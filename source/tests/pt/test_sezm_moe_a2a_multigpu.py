# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-GPU tests for SeZM MoE All-to-All communication primitive.

Run with:
    torchrun --nproc_per_node=2 source/tests/pt/test_sezm_moe_a2a_multigpu.py
    torchrun --nproc_per_node=4 source/tests/pt/test_sezm_moe_a2a_multigpu.py
    torchrun --nproc_per_node=8 source/tests/pt/test_sezm_moe_a2a_multigpu.py
"""

import unittest

import torch
import torch.distributed as dist

from deepmd.pt.model.descriptor.sezm_nn.moe.a2a_ops import (
    all_to_all_differentiable,
)


def setup_dist():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def cleanup_dist():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def make_cyclic_splits(rank, world_size):
    """Return deterministic asymmetric splits valid for any world size."""
    send_splits = [((rank + 2 * peer) % 5) + 1 for peer in range(world_size)]
    recv_splits = [((peer + 2 * rank) % 5) + 1 for peer in range(world_size)]
    return send_splits, recv_splits


def make_encoded_input(rank, send_splits, device):
    """Build rows whose values encode source rank, target rank, and row id."""
    rows = []
    for peer, count in enumerate(send_splits):
        for row_id in range(count):
            rows.append([float(rank), float(peer), float(row_id)])
    return torch.tensor(rows, dtype=torch.float64, device=device)


def make_expected_encoded_output(rank, world_size, device):
    """Expected all-to-all output for make_encoded_input and make_cyclic_splits."""
    rows = []
    for source_rank in range(world_size):
        source_send_splits, _ = make_cyclic_splits(source_rank, world_size)
        count = source_send_splits[rank]
        for row_id in range(count):
            rows.append([float(source_rank), float(rank), float(row_id)])
    return torch.tensor(rows, dtype=torch.float64, device=device)


class TestAllToAllMultiGPU(unittest.TestCase):
    """Multi-GPU tests for _AllToAllDouble communication primitive."""

    @classmethod
    def setUpClass(cls):
        """Set up distributed environment once for all tests."""
        cls.rank, cls.world_size, cls.device = setup_dist()
        cls.group = dist.group.WORLD

    @classmethod
    def tearDownClass(cls):
        """Keep the process group alive until run_tests aggregates results."""

    def test_forward_values_and_shape(self):
        """Forward pass should move the correct rows across ranks."""
        send_splits, recv_splits = make_cyclic_splits(self.rank, self.world_size)

        total_send = sum(send_splits)
        total_recv = sum(recv_splits)

        x = make_encoded_input(self.rank, send_splits, self.device).requires_grad_(True)
        out = all_to_all_differentiable(x, send_splits, recv_splits, self.group)
        expected = make_expected_encoded_output(self.rank, self.world_size, self.device)

        self.assertEqual(
            out.shape[0],
            total_recv,
            f"Rank {self.rank}: expected first dim {total_recv}, got {out.shape[0]}",
        )
        self.assertEqual(
            out.shape[1:],
            x.shape[1:],
            f"Rank {self.rank}: trailing dimensions should be preserved",
        )
        torch.testing.assert_close(out, expected)

    def test_backward_no_deadlock(self):
        """Backward pass should not deadlock."""
        send_splits = [2] * self.world_size
        recv_splits = [2] * self.world_size

        total_send = sum(send_splits)
        x = torch.randn(
            total_send, 8, device=self.device, dtype=torch.float64, requires_grad=True
        )

        out = all_to_all_differentiable(x, send_splits, recv_splits, self.group)
        loss = (out**2).sum()
        loss.backward()

        # If we reach here without hanging, backward succeeded
        self.assertIsNotNone(x.grad, f"Rank {self.rank}: gradient should exist")
        self.assertTrue(
            (x.grad.abs() > 1e-6).any(),
            f"Rank {self.rank}: gradient should be non-zero",
        )

    def test_second_backward_no_deadlock(self):
        """Second backward (create_graph=True) should not deadlock."""
        send_splits = [2] * self.world_size
        recv_splits = [2] * self.world_size

        total_send = sum(send_splits)
        x = torch.randn(
            total_send, 8, device=self.device, dtype=torch.float64, requires_grad=True
        )

        # First forward
        out = all_to_all_differentiable(x, send_splits, recv_splits, self.group)
        loss = (out**2).sum()

        # First backward with create_graph=True
        (grad_x,) = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)

        self.assertTrue(
            grad_x.requires_grad,
            f"Rank {self.rank}: first-order gradient should require grad",
        )

        # Second backward
        loss2 = (grad_x**2).sum()
        loss2.backward()

        # If we reach here without hanging, second backward succeeded
        self.assertIsNotNone(x.grad, f"Rank {self.rank}: second-order gradient exists")
        self.assertTrue(
            (x.grad.abs() > 1e-6).any(),
            f"Rank {self.rank}: second-order gradient should be non-zero",
        )

    def test_asymmetric_splits(self):
        """Test with asymmetric send/recv splits across ranks."""
        send_splits, recv_splits = make_cyclic_splits(self.rank, self.world_size)
        self.assertNotEqual(
            send_splits,
            recv_splits,
            f"Rank {self.rank}: split pattern should be asymmetric",
        )

        total_send = sum(send_splits)
        total_recv = sum(recv_splits)

        x = torch.randn(
            total_send, 16, device=self.device, dtype=torch.float64, requires_grad=True
        )
        out = all_to_all_differentiable(x, send_splits, recv_splits, self.group)

        # Check shape
        self.assertEqual(out.shape[0], total_recv)
        self.assertEqual(out.shape[1], 16)

        # Check backward
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_three_layer_second_backward_no_deadlock(self):
        """Three chained A2A ops should support second backward."""
        send_splits = [1] * self.world_size
        recv_splits = [1] * self.world_size
        x = torch.randn(
            self.world_size,
            4,
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )

        y = x
        for _ in range(3):
            y = all_to_all_differentiable(y, send_splits, recv_splits, self.group)

        loss = (y**2).sum()
        (grad_x,) = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)
        (grad_x**2).sum().backward()
        self.assertIsNotNone(x.grad, f"Rank {self.rank}: second-order grad missing")
        self.assertTrue(
            (x.grad.abs() > 1e-6).any(),
            f"Rank {self.rank}: second-order grad should be non-zero",
        )

    def test_gradgradcheck_fp64_world_group(self):
        """Gradgradcheck should exercise _AllToAllDouble with WORLD group."""
        torch.manual_seed(20260518)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(20260518)

        send_splits = [1] * self.world_size
        recv_splits = [1] * self.world_size
        x = torch.randn(
            self.world_size,
            2,
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )

        def func(inp):
            out = all_to_all_differentiable(
                inp, send_splits, recv_splits, group=self.group
            )
            # Pick the row sourced from this rank so per-rank gradgradcheck
            # perturbs only the input that can affect the local output.
            return out.narrow(0, self.rank, 1)

        result = torch.autograd.gradgradcheck(
            func,
            (x,),
            eps=1e-6,
            atol=1e-4,
            raise_exception=False,
        )
        self.assertTrue(
            result,
            f"Rank {self.rank}: distributed gradgradcheck failed",
        )


def run_tests():
    """Run all tests and report results."""
    import sys

    rank, world_size, device = setup_dist()

    # Only rank 0 prints header
    if rank == 0:
        sys.stdout.write(f"\n{'=' * 70}\n")
        sys.stdout.write(f"Running multi-GPU A2A tests with {world_size} processes\n")
        sys.stdout.write(f"{'=' * 70}\n\n")

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAllToAllMultiGPU)
    runner = unittest.TextTestRunner(verbosity=2 if rank == 0 else 0)
    result = runner.run(suite)

    # Synchronize results across ranks (before cleanup)
    success = torch.tensor(
        [1 if result.wasSuccessful() else 0], dtype=torch.int32, device=device
    )
    if dist.is_initialized():
        dist.all_reduce(success, op=dist.ReduceOp.MIN)

        if rank == 0:
            if success.item() == 1:
                sys.stdout.write(f"\n{'=' * 70}\n")
                sys.stdout.write(f"PASS: all tests passed on all {world_size} ranks\n")
                sys.stdout.write(f"{'=' * 70}\n\n")
            else:
                sys.stdout.write(f"\n{'=' * 70}\n")
                sys.stdout.write("FAIL: tests failed on at least one rank\n")
                sys.stdout.write(f"{'=' * 70}\n\n")

        cleanup_dist()

    return success.item() == 1


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
