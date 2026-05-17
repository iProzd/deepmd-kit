# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-GPU tests for SeZM MoE All-to-All communication primitive.

Run with:
    torchrun --nproc_per_node=2 source/tests/pt/test_sezm_moe_a2a_multigpu.py
    torchrun --nproc_per_node=4 source/tests/pt/test_sezm_moe_a2a_multigpu.py
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
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Use CPU for multi-GPU tests (gloo backend)
    device = torch.device("cpu")
    return rank, world_size, device


def cleanup_dist():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


class TestAllToAllMultiGPU(unittest.TestCase):
    """Multi-GPU tests for _AllToAllDouble communication primitive."""

    @classmethod
    def setUpClass(cls):
        """Set up distributed environment once for all tests."""
        cls.rank, cls.world_size, cls.device = setup_dist()
        cls.group = dist.group.WORLD

    @classmethod
    def tearDownClass(cls):
        """Clean up distributed environment."""
        cleanup_dist()

    def test_forward_shape(self):
        """Forward pass should produce correct output shape across ranks."""
        # Each rank sends different amounts
        # Constraint: rank i's send_splits[j] == rank j's recv_splits[i]
        if self.world_size == 2:
            send_splits = [3, 5] if self.rank == 0 else [2, 6]
            recv_splits = [3, 2] if self.rank == 0 else [5, 6]
        elif self.world_size == 4:
            # Matrix: send[i][j] = recv[j][i]
            # rank 0 sends: [2, 3, 1, 4] -> rank 0 recvs: [2, 5, 3, 7]
            # rank 1 sends: [5, 2, 4, 3] -> rank 1 recvs: [3, 2, 6, 4]
            # rank 2 sends: [3, 6, 1, 2] -> rank 2 recvs: [1, 4, 1, 5]
            # rank 3 sends: [7, 4, 5, 1] -> rank 3 recvs: [4, 3, 2, 1]
            if self.rank == 0:
                send_splits = [2, 3, 1, 4]
                recv_splits = [2, 5, 3, 7]
            elif self.rank == 1:
                send_splits = [5, 2, 4, 3]
                recv_splits = [3, 2, 6, 4]
            elif self.rank == 2:
                send_splits = [3, 6, 1, 2]
                recv_splits = [1, 4, 1, 5]
            else:  # rank 3
                send_splits = [7, 4, 5, 1]
                recv_splits = [4, 3, 2, 1]
        else:
            self.skipTest(f"Test not configured for world_size={self.world_size}")

        total_send = sum(send_splits)
        total_recv = sum(recv_splits)

        x = torch.randn(total_send, 8, device=self.device, requires_grad=True)
        out = all_to_all_differentiable(x, send_splits, recv_splits, self.group)

        # Check output shape
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

    def test_backward_no_deadlock(self):
        """Backward pass should not deadlock."""
        if self.world_size == 2:
            send_splits = [4, 4]
            recv_splits = [4, 4]
        elif self.world_size == 4:
            send_splits = [2, 2, 2, 2]
            recv_splits = [2, 2, 2, 2]
        else:
            self.skipTest(f"Test not configured for world_size={self.world_size}")

        total_send = sum(send_splits)
        x = torch.randn(total_send, 8, device=self.device, requires_grad=True)

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
        if self.world_size == 2:
            send_splits = [3, 3]
            recv_splits = [3, 3]
        elif self.world_size == 4:
            send_splits = [2, 2, 2, 2]
            recv_splits = [2, 2, 2, 2]
        else:
            self.skipTest(f"Test not configured for world_size={self.world_size}")

        total_send = sum(send_splits)
        x = torch.randn(total_send, 8, device=self.device, requires_grad=True)

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
        # Constraint: rank i's send_splits[j] == rank j's recv_splits[i]
        if self.world_size == 2:
            # Rank 0 sends more to rank 1, rank 1 sends more to rank 0
            send_splits = [2, 6] if self.rank == 0 else [5, 3]
            recv_splits = [2, 5] if self.rank == 0 else [6, 3]
        elif self.world_size == 4:
            # Matrix: send[i][j] = recv[j][i]
            # rank 0 sends: [1, 2, 3, 4] -> rank 0 recvs: [1, 3, 2, 4]
            # rank 1 sends: [3, 2, 1, 4] -> rank 1 recvs: [2, 2, 3, 3]
            # rank 2 sends: [2, 3, 4, 1] -> rank 2 recvs: [3, 1, 4, 2]
            # rank 3 sends: [4, 3, 2, 1] -> rank 3 recvs: [4, 4, 1, 1]
            if self.rank == 0:
                send_splits = [1, 2, 3, 4]
                recv_splits = [1, 3, 2, 4]
            elif self.rank == 1:
                send_splits = [3, 2, 1, 4]
                recv_splits = [2, 2, 3, 3]
            elif self.rank == 2:
                send_splits = [2, 3, 4, 1]
                recv_splits = [3, 1, 4, 2]
            else:  # rank 3
                send_splits = [4, 3, 2, 1]
                recv_splits = [4, 4, 1, 1]
        else:
            self.skipTest(f"Test not configured for world_size={self.world_size}")

        total_send = sum(send_splits)
        total_recv = sum(recv_splits)

        x = torch.randn(total_send, 16, device=self.device, requires_grad=True)
        out = all_to_all_differentiable(x, send_splits, recv_splits, self.group)

        # Check shape
        self.assertEqual(out.shape[0], total_recv)
        self.assertEqual(out.shape[1], 16)

        # Check backward
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


def run_tests():
    """Run all tests and report results."""
    import sys

    rank, world_size, _ = setup_dist()

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
    success = torch.tensor([1 if result.wasSuccessful() else 0], dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(success, op=dist.ReduceOp.MIN)

        if rank == 0:
            if success.item() == 1:
                sys.stdout.write(f"\n{'=' * 70}\n")
                sys.stdout.write(f"✓ All tests passed on all {world_size} ranks\n")
                sys.stdout.write(f"{'=' * 70}\n\n")
            else:
                sys.stdout.write(f"\n{'=' * 70}\n")
                sys.stdout.write("✗ Tests failed on at least one rank\n")
                sys.stdout.write(f"{'=' * 70}\n\n")

        cleanup_dist()

    return success.item() == 1


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
