# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for SeZM MoE All-to-All communication primitive (single-GPU)."""

import unittest

import torch

from deepmd.pt.model.descriptor.sezm_nn.moe.a2a_ops import (
    all_to_all_differentiable,
)


class TestAllToAllSingleGPU(unittest.TestCase):
    """Single-GPU tests for _AllToAllDouble communication primitive."""

    def test_single_gpu_passthrough(self):
        """group=None should return x unchanged with gradients flowing through."""
        x = torch.randn(10, 8, requires_grad=True, device="cpu")
        send_splits = [3, 3, 4]
        recv_splits = [2, 5, 3]

        out = all_to_all_differentiable(x, send_splits, recv_splits, group=None)

        # Output should be identical to input
        self.assertIs(out, x, "group=None should return input tensor unchanged")

        # Gradient should flow through
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad, "Gradient should flow through when group=None")
        self.assertTrue(
            torch.allclose(x.grad, torch.ones_like(x)),
            "Gradient should be all ones for sum() loss",
        )

    def test_shape_preservation(self):
        """Forward pass should preserve trailing dimensions."""
        # Test various shapes
        test_cases = [
            ((10, 8), [3, 3, 4], [2, 5, 3]),
            ((15, 16, 32), [5, 5, 5], [4, 6, 5]),
            ((8, 4, 4, 64), [2, 3, 3], [3, 2, 3]),
        ]

        for shape, send_splits, recv_splits in test_cases:
            with self.subTest(shape=shape):
                x = torch.randn(*shape, device="cpu")
                out = all_to_all_differentiable(x, send_splits, recv_splits, group=None)

                # First dimension should match sum(recv_splits)
                expected_shape = (sum(recv_splits), *shape[1:])
                self.assertEqual(
                    out.shape,
                    expected_shape,
                    f"Output shape mismatch for input shape {shape}",
                )

    def test_first_backward(self):
        """loss.backward() should produce non-zero gradients."""
        x = torch.randn(10, 8, requires_grad=True, device="cpu")
        send_splits = [3, 3, 4]
        recv_splits = [2, 5, 3]

        out = all_to_all_differentiable(x, send_splits, recv_splits, group=None)
        loss = (out**2).sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "Gradient should exist after backward")
        self.assertTrue(
            (x.grad.abs() > 1e-6).any(), "Gradient should contain non-zero values"
        )

    def test_second_backward(self):
        """create_graph=True + second backward should produce non-zero gradients."""
        x = torch.randn(10, 8, requires_grad=True, device="cpu")
        send_splits = [3, 3, 4]
        recv_splits = [2, 5, 3]

        # First forward
        out = all_to_all_differentiable(x, send_splits, recv_splits, group=None)
        loss = (out**2).sum()

        # First backward with create_graph=True
        (grad_x,) = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)

        self.assertIsNotNone(grad_x, "First-order gradient should exist")
        self.assertTrue(
            grad_x.requires_grad, "First-order gradient should require grad"
        )

        # Second backward
        loss2 = (grad_x**2).sum()
        loss2.backward()

        self.assertIsNotNone(x.grad, "Second-order gradient should exist")
        self.assertTrue(
            (x.grad.abs() > 1e-6).any(),
            "Second-order gradient should contain non-zero values",
        )

    def test_short_circuit_gradgradcheck_fp64(self):
        """group=None short-circuit should pass gradgradcheck in fp64."""
        # This verifies the single-process passthrough path only.  The real
        # _AllToAllDouble gradgradcheck lives in the multi-GPU test file.
        x = torch.randn(6, 4, dtype=torch.float64, requires_grad=True, device="cpu")
        send_splits = [2, 2, 2]
        recv_splits = [1, 3, 2]

        def func(inp):
            return all_to_all_differentiable(inp, send_splits, recv_splits, group=None)

        # gradgradcheck verifies second-order derivatives
        result = torch.autograd.gradgradcheck(
            func, x, eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=False
        )

        self.assertTrue(
            result, "gradgradcheck failed: second-order derivatives are incorrect"
        )


if __name__ == "__main__":
    unittest.main()
