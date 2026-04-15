# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single-GPU unit tests for moe_ep_ops.

Tests the ``group=None`` path of ``all_to_all_differentiable`` which must
behave as identity with full gradient pass-through.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest source/tests/pt/test_moe_ep_ops.py -v
"""

import torch
import pytest

from deepmd.pt.model.network.moe_ep_ops import all_to_all_differentiable

# Use CPU for single-GPU tests to avoid deepmd device context issues.
DEVICE = torch.device("cpu")


class TestAllToAllSingleGPU:
    """Tests for group=None (single-GPU noop) behaviour."""

    def test_noop_returns_same_tensor(self):
        """group=None should return the exact same tensor object."""
        x = torch.randn(8, 16, dtype=torch.float64, device=DEVICE)
        y = all_to_all_differentiable(x, [8], [8], group=None)
        assert y is x

    def test_noop_gradient_passthrough(self):
        """Gradient must flow through unchanged when group=None."""
        x = torch.randn(6, 4, dtype=torch.float64, device=DEVICE, requires_grad=True)
        y = all_to_all_differentiable(x, [6], [6], group=None)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_noop_create_graph(self):
        """create_graph=True should work with group=None (trivial case)."""
        x = torch.randn(4, 3, dtype=torch.float64, device=DEVICE, requires_grad=True)
        y = all_to_all_differentiable(x, [4], [4], group=None)
        loss = (y ** 2).sum()
        (grad,) = torch.autograd.grad(loss, x, create_graph=True)
        # grad = 2*x, differentiable w.r.t. x
        grad_sum = grad.sum()
        grad_sum.backward()
        assert x.grad is not None
        # d/dx (sum(2*x)) = 2 for each element
        torch.testing.assert_close(x.grad, 2.0 * torch.ones_like(x))

    def test_noop_preserves_dtype_and_device(self):
        """Returned tensor must keep original dtype and device."""
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(3, 5, dtype=dtype, device=DEVICE)
            y = all_to_all_differentiable(x, [3], [3], group=None)
            assert y.dtype == dtype
            assert y.device == DEVICE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
