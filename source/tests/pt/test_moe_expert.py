# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single-GPU unit tests for ExpertMLPLayer and MoEExpertCollection.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest source/tests/pt/test_moe_expert.py -v

Tests:
  1. ExpertMLPLayer forward: shape and numerical correctness
  2. MoEExpertCollection.forward_shared: multi-shared-expert sum
  3. Backward: all expert parameters receive gradients
  4. create_graph=True: second-order derivative support
  5. No shared experts: forward_shared returns zeros
  6. Different experts produce different outputs
  7. Deterministic with same seed
"""

import torch
import pytest

from deepmd.pt.model.network.moe_expert import (
    ExpertMLPLayer,
    MoEExpertCollection,
)

DEVICE = torch.device("cpu")
PREC = "float64"


# ---------------------------------------------------------------------------
# ExpertMLPLayer tests
# ---------------------------------------------------------------------------


class TestExpertMLPLayer:
    """Tests for a single ExpertMLPLayer."""

    def _make_expert(self, num_in=8, num_out=4, activation="silu", seed=42):
        return ExpertMLPLayer(
            num_in=num_in,
            num_out=num_out,
            activation_function=activation,
            precision=PREC,
            seed=seed,
        ).to(DEVICE)

    def test_forward_shape(self):
        """Output shape must be [N_tokens, num_out]."""
        expert = self._make_expert(num_in=8, num_out=4)
        x = torch.randn(10, 8, dtype=torch.float64, device=DEVICE)
        y = expert(x)
        assert y.shape == (10, 4)

    def test_forward_numerical(self):
        """Manually verify act(x @ W + b) matches MLPLayer forward."""
        expert = self._make_expert(num_in=4, num_out=3, activation="silu")
        x = torch.randn(5, 4, dtype=torch.float64, device=DEVICE)
        y = expert(x)
        # Manual computation using the same linear + activation
        linear_out = torch.nn.functional.linear(x, expert.matrix.t(), expert.bias)
        expected = torch.nn.functional.silu(linear_out)
        torch.testing.assert_close(y, expected, atol=1e-12, rtol=0)

    def test_has_independent_w_and_b(self):
        """Expert must have both matrix (W) and bias (b) as parameters."""
        expert = self._make_expert()
        assert hasattr(expert, "matrix")
        assert hasattr(expert, "bias")
        assert isinstance(expert.matrix, torch.nn.Parameter)
        assert isinstance(expert.bias, torch.nn.Parameter)
        assert expert.matrix.shape == (8, 4)
        assert expert.bias.shape == (4,)

    def test_bias_init_finite(self):
        """Bias should be initialised to finite values (MLPLayer default_normal_init)."""
        expert = self._make_expert()
        assert torch.isfinite(expert.bias.data).all()

    def test_backward_creates_grad(self):
        """Both W and b must receive gradients."""
        expert = self._make_expert(num_in=6, num_out=3)
        x = torch.randn(4, 6, dtype=torch.float64, device=DEVICE)
        y = expert(x)
        loss = y.sum()
        loss.backward()
        assert expert.matrix.grad is not None
        assert expert.bias.grad is not None
        assert expert.matrix.grad.abs().sum() > 0
        assert expert.bias.grad.abs().sum() > 0

    def test_backward_through_input(self):
        """Gradients must flow to the input tensor."""
        expert = self._make_expert(num_in=6, num_out=3)
        x = torch.randn(4, 6, dtype=torch.float64, device=DEVICE,
                         requires_grad=True)
        y = expert(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_create_graph_2nd_order(self):
        """create_graph=True must work for second-order derivatives."""
        expert = self._make_expert(num_in=4, num_out=3)
        x = torch.randn(3, 4, dtype=torch.float64, device=DEVICE,
                         requires_grad=True)
        y = expert(x)
        loss = (y ** 2).sum()

        params = [x, expert.matrix, expert.bias]
        grads = torch.autograd.grad(loss, params, create_graph=True)
        for g in grads:
            assert g is not None
            assert g.requires_grad

        grad_sum = sum(g.sum() for g in grads)
        grad_sum.backward()
        assert x.grad is not None
        assert expert.matrix.grad is not None
        assert expert.bias.grad is not None

    def test_deterministic_same_seed(self):
        """Same seed must produce identical parameters."""
        e1 = self._make_expert(seed=77)
        e2 = self._make_expert(seed=77)
        torch.testing.assert_close(e1.matrix.data, e2.matrix.data)
        torch.testing.assert_close(e1.bias.data, e2.bias.data)

    def test_different_seeds_differ(self):
        """Different seeds must produce different parameters."""
        e1 = self._make_expert(seed=1)
        e2 = self._make_expert(seed=2)
        assert not torch.allclose(e1.matrix.data, e2.matrix.data)


# ---------------------------------------------------------------------------
# MoEExpertCollection tests
# ---------------------------------------------------------------------------


class TestMoEExpertCollection:
    """Tests for MoEExpertCollection."""

    def _make_collection(
        self, num_in=8, num_out=4, experts_per_gpu=4,
        n_shared_experts=0, seed=42,
    ):
        return MoEExpertCollection(
            num_in=num_in,
            num_out=num_out,
            experts_per_gpu=experts_per_gpu,
            n_shared_experts=n_shared_experts,
            activation_function="silu",
            precision=PREC,
            seed=seed,
        ).to(DEVICE)

    def test_forward_expert_shape(self):
        """forward_expert output shape must be [N_tokens, num_out]."""
        coll = self._make_collection(num_in=8, num_out=4, experts_per_gpu=4)
        x = torch.randn(6, 8, dtype=torch.float64, device=DEVICE)
        for eid in range(4):
            y = coll.forward_expert(x, eid)
            assert y.shape == (6, 4), f"expert {eid} shape: {y.shape}"

    def test_different_experts_differ(self):
        """Different routing experts should produce different outputs."""
        coll = self._make_collection(num_in=8, num_out=4, experts_per_gpu=4)
        x = torch.randn(6, 8, dtype=torch.float64, device=DEVICE)
        y0 = coll.forward_expert(x, 0)
        y1 = coll.forward_expert(x, 1)
        assert not torch.allclose(y0, y1), "expert 0 and 1 produced same output"

    def test_forward_shared_single(self):
        """forward_shared with 1 shared expert returns that expert's output."""
        coll = self._make_collection(
            num_in=8, num_out=4, experts_per_gpu=2, n_shared_experts=1,
        )
        x = torch.randn(5, 8, dtype=torch.float64, device=DEVICE)
        shared_out = coll.forward_shared(x)
        # Must equal the single shared expert's output
        expected = coll.shared_experts[0](x)
        torch.testing.assert_close(shared_out, expected, atol=1e-12, rtol=0)

    def test_forward_shared_sum(self):
        """forward_shared with multiple shared experts returns their sum."""
        coll = self._make_collection(
            num_in=6, num_out=3, experts_per_gpu=2, n_shared_experts=3,
        )
        x = torch.randn(4, 6, dtype=torch.float64, device=DEVICE)
        shared_out = coll.forward_shared(x)
        # Manual sum
        expected = sum(se(x) for se in coll.shared_experts)
        torch.testing.assert_close(shared_out, expected, atol=1e-12, rtol=0)

    def test_forward_shared_no_shared_returns_zeros(self):
        """forward_shared with 0 shared experts returns a zero tensor."""
        coll = self._make_collection(
            num_in=8, num_out=4, experts_per_gpu=3, n_shared_experts=0,
        )
        x = torch.randn(5, 8, dtype=torch.float64, device=DEVICE)
        shared_out = coll.forward_shared(x)
        assert shared_out.shape == (5, 4)
        torch.testing.assert_close(
            shared_out,
            torch.zeros(5, 4, dtype=torch.float64, device=DEVICE),
        )

    def test_forward_shared_zero_does_not_block_grad(self):
        """When n_shared_experts=0, forward_shared returns zeros but
        must not break the overall computation graph."""
        coll = self._make_collection(
            num_in=8, num_out=4, experts_per_gpu=2, n_shared_experts=0,
        )
        x = torch.randn(3, 8, dtype=torch.float64, device=DEVICE,
                         requires_grad=True)
        # Route through a routing expert and add shared (zeros)
        routed = coll.forward_expert(x, 0)
        total = routed + coll.forward_shared(x)
        loss = total.sum()
        loss.backward()
        assert x.grad is not None

    def test_backward_all_experts_have_grad(self):
        """All routing and shared expert parameters must receive gradients."""
        coll = self._make_collection(
            num_in=6, num_out=3, experts_per_gpu=3, n_shared_experts=2,
        )
        x = torch.randn(4, 6, dtype=torch.float64, device=DEVICE)
        # Sum of all expert outputs to ensure all participate in the loss
        total = torch.zeros(4, 3, dtype=torch.float64, device=DEVICE)
        for eid in range(3):
            total = total + coll.forward_expert(x, eid)
        total = total + coll.forward_shared(x)
        loss = total.sum()
        loss.backward()
        for i, e in enumerate(coll.routing_experts):
            assert e.matrix.grad is not None, f"routing_expert[{i}].matrix no grad"
            assert e.bias.grad is not None, f"routing_expert[{i}].bias no grad"
        for i, e in enumerate(coll.shared_experts):
            assert e.matrix.grad is not None, f"shared_expert[{i}].matrix no grad"
            assert e.bias.grad is not None, f"shared_expert[{i}].bias no grad"

    def test_create_graph_2nd_order(self):
        """create_graph=True must work through the collection."""
        coll = self._make_collection(
            num_in=6, num_out=3, experts_per_gpu=2, n_shared_experts=1,
        )
        x = torch.randn(3, 6, dtype=torch.float64, device=DEVICE,
                         requires_grad=True)
        y = coll.forward_expert(x, 0) + coll.forward_shared(x)
        loss = (y ** 2).sum()

        all_params = [x] + list(coll.parameters())
        # allow_unused=True because expert 1's params are not in the graph
        # (only expert 0 and shared experts are used)
        grads = torch.autograd.grad(loss, all_params, create_graph=True,
                                    allow_unused=True)
        used_grads = [g for g in grads if g is not None]
        assert len(used_grads) > 0, "no grads computed at all"
        for g in used_grads:
            assert g.requires_grad, "a used grad is not differentiable"

        grad_sum = sum(g.sum() for g in used_grads)
        grad_sum.backward()
        assert x.grad is not None, "2nd order x.grad is None"
        # Check at least some expert params got 2nd-order grads
        has_2nd = any(p.grad is not None for p in coll.parameters())
        assert has_2nd, "no 2nd-order grads on any expert param"

    def test_routing_experts_count(self):
        """Correct number of routing and shared experts created."""
        coll = self._make_collection(
            experts_per_gpu=5, n_shared_experts=2,
        )
        assert len(coll.routing_experts) == 5
        assert len(coll.shared_experts) == 2

    def test_num_in_num_out_stored(self):
        """num_in and num_out must be accessible."""
        coll = self._make_collection(num_in=10, num_out=7)
        assert coll.num_in == 10
        assert coll.num_out == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
