# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single-GPU unit tests for MoERouter.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest source/tests/pt/test_moe_router.py -v

Tests:
  1. Output shape: [nb*nloc, topk]
  2. Softmax row-sum ≈ 1
  3. Indices ∈ [0, n_routing_experts)
  4. Backward: router gate parameters receive gradients
  5. Deterministic: same seed → same output
  6. Different n_routing_experts / topk combos
"""

import torch
import pytest

from deepmd.pt.model.network.moe_router import MoERouter

DEVICE = torch.device("cpu")
PREC = "float64"


class TestMoERouter:
    """Tests for MoERouter."""

    def _make_router(self, input_dim=16, n_routing_experts=8, topk=2, seed=42):
        # MLPLayer creates params on env.DEVICE (cuda), move to CPU for testing.
        return MoERouter(
            input_dim=input_dim,
            n_routing_experts=n_routing_experts,
            topk=topk,
            precision=PREC,
            seed=seed,
        ).to(DEVICE)

    def test_output_shape(self):
        """topk_weights and topk_indices must be [nb*nloc, topk]."""
        nb, nloc, input_dim = 2, 10, 16
        n_routing_experts, topk = 8, 2
        router = self._make_router(input_dim, n_routing_experts, topk)
        type_emb = torch.randn(nb, nloc, input_dim, dtype=torch.float64, device=DEVICE)
        weights, indices = router(type_emb)
        expected_shape = (nb * nloc, topk)
        assert weights.shape == expected_shape, f"weights shape: {weights.shape}"
        assert indices.shape == expected_shape, f"indices shape: {indices.shape}"

    def test_softmax_sums_to_one(self):
        """Each row of topk_weights must sum to 1.0."""
        nb, nloc, input_dim = 3, 8, 16
        router = self._make_router(input_dim, n_routing_experts=6, topk=3)
        type_emb = torch.randn(nb, nloc, input_dim, dtype=torch.float64, device=DEVICE)
        weights, _ = router(type_emb)
        row_sums = weights.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones(nb * nloc, dtype=torch.float64, device=DEVICE),
            atol=1e-12,
            rtol=0,
        )

    def test_indices_in_range(self):
        """All expert indices must be in [0, n_routing_experts)."""
        nb, nloc, input_dim = 2, 12, 16
        n_routing_experts = 10
        router = self._make_router(input_dim, n_routing_experts, topk=4)
        type_emb = torch.randn(nb, nloc, input_dim, dtype=torch.float64, device=DEVICE)
        _, indices = router(type_emb)
        assert (indices >= 0).all()
        assert (indices < n_routing_experts).all()

    def test_indices_unique_per_row(self):
        """Within each row, the topk indices should be distinct."""
        nb, nloc, input_dim = 2, 8, 16
        router = self._make_router(input_dim, n_routing_experts=8, topk=3)
        type_emb = torch.randn(nb, nloc, input_dim, dtype=torch.float64, device=DEVICE)
        _, indices = router(type_emb)
        for row in range(indices.shape[0]):
            unique = indices[row].unique()
            assert len(unique) == indices.shape[1], (
                f"row {row}: duplicate indices {indices[row]}"
            )

    def test_weights_positive(self):
        """Softmax outputs should all be strictly positive."""
        nb, nloc, input_dim = 2, 6, 16
        router = self._make_router(input_dim, n_routing_experts=8, topk=2)
        type_emb = torch.randn(nb, nloc, input_dim, dtype=torch.float64, device=DEVICE)
        weights, _ = router(type_emb)
        assert (weights > 0).all()

    def test_backward_gate_has_grad(self):
        """Gate parameters must receive gradients through the softmax path."""
        nb, nloc, input_dim = 2, 5, 16
        router = self._make_router(input_dim, n_routing_experts=6, topk=2)
        type_emb = torch.randn(nb, nloc, input_dim, dtype=torch.float64, device=DEVICE)
        weights, _ = router(type_emb)
        loss = weights.sum()
        loss.backward()
        gate_param = router.gate.matrix
        assert gate_param.grad is not None, "gate.matrix has no grad"
        assert gate_param.grad.abs().sum() > 0, "gate.matrix grad is all zeros"

    def test_backward_through_input(self):
        """Gradients must flow back to the input type_embedding."""
        nb, nloc, input_dim = 2, 5, 16
        router = self._make_router(input_dim, n_routing_experts=6, topk=2)
        type_emb = torch.randn(
            nb, nloc, input_dim, dtype=torch.float64, device=DEVICE,
            requires_grad=True,
        )
        weights, _ = router(type_emb)
        loss = weights.sum()
        loss.backward()
        assert type_emb.grad is not None, "input grad is None"
        assert type_emb.grad.abs().sum() > 0, "input grad is all zeros"

    def test_gate_has_no_bias(self):
        """Gate must be configured without bias."""
        router = self._make_router(input_dim=16, n_routing_experts=8, topk=2)
        assert router.gate.bias is None, "gate should have no bias"

    def test_topk_equals_n_experts(self):
        """Edge case: topk == n_routing_experts (select all)."""
        nb, nloc, input_dim = 1, 4, 8
        n_routing_experts = 3
        router = self._make_router(input_dim, n_routing_experts, topk=n_routing_experts)
        type_emb = torch.randn(nb, nloc, input_dim, dtype=torch.float64, device=DEVICE)
        weights, indices = router(type_emb)
        assert weights.shape == (nb * nloc, n_routing_experts)
        # When all experts are selected, each row should contain all indices 0..E-1
        for row in range(indices.shape[0]):
            assert set(indices[row].tolist()) == set(range(n_routing_experts))

    def test_deterministic_with_same_seed(self):
        """Two routers with the same seed must produce identical outputs."""
        nb, nloc, input_dim = 2, 6, 16
        r1 = self._make_router(input_dim, n_routing_experts=8, topk=2, seed=123)
        r2 = self._make_router(input_dim, n_routing_experts=8, topk=2, seed=123)
        type_emb = torch.randn(nb, nloc, input_dim, dtype=torch.float64, device=DEVICE)
        w1, i1 = r1(type_emb)
        w2, i2 = r2(type_emb)
        torch.testing.assert_close(w1, w2)
        assert (i1 == i2).all()

    def test_different_seeds_differ(self):
        """Two routers with different seeds should (almost certainly) differ."""
        nb, nloc, input_dim = 2, 6, 16
        r1 = self._make_router(input_dim, n_routing_experts=8, topk=2, seed=1)
        r2 = self._make_router(input_dim, n_routing_experts=8, topk=2, seed=2)
        type_emb = torch.randn(nb, nloc, input_dim, dtype=torch.float64, device=DEVICE)
        w1, _ = r1(type_emb)
        w2, _ = r2(type_emb)
        assert not torch.allclose(w1, w2), "different seeds produced identical weights"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
