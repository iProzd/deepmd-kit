# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single-GPU unit tests for MoEDispatchCombine (Step 6).

Run with:
    CUDA_VISIBLE_DEVICES=0 python -m pytest source/tests/pt/test_moe_layer.py -v
"""

from __future__ import annotations

import unittest

import torch

from deepmd.pt.model.network.moe_layer import (
    MoEDispatchCombine,
    _topk_expand_sort,
    _weighted_sum_topk,
)
from deepmd.pt.utils.env import (
    DEVICE,
)

CPU_DEVICE = torch.device("cpu")

# Test dimensions: a_dim=4, so nd=16, ne=8, na=4.
A_DIM = 4
N_DIM = 4 * A_DIM      # 16
E_DIM = 2 * A_DIM      # 8
N_SYM_DIM = 24 * A_DIM  # 96  (nd*axis_neuron + ne*axis_neuron = 16*4 + 8*4)
EDGE_INFO_DIM = 10 * A_DIM  # 40  (nd + nd + ne)
ANGLE_DIM = 4 * A_DIM   # 16  (na + na + na + na)

# MoE config.
N_ROUTING_EXPERTS = 4
TOPK = 2
N_SHARED_EXPERTS = 1
EXPERTS_PER_GPU = 4  # Single GPU holds all experts.


def _make_layer(
    n_routing_experts: int = N_ROUTING_EXPERTS,
    topk: int = TOPK,
    n_shared_experts: int = N_SHARED_EXPERTS,
    experts_per_gpu: int | None = None,
    seed: int = 42,
) -> MoEDispatchCombine:
    """Create a MoEDispatchCombine for single-GPU testing."""
    if experts_per_gpu is None:
        experts_per_gpu = n_routing_experts
    layer = MoEDispatchCombine(
        n_dim=N_DIM,
        e_dim=E_DIM,
        a_dim=A_DIM,
        n_sym_dim=N_SYM_DIM,
        edge_info_dim=EDGE_INFO_DIM,
        angle_dim=ANGLE_DIM,
        n_routing_experts=n_routing_experts,
        topk=topk,
        n_shared_experts=n_shared_experts,
        ep_group=None,
        ep_rank=0,
        ep_size=1,
        experts_per_gpu=experts_per_gpu,
        activation_function="silu",
        precision="float64",
        seed=seed,
    )
    # Move to CPU for single-GPU tests.
    return layer.to(CPU_DEVICE)


def _make_inputs(
    n_node: int = 8,
    n_edge: int = 20,
    n_angle: int = 30,
    topk: int = TOPK,
    n_routing_experts: int = N_ROUTING_EXPERTS,
    requires_grad: bool = False,
    seed: int = 0,
) -> dict:
    """Create random input tensors for MoEDispatchCombine."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    node_m1 = torch.randn(n_node, N_DIM, device=CPU_DEVICE, dtype=torch.float64,
                           generator=gen, requires_grad=requires_grad)
    node_m2 = torch.randn(n_node, N_SYM_DIM, device=CPU_DEVICE, dtype=torch.float64,
                           generator=gen, requires_grad=requires_grad)
    edge = torch.randn(n_edge, EDGE_INFO_DIM, device=CPU_DEVICE, dtype=torch.float64,
                        generator=gen, requires_grad=requires_grad)
    angle = torch.randn(n_angle, ANGLE_DIM, device=CPU_DEVICE, dtype=torch.float64,
                         generator=gen, requires_grad=requires_grad)

    # Generate routing outputs (fake router output).
    # topk_weights: softmax-normalized.
    node_logits = torch.randn(n_node, n_routing_experts, device=CPU_DEVICE,
                               dtype=torch.float64, generator=gen)
    node_topk_logits, node_topk_indices = torch.topk(node_logits, k=topk, dim=-1)
    node_topk_weights = torch.softmax(node_topk_logits, dim=-1)
    if requires_grad:
        node_topk_weights = node_topk_weights.detach().requires_grad_(True)

    edge_logits = torch.randn(n_node, n_routing_experts, device=CPU_DEVICE,
                               dtype=torch.float64, generator=gen)
    edge_topk_logits, edge_topk_indices = torch.topk(edge_logits, k=topk, dim=-1)
    edge_topk_weights = torch.softmax(edge_topk_logits, dim=-1)
    if requires_grad:
        edge_topk_weights = edge_topk_weights.detach().requires_grad_(True)

    angle_logits = torch.randn(n_node, n_routing_experts, device=CPU_DEVICE,
                                dtype=torch.float64, generator=gen)
    angle_topk_logits, angle_topk_indices = torch.topk(angle_logits, k=topk, dim=-1)
    angle_topk_weights = torch.softmax(angle_topk_logits, dim=-1)
    if requires_grad:
        angle_topk_weights = angle_topk_weights.detach().requires_grad_(True)

    # Index tensors mapping edges/angles to center nodes.
    n2e_index = torch.randint(0, n_node, (n_edge,), device=CPU_DEVICE)
    n2a_index = torch.randint(0, n_node, (n_angle,), device=CPU_DEVICE)

    return {
        "node_m1_input": node_m1,
        "node_m2_input": node_m2,
        "edge_input": edge,
        "angle_input": angle,
        "node_router_out": (node_topk_weights, node_topk_indices),
        "edge_router_out": (edge_topk_weights, edge_topk_indices),
        "angle_router_out": (angle_topk_weights, angle_topk_indices),
        "n2e_index": n2e_index,
        "n2a_index": n2a_index,
    }


# ======================================================================
# Test _topk_expand_sort helper
# ======================================================================


class TestTopkExpandSort(unittest.TestCase):
    """Tests for the _topk_expand_sort helper function."""

    def test_output_shapes(self):
        N, feat_dim, topk = 10, 8, 2
        experts_per_gpu = 4
        features = torch.randn(N, feat_dim, device=CPU_DEVICE)
        indices = torch.randint(0, 8, (N, topk), device=CPU_DEVICE)
        weights = torch.softmax(torch.randn(N, topk, device=CPU_DEVICE), dim=-1)

        sorted_feat, sorted_eids, sorted_w, unsort_idx, counts, ep_sz = \
            _topk_expand_sort(features, indices, weights, experts_per_gpu)

        self.assertEqual(sorted_feat.shape, (N * topk, feat_dim))
        self.assertEqual(sorted_eids.shape, (N * topk,))
        self.assertEqual(sorted_w.shape, (N * topk,))
        self.assertEqual(unsort_idx.shape, (N * topk,))
        self.assertEqual(sum(counts), N * topk)

    def test_unsort_roundtrip(self):
        """Unsorted features match original expanded features."""
        N, feat_dim, topk = 6, 4, 3
        experts_per_gpu = 2
        features = torch.randn(N, feat_dim, device=CPU_DEVICE, dtype=torch.float64)
        indices = torch.randint(0, 4, (N, topk), device=CPU_DEVICE)
        weights = torch.softmax(torch.randn(N, topk, device=CPU_DEVICE, dtype=torch.float64), dim=-1)

        sorted_feat, _, sorted_w, unsort_idx, _, _ = \
            _topk_expand_sort(features, indices, weights, experts_per_gpu)

        expanded = features.repeat_interleave(topk, dim=0)
        flat_w = weights.reshape(-1)
        # After sort then unsort, should match original.
        restored = sorted_feat[unsort_idx]
        torch.testing.assert_close(restored, expanded)
        restored_w = sorted_w[unsort_idx]
        torch.testing.assert_close(restored_w, flat_w)

    def test_sorted_by_target_gpu(self):
        """Tokens are sorted by target GPU (non-decreasing)."""
        N, feat_dim, topk = 8, 4, 2
        experts_per_gpu = 2
        features = torch.randn(N, feat_dim, device=CPU_DEVICE)
        indices = torch.randint(0, 6, (N, topk), device=CPU_DEVICE)
        weights = torch.softmax(torch.randn(N, topk, device=CPU_DEVICE), dim=-1)

        sorted_feat, sorted_eids, _, _, counts, _ = \
            _topk_expand_sort(features, indices, weights, experts_per_gpu)

        target_gpus = sorted_eids // experts_per_gpu
        # Should be non-decreasing.
        diffs = target_gpus[1:] - target_gpus[:-1]
        self.assertTrue((diffs >= 0).all())

    def test_counts_consistency(self):
        """Counts per GPU sum to N*topk and match actual token counts."""
        N, feat_dim, topk = 10, 4, 2
        experts_per_gpu = 3
        features = torch.randn(N, feat_dim, device=CPU_DEVICE)
        # Force all experts into range [0, 6).
        indices = torch.randint(0, 6, (N, topk), device=CPU_DEVICE)
        weights = torch.softmax(torch.randn(N, topk, device=CPU_DEVICE), dim=-1)

        _, sorted_eids, _, _, counts, ep_sz = \
            _topk_expand_sort(features, indices, weights, experts_per_gpu)

        self.assertEqual(sum(counts), N * topk)
        target_gpus = sorted_eids // experts_per_gpu
        for g in range(ep_sz):
            expected = int((target_gpus == g).sum().item())
            self.assertEqual(counts[g], expected)


# ======================================================================
# Test _weighted_sum_topk helper
# ======================================================================


class TestWeightedSumTopk(unittest.TestCase):
    """Tests for the _weighted_sum_topk helper function."""

    def test_basic(self):
        n_orig, topk, out_dim = 4, 2, 3
        expanded = torch.randn(n_orig * topk, out_dim, device=CPU_DEVICE, dtype=torch.float64)
        weights = torch.softmax(
            torch.randn(n_orig * topk, device=CPU_DEVICE, dtype=torch.float64).reshape(n_orig, topk),
            dim=-1,
        ).reshape(-1)

        result = _weighted_sum_topk(expanded, weights, n_orig, topk)
        self.assertEqual(result.shape, (n_orig, out_dim))

        # Manual check.
        for i in range(n_orig):
            expected = torch.zeros(out_dim, device=CPU_DEVICE, dtype=torch.float64)
            for k in range(topk):
                expected += weights[i * topk + k] * expanded[i * topk + k]
            torch.testing.assert_close(result[i], expected)

    def test_uniform_weights(self):
        """Uniform weights should give average."""
        n_orig, topk, out_dim = 3, 2, 4
        expanded = torch.randn(n_orig * topk, out_dim, device=CPU_DEVICE, dtype=torch.float64)
        weights = torch.full((n_orig * topk,), 0.5, device=CPU_DEVICE, dtype=torch.float64)

        result = _weighted_sum_topk(expanded, weights, n_orig, topk)
        for i in range(n_orig):
            expected = 0.5 * expanded[2 * i] + 0.5 * expanded[2 * i + 1]
            torch.testing.assert_close(result[i], expected)


# ======================================================================
# Test MoEDispatchCombine forward
# ======================================================================


class TestMoEDispatchCombineForwardShape(unittest.TestCase):
    """Test forward output shapes."""

    def test_basic_shapes(self):
        layer = _make_layer()
        inputs = _make_inputs()
        m1, m2, e, a = layer(**inputs)

        n_node = inputs["node_m1_input"].shape[0]
        n_edge = inputs["edge_input"].shape[0]
        n_angle = inputs["angle_input"].shape[0]

        self.assertEqual(m1.shape, (n_node, N_DIM))
        self.assertEqual(m2.shape, (n_node, N_DIM))
        self.assertEqual(e.shape, (n_edge, N_DIM + E_DIM))
        self.assertEqual(a.shape, (n_angle, E_DIM + A_DIM))

    def test_zero_edge_angle(self):
        """Works with zero edges and angles."""
        layer = _make_layer()
        inputs = _make_inputs(n_edge=0, n_angle=0)
        m1, m2, e, a = layer(**inputs)

        self.assertEqual(m1.shape, (8, N_DIM))
        self.assertEqual(m2.shape, (8, N_DIM))
        self.assertEqual(e.shape, (0, N_DIM + E_DIM))
        self.assertEqual(a.shape, (0, E_DIM + A_DIM))

    def test_single_node(self):
        """Works with single node."""
        layer = _make_layer()
        inputs = _make_inputs(n_node=1, n_edge=3, n_angle=2)
        m1, m2, e, a = layer(**inputs)

        self.assertEqual(m1.shape, (1, N_DIM))
        self.assertEqual(m2.shape, (1, N_DIM))
        self.assertEqual(e.shape, (3, N_DIM + E_DIM))
        self.assertEqual(a.shape, (2, E_DIM + A_DIM))

    def test_topk_1(self):
        """Works with topk=1."""
        layer = _make_layer(topk=1, experts_per_gpu=N_ROUTING_EXPERTS)
        inputs = _make_inputs(topk=1)
        m1, m2, e, a = layer(**inputs)

        self.assertEqual(m1.shape, (8, N_DIM))
        self.assertEqual(m2.shape, (8, N_DIM))

    def test_many_experts(self):
        """Works with more experts."""
        n_experts = 8
        layer = _make_layer(n_routing_experts=n_experts, topk=3, experts_per_gpu=n_experts)
        inputs = _make_inputs(topk=3, n_routing_experts=n_experts)
        m1, m2, e, a = layer(**inputs)

        self.assertEqual(m1.shape, (8, N_DIM))

    def test_no_shared_experts(self):
        """Works with no shared experts."""
        layer = _make_layer(n_shared_experts=0)
        inputs = _make_inputs()
        m1, m2, e, a = layer(**inputs)

        self.assertEqual(m1.shape, (8, N_DIM))

    def test_deterministic(self):
        """Same seed + inputs produce same output."""
        layer1 = _make_layer(seed=123)
        layer2 = _make_layer(seed=123)
        inputs = _make_inputs(seed=456)

        m1a, m2a, ea, aa = layer1(**inputs)
        m1b, m2b, eb, ab = layer2(**inputs)

        torch.testing.assert_close(m1a, m1b)
        torch.testing.assert_close(m2a, m2b)
        torch.testing.assert_close(ea, eb)
        torch.testing.assert_close(aa, ab)


# ======================================================================
# Test backward (gradient propagation)
# ======================================================================


class TestMoEDispatchCombineBackward(unittest.TestCase):
    """Test that gradients propagate to all parameters and inputs."""

    def test_expert_params_have_grad(self):
        """All expert parameters receive gradients."""
        layer = _make_layer()
        inputs = _make_inputs(requires_grad=True)
        m1, m2, e, a = layer(**inputs)

        loss = (m1 ** 2).sum() + (m2 ** 2).sum() + (e ** 2).sum() + (a ** 2).sum()
        loss.backward()

        for name, param in layer.named_parameters():
            self.assertIsNotNone(
                param.grad, f"Parameter {name} has no gradient"
            )
            self.assertTrue(
                (param.grad.abs() > 0).any(),
                f"Parameter {name} has all-zero gradient",
            )

    def test_input_grads(self):
        """Inputs receive gradients."""
        layer = _make_layer()
        inputs = _make_inputs(requires_grad=True)
        m1, m2, e, a = layer(**inputs)

        loss = (m1 ** 2).sum() + (m2 ** 2).sum() + (e ** 2).sum() + (a ** 2).sum()
        loss.backward()

        for key in ["node_m1_input", "node_m2_input", "edge_input", "angle_input"]:
            self.assertIsNotNone(
                inputs[key].grad, f"Input {key} has no gradient"
            )

    def test_router_weight_grads(self):
        """Router weights receive gradients."""
        layer = _make_layer()
        inputs = _make_inputs(requires_grad=True)
        m1, m2, e, a = layer(**inputs)

        loss = (m1 ** 2).sum() + (m2 ** 2).sum() + (e ** 2).sum() + (a ** 2).sum()
        loss.backward()

        node_w = inputs["node_router_out"][0]
        edge_w = inputs["edge_router_out"][0]
        angle_w = inputs["angle_router_out"][0]
        self.assertIsNotNone(node_w.grad, "Node router weights have no gradient")
        self.assertIsNotNone(edge_w.grad, "Edge router weights have no gradient")
        self.assertIsNotNone(angle_w.grad, "Angle router weights have no gradient")


# ======================================================================
# Test second-order derivatives
# ======================================================================


class TestMoEDispatchCombineSecondOrder(unittest.TestCase):
    """Test create_graph=True second-order derivative support."""

    def test_second_order_grad(self):
        """Second-order derivative through MoE pipeline."""
        layer = _make_layer()
        inputs = _make_inputs(requires_grad=True)
        m1, m2, e, a = layer(**inputs)

        loss = (m1 ** 2).sum() + (m2 ** 2).sum() + (e ** 2).sum() + (a ** 2).sum()

        # First-order gradient with create_graph=True.
        (grad_m1,) = torch.autograd.grad(
            loss, inputs["node_m1_input"], create_graph=True,
        )
        self.assertIsNotNone(grad_m1)
        self.assertTrue(grad_m1.requires_grad)

        # Second-order: differentiate the gradient w.r.t. parameters.
        grad_sum = (grad_m1 ** 2).sum()
        grad_sum.backward()

        # At least some expert params should have nonzero 2nd-order grads.
        has_nonzero = False
        for name, param in layer.named_parameters():
            if param.grad is not None and (param.grad.abs() > 0).any():
                has_nonzero = True
                break
        self.assertTrue(has_nonzero, "No parameter received 2nd-order gradient")


# ======================================================================
# Test single-expert special case
# ======================================================================


class TestMoEDispatchCombineSingleExpert(unittest.TestCase):
    """Test with n_routing_experts=1 (degenerate case)."""

    def test_single_expert_topk1(self):
        """Single expert with topk=1: all tokens go to expert 0."""
        layer = _make_layer(n_routing_experts=1, topk=1, n_shared_experts=0,
                            experts_per_gpu=1)
        inputs = _make_inputs(topk=1, n_routing_experts=1)
        m1, m2, e, a = layer(**inputs)

        n_node = inputs["node_m1_input"].shape[0]
        self.assertEqual(m1.shape, (n_node, N_DIM))

        # All weights should be 1.0 (softmax of single value).
        node_w = inputs["node_router_out"][0]
        torch.testing.assert_close(
            node_w, torch.ones_like(node_w),
        )


# ======================================================================
# Test consistency with different topk values
# ======================================================================


class TestMoEDispatchCombineConsistency(unittest.TestCase):
    """Cross-validation tests."""

    def test_shared_expert_only_with_zero_weights(self):
        """When routing weights are zero, output equals shared expert output."""
        layer = _make_layer(n_routing_experts=2, topk=1, n_shared_experts=1,
                            experts_per_gpu=2)
        inputs = _make_inputs(topk=1, n_routing_experts=2)

        # Force zero routing weights.
        zero_w = torch.zeros_like(inputs["node_router_out"][0])
        inputs["node_router_out"] = (zero_w, inputs["node_router_out"][1])
        inputs["edge_router_out"] = (
            torch.zeros_like(inputs["edge_router_out"][0]),
            inputs["edge_router_out"][1],
        )
        inputs["angle_router_out"] = (
            torch.zeros_like(inputs["angle_router_out"][0]),
            inputs["angle_router_out"][1],
        )

        m1, m2, e, a = layer(**inputs)

        # Expected: only shared expert output.
        expected_m1 = layer.node_self_experts.forward_shared(inputs["node_m1_input"])
        expected_m2 = layer.node_sym_experts.forward_shared(inputs["node_m2_input"])
        expected_e = layer.edge_experts.forward_shared(inputs["edge_input"])
        expected_a = layer.angle_experts.forward_shared(inputs["angle_input"])

        torch.testing.assert_close(m1, expected_m1)
        torch.testing.assert_close(m2, expected_m2)
        torch.testing.assert_close(e, expected_e)
        torch.testing.assert_close(a, expected_a)


# ======================================================================
# Test validate_dim_ratio is enforced
# ======================================================================


class TestDimRatioValidation(unittest.TestCase):
    """Test that invalid dim ratios are rejected."""

    def test_invalid_ratio_raises(self):
        with self.assertRaises(ValueError):
            MoEDispatchCombine(
                n_dim=16, e_dim=8, a_dim=5,  # not 4:2:1
                n_sym_dim=96, edge_info_dim=40, angle_dim=16,
                n_routing_experts=4, topk=2,
            )


if __name__ == "__main__":
    unittest.main()
