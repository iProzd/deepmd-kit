# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single-GPU unit tests for RepFlowLayer with MoE (Step 7).

Run with:
    CUDA_VISIBLE_DEVICES=0 python -m pytest source/tests/pt/test_repflow_moe.py -v
"""

from __future__ import annotations

import unittest

import torch

from deepmd.pt.model.descriptor.repflow_layer import (
    RepFlowLayer,
)
from deepmd.pt.utils.env import (
    DEVICE,
)

DTYPE = torch.float64
CPU_DEVICE = torch.device("cpu")

# ======================================================================
# Layer config matching MoE constraints:
#   dynamic_sel=True, optim_update=False, update_angle=True,
#   n_multi_edge_message=1, a_compress_use_split=True,
#   n_dim:e_dim:a_dim = 4:2:1
# ======================================================================
A_DIM = 4
N_DIM = 4 * A_DIM       # 16
E_DIM = 2 * A_DIM       # 8
E_RCUT = 6.0
E_RCUT_SMTH = 5.0
E_SEL = 20
A_RCUT = 4.0
A_RCUT_SMTH = 3.5
A_SEL = 10
NTYPES = 2
AXIS_NEURON = 4
A_COMPRESS_RATE = 1
A_COMPRESS_E_RATE = 2

# Default MoE params.
N_ROUTING_EXPERTS = 4
MOE_TOPK = 2
N_SHARED_EXPERTS = 0

# Test sizes.
NB = 1
NLOC = 6
NALL = NLOC + 4  # ghost atoms
N_EDGE = 30
N_ANGLE = 50

# Shared kwargs for RepFlowLayer (non-MoE-specific).
_BASE_KWARGS = dict(
    e_rcut=E_RCUT,
    e_rcut_smth=E_RCUT_SMTH,
    e_sel=E_SEL,
    a_rcut=A_RCUT,
    a_rcut_smth=A_RCUT_SMTH,
    a_sel=A_SEL,
    ntypes=NTYPES,
    n_dim=N_DIM,
    e_dim=E_DIM,
    a_dim=A_DIM,
    a_compress_rate=A_COMPRESS_RATE,
    a_compress_use_split=True,
    a_compress_e_rate=A_COMPRESS_E_RATE,
    n_multi_edge_message=1,
    axis_neuron=AXIS_NEURON,
    update_angle=True,
    optim_update=False,
    use_dynamic_sel=True,
    smooth_edge_update=True,
    activation_function="silu",
    update_style="res_residual",
    update_residual=0.1,
    update_residual_init="const",
    precision="float64",
)


def _make_layer(
    use_moe: bool = True,
    n_routing_experts: int = N_ROUTING_EXPERTS,
    moe_topk: int = MOE_TOPK,
    n_shared_experts: int = N_SHARED_EXPERTS,
    seed: int = 42,
) -> RepFlowLayer:
    """Create a RepFlowLayer for single-GPU testing."""
    layer = RepFlowLayer(
        **_BASE_KWARGS,
        seed=seed,
        use_moe=use_moe,
        n_routing_experts=n_routing_experts if use_moe else 0,
        moe_topk=moe_topk if use_moe else 0,
        n_shared_experts=n_shared_experts if use_moe else 0,
        ep_group=None,
        ep_rank=0,
        ep_size=1,
    )
    return layer.to(CPU_DEVICE)


def _make_inputs(
    nb: int = NB,
    nloc: int = NLOC,
    nall: int = NALL,
    n_edge: int = N_EDGE,
    n_angle: int = N_ANGLE,
    requires_grad: bool = False,
    seed: int = 0,
) -> dict:
    """Create random inputs matching dynamic_sel format."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    node_ebd_ext = torch.randn(
        nb, nall, N_DIM, device=CPU_DEVICE, dtype=DTYPE,
        generator=gen, requires_grad=requires_grad,
    )
    edge_ebd = torch.randn(
        n_edge, E_DIM, device=CPU_DEVICE, dtype=DTYPE,
        generator=gen, requires_grad=requires_grad,
    )
    h2 = torch.randn(
        n_edge, 3, device=CPU_DEVICE, dtype=DTYPE,
        generator=gen, requires_grad=requires_grad,
    )
    angle_ebd = torch.randn(
        n_angle, A_DIM, device=CPU_DEVICE, dtype=DTYPE,
        generator=gen, requires_grad=requires_grad,
    )

    nlist = torch.randint(0, nall, (nb, nloc, E_SEL), device=CPU_DEVICE, generator=gen)
    nlist_mask = torch.ones(nb, nloc, E_SEL, device=CPU_DEVICE, dtype=DTYPE)
    sw = torch.rand(n_edge, device=CPU_DEVICE, dtype=DTYPE, generator=gen)
    if requires_grad:
        sw = sw.detach().requires_grad_(True)

    a_nlist = torch.randint(0, max(nloc, 1), (nb, nloc, A_SEL), device=CPU_DEVICE, generator=gen)
    a_nlist_mask = torch.ones(nb, nloc, A_SEL, device=CPU_DEVICE, dtype=DTYPE)
    a_sw = torch.rand(max(n_angle, 1), device=CPU_DEVICE, dtype=DTYPE, generator=gen)[:n_angle]
    if requires_grad and n_angle > 0:
        a_sw = a_sw.detach().requires_grad_(True)

    gen_idx = torch.Generator(device="cpu")
    gen_idx.manual_seed(seed + 100)
    n2e_index = torch.randint(0, nb * nloc, (n_edge,), device=CPU_DEVICE, generator=gen_idx)
    n_ext2e_index = torch.randint(0, nb * nall, (n_edge,), device=CPU_DEVICE, generator=gen_idx)
    edge_index = torch.stack([n2e_index, n_ext2e_index], dim=0)

    if n_angle > 0:
        n2a_index = torch.randint(0, nb * nloc, (n_angle,), device=CPU_DEVICE, generator=gen_idx)
        eij2a_index = torch.randint(0, max(n_edge, 1), (n_angle,), device=CPU_DEVICE, generator=gen_idx)
        eik2a_index = torch.randint(0, max(n_edge, 1), (n_angle,), device=CPU_DEVICE, generator=gen_idx)
    else:
        n2a_index = torch.zeros(0, dtype=torch.long, device=CPU_DEVICE)
        eij2a_index = torch.zeros(0, dtype=torch.long, device=CPU_DEVICE)
        eik2a_index = torch.zeros(0, dtype=torch.long, device=CPU_DEVICE)
    angle_index = torch.stack([n2a_index, eij2a_index, eik2a_index], dim=0)

    type_embedding = torch.randn(
        nb, nloc, N_DIM, device=CPU_DEVICE, dtype=DTYPE,
        generator=gen, requires_grad=requires_grad,
    )

    return {
        "node_ebd_ext": node_ebd_ext,
        "edge_ebd": edge_ebd,
        "h2": h2,
        "angle_ebd": angle_ebd,
        "nlist": nlist,
        "nlist_mask": nlist_mask,
        "sw": sw,
        "a_nlist": a_nlist,
        "a_nlist_mask": a_nlist_mask,
        "a_sw": a_sw,
        "edge_index": edge_index,
        "angle_index": angle_index,
        "type_embedding": type_embedding,
    }


# ======================================================================
# Test forward output shapes
# ======================================================================


class TestRepFlowMoEForwardShape(unittest.TestCase):
    """Test MoE forward produces correct output shapes."""

    def test_basic_shapes(self):
        layer = _make_layer(use_moe=True)
        inputs = _make_inputs()
        n_out, e_out, a_out = layer(**inputs)

        self.assertEqual(n_out.shape, (NB, NLOC, N_DIM))
        self.assertEqual(e_out.shape, (N_EDGE, E_DIM))
        self.assertEqual(a_out.shape, (N_ANGLE, A_DIM))

    def test_different_sizes(self):
        """Works with various node/edge/angle sizes."""
        layer = _make_layer(use_moe=True)
        inputs = _make_inputs(nb=2, nloc=4, nall=8, n_edge=16, n_angle=24)
        n_out, e_out, a_out = layer(**inputs)

        self.assertEqual(n_out.shape, (2, 4, N_DIM))
        self.assertEqual(e_out.shape, (16, E_DIM))
        self.assertEqual(a_out.shape, (24, A_DIM))

    def test_zero_edges_angles(self):
        """Works with zero edges and angles."""
        layer = _make_layer(use_moe=True)
        inputs = _make_inputs(n_edge=0, n_angle=0)
        n_out, e_out, a_out = layer(**inputs)

        self.assertEqual(n_out.shape, (NB, NLOC, N_DIM))
        self.assertEqual(e_out.shape, (0, E_DIM))
        self.assertEqual(a_out.shape, (0, A_DIM))

    def test_deterministic(self):
        """Same seed + inputs produce same output."""
        layer1 = _make_layer(seed=123)
        layer2 = _make_layer(seed=123)
        inputs = _make_inputs(seed=456)

        n1, e1, a1 = layer1(**inputs)
        n2, e2, a2 = layer2(**inputs)

        torch.testing.assert_close(n1, n2)
        torch.testing.assert_close(e1, e2)
        torch.testing.assert_close(a1, a2)

    def test_with_shared_experts(self):
        """Works with shared experts."""
        layer = _make_layer(use_moe=True, n_routing_experts=4, n_shared_experts=1)
        inputs = _make_inputs()
        n_out, e_out, a_out = layer(**inputs)

        self.assertEqual(n_out.shape, (NB, NLOC, N_DIM))
        self.assertEqual(e_out.shape, (N_EDGE, E_DIM))
        self.assertEqual(a_out.shape, (N_ANGLE, A_DIM))

    def test_no_non_moe_params(self):
        """MoE layer should not have non-MoE MLP params."""
        layer = _make_layer(use_moe=True)
        self.assertIsNone(layer.node_self_mlp)
        self.assertIsNone(layer.node_sym_linear)
        self.assertIsNone(layer.node_edge_linear)
        self.assertIsNone(layer.edge_self_linear)
        self.assertIsNone(layer.edge_angle_linear1)
        self.assertIsNone(layer.edge_angle_linear2)
        self.assertIsNone(layer.angle_self_linear)
        # MoE attrs should exist.
        self.assertIsNotNone(layer.node_router)
        self.assertIsNotNone(layer.moe_phase1)
        self.assertIsNotNone(layer.edge_angle_linear2_moe)

    def test_no_moe_params_when_disabled(self):
        """Non-MoE layer should not have MoE params."""
        layer = _make_layer(use_moe=False)
        self.assertIsNone(layer.node_router)
        self.assertIsNone(layer.edge_router)
        self.assertIsNone(layer.angle_router)
        self.assertIsNone(layer.moe_phase1)
        self.assertIsNone(layer.edge_angle_linear2_moe)
        # Non-MoE attrs should exist.
        self.assertIsNotNone(layer.node_self_mlp)
        self.assertIsNotNone(layer.node_edge_linear)


# ======================================================================
# Test backward (gradient propagation)
# ======================================================================


class TestRepFlowMoEBackward(unittest.TestCase):
    """Test gradients flow through MoE forward."""

    def test_moe_param_grads(self):
        """All MoE-related parameters receive gradients."""
        layer = _make_layer(use_moe=True)
        inputs = _make_inputs(requires_grad=True)
        n_out, e_out, a_out = layer(**inputs)

        loss = (n_out ** 2).sum() + (e_out ** 2).sum() + (a_out ** 2).sum()
        loss.backward()

        moe_prefixes = (
            "node_router.", "edge_router.", "angle_router.",
            "moe_phase1.", "edge_angle_linear2_moe.",
            "n_residual.", "e_residual.", "a_residual.",
        )
        for name, param in layer.named_parameters():
            if any(name.startswith(p) for p in moe_prefixes):
                self.assertIsNotNone(
                    param.grad, f"MoE parameter {name} has no gradient"
                )
                self.assertTrue(
                    (param.grad.abs() > 0).any(),
                    f"MoE parameter {name} has all-zero gradient",
                )

    def test_input_grads(self):
        """Key inputs receive gradients."""
        layer = _make_layer(use_moe=True)
        inputs = _make_inputs(requires_grad=True)
        n_out, e_out, a_out = layer(**inputs)

        loss = (n_out ** 2).sum() + (e_out ** 2).sum() + (a_out ** 2).sum()
        loss.backward()

        for key in ["node_ebd_ext", "edge_ebd", "h2", "angle_ebd", "type_embedding"]:
            self.assertIsNotNone(
                inputs[key].grad, f"Input {key} has no gradient"
            )


# ======================================================================
# Test second-order derivatives
# ======================================================================


class TestRepFlowMoESecondOrder(unittest.TestCase):
    """Test create_graph=True second-order derivative support."""

    def test_second_order(self):
        """Second-order derivative through MoE pipeline."""
        layer = _make_layer(use_moe=True)
        inputs = _make_inputs(requires_grad=True)
        n_out, e_out, a_out = layer(**inputs)

        loss = (n_out ** 2).sum() + (e_out ** 2).sum() + (a_out ** 2).sum()

        (grad_node,) = torch.autograd.grad(
            loss, inputs["node_ebd_ext"], create_graph=True,
        )
        self.assertIsNotNone(grad_node)
        self.assertTrue(grad_node.requires_grad)

        grad_sum = (grad_node ** 2).sum()
        grad_sum.backward()

        has_nonzero = False
        for name, param in layer.named_parameters():
            if param.grad is not None and (param.grad.abs() > 0).any():
                has_nonzero = True
                break
        self.assertTrue(has_nonzero, "No parameter received 2nd-order gradient")


# ======================================================================
# Test original forward still works (regression)
# ======================================================================


class TestOriginalForwardUnchanged(unittest.TestCase):
    """use_moe=False forward still works correctly."""

    def test_non_moe_forward(self):
        """Non-MoE layer with dynamic_sel=True, optim_update=False."""
        layer = _make_layer(use_moe=False)
        inputs = _make_inputs()
        del inputs["type_embedding"]
        n_out, e_out, a_out = layer(**inputs)

        self.assertEqual(n_out.shape, (NB, NLOC, N_DIM))
        self.assertEqual(e_out.shape, (N_EDGE, E_DIM))
        self.assertEqual(a_out.shape, (N_ANGLE, A_DIM))

    def test_non_moe_jit_script(self):
        """Non-MoE layer can be JIT-scripted without error."""
        layer = _make_layer(use_moe=False)
        scripted = torch.jit.script(layer)

        inputs = _make_inputs()
        del inputs["type_embedding"]
        n_out, e_out, a_out = scripted(**inputs)

        self.assertEqual(n_out.shape, (NB, NLOC, N_DIM))
        self.assertEqual(e_out.shape, (N_EDGE, E_DIM))
        self.assertEqual(a_out.shape, (N_ANGLE, A_DIM))


# ======================================================================
# Test config validation
# ======================================================================


class TestMoEConfigValidation(unittest.TestCase):
    """Invalid MoE configs should raise ValueError."""

    def test_requires_dynamic_sel(self):
        with self.assertRaises(ValueError):
            RepFlowLayer(
                **{**_BASE_KWARGS, "use_dynamic_sel": False},
                seed=42,
                use_moe=True, n_routing_experts=4, moe_topk=2,
            )

    def test_requires_no_optim_update(self):
        with self.assertRaises(ValueError):
            RepFlowLayer(
                **{**_BASE_KWARGS, "optim_update": True},
                seed=42,
                use_moe=True, n_routing_experts=4, moe_topk=2,
            )

    def test_requires_update_angle(self):
        with self.assertRaises(ValueError):
            RepFlowLayer(
                **{**_BASE_KWARGS, "update_angle": False},
                seed=42,
                use_moe=True, n_routing_experts=4, moe_topk=2,
            )

    def test_requires_dim_ratio(self):
        with self.assertRaises(ValueError):
            RepFlowLayer(
                **{**_BASE_KWARGS, "n_dim": 16, "e_dim": 8, "a_dim": 5},
                seed=42,
                use_moe=True, n_routing_experts=4, moe_topk=2,
            )


# ======================================================================
# Single-expert MoE vs non-MoE consistency
# ======================================================================


def _copy_non_moe_to_moe(non_moe_layer, moe_layer):
    """Copy non-MoE weights into MoE single-expert layer.

    Weight mapping:
    - M1 (node_self_mlp) -> moe_phase1.node_self_experts.routing_experts.0.mlp
    - M2 (node_sym_linear) -> moe_phase1.node_sym_experts.routing_experts.0.mlp
    - M3 (node_edge_linear) -> edge_experts merged output[:, :n_dim]
    - M4 (edge_self_linear) -> edge_experts merged output[:, n_dim:]
    - M5 (edge_angle_linear1) -> angle_experts merged output[:, :e_dim]
    - M7 (angle_self_linear) -> angle_experts merged output[:, e_dim:]
    - M6 (edge_angle_linear2) -> edge_angle_linear2_moe
    - Residuals: same names, direct copy
    """
    with torch.no_grad():
        # M1: node self.
        moe_e0_m1 = moe_layer.moe_phase1.node_self_experts.routing_experts[0].mlp
        moe_e0_m1.matrix.copy_(non_moe_layer.node_self_mlp.matrix)
        moe_e0_m1.bias.copy_(non_moe_layer.node_self_mlp.bias)

        # M2: node sym.
        moe_e0_m2 = moe_layer.moe_phase1.node_sym_experts.routing_experts[0].mlp
        moe_e0_m2.matrix.copy_(non_moe_layer.node_sym_linear.matrix)
        moe_e0_m2.bias.copy_(non_moe_layer.node_sym_linear.bias)

        # M3+M4: merged edge expert. W_merged = [W_M3 | W_M4], b_merged = [b_M3 | b_M4].
        moe_e0_edge = moe_layer.moe_phase1.edge_experts.routing_experts[0].mlp
        w_m3 = non_moe_layer.node_edge_linear.matrix  # [edge_info_dim, n_dim]
        w_m4 = non_moe_layer.edge_self_linear.matrix   # [edge_info_dim, e_dim]
        moe_e0_edge.matrix.copy_(torch.cat([w_m3, w_m4], dim=-1))
        b_m3 = non_moe_layer.node_edge_linear.bias  # [n_dim]
        b_m4 = non_moe_layer.edge_self_linear.bias   # [e_dim]
        moe_e0_edge.bias.copy_(torch.cat([b_m3, b_m4], dim=-1))

        # M5+M7: merged angle expert. W_merged = [W_M5 | W_M7], b_merged = [b_M5 | b_M7].
        moe_e0_angle = moe_layer.moe_phase1.angle_experts.routing_experts[0].mlp
        w_m5 = non_moe_layer.edge_angle_linear1.matrix  # [angle_dim, e_dim]
        w_m7 = non_moe_layer.angle_self_linear.matrix    # [angle_dim, a_dim]
        moe_e0_angle.matrix.copy_(torch.cat([w_m5, w_m7], dim=-1))
        b_m5 = non_moe_layer.edge_angle_linear1.bias  # [e_dim]
        b_m7 = non_moe_layer.angle_self_linear.bias    # [a_dim]
        moe_e0_angle.bias.copy_(torch.cat([b_m5, b_m7], dim=-1))

        # M6: edge_angle_linear2 -> edge_angle_linear2_moe.
        moe_layer.edge_angle_linear2_moe.matrix.copy_(
            non_moe_layer.edge_angle_linear2.matrix
        )
        moe_layer.edge_angle_linear2_moe.bias.copy_(
            non_moe_layer.edge_angle_linear2.bias
        )

        # Residuals: same ParameterList structure, direct copy.
        for src, dst in zip(non_moe_layer.n_residual, moe_layer.n_residual):
            dst.copy_(src)
        for src, dst in zip(non_moe_layer.e_residual, moe_layer.e_residual):
            dst.copy_(src)
        for src, dst in zip(non_moe_layer.a_residual, moe_layer.a_residual):
            dst.copy_(src)


class TestSingleExpertConsistency(unittest.TestCase):
    """Single expert (n_routing_experts=1, topk=1) MoE must match non-MoE exactly."""

    def _get_layers_and_inputs(self, seed=42):
        """Create matched non-MoE and MoE layers with copied weights."""
        non_moe = _make_layer(use_moe=False, seed=seed)
        moe = _make_layer(
            use_moe=True,
            n_routing_experts=1,
            moe_topk=1,
            n_shared_experts=0,
            seed=seed + 1000,  # different seed, weights will be overwritten
        )
        _copy_non_moe_to_moe(non_moe, moe)
        return non_moe, moe

    def test_forward_consistency(self):
        """Forward outputs match between non-MoE and single-expert MoE."""
        non_moe, moe = self._get_layers_and_inputs()
        inputs = _make_inputs(seed=99)

        # Non-MoE forward (no type_embedding).
        inputs_no_te = {k: v for k, v in inputs.items() if k != "type_embedding"}
        n_ref, e_ref, a_ref = non_moe(**inputs_no_te)

        # MoE forward.
        n_moe, e_moe, a_moe = moe(**inputs)

        torch.testing.assert_close(n_moe, n_ref, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(e_moe, e_ref, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(a_moe, a_ref, atol=1e-12, rtol=1e-10)

    def test_backward_consistency(self):
        """First-order gradients match between non-MoE and single-expert MoE."""
        non_moe, moe = self._get_layers_and_inputs()

        # Non-MoE.
        inputs_ref = _make_inputs(seed=99, requires_grad=True)
        inputs_ref_no_te = {k: v for k, v in inputs_ref.items() if k != "type_embedding"}
        n_ref, e_ref, a_ref = non_moe(**inputs_ref_no_te)
        loss_ref = (n_ref ** 2).sum() + (e_ref ** 2).sum() + (a_ref ** 2).sum()
        loss_ref.backward()

        # MoE.
        inputs_moe = _make_inputs(seed=99, requires_grad=True)
        n_moe, e_moe, a_moe = moe(**inputs_moe)
        loss_moe = (n_moe ** 2).sum() + (e_moe ** 2).sum() + (a_moe ** 2).sum()
        loss_moe.backward()

        # Compare input grads.
        for key in ["node_ebd_ext", "edge_ebd", "h2", "angle_ebd"]:
            torch.testing.assert_close(
                inputs_moe[key].grad, inputs_ref[key].grad,
                atol=1e-12, rtol=1e-10,
                msg=f"Gradient mismatch for {key}",
            )

    def test_second_order_consistency(self):
        """Second-order gradients match between non-MoE and single-expert MoE."""
        non_moe, moe = self._get_layers_and_inputs()

        # Non-MoE 2nd order.
        inputs_ref = _make_inputs(seed=99, requires_grad=True)
        inputs_ref_no_te = {k: v for k, v in inputs_ref.items() if k != "type_embedding"}
        n_ref, e_ref, a_ref = non_moe(**inputs_ref_no_te)
        loss_ref = (n_ref ** 2).sum() + (e_ref ** 2).sum() + (a_ref ** 2).sum()
        (grad_ref,) = torch.autograd.grad(
            loss_ref, inputs_ref["node_ebd_ext"], create_graph=True,
        )
        grad_loss_ref = (grad_ref ** 2).sum()
        grad_loss_ref.backward()
        grad2_ref = inputs_ref["node_ebd_ext"].grad.clone()

        # MoE 2nd order.
        inputs_moe = _make_inputs(seed=99, requires_grad=True)
        n_moe, e_moe, a_moe = moe(**inputs_moe)
        loss_moe = (n_moe ** 2).sum() + (e_moe ** 2).sum() + (a_moe ** 2).sum()
        (grad_moe,) = torch.autograd.grad(
            loss_moe, inputs_moe["node_ebd_ext"], create_graph=True,
        )
        grad_loss_moe = (grad_moe ** 2).sum()
        grad_loss_moe.backward()
        grad2_moe = inputs_moe["node_ebd_ext"].grad.clone()

        # Compare 1st-order grads.
        torch.testing.assert_close(
            grad_moe.detach(), grad_ref.detach(), atol=1e-12, rtol=1e-10,
        )
        # Compare 2nd-order grads.
        torch.testing.assert_close(grad2_moe, grad2_ref, atol=1e-12, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
