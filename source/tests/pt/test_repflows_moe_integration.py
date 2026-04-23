# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for MoE in DescrptBlockRepflows and DescrptDPA3 (Step 9).

Tests the full stack: RepFlowArgs → DescrptDPA3 → DescrptBlockRepflows → RepFlowLayer
with MoE enabled (use_moe=True).

Run with:
    CUDA_VISIBLE_DEVICES=0 python -m pytest source/tests/pt/test_repflows_moe_integration.py -v
"""

from __future__ import annotations

import unittest

import torch

from deepmd.dpmodel.descriptor.dpa3 import (
    RepFlowArgs,
)
from deepmd.pt.model.descriptor.dpa3 import (
    DescrptDPA3,
)
from deepmd.pt.model.descriptor.repflows import (
    DescrptBlockRepflows,
)
from deepmd.pt.utils.env import (
    DEVICE,
)

DTYPE = torch.float64
CPU_DEVICE = torch.device("cpu")

# ======================================================================
# Config (small, matches MoE constraints)
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
NLAYERS = 2

N_ROUTING_EXPERTS = 4
MOE_TOPK = 2
N_SHARED_EXPERTS = 1

NB = 1
NLOC = 6
NALL = NLOC + 4
NNEI = E_SEL


def _make_repflow_args(use_moe=True):
    """Create RepFlowArgs with MoE config."""
    return RepFlowArgs(
        n_dim=N_DIM,
        e_dim=E_DIM,
        a_dim=A_DIM,
        nlayers=NLAYERS,
        e_rcut=E_RCUT,
        e_rcut_smth=E_RCUT_SMTH,
        e_sel=E_SEL,
        a_rcut=A_RCUT,
        a_rcut_smth=A_RCUT_SMTH,
        a_sel=A_SEL,
        a_compress_rate=A_COMPRESS_RATE,
        a_compress_e_rate=A_COMPRESS_E_RATE,
        a_compress_use_split=True,
        n_multi_edge_message=1,
        axis_neuron=AXIS_NEURON,
        update_angle=True,
        update_style="res_residual",
        update_residual=0.1,
        update_residual_init="const",
        fix_stat_std=0.3,
        optim_update=False,
        smooth_edge_update=True,
        use_dynamic_sel=True,
        use_moe=use_moe,
        n_routing_experts=N_ROUTING_EXPERTS if use_moe else 0,
        moe_topk=MOE_TOPK if use_moe else 0,
        n_shared_experts=N_SHARED_EXPERTS if use_moe else 0,
    )


def _make_repflows_block(use_moe=True, seed=42):
    """Create a DescrptBlockRepflows with MoE."""
    block = DescrptBlockRepflows(
        e_rcut=E_RCUT,
        e_rcut_smth=E_RCUT_SMTH,
        e_sel=E_SEL,
        a_rcut=A_RCUT,
        a_rcut_smth=A_RCUT_SMTH,
        a_sel=A_SEL,
        ntypes=NTYPES,
        nlayers=NLAYERS,
        n_dim=N_DIM,
        e_dim=E_DIM,
        a_dim=A_DIM,
        a_compress_rate=A_COMPRESS_RATE,
        a_compress_e_rate=A_COMPRESS_E_RATE,
        a_compress_use_split=True,
        n_multi_edge_message=1,
        axis_neuron=AXIS_NEURON,
        update_angle=True,
        activation_function="silu",
        update_style="res_residual",
        update_residual=0.1,
        update_residual_init="const",
        fix_stat_std=0.3,
        optim_update=False,
        smooth_edge_update=True,
        use_dynamic_sel=True,
        precision="float64",
        seed=seed,
        use_moe=use_moe,
        n_routing_experts=N_ROUTING_EXPERTS if use_moe else 0,
        moe_topk=MOE_TOPK if use_moe else 0,
        n_shared_experts=N_SHARED_EXPERTS if use_moe else 0,
    )
    return block.to(CPU_DEVICE)


def _make_dpa3(use_moe=True, seed=42):
    """Create a DescrptDPA3 with MoE."""
    repflow_args = _make_repflow_args(use_moe)
    dpa3 = DescrptDPA3(
        ntypes=NTYPES,
        repflow=repflow_args,
        activation_function="silu",
        precision="float64",
        seed=seed,
        use_loc_mapping=True,
        type_map=["H", "O"],
    )
    return dpa3.to(CPU_DEVICE)


def _make_repflows_inputs(seed=0, requires_grad=False):
    """Create inputs for DescrptBlockRepflows.forward().

    Returns nlist, extended_coord, extended_atype, extended_atype_embd, mapping.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    # Coordinates: nf x nall x 3
    extended_coord = torch.randn(
        NB, NALL * 3, device=CPU_DEVICE, dtype=DTYPE, generator=gen,
    )
    if requires_grad:
        extended_coord = extended_coord.detach().requires_grad_(True)

    # Atom types: nf x nall (int)
    extended_atype = torch.randint(
        0, NTYPES, (NB, NALL), device=CPU_DEVICE, generator=gen,
    )

    # Neighbor list: nf x nloc x nnei
    nlist = torch.randint(
        0, NALL, (NB, NLOC, NNEI), device=CPU_DEVICE, generator=gen,
    )
    # Mark some as -1 (padding)
    mask = torch.rand(NB, NLOC, NNEI, device=CPU_DEVICE, generator=gen) > 0.2
    nlist = torch.where(mask, nlist, torch.tensor(-1, device=CPU_DEVICE))

    # extended_atype_embd: nf x nall x n_dim (simulates type embedding)
    extended_atype_embd = torch.randn(
        NB, NALL, N_DIM, device=CPU_DEVICE, dtype=DTYPE, generator=gen,
    )
    if requires_grad:
        extended_atype_embd = extended_atype_embd.detach().requires_grad_(True)

    # mapping: nf x nall (maps extended to local index)
    mapping = torch.zeros(NB, NALL, device=CPU_DEVICE, dtype=torch.long)
    mapping[:, :NLOC] = torch.arange(NLOC, device=CPU_DEVICE)
    # Ghost atoms map back to random local atoms
    for i in range(NLOC, NALL):
        mapping[:, i] = torch.randint(
            0, NLOC, (NB,), device=CPU_DEVICE, generator=gen,
        )

    return nlist, extended_coord, extended_atype, extended_atype_embd, mapping


def _make_dpa3_inputs(seed=0, requires_grad=False):
    """Create inputs for DescrptDPA3.forward().

    Returns extended_coord, extended_atype, nlist, mapping.
    Uses well-separated coordinates and self-excluding nlist
    to avoid NaN from 1/r in prod_env_mat.
    """
    # Place atoms on a grid (spacing=3.0) to avoid near-zero distances.
    coords = []
    for i in range(NALL):
        x = float(i % 4) * 3.0
        y = float((i // 4) % 4) * 3.0
        z = float(i // 16) * 3.0
        coords.extend([x, y, z])
    extended_coord = torch.tensor(
        coords, device=CPU_DEVICE, dtype=DTYPE,
    ).unsqueeze(0).expand(NB, -1).clone()
    if requires_grad:
        extended_coord = extended_coord.detach().requires_grad_(True)

    extended_atype = torch.zeros(NB, NALL, device=CPU_DEVICE, dtype=torch.long)
    extended_atype[:, NALL // 2:] = 1

    # Build nlist: each local atom i has neighbors from other atoms (no self-ref).
    nlist = torch.full(
        (NB, NLOC, NNEI), -1, device=CPU_DEVICE, dtype=torch.long,
    )
    for i in range(NLOC):
        nbrs = [j for j in range(NALL) if j != i][:NNEI]
        for k, j in enumerate(nbrs):
            nlist[0, i, k] = j

    mapping = torch.zeros(NB, NALL, device=CPU_DEVICE, dtype=torch.long)
    mapping[:, :NLOC] = torch.arange(NLOC, device=CPU_DEVICE)
    for i in range(NLOC, NALL):
        mapping[:, i] = i % NLOC

    return extended_coord, extended_atype, nlist, mapping


# ======================================================================
# Test RepFlowArgs
# ======================================================================


class TestRepFlowArgsMoE(unittest.TestCase):
    """Test RepFlowArgs MoE serialization."""

    def test_moe_params_stored(self):
        args = _make_repflow_args(use_moe=True)
        self.assertTrue(args.use_moe)
        self.assertEqual(args.n_routing_experts, N_ROUTING_EXPERTS)
        self.assertEqual(args.moe_topk, MOE_TOPK)
        self.assertEqual(args.n_shared_experts, N_SHARED_EXPERTS)

    def test_serialize_roundtrip(self):
        args = _make_repflow_args(use_moe=True)
        data = args.serialize()
        args2 = RepFlowArgs.deserialize(data)

        self.assertEqual(args2.use_moe, args.use_moe)
        self.assertEqual(args2.n_routing_experts, args.n_routing_experts)
        self.assertEqual(args2.moe_topk, args.moe_topk)
        self.assertEqual(args2.n_shared_experts, args.n_shared_experts)

    def test_serialize_no_moe(self):
        args = _make_repflow_args(use_moe=False)
        data = args.serialize()
        self.assertFalse(data["use_moe"])
        self.assertEqual(data["n_routing_experts"], 0)

    def test_dict_constructor(self):
        """RepFlowArgs can be passed as dict to DescrptDPA3."""
        args = _make_repflow_args(use_moe=True)
        data = args.serialize()
        # Constructing DescrptDPA3 with dict should work.
        dpa3 = DescrptDPA3(
            ntypes=NTYPES,
            repflow=data,
            precision="float64",
            type_map=["H", "O"],
        )
        self.assertTrue(dpa3.repflow_args.use_moe)
        self.assertEqual(dpa3.repflow_args.n_routing_experts, N_ROUTING_EXPERTS)


# ======================================================================
# Test DescrptBlockRepflows with MoE
# ======================================================================


class TestDescrptBlockRepflowsMoE(unittest.TestCase):
    """Test DescrptBlockRepflows forward/backward with MoE."""

    def test_forward_shapes(self):
        block = _make_repflows_block(use_moe=True)
        nlist, ext_coord, ext_atype, ext_embd, mapping = _make_repflows_inputs()

        node_ebd, edge_ebd, h2, rot_mat, sw = block(
            nlist, ext_coord, ext_atype, ext_embd, mapping,
        )

        self.assertEqual(node_ebd.shape, (NB, NLOC, N_DIM))
        self.assertIsNotNone(edge_ebd)
        self.assertIsNotNone(h2)
        self.assertIsNotNone(rot_mat)
        self.assertIsNotNone(sw)

    def test_backward(self):
        block = _make_repflows_block(use_moe=True)
        nlist, ext_coord, ext_atype, ext_embd, mapping = _make_repflows_inputs(
            requires_grad=True,
        )

        node_ebd, edge_ebd, h2, rot_mat, sw = block(
            nlist, ext_coord, ext_atype, ext_embd, mapping,
        )
        loss = node_ebd.pow(2).sum()
        loss.backward()

        # ext_embd feeds directly into the MoE layers via type_embedding.
        self.assertIsNotNone(ext_embd.grad, "ext_embd has no gradient")

    def test_moe_params_present(self):
        """MoE layers should have MoE-specific parameters."""
        block = _make_repflows_block(use_moe=True)
        param_names = [n for n, _ in block.named_parameters()]
        # With shared 3D tensor: routing_matrix and routing_bias
        has_routing_expert = any(
            ".routing_matrix" in n or ".routing_bias" in n
            for n in param_names
        )
        has_router = any("router" in n for n in param_names)
        self.assertTrue(has_routing_expert, "No routing expert parameters found")
        self.assertTrue(has_router, "No router parameters found")

    def test_non_moe_still_works(self):
        """Non-MoE DescrptBlockRepflows still works correctly."""
        block = _make_repflows_block(use_moe=False)
        nlist, ext_coord, ext_atype, ext_embd, mapping = _make_repflows_inputs()

        node_ebd, edge_ebd, h2, rot_mat, sw = block(
            nlist, ext_coord, ext_atype, ext_embd, mapping,
        )
        self.assertEqual(node_ebd.shape, (NB, NLOC, N_DIM))

    def test_multi_layer(self):
        """MoE works across multiple layers."""
        block = _make_repflows_block(use_moe=True)
        # Verify all layers are MoE-enabled.
        for layer in block.layers:
            self.assertTrue(layer.use_moe)
            self.assertIsNotNone(layer.node_router)

    def test_second_order(self):
        """Second-order derivatives work through DescrptBlockRepflows with MoE."""
        block = _make_repflows_block(use_moe=True)
        nlist, ext_coord, ext_atype, ext_embd, mapping = _make_repflows_inputs(
            requires_grad=True,
        )

        node_ebd, edge_ebd, h2, rot_mat, sw = block(
            nlist, ext_coord, ext_atype, ext_embd, mapping,
        )
        loss = node_ebd.pow(2).sum()

        (grad_embd,) = torch.autograd.grad(
            loss, ext_embd, create_graph=True,
        )
        self.assertTrue(grad_embd.requires_grad)

        grad_loss = grad_embd.pow(2).sum()
        grad_loss.backward()

        self.assertIsNotNone(ext_embd.grad, "ext_embd has no 2nd-order gradient")


# ======================================================================
# Test DescrptDPA3 with MoE
# ======================================================================


class TestDescrptDPA3MoE(unittest.TestCase):
    """Test DescrptDPA3 forward/backward with MoE."""

    def test_forward_shapes(self):
        dpa3 = _make_dpa3(use_moe=True)
        ext_coord, ext_atype, nlist, mapping = _make_dpa3_inputs()

        node_ebd, rot_mat, edge_ebd, h2, sw = dpa3(
            ext_coord, ext_atype, nlist, mapping,
        )

        self.assertEqual(node_ebd.shape[0], NB)
        self.assertEqual(node_ebd.shape[1], NLOC)
        # dim_out = n_dim (no concat_output_tebd)
        self.assertEqual(node_ebd.shape[2], N_DIM)

    def test_backward(self):
        dpa3 = _make_dpa3(use_moe=True)
        ext_coord, ext_atype, nlist, mapping = _make_dpa3_inputs(requires_grad=True)

        node_ebd, rot_mat, edge_ebd, h2, sw = dpa3(
            ext_coord, ext_atype, nlist, mapping,
        )
        loss = node_ebd.pow(2).sum()
        loss.backward()

        self.assertIsNotNone(ext_coord.grad)

    def test_second_order(self):
        """Second-order derivatives through DescrptDPA3 with MoE."""
        dpa3 = _make_dpa3(use_moe=True)
        ext_coord, ext_atype, nlist, mapping = _make_dpa3_inputs(requires_grad=True)

        node_ebd, rot_mat, edge_ebd, h2, sw = dpa3(
            ext_coord, ext_atype, nlist, mapping,
        )
        loss = node_ebd.pow(2).sum()

        (grad_coord,) = torch.autograd.grad(
            loss, ext_coord, create_graph=True,
        )
        self.assertTrue(grad_coord.requires_grad)

        grad_loss = grad_coord.pow(2).sum()
        grad_loss.backward()

        self.assertIsNotNone(ext_coord.grad)

    def test_non_moe_forward(self):
        """Non-MoE DescrptDPA3 still works."""
        dpa3 = _make_dpa3(use_moe=False)
        ext_coord, ext_atype, nlist, mapping = _make_dpa3_inputs()

        node_ebd, rot_mat, edge_ebd, h2, sw = dpa3(
            ext_coord, ext_atype, nlist, mapping,
        )
        self.assertEqual(node_ebd.shape[0], NB)
        self.assertEqual(node_ebd.shape[1], NLOC)

    def test_moe_config_propagated(self):
        """MoE config propagates from DescrptDPA3 to RepFlowLayers."""
        dpa3 = _make_dpa3(use_moe=True)
        self.assertTrue(dpa3.repflow_args.use_moe)
        self.assertTrue(dpa3.repflows.use_moe)
        for layer in dpa3.repflows.layers:
            self.assertTrue(layer.use_moe)

    def test_deterministic(self):
        """Same seed produces same outputs."""
        dpa3_a = _make_dpa3(use_moe=True, seed=42)
        dpa3_b = _make_dpa3(use_moe=True, seed=42)

        ext_coord, ext_atype, nlist, mapping = _make_dpa3_inputs(seed=99)

        n_a, _, _, _, _ = dpa3_a(ext_coord, ext_atype, nlist, mapping)
        n_b, _, _, _, _ = dpa3_b(ext_coord, ext_atype, nlist, mapping)

        torch.testing.assert_close(n_a, n_b)

    def test_ep_params_propagated(self):
        """EP params propagate: MoE layers have MoE packer with ep_size=1."""
        dpa3 = _make_dpa3(use_moe=True)
        for layer in dpa3.repflows.layers:
            self.assertTrue(layer.use_moe)
            # Verify the MoE packer exists (it receives ep_group/ep_rank/ep_size).
            self.assertIsNotNone(layer.moe_phase1)


if __name__ == "__main__":
    unittest.main()
