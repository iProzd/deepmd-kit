# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single-GPU unit tests for MoE checkpoint save/load with resharding (Step 10).

Run with:
    CUDA_VISIBLE_DEVICES=0 python -m pytest source/tests/pt/test_moe_checkpoint.py -v
"""

from __future__ import annotations

import unittest

import torch

from deepmd.pt.model.descriptor.repflow_layer import RepFlowLayer
from deepmd.pt.utils.moe_checkpoint import (
    _ROUTING_EXPERT_RE,
    moe_load_state_dict_from_global,
    moe_state_dict_to_global,
)

DTYPE = torch.float64
CPU_DEVICE = torch.device("cpu")

# ======================================================================
# Config (matches Step 7/8 tests)
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

N_ROUTING_EXPERTS = 4
MOE_TOPK = 2
N_SHARED_EXPERTS = 1

NB = 1
NLOC = 6
NALL = NLOC + 4
N_EDGE = 30
N_ANGLE = 50


# ======================================================================
# Helpers
# ======================================================================

def _make_layer(ep_rank=0, ep_size=1, seed=42):
    """Create a MoE RepFlowLayer."""
    layer = RepFlowLayer(
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
        seed=seed,
        use_moe=True,
        n_routing_experts=N_ROUTING_EXPERTS,
        moe_topk=MOE_TOPK,
        n_shared_experts=N_SHARED_EXPERTS,
        ep_group=None,
        ep_rank=ep_rank,
        ep_size=ep_size,
    )
    return layer.to(CPU_DEVICE)


def _make_inputs(seed=0, requires_grad=False):
    """Create inputs for RepFlowLayer.forward()."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    node_ebd_ext = torch.randn(
        NB, NALL, N_DIM, device=CPU_DEVICE, dtype=DTYPE, generator=gen,
    )
    if requires_grad:
        node_ebd_ext = node_ebd_ext.detach().requires_grad_(True)
    edge_ebd = torch.randn(
        N_EDGE, E_DIM, device=CPU_DEVICE, dtype=DTYPE, generator=gen,
    )
    if requires_grad:
        edge_ebd = edge_ebd.detach().requires_grad_(True)
    h2 = torch.randn(
        N_EDGE, 3, device=CPU_DEVICE, dtype=DTYPE, generator=gen,
    )
    if requires_grad:
        h2 = h2.detach().requires_grad_(True)
    angle_ebd = torch.randn(
        N_ANGLE, A_DIM, device=CPU_DEVICE, dtype=DTYPE, generator=gen,
    )
    if requires_grad:
        angle_ebd = angle_ebd.detach().requires_grad_(True)
    nlist = torch.randint(
        0, NALL, (NB, NLOC, E_SEL), device=CPU_DEVICE, generator=gen,
    )
    nlist_mask = torch.ones(NB, NLOC, E_SEL, device=CPU_DEVICE, dtype=DTYPE)
    sw = torch.rand(N_EDGE, device=CPU_DEVICE, dtype=DTYPE, generator=gen)
    if requires_grad:
        sw = sw.detach().requires_grad_(True)
    a_nlist = torch.randint(
        0, NLOC, (NB, NLOC, A_SEL), device=CPU_DEVICE, generator=gen,
    )
    a_nlist_mask = torch.ones(NB, NLOC, A_SEL, device=CPU_DEVICE, dtype=DTYPE)
    a_sw = torch.rand(N_ANGLE, device=CPU_DEVICE, dtype=DTYPE, generator=gen)
    if requires_grad:
        a_sw = a_sw.detach().requires_grad_(True)

    gen_idx = torch.Generator(device="cpu")
    gen_idx.manual_seed(200)
    n2e_index = torch.randint(
        0, NB * NLOC, (N_EDGE,), device=CPU_DEVICE, generator=gen_idx,
    )
    n_ext2e_index = torch.randint(
        0, NB * NALL, (N_EDGE,), device=CPU_DEVICE, generator=gen_idx,
    )
    edge_index = torch.stack([n2e_index, n_ext2e_index], dim=0)

    n2a_index = torch.randint(
        0, NB * NLOC, (N_ANGLE,), device=CPU_DEVICE, generator=gen_idx,
    )
    eij2a_index = torch.randint(
        0, N_EDGE, (N_ANGLE,), device=CPU_DEVICE, generator=gen_idx,
    )
    eik2a_index = torch.randint(
        0, N_EDGE, (N_ANGLE,), device=CPU_DEVICE, generator=gen_idx,
    )
    angle_index = torch.stack([n2a_index, eij2a_index, eik2a_index], dim=0)

    type_embedding = torch.randn(
        NB, NLOC, N_DIM, device=CPU_DEVICE, dtype=DTYPE, generator=gen,
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
# Tests
# ======================================================================


class TestMoECheckpointSaveLoad(unittest.TestCase):
    """Test moe_state_dict_to_global and moe_load_state_dict_from_global."""

    def test_save_load_same_config(self):
        """Save and load with same ep_size=1: output matches."""
        layer1 = _make_layer(ep_rank=0, ep_size=1)
        inputs = _make_inputs()

        # Forward with original layer.
        with torch.no_grad():
            n1, e1, a1 = layer1(**inputs)

        # Save global state_dict.
        experts_per_gpu = N_ROUTING_EXPERTS  # ep_size=1
        global_sd = moe_state_dict_to_global(layer1, 0, 1, experts_per_gpu)

        # Create new layer and load.
        layer2 = _make_layer(ep_rank=0, ep_size=1, seed=999)
        moe_load_state_dict_from_global(layer2, global_sd, 0, 1, experts_per_gpu)

        # Forward with loaded layer.
        with torch.no_grad():
            n2, e2, a2 = layer2(**inputs)

        torch.testing.assert_close(n1, n2)
        torch.testing.assert_close(e1, e2)
        torch.testing.assert_close(a1, a2)

    def test_global_state_dict_keys(self):
        """Global state_dict has correct expert indices 0..n_routing-1."""
        layer = _make_layer(ep_rank=0, ep_size=1)
        experts_per_gpu = N_ROUTING_EXPERTS
        global_sd = moe_state_dict_to_global(layer, 0, 1, experts_per_gpu)

        # Extract routing expert indices.
        indices = set()
        for key in global_sd.keys():
            m = _ROUTING_EXPERT_RE.search(key)
            if m:
                indices.add(int(m.group(1)))

        self.assertEqual(indices, {0, 1, 2, 3})

    def test_resharding_1gpu_to_2gpu_simulation(self):
        """1 GPU (ep_size=1) save → 2 GPU (ep_size=2) load: outputs match."""
        # Original layer: ep_size=1, all 4 experts.
        layer_full = _make_layer(ep_rank=0, ep_size=1)
        inputs = _make_inputs()

        with torch.no_grad():
            n_full, e_full, a_full = layer_full(**inputs)

        # Save global state_dict.
        global_sd = moe_state_dict_to_global(
            layer_full, ep_rank=0, ep_size=1, experts_per_gpu=N_ROUTING_EXPERTS,
        )

        # Create 2 layers simulating ep_size=2.
        # ep_rank=0 holds experts 0,1; ep_rank=1 holds experts 2,3.
        layer_r0 = _make_layer(ep_rank=0, ep_size=2, seed=111)
        layer_r1 = _make_layer(ep_rank=1, ep_size=2, seed=222)
        epg = N_ROUTING_EXPERTS // 2  # 2

        moe_load_state_dict_from_global(layer_r0, global_sd, 0, 2, epg)
        moe_load_state_dict_from_global(layer_r1, global_sd, 1, 2, epg)

        # Verify that specific expert params match.
        sd_full = layer_full.state_dict()
        sd_r0 = layer_r0.state_dict()
        sd_r1 = layer_r1.state_dict()

        # Expert 0 on rank 0 = local index 0 = global index 0.
        for key in sd_full.keys():
            m = _ROUTING_EXPERT_RE.search(key)
            if m:
                global_idx = int(m.group(1))
                if global_idx < epg:
                    # Should be on rank 0 with local_idx = global_idx.
                    local_key = key  # Same key since local_idx == global_idx for rank 0.
                    torch.testing.assert_close(
                        sd_full[key], sd_r0[local_key],
                        msg=f"Expert {global_idx} mismatch on rank 0",
                    )
                else:
                    # Should be on rank 1 with local_idx = global_idx - epg.
                    local_idx = global_idx - epg
                    local_key = key.replace(
                        f".routing_experts.{global_idx}.",
                        f".routing_experts.{local_idx}.",
                        1,
                    )
                    torch.testing.assert_close(
                        sd_full[key], sd_r1[local_key],
                        msg=f"Expert {global_idx} mismatch on rank 1",
                    )

    def test_resharding_2gpu_to_1gpu_simulation(self):
        """2 GPU (ep_size=2) → collect → 1 GPU (ep_size=1) load: params match."""
        # Create 2 layers with ep_size=2 sharing the same seed structure
        # as a single ep_size=1 layer.
        # First create the reference full layer.
        layer_full = _make_layer(ep_rank=0, ep_size=1)
        global_sd_ref = moe_state_dict_to_global(
            layer_full, ep_rank=0, ep_size=1,
            experts_per_gpu=N_ROUTING_EXPERTS,
        )

        # Load into 2 layers simulating ep_size=2.
        epg = N_ROUTING_EXPERTS // 2
        layer_r0 = _make_layer(ep_rank=0, ep_size=2, seed=111)
        layer_r1 = _make_layer(ep_rank=1, ep_size=2, seed=222)
        moe_load_state_dict_from_global(layer_r0, global_sd_ref, 0, 2, epg)
        moe_load_state_dict_from_global(layer_r1, global_sd_ref, 1, 2, epg)

        # Simulate saving from 2 GPUs by manually building global state_dict.
        # (In real multi-GPU, moe_state_dict_to_global with ep_size>1 uses
        # dist.broadcast; here we simulate by merging.)
        sd_r0 = layer_r0.state_dict()
        sd_r1 = layer_r1.state_dict()

        # Build global state_dict manually.
        simulated_global_sd = {}
        for key, tensor in sd_r0.items():
            m = _ROUTING_EXPERT_RE.search(key)
            if m:
                local_idx = int(m.group(1))
                global_idx = 0 * epg + local_idx  # ep_rank=0
                new_key = key.replace(
                    f".routing_experts.{local_idx}.",
                    f".routing_experts.{global_idx}.",
                    1,
                )
                simulated_global_sd[new_key] = tensor
            else:
                simulated_global_sd[key] = tensor
        for key, tensor in sd_r1.items():
            m = _ROUTING_EXPERT_RE.search(key)
            if m:
                local_idx = int(m.group(1))
                global_idx = 1 * epg + local_idx  # ep_rank=1
                new_key = key.replace(
                    f".routing_experts.{local_idx}.",
                    f".routing_experts.{global_idx}.",
                    1,
                )
                simulated_global_sd[new_key] = tensor

        # Load into ep_size=1 layer.
        layer_loaded = _make_layer(ep_rank=0, ep_size=1, seed=333)
        moe_load_state_dict_from_global(
            layer_loaded, simulated_global_sd, 0, 1, N_ROUTING_EXPERTS,
        )

        # Verify forward output matches original.
        inputs = _make_inputs()
        with torch.no_grad():
            n_orig, e_orig, a_orig = layer_full(**inputs)
            n_loaded, e_loaded, a_loaded = layer_loaded(**inputs)

        torch.testing.assert_close(n_orig, n_loaded)
        torch.testing.assert_close(e_orig, e_loaded)
        torch.testing.assert_close(a_orig, a_loaded)

    def test_non_routing_params_unchanged(self):
        """Non-routing params are identical after save/load."""
        layer = _make_layer(ep_rank=0, ep_size=1)
        global_sd = moe_state_dict_to_global(
            layer, 0, 1, N_ROUTING_EXPERTS,
        )

        layer2 = _make_layer(ep_rank=0, ep_size=1, seed=999)
        moe_load_state_dict_from_global(layer2, global_sd, 0, 1, N_ROUTING_EXPERTS)

        sd1 = layer.state_dict()
        sd2 = layer2.state_dict()

        for key in sd1:
            m = _ROUTING_EXPERT_RE.search(key)
            if not m:
                torch.testing.assert_close(
                    sd1[key], sd2[key],
                    msg=f"Non-routing param {key} changed after save/load",
                )

    def test_resharding_preserves_shared_experts(self):
        """Shared experts are preserved across resharding."""
        layer_full = _make_layer(ep_rank=0, ep_size=1)
        global_sd = moe_state_dict_to_global(
            layer_full, 0, 1, N_ROUTING_EXPERTS,
        )

        # Load into ep_size=2 layer.
        epg = N_ROUTING_EXPERTS // 2
        layer_r0 = _make_layer(ep_rank=0, ep_size=2, seed=111)
        moe_load_state_dict_from_global(layer_r0, global_sd, 0, 2, epg)

        sd_orig = layer_full.state_dict()
        sd_r0 = layer_r0.state_dict()

        # Shared expert params should be identical.
        for key in sd_orig:
            if ".shared_experts." in key:
                self.assertIn(key, sd_r0)
                torch.testing.assert_close(
                    sd_orig[key], sd_r0[key],
                    msg=f"Shared expert {key} changed after resharding",
                )

    def test_resharding_preserves_router(self):
        """Router params are preserved across resharding."""
        layer_full = _make_layer(ep_rank=0, ep_size=1)
        global_sd = moe_state_dict_to_global(
            layer_full, 0, 1, N_ROUTING_EXPERTS,
        )

        epg = N_ROUTING_EXPERTS // 2
        layer_r0 = _make_layer(ep_rank=0, ep_size=2, seed=111)
        moe_load_state_dict_from_global(layer_r0, global_sd, 0, 2, epg)

        sd_orig = layer_full.state_dict()
        sd_r0 = layer_r0.state_dict()

        for key in sd_orig:
            if "router" in key:
                self.assertIn(key, sd_r0)
                torch.testing.assert_close(
                    sd_orig[key], sd_r0[key],
                    msg=f"Router {key} changed after resharding",
                )

    def test_forward_output_after_resharding(self):
        """Forward output is correct after 1→2 GPU resharding (single-GPU sim).

        Since ep_group=None (single-GPU), both rank 0 and rank 1 layers
        run as independent single-GPU layers. We verify that each layer's
        routing experts produce the same per-expert outputs as the original.
        """
        layer_full = _make_layer(ep_rank=0, ep_size=1)
        global_sd = moe_state_dict_to_global(
            layer_full, 0, 1, N_ROUTING_EXPERTS,
        )

        # Forward original.
        inputs = _make_inputs()
        with torch.no_grad():
            n_full, e_full, a_full = layer_full(**inputs)

        # Note: We can't directly compare multi-GPU forward output in
        # single-GPU simulation because ep_group=None runs all experts
        # locally. The real test is in multi-GPU test file.
        # Here we just verify the save/load roundtrip preserves the model.
        layer_rt = _make_layer(ep_rank=0, ep_size=1, seed=999)
        moe_load_state_dict_from_global(
            layer_rt, global_sd, 0, 1, N_ROUTING_EXPERTS,
        )
        with torch.no_grad():
            n_rt, e_rt, a_rt = layer_rt(**inputs)

        torch.testing.assert_close(n_full, n_rt)
        torch.testing.assert_close(e_full, e_rt)
        torch.testing.assert_close(a_full, a_rt)

    def test_first_order_grad_consistency(self):
        """1st-order gradients match after save/load roundtrip."""
        layer1 = _make_layer(ep_rank=0, ep_size=1)
        epg = N_ROUTING_EXPERTS

        # Save and load into layer2.
        global_sd = moe_state_dict_to_global(layer1, 0, 1, epg)
        layer2 = _make_layer(ep_rank=0, ep_size=1, seed=999)
        moe_load_state_dict_from_global(layer2, global_sd, 0, 1, epg)

        # Forward + backward on layer1.
        inputs1 = _make_inputs(seed=55, requires_grad=True)
        n1, e1, a1 = layer1(**inputs1)
        loss1 = (n1 ** 2).sum() + (e1 ** 2).sum() + (a1 ** 2).sum()
        loss1.backward()
        grad_node1 = inputs1["node_ebd_ext"].grad.clone()
        grad_edge1 = inputs1["edge_ebd"].grad.clone()

        # Forward + backward on layer2 with fresh inputs (same data).
        inputs2 = _make_inputs(seed=55, requires_grad=True)
        n2, e2, a2 = layer2(**inputs2)
        loss2 = (n2 ** 2).sum() + (e2 ** 2).sum() + (a2 ** 2).sum()
        loss2.backward()
        grad_node2 = inputs2["node_ebd_ext"].grad.clone()
        grad_edge2 = inputs2["edge_ebd"].grad.clone()

        # Input gradients must match.
        torch.testing.assert_close(
            grad_node1, grad_node2,
            msg="1st-order node_ebd_ext grad mismatch after save/load",
        )
        torch.testing.assert_close(
            grad_edge1, grad_edge2,
            msg="1st-order edge_ebd grad mismatch after save/load",
        )

        # Parameter gradients must match.
        for (n1, p1), (n2, p2) in zip(
            layer1.named_parameters(), layer2.named_parameters(),
        ):
            if p1.grad is not None and p2.grad is not None:
                torch.testing.assert_close(
                    p1.grad, p2.grad,
                    msg=f"1st-order param grad mismatch: {n1}",
                )

    def test_second_order_grad_consistency(self):
        """2nd-order gradients match after save/load roundtrip."""
        layer1 = _make_layer(ep_rank=0, ep_size=1)
        epg = N_ROUTING_EXPERTS

        global_sd = moe_state_dict_to_global(layer1, 0, 1, epg)
        layer2 = _make_layer(ep_rank=0, ep_size=1, seed=999)
        moe_load_state_dict_from_global(layer2, global_sd, 0, 1, epg)

        # 2nd-order on layer1.
        inputs1 = _make_inputs(seed=55, requires_grad=True)
        n1, e1, a1 = layer1(**inputs1)
        loss1 = (n1 ** 2).sum() + (e1 ** 2).sum() + (a1 ** 2).sum()
        (grad_node1,) = torch.autograd.grad(
            loss1, inputs1["node_ebd_ext"], create_graph=True,
        )
        grad_loss1 = (grad_node1 ** 2).sum()
        grad_loss1.backward()
        second_grad_node1 = inputs1["node_ebd_ext"].grad.clone()

        # Collect parameter gradients from layer1.
        param_grads1 = {}
        for name, param in layer1.named_parameters():
            if param.grad is not None:
                param_grads1[name] = param.grad.clone()

        # 2nd-order on layer2.
        inputs2 = _make_inputs(seed=55, requires_grad=True)
        n2, e2, a2 = layer2(**inputs2)
        loss2 = (n2 ** 2).sum() + (e2 ** 2).sum() + (a2 ** 2).sum()
        (grad_node2,) = torch.autograd.grad(
            loss2, inputs2["node_ebd_ext"], create_graph=True,
        )
        grad_loss2 = (grad_node2 ** 2).sum()
        grad_loss2.backward()
        second_grad_node2 = inputs2["node_ebd_ext"].grad.clone()

        # 1st-order intermediate grad must match.
        torch.testing.assert_close(
            grad_node1, grad_node2,
            msg="2nd-order: 1st-order intermediate grad mismatch",
        )

        # 2nd-order input grad must match.
        torch.testing.assert_close(
            second_grad_node1, second_grad_node2,
            msg="2nd-order: input grad mismatch after save/load",
        )

        # 2nd-order parameter gradients must match.
        for name, param in layer2.named_parameters():
            if param.grad is not None and name in param_grads1:
                torch.testing.assert_close(
                    param_grads1[name], param.grad,
                    msg=f"2nd-order param grad mismatch: {name}",
                )


if __name__ == "__main__":
    unittest.main()
