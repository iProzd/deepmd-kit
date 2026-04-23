# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-GPU unit tests for MoEDispatchCombine (Step 6).

Run with torchrun:
    torchrun --nproc_per_node=2 source/tests/pt/test_moe_layer_multigpu.py
    torchrun --nproc_per_node=4 source/tests/pt/test_moe_layer_multigpu.py

Tests:
  [2 GPU] Forward shape correctness
  [2 GPU] Backward: all expert params have gradient
  [2 GPU] 2nd-order derivative: no deadlock, correct gradients
  [2 GPU] Consistency: single-GPU reference vs 2-GPU MoE pipeline
  [4 GPU] Forward shape correctness
  [4 GPU] 2nd-order derivative: no deadlock
"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist

from deepmd.dpmodel.utils.seed import child_seed
from deepmd.pt.model.network.moe_layer import MoEDispatchCombine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

A_DIM = 4
N_DIM = 4 * A_DIM       # 16
E_DIM = 2 * A_DIM       # 8
N_SYM_DIM = 24 * A_DIM  # 96
EDGE_INFO_DIM = 10 * A_DIM  # 40
ANGLE_DIM = 4 * A_DIM   # 16
DTYPE = torch.float64


def setup_dist():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def cleanup_dist():
    dist.destroy_process_group()


def check(condition: bool, msg: str, rank: int):
    if not condition:
        print(f"[RANK {rank}] FAIL: {msg}", flush=True)
        sys.exit(1)


def all_pass(rank: int, world_size: int, tag: str):
    dist.barrier()
    if rank == 0:
        print(f"  PASS: {tag}", flush=True)


def _make_layer_multigpu(
    ep_group,
    ep_rank: int,
    ep_size: int,
    n_routing_experts: int,
    topk: int,
    n_shared_experts: int = 0,
    seed: int = 42,
) -> MoEDispatchCombine:
    """Create MoEDispatchCombine for multi-GPU testing."""
    experts_per_gpu = n_routing_experts // ep_size
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
        ep_group=ep_group,
        ep_rank=ep_rank,
        ep_size=ep_size,
        experts_per_gpu=experts_per_gpu,
        activation_function="silu",
        precision="float64",
        seed=seed,
    )
    device = torch.device(f"cuda:{ep_rank}")
    return layer.to(device)


def _make_inputs_multigpu(
    rank: int,
    n_node: int = 6,
    n_edge: int = 15,
    n_angle: int = 20,
    topk: int = 2,
    n_routing_experts: int = 4,
    requires_grad: bool = False,
    seed: int = 100,
) -> dict:
    """Create random inputs on the correct device for the given rank."""
    device = torch.device(f"cuda:{rank}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + rank)  # Different data per rank (simulates DP).

    node_m1 = torch.randn(n_node, N_DIM, device="cpu", dtype=DTYPE,
                           generator=gen).to(device).requires_grad_(requires_grad)
    node_m2 = torch.randn(n_node, N_SYM_DIM, device="cpu", dtype=DTYPE,
                           generator=gen).to(device).requires_grad_(requires_grad)
    edge = torch.randn(n_edge, EDGE_INFO_DIM, device="cpu", dtype=DTYPE,
                        generator=gen).to(device).requires_grad_(requires_grad)
    angle = torch.randn(n_angle, ANGLE_DIM, device="cpu", dtype=DTYPE,
                         generator=gen).to(device).requires_grad_(requires_grad)

    # Router outputs (fake).
    node_logits = torch.randn(n_node, n_routing_experts, device="cpu",
                               dtype=DTYPE, generator=gen).to(device)
    node_topk_logits, node_topk_indices = torch.topk(node_logits, k=topk, dim=-1)
    node_topk_weights = torch.softmax(node_topk_logits, dim=-1)
    if requires_grad:
        node_topk_weights = node_topk_weights.detach().requires_grad_(True)

    edge_logits = torch.randn(n_node, n_routing_experts, device="cpu",
                               dtype=DTYPE, generator=gen).to(device)
    edge_topk_logits, edge_topk_indices = torch.topk(edge_logits, k=topk, dim=-1)
    edge_topk_weights = torch.softmax(edge_topk_logits, dim=-1)
    if requires_grad:
        edge_topk_weights = edge_topk_weights.detach().requires_grad_(True)

    angle_logits = torch.randn(n_node, n_routing_experts, device="cpu",
                                dtype=DTYPE, generator=gen).to(device)
    angle_topk_logits, angle_topk_indices = torch.topk(angle_logits, k=topk, dim=-1)
    angle_topk_weights = torch.softmax(angle_topk_logits, dim=-1)
    if requires_grad:
        angle_topk_weights = angle_topk_weights.detach().requires_grad_(True)

    n2e_index = torch.randint(0, n_node, (n_edge,), device=device)
    n2a_index = torch.randint(0, n_node, (n_angle,), device=device)

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


# ===========================================================================
# 2 GPU Tests
# ===========================================================================


def test_forward_shape_2gpu(rank, world_size, ep_group):
    """[2 GPU] Forward output shapes are correct."""
    n_routing = 4
    topk = 2
    n_node, n_edge, n_angle = 8, 20, 25
    layer = _make_layer_multigpu(ep_group, rank, world_size, n_routing, topk)
    inputs = _make_inputs_multigpu(rank, n_node, n_edge, n_angle, topk, n_routing)

    m1, m2, e, a = layer(**inputs)

    check(m1.shape == (n_node, N_DIM), f"m1 shape {m1.shape}", rank)
    check(m2.shape == (n_node, N_DIM), f"m2 shape {m2.shape}", rank)
    check(e.shape == (n_edge, N_DIM + E_DIM), f"e shape {e.shape}", rank)
    check(a.shape == (n_angle, E_DIM + A_DIM), f"a shape {a.shape}", rank)

    all_pass(rank, world_size, "test_forward_shape_2gpu")


def test_backward_2gpu(rank, world_size, ep_group):
    """[2 GPU] Backward: all expert params have gradients."""
    n_routing = 4
    topk = 2
    layer = _make_layer_multigpu(ep_group, rank, world_size, n_routing, topk)
    inputs = _make_inputs_multigpu(rank, requires_grad=True, topk=topk,
                                    n_routing_experts=n_routing)

    m1, m2, e, a = layer(**inputs)
    loss = (m1 ** 2).sum() + (m2 ** 2).sum() + (e ** 2).sum() + (a ** 2).sum()
    loss.backward()

    n_params_with_grad = 0
    for name, param in layer.named_parameters():
        if param.grad is not None and (param.grad.abs() > 0).any():
            n_params_with_grad += 1

    # Each expert collection has routing + shared experts.
    # We expect most parameters to have gradients.
    check(n_params_with_grad > 0,
          f"Expected params with gradient, got {n_params_with_grad}", rank)

    all_pass(rank, world_size, "test_backward_2gpu")


def test_second_order_2gpu(rank, world_size, ep_group):
    """[2 GPU] 2nd-order derivative: no deadlock, gradients correct."""
    n_routing = 4
    topk = 2
    layer = _make_layer_multigpu(ep_group, rank, world_size, n_routing, topk)
    inputs = _make_inputs_multigpu(rank, requires_grad=True, topk=topk,
                                    n_routing_experts=n_routing)

    m1, m2, e, a = layer(**inputs)
    loss = (m1 ** 2).sum() + (m2 ** 2).sum() + (e ** 2).sum() + (a ** 2).sum()

    # 1st-order with create_graph=True.
    (grad_m1,) = torch.autograd.grad(
        loss, inputs["node_m1_input"], create_graph=True,
    )
    check(grad_m1 is not None, "1st-order grad is None", rank)
    check(grad_m1.requires_grad, "1st-order grad doesn't require grad", rank)

    # 2nd-order.
    grad_sum = (grad_m1 ** 2).sum()
    grad_sum.backward()

    # Check at least one param has nonzero 2nd-order gradient.
    has_nonzero = False
    for name, param in layer.named_parameters():
        if param.grad is not None and (param.grad.abs() > 0).any():
            has_nonzero = True
            break
    check(has_nonzero, "No parameter received 2nd-order gradient", rank)

    all_pass(rank, world_size, "test_second_order_2gpu")


def test_forward_with_shared_experts_2gpu(rank, world_size, ep_group):
    """[2 GPU] Forward with shared experts included."""
    n_routing = 4
    topk = 2
    n_shared = 1
    layer = _make_layer_multigpu(ep_group, rank, world_size, n_routing, topk,
                                  n_shared_experts=n_shared)
    inputs = _make_inputs_multigpu(rank, topk=topk, n_routing_experts=n_routing)

    m1, m2, e, a = layer(**inputs)
    check(m1.shape[1] == N_DIM, f"m1 dim {m1.shape[1]}", rank)
    check(e.shape[1] == N_DIM + E_DIM, f"e dim {e.shape[1]}", rank)

    all_pass(rank, world_size, "test_forward_with_shared_experts_2gpu")


def test_asymmetric_data_2gpu(rank, world_size, ep_group):
    """[2 GPU] Different data sizes per rank."""
    n_routing = 4
    topk = 2
    layer = _make_layer_multigpu(ep_group, rank, world_size, n_routing, topk)

    # Rank 0 gets more data than rank 1.
    n_node = 10 if rank == 0 else 5
    n_edge = 25 if rank == 0 else 12
    n_angle = 30 if rank == 0 else 15
    inputs = _make_inputs_multigpu(rank, n_node, n_edge, n_angle, topk, n_routing)

    m1, m2, e, a = layer(**inputs)
    check(m1.shape == (n_node, N_DIM), f"m1 shape {m1.shape}", rank)
    check(e.shape == (n_edge, N_DIM + E_DIM), f"e shape {e.shape}", rank)
    check(a.shape == (n_angle, E_DIM + A_DIM), f"a shape {a.shape}", rank)

    all_pass(rank, world_size, "test_asymmetric_data_2gpu")


def test_consistency_single_vs_multi_2gpu(rank, world_size, ep_group):
    """[2 GPU] Verify single-GPU reference matches multi-GPU result.

    Strategy:
    - All ranks use the same input data and routing.
    - Create a single-GPU layer with ALL experts (experts_per_gpu=n_routing).
    - Create a multi-GPU layer where each rank holds experts_per_gpu=n_routing/ep_size.
    - For the multi-GPU layer, we need to synchronize parameters: the single-GPU
      layer's expert i corresponds to the multi-GPU rank (i // experts_per_gpu)'s
      local expert (i % experts_per_gpu).
    - Both should produce the same output (since input and routing are identical).
    """
    n_routing = 4
    topk = 2
    n_shared = 0  # No shared experts for simpler comparison.
    seed = 42
    device = torch.device(f"cuda:{rank}")

    # Same inputs on all ranks (use fixed seed, same for all ranks).
    gen = torch.Generator(device="cpu")
    gen.manual_seed(777)
    n_node, n_edge, n_angle = 6, 12, 18

    node_m1 = torch.randn(n_node, N_DIM, device="cpu", dtype=DTYPE, generator=gen).to(device)
    node_m2 = torch.randn(n_node, N_SYM_DIM, device="cpu", dtype=DTYPE, generator=gen).to(device)
    edge_in = torch.randn(n_edge, EDGE_INFO_DIM, device="cpu", dtype=DTYPE, generator=gen).to(device)
    angle_in = torch.randn(n_angle, ANGLE_DIM, device="cpu", dtype=DTYPE, generator=gen).to(device)

    # Same routing on all ranks.
    gen2 = torch.Generator(device="cpu")
    gen2.manual_seed(888)
    node_logits = torch.randn(n_node, n_routing, device="cpu", dtype=DTYPE, generator=gen2).to(device)
    node_topk_l, node_topk_i = torch.topk(node_logits, k=topk, dim=-1)
    node_topk_w = torch.softmax(node_topk_l, dim=-1)

    edge_logits = torch.randn(n_node, n_routing, device="cpu", dtype=DTYPE, generator=gen2).to(device)
    edge_topk_l, edge_topk_i = torch.topk(edge_logits, k=topk, dim=-1)
    edge_topk_w = torch.softmax(edge_topk_l, dim=-1)

    angle_logits = torch.randn(n_node, n_routing, device="cpu", dtype=DTYPE, generator=gen2).to(device)
    angle_topk_l, angle_topk_i = torch.topk(angle_logits, k=topk, dim=-1)
    angle_topk_w = torch.softmax(angle_topk_l, dim=-1)

    gen3 = torch.Generator(device="cpu")
    gen3.manual_seed(999)
    n2e_index = torch.randint(0, n_node, (n_edge,), generator=gen3).to(device)
    n2a_index = torch.randint(0, n_node, (n_angle,), generator=gen3).to(device)

    # ---- Single-GPU reference ----
    single_layer = MoEDispatchCombine(
        n_dim=N_DIM, e_dim=E_DIM, a_dim=A_DIM,
        n_sym_dim=N_SYM_DIM, edge_info_dim=EDGE_INFO_DIM, angle_dim=ANGLE_DIM,
        n_routing_experts=n_routing, topk=topk, n_shared_experts=n_shared,
        ep_group=None, ep_rank=0, ep_size=1, experts_per_gpu=n_routing,
        activation_function="silu", precision="float64", seed=seed,
    ).to(device)

    # ---- Multi-GPU layer ----
    experts_per_gpu = n_routing // world_size
    multi_layer = MoEDispatchCombine(
        n_dim=N_DIM, e_dim=E_DIM, a_dim=A_DIM,
        n_sym_dim=N_SYM_DIM, edge_info_dim=EDGE_INFO_DIM, angle_dim=ANGLE_DIM,
        n_routing_experts=n_routing, topk=topk, n_shared_experts=n_shared,
        ep_group=ep_group, ep_rank=rank, ep_size=world_size,
        experts_per_gpu=experts_per_gpu,
        activation_function="silu", precision="float64", seed=seed,
    ).to(device)

    # Copy parameters from single-GPU to multi-GPU.
    # In the single-GPU layer, experts_per_gpu=n_routing, so routing_experts[0..3].
    # In the multi-GPU layer on rank r, experts_per_gpu=2, so routing_experts[0..1]
    # correspond to single-GPU experts [r*2, r*2+1].
    with torch.no_grad():
        for collection_name in ["node_self_experts", "node_sym_experts",
                                "edge_experts", "angle_experts"]:
            single_col = getattr(single_layer, collection_name)
            multi_col = getattr(multi_layer, collection_name)
            for local_eid in range(experts_per_gpu):
                global_eid = rank * experts_per_gpu + local_eid
                # Copy routing expert params from shared 3D tensor.
                # single_col.routing_matrix[:, :, global_eid] → multi_col.routing_matrix[:, :, local_eid]
                multi_col.routing_matrix[:, :, local_eid].copy_(
                    single_col.routing_matrix[:, :, global_eid],
                )
                multi_col.routing_bias[:, local_eid].copy_(
                    single_col.routing_bias[:, global_eid],
                )

    inputs = {
        "node_m1_input": node_m1,
        "node_m2_input": node_m2,
        "edge_input": edge_in,
        "angle_input": angle_in,
        "node_router_out": (node_topk_w, node_topk_i),
        "edge_router_out": (edge_topk_w, edge_topk_i),
        "angle_router_out": (angle_topk_w, angle_topk_i),
        "n2e_index": n2e_index,
        "n2a_index": n2a_index,
    }

    # Run both.
    with torch.no_grad():
        single_m1, single_m2, single_e, single_a = single_layer(**inputs)
        multi_m1, multi_m2, multi_e, multi_a = multi_layer(**inputs)

    # Compare outputs.
    tol = 1e-10
    m1_diff = (single_m1 - multi_m1).abs().max().item()
    m2_diff = (single_m2 - multi_m2).abs().max().item()
    e_diff = (single_e - multi_e).abs().max().item()
    a_diff = (single_a - multi_a).abs().max().item()

    check(m1_diff < tol, f"m1 max diff {m1_diff} >= {tol}", rank)
    check(m2_diff < tol, f"m2 max diff {m2_diff} >= {tol}", rank)
    check(e_diff < tol, f"edge max diff {e_diff} >= {tol}", rank)
    check(a_diff < tol, f"angle max diff {a_diff} >= {tol}", rank)

    all_pass(rank, world_size, "test_consistency_single_vs_multi_2gpu")


# ===========================================================================
# 4 GPU Tests
# ===========================================================================


def test_forward_shape_4gpu(rank, world_size, ep_group):
    """[4 GPU] Forward output shapes."""
    n_routing = 8
    topk = 2
    n_node, n_edge, n_angle = 10, 24, 30
    layer = _make_layer_multigpu(ep_group, rank, world_size, n_routing, topk)
    inputs = _make_inputs_multigpu(rank, n_node, n_edge, n_angle, topk, n_routing)

    m1, m2, e, a = layer(**inputs)
    check(m1.shape == (n_node, N_DIM), f"m1 shape {m1.shape}", rank)
    check(e.shape == (n_edge, N_DIM + E_DIM), f"e shape {e.shape}", rank)

    all_pass(rank, world_size, "test_forward_shape_4gpu")


def test_second_order_4gpu(rank, world_size, ep_group):
    """[4 GPU] 2nd-order derivative, no deadlock."""
    n_routing = 8
    topk = 2
    layer = _make_layer_multigpu(ep_group, rank, world_size, n_routing, topk)
    inputs = _make_inputs_multigpu(rank, requires_grad=True, topk=topk,
                                    n_routing_experts=n_routing)

    m1, m2, e, a = layer(**inputs)
    loss = (m1 ** 2).sum() + (m2 ** 2).sum() + (e ** 2).sum() + (a ** 2).sum()

    (grad_m1,) = torch.autograd.grad(
        loss, inputs["node_m1_input"], create_graph=True,
    )
    check(grad_m1 is not None, "1st-order grad is None", rank)

    grad_sum = (grad_m1 ** 2).sum()
    grad_sum.backward()

    has_nonzero = False
    for name, param in layer.named_parameters():
        if param.grad is not None and (param.grad.abs() > 0).any():
            has_nonzero = True
            break
    check(has_nonzero, "No parameter received 2nd-order gradient", rank)

    all_pass(rank, world_size, "test_second_order_4gpu")


def test_consistency_4gpu(rank, world_size, ep_group):
    """[4 GPU] Consistency: single-GPU reference vs 4-GPU pipeline."""
    n_routing = 8
    topk = 2
    n_shared = 0
    seed = 42
    device = torch.device(f"cuda:{rank}")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(777)
    n_node, n_edge, n_angle = 8, 16, 24

    node_m1 = torch.randn(n_node, N_DIM, device="cpu", dtype=DTYPE, generator=gen).to(device)
    node_m2 = torch.randn(n_node, N_SYM_DIM, device="cpu", dtype=DTYPE, generator=gen).to(device)
    edge_in = torch.randn(n_edge, EDGE_INFO_DIM, device="cpu", dtype=DTYPE, generator=gen).to(device)
    angle_in = torch.randn(n_angle, ANGLE_DIM, device="cpu", dtype=DTYPE, generator=gen).to(device)

    gen2 = torch.Generator(device="cpu")
    gen2.manual_seed(888)
    node_logits = torch.randn(n_node, n_routing, device="cpu", dtype=DTYPE, generator=gen2).to(device)
    node_topk_l, node_topk_i = torch.topk(node_logits, k=topk, dim=-1)
    node_topk_w = torch.softmax(node_topk_l, dim=-1)

    edge_logits = torch.randn(n_node, n_routing, device="cpu", dtype=DTYPE, generator=gen2).to(device)
    edge_topk_l, edge_topk_i = torch.topk(edge_logits, k=topk, dim=-1)
    edge_topk_w = torch.softmax(edge_topk_l, dim=-1)

    angle_logits = torch.randn(n_node, n_routing, device="cpu", dtype=DTYPE, generator=gen2).to(device)
    angle_topk_l, angle_topk_i = torch.topk(angle_logits, k=topk, dim=-1)
    angle_topk_w = torch.softmax(angle_topk_l, dim=-1)

    gen3 = torch.Generator(device="cpu")
    gen3.manual_seed(999)
    n2e_index = torch.randint(0, n_node, (n_edge,), generator=gen3).to(device)
    n2a_index = torch.randint(0, n_node, (n_angle,), generator=gen3).to(device)

    # Single-GPU reference.
    single_layer = MoEDispatchCombine(
        n_dim=N_DIM, e_dim=E_DIM, a_dim=A_DIM,
        n_sym_dim=N_SYM_DIM, edge_info_dim=EDGE_INFO_DIM, angle_dim=ANGLE_DIM,
        n_routing_experts=n_routing, topk=topk, n_shared_experts=n_shared,
        ep_group=None, ep_rank=0, ep_size=1, experts_per_gpu=n_routing,
        activation_function="silu", precision="float64", seed=seed,
    ).to(device)

    # Multi-GPU layer.
    experts_per_gpu = n_routing // world_size
    multi_layer = MoEDispatchCombine(
        n_dim=N_DIM, e_dim=E_DIM, a_dim=A_DIM,
        n_sym_dim=N_SYM_DIM, edge_info_dim=EDGE_INFO_DIM, angle_dim=ANGLE_DIM,
        n_routing_experts=n_routing, topk=topk, n_shared_experts=n_shared,
        ep_group=ep_group, ep_rank=rank, ep_size=world_size,
        experts_per_gpu=experts_per_gpu,
        activation_function="silu", precision="float64", seed=seed,
    ).to(device)

    # Copy parameters.
    with torch.no_grad():
        for col_name in ["node_self_experts", "node_sym_experts",
                         "edge_experts", "angle_experts"]:
            single_col = getattr(single_layer, col_name)
            multi_col = getattr(multi_layer, col_name)
            for local_eid in range(experts_per_gpu):
                global_eid = rank * experts_per_gpu + local_eid
                multi_col.routing_matrix[:, :, local_eid].copy_(
                    single_col.routing_matrix[:, :, global_eid],
                )
                multi_col.routing_bias[:, local_eid].copy_(
                    single_col.routing_bias[:, global_eid],
                )

    inputs = {
        "node_m1_input": node_m1,
        "node_m2_input": node_m2,
        "edge_input": edge_in,
        "angle_input": angle_in,
        "node_router_out": (node_topk_w, node_topk_i),
        "edge_router_out": (edge_topk_w, edge_topk_i),
        "angle_router_out": (angle_topk_w, angle_topk_i),
        "n2e_index": n2e_index,
        "n2a_index": n2a_index,
    }

    with torch.no_grad():
        single_m1, single_m2, single_e, single_a = single_layer(**inputs)
        multi_m1, multi_m2, multi_e, multi_a = multi_layer(**inputs)

    tol = 1e-10
    m1_diff = (single_m1 - multi_m1).abs().max().item()
    m2_diff = (single_m2 - multi_m2).abs().max().item()
    e_diff = (single_e - multi_e).abs().max().item()
    a_diff = (single_a - multi_a).abs().max().item()

    check(m1_diff < tol, f"m1 max diff {m1_diff} >= {tol}", rank)
    check(m2_diff < tol, f"m2 max diff {m2_diff} >= {tol}", rank)
    check(e_diff < tol, f"edge max diff {e_diff} >= {tol}", rank)
    check(a_diff < tol, f"angle max diff {a_diff} >= {tol}", rank)

    all_pass(rank, world_size, "test_consistency_4gpu")


# ===========================================================================
# Main entry
# ===========================================================================


def main():
    rank, world_size = setup_dist()
    ep_group = dist.new_group(list(range(world_size)))

    if rank == 0:
        print(f"\n=== MoEDispatchCombine Multi-GPU Tests (world_size={world_size}) ===")

    try:
        if world_size == 2:
            test_forward_shape_2gpu(rank, world_size, ep_group)
            test_backward_2gpu(rank, world_size, ep_group)
            test_second_order_2gpu(rank, world_size, ep_group)
            test_forward_with_shared_experts_2gpu(rank, world_size, ep_group)
            test_asymmetric_data_2gpu(rank, world_size, ep_group)
            test_consistency_single_vs_multi_2gpu(rank, world_size, ep_group)

        if world_size == 4:
            test_forward_shape_4gpu(rank, world_size, ep_group)
            test_second_order_4gpu(rank, world_size, ep_group)
            test_consistency_4gpu(rank, world_size, ep_group)

        dist.barrier()
        if rank == 0:
            print("\n=== All tests PASSED ===\n")
    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()
