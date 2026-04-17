# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-GPU unit tests for RepFlowLayer with MoE (Step 7).

Run with:
    torchrun --nproc_per_node=2 source/tests/pt/test_repflow_moe_multigpu.py
    torchrun --nproc_per_node=4 source/tests/pt/test_repflow_moe_multigpu.py
"""

from __future__ import annotations

import sys
import traceback

import torch
import torch.distributed as dist

DTYPE = torch.float64

# ======================================================================
# Config (matches single-GPU tests)
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

NB = 1
NLOC = 6
NALL = NLOC + 4
N_EDGE = 30
N_ANGLE = 50


# ======================================================================
# Helpers
# ======================================================================

def setup_dist():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def cleanup_dist():
    dist.destroy_process_group()


def _make_layer(
    n_routing_experts: int,
    topk: int,
    n_shared: int,
    ep_group,
    ep_rank: int,
    ep_size: int,
    seed: int = 42,
):
    from deepmd.pt.model.descriptor.repflow_layer import RepFlowLayer

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
        n_routing_experts=n_routing_experts,
        moe_topk=topk,
        n_shared_experts=n_shared,
        ep_group=ep_group,
        ep_rank=ep_rank,
        ep_size=ep_size,
    )
    device = torch.device(f"cuda:{ep_rank}")
    return layer.to(device)


def _make_inputs(rank: int, requires_grad: bool = False):
    """Create random inputs for the given rank (rank-specific seed)."""
    device = torch.device(f"cuda:{rank}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(100 + rank)

    node_ebd_ext = torch.randn(
        NB, NALL, N_DIM, device="cpu", dtype=DTYPE, generator=gen,
    ).to(device).requires_grad_(requires_grad)

    edge_ebd = torch.randn(
        N_EDGE, E_DIM, device="cpu", dtype=DTYPE, generator=gen,
    ).to(device).requires_grad_(requires_grad)

    h2 = torch.randn(
        N_EDGE, 3, device="cpu", dtype=DTYPE, generator=gen,
    ).to(device).requires_grad_(requires_grad)

    angle_ebd = torch.randn(
        N_ANGLE, A_DIM, device="cpu", dtype=DTYPE, generator=gen,
    ).to(device).requires_grad_(requires_grad)

    nlist = torch.randint(0, NALL, (NB, NLOC, E_SEL), generator=gen).to(device)
    nlist_mask = torch.ones(NB, NLOC, E_SEL, device=device, dtype=DTYPE)

    sw = torch.rand(N_EDGE, device="cpu", dtype=DTYPE, generator=gen).to(device)
    if requires_grad:
        sw = sw.detach().requires_grad_(True)

    a_nlist = torch.randint(0, NLOC, (NB, NLOC, A_SEL), generator=gen).to(device)
    a_nlist_mask = torch.ones(NB, NLOC, A_SEL, device=device, dtype=DTYPE)
    a_sw = torch.rand(N_ANGLE, device="cpu", dtype=DTYPE, generator=gen).to(device)
    if requires_grad:
        a_sw = a_sw.detach().requires_grad_(True)

    gen_idx = torch.Generator(device="cpu")
    gen_idx.manual_seed(200 + rank)
    n2e_index = torch.randint(0, NB * NLOC, (N_EDGE,), generator=gen_idx).to(device)
    n_ext2e_index = torch.randint(0, NB * NALL, (N_EDGE,), generator=gen_idx).to(device)
    edge_index = torch.stack([n2e_index, n_ext2e_index], dim=0)

    n2a_index = torch.randint(0, NB * NLOC, (N_ANGLE,), generator=gen_idx).to(device)
    eij2a_index = torch.randint(0, N_EDGE, (N_ANGLE,), generator=gen_idx).to(device)
    eik2a_index = torch.randint(0, N_EDGE, (N_ANGLE,), generator=gen_idx).to(device)
    angle_index = torch.stack([n2a_index, eij2a_index, eik2a_index], dim=0)

    type_embedding = torch.randn(
        NB, NLOC, N_DIM, device="cpu", dtype=DTYPE, generator=gen,
    ).to(device).requires_grad_(requires_grad)

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


def _clear_grads(layer, inputs):
    layer.zero_grad()
    for key in ["node_ebd_ext", "edge_ebd", "h2", "angle_ebd", "type_embedding"]:
        if inputs[key].grad is not None:
            inputs[key].grad = None


# ======================================================================
# Tests
# ======================================================================

def test_forward_shape(rank, world_size, ep_group):
    """Forward produces correct output shapes."""
    n_routing = world_size * 2
    layer = _make_layer(n_routing, topk=2, n_shared=0,
                        ep_group=ep_group, ep_rank=rank, ep_size=world_size)
    inputs = _make_inputs(rank)
    n_out, e_out, a_out = layer(**inputs)

    assert n_out.shape == (NB, NLOC, N_DIM), f"node shape {n_out.shape}"
    assert e_out.shape == (N_EDGE, E_DIM), f"edge shape {e_out.shape}"
    assert a_out.shape == (N_ANGLE, A_DIM), f"angle shape {a_out.shape}"
    return True


def test_backward(rank, world_size, ep_group):
    """Backward completes without hang."""
    n_routing = world_size * 2
    layer = _make_layer(n_routing, topk=2, n_shared=0,
                        ep_group=ep_group, ep_rank=rank, ep_size=world_size)
    inputs = _make_inputs(rank, requires_grad=True)
    n_out, e_out, a_out = layer(**inputs)

    loss = (n_out ** 2).sum() + (e_out ** 2).sum() + (a_out ** 2).sum()
    loss.backward()

    # Verify at least some MoE params have gradients.
    has_grad = False
    for name, param in layer.named_parameters():
        if "moe_phase1" in name and param.grad is not None:
            if (param.grad.abs() > 0).any():
                has_grad = True
                break
    assert has_grad, "No moe_phase1 parameter received gradient"
    return True


def test_second_order(rank, world_size, ep_group):
    """Second-order derivative through MoE pipeline."""
    n_routing = world_size * 2
    layer = _make_layer(n_routing, topk=2, n_shared=0,
                        ep_group=ep_group, ep_rank=rank, ep_size=world_size)
    inputs = _make_inputs(rank, requires_grad=True)
    n_out, e_out, a_out = layer(**inputs)

    loss = (n_out ** 2).sum() + (e_out ** 2).sum() + (a_out ** 2).sum()

    (grad_node,) = torch.autograd.grad(
        loss, inputs["node_ebd_ext"], create_graph=True,
    )
    grad_sum = (grad_node ** 2).sum()
    grad_sum.backward()

    _clear_grads(layer, inputs)
    return True


def test_shared_experts(rank, world_size, ep_group):
    """Forward with shared experts produces correct shapes."""
    n_routing = world_size * 2
    n_shared = 1
    layer = _make_layer(n_routing, topk=2, n_shared=n_shared,
                        ep_group=ep_group, ep_rank=rank, ep_size=world_size)
    inputs = _make_inputs(rank)
    n_out, e_out, a_out = layer(**inputs)

    assert n_out.shape == (NB, NLOC, N_DIM)
    assert e_out.shape == (N_EDGE, E_DIM)
    assert a_out.shape == (N_ANGLE, A_DIM)
    return True


# ======================================================================
# Runner
# ======================================================================

def run_test(test_fn, name, rank, world_size, ep_group):
    """Run a test function with error handling and barrier sync."""
    try:
        result = test_fn(rank, world_size, ep_group)
        dist.barrier()
        if rank == 0:
            print(f"  PASS: {name}")
        return True
    except Exception as ex:
        print(f"  FAIL: {name} (rank {rank}): {ex}")
        traceback.print_exc()
        dist.barrier()
        return False


def main():
    rank, world_size = setup_dist()

    ep_group = dist.new_group(list(range(world_size)))

    if rank == 0:
        print(f"\nRepFlowLayer MoE Multi-GPU Tests ({world_size} GPUs)")
        print("=" * 60)

    passed = 0
    failed = 0

    # All world_size tests.
    tests = [
        (test_forward_shape, "forward shape"),
        (test_backward, "backward"),
        (test_second_order, "second order"),
        (test_shared_experts, "shared experts"),
    ]

    for test_fn, name in tests:
        if run_test(test_fn, name, rank, world_size, ep_group):
            passed += 1
        else:
            failed += 1
        torch.cuda.empty_cache()

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
        print(f"{'=' * 60}\n")

    cleanup_dist()
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
