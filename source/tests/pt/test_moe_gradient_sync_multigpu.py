# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-GPU unit tests for MoE EP+DP process groups and gradient sync (Step 8).

Requires exactly 4 GPUs (EP=2, DP=2).

Run with:
    torchrun --nproc_per_node=4 source/tests/pt/test_moe_gradient_sync_multigpu.py

GPU layout (ep_size=2, dp_size=2):

         EP rank 0  EP rank 1
DP rank 0:  GPU 0     GPU 1    <- ep_group_0
DP rank 1:  GPU 2     GPU 3    <- ep_group_1
            ^ dp_group_0  ^ dp_group_1
"""

from __future__ import annotations

import sys
import traceback

import torch
import torch.distributed as dist

DTYPE = torch.float64

# ======================================================================
# RepFlowLayer config (matches Step 7 tests)
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


def _make_layer(ep_group, ep_rank, ep_size, seed=42):
    """Create a MoE RepFlowLayer with n_routing_experts=ep_size*2, topk=2."""
    from deepmd.pt.model.descriptor.repflow_layer import RepFlowLayer

    n_routing = ep_size * 2  # 2 routing experts per GPU
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
        n_routing_experts=n_routing,
        moe_topk=2,
        n_shared_experts=1,
        ep_group=ep_group,
        ep_rank=ep_rank,
        ep_size=ep_size,
    )
    device = torch.device(f"cuda:{dist.get_rank()}")
    return layer.to(device)


def _make_inputs(rank, requires_grad=False):
    """Create rank-specific random inputs on the correct GPU."""
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


def _gather_tensor(t: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    """All-gather a tensor from all ranks, return list of tensors."""
    gathered = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    return gathered


# ======================================================================
# Tests
# ======================================================================

def test_group_creation(rank, world_size):
    """Verify EP/DP group creation and basic communication."""
    from deepmd.pt.utils.moe_ep_dp import init_ep_dp_groups

    ep_group, dp_group, ep_rank, ep_sz, dp_rank, dp_sz = init_ep_dp_groups(
        ep_size=2,
    )

    # Verify dimensions.
    assert ep_sz == 2, f"ep_size={ep_sz}"
    assert dp_sz == 2, f"dp_size={dp_sz}"

    # Verify rank mapping: world_rank = dp_rank * ep_size + ep_rank.
    expected_ep_rank = rank % 2
    expected_dp_rank = rank // 2
    assert ep_rank == expected_ep_rank, (
        f"rank {rank}: ep_rank={ep_rank}, expected={expected_ep_rank}"
    )
    assert dp_rank == expected_dp_rank, (
        f"rank {rank}: dp_rank={dp_rank}, expected={expected_dp_rank}"
    )

    # Test EP group communication: sum across EP group should give
    # sum of ep ranks in that group.
    device = torch.device(f"cuda:{rank}")
    t_ep = torch.tensor([float(ep_rank)], device=device, dtype=DTYPE)
    dist.all_reduce(t_ep, op=dist.ReduceOp.SUM, group=ep_group)
    # EP group has ep_rank 0 and 1, so sum = 0 + 1 = 1.
    assert t_ep.item() == 1.0, f"EP all-reduce: {t_ep.item()}"

    # Test DP group communication: sum across DP group should give
    # sum of dp ranks in that group.
    t_dp = torch.tensor([float(dp_rank)], device=device, dtype=DTYPE)
    dist.all_reduce(t_dp, op=dist.ReduceOp.SUM, group=dp_group)
    # DP group has dp_rank 0 and 1, so sum = 0 + 1 = 1.
    assert t_dp.item() == 1.0, f"DP all-reduce: {t_dp.item()}"

    return True


def test_group_creation_trivial(rank, world_size):
    """ep_size=1 returns None groups."""
    from deepmd.pt.utils.moe_ep_dp import init_ep_dp_groups

    ep_group, dp_group, ep_rank, ep_sz, dp_rank, dp_sz = init_ep_dp_groups(
        ep_size=1,
    )
    assert ep_group is None
    assert dp_group is None
    assert ep_rank == 0
    assert ep_sz == 1
    assert dp_rank == rank
    assert dp_sz == world_size
    return True


def test_routing_expert_grad_sync(rank, world_size):
    """Routing expert grads: identical within dp_group, different across ep_groups."""
    from deepmd.pt.utils.moe_ep_dp import (
        _is_routing_expert_param,
        init_ep_dp_groups,
        sync_moe_gradients,
    )

    ep_group, dp_group, ep_rank, ep_sz, dp_rank, dp_sz = init_ep_dp_groups(
        ep_size=2,
    )

    # Create layer and compute gradients with rank-specific data.
    layer = _make_layer(ep_group, ep_rank, ep_sz)
    inputs = _make_inputs(rank, requires_grad=True)

    n_out, e_out, a_out = layer(**inputs)
    loss = (n_out ** 2).sum() + (e_out ** 2).sum() + (a_out ** 2).sum()
    loss.backward()

    # Sync gradients.
    sync_moe_gradients(layer, dp_group, None, dp_sz, world_size)

    # Gather routing expert grads from all ranks and compare.
    for name, param in layer.named_parameters():
        if not _is_routing_expert_param(name):
            continue
        if param.grad is None:
            continue

        grads = _gather_tensor(param.grad, world_size)

        # Same dp_group (same ep_rank) should be identical.
        # GPU 0 (ep=0,dp=0) and GPU 2 (ep=0,dp=1) share dp_group_0.
        # GPU 1 (ep=1,dp=0) and GPU 3 (ep=1,dp=1) share dp_group_1.
        if rank == 0:
            # Check dp_group_0: GPU 0 == GPU 2.
            assert torch.equal(grads[0], grads[2]), (
                f"Routing expert {name}: GPU0 != GPU2 in dp_group_0"
            )
            # Check dp_group_1: GPU 1 == GPU 3.
            assert torch.equal(grads[1], grads[3]), (
                f"Routing expert {name}: GPU1 != GPU3 in dp_group_1"
            )

    return True


def test_non_routing_grad_sync(rank, world_size):
    """Non-routing param grads should be identical across all 4 GPUs."""
    from deepmd.pt.utils.moe_ep_dp import (
        _is_routing_expert_param,
        init_ep_dp_groups,
        sync_moe_gradients,
    )

    ep_group, dp_group, ep_rank, ep_sz, dp_rank, dp_sz = init_ep_dp_groups(
        ep_size=2,
    )

    layer = _make_layer(ep_group, ep_rank, ep_sz)
    inputs = _make_inputs(rank, requires_grad=True)

    n_out, e_out, a_out = layer(**inputs)
    loss = (n_out ** 2).sum() + (e_out ** 2).sum() + (a_out ** 2).sum()
    loss.backward()

    sync_moe_gradients(layer, dp_group, None, dp_sz, world_size)

    # Check non-routing params are identical across all ranks.
    for name, param in layer.named_parameters():
        if _is_routing_expert_param(name):
            continue
        if param.grad is None:
            continue

        grads = _gather_tensor(param.grad, world_size)
        if rank == 0:
            for i in range(1, world_size):
                assert torch.equal(grads[0], grads[i]), (
                    f"Non-routing {name}: GPU0 != GPU{i}"
                )

    return True


def test_div_by_correct_group_size(rank, world_size):
    """Verify gradients are divided by the correct group size."""
    from deepmd.pt.utils.moe_ep_dp import (
        _is_routing_expert_param,
        init_ep_dp_groups,
        sync_moe_gradients,
    )

    ep_group, dp_group, ep_rank, ep_sz, dp_rank, dp_sz = init_ep_dp_groups(
        ep_size=2,
    )

    layer = _make_layer(ep_group, ep_rank, ep_sz)

    # Set all gradients to a known value: rank + 1.
    for param in layer.parameters():
        param.grad = torch.full_like(param, float(rank + 1))

    # Save pre-sync grads per rank via all-gather.
    pre_grads = {}
    for name, param in layer.named_parameters():
        pre_grads[name] = _gather_tensor(param.grad.clone(), world_size)

    sync_moe_gradients(layer, dp_group, None, dp_sz, world_size)

    if rank == 0:
        for name, param in layer.named_parameters():
            if param.grad is None:
                continue

            post = param.grad
            pre = pre_grads[name]  # list of tensors from all ranks

            if _is_routing_expert_param(name):
                # Routing expert: sum over dp_group / dp_size.
                # GPU 0 is in dp_group_0 with GPU 2.
                # Pre: GPU0=1.0, GPU2=3.0. Sum=4.0. Avg=4.0/2=2.0.
                expected = (pre[0] + pre[2]) / dp_sz
                assert torch.allclose(post, expected, atol=1e-12), (
                    f"Routing {name}: expected {expected.flatten()[:3]}, "
                    f"got {post.flatten()[:3]}"
                )
            else:
                # Non-routing: sum over all / world_size.
                # Pre: GPU0=1, GPU1=2, GPU2=3, GPU3=4. Sum=10. Avg=10/4=2.5.
                expected = sum(pre) / world_size
                assert torch.allclose(post, expected, atol=1e-12), (
                    f"Non-routing {name}: expected {expected.flatten()[:3]}, "
                    f"got {post.flatten()[:3]}"
                )

    return True


def test_second_order_routing_grad_sync(rank, world_size):
    """Routing expert grads under 2nd-order: identical within dp_group."""
    from deepmd.pt.utils.moe_ep_dp import (
        _is_routing_expert_param,
        init_ep_dp_groups,
        sync_moe_gradients,
    )

    ep_group, dp_group, ep_rank, ep_sz, dp_rank, dp_sz = init_ep_dp_groups(
        ep_size=2,
    )

    layer = _make_layer(ep_group, ep_rank, ep_sz)
    inputs = _make_inputs(rank, requires_grad=True)

    # 2nd-order: forward -> 1st grad with create_graph -> 2nd backward.
    n_out, e_out, a_out = layer(**inputs)
    loss = (n_out ** 2).sum() + (e_out ** 2).sum() + (a_out ** 2).sum()

    (grad_node,) = torch.autograd.grad(
        loss, inputs["node_ebd_ext"], create_graph=True,
    )
    grad_loss = (grad_node ** 2).sum()
    grad_loss.backward()

    # Sync gradients.
    sync_moe_gradients(layer, dp_group, None, dp_sz, world_size)

    # Routing expert grads: identical within dp_group.
    for name, param in layer.named_parameters():
        if not _is_routing_expert_param(name):
            continue
        if param.grad is None:
            continue

        grads = _gather_tensor(param.grad, world_size)
        if rank == 0:
            # dp_group_0: GPU 0 == GPU 2.
            assert torch.equal(grads[0], grads[2]), (
                f"2nd-order routing {name}: GPU0 != GPU2"
            )
            # dp_group_1: GPU 1 == GPU 3.
            assert torch.equal(grads[1], grads[3]), (
                f"2nd-order routing {name}: GPU1 != GPU3"
            )

    return True


def test_second_order_non_routing_grad_sync(rank, world_size):
    """Non-routing grads under 2nd-order: identical across all GPUs."""
    from deepmd.pt.utils.moe_ep_dp import (
        _is_routing_expert_param,
        init_ep_dp_groups,
        sync_moe_gradients,
    )

    ep_group, dp_group, ep_rank, ep_sz, dp_rank, dp_sz = init_ep_dp_groups(
        ep_size=2,
    )

    layer = _make_layer(ep_group, ep_rank, ep_sz)
    inputs = _make_inputs(rank, requires_grad=True)

    # 2nd-order computation.
    n_out, e_out, a_out = layer(**inputs)
    loss = (n_out ** 2).sum() + (e_out ** 2).sum() + (a_out ** 2).sum()

    (grad_node,) = torch.autograd.grad(
        loss, inputs["node_ebd_ext"], create_graph=True,
    )
    grad_loss = (grad_node ** 2).sum()
    grad_loss.backward()

    sync_moe_gradients(layer, dp_group, None, dp_sz, world_size)

    # Non-routing grads: identical across all 4 GPUs.
    for name, param in layer.named_parameters():
        if _is_routing_expert_param(name):
            continue
        if param.grad is None:
            continue

        grads = _gather_tensor(param.grad, world_size)
        if rank == 0:
            for i in range(1, world_size):
                assert torch.equal(grads[0], grads[i]), (
                    f"2nd-order non-routing {name}: GPU0 != GPU{i}"
                )

    return True


def test_second_order_div_by_correct_group_size(rank, world_size):
    """2nd-order gradients are divided by the correct group size."""
    from deepmd.pt.utils.moe_ep_dp import (
        _is_routing_expert_param,
        init_ep_dp_groups,
        sync_moe_gradients,
    )

    ep_group, dp_group, ep_rank, ep_sz, dp_rank, dp_sz = init_ep_dp_groups(
        ep_size=2,
    )

    layer = _make_layer(ep_group, ep_rank, ep_sz)
    inputs = _make_inputs(rank, requires_grad=True)

    # 2nd-order computation.
    n_out, e_out, a_out = layer(**inputs)
    loss = (n_out ** 2).sum() + (e_out ** 2).sum() + (a_out ** 2).sum()

    (grad_node,) = torch.autograd.grad(
        loss, inputs["node_ebd_ext"], create_graph=True,
    )
    grad_loss = (grad_node ** 2).sum()
    grad_loss.backward()

    # Save pre-sync grads from all ranks.
    pre_grads = {}
    for name, param in layer.named_parameters():
        if param.grad is not None:
            pre_grads[name] = _gather_tensor(param.grad.clone(), world_size)

    sync_moe_gradients(layer, dp_group, None, dp_sz, world_size)

    if rank == 0:
        for name, param in layer.named_parameters():
            if param.grad is None or name not in pre_grads:
                continue

            post = param.grad
            pre = pre_grads[name]

            if _is_routing_expert_param(name):
                # Routing: avg over dp_group (GPU 0 + GPU 2) / 2.
                expected = (pre[0] + pre[2]) / dp_sz
                assert torch.allclose(post, expected, atol=1e-12), (
                    f"2nd-order routing {name}: division mismatch"
                )
            else:
                # Non-routing: avg over world (all 4 GPUs) / 4.
                expected = sum(pre) / world_size
                assert torch.allclose(post, expected, atol=1e-12), (
                    f"2nd-order non-routing {name}: division mismatch"
                )

    return True


# ======================================================================
# Runner
# ======================================================================

def run_test(test_fn, name, rank, world_size):
    """Run a test function with error handling and barrier sync."""
    try:
        result = test_fn(rank, world_size)
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

    if world_size != 4:
        if rank == 0:
            print(f"ERROR: This test requires exactly 4 GPUs, got {world_size}")
        cleanup_dist()
        sys.exit(1)

    if rank == 0:
        print(f"\nMoE Gradient Sync Multi-GPU Tests ({world_size} GPUs, EP=2, DP=2)")
        print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        (test_group_creation, "group creation (EP=2, DP=2)"),
        (test_group_creation_trivial, "group creation trivial (ep_size=1)"),
        (test_routing_expert_grad_sync, "routing expert grad sync"),
        (test_non_routing_grad_sync, "non-routing grad sync"),
        (test_div_by_correct_group_size, "division by correct group size"),
        (test_second_order_routing_grad_sync, "2nd-order routing grad sync"),
        (test_second_order_non_routing_grad_sync, "2nd-order non-routing grad sync"),
        (test_second_order_div_by_correct_group_size, "2nd-order division correctness"),
    ]

    for test_fn, name in tests:
        if run_test(test_fn, name, rank, world_size):
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
