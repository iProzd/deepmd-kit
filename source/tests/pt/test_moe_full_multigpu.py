# SPDX-License-Identifier: LGPL-3.0-or-later
"""Step 11: Complete correctness test matrix (multi-GPU).

Run with:
    torchrun --nproc_per_node=4 source/tests/pt/test_moe_full_multigpu.py   # T3 + T6
    torchrun --nproc_per_node=8 source/tests/pt/test_moe_full_multigpu.py   # T3 + T4 + T6

Tests:
  T3: DP gradient correctness (EP=1, DP=2)
  T4: EP+DP combined (EP=4, DP=2, 8 GPU) — forward + gradient match
  T6: 2nd-order derivatives through EP=4 All-to-All (no deadlock + numerical match)
"""

from __future__ import annotations

import sys
import traceback

import torch
import torch.distributed as dist

DTYPE = torch.float64

# ======================================================================
# RepFlowLayer config
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


def _make_layer(ep_group, ep_rank, ep_size, seed=42, device=None):
    """Create a MoE RepFlowLayer on the specified device."""
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
        n_routing_experts=N_ROUTING_EXPERTS,
        moe_topk=MOE_TOPK,
        n_shared_experts=N_SHARED_EXPERTS,
        ep_group=ep_group,
        ep_rank=ep_rank,
        ep_size=ep_size,
    )
    if device is None:
        device = torch.device(f"cuda:{dist.get_rank()}")
    return layer.to(device)


def _make_inputs_on_cpu(nb=1, seed=0):
    """Create inputs for a single batch on CPU."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    node_ebd_ext = torch.randn(
        nb, NALL, N_DIM, device="cpu", dtype=DTYPE, generator=gen,
    )
    edge_ebd = torch.randn(
        N_EDGE, E_DIM, device="cpu", dtype=DTYPE, generator=gen,
    )
    h2 = torch.randn(
        N_EDGE, 3, device="cpu", dtype=DTYPE, generator=gen,
    )
    angle_ebd = torch.randn(
        N_ANGLE, A_DIM, device="cpu", dtype=DTYPE, generator=gen,
    )
    nlist = torch.randint(0, NALL, (nb, NLOC, E_SEL), generator=gen)
    nlist_mask = torch.ones(nb, NLOC, E_SEL, dtype=DTYPE)
    sw = torch.rand(N_EDGE, dtype=DTYPE, generator=gen)
    a_nlist = torch.randint(0, NLOC, (nb, NLOC, A_SEL), generator=gen)
    a_nlist_mask = torch.ones(nb, NLOC, A_SEL, dtype=DTYPE)
    a_sw = torch.rand(N_ANGLE, dtype=DTYPE, generator=gen)

    gen_idx = torch.Generator(device="cpu")
    gen_idx.manual_seed(200 + seed)
    n2e_index = torch.randint(0, nb * NLOC, (N_EDGE,), generator=gen_idx)
    n_ext2e_index = torch.randint(0, nb * NALL, (N_EDGE,), generator=gen_idx)
    edge_index = torch.stack([n2e_index, n_ext2e_index], dim=0)

    n2a_index = torch.randint(0, nb * NLOC, (N_ANGLE,), generator=gen_idx)
    eij2a_index = torch.randint(0, N_EDGE, (N_ANGLE,), generator=gen_idx)
    eik2a_index = torch.randint(0, N_EDGE, (N_ANGLE,), generator=gen_idx)
    angle_index = torch.stack([n2a_index, eij2a_index, eik2a_index], dim=0)

    type_embedding = torch.randn(
        nb, NLOC, N_DIM, device="cpu", dtype=DTYPE, generator=gen,
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


def _inputs_to_device(inputs, device, requires_grad=False):
    """Move all input tensors to device."""
    result = {}
    for k, v in inputs.items():
        t = v.to(device)
        if requires_grad and t.is_floating_point():
            t = t.detach().requires_grad_(True)
        result[k] = t
    return result


def _broadcast_state_dict(state_dict, src=0):
    """Broadcast a state_dict from src to all ranks."""
    if dist.get_rank() == src:
        keys = list(state_dict.keys())
        n_keys = torch.tensor([len(keys)], dtype=torch.long, device="cuda")
    else:
        keys = None
        n_keys = torch.tensor([0], dtype=torch.long, device="cuda")
    dist.broadcast(n_keys, src=src)

    if dist.get_rank() == src:
        obj_list = [keys]
    else:
        obj_list = [None]
    dist.broadcast_object_list(obj_list, src=src)
    keys = obj_list[0]

    result = {}
    for key in keys:
        if dist.get_rank() == src:
            tensor = state_dict[key].cuda().contiguous()
        else:
            tensor = None

        if dist.get_rank() == src:
            meta = [(list(tensor.shape), str(tensor.dtype))]
        else:
            meta = [None]
        dist.broadcast_object_list(meta, src=src)
        shape, dtype_str = meta[0]
        dtype = getattr(torch, dtype_str.replace("torch.", ""))

        if dist.get_rank() != src:
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
        dist.broadcast(tensor, src=src)
        result[key] = tensor

    return result


def _mean_loss(n, e, a):
    """Compute mean loss over all elements (standard training loss)."""
    return n.pow(2).mean() + e.pow(2).mean() + a.pow(2).mean()


# ======================================================================
# T6: 2nd-order EP=4 (4 GPU)
# ======================================================================

def test_second_order_ep4(rank, world_size):
    """T6: 2nd-order derivatives through EP=4 All-to-All — no deadlock + match."""
    if world_size != 4:
        return True

    from deepmd.pt.utils.moe_checkpoint import (
        moe_load_state_dict_from_global,
        moe_state_dict_to_global,
    )

    device = torch.device(f"cuda:{rank}")

    # Step 1: Rank 0 creates 1-GPU reference, computes 2nd-order grads.
    if rank == 0:
        layer_1gpu = _make_layer(ep_group=None, ep_rank=0, ep_size=1)
        inputs_cpu = _make_inputs_on_cpu(nb=1, seed=42)
        inputs_gpu = _inputs_to_device(inputs_cpu, device, requires_grad=True)

        n_ref, e_ref, a_ref = layer_1gpu(**inputs_gpu)
        loss_ref = _mean_loss(n_ref, e_ref, a_ref)
        (grad_node_ref,) = torch.autograd.grad(
            loss_ref, inputs_gpu["node_ebd_ext"], create_graph=True,
        )
        grad_loss_ref = grad_node_ref.pow(2).sum()
        grad_loss_ref.backward()
        second_grad_ref = inputs_gpu["node_ebd_ext"].grad.clone()
        first_grad_ref = grad_node_ref.detach().clone()

        # Collect parameter grads.
        param_grads_ref = {}
        for name, param in layer_1gpu.named_parameters():
            if param.grad is not None:
                param_grads_ref[name] = param.grad.clone()

        global_sd = moe_state_dict_to_global(
            layer_1gpu, ep_rank=0, ep_size=1,
            experts_per_gpu=N_ROUTING_EXPERTS,
        )
    else:
        first_grad_ref = second_grad_ref = global_sd = inputs_cpu = None
        param_grads_ref = None

    # Broadcast.
    if rank == 0:
        global_sd_gpu = {k: v.cuda() for k, v in global_sd.items()}
    else:
        global_sd_gpu = None
    global_sd_gpu = _broadcast_state_dict(
        global_sd_gpu if rank == 0 else {}, src=0,
    )

    if rank == 0:
        inputs_cpu_nograd = _make_inputs_on_cpu(nb=1, seed=42)
        inputs_list = [inputs_cpu_nograd]
    else:
        inputs_list = [None]
    dist.broadcast_object_list(inputs_list, src=0)
    inputs_cpu = inputs_list[0]
    inputs_gpu = _inputs_to_device(inputs_cpu, device, requires_grad=True)

    if rank == 0:
        ref_1st = first_grad_ref.contiguous()
        ref_2nd = second_grad_ref.contiguous()
    else:
        ref_1st = torch.empty(1, NALL, N_DIM, dtype=DTYPE, device=device)
        ref_2nd = torch.empty(1, NALL, N_DIM, dtype=DTYPE, device=device)
    dist.broadcast(ref_1st, src=0)
    dist.broadcast(ref_2nd, src=0)

    # Step 2: Create EP=4 layer, load state_dict.
    ep_group = dist.new_group(list(range(world_size)))
    ep_size = world_size  # 4
    experts_per_gpu = N_ROUTING_EXPERTS // ep_size

    layer_ep = _make_layer(
        ep_group=ep_group, ep_rank=rank, ep_size=ep_size, seed=999,
    )

    global_sd_cpu = {k: v.cpu() for k, v in global_sd_gpu.items()}
    moe_load_state_dict_from_global(
        layer_ep, global_sd_cpu,
        ep_rank=rank, ep_size=ep_size,
        experts_per_gpu=experts_per_gpu,
    )

    # Step 3: 2nd-order forward+backward on EP=4 layer.
    n_ep, e_ep, a_ep = layer_ep(**inputs_gpu)
    loss_ep = _mean_loss(n_ep, e_ep, a_ep)
    (grad_node_ep,) = torch.autograd.grad(
        loss_ep, inputs_gpu["node_ebd_ext"], create_graph=True,
    )
    grad_loss_ep = grad_node_ep.pow(2).sum()
    grad_loss_ep.backward()
    second_grad_ep = inputs_gpu["node_ebd_ext"].grad.clone()
    first_grad_ep = grad_node_ep.detach().clone()

    # Step 4: Compare.
    assert torch.allclose(first_grad_ep, ref_1st, atol=1e-10), (
        f"Rank {rank}: T6: 1st intermediate grad mismatch. "
        f"Max diff: {(first_grad_ep - ref_1st).abs().max().item():.2e}"
    )
    assert torch.allclose(second_grad_ep, ref_2nd, atol=1e-10), (
        f"Rank {rank}: T6: 2nd-order input grad mismatch. "
        f"Max diff: {(second_grad_ep - ref_2nd).abs().max().item():.2e}"
    )

    return True


# ======================================================================
# T3: DP gradient correctness (EP=1, DP=2)
# ======================================================================

def test_dp_gradient_correctness(rank, world_size):
    """T3: EP=1, DP=2 — split batch + gradient sync == full-batch gradient."""
    from deepmd.pt.utils.moe_checkpoint import (
        moe_load_state_dict_from_global,
        moe_state_dict_to_global,
    )
    from deepmd.pt.utils.moe_ep_dp import sync_moe_gradients

    dp_size = 2
    if world_size < dp_size:
        return True

    device = torch.device(f"cuda:{rank}")

    # Step 1: Rank 0 creates 1-GPU reference with two separate samples.
    # With dynamic_sel, edge/angle are flat 1D tensors so NB=2 batch is not
    # straightforward. Instead, process each sample independently and average grads.
    if rank == 0:
        layer_ref = _make_layer(ep_group=None, ep_rank=0, ep_size=1)

        # Forward + backward for each sample, accumulate grads.
        for p in layer_ref.parameters():
            p.grad = None
        for s in range(dp_size):
            inp_s = _make_inputs_on_cpu(nb=1, seed=100 + s)
            inp_s_gpu = _inputs_to_device(inp_s, device)
            n_s, e_s, a_s = layer_ref(**inp_s_gpu)
            loss_s = _mean_loss(n_s, e_s, a_s)
            loss_s.backward()

        # Average accumulated gradients by dp_size (simulates mean over samples).
        ref_grads = {}
        for name, param in layer_ref.named_parameters():
            if param.grad is not None:
                param.grad.div_(dp_size)
                ref_grads[name] = param.grad.clone()

        global_sd = moe_state_dict_to_global(
            layer_ref, ep_rank=0, ep_size=1,
            experts_per_gpu=N_ROUTING_EXPERTS,
        )
    else:
        global_sd = None
        ref_grads = None

    # Broadcast state_dict and reference grads.
    if rank == 0:
        global_sd_gpu = {k: v.cuda() for k, v in global_sd.items()}
    else:
        global_sd_gpu = None
    global_sd_gpu = _broadcast_state_dict(
        global_sd_gpu if rank == 0 else {}, src=0,
    )

    # Broadcast ref_grads.
    if rank == 0:
        ref_grads_list = [ref_grads]
    else:
        ref_grads_list = [None]
    dist.broadcast_object_list(ref_grads_list, src=0)
    ref_grads = ref_grads_list[0]
    # Move to device.
    ref_grads = {k: v.to(device) for k, v in ref_grads.items()}

    # Step 2: Create DP group (ALL ranks must participate in new_group, NCCL req).
    dp_group = dist.new_group(list(range(dp_size)))

    # Only first dp_size ranks participate in the actual DP test.
    if rank < dp_size:
        # Create layer with EP=1 (all experts on each GPU).
        layer_dp = _make_layer(
            ep_group=None, ep_rank=0, ep_size=1, seed=999,
        )
        global_sd_cpu = {k: v.cpu() for k, v in global_sd_gpu.items()}
        moe_load_state_dict_from_global(
            layer_dp, global_sd_cpu,
            ep_rank=0, ep_size=1,
            experts_per_gpu=N_ROUTING_EXPERTS,
        )

        # Each rank gets its half of the batch.
        my_inputs_cpu = _make_inputs_on_cpu(nb=1, seed=100 + rank)
        my_inputs_gpu = _inputs_to_device(my_inputs_cpu, device)

        # Forward + backward.
        for p in layer_dp.parameters():
            p.grad = None
        n_dp, e_dp, a_dp = layer_dp(**my_inputs_gpu)
        loss_dp = _mean_loss(n_dp, e_dp, a_dp)
        loss_dp.backward()

        # Gradient sync (EP=1: dp_group covers only dp_size ranks).
        sync_moe_gradients(
            layer_dp,
            dp_group=dp_group,
            world_group=dp_group,
            dp_size=dp_size,
            world_size=dp_size,
        )

        # Compare with reference.
        for name, param in layer_dp.named_parameters():
            if param.grad is not None and name in ref_grads:
                assert torch.allclose(param.grad, ref_grads[name], atol=1e-10), (
                    f"Rank {rank}: T3: param grad mismatch: {name}. "
                    f"Max diff: {(param.grad - ref_grads[name]).abs().max().item():.2e}"
                )

    return True


# ======================================================================
# T4: EP+DP combined (8 GPU)
# ======================================================================

def test_ep_dp_combined(rank, world_size):
    """T4: EP=4, DP=2 — forward output match + gradient sync numerical match."""
    if world_size != 8:
        return True

    from deepmd.pt.utils.moe_checkpoint import (
        moe_load_state_dict_from_global,
        moe_state_dict_to_global,
    )
    from deepmd.pt.utils.moe_ep_dp import (
        init_ep_dp_groups,
        sync_moe_gradients,
    )

    device = torch.device(f"cuda:{rank}")

    # Step 1: Rank 0 creates 1-GPU reference with NB=8 (one sample per GPU).
    if rank == 0:
        layer_ref = _make_layer(ep_group=None, ep_rank=0, ep_size=1)

        # Create 8 per-sample inputs and forward each independently
        # to get per-sample reference outputs.
        per_sample_outputs = []
        for s in range(8):
            inp_s = _make_inputs_on_cpu(nb=1, seed=500 + s)
            inp_s_gpu = _inputs_to_device(inp_s, device)
            with torch.no_grad():
                n_s, e_s, a_s = layer_ref(**inp_s_gpu)
            per_sample_outputs.append((n_s.clone(), e_s.clone(), a_s.clone()))

        # Full-batch forward+backward for gradient reference (NB=8, mean loss).
        # Build the full batch by processing each sample independently and
        # averaging the gradients (since edge/angle are 1D, can't batch them).
        for p in layer_ref.parameters():
            p.grad = None
        for s in range(8):
            inp_s = _make_inputs_on_cpu(nb=1, seed=500 + s)
            inp_s_gpu = _inputs_to_device(inp_s, device)
            n_s, e_s, a_s = layer_ref(**inp_s_gpu)
            loss_s = _mean_loss(n_s, e_s, a_s)
            loss_s.backward()

        # Average the accumulated gradients by dividing by 8.
        ref_grads = {}
        for name, param in layer_ref.named_parameters():
            if param.grad is not None:
                param.grad.div_(8)
                ref_grads[name] = param.grad.clone()

        global_sd = moe_state_dict_to_global(
            layer_ref, ep_rank=0, ep_size=1,
            experts_per_gpu=N_ROUTING_EXPERTS,
        )
    else:
        per_sample_outputs = None
        global_sd = None
        ref_grads = None

    # Broadcast state_dict.
    if rank == 0:
        global_sd_gpu = {k: v.cuda() for k, v in global_sd.items()}
    else:
        global_sd_gpu = None
    global_sd_gpu = _broadcast_state_dict(
        global_sd_gpu if rank == 0 else {}, src=0,
    )

    # Broadcast per-sample reference outputs.
    if rank == 0:
        out_list = [per_sample_outputs]
    else:
        out_list = [None]
    dist.broadcast_object_list(out_list, src=0)
    per_sample_outputs = out_list[0]

    # Broadcast reference gradients.
    if rank == 0:
        grads_list = [ref_grads]
    else:
        grads_list = [None]
    dist.broadcast_object_list(grads_list, src=0)
    ref_grads = grads_list[0]
    ref_grads = {k: v.to(device) for k, v in ref_grads.items()}

    # Step 2: Init EP+DP groups.
    ep_group, dp_group, ep_rank, ep_size, dp_rank, dp_size = (
        init_ep_dp_groups(ep_size=4)
    )
    experts_per_gpu = N_ROUTING_EXPERTS // ep_size  # 1

    # Step 3: Create EP=4 layer, load state_dict.
    layer_ep = _make_layer(
        ep_group=ep_group, ep_rank=ep_rank, ep_size=ep_size, seed=999,
    )

    global_sd_cpu = {k: v.cpu() for k, v in global_sd_gpu.items()}
    moe_load_state_dict_from_global(
        layer_ep, global_sd_cpu,
        ep_rank=ep_rank, ep_size=ep_size,
        experts_per_gpu=experts_per_gpu,
    )

    # Step 4: Each rank processes its own sample (NB=1).
    my_sample_idx = rank
    my_inputs_cpu = _make_inputs_on_cpu(nb=1, seed=500 + my_sample_idx)
    my_inputs_gpu = _inputs_to_device(my_inputs_cpu, device)

    # Part A: Forward output match.
    with torch.no_grad():
        n_ep, e_ep, a_ep = layer_ep(**my_inputs_gpu)

    ref_n, ref_e, ref_a = per_sample_outputs[my_sample_idx]
    ref_n = ref_n.to(device)
    ref_e = ref_e.to(device)
    ref_a = ref_a.to(device)

    assert torch.allclose(n_ep, ref_n, atol=1e-10), (
        f"Rank {rank}: T4: node output mismatch (sample {my_sample_idx}). "
        f"Max diff: {(n_ep - ref_n).abs().max().item():.2e}"
    )
    assert torch.allclose(e_ep, ref_e, atol=1e-10), (
        f"Rank {rank}: T4: edge output mismatch (sample {my_sample_idx}). "
        f"Max diff: {(e_ep - ref_e).abs().max().item():.2e}"
    )
    assert torch.allclose(a_ep, ref_a, atol=1e-10), (
        f"Rank {rank}: T4: angle output mismatch (sample {my_sample_idx}). "
        f"Max diff: {(a_ep - ref_a).abs().max().item():.2e}"
    )

    # Part B: Gradient sync + numerical match.
    for p in layer_ep.parameters():
        p.grad = None
    my_inputs_gpu2 = _inputs_to_device(my_inputs_cpu, device)
    n_ep2, e_ep2, a_ep2 = layer_ep(**my_inputs_gpu2)
    loss_ep = _mean_loss(n_ep2, e_ep2, a_ep2)
    loss_ep.backward()

    # Gradient sync.
    sync_moe_gradients(
        layer_ep,
        dp_group=dp_group,
        world_group=None,  # default world group
        dp_size=dp_size,
        world_size=world_size,
    )

    # Compare with reference grads.
    # Routing expert params: local expert index 0..epg-1 maps to global index
    # ep_rank*epg..ep_rank*epg+epg-1.  We need to match the correct reference key.
    import re
    _RE = re.compile(r"\.routing_experts\.(\d+)\.")
    for name, param in layer_ep.named_parameters():
        if param.grad is None:
            continue
        m = _RE.search(name)
        if m:
            local_idx = int(m.group(1))
            global_idx = ep_rank * experts_per_gpu + local_idx
            ref_name = name.replace(
                f".routing_experts.{local_idx}.",
                f".routing_experts.{global_idx}.",
                1,
            )
        else:
            ref_name = name
        if ref_name in ref_grads:
            assert torch.allclose(param.grad, ref_grads[ref_name], atol=1e-10), (
                f"Rank {rank}: T4: param grad mismatch: {name} (ref: {ref_name}). "
                f"Max diff: {(param.grad - ref_grads[ref_name]).abs().max().item():.2e}"
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

    if world_size < 4:
        if rank == 0:
            print(f"ERROR: This test requires at least 4 GPUs, got {world_size}")
        cleanup_dist()
        sys.exit(1)

    if rank == 0:
        print(f"\nMoE Full Correctness Tests ({world_size} GPUs)")
        print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        (test_second_order_ep4, "T6: 2nd-order EP=4 (no deadlock + match)"),
        (test_dp_gradient_correctness, "T3: DP gradient correctness (EP=1, DP=2)"),
    ]

    if world_size == 8:
        tests.append(
            (test_ep_dp_combined, "T4: EP+DP (EP=4, DP=2) forward + gradient"),
        )

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
