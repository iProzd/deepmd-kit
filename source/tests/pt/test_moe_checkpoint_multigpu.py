# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-GPU unit tests for MoE checkpoint save/load with resharding (Step 10).

Requires exactly 4 GPUs.

Run with:
    torchrun --nproc_per_node=4 source/tests/pt/test_moe_checkpoint_multigpu.py

Core test (T2): 1 GPU save → 4 GPU (EP=4) load → same input → output matches.
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
    """Create a MoE RepFlowLayer on the correct GPU."""
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
    device = torch.device(f"cuda:{dist.get_rank()}")
    return layer.to(device)


def _make_inputs_on_cpu(seed=0, requires_grad=False):
    """Create inputs on CPU (will be broadcast to all ranks)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    node_ebd_ext = torch.randn(
        NB, NALL, N_DIM, device="cpu", dtype=DTYPE, generator=gen,
    )
    if requires_grad:
        node_ebd_ext = node_ebd_ext.detach().requires_grad_(True)
    edge_ebd = torch.randn(
        N_EDGE, E_DIM, device="cpu", dtype=DTYPE, generator=gen,
    )
    if requires_grad:
        edge_ebd = edge_ebd.detach().requires_grad_(True)
    h2 = torch.randn(
        N_EDGE, 3, device="cpu", dtype=DTYPE, generator=gen,
    )
    if requires_grad:
        h2 = h2.detach().requires_grad_(True)
    angle_ebd = torch.randn(
        N_ANGLE, A_DIM, device="cpu", dtype=DTYPE, generator=gen,
    )
    if requires_grad:
        angle_ebd = angle_ebd.detach().requires_grad_(True)
    nlist = torch.randint(0, NALL, (NB, NLOC, E_SEL), generator=gen)
    nlist_mask = torch.ones(NB, NLOC, E_SEL, dtype=DTYPE)
    sw = torch.rand(N_EDGE, dtype=DTYPE, generator=gen)
    a_nlist = torch.randint(0, NLOC, (NB, NLOC, A_SEL), generator=gen)
    a_nlist_mask = torch.ones(NB, NLOC, A_SEL, dtype=DTYPE)
    a_sw = torch.rand(N_ANGLE, dtype=DTYPE, generator=gen)

    gen_idx = torch.Generator(device="cpu")
    gen_idx.manual_seed(200)
    n2e_index = torch.randint(0, NB * NLOC, (N_EDGE,), generator=gen_idx)
    n_ext2e_index = torch.randint(0, NB * NALL, (N_EDGE,), generator=gen_idx)
    edge_index = torch.stack([n2e_index, n_ext2e_index], dim=0)

    n2a_index = torch.randint(0, NB * NLOC, (N_ANGLE,), generator=gen_idx)
    eij2a_index = torch.randint(0, N_EDGE, (N_ANGLE,), generator=gen_idx)
    eik2a_index = torch.randint(0, N_EDGE, (N_ANGLE,), generator=gen_idx)
    angle_index = torch.stack([n2a_index, eij2a_index, eik2a_index], dim=0)

    type_embedding = torch.randn(
        NB, NLOC, N_DIM, device="cpu", dtype=DTYPE, generator=gen,
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
    """Move all input tensors to device, optionally enabling requires_grad."""
    result = {}
    for k, v in inputs.items():
        t = v.to(device)
        if requires_grad and t.is_floating_point():
            t = t.detach().requires_grad_(True)
        result[k] = t
    return result


def _broadcast_state_dict(state_dict, src=0):
    """Broadcast a state_dict from src to all ranks."""
    # Broadcast number of keys.
    if dist.get_rank() == src:
        keys = list(state_dict.keys())
        n_keys = torch.tensor([len(keys)], dtype=torch.long, device="cuda")
    else:
        keys = None
        n_keys = torch.tensor([0], dtype=torch.long, device="cuda")
    dist.broadcast(n_keys, src=src)

    # Broadcast keys via object list.
    if dist.get_rank() == src:
        obj_list = [keys]
    else:
        obj_list = [None]
    dist.broadcast_object_list(obj_list, src=src)
    keys = obj_list[0]

    # Broadcast tensors.
    result = {}
    for key in keys:
        if dist.get_rank() == src:
            tensor = state_dict[key].cuda().contiguous()
        else:
            # Need shape and dtype from src. Use broadcast_object_list.
            tensor = None

        # Broadcast metadata (shape, dtype).
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


# ======================================================================
# Tests
# ======================================================================

def test_1gpu_save_4gpu_load(rank, world_size):
    """T2: 1 GPU save → 4 GPU (EP=4) load → same input → output matches."""
    from deepmd.pt.utils.moe_checkpoint import (
        moe_load_state_dict_from_global,
        moe_state_dict_to_global,
    )

    device = torch.device(f"cuda:{rank}")

    # Step 1: Rank 0 creates a single-GPU layer and computes reference output.
    if rank == 0:
        layer_1gpu = _make_layer(ep_group=None, ep_rank=0, ep_size=1)
        inputs_cpu = _make_inputs_on_cpu(seed=42)
        inputs_gpu = _inputs_to_device(inputs_cpu, device)

        with torch.no_grad():
            n_ref, e_ref, a_ref = layer_1gpu(**inputs_gpu)

        # Save global state_dict.
        global_sd = moe_state_dict_to_global(
            layer_1gpu, ep_rank=0, ep_size=1,
            experts_per_gpu=N_ROUTING_EXPERTS,
        )
    else:
        n_ref = None
        e_ref = None
        a_ref = None
        global_sd = None
        inputs_cpu = None

    # Step 2: Broadcast global state_dict and inputs to all ranks.
    if rank == 0:
        global_sd_gpu = {k: v.cuda() for k, v in global_sd.items()}
    else:
        global_sd_gpu = None
    global_sd_gpu = _broadcast_state_dict(
        global_sd_gpu if rank == 0 else {}, src=0,
    )

    # Broadcast inputs.
    if rank == 0:
        inputs_list = [inputs_cpu]
    else:
        inputs_list = [None]
    dist.broadcast_object_list(inputs_list, src=0)
    inputs_cpu = inputs_list[0]
    inputs_gpu = _inputs_to_device(inputs_cpu, device)

    # Broadcast reference outputs.
    if rank == 0:
        ref_n = n_ref.contiguous()
        ref_e = e_ref.contiguous()
        ref_a = a_ref.contiguous()
    else:
        ref_n = torch.empty(NB, NLOC, N_DIM, dtype=DTYPE, device=device)
        ref_e = torch.empty(N_EDGE, E_DIM, dtype=DTYPE, device=device)
        ref_a = torch.empty(N_ANGLE, A_DIM, dtype=DTYPE, device=device)
    dist.broadcast(ref_n, src=0)
    dist.broadcast(ref_e, src=0)
    dist.broadcast(ref_a, src=0)

    # Step 3: Create EP=4 layer on each rank.
    # Build ep_group for all 4 ranks.
    ep_group = dist.new_group(list(range(world_size)))
    ep_rank_local = rank
    ep_size = world_size  # 4
    experts_per_gpu = N_ROUTING_EXPERTS // ep_size  # 1

    layer_ep = _make_layer(
        ep_group=ep_group, ep_rank=ep_rank_local, ep_size=ep_size, seed=999,
    )

    # Step 4: Load global state_dict.
    # Move global_sd to CPU for load.
    global_sd_cpu = {k: v.cpu() for k, v in global_sd_gpu.items()}
    moe_load_state_dict_from_global(
        layer_ep, global_sd_cpu,
        ep_rank=ep_rank_local, ep_size=ep_size,
        experts_per_gpu=experts_per_gpu,
    )

    # Step 5: Forward with same input.
    with torch.no_grad():
        n_ep, e_ep, a_ep = layer_ep(**inputs_gpu)

    # Step 6: Compare with reference.
    # Node output should match exactly.
    assert torch.allclose(n_ep, ref_n, atol=1e-10), (
        f"Rank {rank}: node output mismatch. "
        f"Max diff: {(n_ep - ref_n).abs().max().item():.2e}"
    )
    assert torch.allclose(e_ep, ref_e, atol=1e-10), (
        f"Rank {rank}: edge output mismatch. "
        f"Max diff: {(e_ep - ref_e).abs().max().item():.2e}"
    )
    assert torch.allclose(a_ep, ref_a, atol=1e-10), (
        f"Rank {rank}: angle output mismatch. "
        f"Max diff: {(a_ep - ref_a).abs().max().item():.2e}"
    )

    return True


def test_4gpu_save_1gpu_load(rank, world_size):
    """T2b: 4 GPU (EP=4) save → 1 GPU load → output matches."""
    from deepmd.pt.utils.moe_checkpoint import (
        moe_load_state_dict_from_global,
        moe_state_dict_to_global,
    )

    device = torch.device(f"cuda:{rank}")

    # Step 1: Create EP=4 layer on each rank with specific seed so each
    # rank has different expert params.
    ep_group = dist.new_group(list(range(world_size)))
    ep_size = world_size  # 4
    experts_per_gpu = N_ROUTING_EXPERTS // ep_size  # 1

    layer_ep = _make_layer(
        ep_group=ep_group, ep_rank=rank, ep_size=ep_size, seed=42,
    )

    # Create inputs (same on all ranks).
    inputs_cpu = _make_inputs_on_cpu(seed=42)
    inputs_gpu = _inputs_to_device(inputs_cpu, device)

    # Forward on 4 GPUs.
    with torch.no_grad():
        n_ep, e_ep, a_ep = layer_ep(**inputs_gpu)

    # Step 2: Collect all expert params into global state_dict.
    global_sd = moe_state_dict_to_global(
        layer_ep, ep_rank=rank, ep_size=ep_size,
        experts_per_gpu=experts_per_gpu, ep_group=ep_group,
    )

    # Step 3: Rank 0 loads into single-GPU layer and compares.
    if rank == 0:
        global_sd_cpu = {k: v.cpu() for k, v in global_sd.items()}

        layer_1gpu = _make_layer(
            ep_group=None, ep_rank=0, ep_size=1, seed=999,
        )
        # Move to CPU for loading, then back to GPU.
        layer_1gpu_cpu = layer_1gpu.cpu()
        moe_load_state_dict_from_global(
            layer_1gpu_cpu, global_sd_cpu,
            ep_rank=0, ep_size=1,
            experts_per_gpu=N_ROUTING_EXPERTS,
        )
        layer_1gpu_gpu = layer_1gpu_cpu.to(device)

        with torch.no_grad():
            n_1gpu, e_1gpu, a_1gpu = layer_1gpu_gpu(**inputs_gpu)

        assert torch.allclose(n_1gpu, n_ep, atol=1e-10), (
            f"Node output mismatch. "
            f"Max diff: {(n_1gpu - n_ep).abs().max().item():.2e}"
        )
        assert torch.allclose(e_1gpu, e_ep, atol=1e-10), (
            f"Edge output mismatch. "
            f"Max diff: {(e_1gpu - e_ep).abs().max().item():.2e}"
        )
        assert torch.allclose(a_1gpu, a_ep, atol=1e-10), (
            f"Angle output mismatch. "
            f"Max diff: {(a_1gpu - a_ep).abs().max().item():.2e}"
        )

    return True


def test_1gpu_save_4gpu_load_first_order_grad(rank, world_size):
    """1st-order gradients match: 1 GPU save → 4 GPU (EP=4) load."""
    from deepmd.pt.utils.moe_checkpoint import (
        moe_load_state_dict_from_global,
        moe_state_dict_to_global,
    )

    device = torch.device(f"cuda:{rank}")

    # Step 1: Rank 0 creates single-GPU layer, saves state_dict.
    if rank == 0:
        layer_1gpu = _make_layer(ep_group=None, ep_rank=0, ep_size=1)
        inputs_cpu = _make_inputs_on_cpu(seed=42)
        inputs_gpu = _inputs_to_device(inputs_cpu, device, requires_grad=True)

        # Forward + backward on 1-GPU layer.
        n_ref, e_ref, a_ref = layer_1gpu(**inputs_gpu)
        loss_ref = (n_ref ** 2).sum() + (e_ref ** 2).sum() + (a_ref ** 2).sum()
        loss_ref.backward()
        grad_node_ref = inputs_gpu["node_ebd_ext"].grad.clone()
        grad_edge_ref = inputs_gpu["edge_ebd"].grad.clone()

        # Save global state_dict.
        global_sd = moe_state_dict_to_global(
            layer_1gpu, ep_rank=0, ep_size=1,
            experts_per_gpu=N_ROUTING_EXPERTS,
        )
    else:
        grad_node_ref = None
        grad_edge_ref = None
        global_sd = None
        inputs_cpu = None

    # Broadcast global state_dict.
    if rank == 0:
        global_sd_gpu = {k: v.cuda() for k, v in global_sd.items()}
    else:
        global_sd_gpu = None
    global_sd_gpu = _broadcast_state_dict(
        global_sd_gpu if rank == 0 else {}, src=0,
    )

    # Broadcast inputs (no requires_grad for broadcasting).
    if rank == 0:
        inputs_cpu_nograd = _make_inputs_on_cpu(seed=42, requires_grad=False)
        inputs_list = [inputs_cpu_nograd]
    else:
        inputs_list = [None]
    dist.broadcast_object_list(inputs_list, src=0)
    inputs_cpu = inputs_list[0]
    inputs_gpu = _inputs_to_device(inputs_cpu, device, requires_grad=True)

    # Broadcast reference gradients.
    if rank == 0:
        ref_gn = grad_node_ref.contiguous()
        ref_ge = grad_edge_ref.contiguous()
    else:
        ref_gn = torch.empty(NB, NALL, N_DIM, dtype=DTYPE, device=device)
        ref_ge = torch.empty(N_EDGE, E_DIM, dtype=DTYPE, device=device)
    dist.broadcast(ref_gn, src=0)
    dist.broadcast(ref_ge, src=0)

    # Step 2: Create EP=4 layer, load global state_dict.
    ep_group = dist.new_group(list(range(world_size)))
    ep_rank_local = rank
    ep_size = world_size  # 4
    experts_per_gpu = N_ROUTING_EXPERTS // ep_size  # 1

    layer_ep = _make_layer(
        ep_group=ep_group, ep_rank=ep_rank_local, ep_size=ep_size, seed=999,
    )

    global_sd_cpu = {k: v.cpu() for k, v in global_sd_gpu.items()}
    moe_load_state_dict_from_global(
        layer_ep, global_sd_cpu,
        ep_rank=ep_rank_local, ep_size=ep_size,
        experts_per_gpu=experts_per_gpu,
    )

    # Step 3: Forward + backward on EP=4 layer.
    n_ep, e_ep, a_ep = layer_ep(**inputs_gpu)
    loss_ep = (n_ep ** 2).sum() + (e_ep ** 2).sum() + (a_ep ** 2).sum()
    loss_ep.backward()
    grad_node_ep = inputs_gpu["node_ebd_ext"].grad.clone()
    grad_edge_ep = inputs_gpu["edge_ebd"].grad.clone()

    # Step 4: Compare input gradients with reference.
    assert torch.allclose(grad_node_ep, ref_gn, atol=1e-10), (
        f"Rank {rank}: 1st-order node_ebd_ext grad mismatch. "
        f"Max diff: {(grad_node_ep - ref_gn).abs().max().item():.2e}"
    )
    assert torch.allclose(grad_edge_ep, ref_ge, atol=1e-10), (
        f"Rank {rank}: 1st-order edge_ebd grad mismatch. "
        f"Max diff: {(grad_edge_ep - ref_ge).abs().max().item():.2e}"
    )

    return True


def test_1gpu_save_4gpu_load_second_order_grad(rank, world_size):
    """2nd-order gradients match: 1 GPU save → 4 GPU (EP=4) load."""
    from deepmd.pt.utils.moe_checkpoint import (
        moe_load_state_dict_from_global,
        moe_state_dict_to_global,
    )

    device = torch.device(f"cuda:{rank}")

    # Step 1: Rank 0 creates single-GPU layer, computes 2nd-order grads.
    if rank == 0:
        layer_1gpu = _make_layer(ep_group=None, ep_rank=0, ep_size=1)
        inputs_cpu = _make_inputs_on_cpu(seed=42)
        inputs_gpu = _inputs_to_device(inputs_cpu, device, requires_grad=True)

        n_ref, e_ref, a_ref = layer_1gpu(**inputs_gpu)
        loss_ref = (n_ref ** 2).sum() + (e_ref ** 2).sum() + (a_ref ** 2).sum()
        (grad_node_ref,) = torch.autograd.grad(
            loss_ref, inputs_gpu["node_ebd_ext"], create_graph=True,
        )
        grad_loss_ref = (grad_node_ref ** 2).sum()
        grad_loss_ref.backward()
        second_grad_node_ref = inputs_gpu["node_ebd_ext"].grad.clone()
        first_grad_node_ref = grad_node_ref.detach().clone()

        global_sd = moe_state_dict_to_global(
            layer_1gpu, ep_rank=0, ep_size=1,
            experts_per_gpu=N_ROUTING_EXPERTS,
        )
    else:
        second_grad_node_ref = None
        first_grad_node_ref = None
        global_sd = None
        inputs_cpu = None

    # Broadcast global state_dict.
    if rank == 0:
        global_sd_gpu = {k: v.cuda() for k, v in global_sd.items()}
    else:
        global_sd_gpu = None
    global_sd_gpu = _broadcast_state_dict(
        global_sd_gpu if rank == 0 else {}, src=0,
    )

    # Broadcast inputs.
    if rank == 0:
        inputs_cpu_nograd = _make_inputs_on_cpu(seed=42, requires_grad=False)
        inputs_list = [inputs_cpu_nograd]
    else:
        inputs_list = [None]
    dist.broadcast_object_list(inputs_list, src=0)
    inputs_cpu = inputs_list[0]
    inputs_gpu = _inputs_to_device(inputs_cpu, device, requires_grad=True)

    # Broadcast reference gradients.
    if rank == 0:
        ref_1st = first_grad_node_ref.contiguous()
        ref_2nd = second_grad_node_ref.contiguous()
    else:
        ref_1st = torch.empty(NB, NALL, N_DIM, dtype=DTYPE, device=device)
        ref_2nd = torch.empty(NB, NALL, N_DIM, dtype=DTYPE, device=device)
    dist.broadcast(ref_1st, src=0)
    dist.broadcast(ref_2nd, src=0)

    # Step 2: Create EP=4 layer, load global state_dict.
    ep_group = dist.new_group(list(range(world_size)))
    ep_rank_local = rank
    ep_size = world_size  # 4
    experts_per_gpu = N_ROUTING_EXPERTS // ep_size  # 1

    layer_ep = _make_layer(
        ep_group=ep_group, ep_rank=ep_rank_local, ep_size=ep_size, seed=999,
    )

    global_sd_cpu = {k: v.cpu() for k, v in global_sd_gpu.items()}
    moe_load_state_dict_from_global(
        layer_ep, global_sd_cpu,
        ep_rank=ep_rank_local, ep_size=ep_size,
        experts_per_gpu=experts_per_gpu,
    )

    # Step 3: 2nd-order forward+backward on EP=4 layer.
    n_ep, e_ep, a_ep = layer_ep(**inputs_gpu)
    loss_ep = (n_ep ** 2).sum() + (e_ep ** 2).sum() + (a_ep ** 2).sum()
    (grad_node_ep,) = torch.autograd.grad(
        loss_ep, inputs_gpu["node_ebd_ext"], create_graph=True,
    )
    grad_loss_ep = (grad_node_ep ** 2).sum()
    grad_loss_ep.backward()
    second_grad_node_ep = inputs_gpu["node_ebd_ext"].grad.clone()
    first_grad_node_ep = grad_node_ep.detach().clone()

    # Step 4: Compare.
    assert torch.allclose(first_grad_node_ep, ref_1st, atol=1e-10), (
        f"Rank {rank}: 2nd-order: 1st intermediate grad mismatch. "
        f"Max diff: {(first_grad_node_ep - ref_1st).abs().max().item():.2e}"
    )
    assert torch.allclose(second_grad_node_ep, ref_2nd, atol=1e-10), (
        f"Rank {rank}: 2nd-order: input grad mismatch. "
        f"Max diff: {(second_grad_node_ep - ref_2nd).abs().max().item():.2e}"
    )

    return True


def test_1gpu_save_2gpu_load(rank, world_size):
    """1 GPU save → 2 GPU (EP=2) load → output matches (uses first 2 ranks)."""
    if world_size < 2:
        return True

    from deepmd.pt.utils.moe_checkpoint import (
        moe_load_state_dict_from_global,
        moe_state_dict_to_global,
    )

    device = torch.device(f"cuda:{rank}")

    # Rank 0 creates 1-GPU layer and computes reference.
    if rank == 0:
        layer_1gpu = _make_layer(ep_group=None, ep_rank=0, ep_size=1)
        inputs_cpu = _make_inputs_on_cpu(seed=77)
        inputs_gpu = _inputs_to_device(inputs_cpu, device)

        with torch.no_grad():
            n_ref, e_ref, a_ref = layer_1gpu(**inputs_gpu)

        global_sd = moe_state_dict_to_global(
            layer_1gpu, 0, 1, N_ROUTING_EXPERTS,
        )
    else:
        n_ref = e_ref = a_ref = global_sd = inputs_cpu = None

    # Broadcast.
    if rank == 0:
        global_sd_gpu = {k: v.cuda() for k, v in global_sd.items()}
    else:
        global_sd_gpu = None
    global_sd_gpu = _broadcast_state_dict(
        global_sd_gpu if rank == 0 else {}, src=0,
    )

    if rank == 0:
        inputs_list = [inputs_cpu]
    else:
        inputs_list = [None]
    dist.broadcast_object_list(inputs_list, src=0)
    inputs_cpu = inputs_list[0]
    inputs_gpu = _inputs_to_device(inputs_cpu, device)

    if rank == 0:
        ref_n = n_ref.contiguous()
        ref_e = e_ref.contiguous()
        ref_a = a_ref.contiguous()
    else:
        ref_n = torch.empty(NB, NLOC, N_DIM, dtype=DTYPE, device=device)
        ref_e = torch.empty(N_EDGE, E_DIM, dtype=DTYPE, device=device)
        ref_a = torch.empty(N_ANGLE, A_DIM, dtype=DTYPE, device=device)
    dist.broadcast(ref_n, src=0)
    dist.broadcast(ref_e, src=0)
    dist.broadcast(ref_a, src=0)

    # Create EP=2 group with ranks [0,1]. Ranks 2,3 do not participate.
    ep_size = 2
    ep_ranks = list(range(ep_size))
    ep_group = dist.new_group(ep_ranks)
    experts_per_gpu = N_ROUTING_EXPERTS // ep_size  # 2

    if rank < ep_size:
        layer_ep = _make_layer(
            ep_group=ep_group, ep_rank=rank, ep_size=ep_size, seed=999,
        )
        global_sd_cpu = {k: v.cpu() for k, v in global_sd_gpu.items()}
        moe_load_state_dict_from_global(
            layer_ep, global_sd_cpu,
            ep_rank=rank, ep_size=ep_size,
            experts_per_gpu=experts_per_gpu,
        )

        with torch.no_grad():
            n_ep, e_ep, a_ep = layer_ep(**inputs_gpu)

        assert torch.allclose(n_ep, ref_n, atol=1e-10), (
            f"Rank {rank}: node mismatch (EP=2). "
            f"Max diff: {(n_ep - ref_n).abs().max().item():.2e}"
        )
        assert torch.allclose(e_ep, ref_e, atol=1e-10), (
            f"Rank {rank}: edge mismatch (EP=2). "
            f"Max diff: {(e_ep - ref_e).abs().max().item():.2e}"
        )
        assert torch.allclose(a_ep, ref_a, atol=1e-10), (
            f"Rank {rank}: angle mismatch (EP=2). "
            f"Max diff: {(a_ep - ref_a).abs().max().item():.2e}"
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
        print(f"\nMoE Checkpoint Multi-GPU Tests ({world_size} GPUs)")
        print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        (test_1gpu_save_4gpu_load, "T2: 1 GPU save -> 4 GPU (EP=4) load"),
        (test_4gpu_save_1gpu_load, "T2b: 4 GPU (EP=4) save -> 1 GPU load"),
        (test_1gpu_save_4gpu_load_first_order_grad,
         "T2 1st-order grad: 1 GPU save -> 4 GPU (EP=4) load"),
        (test_1gpu_save_4gpu_load_second_order_grad,
         "T2 2nd-order grad: 1 GPU save -> 4 GPU (EP=4) load"),
        (test_1gpu_save_2gpu_load, "1 GPU save -> 2 GPU (EP=2) load"),
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
