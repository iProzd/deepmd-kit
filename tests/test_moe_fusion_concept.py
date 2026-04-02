#!/usr/bin/env python3
"""Verify MLP fusion equivalence: fused wide MoE == two separate MoEs.

Launch: torchrun --nproc_per_node=2 tests/test_moe_fusion_concept.py
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from deepmd.pt.model.network.moe import MoELayer
from deepmd.pt.model.network.moe_fused import FusedMoELayer


def setup():
    dist.init_process_group("nccl")
    rank = int(dist.get_rank())
    torch.cuda.set_device(rank)
    return rank


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def _copy_weights_to_fused(moe1, moe2, fused, n_experts_local, out1):
    """Copy gate and expert weights from two separate MoEs into one fused MoE."""
    with torch.no_grad():
        fused.fused_moe.gate.matrix.data.copy_(moe1.gate.matrix.data)
        for i in range(n_experts_local):
            # matrix shape: (num_in, num_out) — output dim on axis 1
            fused.fused_moe.experts[i].matrix.data[:, :out1] = moe1.experts[i].matrix.data
            fused.fused_moe.experts[i].matrix.data[:, out1:] = moe2.experts[i].matrix.data
            if moe1.experts[i].bias is not None:
                fused.fused_moe.experts[i].bias.data[:out1] = moe1.experts[i].bias.data
                fused.fused_moe.experts[i].bias.data[out1:] = moe2.experts[i].bias.data


def test_fusion_local(rank):
    """Test fusion equivalence on single GPU (no EP)."""
    if rank != 0:
        return

    torch.manual_seed(42)
    num_in, out1, out2 = 64, 32, 16
    n_experts, top_k, tebd_dim, ntypes = 4, 2, 32, 4
    nb, nloc = 2, 8
    act_fn = "silu"  # match RepFlowLayer convention

    x = torch.randn(nb, nloc, num_in, device="cuda")
    type_emb = torch.randn(ntypes, tebd_dim, device="cuda")
    atom_types = torch.randint(0, ntypes, (nb, nloc), device="cuda")

    # Two separate MoEs with same gate and activation
    moe1 = MoELayer(num_in, out1, n_experts, top_k, tebd_dim,
                     activation_function=act_fn, seed=100).cuda()
    moe2 = MoELayer(num_in, out2, n_experts, top_k, tebd_dim,
                     activation_function=act_fn, seed=200).cuda()
    moe2.gate.matrix.data.copy_(moe1.gate.matrix.data)

    with torch.no_grad():
        y1_sep = moe1(x, type_emb, atom_types)
        y2_sep = moe2(x, type_emb, atom_types)

    # Fused MoE: copy gate + expert weights
    fused = FusedMoELayer(num_in, [out1, out2], n_experts, top_k, tebd_dim,
                          activation_function=act_fn, seed=300).cuda()
    _copy_weights_to_fused(moe1, moe2, fused, n_experts, out1)

    with torch.no_grad():
        y1_fused, y2_fused = fused(x, type_emb, atom_types)

    diff1 = (y1_sep - y1_fused).abs().max().item()
    diff2 = (y2_sep - y2_fused).abs().max().item()

    print(f"[Local] Output 1 diff: {diff1:.2e}")
    print(f"[Local] Output 2 diff: {diff2:.2e}")
    assert diff1 < 1e-5, f"Local fusion diff1 too large: {diff1}"
    assert diff2 < 1e-5, f"Local fusion diff2 too large: {diff2}"
    print("[PASS] Local fusion equivalence")


def test_fusion_ep(rank, ep_group):
    """Test fusion equivalence with EP."""
    torch.manual_seed(42)
    num_in, out1, out2 = 64, 32, 16
    n_experts, top_k, tebd_dim, ntypes = 4, 2, 32, 4
    nb, nloc = 2, 8
    act_fn = "silu"
    ep_size = dist.get_world_size(group=ep_group)
    experts_per_gpu = n_experts // ep_size

    x = torch.randn(nb, nloc, num_in, device="cuda")
    type_emb = torch.randn(ntypes, tebd_dim, device="cuda")
    atom_types = torch.randint(0, ntypes, (nb, nloc), device="cuda")
    dist.broadcast(x, src=0, group=ep_group)
    dist.broadcast(type_emb, src=0, group=ep_group)
    dist.broadcast(atom_types, src=0, group=ep_group)

    # Two separate EP MoEs with same gate
    moe1 = MoELayer(num_in, out1, n_experts, top_k, tebd_dim,
                     activation_function=act_fn, ep_group=ep_group, seed=100).cuda()
    moe2 = MoELayer(num_in, out2, n_experts, top_k, tebd_dim,
                     activation_function=act_fn, ep_group=ep_group, seed=200).cuda()
    dist.broadcast(moe1.gate.matrix.data, src=0, group=ep_group)
    moe2.gate.matrix.data.copy_(moe1.gate.matrix.data)

    with torch.no_grad():
        y1_sep = moe1(x, type_emb, atom_types)
        y2_sep = moe2(x, type_emb, atom_types)

    # Fused EP MoE: copy gate + local expert weights
    fused = FusedMoELayer(num_in, [out1, out2], n_experts, top_k, tebd_dim,
                          activation_function=act_fn, ep_group=ep_group, seed=300).cuda()
    _copy_weights_to_fused(moe1, moe2, fused, experts_per_gpu, out1)

    with torch.no_grad():
        y1_fused, y2_fused = fused(x, type_emb, atom_types)

    diff1 = (y1_sep - y1_fused).abs().max().item()
    diff2 = (y2_sep - y2_fused).abs().max().item()

    log(rank, f"[EP] Output 1 diff: {diff1:.2e}")
    log(rank, f"[EP] Output 2 diff: {diff2:.2e}")
    assert diff1 < 1e-5, f"EP fusion diff1 too large: {diff1}"
    assert diff2 < 1e-5, f"EP fusion diff2 too large: {diff2}"
    log(rank, "[PASS] EP fusion equivalence")


def test_fusion_gradient(rank, ep_group):
    """Test that fused MoE produces valid gradients for force training."""
    torch.manual_seed(42)
    num_in, out1, out2 = 64, 32, 16
    n_experts, top_k, tebd_dim, ntypes = 4, 2, 32, 4
    nb, nloc = 2, 8
    act_fn = "silu"

    type_emb = torch.randn(ntypes, tebd_dim, device="cuda")
    atom_types = torch.randint(0, ntypes, (nb, nloc), device="cuda")
    dist.broadcast(type_emb, src=0, group=ep_group)
    dist.broadcast(atom_types, src=0, group=ep_group)

    encoder = torch.nn.Linear(3, num_in, bias=False).cuda()
    dist.broadcast(encoder.weight.data, src=0, group=ep_group)

    fused = FusedMoELayer(num_in, [out1, out2], n_experts, top_k, tebd_dim,
                          activation_function=act_fn, ep_group=ep_group, seed=300).cuda()
    dist.broadcast(fused.fused_moe.gate.matrix.data, src=0, group=ep_group)

    # Forward with gradient tracking
    pos = torch.randn(nb, nloc, 3, device="cuda", requires_grad=True)
    hidden = encoder(pos)
    y1, y2 = fused(hidden, type_emb, atom_types)
    energy = y1.sum() + y2.sum()
    force = -torch.autograd.grad(energy, pos, create_graph=True, retain_graph=True)[0]
    loss = energy + force.sum()
    loss.backward()

    # Check gradients
    has_grad = all(p.grad is not None and not p.grad.isnan().any()
                   for p in fused.parameters() if p.requires_grad)
    pos_grad_ok = pos.grad is not None and not pos.grad.isnan().any()

    log(rank, f"[Grad] All param grads valid: {has_grad}")
    log(rank, f"[Grad] Pos grad valid: {pos_grad_ok}")
    assert has_grad, "Some fused MoE parameters have no or NaN gradients"
    assert pos_grad_ok, "Position gradient is invalid"
    log(rank, "[PASS] Fused MoE gradient test")


def test_fusion_speedup(rank, ep_group):
    """Benchmark fused vs separate MoE with EP."""
    import time

    torch.manual_seed(42)
    num_in, out1, out2 = 256, 128, 64
    n_experts, top_k, tebd_dim, ntypes = 8, 2, 32, 8
    nb, nloc = 8, 128
    act_fn = "silu"
    warmup, steps = 20, 100

    type_emb = torch.randn(ntypes, tebd_dim, device="cuda")
    atom_types = torch.randint(0, ntypes, (nb, nloc), device="cuda")
    dist.broadcast(type_emb, src=0, group=ep_group)
    dist.broadcast(atom_types, src=0, group=ep_group)

    # Separate MoEs
    moe1 = MoELayer(num_in, out1, n_experts, top_k, tebd_dim,
                     activation_function=act_fn, ep_group=ep_group, seed=100).cuda()
    moe2 = MoELayer(num_in, out2, n_experts, top_k, tebd_dim,
                     activation_function=act_fn, ep_group=ep_group, seed=200).cuda()
    dist.broadcast(moe1.gate.matrix.data, src=0, group=ep_group)
    moe2.gate.matrix.data.copy_(moe1.gate.matrix.data)

    # Fused MoE
    fused = FusedMoELayer(num_in, [out1, out2], n_experts, top_k, tebd_dim,
                          activation_function=act_fn, ep_group=ep_group, seed=300).cuda()
    dist.broadcast(fused.fused_moe.gate.matrix.data, src=0, group=ep_group)

    def run_separate():
        x = torch.randn(nb, nloc, num_in, device="cuda")
        y1 = moe1(x, type_emb, atom_types)
        y2 = moe2(x, type_emb, atom_types)
        return y1.sum() + y2.sum()

    def run_fused():
        x = torch.randn(nb, nloc, num_in, device="cuda")
        outs = fused(x, type_emb, atom_types)
        return outs[0].sum() + outs[1].sum()

    # Warmup
    for _ in range(warmup):
        run_separate()
        run_fused()

    # Benchmark separate
    torch.cuda.synchronize()
    dist.barrier(group=ep_group)
    t0 = time.perf_counter()
    for _ in range(steps):
        run_separate()
    torch.cuda.synchronize()
    sep_time = time.perf_counter() - t0

    # Benchmark fused
    torch.cuda.synchronize()
    dist.barrier(group=ep_group)
    t0 = time.perf_counter()
    for _ in range(steps):
        run_fused()
    torch.cuda.synchronize()
    fused_time = time.perf_counter() - t0

    log(rank, f"[Speedup] Separate: {steps/sep_time:.1f} steps/s ({sep_time/steps*1000:.2f} ms/step)")
    log(rank, f"[Speedup] Fused:    {steps/fused_time:.1f} steps/s ({fused_time/steps*1000:.2f} ms/step)")
    speedup = sep_time / fused_time
    log(rank, f"[Speedup] Fused/Separate: {speedup:.2f}x")
    log(rank, f"[Speedup] A2A calls: Separate=4, Fused=2 (50% reduction)")


def main():
    rank = setup()
    mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("dp", "ep"))
    ep_group = mesh["ep"].get_group()

    log(rank, "=" * 70)
    log(rank, "MoE Fusion Tests")
    log(rank, "=" * 70)

    test_fusion_local(rank)
    dist.barrier(group=ep_group)

    test_fusion_ep(rank, ep_group)
    dist.barrier(group=ep_group)

    test_fusion_gradient(rank, ep_group)
    dist.barrier(group=ep_group)

    test_fusion_speedup(rank, ep_group)

    log(rank, "\n" + "=" * 70)
    log(rank, "All fusion tests passed")
    log(rank, "=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
