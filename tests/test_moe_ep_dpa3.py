#!/usr/bin/env python3
"""MoE Expert Parallelism tests and benchmarks for DPA3 MoELayer.

Launch: torchrun --nproc_per_node=2 tests/test_moe_ep_dpa3.py [--bench]

Tests:
  1. All-to-All op: forward + 2nd-order gradient
  2. MoELayer EP: forward correctness vs single-GPU reference
  3. MoELayer EP: 2nd-order gradient flow (force training pattern)
  4. Multi-step gradient stability

Benchmarks (--bench):
  Sweep expert counts [2,4,6,8] x model sizes, EP vs single-GPU
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh


def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


# ======================================================================
# Test 1: All-to-All forward + 2nd-order through non-linear chain
# ======================================================================
def test_all_to_all_double_backward(rank, ep_group):
    """Verify All-to-All forward is correct and supports 2nd-order grad."""
    from deepmd.pt.model.network.moe_ep_ops import all_to_all_differentiable

    ep_size = dist.get_world_size(group=ep_group)
    D = 16
    N = 8  # tokens per GPU

    x = torch.randn(N, D, device="cuda", requires_grad=True)

    tokens_per_peer = N // ep_size
    send_splits = [tokens_per_peer] * ep_size
    recv_splits = [tokens_per_peer] * ep_size

    y = all_to_all_differentiable(x, send_splits, recv_splits, ep_group)

    # Forward correctness: 1st-order grad should exist
    loss = y.sum()
    grad1 = torch.autograd.grad(loss, x, create_graph=True)[0]
    assert grad1.shape == x.shape, f"grad1 shape mismatch: {grad1.shape}"

    # Chain with non-linear op and verify 2nd-order grad
    w = torch.randn(D, D, device="cuda", requires_grad=True)
    y2 = all_to_all_differentiable(x, send_splits, recv_splits, ep_group)
    z = torch.nn.functional.silu(y2 @ w)
    scalar = z.sum()

    grad_x = torch.autograd.grad(scalar, x, create_graph=True)[0]
    assert grad_x.shape == x.shape
    grad2_x = torch.autograd.grad(grad_x.sum(), x, allow_unused=True)[0]
    assert grad2_x is not None, "2nd-order grad through A2A + SiLU should not be None"
    assert grad2_x.shape == x.shape

    log(rank, "[PASS] test_all_to_all_double_backward")


# ======================================================================
# Test 2: EP MoELayer forward correctness vs single-GPU reference
# ======================================================================
def test_ep_vs_local_correctness(rank, ep_group, num_experts=4):
    """Compare EP MoELayer output against single-GPU reference with same weights."""
    from deepmd.pt.model.network.moe import MoELayer

    torch.manual_seed(42)
    num_in, num_out = 32, 32
    tebd_dim = 32
    top_k = 2
    ntypes = 3
    nb, nloc = 2, 8

    ep_size = dist.get_world_size(group=ep_group)
    ep_rank = dist.get_rank(group=ep_group)
    experts_per_gpu = num_experts // ep_size

    # --- EP model ---
    moe_ep = MoELayer(
        num_in, num_out, num_experts, top_k, tebd_dim,
        ep_group=ep_group, seed=123,
    ).cuda()

    # Sync gate weights across ranks
    dist.broadcast(moe_ep.gate.matrix.data, src=0)
    # Sync shared params not needed (no shared experts in this test)

    # --- Build single-GPU reference with ALL experts ---
    moe_local = MoELayer(
        num_in, num_out, num_experts, top_k, tebd_dim,
        ep_group=None, seed=123,
    ).cuda()

    # Copy gate from EP model
    moe_local.gate.matrix.data.copy_(moe_ep.gate.matrix.data)

    # Gather and copy expert weights from EP model
    # Each EP rank has experts_per_gpu local experts
    for rank_i in range(ep_size):
        for local_i in range(experts_per_gpu):
            global_expert_id = rank_i * experts_per_gpu + local_i
            if rank_i == ep_rank:
                # I own this expert — broadcast my weights
                for p_local, p_ref in zip(
                    moe_ep.experts[local_i].parameters(),
                    moe_local.experts[global_expert_id].parameters(),
                ):
                    src_data = p_local.data.clone()
                    dist.broadcast(src_data, src=rank_i, group=ep_group)
                    p_ref.data.copy_(src_data)
            else:
                # Someone else owns this expert — receive weights
                for p_ref in moe_local.experts[global_expert_id].parameters():
                    src_data = torch.empty_like(p_ref.data)
                    dist.broadcast(src_data, src=rank_i, group=ep_group)
                    p_ref.data.copy_(src_data)

    # Same input on all ranks
    torch.manual_seed(999)
    type_emb = torch.randn(ntypes, tebd_dim, device="cuda")
    atom_types = torch.randint(0, ntypes, (nb, nloc), device="cuda")

    # Test node-level input [nb, nloc, dim]
    x_node = torch.randn(nb, nloc, num_in, device="cuda")
    with torch.no_grad():
        out_ep = moe_ep(x_node, type_emb, atom_types)
        out_local = moe_local(x_node, type_emb, atom_types)
    diff_node = (out_ep - out_local).abs().max().item()

    # Test edge-level input [nb, nloc, nnei, dim]
    nnei = 10
    x_edge = torch.randn(nb, nloc, nnei, num_in, device="cuda")
    with torch.no_grad():
        out_ep_e = moe_ep(x_edge, type_emb, atom_types)
        out_local_e = moe_local(x_edge, type_emb, atom_types)
    diff_edge = (out_ep_e - out_local_e).abs().max().item()

    log(rank, f"  [EP-correctness] E={num_experts} node_diff={diff_node:.2e} edge_diff={diff_edge:.2e}")
    assert diff_node < 1e-5, f"Node EP vs local diff too large: {diff_node}"
    assert diff_edge < 1e-5, f"Edge EP vs local diff too large: {diff_edge}"

    log(rank, "[PASS] test_ep_vs_local_correctness")


# ======================================================================
# Test 3: 2nd-order gradient flow (force training pattern)
# ======================================================================
def test_ep_2nd_order_grad(rank, ep_group, num_experts=4):
    """Verify EP MoELayer supports 2nd-order autograd (force training)."""
    from deepmd.pt.model.network.moe import MoELayer

    torch.manual_seed(42)
    num_in, num_out = 32, 32
    tebd_dim = 32
    top_k = 2
    ntypes = 3
    nb, nloc = 2, 8

    moe = MoELayer(
        num_in, num_out, num_experts, top_k, tebd_dim,
        ep_group=ep_group, seed=42,
    ).cuda()

    # Sync gate weights
    dist.broadcast(moe.gate.matrix.data, src=0)

    type_emb = torch.randn(ntypes, tebd_dim, device="cuda")
    atom_types = torch.randint(0, ntypes, (nb, nloc), device="cuda")
    dist.broadcast(type_emb, src=0)
    dist.broadcast(atom_types, src=0)

    # Simulate positions -> MoE -> energy -> force pattern
    positions = torch.randn(nb, nloc, 3, device="cuda", requires_grad=True)

    # Simple encoder: positions -> hidden
    encoder = nn.Linear(3, num_in, bias=False).cuda()
    dist.broadcast(encoder.weight.data, src=0)

    hidden = encoder(positions)
    out = moe(hidden, type_emb, atom_types)
    energy = out.sum(dim=-1).sum(dim=-1)  # [nb]

    # 1st-order: force
    force = -torch.autograd.grad(
        energy.sum(), positions, create_graph=True, retain_graph=True
    )[0]
    assert force.shape == positions.shape, f"Force shape mismatch: {force.shape}"
    assert not torch.isnan(force).any(), "Force has NaN"

    # 2nd-order: loss backward through force
    force_target = torch.randn_like(force)
    force_loss = nn.functional.mse_loss(force, force_target)
    force_loss.backward()

    # Check gradients exist and are non-NaN
    has_grad = 0
    for name, p in moe.named_parameters():
        if p.grad is not None and p.grad.abs().max() > 0:
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
            has_grad += 1

    log(rank, f"  [2nd-order-grad] E={num_experts} params_with_grad={has_grad} force_loss={force_loss.item():.4f}")
    assert has_grad > 0, "No parameter received gradient through 2nd-order path"

    log(rank, "[PASS] test_ep_2nd_order_grad")


# ======================================================================
# Test 4: Multi-step gradient stability
# ======================================================================
def test_gradient_stability(rank, ep_group, num_experts=4, steps=20):
    """Verify no NaN/Inf gradients over multiple training steps."""
    from deepmd.pt.model.network.moe import MoELayer

    torch.manual_seed(42)
    num_in, num_out = 32, 32
    tebd_dim = 32
    top_k = 2
    ntypes = 3
    nb, nloc = 4, 16

    moe = MoELayer(
        num_in, num_out, num_experts, top_k, tebd_dim,
        ep_group=ep_group, seed=42,
    ).cuda()
    dist.broadcast(moe.gate.matrix.data, src=0)

    encoder = nn.Linear(3, num_in, bias=False).cuda()
    dist.broadcast(encoder.weight.data, src=0)

    all_params = list(moe.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(all_params, lr=1e-3)

    type_emb = torch.randn(ntypes, tebd_dim, device="cuda")
    atom_types = torch.randint(0, ntypes, (nb, nloc), device="cuda")
    dist.broadcast(type_emb, src=0)
    dist.broadcast(atom_types, src=0)

    losses = []
    for step in range(steps):
        pos = torch.randn(nb, nloc, 3, device="cuda", requires_grad=True)
        hidden = encoder(pos)
        out = moe(hidden, type_emb, atom_types)
        energy = out.sum(dim=-1).sum(dim=-1)

        force = -torch.autograd.grad(
            energy.sum(), pos, create_graph=True, retain_graph=True
        )[0]

        e_tgt = torch.randn(nb, device="cuda")
        f_tgt = torch.randn_like(force)
        loss = nn.functional.mse_loss(energy, e_tgt) + nn.functional.mse_loss(force, f_tgt)

        opt.zero_grad()
        loss.backward()

        # Check for NaN/Inf
        for name, p in moe.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN at step {step} in {name}"
                assert not torch.isinf(p.grad).any(), f"Inf at step {step} in {name}"

        # Sync shared gradients across EP ranks
        for name, p in moe.named_parameters():
            if p.grad is not None and "experts" not in name:
                dist.all_reduce(p.grad, group=ep_group)
                p.grad /= dist.get_world_size(group=ep_group)
        for p in encoder.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, group=ep_group)
                p.grad /= dist.get_world_size(group=ep_group)

        opt.step()
        losses.append(loss.item())

    log(rank, f"  [stability] E={num_experts} {steps} steps loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    log(rank, "[PASS] test_gradient_stability")


# ======================================================================
# Test 5: GPU-level A2A correctness vs expert-level A2A
# ======================================================================
def test_gpu_level_vs_expert_level(rank, ep_group, num_experts=4):
    """Compare GPU-level A2A MoELayer output against expert-level A2A reference."""
    from deepmd.pt.model.network.moe import MoELayer

    torch.manual_seed(42)
    num_in, num_out = 32, 32
    tebd_dim = 32
    top_k = 2
    ntypes = 3
    nb, nloc = 2, 8

    ep_size = dist.get_world_size(group=ep_group)
    ep_rank = dist.get_rank(group=ep_group)
    experts_per_gpu = num_experts // ep_size

    # --- Expert-level EP model ---
    moe_expert_lvl = MoELayer(
        num_in, num_out, num_experts, top_k, tebd_dim,
        ep_group=ep_group, seed=123, gpu_level_a2a=False,
    ).cuda()

    # --- GPU-level EP model (same weights) ---
    moe_gpu_lvl = MoELayer(
        num_in, num_out, num_experts, top_k, tebd_dim,
        ep_group=ep_group, seed=123, gpu_level_a2a=True,
    ).cuda()

    # Copy weights from expert_lvl to gpu_lvl
    moe_gpu_lvl.load_state_dict(moe_expert_lvl.state_dict())

    # Sync gate weights across ranks
    dist.broadcast(moe_expert_lvl.gate.matrix.data, src=0)
    dist.broadcast(moe_gpu_lvl.gate.matrix.data, src=0)

    # Sync expert weights
    for local_i in range(experts_per_gpu):
        for p_el, p_gl in zip(
            moe_expert_lvl.experts[local_i].parameters(),
            moe_gpu_lvl.experts[local_i].parameters(),
        ):
            dist.broadcast(p_el.data, src=0, group=ep_group)
            p_gl.data.copy_(p_el.data)

    # Same input on all ranks
    torch.manual_seed(999)
    type_emb = torch.randn(ntypes, tebd_dim, device="cuda")
    atom_types = torch.randint(0, ntypes, (nb, nloc), device="cuda")
    dist.broadcast(type_emb, src=0, group=ep_group)
    dist.broadcast(atom_types, src=0, group=ep_group)

    # Test node-level input [nb, nloc, dim]
    x_node = torch.randn(nb, nloc, num_in, device="cuda")
    dist.broadcast(x_node, src=0, group=ep_group)
    with torch.no_grad():
        out_el = moe_expert_lvl(x_node, type_emb, atom_types)
        out_gl = moe_gpu_lvl(x_node, type_emb, atom_types)
    diff_node = (out_el - out_gl).abs().max().item()

    # Test edge-level input [nb, nloc, nnei, dim]
    nnei = 10
    x_edge = torch.randn(nb, nloc, nnei, num_in, device="cuda")
    dist.broadcast(x_edge, src=0, group=ep_group)
    with torch.no_grad():
        out_el_e = moe_expert_lvl(x_edge, type_emb, atom_types)
        out_gl_e = moe_gpu_lvl(x_edge, type_emb, atom_types)
    diff_edge = (out_el_e - out_gl_e).abs().max().item()

    log(rank, f"  [GPU-lvl-correctness] E={num_experts} node_diff={diff_node:.2e} edge_diff={diff_edge:.2e}")
    assert diff_node < 1e-4, f"Node GPU-level vs expert-level diff too large: {diff_node}"
    assert diff_edge < 1e-4, f"Edge GPU-level vs expert-level diff too large: {diff_edge}"

    log(rank, "[PASS] test_gpu_level_vs_expert_level")


# ======================================================================
# Test 6: GPU-level A2A 2nd-order gradient flow
# ======================================================================
def test_gpu_level_2nd_order_grad(rank, ep_group, num_experts=4):
    """Verify GPU-level A2A MoELayer supports 2nd-order autograd (force training)."""
    from deepmd.pt.model.network.moe import MoELayer

    torch.manual_seed(42)
    num_in, num_out = 32, 32
    tebd_dim = 32
    top_k = 2
    ntypes = 3
    nb, nloc = 2, 8

    moe = MoELayer(
        num_in, num_out, num_experts, top_k, tebd_dim,
        ep_group=ep_group, seed=42, gpu_level_a2a=True,
    ).cuda()

    dist.broadcast(moe.gate.matrix.data, src=0)

    type_emb = torch.randn(ntypes, tebd_dim, device="cuda")
    atom_types = torch.randint(0, ntypes, (nb, nloc), device="cuda")
    dist.broadcast(type_emb, src=0, group=ep_group)
    dist.broadcast(atom_types, src=0, group=ep_group)

    positions = torch.randn(nb, nloc, 3, device="cuda", requires_grad=True)
    encoder = nn.Linear(3, num_in, bias=False).cuda()
    dist.broadcast(encoder.weight.data, src=0, group=ep_group)

    hidden = encoder(positions)
    out = moe(hidden, type_emb, atom_types)
    energy = out.sum(dim=-1).sum(dim=-1)

    force = -torch.autograd.grad(
        energy.sum(), positions, create_graph=True, retain_graph=True
    )[0]
    assert force.shape == positions.shape, f"Force shape mismatch: {force.shape}"
    assert not torch.isnan(force).any(), "Force has NaN"

    force_target = torch.randn_like(force)
    force_loss = nn.functional.mse_loss(force, force_target)
    force_loss.backward()

    has_grad = 0
    for name, p in moe.named_parameters():
        if p.grad is not None and p.grad.abs().max() > 0:
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
            has_grad += 1

    log(rank, f"  [GPU-lvl-2nd-order] E={num_experts} params_with_grad={has_grad} force_loss={force_loss.item():.4f}")
    assert has_grad > 0, "No parameter received gradient through 2nd-order path"

    log(rank, "[PASS] test_gpu_level_2nd_order_grad")


def bench_moelayer(rank, ep_group, num_experts, dim, nb, nloc, ntypes,
                   warmup=10, steps=50):
    """Benchmark MoELayer: EP (all ranks) then local (rank 0 only).

    Returns (ep_throughput, local_throughput, speedup) on rank 0,
    (ep_throughput, None, None) on other ranks.
    """
    from deepmd.pt.model.network.moe import MoELayer

    top_k = min(2, num_experts)
    ep_size = dist.get_world_size(group=ep_group)

    # --- Phase 1: EP benchmark (all ranks) ---
    torch.manual_seed(42)
    moe_ep = MoELayer(
        dim, dim, num_experts, top_k, dim,
        ep_group=ep_group, seed=42,
    ).cuda()
    dist.broadcast(moe_ep.gate.matrix.data, src=0, group=ep_group)

    encoder_ep = nn.Linear(3, dim, bias=False).cuda()
    dist.broadcast(encoder_ep.weight.data, src=0, group=ep_group)
    opt_ep = torch.optim.AdamW(
        list(moe_ep.parameters()) + list(encoder_ep.parameters()), lr=1e-3
    )

    type_emb = torch.randn(ntypes, dim, device="cuda")
    atom_types = torch.randint(0, ntypes, (nb, nloc), device="cuda")
    dist.broadcast(type_emb, src=0, group=ep_group)
    dist.broadcast(atom_types, src=0, group=ep_group)

    def run_step_ep():
        pos = torch.randn(nb, nloc, 3, device="cuda", requires_grad=True)
        hidden = encoder_ep(pos)
        out = moe_ep(hidden, type_emb, atom_types)
        energy = out.sum()
        force = -torch.autograd.grad(
            energy, pos, create_graph=True, retain_graph=True
        )[0]
        loss = energy + force.sum()
        opt_ep.zero_grad()
        loss.backward()
        # Sync shared parameter gradients across EP ranks
        for name, p in moe_ep.named_parameters():
            if p.grad is not None and "experts" not in name:
                dist.all_reduce(p.grad, group=ep_group)
                p.grad /= ep_size
        for p in encoder_ep.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, group=ep_group)
                p.grad /= ep_size
        opt_ep.step()

    for _ in range(warmup):
        run_step_ep()
    torch.cuda.synchronize()
    dist.barrier(group=ep_group)
    t0 = time.perf_counter()
    for _ in range(steps):
        run_step_ep()
    torch.cuda.synchronize()
    ep_tp = steps / (time.perf_counter() - t0)

    del moe_ep, encoder_ep, opt_ep
    torch.cuda.empty_cache()
    dist.barrier(group=ep_group)

    # --- Phase 2: Local benchmark (rank 0 only) ---
    local_tp = None
    speedup = None
    if rank == 0:
        torch.manual_seed(42)
        moe_local = MoELayer(
            dim, dim, num_experts, top_k, dim,
            ep_group=None, seed=42,
        ).cuda()

        encoder_local = nn.Linear(3, dim, bias=False).cuda()
        opt_local = torch.optim.AdamW(
            list(moe_local.parameters()) + list(encoder_local.parameters()),
            lr=1e-3,
        )

        type_emb_l = torch.randn(ntypes, dim, device="cuda")
        atom_types_l = torch.randint(0, ntypes, (nb, nloc), device="cuda")

        def run_step_local():
            pos = torch.randn(nb, nloc, 3, device="cuda", requires_grad=True)
            hidden = encoder_local(pos)
            out = moe_local(hidden, type_emb_l, atom_types_l)
            energy = out.sum()
            force = -torch.autograd.grad(
                energy, pos, create_graph=True, retain_graph=True
            )[0]
            loss = energy + force.sum()
            opt_local.zero_grad()
            loss.backward()
            opt_local.step()

        for _ in range(warmup):
            run_step_local()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            run_step_local()
        torch.cuda.synchronize()
        local_tp = steps / (time.perf_counter() - t0)
        speedup = ep_tp / local_tp if local_tp > 0 else 0.0

        del moe_local, encoder_local, opt_local
        torch.cuda.empty_cache()

    dist.barrier(group=ep_group)
    return ep_tp, local_tp, speedup


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="Run benchmarks")
    parser.add_argument("--skip-tests", action="store_true", help="Skip correctness tests")
    args = parser.parse_args()

    rank = setup()
    mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("dp", "ep"))
    ep_group = mesh["ep"].get_group()

    log(rank, "=" * 70)
    log(rank, "MoE Expert Parallelism — DPA3 MoELayer Tests (2 GPUs, EP=2)")
    log(rank, "=" * 70)

    # --- Correctness Tests ---
    if not args.skip_tests:
        log(rank, "\n--- Correctness Tests ---")

        test_all_to_all_double_backward(rank, ep_group)

        for n_exp in [2, 4, 6]:
            log(rank, f"\n  num_experts={n_exp}")
            test_ep_vs_local_correctness(rank, ep_group, num_experts=n_exp)
            test_ep_2nd_order_grad(rank, ep_group, num_experts=n_exp)

        test_gradient_stability(rank, ep_group, num_experts=4, steps=20)

        # GPU-level A2A tests
        log(rank, "\n--- GPU-level A2A Tests ---")
        for n_exp in [2, 4, 6]:
            log(rank, f"\n  num_experts={n_exp}")
            test_gpu_level_vs_expert_level(rank, ep_group, num_experts=n_exp)
            test_gpu_level_2nd_order_grad(rank, ep_group, num_experts=n_exp)

        log(rank, "\n" + "=" * 70)
        log(rank, "[PASS] All correctness tests passed")
        log(rank, "=" * 70)

    if args.bench:
        log(rank, "\n--- Performance Benchmarks ---")

        configs = [
            # (num_experts, dim, nb, nloc, ntypes)
            # --- Small model (dim=32) ---
            (2, 32, 4, 64, 4),
            (4, 32, 4, 64, 4),
            (6, 32, 4, 64, 6),
            (8, 32, 4, 64, 8),
            # --- Medium model (dim=64) ---
            (2, 64, 4, 64, 4),
            (4, 64, 4, 64, 4),
            (6, 64, 4, 64, 6),
            (8, 64, 4, 64, 8),
            # --- Larger model (dim=128) ---
            (2, 128, 4, 64, 4),
            (4, 128, 4, 64, 4),
            (6, 128, 4, 64, 6),
            (8, 128, 4, 64, 8),
            # --- Large batch (dim=128, more tokens) ---
            (4, 128, 8, 128, 4),
            (8, 128, 8, 128, 8),
            # --- Larger dim (dim=256) ---
            (4, 256, 4, 64, 4),
            (8, 256, 4, 64, 8),
            (4, 256, 8, 128, 4),
            (8, 256, 8, 128, 8),
        ]

        log(rank, f"\n{'E':>3} {'dim':>5} {'nb':>3} {'nloc':>5} {'nt':>3} {'EP(s/s)':>9} {'Local(s/s)':>10} {'Speedup':>8}")
        log(rank, "-" * 60)
        for n_exp, dim, nb, nloc, nt in configs:
            torch.cuda.empty_cache()
            dist.barrier(group=ep_group)
            ep_tp, local_tp, speedup = bench_moelayer(
                rank, ep_group, n_exp, dim, nb, nloc, nt
            )
            if rank == 0:
                print(
                    f"{n_exp:>3} {dim:>5} {nb:>3} {nloc:>5} {nt:>3} "
                    f"{ep_tp:>9.1f} {local_tp:>10.1f} {speedup:>7.2f}x",
                    flush=True,
                )

        log(rank, "\n" + "=" * 70)
        log(rank, "Benchmarks complete")
        log(rank, "=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
