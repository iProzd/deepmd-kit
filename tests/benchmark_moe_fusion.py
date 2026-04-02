#!/usr/bin/env python3
"""Comprehensive DPA3 MoE benchmark: 6-layer model, 4 configurations, A2A profiling.

Launch: torchrun --nproc_per_node=2 tests/benchmark_moe_fusion.py

Configurations tested:
  A. Single expert, single GPU    (baseline, no MoE)
  B. Multi-expert, single GPU     (MoE, no EP)
  C. Multi-expert, 2-GPU EP       (MoE + EP, unfused)
  D. Multi-expert, 2-GPU EP fused (MoE + EP + fusion)

Expert counts: [1, 2, 4, 8]
Steps: 100 (+ warmup)
"""

import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


# ---------------------------------------------------------------------------
# A2A profiling infrastructure — non-blocking event-based
# ---------------------------------------------------------------------------
_a2a_events: list[tuple] = []   # (start_event, end_event)
_a2a_enabled = False
_original_a2a_fn = None


def _profiled_a2a(x, send_splits, recv_splits, group):
    """Wrapper that records CUDA events around A2A (non-blocking)."""
    if _a2a_enabled:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = _original_a2a_fn(x, send_splits, recv_splits, group)
        end.record()
        _a2a_events.append((start, end))
        return result
    return _original_a2a_fn(x, send_splits, recv_splits, group)


def install_a2a_profiler():
    """Monkey-patch all_to_all_differentiable to record timing."""
    global _original_a2a_fn
    import deepmd.pt.model.network.moe_ep_ops as ops
    import deepmd.pt.model.network.moe as moe_mod
    _original_a2a_fn = ops.all_to_all_differentiable
    ops.all_to_all_differentiable = _profiled_a2a
    moe_mod.all_to_all_differentiable = _profiled_a2a


def enable_a2a_profiling():
    global _a2a_enabled
    _a2a_enabled = True
    _a2a_events.clear()


def disable_a2a_profiling():
    global _a2a_enabled
    _a2a_enabled = False


def get_a2a_stats():
    """Sync events and return (total_ms, count, avg_ms)."""
    if not _a2a_events:
        return 0.0, 0, 0.0
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in _a2a_events]
    total = sum(times)
    return total, len(times), total / len(times)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup():
    dist.init_process_group("nccl")
    rank = int(os.environ.get("RANK", dist.get_rank()))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, local_rank


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def make_water_system(n_molecules, e_rcut, e_sel, device="cuda"):
    """Create a water box with n_molecules H2O, properly extending + building nlist."""
    from deepmd.pt.utils.nlist import build_neighbor_list, extend_coord_with_ghosts

    coords = []
    atypes = []
    n_per_side = int(np.ceil(n_molecules ** (1.0 / 3.0)))
    spacing = 3.0
    count = 0
    for ix in range(n_per_side):
        for iy in range(n_per_side):
            for iz in range(n_per_side):
                if count >= n_molecules:
                    break
                base = np.array([ix * spacing, iy * spacing, iz * spacing])
                coords.append(base)
                coords.append(base + [0.0, 0.96, 0.0])
                coords.append(base + [0.0, 0.0, 0.96])
                atypes.extend([0, 1, 1])
                count += 1
            if count >= n_molecules:
                break
        if count >= n_molecules:
            break

    nloc = len(coords)
    box_size = n_per_side * spacing + e_rcut + 1.0
    coord_np = np.array(coords, dtype=np.float64)
    coord = torch.tensor(coord_np, dtype=torch.float64, device=device).unsqueeze(0)
    atype = torch.tensor(atypes, dtype=torch.long, device=device).unsqueeze(0)
    cell = torch.zeros(1, 9, dtype=torch.float64, device=device)
    cell[0, 0] = box_size
    cell[0, 4] = box_size
    cell[0, 8] = box_size

    coord_flat = coord.reshape(1, nloc * 3)
    ext_coord, ext_atype, mapping = extend_coord_with_ghosts(
        coord_flat, atype, cell, e_rcut
    )
    nlist = build_neighbor_list(
        ext_coord, ext_atype, nloc, e_rcut, e_sel, distinguish_types=False,
    )
    return ext_coord, ext_atype, nlist, mapping, nloc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_training(
    model, ext_coord, ext_atype, nlist, mapping,
    nsteps, warmup, ep_group, ep_size, rank,
    profile_a2a=False,
):
    """Run forward+backward training loop and return timing info."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    nall = ext_atype.shape[1]

    losses_list = []
    t0 = None

    for step in range(nsteps + warmup):
        if step == warmup:
            torch.cuda.synchronize()
            if ep_group is not None:
                dist.barrier(group=ep_group)
            if profile_a2a:
                enable_a2a_profiling()
            t0 = time.perf_counter()

        optimizer.zero_grad()
        noise = torch.randn(1, nall * 3, device="cuda", dtype=ext_coord.dtype) * 0.01
        coord_noisy = (ext_coord + noise).detach().requires_grad_(True)

        result = model(coord_noisy, ext_atype, nlist=nlist, mapping=mapping)
        desc = result[0]
        energy = desc.sum()
        force = -torch.autograd.grad(
            energy, coord_noisy, create_graph=True, retain_graph=True
        )[0]
        loss = energy + force.sum() * 0.01
        loss.backward()

        if ep_group is not None:
            for name, p in model.named_parameters():
                if "experts" not in name:
                    # Always all_reduce non-expert params, treating None grads
                    # as zeros. With create_graph=True, different ranks can have
                    # different grad patterns, so we must be symmetric.
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    dist.all_reduce(p.grad, group=ep_group)
                    p.grad /= ep_size

        optimizer.step()

        if step >= warmup:
            losses_list.append(loss.item())

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    a2a_total_ms, a2a_count, a2a_avg_ms = 0.0, 0, 0.0
    if profile_a2a:
        disable_a2a_profiling()
        a2a_total_ms, a2a_count, a2a_avg_ms = get_a2a_stats()

    has_nan = any(np.isnan(l) or np.isinf(l) for l in losses_list)

    return {
        "elapsed": elapsed,
        "steps_per_sec": nsteps / elapsed,
        "loss_first": losses_list[0] if losses_list else float("nan"),
        "loss_last": losses_list[-1] if losses_list else float("nan"),
        "has_nan": has_nan,
        "a2a_total_ms": a2a_total_ms,
        "a2a_count": a2a_count,
        "a2a_avg_ms": a2a_avg_ms,
        "a2a_pct": (a2a_total_ms / (elapsed * 1000)) * 100 if elapsed > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Model creation helper
# ---------------------------------------------------------------------------
def make_model(n_experts, fuse, ep_group, nlayers, n_dim, e_dim, a_dim,
               e_rcut, e_sel, a_rcut, a_sel, moe_ep_size):
    from deepmd.pt.model.descriptor.dpa3 import DescrptDPA3
    use_moe = n_experts > 1
    return DescrptDPA3(
        repflow={
            "n_dim": n_dim, "e_dim": e_dim, "a_dim": a_dim,
            "nlayers": nlayers,
            "e_rcut": e_rcut, "e_rcut_smth": e_rcut - 1.0, "e_sel": e_sel,
            "a_rcut": a_rcut, "a_rcut_smth": a_rcut - 0.5, "a_sel": a_sel,
            "axis_neuron": 4,
            "update_angle": True, "smooth_edge_update": True,
            "update_style": "res_residual", "update_residual": 0.1,
            "update_residual_init": "const",
            "n_experts": n_experts, "moe_top_k": min(2, n_experts),
            "use_node_moe": use_moe, "use_edge_moe": use_moe,
            "use_angle_moe": use_moe,
            "share_expert": 0,
            "fuse_moe_mlps": fuse,
            "moe_ep_size": moe_ep_size,
        },
        ntypes=2, precision="float32", seed=1,
        ep_group=ep_group,
    ).cuda()


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main():
    rank, local_rank = setup()
    mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("dp", "ep"))
    ep_group = mesh["ep"].get_group()
    ep_size = dist.get_world_size(group=ep_group)

    install_a2a_profiler()

    # Benchmark parameters
    NLAYERS = 6
    N_DIM = 128
    E_DIM = 32
    A_DIM = 16
    E_RCUT = 6.0
    E_SEL = 40
    A_RCUT = 4.0
    A_SEL = 16
    NSTEPS = 100
    WARMUP = 10
    N_MOLECULES = 8
    EXPERT_COUNTS = [8]

    log(rank, "=" * 80)
    log(rank, "DPA3 MoE Comprehensive Benchmark")
    log(rank, "=" * 80)
    log(rank, f"  Model: {NLAYERS} layers, n_dim={N_DIM}, e_dim={E_DIM}, a_dim={A_DIM}")
    log(rank, f"  System: {N_MOLECULES} H2O ({N_MOLECULES * 3} atoms)")
    log(rank, f"  e_rcut={E_RCUT}, e_sel={E_SEL}, a_rcut={A_RCUT}, a_sel={A_SEL}")
    log(rank, f"  Steps: {NSTEPS} (+ {WARMUP} warmup), GPUs: 2x NVIDIA H20")
    log(rank, f"  Expert counts: {EXPERT_COUNTS}")
    log(rank, "=" * 80)

    ext_coord, ext_atype, nlist, mapping, nloc = make_water_system(
        N_MOLECULES, E_RCUT, E_SEL
    )
    dist.broadcast(ext_coord, src=0, group=ep_group)
    dist.broadcast(ext_atype, src=0, group=ep_group)
    dist.broadcast(nlist, src=0, group=ep_group)
    dist.broadcast(mapping, src=0, group=ep_group)
    nall = ext_atype.shape[1]
    log(rank, f"\n  Data: nloc={nloc}, nall={nall}, nlist={list(nlist.shape)}")

    model_args = dict(
        nlayers=NLAYERS, n_dim=N_DIM, e_dim=E_DIM, a_dim=A_DIM,
        e_rcut=E_RCUT, e_sel=E_SEL, a_rcut=A_RCUT, a_sel=A_SEL,
    )
    all_results = {}

    for n_experts in EXPERT_COUNTS:
        log(rank, f"\n{'─' * 80}")
        log(rank, f"  n_experts = {n_experts}")
        log(rank, f"{'─' * 80}")

        configs = []
        if n_experts == 1:
            configs.append(("A_baseline", False, None, 1))
        if n_experts > 1:
            configs.append(("B_single_gpu", False, None, 1))
            configs.append(("C_ep_unfused", False, ep_group, 2))
            configs.append(("D_ep_fused", True, ep_group, 2))

        for config_name, fuse, grp, moe_ep in configs:
            label = {
                "A_baseline": "Single expert baseline",
                "B_single_gpu": f"{n_experts} experts, 1 GPU",
                "C_ep_unfused": f"{n_experts} experts, EP unfused",
                "D_ep_fused": f"{n_experts} experts, EP fused",
            }[config_name]

            log(rank, f"\n  [{config_name}] {label}...")

            torch.manual_seed(42)
            torch.cuda.manual_seed(42)

            model = make_model(
                n_experts=n_experts, fuse=fuse, ep_group=grp,
                moe_ep_size=moe_ep, **model_args,
            )

            # Sync params for EP configs
            if grp is not None:
                for _, p in model.named_parameters():
                    dist.broadcast(p.data, src=0, group=grp)

            n_params = sum(p.numel() for p in model.parameters())
            log(rank, f"      params={n_params}" +
                (" (per GPU)" if grp is not None else ""))

            use_profile = grp is not None  # only profile A2A for EP configs
            res = run_training(
                model, ext_coord, ext_atype, nlist, mapping,
                NSTEPS, WARMUP,
                ep_group=grp, ep_size=ep_size if grp else 1,
                rank=rank, profile_a2a=use_profile,
            )
            all_results[(config_name, n_experts)] = res

            log(rank, f"      {NSTEPS} steps in {res['elapsed']:.2f}s "
                f"({res['steps_per_sec']:.1f} steps/s), "
                f"loss: {res['loss_first']:.4f}->{res['loss_last']:.4f}")
            if res["a2a_count"] > 0:
                log(rank, f"      A2A: {res['a2a_count']} calls, "
                    f"total={res['a2a_total_ms']:.1f}ms, "
                    f"avg={res['a2a_avg_ms']:.3f}ms, "
                    f"fraction={res['a2a_pct']:.1f}%")

            del model
            torch.cuda.empty_cache()
            dist.barrier(group=ep_group)

    # ====================================================================
    # Summary
    # ====================================================================
    if rank == 0:
        baseline = all_results.get(("A_baseline", 1))
        baseline_time = baseline["elapsed"] if baseline else float("nan")

        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)
        if baseline:
            print(f"\nBaseline: 1 expert, 1 GPU, {NSTEPS} steps = {baseline_time:.2f}s "
                  f"({baseline['steps_per_sec']:.1f} steps/s)\n")
        else:
            print(f"\nBaseline: not available (1-expert not tested)\n")

        hdr = (f"{'Config':<28} {'Exp':>4} {'Time(s)':>8} {'steps/s':>8} "
               f"{'vs Base':>8} {'A2A_ms':>8} {'A2A%':>6} {'A2A#':>7}")
        print(hdr)
        print("-" * len(hdr))

        for key in sorted(all_results.keys()):
            cfg, ne = key
            r = all_results[key]
            sp = baseline_time / r["elapsed"] if r["elapsed"] > 0 else 0
            print(f"{cfg:<28} {ne:>4} {r['elapsed']:>8.2f} "
                  f"{r['steps_per_sec']:>8.1f} {sp:>7.2f}x "
                  f"{r['a2a_total_ms']:>8.1f} {r['a2a_pct']:>5.1f}% "
                  f"{r['a2a_count']:>7}")

        # Per-expert-count detail
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS BY EXPERT COUNT")
        print("=" * 80)

        for ne in EXPERT_COUNTS:
            print(f"\n--- n_experts = {ne} ---")
            entries = {k: v for k, v in all_results.items() if k[1] == ne}
            for (cfg, _), r in sorted(entries.items()):
                sp = baseline_time / r["elapsed"] if r["elapsed"] > 0 else 0
                a2a = ""
                if r["a2a_count"] > 0:
                    a2a = (f"  | A2A: {r['a2a_count']} calls, "
                           f"{r['a2a_total_ms']:.1f}ms ({r['a2a_pct']:.1f}%)")
                print(f"  {cfg:<26} {r['elapsed']:>7.2f}s  "
                      f"{r['steps_per_sec']:>6.1f} stp/s  "
                      f"({sp:.2f}x){a2a}")

            c_key = ("C_ep_unfused", ne)
            d_key = ("D_ep_fused", ne)
            if c_key in all_results and d_key in all_results:
                c = all_results[c_key]
                d = all_results[d_key]
                fsp = c["elapsed"] / d["elapsed"] if d["elapsed"] > 0 else 0
                a2a_red = ((1 - d["a2a_count"] / c["a2a_count"]) * 100
                           if c["a2a_count"] > 0 else 0)
                print(f"\n  Fusion effect:")
                print(f"    Speedup (fused/unfused):  {fsp:.2f}x")
                print(f"    A2A calls reduction:      {a2a_red:.0f}% "
                      f"({c['a2a_count']}->{d['a2a_count']})")
                print(f"    A2A time saved:           "
                      f"{c['a2a_total_ms']:.1f}ms->{d['a2a_total_ms']:.1f}ms "
                      f"(Δ{c['a2a_total_ms']-d['a2a_total_ms']:.1f}ms)")

        # EP efficiency analysis
        print("\n" + "=" * 80)
        print("EP EFFICIENCY ANALYSIS")
        print("=" * 80)
        print("\nCompare single-GPU MoE (B) vs 2-GPU EP (C/D):")
        print("  EP ideal speedup = 2.0x (experts split across 2 GPUs)")
        print(f"\n  {'Experts':>7} {'B(1GPU)':>9} {'C(EP)':>9} {'D(EPfuse)':>9} "
              f"{'C/B':>7} {'D/B':>7} {'EP_eff_C':>9} {'EP_eff_D':>9}")
        print(f"  {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*7} {'-'*7} {'-'*9} {'-'*9}")
        for ne in [2, 4, 8]:
            b = all_results.get(("B_single_gpu", ne))
            c = all_results.get(("C_ep_unfused", ne))
            d = all_results.get(("D_ep_fused", ne))
            if b and c and d:
                c_sp = b["elapsed"] / c["elapsed"] if c["elapsed"] > 0 else 0
                d_sp = b["elapsed"] / d["elapsed"] if d["elapsed"] > 0 else 0
                c_eff = c_sp / 2.0 * 100
                d_eff = d_sp / 2.0 * 100
                print(f"  {ne:>7} {b['elapsed']:>8.2f}s {c['elapsed']:>8.2f}s "
                      f"{d['elapsed']:>8.2f}s {c_sp:>6.2f}x {d_sp:>6.2f}x "
                      f"{c_eff:>8.1f}% {d_eff:>8.1f}%")

        # Save JSON
        json_results = {}
        for (cfg, ne), r in all_results.items():
            json_results[f"{cfg}_{ne}"] = {"config": cfg, "n_experts": ne, **r}
        results_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json"
        )
        with open(results_path, "w") as f:
            json.dump(json_results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
