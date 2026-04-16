# SPDX-License-Identifier: LGPL-3.0-or-later
"""Benchmark for MoEPacker pack/unpack + A2A at realistic scale.

Run:
    torchrun --nproc_per_node=4 source/tests/pt/benchmark_moe_packer.py
    torchrun --nproc_per_node=8 source/tests/pt/benchmark_moe_packer.py

Measures wall-clock time (with CUDA synchronization) for each stage of the
MoE dispatch/combine pipeline and compares against a raw A2A baseline to
isolate packing overhead.

Scenarios simulate DPA3-like token counts:
  - nodes  ~ nb * nloc  (atoms)
  - edges  ~ nb * nloc * sel  (neighbor pairs)
  - angles ~ nb * nloc * sel_a (angular triples)
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from deepmd.pt.model.network.moe_ep_ops import all_to_all_differentiable
from deepmd.pt.model.network.moe_packer import MoEPacker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

A_DIM = 16          # realistic base dimension
DTYPE = torch.float64
WARMUP = 5
REPEAT = 20


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_dist():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def cleanup_dist():
    dist.destroy_process_group()


def _compute_send_splits(node_counts, edge_counts, angle_counts):
    """Same logic as MoEPacker."""
    splits = []
    for g in range(len(node_counts)):
        n_node = node_counts[g]
        n_edge_rows = _ceildiv(edge_counts[g], 4) if edge_counts[g] > 0 else 0
        n_angle_rows = _ceildiv(angle_counts[g], 10) if angle_counts[g] > 0 else 0
        splits.append(n_node + n_edge_rows + n_angle_rows)
    return splits


def exchange_counts(send_counts, world_size, group, device):
    """All-to-All exchange of per-GPU token counts."""
    send_t = torch.tensor(send_counts, dtype=torch.int64, device=device)
    recv_t = torch.zeros_like(send_t)
    for field_idx in range(3):
        field_send = send_t[:, field_idx].contiguous()
        field_recv = torch.zeros(world_size, dtype=torch.int64, device=device)
        dist.all_to_all_single(
            field_recv, field_send,
            output_split_sizes=[1] * world_size,
            input_split_sizes=[1] * world_size,
            group=group,
        )
        recv_t[:, field_idx] = field_recv
    return recv_t.tolist()


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

class Timer:
    """CUDA-synchronized timer that accumulates measurements."""

    def __init__(self, device):
        self.device = device
        self.records: dict[str, list[float]] = {}

    def _sync(self):
        torch.cuda.synchronize(self.device)

    def measure(self, name: str):
        """Context manager that records elapsed time under *name*."""
        return _TimerCtx(self, name)

    def stats(self, name: str):
        vals = self.records.get(name, [])
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return mean, var ** 0.5


class _TimerCtx:
    def __init__(self, timer: Timer, name: str):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.timer._sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.timer._sync()
        elapsed = time.perf_counter() - self.t0
        self.timer.records.setdefault(self.name, []).append(elapsed)


# ---------------------------------------------------------------------------
# Benchmark: full packer pipeline
# ---------------------------------------------------------------------------

def bench_pipeline(
    rank, world_size, group, device, packer, timer,
    n_node, n_edge, n_angle, label,
):
    """Run WARMUP + REPEAT iterations of the full pack → A2A → unpack pipeline.

    Distribution: roughly uniform across GPUs (each rank sends ~1/ws to each dest).
    """
    # Distribute tokens uniformly.
    base_n = n_node // world_size
    base_e = n_edge // world_size
    base_a = n_angle // world_size
    node_counts = [base_n] * world_size
    edge_counts = [base_e] * world_size
    angle_counts = [base_a] * world_size
    node_counts[-1] += n_node - base_n * world_size
    edge_counts[-1] += n_edge - base_e * world_size
    angle_counts[-1] += n_angle - base_a * world_size

    total_node = sum(node_counts)
    total_edge = sum(edge_counts)
    total_angle = sum(angle_counts)

    # Generate data.
    torch.manual_seed(42 + rank)
    node_in = torch.randn(total_node, 28 * A_DIM, dtype=DTYPE, device=device)
    edge_in = torch.randn(total_edge, 10 * A_DIM, dtype=DTYPE, device=device)
    angle_in = torch.randn(total_angle, 4 * A_DIM, dtype=DTYPE, device=device)

    # Pre-exchange metadata.
    send_info = [[node_counts[g], edge_counts[g], angle_counts[g]]
                 for g in range(world_size)]
    recv_info = exchange_counts(send_info, world_size, group, device)
    recv_nc = [recv_info[g][0] for g in range(world_size)]
    recv_ec = [recv_info[g][1] for g in range(world_size)]
    recv_ac = [recv_info[g][2] for g in range(world_size)]
    send_splits = _compute_send_splits(node_counts, edge_counts, angle_counts)
    recv_splits = _compute_send_splits(recv_nc, recv_ec, recv_ac)

    # Also prepare a baseline raw tensor with the same total row count
    # to measure pure A2A cost.
    total_send_rows = sum(send_splits)
    total_recv_rows = sum(recv_splits)
    raw_tensor = torch.randn(total_send_rows, 40 * A_DIM, dtype=DTYPE, device=device)

    timer.records.clear()

    for i in range(WARMUP + REPEAT):
        recording = i >= WARMUP
        dist.barrier()

        # --- Full pipeline ---
        if recording:
            with timer.measure("total"):
                with timer.measure("pack_dispatch"):
                    packed, _ = packer.pack_for_dispatch(
                        node_in, edge_in, angle_in,
                        node_counts, edge_counts, angle_counts,
                    )
                with timer.measure("a2a_dispatch"):
                    recv_tensor = all_to_all_differentiable(
                        packed, send_splits, recv_splits, group,
                    )
                with timer.measure("unpack_dispatch"):
                    node_r, edge_r, angle_r = packer.unpack_from_dispatch(
                        recv_tensor, recv_nc, recv_ec, recv_ac,
                    )
                # Simulated expert (truncate).
                node_o = node_r[:, :8 * A_DIM]
                edge_o = edge_r[:, :6 * A_DIM]
                angle_o = angle_r[:, :3 * A_DIM]
                with timer.measure("pack_combine"):
                    packed_o = packer.pack_for_combine(
                        node_o, edge_o, angle_o,
                        recv_nc, recv_ec, recv_ac,
                    )
                with timer.measure("a2a_combine"):
                    returned = all_to_all_differentiable(
                        packed_o, recv_splits, send_splits, group,
                    )
                with timer.measure("unpack_combine"):
                    packer.unpack_from_combine(
                        returned, node_counts, edge_counts, angle_counts,
                    )
        else:
            # Warmup (no recording).
            packed, _ = packer.pack_for_dispatch(
                node_in, edge_in, angle_in,
                node_counts, edge_counts, angle_counts,
            )
            recv_tensor = all_to_all_differentiable(packed, send_splits, recv_splits, group)
            node_r, edge_r, angle_r = packer.unpack_from_dispatch(
                recv_tensor, recv_nc, recv_ec, recv_ac,
            )
            node_o = node_r[:, :8 * A_DIM]
            edge_o = edge_r[:, :6 * A_DIM]
            angle_o = angle_r[:, :3 * A_DIM]
            packed_o = packer.pack_for_combine(
                node_o, edge_o, angle_o, recv_nc, recv_ec, recv_ac,
            )
            returned = all_to_all_differentiable(packed_o, recv_splits, send_splits, group)
            packer.unpack_from_combine(returned, node_counts, edge_counts, angle_counts)

        # --- Raw A2A baseline (same data volume, no packing) ---
        dist.barrier()
        if recording:
            with timer.measure("raw_a2a_dispatch"):
                _ = all_to_all_differentiable(raw_tensor, send_splits, recv_splits, group)
        else:
            _ = all_to_all_differentiable(raw_tensor, send_splits, recv_splits, group)

    # Report from rank 0.
    if rank == 0:
        total_tokens = n_node + n_edge + n_angle
        packed_rows = sum(send_splits)
        mem_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        print(f"\n{'=' * 78}")
        print(f"  {label}")
        print(f"  Per-rank tokens: node={n_node:,} edge={n_edge:,} angle={n_angle:,}"
              f"  total={total_tokens:,}")
        print(f"  Packed rows/rank (dispatch): {packed_rows:,}"
              f"  ({40 * A_DIM} cols = {40 * A_DIM * 8 / 1024:.0f} KB/row f64)")
        print(f"  Data volume/rank (dispatch): "
              f"{packed_rows * 40 * A_DIM * 8 / 1024 / 1024:.1f} MB")
        print(f"  GPUs: {world_size}  a_dim: {A_DIM}  dtype: {DTYPE}"
              f"  warmup: {WARMUP}  repeat: {REPEAT}")
        print(f"  Peak GPU memory: {mem_alloc:.0f} MB")
        print(f"{'=' * 78}")

        ops = [
            "pack_dispatch", "a2a_dispatch", "unpack_dispatch",
            "pack_combine", "a2a_combine", "unpack_combine",
        ]

        total_mean, total_std = timer.stats("total")
        packing_mean = 0.0

        print(f"  {'Operation':<25} {'Mean':>10} {'Std':>10} {'%Total':>8}")
        print(f"  {'-' * 55}")
        for op in ops:
            m, s = timer.stats(op)
            pct = m / total_mean * 100 if total_mean > 0 else 0
            print(f"  {op:<25} {m*1000:>9.3f}ms {s*1000:>9.3f}ms {pct:>7.1f}%")
            if "a2a" not in op:
                packing_mean += m
        print(f"  {'-' * 55}")
        print(f"  {'TOTAL':<25} {total_mean*1000:>9.3f}ms {total_std*1000:>9.3f}ms"
              f" {100.0:>7.1f}%")

        raw_m, raw_s = timer.stats("raw_a2a_dispatch")
        a2a_m = sum(timer.stats(k)[0] for k in ["a2a_dispatch", "a2a_combine"])

        print()
        print(f"  {'--- Summary ---':^55}")
        print(f"  Packing overhead (4 ops):   {packing_mean*1000:>9.3f}ms"
              f"  ({packing_mean/total_mean*100:.1f}% of total)")
        print(f"  A2A communication (2 ops):  {a2a_m*1000:>9.3f}ms"
              f"  ({a2a_m/total_mean*100:.1f}% of total)")
        print(f"  Raw A2A baseline (1 trip):  {raw_m*1000:>9.3f}ms"
              f"  (vs dispatch A2A: {timer.stats('a2a_dispatch')[0]*1000:.3f}ms)")


# ---------------------------------------------------------------------------
# Benchmark: with backward (gradient)
# ---------------------------------------------------------------------------

def bench_pipeline_backward(
    rank, world_size, group, device, packer, timer,
    n_node, n_edge, n_angle, label,
):
    """Measure forward + backward through the full pipeline."""
    base_n = n_node // world_size
    base_e = n_edge // world_size
    base_a = n_angle // world_size
    node_counts = [base_n] * world_size
    edge_counts = [base_e] * world_size
    angle_counts = [base_a] * world_size
    node_counts[-1] += n_node - base_n * world_size
    edge_counts[-1] += n_edge - base_e * world_size
    angle_counts[-1] += n_angle - base_a * world_size

    total_node = sum(node_counts)
    total_edge = sum(edge_counts)
    total_angle = sum(angle_counts)

    send_info = [[node_counts[g], edge_counts[g], angle_counts[g]]
                 for g in range(world_size)]
    recv_info = exchange_counts(send_info, world_size, group, device)
    recv_nc = [recv_info[g][0] for g in range(world_size)]
    recv_ec = [recv_info[g][1] for g in range(world_size)]
    recv_ac = [recv_info[g][2] for g in range(world_size)]
    send_splits = _compute_send_splits(node_counts, edge_counts, angle_counts)
    recv_splits = _compute_send_splits(recv_nc, recv_ec, recv_ac)

    timer.records.clear()

    for i in range(WARMUP + REPEAT):
        recording = i >= WARMUP

        # Fresh tensors each iteration (grad accumulation).
        torch.manual_seed(42 + rank + i * 100)
        node_in = torch.randn(total_node, 28 * A_DIM, dtype=DTYPE, device=device,
                              requires_grad=True)
        edge_in = torch.randn(total_edge, 10 * A_DIM, dtype=DTYPE, device=device,
                              requires_grad=True)
        angle_in = torch.randn(total_angle, 4 * A_DIM, dtype=DTYPE, device=device,
                               requires_grad=True)

        dist.barrier()

        if recording:
            with timer.measure("fwd+bwd_total"):
                # Forward.
                with timer.measure("fwd_total"):
                    packed, _ = packer.pack_for_dispatch(
                        node_in, edge_in, angle_in,
                        node_counts, edge_counts, angle_counts,
                    )
                    recv_tensor = all_to_all_differentiable(
                        packed, send_splits, recv_splits, group,
                    )
                    node_r, edge_r, angle_r = packer.unpack_from_dispatch(
                        recv_tensor, recv_nc, recv_ec, recv_ac,
                    )
                    # Non-trivial expert: scale by 2.
                    node_o = node_r[:, :8 * A_DIM] * 2.0
                    edge_o = edge_r[:, :6 * A_DIM] * 2.0
                    angle_o = angle_r[:, :3 * A_DIM] * 2.0
                    packed_o = packer.pack_for_combine(
                        node_o, edge_o, angle_o,
                        recv_nc, recv_ec, recv_ac,
                    )
                    returned = all_to_all_differentiable(
                        packed_o, recv_splits, send_splits, group,
                    )
                    nf, ef, af = packer.unpack_from_combine(
                        returned, node_counts, edge_counts, angle_counts,
                    )
                    loss = nf.sum() + ef.sum() + af.sum()

                # Backward.
                with timer.measure("bwd_total"):
                    loss.backward()
        else:
            packed, _ = packer.pack_for_dispatch(
                node_in, edge_in, angle_in,
                node_counts, edge_counts, angle_counts,
            )
            recv_tensor = all_to_all_differentiable(
                packed, send_splits, recv_splits, group,
            )
            node_r, edge_r, angle_r = packer.unpack_from_dispatch(
                recv_tensor, recv_nc, recv_ec, recv_ac,
            )
            node_o = node_r[:, :8 * A_DIM] * 2.0
            edge_o = edge_r[:, :6 * A_DIM] * 2.0
            angle_o = angle_r[:, :3 * A_DIM] * 2.0
            packed_o = packer.pack_for_combine(
                node_o, edge_o, angle_o, recv_nc, recv_ec, recv_ac,
            )
            returned = all_to_all_differentiable(
                packed_o, recv_splits, send_splits, group,
            )
            nf, ef, af = packer.unpack_from_combine(
                returned, node_counts, edge_counts, angle_counts,
            )
            loss = nf.sum() + ef.sum() + af.sum()
            loss.backward()

    if rank == 0:
        fwd_m, fwd_s = timer.stats("fwd_total")
        bwd_m, bwd_s = timer.stats("bwd_total")
        tot_m, tot_s = timer.stats("fwd+bwd_total")

        print(f"\n{'=' * 78}")
        print(f"  {label}")
        print(f"  Per-rank: node={n_node:,} edge={n_edge:,} angle={n_angle:,}")
        print(f"  GPUs: {world_size}  a_dim: {A_DIM}")
        print(f"{'=' * 78}")
        print(f"  {'Forward':<25} {fwd_m*1000:>9.3f}ms {fwd_s*1000:>9.3f}ms"
              f"  ({fwd_m/tot_m*100:.1f}%)")
        print(f"  {'Backward':<25} {bwd_m*1000:>9.3f}ms {bwd_s*1000:>9.3f}ms"
              f"  ({bwd_m/tot_m*100:.1f}%)")
        print(f"  {'Total (fwd+bwd)':<25} {tot_m*1000:>9.3f}ms {tot_s*1000:>9.3f}ms")


# ---------------------------------------------------------------------------
# Benchmark: skewed distribution
# ---------------------------------------------------------------------------

def bench_pipeline_skewed(
    rank, world_size, group, device, packer, timer,
    n_node, n_edge, n_angle, label,
):
    """Skewed routing: 60% of tokens go to GPU 0, rest uniform."""
    heavy_frac = 0.6
    light_frac = (1.0 - heavy_frac) / max(1, world_size - 1)

    node_counts = [0] * world_size
    edge_counts = [0] * world_size
    angle_counts = [0] * world_size

    node_counts[0] = int(n_node * heavy_frac)
    edge_counts[0] = int(n_edge * heavy_frac)
    angle_counts[0] = int(n_angle * heavy_frac)

    for g in range(1, world_size):
        node_counts[g] = int(n_node * light_frac)
        edge_counts[g] = int(n_edge * light_frac)
        angle_counts[g] = int(n_angle * light_frac)

    # Fix rounding.
    node_counts[-1] += n_node - sum(node_counts)
    edge_counts[-1] += n_edge - sum(edge_counts)
    angle_counts[-1] += n_angle - sum(angle_counts)

    total_node = sum(node_counts)
    total_edge = sum(edge_counts)
    total_angle = sum(angle_counts)

    torch.manual_seed(42 + rank)
    node_in = torch.randn(total_node, 28 * A_DIM, dtype=DTYPE, device=device)
    edge_in = torch.randn(total_edge, 10 * A_DIM, dtype=DTYPE, device=device)
    angle_in = torch.randn(total_angle, 4 * A_DIM, dtype=DTYPE, device=device)

    send_info = [[node_counts[g], edge_counts[g], angle_counts[g]]
                 for g in range(world_size)]
    recv_info = exchange_counts(send_info, world_size, group, device)
    recv_nc = [recv_info[g][0] for g in range(world_size)]
    recv_ec = [recv_info[g][1] for g in range(world_size)]
    recv_ac = [recv_info[g][2] for g in range(world_size)]
    send_splits = _compute_send_splits(node_counts, edge_counts, angle_counts)
    recv_splits = _compute_send_splits(recv_nc, recv_ec, recv_ac)

    timer.records.clear()

    for i in range(WARMUP + REPEAT):
        recording = i >= WARMUP
        dist.barrier()

        if recording:
            with timer.measure("total"):
                with timer.measure("pack_dispatch"):
                    packed, _ = packer.pack_for_dispatch(
                        node_in, edge_in, angle_in,
                        node_counts, edge_counts, angle_counts,
                    )
                with timer.measure("a2a_dispatch"):
                    recv_tensor = all_to_all_differentiable(
                        packed, send_splits, recv_splits, group,
                    )
                with timer.measure("unpack_dispatch"):
                    node_r, edge_r, angle_r = packer.unpack_from_dispatch(
                        recv_tensor, recv_nc, recv_ec, recv_ac,
                    )
                node_o = node_r[:, :8 * A_DIM]
                edge_o = edge_r[:, :6 * A_DIM]
                angle_o = angle_r[:, :3 * A_DIM]
                with timer.measure("pack_combine"):
                    packed_o = packer.pack_for_combine(
                        node_o, edge_o, angle_o,
                        recv_nc, recv_ec, recv_ac,
                    )
                with timer.measure("a2a_combine"):
                    returned = all_to_all_differentiable(
                        packed_o, recv_splits, send_splits, group,
                    )
                with timer.measure("unpack_combine"):
                    packer.unpack_from_combine(
                        returned, node_counts, edge_counts, angle_counts,
                    )
        else:
            packed, _ = packer.pack_for_dispatch(
                node_in, edge_in, angle_in,
                node_counts, edge_counts, angle_counts,
            )
            recv_tensor = all_to_all_differentiable(packed, send_splits, recv_splits, group)
            node_r, edge_r, angle_r = packer.unpack_from_dispatch(
                recv_tensor, recv_nc, recv_ec, recv_ac,
            )
            node_o = node_r[:, :8 * A_DIM]
            edge_o = edge_r[:, :6 * A_DIM]
            angle_o = angle_r[:, :3 * A_DIM]
            packed_o = packer.pack_for_combine(
                node_o, edge_o, angle_o, recv_nc, recv_ec, recv_ac,
            )
            returned = all_to_all_differentiable(packed_o, recv_splits, send_splits, group)
            packer.unpack_from_combine(returned, node_counts, edge_counts, angle_counts)

    if rank == 0:
        total_mean, total_std = timer.stats("total")
        packing_mean = 0.0

        print(f"\n{'=' * 78}")
        print(f"  {label}")
        print(f"  Per-rank: node={n_node:,} edge={n_edge:,} angle={n_angle:,}")
        print(f"  Distribution: 60% to GPU0, rest uniform")
        print(f"  Actual send splits (rank 0): {send_splits}")
        print(f"  GPUs: {world_size}")
        print(f"{'=' * 78}")
        print(f"  {'Operation':<25} {'Mean':>10} {'Std':>10} {'%Total':>8}")
        print(f"  {'-' * 55}")
        for op in ["pack_dispatch", "a2a_dispatch", "unpack_dispatch",
                    "pack_combine", "a2a_combine", "unpack_combine"]:
            m, s = timer.stats(op)
            pct = m / total_mean * 100 if total_mean > 0 else 0
            print(f"  {op:<25} {m*1000:>9.3f}ms {s*1000:>9.3f}ms {pct:>7.1f}%")
            if "a2a" not in op:
                packing_mean += m
        print(f"  {'-' * 55}")
        print(f"  {'TOTAL':<25} {total_mean*1000:>9.3f}ms {total_std*1000:>9.3f}ms")
        print(f"  Packing overhead: {packing_mean*1000:.3f}ms"
              f"  ({packing_mean/total_mean*100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rank, world_size = setup_dist()
    device = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD
    packer = MoEPacker(A_DIM)
    timer = Timer(device)

    if rank == 0:
        print(f"{'#' * 78}")
        print(f"#  MoEPacker Benchmark — {world_size} GPUs, a_dim={A_DIM}, dtype={DTYPE}")
        print(f"#  warmup={WARMUP}, repeat={REPEAT}")
        print(f"{'#' * 78}")

    # -----------------------------------------------------------------------
    # Part 1: Forward-only pipeline at different scales.
    # -----------------------------------------------------------------------
    # Scenarios: (n_node, n_edge, n_angle, label)
    # Simulate DPA3-like ratios: edges ~ 4x nodes, angles ~ 2x nodes.
    scenarios = [
        (1_000, 4_000, 2_000, "[FWD-S]  Small: 7k tokens/rank"),
        (10_000, 40_000, 20_000, "[FWD-M]  Medium: 70k tokens/rank"),
        (50_000, 200_000, 100_000, "[FWD-L]  Large: 350k tokens/rank"),
        (100_000, 400_000, 200_000, "[FWD-XL] XL: 700k tokens/rank"),
    ]

    for n_node, n_edge, n_angle, label in scenarios:
        torch.cuda.reset_peak_memory_stats(device)
        try:
            bench_pipeline(rank, world_size, group, device, packer, timer,
                           n_node, n_edge, n_angle, label)
        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"\n  {label}: OOM — skipping")
        dist.barrier()

    # -----------------------------------------------------------------------
    # Part 2: Forward + backward at medium scale.
    # -----------------------------------------------------------------------
    if rank == 0:
        print(f"\n\n{'#' * 78}")
        print(f"#  Part 2: Forward + Backward")
        print(f"{'#' * 78}")

    for n_node, n_edge, n_angle, label in [
        (10_000, 40_000, 20_000, "[FWD+BWD-M] Medium: 70k tokens/rank"),
        (50_000, 200_000, 100_000, "[FWD+BWD-L] Large: 350k tokens/rank"),
    ]:
        torch.cuda.reset_peak_memory_stats(device)
        try:
            bench_pipeline_backward(rank, world_size, group, device, packer, timer,
                                    n_node, n_edge, n_angle, label)
        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"\n  {label}: OOM — skipping")
        dist.barrier()

    # -----------------------------------------------------------------------
    # Part 3: Skewed distribution at medium scale.
    # -----------------------------------------------------------------------
    if rank == 0:
        print(f"\n\n{'#' * 78}")
        print(f"#  Part 3: Skewed Distribution (60% to GPU 0)")
        print(f"{'#' * 78}")

    for n_node, n_edge, n_angle, label in [
        (10_000, 40_000, 20_000, "[SKEW-M] Medium skewed: 70k tokens/rank"),
        (50_000, 200_000, 100_000, "[SKEW-L] Large skewed: 350k tokens/rank"),
    ]:
        torch.cuda.reset_peak_memory_stats(device)
        try:
            bench_pipeline_skewed(rank, world_size, group, device, packer, timer,
                                  n_node, n_edge, n_angle, label)
        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"\n  {label}: OOM — skipping")
        dist.barrier()

    if rank == 0:
        print(f"\n{'#' * 78}")
        print(f"#  Benchmark complete.")
        print(f"{'#' * 78}")

    cleanup_dist()


if __name__ == "__main__":
    main()
