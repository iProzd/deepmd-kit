#!/usr/bin/env python3
"""Detailed profiling of MoE EP implementation with communication cost breakdown.

Launch: torchrun --nproc_per_node=2 tests/test_moe_ep_profile.py
"""

import argparse
import os
import time
from collections import defaultdict

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


class CommProfiler:
    """Profile communication costs in EP MoE."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.timings = defaultdict(list)
        self.counts = defaultdict(int)
        self.bytes = defaultdict(int)

    def record(self, name, elapsed_ms, num_bytes=0):
        self.timings[name].append(elapsed_ms)
        self.counts[name] += 1
        self.bytes[name] += num_bytes

    def summary(self):
        result = {}
        for name in self.timings:
            times = self.timings[name]
            result[name] = {
                'count': self.counts[name],
                'total_ms': sum(times),
                'avg_ms': sum(times) / len(times) if times else 0,
                'total_mb': self.bytes[name] / 1e6,
            }
        return result


def profile_ep_step(rank, ep_group, num_experts, dim, nb, nloc, ntypes, steps=50):
    """Profile a single EP configuration with detailed timing."""
    from deepmd.pt.model.network.moe import MoELayer

    top_k = min(2, num_experts)
    ep_size = dist.get_world_size(group=ep_group)

    torch.manual_seed(42)
    moe = MoELayer(dim, dim, num_experts, top_k, dim, ep_group=ep_group, seed=42).cuda()
    dist.broadcast(moe.gate.matrix.data, src=0, group=ep_group)

    encoder = nn.Linear(3, dim, bias=False).cuda()
    dist.broadcast(encoder.weight.data, src=0, group=ep_group)

    type_emb = torch.randn(ntypes, dim, device="cuda")
    atom_types = torch.randint(0, ntypes, (nb, nloc), device="cuda")
    dist.broadcast(type_emb, src=0, group=ep_group)
    dist.broadcast(atom_types, src=0, group=ep_group)

    profiler = CommProfiler()

    # Warmup
    for _ in range(10):
        pos = torch.randn(nb, nloc, 3, device="cuda", requires_grad=True)
        hidden = encoder(pos)
        out = moe(hidden, type_emb, atom_types)
        energy = out.sum()
        force = -torch.autograd.grad(energy, pos, create_graph=True)[0]
        loss = (energy + force.sum())
        loss.backward()

    torch.cuda.synchronize()
    dist.barrier(group=ep_group)

    # Profile forward + backward
    total_time = 0
    for _ in range(steps):
        pos = torch.randn(nb, nloc, 3, device="cuda", requires_grad=True)

        t0 = time.perf_counter()
        hidden = encoder(pos)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        profiler.record('encoder', (t1-t0)*1000)

        # MoE forward (includes A2A)
        t0 = time.perf_counter()
        out = moe(hidden, type_emb, atom_types)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        moe_fwd_time = (t1-t0)*1000
        profiler.record('moe_forward', moe_fwd_time)

        # Energy + force
        t0 = time.perf_counter()
        energy = out.sum()
        force = -torch.autograd.grad(energy, pos, create_graph=True, retain_graph=True)[0]
        loss = energy + force.sum()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        profiler.record('energy_force', (t1-t0)*1000)

        # Backward (includes A2A)
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        profiler.record('backward', (t1-t0)*1000)

        # Gradient sync
        t0 = time.perf_counter()
        for name, p in moe.named_parameters():
            if p.grad is not None and "experts" not in name:
                dist.all_reduce(p.grad, group=ep_group)
                p.grad /= ep_size
        for p in encoder.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, group=ep_group)
                p.grad /= ep_size
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        profiler.record('grad_sync', (t1-t0)*1000)

        step_time = sum([
            profiler.timings['encoder'][-1],
            profiler.timings['moe_forward'][-1],
            profiler.timings['energy_force'][-1],
            profiler.timings['backward'][-1],
            profiler.timings['grad_sync'][-1],
        ])
        total_time += step_time

    summary = profiler.summary()
    throughput = steps / (total_time / 1000)

    # Estimate communication bytes
    n_tokens = nb * nloc
    n_expanded = n_tokens * top_k
    bytes_per_token = dim * 4  # float32

    # Forward A2A: dispatch + combine
    comm_bytes_fwd = 2 * n_expanded * bytes_per_token
    # Backward A2A: same
    comm_bytes_bwd = 2 * n_expanded * bytes_per_token

    summary['comm_estimate'] = {
        'fwd_mb': comm_bytes_fwd / 1e6,
        'bwd_mb': comm_bytes_bwd / 1e6,
        'total_mb_per_step': (comm_bytes_fwd + comm_bytes_bwd) / 1e6,
    }

    return throughput, summary


def main():
    rank = setup()
    mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("dp", "ep"))
    ep_group = mesh["ep"].get_group()

    log(rank, "=" * 80)
    log(rank, "MoE EP Detailed Profiling (2 GPUs, EP=2)")
    log(rank, "=" * 80)

    configs = [
        # (num_experts, dim, nb, nloc, ntypes)
        (4, 32, 4, 64, 4),
        (8, 32, 4, 64, 8),
        (4, 128, 4, 64, 4),
        (8, 128, 4, 64, 8),
        (4, 256, 4, 64, 4),
        (8, 256, 4, 64, 8),
        (8, 256, 8, 128, 8),
    ]

    results = []
    for n_exp, dim, nb, nloc, nt in configs:
        torch.cuda.empty_cache()
        dist.barrier(group=ep_group)

        tp, summary = profile_ep_step(rank, ep_group, n_exp, dim, nb, nloc, nt)

        if rank == 0:
            results.append({
                'config': (n_exp, dim, nb, nloc, nt),
                'throughput': tp,
                'summary': summary,
            })

    if rank == 0:
        log(rank, "\n" + "=" * 80)
        log(rank, "DETAILED PROFILING RESULTS")
        log(rank, "=" * 80)

        for res in results:
            n_exp, dim, nb, nloc, nt = res['config']
            tp = res['throughput']
            s = res['summary']

            log(rank, f"\nConfig: E={n_exp}, dim={dim}, nb={nb}, nloc={nloc}, ntypes={nt}")
            log(rank, f"  Throughput: {tp:.1f} steps/s")
            log(rank, f"  Time breakdown (avg per step):")
            log(rank, f"    Encoder:     {s['encoder']['avg_ms']:>6.2f} ms")
            log(rank, f"    MoE forward: {s['moe_forward']['avg_ms']:>6.2f} ms")
            log(rank, f"    Energy+Force:{s['energy_force']['avg_ms']:>6.2f} ms")
            log(rank, f"    Backward:    {s['backward']['avg_ms']:>6.2f} ms")
            log(rank, f"    Grad sync:   {s['grad_sync']['avg_ms']:>6.2f} ms")
            total_avg = sum([s[k]['avg_ms'] for k in ['encoder', 'moe_forward', 'energy_force', 'backward', 'grad_sync']])
            log(rank, f"    Total:       {total_avg:>6.2f} ms")

            comm = s['comm_estimate']
            log(rank, f"  Communication (estimated per step):")
            log(rank, f"    Forward A2A:  {comm['fwd_mb']:.2f} MB")
            log(rank, f"    Backward A2A: {comm['bwd_mb']:.2f} MB")
            log(rank, f"    Total A2A:    {comm['total_mb_per_step']:.2f} MB")

            # Compute percentage
            moe_pct = s['moe_forward']['avg_ms'] / total_avg * 100
            bwd_pct = s['backward']['avg_ms'] / total_avg * 100
            comm_pct = (s['moe_forward']['avg_ms'] + s['backward']['avg_ms']) / total_avg * 100
            log(rank, f"  MoE forward: {moe_pct:.1f}% of total time")
            log(rank, f"  Backward: {bwd_pct:.1f}% of total time")
            log(rank, f"  Communication overhead: ~{comm_pct:.1f}% of total time")

    log(rank, "\n" + "=" * 80)
    log(rank, "Profiling complete")
    log(rank, "=" * 80)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
