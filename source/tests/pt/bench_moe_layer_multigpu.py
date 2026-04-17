# SPDX-License-Identifier: LGPL-3.0-or-later
"""Benchmark for MoEDispatchCombine (Step 6) - Multi-GPU.

Measures wall-clock time for forward, forward+backward, and 2nd-order
derivative with real multi-GPU All-to-All communication.

Run with:
    torchrun --nproc_per_node=2 source/tests/pt/bench_moe_layer_multigpu.py
    torchrun --nproc_per_node=4 source/tests/pt/bench_moe_layer_multigpu.py
    torchrun --nproc_per_node=8 source/tests/pt/bench_moe_layer_multigpu.py
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from deepmd.pt.model.network.moe_layer import MoEDispatchCombine


DTYPE = torch.float64


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
    a_dim: int,
    n_routing_experts: int,
    topk: int,
    n_shared_experts: int,
    ep_group,
    ep_rank: int,
    ep_size: int,
) -> MoEDispatchCombine:
    n_dim = 4 * a_dim
    e_dim = 2 * a_dim
    n_sym_dim = 24 * a_dim
    edge_info_dim = 10 * a_dim
    angle_dim = 4 * a_dim
    experts_per_gpu = n_routing_experts // ep_size

    layer = MoEDispatchCombine(
        n_dim=n_dim, e_dim=e_dim, a_dim=a_dim,
        n_sym_dim=n_sym_dim, edge_info_dim=edge_info_dim, angle_dim=angle_dim,
        n_routing_experts=n_routing_experts, topk=topk,
        n_shared_experts=n_shared_experts,
        ep_group=ep_group, ep_rank=ep_rank, ep_size=ep_size,
        experts_per_gpu=experts_per_gpu,
        activation_function="silu", precision="float64", seed=42,
    )
    device = torch.device(f"cuda:{ep_rank}")
    return layer.to(device)


def _make_inputs(
    a_dim: int,
    n_node: int, n_edge: int, n_angle: int,
    n_routing_experts: int, topk: int,
    rank: int,
    requires_grad: bool = False,
) -> dict:
    device = torch.device(f"cuda:{rank}")
    n_dim = 4 * a_dim
    n_sym_dim = 24 * a_dim
    edge_info_dim = 10 * a_dim
    angle_dim = 4 * a_dim

    # Use rank-specific seed (simulates different data per GPU in DP).
    gen = torch.Generator(device="cpu")
    gen.manual_seed(100 + rank)

    node_m1 = torch.randn(n_node, n_dim, device="cpu", dtype=DTYPE,
                           generator=gen).to(device).requires_grad_(requires_grad)
    node_m2 = torch.randn(n_node, n_sym_dim, device="cpu", dtype=DTYPE,
                           generator=gen).to(device).requires_grad_(requires_grad)
    edge = torch.randn(n_edge, edge_info_dim, device="cpu", dtype=DTYPE,
                        generator=gen).to(device).requires_grad_(requires_grad)
    angle = torch.randn(n_angle, angle_dim, device="cpu", dtype=DTYPE,
                         generator=gen).to(device).requires_grad_(requires_grad)

    logits_n = torch.randn(n_node, n_routing_experts, device="cpu",
                            dtype=DTYPE, generator=gen).to(device)
    topk_l, topk_i = torch.topk(logits_n, k=topk, dim=-1)
    node_w = torch.softmax(topk_l, dim=-1)
    if requires_grad:
        node_w = node_w.detach().requires_grad_(True)

    logits_e = torch.randn(n_node, n_routing_experts, device="cpu",
                            dtype=DTYPE, generator=gen).to(device)
    topk_l, topk_i_e = torch.topk(logits_e, k=topk, dim=-1)
    edge_w = torch.softmax(topk_l, dim=-1)
    if requires_grad:
        edge_w = edge_w.detach().requires_grad_(True)

    logits_a = torch.randn(n_node, n_routing_experts, device="cpu",
                            dtype=DTYPE, generator=gen).to(device)
    topk_l, topk_i_a = torch.topk(logits_a, k=topk, dim=-1)
    angle_w = torch.softmax(topk_l, dim=-1)
    if requires_grad:
        angle_w = angle_w.detach().requires_grad_(True)

    n2e_index = torch.randint(0, n_node, (n_edge,), generator=gen).to(device)
    n2a_index = torch.randint(0, n_node, (n_angle,), generator=gen).to(device)

    return {
        "node_m1_input": node_m1,
        "node_m2_input": node_m2,
        "edge_input": edge,
        "angle_input": angle,
        "node_router_out": (node_w, topk_i),
        "edge_router_out": (edge_w, topk_i_e),
        "angle_router_out": (angle_w, topk_i_a),
        "n2e_index": n2e_index,
        "n2a_index": n2a_index,
    }


def _clear_grads(layer, inputs):
    layer.zero_grad()
    for key in ["node_m1_input", "node_m2_input", "edge_input", "angle_input"]:
        if inputs[key].grad is not None:
            inputs[key].grad = None


# ======================================================================
# Benchmark functions
# ======================================================================

def bench_forward(layer, inputs, n_warmup=3, n_iter=10):
    for _ in range(n_warmup):
        layer(**inputs)
    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        layer(**inputs)
    torch.cuda.synchronize()
    dist.barrier()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000


def bench_forward_backward(layer, inputs, n_warmup=3, n_iter=10):
    for _ in range(n_warmup):
        m1, m2, e, a = layer(**inputs)
        loss = (m1**2).sum() + (m2**2).sum() + (e**2).sum() + (a**2).sum()
        loss.backward()
        _clear_grads(layer, inputs)
    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        m1, m2, e, a = layer(**inputs)
        loss = (m1**2).sum() + (m2**2).sum() + (e**2).sum() + (a**2).sum()
        loss.backward()
        _clear_grads(layer, inputs)
    torch.cuda.synchronize()
    dist.barrier()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000


def bench_second_order(layer, inputs, n_warmup=2, n_iter=5):
    def run_once():
        m1, m2, e, a = layer(**inputs)
        loss = (m1**2).sum() + (m2**2).sum() + (e**2).sum() + (a**2).sum()
        (grad_m1,) = torch.autograd.grad(
            loss, inputs["node_m1_input"], create_graph=True,
        )
        grad_loss = (grad_m1**2).sum()
        grad_loss.backward()
        _clear_grads(layer, inputs)

    for _ in range(n_warmup):
        run_once()
    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        run_once()
    torch.cuda.synchronize()
    dist.barrier()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000


# ======================================================================
# Scenarios
# ======================================================================

# Per-GPU data sizes (each GPU processes its own data shard).
# n_experts=None means world_size * 2 (2 experts per GPU).
# n_experts=int is the total; experts_per_gpu = n_experts // world_size.
SCENARIOS = [
    # (label, a_dim, n_node, n_edge, n_angle, n_experts, topk, n_shared)
    # --- Section 1: scaling with system size, fixed topk=2, default experts ---
    ("Small (64 atoms)",        4,    192,     2400,     6000,   None, 2, 0),
    ("Medium (256 atoms)",      4,    768,     9600,    24000,   None, 2, 0),
    ("Large (1024 atoms)",      4,   3072,    38400,    96000,   None, 2, 0),
    ("XL (4096 atoms)",         4,  12288,   153600,   384000,   None, 2, 0),
    # --- Section 2: Large system, topk=2, varying n_experts ---
    ("L k2 E=ws*1",            4,   3072,    38400,    96000, "ws*1", 2, 0),
    ("L k2 E=ws*2",            4,   3072,    38400,    96000, "ws*2", 2, 0),
    ("L k2 E=ws*4",            4,   3072,    38400,    96000, "ws*4", 2, 0),
    ("L k2 E=ws*8",            4,   3072,    38400,    96000, "ws*8", 2, 0),
    # --- Section 3: Large system, topk=4, varying n_experts ---
    ("L k4 E=ws*1",            4,   3072,    38400,    96000, "ws*1", 4, 0),
    ("L k4 E=ws*2",            4,   3072,    38400,    96000, "ws*2", 4, 0),
    ("L k4 E=ws*4",            4,   3072,    38400,    96000, "ws*4", 4, 0),
    ("L k4 E=ws*8",            4,   3072,    38400,    96000, "ws*8", 4, 0),
    # --- Section 4: Large system, topk=8, varying n_experts ---
    ("L k8 E=ws*2",            4,   3072,    38400,    96000, "ws*2", 8, 0),
    ("L k8 E=ws*4",            4,   3072,    38400,    96000, "ws*4", 8, 0),
    ("L k8 E=ws*8",            4,   3072,    38400,    96000, "ws*8", 8, 0),
    # --- Section 5: shared experts ---
    ("L k2 shared=1",          4,   3072,    38400,    96000,   None, 2, 1),
    ("L k4 shared=1",          4,   3072,    38400,    96000,   None, 4, 1),
    # --- Section 6: Medium system, more topk/experts combos ---
    ("M k2 E=ws*2",            4,    768,     9600,    24000, "ws*2", 2, 0),
    ("M k4 E=ws*4",            4,    768,     9600,    24000, "ws*4", 4, 0),
    ("M k8 E=ws*4",            4,    768,     9600,    24000, "ws*4", 8, 0),
]


def main():
    rank, world_size = setup_dist()
    ep_group = dist.new_group(list(range(world_size)))
    device_name = torch.cuda.get_device_name(rank)

    if rank == 0:
        print(f"\n{'='*110}")
        print(f"MoEDispatchCombine Benchmark (Multi-GPU) — {world_size} GPUs — {device_name}")
        print(f"{'='*110}")
        print(f"\n{'Scenario':<26} {'N_node':>8} {'N_edge':>8} {'N_angle':>8} "
              f"{'E':>4} {'E/G':>4} {'k':>2} {'Forward':>10} {'Fwd+Bwd':>10} {'2nd-Order':>10}")
        print(f"{'':<26} {'(per GPU)':>8} {'(per GPU)':>8} {'(per GPU)':>8} "
              f"{'':>4} {'':>4} {'':>2} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10}")
        print(f"{'-'*110}")

    for label, a_dim, n_node, n_edge, n_angle, n_exp_raw, topk, n_shared in SCENARIOS:
        # Resolve n_experts.
        # None -> world_size * 2 (default: 2 experts per GPU)
        # "ws*N" -> world_size * N
        # int -> use directly (must be divisible by world_size)
        if n_exp_raw is None:
            n_exp = world_size * 2
        elif isinstance(n_exp_raw, str) and n_exp_raw.startswith("ws*"):
            n_exp = world_size * int(n_exp_raw[3:])
        else:
            n_exp = n_exp_raw

        try:
            experts_per_gpu = n_exp // world_size
            layer = _make_layer(a_dim, n_exp, topk, n_shared, ep_group, rank, world_size)

            # Forward.
            inputs_nograd = _make_inputs(
                a_dim, n_node, n_edge, n_angle, n_exp, topk, rank,
                requires_grad=False,
            )
            t_fwd = bench_forward(layer, inputs_nograd)

            # Forward + backward.
            inputs_grad = _make_inputs(
                a_dim, n_node, n_edge, n_angle, n_exp, topk, rank,
                requires_grad=True,
            )
            t_fwd_bwd = bench_forward_backward(layer, inputs_grad)

            # 2nd-order.
            if n_node <= 4000:
                inputs_grad2 = _make_inputs(
                    a_dim, n_node, n_edge, n_angle, n_exp, topk, rank,
                    requires_grad=True,
                )
                t_2nd = bench_second_order(layer, inputs_grad2)
                t_2nd_str = f"{t_2nd:>10.2f}"
            else:
                inputs_grad2 = _make_inputs(
                    a_dim, n_node, n_edge, n_angle, n_exp, topk, rank,
                    requires_grad=True,
                )
                t_2nd = bench_second_order(layer, inputs_grad2, n_warmup=1, n_iter=2)
                t_2nd_str = f"{t_2nd:>10.2f}"

            if rank == 0:
                print(f"{label:<26} {n_node:>8d} {n_edge:>8d} {n_angle:>8d} "
                      f"{n_exp:>4d} {experts_per_gpu:>4d} {topk:>2d} "
                      f"{t_fwd:>10.2f} {t_fwd_bwd:>10.2f} {t_2nd_str}")

        except Exception as ex:
            if rank == 0:
                print(f"{label:<26} ERROR: {ex}")

        del layer
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        print(f"{'='*110}\n")

    cleanup_dist()


if __name__ == "__main__":
    main()
