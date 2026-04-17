# SPDX-License-Identifier: LGPL-3.0-or-later
"""Benchmark for MoEDispatchCombine (Step 6) - Single GPU.

Measures wall-clock time for forward, forward+backward, and 2nd-order
derivative under various problem sizes simulating real DPA3 workloads.

Run with:
    CUDA_VISIBLE_DEVICES=0 python source/tests/pt/bench_moe_layer.py
"""

from __future__ import annotations

import time

import torch

from deepmd.pt.model.network.moe_layer import MoEDispatchCombine


# ======================================================================
# Helpers
# ======================================================================

def _make_layer(
    a_dim: int,
    n_routing_experts: int,
    topk: int,
    n_shared_experts: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> MoEDispatchCombine:
    n_dim = 4 * a_dim
    e_dim = 2 * a_dim
    n_sym_dim = 24 * a_dim
    edge_info_dim = 10 * a_dim
    angle_dim = 4 * a_dim
    layer = MoEDispatchCombine(
        n_dim=n_dim, e_dim=e_dim, a_dim=a_dim,
        n_sym_dim=n_sym_dim, edge_info_dim=edge_info_dim, angle_dim=angle_dim,
        n_routing_experts=n_routing_experts, topk=topk,
        n_shared_experts=n_shared_experts,
        ep_group=None, ep_rank=0, ep_size=1,
        experts_per_gpu=n_routing_experts,
        activation_function="silu", precision="float64", seed=42,
    )
    return layer.to(device)


def _make_inputs(
    a_dim: int,
    n_node: int, n_edge: int, n_angle: int,
    n_routing_experts: int, topk: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = False,
) -> dict:
    n_dim = 4 * a_dim
    n_sym_dim = 24 * a_dim
    edge_info_dim = 10 * a_dim
    angle_dim = 4 * a_dim

    node_m1 = torch.randn(n_node, n_dim, device=device, dtype=dtype,
                           requires_grad=requires_grad)
    node_m2 = torch.randn(n_node, n_sym_dim, device=device, dtype=dtype,
                           requires_grad=requires_grad)
    edge = torch.randn(n_edge, edge_info_dim, device=device, dtype=dtype,
                        requires_grad=requires_grad)
    angle = torch.randn(n_angle, angle_dim, device=device, dtype=dtype,
                         requires_grad=requires_grad)

    logits_n = torch.randn(n_node, n_routing_experts, device=device, dtype=dtype)
    topk_l, topk_i = torch.topk(logits_n, k=topk, dim=-1)
    node_w = torch.softmax(topk_l, dim=-1)
    if requires_grad:
        node_w = node_w.detach().requires_grad_(True)

    logits_e = torch.randn(n_node, n_routing_experts, device=device, dtype=dtype)
    topk_l, topk_i_e = torch.topk(logits_e, k=topk, dim=-1)
    edge_w = torch.softmax(topk_l, dim=-1)
    if requires_grad:
        edge_w = edge_w.detach().requires_grad_(True)

    logits_a = torch.randn(n_node, n_routing_experts, device=device, dtype=dtype)
    topk_l, topk_i_a = torch.topk(logits_a, k=topk, dim=-1)
    angle_w = torch.softmax(topk_l, dim=-1)
    if requires_grad:
        angle_w = angle_w.detach().requires_grad_(True)

    n2e_index = torch.randint(0, n_node, (n_edge,), device=device)
    n2a_index = torch.randint(0, n_node, (n_angle,), device=device)

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


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ======================================================================
# Benchmark runner
# ======================================================================

def bench_forward(layer, inputs, n_warmup=3, n_iter=10):
    """Benchmark forward pass."""
    for _ in range(n_warmup):
        layer(**inputs)
    _sync_cuda()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        layer(**inputs)
    _sync_cuda()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000  # ms


def bench_forward_backward(layer, inputs, n_warmup=3, n_iter=10):
    """Benchmark forward + backward."""
    for _ in range(n_warmup):
        m1, m2, e, a = layer(**inputs)
        loss = (m1**2).sum() + (m2**2).sum() + (e**2).sum() + (a**2).sum()
        loss.backward()
        layer.zero_grad()
        for key in ["node_m1_input", "node_m2_input", "edge_input", "angle_input"]:
            if inputs[key].grad is not None:
                inputs[key].grad = None
    _sync_cuda()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        m1, m2, e, a = layer(**inputs)
        loss = (m1**2).sum() + (m2**2).sum() + (e**2).sum() + (a**2).sum()
        loss.backward()
        layer.zero_grad()
        for key in ["node_m1_input", "node_m2_input", "edge_input", "angle_input"]:
            if inputs[key].grad is not None:
                inputs[key].grad = None
    _sync_cuda()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000  # ms


def bench_second_order(layer, inputs, n_warmup=2, n_iter=5):
    """Benchmark forward + 1st backward (create_graph) + 2nd backward."""
    def run_once():
        m1, m2, e, a = layer(**inputs)
        loss = (m1**2).sum() + (m2**2).sum() + (e**2).sum() + (a**2).sum()
        (grad_m1,) = torch.autograd.grad(
            loss, inputs["node_m1_input"], create_graph=True,
        )
        grad_loss = (grad_m1**2).sum()
        grad_loss.backward()
        layer.zero_grad()
        for key in ["node_m1_input", "node_m2_input", "edge_input", "angle_input"]:
            if inputs[key].grad is not None:
                inputs[key].grad = None

    for _ in range(n_warmup):
        run_once()
    _sync_cuda()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        run_once()
    _sync_cuda()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000  # ms


# ======================================================================
# Scenarios
# ======================================================================

SCENARIOS = [
    # (label, a_dim, n_node, n_edge, n_angle, n_experts, topk, n_shared)
    ("Small (water 64)",    4,    192,     2400,     6000,   4, 2, 0),
    ("Medium (water 256)",  4,    768,     9600,    24000,   4, 2, 0),
    ("Large (water 1024)",  4,   3072,    38400,    96000,   4, 2, 0),
    ("Large + shared",      4,   3072,    38400,    96000,   4, 2, 1),
    ("Large 8-expert",      4,   3072,    38400,    96000,   8, 2, 0),
    ("Large topk=4",        4,   3072,    38400,    96000,   8, 4, 0),
    ("XL (water 4096)",     4,  12288,   153600,   384000,   4, 2, 0),
    ("Small a_dim=8",       8,    192,     2400,     6000,   4, 2, 0),
    ("Medium a_dim=8",      8,    768,     9600,    24000,   4, 2, 0),
    # --- topk=4, varying n_experts (computation should depend on topk, not n_experts) ---
    ("Large k4 E4",         4,   3072,    38400,    96000,   4, 4, 0),
    ("Large k4 E8",         4,   3072,    38400,    96000,   8, 4, 0),
    ("Large k4 E16",        4,   3072,    38400,    96000,  16, 4, 0),
    ("Large k4 E32",        4,   3072,    38400,    96000,  32, 4, 0),
    ("Large k4 E64",        4,   3072,    38400,    96000,  64, 4, 0),
    # --- topk=8, varying n_experts ---
    ("Large k8 E8",         4,   3072,    38400,    96000,   8, 8, 0),
    ("Large k8 E16",        4,   3072,    38400,    96000,  16, 8, 0),
    ("Large k8 E32",        4,   3072,    38400,    96000,  32, 8, 0),
    ("Large k8 E64",        4,   3072,    38400,    96000,  64, 8, 0),
]


def main():
    # Determine device.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"\n{'='*90}")
    print(f"MoEDispatchCombine Benchmark (Single GPU) — Device: {device_name}")
    print(f"{'='*90}")
    print(f"\n{'Scenario':<24} {'N_node':>8} {'N_edge':>8} {'N_angle':>8} "
          f"{'E':>3} {'k':>2} {'Forward':>10} {'Fwd+Bwd':>10} {'2nd-Order':>10}")
    print(f"{'':<24} {'':>8} {'':>8} {'':>8} "
          f"{'':>3} {'':>2} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10}")
    print(f"{'-'*90}")

    for label, a_dim, n_node, n_edge, n_angle, n_exp, topk, n_shared in SCENARIOS:
        # Skip XL on CPU (too slow).
        if device.type == "cpu" and n_node > 4000:
            print(f"{label:<24} {'(skipped on CPU)':>60}")
            continue

        try:
            layer = _make_layer(a_dim, n_exp, topk, n_shared, device)

            # Forward-only benchmark.
            inputs_nograd = _make_inputs(
                a_dim, n_node, n_edge, n_angle, n_exp, topk,
                device, requires_grad=False,
            )
            t_fwd = bench_forward(layer, inputs_nograd)

            # Forward+backward benchmark.
            inputs_grad = _make_inputs(
                a_dim, n_node, n_edge, n_angle, n_exp, topk,
                device, requires_grad=True,
            )
            t_fwd_bwd = bench_forward_backward(layer, inputs_grad)

            # 2nd-order benchmark (fewer iterations for large sizes).
            if n_node <= 4000:
                inputs_grad2 = _make_inputs(
                    a_dim, n_node, n_edge, n_angle, n_exp, topk,
                    device, requires_grad=True,
                )
                t_2nd = bench_second_order(layer, inputs_grad2)
                t_2nd_str = f"{t_2nd:>10.2f}"
            else:
                # For very large, do just 1 iteration.
                inputs_grad2 = _make_inputs(
                    a_dim, n_node, n_edge, n_angle, n_exp, topk,
                    device, requires_grad=True,
                )
                t_2nd = bench_second_order(layer, inputs_grad2, n_warmup=1, n_iter=2)
                t_2nd_str = f"{t_2nd:>10.2f}"

            print(f"{label:<24} {n_node:>8d} {n_edge:>8d} {n_angle:>8d} "
                  f"{n_exp:>3d} {topk:>2d} {t_fwd:>10.2f} {t_fwd_bwd:>10.2f} {t_2nd_str}")

        except Exception as ex:
            print(f"{label:<24} ERROR: {ex}")

        # Free memory.
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
