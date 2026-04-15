# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-GPU unit tests for moe_ep_ops.

Run with torchrun, e.g.:
    torchrun --nproc_per_node=2 source/tests/pt/test_moe_ep_ops_multigpu.py
    torchrun --nproc_per_node=4 source/tests/pt/test_moe_ep_ops_multigpu.py

Tests:
  1. [2 GPU] forward: data arrives at the correct peer
  2. [2 GPU] 1st backward: gradients flow back via reverse A2A
  3. [2 GPU] 2nd backward: create_graph=True -> second .backward() works
  4. [4 GPU] same as above for 4-GPU scenario
  5. [2 GPU] multi-layer chain: 3 layers of dispatch+combine without deadlock
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

from deepmd.pt.model.network.moe_ep_ops import all_to_all_differentiable
from deepmd.utils.ddebug import ddebug


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_dist():
    """Initialize NCCL distributed backend from torchrun env vars."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def cleanup_dist():
    dist.destroy_process_group()


def check(condition: bool, msg: str, rank: int):
    """Assert-like helper that prints rank info on failure."""
    if not condition:
        print(f"[RANK {rank}] FAIL: {msg}", flush=True)
        sys.exit(1)


def all_pass(rank: int, world_size: int, tag: str):
    """Barrier + success print."""
    dist.barrier()
    if rank == 0:
        print(f"  PASS: {tag}", flush=True)


# ---------------------------------------------------------------------------
# Test: forward correctness (2 GPU)
# ---------------------------------------------------------------------------

def test_forward_2gpu(rank, world_size, group):
    """Each rank sends rank-specific data; verify correct receipt."""
    assert world_size == 2
    device = torch.device(f"cuda:{rank}")
    # Rank 0 sends [1,1,...] to rank 0 and [2,2,...] to rank 1
    # Rank 1 sends [3,3,...] to rank 0 and [4,4,...] to rank 1
    rows_per_dest = 3
    dim = 8
    send_splits = [rows_per_dest, rows_per_dest]
    recv_splits = [rows_per_dest, rows_per_dest]

    if rank == 0:
        part0 = torch.full((rows_per_dest, dim), 1.0, device=device, dtype=torch.float64)
        part1 = torch.full((rows_per_dest, dim), 2.0, device=device, dtype=torch.float64)
    else:
        part0 = torch.full((rows_per_dest, dim), 3.0, device=device, dtype=torch.float64)
        part1 = torch.full((rows_per_dest, dim), 4.0, device=device, dtype=torch.float64)
    x = torch.cat([part0, part1], dim=0)

    y = all_to_all_differentiable(x, send_splits, recv_splits, group)

    # Rank 0 should receive: [1,1,...] from rank 0, [3,3,...] from rank 1
    # Rank 1 should receive: [2,2,...] from rank 0, [4,4,...] from rank 1
    if rank == 0:
        expected = torch.cat([
            torch.full((rows_per_dest, dim), 1.0, device=device, dtype=torch.float64),
            torch.full((rows_per_dest, dim), 3.0, device=device, dtype=torch.float64),
        ])
    else:
        expected = torch.cat([
            torch.full((rows_per_dest, dim), 2.0, device=device, dtype=torch.float64),
            torch.full((rows_per_dest, dim), 4.0, device=device, dtype=torch.float64),
        ])

    check(torch.allclose(y, expected), "forward data mismatch", rank)
    all_pass(rank, world_size, "test_forward_2gpu")


# ---------------------------------------------------------------------------
# Test: 1st backward (2 GPU)
# ---------------------------------------------------------------------------

def test_backward_2gpu(rank, world_size, group):
    """Verify gradients flow back through reverse A2A."""
    device = torch.device(f"cuda:{rank}")
    rows_per_dest = 4
    dim = 5
    send_splits = [rows_per_dest, rows_per_dest]
    recv_splits = [rows_per_dest, rows_per_dest]

    x = torch.randn(rows_per_dest * 2, dim, device=device, dtype=torch.float64,
                     requires_grad=True)
    y = all_to_all_differentiable(x, send_splits, recv_splits, group)

    # Use a rank-dependent coefficient so gradients differ per rank
    coeff = float(rank + 1)
    loss = (y * coeff).sum()
    loss.backward()

    check(x.grad is not None, "x.grad is None", rank)
    check(x.grad.shape == x.shape, "x.grad shape mismatch", rank)

    # The gradient of loss w.r.t. y is coeff everywhere on the receiving rank.
    # The backward A2A reverses the communication, so each rank gets back
    # the coefficient from the rank that *received* its data.
    # Rank 0 sent rows [0:rows] to rank 0 and rows [rows:2*rows] to rank 1.
    # grad from rank 0 = 1.0, grad from rank 1 = 2.0
    if rank == 0:
        expected_grad = torch.cat([
            torch.full((rows_per_dest, dim), 1.0, device=device, dtype=torch.float64),
            torch.full((rows_per_dest, dim), 2.0, device=device, dtype=torch.float64),
        ])
    else:
        expected_grad = torch.cat([
            torch.full((rows_per_dest, dim), 1.0, device=device, dtype=torch.float64),
            torch.full((rows_per_dest, dim), 2.0, device=device, dtype=torch.float64),
        ])

    check(
        torch.allclose(x.grad, expected_grad),
        f"gradient mismatch: got {x.grad}, expected {expected_grad}",
        rank,
    )
    all_pass(rank, world_size, "test_backward_2gpu")


# ---------------------------------------------------------------------------
# Test: 2nd backward / create_graph (2 GPU)
# ---------------------------------------------------------------------------

def test_2nd_order_2gpu(rank, world_size, group):
    """Verify second-order derivatives through A2A with create_graph=True.

    Compute: y = A2A(x) -> loss = sum(y^2)
    1st grad: d(loss)/dx via create_graph=True -> this itself has an A2A node
    2nd backward: backward through the 1st grad graph -> another A2A
    """
    device = torch.device(f"cuda:{rank}")
    rows = 3
    dim = 4
    send_splits = [rows, rows]
    recv_splits = [rows, rows]

    x = torch.randn(rows * 2, dim, device=device, dtype=torch.float64,
                     requires_grad=True)
    y = all_to_all_differentiable(x, send_splits, recv_splits, group)
    loss = (y ** 2).sum()

    # 1st backward with create_graph=True
    (grad_x,) = torch.autograd.grad(loss, x, create_graph=True)

    check(grad_x is not None, "1st grad is None", rank)
    check(grad_x.requires_grad, "1st grad does not require_grad", rank)

    # 2nd backward: differentiate grad_x.sum() w.r.t. x
    grad_x_sum = grad_x.sum()
    grad_x_sum.backward()

    check(x.grad is not None, "2nd order x.grad is None", rank)
    # For loss = sum(y^2), d(loss)/dy = 2*y.
    # 1st backward A2A gives grad_x = A2A_reverse(2*y).
    # 2nd backward of grad_x.sum() w.r.t. x:
    #   d(sum(grad_x))/dx = d(sum(A2A_reverse(2*y)))/dx
    #                     = d(sum(A2A_reverse(2*A2A(x))))/dx
    #   = 2 (via chain of two A2As that cancel + scalar derivative of y^2)
    expected_2nd = torch.full_like(x, 2.0)
    check(
        torch.allclose(x.grad, expected_2nd, atol=1e-10),
        f"2nd order grad mismatch: got {x.grad}, expected {expected_2nd}",
        rank,
    )
    all_pass(rank, world_size, "test_2nd_order_2gpu")


# ---------------------------------------------------------------------------
# Test: forward + backward + 2nd order (4 GPU)
# ---------------------------------------------------------------------------

def test_forward_backward_4gpu(rank, world_size, group):
    """Same logic as 2-GPU tests but with 4 ranks and asymmetric splits."""
    assert world_size == 4
    device = torch.device(f"cuda:{rank}")
    dim = 6

    # Each rank sends different amounts to different ranks
    # Rank r sends (r+1) rows to each destination
    rows_per_dest = rank + 1
    send_splits = [rows_per_dest] * world_size
    total_send = sum(send_splits)

    # Gather recv_splits: rank r will receive (src_rank+1) rows from each src
    send_count_tensor = torch.tensor([rows_per_dest], device=device, dtype=torch.int64)
    all_counts = [torch.zeros(1, device=device, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(all_counts, send_count_tensor)
    recv_splits = [int(c.item()) for c in all_counts]

    x = torch.randn(total_send, dim, device=device, dtype=torch.float64,
                     requires_grad=True)

    # Forward
    y = all_to_all_differentiable(x, send_splits, recv_splits, group)
    expected_recv_rows = sum(recv_splits)
    check(y.shape[0] == expected_recv_rows, "forward shape mismatch", rank)

    # 1st backward with create_graph
    loss = (y ** 2).sum()
    (grad_x,) = torch.autograd.grad(loss, x, create_graph=True)
    check(grad_x.shape == x.shape, "1st grad shape mismatch", rank)
    check(grad_x.requires_grad, "1st grad does not require_grad", rank)


    # 2nd backward
    grad_x.sum().backward()
    check(x.grad is not None, "2nd order x.grad is None", rank)
    expected_2nd = torch.full_like(x, 2.0)
    check(
        torch.allclose(x.grad, expected_2nd, atol=1e-10),
        f"2nd order grad mismatch on rank {rank}",
        rank,
    )
    all_pass(rank, world_size, "test_forward_backward_4gpu")


# ---------------------------------------------------------------------------
# Test: multi-layer chain (2 GPU) — no deadlock
# ---------------------------------------------------------------------------

def test_multilayer_chain_2gpu(rank, world_size, group):
    """Simulate 3 layers of dispatch+combine to verify no deadlock.

    Each layer: dispatch (A2A) -> local compute -> combine (A2A_reverse).
    Then loss.backward() with create_graph=True, then 2nd backward.
    """
    assert world_size == 2
    device = torch.device(f"cuda:{rank}")
    rows = 4
    dim = 5
    send_splits = [rows, rows]
    recv_splits = [rows, rows]
    n_layers = 3

    x = torch.randn(rows * 2, dim, device=device, dtype=torch.float64,
                     requires_grad=True)
    # Simple linear layers to create differentiable parameters per layer
    layers = [
        torch.nn.Linear(dim, dim, dtype=torch.float64).to(device)
        for _ in range(n_layers)
    ]

    h = x
    for layer in layers:
        # Dispatch A2A
        dispatched = all_to_all_differentiable(h, send_splits, recv_splits, group)
        # Local compute
        computed = layer(dispatched)
        # Combine A2A (reverse direction: recv_splits <-> send_splits)
        h = all_to_all_differentiable(computed, recv_splits, send_splits, group)

    loss = (h ** 2).sum()

    # 1st backward with create_graph=True
    all_params = [x] + [p for l in layers for p in l.parameters()]
    grads = torch.autograd.grad(loss, all_params, create_graph=True)
    for i, g in enumerate(grads):
        check(g is not None, f"1st grad[{i}] is None", rank)
        check(g.requires_grad, f"1st grad[{i}] not differentiable", rank)

    # 2nd backward
    grad_sum = sum(g.sum() for g in grads)
    grad_sum.backward()

    check(x.grad is not None, "2nd order x.grad is None after multi-layer", rank)
    for i, layer in enumerate(layers):
        for pname, p in layer.named_parameters():
            check(
                p.grad is not None,
                f"layer {i} param {pname} has no 2nd-order grad",
                rank,
            )

    all_pass(rank, world_size, "test_multilayer_chain_2gpu")


# ---------------------------------------------------------------------------
# Test: multi-layer 2nd-order single-GPU vs multi-GPU consistency (2 GPU)
# ---------------------------------------------------------------------------

def test_multilayer_2nd_order_consistency_2gpu(rank, world_size, group):
    """Compare single-GPU simulated A2A vs real 2-GPU NCCL A2A.

    Each rank has its own *different* input x AND its own *different* layer
    parameters (like real EP where each rank holds different experts).

    Strategy (world_size=2):
    - Rank 0 collects all ranks' x data and layer parameters, then
      simulates the full 2-rank computation locally using tensor
      split+reorder to mimic A2A, with per-rank layers applied to
      data arriving at that simulated rank.
    - Both real ranks run the same chain with real NCCL A2A.
    - Compare forward output, 1st-order grad (create_graph=True),
      2nd-order grad, and layer param gradients.
    """
    assert world_size == 2
    device = torch.device(f"cuda:{rank}")
    N = 6   # rows per rank
    dim = 5
    n_layers = 2
    rows_per_dest = N // world_size
    assert N % world_size == 0
    send_splits = [rows_per_dest] * world_size
    recv_splits = [rows_per_dest] * world_size

    # ---- Each rank gets DIFFERENT x (seed = 100 + rank) ----
    torch.manual_seed(100 + rank)
    x_real = torch.randn(N, dim, device=device, dtype=torch.float64,
                         requires_grad=True)

    # ---- Each rank gets DIFFERENT layer params (seed = 500 + rank) ----
    torch.manual_seed(500 + rank)
    layers_real = []
    for _ in range(n_layers):
        l = torch.nn.Linear(dim, dim, bias=True, dtype=torch.float64,
                            device=device)
        layers_real.append(l)

    # ---- Real multi-GPU forward ----
    # dispatch → each rank applies its OWN layer → combine
    h = x_real
    for layer in layers_real:
        dispatched = all_to_all_differentiable(h, send_splits, recv_splits, group)
        computed = layer(dispatched)
        h = all_to_all_differentiable(computed, recv_splits, send_splits, group)
    loss_real = (h ** 2).sum()

    params_real = [x_real] + [p for l in layers_real for p in l.parameters()]
    grads_real = torch.autograd.grad(loss_real, params_real, create_graph=True)
    grad_sum_real = sum(g.sum() for g in grads_real)
    grad_sum_real.backward()

    fwd_real = h.detach().clone()
    g1_x_real = grads_real[0].detach().clone()
    # grads_real[1:] are layer param grads (weight, bias per layer)
    g1_layer_real = [g.detach().clone() for g in grads_real[1:]]
    g2_x_real = x_real.grad.detach().clone()
    g2_layer_real = []
    for l in layers_real:
        for p in l.parameters():
            g2_layer_real.append(p.grad.detach().clone())

    # ---- Gather all data to rank 0 for simulation ----
    x_data_list = [torch.zeros_like(x_real.data) for _ in range(world_size)]
    dist.all_gather(x_data_list, x_real.data, group=group)

    # Gather layer params: for each layer, gather weight and bias
    layer_params_all = []  # layer_params_all[layer_idx] = (w_list, b_list), indexed by [li][0=w|1=b][rank]
    for li, layer in enumerate(layers_real):
        w_list = [torch.zeros_like(layer.weight.data) for _ in range(world_size)]
        b_list = [torch.zeros_like(layer.bias.data) for _ in range(world_size)]
        dist.all_gather(w_list, layer.weight.data, group=group)
        dist.all_gather(b_list, layer.bias.data, group=group)
        layer_params_all.append((w_list, b_list))

    if rank == 0:
        # Build per-rank x tensors
        xs_sim = [d.clone().requires_grad_(True) for d in x_data_list]

        # Build per-rank layer lists (each rank has its own independent layers)
        layers_sim_per_rank = []  # [rank][layer_idx] = Linear
        for r in range(world_size):
            rank_layers = []
            for li in range(n_layers):
                l = torch.nn.Linear(dim, dim, bias=True, dtype=torch.float64,
                                    device=device)
                l.weight.data.copy_(layer_params_all[li][0][r])
                l.bias.data.copy_(layer_params_all[li][1][r])
                rank_layers.append(l)
            layers_sim_per_rank.append(rank_layers)

        def simulated_a2a(tensors):
            """Simulate A2A: split each tensor by rows_per_dest, reorder."""
            chunks = [t.split(rows_per_dest, dim=0) for t in tensors]
            result = []
            for dest in range(world_size):
                parts = [chunks[sender][dest] for sender in range(world_size)]
                result.append(torch.cat(parts, dim=0))
            return result

        # Forward: each simulated rank applies its OWN layer
        hs = list(xs_sim)
        for li in range(n_layers):
            dispatched = simulated_a2a(hs)
            computed = [layers_sim_per_rank[r][li](dispatched[r])
                        for r in range(world_size)]
            hs = simulated_a2a(computed)

        loss_sim = sum((h_r ** 2).sum() for h_r in hs)

        # Collect all differentiable params: xs + per-rank layer params
        all_sim_params = list(xs_sim)
        for r in range(world_size):
            for l in layers_sim_per_rank[r]:
                all_sim_params.extend(l.parameters())

        grads_sim = torch.autograd.grad(loss_sim, all_sim_params, create_graph=True)
        grad_sum_sim = sum(g.sum() for g in grads_sim)
        grad_sum_sim.backward()

        # Extract per-rank results
        fwd_sim = [h_r.detach().clone() for h_r in hs]
        g1_x_sim = [grads_sim[r].detach().clone() for r in range(world_size)]
        g2_x_sim = [xs_sim[r].grad.detach().clone() for r in range(world_size)]

        # Layer param grads: grads_sim layout is
        #   [x0, x1, r0_l0_w, r0_l0_b, r0_l1_w, r0_l1_b,
        #               r1_l0_w, r1_l0_b, r1_l1_w, r1_l1_b]
        n_layer_params = 2 * n_layers  # weight + bias per layer
        g1_layer_sim = {}  # g1_layer_sim[r] = list of param grads
        g2_layer_sim = {}
        offset = world_size  # skip x grads
        for r in range(world_size):
            g1_layer_sim[r] = [grads_sim[offset + i].detach().clone()
                               for i in range(n_layer_params)]
            g2_layer_sim[r] = []
            for l in layers_sim_per_rank[r]:
                for p in l.parameters():
                    g2_layer_sim[r].append(p.grad.detach().clone())
            offset += n_layer_params
    else:
        fwd_sim = [None] * world_size
        g1_x_sim = [None] * world_size
        g2_x_sim = [None] * world_size
        g1_layer_sim = {r: None for r in range(world_size)}
        g2_layer_sim = {r: None for r in range(world_size)}

    # ---- Broadcast simulated results from rank 0 ----
    for r in range(world_size):
        if fwd_sim[r] is None:
            fwd_sim[r] = torch.zeros_like(fwd_real)
            g1_x_sim[r] = torch.zeros_like(g1_x_real)
            g2_x_sim[r] = torch.zeros_like(g2_x_real)
        dist.broadcast(fwd_sim[r], src=0, group=group)
        dist.broadcast(g1_x_sim[r], src=0, group=group)
        dist.broadcast(g2_x_sim[r], src=0, group=group)

    # Broadcast layer param grads for each rank
    n_layer_params = 2 * n_layers
    for r in range(world_size):
        if g1_layer_sim[r] is None:
            g1_layer_sim[r] = [torch.zeros_like(g) for g in g1_layer_real]
            g2_layer_sim[r] = [torch.zeros_like(g) for g in g2_layer_real]
        for t in g1_layer_sim[r]:
            dist.broadcast(t, src=0, group=group)
        for t in g2_layer_sim[r]:
            dist.broadcast(t, src=0, group=group)

    # ---- Each rank compares its own result with simulation ----
    atol = 1e-10

    check(
        torch.allclose(fwd_real, fwd_sim[rank], atol=atol),
        f"forward mismatch: max diff = {(fwd_real - fwd_sim[rank]).abs().max().item():.2e}",
        rank,
    )
    check(
        torch.allclose(g1_x_real, g1_x_sim[rank], atol=atol),
        f"1st-order x.grad mismatch: max diff = {(g1_x_real - g1_x_sim[rank]).abs().max().item():.2e}",
        rank,
    )
    check(
        torch.allclose(g2_x_real, g2_x_sim[rank], atol=atol),
        f"2nd-order x.grad mismatch: max diff = {(g2_x_real - g2_x_sim[rank]).abs().max().item():.2e}",
        rank,
    )

    for i, (g_real, g_sim) in enumerate(zip(g1_layer_real, g1_layer_sim[rank])):
        check(
            torch.allclose(g_real, g_sim, atol=atol),
            f"1st-order layer param[{i}] grad mismatch: max diff = {(g_real - g_sim).abs().max().item():.2e}",
            rank,
        )
    for i, (g_real, g_sim) in enumerate(zip(g2_layer_real, g2_layer_sim[rank])):
        check(
            torch.allclose(g_real, g_sim, atol=atol),
            f"2nd-order layer param[{i}] grad mismatch: max diff = {(g_real - g_sim).abs().max().item():.2e}",
            rank,
        )

    all_pass(rank, world_size, "test_multilayer_2nd_order_consistency_2gpu")


# ---------------------------------------------------------------------------
# Test: asymmetric splits (2 GPU)
# ---------------------------------------------------------------------------

def test_asymmetric_splits_2gpu(rank, world_size, group):
    """Test with different send sizes to each rank."""
    assert world_size == 2
    device = torch.device(f"cuda:{rank}")
    dim = 4

    if rank == 0:
        send_splits = [2, 5]  # send 2 to rank0, 5 to rank1
    else:
        send_splits = [3, 1]  # send 3 to rank0, 1 to rank1

    # Exchange to get recv_splits
    send_tensor = torch.tensor(send_splits, device=device, dtype=torch.int64)
    recv_tensor = torch.zeros(2, device=device, dtype=torch.int64)
    dist.all_to_all_single(
        recv_tensor, send_tensor,
        output_split_sizes=[1, 1],
        input_split_sizes=[1, 1],
        group=group,
    )
    recv_splits = recv_tensor.tolist()

    total_send = sum(send_splits)
    x = torch.randn(total_send, dim, device=device, dtype=torch.float64,
                     requires_grad=True)

    y = all_to_all_differentiable(x, send_splits, recv_splits, group)
    check(y.shape[0] == sum(recv_splits), "shape mismatch", rank)

    # Backward
    loss = y.sum()
    loss.backward()
    check(x.grad is not None, "grad is None", rank)
    check(x.grad.shape == x.shape, "grad shape mismatch", rank)

    all_pass(rank, world_size, "test_asymmetric_splits_2gpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rank, world_size = setup_dist()
    group = dist.group.WORLD

    if rank == 0:
        print(f"Running moe_ep_ops multi-GPU tests with {world_size} GPUs", flush=True)

    try:
        if world_size == 2:
            test_forward_2gpu(rank, world_size, group)
            test_backward_2gpu(rank, world_size, group)
            test_2nd_order_2gpu(rank, world_size, group)
            test_asymmetric_splits_2gpu(rank, world_size, group)
            test_multilayer_chain_2gpu(rank, world_size, group)
            test_multilayer_2nd_order_consistency_2gpu(rank, world_size, group)

        if world_size == 4:
            test_forward_backward_4gpu(rank, world_size, group)

        if rank == 0:
            print(f"\nAll tests passed with {world_size} GPUs!", flush=True)
    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()
