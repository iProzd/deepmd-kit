# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-GPU unit tests for MoEPacker.

Run with torchrun:
    torchrun --nproc_per_node=2 source/tests/pt/test_moe_packer_multigpu.py
    torchrun --nproc_per_node=4 source/tests/pt/test_moe_packer_multigpu.py

Tests:
  [2 GPU] Full roundtrip: pack → A2A → unpack → expert → repack → A2A → unpack
  [2 GPU] Gradient through full A2A + packer pipeline
  [2 GPU] 2nd-order derivative through full pipeline
  [2 GPU] Asymmetric counts (different node/edge/angle per rank)
  [4 GPU] Full roundtrip with 4 GPUs
  [4 GPU] 2nd-order derivative with 4 GPUs
"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist

from deepmd.pt.model.network.moe_ep_ops import all_to_all_differentiable
from deepmd.pt.model.network.moe_packer import MoEPacker
from deepmd.utils.ddebug import ddebug


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_moe_ep_ops_multigpu.py)
# ---------------------------------------------------------------------------

A_DIM = 3  # base unit for packer tests
DTYPE = torch.float64


def setup_dist():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def cleanup_dist():
    dist.destroy_process_group()


def check(condition: bool, msg: str, rank: int):
    if not condition:
        print(f"[RANK {rank}] FAIL: {msg}", flush=True)
        sys.exit(1)


def all_pass(rank: int, world_size: int, tag: str):
    dist.barrier()
    if rank == 0:
        print(f"  PASS: {tag}", flush=True)


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _compute_send_splits(node_counts, edge_counts, angle_counts):
    """Compute packed row counts per GPU (same logic as MoEPacker)."""
    splits = []
    for g in range(len(node_counts)):
        n_node = node_counts[g]
        n_edge_rows = _ceildiv(edge_counts[g], 4) if edge_counts[g] > 0 else 0
        n_angle_rows = _ceildiv(angle_counts[g], 10) if angle_counts[g] > 0 else 0
        splits.append(n_node + n_edge_rows + n_angle_rows)
    return splits


def _exchange_counts(send_counts, rank, world_size, group, device):
    """Exchange per-GPU counts via all_to_all.

    send_counts: [world_size, 3] int tensor (node, edge, angle per dest GPU)
    Returns recv_counts: [world_size, 3] (node, edge, angle per src GPU)
    """
    send_t = torch.tensor(send_counts, dtype=torch.int64, device=device)
    recv_t = torch.zeros_like(send_t)
    # All-to-all each field: send_t[g] goes to GPU g, recv_t[g] comes from GPU g
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
# Test: Full roundtrip (2 GPU)
# ---------------------------------------------------------------------------

def test_full_roundtrip_2gpu(rank, world_size, group):
    """Pack → A2A dispatch → unpack → identity expert → repack → A2A combine → unpack.

    Each rank generates unique data, sends to the other, and verifies
    everything comes back correctly.
    """
    assert world_size == 2
    device = torch.device(f"cuda:{rank}")
    packer = MoEPacker(A_DIM)

    # Each rank has different counts.
    if rank == 0:
        n_node, n_edge, n_angle = 5, 7, 13
    else:
        n_node, n_edge, n_angle = 3, 12, 4

    # Generate per-rank unique data.
    torch.manual_seed(1000 + rank)
    node_in = torch.randn(n_node, 28 * A_DIM, dtype=DTYPE, device=device)
    edge_in = torch.randn(n_edge, 10 * A_DIM, dtype=DTYPE, device=device)
    angle_in = torch.randn(n_angle, 4 * A_DIM, dtype=DTYPE, device=device)

    # For simplicity: each rank sends ALL its data to the OTHER rank.
    # node_counts_per_gpu[g] = how many of my nodes go to GPU g.
    if rank == 0:
        node_counts = [0, n_node]
        edge_counts = [0, n_edge]
        angle_counts = [0, n_angle]
    else:
        node_counts = [n_node, 0]
        edge_counts = [n_edge, 0]
        angle_counts = [n_angle, 0]

    # Pack for dispatch.
    packed, send_splits = packer.pack_for_dispatch(
        node_in, edge_in, angle_in,
        node_counts, edge_counts, angle_counts,
    )

    # Exchange metadata to get recv splits.
    send_info = [[node_counts[g], edge_counts[g], angle_counts[g]]
                 for g in range(world_size)]
    recv_info = _exchange_counts(send_info, rank, world_size, group, device)
    recv_node_counts = [recv_info[g][0] for g in range(world_size)]
    recv_edge_counts = [recv_info[g][1] for g in range(world_size)]
    recv_angle_counts = [recv_info[g][2] for g in range(world_size)]
    recv_splits = _compute_send_splits(recv_node_counts, recv_edge_counts, recv_angle_counts)

    # Dispatch A2A.
    recv_tensor = all_to_all_differentiable(packed, send_splits, recv_splits, group)
    check(recv_tensor.shape[0] == sum(recv_splits), "recv shape mismatch", rank)

    # Unpack from dispatch.
    node_recv, edge_recv, angle_recv = packer.unpack_from_dispatch(
        recv_tensor, recv_node_counts, recv_edge_counts, recv_angle_counts,
    )

    total_recv_node = sum(recv_node_counts)
    total_recv_edge = sum(recv_edge_counts)
    total_recv_angle = sum(recv_angle_counts)
    check(node_recv.shape == (total_recv_node, 28 * A_DIM),
          f"node_recv shape: {node_recv.shape}", rank)
    check(edge_recv.shape == (total_recv_edge, 10 * A_DIM),
          f"edge_recv shape: {edge_recv.shape}", rank)
    check(angle_recv.shape == (total_recv_angle, 4 * A_DIM),
          f"angle_recv shape: {angle_recv.shape}", rank)

    # Simulate identity expert: output = truncated input (node 28a→8a, edge 10a→6a, angle 4a→3a).
    node_out = node_recv[:, :8 * A_DIM]
    edge_out = edge_recv[:, :6 * A_DIM]
    angle_out = angle_recv[:, :3 * A_DIM]

    # Repack for combine.
    packed_out = packer.pack_for_combine(
        node_out, edge_out, angle_out,
        recv_node_counts, recv_edge_counts, recv_angle_counts,
    )
    # Row counts must match dispatch recv.
    check(packed_out.shape[0] == recv_tensor.shape[0],
          "combine rows != dispatch recv rows", rank)

    # Combine A2A (reverse direction).
    returned = all_to_all_differentiable(packed_out, recv_splits, send_splits, group)
    check(returned.shape[0] == sum(send_splits),
          f"returned shape: {returned.shape}", rank)

    # Unpack from combine.
    node_final, edge_final, angle_final = packer.unpack_from_combine(
        returned, node_counts, edge_counts, angle_counts,
    )
    check(node_final.shape == (n_node, 8 * A_DIM),
          f"node_final shape: {node_final.shape}", rank)
    check(edge_final.shape == (n_edge, 6 * A_DIM),
          f"edge_final shape: {edge_final.shape}", rank)
    check(angle_final.shape == (n_angle, 3 * A_DIM),
          f"angle_final shape: {angle_final.shape}", rank)

    # Verify data: node_final should be node_in[:, :8a], etc.
    check(
        torch.allclose(node_final, node_in[:, :8 * A_DIM]),
        f"node data mismatch: max diff = {(node_final - node_in[:, :8*A_DIM]).abs().max().item():.2e}",
        rank,
    )
    check(
        torch.allclose(edge_final, edge_in[:, :6 * A_DIM]),
        f"edge data mismatch: max diff = {(edge_final - edge_in[:, :6*A_DIM]).abs().max().item():.2e}",
        rank,
    )
    check(
        torch.allclose(angle_final, angle_in[:, :3 * A_DIM]),
        f"angle data mismatch: max diff = {(angle_final - angle_in[:, :3*A_DIM]).abs().max().item():.2e}",
        rank,
    )

    all_pass(rank, world_size, "test_full_roundtrip_2gpu")


# ---------------------------------------------------------------------------
# Test: Gradient through full pipeline (2 GPU)
# ---------------------------------------------------------------------------

def test_gradient_pipeline_2gpu(rank, world_size, group):
    """Verify gradients flow through pack → A2A → unpack → compute → repack → A2A → unpack."""
    assert world_size == 2
    device = torch.device(f"cuda:{rank}")
    packer = MoEPacker(A_DIM)

    if rank == 0:
        n_node, n_edge, n_angle = 4, 8, 10
    else:
        n_node, n_edge, n_angle = 6, 5, 3

    torch.manual_seed(2000 + rank)
    node_in = torch.randn(n_node, 28 * A_DIM, dtype=DTYPE, device=device, requires_grad=True)
    edge_in = torch.randn(n_edge, 10 * A_DIM, dtype=DTYPE, device=device, requires_grad=True)
    angle_in = torch.randn(n_angle, 4 * A_DIM, dtype=DTYPE, device=device, requires_grad=True)

    # Send everything to the other rank.
    if rank == 0:
        node_counts = [0, n_node]
        edge_counts = [0, n_edge]
        angle_counts = [0, n_angle]
    else:
        node_counts = [n_node, 0]
        edge_counts = [n_edge, 0]
        angle_counts = [n_angle, 0]

    packed, send_splits = packer.pack_for_dispatch(
        node_in, edge_in, angle_in,
        node_counts, edge_counts, angle_counts,
    )

    send_info = [[node_counts[g], edge_counts[g], angle_counts[g]]
                 for g in range(world_size)]
    recv_info = _exchange_counts(send_info, rank, world_size, group, device)
    recv_node_counts = [recv_info[g][0] for g in range(world_size)]
    recv_edge_counts = [recv_info[g][1] for g in range(world_size)]
    recv_angle_counts = [recv_info[g][2] for g in range(world_size)]
    recv_splits = _compute_send_splits(recv_node_counts, recv_edge_counts, recv_angle_counts)

    recv_tensor = all_to_all_differentiable(packed, send_splits, recv_splits, group)
    node_recv, edge_recv, angle_recv = packer.unpack_from_dispatch(
        recv_tensor, recv_node_counts, recv_edge_counts, recv_angle_counts,
    )

    # Differentiable "expert": scale by 2.
    node_out = node_recv[:, :8 * A_DIM] * 2.0
    edge_out = edge_recv[:, :6 * A_DIM] * 2.0
    angle_out = angle_recv[:, :3 * A_DIM] * 2.0

    packed_out = packer.pack_for_combine(
        node_out, edge_out, angle_out,
        recv_node_counts, recv_edge_counts, recv_angle_counts,
    )

    returned = all_to_all_differentiable(packed_out, recv_splits, send_splits, group)
    node_final, edge_final, angle_final = packer.unpack_from_combine(
        returned, node_counts, edge_counts, angle_counts,
    )

    loss = node_final.sum() + edge_final.sum() + angle_final.sum()
    loss.backward()

    check(node_in.grad is not None, "node_in grad is None", rank)
    check(node_in.grad.abs().sum() > 0, "node_in grad is all zeros", rank)
    check(edge_in.grad is not None, "edge_in grad is None", rank)
    check(edge_in.grad.abs().sum() > 0, "edge_in grad is all zeros", rank)
    check(angle_in.grad is not None, "angle_in grad is None", rank)
    check(angle_in.grad.abs().sum() > 0, "angle_in grad is all zeros", rank)

    all_pass(rank, world_size, "test_gradient_pipeline_2gpu")


# ---------------------------------------------------------------------------
# Test: 2nd-order derivative through pipeline (2 GPU)
# ---------------------------------------------------------------------------

def test_2nd_order_pipeline_2gpu(rank, world_size, group):
    """create_graph=True through pack → A2A → unpack → compute → repack → A2A → unpack."""
    assert world_size == 2
    device = torch.device(f"cuda:{rank}")
    packer = MoEPacker(A_DIM)

    if rank == 0:
        n_node, n_edge, n_angle = 3, 5, 7
    else:
        n_node, n_edge, n_angle = 4, 6, 2

    torch.manual_seed(3000 + rank)
    node_in = torch.randn(n_node, 28 * A_DIM, dtype=DTYPE, device=device, requires_grad=True)
    edge_in = torch.randn(n_edge, 10 * A_DIM, dtype=DTYPE, device=device, requires_grad=True)
    angle_in = torch.randn(n_angle, 4 * A_DIM, dtype=DTYPE, device=device, requires_grad=True)

    if rank == 0:
        node_counts = [0, n_node]
        edge_counts = [0, n_edge]
        angle_counts = [0, n_angle]
    else:
        node_counts = [n_node, 0]
        edge_counts = [n_edge, 0]
        angle_counts = [n_angle, 0]

    packed, send_splits = packer.pack_for_dispatch(
        node_in, edge_in, angle_in,
        node_counts, edge_counts, angle_counts,
    )

    send_info = [[node_counts[g], edge_counts[g], angle_counts[g]]
                 for g in range(world_size)]
    recv_info = _exchange_counts(send_info, rank, world_size, group, device)
    recv_node_counts = [recv_info[g][0] for g in range(world_size)]
    recv_edge_counts = [recv_info[g][1] for g in range(world_size)]
    recv_angle_counts = [recv_info[g][2] for g in range(world_size)]
    recv_splits = _compute_send_splits(recv_node_counts, recv_edge_counts, recv_angle_counts)

    recv_tensor = all_to_all_differentiable(packed, send_splits, recv_splits, group)
    node_recv, edge_recv, angle_recv = packer.unpack_from_dispatch(
        recv_tensor, recv_node_counts, recv_edge_counts, recv_angle_counts,
    )

    # Differentiable expert: square (non-trivial for 2nd-order).
    node_out = (node_recv[:, :8 * A_DIM]) ** 2
    edge_out = (edge_recv[:, :6 * A_DIM]) ** 2
    angle_out = (angle_recv[:, :3 * A_DIM]) ** 2

    packed_out = packer.pack_for_combine(
        node_out, edge_out, angle_out,
        recv_node_counts, recv_edge_counts, recv_angle_counts,
    )

    returned = all_to_all_differentiable(packed_out, recv_splits, send_splits, group)
    node_final, edge_final, angle_final = packer.unpack_from_combine(
        returned, node_counts, edge_counts, angle_counts,
    )

    loss = node_final.sum() + edge_final.sum() + angle_final.sum()

    # 1st backward with create_graph=True.
    inputs = [node_in, edge_in, angle_in]
    grads = torch.autograd.grad(loss, inputs, create_graph=True)
    for i, g in enumerate(grads):
        check(g is not None, f"1st grad[{i}] is None", rank)
        check(g.requires_grad, f"1st grad[{i}] not differentiable", rank)

    # 2nd backward.
    grad_sum = sum(g.sum() for g in grads)
    grad_sum.backward()

    for i, inp in enumerate(inputs):
        check(inp.grad is not None, f"2nd-order grad[{i}] is None", rank)
        check(inp.grad.abs().sum() > 0, f"2nd-order grad[{i}] is all zeros", rank)

    all_pass(rank, world_size, "test_2nd_order_pipeline_2gpu")


# ---------------------------------------------------------------------------
# Test: Asymmetric counts, some features empty (2 GPU)
# ---------------------------------------------------------------------------

def test_asymmetric_counts_2gpu(rank, world_size, group):
    """Each rank sends partial features: rank 0 sends only nodes+edges, rank 1 only angles."""
    assert world_size == 2
    device = torch.device(f"cuda:{rank}")
    packer = MoEPacker(A_DIM)

    if rank == 0:
        n_node, n_edge, n_angle = 6, 9, 0
    else:
        n_node, n_edge, n_angle = 0, 0, 15

    torch.manual_seed(4000 + rank)
    node_in = torch.randn(n_node, 28 * A_DIM, dtype=DTYPE, device=device)
    edge_in = torch.randn(n_edge, 10 * A_DIM, dtype=DTYPE, device=device)
    angle_in = torch.randn(n_angle, 4 * A_DIM, dtype=DTYPE, device=device)

    # Send all to the other rank.
    if rank == 0:
        node_counts = [0, n_node]
        edge_counts = [0, n_edge]
        angle_counts = [0, n_angle]
    else:
        node_counts = [n_node, 0]
        edge_counts = [n_edge, 0]
        angle_counts = [n_angle, 0]

    packed, send_splits = packer.pack_for_dispatch(
        node_in, edge_in, angle_in,
        node_counts, edge_counts, angle_counts,
    )

    send_info = [[node_counts[g], edge_counts[g], angle_counts[g]]
                 for g in range(world_size)]
    recv_info = _exchange_counts(send_info, rank, world_size, group, device)
    recv_node_counts = [recv_info[g][0] for g in range(world_size)]
    recv_edge_counts = [recv_info[g][1] for g in range(world_size)]
    recv_angle_counts = [recv_info[g][2] for g in range(world_size)]
    recv_splits = _compute_send_splits(recv_node_counts, recv_edge_counts, recv_angle_counts)

    recv_tensor = all_to_all_differentiable(packed, send_splits, recv_splits, group)
    node_recv, edge_recv, angle_recv = packer.unpack_from_dispatch(
        recv_tensor, recv_node_counts, recv_edge_counts, recv_angle_counts,
    )

    total_recv_node = sum(recv_node_counts)
    total_recv_edge = sum(recv_edge_counts)
    total_recv_angle = sum(recv_angle_counts)

    check(node_recv.shape == (total_recv_node, 28 * A_DIM),
          f"node_recv shape mismatch: {node_recv.shape}", rank)
    check(edge_recv.shape == (total_recv_edge, 10 * A_DIM),
          f"edge_recv shape mismatch: {edge_recv.shape}", rank)
    check(angle_recv.shape == (total_recv_angle, 4 * A_DIM),
          f"angle_recv shape mismatch: {angle_recv.shape}", rank)

    # Identity expert on output dims.
    node_out = node_recv[:, :8 * A_DIM]
    edge_out = edge_recv[:, :6 * A_DIM]
    angle_out = angle_recv[:, :3 * A_DIM]

    packed_out = packer.pack_for_combine(
        node_out, edge_out, angle_out,
        recv_node_counts, recv_edge_counts, recv_angle_counts,
    )

    returned = all_to_all_differentiable(packed_out, recv_splits, send_splits, group)
    node_final, edge_final, angle_final = packer.unpack_from_combine(
        returned, node_counts, edge_counts, angle_counts,
    )

    check(node_final.shape == (n_node, 8 * A_DIM),
          f"node_final shape: {node_final.shape}", rank)
    check(edge_final.shape == (n_edge, 6 * A_DIM),
          f"edge_final shape: {edge_final.shape}", rank)
    check(angle_final.shape == (n_angle, 3 * A_DIM),
          f"angle_final shape: {angle_final.shape}", rank)

    # Verify data roundtrip.
    if n_node > 0:
        check(
            torch.allclose(node_final, node_in[:, :8 * A_DIM]),
            "node data mismatch", rank,
        )
    if n_edge > 0:
        check(
            torch.allclose(edge_final, edge_in[:, :6 * A_DIM]),
            "edge data mismatch", rank,
        )
    if n_angle > 0:
        check(
            torch.allclose(angle_final, angle_in[:, :3 * A_DIM]),
            "angle data mismatch", rank,
        )

    all_pass(rank, world_size, "test_asymmetric_counts_2gpu")


# ---------------------------------------------------------------------------
# Test: Full roundtrip (4 GPU)
# ---------------------------------------------------------------------------

def test_full_roundtrip_4gpu(rank, world_size, group):
    """4-GPU full roundtrip: each rank sends data to ALL other ranks."""
    assert world_size == 4
    device = torch.device(f"cuda:{rank}")
    packer = MoEPacker(A_DIM)

    # Each rank has different amounts of data.
    per_rank = {
        0: (5, 7, 13),
        1: (3, 12, 4),
        2: (8, 1, 20),
        3: (2, 9, 0),
    }
    n_node, n_edge, n_angle = per_rank[rank]

    torch.manual_seed(5000 + rank)
    node_in = torch.randn(n_node, 28 * A_DIM, dtype=DTYPE, device=device)
    edge_in = torch.randn(n_edge, 10 * A_DIM, dtype=DTYPE, device=device)
    angle_in = torch.randn(n_angle, 4 * A_DIM, dtype=DTYPE, device=device)

    # Distribute evenly: send roughly equal amounts to each GPU.
    # For simplicity: all tokens go to GPU (rank+1) % world_size.
    dest = (rank + 1) % world_size
    node_counts = [0] * world_size
    edge_counts = [0] * world_size
    angle_counts = [0] * world_size
    node_counts[dest] = n_node
    edge_counts[dest] = n_edge
    angle_counts[dest] = n_angle

    packed, send_splits = packer.pack_for_dispatch(
        node_in, edge_in, angle_in,
        node_counts, edge_counts, angle_counts,
    )

    send_info = [[node_counts[g], edge_counts[g], angle_counts[g]]
                 for g in range(world_size)]
    recv_info = _exchange_counts(send_info, rank, world_size, group, device)
    recv_node_counts = [recv_info[g][0] for g in range(world_size)]
    recv_edge_counts = [recv_info[g][1] for g in range(world_size)]
    recv_angle_counts = [recv_info[g][2] for g in range(world_size)]
    recv_splits = _compute_send_splits(recv_node_counts, recv_edge_counts, recv_angle_counts)

    recv_tensor = all_to_all_differentiable(packed, send_splits, recv_splits, group)
    node_recv, edge_recv, angle_recv = packer.unpack_from_dispatch(
        recv_tensor, recv_node_counts, recv_edge_counts, recv_angle_counts,
    )

    # Identity expert output.
    node_out = node_recv[:, :8 * A_DIM]
    edge_out = edge_recv[:, :6 * A_DIM]
    angle_out = angle_recv[:, :3 * A_DIM]

    packed_out = packer.pack_for_combine(
        node_out, edge_out, angle_out,
        recv_node_counts, recv_edge_counts, recv_angle_counts,
    )

    returned = all_to_all_differentiable(packed_out, recv_splits, send_splits, group)
    node_final, edge_final, angle_final = packer.unpack_from_combine(
        returned, node_counts, edge_counts, angle_counts,
    )

    check(node_final.shape == (n_node, 8 * A_DIM),
          f"node_final shape: {node_final.shape}", rank)
    check(edge_final.shape == (n_edge, 6 * A_DIM),
          f"edge_final shape: {edge_final.shape}", rank)
    check(angle_final.shape == (n_angle, 3 * A_DIM),
          f"angle_final shape: {angle_final.shape}", rank)

    if n_node > 0:
        check(torch.allclose(node_final, node_in[:, :8 * A_DIM]),
              "node data mismatch", rank)
    if n_edge > 0:
        check(torch.allclose(edge_final, edge_in[:, :6 * A_DIM]),
              "edge data mismatch", rank)
    if n_angle > 0:
        check(torch.allclose(angle_final, angle_in[:, :3 * A_DIM]),
              "angle data mismatch", rank)

    all_pass(rank, world_size, "test_full_roundtrip_4gpu")


# ---------------------------------------------------------------------------
# Test: 2nd-order derivative (4 GPU)
# ---------------------------------------------------------------------------

def test_2nd_order_4gpu(rank, world_size, group):
    """2nd-order derivative through pack → A2A → expert → A2A → unpack with 4 GPUs."""
    assert world_size == 4
    device = torch.device(f"cuda:{rank}")
    packer = MoEPacker(A_DIM)

    per_rank = {0: (3, 5, 7), 1: (4, 6, 2), 2: (2, 8, 11), 3: (5, 3, 0)}
    n_node, n_edge, n_angle = per_rank[rank]

    torch.manual_seed(6000 + rank)
    node_in = torch.randn(n_node, 28 * A_DIM, dtype=DTYPE, device=device, requires_grad=True)
    edge_in = torch.randn(n_edge, 10 * A_DIM, dtype=DTYPE, device=device, requires_grad=True)
    angle_in = torch.randn(n_angle, 4 * A_DIM, dtype=DTYPE, device=device, requires_grad=True)

    dest = (rank + 1) % world_size
    node_counts = [0] * world_size
    edge_counts = [0] * world_size
    angle_counts = [0] * world_size
    node_counts[dest] = n_node
    edge_counts[dest] = n_edge
    angle_counts[dest] = n_angle

    packed, send_splits = packer.pack_for_dispatch(
        node_in, edge_in, angle_in,
        node_counts, edge_counts, angle_counts,
    )

    send_info = [[node_counts[g], edge_counts[g], angle_counts[g]]
                 for g in range(world_size)]
    recv_info = _exchange_counts(send_info, rank, world_size, group, device)
    recv_node_counts = [recv_info[g][0] for g in range(world_size)]
    recv_edge_counts = [recv_info[g][1] for g in range(world_size)]
    recv_angle_counts = [recv_info[g][2] for g in range(world_size)]
    recv_splits = _compute_send_splits(recv_node_counts, recv_edge_counts, recv_angle_counts)

    recv_tensor = all_to_all_differentiable(packed, send_splits, recv_splits, group)
    node_recv, edge_recv, angle_recv = packer.unpack_from_dispatch(
        recv_tensor, recv_node_counts, recv_edge_counts, recv_angle_counts,
    )

    # Non-trivial expert: square for 2nd-order.
    node_out = (node_recv[:, :8 * A_DIM]) ** 2
    edge_out = (edge_recv[:, :6 * A_DIM]) ** 2
    angle_out = (angle_recv[:, :3 * A_DIM]) ** 2

    packed_out = packer.pack_for_combine(
        node_out, edge_out, angle_out,
        recv_node_counts, recv_edge_counts, recv_angle_counts,
    )

    returned = all_to_all_differentiable(packed_out, recv_splits, send_splits, group)
    node_final, edge_final, angle_final = packer.unpack_from_combine(
        returned, node_counts, edge_counts, angle_counts,
    )

    loss = node_final.sum() + edge_final.sum() + angle_final.sum()

    inputs = [inp for inp in [node_in, edge_in, angle_in] if inp.shape[0] > 0]
    if inputs:
        grads = torch.autograd.grad(loss, inputs, create_graph=True)
        for i, g in enumerate(grads):
            check(g is not None, f"1st grad[{i}] is None", rank)
            check(g.requires_grad, f"1st grad[{i}] not differentiable", rank)

        grad_sum = sum(g.sum() for g in grads)
        grad_sum.backward()

        for i, inp in enumerate(inputs):
            check(inp.grad is not None, f"2nd-order grad[{i}] is None", rank)

    all_pass(rank, world_size, "test_2nd_order_4gpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rank, world_size = setup_dist()
    group = dist.group.WORLD

    if rank == 0:
        print(f"Running moe_packer multi-GPU tests with {world_size} GPUs", flush=True)

    try:
        if world_size == 2:
            test_full_roundtrip_2gpu(rank, world_size, group)
            test_gradient_pipeline_2gpu(rank, world_size, group)
            test_2nd_order_pipeline_2gpu(rank, world_size, group)
            test_asymmetric_counts_2gpu(rank, world_size, group)

        if world_size == 4:
            test_full_roundtrip_4gpu(rank, world_size, group)
            test_2nd_order_4gpu(rank, world_size, group)

        if rank == 0:
            print(f"\nAll tests passed with {world_size} GPUs!", flush=True)
    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()
