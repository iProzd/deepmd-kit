# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fused CUDA implementation of MoE pack-for-dispatch.

Drop-in replacement for ``MoEPacker.pack_for_dispatch``.  Builds the per-GPU
input/output offset tables on CPU (cheap, ep_size is small), uploads them
once to the GPU, then calls a single fused CUDA op that scatters the three
sorted feature tensors into the [total_rows, D_packed] layout consumed by
the All-to-All.
"""

from __future__ import annotations

import torch

import deepmd.pt.cxx_op  # noqa: F401  registers torch.ops.deepmd


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def fused_pack_for_dispatch(
    node_sorted: torch.Tensor,
    edge_sorted: torch.Tensor,
    angle_sorted: torch.Tensor,
    node_counts: list[int],
    edge_counts: list[int],
    angle_counts: list[int],
    edge_concat: int,
    angle_concat: int,
    D_packed: int,
) -> tuple[torch.Tensor, list[int]]:
    """Fused CUDA pack-for-dispatch.

    Parameters
    ----------
    node_sorted, edge_sorted, angle_sorted : Tensor
        Sorted feature tensors, shapes
        ``[sum(node_counts), 28a]``, ``[sum(edge_counts), 10a]``,
        ``[sum(angle_counts), 4a]``.
    node_counts, edge_counts, angle_counts : list[int]
        Per-GPU token counts (length ``ep_size``).
    edge_concat, angle_concat : int
        Group sizes for edge/angle concatenation (4 and 10 by default).
    D_packed : int
        Output row width (``40 * a_dim``).

    Returns
    -------
    packed : Tensor ``[total_rows, D_packed]``
    send_splits : list[int]
        Number of rows destined for each GPU.
    """
    ep_size = len(node_counts)
    assert len(edge_counts) == ep_size and len(angle_counts) == ep_size

    # Build offset tables on CPU first — ep_size is small (single digits to
    # ~tens), so the Python loop is negligible compared to the GPU work it
    # replaces.
    node_in_off = [0]
    edge_in_off = [0]
    angle_in_off = [0]
    node_out_off: list[int] = []
    edge_out_off: list[int] = []
    angle_out_off: list[int] = []
    send_splits: list[int] = []
    total_rows = 0
    for g in range(ep_size):
        nn, ne, na = node_counts[g], edge_counts[g], angle_counts[g]
        n_e_rows = _ceildiv(ne, edge_concat) if ne > 0 else 0
        n_a_rows = _ceildiv(na, angle_concat) if na > 0 else 0
        node_out_off.append(total_rows)
        edge_out_off.append(total_rows + nn)
        angle_out_off.append(total_rows + nn + n_e_rows)
        gpu_rows = nn + n_e_rows + n_a_rows
        send_splits.append(gpu_rows)
        total_rows += gpu_rows
        node_in_off.append(node_in_off[-1] + nn)
        edge_in_off.append(edge_in_off[-1] + ne)
        angle_in_off.append(angle_in_off[-1] + na)

    device = node_sorted.device
    int_opt = dict(dtype=torch.int64, device=device)
    node_in_t = torch.tensor(node_in_off, **int_opt)
    edge_in_t = torch.tensor(edge_in_off, **int_opt)
    angle_in_t = torch.tensor(angle_in_off, **int_opt)
    node_out_t = torch.tensor(node_out_off, **int_opt)
    edge_out_t = torch.tensor(edge_out_off, **int_opt)
    angle_out_t = torch.tensor(angle_out_off, **int_opt)

    packed = torch.ops.deepmd.moe_pack_for_dispatch(
        node_sorted, edge_sorted, angle_sorted,
        node_in_t, edge_in_t, angle_in_t,
        node_out_t, edge_out_t, angle_out_t,
        int(total_rows), int(ep_size), int(D_packed),
        int(edge_concat), int(angle_concat),
    )
    return packed, send_splits
