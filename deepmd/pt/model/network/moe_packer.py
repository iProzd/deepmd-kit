# SPDX-License-Identifier: LGPL-3.0-or-later
"""MoE Feature Packer for Expert Parallelism.

Provides:
- ``validate_dim_ratio``: check n_dim:e_dim:a_dim = 4:2:1
- ``MoEPacker``: pure formatting pack/unpack for All-to-All communication.
  Completely unaware of routing, experts, or topk.
- ``exchange_metadata``: exchange per-GPU send counts via All-to-All.

Packing rules (a = a_dim, base unit):

Input (40a per row):
  Node  [N, 28a] → pad +12a → [N, 40a],            1 row per node
  Edge  [N, 10a] → 4 concat  → [ceil(N/4), 40a],    1 row per 4 edges
  Angle [N, 4a]  → 10 concat → [ceil(N/10), 40a],   1 row per 10 angles

Output (30a per row):
  Node  [N, 8a]  → pad +22a → [N, 30a],             1 row per node
  Edge  [N, 6a]  → 4 concat → 24a, pad +6a → [ceil(N/4), 30a]
  Angle [N, 3a]  → 10 concat → [ceil(N/10), 30a],   exact fit
"""

from __future__ import annotations

from typing import (
    Optional,
)

import torch
import torch.distributed as dist
import torch.nn.functional as F


def validate_dim_ratio(n_dim: int, e_dim: int, a_dim: int) -> None:
    """Validate n_dim:e_dim:a_dim = 4:2:1.

    Parameters
    ----------
    n_dim, e_dim, a_dim : int
        Feature dimensions.

    Raises
    ------
    ValueError
        If the ratio is not 4:2:1.
    """
    if not (n_dim == 4 * a_dim and e_dim == 2 * a_dim):
        raise ValueError(
            f"MoE requires n_dim:e_dim:a_dim = 4:2:1, "
            f"got {n_dim}:{e_dim}:{a_dim}"
        )


def _ceildiv(a: int, b: int) -> int:
    """Integer ceiling division."""
    return (a + b - 1) // b


def _concat_pad_groups(
    tensor: torch.Tensor,
    group_size: int,
    packed_width: int,
) -> torch.Tensor:
    """Concat ``group_size`` consecutive rows and pad to ``packed_width``.

    Parameters
    ----------
    tensor : Tensor, shape ``[N, feat_dim]``
    group_size : int
        Number of rows to concatenate per group.
    packed_width : int
        Target column width after padding.

    Returns
    -------
    Tensor, shape ``[ceil(N/group_size), packed_width]``
    """
    n, feat_dim = tensor.shape
    concat_width = group_size * feat_dim
    assert packed_width >= concat_width, (
        f"packed_width {packed_width} < concat_width {concat_width}"
    )

    if n == 0:
        return tensor.new_zeros(0, packed_width)

    # Pad rows to a multiple of group_size.
    n_groups = _ceildiv(n, group_size)
    n_padded = n_groups * group_size
    if n_padded > n:
        # F.pad pads last dim first; we want to pad dim=0 (rows).
        # pad = (left, right, top, bottom)
        tensor = F.pad(tensor, (0, 0, 0, n_padded - n))

    # Reshape: [n_groups, group_size * feat_dim]
    grouped = tensor.reshape(n_groups, concat_width)

    # Pad columns if needed.
    col_pad = packed_width - concat_width
    if col_pad > 0:
        grouped = F.pad(grouped, (0, col_pad))

    return grouped


def _split_unpad_groups(
    packed: torch.Tensor,
    group_size: int,
    feat_dim: int,
    n_valid: int,
) -> torch.Tensor:
    """Inverse of ``_concat_pad_groups``.

    Parameters
    ----------
    packed : Tensor, shape ``[N_groups, packed_width]``
    group_size : int
    feat_dim : int
        Original per-item feature dimension.
    n_valid : int
        Number of valid items to keep (removing padding rows).

    Returns
    -------
    Tensor, shape ``[n_valid, feat_dim]``
    """
    if packed.shape[0] == 0 or n_valid == 0:
        return packed.new_zeros(0, feat_dim)

    concat_width = group_size * feat_dim
    # Remove column padding, then reshape to individual items.
    items = packed[:, :concat_width].reshape(-1, feat_dim)
    # Remove row padding.
    return items[:n_valid]


class MoEPacker:
    """Pure formatting packer for MoE All-to-All communication.

    Does not know about routing, experts, or topk. Receives pre-sorted
    tensors and per-GPU counts from the upper layer
    (``MoEDispatchCombine``).

    Parameters
    ----------
    a_dim : int
        Base unit dimension (a_dim). All feature dims are multiples of this.
    """

    def __init__(self, a_dim: int) -> None:
        self.a = a_dim

        # Input dims (dispatch).
        self.D_node_in = 28 * a_dim
        self.D_edge_in = 10 * a_dim
        self.D_angle_in = 4 * a_dim
        self.D_packed_in = 40 * a_dim
        self.edge_concat_in = 4   # 4 × 10a = 40a
        self.angle_concat_in = 10  # 10 × 4a = 40a

        # Output dims (combine).
        self.D_node_out = 8 * a_dim
        self.D_edge_out = 6 * a_dim
        self.D_angle_out = 3 * a_dim
        self.D_packed_out = 30 * a_dim
        self.edge_concat_out = 4   # 4 × 6a = 24a → pad to 30a
        self.angle_concat_out = 10  # 10 × 3a = 30a exact

    # ------------------------------------------------------------------
    # Dispatch packing (input side, 40a per row)
    # ------------------------------------------------------------------

    def pack_for_dispatch(
        self,
        node_input_sorted: torch.Tensor,   # [N_node_exp, 28a]
        edge_input_sorted: torch.Tensor,   # [N_edge_exp, 10a]
        angle_input_sorted: torch.Tensor,  # [N_angle_exp, 4a]
        node_counts_per_gpu: list[int],    # [ep_size]
        edge_counts_per_gpu: list[int],    # [ep_size]
        angle_counts_per_gpu: list[int],   # [ep_size]
    ) -> tuple[torch.Tensor, list[int]]:
        """Pack sorted features into ``[N_total_rows, 40a]``.

        Parameters
        ----------
        node_input_sorted : Tensor, shape ``[sum(node_counts), 28a]``
        edge_input_sorted : Tensor, shape ``[sum(edge_counts), 10a]``
        angle_input_sorted : Tensor, shape ``[sum(angle_counts), 4a]``
        node_counts_per_gpu : list[int]
            Number of node tokens destined for each GPU.
        edge_counts_per_gpu : list[int]
            Number of edge tokens destined for each GPU.
        angle_counts_per_gpu : list[int]
            Number of angle tokens destined for each GPU.

        Returns
        -------
        packed : Tensor, shape ``[N_total_rows, 40a]``
        send_splits : list[int], length ``ep_size``
            Number of rows destined for each GPU.
        """
        ep_size = len(node_counts_per_gpu)
        blocks: list[torch.Tensor] = []
        send_splits: list[int] = []

        node_offset = 0
        edge_offset = 0
        angle_offset = 0

        for g in range(ep_size):
            n_node = node_counts_per_gpu[g]
            n_edge = edge_counts_per_gpu[g]
            n_angle = angle_counts_per_gpu[g]

            gpu_parts: list[torch.Tensor] = []

            # Node: [n_node, 28a] → pad to [n_node, 40a]
            if n_node > 0:
                node_slice = node_input_sorted[node_offset:node_offset + n_node]
                node_padded = F.pad(node_slice, (0, self.D_packed_in - self.D_node_in))
                gpu_parts.append(node_padded)
            node_offset += n_node

            # Edge: [n_edge, 10a] → groups of 4 → [ceil(n_edge/4), 40a]
            if n_edge > 0:
                edge_slice = edge_input_sorted[edge_offset:edge_offset + n_edge]
                edge_packed = _concat_pad_groups(
                    edge_slice, self.edge_concat_in, self.D_packed_in
                )
                gpu_parts.append(edge_packed)
            edge_offset += n_edge

            # Angle: [n_angle, 4a] → groups of 10 → [ceil(n_angle/10), 40a]
            if n_angle > 0:
                angle_slice = angle_input_sorted[angle_offset:angle_offset + n_angle]
                angle_packed = _concat_pad_groups(
                    angle_slice, self.angle_concat_in, self.D_packed_in
                )
                gpu_parts.append(angle_packed)
            angle_offset += n_angle

            # Row count for this GPU.
            n_edge_rows = _ceildiv(n_edge, self.edge_concat_in) if n_edge > 0 else 0
            n_angle_rows = _ceildiv(n_angle, self.angle_concat_in) if n_angle > 0 else 0
            send_splits.append(n_node + n_edge_rows + n_angle_rows)

            if gpu_parts:
                blocks.append(torch.cat(gpu_parts, dim=0))

        if blocks:
            packed = torch.cat(blocks, dim=0)
        else:
            packed = node_input_sorted.new_zeros(0, self.D_packed_in)

        return packed, send_splits

    # ------------------------------------------------------------------
    # Dispatch unpacking (recv side, 40a per row → individual features)
    # ------------------------------------------------------------------

    def unpack_from_dispatch(
        self,
        recv_tensor: torch.Tensor,    # [N_total_recv, 40a]
        node_counts: list[int],        # [ep_size] per-source-GPU node counts
        edge_counts: list[int],        # [ep_size] per-source-GPU edge counts
        angle_counts: list[int],       # [ep_size] per-source-GPU angle counts
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack received tensor into node/edge/angle features.

        Parameters
        ----------
        recv_tensor : Tensor, shape ``[N_total_recv, 40a]``
        node_counts, edge_counts, angle_counts : list[int]
            Per-source-GPU original (unpadded) token counts.

        Returns
        -------
        node_inputs : Tensor ``[N_node_recv, 28a]``
        edge_inputs : Tensor ``[N_edge_recv, 10a]``
        angle_inputs : Tensor ``[N_angle_recv, 4a]``
        """
        node_parts: list[torch.Tensor] = []
        edge_parts: list[torch.Tensor] = []
        angle_parts: list[torch.Tensor] = []

        row_offset = 0
        ep_size = len(node_counts)

        for g in range(ep_size):
            n_node = node_counts[g]
            n_edge = edge_counts[g]
            n_angle = angle_counts[g]
            n_edge_rows = _ceildiv(n_edge, self.edge_concat_in) if n_edge > 0 else 0
            n_angle_rows = _ceildiv(n_angle, self.angle_concat_in) if n_angle > 0 else 0

            # Node rows: slice columns [:28a]
            if n_node > 0:
                node_rows = recv_tensor[row_offset:row_offset + n_node]
                node_parts.append(node_rows[:, :self.D_node_in])
            row_offset += n_node

            # Edge rows: _split_unpad_groups
            if n_edge_rows > 0:
                edge_rows = recv_tensor[row_offset:row_offset + n_edge_rows]
                edge_parts.append(
                    _split_unpad_groups(
                        edge_rows, self.edge_concat_in, self.D_edge_in, n_edge
                    )
                )
            row_offset += n_edge_rows

            # Angle rows: _split_unpad_groups
            if n_angle_rows > 0:
                angle_rows = recv_tensor[row_offset:row_offset + n_angle_rows]
                angle_parts.append(
                    _split_unpad_groups(
                        angle_rows, self.angle_concat_in, self.D_angle_in, n_angle
                    )
                )
            row_offset += n_angle_rows

        node_out = torch.cat(node_parts, dim=0) if node_parts else recv_tensor.new_zeros(0, self.D_node_in)
        edge_out = torch.cat(edge_parts, dim=0) if edge_parts else recv_tensor.new_zeros(0, self.D_edge_in)
        angle_out = torch.cat(angle_parts, dim=0) if angle_parts else recv_tensor.new_zeros(0, self.D_angle_in)

        return node_out, edge_out, angle_out

    # ------------------------------------------------------------------
    # Combine packing (output side, 30a per row)
    # ------------------------------------------------------------------

    def pack_for_combine(
        self,
        node_output: torch.Tensor,     # [N_node_recv, 8a]
        edge_output: torch.Tensor,     # [N_edge_recv, 6a]
        angle_output: torch.Tensor,    # [N_angle_recv, 3a]
        node_counts: list[int],
        edge_counts: list[int],
        angle_counts: list[int],
    ) -> torch.Tensor:
        """Pack expert outputs into ``[N_total_rows, 30a]``.

        Row count is identical to the dispatch recv row count.

        Parameters
        ----------
        node_output : Tensor ``[sum(node_counts), 8a]``
        edge_output : Tensor ``[sum(edge_counts), 6a]``
        angle_output : Tensor ``[sum(angle_counts), 3a]``
        node_counts, edge_counts, angle_counts : list[int]
            Per-source-GPU token counts (same as used in unpack_from_dispatch).

        Returns
        -------
        packed : Tensor ``[N_total_rows, 30a]``
        """
        ep_size = len(node_counts)
        blocks: list[torch.Tensor] = []

        node_offset = 0
        edge_offset = 0
        angle_offset = 0

        for g in range(ep_size):
            n_node = node_counts[g]
            n_edge = edge_counts[g]
            n_angle = angle_counts[g]

            gpu_parts: list[torch.Tensor] = []

            # Node: [n_node, 8a] → pad to [n_node, 30a]
            if n_node > 0:
                node_slice = node_output[node_offset:node_offset + n_node]
                node_padded = F.pad(node_slice, (0, self.D_packed_out - self.D_node_out))
                gpu_parts.append(node_padded)
            node_offset += n_node

            # Edge: [n_edge, 6a] → groups of 4 → pad to 30a
            if n_edge > 0:
                edge_slice = edge_output[edge_offset:edge_offset + n_edge]
                edge_packed = _concat_pad_groups(
                    edge_slice, self.edge_concat_out, self.D_packed_out
                )
                gpu_parts.append(edge_packed)
            edge_offset += n_edge

            # Angle: [n_angle, 3a] → groups of 10 → 30a (exact fit)
            if n_angle > 0:
                angle_slice = angle_output[angle_offset:angle_offset + n_angle]
                angle_packed = _concat_pad_groups(
                    angle_slice, self.angle_concat_out, self.D_packed_out
                )
                gpu_parts.append(angle_packed)
            angle_offset += n_angle

            if gpu_parts:
                blocks.append(torch.cat(gpu_parts, dim=0))

        if blocks:
            return torch.cat(blocks, dim=0)
        else:
            return node_output.new_zeros(0, self.D_packed_out)

    # ------------------------------------------------------------------
    # Combine unpacking (returned side, 30a per row → individual outputs)
    # ------------------------------------------------------------------

    def unpack_from_combine(
        self,
        returned: torch.Tensor,        # [N_total_send, 30a]
        node_counts: list[int],
        edge_counts: list[int],
        angle_counts: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack returned tensor into node/edge/angle outputs.

        Parameters
        ----------
        returned : Tensor ``[N_total_send, 30a]``
        node_counts, edge_counts, angle_counts : list[int]
            Per-GPU token counts (same as used in pack_for_dispatch).

        Returns
        -------
        node_output : Tensor ``[N_node_exp, 8a]``
        edge_output : Tensor ``[N_edge_exp, 6a]``
        angle_output : Tensor ``[N_angle_exp, 3a]``
        """
        node_parts: list[torch.Tensor] = []
        edge_parts: list[torch.Tensor] = []
        angle_parts: list[torch.Tensor] = []

        row_offset = 0
        ep_size = len(node_counts)

        for g in range(ep_size):
            n_node = node_counts[g]
            n_edge = edge_counts[g]
            n_angle = angle_counts[g]
            n_edge_rows = _ceildiv(n_edge, self.edge_concat_out) if n_edge > 0 else 0
            n_angle_rows = _ceildiv(n_angle, self.angle_concat_out) if n_angle > 0 else 0

            # Node rows: slice columns [:8a]
            if n_node > 0:
                node_rows = returned[row_offset:row_offset + n_node]
                node_parts.append(node_rows[:, :self.D_node_out])
            row_offset += n_node

            # Edge rows
            if n_edge_rows > 0:
                edge_rows = returned[row_offset:row_offset + n_edge_rows]
                edge_parts.append(
                    _split_unpad_groups(
                        edge_rows, self.edge_concat_out, self.D_edge_out, n_edge
                    )
                )
            row_offset += n_edge_rows

            # Angle rows
            if n_angle_rows > 0:
                angle_rows = returned[row_offset:row_offset + n_angle_rows]
                angle_parts.append(
                    _split_unpad_groups(
                        angle_rows, self.angle_concat_out, self.D_angle_out, n_angle
                    )
                )
            row_offset += n_angle_rows

        node_out = torch.cat(node_parts, dim=0) if node_parts else returned.new_zeros(0, self.D_node_out)
        edge_out = torch.cat(edge_parts, dim=0) if edge_parts else returned.new_zeros(0, self.D_edge_out)
        angle_out = torch.cat(angle_parts, dim=0) if angle_parts else returned.new_zeros(0, self.D_angle_out)

        return node_out, edge_out, angle_out


# ======================================================================
# Step 5: Metadata exchange
# ======================================================================


def exchange_metadata(
    send_info: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    """Exchange per-GPU metadata (token counts) via All-to-All.

    Each GPU prepares ``send_info[g] = (node_count, edge_count, angle_count)``
    describing what it will send to GPU *g*.  After the exchange,
    ``recv_info[g]`` holds what GPU *g* will send to the current GPU.

    This is a non-differentiable integer exchange used to set up the
    ``send_splits`` / ``recv_splits`` for the subsequent data All-to-All.

    Parameters
    ----------
    send_info : Tensor, shape ``[ep_size, n_fields]``, dtype int64
        Row *g* contains the metadata this GPU intends to send to GPU *g*.
        Typical fields: ``(node_count, edge_count, angle_count)``.
    ep_group : ProcessGroup or None
        Expert-parallelism communication group.
        When ``None`` (single-GPU / no EP), *send_info* is returned as-is.

    Returns
    -------
    recv_info : Tensor, shape ``[ep_size, n_fields]``, dtype int64
        Row *g* contains the metadata GPU *g* will send to this GPU.
    """
    if ep_group is None:
        return send_info

    ep_size = dist.get_world_size(group=ep_group)

    recv_info = torch.empty_like(send_info)
    send_list = list(send_info.chunk(ep_size, dim=0))
    recv_list = list(recv_info.chunk(ep_size, dim=0))
    dist.all_to_all(recv_list, send_list, group=ep_group)

    return torch.cat(recv_list, dim=0)


def counts_to_packed_rows(
    node_counts: list[int],
    edge_counts: list[int],
    angle_counts: list[int],
    edge_group_size: int = 4,
    angle_group_size: int = 10,
) -> list[int]:
    """Convert per-GPU token counts to packed row counts.

    Given the number of node / edge / angle tokens for each GPU, compute
    the total number of packed rows destined for (or received from) each GPU.

    Parameters
    ----------
    node_counts, edge_counts, angle_counts : list[int]
        Per-GPU token counts (length ``ep_size``).
    edge_group_size : int
        Number of edges packed per row (default 4).
    angle_group_size : int
        Number of angles packed per row (default 10).

    Returns
    -------
    list[int]
        Packed row count per GPU (length ``ep_size``).
    """
    splits: list[int] = []
    for g in range(len(node_counts)):
        n_n = node_counts[g]
        n_e = _ceildiv(edge_counts[g], edge_group_size) if edge_counts[g] > 0 else 0
        n_a = _ceildiv(angle_counts[g], angle_group_size) if angle_counts[g] > 0 else 0
        splits.append(n_n + n_e + n_a)
    return splits
