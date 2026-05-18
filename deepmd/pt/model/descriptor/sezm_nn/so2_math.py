# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared SO(2) m-major layout and block application helpers."""

from __future__ import (
    annotations,
)

import torch


def build_m_major_layout(
    lmax: int,
    mmax: int,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Build the m-major reduced layout used by SeZM SO(2) layers."""
    m0_size = int(lmax) + 1
    m0_idx = torch.arange(m0_size, device=device, dtype=torch.long)
    pos_indices_list: list[torch.Tensor] = []
    neg_indices_list: list[torch.Tensor] = []
    m_ranges: list[tuple[int, int, int]] = []

    offset = m0_size
    for m in range(1, int(mmax) + 1):
        num_l = int(lmax) - m + 1
        neg_start = offset
        pos_start = offset + num_l
        neg_indices_list.append(
            torch.arange(neg_start, neg_start + num_l, device=device, dtype=torch.long)
        )
        pos_indices_list.append(
            torch.arange(pos_start, pos_start + num_l, device=device, dtype=torch.long)
        )
        m_ranges.append((neg_start, pos_start, num_l))
        offset += 2 * num_l

    if pos_indices_list:
        pos_indices = torch.cat(pos_indices_list)
        neg_indices = torch.cat(neg_indices_list)
    else:
        pos_indices = torch.empty(0, device=device, dtype=torch.long)
        neg_indices = torch.empty(0, device=device, dtype=torch.long)

    block_slices: list[tuple[int, int, int, int, int, int, int, int]] = []
    for neg_start, pos_start, num_l in m_ranges:
        block_slices.append(
            (
                neg_start,
                neg_start + num_l,
                pos_start,
                pos_start + num_l,
                neg_start,
                neg_start + num_l,
                pos_start,
                pos_start + num_l,
            )
        )

    return {
        "m0_idx": m0_idx,
        "pos_indices": pos_indices,
        "neg_indices": neg_indices,
        "m_ranges": m_ranges,
        "block_slices": block_slices,
        "reduced_dim": int(offset),
        "m0_size": m0_size,
    }


def build_so2_weight(
    *,
    weight_m0: torch.Tensor,
    weight_m: list[torch.Tensor],
    reduced_dim: int,
    in_channels: int,
    out_channels: int,
    n_focus: int,
    m0_size: int,
    block_slices: list[tuple[int, int, int, int, int, int, int, int]],
) -> torch.Tensor:
    """Assemble a full block-diagonal SO(2) weight tensor."""
    in_total = reduced_dim * in_channels
    out_total = reduced_dim * out_channels
    weight = weight_m0.new_zeros(in_total, n_focus, out_total)
    m0_in = m0_size * in_channels
    m0_out = m0_size * out_channels
    weight[0:m0_in, :, 0:m0_out] = weight_m0.view(m0_in, n_focus, m0_out)

    for m_idx, w in enumerate(weight_m):
        ni0, ni1, pi0, pi1, no0, no1, po0, po1 = block_slices[m_idx]
        ib = ni1 - ni0
        ob = no1 - no0
        w = w.view(ib, n_focus, 2 * ob)
        w_u = w[:, :, :ob]
        w_v = w[:, :, ob:]
        weight[ni0:ni1, :, no0:no1] = w_u
        weight[ni0:ni1, :, po0:po1] = w_v
        weight[pi0:pi1, :, no0:no1] = -w_v
        weight[pi0:pi1, :, po0:po1] = w_u
    return weight


def apply_so2_blocks_one(
    x: torch.Tensor,
    *,
    matrix_m0: torch.Tensor,
    matrices_m: list[torch.Tensor],
    m0_size: int,
    m_ranges: list[tuple[int, int, int]],
    out_channels: int,
) -> torch.Tensor:
    """Apply SO(2) blocks for one expert to ``x`` with shape ``(N,D,C)``."""
    n_token = x.shape[0]
    out = x.new_empty(n_token, x.shape[1], out_channels)
    m0_in = matrix_m0.shape[0]
    x_m0 = x[:, :m0_size, :].reshape(n_token, m0_in)
    y_m0 = x_m0.matmul(matrix_m0)
    out[:, :m0_size, :] = y_m0.reshape(n_token, m0_size, out_channels)

    for m_idx, (neg_start, pos_start, num_l) in enumerate(m_ranges):
        block_in = matrices_m[m_idx].shape[0]
        block_out = matrices_m[m_idx].shape[1] // 2
        w_u = matrices_m[m_idx][:, :block_out]
        w_v = matrices_m[m_idx][:, block_out:]
        x_neg = x[:, neg_start : neg_start + num_l, :].reshape(n_token, block_in)
        x_pos = x[:, pos_start : pos_start + num_l, :].reshape(n_token, block_in)
        y_neg = x_neg.matmul(w_u) - x_pos.matmul(w_v)
        y_pos = x_neg.matmul(w_v) + x_pos.matmul(w_u)
        out[:, neg_start : neg_start + num_l, :] = y_neg.reshape(
            n_token, num_l, out_channels
        )
        out[:, pos_start : pos_start + num_l, :] = y_pos.reshape(
            n_token, num_l, out_channels
        )
    return out


def apply_so2_blocks_batched(
    x: torch.Tensor,
    *,
    matrix_m0: torch.Tensor,
    matrices_m: list[torch.Tensor],
    m0_size: int,
    m_ranges: list[tuple[int, int, int]],
    out_channels: int,
) -> torch.Tensor:
    """Apply SO(2) blocks to ``x`` with shape ``(E,S,D,C)``."""
    n_edge = x.shape[0]
    n_stream = x.shape[1]
    out = x.new_empty(n_edge, n_stream, x.shape[2], out_channels)
    m0_in = matrix_m0.shape[1]
    x_m0 = x[:, :, :m0_size, :].reshape(n_edge, n_stream, m0_in)
    y_m0 = torch.einsum("esi,sio->eso", x_m0, matrix_m0)
    out[:, :, :m0_size, :] = y_m0.reshape(n_edge, n_stream, m0_size, out_channels)

    for m_idx, (neg_start, pos_start, num_l) in enumerate(m_ranges):
        block_in = matrices_m[m_idx].shape[1]
        block_out = matrices_m[m_idx].shape[2] // 2
        w_u = matrices_m[m_idx][:, :, :block_out]
        w_v = matrices_m[m_idx][:, :, block_out:]
        x_neg = x[:, :, neg_start : neg_start + num_l, :].reshape(
            n_edge, n_stream, block_in
        )
        x_pos = x[:, :, pos_start : pos_start + num_l, :].reshape(
            n_edge, n_stream, block_in
        )
        y_neg = torch.einsum("esi,sio->eso", x_neg, w_u) - torch.einsum(
            "esi,sio->eso", x_pos, w_v
        )
        y_pos = torch.einsum("esi,sio->eso", x_neg, w_v) + torch.einsum(
            "esi,sio->eso", x_pos, w_u
        )
        out[:, :, neg_start : neg_start + num_l, :] = y_neg.reshape(
            n_edge, n_stream, num_l, out_channels
        )
        out[:, :, pos_start : pos_start + num_l, :] = y_pos.reshape(
            n_edge, n_stream, num_l, out_channels
        )
    return out
