# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Attention utilities for SeZM message passing.

This module implements the destination-wise envelope-gated softmax used by the
SO(2) attention path in the SeZM descriptor.
"""

from __future__ import (
    annotations,
)

import torch
import torch.nn.functional as F


@torch.amp.autocast("cuda", enabled=False)
def segment_envelope_gated_softmax(
    logits: torch.Tensor,
    edge_env: torch.Tensor,
    dst: torch.Tensor,
    n_nodes: int,
    z_bias_raw: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Compute destination-wise envelope-gated softmax attention.

    Parameters
    ----------
    logits
        Attention logits with shape (E, F, H).
    edge_env
        Cutoff envelope weights with shape (E, 1) or (E,).
    dst
        Destination node indices with shape (E,).
    n_nodes
        Number of nodes.
    z_bias_raw
        Unconstrained denominator bias with shape (F, H).
        Softplus is applied to keep the bias strictly positive.
    eps
        Small epsilon for denominator stability.

    Returns
    -------
    torch.Tensor
        Normalized edge weights with shape (E, F, H), computed as
        ``edge_env**2 * exp(logits) / (zeta + sum(edge_env**2 * exp(logits)))``.
    """
    n_edge, n_focus, n_head = logits.shape
    n_channel = n_focus * n_head
    eps_f = float(eps)

    # === Step 1. Flatten (F, H) to reduce index tensor traffic ===
    logits_2d = logits.reshape(n_edge, n_channel)
    edge_env_1d = edge_env.squeeze(-1).to(dtype=logits.dtype).clamp_min(0.0)
    zeta = F.softplus(z_bias_raw).reshape(1, n_channel).to(dtype=logits.dtype)
    has_weight = edge_env_1d > 0.0
    logits_for_max = torch.where(
        has_weight.reshape(n_edge, 1),
        logits_2d,
        torch.full_like(logits_2d, float("-inf")),
    )

    # === Step 2. Destination-wise max for stable exponentials ===
    group_max = torch.full(
        (n_nodes, n_channel),
        float("-inf"),
        dtype=logits.dtype,
        device=logits.device,
    )
    group_max.index_reduce_(0, dst, logits_for_max, reduce="amax", include_self=True)
    edge_max = group_max.index_select(0, dst)
    edge_max = torch.where(
        torch.isfinite(edge_max), edge_max, torch.zeros_like(edge_max)
    )
    group_max_safe = torch.where(
        torch.isfinite(group_max), group_max, torch.zeros_like(group_max)
    )

    # === Step 3. Envelope-gated exponential terms ===
    exp_shifted = torch.exp(logits_2d - edge_max)
    edge_weighted_exp = edge_env_1d.square().reshape(n_edge, 1) * exp_shifted

    # === Step 4. Destination-wise normalization with positive denominator bias ===
    denom_sum = torch.zeros(
        n_nodes,
        n_channel,
        dtype=logits.dtype,
        device=logits.device,
    )
    denom_sum.index_add_(0, dst, edge_weighted_exp)
    denom = denom_sum + zeta * torch.exp(-group_max_safe)

    alpha = edge_weighted_exp / (denom.index_select(0, dst) + eps_f)
    return alpha.reshape(n_edge, n_focus, n_head)
