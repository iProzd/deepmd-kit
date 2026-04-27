# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fused CUDA implementation of the MoE topk-expand-sort op.

Drop-in replacement for ``_topk_expand_sort`` in :mod:`moe_layer`, which
was a chain of ``repeat_interleave`` + stable ``argsort`` + multiple
gathers + inverse-permutation + ``bincount`` + ``.max().item()`` + ``.tolist()``.

The CUDA kernel uses a counting sort (faster than comparison sort for
small expert counts) and fuses the per-token feature scatter into the
same pass.  Forward / backward / double-backward are all implemented in
CUDA and registered as two ``torch::autograd::Function`` classes inside
``source/op/pt/moe_topk_expand_sort.cc`` so that DeePMD's energy → force
→ force-loss double-backward path goes through CUDA end-to-end.
"""

from __future__ import annotations

import torch

import deepmd.pt.cxx_op  # noqa: F401  registers torch.ops.deepmd


def fused_topk_expand_sort(
    features: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    experts_per_gpu: int,
    n_routing_experts: int,
    ep_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    int,
]:
    """Fused CUDA topk-expand-sort.

    Parameters mirror the original ``_topk_expand_sort``.  The last
    ``ep_size`` argument is required (the reference implementation
    inferred it from ``sorted_expert_ids.max()``, which forced a
    device→host sync; here it must be passed explicitly).

    Returns
    -------
    sorted_features : Tensor ``[N*topk, feat_dim]``
    sorted_expert_ids : Tensor ``[N*topk]`` int64
    sorted_weights : Tensor ``[N*topk]``
    unsort_idx : Tensor ``[N*topk]`` int64
    counts_per_gpu : list[int]
        Length ``ep_size``.  This still triggers a device→host sync
        because downstream code (split sizes for All-to-All) needs Python
        ints; the sync is unavoidable, but at least the histogram and
        per-GPU summation now run in CUDA.
    ep_size : int
        Echoed back for API compatibility with the reference.
    """
    out = torch.ops.deepmd.moe_topk_expand_sort(
        features,
        topk_indices,
        topk_weights,
        int(experts_per_gpu),
        int(n_routing_experts),
        int(ep_size),
    )
    sorted_features, sorted_expert_ids, sorted_weights, unsort_idx, gpu_counts = out
    counts_per_gpu = gpu_counts.tolist()
    return (
        sorted_features,
        sorted_expert_ids,
        sorted_weights,
        unsort_idx,
        counts_per_gpu,
        ep_size,
    )
