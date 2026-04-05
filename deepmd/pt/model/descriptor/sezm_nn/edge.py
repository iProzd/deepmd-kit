# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Edge cache data structures and edge-local helpers for SeZM.

This module defines the shared edge cache container together with helper
functions used to assemble and cast per-edge descriptor data.
"""

from __future__ import (
    annotations,
)

from typing import (
    NamedTuple,
)

import torch
import torch.nn.functional as F


def build_edge_type_feat(
    type_ebed: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    num_edges: int | None = None,
) -> torch.Tensor:
    """
    Build per-edge type features by summing src/dst embeddings.

    Parameters
    ----------
    type_ebed
        Per-node type embedding with shape (N, C).
    src
        Source node indices with shape (E,).
    dst
        Destination node indices with shape (E,).
    num_edges
        Number of edges. When provided, avoids calling `numel()` on GPU tensors.

    Returns
    -------
    torch.Tensor
        Per-edge type features with shape (E, C).
    """
    if num_edges is None:
        num_edges = int(src.numel())
    if num_edges == 0:
        return type_ebed.new_empty((0, type_ebed.shape[1]))

    # === Step 1. Normalize index dtypes ===
    if src.dtype != torch.long:
        src = src.to(dtype=torch.long)
    if dst.dtype != torch.long:
        dst = dst.to(dtype=torch.long)

    # === Step 2. Pack edge indices as a 2-item bag ===
    pair_index = torch.stack((src, dst), dim=1).reshape(-1)

    # === Step 3. Reduce with a single embedding_bag ===
    offsets = torch.arange(
        0, 2 * num_edges, step=2, device=src.device, dtype=torch.long
    )
    return F.embedding_bag(
        pair_index,
        type_ebed,
        offsets,
        mode="sum",
        include_last_offset=False,
    )


class EdgeFeatureCache(NamedTuple):
    """
    Global edge feature cache created once per forward().

    All tensors are aligned on the same edge axis (E = number of valid edges).

    Parameters
    ----------
    src
        Source node indices with shape (E,).
    dst
        Destination node indices with shape (E,).
    edge_type_feat
        Per-edge type embeddings with shape (E, C), computed as src+dst.
    edge_vec
        Edge vectors with shape (E, 3) in Å.
    edge_rbf
        Radial basis with shape (E, n_radial).
        The C^3 cutoff envelope is already baked in.
    edge_env
        C^3 cutoff envelope weights with shape (E, 1).
    deg
        Envelope-squared smooth degree with shape (N,), computed as
        ``sum(edge_env**2)`` over incoming edges.
        Used for smooth normalization in EnvironmentInitialEmbedding.
    inv_sqrt_deg
        Inverse square root smooth degree normalization with shape (N, 1, 1).
    D_full
        Block-diagonal Wigner-D matrix with shape (E, D, D) where D=(lmax+1)^2.
        Used for efficient batched rotation. None if not available.
    Dt_full
        Transpose of D_full with shape (E, D, D). None if not available.
    D_to_m_cache
        Lazy cache for projected D matrices keyed by a normalized
        ``"lmax:mmax"`` identifier.
    Dt_from_m_cache
        Lazy cache for projected Dt matrices keyed by a normalized
        ``"lmax:mmax"`` identifier.
    """

    src: torch.Tensor
    dst: torch.Tensor
    edge_type_feat: torch.Tensor
    edge_vec: torch.Tensor
    edge_rbf: torch.Tensor
    edge_env: torch.Tensor
    deg: torch.Tensor
    inv_sqrt_deg: torch.Tensor
    D_full: torch.Tensor | None = None
    Dt_full: torch.Tensor | None = None
    D_to_m_cache: dict[str, torch.Tensor] | None = None
    Dt_from_m_cache: dict[str, torch.Tensor] | None = None


def edge_cache_to_dtype(
    cache: EdgeFeatureCache, dtype: torch.dtype
) -> EdgeFeatureCache:
    """
    Convert all floating-point tensors in EdgeFeatureCache to the specified dtype.

    Integer tensors (src, dst) are unchanged. This is a standalone function
    (not a method) to keep it side-effect free.

    Parameters
    ----------
    cache
        The edge feature cache to convert.
    dtype
        Target dtype for floating-point tensors.

    Returns
    -------
    EdgeFeatureCache
        New cache with converted tensors.
    """
    # Handle Optional tensors explicitly.
    # Use local variables with explicit None check and assignment.
    _D_full = cache.D_full
    _Dt_full = cache.Dt_full
    D_full: torch.Tensor | None = None
    Dt_full: torch.Tensor | None = None
    if _D_full is not None:
        D_full = _D_full.to(dtype=dtype)
    if _Dt_full is not None:
        Dt_full = _Dt_full.to(dtype=dtype)

    return EdgeFeatureCache(
        src=cache.src,
        dst=cache.dst,
        edge_type_feat=cache.edge_type_feat.to(dtype=dtype),
        edge_vec=cache.edge_vec.to(dtype=dtype),
        edge_rbf=cache.edge_rbf.to(dtype=dtype),
        edge_env=cache.edge_env.to(dtype=dtype),
        deg=cache.deg.to(dtype=dtype),
        inv_sqrt_deg=cache.inv_sqrt_deg.to(dtype=dtype),
        D_full=D_full,
        Dt_full=Dt_full,
        D_to_m_cache=None if cache.D_to_m_cache is None else {},
        Dt_from_m_cache=None if cache.Dt_from_m_cache is None else {},
    )
