# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Edge cache construction utilities for SeZM.

This module defines the shared procedures that assemble per-edge geometry,
radial features, rotation blocks, and normalization terms used by the SeZM
descriptor.
"""

from __future__ import (
    annotations,
)

import math
from collections.abc import (
    Callable,
)

import torch
from einops import (
    rearrange,
)

from .edge import (
    EdgeFeatureCache,
    build_edge_type_feat,
)
from .utils import (
    get_promoted_dtype,
    nvtx_range,
    safe_norm,
)
from .wignerd import (
    build_edge_quaternion,
    quaternion_multiply,
    quaternion_z_rotation,
)

WignerCalculatorFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
EdgeTypeKeepMaskFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _get_empty_edge_cache(
    *,
    n_nodes: int,
    n_radial: int,
    n_channel: int,
    device: torch.device,
    dtype: torch.dtype,
) -> EdgeFeatureCache:
    """
    Allocate an empty edge cache for one SeZM forward pass.

    Parameters
    ----------
    n_nodes
        Number of local nodes in the flattened frame-major layout.
    n_radial
        Number of radial basis channels.
    n_channel
        Edge type feature width.
    device
        Target device for the cache tensors.
    dtype
        Target floating-point dtype for the cache tensors.

    Returns
    -------
    EdgeFeatureCache
        Empty cache with valid tensor shapes and neutral degree normalization.
    """
    empty_long = torch.empty(0, dtype=torch.long, device=device)
    empty_vec = torch.empty(0, 3, dtype=dtype, device=device)
    empty_rbf = torch.empty(0, n_radial, dtype=dtype, device=device)
    empty_type_feat = torch.empty(0, n_channel, dtype=dtype, device=device)
    deg = torch.zeros(n_nodes, dtype=dtype, device=device)
    inv_sqrt_deg = torch.ones(n_nodes, 1, 1, dtype=dtype, device=device)
    return EdgeFeatureCache(
        src=empty_long,
        dst=empty_long,
        edge_type_feat=empty_type_feat,
        edge_vec=empty_vec,
        edge_rbf=empty_rbf,
        edge_env=torch.empty(0, 1, dtype=dtype, device=device),
        deg=deg,
        inv_sqrt_deg=inv_sqrt_deg,
        D_full=None,
        Dt_full=None,
        D_to_m_cache={},
        Dt_from_m_cache={},
    )


def _build_standard_edge_index(
    *,
    nlist: torch.Tensor,
    mapping: torch.Tensor | None,
    pair_keep_mask: torch.Tensor,
    nall: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flatten DeePMD valid neighbor slots into per-edge indices.

    This helper keeps the original edge semantics used by the eager standard path:

    - padding slots (``nlist == -1``) are removed
    - excluded type pairs are removed
    - no distance-based filtering is applied here; edges beyond ``rcut`` remain
      in the cache and are later zeroed naturally by the smooth envelope

    Parameters
    ----------
    nlist
        DeePMD neighbor list with shape ``(nf, nloc, nnei)``.
    mapping
        Optional extended-to-local mapping with shape ``(nf, nall)``.
    pair_keep_mask
        Pair exclusion keep mask with shape ``(nf, nloc, nnei)``.
    nall
        Number of atoms on the extended axis per frame.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(src, dst, center_coord_index, neighbor_coord_index)`` for the valid
        standard-path edges. All tensors have shape ``(E,)``.
    """
    nf, nloc, nnei = nlist.shape
    nlist_flat = nlist.reshape(-1)

    # === Step 1. Identify valid edge slots ===
    # An edge is valid if:
    #   - it is not padding (nlist >= 0)
    #   - the type pair is allowed (pair_keep_mask)
    # Note: We do NOT filter by distance here. Edges beyond rcut stay in the
    # cache and will later get edge_env=0 from the cutoff envelope.
    valid_nlist = nlist >= 0
    edge_keep = (valid_nlist & pair_keep_mask).reshape(-1)
    edge_slot = torch.nonzero(edge_keep).squeeze(-1).to(dtype=torch.long)

    if edge_slot.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=nlist.device)
        return empty, empty, empty, empty

    # === Step 2. Decode flat edge slots ===
    # edge_slot indexes the flattened (nf, nloc, nnei) axis in row-major order.
    # Convert it back to:
    #   frame_idx   in [0, nf)
    #   center_local in [0, nloc)
    #   neighbor_ext from the extended axis in [0, nall)
    frame_idx = edge_slot // (nloc * nnei)
    rem = edge_slot % (nloc * nnei)
    center_local = rem // nnei
    neighbor_ext = nlist_flat.index_select(0, edge_slot)

    if mapping is None:
        # Neighbor indices are already local indices in [0, nloc).
        src_local = neighbor_ext
    else:
        # Map extended index -> local index for each frame.
        # mapping_flat packs (nf, nall), so frame k uses offset k * nall.
        mapping_flat = mapping.reshape(-1)
        src_local = mapping_flat.index_select(0, frame_idx * nall + neighbor_ext)

    src_ok = (src_local >= 0) & (src_local < nloc)
    if not bool(src_ok.all()):
        # Drop edges that map outside the local range, e.g. broken mapping
        # or ghost-only neighbors.
        frame_idx = frame_idx[src_ok]
        center_local = center_local[src_ok]
        neighbor_ext = neighbor_ext[src_ok]
        src_local = src_local[src_ok]

    if src_local.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=nlist.device)
        return empty, empty, empty, empty

    # === Step 3. Build node and coordinate indices ===
    # dst is the center atom: per-frame local index -> global node index.
    # src is the neighbor atom: per-frame local index -> global node index.
    # The coordinate indices still point to the extended coordinate tensor.
    src = frame_idx * nloc + src_local
    dst = frame_idx * nloc + center_local
    center_coord_index = frame_idx * nall + center_local
    neighbor_coord_index = frame_idx * nall + neighbor_ext
    return src, dst, center_coord_index, neighbor_coord_index


def build_edge_cache(
    *,
    type_ebed: torch.Tensor,
    extended_coord: torch.Tensor,
    nlist: torch.Tensor,
    mapping: torch.Tensor | None,
    pair_keep_mask: torch.Tensor,
    eps: float,
    inner_clamp: Callable[[torch.Tensor], torch.Tensor] | None,
    edge_envelope: Callable[[torch.Tensor], torch.Tensor],
    radial_basis: Callable[[torch.Tensor], torch.Tensor],
    n_radial: int,
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
) -> EdgeFeatureCache:
    """
    Build the global edge cache from DeePMD padded neighbor list.

    This converts DeePMD's per-frame padded neighbor list into a flat list of
    valid edges used by message passing, and computes all per-edge tensors that
    are reused across blocks.

    The resulting cache contains:

    - per-edge endpoints: ``src``, ``dst`` and per-edge type features: ``edge_type_feat`` (src+dst)
    - per-edge geometry: ``edge_vec``
    - per-edge smooth weights: C^3 cutoff envelope ``edge_env``
    - per-edge radial basis: ``edge_rbf`` (envelope already baked in)
    - per-edge rotation blocks: block-diagonal Wigner-D matrices ``D_full`` and ``Dt_full``
    - destination-node smooth normalization: ``inv_sqrt_deg`` from
      envelope-squared degree ``sum(edge_env**2)``

    Notes
    -----
    Input formats follow DeePMD conventions:

    - ``extended_coord`` has shape ``(nf, nall, 3)``.
    - ``nlist`` has shape ``(nf, nloc, nnei)`` and stores indices into the extended axis
      (``0..nall-1``), with ``-1`` indicating padding.
    - ``mapping`` (when provided) maps extended indices to local indices ``0..nloc-1``.
      When ``mapping`` is ``None``, the function assumes the neighbor indices are already local.

    This function builds the edge cache directly on the valid edge set, so
    padded or excluded neighbor slots never enter the geometry, radial basis,
    or Wigner-D evaluation.

    Parameters
    ----------
    type_ebed
        Per-node type embedding with shape (N, C), where N=nf*nloc.
    extended_coord
        Extended coordinates with shape (nf, nall, 3).
    nlist
        Neighbor list with shape (nf, nloc, nnei).
    mapping
        Mapping from extended indices to local indices with shape (nf, nall), or None.
    pair_keep_mask
        Pair keep mask from `PairExcludeMask` with shape (nf, nloc, nnei). True means keep.
    eps
        Small positive epsilon for safe norm and degree normalization.
    inner_clamp
        Optional inner clamp used to freeze short-range geometry below `r_inner`.
    edge_envelope
        C^3 edge envelope module.
    radial_basis
        Radial basis module.
    n_radial
        Number of radial basis channels used for empty-cache allocation.
    random_gamma
        Whether to apply a random roll around the local +Z axis before
        constructing Wigner-D blocks.
    wigner_calc
        Callable that converts edge-aligned quaternions into packed Wigner-D
        blocks.

    Returns
    -------
    EdgeFeatureCache
        Per-edge cache.
    """
    nf, nloc, nnei = nlist.shape
    n_nodes = int(nf * nloc)

    # === Step 1. Force fp32+ for geometry ===
    geom_dtype = get_promoted_dtype(extended_coord.dtype)
    coord = extended_coord.to(dtype=geom_dtype)  # (nf, nall, 3)
    nall = coord.shape[1]

    # === Step 2. Build valid edge indices once ===
    with nvtx_range("index"):
        src, dst, center_coord_index, neighbor_coord_index = _build_standard_edge_index(
            nlist=nlist,
            mapping=mapping,
            pair_keep_mask=pair_keep_mask,
            nall=nall,
        )

    if src.numel() == 0:
        return _get_empty_edge_cache(
            n_nodes=n_nodes,
            n_radial=n_radial,
            n_channel=type_ebed.shape[1],
            device=extended_coord.device,
            dtype=extended_coord.dtype,
        )

    # === Step 3. Gather per-edge geometry ===
    # edge_vec points from center -> neighbor: r_ij = r_j - r_i (in Å).
    # edge_len is the (possibly clamped) scalar distance.
    with nvtx_range("edge_geom"):
        coord_flat = coord.reshape(nf * nall, 3)
        center_pos = coord_flat.index_select(0, center_coord_index)
        neighbor_pos = coord_flat.index_select(0, neighbor_coord_index)
        edge_vec = neighbor_pos - center_pos  # (E, 3)
        edge_len = safe_norm(edge_vec, eps)  # (E, 1)

    # === Step 4. Inner clamping for zone bridging ===
    # Freeze the descriptor below r_inner: both scalar distance and
    # displacement vector are clamped so the descriptor sees no
    # information about the true distance when r < r_inner.
    if inner_clamp is not None:
        clamped = inner_clamp(edge_len)  # (E, 1)
        scale = clamped / edge_len
        edge_vec = edge_vec * scale  # direction preserved, length -> clamped
        edge_len = clamped

    # === Step 5. C^3 envelope weight ===
    # Edges with r >= rcut are not removed from the cache. Their envelope is
    # exactly zero, so messages vanish naturally while degree normalization
    # remains smooth at the cutoff boundary.
    with nvtx_range("envelope"):
        edge_env = edge_envelope(edge_len)  # (E, 1)

    # === Step 6. Radial basis (envelope already baked in) ===
    with nvtx_range("radial_basis"):
        edge_rbf = radial_basis(edge_len)  # (E, n_radial)

    # === Step 7. Edge quaternion -> Wigner-D blocks ===
    with nvtx_range("wigner_d"):
        D_full, Dt_full = _build_edge_wigner(
            edge_vec=edge_vec,
            edge_len=edge_len,
            eps=eps,
            random_gamma=random_gamma,
            wigner_calc=wigner_calc,
        )  # (E, D, D), (E, D, D)

    edge_type_feat = build_edge_type_feat(
        type_ebed, src, dst, num_edges=int(src.numel())
    )  # (E, C)

    return _finalize_edge_cache(
        n_nodes=n_nodes,
        src=src,
        dst=dst,
        edge_type_feat=edge_type_feat,
        edge_vec=edge_vec,
        edge_rbf=edge_rbf,
        edge_env=edge_env,
        D_full=D_full,
        Dt_full=Dt_full,
        eps=eps,
    )


def build_edge_cache_from_edges(
    *,
    type_ebed: torch.Tensor,
    atype_flat: torch.Tensor,
    edge_index: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_mask: torch.Tensor,
    compute_dtype: torch.dtype,
    eps: float,
    inner_clamp: Callable[[torch.Tensor], torch.Tensor] | None,
    edge_envelope: Callable[[torch.Tensor], torch.Tensor],
    radial_basis: Callable[[torch.Tensor], torch.Tensor],
    has_exclude_types: bool,
    edge_type_keep_mask: EdgeTypeKeepMaskFn,
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
) -> EdgeFeatureCache:
    """
    Build the global edge cache from a fixed-shape edge list.

    Parameters
    ----------
    type_ebed
        Per-node type embedding with shape (N, C), where N=nf*nloc.
    atype_flat
        Flattened local atom types with shape (N,).
    edge_index
        Edge indices with shape (2, E).
    edge_vec
        Edge vectors with shape (E, 3) in Å.
    edge_mask
        Edge mask with shape (E,). True means keep.
    compute_dtype
        Promoted compute dtype used for geometry and radial features.
    eps
        Small positive epsilon for safe norm and degree normalization.
    inner_clamp
        Optional inner clamp used to freeze short-range geometry below `r_inner`.
    edge_envelope
        C^3 edge envelope module.
    radial_basis
        Radial basis module.
    has_exclude_types
        Whether excluded type pairs should be filtered in this path.
    edge_type_keep_mask
        Callable that builds the keep mask for edge type exclusions.
    random_gamma
        Whether to apply a random roll around the local +Z axis before
        constructing Wigner-D blocks.
    wigner_calc
        Callable that converts edge-aligned quaternions into packed Wigner-D
        blocks.

    Returns
    -------
    EdgeFeatureCache
        Per-edge cache.
    """
    n_nodes = int(type_ebed.shape[0])
    src = edge_index[0].to(dtype=torch.long)
    dst = edge_index[1].to(dtype=torch.long)

    # === Step 1. Normalize mask and apply type exclusions ===
    edge_keep = edge_mask.to(dtype=torch.bool)
    if has_exclude_types:
        edge_keep = edge_keep & edge_type_keep_mask(atype_flat, src, dst)

    # === Step 2. Promote geometry dtype ===
    geom_dtype = compute_dtype
    edge_vec = edge_vec.to(dtype=geom_dtype)
    edge_keep_f = edge_keep.to(dtype=geom_dtype).unsqueeze(-1)
    edge_vec = edge_vec * edge_keep_f
    edge_vec = edge_vec + (1.0 - edge_keep_f) * edge_vec.new_tensor([0.0, 0.0, 1.0])

    # === Step 3. Edge length, envelope, and radial basis ===
    with nvtx_range("envelope"):
        edge_len = safe_norm(edge_vec, eps)
        if inner_clamp is not None:
            clamped = inner_clamp(edge_len)
            scale = clamped / edge_len
            edge_vec = edge_vec * scale
            edge_len = clamped
        edge_env = edge_envelope(edge_len) * edge_keep_f  # (E, 1)
        edge_rbf = radial_basis(edge_len) * edge_keep_f  # (E, n_radial)

    # === Step 4. Edge quaternion -> Wigner-D blocks ===
    with nvtx_range("wigner_d"):
        D_full, Dt_full = _build_edge_wigner(
            edge_vec=edge_vec,
            edge_len=edge_len,
            eps=eps,
            random_gamma=random_gamma,
            wigner_calc=wigner_calc,
        )  # (E, D, D), (E, D, D)

    # === Step 5. Edge type features ===
    edge_type_feat = build_edge_type_feat(
        type_ebed, src, dst, num_edges=int(src.numel())
    )
    edge_type_feat = edge_type_feat * edge_keep_f.to(dtype=edge_type_feat.dtype)

    return _finalize_edge_cache(
        n_nodes=n_nodes,
        src=src,
        dst=dst,
        edge_type_feat=edge_type_feat,
        edge_vec=edge_vec,
        edge_rbf=edge_rbf,
        edge_env=edge_env,
        D_full=D_full,
        Dt_full=Dt_full,
        eps=eps,
    )


def _build_edge_wigner(
    *,
    edge_vec: torch.Tensor,
    edge_len: torch.Tensor,
    eps: float,
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build packed Wigner-D blocks from edge vectors.

    Parameters
    ----------
    edge_vec
        Edge vectors with shape (E, 3) in Å.
    edge_len
        Edge lengths with shape (E, 1).
    eps
        Small positive epsilon used in quaternion construction.
    random_gamma
        Whether to apply a random roll around the local +Z axis.
    wigner_calc
        Callable that converts edge-aligned quaternions into packed Wigner-D
        blocks.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Packed Wigner-D matrices ``(D_full, Dt_full)`` with shape ``(E, D, D)``.
    """
    # === Step 1. Build edge-aligned quaternions ===
    edge_quat = build_edge_quaternion(
        edge_vec,
        edge_len=edge_len,
        eps=eps,
    )

    # === Step 2. Apply optional random local-Z roll ===
    if random_gamma:
        gamma = torch.rand(
            edge_quat.shape[0],
            dtype=edge_quat.dtype,
            device=edge_quat.device,
        ) * (2.0 * math.pi)
        edge_quat = quaternion_multiply(quaternion_z_rotation(gamma), edge_quat)

    # === Step 3. Convert quaternions to packed Wigner-D blocks ===
    return wigner_calc(edge_quat)


def _finalize_edge_cache(
    *,
    n_nodes: int,
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_type_feat: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_rbf: torch.Tensor,
    edge_env: torch.Tensor,
    D_full: torch.Tensor,
    Dt_full: torch.Tensor,
    eps: float,
) -> EdgeFeatureCache:
    """
    Assemble the shared `EdgeFeatureCache` layout.

    Parameters
    ----------
    n_nodes
        Number of local nodes in the flattened frame-major layout.
    src
        Source node indices with shape (E,).
    dst
        Destination node indices with shape (E,).
    edge_type_feat
        Per-edge type features with shape (E, C).
    edge_vec
        Edge vectors with shape (E, 3).
    edge_rbf
        Radial basis features with shape (E, n_radial).
    edge_env
        Smooth edge envelope weights with shape (E, 1).
    D_full
        Packed Wigner-D matrices with shape (E, D, D).
    Dt_full
        Transposed packed Wigner-D matrices with shape (E, D, D).
    eps
        Small positive epsilon used in degree normalization.

    Returns
    -------
    EdgeFeatureCache
        Finalized per-edge cache shared by eager and compile paths.
    """
    # === Step 1. Build smooth destination degrees ===
    with nvtx_range("degree"):
        deg = torch.zeros(n_nodes, dtype=edge_vec.dtype, device=edge_vec.device)  # (N,)
        deg.index_add_(0, dst, edge_env.squeeze(-1).to(dtype=edge_vec.dtype).square())
        inv_sqrt_deg = rearrange(torch.rsqrt(deg + eps), "N -> N 1 1")  # (N, 1, 1)

    return EdgeFeatureCache(
        src=src,
        dst=dst,
        edge_type_feat=edge_type_feat,
        edge_vec=edge_vec,
        edge_rbf=edge_rbf,
        edge_env=edge_env,
        deg=deg,
        inv_sqrt_deg=inv_sqrt_deg,
        D_full=D_full,
        Dt_full=Dt_full,
        D_to_m_cache={},
        Dt_from_m_cache={},
    )
