# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Helper utilities for SeZM-Net PyTorch descriptor.

This module contains:
- ``EdgeFeatureCache``: Shared data structure for edge features
- ``WignerDCalcBase``: Abstract base class for Wigner-D calculators
- ``WignerDCalculator``: Block-diagonal parallel implementation
"""

import math
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    NamedTuple,
)

import numpy as np
import torch
import torch.nn as nn

from deepmd.pt.utils import (
    env,
)

from .se_zm_triton import (
    build_z_rotation_triton,
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
        The C^2 cutoff envelope is already baked in.
    edge_env
        C^2 cutoff envelope weights with shape (E, 1).
    deg
        Destination node degree (number of incoming edges) with shape (N,).
        Used for neighbor normalization in EnvironmentInitialEmbedding.
    inv_sqrt_deg
        Destination degree normalization with shape (N, 1, 1).
    D_full
        Block-diagonal Wigner-D matrix with shape (E, D, D) where D=(lmax+1)^2.
        Used for efficient batched rotation. None if not available.
    Dt_full
        Transpose of D_full with shape (E, D, D). None if not available.
    D_to_m_cache
        Lazy cache for projected D matrices keyed by (lmax, mmax).
    Dt_from_m_cache
        Lazy cache for projected Dt matrices keyed by (lmax, mmax).
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

    def get_D_to_m(
        self,
        *,
        ebed_dim_full: int,
        coeff_index_m: torch.Tensor,
        key_lmax: int,
        key_mmax: int,
    ) -> torch.Tensor:
        """
        Fetch (or build once) the row-projected Wigner-D blocks for m-major layout.

        This selects the subset of rows needed for the m-major truncated layout,
        caches the result keyed by (lmax, mmax), and reuses it across blocks.

        Parameters
        ----------
        ebed_dim_full
            Full SO(3) dimension D=(lmax+1)^2 used to slice the block-diagonal
            Wigner matrix.
        coeff_index_m
            Indices for the m-major reduced layout with shape (D_m_trunc,).
        key_lmax
            lmax used to build ``coeff_index_m`` (cache key).
        key_mmax
            mmax used to build ``coeff_index_m`` (cache key).

        Returns
        -------
        torch.Tensor
            Projected rotation matrix with shape (E, D_m_trunc, D_full).
        """
        cache_key = f"{int(key_lmax)}:{int(key_mmax)}"
        cache_dict = self.D_to_m_cache
        if cache_dict is None:
            raise ValueError("EdgeFeatureCache.D_to_m_cache is None")
        cached = cache_dict.get(cache_key)
        if cached is not None:
            return cached

        D_full = self.D_full
        if D_full is None:
            raise ValueError("EdgeFeatureCache.D_full is None")
        D_block = D_full[:, :ebed_dim_full, :ebed_dim_full]
        D_to_m = D_block.index_select(1, coeff_index_m)
        cache_dict[cache_key] = D_to_m
        return D_to_m

    def get_Dt_from_m(
        self,
        *,
        ebed_dim_full: int,
        coeff_index_m: torch.Tensor,
        key_lmax: int,
        key_mmax: int,
    ) -> torch.Tensor:
        """
        Fetch (or build once) the column-projected Wigner-D^T blocks for inverse rotation.

        This selects the subset of columns needed for the m-major truncated layout,
        caches the result keyed by (lmax, mmax), and reuses it across blocks.

        Parameters
        ----------
        ebed_dim_full
            Full SO(3) dimension D=(lmax+1)^2 used to slice the block-diagonal
            Wigner matrix.
        coeff_index_m
            Indices for the m-major reduced layout with shape (D_m_trunc,).
        key_lmax
            lmax used to build ``coeff_index_m`` (cache key).
        key_mmax
            mmax used to build ``coeff_index_m`` (cache key).

        Returns
        -------
        torch.Tensor
            Projected inverse rotation matrix with shape (E, D_full, D_m_trunc).
        """
        cache_key = (int(key_lmax), int(key_mmax))
        cache_dict = self.Dt_from_m_cache
        if cache_dict is None:
            raise ValueError("EdgeFeatureCache.Dt_from_m_cache is None")
        cached = cache_dict.get(cache_key)
        if cached is not None:
            return cached

        Dt_full = self.Dt_full
        if Dt_full is None:
            raise ValueError("EdgeFeatureCache.Dt_full is None")
        Dt_block = Dt_full[:, :ebed_dim_full, :ebed_dim_full]
        Dt_from_m = Dt_block.index_select(2, coeff_index_m)
        cache_dict[cache_key] = Dt_from_m
        return Dt_from_m


def edge_cache_to_dtype(
    cache: EdgeFeatureCache, dtype: torch.dtype
) -> EdgeFeatureCache:
    """
    Convert all floating-point tensors in EdgeFeatureCache to the specified dtype.

    Integer tensors (src, dst) are unchanged. This is a standalone function
    (not a method) to ensure TorchScript compatibility.

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
    # Handle Optional tensors in a TorchScript-compatible way.
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
        D_to_m_cache={} if cache.D_to_m_cache is None else cache.D_to_m_cache,
        Dt_from_m_cache=(
            {} if cache.Dt_from_m_cache is None else cache.Dt_from_m_cache
        ),
    )


def safe_norm(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute vector norm with an epsilon lower bound.

    Uses float32 for computation when input is fp16/bf16.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape (N, 3), where N is the number of vectors.
    eps : float
        Lower bound for the norm.

    Returns
    -------
    torch.Tensor
        Norm with shape (N, 1), clamped to be >= eps.
    """
    in_dtype = x.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True).clamp(min=eps**2))
    return norm.to(dtype=in_dtype)


def safe_numpy_to_tensor(
    data: Any, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    if isinstance(data, np.ndarray):
        # Handle bfloat16: numpy uses ml_dtypes.bfloat16, which torch.as_tensor
        # cannot convert. Convert to float32 first, then cast to target dtype.
        if hasattr(data.dtype, "name") and "bfloat16" in data.dtype.name:
            data = data.astype(np.float32)
        return torch.as_tensor(data, device=device).to(dtype)
    return torch.as_tensor(data, device=device, dtype=dtype)


def get_promoted_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Get promoted dtype for numerical stability.

    For bf16/fp16, use float32 to ensure numerical stability
    in computation and storage compatibility.
    """
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def np_safe(
    tensor: torch.Tensor | None,
) -> np.ndarray | None:
    """
    Convert tensor to numpy array, promoting low-precision types to fp32.

    For bf16/fp16, converts to fp32 first since NumPy/HDF5 do not natively
    support these formats. fp32/fp64 are kept unchanged.

    Parameters
    ----------
    tensor
        PyTorch tensor to convert. Can be None.

    Returns
    -------
    np.ndarray or None
        numpy array with at least fp32 precision.
    """
    if tensor is None:
        return None
    if tensor.dtype in (torch.float16, torch.bfloat16):
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def get_so3_dim_of_lmax(lmax: int) -> int:
    """
    Return SO(3) representation dimension for given lmax.

    The dimension equals::

        sum_{l<=lmax} (2l+1) = (lmax+1)^2

    which is the number of spherical harmonics basis functions.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.

    Returns
    -------
    int
        The SO(3) dimension D = (lmax+1)^2.
    """
    return int((int(lmax) + 1) ** 2)


def map_degree_idx(lmax: int, *, device: torch.device) -> torch.Tensor:
    """
    Build degree (l) index for each position in the packed (l, m) layout.

    For each spherical harmonic coefficient position in the packed tensor,
    returns the corresponding angular momentum quantum number l.

    Examples
    --------
    For lmax=2, the packed layout has D=9 positions:
    - Position 0: l=0, m=0
    - Positions 1-3: l=1, m=-1,0,+1
    - Positions 4-8: l=2, m=-2,-1,0,+1,+2

    Returns: [0, 1,1,1, 2,2,2,2,2]

    Parameters
    ----------
    lmax
        Maximum angular momentum degree.
    device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Integer tensor with shape (D,), where D=(lmax+1)^2.
        Each element is the l value for that position.
    """
    lmax = int(lmax)
    counts = torch.tensor(
        [2 * l + 1 for l in range(lmax + 1)], device=device, dtype=torch.long
    )
    return torch.repeat_interleave(
        torch.arange(lmax + 1, device=device, dtype=torch.long), counts
    )


def project_D_to_m(
    D_full: torch.Tensor,
    coeff_index_m: torch.Tensor,
    ebed_dim_full: int,
    cache: dict[str, torch.Tensor] | None,
    key_lmax: int,
    key_mmax: int,
) -> torch.Tensor:
    """
    Row-project block-diagonal Wigner-D to the m-major truncated layout.

    Parameters
    ----------
    D_full
        Block-diagonal Wigner-D with shape (E, D_full, D_full).
    coeff_index_m
        Indices for m-major reduced layout with shape (D_m_trunc,).
    ebed_dim_full
        Full SO(3) dimension D_full = (lmax+1)^2 to slice the block.
    cache
        Optional cache mapping (lmax, mmax) -> projected matrix.
    key_lmax
        lmax used to build coeff_index_m (cache key).
    key_mmax
        mmax used to build coeff_index_m (cache key).

    Returns
    -------
    torch.Tensor
        Projected rotation matrix with shape (E, D_m_trunc, D_full).

    Examples
    --------
    For lmax=2, mmax=1 (D_full=9, D_m_trunc=7), coeff_index_m selects
    [0,2,6,1,5,3,7] in packed (l,m) order. The returned tensor keeps only those
    rows of ``D_full`` while retaining all columns, so that rotating and truncating
    is done in a single bmm: ``x_local = D_to_m @ x_global``.
    """
    cache_key = f"{int(key_lmax)}:{int(key_mmax)}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    D_block = D_full[:, :ebed_dim_full, :ebed_dim_full]
    proj = D_block.index_select(1, coeff_index_m)
    if cache is not None:
        cache[cache_key] = proj
    return proj


def project_Dt_from_m(
    Dt_full: torch.Tensor,
    coeff_index_m: torch.Tensor,
    ebed_dim_full: int,
    cache: dict[str, torch.Tensor] | None,
    key_lmax: int,
    key_mmax: int,
) -> torch.Tensor:
    """
    Column-project block-diagonal Wigner-D^T for inverse rotation.

    Parameters
    ----------
    Dt_full
        Block-diagonal Wigner-D^T with shape (E, D_full, D_full).
    coeff_index_m
        Indices for m-major reduced layout with shape (D_m_trunc,).
    ebed_dim_full
        Full SO(3) dimension D_full = (lmax+1)^2 to slice the block.
    cache
        Optional cache mapping (lmax, mmax) -> projected matrix.
    key_lmax
        lmax used to build coeff_index_m (cache key).
    key_mmax
        mmax used to build coeff_index_m (cache key).

    Returns
    -------
    torch.Tensor
        Projected inverse rotation matrix with shape (E, D_full, D_m_trunc).

    Examples
    --------
    Continuing lmax=2, mmax=1, the projection selects the same column subset
    [0,2,6,1,5,3,7] from ``Dt_full``. This enables inverse rotation with missing
    coefficients implicitly zeroed: ``x_global = Dt_from_m @ x_local``.
    """
    cache_key = f"{int(key_lmax)}:{int(key_mmax)}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    Dt_block = Dt_full[:, :ebed_dim_full, :ebed_dim_full]
    proj = Dt_block.index_select(2, coeff_index_m)
    if cache is not None:
        cache[cache_key] = proj
    return proj


def so3_packed_index(l: int, m: int) -> int:
    """
    Compute packed (l, m) index for real spherical harmonics layout.

    The packed layout is l-primary with m ordered as ``-l..+l`` inside each l-block.
    The index formula is::

        idx(l, m) = l^2 + l + m

    Parameters
    ----------
    l
        Degree l.
    m
        Order m, must satisfy ``-l <= m <= l``.

    Returns
    -------
    int
        Packed index.
    """
    l = int(l)
    m = int(m)
    return l * l + l + m


def build_l_major_index(lmax: int, mmax: int, *, device: torch.device) -> torch.Tensor:
    """
    Build coefficient indices for l-major layout truncated by mmax.

    The returned indices select coefficients with ``|m| <= min(mmax, l)`` in the
    standard packed (l, m) layout. The order is l-major:

    - l = 0..lmax
    - within each l, m = -min(mmax, l) .. +min(mmax, l)

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum order (|m|). Must satisfy ``0 <= mmax <= lmax``.
    device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Long tensor of indices with shape (D_m_trunc,), selecting coefficients
        from the full packed layout with D_full=(lmax+1)^2, where D_m_trunc is
        the number of coefficients kept under ``|m| <= min(mmax, l)``.

    Examples
    --------
    For lmax=2, mmax=1:
    - Full packed layout: l=0(0), l=1(1-3), l=2(4-8)
    - Truncated by mmax=1: skip (l=2, m=±2) at indices 4,8
    - Returns: [0, 1, 2, 3, 5, 6, 7]
    """
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    if lmax_i < 0:
        raise ValueError("`lmax` must be non-negative")
    if mmax_i < 0:
        raise ValueError("`mmax` must be non-negative")
    if mmax_i > lmax_i:
        raise ValueError("`mmax` must be <= `lmax`")

    indices: list[int] = []
    for l in range(lmax_i + 1):
        m_keep = min(mmax_i, l)
        for m in range(-m_keep, m_keep + 1):
            indices.append(so3_packed_index(l, m))
    return torch.tensor(indices, device=device, dtype=torch.long)


def build_m_major_index(lmax: int, mmax: int, *, device: torch.device) -> torch.Tensor:
    """
    Build coefficient indices for m-major layout truncated by mmax.

    This layout minimizes rotation cost and avoids gather-heavy indexing:

    - m = 0: l = 0..lmax (single coefficient per l)
    - for each m = 1..mmax:
        - negative part: l = m..lmax, coefficient (l, -m)
        - positive part: l = m..lmax, coefficient (l, +m)

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum order (|m|). Must satisfy ``0 <= mmax <= lmax``.
    device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Long tensor of indices with shape (D_m_trunc,), selecting coefficients
        from the full packed layout with D_full=(lmax+1)^2, where D_m_trunc is
        the number of coefficients kept under ``|m| <= min(mmax, l)``.

    Examples
    --------
    For lmax=2, mmax=1:
    - m=0 group: (l=0,m=0)→0, (l=1,m=0)→2, (l=2,m=0)→6
    - m=1 neg group: (l=1,m=-1)→1, (l=2,m=-1)→5
    - m=1 pos group: (l=1,m=+1)→3, (l=2,m=+1)→7
    - Returns: [0, 2, 6, 1, 5, 3, 7]
    """
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    if lmax_i < 0:
        raise ValueError("`lmax` must be non-negative")
    if mmax_i < 0:
        raise ValueError("`mmax` must be non-negative")
    if mmax_i > lmax_i:
        raise ValueError("`mmax` must be <= `lmax`")

    indices: list[int] = []
    # === Step 1. m = 0 group (l = 0..lmax) ===
    for l in range(lmax_i + 1):
        indices.append(so3_packed_index(l, 0))

    # === Step 2. m > 0 groups (neg then pos) ===
    for m in range(1, mmax_i + 1):
        for l in range(m, lmax_i + 1):
            indices.append(so3_packed_index(l, -m))
        for l in range(m, lmax_i + 1):
            indices.append(so3_packed_index(l, m))

    return torch.tensor(indices, device=device, dtype=torch.long)


def build_m_major_l_index(
    lmax: int, mmax: int, *, device: torch.device
) -> torch.Tensor:
    """
    Build degree (l) index aligned with `build_m_major_index`.

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum order (|m|). Must satisfy ``0 <= mmax <= lmax``.
    device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Long tensor of degrees with shape (D_m_trunc,). Entry i is the degree
        l for the i-th coefficient in the m-major layout.

    Examples
    --------
    For lmax=2, mmax=1:
    - m=0 group: l=0,1,2
    - m=1 neg group: l=1,2
    - m=1 pos group: l=1,2
    - Returns: [0, 1, 2, 1, 2, 1, 2]
    """
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    if lmax_i < 0:
        raise ValueError("`lmax` must be non-negative")
    if mmax_i < 0:
        raise ValueError("`mmax` must be non-negative")
    if mmax_i > lmax_i:
        raise ValueError("`mmax` must be <= `lmax`")

    degrees: list[int] = []
    # === Step 1. m = 0 group ===
    for l in range(lmax_i + 1):
        degrees.append(l)

    # === Step 2. m > 0 groups (neg then pos) ===
    for m in range(1, mmax_i + 1):
        for l in range(m, lmax_i + 1):
            degrees.append(l)
        for l in range(m, lmax_i + 1):
            degrees.append(l)

    return torch.tensor(degrees, device=device, dtype=torch.long)


class WignerDCalcBase(nn.Module, ABC):
    """
    Abstract base class for Wigner-D matrix calculators.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum degree.
    eps : float
        Small epsilon for numerical stability.
    dtype : torch.dtype
        Floating-point dtype for output matrices.
    """

    def __init__(self, lmax: int, *, eps: float = 1e-7, dtype: torch.dtype) -> None:
        super().__init__()
        self.lmax = int(lmax)
        if self.lmax < 0:
            raise ValueError("`lmax` must be non-negative")
        self.dtype = dtype
        self.device = env.DEVICE
        self.eps = float(eps)

    @abstractmethod
    def forward(self, rot_mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Wigner-D blocks for a batch of rotation matrices.

        Parameters
        ----------
        rot_mat : torch.Tensor
            Rotation matrices with shape (n_edges, 3, 3), global->local.

        Returns
        -------
        D_full : torch.Tensor
            Block-diagonal matrix with shape (n_edges, D, D) where D=(lmax+1)^2.
        Dt_full : torch.Tensor
            Transpose of D_full.
        """
        raise NotImplementedError

    def _extract_zyz_euler(
        self, rot_mat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert rotation matrices to ZYZ Euler angles.

        The returned angles satisfy::

            rot_mat = Rz(alpha) @ Ry(beta) @ Rz(gamma)

        where the basic rotation matrices are::

            Rz(alpha) = [[ ca, -sa,  0],
                         [ sa,  ca,  0],
                         [  0,   0,  1]]
            Ry(beta)  = [[ cb,   0, sb],
                         [  0,   1,  0],
                         [-sb,   0, cb]]
            Rz(gamma) = [[ cg, -sg,  0],
                         [ sg,  cg,  0],
                         [  0,   0,  1]]

        with ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta),
        cg = cos(gamma), sg = sin(gamma).

        The full 3x3 rotation matrix (for sin(beta) != 0) is::

            R = [
                [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
                [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
                [-sb * cg, sb * sg, cb],
            ]

        Therefore, the Euler angles are extracted as::

            alpha = atan2(R[1, 2], R[0, 2])
            beta = atan2(sin(beta), R[2, 2])
            gamma = atan2(R[2, 1], -R[2, 0])

        Singular cases (beta -> 0 or pi) are handled by setting ``gamma = 0``
        and folding the residual z-rotation into ``alpha``.

        Parameters
        ----------
        rot_mat : torch.Tensor
            Rotation matrices with shape (..., 3, 3).

        Returns
        -------
        alpha : torch.Tensor
            First z-rotation angle with shape (...,).
        beta : torch.Tensor
            y-rotation angle with shape (...,).
        gamma : torch.Tensor
            Second z-rotation angle with shape (...,).
        """
        # === Step 1. Compute beta with stable atan2(sin(beta), cos(beta)) ===
        # Using acos(cos_beta) creates Inf/NaN gradients near |cos_beta| = 1.
        # Gimbal lock occur when edge directions can align with the global z-axis.
        #
        # For ZYZ convention:
        #   cos(beta) = R[2, 2]
        #   sin(beta) = sqrt(R[0, 2]^2 + R[1, 2]^2)
        #
        # We apply an epsilon floor to sin(beta) to keep beta differentiable
        # at the singular manifolds (beta = 0 or pi).
        cos_beta = rot_mat[..., 2, 2].clamp(-1.0, 1.0)
        r02 = rot_mat[..., 0, 2]
        r12 = rot_mat[..., 1, 2]
        sin_beta_sq = r02 * r02 + r12 * r12
        sin_beta_raw = torch.sqrt(sin_beta_sq.clamp(min=0.0))
        sin_beta_safe = torch.sqrt(sin_beta_sq.clamp(min=self.eps))
        beta = torch.atan2(sin_beta_safe, cos_beta)

        # === Step 2. Detect singular cases via sin(beta) ===
        threshold = math.sqrt(self.eps)
        not_singular = sin_beta_raw > threshold

        # === Step 3. Non-singular extraction (sin(beta) > 0) ===
        # torch.atan2(y, x) has undefined gradient at (y, x) = (0, 0).
        # We use a safe variant that perturbs x by eps when ||(x,y)|| is tiny.
        alpha = self._safe_atan2(rot_mat[..., 1, 2], rot_mat[..., 0, 2])
        gamma = self._safe_atan2(rot_mat[..., 2, 1], -rot_mat[..., 2, 0])

        # === Step 4. Singular extraction (gimbal lock) ===
        # When sin(beta) -> 0, alpha and gamma are not individually identifiable.
        # Two singular manifolds exist:
        #   (1) beta -> 0:  R = Rz(alpha) @ I @ Rz(gamma) = Rz(alpha + gamma)
        #   (2) beta -> pi: R = Rz(alpha) @ Ry(pi) @ Rz(gamma)
        #                  = Rz(alpha - gamma) @ Ry(pi)   (since Ry(pi) conjugates Rz)
        #
        # We fix the gauge by setting gamma = 0 and folding the residual z-rotation
        # into alpha, using stable atan2 formulas on the (x,y) block.
        alpha_beta0 = self._safe_atan2(rot_mat[..., 1, 0], rot_mat[..., 0, 0])
        alpha_betapi = self._safe_atan2(-rot_mat[..., 1, 0], -rot_mat[..., 0, 0])
        alpha_singular = torch.where(cos_beta > 0.0, alpha_beta0, alpha_betapi)

        alpha = torch.where(not_singular, alpha, alpha_singular)
        gamma = torch.where(not_singular, gamma, torch.zeros_like(gamma))
        return alpha, beta, gamma

    def _safe_atan2(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Safe atan2 that avoids NaN gradients at (y, x) = (0, 0).

        Perturbing x by eps when ||(x,y)|| is tiny avoids NaNs without
        affecting non-degenerate rotations.

        Parameters
        ----------
        y : torch.Tensor
            Numerator with shape (...).
        x : torch.Tensor
            Denominator with shape (...).

        Returns
        -------
        torch.Tensor
            arctan(y, x) with shape (...).
        """
        mag2 = x * x + y * y
        x_safe = x + (mag2 < self.eps).to(dtype=x.dtype) * self.eps
        return torch.atan2(y, x_safe)

    def _compute_j_matrix(self, l: int) -> torch.Tensor:
        """
        Compute J_l = D^{(l)}(Rx(pi/2)) in the real spherical harmonics basis.

        Computed on CPU with float64 for numerical precision.

        This matrix enables the ZYZ factorization::

            D^{(l)}(R) = Z(alpha) @ J^T @ Z(beta) @ J @ Z(gamma)

        Parameters
        ----------
        l
            Angular momentum order.

        Returns
        -------
        torch.Tensor
            J_l with shape (2l+1, 2l+1) in float64 on CPU.
        """
        dim = 2 * l + 1

        # === Step 1. Extract ZYZ Euler angles for Rx(pi/2) ===
        # Rx(pi/2): [[1,0,0], [0,0,-1], [0,1,0]]
        Rx90 = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
            device="cpu",
        )
        alpha_t, beta_t, gamma_t = self._extract_zyz_euler(Rx90.unsqueeze(0))
        alpha = float(alpha_t[0].item())
        beta = float(beta_t[0].item())
        gamma = float(gamma_t[0].item())

        # === Step 2. Build D^{(l)} in complex basis ===
        # D^l_{m1,m2} = exp(-i*m1*alpha) * d^l_{m1,m2}(beta) * exp(-i*m2*gamma)
        #            = exp(-i*(m1*alpha + m2*gamma)) * d^l_{m1,m2}(beta)
        D_complex = torch.zeros(dim, dim, dtype=torch.complex128, device="cpu")
        for m1 in range(-l, l + 1):
            for m2 in range(-l, l + 1):
                d_elem = self._wigner_d_y_element(l, m1, m2, beta)
                phase = math.cos(m1 * alpha + m2 * gamma) - 1j * math.sin(
                    m1 * alpha + m2 * gamma
                )
                D_complex[m1 + l, m2 + l] = phase * d_elem

        # === Step 3. Build change-of-basis matrix C (complex -> real SH) ===
        # Y_{l,0}^{real} = Y_l^0
        # Y_{l,m}^{real} = (Y_l^m + (-1)^m Y_l^{-m}) / sqrt(2),     m > 0
        # Y_{l,-m}^{real} = (Y_l^m - (-1)^m Y_l^{-m}) / (i*sqrt(2)), m > 0
        # With this convention, C is unitary and C^{-1} = C^H.
        C = torch.zeros(dim, dim, dtype=torch.complex128, device="cpu")
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        for m in range(-l, l + 1):
            row = m + l
            if m == 0:
                C[row, l] = 1.0
            elif m > 0:
                C[row, m + l] = inv_sqrt2
                C[row, -m + l] = ((-1) ** m) * inv_sqrt2
            else:
                C[row, -m + l] = -1j * inv_sqrt2
                C[row, m + l] = ((-1) ** m) * 1j * inv_sqrt2

        # === Step 4. Convert to real basis: D_real = C @ D_complex @ C^H ===
        C_inv = C.conj().transpose(-1, -2)
        D_real = (C @ D_complex @ C_inv).real

        return D_real

    def _wigner_d_y_element(self, l: int, m1: int, m2: int, beta: float) -> float:
        """
        Compute Wigner d-matrix element d^l_{m1,m2}(beta) for y-axis rotation.

        This is the standard closed-form sum (Varshalovich et al.)::

            d^l_{m1,m2}(beta)
                = sum_s (-1)^{m1-m2+s}
                  * sqrt((l+m2)!(l-m2)!(l+m1)!(l-m1)!)
                  / ((l+m2-s)! s! (m1-m2+s)! (l-m1-s)!)
                  * cos(beta/2)^{2l+m2-m1-2s}
                  * sin(beta/2)^{m1-m2+2s}

        Parameters
        ----------
        l
            Angular momentum order.
        m1
            Row index in [-l, ..., l].
        m2
            Column index in [-l, ..., l].
        beta
            Rotation angle about y-axis in radian.

        Returns
        -------
        float
            The matrix element d^l_{m1,m2}(beta).
        """
        # === Step 1. Validate Indices ===
        # Outside the representation range, the element is identically zero.
        if abs(m1) > l or abs(m2) > l:
            return 0.0

        # === Step 2. Precompute Half-Angle Terms ===
        # The closed-form sum uses powers of cos(beta/2) and sin(beta/2).
        cos_b = math.cos(beta / 2.0)
        sin_b = math.sin(beta / 2.0)

        # === Step 3. Determine Valid Summation Range for s ===
        # Denominator factorials require all arguments to be non-negative:
        #   (l + m2 - s)!   => s <= l + m2
        #   s!              => s >= 0
        #   (m1 - m2 + s)!  => s >= m2 - m1
        #   (l - m1 - s)!   => s <= l - m1
        # Therefore:
        #   s_min = max(0, m2 - m1)
        #   s_max = min(l + m2, l - m1)
        s_min = max(0, m2 - m1)
        s_max = min(l + m2, l - m1)

        # === Step 4. Precompute the Factorial Prefactor ===
        # pref = sqrt((l+m2)!(l-m2)!(l+m1)!(l-m1)!)
        pref = math.sqrt(
            math.factorial(l + m2)
            * math.factorial(l - m2)
            * math.factorial(l + m1)
            * math.factorial(l - m1)
        )

        # === Step 5. Accumulate the Closed-Form Sum ===
        out = 0.0
        for s in range(s_min, s_max + 1):
            # denom = (l+m2-s)! s! (m1-m2+s)! (l-m1-s)!
            denom = (
                math.factorial(l + m2 - s)
                * math.factorial(s)
                * math.factorial(m1 - m2 + s)
                * math.factorial(l - m1 - s)
            )
            # sign = (-1)^(m1 - m2 + s)
            sign = -1.0 if ((m1 - m2 + s) % 2) else 1.0

            # Powers of half-angle terms:
            #   p_cos = 2l + m2 - m1 - 2s
            #   p_sin = m1 - m2 + 2s
            p_cos = 2 * l + m2 - m1 - 2 * s
            p_sin = m1 - m2 + 2 * s
            out += sign * pref / denom * (cos_b**p_cos) * (sin_b**p_sin)
        return out


class WignerDCalculator(WignerDCalcBase):
    """
    Fast Wigner-D blocks in the real spherical harmonics (tesseral) basis.

    This module assembles all D^l blocks into a single block-diagonal matrix
    and computes them in one batched matmul chain. This reduces Python dispatch
    overhead but requires O(n_edges * dim_full^2) memory where dim_full = (lmax+1)^2.

    Notes
    -----
    **Conventions**

    - ``rot_mat`` is a batch of 3x3 rotation matrices with shape ``(n_edges, 3, 3)``.
      It is a global->local transform for 3D vectors::

        v_local = rot_mat @ v_global

    - Euler angles follow the ZYZ convention::

        rot_mat = Rz(alpha) @ Ry(beta) @ Rz(gamma)

      with ``beta in [0, pi]``. Singular cases ``sin(beta) ~ 0`` are resolved by
      setting ``gamma = 0`` and folding the residual z-rotation into ``alpha``
      (stable and differentiable).

    - Within each degree ``l``, channels are ordered by ``m = -l, ..., +l``.
      Index mapping: ``i = m + l``.

    **Representation Matrices**

    - In the complex spherical harmonics basis, z-axis rotations are diagonal::

        D_z^{(l)}(theta)_{m1,m2} = delta_{m1,m2} * exp(-i * m1 * theta)

    - In the real (tesseral) basis used here, each pair ``{+m, -m}`` (for ``m>0``)
      forms a 2x2 rotation by angle ``m*theta``; ``m=0`` is invariant. This block is
      denoted as ``Z^{(l)}(theta)``.

    **Efficient Y-Rotation via Conjugation**

    For ZYZ Euler angles, the Wigner-D matrix is::

        D^{(l)}(rot_mat) = Z^{(l)}(alpha) @ D_y^{(l)}(beta) @ Z^{(l)}(gamma)

    The expensive part is the y-axis rotation ``D_y^{(l)}(beta)``. We use the identity::

        Ry(beta) = Rx(pi/2)^{-1} @ Rz(beta) @ Rx(pi/2)

    Define a per-degree constant::

        J_l = D_x ^ {(l)}(pi / 2)

    In the real basis, representation matrices are orthogonal, so::

        D_x^{(l)}(-pi/2) = J_l^{-1} = J_l^T

    Therefore::

        D_y^{(l)}(beta) = J_l^T @ Z^{(l)}(beta) @ J_l

    and the full block becomes::

        D^{(l)}(rot_mat) = Z^{(l)}(alpha) @ J_l^T @ Z^{(l)}(beta) @ J_l @ Z^{(l)}(gamma)

    **Block-Diagonal Parallel Computation**

    Instead of computing each l separately, we assemble all blocks into a single
    block-diagonal matrix of dimension ``dim_full = (lmax+1)^2``::

        J_full = diag(J_0, J_1, ..., J_lmax)
        Z_full(theta) = diag(Z^{(0)}(theta), Z^{(1)}(theta), ..., Z^{(lmax)}(theta))

    The full computation becomes::

        D_full = Z_full(alpha) @ J_full ^ T @ Z_full(beta) @ J_full @ Z_full(gamma)

    This reduces ``lmax+1`` separate matmul chains to a single chain on larger matrices.

    **Index Layout**

    - Block l occupies rows/columns [l^2, (l+1)^2) in the full matrix.
    - Within block l, the m=0 element is at position l^2 + l (center of block).
    - For m > 0, the 2x2 rotation sub-block occupies:
      - pos = l^2 + (l + m) for +m
      - neg = l^2 + (l - m) for -m

    **Outputs and Usage**

    - ``D_full`` has shape ``(n_edges, D, D)`` and is orthogonal. It represents the
      same global->local rotation as ``rot_mat``.
    - ``Dt_full = D_full.transpose(-1, -2)`` is the inverse (local->global).
    - Apply to packed features as::

        x_local = D_full @ x_global
        x_global = Dt_full @ x_local

    - For degree ``l``, the block is sliced by ``[l^2 : (l+1)^2]`` along both axes.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum degree.
    eps : float
        Small epsilon for numerical stability.
    dtype : torch.dtype
        Floating-point dtype for output matrices.
    use_triton : bool
        If True and Triton is available, use fused Triton kernel for Z-rotation
        matrix construction. Only effective on CUDA devices. Backward uses
        PyTorch trig inside the custom autograd wrapper.
    """

    def __init__(
        self,
        lmax: int,
        *,
        eps: float = 1e-7,
        dtype: torch.dtype,
        use_triton: bool = False,
    ) -> None:
        super().__init__(lmax, eps=eps, dtype=dtype)
        self.use_triton = use_triton

        # === Step 1. Compute block dimension ===
        # dim_full = sum_{l=0..lmax}(2l+1) = (lmax+1)^2
        # Block l occupies indices [l^2, (l+1)^2) in the full matrix.
        self.dim_full = (self.lmax + 1) ** 2

        # === Step 2. Build J_full as block-diagonal matrix ===
        # J_full contains J_l = D^{(l)}(Rx(pi/2)) on diagonal blocks
        J_full = torch.zeros(
            self.dim_full,
            self.dim_full,
            dtype=self.dtype,
            device=self.device,
        )
        for l in range(self.lmax + 1):
            J_l = self._compute_j_matrix(l).to(device=self.device, dtype=self.dtype)
            start, end = l * l, (l + 1) * (l + 1)
            J_full[start:end, start:end] = J_l

        self.register_buffer("J_full", J_full, persistent=True)
        self.register_buffer("Jt_full", J_full.T.contiguous(), persistent=True)

        # === Step 3. Precompute indices for Z_full construction ===
        self._precompute_z_indices()

    def forward(self, rot_mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Wigner-D blocks for a batch of rotation matrices.

        Parameters
        ----------
        rot_mat : torch.Tensor
            Rotation matrices with shape (n_edges, 3, 3), global->local.

        Returns
        -------
        D_full : torch.Tensor
            Block-diagonal matrix with shape (n_edges, D, D) where D=(lmax+1)^2.
        Dt_full : torch.Tensor
            Transpose of D_full.
        """
        rot_mat = rot_mat.to(dtype=self.dtype)

        # === Step 1. Extract ZYZ Euler angles ===
        # Convention: rot_mat = Rz(alpha) @ Ry(beta) @ Rz(gamma)
        alpha, beta, gamma = self._extract_zyz_euler(rot_mat)

        # === Step 2. Build block-diagonal Z matrices ===
        # Each Z_full has shape (n_edges, dim_full, dim_full)
        Za_full = self._build_z_rotation(alpha)
        Zb_full = self._build_z_rotation(beta)
        Zc_full = self._build_z_rotation(gamma)

        # === Step 3. Compute D_full via single matmul chain ===
        # D^{(l)}(R) = Z(alpha) @ J^T @ Z(beta) @ J @ Z(gamma)
        J_full: torch.Tensor = self.J_full
        Jt_full: torch.Tensor = self.Jt_full
        D_full = Za_full @ Jt_full @ Zb_full @ J_full @ Zc_full
        Dt_full = D_full.transpose(-1, -2).contiguous()

        return D_full, Dt_full

    def serialize(self) -> dict[str, Any]:
        """Serialize WignerDCalculator (lmax and dtype are stored by parent)."""
        return {
            "@class": "WignerDCalculator",
            "@version": 1,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> WignerDCalcBase:
        """Deserialize WignerDCalculator - parent handles lmax/dtype reconstruction."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "WignerDCalculator":
            raise ValueError(f"Invalid class for WignerDCalculator: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported WignerDCalculator version: {version}")
        raise NotImplementedError(
            "WignerDCalculator.deserialize should be called by parent with lmax/dtype"
        )

    def _build_z_rotation(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Build block-diagonal Z rotation matrix in real spherical harmonics basis.

        The Z matrix is block-diagonal with blocks Z^{(l)}(theta) for l=0..lmax.
        Each block implements z-axis rotation in the real SH basis.

        In the complex spherical harmonics basis, z-axis rotation is diagonal::

            D_z^{(l)}(theta)_{m1,m2} = delta_{m1,m2} * exp(-i * m1 * theta)

        In the real (tesseral) basis, each pair (m, -m) for m > 0 forms a 2x2
        rotation block by angle m*theta; m=0 is invariant.

        Parameters
        ----------
        angle : torch.Tensor
            Rotation angles with shape (n_edges,).

        Returns
        -------
        torch.Tensor
            Block-diagonal rotation matrices with shape (n_edges, dim_full, dim_full).
        """
        m0_indices: torch.Tensor = self.m0_indices
        m_values: torch.Tensor = self.m_values
        pos_indices: torch.Tensor = self.pos_indices
        neg_indices: torch.Tensor = self.neg_indices

        # === Step 1. Triton path (CUDA only, skip during JIT scripting) ===
        if not torch.jit.is_scripting() and self.use_triton and angle.is_cuda:
            return build_z_rotation_triton(
                angle,
                dim_full=self.dim_full,
                m0_indices=m0_indices,
                pos_indices=pos_indices,
                neg_indices=neg_indices,
                m_values=m_values,
            )

        # === Step 2. PyTorch fallback: allocate Z matrix ===
        n_edges = angle.shape[0]
        Z = torch.zeros(
            n_edges,
            self.dim_full,
            self.dim_full,
            dtype=angle.dtype,
            device=angle.device,
        )

        # === Step 3. Set m=0 diagonal elements to 1 ===
        # Z[:, m0_idx, m0_idx] = 1 for each l's center element
        Z[:, m0_indices, m0_indices] = 1.0

        # === Step 4. Fill m>0 rotation blocks ===
        if m_values.numel() > 0:
            # Compute cos(m*angle) and sin(m*angle) for all (l,m) pairs
            # angles_m: (n_edges, n_blocks) where n_blocks = total (l,m) pairs with m>0
            angles_m = angle[:, None] * m_values[None, :]  # (n_edges, n_blocks)
            c = torch.cos(angles_m)
            s = torch.sin(angles_m)

            # Fill the 2x2 rotation blocks:
            #   Z[pos, pos] =  cos(m*theta)
            #   Z[neg, neg] =  cos(m*theta)
            #   Z[pos, neg] =  sin(m*theta)
            #   Z[neg, pos] = -sin(m*theta)
            Z[:, pos_indices, pos_indices] = c
            Z[:, neg_indices, neg_indices] = c
            Z[:, pos_indices, neg_indices] = s
            Z[:, neg_indices, pos_indices] = -s

        return Z

    def _precompute_z_indices(self) -> None:
        """
        Precompute index arrays for efficient Z_full construction.

        For each block l in the block-diagonal Z matrix:
        - m=0 element is at diagonal position l^2 + l
        - For m > 0, the 2x2 rotation block occupies positions:
            pos = l^2 + (l + m)  ->  +m row/column
            neg = l^2 + (l - m)  ->  -m row/column

        The 2x2 block structure in real spherical harmonics basis::

            Z[pos, pos] = cos(m * theta)
            Z[neg, neg] = cos(m * theta)
            Z[pos, neg] = sin(m * theta)
            Z[neg, pos] = -sin(m * theta)
        """
        # === Step 1. Indices for m=0 diagonal elements ===
        m0_indices = torch.tensor(
            [so3_packed_index(l, 0) for l in range(self.lmax + 1)],
            dtype=torch.long,
            device=self.device,
        )
        self.register_buffer("m0_indices", m0_indices, persistent=True)

        # === Step 2. Indices for m>0 rotation blocks ===
        pos_indices: list[int] = []
        neg_indices: list[int] = []
        m_values: list[int] = []

        for l in range(1, self.lmax + 1):  # l=0 has no m>0 terms
            offset_l = l * l
            for m in range(1, l + 1):
                pos = offset_l + (l + m)  # index for +m
                neg = offset_l + (l - m)  # index for -m
                pos_indices.append(pos)
                neg_indices.append(neg)
                m_values.append(m)

        self.register_buffer(
            "pos_indices",
            torch.tensor(pos_indices, dtype=torch.long, device=self.device),
            persistent=True,
        )
        self.register_buffer(
            "neg_indices",
            torch.tensor(neg_indices, dtype=torch.long, device=self.device),
            persistent=True,
        )
        self.register_buffer(
            "m_values",
            torch.tensor(m_values, dtype=self.dtype, device=self.device),
            persistent=True,
        )
