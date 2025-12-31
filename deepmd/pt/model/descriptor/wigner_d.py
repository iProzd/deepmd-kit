# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wigner-D utilities for SO(3) equivariant descriptors (PyTorch backend).

Two implementations are provided:
- ``WignerDCalc``: Per-l loop implementation, memory efficient for large n_edges.
- ``WignerDCalcParallel``: Block-diagonal parallel implementation, faster for small n_edges
  but requires O(n_edges * dim_full^2) memory where dim_full = (lmax+1)^2.
"""

from __future__ import (
    annotations,
)

import math
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
)

import torch
import torch.nn as nn

from deepmd.pt.utils import (
    env,
)


class WignerDCalcBase(nn.Module, ABC):
    """
    Abstract base class for Wigner-D matrix calculators.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum degree.
    dtype : torch.dtype
        Floating-point dtype for output matrices.
    """

    def __init__(self, lmax: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.lmax = int(lmax)
        if self.lmax < 0:
            raise ValueError("`lmax` must be non-negative")
        self.dtype = dtype
        self.device = env.DEVICE
        self.eps = torch.finfo(self.dtype).eps

    @abstractmethod
    def forward(
        self, rot_mat: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute Wigner-D blocks for a batch of rotation matrices.

        Parameters
        ----------
        rot_mat : torch.Tensor
            Rotation matrices with shape (n_edges, 3, 3), global->local.

        Returns
        -------
        D_list : list[torch.Tensor]
            List of D^l blocks, ``D_list[l]`` has shape (n_edges, 2l+1, 2l+1).
        Dt_list : list[torch.Tensor]
            Transpose blocks, ``Dt_list[l]`` has shape (n_edges, 2l+1, 2l+1).
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
            beta = acos(R[2, 2])
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
        # === Step 1. Compute beta from R[2,2] ===
        # Clamp avoids acos domain errors from tiny numerical drift.
        cos_beta = rot_mat[..., 2, 2].clamp(-1.0, 1.0)
        beta = torch.acos(cos_beta)

        # === Step 2. Detect singular cases via sin(beta) ===
        # sin(beta) = sqrt(1 - cos(beta)^2) is stable after clamping.
        sin_beta = torch.sqrt((1.0 - cos_beta * cos_beta).clamp(min=0.0))

        threshold = math.sqrt(self.eps)
        not_singular = sin_beta > threshold

        # === Step 3. Define a safe atan2 for autograd ===
        # torch.atan2(y, x) has undefined gradient at (y, x) = (0, 0).
        # Perturbing x by eps when ||(x,y)|| is tiny avoids NaNs without
        # affecting non-degenerate rotations.
        def safe_atan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            mag2 = x * x + y * y
            x_safe = x + (mag2 < self.eps).to(dtype=x.dtype) * self.eps
            return torch.atan2(y, x_safe)

        # === Step 4. Non-singular extraction (sin(beta) > 0) ===
        alpha = safe_atan2(rot_mat[..., 1, 2], rot_mat[..., 0, 2])
        gamma = safe_atan2(rot_mat[..., 2, 1], -rot_mat[..., 2, 0])

        # === Step 5. Singular extraction (gimbal lock) ===
        # When sin(beta) -> 0, alpha and gamma are not individually identifiable.
        # Two singular manifolds exist:
        #   (1) beta -> 0:  R = Rz(alpha) @ I @ Rz(gamma) = Rz(alpha + gamma)
        #   (2) beta -> pi: R = Rz(alpha) @ Ry(pi) @ Rz(gamma)
        #                  = Rz(alpha - gamma) @ Ry(pi)   (since Ry(pi) conjugates Rz)
        #
        # We fix the gauge by setting gamma = 0 and folding the residual z-rotation
        # into alpha, using stable atan2 formulas on the (x,y) block.
        alpha_beta0 = safe_atan2(rot_mat[..., 1, 0], rot_mat[..., 0, 0])
        alpha_betapi = safe_atan2(-rot_mat[..., 1, 0], -rot_mat[..., 0, 0])
        alpha_singular = torch.where(cos_beta > 0.0, alpha_beta0, alpha_betapi)

        alpha = torch.where(not_singular, alpha, alpha_singular)
        gamma = torch.where(not_singular, gamma, torch.zeros_like(gamma))
        return alpha, beta, gamma

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
            J_l with shape (2l+1, 2l+1) in self.dtype on CPU.
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

        return D_real.to(dtype=self.dtype)

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


class WignerDCalc(WignerDCalcBase):
    """
    Fast Wigner-D blocks in the real spherical harmonics (tesseral) basis.

    This module precomputes constant J matrices as buffers and constructs
    per-edge D^l blocks from Euler angles using only batched matmuls.

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

    **Outputs and Usage**

    - ``D_list[l]`` has shape ``(n_edges, 2l+1, 2l+1)`` and is orthogonal. It
      represents the same global->local rotation as ``rot_mat``.
    - ``Dt_list[l] = D_list[l].transpose(-1, -2)`` is its inverse.
    - Apply blocks to per-degree features as::

        x_local^{(l)}  = D_list[l]  @ x_global^{(l)}
        x_global^{(l)} = Dt_list[l] @ x_local^{(l)}

    Parameters
    ----------
    lmax : int
        Maximum angular momentum degree.
    dtype : torch.dtype
        Floating-point dtype for output matrices.
    """

    def __init__(self, lmax: int, *, dtype: torch.dtype) -> None:
        super().__init__(lmax, dtype=dtype)

        # Precompute J_l = D^{(l)}(Rx(pi/2)) on CPU, then move to target device
        # Also cache J_l^T (the transpose in real basis equals the inverse)
        for l in range(self.lmax + 1):
            J = self._compute_j_matrix(l).to(device=self.device)
            Jt = J.transpose(-1, -2).contiguous()
            self.register_buffer(f"_J_{l}", J, persistent=True)
            self.register_buffer(f"_Jt_{l}", Jt, persistent=True)

    def forward(
        self, rot_mat: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute Wigner-D blocks for a batch of rotation matrices.

        Parameters
        ----------
        rot_mat : torch.Tensor
            Rotation matrices with shape (n_edges, 3, 3), global->local.

        Returns
        -------
        D_list : list[torch.Tensor]
            List of D^l blocks, ``D_list[l]`` has shape (n_edges, 2l+1, 2l+1).
        Dt_list : list[torch.Tensor]
            Transpose blocks, ``Dt_list[l]`` has shape (n_edges, 2l+1, 2l+1).
        """
        # === Step 1. Extract ZYZ Euler angles ===
        # Convention: rot_mat = Rz(alpha) @ Ry(beta) @ Rz(gamma)
        alpha, beta, gamma = self._extract_zyz_euler(rot_mat)

        # === Step 2. Build real-basis Wigner-D blocks for each l ===
        # D^{(l)}(R) = Z(alpha) @ J^T @ Z(beta) @ J @ Z(gamma)
        D_list: list[torch.Tensor] = []
        Dt_list: list[torch.Tensor] = []

        for l in range(self.lmax + 1):
            J: torch.Tensor = getattr(self, f"_J_{l}")
            Jt: torch.Tensor = getattr(self, f"_Jt_{l}")

            Za = self._build_z_rotation(alpha, l)
            Zb = self._build_z_rotation(beta, l)
            Zc = self._build_z_rotation(gamma, l)

            # J_l = D_x^{(l)}(pi/2) in the real basis, hence:
            #   D_x^{(l)}(-pi/2) = J_l^{-1} = J_l^T
            # and the conjugation identity becomes:
            #   D_y^{(l)}(beta) = J_l^T @ Z^{(l)}(beta) @ J_l
            D = Za @ Jt @ Zb @ J @ Zc
            D_list.append(D.contiguous())
            Dt_list.append(D.transpose(-1, -2).contiguous())

        return D_list, Dt_list

    def serialize(self) -> dict[str, Any]:
        """Serialize WignerDCalc (lmax and dtype are stored by parent)."""
        return {
            "@class": "WignerDCalc",
            "@version": 1,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> WignerDCalcBase:
        """Deserialize WignerDCalc - parent handles lmax/dtype reconstruction."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls not in ("WignerDCalc", "WignerDCalcParallel"):
            raise ValueError(f"Invalid class for WignerDCalc: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported WignerDCalc version: {version}")
        # Parent must reconstruct with actual lmax and dtype
        raise NotImplementedError(
            "WignerDCalc.deserialize should be called by parent with lmax/dtype"
        )

    def _build_z_rotation(self, angle: torch.Tensor, l: int) -> torch.Tensor:
        """
        Build z-rotation matrix in the real spherical harmonics basis.

        In the complex spherical harmonics basis, a z-axis rotation is diagonal:

            D_z^{(l)}(theta)_{m1,m2} = delta_{m1,m2} * exp(-i * m1 * theta)

        In the real (tesseral) basis used here, each pair (m, -m) for m > 0
        spans a 2D subspace that rotates by angle m*theta.

        For each |m| > 0, there's a 2x2 rotation block; for m = 0, the element is 1.

        In real SH basis with ordering m = -l, ..., 0, ..., l::

            Z[l + m, l + m] = cos(m * θ)
            Z[l + m, l - m] = sin(m * θ)
            Z[l - m, l + m] = -sin(m * θ)
            Z[l - m, l - m] = cos(m * θ)

        Parameters
        ----------
        angle : torch.Tensor
            Rotation angles with shape (...,).
        l : int
            Angular momentum order.

        Returns
        -------
        torch.Tensor
            Rotation matrices with shape (..., 2l+1, 2l+1).
        """
        # === Step 1. Handle l=0 Special Case ===
        if l == 0:
            return torch.ones(
                (*angle.shape, 1, 1), dtype=angle.dtype, device=angle.device
            )

        dim = 2 * l + 1
        Z = torch.zeros(
            (*angle.shape, dim, dim), dtype=angle.dtype, device=angle.device
        )

        # === Step 2. Vectorized m Generation ===
        # m_idx: [1, 2, ..., l] for indexing
        # m_float: same values as float for computation
        m_idx = torch.arange(1, l + 1, device=angle.device, dtype=torch.long)
        m_float = m_idx.to(dtype=angle.dtype)

        # === Step 3. Batched Trigonometric Computation ===
        # angle: (...,) -> (..., l) via broadcasting with m_float
        angles_m = angle.unsqueeze(-1) * m_float
        c = torch.cos(angles_m)  # (..., l)
        s = torch.sin(angles_m)  # (..., l)

        # === Step 4. Set Center Element ===
        Z[..., l, l] = 1.0

        # === Step 5. Advanced Indexing for Parallel Fill ===
        idx_pos = l + m_idx  # l + m
        idx_neg = l - m_idx  # l - m

        # Each 2x2 block in ordered subspace [m=-m, m=+m]:
        #   [[ cos(mθ), -sin(mθ)],
        #    [ sin(mθ),  cos(mθ)]]
        Z[..., idx_pos, idx_pos] = c
        Z[..., idx_neg, idx_neg] = c
        Z[..., idx_pos, idx_neg] = s
        Z[..., idx_neg, idx_pos] = -s

        return Z


class WignerDCalcParallel(WignerDCalcBase):
    """
    Fast Wigner-D blocks using block-diagonal parallel computation.

    This implementation assembles all D^l blocks into a single block-diagonal matrix
    and computes them in one batched matmul chain. This reduces Python dispatch overhead
    but requires O(n_edges * dim_full^2) memory where dim_full = (lmax+1)^2.

    For large n_edges, use ``WignerDCalc`` instead which processes each l separately.

    Notes
    -----
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

    See ``WignerDCalc`` docstring for conventions and mathematical details.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum degree.
    dtype : torch.dtype
        Floating-point dtype for output matrices.
    """

    def __init__(self, lmax: int, *, dtype: torch.dtype) -> None:
        super().__init__(lmax, dtype=dtype)

        # === Step 1. Compute block dimension ===
        # dim_full = sum_{l=0..lmax}(2l+1) = (lmax+1)^2
        # Block l occupies indices [l^2, (l+1)^2) in the full matrix.
        self.dim_full = (self.lmax + 1) ** 2

        # === Step 2. Build J_full as block-diagonal matrix ===
        # J_full contains J_l = D^{(l)}(Rx(pi/2)) on diagonal blocks
        J_full = torch.zeros(
            self.dim_full, self.dim_full, dtype=self.dtype, device=self.device
        )
        for l in range(self.lmax + 1):
            J_l = self._compute_j_matrix(l).to(device=self.device)
            start, end = l * l, (l + 1) * (l + 1)
            J_full[start:end, start:end] = J_l

        self.register_buffer("_J_full", J_full, persistent=True)
        self.register_buffer("_Jt_full", J_full.T.contiguous(), persistent=True)

        # === Step 3. Precompute indices for Z_full construction ===
        self._precompute_z_indices()

    def forward(
        self, rot_mat: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute Wigner-D blocks for a batch of rotation matrices.

        Parameters
        ----------
        rot_mat : torch.Tensor
            Rotation matrices with shape (n_edges, 3, 3), global->local.

        Returns
        -------
        D_list : list[torch.Tensor]
            List of D^l blocks, ``D_list[l]`` has shape (n_edges, 2l+1, 2l+1).
        Dt_list : list[torch.Tensor]
            Transpose blocks, ``Dt_list[l]`` has shape (n_edges, 2l+1, 2l+1).
        """
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
        J_full: torch.Tensor = getattr(self, "_J_full")
        Jt_full: torch.Tensor = getattr(self, "_Jt_full")
        D_full = Za_full @ Jt_full @ Zb_full @ J_full @ Zc_full

        # === Step 4. Slice D_full into per-l blocks ===
        # Block l occupies indices [l^2, (l+1)^2) in the full matrix.
        D_list: list[torch.Tensor] = []
        Dt_list: list[torch.Tensor] = []
        for l in range(self.lmax + 1):
            start = l * l
            end = (l + 1) * (l + 1)
            D_l = D_full[:, start:end, start:end].contiguous()
            D_list.append(D_l)
            Dt_list.append(D_l.transpose(-1, -2).contiguous())

        return D_list, Dt_list

    def serialize(self) -> dict[str, Any]:
        """Serialize WignerDCalcParallel (lmax and dtype are stored by parent)."""
        return {
            "@class": "WignerDCalcParallel",
            "@version": 1,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> WignerDCalcBase:
        """Deserialize WignerDCalcParallel - parent handles lmax/dtype reconstruction."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls not in ("WignerDCalc", "WignerDCalcParallel"):
            raise ValueError(f"Invalid class for WignerDCalcParallel: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported WignerDCalcParallel version: {version}")
        raise NotImplementedError(
            "WignerDCalcParallel.deserialize should be called by parent with lmax/dtype"
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
        n_edges = angle.shape[0]
        Z = torch.zeros(
            n_edges,
            self.dim_full,
            self.dim_full,
            dtype=angle.dtype,
            device=angle.device,
        )

        # === Step 1. Set m=0 diagonal elements to 1 ===
        # Z[:, m0_idx, m0_idx] = 1 for each l's center element
        m0_indices: torch.Tensor = getattr(self, "_m0_indices")
        Z[:, m0_indices, m0_indices] = 1.0

        # === Step 2. Fill m>0 rotation blocks ===
        m_values: torch.Tensor = getattr(self, "_m_values")
        if m_values.numel() > 0:
            pos_indices: torch.Tensor = getattr(self, "_pos_indices")
            neg_indices: torch.Tensor = getattr(self, "_neg_indices")

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
            [l * l + l for l in range(self.lmax + 1)],
            dtype=torch.long,
            device=self.device,
        )
        self.register_buffer("_m0_indices", m0_indices, persistent=True)

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
            "_pos_indices",
            torch.tensor(pos_indices, dtype=torch.long, device=self.device),
            persistent=True,
        )
        self.register_buffer(
            "_neg_indices",
            torch.tensor(neg_indices, dtype=torch.long, device=self.device),
            persistent=True,
        )
        self.register_buffer(
            "_m_values",
            torch.tensor(m_values, dtype=self.dtype, device=self.device),
            persistent=True,
        )
