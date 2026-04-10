# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Archive of the pre-fix SeZM helper implementations that were identified as non-smooth.

This module intentionally preserves only the historical implementations that were found
to introduce force/PES roughness in SeZM:

- `WignerDCalculator`
- `init_edge_rot_mat`
- `init_edge_rot_mat_frisvad`

Shared helpers that are still valid are imported from the split `sezm_nn` package.
Only the archived buggy implementations themselves are kept in this file.

Do not use this module for production code. The live implementation should stay in
`sezm_nn`.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    Any,
)

import torch
import torch.nn as nn

from deepmd.pt.utils import (
    env,
)

from .indexing import (
    so3_packed_index,
)
from .utils import (
    nvtx_range,
    safe_norm,
)


class WignerDCalculator(nn.Module):
    """
    Fast Wigner-D blocks in the real spherical harmonics (tesseral) basis.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum degree.
    eps : float
        Small epsilon for numerical stability.
    dtype : torch.dtype
        Floating-point dtype for output matrices.

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
    """

    def __init__(
        self,
        lmax: int,
        *,
        eps: float = 1e-7,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        if self.lmax < 0:
            raise ValueError("`lmax` must be non-negative")
        self.dtype = dtype
        self.device = env.DEVICE
        self.eps = float(eps)

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
            Rotation matrices with shape (E, 3, 3), global->local.

        Returns
        -------
        D_full : torch.Tensor
            Block-diagonal matrix with shape (E, D, D) where D=(lmax+1)^2.
        Dt_full : torch.Tensor
            Transpose of D_full.
        """
        rot_mat = rot_mat.to(dtype=self.dtype)

        # === Step 1. Extract ZYZ Euler angles ===
        # Convention: rot_mat = Rz(alpha) @ Ry(beta) @ Rz(gamma)
        with nvtx_range("WignerD/euler"):
            alpha, beta, gamma = self._extract_zyz_euler(rot_mat)  # (E,), (E,), (E,)

        # === Step 2. Build block-diagonal Z matrices ===
        # Each Z_full has shape (E, dim_full, dim_full)
        with nvtx_range("WignerD/z_rotation"):
            Za_full = self._build_z_rotation(alpha)  # (E, D, D)
            Zb_full = self._build_z_rotation(beta)  # (E, D, D)
            Zc_full = self._build_z_rotation(gamma)  # (E, D, D)

        # === Step 3. Compute D_full via single matmul chain ===
        # D^{(l)}(R) = Z(alpha) @ J^T @ Z(beta) @ J @ Z(gamma)
        with nvtx_range("WignerD/matmul"):
            J_full = self.J_full  # (D, D)
            Jt_full = self.Jt_full  # (D, D)
            D_full = Za_full @ Jt_full @ Zb_full @ J_full @ Zc_full  # (E, D, D)
            Dt_full = D_full.transpose(-1, -2).contiguous()  # (E, D, D)

        return D_full, Dt_full

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
            Rotation angles with shape (E,).

        Returns
        -------
        torch.Tensor
            Block-diagonal rotation matrices with shape (E, D, D).
        """
        m0_indices = self.m0_indices  # (lmax+1,)
        m_values = self.m_values  # (n_blocks,)
        pos_indices = self.pos_indices  # (n_blocks,)
        neg_indices = self.neg_indices  # (n_blocks,)

        # === Step 1. Allocate Z matrix ===
        n_edges = angle.shape[0]
        Z = torch.zeros(
            n_edges,
            self.dim_full,
            self.dim_full,
            dtype=angle.dtype,
            device=angle.device,
        )  # (E, D, D)

        # === Step 2. Set m=0 diagonal elements to 1 ===
        # Z[:, m0_idx, m0_idx] = 1 for each l's center element
        Z[:, m0_indices, m0_indices] = 1.0

        # === Step 3. Fill m>0 rotation blocks ===
        if m_values.numel() > 0:
            # Compute cos(m*angle) and sin(m*angle) for all (l,m) pairs
            # angles_m: (E, n_blocks) where n_blocks = total (l,m) pairs with m>0
            angles_m = angle[:, None] * m_values[None, :]  # (E, n_blocks)
            c = torch.cos(angles_m)  # (E, n_blocks)
            s = torch.sin(angles_m)  # (E, n_blocks)

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

    def serialize(self) -> dict[str, Any]:
        """Serialize WignerDCalculator (lmax and dtype are stored by parent)."""
        return {
            "@class": "WignerDCalculator",
            "@version": 1,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> WignerDCalculator:
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


def init_edge_rot_mat(edge_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation matrices that align each edge to the local + Z axis.

    The returned rotation is a global->local transform: ``v_local = R @ v_global``.
    So, for unit edge direction vector ``u``, ``R @ u = (0, 0, 1)``.

    Notes
    -----
    This routine constructs an orthonormal right-handed frame (x_hat, y_hat, z_hat)
    per edge via a simple Gram-Schmidt process::

        z_hat = edge_vec / ||edge_vec||           # local +z direction
        x_hat = normalize(ref - (ref·z_hat) z_hat)  # orthogonal to z_hat
        y_hat = z_hat x x_hat                       # right-handed

    where ``ref`` is a reference axis that is not nearly colinear with ``z_hat``.

    The rotation matrix stacks these basis vectors as rows::

        R = [x_hat^T; y_hat^T; z_hat^T]

    This makes ``R`` a global->local transform, because each row computes the
    dot product with the corresponding local basis vector.

    The reference-axis switch introduces a piecewise definition. For a smoother
    frame construction (especially for higher-order gradients), consider a
    Householder/Frisvad frame.

    Parameters
    ----------
    edge_vec
        Edge vectors with shape (E, 3).

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape (E, 3, 3).
    """
    # === Step 1. Normalize edge direction (local z) ===
    # z_hat is the unit edge direction (center -> neighbor).
    z_hat = edge_vec / safe_norm(edge_vec)

    # === Step 2. Construct x-axis by Gram-Schmidt against a reference ===
    # Choose a reference axis that is not nearly parallel to z_hat to avoid
    # catastrophic cancellation in the Gram-Schmidt projection.
    candi_1 = torch.tensor(
        [1.0, 0.0, 0.0], dtype=edge_vec.dtype, device=edge_vec.device
    ).expand_as(edge_vec)
    candi_2 = torch.tensor(
        [0.0, 1.0, 0.0], dtype=edge_vec.dtype, device=edge_vec.device
    ).expand_as(edge_vec)
    use_alt = torch.abs(torch.sum(z_hat * candi_1, dim=-1, keepdim=True)) > 0.9
    ref = torch.where(use_alt, candi_2, candi_1)

    # Remove the component along z_hat to obtain a vector orthogonal to z_hat.
    proj = torch.sum(ref * z_hat, dim=-1, keepdim=True) * z_hat
    x_hat = ref - proj
    x_hat = x_hat / safe_norm(x_hat)

    # === Step 3. Construct y-axis (right-handed) ===
    # Cross product enforces a right-handed frame: (x_hat, y_hat, z_hat).
    y_hat = torch.cross(z_hat, x_hat, dim=-1)
    y_hat = y_hat / safe_norm(y_hat)

    # === Step 4. Stack rows to form global->local rotation ===
    # Row-stacking ensures v_local = R @ v_global.
    rot_mat = torch.stack([x_hat, y_hat, z_hat], dim=-2)
    return rot_mat


def init_edge_rot_mat_frisvad(
    edge_vec: torch.Tensor,
    edge_len: torch.Tensor | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute rotation matrices that align each edge to the local + Z axis.

    The returned rotation is a global->local transform: ``v_local = R @ v_global``.
    So, for unit edge direction vector ``u``, ``R @ u = (0, 0, 1)``.

    Notes
    -----
    This routine constructs an orthonormal right-handed frame (x_hat, y_hat, z_hat)
    per edge using the Frisvad method (closed-form ONB from a unit vector).

    The Frisvad closed-form is singular at ``z_hat = (0, 0, -1)``, due to the
    ``1 / (1 + nz)`` denominator. For the singular neighborhood near ``-Z``, the
    basis must NOT fall back to fixed axes, otherwise x_hat/y_hat may not be
    exactly perpendicular to the current ``z_hat``. Instead, we build a strict
    orthonormal pair from the current ``z_hat`` via cross products, guaranteeing
    that the returned matrix is a proper rotation and that ``R @ z_hat = (0,0,1)``
    up to floating-point error.

    Given unit vector z_hat = (nx, ny, nz), for nz > -1, define::

        a = 1 / (1 + nz)
        b = -nx * ny * a
        x_hat = (1 - nx ^ 2 * a, b, -nx)
        y_hat = (b, 1 - ny ^ 2 * a, -ny)

    This yields an orthonormal basis with x_hat ⟂ z_hat, y_hat ⟂ z_hat and
    x_hat X y_hat = z_hat (right-handed). For nz close to -1, we fall back to a
    strict cross-product basis built from the current z_hat.

    The rotation matrix stacks these basis vectors as rows::

        R = [x_hat^T; y_hat^T; z_hat^T]

    This makes ``R`` a global->local transform, because each row computes the
    dot product with the corresponding local basis vector.

    Parameters
    ----------
    edge_vec
        Edge vectors with shape (E, 3).
    edge_len
        Precomputed edge lengths with shape (E, 1). If None, recompute from edge_vec.
    eps
        Small epsilon for numerical stability.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape (E, 3, 3).
    """
    # === Step 1. Normalize edge direction (local z) ===
    # z_hat is the unit edge direction (center -> neighbor).
    if edge_len is None:
        edge_len = safe_norm(edge_vec, eps)
    else:
        edge_len = edge_len.clamp(min=eps)
    z_hat = edge_vec / edge_len  # (E, 3)
    nx = z_hat[..., 0:1]  # (E, 1)
    ny = z_hat[..., 1:2]  # (E, 1)
    nz = z_hat[..., 2:3]  # (E, 1)

    # === Step 2. Frisvad closed-form orthonormal basis (non-singular) ===
    # The closed-form uses a = 1 / (1 + nz), which is singular at nz = -1.
    # Compute it with a safe denominator, then select by a singular mask.
    # Use a fixed threshold for singular detection (1e-6 is sufficient for all precisions).
    singular_threshold = 1.0e-6
    singular = nz < (-1.0 + singular_threshold)  # (E, 1)

    denom = 1.0 + nz  # (E, 1)
    denom_safe = torch.where(singular, torch.ones_like(denom), denom)  # (E, 1)
    a = 1.0 / denom_safe  # (E, 1)
    b = -nx * ny * a  # (E, 1)

    x_main = torch.cat([1.0 - nx * nx * a, b, -nx], dim=-1)  # (E, 3)
    y_main = torch.cat([b, 1.0 - ny * ny * a, -ny], dim=-1)  # (E, 3)

    # === Step 3. Strict fallback for the singular neighborhood (z_hat ~= -Z) ===
    # Build x_hat/y_hat from the current z_hat so that:
    #   x_hat ⟂ z_hat, y_hat ⟂ z_hat, and (x_hat, y_hat, z_hat) is right-handed.
    # In the singular neighborhood near -Z, ref = +X is guaranteed not parallel to z_hat.
    ref = torch.tensor(
        [1.0, 0.0, 0.0], dtype=edge_vec.dtype, device=edge_vec.device
    ).expand_as(edge_vec)  # (E, 3)
    x_fb = torch.cross(ref, z_hat, dim=-1)  # (E, 3)
    x_fb = x_fb / safe_norm(x_fb, eps)
    y_fb = torch.cross(z_hat, x_fb, dim=-1)  # (E, 3)
    y_fb = y_fb / safe_norm(y_fb, eps)

    mask3 = singular.expand_as(edge_vec)  # (E, 3)
    x_hat = torch.where(mask3, x_fb, x_main)
    y_hat = torch.where(mask3, y_fb, y_main)

    # Normalize to protect against numerical drift (and to match your existing style).
    x_hat = x_hat / safe_norm(x_hat, eps)
    y_hat = y_hat / safe_norm(y_hat, eps)

    # === Step 4. Stack rows to form global->local rotation ===
    # Row-stacking ensures v_local = R @ v_global.
    rot_mat = torch.stack([x_hat, y_hat, z_hat], dim=-2)  # (E, 3, 3)
    return rot_mat
