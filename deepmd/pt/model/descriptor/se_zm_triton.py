# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Optional Triton kernels for SeZM-Net.

This module is safe to import even when Triton is not installed.

Notes
-----
SeZM-Net training needs higher-order gradients (double backward) because forces
are derived from autograd and force losses backpropagate through the force
computation. Triton kernels are not tracked by PyTorch autograd, so every Triton
accelerated path in this module is wrapped by a custom :class:`torch.autograd.Function`.

To keep double-backward correctness without falling back to PyTorch eager graphs,
each forward operator has a paired backward operator implemented as another
custom autograd Function. The backward Function runs Triton kernels in its
forward and implements the double-backward (gradgrad) in its own backward.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import torch

# === Step 0. Optional Triton import ===
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _TRITON_AVAILABLE: bool = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    """
    Return whether Triton is importable.

    Returns
    -------
    bool
        True if Triton can be imported, otherwise False.
    """
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[misc]
    def _outer_scatter_sum_kernel(
        r_ptr,  # noqa: ANN001
        g_ptr,  # noqa: ANN001
        dst_ptr,  # noqa: ANN001
        out_ptr,  # noqa: ANN001
        EMBED,  # noqa: ANN001
        BLOCK: tl.constexpr,  # type: ignore[valid-type]
    ) -> None:
        """
        Fused outer-product scatter-sum kernel.

        This kernel computes, for each edge `e` and each `j` block::

            out[dst[e], 0, j] += r[e, 0] * g[e, j]
            out[dst[e], 1, j] += r[e, 1] * g[e, j]
            out[dst[e], 2, j] += r[e, 2] * g[e, j]
            out[dst[e], 3, j] += r[e, 3] * g[e, j]

        All accumulations use atomic adds.
        """
        # === Step 1. Program ids ===
        pid_e = tl.program_id(0)
        pid_j = tl.program_id(1)

        # === Step 2. Column offsets ===
        offs_j = pid_j * BLOCK + tl.arange(0, BLOCK)
        mask_j = offs_j < EMBED

        # === Step 3. Load dst index and g block ===
        dst = tl.load(dst_ptr + pid_e).to(tl.int32)
        g = tl.load(g_ptr + pid_e * EMBED + offs_j, mask=mask_j, other=0.0).to(
            tl.float32
        )

        # === Step 4. Load r_tilde scalars (4,) ===
        r0 = tl.load(r_ptr + pid_e * 4 + 0).to(tl.float32)
        r1 = tl.load(r_ptr + pid_e * 4 + 1).to(tl.float32)
        r2 = tl.load(r_ptr + pid_e * 4 + 2).to(tl.float32)
        r3 = tl.load(r_ptr + pid_e * 4 + 3).to(tl.float32)

        # === Step 5. Atomic add to out (flattened as N x (4*EMBED)) ===
        base = out_ptr + dst * (4 * EMBED) + offs_j
        tl.atomic_add(base + 0 * EMBED, r0 * g, mask=mask_j)
        tl.atomic_add(base + 1 * EMBED, r1 * g, mask=mask_j)
        tl.atomic_add(base + 2 * EMBED, r2 * g, mask=mask_j)
        tl.atomic_add(base + 3 * EMBED, r3 * g, mask=mask_j)

    @triton.jit  # type: ignore[misc]
    def _outer_scatter_sum_reduce_r_kernel(
        grad_out_ptr,  # noqa: ANN001
        g_ptr,  # noqa: ANN001
        dst_ptr,  # noqa: ANN001
        grad_r_ptr,  # noqa: ANN001
        EMBED,  # noqa: ANN001
        BLOCK: tl.constexpr,  # type: ignore[valid-type]
    ) -> None:
        """
        Reduce over embedding dimension to compute grad_r (E, 4).

        Each program handles one edge and one C-block; partial sums are
        accumulated with atomic adds.
        """
        # === Step 1. Program ids ===
        pid_e = tl.program_id(0)
        pid_j = tl.program_id(1)

        # === Step 2. Column offsets ===
        offs_j = pid_j * BLOCK + tl.arange(0, BLOCK)
        mask_j = offs_j < EMBED

        # === Step 3. Load dst and g block ===
        dst = tl.load(dst_ptr + pid_e).to(tl.int32)
        g = tl.load(g_ptr + pid_e * EMBED + offs_j, mask=mask_j, other=0.0).to(
            tl.float32
        )

        # === Step 4. Load grad_out block ===
        base = grad_out_ptr + dst * (4 * EMBED) + offs_j
        go0 = tl.load(base + 0 * EMBED, mask=mask_j, other=0.0).to(tl.float32)
        go1 = tl.load(base + 1 * EMBED, mask=mask_j, other=0.0).to(tl.float32)
        go2 = tl.load(base + 2 * EMBED, mask=mask_j, other=0.0).to(tl.float32)
        go3 = tl.load(base + 3 * EMBED, mask=mask_j, other=0.0).to(tl.float32)

        # === Step 5. Partial reductions and atomic add ===
        s0 = tl.sum(go0 * g, axis=0)
        s1 = tl.sum(go1 * g, axis=0)
        s2 = tl.sum(go2 * g, axis=0)
        s3 = tl.sum(go3 * g, axis=0)

        grad_base = grad_r_ptr + pid_e * 4
        tl.atomic_add(grad_base + 0, s0)
        tl.atomic_add(grad_base + 1, s1)
        tl.atomic_add(grad_base + 2, s2)
        tl.atomic_add(grad_base + 3, s3)

    @triton.jit  # type: ignore[misc]
    def _outer_scatter_sum_reduce_g_kernel(
        grad_out_ptr,  # noqa: ANN001
        r_ptr,  # noqa: ANN001
        dst_ptr,  # noqa: ANN001
        grad_g_ptr,  # noqa: ANN001
        EMBED,  # noqa: ANN001
        BLOCK: tl.constexpr,  # type: ignore[valid-type]
    ) -> None:
        """Compute grad_g (E, C) from grad_out and r."""
        # === Step 1. Program ids ===
        pid_e = tl.program_id(0)
        pid_j = tl.program_id(1)

        # === Step 2. Column offsets ===
        offs_j = pid_j * BLOCK + tl.arange(0, BLOCK)
        mask_j = offs_j < EMBED

        # === Step 3. Load dst and r ===
        dst = tl.load(dst_ptr + pid_e).to(tl.int32)
        r0 = tl.load(r_ptr + pid_e * 4 + 0).to(tl.float32)
        r1 = tl.load(r_ptr + pid_e * 4 + 1).to(tl.float32)
        r2 = tl.load(r_ptr + pid_e * 4 + 2).to(tl.float32)
        r3 = tl.load(r_ptr + pid_e * 4 + 3).to(tl.float32)

        # === Step 4. Load grad_out block and compute grad_g ===
        base = grad_out_ptr + dst * (4 * EMBED) + offs_j
        go0 = tl.load(base + 0 * EMBED, mask=mask_j, other=0.0).to(tl.float32)
        go1 = tl.load(base + 1 * EMBED, mask=mask_j, other=0.0).to(tl.float32)
        go2 = tl.load(base + 2 * EMBED, mask=mask_j, other=0.0).to(tl.float32)
        go3 = tl.load(base + 3 * EMBED, mask=mask_j, other=0.0).to(tl.float32)

        grad = go0 * r0 + go1 * r1 + go2 * r2 + go3 * r3
        tl.store(
            grad_g_ptr + pid_e * EMBED + offs_j,
            grad,
            mask=mask_j,
        )

    @triton.jit  # type: ignore[misc]
    def _outer_scatter_sum_gradgrad_out_kernel(
        r_ptr,  # noqa: ANN001
        g_ptr,  # noqa: ANN001
        gradgrad_r_ptr,  # noqa: ANN001
        gradgrad_g_ptr,  # noqa: ANN001
        dst_ptr,  # noqa: ANN001
        out_ptr,  # noqa: ANN001
        EMBED,  # noqa: ANN001
        BLOCK: tl.constexpr,  # type: ignore[valid-type]
    ) -> None:
        """Scatter-add gradgrad contributions to grad_out (N, 4, C)."""
        # === Step 1. Program ids ===
        pid_e = tl.program_id(0)
        pid_j = tl.program_id(1)

        # === Step 2. Column offsets ===
        offs_j = pid_j * BLOCK + tl.arange(0, BLOCK)
        mask_j = offs_j < EMBED

        # === Step 3. Load dst, r, g, and gradgrad inputs ===
        dst = tl.load(dst_ptr + pid_e).to(tl.int32)
        g = tl.load(g_ptr + pid_e * EMBED + offs_j, mask=mask_j, other=0.0).to(
            tl.float32
        )
        gg = tl.load(
            gradgrad_g_ptr + pid_e * EMBED + offs_j,
            mask=mask_j,
            other=0.0,
        ).to(tl.float32)

        r0 = tl.load(r_ptr + pid_e * 4 + 0).to(tl.float32)
        r1 = tl.load(r_ptr + pid_e * 4 + 1).to(tl.float32)
        r2 = tl.load(r_ptr + pid_e * 4 + 2).to(tl.float32)
        r3 = tl.load(r_ptr + pid_e * 4 + 3).to(tl.float32)

        gg0 = tl.load(gradgrad_r_ptr + pid_e * 4 + 0).to(tl.float32)
        gg1 = tl.load(gradgrad_r_ptr + pid_e * 4 + 1).to(tl.float32)
        gg2 = tl.load(gradgrad_r_ptr + pid_e * 4 + 2).to(tl.float32)
        gg3 = tl.load(gradgrad_r_ptr + pid_e * 4 + 3).to(tl.float32)

        # === Step 4. Atomic add to grad_out ===
        base = out_ptr + dst * (4 * EMBED) + offs_j
        tl.atomic_add(base + 0 * EMBED, gg0 * g + r0 * gg, mask=mask_j)
        tl.atomic_add(base + 1 * EMBED, gg1 * g + r1 * gg, mask=mask_j)
        tl.atomic_add(base + 2 * EMBED, gg2 * g + r2 * gg, mask=mask_j)
        tl.atomic_add(base + 3 * EMBED, gg3 * g + r3 * gg, mask=mask_j)


class _OuterScatterSumBwdFn(torch.autograd.Function):
    """
    Triton backward for outer_scatter_sum.

    This Function computes first-order gradients in its forward, and implements
    the double-backward in its backward.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        grad_out: torch.Tensor,
        r_tilde: torch.Tensor,
        g: torch.Tensor,
        dst: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")
        if not grad_out.is_cuda:
            raise RuntimeError("OuterScatterSumBwd requires CUDA tensors")

        # === Step 1. Validate shapes ===
        if grad_out.dim() != 3 or grad_out.size(1) != 4:
            raise ValueError("grad_out must have shape (N, 4, C)")
        if r_tilde.dim() != 2 or r_tilde.size(1) != 4:
            raise ValueError("r_tilde must have shape (E, 4)")
        if g.dim() != 2:
            raise ValueError("g must have shape (E, C)")
        if dst.dim() != 1:
            raise ValueError("dst must have shape (E,)")
        if r_tilde.size(0) != g.size(0) or r_tilde.size(0) != dst.size(0):
            raise ValueError("E mismatch among r_tilde, g, dst")

        # === Step 2. Save tensors for double-backward ===
        ctx.save_for_backward(grad_out, r_tilde, g, dst)

        E = r_tilde.size(0)
        C = g.size(1)

        # === Step 3. Allocate outputs (fp32 accumulation) ===
        grad_r = torch.zeros(
            (E, 4),
            device=grad_out.device,
            dtype=torch.float32,
        )
        grad_g = torch.empty(
            (E, C),
            device=grad_out.device,
            dtype=torch.float32,
        )

        # === Step 4. Launch Triton kernels ===
        BLOCK = 128
        grid = (E, triton.cdiv(C, BLOCK))  # type: ignore[attr-defined]

        _outer_scatter_sum_reduce_r_kernel[grid](  # type: ignore[index]
            grad_out.contiguous(),
            g.contiguous(),
            dst.contiguous(),
            grad_r,
            EMBED=C,
            BLOCK=BLOCK,
            num_warps=4,
        )
        _outer_scatter_sum_reduce_g_kernel[grid](  # type: ignore[index]
            grad_out.contiguous(),
            r_tilde.contiguous(),
            dst.contiguous(),
            grad_g,
            EMBED=C,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # === Step 5. Cast to input dtypes ===
        return grad_r.to(dtype=r_tilde.dtype), grad_g.to(dtype=g.dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_grad_r: torch.Tensor | None,
        grad_grad_g: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
    ]:
        grad_out, r_tilde, g, dst = ctx.saved_tensors

        if grad_grad_r is None and grad_grad_g is None:
            return None, None, None, None

        # === Step 1. Prepare inputs ===
        grad_out = grad_out.contiguous()
        r_tilde = r_tilde.contiguous()
        g = g.contiguous()
        dst = dst.contiguous()

        E = r_tilde.size(0)
        C = g.size(1)
        N = grad_out.size(0)

        gg_r = (
            grad_grad_r
            if grad_grad_r is not None
            else torch.zeros((E, 4), device=grad_out.device, dtype=grad_out.dtype)
        )
        gg_g = (
            grad_grad_g
            if grad_grad_g is not None
            else torch.zeros((E, C), device=grad_out.device, dtype=grad_out.dtype)
        )

        gg_r = gg_r.contiguous()
        gg_g = gg_g.contiguous()

        # === Step 2. Compute grad w.r.t grad_out (scatter-add) ===
        grad_grad_out = torch.zeros(
            (N, 4, C),
            device=grad_out.device,
            dtype=torch.float32,
        )
        grad_grad_out_flat = grad_grad_out.view(N, 4 * C)

        BLOCK = 128
        grid = (E, triton.cdiv(C, BLOCK))  # type: ignore[attr-defined]
        _outer_scatter_sum_gradgrad_out_kernel[grid](  # type: ignore[index]
            r_tilde,
            g,
            gg_r,
            gg_g,
            dst,
            grad_grad_out_flat,
            EMBED=C,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # === Step 3. Compute grad w.r.t r_tilde and g ===
        grad_r_in = None
        grad_g_in = None

        if grad_grad_g is not None:
            grad_r_tmp = torch.zeros(
                (E, 4),
                device=grad_out.device,
                dtype=torch.float32,
            )
            _outer_scatter_sum_reduce_r_kernel[grid](  # type: ignore[index]
                grad_out,
                gg_g,
                dst,
                grad_r_tmp,
                EMBED=C,
                BLOCK=BLOCK,
                num_warps=4,
            )
            grad_r_in = grad_r_tmp.to(dtype=r_tilde.dtype)

        if grad_grad_r is not None:
            grad_g_tmp = torch.empty(
                (E, C),
                device=grad_out.device,
                dtype=torch.float32,
            )
            _outer_scatter_sum_reduce_g_kernel[grid](  # type: ignore[index]
                grad_out,
                gg_r,
                dst,
                grad_g_tmp,
                EMBED=C,
                BLOCK=BLOCK,
                num_warps=4,
            )
            grad_g_in = grad_g_tmp.to(dtype=g.dtype)

        return grad_grad_out, grad_r_in, grad_g_in, None


class _OuterScatterSumFn(torch.autograd.Function):
    """
    Autograd wrapper for Triton fused outer-product scatter-sum.

    Parameters
    ----------
    r_tilde
        Edge-wise 4D features with shape (E, 4).
    g
        Edge-wise features with shape (E, C).
    dst
        Destination indices with shape (E,), int64.
    n_nodes
        Number of destination nodes N.

    Returns
    -------
    torch.Tensor
        Aggregated tensor with shape (N, 4, C) in dtype float32.

    Notes
    -----
    - Forward uses Triton (when available).
    - Backward uses Triton kernels with a custom double-backward.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        r_tilde: torch.Tensor,
        g: torch.Tensor,
        dst: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        if not _TRITON_AVAILABLE:
            raise RuntimeError(
                "Triton is not available but _OuterScatterSumFn was called"
            )
        if not r_tilde.is_cuda or not g.is_cuda or not dst.is_cuda:
            raise ValueError("Triton path requires CUDA tensors")
        if r_tilde.dim() != 2 or r_tilde.size(1) != 4:
            raise ValueError("r_tilde must have shape (E, 4)")
        if g.dim() != 2:
            raise ValueError("g must have shape (E, C)")
        if dst.dim() != 1:
            raise ValueError("dst must have shape (E,)")
        if r_tilde.size(0) != g.size(0) or r_tilde.size(0) != dst.size(0):
            raise ValueError("E mismatch among r_tilde, g, dst")

        # === Step 1. Save tensors for backward ===
        ctx.save_for_backward(r_tilde, g, dst)
        ctx.n_nodes = int(n_nodes)

        E = r_tilde.size(0)
        C = g.size(1)

        # === Step 2. Allocate output in fp32 for stable accumulation ===
        out = torch.zeros(
            (ctx.n_nodes, 4, C),
            device=r_tilde.device,
            dtype=torch.float32,
        )
        out_flat = out.view(ctx.n_nodes, 4 * C)

        # === Step 3. Launch Triton kernel ===
        BLOCK = 128
        grid = (E, triton.cdiv(C, BLOCK))  # type: ignore[attr-defined]
        _outer_scatter_sum_kernel[grid](  # type: ignore[index]
            r_tilde.contiguous(),
            g.contiguous(),
            dst.contiguous(),
            out_flat,
            EMBED=C,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        r_tilde, g, dst = ctx.saved_tensors
        grad_r, grad_g = _OuterScatterSumBwdFn.apply(grad_out, r_tilde, g, dst)
        return grad_r, grad_g, None, None


def outer_scatter_sum(
    r_tilde: torch.Tensor,
    g: torch.Tensor,
    dst: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """
    Compute fused outer-product scatter-sum with optional Triton.

    Parameters
    ----------
    r_tilde
        Edge-wise tensor with shape (E, 4).
    g
        Edge-wise tensor with shape (E, C).
    dst
        Destination node indices with shape (E,).
    n_nodes
        Number of nodes N.

    Returns
    -------
    torch.Tensor
        Aggregated tensor with shape (N, 4, C).

    Raises
    ------
    ValueError
        If input shapes are invalid.
    RuntimeError
        If Triton is unavailable and this function is called.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not available. Install triton or disable use_triton."
        )
    return _OuterScatterSumFn.apply(r_tilde, g, dst, int(n_nodes))


# =============================================================================
# Z-Rotation Triton Kernel for WignerDCalculator
# =============================================================================
# This kernel accelerates the construction of dense block-diagonal Z_full(angle)
# matrices in the real (tesseral) spherical harmonics basis. It replaces the
# slow Python/advanced-indexing filling of Z with a single Triton store kernel.
#
# Key constraint: It does NOT alter the dense (E, D, D) Wigner-D assembly logic
# (the matmul chain stays the same). It only replaces the slow filling of Z.
# =============================================================================

if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[misc]
    def _z_store_kernel(
        values_ptr,  # noqa: ANN001
        offsets_ptr,  # noqa: ANN001
        out_ptr,  # noqa: ANN001
        K,  # noqa: ANN001
        D2,  # noqa: ANN001
        BLOCK_K: tl.constexpr,  # type: ignore[valid-type]
    ) -> None:
        """
        Store per-edge sparse entries into dense Z_flat.

        Parameters
        ----------
        values_ptr
            Pointer to values with shape (E, K).
        offsets_ptr
            Pointer to offsets with shape (K,), int32.
        out_ptr
            Pointer to output Z_flat with shape (E, D2).
        K
            Number of written entries per edge (dynamic).
        D2
            Flattened matrix size: D*D (dynamic).
        BLOCK_K
            Tile size over K (constexpr).
        """
        # === Step 1. Program ids ===
        pid_e = tl.program_id(0)
        pid_k = tl.program_id(1)

        # === Step 2. K tile ===
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # === Step 3. Load offsets and values ===
        off = tl.load(offsets_ptr + offs_k, mask=mask_k, other=0).to(tl.int32)
        val = tl.load(values_ptr + pid_e * K + offs_k, mask=mask_k, other=0.0)

        # === Step 4. Store to Z_flat ===
        # Each (edge, k) writes a unique location, so no atomics are needed.
        tl.store(out_ptr + pid_e * D2 + off, val, mask=mask_k)


class _ZRotationRealSHFn(torch.autograd.Function):
    """
    Build dense Z_full(angle) using Triton store.

    Parameters
    ----------
    angle
        Rotation angles with shape (E,).
    dim_full
        Full matrix dimension D.
    m0_idx
        Indices for m=0 diagonal entries, shape (n_m0,), long.
    pos_idx
        Indices for +m entries, shape (n_blk,), long.
    neg_idx
        Indices for -m entries, shape (n_blk,), long.
    m_values
        Positive m values, shape (n_blk,), dtype matches angle.

    Returns
    -------
    torch.Tensor
        Dense Z matrices with shape (E, D, D).

    Raises
    ------
    RuntimeError
        If Triton is unavailable or inputs are not CUDA tensors.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        angle: torch.Tensor,
        dim_full: int,
        m0_idx: torch.Tensor,
        pos_idx: torch.Tensor,
        neg_idx: torch.Tensor,
        m_values: torch.Tensor,
    ) -> torch.Tensor:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")
        if not angle.is_cuda:
            raise RuntimeError("ZRotationRealSH requires CUDA tensors")

        # === Step 1. Shapes ===
        E = int(angle.numel())
        D = int(dim_full)
        D2 = int(D * D)
        n_m0 = int(m0_idx.numel())
        n_blk = int(m_values.numel())
        K = n_m0 + 4 * n_blk

        # === Step 2. Precompute write offsets (flattened indices) ===
        m0_off = (m0_idx * D + m0_idx).to(torch.int32)
        if n_blk > 0:
            pospos = (pos_idx * D + pos_idx).to(torch.int32)
            negneg = (neg_idx * D + neg_idx).to(torch.int32)
            posneg = (pos_idx * D + neg_idx).to(torch.int32)
            negpos = (neg_idx * D + pos_idx).to(torch.int32)
            offsets = torch.cat([m0_off, pospos, negneg, posneg, negpos], dim=0)
        else:
            offsets = m0_off

        # === Step 3. Build per-edge values to write ===
        values = torch.zeros((E, K), device=angle.device, dtype=angle.dtype)
        values[:, :n_m0] = 1.0
        if n_blk > 0:
            angles_m = angle[:, None] * m_values[None, :]
            c = torch.cos(angles_m)
            s = torch.sin(angles_m)
            # Layout must match offsets concatenation order.
            base = n_m0
            values[:, base : base + n_blk] = c
            values[:, base + n_blk : base + 2 * n_blk] = c
            values[:, base + 2 * n_blk : base + 3 * n_blk] = s
            values[:, base + 3 * n_blk : base + 4 * n_blk] = -s

        # === Step 4. Allocate output and launch store kernel ===
        Z_flat = torch.zeros((E, D2), device=angle.device, dtype=angle.dtype)

        BLOCK_K = 128
        grid = (E, triton.cdiv(K, BLOCK_K))  # type: ignore[attr-defined]
        _z_store_kernel[grid](  # type: ignore[index]
            values,
            offsets,
            Z_flat,
            K=K,
            D2=D2,
            BLOCK_K=BLOCK_K,
            num_warps=2,
        )

        Z = Z_flat.view(E, D, D)

        # === Step 5. Save for backward ===
        ctx.dim_full = D
        ctx.save_for_backward(angle, m0_idx, pos_idx, neg_idx, m_values)
        return Z

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_Z: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, None, None, None, None]:
        angle, m0_idx, pos_idx, neg_idx, m_values = ctx.saved_tensors
        D = int(ctx.dim_full)

        # === Step 1. Grad wrt angle only ===
        # Z entries:
        #  m=0 diag is constant 1 -> contributes no grad to angle.
        #  For each (l,m>0):
        #    Z[pos,pos] =  c
        #    Z[neg,neg] =  c
        #    Z[pos,neg] =  s
        #    Z[neg,pos] = -s
        #  where c = cos(m*angle), s = sin(m*angle).

        E = int(angle.numel())
        n_blk = int(m_values.numel())
        if n_blk == 0:
            return grad_Z.new_zeros((E,)), None, None, None, None, None

        # === Step 2. Gather relevant grad entries ===
        grad_flat = grad_Z.reshape(E, D * D)

        pospos_off = pos_idx * D + pos_idx
        negneg_off = neg_idx * D + neg_idx
        posneg_off = pos_idx * D + neg_idx
        negpos_off = neg_idx * D + pos_idx

        g_pospos = grad_flat.index_select(1, pospos_off)
        g_negneg = grad_flat.index_select(1, negneg_off)
        g_posneg = grad_flat.index_select(1, posneg_off)
        g_negpos = grad_flat.index_select(1, negpos_off)

        # dL/dc = g_pospos + g_negneg
        # dL/ds = g_posneg - g_negpos  (because Z[neg,pos] = -s)
        grad_c = g_pospos + g_negneg
        grad_s = g_posneg - g_negpos

        # === Step 3. Recompute trig terms for double-backward support ===
        # This recomputation happens inside backward, so it *is* tracked when
        # create_graph=True.
        angles_m = angle[:, None] * m_values[None, :]
        c = torch.cos(angles_m)
        s = torch.sin(angles_m)

        # d/dangle cos(m a) = -m sin(m a)
        # d/dangle sin(m a) =  m cos(m a)
        m = m_values[None, :]
        grad_angle = (grad_c * (-m * s) + grad_s * (m * c)).sum(dim=1)

        return grad_angle, None, None, None, None, None


def build_z_rotation_triton(
    angle: torch.Tensor,
    *,
    dim_full: int,
    m0_indices: torch.Tensor,
    pos_indices: torch.Tensor,
    neg_indices: torch.Tensor,
    m_values: torch.Tensor,
) -> torch.Tensor:
    """
    Build dense block-diagonal Z rotation matrices using Triton.

    Parameters
    ----------
    angle
        Rotation angles with shape (E,).
    dim_full
        Full matrix dimension D=(lmax+1)^2.
    m0_indices
        Indices for m=0 diagonal entries.
    pos_indices
        Indices for +m entries.
    neg_indices
        Indices for -m entries.
    m_values
        Positive m values corresponding to (pos_indices, neg_indices).

    Returns
    -------
    torch.Tensor
        Z matrices with shape (E, D, D).

    Raises
    ------
    RuntimeError
        If Triton is not available.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")

    return _ZRotationRealSHFn.apply(
        angle,
        int(dim_full),
        m0_indices,
        pos_indices,
        neg_indices,
        m_values,
    )


# =============================================================================
# Triton-accelerated SeparableRMSNorm
# =============================================================================

if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[misc]
    def _separable_rmsnorm_fwd_kernel(
        x_ptr,  # noqa: ANN001
        out_ptr,  # noqa: ANN001
        balance_w_ptr,  # noqa: ANN001
        weight_ptr,  # noqa: ANN001
        bias_ptr,  # noqa: ANN001
        expand_idx_ptr,  # noqa: ANN001
        N,  # noqa: ANN001
        D,  # noqa: ANN001
        C,  # noqa: ANN001
        eps,  # noqa: ANN001
        HAS_AFFINE: tl.constexpr,  # type: ignore[valid-type]
        HAS_CENTERING: tl.constexpr,  # type: ignore[valid-type]
        BLOCK_C: tl.constexpr,  # type: ignore[valid-type]
    ) -> None:
        """
        Fused forward kernel for SeparableRMSNorm.

        Each program handles one sample (row n).
        """
        pid_n = tl.program_id(0)

        # === Step 1. l=0 normalization ===
        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        # Load x0 (l=0 features)
        x0_ptr = x_ptr + pid_n * D * C + offs_c
        x0 = tl.load(x0_ptr, mask=mask_c, other=0.0).to(tl.float32)

        # Centering (optional)
        if HAS_CENTERING:
            x0_sum = tl.sum(x0, axis=0)
            x0_mean = x0_sum / C
            x0 = x0 - x0_mean

        # RMS for l=0
        x0_sq_sum = tl.sum(x0 * x0, axis=0)
        rms0 = tl.sqrt(x0_sq_sum / C + eps)
        x0_norm = x0 / rms0

        # Affine for l=0
        if HAS_AFFINE:
            w0 = tl.load(weight_ptr + offs_c, mask=mask_c, other=1.0)
            x0_norm = x0_norm * w0
            if HAS_CENTERING:
                b0 = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
                x0_norm = x0_norm + b0

        # Store l=0 output
        out0_ptr = out_ptr + pid_n * D * C + offs_c
        tl.store(out0_ptr, x0_norm.to(tl.float32), mask=mask_c)

        # === Step 2. l>0 normalization (if D > 1) ===
        if D > 1:
            # Compute weighted variance for l>0
            D_non_scalar = D - 1
            var_sum = tl.zeros([1], dtype=tl.float32)

            for d in range(D_non_scalar):
                xt_base = x_ptr + pid_n * D * C + (d + 1) * C + offs_c
                xt_d = tl.load(xt_base, mask=mask_c, other=0.0).to(tl.float32)
                xt_sq = xt_d * xt_d
                xt_sq_sum = tl.sum(xt_sq, axis=0)
                bw = tl.load(balance_w_ptr + d)
                var_sum += xt_sq_sum * bw

            rmst = tl.sqrt(var_sum + eps)

            # Normalize and apply affine for l>0
            for d in range(D_non_scalar):
                xt_base = x_ptr + pid_n * D * C + (d + 1) * C + offs_c
                xt_d = tl.load(xt_base, mask=mask_c, other=0.0).to(tl.float32)
                xt_norm = xt_d / rmst

                if HAS_AFFINE:
                    l_idx = tl.load(expand_idx_ptr + d)
                    wt = tl.load(
                        weight_ptr + l_idx * C + offs_c, mask=mask_c, other=1.0
                    )
                    xt_norm = xt_norm * wt

                out_base = out_ptr + pid_n * D * C + (d + 1) * C + offs_c
                tl.store(out_base, xt_norm.to(tl.float32), mask=mask_c)


class _SeparableRMSNormFn(torch.autograd.Function):
    """
    Triton-accelerated SeparableRMSNorm forward/backward.

    Backward uses PyTorch ops for double-backward support.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        balance_weight: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        expand_index: torch.Tensor,
        eps: float,
        centering: bool,
    ) -> torch.Tensor:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")
        if not x.is_cuda:
            raise RuntimeError("SeparableRMSNorm Triton requires CUDA tensors")

        N, D, C = x.shape
        out = torch.empty_like(x)

        HAS_AFFINE = weight is not None
        HAS_CENTERING = centering

        # Determine BLOCK_C (power of 2, at least C)
        BLOCK_C = triton.next_power_of_2(C)  # type: ignore[attr-defined]

        grid = (N,)
        _separable_rmsnorm_fwd_kernel[grid](  # type: ignore[index]
            x,
            out,
            balance_weight if balance_weight.numel() > 0 else x,  # Dummy if empty
            weight if weight is not None else x,  # Dummy if None
            bias if bias is not None else x,  # Dummy if None
            expand_index if expand_index.numel() > 0 else x,  # Dummy if empty
            N,
            D,
            C,
            eps,
            HAS_AFFINE=HAS_AFFINE,
            HAS_CENTERING=HAS_CENTERING,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        # Save for backward
        ctx.save_for_backward(x, balance_weight, weight, expand_index)
        ctx.eps = eps
        ctx.centering = centering
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_out: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
    ]:
        """Backward pass using PyTorch ops for double-backward support."""
        x, balance_weight, weight, expand_index = ctx.saved_tensors
        eps = ctx.eps
        centering = ctx.centering

        N, D, C = x.shape
        x0 = x[:, :1, :]
        xt = x[:, 1:, :]
        grad_out0 = grad_out[:, :1, :]
        grad_outt = grad_out[:, 1:, :]

        grad_x = torch.zeros_like(x)
        grad_weight = None
        grad_bias = None

        # === Step 1. l=0 backward ===
        x0_c = x0.float()
        if centering:
            x0_c = x0_c - x0_c.mean(dim=-1, keepdim=True)

        rms0_sq = x0_c.pow(2).mean(dim=-1, keepdim=True) + eps
        rms0 = torch.sqrt(rms0_sq)
        x0_norm = x0_c / rms0

        if weight is not None:
            w0 = weight[0].view(1, 1, -1)
            grad_x0_norm = grad_out0 * w0
            grad_weight = torch.zeros_like(weight)
            grad_weight[0] = (grad_out0 * x0_norm).sum(dim=(0, 1))
            if centering:
                grad_bias = grad_out0.sum(dim=(0, 1))
        else:
            grad_x0_norm = grad_out0

        # d(x_norm)/d(x) for RMSNorm
        grad_x0_c = (
            grad_x0_norm - x0_norm * (grad_x0_norm * x0_norm).mean(dim=-1, keepdim=True)
        ) / rms0
        if centering:
            grad_x0_c = grad_x0_c - grad_x0_c.mean(dim=-1, keepdim=True)
        grad_x[:, :1, :] = grad_x0_c.to(x.dtype)

        # === Step 2. l>0 backward ===
        if D > 1 and xt.numel() > 0:
            xt_c = xt.float()
            mean_var = torch.einsum("ndc,d->n", xt_c * xt_c, balance_weight.float())
            rmst = torch.sqrt(mean_var + eps).view(-1, 1, 1)
            xt_norm = xt_c / rmst

            if weight is not None:
                wt = torch.index_select(weight, dim=0, index=expand_index)  # (D-1, C)
                grad_xt_norm = grad_outt * wt.unsqueeze(0)
                # Accumulate grad_weight for l>0
                for l in range(1, int(expand_index.max().item()) + 2):
                    mask = expand_index == l
                    if mask.any():
                        grad_weight[l] = (
                            grad_outt[:, mask, :] * xt_norm[:, mask, :]
                        ).sum(dim=(0, 1))
            else:
                grad_xt_norm = grad_outt

            # d(xt_norm)/d(xt) for weighted RMSNorm
            # xt_norm = xt / rmst, rmst = sqrt(sum(xt^2 * bw) + eps)
            # d(xt_norm)/d(xt) = 1/rmst - xt * d(rmst)/d(xt) / rmst^2
            # d(rmst)/d(xt) = xt * bw / rmst
            bw = balance_weight.float().view(1, -1, 1)
            grad_rmst = -(grad_xt_norm * xt_norm / rmst).sum(dim=(1, 2), keepdim=True)
            grad_xt_c = grad_xt_norm / rmst + grad_rmst * xt_c * bw / rmst
            grad_x[:, 1:, :] = grad_xt_c.to(x.dtype)

        return grad_x, None, grad_weight, grad_bias, None, None, None


def separable_rmsnorm_triton(
    x: torch.Tensor,
    balance_weight: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    expand_index: torch.Tensor,
    eps: float,
    centering: bool,
) -> torch.Tensor:
    """
    Triton-accelerated SeparableRMSNorm.

    Parameters
    ----------
    x
        Input features with shape (N, D, C).
    balance_weight
        Degree balancing weights with shape (D-1,).
    weight
        Per-l affine weights with shape (lmax+1, C), or None.
    bias
        Bias for l=0 with shape (C,), or None.
    expand_index
        Index mapping from d to l for l>0, shape (D-1,).
    eps
        Epsilon for numerical stability.
    centering
        Whether to apply mean centering for l=0.

    Returns
    -------
    torch.Tensor
        Normalized features with shape (N, D, C).
    """
    if not _TRITON_AVAILABLE or not x.is_cuda:
        raise RuntimeError("Triton SeparableRMSNorm requires CUDA and Triton")

    return _SeparableRMSNormFn.apply(
        x,
        balance_weight,
        weight,
        bias,
        expand_index,
        eps,
        centering,
    )


# =============================================================================
# Triton-accelerated SO2Convolution Scatter Operations
# =============================================================================
# These kernels fuse element-wise multiplication with scatter-add aggregation,
# eliminating intermediate tensor materialization and reducing memory bandwidth.
# =============================================================================

if _TRITON_AVAILABLE:

    @triton.jit  # type: ignore[misc]
    def _so2_baseline_scatter_kernel(
        x_msg_ptr,  # noqa: ANN001
        edge_env_ptr,  # noqa: ANN001
        dst_ptr,  # noqa: ANN001
        out_ptr,  # noqa: ANN001
        D,  # noqa: ANN001
        C,  # noqa: ANN001
        BLOCK_C: tl.constexpr,  # type: ignore[valid-type]
    ) -> None:
        """
        Fused envelope-weighted scatter-add for SO2Conv baseline path.

        Computes: out[dst[e], d, :] += edge_env[e] * x_message[e, d, :]

        Grid: (E, D, cdiv(C, BLOCK_C))
        """
        # === Step 1. Program ids ===
        pid_e = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_c = tl.program_id(2)

        # === Step 2. Column offsets ===
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        # === Step 3. Load dst index and edge_env ===
        dst = tl.load(dst_ptr + pid_e).to(tl.int32)
        env = tl.load(edge_env_ptr + pid_e).to(tl.float32)

        # === Step 4. Load x_message block ===
        x_base = pid_e * D * C + pid_d * C + offs_c
        x = tl.load(x_msg_ptr + x_base, mask=mask_c, other=0.0).to(tl.float32)

        # === Step 5. Atomic add to output ===
        out_base = dst * D * C + pid_d * C + offs_c
        tl.atomic_add(out_ptr + out_base, env * x, mask=mask_c)

    @triton.jit  # type: ignore[misc]
    def _so2_head_scatter_kernel(
        V_ptr,  # noqa: ANN001
        edge_weight_ptr,  # noqa: ANN001
        dst_ptr,  # noqa: ANN001
        out_ptr,  # noqa: ANN001
        D,  # noqa: ANN001
        H,  # noqa: ANN001
        Hd,  # noqa: ANN001
        BLOCK_Hd: tl.constexpr,  # type: ignore[valid-type]
    ) -> None:
        """
        Fused head-weighted scatter-add for SO2Conv attention path.

        Computes: out[dst[e], d, h, :] += edge_weight[e, h] * V[e, d, h, :]

        Grid: (E, D*H, cdiv(Hd, BLOCK_Hd))
        """
        # === Step 1. Program ids ===
        pid_e = tl.program_id(0)
        pid_dh = tl.program_id(1)
        pid_hd = tl.program_id(2)

        # === Step 2. Decode (d, h) from pid_dh ===
        pid_d = pid_dh // H
        pid_h = pid_dh % H

        # === Step 3. Hd offsets ===
        offs_hd = pid_hd * BLOCK_Hd + tl.arange(0, BLOCK_Hd)
        mask_hd = offs_hd < Hd

        # === Step 4. Load dst index and edge_weight ===
        dst = tl.load(dst_ptr + pid_e).to(tl.int32)
        w = tl.load(edge_weight_ptr + pid_e * H + pid_h).to(tl.float32)

        # === Step 5. Load V block ===
        # V layout: (E, D, H, Hd) -> flat index: e*D*H*Hd + d*H*Hd + h*Hd + hd
        V_base = pid_e * D * H * Hd + pid_d * H * Hd + pid_h * Hd + offs_hd
        v = tl.load(V_ptr + V_base, mask=mask_hd, other=0.0).to(tl.float32)

        # === Step 6. Atomic add to output ===
        out_base = dst * D * H * Hd + pid_d * H * Hd + pid_h * Hd + offs_hd
        tl.atomic_add(out_ptr + out_base, w * v, mask=mask_hd)

    @triton.jit  # type: ignore[misc]
    def _so2_baseline_gather_mul_kernel(
        grad_out_ptr,  # noqa: ANN001
        scalar_ptr,  # noqa: ANN001
        dst_ptr,  # noqa: ANN001
        out_ptr,  # noqa: ANN001
        D,  # noqa: ANN001
        C,  # noqa: ANN001
        BLOCK_C: tl.constexpr,  # type: ignore[valid-type]
    ) -> None:
        """
        Gather grad_out by dst and multiply by a per-edge scalar.

        Computes: out[e, d, c] = grad_out[dst[e], d, c] * scalar[e]
        """
        # === Step 1. Program ids ===
        pid_e = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_c = tl.program_id(2)

        # === Step 2. Column offsets ===
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        # === Step 3. Load dst and scalar ===
        dst = tl.load(dst_ptr + pid_e).to(tl.int32)
        s = tl.load(scalar_ptr + pid_e).to(tl.float32)

        # === Step 4. Load grad_out block and store ===
        go_base = dst * D * C + pid_d * C + offs_c
        go = tl.load(grad_out_ptr + go_base, mask=mask_c, other=0.0).to(tl.float32)

        out_base = pid_e * D * C + pid_d * C + offs_c
        tl.store(out_ptr + out_base, go * s, mask=mask_c)

    @triton.jit  # type: ignore[misc]
    def _so2_baseline_reduce_env_kernel(
        grad_out_ptr,  # noqa: ANN001
        x_ptr,  # noqa: ANN001
        dst_ptr,  # noqa: ANN001
        grad_env_ptr,  # noqa: ANN001
        D,  # noqa: ANN001
        C,  # noqa: ANN001
        BLOCK_C: tl.constexpr,  # type: ignore[valid-type]
    ) -> None:
        """Reduce over (d, c) to compute per-edge scalar gradients."""
        # === Step 1. Program ids ===
        pid_e = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_c = tl.program_id(2)

        # === Step 2. Column offsets ===
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        # === Step 3. Load dst ===
        dst = tl.load(dst_ptr + pid_e).to(tl.int32)

        # === Step 4. Load grad_out and x blocks ===
        go_base = dst * D * C + pid_d * C + offs_c
        go = tl.load(grad_out_ptr + go_base, mask=mask_c, other=0.0).to(tl.float32)

        x_base = pid_e * D * C + pid_d * C + offs_c
        x = tl.load(x_ptr + x_base, mask=mask_c, other=0.0).to(tl.float32)

        # === Step 5. Partial reduction and atomic add ===
        partial = tl.sum(go * x, axis=0)
        tl.atomic_add(grad_env_ptr + pid_e, partial)

    @triton.jit  # type: ignore[misc]
    def _so2_baseline_gradgrad_out_kernel(
        ggx_ptr,  # noqa: ANN001
        gg_env_ptr,  # noqa: ANN001
        x_ptr,  # noqa: ANN001
        edge_env_ptr,  # noqa: ANN001
        dst_ptr,  # noqa: ANN001
        out_ptr,  # noqa: ANN001
        D,  # noqa: ANN001
        C,  # noqa: ANN001
        BLOCK_C: tl.constexpr,  # type: ignore[valid-type]
    ) -> None:
        """Scatter-add gradgrad contributions to grad_out."""
        # === Step 1. Program ids ===
        pid_e = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_c = tl.program_id(2)

        # === Step 2. Column offsets ===
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        # === Step 3. Load dst and scalars ===
        dst = tl.load(dst_ptr + pid_e).to(tl.int32)
        env = tl.load(edge_env_ptr + pid_e).to(tl.float32)
        gg_env = tl.load(gg_env_ptr + pid_e).to(tl.float32)

        # === Step 4. Load GGx and x blocks ===
        ggx_base = pid_e * D * C + pid_d * C + offs_c
        ggx = tl.load(ggx_ptr + ggx_base, mask=mask_c, other=0.0).to(tl.float32)

        x_base = pid_e * D * C + pid_d * C + offs_c
        x = tl.load(x_ptr + x_base, mask=mask_c, other=0.0).to(tl.float32)

        # === Step 5. Scatter-add ===
        out_base = dst * D * C + pid_d * C + offs_c
        val = ggx * env + x * gg_env
        tl.atomic_add(out_ptr + out_base, val, mask=mask_c)

    @triton.jit  # type: ignore[misc]
    def _so2_head_gather_v_kernel(
        grad_out_ptr,  # noqa: ANN001
        edge_weight_ptr,  # noqa: ANN001
        dst_ptr,  # noqa: ANN001
        out_ptr,  # noqa: ANN001
        D,  # noqa: ANN001
        H,  # noqa: ANN001
        Hd,  # noqa: ANN001
        BLOCK_Hd: tl.constexpr,  # type: ignore[valid-type]
    ) -> None:
        """Gather grad_out by dst and multiply by per-head edge weights."""
        # === Step 1. Program ids ===
        pid_e = tl.program_id(0)
        pid_dh = tl.program_id(1)
        pid_hd = tl.program_id(2)

        # === Step 2. Decode (d, h) ===
        pid_d = pid_dh // H
        pid_h = pid_dh % H

        # === Step 3. Hd offsets ===
        offs_hd = pid_hd * BLOCK_Hd + tl.arange(0, BLOCK_Hd)
        mask_hd = offs_hd < Hd

        # === Step 4. Load dst and weight ===
        dst = tl.load(dst_ptr + pid_e).to(tl.int32)
        w = tl.load(edge_weight_ptr + pid_e * H + pid_h).to(tl.float32)

        # === Step 5. Load grad_out and store ===
        go_base = dst * D * H * Hd + pid_d * H * Hd + pid_h * Hd + offs_hd
        go = tl.load(grad_out_ptr + go_base, mask=mask_hd, other=0.0).to(tl.float32)

        out_base = pid_e * D * H * Hd + pid_d * H * Hd + pid_h * Hd + offs_hd
        tl.store(out_ptr + out_base, go * w, mask=mask_hd)

    @triton.jit  # type: ignore[misc]
    def _so2_head_reduce_weight_kernel(
        grad_out_ptr,  # noqa: ANN001
        V_ptr,  # noqa: ANN001
        dst_ptr,  # noqa: ANN001
        grad_w_ptr,  # noqa: ANN001
        D,  # noqa: ANN001
        H,  # noqa: ANN001
        Hd,  # noqa: ANN001
        BLOCK_Hd: tl.constexpr,  # type: ignore[valid-type]
    ) -> None:
        """Reduce over (d, hd) to compute per-edge head gradients."""
        # === Step 1. Program ids ===
        pid_e = tl.program_id(0)
        pid_dh = tl.program_id(1)
        pid_hd = tl.program_id(2)

        # === Step 2. Decode (d, h) ===
        pid_d = pid_dh // H
        pid_h = pid_dh % H

        # === Step 3. Hd offsets ===
        offs_hd = pid_hd * BLOCK_Hd + tl.arange(0, BLOCK_Hd)
        mask_hd = offs_hd < Hd

        # === Step 4. Load dst ===
        dst = tl.load(dst_ptr + pid_e).to(tl.int32)

        # === Step 5. Partial reduction for this (d, h, hd-block) ===
        go_base = dst * D * H * Hd + pid_d * H * Hd + pid_h * Hd + offs_hd
        go = tl.load(grad_out_ptr + go_base, mask=mask_hd, other=0.0).to(tl.float32)
        v_base = pid_e * D * H * Hd + pid_d * H * Hd + pid_h * Hd + offs_hd
        v = tl.load(V_ptr + v_base, mask=mask_hd, other=0.0).to(tl.float32)
        partial = tl.sum(go * v, axis=0)

        # === Step 6. Atomic add ===
        tl.atomic_add(grad_w_ptr + pid_e * H + pid_h, partial)

    @triton.jit  # type: ignore[misc]
    def _so2_head_gradgrad_out_kernel(
        gg_V_ptr,  # noqa: ANN001
        gg_w_ptr,  # noqa: ANN001
        V_ptr,  # noqa: ANN001
        edge_weight_ptr,  # noqa: ANN001
        dst_ptr,  # noqa: ANN001
        out_ptr,  # noqa: ANN001
        D,  # noqa: ANN001
        H,  # noqa: ANN001
        Hd,  # noqa: ANN001
        BLOCK_Hd: tl.constexpr,  # type: ignore[valid-type]
    ) -> None:
        """Scatter-add gradgrad contributions to grad_out."""
        # === Step 1. Program ids ===
        pid_e = tl.program_id(0)
        pid_dh = tl.program_id(1)
        pid_hd = tl.program_id(2)

        # === Step 2. Decode (d, h) ===
        pid_d = pid_dh // H
        pid_h = pid_dh % H

        # === Step 3. Hd offsets ===
        offs_hd = pid_hd * BLOCK_Hd + tl.arange(0, BLOCK_Hd)
        mask_hd = offs_hd < Hd

        # === Step 4. Load dst and weights ===
        dst = tl.load(dst_ptr + pid_e).to(tl.int32)
        w = tl.load(edge_weight_ptr + pid_e * H + pid_h).to(tl.float32)
        gg_w = tl.load(gg_w_ptr + pid_e * H + pid_h).to(tl.float32)

        # === Step 5. Load GG_V and V blocks ===
        gg_base = pid_e * D * H * Hd + pid_d * H * Hd + pid_h * Hd + offs_hd
        gg_v = tl.load(gg_V_ptr + gg_base, mask=mask_hd, other=0.0).to(tl.float32)

        v_base = pid_e * D * H * Hd + pid_d * H * Hd + pid_h * Hd + offs_hd
        v = tl.load(V_ptr + v_base, mask=mask_hd, other=0.0).to(tl.float32)

        # === Step 6. Scatter-add ===
        out_base = dst * D * H * Hd + pid_d * H * Hd + pid_h * Hd + offs_hd
        val = gg_v * w + v * gg_w
        tl.atomic_add(out_ptr + out_base, val, mask=mask_hd)


class _SO2BaselineScatterBwdFn(torch.autograd.Function):
    """Triton backward for SO2 baseline scatter."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        grad_out: torch.Tensor,
        x_message: torch.Tensor,
        edge_env: torch.Tensor,
        dst: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")
        if not grad_out.is_cuda:
            raise RuntimeError("SO2BaselineScatterBwd requires CUDA tensors")

        # === Step 1. Validate shapes ===
        if grad_out.dim() != 3:
            raise ValueError("grad_out must have shape (N, D, C)")
        if x_message.dim() != 3:
            raise ValueError("x_message must have shape (E, D, C)")
        if edge_env.dim() != 2 or edge_env.size(1) != 1:
            raise ValueError("edge_env must have shape (E, 1)")
        if dst.dim() != 1:
            raise ValueError("dst must have shape (E,)")
        if x_message.size(0) != edge_env.size(0) or x_message.size(0) != dst.size(0):
            raise ValueError("E mismatch among x_message, edge_env, dst")

        # === Step 2. Save tensors for double-backward ===
        ctx.save_for_backward(grad_out, x_message, edge_env, dst)

        E, D, C = x_message.shape

        # === Step 3. Allocate outputs ===
        grad_x_message = torch.empty(
            (E, D, C),
            device=grad_out.device,
            dtype=torch.float32,
        )
        grad_edge_env = torch.zeros(
            (E,),
            device=grad_out.device,
            dtype=torch.float32,
        )

        # === Step 4. Launch Triton kernels ===
        BLOCK_C = 128
        grid = (E, D, triton.cdiv(C, BLOCK_C))  # type: ignore[attr-defined]

        _so2_baseline_gather_mul_kernel[grid](  # type: ignore[index]
            grad_out.contiguous(),
            edge_env.contiguous().view(-1),
            dst.contiguous(),
            grad_x_message,
            D=D,
            C=C,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        _so2_baseline_reduce_env_kernel[grid](  # type: ignore[index]
            grad_out.contiguous(),
            x_message.contiguous(),
            dst.contiguous(),
            grad_edge_env,
            D=D,
            C=C,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        return (
            grad_x_message.to(dtype=x_message.dtype),
            grad_edge_env.view(-1, 1).to(dtype=edge_env.dtype),
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_grad_x: torch.Tensor | None,
        grad_grad_env: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
    ]:
        grad_out, x_message, edge_env, dst = ctx.saved_tensors

        if grad_grad_x is None and grad_grad_env is None:
            return None, None, None, None

        # === Step 1. Prepare inputs ===
        grad_out = grad_out.contiguous()
        x_message = x_message.contiguous()
        edge_env = edge_env.contiguous()
        dst = dst.contiguous()

        E, D, C = x_message.shape
        N = grad_out.size(0)

        ggx = (
            grad_grad_x
            if grad_grad_x is not None
            else torch.zeros((E, D, C), device=grad_out.device, dtype=grad_out.dtype)
        )
        gg_env = (
            grad_grad_env
            if grad_grad_env is not None
            else torch.zeros((E, 1), device=grad_out.device, dtype=grad_out.dtype)
        )
        ggx = ggx.contiguous()
        gg_env = gg_env.contiguous().view(-1)

        # === Step 2. grad w.r.t grad_out (scatter-add) ===
        grad_grad_out = torch.zeros(
            (N, D, C),
            device=grad_out.device,
            dtype=torch.float32,
        )

        BLOCK_C = 128
        grid = (E, D, triton.cdiv(C, BLOCK_C))  # type: ignore[attr-defined]

        _so2_baseline_gradgrad_out_kernel[grid](  # type: ignore[index]
            ggx,
            gg_env,
            x_message,
            edge_env.view(-1),
            dst,
            grad_grad_out,
            D=D,
            C=C,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        # === Step 3. grad w.r.t x_message and edge_env ===
        grad_x_in = None
        grad_env_in = None

        if grad_grad_env is not None:
            grad_x_tmp = torch.empty(
                (E, D, C),
                device=grad_out.device,
                dtype=torch.float32,
            )
            _so2_baseline_gather_mul_kernel[grid](  # type: ignore[index]
                grad_out,
                gg_env,
                dst,
                grad_x_tmp,
                D=D,
                C=C,
                BLOCK_C=BLOCK_C,
                num_warps=4,
            )
            grad_x_in = grad_x_tmp.to(dtype=x_message.dtype)

        if grad_grad_x is not None:
            grad_env_tmp = torch.zeros(
                (E,),
                device=grad_out.device,
                dtype=torch.float32,
            )
            _so2_baseline_reduce_env_kernel[grid](  # type: ignore[index]
                grad_out,
                ggx,
                dst,
                grad_env_tmp,
                D=D,
                C=C,
                BLOCK_C=BLOCK_C,
                num_warps=4,
            )
            grad_env_in = grad_env_tmp.view(-1, 1).to(dtype=edge_env.dtype)

        return grad_grad_out, grad_x_in, grad_env_in, None


class _SO2BaselineScatterFn(torch.autograd.Function):
    """
    Autograd wrapper for Triton fused baseline scatter.

    Forward: out[dst[e], d, c] += edge_env[e] * x_message[e, d, c]
    Backward: Triton kernels with custom double-backward.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x_message: torch.Tensor,
        edge_env: torch.Tensor,
        dst: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")
        if not x_message.is_cuda:
            raise RuntimeError("SO2BaselineScatter requires CUDA tensors")

        E, D, C = x_message.shape
        ctx.save_for_backward(x_message, edge_env, dst)
        ctx.n_nodes = n_nodes

        # === Step 1. Allocate output in fp32 for stable accumulation ===
        out = torch.zeros(
            (n_nodes, D, C),
            device=x_message.device,
            dtype=torch.float32,
        )

        # === Step 2. Launch Triton kernel ===
        BLOCK_C = 128
        grid = (E, D, triton.cdiv(C, BLOCK_C))  # type: ignore[attr-defined]
        _so2_baseline_scatter_kernel[grid](  # type: ignore[index]
            x_message.contiguous(),
            edge_env.contiguous().view(-1),
            dst.contiguous(),
            out,
            D=D,
            C=C,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        x_message, edge_env, dst = ctx.saved_tensors
        grad_x_message, grad_edge_env = _SO2BaselineScatterBwdFn.apply(
            grad_out, x_message, edge_env, dst
        )
        return grad_x_message, grad_edge_env, None, None


class _SO2HeadScatterBwdFn(torch.autograd.Function):
    """Triton backward for SO2 head scatter."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        grad_out: torch.Tensor,
        V: torch.Tensor,
        edge_weight: torch.Tensor,
        dst: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")
        if not grad_out.is_cuda:
            raise RuntimeError("SO2HeadScatterBwd requires CUDA tensors")

        # === Step 1. Validate shapes ===
        if grad_out.dim() != 4:
            raise ValueError("grad_out must have shape (N, D, H, Hd)")
        if V.dim() != 4:
            raise ValueError("V must have shape (E, D, H, Hd)")
        if edge_weight.dim() != 2:
            raise ValueError("edge_weight must have shape (E, H)")
        if dst.dim() != 1:
            raise ValueError("dst must have shape (E,)")
        if V.size(0) != edge_weight.size(0) or V.size(0) != dst.size(0):
            raise ValueError("E mismatch among V, edge_weight, dst")

        # === Step 2. Save tensors for double-backward ===
        ctx.save_for_backward(grad_out, V, edge_weight, dst)

        E, D, H, Hd = V.shape

        # === Step 3. Allocate outputs ===
        grad_V = torch.empty(
            (E, D, H, Hd),
            device=grad_out.device,
            dtype=torch.float32,
        )
        grad_edge_weight = torch.zeros(
            (E, H),
            device=grad_out.device,
            dtype=torch.float32,
        )

        # === Step 4. Launch Triton kernels ===
        BLOCK_Hd = 64
        grid_v = (E, D * H, triton.cdiv(Hd, BLOCK_Hd))  # type: ignore[attr-defined]
        _so2_head_gather_v_kernel[grid_v](  # type: ignore[index]
            grad_out.contiguous(),
            edge_weight.contiguous(),
            dst.contiguous(),
            grad_V,
            D=D,
            H=H,
            Hd=Hd,
            BLOCK_Hd=BLOCK_Hd,
            num_warps=4,
        )

        grid_w = (E, D * H, triton.cdiv(Hd, BLOCK_Hd))  # type: ignore[attr-defined]
        _so2_head_reduce_weight_kernel[grid_w](  # type: ignore[index]
            grad_out.contiguous(),
            V.contiguous(),
            dst.contiguous(),
            grad_edge_weight,
            D=D,
            H=H,
            Hd=Hd,
            BLOCK_Hd=BLOCK_Hd,
            num_warps=4,
        )

        return grad_V.to(dtype=V.dtype), grad_edge_weight.to(dtype=edge_weight.dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_grad_V: torch.Tensor | None,
        grad_grad_w: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
    ]:
        grad_out, V, edge_weight, dst = ctx.saved_tensors

        if grad_grad_V is None and grad_grad_w is None:
            return None, None, None, None

        # === Step 1. Prepare inputs ===
        grad_out = grad_out.contiguous()
        V = V.contiguous()
        edge_weight = edge_weight.contiguous()
        dst = dst.contiguous()

        E, D, H, Hd = V.shape
        N = grad_out.size(0)

        gg_V = (
            grad_grad_V
            if grad_grad_V is not None
            else torch.zeros(
                (E, D, H, Hd), device=grad_out.device, dtype=grad_out.dtype
            )
        )
        gg_w = (
            grad_grad_w
            if grad_grad_w is not None
            else torch.zeros((E, H), device=grad_out.device, dtype=grad_out.dtype)
        )
        gg_V = gg_V.contiguous()
        gg_w = gg_w.contiguous()

        # === Step 2. grad w.r.t grad_out (scatter-add) ===
        grad_grad_out = torch.zeros(
            (N, D, H, Hd),
            device=grad_out.device,
            dtype=torch.float32,
        )

        BLOCK_Hd = 64
        grid = (E, D * H, triton.cdiv(Hd, BLOCK_Hd))  # type: ignore[attr-defined]
        _so2_head_gradgrad_out_kernel[grid](  # type: ignore[index]
            gg_V,
            gg_w,
            V,
            edge_weight,
            dst,
            grad_grad_out,
            D=D,
            H=H,
            Hd=Hd,
            BLOCK_Hd=BLOCK_Hd,
            num_warps=4,
        )

        # === Step 3. grad w.r.t V and edge_weight ===
        grad_V_in = None
        grad_w_in = None

        if grad_grad_w is not None:
            grad_V_tmp = torch.empty(
                (E, D, H, Hd),
                device=grad_out.device,
                dtype=torch.float32,
            )
            _so2_head_gather_v_kernel[grid](  # type: ignore[index]
                grad_out,
                gg_w,
                dst,
                grad_V_tmp,
                D=D,
                H=H,
                Hd=Hd,
                BLOCK_Hd=BLOCK_Hd,
                num_warps=4,
            )
            grad_V_in = grad_V_tmp.to(dtype=V.dtype)

        if grad_grad_V is not None:
            grad_w_tmp = torch.zeros(
                (E, H),
                device=grad_out.device,
                dtype=torch.float32,
            )
            grid_w = (E, D * H, triton.cdiv(Hd, BLOCK_Hd))  # type: ignore[attr-defined]
            _so2_head_reduce_weight_kernel[grid_w](  # type: ignore[index]
                grad_out,
                gg_V,
                dst,
                grad_w_tmp,
                D=D,
                H=H,
                Hd=Hd,
                BLOCK_Hd=BLOCK_Hd,
                num_warps=4,
            )
            grad_w_in = grad_w_tmp.to(dtype=edge_weight.dtype)

        return grad_grad_out, grad_V_in, grad_w_in, None


class _SO2HeadScatterFn(torch.autograd.Function):
    """
    Autograd wrapper for Triton fused head scatter.

    Forward: out[dst[e], d, h, hd] += edge_weight[e, h] * V[e, d, h, hd]
    Backward: Triton kernels with custom double-backward.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        V: torch.Tensor,
        edge_weight: torch.Tensor,
        dst: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")
        if not V.is_cuda:
            raise RuntimeError("SO2HeadScatter requires CUDA tensors")

        E, D, H, Hd = V.shape
        ctx.save_for_backward(V, edge_weight, dst)
        ctx.n_nodes = n_nodes

        # === Step 1. Allocate output in fp32 for stable accumulation ===
        out = torch.zeros(
            (n_nodes, D, H, Hd),
            device=V.device,
            dtype=torch.float32,
        )

        # === Step 2. Launch Triton kernel ===
        BLOCK_Hd = 64
        grid = (E, D * H, triton.cdiv(Hd, BLOCK_Hd))  # type: ignore[attr-defined]
        _so2_head_scatter_kernel[grid](  # type: ignore[index]
            V.contiguous(),
            edge_weight.contiguous(),
            dst.contiguous(),
            out,
            D=D,
            H=H,
            Hd=Hd,
            BLOCK_Hd=BLOCK_Hd,
            num_warps=4,
        )
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        V, edge_weight, dst = ctx.saved_tensors
        grad_V, grad_edge_weight = _SO2HeadScatterBwdFn.apply(
            grad_out, V, edge_weight, dst
        )
        return grad_V, grad_edge_weight, None, None


def so2_baseline_scatter_triton(
    x_message: torch.Tensor,
    edge_env: torch.Tensor,
    dst: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """
    Triton-accelerated fused envelope-weighted scatter-add.

    Parameters
    ----------
    x_message
        Edge messages with shape (E, D, C).
    edge_env
        Envelope weights with shape (E, 1).
    dst
        Destination node indices with shape (E,).
    n_nodes
        Number of nodes N.

    Returns
    -------
    torch.Tensor
        Aggregated output with shape (N, D, C) in float32.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    return _SO2BaselineScatterFn.apply(x_message, edge_env, dst, n_nodes)


def so2_head_scatter_triton(
    V: torch.Tensor,
    edge_weight: torch.Tensor,
    dst: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """
    Triton-accelerated fused head-weighted scatter-add.

    Parameters
    ----------
    V
        Value tensor with shape (E, D, H, Hd).
    edge_weight
        Per-head weights with shape (E, H).
    dst
        Destination node indices with shape (E,).
    n_nodes
        Number of nodes N.

    Returns
    -------
    torch.Tensor
        Aggregated output with shape (N, D, H, Hd) in float32.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    return _SO2HeadScatterFn.apply(V, edge_weight, dst, n_nodes)
