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
