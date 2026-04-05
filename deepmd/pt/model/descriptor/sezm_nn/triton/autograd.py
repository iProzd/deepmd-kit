# SPDX-License-Identifier: LGPL-3.0-or-later
"""Autograd and public API for SeZM Triton rotation kernels."""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import torch
from torch import (
    Tensor,
)

from .constants import (
    SEZM_TRITON_AVAILABLE,
    TritonRotationMode,
)
from .dispatch import (
    coerce_rotation_mode,
    resolve_triton_rotation_mode,
)

if SEZM_TRITON_AVAILABLE:
    from . import custom_ops as _custom_ops  # noqa: F401


def _rotate_to_local_eager(
    *,
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> Tensor:
    """Reference eager implementation for ``D_to_m @ x[src]``."""
    D_to_m = wigner[:, :dim_full, :dim_full].index_select(1, coeff_index)
    return torch.bmm(D_to_m, x.index_select(0, src))


def _rotate_back_eager(
    *,
    x_local: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
) -> Tensor:
    """Reference eager implementation for ``Dt_from_m @ x_local``."""
    Dt_from_m = wigner[:, :dim_full, :dim_full].index_select(2, coeff_index)
    return torch.bmm(Dt_from_m, x_local)


def _resolve_rotation_mode_for_call(
    *,
    dim_full: int,
    coeff_index: Tensor,
    rotation_mode: int | TritonRotationMode | None,
) -> TritonRotationMode:
    """Resolve the effective dispatch mode for one public API call."""
    if rotation_mode is None:
        return resolve_triton_rotation_mode(
            dim_full=int(dim_full),
            reduced_dim=int(coeff_index.numel()),
        )
    return coerce_rotation_mode(rotation_mode)


if SEZM_TRITON_AVAILABLE:

    class _RotateToLocalFunction(torch.autograd.Function):
        """Autograd wrapper for the fused ``global -> local reduced`` rotation."""

        @staticmethod
        def forward(
            ctx: Any,
            x: Tensor,
            src: Tensor,
            wigner: Tensor,
            coeff_index: Tensor,
            dim_full: int,
            rotation_mode: int,
        ) -> Tensor:
            reduced_dim = int(coeff_index.numel())
            out = torch.empty(
                src.shape[0],
                reduced_dim,
                x.shape[2],
                dtype=x.dtype,
                device=x.device,
            )
            torch.ops.deepmd._kernel_sezm_rotate_to_local(
                x,
                src,
                wigner,
                coeff_index,
                out,
                dim_full,
                rotation_mode,
            )
            ctx.save_for_backward(x, src, wigner, coeff_index)
            ctx.dim_full = int(dim_full)
            ctx.rotation_mode = int(rotation_mode)
            return out

        @staticmethod
        def backward(
            ctx: Any,
            grad_out: Tensor,
        ) -> tuple[Tensor, None, Tensor, None, None, None]:
            x, src, wigner, coeff_index = ctx.saved_tensors
            dim_full = int(ctx.dim_full)
            rotation_mode = coerce_rotation_mode(int(ctx.rotation_mode))
            grad_out = grad_out.contiguous()
            grad_edge = torch.empty(
                src.shape[0],
                dim_full,
                x.shape[2],
                dtype=grad_out.dtype,
                device=grad_out.device,
            )
            torch.ops.deepmd._kernel_sezm_rotate_to_local_bwd_dx(
                grad_out,
                wigner,
                coeff_index,
                grad_edge,
                dim_full,
                int(rotation_mode),
            )
            grad_x = torch.zeros_like(x)
            grad_x.index_add_(0, src, grad_edge)

            if rotation_mode == TritonRotationMode.GENERIC_TILED:
                grad_rows = torch.empty(
                    src.shape[0],
                    coeff_index.numel(),
                    dim_full,
                    dtype=wigner.dtype,
                    device=grad_out.device,
                )
                torch.ops.deepmd._kernel_sezm_rotate_to_local_bwd_dw(
                    grad_out,
                    x,
                    src,
                    coeff_index,
                    grad_rows,
                    dim_full,
                    int(rotation_mode),
                )
                grad_wigner = torch.zeros_like(wigner)
                grad_wigner[:, coeff_index, :dim_full] = grad_rows
            else:
                grad_wigner = torch.zeros_like(wigner)
                torch.ops.deepmd._kernel_sezm_rotate_to_local_bwd_dw(
                    grad_out,
                    x,
                    src,
                    coeff_index,
                    grad_wigner,
                    dim_full,
                    int(rotation_mode),
                )
            return grad_x, None, grad_wigner, None, None, None

    class _RotateBackFunction(torch.autograd.Function):
        """Autograd wrapper for the fused ``local reduced -> global`` rotation."""

        @staticmethod
        def forward(
            ctx: Any,
            x_local: Tensor,
            wigner: Tensor,
            coeff_index: Tensor,
            dim_full: int,
            rotation_mode: int,
        ) -> Tensor:
            out = torch.empty(
                x_local.shape[0],
                dim_full,
                x_local.shape[2],
                dtype=x_local.dtype,
                device=x_local.device,
            )
            torch.ops.deepmd._kernel_sezm_rotate_back(
                x_local,
                wigner,
                coeff_index,
                out,
                dim_full,
                rotation_mode,
            )
            ctx.save_for_backward(x_local, wigner, coeff_index)
            ctx.dim_full = int(dim_full)
            ctx.rotation_mode = int(rotation_mode)
            return out

        @staticmethod
        def backward(
            ctx: Any,
            grad_out: Tensor,
        ) -> tuple[Tensor, Tensor, None, None, None]:
            x_local, wigner, coeff_index = ctx.saved_tensors
            dim_full = int(ctx.dim_full)
            rotation_mode = coerce_rotation_mode(int(ctx.rotation_mode))
            grad_out = grad_out.contiguous()
            grad_x_local = torch.empty_like(x_local)
            torch.ops.deepmd._kernel_sezm_rotate_back_bwd_dx(
                grad_out,
                wigner,
                coeff_index,
                grad_x_local,
                dim_full,
                int(rotation_mode),
            )

            if rotation_mode == TritonRotationMode.GENERIC_TILED:
                grad_cols = torch.empty(
                    x_local.shape[0],
                    dim_full,
                    coeff_index.numel(),
                    dtype=wigner.dtype,
                    device=grad_out.device,
                )
                torch.ops.deepmd._kernel_sezm_rotate_back_bwd_dw(
                    grad_out,
                    x_local,
                    coeff_index,
                    grad_cols,
                    dim_full,
                    int(rotation_mode),
                )
                grad_wigner = torch.zeros_like(wigner)
                grad_wigner[:, :dim_full, coeff_index] = grad_cols
            else:
                grad_wigner = torch.zeros_like(wigner)
                torch.ops.deepmd._kernel_sezm_rotate_back_bwd_dw(
                    grad_out,
                    x_local,
                    coeff_index,
                    grad_wigner,
                    dim_full,
                    int(rotation_mode),
                )
            return grad_x_local, grad_wigner, None, None, None


def rotate_to_local_triton(
    x: Tensor,
    src: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
    rotation_mode: int | TritonRotationMode | None = None,
) -> Tensor:
    """
    Apply the fused ``global -> local reduced`` rotation.

    Parameters
    ----------
    x
        Node features with shape ``(N, D, C)``.
    src
        Source-node indices with shape ``(E,)``.
    wigner
        Packed Wigner matrices with shape ``(E, D, D)``.
    coeff_index
        Reduced-layout row indices with shape ``(D_m,)``.
    dim_full
        Full packed SO(3) dimension.
    rotation_mode
        Optional pre-resolved dispatch mode.

    Returns
    -------
    Tensor
        Rotated reduced-layout edge features with shape ``(E, D_m, C)``.
    """
    if not SEZM_TRITON_AVAILABLE:
        raise RuntimeError("SeZM Triton kernels are not available in this environment.")
    src = src.contiguous()
    coeff_index = coeff_index.contiguous()
    resolved_mode = _resolve_rotation_mode_for_call(
        dim_full=int(dim_full),
        coeff_index=coeff_index,
        rotation_mode=rotation_mode,
    )
    if resolved_mode == TritonRotationMode.EAGER_REFERENCE:
        return _rotate_to_local_eager(
            x=x,
            src=src,
            wigner=wigner,
            coeff_index=coeff_index,
            dim_full=int(dim_full),
        )
    return _RotateToLocalFunction.apply(
        x,
        src,
        wigner,
        coeff_index,
        int(dim_full),
        int(resolved_mode),
    )


def rotate_back_triton(
    x_local: Tensor,
    wigner: Tensor,
    coeff_index: Tensor,
    dim_full: int,
    rotation_mode: int | TritonRotationMode | None = None,
) -> Tensor:
    """
    Apply the fused ``local reduced -> global`` rotation.

    Parameters
    ----------
    x_local
        Reduced-layout edge features with shape ``(E, D_m, C)``.
    wigner
        Packed Wigner matrices with shape ``(E, D, D)``.
    coeff_index
        Reduced-layout column indices with shape ``(D_m,)``.
    dim_full
        Full packed SO(3) dimension.
    rotation_mode
        Optional pre-resolved dispatch mode.

    Returns
    -------
    Tensor
        Lifted global-layout edge features with shape ``(E, D, C)``.
    """
    if not SEZM_TRITON_AVAILABLE:
        raise RuntimeError("SeZM Triton kernels are not available in this environment.")
    coeff_index = coeff_index.contiguous()
    resolved_mode = _resolve_rotation_mode_for_call(
        dim_full=int(dim_full),
        coeff_index=coeff_index,
        rotation_mode=rotation_mode,
    )
    if resolved_mode == TritonRotationMode.EAGER_REFERENCE:
        return _rotate_back_eager(
            x_local=x_local,
            wigner=wigner,
            coeff_index=coeff_index,
            dim_full=int(dim_full),
        )
    return _RotateBackFunction.apply(
        x_local,
        wigner,
        coeff_index,
        int(dim_full),
        int(resolved_mode),
    )
