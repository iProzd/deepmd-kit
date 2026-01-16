# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SeZM-Net interaction blocks for DeePMD-kit (PyTorch backend).

This module contains the per-block message passing and nonlinearities used by
`DescrptSeZMNet`.

Design notes
------------
- The caller is responsible for building an `EdgeFeatureCache` once per forward
  pass and reusing it across blocks.
- Features use packed (l, m) layout with ebed_dim=(lmax+1)^2.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    Any,
    cast,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
    get_generator,
)

from .se_zm_helper import (
    EdgeFeatureCache,
    build_m_major_index,
    build_m_major_l_index,
    get_promoted_dtype,
    get_so3_dim_of_lmax,
    map_degree_idx,
    np_safe,
    nvtx_range,
    project_D_to_m,
    project_Dt_from_m,
    safe_numpy_to_tensor,
)
from .se_zm_triton import (
    so2_baseline_scatter_triton,
    so2_head_scatter_triton,
)


class GatedActivation(nn.Module):
    """
    Gated activation for SO(3) equivariant features with per-l independent gates.

    - l=0: Uses the specified activation function
    - l>0: Each degree l has an independent gate derived from the l=0 scalar features.
           The gate for each l is expanded to all m components within that l-block.

    This module also supports the m-major reduced layout used inside SO(2) blocks.
    If `mmax` is provided, the coefficient axis is assumed to follow the truncated
    m-major order built by `build_m_major_index(lmax, mmax)`; otherwise, it is assumed
    to be the full packed (l, m) layout with D=(lmax+1)^2.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum order (|m|) for the m-major reduced layout. If None, use the full
        packed layout with D=(lmax+1)^2.
    channels
        Number of channels per (l, m) coefficient.
    dtype
        Parameter dtype.
    activation_function
        Activation function for l=0 components (e.g., "silu", "tanh", "gelu").
    trainable
        Whether parameters are trainable.
    seed
        Random seed for weight initialization.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        channels: int,
        dtype: torch.dtype,
        activation_function: str = "silu",
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = None if mmax is None else int(mmax)
        if self.mmax is not None:
            if self.mmax < 0:
                raise ValueError("`mmax` must be non-negative")
            if self.mmax > self.lmax:
                raise ValueError("`mmax` must be <= `lmax`")
        self.channels = int(channels)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        self.scalar_act = ActivationFn(activation_function)

        # === Build expand_index for mapping per-l gates to all m components ===
        if self.lmax > 0:
            # expand_index[k] = l-1 for the k-th component in the non-scalar (l>0) portion.
            # This maps each coefficient position to its corresponding gate index.
            if self.mmax is None:
                expand_index = map_degree_idx(self.lmax, device=self.device)[1:] - 1
            else:
                degree_index = build_m_major_l_index(
                    self.lmax, self.mmax, device=self.device
                )
                expand_index = degree_index[1:] - 1
            self.register_buffer("expand_index", expand_index, persistent=True)

            # Linear to generate lmax independent gates from scalar features
            self.gate_linear: nn.Module = MLPLayer(
                self.channels,
                self.lmax * self.channels,
                bias=True,
                activation_function=None,
                precision=self.precision,
                seed=seed,
                trainable=trainable,
            )

            # === Optimized Init: Gate output ~0 => Sigmoid => 0.5 ===
            # Ensures maximum gradient flow (sigmoid'(0) = 0.25 is maximal, sigmoid(0) = 0.5) and unbiased
            # feature scaling at initialization. All high-order features start with
            # equal gating weight (0.5), allowing the model to learn which to suppress
            # or amplify based on the loss signal.
            gen_gate = get_generator(child_seed(seed, 100))
            nn.init.normal_(
                self.gate_linear.matrix, mean=0.0, std=0.01, generator=gen_gate
            )
            if self.gate_linear.bias is not None:
                nn.init.zeros_(self.gate_linear.bias)
        else:
            self.register_buffer(
                "expand_index",
                torch.zeros(0, dtype=torch.long, device=self.device),
                persistent=True,
            )
            self.gate_linear = nn.Identity()

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Features with shape (N, D, C) where D=(lmax+1)^2.

        Returns
        -------
        torch.Tensor
            Gated features with shape (N, D, C).
        """
        # === Step 1. l=0: SiLU activation ===
        x0 = self.scalar_act(x[:, 0:1, :])

        if self.lmax == 0:
            return x0

        # === Step 2. Generate per-l gates from scalar features ===
        # x[:, 0, :] has shape (N, C)
        # gate_linear outputs (N, lmax * C)
        gating_scalars = torch.sigmoid(self.gate_linear(x[:, 0, :]))  # (N, lmax * C)

        # Reshape to (N, lmax, C) then expand to (N, D-1, C)
        gating_scalars = gating_scalars.view(x.shape[0], self.lmax, -1)
        gates = gating_scalars.index_select(
            dim=1, index=self.expand_index
        )  # (N, D-1, C)

        # === Step 3. Apply gates to l>0 components ===
        out = x.new_empty(x.shape)
        out[:, 0:1, :] = x0
        out[:, 1:, :] = x[:, 1:, :] * gates
        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "GatedActivation",
            "@version": 1,
            "lmax": self.lmax,
            "mmax": self.mmax,
            "channels": self.channels,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "activation_function": self.scalar_act.activation,
            "gate_linear": (
                cast("MLPLayer", self.gate_linear).serialize()
                if self.lmax > 0
                else None
            ),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> GatedActivation:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "GatedActivation":
            raise ValueError(f"Invalid class for GatedActivation: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported GatedActivation version: {version}")

        # === Extract gate_linear before creating instance ===
        gate_linear_data = data.pop("gate_linear")
        if "mmax" not in data:
            data["mmax"] = None

        # === Convert precision to dtype ===
        precision = data.pop("precision")
        data["dtype"] = PRECISION_DICT[precision]

        # === Set activation_function default ===
        if "activation_function" not in data:
            data["activation_function"] = "silu"

        # === Set required args ===
        data["trainable"] = True
        data["seed"] = None

        obj = cls(**data)

        # === Restore gate_linear ===
        if gate_linear_data is not None and obj.lmax > 0:
            obj.gate_linear = MLPLayer.deserialize(gate_linear_data)
        return obj


class SeparableRMSNorm(nn.Module):
    """
    Separable RMSNorm with Degree Balancing.

    Features:
        1. Separable: l=0 and l>0 are normalized independently.
        2. Degree Balancing: For l>0, the RMS is computed such that each degree l
           contributes equally, regardless of the number of m components (2l+1).

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    channels
        Number of channels per (l, m) coefficient.
    affine
        Whether to apply per-l learnable scale and bias.
    centering
        Whether to apply mean centering for l=0 features.
    eps
        Small epsilon for numerical stability.
    dtype
        Parameter dtype.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        lmax: int,
        channels: int,
        affine: bool = True,
        centering: bool = True,
        *,
        eps: float = 1e-7,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.affine = affine
        self.centering = centering
        self.dtype = dtype
        self.device = env.DEVICE
        self.eps = float(eps)
        self.compute_dtype = get_promoted_dtype(self.dtype)

        # === Step 1. Learnable Parameters ===
        if self.affine:
            # Per-l affine weights with shape (lmax+1, C)
            self.weight = nn.Parameter(
                torch.ones(
                    self.lmax + 1, self.channels, dtype=self.dtype, device=self.device
                )
            )
            if self.centering:
                # Bias only for l=0
                self.bias = nn.Parameter(
                    torch.zeros(self.channels, dtype=self.dtype, device=self.device)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # === Step 2. Index and Weight Buffers ===
        if self.lmax > 0:
            # Expand index for weight application
            expand_index = map_degree_idx(self.lmax, device=self.device)[1:]
            self.register_buffer("expand_index", expand_index, persistent=True)

            # Degree Balancing: weight each m component by 1/(2l+1) so that each
            # degree l contributes equally to the variance, regardless of the
            # number of m components.
            # Pre-fuse divisors: w_d = 1/(2l+1) / lmax / C, so that
            #   mean_variance = einsum('ndc,d->n', x^2, balance_weight)
            # avoids allocating the intermediate (N, D-1, C) tensor.
            weights_list = []
            scale = 1.0 / (self.lmax * self.channels)
            for l in range(1, self.lmax + 1):
                w = scale / (2 * l + 1)
                weights_list.extend([w] * (2 * l + 1))
            # Shape: (D_non_scalar,) for fused einsum
            balance_weight = torch.tensor(
                weights_list, dtype=self.dtype, device=self.device
            )
            self.register_buffer("balance_weight", balance_weight, persistent=True)
            balance_weight_c = (
                balance_weight
                if self.compute_dtype == self.dtype
                else balance_weight.to(dtype=self.compute_dtype)
            )
            self.register_buffer("balance_weight_c", balance_weight_c, persistent=True)
        else:
            self.register_buffer(
                "expand_index",
                torch.zeros(0, dtype=torch.long, device=self.device),
                persistent=True,
            )
            self.register_buffer(
                "balance_weight",
                torch.zeros(0, dtype=self.dtype, device=self.device),
                persistent=True,
            )
            self.register_buffer(
                "balance_weight_c",
                torch.zeros(0, dtype=self.compute_dtype, device=self.device),
                persistent=True,
            )

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Features with shape (N, D, C) where D=(lmax+1)^2.

        Returns
        -------
        torch.Tensor
            Normalized features with shape (N, D, C).
        """
        in_dtype = x.dtype
        x0 = x[:, :1, :]
        xt = x[:, 1:, :]

        # High precision for RMS calculation
        compute_dtype = self.compute_dtype
        x0_c = x0.to(dtype=compute_dtype)
        xt_c = xt.to(dtype=compute_dtype)

        # === Step 1. l=0: Standard RMS Norm ===
        if self.centering:
            x0_c = x0_c - x0_c.mean(dim=-1, keepdim=True)
        rms0 = torch.sqrt(x0_c.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x0 = (x0_c / rms0).to(dtype=in_dtype)

        if self.affine:
            x0 = x0 * self.weight[0].view(1, 1, -1)
            if self.bias is not None:
                x0 = x0 + self.bias.view(1, 1, -1)

        if xt.numel() == 0:
            return x0

        # === Step 2. l>0: Degree-Balanced RMS Norm ===
        # Fused weighted sum: einsum avoids allocating intermediate (N, D-1, C) tensor.
        # balance_weight already pre-fused with 1/(lmax * C).
        balance_w = self.balance_weight_c
        mean_variance = torch.einsum("ndc,d->n", xt_c * xt_c, balance_w)
        rmst = torch.sqrt(mean_variance + self.eps).view(-1, 1, 1)
        xt = (xt_c / rmst).to(dtype=in_dtype)

        if self.affine:
            wt = torch.index_select(
                self.weight, dim=0, index=self.expand_index
            )  # (D-1, C)
            xt = xt * wt.unsqueeze(0)

        out = x.new_empty(x.shape)
        out[:, :1, :] = x0
        out[:, 1:, :] = xt
        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SeparableRMSNorm",
            "@version": 1,
            "lmax": self.lmax,
            "channels": self.channels,
            "affine": self.affine,
            "centering": self.centering,
            "eps": self.eps,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "weight": np_safe(self.weight) if self.weight is not None else None,
            "bias": np_safe(self.bias) if self.bias is not None else None,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SeparableRMSNorm:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SeparableRMSNorm":
            raise ValueError(f"Invalid class for SeparableRMSNorm: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported SeparableRMSNorm version: {version}")

        # === Extract weight/bias before creating instance ===
        weight_data = data.pop("weight")
        bias_data = data.pop("bias")

        # === Convert precision to dtype ===
        precision = data.pop("precision")
        data["dtype"] = PRECISION_DICT[precision]

        # === Set required args ===
        data["trainable"] = True

        obj = cls(**data)

        # === Restore weight/bias ===
        if obj.weight is not None and weight_data is not None:
            obj.weight.data.copy_(
                safe_numpy_to_tensor(weight_data, device=obj.device, dtype=obj.dtype)
            )
        if obj.bias is not None and bias_data is not None:
            obj.bias.data.copy_(
                safe_numpy_to_tensor(bias_data, device=obj.device, dtype=obj.dtype)
            )
        return obj


class ReducedSeparableRMSNorm(nn.Module):
    """
    Separable RMSNorm for m-major truncated SO(2) layout.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum order kept in the truncated layout.
    channels
        Number of channels per coefficient.
    degree_index_m
        Degree index per coefficient in m-major truncated layout, with shape (D_m_trunc,).
    affine
        Whether to apply per-l learnable scale.
    centering
        Whether to mean-center scalar (l=0) features.
    eps
        Epsilon for numerical stability.
    dtype
        Parameter dtype.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int,
        channels: int,
        degree_index_m: torch.Tensor,
        affine: bool = True,
        centering: bool = True,
        eps: float = 1e-7,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.channels = int(channels)
        self.affine = bool(affine)
        self.centering = bool(centering)
        self.eps = float(eps)
        self.dtype = dtype
        self.device = env.DEVICE
        self.compute_dtype = get_promoted_dtype(self.dtype)

        if degree_index_m.dtype != torch.long:
            degree_index_m = degree_index_m.to(dtype=torch.long)
        self.register_buffer("degree_index_m", degree_index_m, persistent=True)

        deg_ns = degree_index_m[1:]
        weights = torch.zeros(
            deg_ns.numel(), dtype=self.compute_dtype, device=self.device
        )
        scale = 1.0 / (max(1, self.lmax) * max(1, self.channels))
        for l in range(1, self.lmax + 1):
            n_coeff_l = 2 * min(l, self.mmax) + 1
            w_l = scale / float(n_coeff_l)
            weights[deg_ns == l] = w_l
        if torch.any(weights == 0):
            raise ValueError(
                "ReducedSeparableRMSNorm: balance_weight has zeros; degree_index_m may be invalid."
            )
        self.register_buffer("balance_weight", weights, persistent=True)

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones(
                    self.lmax + 1, self.channels, dtype=self.dtype, device=self.device
                )
            )
            self.bias0 = (
                nn.Parameter(
                    torch.zeros(self.channels, dtype=self.dtype, device=self.device)
                )
                if self.centering
                else None
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias0", None)

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor with shape (E, D_m_trunc, C).

        Returns
        -------
        torch.Tensor
            Normalized tensor with shape (E, D_m_trunc, C).

        Raises
        ------
        ValueError
            If the coefficient dimension mismatches degree_index_m.
        """
        if x.shape[1] != self.degree_index_m.numel():
            raise ValueError(
                "Coefficient dimension mismatch for ReducedSeparableRMSNorm."
            )

        in_dtype = x.dtype
        x0 = x[:, :1, :].to(dtype=self.compute_dtype)
        xt = x[:, 1:, :].to(dtype=self.compute_dtype)

        if self.centering:
            x0 = x0 - x0.mean(dim=-1, keepdim=True)
        rms0 = torch.sqrt(x0.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x0n = (x0 / rms0).to(dtype=in_dtype)

        if xt.numel() == 0:
            out = x.new_empty(x.shape)
            out[:, :1, :] = x0n
            return out

        mean_var = torch.einsum("edc,d->e", xt * xt, self.balance_weight)
        rmst = torch.sqrt(mean_var + self.eps).view(-1, 1, 1)
        xtn = (xt / rmst).to(dtype=in_dtype)

        out = x.new_empty(x.shape)
        out[:, :1, :] = x0n
        out[:, 1:, :] = xtn

        if self.affine and self.weight is not None:
            w = torch.index_select(self.weight, dim=0, index=self.degree_index_m)
            out = out * w.unsqueeze(0)
            if self.centering and self.bias0 is not None:
                out[:, 0, :] = out[:, 0, :] + self.bias0
        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "ReducedSeparableRMSNorm",
            "@version": 1,
            "lmax": self.lmax,
            "mmax": self.mmax,
            "channels": self.channels,
            "affine": self.affine,
            "centering": self.centering,
            "eps": self.eps,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "degree_index_m": np_safe(self.degree_index_m),
            "weight": np_safe(self.weight) if self.weight is not None else None,
            "bias0": np_safe(self.bias0) if self.bias0 is not None else None,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> ReducedSeparableRMSNorm:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "ReducedSeparableRMSNorm":
            raise ValueError(f"Invalid class for ReducedSeparableRMSNorm: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported ReducedSeparableRMSNorm version: {version}")

        degree_index_m_data = data.pop("degree_index_m")
        weight_data = data.pop("weight")
        bias_data = data.pop("bias0")
        precision = data.pop("precision")
        data["dtype"] = PRECISION_DICT[precision]
        data["trainable"] = True
        degree_index_m = safe_numpy_to_tensor(
            degree_index_m_data, device=env.DEVICE, dtype=torch.long
        )
        data["degree_index_m"] = degree_index_m

        obj = cls(**data)

        if obj.weight is not None and weight_data is not None:
            obj.weight.data.copy_(
                safe_numpy_to_tensor(weight_data, device=obj.device, dtype=obj.dtype)
            )
        if obj.bias0 is not None and bias_data is not None:
            obj.bias0.data.copy_(
                safe_numpy_to_tensor(bias_data, device=obj.device, dtype=obj.dtype)
            )
        return obj


class ScalarRMSNorm(nn.Module):
    """
    Lightweight RMSNorm for scalar branches.

    Normalizes the last dimension only and applies a learnable scale. Bias is
    intentionally omitted to keep the gate paths minimal.

    Parameters
    ----------
    channels
        Feature dimension of the last axis.
    eps
        Small epsilon for numerical stability.
    dtype
        Parameter dtype.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        channels: int,
        eps: float = 1e-7,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.dtype = dtype
        self.device = env.DEVICE
        self.eps = float(eps)

        self.weight = nn.Parameter(
            torch.ones(self.channels, dtype=self.dtype, device=self.device)
        )

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor with shape (..., C).

        Returns
        -------
        torch.Tensor
            Normalized tensor with the same shape as input.
        """
        in_dtype = x.dtype
        compute_dtype = get_promoted_dtype(self.dtype)
        x_c = x.to(dtype=compute_dtype)

        rms = torch.sqrt(x_c.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = (x_c / rms).to(dtype=in_dtype)

        view_shape = (1,) * (x.dim() - 1) + (self.channels,)
        return x_norm * self.weight.view(*view_shape)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "ScalarRMSNorm",
            "@version": 1,
            "channels": self.channels,
            "eps": self.eps,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "weight": np_safe(self.weight),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> ScalarRMSNorm:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "ScalarRMSNorm":
            raise ValueError(f"Invalid class for ScalarRMSNorm: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported ScalarRMSNorm version: {version}")

        weight_data = data.pop("weight")
        precision = data.pop("precision")
        data["dtype"] = PRECISION_DICT[precision]
        data["trainable"] = True

        obj = cls(**data)
        obj.weight.data.copy_(
            safe_numpy_to_tensor(weight_data, device=obj.device, dtype=obj.dtype)
        )
        return obj


class SO3Linear(nn.Module):
    """
    Degree-wise linear self-interaction using einsum for efficiency.

    This vectorized implementation avoids Python loops by using torch.einsum
    and index_select. The key insight is that weights are shared across all
    m components within each l-block.

    NOTE
    ----
    - weight shape: (lmax+1, C_out, C_in) - per-l linear transformation
    - bias shape: (C_out,) - only applied to l=0 (scalar) components
    - expand_index: maps each (l,m) position to its l value for weight lookup
    - Uses einsum 'bmi, lci -> blmc' pattern for batched per-l matmul
    - Avoids Python for-loops and torch.cat operations

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    in_channels
        Number of input channels per (l, m) coefficient.
    out_channels
        Number of output channels per (l, m) coefficient.
    dtype
        Parameter dtype.
    trainable
        Whether parameters are trainable.
    seed
        Random seed for weight initialization.
    """

    def __init__(
        self,
        *,
        lmax: int,
        in_channels: int,
        out_channels: int,
        dtype: torch.dtype,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.ebed_dim = get_so3_dim_of_lmax(self.lmax)

        # === Step 1. Per-l weight matrix: (lmax+1, C_out, C_in) ===
        # Each l has an independent C_out x C_in linear transformation
        # that is shared across all 2l+1 m components.
        bound = 1.0 / math.sqrt(self.in_channels)
        self.weight = nn.Parameter(
            torch.empty(
                self.lmax + 1,
                self.out_channels,
                self.in_channels,
                dtype=self.dtype,
                device=self.device,
            )
        )
        generator = get_generator(seed)
        nn.init.uniform_(self.weight, -bound, bound, generator=generator)

        # === Step 2. Bias only for l=0 (scalar components) ===
        self.bias = nn.Parameter(
            torch.zeros(self.out_channels, dtype=self.dtype, device=self.device)
        )

        # === Step 3. Precompute expand_index for weight lookup ===
        # expand_index[i] = l for position i in the packed (l,m) layout
        # This maps each (l,m) component to its corresponding weight index.
        self.register_buffer(
            "expand_index",
            map_degree_idx(self.lmax, device=self.device),
            persistent=True,
        )

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input features with shape (N, D, C_in) where D=(lmax+1)^2.

        Returns
        -------
        torch.Tensor
            Order-wise mixed features with shape (N, D, C_out).
        """
        # === Step 1. Expand weight: (lmax+1, C_out, C_in) -> (D, C_out, C_in) ===
        # Use index_select to duplicate each weight matrix to all m components.
        weight_expanded = torch.index_select(
            self.weight, dim=0, index=self.expand_index
        )  # (D, C_out, C_in)

        # === Step 2. Batched per-l matmul using einsum ===
        # Pattern explanation:
        #   b: batch dimension (N)
        #   m: m-component dimension (D)
        #   i: input channels (C_in)
        #   l: lookup dimension (D, same as m after expansion)
        #   c: output channels (C_out)
        # Result: (N, D, C_out) where each position (b, m, :) = weight[l] @ x[b, m, :]
        out = torch.einsum("bmi,mci->bmc", x, weight_expanded)  # (N, D, C_out)

        # === Step 3. Add bias only to l=0 (index 0) ===
        out[:, 0, :] = out[:, 0, :] + self.bias.view(1, -1)

        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SO3Linear",
            "@version": 1,
            "lmax": self.lmax,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "weight": np_safe(self.weight),
            "bias": np_safe(self.bias),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SO3Linear:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SO3Linear":
            raise ValueError(f"Invalid class for SO3Linear: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported SO3Linear version: {version}")

        # === Extract weight/bias before creating instance ===
        weight_data = data.pop("weight")
        bias_data = data.pop("bias")

        # === Convert precision to dtype ===
        precision = data.pop("precision")
        data["dtype"] = PRECISION_DICT[precision]

        # === Create instance with remaining args ===
        data["trainable"] = True
        data["seed"] = None
        obj = cls(**data)

        # === Restore weight/bias ===
        obj.weight.data.copy_(
            safe_numpy_to_tensor(weight_data, device=obj.device, dtype=obj.dtype)
        )
        obj.bias.data.copy_(
            safe_numpy_to_tensor(bias_data, device=obj.device, dtype=obj.dtype)
        )
        return obj


class SO2Linear(nn.Module):
    """
    SO(2) linear mixing in the edge-aligned frame.

    Layout invariant:
    - Coefficient axis uses an m-major layout truncated by mmax.
    - m = 0: l = 0..lmax (single coefficient per l)
    - for each m = 1..mmax:
        - negative part: l = m..lmax, coefficient (l, -m)
        - positive part: l = m..lmax, coefficient (l, +m)

    The m-major layout keeps each |m| group contiguous.
    Mixing preserves SO(2) equivariance via the constrained 2x2 coupling of the
    (-m, +m) pair.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum SO(2) order (|m|) to mix. If None, defaults to `lmax`.
    in_channels
        Number of input channels per (l, m) coefficient.
    out_channels
        Number of output channels per (l, m) coefficient.
    dtype
        Parameter dtype.
    seed
        Random seed for weight initialization.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        in_channels: int,
        out_channels: int,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        # === Step 1. Precompute index buffers for the m-major layout ===
        # Indices refer to the coefficient axis of the m-major reduced layout.
        m0_size = self.lmax + 1
        self.register_buffer(
            "m0_idx",
            torch.arange(m0_size, device=self.device, dtype=torch.long),
            persistent=True,
        )

        pos_indices_list: list[torch.Tensor] = []
        neg_indices_list: list[torch.Tensor] = []
        num_l_list: list[int] = []

        offset = m0_size
        for m in range(1, self.mmax + 1):
            num_l = self.lmax - m + 1
            neg_idx = torch.arange(
                offset, offset + num_l, device=self.device, dtype=torch.long
            )
            pos_idx = torch.arange(
                offset + num_l, offset + 2 * num_l, device=self.device, dtype=torch.long
            )
            neg_indices_list.append(neg_idx)
            pos_indices_list.append(pos_idx)
            num_l_list.append(num_l)
            offset += 2 * num_l

        self.reduced_dim = int(offset)

        if len(pos_indices_list) > 0:
            self.register_buffer(
                "pos_indices", torch.cat(pos_indices_list), persistent=True
            )
            self.register_buffer(
                "neg_indices", torch.cat(neg_indices_list), persistent=True
            )
            offsets = [0]
            for n in num_l_list:
                offsets.append(offsets[-1] + n)
            self._m_offsets = offsets
        else:
            self.register_buffer(
                "pos_indices",
                torch.empty(0, device=self.device, dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                "neg_indices",
                torch.empty(0, device=self.device, dtype=torch.long),
                persistent=True,
            )
            self._m_offsets = [0]

        # === Step 2. Mixing per |m| group, cross-l allowed, bias only for scalar index ===
        num_m0 = self.lmax + 1
        self.linear_m0 = MLPLayer(
            num_m0 * self.in_channels,
            num_m0 * self.out_channels,
            bias=False,
            activation_function=None,
            precision=self.precision,
            seed=child_seed(seed, 0),
            trainable=trainable,
        )
        self.bias0 = nn.Parameter(
            torch.zeros(self.out_channels, device=self.device, dtype=self.dtype)
        )

        # For |m|>0, SO(2) equivariance requires 2x2 block structure on (Re, Im) pairs.
        # Output dimension is doubled for complex mixing: (a, -b; b, a) constraint.
        self.linears_m: nn.ModuleList = nn.ModuleList()
        for m in range(1, self.mmax + 1):
            num_l = self.lmax - m + 1
            fc = MLPLayer(
                num_l * self.in_channels,
                2 * num_l * self.out_channels,
                bias=False,
                activation_function=None,
                precision=self.precision,
                seed=child_seed(seed, m),
                trainable=trainable,
            )
            # Apply scaling for SO(2) equivariance
            fc.matrix.data.mul_(1.0 / math.sqrt(2.0))
            self.linears_m.append(fc)

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input with shape (N, D_m_trunc, Cin), where D_m_trunc is the
            coefficient dimension of the m-major layout truncated by `mmax`.

        Returns
        -------
        torch.Tensor
            Output with shape (N, D_m_trunc, Cout), where Cout is output channels.
        """
        n_atom, D_m_trunc, Cin = x.shape
        out = x.new_empty(n_atom, D_m_trunc, self.out_channels)

        # === Step 1. m = 0 group ===
        m0_idx: torch.Tensor = self.m0_idx
        x_m0 = x[:, m0_idx, :].reshape(n_atom, -1)
        y_m0 = self.linear_m0(x_m0).reshape(n_atom, m0_idx.numel(), self.out_channels)
        out[:, m0_idx, :] = y_m0
        out[:, 0, :].add_(self.bias0)

        # === Step 2. |m| > 0 groups (complex mixing via 2x2 block structure) ===
        pos_indices: torch.Tensor = self.pos_indices
        neg_indices: torch.Tensor = self.neg_indices

        for m, linear in enumerate(self.linears_m, start=1):
            start_off = self._m_offsets[m - 1]
            end_off = self._m_offsets[m]
            pos_idx = pos_indices[start_off:end_off]
            neg_idx = neg_indices[start_off:end_off]
            num_l = int(end_off - start_off)

            # Treat (neg, pos) as (Re, Im) complex pair for |m| = const.
            x_neg = x[:, neg_idx, :].reshape(n_atom, -1)
            x_pos = x[:, pos_idx, :].reshape(n_atom, -1)
            x_pair = torch.stack([x_neg, x_pos], dim=1)  # (n_atom, 2, num_l * Cin)

            # Linear produces two complex coefficients per output channel.
            # (n_atom, 2, num_l * Cin) -> (n_atom, 2, 2 * num_l * Cout)
            x_pair = linear(x_pair)

            # Split into 4 components for complex mixing
            out_half = num_l * self.out_channels
            x_pair = x_pair.view(n_atom, 4, out_half)
            x_r_0, x_i_0, x_r_1, x_i_1 = x_pair.unbind(dim=1)

            # Complex multiplication formula: y_r = a*x_r - b*x_i, y_i = b*x_r + a*x_i
            # With learned (a, b) encoded in linear weights:
            y_r = x_r_0 - x_i_1  # (n_atom, out_half)
            y_i = x_r_1 + x_i_0

            # Reshape back to (n_atom, num_l, Cout)
            out[:, neg_idx, :] = y_r.view(n_atom, num_l, self.out_channels)
            out[:, pos_idx, :] = y_i.view(n_atom, num_l, self.out_channels)

        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SO2Linear",
            "@version": 1,
            "lmax": self.lmax,
            "mmax": self.mmax,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "bias0": np_safe(self.bias0),
            "linear_m0": self.linear_m0.serialize(),
            "linears_m": [net.serialize() for net in self.linears_m],
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SO2Linear:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SO2Linear":
            raise ValueError(f"Invalid class for SO2Linear: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported SO2Linear version: {version}")

        # === Extract linear_m0, bias0, linears_m before creating instance ===
        linear_m0_data = data.pop("linear_m0")
        bias0_data = data.pop("bias0")
        linears_m_data = data.pop("linears_m")

        # === Convert precision to dtype ===
        precision = data.pop("precision")
        data["dtype"] = PRECISION_DICT[precision]

        # === Set required args ===
        data["trainable"] = True
        data["seed"] = None

        obj = cls(**data)

        # === Restore linear_m0, bias0, linears_m ===
        obj.linear_m0 = MLPLayer.deserialize(linear_m0_data)
        obj.bias0.data.copy_(
            safe_numpy_to_tensor(bias0_data, device=obj.device, dtype=obj.dtype)
        )
        obj.linears_m = nn.ModuleList(
            [MLPLayer.deserialize(state) for state in linears_m_data]
        )
        return obj


class SO2Convolution(nn.Module):
    """
    Linearized SO(2) convolution with precomputed edge cache.

    Supports multi-layer SO(2) mixing:
    - so2_layers=1: Standard single-layer SO2Linear
    - so2_layers>=2: SO2Linear -> (Edge-gated Non-linearity -> SO2Linear) x (n-1)

    The intermediate non-linearity uses edge invariants (radial features) to
    generate gates, enabling the model to learn more expressive edge-wise
    message functions.

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum SO(2) order (|m|). If None, defaults to lmax.
    channels
        Number of channels per (l, m) coefficient.
    so2_norm
        If True, apply intermediate ReducedSeparableRMSNorm between SO(2) mixing layers.
        When False (default), no normalization is applied between layers.
    so2_layers
        Number of SO2Linear layers per convolution (default: 2).
    n_atten_head
        Number of gated attention heads when aggregating messages in SO(2) convolution.
        0 means a plain envelope-weighted scatter-sum is applied.
    use_triton
        Whether to use Triton kernels for scatter operations.
    eps
        Small epsilon for gate-side RMSNorm.
    dtype
        Parameter dtype.
    seed
        Random seed for weight initialization.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        channels: int,
        so2_norm: bool = False,
        so2_layers: int = 1,
        n_atten_head: int = 0,
        use_triton: bool = False,
        eps: float = 1e-7,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.channels = int(channels)
        self.so2_norm = bool(so2_norm)
        self.so2_layers = int(so2_layers)
        if self.so2_layers < 1:
            raise ValueError("`so2_layers` must be >= 1")
        self.n_atten_head = int(n_atten_head)
        if self.n_atten_head < 0:
            raise ValueError("`n_atten_head` must be non-negative")
        if self.n_atten_head > 0 and self.channels % self.n_atten_head != 0:
            raise ValueError(
                "`channels` must be divisible by `n_atten_head` when attention is enabled"
            )
        self.head_dim = (
            None if self.n_atten_head == 0 else int(self.channels // self.n_atten_head)
        )
        self.use_triton = bool(use_triton)
        self.eps = float(eps)
        self.ebed_dim_full = get_so3_dim_of_lmax(self.lmax)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.compute_dtype = get_promoted_dtype(self.dtype)

        # === Step 1. Precompute coefficient indices for m-major reduced layout ===
        coeff_index_m = build_m_major_index(self.lmax, self.mmax, device=self.device)
        degree_index_m = build_m_major_l_index(self.lmax, self.mmax, device=self.device)
        self.register_buffer("coeff_index_m", coeff_index_m, persistent=True)
        self.register_buffer("degree_index_m", degree_index_m, persistent=True)
        self.reduced_dim = int(coeff_index_m.numel())

        # === Step 2. Split deterministic seeds at the module top-level ===
        seed_so2_stack = child_seed(seed, 0)
        seed_non_linearities = child_seed(seed, 1)
        seed_so3 = child_seed(seed, 2)
        seed_gate = child_seed(seed, 3)

        # === Step 3. Multiple SO2Linear layers ===
        self.so2_linears = nn.ModuleList(
            [
                SO2Linear(
                    lmax=self.lmax,
                    mmax=self.mmax,
                    in_channels=self.channels,
                    out_channels=self.channels,
                    dtype=self.dtype,
                    seed=child_seed(seed_so2_stack, i),
                    trainable=trainable,
                )
                for i in range(self.so2_layers)
            ]
        )

        # === Step 4. Intermediate norms (Optional) ===
        inter_norms: list[nn.Module] = []
        if self.so2_norm:
            for _ in range(max(0, self.so2_layers - 1)):
                inter_norms.append(
                    ReducedSeparableRMSNorm(
                        lmax=self.lmax,
                        mmax=self.mmax,
                        channels=self.channels,
                        degree_index_m=self.degree_index_m,
                        eps=self.eps,
                        dtype=self.dtype,
                        trainable=trainable,
                    )
                )
        else:
            for _ in range(max(0, self.so2_layers - 1)):
                inter_norms.append(nn.Identity())
        inter_norms.append(nn.Identity())
        self.so2_inter_norms = nn.ModuleList(inter_norms)

        # === Step 5. Intermediate non-linearity ===
        non_linearities: list[nn.Module] = []
        for i in range(max(0, self.so2_layers - 1)):
            non_linearities.append(
                GatedActivation(
                    lmax=self.lmax,
                    mmax=self.mmax,
                    channels=self.channels,
                    dtype=self.dtype,
                    trainable=trainable,
                    seed=child_seed(seed_non_linearities, i),
                )
            )
        non_linearities.append(nn.Identity())
        self.non_linearities = nn.ModuleList(non_linearities)

        # === Step 6. Optional head-wise gating components ===
        # Edge gate: dst (normalized) + radial (raw) + message scalar (normalized)
        # Head gate: post-aggregate per-head scaling with normalization + bounded gamma
        self.norm_dst_for_gate: ScalarRMSNorm | None = None
        self.norm_msg_for_gate: ScalarRMSNorm | None = None
        self.norm_dst_for_head: ScalarRMSNorm | None = None
        self.proj_dst: MLPLayer | None = None
        self.proj_rad: MLPLayer | None = None
        self.proj_msg: MLPLayer | None = None
        self.proj_head: MLPLayer | None = None
        if self.n_atten_head > 0:
            self.norm_dst_for_gate = ScalarRMSNorm(
                channels=self.channels,
                eps=self.eps,
                dtype=self.dtype,
                trainable=trainable,
            )
            self.norm_dst_for_head = ScalarRMSNorm(
                channels=self.channels,
                eps=self.eps,
                dtype=self.dtype,
                trainable=trainable,
            )
            self.norm_msg_for_gate = ScalarRMSNorm(
                channels=self.channels,
                eps=self.eps,
                dtype=self.dtype,
                trainable=trainable,
            )
            # Edge gate projections: dst baseline (normalized), radial discriminator (raw), message scalar (normalized)
            self.proj_dst = MLPLayer(
                self.channels,
                self.n_atten_head,
                activation_function=None,
                bias=True,
                precision=self.precision,
                seed=child_seed(seed_gate, 0),
                trainable=trainable,
            )
            self.proj_rad = MLPLayer(
                self.channels,
                self.n_atten_head,
                activation_function=None,
                bias=True,
                precision=self.precision,
                seed=child_seed(seed_gate, 1),
                trainable=trainable,
            )
            self.proj_msg = MLPLayer(
                self.channels,
                self.n_atten_head,
                activation_function=None,
                bias=True,
                precision=self.precision,
                seed=child_seed(seed_gate, 4),
                trainable=trainable,
            )
            # Post-aggregate head gate projection
            self.proj_head = MLPLayer(
                self.channels,
                self.n_atten_head,
                activation_function=None,
                bias=True,
                precision=self.precision,
                seed=child_seed(seed_gate, 2),
                trainable=trainable,
            )
            # Initialization: Normal(0, 0.01) for weights, zeros for bias (logits≈0)
            gen_proj_dst = get_generator(child_seed(seed_gate, 10))
            gen_proj_rad = get_generator(child_seed(seed_gate, 11))
            gen_proj_msg = get_generator(child_seed(seed_gate, 12))
            gen_proj_head = get_generator(child_seed(seed_gate, 13))
            nn.init.normal_(
                self.proj_dst.matrix, mean=0.0, std=0.01, generator=gen_proj_dst
            )
            if self.proj_dst.bias is not None:
                nn.init.zeros_(self.proj_dst.bias)
            nn.init.normal_(
                self.proj_rad.matrix, mean=0.0, std=0.01, generator=gen_proj_rad
            )
            if self.proj_rad.bias is not None:
                nn.init.zeros_(self.proj_rad.bias)
            nn.init.normal_(
                self.proj_msg.matrix, mean=0.0, std=0.01, generator=gen_proj_msg
            )
            if self.proj_msg.bias is not None:
                nn.init.zeros_(self.proj_msg.bias)
            nn.init.normal_(
                self.proj_head.matrix, mean=0.0, std=0.01, generator=gen_proj_head
            )
            if self.proj_head.bias is not None:
                nn.init.zeros_(self.proj_head.bias)
            # gamma_head_raw parameterized to keep gamma_eff in (0, 2)
            self.gamma_head_raw = nn.Parameter(
                torch.zeros(
                    self.n_atten_head,
                    dtype=self.compute_dtype,
                    device=self.device,
                )
            )
            self.msg_gate_scale_max = 0.2
            self.msg_gate_scale_raw = nn.Parameter(
                torch.full(
                    (self.n_atten_head,),
                    -2.0,
                    dtype=self.compute_dtype,
                    device=self.device,
                )
            )
        else:
            self.msg_gate_scale_max = 0.2
            self.register_parameter("msg_gate_scale_raw", None)

        # === Step 7. Final SO3Linear to mix channels ===
        self.so3_linear = SO3Linear(
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=self.channels,
            dtype=dtype,
            trainable=trainable,
            seed=seed_so3,
        )
        nn.init.zeros_(self.so3_linear.weight)
        nn.init.zeros_(self.so3_linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_cache: EdgeFeatureCache,
        radial_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Node features with shape (N, D, C) where D=(lmax+1)^2 is the SO(3) dimension.
        edge_cache
            Precomputed edge cache. Must be compatible with this block's lmax.
        radial_feat
            Per-edge radial features with shape (E, lmax+1, C), already fused with
            edge type features.

        Returns
        -------
        torch.Tensor
            Message updates with shape (N, D, C).
        """
        num_edges = edge_cache.src.size(0)
        if num_edges == 0:
            return torch.zeros_like(x)

        src, dst = edge_cache.src, edge_cache.dst
        x_src = x[src]  # (E, D, C)

        # === Step 1. Rotate to edge-aligned local frame ===
        with nvtx_range("SO2Conv/rotate_to_local"):
            D_full = edge_cache.D_full
            assert D_full is not None
            D_m_prime = project_D_to_m(
                D_full=D_full,
                coeff_index_m=self.coeff_index_m,
                ebed_dim_full=self.ebed_dim_full,
                cache=edge_cache.D_to_m_cache,
                key_lmax=self.lmax,
                key_mmax=self.mmax,
            )
            x_local = torch.bmm(D_m_prime, x_src)  # (E, D_m_trunc, C)

        # === Step 2. Select radial/type features for reduced layout ===
        with nvtx_range("SO2Conv/radial_fuse"):
            rad_feat = radial_feat[:, self.degree_index_m, :]  # (E, D_m_trunc, C)
            x_local = x_local * rad_feat  # (E, D_m_trunc, C)

        # === Step 3. Multi-layer SO(2) mixing ===
        with nvtx_range("SO2Conv/so2_layers"):
            for layer_idx, (so2_linear, inter_norm, non_linear) in enumerate(
                zip(self.so2_linears, self.so2_inter_norms, self.non_linearities)
            ):
                x_local = so2_linear(x_local)

                if layer_idx == 0:
                    bias_correction = so2_linear.bias0 * (
                        rad_feat[:, 0, :] * edge_cache.edge_env - 1.0
                    )
                    x_local[:, 0, :].add_(bias_correction)

                x_local = inter_norm(x_local)
                x_local = non_linear(x_local)

        # === Step 4. Rotate back to global frame ===
        with nvtx_range("SO2Conv/rotate_back"):
            Dt_full = edge_cache.Dt_full
            assert Dt_full is not None
            Dt_from_m = project_Dt_from_m(
                Dt_full=Dt_full,
                coeff_index_m=self.coeff_index_m,
                ebed_dim_full=self.ebed_dim_full,
                cache=edge_cache.Dt_from_m_cache,
                key_lmax=self.lmax,
                key_mmax=self.mmax,
            )
            x_message = torch.bmm(Dt_from_m, x_local)  # (E, D, C)

        # === Step 5. Aggregate with optional head-wise gating ===
        with nvtx_range("SO2Conv/aggregate"):
            if self.n_atten_head == 0:
                # Baseline path: fused envelope-weighted scatter add -> degree norm
                # Custom backward/gradgrad keeps force training in Triton.
                if self.use_triton and x_message.is_cuda:
                    out = so2_baseline_scatter_triton(
                        x_message, edge_cache.edge_env, dst, x.shape[0]
                    )
                    out = out.to(dtype=self.dtype) * edge_cache.inv_sqrt_deg
                else:
                    x_message = x_message * edge_cache.edge_env.view(-1, 1, 1)
                    out = x.new_zeros(x.shape)
                    out.index_add_(0, dst, x_message)
                    out = out * edge_cache.inv_sqrt_deg
            else:
                assert self.head_dim is not None
                assert self.proj_dst is not None
                assert self.proj_rad is not None
                assert self.proj_head is not None
                assert self.norm_dst_for_gate is not None
                assert self.norm_msg_for_gate is not None
                assert self.norm_dst_for_head is not None

                # === Step 5.1. Extract scalar features for gating ===
                x_l0 = x[:, 0, :]  # (N, C) in self.dtype
                radial_l0 = radial_feat[:, 0, :]  # (E, C) in self.dtype
                msg_l0 = x_message[:, 0, :]  # (E, C) message scalar

                # === Step 5.2. Compute edge gate (fp32+ logits path) ===
                # Edge gate logits: dst scalar normalized, radial unchanged, msg scalar normalized with bounded feedback
                compute_dtype = self.compute_dtype
                dst_logits = self.proj_dst(self.norm_dst_for_gate(x_l0)).to(
                    dtype=compute_dtype
                )  # (N, H) fp32
                radial_logits = self.proj_rad(radial_l0).to(
                    dtype=compute_dtype
                )  # (E, H)
                msg_logits = self.proj_msg(self.norm_msg_for_gate(msg_l0)).to(
                    dtype=compute_dtype
                )  # (E, H)
                assert self.msg_gate_scale_raw is not None
                msg_gate_scale = (
                    self.msg_gate_scale_max * torch.sigmoid(self.msg_gate_scale_raw)
                ).view(1, -1)  # (1, H) fp32+
                edge_logits = (
                    dst_logits.index_select(0, dst)
                    + radial_logits
                    + msg_gate_scale * msg_logits
                )  # (E, H) fp32+
                edge_gate = 2.0 * torch.sigmoid(edge_logits.clamp(-6.0, 6.0))
                edge_weight = (
                    edge_cache.edge_env.view(-1, 1).to(dtype=compute_dtype) * edge_gate
                )  # (E, H) in compute_dtype

                # === Step 5.3. Head-wise scatter aggregation (fp32+ accumulation) ===
                V = x_message.reshape(
                    x_message.shape[0],
                    self.ebed_dim_full,
                    self.n_atten_head,
                    self.head_dim,
                )
                # Custom backward/gradgrad keeps force training in Triton.
                if self.use_triton and V.is_cuda:
                    out_heads = so2_head_scatter_triton(
                        V.to(dtype=compute_dtype), edge_weight, dst, x.shape[0]
                    )
                else:
                    msg = V.to(dtype=compute_dtype) * edge_weight.view(
                        -1, 1, self.n_atten_head, 1
                    )
                    out_heads = torch.zeros(
                        x.shape[0],
                        self.ebed_dim_full,
                        self.n_atten_head,
                        self.head_dim,
                        device=x.device,
                        dtype=compute_dtype,
                    )
                    out_heads.index_add_(0, dst, msg)

                # === Step 5.4. Compute post-aggregate head gate (fp32+ logits path) ===
                head_logits = self.proj_head(self.norm_dst_for_head(x_l0)).to(
                    dtype=compute_dtype
                )  # (N, H)
                head_gate = 2.0 * torch.sigmoid(head_logits.clamp(-6.0, 6.0))  # (N, H)
                gamma_eff = 2.0 * torch.sigmoid(self.gamma_head_raw)  # (H,)
                scale = head_gate * gamma_eff.view(1, -1)  # (N, H) compute_dtype

                # === Step 5.5. Apply head-wise scaling ===
                out_heads = out_heads * scale.view(-1, 1, self.n_atten_head, 1)

                # === Step 5.6. Apply degree normalization ===
                out = out_heads.view(x.shape[0], self.ebed_dim_full, self.channels)
                out = out * edge_cache.inv_sqrt_deg.to(dtype=compute_dtype)
                out = out.to(dtype=self.dtype)

        # === Step 6. Final channel mixing ===
        with nvtx_range("SO2Conv/so3_linear"):
            out = self.so3_linear(out)  # (N, D, C)
        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SO2Convolution",
            "@version": 1,
            "lmax": self.lmax,
            "mmax": self.mmax,
            "channels": self.channels,
            "so2_norm": self.so2_norm,
            "so2_layers": self.so2_layers,
            "n_atten_head": self.n_atten_head,
            "use_triton": self.use_triton,
            "eps": self.eps,
            "so2_linears": [net.serialize() for net in self.so2_linears],
            "non_linearities": (
                [act.serialize() for act in self.non_linearities[:-1]]
                if self.so2_layers > 1
                else None
            ),
            "so2_inter_norms": [
                norm.serialize() if isinstance(norm, ReducedSeparableRMSNorm) else None
                for norm in self.so2_inter_norms
            ],
            "so3_linear": self.so3_linear.serialize(),
            "gate_state": (
                None
                if self.n_atten_head == 0
                else {
                    "norm_dst_for_gate": self.norm_dst_for_gate.serialize(),
                    "norm_msg_for_gate": self.norm_msg_for_gate.serialize(),
                    "norm_dst_for_head": self.norm_dst_for_head.serialize(),
                    "proj_dst": cast("MLPLayer", self.proj_dst).serialize(),
                    "proj_rad": cast("MLPLayer", self.proj_rad).serialize(),
                    "proj_msg": cast("MLPLayer", self.proj_msg).serialize(),
                    "proj_head": cast("MLPLayer", self.proj_head).serialize(),
                    "gamma_head_raw": np_safe(self.gamma_head_raw),
                    "msg_gate_scale_max": self.msg_gate_scale_max,
                    "msg_gate_scale_raw": np_safe(self.msg_gate_scale_raw),
                }
            ),
            "precision": RESERVED_PRECISION_DICT[self.dtype],
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SO2Convolution:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SO2Convolution":
            raise ValueError(f"Invalid class for SO2Convolution: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported SO2Convolution version: {version}")

        so2_linears_data = data.pop("so2_linears")
        non_linearities_data = data.pop("non_linearities")
        so2_inter_norms_data = data.pop("so2_inter_norms", None)
        so3_linear_data = data.pop("so3_linear")
        gate_state = data.pop("gate_state", None)

        precision = data.pop("precision")
        data["dtype"] = PRECISION_DICT[precision]
        data["trainable"] = True
        data["seed"] = None
        if "eps" not in data:
            data["eps"] = 1e-7

        obj = cls(**data)

        obj.so2_linears = nn.ModuleList(
            [SO2Linear.deserialize(state) for state in so2_linears_data]
        )
        if non_linearities_data is not None:
            non_linearities = [
                GatedActivation.deserialize(state) for state in non_linearities_data
            ]
            non_linearities.append(nn.Identity())
            obj.non_linearities = nn.ModuleList(non_linearities)
        if so2_inter_norms_data is not None:
            inter_norms: list[nn.Module] = []
            for state in so2_inter_norms_data:
                if state is None:
                    inter_norms.append(nn.Identity())
                else:
                    inter_norms.append(ReducedSeparableRMSNorm.deserialize(state))
            obj.so2_inter_norms = nn.ModuleList(inter_norms)
        obj.so3_linear = SO3Linear.deserialize(so3_linear_data)
        if gate_state is not None and obj.n_atten_head > 0:
            obj.norm_dst_for_gate = ScalarRMSNorm.deserialize(
                gate_state["norm_dst_for_gate"]
            )
            obj.norm_msg_for_gate = ScalarRMSNorm.deserialize(
                gate_state["norm_msg_for_gate"]
            )
            obj.norm_dst_for_head = ScalarRMSNorm.deserialize(
                gate_state["norm_dst_for_head"]
            )
            obj.proj_dst = MLPLayer.deserialize(gate_state["proj_dst"])
            obj.proj_rad = MLPLayer.deserialize(gate_state["proj_rad"])
            obj.proj_msg = MLPLayer.deserialize(gate_state["proj_msg"])
            obj.proj_head = MLPLayer.deserialize(gate_state["proj_head"])
            gamma_head_raw = gate_state.get("gamma_head_raw")
            if gamma_head_raw is not None:
                obj.gamma_head_raw = nn.Parameter(
                    safe_numpy_to_tensor(
                        gamma_head_raw, device=obj.device, dtype=obj.compute_dtype
                    ).view(obj.n_atten_head)
                )
            msg_gate_scale_raw = gate_state.get("msg_gate_scale_raw")
            msg_gate_scale_max = gate_state.get(
                "msg_gate_scale_max", obj.msg_gate_scale_max
            )
            obj.msg_gate_scale_max = float(msg_gate_scale_max)
            if msg_gate_scale_raw is not None:
                obj.msg_gate_scale_raw = nn.Parameter(
                    safe_numpy_to_tensor(
                        msg_gate_scale_raw, device=obj.device, dtype=obj.compute_dtype
                    ).view(obj.n_atten_head)
                )
        return obj


class EquivariantFFN(nn.Module):
    """
    Full equivariant FFN operating on all spherical harmonic degrees.

    Structure: SO3 linear (in) -> GatedActivation -> SO3 linear (out)

    GatedActivation serves as the unified "activation" for equivariant networks,
    analogous to SiLU in standard MLPs, but respecting SO(3) equivariance:
    - l=0: Uses the specified activation function
    - l>0: sigmoid gate from l=0 scalar features

    Parameters
    ----------
    lmax
        Maximum degree.
    channels
        Number of channels per (l, m) coefficient.
    hidden_channels
        Hidden dimension for the FFN.
    dtype
        Parameter dtype.
    activation_function
        Activation function for l=0 components (e.g., "silu", "tanh", "gelu").
    trainable
        Whether parameters are trainable.
    seed
        Random seed for weight initialization.
    """

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        hidden_channels: int,
        dtype: torch.dtype,
        activation_function: str = "silu",
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.activation_function = activation_function
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        # === Step 0. Split deterministic seeds at the module top-level ===
        seed_so3_in = child_seed(seed, 0)
        seed_act = child_seed(seed, 1)
        seed_so3_out = child_seed(seed, 2)

        # === First SO3Linear for channel mixing ===
        self.so3_linear_1 = SO3Linear(
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=self.hidden_channels,
            dtype=dtype,
            trainable=trainable,
            seed=seed_so3_in,
        )

        # === Equivariant activation ===
        self.act = GatedActivation(
            lmax=self.lmax,
            channels=self.hidden_channels,
            dtype=dtype,
            activation_function=activation_function,
            trainable=trainable,
            seed=seed_act,
        )

        # === Second SO3Linear for channel mixing ===
        self.so3_linear_2 = SO3Linear(
            lmax=self.lmax,
            in_channels=self.hidden_channels,
            out_channels=self.channels,
            dtype=dtype,
            trainable=trainable,
            seed=seed_so3_out,
        )
        nn.init.zeros_(self.so3_linear_2.weight)
        nn.init.zeros_(self.so3_linear_2.bias)

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input with shape (N, D, C) where D=(lmax+1)^2.

        Returns
        -------
        torch.Tensor
            Output with shape (N, D, C).
        """
        # === Step 1. Input up projection ===
        x = self.so3_linear_1(x)

        # === Step 2. Equivariant activation ===
        x = self.act(x)

        # === Step 3. Per-degree output projection ===
        x = self.so3_linear_2(x)

        return x

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "EquivariantFFN",
            "@version": 1,
            "lmax": self.lmax,
            "channels": self.channels,
            "hidden_channels": self.hidden_channels,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "activation_function": self.activation_function,
            "so3_linear_1": self.so3_linear_1.serialize(),
            "so3_linear_2": self.so3_linear_2.serialize(),
            "act": self.act.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EquivariantFFN:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "EquivariantFFN":
            raise ValueError(f"Invalid class for EquivariantFFN: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported EquivariantFFN version: {version}")

        # === Extract sub-module data before creating instance ===
        so3_linear_1_data = data.pop("so3_linear_1")
        so3_linear_2_data = data.pop("so3_linear_2")
        act_data = data.pop("act")

        # === Convert precision to dtype ===
        precision = data.pop("precision")
        data["dtype"] = PRECISION_DICT[precision]

        # === Set activation_function default ===
        if "activation_function" not in data:
            data["activation_function"] = "silu"

        # === Set required args ===
        data["trainable"] = True
        data["seed"] = None

        obj = cls(**data)

        # === Restore sub-modules ===
        obj.so3_linear_1 = SO3Linear.deserialize(so3_linear_1_data)
        obj.so3_linear_2 = SO3Linear.deserialize(so3_linear_2_data)
        obj.act = GatedActivation.deserialize(act_data)
        return obj


class SeZMInteractionBlock(nn.Module):
    """
    SeZM interaction block: pre-norm, SO(2) conv, gating, full equivariant FFN.

    The FFN operates on ALL degrees (l=0 to lmax), using a gated activation where
    scalar features (l=0) control the gating of higher-degree features (l>0).

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum SO(2) order (|m|) mixed inside SO(2) convolution.
    channels
        Number of channels per (l, m) coefficient.
    so2_norm
        If True, apply intermediate ReducedSeparableRMSNorm between SO(2) mixing layers.
        When False (default), no normalization is applied between layers.
    so2_layers
        Number of SO(2) mixing layers.
    n_atten_head
        Number of gated attention heads when aggregating messages in SO(2) convolution.
        0 means no attention is used.
    ffn_neurons
        Hidden layer sizes for FFN.
    activation_function
        Activation function for l=0 components.
    use_triton
        Whether to use Triton kernels for scatter operations.
    eps
        Small epsilon for numerical stability.
    dtype
        Parameter dtype.
    seed
        Random seed for weight initialization.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        channels: int,
        so2_norm: bool = False,
        so2_layers: int = 2,
        n_atten_head: int = 0,
        ffn_neurons: int = 128,
        activation_function: str,
        use_triton: bool = False,
        eps: float = 1e-7,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.channels = int(channels)
        self.so2_norm = bool(so2_norm)
        self.so2_layers = int(so2_layers)
        self.n_atten_head = int(n_atten_head)
        self.ffn_neurons = int(ffn_neurons)
        self.activation_function = activation_function
        self.use_triton = bool(use_triton)
        self.eps = float(eps)
        self.dtype = dtype
        self.precision = RESERVED_PRECISION_DICT[dtype]

        # === Step 0. Split deterministic seeds at the block top-level ===
        seed_so2_conv = child_seed(seed, 0)
        seed_ffn = child_seed(seed, 1)

        self.pre_so2_norm = SeparableRMSNorm(
            self.lmax,
            self.channels,
            eps=eps,
            dtype=dtype,
            trainable=trainable,
        )

        self.pre_ffn_norm = SeparableRMSNorm(
            self.lmax,
            self.channels,
            eps=eps,
            dtype=dtype,
            trainable=trainable,
        )

        self.so2_conv = SO2Convolution(
            lmax=self.lmax,
            mmax=self.mmax,
            channels=self.channels,
            so2_norm=self.so2_norm,
            so2_layers=self.so2_layers,
            n_atten_head=n_atten_head,
            use_triton=self.use_triton,
            eps=self.eps,
            dtype=dtype,
            seed=seed_so2_conv,
            trainable=trainable,
        )

        self.ffn = EquivariantFFN(
            lmax=self.lmax,
            channels=self.channels,
            hidden_channels=ffn_neurons,
            dtype=dtype,
            activation_function=activation_function,
            trainable=trainable,
            seed=seed_ffn,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_cache: EdgeFeatureCache,
        radial_feat: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Features with shape (N, D, C).
        edge_cache
            Edge cache.
        radial_feat
            Per-edge radial features with shape (E, lmax+1, C), or None if no edges.

        Returns
        -------
        torch.Tensor
            Updated features with shape (N, D, C).
        """
        x_res = x

        # === Step 1. Pre-Norm ===
        with nvtx_range("SeZMBlock/pre_norm"):
            x = self.pre_so2_norm(x)

        # === Step 2. SO(2) convolution ===
        with nvtx_range("SeZMBlock/so2_conv"):
            if edge_cache.src.numel() > 0 and radial_feat is not None:
                x = self.so2_conv(x, edge_cache, radial_feat)

        # === Step 2.5 Residual connection ===
        x = x + x_res
        x_res = x

        # === Step 3. Pre-Norm ===
        with nvtx_range("SeZMBlock/pre_ffn_norm"):
            x = self.pre_ffn_norm(x)

        # === Step 4. Nodewise Feed-Forward ===
        with nvtx_range("SeZMBlock/ffn"):
            x = self.ffn(x)

        # === Step 4.5 Residual connection ===
        x = x + x_res
        return x

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SeZMInteractionBlock",
            "@version": 1,
            "lmax": self.lmax,
            "mmax": self.mmax,
            "channels": self.channels,
            "so2_norm": self.so2_norm,
            "so2_layers": self.so2_layers,
            "n_atten_head": self.n_atten_head,
            "ffn_neurons": self.ffn_neurons,
            "activation_function": self.activation_function,
            "use_triton": self.use_triton,
            "eps": self.eps,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "pre_so2_norm": self.pre_so2_norm.serialize(),
            "pre_ffn_norm": self.pre_ffn_norm.serialize(),
            "so2_conv": self.so2_conv.serialize(),
            "ffn": self.ffn.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SeZMInteractionBlock:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SeZMInteractionBlock":
            raise ValueError(f"Invalid class for SeZMInteractionBlock: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported SeZMInteractionBlock version: {version}")

        pre_so2_norm_data = data.pop("pre_so2_norm")
        pre_ffn_norm_data = data.pop("pre_ffn_norm")
        so2_conv_data = data.pop("so2_conv")
        ffn_data = data.pop("ffn")

        precision = data.pop("precision")
        data["dtype"] = PRECISION_DICT[precision]
        data["trainable"] = True
        data["seed"] = None

        obj = cls(**data)

        obj.pre_so2_norm = SeparableRMSNorm.deserialize(pre_so2_norm_data)
        obj.pre_ffn_norm = SeparableRMSNorm.deserialize(pre_ffn_norm_data)
        obj.so2_conv = SO2Convolution.deserialize(so2_conv_data)
        obj.ffn = EquivariantFFN.deserialize(ffn_data)
        return obj
