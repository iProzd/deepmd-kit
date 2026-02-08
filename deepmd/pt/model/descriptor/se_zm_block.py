# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SeZM-Net interaction blocks for DeePMD-kit (PyTorch backend).

This module implements per-block message passing, SO(2)/SO(3) transforms, and
nonlinearities used by `DescrptSeZMNet`. Shared geometry, caches, and projection
helpers are provided by `se_zm_helper.py`.

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
    init_trunc_normal_fan_in_out,
    map_degree_idx,
    np_safe,
    nvtx_range,
    project_D_to_m,
    project_Dt_from_m,
    safe_numpy_to_tensor,
)


class GatedActivation(nn.Module):
    """
    Gated activation for SO(3) equivariant features with per-l independent gates.

    Standard mode (gate=None in forward):
        - l=0: Uses the specified activation function
        - l>0: Each degree l has an independent gate derived from the l=0 scalar features.
               The gate for each l is expanded to all m components within that l-block.

    GLU mode (gate provided in forward, e.g., from split linear output):
        - l=0: x0 * act(g0) (SwiGLU-style when act=silu, GeGLU when act=gelu, etc.)
        - l>0: Uses gate's scalar (g0) to generate sigmoid gates for x's vector components.
               This preserves SO(3) equivariance (scalar gates vector, not vector gates vector).

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
    mlp_bias
        Whether to use bias in the gate linear layer.
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
        mlp_bias: bool = True,
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
        self.mlp_bias = bool(mlp_bias)

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
                bias=self.mlp_bias,
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
            # Use small std (0.01) to keep gate logits near zero at init.
            gen_gate = get_generator(child_seed(seed, 1))
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

    def forward(
        self, x: torch.Tensor, gate: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Value features with shape (N, D, C) where D=(lmax+1)^2.
        gate
            Optional gate features with shape (N, D, C). When provided, enables GLU mode:
            - l=0: x0 * act(g0) (e.g., SwiGLU when act=silu)
            - l>0: sigmoid(Linear(g0)) gates x's vector components
            When None (default), uses standard mode where gates are derived from x itself.

        Returns
        -------
        torch.Tensor
            Gated features with shape (N, D, C).
        """
        # === Determine gate source ===
        # GLU mode: use external gate's scalar; Standard mode: use x's scalar
        gate_scalar_source = (  # (N, C)
            gate[:, 0, :] if gate is not None else x[:, 0, :]
        )

        # === Step 1. l=0 activation ===
        if gate is not None:
            # GLU mode: x0 * act(g0) (e.g., SwiGLU, GeGLU)
            x0 = x[:, 0:1, :] * self.scalar_act(gate[:, 0:1, :])  # (N, 1, C)
        else:
            # Standard mode: act(x0)
            x0 = self.scalar_act(x[:, 0:1, :])

        if self.lmax == 0:
            return x0

        # === Step 2. Generate per-l gates from scalar features ===
        # gate_scalar_source has shape (N, C)
        # gate_linear outputs (N, lmax * C)
        gating_scalars = torch.sigmoid(  # (N, lmax * C)
            self.gate_linear(gate_scalar_source)
        )

        # Reshape to (N, lmax, C) then expand to (N, D-1, C)
        gating_scalars = gating_scalars.reshape(  # (N, lmax, C)
            x.shape[0], self.lmax, self.channels
        )
        gates = gating_scalars.index_select(  # (N, D-1, C)
            dim=1, index=self.expand_index
        )

        # === Step 3. Apply gates to l>0 components ===
        out = x.new_empty(x.shape)  # (N, D, C)
        out[:, 0:1, :] = x0
        out[:, 1:, :] = x[:, 1:, :] * gates
        return out

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "GatedActivation",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "channels": self.channels,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "activation_function": self.scalar_act.activation,
                "mlp_bias": self.mlp_bias,
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
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
        config = data.pop("config")
        variables = data.pop("@variables")
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
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
    centering
        Whether to apply mean centering for l=0 features.
    eps
        Small epsilon for numerical stability.
    dtype
        Parameter and computation dtype. Caller should pass compute_dtype (fp32+)
        for numerical stability and handle input/output conversion at boundaries.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        lmax: int,
        channels: int,
        centering: bool = True,
        *,
        eps: float = 1e-7,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.centering = centering
        self.dtype = dtype
        self.device = env.DEVICE
        self.eps = float(eps)

        # === Step 1. Learnable Parameters ===
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
            Normalized features with shape (N, D, C), same dtype as input.
        """
        in_dtype = x.dtype
        x = x.to(dtype=self.dtype)
        x0 = x[:, :1, :]  # (N, 1, C)
        xt = x[:, 1:, :]  # (N, D-1, C)

        # === Step 1. l=0: Standard RMS Norm ===
        if self.centering:
            x0 = x0 - x0.mean(dim=-1, keepdim=True)
        inv_rms0 = torch.rsqrt(  # (N, 1, 1)
            x0.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        x0 = x0 * inv_rms0

        weight0 = self.weight[0].reshape(1, 1, -1)  # (1, 1, C)
        x0 = x0 * weight0
        if self.bias is not None:
            bias0 = self.bias.reshape(1, 1, -1)  # (1, 1, C)
            x0 = x0 + bias0

        if xt.numel() == 0:
            return x0.to(dtype=in_dtype)

        # === Step 2. l>0: Degree-Balanced RMS Norm ===
        # Fused weighted sum: einsum avoids allocating intermediate (N, D-1, C) tensor.
        # balance_weight already pre-fused with 1/(lmax * C).
        mean_variance = torch.einsum(  # (N,)
            "ndc,d->n", xt * xt, self.balance_weight
        )
        inv_rmst = torch.rsqrt(  # (N, 1, 1)
            mean_variance + self.eps
        ).reshape(-1, 1, 1)
        xt = xt * inv_rmst

        wt = torch.index_select(  # (D-1, C)
            self.weight, dim=0, index=self.expand_index
        )
        xt = xt * wt.unsqueeze(0)

        return torch.cat([x0, xt], dim=1).to(dtype=in_dtype)

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "SeparableRMSNorm",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "channels": self.channels,
                "centering": self.centering,
                "eps": self.eps,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "trainable": trainable,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
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
        config = data.pop("config")
        variables = data.pop("@variables")
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
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
        Parameter and computation dtype. Caller should pass compute_dtype (fp32+)
        for numerical stability.
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

        if degree_index_m.dtype != torch.long:
            degree_index_m = degree_index_m.to(dtype=torch.long)
        self.register_buffer("degree_index_m", degree_index_m, persistent=True)

        deg_ns = degree_index_m[1:]
        weights = torch.zeros(deg_ns.numel(), dtype=self.dtype, device=self.device)
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
            Normalized tensor with shape (E, D_m_trunc, C), same dtype as input.

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
        x = x.to(dtype=self.dtype)
        x0 = x[:, :1, :]  # (E, 1, C)
        xt = x[:, 1:, :]  # (E, D_m_trunc-1, C)

        if self.centering:
            x0 = x0 - x0.mean(dim=-1, keepdim=True)
        inv_rms0 = torch.rsqrt(  # (E, 1, 1)
            x0.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        x0 = x0 * inv_rms0

        if xt.numel() == 0:
            if self.affine and self.weight is not None:
                x0.mul_(self.weight[0].reshape(1, 1, -1))
                if self.centering and self.bias0 is not None:
                    x0 += self.bias0.reshape(1, 1, -1)
            return x0.to(dtype=in_dtype)

        mean_var = torch.einsum(  # (E,)
            "edc,d->e", xt * xt, self.balance_weight
        )
        inv_rmst = torch.rsqrt(mean_var + self.eps).reshape(  # (E, 1, 1)
            -1, 1, 1
        )
        xt = xt * inv_rmst

        if self.affine and self.weight is not None:
            w = torch.index_select(  # (D_m_trunc, C)
                self.weight, dim=0, index=self.degree_index_m
            )
            w0 = w[0].reshape(1, 1, -1)  # (1, 1, C)
            wt = w[1:].unsqueeze(0)  # (1, D_m_trunc-1, C)
            x0.mul_(w0)
            xt.mul_(wt)
            if self.centering and self.bias0 is not None:
                bias0 = self.bias0.reshape(1, 1, -1)  # (1, 1, C)
                x0 += bias0

        return torch.cat([x0, xt], dim=1).to(dtype=in_dtype)

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "ReducedSeparableRMSNorm",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "channels": self.channels,
                "affine": self.affine,
                "centering": self.centering,
                "eps": self.eps,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "trainable": trainable,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
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
        config = data.pop("config")
        variables = data.pop("@variables")
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
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
        Parameter and computation dtype. Caller should pass compute_dtype (fp32+)
        for numerical stability.
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
            Normalized tensor with the same shape as input, same dtype as input.
        """
        in_dtype = x.dtype
        x = x.to(dtype=self.dtype)

        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * inv_rms

        weight = self.weight  # (C,)
        x = x * weight
        return x.to(dtype=in_dtype)

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "ScalarRMSNorm",
            "@version": 1,
            "config": {
                "channels": self.channels,
                "eps": self.eps,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "trainable": trainable,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
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
        config = data.pop("config")
        variables = data.pop("@variables")
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
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
    mlp_bias
        Whether to use bias for l=0 (scalar) components.
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
        mlp_bias: bool = True,
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
        self.mlp_bias = bool(mlp_bias)

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
        if self.mlp_bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(self.out_channels, dtype=self.dtype, device=self.device)
            )
        else:
            self.bias = None

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
        if self.bias is not None:
            bias0 = self.bias.reshape(1, -1)  # (1, C_out)
            out[:, 0, :] = out[:, 0, :] + bias0

        return out

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "SO3Linear",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "mlp_bias": self.mlp_bias,
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
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
        config = data.pop("config")
        variables = data.pop("@variables")
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
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

    Note
    ----
    A single block-diagonal matmul is used for all |m|>0 groups to reduce kernel
    launches and CPU overhead. Each block enforces the SO(2) coupling:
    [W_u, -W_v; W_v, W_u] built from the per-m linear weights.

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
    mlp_bias
        Whether to use bias for l=0 (scalar) components.
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
        mlp_bias: bool = True,
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
        self.mlp_bias = bool(mlp_bias)

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
        # Each entry: (neg_start, pos_start, num_l) for a fixed |m|.
        # These ranges are contiguous in m-major layout.
        m_ranges: list[tuple[int, int, int]] = []

        offset = m0_size
        for m in range(1, self.mmax + 1):
            num_l = self.lmax - m + 1
            neg_start = offset
            pos_start = offset + num_l
            neg_idx = torch.arange(
                neg_start, neg_start + num_l, device=self.device, dtype=torch.long
            )
            pos_idx = torch.arange(
                pos_start, pos_start + num_l, device=self.device, dtype=torch.long
            )
            neg_indices_list.append(neg_idx)
            pos_indices_list.append(pos_idx)
            m_ranges.append((neg_start, pos_start, num_l))
            offset += 2 * num_l

        self.reduced_dim = int(offset)

        if len(pos_indices_list) > 0:
            self.register_buffer(
                "pos_indices", torch.cat(pos_indices_list), persistent=True
            )
            self.register_buffer(
                "neg_indices", torch.cat(neg_indices_list), persistent=True
            )
            self._m_ranges = m_ranges
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
            self._m_ranges = []

        # === Step 2. Mixing per |m| group, cross-l allowed, bias only for scalar index ===
        num_m0 = self.lmax + 1
        num_in_m0 = num_m0 * self.in_channels
        num_out_m0 = num_m0 * self.out_channels
        self.weight_m0 = nn.Parameter(
            torch.empty(num_in_m0, num_out_m0, device=self.device, dtype=self.dtype)
        )
        init_trunc_normal_fan_in_out(self.weight_m0, child_seed(seed, 0))
        if self.mlp_bias:
            self.bias0: nn.Parameter | None = nn.Parameter(
                torch.zeros(self.out_channels, device=self.device, dtype=self.dtype)
            )
        else:
            self.bias0 = None

        # For |m|>0, SO(2) equivariance requires 2x2 block structure on (Re, Im) pairs.
        # Output dimension is doubled for complex mixing: (a, -b; b, a) constraint.
        self.weight_m: nn.ParameterList = nn.ParameterList()
        for m in range(1, self.mmax + 1):
            num_l = self.lmax - m + 1
            num_in = num_l * self.in_channels
            num_out = 2 * num_l * self.out_channels
            weight = nn.Parameter(
                torch.empty(num_in, num_out, device=self.device, dtype=self.dtype)
            )
            init_trunc_normal_fan_in_out(weight, child_seed(seed, m))
            # Apply scaling for SO(2) equivariance
            weight.data.mul_(1.0 / math.sqrt(2.0))
            self.weight_m.append(weight)

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input with shape (N, D_m_trunc, Cin), where D_m_trunc is the
            coefficient dimension of the m-major layout truncated by `mmax`.

        Returns
        -------
        torch.Tensor
            Output with shape (N, D_m_trunc, Cout), where Cout is output channels.
        """
        # === Step 1. Flatten input in m-major order ===
        # Layout: [m=0 (l=0..lmax), m=1 neg, m=1 pos, m=2 neg, m=2 pos, ...],
        # each coefficient contributes a Cin-sized block in the flattened axis.
        n_atom, D_m_trunc, Cin = x.shape
        x_flat = x.reshape(n_atom, -1)  # (N, D_m_trunc * Cin)

        # === Step 2. Build block-diagonal weight (m=0 + all |m|>0 groups) ===
        # m=0: unconstrained linear over (lmax+1) coefficients.
        # |m|>0: enforce SO(2) coupling using [W_u, -W_v; W_v, W_u] per m.
        weight_blocks: list[torch.Tensor] = [self.weight_m0.t()]
        for w in self.weight_m:
            out_half = w.shape[1] // 2
            w_u = w[:, :out_half]
            w_v = w[:, out_half:]
            w_u_t = w_u.t()
            w_v_t = w_v.t()
            top = torch.cat([w_u_t, -w_v_t], dim=1)
            bottom = torch.cat([w_v_t, w_u_t], dim=1)
            weight_blocks.append(torch.cat([top, bottom], dim=0))
        weight = torch.block_diag(*weight_blocks)

        # === Step 3. Single matmul + reshape ===
        out_flat = torch.matmul(x_flat, weight.t())  # (N, D_m_trunc * Cout)
        out = out_flat.reshape(  # (N, D_m_trunc, Cout)
            n_atom, D_m_trunc, self.out_channels
        )

        # === Step 4. Bias on l=0 scalar index ===
        if self.bias0 is not None:
            out[:, 0, :].add_(self.bias0)
        return out

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "SO2Linear",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "mlp_bias": self.mlp_bias,
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
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
        config = data.pop("config")
        variables = data.pop("@variables")
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
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
    mlp_bias
        Whether to use bias in SO2Linear (l=0 bias), GatedActivation (gate linear bias),
        and ReducedSeparableRMSNorm (centering bias).
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
        mlp_bias: bool = True,
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
        self.mlp_bias = bool(mlp_bias)
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
                    mlp_bias=self.mlp_bias,
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
                        centering=self.mlp_bias,
                        eps=self.eps,
                        dtype=self.compute_dtype,
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
                    mlp_bias=self.mlp_bias,
                    trainable=trainable,
                    seed=child_seed(seed_non_linearities, i),
                )
            )
        non_linearities.append(nn.Identity())
        self.non_linearities = nn.ModuleList(non_linearities)

        # === Step 6. Optional head-wise gating components ===
        # Edge gate: normalized dst/msg logits, radial stays raw to preserve geometry.
        self.norm_dst_for_gate: ScalarRMSNorm | None = None
        self.norm_msg_for_gate: ScalarRMSNorm | None = None
        self.proj_dst: MLPLayer | None = None
        self.proj_rad: MLPLayer | None = None
        self.proj_msg: MLPLayer | None = None
        self.gate_tau_log: nn.Parameter | None = None
        self.msg_gate_alpha_log: nn.Parameter | None = None
        if self.n_atten_head > 0:
            self.norm_dst_for_gate = ScalarRMSNorm(
                channels=self.channels,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
            self.norm_msg_for_gate = ScalarRMSNorm(
                channels=self.channels,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
            # Edge gate projections: dst/msg scalars normalized, radial scalars raw
            self.proj_dst = MLPLayer(
                self.channels,
                self.n_atten_head,
                activation_function=None,
                bias=self.mlp_bias,
                precision=self.precision,
                seed=child_seed(seed_gate, 0),
                trainable=trainable,
            )
            self.proj_rad = MLPLayer(
                self.channels,
                self.n_atten_head,
                activation_function=None,
                bias=self.mlp_bias,
                precision=self.precision,
                seed=child_seed(seed_gate, 1),
                trainable=trainable,
            )
            self.proj_msg = MLPLayer(
                self.channels,
                self.n_atten_head,
                activation_function=None,
                bias=self.mlp_bias,
                precision=self.precision,
                seed=child_seed(seed_gate, 4),
                trainable=trainable,
            )
            # Initialization: Normal(0, 0.01) for weights, zeros for bias
            # Small std keeps gate logits near zero at init => sigmoid(0) ≈ 0.5
            gen_proj_dst = get_generator(child_seed(seed_gate, 10))
            gen_proj_rad = get_generator(child_seed(seed_gate, 11))
            gen_proj_msg = get_generator(child_seed(seed_gate, 12))
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
            self.gate_tau_log = nn.Parameter(
                torch.zeros(
                    self.n_atten_head,
                    dtype=self.compute_dtype,
                    device=self.device,
                )
            )
            gen_alpha = get_generator(child_seed(seed_gate, 14))
            alpha_mean = math.log(0.01)
            alpha_std = 0.05 if self.n_atten_head > 1 else 0.0
            alpha_log = torch.empty(
                (self.n_atten_head,),
                device=self.device,
                dtype=self.compute_dtype,
            )
            if alpha_std == 0.0:
                alpha_log.fill_(alpha_mean)
            else:
                nn.init.trunc_normal_(
                    alpha_log,
                    mean=alpha_mean,
                    std=alpha_std,
                    a=alpha_mean - 3.0 * alpha_std,
                    b=alpha_mean + 3.0 * alpha_std,
                    generator=gen_alpha,
                )
            self.msg_gate_alpha_log = nn.Parameter(alpha_log)
        else:
            self.gate_tau_log = None
            self.msg_gate_alpha_log = None

        # === Step 7. Final SO3Linear to mix channels ===
        self.so3_linear = SO3Linear(
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=self.channels,
            dtype=dtype,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3,
        )
        nn.init.zeros_(self.so3_linear.weight)
        if self.so3_linear.bias is not None:
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
        src, dst = edge_cache.src, edge_cache.dst
        x_src = x[src]  # (E, D, C)

        # === Step 1. Rotate to edge-aligned local frame ===
        with nvtx_range("SO2Conv/rotate_to_local"):
            D_full = edge_cache.D_full
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
            rad_feat = radial_feat[  # (E, D_m_trunc, C)
                :, self.degree_index_m, :
            ]
            x_local.mul_(rad_feat)

        # === Step 3. Multi-layer SO(2) mixing ===
        with nvtx_range("SO2Conv/so2_layers"):
            for layer_idx, (so2_linear, inter_norm, non_linear) in enumerate(
                zip(self.so2_linears, self.so2_inter_norms, self.non_linearities)
            ):
                x_local = so2_linear(x_local)

                if layer_idx == 0 and so2_linear.bias0 is not None:
                    bias_correction = so2_linear.bias0 * (  # (E, C)
                        rad_feat[:, 0, :] * edge_cache.edge_env - 1.0
                    )
                    x_local[:, 0, :].add_(bias_correction)

                x_local = inter_norm(x_local)
                x_local = non_linear(x_local)

        # === Step 4. Rotate back to global frame ===
        with nvtx_range("SO2Conv/rotate_back"):
            Dt_full = edge_cache.Dt_full
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
                x_message = x_message * edge_cache.edge_env.unsqueeze(-1)
                out = x.new_zeros(x.shape).to(dtype=self.compute_dtype)
                out.index_add_(0, dst, x_message.to(dtype=self.compute_dtype))
                out = out.to(dtype=self.dtype)
                out.mul_(edge_cache.inv_sqrt_deg)
            else:
                # === Step 5.1. Extract scalar features for gating ===
                x_l0 = x[:, 0, :]  # (N, C)
                radial_l0 = radial_feat[:, 0, :]  # (E, C)
                msg_l0 = x_message[:, 0, :]  # (E, C)

                # === Step 5.2. Compute edge gate (fp32+ logits path) ===
                # Edge gate logits: dst/msg normalized inputs, radial stays raw
                compute_dtype = self.compute_dtype
                dst_logits = self.proj_dst(  # (N, H)
                    self.norm_dst_for_gate(x_l0)
                ).to(dtype=compute_dtype)
                radial_logits = self.proj_rad(radial_l0).to(  # (E, H)
                    dtype=compute_dtype
                )
                msg_logits = self.proj_msg(  # (E, H)
                    self.norm_msg_for_gate(msg_l0)
                ).to(dtype=compute_dtype)
                msg_alpha = torch.exp(  # (1, H)
                    self.msg_gate_alpha_log
                ).view(1, -1)
                edge_logits = (  # (E, H)
                    dst_logits.index_select(0, dst)
                    + radial_logits
                    + msg_alpha * msg_logits
                )
                tau = torch.exp(self.gate_tau_log).view(1, -1)  # (1, H)
                edge_gate = torch.sigmoid(edge_logits / tau)  # (E, H)
                edge_weight = (  # (E, H)
                    edge_cache.edge_env.to(dtype=compute_dtype) * edge_gate
                )

                # === Step 5.3. Head-wise scatter aggregation (fp32+ accumulation) ===
                # Mixed-precision strategy: V stays in its native dtype (bf16/fp32/fp64),
                E = x_message.shape[0]
                V = x_message.reshape(  # (E, D, H, head_dim)
                    E, self.ebed_dim_full, self.n_atten_head, self.head_dim
                )

                # Multiply in V's dtype (cheap), then cast contribution for stable accumulation
                edge_weight_4d = edge_weight.reshape(  # (E, 1, H, 1)
                    E, 1, self.n_atten_head, 1
                )
                msg = V * edge_weight_4d  # (E, D, H, head_dim)
                msg_acc = msg.to(  # (E, D, H, head_dim)
                    dtype=compute_dtype
                )
                out_heads = torch.zeros(  # (N, D, H, head_dim)
                    x.shape[0],
                    self.ebed_dim_full,
                    self.n_atten_head,
                    self.head_dim,
                    device=x.device,
                    dtype=compute_dtype,
                )
                out_heads.index_add_(0, dst, msg_acc)

                # === Step 5.4. Apply degree normalization ===
                out = out_heads.reshape(  # (N, D, C)
                    x.shape[0], self.ebed_dim_full, self.channels
                )
                out.mul_(edge_cache.inv_sqrt_deg.to(dtype=compute_dtype))
                out = out.to(dtype=self.dtype)

        # === Step 6. Final channel mixing ===
        with nvtx_range("SO2Conv/so3_linear"):
            out = self.so3_linear(out)  # (N, D, C)
        return out

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "SO2Convolution",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "channels": self.channels,
                "so2_norm": self.so2_norm,
                "so2_layers": self.so2_layers,
                "n_atten_head": self.n_atten_head,
                "mlp_bias": self.mlp_bias,
                "eps": self.eps,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
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
        config = data.pop("config")
        variables = data.pop("@variables")
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
        return obj


class EquivariantFFN(nn.Module):
    """
    Full equivariant FFN operating on all spherical harmonic degrees.

    Structure (glu_activation=False):
        SO3 linear (in -> hidden) -> GatedActivation -> SO3 linear (hidden -> out)

    Structure (glu_activation=True):
        SO3 linear (in -> 2*hidden) -> split -> GatedActivation(val, gate) -> SO3 linear (hidden -> out)

    GatedActivation serves as the unified "activation" for equivariant networks,
    analogous to SiLU in standard MLPs, but respecting SO(3) equivariance:
    - l=0: Uses the specified activation function (or GLU variant when glu_activation=True)
    - l>0: sigmoid gate from l=0 scalar features

    When glu_activation=True, the first linear outputs 2*hidden_channels, then splits into
    value and gate branches. This transforms activations like silu->swiglu, gelu->geglu.
    The split approach is more efficient than two separate linear layers.

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
    glu_activation
        If True, use GLU-style gating (e.g., silu -> swiglu, gelu -> geglu).
    mlp_bias
        Whether to use bias in SO3Linear (l=0 bias) and GatedActivation (gate linear bias).
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
        glu_activation: bool = True,
        mlp_bias: bool = True,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.activation_function = activation_function
        self.glu_activation = bool(glu_activation)
        self.mlp_bias = bool(mlp_bias)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        # === Step 0. Split deterministic seeds at the module top-level ===
        seed_so3_in = child_seed(seed, 0)
        seed_act = child_seed(seed, 1)
        seed_so3_out = child_seed(seed, 2)

        # === First SO3Linear for channel mixing ===
        # When glu_activation=True, output 2*hidden_channels for split
        linear1_out_channels = (
            2 * self.hidden_channels if self.glu_activation else self.hidden_channels
        )
        self.so3_linear_1 = SO3Linear(
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=linear1_out_channels,
            dtype=dtype,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3_in,
        )

        # === Equivariant activation ===
        self.act = GatedActivation(
            lmax=self.lmax,
            channels=self.hidden_channels,
            dtype=dtype,
            activation_function=activation_function,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_act,
        )

        # === Second SO3Linear for channel mixing ===
        self.so3_linear_2 = SO3Linear(
            lmax=self.lmax,
            in_channels=self.hidden_channels,
            out_channels=self.channels,
            dtype=dtype,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3_out,
        )
        nn.init.zeros_(self.so3_linear_2.weight)
        if self.so3_linear_2.bias is not None:
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

        # === Step 2. Equivariant activation (with optional GLU) ===
        if self.glu_activation:
            # Split into value and gate branches along channel dimension
            x_val, x_gate = x.chunk(2, dim=-1)
            # Pass gate to GatedActivation for GLU-style gating
            x = self.act(x_val, gate=x_gate)
        else:
            x = self.act(x)

        # === Step 3. Per-degree output projection ===
        x = self.so3_linear_2(x)

        return x

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "EquivariantFFN",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "channels": self.channels,
                "hidden_channels": self.hidden_channels,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "activation_function": self.activation_function,
                "glu_activation": self.glu_activation,
                "mlp_bias": self.mlp_bias,
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
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
        config = data.pop("config")
        variables = data.pop("@variables")
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
        return obj


class SeZMInteractionBlock(nn.Module):
    """
    SeZM interaction block: pre/post-norm, SO(2) conv, full equivariant FFN.

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
    so2_pre_norm
        If True, apply pre-norm before SO(2) convolution.
    so2_post_norm
        If True, apply post-norm on SO(2) output before the residual add.
    ffn_pre_norm
        If True, apply pre-norm before FFN.
    ffn_post_norm
        If True, apply post-norm on FFN output before the residual add.
    ffn_neurons
        Hidden layer sizes for FFN.
    activation_function
        Activation function for l=0 components.
    glu_activation
        If True, use GLU-style gating in FFN (e.g., silu -> swiglu, gelu -> geglu).
    mlp_bias
        Whether to use bias in equivariant layers. Controls:
        - SO3Linear: l=0 bias
        - SO2Linear: l=0 bias
        - GatedActivation: gate linear bias
        - SeparableRMSNorm: centering bias
        - ReducedSeparableRMSNorm: centering bias
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
        so2_pre_norm: bool = True,
        so2_post_norm: bool = False,
        ffn_pre_norm: bool = True,
        ffn_post_norm: bool = False,
        ffn_neurons: int = 96,
        activation_function: str,
        glu_activation: bool = True,
        mlp_bias: bool = True,
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
        self.so2_pre_norm = bool(so2_pre_norm)
        self.so2_post_norm = bool(so2_post_norm)
        self.ffn_pre_norm = bool(ffn_pre_norm)
        self.ffn_post_norm = bool(ffn_post_norm)
        self.ffn_neurons = int(ffn_neurons)
        self.activation_function = activation_function
        self.glu_activation = bool(glu_activation)
        self.mlp_bias = bool(mlp_bias)
        self.eps = float(eps)
        self.dtype = dtype
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.compute_dtype = get_promoted_dtype(self.dtype)

        # === Step 0. Split deterministic seeds at the block top-level ===
        seed_so2_conv = child_seed(seed, 0)
        seed_ffn = child_seed(seed, 1)

        if self.so2_pre_norm:
            self.pre_so2_norm: nn.Module = SeparableRMSNorm(
                self.lmax,
                self.channels,
                centering=self.mlp_bias,
                eps=eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
        else:
            self.pre_so2_norm = nn.Identity()

        if self.so2_post_norm:
            self.post_so2_norm: nn.Module = SeparableRMSNorm(
                self.lmax,
                self.channels,
                centering=self.mlp_bias,
                eps=eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
        else:
            self.post_so2_norm = nn.Identity()

        if self.ffn_pre_norm:
            self.pre_ffn_norm: nn.Module = SeparableRMSNorm(
                self.lmax,
                self.channels,
                centering=self.mlp_bias,
                eps=eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
        else:
            self.pre_ffn_norm = nn.Identity()

        if self.ffn_post_norm:
            self.post_ffn_norm: nn.Module = SeparableRMSNorm(
                self.lmax,
                self.channels,
                centering=self.mlp_bias,
                eps=eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
        else:
            self.post_ffn_norm = nn.Identity()

        self.so2_conv = SO2Convolution(
            lmax=self.lmax,
            mmax=self.mmax,
            channels=self.channels,
            so2_norm=self.so2_norm,
            so2_layers=self.so2_layers,
            n_atten_head=n_atten_head,
            mlp_bias=self.mlp_bias,
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
            glu_activation=self.glu_activation,
            mlp_bias=self.mlp_bias,
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
            Per-edge radial features with shape (E, lmax+1, C).

        Returns
        -------
        torch.Tensor
            Updated features with shape (N, D, C).
        """
        x_res = x  # (N, D, C)

        # === Step 1. Pre-Norm (SO2) ===
        with nvtx_range("SeZMBlock/pre_norm"):
            x_pre = self.pre_so2_norm(x)  # (N, D, C)

        # === Step 2. SO(2) convolution ===
        with nvtx_range("SeZMBlock/so2_conv"):
            y = self.so2_conv(x_pre, edge_cache, radial_feat)  # (N, D, C)

        # === Step 3. Post-Norm (SO2) ===
        with nvtx_range("SeZMBlock/post_so2_norm"):
            y = self.post_so2_norm(y)

        # === Step 4. Residual connection (SO2) ===
        x = x_res + y
        x_res = x

        # === Step 5. Pre-Norm (FFN) ===
        with nvtx_range("SeZMBlock/pre_ffn_norm"):
            x_pre = self.pre_ffn_norm(x)

        # === Step 6. Nodewise Feed-Forward ===
        with nvtx_range("SeZMBlock/ffn"):
            y = self.ffn(x_pre)

        # === Step 7. Post-Norm (FFN) ===
        with nvtx_range("SeZMBlock/post_ffn_norm"):
            y = self.post_ffn_norm(y)

        # === Step 8. Residual connection (FFN) ===
        x = x_res + y
        return x

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "SeZMInteractionBlock",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "channels": self.channels,
                "so2_norm": self.so2_norm,
                "so2_layers": self.so2_layers,
                "n_atten_head": self.n_atten_head,
                "so2_pre_norm": self.so2_pre_norm,
                "so2_post_norm": self.so2_post_norm,
                "ffn_pre_norm": self.ffn_pre_norm,
                "ffn_post_norm": self.ffn_post_norm,
                "ffn_neurons": self.ffn_neurons,
                "activation_function": self.activation_function,
                "glu_activation": self.glu_activation,
                "mlp_bias": self.mlp_bias,
                "eps": self.eps,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
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
        config = data.pop("config")
        variables = data.pop("@variables")
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
        return obj
