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
- Node-level equivariant operators use `(N, D, F, Cf)` convention.
- Edge-level SO(2) internal operators use m-major reduced
  `(E, F, D_m_trunc, Cf)` convention.
- `SO2Convolution` includes both `pre_focus_mix` and `post_focus_mix`, so
  channel projection around the SO(2) stack is encapsulated in one module.
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
    segment_softmax_with_env_weight,
)


class FocusLinear(nn.Module):
    """
    Per-focus linear projection on the last feature axis.

    Notes
    -----
    Parameters are stored in (in, out) convention to match Muon's rectangular
    correction assumption (rows=fan_in, cols=fan_out):
    - weight: (in_channels, n_focus * out_channels)
    - bias: (n_focus * out_channels,)

    Parameters
    ----------
    in_channels
        Input feature dimension.
    out_channels
        Output feature dimension.
    n_focus
        Number of focus streams.
    dtype
        Parameter dtype.
    bias
        Whether to use bias.
    trainable
        Whether parameters are trainable.
    seed
        Random seed for initialization.
    init_std
        If given, use normal(0, init_std) instead of default uniform init.
        Useful for gate projections where small initial logits are desired.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        n_focus: int,
        dtype: torch.dtype,
        bias: bool = True,
        trainable: bool,
        seed: int | list[int] | None = None,
        init_std: float | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.dtype = dtype
        self.device = env.DEVICE
        self.use_bias = bool(bias)
        self.weight = nn.Parameter(
            torch.empty(
                self.in_channels,
                self.n_focus * self.out_channels,
                device=self.device,
                dtype=self.dtype,
            )
        )
        gen = get_generator(seed)
        if init_std is not None:
            nn.init.normal_(self.weight, mean=0.0, std=init_std, generator=gen)
        else:
            bound = 1.0 / math.sqrt(self.in_channels)
            nn.init.uniform_(self.weight, -bound, bound, generator=gen)
        if self.use_bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(
                    self.n_focus * self.out_channels,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        else:
            self.bias = None
        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor with shape (B, F, Cin).

        Returns
        -------
        torch.Tensor
            Projected tensor with shape (B, F, Cout).
        """
        weight = self.weight.view(self.in_channels, self.n_focus, self.out_channels)
        out = torch.einsum("bfi,ifo->bfo", x, weight)
        if self.use_bias:
            bias = self.bias.view(self.n_focus, self.out_channels)
            out = out + bias.unsqueeze(0)
        return out


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
        Number of channels per focus stream.
    n_focus
        Number of focus streams.
    dtype
        Parameter dtype.
    activation_function
        Activation function for l=0 components (e.g., "silu", "tanh", "gelu").
    mlp_bias
        Whether to use bias in the gate linear layer.
    layout
        Tensor layout convention. ``"nfdc"`` means input shape (N, F, D, C);
        ``"ndfc"`` means input shape (N, D, F, C).
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
        n_focus: int = 1,
        dtype: torch.dtype,
        activation_function: str = "silu",
        mlp_bias: bool = True,
        layout: str = "nfdc",
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
        self.n_focus = int(n_focus)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.mlp_bias = bool(mlp_bias)
        self.layout = str(layout).lower()
        if self.layout not in {"nfdc", "ndfc"}:
            raise ValueError("`layout` must be either 'nfdc' or 'ndfc'")

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
            self.gate_linear: nn.Module = FocusLinear(
                in_channels=self.channels,
                out_channels=self.lmax * self.channels,
                n_focus=self.n_focus,
                dtype=self.dtype,
                bias=self.mlp_bias,
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
                self.gate_linear.weight, mean=0.0, std=0.01, generator=gen_gate
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
            Value features. Shape is (N, F, D, C) when ``layout='nfdc'``,
            or (N, D, F, C) when ``layout='ndfc'``.
        gate
            Optional gate features with the same layout as ``x``.
            When provided, enables GLU mode:
            - l=0: x0 * act(g0) (e.g., SwiGLU when act=silu)
            - l>0: sigmoid(Linear(g0)) gates x's vector components
            When None (default), uses standard mode where gates are derived from x itself.

        Returns
        -------
        torch.Tensor
            Gated features with the same layout as ``x``.
        """
        if self.layout == "nfdc":
            focus_axis = 1
            degree_axis = 2
        else:
            focus_axis = 2
            degree_axis = 1

        # === Determine gate source ===
        # GLU mode: use external gate's scalar; Standard mode: use x's scalar
        if gate is not None:
            gate_scalar_source = gate.select(dim=degree_axis, index=0)  # (N, F, C)
        else:
            gate_scalar_source = x.select(dim=degree_axis, index=0)  # (N, F, C)

        # === Step 1. l=0 activation ===
        if gate is not None:
            # GLU mode: x0 * act(g0) (e.g., SwiGLU, GeGLU)
            x0 = x.narrow(degree_axis, 0, 1) * self.scalar_act(
                gate.narrow(degree_axis, 0, 1)
            )
        else:
            # Standard mode: act(x0)
            x0 = self.scalar_act(x.narrow(degree_axis, 0, 1))

        if self.lmax == 0:
            return x0

        # === Step 2. Generate per-l gates from scalar features ===
        # gate_scalar_source has shape (N, F, C)
        # gate_linear outputs (N, F, lmax * C)
        gating_scalars = torch.sigmoid(self.gate_linear(gate_scalar_source))

        # Reshape to (N, F, lmax, C) then expand to (N, F, D-1, C)
        gating_scalars = gating_scalars.reshape(
            x.shape[0], gate_scalar_source.shape[1], self.lmax, self.channels
        )
        gates = gating_scalars.index_select(dim=2, index=self.expand_index)
        if self.layout == "ndfc":
            gates = gates.transpose(1, 2)  # (N, D-1, F, C)

        # === Step 3. Apply gates to l>0 components ===
        out = x.new_empty(x.shape)
        out.narrow(degree_axis, 0, 1).copy_(x0)
        out.narrow(degree_axis, 1, x.shape[degree_axis] - 1).copy_(
            x.narrow(degree_axis, 1, x.shape[degree_axis] - 1) * gates
        )
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
                "n_focus": self.n_focus,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "activation_function": self.scalar_act.activation,
                "mlp_bias": self.mlp_bias,
                "layout": self.layout,
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
        Channels per (l, m) coefficient in each focus stream.
    n_focus
        Number of focus streams. Norm affine parameters are independent per focus.
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
        n_focus: int = 1,
        centering: bool = True,
        *,
        eps: float = 1e-7,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.centering = centering
        self.dtype = dtype
        self.device = env.DEVICE
        self.eps = float(eps)

        # === Step 1. Learnable Parameters ===
        # Per-focus, per-l affine scales with shape (F, lmax+1, C)
        # adam_ prefix routes this to Adam (no weight decay) in HybridMuon.
        self.adam_scale = nn.Parameter(
            torch.ones(
                self.n_focus,
                self.lmax + 1,
                self.channels,
                dtype=self.dtype,
                device=self.device,
            )
        )
        if self.centering:
            # Bias only for l=0, independent per focus.
            self.bias = nn.Parameter(
                torch.zeros(
                    self.n_focus, self.channels, dtype=self.dtype, device=self.device
                )
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
            #   mean_variance = einsum('ndfc,d->nf', x^2, balance_weight)
            # avoids allocating the intermediate (N, D-1, F, C) tensor.
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
            Features with shape (N, D, F, C) where D=(lmax+1)^2.

        Returns
        -------
        torch.Tensor
            Normalized features with shape (N, D, F, C), same dtype as input.
        """
        in_dtype = x.dtype
        x = x.to(dtype=self.dtype)
        x0 = x[:, :1, :, :]  # (N, 1, F, C)
        xt = x[:, 1:, :, :]  # (N, D-1, F, C)

        # === Step 1. l=0: Standard RMS Norm ===
        if self.centering:
            x0 = x0 - x0.mean(dim=-1, keepdim=True)
        inv_rms0 = torch.rsqrt(  # (N, 1, F, 1)
            x0.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        x0 = x0 * inv_rms0

        scale0 = self.adam_scale[:, 0, :].reshape(
            1, 1, self.n_focus, -1
        )  # (1, 1, F, C)
        x0 = x0 * scale0
        if self.centering:
            bias0 = self.bias.reshape(1, 1, self.n_focus, -1)  # (1, 1, F, C)
            x0 = x0 + bias0

        if xt.numel() == 0:
            return x0.to(dtype=in_dtype)

        # === Step 2. l>0: Degree-Balanced RMS Norm ===
        # Fused weighted sum: einsum avoids allocating intermediate (N, D-1, F, C) tensor.
        # balance_weight already pre-fused with 1/(lmax * C).
        mean_variance = torch.einsum(  # (N, F)
            "ndfc,d->nf", xt * xt, self.balance_weight
        )
        inv_rmst = torch.rsqrt(mean_variance + self.eps).unsqueeze(1).unsqueeze(-1)
        xt = xt * inv_rmst

        wt = torch.index_select(  # (F, D-1, C)
            self.adam_scale, dim=1, index=self.expand_index
        )
        xt = xt * wt.permute(1, 0, 2).reshape(1, wt.shape[1], self.n_focus, wt.shape[2])

        out = torch.cat([x0, xt], dim=1).to(dtype=in_dtype)
        return out

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "SeparableRMSNorm",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "channels": self.channels,
                "n_focus": self.n_focus,
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
    n_focus
        Number of focus streams.
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
        n_focus: int = 1,
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
        self.n_focus = int(n_focus)
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
            # adam_ prefix routes this to Adam (no weight decay) in HybridMuon.
            self.adam_scale = nn.Parameter(
                torch.ones(
                    self.n_focus,
                    self.lmax + 1,
                    self.channels,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            self.bias0 = (
                nn.Parameter(
                    torch.zeros(
                        self.n_focus,
                        self.channels,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
                if self.centering
                else None
            )
        else:
            self.register_parameter("adam_scale", None)
            self.register_parameter("bias0", None)

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor with shape (E, F, D_m_trunc, C).

        Returns
        -------
        torch.Tensor
            Normalized tensor with shape (E, F, D_m_trunc, C), same dtype as input.
        """
        in_dtype = x.dtype
        x = x.to(dtype=self.dtype)
        x0 = x[:, :, :1, :]  # (E, F, 1, C)
        xt = x[:, :, 1:, :]  # (E, F, D_m_trunc-1, C)

        if self.centering:
            x0 = x0 - x0.mean(dim=-1, keepdim=True)
        inv_rms0 = torch.rsqrt(  # (E, 1, 1)
            x0.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        x0 = x0 * inv_rms0

        if xt.numel() == 0:
            if self.affine:
                x0.mul_(self.adam_scale[:, 0, :].reshape(1, self.n_focus, 1, -1))
                if self.centering:
                    x0 += self.bias0.reshape(1, self.n_focus, 1, -1)
            return x0.to(dtype=in_dtype)

        mean_var = torch.einsum(  # (E, F)
            "efdc,d->ef", xt * xt, self.balance_weight
        )
        inv_rmst = torch.rsqrt(mean_var + self.eps).unsqueeze(-1).unsqueeze(-1)
        xt = xt * inv_rmst

        if self.affine:
            w = torch.index_select(  # (D_m_trunc, C)
                self.adam_scale, dim=1, index=self.degree_index_m
            )
            w0 = w[:, 0, :].reshape(1, self.n_focus, 1, -1)  # (1, F, 1, C)
            wt = w[:, 1:, :].reshape(
                1, self.n_focus, w.shape[1] - 1, w.shape[2]
            )  # (1, F, D_m_trunc-1, C)
            x0.mul_(w0)
            xt.mul_(wt)
            if self.centering:
                bias0 = self.bias0.reshape(1, self.n_focus, 1, -1)  # (1, F, 1, C)
                x0 += bias0

        out = torch.cat([x0, xt], dim=2).to(dtype=in_dtype)
        return out

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
                "n_focus": self.n_focus,
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
    Lightweight per-focus RMSNorm for scalar branches.

    This is the unified scalar norm used by SeZM:
    - `n_focus=1` naturally degenerates to the single-stream behavior.
    - `n_focus>1` uses independent learnable scales per focus stream.
    Bias is intentionally omitted to keep the gate paths minimal.

    Parameters
    ----------
    channels
        Feature dimension of the last axis.
    n_focus
        Number of focus streams.
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
        n_focus: int = 1,
        eps: float = 1e-7,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.dtype = dtype
        self.device = env.DEVICE
        self.eps = float(eps)

        # adam_ prefix routes this to Adam (no weight decay) in HybridMuon.
        self.adam_scale = nn.Parameter(
            torch.ones(
                self.n_focus,
                self.channels,
                dtype=self.dtype,
                device=self.device,
            )
        )

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor with shape (B, F, C).

        Returns
        -------
        torch.Tensor
            Normalized tensor with shape (B, F, C), same dtype as input.
        """
        in_dtype = x.dtype
        x = x.to(dtype=self.dtype)

        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * inv_rms

        x = x * self.adam_scale.unsqueeze(0)
        return x.to(dtype=in_dtype)

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "ScalarRMSNorm",
            "@version": 1,
            "config": {
                "channels": self.channels,
                "n_focus": self.n_focus,
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
    Focus-aware degree-wise linear self-interaction.

    This vectorized implementation avoids Python loops by using ``torch.einsum``
    and ``index_select``. The key insight is that weights are shared across all
    ``m`` components within each ``l`` block.

    Notes
    -----
    - Weight storage: ``(lmax+1, C_in, F*C_out)``.
    - Bias storage: ``(F*C_out,)``, only applied to ``l=0`` scalar components.
    - Runtime view restores weights to ``(lmax+1, C_in, F, C_out)`` via reshape.
    - ``expand_index`` maps each packed ``(l,m)`` position to its ``l`` value.
    - Einsum ``ndfi,difo->ndfo`` keeps the whole multi-focus path vectorized.
    - In HybridMuon slice mode, each ``(C_in, F*C_out)`` slice gets independent
      NS update with stable rectangular scaling.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    in_channels
        Number of input channels per (l, m) coefficient.
    out_channels
        Number of output channels per (l, m) coefficient.
    n_focus
        Number of focus streams.
    dtype
        Parameter dtype.
    mlp_bias
        Whether to use bias for l=0 (scalar) components.
    trainable
        Whether parameters are trainable.
    seed
        Random seed for weight initialization.
    init_std
        If given, use normal(0, init_std) for all weights instead of default
        trunc-normal fan-in/fan-out init. Use 0.0 for zero initialization.
    """

    def __init__(
        self,
        *,
        lmax: int,
        in_channels: int,
        out_channels: int,
        n_focus: int = 1,
        dtype: torch.dtype,
        mlp_bias: bool = True,
        trainable: bool,
        seed: int | list[int] | None = None,
        init_std: float | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.ebed_dim = get_so3_dim_of_lmax(self.lmax)
        self.mlp_bias = bool(mlp_bias)

        # === Step 1. Per-l weight matrix with focus folded on output axis ===
        # Storage: (lmax+1, C_in, F*C_out); runtime view: (lmax+1, C_in, F, C_out).
        num_l = self.lmax + 1
        self.weight = nn.Parameter(
            torch.empty(
                num_l,
                self.in_channels,
                self.n_focus * self.out_channels,
                dtype=self.dtype,
                device=self.device,
            )
        )
        if init_std is not None:
            if init_std == 0.0:
                nn.init.zeros_(self.weight)
            else:
                nn.init.normal_(
                    self.weight,
                    mean=0.0,
                    std=init_std,
                    generator=get_generator(seed),
                )
        else:
            for l_idx in range(num_l):
                init_trunc_normal_fan_in_out(
                    self.weight[l_idx],
                    child_seed(seed, 1000 + l_idx),
                )

        # === Step 2. Bias only for l=0 (scalar components) ===
        if self.mlp_bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(
                    self.n_focus * self.out_channels,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        else:
            self.bias = None

        # === Step 3. Precompute expand_index for weight lookup ===
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
            Input features with shape (N, D, F, C_in) where D=(lmax+1)^2.

        Returns
        -------
        torch.Tensor
            Order-wise mixed features with shape (N, D, F, C_out).
        """
        # === Step 1. Expand per-l weights to packed coefficient layout ===
        # (L, Cin, F*Cout) -> (L, Cin, F, Cout)
        weight = self.weight.view(
            self.lmax + 1,
            self.in_channels,
            self.n_focus,
            self.out_channels,
        )  # (L, Cin, F, Cout)
        # (L, Cin, F, Cout) -> (D, Cin, F, Cout)
        weight_expanded = torch.index_select(
            weight, dim=0, index=self.expand_index
        )  # (D, Cin, F, Cout)

        # === Step 2. Per-focus, per-degree channel mixing ===
        out = torch.einsum("ndfi,difo->ndfo", x, weight_expanded)

        # === Step 3. Add l=0 bias ===
        if self.mlp_bias:
            bias = self.bias.view(self.n_focus, self.out_channels)
            out[:, 0, :, :] = out[:, 0, :, :] + bias.unsqueeze(0)

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
                "n_focus": self.n_focus,
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
    SO(2)-equivariant linear mixing in the edge-aligned local frame.

    Coefficient layout (m-major, truncated by mmax)
    ------------------------------------------------
    The coefficient axis D_m_trunc is ordered by |m| groups::

        [  m=0: l=0..lmax  |  m=1: (l,-1) then (l,+1)  |  ...  |  m=mmax: ... ]
         |___ lmax+1 ____|   |_______ 2*(lmax) ________|

    Each |m| group is contiguous, enabling a single block-diagonal matmul.

    Block-diagonal weight structure
    -------------------------------
    The full weight matrix W has shape ``(F, D_m_trunc*Cout, D_m_trunc*Cin)``
    and is block-diagonal over |m| groups::

        W = diag[W_m0, B_m1, B_m2, ..., B_mmax]

    - ``W_m0``: unconstrained ``(num_l*Cout, num_l*Cin)`` block for m=0.
      Cross-l mixing is allowed since m=0 coefficients are real scalars.

    - ``B_m`` (|m|>0): SO(2)-constrained 2x2 block coupling (-m, +m) pairs::

          B_m = [ W_u^T , -W_v^T ]     where W_u, W_v are learnable
                [ W_v^T ,  W_u^T ]     (num_l*Cin, num_l*Cout) each.

      This structure is the real-valued form of complex multiplication
      ``(u + iv)(a + ib) = (ua - vb) + i(va + ub)``, which guarantees
      SO(2) equivariance: rotating the input by angle phi around z
      rotates the output by the same angle.

    The weight is assembled once per forward (training) or cached (eval)
    by ``_build_so2_weight()``, then applied via a single batched matmul
    over all focus streams: ``einsum("efi,foi->efo")``.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum SO(2) order (|m|) to mix. If None, defaults to ``lmax``.
    in_channels
        Number of input channels per (l, m) coefficient.
    out_channels
        Number of output channels per (l, m) coefficient.
    n_focus
        Number of independent focus streams. Each stream has its own
        weight matrices; the batched matmul vectorizes over all streams.
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
        n_focus: int = 1,
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
        self.n_focus = int(n_focus)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.mlp_bias = bool(mlp_bias)

        # === Step 1. Build m-major coefficient layout ===
        # Map each |m| group to contiguous index ranges in the flattened axis.
        # Example for lmax=2, mmax=2:
        #   m=0 : indices [0, 1, 2]        (l=0,1,2)
        #   m=1-: indices [3, 4]            (l=1,2 with -m)
        #   m=1+: indices [5, 6]            (l=1,2 with +m)
        #   m=2-: index  [7]               (l=2   with -m)
        #   m=2+: index  [8]               (l=2   with +m)
        #   => reduced_dim = 9
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

        # === Step 2. Learnable weight parameters ===
        # weight_m0: folded (num_l*Cin, F*num_l*Cout) storage — (in, out) convention.
        #   Runtime view: (num_l*Cin, F, num_l*Cout).
        #   Cross-l mixing is allowed because m=0 coefficients are real.
        num_m0 = self.lmax + 1
        num_in_m0 = num_m0 * self.in_channels
        num_out_m0 = num_m0 * self.out_channels
        self.weight_m0 = nn.Parameter(
            torch.empty(
                num_in_m0,
                self.n_focus * num_out_m0,
                device=self.device,
                dtype=self.dtype,
            )
        )
        weight_m0_view = self.weight_m0.view(num_in_m0, self.n_focus, num_out_m0)
        for focus_idx in range(self.n_focus):
            init_trunc_normal_fan_in_out(
                weight_m0_view[:, focus_idx, :], child_seed(seed, 1000 + focus_idx)
            )
        if self.mlp_bias:
            self.bias0: nn.Parameter | None = nn.Parameter(
                torch.zeros(
                    self.n_focus * self.out_channels,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        else:
            self.bias0 = None

        # weight_m[i]: folded (num_l*Cin, F*2*num_l*Cout) storage — (in, out) convention.
        #   Runtime view: (num_l*Cin, F, 2*num_l*Cout).
        #   The factor of 2 comes from storing W_u and W_v concatenated along the
        #   output axis. _build_so2_weight() splits them and fills the 2x2 block.
        #   Scaling by 1/sqrt(2) compensates for the doubled parameter count.
        self.weight_m: nn.ParameterList = nn.ParameterList()
        for m in range(1, self.mmax + 1):
            num_l = self.lmax - m + 1
            num_in = num_l * self.in_channels
            num_out = 2 * num_l * self.out_channels
            weight = nn.Parameter(
                torch.empty(
                    num_in,
                    self.n_focus * num_out,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            weight_view = weight.view(num_in, self.n_focus, num_out)
            for focus_idx in range(self.n_focus):
                init_trunc_normal_fan_in_out(
                    weight_view[:, focus_idx, :],
                    child_seed(seed, 2000 + m * 100 + focus_idx),
                )
            # Apply scaling for SO(2) equivariance
            weight.data.mul_(1.0 / math.sqrt(2.0))
            self.weight_m.append(weight)

        for p in self.parameters():
            p.requires_grad = trainable

        # === Step 3. Precompute flattened slice ranges for _build_so2_weight ===
        # Each |m|>0 group occupies two sub-blocks (neg, pos) in the flattened
        # weight matrix. Pre-computing the row/col ranges avoids repeated
        # arithmetic in the hot path.
        # Tuple layout: (neg_i0, neg_i1, pos_i0, pos_i1,   <- input row ranges
        #                neg_o0, neg_o1, pos_o0, pos_o1)   <- output col ranges
        self._m0_in = (self.lmax + 1) * self.in_channels
        self._m0_out = (self.lmax + 1) * self.out_channels
        self._block_slices: list[tuple[int, int, int, int, int, int, int, int]] = []
        for neg_start, pos_start, num_l in self._m_ranges:
            ib = num_l * self.in_channels
            ob = num_l * self.out_channels
            self._block_slices.append(
                (
                    neg_start * self.in_channels,
                    neg_start * self.in_channels + ib,
                    pos_start * self.in_channels,
                    pos_start * self.in_channels + ib,
                    neg_start * self.out_channels,
                    neg_start * self.out_channels + ob,
                    pos_start * self.out_channels,
                    pos_start * self.out_channels + ob,
                )
            )

        # Weight cache: only used in eval + no_grad (inference).
        # Invalidated on train() via overridden method below.
        self._cached_weight: torch.Tensor | None = None

    def train(self, mode: bool = True) -> SO2Linear:
        """Invalidate weight cache when switching to training mode."""
        self._cached_weight = None
        return super().train(mode)

    def _build_so2_weight(self) -> torch.Tensor:
        """
        Assemble the per-focus block-diagonal SO(2) weight matrix.

        The flattened weight has shape ``(D_m*Cin, F, D_m*Cout)`` (in, out)
        where both axes follow the same m-major coefficient ordering.
        Off-diagonal blocks (cross-|m|) are zero, enforcing SO(2) equivariance.

        Returns
        -------
        torch.Tensor
            Weight with shape (D_m*Cin, F, D_m*Cout).
        """
        in_total = self.reduced_dim * self.in_channels
        out_total = self.reduced_dim * self.out_channels
        weight = self.weight_m0.new_zeros(in_total, self.n_focus, out_total)
        num_in_m0 = (self.lmax + 1) * self.in_channels
        num_out_m0 = (self.lmax + 1) * self.out_channels
        weight_m0 = self.weight_m0.view(num_in_m0, self.n_focus, num_out_m0)

        # m=0 block: (Cin_blk, F, Cout_blk) — (in, out) convention.
        weight[: self._m0_in, :, : self._m0_out] = weight_m0

        # |m|>0 blocks: fill the 2x2 SO(2) coupling structure.
        # For each |m|, the learnable param w has shape (in_blk, F, 2*out_blk)
        # which is split into W_u and W_v along the output axis.
        for m_idx, w in enumerate(self.weight_m):
            ni0, ni1, pi0, pi1, no0, no1, po0, po1 = self._block_slices[m_idx]
            ib = ni1 - ni0  # in_block size
            ob = no1 - no0  # out_block size
            w = w.view(ib, self.n_focus, 2 * ob)
            w_u = w[:, :, :ob]  # (in_blk, F, out_blk)
            w_v = w[:, :, ob:]  # (in_blk, F, out_blk)
            # Fill the 2x2 coupling:
            #   Row = input (neg/pos), Col = output (neg/pos).
            #   [ W_u^T, -W_v^T ]^T  =>  row=neg_in: W_u to neg_out, W_v to pos_out
            #   [ W_v^T,  W_u^T ]^T  =>  row=pos_in: -W_v to neg_out, W_u to pos_out
            weight[ni0:ni1, :, no0:no1] = w_u  # neg_in -> neg_out
            weight[ni0:ni1, :, po0:po1] = w_v  # neg_in -> pos_out
            weight[pi0:pi1, :, no0:no1] = -w_v  # pos_in -> neg_out
            weight[pi0:pi1, :, po0:po1] = w_u  # pos_in -> pos_out
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input with shape (E, F, D_m_trunc, Cin), where D_m_trunc is the
            coefficient dimension of the m-major layout truncated by `mmax`.

        Returns
        -------
        torch.Tensor
            Output with shape (E, F, D_m_trunc, Cout), where Cout is output channels.
        """
        # === Step 1. Flatten coefficient + channel axes for matmul ===
        # (E, F, D_m, Cin) -> (E, F, D_m*Cin)
        n_edge = x.shape[0]
        in_dim_total = self.reduced_dim * self.in_channels
        x_flat = x.reshape(n_edge, self.n_focus, in_dim_total)

        # === Step 2. Get block-diagonal weight (cached in eval+no_grad) ===
        if self._cached_weight is not None:
            weight = self._cached_weight
        else:
            weight = self._build_so2_weight()
            # Cache only in eval mode with grad disabled (pure inference).
            if not self.training and not torch.is_grad_enabled():
                self._cached_weight = weight.detach()

        # === Step 3. Batched matmul over focus streams + reshape back ===
        # einsum "efi,ifo->efo": (E,F,D_m*Cin) x (D_m*Cin,F,D_m*Cout) -> (E,F,D_m*Cout)
        out_flat = torch.einsum("efi,ifo->efo", x_flat, weight)
        out = out_flat.reshape(
            n_edge, self.n_focus, self.reduced_dim, self.out_channels
        )

        # === Step 4. Bias on l=0 scalar index ===
        if self.mlp_bias:
            bias0 = self.bias0.view(self.n_focus, self.out_channels)
            out[:, :, 0, :] = out[:, :, 0, :] + bias0.unsqueeze(0)
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
                "n_focus": self.n_focus,
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
    SO(2)-equivariant edge convolution with cached geometry and rotations.

    This module consumes node features in packed SO(3) layout `(N, D, C)` and
    performs edge message passing in the reduced m-major local layout. The
    operation pipeline is:

    1. `pre_focus_mix`: full-channel mixing on node features `(N, D, C)`.
    2. rotate global -> local reduced basis with cached `D_to_m`.
    3. radial modulation in reduced layout.
    4. `so2_layers` stacked local mixers:
       `inter_norm -> SO2Linear -> non_linearity -> residual(+LayerScale)`.
    5. rotate local -> global with cached `Dt_from_m`.
    6. edge aggregation (plain envelope scatter or envelope-aware grouped
       softmax attention with envelope-weighted competition, value envelope
       decay, and output-side head gate).
    7. `post_focus_mix`: full-channel mixing on aggregated messages.

    Equivariance is preserved because both `pre_focus_mix` and `post_focus_mix`
    only mix the channel axis for each `(l, m)` coefficient and never mix
    coefficient indices across `(l, m)`.

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum SO(2) order (|m|). If None, defaults to lmax.
    channels
        Number of channels per (l, m) coefficient.
    n_focus
        Number of focus streams. Internal width is
        ``focus_dim = channels // n_focus``.
    focus_compete
        If True, apply cross-focus softmax competition in SO(2) local layout.
        Competition logits are constructed only from l=0 scalar channels and the
        resulting invariant weights are broadcast to all (l, m) components.
    so2_norm
        If True, apply intermediate ReducedSeparableRMSNorm as pre-norm before
        each SO(2) mixing layer. The last SO(2) layer always uses Identity.
    so2_layers
        Number of SO2Linear layers per convolution (default: 1).
    layer_scale
        If True, apply per-layer learnable LayerScale (per-focus-channel,
        init 1e-3) on each SO(2) residual branch.
    n_atten_head
        Number of attention heads used during aggregation.
        - 0: plain envelope-weighted scatter-sum.
        - >0: envelope-aware grouped softmax attention with output-side head
          gates.
        Requires ``focus_dim % n_atten_head == 0``.
    mlp_bias
        Whether to use bias in SO2Linear (l=0 bias), GatedActivation (gate linear bias),
        and ReducedSeparableRMSNorm (centering bias).
    eps
        Small epsilon for normalization modules.
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
        n_focus: int = 1,
        focus_compete: bool = True,
        so2_norm: bool = False,
        so2_layers: int = 4,
        layer_scale: bool = False,
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
        self.n_focus = int(n_focus)
        if self.channels % self.n_focus != 0:
            raise ValueError("`channels` must be divisible by `n_focus`")
        self.focus_dim = self.channels // self.n_focus
        self.focus_compete = bool(focus_compete)
        self.focus_softmax_tau = 1.0
        self.focus_label_smoothing = 0.02
        self.attn_env_logit_power = 0.5
        self.so2_norm = bool(so2_norm)
        self.so2_layers = int(so2_layers)
        if self.so2_layers < 1:
            raise ValueError("`so2_layers` must be >= 1")
        self.layer_scale = bool(layer_scale)
        self.n_atten_head = int(n_atten_head)
        if self.n_atten_head < 0:
            raise ValueError("`n_atten_head` must be non-negative")
        if self.n_atten_head > 0 and self.focus_dim % self.n_atten_head != 0:
            raise ValueError(
                "`focus_dim` must be divisible by `n_atten_head` when attention is enabled"
            )
        self.head_dim = (
            None if self.n_atten_head == 0 else int(self.focus_dim // self.n_atten_head)
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
        seed_so3_pre = child_seed(seed, 2)
        seed_so3_post = child_seed(seed, 3)
        seed_gate = child_seed(seed, 4)

        # === Step 3. Multiple SO2Linear layers ===
        self.so2_linears = nn.ModuleList(
            [
                SO2Linear(
                    lmax=self.lmax,
                    mmax=self.mmax,
                    in_channels=self.focus_dim,
                    out_channels=self.focus_dim,
                    n_focus=self.n_focus,
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
                        channels=self.focus_dim,
                        degree_index_m=self.degree_index_m,
                        n_focus=self.n_focus,
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
                    channels=self.focus_dim,
                    n_focus=self.n_focus,
                    dtype=self.dtype,
                    mlp_bias=self.mlp_bias,
                    layout="nfdc",
                    trainable=trainable,
                    seed=child_seed(seed_non_linearities, i),
                )
            )
        non_linearities.append(nn.Identity())
        self.non_linearities = nn.ModuleList(non_linearities)

        # === Step 6. Optional per-layer LayerScale for SO(2) residual branches ===
        if self.layer_scale:
            self.adam_so2_layer_scales = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.ones(
                            self.n_focus,
                            self.focus_dim,
                            dtype=self.dtype,
                            device=self.device,
                        )
                        * 1e-3,
                        requires_grad=trainable,
                    )
                    for _ in range(self.so2_layers)
                ]
            )
        else:
            self.adam_so2_layer_scales = None

        # === Step 7. Optional attention projections (n_atten_head > 0) ===
        self.attn_qk_norm: ScalarRMSNorm | None = None
        self.attn_q_proj: FocusLinear | None = None
        self.attn_k_proj: FocusLinear | None = None
        self.adamw_attn_logit_w: nn.Parameter | None = None
        self.attn_output_gate_norm: ScalarRMSNorm | None = None
        self.adamw_attn_gate_w: nn.Parameter | None = None
        if self.n_atten_head > 0:
            self.attn_qk_norm = ScalarRMSNorm(
                channels=self.focus_dim,
                n_focus=self.n_focus,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
            self.attn_q_proj = FocusLinear(
                in_channels=self.focus_dim,
                out_channels=self.focus_dim,
                n_focus=self.n_focus,
                dtype=self.compute_dtype,
                bias=False,
                seed=child_seed(seed_gate, 0),
                trainable=trainable,
            )
            self.attn_k_proj = FocusLinear(
                in_channels=self.focus_dim,
                out_channels=self.focus_dim,
                n_focus=self.n_focus,
                dtype=self.compute_dtype,
                bias=False,
                seed=child_seed(seed_gate, 1),
                trainable=trainable,
            )
            self.adamw_attn_logit_w = nn.Parameter(
                torch.empty(
                    self.focus_dim,
                    self.n_focus,
                    self.n_atten_head,
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=trainable,
            )
            nn.init.normal_(
                self.adamw_attn_logit_w,
                mean=0.0,
                std=0.01,
                generator=get_generator(child_seed(seed_gate, 2)),
            )
            self.attn_output_gate_norm = ScalarRMSNorm(
                channels=self.focus_dim,
                n_focus=self.n_focus,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
            self.adamw_attn_gate_w = nn.Parameter(
                torch.empty(
                    self.focus_dim,
                    self.n_focus,
                    self.n_atten_head,
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=trainable,
            )
            nn.init.normal_(
                self.adamw_attn_gate_w,
                mean=0.0,
                std=0.01,
                generator=get_generator(child_seed(seed_gate, 3)),
            )

        # === Step 7.5. Optional cross-focus competition ===
        self.focus_compete_norm: ScalarRMSNorm | None = None
        self.adamw_focus_compete_w: nn.Parameter | None = None
        self.focus_compete_bias: nn.Parameter | None = None
        if self.focus_compete and self.n_focus > 1:
            self.focus_compete_norm = ScalarRMSNorm(
                channels=self.focus_dim,
                n_focus=self.n_focus,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
            self.adamw_focus_compete_w = nn.Parameter(
                torch.empty(
                    self.focus_dim,
                    self.n_focus,
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=trainable,
            )
            nn.init.normal_(
                self.adamw_focus_compete_w,
                mean=0.0,
                std=0.01,
                generator=get_generator(child_seed(seed_gate, 4)),
            )
            if self.mlp_bias:
                self.focus_compete_bias = nn.Parameter(
                    torch.zeros(
                        self.n_focus,
                        dtype=self.compute_dtype,
                        device=self.device,
                    ),
                    requires_grad=trainable,
                )

        # === Step 8. Pre-focus channel mixing ===
        # This mixes the full channel width before focus slicing.
        self.pre_focus_mix = SO3Linear(
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=self.channels,
            n_focus=1,
            dtype=dtype,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3_pre,
        )

        # === Step 9. Post-focus channel mixing ===
        self.post_focus_mix = SO3Linear(
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=self.channels,
            n_focus=1,
            dtype=dtype,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3_post,
            init_std=0.0,
        )

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
            Node features with shape (N, D, C), where D=(lmax+1)^2 is the
            SO(3) coefficient dimension.
        edge_cache
            Precomputed edge cache. Must be compatible with this block's lmax.
        radial_feat
            Per-edge radial features with shape (E, lmax+1, C), already fused
            with edge type features.

        Returns
        -------
        torch.Tensor
            Message updates with shape (N, D, C).
        """
        src, dst = edge_cache.src, edge_cache.dst
        n_node = x.shape[0]
        n_edge = src.numel()

        # === Step 1. Pre-focus channel mixing on full width ===
        with nvtx_range("SO2Conv/pre_focus_mix"):
            x = self.pre_focus_mix(x.unsqueeze(2)).squeeze(2)

        # === Step 2. Rotate to edge-aligned local frame ===
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
            x_src = x.index_select(0, src)  # (E, D, C)
            x_local = torch.bmm(D_m_prime, x_src)  # (E, D_m, C)

        # === Step 3. Select radial/type features for reduced layout ===
        with nvtx_range("SO2Conv/radial_fuse"):
            rad_feat = radial_feat[:, self.degree_index_m, :]  # (E, D_m, C)
            x_local.mul_(rad_feat)
            rad_feat_l0_focus = rad_feat[:, 0, :].reshape(
                n_edge, self.n_focus, self.focus_dim
            )  # (E, F, Cf)

        # === Step 4. Convert to SO(2) internal focus layout ===
        with nvtx_range("SO2Conv/reshape_for_so2"):
            x_local = x_local.reshape(
                n_edge, self.reduced_dim, self.n_focus, self.focus_dim
            ).transpose(1, 2)  # (E, F, D_m, Cf), strided
            if self.focus_compete and self.n_focus > 1:
                focus_gate_src = x_local[:, :, 0, :]

        # === Step 5. Multi-layer SO(2) mixing (pre-norm + residual + LayerScale) ===
        with nvtx_range("SO2Conv/so2_layers"):
            for layer_idx, (so2_linear, inter_norm, non_linear) in enumerate(
                zip(self.so2_linears, self.so2_inter_norms, self.non_linearities)
            ):
                residual = x_local
                x_local = inter_norm(x_local)
                x_local = so2_linear(x_local)

                if layer_idx == 0 and so2_linear.bias0 is not None:
                    # bias0: (F*Cf,) → (1, F, Cf) for broadcasting with (E, F, Cf)
                    bias0 = so2_linear.bias0.view(
                        self.n_focus, self.focus_dim
                    ).unsqueeze(0)
                    bias_correction = bias0 * (
                        rad_feat_l0_focus * edge_cache.edge_env.reshape(-1, 1, 1) - 1.0
                    )  # (E, F, Cf)
                    x_local[:, :, 0, :].add_(bias_correction)

                x_local = non_linear(x_local)

                if self.layer_scale:
                    scale = self.adam_so2_layer_scales[layer_idx].reshape(
                        1, self.n_focus, 1, self.focus_dim
                    )
                    x_local = residual + scale * x_local
                else:
                    x_local = residual + x_local

        # === Step 5.5. Cross-focus softmax competition ===
        if self.focus_compete and self.n_focus > 1:
            focus_gate_src = focus_gate_src.to(dtype=self.compute_dtype)
            focus_logits = torch.einsum(
                "efi,if->ef",
                self.focus_compete_norm(focus_gate_src),
                self.adamw_focus_compete_w,
            )
            if self.mlp_bias:
                focus_logits = focus_logits + self.focus_compete_bias.unsqueeze(0)
            alpha = torch.softmax(focus_logits / self.focus_softmax_tau, dim=1).to(
                dtype=x_local.dtype
            )
            alpha = alpha * (1.0 - self.focus_label_smoothing) + (
                self.focus_label_smoothing / float(self.n_focus)
            )
            x_local = x_local * alpha.unsqueeze(-1).unsqueeze(-1)

        # === Step 6. Restore reduced global layout for inverse rotation ===
        with nvtx_range("SO2Conv/reshape_for_rotate_back"):
            x_local = x_local.transpose(1, 2).contiguous()  # (E, D_m, F, Cf)
            x_local = x_local.reshape(
                n_edge, self.reduced_dim, self.channels
            )  # (E, D_m, C)

        # === Step 7. Rotate back to global frame ===
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

        # === Step 8. Aggregate with optional head-wise gating ===
        with nvtx_range("SO2Conv/aggregate"):
            if self.n_atten_head == 0:
                # Baseline path: fused envelope-weighted scatter add -> degree norm
                x_message = x_message * edge_cache.edge_env.unsqueeze(-1)
                out = x.new_zeros(x.shape, dtype=self.compute_dtype)
                out.index_add_(0, dst, x_message.to(dtype=self.compute_dtype))
                out.mul_(edge_cache.inv_sqrt_deg.to(dtype=self.compute_dtype))
                out = out.to(dtype=self.dtype)  # (N, D, C)
            else:
                # === Step 8.1. Build attention logits from scalar channels ===
                compute_dtype = self.compute_dtype
                x_l0_node = x[:, 0, :].reshape(
                    n_node, self.n_focus, self.focus_dim
                )  # (N, F, Cf)
                qk_input = self.attn_qk_norm(x_l0_node.to(dtype=compute_dtype))
                q_node = self.attn_q_proj(qk_input)  # (N, F, Cf)
                k_node = self.attn_k_proj(qk_input)  # (N, F, Cf)
                q_edge = q_node.index_select(0, dst).reshape(
                    n_edge, self.n_focus, self.n_atten_head, self.head_dim
                )  # (E, F, H, Dh)
                k_edge = k_node.index_select(0, src).reshape(
                    n_edge, self.n_focus, self.n_atten_head, self.head_dim
                )  # (E, F, H, Dh)
                radial_l0 = radial_feat[:, 0, :].reshape(
                    n_edge, self.n_focus, self.focus_dim
                )  # (E, F, Cf)
                radial_bias = torch.einsum(
                    "efi,ifo->efo",
                    radial_l0.to(dtype=compute_dtype),
                    self.adamw_attn_logit_w,
                )  # (E, F, H)
                attn_logits = (q_edge * k_edge).sum(-1) * (self.head_dim**-0.5)
                attn_logits = attn_logits + radial_bias

                # === Step 8.2. Destination-wise stable softmax with envelope-aware competition ===
                attn_alpha = segment_softmax_with_env_weight(
                    logits=attn_logits,
                    edge_env=edge_cache.edge_env.to(dtype=compute_dtype),
                    dst=dst,
                    n_nodes=n_node,
                    eps=self.eps,
                    env_logit_power=self.attn_env_logit_power,
                )  # (E, F, H)

                # === Step 8.3. Head-wise value aggregation with envelope amplitude decay ===
                value_heads = x_message.reshape(
                    n_edge,
                    self.ebed_dim_full,
                    self.n_focus,
                    self.n_atten_head,
                    self.head_dim,
                ).to(dtype=compute_dtype)  # (E, D, F, H, Dh)
                env_value_weight = edge_cache.edge_env.to(dtype=compute_dtype).squeeze(
                    -1
                )  # (E,)
                weighted_value = value_heads * attn_alpha.reshape(
                    n_edge, 1, self.n_focus, self.n_atten_head, 1
                )
                weighted_value = weighted_value * env_value_weight.reshape(
                    n_edge, 1, 1, 1, 1
                )
                out_heads = torch.zeros(
                    n_node,
                    self.ebed_dim_full,
                    self.n_focus,
                    self.n_atten_head,
                    self.head_dim,
                    device=x.device,
                    dtype=compute_dtype,
                )  # (N, D, F, H, Dh)
                out_heads.index_add_(0, dst, weighted_value)

                # === Step 8.4. Output-side head gate (G1 style) ===
                attn_output_gate = torch.sigmoid(
                    torch.einsum(
                        "nfi,ifo->nfo",
                        self.attn_output_gate_norm(x_l0_node.to(dtype=compute_dtype)),
                        self.adamw_attn_gate_w,
                    )
                )  # (N, F, H)
                out_heads = out_heads * attn_output_gate.reshape(
                    n_node, 1, self.n_focus, self.n_atten_head, 1
                )  # (N, D, F, H, Dh)

                # === Step 8.5. Merge heads (softmax path has no degree norm) ===
                out = out_heads.reshape(n_node, self.ebed_dim_full, self.channels).to(
                    dtype=self.dtype
                )  # (N, D, C)

        # === Step 9. Final channel mixing ===
        with nvtx_range("SO2Conv/post_focus_mix"):
            out = self.post_focus_mix(out.unsqueeze(2)).squeeze(2)
        return out  # (N, D, C)

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
                "n_focus": self.n_focus,
                "focus_compete": self.focus_compete,
                "so2_norm": self.so2_norm,
                "so2_layers": self.so2_layers,
                "layer_scale": self.layer_scale,
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
            n_focus=1,
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
            layout="ndfc",
            trainable=trainable,
            seed=seed_act,
        )

        # === Second SO3Linear for channel mixing ===
        # Zero-initialized so residual path starts near-identity.
        self.so3_linear_2 = SO3Linear(
            lmax=self.lmax,
            in_channels=self.hidden_channels,
            out_channels=self.channels,
            n_focus=1,
            dtype=dtype,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3_out,
            init_std=0.0,
        )

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input with shape (N, D, F, C) where D=(lmax+1)^2.

        Returns
        -------
        torch.Tensor
            Output with shape (N, D, F, C).
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
    SeZM interaction block with SO(2) message passing and equivariant FFN stack.

    Branch order:
    1. SO(2) branch: optional pre-norm -> `SO2Convolution` -> optional post-norm
       -> residual add.
    2. FFN branch: repeated subblocks of
       optional pre-norm -> `EquivariantFFN` -> optional post-norm -> residual add.

    `SO2Convolution` internally handles `pre_focus_mix`/`post_focus_mix`, so this
    block operates on the canonical node layout `(N, D, F, Cf)` at boundaries.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum SO(2) order (|m|) mixed inside SO(2) convolution.
    channels
        Total channels per (l, m) coefficient.
    n_focus
        Number of multi-focus streams used by SO(2) branch. Per-focus width is
        ``focus_dim = channels // n_focus``.
    focus_compete
        If True, enable cross-focus softmax competition in SO(2) convolution.
    so2_norm
        If True, apply intermediate ReducedSeparableRMSNorm between SO(2) mixing layers.
        When False (default), no normalization is applied between layers.
    so2_layers
        Number of SO(2) mixing layers.
    n_atten_head
        Number of attention heads when aggregating messages in SO(2) convolution.
        0 means no attention is used; >0 enables envelope-aware grouped softmax
        attention with output-side head gate.
    so2_pre_norm
        If True, apply pre-norm before SO(2) convolution.
    so2_post_norm
        If True, apply post-norm on SO(2) output before the residual add.
    ffn_pre_norm
        If True, apply pre-norm before each FFN subblock.
    ffn_post_norm
        If True, apply post-norm on each FFN subblock output before the residual add.
    ffn_neurons
        Hidden dimension for each FFN subblock.
    ffn_blocks
        Number of FFN subblocks per block.
    layer_scale
        If True, apply learnable LayerScale (init 1e-3) on residual branches:
        - SO(2) branch: per-focus-channel scales `(n_focus, focus_dim)`
          on each SO(2) mixing layer.
        - FFN branch: per-channel scales `(channels,)` on each FFN subblock.
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
        n_focus: int = 1,
        focus_compete: bool = True,
        so2_norm: bool = False,
        so2_layers: int = 4,
        n_atten_head: int = 0,
        so2_pre_norm: bool = True,
        so2_post_norm: bool = False,
        ffn_pre_norm: bool = True,
        ffn_post_norm: bool = False,
        ffn_neurons: int = 96,
        ffn_blocks: int = 1,
        layer_scale: bool = False,
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
        self.n_focus = int(n_focus)
        if self.channels % self.n_focus != 0:
            raise ValueError("`channels` must be divisible by `n_focus`")
        self.focus_dim = self.channels // self.n_focus
        self.focus_compete = bool(focus_compete)
        self.so2_norm = bool(so2_norm)
        self.so2_layers = int(so2_layers)
        self.n_atten_head = int(n_atten_head)
        self.so2_pre_norm = bool(so2_pre_norm)
        self.so2_post_norm = bool(so2_post_norm)
        self.ffn_pre_norm = bool(ffn_pre_norm)
        self.ffn_post_norm = bool(ffn_post_norm)
        self.ffn_neurons = int(ffn_neurons)
        self.ffn_blocks = int(ffn_blocks)
        if self.ffn_blocks < 1:
            raise ValueError("`ffn_blocks` must be >= 1")
        self.layer_scale = bool(layer_scale)
        self.activation_function = activation_function
        self.glu_activation = bool(glu_activation)
        self.mlp_bias = bool(mlp_bias)
        self.eps = float(eps)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.compute_dtype = get_promoted_dtype(self.dtype)

        # === Step 0. Split deterministic seeds at the block top-level ===
        seed_so2_conv = child_seed(seed, 0)
        seed_ffn = child_seed(seed, 1)

        # === Step 1. SO(2) convolution branch norms ===
        if self.so2_pre_norm:
            self.pre_so2_norm: nn.Module = SeparableRMSNorm(
                self.lmax,
                self.focus_dim,
                n_focus=self.n_focus,
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
                self.focus_dim,
                n_focus=self.n_focus,
                centering=self.mlp_bias,
                eps=eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
        else:
            self.post_so2_norm = nn.Identity()

        self.so2_conv = SO2Convolution(
            lmax=self.lmax,
            mmax=self.mmax,
            channels=self.channels,
            n_focus=self.n_focus,
            focus_compete=self.focus_compete,
            so2_norm=self.so2_norm,
            so2_layers=self.so2_layers,
            layer_scale=self.layer_scale,
            n_atten_head=n_atten_head,
            mlp_bias=self.mlp_bias,
            eps=self.eps,
            dtype=dtype,
            seed=seed_so2_conv,
            trainable=trainable,
        )

        # === Step 2. FFN subblock sequence ===
        pre_ffn_norms: list[nn.Module] = []
        post_ffn_norms: list[nn.Module] = []
        ffns: list[EquivariantFFN] = []

        for i in range(self.ffn_blocks):
            seed_ffn_i = child_seed(seed_ffn, i)

            if self.ffn_pre_norm:
                pre_ffn_norms.append(
                    SeparableRMSNorm(
                        self.lmax,
                        self.channels,
                        n_focus=1,
                        centering=self.mlp_bias,
                        eps=eps,
                        dtype=self.compute_dtype,
                        trainable=trainable,
                    )
                )
            else:
                pre_ffn_norms.append(nn.Identity())

            if self.ffn_post_norm:
                post_ffn_norms.append(
                    SeparableRMSNorm(
                        self.lmax,
                        self.channels,
                        n_focus=1,
                        centering=self.mlp_bias,
                        eps=eps,
                        dtype=self.compute_dtype,
                        trainable=trainable,
                    )
                )
            else:
                post_ffn_norms.append(nn.Identity())

            ffns.append(
                EquivariantFFN(
                    lmax=self.lmax,
                    channels=self.channels,
                    hidden_channels=ffn_neurons,
                    dtype=dtype,
                    activation_function=activation_function,
                    glu_activation=self.glu_activation,
                    mlp_bias=self.mlp_bias,
                    trainable=trainable,
                    seed=seed_ffn_i,
                )
            )

        self.pre_ffn_norms = nn.ModuleList(pre_ffn_norms)
        self.post_ffn_norms = nn.ModuleList(post_ffn_norms)
        self.ffns = nn.ModuleList(ffns)

        # Optional per-channel LayerScale on each FFN residual branch
        if self.layer_scale:
            self.adam_ffn_layer_scales = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.ones(self.channels, dtype=self.dtype, device=self.device)
                        * 1e-3,
                        requires_grad=trainable,
                    )
                    for _ in range(self.ffn_blocks)
                ]
            )
        else:
            self.adam_ffn_layer_scales = None

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
            Features with shape (N, D, F, Cf), where F*Cf=channels.
        edge_cache
            Edge cache.
        radial_feat
            Per-edge radial features with shape (E, lmax+1, C).

        Returns
        -------
        torch.Tensor
            Updated features with shape (N, D, F, Cf).
        """
        n_node = x.shape[0]
        ebed_dim = x.shape[1]
        channels = self.channels

        # === Step 1. SO(2) convolution branch ===
        with nvtx_range("so2_conv"):
            x_pre = self.pre_so2_norm(x)
            y = self.so2_conv(
                x_pre.reshape(n_node, ebed_dim, channels), edge_cache, radial_feat
            )
            y = self.post_so2_norm(
                y.reshape(n_node, ebed_dim, self.n_focus, self.focus_dim)
            )
            x = x + y

        # === Step 2. FFN sublayer sequence ===
        x_ffn = x.reshape(n_node, ebed_dim, 1, channels)  # (N, D, 1, C)
        for i in range(self.ffn_blocks):
            with nvtx_range(f"ffn_{i}/pre_norm"):
                x_pre = self.pre_ffn_norms[i](x_ffn)

            with nvtx_range(f"ffn_{i}/ffn"):
                y = self.ffns[i](x_pre)

            with nvtx_range(f"ffn_{i}/post_norm"):
                y = self.post_ffn_norms[i](y)

            if self.layer_scale:
                y = y * self.adam_ffn_layer_scales[i]

            x_ffn = x_ffn + y

        return x_ffn.squeeze(2).reshape(n_node, ebed_dim, self.n_focus, self.focus_dim)

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
                "n_focus": self.n_focus,
                "focus_compete": self.focus_compete,
                "so2_norm": self.so2_norm,
                "so2_layers": self.so2_layers,
                "n_atten_head": self.n_atten_head,
                "so2_pre_norm": self.so2_pre_norm,
                "so2_post_norm": self.so2_post_norm,
                "ffn_pre_norm": self.ffn_pre_norm,
                "ffn_post_norm": self.ffn_post_norm,
                "ffn_neurons": self.ffn_neurons,
                "ffn_blocks": self.ffn_blocks,
                "activation_function": self.activation_function,
                "glu_activation": self.glu_activation,
                "mlp_bias": self.mlp_bias,
                "layer_scale": self.layer_scale,
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
