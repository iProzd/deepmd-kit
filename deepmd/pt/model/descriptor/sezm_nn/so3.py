# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SO(3)-equivariant linear and gating layers for SeZM.

This module defines the focus-aware linear maps and gated nonlinearities
used by SeZM SO(3) feature transformations.
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

from .indexing import (
    build_m_major_l_index,
    get_so3_dim_of_lmax,
    map_degree_idx,
)
from .utils import (
    init_trunc_normal_fan_in_out,
    np_safe,
    safe_numpy_to_tensor,
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
            self.register_buffer("expand_index", expand_index, persistent=False)

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
                persistent=False,
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
            persistent=False,
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
