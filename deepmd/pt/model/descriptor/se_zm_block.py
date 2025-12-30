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
)

import torch
import torch.nn as nn

from deepmd.pt.model.network.mlp import (
    EmbeddingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)


def _so3_dim_of_lmax(lmax: int) -> int:
    """
    Return SO(3) representation dimension for given lmax.

    The dimension equals sum_{l<=lmax} (2l+1) = (lmax+1)^2,
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


def _build_degree_index(lmax: int, *, device: torch.device) -> torch.Tensor:
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


class GatedActivation(nn.Module):
    """
    Gated activation for SO(3) equivariant features with per-l independent gates.

    - l=0: SiLU activation
    - l>0: Each degree l has an independent gate derived from the l=0 scalar features.
           The gate for each l is expanded to all m components within that l-block.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    channels
        Number of channels per (l, m) coefficient.
    dtype
        Parameter dtype.
    """

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.dtype = dtype
        self.device = env.DEVICE

        self.scalar_act = torch.nn.SiLU()
        self.gate_act = torch.nn.Sigmoid()

        # === Build expand_index for mapping per-l gates to all m components ===
        if self.lmax > 0:
            # expand_index[k] = l-1 for the k-th component in the l>0 portion
            # This maps each (l, m) component to its corresponding gate index
            num_components = 0
            for l in range(1, self.lmax + 1):
                num_components += 2 * l + 1  # all m from -l to l
            expand_index = torch.zeros(
                num_components, dtype=torch.long, device=self.device
            )
            start_idx = 0
            for l in range(1, self.lmax + 1):
                length = 2 * l + 1
                expand_index[start_idx : start_idx + length] = l - 1
                start_idx += length
            self.register_buffer("expand_index", expand_index, persistent=True)

            # Linear to generate lmax independent gates from scalar features
            self.gate_linear: nn.Linear | None = nn.Linear(
                self.channels,
                self.lmax * self.channels,
                bias=True,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.register_buffer(
                "expand_index",
                torch.zeros(0, dtype=torch.long, device=self.device),
                persistent=True,
            )
            self.gate_linear = None

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
        gating_scalars = self.gate_linear(x[:, 0, :])  # (N, lmax * C)
        gating_scalars = self.gate_act(gating_scalars)

        # Reshape to (N, lmax, C) then expand to (N, D-1, C)
        gating_scalars = gating_scalars.view(x.shape[0], self.lmax, -1)
        expand_idx = self.expand_index.to(device=x.device)
        gates = torch.index_select(
            gating_scalars, dim=1, index=expand_idx
        )  # (N, D-1, C)

        # === Step 3. Apply gates to l>0 components ===
        xt = x[:, 1:, :] * gates

        return torch.cat([x0, xt], dim=1)


class SeparableRMSNorm(nn.Module):
    """
    Separable RMSNorm for scalar (l=0) and non-scalar (l>0) features.

    - l=0 and l>0 compute RMS separately (separable design).
    - affine: per-l learnable scale for all degrees.
    - centering: subtract mean for l=0 only (l>0 must remain zero-mean for equivariance).
    """

    def __init__(
        self,
        lmax: int,
        channels: int,
        affine: bool = True,
        centering: bool = True,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.affine = affine
        self.centering = centering
        self.dtype = dtype
        self.device = env.DEVICE
        self.eps = torch.finfo(self.dtype).eps

        if self.affine:
            # Per-l affine weights: (lmax+1, C)
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

            # Build expand_index for l>0 portion
            if self.lmax > 0:
                num_components = (self.lmax + 1) ** 2 - 1  # D - 1
                expand_index = torch.zeros(
                    num_components, dtype=torch.long, device=self.device
                )
                start_idx = 0
                for l in range(1, self.lmax + 1):
                    length = 2 * l + 1
                    expand_index[start_idx : start_idx + length] = l
                    start_idx += length
                self.register_buffer("expand_index", expand_index, persistent=True)
            else:
                self.register_buffer(
                    "expand_index",
                    torch.zeros(0, dtype=torch.long, device=self.device),
                    persistent=True,
                )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self.register_buffer(
                "expand_index",
                torch.zeros(0, dtype=torch.long, device=self.device),
                persistent=True,
            )

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
        x0 = x[:, :1, :]
        xt = x[:, 1:, :]

        # === Step 1. l=0: RMS norm with optional centering ===
        if self.centering:
            x0 = x0 - x0.mean(dim=-1, keepdim=True)
        rms0 = torch.sqrt(torch.mean(x0 * x0, dim=-1, keepdim=True) + self.eps)
        x0 = x0 / rms0
        if self.affine:
            x0 = x0 * self.weight[0].view(1, 1, -1)
            if self.bias is not None:
                x0 = x0 + self.bias.view(1, 1, -1)

        if xt.numel() == 0:
            return x0

        # === Step 2. l>0: RMS norm (no centering to preserve equivariance) ===
        rmst = torch.sqrt(torch.mean(xt * xt, dim=(1, 2), keepdim=True) + self.eps)
        xt = xt / rmst
        if self.affine:
            wt = torch.index_select(
                self.weight, dim=0, index=self.expand_index
            )  # (D-1, C)
            xt = xt * wt.unsqueeze(0)

        return torch.cat([x0, xt], dim=1)


class PerDegreeLinear(nn.Module):
    """Degree-wise linear self-interaction shared across m within each l-block."""

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        linears: list[nn.Linear] = []
        for l in range(self.lmax + 1):
            bias = l == 0
            linears.append(
                nn.Linear(
                    self.channels,
                    self.channels,
                    bias=bias,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        self.linears = nn.ModuleList(linears)

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input features with shape (N, D, C) where D=(lmax+1)^2.

        Returns
        -------
        torch.Tensor
            Order-wise mixed features with shape (N, D, C).
        """
        out = x.new_empty(x.shape)
        offset = 0
        for l, linear in enumerate(self.linears):
            dim = 2 * l + 1
            seg = x[:, offset : offset + dim, :]
            out[:, offset : offset + dim, :] = linear(seg)
            offset += dim
        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "PerDegreeLinear",
            "@version": 1,  # keep 1 at devel stage
            "lmax": self.lmax,
            "channels": self.channels,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "weights": [
                {
                    "weight": to_numpy_array(layer.weight),
                    "bias": to_numpy_array(layer.bias)
                    if layer.bias is not None
                    else None,
                }
                for layer in self.linears
            ],
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> PerDegreeLinear:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls not in (
            "PerDegreeLinear",
            "PerOrderLinear",
        ):  # Accept both for transition
            raise ValueError(f"Invalid class for PerDegreeLinear: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported PerDegreeLinear version: {version}")

        precision = data["precision"]
        dtype = PRECISION_DICT[precision]
        obj = cls(
            lmax=int(data["lmax"]),
            channels=int(data["channels"]),
            dtype=dtype,
            trainable=True,
        )
        weight_data = data["weights"]
        for layer, state in zip(obj.linears, weight_data):
            layer.weight.data.copy_(
                torch.as_tensor(state["weight"], device=obj.device, dtype=obj.dtype)
            )
            if state["bias"] is not None and layer.bias is not None:
                layer.bias.data.copy_(
                    torch.as_tensor(state["bias"], device=obj.device, dtype=obj.dtype)
                )
        return obj


class PerDegreeLinearV2(nn.Module):
    """
    Degree-wise linear self-interaction using einsum for efficiency.

    This is a vectorized version of PerDegreeLinear that avoids Python loops
    by using torch.einsum and index_select. The key insight is that weights
    are shared across all m components within each l-block.

    Design notes
    ------------
    - weight shape: (lmax+1, C, C) - per-l linear transformation
    - bias shape: (C,) - only applied to l=0 (scalar) components
    - expand_index: maps each (l,m) position to its l value for weight lookup
    - Uses einsum 'bmi, lci -> blmc' pattern for batched per-l matmul
    - Avoids Python for-loops and torch.cat operations
    """

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.ebed_dim = _so3_dim_of_lmax(self.lmax)

        # === Step 1. Per-l weight matrix: (lmax+1, C, C) ===
        # Each l has an independent C x C linear transformation
        # that is shared across all 2l+1 m components.
        bound = 1.0 / math.sqrt(self.channels)
        self.weight = nn.Parameter(
            torch.empty(
                self.lmax + 1,
                self.channels,
                self.channels,
                dtype=self.dtype,
                device=self.device,
            )
        )
        nn.init.uniform_(self.weight, -bound, bound)

        # === Step 2. Bias only for l=0 (scalar components) ===
        self.bias = nn.Parameter(
            torch.zeros(self.channels, dtype=self.dtype, device=self.device)
        )

        # === Step 3. Precompute expand_index for weight lookup ===
        # expand_index[i] = l for position i in the packed (l,m) layout
        # This maps each (l,m) component to its corresponding weight index.
        self.register_buffer(
            "_expand_index",
            _build_degree_index(self.lmax, device=self.device),
            persistent=True,
        )

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input features with shape (N, D, C) where D=(lmax+1)^2.

        Returns
        -------
        torch.Tensor
            Order-wise mixed features with shape (N, D, C).
        """
        # === Step 1. Expand weight: (lmax+1, C, C) -> (D, C, C) ===
        # Use index_select to duplicate each weight matrix to all m components.
        weight_expanded = torch.index_select(
            self.weight, dim=0, index=self._expand_index
        )  # (D, C, C)

        # === Step 2. Batched per-l matmul using einsum ===
        # Pattern explanation:
        #   b: batch dimension (N)
        #   m: m-component dimension (D)
        #   i: input channels (C)
        #   l: lookup dimension (D, same as m after expansion)
        #   c: output channels (C)
        # Result: (N, D, C) where each position (b, m, :) = weight[l] @ x[b, m, :]
        out = torch.einsum("bmi,mci->bmc", x, weight_expanded)  # (N, D, C)

        # === Step 3. Add bias only to l=0 (index 0) ===
        out[:, 0, :] = out[:, 0, :] + self.bias.view(1, -1)

        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "PerDegreeLinearV2",
            "@version": 1,
            "lmax": self.lmax,
            "channels": self.channels,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "weight": to_numpy_array(self.weight),
            "bias": to_numpy_array(self.bias),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> PerDegreeLinearV2:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls not in (
            "PerDegreeLinearV2",
            "PerOrderLinearV2",
        ):  # Accept both for transition
            raise ValueError(f"Invalid class for PerDegreeLinearV2: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported PerDegreeLinearV2 version: {version}")

        precision = data["precision"]
        dtype = PRECISION_DICT[precision]
        obj = cls(
            lmax=int(data["lmax"]),
            channels=int(data["channels"]),
            dtype=dtype,
            trainable=True,
        )
        obj.weight.data.copy_(
            torch.as_tensor(data["weight"], device=obj.device, dtype=obj.dtype)
        )
        obj.bias.data.copy_(
            torch.as_tensor(data["bias"], device=obj.device, dtype=obj.dtype)
        )
        return obj


class SO2Linear(nn.Module):
    """
    SO(2) linear mixing in the edge-aligned frame.

    Layout invariant:
    - input and output are packed by (l,m) with m=-l..l per l-block
    - mixing respects SO(2) symmetry: parameters are shared between +m and -m
    """

    def __init__(
        self,
        *,
        lmax: int,
        in_channels: int,
        out_channels: int,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        # === Step 1. Precompute packed indices (buffers) ===
        m0 = [l * l + l for l in range(self.lmax + 1)]
        self.register_buffer(
            "_m0_idx",
            torch.tensor(m0, device=self.device, dtype=torch.long),
            persistent=True,
        )

        for m in range(1, self.lmax + 1):
            pos_idx = [l * l + l + m for l in range(m, self.lmax + 1)]
            neg_idx = [l * l + l - m for l in range(m, self.lmax + 1)]
            self.register_buffer(
                f"_pos_idx_{m}",
                torch.tensor(pos_idx, device=self.device, dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                f"_neg_idx_{m}",
                torch.tensor(neg_idx, device=self.device, dtype=torch.long),
                persistent=True,
            )
        del seed

        # === Step 2. Mixing per |m| group, cross-l allowed, bias only for scalar index ===
        num_m0 = len(m0)
        self.linear_m0 = nn.Linear(
            num_m0 * self.in_channels,
            num_m0 * self.out_channels,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )
        self.bias0 = nn.Parameter(
            torch.zeros(self.out_channels, device=self.device, dtype=self.dtype)
        )

        # For |m|>0, SO(2) equivariance requires 2x2 block structure on (Re, Im) pairs.
        # Output dimension is doubled for complex mixing: (a, -b; b, a) constraint.
        self.linears_m: nn.ModuleList = nn.ModuleList()
        for m in range(1, self.lmax + 1):
            num_l = self.lmax - m + 1
            fc = nn.Linear(
                num_l * self.in_channels,
                2 * num_l * self.out_channels,
                bias=False,
                device=self.device,
                dtype=self.dtype,
            )
            fc.weight.data.mul_(1.0 / math.sqrt(2.0))
            self.linears_m.append(fc)

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input with shape (..., D, Cin).

        Returns
        -------
        torch.Tensor
            Output with shape (..., D, Cout).
        """
        *lead, ebed_dim, Cin = x.shape
        batch = math.prod(lead) if lead else 1
        x2 = x.reshape(batch, ebed_dim, Cin)
        out = x2.new_zeros(batch, ebed_dim, self.out_channels)

        # m = 0 group
        m0_idx = self._m0_idx.to(device=x.device)
        x_m0 = x2[:, m0_idx, :].reshape(batch, -1)
        y_m0 = self.linear_m0(x_m0).reshape(batch, m0_idx.numel(), self.out_channels)
        out[:, m0_idx, :] = y_m0
        out[:, 0, :] = out[:, 0, :] + self.bias0

        # |m| > 0 groups: complex mixing via 2x2 block structure
        for m, linear in enumerate(self.linears_m, start=1):
            pos_idx = getattr(self, f"_pos_idx_{m}").to(device=x.device)
            neg_idx = getattr(self, f"_neg_idx_{m}").to(device=x.device)
            num_l = pos_idx.numel()

            # Treat (neg, pos) as (Re, Im) complex pair
            x_neg = x2[:, neg_idx, :].reshape(batch, -1)  # (batch, num_l * Cin)
            x_pos = x2[:, pos_idx, :].reshape(batch, -1)
            x_pair = torch.stack([x_neg, x_pos], dim=1)  # (batch, 2, num_l * Cin)

            # Linear: (batch, 2, num_l * Cin) -> (batch, 2, 2 * num_l * Cout)
            x_pair = linear(x_pair)

            # Split into 4 components for complex mixing
            out_half = num_l * self.out_channels
            x_pair = x_pair.view(batch, 4, out_half)
            x_r_0, x_i_0, x_r_1, x_i_1 = x_pair.unbind(dim=1)

            # Complex multiplication formula: y_r = a*x_r - b*x_i, y_i = b*x_r + a*x_i
            # With learned (a, b) encoded in linear weights:
            y_r = x_r_0 - x_i_1  # (batch, out_half)
            y_i = x_r_1 + x_i_0

            # Reshape back to (batch, num_l, Cout)
            y_neg = y_r.view(batch, num_l, self.out_channels)
            y_pos = y_i.view(batch, num_l, self.out_channels)

            out[:, neg_idx, :] = y_neg
            out[:, pos_idx, :] = y_pos

        return out.reshape(*lead, ebed_dim, self.out_channels)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SO2Linear",
            "@version": 1,  # keep 1 at devel stage
            "lmax": self.lmax,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "bias0": to_numpy_array(self.bias0),
            "linear_m0": {
                "weight": to_numpy_array(self.linear_m0.weight),
            },
            "linears_m": [
                {"weight": to_numpy_array(net.weight)} for net in self.linears_m
            ],
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

        lmax = int(data["lmax"])
        in_channels = int(data["in_channels"])
        out_channels = int(data["out_channels"])
        precision = data["precision"]
        dtype = PRECISION_DICT[precision]
        obj = cls(
            lmax=lmax,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype=dtype,
            seed=None,
            trainable=True,
        )
        obj.linear_m0.weight.data.copy_(
            torch.as_tensor(
                data["linear_m0"]["weight"], device=obj.device, dtype=obj.dtype
            )
        )
        obj.bias0.data.copy_(
            torch.as_tensor(data["bias0"], device=obj.device, dtype=obj.dtype)
        )
        for net, state in zip(obj.linears_m, data["linears_m"]):
            net.weight.data.copy_(
                torch.as_tensor(state["weight"], device=obj.device, dtype=obj.dtype)
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
    """

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        n_radial: int,
        radial_hidden: list[int],
        so2_layers: int = 1,
        activation_function: str,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.n_radial = int(n_radial)
        self.so2_layers = int(so2_layers)
        if self.so2_layers < 1:
            raise ValueError("`so2_layers` must be >= 1")
        self.ebed_dim = _so3_dim_of_lmax(self.lmax)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        # === Step 1. Multiple SO2Linear layers ===
        self.so2_linears = nn.ModuleList(
            [
                SO2Linear(
                    lmax=self.lmax,
                    in_channels=self.channels,
                    out_channels=self.channels,
                    dtype=self.dtype,
                    seed=seed,
                    trainable=trainable,
                )
                for _ in range(self.so2_layers)
            ]
        )

        # === Step 2. Intermediate activation (for layers 2+) ===
        if self.so2_layers > 1:
            self.intermediate_activations = nn.ModuleList(
                [
                    GatedActivation(
                        lmax=self.lmax, channels=self.channels, dtype=self.dtype
                    )
                    for _ in range(self.so2_layers - 1)
                ]
            )
        else:
            self.intermediate_activations = None

        # === Step 3. Radial weights per l and output channel: (E, (lmax+1)*C) ===
        self.radial_net = EmbeddingNet(
            self.n_radial,
            [*radial_hidden, (self.lmax + 1) * self.channels],
            activation_function=activation_function,
            precision=self.precision,
            resnet_dt=False,
            seed=seed,
            trainable=trainable,
        )

        self.register_buffer(
            "_degree_index",
            _build_degree_index(self.lmax, device=self.device),
            persistent=True,
        )

    def forward(self, x: torch.Tensor, edge_cache: Any) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Node features with shape (N, D, C) where D=(lmax+1)^2 is the SO(3) dimension.
        edge_cache
            Precomputed edge cache. Must be compatible with this block's lmax.

        Returns
        -------
        torch.Tensor
            Message updates with shape (N, D, C).
        """
        if edge_cache.num_edges == 0:
            return torch.zeros_like(x)

        src, dst = edge_cache.src, edge_cache.dst
        x_src = x[src]  # (E, D, C)

        # === Step 1. Rotate to edge-aligned local frame (block-wise) ===
        x_local_parts: list[torch.Tensor] = []
        start = 0
        for l in range(self.lmax + 1):
            dim = 2 * l + 1
            seg = x_src[:, start : start + dim, :]
            x_local_parts.append(
                torch.einsum("eij,ejc->eic", edge_cache.D_list[l], seg)
            )
            start += dim
        x_local = torch.cat(x_local_parts, dim=1)  # (E, D, C)

        # === Step 2. Radial interaction (per-l) + strict smooth cutoff ===
        rad = self.radial_net(edge_cache.edge_rbf)  # (E, (lmax+1)*C)
        rad = rad.view(edge_cache.num_edges, self.lmax + 1, self.channels)
        degree_idx = self._degree_index.to(device=x.device)[: self.ebed_dim]
        rad_expanded = rad[:, degree_idx, :]  # (E, D, C)

        # Apply smooth weight.
        edge_sw = edge_cache.edge_sw
        x_local = x_local * rad_expanded * edge_sw.unsqueeze(1)

        # === Step 3. Multi-layer SO(2) mixing ===
        for layer_idx, so2_linear in enumerate(self.so2_linears):
            x_local = so2_linear(x_local)

            # Bias correction for first layer to preserve strict smoothness at rcut.
            if layer_idx == 0:
                x_local[:, 0, :] = (
                    x_local[:, 0, :]
                    - so2_linear.bias0
                    + so2_linear.bias0 * rad_expanded[:, 0, :] * edge_sw
                )

            # Apply non-linearity between SO2 layers (not after the last).
            if (
                layer_idx < self.so2_layers - 1
                and self.intermediate_activations is not None
            ):
                x_local = self.intermediate_activations[layer_idx](x_local)

        # === Step 4. Rotate back to global frame (block-wise) ===
        x_global_parts: list[torch.Tensor] = []
        start = 0
        for l in range(self.lmax + 1):
            dim = 2 * l + 1
            seg = x_local[:, start : start + dim, :]
            x_global_parts.append(
                torch.einsum("eij,ejc->eic", edge_cache.Dt_list[l], seg)
            )
            start += dim
        x_global = torch.cat(x_global_parts, dim=1)  # (E, D, C)

        # === Step 5. Aggregate with neighbor normalization ===
        out = x.new_zeros(x.shape)
        out.index_add_(0, dst, x_global)
        out = out * edge_cache.inv_sqrt_deg
        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SO2Convolution",
            "@version": 1,
            "lmax": self.lmax,
            "channels": self.channels,
            "n_radial": self.n_radial,
            "so2_layers": self.so2_layers,
            "so2_linears": [net.serialize() for net in self.so2_linears],
            "radial_net": self.radial_net.serialize(),
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

        precision = data["precision"]
        dtype = PRECISION_DICT[precision]
        obj = cls(
            lmax=int(data["lmax"]),
            channels=int(data["channels"]),
            n_radial=int(data["n_radial"]),
            so2_layers=int(data.get("so2_layers", 1)),
            radial_hidden=[],
            activation_function="silu",
            dtype=dtype,
            seed=None,
            trainable=True,
        )

        # Restore SO2Linear layers
        so2_linears_data = data.get("so2_linears")
        if so2_linears_data is not None:
            for net, state in zip(obj.so2_linears, so2_linears_data):
                restored = SO2Linear.deserialize(state)
                net.load_state_dict(restored.state_dict())
        else:
            # Legacy format with single so2_linear
            if "so2_linear" in data:
                restored = SO2Linear.deserialize(data["so2_linear"])
                obj.so2_linears[0].load_state_dict(restored.state_dict())

        obj.radial_net = EmbeddingNet.deserialize(data["radial_net"])
        return obj


class EquivariantFFN(nn.Module):
    """
    Full equivariant FFN operating on all spherical harmonic degrees.

    Structure: per-degree linear (in) -> GatedActivation -> per-degree linear (out)

    GatedActivation serves as the unified "activation" for equivariant networks,
    analogous to SiLU in standard MLPs, but respecting SO(3) equivariance:
    - l=0: SiLU activation
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
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        hidden_channels: int,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.dtype = dtype
        self.device = env.DEVICE

        # === Per-degree input projection: C -> hidden ===
        self.linear_in = nn.ModuleList()
        for l in range(self.lmax + 1):
            self.linear_in.append(
                nn.Linear(
                    self.channels,
                    self.hidden_channels,
                    bias=(l == 0),
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        # === Equivariant activation ===
        self.act = GatedActivation(
            lmax=self.lmax, channels=self.hidden_channels, dtype=self.dtype
        )

        # === Per-degree output projection: hidden -> C ===
        self.linear_out = nn.ModuleList()
        for l in range(self.lmax + 1):
            self.linear_out.append(
                nn.Linear(
                    self.hidden_channels,
                    self.channels,
                    bias=(l == 0),
                    device=self.device,
                    dtype=self.dtype,
                )
            )

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
        N, D, C = x.shape

        # === Step 1. Per-degree input projection ===
        h = x.new_empty(N, D, self.hidden_channels)
        offset = 0
        for l, linear in enumerate(self.linear_in):
            dim = 2 * l + 1
            h[:, offset : offset + dim, :] = linear(x[:, offset : offset + dim, :])
            offset += dim

        # === Step 2. Equivariant activation ===
        h = self.act(h)

        # === Step 3. Per-degree output projection ===
        out = x.new_empty(N, D, C)
        offset = 0
        for l, linear in enumerate(self.linear_out):
            dim = 2 * l + 1
            out[:, offset : offset + dim, :] = linear(h[:, offset : offset + dim, :])
            offset += dim

        return out

    def serialize(self) -> dict[str, Any]:
        """Serialize EquivariantFFN parameters."""
        return {
            "@class": "EquivariantFFN",
            "@version": 1,
            "lmax": self.lmax,
            "channels": self.channels,
            "hidden_channels": self.hidden_channels,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "linear_in": [
                {
                    "weight": to_numpy_array(layer.weight),
                    "bias": to_numpy_array(layer.bias)
                    if layer.bias is not None
                    else None,
                }
                for layer in self.linear_in
            ],
            "linear_out": [
                {
                    "weight": to_numpy_array(layer.weight),
                    "bias": to_numpy_array(layer.bias)
                    if layer.bias is not None
                    else None,
                }
                for layer in self.linear_out
            ],
            "gate_linear": {
                "weight": to_numpy_array(self.act.gate_linear.weight),
                "bias": to_numpy_array(self.act.gate_linear.bias),
            }
            if self.act.gate_linear is not None
            else None,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EquivariantFFN:
        """Deserialize EquivariantFFN parameters."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "EquivariantFFN":
            raise ValueError(f"Invalid class for EquivariantFFN: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported EquivariantFFN version: {version}")

        precision = data["precision"]
        obj = cls(
            lmax=int(data["lmax"]),
            channels=int(data["channels"]),
            hidden_channels=int(data["hidden_channels"]),
            dtype=PRECISION_DICT[precision],
            trainable=True,
        )

        for layer, state in zip(obj.linear_in, data["linear_in"]):
            layer.weight.data.copy_(
                torch.as_tensor(state["weight"], device=obj.device, dtype=obj.dtype)
            )
            if state["bias"] is not None and layer.bias is not None:
                layer.bias.data.copy_(
                    torch.as_tensor(state["bias"], device=obj.device, dtype=obj.dtype)
                )

        for layer, state in zip(obj.linear_out, data["linear_out"]):
            layer.weight.data.copy_(
                torch.as_tensor(state["weight"], device=obj.device, dtype=obj.dtype)
            )
            if state["bias"] is not None and layer.bias is not None:
                layer.bias.data.copy_(
                    torch.as_tensor(state["bias"], device=obj.device, dtype=obj.dtype)
                )

        gate_linear_data = data["gate_linear"]
        if (gate_linear_data is None) != (obj.act.gate_linear is None):
            raise ValueError("EquivariantFFN gate_linear mismatch")

        if obj.act.gate_linear is not None:
            obj.act.gate_linear.weight.data.copy_(
                torch.as_tensor(
                    gate_linear_data["weight"], device=obj.device, dtype=obj.dtype
                )
            )
            obj.act.gate_linear.bias.data.copy_(
                torch.as_tensor(
                    gate_linear_data["bias"], device=obj.device, dtype=obj.dtype
                )
            )

        return obj


class SeZMInteractionBlock(nn.Module):
    """
    SeZM interaction block: pre-norm, SO(2) conv, gating, full equivariant FFN.

    The FFN operates on ALL degrees (l=0 to lmax), using a gated activation where
    scalar features (l=0) control the gating of higher-degree features (l>0).
    """

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        n_radial: int,
        radial_hidden: list[int],
        so2_layers: int = 2,
        ffn_neuron: list[int],
        activation_function: str,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.so2_layers = int(so2_layers)
        self.precision = RESERVED_PRECISION_DICT[dtype]

        self.norm = SeparableRMSNorm(self.lmax, self.channels, dtype=dtype)
        self.gating = GatedActivation(
            lmax=self.lmax, channels=self.channels, dtype=dtype
        )

        self.per_l_linear = PerDegreeLinearV2(
            lmax=self.lmax,
            channels=self.channels,
            dtype=dtype,
            trainable=trainable,
        )

        self.conv = SO2Convolution(
            lmax=self.lmax,
            channels=self.channels,
            n_radial=n_radial,
            radial_hidden=radial_hidden,
            so2_layers=self.so2_layers,
            activation_function=activation_function,
            dtype=dtype,
            seed=seed,
            trainable=trainable,
        )

        hidden_channels = ffn_neuron[0] if ffn_neuron else self.channels
        self.ffn = EquivariantFFN(
            lmax=self.lmax,
            channels=self.channels,
            hidden_channels=hidden_channels,
            dtype=dtype,
            trainable=trainable,
        )

    def forward(self, x: torch.Tensor, edge_cache: Any) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Features with shape (N, D, C).
        edge_cache
            Edge cache.

        Returns
        -------
        torch.Tensor
            Updated features with shape (N, D, C).
        """
        x = self.norm(x)
        x = self.per_l_linear(x)

        if edge_cache.num_edges > 0:
            x = x + self.conv(x, edge_cache)

        x = self.gating(x)
        x = x + self.ffn(x)
        return x
