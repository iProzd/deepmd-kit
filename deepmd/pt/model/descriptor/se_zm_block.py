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
    EmbeddingNet,
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
    to_numpy_array,
)

from .sel_zm_helper import EdgeFeatureCache  # noqa: TC001


def _so3_dim_of_lmax(lmax: int) -> int:
    """
    Return SO(3) representation dimension for given lmax.

    The dimension equals::

        sum_{l<=lmax} (2l+1) = (lmax+1)^2

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


def _map_degree_idx(lmax: int, *, device: torch.device) -> torch.Tensor:
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

    - l=0: Uses the specified activation function
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
        dtype: torch.dtype,
        activation_function: str = "silu",
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        self.scalar_act = ActivationFn(activation_function)
        self.gate_act = torch.nn.Sigmoid()

        # === Build expand_index for mapping per-l gates to all m components ===
        if self.lmax > 0:
            # expand_index[k] = l-1 for the k-th component in the l>0 portion
            # This maps each (l, m) component to its corresponding gate index
            expand_index = _map_degree_idx(self.lmax, device=self.device)[1:] - 1
            self.register_buffer("expand_index", expand_index, persistent=True)

            # Linear to generate lmax independent gates from scalar features
            self.gate_linear: nn.Module = MLPLayer(
                self.channels,
                self.lmax * self.channels,
                bias=True,
                activation_function="linear",
                precision=self.precision,
                seed=seed,
                trainable=trainable,
            )
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
        gating_scalars = self.gate_linear(x[:, 0, :])  # (N, lmax * C)
        gating_scalars = self.gate_act(gating_scalars)

        # Reshape to (N, lmax, C) then expand to (N, D-1, C)
        gating_scalars = gating_scalars.view(x.shape[0], self.lmax, -1)
        expand_idx = self.expand_index
        gates = torch.index_select(
            gating_scalars, dim=1, index=expand_idx
        )  # (N, D-1, C)

        # === Step 3. Apply gates to l>0 components ===
        xt = x[:, 1:, :] * gates

        return torch.cat([x0, xt], dim=1)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "GatedActivation",
            "@version": 1,
            "lmax": self.lmax,
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

        precision = data["precision"]
        dtype = PRECISION_DICT[precision]
        activation_function = data.get("activation_function", "silu")
        obj = cls(
            lmax=int(data["lmax"]),
            channels=int(data["channels"]),
            dtype=dtype,
            activation_function=activation_function,
            trainable=True,
            seed=None,
        )
        gate_linear_data = data["gate_linear"]
        if gate_linear_data is not None and obj.lmax > 0:
            obj.gate_linear = MLPLayer.deserialize(gate_linear_data)
        return obj


class SeparableRMSNorm(nn.Module):
    """
    Separable RMSNorm for scalar (l=0) and non-scalar (l>0) features.

    - l=0 and l>0 compute RMS separately (separable design).

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
        eps: float = 1e-10,
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
                expand_index = _map_degree_idx(self.lmax, device=self.device)[1:]
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
            "weight": to_numpy_array(self.weight) if self.weight is not None else None,
            "bias": to_numpy_array(self.bias) if self.bias is not None else None,
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

        precision = data["precision"]
        dtype = PRECISION_DICT[precision]
        obj = cls(
            lmax=int(data["lmax"]),
            channels=int(data["channels"]),
            affine=bool(data["affine"]),
            centering=bool(data["centering"]),
            eps=float(data.get("eps", 0.0)),
            dtype=dtype,
            trainable=True,
        )
        if obj.weight is not None and data["weight"] is not None:
            obj.weight.data.copy_(
                torch.as_tensor(data["weight"], device=obj.device, dtype=obj.dtype)
            )
        if obj.bias is not None and data["bias"] is not None:
            obj.bias.data.copy_(
                torch.as_tensor(data["bias"], device=obj.device, dtype=obj.dtype)
            )
        return obj


class PerDegreeLinear(nn.Module):
    """
    Degree-wise linear self-interaction shared across m within each l-block.

    NOTE
    ----
    - Each degree l has an independent C x C linear transformation.
    - Within each l-block, the same linear transformation is shared across all 2l+1 m components.
    - Bias is only applied to l=0 (scalar) components to preserve equivariance.
    - Uses Python loop over l, less efficient than PerDegreeLinearV2 but memory friendly.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    channels
        Number of channels per (l, m) coefficient.
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
        channels: int,
        dtype: torch.dtype,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        linears: list[MLPLayer] = []
        for l in range(self.lmax + 1):
            bias = l == 0
            linears.append(
                MLPLayer(
                    self.channels,
                    self.channels,
                    bias=bias,
                    activation_function="linear",
                    precision=self.precision,
                    seed=child_seed(seed, l),
                    trainable=trainable,
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
            "@version": 1,
            "lmax": self.lmax,
            "channels": self.channels,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "weights": [layer.serialize() for layer in self.linears],
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
            seed=None,
        )
        obj.linears = nn.ModuleList(
            [MLPLayer.deserialize(state) for state in data["weights"]]
        )
        return obj


class PerDegreeLinearV2(nn.Module):
    """
    Degree-wise linear self-interaction using einsum for efficiency.

    This is a vectorized version of PerDegreeLinear that avoids Python loops
    by using torch.einsum and index_select. The key insight is that weights
    are shared across all m components within each l-block.

    NOTE
    ----
    - weight shape: (lmax+1, C, C) - per-l linear transformation
    - bias shape: (C,) - only applied to l=0 (scalar) components
    - expand_index: maps each (l,m) position to its l value for weight lookup
    - Uses einsum 'bmi, lci -> blmc' pattern for batched per-l matmul
    - Avoids Python for-loops and torch.cat operations

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    channels
        Number of channels per (l, m) coefficient.
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
        channels: int,
        dtype: torch.dtype,
        trainable: bool,
        seed: int | list[int] | None = None,
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
        generator = get_generator(seed)
        nn.init.uniform_(self.weight, -bound, bound, generator=generator)

        # === Step 2. Bias only for l=0 (scalar components) ===
        self.bias = nn.Parameter(
            torch.zeros(self.channels, dtype=self.dtype, device=self.device)
        )

        # === Step 3. Precompute expand_index for weight lookup ===
        # expand_index[i] = l for position i in the packed (l,m) layout
        # This maps each (l,m) component to its corresponding weight index.
        self.register_buffer(
            "expand_index",
            _map_degree_idx(self.lmax, device=self.device),
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
            self.weight, dim=0, index=self.expand_index
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
            seed=None,
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

        # === Step 1. Mixing per |m| group, cross-l allowed, bias only for scalar index ===
        num_m0 = self.lmax + 1
        self.linear_m0 = MLPLayer(
            num_m0 * self.in_channels,
            num_m0 * self.out_channels,
            bias=False,
            activation_function="linear",
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
                activation_function="linear",
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
            Input with shape (N, D, Cin), where N is the number of atoms,
            D is the SO(3) dimension (lmax+1)^2, and Cin is input channels.

        Returns
        -------
        torch.Tensor
            Output with shape (N, D, Cout), where Cout is output channels.
        """
        n_atom, ebed_dim, Cin = x.shape
        out = x.new_zeros(n_atom, ebed_dim, self.out_channels)

        # === Step 1. Build packed indices on-the-fly (TorchScript-friendly) ===
        # Packed index mapping is: idx(l, m) = l*l + l + m, with m in [-l, l].
        l_values_m0 = torch.arange(0, self.lmax + 1, device=x.device, dtype=torch.long)
        m0_idx = l_values_m0 * l_values_m0 + l_values_m0

        # === Step 2. m = 0 group ===
        x_m0 = x[:, m0_idx, :].reshape(n_atom, -1)
        y_m0 = self.linear_m0(x_m0).reshape(n_atom, m0_idx.numel(), self.out_channels)
        out[:, m0_idx, :] = y_m0
        out[:, 0, :] = out[:, 0, :] + self.bias0

        # === Step 3. |m| > 0 groups (complex mixing via 2x2 block structure) ===
        for m, linear in enumerate(self.linears_m, start=1):
            l_values = torch.arange(m, self.lmax + 1, device=x.device, dtype=torch.long)
            pos_idx = l_values * l_values + l_values + m
            neg_idx = l_values * l_values + l_values - m
            num_l = pos_idx.numel()

            # Treat (neg, pos) as (Re, Im) complex pair
            x_neg = x[:, neg_idx, :].reshape(n_atom, -1)  # (n_atom, num_l * Cin)
            x_pos = x[:, pos_idx, :].reshape(n_atom, -1)
            x_pair = torch.stack([x_neg, x_pos], dim=1)  # (n_atom, 2, num_l * Cin)

            # Linear: (n_atom, 2, num_l * Cin) -> (n_atom, 2, 2 * num_l * Cout)
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
            y_neg = y_r.view(n_atom, num_l, self.out_channels)
            y_pos = y_i.view(n_atom, num_l, self.out_channels)

            out[:, neg_idx, :] = y_neg
            out[:, pos_idx, :] = y_pos

        return out  # shape already (n_atom, ebed_dim, self.out_channels)

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SO2Linear",
            "@version": 1,
            "lmax": self.lmax,
            "mmax": self.mmax,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "bias0": to_numpy_array(self.bias0),
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

        lmax = int(data["lmax"])
        mmax = int(data["mmax"])
        in_channels = int(data["in_channels"])
        out_channels = int(data["out_channels"])
        precision = data["precision"]
        dtype = PRECISION_DICT[precision]
        obj = cls(
            lmax=lmax,
            mmax=mmax,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype=dtype,
            seed=None,
            trainable=True,
        )
        obj.linear_m0 = MLPLayer.deserialize(data["linear_m0"])
        obj.bias0.data.copy_(
            torch.as_tensor(data["bias0"], device=obj.device, dtype=obj.dtype)
        )
        obj.linears_m = nn.ModuleList(
            [MLPLayer.deserialize(state) for state in data["linears_m"]]
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
        mmax: int | None = None,
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
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
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
                    mmax=self.mmax,
                    in_channels=self.channels,
                    out_channels=self.channels,
                    dtype=self.dtype,
                    seed=child_seed(seed, i),
                    trainable=trainable,
                )
                for i in range(self.so2_layers)
            ]
        )

        # === Step 2. Non-linearity Operators (for layers 2+) ===
        non_linearities: list[nn.Module] = []
        for i in range(max(0, self.so2_layers - 1)):
            non_linearities.append(
                GatedActivation(
                    lmax=self.lmax,
                    channels=self.channels,
                    dtype=self.dtype,
                    trainable=trainable,
                    seed=child_seed(seed, self.so2_layers + i),
                )
            )
        non_linearities.append(nn.Identity())
        self.non_linearities = nn.ModuleList(non_linearities)

        # === Step 3. Radial weights per l and output channel: (E, (lmax+1)*C) ===
        self.radial_net = EmbeddingNet(
            self.n_radial + 2 * self.channels,
            [*radial_hidden, (self.lmax + 1) * self.channels],
            activation_function=activation_function,
            precision=self.precision,
            resnet_dt=False,
            seed=child_seed(seed, 2 * self.so2_layers),
            trainable=trainable,
        )

        self.register_buffer(
            "degree_index",
            _map_degree_idx(self.lmax, device=self.device),
            persistent=True,
        )

    def forward(self, x: torch.Tensor, edge_cache: EdgeFeatureCache) -> torch.Tensor:
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
        num_edges = edge_cache.src.size(0)
        if num_edges == 0:
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
        # With atom type embeddings as additional edge features.
        src_type_feat = edge_cache.node_type_feat.index_select(0, src)
        dst_type_feat = edge_cache.node_type_feat.index_select(0, dst)
        rad_in = torch.cat(
            [edge_cache.edge_rbf, src_type_feat, dst_type_feat],
            dim=-1,
        )  # (E, n_rbf + 2C)
        rad = self.radial_net(rad_in)  # (E, (lmax+1)*C)
        rad = rad.view(num_edges, self.lmax + 1, self.channels)
        degree_idx = self.degree_index[: self.ebed_dim]  # for pyramid l_schedule
        rad_expanded = rad[:, degree_idx, :]  # (E, D, C)

        # Apply smooth weight.
        edge_sw = edge_cache.edge_sw
        rad_feat = rad_expanded * edge_sw.unsqueeze(1)  # (E, D, C)
        x_local = x_local * rad_feat  # (E, D, C)

        # === Step 3. Multi-layer SO(2) mixing ===
        for layer_idx, (so2_linear, non_linear) in enumerate(
            zip(self.so2_linears, self.non_linearities)
        ):
            x_local = so2_linear(x_local)

            # Bias correction for first layer to preserve strict smoothness at rcut.
            if layer_idx == 0:
                x_local[:, 0, :] = (
                    x_local[:, 0, :]
                    - so2_linear.bias0
                    + so2_linear.bias0 * rad_expanded[:, 0, :] * edge_sw
                )

            # Apply non-linearity between SO2 layers
            # The last non-linearity is Identity by construction.
            x_local = non_linear(x_local)

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
            "mmax": self.mmax,
            "channels": self.channels,
            "n_radial": self.n_radial,
            "so2_layers": self.so2_layers,
            "so2_linears": [net.serialize() for net in self.so2_linears],
            "non_linearities": (
                [act.serialize() for act in self.non_linearities[:-1]]
                if self.so2_layers > 1
                else None
            ),
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
            mmax=int(data["mmax"]),
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

        # Restore non-linearity operators
        non_linearities_data = data.get("non_linearities")
        if non_linearities_data is not None:
            for act, state in zip(obj.non_linearities[:-1], non_linearities_data):
                restored = GatedActivation.deserialize(state)
                act.load_state_dict(restored.state_dict())

        obj.radial_net = EmbeddingNet.deserialize(data["radial_net"])
        return obj


class EquivariantFFN(nn.Module):
    """
    Full equivariant FFN operating on all spherical harmonic degrees.

    Structure: per-degree linear (in) -> GatedActivation -> per-degree linear (out)

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
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        # === Per-degree input projection: C -> hidden ===
        self.linear_in = nn.ModuleList()
        for l in range(self.lmax + 1):
            self.linear_in.append(
                MLPLayer(
                    self.channels,
                    self.hidden_channels,
                    bias=(l == 0),
                    activation_function="linear",
                    precision=self.precision,
                    seed=child_seed(seed, l),
                    trainable=trainable,
                )
            )

        # === Equivariant activation ===
        self.act = GatedActivation(
            lmax=self.lmax,
            channels=self.hidden_channels,
            dtype=dtype,
            activation_function=activation_function,
            trainable=trainable,
            seed=child_seed(seed, self.lmax + 1),
        )

        # === Per-degree output projection: hidden -> C ===
        self.linear_out = nn.ModuleList()
        for l in range(self.lmax + 1):
            self.linear_out.append(
                MLPLayer(
                    self.hidden_channels,
                    self.channels,
                    bias=(l == 0),
                    activation_function="linear",
                    precision=self.precision,
                    seed=child_seed(seed, self.lmax + 1 + l),
                    trainable=trainable,
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
        return {
            "@class": "EquivariantFFN",
            "@version": 1,
            "lmax": self.lmax,
            "channels": self.channels,
            "hidden_channels": self.hidden_channels,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "activation_function": self.act.scalar_act.activation,
            "linear_in": [layer.serialize() for layer in self.linear_in],
            "linear_out": [layer.serialize() for layer in self.linear_out],
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

        precision = data["precision"]
        activation_function = data.get("activation_function", "silu")
        obj = cls(
            lmax=int(data["lmax"]),
            channels=int(data["channels"]),
            hidden_channels=int(data["hidden_channels"]),
            dtype=PRECISION_DICT[precision],
            activation_function=activation_function,
            trainable=True,
            seed=None,
        )

        obj.linear_in = nn.ModuleList(
            [MLPLayer.deserialize(state) for state in data["linear_in"]]
        )
        obj.linear_out = nn.ModuleList(
            [MLPLayer.deserialize(state) for state in data["linear_out"]]
        )
        obj.act = GatedActivation.deserialize(data["act"])
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
    n_radial
        Number of radial basis functions.
    radial_hidden
        Hidden layer sizes for radial network.
    so2_layers
        Number of SO(2) mixing layers.
    ffn_neuron
        Hidden layer sizes for FFN.
    activation_function
        Activation function for l=0 components.
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
        n_radial: int,
        radial_hidden: list[int],
        so2_layers: int = 2,
        ffn_neuron: list[int],
        activation_function: str,
        eps: float = 1e-10,
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
        self.so2_layers = int(so2_layers)
        self.dtype = dtype
        self.precision = RESERVED_PRECISION_DICT[dtype]

        # === Step 0. Truncation mask for |m| > mmax ===
        # The packed layout is l-primary: for each l, coefficients are ordered by m=-l..+l.
        # When mmax < lmax, coefficients with |m| > mmax are physically discarded.
        ebed_dim = _so3_dim_of_lmax(self.lmax)
        m_mask = torch.ones(ebed_dim, device=env.DEVICE, dtype=self.dtype)
        if self.mmax < self.lmax:
            start = 0
            for l in range(self.lmax + 1):
                dim = 2 * l + 1
                m = torch.arange(-l, l + 1, device=env.DEVICE, dtype=torch.long)
                keep = (m.abs() <= self.mmax).to(dtype=self.dtype)
                m_mask[start : start + dim] = keep
                start += dim
        self.register_buffer("m_mask", m_mask.view(1, -1, 1), persistent=True)

        self.norm = SeparableRMSNorm(
            self.lmax,
            self.channels,
            eps=eps,
            dtype=dtype,
            trainable=trainable,
        )
        self.gating = GatedActivation(
            lmax=self.lmax,
            channels=self.channels,
            dtype=dtype,
            activation_function=activation_function,
            trainable=trainable,
            seed=child_seed(seed, 0),
        )

        self.per_l_linear = PerDegreeLinearV2(
            lmax=self.lmax,
            channels=self.channels,
            dtype=dtype,
            trainable=trainable,
            seed=child_seed(seed, 1),
        )

        self.conv = SO2Convolution(
            lmax=self.lmax,
            mmax=self.mmax,
            channels=self.channels,
            n_radial=n_radial,
            radial_hidden=radial_hidden,
            so2_layers=self.so2_layers,
            activation_function=activation_function,
            dtype=dtype,
            seed=child_seed(seed, 2),
            trainable=trainable,
        )

        hidden_channels = ffn_neuron[0] if ffn_neuron else self.channels
        self.ffn = EquivariantFFN(
            lmax=self.lmax,
            channels=self.channels,
            hidden_channels=hidden_channels,
            dtype=dtype,
            activation_function=activation_function,
            trainable=trainable,
            seed=child_seed(seed, 3),
        )

    def forward(self, x: torch.Tensor, edge_cache: EdgeFeatureCache) -> torch.Tensor:
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
        if self.mmax < self.lmax:
            x = x * self.m_mask
        x_norm = self.norm(x)
        x_norm = self.per_l_linear(x_norm)

        if edge_cache.src.numel() > 0:
            x = x + self.conv(x_norm, edge_cache)

        x = self.gating(x)
        x = x + self.ffn(x)
        return x

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SeZMInteractionBlock",
            "@version": 1,
            "lmax": self.lmax,
            "mmax": self.mmax,
            "channels": self.channels,
            "n_radial": self.conv.n_radial,
            "so2_layers": self.so2_layers,
            "ffn_neuron": [self.ffn.hidden_channels],
            "activation_function": self.gating.scalar_act.activation,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "norm": self.norm.serialize(),
            "gating": self.gating.serialize(),
            "per_l_linear": self.per_l_linear.serialize(),
            "conv": self.conv.serialize(),
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

        precision = data["precision"]
        dtype = PRECISION_DICT[precision]
        activation_function = data.get("activation_function", "silu")
        obj = cls(
            lmax=int(data["lmax"]),
            mmax=int(data["mmax"]),
            channels=int(data["channels"]),
            n_radial=int(data["n_radial"]),
            radial_hidden=[],
            so2_layers=int(data["so2_layers"]),
            ffn_neuron=data["ffn_neuron"],
            activation_function=activation_function,
            dtype=dtype,
            seed=None,
            trainable=True,
        )

        obj.norm = SeparableRMSNorm.deserialize(data["norm"])
        obj.gating = GatedActivation.deserialize(data["gating"])
        obj.per_l_linear = PerDegreeLinearV2.deserialize(data["per_l_linear"])
        obj.conv = SO2Convolution.deserialize(data["conv"])
        obj.ffn = EquivariantFFN.deserialize(data["ffn"])
        return obj
