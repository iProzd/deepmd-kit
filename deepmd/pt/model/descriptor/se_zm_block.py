# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SeZM-Net interaction blocks for DeePMD-kit (PyTorch backend).

This module contains the per-block message passing and nonlinearities used by
`DescrptSeZMNet`.

Design notes
------------
- The caller is responsible for building an `EdgeFeatureCache` once per forward
  pass and reusing it across blocks.
- Features use packed (l, m) layout with K=(lmax+1)^2.
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


def _k_of_lmax(lmax: int) -> int:
    """Return K = sum_{l<=lmax} (2l+1) = (lmax+1)^2."""
    return int((int(lmax) + 1) ** 2)


def _build_l_of_k(lmax: int, *, device: torch.device) -> torch.Tensor:
    """
    Build `l_of_k` mapping for (l, m) packed layout.

    Layout convention:
    - coefficients are packed by increasing l
    - within each l-block, m is ordered from -l .. +l

    Parameters
    ----------
    lmax
        Maximum order.
    device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Integer tensor with shape (K,), where K=(lmax+1)^2 and `l_of_k[k] == l`.
    """
    lmax = int(lmax)
    counts = torch.tensor(
        [2 * l + 1 for l in range(lmax + 1)], device=device, dtype=torch.long
    )
    return torch.repeat_interleave(
        torch.arange(lmax + 1, device=device, dtype=torch.long), counts
    )


class SeparableRMSNorm(nn.Module):
    """
    Separable RMSNorm for scalar (l=0) and non-scalar (l>0) features.

    - Scalar part has affine scale (per-channel).
    - Non-scalar part has no affine parameters (speed + stability).
    """

    def __init__(
        self,
        channels: int,
        *,
        dtype: torch.dtype,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.dtype = dtype
        self.device = env.DEVICE
        self.weight0 = nn.Parameter(
            torch.ones(
                self.channels,
                dtype=self.dtype,
                device=self.device,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Features with shape (N, K, C).

        Returns
        -------
        torch.Tensor
            Normalized features with shape (N, K, C).
        """
        x0 = x[:, :1, :]
        xt = x[:, 1:, :]

        rms0 = torch.sqrt(torch.mean(x0 * x0, dim=-1, keepdim=True) + self.eps)
        x0 = (x0 / rms0) * self.weight0.view(1, 1, -1)

        if xt.numel() == 0:
            return x0

        rmst = torch.sqrt(torch.mean(xt * xt, dim=(1, 2), keepdim=True) + self.eps)
        xt = xt / rmst
        return torch.cat([x0, xt], dim=1)


class AnalyticGating(nn.Module):
    """Analytic gating: gate derived from l=0 controls l>0."""

    def __init__(
        self,
        channels: int,
        *,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.gate = EmbeddingNet(
            self.channels,
            [self.channels],
            activation_function="Linear",
            precision=self.precision,
            resnet_dt=False,
            seed=seed,
            trainable=trainable,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Features with shape (N, K, C).

        Returns
        -------
        torch.Tensor
            Gated features with shape (N, K, C).
        """
        x0 = x[:, 0, :]
        gate = torch.sigmoid(self.gate(x0))
        if x.shape[1] == 1:
            return x
        xt = x[:, 1:, :] * gate.unsqueeze(1)
        return torch.cat([x0.unsqueeze(1), xt], dim=1)


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
        self.K = _k_of_lmax(self.lmax)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        # === Step 1. Precompute packed indices (buffers) ===
        m0 = [l * l + l for l in range(self.lmax + 1)]
        self.register_buffer(
            "_m0_idx",
            torch.tensor(m0, device=torch.device("cpu"), dtype=torch.long),
            persistent=True,
        )

        for m in range(1, self.lmax + 1):
            pos_idx = [l * l + l + m for l in range(m, self.lmax + 1)]
            neg_idx = [l * l + l - m for l in range(m, self.lmax + 1)]
            self.register_buffer(
                f"_pos_idx_{m}",
                torch.tensor(pos_idx, device=torch.device("cpu"), dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                f"_neg_idx_{m}",
                torch.tensor(neg_idx, device=torch.device("cpu"), dtype=torch.long),
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

        self.linears_m: nn.ModuleList = nn.ModuleList()
        for m in range(1, self.lmax + 1):
            num_l = self.lmax - m + 1
            self.linears_m.append(
                nn.Linear(
                    num_l * self.in_channels,
                    num_l * self.out_channels,
                    bias=False,
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
            Input with shape (..., K, Cin).

        Returns
        -------
        torch.Tensor
            Output with shape (..., K, Cout).
        """
        *lead, K, Cin = x.shape
        if K != self.K or Cin != self.in_channels:
            raise ValueError(
                f"SO2Linear expects (..., {self.K}, {self.in_channels}), got {tuple(x.shape)}"
            )

        batch = math.prod(lead) if lead else 1
        x2 = x.reshape(batch, K, Cin)
        out = x2.new_zeros(batch, K, self.out_channels)

        # m = 0 group
        m0_idx = self._m0_idx.to(device=x.device)
        x_m0 = x2[:, m0_idx, :].reshape(batch, -1)
        y_m0 = self.linear_m0(x_m0).reshape(batch, m0_idx.numel(), self.out_channels)
        out[:, m0_idx, :] = y_m0
        out[:, 0, :] = out[:, 0, :] + self.bias0

        # |m| > 0 groups
        for m, linear in enumerate(self.linears_m, start=1):
            pos_idx = getattr(self, f"_pos_idx_{m}").to(device=x.device)
            neg_idx = getattr(self, f"_neg_idx_{m}").to(device=x.device)

            x_pos = x2[:, pos_idx, :].reshape(batch, -1)
            x_neg = x2[:, neg_idx, :].reshape(batch, -1)
            y_pos = linear(x_pos).reshape(batch, pos_idx.numel(), self.out_channels)
            y_neg = linear(x_neg).reshape(batch, neg_idx.numel(), self.out_channels)

            out[:, pos_idx, :] = y_pos
            out[:, neg_idx, :] = y_neg

        return out.reshape(*lead, K, self.out_channels)

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
    """Linearized SO(2) convolution with precomputed edge cache."""

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        n_radial: int,
        radial_hidden: list[int],
        activation_function: str,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
        neighbor_norm: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.n_radial = int(n_radial)
        self.neighbor_norm = bool(neighbor_norm)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        self.so2_linear = SO2Linear(
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=self.channels,
            dtype=self.dtype,
            seed=seed,
            trainable=trainable,
        )

        # Radial weights per l and output channel: (E, (lmax+1)*C)
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
            "_l_of_k",
            _build_l_of_k(self.lmax, device=torch.device("cpu")),
            persistent=True,
        )
        self.K = _k_of_lmax(self.lmax)

    def forward(self, x: torch.Tensor, cache: Any) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Node features with shape (N, K, C) where K=(lmax+1)^2.
        cache
            Precomputed edge cache. Must be compatible with this block's lmax.

        Returns
        -------
        torch.Tensor
            Message updates with shape (N, K, C).
        """
        if x.shape[1] != self.K:
            raise ValueError(f"SO2Convolution expects K={self.K}, got K={x.shape[1]}")
        if cache.num_edges == 0:
            return torch.zeros_like(x)

        src, dst = cache.src, cache.dst
        x_src = x[src]  # (E, K, C)

        # === Step 1. Rotate to edge-aligned local frame (block-wise) ===
        x_local_parts: list[torch.Tensor] = []
        start = 0
        for l in range(self.lmax + 1):
            dim = 2 * l + 1
            seg = x_src[:, start : start + dim, :]
            x_local_parts.append(torch.einsum("eij,ejc->eic", cache.Dt_list[l], seg))
            start += dim
        x_local = torch.cat(x_local_parts, dim=1)  # (E, K, C)

        # === Step 2. Radial interaction (per-l) + strict smooth cutoff ===
        rad = self.radial_net(cache.edge_rbf)  # (E, (lmax+1)*C)
        rad = rad.view(cache.num_edges, self.lmax + 1, self.channels)
        l_of_k = self._l_of_k.to(device=x.device)[: self.K]
        rad_expanded = rad[:, l_of_k, :]  # (E, K, C)

        # Strict smooth cutoff: multiply message by envelope and smooth weight explicitly.
        edge_weight = cache.edge_envelop * cache.edge_sw
        x_local = x_local * rad_expanded * edge_weight.unsqueeze(1)

        # === Step 3. SO(2) mixing (m-grouped) ===
        x_local = self.so2_linear(x_local)
        # SO2Linear adds a constant bias to the scalar (l=0, m=0) channel. Scale it
        # by radial weights and smooth cutoff to preserve strict smoothness at rcut.
        x_local[:, 0, :] = (
            x_local[:, 0, :]
            - self.so2_linear.bias0
            + self.so2_linear.bias0 * rad_expanded[:, 0, :] * edge_weight
        )

        # === Step 4. Rotate back to global frame (block-wise) ===
        x_global_parts: list[torch.Tensor] = []
        start = 0
        for l in range(self.lmax + 1):
            dim = 2 * l + 1
            seg = x_local[:, start : start + dim, :]
            x_global_parts.append(torch.einsum("eij,ejc->eic", cache.D_list[l], seg))
            start += dim
        x_global = torch.cat(x_global_parts, dim=1)  # (E, K, C)

        # === Step 5. Aggregate ===
        out = x.new_zeros(x.shape)
        out.index_add_(0, dst, x_global)
        if self.neighbor_norm and cache.inv_sqrt_deg is not None:
            out = out * cache.inv_sqrt_deg
        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "SO2Convolution",
            "@version": 1,  # keep 1 at devel stage
            "lmax": self.lmax,
            "channels": self.channels,
            "n_radial": self.n_radial,
            "so2_linear": self.so2_linear.serialize(),
            "radial_net": self.radial_net.serialize(),
            "neighbor_norm": self.neighbor_norm,
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
            radial_hidden=[],
            activation_function="silu",
            dtype=dtype,
            seed=None,
            trainable=True,
            neighbor_norm=bool(data["neighbor_norm"]),
        )
        obj.so2_linear = SO2Linear.deserialize(data["so2_linear"])
        obj.radial_net = EmbeddingNet.deserialize(data["radial_net"])
        return obj


class PerDegreeLinear(nn.Module):
    """Order-wise linear self-interaction shared across m within each l-block."""

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
        self.K = _k_of_lmax(self.lmax)
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
            Input features with shape (N, K, C) where K=(lmax+1)^2.

        Returns
        -------
        torch.Tensor
            Order-wise mixed features with shape (N, K, C).
        """
        if x.shape[1] != self.K:
            raise ValueError(f"PerDegreeLinear expects K={self.K}, got {x.shape[1]}")
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
        if data_cls != "PerDegreeLinear":
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


class EquivariantFFN(nn.Module):
    """
    Full equivariant FFN operating on all spherical harmonic orders.

    This module applies a gated nonlinear feedforward network to features of
    all orders (l=0 to lmax), following the NequIP/eSEN gating pattern:

    1. Per-order linear mixing (input projection)
    2. Separable gating: a dedicated l=0 branch produces gates for l>0 via sigmoid,
       while a separate l=0 content branch passes through SiLU.
    3. Per-order linear mixing (output projection)

    The architecture ensures equivariance by only applying pointwise nonlinearities
    to scalar (l=0) features, while l>0 features are linearly transformed and gated.

    Parameters
    ----------
    lmax
        Maximum order.
    channels
        Number of channels per (l, m) coefficient.
    hidden_channels
        Hidden dimension for the scalar MLP and gating.
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
        self.K = _k_of_lmax(self.lmax)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        # === Step 1. Per-order input projection: C -> hidden ===
        self.linear_in = nn.ModuleList()
        for l in range(self.lmax + 1):
            bias = l == 0  # bias only for scalar
            self.linear_in.append(
                nn.Linear(
                    self.channels,
                    self.hidden_channels,
                    bias=bias,
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        # === Step 2. Scalar activation + gating linear ===
        self.scalar_act = nn.SiLU()
        if self.lmax > 0:
            self.gate_linear = nn.Linear(
                self.channels,
                self.lmax * self.hidden_channels,
                bias=True,
                device=self.device,
                dtype=self.dtype,
            )
            num_nonscalar = self.K - 1
            expand_index = torch.zeros(num_nonscalar, dtype=torch.long, device="cpu")
            offset = 0
            for l in range(1, self.lmax + 1):
                dim = 2 * l + 1
                expand_index[offset : offset + dim] = l - 1
                offset += dim
            self.register_buffer("_expand_index", expand_index, persistent=True)
        else:
            self.gate_linear = None
            self._expand_index = None

        # === Step 3. Per-order output projection: hidden -> C ===
        self.linear_out = nn.ModuleList()
        for l in range(self.lmax + 1):
            bias = l == 0
            self.linear_out.append(
                nn.Linear(
                    self.hidden_channels,
                    self.channels,
                    bias=bias,
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
            Input features with shape (N, K, C) where K=(lmax+1)^2.

        Returns
        -------
        torch.Tensor
            Output features with shape (N, K, C).
        """
        N, K, C = x.shape
        if K != self.K:
            raise ValueError(f"EquivariantFFN expects K={self.K}, got {K}")

        # === Step 1. Per-order input projection ===
        h = x.new_empty(N, K, self.hidden_channels)
        offset = 0
        for l, linear in enumerate(self.linear_in):
            dim = 2 * l + 1
            seg = x[:, offset : offset + dim, :]
            h[:, offset : offset + dim, :] = linear(seg)
            offset += dim

        # === Step 2. Gated activation ===
        h0 = self.scalar_act(h[:, 0:1, :])

        if (
            self.lmax > 0
            and self.gate_linear is not None
            and self._expand_index is not None
        ):
            gate_input = x[:, 0, :]
            gates = torch.sigmoid(self.gate_linear(gate_input))
            gates = gates.view(N, self.lmax, self.hidden_channels)

            expand_idx = self._expand_index.to(device=x.device)
            gates_expanded = gates[:, expand_idx, :]
            ht = h[:, 1:, :] * gates_expanded
            h = torch.cat([h0, ht], dim=1)
        else:
            h = h0

        # === Step 3. Per-order output projection ===
        out = x.new_empty(N, K, C)
        offset = 0
        for l, linear in enumerate(self.linear_out):
            dim = 2 * l + 1
            seg = h[:, offset : offset + dim, :]
            out[:, offset : offset + dim, :] = linear(seg)
            offset += dim

        return out

    def serialize(self) -> dict[str, Any]:
        """Serialize EquivariantFFN parameters."""
        return {
            "@class": "EquivariantFFN",
            "@version": 1,  # keep 1 at devel stage
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
                "weight": to_numpy_array(self.gate_linear.weight),
                "bias": to_numpy_array(self.gate_linear.bias),
            }
            if self.gate_linear is not None
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
        if (gate_linear_data is None) != (obj.gate_linear is None):
            raise ValueError("EquivariantFFN gate_linear mismatch")

        if obj.gate_linear is not None:
            obj.gate_linear.weight.data.copy_(
                torch.as_tensor(
                    gate_linear_data["weight"], device=obj.device, dtype=obj.dtype
                )
            )
            obj.gate_linear.bias.data.copy_(
                torch.as_tensor(
                    gate_linear_data["bias"], device=obj.device, dtype=obj.dtype
                )
            )

        return obj


class SeZMInteractionBlock(nn.Module):
    """
    SeZM interaction block: pre-norm, SO(2) conv, gating, full equivariant FFN.

    The FFN operates on ALL orders (l=0 to lmax), using a gated activation where
    scalar features (l=0) control the gating of higher-order features (l>0).
    """

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        n_radial: int,
        radial_hidden: list[int],
        ffn_neuron: list[int],
        activation_function: str,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
        neighbor_norm: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.neighbor_norm = bool(neighbor_norm)
        self.K = _k_of_lmax(self.lmax)
        self.precision = RESERVED_PRECISION_DICT[dtype]

        self.norm = SeparableRMSNorm(self.channels, dtype=dtype)
        self.gating = AnalyticGating(
            self.channels, dtype=dtype, seed=seed, trainable=trainable
        )

        self.pre_linear = PerDegreeLinear(
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
            activation_function=activation_function,
            dtype=dtype,
            seed=seed,
            trainable=trainable,
            neighbor_norm=self.neighbor_norm,
        )

        hidden_channels = ffn_neuron[0] if ffn_neuron else self.channels
        self.ffn = EquivariantFFN(
            lmax=self.lmax,
            channels=self.channels,
            hidden_channels=hidden_channels,
            dtype=dtype,
            trainable=trainable,
        )

    def forward(self, x: torch.Tensor, cache: Any) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Features with shape (N, K, C).
        cache
            Edge cache.

        Returns
        -------
        torch.Tensor
            Updated features with shape (N, K, C).
        """
        x = self.norm(x)
        x = self.pre_linear(x)

        if cache.num_edges > 0:
            x = x + self.conv(x, cache)

        x = self.gating(x)
        x = x + self.ffn(x)
        return x
