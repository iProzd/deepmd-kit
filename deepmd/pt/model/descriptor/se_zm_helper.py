# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Helper utilities for SeZM-Net PyTorch descriptor.

This module collects shared building blocks and math utilities, including:
- edge caches, radial bases/envelopes, and initial embeddings
- edge-frame construction and Wigner-D rotation blocks
- SO(3) index/projection helpers and dtype/serialization utilities
"""

from __future__ import (
    annotations,
)

import math
from collections.abc import (
    Generator,
)
from contextlib import (
    contextmanager,
)
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import (
    rearrange,
)

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
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

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
    )


def init_trunc_normal_fan_in_out(
    weight: torch.Tensor, seed: int | list[int] | None
) -> None:
    """Initialize weight with truncated normal distribution.

    Uses Xavier-like variance scaling: std = 1.0 / sqrt(fan_in + fan_out).
    Truncation at +/-3*std prevents extreme outliers.

    Parameters
    ----------
    weight : torch.Tensor
        Weight tensor with shape (fan_in, fan_out).
    seed : int | list[int] | None
        Random seed for reproducibility.
    """
    fan_in, fan_out = weight.shape
    std = 1.0 / math.sqrt(fan_in + fan_out)
    nn.init.trunc_normal_(
        weight,
        mean=0.0,
        std=std,
        a=-3.0 * std,
        b=3.0 * std,
        generator=get_generator(seed),
    )


def build_edge_type_feat(
    type_ebed: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    num_edges: int | None = None,
) -> torch.Tensor:
    """
    Build per-edge type features by summing src/dst embeddings.

    Parameters
    ----------
    type_ebed
        Per-node type embedding with shape (N, C).
    src
        Source node indices with shape (E,).
    dst
        Destination node indices with shape (E,).
    num_edges
        Number of edges. When provided, avoids calling `numel()` on GPU tensors.

    Returns
    -------
    torch.Tensor
        Per-edge type features with shape (E, C).
    """
    if num_edges is None:
        num_edges = int(src.numel())
    if num_edges == 0:
        return type_ebed.new_empty((0, type_ebed.shape[1]))

    # === Step 1. Normalize index dtypes ===
    if src.dtype != torch.long:
        src = src.to(dtype=torch.long)
    if dst.dtype != torch.long:
        dst = dst.to(dtype=torch.long)

    # === Step 2. Pack edge indices as a 2-item bag ===
    pair_index = torch.stack((src, dst), dim=1).reshape(-1)

    # === Step 3. Reduce with a single embedding_bag ===
    offsets = torch.arange(
        0, 2 * num_edges, step=2, device=src.device, dtype=torch.long
    )
    return F.embedding_bag(
        pair_index,
        type_ebed,
        offsets,
        mode="sum",
        include_last_offset=False,
    )


class EdgeFeatureCache(NamedTuple):
    """
    Global edge feature cache created once per forward().

    All tensors are aligned on the same edge axis (E = number of valid edges).

    Parameters
    ----------
    src
        Source node indices with shape (E,).
    dst
        Destination node indices with shape (E,).
    edge_type_feat
        Per-edge type embeddings with shape (E, C), computed as src+dst.
    edge_vec
        Edge vectors with shape (E, 3) in Å.
    edge_rbf
        Radial basis with shape (E, n_radial).
        The C^2 cutoff envelope is already baked in.
    edge_env
        C^2 cutoff envelope weights with shape (E, 1).
    deg
        Destination node degree (number of incoming edges) with shape (N,).
        Used for neighbor normalization in EnvironmentInitialEmbedding.
    inv_sqrt_deg
        Destination degree normalization with shape (N, 1, 1).
    D_full
        Block-diagonal Wigner-D matrix with shape (E, D, D) where D=(lmax+1)^2.
        Used for efficient batched rotation. None if not available.
    Dt_full
        Transpose of D_full with shape (E, D, D). None if not available.
    D_to_m_cache
        Lazy cache for projected D matrices keyed by (lmax, mmax).
    Dt_from_m_cache
        Lazy cache for projected Dt matrices keyed by (lmax, mmax).
    """

    src: torch.Tensor
    dst: torch.Tensor
    edge_type_feat: torch.Tensor
    edge_vec: torch.Tensor
    edge_rbf: torch.Tensor
    edge_env: torch.Tensor
    deg: torch.Tensor
    inv_sqrt_deg: torch.Tensor
    D_full: torch.Tensor | None = None
    Dt_full: torch.Tensor | None = None
    D_to_m_cache: dict[str, torch.Tensor] | None = None
    Dt_from_m_cache: dict[str, torch.Tensor] | None = None

    def get_D_to_m(
        self,
        *,
        ebed_dim_full: int,
        coeff_index_m: torch.Tensor,
        key_lmax: int,
        key_mmax: int,
    ) -> torch.Tensor:
        """
        Fetch (or build once) the row-projected Wigner-D blocks for m-major layout.

        This selects the subset of rows needed for the m-major truncated layout,
        caches the result keyed by (lmax, mmax), and reuses it across blocks.

        Parameters
        ----------
        ebed_dim_full
            Full SO(3) dimension D=(lmax+1)^2 used to slice the block-diagonal
            Wigner matrix.
        coeff_index_m
            Indices for the m-major reduced layout with shape (D_m_trunc,).
        key_lmax
            lmax used to build ``coeff_index_m`` (cache key).
        key_mmax
            mmax used to build ``coeff_index_m`` (cache key).

        Returns
        -------
        torch.Tensor
            Projected rotation matrix with shape (E, D_m_trunc, D).
        """
        cache_key = f"{int(key_lmax)}:{int(key_mmax)}"
        cache_dict = self.D_to_m_cache
        if cache_dict is None:
            raise ValueError("EdgeFeatureCache.D_to_m_cache is None")
        cached = cache_dict.get(cache_key)
        if cached is not None:
            return cached

        D_full = self.D_full  # (E, D, D)
        if D_full is None:
            raise ValueError("EdgeFeatureCache.D_full is None")
        D_block = D_full[:, :ebed_dim_full, :ebed_dim_full]  # (E, D, D)
        D_to_m = D_block.index_select(1, coeff_index_m)  # (E, D_m_trunc, D)
        cache_dict[cache_key] = D_to_m
        return D_to_m

    def get_Dt_from_m(
        self,
        *,
        ebed_dim_full: int,
        coeff_index_m: torch.Tensor,
        key_lmax: int,
        key_mmax: int,
    ) -> torch.Tensor:
        """
        Fetch (or build once) the column-projected Wigner-D^T blocks for inverse rotation.

        This selects the subset of columns needed for the m-major truncated layout,
        caches the result keyed by (lmax, mmax), and reuses it across blocks.

        Parameters
        ----------
        ebed_dim_full
            Full SO(3) dimension D=(lmax+1)^2 used to slice the block-diagonal
            Wigner matrix.
        coeff_index_m
            Indices for the m-major reduced layout with shape (D_m_trunc,).
        key_lmax
            lmax used to build ``coeff_index_m`` (cache key).
        key_mmax
            mmax used to build ``coeff_index_m`` (cache key).

        Returns
        -------
        torch.Tensor
            Projected inverse rotation matrix with shape (E, D, D_m_trunc).
        """
        cache_key = (int(key_lmax), int(key_mmax))
        cache_dict = self.Dt_from_m_cache
        if cache_dict is None:
            raise ValueError("EdgeFeatureCache.Dt_from_m_cache is None")
        cached = cache_dict.get(cache_key)
        if cached is not None:
            return cached

        Dt_full = self.Dt_full  # (E, D, D)
        if Dt_full is None:
            raise ValueError("EdgeFeatureCache.Dt_full is None")
        Dt_block = Dt_full[:, :ebed_dim_full, :ebed_dim_full]  # (E, D, D)
        Dt_from_m = Dt_block.index_select(2, coeff_index_m)  # (E, D, D_m_trunc)
        cache_dict[cache_key] = Dt_from_m
        return Dt_from_m


class RadialMLP(nn.Module):
    """
    Radial MLP with LayerNorm and configurable activation.

    Parameters
    ----------
    mlp_layers : list[int]
        Layer sizes including input and output dimensions.
        E.g., [in_dim, hidden1, hidden2, out_dim].
    activation_function : str
        Activation function name (e.g., "silu", "tanh", "gelu").
    dtype : torch.dtype
        Floating point dtype for the linear layers.
    trainable : bool
        Whether the parameters are trainable.

    Architecture
    ------------
    Linear → LayerNorm → Activation for all hidden layers,
    with the final layer being a plain Linear (no LN, no activation).
    The first layer's bias is initialized to zero.

    Notes
    -----
    LayerNorm provides stable gradients. The first layer bias is zero-initialized
    to ensure smooth gradient flow at initialization.
    """

    def __init__(
        self,
        mlp_layers: list[int],
        *,
        activation_function: str = "silu",
        dtype: torch.dtype = torch.float32,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        if len(mlp_layers) < 2:
            raise ValueError("`mlp_layers` must have at least 2 elements")
        self.mlp_layers = list(mlp_layers)
        self.activation_function = str(activation_function)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[self.dtype]
        self.trainable = bool(trainable)

        modules: list[nn.Module] = []
        n_layers = len(mlp_layers)
        for i in range(n_layers - 1):
            linear = MLPLayer(
                mlp_layers[i],
                mlp_layers[i + 1],
                bias=True,
                activation_function=None,
                precision=self.precision,
                seed=child_seed(seed, i),
                trainable=trainable,
            )
            # First layer: zero-initialize bias for smooth gradient flow
            if i == 0 and linear.bias is not None:
                nn.init.zeros_(linear.bias)
            modules.append(linear)
            # Last layer: no LayerNorm/activation
            if i < n_layers - 2:
                modules.append(
                    nn.LayerNorm(
                        mlp_layers[i + 1], dtype=self.dtype, device=self.device
                    )
                )
                modules.append(ActivationFn(self.activation_function))

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (..., mlp_layers[0]).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (..., mlp_layers[-1]).
        """
        return self.net(x)

    def serialize(self) -> dict[str, Any]:
        """Serialize the RadialMLP to a dict."""
        state = self.net.state_dict()
        return {
            "@class": "RadialMLP",
            "@version": 1,
            "mlp_layers": self.mlp_layers.copy(),
            "activation_function": self.activation_function,
            "dtype": RESERVED_PRECISION_DICT[self.dtype],
            "trainable": self.trainable,
            "@variables": {k: np_safe(v) for k, v in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> RadialMLP:
        """Deserialize a RadialMLP from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "RadialMLP":
            raise ValueError(f"Invalid class for RadialMLP: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported RadialMLP version: {version}")
        variables = data.pop("@variables")
        data["dtype"] = PRECISION_DICT[data["dtype"]]
        obj = cls(**data)
        state = {
            k: safe_numpy_to_tensor(v, device=env.DEVICE, dtype=obj.dtype)
            for k, v in variables.items()
        }
        obj.net.load_state_dict(state)
        return obj


class C2CutoffEnvelope(torch.nn.Module):
    """
    C^2-continuous polynomial cutoff envelope function.

    As proposed in DimeNet: https://arxiv.org/abs/2003.03123

    This envelope provides a smooth transition to zero at the cutoff radius,
    ensuring continuity of the function value, first derivative, and second
    derivative.

    Notes
    -----
    The envelope function is defined for scaled distance ``x = r / rcut`` as::

        E(x) = 1 + x^p * (a + b*x + c*x^2),  for x < 1
        E(x) = 0,                            for x >= 1

    where the coefficients are chosen to satisfy::

        E(0) = 1,    E(1) = 0
        E'(1) = 0,   E''(1) = 0

    This ensures C^2 continuity at the cutoff boundary. The coefficients are::

        a = -(p + 1)(p + 2) / 2
        b = p(p + 2)
        c = -p(p + 1) / 2

    For the default exponent p=5, the coefficients are a=-21, b=35, c=-15::

        E(x) = 1 + x^5 * (-21 + 35*x - 15*x^2)
             = 1 - 21*x^5 + 35*x^6 - 15*x^7

    Parameters
    ----------
    rcut : float
        Cutoff radius in Å.
    exponent : int, optional
        Polynomial exponent (p), must be positive. Default is 5.

    Attributes
    ----------
    rcut : float
        Cutoff radius in Å.
    p : float
        Polynomial exponent.
    a : float
        Quadratic coefficient for x^p term.
    b : float
        Linear coefficient for x^(p+1) term.
    c : float
        Constant coefficient for x^(p+2) term.
    """

    def __init__(self, rcut: float, exponent: int = 5) -> None:
        super().__init__()
        assert exponent > 0
        self.rcut = float(rcut)
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, dst: torch.Tensor) -> torch.Tensor:
        """Compute the envelope value for given distances."""
        d_scaled = (dst / self.rcut).clamp(min=0.0, max=1.0)
        env_val = 1 + (d_scaled**self.p) * (
            self.a + d_scaled * (self.b + self.c * d_scaled)
        )
        return env_val * ((d_scaled < 1.0).to(dst.dtype))


class RadialBasis(nn.Module):
    """
    Spherical Bessel radial basis with C^2 cutoff envelope.

    Frequencies are trainable nn.Parameter, allowing the model
    to learn optimal radial basis spacing during training.

    Notes
    -----
    This implementation computes the spherical Bessel radial basis
    using PyTorch's sinc function for numerical stability::

        phi_n(r) = w_n * sinc(w_n * r / π)

    where ``torch.sinc(z) = sin(π*z) / (π*z)``. This is mathematically
    equivalent to the standard form ``sin(w_n * r) / r``, but sinc handles
    the r->0 limit via Taylor expansion, providing continuous gradients
    without explicit epsilon clamping.

    The ``r -> 0`` limit is finite::

        lim_{r->0} w_n * sinc(w_n * r / π) = w_n

    The initial frequencies follow a common "Bessel" spacing::

        w_n = n * π / rcut, for n = 1..n_radial (in 1/Å)

    The C^2 cutoff envelope is multiplied directly into the output to ensure
    strict smoothness at ``rcut``.

    Parameters
    ----------
    rcut : float
        Cutoff radius in Å.
    n_radial : int
        Number of basis functions.
    dtype : torch.dtype
        Floating-point dtype for the radial basis frequencies and outputs.
    exponent : int, optional
        Exponent for the C^2 cutoff envelope polynomial. Default is 7.
    """

    def __init__(
        self,
        rcut: float,
        n_radial: int,
        *,
        dtype: torch.dtype,
        exponent: int = 7,
    ) -> None:
        super().__init__()
        self.rcut = float(rcut)
        self.n_radial = int(n_radial)
        if self.n_radial <= 0:
            raise ValueError("`n_radial` must be positive")
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[self.dtype]
        self.exponent = int(exponent)

        # === Frequencies: n*π/rcut, n=1..n_radial ===
        # Shape: (1, n_radial), stored as trainable nn.Parameter
        freqs = torch.arange(
            1,
            self.n_radial + 1,
            device=self.device,
            dtype=self.dtype,
        ) * (math.pi / self.rcut)
        self.freqs = nn.Parameter(
            rearrange(freqs, "n_radial -> 1 n_radial"), requires_grad=True
        )

        self.envelope = C2CutoffEnvelope(rcut=self.rcut, exponent=self.exponent)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Compute radial basis functions.

        Parameters
        ----------
        r : torch.Tensor
            Pair distances with shape (N, 1) in Å, where N is the number of pairs.

        Returns
        -------
        torch.Tensor
            Radial basis multiplied by C^2 cutoff envelope with shape (N, n_rbf).
            The output is smoothly truncated to zero at r = rcut.
        """
        # === Step 1. Bessel Basis via Sinc ===
        # phi_n(r) = w_n * sinc(w_n * r / π)
        # Shape: (N, 1) * (1, n_radial) -> (N, n_radial)
        x = r * self.freqs  # (N, n_rbf)
        raw = self.freqs * torch.sinc(x / math.pi)  # (N, n_rbf)

        # === Step 2. Apply C^2 envelope for smooth cutoff ===
        envelope = self.envelope(r)  # (N, 1)
        return raw * envelope

    def serialize(self) -> dict[str, Any]:
        """Serialize RadialBasis including trainable frequencies."""
        return {
            "@class": "RadialBasis",
            "@version": 1,  # keep 1 at devel stage
            "rcut": self.rcut,
            "n_radial": self.n_radial,
            "exponent": self.exponent,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "freqs": np_safe(self.freqs),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> RadialBasis:
        """Deserialize RadialBasis including trainable frequencies."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "RadialBasis":
            raise ValueError(f"Invalid class for RadialBasis: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported RadialBasis version: {version}")
        precision = data["precision"]
        dtype = PRECISION_DICT[precision]
        obj = cls(
            rcut=float(data["rcut"]),
            n_radial=int(data["n_radial"]),
            exponent=int(data.get("exponent", 7)),
            dtype=dtype,
        )
        obj.freqs.data.copy_(
            safe_numpy_to_tensor(data["freqs"], device=obj.device, dtype=obj.dtype)
        )
        return obj


class GeometricInitialEmbedding(nn.Module):
    """
    Geometric initial embedding that adds zonal (m=0) rotated features.

    This module rotates pre-computed radial features for each degree l >= 1 using the
    zonal (m=0) column of the cached inverse Wigner-D blocks (local->global).
    The l=0 component is not computed here since it comes from type embedding.

    Parameters
    ----------
    lmax
        Maximum degree.
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
        self.ebed_dim = get_so3_dim_of_lmax(self.lmax)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        if self.lmax > 0:
            degree_index = map_degree_idx(self.lmax, device=self.device)
            # row_index covers all packed rows with l >= 1 (skip the scalar l=0 row).
            row_index = torch.arange(
                1, self.ebed_dim, device=self.device, dtype=torch.long
            )
            degree_row = degree_index[1:]
            # For each packed row, col_index picks the m=0 column of the same l-block:
            # so3_packed_index(l, 0) = l^2 + l = l * (l + 1).
            col_index = degree_row * (degree_row + 1)
            # radial_feat stores l=1..lmax at indices 0..lmax-1, so map l -> l-1.
            radial_index = degree_row - 1
            self.register_buffer("row_index", row_index, persistent=True)
            self.register_buffer("col_index", col_index, persistent=True)
            self.register_buffer("radial_index", radial_index, persistent=True)
        else:
            self.register_buffer(
                "row_index",
                torch.empty(0, device=self.device, dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                "col_index",
                torch.empty(0, device=self.device, dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                "radial_index",
                torch.empty(0, device=self.device, dtype=torch.long),
                persistent=True,
            )

    def forward(
        self,
        *,
        n_nodes: int,
        edge_cache: EdgeFeatureCache,
        radial_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        n_nodes
            Number of nodes (nf*nloc).
        edge_cache
            Per-edge cache containing geometry, weights, and Wigner-D blocks.
        radial_feat
            Per-edge radial features with shape (E, lmax, C) for l=1..lmax.

        Returns
        -------
        torch.Tensor
            Initial features to add with shape (N, D, C). l=0 is guaranteed zero.
        """
        # === Step 1. Early exit ===
        num_edges = edge_cache.src.size(0)
        if num_edges == 0:
            return torch.zeros(
                n_nodes,
                self.ebed_dim,
                self.channels,
                device=edge_cache.edge_vec.device,
                dtype=edge_cache.edge_vec.dtype,
            )

        device = edge_cache.edge_vec.device
        dtype = edge_cache.edge_vec.dtype
        out = torch.zeros(
            n_nodes, self.ebed_dim, self.channels, device=device, dtype=dtype
        )  # (N, D, C)
        if self.lmax == 0:
            return out

        # === Step 2. Gather all m=0 columns (l >= 1) in one shot ===
        # row_index selects all (l,m) rows with l>=1 in packed order.
        # col_index maps each row to its m=0 column index for that l-block.
        # Advanced indexing pairs (row_index[i], col_index[i]) so each row pulls
        # the correct m=0 value for its own l.
        Dt_full = edge_cache.Dt_full  # (E, D, D)
        d_col = Dt_full[:, self.row_index, self.col_index]  # (E, D-1)

        # === Step 3. Broadcast radial features per row ===
        # radial_index maps each packed row to its (l-1) radial slot.
        # This repeats the same radial feature across the (2l+1) rows of the l-block.
        radial_row = radial_feat.index_select(1, self.radial_index)  # (E, D-1, C)
        msg_global = d_col.unsqueeze(-1) * radial_row  # (E, D-1, C)

        # === Step 4. Scatter to nodes and normalize ===
        # Avoid advanced-index writeback (out[:, row_index, :]) which produces a copy.
        msg_out = out.new_zeros(
            n_nodes, self.row_index.numel(), self.channels
        )  # (N, D-1, C)
        msg_out.index_add_(0, edge_cache.dst, msg_global)
        out[:, self.row_index, :] = msg_out
        out.mul_(edge_cache.inv_sqrt_deg)
        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "GeometricInitialEmbedding",
            "@version": 1,
            "lmax": self.lmax,
            "channels": self.channels,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> GeometricInitialEmbedding:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "GeometricInitialEmbedding":
            raise ValueError(f"Invalid class for GeometricInitialEmbedding: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(
                f"Unsupported GeometricInitialEmbedding version: {version}"
            )
        precision = data.pop("precision")
        data["dtype"] = PRECISION_DICT[precision]
        return cls(**data)


class EnvironmentInitialEmbedding(nn.Module):
    """
    Environment matrix initial embedding for l=0 features.

    Computes an initial embedding based on the 4D environment matrix::

        [s, s * rx, s * ry, s * rz]

    Combined with independent type embeddings (individual type embedding),
    providing physical inductive bias for l=0 features.

    The computation follows the environment matrix approach where::

        1. Build `r_tilde = [s, s*r_hat]` where `s = edge_env / r` and `r_hat = edge_vec / r`
        2. G network: `g = G(rbf_proj(edge_rbf), type_src, type_dst)` produces per-edge features
           - Uses independent `env_type_embed` instead of projecting from main type embedding
           - Uses `rbf_proj` to project edge_rbf to `rbf_out_dim`
        3. env_agg: aggregate outer product `r_tilde ⊗ g` by destination node
        4. D matrix: `D = env_agg^T @ env_agg[:, :, :axis_dim]`
        5. Output: projection of flattened D matrix into FiLM logits

    Parameters
    ----------
    ntypes : int
        Number of atom types.
    n_radial : int
        Number of radial basis functions.
    channels : int
        Output channel dimension per FiLM branch (final output is 2*channels).
    embed_dim : int
        G network output dimension (filter width).
    axis_dim : int
        D matrix axis dimension (must be < embed_dim).
    type_dim : int
        Dimension for independent type embeddings in env_seed.
    hidden_dim : int
        Hidden layer size for G network.
    activation_function : str
        Activation function for G network hidden layer.
    eps : float
        Small epsilon for numerical stability.
    dtype : torch.dtype
        Parameter dtype.
    trainable : bool
        Whether parameters are trainable.
    seed : int | list[int] | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        *,
        ntypes: int,
        n_radial: int,
        channels: int,
        embed_dim: int = 64,
        axis_dim: int = 8,
        type_dim: int = 16,
        hidden_dim: int = 64,
        activation_function: str = "silu",
        eps: float = 1e-7,
        dtype: torch.dtype,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()

        # === Validate parameters ===
        if axis_dim >= embed_dim:
            raise ValueError(
                f"`axis_dim` ({axis_dim}) must be < `embed_dim` ({embed_dim})"
            )

        self.ntypes = int(ntypes)
        self.n_radial = int(n_radial)
        self.channels = int(channels)
        self.embed_dim = int(embed_dim)
        self.axis_dim = int(axis_dim)
        self.type_dim = int(type_dim)
        self.hidden_dim = int(hidden_dim)
        self.activation_function = str(activation_function)
        self.eps = float(eps)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        # === RBF projection: n_radial -> rbf_out_dim (two-layer MLP) ===
        # rbf_out_dim = max(32, embed_dim - 2*type_dim) to align G-network width to embed_dim
        # First layer: n_radial -> rbf_out_dim with activation
        # Second layer: rbf_out_dim -> rbf_out_dim linear
        self.rbf_out_dim = max(32, self.embed_dim - 2 * self.type_dim)
        seed_rbf_proj = child_seed(seed, 0)
        self.rbf_proj_layer1 = MLPLayer(
            self.n_radial,
            self.rbf_out_dim,
            bias=True,
            activation_function=self.activation_function,
            precision=self.precision,
            seed=child_seed(seed_rbf_proj, 0),
        )
        self.rbf_proj_layer2 = MLPLayer(
            self.rbf_out_dim,
            self.rbf_out_dim,
            bias=True,
            activation_function=None,
            precision=self.precision,
            seed=child_seed(seed_rbf_proj, 1),
        )

        # === Independent type embedding: ntypes -> type_dim ===
        # Individual type embedding
        seed_type_embed = child_seed(seed, 1)
        self.env_type_embed = TypeEmbedNet(
            type_nums=self.ntypes,
            embed_dim=self.type_dim,
            precision=self.precision,
            seed=seed_type_embed,
            trainable=trainable,
        )

        # === G network: (rbf_out_dim + 2*type_dim) -> hidden_dim -> embed_dim ===
        seed_g_net = child_seed(seed, 2)
        g_in_dim = self.rbf_out_dim + 2 * self.type_dim
        self.g_layer1 = MLPLayer(
            g_in_dim,
            self.hidden_dim,
            bias=True,
            activation_function=self.activation_function,
            precision=self.precision,
            seed=child_seed(seed_g_net, 0),
        )
        self.g_layer2 = MLPLayer(
            self.hidden_dim,
            self.embed_dim,
            bias=True,
            activation_function=None,
            precision=self.precision,
            seed=child_seed(seed_g_net, 1),
        )

        # === Output projection: embed_dim * axis_dim -> 2*channels ===
        # Zero init so FiLM logits start at zero; strengths control magnitude.
        seed_out = child_seed(seed, 3)
        self.output_proj = MLPLayer(
            self.embed_dim * self.axis_dim,
            2 * self.channels,
            bias=False,
            activation_function=None,
            init="final",
            precision=self.precision,
            seed=seed_out,
        )

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(
        self,
        *,
        edge_cache: EdgeFeatureCache,
        atype_flat: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """
        Compute environment FiLM logits for l=0 conditioning.

        Parameters
        ----------
        edge_cache : EdgeFeatureCache
            Edge cache containing src, dst, edge_vec, edge_rbf, edge_env.
        atype_flat : torch.Tensor
            Flattened atom types with shape (N,), where N = nf * nloc.
        n_nodes : int
            Number of nodes (N = nf * nloc).

        Returns
        -------
        torch.Tensor
            FiLM logits with shape (N, 2*channels).
        """
        num_edges = edge_cache.src.numel()
        if num_edges == 0:
            return torch.zeros(
                n_nodes, 2 * self.channels, dtype=self.dtype, device=self.device
            )

        src, dst = edge_cache.src, edge_cache.dst
        edge_vec = edge_cache.edge_vec  # (E, 3)
        edge_rbf = edge_cache.edge_rbf  # (E, n_radial)
        edge_env = edge_cache.edge_env  # (E, 1)

        # === Step 1. Construct r_tilde = [s, s*r_hat] ===
        # s = edge_env * (1/r), r_hat = edge_vec / r
        r_sq = (edge_vec * edge_vec).sum(dim=-1, keepdim=True)  # (E, 1)
        inv_r = torch.rsqrt(r_sq.clamp(min=self.eps * self.eps))  # (E, 1)
        s = edge_env * inv_r  # (E, 1)
        r_hat = edge_vec * inv_r  # (E, 3)
        r_tilde = torch.cat([s, s * r_hat], dim=-1)  # (E, 4)

        # === Step 2. Compute G network input and output ===
        # Use independent type embeddings (decoupled from main type embedding)
        atype_src = atype_flat.index_select(0, src)  # (E,)
        atype_dst = atype_flat.index_select(0, dst)  # (E,)
        type_src = self.env_type_embed(atype_src)  # (E, type_dim)
        type_dst = self.env_type_embed(atype_dst)  # (E, type_dim)

        # Project edge_rbf to rbf_out_dim (two-layer MLP)
        rbf_proj = self.rbf_proj_layer2(
            self.rbf_proj_layer1(edge_rbf)
        )  # (E, rbf_out_dim)

        # G network input: concat projected RBF and type embeddings
        g_input = torch.cat([rbf_proj, type_src, type_dst], dim=-1)  # (E, g_in_dim)
        g = self.g_layer2(self.g_layer1(g_input))  # (E, embed_dim)

        # === Step 3. Aggregate outer product by destination node ===
        # outer = r_tilde[:, :, None] * g[:, None, :]  # (E, 4, embed_dim)
        outer = torch.einsum("ei,ej->eij", r_tilde, g)  # (E, 4, embed_dim)
        outer_flat = outer.reshape(num_edges, 4 * self.embed_dim)  # (E, 4*embed_dim)
        env_agg = outer_flat.new_zeros(n_nodes, 4 * self.embed_dim)  # (N, 4*embed_dim)
        env_agg.index_add_(0, dst, outer_flat)
        env_agg = env_agg.reshape(n_nodes, 4, self.embed_dim)  # (N, 4, embed_dim)

        # === Step 4. Normalization by actual neighbor count ===
        # Use cached deg from edge_cache (already computed in build_edge_cache)
        deg_clamped = edge_cache.deg.clamp(min=1.0)  # (N,)
        deg_scale = torch.rsqrt(deg_clamped).reshape(-1, 1, 1)  # (N, 1, 1)
        env_agg = env_agg * deg_scale

        # === Step 5. D matrix construction: D = env_agg^T @ env_agg[:,:,:axis_dim] ===
        env_agg_t = env_agg.permute(0, 2, 1)  # (N, embed_dim, 4)
        env_agg_axis = env_agg[:, :, : self.axis_dim]  # (N, 4, axis_dim)
        D = torch.bmm(env_agg_t, env_agg_axis)  # (N, embed_dim, axis_dim)

        # === Step 6. Output projection for FiLM logits ===
        D_flat = D.reshape(
            n_nodes, self.embed_dim * self.axis_dim
        )  # (N, embed_dim*axis_dim)
        return self.output_proj(D_flat)

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "EnvironmentInitialEmbedding",
            "@version": 1,
            "config": {
                "ntypes": self.ntypes,
                "n_radial": self.n_radial,
                "channels": self.channels,
                "embed_dim": self.embed_dim,
                "axis_dim": self.axis_dim,
                "type_dim": self.type_dim,
                "hidden_dim": self.hidden_dim,
                "activation_function": self.activation_function,
                "eps": self.eps,
                "precision": self.precision,
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EnvironmentInitialEmbedding:
        """Deserialize from dictionary."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "EnvironmentInitialEmbedding":
            raise ValueError(f"Invalid class: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
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


class WignerDCalculator(nn.Module):
    """
    Fast Wigner-D blocks in the real spherical harmonics (tesseral) basis.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum degree.
    eps : float
        Small epsilon for numerical stability.
    dtype : torch.dtype
        Floating-point dtype for output matrices.

    This module assembles all D^l blocks into a single block-diagonal matrix
    and computes them in one batched matmul chain. This reduces Python dispatch
    overhead but requires O(n_edges * dim_full^2) memory where dim_full = (lmax+1)^2.

    Notes
    -----
    **Conventions**

    - ``rot_mat`` is a batch of 3x3 rotation matrices with shape ``(n_edges, 3, 3)``.
      It is a global->local transform for 3D vectors::

        v_local = rot_mat @ v_global

    - Euler angles follow the ZYZ convention::

        rot_mat = Rz(alpha) @ Ry(beta) @ Rz(gamma)

      with ``beta in [0, pi]``. Singular cases ``sin(beta) ~ 0`` are resolved by
      setting ``gamma = 0`` and folding the residual z-rotation into ``alpha``
      (stable and differentiable).

    - Within each degree ``l``, channels are ordered by ``m = -l, ..., +l``.
      Index mapping: ``i = m + l``.

    **Representation Matrices**

    - In the complex spherical harmonics basis, z-axis rotations are diagonal::

        D_z^{(l)}(theta)_{m1,m2} = delta_{m1,m2} * exp(-i * m1 * theta)

    - In the real (tesseral) basis used here, each pair ``{+m, -m}`` (for ``m>0``)
      forms a 2x2 rotation by angle ``m*theta``; ``m=0`` is invariant. This block is
      denoted as ``Z^{(l)}(theta)``.

    **Efficient Y-Rotation via Conjugation**

    For ZYZ Euler angles, the Wigner-D matrix is::

        D^{(l)}(rot_mat) = Z^{(l)}(alpha) @ D_y^{(l)}(beta) @ Z^{(l)}(gamma)

    The expensive part is the y-axis rotation ``D_y^{(l)}(beta)``. We use the identity::

        Ry(beta) = Rx(pi/2)^{-1} @ Rz(beta) @ Rx(pi/2)

    Define a per-degree constant::

        J_l = D_x ^ {(l)}(pi / 2)

    In the real basis, representation matrices are orthogonal, so::

        D_x^{(l)}(-pi/2) = J_l^{-1} = J_l^T

    Therefore::

        D_y^{(l)}(beta) = J_l^T @ Z^{(l)}(beta) @ J_l

    and the full block becomes::

        D^{(l)}(rot_mat) = Z^{(l)}(alpha) @ J_l^T @ Z^{(l)}(beta) @ J_l @ Z^{(l)}(gamma)

    **Block-Diagonal Parallel Computation**

    Instead of computing each l separately, we assemble all blocks into a single
    block-diagonal matrix of dimension ``dim_full = (lmax+1)^2``::

        J_full = diag(J_0, J_1, ..., J_lmax)
        Z_full(theta) = diag(Z^{(0)}(theta), Z^{(1)}(theta), ..., Z^{(lmax)}(theta))

    The full computation becomes::

        D_full = Z_full(alpha) @ J_full ^ T @ Z_full(beta) @ J_full @ Z_full(gamma)

    This reduces ``lmax+1`` separate matmul chains to a single chain on larger matrices.

    **Index Layout**

    - Block l occupies rows/columns [l^2, (l+1)^2) in the full matrix.
    - Within block l, the m=0 element is at position l^2 + l (center of block).
    - For m > 0, the 2x2 rotation sub-block occupies:
      - pos = l^2 + (l + m) for +m
      - neg = l^2 + (l - m) for -m

    **Outputs and Usage**

    - ``D_full`` has shape ``(n_edges, D, D)`` and is orthogonal. It represents the
      same global->local rotation as ``rot_mat``.
    - ``Dt_full = D_full.transpose(-1, -2)`` is the inverse (local->global).
    - Apply to packed features as::

        x_local = D_full @ x_global
        x_global = Dt_full @ x_local

    - For degree ``l``, the block is sliced by ``[l^2 : (l+1)^2]`` along both axes.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum degree.
    eps : float
        Small epsilon for numerical stability.
    dtype : torch.dtype
        Floating-point dtype for output matrices.
    """

    def __init__(
        self,
        lmax: int,
        *,
        eps: float = 1e-7,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        if self.lmax < 0:
            raise ValueError("`lmax` must be non-negative")
        self.dtype = dtype
        self.device = env.DEVICE
        self.eps = float(eps)

        # === Step 1. Compute block dimension ===
        # dim_full = sum_{l=0..lmax}(2l+1) = (lmax+1)^2
        # Block l occupies indices [l^2, (l+1)^2) in the full matrix.
        self.dim_full = (self.lmax + 1) ** 2

        # === Step 2. Build J_full as block-diagonal matrix ===
        # J_full contains J_l = D^{(l)}(Rx(pi/2)) on diagonal blocks
        J_full = torch.zeros(
            self.dim_full,
            self.dim_full,
            dtype=self.dtype,
            device=self.device,
        )
        for l in range(self.lmax + 1):
            J_l = self._compute_j_matrix(l).to(device=self.device, dtype=self.dtype)
            start, end = l * l, (l + 1) * (l + 1)
            J_full[start:end, start:end] = J_l

        self.register_buffer("J_full", J_full, persistent=True)
        self.register_buffer("Jt_full", J_full.T.contiguous(), persistent=True)

        # === Step 3. Precompute indices for Z_full construction ===
        self._precompute_z_indices()

    def forward(self, rot_mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Wigner-D blocks for a batch of rotation matrices.

        Parameters
        ----------
        rot_mat : torch.Tensor
            Rotation matrices with shape (E, 3, 3), global->local.

        Returns
        -------
        D_full : torch.Tensor
            Block-diagonal matrix with shape (E, D, D) where D=(lmax+1)^2.
        Dt_full : torch.Tensor
            Transpose of D_full.
        """
        rot_mat = rot_mat.to(dtype=self.dtype)

        # === Step 1. Extract ZYZ Euler angles ===
        # Convention: rot_mat = Rz(alpha) @ Ry(beta) @ Rz(gamma)
        with nvtx_range("WignerD/euler"):
            alpha, beta, gamma = self._extract_zyz_euler(rot_mat)  # (E,), (E,), (E,)

        # === Step 2. Build block-diagonal Z matrices ===
        # Each Z_full has shape (E, dim_full, dim_full)
        with nvtx_range("WignerD/z_rotation"):
            Za_full = self._build_z_rotation(alpha)  # (E, D, D)
            Zb_full = self._build_z_rotation(beta)  # (E, D, D)
            Zc_full = self._build_z_rotation(gamma)  # (E, D, D)

        # === Step 3. Compute D_full via single matmul chain ===
        # D^{(l)}(R) = Z(alpha) @ J^T @ Z(beta) @ J @ Z(gamma)
        with nvtx_range("WignerD/matmul"):
            J_full = self.J_full  # (D, D)
            Jt_full = self.Jt_full  # (D, D)
            D_full = Za_full @ Jt_full @ Zb_full @ J_full @ Zc_full  # (E, D, D)
            Dt_full = D_full.transpose(-1, -2).contiguous()  # (E, D, D)

        return D_full, Dt_full

    def _extract_zyz_euler(
        self, rot_mat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert rotation matrices to ZYZ Euler angles.

        The returned angles satisfy::

            rot_mat = Rz(alpha) @ Ry(beta) @ Rz(gamma)

        where the basic rotation matrices are::

            Rz(alpha) = [[ ca, -sa,  0],
                         [ sa,  ca,  0],
                         [  0,   0,  1]]
            Ry(beta)  = [[ cb,   0, sb],
                         [  0,   1,  0],
                         [-sb,   0, cb]]
            Rz(gamma) = [[ cg, -sg,  0],
                         [ sg,  cg,  0],
                         [  0,   0,  1]]

        with ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta),
        cg = cos(gamma), sg = sin(gamma).

        The full 3x3 rotation matrix (for sin(beta) != 0) is::

            R = [
                [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
                [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
                [-sb * cg, sb * sg, cb],
            ]

        Therefore, the Euler angles are extracted as::

            alpha = atan2(R[1, 2], R[0, 2])
            beta = atan2(sin(beta), R[2, 2])
            gamma = atan2(R[2, 1], -R[2, 0])

        Singular cases (beta -> 0 or pi) are handled by setting ``gamma = 0``
        and folding the residual z-rotation into ``alpha``.

        Parameters
        ----------
        rot_mat : torch.Tensor
            Rotation matrices with shape (..., 3, 3).

        Returns
        -------
        alpha : torch.Tensor
            First z-rotation angle with shape (...,).
        beta : torch.Tensor
            y-rotation angle with shape (...,).
        gamma : torch.Tensor
            Second z-rotation angle with shape (...,).
        """
        # === Step 1. Compute beta with stable atan2(sin(beta), cos(beta)) ===
        # Using acos(cos_beta) creates Inf/NaN gradients near |cos_beta| = 1.
        # Gimbal lock occur when edge directions can align with the global z-axis.
        #
        # For ZYZ convention:
        #   cos(beta) = R[2, 2]
        #   sin(beta) = sqrt(R[0, 2]^2 + R[1, 2]^2)
        #
        # We apply an epsilon floor to sin(beta) to keep beta differentiable
        # at the singular manifolds (beta = 0 or pi).
        cos_beta = rot_mat[..., 2, 2].clamp(-1.0, 1.0)
        r02 = rot_mat[..., 0, 2]
        r12 = rot_mat[..., 1, 2]
        sin_beta_sq = r02 * r02 + r12 * r12
        sin_beta_raw = torch.sqrt(sin_beta_sq.clamp(min=0.0))
        sin_beta_safe = torch.sqrt(sin_beta_sq.clamp(min=self.eps))
        beta = torch.atan2(sin_beta_safe, cos_beta)

        # === Step 2. Detect singular cases via sin(beta) ===
        threshold = math.sqrt(self.eps)
        not_singular = sin_beta_raw > threshold

        # === Step 3. Non-singular extraction (sin(beta) > 0) ===
        # torch.atan2(y, x) has undefined gradient at (y, x) = (0, 0).
        # We use a safe variant that perturbs x by eps when ||(x,y)|| is tiny.
        alpha = self._safe_atan2(rot_mat[..., 1, 2], rot_mat[..., 0, 2])
        gamma = self._safe_atan2(rot_mat[..., 2, 1], -rot_mat[..., 2, 0])

        # === Step 4. Singular extraction (gimbal lock) ===
        # When sin(beta) -> 0, alpha and gamma are not individually identifiable.
        # Two singular manifolds exist:
        #   (1) beta -> 0:  R = Rz(alpha) @ I @ Rz(gamma) = Rz(alpha + gamma)
        #   (2) beta -> pi: R = Rz(alpha) @ Ry(pi) @ Rz(gamma)
        #                  = Rz(alpha - gamma) @ Ry(pi)   (since Ry(pi) conjugates Rz)
        #
        # We fix the gauge by setting gamma = 0 and folding the residual z-rotation
        # into alpha, using stable atan2 formulas on the (x,y) block.
        alpha_beta0 = self._safe_atan2(rot_mat[..., 1, 0], rot_mat[..., 0, 0])
        alpha_betapi = self._safe_atan2(-rot_mat[..., 1, 0], -rot_mat[..., 0, 0])
        alpha_singular = torch.where(cos_beta > 0.0, alpha_beta0, alpha_betapi)

        alpha = torch.where(not_singular, alpha, alpha_singular)
        gamma = torch.where(not_singular, gamma, torch.zeros_like(gamma))
        return alpha, beta, gamma

    def _safe_atan2(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Safe atan2 that avoids NaN gradients at (y, x) = (0, 0).

        Perturbing x by eps when ||(x,y)|| is tiny avoids NaNs without
        affecting non-degenerate rotations.

        Parameters
        ----------
        y : torch.Tensor
            Numerator with shape (...).
        x : torch.Tensor
            Denominator with shape (...).

        Returns
        -------
        torch.Tensor
            arctan(y, x) with shape (...).
        """
        mag2 = x * x + y * y
        x_safe = x + (mag2 < self.eps).to(dtype=x.dtype) * self.eps
        return torch.atan2(y, x_safe)

    def _compute_j_matrix(self, l: int) -> torch.Tensor:
        """
        Compute J_l = D^{(l)}(Rx(pi/2)) in the real spherical harmonics basis.

        Computed on CPU with float64 for numerical precision.

        This matrix enables the ZYZ factorization::

            D^{(l)}(R) = Z(alpha) @ J^T @ Z(beta) @ J @ Z(gamma)

        Parameters
        ----------
        l
            Angular momentum order.

        Returns
        -------
        torch.Tensor
            J_l with shape (2l+1, 2l+1) in float64 on CPU.
        """
        dim = 2 * l + 1

        # === Step 1. Extract ZYZ Euler angles for Rx(pi/2) ===
        # Rx(pi/2): [[1,0,0], [0,0,-1], [0,1,0]]
        Rx90 = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
            device="cpu",
        )
        alpha_t, beta_t, gamma_t = self._extract_zyz_euler(Rx90.unsqueeze(0))
        alpha = float(alpha_t[0].item())
        beta = float(beta_t[0].item())
        gamma = float(gamma_t[0].item())

        # === Step 2. Build D^{(l)} in complex basis ===
        # D^l_{m1,m2} = exp(-i*m1*alpha) * d^l_{m1,m2}(beta) * exp(-i*m2*gamma)
        #            = exp(-i*(m1*alpha + m2*gamma)) * d^l_{m1,m2}(beta)
        D_complex = torch.zeros(dim, dim, dtype=torch.complex128, device="cpu")
        for m1 in range(-l, l + 1):
            for m2 in range(-l, l + 1):
                d_elem = self._wigner_d_y_element(l, m1, m2, beta)
                phase = math.cos(m1 * alpha + m2 * gamma) - 1j * math.sin(
                    m1 * alpha + m2 * gamma
                )
                D_complex[m1 + l, m2 + l] = phase * d_elem

        # === Step 3. Build change-of-basis matrix C (complex -> real SH) ===
        # Y_{l,0}^{real} = Y_l^0
        # Y_{l,m}^{real} = (Y_l^m + (-1)^m Y_l^{-m}) / sqrt(2),     m > 0
        # Y_{l,-m}^{real} = (Y_l^m - (-1)^m Y_l^{-m}) / (i*sqrt(2)), m > 0
        # With this convention, C is unitary and C^{-1} = C^H.
        C = torch.zeros(dim, dim, dtype=torch.complex128, device="cpu")
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        for m in range(-l, l + 1):
            row = m + l
            if m == 0:
                C[row, l] = 1.0
            elif m > 0:
                C[row, m + l] = inv_sqrt2
                C[row, -m + l] = ((-1) ** m) * inv_sqrt2
            else:
                C[row, -m + l] = -1j * inv_sqrt2
                C[row, m + l] = ((-1) ** m) * 1j * inv_sqrt2

        # === Step 4. Convert to real basis: D_real = C @ D_complex @ C^H ===
        C_inv = C.conj().transpose(-1, -2)
        D_real = (C @ D_complex @ C_inv).real

        return D_real

    def _wigner_d_y_element(self, l: int, m1: int, m2: int, beta: float) -> float:
        """
        Compute Wigner d-matrix element d^l_{m1,m2}(beta) for y-axis rotation.

        This is the standard closed-form sum (Varshalovich et al.)::

            d^l_{m1,m2}(beta)
                = sum_s (-1)^{m1-m2+s}
                  * sqrt((l+m2)!(l-m2)!(l+m1)!(l-m1)!)
                  / ((l+m2-s)! s! (m1-m2+s)! (l-m1-s)!)
                  * cos(beta/2)^{2l+m2-m1-2s}
                  * sin(beta/2)^{m1-m2+2s}

        Parameters
        ----------
        l
            Angular momentum order.
        m1
            Row index in [-l, ..., l].
        m2
            Column index in [-l, ..., l].
        beta
            Rotation angle about y-axis in radian.

        Returns
        -------
        float
            The matrix element d^l_{m1,m2}(beta).
        """
        # === Step 1. Validate Indices ===
        # Outside the representation range, the element is identically zero.
        if abs(m1) > l or abs(m2) > l:
            return 0.0

        # === Step 2. Precompute Half-Angle Terms ===
        # The closed-form sum uses powers of cos(beta/2) and sin(beta/2).
        cos_b = math.cos(beta / 2.0)
        sin_b = math.sin(beta / 2.0)

        # === Step 3. Determine Valid Summation Range for s ===
        # Denominator factorials require all arguments to be non-negative:
        #   (l + m2 - s)!   => s <= l + m2
        #   s!              => s >= 0
        #   (m1 - m2 + s)!  => s >= m2 - m1
        #   (l - m1 - s)!   => s <= l - m1
        # Therefore:
        #   s_min = max(0, m2 - m1)
        #   s_max = min(l + m2, l - m1)
        s_min = max(0, m2 - m1)
        s_max = min(l + m2, l - m1)

        # === Step 4. Precompute the Factorial Prefactor ===
        # pref = sqrt((l+m2)!(l-m2)!(l+m1)!(l-m1)!)
        pref = math.sqrt(
            math.factorial(l + m2)
            * math.factorial(l - m2)
            * math.factorial(l + m1)
            * math.factorial(l - m1)
        )

        # === Step 5. Accumulate the Closed-Form Sum ===
        out = 0.0
        for s in range(s_min, s_max + 1):
            # denom = (l+m2-s)! s! (m1-m2+s)! (l-m1-s)!
            denom = (
                math.factorial(l + m2 - s)
                * math.factorial(s)
                * math.factorial(m1 - m2 + s)
                * math.factorial(l - m1 - s)
            )
            # sign = (-1)^(m1 - m2 + s)
            sign = -1.0 if ((m1 - m2 + s) % 2) else 1.0

            # Powers of half-angle terms:
            #   p_cos = 2l + m2 - m1 - 2s
            #   p_sin = m1 - m2 + 2s
            p_cos = 2 * l + m2 - m1 - 2 * s
            p_sin = m1 - m2 + 2 * s
            out += sign * pref / denom * (cos_b**p_cos) * (sin_b**p_sin)
        return out

    def _build_z_rotation(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Build block-diagonal Z rotation matrix in real spherical harmonics basis.

        The Z matrix is block-diagonal with blocks Z^{(l)}(theta) for l=0..lmax.
        Each block implements z-axis rotation in the real SH basis.

        In the complex spherical harmonics basis, z-axis rotation is diagonal::

            D_z^{(l)}(theta)_{m1,m2} = delta_{m1,m2} * exp(-i * m1 * theta)

        In the real (tesseral) basis, each pair (m, -m) for m > 0 forms a 2x2
        rotation block by angle m*theta; m=0 is invariant.

        Parameters
        ----------
        angle : torch.Tensor
            Rotation angles with shape (E,).

        Returns
        -------
        torch.Tensor
            Block-diagonal rotation matrices with shape (E, D, D).
        """
        m0_indices = self.m0_indices  # (lmax+1,)
        m_values = self.m_values  # (n_blocks,)
        pos_indices = self.pos_indices  # (n_blocks,)
        neg_indices = self.neg_indices  # (n_blocks,)

        # === Step 1. Allocate Z matrix ===
        n_edges = angle.shape[0]
        Z = torch.zeros(
            n_edges,
            self.dim_full,
            self.dim_full,
            dtype=angle.dtype,
            device=angle.device,
        )  # (E, D, D)

        # === Step 2. Set m=0 diagonal elements to 1 ===
        # Z[:, m0_idx, m0_idx] = 1 for each l's center element
        Z[:, m0_indices, m0_indices] = 1.0

        # === Step 3. Fill m>0 rotation blocks ===
        if m_values.numel() > 0:
            # Compute cos(m*angle) and sin(m*angle) for all (l,m) pairs
            # angles_m: (E, n_blocks) where n_blocks = total (l,m) pairs with m>0
            angles_m = angle[:, None] * m_values[None, :]  # (E, n_blocks)
            c = torch.cos(angles_m)  # (E, n_blocks)
            s = torch.sin(angles_m)  # (E, n_blocks)

            # Fill the 2x2 rotation blocks:
            #   Z[pos, pos] =  cos(m*theta)
            #   Z[neg, neg] =  cos(m*theta)
            #   Z[pos, neg] =  sin(m*theta)
            #   Z[neg, pos] = -sin(m*theta)
            Z[:, pos_indices, pos_indices] = c
            Z[:, neg_indices, neg_indices] = c
            Z[:, pos_indices, neg_indices] = s
            Z[:, neg_indices, pos_indices] = -s

        return Z

    def _precompute_z_indices(self) -> None:
        """
        Precompute index arrays for efficient Z_full construction.

        For each block l in the block-diagonal Z matrix:
        - m=0 element is at diagonal position l^2 + l
        - For m > 0, the 2x2 rotation block occupies positions:
            pos = l^2 + (l + m)  ->  +m row/column
            neg = l^2 + (l - m)  ->  -m row/column

        The 2x2 block structure in real spherical harmonics basis::

            Z[pos, pos] = cos(m * theta)
            Z[neg, neg] = cos(m * theta)
            Z[pos, neg] = sin(m * theta)
            Z[neg, pos] = -sin(m * theta)
        """
        # === Step 1. Indices for m=0 diagonal elements ===
        m0_indices = torch.tensor(
            [so3_packed_index(l, 0) for l in range(self.lmax + 1)],
            dtype=torch.long,
            device=self.device,
        )
        self.register_buffer("m0_indices", m0_indices, persistent=True)

        # === Step 2. Indices for m>0 rotation blocks ===
        pos_indices: list[int] = []
        neg_indices: list[int] = []
        m_values: list[int] = []

        for l in range(1, self.lmax + 1):  # l=0 has no m>0 terms
            offset_l = l * l
            for m in range(1, l + 1):
                pos = offset_l + (l + m)  # index for +m
                neg = offset_l + (l - m)  # index for -m
                pos_indices.append(pos)
                neg_indices.append(neg)
                m_values.append(m)

        self.register_buffer(
            "pos_indices",
            torch.tensor(pos_indices, dtype=torch.long, device=self.device),
            persistent=True,
        )
        self.register_buffer(
            "neg_indices",
            torch.tensor(neg_indices, dtype=torch.long, device=self.device),
            persistent=True,
        )
        self.register_buffer(
            "m_values",
            torch.tensor(m_values, dtype=self.dtype, device=self.device),
            persistent=True,
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize WignerDCalculator (lmax and dtype are stored by parent)."""
        return {
            "@class": "WignerDCalculator",
            "@version": 1,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> WignerDCalculator:
        """Deserialize WignerDCalculator - parent handles lmax/dtype reconstruction."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "WignerDCalculator":
            raise ValueError(f"Invalid class for WignerDCalculator: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported WignerDCalculator version: {version}")
        raise NotImplementedError(
            "WignerDCalculator.deserialize should be called by parent with lmax/dtype"
        )


@contextmanager
def nvtx_range(name: str) -> Generator[None, None, None]:
    """
    Create an NVTX range when CUDA is available; otherwise, no-op.

    Parameters
    ----------
    name
        Range name shown in Nsight Systems/Compute.
    """
    if torch.cuda.is_available():
        nvtx = torch.cuda.nvtx
        if hasattr(nvtx, "range"):
            with nvtx.range(name):
                yield
            return
    yield


def edge_cache_to_dtype(
    cache: EdgeFeatureCache, dtype: torch.dtype
) -> EdgeFeatureCache:
    """
    Convert all floating-point tensors in EdgeFeatureCache to the specified dtype.

    Integer tensors (src, dst) are unchanged. This is a standalone function
    (not a method) to keep it side-effect free.

    Parameters
    ----------
    cache
        The edge feature cache to convert.
    dtype
        Target dtype for floating-point tensors.

    Returns
    -------
    EdgeFeatureCache
        New cache with converted tensors.
    """
    # Handle Optional tensors explicitly.
    # Use local variables with explicit None check and assignment.
    _D_full = cache.D_full
    _Dt_full = cache.Dt_full
    D_full: torch.Tensor | None = None
    Dt_full: torch.Tensor | None = None
    if _D_full is not None:
        D_full = _D_full.to(dtype=dtype)
    if _Dt_full is not None:
        Dt_full = _Dt_full.to(dtype=dtype)

    return EdgeFeatureCache(
        src=cache.src,
        dst=cache.dst,
        edge_type_feat=cache.edge_type_feat.to(dtype=dtype),
        edge_vec=cache.edge_vec.to(dtype=dtype),
        edge_rbf=cache.edge_rbf.to(dtype=dtype),
        edge_env=cache.edge_env.to(dtype=dtype),
        deg=cache.deg.to(dtype=dtype),
        inv_sqrt_deg=cache.inv_sqrt_deg.to(dtype=dtype),
        D_full=D_full,
        Dt_full=Dt_full,
        D_to_m_cache={} if cache.D_to_m_cache is None else cache.D_to_m_cache,
        Dt_from_m_cache=(
            {} if cache.Dt_from_m_cache is None else cache.Dt_from_m_cache
        ),
    )


def safe_norm(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute vector norm with an epsilon lower bound.

    Uses float32 for computation when input is fp16/bf16.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape (N, 3), where N is the number of vectors.
    eps : float
        Lower bound for the norm.

    Returns
    -------
    torch.Tensor
        Norm with shape (N, 1), clamped to be >= eps.
    """
    in_dtype = x.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True).clamp(min=eps**2))
    return norm.to(dtype=in_dtype)


def safe_numpy_to_tensor(
    data: Any, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    if isinstance(data, np.ndarray):
        # Handle bfloat16: numpy uses ml_dtypes.bfloat16, which torch.as_tensor
        # cannot convert. Convert to float32 first, then cast to target dtype.
        if hasattr(data.dtype, "name") and "bfloat16" in data.dtype.name:
            data = data.astype(np.float32)
        return torch.as_tensor(data, device=device).to(dtype)
    return torch.as_tensor(data, device=device, dtype=dtype)


def get_promoted_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Get promoted dtype for numerical stability.

    For bf16/fp16, use float32 to ensure numerical stability
    in computation and storage compatibility.
    """
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def np_safe(
    tensor: torch.Tensor | None,
) -> np.ndarray | None:
    """
    Convert tensor to numpy array, promoting low-precision types to fp32.

    For bf16/fp16, converts to fp32 first since NumPy/HDF5 do not natively
    support these formats. fp32/fp64 are kept unchanged.

    Parameters
    ----------
    tensor
        PyTorch tensor to convert. Can be None.

    Returns
    -------
    np.ndarray or None
        numpy array with at least fp32 precision.
    """
    if tensor is None:
        return None
    if tensor.dtype in (torch.float16, torch.bfloat16):
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def get_so3_dim_of_lmax(lmax: int) -> int:
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


def map_degree_idx(lmax: int, *, device: torch.device) -> torch.Tensor:
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


def project_D_to_m(
    D_full: torch.Tensor,
    coeff_index_m: torch.Tensor,
    ebed_dim_full: int,
    cache: dict[str, torch.Tensor] | None,
    key_lmax: int,
    key_mmax: int,
) -> torch.Tensor:
    """
    Row-project block-diagonal Wigner-D to the m-major truncated layout.

    Parameters
    ----------
    D_full
        Block-diagonal Wigner-D with shape (E, D, D).
    coeff_index_m
        Indices for m-major reduced layout with shape (D_m_trunc,).
    ebed_dim_full
        Full SO(3) dimension D_full = (lmax+1)^2 to slice the block.
    cache
        Optional cache mapping (lmax, mmax) -> projected matrix.
    key_lmax
        lmax used to build coeff_index_m (cache key).
    key_mmax
        mmax used to build coeff_index_m (cache key).

    Returns
    -------
    torch.Tensor
        Projected rotation matrix with shape (E, D_m_trunc, D).

    Examples
    --------
    For lmax=2, mmax=1 (D=9, D_m_trunc=7), coeff_index_m selects
    [0,2,6,1,5,3,7] in packed (l,m) order. The returned tensor keeps only those
    rows of ``D_full`` while retaining all columns, so that rotating and truncating
    is done in a single bmm: ``x_local = D_to_m @ x_global``.
    """
    cache_key = f"{int(key_lmax)}:{int(key_mmax)}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    D_block = D_full[:, :ebed_dim_full, :ebed_dim_full]
    proj = D_block.index_select(1, coeff_index_m)
    if cache is not None:
        cache[cache_key] = proj
    return proj


def project_Dt_from_m(
    Dt_full: torch.Tensor,
    coeff_index_m: torch.Tensor,
    ebed_dim_full: int,
    cache: dict[str, torch.Tensor] | None,
    key_lmax: int,
    key_mmax: int,
) -> torch.Tensor:
    """
    Column-project block-diagonal Wigner-D^T for inverse rotation.

    Parameters
    ----------
    Dt_full
        Block-diagonal Wigner-D^T with shape (E, D, D).
    coeff_index_m
        Indices for m-major reduced layout with shape (D_m_trunc,).
    ebed_dim_full
        Full SO(3) dimension D_full = (lmax+1)^2 to slice the block.
    cache
        Optional cache mapping (lmax, mmax) -> projected matrix.
    key_lmax
        lmax used to build coeff_index_m (cache key).
    key_mmax
        mmax used to build coeff_index_m (cache key).

    Returns
    -------
    torch.Tensor
        Projected inverse rotation matrix with shape (E, D, D_m_trunc).

    Examples
    --------
    Continuing lmax=2, mmax=1, the projection selects the same column subset
    [0,2,6,1,5,3,7] from ``Dt_full``. This enables inverse rotation with missing
    coefficients implicitly zeroed: ``x_global = Dt_from_m @ x_local``.
    """
    cache_key = f"{int(key_lmax)}:{int(key_mmax)}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    Dt_block = Dt_full[:, :ebed_dim_full, :ebed_dim_full]
    proj = Dt_block.index_select(2, coeff_index_m)
    if cache is not None:
        cache[cache_key] = proj
    return proj


def so3_packed_index(l: int, m: int) -> int:
    """
    Compute packed (l, m) index for real spherical harmonics layout.

    The packed layout is l-primary with m ordered as ``-l..+l`` inside each l-block.
    The index formula is::

        idx(l, m) = l^2 + l + m

    Parameters
    ----------
    l
        Degree l.
    m
        Order m, must satisfy ``-l <= m <= l``.

    Returns
    -------
    int
        Packed index.
    """
    l = int(l)
    m = int(m)
    return l * l + l + m


def build_l_major_index(lmax: int, mmax: int, *, device: torch.device) -> torch.Tensor:
    """
    Build coefficient indices for l-major layout truncated by mmax.

    The returned indices select coefficients with ``|m| <= min(mmax, l)`` in the
    standard packed (l, m) layout. The order is l-major:

    - l = 0..lmax
    - within each l, m = -min(mmax, l) .. +min(mmax, l)

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum order (|m|). Must satisfy ``0 <= mmax <= lmax``.
    device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Long tensor of indices with shape (D_m_trunc,), selecting coefficients
        from the full packed layout with D=(lmax+1)^2, where D_m_trunc is
        the number of coefficients kept under ``|m| <= min(mmax, l)``.

    Examples
    --------
    For lmax=2, mmax=1:
    - Full packed layout: l=0(0), l=1(1-3), l=2(4-8)
    - Truncated by mmax=1: skip (l=2, m=±2) at indices 4,8
    - Returns: [0, 1, 2, 3, 5, 6, 7]
    """
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    if lmax_i < 0:
        raise ValueError("`lmax` must be non-negative")
    if mmax_i < 0:
        raise ValueError("`mmax` must be non-negative")
    if mmax_i > lmax_i:
        raise ValueError("`mmax` must be <= `lmax`")

    indices: list[int] = []
    for l in range(lmax_i + 1):
        m_keep = min(mmax_i, l)
        for m in range(-m_keep, m_keep + 1):
            indices.append(so3_packed_index(l, m))
    return torch.tensor(indices, device=device, dtype=torch.long)


def build_m_major_index(lmax: int, mmax: int, *, device: torch.device) -> torch.Tensor:
    """
    Build coefficient indices for m-major layout truncated by mmax.

    This layout minimizes rotation cost and avoids gather-heavy indexing:

    - m = 0: l = 0..lmax (single coefficient per l)
    - for each m = 1..mmax:
        - negative part: l = m..lmax, coefficient (l, -m)
        - positive part: l = m..lmax, coefficient (l, +m)

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum order (|m|). Must satisfy ``0 <= mmax <= lmax``.
    device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Long tensor of indices with shape (D_m_trunc,), selecting coefficients
        from the full packed layout with D=(lmax+1)^2, where D_m_trunc is
        the number of coefficients kept under ``|m| <= min(mmax, l)``.

    Examples
    --------
    For lmax=2, mmax=1:
    - m=0 group: (l=0,m=0)→0, (l=1,m=0)→2, (l=2,m=0)→6
    - m=1 neg group: (l=1,m=-1)→1, (l=2,m=-1)→5
    - m=1 pos group: (l=1,m=+1)→3, (l=2,m=+1)→7
    - Returns: [0, 2, 6, 1, 5, 3, 7]
    """
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    if lmax_i < 0:
        raise ValueError("`lmax` must be non-negative")
    if mmax_i < 0:
        raise ValueError("`mmax` must be non-negative")
    if mmax_i > lmax_i:
        raise ValueError("`mmax` must be <= `lmax`")

    indices: list[int] = []
    # === Step 1. m = 0 group (l = 0..lmax) ===
    for l in range(lmax_i + 1):
        indices.append(so3_packed_index(l, 0))

    # === Step 2. m > 0 groups (neg then pos) ===
    for m in range(1, mmax_i + 1):
        for l in range(m, lmax_i + 1):
            indices.append(so3_packed_index(l, -m))
        for l in range(m, lmax_i + 1):
            indices.append(so3_packed_index(l, m))

    return torch.tensor(indices, device=device, dtype=torch.long)


def build_m_major_l_index(
    lmax: int, mmax: int, *, device: torch.device
) -> torch.Tensor:
    """
    Build degree (l) index aligned with `build_m_major_index`.

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum order (|m|). Must satisfy ``0 <= mmax <= lmax``.
    device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Long tensor of degrees with shape (D_m_trunc,). Entry i is the degree
        l for the i-th coefficient in the m-major layout.

    Examples
    --------
    For lmax=2, mmax=1:
    - m=0 group: l=0,1,2
    - m=1 neg group: l=1,2
    - m=1 pos group: l=1,2
    - Returns: [0, 1, 2, 1, 2, 1, 2]
    """
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    if lmax_i < 0:
        raise ValueError("`lmax` must be non-negative")
    if mmax_i < 0:
        raise ValueError("`mmax` must be non-negative")
    if mmax_i > lmax_i:
        raise ValueError("`mmax` must be <= `lmax`")

    degrees: list[int] = []
    # === Step 1. m = 0 group ===
    for l in range(lmax_i + 1):
        degrees.append(l)

    # === Step 2. m > 0 groups (neg then pos) ===
    for m in range(1, mmax_i + 1):
        for l in range(m, lmax_i + 1):
            degrees.append(l)
        for l in range(m, lmax_i + 1):
            degrees.append(l)

    return torch.tensor(degrees, device=device, dtype=torch.long)


def init_edge_rot_mat(edge_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation matrices that align each edge to the local + Z axis.

    The returned rotation is a global->local transform: ``v_local = R @ v_global``.
    So, for unit edge direction vector ``u``, ``R @ u = (0, 0, 1)``.

    Notes
    -----
    This routine constructs an orthonormal right-handed frame (x_hat, y_hat, z_hat)
    per edge via a simple Gram-Schmidt process::

        z_hat = edge_vec / ||edge_vec||           # local +z direction
        x_hat = normalize(ref - (ref·z_hat) z_hat)  # orthogonal to z_hat
        y_hat = z_hat x x_hat                       # right-handed

    where ``ref`` is a reference axis that is not nearly colinear with ``z_hat``.

    The rotation matrix stacks these basis vectors as rows::

        R = [x_hat^T; y_hat^T; z_hat^T]

    This makes ``R`` a global->local transform, because each row computes the
    dot product with the corresponding local basis vector.

    The reference-axis switch introduces a piecewise definition. For a smoother
    frame construction (especially for higher-order gradients), consider a
    Householder/Frisvad frame.

    Parameters
    ----------
    edge_vec
        Edge vectors with shape (E, 3).

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape (E, 3, 3).
    """
    # === Step 1. Normalize edge direction (local z) ===
    # z_hat is the unit edge direction (center -> neighbor).
    z_hat = edge_vec / safe_norm(edge_vec)

    # === Step 2. Construct x-axis by Gram-Schmidt against a reference ===
    # Choose a reference axis that is not nearly parallel to z_hat to avoid
    # catastrophic cancellation in the Gram-Schmidt projection.
    candi_1 = torch.tensor(
        [1.0, 0.0, 0.0], dtype=edge_vec.dtype, device=edge_vec.device
    ).expand_as(edge_vec)
    candi_2 = torch.tensor(
        [0.0, 1.0, 0.0], dtype=edge_vec.dtype, device=edge_vec.device
    ).expand_as(edge_vec)
    use_alt = torch.abs(torch.sum(z_hat * candi_1, dim=-1, keepdim=True)) > 0.9
    ref = torch.where(use_alt, candi_2, candi_1)

    # Remove the component along z_hat to obtain a vector orthogonal to z_hat.
    proj = torch.sum(ref * z_hat, dim=-1, keepdim=True) * z_hat
    x_hat = ref - proj
    x_hat = x_hat / safe_norm(x_hat)

    # === Step 3. Construct y-axis (right-handed) ===
    # Cross product enforces a right-handed frame: (x_hat, y_hat, z_hat).
    y_hat = torch.cross(z_hat, x_hat, dim=-1)
    y_hat = y_hat / safe_norm(y_hat)

    # === Step 4. Stack rows to form global->local rotation ===
    # Row-stacking ensures v_local = R @ v_global.
    rot_mat = torch.stack([x_hat, y_hat, z_hat], dim=-2)
    return rot_mat


def init_edge_rot_mat_frisvad(
    edge_vec: torch.Tensor,
    edge_len: torch.Tensor | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute rotation matrices that align each edge to the local + Z axis.

    The returned rotation is a global->local transform: ``v_local = R @ v_global``.
    So, for unit edge direction vector ``u``, ``R @ u = (0, 0, 1)``.

    Notes
    -----
    This routine constructs an orthonormal right-handed frame (x_hat, y_hat, z_hat)
    per edge using the Frisvad method (closed-form ONB from a unit vector).

    The Frisvad closed-form is singular at ``z_hat = (0, 0, -1)``, due to the
    ``1 / (1 + nz)`` denominator. For the singular neighborhood near ``-Z``, the
    basis must NOT fall back to fixed axes, otherwise x_hat/y_hat may not be
    exactly perpendicular to the current ``z_hat``. Instead, we build a strict
    orthonormal pair from the current ``z_hat`` via cross products, guaranteeing
    that the returned matrix is a proper rotation and that ``R @ z_hat = (0,0,1)``
    up to floating-point error.

    Given unit vector z_hat = (nx, ny, nz), for nz > -1, define::

        a = 1 / (1 + nz)
        b = -nx * ny * a
        x_hat = (1 - nx ^ 2 * a, b, -nx)
        y_hat = (b, 1 - ny ^ 2 * a, -ny)

    This yields an orthonormal basis with x_hat ⟂ z_hat, y_hat ⟂ z_hat and
    x_hat X y_hat = z_hat (right-handed). For nz close to -1, we fall back to a
    strict cross-product basis built from the current z_hat.

    The rotation matrix stacks these basis vectors as rows::

        R = [x_hat^T; y_hat^T; z_hat^T]

    This makes ``R`` a global->local transform, because each row computes the
    dot product with the corresponding local basis vector.

    Parameters
    ----------
    edge_vec
        Edge vectors with shape (E, 3).
    edge_len
        Precomputed edge lengths with shape (E, 1). If None, recompute from edge_vec.
    eps
        Small epsilon for numerical stability.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape (E, 3, 3).
    """
    # === Step 1. Normalize edge direction (local z) ===
    # z_hat is the unit edge direction (center -> neighbor).
    if edge_len is None:
        edge_len = safe_norm(edge_vec, eps)
    else:
        edge_len = edge_len.clamp(min=eps)
    z_hat = edge_vec / edge_len  # (E, 3)
    nx = z_hat[..., 0:1]  # (E, 1)
    ny = z_hat[..., 1:2]  # (E, 1)
    nz = z_hat[..., 2:3]  # (E, 1)

    # === Step 2. Frisvad closed-form orthonormal basis (non-singular) ===
    # The closed-form uses a = 1 / (1 + nz), which is singular at nz = -1.
    # Compute it with a safe denominator, then select by a singular mask.
    # Use a fixed threshold for singular detection (1e-6 is sufficient for all precisions).
    singular_threshold = 1.0e-6
    singular = nz < (-1.0 + singular_threshold)  # (E, 1)

    denom = 1.0 + nz  # (E, 1)
    denom_safe = torch.where(singular, torch.ones_like(denom), denom)  # (E, 1)
    a = 1.0 / denom_safe  # (E, 1)
    b = -nx * ny * a  # (E, 1)

    x_main = torch.cat([1.0 - nx * nx * a, b, -nx], dim=-1)  # (E, 3)
    y_main = torch.cat([b, 1.0 - ny * ny * a, -ny], dim=-1)  # (E, 3)

    # === Step 3. Strict fallback for the singular neighborhood (z_hat ~= -Z) ===
    # Build x_hat/y_hat from the current z_hat so that:
    #   x_hat ⟂ z_hat, y_hat ⟂ z_hat, and (x_hat, y_hat, z_hat) is right-handed.
    # In the singular neighborhood near -Z, ref = +X is guaranteed not parallel to z_hat.
    ref = torch.tensor(
        [1.0, 0.0, 0.0], dtype=edge_vec.dtype, device=edge_vec.device
    ).expand_as(edge_vec)  # (E, 3)
    x_fb = torch.cross(ref, z_hat, dim=-1)  # (E, 3)
    x_fb = x_fb / safe_norm(x_fb, eps)
    y_fb = torch.cross(z_hat, x_fb, dim=-1)  # (E, 3)
    y_fb = y_fb / safe_norm(y_fb, eps)

    mask3 = singular.expand_as(edge_vec)  # (E, 3)
    x_hat = torch.where(mask3, x_fb, x_main)
    y_hat = torch.where(mask3, y_fb, y_main)

    # Normalize to protect against numerical drift (and to match your existing style).
    x_hat = x_hat / safe_norm(x_hat, eps)
    y_hat = y_hat / safe_norm(y_hat, eps)

    # === Step 4. Stack rows to form global->local rotation ===
    # Row-stacking ensures v_local = R @ v_global.
    rot_mat = torch.stack([x_hat, y_hat, z_hat], dim=-2)  # (E, 3, 3)
    return rot_mat
