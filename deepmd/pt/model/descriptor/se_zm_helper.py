# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Helper utilities for SeZM PyTorch descriptor.

This module collects shared building blocks and math utilities, including:
- edge caches, radial bases/envelopes, and initial embeddings
- SO(3) index/projection helpers and dtype/serialization utilities

Role boundary:
- this file owns geometry/math helpers and reusable cache containers;
- `SeZM_WignerD.py` owns quaternion edge frames and Wigner-D rotation blocks;
- `se_zm_block.py` owns feature transforms and message-passing operators;
- `se_zm.py` owns descriptor orchestration and per-forward cache building.
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


ATTN_RES_MODES = ("none", "independent", "dependent")


def init_trunc_normal_fan_in_out(
    weight: torch.Tensor,
    seed: int | list[int] | None,
    scale: float = 1.0,
) -> None:
    """Initialize weight with truncated normal distribution.

    Uses Xavier-like variance scaling: std = scale / sqrt(fan_in + fan_out).
    Truncation at +/-3*std prevents extreme outliers.

    Parameters
    ----------
    weight : torch.Tensor
        Weight tensor with shape (out_features, in_features).
    seed : int | list[int] | None
        Random seed for reproducibility.
    scale : float, default=1.0
        Multiplicative scale factor in the standard deviation numerator.
    """
    if weight.ndim != 2:
        raise ValueError("`weight` must be a 2D tensor")
    if scale <= 0:
        raise ValueError("`scale` must be positive")
    fan_out, fan_in = weight.shape
    std = float(scale) / math.sqrt(fan_in + fan_out)
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


def segment_envelope_gated_softmax(
    logits: torch.Tensor,
    edge_env: torch.Tensor,
    dst: torch.Tensor,
    n_nodes: int,
    z_bias_raw: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Compute destination-wise envelope-gated softmax attention.

    Parameters
    ----------
    logits
        Attention logits with shape (E, F, H).
    edge_env
        Cutoff envelope weights with shape (E, 1) or (E,).
    dst
        Destination node indices with shape (E,).
    n_nodes
        Number of nodes.
    z_bias_raw
        Unconstrained denominator bias with shape (F, H).
        Softplus is applied to keep the bias strictly positive.
    eps
        Small epsilon for denominator stability.

    Returns
    -------
    torch.Tensor
        Normalized edge weights with shape (E, F, H), computed as
        ``edge_env**2 * exp(logits) / (zeta + sum(edge_env**2 * exp(logits)))``.
    """
    n_edge, n_focus, n_head = logits.shape
    n_channel = n_focus * n_head
    eps_f = float(eps)

    # === Step 1. Flatten (F, H) to reduce index tensor traffic ===
    logits_2d = logits.reshape(n_edge, n_channel)
    edge_env_1d = edge_env.squeeze(-1).to(dtype=logits.dtype).clamp_min(0.0)
    zeta = F.softplus(z_bias_raw).reshape(1, n_channel).to(dtype=logits.dtype)
    has_weight = edge_env_1d > 0.0
    logits_for_max = torch.where(
        has_weight.reshape(n_edge, 1),
        logits_2d,
        torch.full_like(logits_2d, float("-inf")),
    )

    # === Step 2. Destination-wise max for stable exponentials ===
    group_max = torch.full(
        (n_nodes, n_channel),
        float("-inf"),
        dtype=logits.dtype,
        device=logits.device,
    )
    group_max.index_reduce_(0, dst, logits_for_max, reduce="amax", include_self=True)
    edge_max = group_max.index_select(0, dst)
    edge_max = torch.where(
        torch.isfinite(edge_max), edge_max, torch.zeros_like(edge_max)
    )
    group_max_safe = torch.where(
        torch.isfinite(group_max), group_max, torch.zeros_like(group_max)
    )

    # === Step 3. Envelope-gated exponential terms ===
    exp_shifted = torch.exp(logits_2d - edge_max)
    edge_weighted_exp = edge_env_1d.square().reshape(n_edge, 1) * exp_shifted

    # === Step 4. Destination-wise normalization with positive denominator bias ===
    denom_sum = torch.zeros(
        n_nodes,
        n_channel,
        dtype=logits.dtype,
        device=logits.device,
    )
    denom_sum.index_add_(0, dst, edge_weighted_exp)
    denom = denom_sum + zeta * torch.exp(-group_max_safe)

    alpha = edge_weighted_exp / (denom.index_select(0, dst) + eps_f)
    return alpha.reshape(n_edge, n_focus, n_head)


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
        The C^3 cutoff envelope is already baked in.
    edge_env
        C^3 cutoff envelope weights with shape (E, 1).
    deg
        Envelope-squared smooth degree with shape (N,), computed as
        ``sum(edge_env**2)`` over incoming edges.
        Used for smooth normalization in EnvironmentInitialEmbedding.
    inv_sqrt_deg
        Inverse square root smooth degree normalization with shape (N, 1, 1).
    D_full
        Block-diagonal Wigner-D matrix with shape (E, D, D) where D=(lmax+1)^2.
        Used for efficient batched rotation. None if not available.
    Dt_full
        Transpose of D_full with shape (E, D, D). None if not available.
    D_to_m_cache
        Lazy cache for projected D matrices keyed by a normalized
        ``"lmax:mmax"`` identifier.
    Dt_from_m_cache
        Lazy cache for projected Dt matrices keyed by a normalized
        ``"lmax:mmax"`` identifier.
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


class SeZMTypeEmbedding(nn.Module):
    """
    Minimal SeZM type embedding with Adam-routed parameter naming.

    Parameters
    ----------
    ntypes
        Number of atom types.
    embed_dim
        Embedding dimension.
    dtype
        Parameter dtype.
    seed
        Random seed for initialization.
    trainable
        Whether parameters are trainable.
    padding
        Whether to append one all-zero padding row.

    Notes
    -----
    The parameter is named with ``adam_`` prefix so HybridMuon routes it to Adam.
    """

    def __init__(
        self,
        *,
        ntypes: int,
        embed_dim: int,
        dtype: torch.dtype,
        seed: int | list[int] | None = None,
        trainable: bool,
        padding: bool = True,
    ) -> None:
        super().__init__()
        self.ntypes = int(ntypes)
        self.embed_dim = int(embed_dim)
        self.dtype = dtype
        self.seed = seed
        self.device = env.DEVICE
        self.padding = bool(padding)
        if self.ntypes <= 0:
            raise ValueError("`ntypes` must be positive")
        if self.embed_dim <= 0:
            raise ValueError("`embed_dim` must be positive")

        # === Step 1. Build embedding table parameter ===
        n_rows = self.ntypes + int(self.padding)
        self.adam_type_embedding = nn.Parameter(
            torch.empty(
                n_rows,
                self.embed_dim,
                device=self.device,
                dtype=self.dtype,
            )
        )

        # === Step 2. Initialize active type rows with default normal scale ===
        init_std = 1.0 / math.sqrt(float(self.ntypes + self.embed_dim))
        nn.init.normal_(
            self.adam_type_embedding[: self.ntypes],
            mean=0.0,
            std=init_std,
            generator=get_generator(child_seed(seed, 0)),
        )
        if self.padding:
            with torch.no_grad():
                self.adam_type_embedding[self.ntypes].zero_()

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, atype: torch.Tensor) -> torch.Tensor:
        """
        Gather type embeddings.

        Parameters
        ----------
        atype
            Atom types with shape (...,). Valid type range is [0, ntypes-1].

        Returns
        -------
        torch.Tensor
            Type embeddings with shape (..., embed_dim).
        """
        return torch.embedding(self.adam_type_embedding, atype)


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

    Notes
    -----
    All bias terms are disabled (Linear bias=False, LayerNorm bias=False) to
    guarantee ``RadialMLP(0) = 0``. This is required because the compile path
    pads masked edges with zero ``edge_rbf``; any non-zero bias would leak
    spurious features into GIE scatter, causing energy divergence between
    compile and non-compile paths.
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
                bias=False,
                activation_function=None,
                precision=self.precision,
                seed=child_seed(seed, i),
                trainable=trainable,
            )
            modules.append(linear)
            # Last layer: no LayerNorm/activation
            if i < n_layers - 2:
                modules.append(
                    nn.LayerNorm(
                        mlp_layers[i + 1],
                        bias=False,
                        dtype=self.dtype,
                        device=self.device,
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


class C3CutoffEnvelope(torch.nn.Module):
    """
    C^3-continuous polynomial cutoff envelope function.

    This envelope provides a smooth transition to zero at the cutoff radius,
    ensuring continuity of the function value and the first three derivatives.

    Notes
    -----
    The envelope function is defined for scaled distance ``x = r / rcut`` as::

        E(x) = 1 + x^p * (a + b*x + c*x^2 + d*x^3),  for x < 1
        E(x) = 0,                                     for x >= 1

    where the coefficients are chosen to satisfy::

        E(0) = 1,    E(1) = 0
        E'(1) = 0,   E''(1) = 0,   E'''(1) = 0

    This ensures C^3 continuity at the cutoff boundary. The coefficients are::

        a = -(p + 1)(p + 2)(p + 3) / 6
        b = p(p + 2)(p + 3) / 2
        c = -p(p + 1)(p + 3) / 2
        d = p(p + 1)(p + 2) / 6

    For the default exponent p=5, the coefficients are a=-56, b=140, c=-120,
    d=35::

        E(x) = 1 + x^5 * (-56 + 140*x - 120*x^2 + 35*x^3)
             = 1 - 56*x^5 + 140*x^6 - 120*x^7 + 35*x^8

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
        Quadratic coefficient for x^(p+2) term.
    d : float
        Cubic coefficient for x^(p+3) term.
    """

    def __init__(self, rcut: float, exponent: int = 5) -> None:
        super().__init__()
        if exponent <= 0:
            raise ValueError("`exponent` must be positive")
        self.rcut = float(rcut)
        self.p: float = float(exponent)
        self.a: float = -((self.p + 1) * (self.p + 2) * (self.p + 3)) / 6.0
        self.b: float = (self.p * (self.p + 2) * (self.p + 3)) / 2.0
        self.c: float = -(self.p * (self.p + 1) * (self.p + 3)) / 2.0
        self.d: float = (self.p * (self.p + 1) * (self.p + 2)) / 6.0

    def forward(self, dst: torch.Tensor) -> torch.Tensor:
        """Compute the envelope value for given distances."""
        d_scaled = (dst / self.rcut).clamp(min=0.0, max=1.0)
        poly = self.a + d_scaled * (self.b + d_scaled * (self.c + d_scaled * self.d))
        env_val = 1 + (d_scaled**self.p) * poly
        return env_val * ((d_scaled < 1.0).to(dst.dtype))


class InnerClamp(nn.Module):
    """
    C3-continuous inner distance clamping for zone bridging.

    Applies a septic Hermite polynomial transition that freezes distances
    below ``r_inner`` to the constant ``r_inner``, then smoothly transitions
    back to identity at ``r_outer``::

        r̃(r) = r_inner                                    if r <= r_inner
        r̃(r) = r_inner + (r_outer - r_inner) * h(t)       if r_inner < r < r_outer
        r̃(r) = r                                          if r >= r_outer

        h(t) = 20t^4 - 45t^5 + 36t^6 - 10t^7,  t = (r - r_inner) / (r_outer - r_inner)

    Boundary conditions:
    ``h(0)=0``, ``h(1)=1``, ``h'(0)=0``, ``h'(1)=1``,
    ``h''(0)=0``, ``h''(1)=0``, ``h'''(0)=0``, ``h'''(1)=0``.
    This ensures C3 continuity: ``dr̃/dr = 0`` at r_inner (frozen zone) and
    ``dr̃/dr = 1`` at r_outer (identity zone), with matched second and third
    derivatives at both boundaries.

    Parameters
    ----------
    r_inner : float
        Freeze radius in Å. Distances below this are clamped to ``r_inner``.
    r_outer : float
        Outer boundary of the transition zone in Å. Above this, ``r̃ = r``.

    Raises
    ------
    ValueError
        If ``r_inner >= r_outer`` or either is non-positive.
    """

    def __init__(self, r_inner: float, r_outer: float) -> None:
        super().__init__()
        if r_inner <= 0 or r_outer <= 0:
            raise ValueError("r_inner and r_outer must be positive")
        if r_inner >= r_outer:
            raise ValueError(f"r_inner ({r_inner}) must be < r_outer ({r_outer})")
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Apply inner distance clamping.

        Parameters
        ----------
        r : torch.Tensor
            Pair distances with shape (...) or (..., 1) in Å.

        Returns
        -------
        torch.Tensor
            Clamped distances r̃ with the same shape as input.
        """
        t = ((r - self.r_inner) / (self.r_outer - self.r_inner)).clamp(0.0, 1.0)
        t2 = t * t
        t4 = t2 * t2
        # h(t) = 20t^4 - 45t^5 + 36t^6 - 10t^7
        # Satisfies:
        #   h(0)=0, h(1)=1
        #   h'(0)=0, h'(1)=1
        #   h''(0)=0, h''(1)=0
        #   h'''(0)=0, h'''(1)=0
        h = t4 * (20.0 + t * (-45.0 + t * (36.0 - 10.0 * t)))
        interpolated = self.r_inner + (self.r_outer - self.r_inner) * h
        # Identity zone: r >= r_outer returns r directly.
        # Both branches have matching first three derivatives at r_outer,
        # so torch.where preserves C3 continuity here.
        return torch.where(r >= self.r_outer, r, interpolated)


class RadialBasis(nn.Module):
    """
    Spherical Bessel radial basis with C^3 cutoff envelope.

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

    The C^3 cutoff envelope is multiplied directly into the output to ensure
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
        Exponent for the C^3 cutoff envelope polynomial. Default is 7.
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
        self.adam_freqs = nn.Parameter(
            rearrange(freqs, "n_radial -> 1 n_radial"), requires_grad=True
        )

        self.envelope = C3CutoffEnvelope(rcut=self.rcut, exponent=self.exponent)

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
            Radial basis multiplied by C^3 cutoff envelope with shape (N, n_rbf).
            The output is smoothly truncated to zero at r = rcut.
        """
        # === Step 1. Bessel Basis via Sinc ===
        # phi_n(r) = w_n * sinc(w_n * r / π)
        # Shape: (N, 1) * (1, n_radial) -> (N, n_radial)
        x = r * self.adam_freqs  # (N, n_rbf)
        raw = self.adam_freqs * torch.sinc(x / math.pi)  # (N, n_rbf)

        # === Step 2. Apply C^3 envelope for smooth cutoff ===
        envelope = self.envelope(r)  # (N, 1)
        return raw * envelope

    def serialize(self) -> dict[str, Any]:
        """Serialize RadialBasis including trainable frequencies."""
        state = self.state_dict()
        return {
            "@class": "RadialBasis",
            "@version": 1,
            "config": {
                "rcut": self.rcut,
                "n_radial": self.n_radial,
                "exponent": self.exponent,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
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
        config = data.pop("config", data)
        variables = data.pop("@variables", None)
        precision = config["precision"]
        dtype = PRECISION_DICT[precision]
        obj = cls(
            rcut=float(config["rcut"]),
            n_radial=int(config["n_radial"]),
            exponent=int(config.get("exponent", 7)),
            dtype=dtype,
        )
        if variables is not None:
            template = obj.state_dict()
            state = {
                key: safe_numpy_to_tensor(
                    value, device=template[key].device, dtype=template[key].dtype
                )
                for key, value in variables.items()
            }
            obj.load_state_dict(state)
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
        Maximum degree, should match ``l_schedule[0]``.
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
            packed_degree_by_row = map_degree_idx(self.lmax, device=self.device)
            # These aligned arrays describe one packed non-scalar row at a time.
            # non_scalar_row_index[k] picks the output row in the packed SO(3) layout.
            # zonal_m0_col_index_for_row[k] picks the matching m=0 column in Dt_full.
            # radial_slot_index_for_row[k] picks the matching degree slot in radial_feat.
            non_scalar_row_index = torch.arange(
                1, self.ebed_dim, device=self.device, dtype=torch.long
            )
            non_scalar_degree_by_row = packed_degree_by_row[1:]
            zonal_m0_col_index_for_row = non_scalar_degree_by_row * (
                non_scalar_degree_by_row + 1
            )
            radial_slot_index_for_row = non_scalar_degree_by_row - 1
            self.register_buffer(
                "non_scalar_row_index", non_scalar_row_index, persistent=True
            )
            self.register_buffer(
                "zonal_m0_col_index_for_row",
                zonal_m0_col_index_for_row,
                persistent=True,
            )
            self.register_buffer(
                "radial_slot_index_for_row",
                radial_slot_index_for_row,
                persistent=True,
            )
        else:
            self.register_buffer(
                "non_scalar_row_index",
                torch.empty(0, device=self.device, dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                "zonal_m0_col_index_for_row",
                torch.empty(0, device=self.device, dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                "radial_slot_index_for_row",
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
        # Advanced indexing pairs one packed non-scalar row with the zonal m=0 column
        # from the same degree block in Dt_full.
        Dt_full = edge_cache.Dt_full  # (E, D, D)
        zonal_m0_value_for_row = Dt_full[
            :,
            self.non_scalar_row_index,
            self.zonal_m0_col_index_for_row,
        ]  # (E, D-1)

        # === Step 3. Broadcast radial features per row ===
        # Each non-scalar packed row reuses the radial feature of its degree l.
        radial_value_for_row = radial_feat.index_select(
            1, self.radial_slot_index_for_row
        )  # (E, D-1, C)
        non_scalar_message = (
            zonal_m0_value_for_row.unsqueeze(-1) * radial_value_for_row
        )  # (E, D-1, C)

        # === Step 4. Scatter to nodes and normalize ===
        # Avoid advanced-index writeback (out[:, non_scalar_row_index, :]) which produces a copy.
        non_scalar_out = out.new_zeros(
            n_nodes, self.non_scalar_row_index.numel(), self.channels
        )  # (N, D-1, C)
        non_scalar_out.index_add_(0, edge_cache.dst, non_scalar_message)
        out[:, self.non_scalar_row_index, :] = non_scalar_out
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
    mlp_bias : bool
        Whether to enable bias terms in env-seed MLP layers
        (`rbf_proj_layer1/2` and `g_layer1/2`).
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
        mlp_bias: bool = True,
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
        self.mlp_bias = bool(mlp_bias)
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
            bias=self.mlp_bias,
            activation_function=self.activation_function,
            precision=self.precision,
            seed=child_seed(seed_rbf_proj, 0),
        )
        self.rbf_proj_layer2 = MLPLayer(
            self.rbf_out_dim,
            self.rbf_out_dim,
            bias=self.mlp_bias,
            activation_function=None,
            precision=self.precision,
            seed=child_seed(seed_rbf_proj, 1),
        )

        # === Independent type embedding: ntypes -> type_dim ===
        # Individual type embedding
        seed_type_embed = child_seed(seed, 1)
        self.env_type_embed = SeZMTypeEmbedding(
            ntypes=self.ntypes,
            embed_dim=self.type_dim,
            dtype=self.dtype,
            seed=seed_type_embed,
            trainable=trainable,
        )

        # === G network: (rbf_out_dim + 2*type_dim) -> hidden_dim -> embed_dim ===
        seed_g_net = child_seed(seed, 2)
        g_in_dim = self.rbf_out_dim + 2 * self.type_dim
        self.g_layer1 = MLPLayer(
            g_in_dim,
            self.hidden_dim,
            bias=self.mlp_bias,
            activation_function=self.activation_function,
            precision=self.precision,
            seed=child_seed(seed_g_net, 0),
        )
        self.g_layer2 = MLPLayer(
            self.hidden_dim,
            self.embed_dim,
            bias=self.mlp_bias,
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
        inv_r = torch.rsqrt(r_sq + self.eps * self.eps)  # (E, 1)
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

        # === Step 4. Smooth normalization by envelope-squared degree ===
        deg_scale = torch.rsqrt(edge_cache.deg + self.eps).reshape(
            -1, 1, 1
        )  # (N, 1, 1)
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
                "mlp_bias": self.mlp_bias,
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
        D_to_m_cache=None if cache.D_to_m_cache is None else {},
        Dt_from_m_cache=None if cache.Dt_from_m_cache is None else {},
    )


def safe_norm(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute vector norm with smooth epsilon regularization.

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
        Norm with shape (N, 1).
    """
    in_dtype = x.dtype
    if in_dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps**2)
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


def build_rotate_inv_rescale(
    lmax: int,
    mmax: int,
    degree_index: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build reduced-layout inverse-rotation rescale factors.

    When ``mmax < lmax``, the reduced local layout keeps only ``2*mmax+1`` orders
    for each degree ``l > mmax``. The inverse rotation rescales those truncated
    degrees by ``sqrt((2*l+1)/(2*mmax+1))`` so the reduced representation matches
    the amplitude expected by the full SO(3) basis.

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum order (|m|). Must satisfy ``0 <= mmax <= lmax``.
    degree_index
        Degree index aligned with the reduced coefficient layout, typically
        returned by ``build_m_major_l_index``.
    device
        Device for the returned tensor.
    dtype
        Floating-point dtype for the returned tensor.

    Returns
    -------
    torch.Tensor
        Rescale vector with shape (D_m_trunc,), aligned with the reduced
        coefficient layout.
    """
    lmax_i = int(lmax)
    mmax_i = int(mmax)
    if lmax_i < 0:
        raise ValueError("`lmax` must be non-negative")
    if mmax_i < 0:
        raise ValueError("`mmax` must be non-negative")
    if mmax_i > lmax_i:
        raise ValueError("`mmax` must be <= `lmax`")

    degrees = degree_index.to(device=device, dtype=torch.long)
    rescale = torch.ones(degrees.shape[0], device=device, dtype=dtype)
    if mmax_i == lmax_i:
        return rescale

    mask = degrees > mmax_i
    if mask.any():
        denom = float(2 * mmax_i + 1)
        degree_values = degrees[mask].to(dtype=dtype)
        rescale[mask] = torch.sqrt((2.0 * degree_values + 1.0) / denom)
    return rescale
