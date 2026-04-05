# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Radial building blocks for the SeZM descriptor.

This module defines the cutoff envelope, inner-distance clamp, radial basis,
and radial multilayer perceptron used by SeZM.
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
)

from .utils import (
    np_safe,
    safe_numpy_to_tensor,
)


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
