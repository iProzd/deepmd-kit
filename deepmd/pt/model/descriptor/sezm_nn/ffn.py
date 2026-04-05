# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Equivariant feed-forward layers for SeZM.

This module defines the full SO(3)-equivariant feed-forward network used
inside SeZM interaction blocks and the descriptor output head.
"""

from __future__ import (
    annotations,
)

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

from .so3 import (
    GatedActivation,
    SO3Linear,
)
from .utils import (
    np_safe,
    safe_numpy_to_tensor,
)


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
        Whether to use bias in SO3Linear (l=0 bias) and GatedActivation
        (gate linear bias).
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
