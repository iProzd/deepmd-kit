# SPDX-License-Identifier: LGPL-3.0-or-later
"""MoE Expert MLP layers for Expert Parallelism.

Provides:
- ``ExpertMLPLayer``: a single expert wrapping ``MLPLayer`` to compute
  ``act(x @ W + b)`` with independent W and b parameters.
- ``MoEExpertCollection``: manages a set of local routing experts plus
  optional shared experts for one MLP position in a RepFlowLayer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
)


class ExpertMLPLayer(nn.Module):
    """Single expert MLP: ``act(x @ W + b)``.

    Each expert has its own independent W and b (not shared).
    Internally wraps ``MLPLayer`` for consistent initialisation, device
    handling and precision management.

    Parameters
    ----------
    num_in : int
        Input feature dimension.
    num_out : int
        Output feature dimension.
    activation_function : str
        Name of the activation function (e.g. ``"silu"``).
    precision : str
        Floating-point precision (e.g. ``"float64"``).
    seed : int, list[int], or None
        Random seed for reproducible weight initialisation.
    """

    def __init__(
        self,
        num_in: int,
        num_out: int,
        activation_function: str = "silu",
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.activation_function = activation_function
        # MLPLayer handles weight init, device placement, and precision.
        # activation_function=None so we control activation ourselves —
        # important for merged-MLP scenarios (M_edge, M_angle) where
        # activation is applied after the merged linear, not inside MLPLayer.
        self.mlp = MLPLayer(
            num_in,
            num_out,
            bias=True,
            activation_function=None,
            precision=precision,
            seed=seed,
        )
        self.activate = ActivationFn(activation_function)

    @property
    def matrix(self) -> nn.Parameter:
        """Weight matrix W, shape ``[num_in, num_out]``."""
        return self.mlp.matrix

    @property
    def bias(self) -> nn.Parameter:
        """Bias vector b, shape ``[num_out]``."""
        return self.mlp.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape ``[N_tokens, num_in]``

        Returns
        -------
        Tensor, shape ``[N_tokens, num_out]``
        """
        return self.activate(self.mlp(x))


class MoEExpertCollection(nn.Module):
    """Collection of local routing experts and shared experts for one MLP position.

    Parameters
    ----------
    num_in : int
        Input feature dimension.
    num_out : int
        Output feature dimension.
    experts_per_gpu : int
        Number of routing experts held on this GPU.
    n_shared_experts : int
        Number of shared experts (replicated on every GPU).
    activation_function : str
        Activation function name.
    precision : str
        Floating-point precision.
    seed : int, list[int], or None
        Parent seed; child seeds are derived per expert.
    """

    def __init__(
        self,
        num_in: int,
        num_out: int,
        experts_per_gpu: int,
        n_shared_experts: int = 0,
        activation_function: str = "silu",
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.experts_per_gpu = experts_per_gpu
        self.n_shared_experts = n_shared_experts

        self.routing_experts = nn.ModuleList([
            ExpertMLPLayer(
                num_in, num_out, activation_function, precision,
                seed=child_seed(seed, i),
            )
            for i in range(experts_per_gpu)
        ])

        if n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                ExpertMLPLayer(
                    num_in, num_out, activation_function, precision,
                    seed=child_seed(seed, experts_per_gpu + i),
                )
                for i in range(n_shared_experts)
            ])
        else:
            self.shared_experts = nn.ModuleList()

    def forward_expert(
        self, x: torch.Tensor, local_expert_idx: int
    ) -> torch.Tensor:
        """Compute output using the specified routing expert.

        Parameters
        ----------
        x : Tensor, shape ``[N_tokens, num_in]``
        local_expert_idx : int
            Index into ``self.routing_experts``.

        Returns
        -------
        Tensor, shape ``[N_tokens, num_out]``
        """
        return self.routing_experts[local_expert_idx](x)

    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """Compute output using all shared experts, summed.

        Parameters
        ----------
        x : Tensor, shape ``[N_tokens, num_in]``

        Returns
        -------
        Tensor, shape ``[N_tokens, num_out]``
            Sum of all shared expert outputs.
            Zero tensor if no shared experts.
        """
        if len(self.shared_experts) == 0:
            return torch.zeros(
                x.shape[0], self.num_out, dtype=x.dtype, device=x.device,
            )
        result = self.shared_experts[0](x)
        for se in self.shared_experts[1:]:
            result = result + se(x)
        return result
