# SPDX-License-Identifier: LGPL-3.0-or-later
"""MoE Router for Expert Parallelism.

Each feature group (node / edge / angle) has its own independent MoERouter
instance, but they all receive the same input: the center-atom type embedding.

The router produces top-k expert indices and softmax-normalised weights.
Edge / angle routing is derived from node routing via index broadcasting
(done at call site, not inside this module).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepmd.pt.model.network.mlp import (
    MLPLayer,
)


class MoERouter(nn.Module):
    """Top-k gating router for MoE.

    Parameters
    ----------
    input_dim : int
        Dimension of the input type embedding (= n_dim in DPA3).
    n_routing_experts : int
        Number of routing experts (excluding shared experts).
    topk : int
        Number of experts selected per token.
    precision : str
        Floating-point precision for the gate linear layer.
    seed : int or list[int] or None
        Random seed for reproducible initialisation.
    """

    def __init__(
        self,
        input_dim: int,
        n_routing_experts: int,
        topk: int,
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_routing_experts = n_routing_experts
        self.topk = topk
        # Gate: linear projection, no activation, no bias (standard MoE practice).
        self.gate = MLPLayer(
            input_dim,
            n_routing_experts,
            activation_function=None,
            bias=False,
            precision=precision,
            seed=seed,
        )

    def forward(self, type_embedding: torch.Tensor):
        """Compute top-k routing weights and indices.

        Parameters
        ----------
        type_embedding : Tensor, shape ``[nb, nloc, input_dim]``
            Center-atom type embedding.

        Returns
        -------
        topk_weights : Tensor, shape ``[nb*nloc, topk]``
            Softmax-normalised routing weights.
        topk_indices : Tensor, shape ``[nb*nloc, topk]``
            Selected expert global indices in ``[0, n_routing_experts)``.
        """
        logits = self.gate(type_embedding)  # [nb, nloc, n_routing_experts]
        logits = logits.reshape(-1, self.n_routing_experts)  # [N_node, n_routing_experts]
        topk_logits, topk_indices = torch.topk(logits, k=self.topk, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)
        return topk_weights, topk_indices
