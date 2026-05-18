# SPDX-License-Identifier: LGPL-3.0-or-later
"""Top-k router for SeZM SO(2) MoE experts."""

from __future__ import (
    annotations,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepmd.pt.model.network.mlp import (
    MLPLayer,
)

_VALID_ROUTING_INPUTS = frozenset({"dst", "src", "src+dst"})


class MoESO2Router(nn.Module):
    """Top-k gating router for SeZM SO(2) MoE.

    The caller is responsible for constructing the per-edge routing key from
    source and/or destination type embeddings. This module only maps a
    ``(E, input_dim)`` tensor to top-k expert weights and global expert indices.
    """

    def __init__(
        self,
        input_dim: int,
        n_routing_experts: int,
        topk: int,
        routing_input: str,
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        if routing_input not in _VALID_ROUTING_INPUTS:
            valid = "', '".join(sorted(_VALID_ROUTING_INPUTS))
            raise ValueError(
                "SeZM MoE router requires `routing_input` to be one of "
                f"'{valid}', got {routing_input!r}."
            )
        if not 1 <= topk <= n_routing_experts:
            raise ValueError(
                "SeZM MoE router requires `1 <= topk <= n_routing_experts`, "
                f"got topk={topk}, n_routing_experts={n_routing_experts}."
            )

        self.input_dim = input_dim
        self.n_routing_experts = n_routing_experts
        self.topk = topk
        self.routing_input = routing_input
        self.gate = MLPLayer(
            input_dim,
            n_routing_experts,
            activation_function=None,
            bias=False,
            precision=precision,
            seed=seed,
        )

    def forward(
        self,
        type_emb_per_edge: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k routing weights and global expert indices.

        Parameters
        ----------
        type_emb_per_edge
            Per-edge routing keys with shape ``(E, input_dim)``.
        """
        if (
            type_emb_per_edge.dim() != 2
            or type_emb_per_edge.shape[-1] != self.input_dim
        ):
            raise ValueError(
                "SeZM MoE router expects `type_emb_per_edge` with shape "
                f"(E, {self.input_dim}), got {tuple(type_emb_per_edge.shape)}."
            )

        logits = self.gate(type_emb_per_edge)  # (E, n_routing_experts)
        topk_logits, topk_indices = torch.topk(
            logits,
            k=self.topk,
            dim=-1,
        )  # (E, topk), (E, topk)
        topk_weights = F.softmax(topk_logits, dim=-1)  # (E, topk)
        return topk_weights, topk_indices


__all__ = ["MoESO2Router"]
