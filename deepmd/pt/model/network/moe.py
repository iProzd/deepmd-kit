# SPDX-License-Identifier: LGPL-3.0-or-later

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class MoELayer(nn.Module):
    """Mixture of Experts layer that replaces MLPLayer.

    The router uses type embeddings (not input features) to compute expert
    weights. Since there are only `ntypes` unique type embeddings, the router
    computes a [ntypes, n_experts] weight matrix once, then indexes by atom
    type to get per-atom weights. This ensures MD continuity.

    Parameters
    ----------
    num_in : int
        Input dimension.
    num_out : int
        Output dimension.
    n_experts : int
        Total number of experts (routed + shared).
    top_k : int
        Number of experts to activate per token (routed + shared).
    tebd_dim : int
        Dimension of type embeddings (for the router gate).
    share_expert : int
        Number of shared experts (always active, counted within top_k).
    bias : bool
        Whether to use bias in expert MLPLayers.
    activation_function : str or None
        Activation function inside each expert.
    precision : str
        Precision for parameters.
    seed : int or list[int] or None
        Random seed.
    trainable : bool
        Whether parameters are trainable.
    """

    def __init__(
        self,
        num_in: int,
        num_out: int,
        n_experts: int,
        top_k: int,
        tebd_dim: int,
        share_expert: int = 0,
        bias: bool = True,
        activation_function: str | None = None,
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        assert n_experts > 1, "MoELayer requires n_experts > 1"
        assert share_expert < top_k, (
            f"share_expert ({share_expert}) must be < top_k ({top_k})"
        )
        self.num_in = num_in
        self.num_out = num_out
        self.n_experts = n_experts
        self.top_k = top_k
        self.tebd_dim = tebd_dim
        self.share_expert = share_expert
        self.precision = precision
        self.activation_function = activation_function
        self.routed_experts: int = n_experts - share_expert
        self.routed_top_k: int = top_k - share_expert

        # Router gate: maps type embedding to routed expert logits
        self.gate = MLPLayer(
            tebd_dim,
            self.routed_experts,
            bias=False,
            activation_function=None,
            precision=precision,
            seed=child_seed(seed, 0),
            trainable=trainable,
        )

        # Routed experts
        self.experts = nn.ModuleList([
            MLPLayer(
                num_in,
                num_out,
                bias=bias,
                activation_function=activation_function,
                precision=precision,
                seed=child_seed(child_seed(seed, 1), i),
                trainable=trainable,
            )
            for i in range(self.routed_experts)
        ])

        # Shared experts (always active)
        if self.share_expert > 0:
            self.shared_experts = nn.ModuleList([
                MLPLayer(
                    num_in,
                    num_out,
                    bias=bias,
                    activation_function=activation_function,
                    precision=precision,
                    seed=child_seed(child_seed(seed, 2), i),
                    trainable=trainable,
                )
                for i in range(self.share_expert)
            ])
        else:
            self.shared_experts = nn.ModuleList()

    def _route(
        self,
        type_embeddings: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights and indices per atom.

        Parameters
        ----------
        type_embeddings : torch.Tensor
            [ntypes, tebd_dim]
        atom_types : torch.Tensor
            [nb, nloc]

        Returns
        -------
        atom_weights : torch.Tensor
            [nb, nloc, routed_top_k]
        atom_indices : torch.Tensor
            [nb, nloc, routed_top_k]
        """
        # logits: [ntypes, routed_experts]
        logits = self.gate(type_embeddings)
        topk_logits, topk_indices = torch.topk(
            logits, k=self.routed_top_k, dim=-1
        )
        # weights: [ntypes, routed_top_k]
        weights = F.softmax(topk_logits, dim=-1)
        # Map to atoms: [nb, nloc, routed_top_k]
        atom_weights = weights[atom_types]
        atom_indices = topk_indices[atom_types]
        return atom_weights, atom_indices

    def forward(
        self,
        x: torch.Tensor,
        type_embeddings: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Can be:
            - Node: [nb, nloc, dim]
            - Edge (non-dynamic): [nb, nloc, nnei, dim]
            - Edge (dynamic): [n_edge, dim]
            - Angle (non-dynamic): [nb, nloc, a_nnei, a_nnei, dim]
            - Angle (dynamic): [n_angle, dim]
        type_embeddings : torch.Tensor
            Type embedding table, [ntypes, tebd_dim].
        atom_types : torch.Tensor
            Atom types, [nb, nloc].
        edge_index : torch.Tensor or None
            For dynamic_sel edge/angle inputs, maps flat indices to center
            atom indices in [0, nb*nloc). Shape: [n_edge] or [n_angle].

        Returns
        -------
        torch.Tensor
            Output with same shape as input except last dim = num_out.
        """
        atom_weights, atom_indices = self._route(type_embeddings, atom_types)
        nb, nloc = atom_types.shape

        if edge_index is not None:
            # Dynamic sel: x is [n_flat, dim], edge_index maps to center atom
            flat_weights = atom_weights.reshape(nb * nloc, self.routed_top_k)[
                edge_index
            ]
            flat_indices = atom_indices.reshape(nb * nloc, self.routed_top_k)[
                edge_index
            ]
            return self._moe_flat(x, flat_weights, flat_indices)
        else:
            return self._moe_batched(x, atom_weights, atom_indices)

    def _moe_flat(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """MoE for flat (dynamic_sel) inputs: x [n_flat, dim]."""
        n_flat = x.shape[0]
        output = torch.zeros(
            n_flat, self.num_out, dtype=x.dtype, device=x.device
        )
        for k in range(self.routed_top_k):
            w_k = weights[:, k]
            idx_k = indices[:, k]
            for e_idx, expert in enumerate(self.experts):
                mask = idx_k == e_idx
                if mask.any():
                    expert_out = expert(x[mask])
                    output[mask] = (
                        output[mask] + w_k[mask].unsqueeze(-1) * expert_out
                    )
        for se in self.shared_experts:
            output = output + se(x)
        return output

    def _moe_batched(
        self,
        x: torch.Tensor,
        atom_weights: torch.Tensor,
        atom_indices: torch.Tensor,
    ) -> torch.Tensor:
        """MoE for batched inputs with arbitrary middle dims.

        Since routing is per-atom (type-based), all edges/angles of the same
        atom share the same routing. We compute each expert on the full input,
        then weight by routing assignment. This is efficient because n_experts
        is small and avoids dynamic shape manipulation (JIT-friendly).

        x: [nb, nloc, *mid, dim_in]
        atom_weights: [nb, nloc, routed_top_k]
        atom_indices: [nb, nloc, routed_top_k]
        """
        # Compute all expert outputs: list of [nb, nloc, *mid, num_out]
        expert_outputs: list[torch.Tensor] = []
        for _e_idx, expert in enumerate(self.experts):
            expert_outputs.append(expert(x))

        # Build per-expert weight mask from routing
        # For each expert e, weight_e[nb, nloc] = sum of w_k where idx_k == e
        nb, nloc = atom_weights.shape[:2]
        # per_expert_weight: [nb, nloc, routed_experts]
        per_expert_weight = torch.zeros(
            nb, nloc, self.routed_experts,
            dtype=x.dtype, device=x.device,
        )
        for k in range(self.routed_top_k):
            w_k = atom_weights[:, :, k]  # [nb, nloc]
            idx_k = atom_indices[:, :, k]  # [nb, nloc]
            per_expert_weight.scatter_add_(
                2, idx_k.unsqueeze(-1), w_k.unsqueeze(-1)
            )

        # Weighted sum of expert outputs
        # per_expert_weight[:, :, e] needs to broadcast over mid dims
        output = torch.zeros_like(expert_outputs[0])
        n_expand = x.dim() - 3  # number of mid dims
        for e_idx in range(len(expert_outputs)):
            # w_e: [nb, nloc] -> broadcast to [nb, nloc, *mid, 1]
            w_e = per_expert_weight[:, :, e_idx]
            # Expand dims for mid dimensions
            for _ in range(n_expand):
                w_e = w_e.unsqueeze(-1)
            w_e = w_e.unsqueeze(-1)  # for output dim
            output = output + w_e * expert_outputs[e_idx]

        for se in self.shared_experts:
            output = output + se(x)
        return output

    def serialize(self) -> dict:
        """Serialize the MoE layer to a dict."""
        data = {
            "@class": "MoELayer",
            "@version": 1,
            "num_in": self.num_in,
            "num_out": self.num_out,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "tebd_dim": self.tebd_dim,
            "share_expert": self.share_expert,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "gate": self.gate.serialize(),
            "experts": [e.serialize() for e in self.experts],
            "shared_experts": [e.serialize() for e in self.shared_experts],
        }
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "MoELayer":
        """Deserialize the MoE layer from a dict."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        gate_data = data.pop("gate")
        experts_data = data.pop("experts")
        shared_experts_data = data.pop("shared_experts")

        obj = cls(**data)
        obj.gate = MLPLayer.deserialize(gate_data)
        obj.experts = nn.ModuleList(
            [MLPLayer.deserialize(e) for e in experts_data]
        )
        obj.shared_experts = nn.ModuleList(
            [MLPLayer.deserialize(e) for e in shared_experts_data]
        )
        return obj
