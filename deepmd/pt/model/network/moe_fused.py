# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fused MoE layer for reducing All-to-All communication overhead.

When multiple MoE layers share the same input, we can fuse them into a single
wider MoE layer to reduce communication rounds from 2N to 2 (N dispatch + N combine
becomes 1 dispatch + 1 combine).
"""

import torch
import torch.distributed as dist

from deepmd.pt.model.network.moe import MoELayer


class FusedMoELayer(torch.nn.Module):
    """Fused MoE layer that combines multiple MoE outputs in one pass.

    This reduces All-to-All communication overhead when multiple MoE layers
    share the same input tensor.
    """

    def __init__(
        self,
        num_in: int,
        output_dims: list[int],
        n_experts: int = 1,
        top_k: int = 1,
        tebd_dim: int = 0,
        share_expert: int = 0,
        bias: bool = True,
        activation_function: str | None = None,
        precision: str = "float64",
        seed: int | list[int] | None = None,
        trainable: bool = True,
        ep_group: "torch.distributed.ProcessGroup | None" = None,
        gpu_level_a2a: bool = False,
    ):
        super().__init__()
        self.num_in = num_in
        self.output_dims = output_dims
        self.num_outputs = len(output_dims)
        self.total_out = sum(output_dims)

        # Single wide MoE layer
        self.fused_moe = MoELayer(
            num_in=num_in,
            num_out=self.total_out,
            n_experts=n_experts,
            top_k=top_k,
            tebd_dim=tebd_dim,
            share_expert=share_expert,
            bias=bias,
            activation_function=activation_function,
            precision=precision,
            seed=seed,
            trainable=trainable,
            ep_group=ep_group,
            gpu_level_a2a=gpu_level_a2a,
        )

    def forward(
        self,
        x: torch.Tensor,
        type_embeddings: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Forward pass returning list of outputs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        type_embeddings : torch.Tensor
            Type embedding table.
        atom_types : torch.Tensor
            Atom types.
        edge_index : torch.Tensor or None
            For dynamic_sel inputs.

        Returns
        -------
        list[torch.Tensor]
            List of output tensors, one per output_dim.
        """
        # Single MoE forward (1 A2A dispatch + 1 A2A combine)
        fused_out = self.fused_moe(x, type_embeddings, atom_types, edge_index)

        # Split into individual outputs
        outputs = torch.split(fused_out, self.output_dims, dim=-1)
        return list(outputs)

    def serialize(self) -> dict:
        """Serialize the fused MoE layer."""
        return {
            "@class": "FusedMoELayer",
            "@version": 1,
            "num_in": self.num_in,
            "output_dims": self.output_dims,
            "fused_moe": self.fused_moe.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "FusedMoELayer":
        """Deserialize the fused MoE layer."""
        from deepmd.utils.version import check_version_compatibility

        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        fused_moe_data = data.pop("fused_moe")

        obj = cls(**data)
        obj.fused_moe = MoELayer.deserialize(fused_moe_data)
        return obj
