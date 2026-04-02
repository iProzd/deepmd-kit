# SPDX-License-Identifier: LGPL-3.0-or-later

import torch
import torch.distributed as dist
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

    Supports Expert Parallelism (EP) when ``ep_group`` is provided.
    In EP mode, only ``routed_experts // ep_size`` local experts are created
    per GPU. Token dispatch and combine use differentiable All-to-All that
    supports 2nd-order autograd (required for force training).

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
    ep_group : dist.ProcessGroup or None
        Process group for Expert Parallelism. None = single-GPU (all experts local).
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
        ep_group: dist.ProcessGroup | None = None,
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

        # Expert Parallelism config
        self.ep_group = ep_group
        if ep_group is not None:
            self.ep_size: int = dist.get_world_size(group=ep_group)
            self.ep_rank: int = dist.get_rank(group=ep_group)
            assert self.routed_experts % self.ep_size == 0, (
                f"routed_experts ({self.routed_experts}) must be divisible by "
                f"ep_size ({self.ep_size})"
            )
            self.experts_per_gpu: int = self.routed_experts // self.ep_size
        else:
            self.ep_size = 1
            self.ep_rank = 0
            self.experts_per_gpu = self.routed_experts

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

        # Routed experts — only local subset when EP is active
        self.experts = nn.ModuleList([
            MLPLayer(
                num_in,
                num_out,
                bias=bias,
                activation_function=activation_function,
                precision=precision,
                seed=child_seed(child_seed(seed, 1),
                                self.ep_rank * self.experts_per_gpu + i),
                trainable=trainable,
            )
            for i in range(self.experts_per_gpu)
        ])

        # Shared experts (always active, replicated on all GPUs)
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
            [nb, nloc, routed_top_k] — global expert IDs
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
            if self.ep_group is not None:
                return self._moe_ep_flat(x, flat_weights, flat_indices)
            return self._moe_flat(x, flat_weights, flat_indices)
        else:
            if self.ep_group is not None:
                return self._moe_ep_batched(x, atom_weights, atom_indices)
            return self._moe_batched(x, atom_weights, atom_indices)

    # ------------------------------------------------------------------ #
    #  Local (single-GPU) paths — unchanged from original
    # ------------------------------------------------------------------ #

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
        nb, nloc = atom_weights.shape[:2]
        per_expert_weight = torch.zeros(
            nb, nloc, self.routed_experts,
            dtype=x.dtype, device=x.device,
        )
        for k in range(self.routed_top_k):
            w_k = atom_weights[:, :, k]
            idx_k = atom_indices[:, :, k]
            per_expert_weight.scatter_add_(
                2, idx_k.unsqueeze(-1), w_k.unsqueeze(-1)
            )

        # Weighted sum of expert outputs
        output = torch.zeros_like(expert_outputs[0])
        n_expand = x.dim() - 3  # number of mid dims
        for e_idx in range(len(expert_outputs)):
            w_e = per_expert_weight[:, :, e_idx]
            for _ in range(n_expand):
                w_e = w_e.unsqueeze(-1)
            w_e = w_e.unsqueeze(-1)  # for output dim
            output = output + w_e * expert_outputs[e_idx]

        for se in self.shared_experts:
            output = output + se(x)
        return output

    # ------------------------------------------------------------------ #
    #  Expert Parallelism paths
    # ------------------------------------------------------------------ #

    def _ep_dispatch_info(
        self,
        flat_expert_ids: torch.Tensor,
    ) -> tuple[list[int], list[int], torch.Tensor, torch.Tensor]:
        """Compute All-to-All dispatch metadata (no gradients).

        Parameters
        ----------
        flat_expert_ids : torch.Tensor
            [N_tokens_expanded] — global expert ID for each expanded token.

        Returns
        -------
        send_splits : list[int]
            Number of tokens to send to each rank.
        recv_splits : list[int]
            Number of tokens to receive from each rank.
        send_perm : torch.Tensor
            Permutation indices to sort tokens by target GPU.
        recv_expert_ids : torch.Tensor
            Local expert IDs for received tokens.
        """
        target_gpu = flat_expert_ids // self.experts_per_gpu
        local_eid = flat_expert_ids % self.experts_per_gpu

        send_perm = torch.argsort(target_gpu, stable=True)
        send_splits = [
            (target_gpu == i).sum().item() for i in range(self.ep_size)
        ]

        # Exchange recv counts
        send_t = torch.tensor(
            send_splits, device=flat_expert_ids.device, dtype=torch.int64
        )
        recv_t = torch.zeros_like(send_t)
        dist.all_to_all_single(recv_t, send_t, group=self.ep_group)
        recv_splits = recv_t.tolist()

        # Exchange local expert IDs
        sorted_eids = local_eid[send_perm]
        recv_eids = torch.empty(
            sum(recv_splits),
            dtype=sorted_eids.dtype,
            device=sorted_eids.device,
        )
        dist.all_to_all_single(
            recv_eids,
            sorted_eids.contiguous(),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.ep_group,
        )
        return send_splits, recv_splits, send_perm, recv_eids

    def _ep_local_experts(
        self,
        h: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run each local expert on its assigned tokens.

        Uses clone-before-scatter to avoid in-place ops for autograd.
        Adds zero-contribution from unused experts to maintain autograd graph
        connectivity for 2nd-order derivatives.
        """
        out = torch.zeros(
            h.shape[0], self.num_out, dtype=h.dtype, device=h.device
        )
        # Maintain graph connectivity: h -> out.
        # When h is 0-element (this GPU received no tokens), h.sum() is a
        # scalar 0 with grad_fn, keeping the dispatch A2A in the autograd
        # graph so that gradients flow back through A2A to the original input.
        ghost = h.sum() * 0.0
        # Track unused experts for graph connectivity
        unused_expert_sum = torch.tensor(
            0.0, dtype=h.dtype, device=h.device
        )
        for i, expert in enumerate(self.experts):
            mask = expert_ids == i
            if mask.any():
                result = expert(h[mask])
                out = out.clone()  # avoid in-place for autograd
                out[mask] = result
            else:
                # Add zero-contribution to keep expert in computation graph.
                # This ensures 2nd-order gradients flow through all params.
                # IMPORTANT: Always use torch.zeros (not h[:1]) to avoid
                # creating backward paths through A2A that would differ
                # between ranks, causing NCCL deadlocks with create_graph=True.
                # The `ghost` term already handles A2A graph connectivity.
                dummy = expert(torch.zeros(
                    1, h.shape[-1], dtype=h.dtype, device=h.device
                ))
                unused_expert_sum = unused_expert_sum + dummy.sum() * 0.0
        # Add zero-valued terms that connect h and unused experts to the graph
        out = out + ghost + unused_expert_sum
        return out

    def _moe_ep_flat(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """EP MoE for flat (dynamic_sel) inputs: x [n_flat, dim].

        Each token is expanded for top_k experts, dispatched to expert-owning
        GPUs via differentiable All-to-All, computed locally, and combined back.

        Uses fused dispatch+compute+combine (ep_moe_forward) to prevent NCCL
        deadlocks from autograd topological sort reordering.
        """
        from deepmd.pt.model.network.moe_ep_ops import ep_moe_forward

        n_flat = x.shape[0]
        dim = x.shape[-1]

        # Expand tokens for top_k
        # x_expanded: [n_flat * routed_top_k, dim]
        x_expanded = x.unsqueeze(1).expand(
            -1, self.routed_top_k, -1
        ).reshape(-1, dim)
        flat_expert_ids = indices.reshape(-1)
        flat_weights = weights.reshape(-1)  # [n_flat * routed_top_k]

        # Dispatch metadata
        send_splits, recv_splits, send_perm, recv_eids = (
            self._ep_dispatch_info(flat_expert_ids)
        )

        # Sort tokens by target GPU (indexing is differentiable)
        x_sorted = x_expanded[send_perm]

        # Fused dispatch A2A + expert compute + combine A2A
        h_ret, used_experts = ep_moe_forward(
            x_sorted, send_splits, recv_splits, self.ep_group,
            self.experts, recv_eids, self.num_out,
        )

        # Add ghost terms for ALL experts (not just unused) to ensure
        # identical autograd graph structure on all ranks. Different ranks
        # have different used_experts sets, so conditionally skipping
        # experts would create asymmetric graphs and cause NCCL deadlocks
        # from topological sort reordering during loss.backward().
        ghost_sum = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        for expert in self.experts:
            dummy = expert(torch.zeros(
                1, dim, dtype=x.dtype, device=x.device
            ))
            ghost_sum = ghost_sum + dummy.sum() * 0.0
        h_ret = h_ret + ghost_sum

        # Un-permute
        h_unperm = torch.zeros_like(x_expanded[:, :self.num_out])
        h_unperm = h_unperm.clone()
        h_unperm[send_perm] = h_ret

        # Weighted sum over top_k
        h_topk = h_unperm.reshape(n_flat, self.routed_top_k, self.num_out)
        w_topk = flat_weights.reshape(n_flat, self.routed_top_k)
        output = (h_topk * w_topk.unsqueeze(-1)).sum(dim=1)

        # Add shared experts (local, no dispatch)
        for se in self.shared_experts:
            output = output + se(x)
        return output

    def _moe_ep_batched(
        self,
        x: torch.Tensor,
        atom_weights: torch.Tensor,
        atom_indices: torch.Tensor,
    ) -> torch.Tensor:
        """EP MoE for batched inputs with arbitrary middle dims.

        x: [nb, nloc, *mid, dim_in]
        atom_weights: [nb, nloc, routed_top_k]
        atom_indices: [nb, nloc, routed_top_k]

        For batched inputs (edge: [nb,nloc,nnei,dim], angle: [nb,nloc,a,a,dim]),
        we flatten to [N_tokens, dim], do EP dispatch/compute/combine, then
        reshape back. The routing weights are broadcast from [nb,nloc] to match
        the mid dimensions.

        Uses fused dispatch+compute+combine (ep_moe_forward) to prevent NCCL
        deadlocks from autograd topological sort reordering.
        """
        from deepmd.pt.model.network.moe_ep_ops import ep_moe_forward

        orig_shape = x.shape  # [nb, nloc, *mid, dim_in]
        nb, nloc = atom_weights.shape[:2]
        dim_in = x.shape[-1]
        mid_shape = orig_shape[2:-1]  # e.g., (nnei,) or (a_nnei, a_nnei)
        mid_size = 1
        for s in mid_shape:
            mid_size *= s

        # Flatten: [nb, nloc, *mid, dim_in] -> [nb * nloc * mid_size, dim_in]
        x_flat = x.reshape(nb * nloc * mid_size, dim_in)

        # Expand routing to match flattened tokens
        # atom_indices: [nb, nloc, routed_top_k] -> repeat for mid dims
        # Each atom's mid_size tokens share the same routing
        # [nb * nloc, routed_top_k] -> [nb * nloc * mid_size, routed_top_k]
        atom_indices_flat = atom_indices.reshape(nb * nloc, self.routed_top_k)
        atom_weights_flat = atom_weights.reshape(nb * nloc, self.routed_top_k)
        if mid_size > 1:
            # Repeat each atom's routing for all its edges/angles
            atom_indices_flat = atom_indices_flat.unsqueeze(1).expand(
                -1, mid_size, -1
            ).reshape(-1, self.routed_top_k)
            atom_weights_flat = atom_weights_flat.unsqueeze(1).expand(
                -1, mid_size, -1
            ).reshape(-1, self.routed_top_k)

        n_tokens = x_flat.shape[0]

        # Expand for top_k: [n_tokens * routed_top_k, dim_in]
        x_expanded = x_flat.unsqueeze(1).expand(
            -1, self.routed_top_k, -1
        ).reshape(-1, dim_in)
        flat_expert_ids = atom_indices_flat.reshape(-1)
        flat_weights = atom_weights_flat.reshape(-1)

        # Dispatch metadata
        send_splits, recv_splits, send_perm, recv_eids = (
            self._ep_dispatch_info(flat_expert_ids)
        )

        # Sort by target GPU
        x_sorted = x_expanded[send_perm]

        # Fused dispatch A2A + expert compute + combine A2A
        h_ret, used_experts = ep_moe_forward(
            x_sorted, send_splits, recv_splits, self.ep_group,
            self.experts, recv_eids, self.num_out,
        )

        # Add ghost terms for ALL experts (not just unused) to ensure
        # identical autograd graph structure on all ranks. See _moe_ep_flat.
        ghost_sum = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        for expert in self.experts:
            dummy = expert(torch.zeros(
                1, dim_in, dtype=x.dtype, device=x.device
            ))
            ghost_sum = ghost_sum + dummy.sum() * 0.0
        h_ret = h_ret + ghost_sum

        # Un-permute
        h_unperm = torch.zeros(
            n_tokens * self.routed_top_k, self.num_out,
            dtype=x.dtype, device=x.device,
        )
        h_unperm = h_unperm.clone()
        h_unperm[send_perm] = h_ret

        # Weighted sum over top_k
        h_topk = h_unperm.reshape(n_tokens, self.routed_top_k, self.num_out)
        w_topk = flat_weights.reshape(n_tokens, self.routed_top_k)
        output_flat = (h_topk * w_topk.unsqueeze(-1)).sum(dim=1)

        # Reshape back to original shape
        output = output_flat.reshape(nb, nloc, *mid_shape, self.num_out)

        # Add shared experts (local, no dispatch)
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
