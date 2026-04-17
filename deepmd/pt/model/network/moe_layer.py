# SPDX-License-Identifier: LGPL-3.0-or-later
"""MoE Dispatch-Compute-Combine pipeline for Expert Parallelism.

Provides ``MoEDispatchCombine``, which manages one RepFlowLayer's Phase 1
MoE computation: routing, topk expansion, sorting, packing, All-to-All
dispatch, expert compute, All-to-All combine, unsort, weighted sum, and
shared experts.

The module handles four MoE MLP positions:
- M1 (node_self): nd -> nd
- M2 (node_sym):  n_sym_dim -> nd
- M_edge (merged M3+M4): edge_info_dim -> nd+ne
- M_angle (merged M5+M7): angle_dim -> ne+na

Single-GPU path (``ep_group is None``) bypasses all packing / A2A logic
and runs a simple per-expert for-loop with weighted aggregation.
"""

from __future__ import annotations

from typing import (
    Optional,
)

import torch
import torch.distributed as dist
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.moe_ep_ops import (
    all_to_all_differentiable,
)
from deepmd.pt.model.network.moe_expert import (
    MoEExpertCollection,
)
from deepmd.pt.model.network.moe_packer import (
    MoEPacker,
    counts_to_packed_rows,
    exchange_metadata,
    validate_dim_ratio,
)


def _topk_expand_sort(
    features: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    experts_per_gpu: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], int]:
    """Expand tokens by topk and sort by global expert ID.

    Sorting by global expert ID (instead of just target GPU) ensures that
    within each GPU's chunk, tokens are additionally sorted by local expert
    ID.  This allows the recv side to skip argsort entirely and reorder
    via a structured O(N) gather built from per-sender per-expert counts.

    Parameters
    ----------
    features : Tensor, shape ``[N, feat_dim]``
        Token features.
    topk_indices : Tensor, shape ``[N, topk]``
        Global expert indices per token.
    topk_weights : Tensor, shape ``[N, topk]``
        Routing weights per token.
    experts_per_gpu : int
        Number of routing experts on each GPU.

    Returns
    -------
    sorted_features : Tensor ``[N*topk, feat_dim]``
    sorted_expert_ids : Tensor ``[N*topk]``
        Global expert IDs in sorted order.
    sorted_weights : Tensor ``[N*topk]``
        Flattened weights in sorted order (for weighted sum after combine).
    unsort_idx : Tensor ``[N*topk]``
        Index to restore original order from sorted order.
    counts_per_gpu : list[int]
        Number of expanded tokens destined for each GPU.
    ep_size : int
        Number of GPUs (derived from max expert index).
    """
    N, topk = topk_indices.shape

    # Flatten: [N, topk] -> [N*topk]
    flat_indices = topk_indices.reshape(-1)  # global expert ids
    flat_weights = topk_weights.reshape(-1)

    # Expand features: [N, feat_dim] -> [N*topk, feat_dim]
    expanded = features.repeat_interleave(topk, dim=0)

    # Sort by global expert ID (stable).  Since global_eid =
    # target_gpu * experts_per_gpu + local_eid, this is a refinement
    # of sorting by target_gpu: tokens for GPU 0 come first (sorted
    # by local_eid within), then GPU 1, etc.
    sort_idx = torch.argsort(flat_indices, stable=True)
    sorted_features = expanded[sort_idx]
    sorted_expert_ids = flat_indices[sort_idx]
    sorted_weights = flat_weights[sort_idx]

    # Compute inverse permutation for unsort.
    unsort_idx = torch.empty_like(sort_idx)
    unsort_idx[sort_idx] = torch.arange(len(sort_idx), device=sort_idx.device)

    # Counts per GPU via bincount on target_gpu.
    sorted_target_gpu = sorted_expert_ids // experts_per_gpu
    ep_size_inferred = int(sorted_target_gpu.max().item()) + 1 if len(sorted_target_gpu) > 0 else 1
    gpu_counts = torch.bincount(sorted_target_gpu, minlength=ep_size_inferred)
    counts_per_gpu = gpu_counts.tolist()

    return sorted_features, sorted_expert_ids, sorted_weights, unsort_idx, counts_per_gpu, ep_size_inferred


def _weighted_sum_topk(
    expanded_output: torch.Tensor,
    weights: torch.Tensor,
    n_orig: int,
    topk: int,
) -> torch.Tensor:
    """Aggregate topk outputs via weighted sum.

    Parameters
    ----------
    expanded_output : Tensor ``[N_orig * topk, out_dim]``
        Expert outputs in original (unsorted) token order.
    weights : Tensor ``[N_orig * topk]``
        Routing weights in original order.
    n_orig : int
        Number of original tokens.
    topk : int

    Returns
    -------
    Tensor ``[N_orig, out_dim]``
    """
    out_dim = expanded_output.shape[-1]
    # [N_orig, topk, out_dim]
    reshaped = expanded_output.reshape(n_orig, topk, out_dim)
    # [N_orig, topk, 1]
    w = weights.reshape(n_orig, topk, 1)
    return (reshaped * w).sum(dim=1)


class MoEDispatchCombine(nn.Module):
    """Complete MoE dispatch -> expert compute -> combine pipeline.

    Manages one RepFlowLayer's Phase 1 MoE MLPs:
    - node_self_experts (M1): nd -> nd
    - node_sym_experts (M2): n_sym_dim -> nd
    - edge_experts (merged M3+M4): edge_info_dim -> nd+ne
    - angle_experts (merged M5+M7): angle_dim -> ne+na

    Responsibilities:
    - topk expansion, sorting by target GPU
    - Calling MoEPacker for format packing/unpacking
    - All-to-All dispatch and combine via all_to_all_differentiable
    - Expert computation (for-loop over local experts)
    - Weighted sum + shared expert aggregation

    Parameters
    ----------
    n_dim : int
        Node feature dimension (nd = 4a).
    e_dim : int
        Edge feature dimension (ne = 2a).
    a_dim : int
        Angle feature dimension (na = a).
    n_sym_dim : int
        Symmetry input dim for M2 (24a).
    edge_info_dim : int
        Edge info input dim for merged M3+M4 (10a).
    angle_dim : int
        Angle info input dim for merged M5+M7 (4a).
    n_routing_experts : int
        Total number of routing experts across all GPUs.
    topk : int
        Number of experts each token is routed to.
    n_shared_experts : int
        Number of shared experts (replicated on every GPU).
    ep_group : ProcessGroup or None
        Expert parallelism communication group.
    ep_rank : int
        This GPU's rank within the EP group.
    ep_size : int
        Number of GPUs in the EP group.
    experts_per_gpu : int
        Number of routing experts per GPU.
    activation_function : str
    precision : str
    seed : int, list[int], or None
    """

    def __init__(
        self,
        n_dim: int,
        e_dim: int,
        a_dim: int,
        n_sym_dim: int,
        edge_info_dim: int,
        angle_dim: int,
        n_routing_experts: int,
        topk: int,
        n_shared_experts: int = 0,
        ep_group: Optional[dist.ProcessGroup] = None,
        ep_rank: int = 0,
        ep_size: int = 1,
        experts_per_gpu: int = 1,
        activation_function: str = "silu",
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        validate_dim_ratio(n_dim, e_dim, a_dim)

        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.n_sym_dim = n_sym_dim
        self.edge_info_dim = edge_info_dim
        self.angle_dim = angle_dim
        self.n_routing_experts = n_routing_experts
        self.topk = topk
        self.n_shared_experts = n_shared_experts
        self.ep_group = ep_group
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.experts_per_gpu = experts_per_gpu

        # Packer (pure formatting tool).
        self.packer = MoEPacker(a_dim)

        # Output dimensions.
        self.node_out_dim = n_dim           # M1 output = nd
        self.node_sym_out_dim = n_dim       # M2 output = nd
        self.edge_out_dim = n_dim + e_dim   # merged M3+M4 output = nd+ne
        self.angle_out_dim = e_dim + a_dim  # merged M5+M7 output = ne+na

        # 4 MoE Expert Collections.
        self.node_self_experts = MoEExpertCollection(
            n_dim, self.node_out_dim,
            experts_per_gpu, n_shared_experts,
            activation_function, precision,
            seed=child_seed(seed, 0),
        )
        self.node_sym_experts = MoEExpertCollection(
            n_sym_dim, self.node_sym_out_dim,
            experts_per_gpu, n_shared_experts,
            activation_function, precision,
            seed=child_seed(seed, 1),
        )
        self.edge_experts = MoEExpertCollection(
            edge_info_dim, self.edge_out_dim,
            experts_per_gpu, n_shared_experts,
            activation_function, precision,
            seed=child_seed(seed, 2),
        )
        self.angle_experts = MoEExpertCollection(
            angle_dim, self.angle_out_dim,
            experts_per_gpu, n_shared_experts,
            activation_function, precision,
            seed=child_seed(seed, 3),
        )

    def forward(
        self,
        node_m1_input: torch.Tensor,
        node_m2_input: torch.Tensor,
        edge_input: torch.Tensor,
        angle_input: torch.Tensor,
        node_router_out: tuple[torch.Tensor, torch.Tensor],
        edge_router_out: tuple[torch.Tensor, torch.Tensor],
        angle_router_out: tuple[torch.Tensor, torch.Tensor],
        n2e_index: torch.Tensor,
        n2a_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run MoE dispatch-compute-combine.

        Parameters
        ----------
        node_m1_input : Tensor ``[N_node, nd]``
            M1 input (node_ebd).
        node_m2_input : Tensor ``[N_node, n_sym_dim]``
            M2 input (cat(grrg, drrd)).
        edge_input : Tensor ``[N_edge, edge_info_dim]``
            Merged edge input (edge_info).
        angle_input : Tensor ``[N_angle, angle_dim]``
            Merged angle input (angle_info).
        node_router_out : tuple(Tensor, Tensor)
            (topk_weights ``[N_node, topk]``, topk_indices ``[N_node, topk]``)
        edge_router_out : tuple(Tensor, Tensor)
            (topk_weights ``[N_node, topk]``, topk_indices ``[N_node, topk]``)
            Node-level routing; edge routing derived via n2e_index.
        angle_router_out : tuple(Tensor, Tensor)
            (topk_weights ``[N_node, topk]``, topk_indices ``[N_node, topk]``)
            Node-level routing; angle routing derived via n2a_index.
        n2e_index : Tensor ``[N_edge]``
            Maps each edge to its center node index.
        n2a_index : Tensor ``[N_angle]``
            Maps each angle to its center node index.

        Returns
        -------
        node_m1_out : Tensor ``[N_node, nd]``
        node_m2_out : Tensor ``[N_node, nd]``
        edge_out : Tensor ``[N_edge, nd+ne]``
        angle_out : Tensor ``[N_angle, ne+na]``
        """
        if self.ep_group is None:
            return self._forward_single_gpu(
                node_m1_input, node_m2_input, edge_input, angle_input,
                node_router_out, edge_router_out, angle_router_out,
                n2e_index, n2a_index,
            )
        else:
            return self._forward_multi_gpu(
                node_m1_input, node_m2_input, edge_input, angle_input,
                node_router_out, edge_router_out, angle_router_out,
                n2e_index, n2a_index,
            )

    # ------------------------------------------------------------------
    # Single-GPU path (no A2A, simple for-loop)
    # ------------------------------------------------------------------

    def _forward_single_gpu(
        self,
        node_m1_input: torch.Tensor,
        node_m2_input: torch.Tensor,
        edge_input: torch.Tensor,
        angle_input: torch.Tensor,
        node_router_out: tuple[torch.Tensor, torch.Tensor],
        edge_router_out: tuple[torch.Tensor, torch.Tensor],
        angle_router_out: tuple[torch.Tensor, torch.Tensor],
        n2e_index: torch.Tensor,
        n2a_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-GPU path: sort by expert → split → forward → cat → unsort.

        Steps:
        1. Flatten [N, topk] → [N*topk], expand features via repeat_interleave
        2. Sort by expert_id (one argsort), compute split sizes via bincount
        3. torch.split into contiguous chunks per expert
        4. Forward each expert on its contiguous chunk
        5. cat back, unsort to original [N*topk] order
        6. Reshape [N, topk, dim], weighted sum → [N, dim]
        """
        N_node = node_m1_input.shape[0]
        N_edge = edge_input.shape[0]
        N_angle = angle_input.shape[0]
        topk = self.topk

        node_weights, node_indices = node_router_out   # [N_node, topk]
        edge_weights_node, edge_indices_node = edge_router_out
        angle_weights_node, angle_indices_node = angle_router_out

        # Broadcast node-level routing to edge/angle tokens.
        edge_weights = edge_weights_node[n2e_index]   # [N_edge, topk]
        edge_indices = edge_indices_node[n2e_index]
        angle_weights = angle_weights_node[n2a_index]  # [N_angle, topk]
        angle_indices = angle_indices_node[n2a_index]

        # ── Node M1 + M2 ──
        node_m1_out, node_m2_out = self._sort_split_forward_node(
            node_m1_input, node_m2_input, node_indices, node_weights,
            N_node, topk,
        )

        # ── Edge ──
        edge_out = self._sort_split_forward_feature(
            edge_input, edge_indices, edge_weights,
            N_edge, topk, self.edge_experts,
        )

        # ── Angle ──
        angle_out = self._sort_split_forward_feature(
            angle_input, angle_indices, angle_weights,
            N_angle, topk, self.angle_experts,
        )

        # Add shared expert contribution.
        node_m1_out = node_m1_out + self.node_self_experts.forward_shared(node_m1_input)
        node_m2_out = node_m2_out + self.node_sym_experts.forward_shared(node_m2_input)
        edge_out = edge_out + self.edge_experts.forward_shared(edge_input)
        angle_out = angle_out + self.angle_experts.forward_shared(angle_input)

        return node_m1_out, node_m2_out, edge_out, angle_out

    def _sort_split_forward_node(
        self,
        m1_input: torch.Tensor,      # [N, nd]
        m2_input: torch.Tensor,      # [N, n_sym_dim]
        indices: torch.Tensor,       # [N, topk]
        weights: torch.Tensor,       # [N, topk]
        N: int,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sort-split-forward for node M1 and M2."""
        device = m1_input.device
        dtype = m1_input.dtype

        if N == 0:
            return (
                m1_input.new_zeros(0, self.node_out_dim),
                m2_input.new_zeros(0, self.node_sym_out_dim),
            )

        # Flatten and expand.
        flat_eids = indices.reshape(-1)        # [N*topk]
        flat_weights = weights.reshape(-1)     # [N*topk]
        m1_exp = m1_input.repeat_interleave(topk, dim=0)  # [N*topk, nd]
        m2_exp = m2_input.repeat_interleave(topk, dim=0)  # [N*topk, n_sym_dim]

        # Sort by expert_id.
        sort_idx = torch.argsort(flat_eids, stable=True)
        m1_sorted = m1_exp[sort_idx]
        m2_sorted = m2_exp[sort_idx]
        eids_sorted = flat_eids[sort_idx]

        # Split sizes.
        counts = torch.bincount(eids_sorted, minlength=self.n_routing_experts)
        split_sizes = counts.tolist()

        # Forward each expert on its contiguous chunk.
        m1_chunks = torch.split(m1_sorted, split_sizes)
        m2_chunks = torch.split(m2_sorted, split_sizes)
        m1_out_parts: list[torch.Tensor] = []
        m2_out_parts: list[torch.Tensor] = []
        for eid, (m1c, m2c) in enumerate(zip(m1_chunks, m2_chunks)):
            if m1c.shape[0] > 0:
                m1_out_parts.append(self.node_self_experts.forward_expert(m1c, eid))
                m2_out_parts.append(self.node_sym_experts.forward_expert(m2c, eid))
            else:
                m1_out_parts.append(m1c.new_zeros(0, self.node_out_dim))
                m2_out_parts.append(m2c.new_zeros(0, self.node_sym_out_dim))

        m1_sorted_out = torch.cat(m1_out_parts, dim=0)  # [N*topk, nd]
        m2_sorted_out = torch.cat(m2_out_parts, dim=0)

        # Unsort back to original [N*topk] order.
        unsort_idx = torch.empty_like(sort_idx)
        unsort_idx[sort_idx] = torch.arange(len(sort_idx), device=device)
        m1_orig = m1_sorted_out[unsort_idx]
        m2_orig = m2_sorted_out[unsort_idx]

        # Weighted sum: [N*topk, dim] → [N, topk, dim] * w → sum → [N, dim]
        m1_out = _weighted_sum_topk(m1_orig, flat_weights, N, topk)
        m2_out = _weighted_sum_topk(m2_orig, flat_weights, N, topk)
        return m1_out, m2_out

    def _sort_split_forward_feature(
        self,
        features: torch.Tensor,         # [N, feat_dim]
        indices: torch.Tensor,          # [N, topk]
        weights: torch.Tensor,          # [N, topk]
        N: int,
        topk: int,
        expert_collection: MoEExpertCollection,
    ) -> torch.Tensor:
        """Sort-split-forward for edge or angle features."""
        device = features.device
        out_dim = expert_collection.routing_experts[0].num_out

        if N == 0:
            return features.new_zeros(0, out_dim)

        flat_eids = indices.reshape(-1)
        flat_weights = weights.reshape(-1)
        feat_exp = features.repeat_interleave(topk, dim=0)  # [N*topk, feat_dim]

        sort_idx = torch.argsort(flat_eids, stable=True)
        feat_sorted = feat_exp[sort_idx]
        eids_sorted = flat_eids[sort_idx]

        counts = torch.bincount(eids_sorted, minlength=self.n_routing_experts)
        split_sizes = counts.tolist()
        chunks = torch.split(feat_sorted, split_sizes)

        out_parts: list[torch.Tensor] = []
        for eid, chunk in enumerate(chunks):
            if chunk.shape[0] > 0:
                out_parts.append(expert_collection.forward_expert(chunk, eid))
            else:
                out_parts.append(chunk.new_zeros(0, out_dim))

        sorted_out = torch.cat(out_parts, dim=0)

        unsort_idx = torch.empty_like(sort_idx)
        unsort_idx[sort_idx] = torch.arange(len(sort_idx), device=device)
        orig_out = sorted_out[unsort_idx]

        return _weighted_sum_topk(orig_out, flat_weights, N, topk)

    # ------------------------------------------------------------------
    # Multi-GPU path (topk expand -> sort -> Pack -> A2A -> Expert -> A2A -> Unpack -> weighted sum)
    # ------------------------------------------------------------------

    def _forward_multi_gpu(
        self,
        node_m1_input: torch.Tensor,
        node_m2_input: torch.Tensor,
        edge_input: torch.Tensor,
        angle_input: torch.Tensor,
        node_router_out: tuple[torch.Tensor, torch.Tensor],
        edge_router_out: tuple[torch.Tensor, torch.Tensor],
        angle_router_out: tuple[torch.Tensor, torch.Tensor],
        n2e_index: torch.Tensor,
        n2a_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Multi-GPU path with All-to-All dispatch and combine."""
        N_node = node_m1_input.shape[0]
        N_edge = edge_input.shape[0]
        N_angle = angle_input.shape[0]
        device = node_m1_input.device
        dtype = node_m1_input.dtype
        topk = self.topk

        node_weights, node_indices = node_router_out
        edge_weights_node, edge_indices_node = edge_router_out
        angle_weights_node, angle_indices_node = angle_router_out

        # Broadcast node-level routing to edge/angle.
        edge_weights = edge_weights_node[n2e_index]   # [N_edge, topk]
        edge_indices = edge_indices_node[n2e_index]   # [N_edge, topk]
        angle_weights = angle_weights_node[n2a_index]  # [N_angle, topk]
        angle_indices = angle_indices_node[n2a_index]  # [N_angle, topk]

        # ── Step 3a: topk expand + sort ──
        # Concatenate node M1 and M2 inputs for packing: [N_node, nd + n_sym_dim] = [N_node, 28a]
        node_combined = torch.cat([node_m1_input, node_m2_input], dim=-1)  # [N_node, 28a]

        (node_sorted, node_expert_ids_sorted, node_weights_sorted,
         node_unsort_idx, node_counts, _) = _topk_expand_sort(
            node_combined, node_indices, node_weights, self.experts_per_gpu,
        )
        (edge_sorted, edge_expert_ids_sorted, edge_weights_sorted,
         edge_unsort_idx, edge_counts, _) = _topk_expand_sort(
            edge_input, edge_indices, edge_weights, self.experts_per_gpu,
        )
        (angle_sorted, angle_expert_ids_sorted, angle_weights_sorted,
         angle_unsort_idx, angle_counts, _) = _topk_expand_sort(
            angle_input, angle_indices, angle_weights, self.experts_per_gpu,
        )

        # Pad counts to ep_size if needed (some GPUs may get 0 tokens).
        while len(node_counts) < self.ep_size:
            node_counts.append(0)
        while len(edge_counts) < self.ep_size:
            edge_counts.append(0)
        while len(angle_counts) < self.ep_size:
            angle_counts.append(0)

        # ── Step 3b: Pack for dispatch ──
        packed, send_splits = self.packer.pack_for_dispatch(
            node_sorted, edge_sorted, angle_sorted,
            node_counts, edge_counts, angle_counts,
        )

        # ── Step 3c: Exchange metadata ──
        # Build send_info: [ep_size, 3] with (node_count, edge_count, angle_count).
        send_info = torch.tensor(
            [[node_counts[g], edge_counts[g], angle_counts[g]]
             for g in range(self.ep_size)],
            dtype=torch.int64, device=device,
        )
        recv_info = exchange_metadata(send_info, self.ep_group)
        recv_node_counts = recv_info[:, 0].tolist()
        recv_edge_counts = recv_info[:, 1].tolist()
        recv_angle_counts = recv_info[:, 2].tolist()

        recv_splits = counts_to_packed_rows(
            recv_node_counts, recv_edge_counts, recv_angle_counts,
            edge_group_size=self.packer.edge_concat_in,
            angle_group_size=self.packer.angle_concat_in,
        )

        # ── Step 3d: Dispatch A2A ──
        recv_tensor = all_to_all_differentiable(
            packed, send_splits, recv_splits, self.ep_group,
        )

        # ── Step 3e: Unpack + Expert Compute ──
        node_recv, edge_recv, angle_recv = self.packer.unpack_from_dispatch(
            recv_tensor, recv_node_counts, recv_edge_counts, recv_angle_counts,
        )

        # Exchange expert IDs via separate A2A (non-differentiable, int).
        node_eid_recv = self._exchange_expert_ids(
            node_expert_ids_sorted, node_counts,
            recv_node_counts, device,
        )
        edge_eid_recv = self._exchange_expert_ids(
            edge_expert_ids_sorted, edge_counts,
            recv_edge_counts, device,
        )
        angle_eid_recv = self._exchange_expert_ids(
            angle_expert_ids_sorted, angle_counts,
            recv_angle_counts, device,
        )

        # Split node_recv into M1 and M2 inputs.
        node_m1_recv = node_recv[:, :self.n_dim]       # [N_node_recv, nd]
        node_m2_recv = node_recv[:, self.n_dim:]        # [N_node_recv, n_sym_dim]

        # Compute experts: for each local expert, process its tokens.
        node_m1_output, node_m2_output = self._compute_node_experts(
            node_m1_recv, node_m2_recv, node_eid_recv, recv_node_counts,
        )
        edge_output = self._compute_feature_experts(
            edge_recv, edge_eid_recv, self.edge_experts, recv_edge_counts,
        )
        angle_output = self._compute_feature_experts(
            angle_recv, angle_eid_recv, self.angle_experts, recv_angle_counts,
        )

        # ── Step 3f: Repack output ──
        # Combine node M1 and M2 outputs: [N_node_recv, nd + nd] = [N_node_recv, 8a]
        node_output_combined = torch.cat([node_m1_output, node_m2_output], dim=-1)

        packed_out = self.packer.pack_for_combine(
            node_output_combined, edge_output, angle_output,
            recv_node_counts, recv_edge_counts, recv_angle_counts,
        )

        # ── Step 3g: Combine A2A (reverse direction) ──
        returned = all_to_all_differentiable(
            packed_out, recv_splits, send_splits, self.ep_group,
        )

        # ── Step 3h: Unpack + Unsort + Weighted Sum ──
        node_ret, edge_ret, angle_ret = self.packer.unpack_from_combine(
            returned, node_counts, edge_counts, angle_counts,
        )

        # Split node output back into M1 and M2.
        node_m1_ret = node_ret[:, :self.n_dim]   # [N_node_exp, nd]
        node_m2_ret = node_ret[:, self.n_dim:]   # [N_node_exp, nd]

        # Unsort to restore original token order.
        node_m1_ret = node_m1_ret[node_unsort_idx]
        node_m2_ret = node_m2_ret[node_unsort_idx]
        node_weights_orig = node_weights_sorted[node_unsort_idx]

        edge_ret = edge_ret[edge_unsort_idx]
        edge_weights_orig = edge_weights_sorted[edge_unsort_idx]

        angle_ret = angle_ret[angle_unsort_idx]
        angle_weights_orig = angle_weights_sorted[angle_unsort_idx]

        # Weighted sum: [N_orig*topk, dim] -> [N_orig, topk, dim] -> sum.
        node_m1_out = _weighted_sum_topk(node_m1_ret, node_weights_orig, N_node, topk)
        node_m2_out = _weighted_sum_topk(node_m2_ret, node_weights_orig, N_node, topk)
        edge_out = _weighted_sum_topk(edge_ret, edge_weights_orig, N_edge, topk)
        angle_out = _weighted_sum_topk(angle_ret, angle_weights_orig, N_angle, topk)

        # Add shared expert contribution (on original inputs, no A2A).
        node_m1_out = node_m1_out + self.node_self_experts.forward_shared(node_m1_input)
        node_m2_out = node_m2_out + self.node_sym_experts.forward_shared(node_m2_input)
        edge_out = edge_out + self.edge_experts.forward_shared(edge_input)
        angle_out = angle_out + self.angle_experts.forward_shared(angle_input)

        return node_m1_out, node_m2_out, edge_out, angle_out

    # ------------------------------------------------------------------
    # Helper: exchange expert IDs via A2A (non-differentiable)
    # ------------------------------------------------------------------

    def _exchange_expert_ids(
        self,
        expert_ids_sorted: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Exchange expert IDs via All-to-All (non-differentiable).

        Parameters
        ----------
        expert_ids_sorted : Tensor ``[N_expanded]``
            Global expert IDs sorted by target GPU.
        send_counts : list[int]
            Tokens sent to each GPU.
        recv_counts : list[int]
            Tokens received from each GPU.
        device : torch.device

        Returns
        -------
        Tensor ``[sum(recv_counts)]``
            Global expert IDs received, which can be converted to local
            expert IDs via ``% self.experts_per_gpu``.
        """
        total_send = sum(send_counts)
        total_recv = sum(recv_counts)

        if total_send == 0 and total_recv == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        # Use int64 tensor for exchange.
        send_tensor = expert_ids_sorted.long().contiguous()

        if self.ep_group is None:
            return send_tensor

        recv_tensor = torch.empty(total_recv, dtype=torch.long, device=device)
        dist.all_to_all_single(
            recv_tensor, send_tensor,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=self.ep_group,
        )
        return recv_tensor

    # ------------------------------------------------------------------
    # Expert computation helpers (multi-GPU recv side, no sort needed)
    # ------------------------------------------------------------------

    def _build_expert_gather_idx(
        self,
        local_eids: torch.Tensor,
        recv_counts: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        """Build gather index to reorder recv buffer into expert-contiguous layout.

        The recv buffer is the concatenation of segments from each sender.
        Because ``_topk_expand_sort`` sorts by global expert ID, each
        sender's segment is already sorted by local_eid.  This function
        exploits that structure to build a permutation in O(N) without
        argsort.

        Parameters
        ----------
        local_eids : Tensor ``[N]``
            Local expert IDs (``global_eid % experts_per_gpu``).
        recv_counts : list[int]
            Number of tokens received from each sender GPU.
        device : torch.device

        Returns
        -------
        gather_idx : Tensor ``[N]``
            Permutation: ``features[gather_idx]`` is expert-contiguous.
        split_sizes : list[int]
            Number of tokens for each local expert (length = experts_per_gpu).
        ungather_idx : Tensor ``[N]``
            Inverse permutation to restore recv order after expert compute.
        """
        N = local_eids.shape[0]
        ep_size = len(recv_counts)
        epg = self.experts_per_gpu

        if N == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, [0] * epg, empty

        # Compute per-sender per-expert counts and offsets.
        # sender_offsets[s][eid] = start index within recv buffer for
        #   sender s, expert eid.
        # sender_counts[s][eid] = how many tokens from sender s for expert eid.
        sender_counts: list[list[int]] = []
        sender_offsets: list[list[int]] = []
        offset = 0
        for s in range(ep_size):
            seg_count = recv_counts[s]
            if seg_count == 0:
                sender_counts.append([0] * epg)
                sender_offsets.append([offset] * epg)
            else:
                seg_eids = local_eids[offset : offset + seg_count]
                cnts = torch.bincount(seg_eids, minlength=epg).tolist()
                offs: list[int] = []
                seg_off = offset
                for eid in range(epg):
                    offs.append(seg_off)
                    seg_off += cnts[eid]
                sender_counts.append(cnts)
                sender_offsets.append(offs)
            offset += seg_count

        # Build gather index: expert-major ordering.
        # For each expert, collect token indices from all senders.
        gather_parts: list[torch.Tensor] = []
        split_sizes: list[int] = []
        for eid in range(epg):
            expert_total = 0
            for s in range(ep_size):
                cnt = sender_counts[s][eid]
                if cnt > 0:
                    start = sender_offsets[s][eid]
                    gather_parts.append(
                        torch.arange(start, start + cnt, device=device)
                    )
                    expert_total += cnt
            split_sizes.append(expert_total)

        gather_idx = (
            torch.cat(gather_parts)
            if gather_parts
            else torch.empty(0, dtype=torch.long, device=device)
        )

        # Inverse permutation: ungather_idx[gather_idx[i]] = i.
        ungather_idx = torch.empty(N, dtype=torch.long, device=device)
        ungather_idx[gather_idx] = torch.arange(N, device=device)

        return gather_idx, split_sizes, ungather_idx

    def _compute_node_experts(
        self,
        node_m1_input: torch.Tensor,
        node_m2_input: torch.Tensor,
        expert_ids: torch.Tensor,
        recv_counts: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute M1 and M2 experts for received node tokens.

        Uses structured O(N) gather (no sort) because each sender's
        segment is already sorted by local expert ID.

        Parameters
        ----------
        node_m1_input : Tensor ``[N, nd]``
        node_m2_input : Tensor ``[N, n_sym_dim]``
        expert_ids : Tensor ``[N]``
            Global expert IDs.
        recv_counts : list[int]
            Per-sender token counts (for structured gather).

        Returns
        -------
        m1_output : Tensor ``[N, nd]``
        m2_output : Tensor ``[N, nd]``
        """
        N = node_m1_input.shape[0]
        device = node_m1_input.device

        if N == 0:
            return (
                node_m1_input.new_zeros(0, self.node_out_dim),
                node_m2_input.new_zeros(0, self.node_sym_out_dim),
            )

        local_eids = expert_ids % self.experts_per_gpu
        gather_idx, split_sizes, ungather_idx = self._build_expert_gather_idx(
            local_eids, recv_counts, device,
        )

        # Gather into expert-contiguous layout.
        m1_gathered = node_m1_input[gather_idx]
        m2_gathered = node_m2_input[gather_idx]

        # Split + forward each expert.
        m1_chunks = torch.split(m1_gathered, split_sizes)
        m2_chunks = torch.split(m2_gathered, split_sizes)
        m1_out_parts: list[torch.Tensor] = []
        m2_out_parts: list[torch.Tensor] = []
        for eid, (m1c, m2c) in enumerate(zip(m1_chunks, m2_chunks)):
            if m1c.shape[0] > 0:
                m1_out_parts.append(self.node_self_experts.forward_expert(m1c, eid))
                m2_out_parts.append(self.node_sym_experts.forward_expert(m2c, eid))
            else:
                m1_out_parts.append(m1c.new_zeros(0, self.node_out_dim))
                m2_out_parts.append(m2c.new_zeros(0, self.node_sym_out_dim))

        m1_cat = torch.cat(m1_out_parts, dim=0)
        m2_cat = torch.cat(m2_out_parts, dim=0)

        # Ungather back to recv order.
        return m1_cat[ungather_idx], m2_cat[ungather_idx]

    def _compute_feature_experts(
        self,
        features: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_collection: MoEExpertCollection,
        recv_counts: list[int],
    ) -> torch.Tensor:
        """Compute experts for received edge or angle tokens.

        Uses structured O(N) gather (no sort) because each sender's
        segment is already sorted by local expert ID.

        Parameters
        ----------
        features : Tensor ``[N, feat_dim]``
        expert_ids : Tensor ``[N]``
            Global expert IDs.
        expert_collection : MoEExpertCollection
        recv_counts : list[int]
            Per-sender token counts (for structured gather).

        Returns
        -------
        Tensor ``[N, out_dim]``
        """
        N = features.shape[0]
        device = features.device
        out_dim = expert_collection.routing_experts[0].num_out

        if N == 0:
            return features.new_zeros(0, out_dim)

        local_eids = expert_ids % self.experts_per_gpu
        gather_idx, split_sizes, ungather_idx = self._build_expert_gather_idx(
            local_eids, recv_counts, device,
        )

        # Gather into expert-contiguous layout.
        feat_gathered = features[gather_idx]

        # Split + forward each expert.
        chunks = torch.split(feat_gathered, split_sizes)
        out_parts: list[torch.Tensor] = []
        for eid, chunk in enumerate(chunks):
            if chunk.shape[0] > 0:
                out_parts.append(expert_collection.forward_expert(chunk, eid))
            else:
                out_parts.append(chunk.new_zeros(0, out_dim))

        cat_out = torch.cat(out_parts, dim=0)

        # Ungather back to recv order.
        return cat_out[ungather_idx]
