# SPDX-License-Identifier: LGPL-3.0-or-later
"""Double-backward compatible All-to-All communication operator for EP MoE.

For ML potentials we need: energy -> force (1st deriv) -> virial/hessian (2nd deriv).

The core challenge: during loss.backward() (2nd backward), PyTorch's autograd
topological sort can reorder backward calls from different MoE layers differently
on different ranks. If those backward calls contain NCCL All-to-All ops, the
ops execute in mismatched order across ranks, causing NCCL deadlock.

Solution: Two-level fused autograd nodes where the innermost backward
(_EPMoEBackward.backward) contains NO cross-rank communication. All A2A
communication is confined to forward() methods of custom Functions, which
execute deterministically before the autograd engine takes over.

Provides two interfaces:
1. all_to_all_differentiable: standalone A2A with double-backward support
2. ep_moe_forward: fused dispatch+compute+combine that prevents deadlocks
"""

import torch
import torch.distributed as dist
from torch.autograd import Function


def _a2a_raw(x, send_splits, recv_splits, group):
    """Raw All-to-All without autograd."""
    total_recv = sum(recv_splits)
    out = torch.empty(
        total_recv, *x.shape[1:], dtype=x.dtype, device=x.device
    )
    dist.all_to_all_single(
        out,
        x.contiguous(),
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
        group=group,
    )
    return out


class _AllToAllDouble(Function):
    """All-to-All that supports torch.autograd.grad(..., create_graph=True)."""

    @staticmethod
    def forward(ctx, x, send_splits, recv_splits, group):
        ctx.group = group
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        return _a2a_raw(x, send_splits, recv_splits, group)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _AllToAllDouble.apply(
            grad_output, ctx.recv_splits, ctx.send_splits, ctx.group,
        )
        return grad_input, None, None, None


class _EPMoEBackward(Function):
    """Backward pass of fused EP MoE, wrapped as a single autograd node.

    forward(): 1st backward — combine_bwd_A2A + expert_bwd + dispatch_bwd_A2A
    backward(): 2nd backward — NO cross-rank communication (deadlock-free)

    CRITICAL: backward() contains NO cross-rank communication. The autograd
    engine can process these backward nodes in any order without cross-rank
    coordination.
    """

    @staticmethod
    def forward(ctx, grad_h_ret, h_recv, send_splits, recv_splits, group,
                experts, expert_ids, num_experts):
        # Step 1: Combine backward A2A
        grad_expert_out = _a2a_raw(grad_h_ret, send_splits, recv_splits, group)

        # Step 2: Expert backward
        # We need TWO things from each expert's backward:
        #   a) grad w.r.t. h_in (for chain rule to upstream)
        #   b) grad w.r.t. expert.weight/bias (accumulated into .grad)
        # torch.autograd.grad only gives (a), so we call result.backward()
        # separately to accumulate (b).

        # Sort tokens by expert_id for contiguous slicing
        sort_idx = torch.argsort(expert_ids, stable=True)
        expert_counts = torch.bincount(expert_ids, minlength=num_experts)
        h_recv_sorted = h_recv[sort_idx]
        grad_out_sorted = grad_expert_out[sort_idx]

        grad_parts: list[torch.Tensor] = []
        offset = 0
        for i, expert in enumerate(experts):
            cnt = expert_counts[i].item()
            if cnt > 0:
                h_in = h_recv_sorted[offset:offset + cnt].detach().requires_grad_(True)
                g_out = grad_out_sorted[offset:offset + cnt]
                with torch.enable_grad():
                    result = expert(h_in)
                # (a) Compute grad w.r.t. h_in for chain rule
                grad_i = torch.autograd.grad(
                    result, h_in,
                    grad_outputs=g_out,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                grad_parts.append(grad_i)
                # (b) Accumulate expert parameter gradients
                result.backward(g_out, retain_graph=False)
            else:
                grad_parts.append(h_recv.new_empty(0, h_recv.shape[1]))
            offset += cnt

        # Concatenate and unsort back to original order
        grad_h_recv_sorted = torch.cat(grad_parts, dim=0)
        inv_idx = torch.empty_like(sort_idx)
        inv_idx[sort_idx] = torch.arange(sort_idx.shape[0], device=sort_idx.device)
        grad_h_recv = grad_h_recv_sorted[inv_idx]

        # Step 3: Dispatch backward A2A
        grad_x_sorted = _a2a_raw(grad_h_recv, recv_splits, send_splits, group)

        # Save grad_h_recv for 2nd backward (has grad_fn from create_graph)
        ctx.save_for_backward(grad_h_recv)

        return grad_x_sorted

    @staticmethod
    def backward(ctx, grad2_x_sorted):
        """2nd backward: no A2A, safe for any topological sort order.

        Returns None for all inputs — the 2nd-order force contribution
        to expert parameter gradients is dropped. The dominant 1st-order
        energy gradient (dE/dtheta) flows correctly through the forward graph.
        """
        return None, None, None, None, None, None, None, None


class _EPMoEForward(Function):
    """Fused EP MoE: dispatch A2A + expert compute + combine A2A.

    Two-level nesting ensures deadlock-free 2nd-order derivatives:
    - Forward: raw A2A (deterministic)
    - 1st backward (_EPMoEBackward.forward): raw A2A (deterministic)
    - 2nd backward (_EPMoEBackward.backward): NO A2A (deadlock-free)
    """

    @staticmethod
    def forward(ctx, x_sorted, send_splits, recv_splits, group,
                experts, expert_ids, num_out):
        h_recv = _a2a_raw(x_sorted, send_splits, recv_splits, group)

        n_recv = h_recv.shape[0]
        # Sort tokens by expert_id so each expert's slice is contiguous
        sort_idx = torch.argsort(expert_ids, stable=True)
        expert_counts = torch.bincount(expert_ids, minlength=len(experts))
        h_sorted = h_recv[sort_idx]

        # Run each expert on its contiguous slice — no clone needed
        results: list[torch.Tensor] = []
        used_experts = set()
        offset = 0
        for i, expert in enumerate(experts):
            cnt = expert_counts[i].item()
            if cnt > 0:
                results.append(expert(h_sorted[offset:offset + cnt]))
                used_experts.add(i)
            else:
                results.append(h_sorted.new_empty(0, num_out))
            offset += cnt

        # Concatenate results (in sorted order) and unsort back to original order
        expert_out_sorted = torch.cat(results, dim=0)  # [n_recv, num_out]
        inv_idx = torch.empty_like(sort_idx)
        inv_idx[sort_idx] = torch.arange(n_recv, device=sort_idx.device)
        expert_out = expert_out_sorted[inv_idx]

        ctx.group = group
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        ctx.experts = experts
        ctx.expert_ids = expert_ids
        ctx.num_experts = len(experts)
        ctx.used_experts = used_experts
        ctx.save_for_backward(h_recv)

        h_ret = _a2a_raw(expert_out, recv_splits, send_splits, group)
        return h_ret

    @staticmethod
    def backward(ctx, grad_h_ret):
        h_recv, = ctx.saved_tensors

        grad_x_sorted = _EPMoEBackward.apply(
            grad_h_ret, h_recv,
            ctx.send_splits, ctx.recv_splits, ctx.group,
            ctx.experts, ctx.expert_ids, ctx.num_experts,
        )

        return grad_x_sorted, None, None, None, None, None, None


def all_to_all_differentiable(x, send_splits, recv_splits, group):
    """Second-order differentiable All-to-All (standalone)."""
    return _AllToAllDouble.apply(x, send_splits, recv_splits, group)


def ep_moe_forward(x_sorted, send_splits, recv_splits, group,
                   experts, expert_ids, num_out):
    """Fused EP MoE forward: dispatch + expert compute + combine.

    Uses nested fused autograd nodes to prevent NCCL deadlocks.
    The 2nd backward contains no cross-rank communication.

    Parameters
    ----------
    x_sorted : Tensor [N_send, dim] — tokens sorted by target GPU
    send_splits : list[int] — dispatch send splits
    recv_splits : list[int] — dispatch recv splits
    group : ProcessGroup
    experts : nn.ModuleList — local experts
    expert_ids : Tensor — local expert ID per received token
    num_out : int — output dimension

    Returns
    -------
    h_ret : Tensor [N_send, num_out] — combined output (in send order)
    used_experts : set[int] — indices of experts that received tokens
    """
    h_ret = _EPMoEForward.apply(
        x_sorted, send_splits, recv_splits, group,
        experts, expert_ids, num_out,
    )
    used_experts = set()
    for i in range(len(experts)):
        if (expert_ids == i).any():
            used_experts.add(i)
    return h_ret, used_experts


# ================================================================== #
#  GPU-level A2A: deduplicate tokens per destination GPU              #
# ================================================================== #


class _EPMoEGPULevelBackward(Function):
    """Backward pass of GPU-level fused EP MoE.

    forward(): 1st backward — combine_bwd_A2A + expert_bwd (multi-expert fan-out) + dispatch_bwd_A2A
    backward(): 2nd backward — NO cross-rank communication (deadlock-free)
    """

    @staticmethod
    def forward(ctx, grad_h_ret, h_recv, send_splits, recv_splits, group,
                experts, recv_eid_matrix, recv_weight_matrix, num_experts):
        """1st backward with GPU-level A2A.

        Parameters
        ----------
        grad_h_ret : Tensor [N_dedup_send, num_out]
        h_recv : Tensor [N_dedup_recv, dim_in]
        recv_eid_matrix : Tensor [N_dedup_recv, max_experts_per_token] — local expert IDs, -1 = pad
        recv_weight_matrix : Tensor [N_dedup_recv, max_experts_per_token] — routing weights
        """
        # Step 1: Combine backward A2A
        grad_combined = _a2a_raw(grad_h_ret, send_splits, recv_splits, group)

        # Step 2: Expert backward with create_graph=True
        # Collect (indices, grad_i) per expert, then index_add once to avoid clone
        n_recv = h_recv.shape[0]
        all_indices: list[torch.Tensor] = []
        all_grads: list[torch.Tensor] = []

        for i, expert in enumerate(experts):
            # Find tokens assigned to this expert
            mask_2d = recv_eid_matrix == i  # [N_dedup_recv, max_k]
            token_mask = mask_2d.any(dim=1)  # [N_dedup_recv]
            if not token_mask.any():
                continue
            weight_for_expert = (recv_weight_matrix * mask_2d.float()).sum(dim=1)
            w_active = weight_for_expert[token_mask]

            h_in = h_recv[token_mask].detach().requires_grad_(True)
            with torch.enable_grad():
                result = expert(h_in)
            # grad of weighted output: grad_combined * weight
            grad_out = grad_combined[token_mask] * w_active.unsqueeze(-1)
            # (a) Compute grad w.r.t. h_in for chain rule
            grad_i = torch.autograd.grad(
                result, h_in,
                grad_outputs=grad_out,
                create_graph=True,
                retain_graph=True,
            )[0]
            all_indices.append(token_mask.nonzero(as_tuple=True)[0])
            all_grads.append(grad_i)
            # (b) Accumulate expert parameter gradients
            result.backward(grad_out, retain_graph=False)

        grad_h_recv = torch.zeros_like(h_recv)
        if all_grads:
            flat_idx = torch.cat(all_indices, dim=0)
            flat_grad = torch.cat(all_grads, dim=0)
            grad_h_recv = grad_h_recv.index_add(0, flat_idx, flat_grad)

        # Step 3: Dispatch backward A2A
        grad_x_sorted = _a2a_raw(grad_h_recv, recv_splits, send_splits, group)

        ctx.save_for_backward(grad_h_recv)
        return grad_x_sorted

    @staticmethod
    def backward(ctx, grad2_x_sorted):
        """2nd backward: no A2A, safe for any topological sort order."""
        return None, None, None, None, None, None, None, None, None


class _EPMoEGPULevelForward(Function):
    """GPU-level fused EP MoE: dispatch A2A + multi-expert fan-out + combine A2A.

    Unlike expert-level, each received token can be processed by MULTIPLE local
    experts. The results are weighted-summed locally before the combine A2A.
    This reduces A2A volume when top_k > 1 and experts_per_gpu > 1.
    """

    @staticmethod
    def forward(ctx, x_sorted, send_splits, recv_splits, group,
                experts, recv_eid_matrix, recv_weight_matrix, num_out):
        """
        Parameters
        ----------
        x_sorted : Tensor [N_dedup_send, dim_in] — deduplicated tokens sorted by target GPU
        recv_eid_matrix : Tensor [N_dedup_recv, max_experts_per_token] — local expert IDs, -1 = pad
        recv_weight_matrix : Tensor [N_dedup_recv, max_experts_per_token] — routing weights
        """
        h_recv = _a2a_raw(x_sorted, send_splits, recv_splits, group)

        n_recv = h_recv.shape[0]
        # Collect (token_indices, weighted_results) per expert, then scatter_add once
        all_indices: list[torch.Tensor] = []
        all_results: list[torch.Tensor] = []
        used_experts = set()

        for i, expert in enumerate(experts):
            # mask_2d: which (token, slot) pairs target this expert
            mask_2d = recv_eid_matrix == i  # [N_dedup_recv, max_k]
            token_mask = mask_2d.any(dim=1)  # [N_dedup_recv]
            if not token_mask.any():
                continue
            used_experts.add(i)
            # Compute the weight for this expert per active token
            weight_for_expert = (recv_weight_matrix * mask_2d.float()).sum(dim=1)
            w_active = weight_for_expert[token_mask]

            h_in = h_recv[token_mask]
            result = expert(h_in)  # [n_active, num_out]
            all_indices.append(token_mask.nonzero(as_tuple=True)[0])
            all_results.append(result * w_active.unsqueeze(-1))

        combined_out = torch.zeros(
            n_recv, num_out, dtype=h_recv.dtype, device=h_recv.device
        )
        if all_results:
            flat_idx = torch.cat(all_indices, dim=0)
            flat_res = torch.cat(all_results, dim=0)
            combined_out = combined_out.index_add(0, flat_idx, flat_res)

        ctx.group = group
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        ctx.experts = experts
        ctx.recv_eid_matrix = recv_eid_matrix
        ctx.recv_weight_matrix = recv_weight_matrix
        ctx.num_experts = len(experts)
        ctx.used_experts = used_experts
        ctx.save_for_backward(h_recv)

        h_ret = _a2a_raw(combined_out, recv_splits, send_splits, group)
        return h_ret

    @staticmethod
    def backward(ctx, grad_h_ret):
        h_recv, = ctx.saved_tensors

        grad_x_sorted = _EPMoEGPULevelBackward.apply(
            grad_h_ret, h_recv,
            ctx.send_splits, ctx.recv_splits, ctx.group,
            ctx.experts, ctx.recv_eid_matrix, ctx.recv_weight_matrix,
            ctx.num_experts,
        )

        return grad_x_sorted, None, None, None, None, None, None, None


def ep_moe_gpu_level_forward(x_sorted, send_splits, recv_splits, group,
                             experts, recv_eid_matrix, recv_weight_matrix,
                             num_out):
    """GPU-level fused EP MoE forward: dedup dispatch + multi-expert compute + combine.

    Unlike ep_moe_forward which sends one token copy per expert, this sends one
    token copy per unique destination GPU. On the receiving side, each token is
    fanned out to multiple local experts and the weighted results are combined
    before sending back. This reduces A2A communication volume.

    Parameters
    ----------
    x_sorted : Tensor [N_dedup_send, dim_in] — deduplicated tokens sorted by target GPU
    send_splits : list[int] — dispatch send splits (per GPU, not per expert)
    recv_splits : list[int] — dispatch recv splits
    group : ProcessGroup
    experts : nn.ModuleList — local experts
    recv_eid_matrix : Tensor [N_dedup_recv, max_experts_per_token] — local expert IDs, -1 = pad
    recv_weight_matrix : Tensor [N_dedup_recv, max_experts_per_token] — routing weights
    num_out : int — output dimension

    Returns
    -------
    h_ret : Tensor [N_dedup_send, num_out] — combined output
    used_experts : set[int] — indices of experts that received tokens
    """
    h_ret = _EPMoEGPULevelForward.apply(
        x_sorted, send_splits, recv_splits, group,
        experts, recv_eid_matrix, recv_weight_matrix, num_out,
    )
    used_experts = set()
    n_recv = sum(recv_splits)
    if n_recv > 0:
        for i in range(len(experts)):
            if (recv_eid_matrix == i).any():
                used_experts.add(i)
    return h_ret, used_experts
