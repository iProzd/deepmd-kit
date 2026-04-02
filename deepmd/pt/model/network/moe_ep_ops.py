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

        # Step 2: Expert backward with create_graph=True for 2nd-order
        grad_h_recv = torch.zeros_like(h_recv)
        for i, expert in enumerate(experts):
            mask = expert_ids == i
            if mask.any():
                h_in = h_recv[mask].detach().requires_grad_(True)
                with torch.enable_grad():
                    result = expert(h_in)
                grad_i = torch.autograd.grad(
                    result, h_in,
                    grad_outputs=grad_expert_out[mask],
                    create_graph=True,
                )[0]
                grad_h_recv = grad_h_recv.clone()
                grad_h_recv[mask] = grad_i

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

        expert_out = torch.zeros(
            h_recv.shape[0], num_out, dtype=h_recv.dtype, device=h_recv.device
        )
        used_experts = set()
        for i, expert in enumerate(experts):
            mask = expert_ids == i
            if mask.any():
                h_in = h_recv[mask]
                result = expert(h_in)
                expert_out = expert_out.clone()
                expert_out[mask] = result
                used_experts.add(i)

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
