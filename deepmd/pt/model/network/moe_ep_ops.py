# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differentiable All-to-All communication operators for MoE Expert Parallelism.

Provides `_AllToAllDouble`, a recursive autograd Function whose backward
calls `.apply()` again, creating a fresh autograd node so that
`create_graph=True` (required for force -> virial second derivatives)
works correctly to arbitrary order.

Public API
----------
all_to_all_differentiable(x, send_splits, recv_splits, group)
    When *group* is ``None`` (single-GPU / no EP), returns *x* unchanged.
    Otherwise dispatches through ``_AllToAllDouble``.
"""

from __future__ import annotations

from typing import (
    Optional,
)

import torch
import torch.distributed as dist
from torch.autograd import (
    Function,
)


def _a2a_raw(
    x: torch.Tensor,
    send_splits: list[int],
    recv_splits: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Raw All-to-All without autograd.

    Parameters
    ----------
    x : Tensor
        Input tensor whose first dimension equals ``sum(send_splits)``.
    send_splits : list[int]
        Number of rows to send to each rank.
    recv_splits : list[int]
        Number of rows to receive from each rank.
    group : ProcessGroup
        The communication group.

    Returns
    -------
    Tensor
        Output tensor with first dimension ``sum(recv_splits)``.
    """
    total_recv = sum(recv_splits)
    out = torch.empty(
        (total_recv, *x.shape[1:]), dtype=x.dtype, device=x.device
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
    """Recursively differentiable All-to-All.

    The backward pass calls ``.apply()`` with swapped send/recv splits,
    which creates a *new* autograd node.  This means the graph built by
    ``create_graph=True`` (1st backward) can itself be differentiated
    (2nd backward), giving correct second-order derivatives through
    the communication boundary.

    The layer-sequential structure of DPA3 guarantees that all ranks
    execute A2A calls in the same order, so deadlocks cannot occur.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        send_splits: list[int],
        recv_splits: list[int],
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        return _a2a_raw(x, send_splits, recv_splits, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Recursive call: backward of this node is itself an A2A with
        # swapped splits.  Because we call .apply(), a new autograd node
        # is inserted into the graph, enabling higher-order derivatives.
        grad_input = _AllToAllDouble.apply(
            grad_output,
            ctx.recv_splits,
            ctx.send_splits,
            ctx.group,
        )
        return grad_input, None, None, None


def all_to_all_differentiable(
    x: torch.Tensor,
    send_splits: list[int],
    recv_splits: list[int],
    group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    """Public API for differentiable All-to-All.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    send_splits : list[int]
        Number of rows to send to each rank.
    recv_splits : list[int]
        Number of rows to receive from each rank.
    group : ProcessGroup or None
        Communication group.  When ``None`` (single-GPU / no EP),
        *x* is returned unchanged with gradients flowing through.

    Returns
    -------
    Tensor
        Result of All-to-All, or *x* itself when ``group is None``.
    """
    if group is None:
        return x
    return _AllToAllDouble.apply(x, send_splits, recv_splits, group)
