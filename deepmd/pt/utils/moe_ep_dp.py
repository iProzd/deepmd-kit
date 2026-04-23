# SPDX-License-Identifier: LGPL-3.0-or-later
"""MoE Expert-Parallelism + Data-Parallelism process group management.

Provides:
- ``init_ep_dp_groups``: create EP and DP process groups from a flat world.
- ``sync_moe_gradients``: all-reduce gradients with correct group/divisor.
- ``_is_routing_expert_param``: classify parameter names.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def init_ep_dp_groups(
    ep_size: int = 1,
) -> tuple[object | None, object | None, int, int, int, int]:
    """Initialize EP and DP process groups from a flat world.

    The world of ``world_size`` GPUs is viewed as a 2-D grid::

        world_size = ep_size × dp_size

        GPU layout (ep_size=2, dp_size=2, world_size=4):

                  EP rank 0  EP rank 1
        DP rank 0:  GPU 0     GPU 1    ← ep_group_0
        DP rank 1:  GPU 2     GPU 3    ← ep_group_1
                    ↑ dp_group_0  ↑ dp_group_1

    Parameters
    ----------
    ep_size : int
        Number of GPUs per expert-parallel group.  When ``ep_size <= 1``
        or distributed is not initialised, no groups are created.

    Returns
    -------
    ep_group : ProcessGroup or None
        The EP group this rank belongs to (for All-to-All).
    dp_group : ProcessGroup or None
        The DP group this rank belongs to (for routing-expert gradient sync).
    ep_rank : int
        This rank's position inside its EP group.
    ep_size : int
        Size of the EP group (echoed back, or 1 if disabled).
    dp_rank : int
        This rank's position inside its DP group.
    dp_size : int
        Size of the DP group.
    """
    if ep_size <= 1 or not dist.is_initialized():
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        return (None, None, 0, 1, rank, world_size)

    world_size = dist.get_world_size()
    world_rank = dist.get_rank()

    if world_size % ep_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by ep_size ({ep_size})"
        )

    dp_size = world_size // ep_size

    # Build EP groups: each row of the GPU grid.
    # ALL ranks must call new_group for every group (NCCL requirement).
    my_ep_group = None
    for dp_idx in range(dp_size):
        ranks = [dp_idx * ep_size + i for i in range(ep_size)]
        group = dist.new_group(ranks)
        if world_rank in ranks:
            my_ep_group = group

    # Build DP groups: each column of the GPU grid.
    my_dp_group = None
    for ep_idx in range(ep_size):
        ranks = [dp_idx * ep_size + ep_idx for dp_idx in range(dp_size)]
        group = dist.new_group(ranks)
        if world_rank in ranks:
            my_dp_group = group

    ep_rank = world_rank % ep_size
    dp_rank = world_rank // ep_size

    return (my_ep_group, my_dp_group, ep_rank, ep_size, dp_rank, dp_size)


def _is_routing_expert_param(name: str) -> bool:
    """Check whether a parameter belongs to a routing expert.

    Routing expert parameters are identified by the presence of
    ``routing_matrix`` or ``routing_bias`` in their fully-qualified name
    (for the shared 3D tensor layout), or the legacy ``.routing_experts.``
    pattern (for backward compatibility).

    Examples::

        moe_phase1.node_self_experts.routing_matrix         → True
        moe_phase1.node_self_experts.routing_bias            → True
        moe_phase1.node_self_experts.routing_experts.0.mlp.matrix  → True (legacy)
        moe_phase1.edge_experts.shared_experts.0.mlp.matrix  → False
        node_router.gate.matrix                               → False
        n_residual.0                                          → False
    """
    return (
        ".routing_matrix" in name
        or ".routing_bias" in name
        or ".routing_experts." in name
    )


def sync_moe_gradients(
    model: torch.nn.Module,
    dp_group: object | None,
    world_group: object | None,
    dp_size: int,
    world_size: int,
) -> None:
    """All-reduce gradients with the correct group and divisor.

    Must be called **after** ``loss.backward()`` and **before**
    ``optimizer.step()``.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameter gradients should be synchronised.
    dp_group : ProcessGroup or None
        DP group for routing-expert gradient all-reduce.
    world_group : ProcessGroup or None
        World group for all other parameters.  ``None`` uses the
        default process group (all ranks).
    dp_size : int
        Number of ranks in the DP group.
    world_size : int
        Total number of ranks.
    """
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if _is_routing_expert_param(name):
            # Routing expert grads: all-reduce across DP group only (same expert
            # exists only on dp_size ranks in the same DP column).
            # Divide by world_size (not dp_size) because All-to-All backward
            # already aggregates gradients from ep_size ranks within the EP group.
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=dp_group)
            param.grad.div_(world_size)
        else:
            dist.all_reduce(
                param.grad, op=dist.ReduceOp.SUM, group=world_group
            )
            param.grad.div_(world_size)
