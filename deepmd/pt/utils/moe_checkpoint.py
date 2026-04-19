# SPDX-License-Identifier: LGPL-3.0-or-later
"""MoE Checkpoint save/load with EP resharding support.

Provides:
- ``moe_state_dict_to_global``: convert local state_dict to global expert indices.
- ``moe_load_state_dict_from_global``: load global state_dict into local model.

When using Expert Parallelism (EP), each GPU only holds a subset of routing
experts.  These functions handle the index renaming needed to save all experts
in a single checkpoint with global indices, and to distribute the correct
subset when loading with a different ``ep_size``.
"""

from __future__ import annotations

import re
from collections import OrderedDict

import torch
import torch.distributed as dist

# Regex to match `.routing_experts.{idx}.` in state_dict keys.
_ROUTING_EXPERT_RE = re.compile(r"\.routing_experts\.(\d+)\.")


def _rename_routing_expert_key(key: str, old_idx: int, new_idx: int) -> str:
    """Rename a routing expert index in a state_dict key.

    Replaces ``.routing_experts.{old_idx}.`` with ``.routing_experts.{new_idx}.``
    (first occurrence only).

    Parameters
    ----------
    key : str
        Original state_dict key.
    old_idx : int
        Current expert index in the key.
    new_idx : int
        Target expert index.

    Returns
    -------
    str
        Key with the expert index replaced.
    """
    return key.replace(
        f".routing_experts.{old_idx}.",
        f".routing_experts.{new_idx}.",
        1,
    )


def moe_state_dict_to_global(
    model: torch.nn.Module,
    ep_rank: int,
    ep_size: int,
    experts_per_gpu: int,
    ep_group: object | None = None,
) -> OrderedDict:
    """Convert a model's local state_dict to global expert indices.

    When ``ep_size == 1``, the local indices are already global (0..n_routing-1),
    so the state_dict is returned as-is.

    When ``ep_size > 1``, this function:
    1. Renames local expert indices to global indices on each rank.
    2. All-gathers routing expert tensors from all EP ranks.
    3. Returns a complete global state_dict (on all ranks).

    Parameters
    ----------
    model : torch.nn.Module
        The model whose state_dict should be saved.
    ep_rank : int
        This GPU's rank within the EP group.
    ep_size : int
        Number of GPUs in the EP group.
    experts_per_gpu : int
        Number of routing experts per GPU.
    ep_group : ProcessGroup or None
        EP communication group (required when ``ep_size > 1``).

    Returns
    -------
    OrderedDict
        Global state_dict with expert indices 0..n_routing_experts-1.
    """
    local_sd = model.state_dict()

    if ep_size <= 1:
        # ep_size=1: local indices are already global.
        return OrderedDict(local_sd)

    # ep_size > 1: rename local → global on this rank.
    n_routing_experts = ep_size * experts_per_gpu

    # Step 1: Rename local indices to global indices.
    renamed_sd = OrderedDict()
    for key, tensor in local_sd.items():
        m = _ROUTING_EXPERT_RE.search(key)
        if m:
            local_idx = int(m.group(1))
            global_idx = ep_rank * experts_per_gpu + local_idx
            new_key = _rename_routing_expert_key(key, local_idx, global_idx)
            renamed_sd[new_key] = tensor
        else:
            renamed_sd[key] = tensor

    # Step 2: All-gather routing expert params from all EP ranks.
    # Collect all routing expert keys and tensors.
    # Each rank broadcasts its routing expert tensors to all other ranks.
    global_sd = OrderedDict()

    # First, add non-routing params (same on all ranks).
    for key, tensor in renamed_sd.items():
        if not _ROUTING_EXPERT_RE.search(key):
            global_sd[key] = tensor

    # Then, gather routing expert params from all ranks.
    # Each rank broadcasts its renamed (global-indexed) routing expert params.
    # Compute this rank's DP rank to convert EP group ranks to global ranks.
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    dp_rank = (global_rank - ep_rank) // ep_size

    for r in range(ep_size):
        # Convert EP group rank r to global rank for broadcast src.
        # EP group layout: [dp_rank * ep_size + 0, ..., dp_rank * ep_size + (ep_size-1)]
        src_global_rank = dp_rank * ep_size + r

        for local_idx in range(experts_per_gpu):
            global_idx = r * experts_per_gpu + local_idx

            # Find all keys for this global expert on the source rank.
            # We need to broadcast each tensor from rank r to all ranks.
            if r == ep_rank:
                # This rank owns this expert; collect keys.
                for key, tensor in renamed_sd.items():
                    m = _ROUTING_EXPERT_RE.search(key)
                    if m and int(m.group(1)) == global_idx:
                        # Broadcast this tensor from this rank.
                        tensor = tensor.contiguous().clone()
                        dist.broadcast(tensor, src=src_global_rank, group=ep_group)
                        global_sd[key] = tensor
            else:
                # This rank doesn't own this expert; receive via broadcast.
                # We need to know the keys and shapes. Since all ranks have
                # the same model structure, we can construct the expected keys.
                # Find local keys that would match this global_idx if we owned it.
                for key, tensor in renamed_sd.items():
                    m = _ROUTING_EXPERT_RE.search(key)
                    if m:
                        local_idx_in_key = int(m.group(1))
                        # This key has our ep_rank's global index. We need the
                        # equivalent key with global_idx instead.
                        our_global = ep_rank * experts_per_gpu + (
                            local_idx_in_key - ep_rank * experts_per_gpu
                        )
                        # Only process if this local key maps to the same
                        # "position" (e.g., same expert collection + suffix).
                        if local_idx_in_key - ep_rank * experts_per_gpu == local_idx:
                            target_key = _rename_routing_expert_key(
                                key, local_idx_in_key, global_idx,
                            )
                            recv_tensor = torch.empty_like(tensor)
                            dist.broadcast(recv_tensor, src=src_global_rank, group=ep_group)
                            global_sd[target_key] = recv_tensor

    return global_sd


def moe_load_state_dict_from_global(
    model: torch.nn.Module,
    global_state_dict: dict,
    ep_rank: int,
    ep_size: int,
    experts_per_gpu: int,
) -> None:
    """Load a global state_dict into a local model with EP resharding.

    Selects the routing experts that belong to this EP rank and renames
    their global indices to local indices.  Non-routing params are loaded
    as-is.

    Parameters
    ----------
    model : torch.nn.Module
        The model to load into.
    global_state_dict : dict
        State dict with global expert indices (0..n_routing_experts-1).
    ep_rank : int
        This GPU's EP rank.
    ep_size : int
        Number of GPUs in the EP group.
    experts_per_gpu : int
        Number of routing experts this GPU should hold.
    """
    n_routing_experts = ep_size * experts_per_gpu

    local_sd = OrderedDict()
    for key, tensor in global_state_dict.items():
        m = _ROUTING_EXPERT_RE.search(key)
        if m:
            global_idx = int(m.group(1))
            # Determine which EP rank owns this expert.
            owner_rank = global_idx // experts_per_gpu
            if owner_rank == ep_rank:
                local_idx = global_idx % experts_per_gpu
                new_key = _rename_routing_expert_key(key, global_idx, local_idx)
                local_sd[new_key] = tensor
            # else: skip, belongs to another GPU.
        else:
            local_sd[key] = tensor

    model.load_state_dict(local_sd, strict=True)
