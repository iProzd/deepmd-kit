# SPDX-License-Identifier: LGPL-3.0-or-later
"""MoE Checkpoint save/load with EP resharding support.

Provides:
- ``moe_state_dict_to_global``: convert local state_dict to global expert indices.
- ``moe_load_state_dict_from_global``: load global state_dict into local model.

When using Expert Parallelism (EP), each GPU only holds a subset of routing
experts.  With the shared 3D tensor layout, each GPU's ``routing_matrix``
has shape ``[num_in, num_out, experts_per_gpu]`` and ``routing_bias`` has
shape ``[num_out, experts_per_gpu]``.  These functions handle gathering all
expert slices into global tensors for saving, and distributing the correct
slices when loading with a potentially different ``ep_size``.

The ``_ExpertView`` sub-modules (``routing_experts.{idx}``) have NO
parameters of their own — they're lightweight facades.  The checkpoint
code operates on the actual ``routing_matrix`` / ``routing_bias`` tensors.

Legacy support: the old checkpoint format with ``routing_experts.{idx}.mlp.matrix``
keys is also handled for backward compatibility.
"""

from __future__ import annotations

import re
from collections import OrderedDict

import torch
import torch.distributed as dist

# Regex to match `.routing_experts.{idx}.` in state_dict keys (legacy format).
_ROUTING_EXPERT_RE = re.compile(r"\.routing_experts\.(\d+)\.")

# Regex to match `.routing_matrix` or `.routing_bias` (new shared 3D format).
_ROUTING_3D_RE = re.compile(r"\.(routing_matrix|routing_bias)$")


def _is_routing_3d_param(key: str) -> bool:
    """Check if key is a shared 3D routing tensor (routing_matrix or routing_bias)."""
    return key.endswith(".routing_matrix") or key.endswith(".routing_bias")


def _rename_routing_expert_key(key: str, old_idx: int, new_idx: int) -> str:
    """Rename a routing expert index in a state_dict key (legacy format).

    Replaces ``.routing_experts.{old_idx}.`` with ``.routing_experts.{new_idx}.``
    (first occurrence only).
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

    When ``ep_size == 1``, the local state_dict is already global (all experts
    on one GPU), so it is returned as-is.

    When ``ep_size > 1``, this function:
    1. All-gathers ``routing_matrix`` and ``routing_bias`` tensors from all
       EP ranks, concatenating along the expert dimension.
    2. Returns a complete global state_dict (on all ranks).

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
        Global state_dict with all routing expert parameters.
    """
    local_sd = model.state_dict()

    if ep_size <= 1:
        return OrderedDict(local_sd)

    # ep_size > 1: gather routing_matrix/routing_bias from all ranks.
    global_sd = OrderedDict()
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    dp_rank = (global_rank - ep_rank) // ep_size

    for key, tensor in local_sd.items():
        if _is_routing_3d_param(key):
            # Shared 3D tensor: gather from all EP ranks.
            # routing_matrix: [I, O, experts_per_gpu] → cat to [I, O, n_routing]
            # routing_bias:   [O, experts_per_gpu] → cat to [O, n_routing]
            expert_dim = tensor.dim() - 1  # last dim is expert dim

            # All-gather: collect tensors from all EP ranks.
            gathered = [torch.empty_like(tensor) for _ in range(ep_size)]
            dist.all_gather(gathered, tensor.contiguous(), group=ep_group)

            # Concatenate along expert dimension.
            global_tensor = torch.cat(gathered, dim=expert_dim)
            global_sd[key] = global_tensor
        elif _ROUTING_EXPERT_RE.search(key):
            # Skip _ExpertView sub-module keys (they have no parameters,
            # but if any slip through, ignore them).
            pass
        else:
            # Non-routing params: same on all ranks.
            global_sd[key] = tensor

    return global_sd


def moe_load_state_dict_from_global(
    model: torch.nn.Module,
    global_state_dict: dict,
    ep_rank: int,
    ep_size: int,
    experts_per_gpu: int,
) -> None:
    """Load a global state_dict into a local model with EP resharding.

    Selects the routing expert slice that belongs to this EP rank by
    indexing into the shared 3D tensor's expert dimension.

    Parameters
    ----------
    model : torch.nn.Module
        The model to load into.
    global_state_dict : dict
        State dict with global routing parameters.
    ep_rank : int
        This GPU's EP rank.
    ep_size : int
        Number of GPUs in the EP group.
    experts_per_gpu : int
        Number of routing experts this GPU should hold.
    """
    local_sd = OrderedDict()

    for key, tensor in global_state_dict.items():
        if _is_routing_3d_param(key):
            # Shared 3D tensor: slice out this rank's experts.
            expert_dim = tensor.dim() - 1
            start = ep_rank * experts_per_gpu
            end = start + experts_per_gpu
            if expert_dim == 2:
                # routing_matrix: [I, O, n_routing] → [I, O, experts_per_gpu]
                local_sd[key] = tensor[:, :, start:end].contiguous()
            elif expert_dim == 1:
                # routing_bias: [O, n_routing] → [O, experts_per_gpu]
                local_sd[key] = tensor[:, start:end].contiguous()
            else:
                # Fallback: use narrow on last dim
                local_sd[key] = tensor.narrow(expert_dim, start, experts_per_gpu).contiguous()
        elif _ROUTING_EXPERT_RE.search(key):
            # Legacy format: routing_experts.{idx}.mlp.matrix / .bias
            # Handle by extracting the global index and mapping to local.
            m = _ROUTING_EXPERT_RE.search(key)
            if m:
                global_idx = int(m.group(1))
                owner_rank = global_idx // experts_per_gpu
                if owner_rank == ep_rank:
                    local_idx = global_idx % experts_per_gpu
                    new_key = _rename_routing_expert_key(key, global_idx, local_idx)
                    local_sd[new_key] = tensor
        else:
            local_sd[key] = tensor

    model.load_state_dict(local_sd, strict=True)
