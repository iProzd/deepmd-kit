# SPDX-License-Identifier: LGPL-3.0-or-later
"""Optimized batched expert forward for MoE expert collection.

Uses torch.bmm with padding when expert loads are balanced, otherwise
falls back to a per-expert loop.
"""

from __future__ import annotations

import torch
from torch.profiler import record_function


def batched_expert_forward_optimized(
    sorted_features: torch.Tensor,
    split_sizes: list[int],
    W: torch.Tensor,
    b: torch.Tensor,
    activate_fn,
    use_bmm: bool = True,
) -> torch.Tensor:
    """Batched expert forward.

    Uses torch.bmm for balanced loads, falls back to loop for unbalanced.

    Parameters
    ----------
    sorted_features : Tensor [N_total, num_in]
    split_sizes : list[int]
    W : Tensor [num_experts, num_in, num_out]
    b : Tensor [num_out, num_experts]
    activate_fn : callable
    use_bmm : bool

    Returns
    -------
    Tensor [N_total, num_out]
    """
    N_total = sorted_features.shape[0]
    if N_total == 0:
        return sorted_features.new_zeros(0, W.shape[2])

    num_experts = len(split_sizes)
    max_tokens = max(split_sizes) if split_sizes else 0
    min_tokens = min(s for s in split_sizes if s > 0) if any(s > 0 for s in split_sizes) else 0

    if max_tokens == 0:
        return sorted_features.new_zeros(0, W.shape[2])

    # Check load balance - relaxed threshold (50%) and lower min_tokens (8)
    # This allows more cases to use bmm, trading some padding overhead for parallelism
    load_balanced = (use_bmm and max_tokens >= 8 and
                    (max_tokens - min_tokens) / max(max_tokens, 1) < 0.5)

    with record_function("batched_expert_forward_opt"):
        if load_balanced:
            # Use batched matmul for balanced loads
            with record_function("bmm_path"):
                chunks = torch.split(sorted_features, split_sizes)

                # Pre-allocate batched tensor to avoid repeated padding
                num_in = sorted_features.shape[1]
                batched_input = torch.zeros(
                    num_experts, max_tokens, num_in,
                    device=sorted_features.device, dtype=sorted_features.dtype
                )

                # Copy chunks into pre-allocated tensor (faster than pad + stack)
                for eid, chunk in enumerate(chunks):
                    if chunk.shape[0] > 0:
                        batched_input[eid, :chunk.shape[0], :] = chunk

                # Batched matmul: [E, N, I] @ [E, I, O] -> [E, N, O]
                batched_output = torch.bmm(batched_input, W)

                # Add bias
                batched_output = batched_output + b.t().unsqueeze(1)

                # Unpad and concatenate
                out_parts = []
                for eid, size in enumerate(split_sizes):
                    if size > 0:
                        out_parts.append(batched_output[eid, :size, :])
                    else:
                        out_parts.append(batched_output.new_zeros(0, batched_output.shape[2]))

                sorted_out = torch.cat(out_parts, dim=0)
        else:
            # Fall back to loop for unbalanced loads
            with record_function("loop_path"):
                chunks = torch.split(sorted_features, split_sizes)
                out_parts = []
                for eid, chunk in enumerate(chunks):
                    if chunk.shape[0] > 0:
                        y = torch.matmul(chunk, W[eid]) + b[:, eid]
                        out_parts.append(y)
                    else:
                        out_parts.append(chunk.new_zeros(0, W.shape[2]))
                sorted_out = torch.cat(out_parts, dim=0)

        # Apply activation
        return activate_fn(sorted_out)
