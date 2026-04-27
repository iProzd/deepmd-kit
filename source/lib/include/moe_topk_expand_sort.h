// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdint>

namespace deepmd {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Forward: counting-sort based topk expand + sort fused with feature scatter.
// Replaces argsort + repeat_interleave + bincount + inverse-permutation.
//
// Inputs:
//   features        : [N, feat_dim]
//   topk_indices    : [N * topk] (int64, global expert IDs)
//   topk_weights    : [N * topk] (already flattened by caller)
// Outputs:
//   sorted_features    : [N * topk, feat_dim]
//   sorted_expert_ids  : [N * topk] int64
//   sorted_weights     : [N * topk]
//   unsort_idx         : [N * topk] int64; unsort_idx[i*topk+k] = sorted_position
//   gpu_counts         : [ep_size] int64; tokens destined per GPU
// Workspace (device, caller-provided):
//   d_hist             : [n_routing_experts]   int32, will be zeroed inside
//   d_offsets          : [n_routing_experts+1] int32
//   d_running          : [n_routing_experts]   int32, will be zeroed inside
template <typename FPTYPE>
void moe_topk_expand_sort_forward_gpu(FPTYPE* sorted_features,
                                      int64_t* sorted_expert_ids,
                                      FPTYPE* sorted_weights,
                                      int64_t* unsort_idx,
                                      int64_t* gpu_counts,
                                      int* d_hist,
                                      int* d_offsets,
                                      int* d_running,
                                      const FPTYPE* features,
                                      const int64_t* topk_indices,
                                      const FPTYPE* topk_weights,
                                      const int N,
                                      const int topk,
                                      const int feat_dim,
                                      const int n_routing_experts,
                                      const int experts_per_gpu,
                                      const int ep_size);

// Backward: gather-and-sum on features; gather on weights.
//   grad_features[i, d] = sum_k grad_sorted_features[unsort_idx[i*topk+k], d]
//   grad_topk_weights[i*topk+k] = grad_sorted_weights[unsort_idx[i*topk+k]]
template <typename FPTYPE>
void moe_topk_expand_sort_backward_gpu(FPTYPE* grad_features,
                                       FPTYPE* grad_topk_weights,
                                       const FPTYPE* grad_sorted_features,
                                       const FPTYPE* grad_sorted_weights,
                                       const int64_t* unsort_idx,
                                       const int N,
                                       const int topk,
                                       const int feat_dim);

// Double backward: scatter (same pattern as forward's feature scatter).
//   ggrad_sorted_features[unsort_idx[i*topk+k], :] = ggrad_features[i, :]
//   ggrad_sorted_weights[unsort_idx[i*topk+k]] = ggrad_topk_weights[i*topk+k]
template <typename FPTYPE>
void moe_topk_expand_sort_double_backward_gpu(
    FPTYPE* ggrad_sorted_features,
    FPTYPE* ggrad_sorted_weights,
    const FPTYPE* ggrad_features,
    const FPTYPE* ggrad_topk_weights,
    const int64_t* unsort_idx,
    const int N,
    const int topk,
    const int feat_dim);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace deepmd
