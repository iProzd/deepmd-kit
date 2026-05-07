// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdint>

namespace deepmd {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Fused MoE pack-for-dispatch.
//
// Input:
//   node_sorted  : [N_node , D_node ]  (sorted by destination GPU)
//   edge_sorted  : [N_edge , D_edge ]
//   angle_sorted : [N_angle, D_angle]
//
// Output:
//   packed       : [total_packed_rows, D_packed]   (zero-padded)
//
// Layout per GPU g (rows are concatenated in g order):
//   - n_node[g] rows: cols [0:D_node) = node feat, cols [D_node:D_packed) = 0
//   - n_edge_rows[g] = ceil(n_edge[g]/edge_concat) rows:
//       cols [k*D_edge : (k+1)*D_edge) = edge_sorted[..]; remainder zero
//   - n_angle_rows[g] = ceil(n_angle[g]/angle_concat) rows: same idea
//
// Offsets (all int64, on GPU, length ep_size or ep_size+1):
//   *_in_offset [ep_size+1] : prefix sum of *_per_gpu counts (input row indexing)
//   *_out_offset[ep_size]   : starting row in packed for that block
template <typename FPTYPE>
void moe_pack_for_dispatch_forward_gpu(
    FPTYPE* packed,
    const FPTYPE* node_sorted,
    const FPTYPE* edge_sorted,
    const FPTYPE* angle_sorted,
    const int64_t* node_in_offset,
    const int64_t* edge_in_offset,
    const int64_t* angle_in_offset,
    const int64_t* node_out_offset,
    const int64_t* edge_out_offset,
    const int64_t* angle_out_offset,
    const int total_packed_rows,
    const int ep_size,
    const int N_node,
    const int N_edge,
    const int N_angle,
    const int D_node,
    const int D_edge,
    const int D_angle,
    const int D_packed,
    const int edge_concat,
    const int angle_concat);

// Backward: gather grads from packed back into the three sorted-grad tensors.
template <typename FPTYPE>
void moe_pack_for_dispatch_backward_gpu(
    FPTYPE* grad_node_sorted,
    FPTYPE* grad_edge_sorted,
    FPTYPE* grad_angle_sorted,
    const FPTYPE* grad_packed,
    const int64_t* node_in_offset,
    const int64_t* edge_in_offset,
    const int64_t* angle_in_offset,
    const int64_t* node_out_offset,
    const int64_t* edge_out_offset,
    const int64_t* angle_out_offset,
    const int ep_size,
    const int N_node,
    const int N_edge,
    const int N_angle,
    const int D_node,
    const int D_edge,
    const int D_angle,
    const int D_packed,
    const int edge_concat,
    const int angle_concat);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace deepmd
