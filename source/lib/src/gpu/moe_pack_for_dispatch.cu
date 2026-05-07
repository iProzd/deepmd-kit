// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused MoE pack-for-dispatch kernel.
//
// Replaces the per-GPU Python loop in `MoEPacker.pack_for_dispatch`, which
// was a sequence of slice + F.pad + reshape + concat repeated `ep_size`
// times.  The output `packed` tensor is a [total_rows, D_packed] block-
// concatenation of, for each GPU g:
//   - `n_node[g]` rows holding node features padded with zeros, then
//   - `ceil(n_edge[g]/edge_concat)` rows holding `edge_concat` edge tokens
//     concatenated horizontally (zero-padded on the trailing slot if not
//     divisible), then
//   - `ceil(n_angle[g]/angle_concat)` rows holding angle tokens, same idea.
//
// The kernel is *input-driven*: each input row writes to exactly one output
// position, so we simply launch one CUDA block per input row.  Output
// padding (both row-tail and column-tail) is taken care of by zero-init'ing
// `packed` once via cudaMemset.

#include <cstdint>

#include "device.h"
#include "moe_pack_for_dispatch.h"

namespace deepmd {

namespace {

// For a global input row `r` in [0, *_in_offset[ep_size]), find the GPU
// index g such that *_in_offset[g] <= r < *_in_offset[g+1].  ep_size is
// small (single digits typical, up to a few tens), so a serial scan in
// each thread is fine and avoids divergent binary search.
__device__ __forceinline__ int find_gpu(const int64_t* __restrict__ in_offset,
                                        const int ep_size,
                                        const int r) {
  int g = 0;
#pragma unroll 1
  while (g + 1 < ep_size && r >= static_cast<int>(in_offset[g + 1])) {
    ++g;
  }
  return g;
}

// ---------------------------------------------------------------------------
// Forward (scatter): node copy with column padding.
//
// For input row r in [0, N_node):
//   g       = find_gpu(node_in_offset, ep_size, r)
//   local   = r - node_in_offset[g]
//   dst_row = node_out_offset[g] + local
//   packed[dst_row, 0:D_node] = node_sorted[r, 0:D_node]
//   (cols [D_node:D_packed] left as zero by memset.)
// ---------------------------------------------------------------------------
template <typename FPTYPE>
__global__ void pack_node_scatter_kernel(
    FPTYPE* __restrict__ packed,
    const FPTYPE* __restrict__ node_sorted,
    const int64_t* __restrict__ node_in_offset,
    const int64_t* __restrict__ node_out_offset,
    const int ep_size,
    const int N_node,
    const int D_node,
    const int D_packed) {
  const int r = blockIdx.x;
  if (r >= N_node) {
    return;
  }
  const int g = find_gpu(node_in_offset, ep_size, r);
  const int local = r - static_cast<int>(node_in_offset[g]);
  const int dst_row = static_cast<int>(node_out_offset[g]) + local;
  const FPTYPE* src = node_sorted + r * D_node;
  FPTYPE* dst = packed + dst_row * D_packed;
  for (int c = threadIdx.x; c < D_node; c += blockDim.x) {
    dst[c] = src[c];
  }
}

// Backward (gather) for node:
//   grad_node_sorted[r, c] = grad_packed[dst_row, c]
template <typename FPTYPE>
__global__ void pack_node_gather_kernel(
    FPTYPE* __restrict__ grad_node_sorted,
    const FPTYPE* __restrict__ grad_packed,
    const int64_t* __restrict__ node_in_offset,
    const int64_t* __restrict__ node_out_offset,
    const int ep_size,
    const int N_node,
    const int D_node,
    const int D_packed) {
  const int r = blockIdx.x;
  if (r >= N_node) {
    return;
  }
  const int g = find_gpu(node_in_offset, ep_size, r);
  const int local = r - static_cast<int>(node_in_offset[g]);
  const int dst_row = static_cast<int>(node_out_offset[g]) + local;
  const FPTYPE* src = grad_packed + dst_row * D_packed;
  FPTYPE* dst = grad_node_sorted + r * D_node;
  for (int c = threadIdx.x; c < D_node; c += blockDim.x) {
    dst[c] = src[c];
  }
}

// ---------------------------------------------------------------------------
// Forward (scatter): edge / angle concat (group_size rows of width D_in
// pack into a single [D_packed] row, with zero-pad on trailing slots).
//
// For input row r in [0, N_in):
//   g       = find_gpu(in_offset, ep_size, r)
//   local   = r - in_offset[g]
//   j       = local / group_size              (which output row)
//   k       = local % group_size              (which slot inside that row)
//   dst_row = out_offset[g] + j
//   packed[dst_row, k*D_in : (k+1)*D_in] = sorted[r, 0:D_in]
// ---------------------------------------------------------------------------
template <typename FPTYPE>
__global__ void pack_concat_scatter_kernel(
    FPTYPE* __restrict__ packed,
    const FPTYPE* __restrict__ sorted,
    const int64_t* __restrict__ in_offset,
    const int64_t* __restrict__ out_offset,
    const int ep_size,
    const int N_in,
    const int D_in,
    const int D_packed,
    const int group_size) {
  const int r = blockIdx.x;
  if (r >= N_in) {
    return;
  }
  const int g = find_gpu(in_offset, ep_size, r);
  const int local = r - static_cast<int>(in_offset[g]);
  const int j = local / group_size;
  const int k = local % group_size;
  const int dst_row = static_cast<int>(out_offset[g]) + j;
  const int dst_col_base = k * D_in;
  const FPTYPE* src = sorted + r * D_in;
  FPTYPE* dst = packed + dst_row * D_packed + dst_col_base;
  for (int c = threadIdx.x; c < D_in; c += blockDim.x) {
    dst[c] = src[c];
  }
}

// Backward (gather) for concat group:
//   grad_sorted[r, c] = grad_packed[dst_row, k*D_in + c]
template <typename FPTYPE>
__global__ void pack_concat_gather_kernel(
    FPTYPE* __restrict__ grad_sorted,
    const FPTYPE* __restrict__ grad_packed,
    const int64_t* __restrict__ in_offset,
    const int64_t* __restrict__ out_offset,
    const int ep_size,
    const int N_in,
    const int D_in,
    const int D_packed,
    const int group_size) {
  const int r = blockIdx.x;
  if (r >= N_in) {
    return;
  }
  const int g = find_gpu(in_offset, ep_size, r);
  const int local = r - static_cast<int>(in_offset[g]);
  const int j = local / group_size;
  const int k = local % group_size;
  const int dst_row = static_cast<int>(out_offset[g]) + j;
  const int dst_col_base = k * D_in;
  const FPTYPE* src = grad_packed + dst_row * D_packed + dst_col_base;
  FPTYPE* dst = grad_sorted + r * D_in;
  for (int c = threadIdx.x; c < D_in; c += blockDim.x) {
    dst[c] = src[c];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

template <typename FPTYPE>
void moe_pack_for_dispatch_forward_gpu(FPTYPE* packed,
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
                                        const int angle_concat) {
  // Zero-init the entire output: any padding (column tails for node block,
  // row tails for the trailing partial group of edge/angle, column tails
  // for edge that does not exactly fill 40a) ends up as zeros automatically.
  if (total_packed_rows > 0) {
    DPErrcheck(cudaMemset(
        packed, 0,
        sizeof(FPTYPE) * static_cast<size_t>(total_packed_rows) * D_packed));
  }

  if (N_node > 0) {
    pack_node_scatter_kernel<FPTYPE><<<N_node, TPB>>>(
        packed, node_sorted, node_in_offset, node_out_offset, ep_size, N_node,
        D_node, D_packed);
    DPErrcheck(cudaGetLastError());
  }
  if (N_edge > 0) {
    pack_concat_scatter_kernel<FPTYPE><<<N_edge, TPB>>>(
        packed, edge_sorted, edge_in_offset, edge_out_offset, ep_size, N_edge,
        D_edge, D_packed, edge_concat);
    DPErrcheck(cudaGetLastError());
  }
  if (N_angle > 0) {
    pack_concat_scatter_kernel<FPTYPE><<<N_angle, TPB>>>(
        packed, angle_sorted, angle_in_offset, angle_out_offset, ep_size,
        N_angle, D_angle, D_packed, angle_concat);
    DPErrcheck(cudaGetLastError());
  }
}

template <typename FPTYPE>
void moe_pack_for_dispatch_backward_gpu(FPTYPE* grad_node_sorted,
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
                                         const int angle_concat) {
  // Each input row reads from exactly one slice of grad_packed; writes are
  // all disjoint across input rows, so no atomics or memset of the
  // grad_*_sorted outputs is needed.
  if (N_node > 0) {
    pack_node_gather_kernel<FPTYPE><<<N_node, TPB>>>(
        grad_node_sorted, grad_packed, node_in_offset, node_out_offset,
        ep_size, N_node, D_node, D_packed);
    DPErrcheck(cudaGetLastError());
  }
  if (N_edge > 0) {
    pack_concat_gather_kernel<FPTYPE><<<N_edge, TPB>>>(
        grad_edge_sorted, grad_packed, edge_in_offset, edge_out_offset,
        ep_size, N_edge, D_edge, D_packed, edge_concat);
    DPErrcheck(cudaGetLastError());
  }
  if (N_angle > 0) {
    pack_concat_gather_kernel<FPTYPE><<<N_angle, TPB>>>(
        grad_angle_sorted, grad_packed, angle_in_offset, angle_out_offset,
        ep_size, N_angle, D_angle, D_packed, angle_concat);
    DPErrcheck(cudaGetLastError());
  }
}

// Explicit instantiations.
template void moe_pack_for_dispatch_forward_gpu<float>(
    float*, const float*, const float*, const float*,
    const int64_t*, const int64_t*, const int64_t*,
    const int64_t*, const int64_t*, const int64_t*,
    const int, const int, const int, const int, const int,
    const int, const int, const int, const int, const int, const int);
template void moe_pack_for_dispatch_forward_gpu<double>(
    double*, const double*, const double*, const double*,
    const int64_t*, const int64_t*, const int64_t*,
    const int64_t*, const int64_t*, const int64_t*,
    const int, const int, const int, const int, const int,
    const int, const int, const int, const int, const int, const int);

template void moe_pack_for_dispatch_backward_gpu<float>(
    float*, float*, float*, const float*,
    const int64_t*, const int64_t*, const int64_t*,
    const int64_t*, const int64_t*, const int64_t*,
    const int, const int, const int, const int,
    const int, const int, const int, const int, const int, const int);
template void moe_pack_for_dispatch_backward_gpu<double>(
    double*, double*, double*, const double*,
    const int64_t*, const int64_t*, const int64_t*,
    const int64_t*, const int64_t*, const int64_t*,
    const int, const int, const int, const int,
    const int, const int, const int, const int, const int, const int);

}  // namespace deepmd
