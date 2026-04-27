// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused MoE topk-expand-sort kernel.
//
// Replaces the PyTorch reference (`_topk_expand_sort` in moe_layer.py),
// which is a chain of repeat_interleave + argsort + multiple gather +
// inverse-permutation + bincount + .max().item() + .tolist().
//
// Strategy: the number of routing experts is small (tens to ~hundreds),
// so a counting sort beats a comparison-based stable argsort and lets us
// fuse the per-token feature gather into the same scatter pass.  The
// resulting permutation is *not* required to be stable (the consumer
// only needs each expert's segment to be contiguous, and unsort_idx is
// constructed from the same permutation we choose, so forward/backward
// are consistent).

#include <cstdint>

#include "device.h"
#include "moe_topk_expand_sort.h"

namespace deepmd {

namespace {

// Histogram per global expert ID.  total = N * topk.
__global__ void moe_topk_histogram_kernel(const int64_t* __restrict__ topk_indices,
                                          int* __restrict__ hist,
                                          const int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int eid = static_cast<int>(topk_indices[idx]);
  atomicAdd(&hist[eid], 1);
}

// Single-block exclusive scan over [E] -> [E+1].
// Requires blockDim.x >= E + 1 (E is small, typically 8..128).
__global__ void moe_exclusive_scan_kernel(const int* __restrict__ hist,
                                          int* __restrict__ offsets,
                                          const int E) {
  extern __shared__ int sdata[];
  const int tid = threadIdx.x;
  // sdata[t] starts as hist[t-1] for t in [1, E], 0 for t == 0.
  int v = 0;
  if (tid > 0 && tid <= E) {
    v = hist[tid - 1];
  }
  if (tid <= E) {
    sdata[tid] = v;
  }
  __syncthreads();

  // Hillis-Steele inclusive scan over the (E+1)-element array.
  for (int offset = 1; offset <= E; offset <<= 1) {
    int add = 0;
    if (tid <= E && tid >= offset) {
      add = sdata[tid - offset];
    }
    __syncthreads();
    if (tid <= E) {
      sdata[tid] += add;
    }
    __syncthreads();
  }

  if (tid <= E) {
    offsets[tid] = sdata[tid];
  }
}

// Sum hist over each GPU's experts.  ep_size threads.
__global__ void moe_gpu_counts_kernel(const int* __restrict__ hist,
                                      int64_t* __restrict__ gpu_counts,
                                      const int experts_per_gpu,
                                      const int ep_size) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= ep_size) {
    return;
  }
  int sum = 0;
  const int base = g * experts_per_gpu;
#pragma unroll 1
  for (int e = 0; e < experts_per_gpu; ++e) {
    sum += hist[base + e];
  }
  gpu_counts[g] = static_cast<int64_t>(sum);
}

// One thread per (i, k).  Atomically claim a slot within the destination
// expert's segment, write expert ID + unsort_idx mapping.  total = N*topk.
__global__ void moe_assign_dst_kernel(const int64_t* __restrict__ topk_indices,
                                      const int* __restrict__ offsets,
                                      int* __restrict__ running,
                                      int64_t* __restrict__ sorted_expert_ids,
                                      int64_t* __restrict__ unsort_idx,
                                      const int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int eid = static_cast<int>(topk_indices[idx]);
  const int rank = atomicAdd(&running[eid], 1);
  const int dst = offsets[eid] + rank;
  sorted_expert_ids[dst] = static_cast<int64_t>(eid);
  unsort_idx[idx] = static_cast<int64_t>(dst);
}

// One block per (i, k); threads cooperate over feat_dim.
template <typename FPTYPE>
__global__ void moe_scatter_features_kernel(
    const FPTYPE* __restrict__ features,
    const int64_t* __restrict__ unsort_idx,
    FPTYPE* __restrict__ sorted_features,
    const int N,
    const int topk,
    const int feat_dim) {
  const int idx = blockIdx.x;  // i*topk + k
  if (idx >= N * topk) {
    return;
  }
  const int i = idx / topk;
  const int dst = static_cast<int>(unsort_idx[idx]);

  const FPTYPE* src_ptr = features + static_cast<size_t>(i) * feat_dim;
  FPTYPE* dst_ptr = sorted_features + static_cast<size_t>(dst) * feat_dim;

  for (int d = threadIdx.x; d < feat_dim; d += blockDim.x) {
    dst_ptr[d] = src_ptr[d];
  }
}

// One thread per (i, k) writing one float.
template <typename FPTYPE>
__global__ void moe_scatter_weights_kernel(
    const FPTYPE* __restrict__ topk_weights,
    const int64_t* __restrict__ unsort_idx,
    FPTYPE* __restrict__ sorted_weights,
    const int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  sorted_weights[unsort_idx[idx]] = topk_weights[idx];
}

// Backward feature gather-sum.  Block grid: (N, ceil(feat_dim/TPB)).
template <typename FPTYPE>
__global__ void moe_grad_features_kernel(
    const FPTYPE* __restrict__ grad_sorted_features,
    const int64_t* __restrict__ unsort_idx,
    FPTYPE* __restrict__ grad_features,
    const int N,
    const int topk,
    const int feat_dim) {
  const int i = blockIdx.x;
  if (i >= N) {
    return;
  }
  const int d = threadIdx.x + blockIdx.y * blockDim.x;
  if (d >= feat_dim) {
    return;
  }

  FPTYPE sum = static_cast<FPTYPE>(0);
  const int64_t* unsort_row = unsort_idx + static_cast<size_t>(i) * topk;
#pragma unroll 1
  for (int k = 0; k < topk; ++k) {
    const int j = static_cast<int>(unsort_row[k]);
    sum += grad_sorted_features[static_cast<size_t>(j) * feat_dim + d];
  }
  grad_features[static_cast<size_t>(i) * feat_dim + d] = sum;
}

// Backward weights gather (one-to-one).  total = N*topk.
template <typename FPTYPE>
__global__ void moe_grad_weights_kernel(
    const FPTYPE* __restrict__ grad_sorted_weights,
    const int64_t* __restrict__ unsort_idx,
    FPTYPE* __restrict__ grad_topk_weights,
    const int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  grad_topk_weights[idx] = grad_sorted_weights[unsort_idx[idx]];
}

}  // namespace

// ======================================================================
//                       Public entry points
// ======================================================================

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
                                      const int ep_size) {
  const int total = N * topk;

  // Always zero-init gpu_counts (may be needed even if N == 0).
  DPErrcheck(cudaMemset(gpu_counts, 0, sizeof(int64_t) * ep_size));

  if (total == 0) {
    return;
  }

  // 1. Zero counters.
  DPErrcheck(cudaMemset(d_hist, 0, sizeof(int) * n_routing_experts));
  DPErrcheck(cudaMemset(d_running, 0, sizeof(int) * n_routing_experts));

  // 2. Histogram.
  {
    const int blocks = (total + TPB - 1) / TPB;
    moe_topk_histogram_kernel<<<blocks, TPB>>>(topk_indices, d_hist, total);
    DPErrcheck(cudaGetLastError());
  }

  // 3. Exclusive scan.  Single block; pad threads up to power-of-two.
  {
    int threads = 1;
    while (threads < n_routing_experts + 1) {
      threads <<= 1;
    }
    if (threads < 32) {
      threads = 32;
    }
    moe_exclusive_scan_kernel<<<1, threads, sizeof(int) * threads>>>(
        d_hist, d_offsets, n_routing_experts);
    DPErrcheck(cudaGetLastError());
  }

  // 4. Per-GPU counts.
  {
    const int blocks = (ep_size + TPB - 1) / TPB;
    moe_gpu_counts_kernel<<<blocks, TPB>>>(d_hist, gpu_counts,
                                           experts_per_gpu, ep_size);
    DPErrcheck(cudaGetLastError());
  }

  // 5. Assign destinations + write expert IDs and unsort_idx.
  {
    const int blocks = (total + TPB - 1) / TPB;
    moe_assign_dst_kernel<<<blocks, TPB>>>(topk_indices, d_offsets, d_running,
                                           sorted_expert_ids, unsort_idx,
                                           total);
    DPErrcheck(cudaGetLastError());
  }

  // 6. Scatter features.
  {
    const int threads = (feat_dim < TPB) ? ((feat_dim + 31) & ~31) : TPB;
    const int t = threads > 0 ? threads : 32;
    moe_scatter_features_kernel<FPTYPE><<<total, t>>>(
        features, unsort_idx, sorted_features, N, topk, feat_dim);
    DPErrcheck(cudaGetLastError());
  }

  // 7. Scatter weights.
  {
    const int blocks = (total + TPB - 1) / TPB;
    moe_scatter_weights_kernel<FPTYPE><<<blocks, TPB>>>(
        topk_weights, unsort_idx, sorted_weights, total);
    DPErrcheck(cudaGetLastError());
  }
}

template <typename FPTYPE>
void moe_topk_expand_sort_backward_gpu(FPTYPE* grad_features,
                                       FPTYPE* grad_topk_weights,
                                       const FPTYPE* grad_sorted_features,
                                       const FPTYPE* grad_sorted_weights,
                                       const int64_t* unsort_idx,
                                       const int N,
                                       const int topk,
                                       const int feat_dim) {
  if (N == 0 || topk == 0) {
    return;
  }
  const int total = N * topk;

  // 1. Gather-sum into grad_features.
  {
    dim3 block(TPB, 1, 1);
    dim3 grid(N, (feat_dim + TPB - 1) / TPB, 1);
    moe_grad_features_kernel<FPTYPE><<<grid, block>>>(
        grad_sorted_features, unsort_idx, grad_features, N, topk, feat_dim);
    DPErrcheck(cudaGetLastError());
  }

  // 2. Gather weights.
  {
    const int blocks = (total + TPB - 1) / TPB;
    moe_grad_weights_kernel<FPTYPE><<<blocks, TPB>>>(
        grad_sorted_weights, unsort_idx, grad_topk_weights, total);
    DPErrcheck(cudaGetLastError());
  }
}

template <typename FPTYPE>
void moe_topk_expand_sort_double_backward_gpu(
    FPTYPE* ggrad_sorted_features,
    FPTYPE* ggrad_sorted_weights,
    const FPTYPE* ggrad_features,
    const FPTYPE* ggrad_topk_weights,
    const int64_t* unsort_idx,
    const int N,
    const int topk,
    const int feat_dim) {
  if (N == 0 || topk == 0) {
    return;
  }
  const int total = N * topk;

  // Same scatter pattern as the forward (one-to-one permutation).
  {
    const int threads = (feat_dim < TPB) ? ((feat_dim + 31) & ~31) : TPB;
    const int t = threads > 0 ? threads : 32;
    moe_scatter_features_kernel<FPTYPE><<<total, t>>>(
        ggrad_features, unsort_idx, ggrad_sorted_features, N, topk, feat_dim);
    DPErrcheck(cudaGetLastError());
  }
  {
    const int blocks = (total + TPB - 1) / TPB;
    moe_scatter_weights_kernel<FPTYPE><<<blocks, TPB>>>(
        ggrad_topk_weights, unsort_idx, ggrad_sorted_weights, total);
    DPErrcheck(cudaGetLastError());
  }
}

// ----- Explicit instantiations -----
template void moe_topk_expand_sort_forward_gpu<float>(
    float*, int64_t*, float*, int64_t*, int64_t*, int*, int*, int*,
    const float*, const int64_t*, const float*, int, int, int, int, int, int);
template void moe_topk_expand_sort_forward_gpu<double>(
    double*, int64_t*, double*, int64_t*, int64_t*, int*, int*, int*,
    const double*, const int64_t*, const double*, int, int, int, int, int, int);

template void moe_topk_expand_sort_backward_gpu<float>(
    float*, float*, const float*, const float*, const int64_t*, int, int, int);
template void moe_topk_expand_sort_backward_gpu<double>(
    double*, double*, const double*, const double*, const int64_t*, int, int,
    int);

template void moe_topk_expand_sort_double_backward_gpu<float>(
    float*, float*, const float*, const float*, const int64_t*, int, int, int);
template void moe_topk_expand_sort_double_backward_gpu<double>(
    double*, double*, const double*, const double*, const int64_t*, int, int,
    int);

}  // namespace deepmd
