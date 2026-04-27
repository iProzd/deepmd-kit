// SPDX-License-Identifier: LGPL-3.0-or-later
//
// PyTorch op wrapping the fused MoE topk-expand-sort CUDA kernel.
// Drop-in replacement for the Python `_topk_expand_sort` reference in
// `deepmd/pt/model/network/moe_layer.py`.
//
// Two autograd::Function classes are used so that the backward itself
// is differentiable (the DeePMD model needs second-order gradients
// through MoE for the force-loss path).
//
// Forward call signature (Python):
//   sorted_features, sorted_expert_ids, sorted_weights, unsort_idx, gpu_counts
//     = torch.ops.deepmd.moe_topk_expand_sort(
//           features, topk_indices, topk_weights,
//           experts_per_gpu, n_routing_experts, ep_size)

#include <torch/torch.h>

#include <vector>

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "device.h"
#include "moe_topk_expand_sort.h"
#endif

namespace {

class MoeTopkExpandSortGradOp;  // forward decl

class MoeTopkExpandSortOp
    : public torch::autograd::Function<MoeTopkExpandSortOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& features,
      const torch::Tensor& topk_indices,
      const torch::Tensor& topk_weights,
      int64_t experts_per_gpu,
      int64_t n_routing_experts,
      int64_t ep_size) {
    TORCH_CHECK(features.is_cuda(), "features must be on CUDA device");
    TORCH_CHECK(topk_indices.is_cuda(), "topk_indices must be on CUDA device");
    TORCH_CHECK(topk_weights.is_cuda(), "topk_weights must be on CUDA device");
    TORCH_CHECK(features.dim() == 2, "features must be 2D [N, feat_dim]");
    TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be 2D [N, topk]");
    TORCH_CHECK(topk_weights.dim() == 2, "topk_weights must be 2D [N, topk]");
    TORCH_CHECK(topk_indices.scalar_type() == torch::kLong,
                "topk_indices must be int64");
    TORCH_CHECK(features.scalar_type() == topk_weights.scalar_type(),
                "features and topk_weights must share dtype");

    const int64_t N = features.size(0);
    const int64_t feat_dim = features.size(1);
    const int64_t topk = topk_indices.size(1);
    TORCH_CHECK(topk_indices.size(0) == N, "topk_indices first dim mismatch");
    TORCH_CHECK(topk_weights.size(0) == N, "topk_weights first dim mismatch");
    TORCH_CHECK(topk_weights.size(1) == topk, "topk_weights second dim mismatch");

    const auto float_options =
        torch::TensorOptions().dtype(features.dtype()).device(features.device());
    const auto int_options =
        torch::TensorOptions().dtype(torch::kLong).device(features.device());
    const auto i32_options =
        torch::TensorOptions().dtype(torch::kInt32).device(features.device());

    torch::Tensor sorted_features =
        torch::empty({N * topk, feat_dim}, float_options);
    torch::Tensor sorted_expert_ids = torch::empty({N * topk}, int_options);
    torch::Tensor sorted_weights = torch::empty({N * topk}, float_options);
    torch::Tensor unsort_idx = torch::empty({N * topk}, int_options);
    torch::Tensor gpu_counts = torch::empty({ep_size}, int_options);

    // Workspaces (small, transient).
    torch::Tensor d_hist = torch::empty({n_routing_experts}, i32_options);
    torch::Tensor d_offsets = torch::empty({n_routing_experts + 1}, i32_options);
    torch::Tensor d_running = torch::empty({n_routing_experts}, i32_options);

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
    auto features_c = features.contiguous();
    auto topk_indices_c = topk_indices.contiguous();
    auto topk_weights_c = topk_weights.contiguous();

    if (features.scalar_type() == torch::kDouble) {
      deepmd::moe_topk_expand_sort_forward_gpu<double>(
          sorted_features.data_ptr<double>(),
          sorted_expert_ids.data_ptr<int64_t>(),
          sorted_weights.data_ptr<double>(),
          unsort_idx.data_ptr<int64_t>(),
          gpu_counts.data_ptr<int64_t>(),
          d_hist.data_ptr<int>(),
          d_offsets.data_ptr<int>(),
          d_running.data_ptr<int>(),
          features_c.data_ptr<double>(),
          topk_indices_c.data_ptr<int64_t>(),
          topk_weights_c.data_ptr<double>(),
          static_cast<int>(N),
          static_cast<int>(topk),
          static_cast<int>(feat_dim),
          static_cast<int>(n_routing_experts),
          static_cast<int>(experts_per_gpu),
          static_cast<int>(ep_size));
    } else if (features.scalar_type() == torch::kFloat) {
      deepmd::moe_topk_expand_sort_forward_gpu<float>(
          sorted_features.data_ptr<float>(),
          sorted_expert_ids.data_ptr<int64_t>(),
          sorted_weights.data_ptr<float>(),
          unsort_idx.data_ptr<int64_t>(),
          gpu_counts.data_ptr<int64_t>(),
          d_hist.data_ptr<int>(),
          d_offsets.data_ptr<int>(),
          d_running.data_ptr<int>(),
          features_c.data_ptr<float>(),
          topk_indices_c.data_ptr<int64_t>(),
          topk_weights_c.data_ptr<float>(),
          static_cast<int>(N),
          static_cast<int>(topk),
          static_cast<int>(feat_dim),
          static_cast<int>(n_routing_experts),
          static_cast<int>(experts_per_gpu),
          static_cast<int>(ep_size));
    } else {
      TORCH_CHECK(false, "moe_topk_expand_sort: only float32/float64 supported");
    }
#else
    TORCH_CHECK(false,
                "moe_topk_expand_sort: built without CUDA/ROCm support");
#endif

    // Save tensors and shapes for backward.  Only unsort_idx is needed
    // by the gradient computation; saving features/topk_weights would
    // be wasteful since the backward only cares about the permutation.
    ctx->save_for_backward({unsort_idx});
    ctx->saved_data["N"] = N;
    ctx->saved_data["topk"] = topk;
    ctx->saved_data["feat_dim"] = feat_dim;

    return {sorted_features, sorted_expert_ids, sorted_weights, unsort_idx,
            gpu_counts};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output);
};

class MoeTopkExpandSortGradOp
    : public torch::autograd::Function<MoeTopkExpandSortGradOp> {
 public:
  // forward of grad-op == backward of the forward op.
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& grad_sorted_features,
      const torch::Tensor& grad_sorted_weights,
      const torch::Tensor& unsort_idx,
      int64_t N,
      int64_t topk,
      int64_t feat_dim) {
    const auto float_options = torch::TensorOptions()
                                   .dtype(grad_sorted_features.dtype())
                                   .device(grad_sorted_features.device());
    torch::Tensor grad_features = torch::empty({N, feat_dim}, float_options);
    torch::Tensor grad_topk_weights = torch::empty({N, topk}, float_options);

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
    auto gsf = grad_sorted_features.contiguous();
    auto gsw = grad_sorted_weights.contiguous();

    if (grad_sorted_features.scalar_type() == torch::kDouble) {
      deepmd::moe_topk_expand_sort_backward_gpu<double>(
          grad_features.data_ptr<double>(),
          grad_topk_weights.data_ptr<double>(),
          gsf.data_ptr<double>(),
          gsw.data_ptr<double>(),
          unsort_idx.data_ptr<int64_t>(),
          static_cast<int>(N),
          static_cast<int>(topk),
          static_cast<int>(feat_dim));
    } else if (grad_sorted_features.scalar_type() == torch::kFloat) {
      deepmd::moe_topk_expand_sort_backward_gpu<float>(
          grad_features.data_ptr<float>(),
          grad_topk_weights.data_ptr<float>(),
          gsf.data_ptr<float>(),
          gsw.data_ptr<float>(),
          unsort_idx.data_ptr<int64_t>(),
          static_cast<int>(N),
          static_cast<int>(topk),
          static_cast<int>(feat_dim));
    } else {
      TORCH_CHECK(false,
                  "moe_topk_expand_sort: only float32/float64 supported");
    }
#else
    TORCH_CHECK(false,
                "moe_topk_expand_sort: built without CUDA/ROCm support");
#endif

    ctx->save_for_backward({unsort_idx});
    ctx->saved_data["N"] = N;
    ctx->saved_data["topk"] = topk;
    ctx->saved_data["feat_dim"] = feat_dim;

    return {grad_features, grad_topk_weights};
  }

  // backward of grad-op == double backward of the forward op.
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    auto saved = ctx->get_saved_variables();
    torch::Tensor unsort_idx = saved[0];
    const int64_t N = ctx->saved_data["N"].toInt();
    const int64_t topk = ctx->saved_data["topk"].toInt();
    const int64_t feat_dim = ctx->saved_data["feat_dim"].toInt();

    torch::Tensor ggrad_features = grad_output[0].contiguous();
    torch::Tensor ggrad_topk_weights = grad_output[1].contiguous();

    const auto float_options = torch::TensorOptions()
                                   .dtype(ggrad_features.dtype())
                                   .device(ggrad_features.device());
    torch::Tensor ggrad_sorted_features =
        torch::empty({N * topk, feat_dim}, float_options);
    torch::Tensor ggrad_sorted_weights =
        torch::empty({N * topk}, float_options);

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
    if (ggrad_features.scalar_type() == torch::kDouble) {
      deepmd::moe_topk_expand_sort_double_backward_gpu<double>(
          ggrad_sorted_features.data_ptr<double>(),
          ggrad_sorted_weights.data_ptr<double>(),
          ggrad_features.data_ptr<double>(),
          ggrad_topk_weights.data_ptr<double>(),
          unsort_idx.data_ptr<int64_t>(),
          static_cast<int>(N),
          static_cast<int>(topk),
          static_cast<int>(feat_dim));
    } else if (ggrad_features.scalar_type() == torch::kFloat) {
      deepmd::moe_topk_expand_sort_double_backward_gpu<float>(
          ggrad_sorted_features.data_ptr<float>(),
          ggrad_sorted_weights.data_ptr<float>(),
          ggrad_features.data_ptr<float>(),
          ggrad_topk_weights.data_ptr<float>(),
          unsort_idx.data_ptr<int64_t>(),
          static_cast<int>(N),
          static_cast<int>(topk),
          static_cast<int>(feat_dim));
    } else {
      TORCH_CHECK(false,
                  "moe_topk_expand_sort: only float32/float64 supported");
    }
#else
    TORCH_CHECK(false,
                "moe_topk_expand_sort: built without CUDA/ROCm support");
#endif

    // Inputs to forward(grad_sorted_features, grad_sorted_weights,
    //                   unsort_idx, N, topk, feat_dim).
    return {ggrad_sorted_features, ggrad_sorted_weights, at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

torch::autograd::variable_list MoeTopkExpandSortOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_output) {
  auto saved = ctx->get_saved_variables();
  torch::Tensor unsort_idx = saved[0];
  const int64_t N = ctx->saved_data["N"].toInt();
  const int64_t topk = ctx->saved_data["topk"].toInt();
  const int64_t feat_dim = ctx->saved_data["feat_dim"].toInt();

  // grad_output[0] : grad of sorted_features
  // grad_output[1] : grad of sorted_expert_ids (non-diff, ignored)
  // grad_output[2] : grad of sorted_weights
  // grad_output[3] : grad of unsort_idx (non-diff, ignored)
  // grad_output[4] : grad of gpu_counts (non-diff, ignored)
  torch::Tensor grad_sorted_features = grad_output[0];
  torch::Tensor grad_sorted_weights = grad_output[2];

  // If a downstream consumer didn't touch sorted_features/sorted_weights,
  // PyTorch may pass an undefined tensor. Replace with zeros so the
  // sub-Function sees a real tensor.
  if (!grad_sorted_features.defined()) {
    const auto float_options = torch::TensorOptions()
                                   .dtype(unsort_idx.options().dtype())
                                   .device(unsort_idx.device());
    // dtype guess from saved_data is unsafe; fall back to the other grad.
    if (grad_sorted_weights.defined()) {
      grad_sorted_features = torch::zeros(
          {N * topk, feat_dim},
          torch::TensorOptions()
              .dtype(grad_sorted_weights.dtype())
              .device(grad_sorted_weights.device()));
    } else {
      // Both undefined: nothing to do.
      return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
              at::Tensor(), at::Tensor()};
    }
  }
  if (!grad_sorted_weights.defined()) {
    grad_sorted_weights = torch::zeros(
        {N * topk}, torch::TensorOptions()
                        .dtype(grad_sorted_features.dtype())
                        .device(grad_sorted_features.device()));
  }

  // Call the sub-Function so backward is differentiable (double-backward
  // works through MoeTopkExpandSortGradOp::backward).
  auto outs = MoeTopkExpandSortGradOp::apply(
      grad_sorted_features, grad_sorted_weights, unsort_idx, N, topk, feat_dim);

  torch::Tensor grad_features = outs[0];
  torch::Tensor grad_topk_weights = outs[1];

  // forward signature was: features, topk_indices, topk_weights,
  //                        experts_per_gpu, n_routing_experts, ep_size
  return {grad_features, at::Tensor(), grad_topk_weights, at::Tensor(),
          at::Tensor(), at::Tensor()};
}

// Public function bound to torch ops.
std::vector<torch::Tensor> moe_topk_expand_sort(
    const torch::Tensor& features,
    const torch::Tensor& topk_indices,
    const torch::Tensor& topk_weights,
    int64_t experts_per_gpu,
    int64_t n_routing_experts,
    int64_t ep_size) {
  auto outs = MoeTopkExpandSortOp::apply(features, topk_indices, topk_weights,
                                          experts_per_gpu, n_routing_experts,
                                          ep_size);
  return std::vector<torch::Tensor>(outs.begin(), outs.end());
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def("moe_topk_expand_sort", moe_topk_expand_sort);
}
