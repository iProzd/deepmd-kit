// SPDX-License-Identifier: LGPL-3.0-or-later
//
// PyTorch op wrapping the fused MoE pack-for-dispatch CUDA kernel.
// Drop-in replacement for `MoEPacker.pack_for_dispatch` in
// `deepmd/pt/model/network/moe_packer.py`.
//
// Two autograd::Function classes are used so that backward itself is
// differentiable (DeePMD needs second-order gradients through MoE for the
// force-loss path).
//
// Forward call signature (Python):
//   packed = torch.ops.deepmd.moe_pack_for_dispatch(
//       node_sorted, edge_sorted, angle_sorted,
//       node_in_offset, edge_in_offset, angle_in_offset,    # int64 [ep+1] on GPU
//       node_out_offset, edge_out_offset, angle_out_offset, # int64 [ep]   on GPU
//       total_packed_rows, ep_size,
//       D_packed, edge_concat, angle_concat)

#include <torch/torch.h>

#include <vector>

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "device.h"
#include "moe_pack_for_dispatch.h"
#endif

namespace {

class MoEPackForDispatchGradOp;  // forward decl

class MoEPackForDispatchOp
    : public torch::autograd::Function<MoEPackForDispatchOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& node_sorted,
      const torch::Tensor& edge_sorted,
      const torch::Tensor& angle_sorted,
      const torch::Tensor& node_in_offset,
      const torch::Tensor& edge_in_offset,
      const torch::Tensor& angle_in_offset,
      const torch::Tensor& node_out_offset,
      const torch::Tensor& edge_out_offset,
      const torch::Tensor& angle_out_offset,
      int64_t total_packed_rows,
      int64_t ep_size,
      int64_t D_packed,
      int64_t edge_concat,
      int64_t angle_concat) {
    TORCH_CHECK(node_sorted.is_cuda(), "node_sorted must be CUDA");
    TORCH_CHECK(edge_sorted.is_cuda(), "edge_sorted must be CUDA");
    TORCH_CHECK(angle_sorted.is_cuda(), "angle_sorted must be CUDA");
    TORCH_CHECK(node_sorted.dim() == 2 && edge_sorted.dim() == 2 &&
                    angle_sorted.dim() == 2,
                "all sorted inputs must be 2D");
    const auto dtype = node_sorted.scalar_type();
    TORCH_CHECK(edge_sorted.scalar_type() == dtype &&
                    angle_sorted.scalar_type() == dtype,
                "node/edge/angle dtype must match");
    TORCH_CHECK(node_in_offset.scalar_type() == torch::kLong &&
                    edge_in_offset.scalar_type() == torch::kLong &&
                    angle_in_offset.scalar_type() == torch::kLong &&
                    node_out_offset.scalar_type() == torch::kLong &&
                    edge_out_offset.scalar_type() == torch::kLong &&
                    angle_out_offset.scalar_type() == torch::kLong,
                "offsets must be int64");

    const auto float_options =
        torch::TensorOptions().dtype(dtype).device(node_sorted.device());
    torch::Tensor packed =
        torch::empty({total_packed_rows, D_packed}, float_options);

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
    auto ns = node_sorted.contiguous();
    auto es = edge_sorted.contiguous();
    auto as_ = angle_sorted.contiguous();

    if (dtype == torch::kDouble) {
      deepmd::moe_pack_for_dispatch_forward_gpu<double>(
          packed.data_ptr<double>(),
          ns.data_ptr<double>(), es.data_ptr<double>(), as_.data_ptr<double>(),
          node_in_offset.data_ptr<int64_t>(),
          edge_in_offset.data_ptr<int64_t>(),
          angle_in_offset.data_ptr<int64_t>(),
          node_out_offset.data_ptr<int64_t>(),
          edge_out_offset.data_ptr<int64_t>(),
          angle_out_offset.data_ptr<int64_t>(),
          static_cast<int>(total_packed_rows),
          static_cast<int>(ep_size),
          static_cast<int>(ns.size(0)),
          static_cast<int>(es.size(0)),
          static_cast<int>(as_.size(0)),
          static_cast<int>(ns.size(1)),
          static_cast<int>(es.size(1)),
          static_cast<int>(as_.size(1)),
          static_cast<int>(D_packed),
          static_cast<int>(edge_concat),
          static_cast<int>(angle_concat));
    } else if (dtype == torch::kFloat) {
      deepmd::moe_pack_for_dispatch_forward_gpu<float>(
          packed.data_ptr<float>(),
          ns.data_ptr<float>(), es.data_ptr<float>(), as_.data_ptr<float>(),
          node_in_offset.data_ptr<int64_t>(),
          edge_in_offset.data_ptr<int64_t>(),
          angle_in_offset.data_ptr<int64_t>(),
          node_out_offset.data_ptr<int64_t>(),
          edge_out_offset.data_ptr<int64_t>(),
          angle_out_offset.data_ptr<int64_t>(),
          static_cast<int>(total_packed_rows),
          static_cast<int>(ep_size),
          static_cast<int>(ns.size(0)),
          static_cast<int>(es.size(0)),
          static_cast<int>(as_.size(0)),
          static_cast<int>(ns.size(1)),
          static_cast<int>(es.size(1)),
          static_cast<int>(as_.size(1)),
          static_cast<int>(D_packed),
          static_cast<int>(edge_concat),
          static_cast<int>(angle_concat));
    } else {
      TORCH_CHECK(false,
                  "moe_pack_for_dispatch: only float32/float64 supported");
    }
#else
    TORCH_CHECK(false,
                "moe_pack_for_dispatch: built without CUDA/ROCm support");
#endif

    ctx->save_for_backward({node_in_offset, edge_in_offset, angle_in_offset,
                            node_out_offset, edge_out_offset,
                            angle_out_offset});
    ctx->saved_data["ep_size"] = ep_size;
    ctx->saved_data["D_packed"] = D_packed;
    ctx->saved_data["edge_concat"] = edge_concat;
    ctx->saved_data["angle_concat"] = angle_concat;
    ctx->saved_data["N_node"] = static_cast<int64_t>(node_sorted.size(0));
    ctx->saved_data["N_edge"] = static_cast<int64_t>(edge_sorted.size(0));
    ctx->saved_data["N_angle"] = static_cast<int64_t>(angle_sorted.size(0));
    ctx->saved_data["D_node"] = static_cast<int64_t>(node_sorted.size(1));
    ctx->saved_data["D_edge"] = static_cast<int64_t>(edge_sorted.size(1));
    ctx->saved_data["D_angle"] = static_cast<int64_t>(angle_sorted.size(1));

    return {packed};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output);
};

class MoEPackForDispatchGradOp
    : public torch::autograd::Function<MoEPackForDispatchGradOp> {
 public:
  // forward of grad-op == backward of the forward op.
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& grad_packed,
      const torch::Tensor& node_in_offset,
      const torch::Tensor& edge_in_offset,
      const torch::Tensor& angle_in_offset,
      const torch::Tensor& node_out_offset,
      const torch::Tensor& edge_out_offset,
      const torch::Tensor& angle_out_offset,
      int64_t ep_size,
      int64_t N_node,
      int64_t N_edge,
      int64_t N_angle,
      int64_t D_node,
      int64_t D_edge,
      int64_t D_angle,
      int64_t D_packed,
      int64_t edge_concat,
      int64_t angle_concat) {
    const auto dtype = grad_packed.scalar_type();
    const auto float_options =
        torch::TensorOptions().dtype(dtype).device(grad_packed.device());

    torch::Tensor grad_node_sorted = torch::empty({N_node, D_node}, float_options);
    torch::Tensor grad_edge_sorted = torch::empty({N_edge, D_edge}, float_options);
    torch::Tensor grad_angle_sorted = torch::empty({N_angle, D_angle}, float_options);

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
    auto gp = grad_packed.contiguous();

    if (dtype == torch::kDouble) {
      deepmd::moe_pack_for_dispatch_backward_gpu<double>(
          grad_node_sorted.data_ptr<double>(),
          grad_edge_sorted.data_ptr<double>(),
          grad_angle_sorted.data_ptr<double>(),
          gp.data_ptr<double>(),
          node_in_offset.data_ptr<int64_t>(),
          edge_in_offset.data_ptr<int64_t>(),
          angle_in_offset.data_ptr<int64_t>(),
          node_out_offset.data_ptr<int64_t>(),
          edge_out_offset.data_ptr<int64_t>(),
          angle_out_offset.data_ptr<int64_t>(),
          static_cast<int>(ep_size),
          static_cast<int>(N_node), static_cast<int>(N_edge),
          static_cast<int>(N_angle),
          static_cast<int>(D_node), static_cast<int>(D_edge),
          static_cast<int>(D_angle), static_cast<int>(D_packed),
          static_cast<int>(edge_concat), static_cast<int>(angle_concat));
    } else if (dtype == torch::kFloat) {
      deepmd::moe_pack_for_dispatch_backward_gpu<float>(
          grad_node_sorted.data_ptr<float>(),
          grad_edge_sorted.data_ptr<float>(),
          grad_angle_sorted.data_ptr<float>(),
          gp.data_ptr<float>(),
          node_in_offset.data_ptr<int64_t>(),
          edge_in_offset.data_ptr<int64_t>(),
          angle_in_offset.data_ptr<int64_t>(),
          node_out_offset.data_ptr<int64_t>(),
          edge_out_offset.data_ptr<int64_t>(),
          angle_out_offset.data_ptr<int64_t>(),
          static_cast<int>(ep_size),
          static_cast<int>(N_node), static_cast<int>(N_edge),
          static_cast<int>(N_angle),
          static_cast<int>(D_node), static_cast<int>(D_edge),
          static_cast<int>(D_angle), static_cast<int>(D_packed),
          static_cast<int>(edge_concat), static_cast<int>(angle_concat));
    } else {
      TORCH_CHECK(false,
                  "moe_pack_for_dispatch: only float32/float64 supported");
    }
#else
    TORCH_CHECK(false,
                "moe_pack_for_dispatch: built without CUDA/ROCm support");
#endif

    ctx->save_for_backward({node_in_offset, edge_in_offset, angle_in_offset,
                            node_out_offset, edge_out_offset,
                            angle_out_offset});
    ctx->saved_data["ep_size"] = ep_size;
    ctx->saved_data["N_node"] = N_node;
    ctx->saved_data["N_edge"] = N_edge;
    ctx->saved_data["N_angle"] = N_angle;
    ctx->saved_data["D_node"] = D_node;
    ctx->saved_data["D_edge"] = D_edge;
    ctx->saved_data["D_angle"] = D_angle;
    ctx->saved_data["D_packed"] = D_packed;
    ctx->saved_data["edge_concat"] = edge_concat;
    ctx->saved_data["angle_concat"] = angle_concat;
    ctx->saved_data["total_packed_rows"] =
        static_cast<int64_t>(grad_packed.size(0));

    return {grad_node_sorted, grad_edge_sorted, grad_angle_sorted};
  }

  // backward of grad-op == double backward of the forward op.
  // The backward of a linear gather is the corresponding scatter, i.e. the
  // *forward* operation on the upstream double-grads.
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    auto saved = ctx->get_saved_variables();
    torch::Tensor node_in_offset = saved[0];
    torch::Tensor edge_in_offset = saved[1];
    torch::Tensor angle_in_offset = saved[2];
    torch::Tensor node_out_offset = saved[3];
    torch::Tensor edge_out_offset = saved[4];
    torch::Tensor angle_out_offset = saved[5];

    const int64_t ep_size = ctx->saved_data["ep_size"].toInt();
    const int64_t N_node = ctx->saved_data["N_node"].toInt();
    const int64_t N_edge = ctx->saved_data["N_edge"].toInt();
    const int64_t N_angle = ctx->saved_data["N_angle"].toInt();
    const int64_t D_node = ctx->saved_data["D_node"].toInt();
    const int64_t D_edge = ctx->saved_data["D_edge"].toInt();
    const int64_t D_angle = ctx->saved_data["D_angle"].toInt();
    const int64_t D_packed = ctx->saved_data["D_packed"].toInt();
    const int64_t edge_concat = ctx->saved_data["edge_concat"].toInt();
    const int64_t angle_concat = ctx->saved_data["angle_concat"].toInt();
    const int64_t total_packed_rows =
        ctx->saved_data["total_packed_rows"].toInt();

    // grad_output[0] : ggrad_node_sorted
    // grad_output[1] : ggrad_edge_sorted
    // grad_output[2] : ggrad_angle_sorted
    // Any may be undefined if the downstream consumer didn't touch the
    // corresponding output.  Replace with zeros to keep the kernel happy.
    auto ggn = grad_output[0];
    auto gge = grad_output[1];
    auto gga = grad_output[2];

    auto pick_dtype = [&]() {
      if (ggn.defined()) return ggn.scalar_type();
      if (gge.defined()) return gge.scalar_type();
      if (gga.defined()) return gga.scalar_type();
      return torch::kFloat;
    };
    auto pick_device = [&]() {
      if (ggn.defined()) return ggn.device();
      if (gge.defined()) return gge.device();
      if (gga.defined()) return gga.device();
      return node_in_offset.device();
    };
    const auto dtype = pick_dtype();
    const auto device = pick_device();
    const auto float_options =
        torch::TensorOptions().dtype(dtype).device(device);
    if (!ggn.defined())
      ggn = torch::zeros({N_node, D_node}, float_options);
    if (!gge.defined())
      gge = torch::zeros({N_edge, D_edge}, float_options);
    if (!gga.defined())
      gga = torch::zeros({N_angle, D_angle}, float_options);
    ggn = ggn.contiguous();
    gge = gge.contiguous();
    gga = gga.contiguous();

    torch::Tensor ggrad_packed =
        torch::empty({total_packed_rows, D_packed}, float_options);

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
    if (dtype == torch::kDouble) {
      deepmd::moe_pack_for_dispatch_forward_gpu<double>(
          ggrad_packed.data_ptr<double>(),
          ggn.data_ptr<double>(), gge.data_ptr<double>(), gga.data_ptr<double>(),
          node_in_offset.data_ptr<int64_t>(),
          edge_in_offset.data_ptr<int64_t>(),
          angle_in_offset.data_ptr<int64_t>(),
          node_out_offset.data_ptr<int64_t>(),
          edge_out_offset.data_ptr<int64_t>(),
          angle_out_offset.data_ptr<int64_t>(),
          static_cast<int>(total_packed_rows),
          static_cast<int>(ep_size),
          static_cast<int>(N_node), static_cast<int>(N_edge),
          static_cast<int>(N_angle),
          static_cast<int>(D_node), static_cast<int>(D_edge),
          static_cast<int>(D_angle), static_cast<int>(D_packed),
          static_cast<int>(edge_concat), static_cast<int>(angle_concat));
    } else if (dtype == torch::kFloat) {
      deepmd::moe_pack_for_dispatch_forward_gpu<float>(
          ggrad_packed.data_ptr<float>(),
          ggn.data_ptr<float>(), gge.data_ptr<float>(), gga.data_ptr<float>(),
          node_in_offset.data_ptr<int64_t>(),
          edge_in_offset.data_ptr<int64_t>(),
          angle_in_offset.data_ptr<int64_t>(),
          node_out_offset.data_ptr<int64_t>(),
          edge_out_offset.data_ptr<int64_t>(),
          angle_out_offset.data_ptr<int64_t>(),
          static_cast<int>(total_packed_rows),
          static_cast<int>(ep_size),
          static_cast<int>(N_node), static_cast<int>(N_edge),
          static_cast<int>(N_angle),
          static_cast<int>(D_node), static_cast<int>(D_edge),
          static_cast<int>(D_angle), static_cast<int>(D_packed),
          static_cast<int>(edge_concat), static_cast<int>(angle_concat));
    } else {
      TORCH_CHECK(false,
                  "moe_pack_for_dispatch: only float32/float64 supported");
    }
#endif

    // forward(grad_packed, node_in_offset, edge_in_offset, angle_in_offset,
    //         node_out_offset, edge_out_offset, angle_out_offset,
    //         ep_size, N_node, N_edge, N_angle,
    //         D_node, D_edge, D_angle, D_packed, edge_concat, angle_concat)
    return {ggrad_packed, at::Tensor(), at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor()};
  }
};

torch::autograd::variable_list MoEPackForDispatchOp::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_output) {
  auto saved = ctx->get_saved_variables();
  torch::Tensor node_in_offset = saved[0];
  torch::Tensor edge_in_offset = saved[1];
  torch::Tensor angle_in_offset = saved[2];
  torch::Tensor node_out_offset = saved[3];
  torch::Tensor edge_out_offset = saved[4];
  torch::Tensor angle_out_offset = saved[5];

  const int64_t ep_size = ctx->saved_data["ep_size"].toInt();
  const int64_t D_packed = ctx->saved_data["D_packed"].toInt();
  const int64_t edge_concat = ctx->saved_data["edge_concat"].toInt();
  const int64_t angle_concat = ctx->saved_data["angle_concat"].toInt();
  const int64_t N_node = ctx->saved_data["N_node"].toInt();
  const int64_t N_edge = ctx->saved_data["N_edge"].toInt();
  const int64_t N_angle = ctx->saved_data["N_angle"].toInt();
  const int64_t D_node = ctx->saved_data["D_node"].toInt();
  const int64_t D_edge = ctx->saved_data["D_edge"].toInt();
  const int64_t D_angle = ctx->saved_data["D_angle"].toInt();

  torch::Tensor grad_packed = grad_output[0];
  if (!grad_packed.defined()) {
    // packed was returned but never used downstream — nothing to back-prop.
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor()};
  }
  grad_packed = grad_packed.contiguous();

  // Call the sub-Function so backward is differentiable (double backward
  // works through MoEPackForDispatchGradOp::backward).
  auto outs = MoEPackForDispatchGradOp::apply(
      grad_packed, node_in_offset, edge_in_offset, angle_in_offset,
      node_out_offset, edge_out_offset, angle_out_offset,
      ep_size, N_node, N_edge, N_angle, D_node, D_edge, D_angle, D_packed,
      edge_concat, angle_concat);

  // forward signature was:
  //   (node_sorted, edge_sorted, angle_sorted,
  //    node_in_offset, edge_in_offset, angle_in_offset,
  //    node_out_offset, edge_out_offset, angle_out_offset,
  //    total_packed_rows, ep_size, D_packed, edge_concat, angle_concat)
  return {outs[0], outs[1], outs[2],
          at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor()};
}

torch::Tensor moe_pack_for_dispatch(const torch::Tensor& node_sorted,
                                     const torch::Tensor& edge_sorted,
                                     const torch::Tensor& angle_sorted,
                                     const torch::Tensor& node_in_offset,
                                     const torch::Tensor& edge_in_offset,
                                     const torch::Tensor& angle_in_offset,
                                     const torch::Tensor& node_out_offset,
                                     const torch::Tensor& edge_out_offset,
                                     const torch::Tensor& angle_out_offset,
                                     int64_t total_packed_rows,
                                     int64_t ep_size,
                                     int64_t D_packed,
                                     int64_t edge_concat,
                                     int64_t angle_concat) {
  auto outs = MoEPackForDispatchOp::apply(
      node_sorted, edge_sorted, angle_sorted,
      node_in_offset, edge_in_offset, angle_in_offset,
      node_out_offset, edge_out_offset, angle_out_offset,
      total_packed_rows, ep_size, D_packed, edge_concat, angle_concat);
  return outs[0];
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def("moe_pack_for_dispatch", moe_pack_for_dispatch);
}
