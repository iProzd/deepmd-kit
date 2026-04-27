# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the fused CUDA MoE topk-expand-sort op.

Covers:
  - forward correctness vs. PyTorch reference (float32 + float64)
  - backward correctness vs. PyTorch reference autograd
  - double backward (energy -> force -> force-loss path)
  - torch.autograd.gradcheck (float64)
  - torch.autograd.gradgradcheck (float64) — must pass before integration
  - empty-input and small-feat-dim edge cases
"""

from __future__ import annotations

import pytest
import torch

import deepmd.pt.cxx_op  # noqa: F401  registers torch.ops.deepmd


# ----------------------------------------------------------------------
# Reference implementation copied verbatim from moe_layer.py
# ----------------------------------------------------------------------
def _ref_topk_expand_sort(
    features: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    experts_per_gpu: int,
):
    N, topk = topk_indices.shape
    flat_indices = topk_indices.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    expanded = features.repeat_interleave(topk, dim=0)
    sort_idx = torch.argsort(flat_indices, stable=True)
    sorted_features = expanded[sort_idx]
    sorted_expert_ids = flat_indices[sort_idx]
    sorted_weights = flat_weights[sort_idx]
    unsort_idx = torch.empty_like(sort_idx)
    unsort_idx[sort_idx] = torch.arange(len(sort_idx), device=sort_idx.device)
    sorted_target_gpu = sorted_expert_ids // experts_per_gpu
    ep_size_inferred = (
        int(sorted_target_gpu.max().item()) + 1
        if len(sorted_target_gpu) > 0
        else 1
    )
    gpu_counts = torch.bincount(sorted_target_gpu, minlength=ep_size_inferred)
    return (
        sorted_features,
        sorted_expert_ids,
        sorted_weights,
        unsort_idx,
        gpu_counts.tolist(),
        ep_size_inferred,
    )


def _cuda_op(features, topk_indices, topk_weights, experts_per_gpu,
             n_routing_experts, ep_size):
    return torch.ops.deepmd.moe_topk_expand_sort(
        features, topk_indices, topk_weights,
        int(experts_per_gpu), int(n_routing_experts), int(ep_size),
    )


# ----------------------------------------------------------------------
# Helpers: invariants that must hold even without sort stability
# ----------------------------------------------------------------------
def _check_segment_invariant(
    cuda_sorted_features, cuda_sorted_weights, cuda_sorted_eids, cuda_unsort,
    ref_sorted_eids, features, topk_weights, atol=1e-6, rtol=1e-6,
):
    """Check that:
    1. Each expert's segment in sorted_expert_ids matches the reference
       (only ordering within a segment may differ).
    2. The full mapping unsort_idx is a valid permutation, and inverting
       it reconstructs the per-token features and weights.
    """
    # 1. expert-id histograms identical
    assert torch.equal(
        torch.bincount(cuda_sorted_eids), torch.bincount(ref_sorted_eids)
    ), "expert-id distribution differs from reference"

    # 2. sorted_eids are non-decreasing (groups are contiguous)
    if cuda_sorted_eids.numel() > 1:
        diffs = cuda_sorted_eids[1:] - cuda_sorted_eids[:-1]
        assert (diffs >= 0).all(), "sorted_expert_ids must be non-decreasing"

    # 3. unsort_idx is a permutation of [0, N*topk)
    n = cuda_unsort.numel()
    if n > 0:
        sorted_unsort, _ = cuda_unsort.sort()
        assert torch.equal(
            sorted_unsort, torch.arange(n, device=cuda_unsort.device)
        ), "unsort_idx is not a permutation"

    # 4. Inverse permutation reconstructs originals.
    N, topk = topk_weights.shape
    flat_weights = topk_weights.reshape(-1)
    expanded = features.repeat_interleave(topk, dim=0)
    torch.testing.assert_close(
        cuda_sorted_features[cuda_unsort], expanded, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        cuda_sorted_weights[cuda_unsort], flat_weights, atol=atol, rtol=rtol
    )


# ======================================================================
#                            Forward correctness
# ======================================================================
class TestForward:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "N,F,topk,epg,eps",
        [
            (100, 32, 4, 8, 4),       # typical
            (1024, 64, 8, 4, 8),       # larger batch
            (33, 28, 1, 4, 4),         # topk=1
            (7, 5, 6, 2, 3),           # weird shapes
            (256, 1, 3, 8, 2),         # feat_dim=1
        ],
    )
    def test_forward(self, dtype, N, F, topk, epg, eps):
        torch.manual_seed(42)
        n_re = epg * eps
        features = torch.randn(N, F, device="cuda", dtype=dtype)
        topk_indices = torch.randint(0, n_re, (N, topk), device="cuda")
        topk_weights = torch.randn(N, topk, device="cuda", dtype=dtype)

        sf, sei, sw, ui, gpc = _cuda_op(
            features, topk_indices, topk_weights, epg, n_re, eps
        )
        rf, rei, rw, rui, rgpc, _ = _ref_topk_expand_sort(
            features, topk_indices, topk_weights, epg
        )

        atol = 1e-6 if dtype == torch.float64 else 1e-5
        _check_segment_invariant(
            sf, sw, sei, ui, rei, features, topk_weights,
            atol=atol, rtol=atol,
        )

        # gpu_counts: pad reference up to eps
        ref_gpc = list(rgpc) + [0] * (eps - len(rgpc))
        assert gpc.tolist() == ref_gpc, f"gpu_counts mismatch: {gpc.tolist()} vs {ref_gpc}"

    def test_forward_empty(self):
        n_re, epg, eps = 8, 2, 4
        features = torch.empty(0, 16, device="cuda", dtype=torch.float32)
        topk_indices = torch.empty(0, 4, device="cuda", dtype=torch.long)
        topk_weights = torch.empty(0, 4, device="cuda", dtype=torch.float32)
        sf, sei, sw, ui, gpc = _cuda_op(
            features, topk_indices, topk_weights, epg, n_re, eps
        )
        assert sf.shape == (0, 16)
        assert sei.shape == (0,)
        assert sw.shape == (0,)
        assert ui.shape == (0,)
        assert gpc.tolist() == [0, 0, 0, 0]


# ======================================================================
#                          Backward correctness
# ======================================================================
class TestBackward:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_backward_features_and_weights(self, dtype):
        torch.manual_seed(0)
        N, F, topk, epg, eps = 64, 24, 3, 4, 4
        n_re = epg * eps
        features = torch.randn(N, F, device="cuda", dtype=dtype, requires_grad=True)
        topk_indices = torch.randint(0, n_re, (N, topk), device="cuda")
        topk_weights = torch.randn(N, topk, device="cuda", dtype=dtype, requires_grad=True)

        # Use the same indices for both reference and CUDA to make the
        # *outputs* directly comparable element-wise after we account for
        # the (potentially different) segment ordering.

        # CUDA path
        sf, sei, sw, ui, _ = _cuda_op(
            features, topk_indices, topk_weights, epg, n_re, eps
        )
        # Backward via a known-shape upstream gradient.  Grad via unsort:
        # simpler to verify by computing grads w.r.t. the reconstructed
        # tensor (sf[ui], sw[ui]) which equals the reference impl exactly.
        loss_cuda = (sf[ui] * 2.5).sum() + (sw[ui] * 1.7).sum()
        loss_cuda.backward()
        gf_cuda = features.grad.clone()
        gw_cuda = topk_weights.grad.clone()
        features.grad = None
        topk_weights.grad = None

        # Reference path: same loss formulation
        rf, rei, rw, rui, _, _ = _ref_topk_expand_sort(
            features, topk_indices, topk_weights, epg
        )
        loss_ref = (rf[rui] * 2.5).sum() + (rw[rui] * 1.7).sum()
        loss_ref.backward()
        gf_ref = features.grad.clone()
        gw_ref = topk_weights.grad.clone()

        atol = 1e-6 if dtype == torch.float64 else 1e-5
        torch.testing.assert_close(gf_cuda, gf_ref, atol=atol, rtol=atol)
        torch.testing.assert_close(gw_cuda, gw_ref, atol=atol, rtol=atol)

    def test_backward_with_sorted_loss(self):
        """Compare grads using the sorted tensors directly.

        The reference's sorted output may have a different *order* within
        each expert's segment than ours, so loss values differ.  But
        bilinear losses that are invariant to within-segment permutation
        (e.g. sum of all elements, or sum of weighted-by-expert-id) give
        identical gradients.
        """
        torch.manual_seed(1)
        N, F, topk, epg, eps = 50, 16, 4, 4, 2
        n_re = epg * eps
        dtype = torch.float64
        features = torch.randn(N, F, device="cuda", dtype=dtype, requires_grad=True)
        topk_indices = torch.randint(0, n_re, (N, topk), device="cuda")
        topk_weights = torch.randn(N, topk, device="cuda", dtype=dtype, requires_grad=True)

        # CUDA
        sf, sei, sw, ui, _ = _cuda_op(
            features, topk_indices, topk_weights, epg, n_re, eps
        )
        # Permutation-invariant loss: sum of all features, weighted by the
        # expert id (which is fixed regardless of within-segment ordering).
        coeff = sei.to(dtype) * 0.1 + 1.0  # [N*topk]
        loss_cuda = (sf * coeff.unsqueeze(-1)).sum() + (sw * coeff).sum()
        loss_cuda.backward()
        gf_cuda = features.grad.clone()
        gw_cuda = topk_weights.grad.clone()
        features.grad = None
        topk_weights.grad = None

        # Reference
        rf, rei, rw, rui, _, _ = _ref_topk_expand_sort(
            features, topk_indices, topk_weights, epg
        )
        coeff_r = rei.to(dtype) * 0.1 + 1.0
        loss_ref = (rf * coeff_r.unsqueeze(-1)).sum() + (rw * coeff_r).sum()
        loss_ref.backward()
        gf_ref = features.grad.clone()
        gw_ref = topk_weights.grad.clone()

        torch.testing.assert_close(gf_cuda, gf_ref, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(gw_cuda, gw_ref, atol=1e-9, rtol=1e-9)


# ======================================================================
#                       Double backward (DeePMD use case)
# ======================================================================
class TestDoubleBackward:
    """Simulate energy -> force -> force-loss -> param grad (DeePMD path).

    Need ``create_graph=True`` on the inner backward and require the same
    final gradients as the PyTorch reference.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_double_backward_features(self, dtype):
        torch.manual_seed(2)
        N, F, topk, epg, eps = 32, 12, 3, 4, 2
        n_re = epg * eps
        coords = torch.randn(N, F, device="cuda", dtype=dtype, requires_grad=True)
        topk_indices = torch.randint(0, n_re, (N, topk), device="cuda")
        topk_weights = torch.randn(N, topk, device="cuda", dtype=dtype)

        # A tiny "model": features = some_fn(coords), then run the op.
        weight_layer = torch.randn(F, F, device="cuda", dtype=dtype)

        def model_cuda():
            features = coords @ weight_layer  # depends on coords twice (2nd derivative non-trivial)
            sf, sei, sw, ui, _ = _cuda_op(
                features, topk_indices, topk_weights, epg, n_re, eps
            )
            # Permutation-invariant scalar
            coeff = (sei.to(dtype) * 0.1 + 1.0).unsqueeze(-1)
            energy = (sf * coeff).pow(2).sum()
            return energy

        def model_ref():
            features = coords @ weight_layer
            rf, rei, rw, rui, _, _ = _ref_topk_expand_sort(
                features, topk_indices, topk_weights, epg
            )
            coeff = (rei.to(dtype) * 0.1 + 1.0).unsqueeze(-1)
            return (rf * coeff).pow(2).sum()

        # CUDA path: energy -> force = d(energy)/d(coords), keep graph
        energy_c = model_cuda()
        force_c = torch.autograd.grad(energy_c, coords, create_graph=True)[0]
        force_loss_c = force_c.pow(2).sum()
        # outer backward: param-grad style — gradient w.r.t. coords (since
        # there are no params in this toy model, we use coords as the
        # "param" and check torch's d(force_loss)/d(coords)).
        gg_c = torch.autograd.grad(force_loss_c, coords)[0]

        energy_r = model_ref()
        force_r = torch.autograd.grad(energy_r, coords, create_graph=True)[0]
        force_loss_r = force_r.pow(2).sum()
        gg_r = torch.autograd.grad(force_loss_r, coords)[0]

        atol = 1e-4 if dtype == torch.float32 else 1e-8
        torch.testing.assert_close(gg_c, gg_r, atol=atol, rtol=atol)

    def test_double_backward_weights(self):
        """Same pattern but exercising the weights gradient path."""
        torch.manual_seed(3)
        N, F, topk, epg, eps = 24, 10, 4, 2, 2
        n_re = epg * eps
        dtype = torch.float64
        features = torch.randn(N, F, device="cuda", dtype=dtype)
        topk_indices = torch.randint(0, n_re, (N, topk), device="cuda")
        # treat topk_weights as the "param" we backprop into twice
        param = torch.randn(N, topk, device="cuda", dtype=dtype, requires_grad=True)

        def model_cuda():
            tw = param * 1.5 + 0.1
            sf, sei, sw, ui, _ = _cuda_op(
                features, topk_indices, tw, epg, n_re, eps
            )
            coeff = sei.to(dtype) * 0.1 + 1.0
            return (sw * coeff).pow(2).sum()

        def model_ref():
            tw = param * 1.5 + 0.1
            rf, rei, rw, rui, _, _ = _ref_topk_expand_sort(
                features, topk_indices, tw, epg
            )
            coeff = rei.to(dtype) * 0.1 + 1.0
            return (rw * coeff).pow(2).sum()

        e_c = model_cuda()
        f_c = torch.autograd.grad(e_c, param, create_graph=True)[0]
        gg_c = torch.autograd.grad(f_c.pow(2).sum(), param)[0]

        e_r = model_ref()
        f_r = torch.autograd.grad(e_r, param, create_graph=True)[0]
        gg_r = torch.autograd.grad(f_r.pow(2).sum(), param)[0]

        torch.testing.assert_close(gg_c, gg_r, atol=1e-8, rtol=1e-8)


# ======================================================================
#                          gradcheck / gradgradcheck
# ======================================================================
class TestGradCheck:
    """Numerical-vs-analytical Jacobian checks.

    Gradcheck must operate on a function that takes only *differentiable*
    tensors as inputs and returns only differentiable outputs.  We build a
    closure that captures the non-differentiable indices/sizes.
    """

    def _make_fn(self, topk_indices, n_re, epg, eps, sei_cached_holder):
        # sei_cached_holder: list to capture sorted_expert_ids from the
        # first call so the per-segment loss is permutation-stable.
        # gradcheck calls the function many times; use a deterministic
        # expert-id-based weighting so the *value* of each output element
        # depends only on the input feature value, not on within-segment
        # ordering.
        def fn(features, topk_weights):
            sf, sei, sw, ui, _ = _cuda_op(
                features, topk_indices, topk_weights, epg, n_re, eps
            )
            # Reduce along the [N*topk] dimension via segment-wise mean to
            # eliminate ordering ambiguity: scatter_add into per-expert
            # buckets, then return that.  This makes the function pure
            # of the within-expert ordering.
            n_re_local = n_re
            buf_f = torch.zeros(
                n_re_local, sf.shape[-1],
                device=sf.device, dtype=sf.dtype,
            )
            buf_w = torch.zeros(
                n_re_local, device=sw.device, dtype=sw.dtype,
            )
            buf_f.index_add_(0, sei, sf)
            buf_w.index_add_(0, sei, sw)
            return buf_f, buf_w

        return fn

    def test_gradcheck(self):
        torch.manual_seed(7)
        N, F, topk, epg, eps = 6, 5, 3, 2, 2
        n_re = epg * eps
        features = torch.randn(N, F, device="cuda", dtype=torch.float64,
                               requires_grad=True)
        topk_indices = torch.randint(0, n_re, (N, topk), device="cuda")
        topk_weights = torch.randn(N, topk, device="cuda", dtype=torch.float64,
                                   requires_grad=True)
        fn = self._make_fn(topk_indices, n_re, epg, eps, [])
        assert torch.autograd.gradcheck(
            fn, (features, topk_weights),
            eps=1e-6, atol=1e-5, rtol=1e-3,
        )

    def test_gradgradcheck(self):
        """MUST pass — DeePMD requires double backward through MoE."""
        torch.manual_seed(11)
        N, F, topk, epg, eps = 5, 4, 2, 2, 2
        n_re = epg * eps
        features = torch.randn(N, F, device="cuda", dtype=torch.float64,
                               requires_grad=True)
        topk_indices = torch.randint(0, n_re, (N, topk), device="cuda")
        topk_weights = torch.randn(N, topk, device="cuda", dtype=torch.float64,
                                   requires_grad=True)
        fn = self._make_fn(topk_indices, n_re, epg, eps, [])
        assert torch.autograd.gradgradcheck(
            fn, (features, topk_weights),
            eps=1e-6, atol=1e-4, rtol=1e-3,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
