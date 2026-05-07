# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the fused CUDA MoE pack-for-dispatch op.

Covers:
  - forward correctness vs. the pure-PyTorch ``MoEPacker.pack_for_dispatch``
    reference (float32 + float64)
  - backward correctness vs. autograd reference
  - double backward (energy → force → force-loss path)
  - torch.autograd.gradcheck (float64)
  - torch.autograd.gradgradcheck (float64) — must pass before integration
  - empty-input edge case (some GPUs receive 0 tokens)
"""

from __future__ import annotations

import pytest
import torch

import deepmd.pt.cxx_op  # noqa: F401  registers torch.ops.deepmd
from deepmd.pt.model.network.moe_pack_dispatch_cuda import (
    fused_pack_for_dispatch,
)
from deepmd.pt.model.network.moe_packer import MoEPacker


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _make_inputs(node_counts, edge_counts, angle_counts, a_dim, dtype, seed=0):
    torch.manual_seed(seed)
    Nn = sum(node_counts)
    Ne = sum(edge_counts)
    Na = sum(angle_counts)
    D_node = 28 * a_dim
    D_edge = 10 * a_dim
    D_angle = 4 * a_dim
    node_sorted = torch.randn(Nn, D_node, device="cuda", dtype=dtype)
    edge_sorted = torch.randn(Ne, D_edge, device="cuda", dtype=dtype)
    angle_sorted = torch.randn(Na, D_angle, device="cuda", dtype=dtype)
    return node_sorted, edge_sorted, angle_sorted


def _ref_pack(packer, ns, es, as_, nc, ec, ac):
    return packer.pack_for_dispatch(ns, es, as_, nc, ec, ac)


# ======================================================================
#                            Forward correctness
# ======================================================================
class TestForward:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "a_dim,node_counts,edge_counts,angle_counts",
        [
            # typical: ep_size=4, even-ish distribution, divisible groups
            (8, [16, 24, 8, 32], [40, 60, 20, 80], [100, 150, 50, 200]),
            # odd remainders force trailing-padding logic
            (4, [3, 5, 7, 11], [9, 13, 17, 19], [11, 13, 17, 23]),
            # ep_size=8 (matches production)
            (16, [4, 8, 12, 16, 20, 24, 28, 32],
                 [10, 14, 18, 22, 26, 30, 34, 38],
                 [50, 60, 70, 80, 90, 100, 110, 120]),
            # some GPUs empty
            (8, [0, 16, 0, 8], [0, 40, 0, 20], [0, 100, 0, 50]),
            # single GPU
            (8, [16], [40], [100]),
            # a_dim=1 (degenerate small feat dim)
            (1, [4, 6], [8, 12], [20, 30]),
        ],
    )
    def test_forward(self, dtype, a_dim, node_counts, edge_counts, angle_counts):
        ns, es, as_ = _make_inputs(node_counts, edge_counts, angle_counts,
                                   a_dim, dtype, seed=42)
        packer = MoEPacker(a_dim=a_dim)
        ref_packed, ref_splits = _ref_pack(
            packer, ns, es, as_, node_counts, edge_counts, angle_counts,
        )
        cu_packed, cu_splits = fused_pack_for_dispatch(
            ns, es, as_, node_counts, edge_counts, angle_counts,
            packer.edge_concat_in, packer.angle_concat_in, packer.D_packed_in,
        )
        atol = 1e-6 if dtype == torch.float64 else 1e-5
        assert cu_splits == ref_splits, f"splits mismatch: {cu_splits} vs {ref_splits}"
        assert cu_packed.shape == ref_packed.shape, (
            f"shape mismatch: {cu_packed.shape} vs {ref_packed.shape}"
        )
        torch.testing.assert_close(cu_packed, ref_packed, atol=atol, rtol=atol)

    def test_forward_all_empty(self):
        a_dim = 4
        packer = MoEPacker(a_dim=a_dim)
        ns = torch.empty(0, 28 * a_dim, device="cuda", dtype=torch.float32)
        es = torch.empty(0, 10 * a_dim, device="cuda", dtype=torch.float32)
        as_ = torch.empty(0, 4 * a_dim, device="cuda", dtype=torch.float32)
        cu_packed, cu_splits = fused_pack_for_dispatch(
            ns, es, as_, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
            packer.edge_concat_in, packer.angle_concat_in, packer.D_packed_in,
        )
        assert cu_packed.shape == (0, packer.D_packed_in)
        assert cu_splits == [0, 0, 0, 0]


# ======================================================================
#                          Backward correctness
# ======================================================================
class TestBackward:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_backward(self, dtype):
        a_dim = 8
        node_counts = [3, 5, 7, 11]
        edge_counts = [9, 13, 17, 19]
        angle_counts = [11, 13, 17, 23]
        packer = MoEPacker(a_dim=a_dim)

        ns, es, as_ = _make_inputs(node_counts, edge_counts, angle_counts,
                                   a_dim, dtype, seed=1)

        # CUDA path
        ns_c = ns.detach().clone().requires_grad_(True)
        es_c = es.detach().clone().requires_grad_(True)
        as_c = as_.detach().clone().requires_grad_(True)
        packed_c, _ = fused_pack_for_dispatch(
            ns_c, es_c, as_c, node_counts, edge_counts, angle_counts,
            packer.edge_concat_in, packer.angle_concat_in, packer.D_packed_in,
        )
        # Loss with non-uniform weights so backward exercises both columns
        # AND rows of the packed tensor.
        weights = torch.randn_like(packed_c)
        (packed_c * weights).sum().backward()
        gn_c, ge_c, ga_c = ns_c.grad.clone(), es_c.grad.clone(), as_c.grad.clone()

        # Reference path
        ns_r = ns.detach().clone().requires_grad_(True)
        es_r = es.detach().clone().requires_grad_(True)
        as_r = as_.detach().clone().requires_grad_(True)
        packed_r, _ = packer.pack_for_dispatch(
            ns_r, es_r, as_r, node_counts, edge_counts, angle_counts,
        )
        (packed_r * weights).sum().backward()
        gn_r, ge_r, ga_r = ns_r.grad.clone(), es_r.grad.clone(), as_r.grad.clone()

        atol = 1e-6 if dtype == torch.float64 else 1e-5
        torch.testing.assert_close(gn_c, gn_r, atol=atol, rtol=atol)
        torch.testing.assert_close(ge_c, ge_r, atol=atol, rtol=atol)
        torch.testing.assert_close(ga_c, ga_r, atol=atol, rtol=atol)


# ======================================================================
#                       Double backward (DeePMD use case)
# ======================================================================
class TestDoubleBackward:
    """Simulate energy → force → force-loss → param grad (DeePMD path)."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_double_backward_features(self, dtype):
        a_dim = 4
        node_counts = [3, 5, 0, 4]
        edge_counts = [9, 0, 7, 11]
        angle_counts = [11, 13, 0, 17]
        packer = MoEPacker(a_dim=a_dim)

        torch.manual_seed(2)
        Nn = sum(node_counts)
        Ne = sum(edge_counts)
        Na = sum(angle_counts)
        D_node, D_edge, D_angle = 28 * a_dim, 10 * a_dim, 4 * a_dim

        coords_n = torch.randn(Nn, D_node, device="cuda", dtype=dtype, requires_grad=True)
        coords_e = torch.randn(Ne, D_edge, device="cuda", dtype=dtype, requires_grad=True)
        coords_a = torch.randn(Na, D_angle, device="cuda", dtype=dtype, requires_grad=True)

        # Tiny "model" — coords passed through a Linear-like map so the 2nd
        # derivative is non-trivial.
        Wn = torch.randn(D_node, D_node, device="cuda", dtype=dtype)
        We = torch.randn(D_edge, D_edge, device="cuda", dtype=dtype)
        Wa = torch.randn(D_angle, D_angle, device="cuda", dtype=dtype)

        def model_cuda():
            ns = coords_n @ Wn
            es = coords_e @ We
            as_ = coords_a @ Wa
            packed, _ = fused_pack_for_dispatch(
                ns, es, as_, node_counts, edge_counts, angle_counts,
                packer.edge_concat_in, packer.angle_concat_in, packer.D_packed_in,
            )
            return packed.pow(2).sum()

        def model_ref():
            ns = coords_n @ Wn
            es = coords_e @ We
            as_ = coords_a @ Wa
            packed, _ = packer.pack_for_dispatch(
                ns, es, as_, node_counts, edge_counts, angle_counts,
            )
            return packed.pow(2).sum()

        # CUDA: energy → force = d(energy)/d(coords_n), keep graph
        ec = model_cuda()
        force_c = torch.autograd.grad(ec, coords_n, create_graph=True)[0]
        gg_c = torch.autograd.grad(force_c.pow(2).sum(), coords_n)[0]

        er = model_ref()
        force_r = torch.autograd.grad(er, coords_n, create_graph=True)[0]
        gg_r = torch.autograd.grad(force_r.pow(2).sum(), coords_n)[0]

        atol = 1e-4 if dtype == torch.float32 else 1e-8
        torch.testing.assert_close(gg_c, gg_r, atol=atol, rtol=atol)


# ======================================================================
#                          gradcheck / gradgradcheck
# ======================================================================
class TestGradCheck:
    def _make_fn(self, node_counts, edge_counts, angle_counts, packer):
        def fn(ns, es, as_):
            packed, _ = fused_pack_for_dispatch(
                ns, es, as_, node_counts, edge_counts, angle_counts,
                packer.edge_concat_in, packer.angle_concat_in,
                packer.D_packed_in,
            )
            return packed
        return fn

    def test_gradcheck(self):
        a_dim = 2
        node_counts = [2, 3]
        edge_counts = [5, 6]
        angle_counts = [4, 7]
        packer = MoEPacker(a_dim=a_dim)
        ns, es, as_ = _make_inputs(node_counts, edge_counts, angle_counts,
                                   a_dim, torch.float64, seed=7)
        ns.requires_grad_(True)
        es.requires_grad_(True)
        as_.requires_grad_(True)
        fn = self._make_fn(node_counts, edge_counts, angle_counts, packer)
        assert torch.autograd.gradcheck(
            fn, (ns, es, as_), eps=1e-6, atol=1e-5, rtol=1e-3,
        )

    def test_gradgradcheck(self):
        """MUST pass — DeePMD requires double backward through MoE."""
        a_dim = 2
        node_counts = [2, 3]
        edge_counts = [3, 4]
        angle_counts = [4, 5]
        packer = MoEPacker(a_dim=a_dim)
        ns, es, as_ = _make_inputs(node_counts, edge_counts, angle_counts,
                                   a_dim, torch.float64, seed=11)
        ns.requires_grad_(True)
        es.requires_grad_(True)
        as_.requires_grad_(True)
        fn = self._make_fn(node_counts, edge_counts, angle_counts, packer)
        assert torch.autograd.gradgradcheck(
            fn, (ns, es, as_), eps=1e-6, atol=1e-4, rtol=1e-3,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
