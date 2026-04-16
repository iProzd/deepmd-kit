# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single-GPU unit tests for MoEPacker.

Run: CUDA_VISIBLE_DEVICES=0 python -m pytest source/tests/pt/test_moe_packer.py -v

Tests cover:
  - validate_dim_ratio
  - Dispatch pack/unpack roundtrip (node, edge, angle)
  - Combine pack/unpack roundtrip
  - Non-divisible edge/angle counts (padding correctness)
  - Multi-GPU counts simulation
  - Input/output row count identity
  - Gradient passthrough and 2nd-order derivatives
  - Empty feature handling
  - Edge cases (count=1)
  - Full dispatch→combine cycle
  - exchange_metadata (group=None noop)
  - counts_to_packed_rows
"""

import torch
import pytest

from deepmd.pt.model.network.moe_packer import (
    MoEPacker,
    counts_to_packed_rows,
    exchange_metadata,
    validate_dim_ratio,
)

DEVICE = torch.device("cpu")
DTYPE = torch.float64
# Use a_dim=3 for manageable tensor sizes.
A = 3


# ---------------------------------------------------------------------------
# validate_dim_ratio
# ---------------------------------------------------------------------------


class TestValidateDimRatio:
    def test_valid(self):
        validate_dim_ratio(n_dim=12, e_dim=6, a_dim=3)
        validate_dim_ratio(n_dim=4, e_dim=2, a_dim=1)
        validate_dim_ratio(n_dim=20, e_dim=10, a_dim=5)

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="4:2:1"):
            validate_dim_ratio(n_dim=10, e_dim=6, a_dim=3)

    def test_invalid_e(self):
        with pytest.raises(ValueError, match="4:2:1"):
            validate_dim_ratio(n_dim=12, e_dim=5, a_dim=3)

    def test_invalid_both(self):
        with pytest.raises(ValueError, match="4:2:1"):
            validate_dim_ratio(n_dim=7, e_dim=5, a_dim=3)


# ---------------------------------------------------------------------------
# Helper to create a packer and random tensors
# ---------------------------------------------------------------------------

def _make_packer():
    return MoEPacker(a_dim=A)


def _rand(n, dim):
    """Create a random tensor with requires_grad=False."""
    return torch.randn(n, dim, dtype=DTYPE, device=DEVICE)


def _rand_grad(n, dim):
    """Create a random tensor with requires_grad=True."""
    return torch.randn(n, dim, dtype=DTYPE, device=DEVICE, requires_grad=True)


# ---------------------------------------------------------------------------
# Dispatch pack → unpack roundtrip tests
# ---------------------------------------------------------------------------


class TestDispatchRoundtrip:
    """Test pack_for_dispatch → unpack_from_dispatch recovers original data."""

    def test_node_only(self):
        """Node features round-trip through pack/unpack."""
        p = _make_packer()
        N_node = 5
        node = _rand(N_node, 28 * A)
        edge = _rand(0, 10 * A)
        angle = _rand(0, 4 * A)

        packed, splits = p.pack_for_dispatch(
            node, edge, angle, [N_node], [0], [0]
        )
        assert packed.shape == (N_node, 40 * A)
        assert splits == [N_node]

        n_out, e_out, a_out = p.unpack_from_dispatch(
            packed, [N_node], [0], [0]
        )
        torch.testing.assert_close(n_out, node)
        assert e_out.shape == (0, 10 * A)
        assert a_out.shape == (0, 4 * A)

    def test_edge_divisible(self):
        """Edge count exactly divisible by 4."""
        p = _make_packer()
        N_edge = 8  # 8 / 4 = 2 rows
        node = _rand(0, 28 * A)
        edge = _rand(N_edge, 10 * A)
        angle = _rand(0, 4 * A)

        packed, splits = p.pack_for_dispatch(
            node, edge, angle, [0], [N_edge], [0]
        )
        assert packed.shape == (2, 40 * A)
        assert splits == [2]

        _, e_out, _ = p.unpack_from_dispatch(packed, [0], [N_edge], [0])
        torch.testing.assert_close(e_out, edge)

    def test_edge_non_divisible(self):
        """Edge count not divisible by 4: N_edge=7 → 2 rows, last has 3 valid."""
        p = _make_packer()
        N_edge = 7  # ceil(7/4) = 2 rows
        edge = _rand(N_edge, 10 * A)

        packed, splits = p.pack_for_dispatch(
            _rand(0, 28 * A), edge, _rand(0, 4 * A),
            [0], [N_edge], [0],
        )
        assert packed.shape == (2, 40 * A)
        assert splits == [2]

        _, e_out, _ = p.unpack_from_dispatch(packed, [0], [N_edge], [0])
        assert e_out.shape == (N_edge, 10 * A)
        torch.testing.assert_close(e_out, edge)

    def test_angle_divisible(self):
        """Angle count exactly divisible by 10."""
        p = _make_packer()
        N_angle = 20  # 20 / 10 = 2 rows
        angle = _rand(N_angle, 4 * A)

        packed, splits = p.pack_for_dispatch(
            _rand(0, 28 * A), _rand(0, 10 * A), angle,
            [0], [0], [N_angle],
        )
        assert packed.shape == (2, 40 * A)
        assert splits == [2]

        _, _, a_out = p.unpack_from_dispatch(packed, [0], [0], [N_angle])
        torch.testing.assert_close(a_out, angle)

    def test_angle_non_divisible(self):
        """Angle count not divisible by 10: N_angle=13 → 2 rows."""
        p = _make_packer()
        N_angle = 13  # ceil(13/10) = 2 rows
        angle = _rand(N_angle, 4 * A)

        packed, splits = p.pack_for_dispatch(
            _rand(0, 28 * A), _rand(0, 10 * A), angle,
            [0], [0], [N_angle],
        )
        assert packed.shape == (2, 40 * A)

        _, _, a_out = p.unpack_from_dispatch(packed, [0], [0], [N_angle])
        assert a_out.shape == (N_angle, 4 * A)
        torch.testing.assert_close(a_out, angle)

    def test_all_features_single_gpu(self):
        """All three feature types packed together for 1 GPU."""
        p = _make_packer()
        N_node, N_edge, N_angle = 3, 7, 13
        node = _rand(N_node, 28 * A)
        edge = _rand(N_edge, 10 * A)
        angle = _rand(N_angle, 4 * A)

        # Expected rows: 3 + ceil(7/4) + ceil(13/10) = 3 + 2 + 2 = 7
        packed, splits = p.pack_for_dispatch(
            node, edge, angle,
            [N_node], [N_edge], [N_angle],
        )
        assert packed.shape == (7, 40 * A)
        assert splits == [7]

        n_out, e_out, a_out = p.unpack_from_dispatch(
            packed, [N_node], [N_edge], [N_angle]
        )
        torch.testing.assert_close(n_out, node)
        torch.testing.assert_close(e_out, edge)
        torch.testing.assert_close(a_out, angle)

    def test_multi_gpu_counts(self):
        """Simulate ep_size=4 with different counts per GPU."""
        p = _make_packer()
        node_counts = [3, 5, 0, 2]
        edge_counts = [7, 0, 12, 1]
        angle_counts = [13, 20, 0, 3]

        total_node = sum(node_counts)
        total_edge = sum(edge_counts)
        total_angle = sum(angle_counts)

        node = _rand(total_node, 28 * A)
        edge = _rand(total_edge, 10 * A)
        angle = _rand(total_angle, 4 * A)

        packed, splits = p.pack_for_dispatch(
            node, edge, angle,
            node_counts, edge_counts, angle_counts,
        )

        # Verify splits.
        expected_splits = []
        for g in range(4):
            nr = node_counts[g]
            er = (edge_counts[g] + 3) // 4 if edge_counts[g] > 0 else 0
            ar = (angle_counts[g] + 9) // 10 if angle_counts[g] > 0 else 0
            expected_splits.append(nr + er + ar)
        assert splits == expected_splits
        assert packed.shape == (sum(expected_splits), 40 * A)

        # Roundtrip.
        n_out, e_out, a_out = p.unpack_from_dispatch(
            packed, node_counts, edge_counts, angle_counts,
        )
        torch.testing.assert_close(n_out, node)
        torch.testing.assert_close(e_out, edge)
        torch.testing.assert_close(a_out, angle)


# ---------------------------------------------------------------------------
# Combine pack → unpack roundtrip tests
# ---------------------------------------------------------------------------


class TestCombineRoundtrip:
    """Test pack_for_combine → unpack_from_combine recovers original data."""

    def test_node_only(self):
        p = _make_packer()
        N_node = 5
        node = _rand(N_node, 8 * A)

        packed = p.pack_for_combine(
            node, _rand(0, 6 * A), _rand(0, 3 * A),
            [N_node], [0], [0],
        )
        assert packed.shape == (N_node, 30 * A)

        n_out, e_out, a_out = p.unpack_from_combine(
            packed, [N_node], [0], [0]
        )
        torch.testing.assert_close(n_out, node)
        assert e_out.shape == (0, 6 * A)
        assert a_out.shape == (0, 3 * A)

    def test_edge_roundtrip(self):
        """Edge output: 6a → 4concat → 24a → pad+6a → 30a → unpack."""
        p = _make_packer()
        N_edge = 9  # ceil(9/4) = 3 rows
        edge = _rand(N_edge, 6 * A)

        packed = p.pack_for_combine(
            _rand(0, 8 * A), edge, _rand(0, 3 * A),
            [0], [N_edge], [0],
        )
        assert packed.shape == (3, 30 * A)

        _, e_out, _ = p.unpack_from_combine(packed, [0], [N_edge], [0])
        assert e_out.shape == (N_edge, 6 * A)
        torch.testing.assert_close(e_out, edge)

    def test_angle_roundtrip(self):
        """Angle output: 3a → 10concat → 30a (exact) → unpack."""
        p = _make_packer()
        N_angle = 15  # ceil(15/10) = 2 rows
        angle = _rand(N_angle, 3 * A)

        packed = p.pack_for_combine(
            _rand(0, 8 * A), _rand(0, 6 * A), angle,
            [0], [0], [N_angle],
        )
        assert packed.shape == (2, 30 * A)

        _, _, a_out = p.unpack_from_combine(packed, [0], [0], [N_angle])
        assert a_out.shape == (N_angle, 3 * A)
        torch.testing.assert_close(a_out, angle)

    def test_non_divisible(self):
        """Edge and angle with non-divisible counts."""
        p = _make_packer()
        N_edge, N_angle = 5, 7
        edge = _rand(N_edge, 6 * A)
        angle = _rand(N_angle, 3 * A)

        packed = p.pack_for_combine(
            _rand(0, 8 * A), edge, angle,
            [0], [N_edge], [N_angle],
        )
        # ceil(5/4) = 2, ceil(7/10) = 1 → 3 rows
        assert packed.shape == (3, 30 * A)

        _, e_out, a_out = p.unpack_from_combine(packed, [0], [N_edge], [N_angle])
        torch.testing.assert_close(e_out, edge)
        torch.testing.assert_close(a_out, angle)


# ---------------------------------------------------------------------------
# Row count identity
# ---------------------------------------------------------------------------


class TestRowCountIdentity:
    """Input (40a) and output (30a) packed row counts must be identical."""

    def test_same_row_count(self):
        p = _make_packer()
        node_counts = [3, 5]
        edge_counts = [7, 12]
        angle_counts = [13, 20]

        # Dispatch.
        node_in = _rand(sum(node_counts), 28 * A)
        edge_in = _rand(sum(edge_counts), 10 * A)
        angle_in = _rand(sum(angle_counts), 4 * A)
        packed_in, send_splits = p.pack_for_dispatch(
            node_in, edge_in, angle_in,
            node_counts, edge_counts, angle_counts,
        )

        # Combine.
        node_out = _rand(sum(node_counts), 8 * A)
        edge_out = _rand(sum(edge_counts), 6 * A)
        angle_out = _rand(sum(angle_counts), 3 * A)
        packed_out = p.pack_for_combine(
            node_out, edge_out, angle_out,
            node_counts, edge_counts, angle_counts,
        )

        assert packed_in.shape[0] == packed_out.shape[0]


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


class TestGradient:
    """Pack/unpack must not block gradient flow."""

    def test_dispatch_gradient(self):
        """Gradients flow through dispatch pack → unpack."""
        p = _make_packer()
        N_node, N_edge, N_angle = 3, 7, 5
        node = _rand_grad(N_node, 28 * A)
        edge = _rand_grad(N_edge, 10 * A)
        angle = _rand_grad(N_angle, 4 * A)

        packed, _ = p.pack_for_dispatch(
            node, edge, angle,
            [N_node], [N_edge], [N_angle],
        )
        n_out, e_out, a_out = p.unpack_from_dispatch(
            packed, [N_node], [N_edge], [N_angle],
        )

        loss = n_out.sum() + e_out.sum() + a_out.sum()
        loss.backward()

        assert node.grad is not None, "node grad is None"
        assert node.grad.abs().sum() > 0, "node grad is all zeros"
        assert edge.grad is not None, "edge grad is None"
        assert edge.grad.abs().sum() > 0, "edge grad is all zeros"
        assert angle.grad is not None, "angle grad is None"
        assert angle.grad.abs().sum() > 0, "angle grad is all zeros"

    def test_combine_gradient(self):
        """Gradients flow through combine pack → unpack."""
        p = _make_packer()
        N_node, N_edge, N_angle = 4, 6, 11
        node = _rand_grad(N_node, 8 * A)
        edge = _rand_grad(N_edge, 6 * A)
        angle = _rand_grad(N_angle, 3 * A)

        packed = p.pack_for_combine(
            node, edge, angle,
            [N_node], [N_edge], [N_angle],
        )
        n_out, e_out, a_out = p.unpack_from_combine(
            packed, [N_node], [N_edge], [N_angle],
        )

        loss = n_out.sum() + e_out.sum() + a_out.sum()
        loss.backward()

        assert node.grad is not None
        assert node.grad.abs().sum() > 0
        assert edge.grad is not None
        assert edge.grad.abs().sum() > 0
        assert angle.grad is not None
        assert angle.grad.abs().sum() > 0

    def test_create_graph_2nd_order(self):
        """create_graph=True works through full pack → unpack cycle."""
        p = _make_packer()
        N_node, N_edge, N_angle = 2, 5, 3
        node = _rand_grad(N_node, 28 * A)
        edge = _rand_grad(N_edge, 10 * A)
        angle = _rand_grad(N_angle, 4 * A)

        packed, _ = p.pack_for_dispatch(
            node, edge, angle,
            [N_node], [N_edge], [N_angle],
        )
        n_out, e_out, a_out = p.unpack_from_dispatch(
            packed, [N_node], [N_edge], [N_angle],
        )
        loss = (n_out ** 2).sum() + (e_out ** 2).sum() + (a_out ** 2).sum()

        inputs = [node, edge, angle]
        grads = torch.autograd.grad(loss, inputs, create_graph=True)
        for g in grads:
            assert g is not None
            assert g.requires_grad

        grad_sum = sum(g.sum() for g in grads)
        grad_sum.backward()
        for inp in inputs:
            assert inp.grad is not None, "2nd order grad is None"


# ---------------------------------------------------------------------------
# Empty feature handling
# ---------------------------------------------------------------------------


class TestEmptyFeatures:
    """Handle N=0 for any feature type gracefully."""

    def test_all_empty(self):
        p = _make_packer()
        packed, splits = p.pack_for_dispatch(
            _rand(0, 28 * A), _rand(0, 10 * A), _rand(0, 4 * A),
            [0], [0], [0],
        )
        assert packed.shape == (0, 40 * A)
        assert splits == [0]

        n, e, a = p.unpack_from_dispatch(packed, [0], [0], [0])
        assert n.shape == (0, 28 * A)
        assert e.shape == (0, 10 * A)
        assert a.shape == (0, 4 * A)

    def test_only_edges(self):
        """Only edge features, no node or angle."""
        p = _make_packer()
        N_edge = 6
        edge = _rand(N_edge, 10 * A)

        packed, _ = p.pack_for_dispatch(
            _rand(0, 28 * A), edge, _rand(0, 4 * A),
            [0], [N_edge], [0],
        )
        _, e_out, _ = p.unpack_from_dispatch(packed, [0], [N_edge], [0])
        torch.testing.assert_close(e_out, edge)

    def test_only_angles(self):
        """Only angle features, no node or edge."""
        p = _make_packer()
        N_angle = 11
        angle = _rand(N_angle, 4 * A)

        packed, _ = p.pack_for_dispatch(
            _rand(0, 28 * A), _rand(0, 10 * A), angle,
            [0], [0], [N_angle],
        )
        _, _, a_out = p.unpack_from_dispatch(packed, [0], [0], [N_angle])
        torch.testing.assert_close(a_out, angle)

    def test_some_gpus_empty(self):
        """Some GPUs have 0 tokens for certain feature types."""
        p = _make_packer()
        # GPU 0: only nodes, GPU 1: only edges, GPU 2: only angles
        node_counts = [3, 0, 0]
        edge_counts = [0, 5, 0]
        angle_counts = [0, 0, 7]

        node = _rand(3, 28 * A)
        edge = _rand(5, 10 * A)
        angle = _rand(7, 4 * A)

        packed, splits = p.pack_for_dispatch(
            node, edge, angle,
            node_counts, edge_counts, angle_counts,
        )

        n_out, e_out, a_out = p.unpack_from_dispatch(
            packed, node_counts, edge_counts, angle_counts,
        )
        torch.testing.assert_close(n_out, node)
        torch.testing.assert_close(e_out, edge)
        torch.testing.assert_close(a_out, angle)


# ---------------------------------------------------------------------------
# Edge cases: count = 1
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary cases with minimal counts."""

    def test_edge_count_1(self):
        """N_edge=1 → 1 row with 1 valid edge + 3 padding."""
        p = _make_packer()
        edge = _rand(1, 10 * A)

        packed, splits = p.pack_for_dispatch(
            _rand(0, 28 * A), edge, _rand(0, 4 * A),
            [0], [1], [0],
        )
        assert packed.shape == (1, 40 * A)

        _, e_out, _ = p.unpack_from_dispatch(packed, [0], [1], [0])
        assert e_out.shape == (1, 10 * A)
        torch.testing.assert_close(e_out, edge)

    def test_angle_count_1(self):
        """N_angle=1 → 1 row with 1 valid angle + 9 padding."""
        p = _make_packer()
        angle = _rand(1, 4 * A)

        packed, splits = p.pack_for_dispatch(
            _rand(0, 28 * A), _rand(0, 10 * A), angle,
            [0], [0], [1],
        )
        assert packed.shape == (1, 40 * A)

        _, _, a_out = p.unpack_from_dispatch(packed, [0], [0], [1])
        assert a_out.shape == (1, 4 * A)
        torch.testing.assert_close(a_out, angle)

    def test_single_node(self):
        """N_node=1."""
        p = _make_packer()
        node = _rand(1, 28 * A)

        packed, splits = p.pack_for_dispatch(
            node, _rand(0, 10 * A), _rand(0, 4 * A),
            [1], [0], [0],
        )
        assert packed.shape == (1, 40 * A)

        n_out, _, _ = p.unpack_from_dispatch(packed, [1], [0], [0])
        torch.testing.assert_close(n_out, node)


# ---------------------------------------------------------------------------
# Full dispatch → combine cycle
# ---------------------------------------------------------------------------


class TestFullCycle:
    """Test the complete dispatch → (identity A2A) → unpack → (identity expert)
    → combine → (identity A2A) → unpack_combine cycle."""

    def test_full_dispatch_combine_cycle(self):
        """Full cycle with all feature types and multi-GPU counts."""
        p = _make_packer()
        node_counts = [4, 3]
        edge_counts = [7, 5]
        angle_counts = [13, 8]

        total_node = sum(node_counts)
        total_edge = sum(edge_counts)
        total_angle = sum(angle_counts)

        node_in = _rand(total_node, 28 * A)
        edge_in = _rand(total_edge, 10 * A)
        angle_in = _rand(total_angle, 4 * A)

        # 1. Pack for dispatch.
        packed_dispatch, send_splits = p.pack_for_dispatch(
            node_in, edge_in, angle_in,
            node_counts, edge_counts, angle_counts,
        )

        # 2. Simulate identity A2A (single GPU: returned = sent).
        recv_tensor = packed_dispatch

        # 3. Unpack from dispatch.
        n_recv, e_recv, a_recv = p.unpack_from_dispatch(
            recv_tensor, node_counts, edge_counts, angle_counts,
        )
        torch.testing.assert_close(n_recv, node_in)
        torch.testing.assert_close(e_recv, edge_in)
        torch.testing.assert_close(a_recv, angle_in)

        # 4. Simulate expert compute: truncate to output dims.
        #    node: 28a → 8a (take first 8a), edge: 10a → 6a, angle: 4a → 3a
        n_out = n_recv[:, :8 * A]
        e_out = e_recv[:, :6 * A]
        a_out = a_recv[:, :3 * A]

        # 5. Pack for combine.
        packed_combine = p.pack_for_combine(
            n_out, e_out, a_out,
            node_counts, edge_counts, angle_counts,
        )

        # Row counts must match.
        assert packed_dispatch.shape[0] == packed_combine.shape[0]

        # 6. Simulate identity A2A return.
        returned = packed_combine

        # 7. Unpack from combine.
        n_final, e_final, a_final = p.unpack_from_combine(
            returned, node_counts, edge_counts, angle_counts,
        )

        # Verify data matches the "expert output".
        torch.testing.assert_close(n_final, n_out)
        torch.testing.assert_close(e_final, e_out)
        torch.testing.assert_close(a_final, a_out)

    def test_full_cycle_gradient(self):
        """Gradients flow through the full dispatch→combine cycle."""
        p = _make_packer()
        N_node, N_edge, N_angle = 3, 7, 5
        node_in = _rand_grad(N_node, 28 * A)
        edge_in = _rand_grad(N_edge, 10 * A)
        angle_in = _rand_grad(N_angle, 4 * A)

        # Dispatch pack/unpack.
        packed, _ = p.pack_for_dispatch(
            node_in, edge_in, angle_in,
            [N_node], [N_edge], [N_angle],
        )
        n_recv, e_recv, a_recv = p.unpack_from_dispatch(
            packed, [N_node], [N_edge], [N_angle],
        )

        # Simulated expert: scale by 2 (differentiable operation).
        n_out = n_recv[:, :8 * A] * 2.0
        e_out = e_recv[:, :6 * A] * 2.0
        a_out = a_recv[:, :3 * A] * 2.0

        # Combine pack/unpack.
        packed_out = p.pack_for_combine(
            n_out, e_out, a_out,
            [N_node], [N_edge], [N_angle],
        )
        n_final, e_final, a_final = p.unpack_from_combine(
            packed_out, [N_node], [N_edge], [N_angle],
        )

        loss = n_final.sum() + e_final.sum() + a_final.sum()
        loss.backward()

        assert node_in.grad is not None, "node_in grad is None"
        assert node_in.grad.abs().sum() > 0
        assert edge_in.grad is not None, "edge_in grad is None"
        assert edge_in.grad.abs().sum() > 0
        assert angle_in.grad is not None, "angle_in grad is None"
        assert angle_in.grad.abs().sum() > 0

    def test_large_multi_gpu_mixed(self):
        """Large test with 4 GPU groups, mixed counts, all non-divisible."""
        p = _make_packer()
        node_counts = [10, 7, 0, 3]
        edge_counts = [15, 1, 9, 0]
        angle_counts = [23, 0, 4, 11]

        total_node = sum(node_counts)
        total_edge = sum(edge_counts)
        total_angle = sum(angle_counts)

        node_in = _rand(total_node, 28 * A)
        edge_in = _rand(total_edge, 10 * A)
        angle_in = _rand(total_angle, 4 * A)

        # Dispatch roundtrip.
        packed, splits = p.pack_for_dispatch(
            node_in, edge_in, angle_in,
            node_counts, edge_counts, angle_counts,
        )
        n_recv, e_recv, a_recv = p.unpack_from_dispatch(
            packed, node_counts, edge_counts, angle_counts,
        )
        torch.testing.assert_close(n_recv, node_in)
        torch.testing.assert_close(e_recv, edge_in)
        torch.testing.assert_close(a_recv, angle_in)

        # Combine roundtrip.
        node_out = _rand(total_node, 8 * A)
        edge_out = _rand(total_edge, 6 * A)
        angle_out = _rand(total_angle, 3 * A)

        packed_out = p.pack_for_combine(
            node_out, edge_out, angle_out,
            node_counts, edge_counts, angle_counts,
        )
        n_final, e_final, a_final = p.unpack_from_combine(
            packed_out, node_counts, edge_counts, angle_counts,
        )
        torch.testing.assert_close(n_final, node_out)
        torch.testing.assert_close(e_final, edge_out)
        torch.testing.assert_close(a_final, angle_out)

        # Row count identity.
        assert packed.shape[0] == packed_out.shape[0]


# ---------------------------------------------------------------------------
# exchange_metadata (group=None noop)
# ---------------------------------------------------------------------------


class TestExchangeMetadataSingleGPU:
    """exchange_metadata with group=None must return input unchanged."""

    def test_noop_returns_same(self):
        """group=None returns the exact same tensor."""
        send = torch.tensor([[3, 7, 13]], dtype=torch.int64, device=DEVICE)
        recv = exchange_metadata(send, ep_group=None)
        assert recv is send

    def test_noop_multi_row(self):
        """group=None with multi-row input returns the same tensor."""
        send = torch.tensor([
            [5, 8, 20],
            [2, 0, 11],
            [0, 3, 0],
        ], dtype=torch.int64, device=DEVICE)
        recv = exchange_metadata(send, ep_group=None)
        assert recv is send


# ---------------------------------------------------------------------------
# counts_to_packed_rows
# ---------------------------------------------------------------------------


class TestCountsToPackedRows:
    """Test the counts_to_packed_rows utility."""

    def test_basic(self):
        """Standard case: node + edge + angle counts → packed rows."""
        node_counts = [3, 5]
        edge_counts = [7, 12]
        angle_counts = [13, 20]
        result = counts_to_packed_rows(node_counts, edge_counts, angle_counts)
        # GPU 0: 3 + ceil(7/4) + ceil(13/10) = 3 + 2 + 2 = 7
        # GPU 1: 5 + ceil(12/4) + ceil(20/10) = 5 + 3 + 2 = 10
        assert result == [7, 10]

    def test_zeros(self):
        """All-zero counts give all-zero rows."""
        result = counts_to_packed_rows([0, 0], [0, 0], [0, 0])
        assert result == [0, 0]

    def test_edge_only(self):
        """Only edge counts, rest zero."""
        result = counts_to_packed_rows([0], [7], [0])
        assert result == [2]  # ceil(7/4) = 2

    def test_angle_only(self):
        """Only angle counts, rest zero."""
        result = counts_to_packed_rows([0], [0], [13])
        assert result == [2]  # ceil(13/10) = 2

    def test_node_only(self):
        """Only node counts, rest zero."""
        result = counts_to_packed_rows([5], [0], [0])
        assert result == [5]

    def test_consistent_with_packer(self):
        """counts_to_packed_rows must agree with MoEPacker.pack_for_dispatch."""
        p = _make_packer()
        node_counts = [3, 5, 0, 2]
        edge_counts = [7, 0, 12, 1]
        angle_counts = [13, 20, 0, 3]

        node = _rand(sum(node_counts), 28 * A)
        edge = _rand(sum(edge_counts), 10 * A)
        angle = _rand(sum(angle_counts), 4 * A)

        _, send_splits = p.pack_for_dispatch(
            node, edge, angle,
            node_counts, edge_counts, angle_counts,
        )
        computed = counts_to_packed_rows(node_counts, edge_counts, angle_counts)
        assert send_splits == computed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
