# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for MoE support in DPA3 RepFlowLayer."""

import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor.dpa3 import (
    RepFlowArgs,
)
from deepmd.pt.model.descriptor import (
    DescrptDPA3,
)
from deepmd.pt.model.network.moe import (
    MoELayer,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

class TestMoELayer(unittest.TestCase):
    """Test MoELayer standalone."""

    def test_node_forward_backward(self):
        """Node input [nb, nloc, dim]."""
        layer = MoELayer(
            num_in=64, num_out=64, n_experts=4, top_k=2,
            tebd_dim=64, share_expert=0, activation_function="silu",
            precision="float64",
        ).to(device)
        x = torch.randn(2, 10, 64, dtype=torch.float64, device=device, requires_grad=True)
        type_emb = torch.randn(3, 64, dtype=torch.float64, device=device)
        atom_types = torch.randint(0, 3, (2, 10), device=device)
        out = layer(x, type_emb, atom_types)
        self.assertEqual(out.shape, (2, 10, 64))
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_edge_forward(self):
        """Edge input [nb, nloc, nnei, dim]."""
        layer = MoELayer(
            num_in=32, num_out=32, n_experts=3, top_k=2,
            tebd_dim=32, activation_function="silu", precision="float64",
        ).to(device)
        x = torch.randn(2, 10, 20, 32, dtype=torch.float64, device=device)
        type_emb = torch.randn(2, 32, dtype=torch.float64, device=device)
        atom_types = torch.randint(0, 2, (2, 10), device=device)
        out = layer(x, type_emb, atom_types)
        self.assertEqual(out.shape, (2, 10, 20, 32))

    def test_angle_forward(self):
        """Angle input [nb, nloc, a_nnei, a_nnei, dim]."""
        layer = MoELayer(
            num_in=16, num_out=16, n_experts=4, top_k=2,
            tebd_dim=16, activation_function="silu", precision="float64",
        ).to(device)
        x = torch.randn(2, 6, 5, 5, 16, dtype=torch.float64, device=device)
        type_emb = torch.randn(2, 16, dtype=torch.float64, device=device)
        atom_types = torch.randint(0, 2, (2, 6), device=device)
        out = layer(x, type_emb, atom_types)
        self.assertEqual(out.shape, (2, 6, 5, 5, 16))

    def test_flat_dynamic_sel(self):
        """Flat (dynamic_sel) input [n_flat, dim]."""
        layer = MoELayer(
            num_in=64, num_out=64, n_experts=4, top_k=2,
            tebd_dim=64, activation_function="silu", precision="float64",
        ).to(device)
        x = torch.randn(50, 64, dtype=torch.float64, device=device)
        type_emb = torch.randn(3, 64, dtype=torch.float64, device=device)
        atom_types = torch.randint(0, 3, (2, 10), device=device)
        edge_index = torch.randint(0, 20, (50,), device=device)
        out = layer(x, type_emb, atom_types, edge_index=edge_index)
        self.assertEqual(out.shape, (50, 64))

    def test_share_expert(self):
        """Test share_expert > 0."""
        layer = MoELayer(
            num_in=64, num_out=64, n_experts=4, top_k=2,
            tebd_dim=64, share_expert=1, activation_function="silu",
            precision="float64",
        ).to(device)
        self.assertEqual(layer.routed_experts, 3)
        self.assertEqual(layer.routed_top_k, 1)
        self.assertEqual(len(layer.shared_experts), 1)
        x = torch.randn(2, 10, 64, dtype=torch.float64, device=device)
        type_emb = torch.randn(3, 64, dtype=torch.float64, device=device)
        atom_types = torch.randint(0, 3, (2, 10), device=device)
        out = layer(x, type_emb, atom_types)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_serialization_roundtrip(self):
        """Serialize -> deserialize -> same output."""
        layer = MoELayer(
            num_in=32, num_out=32, n_experts=4, top_k=2,
            tebd_dim=32, share_expert=1, activation_function="silu",
            precision="float64",
        ).to(device)
        data = layer.serialize()
        self.assertEqual(data["@class"], "MoELayer")
        restored = MoELayer.deserialize(data).to(device)
        x = torch.randn(2, 5, 32, dtype=torch.float64, device=device)
        type_emb = torch.randn(3, 32, dtype=torch.float64, device=device)
        atom_types = torch.randint(0, 3, (2, 5), device=device)
        out1 = layer(x, type_emb, atom_types)
        out2 = restored(x, type_emb, atom_types)
        np.testing.assert_allclose(
            out1.detach().cpu().numpy(), out2.detach().cpu().numpy(), atol=1e-10,
        )


class TestDPA3MoE(unittest.TestCase, TestCaseSingleFrameWithNlist):
    """Test DPA3 with MoE enabled."""

    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_moe_forward_backward(self):
        """DPA3 with MoE: forward + backward."""
        nf, nloc, nnei = self.nlist.shape
        rng = np.random.default_rng(100)
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(rng.normal(size=(self.nt, nnei, 4)))

        repflow = RepFlowArgs(
            n_dim=20, e_dim=10, a_dim=8, nlayers=2,
            e_rcut=self.rcut, e_rcut_smth=self.rcut_smth, e_sel=nnei,
            a_rcut=self.rcut - 0.1, a_rcut_smth=self.rcut_smth, a_sel=nnei - 1,
            axis_neuron=4, update_angle=True, smooth_edge_update=True,
            optim_update=False,
            n_experts=4, moe_top_k=2,
            use_node_moe=True, use_edge_moe=False, use_angle_moe=False,
        )
        model = DescrptDPA3(
            self.nt, repflow=repflow, precision="float64",
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        model.repflows.mean = torch.tensor(davg, dtype=torch.float64, device=env.DEVICE)
        model.repflows.stddev = torch.tensor(dstd, dtype=torch.float64, device=env.DEVICE)

        coord = torch.tensor(self.coord_ext, dtype=torch.float64, device=env.DEVICE)
        atype = torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE)
        nlist = torch.tensor(self.nlist, dtype=int, device=env.DEVICE)
        mapping = torch.tensor(self.mapping, dtype=int, device=env.DEVICE)

        rd, _, _, _, _ = model(coord, atype, nlist, mapping)
        self.assertEqual(rd.shape[0], nf)
        self.assertEqual(rd.shape[1], nloc)
        self.assertEqual(rd.shape[2], 20)

        # Backward
        loss = rd.sum()
        loss.backward()
        n_with_grad = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().max() > 0
        )
        self.assertGreater(n_with_grad, 0)

    def test_moe_serialization(self):
        """DPA3 MoE: serialize -> deserialize -> same output."""
        nf, nloc, nnei = self.nlist.shape
        rng = np.random.default_rng(100)
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(rng.normal(size=(self.nt, nnei, 4)))

        repflow = RepFlowArgs(
            n_dim=20, e_dim=10, a_dim=8, nlayers=2,
            e_rcut=self.rcut, e_rcut_smth=self.rcut_smth, e_sel=nnei,
            a_rcut=self.rcut - 0.1, a_rcut_smth=self.rcut_smth, a_sel=nnei - 1,
            axis_neuron=4, update_angle=True, smooth_edge_update=True,
            optim_update=False,
            n_experts=4, moe_top_k=2,
            use_node_moe=True, use_edge_moe=False, use_angle_moe=False,
            share_expert=1,
        )
        model = DescrptDPA3(
            self.nt, repflow=repflow, precision="float64",
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        model.repflows.mean = torch.tensor(davg, dtype=torch.float64, device=env.DEVICE)
        model.repflows.stddev = torch.tensor(dstd, dtype=torch.float64, device=env.DEVICE)

        coord = torch.tensor(self.coord_ext, dtype=torch.float64, device=env.DEVICE)
        atype = torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE)
        nlist = torch.tensor(self.nlist, dtype=int, device=env.DEVICE)
        mapping = torch.tensor(self.mapping, dtype=int, device=env.DEVICE)

        rd0, _, _, _, _ = model(coord, atype, nlist, mapping)

        # Serialize -> deserialize
        model2 = DescrptDPA3.deserialize(model.serialize())
        rd1, _, _, _, _ = model2(coord, atype, nlist, mapping)
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy(), rd1.detach().cpu().numpy(), atol=1e-8,
        )

    def test_moe_all_targets(self):
        """DPA3 MoE with node + edge + angle MoE."""
        nf, nloc, nnei = self.nlist.shape
        rng = np.random.default_rng(100)
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(rng.normal(size=(self.nt, nnei, 4)))

        repflow = RepFlowArgs(
            n_dim=20, e_dim=10, a_dim=8, nlayers=2,
            e_rcut=self.rcut, e_rcut_smth=self.rcut_smth, e_sel=nnei,
            a_rcut=self.rcut - 0.1, a_rcut_smth=self.rcut_smth, a_sel=nnei - 1,
            axis_neuron=4, update_angle=True, smooth_edge_update=True,
            optim_update=False,
            n_experts=3, moe_top_k=2,
            use_node_moe=True, use_edge_moe=True, use_angle_moe=True,
        )
        model = DescrptDPA3(
            self.nt, repflow=repflow, precision="float64",
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        model.repflows.mean = torch.tensor(davg, dtype=torch.float64, device=env.DEVICE)
        model.repflows.stddev = torch.tensor(dstd, dtype=torch.float64, device=env.DEVICE)

        coord = torch.tensor(self.coord_ext, dtype=torch.float64, device=env.DEVICE)
        atype = torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE)
        nlist = torch.tensor(self.nlist, dtype=int, device=env.DEVICE)
        mapping = torch.tensor(self.mapping, dtype=int, device=env.DEVICE)

        rd, _, _, _, _ = model(coord, atype, nlist, mapping)
        self.assertEqual(rd.shape, (nf, nloc, 20))
        rd.sum().backward()


if __name__ == "__main__":
    unittest.main()
