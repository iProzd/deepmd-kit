# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for DPA3 V2 (dpa3s_v2_rbf_norm) descriptor."""

import unittest

import numpy as np
import torch

from deepmd.pt.model.descriptor.dpa3s_v2_rbf_norm import (
    DescrptDPA3V2,
)
from deepmd.pt.model.model import (
    get_model,
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
from ...common.test_mixins import (
    TestCaseSingleFrameWithNlist,
    get_tols,
)


dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestDescrptDPA3V2(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def _make_descriptor(self, prec="float64"):
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)
        dt = PRECISION_DICT[prec]

        dd = DescrptDPA3V2(
            self.nt,
            n_dim=20,
            e_dim=10,
            a_dim=8,
            nlayers=2,
            e_rcut=self.rcut,
            e_rcut_smth=self.rcut_smth,
            e_sel=nnei,
            a_rcut=self.rcut - 0.1,
            a_rcut_smth=self.rcut_smth,
            a_sel=nnei - 1,
            axis_neuron=4,
            num_edge_basis=10,
            num_angle_basis=8,
            precision=prec,
            type_map=["O", "H"],
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)

        dd.repflows.mean = torch.tensor(davg, dtype=dt, device=env.DEVICE)
        dd.repflows.stddev = torch.tensor(dstd, dtype=dt, device=env.DEVICE)
        return dd

    def test_forward_and_serialize(self) -> None:
        """Test forward pass + serialize/deserialize round-trip."""
        prec = "float64"
        dt = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        atol = 1e-8  # marginal GPU test cases
        nf, nloc, nnei = self.nlist.shape

        dd0 = self._make_descriptor(prec)
        rd0, _, _, _, _ = dd0(
            torch.tensor(self.coord_ext, dtype=dt, device=env.DEVICE),
            torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
            torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
            torch.tensor(self.mapping, dtype=int, device=env.DEVICE),
        )

        # Check output shape
        self.assertEqual(rd0.shape, (nf, nloc, 20))

        # Serialize round-trip
        dd1 = DescrptDPA3V2.deserialize(dd0.serialize())
        dd1 = dd1.to(env.DEVICE)
        rd1, _, _, _, _ = dd1(
            torch.tensor(self.coord_ext, dtype=dt, device=env.DEVICE),
            torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
            torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
            torch.tensor(self.mapping, dtype=int, device=env.DEVICE),
        )
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy(),
            rd1.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )

    def test_permutation_equivariance(self) -> None:
        """Test that permuting atoms gives permuted output."""
        prec = "float64"
        dt = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)

        dd0 = self._make_descriptor(prec)
        rd0, _, _, _, _ = dd0(
            torch.tensor(self.coord_ext, dtype=dt, device=env.DEVICE),
            torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
            torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
            torch.tensor(self.mapping, dtype=int, device=env.DEVICE),
        )

        rd0_np = rd0.detach().cpu().numpy()
        perm_local = self.perm[: self.nloc]
        np.testing.assert_allclose(
            rd0_np[0, perm_local, :],
            rd0_np[1, :, :],
            rtol=rtol,
            atol=1e-8,
            err_msg="Permutation equivariance failed",
        )

    def test_param_count(self) -> None:
        """Report parameter count for standard configuration."""
        dd = DescrptDPA3V2(
            ntypes=2,
            n_dim=128,
            e_dim=64,
            a_dim=32,
            nlayers=6,
            e_rcut=6.0,
            e_rcut_smth=5.0,
            e_sel=120,
            a_rcut=4.0,
            a_rcut_smth=3.5,
            a_sel=40,
            num_edge_basis=20,
            num_angle_basis=16,
            precision="float32",
            type_map=["O", "H"],
        )
        total = sum(p.numel() for p in dd.parameters())
        repflow_total = sum(p.numel() for p in dd.repflows.parameters())
        tebd_total = sum(p.numel() for p in dd.type_embedding.parameters())
        print(f"\n=== DPA3 V2 Parameter Count ===")
        print(f"Total:       {total:>12,}")
        print(f"  Repflows:  {repflow_total:>12,}")
        print(f"  TypeEmbed: {tebd_total:>12,}")


class TestDPA3V2Model(unittest.TestCase):
    """End-to-end model tests: rotation/translation invariance and smoothness."""

    @classmethod
    def setUpClass(cls):
        cls.model_config = {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "dpa3s_v2_rbf_norm",
                "n_dim": 20,
                "e_dim": 10,
                "a_dim": 8,
                "nlayers": 2,
                "e_rcut": 6.0,
                "e_rcut_smth": 5.0,
                "e_sel": 40,
                "a_rcut": 4.0,
                "a_rcut_smth": 3.5,
                "a_sel": 20,
                "axis_neuron": 4,
                "num_edge_basis": 10,
                "num_angle_basis": 8,
                "activation_function": "silu",
                "precision": "float64",
                "seed": GLOBAL_SEED,
            },
            "fitting_net": {
                "neuron": [24, 24],
                "resnet_dt": True,
                "precision": "float64",
                "seed": GLOBAL_SEED,
            },
            "data_stat_nbatch": 20,
        }
        cls.model = get_model(cls.model_config).to(env.DEVICE)
        cls.model.eval()

    @staticmethod
    def _make_water_system():
        """Create a small 3-atom water-like system (O, H, H) with a box."""
        # Positions in a box of size 10x10x10
        coord = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.96, 1.0],
                    [1.0, 1.0, 1.96],
                ]
            ],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        atype = torch.tensor([[0, 1, 1]], dtype=torch.int32, device=env.DEVICE)
        box = torch.tensor(
            [[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        return coord, atype, box

    @staticmethod
    def _random_rotation_matrix():
        """Generate a random SO(3) rotation matrix via QR decomposition."""
        rng = np.random.default_rng(42)
        m = rng.standard_normal((3, 3))
        q, r = np.linalg.qr(m)
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        return torch.tensor(q, dtype=torch.float64, device=env.DEVICE)

    def test_rotation_invariance(self) -> None:
        """Energy must be invariant under SO(3) rotation."""
        coord, atype, box = self._make_water_system()
        R = self._random_rotation_matrix()

        # Original
        result0 = self.model.forward_common(coord, atype, box=box)
        e0 = result0["energy"].detach().cpu().numpy()

        # Rotated coordinates: coord @ R^T
        coord_rot = torch.matmul(coord, R.T)
        # Rotated box
        box_rot = torch.matmul(box.reshape(1, 3, 3), R.T).reshape(1, 9)

        result1 = self.model.forward_common(coord_rot, atype, box=box_rot)
        e1 = result1["energy"].detach().cpu().numpy()

        np.testing.assert_allclose(
            e0, e1, atol=1e-8,
            err_msg="Rotation invariance of energy failed",
        )

    def test_translation_invariance(self) -> None:
        """Energy must be invariant under translation (with PBC)."""
        coord, atype, box = self._make_water_system()

        result0 = self.model.forward_common(coord, atype, box=box)
        e0 = result0["energy"].detach().cpu().numpy()

        # Translate by random vector, apply PBC
        shift = torch.tensor(
            [[[3.7, -2.1, 5.3]]],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        coord_shift = coord + shift
        # Wrap into box (orthorhombic box, diagonal 10)
        box_diag = torch.tensor(
            [[10.0, 10.0, 10.0]],
            dtype=torch.float64,
            device=env.DEVICE,
        ).unsqueeze(1)
        coord_shift = coord_shift - torch.floor(coord_shift / box_diag) * box_diag

        result1 = self.model.forward_common(coord_shift, atype, box=box)
        e1 = result1["energy"].detach().cpu().numpy()

        np.testing.assert_allclose(
            e0, e1, atol=1e-8,
            err_msg="Translation invariance of energy failed",
        )

    def test_force_finite_difference(self) -> None:
        """Validate forces = -dE/dR via finite difference.

        This tests smoothness: if the potential is smooth, autograd forces
        should match numerical forces from finite difference.
        """
        coord, atype, box = self._make_water_system()
        natom = coord.shape[1]
        eps = 1e-5

        # Get autograd forces
        coord_ad = coord.clone().requires_grad_(True)
        result = self.model.forward_common(coord_ad, atype, box=box)
        energy = result["energy"].sum()
        grad = torch.autograd.grad(energy, coord_ad, create_graph=False)[0]
        force_ad = -grad.detach().cpu().numpy().reshape(natom, 3)

        # Get numerical forces via central difference
        force_num = np.zeros((natom, 3), dtype=np.float64)
        for i in range(natom):
            for d in range(3):
                coord_p = coord.clone()
                coord_m = coord.clone()
                coord_p[0, i, d] += eps
                coord_m[0, i, d] -= eps
                ep = self.model.forward_common(coord_p, atype, box=box)["energy"]
                em = self.model.forward_common(coord_m, atype, box=box)["energy"]
                force_num[i, d] = -(ep.sum().item() - em.sum().item()) / (2 * eps)

        np.testing.assert_allclose(
            force_ad, force_num, rtol=1e-3, atol=1e-6,
            err_msg="Autograd forces vs finite difference mismatch (smoothness test)",
        )


if __name__ == "__main__":
    unittest.main()
