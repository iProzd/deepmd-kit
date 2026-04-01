# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor.dpa3s import DescrptDPA3S as DPDescrptDPA3S
from deepmd.pt.model.descriptor import (
    DescrptDPA3S,
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


class TestDescrptDPA3S(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_consistency(self) -> None:
        """Test PT vs dpmodel consistency through serialize/deserialize."""
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for prec, ect, add_chg_spin in [
            ("float64", False, False),
            ("float64", False, True),
        ]:
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            if prec == "float64":
                atol = 1e-8  # marginal GPU test cases

            dd0 = DescrptDPA3S(
                self.nt,
                n_dim=20,
                e_dim=10,
                a_dim=8,
                nlayers=3,
                e_rcut=self.rcut,
                e_rcut_smth=self.rcut_smth,
                e_sel=nnei,
                a_rcut=self.rcut - 0.1,
                a_rcut_smth=self.rcut_smth,
                a_sel=nnei - 1,
                axis_neuron=4,
                precision=prec,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                add_chg_spin_ebd=add_chg_spin,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)

            dd0.repflows.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)

            # Prepare fparam if needed
            fparam = None
            fparam_np = None
            if add_chg_spin:
                fparam = torch.tensor(
                    [[5, 1]], dtype=dtype, device=env.DEVICE
                ).expand(nf, -1)
                fparam_np = np.array([[5, 1]], dtype=np.float64).repeat(nf, axis=0)

            rd0, _, _, _, _ = dd0(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
                torch.tensor(self.mapping, dtype=int, device=env.DEVICE),
                fparam=fparam,
            )
            # serialization round-trip (PT -> PT)
            dd1 = DescrptDPA3S.deserialize(dd0.serialize())
            rd1, _, _, _, _ = dd1(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
                torch.tensor(self.mapping, dtype=int, device=env.DEVICE),
                fparam=fparam,
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd1.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=f"PT serialize round-trip failed (prec={prec}, ect={ect}, chg={add_chg_spin})",
            )
            # dp impl consistency (PT -> dpmodel)
            dd2 = DPDescrptDPA3S.deserialize(dd0.serialize())
            rd2, _, _, _, _ = dd2.call(
                self.coord_ext,
                self.atype_ext,
                self.nlist,
                self.mapping,
                fparam=fparam_np,
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd2,
                rtol=rtol,
                atol=atol,
                err_msg=f"PT vs dpmodel failed (prec={prec}, ect={ect}, chg={add_chg_spin})",
            )

    def test_jit(self) -> None:
        """Test that the descriptor can be TorchScript-compiled."""
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        prec = "float64"
        dtype = PRECISION_DICT[prec]

        dd0 = DescrptDPA3S(
            self.nt,
            n_dim=20,
            e_dim=10,
            a_dim=8,
            nlayers=3,
            e_rcut=self.rcut,
            e_rcut_smth=self.rcut_smth,
            e_sel=nnei,
            a_rcut=self.rcut - 0.1,
            a_rcut_smth=self.rcut_smth,
            a_sel=nnei - 1,
            axis_neuron=4,
            precision=prec,
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)

        dd0.repflows.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
        dd0.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
        model = torch.jit.script(dd0)

    def test_permutation_equivariance(self) -> None:
        """Test that permuting atoms gives permuted output."""
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        prec = "float64"
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)

        dd0 = DescrptDPA3S(
            self.nt,
            n_dim=20,
            e_dim=10,
            a_dim=8,
            nlayers=3,
            e_rcut=self.rcut,
            e_rcut_smth=self.rcut_smth,
            e_sel=nnei,
            a_rcut=self.rcut - 0.1,
            a_rcut_smth=self.rcut_smth,
            a_sel=nnei - 1,
            axis_neuron=4,
            precision=prec,
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)

        dd0.repflows.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
        dd0.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)

        # The test fixture provides two frames: original and permuted
        # nf == 2, frame[0] is original, frame[1] is permuted by self.perm
        rd0, _, _, _, _ = dd0(
            torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
            torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
            torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
            torch.tensor(self.mapping, dtype=int, device=env.DEVICE),
        )

        # rd0 shape: (nf, nloc, dim_out)
        rd0_np = rd0.detach().cpu().numpy()
        # frame 0 is original, frame 1 is permuted
        # The permuted output for local atoms should match:
        # rd0[1, :, :] should equal rd0[0, perm[:nloc], :]
        perm_local = self.perm[: self.nloc]
        np.testing.assert_allclose(
            rd0_np[0, perm_local, :],
            rd0_np[1, :, :],
            rtol=rtol,
            atol=1e-8,
            err_msg="Permutation equivariance failed",
        )


if __name__ == "__main__":
    unittest.main()
