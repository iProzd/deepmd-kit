# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptDPA3S,
)

from ...seed import (
    GLOBAL_SEED,
)
from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class TestDescrptDPA3S(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(self) -> None:
        """Test dpmodel serialize/deserialize round-trip."""
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        em0 = DescrptDPA3S(
            ntypes=self.nt,
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
            precision="float64",
            seed=GLOBAL_SEED,
        )

        em0.repflows.mean = davg
        em0.repflows.stddev = dstd
        em1 = DescrptDPA3S.deserialize(em0.serialize())
        mm0 = em0.call(self.coord_ext, self.atype_ext, self.nlist, self.mapping)
        mm1 = em1.call(self.coord_ext, self.atype_ext, self.nlist, self.mapping)
        desired_shape = [
            (nf, nloc, em0.get_dim_out()),  # descriptor
            (nf, nloc, em0.get_dim_emb(), 3),  # rot_mat
        ]
        for ii in [0, 1]:
            np.testing.assert_equal(mm0[ii].shape, desired_shape[ii])
            np.testing.assert_allclose(
                mm0[ii],
                mm1[ii],
                err_msg=f"Output {ii} mismatch after serialize round-trip",
            )

    def test_permutation_equivariance(self) -> None:
        """Test that permuting atoms gives permuted output."""
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        em0 = DescrptDPA3S(
            ntypes=self.nt,
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
            precision="float64",
            seed=GLOBAL_SEED,
        )

        em0.repflows.mean = davg
        em0.repflows.stddev = dstd
        mm0 = em0.call(self.coord_ext, self.atype_ext, self.nlist, self.mapping)

        # nf == 2: frame[0] original, frame[1] permuted
        rd0 = mm0[0]  # descriptor, shape (nf, nloc, dim_out)
        perm_local = self.perm[: self.nloc]
        np.testing.assert_allclose(
            rd0[0, perm_local, :],
            rd0[1, :, :],
            atol=1e-8,
            err_msg="Permutation equivariance failed for dpmodel",
        )


if __name__ == "__main__":
    unittest.main()
