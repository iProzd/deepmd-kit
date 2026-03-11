# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for min_pair_dist frame filtering."""

import unittest

import numpy as np

from deepmd.pt.utils.dist_check import (
    compute_min_pair_dist_single,
)


class TestComputeMinPairDistSingle(unittest.TestCase):
    """Test minimum pairwise distance computation."""

    def test_three_atoms_no_pbc(self) -> None:
        """Three atoms, closest pair is 0.3 Å."""
        coord = np.array(
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.3,
                0.0,
                0.0,
            ]
        )
        atype = np.array([0, 0, 1])
        dist = compute_min_pair_dist_single(coord, box=None, atype=atype)
        np.testing.assert_almost_equal(dist, 0.3)

    def test_pbc_minimum_image(self) -> None:
        """Two atoms near opposite edges of a 10 Å cubic box.

        Real-space distance is 9.0 Å, but minimum image distance is 1.0 Å.
        """
        coord = np.array([0.5, 5.0, 5.0, 9.5, 5.0, 5.0])
        box = np.array([10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0])
        atype = np.array([0, 0])
        dist = compute_min_pair_dist_single(coord, box=box, atype=atype)
        np.testing.assert_almost_equal(dist, 1.0)

    def test_pbc_triclinic(self) -> None:
        """Triclinic box with atoms near boundary."""
        # Triclinic box: a=(10,0,0), b=(2,10,0), c=(0,0,10)
        box = np.array([10.0, 0, 0, 2.0, 10.0, 0, 0, 0, 10.0])
        # Atom at (0.5, 5, 5) and (9.5, 5, 5)
        # In fractional: (0.05, 0.49, 0.5) and (0.95, 0.31, 0.5)
        # Fractional diff along a: 0.9 → min image: -0.1
        # Cartesian diff: -0.1*(10,0,0) + delta_b*(2,10,0) = (-1, ..., 0)
        # Use simple case: two atoms along a-axis
        coord = np.array([0.2, 0.0, 0.0, 9.8, 0.0, 0.0])
        atype = np.array([0, 0])
        dist = compute_min_pair_dist_single(coord, box=box, atype=atype)
        # Fractional coords: (0.02, 0, 0) and (0.98, 0, 0)
        # Frac diff: 0.96 → min image: -0.04
        # Cartesian: -0.04 * (10, 0, 0) = (-0.4, 0, 0) → dist = 0.4
        np.testing.assert_almost_equal(dist, 0.4, decimal=5)

    def test_virtual_atoms_excluded(self) -> None:
        """Virtual atoms (type < 0) should be excluded."""
        # Real atoms are 2.0 Å apart, virtual atom is 0.1 Å from first
        coord = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
            ]
        )
        atype = np.array([0, -1, 1])  # second atom is virtual
        dist = compute_min_pair_dist_single(coord, box=None, atype=atype)
        np.testing.assert_almost_equal(dist, 2.0)

    def test_single_real_atom(self) -> None:
        """Only one real atom → should return inf."""
        coord = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        atype = np.array([0, -1])
        dist = compute_min_pair_dist_single(coord, box=None, atype=atype)
        self.assertEqual(dist, float("inf"))

    def test_all_virtual(self) -> None:
        """All virtual atoms → should return inf."""
        coord = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        atype = np.array([-1, -1])
        dist = compute_min_pair_dist_single(coord, box=None, atype=atype)
        self.assertEqual(dist, float("inf"))

    def test_coord_shape_2d(self) -> None:
        """Accept (natoms, 3) shaped coord."""
        coord = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
        atype = np.array([0, 1])
        dist = compute_min_pair_dist_single(coord, box=None, atype=atype)
        np.testing.assert_almost_equal(dist, 0.8)


if __name__ == "__main__":
    unittest.main()
