# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Iterator,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.utils.nlist import (
    extend_coord_with_ghosts,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.neighbor_stat import NeighborStat as BaseNeighborStat


class NeighborStatOP(NativeOP):
    """Class for getting neighbor statistics data information.

    Parameters
    ----------
    ntypes
        The num of atom types
    rcut
        The cut-off radius
    mixed_types : bool, optional
        If True, treat all types as a single type.
    """

    def __init__(
        self,
        ntypes: int,
        rcut: float,
        mixed_types: bool,
    ) -> None:
        self.rcut = rcut
        self.ntypes = ntypes
        self.mixed_types = mixed_types

    def call(
        self,
        coord: Array,
        atype: Array,
        cell: Array | None,
    ) -> tuple[Array, Array]:
        """Calculate the neareest neighbor distance between atoms, maximum nbor size of
        atoms and the output data range of the environment matrix.

        Parameters
        ----------
        coord
            The coordinates of atoms.
        atype
            The atom types.
        cell
            The cell.

        Returns
        -------
        float
            The minimal squared distance between two atoms
        np.ndarray
            The maximal number of neighbors
        """
        min_rr2, nnei = self._compute_nnei(coord, atype, cell)
        max_nnei = array_api_compat.array_namespace(nnei).max(nnei, axis=1)
        return min_rr2, max_nnei

    def call_with_edge_stats(
        self,
        coord: Array,
        atype: Array,
        cell: Array | None,
    ) -> tuple[Array, Array, Array]:
        """Calculate neighbor statistics with per-frame edge counts.

        Parameters
        ----------
        coord
            The coordinates of atoms.
        atype
            The atom types.
        cell
            The cell.

        Returns
        -------
        Array
            The minimal squared distance between two atoms
        Array
            The maximal number of neighbors
        Array
            The total number of valid edges per frame
        """
        # === Step 1. Compute per-atom neighbor counts ===
        min_rr2, nnei = self._compute_nnei(coord, atype, cell)
        xp = array_api_compat.array_namespace(nnei)

        # === Step 2. Reduce to per-frame totals ===
        max_nnei = xp.max(nnei, axis=1)
        edge_per_frame = xp.sum(xp.sum(nnei, axis=-1), axis=1)
        return min_rr2, max_nnei, edge_per_frame

    def _compute_nnei(
        self,
        coord: Array,
        atype: Array,
        cell: Array | None,
    ) -> tuple[Array, Array]:
        """Compute minimal distance and per-atom neighbor counts.

        Parameters
        ----------
        coord
            The coordinates of atoms.
        atype
            The atom types.
        cell
            The cell.

        Returns
        -------
        Array
            The minimal squared distance between two atoms, shape (nframes, nloc)
        Array
            The neighbor counts per atom, shape (nframes, nloc, ntypes or 1)
        """
        xp = array_api_compat.array_namespace(coord, atype)
        nframes = coord.shape[0]
        coord = xp.reshape(coord, (nframes, -1, 3))
        nloc = coord.shape[1]
        coord = xp.reshape(coord, (nframes, nloc * 3))
        extend_coord, extend_atype, _ = extend_coord_with_ghosts(
            coord, atype, cell, self.rcut
        )

        # === Step 1. Pairwise distances ===
        coord1 = xp.reshape(extend_coord, (nframes, -1))
        nall = coord1.shape[1] // 3
        coord0 = coord1[:, : nloc * 3]
        diff = (
            xp.reshape(coord1, (nframes, -1, 3))[:, None, :, :]
            - xp.reshape(coord0, (nframes, -1, 3))[:, :, None, :]
        )
        assert list(diff.shape) == [nframes, nloc, nall, 3]

        # === Step 2. Mask self pairs ===
        mask = xp.eye(nloc, nall, dtype=xp.bool, device=array_api_compat.device(diff))
        mask = xp.tile(mask[None, :, :, None], (nframes, 1, 1, 3))
        diff = xp.where(mask, xp.full_like(diff, xp.inf), diff)
        rr2 = xp.sum(xp.square(diff), axis=-1)
        min_rr2 = xp.min(rr2, axis=-1)

        # === Step 3. Count neighbors within rcut ===
        if not self.mixed_types:
            mask = rr2 < self.rcut**2
            nneis = []
            for ii in range(self.ntypes):
                nneis.append(xp.sum(mask & (extend_atype == ii)[:, None, :], axis=-1))
            nnei = xp.stack(nneis, axis=-1)
        else:
            mask = rr2 < self.rcut**2
            # virtual type (<0) are not counted
            nnei = xp.sum(mask & (extend_atype >= 0)[:, None, :], axis=-1)
            nnei = xp.reshape(nnei, (nframes, nloc, 1))

        return min_rr2, nnei


class NeighborStat(BaseNeighborStat):
    """Neighbor statistics using pure NumPy.

    Parameters
    ----------
    ntypes : int
        The num of atom types
    rcut : float
        The cut-off radius
    mixed_type : bool, optional, default=False
        Treat all types as a single type.
    """

    def __init__(
        self,
        ntypes: int,
        rcut: float,
        mixed_type: bool = False,
    ) -> None:
        super().__init__(ntypes, rcut, mixed_type)
        self.op = NeighborStatOP(ntypes, rcut, mixed_type)

    def iterator(
        self, data: DeepmdDataSystem
    ) -> Iterator[tuple[np.ndarray, float, str]]:
        """Abstract method for producing data.

        Yields
        ------
        np.ndarray
            The maximal number of neighbors
        float
            The squared minimal distance between two atoms
        str
            The directory of the data system
        """
        for ii in range(len(data.system_dirs)):
            for jj in data.data_systems[ii].dirs:
                data_set = data.data_systems[ii]
                data_set_data = data_set._load_set(jj)
                minrr2, max_nnei = self.op(
                    data_set_data["coord"],
                    data_set_data["type"],
                    data_set_data["box"] if data_set.pbc else None,
                )
                yield np.max(max_nnei, axis=0), np.min(minrr2), jj

    def iterator_with_edge(
        self, data: DeepmdDataSystem
    ) -> Iterator[tuple[np.ndarray, float, int, str]]:
        """Iterator method producing neighbor statistics with edge counts.

        Yields
        ------
        np.ndarray
            The maximal number of neighbors
        float
            The squared minimal distance between two atoms
        int
            The maximal number of valid edges per frame
        str
            The directory of the data system
        """
        for ii in range(len(data.system_dirs)):
            for jj in data.data_systems[ii].dirs:
                data_set = data.data_systems[ii]
                data_set_data = data_set._load_set(jj)
                minrr2, max_nnei, edge_per_frame = self.op.call_with_edge_stats(
                    data_set_data["coord"],
                    data_set_data["type"],
                    data_set_data["box"] if data_set.pbc else None,
                )
                max_edge = int(np.max(edge_per_frame))
                yield np.max(max_nnei, axis=0), np.min(minrr2), max_edge, jj
