# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import math
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Iterator,
)

import numpy as np

from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

log = logging.getLogger(__name__)

GNN_FIXED_SHAPE_NODE_ALIGN = 8
GNN_FIXED_SHAPE_EDGE_ALIGN = 64
GNN_FIXED_SHAPE_EDGE_WEIGHT = 2.0
GNN_FIXED_SHAPE_TOP_K = 20


class NeighborStat(ABC):
    """Abstract base class for getting training data information.

    It loads data from DeepmdData object, and measures the data info, including
    neareest nbor distance between atoms, max nbor size of atoms and the output
    data range of the environment matrix.

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
        self.rcut = rcut
        self.ntypes = ntypes
        self.mixed_type = mixed_type

    def get_stat(self, data: DeepmdDataSystem) -> tuple[float, np.ndarray]:
        """Get the data statistics of the training data, including nearest nbor distance between atoms, max nbor size of atoms.

        Parameters
        ----------
        data
            Class for manipulating many data systems. It is implemented with the help of DeepmdData.

        Returns
        -------
        min_nbor_dist
            The nearest distance between neighbor atoms
        max_nbor_size
            An array with ntypes integers, denotes the actual achieved max sel
        """
        min_nbor_dist = 100.0
        max_nbor_size = np.zeros(1 if self.mixed_type else self.ntypes, dtype=int)

        for mn, dt, jj in self.iterator(data):
            if np.isinf(dt):
                log.warning(
                    f"Atoms with no neighbors found in {jj}. Please make sure it's what you expected."
                )
            if dt < min_nbor_dist:
                if math.isclose(dt, 0.0, rel_tol=1e-6):
                    # it's unexpected that the distance between two atoms is zero
                    # zero distance will cause nan (#874)
                    raise RuntimeError(
                        f"Some atoms are overlapping in {jj}. Please check your"
                        " training data to remove duplicated atoms."
                    )
                min_nbor_dist = dt
            max_nbor_size = np.maximum(mn, max_nbor_size)

        # do sqrt in the final
        min_nbor_dist = math.sqrt(min_nbor_dist)
        log.info(
            f"Neighbor statistics: training data with minimal neighbor distance: {min_nbor_dist:f}"
        )
        log.info(
            f"Neighbor statistics: training data with maximum neighbor size: {max_nbor_size!s} (cutoff radius: {self.rcut:f})"
        )
        return min_nbor_dist, max_nbor_size

    def get_stat_with_edge(
        self, data: DeepmdDataSystem
    ) -> tuple[float, np.ndarray, int]:
        """Get the data statistics of the training data with edge counts.

        Parameters
        ----------
        data
            Class for manipulating many data systems. It is implemented with the help of DeepmdData.

        Returns
        -------
        min_nbor_dist
            The nearest distance between neighbor atoms
        max_nbor_size
            An array with ntypes integers, denotes the actual achieved max sel
        max_edge_size
            The maximal number of valid edges per frame
        """
        min_nbor_dist = 100.0
        max_nbor_size = np.zeros(1 if self.mixed_type else self.ntypes, dtype=int)
        max_edge_size = 0

        # === Step 1. Build a set-to-system mapping ===
        set_to_system_idx: dict[str, int] = {}
        system_set_dir: dict[int, str] = {}
        for system_idx, data_sys in enumerate(data.data_systems):
            for set_dir in data_sys.dirs:
                set_dir_str = str(set_dir)
                set_to_system_idx[set_dir_str] = system_idx
                system_set_dir.setdefault(system_idx, set_dir_str)
        per_system_max_edge = [0 for _ in range(len(data.data_systems))]

        # === Step 2. Scan sets and collect stats ===
        for mn, dt, me, jj in self.iterator_with_edge(data):
            if np.isinf(dt):
                log.warning(
                    f"Atoms with no neighbors found in {jj}. Please make sure it's what you expected."
                )
            if dt < min_nbor_dist:
                if math.isclose(dt, 0.0, rel_tol=1e-6):
                    raise RuntimeError(
                        f"Some atoms are overlapping in {jj}. Please check your"
                        " training data to remove duplicated atoms."
                    )
                min_nbor_dist = dt
            max_nbor_size = np.maximum(mn, max_nbor_size)
            if me > max_edge_size:
                max_edge_size = int(me)

            system_idx = set_to_system_idx.get(str(jj))
            if system_idx is not None:
                per_system_max_edge[system_idx] = max(
                    per_system_max_edge[system_idx], int(me)
                )

        # === Step 3. Summarize per-system stats ===
        per_system_stats: list[tuple[str, int, int, int]] = []
        for system_idx, data_sys in enumerate(data.data_systems):
            if system_idx in system_set_dir:
                system_label = _label_from_set_dir(system_set_dir[system_idx])
            else:
                system_label = _format_system_label(str(data.system_dirs[system_idx]))
            n_node = data_sys.get_natoms()
            max_edge = per_system_max_edge[system_idx]
            if max_edge <= 0:
                log.warning(
                    "GNN fixed-shape suggestion: system=%s max_edge=%d; clamping to 1.",
                    system_label,
                    max_edge,
                )
                max_edge = 1
            nframes = data_sys.nframes
            per_system_stats.append((system_label, n_node, max_edge, nframes))

        min_nbor_dist = math.sqrt(min_nbor_dist)
        log.info(
            f"Neighbor statistics: training data with minimal neighbor distance: {min_nbor_dist:f}"
        )
        log.info(
            f"Neighbor statistics: training data with maximum neighbor size: {max_nbor_size!s} (cutoff radius: {self.rcut:f})"
        )
        log.info(
            f"Neighbor statistics: maximum valid edge count per frame: {max_edge_size}"
        )
        if per_system_stats:
            log.info(
                "-----------------------------------------------------------------"
            )
            log.info("GNN System Stats Per System:")
            label_width = max(len(item[0]) for item in per_system_stats)
            node_width = len(str(max(item[1] for item in per_system_stats)))
            edge_width = len(str(max(item[2] for item in per_system_stats)))
            frame_width = len(str(max(item[3] for item in per_system_stats)))
            for system_label, n_node, max_edge, nframes in per_system_stats:
                log.info(
                    "System Stats for GNN: System=%-*s n_atom=%*d max_n_edge=%*d nframes=%*d",
                    label_width,
                    system_label,
                    node_width,
                    n_node,
                    edge_width,
                    max_edge,
                    frame_width,
                    nframes,
                )
            log_gnn_fixed_shape_suggestions(per_system_stats)
        return min_nbor_dist, max_nbor_size, max_edge_size

    @abstractmethod
    def iterator(
        self, data: DeepmdDataSystem
    ) -> Iterator[tuple[np.ndarray, float, str]]:
        """Abstract method for producing data.

        Yields
        ------
        mn : np.ndarray
            The maximal number of neighbors
        dt : float
            The squared minimal distance between two atoms
        jj : str
            The directory of the data system
        """

    def iterator_with_edge(
        self, data: DeepmdDataSystem
    ) -> Iterator[tuple[np.ndarray, float, int, str]]:
        """Iterator method producing neighbor statistics with edge counts.

        Yields
        ------
        mn : np.ndarray
            The maximal number of neighbors
        dt : float
            The squared minimal distance between two atoms
        me : int
            The maximal number of valid edges per frame
        jj : str
            The directory of the data system
        """
        raise NotImplementedError(
            "Edge statistics are not implemented for this backend."
        )


def _ceil_align(value: int, align: int) -> int:
    """Round up to the nearest aligned value."""
    if align <= 0:
        return value
    return math.ceil(value / float(align)) * align


def _format_system_label(system_dir: str) -> str:
    """Build a short label from a system directory path."""
    parts = system_dir.rstrip("/").split("/")
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1] if parts else system_dir


def _label_from_set_dir(set_dir: str) -> str:
    """Build a short label from a set.* directory path."""
    if "#" in set_dir:
        root_path, group_path = set_dir.split("#", 1)
        parts = [item for item in group_path.strip("/").split("/") if item]
        if parts and parts[-1].startswith("set."):
            parts = parts[:-1]
        if parts:
            return "/".join(parts[-2:])
        return _format_system_label(root_path)
    parts = set_dir.rstrip("/").split("/")
    if parts and parts[-1].startswith("set."):
        parts = parts[:-1]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1] if parts else set_dir


def log_gnn_fixed_shape_suggestions(
    per_system_stats: list[tuple[str, int, int, int]],
) -> None:
    """Log fixed-shape batching suggestions for GNN workloads."""
    if not per_system_stats:
        return

    # === Step 1. Prepare per-system arrays ===
    n_nodes = np.array([item[1] for item in per_system_stats], dtype=np.int64)
    max_edges = np.array([item[2] for item in per_system_stats], dtype=np.int64)
    if np.any(max_edges <= 0):
        log.warning(
            "GNN fixed-shape suggestion: max_edge contains non-positive values; clamping to 1 to avoid division by zero."
        )
        max_edges = np.maximum(max_edges, 1)

    # === Step 2. Compute minimal feasible limits ===
    max_node = int(np.max(n_nodes))
    max_edge = int(np.max(max_edges))
    edge_density = max_edges / n_nodes
    node_limit_min = _ceil_align(max_node, GNN_FIXED_SHAPE_NODE_ALIGN)
    # Use actual max edge count across all systems instead of extrapolation
    edge_limit_min = _ceil_align(max_edge, GNN_FIXED_SHAPE_EDGE_ALIGN)

    # === Step 3. Grid search candidates (1x~2x window) ===
    node_window = node_limit_min
    # Step size is simply the alignment size
    node_step_size = GNN_FIXED_SHAPE_NODE_ALIGN
    node_step_max = node_window // node_step_size + 1
    candidates: list[dict[str, float]] = []
    for node_step in range(node_step_max):
        node_limit = node_limit_min + node_step * node_step_size
        # Compute frames per system under node constraint only
        nframes_per_system = node_limit // n_nodes
        if np.any(nframes_per_system <= 0):
            continue
        # Compute weighted avg density using nframes as weights
        weights = nframes_per_system / float(np.sum(nframes_per_system))
        weighted_avg_density = float(np.sum(weights * edge_density))
        if weighted_avg_density <= 0:
            continue
        # Extrapolate edge_limit from node_limit and weighted density
        edge_limit = _ceil_align(
            math.ceil(node_limit * weighted_avg_density),
            GNN_FIXED_SHAPE_EDGE_ALIGN,
        )
        # Ensure edge_limit is at least edge_limit_min
        if edge_limit < edge_limit_min:
            edge_limit = edge_limit_min
        # Compute utilization using the same weights
        edge_used = nframes_per_system * n_nodes * edge_density
        edge_util = float(np.sum(weights * (edge_used / edge_limit)))
        exp_nf = float(np.sum(weights * nframes_per_system))
        candidates.append(
            {
                "node_limit": float(node_limit),
                "edge_limit": float(edge_limit),
                "edge_util": edge_util,
                "exp_nf": exp_nf,
            }
        )

    if not candidates:
        return

    # === Step 4. Rank candidates ===
    edge_sorted = sorted(
        candidates,
        key=lambda item: (-item["edge_util"], item["edge_limit"], item["node_limit"]),
    )
    node_width = len(str(int(max(item["node_limit"] for item in candidates))))
    edge_width = len(str(int(max(item["edge_limit"] for item in candidates))))

    # === Step 5. Log results ===
    log.info(
        "GNN Fixed-shape limits: minimal feasible (node_limit=%d, edge_limit=%d)",
        node_limit_min,
        edge_limit_min,
    )
    log.info("GNN Fixed-shape weights (prob_sys_size):")
    log.info(
        "GNN Fixed-shape search window: node_limit in [%d, %d].",
        node_limit_min,
        node_limit_min + (node_step_max - 1) * node_step_size,
    )
    log.info(
        "GNN Fixed-shape grid step: node_step=%d (aligned), edge_limit derived.",
        node_step_size,
    )
    # Deduplicate by edge_util (rounded to 1 decimal place), prefer smaller node_limit
    seen_util: dict[float, dict[str, float]] = {}
    for item in edge_sorted:
        util_rounded = round(item["edge_util"] * 100.0, 1)
        if util_rounded not in seen_util:
            seen_util[util_rounded] = item
        elif item["node_limit"] < seen_util[util_rounded]["node_limit"]:
            seen_util[util_rounded] = item
    # Sort by edge_util descending, then by node_limit ascending
    unique_candidates = sorted(
        seen_util.values(),
        key=lambda x: (-x["edge_util"], x["node_limit"]),
    )
    log.info(
        "GNN Fixed-shape Top%d (edge util deduplicated, avg density):",
        GNN_FIXED_SHAPE_TOP_K,
    )
    for item in unique_candidates[:GNN_FIXED_SHAPE_TOP_K]:
        log.info(
            "GNN Fixed-shape Edge: node_limit=%*d edge_limit=%*d exp_nf=%5.1f edge_util=%5.1f%%",
            node_width,
            int(item["node_limit"]),
            edge_width,
            int(item["edge_limit"]),
            item["exp_nf"],
            round(item["edge_util"] * 100.0, 1),
        )
