# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
from multiprocessing.dummy import (
    Pool,
)
from typing import (
    Any,
)

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from torch.utils.data._utils.collate import (
    collate_tensor_fn,
)
from torch.utils.data.distributed import (
    DistributedSampler,
)

from deepmd.dpmodel.utils.neighbor_stat import (
    NeighborStatOP,
)
from deepmd.pt.modifier import (
    BaseModifier,
)
from deepmd.pt.utils import (
    dp_random,
    env,
)
from deepmd.pt.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.pt.utils.dataset import (
    DeepmdDataSetForLoader,
)
from deepmd.pt.utils.utils import (
    mix_entropy,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.data_system import (
    print_summary,
    prob_sys_size_ext,
    process_sys_probs,
)

log = logging.getLogger(__name__)
torch.multiprocessing.set_sharing_strategy("file_system")


def setup_seed(seed: int | list[int] | tuple[int, ...]) -> None:
    if isinstance(seed, (list, tuple)):
        mixed_seed = mix_entropy(seed)
    else:
        mixed_seed = seed
    torch.manual_seed(mixed_seed)
    torch.cuda.manual_seed_all(mixed_seed)
    torch.backends.cudnn.deterministic = True
    dp_random.seed(seed)


def _format_system_label(system_dir: str) -> str:
    """Build a short label from a system directory path."""
    parts = system_dir.rstrip("/").split("/")
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1] if parts else system_dir


class DpLoaderSet(Dataset):
    """A dataset for storing DataLoaders to multiple Systems.

    Parameters
    ----------
    sys_path
            Path to the data system
    batch_size
            Max frame count in a batch.
    type_map
            Gives the name of different atom types
    gnn_batch_info
            Optional graph batch info with keys: n_node, n_edge, rcut.
    seed
            Random seed for dataloader
    shuffle
            If the data are shuffled (Only effective in serial mode. Always shuffle in distributed data parallelism)
    """

    def __init__(
        self,
        systems: str | list[str],
        batch_size: int,
        type_map: list[str] | None,
        seed: int | None = None,
        shuffle: bool = True,
        modifier: BaseModifier | None = None,
        gnn_batch_info: dict[str, int | float] | None = None,
    ) -> None:
        if seed is not None:
            setup_seed(seed)
        if isinstance(systems, str):
            with h5py.File(systems) as file:
                systems = [os.path.join(systems, item) for item in file.keys()]

        def construct_dataset(system: str) -> DeepmdDataSetForLoader:
            return DeepmdDataSetForLoader(
                system=system,
                type_map=type_map,
                modifier=modifier,
            )

        self.systems: list[DeepmdDataSetForLoader] = []
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        if global_rank == 0:
            log.info(f"Constructing DataLoaders from {len(systems)} systems")
            with Pool(max(1, env.NUM_WORKERS)) as pool:
                self.systems = pool.map(construct_dataset, systems)
        else:
            self.systems = [None] * len(systems)  # type: ignore
        if dist.is_initialized():
            dist.broadcast_object_list(self.systems)
            assert self.systems[-1] is not None
        self.sampler_list: list[DistributedSampler] = []
        self.index = []
        self.total_batch = 0
        self.max_edges: list[int] | None = None

        self.dataloaders = []
        self.batch_sizes = []
        if isinstance(batch_size, str):
            if batch_size == "gnnmax" or batch_size.startswith("gnnmax:"):
                # === Parse node/edge limits ===
                if batch_size == "gnnmax":
                    # Auto mode: read n_node/n_edge from model config
                    if gnn_batch_info is None:
                        raise ValueError(
                            "batch_size='gnnmax' requires gnn_batch_info from model"
                        )
                    if "n_node" not in gnn_batch_info or "n_edge" not in gnn_batch_info:
                        raise ValueError(
                            "batch_size='gnnmax' requires model to have n_node and n_edge parameters"
                        )
                    node_limit = int(gnn_batch_info["n_node"])
                    edge_limit = int(gnn_batch_info["n_edge"])
                    if node_limit <= 0 or edge_limit <= 0:
                        raise ValueError(
                            "batch_size='gnnmax' requires positive n_node and n_edge in model"
                        )
                else:
                    # Explicit mode: gnnmax:node,edge
                    rule = batch_size.split(":", 1)[1]
                    if "," not in rule:
                        raise ValueError(
                            "gnnmax requires format gnnmax:node,edge with positive integers"
                        )
                    node_str, edge_str = rule.split(",", 1)
                    node_limit = int(node_str)
                    edge_limit = int(edge_str)
                    if node_limit <= 0 or edge_limit <= 0:
                        raise ValueError(
                            "gnnmax requires positive node and edge limits in gnnmax:node,edge"
                        )

                    # Validate against model params if provided
                    if "n_node" in gnn_batch_info and "n_edge" in gnn_batch_info:
                        model_node = int(gnn_batch_info.get("n_node", 0))
                        model_edge = int(gnn_batch_info.get("n_edge", 0))
                        if model_node <= 0 or model_edge <= 0:
                            raise ValueError(
                                "gnnmax requires positive n_node and n_edge in gnn_batch_info"
                            )
                        if model_node != node_limit or model_edge != edge_limit:
                            raise ValueError(
                                "gnnmax node/edge must match model n_node/n_edge when provided"
                            )

                if gnn_batch_info is None or "rcut" not in gnn_batch_info:
                    raise ValueError("gnnmax requires rcut in gnn_batch_info")
                rcut = float(gnn_batch_info["rcut"])

                # === Step 1. Compute per-system max edge count ===
                max_edges = [
                    max(1, compute_max_edge_per_frame(system, rcut))
                    for system in self.systems
                ]
                self.max_edges = max_edges
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # === Step 2. Validate fixed-shape feasibility ===
                max_nodes = [system._natoms for system in self.systems]
                max_node = max(max_nodes)
                max_edge = max(max_edges)
                if node_limit < max_node or edge_limit < max_edge:
                    idx_node = (
                        max_nodes.index(max_node) if node_limit < max_node else None
                    )
                    idx_edge = (
                        max_edges.index(max_edge) if edge_limit < max_edge else None
                    )
                    idx = idx_node if idx_node is not None else idx_edge
                    system = self.systems[idx]
                    system_label = _format_system_label(system.system)
                    raise ValueError(
                        "gnnmax node/edge limits are too small for fixed-shape batching: "
                        f"system={system_label} n_node={system._natoms} max_edge={max_edges[idx]} "
                        f"node_limit={node_limit} edge_limit={edge_limit}. "
                        f"Increase gnnmax:node,edge to at least node_limit>={max_node}, "
                        f"edge_limit>={max_edge}."
                    )

                # === Step 3. Determine batch size by node/edge limits ===
                for system, max_edge in zip(self.systems, max_edges):
                    ni = system._natoms
                    bsi_node = node_limit // ni
                    bsi_edge = edge_limit // max_edge
                    if bsi_node == 0 or bsi_edge == 0:
                        system_label = _format_system_label(system.system)
                        raise ValueError(
                            "gnnmax node/edge limits are too small for fixed-shape batching: "
                            f"system={system_label} n_node={ni} max_edge={max_edge} "
                            f"node_limit={node_limit} edge_limit={edge_limit}. "
                            f"Increase gnnmax:node,edge to at least node_limit>={max_node}, "
                            f"edge_limit>={max_edge}."
                        )
                    bsi = min(bsi_node, bsi_edge)
                    self.batch_sizes.append(bsi)
            else:
                if batch_size == "auto":
                    rule = 32
                    ceiling = True
                elif batch_size.startswith("auto:"):
                    rule = int(batch_size.split(":")[1])
                    ceiling = True
                elif batch_size.startswith("max:"):
                    rule = int(batch_size.split(":")[1])
                    ceiling = False
                elif batch_size.startswith("filter:"):
                    # remove system with more than `filter` atoms
                    rule = int(batch_size.split(":")[1])
                    len_before = len(self.systems)
                    self.systems = [
                        system for system in self.systems if system._natoms <= rule
                    ]
                    len_after = len(self.systems)
                    if len_before != len_after:
                        log.warning(
                            f"Remove {len_before - len_after} systems with more than {rule} atoms"
                        )
                    if len(self.systems) == 0:
                        raise ValueError(
                            f"No system left after removing systems with more than {rule} atoms"
                        )
                    ceiling = False
                else:
                    raise ValueError(f"Unsupported batch size rule: {batch_size}")
                for ii in self.systems:
                    ni = ii._natoms
                    bsi = rule // ni
                    if ceiling:
                        if bsi * ni < rule:
                            bsi += 1
                    else:
                        if bsi == 0:
                            bsi = 1
                    self.batch_sizes.append(bsi)
        elif isinstance(batch_size, list):
            self.batch_sizes = batch_size
        else:
            self.batch_sizes = batch_size * np.ones(len(systems), dtype=int)
        assert len(self.systems) == len(self.batch_sizes)
        for system, batch_size in zip(self.systems, self.batch_sizes):
            if dist.is_available() and dist.is_initialized():
                system_sampler = DistributedSampler(system)
                self.sampler_list.append(system_sampler)
            else:
                system_sampler = None
            system_dataloader = DataLoader(
                dataset=system,
                batch_size=int(batch_size),
                num_workers=0,  # Should be 0 to avoid too many threads forked
                sampler=system_sampler,
                collate_fn=collate_batch,
                shuffle=(
                    not (dist.is_available() and dist.is_initialized())
                )  # distributed sampler will do the shuffling by default
                and shuffle,
            )
            self.dataloaders.append(system_dataloader)
            self.index.append(len(system_dataloader))
            self.total_batch += len(system_dataloader)
        # Initialize iterator instances for DataLoader
        self.iters = []
        with torch.device("cpu"):
            for item in self.dataloaders:
                self.iters.append(iter(item))

    def set_noise(self, noise_settings: dict[str, Any]) -> None:
        # noise_settings['noise_type'] # "trunc_normal", "normal", "uniform"
        # noise_settings['noise'] # float, default 1.0
        # noise_settings['noise_mode'] # "prob", "fix_num"
        # noise_settings['mask_num'] # if "fix_num", int
        # noise_settings['mask_prob'] # if "prob", float
        # noise_settings['same_mask'] # coord and type same mask?
        for system in self.systems:
            system.set_noise(noise_settings)

    def __len__(self) -> int:
        return len(self.dataloaders)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # log.warning(str(torch.distributed.get_rank())+" idx: "+str(idx)+" index: "+str(self.index[idx]))
        with torch.device("cpu"):
            try:
                batch = next(self.iters[idx])
            except StopIteration:
                self.iters[idx] = iter(self.dataloaders[idx])
                batch = next(self.iters[idx])
        batch["sid"] = idx
        return batch

    def add_data_requirement(self, data_requirement: list[DataRequirementItem]) -> None:
        """Add data requirement for each system in multiple systems."""
        for system in self.systems:
            system.add_data_requirement(data_requirement)

    def print_summary(
        self,
        name: str,
        prob: list[float],
    ) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print_summary(
                name,
                len(self.systems),
                [ss.system for ss in self.systems],
                [ss._natoms for ss in self.systems],
                self.batch_sizes,
                [
                    ss._data_system.get_sys_numb_batch(self.batch_sizes[ii])
                    for ii, ss in enumerate(self.systems)
                ],
                prob,
                [ss._data_system.pbc for ss in self.systems],
                e_max=self.max_edges,
            )

    def preload_and_modify_all_data_torch(self) -> None:
        for system in self.systems:
            system.preload_and_modify_all_data_torch()


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    example = batch[0]
    result = {}
    for key in example.keys():
        if "find_" in key:
            result[key] = batch[0][key]
        else:
            if batch[0][key] is None:
                result[key] = None
            elif key == "fid":
                result[key] = [d[key] for d in batch]
            elif key == "type":
                continue
            else:
                result[key] = collate_tensor_fn(
                    [torch.as_tensor(d[key]) for d in batch]
                )
    return result


def get_weighted_sampler(
    training_data: Any, prob_style: str, sys_prob: bool = False
) -> WeightedRandomSampler:
    if sys_prob is False:
        if prob_style == "prob_uniform":
            prob_v = 1.0 / float(training_data.__len__())
            probs = [prob_v for ii in range(training_data.__len__())]
        else:  # prob_sys_size;A:B:p1;C:D:p2 or prob_sys_size = prob_sys_size;0:nsys:1.0
            if prob_style == "prob_sys_size":
                style = f"prob_sys_size;0:{len(training_data)}:1.0"
            else:
                style = prob_style
            probs = prob_sys_size_ext(style, len(training_data), training_data.index)
    else:
        probs = process_sys_probs(prob_style, training_data.index)
    log.debug("Generated weighted sampler with prob array: " + str(probs))
    # training_data.total_batch is the size of one epoch, you can increase it to avoid too many  rebuilding of iterators
    len_sampler = training_data.total_batch * max(env.NUM_WORKERS, 1)
    with torch.device("cpu"):
        sampler = WeightedRandomSampler(
            probs,
            len_sampler,
            replacement=True,
        )
    return sampler


def get_sampler_from_params(_data: Any, _params: dict[str, Any]) -> Any:
    if (
        "sys_probs" in _params and _params["sys_probs"] is not None
    ):  # use sys_probs first
        _sampler = get_weighted_sampler(
            _data,
            _params["sys_probs"],
            sys_prob=True,
        )
    elif "auto_prob" in _params:
        _sampler = get_weighted_sampler(_data, _params["auto_prob"])
    else:
        _sampler = get_weighted_sampler(_data, "prob_sys_size")
    return _sampler


def compute_max_edge_per_frame(system: DeepmdDataSetForLoader, rcut: float) -> int:
    """
    Compute the maximal number of valid edges per frame for a system.

    Parameters
    ----------
    system : DeepmdDataSetForLoader
        The data system wrapper.
    rcut : float
        The cutoff radius in Angstrom.

    Returns
    -------
    int
        The maximal number of valid edges per frame.
    """
    data = system._data_system
    ntypes = data.get_ntypes()
    op = NeighborStatOP(ntypes, rcut, data.mixed_type)
    auto_batch_size = AutoBatchSize()
    max_edge = 0

    def _execute_with_edge(
        coord: np.ndarray,
        atype: np.ndarray,
        cell: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        minrr2, max_nnei, edge_per_frame = op.call_with_edge_stats(
            torch.from_numpy(coord).to(env.DEVICE),
            torch.from_numpy(atype).to(env.DEVICE),
            torch.from_numpy(cell).to(env.DEVICE) if cell is not None else None,
        )
        minrr2 = minrr2.detach().cpu().numpy()
        max_nnei = max_nnei.detach().cpu().numpy()
        edge_per_frame = edge_per_frame.detach().cpu().numpy()
        return minrr2, max_nnei, edge_per_frame

    # Suppress auto batch size logs
    batch_logger = logging.getLogger("deepmd.utils.batch_size")
    old_level = batch_logger.level
    batch_logger.setLevel(logging.WARNING)
    try:
        # Iterate sets and collect max edge count
        for set_dir in data.dirs:
            data_set = data._load_set(set_dir)
            _, _, edge_per_frame = auto_batch_size.execute_all(
                _execute_with_edge,
                data_set["coord"].shape[0],
                data.get_natoms(),
                data_set["coord"],
                data_set["type"],
                data_set["box"] if data.pbc else None,
            )
            max_edge = max(max_edge, int(np.max(edge_per_frame)))
    finally:
        batch_logger.setLevel(old_level)

    system_label = "/".join(system.system.rstrip("/").split("/")[-2:])
    log.debug(
        "System Stats for GNN: System=%s n_atom=%d max_n_edge=%d",
        system_label,
        data.get_natoms(),
        max_edge,
    )
    return max_edge
