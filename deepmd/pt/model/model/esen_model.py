# SPDX-License-Identifier: LGPL-3.0-or-later
import functools

import torch
from typing import (
    Callable,
    Optional,
    Union,
)

from deepmd.pt.model.atomic_model import (
    BaseAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
import numpy as np
from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)
from deepmd.pt.utils.stat import (
    compute_output_stats,
)
import copy
from deepmd.utils.path import (
    DPPath,
)
from fairchem.core.common.registry import registry
from fairchem.core.models.esen.esen_dens import eSEN_DeNS_Backbone
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)
from deepmd.pt.utils import (
    env,
)
from ase import Atoms
from dpdata.periodic_table import ELEMENTS
from torch_geometric.data import Data
from fairchem.core.datasets import data_list_collater
from IPython import embed

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE


@BaseModel.register("esen")
class ESENModel(BaseModel):
    def __init__(
        self,
        model_params,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        model_config_copy = copy.deepcopy(model_params["esen"])
        model_name = model_config_copy.pop("name")
        self.default_fparam = model_config_copy.pop("default_fparam", [0.0, 1.0])
        self.numb_fparam = 2  # spin charge
        self.model = registry.get_model_class(model_name)(
            **model_config_copy,
        )
        self.model_config = model_config_copy
        self.type_map = model_params["type_map"]
        self.atype_to_idx = torch.tensor([ELEMENTS.index(i) for i in self.type_map], dtype=torch.int, device=device)
        ntypes = self.get_ntypes()
        self.max_out_size = 1
        self.bias_keys: list[str] = ["energy"]
        self.n_out = len(self.bias_keys)
        out_bias_data = torch.zeros(
            [self.n_out, ntypes, self.max_out_size], dtype=dtype, device=device
        )
        out_std_data = torch.ones(
            [self.n_out, ntypes, self.max_out_size], dtype=dtype, device=device
        )
        self.register_buffer("out_bias", out_bias_data)
        self.register_buffer("out_std", out_std_data)
        self.register_buffer(
            "default_fparam_tensor",
            torch.tensor(
                np.array(self.default_fparam), dtype=dtype, device=device
            ),
        )

    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.type_map

    def compute_or_load_stat(
        self,
        sampled_func,  # noqa: ANN001
        stat_file_path: Optional[DPPath] = None,
    ) -> None:
        """Compute or load the statistics parameters of the model.

        For example, mean and standard deviation of descriptors or the energy bias of
        the fitting net. When `sampled` is provided, all the statistics parameters will
        be calculated (or re-calculated for update), and saved in the
        `stat_file_path`(s). When `sampled` is not provided, it will check the existence
        of `stat_file_path`(s) and load the calculated statistics parameters.

        Parameters
        ----------
        sampled_func
            The sampled data frames from different data systems.
        stat_file_path
            The path to the statistics files.
        """
        if stat_file_path is not None and self.type_map is not None:
            # descriptors and fitting net with different type_map
            # should not share the same parameters
            stat_file_path /= " ".join(self.type_map)

        @functools.lru_cache
        def wrapped_sampler():
            sampled = sampled_func()
            return sampled

        self.compute_or_load_out_stat(wrapped_sampler, stat_file_path)

    def compute_or_load_out_stat(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        stat_file_path: Optional[DPPath] = None,
    ) -> None:
        """
        Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        stat_file_path : Optional[DPPath]
            The path to the stat file.

        """
        self.change_out_bias(
            merged,
            stat_file_path=stat_file_path,
            bias_adjust_mode="set-by-statistic",
        )

    def change_out_bias(
        self,
        sample_merged,
        stat_file_path: Optional[DPPath] = None,
        bias_adjust_mode="change-by-statistic",
    ) -> None:
        """Change the output bias according to the input data and the pretrained model.

        Parameters
        ----------
        sample_merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        bias_adjust_mode : str
            The mode for changing output bias : ['change-by-statistic', 'set-by-statistic']
            'change-by-statistic' : perform predictions on labels of target dataset,
                    and do least square on the errors to obtain the target shift as bias.
            'set-by-statistic' : directly use the statistic output bias in the target dataset.
        stat_file_path : Optional[DPPath]
            The path to the stat file.
        """
        if bias_adjust_mode == "set-by-statistic":
            bias_out, std_out = compute_output_stats(
                sample_merged,
                len(self.type_map),
                keys=['energy'],
                stat_file_path=stat_file_path,
                stats_distinguish_types=True,
                intensive=False,
            )
            self._store_out_stat(bias_out, std_out)
        else:
            raise RuntimeError("Unknown bias_adjust_mode mode: " + bias_adjust_mode)

    def _store_out_stat(
        self,
        out_bias: dict[str, torch.Tensor],
        out_std: dict[str, torch.Tensor],
        add: bool = False,
    ) -> None:
        ntypes = self.get_ntypes()
        out_bias_data = torch.clone(self.out_bias)
        out_std_data = torch.clone(self.out_std)
        for kk in out_bias.keys():
            assert kk in out_std.keys()
            idx = self._get_bias_index(kk)
            size = self._varsize([1])  # only energy
            if not add:
                out_bias_data[idx, :, :size] = out_bias[kk].view(ntypes, size)
            else:
                out_bias_data[idx, :, :size] += out_bias[kk].view(ntypes, size)
            out_std_data[idx, :, :size] = out_std[kk].view(ntypes, size)
        self.out_bias.copy_(out_bias_data)
        self.out_std.copy_(out_std_data)

    def _varsize(
        self,
        shape: list[int],
    ) -> int:
        output_size = 1
        len_shape = len(shape)
        for i in range(len_shape):
            output_size *= shape[i]
        return output_size

    def _get_bias_index(
        self,
        kk: str,
    ) -> int:
        res: list[int] = []
        for i, e in enumerate(self.bias_keys):
            if e == kk:
                res.append(i)
        assert len(res) == 1
        return res[0]


    @torch.jit.export
    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of developer implemented atomic models."""
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy",
                    shape=[1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
            ],
        )

    @torch.jit.export
    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.model.backbone.cutoff

    @torch.jit.export
    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.type_map

    @torch.jit.export
    def get_sel(self) -> list[int]:
        """Return the number of selected atoms for each type."""
        return [self.sel]

    @torch.jit.export
    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.numb_fparam

    @torch.jit.export
    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return 0

    @torch.jit.export
    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return []

    @torch.jit.export
    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False

    @torch.jit.export
    def mixed_types(self) -> bool:
        """Return whether the model is in mixed-types mode.

        If true, the model
        1. assumes total number of atoms aligned across frames;
        2. uses a neighbor list that does not distinguish different atomic types.
        If false, the model
        1. assumes total number of atoms of each atom type aligned across frames;
        2. uses a neighbor list that distinguishes different atomic types.
        """
        return True

    @torch.jit.export
    def has_message_passing(self) -> bool:
        """Return whether the descriptor has message passing."""
        return False

    def has_default_fparam(self) -> bool:
        return True

    def get_default_fparam(self) -> Optional[torch.Tensor]:
        return self.default_fparam_tensor

    @torch.jit.export
    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        coord : torch.Tensor
            The coordinates of atoms.
        atype : torch.Tensor
            The atomic types of atoms.
        box : torch.Tensor, optional
            The box tensor.
        fparam : torch.Tensor, optional
            The frame parameters.
        aparam : torch.Tensor, optional
            The atomic parameters.
        do_atomic_virial : bool, optional
            Whether to compute atomic virial.
        """
        nf, nloc = atype.shape
        if self.numb_fparam > 0 and fparam is None:
            # use default fparam
            assert self.default_fparam_tensor is not None
            fparam = torch.tile(self.default_fparam_tensor.unsqueeze(0), [nf, 1])
        atomic_number = self.atype_to_idx[atype.view(-1)].view(nf, nloc)
        tags = torch.zeros_like(atomic_number)
        pbc = torch.ones([3], dtype=torch.bool, device=coord.device)
        fixed_idx = torch.zeros_like(atomic_number)
        if box is None:
            box = torch.eye(3, dtype=coord.dtype, device=coord.device).view(1, 9).expand([nf, 9]) * 100.0
        # put the minimum data in torch geometric data object
        data_list = []
        for idx in range(nf):
            data = Data(
                cell=box[idx].view(1, 3, 3).float(),
                pos=coord[idx].view(nloc, 3).float(),
                atomic_numbers=atomic_number[idx],
                natoms=nloc,
                tags=tags[idx],
                pbc=pbc,
                fixed=fixed_idx[idx],
                charge=fparam[idx, 0].int(),
                spin=fparam[idx, 1].int(),
            )
            data_list.append(data)
        batch = data_list_collater(data_list, otf_graph=True).to(coord.device)
        model_ret = self.model(batch)

        # apply energy bias
        model_ret["energy"] = model_ret["energy"] + self.out_bias[0, :, 0][atype].sum(-1)

        model_predict = {}
        model_predict["energy"] = model_ret["energy"]
        model_predict["force"] = model_ret["forces"].view(nf, nloc, 3)
        model_predict["virial"] = model_ret["virial"].view(nf, 9)
        return model_predict

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward_lower_common(
        self,
        nloc: int,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,  # noqa: ARG002
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def serialize(self) -> dict:
        raise NotImplementedError

    @classmethod
    def deserialize(cls, data: dict):
        raise NotImplementedError

    @torch.jit.export
    def get_nnei(self) -> int:
        """Return the total number of selected neighboring atoms in cut-off radius."""
        raise NotImplementedError

    @torch.jit.export
    def get_nsel(self) -> int:
        """Return the total number of selected neighboring atoms in cut-off radius."""
        raise NotImplementedError

    @classmethod
    def update_sel(
        cls,
        train_data,
        type_map: Optional[list[str]],
        local_jdata: dict,
    ) -> tuple[dict, Optional[float]]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statictics
        type_map : list[str], optional
            The name of each type of atoms
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        float
            The minimum distance between two atoms
        """
        raise NotImplementedError

    @torch.jit.export
    def model_output_type(self) -> list[str]:
        """Get the output type for the model."""
        return ["energy"]

    def translated_output_def(self):
        """Get the translated output def for the model."""
        raise NotImplementedError

    def model_output_def(self):
        """Get the output def for the model."""
        raise NotImplementedError

    @classmethod
    def get_model(cls, model_params: dict):
        """Get the model by the parameters.

        Parameters
        ----------
        model_params : dict
            The model parameters

        Returns
        -------
        BaseBaseModel
            The model
        """
        model_config_copy = copy.deepcopy(model_params["esen"])
        model_name = model_config_copy.pop("name")
        return registry.get_model_class(model_name)(
            **model_config_copy,
        )
