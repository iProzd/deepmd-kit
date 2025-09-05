"""SevenNet model implementation for DeePMD-kit."""

import torch
import numpy as np
from torch_geometric.loader.dataloader import Collater

from typing import Any, Optional, List, Union, Tuple
from copy import deepcopy
import json
# deepmd-kit
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.model.model import BaseModel
from deepmd.pt.utils import env
from deepmd.pt.utils.stat import compute_output_stats
from deepmd.pt.utils.update_sel import UpdateSel
from deepmd.pt.utils.utils import to_numpy_array, to_torch_tensor
from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.utils.path import DPPath
from deepmd.utils.version import check_version_compatibility

# ase and sevennet
import sevenn._keys as KEY
from sevenn.model_build import build_E3_equivariant_model
from sevenn.train.dataload import unlabeled_atoms_to_graph
from sevenn.atom_graph_data import AtomGraphData
from ase import Atoms
import os
import pickle

ELEMENTS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

PeriodicTable = {
    **{ee: ii + 1 for ii, ee in enumerate(ELEMENTS)},
    **{f"m{ee}": ii + 1 for ii, ee in enumerate(ELEMENTS)},
    "HW": 1,
    "OW": 8,
}


def deepmd_to_ase_atoms_list(
        coord: torch.Tensor,  # (batch_size, n_atoms, 3)
        atype: torch.Tensor,  # (batch_size, n_atoms)
        cell: Optional[torch.Tensor],  # (batch_size, 9)
        type_map: List[str]  # ['O', 'H']
) -> List[Atoms]:
    """
    dp data to ase atoms list
    """
    batch_size, n_atoms = atype.shape
    atoms_list = []

    for batch_idx in range(batch_size):
        frame_coord = coord[batch_idx].detach().cpu().numpy()  # gradient break
        frame_atype = atype[batch_idx].detach().cpu().numpy()  # gradient break
        symbols = [type_map[int(t)] for t in frame_atype]
        atoms = Atoms(symbols=symbols, positions=frame_coord)
        if cell is not None:
            if cell.dim() == 2 and cell.shape[1] == 9:
                # (batch_size, 9) -> (3, 3)
                frame_cell = cell[batch_idx].view(3, 3).detach().cpu().numpy()
            else:
                # (batch_size, 3, 3)
                frame_cell = cell[batch_idx].detach().cpu().numpy()

            atoms.set_cell(frame_cell)
            atoms.set_pbc(True)

        atoms_list.append(atoms)

    return atoms_list  # [ase.Atoms]


def dp_to_sevennet_batch_graph_native(coord, atype, cell, cutoff, type_map):
    """
    dp data to sevennet batch graph
    Args:
        coord: (batch_size, n_atoms, 3)
        atype: (batch_size, n_atoms)
        cell: (batch_size, 9) or (batch_size, 3, 3) or None
        cutoff: float
        type_map: list[str]

    Returns:
        sevennet batch graph (AtomGraphData)
    """
    device = coord.device

    atoms_list = deepmd_to_ase_atoms_list(coord, atype, cell, type_map)

    graph_list = []
    for atoms in atoms_list:
        graph_dict = unlabeled_atoms_to_graph(
            atoms,
            cutoff=cutoff
        )  # SevenNet/sevenn/util.py

        # numpy array totensor
        graph_data = AtomGraphData.from_numpy_dict(graph_dict)
        graph_list.append(graph_data)

    # pytorch geometric batch
    collater = Collater(dataset=[], follow_batch=None, exclude_keys=None)
    batched_graph = collater(graph_list)

    # move to device
    batched_graph = batched_graph.to(device)
    return batched_graph


@BaseModel.register("sevennet_native_e0")
class SevennetNativeModelE0(BaseModel):
    """

    """

    def __init__(
            self,
            type_map: list[str],
            sel: int | str,
            r_max: float = 4.5,
            channel: int = 32,
            irreps_manual: list[str] = None,
            lmax: int = 1,
            lmax_edge: int = -1,
            lmax_node: int = -1,
            is_parity: bool = True,
            num_convolution_layer: int = 3,
            radial_basis_name: str = "bessel",
            num_radial_basis: int = 8,
            # cutoff_function: dict = None,
            cutoff_function_name: str = "poly_cut",
            poly_cut_p: int = 6,
            cutoff_on: float = 4.0,
            activation_radial: str = "silu",
            activation_scalar: dict[str, str] = {'e': 'silu', 'o': 'tanh'},
            activation_gate: dict[str, str] = {'e': 'silu', 'o': 'tanh'},
            weight_nn_hidden_neurons: list[int] = [64, 64],
            conv_denominator: str | float = "avg_num_neigh",
            train_denominator: bool = False,
            train_shift_scale: bool = False,
            use_bias_in_linear: bool = False,
            use_modal_node_embedding: bool = False,
            use_modal_self_inter_intro: bool = False,
            use_modal_self_inter_outro: bool = False,
            use_modal_output_block: bool = False,
            readout_as_fcn: bool = False,
            readout_fcn_hidden_neurons: list[int] = [30, 30],
            readout_fcn_activation: str = "relu",
            self_connection_type: str = "nequip",
            interaction_type: str = "nequip",
            normalize_sph: bool = True,
            cuequivariance_config: dict = {},
            precision: str = "float32",
            shift: str | float | list[float] = 0.0,
            scale: str | float | list[float] = 1.0,
            use_modal_wise_shift: bool = False,
            use_modal_wise_scale: bool = False,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.params = {
            "type_map": type_map,
            "sel": sel,
            "r_max": r_max,
            "channel": channel,
            "irreps_manual": irreps_manual,
            "lmax": lmax,
            "lmax_edge": lmax_edge,
            "lmax_node": lmax_node,
            "is_parity": is_parity,
            "num_convolution_layer": num_convolution_layer,
            "radial_basis_name": radial_basis_name,
            "num_radial_basis": num_radial_basis,
            "cutoff_function_name": cutoff_function_name,  # Store flat
            "poly_cut_p": poly_cut_p,  # Store flat
            "cutoff_on": cutoff_on,
            "activation_radial": activation_radial,
            "activation_scalar": activation_scalar,
            "activation_gate": activation_gate,
            "weight_nn_hidden_neurons": weight_nn_hidden_neurons,
            "conv_denominator": conv_denominator,
            "train_denominator": train_denominator,
            "train_shift_scale": train_shift_scale,
            "use_bias_in_linear": use_bias_in_linear,
            "use_modal_node_embedding": use_modal_node_embedding,
            "use_modal_self_inter_intro": use_modal_self_inter_intro,
            "use_modal_self_inter_outro": use_modal_self_inter_outro,
            "use_modal_output_block": use_modal_output_block,
            "readout_as_fcn": readout_as_fcn,
            "readout_fcn_hidden_neurons": readout_fcn_hidden_neurons,
            "readout_fcn_activation": readout_fcn_activation,
            "self_connection_type": self_connection_type,
            "interaction_type": interaction_type,
            "normalize_sph": normalize_sph,
            "cuequivariance_config": cuequivariance_config,
            "precision": precision,
            "shift": shift,
            "scale": scale,
            "use_modal_wise_shift": use_modal_wise_shift,
            "use_modal_wise_scale": use_modal_wise_scale,
        }

        self.type_map = type_map
        self.ntypes = len(type_map)
        self.rcut = r_max
        self.sel = sel
        self.preset_out_bias: dict[str, list] = {"energy": []}
        self.mm_types = []

        # Handle MM types
        atomic_numbers = []
        for ii, tt in enumerate(type_map):
            atomic_numbers.append(PeriodicTable[tt])
            if not tt.startswith("m") and tt not in {"HW", "OW"}:
                self.preset_out_bias["energy"].append(None)
            else:
                self.preset_out_bias["energy"].append([0])
                self.mm_types.append(ii)

        self.atomic_numbers = atomic_numbers

        # Initialize SevenNet model
        self._init_sevennet_model()

        # Register energy bias buffer
        self.register_buffer(
            "e0",
            torch.zeros(
                self.ntypes,
                dtype=env.GLOBAL_PT_ENER_FLOAT_PRECISION,
                device=env.DEVICE,
            ),
        )

    def _init_sevennet_model(self):
        """Initialize SevenNet model with configuration."""

        irreps_manual = self.params["irreps_manual"]

        if irreps_manual is None or irreps_manual is False:
            irreps_manual_for_sevennet = False
        else:
            irreps_manual_for_sevennet = irreps_manual
        if self.params["conv_denominator"] == "avg_num_neigh":
            if not isinstance(self.sel, (int, float)):
                conv_denominator_value = 120
            else:
                conv_denominator_value = self.sel
        if isinstance(self.params["conv_denominator"], (int, float)):
            conv_denominator_value = self.params["conv_denominator"]
        cutoff_config = {KEY.CUTOFF_FUNCTION_NAME: self.params["cutoff_function_name"]}

        if self.params["cutoff_function_name"] == "poly_cut":
            cutoff_config[KEY.POLY_CUT_P] = self.params["poly_cut_p"]
        elif self.params["cutoff_function_name"] == "XPLOR":
            cutoff_config["cutoff_on"] = self.params.get("cutoff_on", self.rcut - 0.5)

        config_dict = {
            KEY.CHEMICAL_SPECIES: self.atomic_numbers,
            KEY.NUM_SPECIES: len(self.type_map),
            KEY.TYPE_MAP: {self.atomic_numbers[i]: i for i in range(len(self.type_map))},
            KEY.CUTOFF: self.rcut,
            KEY.NODE_FEATURE_MULTIPLICITY: self.params["channel"],
            KEY.LMAX: self.params["lmax"],
            KEY.LMAX_EDGE: self.params["lmax_edge"],
            KEY.LMAX_NODE: self.params["lmax_node"],
            KEY.NUM_CONVOLUTION: self.params["num_convolution_layer"],
            KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS: self.params["weight_nn_hidden_neurons"],
            KEY.RADIAL_BASIS: {
                KEY.RADIAL_BASIS_NAME: self.params["radial_basis_name"],
                KEY.BESSEL_BASIS_NUM: self.params["num_radial_basis"],
            },
            KEY.SHIFT: self.params["shift"],
            KEY.SCALE: self.params["scale"],
            KEY.CUTOFF_FUNCTION: cutoff_config,
            KEY.IS_PARITY: self.params["is_parity"],
            KEY.SELF_CONNECTION_TYPE: self.params["self_connection_type"],
            KEY.CONV_DENOMINATOR: conv_denominator_value,
            KEY.TRAIN_DENOMINTAOR: self.params["train_denominator"],
            KEY.TRAIN_SHIFT_SCALE: self.params["train_shift_scale"],
            KEY.ACTIVATION_RADIAL: self.params["activation_radial"],
            KEY.ACTIVATION_SCARLAR: self.params["activation_scalar"],
            KEY.ACTIVATION_GATE: self.params["activation_gate"],
            KEY.USE_BIAS_IN_LINEAR: self.params["use_bias_in_linear"],
            KEY.IRREPS_MANUAL: irreps_manual_for_sevennet,
            KEY._NORMALIZE_SPH: self.params["normalize_sph"],
            KEY.INTERACTION_TYPE: self.params["interaction_type"],
            KEY.USE_MODAL_NODE_EMBEDDING: self.params["use_modal_node_embedding"],
            KEY.USE_MODAL_SELF_INTER_INTRO: self.params["use_modal_self_inter_intro"],
            KEY.USE_MODAL_SELF_INTER_OUTRO: self.params["use_modal_self_inter_outro"],
            KEY.USE_MODAL_OUTPUT_BLOCK: self.params["use_modal_output_block"],
            KEY.READOUT_AS_FCN: self.params["readout_as_fcn"],
            KEY.READOUT_FCN_HIDDEN_NEURONS: self.params["readout_fcn_hidden_neurons"],
            KEY.READOUT_FCN_ACTIVATION: self.params["readout_fcn_activation"],
            KEY.CUEQUIVARIANCE_CONFIG: self.params["cuequivariance_config"],
        }

        # print(f"config_dict: {config_dict}")
        self.sevennet_model = build_E3_equivariant_model(config_dict)
        # set batch mode
        self.sevennet_model.set_is_batch_data(True)

    #        sevennet_path = "checkpoint_best.pth"
    #        from sevenn.util import load_checkpoint
    #        if os.path.exists(sevennet_path):
    #            state_dict = load_checkpoint(sevennet_path)
    #            self.sevennet_model.load_state_dict(state_dict.model_state_dict)
    #            print("Loading SevenNet weights from checkpoint")
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
        """Forward pass using gradient-preserving batch processing."""
        if fparam is not None:
            raise ValueError("fparam is unsupported")
        if aparam is not None:
            raise ValueError("aparam is unsupported")

        nf, nloc = atype.shape
        coord = coord.view(nf, nloc, 3)
        coord = coord.to(torch.float32)
        # dp data to sevennet batch graph
        batch_graph = dp_to_sevennet_batch_graph_native(
            coord, atype, box, self.rcut, self.type_map
        )

        # run sevennet model
        result = self.sevennet_model(batch_graph)
        atom_energy = result['atomic_energy']
        if atom_energy is None:
            raise ValueError(f"No atomic energy found. Available keys: {list(result.keys())}")

        atom_energy_batch = atom_energy.view(nf, nloc)

        # add energy bias (e0): reference energy per atom type
        bias_expanded = self.e0[atype].to(atom_energy_batch.dtype).to(atom_energy_batch.device)
        atom_energy_batch = atom_energy_batch + bias_expanded

        # compute total energy
        energy_batch = torch.sum(atom_energy_batch, dim=1).view(nf, 1)

        force_batch = result['inferred_force'].view(nf, nloc, 3)
        # print("✅ use sevennet internal force")

        # compute virial tensor
        coord_for_virial = coord.view(nf, nloc, 3)
        virial_batch = torch.sum(
            force_batch.unsqueeze(-1) @ coord_for_virial.unsqueeze(-2),
            dim=1
        ).view(nf, 9)

        # prepare model output
        model_predict = {
            "atom_energy": atom_energy_batch.unsqueeze(-1),  # (nf, nloc, 1)
            "energy": energy_batch,  # (nf, 1)
            "force": force_batch,  # (nf, nloc, 3)
            "virial": virial_batch,  # (nf, 9)
        }
        # do_atomic_virial = True
        if do_atomic_virial:
            # compute atomic virial
            atomic_virial = force_batch.unsqueeze(-1) @ coord_for_virial.unsqueeze(-2)
            model_predict["atom_virial"] = atomic_virial.view(nf, nloc, 9)

        return model_predict

    # implement all necessary BaseModel methods (same as mace.py)
    @torch.jit.export
    def fitting_output_def(self) -> FittingOutputDef:
        return FittingOutputDef([
            OutputVariableDef(
                name="energy",
                shape=[1],
                reducible=True,
                r_differentiable=True,
                c_differentiable=True,
            ),
        ])

    @torch.jit.export
    def get_rcut(self) -> float:
        return self.rcut

    @torch.jit.export
    def get_type_map(self) -> list[str]:
        return self.type_map

    @torch.jit.export
    def get_sel(self) -> list[int]:
        if isinstance(self.sel, int):
            return [self.sel]
        else:
            return [120]

    @torch.jit.export
    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return 0

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
        return True  # SevenNet应该支持不同原子数的系统

    @torch.jit.export
    def has_message_passing(self) -> bool:
        return True

    @torch.jit.export
    def get_nnei(self) -> int:
        """Return the total number of selected neighboring atoms in cut-off radius."""
        return self.sel

    @torch.jit.export
    def get_nsel(self) -> int:
        """Return the total number of selected neighboring atoms in cut-off radius."""
        return self.sel

    @torch.jit.export
    def model_output_type(self) -> list[str]:
        return ["energy"]

    def compute_or_load_stat(self, sampled_func, stat_file_path: Optional[DPPath] = None) -> None:
        """Compute or load statistics parameters."""
        # Determine which keys to compute based on available data
        keys_to_compute = ["energy"]
        # sample_data = sampled_func() if callable(sampled_func) else sampled_func
        # if len(sample_data) > 0 and "force" in sample_data[0]:
        #    keys_to_compute.append("force")
        if stat_file_path is not None and self.type_map is not None:
            # descriptors and fitting net with different type_map
            # should not share the same parameters
            stat_file_path /= " ".join(self.type_map)

        bias_out, std_out = compute_output_stats(
            sampled_func,
            self.get_ntypes(),
            keys=keys_to_compute,
            stat_file_path=stat_file_path,
            rcond=None,
            preset_bias=self.preset_out_bias,
        )
        # print("bias_out", bias_out)
        # print("std_out", std_out)
        # Set energy bias (e0) - this is the only bias correction we apply
        if "energy" in bias_out:
            self.e0 = (
                bias_out["energy"]
                .view(self.e0.shape)
                .to(self.e0.dtype)
                .to(self.e0.device)
            )

    def serialize(self) -> dict:
        """Serialize the model."""
        return {
            "@class": "Model",
            "@version": 1,
            "type": "sevennet_native_e0",
            **self.params,
            "@variables": {
                "e0": to_numpy_array(self.e0),
                **{
                    kk: to_numpy_array(vv)
                    for kk, vv in self.sevennet_model.state_dict().items()
                },
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "SevennetNativeModelE0":
        """Deserialize the model."""
        data = data.copy()
        if not (data.pop("@class") == "Model" and data.pop("type") == "sevennet_native_e0"):
            raise ValueError("data is not a serialized SevennetNativeModelE0")

        check_version_compatibility(data.pop("@version"), 1, 1)
        variables = {kk: to_torch_tensor(vv) for kk, vv in data.pop("@variables").items()}

        model = cls(**data)
        model.e0 = variables.pop("e0")

        if variables:
            model.sevennet_model.load_state_dict(variables)

        return model

    @classmethod
    def update_sel(
            cls,
            train_data: DeepmdDataSystem,
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
        local_jdata_cpy = local_jdata.copy()
        min_nbor_dist, sel = UpdateSel().update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["r_max"],
            local_jdata_cpy["sel"],
            mixed_type=True,
        )
        local_jdata_cpy["sel"] = sel[0]
        return local_jdata_cpy, min_nbor_dist

    def model_output_def(self) -> ModelOutputDef:
        """Get output definition."""
        return ModelOutputDef(self.fitting_output_def())

    def translated_output_def(self) -> dict[str, Any]:
        """Get translated output definition."""
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": deepcopy(out_def_data["energy"]),
            "energy": deepcopy(out_def_data["energy_redu"]),
        }
        output_def["force"] = deepcopy(out_def_data["energy_derv_r"])
        output_def["force"].squeeze(-2)
        output_def["virial"] = deepcopy(out_def_data["energy_derv_c_redu"])
        output_def["virial"].squeeze(-2)
        output_def["atom_virial"] = deepcopy(out_def_data["energy_derv_c"])
        output_def["atom_virial"].squeeze(-3)
        if "mask" in out_def_data:
            output_def["mask"] = deepcopy(out_def_data["mask"])
        return output_def

    @classmethod
    def get_model(cls, model_params: dict) -> "SevennetNativeModelE0":
        """Get model by parameters."""
        model_params_old = model_params.copy()
        model_params = model_params.copy()
        model_params.pop("type", None)

        precision = model_params.pop("precision", "float64")
        if precision == "float32":
            torch.set_default_dtype(torch.float32)
        elif precision == "float64":
            torch.set_default_dtype(torch.float64)
        else:
            raise ValueError(f"precision {precision} not supported")

        model = cls(**model_params)
        model.model_def_script = json.dumps(model_params_old)
        return model

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
        """Forward lower pass - not implemented for native batch approach."""
        return {"energy": torch.zeros(extended_coord.shape[0], 1)}

