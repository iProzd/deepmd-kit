# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
    TypeEmbedNetConsistent,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
    to_numpy_array,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
    map_pair_exclude_types,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .descriptor import (
    extend_descrpt_stat,
)
from .repflow_s import (
    DescrptBlockRepflowsS,
    RepFlowLayerS,
)


@BaseDescriptor.register("dpa3s")
class DescrptDPA3S(BaseDescriptor, torch.nn.Module):
    r"""The simplified DPA3 descriptor.

    A simplified version of the DPA3 descriptor with flat constructor
    parameters (no nested repflow dict). It wraps DescrptBlockRepflowsS.

    Parameters
    ----------
    ntypes : int
        Number of element types.
    n_dim : int, optional
        The dimension of node representation.
    e_dim : int, optional
        The dimension of edge representation.
    a_dim : int, optional
        The dimension of angle representation.
    nlayers : int, optional
        Number of repflow layers.
    e_rcut : float
        The edge cut-off radius.
    e_rcut_smth : float
        Where to start smoothing for edge.
    e_sel : int
        Maximally possible number of selected edge neighbors.
    a_rcut : float
        The angle cut-off radius.
    a_rcut_smth : float
        Where to start smoothing for angle.
    a_sel : int
        Maximally possible number of selected angle neighbors.
    axis_neuron : int, optional
        The number of dimension of submatrix in the symmetrization ops.
    sel_reduce_factor : float, optional
        Reduction factor applied to neighbor-scale normalization.
    activation_function : str, optional
        The activation function in the embedding net.
    precision : str, optional
        The precision of the embedding net parameters.
    exclude_types : list[tuple[int, int]], optional
        The excluded pairs of types which have no interaction with each other.
    env_protection : float, optional
        Protection parameter to prevent division by zero errors.
    trainable : bool, optional
        If the parameters are trainable.
    seed : int or list[int] or None, optional
        Random seed for parameter initialization.
    use_econf_tebd : bool, optional
        Whether to use electronic configuration type embedding.
    use_tebd_bias : bool, optional
        Whether to use bias in the type embedding layer.
    type_map : list[str] or None, optional
        A list of strings. Give the name to each type of atoms.
    concat_output_tebd : bool, optional
        Whether to concat type embedding at the output of the descriptor.
    add_chg_spin_ebd : bool, optional
        Whether to add charge and spin embedding.
    """

    def __init__(
        self,
        ntypes: int,
        n_dim: int = 128,
        e_dim: int = 64,
        a_dim: int = 32,
        nlayers: int = 6,
        e_rcut: float = 6.0,
        e_rcut_smth: float = 5.0,
        e_sel: int = 120,
        a_rcut: float = 4.0,
        a_rcut_smth: float = 3.5,
        a_sel: int = 40,
        axis_neuron: int = 4,
        sel_reduce_factor: float = 10.0,
        activation_function: str = "silu",
        precision: str = "float64",
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        type_map: list[str] | None = None,
        concat_output_tebd: bool = False,
        add_chg_spin_ebd: bool = False,
    ) -> None:
        super().__init__()

        self.repflows = DescrptBlockRepflowsS(
            e_rcut=e_rcut,
            e_rcut_smth=e_rcut_smth,
            e_sel=e_sel,
            a_rcut=a_rcut,
            a_rcut_smth=a_rcut_smth,
            a_sel=a_sel,
            ntypes=ntypes,
            nlayers=nlayers,
            n_dim=n_dim,
            e_dim=e_dim,
            a_dim=a_dim,
            axis_neuron=axis_neuron,
            sel_reduce_factor=sel_reduce_factor,
            activation_function=activation_function,
            exclude_types=exclude_types,
            env_protection=env_protection,
            precision=precision,
            seed=child_seed(seed, 1),
            trainable=trainable,
        )

        self.type_embedding = TypeEmbedNet(
            ntypes,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 2),
            use_econf_tebd=use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
            trainable=trainable,
        )

        # Store all constructor parameters
        self.ntypes = ntypes
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.nlayers = nlayers
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.e_sel = e_sel
        self.a_rcut = float(a_rcut)
        self.a_rcut_smth = float(a_rcut_smth)
        self.a_sel = a_sel
        self.axis_neuron = axis_neuron
        self.sel_reduce_factor = sel_reduce_factor
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[precision]
        self.exclude_types = exclude_types
        self.env_protection = env_protection
        self.trainable = trainable
        self.use_econf_tebd = use_econf_tebd
        self.use_tebd_bias = use_tebd_bias
        self.type_map = type_map
        self.tebd_dim = n_dim
        self.concat_output_tebd = concat_output_tebd
        self.add_chg_spin_ebd = add_chg_spin_ebd

        # Optional charge/spin embedding
        if self.add_chg_spin_ebd:
            self.act = ActivationFn(activation_function)
            # -100 ~ 100 is a conservative bound
            self.chg_embedding = TypeEmbedNet(
                200,
                n_dim,
                precision=precision,
                seed=child_seed(seed, 3),
            )
            # 100 is a conservative upper bound
            self.spin_embedding = TypeEmbedNet(
                100,
                n_dim,
                precision=precision,
                seed=child_seed(seed, 4),
            )
            self.mix_cs_mlp = MLPLayer(
                2 * n_dim,
                n_dim,
                precision=precision,
                seed=child_seed(seed, 5),
            )
        else:
            self.chg_embedding = None
            self.spin_embedding = None
            self.mix_cs_mlp = None

        # Validate cutoff and selection consistency
        assert e_rcut >= a_rcut, (
            f"Edge radial cutoff (e_rcut: {e_rcut}) "
            f"must be greater than or equal to angular cutoff (a_rcut: {a_rcut})!"
        )
        assert e_sel >= a_sel, (
            f"Edge sel number (e_sel: {e_sel}) "
            f"must be greater than or equal to angular sel (a_sel: {a_sel})!"
        )

        self.rcut = self.repflows.get_rcut()
        self.rcut_smth = self.repflows.get_rcut_smth()
        self.sel = self.repflows.get_sel()

        # Set trainable
        for param in self.parameters():
            param.requires_grad = trainable

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.rcut_smth

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return self.sel

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_type_map(self) -> list[str]:
        """Get the name to each type of atoms."""
        return self.type_map

    def get_dim_out(self) -> int:
        """Returns the output dimension of this descriptor."""
        ret = self.repflows.dim_out
        if self.concat_output_tebd:
            ret += self.tebd_dim
        return ret

    def get_dim_emb(self) -> int:
        """Returns the embedding dimension of this descriptor."""
        return self.repflows.dim_emb

    def mixed_types(self) -> bool:
        """If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return True

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return True

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor needs sorted nlist when using `forward_lower`."""
        return True

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.repflows.get_env_protection()

    @property
    def dim_out(self) -> int:
        return self.get_dim_out()

    @property
    def dim_emb(self) -> int:
        """Returns the embedding dimension e_dim."""
        return self.get_dim_emb()

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        descrpt_list = [self.repflows]
        for ii, descrpt in enumerate(descrpt_list):
            descrpt.compute_input_stats(merged, path)

    def set_stat_mean_and_stddev(
        self,
        mean: list[torch.Tensor],
        stddev: list[torch.Tensor],
    ) -> None:
        """Update mean and stddev for descriptor."""
        descrpt_list = [self.repflows]
        for ii, descrpt in enumerate(descrpt_list):
            descrpt.mean = mean[ii]
            descrpt.stddev = stddev[ii]

    def get_stat_mean_and_stddev(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Get mean and stddev for descriptor."""
        mean_list = [self.repflows.mean]
        stddev_list = [self.repflows.stddev]
        return mean_list, stddev_list

    def share_params(
        self, base_class: Any, shared_level: int, resume: bool = False
    ) -> None:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert self.__class__ == base_class.__class__, (
            "Only descriptors of the same type can share params!"
        )
        # shared_level: 0
        # share all parameters in type_embedding and repflows
        if shared_level == 0:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            self.repflows.share_params(base_class.repflows, 0, resume=resume)
        # shared_level: 1
        # share all parameters in type_embedding only
        elif shared_level == 1:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
        # Other shared levels
        else:
            raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        assert self.type_map is not None, (
            "'type_map' must be defined when performing type changing!"
        )
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        self.type_map = type_map
        self.type_embedding.change_type_map(type_map=type_map)
        self.exclude_types = map_pair_exclude_types(self.exclude_types, remap_index)
        self.ntypes = len(type_map)
        repflow = self.repflows
        if has_new_type:
            # the avg and std of new types need to be updated
            extend_descrpt_stat(
                repflow,
                type_map,
                des_with_stat=model_with_new_type_stat.repflows
                if model_with_new_type_stat is not None
                else None,
            )
        repflow.ntypes = self.ntypes
        repflow.reinit_exclude(self.exclude_types)
        repflow["davg"] = repflow["davg"][remap_index]
        repflow["dstd"] = repflow["dstd"][remap_index]

    def serialize(self) -> dict:
        from deepmd.dpmodel.utils import EnvMat as DPEnvMat

        repflows = self.repflows
        data = {
            "@class": "Descriptor",
            "type": "dpa3s",
            "@version": 1,
            "ntypes": self.ntypes,
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "a_dim": self.a_dim,
            "nlayers": self.nlayers,
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "a_rcut": self.a_rcut,
            "a_rcut_smth": self.a_rcut_smth,
            "a_sel": self.a_sel,
            "axis_neuron": self.axis_neuron,
            "sel_reduce_factor": self.sel_reduce_factor,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "trainable": self.trainable,
            "use_econf_tebd": self.use_econf_tebd,
            "use_tebd_bias": self.use_tebd_bias,
            "concat_output_tebd": self.concat_output_tebd,
            "add_chg_spin_ebd": self.add_chg_spin_ebd,
            "type_map": self.type_map,
            "type_embedding": self.type_embedding.embedding.serialize(),
            "repflow_variable": {
                "edge_embd": repflows.edge_embd.serialize(),
                "angle_embd": repflows.angle_embd.serialize(),
                "repflow_layers": [layer.serialize() for layer in repflows.layers],
                "env_mat": DPEnvMat(repflows.rcut, repflows.rcut_smth).serialize(),
                "@variables": {
                    "davg": to_numpy_array(repflows["davg"]),
                    "dstd": to_numpy_array(repflows["dstd"]),
                },
            },
        }
        if self.add_chg_spin_ebd:
            data["chg_embedding"] = self.chg_embedding.embedding.serialize()
            data["spin_embedding"] = self.spin_embedding.embedding.serialize()
            data["mix_cs_mlp"] = self.mix_cs_mlp.serialize()
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA3S":
        data = data.copy()
        version = data.pop("@version")
        check_version_compatibility(version, 1, 1)
        data.pop("@class")
        data.pop("type")
        repflow_variable = data.pop("repflow_variable").copy()
        type_embedding = data.pop("type_embedding")
        chg_embedding = data.pop("chg_embedding", None)
        spin_embedding = data.pop("spin_embedding", None)
        mix_cs_mlp = data.pop("mix_cs_mlp", None)
        obj = cls(**data)
        obj.type_embedding.embedding = TypeEmbedNetConsistent.deserialize(
            type_embedding
        )

        if obj.add_chg_spin_ebd and chg_embedding is not None:
            obj.chg_embedding.embedding = TypeEmbedNetConsistent.deserialize(
                chg_embedding
            )
            obj.spin_embedding.embedding = TypeEmbedNetConsistent.deserialize(
                spin_embedding
            )
            obj.mix_cs_mlp = MLPLayer.deserialize(mix_cs_mlp)

        def t_cvt(xx: Any) -> torch.Tensor:
            return torch.tensor(xx, dtype=obj.repflows.prec, device=env.DEVICE)

        # Deserialize repflow
        statistic_repflows = repflow_variable.pop("@variables")
        env_mat = repflow_variable.pop("env_mat")
        repflow_layers = repflow_variable.pop("repflow_layers")
        obj.repflows.edge_embd = MLPLayer.deserialize(
            repflow_variable.pop("edge_embd")
        )
        obj.repflows.angle_embd = MLPLayer.deserialize(
            repflow_variable.pop("angle_embd")
        )
        obj.repflows["davg"] = t_cvt(statistic_repflows["davg"])
        obj.repflows["dstd"] = t_cvt(statistic_repflows["dstd"])
        obj.repflows.layers = torch.nn.ModuleList(
            [RepFlowLayerS.deserialize(layer) for layer in repflow_layers]
        )
        return obj

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        fparam: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Compute the descriptor.

        Parameters
        ----------
        extended_coord
            The extended coordinates of atoms. shape: nf x (nallx3)
        extended_atype
            The extended atom types. shape: nf x nall
        nlist
            The neighbor list. shape: nf x nloc x nnei
        mapping
            The index mapping, maps extended region index to local region.
        comm_dict
            The data needed for communication for parallel inference.
        fparam
            The frame parameters. Used for charge/spin embedding when enabled.

        Returns
        -------
        node_ebd
            The output descriptor. shape: nf x nloc x n_dim (or n_dim + tebd_dim)
        rot_mat
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x e_dim x 3
        edge_ebd
            The edge embedding. shape: nf x nloc x nnei x e_dim
        h2
            The rotationally equivariant pair-particle representation.
            shape: nf x nloc x nnei x 3
        sw
            The smooth switch function. shape: nf x nloc x nnei

        """
        parallel_mode = comm_dict is not None
        # Cast the input to internal precision
        extended_coord = extended_coord.to(dtype=self.prec)
        nframes, nloc, nnei = nlist.shape

        # Type embedding (local only when not parallel)
        if not parallel_mode:
            node_ebd_ext = self.type_embedding(extended_atype[:, :nloc])
        else:
            node_ebd_ext = self.type_embedding(extended_atype)

        # Optional charge/spin embedding
        if self.add_chg_spin_ebd and fparam is not None:
            assert self.chg_embedding is not None
            assert self.spin_embedding is not None
            charge = fparam[:, 0].to(dtype=torch.int64) + 100
            spin = fparam[:, 1].to(dtype=torch.int64)
            chg_ebd = self.chg_embedding(charge)
            spin_ebd = self.spin_embedding(spin)
            sys_cs_embd = self.act(
                self.mix_cs_mlp(torch.cat((chg_ebd, spin_ebd), dim=-1))
            )
            node_ebd_ext = node_ebd_ext + sys_cs_embd.unsqueeze(1)

        node_ebd_inp = node_ebd_ext[:, :nloc, :]

        # Repflows
        node_ebd, edge_ebd, h2, rot_mat, sw = self.repflows(
            nlist,
            extended_coord,
            extended_atype,
            node_ebd_ext,
            mapping,
            comm_dict=comm_dict,
        )

        if self.concat_output_tebd:
            node_ebd = torch.cat([node_ebd, node_ebd_inp], dim=-1)

        gp = env.GLOBAL_PT_FLOAT_PRECISION
        return (
            node_ebd.to(dtype=gp),
            rot_mat.to(dtype=gp) if rot_mat is not None else None,
            edge_ebd.to(dtype=gp) if edge_ebd is not None else None,
            h2.to(dtype=gp) if h2 is not None else None,
            sw.to(dtype=gp) if sw is not None else None,
        )

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statistics
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
        update_sel = UpdateSel()
        min_nbor_dist, e_sel = update_sel.update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["e_rcut"],
            local_jdata_cpy["e_sel"],
            True,
        )
        local_jdata_cpy["e_sel"] = e_sel[0]
        min_nbor_dist, a_sel = update_sel.update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["a_rcut"],
            local_jdata_cpy["a_sel"],
            True,
        )
        local_jdata_cpy["a_sel"] = a_sel[0]
        return local_jdata_cpy, min_nbor_dist

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Receive the statistics (distance, max_nbor_size and env_mat_range) of the training data.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        """
        raise NotImplementedError("Compression is unsupported for DPA3S.")
