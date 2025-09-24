# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
    Union,
)

import torch

from deepmd.dpmodel.descriptor.dpa3 import (
    RepFlowArgs,
)
from deepmd.dpmodel.utils import EnvMat as DPEnvMat
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
from .repflow_layer import (
    RepFlowLayer,
)
from .repflows import (
    DescrptBlockRepflows,
)


@BaseDescriptor.register("dpa3")
class DescrptDPA3(BaseDescriptor, torch.nn.Module):
    def __init__(
        self,
        ntypes: int,
        # args for repflow
        repflow: Union[RepFlowArgs, dict],
        # kwargs for descriptor
        concat_output_tebd: bool = False,
        activation_function: str = "silu",
        precision: str = "float64",
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        use_torch_embed: bool = False,
        use_loc_mapping: bool = True,
        type_map: Optional[list[str]] = None,
        add_chg_spin_ebd: bool = False,
    ) -> None:
        r"""The DPA-3 descriptor.

        Parameters
        ----------
        repflow : Union[RepFlowArgs, dict]
            The arguments used to initialize the repflow block, see docstr in `RepFlowArgs` for details information.
        concat_output_tebd : bool, optional
            Whether to concat type embedding at the output of the descriptor.
        activation_function : str, optional
            The activation function in the embedding net.
        precision : str, optional
            The precision of the embedding net parameters.
        exclude_types : list[list[int]], optional
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
        env_protection : float, optional
            Protection parameter to prevent division by zero errors during environment matrix calculations.
            For example, when using paddings, there may be zero distances of neighbors, which may make division by zero error during environment matrix calculations without protection.
        trainable : bool, optional
            If the parameters are trainable.
        seed : int, optional
            Random seed for parameter initialization.
        use_econf_tebd : bool, Optional
            Whether to use electronic configuration type embedding.
        use_tebd_bias : bool, Optional
            Whether to use bias in the type embedding layer.
        type_map : list[str], Optional
            A list of strings. Give the name to each type of atoms.

        Returns
        -------
        descriptor:         torch.Tensor
            the descriptor of shape nb x nloc x n_dim.
            invariant single-atom representation.
        g2:                 torch.Tensor
            invariant pair-atom representation.
        h2:                 torch.Tensor
            equivariant pair-atom representation.
        rot_mat:            torch.Tensor
            rotation matrix for equivariant fittings
        sw:                 torch.Tensor
            The switch function for decaying inverse distance.

        """
        super().__init__()

        def init_subclass_params(sub_data, sub_class):
            if isinstance(sub_data, dict):
                return sub_class(**sub_data)
            elif isinstance(sub_data, sub_class):
                return sub_data
            else:
                raise ValueError(
                    f"Input args must be a {sub_class.__name__} class or a dict!"
                )

        self.repflow_args = init_subclass_params(repflow, RepFlowArgs)
        self.activation_function = activation_function

        self.repflows = DescrptBlockRepflows(
            self.repflow_args.e_rcut,
            self.repflow_args.e_rcut_smth,
            self.repflow_args.e_sel,
            self.repflow_args.a_rcut,
            self.repflow_args.a_rcut_smth,
            self.repflow_args.a_sel,
            ntypes,
            nlayers=self.repflow_args.nlayers,
            n_dim=self.repflow_args.n_dim,
            e_dim=self.repflow_args.e_dim,
            a_dim=self.repflow_args.a_dim,
            a_compress_rate=self.repflow_args.a_compress_rate,
            a_compress_e_rate=self.repflow_args.a_compress_e_rate,
            a_compress_use_split=self.repflow_args.a_compress_use_split,
            n_multi_edge_message=self.repflow_args.n_multi_edge_message,
            axis_neuron=self.repflow_args.axis_neuron,
            update_angle=self.repflow_args.update_angle,
            activation_function=self.activation_function,
            update_style=self.repflow_args.update_style,
            update_residual=self.repflow_args.update_residual,
            update_residual_init=self.repflow_args.update_residual_init,
            optim_update=self.repflow_args.optim_update,
            skip_stat=self.repflow_args.skip_stat,
            smooth_angle_init=self.repflow_args.smooth_angle_init,
            angle_init_use_sin=self.repflow_args.angle_init_use_sin,
            smooth_edge_update=self.repflow_args.smooth_edge_update,
            angle_multi_freq=self.repflow_args.angle_multi_freq,
            use_dynamic_sel=self.repflow_args.use_dynamic_sel,
            sel_reduce_factor=self.repflow_args.sel_reduce_factor,
            use_env_envelope=self.repflow_args.use_env_envelope,
            use_new_sw=self.repflow_args.use_new_sw,
            update_dihedral=self.repflow_args.update_dihedral,
            d_dim=self.repflow_args.d_dim,
            d_sel=self.repflow_args.d_sel,
            d_rcut=self.repflow_args.d_rcut,
            d_rcut_smth=self.repflow_args.d_rcut_smth,
            use_ffn_node_edge_message=self.repflow_args.use_ffn_node_edge_message,
            use_ffn_edge_edge_message=self.repflow_args.use_ffn_edge_edge_message,
            use_ffn_edge_angle_message=self.repflow_args.use_ffn_edge_angle_message,
            use_ffn_angle_angle_message=self.repflow_args.use_ffn_angle_angle_message,
            ffn_hidden_dim=self.repflow_args.ffn_hidden_dim,
            edge_use_concat_rbf=self.repflow_args.edge_use_concat_rbf,
            edge_use_rbf=self.repflow_args.edge_use_rbf,
            edge_use_dist=self.repflow_args.edge_use_dist,
            embed_use_bias=self.repflow_args.embed_use_bias,
            edge_use_attn=self.repflow_args.edge_use_attn,
            edge_attn_hidden=self.repflow_args.edge_attn_hidden,
            edge_attn_head=self.repflow_args.edge_attn_head,
            edge_attn_use_ln=self.repflow_args.edge_attn_use_ln,
            edge_rbf_dot_self=self.repflow_args.edge_rbf_dot_self,
            edge_rbf_dot_message=self.repflow_args.edge_rbf_dot_message,
            edge_use_esen_rbf=self.repflow_args.edge_use_esen_rbf,
            edge_use_esen_atom_ebd=self.repflow_args.edge_use_esen_atom_ebd,
            edge_use_esen_env=self.repflow_args.edge_use_esen_env,
            residual_pref=self.repflow_args.residual_pref,
            tebd_use_act=self.repflow_args.tebd_use_act,
            message_use_self_concat=self.repflow_args.message_use_self_concat,
            use_slim_message=self.repflow_args.use_slim_message,
            use_combined_output=self.repflow_args.use_combined_output,
            use_force_embedding=self.repflow_args.use_force_embedding,
            force_embedding_on_edge=self.repflow_args.force_embedding_on_edge,
            use_gated_mlp=self.repflow_args.use_gated_mlp,
            gated_mlp_norm=self.repflow_args.gated_mlp_norm,
            only_angle_gated_mlp=self.repflow_args.only_angle_gated_mlp,
            node_use_rmsnorm=self.repflow_args.node_use_rmsnorm,
            use_res_gnn=self.repflow_args.use_res_gnn,
            res_gnn_layer=self.repflow_args.res_gnn_layer,
            use_rk_update=self.repflow_args.use_rk_update,
            rk_order=self.repflow_args.rk_order,
            rk_update_diff_layer=self.repflow_args.rk_update_diff_layer,
            angle_use_node=self.repflow_args.angle_use_node,
            use_loc_mapping=use_loc_mapping,
            angle_self_attention=self.repflow_args.angle_self_attention,
            angle_self_attention_gate=self.repflow_args.angle_self_attention_gate,
            rmsnorm_mode=self.repflow_args.rmsnorm_mode,
            edge_rbf_cat_message=self.repflow_args.edge_rbf_cat_message,
            edge_message_use_dropout=self.repflow_args.edge_message_use_dropout,
            angle_message_use_dropout=self.repflow_args.angle_message_use_dropout,
            dropout_rate=self.repflow_args.dropout_rate,
            angle_use_sh_init=self.repflow_args.angle_use_sh_init,
            angle_sh_init_lmax=self.repflow_args.angle_sh_init_lmax,
            angle_use_fixed_gaussian=self.repflow_args.angle_use_fixed_gaussian,
            angle_fixed_gaussian_interpolate=self.repflow_args.angle_fixed_gaussian_interpolate,
            use_e3nn_conv=self.repflow_args.use_e3nn_conv,
            e3nn_conv_pattern=self.repflow_args.e3nn_conv_pattern,
            use_e3nn_denominator=self.repflow_args.use_e3nn_denominator,
            e3nn_conv_l_max=self.repflow_args.e3nn_conv_l_max,
            exclude_types=exclude_types,
            env_protection=env_protection,
            precision=precision,
            seed=child_seed(seed, 1),
        )
        self.act = ActivationFn(activation_function)

        self.use_econf_tebd = use_econf_tebd
        self.add_chg_spin_ebd = add_chg_spin_ebd
        self.use_loc_mapping = use_loc_mapping
        self.use_tebd_bias = use_tebd_bias
        self.use_torch_embed = use_torch_embed
        self.type_map = type_map
        self.tebd_dim = self.repflow_args.n_dim
        self.concat_output_tebd = concat_output_tebd
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.use_force_embedding = self.repflow_args.use_force_embedding
        if self.use_torch_embed:
            self.type_embedding = torch.nn.Embedding(
                ntypes, self.tebd_dim, device=env.DEVICE, dtype=self.prec
            )
        else:
            self.type_embedding = TypeEmbedNet(
                ntypes,
                self.tebd_dim,
                precision=precision,
                seed=child_seed(seed, 2),
                use_econf_tebd=self.use_econf_tebd,
                use_tebd_bias=use_tebd_bias,
                type_map=type_map,
            )
        if self.add_chg_spin_ebd:
            # -100 ~ 100 is a conservative bound
            self.chg_embedding = TypeEmbedNet(
                200,
                self.tebd_dim,
                precision=precision,
                seed=child_seed(seed, 3),
            )
            # 100 is a conservative upper bound
            self.spin_embedding = TypeEmbedNet(
                100,
                self.tebd_dim,
                precision=precision,
                seed=child_seed(seed, 4),
            )
            self.mix_cs_mlp = MLPLayer(
                2 * self.tebd_dim,
                self.tebd_dim,
                precision=precision,
                seed=child_seed(seed, 3),
            )
        else:
            self.chg_embedding = None
            self.spin_embedding = None
            self.mix_cs_mlp = None

        self.exclude_types = exclude_types
        self.env_protection = env_protection
        self.trainable = trainable

        assert self.repflows.e_rcut >= self.repflows.a_rcut
        assert self.repflows.e_sel >= self.repflows.a_sel

        self.rcut = self.repflows.get_rcut()
        self.rcut_smth = self.repflows.get_rcut_smth()
        self.sel = self.repflows.get_sel()
        self.ntypes = ntypes

        # set trainable
        for param in self.parameters():
            param.requires_grad = trainable
        self.compress = False

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

    def get_norm_fact(self) -> list[float]:
        """Returns the norm factor."""
        return self.repflows.get_norm_fact()

    def get_additional_output_for_fitting(self):
        return self.repflows.get_additional_output_for_fitting()

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

    def get_angle_dim(self) -> int:
        """Returns the angle embedding dimension of this descriptor."""
        return self.repflows.a_dim

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
        return self.repflows.has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor needs sorted nlist when using `forward_lower`."""
        return True

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.repflows.get_env_protection()

    def share_params(self, base_class, shared_level, resume=False) -> None:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert (
            self.__class__ == base_class.__class__
        ), "Only descriptors of the same type can share params!"
        # For DPA3 descriptors, the user-defined share-level
        # shared_level: 0
        # share all parameters in type_embedding, repflow
        if shared_level == 0:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            self.repflows.share_params(base_class.repflows, 0, resume=resume)
        # shared_level: 1
        # share all parameters in type_embedding
        elif shared_level == 1:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
        # Other shared levels
        else:
            raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        assert (
            self.type_map is not None
        ), "'type_map' must be defined when performing type changing!"
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

    @property
    def dim_out(self):
        return self.get_dim_out()

    @property
    def dim_emb(self):
        """Returns the embedding dimension g2."""
        return self.get_dim_emb()

    def compute_input_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        path: Optional[DPPath] = None,
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

    def get_stat_mean_and_stddev(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Get mean and stddev for descriptor."""
        mean_list = [self.repflows.mean]
        stddev_list = [self.repflows.stddev]
        return mean_list, stddev_list

    def serialize(self) -> dict:
        repflows = self.repflows
        data = {
            "@class": "Descriptor",
            "type": "dpa3",
            "@version": 1,
            "ntypes": self.ntypes,
            "repflow_args": self.repflow_args.serialize(),
            "concat_output_tebd": self.concat_output_tebd,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "trainable": self.trainable,
            "use_econf_tebd": self.use_econf_tebd,
            "use_tebd_bias": self.use_tebd_bias,
            "type_map": self.type_map,
            "type_embedding": self.type_embedding.embedding.serialize(),
        }
        repflow_variable = {
            "edge_embd": repflows.edge_embd.serialize(),
            "angle_embd": repflows.angle_embd.serialize(),
            "repflow_layers": [layer.serialize() for layer in repflows.layers],
            "env_mat": DPEnvMat(repflows.rcut, repflows.rcut_smth).serialize(),
            "@variables": {
                "davg": to_numpy_array(repflows["davg"]),
                "dstd": to_numpy_array(repflows["dstd"]),
            },
        }
        data.update(
            {
                "repflow_variable": repflow_variable,
            }
        )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA3":
        data = data.copy()
        version = data.pop("@version")
        check_version_compatibility(version, 1, 1)
        data.pop("@class")
        data.pop("type")
        repflow_variable = data.pop("repflow_variable").copy()
        type_embedding = data.pop("type_embedding")
        data["repflow"] = RepFlowArgs(**data.pop("repflow_args"))
        obj = cls(**data)
        obj.type_embedding.embedding = TypeEmbedNetConsistent.deserialize(
            type_embedding
        )

        def t_cvt(xx):
            return torch.tensor(xx, dtype=obj.repflows.prec, device=env.DEVICE)

        # deserialize repflow
        statistic_repflows = repflow_variable.pop("@variables")
        env_mat = repflow_variable.pop("env_mat")
        repflow_layers = repflow_variable.pop("repflow_layers")
        obj.repflows.edge_embd = MLPLayer.deserialize(repflow_variable.pop("edge_embd"))
        obj.repflows.angle_embd = MLPLayer.deserialize(
            repflow_variable.pop("angle_embd")
        )
        obj.repflows["davg"] = t_cvt(statistic_repflows["davg"])
        obj.repflows["dstd"] = t_cvt(statistic_repflows["dstd"])
        obj.repflows.layers = torch.nn.ModuleList(
            [RepFlowLayer.deserialize(layer) for layer in repflow_layers]
        )
        return obj

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
        force_embedding_input: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
    ):
        """Compute the descriptor.

        Parameters
        ----------
        extended_coord
            The extended coordinates of atoms. shape: nf x (nallx3)
        extended_atype
            The extended aotm types. shape: nf x nall
        nlist
            The neighbor list. shape: nf x nloc x nnei
        mapping
            The index mapping, mapps extended region index to local region.
        comm_dict
            The data needed for communication for parallel inference.

        Returns
        -------
        node_ebd
            The output descriptor. shape: nf x nloc x n_dim (or n_dim + tebd_dim)
        rot_mat
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x e_dim x 3
        edge_ebd
            The edge embedding.
            shape: nf x nloc x nnei x e_dim
        h2
            The rotationally equivariant pair-partical representation.
            shape: nf x nloc x nnei x 3
        sw
            The smooth switch function. shape: nf x nloc x nnei

        """
        parrallel_mode = comm_dict is not None
        # cast the input to internal precsion
        extended_coord = extended_coord.to(dtype=self.prec)
        force_embedding_input = (
            force_embedding_input.to(dtype=self.prec)
            if force_embedding_input is not None
            else None
        )
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.view(nframes, -1).shape[1] // 3

        if not parrallel_mode and self.use_loc_mapping:
            node_ebd_ext = self.type_embedding(extended_atype[:, :nloc])
        else:
            node_ebd_ext = self.type_embedding(extended_atype)

        if self.add_chg_spin_ebd:
            assert fparam is not None
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
        # repflows
        node_ebd, edge_ebd, h2, rot_mat, sw = self.repflows(
            nlist,
            extended_coord,
            extended_atype,
            node_ebd_ext,
            mapping,
            comm_dict=comm_dict,
            force_embedding_input=force_embedding_input,
        )
        if self.concat_output_tebd:
            node_ebd = torch.cat([node_ebd, node_ebd_inp], dim=-1)
        return (
            node_ebd.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            rot_mat.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            edge_ebd.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            h2.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            sw.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
        )

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
        min_nbor_dist, repflow_e_sel = update_sel.update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["repflow"]["e_rcut"],
            local_jdata_cpy["repflow"]["e_sel"],
            True,
        )
        local_jdata_cpy["repflow"]["e_sel"] = repflow_e_sel[0]

        min_nbor_dist, repflow_a_sel = update_sel.update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["repflow"]["a_rcut"],
            local_jdata_cpy["repflow"]["a_sel"],
            True,
        )
        local_jdata_cpy["repflow"]["a_sel"] = repflow_a_sel[0]

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
        raise NotImplementedError("Compression is unsupported for DPA3.")
