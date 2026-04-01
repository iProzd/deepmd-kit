# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    Array,
    xp_take_along_axis,
    xp_take_first_n,
)
from deepmd.dpmodel.common import (
    cast_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils import (
    EnvMat,
    PairExcludeMask,
    aggregate,
    get_graph_index,
)
from deepmd.dpmodel.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
    get_activation_fn,
)
from deepmd.dpmodel.utils.safe_gradient import (
    safe_for_vector_norm,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.dpmodel.utils.type_embed import (
    TypeEmbedNet,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
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
    DescriptorBlock,
    extend_descrpt_stat,
)
from .repformers import (
    _cal_grrg,
    get_residual,
)


def _cal_hg_dynamic(
    flat_edge_ebd: Array,
    flat_h2: Array,
    flat_sw: Array,
    owner: Array,
    num_owner: int,
    nb: int,
    nloc: int,
    scale_factor: float,
) -> Array:
    xp = array_api_compat.array_namespace(flat_edge_ebd, flat_h2, flat_sw, owner)
    n_edge, e_dim = flat_edge_ebd.shape
    flat_edge_ebd = flat_edge_ebd * xp.expand_dims(flat_sw, axis=-1)
    flat_h2g2 = xp.reshape(
        xp.expand_dims(flat_h2, axis=-1) * xp.expand_dims(flat_edge_ebd, axis=1),
        (-1, 3 * e_dim),
    )
    h2g2 = aggregate(flat_h2g2, owner, average=False, num_owner=num_owner)
    h2g2 = xp.reshape(h2g2, (nb, nloc, 3, e_dim)) * scale_factor
    return h2g2


def symmetrization_op_dynamic(
    flat_edge_ebd: Array,
    flat_h2: Array,
    flat_sw: Array,
    owner: Array,
    num_owner: int,
    nb: int,
    nloc: int,
    scale_factor: float,
    axis_neuron: int,
) -> Array:
    h2g2 = _cal_hg_dynamic(
        flat_edge_ebd, flat_h2, flat_sw, owner, num_owner, nb, nloc, scale_factor,
    )
    grrg = _cal_grrg(h2g2, axis_neuron)
    return grrg


class RepFlowLayerS(NativeOP):
    def __init__(
        self,
        e_rcut: float,
        e_rcut_smth: float,
        e_sel: int | list[int],
        a_rcut: float,
        a_rcut_smth: float,
        a_sel: int,
        ntypes: int,
        n_dim: int = 128,
        e_dim: int = 64,
        a_dim: int = 32,
        axis_neuron: int = 4,
        activation_function: str = "silu",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        precision: str = "float64",
        sel_reduce_factor: float = 10.0,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.ntypes = ntypes
        e_sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(e_sel)
        self.e_sel = e_sel
        self.a_rcut = float(a_rcut)
        self.a_rcut_smth = float(a_rcut_smth)
        self.a_sel = a_sel
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.axis_neuron = axis_neuron
        self.activation_function = activation_function
        self.act = get_activation_fn(self.activation_function)
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.precision = precision
        self.seed = seed
        self.prec = PRECISION_DICT[precision]
        self.sel_reduce_factor = sel_reduce_factor
        self.dynamic_e_sel = self.nnei / self.sel_reduce_factor
        self.dynamic_a_sel = self.a_sel / self.sel_reduce_factor

        self.edge_info_dim = n_dim * 2 + e_dim
        self.angle_dim = a_dim + n_dim + 2 * e_dim

        self.node_self_mlp = NativeLayer(
            n_dim, n_dim, precision=precision,
            seed=child_seed(seed, 0), trainable=trainable,
        )
        self.n_sym_dim = (n_dim + e_dim) * axis_neuron
        self.node_sym_linear = NativeLayer(
            self.n_sym_dim, n_dim, precision=precision,
            seed=child_seed(seed, 2), trainable=trainable,
        )
        self.node_edge_linear = NativeLayer(
            self.edge_info_dim, n_dim, precision=precision,
            seed=child_seed(seed, 4), trainable=trainable,
        )
        self.edge_self_linear = NativeLayer(
            self.edge_info_dim, e_dim, precision=precision,
            seed=child_seed(seed, 6), trainable=trainable,
        )
        self.edge_angle_linear1 = NativeLayer(
            self.angle_dim, e_dim, precision=precision,
            seed=child_seed(seed, 10), trainable=trainable,
        )
        self.edge_angle_linear2 = NativeLayer(
            e_dim, e_dim, precision=precision,
            seed=child_seed(seed, 11), trainable=trainable,
        )
        self.angle_self_linear = NativeLayer(
            self.angle_dim, a_dim, precision=precision,
            seed=child_seed(seed, 13), trainable=trainable,
        )

        self.n_residual = [
            get_residual(
                n_dim, update_residual, update_residual_init,
                precision=precision, seed=child_seed(seed, 1), trainable=trainable,
            ),
            get_residual(
                n_dim, update_residual, update_residual_init,
                precision=precision, seed=child_seed(seed, 3), trainable=trainable,
            ),
            get_residual(
                n_dim, update_residual, update_residual_init,
                precision=precision, seed=child_seed(seed, 5), trainable=trainable,
            ),
        ]
        self.e_residual = [
            get_residual(
                e_dim, update_residual, update_residual_init,
                precision=precision, seed=child_seed(seed, 7), trainable=trainable,
            ),
            get_residual(
                e_dim, update_residual, update_residual_init,
                precision=precision, seed=child_seed(seed, 12), trainable=trainable,
            ),
        ]
        self.a_residual = [
            get_residual(
                a_dim, update_residual, update_residual_init,
                precision=precision, seed=child_seed(seed, 14), trainable=trainable,
            ),
        ]

    def call(
        self,
        node_ebd_ext: Array,
        edge_ebd: Array,
        h2: Array,
        angle_ebd: Array,
        nlist: Array,
        nlist_mask: Array,
        sw: Array,
        a_nlist: Array,
        a_nlist_mask: Array,
        a_sw: Array,
        edge_index: Array,
        angle_index: Array,
    ) -> tuple[Array, Array, Array]:
        xp = array_api_compat.array_namespace(
            node_ebd_ext, edge_ebd, h2, angle_ebd,
            nlist, nlist_mask, sw, a_nlist, a_nlist_mask, a_sw,
            edge_index, angle_index,
        )
        nb, nloc, nnei = nlist.shape
        nall = node_ebd_ext.shape[1]
        node_ebd = xp_take_first_n(node_ebd_ext, 1, nloc)

        n2e_index = edge_index[0, :]
        n_ext2e_index = edge_index[1, :]
        n2a_index = angle_index[0, :]
        eij2a_index = angle_index[1, :]
        eik2a_index = angle_index[2, :]

        nei_node_ebd = xp.take(
            xp.reshape(node_ebd_ext, (-1, self.n_dim)), n_ext2e_index, axis=0,
        )

        # 1. Node self MLP
        node_self = self.act(self.node_self_mlp(node_ebd))

        # 2. Node symmetrization (grrg + drrd)
        sym_edge = symmetrization_op_dynamic(
            edge_ebd, h2, sw, owner=n2e_index, num_owner=nb * nloc,
            nb=nb, nloc=nloc, scale_factor=self.dynamic_e_sel ** (-0.5),
            axis_neuron=self.axis_neuron,
        )
        sym_node = symmetrization_op_dynamic(
            nei_node_ebd, h2, sw, owner=n2e_index, num_owner=nb * nloc,
            nb=nb, nloc=nloc, scale_factor=self.dynamic_e_sel ** (-0.5),
            axis_neuron=self.axis_neuron,
        )
        node_sym = self.act(
            self.node_sym_linear(xp.concat([sym_edge, sym_node], axis=-1))
        )

        # 3. Node edge message (direct concat, no optim_update)
        node_i_broadcast = xp.take(
            xp.reshape(node_ebd, (-1, self.n_dim)), n2e_index, axis=0,
        )
        edge_info = xp.concat([node_i_broadcast, nei_node_ebd, edge_ebd], axis=-1)
        node_edge_update = self.act(self.node_edge_linear(edge_info)) * xp.expand_dims(
            sw, axis=-1,
        )
        node_edge_msg = xp.reshape(
            aggregate(node_edge_update, n2e_index, average=False, num_owner=nb * nloc),
            (nb, nloc, node_edge_update.shape[-1]),
        ) / self.dynamic_e_sel

        # Update node (res_residual)
        n_updated = (
            node_ebd
            + self.n_residual[0] * node_self
            + self.n_residual[1] * node_sym
            + self.n_residual[2] * node_edge_msg
        )

        # 4. Edge self message
        edge_self_update = self.act(self.edge_self_linear(edge_info))

        # 5. Angle info (no compression)
        node_for_angle = xp.take(
            xp.reshape(node_ebd, (-1, self.n_dim)), n2a_index, axis=0,
        )
        edge_for_angle_ik = xp.take(edge_ebd, eik2a_index, axis=0)
        edge_for_angle_ij = xp.take(edge_ebd, eij2a_index, axis=0)
        angle_info = xp.concat(
            [angle_ebd, node_for_angle, edge_for_angle_ik, edge_for_angle_ij], axis=-1,
        )

        # 6. Edge angle message
        edge_angle_update = self.act(self.edge_angle_linear1(angle_info))
        weighted_edge_angle = edge_angle_update * xp.expand_dims(a_sw, axis=-1)
        n_edge = edge_ebd.shape[0]
        reduced_edge_angle = aggregate(
            weighted_edge_angle, eij2a_index, average=False, num_owner=n_edge,
        ) / (self.dynamic_a_sel ** 0.5)
        edge_angle_msg = self.act(self.edge_angle_linear2(reduced_edge_angle))

        # Update edge (res_residual)
        e_updated = (
            edge_ebd
            + self.e_residual[0] * edge_self_update
            + self.e_residual[1] * edge_angle_msg
        )

        # 7. Angle self message
        angle_self_update = self.act(self.angle_self_linear(angle_info))

        # Update angle (res_residual)
        a_updated = angle_ebd + self.a_residual[0] * angle_self_update

        return n_updated, e_updated, a_updated

    def serialize(self) -> dict:
        return {
            "@class": "RepFlowLayerS",
            "@version": 1,
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "a_rcut": self.a_rcut,
            "a_rcut_smth": self.a_rcut_smth,
            "a_sel": self.a_sel,
            "ntypes": self.ntypes,
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "a_dim": self.a_dim,
            "axis_neuron": self.axis_neuron,
            "activation_function": self.activation_function,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
            "precision": self.precision,
            "sel_reduce_factor": self.sel_reduce_factor,
            "node_self_mlp": self.node_self_mlp.serialize(),
            "node_sym_linear": self.node_sym_linear.serialize(),
            "node_edge_linear": self.node_edge_linear.serialize(),
            "edge_self_linear": self.edge_self_linear.serialize(),
            "edge_angle_linear1": self.edge_angle_linear1.serialize(),
            "edge_angle_linear2": self.edge_angle_linear2.serialize(),
            "angle_self_linear": self.angle_self_linear.serialize(),
            "@variables": {
                "n_residual": [to_numpy_array(t) for t in self.n_residual],
                "e_residual": [to_numpy_array(t) for t in self.e_residual],
                "a_residual": [to_numpy_array(t) for t in self.a_residual],
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "RepFlowLayerS":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        node_self_mlp = data.pop("node_self_mlp")
        node_sym_linear = data.pop("node_sym_linear")
        node_edge_linear = data.pop("node_edge_linear")
        edge_self_linear = data.pop("edge_self_linear")
        edge_angle_linear1 = data.pop("edge_angle_linear1")
        edge_angle_linear2 = data.pop("edge_angle_linear2")
        angle_self_linear = data.pop("angle_self_linear")
        variables = data.pop("@variables")
        n_residual = variables["n_residual"]
        e_residual = variables["e_residual"]
        a_residual = variables["a_residual"]
        obj = cls(**data)
        obj.node_self_mlp = NativeLayer.deserialize(node_self_mlp)
        obj.node_sym_linear = NativeLayer.deserialize(node_sym_linear)
        obj.node_edge_linear = NativeLayer.deserialize(node_edge_linear)
        obj.edge_self_linear = NativeLayer.deserialize(edge_self_linear)
        obj.edge_angle_linear1 = NativeLayer.deserialize(edge_angle_linear1)
        obj.edge_angle_linear2 = NativeLayer.deserialize(edge_angle_linear2)
        obj.angle_self_linear = NativeLayer.deserialize(angle_self_linear)
        obj.n_residual = n_residual
        obj.e_residual = e_residual
        obj.a_residual = a_residual
        return obj


@DescriptorBlock.register("se_repflow_s")
class DescrptBlockRepflowsS(NativeOP, DescriptorBlock):
    def __init__(
        self,
        e_rcut: float,
        e_rcut_smth: float,
        e_sel: int,
        a_rcut: float,
        a_rcut_smth: float,
        a_sel: int,
        ntypes: int,
        nlayers: int = 6,
        n_dim: int = 128,
        e_dim: int = 64,
        a_dim: int = 32,
        axis_neuron: int = 4,
        activation_function: str = "silu",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        set_davg_zero: bool = True,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        precision: str = "float64",
        fix_stat_std: float = 0.3,
        sel_reduce_factor: float = 10.0,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.e_sel = e_sel
        self.a_rcut = float(a_rcut)
        self.a_rcut_smth = float(a_rcut_smth)
        self.a_sel = a_sel
        self.ntypes = ntypes
        self.nlayers = nlayers
        sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4
        assert len(sel) == 1
        self.sel = sel
        self.rcut = e_rcut
        self.rcut_smth = e_rcut_smth
        self.sec = self.sel
        self.split_sel = self.sel
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.axis_neuron = axis_neuron
        self.set_davg_zero = set_davg_zero
        self.fix_stat_std = fix_stat_std
        self.set_stddev_constant = fix_stat_std != 0.0
        self.sel_reduce_factor = sel_reduce_factor
        self.activation_function = activation_function
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.act = get_activation_fn(self.activation_function)
        self.prec = PRECISION_DICT[precision]
        self.reinit_exclude(exclude_types)
        self.env_protection = env_protection
        self.precision = precision
        self.seed = seed

        self.edge_embd = NativeLayer(
            1, self.e_dim, precision=precision,
            seed=child_seed(seed, 0), trainable=trainable,
        )
        self.angle_embd = NativeLayer(
            1, self.a_dim, precision=precision, bias=False,
            seed=child_seed(seed, 1), trainable=trainable,
        )

        layers = []
        for ii in range(nlayers):
            layers.append(
                RepFlowLayerS(
                    e_rcut=self.e_rcut,
                    e_rcut_smth=self.e_rcut_smth,
                    e_sel=self.sel,
                    a_rcut=self.a_rcut,
                    a_rcut_smth=self.a_rcut_smth,
                    a_sel=self.a_sel,
                    ntypes=self.ntypes,
                    n_dim=self.n_dim,
                    e_dim=self.e_dim,
                    a_dim=self.a_dim,
                    axis_neuron=self.axis_neuron,
                    activation_function=self.activation_function,
                    update_residual=self.update_residual,
                    update_residual_init=self.update_residual_init,
                    precision=precision,
                    sel_reduce_factor=self.sel_reduce_factor,
                    seed=child_seed(child_seed(seed, 1), ii),
                    trainable=trainable,
                )
            )
        self.layers = layers

        wanted_shape = (self.ntypes, self.nnei, 4)
        self.env_mat_edge = EnvMat(
            self.e_rcut, self.e_rcut_smth,
            protection=self.env_protection, use_exp_switch=True,
        )
        self.env_mat_angle = EnvMat(
            self.a_rcut, self.a_rcut_smth,
            protection=self.env_protection, use_exp_switch=True,
        )
        self.mean = np.zeros(wanted_shape, dtype=PRECISION_DICT[self.precision])
        self.stddev = np.ones(wanted_shape, dtype=PRECISION_DICT[self.precision])
        if self.set_stddev_constant:
            self.stddev = self.stddev * self.fix_stat_std

    def get_rcut(self) -> float:
        return self.e_rcut

    def get_rcut_smth(self) -> float:
        return self.e_rcut_smth

    def get_nsel(self) -> int:
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        return self.sel

    def get_ntypes(self) -> int:
        return self.ntypes

    def get_dim_out(self) -> int:
        return self.dim_out

    def get_dim_in(self) -> int:
        return self.dim_in

    def get_dim_emb(self) -> int:
        return self.e_dim

    def __setitem__(self, key: str, value: Array) -> None:
        if key in ("avg", "data_avg", "davg"):
            self.mean = value
        elif key in ("std", "data_std", "dstd"):
            self.stddev = value
        else:
            raise KeyError(key)

    def __getitem__(self, key: str) -> Array:
        if key in ("avg", "data_avg", "davg"):
            return self.mean
        elif key in ("std", "data_std", "dstd"):
            return self.stddev
        else:
            raise KeyError(key)

    def mixed_types(self) -> bool:
        return True

    def get_env_protection(self) -> float:
        return self.env_protection

    @property
    def dim_out(self) -> int:
        return self.n_dim

    @property
    def dim_in(self) -> int:
        return self.n_dim

    @property
    def dim_emb(self) -> int:
        return self.get_dim_emb()

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        if self.set_stddev_constant and self.set_davg_zero:
            return
        env_mat_stat = EnvMatStatSe(self)
        if path is not None:
            path = path / env_mat_stat.get_hash()
        if path is None or not path.is_dir():
            if callable(merged):
                sampled = merged()
            else:
                sampled = merged
        else:
            sampled = []
        env_mat_stat.load_or_compute_stats(sampled, path)
        self.stats = env_mat_stat.stats
        mean, stddev = env_mat_stat()
        xp = array_api_compat.array_namespace(self.stddev)
        device = array_api_compat.device(self.stddev)
        if not self.set_davg_zero:
            self.mean = xp.asarray(
                mean, dtype=self.mean.dtype, copy=True, device=device,
            )
        if not self.set_stddev_constant:
            self.stddev = xp.asarray(
                stddev, dtype=self.stddev.dtype, copy=True, device=device,
            )

    def get_stats(self) -> dict[str, StatItem]:
        if self.stats is None:
            raise RuntimeError(
                "The statistics of the descriptor has not been computed."
            )
        return self.stats

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def call(
        self,
        nlist: Array,
        coord_ext: Array,
        atype_ext: Array,
        atype_embd_ext: Array | None = None,
        mapping: Array | None = None,
        type_embedding: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, Array]:
        xp = array_api_compat.array_namespace(nlist, coord_ext, atype_ext)
        nframes, nloc, nnei = nlist.shape
        nall = xp.reshape(coord_ext, (nframes, -1)).shape[1] // 3
        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        exclude_mask = xp.astype(exclude_mask, xp.bool)
        nlist = xp.where(exclude_mask, nlist, xp.full_like(nlist, -1))

        dmatrix, diff, sw = self.env_mat_edge.call(
            coord_ext, atype_ext, nlist, self.mean[...], self.stddev[...],
        )
        nlist_mask = nlist != -1
        sw = xp.reshape(sw, (nframes, nloc, nnei))
        sw = xp.where(nlist_mask, sw, xp.zeros_like(sw))

        a_dist_mask = (safe_for_vector_norm(diff, axis=-1) < self.a_rcut)[
            :, :, : self.a_sel
        ]
        a_nlist = nlist[:, :, : self.a_sel]
        a_nlist = xp.where(a_dist_mask, a_nlist, xp.full_like(a_nlist, -1))

        _, a_diff, a_sw = self.env_mat_angle.call(
            coord_ext, atype_ext, a_nlist,
            self.mean[:, : self.a_sel, :], self.stddev[:, : self.a_sel, :],
        )
        a_nlist_mask = a_nlist != -1
        a_sw = xp.reshape(a_sw, (nframes, nloc, self.a_sel))
        a_sw = xp.where(a_nlist_mask, a_sw, xp.zeros_like(a_sw))

        nlist = xp.where(nlist == -1, xp.zeros_like(nlist), nlist)
        a_nlist = xp.where(a_nlist == -1, xp.zeros_like(a_nlist), a_nlist)

        atype_embd = xp_take_first_n(atype_embd_ext, 1, nloc)
        assert list(atype_embd.shape) == [nframes, nloc, self.n_dim]
        node_ebd = self.act(atype_embd)

        edge_input = safe_for_vector_norm(diff, axis=-1, keepdims=True)
        h2 = dmatrix[:, :, :, 1:]

        normalized_diff_i = a_diff / (
            safe_for_vector_norm(a_diff, axis=-1, keepdims=True) + 1e-6
        )
        normalized_diff_j = xp.matrix_transpose(normalized_diff_i)
        cosine_ij = xp.matmul(normalized_diff_i, normalized_diff_j) * (1 - 1e-6)
        angle_input = xp.reshape(
            cosine_ij, (nframes, nloc, self.a_sel, self.a_sel, 1),
        ) / (xp.pi ** 0.5)

        # loc_mapping
        assert mapping is not None
        flat_map = xp.reshape(mapping, (nframes, -1))
        nlist = xp.reshape(
            xp_take_along_axis(flat_map, xp.reshape(nlist, (nframes, -1)), axis=1),
            nlist.shape,
        )

        # dynamic sel: build graph index
        edge_index, angle_index = get_graph_index(
            nlist, nlist_mask, a_nlist_mask, nall, use_loc_mapping=True,
        )

        # flatten to n_edge / n_angle
        edge_input = edge_input[nlist_mask]
        h2 = h2[nlist_mask]
        sw = sw[nlist_mask]
        a_nlist_mask_2d = xp.logical_and(
            a_nlist_mask[:, :, :, None], a_nlist_mask[:, :, None, :],
        )
        angle_input = angle_input[a_nlist_mask_2d]
        a_sw = (a_sw[:, :, :, None] * a_sw[:, :, None, :])[a_nlist_mask_2d]

        # edge / angle embedding
        edge_ebd = self.edge_embd(edge_input)
        angle_ebd = self.angle_embd(angle_input)

        # layer loop
        mapping_expand = xp.tile(
            xp.expand_dims(mapping, axis=-1), (1, 1, self.n_dim),
        )
        for ll in self.layers:
            node_ebd, edge_ebd, angle_ebd = ll.call(
                node_ebd,
                edge_ebd,
                h2,
                angle_ebd,
                nlist,
                nlist_mask,
                sw,
                a_nlist,
                a_nlist_mask,
                a_sw,
                edge_index=edge_index,
                angle_index=angle_index,
            )

        h2g2 = _cal_hg_dynamic(
            edge_ebd, h2, sw,
            owner=edge_index[0, :],
            num_owner=nframes * nloc,
            nb=nframes, nloc=nloc,
            scale_factor=(self.nnei / self.sel_reduce_factor) ** (-0.5),
        )
        rot_mat = xp.matrix_transpose(h2g2)

        return (
            node_ebd,
            edge_ebd,
            h2,
            xp.reshape(rot_mat, (nframes, nloc, self.dim_emb, 3)),
            sw,
        )

    def has_message_passing(self) -> bool:
        return True

    def need_sorted_nlist_for_lower(self) -> bool:
        return True

    def serialize(self) -> dict:
        return {
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "a_rcut": self.a_rcut,
            "a_rcut_smth": self.a_rcut_smth,
            "a_sel": self.a_sel,
            "ntypes": self.ntypes,
            "nlayers": self.nlayers,
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "a_dim": self.a_dim,
            "axis_neuron": self.axis_neuron,
            "activation_function": self.activation_function,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
            "set_davg_zero": self.set_davg_zero,
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "precision": self.precision,
            "fix_stat_std": self.fix_stat_std,
            "sel_reduce_factor": self.sel_reduce_factor,
            "edge_embd": self.edge_embd.serialize(),
            "angle_embd": self.angle_embd.serialize(),
            "repflow_layers": [layer.serialize() for layer in self.layers],
            "env_mat_edge": self.env_mat_edge.serialize(),
            "env_mat_angle": self.env_mat_angle.serialize(),
            "@variables": {
                "davg": to_numpy_array(self["davg"]),
                "dstd": to_numpy_array(self["dstd"]),
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptBlockRepflowsS":
        data = data.copy()
        edge_embd = NativeLayer.deserialize(data.pop("edge_embd"))
        angle_embd = NativeLayer.deserialize(data.pop("angle_embd"))
        layers = [RepFlowLayerS.deserialize(dd) for dd in data.pop("repflow_layers")]
        env_mat_edge = EnvMat.deserialize(data.pop("env_mat_edge"))
        env_mat_angle = EnvMat.deserialize(data.pop("env_mat_angle"))
        variables = data.pop("@variables")
        davg = variables["davg"]
        dstd = variables["dstd"]
        obj = cls(**data)
        obj.edge_embd = edge_embd
        obj.angle_embd = angle_embd
        obj.layers = layers
        obj.env_mat_edge = env_mat_edge
        obj.env_mat_angle = env_mat_angle
        obj.mean = davg
        obj.stddev = dstd
        return obj


@BaseDescriptor.register("dpa3s")
class DescrptDPA3S(NativeOP, BaseDescriptor):
    _update_sel_cls = UpdateSel

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
        a_sel: int = 20,
        axis_neuron: int = 4,
        sel_reduce_factor: float = 10.0,
        activation_function: str = "silu",
        precision: str = "float64",
        seed: int | list[int] | None = None,
        concat_output_tebd: bool = False,
        add_chg_spin_ebd: bool = False,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        env_protection: float = 0.0,
        exclude_types: list[tuple[int, int]] = [],
        trainable: bool = True,
        type_map: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.ntypes = ntypes
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.nlayers = nlayers
        self.e_rcut = e_rcut
        self.e_rcut_smth = e_rcut_smth
        self.e_sel = e_sel
        self.a_rcut = a_rcut
        self.a_rcut_smth = a_rcut_smth
        self.a_sel = a_sel
        self.axis_neuron = axis_neuron
        self.sel_reduce_factor = sel_reduce_factor
        self.activation_function = activation_function
        self.precision = precision
        self.concat_output_tebd = concat_output_tebd
        self.add_chg_spin_ebd = add_chg_spin_ebd
        self.use_econf_tebd = use_econf_tebd
        self.use_tebd_bias = use_tebd_bias
        self.env_protection = env_protection
        self.exclude_types = exclude_types
        self.trainable = trainable
        self.type_map = type_map
        self.tebd_dim = n_dim

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
            activation_function=activation_function,
            update_residual=0.1,
            update_residual_init="const",
            exclude_types=exclude_types,
            env_protection=env_protection,
            precision=precision,
            sel_reduce_factor=sel_reduce_factor,
            seed=child_seed(seed, 1),
            trainable=trainable,
        )

        self.type_embedding = TypeEmbedNet(
            ntypes=ntypes,
            neuron=[self.tebd_dim],
            padding=True,
            activation_function="Linear",
            precision=precision,
            use_econf_tebd=self.use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
            seed=child_seed(seed, 2),
            trainable=trainable,
        )

        if self.add_chg_spin_ebd:
            self.cs_activation_fn = get_activation_fn(activation_function)
            self.chg_embedding = TypeEmbedNet(
                ntypes=200, neuron=[self.tebd_dim], padding=True,
                activation_function="Linear", precision=precision,
                seed=child_seed(seed, 3),
            )
            self.spin_embedding = TypeEmbedNet(
                ntypes=100, neuron=[self.tebd_dim], padding=True,
                activation_function="Linear", precision=precision,
                seed=child_seed(seed, 4),
            )
            self.mix_cs_mlp = NativeLayer(
                2 * self.tebd_dim, self.tebd_dim,
                precision=precision, seed=child_seed(seed, 5),
            )
        else:
            self.chg_embedding = None
            self.spin_embedding = None
            self.mix_cs_mlp = None

        assert self.repflows.e_rcut >= self.repflows.a_rcut
        assert self.repflows.e_sel >= self.repflows.a_sel

        self.rcut = self.repflows.get_rcut()
        self.rcut_smth = self.repflows.get_rcut_smth()
        self.sel = self.repflows.get_sel()

    def get_rcut(self) -> float:
        return self.rcut

    def get_rcut_smth(self) -> float:
        return self.rcut_smth

    def get_nsel(self) -> int:
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        return self.sel

    def get_ntypes(self) -> int:
        return self.ntypes

    def get_type_map(self) -> list[str]:
        return self.type_map

    def get_dim_out(self) -> int:
        ret = self.repflows.dim_out
        if self.concat_output_tebd:
            ret += self.tebd_dim
        return ret

    def get_dim_emb(self) -> int:
        return self.repflows.dim_emb

    def mixed_types(self) -> bool:
        return True

    def has_message_passing(self) -> bool:
        return self.repflows.has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        return True

    def get_env_protection(self) -> float:
        return self.repflows.get_env_protection()

    def share_params(
        self, base_class: Any, shared_level: int, resume: bool = False,
    ) -> None:
        raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any = None,
    ) -> None:
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
            extend_descrpt_stat(
                repflow, type_map,
                des_with_stat=model_with_new_type_stat.repflows
                if model_with_new_type_stat is not None else None,
            )
        repflow.ntypes = self.ntypes
        repflow.reinit_exclude(self.exclude_types)
        repflow["davg"] = repflow["davg"][remap_index]
        repflow["dstd"] = repflow["dstd"][remap_index]

    @property
    def dim_out(self) -> int:
        return self.get_dim_out()

    @property
    def dim_emb(self) -> int:
        return self.get_dim_emb()

    def compute_input_stats(
        self, merged: list[dict], path: DPPath | None = None,
    ) -> None:
        self.repflows.compute_input_stats(merged, path)

    def set_stat_mean_and_stddev(
        self, mean: list[Array], stddev: list[Array],
    ) -> None:
        self.repflows.mean = mean[0]
        self.repflows.stddev = stddev[0]

    def get_stat_mean_and_stddev(self) -> tuple[list[Array], list[Array]]:
        return [self.repflows.mean], [self.repflows.stddev]

    @cast_precision
    def call(
        self,
        coord_ext: Array,
        atype_ext: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, Array]:
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        nframes, nloc, nnei = nlist.shape

        type_embedding = self.type_embedding.call()
        node_ebd_ext = xp.reshape(
            xp.take(
                type_embedding,
                xp.reshape(xp_take_first_n(atype_ext, 1, nloc), (-1,)),
                axis=0,
            ),
            (nframes, nloc, self.tebd_dim),
        )

        if self.add_chg_spin_ebd:
            assert fparam is not None
            assert self.chg_embedding is not None
            assert self.spin_embedding is not None
            chg_tebd = self.chg_embedding.call()
            spin_tebd = self.spin_embedding.call()
            charge = xp.astype(fparam[:, 0], xp.int64) + 100
            spin = xp.astype(fparam[:, 1], xp.int64)
            chg_ebd = xp.reshape(
                xp.take(chg_tebd, xp.reshape(charge, (-1,)), axis=0),
                (nframes, self.tebd_dim),
            )
            spin_ebd = xp.reshape(
                xp.take(spin_tebd, xp.reshape(spin, (-1,)), axis=0),
                (nframes, self.tebd_dim),
            )
            cs_cat = xp.concat([chg_ebd, spin_ebd], axis=-1)
            sys_cs_embd = self.cs_activation_fn(self.mix_cs_mlp.call(cs_cat))
            node_ebd_ext = node_ebd_ext + xp.expand_dims(sys_cs_embd, axis=1)

        node_ebd_inp = xp_take_first_n(node_ebd_ext, 1, nloc)

        node_ebd, edge_ebd, h2, rot_mat, sw = self.repflows(
            nlist, coord_ext, atype_ext, node_ebd_ext, mapping,
        )
        if self.concat_output_tebd:
            node_ebd = xp.concat([node_ebd, node_ebd_inp], axis=-1)
        return node_ebd, rot_mat, edge_ebd, h2, sw

    def serialize(self) -> dict:
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
            "concat_output_tebd": self.concat_output_tebd,
            "add_chg_spin_ebd": self.add_chg_spin_ebd,
            "use_econf_tebd": self.use_econf_tebd,
            "use_tebd_bias": self.use_tebd_bias,
            "env_protection": self.env_protection,
            "exclude_types": self.exclude_types,
            "trainable": self.trainable,
            "type_map": self.type_map,
            "type_embedding": self.type_embedding.serialize(),
        }
        if self.add_chg_spin_ebd:
            data["chg_embedding"] = self.chg_embedding.serialize()
            data["spin_embedding"] = self.spin_embedding.serialize()
            data["mix_cs_mlp"] = self.mix_cs_mlp.serialize()
        data["repflow_variable"] = {
            "edge_embd": repflows.edge_embd.serialize(),
            "angle_embd": repflows.angle_embd.serialize(),
            "repflow_layers": [layer.serialize() for layer in repflows.layers],
            "env_mat": EnvMat(repflows.rcut, repflows.rcut_smth).serialize(),
            "@variables": {
                "davg": to_numpy_array(repflows["davg"]),
                "dstd": to_numpy_array(repflows["dstd"]),
            },
        }
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
        obj.type_embedding = TypeEmbedNet.deserialize(type_embedding)

        if obj.add_chg_spin_ebd and chg_embedding is not None:
            obj.chg_embedding = TypeEmbedNet.deserialize(chg_embedding)
            obj.spin_embedding = TypeEmbedNet.deserialize(spin_embedding)
            obj.mix_cs_mlp = NativeLayer.deserialize(mix_cs_mlp)

        statistic_repflows = repflow_variable.pop("@variables")
        repflow_variable.pop("env_mat")
        repflow_layers = repflow_variable.pop("repflow_layers")
        obj.repflows.edge_embd = NativeLayer.deserialize(
            repflow_variable.pop("edge_embd"),
        )
        obj.repflows.angle_embd = NativeLayer.deserialize(
            repflow_variable.pop("angle_embd"),
        )
        obj.repflows["davg"] = statistic_repflows["davg"]
        obj.repflows["dstd"] = statistic_repflows["dstd"]
        obj.repflows.layers = [
            RepFlowLayerS.deserialize(layer) for layer in repflow_layers
        ]
        return obj

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[Array, Array]:
        local_jdata_cpy = local_jdata.copy()
        update_sel = cls._update_sel_cls()
        min_nbor_dist, e_sel = update_sel.update_one_sel(
            train_data, type_map, local_jdata_cpy["e_rcut"],
            local_jdata_cpy["e_sel"], True,
        )
        local_jdata_cpy["e_sel"] = e_sel[0]
        min_nbor_dist, a_sel = update_sel.update_one_sel(
            train_data, type_map, local_jdata_cpy["a_rcut"],
            local_jdata_cpy["a_sel"], True,
        )
        local_jdata_cpy["a_sel"] = a_sel[0]
        return local_jdata_cpy, min_nbor_dist
