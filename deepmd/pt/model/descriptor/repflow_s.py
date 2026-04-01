# SPDX-License-Identifier: LGPL-3.0-or-later
"""Simplified repflow descriptor block and layer.

This module provides a streamlined repflow implementation with no feature-flag
branches.  Every forward path uses:

- dynamic_sel (flat tensors: n_edge, n_angle)
- direct concat (optim_update=False)
- res_residual update (inline)
- always update_angle, smooth_edge_update, edge_init_use_dist,
  use_exp_switch, use_loc_mapping
- a_compress_rate=0, n_multi_edge_message=1
"""

from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.descriptor.descriptor import (
    DescriptorBlock,
)
from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pt.model.descriptor.repformer_layer import (
    get_residual,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.model.network.utils import (
    aggregate,
    get_graph_index,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.pt.utils.spin import (
    concat_switch_virtual,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

if not hasattr(torch.ops.deepmd, "border_op"):

    def border_op(
        argument0: Any,
        argument1: Any,
        argument2: Any,
        argument3: Any,
        argument4: Any,
        argument5: Any,
        argument6: Any,
        argument7: Any,
        argument8: Any,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "border_op is not available since customized PyTorch OP library is not built when freezing the model. "
            "See documentation for DPA3 for details."
        )

    # Note: this hack cannot actually save a model that can be run using LAMMPS.
    torch.ops.deepmd.border_op = border_op


class RepFlowLayerS(torch.nn.Module):
    """A single simplified repflow layer.

    This layer always uses dynamic selection (flat tensors), direct concat for
    edge info, res_residual updates, angle updates, and no angular compression.

    Parameters
    ----------
    e_rcut : float
        Edge cut-off radius.
    e_rcut_smth : float
        Edge smoothing start radius.
    e_sel : int or list[int]
        Maximum number of edge neighbors.
    a_rcut : float
        Angle cut-off radius.
    a_rcut_smth : float
        Angle smoothing start radius.
    a_sel : int
        Maximum number of angle neighbors.
    ntypes : int
        Number of element types.
    n_dim : int
        Node representation dimension.
    e_dim : int
        Edge representation dimension.
    a_dim : int
        Angle representation dimension.
    axis_neuron : int
        Number of axis neurons in symmetrization.
    sel_reduce_factor : float
        Reduction factor for dynamic neighbor scaling.
    activation_function : str
        Activation function name.
    precision : str
        Parameter precision.
    seed : int or list[int] or None
        Random seed for initialization.
    trainable : bool
        Whether parameters are trainable.
    """

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
        sel_reduce_factor: float = 10.0,
        activation_function: str = "silu",
        precision: str = "float64",
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.ntypes = ntypes
        e_sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(e_sel)
        assert len(e_sel) == 1
        self.e_sel = e_sel
        self.a_rcut = float(a_rcut)
        self.a_rcut_smth = float(a_rcut_smth)
        self.a_sel = a_sel
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.axis_neuron = axis_neuron
        self.sel_reduce_factor = sel_reduce_factor
        self.activation_function = activation_function
        self.precision = precision
        self.seed = seed
        self.prec = PRECISION_DICT[precision]
        self.act = ActivationFn(activation_function)

        # hardcoded residual config
        self.update_residual = 0.1
        self.update_residual_init = "const"

        # derived dimensions
        self.edge_info_dim = self.n_dim * 2 + self.e_dim
        self.angle_dim = self.a_dim + self.n_dim + 2 * self.e_dim
        self.dynamic_e_sel = self.nnei / self.sel_reduce_factor
        self.dynamic_a_sel = self.a_sel / self.sel_reduce_factor

        # symmetrization input dim: (n_dim + e_dim) * axis_neuron
        self.n_sym_dim = (n_dim + e_dim) * self.axis_neuron

        # -- Linear layers --
        self.node_self_mlp = MLPLayer(
            n_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 0),
            trainable=trainable,
        )
        self.node_sym_linear = MLPLayer(
            self.n_sym_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 1),
            trainable=trainable,
        )
        self.node_edge_linear = MLPLayer(
            self.edge_info_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 2),
            trainable=trainable,
        )
        self.edge_self_linear = MLPLayer(
            self.edge_info_dim,
            e_dim,
            precision=precision,
            seed=child_seed(seed, 3),
            trainable=trainable,
        )
        self.edge_angle_linear1 = MLPLayer(
            self.angle_dim,
            e_dim,
            precision=precision,
            seed=child_seed(seed, 4),
            trainable=trainable,
        )
        self.edge_angle_linear2 = MLPLayer(
            e_dim,
            e_dim,
            precision=precision,
            seed=child_seed(seed, 5),
            trainable=trainable,
        )
        self.angle_self_linear = MLPLayer(
            self.angle_dim,
            a_dim,
            precision=precision,
            seed=child_seed(seed, 6),
            trainable=trainable,
        )

        # -- Residual parameters --
        n_residual: list[nn.Parameter] = []
        # 3 residuals for node: self, sym, edge
        for ii in range(3):
            n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(child_seed(seed, 7), ii),
                    trainable=trainable,
                )
            )
        self.n_residual = nn.ParameterList(n_residual)

        e_residual: list[nn.Parameter] = []
        # 2 residuals for edge: self, angle
        for ii in range(2):
            e_residual.append(
                get_residual(
                    e_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(child_seed(seed, 8), ii),
                    trainable=trainable,
                )
            )
        self.e_residual = nn.ParameterList(e_residual)

        a_residual: list[nn.Parameter] = []
        # 1 residual for angle: self
        a_residual.append(
            get_residual(
                a_dim,
                self.update_residual,
                self.update_residual_init,
                precision=precision,
                seed=child_seed(seed, 9),
                trainable=trainable,
            )
        )
        self.a_residual = nn.ParameterList(a_residual)

    # ------------------------------------------------------------------ #
    #  Static helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cal_hg_dynamic(
        flat_edge_ebd: torch.Tensor,
        flat_h2: torch.Tensor,
        flat_sw: torch.Tensor,
        owner: torch.Tensor,
        num_owner: int,
        nb: int,
        nloc: int,
        scale_factor: float,
    ) -> torch.Tensor:
        """Compute the transposed rotation matrix from flat (dynamic) tensors.

        Parameters
        ----------
        flat_edge_ebd : n_edge x e_dim
        flat_h2 : n_edge x 3
        flat_sw : n_edge
        owner : n_edge  (index into nb*nloc)
        num_owner : int
        nb, nloc : int
        scale_factor : float

        Returns
        -------
        h2g2 : nb x nloc x 3 x e_dim
        """
        n_edge, e_dim = flat_edge_ebd.shape
        # n_edge x e_dim
        flat_edge_ebd = flat_edge_ebd * flat_sw.unsqueeze(-1)
        # n_edge x (3 * e_dim)
        flat_h2g2 = (flat_h2.unsqueeze(-1) * flat_edge_ebd.unsqueeze(-2)).reshape(
            -1, 3 * e_dim
        )
        # nb x nloc x 3 x e_dim
        h2g2 = (
            aggregate(flat_h2g2, owner, average=False, num_owner=num_owner).reshape(
                nb, nloc, 3, e_dim
            )
            * scale_factor
        )
        return h2g2

    @staticmethod
    def _cal_grrg(h2g2: torch.Tensor, axis_neuron: int) -> torch.Tensor:
        """Compute atomic invariant rep from the transposed rotation matrix.

        Parameters
        ----------
        h2g2 : nb x nloc x 3 x e_dim
        axis_neuron : int

        Returns
        -------
        grrg : nb x nloc x (axis_neuron * e_dim)
        """
        nb, nloc, _, e_dim = h2g2.shape
        # nb x nloc x 3 x axis_neuron
        h2g2m = h2g2[..., :axis_neuron]
        # nb x nloc x axis_neuron x e_dim
        g1_13 = torch.matmul(torch.transpose(h2g2m, -1, -2), h2g2) / 3.0
        return g1_13.view(nb, nloc, axis_neuron * e_dim)

    def _symmetrization_op_dynamic(
        self,
        flat_edge_ebd: torch.Tensor,
        flat_h2: torch.Tensor,
        flat_sw: torch.Tensor,
        owner: torch.Tensor,
        num_owner: int,
        nb: int,
        nloc: int,
    ) -> torch.Tensor:
        """Symmetrization operator for dynamic-sel flat tensors.

        Returns
        -------
        grrg : nb x nloc x (axis_neuron * dim)
        """
        h2g2 = self._cal_hg_dynamic(
            flat_edge_ebd,
            flat_h2,
            flat_sw,
            owner,
            num_owner,
            nb,
            nloc,
            self.dynamic_e_sel ** (-0.5),
        )
        return self._cal_grrg(h2g2, self.axis_neuron)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        node_ebd_ext: torch.Tensor,  # nf x nall x n_dim
        edge_ebd: torch.Tensor,  # n_edge x e_dim
        h2: torch.Tensor,  # n_edge x 3
        angle_ebd: torch.Tensor,  # n_angle x a_dim
        nlist: torch.Tensor,  # nf x nloc x nnei
        nlist_mask: torch.Tensor,  # nf x nloc x nnei (unused)
        sw: torch.Tensor,  # n_edge
        a_nlist: torch.Tensor,  # unused
        a_nlist_mask: torch.Tensor,  # unused
        a_sw: torch.Tensor,  # n_angle
        edge_index: torch.Tensor,  # 2 x n_edge
        angle_index: torch.Tensor,  # 3 x n_angle
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of a single simplified repflow layer.

        Returns
        -------
        n_updated : nf x nloc x n_dim
        e_updated : n_edge x e_dim
        a_updated : n_angle x a_dim
        """
        nb, nloc, nnei = nlist.shape
        nall = node_ebd_ext.shape[1]
        node_ebd = node_ebd_ext[:, :nloc, :]
        n_edge = h2.shape[0]
        del a_nlist  # unused

        n2e_index, n_ext2e_index = edge_index[0], edge_index[1]
        n2a_index, eij2a_index, eik2a_index = (
            angle_index[0],
            angle_index[1],
            angle_index[2],
        )

        # n_edge x n_dim
        nei_node_ebd = torch.index_select(
            node_ebd_ext.reshape(-1, self.n_dim), 0, n_ext2e_index
        )

        # ---- 1. Node self MLP ----
        node_self = self.act(self.node_self_mlp(node_ebd))

        # ---- 2. Node symmetrization ----
        sym_edge = self._symmetrization_op_dynamic(
            edge_ebd, h2, sw, n2e_index, nb * nloc, nb, nloc
        )
        sym_node = self._symmetrization_op_dynamic(
            nei_node_ebd, h2, sw, n2e_index, nb * nloc, nb, nloc
        )
        node_sym = self.act(
            self.node_sym_linear(torch.cat([sym_edge, sym_node], dim=-1))
        )

        # ---- 3. Node edge message (direct concat) ----
        node_i = torch.index_select(
            node_ebd.reshape(-1, self.n_dim), 0, n2e_index
        )
        edge_info = torch.cat([node_i, nei_node_ebd, edge_ebd], dim=-1)
        node_edge_update = (
            self.act(self.node_edge_linear(edge_info)) * sw.unsqueeze(-1)
        )
        node_edge_msg = (
            aggregate(
                node_edge_update, n2e_index, average=False, num_owner=nb * nloc
            ).reshape(nb, nloc, -1)
            / self.dynamic_e_sel
        )

        # ---- Update node ----
        n_updated = (
            node_ebd
            + self.n_residual[0] * node_self
            + self.n_residual[1] * node_sym
            + self.n_residual[2] * node_edge_msg
        )

        # ---- 4. Edge self ----
        edge_self = self.act(self.edge_self_linear(edge_info))

        # ---- 5. Angle info (no compression) ----
        node_for_angle = torch.index_select(
            node_ebd.reshape(-1, self.n_dim), 0, n2a_index
        )
        edge_ik = torch.index_select(edge_ebd, 0, eik2a_index)
        edge_ij = torch.index_select(edge_ebd, 0, eij2a_index)
        angle_info = torch.cat(
            [angle_ebd, node_for_angle, edge_ik, edge_ij], dim=-1
        )

        # ---- 6. Edge angle message ----
        edge_angle = self.act(self.edge_angle_linear1(angle_info))
        weighted = edge_angle * a_sw.unsqueeze(-1)
        reduced = aggregate(
            weighted, eij2a_index, average=False, num_owner=n_edge
        ) / (self.dynamic_a_sel**0.5)
        edge_angle_msg = self.act(self.edge_angle_linear2(reduced))

        # ---- Update edge ----
        e_updated = (
            edge_ebd
            + self.e_residual[0] * edge_self
            + self.e_residual[1] * edge_angle_msg
        )

        # ---- 7. Angle self ----
        angle_self = self.act(self.angle_self_linear(angle_info))
        a_updated = angle_ebd + self.a_residual[0] * angle_self

        return n_updated, e_updated, a_updated

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #

    def serialize(self) -> dict:
        """Serialize the layer to a dict."""
        data = {
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
            "sel_reduce_factor": self.sel_reduce_factor,
            "activation_function": self.activation_function,
            "precision": self.precision,
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
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "RepFlowLayerS":
        """Deserialize the layer from a dict."""
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
        variables = data.pop("@variables", {})
        n_residual = variables.get("n_residual", [])
        e_residual = variables.get("e_residual", [])
        a_residual = variables.get("a_residual", [])

        obj = cls(**data)
        obj.node_self_mlp = MLPLayer.deserialize(node_self_mlp)
        obj.node_sym_linear = MLPLayer.deserialize(node_sym_linear)
        obj.node_edge_linear = MLPLayer.deserialize(node_edge_linear)
        obj.edge_self_linear = MLPLayer.deserialize(edge_self_linear)
        obj.edge_angle_linear1 = MLPLayer.deserialize(edge_angle_linear1)
        obj.edge_angle_linear2 = MLPLayer.deserialize(edge_angle_linear2)
        obj.angle_self_linear = MLPLayer.deserialize(angle_self_linear)

        for ii, t in enumerate(obj.n_residual):
            t.data = to_torch_tensor(n_residual[ii])
        for ii, t in enumerate(obj.e_residual):
            t.data = to_torch_tensor(e_residual[ii])
        for ii, t in enumerate(obj.a_residual):
            t.data = to_torch_tensor(a_residual[ii])
        return obj


@DescriptorBlock.register("se_repflow_s")
class DescrptBlockRepflowsS(DescriptorBlock):
    """Simplified repflow descriptor block.

    Always uses dynamic selection, exponential switch, distance-based edge
    initialization, local mapping, res_residual updates, angle updates, and
    no angular compression.

    Parameters
    ----------
    e_rcut : float
        Edge cut-off radius.
    e_rcut_smth : float
        Edge smoothing start radius.
    e_sel : int
        Maximum number of edge neighbors.
    a_rcut : float
        Angle cut-off radius.
    a_rcut_smth : float
        Angle smoothing start radius.
    a_sel : int
        Maximum number of angle neighbors.
    ntypes : int
        Number of element types.
    nlayers : int
        Number of repflow layers.
    n_dim : int
        Node representation dimension.
    e_dim : int
        Edge representation dimension.
    a_dim : int
        Angle representation dimension.
    axis_neuron : int
        Number of axis neurons in symmetrization.
    sel_reduce_factor : float
        Reduction factor for dynamic neighbor scaling.
    activation_function : str
        Activation function name.
    set_davg_zero : bool
        Whether to set the normalization average to zero.
    exclude_types : list[tuple[int, int]]
        Excluded type pairs.
    env_protection : float
        Protection parameter for environment matrix calculations.
    precision : str
        Parameter precision.
    fix_stat_std : float
        If non-zero, use this constant as the normalization std.
    seed : int or list[int] or None
        Random seed for initialization.
    trainable : bool
        Whether parameters are trainable.
    """

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
        sel_reduce_factor: float = 10.0,
        activation_function: str = "silu",
        set_davg_zero: bool = True,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        precision: str = "float64",
        fix_stat_std: float = 0.3,
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

        # for compatibility with DescriptorBlock interface
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
        self.sel_reduce_factor = sel_reduce_factor
        if self.sel_reduce_factor <= 0:
            raise ValueError(
                f"`sel_reduce_factor` must be > 0, got {self.sel_reduce_factor}"
            )

        self.set_davg_zero = set_davg_zero
        self.fix_stat_std = fix_stat_std
        self.set_stddev_constant = fix_stat_std != 0.0
        self.activation_function = activation_function
        self.act = ActivationFn(activation_function)
        self.prec = PRECISION_DICT[precision]

        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)
        self.env_protection = env_protection
        self.precision = precision
        self.epsilon = 1e-4
        self.seed = seed

        # -- Embedding layers --
        self.edge_embd = MLPLayer(
            1,
            self.e_dim,
            precision=precision,
            seed=child_seed(seed, 0),
            trainable=trainable,
        )
        self.angle_embd = MLPLayer(
            1,
            self.a_dim,
            precision=precision,
            bias=False,
            seed=child_seed(seed, 1),
            trainable=trainable,
        )

        # -- RepFlow layers --
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
                    sel_reduce_factor=self.sel_reduce_factor,
                    activation_function=self.activation_function,
                    precision=precision,
                    seed=child_seed(child_seed(seed, 2), ii),
                    trainable=trainable,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

        # -- Stat buffers --
        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(wanted_shape, dtype=self.prec, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=self.prec, device=env.DEVICE)
        if self.set_stddev_constant:
            stddev = stddev * self.fix_stat_std
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.stats = None

    # ------------------------------------------------------------------ #
    #  DescriptorBlock interface
    # ------------------------------------------------------------------ #

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.e_rcut

    def get_rcut_smth(self) -> float:
        """Returns the radius where neighbor info starts to smoothly decay to 0."""
        return self.e_rcut_smth

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return self.sel

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.dim_out

    def get_dim_in(self) -> int:
        """Returns the input dimension."""
        return self.dim_in

    def get_dim_emb(self) -> int:
        """Returns the embedding dimension e_dim."""
        return self.e_dim

    def __setitem__(self, key: str, value: Any) -> None:
        if key in ("avg", "data_avg", "davg"):
            self.mean = value
        elif key in ("std", "data_std", "dstd"):
            self.stddev = value
        else:
            raise KeyError(key)

    def __getitem__(self, key: str) -> Any:
        if key in ("avg", "data_avg", "davg"):
            return self.mean
        elif key in ("std", "data_std", "dstd"):
            return self.stddev
        else:
            raise KeyError(key)

    def mixed_types(self) -> bool:
        """Returns True: the descriptor uses a type-agnostic neighbor list."""
        return True

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.env_protection

    @property
    def dim_out(self) -> int:
        """Returns the output dimension of this descriptor."""
        return self.n_dim

    @property
    def dim_in(self) -> int:
        """Returns the atomic input dimension of this descriptor."""
        return self.n_dim

    @property
    def dim_emb(self) -> int:
        """Returns the embedding dimension e_dim."""
        return self.get_dim_emb()

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: torch.Tensor | None = None,
        mapping: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        parallel_mode = comm_dict is not None
        if not parallel_mode:
            assert mapping is not None
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.view(nframes, -1).shape[1] // 3
        atype = extended_atype[:, :nloc]

        # exclude mask
        exclude_mask = self.emask(nlist, extended_atype)
        nlist = torch.where(exclude_mask != 0, nlist, -1)

        # env_mat for edge
        dmatrix, diff, sw = prod_env_mat(
            extended_coord,
            nlist,
            atype,
            self.mean,
            self.stddev,
            self.e_rcut,
            self.e_rcut_smth,
            protection=self.env_protection,
            use_exp_switch=True,
        )
        nlist_mask = nlist != -1
        sw = torch.squeeze(sw, -1)
        sw = sw.masked_fill(~nlist_mask, 0.0)

        # angle nlist (subset within a_rcut)
        a_dist_mask = (torch.linalg.norm(diff, dim=-1) < self.a_rcut)[
            :, :, : self.a_sel
        ]
        a_nlist = nlist[:, :, : self.a_sel]
        a_nlist = torch.where(a_dist_mask, a_nlist, -1)
        _, a_diff, a_sw = prod_env_mat(
            extended_coord,
            a_nlist,
            atype,
            self.mean[:, : self.a_sel],
            self.stddev[:, : self.a_sel],
            self.a_rcut,
            self.a_rcut_smth,
            protection=self.env_protection,
            use_exp_switch=True,
        )
        a_nlist_mask = a_nlist != -1
        a_sw = torch.squeeze(a_sw, -1)
        a_sw = a_sw.masked_fill(~a_nlist_mask, 0.0)

        # set all padding positions to index 0
        nlist[nlist == -1] = 0
        a_nlist[a_nlist == -1] = 0

        # node embedding from type embedding
        assert extended_atype_embd is not None
        atype_embd = extended_atype_embd[:, :nloc, :]
        assert list(atype_embd.shape) == [nframes, nloc, self.n_dim]
        assert isinstance(atype_embd, torch.Tensor)
        node_ebd = self.act(atype_embd)

        # edge and angle embedding inputs
        edge_input, h2 = torch.split(dmatrix, [1, 3], dim=-1)
        # always use distance for edge init
        edge_input = torch.linalg.norm(diff, dim=-1, keepdim=True)

        # cosine angles
        normalized_diff_i = a_diff / (
            torch.linalg.norm(a_diff, dim=-1, keepdim=True) + 1e-6
        )
        normalized_diff_j = torch.transpose(normalized_diff_i, 2, 3)
        cosine_ij = torch.matmul(normalized_diff_i, normalized_diff_j) * (1 - 1e-6)
        angle_input = cosine_ij.unsqueeze(-1) / (torch.pi**0.5)

        # apply loc_mapping when not parallel
        if not parallel_mode:
            assert mapping is not None
            nlist = torch.gather(
                mapping,
                1,
                index=nlist.reshape(nframes, -1),
            ).reshape(nlist.shape)

        # build graph index
        edge_index, angle_index = get_graph_index(
            nlist,
            nlist_mask,
            a_nlist_mask,
            nall,
            use_loc_mapping=True,
        )

        # flatten all tensors to dynamic-sel form
        # n_edge x 1
        edge_input = edge_input[nlist_mask]
        # n_edge x 3
        h2 = h2[nlist_mask]
        # n_edge
        sw = sw[nlist_mask]
        # nb x nloc x a_nnei x a_nnei mask
        a_nlist_mask_3d = a_nlist_mask[:, :, :, None] & a_nlist_mask[:, :, None, :]
        # n_angle x 1
        angle_input = angle_input[a_nlist_mask_3d]
        # n_angle
        a_sw = (a_sw[:, :, :, None] * a_sw[:, :, None, :])[a_nlist_mask_3d]

        # initial edge / angle embeddings
        # no activation for distance-based init
        edge_ebd = self.edge_embd(edge_input)
        angle_ebd = self.angle_embd(angle_input)

        # prepare mapping for node_ebd_ext construction
        if not parallel_mode:
            assert mapping is not None
            mapping = (
                mapping.view(nframes, nall).unsqueeze(-1).expand(-1, -1, self.n_dim)
            )

        # iterate layers
        for idx, ll in enumerate(self.layers):
            # build node_ebd_ext
            if not parallel_mode:
                assert mapping is not None
                # use_loc_mapping: node_ebd_ext == node_ebd (nloc-indexed)
                node_ebd_ext = node_ebd
            else:
                assert comm_dict is not None
                has_spin = "has_spin" in comm_dict
                if not has_spin:
                    n_padding = nall - nloc
                    node_ebd = torch.nn.functional.pad(
                        node_ebd.squeeze(0), (0, 0, 0, n_padding), value=0.0
                    )
                    real_nloc = nloc
                    real_nall = nall
                else:
                    real_nloc = nloc // 2
                    real_nall = nall // 2
                    real_n_padding = real_nall - real_nloc
                    node_ebd_real, node_ebd_virtual = torch.split(
                        node_ebd, [real_nloc, real_nloc], dim=1
                    )
                    mix_node_ebd = torch.cat(
                        [node_ebd_real, node_ebd_virtual], dim=2
                    )
                    node_ebd = torch.nn.functional.pad(
                        mix_node_ebd.squeeze(0),
                        (0, 0, 0, real_n_padding),
                        value=0.0,
                    )

                assert "send_list" in comm_dict
                assert "send_proc" in comm_dict
                assert "recv_proc" in comm_dict
                assert "send_num" in comm_dict
                assert "recv_num" in comm_dict
                assert "communicator" in comm_dict
                ret = torch.ops.deepmd.border_op(
                    comm_dict["send_list"],
                    comm_dict["send_proc"],
                    comm_dict["recv_proc"],
                    comm_dict["send_num"],
                    comm_dict["recv_num"],
                    node_ebd,
                    comm_dict["communicator"],
                    torch.tensor(
                        real_nloc,
                        dtype=torch.int32,
                        device=torch.device("cpu"),
                    ),
                    torch.tensor(
                        real_nall - real_nloc,
                        dtype=torch.int32,
                        device=torch.device("cpu"),
                    ),
                )
                node_ebd_ext = ret[0].unsqueeze(0)
                if has_spin:
                    n_dim = self.n_dim
                    node_ebd_real_ext, node_ebd_virtual_ext = torch.split(
                        node_ebd_ext, [n_dim, n_dim], dim=2
                    )
                    node_ebd_ext = concat_switch_virtual(
                        node_ebd_real_ext, node_ebd_virtual_ext, real_nloc
                    )

            node_ebd, edge_ebd, angle_ebd = ll.forward(
                node_ebd_ext,
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

        # final rotation matrix
        h2g2 = RepFlowLayerS._cal_hg_dynamic(
            edge_ebd,
            h2,
            sw,
            owner=edge_index[0],
            num_owner=nframes * nloc,
            nb=nframes,
            nloc=nloc,
            scale_factor=(self.nnei / self.sel_reduce_factor) ** (-0.5),
        )
        # (nb x nloc) x e_dim x 3
        rot_mat = torch.permute(h2g2, (0, 1, 3, 2))

        return (
            node_ebd,
            edge_ebd,
            h2,
            rot_mat.view(nframes, nloc, self.dim_emb, 3),
            sw,
        )

    # ------------------------------------------------------------------ #
    #  Statistics
    # ------------------------------------------------------------------ #

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        """Compute the input statistics (e.g. mean and stddev) for the descriptors."""
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
        if not self.set_davg_zero:
            self.mean.copy_(
                torch.tensor(mean, device=env.DEVICE, dtype=self.mean.dtype)
            )
        if not self.set_stddev_constant:
            self.stddev.copy_(
                torch.tensor(stddev, device=env.DEVICE, dtype=self.stddev.dtype)
            )

    def get_stats(self) -> dict[str, StatItem]:
        """Get the statistics of the descriptor."""
        if self.stats is None:
            raise RuntimeError(
                "The statistics of the descriptor has not been computed."
            )
        return self.stats

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor block has message passing."""
        return True

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor block needs sorted nlist when using forward_lower."""
        return True
