# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA3S V7 variant: Complete MatRIS 4-subblock structure.

Self-contained file. Copied from V6, with:
- Symmetrization + 3-way node update REPLACED by atom graph attention + refinement
- Full 4-subblock per layer: line_attn → atom_attn → line_refine → atom_refine
- MatRIS-style residuals: update + weight * old (element-wise, init 1.0)
- All other components (Bessel/Fourier, learnable envelope, GatedMLP) preserved.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import child_seed
from deepmd.pt.model.descriptor.descriptor import DescriptorBlock
from deepmd.pt.model.descriptor.env_mat import prod_env_mat
from deepmd.pt.model.descriptor.repformer_layer import get_residual
from deepmd.pt.model.network.mlp import MLPLayer
from deepmd.pt.model.network.utils import (
    BesselExpansion,
    FourierExpansion,
    GatedMLP,
    aggregate,
    dimwise_softmax,
    get_graph_index,
)
from deepmd.pt.utils import env
from deepmd.pt.utils.env import PRECISION_DICT
from deepmd.pt.utils.env_mat_stat import EnvMatStatSe
from deepmd.pt.utils.exclude_mask import PairExcludeMask
from deepmd.pt.utils.spin import concat_switch_virtual
from deepmd.pt.utils.utils import ActivationFn, to_numpy_array, to_torch_tensor
from deepmd.utils.env_mat_stat import StatItem
from deepmd.utils.path import DPPath
from deepmd.utils.version import check_version_compatibility

if not hasattr(torch.ops.deepmd, "border_op"):

    def border_op(
        argument0, argument1, argument2, argument3, argument4,
        argument5, argument6, argument7, argument8,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "border_op is not available since customized PyTorch OP library "
            "is not built when freezing the model."
        )

    torch.ops.deepmd.border_op = border_op
# SPLIT_MARKER_1


class RepFlowLayerV7(torch.nn.Module):
    """A single repflow layer for V7 variant: complete MatRIS 4-subblock.

    4 sub-modules per layer (executed sequentially):
    1. Line Graph Attention: dimwise softmax on angle graph
    2. Atom Graph Attention: dimwise softmax on atom graph (replaces symmetrization)
    3. Line Graph Refinement: learnable envelope on angle graph
    4. Atom Graph Refinement: learnable envelope on atom graph

    MatRIS-style residuals: update + weight * old (element-wise, init 1.0).
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
        num_edge_basis: int = 7,
        num_angle_basis: int = 7,
        gated_hidden_dim: int = 128,
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
        self.num_edge_basis = num_edge_basis
        self.num_angle_basis = num_angle_basis
        self.gated_hidden_dim = gated_hidden_dim
        self.activation_function = activation_function
        self.precision = precision
        self.seed = seed
        self.prec = PRECISION_DICT[precision]
        self.act = ActivationFn(activation_function)

        self.edge_info_dim = self.n_dim * 2 + self.e_dim
        self.dynamic_e_sel = self.nnei / self.sel_reduce_factor
        self.dynamic_a_sel = self.a_sel / self.sel_reduce_factor

        # ===== Sub-module 1: Line Graph Attention =====
        self.line_attn_edge_mlp = GatedMLP(
            a_dim + 2 * e_dim, gated_hidden_dim, e_dim,
            precision=precision, seed=child_seed(seed, 30),
        )
        self.line_attn_weight_linear = MLPLayer(
            a_dim, e_dim, bias=False, precision=precision,
            seed=child_seed(seed, 31), trainable=trainable,
        )
        self.line_attn_angle_mlp = MLPLayer(
            a_dim + 2 * e_dim, a_dim, precision=precision,
            seed=child_seed(seed, 32), trainable=trainable,
        )

        # ===== Sub-module 2: Atom Graph Attention =====
        self.atom_attn_edge_mlp = GatedMLP(
            self.edge_info_dim, gated_hidden_dim, e_dim,
            precision=precision, seed=child_seed(seed, 40),
        )
        self.atom_attn_weight_linear = MLPLayer(
            e_dim, e_dim, bias=False, precision=precision,
            seed=child_seed(seed, 41), trainable=trainable,
        )
        self.atom_attn_node_mlp = GatedMLP(
            n_dim + e_dim, gated_hidden_dim, n_dim,
            precision=precision, seed=child_seed(seed, 42),
        )

        # ===== Sub-module 3: Line Graph Refinement =====
        self.line_refine_mlp = GatedMLP(
            n_dim + a_dim + 2 * e_dim, gated_hidden_dim, e_dim,
            precision=precision, seed=child_seed(seed, 50),
        )
        self.line_refine_envelope = MLPLayer(
            num_angle_basis, e_dim, bias=False, precision=precision,
            seed=child_seed(seed, 51), trainable=trainable,
        )
        self.line_refine_node_ffn = MLPLayer(
            e_dim, e_dim, precision=precision,
            seed=child_seed(seed, 52), trainable=trainable,
        )
        self.line_refine_edge_ffn = MLPLayer(
            e_dim, a_dim, precision=precision,
            seed=child_seed(seed, 53), trainable=trainable,
        )

        # ===== Sub-module 4: Atom Graph Refinement =====
        self.atom_refine_mlp = GatedMLP(
            self.edge_info_dim, gated_hidden_dim, e_dim,
            precision=precision, seed=child_seed(seed, 60),
        )
        self.atom_refine_envelope = MLPLayer(
            num_edge_basis, e_dim, bias=False, precision=precision,
            seed=child_seed(seed, 61), trainable=trainable,
        )
        self.atom_refine_node_ffn = MLPLayer(
            e_dim, n_dim, precision=precision,
            seed=child_seed(seed, 62), trainable=trainable,
        )
        self.atom_refine_edge_ffn = MLPLayer(
            e_dim, e_dim, precision=precision,
            seed=child_seed(seed, 63), trainable=trainable,
        )

        # ===== MatRIS-style element-wise residual weights (init 1.0) =====
        prec_dt = PRECISION_DICT[precision]
        # Line attn: edge, angle
        self.line_attn_edge_res = nn.Parameter(torch.ones(1, e_dim, dtype=prec_dt, device="cpu"))
        self.line_attn_angle_res = nn.Parameter(torch.ones(1, a_dim, dtype=prec_dt, device="cpu"))
        # Atom attn: node, edge
        self.atom_attn_node_res = nn.Parameter(torch.ones(1, n_dim, dtype=prec_dt, device="cpu"))
        self.atom_attn_edge_res = nn.Parameter(torch.ones(1, e_dim, dtype=prec_dt, device="cpu"))
        # Line refine: edge(=node in line graph), angle(=edge in line graph)
        self.line_refine_edge_res = nn.Parameter(torch.ones(1, e_dim, dtype=prec_dt, device="cpu"))
        self.line_refine_angle_res = nn.Parameter(torch.ones(1, a_dim, dtype=prec_dt, device="cpu"))
        # Atom refine: node, edge
        self.atom_refine_node_res = nn.Parameter(torch.ones(1, n_dim, dtype=prec_dt, device="cpu"))
        self.atom_refine_edge_res = nn.Parameter(torch.ones(1, e_dim, dtype=prec_dt, device="cpu"))
# SPLIT_MARKER_2

    # ------------------------------------------------------------------ #
    #  Static helpers (identical to baseline)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cal_hg_dynamic(
        flat_edge_ebd, flat_h2, flat_sw, owner, num_owner, nb, nloc, scale_factor,
    ):
        n_edge, e_dim = flat_edge_ebd.shape
        flat_edge_ebd = flat_edge_ebd * flat_sw.unsqueeze(-1)
        flat_h2g2 = (flat_h2.unsqueeze(-1) * flat_edge_ebd.unsqueeze(-2)).reshape(
            -1, 3 * e_dim
        )
        h2g2 = (
            aggregate(flat_h2g2, owner, average=False, num_owner=num_owner).reshape(
                nb, nloc, 3, e_dim
            )
            * scale_factor
        )
        return h2g2

    @staticmethod
    def _cal_grrg(h2g2, axis_neuron):
        nb, nloc, _, e_dim = h2g2.shape
        h2g2m = h2g2[..., :axis_neuron]
        g1_13 = torch.matmul(torch.transpose(h2g2m, -1, -2), h2g2) / 3.0
        return g1_13.view(nb, nloc, axis_neuron * e_dim)

    # ------------------------------------------------------------------ #
    #  Forward: MatRIS 4-subblock structure
    # ------------------------------------------------------------------ #

    def forward(
        self,
        node_ebd_ext, edge_ebd, h2, angle_ebd,
        nlist, nlist_mask, sw, a_nlist, a_nlist_mask, a_sw,
        edge_index, angle_index,
        edge_rbf=None,
        angle_rbf=None,
    ):
        nb, nloc, nnei = nlist.shape
        node_ebd = node_ebd_ext[:, :nloc, :]
        n_edge = h2.shape[0]
        del a_nlist

        n2e_index, n_ext2e_index = edge_index[0], edge_index[1]
        n2a_index, eij2a_index, eik2a_index = (
            angle_index[0], angle_index[1], angle_index[2],
        )

        # ===== Step 1: Line Graph Attention =====
        edge_ik = torch.index_select(edge_ebd, 0, eik2a_index)
        edge_ij = torch.index_select(edge_ebd, 0, eij2a_index)
        line_attn_info = torch.cat([angle_ebd, edge_ik, edge_ij], dim=-1)

        # Dimwise softmax attention (multiply a_sw before softmax for smoothness)
        attn_logits = self.line_attn_weight_linear(angle_ebd) * a_sw.unsqueeze(-1)
        alpha = dimwise_softmax(attn_logits, eij2a_index, num_segment=n_edge)

        line_edge_update = self.line_attn_edge_mlp(line_attn_info)
        line_edge_msg = alpha * line_edge_update * a_sw.unsqueeze(-1)
        line_edge_agg = aggregate(
            line_edge_msg, eij2a_index, average=False, num_owner=n_edge
        ) / (self.dynamic_a_sel ** 0.5)

        line_angle_update = self.act(self.line_attn_angle_mlp(line_attn_info))

        # MatRIS residual: update + weight * old
        edge_ebd = line_edge_agg + self.line_attn_edge_res * edge_ebd
        angle_ebd = line_angle_update + self.line_attn_angle_res * angle_ebd

        # ===== Step 2: Atom Graph Attention =====
        nei_node_ebd = torch.index_select(
            node_ebd_ext.reshape(-1, self.n_dim), 0, n_ext2e_index
        )
        node_i = torch.index_select(
            node_ebd.reshape(-1, self.n_dim), 0, n2e_index
        )
        edge_info = torch.cat([node_i, nei_node_ebd, edge_ebd], dim=-1)

        # Edge update
        atom_edge_update = self.atom_attn_edge_mlp(edge_info)

        # Dimwise softmax attention on atom graph (source direction only)
        atom_attn_logits = self.atom_attn_weight_linear(edge_ebd) * sw.unsqueeze(-1)
        atom_alpha = dimwise_softmax(atom_attn_logits, n2e_index, num_segment=nb * nloc)

        atom_msg = atom_alpha * atom_edge_update * sw.unsqueeze(-1)
        atom_node_agg = (
            aggregate(atom_msg, n2e_index, average=False, num_owner=nb * nloc)
            .reshape(nb, nloc, -1)
            / self.dynamic_e_sel
        )

        # Node update via MLP on [node, aggregated_msg]
        node_update = self.atom_attn_node_mlp(
            torch.cat([node_ebd, atom_node_agg], dim=-1)
        )

        # MatRIS residual
        node_ebd = node_update + self.atom_attn_node_res * node_ebd
        edge_ebd = atom_edge_update + self.atom_attn_edge_res * edge_ebd

        # ===== Step 3: Line Graph Refinement =====
        edge_ik = torch.index_select(edge_ebd, 0, eik2a_index)
        edge_ij = torch.index_select(edge_ebd, 0, eij2a_index)
        node_for_angle = torch.index_select(
            node_ebd.reshape(-1, self.n_dim), 0, n2a_index
        )
        refine_line_info = torch.cat(
            [angle_ebd, node_for_angle, edge_ik, edge_ij], dim=-1
        )
        refine_line_feat = self.line_refine_mlp(refine_line_info)

        # Learnable envelope for line graph
        if angle_rbf is not None:
            line_envelope = torch.sigmoid(self.line_refine_envelope(angle_rbf))
            refine_line_smoothed = refine_line_feat * line_envelope * a_sw.unsqueeze(-1)
        else:
            refine_line_smoothed = refine_line_feat * a_sw.unsqueeze(-1)

        line_refine_agg = aggregate(
            refine_line_smoothed, eij2a_index, average=False, num_owner=n_edge
        ) / (self.dynamic_a_sel ** 0.5)

        delta_edge_lr = self.act(self.line_refine_node_ffn(line_refine_agg))
        delta_angle_lr = self.act(self.line_refine_edge_ffn(refine_line_feat))

        edge_ebd = delta_edge_lr + self.line_refine_edge_res * edge_ebd
        angle_ebd = delta_angle_lr + self.line_refine_angle_res * angle_ebd

        # ===== Step 4: Atom Graph Refinement =====
        nei_node_ebd = torch.index_select(
            node_ebd_ext.reshape(-1, self.n_dim), 0, n_ext2e_index
        )
        # Note: use updated node_ebd for node_i
        node_i = torch.index_select(
            node_ebd.reshape(-1, self.n_dim), 0, n2e_index
        )
        refine_atom_info = torch.cat([node_i, nei_node_ebd, edge_ebd], dim=-1)
        refine_atom_feat = self.atom_refine_mlp(refine_atom_info)

        # Learnable envelope for atom graph
        if edge_rbf is not None:
            atom_envelope = torch.sigmoid(self.atom_refine_envelope(edge_rbf))
            refine_atom_smoothed = refine_atom_feat * atom_envelope * sw.unsqueeze(-1)
        else:
            refine_atom_smoothed = refine_atom_feat * sw.unsqueeze(-1)

        atom_refine_agg = (
            aggregate(
                refine_atom_smoothed, n2e_index, average=False, num_owner=nb * nloc
            ).reshape(nb, nloc, -1)
            / self.dynamic_e_sel
        )

        delta_node_ar = self.act(self.atom_refine_node_ffn(atom_refine_agg))
        delta_edge_ar = self.act(self.atom_refine_edge_ffn(refine_atom_feat))

        n_updated = delta_node_ar + self.atom_refine_node_res * node_ebd
        e_updated = delta_edge_ar + self.atom_refine_edge_res * edge_ebd
        a_updated = angle_ebd  # angle already updated in steps 1 and 3

        return n_updated, e_updated, a_updated
# SPLIT_MARKER_3

    def serialize(self) -> dict:
        data = {
            "@class": "RepFlowLayerV7",
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
            "num_edge_basis": self.num_edge_basis,
            "num_angle_basis": self.num_angle_basis,
            "gated_hidden_dim": self.gated_hidden_dim,
            "activation_function": self.activation_function,
            "precision": self.precision,
            # Sub-module 1: Line Graph Attention
            "line_attn_edge_mlp": self.line_attn_edge_mlp.serialize(),
            "line_attn_weight_linear": self.line_attn_weight_linear.serialize(),
            "line_attn_angle_mlp": self.line_attn_angle_mlp.serialize(),
            # Sub-module 2: Atom Graph Attention
            "atom_attn_edge_mlp": self.atom_attn_edge_mlp.serialize(),
            "atom_attn_weight_linear": self.atom_attn_weight_linear.serialize(),
            "atom_attn_node_mlp": self.atom_attn_node_mlp.serialize(),
            # Sub-module 3: Line Graph Refinement
            "line_refine_mlp": self.line_refine_mlp.serialize(),
            "line_refine_envelope": self.line_refine_envelope.serialize(),
            "line_refine_node_ffn": self.line_refine_node_ffn.serialize(),
            "line_refine_edge_ffn": self.line_refine_edge_ffn.serialize(),
            # Sub-module 4: Atom Graph Refinement
            "atom_refine_mlp": self.atom_refine_mlp.serialize(),
            "atom_refine_envelope": self.atom_refine_envelope.serialize(),
            "atom_refine_node_ffn": self.atom_refine_node_ffn.serialize(),
            "atom_refine_edge_ffn": self.atom_refine_edge_ffn.serialize(),
            # Residual weights
            "@variables": {
                "line_attn_edge_res": to_numpy_array(self.line_attn_edge_res),
                "line_attn_angle_res": to_numpy_array(self.line_attn_angle_res),
                "atom_attn_node_res": to_numpy_array(self.atom_attn_node_res),
                "atom_attn_edge_res": to_numpy_array(self.atom_attn_edge_res),
                "line_refine_edge_res": to_numpy_array(self.line_refine_edge_res),
                "line_refine_angle_res": to_numpy_array(self.line_refine_angle_res),
                "atom_refine_node_res": to_numpy_array(self.atom_refine_node_res),
                "atom_refine_edge_res": to_numpy_array(self.atom_refine_edge_res),
            },
        }
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "RepFlowLayerV7":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        # Pop all serialized sub-modules
        line_attn_edge_mlp = data.pop("line_attn_edge_mlp")
        line_attn_weight_linear = data.pop("line_attn_weight_linear")
        line_attn_angle_mlp = data.pop("line_attn_angle_mlp")
        atom_attn_edge_mlp = data.pop("atom_attn_edge_mlp")
        atom_attn_weight_linear = data.pop("atom_attn_weight_linear")
        atom_attn_node_mlp = data.pop("atom_attn_node_mlp")
        line_refine_mlp = data.pop("line_refine_mlp")
        line_refine_envelope = data.pop("line_refine_envelope")
        line_refine_node_ffn = data.pop("line_refine_node_ffn")
        line_refine_edge_ffn = data.pop("line_refine_edge_ffn")
        atom_refine_mlp = data.pop("atom_refine_mlp")
        atom_refine_envelope = data.pop("atom_refine_envelope")
        atom_refine_node_ffn = data.pop("atom_refine_node_ffn")
        atom_refine_edge_ffn = data.pop("atom_refine_edge_ffn")
        variables = data.pop("@variables", {})

        obj = cls(**data)
        obj.line_attn_edge_mlp = GatedMLP.deserialize(line_attn_edge_mlp)
        obj.line_attn_weight_linear = MLPLayer.deserialize(line_attn_weight_linear)
        obj.line_attn_angle_mlp = MLPLayer.deserialize(line_attn_angle_mlp)
        obj.atom_attn_edge_mlp = GatedMLP.deserialize(atom_attn_edge_mlp)
        obj.atom_attn_weight_linear = MLPLayer.deserialize(atom_attn_weight_linear)
        obj.atom_attn_node_mlp = GatedMLP.deserialize(atom_attn_node_mlp)
        obj.line_refine_mlp = GatedMLP.deserialize(line_refine_mlp)
        obj.line_refine_envelope = MLPLayer.deserialize(line_refine_envelope)
        obj.line_refine_node_ffn = MLPLayer.deserialize(line_refine_node_ffn)
        obj.line_refine_edge_ffn = MLPLayer.deserialize(line_refine_edge_ffn)
        obj.atom_refine_mlp = GatedMLP.deserialize(atom_refine_mlp)
        obj.atom_refine_envelope = MLPLayer.deserialize(atom_refine_envelope)
        obj.atom_refine_node_ffn = MLPLayer.deserialize(atom_refine_node_ffn)
        obj.atom_refine_edge_ffn = MLPLayer.deserialize(atom_refine_edge_ffn)

        for key in [
            "line_attn_edge_res", "line_attn_angle_res",
            "atom_attn_node_res", "atom_attn_edge_res",
            "line_refine_edge_res", "line_refine_angle_res",
            "atom_refine_node_res", "atom_refine_edge_res",
        ]:
            if key in variables:
                getattr(obj, key).data = to_torch_tensor(variables[key])
        return obj


# SPLIT_MARKER_4


@DescriptorBlock.register("se_repflow_v7")
class DescrptBlockRepflowV7(DescriptorBlock):
    """V7 repflow descriptor block: Complete MatRIS 4-subblock structure.

    Based on V4, with node_edge, edge_self, edge_angle1 replaced by GatedMLP.
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
        num_edge_basis: int = 7,
        num_angle_basis: int = 7,
        envelope_exponent: int = 8,
        learnable_basis: bool = True,
        gated_hidden_dim: int = 128,
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
        self.num_edge_basis = num_edge_basis
        self.num_angle_basis = num_angle_basis
        self.envelope_exponent = envelope_exponent
        self.learnable_basis = learnable_basis
        self.gated_hidden_dim = gated_hidden_dim

        self.set_davg_zero = set_davg_zero
        self.fix_stat_std = fix_stat_std
        self.set_stddev_constant = fix_stat_std != 0.0
        self.activation_function = activation_function
        self.act = ActivationFn(activation_function)
        self.prec = PRECISION_DICT[precision]

        self.reinit_exclude(exclude_types)
        self.env_protection = env_protection
        self.precision = precision
        self.epsilon = 1e-4
        self.seed = seed

        # -- V4: Bessel + Fourier basis --
        self.edge_basis = BesselExpansion(
            num_radial=num_edge_basis,
            cutoff=self.e_rcut,
            envelope_exponent=envelope_exponent,
            learnable=learnable_basis,
        )
        max_f = (num_angle_basis - 1) // 2
        self.angle_basis = FourierExpansion(
            max_f=max_f, learnable=learnable_basis,
        )
        actual_angle_dim = self.angle_basis.num_output

        # -- Embedding layers --
        self.edge_embd = MLPLayer(
            num_edge_basis, self.e_dim,
            precision=precision, seed=child_seed(seed, 0), trainable=trainable,
        )
        self.angle_embd = MLPLayer(
            actual_angle_dim, self.a_dim, bias=False,
            precision=precision, seed=child_seed(seed, 1), trainable=trainable,
        )

        # -- RepFlow layers --
        layers = []
        for ii in range(nlayers):
            layers.append(
                RepFlowLayerV7(
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
                    num_edge_basis=num_edge_basis,
                    num_angle_basis=actual_angle_dim,
                    gated_hidden_dim=gated_hidden_dim,
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
# SPLIT_MARKER_5

    # ------------------------------------------------------------------ #
    #  DescriptorBlock interface
    # ------------------------------------------------------------------ #

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

    def __setitem__(self, key, value):
        if key in ("avg", "data_avg", "davg"):
            self.mean = value
        elif key in ("std", "data_std", "dstd"):
            self.stddev = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
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

    def reinit_exclude(self, exclude_types: list[tuple[int, int]] = []):
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
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None,
               torch.Tensor | None, torch.Tensor | None]:
        parallel_mode = comm_dict is not None
        if not parallel_mode:
            assert mapping is not None
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.view(nframes, -1).shape[1] // 3
        atype = extended_atype[:, :nloc]

        exclude_mask = self.emask(nlist, extended_atype)
        nlist = torch.where(exclude_mask != 0, nlist, -1)

        dmatrix, diff, sw = prod_env_mat(
            extended_coord, nlist, atype,
            self.mean, self.stddev,
            self.e_rcut, self.e_rcut_smth,
            protection=self.env_protection, use_exp_switch=True,
        )
        nlist_mask = nlist != -1
        sw = torch.squeeze(sw, -1)
        sw = sw.masked_fill(~nlist_mask, 0.0)

        # angle nlist
        a_dist_mask = (torch.linalg.norm(diff, dim=-1) < self.a_rcut)[
            :, :, : self.a_sel
        ]
        a_nlist = nlist[:, :, : self.a_sel]
        a_nlist = torch.where(a_dist_mask, a_nlist, -1)
        _, a_diff, a_sw = prod_env_mat(
            extended_coord, a_nlist, atype,
            self.mean[:, : self.a_sel], self.stddev[:, : self.a_sel],
            self.a_rcut, self.a_rcut_smth,
            protection=self.env_protection, use_exp_switch=True,
        )
        a_nlist_mask = a_nlist != -1
        a_sw = torch.squeeze(a_sw, -1)
        a_sw = a_sw.masked_fill(~a_nlist_mask, 0.0)

        nlist[nlist == -1] = 0
        a_nlist[a_nlist == -1] = 0

        # node embedding
        assert extended_atype_embd is not None
        atype_embd = extended_atype_embd[:, :nloc, :]
        assert list(atype_embd.shape) == [nframes, nloc, self.n_dim]
        node_ebd = self.act(atype_embd)

        # V4: Bessel edge embedding
        _, h2 = torch.split(dmatrix, [1, 3], dim=-1)
        edge_dist = torch.linalg.norm(diff, dim=-1)

        # V4: Fourier angle embedding
        normalized_diff_i = a_diff / (
            torch.linalg.norm(a_diff, dim=-1, keepdim=True) + 1e-6
        )
        normalized_diff_j = torch.transpose(normalized_diff_i, 2, 3)
        cosine_ij = torch.matmul(normalized_diff_i, normalized_diff_j) * (1 - 1e-6)
        # clamp to avoid acos instability at ±1
        cosine_ij = torch.clamp(cosine_ij, -1.0 + 1e-6, 1.0 - 1e-6)
        angle_ij = torch.acos(cosine_ij)

        if not parallel_mode:
            assert mapping is not None
            nlist = torch.gather(
                mapping, 1, index=nlist.reshape(nframes, -1),
            ).reshape(nlist.shape)

        edge_index, angle_index = get_graph_index(
            nlist, nlist_mask, a_nlist_mask, nall, use_loc_mapping=True,
        )

        # flatten to dynamic-sel
        h2 = h2[nlist_mask]
        sw = sw[nlist_mask]
        edge_dist_flat = edge_dist[nlist_mask]
        a_nlist_mask_3d = a_nlist_mask[:, :, :, None] & a_nlist_mask[:, :, None, :]
        angle_flat = angle_ij[a_nlist_mask_3d]
        a_sw = (a_sw[:, :, :, None] * a_sw[:, :, None, :])[a_nlist_mask_3d]

        # V4: basis expansion + embedding
        edge_rbf = self.edge_basis(edge_dist_flat)  # n_edge x num_edge_basis
        edge_ebd = self.edge_embd(edge_rbf)  # n_edge x e_dim
        angle_feat = self.angle_basis(angle_flat)  # n_angle x num_angle_basis
        angle_ebd = self.angle_embd(angle_feat)  # n_angle x a_dim

        if not parallel_mode:
            assert mapping is not None
            mapping = (
                mapping.view(nframes, nall).unsqueeze(-1).expand(-1, -1, self.n_dim)
            )

        # iterate layers
        for idx, ll in enumerate(self.layers):
            if not parallel_mode:
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
                        (0, 0, 0, real_n_padding), value=0.0,
                    )
                ret = torch.ops.deepmd.border_op(
                    comm_dict["send_list"], comm_dict["send_proc"],
                    comm_dict["recv_proc"], comm_dict["send_num"],
                    comm_dict["recv_num"], node_ebd, comm_dict["communicator"],
                    torch.tensor(real_nloc, dtype=torch.int32, device=torch.device("cpu")),
                    torch.tensor(real_nall - real_nloc, dtype=torch.int32, device=torch.device("cpu")),
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
                node_ebd_ext, edge_ebd, h2, angle_ebd,
                nlist, nlist_mask, sw, a_nlist, a_nlist_mask, a_sw,
                edge_index=edge_index, angle_index=angle_index,
                edge_rbf=edge_rbf,
                angle_rbf=angle_feat,  # V7: pass raw Fourier basis for line refine envelope
            )

        # final rotation matrix
        h2g2 = RepFlowLayerV7._cal_hg_dynamic(
            edge_ebd, h2, sw,
            owner=edge_index[0], num_owner=nframes * nloc,
            nb=nframes, nloc=nloc,
            scale_factor=(self.nnei / self.sel_reduce_factor) ** (-0.5),
        )
        rot_mat = torch.permute(h2g2, (0, 1, 3, 2))

        return (
            node_ebd, edge_ebd, h2,
            rot_mat.view(nframes, nloc, self.dim_emb, 3), sw,
        )
# SPLIT_MARKER_6

    def compute_input_stats(self, merged, path=None):
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
            self.mean.copy_(torch.tensor(mean, device=env.DEVICE, dtype=self.mean.dtype))
        if not self.set_stddev_constant:
            self.stddev.copy_(torch.tensor(stddev, device=env.DEVICE, dtype=self.stddev.dtype))

    def get_stats(self):
        if self.stats is None:
            raise RuntimeError("The statistics of the descriptor has not been computed.")
        return self.stats

    def has_message_passing(self) -> bool:
        return True

    def need_sorted_nlist_for_lower(self) -> bool:
        return True


# -- Top-level descriptor --

from deepmd.pt.model.network.network import (  # noqa: E402
    TypeEmbedNet,
    TypeEmbedNetConsistent,
)
from deepmd.pt.utils.update_sel import UpdateSel  # noqa: E402
from deepmd.utils.data_system import DeepmdDataSystem  # noqa: E402
from deepmd.utils.finetune import (  # noqa: E402
    get_index_between_two_maps,
    map_pair_exclude_types,
)

from .base_descriptor import BaseDescriptor  # noqa: E402
from .descriptor import extend_descrpt_stat  # noqa: E402


@BaseDescriptor.register("dpa3s_v7_matris")
class DescrptDPA3V7(BaseDescriptor, torch.nn.Module):
    r"""DPA3S V7 descriptor: Complete MatRIS 4-subblock structure."""

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
        num_edge_basis: int = 7,
        num_angle_basis: int = 7,
        envelope_exponent: int = 8,
        learnable_basis: bool = True,
        gated_hidden_dim: int = 128,
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

        self.repflows = DescrptBlockRepflowV7(
            e_rcut=e_rcut, e_rcut_smth=e_rcut_smth, e_sel=e_sel,
            a_rcut=a_rcut, a_rcut_smth=a_rcut_smth, a_sel=a_sel,
            ntypes=ntypes, nlayers=nlayers,
            n_dim=n_dim, e_dim=e_dim, a_dim=a_dim,
            axis_neuron=axis_neuron, sel_reduce_factor=sel_reduce_factor,
            num_edge_basis=num_edge_basis, num_angle_basis=num_angle_basis,
            envelope_exponent=envelope_exponent, learnable_basis=learnable_basis,
            gated_hidden_dim=gated_hidden_dim,
            activation_function=activation_function,
            exclude_types=exclude_types, env_protection=env_protection,
            precision=precision, seed=child_seed(seed, 1), trainable=trainable,
        )

        self.type_embedding = TypeEmbedNet(
            ntypes, n_dim, precision=precision,
            seed=child_seed(seed, 2),
            use_econf_tebd=use_econf_tebd, use_tebd_bias=use_tebd_bias,
            type_map=type_map, trainable=trainable,
        )

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
        self.num_edge_basis = num_edge_basis
        self.num_angle_basis = num_angle_basis
        self.envelope_exponent = envelope_exponent
        self.learnable_basis = learnable_basis
        self.gated_hidden_dim = gated_hidden_dim
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

        if self.add_chg_spin_ebd:
            self.act = ActivationFn(activation_function)
            self.chg_embedding = TypeEmbedNet(
                200, n_dim, precision=precision, seed=child_seed(seed, 3),
            )
            self.spin_embedding = TypeEmbedNet(
                100, n_dim, precision=precision, seed=child_seed(seed, 4),
            )
            self.mix_cs_mlp = MLPLayer(
                2 * n_dim, n_dim, precision=precision, seed=child_seed(seed, 5),
            )
        else:
            self.chg_embedding = None
            self.spin_embedding = None
            self.mix_cs_mlp = None

        assert e_rcut >= a_rcut
        assert e_sel >= a_sel

        self.rcut = self.repflows.get_rcut()
        self.rcut_smth = self.repflows.get_rcut_smth()
        self.sel = self.repflows.get_sel()

        for param in self.parameters():
            param.requires_grad = trainable
# SPLIT_MARKER_7

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
        return True

    def need_sorted_nlist_for_lower(self) -> bool:
        return True

    def get_env_protection(self) -> float:
        return self.repflows.get_env_protection()

    @property
    def dim_out(self) -> int:
        return self.get_dim_out()

    @property
    def dim_emb(self) -> int:
        return self.get_dim_emb()

    def compute_input_stats(self, merged, path=None):
        descrpt_list = [self.repflows]
        for ii, descrpt in enumerate(descrpt_list):
            descrpt.compute_input_stats(merged, path)

    def set_stat_mean_and_stddev(self, mean, stddev):
        descrpt_list = [self.repflows]
        for ii, descrpt in enumerate(descrpt_list):
            descrpt.mean = mean[ii]
            descrpt.stddev = stddev[ii]

    def get_stat_mean_and_stddev(self):
        return [self.repflows.mean], [self.repflows.stddev]

    def share_params(self, base_class, shared_level, resume=False):
        assert self.__class__ == base_class.__class__
        if shared_level == 0:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            self.repflows.share_params(base_class.repflows, 0, resume=resume)
        elif shared_level == 1:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
        else:
            raise NotImplementedError

    def change_type_map(self, type_map, model_with_new_type_stat=None):
        assert self.type_map is not None
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
# SPLIT_MARKER_8

    def serialize(self) -> dict:
        from deepmd.dpmodel.utils import EnvMat as DPEnvMat

        repflows = self.repflows
        data = {
            "@class": "Descriptor",
            "type": "dpa3s_v7_matris",
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
            "num_edge_basis": self.num_edge_basis,
            "num_angle_basis": self.num_angle_basis,
            "envelope_exponent": self.envelope_exponent,
            "learnable_basis": self.learnable_basis,
            "gated_hidden_dim": self.gated_hidden_dim,
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
                "edge_basis": repflows.edge_basis.serialize(),
                "angle_basis": repflows.angle_basis.serialize(),
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
    def deserialize(cls, data: dict) -> "DescrptDPA3V7":
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
            obj.chg_embedding.embedding = TypeEmbedNetConsistent.deserialize(chg_embedding)
            obj.spin_embedding.embedding = TypeEmbedNetConsistent.deserialize(spin_embedding)
            obj.mix_cs_mlp = MLPLayer.deserialize(mix_cs_mlp)

        def t_cvt(xx):
            return torch.tensor(xx, dtype=obj.repflows.prec, device=env.DEVICE)

        statistic_repflows = repflow_variable.pop("@variables")
        env_mat = repflow_variable.pop("env_mat")
        repflow_layers = repflow_variable.pop("repflow_layers")
        obj.repflows.edge_basis = BesselExpansion.deserialize(
            repflow_variable.pop("edge_basis")
        )
        obj.repflows.angle_basis = FourierExpansion.deserialize(
            repflow_variable.pop("angle_basis")
        )
        obj.repflows.edge_embd = MLPLayer.deserialize(
            repflow_variable.pop("edge_embd")
        )
        obj.repflows.angle_embd = MLPLayer.deserialize(
            repflow_variable.pop("angle_embd")
        )
        obj.repflows["davg"] = t_cvt(statistic_repflows["davg"])
        obj.repflows["dstd"] = t_cvt(statistic_repflows["dstd"])
        obj.repflows.layers = torch.nn.ModuleList(
            [RepFlowLayerV7.deserialize(layer) for layer in repflow_layers]
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
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None,
               torch.Tensor | None, torch.Tensor | None]:
        parallel_mode = comm_dict is not None
        extended_coord = extended_coord.to(dtype=self.prec)
        nframes, nloc, nnei = nlist.shape

        if not parallel_mode:
            node_ebd_ext = self.type_embedding(extended_atype[:, :nloc])
        else:
            node_ebd_ext = self.type_embedding(extended_atype)

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

        node_ebd, edge_ebd, h2, rot_mat, sw = self.repflows(
            nlist, extended_coord, extended_atype, node_ebd_ext,
            mapping, comm_dict=comm_dict,
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
    def update_sel(cls, train_data, type_map, local_jdata):
        local_jdata_cpy = local_jdata.copy()
        update_sel = UpdateSel()
        min_nbor_dist, e_sel = update_sel.update_one_sel(
            train_data, type_map,
            local_jdata_cpy["e_rcut"], local_jdata_cpy["e_sel"], True,
        )
        local_jdata_cpy["e_sel"] = e_sel[0]
        min_nbor_dist, a_sel = update_sel.update_one_sel(
            train_data, type_map,
            local_jdata_cpy["a_rcut"], local_jdata_cpy["a_sel"], True,
        )
        local_jdata_cpy["a_sel"] = a_sel[0]
        return local_jdata_cpy, min_nbor_dist

    def enable_compression(self, min_nbor_dist, table_extrapolate=5,
                           table_stride_1=0.01, table_stride_2=0.1,
                           check_frequency=-1):
        raise NotImplementedError("Compression is unsupported for DPA3S V7.")
