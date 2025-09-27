# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
    Union,
)

import torch

from e3nn.o3 import Irreps, Linear
from e3nn.o3 import FullyConnectedTensorProduct as FCTP
import sevenn.util as util
from sevenn.nn.edge_embedding import (
    SphericalEncoding,
    # BesselBasis,
    PolynomialCutoff,
)
import time
from deepmd.pt.utils import env
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.descriptor.descriptor import (
    DescriptorBlock,
)
from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pt.model.network.mlp import (
    AnglePriorEncoder,
    AngleSH,
    MLPLayer,
)
from deepmd.pt.model.network.utils import (
    BesselBasis,
    GaussianSmearing,
    PolynomialEnvelope,
    RadialMLP,
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
)
from deepmd.utils.env_mat_stat import (
    StatItem,
)
from deepmd.utils.path import (
    DPPath,
)

from .repflow_layer import (
    RepFlowLayer,
)

if not hasattr(torch.ops.deepmd, "border_op"):

    def border_op(
        argument0,
        argument1,
        argument2,
        argument3,
        argument4,
        argument5,
        argument6,
        argument7,
        argument8,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "border_op is not available since customized PyTorch OP library is not built when freezing the model. "
            "See documentation for DPA-3 for details."
        )

    # Note: this hack cannot actually save a model that can be run using LAMMPS.
    torch.ops.deepmd.border_op = border_op


@DescriptorBlock.register("se_repflow")
class DescrptBlockRepflows(DescriptorBlock):
    def __init__(
        self,
        e_rcut,
        e_rcut_smth,
        e_sel: int,
        a_rcut,
        a_rcut_smth,
        a_sel: int,
        ntypes: int,
        nlayers: int = 6,
        n_dim: int = 128,
        e_dim: int = 64,
        a_dim: int = 64,
        a_compress_rate: int = 0,
        a_compress_e_rate: int = 1,
        a_compress_use_split: bool = False,
        n_multi_edge_message: int = 1,
        axis_neuron: int = 4,
        update_angle: bool = True,
        activation_function: str = "silu",
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        set_davg_zero: bool = True,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        precision: str = "float64",
        skip_stat: bool = True,
        smooth_angle_init: bool = False,
        angle_init_use_sin: bool = False,
        smooth_edge_update: bool = False,
        angle_multi_freq: Optional[str] = None,
        use_dynamic_sel: bool = False,
        sel_reduce_factor: float = 10.0,
        use_env_envelope: bool = False,
        use_new_sw: bool = False,
        update_dihedral: bool = False,
        d_dim: int = 32,
        d_sel: int = 10,
        d_rcut: float = 2.8,
        d_rcut_smth: float = 2.0,
        use_ffn_node_edge_message: bool = False,
        use_ffn_edge_edge_message: bool = False,
        use_ffn_edge_angle_message: bool = False,
        use_ffn_angle_angle_message: bool = False,
        ffn_hidden_dim: int = 1024,
        edge_use_concat_rbf: bool = False,
        edge_use_rbf: bool = False,
        edge_use_dist: bool = False,
        embed_use_bias: bool = True,
        edge_use_attn: bool = False,
        edge_attn_hidden: int = 32,
        edge_attn_head: int = 4,
        edge_attn_use_ln: bool = True,
        edge_rbf_dot_self: bool = False,
        edge_rbf_dot_message: bool = False,
        edge_use_esen_rbf: bool = False,
        edge_use_esen_atom_ebd: bool = False,
        edge_use_esen_env: bool = False,
        residual_pref: list = [],
        tebd_use_act: bool = True,
        message_use_self_concat: bool = False,
        use_slim_message: bool = False,
        use_combined_output: bool = False,
        use_force_embedding: bool = False,
        force_embedding_on_edge: bool = False,
        use_gated_mlp: bool = False,
        gated_mlp_norm: str = "none",
        only_angle_gated_mlp: bool = False,
        use_res_gnn: bool = False,
        res_gnn_layer: int = 6,
        node_use_rmsnorm: bool = False,
        use_loc_mapping: bool = True,
        use_rk_update: bool = False,
        rk_order: int = 4,
        rk_update_diff_layer: bool = False,
        angle_use_node: bool = True,
        optim_update: bool = True,
        angle_self_attention: bool = False,
        angle_self_attention_gate: str = "none",
        rmsnorm_mode: str = "none",
        edge_rbf_cat_message: bool = False,
        edge_message_use_dropout: bool = False,
        angle_message_use_dropout: bool = False,
        dropout_rate: float = 0.1,
        angle_use_sh_init: bool = False,
        angle_sh_init_lmax: int = 3,
        angle_use_fixed_gaussian: bool = False,
        angle_fixed_gaussian_interpolate: bool = False,
        use_e3nn_conv: bool = False,
        e3nn_conv_pattern: str = "128x0e+64x1e+32x2e+32x3e",
        use_e3nn_denominator: bool = False,
        e3nn_conv_l_max: int = 3,
        e3nn_use_edge_feat_weights: bool = False,
        use_e3nn_angle_conv: bool = False,
        e3nn_angle_conv_l_max: int = 2,
        e3nn_angle_conv_pattern: str = "64x0e+32x1e+32x2e",
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        r"""
        The repflow descriptor block.

        Parameters
        ----------
        n_dim : int, optional
            The dimension of node representation.
        e_dim : int, optional
            The dimension of edge representation.
        a_dim : int, optional
            The dimension of angle representation.
        nlayers : int, optional
            Number of repflow layers.
        e_rcut : float, optional
            The edge cut-off radius.
        e_rcut_smth : float, optional
            Where to start smoothing for edge. For example the 1/r term is smoothed from rcut to rcut_smth.
        e_sel : int, optional
            Maximally possible number of selected edge neighbors.
        a_rcut : float, optional
            The angle cut-off radius.
        a_rcut_smth : float, optional
            Where to start smoothing for angle. For example the 1/r term is smoothed from rcut to rcut_smth.
        a_sel : int, optional
            Maximally possible number of selected angle neighbors.
        a_compress_rate : int, optional
            The compression rate for angular messages. The default value is 0, indicating no compression.
            If a non-zero integer c is provided, the node and edge dimensions will be compressed
            to a_dim/c and a_dim/2c, respectively, within the angular message.
        a_compress_e_rate : int, optional
            The extra compression rate for edge in angular message compression. The default value is 1.
            When using angular message compression with a_compress_rate c and a_compress_e_rate c_e,
            the edge dimension will be compressed to (c_e * a_dim / 2c) within the angular message.
        a_compress_use_split : bool, optional
            Whether to split first sub-vectors instead of linear mapping during angular message compression.
            The default value is False.
        n_multi_edge_message : int, optional
            The head number of multiple edge messages to update node feature.
            Default is 1, indicating one head edge message.
        axis_neuron : int, optional
            The number of dimension of submatrix in the symmetrization ops.
        update_angle : bool, optional
            Where to update the angle rep. If not, only node and edge rep will be used.
        update_style : str, optional
            Style to update a representation.
            Supported options are:
            -'res_avg': Updates a rep `u` with: u = 1/\\sqrt{n+1} (u + u_1 + u_2 + ... + u_n)
            -'res_incr': Updates a rep `u` with: u = u + 1/\\sqrt{n} (u_1 + u_2 + ... + u_n)
            -'res_residual': Updates a rep `u` with: u = u + (r1*u_1 + r2*u_2 + ... + r3*u_n)
            where `r1`, `r2` ... `r3` are residual weights defined by `update_residual`
            and `update_residual_init`.
        update_residual : float, optional
            When update using residual mode, the initial std of residual vector weights.
        update_residual_init : str, optional
            When update using residual mode, the initialization mode of residual vector weights.
        ntypes : int
            Number of element types
        activation_function : str, optional
            The activation function in the embedding net.
        set_davg_zero : bool, optional
            Set the normalization average to zero.
        precision : str, optional
            The precision of the embedding net parameters.
        exclude_types : list[list[int]], optional
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
        env_protection : float, optional
            Protection parameter to prevent division by zero errors during environment matrix calculations.
            For example, when using paddings, there may be zero distances of neighbors, which may make division by zero error during environment matrix calculations without protection.
        seed : int, optional
            Random seed for parameter initialization.
        """
        super().__init__()
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.e_sel = e_sel
        self.a_rcut = float(a_rcut)
        self.a_rcut_smth = float(a_rcut_smth)
        self.a_sel = a_sel
        self.ntypes = ntypes
        self.nlayers = nlayers
        # for other common desciptor method
        sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4  # use full descriptor.
        assert len(sel) == 1
        self.sel = sel
        self.rcut = e_rcut
        self.rcut_smth = e_rcut_smth
        self.sec = self.sel
        self.split_sel = self.sel
        self.a_compress_rate = a_compress_rate
        self.a_compress_e_rate = a_compress_e_rate
        self.n_multi_edge_message = n_multi_edge_message
        self.axis_neuron = axis_neuron
        self.set_davg_zero = set_davg_zero
        self.skip_stat = skip_stat
        self.a_compress_use_split = a_compress_use_split
        self.optim_update = optim_update
        self.smooth_angle_init = smooth_angle_init
        self.angle_init_use_sin = angle_init_use_sin
        self.smooth_edge_update = smooth_edge_update
        self.use_dynamic_sel = use_dynamic_sel
        self.sel_reduce_factor = sel_reduce_factor
        self.dynamic_e_sel = self.nnei / self.sel_reduce_factor
        self.dynamic_a_sel = self.a_sel / self.sel_reduce_factor
        self.angle_multi_freq = angle_multi_freq
        self.angle_use_multi_freq = angle_multi_freq is not None
        self.angle_multi_freq_list_float = (
            [float(freq) for freq in angle_multi_freq.split(":")]
            if self.angle_use_multi_freq
            else []
        )
        if self.angle_use_multi_freq:
            self.register_buffer(
                "angle_multi_freq_list",
                torch.tensor(
                    self.angle_multi_freq_list_float,
                    dtype=torch.float,
                    device=env.DEVICE,
                ),
            )
        else:
            self.angle_multi_freq_list = None

        self.angle_use_sh_init = angle_use_sh_init
        self.angle_sh_init_lmax = angle_sh_init_lmax
        if self.angle_use_sh_init:
            self.angle_sh = AngleSH(self.angle_sh_init_lmax)
        else:
            self.angle_sh = None

        self.angle_use_fixed_gaussian = angle_use_fixed_gaussian
        self.angle_fixed_gaussian_interpolate = angle_fixed_gaussian_interpolate
        if self.angle_use_fixed_gaussian:
            self.angle_gaussian_encoder = AnglePriorEncoder(
                sigma_deg=6.0,
                learn_sigma=False,
                normalize=None,
                interpolate=self.angle_fixed_gaussian_interpolate,
            )
        else:
            self.angle_gaussian_encoder = None
        self.use_env_envelope = use_env_envelope
        self.use_new_sw = use_new_sw
        self.use_force_embedding = use_force_embedding
        self.force_embedding_on_edge = force_embedding_on_edge
        self.update_dihedral = update_dihedral
        self.d_dim = d_dim
        self.d_sel = d_sel
        self.d_rcut = d_rcut
        self.d_rcut_smth = d_rcut_smth
        self.use_ffn_node_edge_message = use_ffn_node_edge_message
        self.use_ffn_edge_edge_message = use_ffn_edge_edge_message
        self.use_ffn_edge_angle_message = use_ffn_edge_angle_message
        self.use_ffn_angle_angle_message = use_ffn_angle_angle_message
        self.ffn_hidden_dim = ffn_hidden_dim
        self.edge_use_concat_rbf = edge_use_concat_rbf
        self.edge_use_rbf = edge_use_rbf
        self.edge_use_dist = edge_use_dist
        self.embed_use_bias = embed_use_bias
        self.edge_use_attn = edge_use_attn
        self.edge_attn_hidden = edge_attn_hidden
        self.edge_attn_head = edge_attn_head
        self.edge_attn_use_ln = edge_attn_use_ln
        self.edge_rbf_dot_self = edge_rbf_dot_self
        self.edge_rbf_dot_message = edge_rbf_dot_message
        self.edge_use_esen_rbf = edge_use_esen_rbf
        self.edge_use_esen_atom_ebd = edge_use_esen_atom_ebd
        self.edge_use_esen_env = edge_use_esen_env
        self.edge_rbf_cat_message = edge_rbf_cat_message
        if self.edge_rbf_dot_self or self.edge_rbf_dot_message:
            assert self.edge_use_rbf or self.edge_use_concat_rbf, "rbf is not used"
        self.edge_embed_input_dim = 1
        if self.edge_use_esen_atom_ebd or self.edge_use_esen_env:
            assert self.edge_use_esen_rbf, "esen rbf is not used"
        if self.edge_use_esen_rbf:
            self.rbf = GaussianSmearing(
                0.0,
                self.e_rcut,
                10,
                2.0,
            )
            self.edge_embed_input_dim = 10
        elif self.edge_use_concat_rbf:
            self.rbf = BesselBasis(self.e_rcut)
            self.edge_embed_input_dim = 1 + self.rbf.num_basis
        elif self.edge_use_rbf:
            self.rbf = BesselBasis(self.e_rcut)
            self.edge_embed_input_dim = self.rbf.num_basis
        elif self.edge_rbf_cat_message:
            # edge can use dist itself
            self.rbf = BesselBasis(self.e_rcut)
        else:
            self.rbf = None

        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.update_angle = update_angle
        self.residual_pref = residual_pref
        self.tebd_use_act = tebd_use_act
        self.message_use_self_concat = message_use_self_concat
        self.use_slim_message = use_slim_message
        self.use_combined_output = use_combined_output
        self.use_loc_mapping = use_loc_mapping
        self.use_rk_update = use_rk_update
        if self.use_rk_update:
            assert (
                self.use_loc_mapping
            ), "use_loc_mapping must be True when use_rk_update is True"
        self.rk_order = rk_order
        self.rk_update_diff_layer = rk_update_diff_layer
        if self.rk_update_diff_layer:
            assert (
                self.nlayers % self.rk_order == 0
            ), "nlayers must be divisible by rk_order"
            assert self.rk_order == 4, "rk_order must be 4 for now"
        self.use_gated_mlp = use_gated_mlp
        self.gated_mlp_norm = gated_mlp_norm
        self.only_angle_gated_mlp = only_angle_gated_mlp
        self.use_res_gnn = use_res_gnn
        self.res_gnn_layer = res_gnn_layer
        if self.use_res_gnn:
            assert (
                self.nlayers % self.res_gnn_layer == 0
            ), "nlayers must be divisible by res_gnn_layer"
        assert not (
            self.message_use_self_concat and self.use_slim_message
        ), "only one of message_use_self_concat and use_slim_message can be True"

        self.node_use_rmsnorm = node_use_rmsnorm

        self.angle_use_node = angle_use_node
        if not self.angle_use_node:
            assert (
                not self.optim_update
            ), "optim_update must be False when angle_use_node is False"

        if self.edge_rbf_cat_message:
            assert (
                not self.optim_update
            ), "optim_update must be False when edge_rbf_cat_message is True"

        if self.edge_use_esen_atom_ebd:
            self.source_embedding = torch.nn.Embedding(self.ntypes, self.e_dim)
            self.target_embedding = torch.nn.Embedding(self.ntypes, self.e_dim)
            torch.nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            torch.nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_embed_input_dim += 2 * self.e_dim
        else:
            self.source_embedding = None
            self.target_embedding = None

        if self.edge_use_esen_env:
            self.env = PolynomialEnvelope(exponent=5)
        else:
            self.env = None

        self.angle_self_attention = angle_self_attention
        self.angle_self_attention_gate = angle_self_attention_gate
        if self.angle_self_attention:
            assert (
                not self.use_dynamic_sel
            ), "angle_self_attention does not support dynamic selection so far"
            assert self.angle_self_attention_gate in [
                "none",
                "edge",
                "edge_feat",
            ], "angle_self_attention_gate must be 'none', 'edge' or 'edge_feat'"
        self.rmsnorm_mode = rmsnorm_mode
        self.edge_message_use_dropout = edge_message_use_dropout
        self.angle_message_use_dropout = angle_message_use_dropout
        if self.edge_message_use_dropout or self.angle_message_use_dropout:
            assert (
                not self.optim_update
            ), "optim_update must be False when using dropout"
        self.dropout_rate = dropout_rate

        self.activation_function = activation_function
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.act = ActivationFn(activation_function)
        self.prec = PRECISION_DICT[precision]

        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)
        self.env_protection = env_protection
        self.precision = precision
        self.epsilon = 1e-4
        self.seed = seed
        self.use_e3nn_conv = use_e3nn_conv
        self.e3nn_conv_pattern = e3nn_conv_pattern
        self.use_e3nn_denominator = use_e3nn_denominator
        self.e3nn_conv_l_max = e3nn_conv_l_max
        self.e3nn_use_edge_feat_weights = e3nn_use_edge_feat_weights
        self.use_e3nn_angle_conv = use_e3nn_angle_conv
        self.e3nn_angle_conv_l_max = e3nn_angle_conv_l_max
        self.e3nn_angle_conv_pattern = e3nn_angle_conv_pattern

        if not self.edge_use_esen_rbf:
            self.edge_embd = MLPLayer(
                self.edge_embed_input_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, 0),
                bias=self.embed_use_bias,
            )
        else:
            edge_channels_list = [
                self.edge_embed_input_dim,
                self.e_dim,
                self.e_dim,
                self.e_dim,
            ]
            self.edge_embd = RadialMLP(edge_channels_list)

        if self.angle_use_sh_init:
            angle_input_dim = self.angle_sh_init_lmax + 1
        elif self.angle_use_fixed_gaussian:
            angle_input_dim = (
                10 + 1 if not self.angle_fixed_gaussian_interpolate else 12 + 1
            )
        else:
            angle_input_dim = (
                len(self.angle_multi_freq_list_float) + 1
                if not self.angle_init_use_sin
                else 2 * (len(self.angle_multi_freq_list_float) + 1)
            )

        self.angle_embd = MLPLayer(
            angle_input_dim,
            self.a_dim,
            precision=precision,
            bias=False,
            seed=child_seed(seed, 1),
        )
        if self.update_dihedral:
            self.dihedral_embd = MLPLayer(
                1, self.d_dim, precision=precision, seed=child_seed(seed, 2)
            )
        else:
            self.dihedral_embd = None

        if self.use_force_embedding:
            self.force_embedding_linear = MLPLayer(
                1,
                self.n_dim if not self.force_embedding_on_edge else self.e_dim,
                precision=precision,
                seed=child_seed(seed, 3),
            )
        else:
            self.force_embedding_linear = None

        layers = []
        # for node edge e3nn conv
        irreps_x = Irreps(f'{self.n_dim}x0e')
        self.lmax = e3nn_conv_l_max
        self.e3nn_conv_pattern = Irreps("+".join(e3nn_conv_pattern.split("+")[:self.lmax+1]))
        if self.use_e3nn_conv:
            self.edge_rbf_embed = BesselBasis(self.e_rcut)
            self.edge_env = PolynomialEnvelope(exponent=6)
            self.edge_spherical_embd = SphericalEncoding(self.lmax, parity=1, normalize=True)
            self.irreps_filter = self.edge_spherical_embd.irreps_out
        else:
            self.edge_rbf_embed = None
            self.edge_env = None
            self.edge_spherical_embd = None
            self.irreps_filter = "0e"

        # for edge angle e3nn conv
        irreps_edge = Irreps(f'{self.e_dim}x0e')
        self.angle_lmax = e3nn_angle_conv_l_max
        self.e3nn_angle_conv_pattern = Irreps("+".join(e3nn_angle_conv_pattern.split("+")[:self.angle_lmax+1]))
        if self.use_e3nn_angle_conv:
            self.edge_spherical_embd_for_angle = SphericalEncoding(self.angle_lmax, parity=1, normalize=True)
            self.irreps_angle_filter = self.edge_spherical_embd_for_angle.irreps_out
            self.angle_edge_filter_out = util.infer_irreps_out(
                self.irreps_angle_filter,  # type: ignore
                self.irreps_angle_filter,
                self.angle_lmax,  # type: ignore
                'full',
                False,
            )
            # edge_i x edge_j --> linear --> angle_filter
            self.edge_to_angle_filter_prod = FCTP(self.irreps_angle_filter, self.irreps_angle_filter, self.angle_edge_filter_out)
            self.edge_to_angle_filter_linear = Linear(
                irreps_in=self.angle_edge_filter_out,
                irreps_out=self.irreps_angle_filter,
                biases=False,
            )
        else:
            self.edge_spherical_embd_for_angle = None
            self.irreps_angle_filter = "0e"
            self.edge_to_angle_filter_prod = None
            self.edge_to_angle_filter_linear = None


        for ii in range(nlayers):
            # for node edge e3nn conv
            irreps_out = Irreps(self.e3nn_conv_pattern)
            irreps_out_tp = util.infer_irreps_out(
                irreps_x,  # type: ignore
                self.irreps_filter,
                irreps_out.lmax,  # type: ignore
                'full',
                False,
            )
            e3nn_conv_args = {
                "irreps_x": irreps_x,
                "irreps_filter": self.irreps_filter,
                "irreps_out_tp": irreps_out_tp,
                "irreps_out": irreps_out,
                "denominator": 1.0 if not self.use_e3nn_denominator else self.dynamic_e_sel / 4,
                "train_denominator": True,
                "weight_layer_input_to_hidden": [8, 64, 64] if not self.e3nn_use_edge_feat_weights else [self.e_dim],
            }
            irreps_x = irreps_out

            # for edge angle e3nn conv
            irreps_edge_out = Irreps(self.e3nn_angle_conv_pattern)
            irreps_out_tp = util.infer_irreps_out(
                irreps_edge,  # type: ignore
                self.irreps_angle_filter,
                irreps_edge_out.lmax,  # type: ignore
                'full',
                False,
            )
            e3nn_angle_conv_args = {
                "irreps_x": irreps_edge,
                "irreps_filter": self.irreps_angle_filter,
                "irreps_out_tp": irreps_out_tp,
                "irreps_out": irreps_edge_out,
                "denominator": 1.0 if not self.use_e3nn_denominator else self.dynamic_a_sel / 4,
                "train_denominator": True,
                "weight_layer_input_to_hidden": [self.a_dim],
            }
            irreps_edge = irreps_edge_out

            layers.append(
                RepFlowLayer(
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
                    a_compress_rate=self.a_compress_rate,
                    a_compress_use_split=self.a_compress_use_split,
                    a_compress_e_rate=self.a_compress_e_rate,
                    n_multi_edge_message=self.n_multi_edge_message,
                    axis_neuron=self.axis_neuron,
                    update_angle=self.update_angle,
                    activation_function=self.activation_function,
                    update_style=self.update_style,
                    update_residual=self.update_residual,
                    update_residual_init=self.update_residual_init,
                    precision=precision,
                    optim_update=self.optim_update,
                    use_dynamic_sel=self.use_dynamic_sel,
                    sel_reduce_factor=self.sel_reduce_factor,
                    smooth_edge_update=self.smooth_edge_update,
                    update_dihedral=self.update_dihedral,
                    d_dim=self.d_dim,
                    d_sel=self.d_sel,
                    d_rcut=self.d_rcut,
                    d_rcut_smth=self.d_rcut_smth,
                    use_ffn_node_edge_message=self.use_ffn_node_edge_message,
                    use_ffn_edge_edge_message=self.use_ffn_edge_edge_message,
                    use_ffn_edge_angle_message=self.use_ffn_edge_angle_message,
                    use_ffn_angle_angle_message=self.use_ffn_angle_angle_message,
                    ffn_hidden_dim=self.ffn_hidden_dim,
                    edge_use_attn=self.edge_use_attn,
                    edge_attn_hidden=self.edge_attn_hidden,
                    edge_attn_head=self.edge_attn_head,
                    edge_attn_use_ln=self.edge_attn_use_ln,
                    edge_rbf_dot_self=self.edge_rbf_dot_self,
                    edge_rbf_dot_message=self.edge_rbf_dot_message,
                    rbf_dim=self.edge_embed_input_dim
                    if not self.edge_rbf_cat_message
                    else self.rbf.num_basis,
                    residual_pref=self.residual_pref,
                    message_use_self_concat=self.message_use_self_concat,
                    use_slim_message=self.use_slim_message,
                    use_gated_mlp=self.use_gated_mlp,
                    gated_mlp_norm=self.gated_mlp_norm,
                    only_angle_gated_mlp=self.only_angle_gated_mlp,
                    node_use_rmsnorm=self.node_use_rmsnorm,
                    angle_use_node=self.angle_use_node,
                    angle_self_attention=self.angle_self_attention,
                    angle_self_attention_gate=self.angle_self_attention_gate,
                    rmsnorm_mode=self.rmsnorm_mode,
                    edge_rbf_cat_message=self.edge_rbf_cat_message,
                    edge_message_use_dropout=self.edge_message_use_dropout,
                    angle_message_use_dropout=self.angle_message_use_dropout,
                    dropout_rate=self.dropout_rate,
                    use_e3nn_conv=self.use_e3nn_conv,
                    e3nn_conv_pattern=self.e3nn_conv_pattern,
                    e3nn_use_edge_feat_weights=self.e3nn_use_edge_feat_weights,
                    e3nn_conv_args=e3nn_conv_args,
                    use_e3nn_angle_conv=self.use_e3nn_angle_conv,
                    e3nn_angle_conv_args=e3nn_angle_conv_args,
                    e3nn_angle_conv_pattern=self.e3nn_angle_conv_pattern,
                    seed=child_seed(child_seed(seed, 1), ii),
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.additional_output_for_fitting: dict[str, Optional[torch.Tensor]] = {}

        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(wanted_shape, dtype=self.prec, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=self.prec, device=env.DEVICE)
        if self.skip_stat:
            stddev = stddev * 0.3
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.stats = None

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.e_rcut

    additional_output_for_fitting: dict[str, Optional[torch.Tensor]]

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.e_rcut_smth

    def get_norm_fact(self) -> list[float]:
        """Returns the norm factor."""
        return [
            float(self.dynamic_e_sel if self.use_dynamic_sel else self.nnei),
            float(self.dynamic_a_sel if self.use_dynamic_sel else self.a_sel),
        ]

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

    def __setitem__(self, key, value) -> None:
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
        """If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return True

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.env_protection

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        out_dim = self.n_dim
        if self.use_combined_output:
            out_dim += (
                self.e_dim if not self.update_angle else self.e_dim + self.a_dim
            )  # edge or edge + angle
        return out_dim

    @property
    def dim_in(self):
        """Returns the atomic input dimension of this descriptor."""
        return self.n_dim

    @property
    def dim_emb(self):
        """Returns the embedding dimension e_dim."""
        return self.get_dim_emb()

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def get_additional_output_for_fitting(self):
        return self.additional_output_for_fitting

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: Optional[torch.Tensor] = None,
        mapping: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
        force_embedding_input: Optional[torch.Tensor] = None,
    ):
        parrallel_mode = comm_dict is not None
        if not parrallel_mode:
            assert mapping is not None
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.view(nframes, -1).shape[1] // 3
        atype = extended_atype[:, :nloc]
        # nb x nloc x nnei
        exclude_mask = self.emask(nlist, extended_atype)
        nlist = torch.where(exclude_mask != 0, nlist, -1)
        # nb x nloc x nnei x 4, nb x nloc x nnei x 3, nb x nloc x nnei x 1
        dmatrix, diff, sw = prod_env_mat(
            extended_coord,
            nlist,
            atype,
            self.mean,
            self.stddev,
            self.e_rcut,
            self.e_rcut_smth,
            protection=self.env_protection,
            use_env_envelope=self.use_env_envelope,
            use_new_sw=self.use_new_sw,
        )
        nlist_mask = nlist != -1
        if (
            self.edge_use_esen_rbf
            or self.edge_use_concat_rbf
            or self.edge_use_rbf
            or self.edge_use_dist
            or self.use_e3nn_conv
        ):
            # nb x nloc x nnei x 1
            edge_dist = torch.linalg.norm(diff, dim=-1, keepdim=True)
        else:
            edge_dist = None

        if self.edge_use_esen_env:
            assert self.env is not None
            assert edge_dist is not None
            sw = self.env(edge_dist / self.e_rcut)

        sw = torch.squeeze(sw, -1)
        # beyond the cutoff sw should be 0.0
        sw = sw.masked_fill(~nlist_mask, 0.0)

        # get angle nlist (maybe smaller)
        length_nei = torch.linalg.norm(diff, dim=-1)
        a_dist_mask = (length_nei < self.a_rcut)[:, :, : self.a_sel]
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
            use_env_envelope=self.use_env_envelope,
            use_new_sw=self.use_new_sw,
        )
        a_nlist_mask = a_nlist != -1
        if self.edge_use_esen_env:
            assert self.env is not None
            edge_dist_a = torch.linalg.norm(a_diff, dim=-1, keepdim=True)
            a_sw = self.env(edge_dist_a / self.a_rcut)
        a_sw = torch.squeeze(a_sw, -1)
        # beyond the cutoff sw should be 0.0
        a_sw = a_sw.masked_fill(~a_nlist_mask, 0.0)

        # get dihedral nlist (maybe smaller)
        d_dist_mask = (length_nei < self.d_rcut)[:, :, : self.d_sel]
        d_nlist = nlist[:, :, : self.d_sel]
        d_nlist = torch.where(d_dist_mask, d_nlist, -1)
        d_nlist_mask = d_nlist != -1
        # set all padding positions to index of 0
        # if the a neighbor is real or not is indicated by nlist_mask
        nlist[nlist == -1] = 0
        a_nlist[a_nlist == -1] = 0

        # get node embedding
        # [nframes, nloc, tebd_dim]
        assert extended_atype_embd is not None
        atype_embd = extended_atype_embd[:, :nloc, :]
        assert list(atype_embd.shape) == [nframes, nloc, self.n_dim]
        assert isinstance(atype_embd, torch.Tensor)  # for jit
        if not self.tebd_use_act:
            node_ebd = atype_embd
        else:
            node_ebd = self.act(atype_embd)
        n_dim = node_ebd.shape[-1]

        # get edge and angle embedding input
        # nb x nloc x nnei x 1,  nb x nloc x nnei x 3
        edge_input, h2 = torch.split(dmatrix, [1, 3], dim=-1)
        if (
            self.edge_use_esen_rbf
            or self.edge_use_concat_rbf
            or self.edge_use_rbf
            or self.edge_use_dist
        ):
            assert edge_dist is not None
            # nb x nloc x nnei x 1
            edge_input = edge_dist
        # nf x nloc x a_nnei x 3
        normalized_diff_i = a_diff / (
            torch.linalg.norm(a_diff, dim=-1, keepdim=True) + 1e-6
        )
        # nf x nloc x 3 x a_nnei
        normalized_diff_j = torch.transpose(normalized_diff_i, 2, 3)
        # nf x nloc x a_nnei x a_nnei
        # 1 - 1e-6 for torch.acos stability
        cosine_ij = torch.matmul(normalized_diff_i, normalized_diff_j) * (1 - 1e-6)
        sine_ij = torch.sqrt(1 - cosine_ij**2)
        if self.smooth_angle_init:
            cosine_ij = cosine_ij * a_sw.unsqueeze(-1) * a_sw.unsqueeze(-2)
            sine_ij = sine_ij * a_sw.unsqueeze(-1) * a_sw.unsqueeze(-2)

        if self.angle_use_multi_freq:
            assert self.angle_multi_freq_list is not None
            theta = torch.acos(cosine_ij)
            theta_list = theta[..., None] * self.angle_multi_freq_list
        else:
            theta_list = None

        # nf x nloc x a_nnei x a_nnei x 1,  nf x nloc x a_nnei x a_nnei x n_freq
        angle_input_list = [cosine_ij.unsqueeze(-1)] + (
            [torch.cos(theta_list)] if theta_list is not None else []
        )
        if self.angle_init_use_sin:
            angle_input_list += [sine_ij.unsqueeze(-1)] + (
                [torch.sin(theta_list)] if theta_list is not None else []
            )
        angle_input = torch.cat(angle_input_list, dim=-1) / (torch.pi**0.5)

        if self.update_dihedral:
            _, d_diff, d_sw = prod_env_mat(
                extended_coord,
                d_nlist,
                atype,
                self.mean[:, : self.d_sel],
                self.stddev[:, : self.d_sel],
                self.d_rcut,
                self.d_rcut_smth,
                protection=self.env_protection,
                use_env_envelope=self.use_env_envelope,
                use_new_sw=self.use_new_sw,
            )
            d_sw = torch.squeeze(d_sw, -1)
            # beyond the cutoff sw should be 0.0
            d_sw = d_sw.masked_fill(~d_nlist_mask, 0.0)
            d_nlist[d_nlist == -1] = 0

            # compute dihedral
            # nf x nloc x d_nnei x 3
            normalized_d = d_diff / (
                torch.linalg.norm(d_diff, dim=-1, keepdim=True) + 1e-6
            )
            # nf x nloc x d_nnei x [d_nnei] x 3
            normalized_d_ij = normalized_d[:, :, :, None, :].expand(
                [-1, -1, -1, self.d_sel, -1]
            )
            # nf x nloc x [d_nnei] x d_nnei x 3
            normalized_d_ik = normalized_d[:, :, None, :, :].expand(
                [-1, -1, self.d_sel, -1, -1]
            )
            # nf x nloc x d_nnei x d_nnei x 3
            norm_ij_ik = torch.cross(normalized_d_ij, normalized_d_ik, dim=-1)
            norm_ij_ik = norm_ij_ik / (
                torch.linalg.norm(norm_ij_ik, dim=-1, keepdim=True) + 1e-6
            )
            # nf x nloc x d_nnei x d_nnei x 3
            norm_il_ij = -norm_ij_ik

            # nf x nloc x d_nnei x d_nnei x d_nnei
            cos_ijkl = -torch.matmul(norm_ij_ik, norm_il_ij.transpose(-1, -2)) * (
                1 - 1e-6
            )
            dihedral_input = cos_ijkl.unsqueeze(-1) / (torch.pi**0.5)
        else:
            d_sw = None
            dihedral_input = None

        if self.edge_use_esen_atom_ebd:
            # nf x (nl x nnei)
            nlist_index = nlist.reshape(nframes, nloc * nnei)
            # nf x (nl x nnei)
            source_type = torch.gather(
                extended_atype, dim=1, index=nlist_index
            ).reshape(nframes, nloc, nnei)
            target_type = atype.unsqueeze(-1).expand(-1, -1, nnei)
        else:
            source_type = None
            target_type = None

        if not parrallel_mode and self.use_loc_mapping:
            assert mapping is not None
            # convert nlist from nall to nloc index
            nlist = torch.gather(
                mapping,
                1,
                index=nlist.reshape(nframes, -1),
            ).reshape(nlist.shape)

        if self.use_dynamic_sel:
            # get graph index
            edge_index, angle_index, dihedral_index, a_nlist_mask_3d, d_nlist_mask4d = (
                get_graph_index(
                    nlist,
                    nlist_mask,
                    a_nlist_mask,
                    d_nlist_mask,
                    nall,
                    calculate_dihedral=self.update_dihedral,
                    use_loc_mapping=self.use_loc_mapping,
                )
            )
            # flat all the tensors
            # n_edge x 1
            edge_input = edge_input[nlist_mask]
            if edge_dist is not None:
                edge_dist = edge_dist[nlist_mask]
            # n_edge x 3
            h2 = h2[nlist_mask]
            # n_edge
            sw = sw[nlist_mask]
            # n_edge x 4
            dmatrix = dmatrix[nlist_mask]
            # n_edge x 3
            diff = diff[nlist_mask]

            if self.edge_use_esen_atom_ebd:
                assert source_type is not None
                assert target_type is not None
                source_type = source_type[nlist_mask]
                target_type = target_type[nlist_mask]

            # nb x nloc x a_nnei x a_nnei
            a_nlist_mask = a_nlist_mask_3d
            # n_angle x 1
            angle_input = angle_input[a_nlist_mask]
            # n_angle
            a_sw = (a_sw[:, :, :, None] * a_sw[:, :, None, :])[a_nlist_mask]
            if self.update_dihedral:
                assert dihedral_input is not None
                assert d_sw is not None
                assert d_nlist_mask4d is not None
                # nb x nloc x d_nnei x d_nnei x d_nnei
                d_nlist_mask = d_nlist_mask4d
                # n_dihedral x 1
                dihedral_input = dihedral_input[d_nlist_mask]
                # n_dihedral x 1
                d_sw = (
                    d_sw[:, :, :, None, None]
                    * d_sw[:, :, None, :, None]
                    * d_sw[:, :, None, None, :]
                )[d_nlist_mask]
            self.additional_output_for_fitting["edge_index"] = edge_index
            self.additional_output_for_fitting["angle_index"] = angle_index
        else:
            # avoid jit assertion
            edge_index = angle_index = torch.zeros(
                [1, 3], device=nlist.device, dtype=nlist.dtype
            )
            dihedral_index = None
            self.additional_output_for_fitting["edge_index"] = None
            self.additional_output_for_fitting["angle_index"] = None
        self.additional_output_for_fitting["diff"] = diff
        self.additional_output_for_fitting["sw"] = sw
        self.additional_output_for_fitting["a_sw"] = a_sw

        # get edge and angle embedding
        # nb x nloc x nnei x e_dim [OR] n_edge x e_dim
        if self.edge_use_esen_rbf:
            assert self.rbf is not None
            rbf_ebd = self.rbf(edge_input)
            if self.edge_use_esen_atom_ebd:
                assert source_type is not None
                assert target_type is not None
                assert self.source_embedding is not None
                assert self.target_embedding is not None
                source_ebd = self.source_embedding(source_type)
                target_ebd = self.target_embedding(target_type)
                rbf_input = torch.cat((rbf_ebd, source_ebd, target_ebd), dim=-1)
            else:
                rbf_input = rbf_ebd
            edge_ebd = self.edge_embd(rbf_input)
        elif self.edge_use_dist:
            edge_ebd = self.edge_embd(edge_input)
            if not self.edge_rbf_cat_message:
                rbf_ebd = None
            else:
                assert self.rbf is not None
                rbf_ebd = self.rbf(edge_input)
        elif self.edge_use_concat_rbf:
            assert self.rbf is not None
            rbf_ebd = torch.cat([dmatrix[..., :1], self.rbf(edge_input)], dim=-1)
            edge_ebd = self.edge_embd(rbf_ebd)
        elif self.edge_use_rbf:
            assert self.rbf is not None
            rbf_ebd = self.rbf(edge_input)
            edge_ebd = self.edge_embd(rbf_ebd)
        else:
            rbf_ebd = None
            edge_ebd = self.act(self.edge_embd(edge_input))

        if self.angle_use_sh_init:
            assert self.angle_sh is not None
            # nf x nloc x a_nnei x a_nnei x sh_sim [OR] n_angle x sh_sim
            angle_input = self.angle_sh(angle_input * (torch.pi**0.5))
        elif self.angle_use_fixed_gaussian:
            assert not self.angle_init_use_sin and not self.angle_use_multi_freq
            assert self.angle_gaussian_encoder is not None
            # nf x nloc x a_nnei x a_nnei x 11(13) [OR] n_angle x 11(13)
            angle_input = self.angle_gaussian_encoder(angle_input)

        # nf x nloc x a_nnei x a_nnei x a_dim [OR] n_angle x a_dim
        angle_ebd = self.angle_embd(angle_input)

        if self.update_dihedral:
            assert self.dihedral_embd is not None
            assert dihedral_input is not None
            # n_dihedral x d_dim
            dihedral_ebd = self.dihedral_embd(dihedral_input)
        else:
            dihedral_ebd = None

        # add force embedding to node or edge for DeNS
        if self.use_force_embedding:
            assert self.force_embedding_linear is not None
            if force_embedding_input is None:
                force_input = torch.zeros(
                    (nframes, nloc, 3),
                    dtype=node_ebd.dtype,
                    device=node_ebd.device,
                )
            else:
                force_input = force_embedding_input
            if not self.use_dynamic_sel:
                # nb x nloc x nnei x 3
                force_input = force_input.unsqueeze(-2).expand(-1, -1, self.nnei, -1)
                # nb x nloc x nnei x 1
                edge_force_dot = (force_input * diff).sum(-1, keepdim=True) / (
                    torch.linalg.norm(diff, dim=-1, keepdim=True) + 1e-6
                )
                # nb x nloc x nnei x n_dim/e_dim
                edge_force_embedding = self.act(
                    self.force_embedding_linear(edge_force_dot)
                )
                if not self.force_embedding_on_edge:
                    # nb x nloc x n_dim
                    edge_force_embedding = (
                        edge_force_embedding * sw.unsqueeze(-1)
                    ).sum(-2) / self.nnei

            else:
                n2e_index = edge_index[:, 0]
                # nedge x 3
                force_input = torch.index_select(
                    force_input.reshape(-1, 3), 0, n2e_index
                )
                # nedge x 1
                edge_force_dot = (force_input * diff).sum(-1, keepdim=True) / (
                    torch.linalg.norm(diff, dim=-1, keepdim=True) + 1e-6
                )
                # nedge x n_dim/e_dim
                edge_force_embedding = self.act(
                    self.force_embedding_linear(edge_force_dot)
                )
                if not self.force_embedding_on_edge:
                    # nb x nloc x n_dim
                    edge_force_embedding = (
                        aggregate(
                            edge_force_embedding * sw.unsqueeze(-1),
                            n2e_index,
                            average=False,
                            num_owner=nframes * nloc,
                        ).reshape(nframes, nloc, -1)
                        / self.dynamic_e_sel
                    )
            if not self.force_embedding_on_edge:
                node_ebd = node_ebd + edge_force_embedding
            else:
                edge_ebd = edge_ebd + edge_force_embedding

        # nb x nall x n_dim
        if not parrallel_mode:
            assert mapping is not None
            mapping = (
                mapping.view(nframes, nall).unsqueeze(-1).expand(-1, -1, self.n_dim)
            )
        res_node_list = []

        # rk_update_diff_layer
        node_ebd_k_in_ori_diff = node_ebd
        node_ebd_k_in_diff = node_ebd
        edge_ebd_k_in_ori_diff = edge_ebd
        edge_ebd_k_in_diff = edge_ebd
        angle_ebd_k_in_ori_diff = angle_ebd
        angle_ebd_k_in_diff = angle_ebd

        if self.use_e3nn_conv:
            assert self.use_dynamic_sel, "e3nn conv must use dynamic sel"
            assert self.edge_rbf_embed is not None
            assert self.edge_env is not None
            assert self.edge_spherical_embd is not None
            assert edge_dist is not None
            # n_edge x rbf
            edge_rbf_ebd = self.edge_rbf_embed(edge_dist) * self.edge_env(edge_dist/self.e_rcut)
            # n_edge x num_sph(16)
            edge_sph = self.edge_spherical_embd(diff)
            node_sph_embed = node_ebd
        else:
            edge_rbf_ebd = None
            edge_sph = None
            node_sph_embed = None

        if self.use_e3nn_angle_conv:
            assert self.use_dynamic_sel, "e3nn conv must use dynamic sel"
            assert self.edge_spherical_embd_for_angle is not None
            assert self.edge_to_angle_filter_prod is not None
            assert self.edge_to_angle_filter_linear is not None
            edge_sph_for_angle = self.edge_spherical_embd_for_angle(diff)
            edge_i_index = angle_index[:, 1]
            edge_j_index = angle_index[:, 2]
            edge_angle_filter_tp = self.edge_to_angle_filter_prod(edge_sph_for_angle[edge_i_index], edge_sph_for_angle[edge_j_index])
            edge_angle_filter = self.edge_to_angle_filter_linear(edge_angle_filter_tp)
            edge_sph_embed = edge_ebd
        else:
            edge_angle_filter = None
            edge_sph_embed = None

        for idx, ll in enumerate(self.layers):
            # node_ebd:     nb x nloc x n_dim
            # node_ebd_ext: nb x nall x n_dim [OR] nb x nloc x n_dim when not parrallel_mode
            if not parrallel_mode:
                assert mapping is not None
                node_ebd_ext = (
                    torch.gather(node_ebd, 1, mapping)
                    if not self.use_loc_mapping
                    else node_ebd
                )
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
                    # for spin
                    real_nloc = nloc // 2
                    real_nall = nall // 2
                    real_n_padding = real_nall - real_nloc
                    node_ebd_real, node_ebd_virtual = torch.split(
                        node_ebd, [real_nloc, real_nloc], dim=1
                    )
                    # mix_node_ebd: nb x real_nloc x (n_dim * 2)
                    mix_node_ebd = torch.cat([node_ebd_real, node_ebd_virtual], dim=2)
                    # nb x real_nall x (n_dim * 2)
                    node_ebd = torch.nn.functional.pad(
                        mix_node_ebd.squeeze(0), (0, 0, 0, real_n_padding), value=0.0
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
                        device=env.DEVICE,
                    ),  # should be int of c++
                    torch.tensor(
                        real_nall - real_nloc,
                        dtype=torch.int32,
                        device=env.DEVICE,
                    ),  # should be int of c++
                )
                node_ebd_ext = ret[0].unsqueeze(0)
                if has_spin:
                    node_ebd_real_ext, node_ebd_virtual_ext = torch.split(
                        node_ebd_ext, [n_dim, n_dim], dim=2
                    )
                    node_ebd_ext = concat_switch_virtual(
                        node_ebd_real_ext, node_ebd_virtual_ext, real_nloc
                    )

            if self.update_dihedral:
                # for jit
                assert not self.use_rk_update
                assert dihedral_ebd is not None
                node_ebd, edge_ebd, angle_ebd, dihedral_ebd, ___, ___, = ll.forward(
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
                    d_nlist=d_nlist,
                    d_nlist_mask=d_nlist_mask,
                    edge_index=edge_index,
                    angle_index=angle_index,
                    dihedral_index=dihedral_index,
                    dihedral_ebd=dihedral_ebd,
                    d_sw=d_sw,
                    rbf_ebd=rbf_ebd,
                )
            else:
                assert dihedral_ebd is None
                if not self.use_rk_update:
                    node_ebd, edge_ebd, angle_ebd, ___, node_sph_embed, edge_sph_embed = ll.forward(
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
                        d_nlist=d_nlist,
                        d_nlist_mask=d_nlist_mask,
                        edge_index=edge_index,
                        angle_index=angle_index,
                        dihedral_index=dihedral_index,
                        dihedral_ebd=None,
                        d_sw=d_sw,
                        rbf_ebd=rbf_ebd,
                        edge_rbf_ebd=edge_rbf_ebd,
                        edge_sph=edge_sph,
                        node_sph_embed=node_sph_embed,
                        edge_angle_filter=edge_angle_filter,
                        edge_sph_embed=edge_sph_embed,
                    )
                # may cause jit slow, todo fix
                elif not self.rk_update_diff_layer:
                    node_ebd_k_in_ori = node_ebd_ext
                    node_ebd_k_in = node_ebd_ext
                    edge_ebd_k_in_ori = edge_ebd
                    edge_ebd_k_in = edge_ebd
                    angle_ebd_k_in_ori = angle_ebd
                    angle_ebd_k_in = angle_ebd

                    if self.rk_order == 4:
                        # use 4th order Runge-Kutta update

                        for kk in range(4):
                            (
                                node_ebd_k1,
                                edge_ebd_k1,
                                angle_ebd_k1,
                                ___,
                                ___,
                                ___,
                            ) = ll.forward(
                                node_ebd_k_in,
                                edge_ebd_k_in,
                                h2,
                                angle_ebd_k_in,
                                nlist,
                                nlist_mask,
                                sw,
                                a_nlist,
                                a_nlist_mask,
                                a_sw,
                                d_nlist=d_nlist,
                                d_nlist_mask=d_nlist_mask,
                                edge_index=edge_index,
                                angle_index=angle_index,
                                dihedral_index=dihedral_index,
                                dihedral_ebd=None,
                                d_sw=d_sw,
                                rbf_ebd=rbf_ebd,
                            )
                            # h * f(y), k1/2/3/4
                            node_ebd_k1 = node_ebd_k1 - node_ebd_k_in
                            edge_ebd_k1 = edge_ebd_k1 - edge_ebd_k_in
                            angle_ebd_k1 = angle_ebd_k1 - angle_ebd_k_in
                            if kk == 0:
                                # k1
                                node_ebd = node_ebd + node_ebd_k1 / 6.0
                                edge_ebd = edge_ebd + edge_ebd_k1 / 6.0
                                angle_ebd = angle_ebd + angle_ebd_k1 / 6.0
                                # next input: y + k1/2
                                node_ebd_k_in = node_ebd_k_in_ori + node_ebd_k1 / 2.0
                                edge_ebd_k_in = edge_ebd_k_in_ori + edge_ebd_k1 / 2.0
                                angle_ebd_k_in = angle_ebd_k_in_ori + angle_ebd_k1 / 2.0
                            elif kk == 1:
                                # k2
                                node_ebd = node_ebd + node_ebd_k1 / 3.0
                                edge_ebd = edge_ebd + edge_ebd_k1 / 3.0
                                angle_ebd = angle_ebd + angle_ebd_k1 / 3.0
                                # next input: y + k2/2
                                node_ebd_k_in = node_ebd_k_in_ori + node_ebd_k1 / 2.0
                                edge_ebd_k_in = edge_ebd_k_in_ori + edge_ebd_k1 / 2.0
                                angle_ebd_k_in = angle_ebd_k_in_ori + angle_ebd_k1 / 2.0
                            elif kk == 2:
                                # k3
                                node_ebd = node_ebd + node_ebd_k1 / 3.0
                                edge_ebd = edge_ebd + edge_ebd_k1 / 3.0
                                angle_ebd = angle_ebd + angle_ebd_k1 / 3.0
                                # next input: y + k3
                                node_ebd_k_in = node_ebd_k_in_ori + node_ebd_k1
                                edge_ebd_k_in = edge_ebd_k_in_ori + edge_ebd_k1
                                angle_ebd_k_in = angle_ebd_k_in_ori + angle_ebd_k1
                            else:
                                # k4
                                node_ebd = node_ebd + node_ebd_k1 / 6.0
                                edge_ebd = edge_ebd + edge_ebd_k1 / 6.0
                                angle_ebd = angle_ebd + angle_ebd_k1 / 6.0
                    elif self.rk_order == 2:
                        # use 2nd order Runge-Kutta update

                        for kk in range(2):
                            (
                                node_ebd_k1,
                                edge_ebd_k1,
                                angle_ebd_k1,
                                ___,
                            ) = ll.forward(
                                node_ebd_k_in,
                                edge_ebd_k_in,
                                h2,
                                angle_ebd_k_in,
                                nlist,
                                nlist_mask,
                                sw,
                                a_nlist,
                                a_nlist_mask,
                                a_sw,
                                d_nlist=d_nlist,
                                d_nlist_mask=d_nlist_mask,
                                edge_index=edge_index,
                                angle_index=angle_index,
                                dihedral_index=dihedral_index,
                                dihedral_ebd=None,
                                d_sw=d_sw,
                                rbf_ebd=rbf_ebd,
                            )
                            # h * f(y), k1/2
                            node_ebd_k1 = node_ebd_k1 - node_ebd_k_in
                            edge_ebd_k1 = edge_ebd_k1 - edge_ebd_k_in
                            angle_ebd_k1 = angle_ebd_k1 - angle_ebd_k_in
                            if kk == 0:
                                # next input: y + k1/2
                                node_ebd_k_in = node_ebd_k_in_ori + node_ebd_k1 / 2.0
                                edge_ebd_k_in = edge_ebd_k_in_ori + edge_ebd_k1 / 2.0
                                angle_ebd_k_in = angle_ebd_k_in_ori + angle_ebd_k1 / 2.0
                            else:
                                # k2
                                node_ebd = node_ebd + node_ebd_k1
                                edge_ebd = edge_ebd + edge_ebd_k1
                                angle_ebd = angle_ebd + angle_ebd_k1
                    else:
                        raise ValueError(
                            f"Unsupported Runge-Kutta order: {self.rk_order}"
                        )
                else:
                    # use RK update with diff layer
                    (
                        node_ebd_k1,
                        edge_ebd_k1,
                        angle_ebd_k1,
                        ___,
                    ) = ll.forward(
                        node_ebd_k_in_diff,
                        edge_ebd_k_in_diff,
                        h2,
                        angle_ebd_k_in_diff,
                        nlist,
                        nlist_mask,
                        sw,
                        a_nlist,
                        a_nlist_mask,
                        a_sw,
                        d_nlist=d_nlist,
                        d_nlist_mask=d_nlist_mask,
                        edge_index=edge_index,
                        angle_index=angle_index,
                        dihedral_index=dihedral_index,
                        dihedral_ebd=None,
                        d_sw=d_sw,
                        rbf_ebd=rbf_ebd,
                    )
                    # h * f(y), k1/2/3/4
                    node_ebd_k1 = node_ebd_k1 - node_ebd_k_in_diff
                    edge_ebd_k1 = edge_ebd_k1 - edge_ebd_k_in_diff
                    angle_ebd_k1 = angle_ebd_k1 - angle_ebd_k_in_diff
                    if idx % self.rk_order == 0:
                        # k1
                        node_ebd = node_ebd + node_ebd_k1 / 6.0
                        edge_ebd = edge_ebd + edge_ebd_k1 / 6.0
                        angle_ebd = angle_ebd + angle_ebd_k1 / 6.0
                        # next input: y + k1/2
                        node_ebd_k_in_diff = node_ebd_k_in_ori_diff + node_ebd_k1 / 2.0
                        edge_ebd_k_in_diff = edge_ebd_k_in_ori_diff + edge_ebd_k1 / 2.0
                        angle_ebd_k_in_diff = (
                            angle_ebd_k_in_ori_diff + angle_ebd_k1 / 2.0
                        )
                    elif idx % self.rk_order == 1:
                        # k2
                        node_ebd = node_ebd + node_ebd_k1 / 3.0
                        edge_ebd = edge_ebd + edge_ebd_k1 / 3.0
                        angle_ebd = angle_ebd + angle_ebd_k1 / 3.0
                        # next input: y + k2/2
                        node_ebd_k_in_diff = node_ebd_k_in_ori_diff + node_ebd_k1 / 2.0
                        edge_ebd_k_in_diff = edge_ebd_k_in_ori_diff + edge_ebd_k1 / 2.0
                        angle_ebd_k_in_diff = (
                            angle_ebd_k_in_ori_diff + angle_ebd_k1 / 2.0
                        )
                    elif idx % self.rk_order == 2:
                        # k3
                        node_ebd = node_ebd + node_ebd_k1 / 3.0
                        edge_ebd = edge_ebd + edge_ebd_k1 / 3.0
                        angle_ebd = angle_ebd + angle_ebd_k1 / 3.0
                        # next input: y + k3
                        node_ebd_k_in_diff = node_ebd_k_in_ori_diff + node_ebd_k1
                        edge_ebd_k_in_diff = edge_ebd_k_in_ori_diff + edge_ebd_k1
                        angle_ebd_k_in_diff = angle_ebd_k_in_ori_diff + angle_ebd_k1
                    else:
                        # k4
                        node_ebd = node_ebd + node_ebd_k1 / 6.0
                        edge_ebd = edge_ebd + edge_ebd_k1 / 6.0
                        angle_ebd = angle_ebd + angle_ebd_k1 / 6.0

                        # next round of rk
                        node_ebd_k_in_ori_diff = node_ebd
                        node_ebd_k_in_diff = node_ebd
                        edge_ebd_k_in_ori_diff = edge_ebd
                        edge_ebd_k_in_diff = edge_ebd
                        angle_ebd_k_in_ori_diff = angle_ebd
                        angle_ebd_k_in_diff = angle_ebd

            if self.use_res_gnn and (idx + 1) % self.res_gnn_layer == 0:
                res_node_list.append(node_ebd.unsqueeze(-1))

        if self.use_res_gnn:
            node_ebd = torch.concat(res_node_list, dim=-1).mean(dim=-1)

        if self.use_combined_output:
            concat_list = [node_ebd]
            edge_part = edge_ebd * sw.unsqueeze(-1)
            edge_part = (
                (torch.sum(edge_part, dim=-2) / self.nnei)
                if not self.use_dynamic_sel
                else (
                    aggregate(
                        edge_part,
                        edge_index[:, 0],
                        average=False,
                        num_owner=nframes * nloc,
                    ).reshape(nframes, nloc, -1)
                    / self.dynamic_e_sel
                )
            )
            concat_list.append(edge_part)
            if self.update_angle:
                if not self.use_dynamic_sel:
                    angle_part = (
                        angle_ebd
                        * a_sw[:, :, :, None, None]
                        * a_sw[:, :, None, :, None]
                    )
                    angle_part = (
                        torch.sum(torch.sum(angle_part, dim=-2), dim=-2) / self.a_sel
                    )  # (self.a_sel**0.5)**2
                else:
                    angle_part = angle_ebd * a_sw.unsqueeze(-1)
                    angle_part = (
                        aggregate(
                            angle_part,
                            angle_index[:, 0],
                            average=False,
                            num_owner=nframes * nloc,
                        ).reshape(nframes, nloc, -1)
                        / self.dynamic_a_sel
                    )
                concat_list.append(angle_part)
            node_ebd = torch.concat(concat_list, dim=-1)

        # nb x nloc x 3 x e_dim
        h2g2 = (
            RepFlowLayer._cal_hg(edge_ebd, h2, nlist_mask, sw)
            if not self.use_dynamic_sel
            else RepFlowLayer._cal_hg_dynamic(
                edge_ebd,
                h2,
                sw,
                owner=edge_index[:, 0],
                num_owner=nframes * nloc,
                nloc=nloc,
                scale_factor=(self.nnei / self.sel_reduce_factor) ** (-0.5),
            )
        )
        # (nb x nloc) x e_dim x 3
        rot_mat = torch.permute(h2g2, (0, 1, 3, 2))

        self.additional_output_for_fitting["angle_embd"] = angle_ebd

        return node_ebd, edge_ebd, h2, rot_mat.view(nframes, nloc, self.dim_emb, 3), sw

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
        if self.skip_stat and self.set_davg_zero:
            return
        env_mat_stat = EnvMatStatSe(self)
        if path is not None:
            path = path / env_mat_stat.get_hash()
        if path is None or not path.is_dir():
            if callable(merged):
                # only get data for once
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
        """Returns whether the descriptor block needs sorted nlist when using `forward_lower`."""
        return True
