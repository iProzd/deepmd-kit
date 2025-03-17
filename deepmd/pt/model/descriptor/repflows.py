# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
    Union,
)

import torch

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
    MLPLayer,
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
        a_mess_has_n: bool = True,
        a_use_e_mess: bool = False,
        a_compress_use_split: bool = False,
        n_multi_edge_message: int = 1,
        axis_neuron: int = 4,
        update_angle: bool = True,
        activation_function: str = "silu",
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        update_n_has_h1: bool = False,
        update_e_has_h1: bool = False,
        h1_message_sub_axis: int = 4,
        h1_message_idc: bool = False,
        h1_message_only_nei: bool = False,
        h1_dim: int = 16,
        update_n_has_attn: bool = False,
        n_attn_hidden: int = 64,
        n_attn_head: int = 4,
        set_davg_zero: bool = True,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        precision: str = "float64",
        skip_stat: bool = True,
        no_sym: bool = False,
        smooth_angle_init: bool = False,
        angle_init_use_sin: bool = False,
        smooth_edge_update: bool = False,
        pre_ln: bool = False,
        only_e_ln: bool = False,
        pre_bn: bool = False,
        only_e_bn: bool = False,
        use_unet: bool = False,
        use_unet_n: bool = True,
        use_unet_e: bool = True,
        use_unet_a: bool = True,
        unet_rate: float = 0.5,
        unet_norm: str = "None",
        bn_moment: float = 0.1,
        auto_batchsize: int = 0,
        optim_update: bool = True,
        a_norm_use_max_v: bool = False,
        e_norm_use_max_v: bool = False,
        e_a_reduce_use_sqrt: bool = True,
        n_update_has_a: bool = False,
        n_update_has_a_first_sum: bool = False,
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
            to n_dim/c and e_dim/2c, respectively, within the angular message.
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
        self.n_multi_edge_message = n_multi_edge_message
        self.axis_neuron = axis_neuron
        self.set_davg_zero = set_davg_zero
        self.skip_stat = skip_stat
        self.a_mess_has_n = a_mess_has_n
        self.a_use_e_mess = a_use_e_mess
        self.a_compress_e_rate = a_compress_e_rate
        self.a_compress_use_split = a_compress_use_split
        self.update_n_has_h1 = update_n_has_h1
        self.update_e_has_h1 = update_e_has_h1
        self.h1_message_sub_axis = h1_message_sub_axis
        self.h1_message_idc = h1_message_idc
        self.h1_message_only_nei = h1_message_only_nei
        self.h1_dim = h1_dim
        self.update_n_has_attn = update_n_has_attn
        self.n_update_has_a = n_update_has_a
        self.n_update_has_a_first_sum = n_update_has_a_first_sum
        self.n_attn_hidden = n_attn_hidden
        self.n_attn_head = n_attn_head
        self.auto_batchsize = auto_batchsize
        self.optim_update = optim_update
        self.no_sym = no_sym
        self.smooth_angle_init = smooth_angle_init
        self.angle_init_use_sin = angle_init_use_sin
        self.smooth_edge_update = smooth_edge_update

        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.update_angle = update_angle

        self.activation_function = activation_function
        self.act_list = []
        if "," in self.activation_function:
            self.use_mix_act = True
            for act_str in self.activation_function.split(","):
                act_name, act_number = act_str.split(":")
                act_number = int(act_number)
                self.act_list += [act_name] * act_number
            assert len(self.act_list) == self.nlayers
        else:
            self.use_mix_act = False
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.act = (
            ActivationFn(activation_function)
            if not self.use_mix_act
            else ActivationFn(self.act_list[0])
        )
        self.prec = PRECISION_DICT[precision]
        self.a_norm_use_max_v = a_norm_use_max_v
        self.e_norm_use_max_v = e_norm_use_max_v
        self.e_a_reduce_use_sqrt = e_a_reduce_use_sqrt

        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)
        self.env_protection = env_protection
        self.precision = precision
        self.epsilon = 1e-4
        self.seed = seed
        self.pre_ln = pre_ln
        self.only_e_ln = only_e_ln
        self.pre_bn = pre_bn
        self.only_e_bn = only_e_bn
        self.bn_moment = bn_moment
        self.use_unet = use_unet
        self.use_unet_n = use_unet_n
        self.use_unet_e = use_unet_e
        self.use_unet_a = use_unet_a
        self.unet_rate = unet_rate
        self.unet_norm = unet_norm
        # self.out_ln = None
        # if self.pre_ln:
        #     self.out_ln = torch.nn.LayerNorm(
        #         self.n_dim,
        #         device=env.DEVICE,
        #         dtype=self.prec,
        #         elementwise_affine=False,
        #     )

        self.edge_embd = MLPLayer(
            1, self.e_dim, precision=precision, seed=child_seed(seed, 0)
        )
        self.angle_embd = MLPLayer(
            1 if not self.angle_init_use_sin else 2,
            self.a_dim,
            precision=precision,
            bias=False,
            seed=child_seed(seed, 1),
        )
        self.has_h1 = self.update_n_has_h1 or self.update_e_has_h1
        if self.has_h1:
            self.h1_embd = MLPLayer(
                1,
                self.h1_dim,
                precision=precision,
                bias=False,
                stddev=self.h1_dim**0.5,
                seed=child_seed(seed, 2),
            )
        else:
            self.h1_embd = None
        layers = []
        self.unet_scale = [1.0 for _ in range(self.nlayers)]
        self.unet_first_half = int((self.nlayers + 1) / 2)
        self.unet_rest_half = int(self.nlayers / 2)
        self.unet_norm_n = None
        self.unet_norm_e = None
        self.unet_norm_a = None
        if self.use_unet:
            self.unet_scale = [
                (self.unet_rate**i) for i in range(self.unet_first_half)
            ] + [
                (self.unet_rate ** (self.unet_rest_half - 1 - i))
                for i in range(self.unet_rest_half)
            ]
            if self.unet_norm != "None":
                norm_idx = self.unet_first_half - 1
                if self.unet_norm == "batchnorm":
                    self.unet_norm_n = (
                        torch.nn.BatchNorm1d(
                            int(self.n_dim * self.unet_scale[norm_idx]),
                            affine=False,
                            device=env.DEVICE,
                            dtype=self.prec,
                            momentum=self.bn_moment,
                        )
                        if self.use_unet_n
                        else None
                    )
                    self.unet_norm_e = (
                        torch.nn.BatchNorm1d(
                            int(self.e_dim * self.unet_scale[norm_idx]),
                            affine=False,
                            device=env.DEVICE,
                            dtype=self.prec,
                            momentum=self.bn_moment,
                        )
                        if self.use_unet_e
                        else None
                    )
                    self.unet_norm_a = (
                        torch.nn.BatchNorm1d(
                            int(self.a_dim * self.unet_scale[norm_idx]),
                            affine=False,
                            device=env.DEVICE,
                            dtype=self.prec,
                            momentum=self.bn_moment,
                        )
                        if self.use_unet_a
                        else None
                    )
                elif self.unet_norm == "layernorm":
                    self.unet_norm_n = (
                        torch.nn.LayerNorm(
                            int(self.n_dim * self.unet_scale[norm_idx]),
                            device=env.DEVICE,
                            dtype=self.prec,
                            elementwise_affine=False,
                        )
                        if self.use_unet_n
                        else None
                    )
                    self.unet_norm_e = (
                        torch.nn.LayerNorm(
                            int(self.e_dim * self.unet_scale[norm_idx]),
                            device=env.DEVICE,
                            dtype=self.prec,
                            elementwise_affine=False,
                        )
                        if self.use_unet_e
                        else None
                    )
                    self.unet_norm_a = (
                        torch.nn.LayerNorm(
                            int(self.a_dim * self.unet_scale[norm_idx]),
                            device=env.DEVICE,
                            dtype=self.prec,
                            elementwise_affine=False,
                        )
                        if self.use_unet_a
                        else None
                    )
                else:
                    raise ValueError(f"Unsupported unet norm {self.unet_norm}!")

        for ii in range(nlayers):
            layers.append(
                RepFlowLayer(
                    e_rcut=self.e_rcut,
                    e_rcut_smth=self.e_rcut_smth,
                    e_sel=self.sel,
                    a_rcut=self.a_rcut,
                    a_rcut_smth=self.a_rcut_smth,
                    a_sel=self.a_sel,
                    ntypes=self.ntypes,
                    n_dim=self.n_dim
                    if (not self.use_unet or not self.use_unet_n)
                    else int(self.n_dim * self.unet_scale[ii]),
                    e_dim=self.e_dim
                    if (not self.use_unet or not self.use_unet_e)
                    else int(self.e_dim * self.unet_scale[ii]),
                    a_dim=self.a_dim
                    if (not self.use_unet or not self.use_unet_a)
                    else int(self.a_dim * self.unet_scale[ii]),
                    a_compress_rate=self.a_compress_rate,
                    a_mess_has_n=self.a_mess_has_n,
                    a_use_e_mess=self.a_use_e_mess,
                    a_compress_use_split=self.a_compress_use_split,
                    a_compress_e_rate=self.a_compress_e_rate,
                    n_multi_edge_message=self.n_multi_edge_message,
                    axis_neuron=self.axis_neuron,
                    update_angle=self.update_angle,
                    update_n_has_h1=self.update_n_has_h1,
                    update_e_has_h1=self.update_e_has_h1,
                    h1_message_sub_axis=self.h1_message_sub_axis,
                    h1_message_idc=self.h1_message_idc,
                    h1_message_only_nei=self.h1_message_only_nei,
                    h1_dim=self.h1_dim,
                    update_n_has_attn=self.update_n_has_attn,
                    n_attn_hidden=self.n_attn_hidden,
                    n_attn_head=self.n_attn_head,
                    a_norm_use_max_v=self.a_norm_use_max_v,
                    e_norm_use_max_v=self.e_norm_use_max_v,
                    e_a_reduce_use_sqrt=self.e_a_reduce_use_sqrt,
                    activation_function=self.activation_function
                    if not self.use_mix_act
                    else self.act_list[ii],
                    update_style=self.update_style,
                    update_residual=self.update_residual,
                    update_residual_init=self.update_residual_init,
                    n_update_has_a=self.n_update_has_a,
                    n_update_has_a_first_sum=self.n_update_has_a_first_sum,
                    precision=precision,
                    pre_ln=self.pre_ln,
                    only_e_ln=self.only_e_ln,
                    pre_bn=self.pre_bn,
                    only_e_bn=self.only_e_bn,
                    bn_moment=self.bn_moment,
                    optim_update=self.optim_update,
                    no_sym=self.no_sym,
                    smooth_edge_update=self.smooth_edge_update,
                    seed=child_seed(child_seed(seed, 1), ii),
                )
            )
        self.layers = torch.nn.ModuleList(layers)

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

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
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
        return self.n_dim

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

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: Optional[torch.Tensor] = None,
        mapping: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ):
        if comm_dict is None:
            assert mapping is not None
            assert extended_atype_embd is not None
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
        )
        nlist_mask = nlist != -1
        sw = torch.squeeze(sw, -1)
        # beyond the cutoff sw should be 0.0
        sw = sw.masked_fill(~nlist_mask, 0.0)

        # [nframes, nloc, tebd_dim]
        if comm_dict is None:
            assert isinstance(extended_atype_embd, torch.Tensor)  # for jit
            atype_embd = extended_atype_embd[:, :nloc, :]
            assert list(atype_embd.shape) == [nframes, nloc, self.n_dim]
        else:
            atype_embd = extended_atype_embd
        assert isinstance(atype_embd, torch.Tensor)  # for jit
        node_ebd = self.act(atype_embd)
        n_dim = node_ebd.shape[-1]
        # nb x nloc x nnei x 1,  nb x nloc x nnei x 3
        edge_input, h2 = torch.split(dmatrix, [1, 3], dim=-1)
        # nb x nloc x nnei x e_dim
        edge_ebd = self.act(self.edge_embd(edge_input))

        # get angle nlist (maybe smaller)
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
        )
        a_nlist_mask = a_nlist != -1
        a_sw = torch.squeeze(a_sw, -1)
        # beyond the cutoff sw should be 0.0
        a_sw = a_sw.masked_fill(~a_nlist_mask, 0.0)
        a_nlist[a_nlist == -1] = 0

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

        if not self.angle_init_use_sin:
            # nf x nloc x a_nnei x a_nnei x 1
            angle_input = cosine_ij.unsqueeze(-1) / (torch.pi**0.5)
        else:
            angle_input = torch.cat(
                [cosine_ij.unsqueeze(-1), sine_ij.unsqueeze(-1)], dim=-1
            ) / (torch.pi**0.5)

        # nf x nloc x a_nnei x a_nnei x a_dim
        angle_ebd = self.angle_embd(angle_input).reshape(
            nframes, nloc, self.a_sel, self.a_sel, self.a_dim
        )
        if self.has_h1:
            assert self.h1_embd is not None
            h1 = torch.sum(
                self.h1_embd(h2.view([nframes, nloc, nnei, 3, 1])), dim=2
            ) / (nnei**0.5)
        else:
            h1 = None

        # set all padding positions to index of 0
        # if the a neighbor is real or not is indicated by nlist_mask
        nlist[nlist == -1] = 0
        # nb x nall x n_dim
        if comm_dict is None:
            assert mapping is not None
            mapping3 = (
                mapping.view(nframes, nall)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, 3, self.h1_dim)
            )
            mapping = (
                mapping.view(nframes, nall).unsqueeze(-1).expand(-1, -1, self.n_dim)
            )
        else:
            mapping3 = None

        unet_list_node = []
        unet_list_edge = []
        unet_list_angle = []

        for idx, ll in enumerate(self.layers):
            # node_ebd:     nb x nloc x n_dim
            # node_ebd_ext: nb x nall x n_dim
            if comm_dict is None:
                assert mapping is not None
                assert mapping3 is not None
                node_ebd_ext = torch.gather(
                    node_ebd, 1, mapping[:, :, : node_ebd.shape[-1]]
                )
                if self.has_h1:
                    assert h1 is not None
                    h1_ext = torch.gather(h1, 1, mapping3)
                else:
                    h1_ext = None
            else:
                h1_ext = None
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
            if self.auto_batchsize != 0 and nframes * nloc > self.auto_batchsize:
                node_ebd_full, _ = torch.split(node_ebd_ext, [nloc, nall - nloc], dim=1)
                node_ebd_chunks = torch.split(node_ebd_full, self.auto_batchsize, dim=1)
                edge_ebd_chunks = torch.split(edge_ebd, self.auto_batchsize, dim=1)
                h2_chunks = torch.split(h2, self.auto_batchsize, dim=1)
                angle_ebd_chunks = torch.split(angle_ebd, self.auto_batchsize, dim=1)
                nlist_chunks = torch.split(nlist, self.auto_batchsize, dim=1)
                nlist_mask_chunks = torch.split(nlist_mask, self.auto_batchsize, dim=1)
                sw_chunks = torch.split(sw, self.auto_batchsize, dim=1)
                a_nlist_chunks = torch.split(a_nlist, self.auto_batchsize, dim=1)
                a_nlist_mask_chunks = torch.split(
                    a_nlist_mask, self.auto_batchsize, dim=1
                )
                a_sw_chunks = torch.split(a_sw, self.auto_batchsize, dim=1)

                node_ebd_list = []
                edge_ebd_list = []
                angle_ebd_list = []
                for (
                    node_ebd_sub,
                    edge_ebd_sub,
                    h2_sub,
                    angle_ebd_sub,
                    nlist_sub,
                    nlist_mask_sub,
                    sw_sub,
                    a_nlist_sub,
                    a_nlist_mask_sub,
                    a_sw_sub,
                ) in zip(
                    node_ebd_chunks,
                    edge_ebd_chunks,
                    h2_chunks,
                    angle_ebd_chunks,
                    nlist_chunks,
                    nlist_mask_chunks,
                    sw_chunks,
                    a_nlist_chunks,
                    a_nlist_mask_chunks,
                    a_sw_chunks,
                ):
                    node_ebd_tmp, edge_ebd_tmp, angle_ebd_tmp, h1_tmp = ll.forward(
                        node_ebd_ext,
                        edge_ebd_sub,
                        h2_sub,
                        angle_ebd_sub,
                        nlist_sub,
                        nlist_mask_sub,
                        sw_sub,
                        a_nlist_sub,
                        a_nlist_mask_sub,
                        a_sw_sub,
                        h1_ext,
                        node_ebd_split=node_ebd_sub,
                    )
                    node_ebd_list.append(node_ebd_tmp)
                    edge_ebd_list.append(edge_ebd_tmp)
                    angle_ebd_list.append(angle_ebd_tmp)
                node_ebd = torch.cat(node_ebd_list, dim=1)
                edge_ebd = torch.cat(edge_ebd_list, dim=1)
                angle_ebd = torch.cat(angle_ebd_list, dim=1)
            else:
                node_ebd, edge_ebd, angle_ebd, h1 = ll.forward(
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
                    h1_ext,
                )
            if self.use_unet:
                if self.unet_norm != "None" and idx == self.unet_first_half - 1:
                    if self.use_unet_n:
                        assert self.unet_norm_n is not None
                        node_ebd = self.unet_norm_n(
                            node_ebd.view(nframes * nloc, -1)
                        ).view(nframes, nloc, -1)
                    if self.use_unet_e:
                        assert self.unet_norm_e is not None
                        edge_ebd = self.unet_norm_e(
                            edge_ebd.view(nframes * nloc * self.nnei, -1)
                        ).view(nframes, nloc, nnei, -1)
                    if self.use_unet_a:
                        assert self.unet_norm_a is not None
                        angle_ebd = self.unet_norm_a(
                            angle_ebd.view(nframes * nloc * self.a_sel * self.a_sel, -1)
                        ).view(nframes, nloc, self.a_sel, self.a_sel, -1)
                if idx < self.unet_first_half - 1:
                    # stack half output
                    tmp_n_dim = int(self.n_dim * self.unet_scale[idx + 1])
                    tmp_e_dim = int(self.e_dim * self.unet_scale[idx + 1])
                    tmp_a_dim = int(self.a_dim * self.unet_scale[idx + 1])
                    if self.use_unet_n:
                        ori_dim = node_ebd.shape[-1]
                        stack_node_ebd, node_ebd = torch.split(
                            node_ebd, [ori_dim - tmp_n_dim, tmp_n_dim], dim=-1
                        )
                        unet_list_node.append(stack_node_ebd)
                    if self.use_unet_e:
                        ori_dim = edge_ebd.shape[-1]
                        stack_edge_ebd, edge_ebd = torch.split(
                            edge_ebd, [ori_dim - tmp_e_dim, tmp_e_dim], dim=-1
                        )
                        unet_list_edge.append(stack_edge_ebd)
                    if self.use_unet_a:
                        ori_dim = angle_ebd.shape[-1]
                        stack_angle_ebd, angle_ebd = torch.split(
                            angle_ebd, [ori_dim - tmp_a_dim, tmp_a_dim], dim=-1
                        )
                        unet_list_angle.append(stack_angle_ebd)
                elif self.unet_rest_half - 1 < idx < self.nlayers - 1:
                    # skip connection, concat the half output
                    if self.use_unet_n:
                        stack_node_ebd = unet_list_node.pop()
                        node_ebd = torch.cat([stack_node_ebd, node_ebd], dim=-1)
                    if self.use_unet_e:
                        stack_edge_ebd = unet_list_edge.pop()
                        edge_ebd = torch.cat([stack_edge_ebd, edge_ebd], dim=-1)
                    if self.use_unet_a:
                        stack_angle_ebd = unet_list_angle.pop()
                        angle_ebd = torch.cat([stack_angle_ebd, angle_ebd], dim=-1)
        # nb x nloc x 3 x e_dim
        h2g2 = RepFlowLayer._cal_hg(edge_ebd, h2, nlist_mask, sw)
        # (nb x nloc) x e_dim x 3
        rot_mat = torch.permute(h2g2, (0, 1, 3, 2))
        # if self.pre_ln:
        #     assert self.out_ln is not None
        #     node_ebd = self.out_ln(node_ebd)

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
