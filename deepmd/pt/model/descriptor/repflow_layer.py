# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.descriptor.repformer_layer import (
    LocalAtten,
    _apply_nlist_mask,
    _apply_switch,
    _make_nei_g1,
    get_residual,
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
from deepmd.pt.utils.utils import (
    ActivationFn,
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class RepFlowLayer(torch.nn.Module):
    def __init__(
        self,
        e_rcut: float,
        e_rcut_smth: float,
        e_sel: int,
        a_rcut: float,
        a_rcut_smth: float,
        a_sel: int,
        ntypes: int,
        n_dim: int = 128,
        e_dim: int = 16,
        a_dim: int = 64,
        a_compress_rate: int = 0,
        a_mess_has_n: bool = True,
        a_use_e_mess: bool = False,
        a_compress_use_split: bool = False,
        a_compress_e_rate: int = 1,
        n_multi_edge_message: int = 1,
        axis_neuron: int = 4,
        update_angle: bool = True,  # angle
        update_n_has_h1: bool = False,
        update_e_has_h1: bool = False,
        h1_message_sub_axis: int = 4,
        h1_message_idc: bool = False,
        h1_message_only_nei: bool = False,
        h1_dim: int = 16,
        update_n_has_attn: bool = False,
        n_attn_hidden: int = 64,
        n_attn_head: int = 4,
        a_norm_use_max_v: bool = False,
        e_norm_use_max_v: bool = False,
        e_a_reduce_use_sqrt: bool = True,
        n_update_has_a: bool = False,
        n_update_has_a_first_sum: bool = False,
        pre_ln: bool = False,
        activation_function: str = "silu",
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        precision: str = "float64",
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        super().__init__()
        self.epsilon = 1e-4  # protection of 1./nnei
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.ntypes = ntypes
        e_sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(e_sel)
        assert len(e_sel) == 1
        self.e_sel = e_sel
        self.sec = self.e_sel
        self.a_rcut = a_rcut
        self.a_rcut_smth = a_rcut_smth
        self.a_sel = a_sel
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.h1_dim = h1_dim
        self.a_compress_rate = a_compress_rate
        self.a_use_e_mess = a_use_e_mess
        if a_compress_rate != 0:
            assert a_dim % (2 * a_compress_rate) == 0, (
                f"For a_compress_rate of {a_compress_rate}, a_dim must be divisible by {2 * a_compress_rate}. "
                f"Currently, a_dim={a_dim} is not valid."
            )
        self.n_multi_edge_message = n_multi_edge_message
        assert self.n_multi_edge_message >= 1, "n_multi_edge_message must >= 1!"
        self.axis_neuron = axis_neuron
        self.update_angle = update_angle
        self.activation_function = activation_function
        self.act = ActivationFn(activation_function)
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.a_mess_has_n = a_mess_has_n
        self.a_compress_e_rate = a_compress_e_rate
        self.a_compress_use_split = a_compress_use_split
        self.update_n_has_h1 = update_n_has_h1
        self.update_e_has_h1 = update_e_has_h1
        self.h1_message_sub_axis = h1_message_sub_axis
        self.h1_message_idc = h1_message_idc
        self.h1_message_only_nei = h1_message_only_nei
        self.update_n_has_attn = update_n_has_attn
        self.n_attn_hidden = n_attn_hidden
        self.n_attn_head = n_attn_head
        self.has_h1 = self.update_n_has_h1 or self.update_e_has_h1
        self.precision = precision
        self.seed = seed
        self.prec = PRECISION_DICT[precision]
        self.pre_ln = pre_ln
        self.a_norm_use_max_v = a_norm_use_max_v
        self.e_norm_use_max_v = e_norm_use_max_v
        self.e_a_reduce_use_sqrt = e_a_reduce_use_sqrt
        self.n_update_has_a = n_update_has_a
        self.n_update_has_a_first_sum = n_update_has_a_first_sum

        assert update_residual_init in [
            "norm",
            "const",
        ], "'update_residual_init' only support 'norm' or 'const'!"

        if self.pre_ln:
            assert self.update_style == "res_layer"

        if self.update_style == "res_layer":
            self.node_layernorm = nn.LayerNorm(
                self.n_dim,
                device=env.DEVICE,
                dtype=self.prec,
                elementwise_affine=False,
            )
            self.edge_layernorm = nn.LayerNorm(
                self.e_dim,
                device=env.DEVICE,
                dtype=self.prec,
                elementwise_affine=False,
            )
            self.angle_layernorm = nn.LayerNorm(
                self.a_dim,
                device=env.DEVICE,
                dtype=self.prec,
                elementwise_affine=False,
            )
            self.h1_layernorm = nn.LayerNorm(
                self.h1_dim,
                device=env.DEVICE,
                dtype=self.prec,
                elementwise_affine=False,
            )
        else:
            self.node_layernorm = None
            self.edge_layernorm = None
            self.angle_layernorm = None
            self.h1_layernorm = None

        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.n_residual = []
        self.e_residual = []
        self.a_residual = []
        self.h1_residual = []
        self.edge_info_dim = self.n_dim * 2 + self.e_dim
        # h_dim * h1_axis or h_dim * 1
        self.h_info_dim = (
            self.h1_dim * self.h1_message_sub_axis
            if not self.h1_message_idc
            else self.h1_dim
        )
        self.node_h1_dim = self.n_dim * 2 + self.e_dim
        self.node_h1_dim += (
            self.h_info_dim * 3 if not self.h1_message_only_nei else self.h_info_dim
        )
        self.node_h1_out_dim = 0
        self.node_h1_out_dim += self.n_dim if self.update_n_has_h1 else 0
        self.node_h1_out_dim += self.e_dim if self.update_e_has_h1 else 0
        self.node_h1_out_dim += self.h1_dim * 2

        # node self mlp
        self.node_self_mlp = MLPLayer(
            n_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        if self.update_style == "res_residual":
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 1),
                )
            )

        # node sym (grrg + drrd)
        self.n_sym_dim = n_dim * self.axis_neuron + e_dim * self.axis_neuron
        self.node_sym_linear = MLPLayer(
            self.n_sym_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 2),
        )
        if self.update_style == "res_residual":
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 3),
                )
            )

        # node edge message
        self.node_edge_linear = MLPLayer(
            self.edge_info_dim,
            self.n_multi_edge_message * n_dim,
            precision=precision,
            seed=child_seed(seed, 4),
        )
        if self.update_style == "res_residual":
            for head_index in range(self.n_multi_edge_message):
                self.n_residual.append(
                    get_residual(
                        n_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(child_seed(seed, 5), head_index),
                    )
                )

        # node local attention
        if self.update_n_has_attn:
            self.node_attn = LocalAtten(
                self.n_dim,
                self.n_attn_hidden,
                self.n_attn_head,
                True,
                precision=precision,
                seed=child_seed(seed, 6),
            )
            if self.update_style == "res_residual":
                self.n_residual.append(
                    get_residual(
                        n_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 3),
                    )
                )
        else:
            self.node_attn = None

        # h1 message
        if self.has_h1:
            self.h1_linear = MLPLayer(
                self.node_h1_dim,
                self.node_h1_out_dim,
                precision=precision,
                seed=child_seed(seed, 5),
            )
            if self.update_style == "res_residual":
                self.h1_residual.append(
                    get_residual(
                        self.h1_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 16),
                    )
                )
        else:
            self.h1_linear = None

        # node h1 message
        if self.update_n_has_h1 and (self.update_style == "res_residual"):
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 15),
                )
            )

        # edge h1 message
        if self.update_e_has_h1 and (self.update_style == "res_residual"):
            self.e_residual.append(
                get_residual(
                    e_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 16),
                )
            )

        # edge self message
        self.edge_self_linear = MLPLayer(
            self.edge_info_dim,
            e_dim,
            precision=precision,
            seed=child_seed(seed, 6),
        )
        if self.update_style == "res_residual":
            self.e_residual.append(
                get_residual(
                    e_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 7),
                )
            )

        if self.update_angle:
            self.angle_dim = self.a_dim
            if self.a_compress_rate == 0:
                # angle + node + edge * 2
                self.angle_dim += self.n_dim if self.a_mess_has_n else 0
                self.angle_dim += 2 * self.e_dim
                self.a_compress_n_linear = None
                self.a_compress_e_linear = None
                self.e_a_compress_dim = 0
                self.n_a_compress_dim = 0
            else:
                # angle + node/c + edge/2c * 2
                # node : node/c or 0
                self.angle_dim += (
                    self.a_dim // self.a_compress_rate if self.a_mess_has_n else 0
                )
                # edge : edge/2c * 2 * e_rate
                self.angle_dim += (
                    self.a_dim // self.a_compress_rate
                ) * self.a_compress_e_rate
                self.e_a_compress_dim = (
                    self.a_dim // (2 * self.a_compress_rate) * self.a_compress_e_rate
                )
                self.n_a_compress_dim = self.a_dim // self.a_compress_rate
                if not self.a_compress_use_split:
                    self.a_compress_n_linear = MLPLayer(
                        self.n_dim,
                        self.n_a_compress_dim,
                        precision=precision,
                        bias=False,
                        seed=child_seed(seed, 8),
                    )
                    self.a_compress_e_linear = MLPLayer(
                        self.e_dim,
                        self.e_a_compress_dim,
                        precision=precision,
                        bias=False,
                        seed=child_seed(seed, 9),
                    )
                else:
                    self.a_compress_n_linear = None
                    self.a_compress_e_linear = None

            # node angle message
            if self.n_update_has_a:
                self.node_angle_linear = MLPLayer(
                    self.angle_dim,
                    self.n_dim,
                    precision=precision,
                    seed=child_seed(seed, 15),
                )
                if self.update_style == "res_residual":
                    self.n_residual.append(
                        get_residual(
                            self.n_dim,
                            self.update_residual,
                            self.update_residual_init,
                            precision=precision,
                            seed=child_seed(seed, 16),
                        )
                    )
            else:
                self.node_angle_linear = None

            # edge angle message
            self.edge_angle_linear1 = MLPLayer(
                self.angle_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, 10),
            )
            self.edge_angle_linear2 = MLPLayer(
                self.e_dim,
                self.e_dim,
                precision=precision,
                seed=child_seed(seed, 11),
            )
            if self.update_style == "res_residual":
                self.e_residual.append(
                    get_residual(
                        self.e_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 12),
                    )
                )

            # angle self message
            self.angle_self_linear = MLPLayer(
                self.angle_dim,
                self.a_dim,
                precision=precision,
                seed=child_seed(seed, 13),
            )
            if self.update_style == "res_residual":
                self.a_residual.append(
                    get_residual(
                        self.a_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 14),
                    )
                )
        else:
            self.angle_self_linear = None
            self.edge_angle_linear1 = None
            self.edge_angle_linear2 = None
            self.a_compress_n_linear = None
            self.a_compress_e_linear = None
            self.angle_dim = 0

        self.n_residual = nn.ParameterList(self.n_residual)
        self.e_residual = nn.ParameterList(self.e_residual)
        self.a_residual = nn.ParameterList(self.a_residual)
        self.h1_residual = nn.ParameterList(self.h1_residual)

    @staticmethod
    def _cal_hg(
        edge_ebd: torch.Tensor,
        h2: torch.Tensor,
        nlist_mask: torch.Tensor,
        sw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the transposed rotation matrix.

        Parameters
        ----------
        edge_ebd
            Neighbor-wise/Pair-wise edge embeddings, with shape nb x nloc x nnei x e_dim.
        h2
            Neighbor-wise/Pair-wise equivariant rep tensors, with shape nb x nloc x nnei x 3.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.

        Returns
        -------
        hg
            The transposed rotation matrix, with shape nb x nloc x 3 x e_dim.
        """
        # edge_ebd:  nb x nloc x nnei x e_dim
        # h2:  nb x nloc x nnei x 3
        # msk: nb x nloc x nnei
        nb, nloc, nnei, _ = edge_ebd.shape
        e_dim = edge_ebd.shape[-1]
        # nb x nloc x nnei x e_dim
        edge_ebd = _apply_nlist_mask(edge_ebd, nlist_mask)
        edge_ebd = _apply_switch(edge_ebd, sw)
        invnnei = torch.rsqrt(
            float(nnei)
            * torch.ones((nb, nloc, 1, 1), dtype=edge_ebd.dtype, device=edge_ebd.device)
        )
        # nb x nloc x 3 x e_dim
        h2g2 = torch.matmul(torch.transpose(h2, -1, -2), edge_ebd) * invnnei
        return h2g2

    @staticmethod
    def _cal_grrg(h2g2: torch.Tensor, axis_neuron: int) -> torch.Tensor:
        """
        Calculate the atomic invariant rep.

        Parameters
        ----------
        h2g2
            The transposed rotation matrix, with shape nb x nloc x 3 x e_dim.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x e_dim)
        """
        # nb x nloc x 3 x e_dim
        nb, nloc, _, e_dim = h2g2.shape
        # nb x nloc x 3 x axis
        h2g2m = torch.split(h2g2, axis_neuron, dim=-1)[0]
        # nb x nloc x axis x e_dim
        g1_13 = torch.matmul(torch.transpose(h2g2m, -1, -2), h2g2) / (3.0**1)
        # nb x nloc x (axisxng2)
        g1_13 = g1_13.view(nb, nloc, axis_neuron * e_dim)
        return g1_13

    def symmetrization_op(
        self,
        edge_ebd: torch.Tensor,
        h2: torch.Tensor,
        nlist_mask: torch.Tensor,
        sw: torch.Tensor,
        axis_neuron: int,
    ) -> torch.Tensor:
        """
        Symmetrization operator to obtain atomic invariant rep.

        Parameters
        ----------
        edge_ebd
            Neighbor-wise/Pair-wise invariant rep tensors, with shape nb x nloc x nnei x e_dim.
        h2
            Neighbor-wise/Pair-wise equivariant rep tensors, with shape nb x nloc x nnei x 3.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x e_dim)
        """
        # edge_ebd:  nb x nloc x nnei x e_dim
        # h2:  nb x nloc x nnei x 3
        # msk: nb x nloc x nnei
        nb, nloc, nnei, _ = edge_ebd.shape
        # nb x nloc x 3 x e_dim
        h2g2 = self._cal_hg(
            edge_ebd,
            h2,
            nlist_mask,
            sw,
        )
        # nb x nloc x (axisxng2)
        g1_13 = self._cal_grrg(h2g2, axis_neuron)
        return g1_13

    def forward(
        self,
        node_ebd_ext: torch.Tensor,  # nf x nall x n_dim
        edge_ebd: torch.Tensor,  # nf x nloc x nnei x e_dim
        h2: torch.Tensor,  # nf x nloc x nnei x 3
        angle_ebd: torch.Tensor,  # nf x nloc x a_nnei x a_nnei x a_dim
        nlist: torch.Tensor,  # nf x nloc x nnei
        nlist_mask: torch.Tensor,  # nf x nloc x nnei
        sw: torch.Tensor,  # switch func, nf x nloc x nnei
        a_nlist: torch.Tensor,  # nf x nloc x a_nnei
        a_nlist_mask: torch.Tensor,  # nf x nloc x a_nnei
        a_sw: torch.Tensor,  # switch func, nf x nloc x a_nnei
        h1_ext: Optional[torch.Tensor],  # nf x nall x 3 x h1_dim
    ):
        """
        Parameters
        ----------
        node_ebd_ext : nf x nall x n_dim
            Extended node embedding.
        edge_ebd : nf x nloc x nnei x e_dim
            Edge embedding.
        h2 : nf x nloc x nnei x 3
            Pair-atom channel, equivariant.
        angle_ebd : nf x nloc x a_nnei x a_nnei x a_dim
            Angle embedding.
        nlist : nf x nloc x nnei
            Neighbor list. (padded neis are set to 0)
        nlist_mask : nf x nloc x nnei
            Masks of the neighbor list. real nei 1 otherwise 0
        sw : nf x nloc x nnei
            Switch function.
        a_nlist : nf x nloc x a_nnei
            Neighbor list for angle. (padded neis are set to 0)
        a_nlist_mask : nf x nloc x a_nnei
            Masks of the neighbor list for angle. real nei 1 otherwise 0
        a_sw : nf x nloc x a_nnei
            Switch function for angle.

        Returns
        -------
        n_updated:     nf x nloc x n_dim
            Updated node embedding.
        e_updated:     nf x nloc x nnei x e_dim
            Updated edge embedding.
        a_updated : nf x nloc x a_nnei x a_nnei x a_dim
            Updated angle embedding.
        """
        nb, nloc, nnei, _ = edge_ebd.shape
        nall = node_ebd_ext.shape[1]
        node_ebd, _ = torch.split(node_ebd_ext, [nloc, nall - nloc], dim=1)
        assert (nb, nloc) == node_ebd.shape[:2]
        assert (nb, nloc, nnei) == h2.shape[:3]
        del a_nlist  # may be used in the future

        n_update_list: list[torch.Tensor] = [node_ebd]
        e_update_list: list[torch.Tensor] = [edge_ebd]
        a_update_list: list[torch.Tensor] = [angle_ebd]
        h1_update_list: list[torch.Tensor] = []

        # pre layernorm
        if self.pre_ln:
            assert self.node_layernorm is not None
            assert self.edge_layernorm is not None
            assert self.angle_layernorm is not None
            node_ebd_ext = self.node_layernorm(node_ebd_ext)
            node_ebd, _ = torch.split(node_ebd_ext, [nloc, nall - nloc], dim=1)
            edge_ebd = self.edge_layernorm(edge_ebd)
            angle_ebd = self.angle_layernorm(angle_ebd)

        # only norm angle with max absolute value
        if self.a_norm_use_max_v:
            angle_ebd = angle_ebd / (angle_ebd.abs().max(-1)[0] + 1e-5).unsqueeze(-1)

        # only norm edge with max absolute value
        if self.e_norm_use_max_v:
            edge_ebd = edge_ebd / (edge_ebd.abs().max(-1)[0] + 1e-5).unsqueeze(-1)

        # node self mlp
        node_self_mlp = self.act(self.node_self_mlp(node_ebd))
        n_update_list.append(node_self_mlp)

        nei_node_ebd = _make_nei_g1(node_ebd_ext, nlist)

        # node sym (grrg + drrd)
        node_sym_list: list[torch.Tensor] = []
        node_sym_list.append(
            self.symmetrization_op(
                edge_ebd,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
            )
        )
        node_sym_list.append(
            self.symmetrization_op(
                nei_node_ebd,
                h2,
                nlist_mask,
                sw,
                self.axis_neuron,
            )
        )
        node_sym = self.act(self.node_sym_linear(torch.cat(node_sym_list, dim=-1)))
        n_update_list.append(node_sym)

        # nb x nloc x nnei x (n_dim * 2 + e_dim)
        edge_info = torch.cat(
            [
                torch.tile(node_ebd.unsqueeze(-2), [1, 1, self.nnei, 1]),
                nei_node_ebd,
                edge_ebd,
            ],
            dim=-1,
        )

        # node edge message
        # nb x nloc x nnei x (h * n_dim)
        node_edge_update = self.act(self.node_edge_linear(edge_info)) * sw.unsqueeze(-1)
        node_edge_update = torch.sum(node_edge_update, dim=-2) / self.nnei
        if self.n_multi_edge_message > 1:
            # nb x nloc x nnei x h x n_dim
            node_edge_update_mul_head = node_edge_update.view(
                nb, nloc, self.n_multi_edge_message, self.n_dim
            )
            for head_index in range(self.n_multi_edge_message):
                n_update_list.append(node_edge_update_mul_head[:, :, head_index, :])
        else:
            n_update_list.append(node_edge_update)

        # node local attn
        if self.update_n_has_attn:
            assert self.node_attn is not None
            n_update_list.append(self.node_attn(node_ebd, nei_node_ebd, nlist_mask, sw))

        # h1 message
        if self.has_h1:
            assert h1_ext is not None
            assert self.h1_linear is not None
            # nf x nloc x 3 x h1_dim
            h1 = h1_ext[:, :nloc]
            h1_update_list.append(h1)

            if self.pre_ln:
                assert self.h1_layernorm is not None
                h1_ext = self.h1_layernorm(h1_ext)
                h1 = h1_ext[:, :nloc]

            # j: nf x nloc x nnei x 3 x h1_dim
            nei_h1 = _make_nei_g1(h1_ext.view(nb, nall, -1), nlist).view(
                nb, nloc, nnei, 3, -1
            )
            # i: nf x nloc x nnei x 3 x h1_dim
            center_h1 = h1.unsqueeze(2).expand(-1, -1, nnei, -1, -1)
            # hiThj : nf x nloc x nnei x (h1_dim x h1_axis) or nf x nloc x nnei x h1_dim
            h1i_h1j = (
                torch.matmul(
                    center_h1.transpose(3, 4), nei_h1[..., : self.h1_message_sub_axis]
                ).view(nb, nloc, nnei, -1)
                if not self.h1_message_idc
                else ((center_h1 * nei_h1).sum(dim=-2))
            )
            h1_message_list = [
                torch.tile(node_ebd.unsqueeze(-2), [1, 1, self.nnei, 1]),
                nei_node_ebd,
                edge_ebd,
            ]
            if not self.h1_message_only_nei:
                # hiThi : nf x nloc x nnei x (h1_dim x h1_axis) or nf x nloc x nnei x h1_dim
                h1i_h1i = (
                    (
                        torch.matmul(
                            h1.transpose(2, 3), h1[..., : self.h1_message_sub_axis]
                        )
                        .view(nb, nloc, -1)
                        .unsqueeze(2)
                        .expand(-1, -1, nnei, -1)
                    )
                    if not self.h1_message_idc
                    else ((h1 * h1).sum(dim=-2).unsqueeze(2).expand(-1, -1, nnei, -1))
                )
                # hjThj : nf x nloc x nnei x (h1_dim x h1_axis) or nf x nloc x nnei x h1_dim
                h1j_h1j = (
                    torch.matmul(
                        nei_h1.transpose(3, 4), nei_h1[..., : self.h1_message_sub_axis]
                    ).view(nb, nloc, nnei, -1)
                    if not self.h1_message_idc
                    else ((nei_h1 * nei_h1).sum(dim=-2))
                )
                h1_message_list += [h1i_h1i, h1i_h1j, h1j_h1j]
            else:
                h1_message_list += [h1i_h1j]

            # nf x nloc x nnei x (2 * n_dim + e_dim + 3 * (h1_dim x h1_axis))
            h1_info = torch.cat(h1_message_list, dim=-1)
            # nf x nloc x nnei x (n_dim + 2 * h1_dim)
            h1_info_mlp = self.act(self.h1_linear(h1_info)) * sw.unsqueeze(-1)
            h1_out_axis_dim = 0

            # node h1 message
            if self.update_n_has_h1:
                # nf x nloc x nnei x n_dim
                node_h1_update = h1_info_mlp[
                    ..., h1_out_axis_dim : h1_out_axis_dim + self.n_dim
                ]
                h1_out_axis_dim += self.n_dim
                # nf x nloc x n_dim
                node_h1_update = torch.sum(node_h1_update, dim=-2) / self.nnei
                n_update_list.append(node_h1_update)

            # edge h1 message
            if self.update_e_has_h1:
                # nf x nloc x nnei x e_dim
                edge_h1_update = h1_info_mlp[
                    ..., h1_out_axis_dim : h1_out_axis_dim + self.e_dim
                ]
                h1_out_axis_dim += self.e_dim
                e_update_list.append(edge_h1_update)

            # h1 self message
            # nf x nloc x nnei x h1_dim
            h1_i_update = h1_info_mlp[
                ..., h1_out_axis_dim : h1_out_axis_dim + self.h1_dim
            ]
            h1_j_update = h1_info_mlp[..., h1_out_axis_dim + self.h1_dim :]
            # nf x nloc x nnei x 3 x h1_dim
            h1_node_update = (
                h1_i_update.unsqueeze(3) * center_h1 + h1_j_update.unsqueeze(3) * nei_h1
            )
            # nf x nloc x 3 x h1_dim
            h1_node_update = torch.sum(h1_node_update, dim=-3) / self.nnei
            h1_update_list.append(h1_node_update)
            h1_update = self.list_update(h1_update_list, "h1")
        else:
            h1_update = None

        # edge self message
        edge_self_update = self.act(self.edge_self_linear(edge_info))
        e_update_list.append(edge_self_update)

        if self.update_angle:
            assert self.angle_self_linear is not None
            assert self.edge_angle_linear1 is not None
            assert self.edge_angle_linear2 is not None
            if self.a_use_e_mess:
                edge_ebd_for_a_before_cp = edge_self_update
            else:
                edge_ebd_for_a_before_cp = edge_ebd
            # get angle info
            if self.a_compress_rate != 0:
                if not self.a_compress_use_split:
                    assert self.a_compress_n_linear is not None
                    assert self.a_compress_e_linear is not None
                    node_ebd_for_angle = self.a_compress_n_linear(node_ebd)
                    edge_ebd_for_angle = self.a_compress_e_linear(
                        edge_ebd_for_a_before_cp
                    )
                else:
                    # use the first a_compress_dim dim for node and edge
                    node_ebd_for_angle = node_ebd[:, :, : self.n_a_compress_dim]
                    edge_ebd_for_angle = edge_ebd_for_a_before_cp[
                        :, :, :, : self.e_a_compress_dim
                    ]
            else:
                node_ebd_for_angle = node_ebd
                edge_ebd_for_angle = edge_ebd_for_a_before_cp

            # nb x nloc x a_nnei x a_nnei x n_dim
            node_for_angle_info = torch.tile(
                node_ebd_for_angle.unsqueeze(2).unsqueeze(2),
                (1, 1, self.a_sel, self.a_sel, 1),
            )
            # nb x nloc x a_nnei x e_dim
            edge_for_angle = edge_ebd_for_angle[:, :, : self.a_sel, :]
            # nb x nloc x a_nnei x e_dim
            edge_for_angle = torch.where(
                a_nlist_mask.unsqueeze(-1), edge_for_angle, 0.0
            )
            # nb x nloc x (a_nnei) x a_nnei x edge_ebd
            edge_for_angle_i = torch.tile(
                edge_for_angle.unsqueeze(2), (1, 1, self.a_sel, 1, 1)
            )
            # nb x nloc x a_nnei x (a_nnei) x e_dim
            edge_for_angle_j = torch.tile(
                edge_for_angle.unsqueeze(3), (1, 1, 1, self.a_sel, 1)
            )
            # nb x nloc x a_nnei x a_nnei x (e_dim + e_dim)
            edge_for_angle_info = torch.cat(
                [edge_for_angle_i, edge_for_angle_j], dim=-1
            )
            angle_info_list = [angle_ebd]
            if self.a_mess_has_n:
                angle_info_list.append(node_for_angle_info)
            angle_info_list.append(edge_for_angle_info)
            # nb x nloc x a_nnei x a_nnei x (a + n_dim + e_dim*2) or (a + a/c + a/c)
            angle_info = torch.cat(angle_info_list, dim=-1)

            if self.n_update_has_a:
                # node angle message
                assert self.node_angle_linear is not None
                if not self.n_update_has_a_first_sum:
                    node_angle_update = self.act(self.node_angle_linear(angle_info))
                    # nb x nloc x a_nnei x a_nnei x n_dim
                    weighted_node_angle_update = (
                        node_angle_update
                        * a_sw[:, :, :, None, None]
                        * a_sw[:, :, None, :, None]
                    )
                    # nb x nloc x n_dim
                    reduced_node_angle_update = torch.sum(
                        torch.sum(weighted_node_angle_update, dim=-2), dim=-2
                    ) / (self.a_sel**2)
                else:
                    reduced_angle_info = (
                        angle_info
                        * a_sw[:, :, :, None, None]
                        * a_sw[:, :, None, :, None]
                    )
                    # nb x nloc x angle_dim
                    reduced_angle_info = torch.sum(
                        torch.sum(reduced_angle_info, dim=-2), dim=-2
                    ) / (self.a_sel**2)
                    # nb x nloc x n_dim
                    reduced_node_angle_update = self.act(
                        self.node_angle_linear(reduced_angle_info)
                    )
                n_update_list.append(reduced_node_angle_update)

            # edge angle message
            # nb x nloc x a_nnei x a_nnei x e_dim
            edge_angle_update = self.act(self.edge_angle_linear1(angle_info))
            # nb x nloc x a_nnei x a_nnei x e_dim
            weighted_edge_angle_update = (
                edge_angle_update
                * a_sw[:, :, :, None, None]
                * a_sw[:, :, None, :, None]
            )
            # nb x nloc x a_nnei x e_dim
            if self.e_a_reduce_use_sqrt:
                reduced_edge_angle_update = torch.sum(
                    weighted_edge_angle_update, dim=-2
                ) / (self.a_sel**0.5)
            else:
                reduced_edge_angle_update = (
                    torch.sum(weighted_edge_angle_update, dim=-2) / self.a_sel
                )
            # nb x nloc x nnei x e_dim
            padding_edge_angle_update = torch.concat(
                [
                    reduced_edge_angle_update,
                    torch.zeros(
                        [nb, nloc, self.nnei - self.a_sel, self.e_dim],
                        dtype=edge_ebd.dtype,
                        device=edge_ebd.device,
                    ),
                ],
                dim=2,
            )
            full_mask = torch.concat(
                [
                    a_nlist_mask,
                    torch.zeros(
                        [nb, nloc, self.nnei - self.a_sel],
                        dtype=a_nlist_mask.dtype,
                        device=a_nlist_mask.device,
                    ),
                ],
                dim=-1,
            )
            padding_edge_angle_update = torch.where(
                full_mask.unsqueeze(-1), padding_edge_angle_update, edge_ebd
            )
            e_update_list.append(
                self.act(self.edge_angle_linear2(padding_edge_angle_update))
            )
            # update node_ebd
            n_updated = self.list_update(n_update_list, "node")
            # update edge_ebd
            e_updated = self.list_update(e_update_list, "edge")

            # angle self message
            # nb x nloc x a_nnei x a_nnei x dim_a
            angle_self_update = self.act(self.angle_self_linear(angle_info))
            a_update_list.append(angle_self_update)
        else:
            # update node_ebd
            n_updated = self.list_update(n_update_list, "node")
            # update edge_ebd
            e_updated = self.list_update(e_update_list, "edge")

        # update angle_ebd
        a_updated = self.list_update(a_update_list, "angle")
        return n_updated, e_updated, a_updated, h1_update

    @torch.jit.export
    def list_update_res_avg(
        self,
        update_list: list[torch.Tensor],
    ) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        for ii in range(1, nitem):
            uu = uu + update_list[ii]
        return uu / (float(nitem) ** 0.5)

    @torch.jit.export
    def list_update_res_incr(self, update_list: list[torch.Tensor]) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        scale = 1.0 / (float(nitem - 1) ** 0.5) if nitem > 1 else 0.0
        for ii in range(1, nitem):
            uu = uu + scale * update_list[ii]
        return uu

    @torch.jit.export
    def list_update_res_residual(
        self, update_list: list[torch.Tensor], update_name: str = "node"
    ) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        # make jit happy
        if update_name == "node":
            for ii, vv in enumerate(self.n_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "edge":
            for ii, vv in enumerate(self.e_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "angle":
            for ii, vv in enumerate(self.a_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "h1":
            for ii, vv in enumerate(self.h1_residual):
                uu = uu + vv * update_list[ii + 1]
        else:
            raise NotImplementedError
        return uu

    @torch.jit.export
    def list_update_res_layer(
        self, update_list: list[torch.Tensor], update_name: str = "node"
    ) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        for ii in range(1, nitem):
            uu = uu + update_list[ii]
        if not self.pre_ln:
            # make jit happy
            if update_name == "node":
                assert self.node_layernorm is not None
                out = self.node_layernorm(uu)
            elif update_name == "edge":
                assert self.edge_layernorm is not None
                out = self.edge_layernorm(uu)
            elif update_name == "angle":
                assert self.angle_layernorm is not None
                out = self.angle_layernorm(uu)
            elif update_name == "h1":
                assert self.h1_layernorm is not None
                out = self.h1_layernorm(uu)
            else:
                raise NotImplementedError
        else:
            out = uu
        return out

    @torch.jit.export
    def list_update(
        self, update_list: list[torch.Tensor], update_name: str = "node"
    ) -> torch.Tensor:
        if self.update_style == "res_avg":
            return self.list_update_res_avg(update_list)
        elif self.update_style == "res_incr":
            return self.list_update_res_incr(update_list)
        elif self.update_style == "res_residual":
            return self.list_update_res_residual(update_list, update_name=update_name)
        elif self.update_style == "res_layer":
            return self.list_update_res_layer(update_list, update_name=update_name)
        else:
            raise RuntimeError(f"unknown update style {self.update_style}")

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        data = {
            "@class": "RepformerLayer",
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
            "a_compress_rate": self.a_compress_rate,
            "n_multi_edge_message": self.n_multi_edge_message,
            "axis_neuron": self.axis_neuron,
            "activation_function": self.activation_function,
            "update_angle": self.update_angle,
            "update_style": self.update_style,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
            "precision": self.precision,
            "node_self_mlp": self.node_self_mlp.serialize(),
            "node_sym_linear": self.node_sym_linear.serialize(),
            "node_edge_linear": self.node_edge_linear.serialize(),
            "edge_self_linear": self.edge_self_linear.serialize(),
        }
        if self.update_angle:
            data.update(
                {
                    "edge_angle_linear1": self.edge_angle_linear1.serialize(),
                    "edge_angle_linear2": self.edge_angle_linear2.serialize(),
                    "angle_self_linear": self.angle_self_linear.serialize(),
                }
            )
            if self.a_compress_rate != 0:
                data.update(
                    {
                        "a_compress_n_linear": self.a_compress_n_linear.serialize(),
                        "a_compress_e_linear": self.a_compress_e_linear.serialize(),
                    }
                )
        if self.update_style == "res_residual":
            data.update(
                {
                    "@variables": {
                        "n_residual": [to_numpy_array(t) for t in self.n_residual],
                        "e_residual": [to_numpy_array(t) for t in self.e_residual],
                        "a_residual": [to_numpy_array(t) for t in self.a_residual],
                    }
                }
            )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "RepFlowLayer":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        update_angle = data["update_angle"]
        a_compress_rate = data["a_compress_rate"]
        node_self_mlp = data.pop("node_self_mlp")
        node_sym_linear = data.pop("node_sym_linear")
        node_edge_linear = data.pop("node_edge_linear")
        edge_self_linear = data.pop("edge_self_linear")
        edge_angle_linear1 = data.pop("edge_angle_linear1", None)
        edge_angle_linear2 = data.pop("edge_angle_linear2", None)
        angle_self_linear = data.pop("angle_self_linear", None)
        a_compress_n_linear = data.pop("a_compress_n_linear", None)
        a_compress_e_linear = data.pop("a_compress_e_linear", None)
        update_style = data["update_style"]
        variables = data.pop("@variables", {})
        n_residual = variables.get("n_residual", data.pop("n_residual", []))
        e_residual = variables.get("e_residual", data.pop("e_residual", []))
        a_residual = variables.get("a_residual", data.pop("a_residual", []))

        obj = cls(**data)
        obj.node_self_mlp = MLPLayer.deserialize(node_self_mlp)
        obj.node_sym_linear = MLPLayer.deserialize(node_sym_linear)
        obj.node_edge_linear = MLPLayer.deserialize(node_edge_linear)
        obj.edge_self_linear = MLPLayer.deserialize(edge_self_linear)

        if update_angle:
            assert isinstance(edge_angle_linear1, dict)
            assert isinstance(edge_angle_linear2, dict)
            assert isinstance(angle_self_linear, dict)
            obj.edge_angle_linear1 = MLPLayer.deserialize(edge_angle_linear1)
            obj.edge_angle_linear2 = MLPLayer.deserialize(edge_angle_linear2)
            obj.angle_self_linear = MLPLayer.deserialize(angle_self_linear)
            if a_compress_rate != 0:
                assert isinstance(a_compress_n_linear, dict)
                assert isinstance(a_compress_e_linear, dict)
                obj.a_compress_n_linear = MLPLayer.deserialize(a_compress_n_linear)
                obj.a_compress_e_linear = MLPLayer.deserialize(a_compress_e_linear)

        if update_style == "res_residual":
            for ii, t in enumerate(obj.n_residual):
                t.data = to_torch_tensor(n_residual[ii])
            for ii, t in enumerate(obj.e_residual):
                t.data = to_torch_tensor(e_residual[ii])
            for ii, t in enumerate(obj.a_residual):
                t.data = to_torch_tensor(a_residual[ii])
        return obj
