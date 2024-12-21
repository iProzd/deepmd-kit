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
from deepmd.pt.model.network.init import (
    constant_,
    normal_,
)
from deepmd.pt.model.network.layernorm import (
    LayerNorm,
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
    get_generator,
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


def get_residual(
    _dim: int,
    _scale: float,
    _mode: str = "norm",
    trainable: bool = True,
    precision: str = "float64",
    seed: Optional[Union[int, list[int]]] = None,
) -> torch.Tensor:
    r"""
    Get residual tensor for one update vector.

    Parameters
    ----------
    _dim : int
        The dimension of the update vector.
    _scale
        The initial scale of the residual tensor. See `_mode` for details.
    _mode
        The mode of residual initialization for the residual tensor.
        - "norm" (default): init residual using normal with `_scale` std.
        - "const": init residual using element-wise constants of `_scale`.
    trainable
        Whether the residual tensor is trainable.
    precision
        The precision of the residual tensor.
    seed : int, optional
        Random seed for parameter initialization.
    """
    random_generator = get_generator(seed)
    residual = nn.Parameter(
        data=torch.zeros(_dim, dtype=PRECISION_DICT[precision], device=env.DEVICE),
        requires_grad=trainable,
    )
    if _mode == "norm":
        normal_(residual.data, std=_scale, generator=random_generator)
    elif _mode == "const":
        constant_(residual.data, val=_scale)
    else:
        raise RuntimeError(f"Unsupported initialization mode '{_mode}'!")
    return residual


# common ops
def _make_nei_g1(
    g1_ext: torch.Tensor,
    nlist: torch.Tensor,
) -> torch.Tensor:
    """
    Make neighbor-wise atomic invariant rep.

    Parameters
    ----------
    g1_ext
        Extended atomic invariant rep, with shape nb x nall x ng1.
    nlist
        Neighbor list, with shape nb x nloc x nnei.

    Returns
    -------
    gg1: torch.Tensor
        Neighbor-wise atomic invariant rep, with shape nb x nloc x nnei x ng1.

    """
    # nlist: nb x nloc x nnei
    nb, nloc, nnei = nlist.shape
    # g1_ext: nb x nall x ng1
    ng1 = g1_ext.shape[-1]
    # index: nb x (nloc x nnei) x ng1
    index = nlist.reshape(nb, nloc * nnei).unsqueeze(-1).expand(-1, -1, ng1)
    # gg1  : nb x (nloc x nnei) x ng1
    gg1 = torch.gather(g1_ext, dim=1, index=index)
    # gg1  : nb x nloc x nnei x ng1
    gg1 = gg1.view(nb, nloc, nnei, ng1)
    return gg1


def _apply_nlist_mask(
    gg: torch.Tensor,
    nlist_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply nlist mask to neighbor-wise rep tensors.

    Parameters
    ----------
    gg
        Neighbor-wise rep tensors, with shape nf x nloc x nnei x d.
    nlist_mask
        Neighbor list mask, where zero means no neighbor, with shape nf x nloc x nnei.
    """
    # gg:  nf x nloc x nnei x d
    # msk: nf x nloc x nnei
    return gg.masked_fill(~nlist_mask.unsqueeze(-1), 0.0)


def _apply_switch(gg: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """
    Apply switch function to neighbor-wise rep tensors.

    Parameters
    ----------
    gg
        Neighbor-wise rep tensors, with shape nf x nloc x nnei x d.
    sw
        The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
        and remains 0 beyond rcut, with shape nf x nloc x nnei.
    """
    # gg:  nf x nloc x nnei x d
    # sw:  nf x nloc x nnei
    return gg * sw.unsqueeze(-1)


class Atten2Map(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        head_num: int,
        has_gate: bool = False,  # apply gate to attn map
        smooth: bool = True,
        attnw_shift: float = 20.0,
        precision: str = "float64",
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        """Return neighbor-wise multi-head self-attention maps, with gate mechanism."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.mapqk = MLPLayer(
            input_dim,
            hidden_dim * 2 * head_num,
            bias=False,
            precision=precision,
            seed=seed,
        )
        self.has_gate = has_gate
        self.smooth = smooth
        self.attnw_shift = attnw_shift
        self.precision = precision

    def forward(
        self,
        g2: torch.Tensor,  # nb x nloc x nnei x ng2
        h2: torch.Tensor,  # nb x nloc x nnei x 3
        nlist_mask: torch.Tensor,  # nb x nloc x nnei
        sw: torch.Tensor,  # nb x nloc x nnei
    ) -> torch.Tensor:
        (
            nb,
            nloc,
            nnei,
            _,
        ) = g2.shape
        nd, nh = self.hidden_dim, self.head_num
        # nb x nloc x nnei x nd x (nh x 2)
        g2qk = self.mapqk(g2).view(nb, nloc, nnei, nd, nh * 2)
        # nb x nloc x (nh x 2) x nnei x nd
        g2qk = torch.permute(g2qk, (0, 1, 4, 2, 3))
        # nb x nloc x nh x nnei x nd
        g2q, g2k = torch.split(g2qk, nh, dim=2)
        # g2q = torch.nn.functional.normalize(g2q, dim=-1)
        # g2k = torch.nn.functional.normalize(g2k, dim=-1)
        # nb x nloc x nh x nnei x nnei
        attnw = torch.matmul(g2q, torch.transpose(g2k, -1, -2)) / nd**0.5
        if self.has_gate:
            gate = torch.matmul(h2, torch.transpose(h2, -1, -2)).unsqueeze(-3)
            attnw = attnw * gate
        # mask the attenmap, nb x nloc x 1 x 1 x nnei
        attnw_mask = ~nlist_mask.unsqueeze(2).unsqueeze(2)
        # mask the attenmap, nb x nloc x 1 x nnei x 1
        attnw_mask_c = ~nlist_mask.unsqueeze(2).unsqueeze(-1)
        if self.smooth:
            attnw = (attnw + self.attnw_shift) * sw[:, :, None, :, None] * sw[
                :, :, None, None, :
            ] - self.attnw_shift
        else:
            attnw = attnw.masked_fill(
                attnw_mask,
                float("-inf"),
            )
        attnw = torch.softmax(attnw, dim=-1)
        attnw = attnw.masked_fill(
            attnw_mask,
            0.0,
        )
        # nb x nloc x nh x nnei x nnei
        attnw = attnw.masked_fill(
            attnw_mask_c,
            0.0,
        )
        if self.smooth:
            attnw = attnw * sw[:, :, None, :, None] * sw[:, :, None, None, :]
        # nb x nloc x nnei x nnei
        h2h2t = torch.matmul(h2, torch.transpose(h2, -1, -2)) / 3.0**0.5
        # nb x nloc x nh x nnei x nnei
        ret = attnw * h2h2t[:, :, None, :, :]
        # ret = torch.softmax(g2qk, dim=-1)
        # nb x nloc x nnei x nnei x nh
        ret = torch.permute(ret, (0, 1, 3, 4, 2))
        return ret

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "@class": "Atten2Map",
            "@version": 1,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "head_num": self.head_num,
            "has_gate": self.has_gate,
            "smooth": self.smooth,
            "attnw_shift": self.attnw_shift,
            "precision": self.precision,
            "mapqk": self.mapqk.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Atten2Map":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        mapqk = data.pop("mapqk")
        obj = cls(**data)
        obj.mapqk = MLPLayer.deserialize(mapqk)
        return obj


class Atten2MultiHeadApply(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_num: int,
        precision: str = "float64",
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.head_num = head_num
        self.mapv = MLPLayer(
            input_dim,
            input_dim * head_num,
            bias=False,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        self.head_map = MLPLayer(
            input_dim * head_num,
            input_dim,
            precision=precision,
            seed=child_seed(seed, 1),
        )
        self.precision = precision

    def forward(
        self,
        AA: torch.Tensor,  # nf x nloc x nnei x nnei x nh
        g2: torch.Tensor,  # nf x nloc x nnei x ng2
    ) -> torch.Tensor:
        nf, nloc, nnei, ng2 = g2.shape
        nh = self.head_num
        # nf x nloc x nnei x ng2 x nh
        g2v = self.mapv(g2).view(nf, nloc, nnei, ng2, nh)
        # nf x nloc x nh x nnei x ng2
        g2v = torch.permute(g2v, (0, 1, 4, 2, 3))
        # g2v = torch.nn.functional.normalize(g2v, dim=-1)
        # nf x nloc x nh x nnei x nnei
        AA = torch.permute(AA, (0, 1, 4, 2, 3))
        # nf x nloc x nh x nnei x ng2
        ret = torch.matmul(AA, g2v)
        # nf x nloc x nnei x ng2 x nh
        ret = torch.permute(ret, (0, 1, 3, 4, 2)).reshape(nf, nloc, nnei, (ng2 * nh))
        # nf x nloc x nnei x ng2
        return self.head_map(ret)

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "@class": "Atten2MultiHeadApply",
            "@version": 1,
            "input_dim": self.input_dim,
            "head_num": self.head_num,
            "precision": self.precision,
            "mapv": self.mapv.serialize(),
            "head_map": self.head_map.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Atten2MultiHeadApply":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        mapv = data.pop("mapv")
        head_map = data.pop("head_map")
        obj = cls(**data)
        obj.mapv = MLPLayer.deserialize(mapv)
        obj.head_map = MLPLayer.deserialize(head_map)
        return obj


class Atten2EquiVarApply(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_num: int,
        precision: str = "float64",
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.head_num = head_num
        self.head_map = MLPLayer(
            head_num, 1, bias=False, precision=precision, seed=seed
        )
        self.precision = precision

    def forward(
        self,
        AA: torch.Tensor,  # nf x nloc x nnei x nnei x nh
        h2: torch.Tensor,  # nf x nloc x nnei x 3
    ) -> torch.Tensor:
        nf, nloc, nnei, _ = h2.shape
        nh = self.head_num
        # nf x nloc x nh x nnei x nnei
        AA = torch.permute(AA, (0, 1, 4, 2, 3))
        h2m = torch.unsqueeze(h2, dim=2)
        # nf x nloc x nh x nnei x 3
        h2m = torch.tile(h2m, [1, 1, nh, 1, 1])
        # nf x nloc x nh x nnei x 3
        ret = torch.matmul(AA, h2m)
        # nf x nloc x nnei x 3 x nh
        ret = torch.permute(ret, (0, 1, 3, 4, 2)).view(nf, nloc, nnei, 3, nh)
        # nf x nloc x nnei x 3
        return torch.squeeze(self.head_map(ret), dim=-1)

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "@class": "Atten2EquiVarApply",
            "@version": 1,
            "input_dim": self.input_dim,
            "head_num": self.head_num,
            "precision": self.precision,
            "head_map": self.head_map.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Atten2EquiVarApply":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        head_map = data.pop("head_map")
        obj = cls(**data)
        obj.head_map = MLPLayer.deserialize(head_map)
        return obj


class RepformerLayer(torch.nn.Module):
    def __init__(
        self,
        rcut,
        rcut_smth,
        sel: int,
        ntypes: int,
        g1_dim=128,
        g2_dim=16,
        axis_neuron: int = 4,
        update_chnnl_2: bool = True,  # deprecated
        update_g1_has_conv: bool = True,
        update_g1_has_drrd: bool = True,
        update_g1_has_grrg: bool = True,
        update_g1_has_attn: bool = True,  # deprecated
        update_g2_has_g1g1: bool = True,  # deprecated
        update_g2_has_attn: bool = True,
        update_h2: bool = False,  # deprecated
        attn1_hidden: int = 64,
        attn1_nhead: int = 4,
        attn2_hidden: int = 16,
        attn2_nhead: int = 4,
        attn2_has_gate: bool = False,
        activation_function: str = "tanh",
        update_style: str = "res_avg",
        update_residual: float = 0.001,
        update_residual_init: str = "norm",
        smooth: bool = True,
        precision: str = "float64",
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
        use_sqrt_nnei: bool = True,
        g1_out_conv: bool = True,  # deprecated
        g1_out_mlp: bool = True,  # deprecated
        has_angle: bool = True,  # angle
        update_a_has_g1: bool = True,
        update_a_has_g2: bool = True,
        update_g2_has_a: bool = True,
        update_g1_has_edge: bool = True,
        update_g2_has_edge: bool = True,
        a_dim: int = 64,
        num_a: int = 9,
        a_rcut: float = 4.0,
        a_sel: int = 40,
        angle_use_self_g2_padding: bool = True,
        use_undirect_g2: bool = False,
        use_undirect_a: bool = False,
        update_g1_bidirect: bool = False,
        pipeline_update: bool = False,
        pre_ln: bool = False,
        g1_mess_mulmlp: bool = False,
        update_g2_has_ar: bool = False,
        update_g1_has_ar: bool = False,
        update_g2_has_arra: bool = False,
        compress_a: int = 0,
        g1_bi_message: bool = False,
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        super().__init__()
        self.epsilon = 1e-4  # protection of 1./nnei
        self.rcut = float(rcut)
        self.rcut_smth = float(rcut_smth)
        self.ntypes = ntypes
        sel = [sel] if isinstance(sel, int) else sel
        self.nnei = sum(sel)
        assert len(sel) == 1
        self.sel = sel
        self.sec = self.sel
        self.axis_neuron = axis_neuron
        self.activation_function = activation_function
        self.act = ActivationFn(activation_function)
        self.update_g1_has_grrg = update_g1_has_grrg
        self.update_g1_has_drrd = update_g1_has_drrd
        self.update_g1_has_conv = update_g1_has_conv
        self.update_g1_has_attn = update_g1_has_attn
        self.update_chnnl_2 = update_chnnl_2
        self.update_g2_has_g1g1 = update_g2_has_g1g1 if self.update_chnnl_2 else False
        self.update_g2_has_attn = update_g2_has_attn if self.update_chnnl_2 else False
        self.update_h2 = update_h2 if self.update_chnnl_2 else False
        del update_g2_has_g1g1, update_g2_has_attn, update_h2
        self.attn1_hidden = attn1_hidden
        self.attn1_nhead = attn1_nhead
        self.attn2_hidden = attn2_hidden
        self.attn2_nhead = attn2_nhead
        self.attn2_has_gate = attn2_has_gate
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.smooth = smooth
        self.g1_dim = g1_dim
        self.g2_dim = g2_dim
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.pre_ln = pre_ln
        self.precision = precision
        self.seed = seed
        self.use_sqrt_nnei = use_sqrt_nnei
        self.g1_out_conv = g1_out_conv
        self.g1_out_mlp = g1_out_mlp
        # angle information
        self.has_angle = has_angle
        self.update_a_has_g1 = update_a_has_g1
        self.update_a_has_g2 = update_a_has_g2
        self.update_g2_has_a = update_g2_has_a
        self.update_g1_has_edge = update_g1_has_edge
        self.update_g2_has_edge = update_g2_has_edge
        self.a_dim = a_dim
        self.num_a = num_a
        self.a_rcut = a_rcut
        self.a_sel = a_sel
        self.angle_use_self_g2_padding = angle_use_self_g2_padding
        self.use_undirect_g2 = use_undirect_g2
        self.use_undirect_a = use_undirect_a
        self.update_g1_bidirect = update_g1_bidirect
        self.pipeline_update = pipeline_update
        self.g1_mess_mulmlp = g1_mess_mulmlp
        self.update_g2_has_ar = update_g2_has_ar
        self.update_g1_has_ar = update_g1_has_ar
        self.update_g2_has_arra = update_g2_has_arra
        self.compress_a = compress_a
        self.g1_bi_message = g1_bi_message
        self.prec = PRECISION_DICT[precision]
        self.g1_layernorm = None
        self.g2_layernorm = None
        self.angle_layernorm = None

        assert update_residual_init in [
            "norm",
            "const",
        ], "'update_residual_init' only support 'norm' or 'const'!"

        if self.pre_ln:
            assert self.update_style == "res_layer"

        if self.update_style == "res_layer":
            self.g1_layernorm = nn.LayerNorm(
                self.g1_dim,
                device=env.DEVICE,
                dtype=self.prec,
                elementwise_affine=trainable_ln,
            )
            self.g2_layernorm = nn.LayerNorm(
                self.g2_dim,
                device=env.DEVICE,
                dtype=self.prec,
                elementwise_affine=trainable_ln,
            )

        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.g1_residual = []
        self.g2_residual = []
        self.h2_residual = []
        self.a_residual = []
        self.linear2 = None
        self.proj_g1g2 = None
        self.proj_g1g1g2 = None
        self.attn2g_map = None
        self.attn2_mh_apply = None
        self.attn2_lm = None
        self.attn2_ev_apply = None
        self.loc_attn = None
        self.g1_edge_linear1 = None
        self.g1_edge_linear2 = None
        self.edge_info_dim = self.g1_dim * 2 + self.g2_dim

        # g1 self mlp
        self.g1_self_mlp = MLPLayer(
            g1_dim,
            g1_dim,
            precision=precision,
            seed=child_seed(seed, 15),
        )
        if self.update_style == "res_residual":
            self.g1_residual.append(
                get_residual(
                    g1_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 16),
                )
            )

        # g1 conv
        if self.update_g1_has_conv:
            self.proj_g1g2 = MLPLayer(
                g2_dim,
                g1_dim,
                bias=False,
                precision=precision,
                seed=child_seed(seed, 4),
            )
            if self.update_style == "res_residual":
                self.g1_residual.append(
                    get_residual(
                        g1_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 17),
                    )
                )

        self.g1_in_dim = self.cal_1_dim(g1_dim, g2_dim, self.axis_neuron)
        if self.g1_in_dim > 0:
            # g1 concat mlp
            self.linear1 = MLPLayer(
                self.g1_in_dim,
                g1_dim,
                precision=precision,
                seed=child_seed(seed, 1),
            )
            if self.update_style == "res_residual":
                self.g1_residual.append(
                    get_residual(
                        g1_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 0),
                    )
                )
        else:
            self.linear1 = None

        # g1 edge
        if self.update_g1_has_edge:
            self.g1_edge_linear1 = MLPLayer(
                self.edge_info_dim,
                g1_dim,
                precision=precision,
                seed=child_seed(seed, 11),
            )  # need act # receive
            self.g1_edge_linear2 = MLPLayer(
                g1_dim,
                g1_dim,
                precision=precision,
                seed=child_seed(seed, 12),
            )  # need act
            if self.update_g1_bidirect:
                self.g1_edge_linear_send = MLPLayer(
                    self.edge_info_dim,
                    g1_dim,
                    precision=precision,
                    seed=child_seed(seed, 20),
                )  # need act # send
            else:
                self.g1_edge_linear_send = None

            if self.g1_bi_message:
                self.g1_edge_linear_receive_head2 = MLPLayer(
                    self.edge_info_dim,
                    g1_dim,
                    precision=precision,
                    seed=child_seed(seed, 22),
                )  # need act # receive 2
            else:
                self.g1_edge_linear_receive_head2 = None

            if self.update_style == "res_residual":
                self.g1_residual.append(
                    get_residual(
                        g1_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 13),
                    )
                )
                if self.update_g1_bidirect:
                    self.g1_residual.append(
                        get_residual(
                            g1_dim,
                            self.update_residual,
                            self.update_residual_init,
                            precision=precision,
                            seed=child_seed(seed, 21),
                        )
                    )
                if self.g1_bi_message:
                    self.g1_residual.append(
                        get_residual(
                            g1_dim,
                            self.update_residual,
                            self.update_residual_init,
                            precision=precision,
                            seed=child_seed(seed, 23),
                        )
                    )

        # angle for g1
        if self.has_angle and self.update_g1_has_ar:
            self.g1_angle_linear = MLPLayer(
                self.a_dim,
                g1_dim,
                precision=precision,
                seed=child_seed(seed, 13),
            )  # need act
            if self.update_style == "res_residual":
                self.g1_residual.append(
                    get_residual(
                        g1_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 14),
                    )
                )
        else:
            self.g1_angle_linear = None

        if not self.update_g2_has_edge:
            # g2 self mlp
            self.linear2 = MLPLayer(
                g2_dim,
                g2_dim,
                precision=precision,
                seed=child_seed(seed, 2),
            )
        else:
            # g2 edge
            self.linear2 = MLPLayer(
                self.edge_info_dim,
                g2_dim,
                precision=precision,
                seed=child_seed(seed, 2),
            )
        if self.update_style == "res_residual":
            self.g2_residual.append(
                get_residual(
                    g2_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 3),
                )
            )

        # g2 attn
        if self.update_g2_has_attn:
            self.attn2g_map = Atten2Map(
                g2_dim,
                attn2_hidden,
                attn2_nhead,
                attn2_has_gate,
                self.smooth,
                precision=precision,
                seed=child_seed(seed, 7),
            )
            self.attn2_mh_apply = Atten2MultiHeadApply(
                g2_dim, attn2_nhead, precision=precision, seed=child_seed(seed, 8)
            )
            self.attn2_lm = LayerNorm(
                g2_dim,
                eps=ln_eps,
                trainable=trainable_ln,
                precision=precision,
                seed=child_seed(seed, 9),
            )
            if self.update_style == "res_residual":
                self.g2_residual.append(
                    get_residual(
                        g2_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 10),
                    )
                )

        # angle for g2
        if self.has_angle and self.update_g2_has_ar:
            self.g2_angle_linear_ar = MLPLayer(
                self.a_dim,
                g2_dim,
                precision=precision,
                seed=child_seed(seed, 21),
            )  # need act
            if self.update_style == "res_residual":
                self.g2_residual.append(
                    get_residual(
                        g2_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 22),
                    )
                )
        else:
            self.g2_angle_linear_ar = None

        if self.has_angle:
            if self.update_style == "res_layer":
                self.angle_layernorm = nn.LayerNorm(
                    self.a_dim,
                    device=env.DEVICE,
                    dtype=self.prec,
                    elementwise_affine=trainable_ln,
                )
            angle_seed = 20
            self.angle_dim = self.a_dim
            if self.compress_a == 0:
                self.angle_dim += self.g1_dim if self.update_a_has_g1 else 0
                self.angle_dim += 2 * self.g2_dim if self.update_a_has_g2 else 0
                self.compress_n_linear = None
                self.compress_e_linear = None
            else:
                self.angle_dim += (
                    self.a_dim // self.compress_a if self.update_a_has_g1 else 0
                )
                self.angle_dim += (
                    self.a_dim // self.compress_a if self.update_a_has_g2 else 0
                )
                self.compress_n_linear = MLPLayer(
                    self.g1_dim,
                    self.a_dim // self.compress_a,
                    precision=precision,
                    bias=False,
                    seed=child_seed(seed, angle_seed + 3),
                )
                self.compress_e_linear = MLPLayer(
                    self.g2_dim,
                    self.a_dim // (2 * self.compress_a),
                    precision=precision,
                    bias=False,
                    seed=child_seed(seed, angle_seed + 2),
                )

            self.g2_angle_dim = self.angle_dim
            self.angle_linear = MLPLayer(
                self.angle_dim,
                self.a_dim,
                precision=precision,
                seed=child_seed(seed, angle_seed + 1),
            )  # need act
            if self.update_style == "res_residual":
                self.a_residual.append(
                    get_residual(
                        self.a_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, angle_seed + 2),
                    )
                )

            self.g2_angle_linear1 = MLPLayer(
                self.g2_angle_dim,
                self.g2_dim,
                precision=precision,
                seed=child_seed(seed, angle_seed + 3),
            )  # need act
            self.g2_angle_linear2 = MLPLayer(
                self.g2_dim,
                self.g2_dim,
                precision=precision,
                seed=child_seed(seed, angle_seed + 4),
            )
            if self.update_style == "res_residual":
                self.g2_residual.append(
                    get_residual(
                        self.g2_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, angle_seed + 5),
                    )
                )

        else:
            self.angle_linear = None
            self.g2_angle_linear1 = None
            self.g2_angle_linear2 = None
            self.angle_dim = 0
            self.g2_angle_dim = 0

        self.g1_residual = nn.ParameterList(self.g1_residual)
        self.g2_residual = nn.ParameterList(self.g2_residual)
        self.h2_residual = nn.ParameterList(self.h2_residual)
        self.a_residual = nn.ParameterList(self.a_residual)

    def cal_1_dim(self, g1d: int, g2d: int, ax: int) -> int:
        ret = 0
        if self.update_g1_has_grrg:
            ret += g2d * ax
        if self.update_g1_has_drrd:
            ret += g1d * ax
        return ret

    def _update_h2(
        self,
        h2: torch.Tensor,
        attn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the attention weights update for pair-wise equivariant rep.

        Parameters
        ----------
        h2
            Pair-wise equivariant rep tensors, with shape nf x nloc x nnei x 3.
        attn
            Attention weights from g2 attention, with shape nf x nloc x nnei x nnei x nh2.
        """
        assert self.attn2_ev_apply is not None
        # nf x nloc x nnei x nh2
        h2_1 = self.attn2_ev_apply(attn, h2)
        return h2_1

    def _update_g1_conv(
        self,
        gg1: torch.Tensor,
        g2: torch.Tensor,
        nlist_mask: torch.Tensor,
        sw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the convolution update for atomic invariant rep.

        Parameters
        ----------
        gg1
            Neighbor-wise atomic invariant rep, with shape nb x nloc x nnei x ng1.
        g2
            Pair invariant rep, with shape nb x nloc x nnei x ng2.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.
        """
        assert self.proj_g1g2 is not None
        nb, nloc, nnei, _ = g2.shape
        ng1 = gg1.shape[-1]
        ng2 = g2.shape[-1]
        gg1 = gg1.view(nb, nloc, nnei, ng1)
        # nb x nloc x nnei x ng2/ng1
        gg1 = _apply_nlist_mask(gg1, nlist_mask)
        if not self.smooth:
            # normalized by number of neighbors, not smooth
            # nb x nloc x 1
            # must use type_as here to convert bool to float, otherwise there will be numerical difference from numpy
            invnnei = 1.0 / (
                self.epsilon + torch.sum(nlist_mask.type_as(gg1), dim=-1)
            ).unsqueeze(-1)
        else:
            gg1 = _apply_switch(gg1, sw)
            invnnei = (1.0 / float(nnei)) * torch.ones(
                (nb, nloc, 1), dtype=gg1.dtype, device=gg1.device
            )
        g2 = self.proj_g1g2(g2).view(nb, nloc, nnei, ng1)
        # nb x nloc x ng1
        g1_11 = torch.sum(g2 * gg1, dim=2) * invnnei
        return g1_11

    @staticmethod
    def _cal_hg(
        g2: torch.Tensor,
        h2: torch.Tensor,
        nlist_mask: torch.Tensor,
        sw: torch.Tensor,
        smooth: bool = True,
        epsilon: float = 1e-4,
        use_sqrt_nnei: bool = True,
    ) -> torch.Tensor:
        """
        Calculate the transposed rotation matrix.

        Parameters
        ----------
        g2
            Neighbor-wise/Pair-wise invariant rep tensors, with shape nb x nloc x nnei x ng2.
        h2
            Neighbor-wise/Pair-wise equivariant rep tensors, with shape nb x nloc x nnei x 3.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.
        smooth
            Whether to use smoothness in processes such as attention weights calculation.
        epsilon
            Protection of 1./nnei.

        Returns
        -------
        hg
            The transposed rotation matrix, with shape nb x nloc x 3 x ng2.
        """
        # g2:  nb x nloc x nnei x ng2
        # h2:  nb x nloc x nnei x 3
        # msk: nb x nloc x nnei
        nb, nloc, nnei, _ = g2.shape
        ng2 = g2.shape[-1]
        # nb x nloc x nnei x ng2
        g2 = _apply_nlist_mask(g2, nlist_mask)
        if not smooth:
            # nb x nloc
            # must use type_as here to convert bool to float, otherwise there will be numerical difference from numpy
            if not use_sqrt_nnei:
                invnnei = 1.0 / (epsilon + torch.sum(nlist_mask.type_as(g2), dim=-1))
            else:
                invnnei = 1.0 / (
                    epsilon + torch.sqrt(torch.sum(nlist_mask.type_as(g2), dim=-1))
                )
            # nb x nloc x 1 x 1
            invnnei = invnnei.unsqueeze(-1).unsqueeze(-1)
        else:
            g2 = _apply_switch(g2, sw)
            if not use_sqrt_nnei:
                invnnei = (1.0 / float(nnei)) * torch.ones(
                    (nb, nloc, 1, 1), dtype=g2.dtype, device=g2.device
                )
            else:
                invnnei = torch.rsqrt(
                    float(nnei)
                    * torch.ones((nb, nloc, 1, 1), dtype=g2.dtype, device=g2.device)
                )
        # nb x nloc x 3 x ng2
        h2g2 = torch.matmul(torch.transpose(h2, -1, -2), g2) * invnnei
        return h2g2

    @staticmethod
    def _cal_grrg(h2g2: torch.Tensor, axis_neuron: int) -> torch.Tensor:
        """
        Calculate the atomic invariant rep.

        Parameters
        ----------
        h2g2
            The transposed rotation matrix, with shape nb x nloc x 3 x ng2.
        axis_neuron
            Size of the submatrix.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x ng2)
        """
        # nb x nloc x 3 x ng2
        nb, nloc, _, ng2 = h2g2.shape
        # nb x nloc x 3 x axis
        h2g2m = torch.split(h2g2, axis_neuron, dim=-1)[0]
        # nb x nloc x axis x ng2
        g1_13 = torch.matmul(torch.transpose(h2g2m, -1, -2), h2g2) / (3.0**1)
        # nb x nloc x (axisxng2)
        g1_13 = g1_13.view(nb, nloc, axis_neuron * ng2)
        return g1_13

    def symmetrization_op(
        self,
        g2: torch.Tensor,
        h2: torch.Tensor,
        nlist_mask: torch.Tensor,
        sw: torch.Tensor,
        axis_neuron: int,
        smooth: bool = True,
        epsilon: float = 1e-4,
    ) -> torch.Tensor:
        """
        Symmetrization operator to obtain atomic invariant rep.

        Parameters
        ----------
        g2
            Neighbor-wise/Pair-wise invariant rep tensors, with shape nb x nloc x nnei x ng2.
        h2
            Neighbor-wise/Pair-wise equivariant rep tensors, with shape nb x nloc x nnei x 3.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nb x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nb x nloc x nnei.
        axis_neuron
            Size of the submatrix.
        smooth
            Whether to use smoothness in processes such as attention weights calculation.
        epsilon
            Protection of 1./nnei.

        Returns
        -------
        grrg
            Atomic invariant rep, with shape nb x nloc x (axis_neuron x ng2)
        """
        # g2:  nb x nloc x nnei x ng2
        # h2:  nb x nloc x nnei x 3
        # msk: nb x nloc x nnei
        nb, nloc, nnei, _ = g2.shape
        # nb x nloc x 3 x ng2
        h2g2 = self._cal_hg(
            g2,
            h2,
            nlist_mask,
            sw,
            smooth=smooth,
            epsilon=epsilon,
            use_sqrt_nnei=self.use_sqrt_nnei,
        )
        # nb x nloc x (axisxng2)
        g1_13 = self._cal_grrg(h2g2, axis_neuron)
        return g1_13

    def forward(
        self,
        g1_ext: torch.Tensor,  # nf x nall x ng1
        g2: torch.Tensor,  # nf x nloc x nnei x ng2
        h2: torch.Tensor,  # nf x nloc x nnei x 3
        angle_embed: torch.Tensor,  # nf x nloc x a_nnei x a_nnei x a_dim
        nlist: torch.Tensor,  # nf x nloc x nnei
        nlist_mask: torch.Tensor,  # nf x nloc x nnei
        sw: torch.Tensor,  # switch func, nf x nloc x nnei
        angle_nlist: torch.Tensor,  # nf x nloc x a_nnei
        angle_nlist_mask: torch.Tensor,  # nf x nloc x a_nnei
        angle_sw: torch.Tensor,  # switch func, nf x nloc x a_nnei
        nlist_loc: Optional[torch.Tensor] = None,
        cosine_ij: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        g1_ext : nf x nall x ng1         extended single-atom channel
        g2 : nf x nloc x nnei x ng2  pair-atom channel, invariant
        h2 : nf x nloc x nnei x 3    pair-atom channel, equivariant
        nlist : nf x nloc x nnei        neighbor list (padded neis are set to 0)
        nlist_mask : nf x nloc x nnei   masks of the neighbor list. real nei 1 otherwise 0
        sw : nf x nloc x nnei        switch function

        Returns
        -------
        g1:     nf x nloc x ng1         updated single-atom channel
        g2:     nf x nloc x nnei x ng2  updated pair-atom channel, invariant
        h2:     nf x nloc x nnei x 3    updated pair-atom channel, equivariant
        """
        cal_gg1 = (
            self.update_g1_has_drrd
            or self.update_g1_has_conv
            or self.update_g1_has_attn
            or self.update_g2_has_g1g1
            or self.update_g2_has_edge
            or self.update_g1_has_edge
        )

        nb, nloc, nnei, _ = g2.shape
        nall = g1_ext.shape[1]
        g1, _ = torch.split(g1_ext, [nloc, nall - nloc], dim=1)
        assert (nb, nloc) == g1.shape[:2]
        assert (nb, nloc, nnei) == h2.shape[:3]

        g2_update: list[torch.Tensor] = [g2]
        h2_update: list[torch.Tensor] = [h2]
        g1_update: list[torch.Tensor] = [g1]
        g1_mlp: list[torch.Tensor] = []

        # angle
        a_update: list[torch.Tensor] = [angle_embed]

        if self.pre_ln:
            assert self.g1_layernorm is not None
            assert self.g2_layernorm is not None
            g1 = self.g1_layernorm(g1)
            g2 = self.g2_layernorm(g2)

        # g1 self mlp
        g1_self_mlp = self.act(self.g1_self_mlp(g1))
        g1_update.append(g1_self_mlp)

        if cal_gg1:
            gg1 = _make_nei_g1(g1_ext, nlist)
        else:
            gg1 = None

        # g1 conv
        if self.update_g1_has_conv:
            assert gg1 is not None
            g1_conv = self._update_g1_conv(gg1, g2, nlist_mask, sw)
            g1_update.append(g1_conv)

        # g1 concat mlp
        if self.update_g1_has_grrg:
            g1_mlp.append(
                self.symmetrization_op(
                    g2,
                    h2,
                    nlist_mask,
                    sw,
                    self.axis_neuron,
                    smooth=self.smooth,
                    epsilon=self.epsilon,
                )
            )

        if self.update_g1_has_drrd:
            assert gg1 is not None
            g1_mlp.append(
                self.symmetrization_op(
                    gg1,
                    h2,
                    nlist_mask,
                    sw,
                    self.axis_neuron,
                    smooth=self.smooth,
                    epsilon=self.epsilon,
                )
            )

        if self.g1_in_dim > 0:
            assert self.linear1 is not None
            # nb x nloc x [ng1+ng2+(axisxng2)+(axisxng1)]
            #                  conv   grrg      drrd
            g1_1 = self.act(self.linear1(torch.cat(g1_mlp, dim=-1)))
            g1_update.append(g1_1)

        if self.update_g2_has_edge or self.update_g1_has_edge:
            assert gg1 is not None
            edge_info = torch.cat(
                [torch.tile(g1.unsqueeze(-2), [1, 1, self.nnei, 1]), gg1, g2], dim=-1
            )
        else:
            edge_info = None

        # g1 edge update
        if self.update_g1_has_edge:
            assert edge_info is not None
            assert self.g1_edge_linear1 is not None
            assert self.g1_edge_linear2 is not None
            # nb x nloc x nnei x ng1
            # receive
            g1_edge_info = self.act(self.g1_edge_linear1(edge_info))
            if self.g1_mess_mulmlp:
                g1_edge_info = self.act(self.g1_edge_linear2(g1_edge_info))
            g1_edge_info = g1_edge_info * sw.unsqueeze(-1)
            g1_edge_update = torch.sum(g1_edge_info, dim=-2) / self.nnei
            g1_update.append(g1_edge_update)

            if self.g1_bi_message:
                # reveive multihead
                assert self.g1_edge_linear_receive_head2 is not None
                g1_edge_info_reveive2 = self.act(
                    self.g1_edge_linear_receive_head2(edge_info)
                )
                g1_edge_info_reveive2 = g1_edge_info_reveive2 * sw.unsqueeze(-1)
                g1_edge_update_reveive2 = (
                    torch.sum(g1_edge_info_reveive2, dim=-2) / self.nnei
                )
                g1_update.append(g1_edge_update_reveive2)

            if self.update_g1_bidirect:
                # send message
                assert self.g1_edge_linear_send is not None
                # nb x nloc x nnei x ng1
                g1_edge_info_send = self.act(
                    self.g1_edge_linear_send(edge_info)
                ) * sw.unsqueeze(-1)
                assert nlist_loc is not None
                # nb x (nloc+1) x nnei x ng1
                scattered_message = torch.zeros(
                    size=[nb, nloc + 1, nnei, self.g1_dim],
                    device=g1_edge_info_send.device,
                    dtype=g1_edge_info_send.dtype,
                )
                # nb x nloc x nnei x ng1
                scatter_index = nlist_loc.unsqueeze(-1).expand(-1, -1, -1, self.g1_dim)
                # nb x nloc x nnei x ng1
                scattered_message = torch.scatter_reduce(
                    scattered_message,
                    dim=1,
                    index=scatter_index,
                    src=g1_edge_info_send,
                    reduce="sum",
                )[:, :-1, :, :]
                # nb x nloc x ng1
                g1_edge_update_send = torch.sum(scattered_message, dim=-2) / self.nnei
                g1_update.append(g1_edge_update_send)

        if self.has_angle and (self.update_g1_has_ar or self.update_g2_has_ar):
            assert cosine_ij is not None
            assert angle_embed is not None
            # nb x nloc x a_nnei x a_nnei x a
            angle_ar = cosine_ij.unsqueeze(-1) * angle_embed
        else:
            angle_ar = None

        # angle for g1
        if self.has_angle and self.update_g1_has_ar:
            assert self.g1_angle_linear is not None
            assert angle_ar is not None
            # nb x nloc x g1_dim
            g1_ar = self.act(self.g1_angle_linear(angle_ar)).sum(-2).sum(-2) / (
                float(self.a_sel) * float(self.a_sel)
            )
            g1_update.append(g1_ar)

        # update g1 for pipeline_update
        g1_new = self.list_update(g1_update, "g1")
        if self.pipeline_update:
            # update all g1 related tensor
            assert gg1 is not None
            assert nlist_loc is not None
            # new gg1
            gg1 = _make_nei_g1(g1_new, torch.where(nlist_mask, nlist_loc, 0))
            # g1
            g1 = g1_new

        assert self.linear2 is not None
        if not self.update_g2_has_edge:
            # g2 self mlp
            # nb x nloc x nnei x ng2
            g2_1 = self.act(self.linear2(g2))
            g2_update.append(g2_1)
        else:
            # g2 edge update
            assert edge_info is not None
            g2_edge_info = self.act(self.linear2(edge_info))
            if self.use_undirect_g2:
                assert gg1 is not None
                edge_info_2 = torch.cat(
                    [gg1, torch.tile(g1.unsqueeze(-2), [1, 1, self.nnei, 1]), g2],
                    dim=-1,
                )
                g2_edge_info_2 = self.act(self.linear2(edge_info_2))
                g2_edge_info = (g2_edge_info + g2_edge_info_2) / 2
            g2_update.append(g2_edge_info)

        # g2 attn
        if self.update_g2_has_attn:
            # gated_attention(g2, h2)
            assert self.attn2g_map is not None
            # nb x nloc x nnei x nnei x nh
            AAg = self.attn2g_map(g2, h2, nlist_mask, sw)
            assert self.attn2_mh_apply is not None
            assert self.attn2_lm is not None
            # nb x nloc x nnei x ng2
            g2_2 = self.attn2_mh_apply(AAg, g2)
            g2_2 = self.attn2_lm(g2_2)
            g2_update.append(g2_2)

        # angle for g2
        if self.has_angle and self.update_g2_has_ar:
            assert self.g2_angle_linear_ar is not None
            assert angle_ar is not None
            # nb x nloc x a_nnei x g2_dim
            g2_ar = self.act(self.g2_angle_linear_ar(angle_ar)).sum(-2) / float(
                self.a_sel
            )
            # nb x nloc x nnei x g2
            padding_g2_ar = torch.concat(
                [
                    g2_ar,
                    torch.zeros(
                        [nb, nloc, self.nnei - self.a_sel, self.g2_dim],
                        dtype=g2.dtype,
                        device=g2.device,
                    ),
                ],
                dim=2,
            )
            if self.angle_use_self_g2_padding:
                full_mask = torch.concat(
                    [
                        angle_nlist_mask,
                        torch.zeros(
                            [nb, nloc, self.nnei - self.a_sel],
                            dtype=angle_nlist_mask.dtype,
                            device=angle_nlist_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                padding_g2_ar = torch.where(full_mask.unsqueeze(-1), padding_g2_ar, g2)
            g2_update.append(padding_g2_ar)

        if self.has_angle:
            if self.pre_ln:
                assert self.angle_layernorm is not None
                angle_embed = self.angle_layernorm(angle_embed)
            assert self.angle_linear is not None
            assert self.g2_angle_linear1 is not None
            assert self.g2_angle_linear2 is not None
            assert (
                not self.use_undirect_g2
            ), "use angle update can not use undirect g2 yet"
            if self.compress_a != 0:
                assert self.compress_n_linear is not None
                assert self.compress_e_linear is not None
                # nb x nloc x a/c
                g1_for_angle = self.compress_n_linear(g1)
                # nb x nloc x nnei x a/2c
                g2_for_angle = self.compress_e_linear(g2)
            else:
                g1_for_angle = g1
                g2_for_angle = g2

            # nb x nloc x a_nnei x a_nnei x (g1 or a/c)
            g1_angle_embed = torch.tile(
                g1_for_angle.unsqueeze(2).unsqueeze(2),
                (1, 1, self.a_sel, self.a_sel, 1),
            )
            # nb x nloc x a_nnei x (g2 or a/2c)
            g2_angle = g2_for_angle[:, :, : self.a_sel, :]
            # nb x nloc x a_nnei x (g2 or a/2c)
            g2_angle = torch.where(angle_nlist_mask.unsqueeze(-1), g2_angle, 0.0)
            # nb x nloc x (a_nnei) x a_nnei x (g2 or a/2c)
            g2_angle_i = torch.tile(g2_angle.unsqueeze(2), (1, 1, self.a_sel, 1, 1))
            # nb x nloc x a_nnei x (a_nnei) x (g2 or a/2c)
            g2_angle_j = torch.tile(g2_angle.unsqueeze(3), (1, 1, 1, self.a_sel, 1))
            # nb x nloc x a_nnei x a_nnei x (g2 + g2 or a/c)
            g2_angle_embed = torch.cat([g2_angle_i, g2_angle_j], dim=-1)

            # angle for g2:
            updated_g2_angle_list = [angle_embed]
            # nb x nloc x a_nnei x a_nnei x (a + g1 + g2*2) or (a + a/c + a/c)
            if self.update_a_has_g1:
                updated_g2_angle_list += [g1_angle_embed]
            if self.update_a_has_g2:
                updated_g2_angle_list += [g2_angle_embed]
            updated_g2_angle = torch.cat(updated_g2_angle_list, dim=-1)
            # nb x nloc x a_nnei x a_nnei x g2
            updated_angle_g2 = self.act(self.g2_angle_linear1(updated_g2_angle))
            # nb x nloc x a_nnei x a_nnei x g2
            weighted_updated_angle_g2 = (
                updated_angle_g2
                * angle_sw[:, :, :, None, None]
                * angle_sw[:, :, None, :, None]
            )
            # nb x nloc x a_nnei x g2
            reduced_updated_angle_g2 = torch.sum(weighted_updated_angle_g2, dim=-2) / (
                self.a_sel**0.5
            )
            # nb x nloc x nnei x g2
            padding_updated_angle_g2 = torch.concat(
                [
                    reduced_updated_angle_g2,
                    torch.zeros(
                        [nb, nloc, self.nnei - self.a_sel, self.g2_dim],
                        dtype=g2.dtype,
                        device=g2.device,
                    ),
                ],
                dim=2,
            )
            if self.angle_use_self_g2_padding:
                full_mask = torch.concat(
                    [
                        angle_nlist_mask,
                        torch.zeros(
                            [nb, nloc, self.nnei - self.a_sel],
                            dtype=angle_nlist_mask.dtype,
                            device=angle_nlist_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                padding_updated_angle_g2 = torch.where(
                    full_mask.unsqueeze(-1), padding_updated_angle_g2, g2
                )
            g2_update.append(self.act(self.g2_angle_linear2(padding_updated_angle_g2)))

            # update g2 for pipeline_update
            g2_new = self.list_update(g2_update, "g2")
            if self.pipeline_update:
                g2 = g2_new

            # angle for angle
            if not self.pipeline_update and self.g2_angle_dim == self.angle_dim:
                updated_angle = updated_g2_angle
            else:
                # copy from above
                # use new g2
                if self.pipeline_update:
                    # nb x nloc x a_nnei x g2
                    g2_angle = g2[:, :, : self.a_sel, :]
                    # nb x nloc x a_nnei x g2
                    g2_angle = torch.where(
                        angle_nlist_mask.unsqueeze(-1), g2_angle, 0.0
                    )
                    # nb x nloc x (a_nnei) x a_nnei x g2
                    g2_angle_i = torch.tile(
                        g2_angle.unsqueeze(2), (1, 1, self.a_sel, 1, 1)
                    )
                    # nb x nloc x a_nnei x (a_nnei) x g2
                    g2_angle_j = torch.tile(
                        g2_angle.unsqueeze(3), (1, 1, 1, self.a_sel, 1)
                    )
                    # nb x nloc x a_nnei x a_nnei x (g2 + g2)
                    g2_angle_embed = torch.cat([g2_angle_i, g2_angle_j], dim=-1)
                updated_angle_list = [angle_embed]
                if self.update_a_has_g1:
                    updated_angle_list.append(g1_angle_embed)
                if self.update_a_has_g2:
                    updated_angle_list.append(g2_angle_embed)
                # nb x nloc x a_nnei x a_nnei x (a + g1 + g2*2)
                updated_angle = torch.cat(updated_angle_list, dim=-1)

            # nb x nloc x a_nnei x a_nnei x dim_a
            angle_message = self.act(self.angle_linear(updated_angle))
            if self.use_undirect_a:
                angle_message = (
                    angle_message + angle_message.permute(0, 1, 3, 2, 4)
                ) / 2
            # angle update
            a_update.append(angle_message)
        else:
            g2_new = self.list_update(g2_update, "g2")

        # update
        h2_new = self.list_update(h2_update, "h2")
        a_new = self.list_update(a_update, "a")
        return g1_new, g2_new, h2_new, a_new

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
    def list_update_res_layer(
        self, update_list: list[torch.Tensor], update_name: str = "g1"
    ) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        for ii in range(1, nitem):
            uu = uu + update_list[ii]
        if not self.pre_ln:
            if update_name == "g1":
                assert self.g1_layernorm is not None
                out = self.g1_layernorm(uu)
            elif update_name == "g2":
                assert self.g2_layernorm is not None
                out = self.g2_layernorm(uu)
            elif update_name == "h2":
                # not update h2
                out = uu
            elif update_name == "a":
                assert self.angle_layernorm is not None
                out = self.angle_layernorm(uu)
            else:
                raise NotImplementedError
        else:
            out = uu
        return out

    @torch.jit.export
    def list_update_res_residual(
        self, update_list: list[torch.Tensor], update_name: str = "g1"
    ) -> torch.Tensor:
        nitem = len(update_list)
        uu = update_list[0]
        # make jit happy
        if update_name == "g1":
            for ii, vv in enumerate(self.g1_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "g2":
            for ii, vv in enumerate(self.g2_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "h2":
            for ii, vv in enumerate(self.h2_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "a":
            for ii, vv in enumerate(self.a_residual):
                uu = uu + vv * update_list[ii + 1]
        else:
            raise NotImplementedError
        return uu

    @torch.jit.export
    def list_update(
        self, update_list: list[torch.Tensor], update_name: str = "g1"
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
            "@version": 2,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
            "ntypes": self.ntypes,
            "g1_dim": self.g1_dim,
            "g2_dim": self.g2_dim,
            "axis_neuron": self.axis_neuron,
            "update_chnnl_2": self.update_chnnl_2,
            "update_g1_has_conv": self.update_g1_has_conv,
            "update_g1_has_drrd": self.update_g1_has_drrd,
            "update_g1_has_grrg": self.update_g1_has_grrg,
            "update_g1_has_attn": self.update_g1_has_attn,
            "update_g2_has_g1g1": self.update_g2_has_g1g1,
            "update_g2_has_attn": self.update_g2_has_attn,
            "update_h2": self.update_h2,
            "attn1_hidden": self.attn1_hidden,
            "attn1_nhead": self.attn1_nhead,
            "attn2_hidden": self.attn2_hidden,
            "attn2_nhead": self.attn2_nhead,
            "attn2_has_gate": self.attn2_has_gate,
            "activation_function": self.activation_function,
            "update_style": self.update_style,
            "smooth": self.smooth,
            "precision": self.precision,
            "trainable_ln": self.trainable_ln,
            "use_sqrt_nnei": self.use_sqrt_nnei,
            "g1_out_conv": self.g1_out_conv,
            "g1_out_mlp": self.g1_out_mlp,
            "ln_eps": self.ln_eps,
            "linear1": self.linear1.serialize(),
        }
        if self.update_chnnl_2:
            data.update(
                {
                    "linear2": self.linear2.serialize(),
                }
            )
        if self.update_g1_has_conv:
            data.update(
                {
                    "proj_g1g2": self.proj_g1g2.serialize(),
                }
            )
        if self.update_g2_has_g1g1:
            data.update(
                {
                    "proj_g1g1g2": self.proj_g1g1g2.serialize(),
                }
            )
        if self.update_g2_has_attn or self.update_h2:
            data.update(
                {
                    "attn2g_map": self.attn2g_map.serialize(),
                }
            )
            if self.update_g2_has_attn:
                data.update(
                    {
                        "attn2_mh_apply": self.attn2_mh_apply.serialize(),
                        "attn2_lm": self.attn2_lm.serialize(),
                    }
                )

            if self.update_h2:
                data.update(
                    {
                        "attn2_ev_apply": self.attn2_ev_apply.serialize(),
                    }
                )
        if self.update_g1_has_attn:
            data.update(
                {
                    "loc_attn": self.loc_attn.serialize(),
                }
            )
        if self.g1_out_mlp:
            data.update(
                {
                    "g1_self_mlp": self.g1_self_mlp.serialize(),
                }
            )
        if self.update_style == "res_residual":
            data.update(
                {
                    "@variables": {
                        "g1_residual": [to_numpy_array(t) for t in self.g1_residual],
                        "g2_residual": [to_numpy_array(t) for t in self.g2_residual],
                        "h2_residual": [to_numpy_array(t) for t in self.h2_residual],
                    }
                }
            )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "RepformerLayer":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 2, 1)
        data.pop("@class")
        linear1 = data.pop("linear1")
        update_chnnl_2 = data["update_chnnl_2"]
        update_g1_has_conv = data["update_g1_has_conv"]
        update_g2_has_g1g1 = data["update_g2_has_g1g1"]
        update_g2_has_attn = data["update_g2_has_attn"]
        update_h2 = data["update_h2"]
        update_g1_has_attn = data["update_g1_has_attn"]
        update_style = data["update_style"]
        g1_out_mlp = data["g1_out_mlp"]

        linear2 = data.pop("linear2", None)
        proj_g1g2 = data.pop("proj_g1g2", None)
        proj_g1g1g2 = data.pop("proj_g1g1g2", None)
        attn2g_map = data.pop("attn2g_map", None)
        attn2_mh_apply = data.pop("attn2_mh_apply", None)
        attn2_lm = data.pop("attn2_lm", None)
        attn2_ev_apply = data.pop("attn2_ev_apply", None)
        loc_attn = data.pop("loc_attn", None)
        g1_self_mlp = data.pop("g1_self_mlp", None)
        variables = data.pop("@variables", {})
        g1_residual = variables.get("g1_residual", data.pop("g1_residual", []))
        g2_residual = variables.get("g2_residual", data.pop("g2_residual", []))
        h2_residual = variables.get("h2_residual", data.pop("h2_residual", []))

        obj = cls(**data)
        obj.linear1 = MLPLayer.deserialize(linear1)
        if update_chnnl_2:
            assert isinstance(linear2, dict)
            obj.linear2 = MLPLayer.deserialize(linear2)
        if update_g1_has_conv:
            assert isinstance(proj_g1g2, dict)
            obj.proj_g1g2 = MLPLayer.deserialize(proj_g1g2)
        if update_g2_has_g1g1:
            assert isinstance(proj_g1g1g2, dict)
            obj.proj_g1g1g2 = MLPLayer.deserialize(proj_g1g1g2)
        if update_g2_has_attn or update_h2:
            assert isinstance(attn2g_map, dict)
            obj.attn2g_map = Atten2Map.deserialize(attn2g_map)
            if update_g2_has_attn:
                assert isinstance(attn2_mh_apply, dict)
                assert isinstance(attn2_lm, dict)
                obj.attn2_mh_apply = Atten2MultiHeadApply.deserialize(attn2_mh_apply)
                obj.attn2_lm = LayerNorm.deserialize(attn2_lm)
            if update_h2:
                assert isinstance(attn2_ev_apply, dict)
                obj.attn2_ev_apply = Atten2EquiVarApply.deserialize(attn2_ev_apply)
        if g1_out_mlp:
            assert isinstance(g1_self_mlp, dict)
            obj.g1_self_mlp = MLPLayer.deserialize(g1_self_mlp)
        if update_style == "res_residual":
            for ii, t in enumerate(obj.g1_residual):
                t.data = to_torch_tensor(g1_residual[ii])
            for ii, t in enumerate(obj.g2_residual):
                t.data = to_torch_tensor(g2_residual[ii])
            for ii, t in enumerate(obj.h2_residual):
                t.data = to_torch_tensor(h2_residual[ii])
        return obj
