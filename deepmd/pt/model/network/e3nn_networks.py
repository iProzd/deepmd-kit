import torch
import torch.nn as nn
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import Irreps, TensorProduct, Linear
from deepmd.pt.utils.utils import (
    ActivationFn,
)
from typing import (
    Optional,
)


def broadcast(
    src: torch.Tensor,
    other: torch.Tensor,
    dim: int
) -> torch.Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def message_gather(
    node_features: torch.Tensor,
    edge_dst: torch.Tensor,
    message: torch.Tensor
) -> torch.Tensor:
    index = broadcast(edge_dst, message, 0)
    out_shape = [len(node_features)] + list(message.shape[1:])
    out = torch.zeros(
        out_shape,
        dtype=node_features.dtype,
        device=node_features.device
    )
    out.scatter_reduce_(0, index, message, reduce='sum')
    return out


# @compile_mode('script')
class IrrepsConvolutionBlock(nn.Module):

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: list[int] = [8, 64, 64],
        weight_layer_act="tanh",
        denominator: float = 1.0,
        train_denominator: bool = False,
    ) -> None:
        super().__init__()
        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )
        instructions = []
        irreps_mid = []
        weight_numel = 0
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (_, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:  # here we drop l > lmax
                        k = len(irreps_mid)
                        weight_numel += mul_x * 1  # path shape
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, 'uvu', True))

        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()  # type: ignore
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        # From v0.11.x, to compatible with cuEquivariance
        self._instructions_before_sort = instructions
        instructions = sorted(instructions, key=lambda x: x[2])

        self.convolution_kwargs = dict(
            irreps_in1=irreps_x,
            irreps_in2=irreps_filter,
            irreps_out=irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.weight_nn_kwargs = dict(
            hs=weight_layer_input_to_hidden + [weight_numel],
            act=ActivationFn(weight_layer_act)
        )

        self.convolution = None
        self.weight_nn = None
        self.convolution_cls = TensorProduct
        self.weight_nn_cls = FullyConnectedNet

        self.instantiate()
        self._comm_size = irreps_x.dim  # used in parallel

    def instantiate(self) -> None:
        if self.convolution is not None:
            raise ValueError('Convolution layer already exists')
        if self.weight_nn is not None:
            raise ValueError('Weight_nn layer already exists')

        self.convolution = self.convolution_cls(**self.convolution_kwargs)
        self.weight_nn = self.weight_nn_cls(**self.weight_nn_kwargs)

    def forward(
            self,
            node_sph_embed: torch.Tensor,
            edge_sph: torch.Tensor,
            edge_rbf_ebd: torch.Tensor,
            edge_index: torch.Tensor,
    ):
        assert self.convolution is not None, 'Convolution is not instantiated'
        assert self.weight_nn is not None, 'Weight_nn is not instantiated'
        # from IPython import embed
        # embed()
        weight = self.weight_nn(edge_rbf_ebd)
        nf, nall, node_dim = node_sph_embed.shape
        x = node_sph_embed.reshape(nf * nall, -1)

        # note that 1 -> src 0 -> dst
        edge_src = edge_index[:, 1]
        edge_dst = edge_index[:, 0]
        # from IPython import embed
        # embed()

        message = self.convolution(x[edge_src], edge_sph, weight)

        x = message_gather(x, edge_dst, message)
        # from IPython import embed
        # embed()
        x = x.div(self.denominator).reshape(nf, nall, -1)
        return x, message


class IrrepsConvolutionAngleBlock(nn.Module):

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: list[int] = [8, 64, 64],
        weight_layer_act="tanh",
        denominator: float = 1.0,
        train_denominator: bool = False,
    ) -> None:
        super().__init__()
        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )
        instructions = []
        irreps_mid = []
        weight_numel = 0
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (_, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:  # here we drop l > lmax
                        k = len(irreps_mid)
                        weight_numel += mul_x * 1  # path shape
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, 'uvu', True))

        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()  # type: ignore
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        # From v0.11.x, to compatible with cuEquivariance
        self._instructions_before_sort = instructions
        instructions = sorted(instructions, key=lambda x: x[2])

        self.convolution_kwargs = dict(
            irreps_in1=irreps_x,
            irreps_in2=irreps_filter,
            irreps_out=irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.weight_nn_kwargs = dict(
            hs=weight_layer_input_to_hidden + [weight_numel],
            act=ActivationFn(weight_layer_act)
        )

        self.convolution = None
        self.weight_nn = None
        self.convolution_cls = TensorProduct
        self.weight_nn_cls = FullyConnectedNet

        self.instantiate()
        self._comm_size = irreps_x.dim  # used in parallel

    def instantiate(self) -> None:
        if self.convolution is not None:
            raise ValueError('Convolution layer already exists')
        if self.weight_nn is not None:
            raise ValueError('Weight_nn layer already exists')

        self.convolution = self.convolution_cls(**self.convolution_kwargs)
        self.weight_nn = self.weight_nn_cls(**self.weight_nn_kwargs)

    def forward(
            self,
            edge_sph_embed: torch.Tensor,
            angle_sph: torch.Tensor,
            angle_feat_ebd: torch.Tensor,
            angle_index: torch.Tensor,
            a_sw: torch.Tensor,
    ):
        assert self.convolution is not None, 'Convolution is not instantiated'
        assert self.weight_nn is not None, 'Weight_nn is not instantiated'
        weight = self.weight_nn(angle_feat_ebd)
        nedge, edim = edge_sph_embed.shape
        x = edge_sph_embed

        # note that 2 -> src 1 -> dst
        angle_src = angle_index[:, 2]
        angle_dst = angle_index[:, 1]
        # from IPython import embed
        # embed()

        message = self.convolution(x[angle_src], angle_sph, weight) * a_sw.unsqueeze(-1)

        x = message_gather(x, angle_dst, message)
        x = x.div(self.denominator)
        return x


# @compile_mode('script')
class IrrepsLinear(nn.Module):
    """
    wrapper class of e3nn Linear to operate on AtomGraphData
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        **linear_kwargs,
    ) -> None:
        super().__init__()
        self._irreps_in_wo_modal = irreps_in
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.linear_kwargs = linear_kwargs

        self.linear = None
        self.layer_instantiated = False

        # use getter setter
        self.linear_cls = Linear
        self.instantiate()

    def instantiate(self) -> None:
        if self.linear is not None:
            raise ValueError('Linear layer already exists')
        self.linear = self.linear_cls(
            self.irreps_in, self.irreps_out, **self.linear_kwargs
        )
        self.layer_instantiated = True

    def set_num_modalities(self, num_modalities: int) -> None:
        if self.layer_instantiated:
            raise ValueError('Layer already instantiated, can not change modalities')
        irreps_in = self._irreps_in_wo_modal + Irreps(f'{num_modalities}x0e')
        self.num_modalities = num_modalities
        self.irreps_in = irreps_in

    def forward(self, x):
        assert self.linear is not None, 'Layer is not instantiated'
        x = self.linear(x)
        return x


# @compile_mode('script')
class EquivariantGate(nn.Module):
    def __init__(
        self,
        irreps_x: Irreps,
        act_scalar_dict: dict[str, callable],
        act_gate_dict: dict[str, callable],
    ) -> None:
        super().__init__()
        parity_map = {1: 'e', -1: 'o'}

        irreps_gated_elem = []
        irreps_scalars_elem = []
        # non scalar irreps > gated / scalar irreps > scalars
        for mul, irreps in irreps_x:
            if irreps.l > 0:
                irreps_gated_elem.append((mul, irreps))
            else:
                irreps_scalars_elem.append((mul, irreps))
        irreps_scalars = Irreps(irreps_scalars_elem)
        irreps_gated = Irreps(irreps_gated_elem)

        irreps_gates_parity = 1 if '0e' in irreps_scalars else -1
        irreps_gates = Irreps(
            [(mul, (0, irreps_gates_parity)) for mul, _ in irreps_gated]
        )

        act_scalars = [
            act_scalar_dict[parity_map[p]] for _, (_, p) in irreps_scalars
        ]
        act_gates = [act_gate_dict[parity_map[p]] for _, (_, p) in irreps_gates]

        self.gate = Gate(
            irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated
        )

    def get_gate_irreps_in(self):
        """
        user must call this function to get proper irreps in for forward
        """
        return self.gate.irreps_in

    def forward(self, x):
        x = self.gate(x)
        return x


# @compile_mode('script')
class IrrepsBlock(nn.Module):
    # conv + linear + eqgate
    def __init__(
            self,
            irreps_x: Irreps,
            irreps_filter: Irreps,
            irreps_out_tp: Irreps,
            irreps_out: Irreps,
            weight_layer_act="tanh",
            denominator: float = 1.0,
            train_denominator: bool = False,
            e3nn_conv_use_edge_sh_feat: bool = False,
            weight_layer_input_to_hidden: list[int] = [8, 64, 64],
            edge_sh_feat_use_rbf_weights: bool = False,
    ) -> None:
        super().__init__()
        self.e3nn_conv_use_edge_sh_feat = e3nn_conv_use_edge_sh_feat
        self.irreps_out_tp = irreps_out_tp

        # 1. conv
        if not self.e3nn_conv_use_edge_sh_feat:
            self.eq_conv = IrrepsConvolutionBlock(
                irreps_x=irreps_x,
                irreps_filter=irreps_filter,
                irreps_out=irreps_out_tp,
                weight_layer_act=weight_layer_act,
                denominator=denominator,
                train_denominator=train_denominator,
                weight_layer_input_to_hidden=weight_layer_input_to_hidden,
            )
            linear_in_dim = irreps_out_tp
        else:
            self.eq_conv = FullLInteractionsUUU(
                irreps1_in=irreps_x,
                denominator=denominator,
                train_denominator=train_denominator,
                weight_layer_input_to_hidden=weight_layer_input_to_hidden,
                use_rbf_weights=edge_sh_feat_use_rbf_weights,
                weight_layer_act=weight_layer_act,
            )
            linear_in_dim = self.eq_conv.irreps_cat

        # 3. gate
        self.gate_act_dict = {
            "e": ActivationFn(weight_layer_act),
            "o": ActivationFn('tanh'),
        }

        self.eq_gate = EquivariantGate(irreps_out, self.gate_act_dict, self.gate_act_dict)

        # 2. linear
        self.linear_after_conv = IrrepsLinear(
            irreps_in=linear_in_dim,
            irreps_out=self.eq_gate.get_gate_irreps_in(),
            biases=False,
        )

    def forward(
            self,
            node_sph_embed: torch.Tensor,
            edge_sph: torch.Tensor,
            edge_rbf_ebd: torch.Tensor,
            edge_index: torch.Tensor,
            edge_sph_embed: Optional[torch.Tensor] = None,
    ):
        # conv
        if not self.e3nn_conv_use_edge_sh_feat:
            node_sph_embed, edge_sph_update = self.eq_conv(
                node_sph_embed,
                edge_sph,
                edge_rbf_ebd,
                edge_index,
            )
        else:
            assert edge_sph_embed is not None
            node_sph_embed, edge_sph_update = self.eq_conv(
                node_sph_embed,
                edge_sph_embed,
                edge_index,
                edge_rbf_ebd=edge_rbf_ebd,
            )


        # linear
        node_sph_embed = self.linear_after_conv(node_sph_embed)
        # eqgate
        node_sph_embed = self.eq_gate(node_sph_embed)
        return node_sph_embed, edge_sph_update


class IrrepsAngleBlock(nn.Module):
    # conv + linear + eqgate
    def __init__(
            self,
            irreps_x: Irreps,
            irreps_filter: Irreps,
            irreps_out_tp: Irreps,
            irreps_out: Irreps,
            weight_layer_act="tanh",
            denominator: float = 1.0,
            train_denominator: bool = False,
            weight_layer_input_to_hidden: list[int] = [32],
    ) -> None:
        super().__init__()

        # 1. conv
        self.eq_conv = IrrepsConvolutionAngleBlock(
            irreps_x=irreps_x,
            irreps_filter=irreps_filter,
            irreps_out=irreps_out_tp,
            weight_layer_act=weight_layer_act,
            denominator=denominator,
            train_denominator=train_denominator,
            weight_layer_input_to_hidden=weight_layer_input_to_hidden,
        )

        # 3. gate
        self.gate_act_dict = {
            "e": ActivationFn(weight_layer_act),
            "o": ActivationFn('tanh'),
        }

        self.eq_gate = EquivariantGate(irreps_out, self.gate_act_dict, self.gate_act_dict)

        # 2. linear
        self.linear_after_conv = IrrepsLinear(
            irreps_in=irreps_out_tp,
            irreps_out=self.eq_gate.get_gate_irreps_in(),
            biases=False,
        )

    def forward(
            self,
            edge_sph_embed: torch.Tensor,
            angle_sph: torch.Tensor,
            angle_feat_ebd: torch.Tensor,
            angle_index: torch.Tensor,
            a_sw: torch.Tensor,
    ):
        # conv
        edge_sph_embed = self.eq_conv(
            edge_sph_embed,
            angle_sph,
            angle_feat_ebd,
            angle_index,
            a_sw,
        )

        # linear
        edge_sph_embed = self.linear_after_conv(edge_sph_embed)

        # eqgate
        edge_sph_embed = self.eq_gate(edge_sph_embed)
        return edge_sph_embed


def _idx_l(irreps: Irreps, l: int):
    for i, (_, ir) in enumerate(irreps):
        if ir.l == l:
            return i
    return None


class FullLInteractionsUUU(nn.Module):
    """
    - irrep1: "128x0e + 64x1e + 32x2e" 或 "128x0e"
    - irreps2_base: 固定 "64x0e + 32x1e + 32x2e"
    - 三份 kernel self-interaction:
        k1: 128x0e + 128x1e + 128x2e   （用于 l1=0 组）
        k2:  64x0e +  64x1e +  64x2e   （用于 l1=1 组）
        k3:  32x0e +  32x1e +  32x2e   （用于 l1=2 组）
    - 每组用 `uuu`，对 (l1, l2) 枚举所有 CG 允许的 l_out ∈ {0,1,2}。
      `uuu` 逐通道：要求 in1[l1]、in2[l2] 与 out[l_out] 的 multiplicity 一致。
      因此每组的输出在各阶的 multiplicity 取自 irrep1 对应阶。
    """
    def __init__(
            self,
            irreps1_in: Irreps,
            l_max: int = 2,
            denominator: float = 1.0,
            train_denominator: bool = False,
            use_rbf_weights: bool = False,
            weight_layer_input_to_hidden: list[int] = [8, 64, 64],
            weight_layer_act="tanh",
    ):
        super().__init__()
        self.ir1 = irreps1_in
        self.l_max = l_max
        self.ir2_base = Irreps("64x0e + 32x1e + 32x2e").simplify()  # edge feature irreps
        self.use_rbf_weights = use_rbf_weights

        # self-interactions on kernel
        self.si1 = Linear(self.ir2_base, "128x0e + 128x1e + 128x2e")
        self.si2 = Linear(self.ir2_base,  "64x0e +  64x1e +  64x2e")
        self.si3 = Linear(self.ir2_base,  "32x0e +  32x1e +  32x2e")

        # 三组 TP：分别针对 l1=0/1/2，输出阶数上限为 2
        self.tp0 = self._build_tp_group(l1=0, ir2=self.si1.irreps_out)
        self.tp1 = self._build_tp_group(l1=1, ir2=self.si2.irreps_out)
        self.tp2 = self._build_tp_group(l1=2, ir2=self.si3.irreps_out)

        if self.use_rbf_weights:
            self.tp_weight_linear = FullyConnectedNet(
                weight_layer_input_to_hidden + [self.tp0.weight_numel + self.tp1.weight_numel + self.tp2.weight_numel],
                act=ActivationFn(weight_layer_act),
            )
        else:
            self.tp_weight_linear = None

        # 拼接后的型谱
        self.irreps_cat = (self.tp0.irreps_out + self.tp1.irreps_out + self.tp2.irreps_out).simplify()

        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )

        # gate for edge
        self.gate_act_dict = {
            "e": ActivationFn("silu"),
            "o": ActivationFn('tanh'),
        }

        self.eq_gate = EquivariantGate(self.ir2_base, self.gate_act_dict, self.gate_act_dict)

        # linear
        self.linear_after_conv_edge = IrrepsLinear(
            irreps_in=self.irreps_cat,
            irreps_out=self.eq_gate.get_gate_irreps_in(),
            biases=False,
        )

    def _build_tp_group(self, l1: int, ir2: Irreps) -> TensorProduct:
        if l1 >= len(self.ir1):
            return TensorProduct(self.ir1, ir2, Irreps([]), instructions=[], shared_weights=not self.use_rbf_weights, internal_weights=not self.use_rbf_weights,)
        ir1_l1 = self.ir1[_idx_l(self.ir1, l1)]
        ir1_mul = ir1_l1.mul
        # ir1_l1 x "ir1_mul x 0e + ir1_mul x 1e + ir1_mul x 2e"
        ir_out = Irreps([])
        instr = []
        for i, (mul, ir) in enumerate(ir2):
            assert mul == ir1_mul
            for ir_single_out in ir1_l1.ir * ir:
                if ir_single_out.l <= self.l_max:
                    ir_out += Irreps(f"{ir1_mul}x{ir_single_out.l}e")
                    instr.append((_idx_l(self.ir1, l1), i, len(ir_out) - 1, "uuu", True))

        ir_out, p, _ = ir_out.sort()  # type: ignore
        instr = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instr
        ]

        return TensorProduct(
            self.ir1, ir2, ir_out,
            instructions=instr,
            internal_weights=not self.use_rbf_weights,
            shared_weights=not self.use_rbf_weights,
        )

    def forward(self, x1, x2, edge_index: torch.Tensor, edge_rbf_ebd: Optional[torch.Tensor] = None):
        """
        x1: (..., self.ir1.dim)
        x2: (..., self.ir2_base.dim)
        return: y_cat with Irreps == self.irreps_cat
        """
        k1 = self.si1(x2)   # for l1=0
        k2 = self.si2(x2)   # for l1=1
        k3 = self.si3(x2)   # for l1=2

        nf, nall, _ = x1.shape
        x1 = x1.reshape(nf * nall, -1)

        # note that 1 -> src 0 -> dst
        edge_src = edge_index[:, 1]
        edge_dst = edge_index[:, 0]

        x1_src = x1[edge_src]
        k_weight = self.tp_weight_linear(edge_rbf_ebd) if self.tp_weight_linear is not None and edge_rbf_ebd is not None else None
        if k_weight is not None:
            k1_weight, k2_weight, k3_weight = torch.split(
                k_weight,
                [self.tp0.weight_numel, self.tp1.weight_numel, self.tp2.weight_numel],
                dim=-1
            )
        else:
            k1_weight = k2_weight = k3_weight = None

        y0 = self.tp0(x1_src, k1, k1_weight)  # 包含 (0,0)->0; (0,1)->1; (0,2)->2
        y1 = self.tp1(x1_src, k2, k2_weight)  # 包含 (1,0)->1; (1,1)->0,1,2; (1,2)->1,2
        y2 = self.tp2(x1_src, k3, k3_weight)  # 包含 (2,0)->2; (2,1)->1,2; (2,2)->0,1,2（裁到≤2）
        y = torch.cat([y0, y1, y2], dim=-1)

        x1 = message_gather(x1, edge_dst, y)
        x1 = x1.div(self.denominator).reshape(nf, nall, -1)

        y = self.linear_after_conv_edge(y)
        y = self.eq_gate(y)
        return x1, y
