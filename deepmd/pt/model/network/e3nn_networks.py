import torch
import torch.nn as nn
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import Irreps, TensorProduct, Linear
from deepmd.pt.utils.utils import (
    ActivationFn,
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
        return x


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
            weight_layer_input_to_hidden: list[int] = [8, 64, 64],
    ) -> None:
        super().__init__()

        # 1. conv
        self.eq_conv = IrrepsConvolutionBlock(
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
            node_sph_embed: torch.Tensor,
            edge_sph: torch.Tensor,
            edge_rbf_ebd: torch.Tensor,
            edge_index: torch.Tensor,
    ):
        # conv
        node_sph_embed = self.eq_conv(
            node_sph_embed,
            edge_sph,
            edge_rbf_ebd,
            edge_index,
        )

        # linear
        node_sph_embed = self.linear_after_conv(node_sph_embed)

        # eqgate
        node_sph_embed = self.eq_gate(node_sph_embed)
        return node_sph_embed


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