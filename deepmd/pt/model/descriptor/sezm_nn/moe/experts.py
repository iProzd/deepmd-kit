# SPDX-License-Identifier: LGPL-3.0-or-later
"""Expert SO(2) layers for SeZM MoE.

Routing expert weights use an expert-major 3D layout:
``(n_per_gpu, in_block, out_block)``.  This differs from the DPA3 reference
layout ``(in_block, out_block, experts_per_gpu)`` and lets the baseline
for-loop path select each local expert with ``matrix[local_eid]`` directly.
"""

from __future__ import (
    annotations,
)

import math

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.descriptor.sezm_nn.activation import (
    GatedActivation,
)
from deepmd.pt.model.descriptor.sezm_nn.so2_math import (
    apply_so2_blocks_batched,
    apply_so2_blocks_one,
    build_m_major_layout,
)
from deepmd.pt.model.descriptor.sezm_nn.utils import (
    init_trunc_normal_fan_in_out,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)


class _ExpertSO2LinearLayer(nn.Module):
    """SO(2)-equivariant linear layer backed by expert-major 3D tensors."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int,
        in_channels: int,
        out_channels: int,
        n_per_gpu: int,
        has_bias: bool,
        is_shared: bool,
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        if lmax < 0:
            raise ValueError("`lmax` must be non-negative")
        if mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if mmax > lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_per_gpu = int(n_per_gpu)
        self.has_bias = bool(has_bias)
        self.is_shared = bool(is_shared)
        self.dtype = PRECISION_DICT[precision]
        self.device = env.DEVICE

        layout = build_m_major_layout(self.lmax, self.mmax, device=self.device)
        self.reduced_dim = layout["reduced_dim"]
        self._m0_size = layout["m0_size"]
        self._m_ranges = layout["m_ranges"]

        num_in_m0 = self._m0_size * self.in_channels
        num_out_m0 = self._m0_size * self.out_channels
        self._register_matrix(
            "matrix_m0",
            torch.empty(
                self.n_per_gpu,
                num_in_m0,
                num_out_m0,
                device=self.device,
                dtype=self.dtype,
            ),
            seed,
            seed_offset=1000,
            scale=1.0,
        )

        for m in range(1, self.mmax + 1):
            num_l = self.lmax - m + 1
            num_in = num_l * self.in_channels
            num_out = 2 * num_l * self.out_channels
            self._register_matrix(
                f"matrix_m_{m}",
                torch.empty(
                    self.n_per_gpu,
                    num_in,
                    num_out,
                    device=self.device,
                    dtype=self.dtype,
                ),
                seed,
                seed_offset=2000 + 100 * m,
                scale=1.0 / math.sqrt(2.0),
            )

        if self.has_bias:
            bias = nn.Parameter(
                torch.zeros(
                    self.n_per_gpu,
                    self.out_channels,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            if self.is_shared:
                self.shared_bias = bias
            else:
                self.routing_bias = bias

    def _register_matrix(
        self,
        suffix: str,
        value: torch.Tensor,
        seed: int | list[int] | None,
        *,
        seed_offset: int,
        scale: float,
    ) -> None:
        """Initialize and register a routing or shared expert matrix."""
        for expert_idx in range(self.n_per_gpu):
            init_trunc_normal_fan_in_out(
                value[expert_idx],
                child_seed(seed, seed_offset + expert_idx),
            )
        value.mul_(scale)
        prefix = "shared" if self.is_shared else "routing"
        self.register_parameter(f"{prefix}_{suffix}", nn.Parameter(value))

    def _matrix(self, suffix: str) -> torch.Tensor:
        prefix = "shared" if self.is_shared else "routing"
        return getattr(self, f"{prefix}_{suffix}")

    def _matrices_m(self) -> list[torch.Tensor]:
        return [self._matrix(f"matrix_m_{m}") for m in range(1, self.mmax + 1)]

    def _forward_one(self, x: torch.Tensor, local_eid: int) -> torch.Tensor:
        """Apply one local expert to a token batch."""
        if x.dim() != 3:
            raise ValueError("`x` must have shape (N, D_m, Cin)")
        if x.shape[1] != self.reduced_dim or x.shape[2] != self.in_channels:
            raise ValueError(
                "Expected `x` with shape "
                f"(N, {self.reduced_dim}, {self.in_channels}), got {tuple(x.shape)}."
            )
        if not 0 <= int(local_eid) < self.n_per_gpu:
            raise ValueError(
                f"`local_eid` must be in [0, {self.n_per_gpu}), got {local_eid}."
            )

        out = apply_so2_blocks_one(
            x,
            matrix_m0=self._matrix("matrix_m0")[local_eid],
            matrices_m=[matrix[local_eid] for matrix in self._matrices_m()],
            m0_size=self._m0_size,
            m_ranges=self._m_ranges,
            out_channels=self.out_channels,
        )
        bias = None
        if self.has_bias:
            bias = self.shared_bias if self.is_shared else self.routing_bias
        if bias is not None:
            out[:, 0, :] = out[:, 0, :] + bias[local_eid]
        return out

    def forward_one_expert(self, x: torch.Tensor, local_eid: int = 0) -> torch.Tensor:
        """Apply one local expert to an unsplit token batch."""
        return self._forward_one(x, int(local_eid))

    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all local experts in parallel along axis 1."""
        if x.dim() != 4:
            raise ValueError("`x` must have shape (E, n_per_gpu, D_m, Cin)")
        if (
            x.shape[1] != self.n_per_gpu
            or x.shape[2] != self.reduced_dim
            or x.shape[3] != self.in_channels
        ):
            raise ValueError(
                "Expected `x` with shape "
                f"(E, {self.n_per_gpu}, {self.reduced_dim}, {self.in_channels}), "
                f"got {tuple(x.shape)}."
            )

        out = apply_so2_blocks_batched(
            x,
            matrix_m0=self._matrix("matrix_m0"),
            matrices_m=self._matrices_m(),
            m0_size=self._m0_size,
            m_ranges=self._m_ranges,
            out_channels=self.out_channels,
        )
        if self.has_bias:
            bias = self.shared_bias if self.is_shared else self.routing_bias
            out[:, :, 0, :] = out[:, :, 0, :] + bias.unsqueeze(0)
        return out


class _ExpertSO2Stack(nn.Module):
    """Stack of expert SO(2) layers and activations."""

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int,
        focus_dim: int,
        n_per_gpu: int,
        so2_layers: int = 4,
        activation_function: str = "silu",
        mlp_bias: bool = False,
        use_layer_scale: bool = False,
        is_shared: bool = False,
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        if so2_layers < 1:
            raise ValueError("`so2_layers` must be >= 1")

        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.focus_dim = int(focus_dim)
        self.n_per_gpu = int(n_per_gpu)
        self.so2_layers = int(so2_layers)
        self.mlp_bias = bool(mlp_bias)
        self.use_layer_scale = bool(use_layer_scale)
        self.is_shared = bool(is_shared)
        self.dtype = PRECISION_DICT[precision]
        self.device = env.DEVICE

        seed_layers = child_seed(seed, 0)
        seed_activations = child_seed(seed, 1)
        self.layers = nn.ModuleList(
            [
                _ExpertSO2LinearLayer(
                    lmax=self.lmax,
                    mmax=self.mmax,
                    in_channels=self.focus_dim,
                    out_channels=self.focus_dim,
                    n_per_gpu=self.n_per_gpu,
                    has_bias=self.mlp_bias and layer_idx == 0,
                    is_shared=self.is_shared,
                    precision=precision,
                    seed=child_seed(seed_layers, layer_idx),
                )
                for layer_idx in range(self.so2_layers)
            ]
        )

        activations: list[nn.Module] = []
        for layer_idx in range(self.so2_layers - 1):
            activations.append(
                GatedActivation(
                    lmax=self.lmax,
                    mmax=self.mmax,
                    channels=self.focus_dim,
                    n_focus=self.n_per_gpu if self.is_shared else 1,
                    dtype=self.dtype,
                    activation_function=activation_function,
                    mlp_bias=self.mlp_bias,
                    layout="nfdc",
                    trainable=True,
                    seed=child_seed(seed_activations, layer_idx),
                )
            )
        activations.append(nn.Identity())
        self.activations = nn.ModuleList(activations)

        if self.use_layer_scale:
            self.adam_layer_scales = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.ones(
                            self.focus_dim,
                            dtype=self.dtype,
                            device=self.device,
                        )
                        * 1e-3
                    )
                    for _ in range(self.so2_layers)
                ]
            )
        else:
            self.adam_layer_scales = None

    def _activate_one_expert(
        self, x: torch.Tensor, layer_idx: int, local_eid: int
    ) -> torch.Tensor:
        if not self.is_shared:
            y = self.activations[layer_idx](x.unsqueeze(1))
            return y.squeeze(1)

        slots = []
        for expert_idx in range(self.n_per_gpu):
            if expert_idx == local_eid:
                slots.append(x)
            else:
                slots.append(x.new_zeros(x.shape))
        y = self.activations[layer_idx](torch.stack(slots, dim=1))
        return y[:, local_eid, :, :]

    def _activate_shared(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.activations[layer_idx](x)

    def _apply_layer_scale(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not self.use_layer_scale:
            return x
        shape = (1,) * (x.dim() - 1) + (self.focus_dim,)
        scale = self.adam_layer_scales[layer_idx].reshape(shape)
        return x * scale

    def _bias0(self) -> torch.Tensor:
        return (
            self.layers[0].shared_bias
            if self.is_shared
            else self.layers[0].routing_bias
        )

    def _apply_one_bias_correction(
        self,
        x: torch.Tensor,
        local_eid: int,
        radial_factor: torch.Tensor | None,
    ) -> torch.Tensor:
        if radial_factor is None:
            return x
        if not self.mlp_bias:
            raise ValueError("`radial_factor` requires `mlp_bias=True`")
        correction = self._bias0()[int(local_eid)].reshape(1, self.focus_dim)
        correction = correction * radial_factor
        x_l0_plus = x[:, 0:1, :] + correction.unsqueeze(1)
        return torch.cat([x_l0_plus, x[:, 1:, :]], dim=1)

    def _apply_shared_bias_correction(
        self,
        x: torch.Tensor,
        radial_factors: torch.Tensor | None,
    ) -> torch.Tensor:
        if radial_factors is None:
            return x
        if not self.mlp_bias:
            raise ValueError("`radial_factors` requires `mlp_bias=True`")
        correction = self._bias0().unsqueeze(0) * radial_factors
        x_l0_plus = x[:, :, 0:1, :] + correction.unsqueeze(2)
        return torch.cat([x_l0_plus, x[:, :, 1:, :]], dim=2)

    def forward_routing(
        self,
        sorted_tokens: torch.Tensor,
        local_eids_sorted: torch.Tensor,
        split_sizes: list[int],
        sorted_radial_factor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run routing experts in expert-major order."""
        if len(split_sizes) != self.n_per_gpu:
            raise ValueError(
                f"`split_sizes` length must be {self.n_per_gpu}, got {len(split_sizes)}."
            )
        if sum(split_sizes) != sorted_tokens.shape[0]:
            raise ValueError("`split_sizes` must sum to the number of tokens")
        if local_eids_sorted.shape[0] != sorted_tokens.shape[0]:
            raise ValueError("`local_eids_sorted` must match token count")

        chunks = torch.split(sorted_tokens, split_sizes, dim=0)
        if sorted_radial_factor is None:
            factor_chunks: list[torch.Tensor | None] = [None] * self.n_per_gpu
        else:
            factor_chunks = list(torch.split(sorted_radial_factor, split_sizes, dim=0))

        output_chunks: list[torch.Tensor] = []
        for local_eid, chunk in enumerate(chunks):
            if chunk.shape[0] == 0:
                output_chunks.append(chunk)
                continue
            x_e = chunk
            factor_e = factor_chunks[local_eid]
            for layer_idx, layer in enumerate(self.layers):
                x_e = layer.forward_one_expert(x_e, local_eid)
                if layer_idx == 0:
                    x_e = self._apply_one_bias_correction(
                        x_e,
                        local_eid,
                        factor_e,
                    )
                x_e = self._activate_one_expert(x_e, layer_idx, local_eid)
                x_e = self._apply_layer_scale(x_e, layer_idx)
            output_chunks.append(x_e)
        return torch.cat(output_chunks, dim=0)

    def forward_shared(
        self,
        x: torch.Tensor,
        radial_factors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run shared experts in batched layer-major order."""
        for layer_idx, layer in enumerate(self.layers):
            x = layer.forward_shared(x)
            if layer_idx == 0:
                x = self._apply_shared_bias_correction(x, radial_factors)
            x = self._activate_shared(x, layer_idx)
            x = self._apply_layer_scale(x, layer_idx)
        return x


class MoESO2ExpertCollection(nn.Module):
    """Routing and shared SO(2) expert stacks for SeZM MoE."""

    def __init__(
        self,
        lmax: int,
        mmax: int,
        focus_dim: int,
        n_experts_per_gpu: int,
        n_shared_experts: int,
        so2_layers: int = 4,
        activation_function: str = "silu",
        mlp_bias: bool = False,
        use_layer_scale: bool = False,
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        if lmax < 0:
            raise ValueError("`lmax` must be non-negative")
        if mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if mmax > lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        if focus_dim <= 0:
            raise ValueError("`focus_dim` must be positive")
        if n_experts_per_gpu < 1:
            raise ValueError("`n_experts_per_gpu` must be >= 1")
        if n_shared_experts < 0:
            raise ValueError("`n_shared_experts` must be >= 0")
        if so2_layers < 1:
            raise ValueError("`so2_layers` must be >= 1")

        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.focus_dim = int(focus_dim)
        self.n_experts_per_gpu = int(n_experts_per_gpu)
        self.n_shared_experts = int(n_shared_experts)
        self.so2_layers = int(so2_layers)
        self.mlp_bias = bool(mlp_bias)
        self.use_layer_scale = bool(use_layer_scale)
        self.reduced_dim = (
            self.lmax
            + 1
            + sum(2 * (self.lmax - m + 1) for m in range(1, self.mmax + 1))
        )

        self.routing_stack = _ExpertSO2Stack(
            lmax=self.lmax,
            mmax=self.mmax,
            focus_dim=self.focus_dim,
            n_per_gpu=self.n_experts_per_gpu,
            so2_layers=self.so2_layers,
            activation_function=activation_function,
            mlp_bias=self.mlp_bias,
            use_layer_scale=self.use_layer_scale,
            is_shared=False,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        if self.n_shared_experts > 0:
            self.shared_stack: _ExpertSO2Stack | None = _ExpertSO2Stack(
                lmax=self.lmax,
                mmax=self.mmax,
                focus_dim=self.focus_dim,
                n_per_gpu=self.n_shared_experts,
                so2_layers=self.so2_layers,
                activation_function=activation_function,
                mlp_bias=self.mlp_bias,
                use_layer_scale=self.use_layer_scale,
                is_shared=True,
                precision=precision,
                seed=child_seed(seed, 1),
            )
        else:
            self.shared_stack = None

    def forward_routing(
        self,
        sorted_tokens: torch.Tensor,
        local_eids_sorted: torch.Tensor,
        split_sizes: list[int],
        sorted_radial_factor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run routing expert tokens through the local routing stack."""
        return self.routing_stack.forward_routing(
            sorted_tokens,
            local_eids_sorted,
            split_sizes,
            sorted_radial_factor,
        )

    def forward_shared(
        self,
        x_shared: torch.Tensor,
        radial_factors_shared: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run local shared experts in parallel over the shared slot axis."""
        if self.shared_stack is None:
            if x_shared.dim() != 4:
                raise ValueError("`x_shared` must have shape (E, 0, D_m, Cf)")
            n_edge, _, reduced_dim, focus_dim = x_shared.shape
            return x_shared.new_empty(n_edge, 0, reduced_dim, focus_dim)
        return self.shared_stack.forward_shared(x_shared, radial_factors_shared)


__all__ = ["MoESO2ExpertCollection", "_ExpertSO2LinearLayer", "_ExpertSO2Stack"]
