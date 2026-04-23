# SPDX-License-Identifier: LGPL-3.0-or-later
"""MoE Expert MLP layers for Expert Parallelism.

Provides:
- ``ExpertMLPLayer``: a single expert wrapping ``MLPLayer`` to compute
  ``act(x @ W + b)`` with independent W and b parameters.
- ``MoEExpertCollection``: manages a set of local routing experts plus
  optional shared experts for one MLP position in a RepFlowLayer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.utils.env import (
    DEVICE,
    PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
    get_generator,
)

try:
    from deepmd.pt.model.network.init import (
        normal_,
    )
except ImportError:
    from deepmd.pt.utils.env import (
        GLOBAL_PT_FLOAT_PRECISION,
    )
    def normal_(tensor, mean=0.0, std=1.0, generator=None):
        """Fallback normal init if deepmd.pt.model.network.init not available."""
        with torch.no_grad():
            if generator is not None:
                tensor.normal_(mean, std, generator=generator)
            else:
                tensor.normal_(mean, std)


def empty_t(shape: tuple[int, ...], precision: str) -> torch.Tensor:
    """Create empty tensor with correct device and dtype."""
    return torch.empty(shape, dtype=PRECISION_DICT[precision], device=DEVICE)


class ExpertMLPLayer(nn.Module):
    """Single expert MLP: ``act(x @ W + b)``.

    Each expert has its own independent W and b (not shared).
    Internally wraps ``MLPLayer`` for consistent initialisation, device
    handling and precision management.

    Parameters
    ----------
    num_in : int
        Input feature dimension.
    num_out : int
        Output feature dimension.
    activation_function : str
        Name of the activation function (e.g. ``"silu"``).
    precision : str
        Floating-point precision (e.g. ``"float64"``).
    seed : int, list[int], or None
        Random seed for reproducible weight initialisation.
    """

    def __init__(
        self,
        num_in: int,
        num_out: int,
        activation_function: str = "silu",
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.activation_function = activation_function
        # MLPLayer handles weight init, device placement, and precision.
        # activation_function=None so we control activation ourselves —
        # important for merged-MLP scenarios (M_edge, M_angle) where
        # activation is applied after the merged linear, not inside MLPLayer.
        self.mlp = MLPLayer(
            num_in,
            num_out,
            bias=True,
            activation_function=None,
            precision=precision,
            seed=seed,
        )
        self.activate = ActivationFn(activation_function)

    @property
    def matrix(self) -> nn.Parameter:
        """Weight matrix W, shape ``[num_in, num_out]``."""
        return self.mlp.matrix

    @property
    def bias(self) -> nn.Parameter:
        """Bias vector b, shape ``[num_out]``."""
        return self.mlp.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape ``[N_tokens, num_in]``

        Returns
        -------
        Tensor, shape ``[N_tokens, num_out]``
        """
        return self.activate(self.mlp(x))


class MoEExpertCollection(nn.Module):
    """Collection of local routing experts and shared experts for one MLP position.

    Uses a shared 3D weight tensor ``[num_in, num_out, experts_per_gpu]``
    for all routing experts, matching the efficient memory layout of the
    old ``mlp_layer_moe_dynamic_sel`` implementation.  This avoids 64
    independent ``nn.Parameter`` objects and enables cache-friendly
    batched expert computation.

    Parameters
    ----------
    num_in : int
        Input feature dimension.
    num_out : int
        Output feature dimension.
    experts_per_gpu : int
        Number of routing experts held on this GPU.
    n_shared_experts : int
        Number of shared experts (replicated on every GPU).
    activation_function : str
        Activation function name.
    precision : str
        Floating-point precision.
    seed : int, list[int], or None
        Parent seed; child seeds are derived per expert.
    """

    def __init__(
        self,
        num_in: int,
        num_out: int,
        experts_per_gpu: int,
        n_shared_experts: int = 0,
        activation_function: str = "silu",
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.experts_per_gpu = experts_per_gpu
        self.n_shared_experts = n_shared_experts
        self.activation_function = activation_function
        self.precision = precision

        # ── Shared 3D tensor for routing experts ──
        # Shape: [num_in, num_out, experts_per_gpu] (matches old code layout)
        self.routing_matrix = nn.Parameter(
            empty_t((num_in, num_out, experts_per_gpu), precision)
        )
        # Per-expert bias: [num_out, experts_per_gpu]
        self.routing_bias = nn.Parameter(
            empty_t((num_out, experts_per_gpu), precision)
        )

        # Activation applied ONCE after all experts compute
        self.activate = ActivationFn(activation_function)

        # Initialize weights (matches old mlp_layer_moe_dynamic_sel default_normal_init)
        self._init_routing_weights(seed)

        # Keep nn.ModuleList of ExpertMLPLayer as a property facade for backward
        # compatibility with tests that access routing_experts[i].matrix/bias.
        # These are lightweight wrappers that reference VIEWS of the 3D tensor.
        self.routing_experts = nn.ModuleList([
            _ExpertView(self, i, activation_function, precision)
            for i in range(experts_per_gpu)
        ])

        # Shared experts remain as independent ExpertMLPLayer instances.
        if n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                ExpertMLPLayer(
                    num_in, num_out, activation_function, precision,
                    seed=child_seed(seed, experts_per_gpu + i),
                )
                for i in range(n_shared_experts)
            ])
        else:
            self.shared_experts = nn.ModuleList()

    def _init_routing_weights(
        self,
        seed: int | list[int] | None = None,
    ) -> None:
        """Initialize routing_matrix and routing_bias to match old code's
        ``default_normal_init``.

        The old code uses:
          std = stddev / sqrt(num_out * numb_experts + num_in)
        for the weight matrix and normal_(bias, mean=0, std=1) for bias.
        """
        random_generator = get_generator(seed)
        stddev = 1.0
        bavg = 0.0
        normal_(
            self.routing_matrix.data,
            std=stddev / (self.num_out * self.experts_per_gpu + self.num_in) ** 0.5,
            generator=random_generator,
        )
        normal_(
            self.routing_bias.data,
            mean=bavg,
            std=stddev,
            generator=random_generator,
        )

    def forward_expert(
        self, x: torch.Tensor, local_expert_idx: int
    ) -> torch.Tensor:
        """Compute output using the specified routing expert.

        This is a compatibility method that uses the shared 3D tensor
        under the hood.

        Parameters
        ----------
        x : Tensor, shape ``[N_tokens, num_in]``
        local_expert_idx : int
            Index into routing experts (0..experts_per_gpu-1).

        Returns
        -------
        Tensor, shape ``[N_tokens, num_out]``
        """
        # Use a view of the shared 3D tensor: W[:, :, eid] → [I, O]
        W = self.routing_matrix[:, :, local_expert_idx]  # [I, O], view
        b = self.routing_bias[:, local_expert_idx]        # [O], view
        return self.activate(torch.matmul(x, W) + b)

    def forward_expert_batched(
        self,
        sorted_features: torch.Tensor,
        expert_ids_sorted: torch.Tensor,
        split_sizes: list[int],
    ) -> torch.Tensor:
        """Batched expert forward using shared 3D tensor.

        All routing experts are computed in a tight loop using views of
        the shared ``routing_matrix``.  Activation is applied ONCE at
        the end (not inside each expert), matching old code behavior.

        Parameters
        ----------
        sorted_features : Tensor ``[N_expanded, num_in]``
            Features sorted by expert ID.
        expert_ids_sorted : Tensor ``[N_expanded]``
            Expert IDs in sorted order.
        split_sizes : list[int]
            Number of tokens per expert (length = n_routing_experts or
            experts_per_gpu).

        Returns
        -------
        Tensor ``[N_expanded, num_out]``
            Expert outputs with activation applied.
        """
        N = sorted_features.shape[0]
        if N == 0:
            return sorted_features.new_zeros(0, self.num_out)

        # W: [E, I, O] — permuted view for efficient per-expert matmul
        W = self.routing_matrix.permute(2, 0, 1)  # [E, I, O]
        b = self.routing_bias                       # [O, E]

        # Split into per-expert chunks (already sorted, so contiguous)
        chunks = torch.split(sorted_features, split_sizes)

        out_parts: list[torch.Tensor] = []
        for eid, chunk in enumerate(chunks):
            if chunk.shape[0] > 0:
                # Map global expert ID to local: eid may be global
                local_eid = eid % self.experts_per_gpu
                # matmul: [N_e, I] @ [I, O] → [N_e, O]
                y = torch.matmul(chunk, W[local_eid]) + b[:, local_eid]
                out_parts.append(y)
            else:
                out_parts.append(chunk.new_zeros(0, self.num_out))

        # Cat all expert outputs and apply activation ONCE
        sorted_out = torch.cat(out_parts, dim=0)
        return self.activate(sorted_out)

    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """Compute output using all shared experts, summed.

        Parameters
        ----------
        x : Tensor, shape ``[N_tokens, num_in]``

        Returns
        -------
        Tensor, shape ``[N_tokens, num_out]``
            Sum of all shared expert outputs.
            Zero tensor if no shared experts.
        """
        if len(self.shared_experts) == 0:
            return torch.zeros(
                x.shape[0], self.num_out, dtype=x.dtype, device=x.device,
            )
        result = self.shared_experts[0](x)
        for se in self.shared_experts[1:]:
            result = result + se(x)
        return result


class _ExpertView(nn.Module):
    """Lightweight facade that provides .matrix / .bias properties pointing
    to slices of the parent ``MoEExpertCollection``'s shared 3D tensor.

    This exists purely for backward-compatible tests that do things like
    ``collection.routing_experts[i].matrix``.  It is NOT used in the
    hot forward path (``forward_expert_batched`` operates on the 3D
    tensor directly).

    NOTE: We store parent_ref via object.__setattr__ to avoid nn.Module
    registering it as a sub-module (which would cause circular parameter
    registration).
    """

    def __init__(
        self,
        parent: MoEExpertCollection,
        idx: int,
        activation_function: str,
        precision: str,
    ) -> None:
        super().__init__()
        # Use object.__setattr__ to avoid nn.Module registering parent
        # as a sub-module (circular reference).
        object.__setattr__(self, "_parent_ref", parent)
        self._idx = idx
        self.num_in = parent.num_in
        self.num_out = parent.num_out
        self.activate = ActivationFn(activation_function)

    @property
    def matrix(self) -> torch.Tensor:
        """Weight matrix W for this expert, shape ``[num_in, num_out]``."""
        return self._parent_ref.routing_matrix[:, :, self._idx]

    @property
    def bias(self) -> torch.Tensor:
        """Bias vector b for this expert, shape ``[num_out]``."""
        return self._parent_ref.routing_bias[:, self._idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: act(x @ W + b)."""
        return self.activate(
            torch.matmul(x, self.matrix) + self.bias
        )
