# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SO(2)-equivariant message-passing layers for SeZM.

This module defines the reduced-layout SO(2) linear operator and the
edge convolution used inside SeZM interaction blocks.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    get_generator,
)

from .attention import (
    segment_envelope_gated_softmax,
)
from .attn_res import (
    DepthAttnRes,
)
from .indexing import (
    build_m_major_index,
    build_m_major_l_index,
    build_rotate_inv_rescale,
    get_so3_dim_of_lmax,
    map_degree_idx,
    project_D_to_m,
    project_Dt_from_m,
)
from .norm import (
    ReducedEquivariantRMSNorm,
    ScalarRMSNorm,
)
from .so3 import (
    FocusLinear,
    GatedActivation,
    SO3Linear,
)
from .triton import (
    resolve_triton_rotation_mode,
    rotate_back_triton,
    rotate_to_local_triton,
    sezm_triton_enabled,
)
from .utils import (
    ATTN_RES_MODES,
    get_promoted_dtype,
    init_trunc_normal_fan_in_out,
    np_safe,
    nvtx_range,
    safe_numpy_to_tensor,
)

if TYPE_CHECKING:
    from .edge_cache import (
        EdgeFeatureCache,
    )


class SO2Linear(nn.Module):
    """
    SO(2)-equivariant linear mixing in the edge-aligned local frame.

    Coefficient layout (m-major, truncated by mmax)
    ------------------------------------------------
    The coefficient axis D_m_trunc is ordered by |m| groups::

        [  m=0: l=0..lmax  |  m=1: (l,-1) then (l,+1)  |  ...  |  m=mmax: ... ]
         |___ lmax+1 ____|   |_______ 2*(lmax) ________|

    Each |m| group is contiguous, enabling a single block-diagonal matmul.

    Block-diagonal weight structure
    -------------------------------
    The full weight matrix W has shape ``(F, D_m_trunc*Cout, D_m_trunc*Cin)``
    and is block-diagonal over |m| groups::

        W = diag[W_m0, B_m1, B_m2, ..., B_mmax]

    - ``W_m0``: unconstrained ``(num_l*Cout, num_l*Cin)`` block for m=0.
      Cross-l mixing is allowed since m=0 coefficients are real scalars.

    - ``B_m`` (|m|>0): SO(2)-constrained 2x2 block coupling (-m, +m) pairs::

          B_m = [ W_u^T , -W_v^T ]     where W_u, W_v are learnable
                [ W_v^T ,  W_u^T ]     (num_l*Cin, num_l*Cout) each.

      This structure is the real-valued form of complex multiplication
      ``(u + iv)(a + ib) = (ua - vb) + i(va + ub)``, which guarantees
      SO(2) equivariance: rotating the input by angle phi around z
      rotates the output by the same angle.

    The weight is assembled once per forward (training) or cached (eval)
    by ``_build_so2_weight()``, then applied via a single batched matmul
    over all focus streams: ``einsum("efi,foi->efo")``.

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum SO(2) order (|m|) to mix. If None, defaults to ``lmax``.
    in_channels
        Number of input channels per (l, m) coefficient.
    out_channels
        Number of output channels per (l, m) coefficient.
    n_focus
        Number of independent focus streams. Each stream has its own
        weight matrices; the batched matmul vectorizes over all streams.
    dtype
        Parameter dtype.
    mlp_bias
        Whether to use bias for l=0 (scalar) components.
    seed
        Random seed for weight initialization.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        in_channels: int,
        out_channels: int,
        n_focus: int = 1,
        dtype: torch.dtype,
        mlp_bias: bool = True,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.mlp_bias = bool(mlp_bias)

        # === Step 1. Build m-major coefficient layout ===
        # Map each |m| group to contiguous index ranges in the flattened axis.
        # Example for lmax=2, mmax=2:
        #   m=0 : indices [0, 1, 2]        (l=0,1,2)
        #   m=1-: indices [3, 4]            (l=1,2 with -m)
        #   m=1+: indices [5, 6]            (l=1,2 with +m)
        #   m=2-: index  [7]               (l=2   with -m)
        #   m=2+: index  [8]               (l=2   with +m)
        #   => reduced_dim = 9
        m0_size = self.lmax + 1
        self.register_buffer(
            "m0_idx",
            torch.arange(m0_size, device=self.device, dtype=torch.long),
            persistent=False,
        )

        pos_indices_list: list[torch.Tensor] = []
        neg_indices_list: list[torch.Tensor] = []
        # Each entry: (neg_start, pos_start, num_l) for a fixed |m|.
        # These ranges are contiguous in m-major layout.
        m_ranges: list[tuple[int, int, int]] = []

        offset = m0_size
        for m in range(1, self.mmax + 1):
            num_l = self.lmax - m + 1
            neg_start = offset
            pos_start = offset + num_l
            neg_idx = torch.arange(
                neg_start, neg_start + num_l, device=self.device, dtype=torch.long
            )
            pos_idx = torch.arange(
                pos_start, pos_start + num_l, device=self.device, dtype=torch.long
            )
            neg_indices_list.append(neg_idx)
            pos_indices_list.append(pos_idx)
            m_ranges.append((neg_start, pos_start, num_l))
            offset += 2 * num_l

        self.reduced_dim = int(offset)

        if len(pos_indices_list) > 0:
            self.register_buffer(
                "pos_indices", torch.cat(pos_indices_list), persistent=False
            )
            self.register_buffer(
                "neg_indices", torch.cat(neg_indices_list), persistent=False
            )
            self._m_ranges = m_ranges
        else:
            self.register_buffer(
                "pos_indices",
                torch.empty(0, device=self.device, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "neg_indices",
                torch.empty(0, device=self.device, dtype=torch.long),
                persistent=False,
            )
            self._m_ranges = []

        # === Step 2. Learnable weight parameters ===
        # weight_m0: folded (num_l*Cin, F*num_l*Cout) storage — (in, out) convention.
        #   Runtime view: (num_l*Cin, F, num_l*Cout).
        #   Cross-l mixing is allowed because m=0 coefficients are real.
        num_m0 = self.lmax + 1
        num_in_m0 = num_m0 * self.in_channels
        num_out_m0 = num_m0 * self.out_channels
        self.weight_m0 = nn.Parameter(
            torch.empty(
                num_in_m0,
                self.n_focus * num_out_m0,
                device=self.device,
                dtype=self.dtype,
            )
        )
        weight_m0_view = self.weight_m0.view(num_in_m0, self.n_focus, num_out_m0)
        for focus_idx in range(self.n_focus):
            init_trunc_normal_fan_in_out(
                weight_m0_view[:, focus_idx, :], child_seed(seed, 1000 + focus_idx)
            )
        if self.mlp_bias:
            self.bias0: nn.Parameter | None = nn.Parameter(
                torch.zeros(
                    self.n_focus * self.out_channels,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        else:
            self.bias0 = None

        # weight_m[i]: folded (num_l*Cin, F*2*num_l*Cout) storage — (in, out) convention.
        #   Runtime view: (num_l*Cin, F, 2*num_l*Cout).
        #   The factor of 2 comes from storing W_u and W_v concatenated along the
        #   output axis. _build_so2_weight() splits them and fills the 2x2 block.
        #   Scaling by 1/sqrt(2) compensates for the doubled parameter count.
        self.weight_m: nn.ParameterList = nn.ParameterList()
        for m in range(1, self.mmax + 1):
            num_l = self.lmax - m + 1
            num_in = num_l * self.in_channels
            num_out = 2 * num_l * self.out_channels
            weight = nn.Parameter(
                torch.empty(
                    num_in,
                    self.n_focus * num_out,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            weight_view = weight.view(num_in, self.n_focus, num_out)
            for focus_idx in range(self.n_focus):
                init_trunc_normal_fan_in_out(
                    weight_view[:, focus_idx, :],
                    child_seed(seed, 2000 + m * 100 + focus_idx),
                )
            # Apply scaling for SO(2) equivariance
            weight.data.mul_(1.0 / math.sqrt(2.0))
            self.weight_m.append(weight)

        for p in self.parameters():
            p.requires_grad = trainable

        # === Step 3. Precompute flattened slice ranges for _build_so2_weight ===
        # Each |m|>0 group occupies two sub-blocks (neg, pos) in the flattened
        # weight matrix. Pre-computing the row/col ranges avoids repeated
        # arithmetic in the hot path.
        # Tuple layout: (neg_i0, neg_i1, pos_i0, pos_i1,   <- input row ranges
        #                neg_o0, neg_o1, pos_o0, pos_o1)   <- output col ranges
        self._m0_in = (self.lmax + 1) * self.in_channels
        self._m0_out = (self.lmax + 1) * self.out_channels
        self._block_slices: list[tuple[int, int, int, int, int, int, int, int]] = []
        for neg_start, pos_start, num_l in self._m_ranges:
            ib = num_l * self.in_channels
            ob = num_l * self.out_channels
            self._block_slices.append(
                (
                    neg_start * self.in_channels,
                    neg_start * self.in_channels + ib,
                    pos_start * self.in_channels,
                    pos_start * self.in_channels + ib,
                    neg_start * self.out_channels,
                    neg_start * self.out_channels + ob,
                    pos_start * self.out_channels,
                    pos_start * self.out_channels + ob,
                )
            )

        # Weight cache: only used in eval + no_grad (inference).
        # Invalidated on train() via overridden method below.
        self._cached_weight: torch.Tensor | None = None

    def train(self, mode: bool = True) -> SO2Linear:
        """Invalidate weight cache when switching to training mode."""
        self._cached_weight = None
        return super().train(mode)

    def _build_so2_weight(self) -> torch.Tensor:
        """
        Assemble the per-focus block-diagonal SO(2) weight matrix.

        The flattened weight has shape ``(D_m*Cin, F, D_m*Cout)`` (in, out)
        where both axes follow the same m-major coefficient ordering.
        Off-diagonal blocks (cross-|m|) are zero, enforcing SO(2) equivariance.

        Returns
        -------
        torch.Tensor
            Weight with shape (D_m*Cin, F, D_m*Cout).
        """
        in_total = self.reduced_dim * self.in_channels
        out_total = self.reduced_dim * self.out_channels
        weight = self.weight_m0.new_zeros(in_total, self.n_focus, out_total)
        num_in_m0 = (self.lmax + 1) * self.in_channels
        num_out_m0 = (self.lmax + 1) * self.out_channels
        weight_m0 = self.weight_m0.view(num_in_m0, self.n_focus, num_out_m0)

        # m=0 block: (Cin_blk, F, Cout_blk) — (in, out) convention.
        weight[: self._m0_in, :, : self._m0_out] = weight_m0

        # |m|>0 blocks: fill the 2x2 SO(2) coupling structure.
        # For each |m|, the learnable param w has shape (in_blk, F, 2*out_blk)
        # which is split into W_u and W_v along the output axis.
        for m_idx, w in enumerate(self.weight_m):
            ni0, ni1, pi0, pi1, no0, no1, po0, po1 = self._block_slices[m_idx]
            ib = ni1 - ni0  # in_block size
            ob = no1 - no0  # out_block size
            w = w.view(ib, self.n_focus, 2 * ob)
            w_u = w[:, :, :ob]  # (in_blk, F, out_blk)
            w_v = w[:, :, ob:]  # (in_blk, F, out_blk)
            # Fill the 2x2 coupling:
            #   Row = input (neg/pos), Col = output (neg/pos).
            #   [ W_u^T, -W_v^T ]^T  =>  row=neg_in: W_u to neg_out, W_v to pos_out
            #   [ W_v^T,  W_u^T ]^T  =>  row=pos_in: -W_v to neg_out, W_u to pos_out
            weight[ni0:ni1, :, no0:no1] = w_u  # neg_in -> neg_out
            weight[ni0:ni1, :, po0:po1] = w_v  # neg_in -> pos_out
            weight[pi0:pi1, :, no0:no1] = -w_v  # pos_in -> neg_out
            weight[pi0:pi1, :, po0:po1] = w_u  # pos_in -> pos_out
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input with shape (E, F, D_m_trunc, Cin), where D_m_trunc is the
            coefficient dimension of the m-major layout truncated by `mmax`.

        Returns
        -------
        torch.Tensor
            Output with shape (E, F, D_m_trunc, Cout), where Cout is output channels.
        """
        # === Step 1. Flatten coefficient + channel axes for matmul ===
        # (E, F, D_m, Cin) -> (E, F, D_m*Cin)
        n_edge = x.shape[0]
        in_dim_total = self.reduced_dim * self.in_channels
        x_flat = x.reshape(n_edge, self.n_focus, in_dim_total)

        # === Step 2. Get block-diagonal weight (cached in eval+no_grad) ===
        if self._cached_weight is not None:
            weight = self._cached_weight
        else:
            weight = self._build_so2_weight()
            # Cache only in eval mode with grad disabled (pure inference).
            if not self.training and not torch.is_grad_enabled():
                self._cached_weight = weight.detach()

        # === Step 3. Batched matmul over focus streams + reshape back ===
        # einsum "efi,ifo->efo": (E,F,D_m*Cin) x (D_m*Cin,F,D_m*Cout) -> (E,F,D_m*Cout)
        out_flat = torch.einsum("efi,ifo->efo", x_flat, weight)
        out = out_flat.reshape(
            n_edge, self.n_focus, self.reduced_dim, self.out_channels
        )

        # === Step 4. Bias on l=0 scalar index ===
        if self.mlp_bias:
            bias0 = self.bias0.view(self.n_focus, self.out_channels)
            out[:, :, 0, :] = out[:, :, 0, :] + bias0.unsqueeze(0)
        return out

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "SO2Linear",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "n_focus": self.n_focus,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "mlp_bias": self.mlp_bias,
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SO2Linear:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SO2Linear":
            raise ValueError(f"Invalid class for SO2Linear: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported SO2Linear version: {version}")
        config = data.pop("config")
        variables = data.pop("@variables")
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
        return obj


class SO2Convolution(nn.Module):
    """
    SO(2)-equivariant edge convolution with cached geometry and rotations.

    This module consumes node features in packed SO(3) layout `(N, D, C)` and
    performs edge message passing in the reduced m-major local layout. The
    operation pipeline is:

    1. `pre_focus_mix`: project node features `(N, D, C)` to the SO(2) hidden width.
    2. rotate global -> local reduced basis with cached `D_to_m`.
    3. radial modulation in reduced layout.
    4. `so2_layers` stacked local mixers:
       `inter_norm -> SO2Linear -> non_linearity -> residual(+LayerScale)`.
    5. rotate local -> global with cached `Dt_from_m`.
    6. edge aggregation (plain envelope scatter or envelope-aware grouped
       softmax attention with exact envelope-gated competition and
       output-side head gate).
    7. `post_focus_mix`: project aggregated hidden messages back to `(N, D, C)`.

    Equivariance is preserved because both `pre_focus_mix` and `post_focus_mix`
    only mix the channel axis for each `(l, m)` coefficient and never mix
    coefficient indices across `(l, m)`.

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum SO(2) order (|m|). If None, defaults to lmax.
    channels
        Number of channels per (l, m) coefficient.
    n_focus
        Number of focus streams inside the SO(2) branch.
    focus_dim
        Hidden width per focus stream inside SO(2).
        ``focus_dim=0`` means using ``channels``.
    focus_compete
        If True, apply cross-focus softmax competition in SO(2) local layout.
        Competition logits are constructed only from l=0 scalar channels and the
        resulting invariant weights are broadcast to all (l, m) components.
    so2_norm
        If True, apply intermediate ReducedEquivariantRMSNorm as pre-norm before
        each SO(2) mixing layer. The last SO(2) layer always uses Identity.
    so2_layers
        Number of SO2Linear layers per convolution (default: 1).
    so2_attn_res
        Depth-wise attention residual mode across the internal SO(2) layer
        history. Must be one of ``"none"``, ``"independent"``, or
        ``"dependent"``. The same scalar weights are broadcast to the full
        reduced equivariant tensor.
    layer_scale
        If True, apply per-layer learnable LayerScale (per-focus-channel,
        init 1e-3) on each SO(2) residual branch.
    n_atten_head
        Number of attention heads used during aggregation.
        - 0: plain envelope-weighted scatter-sum.
        - >0: envelope-gated grouped softmax attention with output-side head
          gates. Attention uses ``w**2 * exp(logit)`` in the numerator and
          ``zeta + sum(w**2 * exp(logit))`` in the denominator.
        Requires the effective per-focus width to satisfy
        ``focus_dim % n_atten_head == 0``.
    mlp_bias
        Whether to use bias in SO2Linear (l=0 bias) and GatedActivation
        (gate linear bias).
    use_triton
        If True, opt into fused Triton SO(2) rotation kernels on supported
        CUDA dtypes. The eager projection path remains the default.
    eps
        Small epsilon for normalization modules.
    dtype
        Parameter dtype.
    seed
        Random seed for weight initialization.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        channels: int,
        n_focus: int = 1,
        focus_dim: int = 0,
        focus_compete: bool = True,
        so2_norm: bool = False,
        so2_layers: int = 4,
        so2_attn_res: str = "none",
        layer_scale: bool = False,
        n_atten_head: int = 0,
        mlp_bias: bool = True,
        use_triton: bool = False,
        eps: float = 1e-7,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        if self.n_focus < 1:
            raise ValueError("`n_focus` must be >= 1")
        self.focus_dim = int(focus_dim)
        if self.focus_dim < 0:
            raise ValueError("`focus_dim` must be >= 0")
        self.so2_focus_dim = self.channels if self.focus_dim == 0 else self.focus_dim
        self.hidden_channels = int(self.n_focus * self.so2_focus_dim)
        self.use_hidden_projection = self.hidden_channels != self.channels
        self.focus_compete = bool(focus_compete)
        self.focus_softmax_tau = 1.0
        self.focus_label_smoothing = 0.02
        self.so2_norm = bool(so2_norm)
        self.so2_layers = int(so2_layers)
        if self.so2_layers < 1:
            raise ValueError("`so2_layers` must be >= 1")
        self.so2_attn_res_mode = str(so2_attn_res).lower()
        if self.so2_attn_res_mode not in ATTN_RES_MODES:
            raise ValueError(
                "`so2_attn_res` must be one of 'none', 'independent', or 'dependent'"
            )
        self.use_so2_attn_res = self.so2_attn_res_mode != "none"
        self.layer_scale = bool(layer_scale)
        self.n_atten_head = int(n_atten_head)
        if self.n_atten_head < 0:
            raise ValueError("`n_atten_head` must be non-negative")
        if self.n_atten_head > 0 and self.so2_focus_dim % self.n_atten_head != 0:
            raise ValueError(
                "`focus_dim` must be divisible by `n_atten_head` when attention is enabled"
            )
        self.head_dim = (
            None
            if self.n_atten_head == 0
            else int(self.so2_focus_dim // self.n_atten_head)
        )
        self.mlp_bias = bool(mlp_bias)
        self.use_triton = bool(use_triton)
        self.eps = float(eps)
        self.ebed_dim_full = get_so3_dim_of_lmax(self.lmax)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.compute_dtype = get_promoted_dtype(self.dtype)
        self.use_triton_rotations = self.use_triton and sezm_triton_enabled(
            device=self.device,
            dtype=self.dtype,
        )

        # === Step 1. Precompute coefficient indices for m-major reduced layout ===
        coeff_index_m = build_m_major_index(self.lmax, self.mmax, device=self.device)
        degree_index_m = build_m_major_l_index(self.lmax, self.mmax, device=self.device)
        degree_index_full = map_degree_idx(self.lmax, device=self.device)
        rotate_inv_rescale_full = build_rotate_inv_rescale(
            lmax=self.lmax,
            mmax=self.mmax,
            degree_index=degree_index_full,
            device=self.device,
            dtype=self.dtype,
        )
        self.register_buffer("coeff_index_m", coeff_index_m, persistent=False)
        self.register_buffer("degree_index_m", degree_index_m, persistent=False)
        self.register_buffer(
            "rotate_inv_rescale_full", rotate_inv_rescale_full, persistent=False
        )
        self.reduced_dim = int(coeff_index_m.numel())
        self.triton_rotation_mode = resolve_triton_rotation_mode(
            dim_full=self.ebed_dim_full,
            reduced_dim=self.reduced_dim,
        )

        # === Step 2. Split deterministic seeds at the module top-level ===
        seed_so2_stack = child_seed(seed, 0)
        seed_non_linearities = child_seed(seed, 1)
        seed_so3_pre = child_seed(seed, 2)
        seed_so3_post = child_seed(seed, 3)
        seed_gate = child_seed(seed, 4)
        seed_depth_attn = child_seed(seed, 5)
        seed_radial_hidden = child_seed(seed, 6)

        # === Step 3. Multiple SO2Linear layers ===
        self.so2_linears = nn.ModuleList(
            [
                SO2Linear(
                    lmax=self.lmax,
                    mmax=self.mmax,
                    in_channels=self.so2_focus_dim,
                    out_channels=self.so2_focus_dim,
                    n_focus=self.n_focus,
                    dtype=self.dtype,
                    mlp_bias=self.mlp_bias,
                    seed=child_seed(seed_so2_stack, i),
                    trainable=trainable,
                )
                for i in range(self.so2_layers)
            ]
        )

        # === Step 4. Intermediate norms (Optional) ===
        inter_norms: list[nn.Module] = []
        if self.so2_norm:
            for _ in range(max(0, self.so2_layers - 1)):
                inter_norms.append(
                    ReducedEquivariantRMSNorm(
                        lmax=self.lmax,
                        mmax=self.mmax,
                        channels=self.so2_focus_dim,
                        degree_index_m=self.degree_index_m,
                        n_focus=self.n_focus,
                        centering=True,
                        dtype=self.compute_dtype,
                        trainable=trainable,
                    )
                )
        else:
            for _ in range(max(0, self.so2_layers - 1)):
                inter_norms.append(nn.Identity())
        inter_norms.append(nn.Identity())
        self.so2_inter_norms = nn.ModuleList(inter_norms)

        # === Step 5. Intermediate non-linearity ===
        non_linearities: list[nn.Module] = []
        for i in range(max(0, self.so2_layers - 1)):
            non_linearities.append(
                GatedActivation(
                    lmax=self.lmax,
                    mmax=self.mmax,
                    channels=self.so2_focus_dim,
                    n_focus=self.n_focus,
                    dtype=self.dtype,
                    mlp_bias=self.mlp_bias,
                    layout="nfdc",
                    trainable=trainable,
                    seed=child_seed(seed_non_linearities, i),
                )
            )
        non_linearities.append(nn.Identity())
        self.non_linearities = nn.ModuleList(non_linearities)

        # === Step 5.5. Optional depth-wise attention residuals across SO(2) layers ===
        if self.use_so2_attn_res:
            self.so2_layer_attn_res: nn.ModuleList | None = nn.ModuleList(
                [
                    DepthAttnRes(
                        channels=self.hidden_channels,
                        input_dependent=self.so2_attn_res_mode == "dependent",
                        eps=self.eps,
                        bias=self.mlp_bias,
                        dtype=self.compute_dtype,
                        trainable=trainable,
                        seed=child_seed(seed_depth_attn, i),
                    )
                    for i in range(self.so2_layers)
                ]
            )
        else:
            self.so2_layer_attn_res = None

        # === Step 6. Optional per-layer LayerScale for SO(2) residual branches ===
        if self.layer_scale:
            self.adam_so2_layer_scales = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.ones(
                            self.n_focus,
                            self.so2_focus_dim,
                            dtype=self.dtype,
                            device=self.device,
                        )
                        * 1e-3,
                        requires_grad=trainable,
                    )
                    for _ in range(self.so2_layers)
                ]
            )
        else:
            self.adam_so2_layer_scales = None

        # === Step 7. Optional attention projections (n_atten_head > 0) ===
        self.attn_qk_norm: ScalarRMSNorm | None = None
        self.attn_q_proj: FocusLinear | None = None
        self.attn_k_proj: FocusLinear | None = None
        self.adamw_attn_logit_w: nn.Parameter | None = None
        self.adamw_attn_z_bias_raw: nn.Parameter | None = None
        self.attn_output_gate_norm: ScalarRMSNorm | None = None
        self.adamw_attn_gate_w: nn.Parameter | None = None
        if self.n_atten_head > 0:
            self.attn_qk_norm = ScalarRMSNorm(
                channels=self.so2_focus_dim,
                n_focus=self.n_focus,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
            self.attn_q_proj = FocusLinear(
                in_channels=self.so2_focus_dim,
                out_channels=self.so2_focus_dim,
                n_focus=self.n_focus,
                dtype=self.compute_dtype,
                bias=False,
                seed=child_seed(seed_gate, 0),
                trainable=trainable,
            )
            self.attn_k_proj = FocusLinear(
                in_channels=self.so2_focus_dim,
                out_channels=self.so2_focus_dim,
                n_focus=self.n_focus,
                dtype=self.compute_dtype,
                bias=False,
                seed=child_seed(seed_gate, 1),
                trainable=trainable,
            )
            self.adamw_attn_logit_w = nn.Parameter(
                torch.empty(
                    self.so2_focus_dim,
                    self.n_focus,
                    self.n_atten_head,
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=trainable,
            )
            nn.init.normal_(
                self.adamw_attn_logit_w,
                mean=0.0,
                std=0.01,
                generator=get_generator(child_seed(seed_gate, 2)),
            )
            # softplus(0.5413) ~= 1.0 provides balanced initial competition.
            self.adamw_attn_z_bias_raw = nn.Parameter(
                torch.full(
                    (self.n_focus, self.n_atten_head),
                    0.5413,
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=trainable,
            )
            self.attn_output_gate_norm = ScalarRMSNorm(
                channels=self.so2_focus_dim,
                n_focus=self.n_focus,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
            self.adamw_attn_gate_w = nn.Parameter(
                torch.empty(
                    self.so2_focus_dim,
                    self.n_focus,
                    self.n_atten_head,
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=trainable,
            )
            nn.init.normal_(
                self.adamw_attn_gate_w,
                mean=0.0,
                std=0.01,
                generator=get_generator(child_seed(seed_gate, 3)),
            )

        # === Step 7.5. Optional cross-focus competition ===
        self.focus_compete_norm: ScalarRMSNorm | None = None
        self.adamw_focus_compete_w: nn.Parameter | None = None
        self.focus_compete_bias: nn.Parameter | None = None
        if self.focus_compete and self.n_focus > 1:
            self.focus_compete_norm = ScalarRMSNorm(
                channels=self.so2_focus_dim,
                n_focus=self.n_focus,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
            self.adamw_focus_compete_w = nn.Parameter(
                torch.empty(
                    self.so2_focus_dim,
                    self.n_focus,
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=trainable,
            )
            nn.init.normal_(
                self.adamw_focus_compete_w,
                mean=0.0,
                std=0.01,
                generator=get_generator(child_seed(seed_gate, 4)),
            )
            if self.mlp_bias:
                self.focus_compete_bias = nn.Parameter(
                    torch.zeros(
                        self.n_focus,
                        dtype=self.compute_dtype,
                        device=self.device,
                    ),
                    requires_grad=trainable,
                )

        # === Step 8. Optional radial hidden projection ===
        self.radial_hidden_proj: FocusLinear | None = None
        if self.use_hidden_projection:
            self.radial_hidden_proj = FocusLinear(
                in_channels=self.channels,
                out_channels=self.hidden_channels,
                n_focus=1,
                dtype=self.dtype,
                bias=False,
                seed=seed_radial_hidden,
                trainable=trainable,
            )

        # === Step 9. Pre-focus channel mixing ===
        # This projects the full channel width before the SO(2) focus split.
        self.pre_focus_mix = SO3Linear(
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=self.hidden_channels,
            n_focus=1,
            dtype=dtype,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3_pre,
        )

        # === Step 10. Post-focus channel mixing ===
        self.post_focus_mix = SO3Linear(
            lmax=self.lmax,
            in_channels=self.hidden_channels,
            out_channels=self.channels,
            n_focus=1,
            dtype=dtype,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3_post,
            init_std=0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_cache: EdgeFeatureCache,
        radial_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Node features with shape (N, D, C), where D=(lmax+1)^2 is the
            SO(3) coefficient dimension.
        edge_cache
            Precomputed edge cache. Must be compatible with this block's lmax.
        radial_feat
            Per-edge radial features with shape (E, lmax+1, C), already fused
            with edge type features.

        Returns
        -------
        torch.Tensor
            Message updates with shape (N, D, C).
        """
        src, dst = edge_cache.src, edge_cache.dst
        n_node = x.shape[0]
        n_edge = src.numel()

        # === Step 1. Pre-focus channel mixing on full width ===
        with nvtx_range("SO2Conv/pre_focus_mix"):
            x = self.pre_focus_mix(x.unsqueeze(2)).squeeze(2)

        # === Step 2. Rotate to edge-aligned local frame ===
        with nvtx_range("SO2Conv/rotate_to_local"):
            D_full = edge_cache.D_full
            if self.use_triton_rotations and not self.training:
                x_local = rotate_to_local_triton(
                    x=x,
                    src=src,
                    wigner=D_full,
                    coeff_index=self.coeff_index_m,
                    dim_full=self.ebed_dim_full,
                    rotation_mode=self.triton_rotation_mode,
                )  # (E, D_m, H)
            else:
                D_m_prime = project_D_to_m(
                    D_full=D_full,
                    coeff_index_m=self.coeff_index_m,
                    ebed_dim_full=self.ebed_dim_full,
                    cache=edge_cache.D_to_m_cache,
                    key_lmax=self.lmax,
                    key_mmax=self.mmax,
                )
                x_src = x.index_select(0, src)  # (E, D, H)
                x_local = torch.bmm(D_m_prime, x_src)  # (E, D_m, H)

        # === Step 3. Select radial/type features for reduced layout ===
        with nvtx_range("SO2Conv/radial_fuse"):
            rad_feat = radial_feat[:, self.degree_index_m, :]  # (E, D_m, C)
            if self.radial_hidden_proj is not None:
                rad_feat = self.radial_hidden_proj(
                    rad_feat.reshape(n_edge * self.reduced_dim, 1, self.channels)
                ).reshape(n_edge, self.reduced_dim, self.hidden_channels)
            x_local.mul_(rad_feat)
            rad_feat_l0_focus = rad_feat[:, 0, :].reshape(
                n_edge, self.n_focus, self.so2_focus_dim
            )  # (E, F, Cf)

        # === Step 4. Convert to SO(2) internal focus layout ===
        with nvtx_range("SO2Conv/reshape_for_so2"):
            x_local = x_local.reshape(
                n_edge, self.reduced_dim, self.n_focus, self.so2_focus_dim
            ).transpose(1, 2)  # (E, F, D_m, Cf), strided
            if self.focus_compete and self.n_focus > 1:
                focus_gate_src = x_local[:, :, 0, :]

        # === Step 5. Multi-layer SO(2) mixing (pre-norm + residual + LayerScale) ===
        with nvtx_range("SO2Conv/so2_layers"):

            def so2_l0_extractor(v: torch.Tensor) -> torch.Tensor:
                """Extract scalar features from SO(2) reduced layout."""
                return v[:, :, 0, :].reshape(v.shape[0], self.hidden_channels)

            if self.use_so2_attn_res:
                so2_depth_sources = [x_local]
                for layer_idx, (so2_linear, inter_norm, non_linear) in enumerate(
                    zip(self.so2_linears, self.so2_inter_norms, self.non_linearities)
                ):
                    x_local: torch.Tensor = self.so2_layer_attn_res[layer_idx](
                        sources=so2_depth_sources,
                        scalar_extractor=so2_l0_extractor,
                        current_x=x_local,
                    )
                    residual = x_local
                    x_local = inter_norm(x_local)
                    x_local = so2_linear(x_local)

                    if layer_idx == 0 and so2_linear.bias0 is not None:
                        # bias0: (F*Cf,) → (1, F, Cf) for broadcasting with (E, F, Cf)
                        bias0 = so2_linear.bias0.view(
                            self.n_focus, self.so2_focus_dim
                        ).unsqueeze(0)
                        bias_correction = bias0 * (
                            rad_feat_l0_focus * edge_cache.edge_env.reshape(-1, 1, 1)
                            - 1.0
                        )  # (E, F, Cf)
                        x_local[:, :, 0, :].add_(bias_correction)

                    x_local = non_linear(x_local)

                    if self.layer_scale:
                        scale: torch.Tensor = self.adam_so2_layer_scales[
                            layer_idx
                        ].reshape(1, self.n_focus, 1, self.so2_focus_dim)
                        x_local = residual + scale * x_local
                    else:
                        x_local = residual + x_local
                    so2_depth_sources.append(x_local - residual)
            else:
                for layer_idx, (so2_linear, inter_norm, non_linear) in enumerate(
                    zip(self.so2_linears, self.so2_inter_norms, self.non_linearities)
                ):
                    residual = x_local
                    x_local = inter_norm(x_local)
                    x_local = so2_linear(x_local)

                    if layer_idx == 0 and so2_linear.bias0 is not None:
                        # bias0: (F*Cf,) → (1, F, Cf) for broadcasting with (E, F, Cf)
                        bias0 = so2_linear.bias0.view(
                            self.n_focus, self.so2_focus_dim
                        ).unsqueeze(0)
                        bias_correction = bias0 * (
                            rad_feat_l0_focus * edge_cache.edge_env.reshape(-1, 1, 1)
                            - 1.0
                        )  # (E, F, Cf)
                        x_local[:, :, 0, :].add_(bias_correction)

                    x_local = non_linear(x_local)

                    if self.layer_scale:
                        scale = self.adam_so2_layer_scales[layer_idx].reshape(
                            1, self.n_focus, 1, self.so2_focus_dim
                        )
                        x_local = residual + scale * x_local
                    else:
                        x_local = residual + x_local

        # === Step 5.5. Cross-focus softmax competition ===
        if self.focus_compete and self.n_focus > 1:
            focus_gate_src = focus_gate_src.to(dtype=self.compute_dtype)
            focus_logits = torch.einsum(
                "efi,if->ef",
                self.focus_compete_norm(focus_gate_src),
                self.adamw_focus_compete_w,
            )
            if self.mlp_bias:
                focus_logits = focus_logits + self.focus_compete_bias.unsqueeze(0)
            alpha = torch.softmax(focus_logits / self.focus_softmax_tau, dim=1).to(
                dtype=x_local.dtype
            )
            alpha = alpha * (1.0 - self.focus_label_smoothing) + (
                self.focus_label_smoothing / float(self.n_focus)
            )
            x_local = x_local * alpha.unsqueeze(-1).unsqueeze(-1)

        # === Step 6. Restore reduced global layout for inverse rotation ===
        with nvtx_range("SO2Conv/reshape_for_rotate_back"):
            x_local = x_local.transpose(1, 2).contiguous()  # (E, D_m, F, Cf)
            x_local = x_local.reshape(
                n_edge, self.reduced_dim, self.hidden_channels
            )  # (E, D_m, H)

        # === Step 7. Rotate back to global frame ===
        with nvtx_range("SO2Conv/rotate_back"):
            Dt_full = edge_cache.Dt_full
            if self.use_triton_rotations and not self.training:
                x_message = rotate_back_triton(
                    x_local=x_local,
                    wigner=Dt_full,
                    coeff_index=self.coeff_index_m,
                    dim_full=self.ebed_dim_full,
                    rotation_mode=self.triton_rotation_mode,
                )  # (E, D, H)
            else:
                Dt_from_m = project_Dt_from_m(
                    Dt_full=Dt_full,
                    coeff_index_m=self.coeff_index_m,
                    ebed_dim_full=self.ebed_dim_full,
                    cache=edge_cache.Dt_from_m_cache,
                    key_lmax=self.lmax,
                    key_mmax=self.mmax,
                )
                x_message = torch.bmm(Dt_from_m, x_local)  # (E, D, H)
            # Reduced layouts keep only 2*mmax+1 orders for l>mmax. Applying the
            # inverse-rotation degree rescale after the global lift restores the
            # full-basis amplitude expected by the block output contract.
            x_message = x_message * self.rotate_inv_rescale_full.view(1, -1, 1)

        # === Step 8. Aggregate with optional head-wise gating ===
        with nvtx_range("SO2Conv/aggregate"):
            if self.n_atten_head == 0:
                # Baseline path: fused envelope-weighted scatter add -> degree norm
                x_message = x_message * edge_cache.edge_env.unsqueeze(-1)
                out = x.new_zeros(x.shape, dtype=self.compute_dtype)
                out.index_add_(0, dst, x_message.to(dtype=self.compute_dtype))
                out.mul_(edge_cache.inv_sqrt_deg.to(dtype=self.compute_dtype))
                out = out.to(dtype=self.dtype)  # (N, D, H)
            else:
                # === Step 8.1. Build attention logits from scalar channels ===
                compute_dtype = self.compute_dtype
                x_l0_node = x[:, 0, :].reshape(
                    n_node, self.n_focus, self.so2_focus_dim
                )  # (N, F, Cf)
                qk_input = self.attn_qk_norm(x_l0_node.to(dtype=compute_dtype))
                q_node = self.attn_q_proj(qk_input)  # (N, F, Cf)
                k_node = self.attn_k_proj(qk_input)  # (N, F, Cf)
                q_edge = q_node.index_select(0, dst).reshape(
                    n_edge, self.n_focus, self.n_atten_head, self.head_dim
                )  # (E, F, H, Dh)
                k_edge = k_node.index_select(0, src).reshape(
                    n_edge, self.n_focus, self.n_atten_head, self.head_dim
                )  # (E, F, H, Dh)
                radial_l0 = rad_feat[:, 0, :].reshape(
                    n_edge, self.n_focus, self.so2_focus_dim
                )  # (E, F, Cf)
                radial_bias = torch.einsum(
                    "efi,ifo->efo",
                    radial_l0.to(dtype=compute_dtype),
                    self.adamw_attn_logit_w,
                )  # (E, F, H)
                attn_logits = (q_edge * k_edge).sum(-1) * (self.head_dim**-0.5)
                attn_logits = attn_logits + radial_bias

                # === Step 8.2. Destination-wise stable envelope-gated softmax ===
                attn_alpha = segment_envelope_gated_softmax(
                    logits=attn_logits,
                    edge_env=edge_cache.edge_env.to(dtype=compute_dtype),
                    dst=dst,
                    n_nodes=n_node,
                    z_bias_raw=self.adamw_attn_z_bias_raw,
                    eps=self.eps,
                )  # (E, F, H)

                # === Step 8.3. Head-wise value aggregation ===
                value_heads = x_message.reshape(
                    n_edge,
                    self.ebed_dim_full,
                    self.n_focus,
                    self.n_atten_head,
                    self.head_dim,
                ).to(dtype=compute_dtype)  # (E, D, F, H, Dh)
                weighted_value = value_heads * attn_alpha.reshape(
                    n_edge, 1, self.n_focus, self.n_atten_head, 1
                )
                out_heads = torch.zeros(
                    n_node,
                    self.ebed_dim_full,
                    self.n_focus,
                    self.n_atten_head,
                    self.head_dim,
                    device=x.device,
                    dtype=compute_dtype,
                )  # (N, D, F, H, Dh)
                out_heads.index_add_(0, dst, weighted_value)

                # === Step 8.4. Output-side head gate (G1 style) ===
                attn_output_gate = torch.sigmoid(
                    torch.einsum(
                        "nfi,ifo->nfo",
                        self.attn_output_gate_norm(x_l0_node.to(dtype=compute_dtype)),
                        self.adamw_attn_gate_w,
                    )
                )  # (N, F, H)
                out_heads = out_heads * attn_output_gate.reshape(
                    n_node, 1, self.n_focus, self.n_atten_head, 1
                )  # (N, D, F, H, Dh)

                # === Step 8.5. Merge heads (softmax path has no degree norm) ===
                out = out_heads.reshape(
                    n_node, self.ebed_dim_full, self.hidden_channels
                ).to(dtype=self.dtype)  # (N, D, H)

        # === Step 9. Final channel mixing ===
        with nvtx_range("SO2Conv/post_focus_mix"):
            out = self.post_focus_mix(out.unsqueeze(2)).squeeze(2)
        return out  # (N, D, C)

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "SO2Convolution",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "channels": self.channels,
                "n_focus": self.n_focus,
                "focus_dim": self.focus_dim,
                "focus_compete": self.focus_compete,
                "so2_norm": self.so2_norm,
                "so2_layers": self.so2_layers,
                "so2_attn_res": self.so2_attn_res_mode,
                "layer_scale": self.layer_scale,
                "n_atten_head": self.n_atten_head,
                "mlp_bias": self.mlp_bias,
                "eps": self.eps,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SO2Convolution:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SO2Convolution":
            raise ValueError(f"Invalid class for SO2Convolution: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported SO2Convolution version: {version}")
        config = data.pop("config")
        variables = data.pop("@variables")
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
        return obj
