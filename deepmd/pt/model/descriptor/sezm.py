# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SeZM: The descriptor of smooth equivariant Zone-bridging Model.

PyTorch backend

This implementation is designed around two non-negotiables:

1) Conservative forces: the descriptor is computed from differentiable energy.
2) Speed-first inference: edge geometry and Wigner-D rotation blocks are computed
   exactly once per `forward()` and reused by all interaction blocks.

Shared descriptor building blocks are re-exported by `sezm_nn/__init__.py`.

Runtime flow at a glance:
1) Build edge cache and radial features once.
2) Run interaction blocks with shared geometric caches.
3) Return scalar (`l=0`) descriptor channels for fitting.

Layout notes
------------
- Node-level backbone features use contiguous `(N, D, F, Cf)` where
  `D=(lmax+1)^2`, `F=n_focus`, `Cf=channels//n_focus`.
- Node-level equivariant operators use `(N, D, F, Cf)` convention.
- Edge-level SO(2) internal operators keep m-major reduced layout
  `(E, F, D_m_trunc, Cf)`.
"""

from __future__ import (
    annotations,
)

import math
from contextlib import (
    contextmanager,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.nn as nn
from einops import (
    rearrange,
)

from deepmd.dpmodel.utils import EnvMat as DPEnvMat
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
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .sezm_nn import (
    ATTN_RES_MODES,
    C3CutoffEnvelope,
    DepthAttnRes,
    EdgeFeatureCache,
    EnvironmentInitialEmbedding,
    EquivariantFFN,
    GeometricInitialEmbedding,
    InnerClamp,
    RadialBasis,
    RadialMLP,
    ScalarRMSNorm,
    SeZMInteractionBlock,
    SeZMTypeEmbedding,
    WignerDCalculator,
    build_edge_cache,
    build_edge_cache_from_edges,
    edge_cache_to_dtype,
    get_promoted_dtype,
    get_so3_dim_of_lmax,
    np_safe,
    nvtx_range,
    safe_numpy_to_tensor,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
    )

    from deepmd.utils.data_system import (
        DeepmdDataSystem,
    )
    from deepmd.utils.path import (
        DPPath,
    )


@BaseDescriptor.register("SeZM")
@BaseDescriptor.register("se_zm")
class DescrptSeZM(BaseDescriptor, nn.Module):
    """
    SeZM: The descriptor of smooth equivariant Zone-bridging Model for DeePMD-kit.

    Execution outline
    -----------------
    1. Build a per-forward `EdgeFeatureCache` (geometry, envelope, Wigner-D).
    2. Build radial/type edge features once and reuse across blocks.
    3. Run `SeZMInteractionBlock` stack with optional l/m schedules.
    4. Extract scalar channels and apply the final scalar FFN.

    Parameters
    ----------
    ntypes
        Number of element types.
    sel
        Maximum number of neighbors per type within `rcut`.
        - int: broadcast to all types, e.g. sel=100 with ntypes=2 → [100, 100]
        - list[int]: sel[i] is the maximum number of type i atoms within `rcut`
    rcut
        Cutoff radius in Å.
    env_exp
        C^3 cutoff envelope exponents `[rbf_env_exp, edge_env_exp]`.
        - `rbf_env_exp`: Controls radial basis function envelope decay.
        - `edge_env_exp`: Controls message passing edge weight envelope decay.
        Larger values give weaker suppression (values stay near 1.0 longer).
    channels
        Total channels per (l,m) coefficient.
    n_radial
        Number of radial basis functions.
    radial_mlp
        Hidden layer sizes for radial networks. An output layer of size
        `(l_schedule[0]+1)*channels` will be automatically appended.
    use_env_seed
        If True, apply environment matrix initial embedding as FiLM conditioning
        on l=0 features using 4D `[s, s*r_hat]` representation. FiLM deltas are
        normalized and scaled with learnable strengths initialized to small values.
        Internal dimensions are derived from `channels`:
        `embed_dim=min(channels, 128)`,
        `axis_dim=min(4 if embed_dim < 64 else 8, embed_dim-1)`,
        `type_dim=clamp(channels//4, 8, 32)`,
        `rbf_out_dim=max(32, embed_dim-2*type_dim)`,
        `hidden_dim=min(256, max(2*embed_dim, rbf_out_dim+2*type_dim))`.
    random_gamma
        If True, apply a random roll about the edge-aligned local ``+Z`` axis
        before building the Wigner-D blocks. The roll is sampled independently
        per edge and per forward call.
    lmax
        Maximum degree, only used when `l_schedule` is None.
    l_schedule
        Pyramid schedule of lmax per block, e.g. [3, 3, 2]. Must be non-increasing.
        If set, lmax and n_blocks will be ignored.
    mmax
        Maximum SO(2) order (|m|), only used when `m_schedule` is None.
        If None, defaults to the per-block `lmax` (i.e. `m_schedule = l_schedule`).
    m_schedule
        Schedule of mmax per block, e.g. [2, 2, 1, 0]. Must satisfy
        `m_schedule[i] <= l_schedule[i]` for every block. A non-increasing schedule is
        recommended but not required. If set, `mmax` will be ignored.
    n_blocks
        Number of blocks (only used when `l_schedule` is None).
    so2_norm
        If True, apply intermediate ReducedEquivariantRMSNorm between SO(2) mixing layers.
        When False (default), no normalization is applied between layers.
    so2_layers
        Number of SO(2) mixing layers per block.
    so2_attn_res
        SO(2)-internal depth-wise attention residual mode inside each interaction
        block. Must be one of ``"none"``, ``"independent"``, or ``"dependent"``.
    n_focus
        Number of parallel focus streams. The per-stream channel width is
        ``focus_dim = channels // n_focus``. Must divide ``channels`` exactly.
    focus_compete
        If True, enable cross-focus softmax competition inside SO(2) convolution.
        Competition logits are built from l=0 scalar channels before SO(2) mixing
        and applied after SO(2) stack to scale full irreps uniformly per focus.
    n_atten_head
        Number of attention heads when aggregating messages in SO(2) convolution.
        0 applies a plain envelope-weighted scatter-sum; >0 enables
        envelope-gated grouped softmax attention with output-side head gate.
        Attention uses ``w**2 * exp(logit)`` in the numerator and
        ``zeta + sum(w**2 * exp(logit))`` in the denominator.
        When enabled, the per-focus stream width
        ``focus_dim = channels // n_focus`` must be divisible by ``n_atten_head``.
    ffn_neurons
        Hidden sizes for the equivariant FFN in each block and the final scalar output FFN.
    ffn_blocks
        Number of FFN subblocks per interaction block.
    sandwich_norm
        Pre/post-norm switches for [SO(2), FFN] residual branches in order:
        [so2_pre, so2_post, ffn_pre, ffn_post], shared across all blocks.
    mlp_bias
        Whether to use bias in equivariant layers. When False, removes bias from:
        - SO3Linear: l=0 bias
        - SO2Linear: l=0 bias
        - GatedActivation: gate linear bias
        - DepthAttnRes: input-dependent query projection
        - EnvironmentInitialEmbedding:
          rbf_proj_layer1/2 and g_layer1/2
        Attention projections in SO2Convolution
        (attn_radial_logit_proj, attn_output_gate_proj) are always bias-free.
    layer_scale
        If True, apply learnable LayerScale (init 1e-3) on residual branches:
        - SO(2) branch: per-focus-channel scales `(n_focus, focus_dim)`
          on each SO(2) mixing layer.
        - FFN branch: per-channel scales `(channels,)` on each FFN subblock.
    full_attn_res
        Descriptor-level full attention residual mode over the unit history
        `[x0, so2_0, ffn_0_0, ffn_0_1, ..., so2_1, ffn_1_0, ffn_1_1, ...]`,
        where each FFN subblock contributes its own completed unit
        representation. `independent` uses learned query vectors, while
        `dependent` derives queries from the current SeZM state before the
        SO(2) unit, before each FFN unit, and before the final aggregation.
        Must be one of ``"none"``, ``"independent"``, or ``"dependent"``.
    block_attn_res
        Descriptor-level block attention residual mode over the block history
        `[x0, b1, b2, ...]`, where each `b_i` is the sum of all unit outputs
        inside one `SeZMInteractionBlock`. `independent` uses learned query
        vectors, while `dependent` derives queries from the current SeZM state
        before the SO(2) unit, before each FFN unit, and before the final block
        aggregation. Must be one of ``"none"``, ``"independent"``, or
        ``"dependent"``. Cannot be enabled together with `full_attn_res`.
    activation_function
        Activation function used by deepmd EmbeddingNet.
    glu_activation
        If True, use GLU-style gating in FFN (e.g., silu -> swiglu, gelu -> geglu).
    use_amp
        If True, use automatic mixed precision (AMP) with bfloat16 on CUDA.
        This does not provide accelerations under fp32 precision but will decrease
        the memory usage, while persevering model accuracy.
    use_triton
        If True, opt into the fused Triton SO(2) rotation kernels on supported
        CUDA dtypes. The default is False because the current Triton rotation
        path is not consistently faster than the eager reference path.
    exclude_types
        List of excluded type pairs.
    precision
        Precision for neural network parameters and computations. Geometry computations
        (edge distances, Wigner-D matrices, rotations, GIE) always run in fp32+ to
        provide accurate geometric information for better convergence. Only the
        interaction blocks use this precision.
    eps
        Small epsilon for numerical stability in division and normalization.
    trainable
        Whether parameters are trainable.
    seed
        Random seed(s).
    type_map
        Type names.

    Notes
    -----
    SeZM does not use the traditional environment matrix (r, a_x, a_y, a_z).
    Instead, it uses radial basis functions and spherical harmonics directly.
    The mean/stddev statistics are kept for interface compatibility but are not
    actively used in the forward pass.
    """

    _ENV_DIM: int = 1  # Use se_r style (radial only) for EnvMatStatSe compatibility

    def __init__(
        self,
        ntypes: int,
        sel: list[int] | int,
        rcut: float,
        env_exp: list[int] | None = None,
        channels: int = 64,
        n_radial: int = 10,
        radial_mlp: list[int] | None = None,
        use_env_seed: bool = True,
        random_gamma: bool = False,
        lmax: int = 2,
        l_schedule: list[int] | None = None,
        mmax: int | None = None,
        m_schedule: list[int] | None = None,
        n_blocks: int = 2,
        so2_norm: bool = False,
        so2_layers: int = 4,
        so2_attn_res: str = "none",
        n_focus: int = 1,
        focus_compete: bool = True,
        n_atten_head: int = 0,
        ffn_neurons: int = 96,
        ffn_blocks: int = 1,
        sandwich_norm: list[bool] | None = None,
        mlp_bias: bool = True,
        layer_scale: bool = False,
        full_attn_res: str = "none",
        block_attn_res: str = "none",
        activation_function: str = "silu",
        glu_activation: bool = True,
        use_amp: bool = True,
        use_triton: bool = False,
        exclude_types: list[tuple[int, int]] | None = None,
        precision: str = "float32",
        eps: float = 1e-7,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        type_map: list[str] | None = None,
        inner_clamp_r_inner: float | None = None,
        inner_clamp_r_outer: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.rcut = float(rcut)
        if env_exp is None:
            env_exp = [7, 5]
        if len(env_exp) != 2:
            raise ValueError(
                "`env_exp` must be a list of two integers: [rbf_env_exp, edge_env_exp]"
            )
        self.env_exp = [int(x) for x in env_exp]
        self.eps = float(eps)

        if isinstance(sel, int):
            sel = [sel]
        self.ntypes = int(ntypes)
        self.sel = [int(x) for x in sel]
        self.type_map = type_map
        self.nnei = int(sum(self.sel))
        self.ndescrpt = int(self.nnei * self._ENV_DIM)

        self.channels = int(channels)
        self.n_focus = int(n_focus)
        if self.n_focus < 1:
            raise ValueError("`n_focus` must be >= 1")
        if self.channels % self.n_focus != 0:
            raise ValueError(
                f"`channels` ({self.channels}) must be divisible by `n_focus` ({self.n_focus})"
            )
        self.focus_dim = self.channels // self.n_focus
        self.focus_compete = bool(focus_compete)
        self.n_radial = int(n_radial)
        if radial_mlp is None:
            radial_mlp = [64]
        self.radial_mlp = list(radial_mlp)
        if sandwich_norm is None:
            sandwich_norm = [True, False, True, False]
        if not isinstance(sandwich_norm, (list, tuple)) or len(sandwich_norm) != 4:
            raise ValueError(
                "sandwich_norm must be a list[bool] of length 4: [so2_pre, so2_post, ffn_pre, ffn_post]"
            )
        self.sandwich_norm = [bool(x) for x in sandwich_norm]
        self.so2_pre_norm = self.sandwich_norm[0]
        self.so2_post_norm = self.sandwich_norm[1]
        self.ffn_pre_norm = self.sandwich_norm[2]
        self.ffn_post_norm = self.sandwich_norm[3]
        self.activation_function = str(activation_function)
        self.glu_activation = bool(glu_activation)
        self.precision = str(precision)
        self.dtype = PRECISION_DICT[self.precision]
        self.device = env.DEVICE
        self.compute_dtype = get_promoted_dtype(self.dtype)
        self.mlp_bias = bool(mlp_bias)
        self.layer_scale = bool(layer_scale)
        self.use_amp = bool(use_amp)  # and self.training
        self.use_triton = bool(use_triton)
        self.trainable = bool(trainable)
        self.seed = seed
        self.random_gamma = bool(random_gamma)

        # === Inner clamping for zone bridging ===
        self.inner_clamp_r_inner = (
            float(inner_clamp_r_inner) if inner_clamp_r_inner is not None else None
        )
        self.inner_clamp_r_outer = (
            float(inner_clamp_r_outer) if inner_clamp_r_outer is not None else None
        )
        if (
            self.inner_clamp_r_inner is not None
            and self.inner_clamp_r_outer is not None
        ):
            self.inner_clamp: InnerClamp | None = InnerClamp(
                self.inner_clamp_r_inner, self.inner_clamp_r_outer
            )
        else:
            self.inner_clamp = None

        # === Env seed parameters ===
        self.use_env_seed = bool(use_env_seed)
        self.env_seed_embed_dim = min(self.channels, 128)
        self.env_seed_type_dim = min(32, max(8, self.channels // 4))
        axis_dim = 4 if self.env_seed_embed_dim < 64 else 8
        self.env_seed_axis_dim = min(axis_dim, max(1, self.env_seed_embed_dim - 1))
        rbf_out_dim = max(32, self.env_seed_embed_dim - 2 * self.env_seed_type_dim)
        g_in_dim = rbf_out_dim + 2 * self.env_seed_type_dim
        self.env_seed_hidden_dim = min(256, max(2 * self.env_seed_embed_dim, g_in_dim))

        # === Split deterministic seeds at the descriptor top-level ===
        seed_type_embedding = child_seed(self.seed, 0)
        seed_blocks = child_seed(self.seed, 1)
        seed_out = child_seed(self.seed, 2)
        seed_radial_embedding = child_seed(self.seed, 3)
        seed_env_seed = child_seed(self.seed, 4)
        seed_full_attn = child_seed(self.seed, 5)
        seed_block_attn = child_seed(self.seed, 6)

        # === L/M schedules ===
        self._init_lm_schedules(lmax, n_blocks, l_schedule, mmax, m_schedule)
        self.ebed_dims = [get_so3_dim_of_lmax(l) for l in self.l_schedule]
        self.rad_sizes_per_block = [l + 1 for l in self.l_schedule]

        self.so2_norm = bool(so2_norm)
        self.so2_layers = int(so2_layers)
        self.so2_attn_res_mode = str(so2_attn_res).lower()
        if self.so2_attn_res_mode not in ATTN_RES_MODES:
            raise ValueError(
                "`so2_attn_res` must be one of 'none', 'independent', or 'dependent'"
            )
        self.ffn_neurons = int(ffn_neurons)
        self.ffn_blocks = int(ffn_blocks)
        if self.ffn_blocks < 1:
            raise ValueError("`ffn_blocks` must be >= 1")
        self.full_attn_res_mode = str(full_attn_res).lower()
        if self.full_attn_res_mode not in ATTN_RES_MODES:
            raise ValueError(
                "`full_attn_res` must be one of 'none', 'independent', or 'dependent'"
            )
        self.block_attn_res_mode = str(block_attn_res).lower()
        if self.block_attn_res_mode not in ATTN_RES_MODES:
            raise ValueError(
                "`block_attn_res` must be one of 'none', 'independent', or 'dependent'"
            )
        self.use_full_attn_res = self.full_attn_res_mode != "none"
        self.use_block_attn_res = self.block_attn_res_mode != "none"
        if self.use_full_attn_res and self.use_block_attn_res:
            raise ValueError(
                "`full_attn_res` and `block_attn_res` cannot both be enabled"
            )
        self.n_atten_head = int(n_atten_head)
        if self.n_atten_head > 0 and self.focus_dim % self.n_atten_head != 0:
            raise ValueError(
                "`focus_dim` must be divisible by `n_atten_head` when attention is enabled"
            )

        # === Excluded type pairs ===
        self.reinit_exclude(exclude_types)

        # === Type embedding ===
        self.type_embedding = SeZMTypeEmbedding(
            ntypes=self.ntypes,
            embed_dim=self.channels,
            dtype=self.compute_dtype,  # force fp32+
            seed=seed_type_embedding,
            trainable=self.trainable,
        )

        # === Env FiLM embedding (optional) ===
        if self.use_env_seed:
            self.env_seed_embedding: EnvironmentInitialEmbedding | None = (
                EnvironmentInitialEmbedding(
                    ntypes=self.ntypes,
                    n_radial=self.n_radial,
                    channels=self.channels,
                    embed_dim=self.env_seed_embed_dim,
                    axis_dim=self.env_seed_axis_dim,
                    type_dim=self.env_seed_type_dim,
                    hidden_dim=self.env_seed_hidden_dim,
                    mlp_bias=self.mlp_bias,
                    activation_function=self.activation_function,
                    eps=self.eps,
                    dtype=self.compute_dtype,  # force fp32+
                    trainable=self.trainable,
                    seed=seed_env_seed,
                )
            )
            self.film_scale_norm = ScalarRMSNorm(
                channels=self.channels,
                n_focus=1,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=self.trainable,
            )
            self.film_shift_norm = ScalarRMSNorm(
                channels=self.channels,
                n_focus=1,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=self.trainable,
            )
            film_strength_init = 0.01
            # Use 1D tensor (not scalar) for FSDP2 compatibility
            self.film_scale_strength_log = nn.Parameter(
                torch.full(
                    (1,),
                    math.log(film_strength_init),
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=self.trainable,
            )
            self.film_shift_strength_log = nn.Parameter(
                torch.full(
                    (1,),
                    math.log(film_strength_init),
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=self.trainable,
            )
        else:
            self.env_seed_embedding = None
            self.film_scale_norm = None
            self.film_shift_norm = None
            self.film_scale_strength_log = None
            self.film_shift_strength_log = None

        self.radial_basis = RadialBasis(
            self.rcut,
            self.n_radial,
            dtype=self.compute_dtype,  # force fp32+
            exponent=self.env_exp[0],
        )

        # === Shared radial embedding: RBF -> per-l radial features ===
        # Output dimension is (lmax+1)*channels, directly usable by GIE and SO2Conv.
        # radial_mlp specifies hidden layer sizes; input/output layers are prepended/appended.
        # Use fp32+ precision (same as RBF output) for numerical stability.
        radial_out_dim = (self.lmax + 1) * self.channels
        radial_mlp_layers = [self.n_radial, *self.radial_mlp, radial_out_dim]
        self.radial_embedding = RadialMLP(
            radial_mlp_layers,
            activation_function=self.activation_function,
            dtype=self.compute_dtype,  # force fp32+
            trainable=self.trainable,
            seed=seed_radial_embedding,
        )

        # === C^3 cutoff envelope for edge weight ===
        self.edge_envelope = C3CutoffEnvelope(rcut=self.rcut, exponent=self.env_exp[1])

        wigner_lmax = self.l_schedule[0]
        # force fp32+
        self.wigner_calc = WignerDCalculator(
            lmax=wigner_lmax,
            eps=self.eps,
            dtype=self.compute_dtype,
        )

        self.use_gie = self.l_schedule[0] > 0
        if self.use_gie:
            self.gie = GeometricInitialEmbedding(
                lmax=self.l_schedule[0],
                channels=self.channels,
                dtype=self.compute_dtype,  # force fp32+
            )
        else:
            self.gie = None

        blocks: list[SeZMInteractionBlock] = []
        for block_idx, (l_b, m_b) in enumerate(zip(self.l_schedule, self.m_schedule)):
            blocks.append(
                SeZMInteractionBlock(
                    lmax=l_b,
                    mmax=m_b,
                    channels=self.channels,
                    n_focus=self.n_focus,
                    focus_compete=self.focus_compete,
                    so2_norm=self.so2_norm,
                    so2_layers=self.so2_layers,
                    so2_attn_res=self.so2_attn_res_mode,
                    ffn_neurons=self.ffn_neurons,
                    ffn_blocks=self.ffn_blocks,
                    layer_scale=self.layer_scale,
                    full_attn_res=self.full_attn_res_mode,
                    block_attn_res=self.block_attn_res_mode,
                    n_atten_head=self.n_atten_head,
                    so2_pre_norm=self.so2_pre_norm,
                    so2_post_norm=self.so2_post_norm,
                    ffn_pre_norm=self.ffn_pre_norm,
                    ffn_post_norm=self.ffn_post_norm,
                    activation_function=self.activation_function,
                    glu_activation=self.glu_activation,
                    mlp_bias=self.mlp_bias,
                    use_triton=self.use_triton,
                    eps=self.eps,
                    dtype=self.dtype,
                    seed=child_seed(seed_blocks, block_idx),
                    trainable=self.trainable,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # === Optional descriptor-level attention residuals ===
        self.final_block_attn_res = None
        if self.use_full_attn_res:
            self.final_full_attn_res: DepthAttnRes | None = DepthAttnRes(
                channels=self.channels,
                input_dependent=self.full_attn_res_mode == "dependent",
                eps=self.eps,
                bias=self.mlp_bias,
                dtype=self.compute_dtype,
                trainable=self.trainable,
                seed=child_seed(seed_full_attn, 2000),
            )
        else:
            self.final_full_attn_res = None
        if self.use_block_attn_res:
            self.final_block_attn_res: DepthAttnRes | None = DepthAttnRes(
                channels=self.channels,
                input_dependent=self.block_attn_res_mode == "dependent",
                eps=self.eps,
                bias=self.mlp_bias,
                dtype=self.compute_dtype,
                trainable=self.trainable,
                seed=child_seed(seed_block_attn, 2000),
            )

        # === Final FFN for l=0 output mixing ===
        self.output_ffn = EquivariantFFN(
            lmax=0,
            channels=self.channels,
            hidden_channels=self.ffn_neurons,
            dtype=self.compute_dtype,
            activation_function=self.activation_function,
            glu_activation=self.glu_activation,
            mlp_bias=self.mlp_bias,
            trainable=self.trainable,
            seed=seed_out,
        )

        for p in self.parameters():
            p.requires_grad = self.trainable

        # Pre-allocate empty tensor for interface compatibility (torch.compile + DDP)
        self.register_buffer(
            "_empty_tensor",
            torch.empty(0, device=env.DEVICE, dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            persistent=False,
        )

        # === Statistics buffers (interface compatibility) ===
        self.stats: dict[str, Any] | None = None
        self.register_buffer(
            "mean",
            torch.zeros(0, dtype=self.dtype, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "stddev",
            torch.ones(0, dtype=self.dtype, device=self.device),
            persistent=False,
        )

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_vec: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        fparam: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Compute the descriptor.

        Parameters
        ----------
        extended_coord
            Extended coordinates of atoms with shape (nf, nall*3) or (nf, nall, 3) in Å.
        extended_atype
            Extended atom types with shape (nf, nall).
        nlist
            Neighbor list with shape (nf, nloc, nnei).
        mapping
            Extended-to-local mapping with shape (nf, nall), or None.
        edge_index
            Fixed-shape edge indices with shape (2, E). If provided, the descriptor
            uses the edge-list path and ignores `nlist` and `mapping`.
        edge_vec
            Fixed-shape edge vectors with shape (E, 3) in Å. Required when
            `edge_index` is provided.
        edge_mask
            Fixed-shape edge mask with shape (E,). Required when `edge_index`
            is provided.
        comm_dict
            Communication dictionary for parallel inference (unused).
        fparam
            Frame parameters with shape (nf, nfp). Not used by SeZM, kept for
            interface compatibility.

        Returns
        -------
        descriptor
            Descriptor with shape (nf, nloc, channels). Only l=0 is returned.
        rot_mat
            Empty tensor (not used).
        g2
            Empty tensor (not used).
        h2
            Empty tensor (not used).
        sw
            Empty tensor (not used).
        """
        if edge_index is not None:
            return self.forward_with_edges(
                extended_coord=extended_coord,
                extended_atype=extended_atype,
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_mask=edge_mask,
            )

        # === Step 1. Setup dimensions ===
        extended_coord = extended_coord.to(self.compute_dtype)
        nf, nloc, nnei = nlist.shape
        if extended_coord.ndim == 2:
            extended_coord = rearrange(extended_coord, "nf (nall c) -> nf nall c", c=3)
        elif extended_coord.ndim != 3:
            raise ValueError(
                "extended_coord must have shape (nf, nall*3) or (nf, nall, 3)"
            )
        nall = extended_coord.shape[1]
        n_nodes = int(nf * nloc)

        # === Step 2. Excluded type pairs ===
        if self.exclude_types:
            # (nf, nloc, nnei), True means keep.
            pair_keep_mask = self.emask(nlist, extended_atype).to(dtype=torch.bool)
        else:
            pair_keep_mask = torch.ones_like(
                nlist, dtype=torch.bool, device=self.device
            )

        # === Step 3. Type embedding (l=0) ===
        with nvtx_range("type_embedding"):
            atype_loc = extended_atype[:, :nloc]  # (nf, nloc)
            type_ebed = self.type_embedding(atype_loc).reshape(
                n_nodes, self.channels
            )  # (N, C)

        # === Step 4. Build edge cache once (geometry + RBF + Wigner-D) ===
        with nvtx_range("build_edge_cache"):
            edge_cache = build_edge_cache(
                type_ebed=type_ebed,
                extended_coord=extended_coord,
                nlist=nlist,
                mapping=mapping,
                pair_keep_mask=pair_keep_mask,
                eps=self.eps,
                inner_clamp=self.inner_clamp,
                edge_envelope=self.edge_envelope,
                radial_basis=self.radial_basis,
                n_radial=self.radial_basis.n_radial,
                random_gamma=self.random_gamma,
                wigner_calc=self.wigner_calc,
            )

        lmax_0 = self.l_schedule[0]
        ebed_dim_0 = get_so3_dim_of_lmax(lmax_0)  # (lmax+1)^2
        x0 = type_ebed  # (N, C)
        x0_out = x0  # (N, C)

        # === Step 5. Compute radial features once (fp32+) ===
        # Shape: (E, (lmax+1)*C) -> (E, lmax+1, C)
        radial_feat = None
        with nvtx_range("radial_embedding"):
            if edge_cache.src.numel() > 0:
                radial_feat = rearrange(
                    self.radial_embedding(edge_cache.edge_rbf),
                    "E (L C) -> E L C",
                    L=self.lmax + 1,
                    C=self.channels,
                )  # (E, lmax+1, C)

        # === Step 6. Env FiLM conditioning (optional, fp32+) ===
        with nvtx_range("env_film"):
            if self.use_env_seed and edge_cache.src.numel() > 0:
                atype_flat = atype_loc.reshape(-1)  # (N,)
                film = self.env_seed_embedding(
                    edge_cache=edge_cache,
                    atype_flat=atype_flat,
                    n_nodes=n_nodes,
                )  # (N, 2*C)
                scale_logits, shift_logits = film.chunk(2, dim=-1)  # (N, C), (N, C)
                # ScalarRMSNorm is unified to focus-aware layout (B, F, C).
                # Env FiLM remains a single scalar stream, so F=1 here.
                scale_hat = self.film_scale_norm(scale_logits.unsqueeze(1)).squeeze(
                    1
                )  # (N, C)
                shift_hat = self.film_shift_norm(shift_logits.unsqueeze(1)).squeeze(
                    1
                )  # (N, C)
                scale_strength = torch.exp(self.film_scale_strength_log)
                shift_strength = torch.exp(self.film_shift_strength_log)
                scale = 1.0 + scale_strength * torch.tanh(scale_hat)  # (N, C)
                shift = shift_strength * torch.tanh(shift_hat)  # (N, C)
                x0_out = x0 * scale + shift

        # === Step 7. Build l=0 features ===
        x = type_ebed.new_zeros(
            n_nodes, ebed_dim_0, self.n_focus, self.focus_dim
        )  # (N, D, F, Cf)
        x[:, 0, :, :] = x0_out.reshape(n_nodes, self.n_focus, self.focus_dim)

        # === Step 8. Geometric Initial Embedding (fp32+) ===
        with nvtx_range("gie"):
            if self.use_gie and radial_feat is not None:
                # GIE only needs l>=1, slice radial_feat[:, 1:, :]
                x = x + self.gie(
                    n_nodes=n_nodes,
                    edge_cache=edge_cache,
                    radial_feat=radial_feat[:, 1:, :],
                ).reshape(n_nodes, ebed_dim_0, self.n_focus, self.focus_dim)

        # === Step 9. Fuse edge type features into radial features (fp32+) ===
        with nvtx_range("radial_fuse"):
            if radial_feat is not None:
                radial_feat = radial_feat + rearrange(
                    edge_cache.edge_type_feat, "E C -> E 1 C"
                )
                radial_feat = radial_feat.to(dtype=self.dtype)
                rad_feat_per_block = [
                    radial_feat[:, :rad_len, :] for rad_len in self.rad_sizes_per_block
                ]  # list of (E, lmax+1, C)
            else:
                rad_feat_per_block = []

        # === Step 10. Convert to self.dtype and run blocks ===
        with nvtx_range("blocks"):
            x = x.to(dtype=self.dtype)  # (N, D, F, Cf)
            if edge_cache.src.numel() > 0:
                edge_cache = edge_cache_to_dtype(edge_cache, self.dtype)
                with self._compute_mode_ctx(extended_coord.device):
                    x = self._forward_blocks(x, edge_cache, rad_feat_per_block)

        # === Step 11. Final l=0 output mixing ===
        # Extract l=0 scalar features and apply FFN in promoted dtype.
        # Residual keeps the output close to identity with zero-initialized FFN output.
        with nvtx_range("output_ffn"):
            x_scalar = (
                x[:, 0:1, :, :]
                .reshape(n_nodes, 1, 1, self.channels)
                .to(dtype=self.compute_dtype)
            )  # (N, 1, 1, C)
            x_scalar = x_scalar + self.output_ffn(x_scalar)

        # === Step 12. Reshape to (nf, nloc, channels) and return ===
        descriptor = rearrange(
            x_scalar, "(nf nloc) 1 1 C -> nf nloc C", nf=nf, nloc=nloc
        )  # (nf, nloc, C)
        return (
            descriptor.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            self._empty_tensor,
            self._empty_tensor,
            self._empty_tensor,
            self._empty_tensor,
        )

    def forward_with_edges(
        self,
        *,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Compute the descriptor from a fixed-shape edge list.

        Parameters
        ----------
        extended_coord
            Coordinates with shape (nf, nloc*3) or (nf, nloc, 3) in Å.
        extended_atype
            Atom types with shape (nf, nloc).
        edge_index
            Edge indices with shape (2, E).
        edge_vec
            Edge vectors with shape (E, 3) in Å.
        edge_mask
            Edge mask with shape (E,).

        Returns
        -------
        descriptor
            Descriptor with shape (nf, nloc, channels). Only l=0 is returned.
        rot_mat
            Empty tensor (not used).
        g2
            Empty tensor (not used).
        h2
            Empty tensor (not used).
        sw
            Empty tensor (not used).
        """
        # === Step 1. Setup dimensions ===
        extended_coord = extended_coord.to(self.compute_dtype)
        if extended_coord.ndim == 2:
            extended_coord = rearrange(extended_coord, "nf (nloc c) -> nf nloc c", c=3)
        elif extended_coord.ndim != 3:
            raise ValueError(
                "extended_coord must have shape (nf, nloc*3) or (nf, nloc, 3)"
            )
        nf, nloc = extended_atype.shape[:2]
        n_nodes = int(nf * nloc)

        # === Step 2. Type embedding (l=0) ===
        with nvtx_range("type_embedding"):
            atype_loc = extended_atype[:, :nloc]  # (nf, nloc)
            type_ebed = self.type_embedding(atype_loc).reshape(
                n_nodes, self.channels
            )  # (N, C)

        # === Step 3. Build edge cache once (fixed-shape edges) ===
        with nvtx_range("build_edge_cache"):
            edge_cache = build_edge_cache_from_edges(
                type_ebed=type_ebed,
                atype_flat=atype_loc.reshape(-1),
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_mask=edge_mask,
                compute_dtype=self.compute_dtype,
                eps=self.eps,
                inner_clamp=self.inner_clamp,
                edge_envelope=self.edge_envelope,
                radial_basis=self.radial_basis,
                has_exclude_types=bool(self.exclude_types),
                edge_type_keep_mask=self._edge_type_keep_mask,
                random_gamma=self.random_gamma,
                wigner_calc=self.wigner_calc,
            )

        lmax_0 = self.l_schedule[0]
        ebed_dim_0 = get_so3_dim_of_lmax(lmax_0)  # (lmax+1)^2
        x0 = type_ebed  # (N, C)
        x0_out = x0  # (N, C)

        # === Step 4. Compute radial features once (fp32+) ===
        radial_feat = None
        with nvtx_range("radial_embedding"):
            if edge_cache.src.numel() > 0:
                radial_feat = rearrange(
                    self.radial_embedding(edge_cache.edge_rbf),
                    "E (L C) -> E L C",
                    L=self.lmax + 1,
                    C=self.channels,
                )  # (E, lmax+1, C)

        # === Step 5. Env FiLM conditioning (optional, fp32+) ===
        with nvtx_range("env_film"):
            if self.use_env_seed and edge_cache.src.numel() > 0:
                atype_flat = atype_loc.reshape(-1)  # (N,)
                film = self.env_seed_embedding(
                    edge_cache=edge_cache,
                    atype_flat=atype_flat,
                    n_nodes=n_nodes,
                )  # (N, 2*C)
                scale_logits, shift_logits = film.chunk(2, dim=-1)  # (N, C), (N, C)
                # ScalarRMSNorm is unified to focus-aware layout (B, F, C).
                # Env FiLM remains a single scalar stream, so F=1 here.
                scale_hat = self.film_scale_norm(scale_logits.unsqueeze(1)).squeeze(
                    1
                )  # (N, C)
                shift_hat = self.film_shift_norm(shift_logits.unsqueeze(1)).squeeze(
                    1
                )  # (N, C)
                scale_strength = torch.exp(self.film_scale_strength_log)
                shift_strength = torch.exp(self.film_shift_strength_log)
                scale = 1.0 + scale_strength * torch.tanh(scale_hat)  # (N, C)
                shift = shift_strength * torch.tanh(shift_hat)  # (N, C)
                x0_out = x0 * scale + shift

        # === Step 6. Build l=0 features ===
        x = type_ebed.new_zeros(
            n_nodes, ebed_dim_0, self.n_focus, self.focus_dim
        )  # (N, D, F, Cf)
        x[:, 0, :, :] = x0_out.reshape(n_nodes, self.n_focus, self.focus_dim)

        # === Step 7. Geometric Initial Embedding (fp32+) ===
        with nvtx_range("gie"):
            if self.use_gie and radial_feat is not None:
                x = x + self.gie(
                    n_nodes=n_nodes,
                    edge_cache=edge_cache,
                    radial_feat=radial_feat[:, 1:, :],
                ).reshape(n_nodes, ebed_dim_0, self.n_focus, self.focus_dim)

        # === Step 8. Fuse edge type features into radial features (fp32+) ===
        with nvtx_range("radial_fuse"):
            if radial_feat is not None:
                radial_feat = radial_feat + rearrange(
                    edge_cache.edge_type_feat, "E C -> E 1 C"
                )

        if radial_feat is not None:
            radial_feat = radial_feat.to(dtype=self.dtype)
            rad_feat_per_block = [
                radial_feat[:, :rad_len, :] for rad_len in self.rad_sizes_per_block
            ]
        else:
            rad_feat_per_block = []

        # === Step 9. Convert to self.dtype and run blocks ===
        with nvtx_range("blocks"):
            x = x.to(dtype=self.dtype)  # (N, D, F, Cf)
            if edge_cache.src.numel() > 0:
                edge_cache = edge_cache_to_dtype(edge_cache, self.dtype)
                with self._compute_mode_ctx(extended_coord.device):
                    x = self._forward_blocks(x, edge_cache, rad_feat_per_block)

        # === Step 10. Final l=0 output mixing ===
        with nvtx_range("output_ffn"):
            x_scalar = (
                x[:, 0:1, :, :]
                .reshape(n_nodes, 1, 1, self.channels)
                .to(dtype=self.compute_dtype)
            )  # (N, 1, 1, C)
            x_scalar = x_scalar + self.output_ffn(x_scalar)

        # === Step 11. Reshape to (nf, nloc, channels) and return ===
        descriptor = rearrange(
            x_scalar, "(nf nloc) 1 1 C -> nf nloc C", nf=nf, nloc=nloc
        )  # (nf, nloc, C)
        return (
            descriptor.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            self._empty_tensor,
            self._empty_tensor,
            self._empty_tensor,
            self._empty_tensor,
        )

    def _forward_blocks(
        self,
        x: torch.Tensor,
        edge_cache: EdgeFeatureCache,
        radial_feat_per_block: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Run the interaction blocks with optional depth attention.

        Parameters
        ----------
        x
            Initial node features with shape (N, D, F, Cf).
        edge_cache
            Per-edge cache.
        radial_feat_per_block
            List of per-block radial features already truncated to l_schedule[i]+1.

        Returns
        -------
        torch.Tensor
            Output features with shape (N, D, F, Cf).
        """
        if not self.use_full_attn_res and not self.use_block_attn_res:
            # === Fast path without descriptor-level attention residuals ===
            for i, block in enumerate(self.blocks):
                x = x[:, : self.ebed_dims[i], :, :]
                blk_radial = radial_feat_per_block[i]
                with nvtx_range(f"block_{i}"):
                    x, _, _, _ = block(x, edge_cache, blk_radial)
            return x

        n_node = x.shape[0]

        def node_l0_extractor(v: torch.Tensor) -> torch.Tensor:
            """Extract scalar features from global SO(3) layout."""
            return v[:, 0, :, :].reshape(n_node, self.channels)

        if self.use_full_attn_res:
            # === Step 1. Maintain descriptor-level unit history ===
            unit_history = [x]

            # === Step 2. Run each block with selective unit-history aggregation ===
            for i, block in enumerate(self.blocks):
                current_dim = self.ebed_dims[i]
                current_x = x[:, :current_dim, :, :]
                truncated_unit_history = [
                    source[:, :current_dim, :, :] for source in unit_history
                ]
                blk_radial = radial_feat_per_block[i]
                with nvtx_range(f"block_{i}"):
                    block_output, _, so2_unit_output, ffn_unit_outputs = block(
                        current_x,
                        edge_cache,
                        blk_radial,
                        unit_history=truncated_unit_history,
                    )
                unit_history.append(so2_unit_output)
                unit_history.extend(ffn_unit_outputs)
                x = block_output

            # === Step 3. Final aggregation over all completed unit representations ===
            final_dim = self.ebed_dims[-1]
            final_sources = [source[:, :final_dim, :, :] for source in unit_history]
            x = self.final_full_attn_res(
                sources=final_sources,
                scalar_extractor=node_l0_extractor,
                current_x=x,
            ).to(dtype=self.dtype)
            return x

        # === Step 1. Maintain descriptor-level block history ===
        block_history = [x]

        # === Step 2. Run each block with selective block-history aggregation ===
        for i, block in enumerate(self.blocks):
            current_dim = self.ebed_dims[i]
            current_x = x[:, :current_dim, :, :]
            truncated_block_history = [
                source[:, :current_dim, :, :] for source in block_history
            ]
            blk_radial = radial_feat_per_block[i]
            with nvtx_range(f"block_{i}"):
                block_output, block_summary, _, _ = block(
                    current_x,
                    edge_cache,
                    blk_radial,
                    unit_history=truncated_block_history,
                )
            block_history.append(block_summary)
            x = block_output

        # === Step 3. Final aggregation over all completed block summaries ===
        final_dim = self.ebed_dims[-1]
        final_sources = [source[:, :final_dim, :, :] for source in block_history]
        x = self.final_block_attn_res(
            sources=final_sources,
            scalar_extractor=node_l0_extractor,
            current_x=x,
        ).to(dtype=self.dtype)
        return x

    def _edge_type_keep_mask(
        self,
        atype_flat: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build keep mask for edge pairs based on excluded type pairs.

        Parameters
        ----------
        atype_flat
            Flattened local atom types with shape (N,).
        src
            Source indices with shape (E,).
        dst
            Destination indices with shape (E,).

        Returns
        -------
        torch.Tensor
            Boolean mask with shape (E,), True means keep.
        """
        if self.emask.no_exclusion:
            return torch.ones_like(src, dtype=torch.bool, device=src.device)
        type_i = atype_flat.index_select(0, dst)
        type_j = atype_flat.index_select(0, src)
        type_i = torch.where(type_i >= 0, type_i, self.ntypes)
        type_j = torch.where(type_j >= 0, type_j, self.ntypes)
        type_ij = type_i * (self.ntypes + 1) + type_j
        type_mask = self.emask.type_mask.to(device=atype_flat.device)
        keep = type_mask.index_select(0, type_ij.to(dtype=torch.long))
        return keep.to(dtype=torch.bool)

    def _init_lm_schedules(
        self,
        lmax: int,
        n_blocks: int,
        l_schedule: list[int] | None,
        mmax: int | None,
        m_schedule: list[int] | None,
    ) -> None:
        """Parse and validate L/M schedules, setting self.l_schedule/m_schedule/lmax/mmax."""
        # === L schedule ===
        if l_schedule is None:
            self.l_schedule = [int(lmax)] * int(n_blocks)
        else:
            self.l_schedule = [int(x) for x in l_schedule]
        if len(self.l_schedule) == 0:
            raise ValueError("`l_schedule` must be non-empty")
        if any(x < 0 for x in self.l_schedule):
            raise ValueError("`l_schedule` entries must be non-negative")
        if any(
            self.l_schedule[i] < self.l_schedule[i + 1]
            for i in range(len(self.l_schedule) - 1)
        ):
            raise ValueError("`l_schedule` must be non-increasing (pyramid schedule)")

        self.lmax = int(self.l_schedule[0])
        self.n_blocks = len(self.l_schedule)

        # === M schedule ===
        if m_schedule is None:
            if mmax is None:
                self.m_schedule = [int(l) for l in self.l_schedule]
            else:
                mmax_i = int(mmax)
                if mmax_i < 0:
                    raise ValueError("`mmax` must be non-negative")
                self.m_schedule = [min(mmax_i, int(l)) for l in self.l_schedule]
        else:
            self.m_schedule = [int(x) for x in m_schedule]
        if len(self.m_schedule) == 0:
            raise ValueError("`m_schedule` must be non-empty")
        if len(self.m_schedule) != len(self.l_schedule):
            raise ValueError("`m_schedule` must have the same length as `l_schedule`")
        if any(x < 0 for x in self.m_schedule):
            raise ValueError("`m_schedule` entries must be non-negative")
        if any(m > l for m, l in zip(self.m_schedule, self.l_schedule)):
            raise ValueError(
                "`m_schedule` entries must satisfy `m_schedule[i] <= l_schedule[i]`"
            )

        self.mmax = int(self.m_schedule[0])

    @contextmanager
    def _compute_mode_ctx(self, device: torch.device) -> Generator[None, None, None]:
        """
        Context manager that applies automatic mixed precision (AMP) for forward().

        Parameters
        ----------
        device
            The device of the input tensors (used to determine if CUDA ops apply).

        Notes
        -----
        - When `use_amp=True` and the model is in training mode, enables
          torch.autocast with bfloat16 on CUDA.
        - Only affects autocast-eligible operations (matmul, conv, etc.).
        - Does nothing during inference (`self.training=False`), on non-CUDA
          devices, or when `use_amp=False`.

        Yields
        ------
        None
            Runs the wrapped region under the configured AMP setting.
        """
        if not self.use_amp or device.type != "cuda" or not self.training:
            yield
            return

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            yield

    # === DeePMD descriptor interface ===
    def get_rcut(self) -> float:
        return self.rcut

    def get_rcut_smth(self) -> float:
        return self.rcut

    def get_sel(self) -> list[int]:
        return self.sel

    def get_nsel(self) -> int:
        return sum(self.sel)

    def get_ntypes(self) -> int:
        return self.ntypes

    def get_type_map(self) -> list[str]:
        return self.type_map if self.type_map is not None else []

    def get_dim_out(self) -> int:
        return self.channels

    def get_dim_emb(self) -> int:
        return self.get_dim_out()

    def mixed_types(self) -> bool:
        """
        If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        SeZM uses SeZMTypeEmbedding for type handling, so it does not require
        a type-distinguished neighbor list.
        """
        return True

    def has_message_passing(self) -> bool:
        return bool(len(self.blocks) > 0 and self.lmax > 0)

    def need_sorted_nlist_for_lower(self) -> bool:
        return False

    def get_env_protection(self) -> float:
        return self.eps

    @property
    def dim_out(self) -> int:
        return self.get_dim_out()

    @property
    def dim_emb(self) -> int:
        return self.get_dim_emb()

    def share_params(
        self, base_class: Any, shared_level: int, resume: bool = False
    ) -> None:
        raise NotImplementedError("share_params is not supported for SeZM")

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Receive the statistics (distance, max_nbor_size and env_mat_range) of the training data.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        """
        raise NotImplementedError("Compression is unsupported for SeZM.")

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
    ) -> None:
        raise NotImplementedError("change_type_map is not supported for SeZM")

    def reinit_exclude(
        self, exclude_types: list[tuple[int, int]] | None = None
    ) -> None:
        if exclude_types is None:
            exclude_types = []
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    # =========================================================================
    # Statistics interface (interface compatibility only)
    # -------------------------------------------------------------------------
    # SeZM uses EquivariantRMSNorm inside blocks for feature normalization,
    # so mean/stddev are NOT used in forward(). These methods are kept for:
    #   1. Interface compatibility with BaseDescriptor
    #   2. Consistent serialization format (davg/dstd in checkpoint)
    # =========================================================================

    def set_stat_mean_and_stddev(
        self, mean: torch.Tensor, stddev: torch.Tensor
    ) -> None:
        """Set mean and stddev (interface compatibility, not used in forward)."""
        self.mean = mean
        self.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get mean and stddev (interface compatibility, not used in forward)."""
        return self.mean, self.stddev

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        """
        Compute statistics (interface compatibility, not used in forward).

        SeZM uses learnable EquivariantRMSNorm for normalization, so these
        statistics do not affect the forward pass. This is a no-op that keeps
        mean/stddev at their initialized values (zero/one) for interface consistency.
        """
        del merged, path
        # No-op: mean and stddev are already initialized to zero/one in __init__
        # and are not used in forward() due to EquivariantRMSNorm.

    def serialize(self) -> dict[str, Any]:
        state = self.state_dict()
        return {
            "@class": "Descriptor",
            "type": "SeZM",
            "@version": 1,
            "config": {
                "ntypes": self.ntypes,
                "sel": self.sel,
                "rcut": self.rcut,
                "env_exp": self.env_exp,
                "type_map": self.type_map,
                "lmax": self.lmax,
                "n_blocks": self.n_blocks,
                "l_schedule": self.l_schedule,
                "mmax": self.mmax,
                "m_schedule": self.m_schedule,
                "channels": self.channels,
                "n_radial": self.n_radial,
                "radial_mlp": self.radial_mlp,
                "use_env_seed": self.use_env_seed,
                "random_gamma": self.random_gamma,
                "so2_norm": self.so2_norm,
                "so2_layers": self.so2_layers,
                "so2_attn_res": self.so2_attn_res_mode,
                "n_focus": self.n_focus,
                "focus_compete": self.focus_compete,
                "ffn_neurons": self.ffn_neurons,
                "ffn_blocks": self.ffn_blocks,
                "layer_scale": self.layer_scale,
                "n_atten_head": self.n_atten_head,
                "sandwich_norm": self.sandwich_norm,
                "full_attn_res": self.full_attn_res_mode,
                "block_attn_res": self.block_attn_res_mode,
                "activation_function": self.activation_function,
                "glu_activation": self.glu_activation,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "mlp_bias": self.mlp_bias,
                "use_amp": self.use_amp,
                "use_triton": self.use_triton,
                "exclude_types": self.exclude_types,
                "eps": self.eps,
                "trainable": self.trainable,
                "seed": self.seed,
                "inner_clamp_r_inner": self.inner_clamp_r_inner,
                "inner_clamp_r_outer": self.inner_clamp_r_outer,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
            "env_mat": DPEnvMat(self.rcut, self.rcut, self.eps).serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> DescrptSeZM:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "Descriptor":
            raise ValueError(f"Invalid class for DescrptSeZM: {data_cls}")
        type_val = data.pop("type")
        if type_val != "SeZM":
            raise ValueError(f"Invalid type for DescrptSeZM: {type_val}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported SeZM version: {version}")
        config = data.pop("config")
        variables = data.pop("@variables")
        data.pop("env_mat", None)
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

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        """
        Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            Data used to do neighbor statistics.
        type_map : list[str] | None
            The name of each type of atoms.
        local_jdata : dict
            The local data refer to the current class.

        Returns
        -------
        dict
            The updated local data.
        float | None
            The minimum distance between two atoms.
        """
        local_jdata_cpy = local_jdata.copy()
        min_nbor_dist, sel = UpdateSel().update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["rcut"],
            local_jdata_cpy["sel"],
            True,  # mixed_type=True for unified sel
        )
        local_jdata_cpy["sel"] = sel[0]
        return local_jdata_cpy, min_nbor_dist

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Ignore legacy checkpoint state that is rebuilt at construction."""
        expected_keys = {prefix + key for key in self.state_dict().keys()}
        for full_key in list(state_dict.keys()):
            if full_key.startswith(prefix) and full_key not in expected_keys:
                state_dict.pop(full_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
