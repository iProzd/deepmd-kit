# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SeZM: The descriptor of smooth equivariant ZBL Message-passing Network.

PyTorch backend

This implementation is designed around two non-negotiables:

1) Conservative forces: the descriptor is computed from differentiable energy.
2) Speed-first inference: edge geometry and Wigner-D rotation blocks are computed
   exactly once per `forward()` and reused by all interaction blocks.

Core math utilities live in `se_zm_helper.py`, while per-block message passing
is implemented in `se_zm_block.py`.

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
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
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
from .se_zm_block import (
    EquivariantFFN,
    ScalarRMSNorm,
    SeZMInteractionBlock,
)
from .se_zm_helper import (
    C2CutoffEnvelope,
    EdgeFeatureCache,
    EnvironmentInitialEmbedding,
    GeometricInitialEmbedding,
    RadialBasis,
    RadialMLP,
    WignerDCalculator,
    build_edge_type_feat,
    edge_cache_to_dtype,
    get_promoted_dtype,
    get_so3_dim_of_lmax,
    init_edge_rot_mat_frisvad,
    np_safe,
    nvtx_range,
    safe_norm,
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
class DescrptSeZMNet(BaseDescriptor, nn.Module):
    """
    SeZM: Smooth equivariant ZBL Message-passing Network descriptor for DeePMD-kit.

    Execution outline
    -----------------
    1. Build a per-forward `EdgeFeatureCache` (geometry, envelope, Wigner-D).
    2. Build radial/type edge features once and reuse across blocks.
    3. Run `SeZMInteractionBlock` stack with optional l/m schedules.
    4. Extract scalar channels and apply the final scalar FFN.

    Parameters
    ----------
    rcut
        Cutoff radius in Å.
    sel
        Maximum number of neighbors per type within `rcut`.
        - int: broadcast to all types, e.g. sel=100 with ntypes=2 → [100, 100]
        - list[int]: sel[i] is the maximum number of type i atoms within `rcut`
    ntypes
        Number of element types.
    lmax
        Maximum degree, only used when `l_schedule` is None.
    n_blocks
        Number of blocks (only used when `l_schedule` is None).
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
    channels
        Total channels per (l,m) coefficient.
    n_focus
        Number of parallel focus streams. The per-stream channel width is
        ``focus_dim = channels // n_focus``. Must divide ``channels`` exactly.
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
    so2_norm
        If True, apply intermediate ReducedSeparableRMSNorm between SO(2) mixing layers.
        When False (default), no normalization is applied between layers.
    so2_layers
        Number of SO(2) mixing layers per block.
    ffn_neurons
        Hidden sizes for the equivariant FFN in each block and the final scalar output FFN.
    ffn_blocks
        Number of FFN subblocks per interaction block.
    focus_compete
        If True, enable cross-focus softmax competition inside SO(2) convolution.
        Competition logits are built from l=0 scalar channels before SO(2) mixing
        and applied after SO(2) stack to scale full irreps uniformly per focus.
    n_atten_head
        Number of attention heads when aggregating messages in SO(2) convolution.
        0 applies a plain envelope-weighted scatter-sum; >0 enables
        envelope-aware grouped softmax attention with output-side head gate.
        Competition is weighted by ``edge_env**0.5`` and value amplitude is
        scaled by ``edge_env``.
        When enabled, the per-focus stream width
        ``focus_dim = channels // n_focus`` must be divisible by ``n_atten_head``.
    sandwich_norm
        Pre/post-norm switches for [SO(2), FFN] residual branches in order:
        [so2_pre, so2_post, ffn_pre, ffn_post], shared across all blocks.
    mlp_bias
        Whether to use bias in equivariant layers. When False, removes bias from:
        - SO3Linear: l=0 bias
        - SO2Linear: l=0 bias
        - GatedActivation: gate linear bias
        - SeparableRMSNorm: centering bias
        - ReducedSeparableRMSNorm: centering bias
        Attention projections in SO2Convolution
        (attn_radial_bias_proj, attn_output_gate_proj) are always bias-free.
    layer_scale
        If True, apply learnable LayerScale (init 1e-3) on residual branches:
        - SO(2) branch: per-focus-channel scales `(n_focus, focus_dim)`
          on each SO(2) mixing layer.
        - FFN branch: per-channel scales `(channels,)` on each FFN subblock.
    activation_function
        Activation function used by deepmd EmbeddingNet.
    glu_activation
        If True, use GLU-style gating in FFN (e.g., silu -> swiglu, gelu -> geglu).
    use_amp
        If True, use automatic mixed precision (AMP) with bfloat16 on CUDA.
        This does not provide accelerations under fp32 precision but will decrease
        the memory usage, while persevering model accuracy.
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
        rcut: float,
        sel: list[int] | int,
        ntypes: int,
        lmax: int = 2,
        n_blocks: int = 2,
        l_schedule: list[int] | None = None,
        mmax: int | None = 2,
        m_schedule: list[int] | None = None,
        channels: int = 64,
        n_focus: int = 1,
        n_radial: int = 10,
        radial_mlp: list[int] | None = None,
        use_env_seed: bool = True,
        so2_norm: bool = False,
        so2_layers: int = 3,
        ffn_neurons: int = 96,
        ffn_blocks: int = 1,
        focus_compete: bool = False,
        n_atten_head: int = 0,
        sandwich_norm: list[bool] | None = None,
        activation_function: str = "silu",
        glu_activation: bool = True,
        precision: str = "float32",
        mlp_bias: bool = True,
        layer_scale: bool = False,
        use_amp: bool = True,
        exclude_types: list[tuple[int, int]] | None = None,
        eps: float = 1e-7,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        type_map: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()

        self.rcut = float(rcut)
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
        self.so2_norm = bool(so2_norm)
        self.so2_layers = int(so2_layers)
        self.ffn_neurons = int(ffn_neurons)
        self.ffn_blocks = int(ffn_blocks)
        if self.ffn_blocks < 1:
            raise ValueError("`ffn_blocks` must be >= 1")
        self.n_atten_head = int(n_atten_head)
        if self.n_atten_head > 0 and self.focus_dim % self.n_atten_head != 0:
            raise ValueError(
                "`focus_dim` must be divisible by `n_atten_head` when attention is enabled"
            )
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
        self.trainable = bool(trainable)
        self.seed = seed

        # === Env seed parameters ===
        self.use_env_seed = bool(use_env_seed)
        self.env_seed_embed_dim = min(self.channels, 128)
        self.env_seed_type_dim = min(32, max(8, self.channels // 4))
        axis_dim = 4 if self.env_seed_embed_dim < 64 else 8
        self.env_seed_axis_dim = min(axis_dim, max(1, self.env_seed_embed_dim - 1))
        rbf_out_dim = max(32, self.env_seed_embed_dim - 2 * self.env_seed_type_dim)
        g_in_dim = rbf_out_dim + 2 * self.env_seed_type_dim
        self.env_seed_hidden_dim = min(256, max(2 * self.env_seed_embed_dim, g_in_dim))

        # === Step 0. Split deterministic seeds at the descriptor top-level ===
        seed_type_embedding = child_seed(self.seed, 0)
        seed_blocks = child_seed(self.seed, 1)
        seed_out = child_seed(self.seed, 2)
        seed_radial_embedding = child_seed(self.seed, 3)
        seed_env_seed = child_seed(self.seed, 4)

        # === Step 1. L/M schedules ===
        self._init_lm_schedules(lmax, n_blocks, l_schedule, mmax, m_schedule)
        self.ebed_dims = [get_so3_dim_of_lmax(l) for l in self.l_schedule]
        self.rad_sizes_per_block = [l + 1 for l in self.l_schedule]

        # === Step 2. Statistics buffers (interface compatibility) ===
        _shape = (self.ntypes, self.nnei, self._ENV_DIM)
        mean = torch.zeros(_shape, dtype=self.dtype, device=self.device)
        stddev = torch.ones(_shape, dtype=self.dtype, device=self.device)
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.stats: dict[str, Any] | None = None

        # === Step 3. Excluded type pairs ===
        self.reinit_exclude(exclude_types)

        # === Step 4. Type embedding ===
        type_embedding_precision = RESERVED_PRECISION_DICT[self.compute_dtype]
        self.type_embedding = TypeEmbedNet(
            type_nums=self.ntypes,
            embed_dim=self.channels,
            precision=type_embedding_precision,  # force fp32+
            seed=seed_type_embedding,
            type_map=type_map,
            trainable=self.trainable,
        )

        # === Step 5. Env FiLM embedding (optional) ===
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
            exponent=7,
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

        # === C^2 cutoff envelope for edge weight ===
        self.c2_envelope = C2CutoffEnvelope(rcut=self.rcut, exponent=5)

        wigner_lmax = self.l_schedule[0]
        # force fp32+
        self.wigner_calc = WignerDCalculator(
            lmax=wigner_lmax,
            eps=self.eps,
            dtype=self.compute_dtype,
        )

        if self.l_schedule[0] > 0:
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
                    ffn_neurons=self.ffn_neurons,
                    ffn_blocks=self.ffn_blocks,
                    layer_scale=self.layer_scale,
                    n_atten_head=self.n_atten_head,
                    so2_pre_norm=self.so2_pre_norm,
                    so2_post_norm=self.so2_post_norm,
                    ffn_pre_norm=self.ffn_pre_norm,
                    ffn_post_norm=self.ffn_post_norm,
                    activation_function=self.activation_function,
                    glu_activation=self.glu_activation,
                    mlp_bias=self.mlp_bias,
                    eps=self.eps,
                    dtype=self.dtype,
                    seed=child_seed(seed_blocks, block_idx),
                    trainable=self.trainable,
                )
            )
        self.blocks = nn.ModuleList(blocks)

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
            edge_cache = self.build_edge_cache(
                type_ebed=type_ebed,
                extended_coord=extended_coord,
                extended_atype=extended_atype,
                nlist=nlist,
                mapping=mapping,
                pair_keep_mask=pair_keep_mask,
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
            if self.env_seed_embedding is not None and edge_cache.src.numel() > 0:
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
            if self.gie is not None and radial_feat is not None:
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
            edge_cache = self.build_edge_cache_from_edges(
                type_ebed=type_ebed,
                atype_flat=atype_loc.reshape(-1),
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_mask=edge_mask,
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
            if self.env_seed_embedding is not None and edge_cache.src.numel() > 0:
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
            if self.gie is not None and radial_feat is not None:
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
        Run the interaction blocks.

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
        # Blocks with pyramid l-schedule slicing
        for i, block in enumerate(self.blocks):
            x = x[:, : self.ebed_dims[i], :, :]
            blk_radial = radial_feat_per_block[i]
            with nvtx_range(f"block_{i}"):
                x = block(x, edge_cache, blk_radial)
        return x

    def build_edge_cache(
        self,
        *,
        type_ebed: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None,
        pair_keep_mask: torch.Tensor,
    ) -> EdgeFeatureCache:
        """
        Build the global edge cache from DeePMD padded neighbor list.

        This converts DeePMD's per-frame padded neighbor list into a flat list of
        valid edges used by message passing, and computes all per-edge tensors that
        are reused across blocks.

        The resulting cache contains:

        - per-edge endpoints: ``src``, ``dst`` and per-edge type features: ``edge_type_feat`` (src+dst)
        - per-edge geometry: ``edge_vec``
        - per-edge smooth weights: C^2 cutoff envelope ``edge_env``
        - per-edge radial basis: ``edge_rbf`` (envelope already baked in)
        - per-edge rotation blocks: block-diagonal Wigner-D matrices ``D_full`` and ``Dt_full``
        - destination-node normalization: ``inv_sqrt_deg`` for neighbor norm

        Notes
        -----
        Input formats follow DeePMD conventions:

        - ``extended_coord`` has shape ``(nf, nall, 3)``.
        - ``nlist`` has shape ``(nf, nloc, nnei)`` and stores indices into the extended axis
          (``0..nall-1``), with ``-1`` indicating padding.
        - ``mapping`` (when provided) maps extended indices to local indices ``0..nloc-1``.
          When ``mapping`` is ``None``, the function assumes the neighbor indices are already local.

        This function avoids branchy gather on ``-1`` indices by appending a sentinel coordinate
        far outside the cutoff and mapping padding entries to that sentinel. Those padded pairs
        are then removed by the keep/within-cutoff masks before any normalization or rotation.

        Parameters
        ----------
        type_ebed
            Per-node type embedding with shape (N, C), where N=nf*nloc.
        extended_coord
            Extended coordinates with shape (nf, nall, 3).
        extended_atype
            Extended atom types with shape (nf, nall).
            Currently unused; reserved for potential type-dependent filtering.
        nlist
            Neighbor list with shape (nf, nloc, nnei).
        mapping
            Mapping from extended indices to local indices with shape (nf, nall), or None.
        pair_keep_mask
            Pair keep mask from `PairExcludeMask` with shape (nf, nloc, nnei). True means keep.

        Returns
        -------
        EdgeFeatureCache
            Per-edge cache.
        """
        nf, nloc, nnei = nlist.shape
        n_nodes = int(nf * nloc)

        # === Step 0. Force fp32+ for geometry ===
        geom_dtype = get_promoted_dtype(extended_coord.dtype)
        coord = extended_coord.to(dtype=geom_dtype)  # (nf, nall, 3)
        nall = coord.shape[1]

        # === Step 1. Build per-pair geometry with padding-safe gather ===
        # DeePMD uses -1 for padding in nlist. torch.gather cannot index -1, so we
        # replace padding indices with 0 (any valid index works since padding positions
        # are masked out by `keep`). This avoids the extra cat operation for sentinel.
        with nvtx_range("geometry"):
            valid_nlist = nlist >= 0  # (nf, nloc, nnei)
            keep = valid_nlist & pair_keep_mask  # (nf, nloc, nnei)

            # Replace padding (-1) with 0 to avoid gather errors.
            # Padding positions are excluded by `keep` mask, so their values don't matter.
            gather_index = torch.where(  # (nf, nloc, nnei)
                valid_nlist, nlist, torch.zeros_like(nlist)
            )
            index = rearrange(  # (nf, nloc*nnei, 3)
                gather_index, "nf nloc nnei -> nf (nloc nnei) 1"
            ).expand(-1, -1, 3)
            nei_pos = rearrange(  # (nf, nloc, nnei, 3)
                torch.gather(coord, 1, index),
                "nf (nloc nnei) c -> nf nloc nnei c",
                nloc=nloc,
                nnei=nnei,
            )
            atom_pos = rearrange(  # (nf, nloc, 1, 3)
                coord[:, :nloc], "nf nloc c -> nf nloc 1 c"
            )
            diff = nei_pos - atom_pos  # (nf, nloc, nnei, 3)
            length = safe_norm(diff, self.eps)  # (nf, nloc, nnei, 1)

        # === Step 2. C^2 envelope weight `sw` ===
        # sw is the C^2-continuous cutoff envelope weight in [0, 1], applied per neighbor pair.
        with nvtx_range("envelope"):
            sw = self.c2_envelope(length)  # (nf, nloc, nnei, 1)
            sw = sw * rearrange(keep, "nf nloc nnei -> nf nloc nnei 1").to(
                dtype=sw.dtype
            )

        # === Step 3. Filter valid edges for message passing ===
        # An edge is valid if:
        #   - it is not padding (nlist >= 0)
        #   - the type pair is allowed (pair_keep_mask)
        # Note: We do NOT filter by `length < rcut` here. Edges beyond rcut have
        # edge_env=0 (from C2CutoffEnvelope), so their messages naturally vanish.
        # This avoids the dynamic-output-size `nonzero` kernel and enables smoother
        # degree/normalization (no discontinuous edge count jumps at rcut boundary).
        with nvtx_range("filter"):
            edge_keep = rearrange(  # (nf*nloc*nnei,)
                keep, "nf nloc nnei -> (nf nloc nnei)"
            )
            edge_idx = torch.nonzero(edge_keep).squeeze(-1)  # (E,)
            edge_env = sw.reshape(-1, 1)[edge_idx]  # (E, 1)

        if edge_idx.numel() == 0:
            # No edges -> empty cache.
            device = extended_coord.device
            dtype = extended_coord.dtype
            empty_long = torch.empty(  # (0,)
                0, dtype=torch.long, device=device
            )
            empty_vec = torch.empty(  # (0, 3)
                0, 3, dtype=dtype, device=device
            )
            empty_rbf = torch.empty(  # (0, n_radial)
                0, self.radial_basis.n_radial, dtype=dtype, device=device
            )
            empty_type_feat = torch.empty(  # (0, C)
                0, type_ebed.shape[1], dtype=dtype, device=device
            )
            deg = torch.zeros(n_nodes, dtype=dtype, device=device)  # (N,)
            inv_sqrt_deg = torch.ones(  # (N, 1, 1)
                n_nodes, 1, 1, dtype=dtype, device=device
            )
            cache = EdgeFeatureCache(
                src=empty_long,
                dst=empty_long,
                edge_type_feat=empty_type_feat,
                edge_vec=empty_vec,
                edge_rbf=empty_rbf,
                edge_env=torch.empty(0, 1, dtype=dtype, device=device),
                deg=deg,
                inv_sqrt_deg=inv_sqrt_deg,
                D_full=None,
                Dt_full=None,
                D_to_m_cache={},
                Dt_from_m_cache={},
            )
            return cache

        # === Step 4. Build flat edge indices and map to (src, dst) nodes ===
        # edge_idx indexes the flattened (nf, nloc, nnei) axis in row-major order.
        # Convert it back to:
        #   f_idx   in [0, nf)
        #   loc_idx in [0, nloc)
        #   neighbor index from nlist (extended axis)
        with nvtx_range("index"):
            nlist_flat = nlist.reshape(-1)  # (nf*nloc*nnei,)
            edge_idx_flat = edge_idx.to(dtype=torch.long)  # (E,)
            valid_f_idx = edge_idx_flat // (nloc * nnei)  # (E,)
            rem = edge_idx_flat % (nloc * nnei)  # (E,)
            valid_loc_idx = rem // nnei  # (E,)
            # neighbor indices from the extended axis (0..nall-1) for valid edges.
            valid_neighbor = nlist_flat[edge_idx_flat]  # (E,)
            if mapping is None:
                # Neighbor indices are already local indices in [0, nloc).
                src_local = valid_neighbor  # (E,)
            else:
                # Map extended index -> local index for each frame.
                # mapping_flat packs (nf, nall) so frame k uses offset k * nall.
                mapping_flat = mapping.reshape(-1)  # (nf*nall,)
                src_local = mapping_flat[valid_f_idx * nall + valid_neighbor]

            # dst is the center atom: per-frame local index -> global node index.
            dst = valid_f_idx * nloc + valid_loc_idx  # (E,)
            src_ok = (src_local >= 0) & (src_local < nloc)  # (E,)
            if not bool(src_ok.all()):
                # Drop edges that map outside the local range (e.g. broken mapping or ghost-only neighbor).
                edge_idx = edge_idx[src_ok]
                valid_f_idx = valid_f_idx[src_ok]
                valid_loc_idx = valid_loc_idx[src_ok]
                dst = dst[src_ok]
                src_local = src_local[src_ok]
                edge_env = edge_env[src_ok]

            # src is the neighbor atom (per-frame local index -> global node index)
            src = valid_f_idx * nloc + src_local  # (E,)
            num_edges = int(src.numel())

        # === Step 5. Gather per-edge geometry ===
        # edge_vec points from center -> neighbor: r_ij = r_j - r_i (in Å).
        with nvtx_range("edge_geom"):
            diff_flat = diff.reshape(-1, 3)  # (nf*nloc*nnei, 3)
            length_flat = length.reshape(-1, 1)  # (nf*nloc*nnei, 1)
            edge_vec = diff_flat[edge_idx]  # (E, 3)
            edge_len = length_flat[edge_idx]  # (E, 1)

        # === Step 6. Radial basis (envelope already baked in) ===
        with nvtx_range("radial_basis"):
            edge_rbf = self.radial_basis(edge_len)  # (E, n_radial)

        # === Step 7. Wigner-D blocks ===
        with nvtx_range("rot_mat"):
            rot_mat = init_edge_rot_mat_frisvad(  # (E, 3, 3)
                edge_vec, edge_len=edge_len, eps=self.eps
            )
        with nvtx_range("wigner_d"):
            D_full, Dt_full = self.wigner_calc(rot_mat)  # (E, D, D), (E, D, D)

        edge_type_feat = build_edge_type_feat(  # (E, C)
            type_ebed, src, dst, num_edges=num_edges
        )

        # === Step 8. Neighbor normalization (destination degree) ===
        # Compute inverse sqrt degree for graph-style message normalization.
        with nvtx_range("degree"):
            deg = torch.zeros(
                n_nodes, dtype=edge_vec.dtype, device=edge_vec.device
            )  # (N,)
            ones = torch.ones_like(dst, dtype=edge_vec.dtype)  # (E,)
            deg.index_add_(0, dst, ones)  # (N,)
            inv_sqrt_deg = rearrange(  # (N, 1, 1)
                torch.rsqrt(deg.clamp(min=1)), "N -> N 1 1"
            )

        cache = EdgeFeatureCache(
            src=src,  # (E,)
            dst=dst,  # (E,)
            edge_type_feat=edge_type_feat,  # (E, C)
            edge_vec=edge_vec,  # (E, 3)
            edge_rbf=edge_rbf,  # (E, n_radial)
            edge_env=edge_env,  # (E, 1)
            deg=deg,  # (N,)
            inv_sqrt_deg=inv_sqrt_deg,  # (N, 1, 1)
            D_full=D_full,  # (E, D, D)
            Dt_full=Dt_full,  # (E, D, D)
            D_to_m_cache={},
            Dt_from_m_cache={},
        )
        return cache

    def build_edge_cache_from_edges(
        self,
        *,
        type_ebed: torch.Tensor,
        atype_flat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> EdgeFeatureCache:
        """
        Build the global edge cache from a fixed-shape edge list.

        Parameters
        ----------
        type_ebed
            Per-node type embedding with shape (N, C), where N=nf*nloc.
        atype_flat
            Flattened local atom types with shape (N,).
        edge_index
            Edge indices with shape (2, E).
        edge_vec
            Edge vectors with shape (E, 3) in Å.
        edge_mask
            Edge mask with shape (E,). True means keep.

        Returns
        -------
        EdgeFeatureCache
            Per-edge cache.
        """
        n_nodes = int(type_ebed.shape[0])
        src = edge_index[0].to(dtype=torch.long)
        dst = edge_index[1].to(dtype=torch.long)

        # === Step 1. Normalize mask and apply type exclusions ===
        edge_keep = edge_mask.to(dtype=torch.bool)
        if self.exclude_types:
            edge_keep = edge_keep & self._edge_type_keep_mask(atype_flat, src, dst)

        # === Step 2. Promote geometry dtype ===
        geom_dtype = self.compute_dtype
        edge_vec = edge_vec.to(dtype=geom_dtype)
        edge_keep_f = edge_keep.to(dtype=geom_dtype).unsqueeze(-1)
        edge_vec = edge_vec * edge_keep_f
        edge_vec = edge_vec + (1.0 - edge_keep_f) * edge_vec.new_tensor([0.0, 0.0, 1.0])

        # === Step 3. Edge length, envelope, and radial basis ===
        with nvtx_range("envelope"):
            edge_len = safe_norm(edge_vec, self.eps)
            edge_env = self.c2_envelope(edge_len) * edge_keep_f  # (E, 1)
            edge_rbf = self.radial_basis(edge_len) * edge_keep_f  # (E, n_radial)

        # === Step 4. Rotation blocks ===
        with nvtx_range("rot_mat"):
            rot_mat = init_edge_rot_mat_frisvad(
                edge_vec, edge_len=edge_len, eps=self.eps
            )
        with nvtx_range("wigner_d"):
            D_full, Dt_full = self.wigner_calc(rot_mat)

        # === Step 5. Edge type features ===
        edge_type_feat = build_edge_type_feat(
            type_ebed, src, dst, num_edges=int(src.numel())
        )
        edge_type_feat = edge_type_feat * edge_keep_f.to(dtype=edge_type_feat.dtype)

        # === Step 6. Neighbor normalization ===
        with nvtx_range("degree"):
            deg = torch.zeros(
                n_nodes, dtype=edge_vec.dtype, device=edge_vec.device
            )  # (N,)
            deg.index_add_(0, dst, edge_keep_f.squeeze(-1))
            inv_sqrt_deg = rearrange(
                torch.rsqrt(deg.clamp(min=1)), "N -> N 1 1"
            )  # (N, 1, 1)

        cache = EdgeFeatureCache(
            src=src,
            dst=dst,
            edge_type_feat=edge_type_feat,
            edge_vec=edge_vec,
            edge_rbf=edge_rbf,
            edge_env=edge_env,
            deg=deg,
            inv_sqrt_deg=inv_sqrt_deg,
            D_full=D_full,
            Dt_full=Dt_full,
            D_to_m_cache={},
            Dt_from_m_cache={},
        )
        return cache

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
        - When `use_amp=True`, enables torch.autocast with bfloat16 on CUDA.
        - Only affects autocast-eligible operations (matmul, conv, etc.).
        - Does nothing on non-CUDA devices or when `use_amp=False`.

        Yields
        ------
        None
            Runs the wrapped region under the configured AMP setting.
        """
        if not self.use_amp or device.type != "cuda":  # and self.training
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

        SeZM uses TypeEmbedNet for type handling, so it does not require
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
    # SeZM uses SeparableRMSNorm inside blocks for feature normalization,
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

        SeZM uses learnable SeparableRMSNorm for normalization, so these
        statistics do not affect the forward pass. This is a no-op that keeps
        mean/stddev at their initialized values (zero/one) for interface consistency.
        """
        del merged, path
        # No-op: mean and stddev are already initialized to zero/one in __init__
        # and are not used in forward() due to SeparableRMSNorm.

    def serialize(self) -> dict[str, Any]:
        state = self.state_dict()
        return {
            "@class": "Descriptor",
            "type": "SeZM",
            "@version": 1,
            "config": {
                "rcut": self.rcut,
                "sel": self.sel,
                "ntypes": self.ntypes,
                "type_map": self.type_map,
                "lmax": self.lmax,
                "n_blocks": self.n_blocks,
                "l_schedule": self.l_schedule,
                "mmax": self.mmax,
                "m_schedule": self.m_schedule,
                "channels": self.channels,
                "n_focus": self.n_focus,
                "focus_compete": self.focus_compete,
                "n_radial": self.n_radial,
                "radial_mlp": self.radial_mlp,
                "use_env_seed": self.use_env_seed,
                "so2_norm": self.so2_norm,
                "so2_layers": self.so2_layers,
                "ffn_neurons": self.ffn_neurons,
                "ffn_blocks": self.ffn_blocks,
                "layer_scale": self.layer_scale,
                "n_atten_head": self.n_atten_head,
                "sandwich_norm": self.sandwich_norm,
                "activation_function": self.activation_function,
                "glu_activation": self.glu_activation,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "mlp_bias": self.mlp_bias,
                "use_amp": self.use_amp,
                "exclude_types": self.exclude_types,
                "eps": self.eps,
                "trainable": self.trainable,
                "seed": self.seed,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
            "env_mat": DPEnvMat(self.rcut, self.rcut, self.eps).serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> DescrptSeZMNet:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "Descriptor":
            raise ValueError(f"Invalid class for DescrptSeZMNet: {data_cls}")
        type_val = data.pop("type")
        if type_val != "SeZM":
            raise ValueError(f"Invalid type for DescrptSeZMNet: {type_val}")
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
