# SeZM: Smooth Equivariant Zone-bridging Model

Technical reference for the SeZM descriptor and model, implemented in the PyTorch backend of DeePMD-kit.

**Source files:**

- `deepmd/pt/model/descriptor/sezm.py` — top-level descriptor (`DescrptSeZM`)
- `deepmd/pt/model/descriptor/sezm_nn/` — all submodules
- `deepmd/pt/model/model/sezm_model.py` — model scaffold, compile path, ZBL injection

______________________________________________________________________

## 1. Overview

SeZM is an SO(3)-equivariant message-passing descriptor designed for molecular dynamics workloads where inference speed and physical correctness jointly matter. The model predicts per-atom energies, derives forces by differentiating the energy with respect to coordinates (conservative forces), and guarantees a smooth potential energy surface (PES) through C³-continuous cutoff envelopes on every edge.

The descriptor maps local atomic environments to rotationally invariant scalar features through a stack of interaction blocks, each containing an SO(2) convolution operating in a per-edge local frame and an equivariant feed-forward network (FFN). Edge geometry — distances, radial basis, Wigner-D rotation matrices — is computed once per forward call and shared across all blocks, avoiding the cost multiplier that plagues per-block geometry recomputation.

The descriptor and model are both registered as `SeZM` (alias `sezm`). The model-level scaffold `SeZMModel` handles energy/force/virial output, optional analytical short-range repulsion, and an end-to-end `torch.compile` path that supports second-order coordinate derivatives through Inductor.

### 1.1 Key Innovations

**Zone Bridging framework.** SeZM introduces a general-purpose inner-bridging mechanism that allows any analytical short-range potential to be additively composed with the learned energy, while guaranteeing that the descriptor contribution on every bridging-zone pair is strictly frozen. The mechanism composes two parameter-free modules — InnerClamp (C³ distance saturation) and SFPG (Source Freeze Propagation Gate, a multiplicative per-node gate) — on the same radial window. Together they close all three leakage channels (scalar distance, direction, multi-hop propagation) that would otherwise break the additive decomposition. The default analytical potential is ZBL, but the framework is agnostic to the specific formula: any pair potential that accepts a pairwise distance can be plugged in.

**Compiled double-backward training.** SeZM achieves the first end-to-end `torch.compile` training path for an equivariant ML potential whose force loss requires second-order coordinate derivatives (`∂²E/∂x∂θ`). The approach uses `make_fx` symbolic tracing to capture `autograd.grad(create_graph=True)` as ordinary FX nodes, then hands the flat graph to Inductor with dynamic shapes. This yields a 2–3× wall-clock speedup over eager training while keeping the full conservative-force guarantee intact. A catalogue of compile invariants (§12.2) documents each non-obvious choice that makes this work under PyTorch's current tracing constraints.

**Model architecture:**

- **Environment Initial Embedding (FiLM).** A physics-motivated 4D environment matrix `[s, s·r̂]` is aggregated into a low-rank local descriptor `D = env_agg^T @ env_agg`, then projected to FiLM scale/shift logits that condition the scalar backbone. This provides geometric inductive bias at layer 0 through a dedicated type-embedding branch, decoupled from the main type embedding, reducing the number of interaction blocks needed to capture local geometry.

- **Envelope-gated softmax attention.** The message aggregation uses `edge_env² × exp(logit)` in the numerator and a learnable positive bias `ζ` in the denominator, so that the attention weight of each edge smoothly decays to zero at the cutoff boundary together with its derivatives. A post-aggregation output-side head gate (query-dependent, per-head sigmoid) further modulates the aggregated message. This attention replaces degree-based normalization when active and naturally integrates with SFPG through the `src_weight` parameter.

- **Multi-focus SO(2) convolution.** Multiple parallel focus streams process the same geometric context inside the SO(2) operator. A cross-focus softmax competition (driven by `l=0` scalar invariants, with label smoothing to prevent dead focuses) re-weights the streams before rotate-back. Unlike MHA, which attends across sequence positions, multi-focus attends across parallel equivariant sub-channels on the same edge. Unlike sparse MoE, all focuses are computed and then soft-weighted, preserving SO(3) equivariance.

- **Trainable-frequency radial basis.** Radial basis frequencies, initialized as integer harmonics `nπ/rcut`, are learnable parameters that adapt to the data distribution during training, improving expressiveness over a fixed grid.

**Training methodology:**

- **HybridMuon with slice mode.** SeZM uses `muon_mode="slice"` for its primary optimizer routing: 2D weight matrices go to Muon, while 3D `SO3Linear` weights `(lmax+1, C_in, C_out)` are sliced along the degree axis so each `l`-block receives an independent Muon Newton-Schulz update with correct rectangular scaling. The Muon learning rate adjustment (`lr_adjust`) is set to 0, letting the base learning rate schedule control the effective step size directly.

- **Magma-lite damping.** A per-block cosine-alignment score between Muon momentum and current gradient is EMA-smoothed and mapped to a damping scale in `[0.1, 1.0]`, reducing update magnitude on blocks with noisy or misaligned gradients. No stochastic Bernoulli masking is used, keeping the damping dense and stable for force-field objectives.

- Norm scales, layer scales, radial frequencies, and type embeddings are routed to Adam (via `adam_` naming prefix) for stability.

______________________________________________________________________

## 2. Code Organization

```
deepmd/pt/model/descriptor/
├── sezm.py                    # DescrptSeZM: config parsing, edge-cache construction,
│                               #   block scheduling, forward(), serialize/deserialize
├── sezm_nn/
│   ├── __init__.py            # public re-export layer
│   ├── utils.py               # NVTX helpers, dtype promotion, serialization helpers
│   ├── edge_cache.py          # EdgeFeatureCache, edge construction, SFPG gate,
│   │                           #   build_edge_cache / build_edge_cache_from_edges
│   ├── indexing.py            # packed (l, m) indexing, reduced-layout maps,
│   │                           #   rotation projection helpers, inv_rescale
│   ├── radial.py              # C3CutoffEnvelope, InnerClamp, BridgingSwitch,
│   │                           #   RadialBasis, RadialMLP
│   ├── activation.py          # GatedActivation, SwiGLU, S2GridProjector,
│   │                           #   SwiGLUS2Activation
│   ├── embedding.py           # SeZMTypeEmbedding, GeometricInitialEmbedding,
│   │                           #   EnvironmentInitialEmbedding
│   ├── norm.py                # EquivariantRMSNorm, ReducedEquivariantRMSNorm,
│   │                           #   ScalarRMSNorm
│   ├── attention.py           # segment_envelope_gated_softmax
│   ├── attn_res.py            # DepthAttnRes
│   ├── so3.py                 # ChannelLinear, FocusLinear, SO3Linear
│   ├── so2.py                 # SO2Linear, SO2Convolution
│   ├── ffn.py                 # EquivariantFFN, PointwiseGridMLP
│   ├── block.py               # SeZMInteractionBlock
│   ├── dens.py                # ForceEmbedding, denoising/direct-force heads
│   ├── wignerd.py             # build_edge_quaternion, WignerDCalculator
│   └── triton/                # Triton kernels for eval-mode acceleration
│       ├── constants.py       # tile/mode constants
│       ├── dispatch.py        # single dispatch policy
│       ├── custom_ops.py      # Triton kernel launchers
│       ├── autograd.py        # public API + eager fallback
│       ├── kernels_small.py   # specialized lmax ≤ 3 kernels
│       ├── kernels_generic.py # tiled kernels for lmax ≥ 4
│       └── kernels_edge_geometry_rbf.py
│                               # fused geometry/RBF chain
deepmd/pt/model/model/
└── sezm_model.py              # SeZMModel, InterPotential,
                                #   trace_and_compile, make_fx path
```

Each submodule is self-contained. `sezm.py` imports everything through `sezm_nn/__init__.py`, which re-exports the public API of each submodule. Tests are split by responsibility: descriptor tests in `test_descriptor_sezm.py`, Triton tests in `test_descriptor_sezm_triton.py`, model/compile tests in `test_sezm_model.py`.

______________________________________________________________________

## 3. Forward Pass Overview

A single forward pass through SeZM proceeds as follows. The text diagram shows the main data flow; subsequent sections expand each stage.

```
SeZMModel main forward path (core_compute):
  Inputs: coord, atype, box
    ├─ build DeePMD neighbor list → extended_coord, extended_atype, mapping, nlist
    ├─ format_nlist (outside the compiled graph)
    ├─ build compact sparse edges (src, dst, edge_vec, edge_mask) from nlist
    └─ graph inputs: local (coord, atype) + sparse edges

  EdgeFeatureCache (built once per forward via build_edge_cache_from_edges):
    ├─ edges: (src, dst) global indices, edge_vec
    ├─ edge_type_feat: per-edge type embedding (src + dst)
    ├─ edge_rbf: Bessel radial basis × C² envelope (trainable frequencies)
    ├─ edge_env: C³ cutoff envelope (flattened to valid edges)
    ├─ D_full, Dt_full: block-diagonal Wigner-D matrices
    ├─ inv_sqrt_deg: inverse sqrt smooth degree for normalization
    └─ edge_src_gate: SFPG per-edge gate η_src (when bridging is active)

  Radial embedding (computed once in fp32+):
    └─ radial_feat: (E, lmax+1, C) via RadialMLP(edge_rbf)
       └─ type features fused once after GIE; per-block slices prebuilt

  Node initialization:
    ├─ l=0: Type embedding + optional EnvironmentInitialEmbedding (FiLM)
    └─ l>0: Zonal (m=0) initial embedding via Wigner-D + radial_feat[:, 1:, :]

  Interaction blocks (pyramid schedule):
    for block i:
      ├─ slice node features to ebed_dim(l_schedule[i])
      ├─ optional depth AttnRes from unit/block history
      ├─ EquivariantRMSNorm (pre-SO2)
      ├─ SO(2) Convolution
      │  ├─ pre_focus_mix: full-channel projection
      │  ├─ rotate to edge-local frame via Wigner-D
      │  ├─ radial modulation + multi-layer SO2Linear stack
      │  ├─ optional cross-focus softmax competition
      │  ├─ rotate back to global frame
      │  └─ scatter-aggregate (envelope-weighted or attention)
      ├─ FFN subblocks (ffn_blocks iterations)
      │  ├─ EquivariantRMSNorm (pre-FFN)
      │  └─ SO3Linear → Activation → SO3Linear (zero-init output)
      └─ optional depth AttnRes updates

  Output:
    └─ Extract l=0 scalar features
    └─ Scalar FFN with residual: x + FFN(x)
    └─ Reshape to (nf, nloc, channels)

  Post-process in SeZMModel:
    ├─ fitting network + output statistics + atom mask
    ├─ optional ZBL energy on sparse-edge path
    ├─ autograd for force / virial
    └─ local outputs (energy, force, virial)
```

Two `forward` entry points exist in `DescrptSeZM`:

- **`forward(...)`** — the standard DeePMD descriptor interface. Accepts `extended_coord`, `extended_atype`, `nlist`, `mapping`. Builds a padded edge cache from the neighbor list. Zone bridging (InnerClamp + SFPG) is excluded because ZBL energy injection is handled only by `SeZMModel` on the sparse-edge path.
- **`forward_with_edges(...)`** — the sparse-edge interface used by `SeZMModel`. Accepts pre-built `(edge_index, edge_vec, edge_mask)`. Supports zone bridging and returns both the scalar descriptor and the full equivariant latent for downstream heads.

Both paths share `_forward_blocks(...)` for the actual interaction-block loop.

______________________________________________________________________

## 4. Edge Feature Cache

### 4.1 Design Rationale

Edge geometry computation — coordinate gathering, distance calculation, quaternion construction, Wigner-D evaluation, radial basis expansion — is the most expensive per-edge workload. SeZM computes all of this exactly once per `forward()` and packs the results into an `EdgeFeatureCache` dataclass. Every interaction block reads from this shared cache; no block is allowed to recompute geometry.

### 4.2 Cache Contents

| Field             | Shape              | Description                                       |
| ----------------- | ------------------ | ------------------------------------------------- |
| `src`, `dst`      | `(E,)`             | Flattened node indices in `[0, N)`                |
| `edge_type_feat`  | `(E, C)`           | Per-edge type embedding (src + dst lookup)        |
| `edge_vec`        | `(E, 3)`           | Displacement vectors in Å                         |
| `edge_rbf`        | `(E, n_radial)`    | Bessel radial basis × C² envelope                 |
| `edge_env`        | `(E, 1)`           | C³ cutoff envelope weights                        |
| `deg`             | `(N, 1)`           | Smooth degree: `Σ_e edge_env²` per destination    |
| `inv_sqrt_deg`    | `(N, 1, 1)`        | `rsqrt(deg + eps)` for normalization              |
| `D_full`          | `(E, D, D)`        | Block-diagonal Wigner-D (global→local)            |
| `Dt_full`         | `(E, D, D)`        | Transpose of `D_full` (local→global)              |
| `D_to_m_cache`    | dict               | Cached m-major projections keyed by `"lmax:mmax"` |
| `Dt_from_m_cache` | dict               | Cached inverse projections keyed by `"lmax:mmax"` |
| `edge_src_gate`   | `(E, 1)` or `None` | SFPG per-edge gate `η_src(e)` (bridging only)     |

### 4.3 Edge Construction

**Neighbor-list path** (`build_edge_cache`): Takes the DeePMD padded neighbor list `(nf, nloc, nnei)`. Padding entries (`nlist == -1`) and excluded type pairs are filtered out before any distance computation. Edges with `r ≥ rcut` are kept — their `edge_env = 0` from the C³ envelope naturally zeros their messages. This avoids the dynamic-output-size `torch.nonzero` kernel that distance filtering would require, and it keeps the smooth degree `Σ edge_env²` free of discontinuous jumps at the cutoff boundary.

**Sparse-edge path** (`build_edge_cache_from_edges`): Takes pre-built `(edge_index, edge_vec, edge_mask)` from `SeZMModel.core_compute`. Masked edges (`edge_mask=False`) have their displacement vector reset to `(0, 0, 1)` to provide a safe normalization target — this prevents NaN gradients from zero-length vector division while contributing zero downstream thanks to `edge_env = 0`.

### 4.4 Smooth Degree Normalization

After cache construction, `_finalize_edge_cache` computes:

```
deg[j] = Σ_{e: dst(e)=j} edge_env[e]²
inv_sqrt_deg[j] = 1 / sqrt(deg[j] + eps)
```

The squared envelope ensures the degree is a smooth function of atomic positions (C⁶ regularity from squaring a C³ function). `inv_sqrt_deg` is applied at every aggregation site in the non-attention path, providing graph-style normalization analogous to GCN's `D^{-1/2}`.

### 4.5 Dtype and Triton Dispatch

Geometry computations always run in fp32+ (`compute_dtype = get_promoted_dtype(dtype)`) regardless of the model's working precision. This ensures accurate distances, quaternions, and Wigner-D matrices for stable training convergence.

When the descriptor is in eval mode and the device supports it, `build_edge_cache` replaces the eager geometry chain with a fused Triton kernel (`kernels_edge_geometry_rbf.py`). Training always uses the eager chain to preserve the full PyTorch autograd graph for force/virial higher-order derivatives. The `edge_cache_to_dtype` helper converts float fields to the working dtype when entering the interaction blocks, and clears the rotation projection caches to prevent dtype mismatches.

______________________________________________________________________

## 5. Radial Functions

### 5.1 C³ Cutoff Envelope

The `C3CutoffEnvelope` (in `radial.py`) enforces a smooth transition to zero at the cutoff radius. For normalized distance `x = r / rcut`:

```
E(x) = 1 + x^p * (a + b*x + c*x² + d*x³)    for x ∈ [0, 1)
E(x) = 0                                       for x ≥ 1
```

The coefficients `(a, b, c, d)` are uniquely determined by the boundary conditions `E(1) = 0`, `E'(1) = 0`, `E''(1) = 0`, `E'''(1) = 0`, which guarantee C³ continuity at the cutoff. The exponent `p` (default 5 for edge envelope, 7 for radial basis envelope) controls how steeply values decay near `rcut`: larger `p` keeps values closer to 1.0 over more of the range before dropping.

With the default `p = 5`:

```
E(x) = 1 − 56x⁵ + 140x⁶ − 120x⁷ + 35x⁸
```

The C³ envelope is applied at two points:

1. **Inside `RadialBasis.forward()`**: multiplied into each radial basis function, making the basis itself vanish at `rcut`.
1. **As `edge_env`**: applied to all edge messages during aggregation.

This double application ensures that a message, its first derivative, its second derivative, and its third derivative with respect to distance all reach exactly zero at `rcut`. Conservative forces (first derivative of energy) and force-loss training (second derivative) therefore see no discontinuity at the cutoff boundary.

### 5.2 Trainable Radial Basis

`RadialBasis` (in `radial.py`) produces `n_radial` basis functions evaluated at each edge distance. The basis uses a sinc form:

```
φ_n(r) = w_n · sinc(w_n · r / π) = sin(w_n · r) / r
```

where `w_n = n · π / rcut` for `n = 1, ..., n_radial`. The sinc form is chosen for numerical stability near `r → 0`: unlike `sin(w·r)/r`, `sinc` is well-defined at zero and produces stable gradients. Each basis function is multiplied by the C² envelope (with `exponent` typically 7) before output.

The frequencies `w_n` are stored as trainable parameters (`adam_freqs`) with the `adam_` prefix so that HybridMuon routes them to the Adam optimizer without weight decay. During training, the frequencies can shift away from their integer-harmonic initialization to better fit the data distribution.

### 5.3 RadialMLP

`RadialMLP` maps the `n_radial`-dimensional radial basis vector to a `(lmax+1) × channels`-dimensional per-edge feature through a sequence of `Linear → ScalarRMSNorm → SiLU` layers. The final layer is a bare linear projection without normalization or activation.

All linear layers use `bias=False`. This is a deliberate choice: with zero bias, a zero input (from masked or padding edges) produces exactly zero output. Bias terms would leak a constant offset into masked edges, which is particularly problematic on the `torch.compile` path where padding edges must contribute exactly zero.

The output is reshaped to `(E, lmax+1, C)` — one radial feature vector per degree `l` and per channel. The `l = 0` slice feeds the scalar branch; `l ≥ 1` slices feed the Geometric Initial Embedding (GIE) and the per-degree modulation inside SO(2) convolution.

______________________________________________________________________

## 6. Initial Feature Construction

### 6.1 Type Embedding

`SeZMTypeEmbedding` (in `embedding.py`) maps discrete atom types to continuous vectors. It stores a learnable embedding table of shape `(ntypes, channels)` named `adam_type_embedding`. The `adam_` prefix routes it to the Adam optimizer in HybridMuon. An optional padding row (index `ntypes`) is zeroed out for masked atoms.

The type embedding provides the sole `l = 0` initial feature:

```
x[:, 0, 0, :] = type_embedding(atype)    # (N, C)
```

All higher-degree coefficients (`l ≥ 1`) start at zero and are seeded by the Geometric Initial Embedding described below.

### 6.2 Geometric Initial Embedding (GIE)

`GeometricInitialEmbedding` (in `embedding.py`) seeds `l > 0` features at layer 0 by projecting radial features through the zonal (`m = 0`) column of the Wigner-D transpose matrix. This gives the initial backbone non-trivial angular information from the start, reducing the number of interaction blocks needed to capture directional dependence.

For each degree `l ≥ 1`, the `m = 0` column of `Dt_full` at row `(l, m')` gives the real spherical harmonic `Y_l^{m'}` evaluated along the edge direction. The GIE multiplies this column element-wise with the radial feature `radial_feat[:, l, :]` (shape `(E, C)`), then scatters the result to destination nodes:

```
For each edge e = (src → dst):
  For each degree l ≥ 1:
    msg_l[m'] = Dt_full[e, l²+l+m', l²+l] × radial_feat[e, l, :]    for m' = -l..+l
  scatter_add msg to node dst
Normalize by inv_sqrt_deg
```

The implementation avoids advanced-index writeback (which can silently produce zero gradients on some PyTorch builds under `make_fx`). Instead, it uses `index_add_` into a compact buffer and then assigns back to the output tensor at the appropriate row indices.

When SFPG is active, each edge message is multiplied by `edge_src_gate` before scatter, so any edge from a frozen-zone source contributes exactly zero.

### 6.3 Environment Initial Embedding (FiLM Conditioning)

`EnvironmentInitialEmbedding` (in `embedding.py`, optional via `use_env_seed=True`) provides a physics-motivated inductive bias for the scalar features by conditioning them on a local environment matrix. The conditioning uses Feature-wise Linear Modulation (FiLM): the environment embedding produces per-atom scale and shift parameters that modulate the type embedding.

**The 4D environment vector.** For each edge, a 4-component vector `r_tilde` encodes both radial decay and angular information in the global frame:

```
s = edge_env / r
r_hat = edge_vec / r
r_tilde = [s, s·r_hat_x, s·r_hat_y, s·r_hat_z]
```

Unlike the edge-local frame used by SO(2) convolution, `r_tilde` uses the global frame direction. This preserves full angular information (three independent components) rather than projecting onto a local axis.

**G-network.** A two-layer MLP processes per-edge features:

1. RBF projection: `Linear(n_radial → rbf_out_dim) → SiLU → Linear(rbf_out_dim → rbf_out_dim)`
1. Concatenate with source and destination type embeddings from an independent `env_type_embed`
1. G-MLP: `Linear(concat_dim → hidden_dim) → SiLU → Linear(hidden_dim → embed_dim)`

The environment type embedding is deliberately independent from the main type embedding, allowing `env_seed` to learn its own type representations optimized for the environment-matrix pathway.

**Aggregation and D-matrix.** The outer product `r_tilde ⊗ g` (shape `(E, 4, embed_dim)`) is scattered to destination nodes and normalized by `inv_sqrt_deg`. The resulting aggregated matrix `env_agg` (shape `(N, 4, embed_dim)`) captures the local geometry around each atom. The matrix product `D = env_agg^T @ env_agg[:, :, :axis_dim]` (shape `(N, embed_dim, axis_dim)`) reduces this to a compact descriptor.

**FiLM application.** `D` is flattened and projected to `2 × channels` logits, split into scale and shift:

```
scale = 1 + scale_strength · tanh(RMSNorm(scale_logits))
shift = shift_strength · tanh(RMSNorm(shift_logits))
x0 = x0 · scale + shift
```

The strengths are learnable log-parameters initialized to `log(0.01)`, so the initial FiLM effect is near-identity (`scale ≈ 1`, `shift ≈ 0`). The `tanh` bounds prevent large-magnitude modulation early in training. RMSNorm on the logits prevents scale collapse. The output projection is zero-initialized, making the entire module a no-op at initialization — the network learns when and how to use the environment conditioning from the loss signal.

______________________________________________________________________

## 7. Wigner-D Rotation System

SeZM operates on SO(3)-equivariant features by rotating them between a global frame and edge-aligned local frames. The rotation is realized through real-basis Wigner-D matrices, computed from edge-aligned quaternions by `WignerDCalculator` (in `wignerd.py`).

### 7.1 Edge-Aligned Quaternion

`build_edge_quaternion(edge_vec)` returns a quaternion `q_edge` whose rotation matrix `R(q_edge)` maps the global frame to a local frame where the edge direction aligns with `+Z`:

```
R(q_edge) @ (edge_vec / ‖edge_vec‖) = (0, 0, 1)
```

The quaternion is built from two exact charts:

- **Chart A**: regular away from the `−Z` pole, handling edges that point roughly upward.
- **Chart B**: regular away from the `+Z` pole, handling edges that point roughly downward.

A C^∞ normalized-linear blend (`quaternion_nlerp` with sign alignment for shortest arc) is applied only inside the overlap region of the two charts. This avoids the singularity that any single chart exhibits at one of the poles, producing a smooth edge rotation across all directions.

**Numerical stability.** Vector and quaternion normalizations clamp squared norms before taking the square root — `sqrt(clamp(‖x‖², eps²))` — to prevent NaN gradients at zero or near-zero vectors. For fp16/bf16, the norm computation is promoted to fp32 before the square root.

### 7.2 Random Roll (random_gamma)

When `random_gamma=True` (default), SeZM samples an independent roll angle `γ ~ U[0, 2π)` per edge per forward call, constructs a local `+Z` roll quaternion `q_γ`, and left-composes it with the edge quaternion:

```
q_total = q_γ · q_edge
```

A local `+Z` roll leaves the edge-direction invariant (`(0,0,1)` is unchanged by rotation about `Z`) while randomizing the in-plane gauge. This data augmentation breaks the correlation between the arbitrary choice of local `x/y` axes and the training data, improving generalization without affecting equivariance.

### 7.3 Block-Diagonal Wigner-D

`WignerDCalculator.forward(quaternions)` returns `(D_full, Dt_full)` where `D_full` is block-diagonal with one `(2l+1) × (2l+1)` block per degree `l`:

```
D_full = diag(D⁰, D¹, D², ..., D^lmax)
```

The block for `l = 0` is the `1 × 1` identity. Higher-degree blocks are computed as follows:

| Degree     | Method                                                                                                                                                      |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `l = 1`    | Direct construction from the quaternion-induced 3×3 rotation matrix, permuted and sign-adjusted to match real spherical harmonic ordering `(m = -1, 0, +1)` |
| `l = 2`    | Dedicated degree-4 quaternion tensor contraction kernel                                                                                                     |
| `l = 3, 4` | Dedicated quaternion monomial kernels; when both are needed, they share one degree-8 matrix multiply                                                        |
| `l ≥ 5`    | Generic quaternion polynomial evaluator using precomputed Wigner polynomial coefficient tables                                                              |

All Wigner-D blocks are stored in the real spherical harmonic basis with `m = -l, ..., +l` ordering within each `l` block. The coefficient tables exploit the symmetry relation `D^l_{-m',-m} = (-1)^{m'-m} conj(D^l_{m',m})` to store only half the coefficients.

`Dt_full = D_full^T` is the inverse rotation (local → global), valid because real orthogonal representations satisfy `D^{-1} = D^T`.

### 7.4 Reduced Rotation Projections

Inside SO(2) convolution, only coefficients with `|m| ≤ mmax` are retained. Rather than extracting a subblock at runtime, SeZM precomputes the row/column index sets and caches the projected blocks:

- `project_D_to_m(D_full, coeff_index_m)` → `D_to_m` of shape `(E, D_m, D)` — row subset of `D_full`
- `project_Dt_from_m(Dt_full, coeff_index_m)` → `Dt_from_m` of shape `(E, D, D_m)` — column subset of `Dt_full`

These projections are cached in the `EdgeFeatureCache` keyed by the string `"lmax:mmax"` and reused across blocks that share the same `(lmax, mmax)`.

When `DP_TRITON=1` is set and the model is in eval mode, these materialized projections are skipped entirely. The Triton kernels in `triton/` fuse the rotation and truncation into a single pass over the Wigner-D matrix, reducing memory traffic.

### 7.5 Inverse Rescale for Truncated Rotation

When `mmax < lmax`, the rotate-to-local step discards `|m| > mmax` coefficients, and the rotate-back step treats them as zero. For degrees `l > mmax`, this truncation loses energy: the `(2l+1)` full coefficients are reduced to `(2·mmax+1)` retained coefficients. To compensate, the rotate-back output is multiplied by a per-coefficient rescale factor:

```
rescale(l) = sqrt((2l+1) / (2·min(l, mmax)+1))
```

This factor restores the expected norm of each degree block after the truncated round-trip. The rescale vector `rotate_inv_rescale_full` is precomputed as a buffer in `SO2Convolution` and applied element-wise after the rotate-back `bmm`.

______________________________________________________________________

## 8. Interaction Blocks

### 8.1 Block Structure

Each `SeZMInteractionBlock` (in `block.py`) follows a two-path residual architecture operating on node features of shape `(N, D, 1, C)`:

```
SeZMInteractionBlock:
  Path 1 — SO(2) Convolution:
    x_pre = pre_so2_norm(x)                          # EquivariantRMSNorm
    y = so2_conv(x_pre, edge_cache, radial_feat)     # message passing
    y = post_so2_norm(y)                              # optional
    x = x + y

  Path 2 — FFN subblock sequence:
    for i in range(ffn_blocks):
        x_pre = pre_ffn_norm[i](x)                   # EquivariantRMSNorm
        y = ffn[i](x_pre)                            # EquivariantFFN
        y = post_ffn_norm[i](y)                       # optional
        if layer_scale:
            y = y × adam_ffn_layer_scale[i]           # per-channel, init 1e-3
        x = x + y

  return x
```

The `sandwich_norm` config `[so2_pre, so2_post, ffn_pre, ffn_post]` controls which norms are active. The default `[True, False, True, False]` uses pre-norm only, matching the Pre-LN Transformer convention.

### 8.2 SO(2) Convolution

`SO2Convolution` (in `so2.py`) is the message-passing engine. It operates in the edge-aligned local frame where only `|m| ≤ mmax` coefficients are retained, converting the full SO(3) rotation to a sequence of SO(2) operations that scale linearly in `lmax` instead of cubically.

**Step-by-step flow:**

1. **Pre-focus mixing.** A `ChannelLinear` (n_focus=1) mixes channels on the node tensor `(N, D, C)` before any edge operation.

1. **Gather and rotate to local frame.** Source features are gathered per edge and rotated: `x_local = bmm(D_to_m, x_src)`, producing `(E, D_m, C)` where `D_m = Σ_l (2·min(l, mmax)+1)`.

1. **Radial modulation.** Each coefficient is multiplied by its degree-specific radial feature: `x_local *= radial_feat[:, degree_index_m, :]`. The `degree_index_m` buffer maps each position in the m-major layout to its degree `l`.

1. **Reshape to multi-focus layout.** `x_local` is reshaped to `(E, F, D_m, Cf)` where `F = n_focus` and `Cf = focus_dim` (or `channels` when `focus_dim = 0`). The hidden width is `H = F × Cf`.

1. **Multi-layer SO(2) stack.** For each of `so2_layers` iterations:

   - Save residual: `residual = x_local`
   - Pre-norm (when `so2_norm=True`): `ReducedEquivariantRMSNorm` on the reduced layout. Identity for the last layer.
   - `SO2Linear`: block-diagonal matmul (see §8.3)
   - Bias correction on layer 0 only: the l=0 bias is modulated by `radial_l0 × edge_env − 1` to keep the bias consistent with the radial/envelope scaling.
   - Nonlinearity (between layers, Identity for last):
     - Default: `GatedActivation` — scalar activation on `l=0`, sigmoid gate from `l=0` applied to `l>0`
     - S2 path (`s2_activation[0]=True`): `SwiGLUS2Activation` — scalar SwiGLU branch + sigmoid gate + S2-grid point-wise multiplication
   - LayerScale + residual: `x_local = residual + scale × x_local`
   - Optional SO(2)-internal `DepthAttnRes` from local layer history

1. **Cross-focus competition** (when `n_focus > 1`). A softmax over focus streams is computed from `l=0` scalars:

   ```
   logits = focus_compete_proj(ScalarRMSNorm(x_local_l0))
   alpha = softmax(logits / tau, dim=focus)
   alpha = (1 − eps) × alpha + eps / n_focus     # label smoothing
   x_local *= alpha[:, :, None, None]
   ```

   Label smoothing (default `eps = 0.02`) prevents dead focuses.

1. **Rotate back.** `x_message = bmm(Dt_from_m, x_local) × rotate_inv_rescale_full`, producing `(E, D, C)`.

1. **Aggregate to destination nodes.**

   - **No attention** (`n_atten_head = 0`): multiply by `edge_env`, scatter-sum by `dst`, multiply by `inv_sqrt_deg`.
   - **With attention** (`n_atten_head > 0`): see §8.4.

1. **Post-focus mixing.** A `ChannelLinear` mixes the full `C` channels back.

### 8.3 SO2Linear

`SO2Linear` (in `so2.py`) implements the core SO(2)-equivariant linear map. It operates on the m-major reduced layout and applies a single block-diagonal matmul that processes all `|m|` groups simultaneously:

- **`m = 0` block**: An unconstrained linear map over all `l = 0..lmax` coefficients (plus optional additive bias).
- **`|m| > 0` blocks**: A constrained 2×2 complex coupling on each `(-m, +m)` pair, treated as `(Re, Im)`:
  ```
  [out_neg_m]   [W_u^T  -W_v^T] [in_neg_m]
  [out_pos_m] = [W_v^T   W_u^T] [in_pos_m]
  ```
  This is the real-number realization of a complex linear map, preserving SO(2) equivariance.

The `|m| > 0` weights are initialized with an extra `1/sqrt(2)` scaling factor to preserve the coupling energy between the real and imaginary parts, compensating for the doubled parameter count relative to the `m = 0` block.

In eval mode with `torch.no_grad()`, SO2Linear caches the assembled block-diagonal weight matrix (`_cached_weight`) to avoid reassembling it every call. The cache is invalidated when gradients are needed.

### 8.4 Attention Aggregation

When `n_atten_head > 0`, SO(2) convolution uses an envelope-gated grouped softmax (`segment_envelope_gated_softmax` in `attention.py`) instead of plain envelope-weighted scatter.

**Logit computation:**

```
q = attn_q_proj(ScalarRMSNorm(x_l0[dst]))     # (E, F, H, Dh)
k = attn_k_proj(ScalarRMSNorm(x_l0[src]))     # (E, F, H, Dh)
logits = dot(q, k) / sqrt(head_dim) + attn_radial_logit_proj(radial_l0)
```

**Destination-wise softmax with envelope gating:**

```
grouped_max = scatter_reduce(logits, dst, reduce="amax")
numerator = edge_env² × exp(logits − grouped_max[dst])
denominator = softplus(z_bias) × exp(−grouped_max) + scatter_sum(numerator, dst)
alpha = numerator / denominator[dst]
```

The `z_bias` term (learnable, initialized to `softplus⁻¹(1)`) prevents the denominator from reaching zero when all edges have small envelope weights, which would produce division-by-zero at the cutoff boundary. Zero-weight edges (from padding or envelope) have their logits set to `−∞` before the grouped max, excluding them from the normalization entirely.

**Output-side head gate:** After scatter-summing the attention-weighted messages, an output gate is applied per head:

```
gate = sigmoid(attn_output_gate_proj(ScalarRMSNorm(x_l0)))
output = output × gate
```

This query-dependent gate (analogous to the G1 gate in AlphaFold) allows the model to selectively suppress or amplify each attention head's contribution based on the destination atom's scalar features.

When attention is active, `inv_sqrt_deg` is not applied — the softmax normalization replaces degree-based normalization.

### 8.5 Equivariant FFN

`EquivariantFFN` (in `ffn.py`) provides per-node channel mixing with degree-aware gating. Three variants exist:

**Standard gated path** (default):

```
h = SO3Linear(x)              # (N, D, C) → (N, D, hidden)
h[l=0] = activation(h[l=0])   # scalar activation (SiLU)
gate = sigmoid(gate_proj(h[l=0]))   # per-l independent gates
h[l>0] = h[l>0] × gate              # gate higher-degree features
out = SO3Linear(h)             # (N, D, hidden) → (N, D, C), zero-init
x = x + out                   # residual
```

Each degree `l` has an independent gate derived from the scalar `l = 0` features. The gate is expanded via `expand_index` to all `2l+1` m-components within that degree.

**S2-grid activation path** (`s2_activation[1]=True`):

The first SO3Linear projects to `2 × hidden` channels. `SwiGLUS2Activation` then:

1. Extracts the `l = 0` slice to build a scalar SwiGLU branch and a sigmoid gate
1. Projects the remaining SO(3) coefficients to an S2 grid
1. Applies point-wise multiplication on the grid (one half gates the other)
1. Projects grid features back to SO(3) coefficients
1. Applies the sigmoid gate and merges the scalar branch back to `l = 0`

**Grid-MLP path** (`grid_mlp=True`):

```
h = SO3Linear(x)                    # up-project
h_scalar = LinearSwiGLU(h[l=0])     # scalar branch
h_grid = to_s2_grid(h)              # project to grid
h_grid = PointwiseGridMLP(h_grid)   # point-wise MLP on grid
h = from_s2_grid(h_grid)            # back to coefficients
h[l=0] += h_scalar                  # merge scalar branch
out = SO3Linear(h)                  # down-project, zero-init
```

In all variants, the output projection is zero-initialized (weights and bias), so the FFN starts as a no-op and the residual path begins near-identity.

### 8.6 Equivariant RMS Normalization

`EquivariantRMSNorm` (in `norm.py`) normalizes SO(3)-equivariant features across all degrees simultaneously, with degree balancing to prevent high-multiplicity degrees from dominating the norm.

**Degree balancing.** Each coefficient from degree `l` is weighted by `1 / ((2l+1) × (lmax+1) × C)` in the RMS computation. This ensures that every degree contributes equally to the norm regardless of its multiplicity `(2l+1)`. Without this balancing, degree `l = 3` (with 7 coefficients) would contribute 7× more to the norm than `l = 0` (with 1 coefficient), causing the scalar features to be suppressed.

**Scalar centering.** Before computing the RMS, the `l = 0` slice is mean-centered across channels. This allows the normalization to be invariant to a uniform shift in the scalar features while preserving the zero-mean property of `l > 0` features (which are already zero-mean by equivariance).

**Per-degree affine.** After normalization, a learnable per-degree scale (`adam_scale`) is applied. The scale is expanded to all coefficients via a precomputed `expand_index`. Additive bias is applied only to the `l = 0` slice.

`ReducedEquivariantRMSNorm` is the variant for the m-major truncated layout inside SO(2) convolution, using `n_coeff_l = 2·min(l, mmax)+1` instead of `2l+1` for the degree weights.

Both norms disable autocast (`@torch.amp.autocast("cuda", enabled=False)`) to ensure the RMS is always computed in full precision.

### 8.7 Pyramid Schedules

**L-schedule.** SeZM supports a non-increasing sequence of `lmax` values across blocks: `l_schedule = [3, 3, 2, 1]`. When the schedule decreases between blocks, the higher-degree slices of the node tensor are physically discarded. Later blocks operate on smaller `D = (l+1)²`, reducing compute. The final block does not need to end at `l = 0` — the output always extracts only the `l = 0` features regardless of the final block's `lmax`.

**M-schedule.** An independent sequence controls `mmax` per block: `m_schedule = [2, 2, 1, 0]`. Each entry must satisfy `m_schedule[i] ≤ l_schedule[i]`. Unlike `l_schedule`, the m-schedule does not change the global node tensor shape `(N, D, 1, C)`. It only affects the edge-local coefficient set retained during SO(2) operations and the corresponding rotation projections.

When both schedules are `None`, SeZM uses constant `lmax` and `mmax = lmax` across all `n_blocks` blocks.

### 8.8 SO(3) Linear Layers

Three linear layer variants handle different dimensionality requirements:

- **`ChannelLinear`**: A shared linear map on the last (channel) axis. Weight shape `(C_in, C_out)`, contraction via `einsum("...i,io->...o")`. Used for pre/post-focus mixing and scalar projections.

- **`FocusLinear`**: Per-focus linear mixing. Weight shape `(C_in, F×C_out)` with the focus dimension folded on the output side. Runtime view `(C_in, F, C_out)`, contraction `einsum("bfi,ifo->bfo")`. Used for gate projections in `GatedActivation`.

- **`SO3Linear`**: Per-degree independent mixing. Weight shape `(lmax+1, C_in, F×C_out)`, expanded to `(D, C_in, F, C_out)` via `index_select` on a precomputed `expand_index` buffer. Contraction `einsum("ndfi,difo->ndfo")`. Bias is applied only to `l = 0`. This is the workhorse linear layer for the FFN and SO(2) channel mixers.

All three store weights in `(in, out)` layout (rows = fan_in, cols = fan_out) to match the Muon optimizer's rectangular correction scaling.

______________________________________________________________________

## 9. Depth Attention Residuals

SeZM provides three independent attention-residual mechanisms that allow later computation stages to selectively attend to earlier representations. All three are implemented by `DepthAttnRes` (in `attn_res.py`) with the same core algorithm but different scopes.

### 9.1 Core Mechanism

`DepthAttnRes` computes a weighted average over a list of source tensors, using softmax attention on `l = 0` scalar features:

```
keys = [ScalarRMSNorm(scalar_extractor(source)) for source in sources]
query = learned_pseudo_query           # "independent" mode
      | ChannelLinear(scalar_extractor(current_x))   # "dependent" mode
logits = Σ_c query_c × key_c          # dot product per source
alpha = softmax(logits)                # over sources
output = Σ_s alpha_s × sources[s]     # weighted sum, full equivariant tensor
```

The query projection is zero-initialized (`init_std=0`), so initial attention weights are uniform across all sources — equivalent to a simple average. The model learns to specialize the attention as training progresses.

When only one source is available, `DepthAttnRes` short-circuits and returns it directly.

### 9.2 Three Attention Scopes

**`so2_attn_res`** — inside each SO(2) convolution. The SO(2) layer stack maintains a local history of intermediate states. Before each SO(2) layer, attention over this local history provides a skip-connection mechanism within the convolution itself. This helps gradient flow through deep SO(2) stacks (`so2_layers` > 2).

**`full_attn_res`** — across the entire descriptor. A global `unit_history` list accumulates every intermediate representation: the initial `x0`, each SO(2) output, and each FFN output across all blocks. Before the SO(2) unit and each FFN unit, attention over this growing history provides long-range skip connections. A final attention aggregation runs before the output FFN.

**`block_attn_res`** — across blocks. A `block_history` list accumulates one summary per block (the sum of all unit outputs within that block). Each FFN unit also attends to `block_history + [partial_block]`, where `partial_block` is the running sum of unit outputs inside the current block.

`full_attn_res` and `block_attn_res` are mutually exclusive. Both accept modes `"none"` (disabled), `"independent"` (learned pseudo-query), and `"dependent"` (query derived from current state).

______________________________________________________________________

## 10. ZBL Zone Bridging

SeZM supplements its learned energy with an analytical short-range repulsion (ZBL potential) that protects MD simulations from unphysical close contacts. The total energy decomposes as:

```
E_total = E_ZBL(r) + E_model(r̃)
```

where `r` is the true pairwise distance and `r̃` is the effective distance seen by the descriptor after the bridging pipeline. The goal is to make the two-body repulsive wall indistinguishable from a pure ZBL potential inside the bridging window.

### 10.1 The Invariance Requirement

For any frozen pair `(j, k)` with `r_{jk} < r_inner`, the model contribution `E_model` must be constant under all motions that keep `(j, k)` in the frozen zone. Only then does differentiating `E_total` reproduce the analytical ZBL force without a parasitic residual from the learned model.

A scalar-distance clamp alone is insufficient. Two channels still leak trajectory-dependence into the descriptor:

1. **Direction channel.** Even with clamped `‖r_j − r_k‖`, the unit direction `r̂_{jk}` still rotates freely. The Wigner-D operator inherits this rotation, and any `l > 0` path picks up angular dependence.

1. **Multi-hop channel.** A third atom `ℓ` connected to both `j` and `k` through unclamped edges acquires frozen-pair information after one message-passing layer. After a second layer, this information propagates back to `j`, making `x_j` depend on the frozen-pair geometry.

### 10.2 InnerClamp

`InnerClamp` (in `radial.py`) maps true distances to effective distances via a C³-continuous septic Hermite interpolant:

```
r̃(r) = r_inner                                             if r ≤ r_inner
      = r_inner + (r_outer − r_inner) × h_clamp(t)         if r_inner < r < r_outer
      = r                                                   if r ≥ r_outer

t = (r − r_inner) / (r_outer − r_inner)
h_clamp(t) = 20t⁴ − 45t⁵ + 36t⁶ − 10t⁷
```

Boundary conditions: `h_clamp(0) = 0`, `h_clamp(1) = 1`, `h_clamp'(0) = h_clamp'(1) = 0`, `h_clamp''(0) = h_clamp''(1) = h_clamp'''(0) = h_clamp'''(1) = 0`.

Both the scalar distance and the displacement vector are clamped. The vector is rescaled to preserve direction while matching the clamped length:

```
scale = r̃ / max(r, eps)
diff = diff × scale
```

This closes the scalar-distance channel: all downstream quantities derived from `length` (radial basis, cutoff envelope, edge type features) see constant geometry for any frozen-zone displacement.

### 10.3 Source Freeze Propagation Gate (SFPG)

SFPG closes both the direction and multi-hop channels. `BridgingSwitch` (in `radial.py`) produces a per-edge C³ switching amplitude:

```
w(r) = 0                                                   if r ≤ r_inner
w(r) = h_switch((r − r_inner) / (r_outer − r_inner))       if r_inner < r < r_outer
w(r) = 1                                                   if r ≥ r_outer

h_switch(t) = 35t⁴ − 84t⁵ + 70t⁶ − 20t⁷
```

The per-node gate is the product of switch values over all outgoing edges:

```
η_j = Π_{e ∈ E(j)} w(r_e)
```

This product is computed with `torch.scatter_reduce(..., reduce="prod", include_self=True)` directly on non-negative reals. No `log`/`exp` detour is needed, so `η_j = 0` is exact when any neighbor enters the frozen zone. Padded/excluded edges contribute the multiplicative identity `w = 1`.

The per-edge broadcast `edge_src_gate[e] = η_{src(e)}` is applied at every aggregation site:

1. **GeometricInitialEmbedding**: zonal message × `edge_src_gate` before `index_add_`
1. **EnvironmentInitialEmbedding**: outer product × `edge_src_gate` before scatter
1. **SO2Convolution (plain path)**: `edge_env × edge_src_gate` in the scatter
1. **SO2Convolution (attention path)**: `edge_src_gate` enters the softmax as `src_weight`, multiplying `edge_env²` in both numerator and denominator. A muted source contributes zero to the normalization, preventing the denominator from "seeing" frozen-zone edges.

**Equivariance.** `η_j` is a product of functions of pairwise distances (SO(3) scalars). Multiplying an equivariant message by this scalar preserves the transformation behavior.

**Smoothness.** `w` is C³. The product `η_j` is C³ in coordinates. `scatter_reduce("prod")` uses the leave-one-out product rule in its backward, which is exact for non-negative inputs including zero.

**Freezing correctness.** By induction on layer index: if `η_j = 0` for frozen nodes `j, k`, then no non-frozen atom `i` can receive frozen-pair information at any layer, so `E_i^{GNN}` is constant under frozen-pair motions. The proof applies identically to both the plain and attention aggregation paths.

### 10.4 Configuration and Defaults

| Parameter          | Type  | Default  | Description                            |
| ------------------ | ----- | -------- | -------------------------------------- |
| `bridging_method`  | str   | `"none"` | `"none"` to disable, `"ZBL"` to enable |
| `bridging_r_inner` | float | 0.8      | Inner radius in Å                      |
| `bridging_r_outer` | float | 1.2      | Outer radius in Å                      |

The default window `[0.8, 1.2]` Å is tuned for general-purpose pretraining on large materials/molecule corpora (OMat24, OMol, MPtraj-class datasets). Three considerations shape the window:

1. **Data coverage.** `r_inner = 0.8 Å` places the lower edge of the transition zone where the combined datasets carry edges. Choosing lower would leave the sub-0.8 Å region as pure extrapolation.

1. **Bond sensitivity.** Common short bonds land on healthy parts of `dr̃/dr`: O-H (0.96 Å, `h'≈1.29`), N-H (1.01 Å), C-H (1.09 Å). Raising `r_inner` above 0.8 would push these into the low-derivative shoulder.

1. **Numerical smoothness.** The 0.4 Å width keeps `max|d²r̃/dr²| ≈ 24 Å⁻¹`, keeping second-derivative training stable. Narrower windows inflate the curvature.

### 10.5 InterPotential (ZBL)

The `InterPotential` module (in `sezm_model.py`) computes the Ziegler-Biersack-Littmark screened nuclear repulsion:

```
V_ZBL(r) = (ke × Zi × Zj / r) × φ(r / a)
a = 0.88534 × a_bohr / (Zi^0.23 + Zj^0.23)
φ(x) = 0.18175·exp(−3.1998x) + 0.50986·exp(−0.94229x)
      + 0.28022·exp(−0.4029x) + 0.02817·exp(−0.20162x)
```

Each pair `(i, j)` contributes `V_ZBL / 2` to atom `i` (symmetric neighbor list). The ZBL energy is added to `atom_energy` before autograd computes forces/virials on the main sparse-edge path, and into `fit_ret["energy"]` before `fit_output_to_model_output` on the lower extended-coordinate path.

______________________________________________________________________

## 11. Fitting Network

The SeZM fitting net (`sezm_ener`) maps the scalar descriptor `(nf, nloc, channels)` to per-atom energies. It uses the same configuration keys as the standard DeePMD energy fitting (`neuron`, `activation_function`, `precision`, `seed`, ...).

- `neuron = []` produces a direct linear projection from `channels` to scalar energy (no hidden layers).
- When `neuron` is non-empty, each hidden layer is a GLU block: `Linear(in, 2×hidden) → split → value × act(gate)`. The internal hidden width is therefore double the user-specified value (e.g., `hidden=256` creates a 512-wide layer before the split).

The `dens` path (direct-force / denoising head, inspired by EquiformerV3 DeNS) reuses the same fitting configuration for its scalar energy branch and adds a parallel direct-force head. This path is activated by `loss.type = "dens"` and materialized lazily — training creates it when the loss type requires it, and checkpoint loading recreates it when `dens` weights are present. The `dens` path is retained for experimental purposes but is not the primary training mode.

______________________________________________________________________

## 12. Model Integration

### 12.1 SeZMModel

Set `model.type = "SeZM"` (alias `"sezm"`) to select the SeZM model scaffold. Internally it is built as `make_model(SeZMAtomicModel)`.

`descriptor.type` follows user input, and `fitting_net.type` is ignored — SeZM always uses `sezm_ener`. The fitting configuration is shared with the `dens` head when present.

**Mode routing** is selected by `loss.type`:

- `loss.type = "ener"` → conservative energy/autograd-force path
- `loss.type = "dens"` → parallel direct-force/denoising path

The LAMMPS-style interface (`forward_lower`) supports only the `ener` path.

### 12.2 torch.compile Path

SeZM supports an end-to-end `torch.compile` path for `ener` force-loss training. The implementation lives in `sezm_model.py` and is designed to survive Inductor's compilation of second-order coordinate derivatives — the gradient of the force loss with respect to model parameters passes through both the energy-to-force `autograd.grad` and the optimizer's backward pass.

**Enabling compile:** Set `model.use_compile = true`. Training follows `use_compile`; eval/inference defaults to eager. Set `DP_COMPILE_INFER=1` before model construction to opt evaluation into the compile path.

**The unified kernel** `core_compute()` handles both eager and compile paths. It builds compact sparse edges from the DeePMD neighbor list (outside the compiled graph), runs the descriptor and fitting, then calls `autograd.grad` for force/virial with `create_graph=self.training`.

**Trace strategy for `ener`:** `make_fx(tracing_mode="symbolic", _allow_non_fake_inputs=True)` captures the inner `autograd.grad` as ordinary FX nodes. Then `torch.compile(dynamic=True, backend="inductor")` compiles the resulting graph. The `make_fx` step is essential: it turns the second-derivative `autograd.grad` into a flat FX graph that Inductor can lower without needing to re-derive the second backward at compile time.

**Key compile invariants** (each maps to a `NOTE:` tag in `sezm_model.py`):

| Invariant                                                 | Why                                                                                                                                                                                                                                      |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Trace inputs use `nf=2`**                               | `nf=1` triggers 0/1 specialization; `nf=3` collides with spatial dim 3 in `extended_coord (nf, nall, 3)`                                                                                                                                 |
| **`silu_backward` decomposed**                            | PyTorch has no registered higher-order derivative for SiLU; the opaque `silu_backward` would crash Inductor's double-backward                                                                                                            |
| **Detach chains stripped in training**                    | `make_fx` under `create_graph=True` wraps saved activations in `fwd → detach → detach → bwd` chains that sever gradient flow. The stripper uses graph topology to remove chain-inner detaches while preserving user-explicit `.detach()` |
| **FX graph rebuilt after stripping**                      | `Graph.erase_node` leaves stale C-level pointers on some builds; a fresh `node_copy` pass prevents segfaults                                                                                                                             |
| **DDPOptimizer disabled**                                 | Set at import time. DDPOptimizer splits graphs at bucket boundaries, producing subgraphs with symbolic-integer outputs that crash AOT Autograd                                                                                           |
| **`index_select` for edge vectors**                       | Advanced indexing under `make_fx` has silently produced zero second-derivative gradients in this project                                                                                                                                 |
| **Dummy edge appended**                                   | `torch.nonzero(valid_mask)` is data-dependent; an empty result cannot be traced symbolically. The dummy edge (`edge_mask=False`) keeps every tensor at length ≥ 1                                                                        |
| **Compiled callable via `object.__setattr__`**            | `nn.Module.__setattr__` would register the wrapper as a submodule, exposing duplicate parameter views to FSDP2/DDP                                                                                                                       |
| **Extended coords rebound per forward**                   | `.detach().requires_grad_(True)` severs upstream autograd and provides a known-shape grad endpoint for symbolic tracing                                                                                                                  |
| **Compile cache keyed on `(training, do_atomic_virial)`** | Both toggle graph structure (second-derivative branch / extra output tensor)                                                                                                                                                             |

**Inductor options** locked in `trace_and_compile`: `max_autotune=False`, `shape_padding=True`, `epilogue_fusion=False`, `triton.cudagraphs=False`, `max_fusion_size=64`, `triton.mix_order_reduction=False`. Each setting addresses a specific failure mode under dynamic shapes or higher-order gradients.

**`dens` compile:** Since the direct-force head needs no second coordinate derivative, `dens` compiles `core_compute_dens` directly with `torch.compile` (no `make_fx`). Limitation: `dens` does not support analytical bridging potentials.

______________________________________________________________________

## 13. Weight Layout and Optimizer Routing

### 13.1 Unified (in, out) Convention

All learnable weight matrices store rows as fan_in and columns as fan_out. Focus-aware modules fold the focus dimension `F` on the output (cols) side:

| Module          | Stored shape                     | Runtime contraction                                           |
| --------------- | -------------------------------- | ------------------------------------------------------------- |
| `ChannelLinear` | `(C_in, C_out)`                  | `einsum("...i,io->...o")`                                     |
| `FocusLinear`   | `(C_in, F×C_out)`                | view `(C_in, F, C_out)` → `einsum("bfi,ifo->bfo")`            |
| `SO3Linear`     | `(lmax+1, C_in, F×C_out)`        | expand to `(D, C_in, F, C_out)` → `einsum("ndfi,difo->ndfo")` |
| `SO2Linear`     | `(D_m×C_in, F×D_m×C_out)` per \` | m                                                             |

This layout ensures the Muon optimizer's rectangular correction `scale = sqrt(max(1, rows/cols))` stays at 1.0 when `C_in ≤ F×C_out` (typical), avoiding step-size inflation. Each semantically independent weight block gets its own Muon Newton-Schulz update.

### 13.2 HybridMuon Routing

HybridMuon uses name-based routing to separate optimizer paths (case-insensitive matching on the final effective parameter name segment):

| Pattern              | Optimizer | Weight decay |
| -------------------- | --------- | ------------ |
| Contains `bias`      | Adam      | none         |
| Starts with `adam_`  | Adam      | none         |
| Starts with `adamw_` | AdamW     | decoupled    |
| 2D tensor (default)  | Muon      | —            |
| Other shape          | AdamW     | decoupled    |

SeZM parameters use `adam_` prefixes for norm scales, layer scales, radial frequencies, and type embeddings (`adam_scale`, `adam_so2_layer_scales`, `adam_ffn_layer_scales`, `adam_freqs`, `adam_type_embedding`) so they route to Adam without weight decay.

**Recommended mode:** `muon_mode = "slice"`:

- 2D weights (ChannelLinear, SO2Linear, FocusLinear): standard Muon
- 3D SO3Linear `(lmax+1, C_in, C_out)`: per-(l) independent Muon with correct rectangular scale
- `adam_`/`bias` parameters: Adam (name-based routing takes priority)

### 13.3 Optional Magma-lite Damping

HybridMuon supports optional Muon-path damping via `training.magma_muon = true`:

1. **Alignment.** Compute per-block cosine alignment between Muon momentum `m_t` and current gradient `g_t`.
1. **Temperature sigmoid.** `raw = sigmoid(cos / tau)` with `tau = 2.0`, then stretch to `[0, 1]`.
1. **EMA smoothing.** `score_t = 0.9 × score_{t-1} + 0.1 × raw`.
1. **Final scale.** `scale = 0.1 + 0.9 × score_t`, always in `[0.1, 1.0]`.
1. **Apply.** `delta_muon = delta_muon × scale`.

Strong momentum-gradient agreement yields larger scales; poor alignment damps the update. The `0.1` floor prevents complete starvation of hard-to-optimize blocks. No Bernoulli skip masking is used (unlike the original Magma paper) because force-field training is sensitive to intermittent block freezing.

______________________________________________________________________

## 14. Initialization Strategy

### 14.1 Deterministic Seeding

All submodules derive seeds from `child_seed(seed, idx)`. Repeated structures (blocks, SO2 layers, FFN subblocks) include loop indices in the seed derivation. When `seed=None`, initialization follows the global RNG.

### 14.2 Weight Initialization

| Component                         | Method                                                                                       |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| SO2Linear                         | `TruncatedNormal(0, 1/sqrt(fan_in+fan_out), ±3σ)`. `\|m\|>0` blocks: extra `1/sqrt(2)` scale |
| SO3Linear                         | `TruncatedNormal(0, 1/sqrt(fan_in+fan_out), ±3σ)`                                            |
| ChannelLinear, FocusLinear        | Same truncated normal scheme                                                                 |
| EnvironmentInitialEmbedding MLPs  | `TruncatedNormal(0, sqrt(2/(fan_in+fan_out)), ±3σ)`                                          |
| Output projections (FFN, SO2Conv) | Zero-initialized (weights + bias) via `init_std=0`                                           |
| Gate projections                  | `Normal(0, 0.01)` for weights, zeros for bias                                                |
| LayerScale                        | Init to `1e-3`                                                                               |
| FiLM strength logs                | Init to `log(0.01)`                                                                          |
| DepthAttnRes query                | Zero-initialized                                                                             |

### 14.3 Near-Identity Start

The zero-initialization of output projections and the small-value initialization of gates and layer scales together ensure that SeZM starts training near identity:

- Each residual branch `x = x + output_proj(...)` starts as `x = x + 0 = x`.
- Gates start at `sigmoid(0) ≈ 0.5`, providing maximum gradient flow (`sigmoid'(0) = 0.25`).
- LayerScale starts at `1e-3`, keeping residual contributions small initially.
- FiLM starts as `scale ≈ 1, shift ≈ 0`, preserving the type embedding.
- DepthAttnRes starts as uniform average over sources.

This strategy prevents early training instability from large random perturbations in the equivariant feature space.

______________________________________________________________________

## 15. Precision and Numerical Safety

### 15.1 Compute Dtype Promotion

SeZM separates the working dtype (for interaction blocks) from the compute dtype (for geometry and critical operations):

```
dtype = PRECISION_DICT[precision]          # e.g., float32
compute_dtype = get_promoted_dtype(dtype)  # float32 if dtype is float32, else float32
```

Geometry computations (edge distances, quaternions, Wigner-D, GIE, radial basis, environment embedding, norms, output FFN) always run in `compute_dtype` (fp32+). Interaction blocks use the working `dtype`. This prevents the accumulation of half-precision rounding errors in geometric quantities that directly affect the PES shape.

### 15.2 Automatic Mixed Precision

When `use_amp=True` (default) and the model is training on CUDA, interaction blocks run under `torch.autocast("cuda", dtype=torch.bfloat16)`. This reduces memory usage for the large edge-level tensors inside SO(2) convolution without affecting model accuracy — autocast-eligible operations (matmul, bmm) use bf16 while reductions and norms stay in fp32.

AMP is disabled during inference (no speed benefit from autocast on the forward-only path). All norm modules explicitly disable autocast to prevent precision loss in RMS computation.

### 15.3 Numerical Safety Measures

| Situation                      | Protection                                                                 |
| ------------------------------ | -------------------------------------------------------------------------- |
| Vector normalization near zero | `sqrt(clamp(‖x‖², eps²))`                                                  |
| Quaternion normalization       | Same clamped sqrt; fp16/bf16 promoted to fp32                              |
| Division by distance           | `max(r, eps)` or `r.clamp(min=eps)`                                        |
| Degree normalization           | `rsqrt(deg + eps)`                                                         |
| Softmax stability              | Per-destination max subtraction                                            |
| Softmax denominator            | `softplus(z_bias) × exp(−max)` prevents zero denominator                   |
| Padding edges                  | Displacement reset to `(0, 0, 1)`, `edge_env = 0`                          |
| RadialMLP bias                 | `bias=False` throughout to prevent constant leakage from padding           |
| SO2Linear bias                 | First-layer bias modulated by `(radial_l0 × edge_env − 1)` for consistency |

### 15.4 Conservative Force Guarantee

The entire geometry chain — `edge_vec → quaternion → Wigner-D → rotation → aggregation` — is fully differentiable. No `.detach()` is applied to edge rotations or geometric quantities. This ensures that forces computed via `autograd.grad(energy, coord)` are truly conservative (path-independent, derivable from a scalar potential).

______________________________________________________________________

## 16. Tensor Layouts

### 16.1 Node Features

The backbone tensor throughout the descriptor is `(N, D, 1, C)` (contiguous), where:

- `N = nf × nloc` — flattened batch×atom
- `D = (lmax+1)²` — SO(3) embedding dimension
- `1` — singleton focus axis (kept for module reuse; real multi-focus lives inside SO2Convolution)
- `C = channels` — per-coefficient channel width

Features are packed by increasing `l`. Within each `l` block, `m` runs from `−l` to `+l`:

| Degree      | Indices         | Count  |
| ----------- | --------------- | ------ |
| `l=0`       | `[0:1]`         | 1      |
| `l=1`       | `[1:4]`         | 3      |
| `l=2`       | `[4:9]`         | 5      |
| `l=3`       | `[9:16]`        | 7      |
| general `l` | `[l² : (l+1)²]` | `2l+1` |

View conventions inside blocks:

- `x.view(N, D, C)` for full-channel rotation and FFN
- `x.view(N, D, 1, C)` at block boundaries
- Inside SO2Convolution: `(E, F, D_m, Cf)` for multi-focus SO(2) stack

### 16.2 Edge Cache Tensors

All edge cache tensors hold **valid edges only** (padding and excluded type pairs removed). The edge count `E` varies per forward call. Key shapes:

| Tensor              | Shape           | Notes                               |
| ------------------- | --------------- | ----------------------------------- |
| `src`, `dst`        | `(E,)`          | Node indices in `[0, N)`            |
| `edge_vec`          | `(E, 3)`        | Displacement in Å                   |
| `edge_type_feat`    | `(E, C)`        | `type_embed[src] + type_embed[dst]` |
| `edge_rbf`          | `(E, n_radial)` | Radial basis with envelope          |
| `edge_env`          | `(E, 1)`        | C³ cutoff envelope                  |
| `D_full`, `Dt_full` | `(E, D, D)`     | Block-diagonal Wigner-D             |
| `inv_sqrt_deg`      | `(N, 1, 1)`     | Degree normalization                |
| `edge_src_gate`     | `(E, 1)`        | SFPG gate (or `None`)               |

Edges with `r ≥ rcut` are kept in the cache (not filtered) because their `edge_env = 0` naturally zeros their messages. This avoids the `torch.nonzero` dynamic-shape kernel and keeps the smooth degree free of discontinuous jumps.

### 16.3 Reduced SO(2) Layout

Inside SO(2) convolution, only coefficients with `|m| ≤ mmax` are retained. The reduced dimension is:

```
D_m = Σ_{l=0}^{lmax} (2 × min(l, mmax) + 1)
```

Coefficients are stored in m-major order: all `l`-values for `m=0`, then all `l`-values for `m=−1,+1`, then `m=−2,+2`, etc. Two precomputed buffers map between the full and reduced layouts:

- `coeff_index_m`: maps reduced positions to full `D` indices (for row-select in `D_to_m`)
- `degree_index_m`: maps reduced positions to degree `l` (for radial feature lookup)

______________________________________________________________________

## 17. VRAM Estimation

### 17.1 Notation

| Symbol | Meaning                                    | Source       |
| ------ | ------------------------------------------ | ------------ |
| P      | Total trainable parameters                 | training log |
| N      | Atoms per frame                            | system       |
| nnei   | Max neighbors (sel)                        | config       |
| E      | Edges = N × nnei                           | derived      |
| L      | max(l_schedule)                            | config       |
| D      | (L+1)²                                     | derived      |
| C      | channels                                   | config       |
| F      | Effective FFN hidden dim                   | config/auto  |
| B      | Number of blocks                           | config       |
| b      | Bytes per element (4 for fp32, 2 for bf16) | dtype        |

### 17.2 Inference

```
M_infer ≈ E × D × b × [2D + (B+1)×C + max(C, 2F)]
                         ^^^   ^^^^^     ^^^^^^^^^^
                       Wigner  radial    transient peak
```

The three terms:

1. **Wigner-D matrices** `2 × E × D² × b`: `D_full + Dt_full`, shared across all blocks.
1. **Radial features** `B × E × D × C × b`: one truncated slice per block.
1. **Transient peak** `E × D × max(C, 2F) × b`: the larger of SO2Conv intermediate and FFN up-projection. Only one block is active at a time during inference.

### 17.3 Training

During training, autograd saves intermediate tensors across all blocks simultaneously:

```
M_train ≈ E × D × b × [2D + B × (1+k) × C]     (k ≈ 5)
```

The factor `k ≈ 4–6` accounts for saved tensors: pre-norm input, rotation result, message, FFN up-projection, residual inputs. Training uses roughly **5–8× inference memory** due to these saved activations.

### 17.4 Scaling Summary

| Factor       | Scaling                           | Note                         |
| ------------ | --------------------------------- | ---------------------------- |
| N (atoms)    | Linear                            | E = N × nnei                 |
| nnei         | Linear                            | E = N × nnei                 |
| C (channels) | Linear                            | dominates E×D×C              |
| L (lmax)     | **Quadratic**                     | D = (L+1)², Wigner-D is E×D² |
| B (blocks)   | Linear (train) / Constant (infer) |                              |
| F (ffn)      | Linear                            | only transient peak          |

**Bottleneck:** Edge-level tensors (E×D×C) and Wigner-D matrices (E×D²). Reducing `nnei` or `L` yields the largest memory savings.

______________________________________________________________________

## 18. Serialization

`DescrptSeZM.serialize()` produces a flat dictionary:

```python
{
    "@class": "Descriptor",
    "type": "SeZM",
    "@version": 1,
    "config": { ... all constructor arguments ... },
    "@variables": { key: numpy_array for key, array in state_dict },
    "env_mat": DPEnvMat(rcut, rcut, eps).serialize(),
}
```

`@variables` contains the full `state_dict()` payload, including all `register_buffer` tensors (precomputed index tables, S2 projection matrices, reduced-layout maps, Wigner coefficient tables, interface-compatibility buffers). Serialization is flat at the descriptor level — no recursive per-submodule packing.

`DescrptSeZM.deserialize(data)` reconstructs the descriptor from `config`, then restores the state dict. Transient buffers rebuilt at construction time are dropped by `_load_from_state_dict` when loading from older checkpoints.

______________________________________________________________________

## 19. DeePMD Interface Compatibility

SeZM follows the new-style descriptor interface (same as `dpa3`), using `extended_coord` / `extended_atype` parameter names.

**Implemented interfaces:**

- `forward()` with the standard 5-tuple return `(descriptor, rot_mat, g2, h2, sw)`. Only `descriptor` is meaningful; the rest are empty tensors.
- `get_rcut()`, `get_sel()`, `get_dim_out()`, `get_dim_emb()`
- `mixed_types() → True` (unified neighbor list, no type distinction)
- `serialize()` / `deserialize()`
- `compute_input_stats()` — no-op (SeZM uses learnable RMSNorm)
- `update_sel()` for automatic neighbor selection

**Interface compatibility details:**

- `_ENV_DIM = 1` (se_r style) for `EnvMatStatSe` compatibility
- `ndescrpt = nnei × 1`
- `mean` and `stddev` buffers are maintained but not used in the forward pass
- Output: only `l=0` features as descriptor `(nf, nloc, channels)`

**Not implemented:** `share_params()`, `change_type_map()`, `enable_compression()`.

______________________________________________________________________

## 20. Configuration Reference

### 20.1 Descriptor Parameters

| Parameter             | Type              | Default     | Description                                                       |
| --------------------- | ----------------- | ----------- | ----------------------------------------------------------------- |
| `rcut`                | float             | —           | Cutoff radius in Å                                                |
| `sel`                 | int \| list[int]  | —           | Max neighbors (int: total, list: per-type)                        |
| `env_exp`             | list[int]         | `[7, 5]`    | Envelope exponents `[rbf_env_exp, edge_env_exp]`                  |
| `channels`            | int               | 64          | Total channels per `(l,m)` coefficient                            |
| `n_radial`            | int               | 10          | Number of radial basis functions                                  |
| `radial_mlp`          | list[int]         | `[64]`      | Hidden sizes for radial MLP                                       |
| `use_env_seed`        | bool              | True        | Enable FiLM conditioning from environment matrix                  |
| `random_gamma`        | bool              | True        | Random edge roll for data augmentation                            |
| `lmax`                | int               | 2           | Max degree (when `l_schedule` is None)                            |
| `n_blocks`            | int               | 2           | Number of blocks (when `l_schedule` is None)                      |
| `l_schedule`          | list[int] \| None | None        | Pyramid schedule of lmax per block (non-increasing)               |
| `mmax`                | int \| None       | None        | Max SO(2) order (when `m_schedule` is None)                       |
| `m_schedule`          | list[int] \| None | None        | Schedule of mmax per block                                        |
| `n_focus`             | int               | 1           | Parallel focus streams in SO(2) convolution                       |
| `focus_dim`           | int               | 0           | Per-focus hidden width (0 = use `channels`)                       |
| `n_atten_head`        | int               | 1           | Attention heads in SO(2) aggregation (0 = plain scatter)          |
| `so2_norm`            | bool              | False       | Pre-norm between SO(2) layers                                     |
| `so2_layers`          | int               | 4           | SO2Linear layers per convolution                                  |
| `so2_attn_res`        | str               | `"none"`    | SO(2)-internal depth attention (`none`/`independent`/`dependent`) |
| `ffn_neurons`         | int               | 0           | FFN hidden width (0 = auto from channels)                         |
| `grid_mlp`            | bool              | False       | Grid-MLP FFN variant                                              |
| `ffn_blocks`          | int               | 1           | FFN subblocks per interaction block                               |
| `sandwich_norm`       | list[bool]        | `[T,F,T,F]` | `[so2_pre, so2_post, ffn_pre, ffn_post]`                          |
| `mlp_bias`            | bool              | False       | Bias in equivariant layers                                        |
| `layer_scale`         | bool              | False       | Learnable LayerScale (init 1e-3)                                  |
| `full_attn_res`       | str               | `"none"`    | Descriptor-level full attention residual                          |
| `block_attn_res`      | str               | `"none"`    | Descriptor-level block attention residual                         |
| `s2_activation`       | list[bool]        | `[F, F]`    | `[so2_s2_enabled, ffn_s2_enabled]`                                |
| `s2_grid_resolution`  | list[int] \| None | None        | `[R_phi, R_theta]` for S2-grid activation                         |
| `activation_function` | str               | `"silu"`    | Base activation                                                   |
| `glu_activation`      | bool              | True        | Base GLU switch for FFN                                           |
| `use_amp`             | bool              | True        | AMP with bf16 during training on CUDA                             |
| `precision`           | str               | `"float32"` | Working precision for interaction blocks                          |
| `eps`                 | float             | 1e-7        | Numerical stability epsilon                                       |
| `exclude_types`       | list[tuple]       | `[]`        | Excluded type pairs                                               |

### 20.2 Model Parameters

| Parameter                | Type  | Default  | Description               |
| ------------------------ | ----- | -------- | ------------------------- |
| `model.type`             | str   | —        | `"SeZM"` / `"sezm"`       |
| `model.use_compile`      | bool  | False    | Enable torch.compile path |
| `model.bridging_method`  | str   | `"none"` | `"none"` or `"ZBL"`       |
| `model.bridging_r_inner` | float | 0.8      | Inner radius in Å         |
| `model.bridging_r_outer` | float | 1.2      | Outer radius in Å         |

### 20.3 Environment Variables

| Variable             | Effect                                           |
| -------------------- | ------------------------------------------------ |
| `DP_TRITON=1`        | Enable Triton SO(2) rotation kernels (eval only) |
| `DP_COMPILE_INFER=1` | Opt eval/inference into the compile path         |

### 20.4 FFN Width Auto-Resolution

When `ffn_neurons = 0`, the hidden width is computed from `channels`:

- With GLU: `ceil_to_32(8/3 × channels)` — e.g., channels=64 → 192, channels=128 → 352
- Without GLU: `ceil_to_32(4 × channels)` — e.g., channels=64 → 256, channels=128 → 512

______________________________________________________________________

## 21. Testing and Validation

### 21.1 Test Organization

| File                                                   | Scope                                                     |
| ------------------------------------------------------ | --------------------------------------------------------- |
| `source/tests/pt/model/test_descriptor_sezm.py`        | Descriptor: forward pass, serialization, smoothness, SFPG |
| `source/tests/pt/model/test_descriptor_sezm_triton.py` | Triton kernels: rotation, geometry/RBF chain              |
| `source/tests/pt/model/test_sezm_model.py`             | Model: compile path, energy/force correctness             |

### 21.2 PES Smoothness Validation

The primary smoothness test uses direct **total-energy scans** rather than force-vs-finite-difference checks. A force check can pass even when the PES contains non-physical kinks.

**Probe setup:** An eight-atom two-sublattice template in fractional coordinates:

```
[0,0,0], [0,½,½], [½,0,½], [½,½,0], [½,½,½], [½,0,0], [0,½,0], [0,0,½]
```

The cubic lattice is scaled so the nearest-neighbor distance matches three boundary conditions:

1. **Near-cutoff** (`r_nn = 4.95 Å`, `rcut = 5.0 Å`): probes the cutoff envelope
1. **Inner boundary** (`r_nn = r_inner`): probes the InnerClamp transition (with ZBL)
1. **Outer boundary** (`r_nn = r_outer`): probes the bridging-to-identity transition (with ZBL)

Atom 0 is displaced along `x` over `[−0.1, 0.1]` Å. The test validates:

1. The second derivative maintains one sign across the scan (no inflection points)
1. The first derivative changes sign once at the center (single extremum)
1. The energy curve is a single smooth bowl

With randomized model weights, the non-bridged curve may be either concave or convex; both are acceptable. Bridged probes should be convex (ZBL repulsion dominates).

### 21.3 SFPG Invariance Test

A three-atom setup verifies that the SFPG correctly freezes information propagation:

1. Fix the frozen-partner atom on a constant-radius sphere inside the frozen zone
1. Slide the partner along the sphere through multiple angular positions
1. Assert the anchor atom's energy is invariant (span < `1e-10` in fp64)
1. Ablation: clear `bridging_switch` and verify the span jumps by orders of magnitude

### 21.4 Caching Strategy

Triton kernel correctness is validated by comparing Triton output against the eager PyTorch reference path with tight numerical tolerances. The comparison covers both forward and backward passes for each rotation kernel family (`SMALL_LE1`, `SMALL_L2`, `SMALL_L3`, `GENERIC_TILED`) and the fused geometry/RBF chain.
