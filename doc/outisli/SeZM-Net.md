# SeZM-Net: Smooth equivariant ZBL Message-passing Network

SeZM-Net is a small-`l` equivariant message passing descriptor designed for molecular dynamics (MD) workloads where **inference speed** and **physical correctness** (conservative forces + smooth PES) dominate.

This document is the **final spec** for the `se_zm_net` PyTorch descriptor implemented in:

- `deepmd/pt/model/descriptor/se_zm_net.py`
- `deepmd/pt/model/descriptor/se_zm_block.py`

---

## Goals (Non-Negotiable)

1. **Conservative forces**
   - SeZM-Net outputs features meant for an energy model; forces must come from `autograd` of energy w.r.t. coordinates.
   - Geometry / rotations are fully differentiable; **no `.detach()`** in edge rotations.

2. **Smooth cutoff**
   - Every edge message is multiplied by a **C² polynomial envelope** that goes to **exactly 0 at `rcut`**, so message and its derivatives vanish at `rcut`.

3. **Strict padded neighbor masking**
   - DeePMD neighbor lists are padded (typically with `-1` indices).
   - Padding and excluded type-pairs must contribute **exactly zero** and must not introduce NaNs.

4. **Speed mandate: single geometry/Wigner pass**
   - Edge geometry and Wigner-D rotation blocks are computed **once per `forward()`** and reused by all blocks.
   - No interaction block is allowed to recompute:
     - `edge_vec`
     - envelope / radial basis
     - edge-aligned rotation matrices
     - Wigner-D blocks

## Non-Goals (Explicitly Out of Scope)

- ZBL gating / short-range repulsion: only a **clean placeholder hook** is provided.
- Top-K neighbor selection: **not allowed**. Use `rcut` and masks only.
- External equivariant libraries (e.g. e3nn): **not allowed**.

---

## High-Level Architecture

Text diagram (single forward pass):

```
Inputs (extended_coord, extended_atype, nlist, mapping)
  └─ EdgeFeatureCache (built once)
       ├─ edges: (src, dst), edge_vec
       ├─ edge_type_feat: per-edge type embedding (src+dst)
       ├─ edge_rbf: Bessel radial basis via sinc × C² envelope (trainable frequencies)
       ├─ edge_sw: C² cutoff envelope weights in flattened edge layout
       ├─ D_full, Dt_full: block-diagonal Wigner-D matrices
       └─ inv_sqrt_deg: inverse sqrt degree for normalization

Radial embedding (computed once):
  └─ radial_feat: (E, lmax+1, C) via RadialMLP(edge_rbf)
     └─ fused once with edge_type_feat after GIE

Node init:
  ├─ l=0: Type embedding + (optional) EnvironmentInitialEmbedding
  └─ l>0: Zonal (m=0) initial embedding via cached Wigner-D + radial_feat[:, 1:, :]

Interaction blocks (pyramid schedule):
  for block i:
    ├─ slice x to ebed_dim(l_schedule[i]) (discard higher-l if needed)
    ├─ SeparableRMSNorm (pre-SO2)
    ├─ SO(2) Convolution (enabled for ALL lmax, including lmax=0)
    │  ├─ Multi-layer SO(2) mixing
    │  └─ Final SO3Linear channel mixing
    ├─ Residual
    ├─ SeparableRMSNorm (pre-FFN)
    └─ Full Equivariant FFN (operates on ALL degrees l=0..lmax)
       ├─ SO3Linear (in projection)
       ├─ GatedActivation (per-l independent gates from l=0)
       └─ SO3Linear (out projection)
      + Residual

Output (forward, promoted dtype):
  └─ Extract x(l=0) from block output
  └─ Convert to promoted dtype (float32+)
  └─ Final SO3Linear (lmax=0) for channel mixing
     └─ (nf, nloc, channels)
```

---

## Key Design Decisions

### 1. Trainable radial basis frequencies

- Frequencies are trainable parameters initialized as `n * pi / rcut` for `n=1..n_radial`.
- Basis uses a sinc form for stable `r -> 0` gradients: `phi_n(r) = w_n * sinc(w_n * r / pi)`.
- Frequencies are serialized/deserialized.

### 2. SeparableRMSNorm with Degree Balancing

- Normalizes `l=0` and `l>0` **separately** (separable design).
- **Degree Balancing** for `l>0`: each degree `l` contributes equally to the variance, regardless of the number of m components (`2l+1`). This is achieved by weighting each m component by `1/(2l+1)`.
- Centering and bias are applied only to `l=0`; `l>0` remains zero-mean.
- Per-l affine scales are expanded to all coefficients via a precomputed degree index.
- Memory optimization: fused einsum avoids allocating an intermediate `(N, D-1, C)` tensor.

### 3. SO3LinearV2

- Degree-wise channel mixing with `weight[l]` shared across all `m` in the `l` block.
- Implemented as `index_select(expand_index)` + `einsum`; bias only for `l=0`.

### 4. Full equivariant FFN

- `SO3LinearV2 -> GatedActivation -> SO3LinearV2` with a residual connection.
- Gates are derived from `l=0` scalars (one gate per `l`) and expanded across all `m`.

### 5. Multi-layer SO(2) convolution

- Uses the edge-local m-major reduced layout controlled by `mmax`.
- Stacks `SO2Linear` with optional `GatedActivation(mmax=...)` for `so2_layers`.
- Ends with a `SO3LinearV2` channel mixer before aggregation.

### 6. Deterministic initialization

- All submodules derive seeds via `child_seed(seed, idx)`; repeated structures include loop indices.
- If `seed=None`, initialization follows the global RNG.

### 7. Gate initialization strategy

All gate projections use a consistent initialization:

- **Matrix**: `Normal(mean=0, std=0.01)` with reproducible generator
- **Bias**: zeros

This ensures gate logits start near 0, making `sigmoid(0) ≈ 0.5` and `2*sigmoid(0) ≈ 1.0`.

**Benefits**:

- Maximum gradient flow: `sigmoid'(0.5) = 0.25` is the maximum derivative value
- Unbiased feature scaling: the model learns which features to suppress/amplify from the loss signal
- Near-identity initialization for `2*sigmoid` gates preserves initial magnitude

### 8. Stability defaults

- Residual branches start near-identity: the output projections of both SO(2) convolution and Equivariant FFN are zero-initialized (weights + bias).
- Attention aggregation runs in promoted dtype (fp32) and clamps logits to `[-6, 6]` before `sigmoid`.
- Message feedback into edge gates is bounded: `edge_logits = dst + radial + λ * msg` with `λ ∈ (0, λ_max)` and `λ_max = 0.2`.
- Multi-layer SO(2) stacks optionally insert a reduced-layout separable RMSNorm between layers (except the last) when `so2_norm=True`. This keeps truncated m-major activations balanced but is disabled by default.

### 9. Optional environment matrix initial embedding (EnvironmentInitialEmbedding)

An optional module that provides physical inductive bias for l=0 features using a 4D environment matrix approach. When `use_env_seed=True`:

**Key design: Type embedding decoupling**

- Uses an **independent** `env_type_embed` (TypeEmbedNet) instead of projecting from the main type embedding
- This allows `env_seed` to learn type representations independent from the main descriptor backbone
- RBF projection (`rbf_proj`) aligns G-network input dimension to approximately `channels`

**Computation pipeline**:

1. **r_tilde construction**: For each edge, build a 4D vector `[s, s*rx, s*ry, s*rz]` where:
   - `s = edge_env / r` (smooth weight divided by distance)
   - `r_hat = edge_vec / r` (unit direction vector)
   - `r_tilde = [s, s * r_hat]` encodes both radial decay and angular information

2. **G network**: Computes per-edge filter features:
   - RBF projection: Two-layer MLP `rbf_proj_layer1 → rbf_proj_layer2` with dimension `rbf_out_dim = max(32, channels - 2*type_dim)`
     - First layer: `n_radial → rbf_out_dim` with activation (SiLU)
     - Second layer: `rbf_out_dim → rbf_out_dim` linear
   - Type embeddings: `type_src, type_dst = env_type_embed(atype[src]), env_type_embed(atype[dst])`
   - Input: `concat([rbf_proj, type_src, type_dst])` with dimension `(E, rbf_out_dim + 2*type_dim) ≈ channels`
   - Two-layer MLP: `hidden_dim` → `embed_dim` with SiLU activation

3. **env_agg (environment aggregation)**: Vectorized outer product and scatter:
   - `outer = r_tilde[:, :, None] * g[:, None, :]` produces `(E, 4, embed_dim)`
   - Scatter-sum by destination node: `env_agg.index_add_(0, dst, outer_flat)`
   - Normalize by neighbor count (degree normalization)

- **Triton optimization**: When `use_triton=True` and Triton is available on CUDA, a fused kernel computes the outer-product and scatter-sum in one pass, avoiding the `(E, 4, embed_dim)` intermediate tensor. Backward and double-backward are implemented as custom Triton kernels to keep force training on the Triton path.

4. **D matrix construction**: Captures local geometry via matrix product:
   - `D = env_agg^T @ env_agg[:, :, :axis_dim]` with shape `(N, embed_dim, axis_dim)`

5. **Output projection to FiLM deltas**:
   - Flatten D to `(N, embed_dim * axis_dim)` and project to `(N, 2*channels)`
   - Split to `(scale_logits, shift)` and apply FiLM on `l=0`:
     `scale = 1 + env_film_scale_delta * (2 * sigmoid(scale_logits) - 1)`
     `x0 = x0 * scale + shift`
   - Output projection is zero-initialized → `scale_logits=0`, `shift=0` at init

**Key properties**:

- Uses **global frame** direction (not edge-aligned local frame) to preserve angular information
- Neighbor count normalization uses actual degree, not `inv_sqrt_deg` from edge cache
- `edge_rbf` already includes envelope; r_tilde also uses envelope; no double envelope issue
- **Identity start** is guaranteed for any `env_film_scale_delta` with zero-initialized logits

### 10. Runtime acceleration flags and serialization

- `use_triton` is serialized for the descriptor and its internal modules that have Triton paths.
- Inference uses the saved flag and automatically falls back to PyTorch when Triton is unavailable or CUDA is not present.

---

## Tensor Layouts and Invariants

### Node features `x`

The core tensor is:

- `x`: `torch.Tensor` with shape `(N, D, C)`
  - `N = nf * nloc`
  - `C = channels`
  - `D = ebed_dim = (lmax + 1)^2 = sum_{l=0..lmax} (2l + 1)` is the SO(3) embedding dimension

Packing convention:

- Features are packed by increasing `l`.
- Within each `l` block, `m` is ordered from `-l .. +l`.
- Slice ranges are contiguous:
  - `l=0`: indices `[0:1]`
  - `l=1`: indices `[1:4]`
  - `l=2`: indices `[4:9]`
  - `l=3`: indices `[9:16]`
  - in general: `l` block is `[l^2 : (l+1)^2]`

### Edge cache tensors

Edge cache holds **valid edges** (non-padding, non-excluded):

- padding (`nlist == -1`) is removed
- excluded type pairs are removed
- edges with `r >= rcut` are **NOT** removed; their `edge_env=0` (from C² envelope) naturally zeros their messages

This design avoids the dynamic-output-size `nonzero` kernel for distance filtering and enables smoother degree/normalization (no discontinuous edge count jumps at rcut boundary).

Let `E` be the number of valid edges:

- `src`, `dst`: `(E,)` flattened node indices in `[0, N)`
- `edge_type_feat`: `(E, C)` per-edge type embeddings (src+dst)
- `edge_vec`: `(E, 3)` in Å
- `edge_rbf`: `(E, n_radial)` Bessel radial basis via sinc × C² envelope (trainable frequencies)
- `edge_sw`: `(E, 1)` C² cutoff envelope weights flattened to valid edges
- `D_full`: `(E, D, D)` block-diagonal Wigner-D matrix
- `Dt_full`: transpose of `D_full`
- `inv_sqrt_deg`: `(N, 1, 1)` inverse sqrt degree for graph-style normalization

---

## Core Operations

### Geometric Initial Embedding (GIE)

Purpose: Seed `l>0` features at layer 0 to reduce the number of blocks required.

Definition:

- `x(l=0)` comes **only** from type embedding.
- For `l>0`, compute per-`l` zonal seeds via the m=0 column of the Wigner-D transpose
  (local->global) and pre-computed radial features `radial_feat[:, 1:, :]` (sliced for l>=1).

Implementation detail:

- Extract the m=0 column from `Dt_full[:, s:e, l*(l+1)]`

The global index for m=0 in l-block is `l^2 + l = l*(l+1)`.

### SO(2) Convolution (linearized)

For each edge `(src -> dst)`:

1. **Rotate to local frame (reduced)**: The edge-local path uses the **m-major reduced layout**
   controlled by `mmax`. Rotation computes only the required coefficients:
   - project once per `(lmax, mmax)` via cached `project_D_to_m(D_full, coeff_index_m)`
     (shared by all blocks; TorchScript-friendly)
2. **Type feature fusion (once, outside blocks)**:
   - `edge_type_feat = type_ebed[src] + type_ebed[dst]` with shape `(E, C)`
   - `radial_feat = radial_feat + edge_type_feat.unsqueeze(1)` with shape `(E, lmax+1, C)`
   - Per-block truncated `radial_feat[:, : l_i+1, :]` is prebuilt according to `l_schedule`
3. **Modulate local features**: multiply by radial/type features
4. **Multi-layer SO(2) mixing**: for each layer in `so2_linears`:
   - Apply `SO2Linear` (group by `|m|`):
     - `m=0`: standard linear with additive bias (modulated by radial weights and cutoff to preserve strict smoothness on first layer); bias uses in-place add on preallocated output
     - `|m|>0`: 2x2 complex mixing on `(-m, +m)` pairs treated as `(Re, Im)`
   - If `so2_norm=True`, apply `ReducedSeparableRMSNorm` (m-major truncated layout) between layers (not after the last)
   - Apply `GatedActivation(mmax=...)` between layers (not after the last):
     - l=0: SiLU activation
     - l>0: sigmoid(l=0) gate; implementation uses preallocated output instead of cat
5. **Final SO(3) channel mixing**: apply `SO3LinearV2` to mix channels across all degrees (zero-initialized for residual stability)
6. **Rotate back (reduced)**:
   - reuse cached `project_Dt_from_m(Dt_full, coeff_index_m)` (shared across blocks and Script/eager)
7. **Aggregate with optional head gates**:
   - `n_atten_head == 0`: multiply by `edge_env`, scatter-sum by `dst`, then multiply by `inv_sqrt_deg`.
   - `n_atten_head > 0`:
     - **Edge gate**: `g = 2 * sigmoid(clamp(dst + radial + λ * msg, [-6, 6]))` where `λ = λ_max * sigmoid(lambda_raw)` with `λ_max = 0.2`.
     - Weight: `w = edge_env * g` (all gate math and scatter aggregation in promoted dtype, fp32)
     - Split value into `H = n_atten_head` heads, scale by `w`, `index_add` by `dst`
     - Apply `inv_sqrt_deg`
     - **Head gate**: `alpha = 2 * sigmoid(clamp(proj_head(RMSNorm(x_l0)), [-6, 6]))` with bounded per-head scale `gamma_head = 2 * sigmoid(gamma_head_raw)` (init ones, bounded in (0,2))
     - Both gates use `2*sigmoid` (no softmax) and clamp logits before activation

### Full Equivariant FFN

The `EquivariantFFN` class implements:

```python
# Input projection (SO3LinearV2)
h = so3_linear_1(x)  # (N, D, C) -> (N, D, hidden_channels)

# GatedActivation with per-l independent gates
h0 = SiLU(h[:, 0:1, :])  # l=0: scalar activation
gating_scalars = sigmoid(gate_linear(h[:, 0, :]))  # (N, lmax * C)
gating_scalars = gating_scalars.view(N, lmax, C)
gates = index_select(gating_scalars, expand_index)  # (N, D-1, C)
ht = h[:, 1:, :] * gates  # gate l>0 features with per-l gates
h = torch.cat([h0, ht], dim=1)

# Output projection (SO3LinearV2)
out = so3_linear_2(h)  # (N, D, hidden_channels) -> (N, D, C)
```

Key properties:

- Bias only on l=0 (scalar) components
- Each `l` has an independent gate, expanded to all `m` within that `l`
- `expand_index` maps per-l gates to all `2l+1` m-components
- Residual connection: `x = x + ffn(x)`
- Output projection is zero-initialized so the residual path starts near-identity

---

## Interaction Block Structure

The `SeZMInteractionBlock` implements a clean two-path residual structure:

```text
SeZMInteractionBlock:
  x_res = x

  # === Path 1: SO(2) Convolution ===
  x = pre_so2_norm(x)
  if edge_cache.src.numel() > 0:
      x = so2_conv(x, edge_cache)
  x = x + x_res  # Residual
  x_res = x

  # === Path 2: Equivariant FFN ===
  x = pre_ffn_norm(x)
  x = ffn(x)
  x = x + x_res  # Residual

  return x
```

Components:

- `pre_so2_norm`: `SeparableRMSNorm` applied before SO(2) convolution
- `so2_conv`: `SO2Convolution` with multi-layer SO(2) mixing and final SO(3) channel mixing
- `pre_ffn_norm`: `SeparableRMSNorm` applied before FFN
- `ffn`: `EquivariantFFN` with SO(3) linear projections and gated activation

---

## Pyramid `l_schedule`

SeZM-Net supports:

1. constant `lmax` (default): `l_schedule = [lmax] * n_blocks`
2. explicit pyramid: `l_schedule = [2, 2, 1, 0]` (example)

Rules:

- `l_schedule` must be **non-increasing**
- final entry does NOT need to be 0 (output always extracts only l=0 features)
- when schedule decreases, higher-`l` channels are **physically discarded**
- later blocks operate on smaller `ebed_dim`, reducing compute

---

## Pyramid `m_schedule`

SeZM-Net supports:

1. constant `mmax` (default None): if `mmax is None`, `m_schedule[i] = l_schedule[i]`, otherwise `m_schedule[i] = min(mmax, l_schedule[i])`
2. explicit pyramid: `m_schedule = [2, 2, 1, 0]` (example)

Rules:

- must satisfy `m_schedule[i] <= l_schedule[i]` for every block
- a non-increasing schedule is recommended but not required
- within a block, the edge-local SO(2) operator only carries coefficients with `|m| <= mmax`
  - coefficients are stored in an **m-major reduced layout** with dimension:
    `D_m_trunc = sum_{l=0..lmax} (2*min(mmax, l)+1)`
  - rotate-to-local computes only these coefficients via a **row-subset** of Wigner-D
  - rotate-back treats omitted coefficients as zero via a **column-subset** of Wigner-D
  - `SO2Linear` operates directly on the reduced layout (no gather from full D)

Note: unlike `l_schedule`, `m_schedule` does NOT change the **global node tensor** packed layout
(`D = (lmax+1)^2`). It only changes the **edge-local** coefficient set used inside the SO(2)
operator and its rotate-to/rotate-back.

---

## Public API and Hyperparameters

Constructor: `DescrptSeZMNet(...)`

Key arguments:

- `rcut: float` — Cutoff radius in Å
- `sel: list[int] | int` — Maximum neighbors (int: total count, list[int]: per-type counts)
- `lmax: int` — Maximum degree (only if `l_schedule` is None)
- `n_blocks: int` — Number of blocks (only if `l_schedule` is None, default: 2)
- `l_schedule: list[int] | None` — Pyramid schedule of lmax per block, e.g. [3, 3, 2]. Must be non-increasing. If set, lmax and n_blocks will be ignored
- `mmax: int | None` — Maximum SO(2) order (|m|), only used when `m_schedule` is None. If None, defaults to the per-block lmax
- `m_schedule: list[int] | None` — Schedule of mmax per block. Must satisfy `m_schedule[i] <= l_schedule[i]`. If set, `mmax` will be ignored
- `channels: int` — Channels per (l,m) coefficient, i.e. feature dimension per degree (default: 64)
- `n_radial: int` — Number of radial basis functions (default: 10)
- `radial_mlp: list[int]` — Hidden layer sizes for radial networks. An output layer of size (l_schedule[0]+1)\*channels is automatically appended (default: [64])
- `so2_norm: bool` — If True, apply intermediate ReducedSeparableRMSNorm between SO(2) mixing layers. When False (default), no normalization is applied between layers
- `so2_layers: int` — Number of SO2Linear layers per convolution (default: 2)
- `ffn_neurons: int` — Hidden size for equivariant FFN (default: 128)
- `n_atten_head: int` — Number of gated attention heads when aggregating messages in SO(2) convolution. 0 applies a plain envelope-weighted scatter-sum (default: 0). When >0, channels must be divisible by `n_atten_head`. Edge gate uses `2 * sigmoid(proj_b(x_l0)[dst] + proj_s(radial_l0))` without normalization; head gate uses `2 * sigmoid(proj_head(RMSNorm(x_l0)))` with per-head scale `gamma_head` (no softmax).
- `exclude_types: list[tuple[int, int]]` — Excluded type pairs
- `precision: str` — `float64` / `float32`
- `use_amp: bool` — If True, use automatic mixed precision (AMP) with bfloat16 on CUDA. This does not provide accelerations under fp32 precision but will decrease the memory usage, while preserving model accuracy (default: False)
- `use_triton: bool` — If True and Triton is available, use fused Triton kernels for outer-product scatter-sum in EnvironmentInitialEmbedding, including custom backward/double-backward. This reduces memory usage by avoiding large intermediate tensors. Only effective on CUDA devices. Falls back to PyTorch if Triton is unavailable (default: False)
- `use_env_seed: bool` — If True, apply environment matrix initial embedding as FiLM on l=0 features using 4D `[s, s*r_hat]` representation (default: False)
- `env_seed_embed_dim: int` — Output dimension of the G network in environment initial embedding. Other dimensions are derived: `axis_dim = min(8, max(4, env_seed_embed_dim//2))`, `type_dim = min(16, max(8, env_seed_embed_dim//2))`, `hidden_dim = min(64, max(32, 2*env_seed_embed_dim))` (default: 64)
- `env_seed_norm: str` — Normalization mode for env_agg aggregation: "deg" (1/degree) or "sqrt_deg" (1/sqrt(degree)) (default: "sqrt_deg")
- `env_film_scale_delta: float` — Symmetric FiLM scale delta around 1 for env_seed. The scale is `1 + env_film_scale_delta * (2*sigmoid(scale_logits) - 1)` (default: 0.5)

Note: Neighbor normalization (graph-style degree normalization) is always enabled.

### Interface Compatibility Notes

SeZM-Net uses `_ENV_DIM = 1` (se_r style) for `EnvMatStatSe` compatibility. This means:

- `ndescrpt = nnei * 1` (only radial statistics are collected)
- `mean` and `stddev` statistics are maintained but not used in the forward pass (SeZM-Net uses radial basis functions directly instead of traditional env_mat)

Output:

- returns only `l=0` features as descriptor: `(nf, nloc, channels)`

---

## Serialization

`serialize()` captures:

- hyperparameters including `l_schedule`, `m_schedule`, `use_env_seed`, and `compute_mode`
- type embedding parameters
- **env_seed_embedding** (if `use_env_seed=True`): independent type embedding (`env_type_embed`), two-layer RBF projection (`rbf_proj_layer1`, `rbf_proj_layer2`), G network layers, output projection (2\*C), zero-init for FiLM deltas
- **radial basis with trainable frequencies**
- **radial embedding** (RadialMLP: edge_rbf -> (lmax+1)\*C, architecture: Linear → LayerNorm → Activation, first layer bias zero-initialized)
- geometric initial embedding (GIE, if lmax > 0)
- block sub-networks:
  - `EquivariantFFN` (SO3LinearV2 projections + GatedActivation)
  - `SO2Convolution` (SO2Linear + SO3LinearV2, receives pre-computed radial_feat)
  - `SeparableRMSNorm` (pre-norms)
- `so3_linear_output`: final SO3LinearV2 with `lmax=0` for l=0 channel mixing, **uses promoted dtype (float32+) for performance**
- `davg` / `dstd` statistics buffers

`deserialize()` reconstructs the model and restores all parameters including trainable frequencies.

Version: `@version: 1`

---

## Physics & Numerics

### C² cutoff envelope

Envelope is DimeNet-style C²-continuous polynomial to enforce smooth PES at `rcut`:

For `x = r / rcut`:

- `E(x) = 1 + x^5 * (-21 + 35*x - 15*x^2)` for `x in [0, 1)`
- `E(x) = 0` for `x >= 1`

The coefficients satisfy `E(0)=1, E(1)=0, E'(1)=0, E''(1)=0`, ensuring C² continuity.

The C² envelope is applied in two places:

1. In `RadialBasis.forward()`: multiplied into the radial basis functions
2. As `edge_sw`: applied to all edge messages

This double-guarantee ensures:

- message is 0 at `rcut`
- d(message)/dr is 0 at `rcut`
- d²(message)/dr² is 0 at `rcut`

### Conservative forces

- Edge rotations are computed from `edge_vec` without detach.
- Wigner-D blocks are computed from those rotations and remain differentiable.
- Vector normalizations clamp squared norms before `sqrt` (e.g. `sqrt(clamp(||x||^2, eps^2))`) to avoid NaN gradients at zero vectors, even in masked branches.

### Wigner-D blocks (real SH basis)

SeZM-Net uses real-basis Wigner-D blocks to rotate per-degree features between the global frame
and the edge-aligned local frame. The block-diagonal matrices are computed by
`WignerDCalculator` in `deepmd/pt/model/descriptor/se_zm_helper.py`.

#### Conventions

- `rot_mat` is a global->local transform for 3D vectors:
  - `v_local = rot_mat @ v_global`
  - It is built by either `init_edge_rot_mat(edge_vec)` (Gram-Schmidt with a reference-axis switch) or `init_edge_rot_mat_frisvad(edge_vec)` (Frisvad ONB with a strict cross-product fallback near `-Z`) so that `rot_mat @ (edge_vec / ||edge_vec||) = (0, 0, 1)`.
- For each degree `l`, real SH channels are ordered by `m=-l..+l` (index `i = m + l`).
- `D_full` is block-diagonal with block `l` occupying indices `[l^2 : (l+1)^2)`.
- `Dt_full = D_full^T` is the inverse rotation (local->global).

#### Z-rotation Triton optimization

When `use_triton=True` and Triton is available on CUDA, `WignerDCalculator._build_z_rotation()` uses a fused Triton kernel to construct the block-diagonal Z rotation matrices. This replaces the slow Python/advanced-indexing filling (`Z[:, idx, idx] = ...`) with a single kernel launch that writes all sparse entries in one pass.

- **Forward**: Triton kernel stores `K = n_m0 + 4*n_blk` entries per edge (m=0 diagonals + 2x2 rotation blocks for m>0) into the flattened `Z_flat` tensor, then reshapes to `(E, D, D)`.
- **Backward**: Uses pure PyTorch operations (gather + analytical gradients) to preserve double-backward correctness for force training.
- **JIT compatibility**: The Triton path is skipped during `torch.jit.script()` via `torch.jit.is_scripting()` check.

The matmul chain `D_full = Za @ Jt @ Zb @ J @ Zc` remains unchanged; only the Z matrix construction is accelerated.

#### Usage in message passing

- Rotate to local (reduced): `x_local = bmm(D_full[:, coeff_index_m, :], x_global)`
- Rotate back (reduced): `x_global = bmm(Dt_full[:, :, coeff_index_m], x_local)`
- `project_D_to_m` / `project_Dt_from_m` cache the projected blocks keyed by `(lmax, mmax)`.

### Padded neighbor safety

- Padding edges (`nlist == -1`) are removed before any normalization or angle computation.
- Padding indices are replaced with 0 (any valid index) during gather; their values are masked out by `keep = valid_nlist & pair_keep_mask`, avoiding the extra `cat` operation for sentinel coordinates.
- `PairExcludeMask` returns a **keep mask** (1=keep, 0=excluded). It does not remove padding by itself, so always combine it with `nlist >= 0`.
- No zero-length vector is normalized for padding edges.

### Precision and device handling

- All submodules use `dtype: torch.dtype` (not `precision: str`) for constructor parameter.
- Device is obtained from global `env.DEVICE` at runtime; submodules store `self.device = env.DEVICE` only as a convenience reference, not for serialization.
- Each submodule stores `self.precision = RESERVED_PRECISION_DICT[dtype]` for serialization compatibility.
- The `env_protection` parameter (stored as `self.eps`) is used for numerical stability in division and normalization. If 0.0 is passed, it defaults to `1e-7`.

---

## Caching Strategy (Critical)

**EdgeFeatureCache is built exactly once per `forward()`**.

Why caching matters:

- The expensive part is edge geometry + Wigner rotations.
- Message passing blocks reuse the same per-edge rotations and radial features.
- This avoids an O(#blocks) multiplier on per-edge trig/matrix ops.

What is cached / reused:

- All per-edge tensors needed by all blocks (geometry, radial basis, envelope, Wigner-D blocks).
- `radial_feat`: computed once in `compute_dtype`, GIE consumes the pure radial part `radial_feat[:, 1:, :]` (no type fusion), then type embeddings are fused (`edge_type_feat = type_ebed[src] + type_ebed[dst]`) and **per-block truncated slices** are prebuilt according to `l_schedule`.
- Parallel rotation projection caches: `project_D_to_m` / `project_Dt_from_m` project block-diagonal Wigner-D to the m-major truncated layout keyed by `(lmax, mmax)`, shared by all blocks and available in both eager and TorchScript.
- Dtypes: `compute_dtype = get_promoted_dtype(dtype)` is set once in `__init__` and reused for geometry, radial basis/MLP, Wigner calculators, and the final l=0 mixer; runtime casts happen once on `extended_coord`, `radial_feat`, and the final scalar output.

---

## DeePMD Interface Compatibility

SeZM-Net follows the **new-style descriptor interface** (same as `dpa3`), using `extended_coord` / `extended_atype` parameter names (instead of `coord_ext` / `atype_ext` used by older descriptors like `se_a` and `se_r`).

- Implements the required `BaseDescriptor` interface (forward, stats accessors, (de)serialization, neighbor info, exclusion updates).
- `_ENV_DIM = 1` for `EnvMatStatSe` compatibility; statistics are stored but not used in forward.
- Not implemented: `share_params()`, `change_type_map()`.
