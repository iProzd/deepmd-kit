# SeZM: Smooth equivariant Zone-bridging Model

SeZM is a small-`l` equivariant message passing descriptor designed for molecular dynamics (MD) workloads where **inference speed** and **physical correctness** (conservative forces + smooth PES) dominate.
SeZM is the model branch built on top of the SeZM descriptor.

This document is the **final spec** for the SeZM descriptor (`SeZM`, alias: `se_zm`) implemented in:

- `deepmd/pt/model/descriptor/sezm.py`
- `deepmd/pt/model/descriptor/sezm_nn/`

### Module Layout

- `deepmd/pt/model/descriptor/sezm.py`
  - top-level `DescrptSeZM`
  - descriptor orchestration, config parsing, edge-cache construction, block scheduling
  - imports all submodules through `deepmd/pt/model/descriptor/sezm_nn/__init__.py`
- `deepmd/pt/model/descriptor/sezm_nn/__init__.py`
  - public re-export layer for descriptor code, model code, and tests
- `deepmd/pt/model/descriptor/sezm_nn/utils.py`
  - NVTX helpers, dtype promotion, serialization helpers, small numerical utilities
- `deepmd/pt/model/descriptor/sezm_nn/edge.py`
  - `EdgeFeatureCache`, edge type feature construction, cache dtype conversion
- `deepmd/pt/model/descriptor/sezm_nn/indexing.py`
  - packed `(l, m)` indexing, reduced-layout indexing, rotation projection helpers
- `deepmd/pt/model/descriptor/sezm_nn/radial.py`
  - `C3CutoffEnvelope`, `InnerClamp`, `RadialBasis`, `RadialMLP`
- `deepmd/pt/model/descriptor/sezm_nn/embedding.py`
  - `SeZMTypeEmbedding`, `GeometricInitialEmbedding`, `EnvironmentInitialEmbedding`
- `deepmd/pt/model/descriptor/sezm_nn/norm.py`
  - `EquivariantRMSNorm`, `ReducedEquivariantRMSNorm`, `ScalarRMSNorm`
- `deepmd/pt/model/descriptor/sezm_nn/attention.py`
  - destination-wise envelope-gated softmax for SO(2) attention
- `deepmd/pt/model/descriptor/sezm_nn/attn_res.py`
  - `DepthAttnRes`
- `deepmd/pt/model/descriptor/sezm_nn/so3.py`
  - `FocusLinear`, `GatedActivation`, `SO3Linear`
- `deepmd/pt/model/descriptor/sezm_nn/so2.py`
  - `SO2Linear`, `SO2Convolution`
- `deepmd/pt/model/descriptor/sezm_nn/ffn.py`
  - `EquivariantFFN`
- `deepmd/pt/model/descriptor/sezm_nn/block.py`
  - `SeZMInteractionBlock`
- `deepmd/pt/model/descriptor/sezm_nn/wignerd.py`
  - quaternion edge-frame construction and `WignerDCalculator`

______________________________________________________________________

## Goals (Non-Negotiable)

1. **Conservative forces**

   - SeZM outputs features meant for an energy model; forces must come from `autograd` of energy w.r.t. coordinates.
   - Geometry / rotations are fully differentiable; **no `.detach()`** in edge rotations.

1. **Smooth cutoff**

   - Every edge message is multiplied by a **C² polynomial envelope** that goes to **exactly 0 at `rcut`**, so message and its derivatives vanish at `rcut`.

1. **Strict padded neighbor masking**

   - DeePMD neighbor lists are padded (typically with `-1` indices).
   - Padding and excluded type-pairs must contribute **exactly zero** and must not introduce NaNs.

1. **Speed mandate: single geometry/Wigner pass**

   - Edge geometry and Wigner-D rotation blocks are computed **once per `forward()`** and reused by all blocks.
   - No interaction block is allowed to recompute:
     - `edge_vec`
     - envelope / radial basis
     - edge-aligned quaternions
     - Wigner-D blocks

## Non-Goals (Explicitly Out of Scope)

- Top-K neighbor selection: **not allowed**. Use `rcut` and masks only.
- External equivariant libraries (e.g. e3nn): **not allowed**.

______________________________________________________________________

## Model Integration (PyTorch)

- Set `model.type = "SeZM"` (aliases: `"se_zm"`, `"sezm"`) to select the SeZM model scaffold. Aliases are resolved during configuration validation.
- `loss.type` still follows the fitting target (e.g., `"ener"`).
- `descriptor.type` follows user input (SeZM is recommended), and `fitting_net.type` is ignored; SeZM always uses `sezm_ener`.
- Internally it is built as `make_model(SeZMAtomicModel)`.

### Optional compile path (fixed-shape sparse edges)

SeZM supports an optional **fixed-shape edge** path that enables `torch.compile` while preserving
the standard DeePMD neighbor list behavior:

- Enable with `model.use_compile = true`.
- Set `model.n_node` to the fixed node count used for padding (must satisfy `n_node >= nf * nloc`).
- Optional: set `model.n_edge` to cap the fixed edge count. `n_edge=0` means
  `n_node * nsel` (default). When `n_edge>0`, valid edges are **globally sorted
  by distance** and only the shortest `n_edge` are kept; extra edges are dropped
  (padding is added if fewer are available).
- The model still builds the DeePMD neighbor list, then **packs it into a fixed-shape edge list**
  `(src, dst, edge_vec, edge_mask)` with size `n_edge` (or `n_node * nsel` when `n_edge=0`).
- The descriptor accepts the fixed edge list directly and runs a **pure tensor graph**.
- When `use_compile=false` (default), the normal DeePMD neighbor list path is used.
- `use_compile` / `n_node` are model-level flags; the descriptor config remains unchanged.

This path is designed for second-order derivatives during training; all geometry (edge vectors,
Wigner-D, radial basis) remains differentiable, and padded edges are fully masked.

______________________________________________________________________

## ZBL Bridging (Optional Short-Range Repulsion)

SeZM supports an optional analytical short-range repulsion potential that supplements the
ML-predicted energy via pure additive energy decomposition:

```
E_total = E_ZBL(r) + E_model(r̃)
```

where `r` is the true interatomic distance and `r̃` is the clamped distance seen by the descriptor.

### Design Principles

1. **Pure additive**: ZBL energy is added to the atomic energy *before* autograd, so forces and
   virials naturally include ZBL contributions with no extra code.
1. **Descriptor-level inner clamping**: When bridging is active, the descriptor sees clamped
   distances — both scalar and vectorial — so its output is completely frozen below `r_inner`.
   This prevents the ML model from learning (or unlearning) the repulsive wall.
1. **Optional**: Controlled by `model.bridging_method` (default `"none"`). Setting it to `"ZBL"`
   enables both the ZBL potential and inner clamping simultaneously.

### Configuration Parameters (model-level)

| Parameter          | Type  | Default  | Description                                              |
| ------------------ | ----- | -------- | -------------------------------------------------------- |
| `bridging_method`  | str   | `"none"` | Bridging formula. `"none"` to disable, `"ZBL"` to enable |
| `bridging_r_inner` | float | 1.0      | Inner clamping radius in Å. Descriptor frozen below this |
| `bridging_r_outer` | float | 1.5      | Outer clamping radius in Å. Transition zone upper bound  |

These are model-level parameters. When `bridging_method != "none"`, the model factory injects
`inner_clamp_r_inner` and `inner_clamp_r_outer` into the descriptor parameters automatically;
they are not user-facing descriptor config keys.

### Inner Clamping

The `InnerClamp` module (in `deepmd/pt/model/descriptor/sezm_nn/radial.py`) implements a C3-continuous septic Hermite
polynomial that maps true distances to effective distances:

```
r̃(r) = r_inner                                           if r <= r_inner   (frozen zone)
      = r_inner + (r_outer - r_inner) * h(t)              if r_inner < r < r_outer  (transition)
      = r                                                  if r >= r_outer  (identity zone)

where t = (r - r_inner) / (r_outer - r_inner)
      h(t) = 20t^4 - 45t^5 + 36t^6 - 10t^7
```

Boundary conditions: `h(0)=0, h(1)=1, h'(0)=0, h'(1)=1, h''(0)=0, h''(1)=0`.

**Vector-level clamping**: Both the scalar distance and the displacement vector are clamped.
The vector is rescaled to preserve direction while matching the clamped distance:

```python
clamped = inner_clamp(length)
scale = clamped / length.clamp(min=eps)
diff = diff * scale  # direction preserved, ||diff|| = clamped
length = clamped
```

The polynomial satisfies `h(0)=0`, `h(1)=1`, `h'(0)=0`, `h'(1)=1`,
`h''(0)=0`, `h''(1)=0`, `h'''(0)=0`, and `h'''(1)=0`, so the frozen zone,
transition zone, and identity zone match with C3 continuity.

This ensures the descriptor receives *no information* about the true distance when `r < r_inner`.
All downstream modules — radial basis, envelope, edge quaternion construction,
`EnvironmentInitialEmbedding` — see clamped values uniformly. No raw distance is preserved anywhere.

### InterPotential (ZBL)

The `InterPotential` module (in `sezm_model.py`) computes the analytical Ziegler-Biersack-Littmark
screened nuclear repulsion potential:

```
V_ZBL(r) = (ke * Zi * Zj / r) * phi(r / a)
a = 0.88534 * a_bohr / (Zi^0.23 + Zj^0.23)
phi(x) = 0.18175*exp(-3.1998*x) + 0.50986*exp(-0.94229*x) + 0.28022*exp(-0.4029*x) + 0.02817*exp(-0.20162*x)
```

Each pair `(i, j)` contributes `V_ZBL / 2` to atom `i` (symmetric neighbor list, avoid double-counting).

Energy injection points:

- **Standard path** (`forward_common_lower`): injected into `atomic_ret["energy"]` before
  `fit_output_to_model_output`.
- **Compile path** (`compile_compute_func`): injected into `atom_energy` before autograd
  computes forces/virials.

______________________________________________________________________

## High-Level Architecture

Text diagram (single forward pass):

```
Standard DeePMD nlist path:
  Inputs: extended_coord, extended_atype, nlist, mapping
    └─ EdgeFeatureCache (built once via build_edge_cache)
       ├─ edges: (src, dst) global indices, edge_vec
       ├─ edge_type_feat: per-edge type embedding (src+dst)
       ├─ edge_rbf: Bessel radial basis via sinc × C² envelope
       ├─ edge_env: C³ cutoff envelope weights (flattened to valid edges)
       ├─ D_full, Dt_full: block-diagonal Wigner-D matrices
       └─ inv_sqrt_deg: inverse sqrt smooth degree (`sum(edge_env²)`) for normalization

  Radial embedding (computed once):
    └─ radial_feat: (E, lmax+1, C) via RadialMLP(edge_rbf)
       └─ fused once with edge_type_feat after GIE

  Node init:
    ├─ l=0: Type embedding + (optional) EnvironmentInitialEmbedding
    └─ l>0: Zonal (m=0) initial embedding via cached Wigner-D + radial_feat[:, 1:, :]

  Interaction blocks (pyramid schedule):
    if `full_attn_res != "none"`:
      unit_history = [x0]
    if `block_attn_res != "none"`:
      block_history = [x0]
    for block i:
      ├─ slice D to ebed_dim(l_schedule[i]) (discard higher-l if needed)
      ├─ truncate every source in the active history to the current `D_i`
      ├─ optional full AttnRes for SO(2) input from `unit_history`
      ├─ optional block AttnRes for SO(2) input from `block_history`
      ├─ main tensor layout is fixed as (N, D, F, Cf), contiguous
      ├─ EquivariantRMSNorm (pre-SO2, per-focus on (N, D, F, Cf))
      ├─ Multi-Focus SO(2) Convolution (enabled for ALL lmax, including lmax=0)
      │  ├─ `pre_focus_mix`: full-channel mixing on (N, D, C)
      │  ├─ rotate/bmm in full width (E, D, C), then SO2 stack on strided (E, F, Dm, Cf)
      │  ├─ optional SO(2) internal AttnRes over local layer history when `so2_attn_res != "none"`
      │  └─ `post_focus_mix` on (N, 1, D, C)
      ├─ FFN subblock sequence (ffn_blocks iterations, global C via view, no permute)
      │  ├─ optional full AttnRes before each FFN unit from `unit_history`
      │  ├─ optional block AttnRes before each FFN unit from `block_history + [partial_block]`
      │  └─ update either `unit_history` or `partial_block`
      ├─ if full mode: append SO(2) output and every FFN unit output to `unit_history`
      └─ if block mode: wrapper returns `block_summary` directly and descriptor appends it once to `block_history`

  Output (forward, promoted dtype):
    └─ optional final full AttnRes over all completed unit representations when `full_attn_res != "none"`
    └─ optional final block AttnRes over all completed block summaries when `block_attn_res != "none"`
    └─ Extract x(l=0) from global block output
    └─ reshape to (N, 1, 1, C)
    └─ Convert to promoted dtype (float32+)
    └─ Scalar FFN (lmax=0) for channel mixing
       └─ Residual: x0 + FFN(x0)
       └─ (nf, nloc, channels)
```

______________________________________________________________________

## SeZM Fitting (GLU)

- The fitting net uses the same configuration keys as the standard energy fitting
  (`neuron`, `activation_function`, `precision`, `seed`, ...).
- `neuron = []` is valid and means a direct linear projection from descriptor
  dimension to scalar energy.
- When `neuron` is non-empty, each hidden layer is a GLU block:
  `Linear(in, 2*hidden) -> split -> value * act(gate)`.
  This makes the internal hidden width double the user-specified value
  (e.g., `hidden=256` becomes `512` before split).

______________________________________________________________________

## Key Design Decisions

### 1. Trainable radial basis frequencies

- Frequencies are trainable parameters initialized as `n * pi / rcut` for `n=1..n_radial`.
- Basis uses a sinc form for stable `r -> 0` gradients: `phi_n(r) = w_n * sinc(w_n * r / pi)`.
- Frequencies are serialized/deserialized.

### 2. EquivariantRMSNorm with Degree Balancing

- Applies one shared RMS over **all retained degrees**, not separate denominators for `l=0` and `l>0`.
- The scalar slice `l=0` can be mean-centered across channels before RMS evaluation.
- **Degree Balancing** weights every coefficient from degree `l` by `1/(2l+1)`, then averages uniformly across degrees with factor `1/(lmax+1)`, so each degree contributes equally regardless of multiplicity.
- Per-l affine scales are expanded to all coefficients via a precomputed degree index.
- Bias is applied only to the scalar slice after normalization.
- Memory optimization: the weighted RMS is computed with fused einsum, and the reduced SO(2) layout uses retained-count weights `2 * min(l, mmax) + 1`.

### 3. SO3Linear

- `SO3Linear` keeps per-focus independent mixing on `(N, D, F, C)` and stores
  weight as `(lmax+1, Cin, F*Cout)` (focus folded on output side).
- Implemented as `index_select(expand_index)` + `einsum`; bias is applied only for `l=0`.

### Multi-focus SO2 + global FFN

- `channels` is split as `C = F * Cf`, where `F = n_focus` and `Cf = focus_dim`.
- Backbone tensor stays in a single contiguous layout `(N, D, F, Cf)` across all blocks.
- SO(2) pre/post norms operate directly on `(N, D, F, Cf)` (per-focus semantics unchanged).
- Geometry cache (`edge_vec`, `edge_rbf`, `D_full`, `Dt_full`) is still built once per forward and shared across all focus streams.
- Radial features keep `(E, L, C)` and are consumed by SO(2) without an explicit focus split.
- FFN branch uses `view(N, D, C) <-> view(N, D, F, Cf)` only; no layout permute is required.

### 4. Full equivariant FFN

- `SO3Linear -> GatedActivation -> SO3Linear` with a residual connection.
- Gates are derived from `l=0` scalars (one gate per `l`) and expanded across all `m`.

### 5. Multi-layer SO(2) convolution

- Uses the edge-local m-major reduced layout controlled by `mmax`.
- Stacks `SO2Linear` with optional `GatedActivation(mmax=...)` for `so2_layers`.
- Each SO(2) layer uses **pre-norm + residual + LayerScale**:
  - `residual = x_local`
  - `x_local = inter_norm(x_local)` (pre-norm; Identity when `so2_norm=False`)
  - `x_local = so2_linear(x_local)`
  - bias correction (layer 0 only, when bias exists)
  - `x_local = non_linear(x_local)`
  - `x_local = residual + layer_scale * x_local` (scalar LayerScale when `layer_scale=True`, otherwise bare residual)
- Ends with a `SO3Linear` channel mixer before aggregation.
- `SO2Linear` uses a single block-diagonal matmul over all m groups:
  - m=0 block: unconstrained linear over `(l=0..lmax)` coefficients
  - |m|>0 blocks: constrained complex coupling `[W_u, -W_v; W_v, W_u]`
  - This reduces kernel launches while preserving SO(2) equivariance.
  - All learnable weights use `(in, out)` layout; focus dimension `F` is folded on the output (cols) side.
  - Weights are stored as raw parameters (no activation/bias); only the l=0 bias is separate.
- Optional depth-wise attention residuals:
  - `so2_attn_res="none"` disables SO(2)-internal AttnRes, `"independent"` uses a learned pseudo-query, and `"dependent"` derives the query from the current reduced-layout `l=0` slice before each SO(2) layer.
  - `full_attn_res="none"` and `block_attn_res="none"` keep the original residual-connected block wrapper.
  - `full_attn_res` keeps `unit_history = [x0, so2_0, ffn_0_0, ffn_0_1, ..., so2_1, ffn_1_0, ffn_1_1, ...]`, truncates every source to the current `D_i`, applies one selective aggregation before the SO(2) unit, applies another selective aggregation before each FFN unit, and runs one final aggregation before `output_ffn`.
  - In SeZM, `full_attn_res="independent"` uses learned query vectors. `full_attn_res="dependent"` derives the query from the current block input before SO(2), from the latest unit output before each FFN unit, and from the final block output before the last aggregation.
  - `block_attn_res` keeps `block_history = [x0, b1, b2, ...]`, where each `b_i` is the sum of the SO(2) unit output and all FFN unit outputs inside one `SeZMInteractionBlock`. The SO(2) unit attends only `block_history`; each FFN unit attends `block_history + [partial_block]`, where `partial_block` is the running sum of unit outputs inside the current block.
  - In SeZM, `block_attn_res="independent"` uses learned query vectors. `block_attn_res="dependent"` derives the query from the current block input before SO(2), from the latest unit output before each FFN unit, and from the final block output before the last aggregation.
  - `full_attn_res` and `block_attn_res` are mutually exclusive.
  - All depth-attention query paths are zero-initialized, so the initial behavior is a uniform average over the available sources.

### 6. Deterministic initialization

- All submodules derive seeds via `child_seed(seed, idx)`; repeated structures include loop indices.
- If `seed=None`, initialization follows the global RNG.
- SO2Linear weights use truncated normal init with std `1/sqrt(fan_in + fan_out)`, cut at +/-3\*std.
- For |m|>0 blocks, an extra `1/sqrt(2)` scale is applied to preserve SO(2) coupling energy.
- SO3Linear uses truncated normal init with std `1/sqrt(fan_in + fan_out)`, cut at +/-3\*std.
- `FocusLinear`, `SO3Linear`, and `SO2Linear` all support an optional `init_std` parameter: when given, weights are initialized with `Normal(0, init_std)` instead of the default scheme. Use `init_std=0.0` for zero initialization (e.g., residual output projections).

### 7. Unified weight layout convention

All learnable weight matrices use **(in, out) convention** (rows = fan_in, cols = fan_out),
and focus-aware modules fold the focus dimension `F` on the **output (cols) side**:

- `FocusLinear`
  - stored weight: `(C_in, F * C_out)` — 2D, Muon
  - runtime view: `(C_in, F, C_out)` with `einsum("bfi,ifo->bfo")`
- `SO3Linear`
  - stored weight: `(lmax+1, Cin, F * Cout)` — 3D, per-l slice Muon
  - stored bias: `(F * Cout,)` — 1D, Adam
  - runtime view: `(D, Cin, F, Cout)` via `index_select`, then `einsum("ndfi,difo->ndfo")`
  - focus streams stay independent in compute while focus is folded on the output side for Muon scaling
- `SO2Linear`
  - stored `weight_m0`: `(num_l*Cin, F * num_l*Cout)` — 2D, Muon
  - stored `weight_m[i]`: `(num_l*Cin, F * 2*num_l*Cout)` — 2D, Muon
  - runtime view: assembled to `(D_m*Cin, F, D_m*Cout)` and applied by `einsum("efi,ifo->efo")`

This (in, out) storage layout ensures Muon's rectangular correction `scale = sqrt(max(1, rows/cols))`
stays at 1.0 when `Cin <= F*Cout` (typical case), avoiding step-size inflation.
Each semantically independent weight block gets its own Muon NS update without
artificial cross-l or cross-focus coupling.

### 8. Gate initialization strategy

All gate projections use a consistent initialization:

- **Matrix**: `Normal(mean=0, std=0.01)` with reproducible generator
- **Bias**: zeros

This ensures gate logits start near 0, making `sigmoid(0) ≈ 0.5`.

**Benefits**:

- Maximum gradient flow: `sigmoid'(0.5) = 0.25` is the maximum derivative value
- Unbiased feature scaling: the model learns which features to suppress/amplify from the loss signal
- Output-side head gates start near `~0.5`, avoiding early saturation

### 9. Stability defaults

- Residual branches start near-identity: the output projections of both SO(2) convolution and Equivariant FFN are zero-initialized (weights + bias).
- When `layer_scale=True`, both SO(2) residual branches (per-focus-channel, init 1e-3) and FFN residual branches (per-channel vector, init 1e-3) use learnable LayerScale for training stability.
- Attention aggregation runs in promoted dtype (fp32) and uses destination-wise stable softmax (group max subtraction) with envelope-weighted unnormalized scores.
- Multi-layer SO(2) stacks use pre-norm residual connections. When `so2_norm=True`, a reduced-layout separable RMSNorm is applied as pre-norm before each SO(2) layer (except the last, which uses Identity). This keeps truncated m-major activations balanced but is disabled by default.

### 10. Optional environment matrix initial embedding (EnvironmentInitialEmbedding)

An optional module that provides physical inductive bias for l=0 features using a 4D environment matrix approach. When `use_env_seed=True`:

**Key design: Type embedding decoupling**

- Uses an **independent** `env_type_embed` (`SeZMTypeEmbedding`) instead of projecting from the main type embedding
- This allows `env_seed` to learn type representations independent from the main descriptor backbone
- RBF projection (`rbf_proj`) aligns G-network input dimension to approximately `embed_dim`

**Computation pipeline**:

1. **r_tilde construction**: For each edge, build a 4D vector `[s, s*rx, s*ry, s*rz]` where:

   - `s = edge_env / r` (smooth weight divided by distance)
   - `r_hat = edge_vec / r` (unit direction vector)
   - `r_tilde = [s, s * r_hat]` encodes both radial decay and angular information

1. **G network**: Computes per-edge filter features:

   - RBF projection: Two-layer MLP `rbf_proj_layer1 → rbf_proj_layer2` with dimension `rbf_out_dim = max(32, embed_dim - 2*type_dim)`

- RBF/G MLP layers use `TruncatedNormal(mean=0, std=sqrt(2/(fan_in+fan_out)), ±3σ)` for weights; `output_proj` is still zero-initialized for FiLM logits
  - First layer: `n_radial → rbf_out_dim` with activation (SiLU)
  - Second layer: `rbf_out_dim → rbf_out_dim` linear
- Type embeddings: `type_src, type_dst = env_type_embed(atype[src]), env_type_embed(atype[dst])`
- Input: `concat([rbf_proj, type_src, type_dst])` with dimension `(E, rbf_out_dim + 2*type_dim) ≈ embed_dim`
- Two-layer MLP: `hidden_dim` → `embed_dim` with SiLU activation

3. **env_agg (environment aggregation)**: Vectorized outer product and scatter:

   - `outer = r_tilde[:, :, None] * g[:, None, :]` produces `(E, 4, embed_dim)`
   - Scatter-sum by destination node: `env_agg.index_add_(0, dst, outer_flat)`
   - Normalize by smooth envelope-squared degree

1. **D matrix construction**: Captures local geometry via matrix product:

   - `D = env_agg^T @ env_agg[:, :, :axis_dim]` with shape `(N, embed_dim, axis_dim)`

1. **Output projection to FiLM logits**:

   - Flatten D to `(N, embed_dim * axis_dim)` and project to `(N, 2*channels)`
   - Split to `(scale_logits, shift_logits)` and apply FiLM on `l=0`:
     - `scale_strength = exp(scale_strength_log)`
     - `shift_strength = exp(shift_strength_log)`
     - `scale = 1 + scale_strength * tanh(RMSNorm(scale_logits))`
     - `shift = shift_strength * tanh(RMSNorm(shift_logits))`
     - `x0 = x0 * scale + shift`
   - Output projection is zero-initialized → logits start at 0; strengths initialize to `1e-2` (near-identity, non-dead gradients)

**Key properties**:

- Uses **global frame** direction (not edge-aligned local frame) to preserve angular information
- Normalization uses smooth envelope-squared degree from edge cache
- `edge_rbf` already includes envelope; r_tilde also uses envelope; no double envelope issue
- **Near-identity start** is guaranteed by small strengths with zero-initialized logits

______________________________________________________________________

## Tensor Layouts and Invariants

### Node features `x`

The backbone tensor is:

- `x`: `torch.Tensor` with shape `(N, D, F, Cf)` (contiguous)
  - `N = nf * nloc`
  - `F = n_focus`
  - `Cf = focus_dim`
  - `C = channels = F * Cf`
  - `D = ebed_dim = (lmax + 1)^2 = sum_{l=0..lmax} (2l + 1)` is the SO(3) embedding dimension

View conventions used inside blocks:

- `x.view(N, D, C)` for full-channel rotate/bmm and FFN mixing
- `x.view(N, D, F, Cf)` for per-focus SO(2) pre/post norms

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
- edges with `r >= rcut` are **NOT** removed; their `edge_env=0` (from C³ envelope) naturally zeros their messages

This design avoids the dynamic-output-size `nonzero` kernel for distance filtering and enables smoother degree/normalization (no discontinuous edge count jumps at rcut boundary).

Let `E` be the number of valid edges:

- `src`, `dst`: `(E,)` flattened node indices in `[0, N)`
- `edge_type_feat`: `(E, C)` per-edge type embeddings (src+dst)
- `edge_vec`: `(E, 3)` in Å
- `edge_rbf`: `(E, n_radial)` Bessel radial basis via sinc × C² envelope (trainable frequencies)
- `edge_env`: `(E, 1)` C³ cutoff envelope weights flattened to valid edges
- `D_full`: `(E, D, D)` block-diagonal Wigner-D matrix
- `Dt_full`: transpose of `D_full`
- `inv_sqrt_deg`: `(N, 1, 1)` inverse sqrt smooth degree for graph-style normalization (`deg = sum(edge_env^2)`)

______________________________________________________________________

## Core Operations

### Geometric Initial Embedding (GIE)

Purpose: Seed `l>0` features at layer 0 to reduce the number of blocks required.

Definition:

- `x(l=0)` comes **only** from type embedding.
- For `l>0`, compute per-`l` zonal seeds via the m=0 column of the Wigner-D transpose
  (local->global) and pre-computed radial features `radial_feat[:, 1:, :]` (sliced for l>=1).

Implementation detail:

- Vectorized gather: collect all rows with `l>=1` in packed order, and map each row to its `m=0` column using the identity `l^2 + l = l*(l+1)`.
- Broadcast `radial_feat` to each packed row via its `l-1` mapping, then scatter once into a compact buffer and assign back to `out[:, row_index, :]` to avoid advanced-index writeback.

### SO(2) Convolution (linearized)

For each edge `(src -> dst)`:

1. **Pre-focus mixing (full C width, node-side)**:
   - `x = pre_focus_mix(x.unsqueeze(2)).squeeze(2)` with `n_focus=1`
   - this is channel-only mixing per `(l,m)` and keeps SO(3) equivariance
1. **Rotate to local frame (full C width first)**:
   - project once per `(lmax, mmax)` via cached `project_D_to_m(D_full, coeff_index_m)`
   - `x_src = x.index_select(0, src)` gives `(E, D, C)`
   - `x_local = bmm(D_to_m, x_src)` gives `(E, D_m, C)` (high-efficiency GEMM)
1. **Type feature fusion (once, outside blocks)**:
   - `edge_type_feat = type_ebed[src] + type_ebed[dst]` with shape `(E, C)`
   - `radial_feat = radial_feat + edge_type_feat.unsqueeze(1)` with shape `(E, lmax+1, C)`
   - Per-block truncated `radial_feat[:, : l_i+1, :]` is prebuilt according to `l_schedule`
1. **Modulate local features**:
   - `rad_feat = radial_feat[:, degree_index_m, :]` with shape `(E, D_m, C)`
   - `x_local *= rad_feat`
1. **Convert to SO(2) internal layout**:
   - `x_local.view(E, D_m, F, Cf).transpose(1, 2)` gives strided `(E, F, D_m, Cf)`
   - no explicit contiguous call here
1. **Multi-layer SO(2) mixing (pre-norm + residual + LayerScale)**: for each layer in `so2_linears`:
   - Save residual: `residual = x_local`
   - Pre-norm: apply `inter_norm(x_local)` (ReducedEquivariantRMSNorm when `so2_norm=True`, Identity otherwise; last layer always Identity)
   - Apply `SO2Linear` (group by `|m|`):
     - `m=0`: standard linear with additive bias (modulated by radial weights and cutoff to preserve strict smoothness on first layer); bias uses in-place add on preallocated output
     - `|m|>0`: 2x2 complex mixing on `(-m, +m)` pairs treated as `(Re, Im)`
   - Apply `non_linear` (GatedActivation between layers, Identity for last layer):
     - l=0: SiLU activation
     - l>0: sigmoid(l=0) gate; implementation uses preallocated output instead of cat
   - LayerScale + Residual: `x_local = residual + scale * x_local` (scalar scale, init 1e-3 when `layer_scale=True`; bare residual otherwise)
1. **Cross-focus competition (optional)**:
   - Enabled only when `focus_compete=True and n_focus>1`
   - Use scalar-invariant source captured before SO(2) stack: `focus_gate_src = x_local_pre[:, :, 0, :]`
   - Compute logits with per-focus scalar projection and temperature:
     - `logits = focus_compete_proj(ScalarRMSNorm(focus_gate_src))`
     - `alpha = softmax(logits / tau, dim=focus)`
   - Apply label smoothing to avoid dead focuses:
     - `alpha = (1 - eps) * alpha + eps / n_focus` (internal default `eps=0.02`)
   - Apply invariant weights to full irreps: `x_local *= alpha[:, :, None, None]`
1. **Rotate back preparation**:
   - `x_local.transpose(1, 2).contiguous().view(E, D_m, C)`
1. **Rotate back (reduced)**:
   - reuse cached `project_Dt_from_m(Dt_full, coeff_index_m)` (shared across blocks and Script/eager)
   - `x_message = bmm(Dt_from_m, x_local)` gives `(E, D, C)`
   - apply the inverse-rotation degree rescale for every `l > mmax` to restore
     the full-basis amplitude after truncated local mixing
1. **Aggregate with optional head gates**:

- `n_atten_head == 0`: multiply by `edge_env`, scatter-sum by `dst`, then multiply by `inv_sqrt_deg`.
- `n_atten_head > 0`:
  - **Edge attention logits**:
    - Pre-norm: `qk_input = attn_qk_norm(x_l0_node)` on destination node scalar channel
    - Independent Q/K projections: `q = attn_q_proj(qk_input)`, `k = attn_k_proj(qk_input)`
    - Gather per-edge: `q_edge = q[dst]`, `k_edge = k[src]`, reshape to `(E, F, H, Dh)`
    - `logits = dot(q_edge, k_edge) / sqrt(head_dim) + attn_radial_logit_proj(radial_l0)`
  - **Stable grouped softmax with envelope-gated competition**:
    - destination-wise max subtraction for numerical stability
    - `numerator = edge_env^2 * exp(logits - grouped_max)`
    - `denominator = softplus(z_bias_raw) * exp(-grouped_max) + sum(edge_env^2 * exp(logits - grouped_max))`
    - normalize per destination (`index_add` denominator)
  - Split value into `H = n_atten_head` heads, scale by `alpha`, `index_add` by `dst`
  - Apply output-side head gate (G1 style):
    - `gate = sigmoid(attn_output_gate_proj(attn_output_gate_norm(x_l0_node)))`
    - gate is head-specific and query-dependent
  - Merge heads back to `(N, D, C)` (**no `inv_sqrt_deg` on this path**)

11. **Post-focus mixing**:

- `out = post_focus_mix(out.unsqueeze(2)).squeeze(2)` with `n_focus=1`, mixing full channel width `C`

### Full Equivariant FFN

The `EquivariantFFN` class implements:

```python
# Input projection (SO3Linear)
h = so3_linear_1(x)  # (N, D, C) -> (N, D, hidden_channels)

# GatedActivation with per-l independent gates
h0 = SiLU(h[:, 0:1, :])  # l=0: scalar activation
gating_scalars = sigmoid(gate_linear(h[:, 0, :]))  # (N, lmax * C)
gating_scalars = gating_scalars.view(N, lmax, C)
gates = index_select(gating_scalars, expand_index)  # (N, D-1, C)
ht = h[:, 1:, :] * gates  # gate l>0 features with per-l gates
h = torch.cat([h0, ht], dim=1)

# Output projection (SO3Linear)
out = so3_linear_2(h)  # (N, D, hidden_channels) -> (N, D, C)
```

Key properties:

- Bias only on l=0 (scalar) components
- Each `l` has an independent gate, expanded to all `m` within that `l`
- `expand_index` maps per-l gates to all `2l+1` m-components
- Residual connection: `x = x + ffn(x)`
- Output projection is zero-initialized so the residual path starts near-identity

______________________________________________________________________

## Interaction Block Structure

The `SeZMInteractionBlock` implements a clean two-path residual structure:

```text
SeZMInteractionBlock:  # x shape: (N, D, F, Cf)
  C = F * Cf

  # === Path 1: SO(2) Convolution ===
  x_pre = pre_so2_norm(x)                          # (N, D, F, Cf)
  y = so2_conv(view(x_pre, N, D, C), edge_cache, radial_feat)  # (N, D, C)
  y = post_so2_norm(view(y, N, D, F, Cf))         # (N, D, F, Cf)
  x = x + y

  # === Path 2: FFN subblock sequence (ffn_blocks iterations) ===
  x_ffn = view(x, N, D, C).unsqueeze(1)      # (N, 1, D, C)
  for i in range(ffn_blocks):
      x_pre = pre_ffn_norms[i](view(x_ffn, N, D, 1, C))
      x_pre = view(x_pre, N, 1, D, C)
      y = ffns[i](x_pre)
      y = post_ffn_norms[i](view(y, N, D, 1, C))
      y = view(y, N, 1, D, C)
      if layer_scale:
          y = y * adam_ffn_layer_scales[i]   # per-channel, init 1e-3
      x_ffn = x_ffn + y

  return view(x_ffn.squeeze(1), N, D, F, Cf)
```

Descriptor-level wrapper around each block:

```text
x = block_i(x, edge_cache, radial_feat_i)  # block internally does:
                                            # pre-norm -> SO2(full C rotate) ->
                                            # FFN(view-only reshape)
```

Components:

- `pre_so2_norm`: `EquivariantRMSNorm` applied before SO(2) convolution
- `so2_conv`: `SO2Convolution` with pre-norm residual SO(2) mixing, optional per-focus-channel LayerScale, and final SO(3) channel mixing
- `pre_ffn_norms[i]`: `EquivariantRMSNorm` applied before each FFN subblock
- `ffns[i]`: `EquivariantFFN` with SO(3) linear projections and gated activation
- `adam_ffn_layer_scales[i]`: optional per-channel learnable scale (init 1e-3) for training stability

______________________________________________________________________

## Pyramid `l_schedule`

SeZM supports:

1. constant `lmax` (default): `l_schedule = [lmax] * n_blocks`
1. explicit pyramid: `l_schedule = [2, 2, 1, 0]` (example)

Rules:

- `l_schedule` must be **non-increasing**
- final entry does NOT need to be 0 (output always extracts only l=0 features)
- when schedule decreases, higher-`l` channels are **physically discarded**
- later blocks operate on smaller `ebed_dim`, reducing compute

______________________________________________________________________

## Pyramid `m_schedule`

SeZM supports:

1. constant `mmax` (default None): if `mmax is None`, `m_schedule[i] = l_schedule[i]`, otherwise `m_schedule[i] = min(mmax, l_schedule[i])`
1. explicit pyramid: `m_schedule = [2, 2, 1, 0]` (example)

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

______________________________________________________________________

## Public API and Hyperparameters

Constructor: `DescrptSeZM(...)`

Key arguments:

- `rcut: float` — Cutoff radius in Å
- `sel: list[int] | int` — Maximum neighbors (int: total count, list\[int\]: per-type counts)
- `lmax: int` — Maximum degree (only if `l_schedule` is None)
- `n_blocks: int` — Number of blocks (only if `l_schedule` is None, default: 2)
- `l_schedule: list[int] | None` — Pyramid schedule of lmax per block, e.g. [3, 3, 2]. Must be non-increasing. If set, lmax and n_blocks will be ignored
- `mmax: int | None` — Maximum SO(2) order (|m|), only used when `m_schedule` is None. If None, defaults to the per-block lmax
- `m_schedule: list[int] | None` — Schedule of mmax per block. Must satisfy `m_schedule[i] <= l_schedule[i]`. If set, `mmax` will be ignored
- `channels: int` — Total channels per (l,m) coefficient (default: 64)
- `n_focus: int` — Number of parallel focus streams. Internal width is `focus_dim = channels // n_focus`; channels must be divisible by `n_focus` (default: 1)
- `focus_compete: bool` — If True, enable cross-focus softmax competition in SO(2) convolution. Logits are built from l=0 scalar channels and normalized across focus streams; weights are broadcast to all `(l, m)` components in each focus (default: True)
- `n_radial: int` — Number of radial basis functions (default: 10)
- `radial_mlp: list[int]` — Hidden layer sizes for radial networks. An output layer of size (l_schedule[0]+1)\*channels is automatically appended (default: [64])
- `so2_norm: bool` — If True, apply ReducedEquivariantRMSNorm as pre-norm before each SO(2) mixing layer (except the last, which uses Identity). When False (default), pre-norm is Identity for all layers
- `so2_layers: int` — Number of SO2Linear layers per convolution (default: 4)
- `so2_attn_res: str` — SO(2)-internal depth-wise attention residual mode inside each interaction block. Allowed values: `none`, `independent`, `dependent` (default: `none`)
- `ffn_neurons: int` — Hidden size for equivariant FFN (default: 96)
- `ffn_blocks: int` — Number of FFN subblocks per interaction block (default: 1)
- `n_atten_head: int` — Number of attention heads when aggregating messages in SO(2) convolution. 0 applies a plain envelope-weighted scatter-sum (default: 0). When >0, the per-focus width `channels // n_focus` must be divisible by `n_atten_head`, and envelope-gated grouped softmax attention with output-side head gate is applied. Attention uses `w^2 * exp(logit)` in the numerator and `zeta + sum(w^2 * exp(logit))` in the denominator.
- `sandwich_norm: list[bool]` — Pre/post-norm switches for residual branches: `[so2_pre, so2_post, ffn_pre, ffn_post]` (default: [True, False, True, False])
- `exclude_types: list[tuple[int, int]]` — Excluded type pairs
- `precision: str` — `float64` / `float32`
- `mlp_bias: bool` — Whether to use bias in equivariant layers (SO3Linear l=0 bias, SO2Linear l=0 bias, GatedActivation gate linear bias, DepthAttnRes input-dependent query projection) and EnvironmentInitialEmbedding MLPs (`rbf_proj_layer1/2`, `g_layer1/2`) (default: True)
- `layer_scale: bool` — If True, apply learnable LayerScale on residual branches for training stability: per-focus-channel scales (init 1e-3) on each SO(2) mixing layer, and per-channel vector (init 1e-3) on each FFN subblock (default: False)
- `full_attn_res: str` — Descriptor-level full AttnRes mode over unit history. `independent` uses learned query vectors; `dependent` derives the query from the current SeZM state before the SO(2) unit, before each FFN unit, and before the final aggregation. Allowed values: `none`, `independent`, `dependent` (default: `none`)
- `block_attn_res: str` — Descriptor-level block AttnRes mode over block history. `independent` uses learned query vectors; `dependent` derives the query from the current SeZM state before the SO(2) unit, before each FFN unit, and before the final block aggregation. Allowed values: `none`, `independent`, `dependent` (default: `none`). Cannot be enabled together with `full_attn_res`
- `use_amp: bool` — If True, use automatic mixed precision (AMP) with bfloat16 on CUDA. This does not provide accelerations under fp32 precision but will decrease the memory usage, while preserving model accuracy (default: True)
- `use_env_seed: bool` — If True, apply environment matrix initial embedding as FiLM on l=0 features using 4D `[s, s*r_hat]` representation. Internal dimensions are derived from `channels`: `embed_dim=min(channels, 128)`, `axis_dim=min(4 if embed_dim < 64 else 8, embed_dim-1)`, `type_dim=clamp(channels//4, 8, 32)`, `rbf_out_dim=max(32, embed_dim-2*type_dim)`, `hidden_dim=min(256, max(2*embed_dim, rbf_out_dim+2*type_dim))` (default: True)
- `random_gamma: bool` — If True, sample an independent random roll `gamma ~ U[0, 2π)` for every edge on every forward call, build a local-`+Z` roll quaternion, and left-compose it with the edge-aligned quaternion before Wigner-D evaluation (default: True)

Optimizer routing note:

- HybridMuon uses name-based routing to separate Adam / AdamW / Muon paths (case-insensitive):
  - final effective parameter name segment containing `bias` → Adam (no weight decay)
  - final effective parameter name segment starting with `adam_` → Adam (no weight decay)
  - final effective parameter name segment starting with `adamw_` → AdamW (decoupled weight decay)
  - trailing numeric `ParameterList` indices are ignored when deriving the effective segment
  - all other parameters follow shape-based routing (2D → Muon, otherwise → AdamW)
- SeZM norm/layer-scale/frequency parameters use `adam_` prefixes (`adam_scale`, `adam_so2_layer_scales`, `adam_ffn_layer_scales`, `adam_freqs`) so HybridMuon routes them to Adam (no weight decay).
- For HybridMuon with SeZM, recommended routing mode is `muon_mode = "slice"`:
  - 2D weights (SO2Linear, FocusLinear): Muon (same as mode=2d)
  - 3D SO3Linear `(F*(lmax+1), C_out, C_in)`: per-(focus, l) independent Muon with correct rectangular scale
  - `adam_`/`bias` parameters: Adam (name-based routing takes priority)
- HybridMuon supports optional Muon-path Magma-lite damping via `magma_muon` (default: `false`):
  - computes block-wise cosine alignment between Muon momentum and current gradients
  - applies EMA smoothing on alignment score (`decay=0.9`)
  - rescales Muon updates to `[0.1, 1.0]` for stability under noisy gradients
  - does not change Adam/AdamW paths and does not use Bernoulli update dropping

Detailed Magma-lite behavior (`training.magma_muon`):

- Scope and insertion point

  - active only on the Muon route after momentum update (`m_t`) is computed
  - damping is applied to Muon update right before parameter update
  - no change to Adam or AdamW branches

- Block definition

  - `muon_mode = "slice"`: one score per matrix slice `(..., m, n)` in Muon view
  - `muon_mode = "2d"` or `"flat"`: one score per parameter
  - state is stored as `optimizer.state[param]["magma_score"]`
    - shape `(batch_size,)` in slice mode
    - shape `(1,)` in 2d/flat mode
  - if block shape changes unexpectedly (e.g., mismatch after load), score is reinitialized to `0.5`

- Alignment and scaling formula (implementation)

  - cosine alignment (FP32):
    - `cos = clamp( <m_t, g_t> / (||m_t|| * ||g_t|| + eps), -1, 1 )`
    - `eps = 1e-12`
  - temperature sigmoid:
    - `raw_sig = sigmoid(cos / tau)`, `tau = 2.0`
  - range stretching to `[0, 1]`:
    - `smin = sigmoid(-1/tau)`, `smax = sigmoid(1/tau)`
    - `raw = clamp((raw_sig - smin) / (smax - smin), 0, 1)`
  - EMA score:
    - `score_t = 0.9 * score_{t-1} + 0.1 * raw`
  - final damping scale:
    - `scale = 0.1 + 0.9 * score_t` (always in `[0.1, 1.0]`)
  - Muon update:
    - `delta_muon = delta_muon * scale`

- Practical interpretation

  - strong momentum-gradient agreement (`cos` near `1`) gives larger scale
  - poor or opposite alignment (`cos` near `0` or `-1`) damps the update
  - `min_scale = 0.1` avoids complete update starvation on hard blocks

- Usage guidance

  - keep `magma_muon = false` when baseline is already stable
  - enable `magma_muon = true` when multi-focus runs show frequent gradient spikes / unstable loss
  - this is a stability-first variant; unlike the paper's full Magma, it intentionally avoids stochastic Bernoulli masking

- Differences from the original Magma paper and rationale

  - No Bernoulli skip masking:
    - paper: `delta <- Q * S * delta`, with `S ~ Bernoulli(0.5)`
    - here: `delta <- scale * delta` (dense every step)
    - rationale: force-field training is more sensitive to intermittent block freezing; dense damping is more stable for noisy second-order/coupled objectives
  - Muon-route-only application:
    - paper: generic wrapper for multiple adaptive optimizers
    - here: only HybridMuon's Muon branch is modulated
    - rationale: observed instability is concentrated on large matrix Muon updates; keeping Adam/AdamW unchanged avoids side effects on bias/norm/layer-scale paths
  - Sigmoid output stretching + minimum scale floor:
    - paper: directly uses `sigmoid(cos/tau)` as alignment score
    - here: score is stretched to `[0, 1]` and mapped to `[0.1, 1.0]`
    - rationale: direct sigmoid with `tau=2` has narrow dynamic range; stretching improves control sensitivity, and the `0.1` floor prevents full starvation of hard blocks
  - Block definition tied to Muon routing:
    - paper: abstract block partition
    - here: block granularity follows Muon view (`slice`: per `(..., m, n)` slice; `2d/flat`: per parameter)
    - rationale: preserves structural independence of SeZM focus/l slices and aligns damping with the actual update operator

Note: Neighbor normalization (graph-style degree normalization) is always enabled.
Note: `focus_softmax_tau` (default `1.0`) and `focus_label_smoothing` (default `0.02`) are internal `SO2Convolution` parameters and are not exposed in descriptor top-level config.

### Interface Compatibility Notes

SeZM uses `_ENV_DIM = 1` (se_r style) for `EnvMatStatSe` compatibility. This means:

- `ndescrpt = nnei * 1` (only radial statistics are collected)
- `mean` and `stddev` statistics are maintained but not used in the forward pass (SeZM uses radial basis functions directly instead of traditional env_mat)

Output:

- returns only `l=0` features as descriptor: `(nf, nloc, channels)`

______________________________________________________________________

## Serialization

`serialize()` captures:

- top-level descriptor config, including `l_schedule`, `m_schedule`, `use_env_seed`,
  `full_attn_res`, `block_attn_res`, `so2_attn_res`, `sandwich_norm`, and inner-clamp settings
- the full descriptor `state_dict()` payload under `@variables`
- an `env_mat` compatibility payload

At the descriptor top level, serialization is flat: `DescrptSeZM.serialize()` does
not recursively pack per-submodule payloads. Instead, `DescrptSeZM.deserialize()`
reconstructs `DescrptSeZM(**config)` and restores the full `state_dict()`.

Individual SeZM submodules such as `DepthAttnRes`, `SO2Convolution`, and
`SeZMInteractionBlock` still expose standalone `serialize()` / `deserialize()`
helpers using the same `config` + `@variables` convention when tested in isolation.

Version: `@version: 1`

______________________________________________________________________

## Physics & Numerics

### C³ cutoff envelope

Envelope is a C³-continuous polynomial to enforce smoother high-order derivatives at `rcut`:

For `x = r / rcut`:

- `E(x) = 1 - 56*x^5 + 140*x^6 - 120*x^7 + 35*x^8` for `x in [0, 1)`
- `E(x) = 0` for `x >= 1`

The coefficients satisfy `E(0)=1, E(1)=0, E'(1)=0, E''(1)=0, E'''(1)=0`, ensuring C³ continuity.

The C³ envelope is applied in two places:

1. In `RadialBasis.forward()`: multiplied into the radial basis functions
1. As `edge_env`: applied to all edge messages

This double-guarantee ensures:

- message is 0 at `rcut`
- d(message)/dr is 0 at `rcut`
- d²(message)/dr² is 0 at `rcut`
- d³(message)/dr³ is 0 at `rcut`

### Conservative forces

- Edge-local geometry stays on a single differentiable runtime chain:
  `edge_vec -> stable edge quaternion -> Wigner-D`.
- The edge quaternion is built directly from `edge_vec` without detach.
- Wigner-D blocks are computed from those quaternions and remain differentiable.
- Vector and quaternion normalizations clamp squared norms before `sqrt`
  (e.g. `sqrt(clamp(||x||^2, eps^2))`) to avoid NaN gradients at zero vectors or masked branches.

### Wigner-D blocks (real SH basis)

SeZM uses real-basis Wigner-D blocks to rotate per-degree features between the global frame
and the edge-aligned local frame. The block-diagonal matrices are computed by
`WignerDCalculator` in `deepmd/pt/model/descriptor/sezm_nn/wignerd.py`.

#### Geometric contract

- `build_edge_quaternion(edge_vec)` returns a **global->local** edge rotation in quaternion form.
- Its rotation matrix `R(q_edge)` satisfies:
  - `v_local = R(q_edge) @ v_global`
  - `R(q_edge) @ (edge_vec / ||edge_vec||) = (0, 0, 1)`
- The quaternion is built from two exact edge-to-`+Z` charts:
  - a chart regular away from the `-Z` pole
  - a chart regular away from the `+Z` pole
- A `C^inf` normalized-linear blend is used only inside their overlap, so the represented
  edge-aligned rotation stays smooth across both pole neighborhoods.
- When `random_gamma=True`, SeZM samples a per-edge roll `gamma` and left-composes
  `q_gamma_z(gamma)` with the edge-aligned quaternion:
  `q_total = q_gamma_z(gamma) * q_edge`.
  This preserves the edge-alignment invariant because a local `+Z` roll leaves
  `(0, 0, 1)` unchanged while randomizing the in-plane gauge.

#### Representation contract

- For each degree `l`, real SH channels are ordered by `m=-l..+l` (index `i = m + l`).
- `D_full` is block-diagonal with block `l` occupying indices `[l^2 : (l+1)^2)`.
- `Dt_full = D_full^T` is the inverse rotation (local->global).
- `l=0` is the scalar identity block.
- `l=1` is computed directly from the quaternion-induced Cartesian rotation.
- `l=2` uses a dedicated degree-4 quaternion tensor-contraction kernel.
- `l=3,4` use dedicated quaternion monomial kernels; when both are required they are
  emitted by one shared degree-8 matrix multiply.
- `l>=5` uses the generic quaternion polynomial evaluator, so arbitrary `lmax` remains
  available without changing the packed representation contract.

#### Usage in message passing

- Rotate to local (reduced): `x_local = bmm(D_full[:, coeff_index_m, :], x_global)`
- Rotate back (reduced): `x_global = bmm(Dt_full[:, :, coeff_index_m], x_local) * rotate_inv_rescale(l)`
- `project_D_to_m` / `project_Dt_from_m` cache the projected blocks keyed by the normalized
  string key `"lmax:mmax"`.

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

______________________________________________________________________

## Caching Strategy (Critical)

**EdgeFeatureCache is built exactly once per `forward()`**.

Why caching matters:

- The expensive part is edge geometry + quaternion/Wigner generation.
- Message passing blocks reuse the same per-edge rotation objects and radial features.
- This avoids an O(#blocks) multiplier on per-edge trig/matrix ops.

What is cached / reused:

- All per-edge tensors needed by all blocks (geometry, radial basis, envelope, Wigner-D blocks).

- Focus streams do not duplicate geometric caches. `n_focus` only adds a feature axis;
  `edge_vec`, `edge_rbf`, `D_full`, and `Dt_full` remain on the original edge axis.

- `radial_feat`: computed once in `compute_dtype`, GIE consumes the pure radial part `radial_feat[:, 1:, :]` (no type fusion), then type embeddings are fused via a single `embedding_bag` reduction and **per-block truncated slices** are prebuilt according to `l_schedule`.

- Parallel rotation projection caches: `project_D_to_m` / `project_Dt_from_m` project block-diagonal Wigner-D to the m-major truncated layout keyed by `"lmax:mmax"`, shared by all blocks and available in both eager and TorchScript.

- Dtypes: `compute_dtype = get_promoted_dtype(dtype)` is set once in `__init__` and reused for geometry, radial basis/MLP, Wigner calculators, and the final l=0 mixer; runtime casts happen once on `extended_coord`, `radial_feat`, and the final scalar output.

______________________________________________________________________

## DeePMD Interface Compatibility

SeZM follows the **new-style descriptor interface** (same as `dpa3`), using `extended_coord` / `extended_atype` parameter names (instead of `coord_ext` / `atype_ext` used by older descriptors like `se_a` and `se_r`).

- Implements the required `BaseDescriptor` interface (forward, stats accessors, (de)serialization, neighbor info, exclusion updates).
- `_ENV_DIM = 1` for `EnvMatStatSe` compatibility; statistics are stored but not used in forward.
- Not implemented: `share_params()`, `change_type_map()`.

______________________________________________________________________

## Quick VRAM Estimation Formulas

This section provides formulas for quick GPU memory estimation given known model parameters. DeePMD-kit prints total parameter count `P` at training start; combine `P` with the config values below to estimate VRAM.

### Notation

| Symbol | Meaning                                          | Source       |
| ------ | ------------------------------------------------ | ------------ |
| P      | Total trainable parameters (printed by DeePMD)   | training log |
| N      | Number of atoms per frame                        | system       |
| nnei   | Max neighbors (sel)                              | config       |
| E      | Number of edges = N × nnei                       | derived      |
| L      | Max angular momentum (max of l_schedule)         | config       |
| D      | Spherical harmonics dimension = (L+1)²           | derived      |
| C      | Channel width                                    | config       |
| F      | FFN hidden dimension (ffn_neurons)               | config       |
| B      | Number of interaction blocks (len of l_schedule) | config       |
| b      | Bytes per element (4 for FP32, 2 for FP16/BF16)  | dtype        |

### 1. Parameter Memory

Parameter count `P` is already known from training log.

```
M_param = P × b
```

For Adam optimizer, training requires 4 copies (param + grad + momentum + variance):

```
M_train_param = 4 × P × b
```

> Example: P = 2M, FP32 → M_param ≈ 8 MB, M_train_param ≈ 32 MB.
> Parameter memory is typically negligible compared to activation memory.

### 2. Activation Memory (Inference)

Activation memory is dominated by **edge-level tensors**. The two major components:

#### 2.1 Persistent Edge Cache (computed once, shared across all blocks)

```
M_wigner  = 2 × E × D² × b          # D_full + Dt_full (Wigner-D matrices)
M_radial  = B × E × D × C × b       # radial_feat, one per block (different lmax truncation)
M_cache   = M_wigner + M_radial      # small terms (edge_vectors, envelope, etc.) ignored
```

> M_wigner uses D = (L+1)² where L = max(l_schedule). Each block's radial_feat uses its own D_block = (l_block+1)², but for quick estimation use D for all blocks (upper bound).

#### 2.2 Per-Block Transient Peak (only one block active at a time during inference)

```
M_so2conv = E × D × C × b           # SO2Conv intermediate (rotation + message)
M_ffn     = E × D × 2F × b          # FFN up-projection (GLU doubles width)
M_block   = max(M_so2conv, M_ffn)    # peak of the two stages (not simultaneous)
```

> If 2F > C (typical: F=96, C=64 → 2F=192 > 64), FFN dominates the per-block peak.

#### 2.3 Total Inference VRAM

```
M_infer ≈ M_param + M_cache + M_block
        = P×b + (2×E×D² + B×E×D×C)×b + E×D×max(C, 2F)×b
```

Simplified (dropping P×b which is small):

```
M_infer ≈ E × D × b × [2D + (B+1)×C + max(C, 2F)]
                         ^^^   ^^^^^     ^^^^^^^^^^
                       Wigner  radial    transient
```

### 3. Activation Memory (Training)

During training, autograd saves intermediate tensors for backward across **all** blocks simultaneously. Each block saves approximately:

```
M_saved_per_block ≈ k × E × D × C × b
```

where k ≈ 4–6 accounts for: pre-norm input, SO2Conv intermediates (rotation result, message), FFN up-projection, residual inputs.

```
M_train_act ≈ M_cache + B × k × E × D × C × b
```

Total training VRAM:

```
M_train ≈ M_train_param + M_train_act
        = 4×P×b + [2×E×D² + B×(1+k)×E×D×C] × b
```

Simplified:

```
M_train ≈ E × D × b × [2D + B×(1+k)×C]     (k ≈ 5)
```

> Training is roughly **5–8× inference** due to saved activations across all blocks.

### 4. Scaling Summary

| Factor       | Scaling                                   | Note                                 |
| ------------ | ----------------------------------------- | ------------------------------------ |
| N (atoms)    | **Linear**                                | E = N × nnei                         |
| nnei         | **Linear**                                | E = N × nnei                         |
| C (channels) | **Linear**                                | dominates edge features E×D×C        |
| L (lmax)     | **Quadratic**                             | D = (L+1)², Wigner-D is E×D²         |
| B (blocks)   | **Linear** (train) / **Constant** (infer) | train saves all blocks; infer reuses |
| F (ffn)      | **Linear**                                | only affects transient peak          |

### 5. Quick Reference Formula (FP32)

For a quick ballpark in **MB**, with FP32 (b=4):

```
M_infer (MB) ≈ N × nnei × (L+1)² × [2(L+1)² + (B+1)×C + 2F] × 4 / 1e6

M_train (MB) ≈ N × nnei × (L+1)² × [2(L+1)² + B×6×C] × 4 / 1e6 + 4P×4/1e6
```

> **Bottleneck**: Edge-level tensors (E×D×C) and Wigner-D matrices (E×D²) dominate. For larger systems, reducing nnei or L yields the most significant memory savings.
