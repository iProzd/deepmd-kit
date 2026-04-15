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
- `deepmd/pt/model/descriptor/sezm_nn/activation.py`
  - `GatedActivation`, `SwiGLU`, `S2GridProjector`, `SwiGLUS2Activation`
- `deepmd/pt/model/descriptor/sezm_nn/embedding.py`
  - `SeZMTypeEmbedding`, `GeometricInitialEmbedding`, `EnvironmentInitialEmbedding`
- `deepmd/pt/model/descriptor/sezm_nn/norm.py`
  - `EquivariantRMSNorm`, `ReducedEquivariantRMSNorm`, `ScalarRMSNorm`
- `deepmd/pt/model/descriptor/sezm_nn/attention.py`
  - destination-wise envelope-gated softmax for SO(2) attention
- `deepmd/pt/model/descriptor/sezm_nn/attn_res.py`
  - `DepthAttnRes`
- `deepmd/pt/model/descriptor/sezm_nn/so3.py`
  - `ChannelLinear`, `FocusLinear`, `SO3Linear`
- `deepmd/pt/model/descriptor/sezm_nn/so2.py`
  - `SO2Linear`, `SO2Convolution`
- `deepmd/pt/model/descriptor/sezm_nn/ffn.py`
  - `EquivariantFFN`
- `deepmd/pt/model/descriptor/sezm_nn/block.py`
  - `SeZMInteractionBlock`
- `deepmd/pt/model/descriptor/sezm_nn/dens.py`
  - `ForceEmbedding`, direct-force / denoising vector heads, parallel `dens` fitting net
- `deepmd/pt/model/descriptor/sezm_nn/wignerd.py`
  - quaternion edge-frame construction and `WignerDCalculator`

______________________________________________________________________

## Goals (Non-Negotiable)

1. **Two explicit force modes**

   - `ener` mode keeps the standard conservative path: forces come from `autograd` of energy w.r.t. coordinates.
   - `dens` mode keeps the same descriptor trunk but swaps to a direct-force / denoising head that predicts forces without coordinate higher-order derivatives.
   - Geometry / rotations remain fully differentiable in both modes; **no `.detach()`** in edge rotations.

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

______________________________________________________________________

## Model Integration (PyTorch)

- Set `model.type = "SeZM"` (aliases: `"se_zm"`, `"sezm"`) to select the SeZM model scaffold. Aliases are resolved during configuration validation.
- `descriptor.type` follows user input (SeZM is recommended), and `fitting_net.type` is ignored; SeZM always uses `sezm_ener`.
- SeZM uses one public fitting config: `fitting_net`. The `dens` head reuses this same fitting configuration for its scalar energy branch; there is no separate user-facing `dens_fitting_net` setting.
- The optional `dens` head is materialized lazily: training creates it when `loss.type = "dens"`, and checkpoint loading recreates it automatically when `dens` weights are present.
- Internally it is built as `make_model(SeZMAtomicModel)`.

### Mode Routing

- Training mode is selected by `loss.type`.
  - `loss.type = "ener"` activates the conservative energy / autograd-force path.
  - `loss.type = "dens"` activates the parallel direct-force / denoising path.
- `forward_lower(...)` / the LAMMPS-style conservative interface only supports `ener`.
- Checkpoints without `dens` weights stay on the standard `ener` path.
- Checkpoints with `dens` weights recreate the `dens` head during loading.

### Descriptor Public Contract

- `DescrptSeZM.forward_with_edges(...)` returns:
  - scalar descriptor `(nf, nloc, channels)`
  - final equivariant latent `(nf * nloc, D_final, 1, channels)`
- The ordinary DeePMD descriptor entry `DescrptSeZM.forward(...)` keeps the standard return ABI and discards the latent internally, so existing descriptor call sites remain unchanged.

### `dens` Loss Configuration

- The `dens` loss keeps energy supervision and replaces the force branch with a mixed target:
  - uncorrupted atoms supervise globally normalized direct force
  - corrupted atoms supervise the normalized injected Gaussian noise vector `noise / dens_std`
- Standard `dens` training perturbs coordinates, passes `noise_mask` through the model, and feeds the clean force label back as `force_input`; the model converts it to a masked external equivariant force embedding for corrupted atoms.
- The initial SeZM defaults follow the EquiformerV3 DeNS recipe:
  - `dens_prob = 0.5`
  - `dens_fixed_noise_std = true`
  - `dens_std = 0.025`
  - `dens_corrupt_ratio = 0.5`
  - `dens_denoising_pos_coefficient = 10`
- The current implementation supports only `dens_fixed_noise_std = true`.
- Output statistics use one dedicated `dens` normalization path that matches the
  EquiformerV3 trainer semantics:
  - the standard `ener` branch still keeps the DeePMD per-type energy bias path
  - `dens` also fits one global direct-force RMSD scale and stores it as
    `rmsd_dforce`
  - the `dens` model outputs stay in normalized space during training:
    reduced energy reuses the standard DeePMD per-type energy bias and the
    broadcast global residual `out_std`, clean direct-force uses
    `rmsd_dforce`, and denoising predictions stay in the normalized
    `noise / dens_std` space
  - the public `SeZMModel(...).energy` / `.force` outputs stay in physical
    units; in `dens` mode the public `force` always denotes the clean direct
    force branch, while denoising predictions remain internal normalized tensors
    consumed by the `dens` loss
  - logging converts normalized predictions back to physical units, and the
    public `mae_f` / `rmse_f` in `dens` mode report only clean-force errors so
    they stay directly comparable with the `ener` mode force logs

### Optional compile path (compact sparse edges)

SeZM supports an optional **compact sparse edge** path that enables `torch.compile` while preserving
the standard DeePMD neighbor list behavior:

- Enable with `model.use_compile = true`.
- The model still builds the DeePMD neighbor list eagerly, formats it outside the compiled graph,
  then **compacts it into a sparse edge list** `(src, dst, edge_vec, edge_mask)` containing only
  valid local edges.
- The main SeZM graph takes **local** `(coord, atype)` together with that sparse edge list as input.
  `edge_vec` enters the graph directly, and the force gradient path is reattached cleanly from
  local coordinates before descriptor evaluation.
- The descriptor accepts the sparse edge list directly and compiles a **pure tensor graph**.
  - `ener` uses `make_fx(tracing_mode="symbolic")` before `torch.compile(...)` to stabilize higher-order autograd capture.
  - `dens` compiles the direct-force compute function directly.
- The compiled graph keeps **dynamic total node / edge counts**.
  - `ener` compile keeps the second-order derivatives needed by conservative force training.
  - `dens` compile uses a dedicated direct-force path that does not require coordinate higher-order derivatives.
- Training follows `model.use_compile`, but `eval()/inference/full_validation` default to the eager path
  even when `use_compile=true`.
- Set `DP_COMPILE_INFER=1` before model construction to make `eval()/inference` use the compile path
  by default without changing any Python call sites. The environment variable is read at model init time.
- `SeZMModel` always starts from the standard DeePMD neighbor list, then converts it to the same
  sparse edge representation `(src, dst, edge_vec, edge_mask)` before entering the SeZM descriptor.
  Eager and compile share the same sparse-edge `core_compute` kernel and the same hand-written
  local energy / force / virial post-process, while the descriptor's ordinary
  `forward(extended_coord, extended_atype, nlist, mapping, ...)` entry remains available for other
  DeePMD models that want to reuse SeZM without going through `SeZMModel`.

In both compile modes, all geometry (edge vectors, Wigner-D, radial basis) remains differentiable,
and no global node/edge padding is introduced.

`dens` compile limitation:

- `dens` mode does not support analytical bridging potentials (`model.bridging_method != "none"`).
- If short-range ZBL bridging is required, use `ener` mode.

______________________________________________________________________

## ZBL Bridging (Optional Short-Range Repulsion)

SeZM supports an optional analytical short-range repulsion potential that supplements the
ML-predicted energy via pure additive energy decomposition:

```
E_total = E_ZBL(r) + E_model(r̃)
```

where `r` is the true interatomic distance and `r̃` is the clamped distance seen by the descriptor.

### Design Principles

1. **Pure additive**: ZBL energy is added to the atomic energy _before_ autograd, so forces and
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
| `bridging_r_inner` | float | 0.9      | Inner clamping radius in Å. Descriptor frozen below this |
| `bridging_r_outer` | float | 1.3      | Outer clamping radius in Å. Transition zone upper bound  |

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

This ensures the descriptor receives _no information_ about the true distance when `r < r_inner`.
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

- **Main sparse-edge path** (`core_compute`): injected into `atom_energy` before autograd
  computes forces/virials.
- **Lower extended-coordinate path** (`forward_common_lower`): injected into `fit_ret["energy"]`
  before `fit_output_to_model_output`.

______________________________________________________________________

## High-Level Architecture

Text diagram (single forward pass):

```
SeZMModel main forward path:
  Inputs: coord, atype, box
    ├─ build DeePMD neighbor list -> extended_coord, extended_atype, mapping, nlist
    ├─ format_nlist outside the compiled graph
    ├─ build compact sparse edges `(src, dst, edge_vec, edge_mask)` from DeePMD nlist
    ├─ graph inputs: local `(coord, atype)` + sparse edges
    └─ EdgeFeatureCache (built once via build_edge_cache_from_edges)
       ├─ edges: (src, dst) global indices, edge_vec
       ├─ edge_type_feat: per-edge type embedding (src+dst)
       ├─ edge_rbf: Bessel radial basis via sinc × C² envelope
       ├─ edge_env: C³ cutoff envelope weights (flattened to valid edges)
       ├─ D_full, Dt_full: block-diagonal Wigner-D matrices
       └─ inv_sqrt_deg: inverse sqrt smooth degree (`sum(edge_env²)`) for normalization

Descriptor compatibility path for other models:
  Inputs: extended_coord, extended_atype, nlist, mapping
    └─ EdgeFeatureCache (built once via build_edge_cache)
       ├─ standard-path geometry/RBF chain (`not training` on supported CUDA+dtypes):
       │  fused `coord_gather -> edge_vec -> edge_len -> inner_clamp -> edge_env -> edge_rbf`
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
      ├─ main tensor layout is fixed as (N, D, 1, C), contiguous
      ├─ EquivariantRMSNorm (pre-SO2, singleton-focus on (N, D, 1, C))
      ├─ Multi-Focus SO(2) Convolution (enabled for ALL lmax, including lmax=0)
      │  ├─ `pre_focus_mix`: full-channel projection on (N, D, C)
      │  ├─ rotate/bmm in hidden width (E, D, H), then SO2 stack on strided (E, F, Dm, Cf)
      │  ├─ optional SO(2) internal AttnRes over local layer history when `so2_attn_res != "none"`
      │  └─ `post_focus_mix`: hidden width H -> channel width C
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

  Post-process in SeZMModel:
    ├─ fitting + output statistics + atom mask
    │  ├─ `ener`: add per-type energy bias
    │  └─ `dens`: keep energy / direct-force / denoising outputs in normalized space, reduce energy with the global `dens` normalizer semantics, then mix clean-force and denoising branches with `noise_mask`
    ├─ optional ZBL energy added on the sparse-edge path
    ├─ hand-written autograd for force / virial on the sparse-edge graph
    └─ local DeePMD-style outputs `(energy, energy_redu, energy_derv_r, energy_derv_c[_redu], mask)`
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

### 3. SO(3) Linear Layers

- `ChannelLinear` applies a shared linear map on the last channel axis without
  introducing a focus dimension.
- `FocusLinear` applies per-focus linear mixing on `(B, F, C)` and stores weight
  as `(Cin, F*Cout)` (focus folded on output side).
- `SO3Linear` keeps per-focus independent mixing on `(N, D, F, C)` and stores
  weight as `(lmax+1, Cin, F*Cout)` (focus folded on output side).
- Implemented as `index_select(expand_index)` + `einsum`; bias is applied only for `l=0`.

### Multi-focus SO2 + global FFN

- The multi-focus SO(2) design is inspired by both MHA and dense MoE:
  multiple focus streams process the same geometric context in parallel,
  while `pre_focus_mix` / `post_focus_mix` provide dense channel projections
  between the backbone width and the internal focus width.
- Backbone tensor stays in a single contiguous layout `(N, D, 1, C)` across all blocks.
- Real multi-focus structure is used only inside `SO2Convolution`, where the hidden width is `H = n_focus * focus_dim`.
- SO(2) pre/post norms on the block boundary operate on `(N, D, 1, C)`.
- Geometry cache (`edge_vec`, `edge_rbf`, `D_full`, `Dt_full`) is still built once per forward and shared across all focus streams.
- Radial features keep `(E, L, C)` at descriptor level; when `H != C`, the SO(2) branch applies an internal radial hidden projection.
- FFN branch stays on the singleton-focus backbone `(N, D, 1, C)` with only view-based reshapes.

### 4. Full equivariant FFN

- Default path: `SO3Linear -> GatedActivation -> SO3Linear` with a residual connection.
  Gates are derived from `l=0` scalars (one gate per `l`) and expanded across all `m`.
- When `s2_activation[1]=true` and `grid_mlp=false`, the first projection outputs
  `2 * hidden_channels`, then `SwiGLUS2Activation` uses the `l=0` slice to
  build a scalar `SwiGLU` branch plus a sigmoid gate, applies point-wise
  multiplication on the S2 grid, gates the reconstructed coefficients, and
  merges the scalar branch back to `l=0`. In the SeZM block wrapper, the FFN
  path uses a square grid resolution `[max(R_phi, R_theta), max(R_phi, R_theta)]`.
- When `grid_mlp=true`, the block-internal FFN switches to
  `SO3Linear(in -> hidden) -> scalar LinearSwiGLU(input l=0) + packed-grid point-wise MLP(hidden -> hidden) -> merge at l=0 -> SO3Linear(hidden -> out)`.

### 5. Multi-layer SO(2) convolution

- Uses the edge-local m-major reduced layout controlled by `mmax`.
- Stacks `SO2Linear` with an intermediate non-linearity for `so2_layers`.
  - Default path uses `GatedActivation(mmax=...)`.
  - When `s2_activation[0]=true`, every intermediate layer outputs `2 * focus_dim`
    channels and `SwiGLUS2Activation` uses the reduced-layout `l=0` slice to
    build the scalar `SwiGLU` branch and sigmoid gate around the S2-grid
    multiplicative path.
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
- `ChannelLinear`, `FocusLinear`, `SO3Linear`, and `SO2Linear` all support an optional `init_std` parameter: when given, weights are initialized with `Normal(0, init_std)` instead of the default scheme. Use `init_std=0.0` for zero initialization (e.g., residual output projections).

### 7. Unified weight layout convention

All learnable weight matrices use **(in, out) convention** (rows = fan_in, cols = fan_out),
and focus-aware modules fold the focus dimension `F` on the **output (cols) side**:

- `ChannelLinear`
  - stored weight: `(C_in, C_out)` — 2D, Muon
  - stored bias: `(C_out,)` — 1D, Adam
  - runtime contraction: `einsum("...i,io->...o")`
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

- `x`: `torch.Tensor` with shape `(N, D, 1, C)` (contiguous)
  - `N = nf * nloc`
  - `C = channels`
  - `D = ebed_dim = (lmax + 1)^2 = sum_{l=0..lmax} (2l + 1)` is the SO(3) embedding dimension
  - the singleton focus axis is kept only for module reuse

View conventions used inside blocks:

- `x.view(N, D, C)` for full-channel rotate/bmm and FFN mixing
- `x.view(N, D, 1, C)` at block boundaries
- inside `SO2Convolution`, hidden features are reshaped to `(E, F, Dm, Cf)` with
  `F = n_focus`, `Cf = focus_dim`, and `H = F * Cf`

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

This design avoids the dynamic-output-size `nonzero` kernel for distance filtering and enables smoother degree/normalization (no discontinuous edge count jumps at `rcut` boundary).

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

- Apply `non_linear` (between layers, Identity for last layer):
  - default path: `GatedActivation`
    - l=0: uses the configured base `activation_function`
    - l>0: sigmoid(l=0) gate; implementation uses preallocated output instead of cat
  - `s2_activation[0]=true`: `SwiGLUS2Activation`
    - `l=0` slice -> scalar `SwiGLU` branch
    - `l=0` slice -> sigmoid gate
    - reduced coefficients -> flattened S2 grid -> point-wise multiplication -> reduced coefficients
    - sigmoid gate modulates the reconstructed coefficients, then the scalar branch is added back to `l=0`
- LayerScale + Residual: `x_local = residual + scale * x_local` (scalar scale, init 1e-3 when `layer_scale=True`; bare residual otherwise)

1. **Cross-focus competition (automatic when multi-focus is enabled)**:
   - Enabled only when `n_focus>1`
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

With `grid_mlp=true`, the block-internal FFN instead keeps a hidden-width
SO(3) projection, projects the packed SO(3) coefficients to the S2 grid,
applies a point-wise grid MLP on the packed S2 grid, projects the grid features
back to packed SO(3) coefficients, adds a separate scalar `LinearSwiGLU`
branch back to `l=0`, and then applies the output SO(3) projection.

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
SeZMInteractionBlock:  # x shape: (N, D, 1, C)

  # === Path 1: SO(2) Convolution ===
  x_pre = pre_so2_norm(x)                          # (N, D, 1, C)
  y = so2_conv(view(x_pre, N, D, C), edge_cache, radial_feat)  # (N, D, C)
  y = post_so2_norm(view(y, N, D, 1, C))          # (N, D, 1, C)
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

  return view(x_ffn.squeeze(1), N, D, 1, C)
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
- `ffns[i]`: `EquivariantFFN` with SO(3) linear projections and either
  the direct `GatedActivation` path, S2-grid SwiGLU activation, or the optional
  grid-MLP FFN path
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
  - when `DP_TRITON=1` environment variable is set and the model is in eval mode (not training), these two subset-rotation paths are fused into Triton custom ops with custom backward on supported CUDA dtypes
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
- `n_focus: int` — Number of parallel focus streams used only inside the SO(2) convolution (default: 1)
- `focus_dim: int` — Hidden width per focus stream inside SO(2). `0` means using `channels` (default: 0)
- Cross-focus softmax competition is enabled automatically when `n_focus > 1`. Logits are built from l=0 scalar channels and normalized across focus streams; weights are broadcast to all `(l, m)` components in each focus
- `n_radial: int` — Number of radial basis functions (default: 10)
- `radial_mlp: list[int]` — Hidden layer sizes for radial networks. An output layer of size (l_schedule[0]+1)\*channels is automatically appended (default: [64])
- `so2_norm: bool` — If True, apply ReducedEquivariantRMSNorm as pre-norm before each SO(2) mixing layer (except the last, which uses Identity). When False (default), pre-norm is Identity for all layers
- `so2_layers: int` — Number of SO2Linear layers per convolution (default: 4)
- `so2_attn_res: str` — SO(2)-internal depth-wise attention residual mode inside each interaction block. Allowed values: `none`, `independent`, `dependent` (default: `none`)
- `ffn_neurons: int` — Hidden width for block FFNs and the final scalar output FFN. `>0` uses the same explicit width for both. `0` lets each path resolve its own width from `channels`: `4 * channels` without GLU, `(8 / 3) * channels` with GLU, then round up to a multiple of 32 (default: 0)
- `grid_mlp: bool` — If True, use the optional grid-MLP structure for block-internal FFN units. The final `l=0` output head is unchanged (default: False)
- `ffn_blocks: int` — Number of FFN subblocks per interaction block (default: 1)
- `n_atten_head: int` — Number of attention heads when aggregating messages in SO(2) convolution. 0 applies a plain envelope-weighted scatter-sum (default: 1). When >0, the effective per-focus width (`focus_dim` or `channels` when `focus_dim=0`) must be divisible by `n_atten_head`, and envelope-gated grouped softmax attention with output-side head gate is applied. Attention uses `w^2 * exp(logit)` in the numerator and `zeta + sum(w^2 * exp(logit))` in the denominator.
- `sandwich_norm: list[bool]` — Pre/post-norm switches for residual branches: `[so2_pre, so2_post, ffn_pre, ffn_post]` (default: [True, False, True, False])
- `exclude_types: list[tuple[int, int]]` — Excluded type pairs
- `precision: str` — `float64` / `float32`
- `s2_activation: list[bool]` — Two booleans `[so2_enabled, ffn_enabled]`. `so2_enabled=true` makes the SO(2) gated activation path use `activation_function="silu"`. `ffn_enabled=true` makes the block-internal FFN path use `activation_function="silu"` and `glu_activation=true`. The final `l=0` output FFN is unchanged (default: `[False, False]`)
- `s2_grid_resolution: list[int]` — Two positive integers `[R_phi, R_theta]` used by the S2-grid activation. If omitted, SeZM resolves it from the first block `(lmax, mmax)` after schedule parsing as `[2 * mmax + 4, ceil_even(3 * lmax + 2)]`
- `mlp_bias: bool` — Whether to use bias in equivariant layers (SO3Linear l=0 bias, SO2Linear l=0 bias, GatedActivation gate linear bias, DepthAttnRes input-dependent query projection) and EnvironmentInitialEmbedding MLPs (`rbf_proj_layer1/2`, `g_layer1/2`) (default: False)
- `layer_scale: bool` — If True, apply learnable LayerScale on residual branches for training stability: per-focus-channel scales (init 1e-3) on each SO(2) mixing layer, and per-channel vector (init 1e-3) on each FFN subblock (default: False)
- `full_attn_res: str` — Descriptor-level full AttnRes mode over unit history. `independent` uses learned query vectors; `dependent` derives the query from the current SeZM state before the SO(2) unit, before each FFN unit, and before the final aggregation. Allowed values: `none`, `independent`, `dependent` (default: `none`)
- `block_attn_res: str` — Descriptor-level block AttnRes mode over block history. `independent` uses learned query vectors; `dependent` derives the query from the current SeZM state before the SO(2) unit, before each FFN unit, and before the final block aggregation. Allowed values: `none`, `independent`, `dependent` (default: `none`). Cannot be enabled together with `full_attn_res`
- `use_amp: bool` — If True, use automatic mixed precision (AMP) with bfloat16 on CUDA. This does not provide accelerations under fp32 precision but will decrease the memory usage, while preserving model accuracy (default: True)
- Triton rotation kernels are controlled by the `DP_TRITON` environment variable. Set `DP_TRITON=1` to enable fused Triton SO(2) rotation kernels on supported CUDA dtypes. This only affects the SO(2) rotation path and only in eval/inference mode (not training); the standard-path edge geometry/RBF chain has its own automatic eval-only Triton gate.
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
  - 2D weights (ChannelLinear, SO2Linear, FocusLinear): Muon (same as mode=2d)
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
- all SeZM `register_buffer()` tensors are persistent model state and therefore live inside
  `@variables`, including precomputed index tables, S2 projection matrices, reduced-layout
  maps, Wigner coefficient tables, and interface-compatibility buffers
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

### Smooth PES validation

For SeZM, the most useful smoothness regression test is **not** force-vs-finite-difference
agreement by itself. A force check can pass even when the PES shape is still not the clean
single-bowl curve expected around a symmetric equilibrium structure.

The recommended unit-test probes are direct **total-energy** scans:

- use one hard-coded symmetric **eight-atom two-sublattice template** in fractional coordinates:
  - `[0,0,0]`, `[0,1/2,1/2]`, `[1/2,0,1/2]`, `[1/2,1/2,0]`, `[1/2,1/2,1/2]`, `[1/2,0,0]`, `[0,1/2,0]`, `[0,0,1/2]`
- scale the cubic lattice so the nearest-neighbor distance matches the target boundary:
  - non-bridged near-cutoff probe: `r_nn = 4.95` Å with `rcut = 5.0` Å
  - bridged inner-boundary probe: `r_nn = r_inner = 0.9` Å
  - bridged outer-boundary probe: `r_nn = r_outer = 1.3` Å
- displace atom `0` along `x` over `[-0.1, 0.1]` Å
- enable `bridging_method="ZBL"` only for the `r_inner` / `r_outer` probes

The test should validate the shape directly from the sampled energy curve:

1. the second derivative keeps one sign over the whole scan window
1. the first derivative keeps one sign on the left branch and the opposite sign on the right branch
1. the equilibrium point is the unique extremum at the center of the scan

Because the probes intentionally use **randomized model weights**, the non-bridged near-cutoff
curve may be either bowl-up or bowl-down; both are acceptable as long as the curve is a single
smooth extremum. For the bridged `r_inner` / `r_outer` probes, the additional ZBL repulsion should
keep the curve bowl-up with a minimum at the symmetric center.

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
- Only valid edges enter coordinate gather and subsequent geometry / radial / rotation evaluation.
- `PairExcludeMask` returns a **keep mask** (1=keep, 0=excluded). It does not remove padding by itself, so always combine it with `nlist >= 0`.
- No zero-length vector is normalized for padding edges.

### Precision and device handling

- All submodules use `dtype: torch.dtype` (not `precision: str`) for constructor parameter.
- Device is obtained from global `env.DEVICE` at runtime; submodules store `self.device = env.DEVICE` only as a convenience reference, not for serialization.
- Each submodule stores `self.precision = RESERVED_PRECISION_DICT[dtype]` for serialization compatibility.
- `GatedActivation` and `SwiGLUS2Activation` are instantiated with the promoted `compute_dtype`. `GatedActivation` casts the scalar gate input to that compute dtype before `sigmoid` and casts the sigmoid result back to the incoming activation dtype before applying the multiplicative gate, while `SwiGLUS2Activation` keeps its original forward path and therefore uses that promoted dtype directly inside the scalar gate and S2-grid activation flow.
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

- Standard-path geometry/RBF Triton path: when the descriptor is in eval/inference mode (`self.training=False`) and the geometry dtype is a supported CUDA dtype, `build_edge_cache()` replaces the eager chain
  `coord_gather -> edge_vec -> edge_len -> inner_clamp -> edge_env -> edge_rbf`
  with the fused Triton geometry/RBF chain under `deepmd/pt/model/descriptor/sezm_nn/triton/`.
  Training always falls back to the eager geometry/RBF chain so the force/virial high-order
  derivative path keeps the original PyTorch autograd behavior.

- Parallel rotation projection caches: `project_D_to_m` / `project_Dt_from_m` project block-diagonal Wigner-D to the m-major truncated layout keyed by `"lmax:mmax"` in the eager fallback path. When `DP_TRITON=1` is set, the model is in eval mode, and CUDA+dtypes are supported, `SO2Convolution` skips these materialized projections and uses the Triton package under `deepmd/pt/model/descriptor/sezm_nn/triton/` for fused `global -> local reduced` and `local reduced -> global` rotations.

- Dtypes: `compute_dtype = get_promoted_dtype(dtype)` is set once in `__init__` and reused for geometry, radial basis/MLP, Wigner calculators, and the final l=0 mixer; runtime casts happen once on `extended_coord`, `radial_feat`, and the final scalar output.

- Triton package layout: Triton code is split by responsibility instead of keeping kernels, launchers, fallback policy, and autograd glue in one file. `constants.py` defines shared tile and mode constants, `dispatch.py` owns the single dispatch policy entry, `custom_ops.py` only launches Triton kernels, `autograd.py` owns the public API and eager fallback glue, and `kernels_small.py` / `kernels_generic.py` / `kernels_edge_geometry_rbf.py` keep the math kernels separated by responsibility.

- Triton dispatch: `DP_TRITON` environment variable is the user-facing opt-in switch. When it is set to `1` (and the model is in eval mode), `SO2Convolution` decides once in `__init__` whether Triton rotations are actually enabled from the module device/dtype, then resolves a fixed `rotation_mode` from `(dim_full, reduced_dim)`. The dispatch modes are `SMALL_LE1` / `SMALL_L2` / `SMALL_L3` for specialized kernels, `GENERIC_TILED` for the large-`l` tiled path, and `EAGER_REFERENCE` for the generic small-`K` case that would violate Triton's `K >= 16` `tl.dot` constraint.

- Edge geometry/RBF dispatch: the fused geometry/RBF chain is not user-configurable. `build_edge_cache()` simply checks `self.training`; eval/inference mode is allowed to use Triton, while training always uses the eager gather/clamp/envelope/radial chain. Unsupported devices or dtypes still fall back to eager inside the public Triton API.

- Small-l Triton kernels: `lmax<=3` uses dedicated kernel families with custom backward. `lmax=0,1` share the `SMALL_LE1` family through the packed full dimensions `1` and `4`; `lmax=2` uses `SMALL_L2`; `lmax=3` uses `SMALL_L3`. These specialized kernels keep one padded `16x16` block in registers and block only along the channel axis.

- Generic tiled Triton kernels: once `dim_full` exceeds the specialized families, SeZM uses the tiled kernels in `kernels_generic.py`. They keep `BLOCK_FULL = BLOCK_REDUCED = 16` so every `tl.dot` sees a legal tile, and all float32 contractions use `input_precision="ieee"` to match eager PyTorch numerics instead of TF32.

- Eager fallback boundary: non-CUDA or unsupported dtypes still use the eager PyTorch projection + `bmm` path. Even on CUDA, the generic path falls back to eager when `reduced_dim < 16`, because that case cannot satisfy Triton's current `tl.dot` contraction constraint.

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
| F      | Effective FFN hidden dimension                   | config/auto  |
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
M_ffn     = E × D × 2F × b          # Default FFN up-projection (GLU doubles width)
M_block   = max(M_so2conv, M_ffn)    # peak of the two stages (not simultaneous)
```

> If 2F > C (typical: F=96, C=64 → 2F=192 > 64), FFN dominates the per-block peak.
> When `grid_mlp=true`, the `2F` activation buffer is replaced by a hidden-width
> SO(3) tensor plus transient S2-grid buffers from the point-wise grid MLP.

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
