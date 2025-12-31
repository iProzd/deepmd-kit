# SeZM-Net — Speed-First SO(2) Equivariant Descriptor for DeePMD-kit (PyTorch)

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
       ├─ node_type_feat: type embedding for nodes
       ├─ edge_rbf: Bessel radial basis × C² envelope (trainable frequencies)
       ├─ edge_sw: DeePMD smooth weights in flattened edge layout
       ├─ D_list[l], Dt_list[l]: Wigner-D blocks per l
       └─ inv_sqrt_deg: inverse sqrt degree for normalization

Node init:
  ├─ l=0: Type embedding
  └─ l>0: Zonal (m=0) initial embedding via cached Wigner-D + radial MLP

Interaction blocks (pyramid schedule):
  for block i:
    ├─ slice x to ebed_dim(l_schedule[i]) (discard higher-l if needed)
    ├─ SeparableRMSNorm (l=0/l>0 separated, per-l affine, l=0 centering)
    ├─ SO(2) Convolution (enabled for ALL lmax, including lmax=0)
    ├─ Residual
    ├─ GatedActivation (per-l independent gates from l=0)
    └─ Full Equivariant FFN (operates on ALL degrees l=0..lmax, per-l gating)

Output:
  └─ return x(l=0) only: (nf, nloc, channels)
```

---

## Key Design Decisions

### 1. Trainable Radial Basis Frequencies (NequIP-style)

The radial basis uses **trainable frequencies** stored as `nn.Parameter`:

```python
# Frequencies: n*pi/rcut, n=1..n_radial
# Stored as trainable nn.Parameter with shape (1, n_radial)
self.freqs = nn.Parameter(freqs.view(1, -1), requires_grad=True)
```

This allows the model to learn optimal radial basis spacing during training, following the NequIP approach. The frequencies are:

- Initialized to standard Bessel frequencies: `n * π / rcut`
- Stored in the model's configured precision (float64 or float32)
- Included in serialization/deserialization

### 1.5 SeparableRMSNorm (Per-l Affine with Centering)

`SeparableRMSNorm` normalizes l=0 and l>0 features **separately** while supporting per-l affine scaling:

```python
# l=0: centering + RMS norm + per-channel affine
x0 = x0 - mean(x0)  # centering (optional, default=True)
x0 = x0 / rms(x0)
x0 = x0 * weight[0] + bias  # affine with bias

# l>0: RMS norm (no centering) + per-l affine (no bias)
xt = xt / rms(xt)  # rms over all (D-1, C)
xt = xt * weight[expand_index]  # per-l scale, expanded to all m
```

Key properties:

- **Separable RMS**: l=0 and l>0 compute RMS independently
- **Centering**: Only for l=0 (l>0 must remain zero-mean for equivariance)
- **Per-l affine**: Weight shape `(lmax+1, C)`, with `expand_index` mapping to all m-components
- **Bias**: Only for l=0 (when both `affine=True` and `centering=True`)

### 2. Full Equivariant FFN (All Degrees, Per-l Gating)

The FFN operates on **ALL spherical harmonic degrees** (l=0 to lmax), not just scalars. It uses a gated FFN with **per-l independent gates**:

```
EquivariantFFN:
  1. Per-degree linear mixing (C → hidden)
  2. GatedActivation:
     - l=0: SiLU activation
     - l>0: Each degree l has an independent gate derived from l=0 scalar features via gate_linear
           Gates are expanded to all m components within each l-block
  3. Per-degree linear mixing (hidden → C)
  4. Residual connection
```

The `GatedActivation` module generates `lmax` independent gates from scalar features using a linear layer (`gate_linear: C → lmax*C`), then expands each per-l gate to all `2l+1` m-components using a precomputed `expand_index` buffer.

### 3. PerDegreeLinearV2 (Vectorized Degree-wise Linear)

`PerDegreeLinearV2` implements degree-wise linear self-interaction shared across all m components within each l-block, using vectorized operations for efficiency:

```python
# Per-l weight matrix: (lmax+1, C, C)
# Each l has an independent C x C linear transformation shared across 2l+1 m components.
weight = nn.Parameter(torch.randn(lmax + 1, C, C))
bias = nn.Parameter(torch.zeros(C))  # Only for l=0

# expand_index: maps each (l,m) position to its l value
expand_index = [0, 1, 1, 1, 2, 2, 2, 2, 2, ...]  # length = (lmax+1)^2

# Forward pass (vectorized):
weight_expanded = index_select(weight, dim=0, index=expand_index)  # (D, C, C)
out = einsum("bmi,mci->bmc", x, weight_expanded)  # (N, D, C)
out[:, 0, :] += bias  # Add bias only to l=0
```

Key properties:

- **Vectorized**: Uses `einsum` + `index_select` instead of Python for-loops
- **Per-l weights**: Each degree l has an independent (C, C) transformation
- **Weight sharing**: Within each l-block, the same weight is shared across all 2l+1 m components
- **Bias only for l=0**: Only scalar components have additive bias (preserves equivariance for l>0)
- **expand_index buffer**: Precomputed mapping from packed (l,m) positions to l values

Compared to `PerDegreeLinear` (which uses `nn.ModuleList` with per-l `nn.Linear`), V2:

- Avoids Python for-loops
- Uses a single combined `weight` parameter instead of `lmax+1` separate weights
- Has identical numerical output (tested in unit tests)

### 4. Multi-layer SO(2) Convolution

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

Edge cache holds **only valid edges**:

- padding (`nlist == -1`) is removed
- excluded type pairs are removed
- edges with `r >= rcut` are removed

Let `E` be the number of valid edges:

- `src`, `dst`: `(E,)` flattened node indices in `[0, N)`
- `node_type_feat`: `(N, C)` type embeddings for nodes (used to derive per-edge type features by indexing)
- `edge_vec`: `(E, 3)` in Å
- `edge_rbf`: `(E, n_radial)` Bessel radial basis × C² envelope (trainable frequencies)
- `edge_sw`: `(E, 1)` DeePMD smooth weight flattened to valid edges
- `D_list[l]`: `(E, 2l+1, 2l+1)` real-basis Wigner-D block
- `Dt_list[l]`: transpose of `D_list[l]`
- `inv_sqrt_deg`: `(N, 1, 1)` inverse sqrt degree for graph-style normalization

---

## Core Operations

### Geometric Initial Embedding (GIE)

Purpose: Seed `l>0` features at layer 0 to reduce the number of blocks required.

Definition:

- `x(l=0)` comes **only** from type embedding.
- For `l>0`, compute per-`l` zonal seeds via the cached `Dt_list[l][:, :, m=0]` column
  (local->global) and a radial MLP.

### SO(2) Convolution (linearized)

For each edge `(src -> dst)`:

1. **Rotate to local frame**: apply `D_list[l]` per degree (l)
2. **Radial interaction**: radial MLP outputs per-`l` weights and modulates local features.
   The radial MLP takes edge invariants plus type priors as input:
   - `rad_in = concat(edge_rbf, src_type_feat, dst_type_feat)` with shape `(E, n_radial + 2C)`
   - `rad = radial_net(rad_in)` with shape `(E, (lmax+1) * C)`
3. **Strict smooth cutoff**: multiply by `edge_sw` (all terms vanish at `rcut`, envelope already baked into `edge_rbf`)
4. **Multi-layer SO(2) mixing**: for each layer in `so2_linears`:
   - Apply `SO2Linear` (group by `|m|`):
     - `m=0`: standard linear with additive bias (modulated by radial weights and cutoff to preserve strict smoothness on first layer)
     - `|m|>0`: 2x2 complex mixing on `(-m, +m)` pairs treated as `(Re, Im)`
   - Apply `GatedActivation` between layers (not after the last):
     - l=0: SiLU activation
     - l>0: sigmoid(l=0) gate
5. **Rotate back**: apply `Dt_list[l]`
6. **Aggregate with normalization**: scatter-sum by `dst`, then multiply by `inv_sqrt_deg`

### Full Equivariant FFN

The `EquivariantFFN` class implements:

```python
# Input projection (per-degree)
h = linear_in[l](x[:, l_slice, :])  # for each l

# GatedActivation with per-l independent gates
h0 = SiLU(h[:, 0:1, :])  # l=0: scalar activation
gating_scalars = sigmoid(gate_linear(h[:, 0, :]))  # (N, lmax * C)
gating_scalars = gating_scalars.view(N, lmax, C)
gates = index_select(gating_scalars, expand_index)  # (N, D-1, C)
ht = h[:, 1:, :] * gates  # gate l>0 features with per-l gates
h = torch.cat([h0, ht], dim=1)

# Output projection (per-degree)
out = linear_out[l](h[:, l_slice, :])  # for each l
```

Key properties:

- Bias only on l=0 (scalar) components
- Each `l` has an independent gate, expanded to all `m` within that `l`
- `expand_index` maps per-l gates to all `2l+1` m-components
- Residual connection: `x = x + ffn(x)`

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

## Public API and Hyperparameters

Constructor: `DescrptSeZMNet(...)`

Key arguments:

- `rcut: float` — Cutoff radius in Å
- `rcut_smth: float` — Smooth weight start in Å
- `sel: list[int] | int` — Maximum neighbors per type
- `lmax: int` — Maximum degree (only if `l_schedule` is None)
- `l_schedule: list[int] | None` — Pyramid schedule
- `channels: int` — Channels per (l,m) coefficient
- `n_radial: int` — Number of radial basis functions
- `radial_mlp: list[int]` — Hidden sizes for radial networks
- `n_blocks: int` — Number of blocks (only if `l_schedule` is None)
- `so2_layers: int` — Number of SO2Linear layers per convolution (default: 2)
- `ffn_neuron: list[int]` — Hidden sizes for equivariant FFN (first element used as hidden_channels)
- `use_parallel: bool` — If True, use the parallel operations, which requires more memory. For example, use block-diagonal parallel Wigner-D (`O(E * D^2)` memory, faster for small `E`)
- `exclude_types: list[tuple[int, int]]` — Excluded type pairs
- `precision: str` — `float64` / `float32`

Note: Neighbor normalization (graph-style degree normalization) is always enabled.

### Interface Compatibility Notes

SeZM-Net uses `_ENV_DIM = 1` (se_r style) for `EnvMatStatSe` compatibility. This means:

- `ndescrpt = nnei * 1` (only radial statistics are collected)
- The `env_protection` parameter is kept for interface compatibility but is not actively used
- `mean` and `stddev` statistics are maintained but not used in the forward pass (SeZM-Net uses radial basis functions directly instead of traditional env_mat)

### Multi-layer SO(2) Convolution

The `SO2Convolution` module supports multiple `SO2Linear` layers with intermediate `GatedActivation`:

- `so2_layers=1`: Single `SO2Linear` layer
- `so2_layers>=2`: `SO2Linear` -> (`GatedActivation` -> `SO2Linear`) x (so2_layers-1)

The intermediate activations are properly serialized/deserialized to ensure model state consistency.

Output:

- returns only `l=0` features as descriptor: `(nf, nloc, channels)`

---

## Serialization

`serialize()` captures:

- hyperparameters including `l_schedule`
- type embedding parameters
- **radial basis with trainable frequencies**
- block sub-networks:
  - `EquivariantFFN` (full equivariant FFN with separable gating: `gate_linear`)
  - `SO2Convolution` (radial net + SO2Linear)
  - `PerDegreeLinear`
- `davg` / `dstd` statistics buffers

`deserialize()` reconstructs the model and restores all parameters including trainable frequencies.

Version: `@version: 1`

---

## Physics & Numerics

### C² cutoff envelope

Envelope is quintic smoothstep to enforce smooth PES at `rcut`:

For `x = r / rcut`:

- `E(x) = 1 - 10 x^3 + 15 x^4 - 6 x^5` for `x in [0, 1]`
- `E(x) = 0` for `x >= 1`

The C² envelope is multiplied directly into the radial basis functions in `RadialBasis.forward()`.
All edge messages are then multiplied by the DeePMD smooth weight `edge_sw`, guaranteeing:

- message is 0 at `rcut`
- d(message)/dr is 0 at `rcut`
- d²(message)/dr² is 0 at `rcut`

### Conservative forces

- Edge rotations are computed from `edge_vec` without detach.
- Wigner-D blocks are computed from those rotations and remain differentiable.

### Wigner-D blocks (real SH basis)

SeZM-Net uses real-basis Wigner-D blocks to rotate per-degree features between the global frame
and the edge-aligned local frame.

#### Two implementations

Two implementations are provided in `deepmd/pt/model/descriptor/wigner_d.py`:

| Class                 | Description               | Memory                  | Speed                    |
| --------------------- | ------------------------- | ----------------------- | ------------------------ |
| `WignerDCalc`         | Per-l loop implementation | O(n_edges × max(2l+1)²) | Moderate                 |
| `WignerDCalcParallel` | Block-diagonal parallel   | O(n_edges × dim_full²)  | Faster for small n_edges |

where `dim_full = (lmax+1)² = ebed_dim`.

**Recommendation**:

- Use `WignerDCalc` (default) for large `n_edges` to avoid memory issues.
- Use `WignerDCalcParallel` for small `n_edges` where dispatch overhead dominates.

#### Conventions

- `rot_mat` is a global->local transform for 3D vectors:
  - `v_local = rot_mat @ v_global`
  - It is built by either `init_edge_rot_mat(edge_vec)` (Gram-Schmidt with a reference-axis switch) or `init_edge_rot_mat_frisvad(edge_vec)` (Frisvad ONB with a strict cross-product fallback near `-Z`) so that `rot_mat @ (edge_vec / ||edge_vec||) = (0, 0, 1)`.
- For each degree `l`, real SH channels are ordered by `m=-l..+l` (index `i = m + l`).
- `D_list[l]` / `Dt_list[l]` are real-basis Wigner-D blocks:
  - `D_list[l]`: `(E, 2l+1, 2l+1)` and represents the same global->local rotation as `rot_mat`.
  - `Dt_list[l] = D_list[l]^T`: inverse blocks (local->global).

#### Key identities

We use the ZYZ decomposition:

- `R = Rz(alpha) @ Ry(beta) @ Rz(gamma)`

In the complex spherical harmonics basis, the standard definition is:

- `D^{(l)}_{m1,m2}(alpha,beta,gamma) = exp(-i*m1*alpha) * d^{(l)}_{m1,m2}(beta) * exp(-i*m2*gamma)`

In the real (tesseral) basis, z-axis rotations become cheap 2x2 `cos/sin` blocks. To avoid explicitly
building the dense y-axis rotation `d^{(l)}(beta)`, we use the conjugation identity:

- `Ry(beta) = Rx(pi/2)^{-1} @ Rz(beta) @ Rx(pi/2)`

Define `J_l = D^{(l)}(Rx(pi/2))` in the real basis. Since the real-basis representation is orthogonal:

- `D_x^{(l)}(-pi/2) = J_l^T`

Then the per-degree block is:

- `D^{(l)}(R) = Z^{(l)}(alpha) @ J_l^T @ Z^{(l)}(beta) @ J_l @ Z^{(l)}(gamma)`
- `D^{(l)}(R)^{-1} = D^{(l)}(R)^T`

#### Block-diagonal parallel computation (`WignerDCalcParallel`)

Instead of computing each `l` separately (which incurs `lmax+1` matmul chains), `WignerDCalcParallel`
assembles all blocks into a single block-diagonal matrix of dimension `dim_full = (lmax+1)^2`:

```
J_full = diag(J_0, J_1, ..., J_lmax)
Z_full(theta) = diag(Z^{(0)}(theta), Z^{(1)}(theta), ..., Z^{(lmax)}(theta))
```

The full computation becomes a single batched matmul chain:

```
D_full = Z_full(alpha) @ J_full^T @ Z_full(beta) @ J_full @ Z_full(gamma)
```

This reduces `lmax+1` separate matmul chains to **one chain** on larger matrices, minimizing Python
dispatch overhead (significant when `lmax <= 10`).

**Index layout**:

- Block `l` occupies rows/columns `[l^2, (l+1)^2)` in the full matrix.
- Within block `l`, the `m=0` element is at position `l^2 + l` (center of block).
- For `m > 0`, the 2×2 rotation sub-block occupies:
  - `pos = l^2 + (l + m)` for `+m`
  - `neg = l^2 + (l - m)` for `-m`

**Precomputed buffers**:

| Buffer         | Shape                  | Description                                   |
| -------------- | ---------------------- | --------------------------------------------- |
| `_J_full`      | `(dim_full, dim_full)` | Block-diagonal `J` matrix                     |
| `_Jt_full`     | `(dim_full, dim_full)` | Transpose of `_J_full`                        |
| `_m0_indices`  | `(lmax+1,)`            | Global indices for `m=0` elements             |
| `_pos_indices` | `(n_blocks,)`          | Global indices for `+m` positions             |
| `_neg_indices` | `(n_blocks,)`          | Global indices for `-m` positions             |
| `_m_values`    | `(n_blocks,)`          | Corresponding `m` values for trig computation |

**Z matrix construction** (`_build_z_rotation`):

1. Initialize zeros `(n_edges, dim_full, dim_full)`.
2. Set `m=0` diagonal elements to 1 using `_m0_indices`.
3. Compute `cos(m*theta)` and `sin(m*theta)` for all `(l, m)` pairs in one vectorized op.
4. Fill 2×2 rotation blocks using precomputed indices:
   ```
   Z[:, pos, pos] =  cos(m*theta)
   Z[:, neg, neg] =  cos(m*theta)
   Z[:, pos, neg] =  sin(m*theta)
   Z[:, neg, pos] = -sin(m*theta)
   ```

#### Implementation mapping

**Common to both implementations**:

- Euler extraction: `_extract_zyz_euler(rot_mat)`
  - Uses stable matrix-entry formulas.
  - Singular cases `beta -> 0` / `beta -> pi` use `gamma = 0` and fold the residual z-rotation into `alpha`.
- Constant `J_l`: `_compute_j_matrix(l)`
  - Built once per `l` in float64 on CPU.
  - Uses the standard closed-form sum `d^{(l)}_{m1,m2}(beta)` for this precomputation.

**`WignerDCalc` (per-l loop)**:

- Z rotation: `_build_z_rotation(angle, l)` builds `(n_edges, 2l+1, 2l+1)` per block.
- J matrices stored as `_J_{l}` and `_Jt_{l}` for each `l`.

**`WignerDCalcParallel` (block-diagonal)**:

- Z rotation: `_build_z_rotation(angle)` builds full `(n_edges, dim_full, dim_full)`.
- J matrices stored as `_J_full` and `_Jt_full` block-diagonal.
- Index precomputation: `_precompute_z_indices()` builds `_m0_indices`, `_pos_indices`, `_neg_indices`, `_m_values`.

#### Usage in message passing

SeZM blocks apply the cached rotations as:

- rotate to local: `x_local^{(l)} = D_list[l] @ x_global^{(l)}`
- rotate back: `x_global^{(l)} = Dt_list[l] @ x_local^{(l)}`

### Padded neighbor safety

- Padding edges are removed before any normalization or angle computation.
- `PairExcludeMask` returns a **keep mask** (1=keep, 0=excluded). It does not remove padding by itself, so always combine it with `nlist >= 0`.
- No zero-length vector is normalized for padding edges.

### Precision and device handling

- All submodules use `dtype: torch.dtype` (not `precision: str`) for constructor parameter.
- Device is obtained from global `env.DEVICE` at runtime; submodules store `self.device = env.DEVICE` only as a convenience reference, not for serialization.
- Each submodule stores `self.precision = RESERVED_PRECISION_DICT[dtype]` for serialization compatibility.
- The `safe_norm` function automatically infers epsilon from input dtype instead of accepting an external `eps` parameter.

---

## Caching Strategy (Critical)

**EdgeFeatureCache is built exactly once per `forward()`**.

Why caching matters:

- The expensive part is edge geometry + Wigner rotations.
- Message passing blocks reuse the same per-edge rotations and radial features.
- This avoids an O(#blocks) multiplier on per-edge trig/matrix ops.

What is cached:

- All per-edge tensors needed by all blocks (geometry, radial basis, envelope, Wigner-D blocks).

What is not cached:

- Block-specific radial MLP outputs (depend on block parameters).

---

## DeePMD Interface Compatibility

SeZM-Net follows the **new-style descriptor interface** (same as `dpa3`), using `extended_coord` / `extended_atype` parameter names (instead of `coord_ext` / `atype_ext` used by older descriptors like `se_a` and `se_r`).

### Implemented abstract methods

All required `BaseDescriptor` abstract methods are implemented:

- `get_rcut()` / `get_rcut_smth()` — Return cutoff radii
- `get_sel()` / `get_ntypes()` / `get_nsel()` / `get_nnei()` — Return neighbor selection info
- `get_type_map()` — Return type name mapping
- `get_dim_out()` / `get_dim_emb()` — Return output dimensions
- `mixed_types()` — Returns `False` (uses type-distinguished neighbor list)
- `has_message_passing()` — Returns `True` (uses edge message passing)
- `need_sorted_nlist_for_lower()` — Returns `False`
- `get_env_protection()` — Return protection parameter
- `set_stat_mean_and_stddev()` / `get_stat_mean_and_stddev()` — Statistics accessors
- `forward()` — Core descriptor computation
- `serialize()` / `deserialize()` — Model persistence
- `update_sel()` — Classmethod for neighbor statistics
- `compute_input_stats()` — Compute env_mat statistics via `EnvMatStatSe`
- `reinit_exclude()` — Update excluded type pairs
- `__setitem__()` / `__getitem__()` — Dict-like access for `davg`/`dstd`

### Not implemented (raises NotImplementedError)

- `share_params()` — Multi-task parameter sharing not implemented yet
- `change_type_map()` — Type map changing not implemented yet

### Interface notes

1. **`_ENV_DIM = 1`** — Uses `se_r` style (radial only) for `EnvMatStatSe` compatibility
2. **Precision handling** — Submodules accept `dtype: torch.dtype` parameter and store both `self.dtype` and `self.precision = RESERVED_PRECISION_DICT[dtype]` for serialization
3. **Statistics buffers** — `mean`/`stddev` are maintained for interface compatibility but not actively used in forward (SeZM-Net uses radial basis functions directly)
