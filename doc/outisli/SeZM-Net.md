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
     - `edge_vec / edge_len / edge_unit`
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
       ├─ edges: (src, dst), edge_vec, edge_len, edge_unit
       ├─ edge_envelop: C² envelope
       ├─ edge_sw: DeePMD smooth weights in flattened edge layout
       ├─ edge_rbf: Bessel radial basis (trainable frequencies)
       ├─ D_list[l], Dt_list[l]: Wigner-D blocks per l
       └─ sw: DeePMD smooth weights for (nf, nloc, nnei, 1)

Node init:
  ├─ l=0: Type embedding
  └─ l>0: Zonal (m=0) initial embedding via cached Wigner-D + radial MLP

Interaction blocks (pyramid schedule):
  for block i:
    ├─ slice x to K(l_schedule[i]) (discard higher-l if needed)
    ├─ SeparableRMSNorm (scalar / tensor separated)
    ├─ SO(2) Convolution (enabled for ALL lmax, including lmax=0)
    ├─ Residual
    ├─ AnalyticGating (gate from l=0)
    └─ Full Equivariant FFN (operates on ALL orders l=0..lmax, separable gating)

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

### 2. Full Equivariant FFN (All Orders, Separable Gating)

The FFN operates on **ALL spherical harmonic orders** (l=0 to lmax), not just scalars. It uses a NequIP/eSEN-style gated FFN with separable gating:

```
EquivariantFFN:
  1. Per-order linear mixing (C → hidden)
  2. Separable gating:
     - l=0 content branch: SiLU applied to scalar hidden features
     - l=0 gate branch: independent projection from scalar inputs produces sigmoid-activated gates
  3. Per-order linear mixing (hidden → C)
  4. Residual connection
```

Separable gating avoids coupling gate gradients with the scalar content pathway while still gating all higher-order features (l>0).

---

## Tensor Layouts and Invariants

### Node features `x`

The core tensor is:

- `x`: `torch.Tensor` with shape `(N, K, C)`
  - `N = nf * nloc`
  - `C = channels`
  - `K = (lmax + 1)^2 = sum_{l=0..lmax} (2l + 1)`

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
- `edge_vec`: `(E, 3)` in Å
- `edge_len`: `(E, 1)` in Å
- `edge_unit`: `(E, 3)` unit vector
- `edge_envelop`: `(E, 1)` C² envelope
- `edge_sw`: `(E, 1)` DeePMD smooth weight flattened to valid edges
- `edge_rbf`: `(E, n_radial)` raw Bessel basis (no envelope baked in)
- `D_list[l]`: `(E, 2l+1, 2l+1)` real-basis Wigner-D block
- `Dt_list[l]`: transpose of `D_list[l]`

---

## Core Operations

### Geometric Initial Embedding (GIE)

Purpose: Seed `l>0` features at layer 0 to reduce the number of blocks required.

Definition:

- `x(l=0)` comes **only** from type embedding.
- For `l>0`, compute per-`l` zonal seeds via cached Wigner-D columns and radial MLP.

### SO(2) Convolution (linearized)

For each edge `(src -> dst)`:

1. **Rotate to local frame**: apply `Dt_list[l]` per order (l)
2. **Radial interaction**: radial MLP outputs per-`l` weights and modulates local features
3. **Strict smooth cutoff**: multiply by `edge_envelop * edge_sw` (all terms vanish at `rcut`)
4. **SO(2) mixing**: group by `|m|`, shared linear maps for `+m` and `-m`
   - Any additive scalar bias is also modulated by radial weights and cutoff to preserve strict smoothness.
5. **Rotate back**: apply `D_list[l]`
6. **Aggregate**: scatter-sum by `dst`

### Full Equivariant FFN

The `EquivariantFFN` class implements:

```python
# Input projection (per-order)
h = linear_in[l](x[:, l_slice, :])  # for each l

# Separable gating
h0 = SiLU(h[:, 0:1, :])  # scalar activation
gate_input = x[:, 0, :]  # scalar inputs
gates = sigmoid(gate_linear(gate_input))  # gates from scalar branch
ht = h[:, 1:, :] * gates_expanded  # gate l>0 features
h = torch.cat([h0, ht], dim=1)

# Output projection (per-order)
out = linear_out[l](h[:, l_slice, :])  # for each l
```

Key properties:

- Bias only on l=0 (scalar) components
- Gates are shared across `m` within each `l`
- Residual connection: `x = x + ffn(x)`

---

## Pyramid `l_schedule`

SeZM-Net supports:

1. constant `lmax` (legacy style): `l_schedule = [lmax] * n_blocks`
2. explicit pyramid: `l_schedule = [2, 2, 1, 0]` (example)

Rules:

- `l_schedule` must be **non-increasing**
- final entry must be **0** to fully drop higher-order components
- when schedule decreases, higher-`l` channels are **physically discarded**
- later blocks operate on smaller `K`, reducing compute

---

## Public API and Hyperparameters

Constructor: `DescrptSeZMNet(...)`

Key arguments:

- `rcut: float` — Cutoff radius in Å
- `rcut_smth: float` — Smooth weight start in Å
- `sel: list[int] | int` — Maximum neighbors per type
- `lmax: int` — Maximum order (only if `l_schedule` is None)
- `l_schedule: list[int] | None` — Pyramid schedule
- `channels: int` — Channels per (l,m) coefficient
- `n_radial: int` — Number of radial basis functions
- `radial_mlp: list[int]` — Hidden sizes for radial networks
- `n_blocks: int` — Number of blocks (only if `l_schedule` is None)
- `ffn_neuron: list[int]` — Hidden sizes for equivariant FFN (first element used as hidden_channels)
- `neighbor_norm: bool` — Normalize by inverse sqrt node degree
- `wigner_parallel: bool` — If True, use block-diagonal parallel Wigner-D (`O(E * K^2)` memory, faster for small `E`)
- `exclude_types: list[tuple[int, int]]` — Excluded type pairs
- `precision: str` — `float64` / `float32`

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
  - `AnalyticGating`
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

All edge messages are multiplied by `edge_envelop` and the DeePMD smooth weight `edge_sw`, guaranteeing:

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

| Class                 | Description               | Memory                 | Speed                   |
| --------------------- | ------------------------- | ---------------------- | ----------------------- |
| `WignerDCalc`         | Per-l loop implementation | O(n_edge × max(2l+1)²) | Moderate                |
| `WignerDCalcParallel` | Block-diagonal parallel   | O(n_edge × dim_full²)  | Faster for small n_edge |

where `dim_full = (lmax+1)²`.

**Recommendation**:

- Use `WignerDCalc` (default) for large `n_edge` to avoid memory issues.
- Use `WignerDCalcParallel` for small `n_edge` where dispatch overhead dominates.

#### Conventions

- `rot_mat` is a global->local transform for 3D vectors:
  - `v_local = rot_mat @ v_global`
  - It is built by `init_edge_rot_mat(edge_vec)` so that `rot_mat @ edge_unit = (0, 0, 1)`.
- For each degree `l`, real SH channels are ordered by `m=-l..+l` (index `i = m + l`).
- `D_list[l]` / `Dt_list[l]` are real-basis Wigner-D blocks:
  - `D_list[l]`: `(E, 2l+1, 2l+1)`
  - `Dt_list[l] = D_list[l]^T`: inverse blocks (orthogonal matrices).

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

1. Initialize zeros `(n_edge, dim_full, dim_full)`.
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

- Z rotation: `_build_z_rotation(angle, l)` builds `(n_edge, 2l+1, 2l+1)` per block.
- J matrices stored as `_J_{l}` and `_Jt_{l}` for each `l`.

**`WignerDCalcParallel` (block-diagonal)**:

- Z rotation: `_build_z_rotation(angle)` builds full `(n_edge, dim_full, dim_full)`.
- J matrices stored as `_J_full` and `_Jt_full` block-diagonal.
- Index precomputation: `_precompute_z_indices()` builds `_m0_indices`, `_pos_indices`, `_neg_indices`, `_m_values`.

#### Usage in message passing

SeZM blocks apply the cached rotations as:

- rotate to local: `x_local^{(l)} = Dt_list[l] @ x_global^{(l)}`
- rotate back: `x_global^{(l)} = D_list[l] @ x_local^{(l)}`

### Padded neighbor safety

- Padding edges are removed before any normalization or angle computation.
- `PairExcludeMask` returns a **keep mask** (1=keep, 0=excluded). It does not remove padding by itself, so always combine it with `nlist >= 0`.
- No zero-length vector is normalized for padding edges.

### Precision and device handling

- All submodules use `dtype: torch.dtype` (not `precision: str`) for constructor parameter.
- Device is obtained from global `env.DEVICE` at runtime; submodules store `self.device = env.DEVICE` only as a convenience reference, not for serialization.
- Each submodule stores `self.precision = RESERVED_PRECISION_DICT[dtype]` for serialization compatibility.
- The `_safe_norm` function automatically infers epsilon from input dtype instead of accepting an external `eps` parameter.

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
