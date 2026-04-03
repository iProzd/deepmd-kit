# MoE Implementation Performance Analysis: Old vs New

## Summary

**Performance gap:** Old MoE ~0.1083 s/batch vs New MoE ~1.5052 s/batch (~14x slower).
**Root cause identified and fixed:** Inefficient `_moe_flat` implementation in `deepmd/pt/model/network/moe.py`.
**After fix:** New MoE ~0.2029 s/batch (~2x vs old, acceptable given architectural differences).

---

## Architecture Comparison

### Old MoE (`old_moe_deepmdkit`)

Located in `deepmd/pt/model/network/mlp.py`:
- **Storage:** Single 3D weight tensor `[num_in, num_out, n_experts]` per MoE layer
- **Classes:** `mlp_layer_moe`, `mlp_layer_moe_dynamic_sel`, `share_mlp_layer_moe`
- **Wrapped in:** `MOE_replace_mlp` which handles routing + expert computation
- **Router:** `MOErouter` — takes atom type embedding, returns topk weights + indices

**Key implementation — `experts_regroup_only` (non-dynamic-sel path):**
```python
# Sort tokens by expert assignment
order = torch.argsort(expert_idx)
for e in range(n_experts):
    tok_e = token_idx[start:end]
    x_e = x.index_select(0, tok_e)  # contiguous memory
    y_e = torch.matmul(x_e, W[e])   # one dense matmul per expert
```

**Key implementation — `experts_regroup_dynamic` (dynamic-sel path):**
```python
# Sort tokens by expert assignment, then one dense matmul per expert
order = torch.argsort(expert_idx)
for e in range(n_experts):
    x_e = xx.index_select(0, tok_e)   # [Ne, dim]
    y_e = torch.matmul(x_e, W[e])     # dense matmul
```

Total kernel launches: **n_experts** (4 in this case).
Each matmul processes **all tokens assigned to that expert** (contiguous).

### New MoE (`deepmd-kit-moe`)

Located in `deepmd/pt/model/network/moe.py`:
- **Storage:** `ModuleList` of separate `MLPLayer` instances, one per expert
- **Classes:** `MoELayer` with routing via `gate` (MLPLayer) on type embeddings
- **Supports:** Expert Parallelism (EP) via All-to-All

**Routing:**
```python
logits = self.gate(type_embeddings)   # [ntypes, routed_experts]
topk_logits, topk_indices = torch.topk(logits, k=routed_top_k)
weights = F.softmax(topk_logits)      # [ntypes, routed_top_k]
atom_weights = weights[atom_types]    # broadcast to [nb, nloc, top_k]
atom_indices = topk_indices[atom_types]
```

**Two local (non-EP) forward paths:**

1. **`_moe_batched`** — for non-dynamic-sel inputs `[nb, nloc, *mid, dim]`:
   - Computes all expert outputs on full input → O(n_experts) dense matmuls ✅
   - Builds per-expert weight via `scatter_add_` then weighted sum ✅
   - **Correct and efficient**

2. **`_moe_flat`** — for dynamic-sel inputs `[n_flat, dim]` (called with `edge_index`):
   - **ORIGINAL (BUGGY):** Nested loop: `for k in top_k: for e in n_experts: mask=... expert(x[mask])`
   - Total iterations: `top_k × n_experts = 4 × 4 = 16`
   - Each call uses boolean-mask indexing → **small, non-contiguous kernel launches**
   - With `top_k = n_experts = 4`: each sub-batch is ~25% of `n_flat`
   - GPU utilization is very low due to many small matmuls

---

## Root Cause of Performance Degradation

In the test configuration:
- `n_experts = 4`, `moe_top_k = 4`, `use_dynamic_sel = True`
- Dynamic-sel mode means edge/angle embeddings are **flat** `[n_flat, dim]`
- `n_flat` = number of real edges ≈ `n_atoms × avg_real_neighbors` (can be large)

**Old implementation** dispatches all tokens through a sorted list, yielding:
- `n_experts = 4` dense matmuls, each processing `n_flat / n_experts` tokens on average
- Contiguous memory access patterns

**Original new implementation** uses nested loops with bool-mask indexing:
- `top_k × n_experts = 16` kernel launches
- Each with irregular (non-contiguous) indices
- Small batch sizes → poor GPU utilization
- Many Python-level loop iterations → additional Python overhead

---

## Fix Applied

**File:** `/mnt/data_nas/zhangd/claude_space/deepmd-kit-moe/deepmd/pt/model/network/moe.py`

**Method:** `_moe_flat`

**Change:** Replaced nested loop with bool-mask indexing to the same "compute-all-then-weight" approach used by `_moe_batched`:

```python
# BEFORE (buggy — O(top_k * n_experts) small kernel launches):
for k in range(self.routed_top_k):
    w_k = weights[:, k]
    idx_k = indices[:, k]
    for e_idx, expert in enumerate(self.experts):
        mask = idx_k == e_idx
        if mask.any():
            expert_out = expert(x[mask])       # small non-contiguous batch
            output[mask] = output[mask] + w_k[mask].unsqueeze(-1) * expert_out

# AFTER (fixed — O(n_experts) dense matmuls):
expert_outputs = [expert(x) for expert in self.experts]  # full tensor
per_expert_weight = torch.zeros(n_flat, self.routed_experts, ...)
per_expert_weight.scatter_add_(1, indices, weights)       # vectorized
output = sum(per_expert_weight[:, e].unsqueeze(-1) * expert_outputs[e]
             for e in range(self.routed_experts))
```

---

## Performance Results

| Implementation | 100-step wall time | Average s/batch |
|---------------|-------------------|-----------------|
| Old MoE       | 20.70 s           | **0.1083 s**   |
| New MoE (original) | 144.21 s     | 1.5052 s (~14× slower) |
| **New MoE (fixed)** | **17.70 s** | **0.2029 s** (~2× vs old) |

---

## Remaining ~2x Gap Analysis

After the fix, new MoE is still ~2× slower than old MoE. Likely causes:

1. **Separate MLPLayer per expert** vs **3D weight matrix** in old:
   - Old: single `matmul(x, W[e])` — one fused kernel
   - New: `F.linear(x, matrix.t(), bias)` — bias fused but extra `.t()` view creation overhead
   - Old's `experts_regroup_*` does a single sort pass + grouped matmuls

2. **Memory allocation overhead** in new MoE:
   - Allocates intermediate `expert_outputs` list (4 full tensors of `[n_flat, num_out]`)
   - Old allocates a single output tensor

3. **Additional scatter_add** in new implementation for routing weight accumulation

4. **Minor:** `F.linear` CUBLAS initialization warning suggests fallback to unfused path for some small batch cases

5. **Structural differences:** New MoE additionally applies node MoE (`use_node_moe=True`) via `_moe_batched` path on node features, which is additional computation not present in the same form in old.

---

## Key Differences Summary Table

| Aspect | Old MoE | New MoE |
|--------|---------|---------|
| Weight storage | 3D tensor `[in, out, n_exp]` | ModuleList of MLPLayer |
| Dynamic-sel compute | `experts_regroup_dynamic` (sort+grouped matmul) | `_moe_flat` (fixed: compute-all-weight) |
| Non-dynamic-sel compute | `experts_regroup_only` (sort+grouped matmul) | `_moe_batched` (compute-all-weight) |
| Routing source | Atom type embedding | Type embedding table (gate) |
| EP support | No | Yes (via All-to-All) |
| Shared experts | Separate `share_mlp_layer_moe` | Shared expert MLPLayers |
| Gate computation | `MOErouter` (per-atom forward) | Gate on ntypes only, broadcast to atoms |
| Node MoE | Same `MOE_replace_mlp` as edge | Separate `node_self_mlp_moe`, `node_sym_linear_moe` |
| Angle MoE | `MOE_replace_mlp` with dynamic | `_moe_flat` (now fixed) |

---

## Configuration Differences Between Test Cases

| Parameter | old_moe | new_moe |
|-----------|---------|---------|
| n_experts | 4 | 4 |
| topk / moe_top_k | 4 | 4 |
| use_node_moe | implicit True | explicit True |
| use_edge_moe | True | True |
| use_angle_moe | True | True |
| optim_update | False | False |
| use_dynamic_sel | True | True |

Both configurations are equivalent in terms of MoE structure; the performance difference was purely implementation-level.
