---
name: sezm-moe-design
description: Task-specific design patterns for SeZM MoE. Covers the F = topk + n_shared_experts invariant, the shared-expert-as-alpha-1-slot mechanism, routing-input options (dst/src/src+dst), per-expert bias_correction packing, and the v1 simplifications. Read before implementing any Step 2+.
---

# SeZM MoE design patterns

## The single most important invariant

```
n_focus  ==  topk  +  n_shared_experts
   F            t            S
```

Every parameter in the system is sized around this. Violating it must raise immediately at `SO2Convolution.__init__`.

## How F = topk + S maps to the data flow

```
x_local: (E, F, D_m, Cf)
                 ↑ F slots, semantically:
                 ├── slot 0..topk-1: routed via Router (each goes to one of topk experts)
                 └── slot topk..topk+S-1: handled by shared experts (always-on)
```

**Routing slots** travel through the A2A:

- For each edge, slot `t` (t < topk) carries the input for the t-th selected expert
- After A2A round-trip, slot `t` returns with that expert's output

**Shared slots** stay local:

- For each edge, slot `topk + s` is processed by `SharedExpert_s` (which lives on every GPU)
- No A2A, no inter-rank communication

After all experts have run, both groups produce `(E, D_m, Cf)` outputs that get concatenated back into the F dimension:

```
full = cat([routing_out (E, topk, D_m, Cf), shared_out (E, S, D_m, Cf)], dim=1)
     = (E, F, D_m, Cf)
```

## Replacing cross-focus competition with routing alpha

In the original SeZM code, Step 5.5 does:

```python
focus_logits = einsum("efi,if->ef", normed_x_l0, focus_compete_w)
alpha = softmax(focus_logits / tau, dim=1) * (1 - eps) + eps / F
x_local = x_local * alpha[:, :, None, None]
```

In MoE mode, the same multiplicative weighting happens with a different alpha source:

```python
alpha_routing = topk_weights  # (E, topk), already softmax-normalized by router
alpha_shared = torch.ones(E, S, device=...)  # always-on
alpha = torch.cat([alpha_routing, alpha_shared], dim=-1)  # (E, F)
x_local = full * alpha[:, :, None, None]  # element-wise weighting, F preserved
```

**Critical**: alpha *weights* the F dimension; **it does not reduce it**. The output is `(E, F, D_m, Cf)` ready for Step 6 (rotate-back), which folds F into the channel dim:

```python
x_local.transpose(1, 2).reshape(E, D_m, F * Cf)  # F concat'd into channels
```

So the F dimension survives all the way to rotate-back, just as in the non-MoE path.

## Why shared experts have alpha = 1 (not normalized)

Two options were considered:

| Option | Description                                         | Behavior                                                                                |
| ------ | --------------------------------------------------- | --------------------------------------------------------------------------------------- |
| A      | shared has fixed alpha = 1, routing alphas sum to 1 | shared experts are "always on" at full magnitude, routing experts share a "budget of 1" |
| B      | shared participates in a joint softmax of size F    | shared and routing compete; each can be down-weighted                                   |

**v1 uses option A**. Rationale:

- Matches the user's mental model ("alpha=1 always-on expert")
- Lets shared experts genuinely act as a "baseline" that the routing experts modulate
- Avoids extra softmax machinery

## Routing input options

`routing_input` selects what's fed into `MoESO2Router.gate`:

| Option            | What                                                | Use case                                                                      |
| ----------------- | --------------------------------------------------- | ----------------------------------------------------------------------------- |
| `"dst"` (default) | type_embedding indexed by `edge.dst`                | Receiver-driven routing; intuitive analogy to attention query coming from dst |
| `"src"`           | type_embedding indexed by `edge.src`                | Sender-driven routing; the source decides where its messages go               |
| `"src+dst"`       | concat(type_emb[src], type_emb[dst]) along channels | Pair-aware routing; finer-grained but doubles router input dim                |

Implementation in `SO2Convolution`:

```python
if self.routing_input == "dst":
    type_emb_per_edge = type_embedding.index_select(0, dst)  # (E, C_type)
elif self.routing_input == "src":
    type_emb_per_edge = type_embedding.index_select(0, src)
elif self.routing_input == "src+dst":
    type_emb_per_edge = torch.cat(
        [
            type_embedding.index_select(0, src),
            type_embedding.index_select(0, dst),
        ],
        dim=-1,
    )  # (E, 2*C_type)
else:
    raise ValueError(...)
```

The router's `input_dim` adjusts:

- `dst` / `src`: `input_dim = C_type`
- `src+dst`: `input_dim = 2 * C_type`

## Each block has its own router (no parameter sharing)

```python
# In SeZMInteractionBlock or DescrptSeZM:
self.blocks = nn.ModuleList(
    [
        SeZMInteractionBlock(..., moe_config=moe_config_for_block_i)
        for i in range(n_blocks)
    ]
)
```

Each `SO2Convolution` (one per block) gets its own `MoESO2Router` instance with independent `gate` parameters. This is intentional — different blocks may want to route based on different aspects of the type embedding.

Routers are **non-routing-expert parameters** (their `name` does not contain `.routing_matrix`), so they are world-synced normally.

## Expert internals: `n_focus = 1` per expert

Each routing/shared expert is a complete 4-layer SO2Linear stack with `n_focus = 1` internally. The outer multi-focus structure (F = topk + S) is realized by:

- Having `n_routing_experts` total routing experts available
- Routing each edge to `topk` of them (their outputs fill slots 0..topk-1)
- Having S shared experts (their outputs fill slots topk..F-1)

This means:

- `SO2Linear` inside expert has `weight_m0 shape (num_l * Cin, num_l * Cout)` — no F factor on output
- The 3D shared tensor storage:
  ```python
  routing_matrix_layer{i}_m0: (n_per_gpu, num_l*Cin, num_l*Cout)
  routing_matrix_layer{i}_m{m}: (n_per_gpu, (lmax-m+1)*Cin, 2*(lmax-m+1)*Cout)
  routing_bias_layer0:          (n_per_gpu, num_out)   # only layer 0 has bias by convention
  ```

## v1 simplifications (all `raise` if violated)

| Feature                                  | v1 forced value                               | Future work                                                     |
| ---------------------------------------- | --------------------------------------------- | --------------------------------------------------------------- |
| `use_compile`                            | `False`                                       | Investigate make_fx tracing of `_AllToAllDouble`                |
| `so2_norm`                               | `False`                                       | Decide whether per-expert or shared `ReducedEquivariantRMSNorm` |
| `use_so2_attn_res`                       | `"none"`                                      | Depth-attention with MoE outputs is non-trivial                 |
| `layer_scale` (when `True`)              | shared shape `(Cf,)` instead of `(F, Cf)`     | Per-expert layer_scale if motivated                             |
| `bias_correction` (when `mlp_bias=True`) | per-expert bias, A2A radial factor with token | Already correct in v1; just verify carefully                    |

## `bias_correction` packing detail (when `mlp_bias=True`)

Original code:

```python
bias_correction = bias0 * (radial_factor * edge_env - 1.0)
x_local[:, :, 0, :] += bias_correction
```

In MoE, `bias0` is **per-expert** (on the remote rank), while `radial_factor` and `edge_env` are **per-edge** (on the source rank). The fix:

1. **Source rank**: compute `multiplier = radial_factor * edge_env - 1.0` for each routing slot, shape `(E, topk, Cf)`.
1. After sort, the multiplier has shape `(E*topk, Cf)`.
1. A2A the multiplier alongside the tokens using the same `send_splits`. (Two separate `_AllToAllDouble.apply` calls with same splits, or pack them — implementer's choice.)
1. **Remote rank**: after `SO2Linear_layer0`, before activation:
   ```python
   y[:, 0, :] += bias0_local_e * received_multiplier_chunk
   ```

In v1 with `mlp_bias=False` (SeZM default), this entire branch is skipped, simplifying the A2A to a single token tensor.

## Single-GPU path (`ep_group=None`)

When EP is disabled, the multi-GPU machinery degenerates:

```python
def _forward_single_gpu(self, x_local, ...):
    topk_w, topk_idx = self.router(routing_key)

    # Routing experts: just for-loop locally (no A2A)
    r_input = x_local[:, :self.topk, :, :].reshape(E*topk, D_m, Cf)
    r_eids = topk_idx.reshape(E*topk)
    sort_order = torch.argsort(r_eids, stable=True)
    # bincount, per-expert chunked matmul (use the 3D tensor as if it were
    # a "local view" of all experts since n_per_gpu == n_routing_experts)
    r_out = ...

    # Shared experts: direct compute
    sh_input = x_local[:, self.topk:, :, :]   # (E, S, D_m, Cf)
    sh_out = self.experts.forward_shared(sh_input)

    full = cat([r_out_reshaped, sh_out], dim=1)
    alpha = cat([topk_w, ones_like(...)], dim=-1)
    return full * alpha[:, :, None, None]
```

**Target overhead: < 15% vs no-MoE baseline.** Optimization tip: keep the topk=1, S=0, n=1 trivial-case path fast — it should reduce to (almost) a single matmul.

## Naming for `sync_moe_gradients` to dispatch correctly

All routing-expert parameter `name` paths must contain `.routing_matrix` or `.routing_bias`:

```
descriptor.blocks.0.so2_conv.moe.experts.routing_matrix_layer0_m0      ✓ routing
descriptor.blocks.0.so2_conv.moe.experts.routing_bias_layer0           ✓ routing
descriptor.blocks.0.so2_conv.moe.experts.shared_experts.0.layer0.weight_m0  ✗ not routing (synced world-wise)
descriptor.blocks.0.so2_conv.moe.router.gate.matrix                    ✗ not routing
descriptor.blocks.0.so2_conv.adam_so2_layer_scales.0                   ✗ not routing
```

`_is_routing_expert_param(name)` returns True iff `".routing_matrix" in name or ".routing_bias" in name`.

## Quick reference: file/class names

| Concept                                      | Location                                                                       |
| -------------------------------------------- | ------------------------------------------------------------------------------ |
| A2A op                                       | `deepmd/pt/model/descriptor/sezm_nn/moe/a2a_ops.py`                            |
| Router                                       | `deepmd/pt/model/descriptor/sezm_nn/moe/router.py` (`MoESO2Router`)            |
| Expert collection                            | `deepmd/pt/model/descriptor/sezm_nn/moe/experts.py` (`MoESO2ExpertCollection`) |
| MoE convolution (replaces inner SO(2) stack) | `deepmd/pt/model/descriptor/sezm_nn/moe/conv.py` (`MoESO2Convolution`)         |
| Process groups + sync                        | `deepmd/pt/utils/sezm_moe_ep_dp.py`                                            |
| Checkpoint resharding                        | `deepmd/pt/utils/sezm_moe_checkpoint.py`                                       |
