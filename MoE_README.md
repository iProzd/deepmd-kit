# MoE (Mixture of Experts) Support for DPA3 RepFlowLayer

## Overview

This implementation adds optional Mixture of Experts (MoE) support to the DPA3 descriptor's RepFlowLayer. When `n_experts > 1` in the repflow config, selected MLPLayers are replaced with MoE layers where multiple expert linear transforms are weighted by a type-embedding-based router.

### Key Design Decisions

**Type-based routing (not input-based):** Unlike LLM MoE where the router uses input features, our router uses type embeddings to compute expert weights. Since there are only `ntypes` unique type embeddings (e.g., 2 for water), the router computes a `[ntypes, n_experts]` weight matrix once, then indexes by atom type. This ensures MD continuity — atoms of the same type always get the same expert weights.

**Experts as nn.ModuleList:** Each expert is a separate `MLPLayer` instance stored in `nn.ModuleList`, not fused tensors. This enables future Expert Parallelism (EP) where experts can be sharded across GPUs.

**Activation inside experts:** Each expert's `MLPLayer` includes its own activation function. This is critical for future EP — each expert must be self-contained on its own GPU.

**JIT compatibility:** Uses separate `Optional[MLPLayer]` and `Optional[MoELayer]` attributes with boolean dispatch flags, avoiding `isinstance` checks that JIT doesn't support.

## Architecture

```
MoELayer
├── gate: MLPLayer(tebd_dim → routed_experts, no bias, no activation)
├── experts: ModuleList[MLPLayer × routed_experts]  (with activation)
└── shared_experts: ModuleList[MLPLayer × share_expert]  (with activation, always active)

Routing flow:
  type_embeddings [ntypes, tebd_dim]
  → gate → logits [ntypes, routed_experts]
  → top-k → softmax → weights [ntypes, routed_top_k]
  → index by atom_types → per-atom weights [nb, nloc, routed_top_k]
  → broadcast to input shape → weighted expert outputs
```

## Parameters

Added to `RepFlowArgs` and `dpa3_repflow_args()`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_experts` | int | 1 | Total number of experts. 1 = no MoE (backward compatible) |
| `moe_top_k` | int | 1 | Experts activated per token (routed + shared) |
| `use_node_moe` | bool | True | Apply MoE to node MLPs (node_self, node_sym, node_edge) |
| `use_edge_moe` | bool | False | Apply MoE to edge MLPs (edge_self, edge_angle_linear2) |
| `use_angle_moe` | bool | False | Apply MoE to angle MLPs (edge_angle_linear1, angle_self) |
| `share_expert` | int | 0 | Number of shared experts (always active, counted in top_k) |

### Parameter Constraints
- `n_experts > 1` to enable MoE
- `share_expert < moe_top_k`
- `routed_experts = n_experts - share_expert`
- `routed_top_k = moe_top_k - share_expert`

## Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `deepmd/pt/model/network/moe.py` | CREATE | Core `MoELayer` class with type-based routing |
| `deepmd/dpmodel/descriptor/dpa3.py` | MODIFY | Added MoE params to `RepFlowArgs` |
| `deepmd/utils/argcheck.py` | MODIFY | Added MoE args to `dpa3_repflow_args()` |
| `deepmd/pt/model/descriptor/repflow_layer.py` | MODIFY | Conditional MoE/MLP creation, JIT-compatible dispatch, per-path optim flags |
| `deepmd/pt/model/descriptor/repflows.py` | MODIFY | Thread MoE config + type_embeddings to layers |
| `deepmd/pt/model/descriptor/dpa3.py` | MODIFY | Extract type embedding table, pass to repflows |
| `deepmd/dpmodel/descriptor/repflows.py` | MODIFY | Pop MoE fields in dpmodel deserialize for compat |
| `source/tests/pt/model/test_moe.py` | CREATE | 9 unit tests for MoELayer + DPA3 MoE |
| `source/tests/pt/test_moe_e2e.py` | CREATE | End-to-end training + freeze test |

## Usage

### JSON Config

Add MoE parameters to the `repflow` section of a DPA3 descriptor config:

```json
{
  "model": {
    "descriptor": {
      "type": "dpa3",
      "repflow": {
        "n_dim": 128,
        "e_dim": 64,
        "a_dim": 32,
        "nlayers": 6,
        "e_rcut": 6.0,
        "e_rcut_smth": 5.3,
        "e_sel": 120,
        "a_rcut": 4.0,
        "a_rcut_smth": 3.5,
        "a_sel": 30,
        "axis_neuron": 4,
        "update_angle": true,
        "smooth_edge_update": true,
        "n_experts": 4,
        "moe_top_k": 2,
        "use_node_moe": true,
        "use_edge_moe": false,
        "use_angle_moe": false,
        "share_expert": 0
      }
    }
  }
}
```

### Backward Compatibility

When `n_experts=1` (default), no MoE layers are created. The model behaves identically to the original implementation. Existing configs and serialized models work without changes.

## Layer Mapping

Which MLPLayers get replaced by MoE when enabled:

| Layer | Controls | MoE Flag |
|-------|----------|----------|
| `node_self_mlp` | Node self-interaction | `use_node_moe` |
| `node_sym_linear` | Node symmetrization | `use_node_moe` |
| `node_edge_linear` | Edge→node message | `use_node_moe` |
| `edge_self_linear` | Edge self-interaction | `use_edge_moe` |
| `edge_angle_linear1` | Angle→edge message (1st) | `use_angle_moe` |
| `edge_angle_linear2` | Angle→edge message (2nd) | `use_edge_moe` |
| `angle_self_linear` | Angle self-interaction | `use_angle_moe` |

## Optimized Update Path

The `optim_update` optimization (matrix decomposition for edge/angle updates) is automatically disabled per-path when MoE is involved:

- `optim_node_edge`: disabled if `node_edge_linear` is MoE
- `optim_edge_self`: disabled if `edge_self_linear` is MoE
- `optim_angle`: disabled if `edge_angle_linear1` or `angle_self_linear` is MoE

Non-MoE layers keep the optimized path even when other layers use MoE.

## Dynamic Selection Support

MoE works with `use_dynamic_sel=True`. For flat tensors (`[n_edge, dim]` or `[n_angle, dim]`), the `edge_index` parameter maps flat indices back to center atoms for routing weight lookup:

```python
# atom_weights: [nb*nloc, top_k]
# edge_index: [n_edge] → center atom index
edge_weights = atom_weights[edge_index]  # [n_edge, top_k]
```

## Serialization

MoE layers serialize/deserialize transparently. Each layer stores a `_type` field (`"MoELayer"` or `"MLPLayer"`) to determine the correct deserialization class. The dpmodel backend ignores MoE-specific fields for cross-backend compatibility.

## Testing

### Run Unit Tests

```bash
# All MoE tests (9 tests)
python -m pytest source/tests/pt/model/test_moe.py -v

# Backward compatibility (existing DPA3 tests)
python -m pytest source/tests/pt/model/test_dpa3.py -v

# End-to-end training + freeze
python source/tests/pt/test_moe_e2e.py
```

### Test Coverage

| Test | What it verifies |
|------|-----------------|
| `TestMoELayer::test_node_forward_backward` | Node [nb,nloc,dim] forward + gradient |
| `TestMoELayer::test_edge_forward` | Edge [nb,nloc,nnei,dim] forward |
| `TestMoELayer::test_angle_forward` | Angle [nb,nloc,a_nnei,a_nnei,dim] forward |
| `TestMoELayer::test_flat_dynamic_sel` | Dynamic sel [n_flat,dim] with edge_index |
| `TestMoELayer::test_share_expert` | share_expert=1 construction + forward |
| `TestMoELayer::test_serialization_roundtrip` | Serialize → deserialize → identical output |
| `TestDPA3MoE::test_moe_forward_backward` | Full DPA3 with node MoE, forward + backward |
| `TestDPA3MoE::test_moe_serialization` | DPA3 MoE serialize → deserialize roundtrip |
| `TestDPA3MoE::test_moe_all_targets` | Node + edge + angle MoE simultaneously |
| `TestDescrptDPA3::test_consistency` | Backward compat: n_experts=1 matches original |
| `TestDescrptDPA3::test_jit` | JIT scripting works with default config |
| E2E test | dp train (20 steps) + dp freeze with MoE config |

## Future: Expert Parallelism (EP)

The implementation is designed for future EP support:

- Experts stored as `nn.ModuleList` (not fused tensors) → can shard across GPUs
- Activation inside each expert → self-contained per GPU
- `ep_group` parameter reserved in design (not yet implemented)
- Reference EP implementation: `/mnt/data_nas/zhangd/claude_test/moe_ep_poc`

EP would add All-to-All communication to dispatch tokens to expert-owning GPUs and collect results back, following the pattern in the PoC.
