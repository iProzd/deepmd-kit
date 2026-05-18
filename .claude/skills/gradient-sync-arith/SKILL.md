---
name: gradient-sync-arith
description: Derivation of why routing-expert gradient sync divides by world_size (not dp_size). The "gotcha" of MoE EP+DP training; if implemented wrong, training silently converges to the wrong answer. Read before implementing Step 6 (sync_moe_gradients) and Step 9 (training loop integration).
---

# Gradient sync arithmetic for MoE EP+DP

## TL;DR

When all-reducing routing-expert gradients across the DP group:

```python
dist.all_reduce(param.grad, op=SUM, group=dp_group)
param.grad.div_(world_size)  # ← world_size, NOT dp_size
```

This skill explains why.

## Setup

```
world_size = ep_size × dp_size

8 GPU example with ep_size=4, dp_size=2:

           EP rank 0   EP rank 1   EP rank 2   EP rank 3
DP rank 0:   GPU 0       GPU 1       GPU 2       GPU 3     ← ep_group_0
DP rank 1:   GPU 4       GPU 5       GPU 6       GPU 7     ← ep_group_1
             ↑ dp_group_0 ↑ dp_group_1 ↑ dp_group_2 ↑ dp_group_3
```

Each routing expert lives on **one rank per dp_group**, so it exists on **`dp_size` ranks** in total (the same EP-rank column). For example expert with `local_id=0` on EP rank 1 lives on GPUs 1 and 5.

Other parameters (router gate, shared experts, non-MoE params) live on **all `world_size` ranks**.

## Sample mean as the target

We want each parameter's gradient to be the **mean of per-sample gradients across all data the world has seen this step**, i.e.

$$\bar{g} = \frac{1}{world\_size} \sum_{r=0}^{world\_size-1} g_r$$

(Here `g_r` is the gradient computed by rank `r` on its local minibatch.)

This is what plain DDP gives you when you call `loss.backward()` with default DDP wrapping.

## Where does each rank's gradient contribution come from for a routing expert?

For a routing-expert parameter `θ_e` that lives on EP rank `er` (and thus on every `(*, er)` GPU in the grid):

1. **Forward**: tokens from every rank in the same EP group (and that's `ep_size` ranks) are A2A-dispatched to the rank holding `θ_e`.
1. The expert computes `act(x @ θ_e + b_e)` on the union of all incoming tokens.
1. The output is A2A-combined back to the original source ranks.
1. **Backward**: 1st backward propagates `dL/dy` back through the combine A2A → expert backward → dispatch A2A.
1. **The expert's backward** computes `dθ_e.grad += dL/dy @ x.T` over all incoming tokens. Since these tokens came from `ep_size` ranks, the contributions of `ep_size` ranks' data are **summed into `θ_e.grad` automatically** by the expert's local backward computation.

So after `loss.backward()` (or rather after `_AllToAllDouble.backward`), on each rank holding `θ_e`:

$$\theta_e.\text{grad}\Big|_{\text{after local backward}} = \sum_{r \in \text{my\_ep\_group}} g_r$$

— `ep_size` ranks' gradients are already summed in.

## What does `dist.all_reduce(grad, group=dp_group, op=SUM)` add?

The DP group is the **column** of the grid (different DP rank, same EP rank). After the EP-internal aggregation, each DP-column rank holds a "sum of `ep_size` ranks' grads". An all-reduce across the dp_group then sums those `dp_size` partial sums:

$$\theta_e.\text{grad}\Big|_{\text{after all\_reduce}} = \sum_{\text{dp\_rank}} \sum_{r \in \text{ep\_group for that dp\_rank}} g_r = \sum_{r=0}^{world\_size - 1} g_r$$

— we've now summed all `world_size = ep_size × dp_size` rank-level gradients.

## Therefore, divide by `world_size`

To get the *mean*:

$$\bar{g} = \frac{1}{world\_size} \cdot \sum_r g_r$$

```python
dist.all_reduce(
    grad, op=SUM, group=dp_group
)  # sums dp_size × ep_size = world_size contributions
grad.div_(world_size)  # mean
```

**Common confusion**: people see "all-reduce across dp_group of size dp_size" and instinctively want `/dp_size`. But the EP-side sum has already happened *inside the expert backward*, so the divisor must account for both factors.

## Sanity check via dimensional analysis

| Quantity                                                          | Value                                                        |
| ----------------------------------------------------------------- | ------------------------------------------------------------ |
| Ranks holding `θ_e`                                               | `dp_size`                                                    |
| Gradient contributions accumulated by EP backward (per such rank) | `ep_size`                                                    |
| Gradient contributions added by DP all-reduce                     | `dp_size - 1` partial sums (each of `ep_size` contributions) |
| Total contributions summed in `θ_e.grad` after all-reduce         | `dp_size × ep_size = world_size`                             |
| Divisor for mean                                                  | `world_size` ✓                                               |

## What about non-routing parameters?

These exist on **all `world_size` ranks**. Plain DDP would do:

```python
dist.all_reduce(grad, op=SUM, group=world)  # sums world_size contributions
grad.div_(world_size)
```

In SeZM MoE we disable DDP's auto-sync (`wrapper.no_sync()`) and do this manually for ALL parameters, separately routing those that are routing-expert through `dp_group`:

```python
for name, param in model.named_parameters():
    if param.grad is None:
        continue
    if _is_routing_expert_param(name):
        dist.all_reduce(param.grad, op=SUM, group=dp_group)
        param.grad.div_(world_size)  # world_size!
    else:
        dist.all_reduce(param.grad, op=SUM, group=world_group)
        param.grad.div_(world_size)
```

Both branches end with `.div_(world_size)`; the only difference is the group.

## Why DDP's auto-sync is wrong here

DDP's standard `loss.backward()` does **world-group** all-reduce on every parameter. For routing experts, this is **wrong**:

- Each rank's local `θ_e.grad` is only meaningful if that rank holds expert `e`.
- For ranks that DON'T hold `e`, their `θ_e.grad` is either zero or undefined (parameter doesn't even exist physically — DDP would just sync the *same view*).
- World-reducing it would average `world_size` copies of an `ep_size`-rank-sum with `(world_size - dp_size)` zeros → wrong magnitude.

That's why the training loop must use `wrapper.no_sync()` around `loss.backward()` and call `sync_moe_gradients` manually.

## Identification of routing-expert parameters

```python
def _is_routing_expert_param(name: str) -> bool:
    return ".routing_matrix" in name or ".routing_bias" in name
```

This requires the implementer to **name every routing-expert parameter** with `routing_matrix` or `routing_bias` in its full module path. See SPEC.md §5.4 for the naming convention.

## Edge cases

| Scenario                                      | Behavior                                                                                                                                                                                                                                                                                         |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `world_size == 1`                             | Skip sync entirely (return early)                                                                                                                                                                                                                                                                |
| `dp_size == 1` and `world_size > 1` (pure EP) | Routing experts: A2A backward already sums all `ep_size = world_size` ranks; **no extra all-reduce needed**, but still `div_(world_size)`. In code: still call `all_reduce(group=dp_group)` (size-1 group is a no-op) and divide; the no-op all-reduce is cheap and keeps the code path uniform. |
| `ep_size == 1` (pure DP)                      | All routing experts exist on all ranks; degenerates to standard DDP. `_is_routing_expert_param` still returns True; dp_group == world_group in this case.                                                                                                                                        |
| New expert added during training (not v1)     | Not supported; raise.                                                                                                                                                                                                                                                                            |

## Where this lives in the SeZM MoE plan

- Step 6 implements `sync_moe_gradients` in `deepmd/pt/utils/sezm_moe_ep_dp.py` (copy from DPA3 `moe_ep_dp.py`).
- Step 9 integrates it into `training.py` after `wrapper.no_sync(): loss.backward()`.

## Reference

DPA3 implementation: `/mnt/data_nas/zhangd/claude_space/deepmd-kit-moe/deepmd/pt/utils/moe_ep_dp.py` lines 115-178. Read the inline comments — they explicitly state "Divide by world_size (not dp_size) because All-to-All backward already aggregates..."
