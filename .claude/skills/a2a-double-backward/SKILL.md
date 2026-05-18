---
name: a2a-double-backward
description: How to write an All-to-All collective that supports second-order derivatives (needed for force-loss training). The single most-important pattern in the SeZM MoE implementation; getting this wrong silently kills training. Read before implementing Step 1 (a2a_ops.py) or any code that touches A2A backward.
---

# Recursively differentiable All-to-All

## The problem in one sentence

SeZM training computes `force = -∂E/∂x`, and `loss.backward()` walks back through `force` — this means **the forward path is differentiated twice**. PyTorch collective ops like `dist.all_to_all_single` are not autograd-aware by default, so wrapping them in a naive `autograd.Function` whose `backward` calls another `all_to_all_single` works for the first backward but **silently drops the gradient on the second**.

## The fix

`backward` must call `cls.apply(...)` (the *same* Function) again, not the raw collective. The recursive `.apply()` inserts a fresh autograd node, so the second `.backward()` can walk over it.

```python
class _AllToAllDouble(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, send_splits, recv_splits, group):
        ctx.group = group
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        return _a2a_raw(x, send_splits, recv_splits, group)

    @staticmethod
    def backward(ctx, grad_output):
        # CRITICAL: recursive .apply(), not _a2a_raw directly.
        grad_input = _AllToAllDouble.apply(
            grad_output,
            ctx.recv_splits,  # swap
            ctx.send_splits,  # swap
            ctx.group,
        )
        return grad_input, None, None, None
```

## What "the same Function" does for the autograd graph

| Call                      | Forward                                                                  | Backward                                           |
| ------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------- |
| 1st forward               | `_a2a_raw(x, send, recv)`                                                | uses ctx → produces grad via `.apply`              |
| 1st backward              | (above) → ANOTHER `_AllToAllDouble` node created                         | this node's own backward is also `_AllToAllDouble` |
| 2nd backward (force-loss) | walks the 1st-backward graph; the A2A node it finds has a valid backward | recurses; arbitrary order works                    |

The key insight: each `.apply()` creates a new autograd node. The recursive call makes the backward graph **self-similar to the forward graph** — both are made of `_AllToAllDouble` nodes — so PyTorch can differentiate any order.

## Anti-pattern (does NOT work)

```python
# WRONG — first backward works, second backward silently produces 0
@staticmethod
def backward(ctx, grad_output):
    return (
        _a2a_raw(grad_output, ctx.recv_splits, ctx.send_splits, ctx.group),
        None,
        None,
        None,
    )
```

The result of `_a2a_raw` is a plain tensor with no autograd history, so `loss.backward()` cannot trace through it on the second pass.

## Why it works through DDP/EP groups

The recursive call respects the `ProcessGroup` (carried in `ctx.group`) so the backward A2A travels the same NCCL group as the forward. Provided every rank executes the chain in the same order (which is guaranteed by SeZM's strictly sequential block structure), no deadlock can occur.

## Quick verification UT

```python
def test_2nd_backward():
    x = torch.randn(local_n, D, requires_grad=True, device="cuda")
    y = _AllToAllDouble.apply(x, send_splits, recv_splits, group)
    loss = y.pow(2).sum()
    # 1st backward with create_graph
    (g,) = torch.autograd.grad(loss, x, create_graph=True)
    # 2nd backward — this is where the recursive .apply() pays off
    g.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0  # non-trivial gradient
```

If you see `x.grad is None` or `x.grad.abs().sum() == 0` after the 2nd `.backward()`, the `.apply()` is missing somewhere.

## Single-GPU short-circuit (`group=None`)

```python
def all_to_all_differentiable(x, send_splits, recv_splits, group):
    if group is None:
        return x  # PyTorch's autograd already tracks x directly
    return _AllToAllDouble.apply(x, send_splits, recv_splits, group)
```

When `group is None`, returning `x` unchanged is **necessary** for the no-MoE/no-EP fallback to also have correct gradients without paying any collective cost.

## Common mistakes summary

| Mistake                                                     | Symptom                                                |
| ----------------------------------------------------------- | ------------------------------------------------------ |
| `backward` calls `_a2a_raw` directly                        | 2nd backward gradient is wrong / zero                  |
| `backward` calls a different `Function`'s `.apply`          | autograd graph topology broken                         |
| Stored `ctx.group` is the *default* group, not the EP group | NCCL group mismatch → deadlock or wrong data           |
| `send_splits` / `recv_splits` not swapped in backward       | shape mismatch, immediate `RuntimeError`               |
| Output of `_a2a_raw` not contiguous                         | NCCL warning, silent data corruption on some platforms |

## Where this lives in the SeZM MoE plan

- Step 1 implements `_AllToAllDouble` in `deepmd/pt/model/descriptor/sezm_nn/moe/a2a_ops.py`
- Steps 4 and 6 use it for dispatch and combine
- Step 9 (training loop) needs `wrapper.no_sync()` around `loss.backward()` so DDP doesn't double-reduce the routing-expert gradients that already traveled through `_AllToAllDouble.backward`

## Reference

DPA3 implementation: `/mnt/data_nas/zhangd/claude_space/deepmd-kit-moe/deepmd/pt/model/network/moe_ep_ops.py`. Copy verbatim, rename namespace.
