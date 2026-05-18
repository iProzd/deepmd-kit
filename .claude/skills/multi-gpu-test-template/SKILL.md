---
name: multi-gpu-test-template
description: Scaffolding template for torchrun-based multi-GPU unit tests in the SeZM MoE project. Use when writing a `test_sezm_moe_<topic>_multigpu.py` for any Step that has a multi-GPU UT requirement. Provides distributed setup/teardown, ep_size/dp_size pickling, save→load consistency pattern, and 2nd-backward deadlock detection.
---

# Multi-GPU UT template

All multi-GPU UTs use a common scaffolding to avoid copy-paste of distributed-init boilerplate. Follow this template.

## Standard file structure

```python
# source/tests/pt/test_sezm_moe_<topic>_multigpu.py
"""<one-line description of what this test verifies>."""

from __future__ import annotations

import os
import sys
import torch
import torch.distributed as dist


def setup_dist():
    """Initialize the default process group from torchrun-set env vars."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world


def teardown_dist():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def test_main(ep_size: int):
    rank, world = setup_dist()
    try:
        # ── your test body here ──
        # 1. Build groups using your sezm_moe_ep_dp.init_ep_dp_groups
        # 2. Construct module(s) under test
        # 3. Run forward/backward
        # 4. Assertions (often by gathering tensors from all ranks)
        # 5. Print "PASS rank={rank}" on success
        ...
        print(f"[rank {rank}] PASS")
    finally:
        teardown_dist()


if __name__ == "__main__":
    # torchrun sets these
    ep_size = int(os.environ.get("EP_SIZE", "2"))
    test_main(ep_size)
```

Invoked as:

```bash
torchrun --nproc_per_node=4 --master_port=29501 \
    source/tests/pt/test_sezm_moe_<topic>_multigpu.py
```

Different ep_size values:

```bash
EP_SIZE=2 torchrun --nproc_per_node=4 ...   # ep=2, dp=2
EP_SIZE=4 torchrun --nproc_per_node=4 ...   # ep=4, dp=1
EP_SIZE=8 torchrun --nproc_per_node=8 ...   # ep=8, dp=1
```

## Pattern 1: forward + backward + 2nd-backward (no deadlock)

For Step 1 (`_AllToAllDouble`):

```python
def test_main(ep_size):
    rank, world = setup_dist()
    try:
        # Construct dummy data with known shape
        local_n = 8
        D = 16
        x = torch.randn(
            local_n, D, device="cuda", requires_grad=True, dtype=torch.float64
        )
        send_splits = [local_n // world] * world  # uniform splits
        recv_splits = send_splits  # symmetric for round-trip

        from deepmd.pt.model.descriptor.sezm_nn.moe.a2a_ops import (
            all_to_all_differentiable,
        )

        y = all_to_all_differentiable(x, send_splits, recv_splits, dist.group.WORLD)
        loss = y.pow(2).sum()

        # 1st backward with create_graph (force-loss training pattern)
        (g,) = torch.autograd.grad(loss, x, create_graph=True)
        assert g.shape == x.shape, f"rank {rank} grad shape mismatch"

        # 2nd backward — this is where _AllToAllDouble.apply() recursion matters
        g.sum().backward()
        assert x.grad is not None, f"rank {rank} got None grad after 2nd backward"
        assert x.grad.abs().sum().item() > 0, f"rank {rank} got zero grad"

        print(f"[rank {rank}] PASS")
    finally:
        teardown_dist()
```

Deadlock detection: every rank should reach `print` within a few seconds. If a rank hangs, the runner (`multi-gpu-tester` agent) will time-out and `pkill`.

## Pattern 2: 1 GPU save → N GPU load → numerical equivalence

For Step 4 (`MoESO2Convolution` ckpt consistency) and Step 10 (resharding):

```python
import pickle

def save_single_gpu_state():
    """Run on 1 GPU; serialize state_dict + a fixed input/output pair."""
    torch.manual_seed(42)
    module_single = MoESO2Convolution(
        ep_group=None, ep_size=1,   # single-GPU mode
        ...
    ).cuda()

    x = torch.randn(E, F, D_m, Cf, device="cuda", dtype=torch.float64)
    type_emb = torch.randn(N, C_type, device="cuda", dtype=torch.float64)
    edge_index = torch.randint(0, N, (2, E), device="cuda")

    y = module_single(x, type_emb, edge_index, ...)

    torch.save({
        "state_dict": module_single.state_dict(),
        "input": (x, type_emb, edge_index),
        "expected_output": y,
    }, "/tmp/sezm_moe_ckpt.pt")


def test_main(ep_size):
    rank, world = setup_dist()
    try:
        # All ranks load the same single-GPU checkpoint
        ckpt = torch.load("/tmp/sezm_moe_ckpt.pt", map_location=f"cuda:{rank}")
        x, type_emb, edge_index = ckpt["input"]
        expected = ckpt["expected_output"]

        # Build EP/DP groups
        from deepmd.pt.utils.sezm_moe_ep_dp import init_ep_dp_groups
        ep_group, dp_group, ep_rank, _, _, _ = init_ep_dp_groups(ep_size)

        # Build multi-GPU module
        module_multi = MoESO2Convolution(
            ep_group=ep_group, ep_size=ep_size,
            ...
        ).cuda()

        # Load with resharding (extract the local slice of routing_matrix)
        local_state = reshard_state_dict_for_ep(ckpt["state_dict"], ep_rank, ep_size)
        module_multi.load_state_dict(local_state)

        # Forward
        y = module_multi(x, type_emb, edge_index, ...)

        # Each rank should compute its own slice (or the full output if non-MoE);
        # depending on design, either gather y from all ranks or compare directly
        torch.testing.assert_close(y, expected, atol=1e-10, rtol=1e-10)
        print(f"[rank {rank}] PASS")
    finally:
        teardown_dist()
```

This pattern is the gold standard for proving "the multi-GPU implementation is consistent with the single-GPU reference".

## Pattern 3: gradient sync verification

For Step 6 (`sync_moe_gradients`):

```python
def test_main(ep_size):
    rank, world = setup_dist()
    try:
        from deepmd.pt.utils.sezm_moe_ep_dp import (
            init_ep_dp_groups,
            sync_moe_gradients,
            _is_routing_expert_param,
        )

        ep_group, dp_group, ep_rank, _, dp_rank, dp_size = init_ep_dp_groups(ep_size)

        # Build a tiny model whose params are clearly named:
        # - .routing_matrix_layer0_m0  (routing — should sync via dp_group)
        # - .other_param               (non-routing — should sync via world)
        model = build_tiny_moe_test_model()

        # Each rank gets DIFFERENT data so per-rank gradients differ
        loss = model(per_rank_data(rank)).sum()
        loss.backward()

        # Record pre-sync routing-expert grad on each rank
        pre_routing = {
            n: p.grad.clone()
            for n, p in model.named_parameters()
            if _is_routing_expert_param(n)
        }
        pre_other = {
            n: p.grad.clone()
            for n, p in model.named_parameters()
            if not _is_routing_expert_param(n)
        }

        sync_moe_gradients(model, dp_group, None, dp_size, world)

        # Verify routing expert: same within dp_group (column), different across ep_groups
        for n, p in model.named_parameters():
            if not _is_routing_expert_param(n):
                continue
            # Within dp_group: all ranks should now have the same grad
            buf = [torch.zeros_like(p.grad) for _ in range(dp_size)]
            dist.all_gather(buf, p.grad, group=dp_group)
            for other in buf:
                torch.testing.assert_close(p.grad, other)

        # Verify non-routing: same across world
        for n, p in model.named_parameters():
            if _is_routing_expert_param(n):
                continue
            buf = [torch.zeros_like(p.grad) for _ in range(world)]
            dist.all_gather(buf, p.grad)
            for other in buf:
                torch.testing.assert_close(p.grad, other)

        # Verify divisor is world_size (compare against manually summed reference)
        # ... (use a known-good per-rank gradient and check post-sync value)

        print(f"[rank {rank}] PASS")
    finally:
        teardown_dist()
```

## Pattern 4: deadlock detection (force-virial 2nd-derivative)

For Step 4 and Step 6 multi-GPU 2nd-backward:

```python
import signal


def deadlock_alarm(signum, frame):
    print(f"[rank {dist.get_rank()}] DEADLOCK timeout")
    sys.exit(1)


def test_main(ep_size):
    signal.signal(signal.SIGALRM, deadlock_alarm)
    signal.alarm(120)  # 120s timeout for the whole test

    rank, world = setup_dist()
    try:
        # ... build model, compute energy, force, force_loss ...
        # ... loss.backward() ...
        # Every rank must reach the print statement within 120s
        print(f"[rank {rank}] PASS")
        signal.alarm(0)  # clear alarm on success
    finally:
        teardown_dist()
```

If any rank hits the alarm, it exits with code 1 → torchrun reports failure → `multi-gpu-tester` picks it up.

## Common gotchas

1. **Set `torch.cuda.set_device(rank % nvis)`** at start, otherwise all ranks land on cuda:0.
1. **Use `dist.barrier()` before destroy_process_group**, otherwise faster ranks may exit and stall slower ranks at their next collective.
1. **Pickle path must be on a shared filesystem if multi-node**, but single-node uses `/tmp/...` which is per-host — fine for `--nnodes=1`.
1. **`torch.manual_seed(42)` is per-rank by default**; for "all ranks same RNG" tests, use `torch.manual_seed(42 + rank)` to give each rank distinct data while keeping reproducibility.
1. **Tests with `dist.group.WORLD` will hang in single-rank runs** if you forgot to init the default group. Always call `setup_dist` first.

## Skeleton checklist before submitting a new multi-GPU UT

- [ ] Imports `setup_dist` / `teardown_dist` (or inlines them per pattern above)
- [ ] Uses `signal.alarm` for deadlock detection on any test involving 2nd backward
- [ ] Has both `EP_SIZE=2` and `EP_SIZE=N` variants if behavior should be consistent
- [ ] Uses `fp64` for gradcheck-style numerical comparisons
- [ ] Prints `PASS rank={rank}` on each rank for clarity in torchrun's interleaved output
- [ ] Calls `dist.barrier()` before `destroy_process_group`
- [ ] All tensors created on `cuda:{rank % nvis}`, not `cuda:0`
