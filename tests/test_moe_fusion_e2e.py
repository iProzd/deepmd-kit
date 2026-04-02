#!/usr/bin/env python3
"""End-to-end 2-GPU EP training test with MLP fusion for DPA3 MoE.

Launch: torchrun --nproc_per_node=2 tests/test_moe_fusion_e2e.py

Tests:
  1. Model construction: verify fused layers exist
  2. Inline 2-GPU EP training: forward+backward with create_graph=True
"""

import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def setup():
    dist.init_process_group("nccl")
    rank = int(os.environ.get("RANK", dist.get_rank()))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, local_rank


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def make_water_system(e_rcut, e_sel, device="cuda"):
    """Create a small water-like system with proper nlist and mapping.

    Uses extend_coord_with_ghosts and build_neighbor_list to produce
    valid inputs for DescrptDPA3.forward().
    Note: extend_coord_with_ghosts requires float64 for einsum precision.
    """
    from deepmd.pt.utils.nlist import build_neighbor_list, extend_coord_with_ghosts

    # 3 water molecules = 9 atoms (3 O + 6 H) in a box
    nloc = 9
    coord = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # O
            [0.0, 0.96, 0.0],  # H
            [0.0, 0.0, 0.96],  # H
            [3.0, 0.0, 0.0],  # O
            [3.0, 0.96, 0.0],  # H
            [3.0, 0.0, 0.96],  # H
            [0.0, 3.0, 0.0],  # O
            [0.0, 3.96, 0.0],  # H
            [0.0, 3.0, 0.96],  # H
        ],
        dtype=torch.float64,
        device=device,
    ).unsqueeze(0)  # (1, nloc, 3)

    atype = torch.tensor(
        [0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=torch.long, device=device
    ).unsqueeze(0)  # (1, nloc)

    cell = torch.tensor(
        [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
        dtype=torch.float64,
        device=device,
    ).unsqueeze(0)  # (1, 9)

    # Flatten coord for extend_coord_with_ghosts: (nf, nloc*3)
    coord_flat = coord.reshape(1, nloc * 3)

    # Extend with ghost atoms
    ext_coord, ext_atype, mapping = extend_coord_with_ghosts(
        coord_flat, atype, cell, e_rcut
    )
    # ext_coord: (1, nall*3), ext_atype: (1, nall), mapping: (1, nall)

    nall = ext_atype.shape[1]

    # Build neighbor list
    nlist = build_neighbor_list(
        ext_coord,
        ext_atype,
        nloc,
        e_rcut,
        e_sel,
        distinguish_types=False,
    )
    # nlist: (1, nloc, e_sel)

    return ext_coord, ext_atype, nlist, mapping, nloc


def test_model_construction(rank, ep_group):
    """Test that DPA3 model constructs correctly with fusion enabled."""
    from deepmd.pt.model.descriptor.dpa3 import DescrptDPA3

    for fuse in [False, True]:
        repflow_args = {
            "n_dim": 32,
            "e_dim": 16,
            "a_dim": 8,
            "nlayers": 2,
            "e_rcut": 6.0,
            "e_rcut_smth": 5.0,
            "e_sel": 40,
            "a_rcut": 4.0,
            "a_rcut_smth": 3.5,
            "a_sel": 16,
            "axis_neuron": 4,
            "update_angle": True,
            "smooth_edge_update": True,
            "n_experts": 4,
            "moe_top_k": 2,
            "use_node_moe": True,
            "use_edge_moe": True,
            "use_angle_moe": True,
            "fuse_moe_mlps": fuse,
            "moe_ep_size": 2,
        }
        model = DescrptDPA3(
            repflow=repflow_args,
            ntypes=2,
            precision="float32",
            seed=1,
            ep_group=ep_group,
        ).cuda()

        # Count MoE and fused layers
        n_params = sum(p.numel() for p in model.parameters())
        n_moe_layers = 0
        n_fused_layers = 0
        for name, module in model.named_modules():
            from deepmd.pt.model.network.moe import MoELayer
            from deepmd.pt.model.network.moe_fused import FusedMoELayer

            if isinstance(module, FusedMoELayer):
                n_fused_layers += 1
            elif isinstance(module, MoELayer):
                n_moe_layers += 1

        fuse_str = "fused" if fuse else "unfused"
        log(
            rank,
            f"  [{fuse_str}] params={n_params}, MoE layers={n_moe_layers}, "
            f"Fused layers={n_fused_layers}",
        )

        if fuse:
            assert (
                n_fused_layers > 0
            ), f"Expected fused layers when fuse_moe_mlps=True, got {n_fused_layers}"
        else:
            assert (
                n_fused_layers == 0
            ), f"Expected no fused layers when fuse_moe_mlps=False, got {n_fused_layers}"

        del model
        torch.cuda.empty_cache()

    log(rank, "[PASS] Model construction test")


def test_e2e_training_ep(rank, ep_group):
    """Inline 2-GPU EP training test comparing fused vs unfused.

    Directly creates DPA3 model with EP support, generates valid water system
    data via extend_coord_with_ghosts + build_neighbor_list, runs
    forward+backward loop on both GPUs, and compares throughput.
    """
    from deepmd.pt.model.descriptor.dpa3 import DescrptDPA3

    ep_size = dist.get_world_size(group=ep_group)
    nsteps = 10
    warmup = 3

    e_rcut = 6.0
    e_sel = 40
    results = {}

    for fuse_label, fuse_flag in [("unfused_ep", False), ("fused_ep", True)]:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        repflow_args = {
            "n_dim": 32,
            "e_dim": 16,
            "a_dim": 8,
            "nlayers": 2,
            "e_rcut": e_rcut,
            "e_rcut_smth": 5.0,
            "e_sel": e_sel,
            "a_rcut": 4.0,
            "a_rcut_smth": 3.5,
            "a_sel": 16,
            "axis_neuron": 4,
            "update_angle": True,
            "smooth_edge_update": True,
            "n_experts": 4,
            "moe_top_k": 2,
            "use_node_moe": True,
            "use_edge_moe": True,
            "use_angle_moe": True,
            "fuse_moe_mlps": fuse_flag,
            "moe_ep_size": 2,
        }
        model = DescrptDPA3(
            repflow=repflow_args,
            ntypes=2,
            precision="float32",
            seed=1,
            ep_group=ep_group,
        ).cuda()

        # Sync all parameters across EP group
        for name, p in model.named_parameters():
            dist.broadcast(p.data, src=0, group=ep_group)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Generate valid test data using proper nlist utilities
        ext_coord, ext_atype, nlist, mapping, nloc = make_water_system(
            e_rcut, e_sel, device="cuda"
        )

        # Broadcast data so all ranks have identical input
        dist.broadcast(ext_coord, src=0, group=ep_group)
        dist.broadcast(ext_atype, src=0, group=ep_group)
        dist.broadcast(nlist, src=0, group=ep_group)
        dist.broadcast(mapping, src=0, group=ep_group)

        log(
            rank,
            f"  [{fuse_label}] Data: nloc={nloc}, nall={ext_atype.shape[1]}, "
            f"nlist={nlist.shape}, mapping={mapping.shape}",
        )

        losses_list = []
        t0 = None

        for step in range(nsteps + warmup):
            if step == warmup:
                torch.cuda.synchronize()
                dist.barrier(group=ep_group)
                t0 = time.perf_counter()

            optimizer.zero_grad()

            # Add small noise to coordinates each step (need grad for force)
            nall = ext_atype.shape[1]
            noise = torch.randn(1, nall * 3, device="cuda", dtype=ext_coord.dtype) * 0.01
            coord_noisy = (ext_coord + noise).detach().requires_grad_(True)

            try:
                result = model(
                    coord_noisy, ext_atype, nlist=nlist, mapping=mapping
                )
                desc = result[0]  # (nf, nloc, n_dim)
                energy = desc.sum()
                force = -torch.autograd.grad(
                    energy, coord_noisy, create_graph=True, retain_graph=True
                )[0]
                loss = energy + force.sum() * 0.01
            except Exception as e:
                log(rank, f"  [{fuse_label}] Step {step} failed: {e}")
                import traceback

                if rank == 0:
                    traceback.print_exc()
                raise

            loss.backward()

            # Sync gradients for shared params (non-expert parameters)
            for name, p in model.named_parameters():
                if p.grad is not None and "experts" not in name:
                    dist.all_reduce(p.grad, group=ep_group)
                    p.grad /= ep_size

            optimizer.step()

            if step >= warmup:
                losses_list.append(loss.item())

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        throughput = nsteps / elapsed

        # Check for NaN
        has_nan = any(np.isnan(l) or np.isinf(l) for l in losses_list)
        log(
            rank,
            f"  [{fuse_label}] losses: {losses_list[0]:.4f} -> {losses_list[-1]:.4f}, "
            f"throughput: {throughput:.1f} steps/s, NaN: {has_nan}",
        )

        assert not has_nan, f"[{fuse_label}] NaN/Inf in losses"

        results[fuse_label] = {
            "losses": losses_list,
            "throughput": throughput,
            "elapsed": elapsed,
        }

        del model, optimizer
        torch.cuda.empty_cache()

    # Summary
    speedup = results["unfused_ep"]["elapsed"] / results["fused_ep"]["elapsed"]
    log(rank, f"\n  EP Training comparison (inline, {nsteps} steps):")
    log(rank, f"    Unfused EP: {results['unfused_ep']['throughput']:.1f} steps/s")
    log(rank, f"    Fused EP:   {results['fused_ep']['throughput']:.1f} steps/s")
    log(rank, f"    Speedup:    {speedup:.2f}x")

    log(rank, "[PASS] E2E EP training test")


def main():
    rank, local_rank = setup()
    mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("dp", "ep"))
    ep_group = mesh["ep"].get_group()

    log(rank, "=" * 70)
    log(rank, "MoE Fusion E2E Tests (2-GPU Expert Parallelism)")
    log(rank, "=" * 70)

    log(rank, "\n--- Test 1: Model construction ---")
    test_model_construction(rank, ep_group)
    dist.barrier(group=ep_group)

    log(rank, "\n--- Test 2: E2E 2-GPU EP training (fused vs unfused) ---")
    test_e2e_training_ep(rank, ep_group)
    dist.barrier(group=ep_group)

    log(rank, "\n" + "=" * 70)
    log(rank, "All fusion E2E tests passed!")
    log(rank, "=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
