#!/usr/bin/env python3
"""Debug 8-expert EP deadlock: fine-grained per-phase sync + memory/NaN checks."""

import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import numpy as np

def setup():
    dist.init_process_group("nccl")
    rank = int(os.environ.get("RANK", dist.get_rank()))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, local_rank

def make_water_system(n_molecules, e_rcut, e_sel, device="cuda"):
    from deepmd.pt.utils.nlist import build_neighbor_list, extend_coord_with_ghosts
    coords = []
    atypes = []
    n_per_side = int(np.ceil(n_molecules ** (1.0 / 3.0)))
    spacing = 3.0
    count = 0
    for ix in range(n_per_side):
        for iy in range(n_per_side):
            for iz in range(n_per_side):
                if count >= n_molecules: break
                base = np.array([ix * spacing, iy * spacing, iz * spacing])
                coords.append(base)
                coords.append(base + [0.0, 0.96, 0.0])
                coords.append(base + [0.0, 0.0, 0.96])
                atypes.extend([0, 1, 1])
                count += 1
            if count >= n_molecules: break
        if count >= n_molecules: break
    nloc = len(coords)
    box_size = n_per_side * spacing + e_rcut + 1.0
    coord_np = np.array(coords, dtype=np.float64)
    coord = torch.tensor(coord_np, dtype=torch.float64, device=device).unsqueeze(0)
    atype = torch.tensor(atypes, dtype=torch.long, device=device).unsqueeze(0)
    cell = torch.zeros(1, 9, dtype=torch.float64, device=device)
    cell[0, 0] = box_size; cell[0, 4] = box_size; cell[0, 8] = box_size
    coord_flat = coord.reshape(1, nloc * 3)
    ext_coord, ext_atype, mapping = extend_coord_with_ghosts(coord_flat, atype, cell, e_rcut)
    nlist = build_neighbor_list(ext_coord, ext_atype, nloc, e_rcut, e_sel, distinguish_types=False)
    return ext_coord, ext_atype, nlist, mapping, nloc

def sync_barrier(ep_group, rank, label):
    """Explicit synchronize + barrier with label for debugging."""
    torch.cuda.synchronize()
    dist.barrier(group=ep_group)

def check_params_synced(model, ep_group, rank):
    """Check if non-expert params are in sync between ranks."""
    for name, p in model.named_parameters():
        if "experts" not in name:
            # Compare via allreduce of diff
            p_clone = p.data.clone()
            dist.broadcast(p_clone, src=0, group=ep_group)
            diff = (p.data - p_clone).abs().max().item()
            if diff > 1e-6:
                print(f"[rank {rank}] PARAM DESYNC: {name}, max_diff={diff}", flush=True)
                return False
    return True

def check_grads(model, rank):
    """Check for NaN/Inf in gradients."""
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                print(f"[rank {rank}] NaN/Inf GRAD: {name}, "
                      f"nan={torch.isnan(p.grad).sum().item()}, "
                      f"inf={torch.isinf(p.grad).sum().item()}", flush=True)
                return False
    return True

def main():
    rank, local_rank = setup()
    mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("dp", "ep"))
    ep_group = mesh["ep"].get_group()
    ep_size = dist.get_world_size(group=ep_group)

    from deepmd.pt.model.descriptor.dpa3 import DescrptDPA3

    ext_coord, ext_atype, nlist, mapping, nloc = make_water_system(8, 6.0, 40)
    dist.broadcast(ext_coord, src=0, group=ep_group)
    dist.broadcast(ext_atype, src=0, group=ep_group)
    dist.broadcast(nlist, src=0, group=ep_group)
    dist.broadcast(mapping, src=0, group=ep_group)
    nall = ext_atype.shape[1]

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = DescrptDPA3(
        repflow={
            "n_dim": 128, "e_dim": 32, "a_dim": 16,
            "nlayers": 6,
            "e_rcut": 6.0, "e_rcut_smth": 5.0, "e_sel": 40,
            "a_rcut": 4.0, "a_rcut_smth": 3.5, "a_sel": 16,
            "axis_neuron": 4,
            "update_angle": True, "smooth_edge_update": True,
            "update_style": "res_residual", "update_residual": 0.1,
            "update_residual_init": "const",
            "n_experts": 8, "moe_top_k": 2,
            "use_node_moe": True, "use_edge_moe": True,
            "use_angle_moe": True,
            "share_expert": 0, "fuse_moe_mlps": False,
            "moe_ep_size": 2,
        },
        ntypes=2, precision="float32", seed=1,
        ep_group=ep_group,
    ).cuda()

    for _, p in model.named_parameters():
        dist.broadcast(p.data, src=0, group=ep_group)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Use SAME noise on both ranks to eliminate randomness as a factor
    torch.manual_seed(123)  # same seed on both ranks

    for step in range(110):
        sync_barrier(ep_group, rank, f"step_{step}_start")

        optimizer.zero_grad()
        # Same noise on both ranks
        noise = torch.randn(1, nall * 3, device="cuda", dtype=ext_coord.dtype) * 0.01
        coord_noisy = (ext_coord + noise).detach().requires_grad_(True)

        # Forward
        result = model(coord_noisy, ext_atype, nlist=nlist, mapping=mapping)
        sync_barrier(ep_group, rank, f"step_{step}_fwd")

        desc = result[0]
        energy = desc.sum()

        # Check energy sync
        e_val = energy.item()
        e_tensor = torch.tensor([e_val], device="cuda")
        e_other = torch.tensor([0.0], device="cuda")
        dist.all_reduce(e_tensor, group=ep_group)  # sum of both ranks
        e_avg = e_tensor.item() / 2.0
        if abs(e_val - e_avg) > 1e-2 and step < 5:
            print(f"[rank {rank}] step {step}: energy={e_val:.4f}, avg={e_avg:.4f}, "
                  f"diff={abs(e_val - e_avg):.6f}", flush=True)

        # 1st backward
        force = -torch.autograd.grad(
            energy, coord_noisy, create_graph=True, retain_graph=True
        )[0]
        sync_barrier(ep_group, rank, f"step_{step}_bwd1")

        loss = energy + force.sum() * 0.01
        loss_val = loss.item()

        # 2nd backward
        loss.backward()
        sync_barrier(ep_group, rank, f"step_{step}_bwd2")

        # Check grads
        grads_ok = check_grads(model, rank)

        # Gradient sync
        for name, p in model.named_parameters():
            if "experts" not in name:
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                dist.all_reduce(p.grad, group=ep_group)
                p.grad /= ep_size
        sync_barrier(ep_group, rank, f"step_{step}_grad_sync")

        optimizer.step()
        sync_barrier(ep_group, rank, f"step_{step}_optim")

        # Memory check
        mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
        mem_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024

        if step < 5 or step % 10 == 0:
            print(f"[rank {rank}] step {step}: loss={loss_val:.4f}, "
                  f"grads_ok={grads_ok}, "
                  f"mem={mem_mb:.0f}MB/{mem_reserved_mb:.0f}MB",
                  flush=True)

        # Check param sync every 20 steps
        if step % 20 == 0:
            params_synced = check_params_synced(model, ep_group, rank)
            if not params_synced:
                print(f"[rank {rank}] PARAMS OUT OF SYNC at step {step}!", flush=True)

    print(f"[rank {rank}] ALL 110 STEPS DONE", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
