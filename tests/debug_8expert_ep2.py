#!/usr/bin/env python3
"""Debug 8-expert EP: trace all NCCL ops with detailed splits info."""

import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import numpy as np

# Patch both the A2A differentiable and the dispatch info
_original_a2a_fn = None
_original_dispatch_fn = None
_nccl_op_log = []

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

def main():
    rank, local_rank = setup()
    mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("dp", "ep"))
    ep_group = mesh["ep"].get_group()

    # Monkey-patch dist.all_to_all_single to count ALL NCCL A2A ops
    original_a2a_single = dist.all_to_all_single
    a2a_single_count = [0]

    def counting_a2a_single(output, input, *args, **kwargs):
        a2a_single_count[0] += 1
        return original_a2a_single(output, input, *args, **kwargs)

    dist.all_to_all_single = counting_a2a_single

    # Also count all_reduce
    original_all_reduce = dist.all_reduce
    all_reduce_count = [0]

    def counting_all_reduce(tensor, *args, **kwargs):
        all_reduce_count[0] += 1
        return original_all_reduce(tensor, *args, **kwargs)

    dist.all_reduce = counting_all_reduce

    from deepmd.pt.model.descriptor.dpa3 import DescrptDPA3

    ext_coord, ext_atype, nlist, mapping, nloc = make_water_system(8, 6.0, 40)
    dist.all_to_all_single = original_a2a_single  # restore during broadcast
    dist.all_reduce = original_all_reduce
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

    # Re-enable counting
    dist.all_to_all_single = counting_a2a_single
    dist.all_reduce = counting_all_reduce

    for step in range(110):
        a2a_single_count[0] = 0
        all_reduce_count[0] = 0

        optimizer.zero_grad()
        noise = torch.randn(1, nall * 3, device="cuda", dtype=ext_coord.dtype) * 0.01
        coord_noisy = (ext_coord + noise).detach().requires_grad_(True)

        result = model(coord_noisy, ext_atype, nlist=nlist, mapping=mapping)
        fwd_a2a = a2a_single_count[0]
        a2a_single_count[0] = 0

        desc = result[0]
        energy = desc.sum()
        force = -torch.autograd.grad(
            energy, coord_noisy, create_graph=True, retain_graph=True
        )[0]
        bwd1_a2a = a2a_single_count[0]
        a2a_single_count[0] = 0

        loss = energy + force.sum() * 0.01
        loss.backward()
        bwd2_a2a = a2a_single_count[0]
        a2a_single_count[0] = 0

        # Gradient sync
        ep_size = dist.get_world_size(group=ep_group)
        for name, p in model.named_parameters():
            if "experts" not in name:
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                dist.all_reduce(p.grad, group=ep_group)
                p.grad /= ep_size
        grad_sync_allreduce = all_reduce_count[0]
        all_reduce_count[0] = 0

        total = fwd_a2a + bwd1_a2a + bwd2_a2a
        if step < 3 or step % 20 == 0:
            print(f"[rank {rank}] step {step}: fwd_a2a={fwd_a2a}, bwd1_a2a={bwd1_a2a}, "
                  f"bwd2_a2a={bwd2_a2a}, total_a2a={total}, grad_allreduce={grad_sync_allreduce}",
                  flush=True)

        optimizer.step()
        # Sync barrier every step
        torch.cuda.synchronize()
        dist.barrier(group=ep_group)

    print(f"[rank {rank}] ALL 110 STEPS DONE", flush=True)
    dist.all_to_all_single = original_a2a_single
    dist.all_reduce = original_all_reduce
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
