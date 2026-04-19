# SPDX-License-Identifier: LGPL-3.0-or-later
"""MoE EP+DP training speed benchmark."""
import json
import os
import shutil
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist

from deepmd.pt.entrypoints.main import get_trainer


def benchmark_ep4_dp2_vs_dp8():
    """Benchmark: EP=4 DP=2 vs EP=1 DP=8 on same model.

    Measures wall-clock time for N training steps. EP should be faster due to
    reduced All-Reduce volume (routing expert grads only sync within DP group).
    """
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8, f"This test requires 8 GPUs, got {world_size}"

    # Use existing water test data
    test_dir = Path(__file__).parent
    data_file = [str(test_dir / "water/data/data_0")]

    n_steps = 50  # Number of steps to benchmark

    base_config = {
        "model": {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "dpa3",
                "repflow": {
                    "n_dim": 16,
                    "e_dim": 8,
                    "a_dim": 4,
                    "nlayers": 2,
                    "e_rcut": 6.0,
                    "e_rcut_smth": 5.0,
                    "e_sel": 20,
                    "a_rcut": 4.0,
                    "a_rcut_smth": 3.5,
                    "a_sel": 10,
                    "axis_neuron": 4,
                    "a_compress_rate": 1,
                    "a_compress_e_rate": 2,
                    "a_compress_use_split": True,
                    "update_angle": True,
                    "update_style": "res_residual",
                    "update_residual": 0.1,
                    "update_residual_init": "const",
                    "smooth_edge_update": True,
                    "use_dynamic_sel": True,  # Required for MoE
                    "sel_reduce_factor": 10.0,
                    "optim_update": False,  # Required for MoE
                    "use_moe": True,
                    "n_routing_experts": 4,
                    "moe_topk": 2,
                    "n_shared_experts": 1,
                },
                "activation_function": "silu",
                "use_tebd_bias": False,
                "precision": "float32",
                "concat_output_tebd": False,
            },
            "fitting_net": {
                "neuron": [24, 24],
                "resnet_dt": True,
                "precision": "float32",
                "activation_function": "silu",
                "seed": 1,
            },
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 100,
            "start_lr": 0.001,
            "stop_lr": 1e-6,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0,
        },
        "optimizer": {
        "type": "Adam",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "weight_decay": 0.0,
    },
    "training": {
            "training_data": {
                "systems": data_file,
                "batch_size": 1,
            },
            "validation_data": {
                "systems": data_file,
                "batch_size": 1,
                "numb_btch": 1,
            },
            "numb_steps": n_steps,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": n_steps + 1,  # Don't print during benchmark
            "save_freq": n_steps + 1,  # Don't save during benchmark
        },
    }

    # === Benchmark 1: EP=4, DP=2 ===
    config_ep4 = deepcopy(base_config)
    config_ep4["training"]["moe_ep_size"] = 4
    config_ep4["training"]["save_ckpt"] = "model_bench_ep4"

    # Warmup
    if rank == 0:
        print(f"Warming up EP=4 DP=2...")
    trainer_ep4 = get_trainer(deepcopy(config_ep4))
    dist.barrier()
    torch.cuda.synchronize()

    start_ep4 = time.time()
    trainer_ep4.run()
    torch.cuda.synchronize()
    time_ep4 = time.time() - start_ep4
    del trainer_ep4

    # === Benchmark 2: EP=1, DP=8 (standard DDP, all experts replicated) ===
    config_dp8 = deepcopy(base_config)
    config_dp8["training"]["moe_ep_size"] = 1  # No EP, standard DDP
    config_dp8["training"]["save_ckpt"] = "model_bench_dp8"

    # Warmup
    if rank == 0:
        print(f"Warming up EP=1 DP=8...")
    trainer_dp8 = get_trainer(deepcopy(config_dp8))
    dist.barrier()
    torch.cuda.synchronize()

    start_dp8 = time.time()
    trainer_dp8.run()
    torch.cuda.synchronize()
    time_dp8 = time.time() - start_dp8
    del trainer_dp8

    if rank == 0:
        steps_per_sec_ep4 = n_steps / time_ep4
        steps_per_sec_dp8 = n_steps / time_dp8
        speedup = time_dp8 / time_ep4

        print(f"\n{'='*60}")
        print(f"MoE Training Speed Benchmark Results")
        print(f"{'='*60}")
        print(f"Steps: {n_steps}")
        print(f"  EP=4, DP=2: {time_ep4:.2f}s ({steps_per_sec_ep4:.1f} steps/s)")
        print(f"  EP=1, DP=8: {time_dp8:.2f}s ({steps_per_sec_dp8:.1f} steps/s)")
        print(f"  Speedup (EP over DP): {speedup:.2f}x")
        print(f"{'='*60}")

    # Cleanup
    dist.barrier()
    if rank == 0:
        for f in os.listdir("."):
            if f.startswith("model_bench_") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f, ignore_errors=True)

    dist.destroy_process_group()


def benchmark_production_scale():
    """Benchmark with production-scale config (64 experts, mptraj)."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8, f"This test requires 8 GPUs, got {world_size}"

    # Load reference config
    reference_input = Path(
        "/mnt/data_nas/zhangd/workplace/dev26/0403_debug_moe/new_moe_EP8/input.json"
    )
    if not reference_input.exists():
        if rank == 0:
            print(f"⚠ Reference input not found: {reference_input}")
            print("Skipping production benchmark")
        dist.destroy_process_group()
        return

    with open(reference_input) as f:
        config = json.load(f)

    # Update MoE params (old → new format)
    repflow = config["model"]["descriptor"]["repflow"]
    repflow["use_moe"] = True
    repflow["n_routing_experts"] = 64
    repflow["moe_topk"] = 4
    repflow["n_shared_experts"] = 2
    repflow.pop("n_experts", None)
    repflow.pop("share_expert", None)
    repflow.pop("moe_top_k", None)
    repflow.pop("use_node_moe", None)
    repflow.pop("use_edge_moe", None)
    repflow.pop("use_angle_moe", None)

    n_steps = 50
    config["training"]["numb_steps"] = n_steps
    config["training"]["disp_freq"] = n_steps + 1
    config["training"]["save_freq"] = n_steps + 1

    # === EP=4, DP=2 ===
    config_ep4 = deepcopy(config)
    config_ep4["training"]["moe_ep_size"] = 4
    config_ep4["training"]["save_ckpt"] = "model_prod_bench_ep4"

    if rank == 0:
        print(f"Production benchmark EP=4 DP=2 ({n_steps} steps)...")
    trainer_ep4 = get_trainer(deepcopy(config_ep4))
    dist.barrier()
    torch.cuda.synchronize()
    start_ep4 = time.time()
    trainer_ep4.run()
    torch.cuda.synchronize()
    time_ep4 = time.time() - start_ep4
    del trainer_ep4

    # === EP=1, DP=8 ===
    config_dp8 = deepcopy(config)
    config_dp8["training"]["moe_ep_size"] = 1
    config_dp8["training"]["save_ckpt"] = "model_prod_bench_dp8"

    if rank == 0:
        print(f"Production benchmark EP=1 DP=8 ({n_steps} steps)...")
    trainer_dp8 = get_trainer(deepcopy(config_dp8))
    dist.barrier()
    torch.cuda.synchronize()
    start_dp8 = time.time()
    trainer_dp8.run()
    torch.cuda.synchronize()
    time_dp8 = time.time() - start_dp8
    del trainer_dp8

    if rank == 0:
        speedup = time_dp8 / time_ep4
        print(f"\n{'='*60}")
        print(f"Production MoE Training Speed Benchmark")
        print(f"{'='*60}")
        print(f"Config: 64 experts, topk=4, shared=2, 6 layers")
        print(f"Steps: {n_steps}")
        print(f"  EP=4, DP=2: {time_ep4:.2f}s ({n_steps/time_ep4:.1f} steps/s)")
        print(f"  EP=1, DP=8: {time_dp8:.2f}s ({n_steps/time_dp8:.1f} steps/s)")
        print(f"  Speedup (EP over DP): {speedup:.2f}x")
        print(f"{'='*60}")

    # Cleanup
    dist.barrier()
    if rank == 0:
        for f in os.listdir("."):
            if f.startswith("model_prod_bench_") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f, ignore_errors=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    import sys

    if "--production" in sys.argv:
        benchmark_production_scale()
    else:
        benchmark_ep4_dp2_vs_dp8()
