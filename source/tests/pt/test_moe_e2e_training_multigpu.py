# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-GPU MoE E2E training tests with EP+DP."""
import os
import re
import shutil
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist

from deepmd.pt.entrypoints.main import get_trainer


def test_ep4_dp2_training_small():
    """8 GPU: EP=4, DP=2. Train for 10 steps with small config."""
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8, f"This test requires 8 GPUs, got {world_size}"

    # Use existing water test data
    test_dir = Path(__file__).parent
    data_file = [str(test_dir / "water/data/data_0")]

    config = {
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
                    # MoE params (small scale for fast testing)
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
            "numb_steps": 10,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": 1,
            "save_freq": 10,
            "save_ckpt": "model_moe_ep4_dp2",
            "moe_ep_size": 4,  # NEW: EP group size
        },
    }

    trainer = get_trainer(deepcopy(config))
    trainer.run()

    # Verify checkpoint saved (rank 0 only)
    if rank == 0:
        ckpt_path = Path("model_moe_ep4_dp2.pt")
        assert ckpt_path.exists(), "Checkpoint not saved"

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model"]

        # Verify routing experts are global (not local)
        # Should have routing_experts.0 through routing_experts.3
        expert_keys = [k for k in state_dict if "routing_experts" in k]
        assert len(expert_keys) > 0, "No routing expert params in checkpoint"

        # Extract expert indices
        expert_indices = set()
        for k in expert_keys:
            m = re.search(r"\.routing_experts\.(\d+)\.", k)
            if m:
                expert_indices.add(int(m.group(1)))

        # Should have all 4 experts (0, 1, 2, 3)
        expected_experts = set(range(4))
        assert (
            expert_indices == expected_experts
        ), f"Missing experts: {expected_experts - expert_indices}"

        print(f"✓ Checkpoint contains all {len(expected_experts)} routing experts")

    # Cleanup
    dist.barrier()
    if rank == 0:
        for f in os.listdir("."):
            if f.startswith("model_moe_ep4_dp2") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f, ignore_errors=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    test_ep4_dp2_training_small()
