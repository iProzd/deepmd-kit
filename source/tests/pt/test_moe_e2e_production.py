# SPDX-License-Identifier: LGPL-3.0-or-later
"""Production-scale MoE E2E training test with reference config."""
import json
import os
import re
import shutil
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist

from deepmd.pt.entrypoints.main import get_trainer


def test_ep4_dp2_training_production():
    """8 GPU: EP=4, DP=2. Train with production config (64 experts, mptraj data)."""
    # Initialize distributed
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
            print("Skipping production test")
        dist.destroy_process_group()
        return

    with open(reference_input) as f:
        config = json.load(f)

    # Update MoE params (old → new format)
    repflow = config["model"]["descriptor"]["repflow"]
    repflow["use_moe"] = True
    repflow["n_routing_experts"] = 64  # old: n_experts
    repflow["moe_topk"] = 4  # old: moe_top_k
    repflow["n_shared_experts"] = 2  # old: share_expert

    # Remove old params
    repflow.pop("n_experts", None)
    repflow.pop("share_expert", None)
    repflow.pop("moe_top_k", None)
    repflow.pop("use_node_moe", None)
    repflow.pop("use_edge_moe", None)
    repflow.pop("use_angle_moe", None)

    # Add EP config and reduce steps for testing
    config["training"]["moe_ep_size"] = 4
    config["training"]["numb_steps"] = 100
    config["training"]["save_ckpt"] = "model_moe_production"

    trainer = get_trainer(deepcopy(config))
    trainer.run()

    # Verify checkpoint
    if rank == 0:
        ckpt_path = Path("model_moe_production.pt")
        assert ckpt_path.exists(), "Checkpoint not saved"

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model"]

        # Verify all 64 routing experts are present
        expert_keys = [k for k in state_dict if "routing_experts" in k]
        assert len(expert_keys) > 0, "No routing expert params in checkpoint"

        expert_indices = set()
        for k in expert_keys:
            m = re.search(r"\.routing_experts\.(\d+)\.", k)
            if m:
                expert_indices.add(int(m.group(1)))

        expected_experts = set(range(64))
        missing = expected_experts - expert_indices
        assert (
            expert_indices == expected_experts
        ), f"Missing experts: {missing}, got {len(expert_indices)}/64"

        print(f"✓ Checkpoint contains all 64 routing experts")

    # Cleanup
    dist.barrier()
    if rank == 0:
        for f in os.listdir("."):
            if f.startswith("model_moe_production") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f, ignore_errors=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    test_ep4_dp2_training_production()
