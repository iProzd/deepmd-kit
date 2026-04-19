# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single-GPU MoE E2E training smoke tests."""
import json
import os
import shutil
import unittest
from copy import deepcopy
from pathlib import Path

import torch

from deepmd.pt.entrypoints.main import get_trainer


class TestMoESingleGPUSmoke(unittest.TestCase):
    """Single-GPU MoE training smoke test with small config."""

    def setUp(self) -> None:
        # Use existing water test data
        data_file = [str(Path(__file__).parent / "water/data/data_0")]

        self.config = {
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
                "numb_steps": 5,
                "seed": 10,
                "disp_file": "lcurve.out",
                "disp_freq": 1,
                "save_freq": 5,
                "save_ckpt": "model_moe_smoke",
            },
        }

    def test_moe_single_gpu_training(self) -> None:
        """Verify MoE model can train for a few steps on 1 GPU."""
        trainer = get_trainer(deepcopy(self.config))
        trainer.run()

        # Verify checkpoint saved
        ckpt_path = Path("model_moe_smoke.pt")
        self.assertTrue(ckpt_path.exists(), "Checkpoint not saved")

        # Verify checkpoint contains MoE params
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model"]

        # Check for routing expert keys
        routing_expert_keys = [k for k in state_dict if "routing_experts" in k]
        self.assertGreater(
            len(routing_expert_keys), 0, "No routing expert params in checkpoint"
        )

        # Check for shared expert keys
        shared_expert_keys = [k for k in state_dict if "shared_experts" in k]
        self.assertGreater(
            len(shared_expert_keys), 0, "No shared expert params in checkpoint"
        )

        # Check for router keys
        router_keys = [k for k in state_dict if "router" in k]
        self.assertGreater(len(router_keys), 0, "No router params in checkpoint")

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model_moe_smoke") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


if __name__ == "__main__":
    unittest.main()
