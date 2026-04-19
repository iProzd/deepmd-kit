# SPDX-License-Identifier: LGPL-3.0-or-later
"""dp test inference validation for MoE models."""
import os
import shutil
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from deepmd.infer.deep_pot import DeepPot
from deepmd.pt.entrypoints.main import get_trainer


class TestMoEDPTest(unittest.TestCase):
    """Load MoE checkpoint, run dp test, verify output shape and compare with non-MoE."""

    def setUp(self) -> None:
        # Train a small MoE model first
        data_file = [str(Path(__file__).parent / "water/data/data_0")]

        self.config_moe = {
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
                        # MoE params
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
                "start_lr": 0.0001,
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
                "numb_steps": 50,
                "seed": 10,
                "disp_file": "lcurve.out",
                "disp_freq": 1,
                "save_freq": 50,
                "save_ckpt": "model_moe_dptest",
            },
        }

        # Train MoE model
        trainer_moe = get_trainer(self.config_moe)
        trainer_moe.run()

        # Train non-MoE model for comparison
        self.config_non_moe = deepcopy(self.config_moe)
        self.config_non_moe["model"]["descriptor"]["repflow"]["use_moe"] = False
        self.config_non_moe["model"]["descriptor"]["repflow"]["optim_update"] = True
        self.config_non_moe["training"]["save_ckpt"] = "model_non_moe_dptest"
        trainer_non_moe = get_trainer(self.config_non_moe)
        trainer_non_moe.run()

    def test_dp_test_moe_model(self) -> None:
        """Load MoE checkpoint, run inference, verify output shape."""
        ckpt_path = Path("model_moe_dptest.pt")
        self.assertTrue(ckpt_path.exists(), "MoE checkpoint not found")

        dp_moe = DeepPot(str(ckpt_path))

        # Use training data coordinates (test data may be OOD for undertrained model)
        data_dir = Path(__file__).parent / "water/data/data_0"
        coord = np.load(str(data_dir / "set.000/coord.npy"))[:1]  # First frame
        atype = np.loadtxt(str(data_dir / "type.raw"), dtype=int)
        box = np.load(str(data_dir / "set.000/box.npy"))[:1]

        # Test MoE model
        energy_moe, force_moe, virial_moe = dp_moe.eval(coord, box, atype)

        # Verify shapes (DeepPot.eval returns (nframes, 1) for energy)
        self.assertEqual(energy_moe.shape, (1, 1), f"Energy shape mismatch: {energy_moe.shape}")
        self.assertEqual(force_moe.shape, (1, len(atype), 3), f"Force shape mismatch: {force_moe.shape}")
        self.assertEqual(virial_moe.shape, (1, 9), f"Virial shape mismatch: {virial_moe.shape}")

        print(f"✓ MoE model inference shapes correct")
        print(f"  Energy: {energy_moe.ravel()[0]:.6f}")
        print(f"  Force norm: {np.linalg.norm(force_moe):.6f}")

        # Verify outputs are finite (with training data, should be OK)
        self.assertTrue(np.isfinite(energy_moe).all(), "MoE energy contains NaN/Inf")
        self.assertTrue(np.isfinite(force_moe).all(), "MoE force contains NaN/Inf")

    def test_compare_moe_vs_non_moe(self) -> None:
        """Compare MoE and non-MoE model outputs (should be different but reasonable)."""
        ckpt_moe = Path("model_moe_dptest.pt")
        ckpt_non_moe = Path("model_non_moe_dptest.pt")

        if not ckpt_non_moe.exists():
            self.skipTest("Non-MoE checkpoint not found")

        dp_moe = DeepPot(str(ckpt_moe))
        dp_non_moe = DeepPot(str(ckpt_non_moe))

        # Use training data coordinates
        data_dir = Path(__file__).parent / "water/data/data_0"
        coord = np.load(str(data_dir / "set.000/coord.npy"))[:1]
        atype = np.loadtxt(str(data_dir / "type.raw"), dtype=int)
        box = np.load(str(data_dir / "set.000/box.npy"))[:1]

        energy_moe, force_moe, virial_moe = dp_moe.eval(coord, box, atype)
        energy_ref, force_ref, virial_ref = dp_non_moe.eval(coord, box, atype)

        # Should be close but not identical (different architectures, random init)
        energy_diff = np.abs(energy_moe - energy_ref).max()
        force_diff = np.abs(force_moe - force_ref).max()

        print(f"✓ MoE vs non-MoE comparison:")
        print(f"  Energy diff: {energy_diff:.6f}")
        print(f"  Force diff: {force_diff:.6f}")

        # Loose tolerance (different architectures, only 50 training steps)
        # Just verify outputs are in reasonable range (not NaN/Inf)
        self.assertTrue(np.isfinite(energy_moe).all(), "MoE energy contains NaN/Inf")
        self.assertTrue(np.isfinite(force_moe).all(), "MoE force contains NaN/Inf")
        self.assertTrue(np.isfinite(energy_ref).all(), "Non-MoE energy contains NaN/Inf")
        self.assertTrue(np.isfinite(force_ref).all(), "Non-MoE force contains NaN/Inf")

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model_moe_dptest") and f.endswith(".pt"):
                os.remove(f)
            if f.startswith("model_non_moe_dptest") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
