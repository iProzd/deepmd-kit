# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from pathlib import (
    Path,
)

import numpy as np

# NOTE: avoid torch thread reconfiguration errors during import.
import torch

torch_set_num_interop_threads = getattr(torch, "set_num_interop_threads", None)
torch_set_num_threads = getattr(torch, "set_num_threads", None)
if torch_set_num_interop_threads is not None:
    torch.set_num_interop_threads = lambda *args, **kwargs: None  # type: ignore[assignment]
if torch_set_num_threads is not None:
    torch.set_num_threads = lambda *args, **kwargs: None  # type: ignore[assignment]

from deepmd.pt.model.model import (
    get_sezm_model,
)
from deepmd.pt.model.model.sezm_model import (
    InterPotential,
)
from deepmd.pt.utils import (
    env,
)


class TestSeZMModelCompile(unittest.TestCase):
    """Test SeZM model compile path consistency."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(2024)

    def _build_model_params(self, *, use_compile: bool, n_node: int) -> dict:
        return {
            "type": "SeZM",
            "type_map": ["A", "B"],
            "descriptor": {
                "type": "SeZM",
                "sel": [2, 2],
                "rcut": 3.0,
                "channels": 4,
                "n_focus": 1,
                "focus_compete": False,
                "n_radial": 3,
                "radial_mlp": [6],
                "use_env_seed": False,
                "l_schedule": [1, 0],
                "mmax": 1,
                "so2_norm": False,
                "so2_layers": 1,
                "n_atten_head": 0,
                "sandwich_norm": [True, False, True, False],
                "ffn_neurons": 8,
                "ffn_blocks": 2,
                "mlp_bias": True,
                "layer_scale": True,
                "use_amp": False,
                "use_triton": False,
                "activation_function": "silu",
                "glu_activation": True,
                "precision": "float32",
                "seed": 7,
            },
            "fitting_net": {
                "neuron": [8],
                "activation_function": "silu",
                "precision": "float32",
                "seed": 7,
            },
            "use_compile": use_compile,
            "n_node": n_node,
        }

    def _load_water_frame(
        self,
        nframe: int = 1,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Load frames from dplr dataset with virial data.

        Parameters
        ----------
        nframe
            Number of frames to load.

        Returns
        -------
        coord : torch.Tensor
            Coordinates with shape (nframe, nloc, 3).
        atype : torch.Tensor
            Atom types with shape (nframe, nloc).
        box : torch.Tensor
            Box tensor with shape (nframe, 9).
        energy : torch.Tensor
            Energy with shape (nframe, 1).
        force : torch.Tensor
            Forces with shape (nframe, nloc, 3).
        virial : torch.Tensor
            Virial tensor with shape (nframe, 9).
        """
        if nframe <= 0:
            raise ValueError("nframe must be positive")

        # Use dplr dataset which contains virial data
        data_root = (
            Path(__file__).parent.parent.parent.parent.parent
            / "examples"
            / "water"
            / "dplr"
            / "train"
            / "data"
        )
        set_dir = data_root / "set.000"

        coord_np = np.load(set_dir / "coord.npy")
        force_np = np.load(set_dir / "force.npy")
        energy_np = np.load(set_dir / "energy.npy")
        box_np = np.load(set_dir / "box.npy")
        virial_np = np.load(set_dir / "virial.npy")
        atype_np = np.loadtxt(data_root / "type.raw", dtype=np.int32).reshape(1, -1)

        coord = torch.from_numpy(coord_np[:nframe].reshape(nframe, -1, 3)).to(
            device=self.device, dtype=torch.float32
        )
        force = torch.from_numpy(force_np[:nframe].reshape(nframe, -1, 3)).to(
            device=self.device, dtype=torch.float32
        )
        energy = torch.from_numpy(energy_np[:nframe].reshape(nframe, 1)).to(
            device=self.device, dtype=torch.float32
        )
        box = torch.from_numpy(box_np[:nframe]).to(
            device=self.device, dtype=torch.float32
        )
        virial = torch.from_numpy(virial_np[:nframe]).to(
            device=self.device, dtype=torch.float32
        )
        atype = torch.from_numpy(np.repeat(atype_np, nframe, axis=0)).to(
            device=self.device, dtype=torch.int32
        )
        return coord, atype, box, energy, force, virial

    def _train_steps(
        self,
        model: torch.nn.Module,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor,
        energy: torch.Tensor,
        force: torch.Tensor,
        virial: torch.Tensor | None = None,
        steps: int = 3,
    ) -> dict[str, torch.Tensor]:
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-7)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            out = model(coord, atype, box=box)
            loss_energy = torch.mean(
                (out["energy"] - energy.to(out["energy"].dtype)) ** 2
            )
            loss_force = torch.mean((out["force"] - force.to(out["force"].dtype)) ** 2)
            loss = loss_energy + loss_force
            if virial is not None and "virial" in out:
                loss_virial = torch.mean(
                    (out["virial"] - virial.to(out["virial"].dtype)) ** 2
                )
                loss = loss + loss_virial
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        return {
            name: param.detach().clone() for name, param in model.named_parameters()
        }

    def test_three_step_training_matches_compile(self) -> None:
        """Train three steps and compare parameters between dynamic and compile paths."""
        coord, atype, box, energy, force, _virial = self._load_water_frame()

        # === Step 1. Build paired models with shared weights ===
        model_dyn = get_sezm_model(
            self._build_model_params(use_compile=False, n_node=512)
        )
        model_cmp = get_sezm_model(
            self._build_model_params(use_compile=True, n_node=512)
        )
        model_cmp.load_state_dict(model_dyn.state_dict())
        model_dyn.train()
        model_cmp.train()
        self.assertTrue(model_cmp.use_compile)

        # === Step 2. Three-step training (first step triggers compile) ===
        params_dyn = self._train_steps(
            model_dyn, coord, atype, box, energy, force, _virial
        )
        params_cmp = self._train_steps(
            model_cmp, coord, atype, box, energy, force, _virial
        )

        # === Step 3. Compare parameters ===
        self.assertEqual(set(params_dyn.keys()), set(params_cmp.keys()))
        for name in params_dyn.keys():
            torch.testing.assert_close(
                params_dyn[name], params_cmp[name], atol=1.0e-7, rtol=1.0e-7
            )

    def test_force_matches_compile(self) -> None:
        """Single forward force should match between dynamic and compile paths."""
        coord, atype, box, _, _, _ = self._load_water_frame()

        # === Step 1. Build paired models with shared weights ===
        model_dyn = get_sezm_model(
            self._build_model_params(use_compile=False, n_node=512)
        )
        model_cmp = get_sezm_model(
            self._build_model_params(use_compile=True, n_node=512)
        )
        model_cmp.load_state_dict(model_dyn.state_dict())
        model_dyn.eval()
        model_cmp.eval()

        # === Step 2. Forward and compare forces ===
        out_dyn = model_dyn(coord, atype, box=box)
        out_cmp = model_cmp(coord, atype, box=box)
        torch.testing.assert_close(
            out_dyn["force"], out_cmp["force"], atol=1.0e-6, rtol=1.0e-6
        )

    def test_multi_frame_energy_matches_compile(self) -> None:
        """Energy output must remain per-frame in compile path."""
        nframe = 2
        coord, atype, box, _, _, _ = self._load_water_frame(nframe=nframe)
        n_node = int(coord.shape[0] * coord.shape[1] + 64)

        # === Step 1. Build paired models with shared weights ===
        model_dyn = get_sezm_model(
            self._build_model_params(use_compile=False, n_node=n_node)
        )
        model_cmp = get_sezm_model(
            self._build_model_params(use_compile=True, n_node=n_node)
        )
        model_cmp.load_state_dict(model_dyn.state_dict())
        model_dyn.train()
        model_cmp.train()

        # === Step 2. Forward and compare per-frame energy ===
        out_dyn = model_dyn(coord, atype, box=box)
        out_cmp = model_cmp(coord, atype, box=box)

        self.assertEqual(out_dyn["energy"].shape, (nframe, 1))
        self.assertEqual(out_cmp["energy"].shape, (nframe, 1))
        torch.testing.assert_close(
            out_dyn["energy"], out_cmp["energy"], atol=1.0e-6, rtol=1.0e-6
        )

    def test_virial_matches_compile(self) -> None:
        """Single forward virial should match between dynamic and compile paths."""
        coord, atype, box, _, _, _ = self._load_water_frame()

        # === Step 1. Build paired models with shared weights ===
        model_dyn = get_sezm_model(
            self._build_model_params(use_compile=False, n_node=512)
        )
        model_cmp = get_sezm_model(
            self._build_model_params(use_compile=True, n_node=512)
        )
        model_cmp.load_state_dict(model_dyn.state_dict())
        model_dyn.eval()
        model_cmp.eval()

        # === Step 2. Forward and compare virials ===
        out_dyn = model_dyn(coord, atype, box=box)
        out_cmp = model_cmp(coord, atype, box=box)

        # Check virial exists in both outputs
        self.assertIn("virial", out_dyn, "Virial not in dynamic model output")
        self.assertIn("virial", out_cmp, "Virial not in compile model output")

        # Compare virial values
        torch.testing.assert_close(
            out_dyn["virial"], out_cmp["virial"], atol=1.0e-5, rtol=1.0e-5
        )

    def test_forward_backward_double_backward_matches_compile(self) -> None:
        """
        Check forward, backward, and double backward consistency vs compile.

        Forward: energy/force outputs should match.
        Backward: d(energy)/d(params) should match.
        Double backward: d(force_loss)/d(params) should match.
        """
        coord, atype, box, _, _, _ = self._load_water_frame()

        # === Step 1. Build paired models with shared weights ===
        model_dyn = get_sezm_model(
            self._build_model_params(use_compile=False, n_node=512)
        )
        model_cmp = get_sezm_model(
            self._build_model_params(use_compile=True, n_node=512)
        )
        model_cmp.load_state_dict(model_dyn.state_dict())
        model_dyn.train()
        model_cmp.train()

        # === Step 2. Forward output consistency ===
        out_dyn = model_dyn(coord, atype, box=box)
        out_cmp = model_cmp(coord, atype, box=box)
        torch.testing.assert_close(
            out_dyn["energy"], out_cmp["energy"], atol=1.0e-6, rtol=1.0e-6
        )
        torch.testing.assert_close(
            out_dyn["force"], out_cmp["force"], atol=1.0e-6, rtol=1.0e-6
        )

        # === Step 3. Backward on energy ===
        model_dyn.zero_grad(set_to_none=True)
        model_cmp.zero_grad(set_to_none=True)
        loss_dyn = out_dyn["energy"].sum()
        loss_cmp = out_cmp["energy"].sum()
        loss_dyn.backward()
        loss_cmp.backward()
        grads_dyn = {
            name: (
                torch.zeros_like(param) if param.grad is None else param.grad.detach()
            )
            for name, param in model_dyn.named_parameters()
        }
        grads_cmp = {
            name: (
                torch.zeros_like(param) if param.grad is None else param.grad.detach()
            )
            for name, param in model_cmp.named_parameters()
        }
        # Inductor Triton kernels use different reduction order vs eager,
        # so float32 gradients can differ by ~1e-3 on GPU.
        grad_atol = 1.0e-5 if self.device == torch.device("cpu") else 2.0e-3
        grad_rtol = 1.0e-5 if self.device == torch.device("cpu") else 1.0e-4
        self.assertEqual(set(grads_dyn.keys()), set(grads_cmp.keys()))
        for name in grads_dyn.keys():
            torch.testing.assert_close(
                grads_dyn[name], grads_cmp[name], atol=grad_atol, rtol=grad_rtol
            )

        # === Step 4. Double backward via force loss ===
        model_dyn.zero_grad(set_to_none=True)
        model_cmp.zero_grad(set_to_none=True)
        out_dyn = model_dyn(coord, atype, box=box)
        out_cmp = model_cmp(coord, atype, box=box)
        loss_dyn = torch.sum(out_dyn["force"] * out_dyn["force"])
        loss_cmp = torch.sum(out_cmp["force"] * out_cmp["force"])
        loss_dyn.backward()
        loss_cmp.backward()
        grads_dyn = {
            name: (
                torch.zeros_like(param) if param.grad is None else param.grad.detach()
            )
            for name, param in model_dyn.named_parameters()
        }
        grads_cmp = {
            name: (
                torch.zeros_like(param) if param.grad is None else param.grad.detach()
            )
            for name, param in model_cmp.named_parameters()
        }
        self.assertEqual(set(grads_dyn.keys()), set(grads_cmp.keys()))
        for name in grads_dyn.keys():
            torch.testing.assert_close(
                grads_dyn[name], grads_cmp[name], atol=grad_atol, rtol=grad_rtol
            )


class TestInterPotential(unittest.TestCase):
    """Test InterPotential ZBL analytical pair potential."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_zbl_known_value_OO(self) -> None:
        """Test ZBL energy for O-O pair at known distance against reference."""
        pot = InterPotential(type_map=["O", "H"], mode="ZBL").to(self.device)

        import math

        z_o = 8.0
        a_bohr = 0.5291772109
        ke = 14.3996
        a_screen = 0.88534 * a_bohr / (z_o**0.23 + z_o**0.23)
        r = 1.0
        x = r / a_screen
        phi = (
            0.18175 * math.exp(-3.1998 * x)
            + 0.50986 * math.exp(-0.94229 * x)
            + 0.28022 * math.exp(-0.4029 * x)
            + 0.028171 * math.exp(-0.20162 * x)
        )
        expected = ke * z_o * z_o / r * phi

        extended_coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=torch.float64,
            device=self.device,
        )
        extended_atype = torch.tensor([[0, 0]], dtype=torch.int64, device=self.device)
        nlist = torch.tensor([[[1], [0]]], dtype=torch.int64, device=self.device)

        pair_e = pot(extended_coord, extended_atype, nlist, nloc=2)
        total_e = pair_e.sum().item()
        self.assertAlmostEqual(total_e, expected, places=5)

    def test_zbl_known_value_OH(self) -> None:
        """Test ZBL energy for O-H pair at known distance."""
        pot = InterPotential(type_map=["O", "H"], mode="ZBL").to(self.device)
        import math

        z_o, z_h = 8.0, 1.0
        a_bohr = 0.5291772109
        ke = 14.3996
        a_screen = 0.88534 * a_bohr / (z_o**0.23 + z_h**0.23)
        r = 0.8
        x = r / a_screen
        phi = (
            0.18175 * math.exp(-3.1998 * x)
            + 0.50986 * math.exp(-0.94229 * x)
            + 0.28022 * math.exp(-0.4029 * x)
            + 0.028171 * math.exp(-0.20162 * x)
        )
        expected = ke * z_o * z_h / r * phi

        extended_coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]],
            dtype=torch.float64,
            device=self.device,
        )
        extended_atype = torch.tensor([[0, 1]], dtype=torch.int64, device=self.device)
        nlist = torch.tensor([[[1], [0]]], dtype=torch.int64, device=self.device)

        pair_e = pot(extended_coord, extended_atype, nlist, nloc=2)
        total_e = pair_e.sum().item()
        self.assertAlmostEqual(total_e, expected, places=5)

    def test_zbl_gradient_exists(self) -> None:
        """Test that ZBL potential produces valid gradients for force computation."""
        pot = InterPotential(type_map=["O", "H"], mode="ZBL").to(self.device)

        extended_coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        extended_atype = torch.tensor([[0, 1]], dtype=torch.int64, device=self.device)
        nlist = torch.tensor([[[1], [0]]], dtype=torch.int64, device=self.device)

        pair_e = pot(extended_coord, extended_atype, nlist, nloc=2)
        pair_e.sum().backward()
        self.assertIsNotNone(extended_coord.grad)
        self.assertTrue(torch.isfinite(extended_coord.grad).all())

    def test_unknown_element_raises(self) -> None:
        """Test that unknown element raises ValueError."""
        with self.assertRaises(ValueError):
            InterPotential(type_map=["O", "Xx"])

    def test_forward_from_edges(self) -> None:
        """Test the compile-path edge-based ZBL computation."""
        pot = InterPotential(type_map=["O", "H"], mode="ZBL").to(self.device)

        edge_vec = torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=self.device,
        )
        edge_index = torch.tensor(
            [[1, 0], [0, 1]], dtype=torch.long, device=self.device
        )
        atype_flat = torch.tensor([0, 1], dtype=torch.long, device=self.device)
        edge_mask = torch.tensor([True, True], device=self.device)

        result = pot.forward_from_edges(edge_vec, edge_index, atype_flat, edge_mask, 2)
        self.assertEqual(result.shape, (1, 2, 1))
        self.assertTrue(torch.isfinite(result).all())

        extended_coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=torch.float64,
            device=self.device,
        )
        extended_atype = torch.tensor([[0, 1]], dtype=torch.int64, device=self.device)
        nlist = torch.tensor([[[1], [0]]], dtype=torch.int64, device=self.device)
        pair_e_nlist = pot(extended_coord, extended_atype, nlist, nloc=2)
        torch.testing.assert_close(
            result.sum(), pair_e_nlist.sum().to(result.dtype), atol=1e-8, rtol=1e-8
        )


class TestSeZMModelBridging(unittest.TestCase):
    """Test SeZM model with ZBL bridging enabled."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(2024)

    def _build_model_params(self, *, bridging_method: str = "none") -> dict:
        return {
            "type": "SeZM",
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "SeZM",
                "sel": [2, 2],
                "rcut": 3.0,
                "channels": 4,
                "n_focus": 1,
                "focus_compete": False,
                "n_radial": 3,
                "radial_mlp": [6],
                "use_env_seed": False,
                "l_schedule": [1, 0],
                "mmax": 1,
                "so2_norm": False,
                "so2_layers": 1,
                "n_atten_head": 0,
                "sandwich_norm": [True, False, True, False],
                "ffn_neurons": 8,
                "ffn_blocks": 1,
                "mlp_bias": True,
                "layer_scale": True,
                "use_amp": False,
                "activation_function": "silu",
                "glu_activation": True,
                "precision": "float32",
                "seed": 7,
            },
            "fitting_net": {
                "neuron": [8],
                "activation_function": "silu",
                "precision": "float32",
                "seed": 7,
            },
            "use_compile": False,
            "n_node": 256,
            "bridging_method": bridging_method,
            "bridging_r_inner": 1.0,
            "bridging_r_outer": 1.5,
        }

    def test_bridging_none_unchanged(self) -> None:
        """Test that bridging_method='none' produces no inter_potential."""
        model = get_sezm_model(self._build_model_params(bridging_method="none"))
        self.assertIsNone(model.inter_potential)
        self.assertEqual(model.bridging_method, "NONE")

    def test_bridging_zbl_creates_potential(self) -> None:
        """Test that bridging_method='ZBL' creates InterPotential and InnerClamp."""
        model = get_sezm_model(self._build_model_params(bridging_method="ZBL"))
        self.assertIsNotNone(model.inter_potential)
        self.assertEqual(model.bridging_method, "ZBL")
        self.assertIsNotNone(model.atomic_model.descriptor.inner_clamp)

    def test_zbl_adds_energy(self) -> None:
        """Test that ZBL bridging adds energy to the model output."""
        model_plain = get_sezm_model(self._build_model_params(bridging_method="none"))
        model_zbl = get_sezm_model(self._build_model_params(bridging_method="ZBL"))

        sd = model_plain.state_dict()
        model_zbl.load_state_dict(sd, strict=False)

        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 2.0, 0.0]]],
            dtype=torch.float32,
            device=self.device,
        )
        atype = torch.tensor([[0, 1, 0]], dtype=torch.int32, device=self.device)
        box = torch.tensor(
            [[10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]],
            dtype=torch.float32,
            device=self.device,
        )

        model_plain.eval()
        model_zbl.eval()

        out_plain = model_plain(coord, atype, box=box)
        out_zbl = model_zbl(coord, atype, box=box)

        energy_diff = (out_zbl["energy"] - out_plain["energy"]).item()
        self.assertGreater(
            energy_diff,
            0.0,
            "ZBL bridging should add positive (repulsive) energy",
        )
