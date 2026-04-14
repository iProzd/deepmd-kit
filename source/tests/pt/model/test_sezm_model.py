# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import tempfile
import unittest
from pathlib import (
    Path,
)
from unittest import (
    mock,
)

import h5py
import numpy as np

# NOTE: avoid torch thread reconfiguration errors during import.
import torch

torch_set_num_interop_threads = getattr(torch, "set_num_interop_threads", None)
torch_set_num_threads = getattr(torch, "set_num_threads", None)
if torch_set_num_interop_threads is not None:
    torch.set_num_interop_threads = lambda *args, **kwargs: None  # type: ignore[assignment]
if torch_set_num_threads is not None:
    torch.set_num_threads = lambda *args, **kwargs: None  # type: ignore[assignment]

from deepmd.pt.loss import (
    DeNSLoss,
    EnergyStdLoss,
)
from deepmd.pt.model.descriptor.sezm_nn import (
    build_edge_cache,
    build_edge_cache_from_edges,
)
from deepmd.pt.model.model import (
    get_sezm_model,
)
from deepmd.pt.model.model.sezm_model import (
    InterPotential,
)
from deepmd.pt.train.training import (
    prepare_model_for_loss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.utils.path import (
    DPPath,
)


class TestSeZMModelCompile(unittest.TestCase):
    """Test SeZM model compile path consistency."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(2024)

    def _build_model_params(self, *, use_compile: bool) -> dict:
        return {
            "type": "SeZM",
            "type_map": ["A", "B"],
            "descriptor": {
                "type": "SeZM",
                "sel": [2, 2],
                "rcut": 3.0,
                "channels": 4,
                "n_focus": 1,
                "n_radial": 3,
                "radial_mlp": [6],
                "use_env_seed": True,
                "l_schedule": [1, 0],
                "mmax": 1,
                "so2_norm": False,
                "so2_layers": 1,
                "n_atten_head": 1,
                "sandwich_norm": [True, False, True, False],
                "ffn_neurons": 8,
                "ffn_blocks": 1,
                "s2_activation": [False, True],
                "mlp_bias": False,
                "layer_scale": False,
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
            "use_compile": use_compile,
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

    def test_eval_outputs_match_compile_and_handle_shape_change(self) -> None:
        """Eval compile path should match eager on first trace and after batch-size growth."""
        coord_1, atype_1, box_1, _, _, _ = self._load_water_frame()
        coord_2, atype_2, box_2, _, _, _ = self._load_water_frame(nframe=2)

        # === Step 1. Build paired models with shared weights ===
        model_dyn = get_sezm_model(self._build_model_params(use_compile=False))
        with mock.patch.dict(os.environ, {"DP_COMPILE_INFER": "1"}, clear=False):
            model_cmp = get_sezm_model(self._build_model_params(use_compile=True))
        model_cmp.load_state_dict(model_dyn.state_dict())
        model_dyn.eval()
        model_cmp.eval()

        # === Step 2. First eval call traces the compile graph on nf=1 ===
        out_dyn_1 = model_dyn(coord_1, atype_1, box=box_1)
        out_cmp_1 = model_cmp(coord_1, atype_1, box=box_1)
        torch.testing.assert_close(
            out_dyn_1["energy"], out_cmp_1["energy"], atol=1.0e-6, rtol=1.0e-6
        )
        torch.testing.assert_close(
            out_dyn_1["force"], out_cmp_1["force"], atol=1.0e-6, rtol=1.0e-6
        )
        torch.testing.assert_close(
            out_dyn_1["virial"], out_cmp_1["virial"], atol=1.0e-5, rtol=1.0e-5
        )

        # === Step 3. Reuse the traced graph on a larger batch ===
        out_dyn_2 = model_dyn(coord_2, atype_2, box=box_2)
        out_cmp_2 = model_cmp(coord_2, atype_2, box=box_2)
        self.assertEqual(out_dyn_2["energy"].shape, (2, 1))
        self.assertEqual(out_cmp_2["energy"].shape, (2, 1))
        torch.testing.assert_close(
            out_dyn_2["energy"], out_cmp_2["energy"], atol=1.0e-6, rtol=1.0e-6
        )
        torch.testing.assert_close(
            out_dyn_2["force"], out_cmp_2["force"], atol=1.0e-6, rtol=1.0e-6
        )
        torch.testing.assert_close(
            out_dyn_2["virial"], out_cmp_2["virial"], atol=1.0e-5, rtol=1.0e-5
        )

    def test_fixed_edge_geometry_matches_standard_cache(self) -> None:
        """Sparse edge geometry should match the standard descriptor cache."""
        coord, atype, box, _, _, _ = self._load_water_frame()
        model = get_sezm_model(self._build_model_params(use_compile=False))
        model.train()
        descriptor = model.atomic_model.descriptor

        cc, bb, fp, ap, _ = model._input_type_cast(
            coord, box=box, fparam=None, aparam=None
        )
        del fp, ap
        if cc.ndim == 2:
            cc = cc.view(coord.shape[0], atype.shape[1], 3)
        extended_coord, extended_atype, mapping, nlist = model.build_neighbor_list(
            cc, atype, bb
        )
        atype_loc = extended_atype[:, : nlist.shape[1]]
        type_ebed = descriptor.type_embedding(atype_loc).reshape(
            -1, descriptor.channels
        )
        pair_keep_mask = torch.ones_like(
            nlist, dtype=torch.bool, device=extended_coord.device
        )

        cache_std = build_edge_cache(
            type_ebed=type_ebed,
            extended_coord=extended_coord.to(descriptor.compute_dtype),
            nlist=nlist,
            mapping=mapping,
            pair_keep_mask=pair_keep_mask,
            eps=descriptor.eps,
            inner_clamp=descriptor.inner_clamp,
            edge_envelope=descriptor.edge_envelope,
            radial_basis=descriptor.radial_basis,
            n_radial=descriptor.radial_basis.n_radial,
            random_gamma=False,
            wigner_calc=descriptor.wigner_calc,
            use_geometry_rbf_triton=False,
        )

        edge_index, edge_vec, edge_mask = model.build_edge_list_from_nlist(
            extended_coord=extended_coord,
            nlist=nlist,
            mapping=mapping,
        )
        cache_sparse = build_edge_cache_from_edges(
            type_ebed=type_ebed,
            atype_flat=atype_loc.reshape(-1),
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            compute_dtype=descriptor.compute_dtype,
            eps=descriptor.eps,
            inner_clamp=descriptor.inner_clamp,
            edge_envelope=descriptor.edge_envelope,
            radial_basis=descriptor.radial_basis,
            has_exclude_types=False,
            edge_type_keep_mask=descriptor._edge_type_keep_mask,
            random_gamma=False,
            wigner_calc=descriptor.wigner_calc,
        )

        self.assertTrue(torch.equal(cache_std.src, cache_sparse.src))
        self.assertTrue(torch.equal(cache_std.dst, cache_sparse.dst))
        torch.testing.assert_close(cache_std.edge_vec, cache_sparse.edge_vec)
        torch.testing.assert_close(cache_std.edge_rbf, cache_sparse.edge_rbf)
        torch.testing.assert_close(cache_std.edge_env, cache_sparse.edge_env)
        torch.testing.assert_close(cache_std.D_full, cache_sparse.D_full)
        torch.testing.assert_close(cache_std.Dt_full, cache_sparse.Dt_full)

    def test_eval_compile_policy(self) -> None:
        """Eval should stay eager by default and compile only with env override."""
        model = get_sezm_model(self._build_model_params(use_compile=True))
        self.assertTrue(model.use_compile)

        model.train()
        self.assertTrue(model._should_use_compile())

        model.eval()
        self.assertFalse(model._should_use_compile())

        with mock.patch.dict(os.environ, {"DP_COMPILE_INFER": "1"}, clear=False):
            model_eval = get_sezm_model(self._build_model_params(use_compile=True))
        model_eval.eval()
        self.assertTrue(model_eval._should_use_compile())

    def test_forward_backward_double_backward_matches_compile(self) -> None:
        """
        Check forward, backward, double backward, and short training consistency.

        Forward: energy/force outputs should match.
        Backward: d(energy)/d(params) should match.
        Double backward: d(force_loss)/d(params) should match.
        Training: three SGD steps and a larger follow-up batch should still match.
        """
        coord, atype, box, energy, force, virial = self._load_water_frame()
        coord_2, atype_2, box_2, _, _, _ = self._load_water_frame(nframe=2)

        # === Step 1. Build paired models with shared weights ===
        model_dyn = get_sezm_model(self._build_model_params(use_compile=False))
        model_cmp = get_sezm_model(self._build_model_params(use_compile=True))
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

        # === Step 5. Reuse the compiled training graph for three optimizer steps ===
        params_dyn = self._train_steps(
            model_dyn, coord, atype, box, energy, force, virial
        )
        params_cmp = self._train_steps(
            model_cmp, coord, atype, box, energy, force, virial
        )
        self.assertEqual(set(params_dyn.keys()), set(params_cmp.keys()))
        for name in params_dyn.keys():
            torch.testing.assert_close(
                params_dyn[name], params_cmp[name], atol=1.0e-7, rtol=1.0e-7
            )

        # === Step 6. The traced training graph should also handle a larger batch ===
        out_dyn = model_dyn(coord_2, atype_2, box=box_2)
        out_cmp = model_cmp(coord_2, atype_2, box=box_2)
        self.assertEqual(out_dyn["energy"].shape, (2, 1))
        self.assertEqual(out_cmp["energy"].shape, (2, 1))
        torch.testing.assert_close(
            out_dyn["energy"], out_cmp["energy"], atol=1.0e-6, rtol=1.0e-6
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
            "bridging_method": bridging_method,
            "bridging_r_inner": 0.9,
            "bridging_r_outer": 1.3,
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


class TestSeZMModelModes(unittest.TestCase):
    """Targeted regression tests for SeZM `ener` / `dens` mode routing."""

    def setUp(self) -> None:
        self.device = env.DEVICE
        torch.manual_seed(2024)

    def _build_model_params(
        self,
        *,
        use_compile: bool = False,
        bridging_method: str = "none",
    ) -> dict:
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
                "l_schedule": [1, 1],
                "mmax": 1,
                "so2_norm": False,
                "so2_layers": 1,
                "n_atten_head": 0,
                "sandwich_norm": [True, False, True, False],
                "ffn_neurons": 8,
                "ffn_blocks": 1,
                "mlp_bias": True,
                "layer_scale": False,
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
            "use_compile": use_compile,
            "bridging_method": bridging_method,
        }

    def _tiny_system(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coord = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.1, 0.2, 0.0],
                    [0.2, 1.0, 0.3],
                ]
            ],
            device=self.device,
            dtype=torch.float32,
        )
        atype = torch.tensor([[0, 1, 0]], device=self.device, dtype=torch.int32)
        box = torch.tensor(
            [[6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0]],
            device=self.device,
            dtype=torch.float32,
        )
        force = torch.tensor(
            [
                [
                    [0.2, -0.1, 0.0],
                    [-0.3, 0.4, 0.1],
                    [0.1, 0.2, -0.2],
                ]
            ],
            device=self.device,
            dtype=torch.float32,
        )
        noise_mask = torch.tensor(
            [[True, False, True]],
            device=self.device,
            dtype=torch.bool,
        )
        return coord, atype, box, force, noise_mask

    def _dens_stat_samples(self) -> list[dict[str, torch.Tensor | np.float32]]:
        """Build a tiny SeZM `dens` statistics set with force labels."""
        return [
            {
                "atype": torch.tensor(
                    [[0, 1]],
                    device=self.device,
                    dtype=torch.int32,
                ),
                "natoms": torch.tensor(
                    [[2, 2, 1, 1]],
                    device=self.device,
                    dtype=torch.int32,
                ),
                "energy": torch.tensor(
                    [[10.0]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "force": torch.tensor(
                    [[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "find_energy": np.float32(1.0),
                "find_force": np.float32(1.0),
            },
            {
                "atype": torch.tensor(
                    [[0, 0]],
                    device=self.device,
                    dtype=torch.int32,
                ),
                "natoms": torch.tensor(
                    [[2, 2, 2, 0]],
                    device=self.device,
                    dtype=torch.int32,
                ),
                "energy": torch.tensor(
                    [[8.0]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "force": torch.tensor(
                    [[[5.0, 6.0, 7.0], [5.0, 6.0, 7.0]]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "find_energy": np.float32(1.0),
                "find_force": np.float32(1.0),
            },
            {
                "atype": torch.tensor(
                    [[1, 1]],
                    device=self.device,
                    dtype=torch.int32,
                ),
                "natoms": torch.tensor(
                    [[2, 2, 0, 2]],
                    device=self.device,
                    dtype=torch.int32,
                ),
                "energy": torch.tensor(
                    [[12.0]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "force": torch.tensor(
                    [[[8.0, 10.0, 12.0], [8.0, 10.0, 12.0]]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "find_energy": np.float32(1.0),
                "find_force": np.float32(1.0),
            },
        ]

    def _expected_dens_force_rmsd(
        self,
        sampled: list[dict[str, torch.Tensor | np.float32]],
    ) -> float:
        """Compute the expected global direct-force RMSD."""
        force_square_sum = 0.0
        force_atom_count = 0
        for sample in sampled:
            force = sample["force"].detach().cpu().numpy()
            force_square_sum += float(np.square(force).sum())
            force_atom_count += int(force.shape[0] * force.shape[1])
        return float(np.sqrt(force_square_sum / force_atom_count))

    def test_training_setup_routes_mode_without_rebuilding_energy_head(self) -> None:
        """Training setup should route SeZM mode without rebuilding the energy head."""
        model = get_sezm_model(self._build_model_params(use_compile=False))
        energy_param_before = (
            next(model.atomic_model.fitting_net.parameters()).detach().clone()
        )
        prepare_model_for_loss(model, {"type": "dens"})
        self.assertEqual(model.get_active_mode(), "dens")
        self.assertIsNotNone(model.atomic_model.dens_fitting_net)
        prepare_model_for_loss(model, {"type": "ener"})
        coord, atype, box, _, _ = self._tiny_system()
        loss_module = EnergyStdLoss(
            starter_learning_rate=1.0e-3,
            start_pref_e=1.0,
            limit_pref_e=1.0,
        )
        _, loss, _ = loss_module(
            {
                "coord": coord,
                "atype": atype,
                "box": box,
            },
            model,
            {
                "energy": torch.zeros((1, 1), device=self.device, dtype=torch.float32),
                "find_energy": 1.0,
            },
            natoms=atype.shape[1],
            learning_rate=1.0e-3,
        )
        energy_param_after = next(model.atomic_model.fitting_net.parameters()).detach()
        torch.testing.assert_close(energy_param_after, energy_param_before)
        self.assertEqual(model.get_active_mode(), "ener")
        self.assertTrue(torch.isfinite(loss))

    def test_checkpoint_loading_handles_optional_dens_head(self) -> None:
        """Checkpoint loading should respect whether `dens` weights exist."""
        params = self._build_model_params(use_compile=False)
        model = get_sezm_model(params)
        state_without_dens = {
            key: value
            for key, value in model.state_dict().items()
            if "dens_fitting_net" not in key
        }
        fresh_model = get_sezm_model(params)
        self.assertIsNone(fresh_model.atomic_model.dens_fitting_net)
        fresh_model.load_state_dict(state_without_dens, strict=True)
        self.assertIsNone(fresh_model.atomic_model.dens_fitting_net)
        self.assertEqual(fresh_model.get_active_mode(), "ener")
        coord, atype, box, _, _ = self._tiny_system()
        out = fresh_model(coord, atype, box=box)
        self.assertIn("energy", out)
        self.assertIn("force", out)
        model = get_sezm_model(self._build_model_params(use_compile=False))
        model.set_active_mode("dens")
        dens_state = model.state_dict()
        fresh_model = get_sezm_model(self._build_model_params(use_compile=False))
        self.assertIsNone(fresh_model.atomic_model.dens_fitting_net)
        fresh_model.load_state_dict(dens_state, strict=True)
        self.assertIsNotNone(fresh_model.atomic_model.dens_fitting_net)
        self.assertEqual(fresh_model.get_active_mode(), "dens")

    def test_dens_forward_returns_direct_force_outputs(self) -> None:
        """`dens` mode should expose direct-force outputs without virial branches."""
        model = get_sezm_model(self._build_model_params(use_compile=False))
        model.set_active_mode("dens")
        coord, atype, box, force, noise_mask = self._tiny_system()
        out = model(
            coord,
            atype,
            box=box,
            force_input=force,
            noise_mask=noise_mask,
        )
        self.assertIn("energy", out)
        self.assertIn("atom_energy", out)
        self.assertIn("force", out)
        self.assertNotIn("virial", out)
        self.assertEqual(out["force"].shape, force.shape)

    def test_dens_loss_forward_smoke(self) -> None:
        """`DeNSLoss` should build noisy inputs and return a finite training loss."""
        model = get_sezm_model(self._build_model_params(use_compile=False))
        prepare_model_for_loss(model, {"type": "dens"})
        loss_module = DeNSLoss(
            starter_learning_rate=1.0e-3,
            start_pref_e=1.0,
            limit_pref_e=1.0,
            start_pref_f=1.0,
            limit_pref_f=1.0,
            dens_prob=1.0,
            dens_std=0.025,
            dens_corrupt_ratio=0.5,
            dens_denoising_pos_coefficient=10.0,
            loss_func="mae",
        )
        coord, atype, box, force, _ = self._tiny_system()
        label = {
            "energy": torch.zeros((1, 1), device=self.device, dtype=torch.float32),
            "force": force,
            "find_energy": 1.0,
            "find_force": 1.0,
        }
        model_pred, loss, more_loss = loss_module(
            {
                "coord": coord,
                "atype": atype,
                "box": box,
            },
            model,
            label,
            natoms=atype.shape[1],
            learning_rate=1.0e-3,
        )
        self.assertEqual(model.get_active_mode(), "dens")
        self.assertIn("force", model_pred)
        self.assertTrue(torch.isfinite(loss))

    def test_dens_stat_roundtrip(self) -> None:
        """`dens` statistics should roundtrip the global direct-force RMSD."""
        sampled = self._dens_stat_samples()
        expected_force_rmsd = self._expected_dens_force_rmsd(sampled)

        model = get_sezm_model(self._build_model_params(use_compile=False))
        prepare_model_for_loss(model, {"type": "dens"})

        with tempfile.TemporaryDirectory() as tmpdir:
            h5file = Path(tmpdir) / "sezm_stat.hdf5"
            with h5py.File(h5file, "w"):
                pass

            stat_path = DPPath(str(h5file), "a")
            try:
                model.atomic_model.compute_or_load_stat(
                    lambda: sampled,
                    stat_file_path=stat_path,
                )
                self.assertAlmostEqual(
                    model.atomic_model.dens_force_rmsd.item(),
                    expected_force_rmsd,
                    places=7,
                )
                self.assertEqual(model.get_active_mode(), "dens")

                stored_force_rmsd = (stat_path / "O H" / "rmsd_dforce").load_numpy()
                self.assertAlmostEqual(
                    float(np.asarray(stored_force_rmsd).reshape(-1)[0]),
                    expected_force_rmsd,
                    places=7,
                )

                fresh_model = get_sezm_model(
                    self._build_model_params(use_compile=False)
                )
                prepare_model_for_loss(fresh_model, {"type": "dens"})

                def raise_error() -> None:
                    raise RuntimeError("statistics should be restored from file")

                fresh_model.atomic_model.compute_or_load_stat(
                    raise_error,
                    stat_file_path=stat_path,
                )
                self.assertAlmostEqual(
                    fresh_model.atomic_model.dens_force_rmsd.item(),
                    expected_force_rmsd,
                    places=7,
                )
                self.assertEqual(fresh_model.get_active_mode(), "dens")
            finally:
                stat_path.root.close()
