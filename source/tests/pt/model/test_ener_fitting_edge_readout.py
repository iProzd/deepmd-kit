# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

model_dpa3_edge_readout = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa3",
        "repflow": {
            "n_dim": 10,
            "e_dim": 8,
            "a_dim": 8,
            "nlayers": 2,
            "e_rcut": 6.0,
            "e_rcut_smth": 3.0,
            "e_sel": 20,
            "a_rcut": 4.0,
            "a_rcut_smth": 2.0,
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
            "optim_update": False,
        },
        "activation_function": "tanh",
        "use_tebd_bias": False,
        "precision": "float64",
        "concat_output_tebd": False,
    },
    "fitting_net": {
        "neuron": [24, 24],
        "resnet_dt": True,
        "precision": "float64",
        "seed": 1,
        "add_edge_readout": True,
    },
}

model_dpa3_edge_readout_custom_neuron = copy.deepcopy(model_dpa3_edge_readout)
model_dpa3_edge_readout_custom_neuron["fitting_net"]["edge_readout_neuron"] = [12, 12]

model_dpa3_no_edge_readout = copy.deepcopy(model_dpa3_edge_readout)
model_dpa3_no_edge_readout["fitting_net"]["add_edge_readout"] = False


class TestEnergyFittingEdgeReadoutConstruction(unittest.TestCase):
    """Test that edge readout model can be constructed."""

    def test_construction_with_edge_readout(self) -> None:
        model_params = copy.deepcopy(model_dpa3_edge_readout)
        model = get_model(model_params).to(device)
        self.assertTrue(model is not None)
        fitting = model.get_fitting_net()
        self.assertTrue(fitting.add_edge_readout)
        self.assertTrue(fitting.edge_embed is not None)

    def test_construction_without_edge_readout(self) -> None:
        model_params = copy.deepcopy(model_dpa3_no_edge_readout)
        model = get_model(model_params).to(device)
        fitting = model.get_fitting_net()
        self.assertFalse(fitting.add_edge_readout)
        self.assertTrue(fitting.edge_embed is None)

    def test_default_neuron(self) -> None:
        model_params = copy.deepcopy(model_dpa3_edge_readout)
        model = get_model(model_params).to(device)
        fitting = model.get_fitting_net()
        self.assertEqual(fitting.edge_readout_neuron, [24, 24])

    def test_custom_neuron(self) -> None:
        model_params = copy.deepcopy(model_dpa3_edge_readout_custom_neuron)
        model = get_model(model_params).to(device)
        fitting = model.get_fitting_net()
        self.assertEqual(fitting.edge_readout_neuron, [12, 12])


def _eval_model(model, coord, cell, atype):
    """Evaluate the model using the standard forward pass."""
    from deepmd.pt.utils.nlist import (
        extend_input_and_build_neighbor_list,
    )

    if coord.dim() == 2:
        coord = coord.unsqueeze(0)
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)

    (
        extended_coord,
        extended_atype,
        mapping,
        nlist,
    ) = extend_input_and_build_neighbor_list(
        coord,
        atype.unsqueeze(0),
        model.get_rcut(),
        model.get_sel(),
        mixed_types=model.mixed_types(),
        box=cell,
    )
    atomic_model = model.atomic_model
    ret = atomic_model.forward_common_atomic(
        extended_coord,
        extended_atype,
        nlist,
        mapping=mapping,
    )
    return ret


class TestEnergyFittingEdgeReadoutForward(unittest.TestCase):
    """Test forward pass with edge readout."""

    def setUp(self) -> None:
        self.natoms = 12
        self.cell = 10.0 * torch.eye(3, dtype=dtype, device=device)
        torch.manual_seed(20240217)
        self.coord = torch.rand([self.natoms, 3], dtype=dtype, device=device) * 10.0
        self.atype = torch.tensor(
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.int32, device=device
        )

    def test_forward_with_edge_readout(self) -> None:
        model_params = copy.deepcopy(model_dpa3_edge_readout)
        model = get_model(model_params).to(device)
        ret = _eval_model(model, self.coord, self.cell, self.atype)
        self.assertIn("energy", ret)
        energy = ret["energy"]
        self.assertEqual(energy.shape, (1, self.natoms, 1))
        self.assertFalse(torch.isnan(energy).any())

    def test_forward_without_edge_readout(self) -> None:
        model_params = copy.deepcopy(model_dpa3_no_edge_readout)
        model = get_model(model_params).to(device)
        ret = _eval_model(model, self.coord, self.cell, self.atype)
        self.assertIn("energy", ret)
        energy = ret["energy"]
        self.assertEqual(energy.shape, (1, self.natoms, 1))
        self.assertFalse(torch.isnan(energy).any())


class TestEnergyFittingEdgeReadoutPermutation(unittest.TestCase):
    """Test permutation invariance with edge readout."""

    def setUp(self) -> None:
        self.natoms = 12
        self.cell = 10.0 * torch.eye(3, dtype=dtype, device=device)
        torch.manual_seed(20240218)
        self.coord = torch.rand([self.natoms, 3], dtype=dtype, device=device) * 10.0
        self.atype = torch.tensor(
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.int32, device=device
        )

    def test_permutation(self) -> None:
        model_params = copy.deepcopy(model_dpa3_edge_readout)
        model = get_model(model_params).to(device)

        idx_perm = [1, 0, 4, 3, 2, 5, 7, 6, 9, 8, 11, 10]
        ret0 = _eval_model(model, self.coord, self.cell, self.atype)
        ret1 = _eval_model(model, self.coord[idx_perm], self.cell, self.atype[idx_perm])

        energy0 = ret0["energy"].squeeze(0)
        energy1 = ret1["energy"].squeeze(0)

        # Total energy should be invariant
        torch.testing.assert_close(energy0.sum(), energy1.sum(), rtol=1e-10, atol=1e-10)
        # Per-atom energy should be permuted
        torch.testing.assert_close(energy0[idx_perm], energy1, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
