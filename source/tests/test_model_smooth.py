# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest

import numpy as np
from common import (
    j_loader,
    run_dp,
    tests_path,
)

from deepmd.infer import (
    DeepPot,
)

from pathlib import (
    Path,
)


def _file_delete(file):
    if os.path.isdir(file):
        os.rmdir(file)
    elif os.path.isfile(file):
        os.remove(file)


def init_models(model_name, jdata):
    data_file = str(tests_path / os.path.join("model_smooth", "data"))
    frozen_model = str(tests_path / "model_smooth_{}.pb".format(model_name))
    ckpt = str(tests_path / "model_smooth_{}.ckpt".format(model_name))
    INPUT = str(tests_path / "input_{}.json".format(model_name))
    jdata["training"]["training_data"]["systems"] = data_file
    jdata["training"]["validation_data"]["systems"] = data_file
    jdata["training"]["save_ckpt"] = ckpt
    with open(INPUT, "w") as fp:
        json.dump(jdata, fp, indent=4)
    ret = run_dp("dp train " + INPUT)
    np.testing.assert_equal(ret, 0, "DP train failed!")
    ret = run_dp("dp freeze -c " + str(tests_path) + " -o " + frozen_model)
    np.testing.assert_equal(ret, 0, "DP freeze failed!")
    dp_model = DeepPot(Path(frozen_model))
    return dp_model, INPUT, ckpt, frozen_model


def init_data(epsilon):
    natoms = 10
    cell = 8.6 * np.eye(3)
    atype = np.random.randint(0, 3, [natoms])
    coord0 = np.array(
        [
            0., 0., 0.,
            4. - .5 * epsilon, 0., 0.,
            0., 4. - .5 * epsilon, 0.,
        ]).reshape([-1, 3])
    coord1 = np.random.rand(natoms - coord0.shape[0], 3)
    coord1 = np.matmul(coord1, cell)
    coord = np.concatenate([coord0, coord1], axis=0)

    coord0 = np.copy(coord)
    coord1 = np.copy(coord)
    coord1[1][0] += epsilon
    coord2 = np.copy(coord)
    coord2[2][1] += epsilon
    coord3 = np.copy(coord)
    coord3[1][0] += epsilon
    coord3[2][1] += epsilon
    return {"natoms": natoms, "cell": cell.reshape([1, -1]), "atype": list(atype),
            "coord0": coord0.reshape([1, -1]), "coord1": coord1.reshape([1, -1]),
            "coord2": coord2.reshape([1, -1]), "coord3": coord3.reshape([1, -1])}


def compare(ret0, ret1, rprec, aprec, test_virial=False):
    np.testing.assert_allclose(ret0['energy'], ret1['energy'], rtol=rprec, atol=aprec)
    # plus 1. to avoid the divided-by-zero issue
    np.testing.assert_allclose(1. + ret0['force'], 1. + ret1['force'], rtol=rprec, atol=aprec)
    if test_virial:
        np.testing.assert_allclose(1. + ret0['virial'], 1. + ret1['virial'], rtol=rprec, atol=aprec)


class TestModelSmoothSeA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jdata = j_loader(str(tests_path / os.path.join("model_smooth", "input.json")))
        cls.dp_model, cls.INPUT, cls.ckpt, cls.frozen_model = init_models('se_a', jdata)
        # displacement of atoms
        cls.epsilon = 1e-5
        # required prec. relative prec is not checked.
        cls.rprec = 0
        cls.aprec = 1e-5
        cls.data = init_data(cls.epsilon)
        cls.test_virial = True

    @classmethod
    def tearDownClass(cls):
        _file_delete(cls.INPUT)
        _file_delete(cls.frozen_model)
        _file_delete("out.json")
        _file_delete(str(tests_path / "checkpoint"))
        _file_delete(cls.ckpt + ".meta")
        _file_delete(cls.ckpt + ".index")
        _file_delete(cls.ckpt + ".data-00000-of-00001")
        _file_delete(cls.ckpt + "-0.meta")
        _file_delete(cls.ckpt + "-0.index")
        _file_delete(cls.ckpt + "-0.data-00000-of-00001")
        _file_delete(cls.ckpt + "-1.meta")
        _file_delete(cls.ckpt + "-1.index")
        _file_delete(cls.ckpt + "-1.data-00000-of-00001")
        _file_delete("input_v2_compat.json")
        _file_delete("lcurve.out")

    def test_smooth(self):
        e0, f0, v0 = self.dp_model.eval(self.data['coord0'], self.data['cell'], self.data['atype'])
        ret0 = {'energy': e0.squeeze(0), 'force': f0.squeeze(0), 'virial': v0.squeeze(0)}
        e1, f1, v1 = self.dp_model.eval(self.data['coord1'], self.data['cell'], self.data['atype'])
        ret1 = {'energy': e1.squeeze(0), 'force': f1.squeeze(0), 'virial': v1.squeeze(0)}
        e2, f2, v2 = self.dp_model.eval(self.data['coord2'], self.data['cell'], self.data['atype'])
        ret2 = {'energy': e2.squeeze(0), 'force': f2.squeeze(0), 'virial': v2.squeeze(0)}
        e3, f3, v3 = self.dp_model.eval(self.data['coord3'], self.data['cell'], self.data['atype'])
        ret3 = {'energy': e3.squeeze(0), 'force': f3.squeeze(0), 'virial': v3.squeeze(0)}

        compare(ret0, ret1, self.rprec, self.aprec, test_virial=self.test_virial)
        compare(ret1, ret2, self.rprec, self.aprec, test_virial=self.test_virial)
        compare(ret0, ret3, self.rprec, self.aprec, test_virial=self.test_virial)


# should be OK
class TestModelSmoothSeAttenL0V1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jdata = j_loader(str(tests_path / os.path.join("model_smooth", "input.json")))
        jdata['model']['descriptor'] = {
            "type": "se_atten",
            "sel": 120,
            "rcut_smth": 2.0,
            "rcut": 9.0,
            "neuron": [
                25,
                50,
                100
            ],
            "resnet_dt": False,
            "axis_neuron": 12,
            "attn": 128,
            "attn_layer": 0,
            "attn_dotr": True,
            "attn_mask": False,
            # "set_davg_zero": True,
            # "stripped_type_embedding": False,
            # "smooth_type_embdding": False
        }
        cls.dp_model, cls.INPUT, cls.ckpt, cls.frozen_model = init_models('se_atten_0_v1', jdata)
        # displacement of atoms
        cls.epsilon = 1e-5
        # required prec. relative prec is not checked.
        cls.rprec = 0
        cls.aprec = 1e-5
        cls.data = init_data(cls.epsilon)
        cls.test_virial = True

    @classmethod
    def tearDownClass(cls):
        _file_delete(cls.INPUT)
        _file_delete(cls.frozen_model)
        _file_delete("out.json")
        _file_delete(str(tests_path / "checkpoint"))
        _file_delete(cls.ckpt + ".meta")
        _file_delete(cls.ckpt + ".index")
        _file_delete(cls.ckpt + ".data-00000-of-00001")
        _file_delete(cls.ckpt + "-0.meta")
        _file_delete(cls.ckpt + "-0.index")
        _file_delete(cls.ckpt + "-0.data-00000-of-00001")
        _file_delete(cls.ckpt + "-1.meta")
        _file_delete(cls.ckpt + "-1.index")
        _file_delete(cls.ckpt + "-1.data-00000-of-00001")
        _file_delete("input_v2_compat.json")
        _file_delete("lcurve.out")

    def test_smooth(self):
        e0, f0, v0 = self.dp_model.eval(self.data['coord0'], self.data['cell'], self.data['atype'])
        ret0 = {'energy': e0.squeeze(0), 'force': f0.squeeze(0), 'virial': v0.squeeze(0)}
        e1, f1, v1 = self.dp_model.eval(self.data['coord1'], self.data['cell'], self.data['atype'])
        ret1 = {'energy': e1.squeeze(0), 'force': f1.squeeze(0), 'virial': v1.squeeze(0)}
        e2, f2, v2 = self.dp_model.eval(self.data['coord2'], self.data['cell'], self.data['atype'])
        ret2 = {'energy': e2.squeeze(0), 'force': f2.squeeze(0), 'virial': v2.squeeze(0)}
        e3, f3, v3 = self.dp_model.eval(self.data['coord3'], self.data['cell'], self.data['atype'])
        ret3 = {'energy': e3.squeeze(0), 'force': f3.squeeze(0), 'virial': v3.squeeze(0)}

        compare(ret0, ret1, self.rprec, self.aprec, test_virial=self.test_virial)
        compare(ret1, ret2, self.rprec, self.aprec, test_virial=self.test_virial)
        compare(ret0, ret3, self.rprec, self.aprec, test_virial=self.test_virial)


# should be OK
class TestModelSmoothSeAttenL0V1Stripped(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jdata = j_loader(str(tests_path / os.path.join("model_smooth", "input.json")))
        jdata['model']['descriptor'] = {
            "type": "se_atten",
            "sel": 120,
            "rcut_smth": 2.0,
            "rcut": 9.0,
            "neuron": [
                25,
                50,
                100
            ],
            "resnet_dt": False,
            "axis_neuron": 12,
            "attn": 128,
            "attn_layer": 0,
            "attn_dotr": True,
            "attn_mask": False,
            # "set_davg_zero": True,
            "stripped_type_embedding": True,
            # "smooth_type_embdding": False
        }
        cls.dp_model, cls.INPUT, cls.ckpt, cls.frozen_model = init_models('se_atten_0_v1_stripped', jdata)
        # displacement of atoms
        cls.epsilon = 1e-5
        # required prec. relative prec is not checked.
        cls.rprec = 0
        cls.aprec = 1e-5
        cls.data = init_data(cls.epsilon)
        cls.test_virial = True

    @classmethod
    def tearDownClass(cls):
        _file_delete(cls.INPUT)
        _file_delete(cls.frozen_model)
        _file_delete("out.json")
        _file_delete(str(tests_path / "checkpoint"))
        _file_delete(cls.ckpt + ".meta")
        _file_delete(cls.ckpt + ".index")
        _file_delete(cls.ckpt + ".data-00000-of-00001")
        _file_delete(cls.ckpt + "-0.meta")
        _file_delete(cls.ckpt + "-0.index")
        _file_delete(cls.ckpt + "-0.data-00000-of-00001")
        _file_delete(cls.ckpt + "-1.meta")
        _file_delete(cls.ckpt + "-1.index")
        _file_delete(cls.ckpt + "-1.data-00000-of-00001")
        _file_delete("input_v2_compat.json")
        _file_delete("lcurve.out")

    def test_smooth(self):
        e0, f0, v0 = self.dp_model.eval(self.data['coord0'], self.data['cell'], self.data['atype'])
        ret0 = {'energy': e0.squeeze(0), 'force': f0.squeeze(0), 'virial': v0.squeeze(0)}
        e1, f1, v1 = self.dp_model.eval(self.data['coord1'], self.data['cell'], self.data['atype'])
        ret1 = {'energy': e1.squeeze(0), 'force': f1.squeeze(0), 'virial': v1.squeeze(0)}
        e2, f2, v2 = self.dp_model.eval(self.data['coord2'], self.data['cell'], self.data['atype'])
        ret2 = {'energy': e2.squeeze(0), 'force': f2.squeeze(0), 'virial': v2.squeeze(0)}
        e3, f3, v3 = self.dp_model.eval(self.data['coord3'], self.data['cell'], self.data['atype'])
        ret3 = {'energy': e3.squeeze(0), 'force': f3.squeeze(0), 'virial': v3.squeeze(0)}

        compare(ret0, ret1, self.rprec, self.aprec, test_virial=self.test_virial)
        compare(ret1, ret2, self.rprec, self.aprec, test_virial=self.test_virial)
        compare(ret0, ret3, self.rprec, self.aprec, test_virial=self.test_virial)


# Not OK, but still pass this UT
class TestModelSmoothSeAttenL0V1Shifted(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jdata = j_loader(str(tests_path / os.path.join("model_smooth", "input.json")))
        jdata['model']['descriptor'] = {
            "type": "se_atten",
            "sel": 120,
            "rcut_smth": 2.0,
            "rcut": 9.0,
            "neuron": [
                25,
                50,
                100
            ],
            "resnet_dt": False,
            "axis_neuron": 12,
            "attn": 128,
            "attn_layer": 0,
            "attn_dotr": True,
            "attn_mask": False,
            "set_davg_zero": False,
            # "stripped_type_embedding": False,
            # "smooth_type_embdding": False
        }
        cls.dp_model, cls.INPUT, cls.ckpt, cls.frozen_model = init_models('se_atten_0_v1_shifted', jdata)
        # displacement of atoms
        cls.epsilon = 1e-5
        # required prec. relative prec is not checked.
        cls.rprec = 0
        cls.aprec = 1e-5
        cls.data = init_data(cls.epsilon)
        cls.test_virial = True

    @classmethod
    def tearDownClass(cls):
        _file_delete(cls.INPUT)
        _file_delete(cls.frozen_model)
        _file_delete("out.json")
        _file_delete(str(tests_path / "checkpoint"))
        _file_delete(cls.ckpt + ".meta")
        _file_delete(cls.ckpt + ".index")
        _file_delete(cls.ckpt + ".data-00000-of-00001")
        _file_delete(cls.ckpt + "-0.meta")
        _file_delete(cls.ckpt + "-0.index")
        _file_delete(cls.ckpt + "-0.data-00000-of-00001")
        _file_delete(cls.ckpt + "-1.meta")
        _file_delete(cls.ckpt + "-1.index")
        _file_delete(cls.ckpt + "-1.data-00000-of-00001")
        _file_delete("input_v2_compat.json")
        _file_delete("lcurve.out")

    def test_smooth(self):
        e0, f0, v0 = self.dp_model.eval(self.data['coord0'], self.data['cell'], self.data['atype'])
        ret0 = {'energy': e0.squeeze(0), 'force': f0.squeeze(0), 'virial': v0.squeeze(0)}
        e1, f1, v1 = self.dp_model.eval(self.data['coord1'], self.data['cell'], self.data['atype'])
        ret1 = {'energy': e1.squeeze(0), 'force': f1.squeeze(0), 'virial': v1.squeeze(0)}
        e2, f2, v2 = self.dp_model.eval(self.data['coord2'], self.data['cell'], self.data['atype'])
        ret2 = {'energy': e2.squeeze(0), 'force': f2.squeeze(0), 'virial': v2.squeeze(0)}
        e3, f3, v3 = self.dp_model.eval(self.data['coord3'], self.data['cell'], self.data['atype'])
        ret3 = {'energy': e3.squeeze(0), 'force': f3.squeeze(0), 'virial': v3.squeeze(0)}

        compare(ret0, ret1, self.rprec, self.aprec, test_virial=self.test_virial)
        compare(ret1, ret2, self.rprec, self.aprec, test_virial=self.test_virial)
        compare(ret0, ret3, self.rprec, self.aprec, test_virial=self.test_virial)


# Not OK, but still pass this UT
class TestModelSmoothSeAttenL0V2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jdata = j_loader(str(tests_path / os.path.join("model_smooth", "input.json")))
        jdata['model']['descriptor'] = {
            "type": "se_atten_v2",
            "sel": 120,
            "rcut_smth": 2.0,
            "rcut": 9.0,
            "neuron": [
                25,
                50,
                100
            ],
            "resnet_dt": False,
            "axis_neuron": 12,
            "attn": 128,
            "attn_layer": 0,
            "attn_dotr": True,
            "attn_mask": False,
            # "set_davg_zero": False,
            # "stripped_type_embedding": True,
            # "smooth_type_embdding": True
        }
        cls.dp_model, cls.INPUT, cls.ckpt, cls.frozen_model = init_models('se_atten_0_v2', jdata)
        # displacement of atoms
        cls.epsilon = 1e-5
        # required prec. relative prec is not checked.
        cls.rprec = 0
        cls.aprec = 1e-5
        cls.data = init_data(cls.epsilon)
        cls.test_virial = True

    @classmethod
    def tearDownClass(cls):
        _file_delete(cls.INPUT)
        _file_delete(cls.frozen_model)
        _file_delete("out.json")
        _file_delete(str(tests_path / "checkpoint"))
        _file_delete(cls.ckpt + ".meta")
        _file_delete(cls.ckpt + ".index")
        _file_delete(cls.ckpt + ".data-00000-of-00001")
        _file_delete(cls.ckpt + "-0.meta")
        _file_delete(cls.ckpt + "-0.index")
        _file_delete(cls.ckpt + "-0.data-00000-of-00001")
        _file_delete(cls.ckpt + "-1.meta")
        _file_delete(cls.ckpt + "-1.index")
        _file_delete(cls.ckpt + "-1.data-00000-of-00001")
        _file_delete("input_v2_compat.json")
        _file_delete("lcurve.out")

    def test_smooth(self):
        e0, f0, v0 = self.dp_model.eval(self.data['coord0'], self.data['cell'], self.data['atype'])
        ret0 = {'energy': e0.squeeze(0), 'force': f0.squeeze(0), 'virial': v0.squeeze(0)}
        e1, f1, v1 = self.dp_model.eval(self.data['coord1'], self.data['cell'], self.data['atype'])
        ret1 = {'energy': e1.squeeze(0), 'force': f1.squeeze(0), 'virial': v1.squeeze(0)}
        e2, f2, v2 = self.dp_model.eval(self.data['coord2'], self.data['cell'], self.data['atype'])
        ret2 = {'energy': e2.squeeze(0), 'force': f2.squeeze(0), 'virial': v2.squeeze(0)}
        e3, f3, v3 = self.dp_model.eval(self.data['coord3'], self.data['cell'], self.data['atype'])
        ret3 = {'energy': e3.squeeze(0), 'force': f3.squeeze(0), 'virial': v3.squeeze(0)}

        compare(ret0, ret1, self.rprec, self.aprec, test_virial=self.test_virial)
        compare(ret1, ret2, self.rprec, self.aprec, test_virial=self.test_virial)
        compare(ret0, ret3, self.rprec, self.aprec, test_virial=self.test_virial)


# Not OK, but still pass this UT
class TestModelSmoothSeAttenL2V1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jdata = j_loader(str(tests_path / os.path.join("model_smooth", "input.json")))
        jdata['model']['descriptor'] = {
            "type": "se_atten",
            "sel": 120,
            "rcut_smth": 2.0,
            "rcut": 9.0,
            "neuron": [
                25,
                50,
                100
            ],
            "resnet_dt": False,
            "axis_neuron": 12,
            "attn": 128,
            "attn_layer": 2,
            "attn_dotr": True,
            "attn_mask": False,
            # "set_davg_zero": True,
            # "stripped_type_embedding": False,
            # "smooth_type_embdding": False
        }
        cls.dp_model, cls.INPUT, cls.ckpt, cls.frozen_model = init_models('se_atten_2_v1', jdata)
        # displacement of atoms
        cls.epsilon = 1e-5
        # required prec. relative prec is not checked.
        cls.rprec = 0
        cls.aprec = 1e-5
        cls.data = init_data(cls.epsilon)
        cls.test_virial = True

    @classmethod
    def tearDownClass(cls):
        _file_delete(cls.INPUT)
        _file_delete(cls.frozen_model)
        _file_delete("out.json")
        _file_delete(str(tests_path / "checkpoint"))
        _file_delete(cls.ckpt + ".meta")
        _file_delete(cls.ckpt + ".index")
        _file_delete(cls.ckpt + ".data-00000-of-00001")
        _file_delete(cls.ckpt + "-0.meta")
        _file_delete(cls.ckpt + "-0.index")
        _file_delete(cls.ckpt + "-0.data-00000-of-00001")
        _file_delete(cls.ckpt + "-1.meta")
        _file_delete(cls.ckpt + "-1.index")
        _file_delete(cls.ckpt + "-1.data-00000-of-00001")
        _file_delete("input_v2_compat.json")
        _file_delete("lcurve.out")

    def test_smooth(self):
        e0, f0, v0 = self.dp_model.eval(self.data['coord0'], self.data['cell'], self.data['atype'])
        ret0 = {'energy': e0.squeeze(0), 'force': f0.squeeze(0), 'virial': v0.squeeze(0)}
        e1, f1, v1 = self.dp_model.eval(self.data['coord1'], self.data['cell'], self.data['atype'])
        ret1 = {'energy': e1.squeeze(0), 'force': f1.squeeze(0), 'virial': v1.squeeze(0)}
        e2, f2, v2 = self.dp_model.eval(self.data['coord2'], self.data['cell'], self.data['atype'])
        ret2 = {'energy': e2.squeeze(0), 'force': f2.squeeze(0), 'virial': v2.squeeze(0)}
        e3, f3, v3 = self.dp_model.eval(self.data['coord3'], self.data['cell'], self.data['atype'])
        ret3 = {'energy': e3.squeeze(0), 'force': f3.squeeze(0), 'virial': v3.squeeze(0)}

        compare(ret0, ret1, self.rprec, self.aprec, test_virial=self.test_virial)
        compare(ret1, ret2, self.rprec, self.aprec, test_virial=self.test_virial)
        compare(ret0, ret3, self.rprec, self.aprec, test_virial=self.test_virial)
