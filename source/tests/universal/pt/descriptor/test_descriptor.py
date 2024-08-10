# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.pt.model.descriptor import (
    DescrptDPA2,
)

from ....consistent.common import (
    parameterized,
)
from ...common.cases.descriptor.descriptor import (
    DescriptorTest,
)
from ...dpmodel.descriptor.test_descriptor import (
    DescriptorParamDPA2TTebd,
)
from ..backend import (
    PTTestCase,
)


@parameterized(
    (
        # (DescriptorParamSeA, DescrptSeA),
        # (DescriptorParamSeR, DescrptSeR),
        # (DescriptorParamSeT, DescrptSeT),
        # (DescriptorParamDPA1, DescrptDPA1),
        # (DescriptorParamDPA2, DescrptDPA2),
        # (DescriptorParamHybrid, DescrptHybrid),
        # (DescriptorParamHybridMixed, DescrptHybrid),
        # (DescriptorParamHybridTTebd, DescrptHybrid),
        (DescriptorParamDPA2TTebd, DescrptDPA2),
    )  # class_param & class
)
class TestDescriptorPT(unittest.TestCase, DescriptorTest, PTTestCase):
    def setUp(self):
        DescriptorTest.setUp(self)
        (DescriptorParam, Descrpt) = self.param[0]
        self.module_class = Descrpt
        self.input_dict = DescriptorParam(
            self.nt, self.rcut, self.rcut_smth, self.sel, ["O", "H"]
        )
        self.module = Descrpt(**self.input_dict)
