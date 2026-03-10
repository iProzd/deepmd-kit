# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM atomic model definitions."""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

from deepmd.pt.model.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.pt.model.task.ener import (
    EnergyFittingNet,
    EnergyFittingNetDirect,
    InvarFitting,
)


class SeZMAtomicModel(DPAtomicModel):
    """Atomic model scaffold for SeZM energy fitting.

    Parameters
    ----------
    descriptor
        Descriptor instance.
    fitting
        Energy fitting network instance.
    type_map
        Atom type map.
    **kwargs
        Additional keyword arguments forwarded to DPAtomicModel.

    Raises
    ------
    TypeError
        If fitting is not an energy fitting network.
    """

    def __init__(
        self, descriptor: Any, fitting: Any, type_map: Any, **kwargs: Any
    ) -> None:
        if not (
            isinstance(fitting, EnergyFittingNet)
            or isinstance(fitting, EnergyFittingNetDirect)
            or isinstance(fitting, InvarFitting)
        ):
            raise TypeError(
                "fitting must be an instance of EnergyFittingNet, EnergyFittingNetDirect "
                "or InvarFitting for SeZMAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)
