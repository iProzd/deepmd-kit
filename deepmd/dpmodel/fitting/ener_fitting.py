# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    TYPE_CHECKING,
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.common import (
    DEFAULT_PRECISION,
)
from deepmd.dpmodel.fitting.invar_fitting import (
    InvarFitting,
)
from deepmd.dpmodel.utils import (
    FittingNet,
    NetworkCollection,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.fitting.general_fitting import (
        GeneralFitting,
    )
from deepmd.utils.version import (
    check_version_compatibility,
)


@InvarFitting.register("ener")
class EnergyFittingNet(InvarFitting):
    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: list[int] = [120, 120, 120],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        rcond: float | None = None,
        tot_ener_zero: bool = False,
        trainable: list[bool] | None = None,
        atom_ener: list[float] | None = None,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        layer_name: list[str | None] | None = None,
        use_aparam_as_mask: bool = False,
        spin: Any = None,
        mixed_types: bool = False,
        exclude_types: list[int] = [],
        type_map: list[str] | None = None,
        seed: int | list[int] | None = None,
        default_fparam: list | None = None,
        add_edge_readout: bool = False,
        edge_readout_neuron: list[int] | None = None,
        embedding_width: int = 0,
        norm_fact: list[float] | None = None,
    ) -> None:
        super().__init__(
            var_name="energy",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=1,
            neuron=neuron,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            rcond=rcond,
            tot_ener_zero=tot_ener_zero,
            trainable=trainable,
            atom_ener=atom_ener,
            activation_function=activation_function,
            precision=precision,
            layer_name=layer_name,
            use_aparam_as_mask=use_aparam_as_mask,
            spin=spin,
            mixed_types=mixed_types,
            exclude_types=exclude_types,
            type_map=type_map,
            seed=seed,
            default_fparam=default_fparam,
        )
        self.add_edge_readout = add_edge_readout
        self.edge_readout_neuron = (
            edge_readout_neuron if edge_readout_neuron is not None else neuron
        )
        self.embedding_width = embedding_width
        self.norm_fact = norm_fact if norm_fact is not None else [1.0]

        if self.add_edge_readout:
            if self.embedding_width <= 0:
                raise ValueError(
                    "embedding_width must be > 0 when add_edge_readout is True. "
                    "This is typically set from the descriptor's get_dim_emb()."
                )
            self.edge_nets = NetworkCollection(
                0,
                self.ntypes,
                network_type="fitting_network",
                networks=[
                    FittingNet(
                        self.embedding_width,
                        1,
                        self.edge_readout_neuron,
                        self.activation_function,
                        self.resnet_dt,
                        self.precision,
                        bias_out=True,
                        seed=child_seed(child_seed(seed, 3), 0),
                    )
                ],
            )
        else:
            self.edge_nets = None

    def need_additional_input(self) -> bool:
        return self.add_edge_readout

    def call(
        self,
        descriptor: Array,
        atype: Array,
        gr: Array | None = None,
        g2: Array | None = None,
        h2: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        sw: Array | None = None,
        edge_index: Array | None = None,
    ) -> dict[str, Array]:
        xp = array_api_compat.array_namespace(descriptor)
        result = self._call_common(descriptor, atype, gr, g2, h2, fparam, aparam)
        result_energy = result[self.var_name]

        if self.add_edge_readout and g2 is not None and self.edge_nets is not None:
            nf, nloc = atype.shape[:2]
            norm_e_fact = self.norm_fact[0]
            # static sel mode: g2 is nf x nloc x nnei x e_dim
            edge_feature = g2
            edge_atomic_contrib = self.edge_nets[()].call(edge_feature)
            # edge_atomic_contrib: nf x nloc x nnei x 1
            if sw is not None:
                # sw: nf x nloc x nnei
                edge_atomic_contrib = edge_atomic_contrib * xp.expand_dims(sw, axis=-1)
            # sum over neighbors
            edge_energy = xp.sum(edge_atomic_contrib, axis=-2)
            # edge_energy: nf x nloc x 1
            result_energy = result_energy + edge_energy / norm_e_fact

        result[self.var_name] = result_energy
        return result

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 4, 1)
        data.pop("var_name")
        data.pop("dim_out")
        # Handle edge readout nets
        edge_nets_data = data.pop("edge_nets", None)
        obj = super().deserialize(data)
        if edge_nets_data is not None and obj.add_edge_readout:
            obj.edge_nets = NetworkCollection.deserialize(edge_nets_data)
        return obj

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        dd = {
            **super().serialize(),
            "type": "ener",
            "add_edge_readout": self.add_edge_readout,
            "edge_readout_neuron": self.edge_readout_neuron,
            "embedding_width": self.embedding_width,
            "norm_fact": self.norm_fact,
        }
        if self.add_edge_readout and self.edge_nets is not None:
            dd["edge_nets"] = self.edge_nets.serialize()
        return dd
