# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM GLU energy fitting networks."""

from __future__ import (
    annotations,
)

from typing import (
    Any,
    ClassVar,
)

import torch

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    GLULayer,
    MLPLayer,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
    GeneralFitting,
)
from deepmd.pt.model.task.invar_fitting import (
    InvarFitting,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class GLUFittingNet(torch.nn.Module):
    """
    GLU-based fitting network for SeZM.

    Parameters
    ----------
    in_dim
        Input dimension.
    out_dim
        Output dimension.
    neuron
        Hidden layer sizes. Empty list means direct linear projection.
    activation_function
        Activation function used for GLU gating.
    resnet_dt
        Reserved for compatibility; not used in GLU layers.
    precision
        Numerical precision.
    bias_out
        Whether the output layer uses bias.
    seed
        Random seed.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        neuron: list[int] | None = None,
        activation_function: str = "silu",
        resnet_dt: bool = False,
        precision: str = DEFAULT_PRECISION,
        bias_out: bool = False,
        seed: int | list[int] | None = None,
        trainable: bool | list[bool] = True,
    ) -> None:
        super().__init__()
        if neuron is None:
            neuron = []
        if isinstance(trainable, list):
            trainable = all(trainable)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.neuron = [int(nn_dim) for nn_dim in neuron]
        self.activation_function = activation_function
        self.resnet_dt = bool(resnet_dt)
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.bias_out = bool(bias_out)

        # === Step 1. Build GLU hidden layers ===
        hidden_layers = []
        dim_in = self.in_dim
        for layer_idx, hidden_dim in enumerate(self.neuron):
            hidden_layers.append(
                GLULayer(
                    dim_in,
                    hidden_dim,
                    activation_function=self.activation_function,
                    precision=self.precision,
                    seed=child_seed(seed, layer_idx),
                    trainable=trainable,
                )
            )
            dim_in = hidden_dim
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)

        # === Step 2. Build output projection ===
        self.output_layer = MLPLayer(
            num_in=dim_in,
            num_out=self.out_dim,
            bias=self.bias_out,
            use_timestep=False,
            activation_function=None,
            resnet=False,
            precision=self.precision,
            seed=child_seed(seed, len(self.neuron)),
            trainable=trainable,
        )

        for param in self.parameters():
            param.requires_grad = trainable

    def forward(self, xx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GLU fitting net.

        Parameters
        ----------
        xx
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        for layer in self.hidden_layers:
            xx = layer(xx)
        return self.output_layer(xx)

    def call_until_last(self, xx: torch.Tensor) -> torch.Tensor:
        """
        Return activations before the output projection.

        Parameters
        ----------
        xx
            Input tensor.

        Returns
        -------
        torch.Tensor
            Hidden activations, or input if no hidden layers exist.
        """
        for layer in self.hidden_layers:
            xx = layer(xx)
        return xx

    def serialize(self) -> dict[str, Any]:
        """Serialize the network to a dict."""
        state = self.state_dict()
        return {
            "@class": "GLUFittingNet",
            "@version": 1,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "neuron": self.neuron.copy(),
            "activation_function": self.activation_function,
            "resnet_dt": self.resnet_dt,
            "precision": self.precision,
            "bias_out": self.bias_out,
            "@variables": {key: to_numpy_array(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict) -> GLUFittingNet:
        """Deserialize the network from a dict."""
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        variables = data.pop("@variables", {})
        obj = cls(**data)
        state = {key: to_torch_tensor(value) for key, value in variables.items()}
        obj.load_state_dict(state)
        return obj


class SeZMNetworkCollection(torch.nn.Module):
    """
    Network collection for SeZM fitting networks.

    Parameters
    ----------
    ndim
        The number of type dimensions.
    ntypes
        Number of atom types.
    network_type
        The network type name. Only "sezm_fitting_network" is supported.
    networks
        The networks to initialize with.
    """

    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "sezm_fitting_network": GLUFittingNet,
    }

    def __init__(
        self,
        ndim: int,
        ntypes: int,
        network_type: str = "sezm_fitting_network",
        networks: list[GLUFittingNet | dict | None] | None = None,
    ) -> None:
        super().__init__()
        self.ndim = int(ndim)
        self.ntypes = int(ntypes)
        if network_type not in self.NETWORK_TYPE_MAP:
            raise ValueError(f"Unknown network_type: {network_type}")
        self.network_type = self.NETWORK_TYPE_MAP[network_type]
        if networks is None:
            networks = []

        total = self.ntypes**self.ndim
        self._networks: list[GLUFittingNet | None] = [None for _ in range(total)]
        for idx, network in enumerate(networks):
            self[idx] = network
        if any(net is None for net in self._networks):
            raise RuntimeError("SeZMNetworkCollection is incomplete.")
        self.networks = torch.nn.ModuleList(self._networks)

    def _convert_key(self, key: int | tuple | str) -> int:
        if isinstance(key, int):
            idx = key
        else:
            if isinstance(key, tuple):
                pass
            elif isinstance(key, str):
                key = tuple([int(tt) for tt in key.split("_")[1:]])
            else:
                raise TypeError(key)
            assert isinstance(key, tuple)
            assert len(key) == self.ndim
            idx = sum([tt * self.ntypes**ii for ii, tt in enumerate(key)])
        return idx

    def __getitem__(self, key: int | tuple | str) -> GLUFittingNet:
        idx = self._convert_key(key)
        nn = self._networks[idx]
        assert nn is not None
        return nn

    def __setitem__(self, key: int | tuple | str, value: GLUFittingNet | dict) -> None:
        if isinstance(value, self.network_type):
            network = value
        elif isinstance(value, dict):
            network = self.network_type.deserialize(value)
        else:
            raise TypeError(value)
        idx = self._convert_key(key)
        self._networks[idx] = network

    def serialize(self) -> dict[str, Any]:
        """Serialize the networks to a dict."""
        network_type_map_inv = {v: k for k, v in self.NETWORK_TYPE_MAP.items()}
        return {
            "@class": "NetworkCollection",
            "@version": 1,
            "ndim": self.ndim,
            "ntypes": self.ntypes,
            "network_type": network_type_map_inv[self.network_type],
            "networks": [
                nn.serialize() if nn is not None else None for nn in self._networks
            ],
        }

    @classmethod
    def deserialize(cls, data: dict) -> SeZMNetworkCollection:
        """Deserialize the networks from a dict."""
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        return cls(**data)


@Fitting.register("sezm_ener")
class SeZMEnergyFittingNet(InvarFitting):
    """
    SeZM energy fitting with GLU hidden layers.

    This uses the same configuration keys as the standard energy fitting
    but replaces hidden MLP layers with GLU blocks.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: list[int] = [128, 128, 128],
        bias_atom_e: torch.Tensor | None = None,
        resnet_dt: bool = False,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "silu",
        bias_out: bool = False,
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        seed: int | list[int] | None = None,
        type_map: list[str] | None = None,
        default_fparam: list | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "energy",
            ntypes,
            dim_descrpt,
            1,
            neuron=neuron,
            bias_atom_e=bias_atom_e,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            seed=seed,
            type_map=type_map,
            default_fparam=default_fparam,
            **kwargs,
        )
        self.bias_out = bool(bias_out)
        self._build_glu_fitting_layers()

    def _build_glu_fitting_layers(self) -> None:
        # === Step 1. Derive input/output dimensions ===
        in_dim = (
            self.dim_descrpt
            + self.numb_fparam
            + (0 if self.use_aparam_as_mask else self.numb_aparam)
            + self.dim_case_embd
        )
        net_dim_out = self._net_out_dim()
        n_networks = self.ntypes if not self.mixed_types else 1

        # === Step 2. Build GLU fitting networks ===
        self.filter_layers = SeZMNetworkCollection(
            1 if not self.mixed_types else 0,
            self.ntypes,
            network_type="sezm_fitting_network",
            networks=[
                GLUFittingNet(
                    in_dim,
                    net_dim_out,
                    self.neuron,
                    activation_function=self.activation_function,
                    resnet_dt=self.resnet_dt,
                    precision=self.precision,
                    bias_out=self.bias_out,
                    seed=child_seed(self.seed, idx),
                    trainable=self.trainable,
                )
                for idx in range(n_networks)
            ],
        )
        for param in self.parameters():
            param.requires_grad = self.trainable

    @classmethod
    def deserialize(cls, data: dict) -> GeneralFitting:
        data = data.copy()
        variables = data.pop("@variables")
        nets = data.pop("nets")
        check_version_compatibility(data.pop("@version", 1), 4, 1)
        data.pop("var_name")
        data.pop("dim_out")
        obj = cls(**data)
        for kk in variables.keys():
            obj[kk] = to_torch_tensor(variables[kk])
        obj.filter_layers = SeZMNetworkCollection.deserialize(nets)
        return obj

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            **super().serialize(),
            "type": "sezm_ener",
        }
