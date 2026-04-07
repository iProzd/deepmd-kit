# SPDX-License-Identifier: LGPL-3.0-or-later

import math

import torch
import torch.nn as nn


@torch.jit.script
def aggregate(
    data: torch.Tensor,
    owners: torch.Tensor,
    average: bool = True,
    num_owner: int | None = None,
) -> torch.Tensor:
    """
    Aggregate rows in data by specifying the owners.

    Parameters
    ----------
    data : data tensor to aggregate [n_row, feature_dim]
    owners : specify the owner of each row [n_row, 1]
    average : if True, average the rows, if False, sum the rows.
        Default = True
    num_owner : the number of owners, this is needed if the
        max idx of owner is not presented in owners tensor
        Default = None

    Returns
    -------
    output: [num_owner, feature_dim]
    """
    if num_owner is None or average:
        # requires bincount
        bin_count = torch.bincount(owners)
        bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))
        if (num_owner is not None) and (bin_count.shape[0] != num_owner):
            difference = num_owner - bin_count.shape[0]
            bin_count = torch.cat([bin_count, bin_count.new_ones(difference)])
        else:
            num_owner = bin_count.shape[0]
    else:
        bin_count = None

    output = data.new_zeros([num_owner, data.shape[1]])
    output = output.index_add_(0, owners, data)
    if average:
        assert bin_count is not None
        output = (output.T / bin_count).T
    return output


@torch.jit.script
def get_graph_index(
    nlist: torch.Tensor,
    nlist_mask: torch.Tensor,
    a_nlist_mask: torch.Tensor,
    nall: int,
    use_loc_mapping: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the index mapping for edge graph and angle graph, ready in `aggregate` or `index_select`.

    Parameters
    ----------
    nlist : nf x nloc x nnei
        Neighbor list. (padded neis are set to 0)
    nlist_mask : nf x nloc x nnei
        Masks of the neighbor list. real nei 1 otherwise 0
    a_nlist_mask : nf x nloc x a_nnei
        Masks of the neighbor list for angle. real nei 1 otherwise 0
    nall
        The number of extended atoms.

    Returns
    -------
    edge_index : 2 x n_edge
        n2e_index : n_edge
            Broadcast indices from node(i) to edge(ij), or reduction indices from edge(ij) to node(i).
        n_ext2e_index : n_edge
            Broadcast indices from extended node(j) to edge(ij).
    angle_index : 3 x n_angle
        n2a_index : n_angle
            Broadcast indices from extended node(j) to angle(ijk).
        eij2a_index : n_angle
            Broadcast indices from extended edge(ij) to angle(ijk), or reduction indices from angle(ijk) to edge(ij).
        eik2a_index : n_angle
            Broadcast indices from extended edge(ik) to angle(ijk).
    """
    nf, nloc, nnei = nlist.shape
    _, _, a_nnei = a_nlist_mask.shape
    # nf x nloc x nnei x nnei
    # nlist_mask_3d = nlist_mask[:, :, :, None] & nlist_mask[:, :, None, :]
    a_nlist_mask_3d = a_nlist_mask[:, :, :, None] & a_nlist_mask[:, :, None, :]
    n_edge = nlist_mask.sum().item()
    # n_angle = a_nlist_mask_3d.sum().item()

    # following: get n2e_index, n_ext2e_index, n2a_index, eij2a_index, eik2a_index

    # 1. atom graph
    # node(i) to edge(ij) index_select; edge(ij) to node aggregate
    nlist_loc_index = torch.arange(0, nf * nloc, dtype=nlist.dtype, device=nlist.device)
    # nf x nloc x nnei
    n2e_index = nlist_loc_index.reshape(nf, nloc, 1).expand(-1, -1, nnei)
    # n_edge
    n2e_index = n2e_index[nlist_mask]  # graph node index, atom_graph[:, 0]

    # node_ext(j) to edge(ij) index_select
    frame_shift = torch.arange(0, nf, dtype=nlist.dtype, device=nlist.device) * (
        nall if not use_loc_mapping else nloc
    )
    shifted_nlist = nlist + frame_shift[:, None, None]
    # n_edge
    n_ext2e_index = shifted_nlist[nlist_mask]  # graph neighbor index, atom_graph[:, 1]

    # 2. edge graph
    # node(i) to angle(ijk) index_select
    n2a_index = nlist_loc_index.reshape(nf, nloc, 1, 1).expand(-1, -1, a_nnei, a_nnei)
    # n_angle
    n2a_index = n2a_index[a_nlist_mask_3d]

    # edge(ij) to angle(ijk) index_select; angle(ijk) to edge(ij) aggregate
    edge_id = torch.arange(0, n_edge, dtype=nlist.dtype, device=nlist.device)
    # nf x nloc x nnei
    edge_index = torch.zeros([nf, nloc, nnei], dtype=nlist.dtype, device=nlist.device)
    edge_index[nlist_mask] = edge_id
    # only cut a_nnei neighbors, to avoid nnei x nnei
    edge_index = edge_index[:, :, :a_nnei]
    edge_index_ij = edge_index.unsqueeze(-1).expand(-1, -1, -1, a_nnei)
    # n_angle
    eij2a_index = edge_index_ij[a_nlist_mask_3d]

    # edge(ik) to angle(ijk) index_select
    edge_index_ik = edge_index.unsqueeze(-2).expand(-1, -1, a_nnei, -1)
    # n_angle
    eik2a_index = edge_index_ik[a_nlist_mask_3d]

    edge_index_result = torch.stack([n2e_index, n_ext2e_index], dim=0)
    angle_index_result = torch.stack([n2a_index, eij2a_index, eik2a_index], dim=0)

    return edge_index_result, angle_index_result


class GaussianRBF(nn.Module):
    """Gaussian radial basis function expansion.

    Expands scalar r into K dimensions: phi_k(r) = exp(-beta_k * (r - mu_k)^2)
    where mu_k are uniformly distributed in [0, rcut]. C-infinity smooth.

    Parameters
    ----------
    num_basis : int
        Number of basis functions K.
    rcut : float
        Cutoff radius.
    trainable : bool
        Whether mu and beta are trainable parameters.
    """

    def __init__(
        self, num_basis: int = 20, rcut: float = 6.0, trainable: bool = False
    ) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.rcut = rcut
        means = torch.linspace(0.0, rcut, num_basis, device="cpu")
        spacing = rcut / (num_basis - 1)
        betas = torch.full((num_basis,), 1.0 / (2.0 * spacing * spacing), device="cpu")
        if trainable:
            self.means = nn.Parameter(means)
            self.betas = nn.Parameter(betas)
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Expand scalar distances to Gaussian RBF features.

        Parameters
        ----------
        r : torch.Tensor
            Input distances. Shape: (...,) or (..., 1).

        Returns
        -------
        torch.Tensor
            RBF features. Shape: (..., K).
        """
        if r.dim() > 0 and r.shape[-1] == 1:
            r = r.squeeze(-1)
        return torch.exp(-self.betas * (r.unsqueeze(-1) - self.means) ** 2)

    def serialize(self) -> dict:
        """Serialize the module to a dict."""
        return {
            "@class": "GaussianRBF",
            "@version": 1,
            "num_basis": self.num_basis,
            "rcut": self.rcut,
            "trainable": isinstance(self.means, nn.Parameter),
            "means": self.means.detach().cpu().numpy().tolist(),
            "betas": self.betas.detach().cpu().numpy().tolist(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "GaussianRBF":
        """Deserialize the module from a dict."""
        obj = cls(
            num_basis=data["num_basis"],
            rcut=data["rcut"],
            trainable=data["trainable"],
        )
        device = obj.means.device
        dtype = obj.means.dtype
        means_tensor = torch.tensor(data["means"], dtype=dtype, device=device)
        betas_tensor = torch.tensor(data["betas"], dtype=dtype, device=device)
        if data["trainable"]:
            obj.means.data.copy_(means_tensor)
            obj.betas.data.copy_(betas_tensor)
        else:
            obj.means.copy_(means_tensor)
            obj.betas.copy_(betas_tensor)
        return obj


class ChebyshevBasis(nn.Module):
    """Chebyshev polynomial basis for cos(theta) in [-1, 1].

    T_0(x) = 1, T_1(x) = x, T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x).
    Polynomial, C-infinity smooth.

    Parameters
    ----------
    num_basis : int
        Number of basis functions (polynomial order K).
    """

    def __init__(self, num_basis: int = 16) -> None:
        super().__init__()
        self.num_basis = num_basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand cosine values to Chebyshev polynomial features.

        Parameters
        ----------
        x : torch.Tensor
            Input cosine values in [-1, 1]. Shape: (...,) or (..., 1).

        Returns
        -------
        torch.Tensor
            Chebyshev features. Shape: (..., K).
        """
        if x.dim() > 0 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        polys = [torch.ones_like(x), x]
        for n in range(2, self.num_basis):
            polys.append(2.0 * x * polys[-1] - polys[-2])
        return torch.stack(polys[: self.num_basis], dim=-1)

    def serialize(self) -> dict:
        """Serialize the module to a dict."""
        return {
            "@class": "ChebyshevBasis",
            "@version": 1,
            "num_basis": self.num_basis,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "ChebyshevBasis":
        """Deserialize the module from a dict."""
        return cls(num_basis=data["num_basis"])


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Parameters
    ----------
    dim : int
        Normalization dimension.
    eps : float
        Numerical stability epsilon.
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Shape: (..., dim).

        Returns
        -------
        torch.Tensor
            Normalized tensor. Shape: (..., dim).
        """
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)

    def serialize(self) -> dict:
        """Serialize the module to a dict."""
        return {
            "@class": "RMSNorm",
            "@version": 1,
            "dim": self.dim,
            "eps": self.eps,
            "weight": self.weight.detach().cpu().numpy().tolist(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "RMSNorm":
        """Deserialize the module from a dict."""
        obj = cls(dim=data["dim"], eps=data["eps"])
        obj.weight.data.copy_(
            torch.tensor(data["weight"], dtype=obj.weight.dtype)
        )
        return obj


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    FFN(x) = W_down * (silu(W_gate * x) . W_up * x)

    Parameters
    ----------
    dim : int
        Input/output dimension.
    hidden_mult : float
        hidden_dim = int(dim * hidden_mult).
    precision : str
        Parameter precision.
    seed : int or list[int] or None
        Random seed for initialization.
    init : str
        Weight initialization method for internal MLPLayers.
        Supported: "default", "glorot", "kaiming_normal".
    """

    def __init__(
        self,
        dim: int,
        hidden_mult: float = 4.0,
        precision: str = "float32",
        seed: int | list[int] | None = None,
        init: str = "default",
    ) -> None:
        from deepmd.dpmodel.utils.seed import (
            child_seed,
        )
        from deepmd.pt.model.network.mlp import (
            MLPLayer,
        )
        from deepmd.pt.utils.utils import (
            ActivationFn,
        )

        super().__init__()
        self.dim = dim
        self.hidden_mult = hidden_mult
        self.precision = precision
        self.init = init
        hidden_dim = int(dim * hidden_mult)
        self.hidden_dim = hidden_dim
        self.w_gate = MLPLayer(
            dim, hidden_dim, bias=False, precision=precision, init=init, seed=child_seed(seed, 0)
        )
        self.w_up = MLPLayer(
            dim, hidden_dim, bias=False, precision=precision, init=init, seed=child_seed(seed, 1)
        )
        self.w_down = MLPLayer(
            hidden_dim, dim, bias=False, precision=precision, init=init, seed=child_seed(seed, 2)
        )
        self.act = ActivationFn("silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU FFN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Shape: (..., dim).

        Returns
        -------
        torch.Tensor
            Output tensor. Shape: (..., dim).
        """
        return self.w_down(self.act(self.w_gate(x)) * self.w_up(x))

    def serialize(self) -> dict:
        """Serialize the module to a dict."""
        return {
            "@class": "SwiGLUFFN",
            "@version": 2,
            "dim": self.dim,
            "hidden_mult": self.hidden_mult,
            "precision": self.precision,
            "init": self.init,
            "w_gate": self.w_gate.serialize(),
            "w_up": self.w_up.serialize(),
            "w_down": self.w_down.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "SwiGLUFFN":
        """Deserialize the module from a dict."""
        from deepmd.pt.model.network.mlp import (
            MLPLayer,
        )

        obj = cls(
            dim=data["dim"],
            hidden_mult=data["hidden_mult"],
            precision=data["precision"],
            init=data.get("init", "default"),
        )
        obj.w_gate = MLPLayer.deserialize(data["w_gate"])
        obj.w_up = MLPLayer.deserialize(data["w_up"])
        obj.w_down = MLPLayer.deserialize(data["w_down"])
        return obj


class BesselExpansion(nn.Module):
    """Bessel radial basis with polynomial envelope (DimeNet-style).

    phi_k(r) = sqrt(2/rcut) * sin(k*pi*r/rcut) / r * envelope(r)
    envelope(r) = 1 + a*(r/rcut)^p + b*(r/rcut)^(p+1) + c*(r/rcut)^(p+2)

    C-infinity smooth within (0, rcut). The envelope ensures smooth decay to 0
    at the cutoff boundary.

    Parameters
    ----------
    num_radial : int
        Number of Bessel basis functions.
    cutoff : float
        Cutoff radius.
    envelope_exponent : int
        Polynomial envelope exponent p. Default 8.
    learnable : bool
        Whether the Bessel frequencies are trainable.
    """

    def __init__(
        self,
        num_radial: int = 7,
        cutoff: float = 6.0,
        envelope_exponent: int = 8,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope_exponent = envelope_exponent
        self.learnable = learnable
        inv_cutoff = 1.0 / cutoff
        self.register_buffer("inv_cutoff", torch.tensor(inv_cutoff, device="cpu"))
        self.register_buffer(
            "norm_const", torch.tensor((2.0 * inv_cutoff) ** 0.5, device="cpu")
        )

        freqs = math.pi * torch.arange(1, num_radial + 1, dtype=torch.float64, device="cpu")
        if learnable:
            self.frequencies = nn.Parameter(freqs)
        else:
            self.register_buffer("frequencies", freqs)

        # Polynomial envelope coefficients
        p = float(envelope_exponent)
        self.register_buffer("p", torch.tensor(p, device="cpu"))
        self.register_buffer("a", torch.tensor(-(p + 1) * (p + 2) / 2, device="cpu"))
        self.register_buffer("b", torch.tensor(p * (p + 2), device="cpu"))
        self.register_buffer("c", torch.tensor(-p * (p + 1) / 2, device="cpu"))

    def _envelope(self, r_scaled: torch.Tensor) -> torch.Tensor:
        """Polynomial envelope that smoothly goes to 0 at r_scaled=1."""
        p = self.p
        env_val = (
            1.0
            + self.a * r_scaled.pow(p)
            + self.b * r_scaled.pow(p + 1)
            + self.c * r_scaled.pow(p + 2)
        )
        return torch.where(r_scaled < 1.0, env_val, torch.zeros_like(env_val))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Expand distances to Bessel basis features.

        Parameters
        ----------
        dist : torch.Tensor
            Input distances. Shape: (...,).

        Returns
        -------
        torch.Tensor
            Bessel features. Shape: (..., num_radial).
        """
        dist = dist.unsqueeze(-1)  # (..., 1)
        d_scaled = dist * self.inv_cutoff
        rbf = self.norm_const * torch.sin(self.frequencies * d_scaled) / (
            dist + 1e-8
        )
        envelope = self._envelope(d_scaled)
        return rbf * envelope

    def serialize(self) -> dict:
        """Serialize the module to a dict."""
        return {
            "@class": "BesselExpansion",
            "@version": 1,
            "num_radial": self.num_radial,
            "cutoff": self.cutoff,
            "envelope_exponent": self.envelope_exponent,
            "learnable": self.learnable,
            "frequencies": self.frequencies.detach().cpu().numpy().tolist(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "BesselExpansion":
        """Deserialize the module from a dict."""
        obj = cls(
            num_radial=data["num_radial"],
            cutoff=data["cutoff"],
            envelope_exponent=data["envelope_exponent"],
            learnable=data["learnable"],
        )
        freq_tensor = torch.tensor(
            data["frequencies"], dtype=obj.frequencies.dtype, device="cpu"
        )
        if data["learnable"]:
            obj.frequencies.data.copy_(freq_tensor)
        else:
            obj.frequencies.copy_(freq_tensor)
        return obj


# PLACEHOLDER_FOURIER


class FourierExpansion(nn.Module):
    """Fourier expansion for angles.

    Output: [1/sqrt(2), sin(1*theta), ..., sin(K*theta),
             cos(1*theta), ..., cos(K*theta)] / sqrt(pi)
    Total dimension: 1 + 2*max_f.

    Parameters
    ----------
    max_f : int
        Maximum frequency. Output dimension = 1 + 2*max_f.
    learnable : bool
        Whether the frequencies are trainable.
    """

    def __init__(self, max_f: int = 3, learnable: bool = True) -> None:
        super().__init__()
        self.max_f = max_f
        self.learnable = learnable
        self.register_buffer("scale", torch.tensor(math.pi**0.5, device="cpu"))
        freqs = torch.arange(1, max_f + 1, dtype=torch.float64, device="cpu")
        if learnable:
            self.frequencies = nn.Parameter(freqs)
        else:
            self.register_buffer("frequencies", freqs)

    @property
    def num_output(self) -> int:
        """Output dimension."""
        return 1 + 2 * self.max_f

    def forward(self, angle: torch.Tensor) -> torch.Tensor:
        """Expand angles (radians) to Fourier features.

        Parameters
        ----------
        angle : torch.Tensor
            Input angles in radians. Shape: (...,).

        Returns
        -------
        torch.Tensor
            Fourier features. Shape: (..., 1 + 2*max_f).
        """
        result = angle.new_zeros(*angle.shape, 1 + 2 * self.max_f)
        result[..., 0] = 1.0 / (2.0**0.5)
        tmp = angle.unsqueeze(-1) * self.frequencies
        result[..., 1 : self.max_f + 1] = torch.sin(tmp)
        result[..., self.max_f + 1 :] = torch.cos(tmp)
        return result / self.scale

    def serialize(self) -> dict:
        """Serialize the module to a dict."""
        return {
            "@class": "FourierExpansion",
            "@version": 1,
            "max_f": self.max_f,
            "learnable": self.learnable,
            "frequencies": self.frequencies.detach().cpu().numpy().tolist(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "FourierExpansion":
        """Deserialize the module from a dict."""
        obj = cls(max_f=data["max_f"], learnable=data["learnable"])
        freq_tensor = torch.tensor(
            data["frequencies"], dtype=obj.frequencies.dtype, device="cpu"
        )
        if data["learnable"]:
            obj.frequencies.data.copy_(freq_tensor)
        else:
            obj.frequencies.copy_(freq_tensor)
        return obj


# PLACEHOLDER_DIMWISE


def dimwise_softmax(
    feas: torch.Tensor,
    segment: torch.Tensor,
    num_segment: int | None = None,
) -> torch.Tensor:
    """Per-dimension softmax over segments (MatRIS-style).

    For each feature dimension independently, compute softmax-normalized
    weights across all entries belonging to the same segment.

    Parameters
    ----------
    feas : torch.Tensor
        Input features. Shape: (N, D).
    segment : torch.Tensor
        Segment index for each row. Shape: (N,).
    num_segment : int or None
        Total number of segments. Inferred from segment if None.

    Returns
    -------
    torch.Tensor
        Normalized weights. Shape: (N, D).
    """
    N, D = feas.shape
    if num_segment is None:
        num_segment = int(segment.max().item()) + 1
    seg_exp = segment.unsqueeze(1).expand(-1, D)

    # Numerically stable softmax: subtract per-segment max
    feas_max = feas.new_full((num_segment, D), float("-inf"))
    feas_max = feas_max.scatter_reduce(
        0, seg_exp, feas, reduce="amax", include_self=False
    )
    feas_max_gathered = feas_max[segment]

    out = (feas - feas_max_gathered).exp()
    out_sum = feas.new_zeros(num_segment, D)
    out_sum = out_sum.scatter_reduce(
        0, seg_exp, out, reduce="sum", include_self=False
    )
    out_sum_gathered = out_sum[segment]

    return out / (out_sum_gathered + 1e-12)


class GatedMLP(nn.Module):
    """MatRIS-style GatedMLP: core * gate.

    core = silu(RMSNorm(MLP_core(x)))
    gate = sigmoid(RMSNorm(MLP_gate(x)))
    output = core * gate

    Each branch is a 2-hidden-layer MLP (input -> hidden -> hidden -> output).

    Parameters
    ----------
    input_dim : int
        Input dimension.
    hidden_dim : int
        Hidden layer dimension for both branches.
    output_dim : int
        Output dimension.
    precision : str
        Parameter precision.
    seed : int or list[int] or None
        Random seed for initialization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        precision: str = "float32",
        seed: int | list[int] | None = None,
    ) -> None:
        from deepmd.dpmodel.utils.seed import child_seed
        from deepmd.pt.model.network.mlp import MLPLayer

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.precision = precision

        # Core branch: input -> hidden -> hidden -> output
        self.core_l1 = MLPLayer(
            input_dim, hidden_dim, bias=False, precision=precision,
            seed=child_seed(seed, 0),
        )
        self.core_l2 = MLPLayer(
            hidden_dim, hidden_dim, bias=False, precision=precision,
            seed=child_seed(seed, 1),
        )
        self.core_l3 = MLPLayer(
            hidden_dim, output_dim, bias=False, precision=precision,
            seed=child_seed(seed, 2),
        )
        # Gate branch: input -> hidden -> hidden -> output
        self.gate_l1 = MLPLayer(
            input_dim, hidden_dim, bias=False, precision=precision,
            seed=child_seed(seed, 3),
        )
        self.gate_l2 = MLPLayer(
            hidden_dim, hidden_dim, bias=False, precision=precision,
            seed=child_seed(seed, 4),
        )
        self.gate_l3 = MLPLayer(
            hidden_dim, output_dim, bias=False, precision=precision,
            seed=child_seed(seed, 5),
        )
        # Norms (create on same device as MLPLayer params)
        from deepmd.pt.utils import env as _env
        with torch.device(_env.DEVICE):
            self.core_norm = RMSNorm(output_dim)
            self.gate_norm = RMSNorm(output_dim)

        from deepmd.pt.utils.utils import ActivationFn
        self.act = ActivationFn("silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Shape: (..., input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor. Shape: (..., output_dim).
        """
        # Core branch
        core = self.act(self.core_l1(x))
        core = self.act(self.core_l2(core))
        core = self.core_l3(core)
        core = torch.nn.functional.silu(self.core_norm(core))

        # Gate branch
        gate = self.act(self.gate_l1(x))
        gate = self.act(self.gate_l2(gate))
        gate = self.gate_l3(gate)
        gate = torch.sigmoid(self.gate_norm(gate))

        return core * gate

    def serialize(self) -> dict:
        """Serialize the module to a dict."""
        return {
            "@class": "GatedMLP",
            "@version": 1,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "precision": self.precision,
            "core_l1": self.core_l1.serialize(),
            "core_l2": self.core_l2.serialize(),
            "core_l3": self.core_l3.serialize(),
            "gate_l1": self.gate_l1.serialize(),
            "gate_l2": self.gate_l2.serialize(),
            "gate_l3": self.gate_l3.serialize(),
            "core_norm": self.core_norm.serialize(),
            "gate_norm": self.gate_norm.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "GatedMLP":
        """Deserialize the module from a dict."""
        from deepmd.pt.model.network.mlp import MLPLayer
        from deepmd.pt.utils import env as _env

        obj = cls(
            input_dim=data["input_dim"],
            hidden_dim=data["hidden_dim"],
            output_dim=data["output_dim"],
            precision=data["precision"],
        )
        obj.core_l1 = MLPLayer.deserialize(data["core_l1"])
        obj.core_l2 = MLPLayer.deserialize(data["core_l2"])
        obj.core_l3 = MLPLayer.deserialize(data["core_l3"])
        obj.gate_l1 = MLPLayer.deserialize(data["gate_l1"])
        obj.gate_l2 = MLPLayer.deserialize(data["gate_l2"])
        obj.gate_l3 = MLPLayer.deserialize(data["gate_l3"])
        with torch.device(_env.DEVICE):
            obj.core_norm = RMSNorm.deserialize(data["core_norm"])
            obj.gate_norm = RMSNorm.deserialize(data["gate_norm"])
        return obj
