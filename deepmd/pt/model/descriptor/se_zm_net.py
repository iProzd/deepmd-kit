# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SeZM-Net (Base) equivariant descriptor for DeePMD-kit (PyTorch backend).

This implementation is designed around two non-negotiables:

1) Conservative forces: the descriptor is computed from differentiable energy.
2) Speed-first inference: edge geometry and Wigner-D rotation blocks are computed
   exactly once per `forward()` and reused by all interaction blocks.

Interaction blocks are implemented in `se_zm_block.py`.
"""

from __future__ import (
    annotations,
)

import math
from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.pt.model.network.mlp import (
    EmbeddingNet,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
    TypeEmbedNetConsistent,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.pt.utils.preprocess import (
    compute_smooth_weight,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .se_zm_block import (
    EquivariantFFN,
    PerDegreeLinear,
    SeZMInteractionBlock,
    SO2Convolution,
    _so3_dim_of_lmax,
)
from .wigner_d import (
    WignerDCalc,
    WignerDCalcBase,
    WignerDCalcParallel,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from deepmd.utils.data_system import (
        DeepmdDataSystem,
    )
    from deepmd.utils.env_mat_stat import (
        StatItem,
    )
    from deepmd.utils.path import (
        DPPath,
    )


def _as_int_list(value: list[int] | int, *, ntypes: int | None) -> list[int]:
    """Convert `sel` to a list of integers."""
    if isinstance(value, int):
        if ntypes is None:
            raise ValueError("`ntypes` must be provided when `sel` is an int")
        return [int(value)] * int(ntypes)
    if len(value) == 0:
        raise ValueError("`sel` must be non-empty")
    return [int(x) for x in value]


def safe_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Compute vector norm with an epsilon lower bound.

    The epsilon is automatically inferred from the input dtype.

    Parameters
    ----------
    x
        Input tensor with shape (..., 3).

    Returns
    -------
    torch.Tensor
        Norm with shape (..., 1), clamped to be >= machine epsilon.
    """
    eps = torch.finfo(x.dtype).eps
    return torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True)).clamp(min=eps)


def init_edge_rot_mat(edge_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation matrices that align each edge to the local + Z axis.

    The returned rotation is a global->local transform: ``v_local = R @ v_global``.
    So, for unit edge direction vector ``u``, ``R @ u = (0, 0, 1)``.

    Notes
    -----
    This routine constructs an orthonormal right-handed frame (x_hat, y_hat, z_hat)
    per edge via a simple Gram-Schmidt process::

        z_hat = edge_vec / ||edge_vec||           # local +z direction
        x_hat = normalize(ref - (ref·z_hat) z_hat)  # orthogonal to z_hat
        y_hat = z_hat x x_hat                       # right-handed

    where ``ref`` is a reference axis that is not nearly colinear with ``z_hat``.

    The rotation matrix stacks these basis vectors as rows::

        R = [x_hat^T; y_hat^T; z_hat^T]

    This makes ``R`` a global->local transform, because each row computes the
    dot product with the corresponding local basis vector.

    The reference-axis switch introduces a piecewise definition. For a smoother
    frame construction (especially for higher-order gradients), consider a
    Householder/Frisvad frame.

    Parameters
    ----------
    edge_vec
        Edge vectors with shape (n_edges, 3).

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape (n_edges, 3, 3).
    """
    # === Step 1. Normalize edge direction (local z) ===
    # z_hat is the unit edge direction (center -> neighbor).
    z_hat = edge_vec / safe_norm(edge_vec)

    # === Step 2. Construct x-axis by Gram-Schmidt against a reference ===
    # Choose a reference axis that is not nearly parallel to z_hat to avoid
    # catastrophic cancellation in the Gram-Schmidt projection.
    candi_1 = edge_vec.new_tensor([1.0, 0.0, 0.0]).expand_as(edge_vec)
    candi_2 = edge_vec.new_tensor([0.0, 1.0, 0.0]).expand_as(edge_vec)
    use_alt = torch.abs(torch.sum(z_hat * candi_1, dim=-1, keepdim=True)) > 0.9
    ref = torch.where(use_alt, candi_2, candi_1)

    # Remove the component along z_hat to obtain a vector orthogonal to z_hat.
    proj = torch.sum(ref * z_hat, dim=-1, keepdim=True) * z_hat
    x_hat = ref - proj
    x_hat = x_hat / safe_norm(x_hat)

    # === Step 3. Construct y-axis (right-handed) ===
    # Cross product enforces a right-handed frame: (x_hat, y_hat, z_hat).
    y_hat = torch.cross(z_hat, x_hat, dim=-1)
    y_hat = y_hat / safe_norm(y_hat)

    # === Step 4. Stack rows to form global->local rotation ===
    # Row-stacking ensures v_local = R @ v_global.
    return torch.stack([x_hat, y_hat, z_hat], dim=-2)


def init_edge_rot_mat_frisvad(edge_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation matrices that align each edge to the local + Z axis.

    The returned rotation is a global->local transform: ``v_local = R @ v_global``.
    So, for unit edge direction vector ``u``, ``R @ u = (0, 0, 1)``.

    Notes
    -----
    This routine constructs an orthonormal right-handed frame (x_hat, y_hat, z_hat)
    per edge using the Frisvad method (closed-form ONB from a unit vector).

    The Frisvad closed-form is singular at ``z_hat = (0, 0, -1)``, due to the
    ``1 / (1 + nz)`` denominator. For the singular neighborhood near ``-Z``, the
    basis must NOT fall back to fixed axes, otherwise x_hat/y_hat may not be
    exactly perpendicular to the current ``z_hat``. Instead, we build a strict
    orthonormal pair from the current ``z_hat`` via cross products, guaranteeing
    that the returned matrix is a proper rotation and that ``R @ z_hat = (0,0,1)``
    up to floating-point error.

    Given unit vector z_hat = (nx, ny, nz), for nz > -1, define::

        a = 1 / (1 + nz)
        b = -nx * ny * a
        x_hat = (1 - nx ^ 2 * a, b, -nx)
        y_hat = (b, 1 - ny ^ 2 * a, -ny)

    This yields an orthonormal basis with x_hat ⟂ z_hat, y_hat ⟂ z_hat and
    x_hat X y_hat = z_hat (right-handed). For nz close to -1, we fall back to a
    strict cross-product basis built from the current z_hat.

    The rotation matrix stacks these basis vectors as rows::

        R = [x_hat^T; y_hat^T; z_hat^T]

    This makes ``R`` a global->local transform, because each row computes the
    dot product with the corresponding local basis vector.

    Parameters
    ----------
    edge_vec
        Edge vectors with shape (n_edges, 3).

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape (n_edges, 3, 3).
    """
    # === Step 1. Normalize edge direction (local z) ===
    # z_hat is the unit edge direction (center -> neighbor).
    z_hat = edge_vec / safe_norm(edge_vec)
    nx = z_hat[..., 0:1]
    ny = z_hat[..., 1:2]
    nz = z_hat[..., 2:3]

    # === Step 2. Frisvad closed-form orthonormal basis (non-singular) ===
    # The closed-form uses a = 1 / (1 + nz), which is singular at nz = -1.
    # Compute it with a safe denominator, then select by a singular mask.
    eps = 1.0e-6
    singular = nz < (-1.0 + eps)

    denom = 1.0 + nz
    denom_safe = torch.where(singular, torch.ones_like(denom), denom)
    a = 1.0 / denom_safe
    b = -nx * ny * a

    x_main = torch.cat([1.0 - nx * nx * a, b, -nx], dim=-1)
    y_main = torch.cat([b, 1.0 - ny * ny * a, -ny], dim=-1)

    # === Step 3. Strict fallback for the singular neighborhood (z_hat ~= -Z) ===
    # Build x_hat/y_hat from the current z_hat so that:
    #   x_hat ⟂ z_hat, y_hat ⟂ z_hat, and (x_hat, y_hat, z_hat) is right-handed.
    # In the singular neighborhood near -Z, ref = +X is guaranteed not parallel to z_hat.
    ref = edge_vec.new_tensor([1.0, 0.0, 0.0]).expand_as(edge_vec)
    x_fb = torch.cross(ref, z_hat, dim=-1)
    x_fb = x_fb / safe_norm(x_fb)
    y_fb = torch.cross(z_hat, x_fb, dim=-1)
    y_fb = y_fb / safe_norm(y_fb)

    mask3 = singular.expand_as(edge_vec)
    x_hat = torch.where(mask3, x_fb, x_main)
    y_hat = torch.where(mask3, y_fb, y_main)

    # Normalize to protect against numerical drift (and to match your existing style).
    x_hat = x_hat / safe_norm(x_hat)
    y_hat = y_hat / safe_norm(y_hat)

    # === Step 4. Stack rows to form global->local rotation ===
    # Row-stacking ensures v_local = R @ v_global.
    return torch.stack([x_hat, y_hat, z_hat], dim=-2)


class C2CutoffEnvelope(nn.Module):
    """
    C^2 polynomial cutoff envelope (quintic smoothstep to zero at rcut).

    E(x) = 1 - 10 x^3 + 15 x^4 - 6 x^5,  for x = r/rcut in [0, 1]
    E(x) = 0,                            for x >= 1
    """

    def __init__(self, rcut: float) -> None:
        super().__init__()
        self.rcut = float(rcut)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        r
            Pair distances with shape (..., 1).

        Returns
        -------
        torch.Tensor
            Envelope values with shape (..., 1).
        """
        x = (r / self.rcut).clamp(min=0.0, max=1.0)
        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2
        x5 = x4 * x
        return 1.0 - 10.0 * x3 + 15.0 * x4 - 6.0 * x5


class RadialBasis(nn.Module):
    """
    Spherical Bessel radial basis with C^2 cutoff envelope.

    Frequencies are trainable nn.Parameter, allowing the model
    to learn optimal radial basis spacing during training.

    Notes
    -----
    This implementation uses the order-0 spherical Bessel function
    ``j0(x) = sin(x) / x`` evaluated at ``x = w_n * r``::

        phi_n(r) = sin(w_n * r) / r

    The initial frequencies follow a common "Bessel" spacing::

        w_n = n * pi / rcut, for n = 1..n_radial (in 1/Å)

    The ``r -> 0`` limit is finite::

        lim_{r->0} sin(w_n * r) / r = w_n

    The C^2 cutoff envelope is multiplied directly into the output to ensure
    strict smoothness at ``rcut``.

    Parameters
    ----------
    rcut
        Cutoff radius in Å.
    n_radial
        Number of basis functions.
    dtype
        Floating-point dtype for the radial basis frequencies and outputs.
    """

    def __init__(
        self,
        rcut: float,
        n_radial: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rcut = float(rcut)
        self.n_radial = int(n_radial)
        if self.n_radial <= 0:
            raise ValueError("`n_radial` must be positive")
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.eps = torch.finfo(self.dtype).eps

        # Frequencies: n*pi/rcut, n=1..n_radial.
        # Stored as trainable nn.Parameter.
        freqs = torch.arange(
            1,
            self.n_radial + 1,
            device=self.device,
            dtype=self.dtype,
        ) * (math.pi / self.rcut)
        self.freqs = nn.Parameter(freqs.view(1, -1), requires_grad=True)

        self.envelope = C2CutoffEnvelope(self.rcut)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        r
            Pair distances with shape (..., 1) in Å.

        Returns
        -------
        torch.Tensor
            Radial basis multiplied by C^2 cutoff envelope with shape (..., n_radial).
            The output is smoothly truncated to zero at r = rcut.
        """
        # === Step 1. Avoid division by zero (preserve r->0 limit) ===
        rr = r.clamp(min=self.eps)

        # === Step 2. Evaluate spherical Bessel-like features ===
        # phi_n(r) = sin(w_n * r) / r
        # r->0 limit: sin(w r)/r -> w
        raw = torch.sin(rr * self.freqs) / rr

        # === Step 3. Apply C^2 envelope for smooth cutoff ===
        envelope = self.envelope(rr)
        return raw * envelope

    def serialize(self) -> dict[str, Any]:
        """Serialize RadialBasis including trainable frequencies."""
        return {
            "@class": "RadialBasis",
            "@version": 1,  # keep 1 at devel stage
            "rcut": self.rcut,
            "n_radial": self.n_radial,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "freqs": to_numpy_array(self.freqs),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> RadialBasis:
        """Deserialize RadialBasis including trainable frequencies."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "RadialBasis":
            raise ValueError(f"Invalid class for RadialBasis: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported RadialBasis version: {version}")
        precision = data["precision"]
        dtype = PRECISION_DICT[precision]
        obj = cls(
            rcut=float(data["rcut"]),
            n_radial=int(data["n_radial"]),
            dtype=dtype,
        )
        obj.freqs.data.copy_(
            torch.as_tensor(data["freqs"], dtype=obj.dtype, device=obj.device)
        )
        return obj


@dataclass(frozen=True, slots=True)
class EdgeFeatureCache:
    """
    Global edge feature cache created once per forward().

    Invariants
    ----------
    - all edges are valid (no padding, no excluded type pair, within rcut)
    - all edge-dependent tensors are aligned on the same edge axis (n_edges)
    - edge_rbf already includes the C^2 cutoff envelope for smoothness
    """

    src: torch.Tensor  # (n_edges,)
    dst: torch.Tensor  # (n_edges,)
    edge_vec: torch.Tensor  # (n_edges, 3)
    edge_len: torch.Tensor  # (n_edges, 1)
    edge_unit: torch.Tensor  # (n_edges, 3)
    edge_rbf: torch.Tensor  # (n_edges, n_rbf)
    edge_sw: torch.Tensor  # (n_edges, 1)
    D_list: list[torch.Tensor]  # D_list[l] : (n_edges, 2l+1, 2l+1)
    Dt_list: list[torch.Tensor]  # transpose/inverse blocks
    sw: torch.Tensor  # (nf, nloc, nnei, 1)
    inv_sqrt_deg: torch.Tensor  # (N, 1, 1)

    @property
    def num_edges(self) -> int:
        return int(self.src.numel())


class GeometricInitialEmbedding(nn.Module):
    """
    Geometric initial embedding that adds zonal (m=0) rotated features.

    This module computes radial-transformed features for each degree l >= 1 and
    rotates them using the zonal (m=0) column of the cached inverse Wigner-D blocks
    (local->global).
    The l=0 component is not computed here since it comes from type embedding.

    Notes
    -----
    The radial network outputs `lmax * channels` features,
    corresponding to l = 1, 2, ..., lmax. In the forward pass, indexing uses `l-1`
    to map from degree l to the radial feature array.
    """

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        n_radial: int,
        radial_hidden: list[int],
        activation_function: str,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.n_radial = int(n_radial)
        self.ebed_dim = _so3_dim_of_lmax(self.lmax)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        # Output only for l >= 1, since l=0 comes from type embedding.
        self.radial_net = EmbeddingNet(
            self.n_radial,
            [*radial_hidden, self.lmax * self.channels],
            activation_function=activation_function,
            precision=self.precision,
            resnet_dt=False,
            seed=seed,
            trainable=trainable,
        )

    def forward(self, *, n_nodes: int, edge_cache: EdgeFeatureCache) -> torch.Tensor:
        """
        Parameters
        ----------
        n_nodes
            Number of nodes (nf*nloc).
        edge_cache
            Per-edge cache containing geometry, weights, radial basis, and Wigner-D blocks.

        Returns
        -------
        torch.Tensor
            Initial features to add with shape (N, D, C). l=0 is guaranteed zero.
        """
        if edge_cache.num_edges == 0:
            return torch.zeros(
                n_nodes,
                self.ebed_dim,
                self.channels,
                device=edge_cache.edge_vec.device,
                dtype=edge_cache.edge_vec.dtype,
            )

        device = edge_cache.edge_vec.device
        dtype = edge_cache.edge_vec.dtype
        edge_sw = edge_cache.edge_sw  # (n_edges, 1)
        # Output shape: (n_edges, lmax, channels) for l=1..lmax
        radial = self.radial_net(edge_cache.edge_rbf).view(
            edge_cache.num_edges, self.lmax, self.channels
        )

        out = torch.zeros(
            n_nodes, self.ebed_dim, self.channels, device=device, dtype=dtype
        )

        start = 0
        for l in range(self.lmax + 1):
            dim = 2 * l + 1
            if l == 0:
                start += dim
                continue
            m0 = l  # packed index of m=0 inside this l-block
            d_col = edge_cache.Dt_list[l][:, :, m0]  # (n_edges, 2l+1)
            msg_global = d_col.unsqueeze(-1) * radial[:, l - 1, :].unsqueeze(1)
            msg_global = msg_global * edge_sw.unsqueeze(-1)  # (n_edges, 2l+1, channels)

            out[:, start : start + dim, :].index_add_(0, edge_cache.dst, msg_global)
            start += dim

        # Apply neighbor normalization (graph-style degree normalization).
        out = out * edge_cache.inv_sqrt_deg

        return out


@BaseDescriptor.register("se_zm_net")
@BaseDescriptor.register("se_zm")
class DescrptSeZMNet(BaseDescriptor, nn.Module):
    """
    SeZM-Net equivariant descriptor for DeePMD-kit.

    Parameters
    ----------
    rcut
        Cutoff radius in Å.
    rcut_smth
        Smooth weight start in Å.
    sel
        Maximum number of neighbors per type within `rcut`.
    lmax
        Maximum order, only used when `l_schedule` is None.
    l_schedule
        Pyramid schedule of lmax per block, e.g. [2, 2, 1, 0]. Must be non-increasing.
    channels
        Channels per (l,m) coefficient.
    n_radial
        Number of radial basis functions.
    radial_mlp
        Hidden sizes for radial networks.
    n_blocks
        Number of blocks (only used when `l_schedule` is None).
    ffn_neuron
        Hidden sizes for the l=0 FFN in each block.
    wigner_parallel
        If True, use the block-diagonal parallel Wigner-D implementation.
    set_davg_zero
        If True, keep mean at zeros when computing stats.
    activation_function
        Activation function used by deepmd EmbeddingNet.
    precision
        Internal precision.
    exclude_types
        List of excluded type pairs.
    env_protection
        Protection parameter for DeePMD env-mat stat (kept for compatibility).
    trainable
        Whether parameters are trainable.
    seed
        Random seed(s).
    ntypes
        Number of element types.
    type_map
        Type names.
    """

    _ENV_DIM: int = 4

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: list[int] | int,
        *,
        lmax: int = 2,
        l_schedule: list[int] | None = None,
        channels: int = 96,
        n_radial: int = 8,
        radial_mlp: list[int] = [64, 64],
        n_blocks: int = 4,
        so2_layers: int = 2,
        ffn_neuron: list[int] = [128],
        wigner_parallel: bool = False,
        set_davg_zero: bool = False,
        activation_function: str = "silu",
        precision: str = "float64",
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        ntypes: int | None = None,
        type_map: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()

        self.rcut = float(rcut)
        self.rcut_smth = float(rcut_smth)
        self.env_protection = float(env_protection)

        self.sel = _as_int_list(sel, ntypes=ntypes)
        self.ntypes = len(self.sel)
        self.type_map = type_map
        self.nnei = int(sum(self.sel))
        self.ndescrpt = int(self.nnei * self._ENV_DIM)

        self.channels = int(channels)
        self.n_radial = int(n_radial)
        self.radial_mlp = list(radial_mlp)
        self.so2_layers = int(so2_layers)
        self.ffn_neuron = list(ffn_neuron)
        self.wigner_parallel = bool(wigner_parallel)

        self.set_davg_zero = bool(set_davg_zero)
        self.activation_function = str(activation_function)
        self.precision = str(precision)
        self.dtype = PRECISION_DICT[self.precision]
        self.device = env.DEVICE

        self.trainable = bool(trainable)
        self.seed = seed

        # === Step 1. L schedule ===
        if l_schedule is None:
            self.l_schedule = [int(lmax)] * int(n_blocks)
        else:
            self.l_schedule = [int(x) for x in l_schedule]
        if len(self.l_schedule) == 0:
            raise ValueError("`l_schedule` must be non-empty")
        if any(x < 0 for x in self.l_schedule):
            raise ValueError("`l_schedule` entries must be non-negative")
        if any(
            self.l_schedule[i] < self.l_schedule[i + 1]
            for i in range(len(self.l_schedule) - 1)
        ):
            raise ValueError("`l_schedule` must be non-increasing (pyramid schedule)")
        if self.l_schedule[-1] != 0:
            raise ValueError("`l_schedule` must end at 0 for stable cutoff handling")

        self.lmax = int(self.l_schedule[0])

        # === Step 2. Statistics buffers ===
        _shape = (self.ntypes, self.nnei, self._ENV_DIM)
        mean = torch.zeros(_shape, dtype=self.dtype, device=self.device)
        stddev = torch.ones(_shape, dtype=self.dtype, device=self.device)
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.stats: dict[str, StatItem] | None = None

        # === Step 3. Excluded type pairs ===
        self.reinit_exclude(exclude_types)

        # === Step 4. Core modules ===
        self.type_embedding = TypeEmbedNet(
            type_nums=self.ntypes,
            embed_dim=self.channels,
            precision=self.precision,
            seed=self.seed,
            type_map=type_map,
            trainable=self.trainable,
        )

        self.radial_basis = RadialBasis(
            self.rcut,
            self.n_radial,
            dtype=self.dtype,
        )
        wigner_lmax = self.l_schedule[0]
        if self.wigner_parallel:
            self.wigner_calc: WignerDCalcBase = WignerDCalcParallel(
                lmax=wigner_lmax, dtype=self.dtype
            )
        else:
            self.wigner_calc = WignerDCalc(lmax=wigner_lmax, dtype=self.dtype)

        if self.l_schedule[0] > 0:
            self.gie = GeometricInitialEmbedding(
                lmax=self.l_schedule[0],
                channels=self.channels,
                n_radial=self.n_radial,
                radial_hidden=self.radial_mlp,
                activation_function=self.activation_function,
                dtype=self.dtype,
                seed=self.seed,
                trainable=self.trainable,
            )
        else:
            self.gie = None

        blocks: list[SeZMInteractionBlock] = []
        for l_b in self.l_schedule:
            blocks.append(
                SeZMInteractionBlock(
                    lmax=l_b,
                    channels=self.channels,
                    n_radial=self.n_radial,
                    radial_hidden=self.radial_mlp,
                    so2_layers=self.so2_layers,
                    ffn_neuron=self.ffn_neuron,
                    activation_function=self.activation_function,
                    dtype=self.dtype,
                    seed=self.seed,
                    trainable=self.trainable,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        for p in self.parameters():
            p.requires_grad = self.trainable

    # === DeePMD descriptor interface ===
    def get_rcut(self) -> float:
        return self.rcut

    def get_rcut_smth(self) -> float:
        return self.rcut_smth

    def get_sel(self) -> list[int]:
        return self.sel

    def get_nsel(self) -> int:
        return sum(self.sel)

    def get_ntypes(self) -> int:
        return self.ntypes

    def get_type_map(self) -> list[str]:
        return self.type_map if self.type_map is not None else []

    def get_dim_out(self) -> int:
        return self.channels

    def get_dim_emb(self) -> int:
        return self.get_dim_out()

    def mixed_types(self) -> bool:
        return False

    def has_message_passing(self) -> bool:
        return bool(len(self.blocks) > 0 and self.lmax > 0)

    def need_sorted_nlist_for_lower(self) -> bool:
        return False

    def get_env_protection(self) -> float:
        return self.env_protection

    def share_params(
        self, base_class: Any, shared_level: int, resume: bool = False
    ) -> None:
        raise NotImplementedError("share_params is not supported for se_zm_net")

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
    ) -> None:
        raise NotImplementedError("change_type_map is not supported for se_zm_net")

    def reinit_exclude(
        self, exclude_types: list[tuple[int, int]] | None = None
    ) -> None:
        if exclude_types is None:
            exclude_types = []
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def set_stat_mean_and_stddev(
        self, mean: torch.Tensor, stddev: torch.Tensor
    ) -> None:
        self.mean = mean
        self.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean, self.stddev

    def compute_input_stats(
        self, merged: Callable[[], list[dict]] | list[dict], path: DPPath | None = None
    ) -> None:
        env_mat_stat = EnvMatStatSe(self)
        if path is not None:
            path = path / env_mat_stat.get_hash()
        if path is None or not path.is_dir():
            sampled = merged() if callable(merged) else merged
        else:
            sampled = []
        env_mat_stat.load_or_compute_stats(sampled, path)
        self.stats = env_mat_stat.stats
        mean, stddev = env_mat_stat()
        if not self.set_davg_zero:
            self.mean.copy_(
                torch.tensor(mean, device=self.device, dtype=self.mean.dtype)
            )
        self.stddev.copy_(
            torch.tensor(stddev, device=self.device, dtype=self.stddev.dtype)
        )

    def get_stats(self) -> dict[str, StatItem]:
        if self.stats is None:
            raise RuntimeError(
                "The statistics of the descriptor has not been computed."
            )
        return self.stats

    def __setitem__(self, key: str, value: Any) -> None:
        if key in ("avg", "data_avg", "davg"):
            self.mean = value
            return
        if key in ("std", "data_std", "dstd"):
            self.stddev = value
            return
        raise KeyError(key)

    def __getitem__(self, key: str) -> Any:
        if key in ("avg", "data_avg", "davg"):
            return self.mean
        if key in ("std", "data_std", "dstd"):
            return self.stddev
        raise KeyError(key)

    def _apply_zbl_gating_hook(
        self, x0: torch.Tensor, edge_cache: EdgeFeatureCache
    ) -> torch.Tensor:
        """
        Placeholder hook for ZBL gating (not implemented).

        Parameters
        ----------
        x0
            Scalar features with shape (N, C).
        edge_cache
            Per-edge cache.

        Returns
        -------
        torch.Tensor
            Updated scalar features with shape (N, C).
        """
        del edge_cache
        return x0

    def _build_edge_cache(
        self,
        *,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None,
        pair_keep_mask: torch.Tensor,
    ) -> EdgeFeatureCache:
        """
        Build the global edge cache from DeePMD padded neighbor list.

        This converts DeePMD's per-frame padded neighbor list into a flat list of
        valid edges used by message passing, and computes all per-edge tensors that
        are reused across blocks.

        The resulting cache contains:

        - per-edge geometry: ``edge_vec``, ``edge_len``, ``edge_unit``
        - per-edge smooth weights: DeePMD smooth weight ``edge_sw`` and C^2 envelope
          ``edge_envelop`` (kept separate for strict smoothness)
        - per-edge radial basis: ``edge_rbf`` (raw, without envelope)
        - per-edge rotation blocks: real-basis Wigner-D matrices ``D_list`` and ``Dt_list``
        - destination-node normalization: ``inv_sqrt_deg`` for neighbor norm

        Notes
        -----
        Input formats follow DeePMD conventions:

        - ``extended_coord`` is flattened as ``(nf, nall*3)`` and reshaped to ``(nf, nall, 3)``.
        - ``nlist`` has shape ``(nf, nloc, nnei)`` and stores indices into the extended axis
          (``0..nall-1``), with ``-1`` indicating padding.
        - ``mapping`` (when provided) maps extended indices to local indices ``0..nloc-1``.
          When ``mapping`` is ``None``, the function assumes the neighbor indices are already local.

        This function avoids branchy gather on ``-1`` indices by appending a sentinel coordinate
        far outside the cutoff and mapping padding entries to that sentinel. Those padded pairs
        are then removed by the keep/within-cutoff masks before any normalization or rotation.

        Parameters
        ----------
        extended_coord
            Extended coordinates with shape (nf, nall*3).
        extended_atype
            Extended atom types with shape (nf, nall).
            Currently unused; reserved for potential type-dependent filtering.
        nlist
            Neighbor list with shape (nf, nloc, nnei).
        mapping
            Mapping from extended indices to local indices with shape (nf, nall), or None.
        pair_keep_mask
            Pair keep mask from `PairExcludeMask` with shape (nf, nloc, nnei). True means keep.
        """
        nf, nloc, nnei = nlist.shape
        n_nodes = int(nf * nloc)
        coord = extended_coord.view(nf, -1, 3)  # (nf, nall, 3)
        nall = coord.shape[1]

        # === Step 1. Build per-pair geometry with padding-safe gather ===
        # DeePMD uses -1 for padding in nlist. torch.gather cannot index -1, so we:
        #   - replace padding indices with a sentinel index (nall)
        #   - append a sentinel coordinate far outside the cutoff
        # This keeps the gather path branch-free and ensures padded pairs have r > rcut.
        valid_nlist = nlist >= 0
        keep = valid_nlist & pair_keep_mask

        # Pad index nall points to a far-away coordinate to force r > rcut -> sw==0.
        pad_index = torch.full_like(nlist, nall)
        gather_index = torch.where(valid_nlist, nlist, pad_index)  # (nf, nloc, nnei)
        index = (
            gather_index.view(nf, -1).unsqueeze(-1).expand(-1, -1, 3)
        )  # (nf, nloc*nnei, 3)
        coord_pad = torch.cat(
            [coord, coord[:, -1:, :] + self.rcut * 2.0], dim=1
        )  # (nf, nall+1, 3)
        nei_pos = torch.gather(coord_pad, 1, index).view(nf, nloc, nnei, 3)
        atom_pos = coord[:, :nloc].view(nf, nloc, 1, 3)
        diff = nei_pos - atom_pos  # (nf, nloc, nnei, 3)
        length = torch.linalg.norm(diff, dim=-1, keepdim=True)  # (nf, nloc, nnei, 1)

        # === Step 2. Smooth weight `sw` ===
        # sw is the DeePMD smooth switching weight in [0, 1], applied per neighbor pair.
        sw = compute_smooth_weight(
            length, self.rcut_smth, self.rcut
        )  # (nf, nloc, nnei, 1)
        sw = sw * keep.unsqueeze(-1).to(dtype=sw.dtype)

        # === Step 3. Filter valid edges for message passing ===
        # An edge is valid if:
        #   - it is not padding (nlist >= 0)
        #   - the type pair is allowed (pair_keep_mask)
        #   - its length is strictly within rcut
        within = length < self.rcut
        edge_keep = (keep & within.squeeze(-1)).view(-1)
        edge_idx = torch.nonzero(edge_keep, as_tuple=False).squeeze(-1)
        edge_sw = sw.reshape(-1, 1)[edge_idx]

        if edge_idx.numel() == 0:
            # No edges -> empty cache, but sw is still returned for compatibility.
            device = extended_coord.device
            dtype = extended_coord.dtype
            empty_long = torch.empty(0, dtype=torch.long, device=device)
            empty_vec = torch.empty(0, 3, dtype=dtype, device=device)
            empty_len = torch.empty(0, 1, dtype=dtype, device=device)
            empty_rbf = torch.empty(
                0, self.radial_basis.n_radial, dtype=dtype, device=device
            )
            inv_sqrt_deg = torch.ones(n_nodes, 1, 1, dtype=dtype, device=device)
            return EdgeFeatureCache(
                src=empty_long,
                dst=empty_long,
                edge_vec=empty_vec,
                edge_len=empty_len,
                edge_unit=empty_vec,
                edge_rbf=empty_rbf,
                edge_sw=torch.empty(0, 1, dtype=dtype, device=device),
                D_list=[torch.empty(0, 1, 1, dtype=dtype, device=device)],
                Dt_list=[torch.empty(0, 1, 1, dtype=dtype, device=device)],
                sw=sw,
                inv_sqrt_deg=inv_sqrt_deg,
            )

        # === Step 4. Build flat edge indices and map to (src, dst) nodes ===
        # edge_idx indexes the flattened (nf, nloc, nnei) axis in row-major order.
        # Convert it back to:
        #   f_idx   in [0, nf)
        #   loc_idx in [0, nloc)
        #   neighbor index from nlist (extended axis)
        nlist_flat = nlist.reshape(-1)  # (nf*nloc*nnei,)
        edge_idx_flat = edge_idx.to(dtype=torch.long)
        valid_f_idx = edge_idx_flat // (nloc * nnei)
        rem = edge_idx_flat % (nloc * nnei)
        valid_loc_idx = rem // nnei
        valid_neighbor = nlist_flat[edge_idx_flat]
        if mapping is None:
            # Neighbor indices are already local indices.
            src_local = valid_neighbor
        else:
            # Map extended index -> local index for each frame.
            mapping_flat = mapping.reshape(-1)
            src_local = mapping_flat[valid_f_idx * nall + valid_neighbor]

        # dst is the center atom (per-frame local index -> global node index)
        dst = valid_f_idx * nloc + valid_loc_idx
        src_ok = (src_local >= 0) & (src_local < nloc)
        if not bool(src_ok.all()):
            # Drop edges that map outside the local range (e.g. broken mapping or ghost-only neighbor).
            edge_idx = edge_idx[src_ok]  # (n_edges,)
            valid_f_idx = valid_f_idx[src_ok]
            valid_loc_idx = valid_loc_idx[src_ok]
            dst = dst[src_ok]
            src_local = src_local[src_ok]
            edge_sw = edge_sw[src_ok]

        # src is the neighbor atom (per-frame local index -> global node index)
        src = valid_f_idx * nloc + src_local

        # === Step 5. Gather per-edge geometry ===
        # edge_vec points from center -> neighbor: r_ij = r_j - r_i (in Å).
        diff_flat = diff.reshape(-1, 3)
        length_flat = length.reshape(-1, 1)
        edge_vec = diff_flat[edge_idx]  # (n_edges, 3)
        edge_len = length_flat[edge_idx]  # (n_edges, 1)

        # Unit edge direction. safe_norm avoids division-by-zero NaNs.
        edge_unit = edge_vec / safe_norm(edge_vec)

        # === Step 6. Radial basis (envelope already baked in) ===
        edge_rbf = self.radial_basis(edge_len)  # (n_edges, n_rbf)

        # === Step 7. Wigner-D blocks ===
        rot_mat = init_edge_rot_mat(edge_vec)
        D_list, Dt_list = self.wigner_calc(rot_mat)

        # === Step 8. Neighbor normalization (destination degree) ===
        # Compute inverse sqrt degree for graph-style message normalization.
        deg = torch.bincount(dst, minlength=n_nodes).to(
            dtype=edge_vec.dtype, device=edge_vec.device
        )
        inv_sqrt_deg = torch.rsqrt(deg.clamp(min=1)).view(n_nodes, 1, 1)

        return EdgeFeatureCache(
            src=src,
            dst=dst,
            edge_vec=edge_vec,
            edge_len=edge_len,
            edge_unit=edge_unit,
            edge_rbf=edge_rbf,
            edge_sw=edge_sw,
            D_list=D_list,
            Dt_list=Dt_list,
            sw=sw,
            inv_sqrt_deg=inv_sqrt_deg,
        )

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, None, None, None, torch.Tensor]:
        """
        Compute the descriptor.

        Parameters
        ----------
        extended_coord
            Extended coordinates of atoms with shape (nf, nall*3).
        extended_atype
            Extended atom types with shape (nf, nall).
        nlist
            Neighbor list with shape (nf, nloc, nnei).
        mapping
            Extended-to-local mapping with shape (nf, nall), or None.
        comm_dict
            Communication dictionary for parallel inference (unused).

        Returns
        -------
        descriptor
            Descriptor with shape (nf, nloc, channels). Only l=0 is returned.
        rot_mat
            None (not used in this descriptor).
        g2
            None (not used).
        h2
            None (not used).
        sw
            Smooth weight with shape (nf, nloc, nnei, 1).
        """
        del comm_dict  # Parallel mode not implemented yet

        # === Step 1. Cast inputs to internal precision ===
        extended_coord = extended_coord.to(dtype=self.dtype)
        nf, nloc, nnei = nlist.shape
        nall = extended_coord.view(nf, -1).shape[1] // 3
        n_nodes = int(nf * nloc)

        # === Step 2. Excluded type pairs ===
        if self.exclude_types:
            # (nf, nloc, nnei), True means keep.
            pair_keep_mask = self.emask(nlist, extended_atype).to(dtype=torch.bool)
        else:
            pair_keep_mask = torch.ones_like(
                nlist, dtype=torch.bool, device=self.device
            )

        # === Step 3. Build edge cache once (geometry + RBF + Wigner-D) ===
        edge_cache = self._build_edge_cache(
            extended_coord=extended_coord,
            extended_atype=extended_atype,
            nlist=nlist,
            mapping=mapping,
            pair_keep_mask=pair_keep_mask,
        )

        # === Step 4. Initial embedding: l=0 from type embedding ===
        atype_loc = extended_atype[:, :nloc]
        x0 = self.type_embedding(atype_loc).reshape(n_nodes, self.channels)
        x0 = self._apply_zbl_gating_hook(x0, edge_cache)

        lmax_0 = self.l_schedule[0]
        ebed_dim_0 = _so3_dim_of_lmax(lmax_0)  # (lmax+1)^2
        x = x0.new_zeros(n_nodes, ebed_dim_0, self.channels)
        x[:, 0, :] = x0

        # === Step 5. Geometric Initial Embedding ===
        if self.gie is not None and edge_cache.num_edges > 0:
            x = x + self.gie(n_nodes=n_nodes, edge_cache=edge_cache)

        # === Step 6. Blocks with pyramid l-schedule slicing ===
        for blk_lmax, block in zip(self.l_schedule, self.blocks):
            x = x[:, : _so3_dim_of_lmax(blk_lmax), :]
            x = block(x, edge_cache)

        # === Step 7. Output l=0 only ===
        descriptor = x[:, 0, :].view(nf, nloc, self.channels)
        return (
            descriptor.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            None,
            None,
            None,
            edge_cache.sw.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "Descriptor",
            "type": "se_zm_net",
            "@version": 1,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
            "ntypes": self.ntypes,
            "type_map": self.type_map,
            "l_schedule": self.l_schedule,
            "channels": self.channels,
            "n_radial": self.n_radial,
            "radial_mlp": self.radial_mlp,
            "so2_layers": self.so2_layers,
            "ffn_neuron": self.ffn_neuron,
            "wigner_parallel": self.wigner_parallel,
            "set_davg_zero": self.set_davg_zero,
            "activation_function": self.activation_function,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "trainable": self.trainable,
            "seed": self.seed,
            "type_embedding": self.type_embedding.embedding.serialize(),
            "radial_basis": self.radial_basis.serialize(),
            "blocks": [
                {
                    "lmax": blk.lmax,
                    "conv": blk.conv.serialize() if blk.conv is not None else None,
                    "ffn": blk.ffn.serialize(),
                    "gate": blk.gating.gate.serialize(),
                    "pre_linear": blk.pre_linear.serialize(),
                }
                for blk in self.blocks
            ],
            "env_mat": DPEnvMat(
                self.rcut, self.rcut_smth, self.env_protection
            ).serialize(),
            "@variables": {
                "davg": to_numpy_array(self.mean),
                "dstd": to_numpy_array(self.stddev),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> DescrptSeZMNet:
        data = data.copy()
        data.pop("@class")
        data.pop("type")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported se_zm_net version: {version}")

        stats = data.pop("@variables")
        data.pop("env_mat")
        type_embedding = data.pop("type_embedding")
        radial_basis_data = data.pop("radial_basis")
        blocks_data = data.pop("blocks")

        obj = cls(**data)
        obj.type_embedding.embedding = TypeEmbedNetConsistent.deserialize(
            type_embedding
        )

        obj.radial_basis = RadialBasis.deserialize(radial_basis_data)

        for blk, blk_data in zip(obj.blocks, blocks_data):
            blk.ffn = EquivariantFFN.deserialize(blk_data["ffn"])
            blk.gating.gate = EmbeddingNet.deserialize(blk_data["gate"])
            if blk_data["conv"] is not None:
                blk.conv = SO2Convolution.deserialize(blk_data["conv"])
            else:
                blk.conv = None
            blk.pre_linear = PerDegreeLinear.deserialize(blk_data["pre_linear"])

        obj.mean = torch.as_tensor(
            stats["davg"], dtype=obj.mean.dtype, device=env.DEVICE
        )
        obj.stddev = torch.as_tensor(
            stats["dstd"], dtype=obj.stddev.dtype, device=env.DEVICE
        )
        return obj

    @classmethod
    def update_sel(
        cls, train_data: DeepmdDataSystem, type_map: list[str] | None, local_jdata: dict
    ) -> tuple[dict, float | None]:
        local_jdata_cpy = local_jdata.copy()
        min_nbor_dist, local_jdata_cpy["sel"] = UpdateSel().update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["rcut"],
            local_jdata_cpy["sel"],
            False,
        )
        return local_jdata_cpy, min_nbor_dist
