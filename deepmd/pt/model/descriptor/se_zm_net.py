# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SeZM-Net: Smooth equivariant ZBL Message-passing Network descriptor for DeePMD-kit
(PyTorch backend).

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
from contextlib import (
    contextmanager,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
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
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .se_zm_block import (
    SeZMInteractionBlock,
    SO3Linear,
)
from .se_zm_helper import (
    EdgeFeatureCache,
    WignerDCalculator,
    edge_cache_to_dtype,
    get_promoted_dtype,
    get_so3_dim_of_lmax,
    np_safe,
    safe_norm,
    safe_numpy_to_tensor,
    so3_packed_index,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
    )

    from deepmd.utils.data_system import (
        DeepmdDataSystem,
    )
    from deepmd.utils.path import (
        DPPath,
    )


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
        Edge vectors with shape (E, 3).

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape (E, 3, 3).
    """
    # === Step 1. Normalize edge direction (local z) ===
    # z_hat is the unit edge direction (center -> neighbor).
    z_hat = edge_vec / safe_norm(edge_vec)

    # === Step 2. Construct x-axis by Gram-Schmidt against a reference ===
    # Choose a reference axis that is not nearly parallel to z_hat to avoid
    # catastrophic cancellation in the Gram-Schmidt projection.
    candi_1 = torch.tensor(
        [1.0, 0.0, 0.0], dtype=edge_vec.dtype, device=edge_vec.device
    ).expand_as(edge_vec)
    candi_2 = torch.tensor(
        [0.0, 1.0, 0.0], dtype=edge_vec.dtype, device=edge_vec.device
    ).expand_as(edge_vec)
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
    rot_mat = torch.stack([x_hat, y_hat, z_hat], dim=-2)
    return rot_mat


def init_edge_rot_mat_frisvad(
    edge_vec: torch.Tensor,
    edge_len: torch.Tensor | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
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
        Edge vectors with shape (E, 3).
    edge_len
        Precomputed edge lengths with shape (E, 1). If None, recompute from edge_vec.
    eps
        Small epsilon for numerical stability.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape (E, 3, 3).
    """
    # === Step 1. Normalize edge direction (local z) ===
    # z_hat is the unit edge direction (center -> neighbor).
    if edge_len is None:
        edge_len = safe_norm(edge_vec, eps)
    else:
        edge_len = edge_len.clamp(min=eps)
    z_hat = edge_vec / edge_len
    nx = z_hat[..., 0:1]
    ny = z_hat[..., 1:2]
    nz = z_hat[..., 2:3]

    # === Step 2. Frisvad closed-form orthonormal basis (non-singular) ===
    # The closed-form uses a = 1 / (1 + nz), which is singular at nz = -1.
    # Compute it with a safe denominator, then select by a singular mask.
    # Use a fixed threshold for singular detection (1e-6 is sufficient for all precisions).
    singular_threshold = 1.0e-6
    singular = nz < (-1.0 + singular_threshold)

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
    ref = torch.tensor(
        [1.0, 0.0, 0.0], dtype=edge_vec.dtype, device=edge_vec.device
    ).expand_as(edge_vec)
    x_fb = torch.cross(ref, z_hat, dim=-1)
    x_fb = x_fb / safe_norm(x_fb, eps)
    y_fb = torch.cross(z_hat, x_fb, dim=-1)
    y_fb = y_fb / safe_norm(y_fb, eps)

    mask3 = singular.expand_as(edge_vec)
    x_hat = torch.where(mask3, x_fb, x_main)
    y_hat = torch.where(mask3, y_fb, y_main)

    # Normalize to protect against numerical drift (and to match your existing style).
    x_hat = x_hat / safe_norm(x_hat, eps)
    y_hat = y_hat / safe_norm(y_hat, eps)

    # === Step 4. Stack rows to form global->local rotation ===
    # Row-stacking ensures v_local = R @ v_global.
    rot_mat = torch.stack([x_hat, y_hat, z_hat], dim=-2)
    return rot_mat


class RadialMLP(nn.Module):
    """
    Radial MLP with LayerNorm and configurable activation.

    Parameters
    ----------
    mlp_layers : list[int]
        Layer sizes including input and output dimensions.
        E.g., [in_dim, hidden1, hidden2, out_dim].
    activation_function : str
        Activation function name (e.g., "silu", "tanh", "gelu").
    dtype : torch.dtype
        Floating point dtype for the linear layers.
    trainable : bool
        Whether the parameters are trainable.

    Architecture
    ------------
    Linear → LayerNorm → Activation for all hidden layers,
    with the final layer being a plain Linear (no LN, no activation).
    The first layer's bias is initialized to zero.

    Notes
    -----
    LayerNorm provides stable gradients. The first layer bias is zero-initialized
    to ensure smooth gradient flow at initialization.
    """

    def __init__(
        self,
        mlp_layers: list[int],
        *,
        activation_function: str = "silu",
        dtype: torch.dtype = torch.float32,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        if len(mlp_layers) < 2:
            raise ValueError("`mlp_layers` must have at least 2 elements")
        self.mlp_layers = list(mlp_layers)
        self.activation_function = str(activation_function)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[self.dtype]
        self.trainable = bool(trainable)

        modules: list[nn.Module] = []
        n_layers = len(mlp_layers)
        for i in range(n_layers - 1):
            linear = MLPLayer(
                mlp_layers[i],
                mlp_layers[i + 1],
                bias=True,
                activation_function=None,
                precision=self.precision,
                seed=child_seed(seed, i),
                trainable=trainable,
            )
            # First layer: zero-initialize bias for smooth gradient flow
            if i == 0 and linear.bias is not None:
                nn.init.zeros_(linear.bias)
            modules.append(linear)
            # Last layer: no LayerNorm/activation
            if i < n_layers - 2:
                modules.append(
                    nn.LayerNorm(
                        mlp_layers[i + 1], dtype=self.dtype, device=self.device
                    )
                )
                modules.append(ActivationFn(self.activation_function))

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (..., mlp_layers[0]).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (..., mlp_layers[-1]).
        """
        return self.net(x)

    def serialize(self) -> dict[str, Any]:
        """Serialize the RadialMLP to a dict."""
        state = self.net.state_dict()
        return {
            "@class": "RadialMLP",
            "@version": 1,
            "mlp_layers": self.mlp_layers.copy(),
            "activation_function": self.activation_function,
            "dtype": RESERVED_PRECISION_DICT[self.dtype],
            "trainable": self.trainable,
            "@variables": {k: np_safe(v) for k, v in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> RadialMLP:
        """Deserialize a RadialMLP from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "RadialMLP":
            raise ValueError(f"Invalid class for RadialMLP: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported RadialMLP version: {version}")
        variables = data.pop("@variables")
        data["dtype"] = PRECISION_DICT[data["dtype"]]
        obj = cls(**data)
        state = {
            k: safe_numpy_to_tensor(v, device=env.DEVICE, dtype=obj.dtype)
            for k, v in variables.items()
        }
        obj.net.load_state_dict(state)
        return obj


class C2CutoffEnvelope(torch.nn.Module):
    """
    C^2-continuous polynomial cutoff envelope function.

    As proposed in DimeNet: https://arxiv.org/abs/2003.03123

    This envelope provides a smooth transition to zero at the cutoff radius,
    ensuring continuity of the function value, first derivative, and second
    derivative.

    Notes
    -----
    The envelope function is defined for scaled distance ``x = r / rcut`` as::

        E(x) = 1 + x^p * (a + b*x + c*x^2),  for x < 1
        E(x) = 0,                            for x >= 1

    where the coefficients are chosen to satisfy::

        E(0) = 1,    E(1) = 0
        E'(1) = 0,   E''(1) = 0

    This ensures C^2 continuity at the cutoff boundary. The coefficients are::

        a = -(p + 1)(p + 2) / 2
        b = p(p + 2)
        c = -p(p + 1) / 2

    For the default exponent p=5, the coefficients are a=-21, b=35, c=-15::

        E(x) = 1 + x^5 * (-21 + 35*x - 15*x^2)
             = 1 - 21*x^5 + 35*x^6 - 15*x^7

    Parameters
    ----------
    rcut : float
        Cutoff radius in Å.
    exponent : int, optional
        Polynomial exponent (p), must be positive. Default is 5.

    Attributes
    ----------
    rcut : float
        Cutoff radius in Å.
    p : float
        Polynomial exponent.
    a : float
        Quadratic coefficient for x^p term.
    b : float
        Linear coefficient for x^(p+1) term.
    c : float
        Constant coefficient for x^(p+2) term.
    """

    def __init__(self, rcut: float, exponent: int = 5) -> None:
        super().__init__()
        assert exponent > 0
        self.rcut = float(rcut)
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, dst: torch.Tensor) -> torch.Tensor:
        """Compute the envelope value for given distances."""
        d_scaled = (dst / self.rcut).clamp(min=0.0, max=1.0)
        env_val = 1 + (d_scaled**self.p) * (
            self.a + d_scaled * (self.b + self.c * d_scaled)
        )
        return env_val * ((d_scaled < 1.0).to(dst.dtype))


class RadialBasis(nn.Module):
    """
    Spherical Bessel radial basis with C^2 cutoff envelope.

    Frequencies are trainable nn.Parameter, allowing the model
    to learn optimal radial basis spacing during training.

    Notes
    -----
    This implementation computes the spherical Bessel radial basis
    using PyTorch's sinc function for numerical stability::

        phi_n(r) = w_n * sinc(w_n * r / π)

    where ``torch.sinc(z) = sin(π*z) / (π*z)``. This is mathematically
    equivalent to the standard form ``sin(w_n * r) / r``, but sinc handles
    the r->0 limit via Taylor expansion, providing continuous gradients
    without explicit epsilon clamping.

    The ``r -> 0`` limit is finite::

        lim_{r->0} w_n * sinc(w_n * r / π) = w_n

    The initial frequencies follow a common "Bessel" spacing::

        w_n = n * π / rcut, for n = 1..n_radial (in 1/Å)

    The C^2 cutoff envelope is multiplied directly into the output to ensure
    strict smoothness at ``rcut``.

    Parameters
    ----------
    rcut : float
        Cutoff radius in Å.
    n_radial : int
        Number of basis functions.
    dtype : torch.dtype
        Floating-point dtype for the radial basis frequencies and outputs.
    exponent : int, optional
        Exponent for the C^2 cutoff envelope polynomial. Default is 7.
    """

    def __init__(
        self,
        rcut: float,
        n_radial: int,
        *,
        dtype: torch.dtype,
        exponent: int = 7,
    ) -> None:
        super().__init__()
        self.rcut = float(rcut)
        self.n_radial = int(n_radial)
        if self.n_radial <= 0:
            raise ValueError("`n_radial` must be positive")
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[self.dtype]
        self.exponent = int(exponent)

        # === Frequencies: n*π/rcut, n=1..n_radial ===
        # Shape: (1, n_radial), stored as trainable nn.Parameter
        freqs = torch.arange(
            1,
            self.n_radial + 1,
            device=self.device,
            dtype=self.dtype,
        ) * (math.pi / self.rcut)
        self.freqs = nn.Parameter(freqs.view(1, -1), requires_grad=True)

        self.envelope = C2CutoffEnvelope(rcut=self.rcut, exponent=self.exponent)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Compute radial basis functions.

        Parameters
        ----------
        r : torch.Tensor
            Pair distances with shape (N, 1) in Å, where N is the number of pairs.

        Returns
        -------
        torch.Tensor
            Radial basis multiplied by C^2 cutoff envelope with shape (N, n_rbf).
            The output is smoothly truncated to zero at r = rcut.
        """
        # === Step 1. Bessel Basis via Sinc ===
        # phi_n(r) = w_n * sinc(w_n * r / π)
        # Shape: (N, 1) * (1, n_radial) -> (N, n_radial)
        x = r * self.freqs
        raw = self.freqs * torch.sinc(x / math.pi)

        # === Step 2. Apply C^2 envelope for smooth cutoff ===
        envelope = self.envelope(r)
        return raw * envelope

    def serialize(self) -> dict[str, Any]:
        """Serialize RadialBasis including trainable frequencies."""
        return {
            "@class": "RadialBasis",
            "@version": 1,  # keep 1 at devel stage
            "rcut": self.rcut,
            "n_radial": self.n_radial,
            "exponent": self.exponent,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "freqs": np_safe(self.freqs),
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
            exponent=int(data.get("exponent", 7)),
            dtype=dtype,
        )
        obj.freqs.data.copy_(
            safe_numpy_to_tensor(data["freqs"], device=obj.device, dtype=obj.dtype)
        )
        return obj


class GeometricInitialEmbedding(nn.Module):
    """
    Geometric initial embedding that adds zonal (m=0) rotated features.

    This module rotates pre-computed radial features for each degree l >= 1 using the
    zonal (m=0) column of the cached inverse Wigner-D blocks (local->global).
    The l=0 component is not computed here since it comes from type embedding.

    Parameters
    ----------
    lmax
        Maximum degree.
    channels
        Number of channels per (l, m) coefficient.
    dtype
        Parameter dtype.
    """

    def __init__(
        self,
        *,
        lmax: int,
        channels: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.ebed_dim = get_so3_dim_of_lmax(self.lmax)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

    def forward(
        self,
        *,
        n_nodes: int,
        edge_cache: EdgeFeatureCache,
        radial_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        n_nodes
            Number of nodes (nf*nloc).
        edge_cache
            Per-edge cache containing geometry, weights, and Wigner-D blocks.
        radial_feat
            Per-edge radial features with shape (E, lmax, C) for l=1..lmax.

        Returns
        -------
        torch.Tensor
            Initial features to add with shape (N, D, C). l=0 is guaranteed zero.
        """
        num_edges = edge_cache.src.size(0)
        if num_edges == 0:
            return torch.zeros(
                n_nodes,
                self.ebed_dim,
                self.channels,
                device=edge_cache.edge_vec.device,
                dtype=edge_cache.edge_vec.dtype,
            )

        device = edge_cache.edge_vec.device
        dtype = edge_cache.edge_vec.dtype

        out = torch.zeros(
            n_nodes, self.ebed_dim, self.channels, device=device, dtype=dtype
        )

        for l in range(1, self.lmax + 1):
            start, end = l * l, (l + 1) * (l + 1)
            # Extract m=0 column from Wigner-D transpose (local->global rotation).
            # Global column index for m=0 in l-block: l^2 + l.
            Dt_full = edge_cache.Dt_full
            assert Dt_full is not None
            d_col = Dt_full[:, start:end, so3_packed_index(l, 0)]  # (E, 2l+1)
            msg_global = d_col.unsqueeze(-1) * radial_feat[:, l - 1, :].unsqueeze(
                1
            )  # (E, 2l+1, C)

            out[:, start:end, :].index_add_(0, edge_cache.dst, msg_global)

        # Apply neighbor normalization (graph-style degree normalization).
        out = out * edge_cache.inv_sqrt_deg

        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "GeometricInitialEmbedding",
            "@version": 1,
            "lmax": self.lmax,
            "channels": self.channels,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> GeometricInitialEmbedding:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "GeometricInitialEmbedding":
            raise ValueError(f"Invalid class for GeometricInitialEmbedding: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(
                f"Unsupported GeometricInitialEmbedding version: {version}"
            )
        precision = data.pop("precision")
        data["dtype"] = PRECISION_DICT[precision]
        return cls(**data)


class EnvironmentInitialEmbedding(nn.Module):
    """
    Environment matrix initial embedding for l=0 features.

    Computes an initial embedding based on the 4D environment matrix::

        [s, s * rx, s * ry, s * rz]

    Combined with independent type embeddings (individual type embedding),
    providing physical inductive bias for l=0 features.

    The computation follows the environment matrix approach where::

        1. Build `r_tilde = [s, s*r_hat]` where `s = edge_env / r` and `r_hat = edge_vec / r`
        2. G network: `g = G(rbf_proj(edge_rbf), type_src, type_dst)` produces per-edge features
           - Uses independent `env_type_embed` instead of projecting from main type embedding
           - Uses `rbf_proj` to project edge_rbf to `rbf_out_dim`
        3. env_agg: aggregate outer product `r_tilde ⊗ g` by destination node
        4. D matrix: `D = env_agg^T @ env_agg[:, :, :axis_dim]`
        5. Output: residual-scaled projection of flattened D matrix

    Parameters
    ----------
    ntypes : int
        Number of atom types.
    n_radial : int
        Number of radial basis functions.
    channels : int
        Output channel dimension (same as type embedding channels).
    embed_dim : int
        G network output dimension (filter width).
    axis_dim : int
        D matrix axis dimension (must be < embed_dim).
    type_dim : int
        Dimension for independent type embeddings in env_seed.
    hidden_dim : int
        Hidden layer size for G network.
    norm : str
        Normalization mode: "deg" (divide by degree) or "sqrt_deg" (multiply by 1/sqrt(deg)).
    activation_function : str
        Activation function for G network hidden layer.
    eps : float
        Small epsilon for numerical stability.
    dtype : torch.dtype
        Parameter dtype.
    trainable : bool
        Whether parameters are trainable.
    seed : int | list[int] | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        *,
        ntypes: int,
        n_radial: int,
        channels: int,
        embed_dim: int = 64,
        axis_dim: int = 8,
        type_dim: int = 16,
        hidden_dim: int = 64,
        env_seed_max: float = 1.0,
        norm: str = "sqrt_deg",
        activation_function: str = "silu",
        eps: float = 1e-7,
        dtype: torch.dtype,
        trainable: bool,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()

        # === Validate parameters ===
        if axis_dim >= embed_dim:
            raise ValueError(
                f"`axis_dim` ({axis_dim}) must be < `embed_dim` ({embed_dim})"
            )
        if norm not in ("deg", "sqrt_deg"):
            raise ValueError(f"`norm` must be 'deg' or 'sqrt_deg', got '{norm}'")

        self.ntypes = int(ntypes)
        self.n_radial = int(n_radial)
        self.channels = int(channels)
        self.embed_dim = int(embed_dim)
        self.axis_dim = int(axis_dim)
        self.type_dim = int(type_dim)
        self.hidden_dim = int(hidden_dim)
        self.env_seed_max = float(env_seed_max)
        self.norm = str(norm)
        self.activation_function = str(activation_function)
        self.eps = float(eps)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]

        # === RBF projection: n_radial -> rbf_out_dim (two-layer MLP) ===
        # rbf_out_dim = max(32, channels - 2*type_dim) to align G-network input ~ channels
        # First layer: n_radial -> rbf_out_dim with activation
        # Second layer: rbf_out_dim -> rbf_out_dim linear
        self.rbf_out_dim = max(32, self.channels - 2 * self.type_dim)
        seed_rbf_proj = child_seed(seed, 0)
        self.rbf_proj_layer1 = MLPLayer(
            self.n_radial,
            self.rbf_out_dim,
            bias=True,
            activation_function=self.activation_function,
            precision=self.precision,
            seed=child_seed(seed_rbf_proj, 0),
        )
        self.rbf_proj_layer2 = MLPLayer(
            self.rbf_out_dim,
            self.rbf_out_dim,
            bias=True,
            activation_function=None,
            precision=self.precision,
            seed=child_seed(seed_rbf_proj, 1),
        )

        # === Independent type embedding: ntypes -> type_dim ===
        # Individual type embedding
        seed_type_embed = child_seed(seed, 1)
        self.env_type_embed = TypeEmbedNet(
            type_nums=self.ntypes,
            embed_dim=self.type_dim,
            precision=self.precision,
            seed=seed_type_embed,
            trainable=trainable,
        )

        # === G network: (rbf_out_dim + 2*type_dim) -> hidden_dim -> embed_dim ===
        seed_g_net = child_seed(seed, 2)
        g_in_dim = self.rbf_out_dim + 2 * self.type_dim
        self.g_layer1 = MLPLayer(
            g_in_dim,
            self.hidden_dim,
            bias=True,
            activation_function=self.activation_function,
            precision=self.precision,
            seed=child_seed(seed_g_net, 0),
        )
        self.g_layer2 = MLPLayer(
            self.hidden_dim,
            self.embed_dim,
            bias=True,
            activation_function=None,
            precision=self.precision,
            seed=child_seed(seed_g_net, 1),
        )

        # === Output projection: embed_dim * axis_dim -> channels ===
        seed_out = child_seed(seed, 3)
        self.output_proj = MLPLayer(
            self.embed_dim * self.axis_dim,
            self.channels,
            bias=True,
            activation_function=None,
            precision=self.precision,
            seed=seed_out,
        )

        # === Learnable gamma for bounded sigmoid scale ===
        # scale = env_seed_max * (2*sigmoid(gamma) - 1) in [-env_seed_max, env_seed_max]
        # Initialized to 0.0 so sigmoid(0) = 0.5 → scale = 0 (safe residual start)
        self.gamma = nn.Parameter(
            torch.tensor(0.0, dtype=self.dtype, device=self.device)
        )

        for p in self.parameters():
            p.requires_grad = trainable

    def forward(
        self,
        *,
        edge_cache: EdgeFeatureCache,
        atype_flat: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """
        Compute environment scalar env_seed embedding.

        Parameters
        ----------
        edge_cache : EdgeFeatureCache
            Edge cache containing src, dst, edge_vec, edge_rbf, edge_env.
        atype_flat : torch.Tensor
            Flattened atom types with shape (N,), where N = nf * nloc.
        n_nodes : int
            Number of nodes (N = nf * nloc).

        Returns
        -------
        torch.Tensor
            Scalar env_seed with shape (N, channels).
        """
        num_edges = edge_cache.src.numel()
        if num_edges == 0:
            return torch.zeros(
                n_nodes, self.channels, dtype=self.dtype, device=self.device
            )

        src, dst = edge_cache.src, edge_cache.dst
        edge_vec = edge_cache.edge_vec  # (E, 3)
        edge_rbf = edge_cache.edge_rbf  # (E, n_radial)
        edge_env = edge_cache.edge_env  # (E, 1)

        # === Step 1. Construct r_tilde = [s, s*r_hat] ===
        # s = edge_env * (1/r), r_hat = edge_vec / r
        r_sq = (edge_vec * edge_vec).sum(dim=-1, keepdim=True)  # (E, 1)
        r = torch.sqrt(r_sq.clamp(min=self.eps * self.eps))  # (E, 1)
        inv_r = 1.0 / r  # (E, 1)
        s = edge_env * inv_r  # (E, 1)
        r_hat = edge_vec * inv_r  # (E, 3)
        r_tilde = torch.cat([s, s * r_hat], dim=-1)  # (E, 4)

        # === Step 2. Compute G network input and output ===
        # Use independent type embeddings (decoupled from main type embedding)
        atype_src = atype_flat.index_select(0, src)  # (E,)
        atype_dst = atype_flat.index_select(0, dst)  # (E,)
        type_src = self.env_type_embed(atype_src)  # (E, type_dim)
        type_dst = self.env_type_embed(atype_dst)  # (E, type_dim)

        # Project edge_rbf to rbf_out_dim (two-layer MLP)
        rbf_proj: torch.Tensor = self.rbf_proj_layer2(
            self.rbf_proj_layer1(edge_rbf)
        )  # (E, rbf_out_dim)

        # G network input: concat projected RBF and type embeddings
        g_input = torch.cat([rbf_proj, type_src, type_dst], dim=-1)  # (E, g_in_dim)
        g = self.g_layer2(self.g_layer1(g_input))  # (E, embed_dim)

        # === Step 3. Aggregate outer product by destination node ===
        # outer = r_tilde[:, :, None] * g[:, None, :]  # (E, 4, embed_dim)
        outer: torch.Tensor = r_tilde.unsqueeze(-1) * g.unsqueeze(
            1
        )  # (E, 4, embed_dim)
        outer_flat = outer.reshape(num_edges, 4 * self.embed_dim)  # (E, 4*embed_dim)

        env_agg = outer_flat.new_zeros(n_nodes, 4 * self.embed_dim)
        env_agg.index_add_(0, dst, outer_flat)  # (N, 4*embed_dim)
        env_agg = env_agg.view(n_nodes, 4, self.embed_dim)  # (N, 4, embed_dim)

        # === Step 4. Normalization by actual neighbor count ===
        node_edges = edge_vec.new_zeros(n_nodes)
        node_edges.index_add_(0, dst, node_edges.new_ones(num_edges))
        node_edges_clamped = node_edges.clamp(min=1.0)

        if self.norm == "deg":
            env_agg = env_agg / node_edges_clamped.view(-1, 1, 1)
        else:  # sqrt_deg
            env_agg = env_agg * torch.rsqrt(node_edges_clamped).view(-1, 1, 1)

        # === Step 5. D matrix construction: D = env_agg^T @ env_agg[:,:,:axis_dim] ===
        env_agg_t = env_agg.permute(0, 2, 1)  # (N, embed_dim, 4)
        env_agg_axis = env_agg[:, :, : self.axis_dim]  # (N, 4, axis_dim)
        D = torch.bmm(env_agg_t, env_agg_axis)  # (N, embed_dim, axis_dim)

        # === Step 6. Output projection with bounded sigmoid scale ===
        D_flat = D.reshape(n_nodes, self.embed_dim * self.axis_dim)  # (N, embed*axis)
        env_seed = self.output_proj(D_flat)  # (N, channels)
        # scale = env_seed_max * (2*sigmoid(gamma) - 1) in [-env_seed_max, env_seed_max]
        scale = self.env_seed_max * (2.0 * torch.sigmoid(self.gamma) - 1.0)
        env_seed = scale * env_seed

        return env_seed

    def serialize(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "@class": "EnvironmentInitialEmbedding",
            "@version": 1,
            "ntypes": self.ntypes,
            "n_radial": self.n_radial,
            "channels": self.channels,
            "embed_dim": self.embed_dim,
            "axis_dim": self.axis_dim,
            "type_dim": self.type_dim,
            "hidden_dim": self.hidden_dim,
            "env_seed_max": self.env_seed_max,
            "norm": self.norm,
            "activation_function": self.activation_function,
            "eps": self.eps,
            "precision": self.precision,
            "rbf_proj_layer1": self.rbf_proj_layer1.serialize(),
            "rbf_proj_layer2": self.rbf_proj_layer2.serialize(),
            "env_type_embed": self.env_type_embed.embedding.serialize(),
            "g_layer1": self.g_layer1.serialize(),
            "g_layer2": self.g_layer2.serialize(),
            "output_proj": self.output_proj.serialize(),
            "gamma": np_safe(self.gamma),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EnvironmentInitialEmbedding:
        """Deserialize from dictionary."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "EnvironmentInitialEmbedding":
            raise ValueError(f"Invalid class: {data_cls}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")

        rbf_proj_layer1_data = data.pop("rbf_proj_layer1")
        rbf_proj_layer2_data = data.pop("rbf_proj_layer2")
        env_type_embed_data = data.pop("env_type_embed")
        g_layer1_data = data.pop("g_layer1")
        g_layer2_data = data.pop("g_layer2")
        output_proj_data = data.pop("output_proj")
        gamma_data = data.pop("gamma")

        precision = data.pop("precision")
        data["dtype"] = PRECISION_DICT[precision]
        data["trainable"] = True
        data["seed"] = None

        obj = cls(**data)
        obj.rbf_proj_layer1 = MLPLayer.deserialize(rbf_proj_layer1_data)
        obj.rbf_proj_layer2 = MLPLayer.deserialize(rbf_proj_layer2_data)
        obj.env_type_embed.embedding = TypeEmbedNetConsistent.deserialize(
            env_type_embed_data
        )
        obj.g_layer1 = MLPLayer.deserialize(g_layer1_data)
        obj.g_layer2 = MLPLayer.deserialize(g_layer2_data)
        obj.output_proj = MLPLayer.deserialize(output_proj_data)
        obj.gamma.data.copy_(
            safe_numpy_to_tensor(gamma_data, device=obj.device, dtype=obj.dtype)
        )
        return obj


@BaseDescriptor.register("se_zm_net")
@BaseDescriptor.register("se_zm")
class DescrptSeZMNet(BaseDescriptor, nn.Module):
    """
    SeZM-Net: Smooth equivariant ZBL Message-passing Network descriptor for DeePMD-kit.

    Parameters
    ----------
    rcut
        Cutoff radius in Å.
    sel
        Maximum number of neighbors per type within `rcut`.
        - int: broadcast to all types, e.g. sel=100 with ntypes=2 → [100, 100]
        - list[int]: sel[i] is the maximum number of type i atoms within `rcut`
    ntypes
        Number of element types.
    lmax
        Maximum degree, only used when `l_schedule` is None.
    n_blocks
        Number of blocks (only used when `l_schedule` is None).
    l_schedule
        Pyramid schedule of lmax per block, e.g. [3, 3, 2]. Must be non-increasing.
        If set, lmax and n_blocks will be ignored.
    mmax
        Maximum SO(2) order (|m|), only used when `m_schedule` is None.
        If None, defaults to the per-block `lmax` (i.e. `m_schedule = l_schedule`).
    m_schedule
        Schedule of mmax per block, e.g. [2, 2, 1, 0]. Must satisfy
        `m_schedule[i] <= l_schedule[i]` for every block. A non-increasing schedule is
        recommended but not required. If set, `mmax` will be ignored.
    channels
        Channels per (l,m) coefficient, i.e. feature dimension per degree.
    n_radial
        Number of radial basis functions.
    radial_mlp
        Hidden layer sizes for radial networks. An output layer of size
        `(l_schedule[0]+1)*channels` will be automatically appended.
    use_env_seed
        If True, add environment matrix initial embedding to l=0 features using
        4D `[s, s*r_hat]` representation. Provides physical inductive bias from
        local atomic environments.
    env_seed_embed_dim
        Output dimension of the G network in environment initial embedding.
        Other dimensions are derived from this value:
        `axis_dim=min(8, max(4, embed_dim//2))`,
        `type_dim=min(16, max(8, embed_dim//2))`,
        `hidden_dim=min(64, max(32, 2*embed_dim))`.
    env_seed_norm
        Normalization mode for env_agg aggregation: `"deg"` (1/degree) or
        `"sqrt_deg"` (1/sqrt(degree)).
    env_seed_max
        Maximum scale magnitude for env_seed injection. The scale is bounded in
        `[-env_seed_max, env_seed_max]` via sigmoid parameterization.
    so2_norm
        If True, apply intermediate ReducedSeparableRMSNorm between SO(2) mixing layers.
        When False (default), no normalization is applied between layers.
    so2_layers
        Number of SO(2) mixing layers per block.
    ffn_neurons
        Hidden sizes for the equivariant FFN in each block.
    n_atten_head
        Number of gated attention heads when aggregating messages in SO(2) convolution.
        0 applies a plain envelope-weighted scatter-sum; >0 uses head-wise gates (channels must be divisible by `n_atten_head`).
    activation_function
        Activation function used by deepmd EmbeddingNet.
    precision
        Precision for neural network parameters and computations. Geometry computations
        (edge distances, Wigner-D matrices, rotations, GIE) always run in fp32+ to
        provide accurate geometric information for better convergence. Only the
        interaction blocks use this precision.
    use_amp
        If True, use automatic mixed precision (AMP) with bfloat16 on CUDA.
        This does not provide accelerations under fp32 precision but will decrease
        the memory usage, while persevering model accuracy.
    exclude_types
        List of excluded type pairs.
    env_protection
        Small epsilon for numerical stability in division and normalization.
    trainable
        Whether parameters are trainable.
    seed
        Random seed(s).
    ntypes
        Number of element types.
    type_map
        Type names.

    Notes
    -----
    SeZM-Net does not use the traditional environment matrix (r, a_x, a_y, a_z).
    Instead, it uses radial basis functions and spherical harmonics directly.
    The mean/stddev statistics are kept for interface compatibility but are not
    actively used in the forward pass.
    """

    _ENV_DIM: int = 1  # Use se_r style (radial only) for EnvMatStatSe compatibility

    def __init__(
        self,
        rcut: float,
        sel: list[int] | int,
        ntypes: int,
        lmax: int = 2,
        n_blocks: int = 2,
        l_schedule: list[int] | None = None,
        mmax: int | None = None,
        m_schedule: list[int] | None = None,
        channels: int = 64,
        n_radial: int = 10,
        radial_mlp: list[int] | None = None,
        use_env_seed: bool = False,
        env_seed_max: float = 1.0,
        env_seed_embed_dim: int = 64,
        env_seed_norm: str = "sqrt_deg",
        so2_norm: bool = False,
        so2_layers: int = 2,
        ffn_neurons: int = 128,
        n_atten_head: int = 0,
        activation_function: str = "silu",
        precision: str = "float32",
        use_amp: bool = False,
        exclude_types: list[tuple[int, int]] | None = None,
        env_protection: float = 1e-7,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        type_map: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()

        self.rcut = float(rcut)

        if isinstance(sel, int):
            sel = [sel]
        self.ntypes = int(ntypes)
        self.sel = [int(x) for x in sel]
        self.type_map = type_map
        self.nnei = int(sum(self.sel))
        self.ndescrpt = int(self.nnei * self._ENV_DIM)

        self.channels = int(channels)
        self.n_radial = int(n_radial)
        if radial_mlp is None:
            radial_mlp = [64]
        self.radial_mlp = list(radial_mlp)
        self.so2_norm = bool(so2_norm)
        self.so2_layers = int(so2_layers)
        self.ffn_neurons = int(ffn_neurons)
        self.n_atten_head = int(n_atten_head)
        self.activation_function = str(activation_function)
        self.precision = str(precision)
        self.dtype = PRECISION_DICT[self.precision]
        self.device = env.DEVICE
        self.compute_dtype = get_promoted_dtype(self.dtype)
        self.eps = float(env_protection)
        self.use_amp = bool(use_amp)
        self.trainable = bool(trainable)
        self.seed = seed

        # === Env seed parameters ===
        self.use_env_seed = bool(use_env_seed)
        self.env_seed_embed_dim = int(env_seed_embed_dim)
        # Derived: axis_dim in [4, 8], scales with embed_dim // 2
        self.env_seed_axis_dim = min(8, max(4, self.env_seed_embed_dim // 2))
        # Derived: type_dim in [8, 16], larger for complex type systems
        self.env_seed_type_dim = min(16, max(8, self.env_seed_embed_dim // 2))
        self.env_seed_hidden_dim = min(64, max(32, 2 * self.env_seed_embed_dim))
        self.env_seed_max = float(env_seed_max)
        self.env_seed_norm = str(env_seed_norm)

        # === Step 0. Split deterministic seeds at the descriptor top-level ===
        seed_type_embedding = child_seed(self.seed, 0)
        seed_blocks = child_seed(self.seed, 1)
        seed_out = child_seed(self.seed, 2)
        seed_radial_embedding = child_seed(self.seed, 3)
        seed_env_seed = child_seed(self.seed, 4)

        # === Step 1. L/M schedules ===
        self._init_lm_schedules(lmax, n_blocks, l_schedule, mmax, m_schedule)
        self.ebed_dims = [get_so3_dim_of_lmax(l) for l in self.l_schedule]
        self.rad_sizes_per_block = [l + 1 for l in self.l_schedule]

        # === Step 2. Statistics buffers (interface compatibility) ===
        _shape = (self.ntypes, self.nnei, self._ENV_DIM)
        mean = torch.zeros(_shape, dtype=self.dtype, device=self.device)
        stddev = torch.ones(_shape, dtype=self.dtype, device=self.device)
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.stats: dict[str, Any] | None = None

        # === Step 3. Excluded type pairs ===
        self.reinit_exclude(exclude_types)

        # === Step 4. Type embedding ===
        type_embedding_precision = RESERVED_PRECISION_DICT[self.compute_dtype]
        self.type_embedding = TypeEmbedNet(
            type_nums=self.ntypes,
            embed_dim=self.channels,
            precision=type_embedding_precision,  # force fp32+
            seed=seed_type_embedding,
            type_map=type_map,
            trainable=self.trainable,
        )

        # === Step 5. Env scalar seed embedding (optional) ===
        if self.use_env_seed:
            self.env_seed_embedding: EnvironmentInitialEmbedding | None = (
                EnvironmentInitialEmbedding(
                    ntypes=self.ntypes,
                    n_radial=self.n_radial,
                    channels=self.channels,
                    embed_dim=self.env_seed_embed_dim,
                    axis_dim=self.env_seed_axis_dim,
                    type_dim=self.env_seed_type_dim,
                    hidden_dim=self.env_seed_hidden_dim,
                    env_seed_max=self.env_seed_max,
                    norm=self.env_seed_norm,
                    activation_function=self.activation_function,
                    eps=self.eps,
                    dtype=self.compute_dtype,  # force fp32+
                    trainable=self.trainable,
                    seed=seed_env_seed,
                )
            )
        else:
            self.env_seed_embedding = None

        self.radial_basis = RadialBasis(
            self.rcut,
            self.n_radial,
            dtype=self.compute_dtype,  # force fp32+
            exponent=7,
        )

        # === Shared radial embedding: RBF -> per-l radial features ===
        # Output dimension is (lmax+1)*channels, directly usable by GIE and SO2Conv.
        # radial_mlp specifies hidden layer sizes; input/output layers are prepended/appended.
        # Use fp32+ precision (same as RBF output) for numerical stability.
        radial_out_dim = (self.lmax + 1) * self.channels
        radial_mlp_layers = [self.n_radial, *self.radial_mlp, radial_out_dim]
        self.radial_embedding = RadialMLP(
            radial_mlp_layers,
            activation_function=self.activation_function,
            dtype=self.compute_dtype,  # force fp32+
            trainable=self.trainable,
            seed=seed_radial_embedding,
        )

        # === C^2 cutoff envelope for edge weight ===
        self.c2_envelope = C2CutoffEnvelope(rcut=self.rcut, exponent=5)

        wigner_lmax = self.l_schedule[0]
        # force fp32+
        self.wigner_calc = WignerDCalculator(
            lmax=wigner_lmax, eps=self.eps, dtype=self.compute_dtype
        )

        if self.l_schedule[0] > 0:
            self.gie = GeometricInitialEmbedding(
                lmax=self.l_schedule[0],
                channels=self.channels,
                dtype=self.compute_dtype,  # force fp32+
            )
        else:
            self.gie = None

        blocks: list[SeZMInteractionBlock] = []
        for block_idx, (l_b, m_b) in enumerate(zip(self.l_schedule, self.m_schedule)):
            blocks.append(
                SeZMInteractionBlock(
                    lmax=l_b,
                    mmax=m_b,
                    channels=self.channels,
                    so2_norm=self.so2_norm,
                    so2_layers=self.so2_layers,
                    ffn_neurons=self.ffn_neurons,
                    n_atten_head=self.n_atten_head,
                    activation_function=self.activation_function,
                    eps=self.eps,
                    dtype=self.dtype,
                    seed=child_seed(seed_blocks, block_idx),
                    trainable=self.trainable,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # === Final SO3Linear for l=0 output mixing ===
        # This mixes channels in the final scalar (l=0) descriptor output.
        # Uses promoted dtype (float32+) for better performance.
        self.so3_linear_output = SO3Linear(
            lmax=0,
            in_channels=self.channels,
            out_channels=self.channels,
            dtype=self.compute_dtype,
            trainable=self.trainable,
            seed=seed_out,
        )

        for p in self.parameters():
            p.requires_grad = self.trainable

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

        # === Step 1. Setup dimensions ===
        extended_coord = extended_coord.to(self.compute_dtype)
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

        # === Step 3. Type embedding (l=0) ===
        atype_loc = extended_atype[:, :nloc]
        type_ebed = self.type_embedding(atype_loc).reshape(n_nodes, self.channels)

        # === Step 4. Build edge cache once (geometry + RBF + Wigner-D) ===
        edge_cache, sw = self.build_edge_cache(
            type_ebed=type_ebed,
            extended_coord=extended_coord,
            extended_atype=extended_atype,
            nlist=nlist,
            mapping=mapping,
            pair_keep_mask=pair_keep_mask,
        )

        # === Step 5. Optional short-range gating hook ===
        type_ebed = self._apply_zbl_gating_hook(type_ebed, edge_cache)

        lmax_0 = self.l_schedule[0]
        ebed_dim_0 = get_so3_dim_of_lmax(lmax_0)  # (lmax+1)^2
        x = type_ebed.new_zeros(n_nodes, ebed_dim_0, self.channels)  # (N, D, C)
        x[:, 0, :] = type_ebed

        # === Step 6. Compute radial features once (fp32+) ===
        # Shape: (E, (lmax+1)*C) -> (E, lmax+1, C)
        if edge_cache.src.numel() > 0:
            radial_feat: torch.Tensor | None = self.radial_embedding(
                edge_cache.edge_rbf
            ).view(-1, self.lmax + 1, self.channels)
        else:
            radial_feat = None

        # === Step 7. Env scalar env_seed embedding (optional, fp32+) ===
        if self.env_seed_embedding is not None and edge_cache.src.numel() > 0:
            atype_flat = atype_loc.reshape(-1)  # (N,)
            env_seed = self.env_seed_embedding(
                edge_cache=edge_cache,
                atype_flat=atype_flat,
                n_nodes=n_nodes,
            )
            x[:, 0, :] = x[:, 0, :] + env_seed

        # === Step 8. Geometric Initial Embedding (fp32+) ===
        if self.gie is not None and radial_feat is not None:
            # GIE only needs l>=1, slice radial_feat[:, 1:, :]
            x = x + self.gie(
                n_nodes=n_nodes,
                edge_cache=edge_cache,
                radial_feat=radial_feat[:, 1:, :],
            )

        # === Step 9. Fuse edge type features into radial features (fp32+) ===
        if radial_feat is not None:
            radial_feat = radial_feat + edge_cache.edge_type_feat.unsqueeze(1)
            radial_feat = radial_feat.to(dtype=self.dtype)
            rad_feat_per_block = [
                radial_feat[:, :rad_len, :] for rad_len in self.rad_sizes_per_block
            ]
        else:
            rad_feat_per_block = None

        # === Step 10. Convert to self.dtype and run blocks ===
        x = x.to(dtype=self.dtype)
        edge_cache = edge_cache_to_dtype(edge_cache, self.dtype)
        if torch.jit.is_scripting():
            x = self._forward_blocks(x, edge_cache, rad_feat_per_block)
        else:
            with self._compute_mode_ctx(extended_coord.device):
                x = self._forward_blocks(x, edge_cache, rad_feat_per_block)

        # === Step 11. Final l=0 output mixing ===
        # Extract l=0 scalar features and apply final channel mixing.
        # Convert to promoted dtype for better performance.
        x_scalar = x[:, 0:1, :].to(dtype=self.compute_dtype)  # (N, 1, C)
        x_scalar = self.so3_linear_output(x_scalar)  # (N, 1, C)

        # === Step 12. Reshape to (nf, nloc, channels) and return ===
        descriptor = x_scalar.squeeze(1).view(nf, nloc, self.channels)  # (nf, nloc, C)
        return (
            descriptor.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            None,
            None,
            None,
            sw.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
        )

    def _forward_blocks(
        self,
        x: torch.Tensor,
        edge_cache: EdgeFeatureCache,
        radial_feat_per_block: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        """
        Run the interaction blocks.

        Parameters
        ----------
        x
            Initial node features with shape (N, D, C).
        edge_cache
            Per-edge cache.
        radial_feat_per_block
            List of per-block radial features already truncated to l_schedule[i]+1,
            or None if no edges.

        Returns
        -------
        torch.Tensor
            Output features with shape (N, D, C).
        """
        # Blocks with pyramid l-schedule slicing
        for i, block in enumerate(self.blocks):
            x = x[:, : self.ebed_dims[i], :]
            blk_radial = (
                None if radial_feat_per_block is None else radial_feat_per_block[i]
            )
            x = block(x, edge_cache, blk_radial)
        return x

    def _init_lm_schedules(
        self,
        lmax: int,
        n_blocks: int,
        l_schedule: list[int] | None,
        mmax: int | None,
        m_schedule: list[int] | None,
    ) -> None:
        """Parse and validate L/M schedules, setting self.l_schedule/m_schedule/lmax/mmax."""
        # === L schedule ===
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

        self.lmax = int(self.l_schedule[0])
        self.n_blocks = len(self.l_schedule)

        # === M schedule ===
        if m_schedule is None:
            if mmax is None:
                self.m_schedule = [int(l) for l in self.l_schedule]
            else:
                mmax_i = int(mmax)
                if mmax_i < 0:
                    raise ValueError("`mmax` must be non-negative")
                self.m_schedule = [min(mmax_i, int(l)) for l in self.l_schedule]
        else:
            self.m_schedule = [int(x) for x in m_schedule]
        if len(self.m_schedule) == 0:
            raise ValueError("`m_schedule` must be non-empty")
        if len(self.m_schedule) != len(self.l_schedule):
            raise ValueError("`m_schedule` must have the same length as `l_schedule`")
        if any(x < 0 for x in self.m_schedule):
            raise ValueError("`m_schedule` entries must be non-negative")
        if any(m > l for m, l in zip(self.m_schedule, self.l_schedule)):
            raise ValueError(
                "`m_schedule` entries must satisfy `m_schedule[i] <= l_schedule[i]`"
            )

        self.mmax = int(self.m_schedule[0])

    def _apply_zbl_gating_hook(
        self, type_ebed: torch.Tensor, edge_cache: EdgeFeatureCache
    ) -> torch.Tensor:
        """
        Placeholder hook for ZBL gating (not implemented).

        Parameters
        ----------
        type_ebed
            Scalar features with shape (N, C).
        edge_cache
            Per-edge cache.

        Returns
        -------
        torch.Tensor
            Updated scalar features with shape (N, C).
        """
        del edge_cache
        return type_ebed

    @contextmanager
    def _compute_mode_ctx(self, device: torch.device) -> Generator[None, None, None]:
        """
        Context manager that applies automatic mixed precision (AMP) for forward().

        Parameters
        ----------
        device
            The device of the input tensors (used to determine if CUDA ops apply).

        Notes
        -----
        - When `use_amp=True`, enables torch.autocast with bfloat16 on CUDA.
        - Only affects autocast-eligible operations (matmul, conv, etc.).
        - Does nothing on non-CUDA devices or when `use_amp=False`.

        Yields
        ------
        None
            Runs the wrapped region under the configured AMP setting.
        """
        if not self.use_amp or device.type != "cuda":
            yield
            return

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            yield

    def build_edge_cache(
        self,
        *,
        type_ebed: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None,
        pair_keep_mask: torch.Tensor,
    ) -> tuple[EdgeFeatureCache, torch.Tensor]:
        """
        Build the global edge cache from DeePMD padded neighbor list.

        This converts DeePMD's per-frame padded neighbor list into a flat list of
        valid edges used by message passing, and computes all per-edge tensors that
        are reused across blocks.

        The resulting cache contains:

        - per-edge endpoints: ``src``, ``dst`` and per-edge type features: ``edge_type_feat`` (src+dst)
        - per-edge geometry: ``edge_vec``
        - per-edge smooth weights: C^2 cutoff envelope ``edge_env``
        - per-edge radial basis: ``edge_rbf`` (envelope already baked in)
        - per-edge rotation blocks: block-diagonal Wigner-D matrices ``D_full`` and ``Dt_full``
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
        type_ebed
            Per-node type embedding with shape (N, C), where N=nf*nloc.
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

        Returns
        -------
        EdgeFeatureCache
            Per-edge cache.
        sw
            Smooth weight with shape (nf, nloc, nnei, 1).
        """
        nf, nloc, nnei = nlist.shape
        n_nodes = int(nf * nloc)
        # === Step 0. Force fp32+ for geometry ===
        geom_dtype = get_promoted_dtype(extended_coord.dtype)
        coord = extended_coord.to(dtype=geom_dtype).view(nf, -1, 3)  # (nf, nall, 3)
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

        # === Step 2. C^2 envelope weight `sw` ===
        # sw is the C^2-continuous cutoff envelope weight in [0, 1], applied per neighbor pair.
        sw = self.c2_envelope(length)  # (nf, nloc, nnei, 1)
        sw = sw * keep.unsqueeze(-1).to(dtype=sw.dtype)

        # === Step 3. Filter valid edges for message passing ===
        # An edge is valid if:
        #   - it is not padding (nlist >= 0)
        #   - the type pair is allowed (pair_keep_mask)
        #   - its length is strictly within rcut
        within = length < self.rcut
        edge_keep = (keep & within.squeeze(-1)).view(-1)
        edge_idx = torch.nonzero(edge_keep).squeeze(-1)
        edge_env = sw.reshape(-1, 1)[edge_idx]

        if edge_idx.numel() == 0:
            # No edges -> empty cache.
            device = extended_coord.device
            dtype = extended_coord.dtype
            empty_long = torch.empty(0, dtype=torch.long, device=device)
            empty_vec = torch.empty(0, 3, dtype=dtype, device=device)
            empty_rbf = torch.empty(
                0, self.radial_basis.n_radial, dtype=dtype, device=device
            )
            empty_type_feat = torch.empty(
                0, type_ebed.shape[1], dtype=dtype, device=device
            )
            inv_sqrt_deg = torch.ones(n_nodes, 1, 1, dtype=dtype, device=device)
            cache = EdgeFeatureCache(
                src=empty_long,
                dst=empty_long,
                edge_type_feat=empty_type_feat,
                edge_vec=empty_vec,
                edge_rbf=empty_rbf,
                edge_env=torch.empty(0, 1, dtype=dtype, device=device),
                inv_sqrt_deg=inv_sqrt_deg,
                D_full=None,
                Dt_full=None,
                D_to_m_cache={},
                Dt_from_m_cache={},
            )
            return cache, sw

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
        # neighbor indices from the extended axis (0..nall-1) for valid edges.
        valid_neighbor = nlist_flat[edge_idx_flat]
        if mapping is None:
            # Neighbor indices are already local indices in [0, nloc).
            src_local = valid_neighbor
        else:
            # Map extended index -> local index for each frame.
            # mapping_flat packs (nf, nall) so frame k uses offset k * nall.
            mapping_flat = mapping.reshape(-1)
            src_local = mapping_flat[valid_f_idx * nall + valid_neighbor]

        # dst is the center atom: per-frame local index -> global node index.
        dst = valid_f_idx * nloc + valid_loc_idx
        src_ok = (src_local >= 0) & (src_local < nloc)
        if not bool(src_ok.all()):
            # Drop edges that map outside the local range (e.g. broken mapping or ghost-only neighbor).
            edge_idx = edge_idx[src_ok]  # (E,)
            valid_f_idx = valid_f_idx[src_ok]
            valid_loc_idx = valid_loc_idx[src_ok]
            dst = dst[src_ok]
            src_local = src_local[src_ok]
            edge_env = edge_env[src_ok]

        # src is the neighbor atom (per-frame local index -> global node index)
        src = valid_f_idx * nloc + src_local
        edge_type_feat = type_ebed.index_select(0, src) + type_ebed.index_select(0, dst)

        # === Step 5. Gather per-edge geometry ===
        # edge_vec points from center -> neighbor: r_ij = r_j - r_i (in Å).
        diff_flat = diff.reshape(-1, 3)
        length_flat = length.reshape(-1, 1)
        edge_vec = diff_flat[edge_idx]  # (E, 3)
        edge_len = length_flat[edge_idx]  # (E, 1)

        # === Step 6. Radial basis (envelope already baked in) ===
        edge_rbf = self.radial_basis(edge_len)  # (E, n_rbf)

        # === Step 7. Wigner-D blocks ===
        rot_mat = init_edge_rot_mat_frisvad(edge_vec, edge_len=edge_len, eps=self.eps)
        D_full, Dt_full = self.wigner_calc(rot_mat)

        # === Step 8. Neighbor normalization (destination degree) ===
        # Compute inverse sqrt degree for graph-style message normalization.
        deg = torch.bincount(dst, minlength=n_nodes).to(
            dtype=edge_vec.dtype, device=edge_vec.device
        )
        inv_sqrt_deg = torch.rsqrt(deg.clamp(min=1)).view(n_nodes, 1, 1)

        cache = EdgeFeatureCache(
            src=src,  # (E,)
            dst=dst,  # (E,)
            edge_type_feat=edge_type_feat,  # (E, C)
            edge_vec=edge_vec,  # (E, 3)
            edge_rbf=edge_rbf,  # (E, n_radial)
            edge_env=edge_env,  # (E, 1)
            inv_sqrt_deg=inv_sqrt_deg,  # (N, 1, 1)
            D_full=D_full,  # (E, D, D)
            Dt_full=Dt_full,  # (E, D, D)
            D_to_m_cache={},
            Dt_from_m_cache={},
        )
        return cache, sw

    # === DeePMD descriptor interface ===
    def get_rcut(self) -> float:
        return self.rcut

    def get_rcut_smth(self) -> float:
        return self.rcut

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
        """
        If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        SeZM-Net uses TypeEmbedNet for type handling, so it does not require
        a type-distinguished neighbor list.
        """
        return True

    def has_message_passing(self) -> bool:
        return bool(len(self.blocks) > 0 and self.lmax > 0)

    def need_sorted_nlist_for_lower(self) -> bool:
        return False

    def get_env_protection(self) -> float:
        return self.eps

    @property
    def dim_out(self) -> int:
        return self.get_dim_out()

    @property
    def dim_emb(self) -> int:
        return self.get_dim_emb()

    def share_params(
        self, base_class: Any, shared_level: int, resume: bool = False
    ) -> None:
        raise NotImplementedError("share_params is not supported for se_zm_net")

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Receive the statistics (distance, max_nbor_size and env_mat_range) of the training data.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        """
        raise NotImplementedError("Compression is unsupported for se_zm_net.")

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
    ) -> None:
        raise NotImplementedError("change_type_map is not supported for se_zm_net")

    def reinit_exclude(
        self, exclude_types: list[tuple[int, int]] | None = None
    ) -> None:
        # Handle torch.jit.Attribute from deserialization
        if isinstance(exclude_types, torch.jit.Attribute):
            exclude_types = exclude_types.value
        if exclude_types is None:
            exclude_types = []
        self.exclude_types = torch.jit.Attribute(exclude_types, list[tuple[int, int]])
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    # =========================================================================
    # Statistics interface (interface compatibility only)
    # -------------------------------------------------------------------------
    # SeZM-Net uses SeparableRMSNorm inside blocks for feature normalization,
    # so mean/stddev are NOT used in forward(). These methods are kept for:
    #   1. Interface compatibility with BaseDescriptor
    #   2. Consistent serialization format (davg/dstd in checkpoint)
    # =========================================================================

    def set_stat_mean_and_stddev(
        self, mean: torch.Tensor, stddev: torch.Tensor
    ) -> None:
        """Set mean and stddev (interface compatibility, not used in forward)."""
        self.mean = mean
        self.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get mean and stddev (interface compatibility, not used in forward)."""
        return self.mean, self.stddev

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        """
        Compute statistics (interface compatibility, not used in forward).

        SeZM-Net uses learnable SeparableRMSNorm for normalization, so these
        statistics do not affect the forward pass. This is a no-op that keeps
        mean/stddev at their initialized values (zero/one) for interface consistency.
        """
        del merged, path
        # No-op: mean and stddev are already initialized to zero/one in __init__
        # and are not used in forward() due to SeparableRMSNorm.

    def serialize(self) -> dict[str, Any]:
        return {
            "@class": "Descriptor",
            "type": "se_zm_net",
            "@version": 1,
            "rcut": self.rcut,
            "sel": self.sel,
            "ntypes": self.ntypes,
            "type_map": self.type_map,
            "l_schedule": self.l_schedule,
            "m_schedule": self.m_schedule,
            "channels": self.channels,
            "n_radial": self.n_radial,
            "radial_mlp": self.radial_mlp,
            "so2_norm": self.so2_norm,
            "so2_layers": self.so2_layers,
            "ffn_neurons": self.ffn_neurons,
            "n_atten_head": self.n_atten_head,
            "activation_function": self.activation_function,
            "precision": RESERVED_PRECISION_DICT[self.dtype],
            "use_amp": self.use_amp,
            "exclude_types": self.exclude_types,
            "env_protection": self.eps,
            "use_env_seed": self.use_env_seed,
            "env_seed_embed_dim": self.env_seed_embed_dim,
            "env_seed_norm": self.env_seed_norm,
            "env_seed_max": self.env_seed_max,
            "trainable": self.trainable,
            "seed": self.seed,
            "type_embedding": self.type_embedding.embedding.serialize(),
            "env_seed_embedding": (
                self.env_seed_embedding.serialize()
                if self.env_seed_embedding is not None
                else None
            ),
            "radial_basis": self.radial_basis.serialize(),
            "radial_embedding": self.radial_embedding.serialize(),
            "gie": self.gie.serialize() if self.gie is not None else None,
            "blocks": [blk.serialize() for blk in self.blocks],
            "so3_linear_output": self.so3_linear_output.serialize(),
            "env_mat": DPEnvMat(self.rcut, self.rcut, self.eps).serialize(),
            "@variables": {
                "davg": np_safe(self.mean),
                "dstd": np_safe(self.stddev),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> DescrptSeZMNet:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "Descriptor":
            raise ValueError(f"Invalid class for DescrptSeZMNet: {data_cls}")
        type_val = data.pop("type")
        if type_val != "se_zm_net":
            raise ValueError(f"Invalid type for DescrptSeZMNet: {type_val}")
        version = int(data.pop("@version"))
        if version != 1:
            raise ValueError(f"Unsupported se_zm_net version: {version}")

        stats = data.pop("@variables")
        data.pop("env_mat")
        type_embedding = data.pop("type_embedding")
        env_seed_embedding_data = data.pop("env_seed_embedding", None)
        radial_basis_data = data.pop("radial_basis")
        radial_embedding_data = data.pop("radial_embedding")
        gie_data = data.pop("gie", None)
        blocks_data = data.pop("blocks")
        so3_linear_output_data = data.pop("so3_linear_output")

        obj = cls(**data)
        obj.type_embedding.embedding = TypeEmbedNetConsistent.deserialize(
            type_embedding
        )
        if env_seed_embedding_data is not None:
            obj.env_seed_embedding = EnvironmentInitialEmbedding.deserialize(
                env_seed_embedding_data
            )
        obj.radial_basis = RadialBasis.deserialize(radial_basis_data)
        obj.radial_embedding = RadialMLP.deserialize(radial_embedding_data)

        if gie_data is not None:
            obj.gie = GeometricInitialEmbedding.deserialize(gie_data)

        obj.blocks = nn.ModuleList(
            [SeZMInteractionBlock.deserialize(blk_data) for blk_data in blocks_data]
        )
        obj.so3_linear_output = SO3Linear.deserialize(so3_linear_output_data)

        obj.mean = safe_numpy_to_tensor(
            stats["davg"], device=obj.device, dtype=obj.mean.dtype
        )
        obj.stddev = safe_numpy_to_tensor(
            stats["dstd"], device=obj.device, dtype=obj.stddev.dtype
        )
        return obj

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        """
        Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            Data used to do neighbor statistics.
        type_map : list[str] | None
            The name of each type of atoms.
        local_jdata : dict
            The local data refer to the current class.

        Returns
        -------
        dict
            The updated local data.
        float | None
            The minimum distance between two atoms.
        """
        local_jdata_cpy = local_jdata.copy()
        min_nbor_dist, sel = UpdateSel().update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["rcut"],
            local_jdata_cpy["sel"],
            True,  # mixed_type=True for unified sel
        )
        local_jdata_cpy["sel"] = sel[0]
        return local_jdata_cpy, min_nbor_dist
