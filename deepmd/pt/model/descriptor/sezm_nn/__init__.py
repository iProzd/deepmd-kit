# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Public building blocks for the SeZM descriptor.

This package re-exports the helper functions, embeddings, equivariant layers,
and quaternion-based Wigner-D utilities used by the SeZM descriptor and model.
"""

from .activation import (
    GatedActivation,
    S2GridProjector,
    SwiGLU,
    SwiGLUS2Activation,
    resolve_s2_grid_resolution,
)
from .attention import (
    segment_envelope_gated_softmax,
)
from .attn_res import (
    DepthAttnRes,
)
from .block import (
    SeZMInteractionBlock,
)
from .dens import (
    ForceEmbedding,
    SeZMDenoisingHead,
    SeZMDeNSFittingNet,
    SeZMDirectForceHead,
)
from .edge_cache import (
    EdgeFeatureCache,
    build_edge_cache,
    build_edge_cache_from_edges,
    build_edge_type_feat,
    edge_cache_to_dtype,
)
from .embedding import (
    EnvironmentInitialEmbedding,
    GeometricInitialEmbedding,
    SeZMTypeEmbedding,
)
from .ffn import (
    EquivariantFFN,
)
from .indexing import (
    build_l_major_index,
    build_m_major_index,
    build_m_major_l_index,
    build_rotate_inv_rescale,
    get_so3_dim_of_lmax,
    map_degree_idx,
    project_D_to_m,
    project_Dt_from_m,
    so3_packed_index,
)
from .norm import (
    EquivariantRMSNorm,
    ReducedEquivariantRMSNorm,
    RMSNorm,
    ScalarRMSNorm,
)
from .radial import (
    C3CutoffEnvelope,
    InnerClamp,
    RadialBasis,
    RadialMLP,
)
from .so2 import (
    SO2Convolution,
    SO2Linear,
)
from .so3 import (
    ChannelLinear,
    FocusLinear,
    SO3Linear,
)
from .utils import (
    ATTN_RES_MODES,
    get_promoted_dtype,
    init_trunc_normal_fan_in_out,
    np_safe,
    nvtx_range,
    safe_norm,
    safe_numpy_to_tensor,
)
from .wignerd import (
    WignerDCalculator,
    build_edge_quaternion,
    quaternion_multiply,
    quaternion_nlerp,
    quaternion_normalize,
    quaternion_to_rotation_matrix,
    quaternion_z_rotation,
)

__all__ = [
    "ATTN_RES_MODES",
    "C3CutoffEnvelope",
    "ChannelLinear",
    "DepthAttnRes",
    "EdgeFeatureCache",
    "EnvironmentInitialEmbedding",
    "EquivariantFFN",
    "EquivariantRMSNorm",
    "FocusLinear",
    "ForceEmbedding",
    "GatedActivation",
    "GeometricInitialEmbedding",
    "InnerClamp",
    "RMSNorm",
    "RadialBasis",
    "RadialMLP",
    "ReducedEquivariantRMSNorm",
    "S2GridProjector",
    "SO2Convolution",
    "SO2Linear",
    "SO3Linear",
    "ScalarRMSNorm",
    "SeZMDeNSFittingNet",
    "SeZMDenoisingHead",
    "SeZMDirectForceHead",
    "SeZMInteractionBlock",
    "SeZMTypeEmbedding",
    "SwiGLU",
    "SwiGLUS2Activation",
    "WignerDCalculator",
    "build_edge_cache",
    "build_edge_cache_from_edges",
    "build_edge_quaternion",
    "build_edge_type_feat",
    "build_l_major_index",
    "build_m_major_index",
    "build_m_major_l_index",
    "build_rotate_inv_rescale",
    "edge_cache_to_dtype",
    "get_promoted_dtype",
    "get_so3_dim_of_lmax",
    "init_trunc_normal_fan_in_out",
    "map_degree_idx",
    "np_safe",
    "nvtx_range",
    "project_D_to_m",
    "project_Dt_from_m",
    "quaternion_multiply",
    "quaternion_nlerp",
    "quaternion_normalize",
    "quaternion_to_rotation_matrix",
    "quaternion_z_rotation",
    "resolve_s2_grid_resolution",
    "safe_norm",
    "safe_numpy_to_tensor",
    "segment_envelope_gated_softmax",
    "so3_packed_index",
]
