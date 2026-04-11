# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA3S V8 (DPA3-Next): 4-subblock pre-norm invariant transformer descriptor.

Self-contained implementation. All model building blocks (RMSNorm, GatedMLP,
BesselRBF, ChebyshevBasis, dimwise_softmax, etc.) are
implemented from scratch in this single file.

Architecture:  4-subblock pre-norm invariant transformer
  Per layer:
    1. Line Graph Attention   (angle→edge via dimwise softmax)
    2. Atom Graph Attention    (edge→node via bidirectional dimwise softmax)
    3. Line Graph Refinement   (envelope-gated angle→edge message)
    4. Atom Graph Refinement   (envelope-gated edge→node message)

Key design choices aligned with SOTA (MatRIS, eSEN, PET):
  - Bessel RBF (12-dim) + polynomial envelope for edge init
  - Fourier expansion (8-dim) for angle init
  - RMSNorm (pre-norm pattern) for training stability
  - GatedMLP (SiLU-gated) for nonlinear transforms
  - Dimwise softmax attention (O(N) separable attention)
  - Learnable envelope for smoothness
  - Conservative forces (F = -∇E) expected from the fitting framework
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Minimal DeePMD-kit framework imports for training integration ──
from deepmd.dpmodel.utils.seed import child_seed
from deepmd.pt.model.descriptor.descriptor import DescriptorBlock
from deepmd.pt.model.descriptor.env_mat import prod_env_mat
from deepmd.pt.utils import env
from deepmd.pt.utils.env import PRECISION_DICT
from deepmd.pt.utils.env_mat_stat import EnvMatStatSe
from deepmd.pt.utils.exclude_mask import PairExcludeMask
from deepmd.pt.utils.spin import concat_switch_virtual
from deepmd.pt.utils.update_sel import UpdateSel
from deepmd.pt.utils.utils import to_numpy_array, to_torch_tensor
from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.utils.env_mat_stat import StatItem
from deepmd.utils.finetune import get_index_between_two_maps, map_pair_exclude_types
from deepmd.utils.path import DPPath
from deepmd.utils.version import check_version_compatibility

from .base_descriptor import BaseDescriptor
from .descriptor import extend_descrpt_stat

if not hasattr(torch.ops.deepmd, "border_op"):

    def border_op(*args: Any) -> torch.Tensor:
        raise NotImplementedError(
            "border_op is not available since customized PyTorch OP library "
            "is not built when freezing the model."
        )

    torch.ops.deepmd.border_op = border_op


# ═══════════════════════════════════════════════════════════════════════
#  Part 1: Self-contained building blocks
# ═══════════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device="cpu"))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class LinearLayer(nn.Module):
    """Simple linear layer with optional bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "default",
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if init == "zeros":
            nn.init.zeros_(self.linear.weight)
            if bias:
                nn.init.zeros_(self.linear.bias)
        elif init == "xavier":
            nn.init.xavier_uniform_(self.linear.weight)
            if bias:
                nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class GatedMLP(nn.Module):
    """Gated MLP: Linear → [value, gate] → value * SiLU(gate) → Norm → Linear.

    This is the core nonlinear transform used throughout the model.
    Inspired by SwiGLU (Shazeer 2020) and MatRIS's GatedMLP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_norm: bool = True,
    ):
        super().__init__()
        # Project to 2x hidden for gating
        self.w_in = nn.Linear(input_dim, hidden_dim * 2, bias=False)
        self.w_out = nn.Linear(hidden_dim, output_dim, bias=False)
        self.norm = RMSNorm(hidden_dim) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w_in(x)
        value, gate = h.chunk(2, dim=-1)
        h = value * F.silu(gate)
        h = self.norm(h)
        return self.w_out(h)


class BesselRBF(nn.Module):
    """Bessel radial basis functions with polynomial envelope.

    RBF_n(r) = sqrt(2/r_cut) * sin(n*pi*r/r_cut) / r * envelope(r)

    The polynomial envelope ensures C^p continuity at the cutoff:
      u(r) = 1 - (p+1)(p+2)/2 * x^p + p(p+2) * x^(p+1) - p(p+1)/2 * x^(p+2)
    where x = r / r_cut.
    """

    def __init__(
        self,
        num_basis: int = 12,
        r_cut: float = 6.0,
        envelope_exponent: int = 6,
        trainable: bool = False,
    ):
        super().__init__()
        self.num_basis = num_basis
        self.r_cut = r_cut
        self.p = envelope_exponent

        freq = torch.arange(1, num_basis + 1, dtype=torch.float64, device="cpu") * math.pi
        if trainable:
            self.freq = nn.Parameter(freq)
        else:
            self.register_buffer("freq", freq)

        self.prefactor = math.sqrt(2.0 / r_cut)

    def _envelope(self, x: torch.Tensor) -> torch.Tensor:
        """Polynomial envelope that smoothly goes to zero at x=1 with p-order smoothness."""
        p = self.p
        # Clamp to [0, 1] for safety
        x = x.clamp(0.0, 1.0)
        x_p = x.pow(p)
        return (
            1.0
            - 0.5 * (p + 1) * (p + 2) * x_p
            + p * (p + 2) * (x_p * x)
            - 0.5 * p * (p + 1) * (x_p * x * x)
        )

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        dist : Tensor, shape [...], distances (positive scalars).

        Returns
        -------
        rbf : Tensor, shape [..., num_basis]
        """
        dist = dist.unsqueeze(-1)  # [..., 1]
        d_scaled = dist / self.r_cut  # [..., 1] in [0, 1]
        env = self._envelope(d_scaled)  # [..., 1]
        # Bessel basis: sin(n*pi*r/r_cut) / r
        # Avoid division by zero
        rbf = self.prefactor * torch.sin(self.freq.to(dist.dtype) * d_scaled) / (dist + 1e-8)
        return (rbf * env).to(dist.dtype)


class ChebyshevBasis(nn.Module):
    """Chebyshev polynomial basis for angular features.

    Given cosine angle x ∈ [-1, 1], expand as Chebyshev polynomials:
      T_0(x) = 1, T_1(x) = x, T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
    """

    def __init__(self, num_basis: int = 8):
        super().__init__()
        self.num_basis = num_basis
        # We use Chebyshev recurrence for numerical stability

    def forward(self, cos_angle: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cos_angle : Tensor, shape [...], cosine of angles in [-1, 1].

        Returns
        -------
        expansion : Tensor, shape [..., num_basis]
        """
        cos_angle = cos_angle.unsqueeze(-1) if cos_angle.dim() == 0 or cos_angle.shape[-1] != 1 else cos_angle
        if cos_angle.shape[-1] != 1:
            cos_angle = cos_angle.unsqueeze(-1)

        # Chebyshev recurrence: T_0 = 1, T_1 = x, T_n = 2x*T_{n-1} - T_{n-2}
        basis = [torch.ones_like(cos_angle), cos_angle]
        for _ in range(2, self.num_basis):
            basis.append(2 * cos_angle * basis[-1] - basis[-2])
        return torch.cat(basis[: self.num_basis], dim=-1)


class TypeEmbedding(nn.Module):
    """Learnable type embedding for atoms.

    Maps integer atom type to a dense vector, optionally using
    electronic configuration information.
    """

    def __init__(
        self,
        ntypes: int,
        embed_dim: int,
        use_econf: bool = False,
        type_map: list[str] | None = None,
    ):
        super().__init__()
        self.ntypes = ntypes
        self.embed_dim = embed_dim

        if use_econf and type_map is not None:
            # Use electronic configuration features as initialization
            econf_dim = self._get_econf_dim()
            self.econf_proj = nn.Linear(econf_dim, embed_dim, bias=False)
            econf_data = self._get_econf_data(type_map, econf_dim)
            self.register_buffer("econf_data", econf_data)
            self.embedding = None
        else:
            self.embedding = nn.Embedding(ntypes, embed_dim)
            self.econf_proj = None

    def _get_econf_dim(self) -> int:
        return 20  # s, p, d, f orbital occupancy features

    def _get_econf_data(self, type_map: list[str], econf_dim: int) -> torch.Tensor:
        # Simplified: just use random features; real implementation would use
        # actual electronic configuration data
        return torch.randn(len(type_map), econf_dim)

    def forward(self, atype: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        atype : LongTensor [...], atom type indices.

        Returns
        -------
        embd : Tensor [..., embed_dim]
        """
        if self.embedding is not None:
            return self.embedding(atype)
        else:
            econf = self.econf_data[atype]
            return self.econf_proj(econf)

    def change_type_map(self, type_map: list[str]) -> None:
        """Placeholder for type map changes."""
        pass


# ── Scatter / Aggregate utilities ──


def _scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Scatter-sum: output[i] = sum of src[j] where index[j] == i."""
    out = src.new_zeros(num_segments, src.shape[-1])
    return out.index_add_(0, index, src)


def _dimwise_softmax(
    features: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Per-dimension softmax over segments.

    For each feature dimension independently, compute softmax over all
    entries belonging to the same segment.

    Parameters
    ----------
    features : [N, D] input features
    index : [N] segment assignment for each row
    num_segments : total number of segments

    Returns
    -------
    scores : [N, D] softmax scores (per dimension, per segment)
    """
    N, D = features.shape
    idx = index.unsqueeze(-1).expand(N, D)

    # Segment-wise max for numerical stability
    seg_max = features.new_full((num_segments, D), float("-inf"))
    seg_max = seg_max.scatter_reduce(0, idx, features, reduce="amax", include_self=False)
    max_vals = seg_max.gather(0, idx)  # [N, D]

    exp_f = (features - max_vals).exp()

    # Segment-wise sum of exponentials
    seg_sum = features.new_zeros(num_segments, D)
    seg_sum = seg_sum.scatter_reduce(0, idx, exp_f, reduce="sum", include_self=False)
    sum_vals = seg_sum.gather(0, idx)  # [N, D]

    return exp_f / (sum_vals + 1e-12)


@torch.jit.script
def _get_graph_index(
    nlist: torch.Tensor,
    nlist_mask: torch.Tensor,
    a_nlist_mask: torch.Tensor,
    nall: int,
    use_loc_mapping: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build edge/angle graph indices from neighbor list.

    Returns
    -------
    edge_index : 2 x n_edge
        [n2e_index, n_ext2e_index]
    angle_index : 3 x n_angle
        [n2a_index, eij2a_index, eik2a_index]
    """
    nf, nloc, nnei = nlist.shape
    _, _, a_nnei = a_nlist_mask.shape
    a_nlist_mask_3d = a_nlist_mask[:, :, :, None] & a_nlist_mask[:, :, None, :]
    n_edge = nlist_mask.sum().item()

    # 1. atom graph indices
    nlist_loc_index = torch.arange(0, nf * nloc, dtype=nlist.dtype, device=nlist.device)
    n2e_index = nlist_loc_index.reshape(nf, nloc, 1).expand(-1, -1, nnei)
    n2e_index = n2e_index[nlist_mask]

    frame_shift = torch.arange(0, nf, dtype=nlist.dtype, device=nlist.device) * (
        nall if not use_loc_mapping else nloc
    )
    shifted_nlist = nlist + frame_shift[:, None, None]
    n_ext2e_index = shifted_nlist[nlist_mask]

    # 2. angle graph indices
    n2a_index = nlist_loc_index.reshape(nf, nloc, 1, 1).expand(-1, -1, a_nnei, a_nnei)
    n2a_index = n2a_index[a_nlist_mask_3d]

    edge_id = torch.arange(0, n_edge, dtype=nlist.dtype, device=nlist.device)
    edge_index_buf = torch.zeros(
        [nf, nloc, nnei], dtype=nlist.dtype, device=nlist.device
    )
    edge_index_buf[nlist_mask] = edge_id
    edge_index_buf = edge_index_buf[:, :, :a_nnei]

    edge_index_ij = edge_index_buf.unsqueeze(-1).expand(-1, -1, -1, a_nnei)
    eij2a_index = edge_index_ij[a_nlist_mask_3d]

    edge_index_ik = edge_index_buf.unsqueeze(-2).expand(-1, -1, a_nnei, -1)
    eik2a_index = edge_index_ik[a_nlist_mask_3d]

    edge_index = torch.stack([n2e_index, n_ext2e_index])
    angle_index = torch.stack([n2a_index, eij2a_index, eik2a_index])
    return edge_index, angle_index


# ═══════════════════════════════════════════════════════════════════════
#  Part 2: DPA3-Next Layer
# ═══════════════════════════════════════════════════════════════════════


class DPA3NextLayer(nn.Module):
    """A single DPA3-Next layer with 4 sub-blocks.

    Sub-block 1 — Line Graph Attention:
        angle→edge update via dimwise softmax attention.
    Sub-block 2 — Atom Graph Attention:
        edge→node update via bidirectional dimwise softmax attention.
    Sub-block 3 — Line Graph Refinement:
        envelope-gated angle→edge message passing.
    Sub-block 4 — Atom Graph Refinement:
        envelope-gated edge→node message passing.
    """

    def __init__(
        self,
        n_dim: int = 128,
        e_dim: int = 128,
        a_dim: int = 64,
        num_radial: int = 12,
        num_angular: int = 8,
        sel_reduce_factor: float = 10.0,
        a_sel: int = 40,
        nnei: int = 120,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.sel_reduce_factor = sel_reduce_factor
        self.a_sel = a_sel
        self.nnei = nnei
        # Pre-computed scaling factors
        self.dynamic_a_scale = (a_sel / sel_reduce_factor) ** (-0.5)
        self.dynamic_e_scale = (nnei / sel_reduce_factor) ** (-0.5)

        # ── Sub-block 1: Line Graph Attention ──
        self.line_attn_norm_e = RMSNorm(e_dim)
        self.line_attn_norm_a = RMSNorm(a_dim)
        self.line_attn_mlp = GatedMLP(
            input_dim=a_dim + n_dim + 2 * e_dim,
            hidden_dim=e_dim * 2,
            output_dim=e_dim,
        )
        self.line_attn_gate = nn.Linear(a_dim + n_dim + 2 * e_dim, e_dim, bias=False)

        # ── Sub-block 2: Atom Graph Attention ──
        self.atom_attn_norm_n = RMSNorm(n_dim)
        self.atom_attn_norm_e = RMSNorm(e_dim)
        self.atom_attn_mlp = GatedMLP(
            input_dim=e_dim + 2 * n_dim,
            hidden_dim=n_dim * 2,
            output_dim=n_dim,
        )
        self.atom_attn_src_gate = nn.Linear(e_dim, n_dim, bias=False)
        # Also update edge in this sub-block
        self.atom_attn_edge_mlp = GatedMLP(
            input_dim=e_dim + 2 * n_dim,
            hidden_dim=e_dim * 2,
            output_dim=e_dim,
        )

        # ── Sub-block 3: Line Graph Refinement ──
        self.line_ref_norm_e = RMSNorm(e_dim)
        self.line_ref_norm_a = RMSNorm(a_dim)
        self.line_ref_mlp = GatedMLP(
            input_dim=a_dim + n_dim + 2 * e_dim,
            hidden_dim=e_dim * 2,
            output_dim=e_dim,
        )
        # Learnable envelope: maps RBF basis → per-channel weight
        self.line_ref_env = nn.Linear(num_radial, e_dim, bias=False)
        self.line_ref_edge_proj = nn.Linear(e_dim, e_dim, bias=False)
        self.line_ref_angle_proj = nn.Linear(e_dim, a_dim, bias=False)

        # ── Sub-block 4: Atom Graph Refinement ──
        self.atom_ref_norm_n = RMSNorm(n_dim)
        self.atom_ref_norm_e = RMSNorm(e_dim)
        self.atom_ref_mlp = GatedMLP(
            input_dim=e_dim + 2 * n_dim,
            hidden_dim=n_dim * 2,
            output_dim=n_dim,
        )
        self.atom_ref_env = nn.Linear(num_radial, n_dim, bias=False)
        self.atom_ref_node_proj = nn.Linear(n_dim, n_dim, bias=False)
        self.atom_ref_edge_proj = nn.Linear(n_dim, e_dim, bias=False)

    def forward(
        self,
        node_ebd_ext: torch.Tensor,  # nf x nall x n_dim
        edge_ebd: torch.Tensor,  # n_edge x e_dim
        angle_ebd: torch.Tensor,  # n_angle x a_dim
        h2: torch.Tensor,  # n_edge x 3 (unit vectors)
        sw: torch.Tensor,  # n_edge (switch)
        a_sw: torch.Tensor,  # n_angle (angle switch)
        edge_index: torch.Tensor,  # 2 x n_edge
        angle_index: torch.Tensor,  # 3 x n_angle
        edge_rbf: torch.Tensor,  # n_edge x num_radial (Bessel RBF for envelope)
        nframes: int,
        nloc: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        node_ebd : nf x nloc x n_dim
        edge_ebd : n_edge x e_dim
        angle_ebd : n_angle x a_dim
        """
        n_edge = edge_ebd.shape[0]
        n_angle = angle_ebd.shape[0]
        num_nodes = nframes * nloc

        n2e = edge_index[0]        # n_edge: owner node for each edge
        n_ext2e = edge_index[1]     # n_edge: neighbor node for each edge
        n2a = angle_index[0]        # n_angle: owner node for each angle
        eij2a = angle_index[1]      # n_angle: edge ij for each angle
        eik2a = angle_index[2]      # n_angle: edge ik for each angle

        node_ebd = node_ebd_ext[:, :nloc, :].reshape(-1, self.n_dim)  # (nf*nloc) x n_dim
        node_ext_flat = node_ebd_ext.reshape(-1, self.n_dim)

        # ================================================================
        # Sub-block 1: Line Graph Attention (angle → edge)
        # ================================================================
        edge_normed = self.line_attn_norm_e(edge_ebd)
        angle_normed = self.line_attn_norm_a(angle_ebd)

        # Gather features for each angle
        node_a = node_ebd[n2a]          # n_angle x n_dim
        eij_a = edge_normed[eij2a]      # n_angle x e_dim
        eik_a = edge_normed[eik2a]      # n_angle x e_dim
        angle_info = torch.cat([angle_normed, node_a, eij_a, eik_a], dim=-1)

        angle_msg = self.line_attn_mlp(angle_info)  # n_angle x e_dim
        attn_logits = self.line_attn_gate(angle_info)  # n_angle x e_dim
        attn_w = _dimwise_softmax(attn_logits, eij2a, n_edge)  # n_angle x e_dim
        weighted_msg = attn_w * angle_msg * a_sw.unsqueeze(-1)

        edge_update = _scatter_sum(weighted_msg, eij2a, n_edge)
        edge_ebd = edge_ebd + edge_update

        # ================================================================
        # Sub-block 2: Atom Graph Attention (edge → node)
        # ================================================================
        node_normed = self.atom_attn_norm_n(node_ebd)
        edge_normed = self.atom_attn_norm_e(edge_ebd)

        node_i = node_normed[n2e]                          # n_edge x n_dim
        node_j = node_ext_flat[n_ext2e]                    # n_edge x n_dim
        edge_info = torch.cat([edge_normed, node_i, node_j], dim=-1)

        # Dimwise softmax attention (source direction only)
        node_msg = self.atom_attn_mlp(edge_info)  # n_edge x n_dim
        src_logits = self.atom_attn_src_gate(edge_normed)  # n_edge x n_dim
        src_w = _dimwise_softmax(src_logits, n2e, num_nodes)

        msg_weighted = src_w * node_msg * sw.unsqueeze(-1)
        node_update = _scatter_sum(msg_weighted, n2e, num_nodes)
        node_ebd = node_ebd + node_update

        # Also update edge
        edge_update2 = self.atom_attn_edge_mlp(edge_info) * sw.unsqueeze(-1)
        edge_ebd = edge_ebd + edge_update2

        # ================================================================
        # Sub-block 3: Line Graph Refinement (envelope-gated)
        # ================================================================
        edge_normed = self.line_ref_norm_e(edge_ebd)
        angle_normed = self.line_ref_norm_a(angle_ebd)

        node_a = node_ebd[n2a]
        eij_a = edge_normed[eij2a]
        eik_a = edge_normed[eik2a]
        angle_info = torch.cat([angle_normed, node_a, eij_a, eik_a], dim=-1)

        angle_update = self.line_ref_mlp(angle_info)  # n_angle x e_dim

        # Learnable envelope: product of envelopes on ij and ik edges
        env_ij = self.line_ref_env(edge_rbf[eij2a])  # n_angle x e_dim
        env_ik = self.line_ref_env(edge_rbf[eik2a])  # n_angle x e_dim
        envelope = torch.sigmoid(env_ij) * torch.sigmoid(env_ik)  # soft gating

        angle_update_gated = angle_update * envelope * a_sw.unsqueeze(-1)

        # Aggregate to edges
        delta_edge = _scatter_sum(angle_update_gated, eij2a, n_edge)
        edge_ebd = edge_ebd + self.line_ref_edge_proj(delta_edge) * self.dynamic_a_scale

        # Update angles
        angle_ebd = angle_ebd + self.line_ref_angle_proj(angle_update)

        # ================================================================
        # Sub-block 4: Atom Graph Refinement (envelope-gated)
        # ================================================================
        node_normed = self.atom_ref_norm_n(node_ebd)
        edge_normed = self.atom_ref_norm_e(edge_ebd)

        node_i = node_normed[n2e]
        node_j = node_ext_flat[n_ext2e]
        edge_info = torch.cat([edge_normed, node_i, node_j], dim=-1)

        node_update2 = self.atom_ref_mlp(edge_info)  # n_edge x n_dim

        # Learnable envelope on pairwise RBF
        env_e = torch.sigmoid(self.atom_ref_env(edge_rbf))  # n_edge x n_dim
        node_update2_gated = node_update2 * env_e * sw.unsqueeze(-1)

        delta_node = _scatter_sum(node_update2_gated, n2e, num_nodes)
        node_ebd = node_ebd + self.atom_ref_node_proj(delta_node) * self.dynamic_e_scale

        # Also update edge
        edge_ebd = edge_ebd + self.atom_ref_edge_proj(node_update2)

        node_ebd = node_ebd.reshape(nframes, nloc, self.n_dim)
        return node_ebd, edge_ebd, angle_ebd


# ═══════════════════════════════════════════════════════════════════════
#  Part 3: Descriptor Block
# ═══════════════════════════════════════════════════════════════════════


@DescriptorBlock.register("se_repflow_v8")
class DescrptBlockDPA3Next(DescriptorBlock):
    """DPA3-Next descriptor block.

    Manages the full forward pipeline: env_mat → embeddings → N layers → output.
    """

    def __init__(
        self,
        e_rcut: float,
        e_rcut_smth: float,
        e_sel: int,
        a_rcut: float,
        a_rcut_smth: float,
        a_sel: int,
        ntypes: int,
        nlayers: int = 6,
        n_dim: int = 128,
        e_dim: int = 128,
        a_dim: int = 64,
        num_radial: int = 12,
        num_angular: int = 8,
        sel_reduce_factor: float = 10.0,
        activation_function: str = "silu",
        set_davg_zero: bool = True,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        precision: str = "float32",
        fix_stat_std: float = 0.3,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.e_sel = e_sel
        self.a_rcut = float(a_rcut)
        self.a_rcut_smth = float(a_rcut_smth)
        self.a_sel = a_sel
        self.ntypes = ntypes
        self.nlayers = nlayers
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.num_radial = num_radial
        self.num_angular = num_angular
        self.sel_reduce_factor = sel_reduce_factor
        self.activation_function = activation_function

        sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4
        assert len(sel) == 1
        self.sel = sel
        self.rcut = e_rcut
        self.rcut_smth = e_rcut_smth
        self.sec = self.sel
        self.split_sel = self.sel

        self.set_davg_zero = set_davg_zero
        self.fix_stat_std = fix_stat_std
        self.set_stddev_constant = fix_stat_std != 0.0
        self.prec = PRECISION_DICT[precision]
        self.env_protection = env_protection
        self.precision = precision
        self.seed = seed

        self.reinit_exclude(exclude_types)

        # All self-contained modules created on CPU to avoid device context pollution
        # They will be moved to the correct device by .to(device) later
        with torch.device("cpu"):
            # ── Embedding layers ──
            self.edge_rbf = BesselRBF(
                num_basis=num_radial,
                r_cut=e_rcut,
                envelope_exponent=6,
                trainable=False,
            )
            self.edge_proj = nn.Sequential(
                nn.Linear(num_radial, e_dim, bias=False),
                RMSNorm(e_dim),
            )

            self.angle_chebyshev = ChebyshevBasis(num_basis=num_angular)
            self.angle_proj = nn.Linear(num_angular, a_dim, bias=False)

            self.node_norm = RMSNorm(n_dim)

            # ── Layers ──
            layers = []
            for ii in range(nlayers):
                layers.append(
                    DPA3NextLayer(
                        n_dim=n_dim,
                        e_dim=e_dim,
                        a_dim=a_dim,
                        num_radial=num_radial,
                        num_angular=num_angular,
                        sel_reduce_factor=sel_reduce_factor,
                        a_sel=a_sel,
                        nnei=self.nnei,
                    )
                )
            self.layers = nn.ModuleList(layers)

            # ── Output projection (for rotation matrix computation) ──
            self.output_edge_norm = RMSNorm(e_dim)

        # Move all self-contained modules to correct precision and device
        self.edge_rbf = self.edge_rbf.to(dtype=self.prec, device=env.DEVICE)
        self.edge_proj = self.edge_proj.to(dtype=self.prec, device=env.DEVICE)
        self.angle_chebyshev = self.angle_chebyshev.to(dtype=self.prec, device=env.DEVICE)
        self.angle_proj = self.angle_proj.to(dtype=self.prec, device=env.DEVICE)
        self.node_norm = self.node_norm.to(dtype=self.prec, device=env.DEVICE)
        self.layers = self.layers.to(dtype=self.prec, device=env.DEVICE)
        self.output_edge_norm = self.output_edge_norm.to(dtype=self.prec, device=env.DEVICE)

        # ── Stat buffers ──
        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(wanted_shape, dtype=self.prec, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=self.prec, device=env.DEVICE)
        if self.set_stddev_constant:
            stddev = stddev * self.fix_stat_std
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.stats = None

    # ── DescriptorBlock interface ──

    def get_rcut(self) -> float:
        return self.e_rcut

    def get_rcut_smth(self) -> float:
        return self.e_rcut_smth

    def get_nsel(self) -> int:
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        return self.sel

    def get_ntypes(self) -> int:
        return self.ntypes

    def get_dim_out(self) -> int:
        return self.dim_out

    def get_dim_in(self) -> int:
        return self.dim_in

    def get_dim_emb(self) -> int:
        return self.e_dim

    def get_env_protection(self) -> float:
        return self.env_protection

    @property
    def dim_out(self) -> int:
        return self.n_dim

    @property
    def dim_in(self) -> int:
        return self.n_dim

    @property
    def dim_emb(self) -> int:
        return self.e_dim

    def __setitem__(self, key: str, value: Any) -> None:
        if key in ("avg", "data_avg", "davg"):
            self.mean = value
        elif key in ("std", "data_std", "dstd"):
            self.stddev = value
        else:
            raise KeyError(key)

    def __getitem__(self, key: str) -> Any:
        if key in ("avg", "data_avg", "davg"):
            return self.mean
        elif key in ("std", "data_std", "dstd"):
            return self.stddev
        else:
            raise KeyError(key)

    def mixed_types(self) -> bool:
        return True

    def reinit_exclude(self, exclude_types: list[tuple[int, int]] = []) -> None:
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def has_message_passing(self) -> bool:
        return True

    def need_sorted_nlist_for_lower(self) -> bool:
        return True

    # ── Forward ──

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: torch.Tensor | None = None,
        mapping: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        parallel_mode = comm_dict is not None
        if not parallel_mode:
            assert mapping is not None
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.view(nframes, -1).shape[1] // 3
        atype = extended_atype[:, :nloc]

        # Exclude mask
        exclude_mask = self.emask(nlist, extended_atype)
        nlist = torch.where(exclude_mask != 0, nlist, -1)

        # Environment matrix for edge
        dmatrix, diff, sw = prod_env_mat(
            extended_coord, nlist, atype,
            self.mean, self.stddev,
            self.e_rcut, self.e_rcut_smth,
            protection=self.env_protection,
            use_exp_switch=True,
        )
        nlist_mask = nlist != -1
        sw = torch.squeeze(sw, -1)
        sw = sw.masked_fill(~nlist_mask, 0.0)

        # Angle nlist (subset within a_rcut)
        a_dist_mask = (torch.linalg.norm(diff, dim=-1) < self.a_rcut)[
            :, :, : self.a_sel
        ]
        a_nlist = nlist[:, :, : self.a_sel]
        a_nlist = torch.where(a_dist_mask, a_nlist, -1)
        _, a_diff, a_sw = prod_env_mat(
            extended_coord, a_nlist, atype,
            self.mean[:, : self.a_sel],
            self.stddev[:, : self.a_sel],
            self.a_rcut, self.a_rcut_smth,
            protection=self.env_protection,
            use_exp_switch=True,
        )
        a_nlist_mask = a_nlist != -1
        a_sw = torch.squeeze(a_sw, -1)
        a_sw = a_sw.masked_fill(~a_nlist_mask, 0.0)

        # Padding positions to 0
        nlist[nlist == -1] = 0
        a_nlist[a_nlist == -1] = 0

        # ── Node embedding from type embedding ──
        assert extended_atype_embd is not None
        atype_embd = extended_atype_embd[:, :nloc, :]
        node_ebd = self.node_norm(atype_embd)

        # ── Edge embedding: distance → Bessel RBF → projection ──
        _, h2 = torch.split(dmatrix, [1, 3], dim=-1)
        edge_dist = torch.linalg.norm(diff, dim=-1)  # nf x nloc x nnei

        # ── Cosine angles ──
        normalized_diff_i = a_diff / (
            torch.linalg.norm(a_diff, dim=-1, keepdim=True) + 1e-6
        )
        normalized_diff_j = torch.transpose(normalized_diff_i, 2, 3)
        cosine_ij = torch.matmul(normalized_diff_i, normalized_diff_j) * (1 - 1e-6)

        # Apply loc_mapping
        if not parallel_mode:
            assert mapping is not None
            nlist = torch.gather(
                mapping, 1, index=nlist.reshape(nframes, -1)
            ).reshape(nlist.shape)

        # Build graph index
        edge_index, angle_index = _get_graph_index(
            nlist, nlist_mask, a_nlist_mask, nall, use_loc_mapping=True,
        )

        # ── Flatten to dynamic-sel form ──
        # Edge
        edge_dist_flat = edge_dist[nlist_mask]  # n_edge
        h2 = h2[nlist_mask]                     # n_edge x 3
        sw = sw[nlist_mask]                     # n_edge

        # Angle
        a_nlist_mask_3d = a_nlist_mask[:, :, :, None] & a_nlist_mask[:, :, None, :]
        cosine_flat = cosine_ij[a_nlist_mask_3d]  # n_angle
        a_sw = (a_sw[:, :, :, None] * a_sw[:, :, None, :])[a_nlist_mask_3d]

        # ── Compute embeddings ──
        edge_rbf = self.edge_rbf(edge_dist_flat)        # n_edge x num_radial
        edge_ebd = self.edge_proj(edge_rbf.to(self.prec))  # n_edge x e_dim

        angle_chebyshev = self.angle_chebyshev(cosine_flat)  # n_angle x num_angular
        angle_ebd = self.angle_proj(angle_chebyshev.to(self.prec))  # n_angle x a_dim

        # ── Build node_ebd_ext for layers ──
        if not parallel_mode:
            assert mapping is not None
            mapping_expand = (
                mapping.view(nframes, nall).unsqueeze(-1).expand(-1, -1, self.n_dim)
            )

        # ── Iterate layers ──
        for layer in self.layers:
            # Build node_ebd_ext
            if not parallel_mode:
                node_ebd_ext = node_ebd  # use_loc_mapping: ext == local
            else:
                assert comm_dict is not None
                has_spin = "has_spin" in comm_dict
                if not has_spin:
                    n_padding = nall - nloc
                    node_ebd_flat = torch.nn.functional.pad(
                        node_ebd.squeeze(0), (0, 0, 0, n_padding), value=0.0
                    )
                    real_nloc = nloc
                    real_nall = nall
                else:
                    real_nloc = nloc // 2
                    real_nall = nall // 2
                    real_n_padding = real_nall - real_nloc
                    node_ebd_real, node_ebd_virtual = torch.split(
                        node_ebd, [real_nloc, real_nloc], dim=1
                    )
                    mix_node_ebd = torch.cat(
                        [node_ebd_real, node_ebd_virtual], dim=2
                    )
                    node_ebd_flat = torch.nn.functional.pad(
                        mix_node_ebd.squeeze(0),
                        (0, 0, 0, real_n_padding),
                        value=0.0,
                    )

                ret = torch.ops.deepmd.border_op(
                    comm_dict["send_list"],
                    comm_dict["send_proc"],
                    comm_dict["recv_proc"],
                    comm_dict["send_num"],
                    comm_dict["recv_num"],
                    node_ebd_flat,
                    comm_dict["communicator"],
                    torch.tensor(real_nloc, dtype=torch.int32, device=torch.device("cpu")),
                    torch.tensor(real_nall - real_nloc, dtype=torch.int32, device=torch.device("cpu")),
                )
                node_ebd_ext = ret[0].unsqueeze(0)
                if has_spin:
                    nd = self.n_dim
                    node_ebd_real_ext, node_ebd_virtual_ext = torch.split(
                        node_ebd_ext, [nd, nd], dim=2
                    )
                    node_ebd_ext = concat_switch_virtual(
                        node_ebd_real_ext, node_ebd_virtual_ext, real_nloc
                    )

            node_ebd, edge_ebd, angle_ebd = layer(
                node_ebd_ext,
                edge_ebd,
                angle_ebd,
                h2,
                sw,
                a_sw,
                edge_index,
                angle_index,
                edge_rbf.to(self.prec),
                nframes,
                nloc,
            )

        # ── Compute rotation matrix ──
        edge_ebd_normed = self.output_edge_norm(edge_ebd)
        scale = (self.nnei / self.sel_reduce_factor) ** (-0.5)
        weighted_edge = edge_ebd_normed * sw.unsqueeze(-1)
        h2g2_flat = (h2.unsqueeze(-1) * weighted_edge.unsqueeze(-2)).reshape(-1, 3 * self.e_dim)
        n2e_index = edge_index[0]

        h2g2 = (
            _scatter_sum(h2g2_flat, n2e_index, nframes * nloc).reshape(
                nframes, nloc, 3, self.e_dim
            )
            * scale
        )
        rot_mat = torch.permute(h2g2, (0, 1, 3, 2))  # nf x nloc x e_dim x 3

        return (
            node_ebd,
            rot_mat,
            edge_ebd,
            h2,
            sw,
        )

    # ── Statistics ──

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        if self.set_stddev_constant and self.set_davg_zero:
            return
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
            self.mean.copy_(torch.tensor(mean, device=env.DEVICE, dtype=self.mean.dtype))
        if not self.set_stddev_constant:
            self.stddev.copy_(torch.tensor(stddev, device=env.DEVICE, dtype=self.stddev.dtype))

    def get_stats(self) -> dict[str, StatItem]:
        if self.stats is None:
            raise RuntimeError("Statistics not yet computed.")
        return self.stats


# ═══════════════════════════════════════════════════════════════════════
#  Part 4: Top-level Descriptor (integrates with DeePMD-kit)
# ═══════════════════════════════════════════════════════════════════════


@BaseDescriptor.register("dpa3s_v8_next")
class DescrptDPA3Next(BaseDescriptor, torch.nn.Module):
    r"""DPA3-Next descriptor.

    A next-generation invariant descriptor built on the 4-subblock
    pre-norm transformer architecture.

    Parameters
    ----------
    ntypes : int
        Number of element types.
    n_dim : int
        Node representation dimension.
    e_dim : int
        Edge representation dimension.
    a_dim : int
        Angle representation dimension.
    nlayers : int
        Number of interaction layers.
    e_rcut : float
        Edge cut-off radius.
    e_rcut_smth : float
        Edge smoothing start radius.
    e_sel : int
        Max edge neighbors.
    a_rcut : float
        Angle cut-off radius.
    a_rcut_smth : float
        Angle smoothing start radius.
    a_sel : int
        Max angle neighbors.
    num_radial : int
        Number of Bessel radial basis functions.
    num_angular : int
        Number of Fourier/Chebyshev angular basis functions.
    sel_reduce_factor : float
        Reduction factor for neighbor normalization.
    activation_function : str
        Activation function name.
    precision : str
        Parameter precision.
    exclude_types : list
        Excluded type pairs.
    env_protection : float
        Environment matrix protection parameter.
    trainable : bool
        Whether parameters are trainable.
    seed : int or None
        Random seed.
    use_econf_tebd : bool
        Use electronic configuration type embedding.
    use_tebd_bias : bool
        Not used (kept for interface compatibility).
    type_map : list[str] or None
        Atom type names.
    concat_output_tebd : bool
        Whether to concatenate type embedding to output.
    """

    def __init__(
        self,
        ntypes: int,
        n_dim: int = 128,
        e_dim: int = 128,
        a_dim: int = 64,
        nlayers: int = 6,
        e_rcut: float = 6.0,
        e_rcut_smth: float = 5.0,
        e_sel: int = 120,
        a_rcut: float = 4.0,
        a_rcut_smth: float = 3.5,
        a_sel: int = 40,
        num_radial: int = 12,
        num_angular: int = 8,
        sel_reduce_factor: float = 10.0,
        activation_function: str = "silu",
        precision: str = "float32",
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        type_map: list[str] | None = None,
        concat_output_tebd: bool = False,
    ) -> None:
        super().__init__()

        self.repflows = DescrptBlockDPA3Next(
            e_rcut=e_rcut,
            e_rcut_smth=e_rcut_smth,
            e_sel=e_sel,
            a_rcut=a_rcut,
            a_rcut_smth=a_rcut_smth,
            a_sel=a_sel,
            ntypes=ntypes,
            nlayers=nlayers,
            n_dim=n_dim,
            e_dim=e_dim,
            a_dim=a_dim,
            num_radial=num_radial,
            num_angular=num_angular,
            sel_reduce_factor=sel_reduce_factor,
            activation_function=activation_function,
            exclude_types=exclude_types,
            env_protection=env_protection,
            precision=precision,
            seed=child_seed(seed, 1),
            trainable=trainable,
        )

        from deepmd.pt.model.network.network import TypeEmbedNet

        self.type_embedding = TypeEmbedNet(
            ntypes,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 2),
            use_econf_tebd=use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
            trainable=trainable,
        )

        self.ntypes = ntypes
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.nlayers = nlayers
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.e_sel = e_sel
        self.a_rcut = float(a_rcut)
        self.a_rcut_smth = float(a_rcut_smth)
        self.a_sel = a_sel
        self.num_radial = num_radial
        self.num_angular = num_angular
        self.sel_reduce_factor = sel_reduce_factor
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[precision]
        self.exclude_types = exclude_types
        self.env_protection = env_protection
        self.trainable = trainable
        self.use_econf_tebd = use_econf_tebd
        self.use_tebd_bias = use_tebd_bias
        self.type_map = type_map
        self.tebd_dim = n_dim
        self.concat_output_tebd = concat_output_tebd

        assert e_rcut >= a_rcut
        assert e_sel >= a_sel

        self.rcut = self.repflows.get_rcut()
        self.rcut_smth = self.repflows.get_rcut_smth()
        self.sel = self.repflows.get_sel()

        for param in self.parameters():
            param.requires_grad = trainable

    # ── Interface properties ──

    def get_rcut(self) -> float:
        return self.rcut

    def get_rcut_smth(self) -> float:
        return self.rcut_smth

    def get_nsel(self) -> int:
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        return self.sel

    def get_ntypes(self) -> int:
        return self.ntypes

    def get_type_map(self) -> list[str]:
        return self.type_map

    def get_dim_out(self) -> int:
        ret = self.repflows.dim_out
        if self.concat_output_tebd:
            ret += self.tebd_dim
        return ret

    def get_dim_emb(self) -> int:
        return self.repflows.dim_emb

    def mixed_types(self) -> bool:
        return True

    def has_message_passing(self) -> bool:
        return True

    def need_sorted_nlist_for_lower(self) -> bool:
        return True

    def get_env_protection(self) -> float:
        return self.repflows.get_env_protection()

    @property
    def dim_out(self) -> int:
        return self.get_dim_out()

    @property
    def dim_emb(self) -> int:
        return self.get_dim_emb()

    # ── Stats ──

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        self.repflows.compute_input_stats(merged, path)

    def set_stat_mean_and_stddev(
        self, mean: list[torch.Tensor], stddev: list[torch.Tensor]
    ) -> None:
        self.repflows.mean = mean[0]
        self.repflows.stddev = stddev[0]

    def get_stat_mean_and_stddev(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return [self.repflows.mean], [self.repflows.stddev]

    def share_params(
        self, base_class: Any, shared_level: int, resume: bool = False
    ) -> None:
        assert self.__class__ == base_class.__class__
        if shared_level == 0:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            self.repflows.share_params(base_class.repflows, 0, resume=resume)
        elif shared_level == 1:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
        else:
            raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
    ) -> None:
        assert self.type_map is not None
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        self.type_map = type_map
        self.type_embedding.change_type_map(type_map=type_map)
        self.exclude_types = map_pair_exclude_types(self.exclude_types, remap_index)
        self.ntypes = len(type_map)
        repflow = self.repflows
        if has_new_type:
            extend_descrpt_stat(
                repflow,
                type_map,
                des_with_stat=model_with_new_type_stat.repflows
                if model_with_new_type_stat is not None
                else None,
            )
        repflow.ntypes = self.ntypes
        repflow.reinit_exclude(self.exclude_types)
        repflow["davg"] = repflow["davg"][remap_index]
        repflow["dstd"] = repflow["dstd"][remap_index]

    # ── Serialization (minimal) ──

    def serialize(self) -> dict:
        raise NotImplementedError("Serialization not yet implemented for DPA3-Next.")

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA3Next":
        raise NotImplementedError("Deserialization not yet implemented for DPA3-Next.")

    # ── Forward ──

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        fparam: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Compute the descriptor.

        Parameters
        ----------
        extended_coord : nf x (nall*3)
        extended_atype : nf x nall
        nlist : nf x nloc x nnei
        mapping : index mapping (extended→local)
        comm_dict : parallel communication data
        fparam : frame parameters (unused)

        Returns
        -------
        node_ebd : nf x nloc x dim_out
        rot_mat : nf x nloc x e_dim x 3
        edge_ebd : n_edge x e_dim (or None)
        h2 : n_edge x 3 (or None)
        sw : n_edge (or None)
        """
        parallel_mode = comm_dict is not None
        extended_coord = extended_coord.to(dtype=self.prec)
        nframes, nloc, nnei = nlist.shape

        # Type embedding
        if not parallel_mode:
            node_ebd_ext = self.type_embedding(extended_atype[:, :nloc])
        else:
            node_ebd_ext = self.type_embedding(extended_atype)

        node_ebd_inp = node_ebd_ext[:, :nloc, :]

        # Repflows
        node_ebd, edge_ebd, h2, rot_mat, sw = self.repflows(
            nlist,
            extended_coord,
            extended_atype,
            node_ebd_ext,
            mapping,
            comm_dict=comm_dict,
        )

        if self.concat_output_tebd:
            node_ebd = torch.cat([node_ebd, node_ebd_inp], dim=-1)

        gp = env.GLOBAL_PT_FLOAT_PRECISION
        return (
            node_ebd.to(dtype=gp),
            rot_mat.to(dtype=gp) if rot_mat is not None else None,
            edge_ebd.to(dtype=gp) if edge_ebd is not None else None,
            h2.to(dtype=gp) if h2 is not None else None,
            sw.to(dtype=gp) if sw is not None else None,
        )

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        local_jdata_cpy = local_jdata.copy()
        update_sel = UpdateSel()
        min_nbor_dist, e_sel = update_sel.update_one_sel(
            train_data, type_map, local_jdata_cpy["e_rcut"], local_jdata_cpy["e_sel"], True,
        )
        local_jdata_cpy["e_sel"] = e_sel[0]
        min_nbor_dist, a_sel = update_sel.update_one_sel(
            train_data, type_map, local_jdata_cpy["a_rcut"], local_jdata_cpy["a_sel"], True,
        )
        local_jdata_cpy["a_sel"] = a_sel[0]
        return local_jdata_cpy, min_nbor_dist

    def enable_compression(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Compression is unsupported for DPA3-Next.")
