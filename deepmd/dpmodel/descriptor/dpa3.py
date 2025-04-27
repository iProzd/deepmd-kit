# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)


class RepFlowArgs:
    def __init__(
        self,
        n_dim: int = 128,
        e_dim: int = 64,
        a_dim: int = 64,
        nlayers: int = 6,
        e_rcut: float = 6.0,
        e_rcut_smth: float = 5.0,
        e_sel: int = 120,
        a_rcut: float = 4.0,
        a_rcut_smth: float = 3.5,
        a_sel: int = 20,
        a_compress_rate: int = 0,
        a_compress_e_rate: int = 1,
        a_compress_use_split: bool = False,
        n_multi_edge_message: int = 1,
        axis_neuron: int = 4,
        update_angle: bool = True,
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        skip_stat: bool = False,
        optim_update: bool = True,
        smooth_angle_init: bool = False,
        angle_init_use_sin: bool = False,
        smooth_edge_update: bool = False,
        angle_multi_freq: Optional[str] = None,
        use_dynamic_sel: bool = False,
        sel_reduce_factor: float = 10.0,
        use_env_envelope: bool = False,
        use_new_sw: bool = False,
        update_dihedral: bool = False,
        d_dim: int = 32,
        d_sel: int = 10,
        d_rcut: float = 2.8,
        d_rcut_smth: float = 2.0,
        use_ffn_node_edge_message: bool = False,
        use_ffn_edge_edge_message: bool = False,
        use_ffn_edge_angle_message: bool = False,
        use_ffn_angle_angle_message: bool = False,
        ffn_hidden_dim: int = 1024,
        edge_use_concat_rbf: bool = False,
        edge_use_rbf: bool = False,
        edge_use_dist: bool = False,
        embed_use_bias: bool = True,
        edge_use_attn: bool = False,
        edge_attn_hidden: int = 32,
        edge_attn_head: int = 4,
        edge_attn_use_ln: bool = True,
        edge_rbf_dot_self: bool = False,
        edge_rbf_dot_message: bool = False,
        edge_use_esen_rbf: bool = False,
        edge_use_esen_atom_ebd: bool = False,
        edge_use_esen_env: bool = False,
        residual_pref: list = [],
        tebd_use_act: bool = True,
    ) -> None:
        r"""The constructor for the RepFlowArgs class which defines the parameters of the repflow block in DPA3 descriptor.

        Parameters
        ----------
        n_dim : int, optional
            The dimension of node representation.
        e_dim : int, optional
            The dimension of edge representation.
        a_dim : int, optional
            The dimension of angle representation.
        nlayers : int, optional
            Number of repflow layers.
        e_rcut : float, optional
            The edge cut-off radius.
        e_rcut_smth : float, optional
            Where to start smoothing for edge. For example the 1/r term is smoothed from rcut to rcut_smth.
        e_sel : int, optional
            Maximally possible number of selected edge neighbors.
        a_rcut : float, optional
            The angle cut-off radius.
        a_rcut_smth : float, optional
            Where to start smoothing for angle. For example the 1/r term is smoothed from rcut to rcut_smth.
        a_sel : int, optional
            Maximally possible number of selected angle neighbors.
        a_compress_rate : int, optional
            The compression rate for angular messages. The default value is 0, indicating no compression.
            If a non-zero integer c is provided, the node and edge dimensions will be compressed
            to a_dim/c and a_dim/2c, respectively, within the angular message.
        a_compress_e_rate : int, optional
            The extra compression rate for edge in angular message compression. The default value is 1.
            When using angular message compression with a_compress_rate c and a_compress_e_rate c_e,
            the edge dimension will be compressed to (c_e * a_dim / 2c) within the angular message.
        a_compress_use_split : bool, optional
            Whether to split first sub-vectors instead of linear mapping during angular message compression.
            The default value is False.
        n_multi_edge_message : int, optional
            The head number of multiple edge messages to update node feature.
            Default is 1, indicating one head edge message.
        axis_neuron : int, optional
            The number of dimension of submatrix in the symmetrization ops.
        update_angle : bool, optional
            Where to update the angle rep. If not, only node and edge rep will be used.
        update_style : str, optional
            Style to update a representation.
            Supported options are:
            -'res_avg': Updates a rep `u` with: u = 1/\\sqrt{n+1} (u + u_1 + u_2 + ... + u_n)
            -'res_incr': Updates a rep `u` with: u = u + 1/\\sqrt{n} (u_1 + u_2 + ... + u_n)
            -'res_residual': Updates a rep `u` with: u = u + (r1*u_1 + r2*u_2 + ... + r3*u_n)
            where `r1`, `r2` ... `r3` are residual weights defined by `update_residual`
            and `update_residual_init`.
        update_residual : float, optional
            When update using residual mode, the initial std of residual vector weights.
        update_residual_init : str, optional
            When update using residual mode, the initialization mode of residual vector weights.
        """
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.nlayers = nlayers
        self.e_rcut = e_rcut
        self.e_rcut_smth = e_rcut_smth
        self.e_sel = e_sel
        self.a_rcut = a_rcut
        self.a_rcut_smth = a_rcut_smth
        self.a_sel = a_sel
        self.a_compress_rate = a_compress_rate
        self.n_multi_edge_message = n_multi_edge_message
        self.axis_neuron = axis_neuron
        self.update_angle = update_angle
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.skip_stat = skip_stat
        self.a_compress_e_rate = a_compress_e_rate
        self.a_compress_use_split = a_compress_use_split
        self.optim_update = optim_update
        self.smooth_angle_init = smooth_angle_init
        self.angle_init_use_sin = angle_init_use_sin
        self.smooth_edge_update = smooth_edge_update
        self.angle_multi_freq = angle_multi_freq
        self.use_dynamic_sel = use_dynamic_sel
        self.sel_reduce_factor = sel_reduce_factor
        self.use_env_envelope = use_env_envelope
        self.use_new_sw = use_new_sw
        self.update_dihedral = update_dihedral
        self.d_dim = d_dim
        self.d_sel = d_sel
        self.d_rcut = d_rcut
        self.d_rcut_smth = d_rcut_smth
        self.use_ffn_node_edge_message = use_ffn_node_edge_message
        self.use_ffn_edge_edge_message = use_ffn_edge_edge_message
        self.use_ffn_edge_angle_message = use_ffn_edge_angle_message
        self.use_ffn_angle_angle_message = use_ffn_angle_angle_message
        self.ffn_hidden_dim = ffn_hidden_dim
        self.edge_use_concat_rbf = edge_use_concat_rbf
        self.edge_use_rbf = edge_use_rbf
        self.edge_use_dist = edge_use_dist
        self.embed_use_bias = embed_use_bias
        self.edge_use_attn = edge_use_attn
        self.edge_attn_hidden = edge_attn_hidden
        self.edge_attn_head = edge_attn_head
        self.edge_attn_use_ln = edge_attn_use_ln
        self.edge_rbf_dot_self = edge_rbf_dot_self
        self.edge_rbf_dot_message = edge_rbf_dot_message
        self.edge_use_esen_rbf = edge_use_esen_rbf
        self.edge_use_esen_atom_ebd = edge_use_esen_atom_ebd
        self.edge_use_esen_env = edge_use_esen_env
        self.residual_pref = residual_pref
        self.tebd_use_act = tebd_use_act

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(key)

    def serialize(self) -> dict:
        return {
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "a_dim": self.a_dim,
            "nlayers": self.nlayers,
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "a_rcut": self.a_rcut,
            "a_rcut_smth": self.a_rcut_smth,
            "a_sel": self.a_sel,
            "a_compress_rate": self.a_compress_rate,
            "a_compress_e_rate": self.a_compress_e_rate,
            "a_compress_use_split": self.a_compress_use_split,
            "n_multi_edge_message": self.n_multi_edge_message,
            "axis_neuron": self.axis_neuron,
            "update_angle": self.update_angle,
            "update_style": self.update_style,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "RepFlowArgs":
        return cls(**data)
