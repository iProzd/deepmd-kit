# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM: Smooth equivariant Zone-bridging Model."""

from __future__ import (
    annotations,
)

import os
from contextlib import (
    contextmanager,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
from einops import (
    rearrange,
)
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from jaxtyping import Float, Int
    from torch import Tensor

from deepmd.pt.model.atomic_model.sezm_atomic_model import (
    SeZMAtomicModel,
)
from deepmd.pt.model.descriptor.sezm_nn import (
    nvtx_range,
)
from deepmd.pt.model.model.dp_model import (
    DPModelCommon,
)
from deepmd.pt.model.model.make_model import (
    make_model,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.model.transform_output import (
    fit_output_to_model_output,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

SeZMModel_ = make_model(SeZMAtomicModel)

# Keep compile-time autotune candidate dumps out of logs by default.
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_REPORT_CHOICES_STATS", "0")
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")


def _parse_optional_env_bool(var_name: str) -> bool | None:
    """
    Parse an optional boolean environment variable.

    Parameters
    ----------
    var_name
        Environment variable name.

    Returns
    -------
    bool | None
        Parsed boolean value, or ``None`` when the variable is unset.

    Raises
    ------
    ValueError
        If the environment variable value is not a supported boolean token.
    """
    raw_value = os.environ.get(var_name)
    if raw_value is None:
        return None
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{var_name} must be one of 1/0/true/false/yes/no/on/off, got {raw_value!r}"
    )


def _strip_saved_tensor_detach(gm: torch.fx.GraphModule) -> None:
    """Strip ``aten.detach`` nodes that ``make_fx`` inserts for saved tensors.

    When ``make_fx`` decomposes ``autograd.grad(..., create_graph=True)``,
    the autograd engine wraps every saved forward activation in a double-detach
    chain (e.g. ``tanh -> detach_A -> detach_B -> tanh_backward``).  These
    detach nodes block the second-order gradient path from the loss back to
    model parameters, causing incorrect parameter updates during force-loss
    training.

    User-explicit ``.detach()`` calls (e.g. inside ``attach_edge_vec_grad``)
    are preserved.  The two categories are distinguished by graph topology
    alone — no hard-coded op names — using three rules:

    * *Chain inner*: input is another detach node.
    * *Dead node*: no downstream users.
    * *Chain head*: *all* users are detach nodes.

    Any detach that does **not** match these rules is treated as user-explicit
    and left untouched.
    """
    _DETACH = torch.ops.aten.detach.default

    def _is_detach(n: torch.fx.Node) -> bool:
        return n.op == "call_function" and n.target == _DETACH

    # Pass 1 — classify on the original, unmodified graph.
    to_remove: list[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if not _is_detach(node):
            continue
        input_node = node.args[0]
        users = list(node.users.keys())
        is_chain_inner = _is_detach(input_node)
        is_dead = len(users) == 0
        is_chain_head = len(users) > 0 and all(_is_detach(u) for u in users)
        if is_chain_inner or is_dead or is_chain_head:
            to_remove.append(node)

    # Pass 2 — remove in one shot so classification stays consistent.
    for node in to_remove:
        node.replace_all_uses_with(node.args[0])
        gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()


@BaseModel.register("SeZM")
@BaseModel.register("se_zm")
@BaseModel.register("sezm")
@BaseModel.register("SeZM-Net")
class SeZMModel(DPModelCommon, SeZMModel_):
    """
    SeZM energy model with an optional compiled sparse-edge path.

    By default it uses the traditional DeePMD neighbor list path with ghost atoms
    and padded neighbor matrix, compatible with LAMMPS and other MD engines.
    When `use_compile=True`, it builds a compact sparse edge list from the
    standard neighbor list and traces the local graph with ``make_fx`` for
    higher-order force training. Evaluation/inference compile usage is
    controlled by the `DP_COMPILE_INFER` environment variable read at model
    initialization time.
    """

    model_type = "SeZM"

    def __init__(
        self,
        *args: Any,
        use_compile: bool = False,
        enable_tf32: bool = False,
        bridging_method: str = "none",
        bridging_r_inner: float = 0.9,
        bridging_r_outer: float = 1.3,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        SeZMModel_.__init__(self, *args, **kwargs)
        self.redu_prec = env.GLOBAL_PT_ENER_FLOAT_PRECISION
        self.use_compile = bool(use_compile)
        self.enable_tf32 = bool(enable_tf32)
        self._dens_compiled = False
        self._core_compute_compiled_train: bool | None = None
        self._core_compute_compiled_atomic_virial: bool | None = None
        # Store compiled callables outside the nn.Module tree so that
        # FSDP2 / DDP do not shard or sync its duplicated parameters.
        object.__setattr__(self, "compiled_core_compute", None)
        object.__setattr__(self, "compiled_dens_compute", None)
        # Training follows `use_compile`. Evaluation/inference reads
        # `DP_COMPILE_INFER` at init time and falls back to eager when unset.
        self._env_use_compile_infer: bool | None = _parse_optional_env_bool(
            "DP_COMPILE_INFER"
        )

        # === Bridging (optional short-range zone bridging) ===
        self.bridging_method: str = str(bridging_method).upper()
        self.bridging_r_inner = float(bridging_r_inner)
        self.bridging_r_outer = float(bridging_r_outer)
        self.inter_potential: InterPotential | None = (
            InterPotential(type_map=self.get_type_map(), mode=self.bridging_method)
            if self.bridging_method != "NONE"
            else None
        )

    # =========================================================================
    # Forward Methods
    # =========================================================================

    def forward(
        self,
        coord: Float[Tensor, "nf nloc 3"] | Float[Tensor, "nf nloc_x3"],
        atype: Int[Tensor, "nf nloc"],
        box: Float[Tensor, "nf 9"] | None = None,
        fparam: Float[Tensor, "nf ndf"] | None = None,
        aparam: Float[Tensor, "nf nloc nda"] | None = None,
        do_atomic_virial: bool = False,
        force_input: Float[Tensor, "nf nloc 3"] | None = None,
        noise_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass using standard neighbor list.

        Parameters
        ----------
        coord
            Coordinates with shape (nf, nloc*3) or (nf, nloc, 3) in Å.
        atype
            Atom types with shape (nf, nloc).
        box
            Box tensor with shape (nf, 9) in Å, or None.
        fparam
            Frame parameters with shape (nf, ndf) or None.
        aparam
            Atomic parameters with shape (nf, nloc, nda) or None.
        do_atomic_virial
            Whether to compute atomic virial.
        force_input
            Optional atom-wise force input tensor with shape `(nf, nloc, 3)`.
            It stays optional at the public model boundary because validation /
            inference and clean `dens` batches may not provide force labels.
        noise_mask
            Optional corruption mask with shape `(nf, nloc)`. It stays optional
            at the public model boundary because validation / inference and
            clean `dens` batches may not provide corruption masks.

        Returns
        -------
        dict[str, torch.Tensor]
            Model predictions including atom_energy, energy, force, virial,
            atom_virial, and mask.
        """
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            force_input=force_input,
            noise_mask=noise_mask,
        )
        if self.get_fitting_net() is not None:
            model_predict: dict[str, torch.Tensor] = {}

            # === Step 1. Energy ===
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]

            # === Step 2. Force (independent branch) ===
            if self.do_grad_r("energy"):
                model_predict["force"] = rearrange(
                    model_ret["energy_derv_r"],
                    "nf nloc 1 three -> nf nloc three",
                    three=3,
                )
            else:
                model_predict["force"] = model_ret["dforce"]

            if self.get_active_mode() == "dens":
                if "energy_norm" in model_ret:
                    model_predict["energy_norm"] = model_ret["energy_norm"]
                if "atom_energy_norm" in model_ret:
                    model_predict["atom_energy_norm"] = model_ret["atom_energy_norm"]
                if "dforce_norm" in model_ret:
                    model_predict["force_norm"] = model_ret["dforce_norm"]
                if "clean_dforce_norm" in model_ret:
                    model_predict["clean_force_norm"] = model_ret["clean_dforce_norm"]
                if "denoising_dforce_norm" in model_ret:
                    model_predict["denoising_force_norm"] = model_ret[
                        "denoising_dforce_norm"
                    ]

            # === Step 3. Virial ===
            if self.do_grad_c("energy"):
                model_predict["virial"] = rearrange(
                    model_ret["energy_derv_c_redu"], "nf 1 nine -> nf nine", nine=9
                )
                if do_atomic_virial:
                    model_predict["atom_virial"] = rearrange(
                        model_ret["energy_derv_c"],
                        "nf nloc 1 nine -> nf nloc nine",
                        nine=9,
                    )

            # === Step 4. Mask ===
            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]

        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord
        return model_predict

    def forward_common(
        self,
        coord: Float[Tensor, "nf nloc 3"] | Float[Tensor, "nf nloc_x3"],
        atype: Int[Tensor, "nf nloc"],
        box: Float[Tensor, "nf 9"] | None = None,
        fparam: Float[Tensor, "nf ndf"] | None = None,
        aparam: Float[Tensor, "nf nloc nda"] | None = None,
        do_atomic_virial: bool = False,
        force_input: Float[Tensor, "nf nloc 3"] | None = None,
        noise_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Return model prediction using standard neighbor list.

        Parameters
        ----------
        coord
            Coordinates with shape (nf, nloc*3) or (nf, nloc, 3) in Å.
        atype
            Atom types with shape (nf, nloc).
        box
            Box tensor with shape (nf, 9) in Å, or None.
        fparam
            Frame parameters with shape (nf, ndf) or None.
        aparam
            Atomic parameters with shape (nf, nloc, nda) or None.
        do_atomic_virial
            Whether to compute atomic virial.
        force_input
            Optional atom-wise force input tensor with shape `(nf, nloc, 3)`.
            It stays optional at the public model boundary because validation /
            inference and clean `dens` batches may not provide force labels.
        noise_mask
            Optional corruption mask with shape `(nf, nloc)`. It stays optional
            at the public model boundary because validation / inference and
            clean `dens` batches may not provide corruption masks.

        Returns
        -------
        dict[str, torch.Tensor]
            Model predictions including energy, forces, etc.
        """
        with nvtx_range("SeZM/forward_common"):
            # === Step 1. Cast inputs to correct dtype ===
            with nvtx_range("SeZM/input_type_cast"):
                cc, bb, fp, ap, input_prec = self._input_type_cast(
                    coord, box=box, fparam=fparam, aparam=aparam
                )
                del coord, box, fparam, aparam
                nf, nloc = atype.shape[:2]
                if cc.ndim == 2:
                    cc = cc.view(nf, nloc, 3)

            # === Step 2. Build neighbor list ===
            with nvtx_range("SeZM/build_neighbor_list"):
                # extended_coord: (nf, nall, 3), extended_atype: (nf, nall)
                # mapping: (nf, nall), nlist: (nf, nloc, nsel)
                extended_coord, extended_atype, mapping, nlist = (
                    self.build_neighbor_list(cc, atype, bb)
                )

            # === Step 3. Format nlist and build sparse edges ===
            with nvtx_range("SeZM/prepare_sparse_edges"):
                nlist = self.format_nlist(
                    extended_coord,
                    extended_atype,
                    nlist,
                    extra_nlist_sort=self.need_sorted_nlist_for_lower(),
                )
                edge_index, edge_vec, edge_mask = self.build_edge_list_from_nlist(
                    extended_coord=extended_coord,
                    nlist=nlist,
                    mapping=mapping,
                )

            active_mode = self.get_active_mode()
            if active_mode == "dens":
                force_input, noise_mask = self.canonicalize_dens_inputs(
                    force_input,
                    noise_mask,
                    nf=nf,
                    nloc=nloc,
                    dtype=cc.dtype,
                    device=cc.device,
                )
                cc = cc.detach()

                # === Step 4. `dens` eager / compile dispatch ===
                if self.should_use_compile():
                    fp, ap = self.convert_fp_ap(
                        fp,
                        ap,
                        nf=nf,
                        nloc=nloc,
                        dtype=cc.dtype,
                        device=cc.device,
                    )
                    with self.tf32_precision_ctx():
                        if (
                            self.compiled_dens_compute is None
                            or not self._dens_compiled
                        ):
                            self.compile_dens()
                        with nvtx_range("SeZM/core_compute_dens"):
                            compute_ret = self.compiled_dens_compute(
                                cc,
                                atype,
                                edge_index,
                                edge_vec,
                                edge_mask,
                                force_input=force_input,
                                noise_mask=noise_mask,
                                fparam=fp,
                                aparam=ap,
                            )
                else:
                    with nvtx_range("SeZM/core_compute_dens"):
                        compute_ret = self.core_compute_dens(
                            cc,
                            atype,
                            edge_index,
                            edge_vec,
                            edge_mask,
                            force_input=force_input,
                            noise_mask=noise_mask,
                            fparam=fp,
                            aparam=ap,
                        )
                # === Step 5. Post-process local outputs ===
                with nvtx_range("SeZM/post_process"):
                    model_predict = self.post_process_output_dens(
                        compute_ret,
                        atype,
                        noise_mask=noise_mask,
                    )
            else:
                need_coord_grad = self.do_grad_r() or self.do_grad_c()
                if need_coord_grad:
                    cc = cc.detach().requires_grad_(True)
                else:
                    cc = cc.detach()

                # === Step 4. `ener` eager / compile dispatch ===
                if self.should_use_compile():
                    fp, ap = self.convert_fp_ap(
                        fp,
                        ap,
                        nf=nf,
                        nloc=nloc,
                        dtype=cc.dtype,
                        device=cc.device,
                    )
                    with self.tf32_precision_ctx():
                        if (
                            self.compiled_core_compute is None
                            or self._core_compute_compiled_train != self.training
                            or self._core_compute_compiled_atomic_virial
                            != do_atomic_virial
                        ):
                            self.trace_and_compile(
                                cc,
                                atype,
                                edge_index,
                                edge_vec,
                                edge_mask,
                                fp,
                                ap,
                                do_atomic_virial,
                            )
                        with nvtx_range("SeZM/core_compute"):
                            compute_ret = self.compiled_core_compute(
                                cc,
                                atype,
                                edge_index,
                                edge_vec,
                                edge_mask,
                                fp,
                                ap,
                            )
                else:
                    with nvtx_range("SeZM/core_compute"):
                        compute_ret = self.core_compute(
                            cc,
                            atype,
                            edge_index,
                            edge_vec,
                            edge_mask,
                            fparam=fp,
                            aparam=ap,
                        )

                # === Step 5. Post-process local outputs ===
                with nvtx_range("SeZM/post_process"):
                    model_predict = self.post_process_output(
                        compute_ret,
                        atype,
                        do_atomic_virial=do_atomic_virial,
                    )

            # === Step 6. Type cast output ===
            with nvtx_range("SeZM/output_type_cast"):
                model_predict = self._output_type_cast(model_predict, input_prec)
                return model_predict

    def core_compute(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute SeZM energy/force/virial tensors from sparse edges.

        Parameters
        ----------
        coord
            Local coordinates with shape (nf, nloc, 3) in Å.
        atype
            Local atom types with shape (nf, nloc).
        edge_index
            Edge source/destination indices with shape (2, E).
        edge_vec
            Edge vectors with shape (E, 3) in Å.
        edge_mask
            Edge validity mask with shape (E,).
        fparam
            Frame parameters with shape (nf, ndf), or None.
        aparam
            Atomic parameters with shape (nf, nloc, nda), or None.

        Returns
        -------
        torch.Tensor
            Concatenated local tensor with shape (nf, nloc, 13):
            `[atom_energy | minus_force | atom_virial]`.
        """
        # === Step 1. Setup dimensions ===
        nf, nloc = atype.shape[:2]
        n_node = nf * nloc
        coord_flat = coord.reshape(n_node, 3)

        # === Step 2. Attach coordinate gradients to edge vectors ===
        edge_vec = self.attach_edge_vec_grad(coord_flat, edge_index, edge_vec)

        # === Step 3. Descriptor forward through the sparse-edge entry ===
        with nvtx_range("SeZM/descriptor"):
            descriptor_model = self.atomic_model.descriptor
            descriptor, _ = descriptor_model.forward_with_edges(
                extended_coord=coord,
                extended_atype=atype,
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_mask=edge_mask,
            )
        if self.atomic_model.enable_eval_descriptor_hook:
            self.atomic_model.eval_descriptor_list.append(descriptor.detach())

        # === Step 4. Fitting net + output statistics ===
        with nvtx_range("SeZM/fitting_net"):
            fit_ret = self.atomic_model.fitting_net(
                descriptor,
                atype,
                fparam=fparam,
                aparam=aparam,
            )
        if self.atomic_model.enable_eval_fitting_last_layer_hook:
            assert "middle_output" in fit_ret, (
                "eval_fitting_last_layer not supported for this fitting net!"
            )
            self.atomic_model.eval_fitting_last_layer_list.append(
                fit_ret.pop("middle_output").detach()
            )
        with nvtx_range("SeZM/apply_out_stat"):
            fit_ret = self.atomic_model.apply_out_stat(fit_ret, atype)

        # === Step 5. Apply the local atom mask ===
        atom_mask = self.atomic_model.make_atom_mask(atype).to(torch.int32)
        if self.atomic_model.atom_excl is not None:
            atom_mask *= self.atomic_model.atom_excl(atype)
        for key in fit_ret.keys():
            out_shape = fit_ret[key].shape
            flat_dim = 1
            for axis_size in out_shape[2:]:
                flat_dim *= axis_size
            fit_ret[key] = (
                fit_ret[key].reshape([out_shape[0], out_shape[1], flat_dim])
                * atom_mask[:, :, None]
            ).view(out_shape)

        atom_energy = fit_ret["energy"]
        if self.inter_potential is not None:
            atom_energy = atom_energy + self.inter_potential.forward_from_edges(
                edge_vec,
                edge_index,
                atype.reshape(-1),
                edge_mask,
                n_node,
            ).view(nf, nloc, 1)

        # === Step 6. Differentiate energy for force / virial ===
        energy_sum = torch.sum(atom_energy.to(self.redu_prec))
        do_grad_r = self.do_grad_r("energy")
        do_grad_c = self.do_grad_c("energy")
        create_graph = self.training

        minus_force = torch.zeros_like(coord)
        atom_virial_flat = torch.zeros(
            n_node,
            9,
            dtype=atom_energy.dtype,
            device=atom_energy.device,
        )

        if do_grad_r and do_grad_c:
            minus_force, edge_grad = torch.autograd.grad(
                outputs=[energy_sum],
                inputs=[coord, edge_vec],
                create_graph=create_graph,
                retain_graph=True,
            )
            edge_keep = edge_mask.to(dtype=edge_grad.dtype).unsqueeze(-1)
            edge_grad = edge_grad * edge_keep
            edge_vec_val = edge_vec.detach()
            edge_virial = -torch.einsum("Ei,Ej->Eij", edge_grad, edge_vec_val)
            edge_virial = edge_virial * edge_keep.unsqueeze(-1)
            edge_virial_flat = edge_virial.view(-1, 9)
            dst = edge_index[1].to(dtype=torch.long)
            atom_virial_flat = torch.zeros(
                n_node,
                9,
                dtype=edge_virial_flat.dtype,
                device=edge_virial_flat.device,
            ).index_add(0, dst, edge_virial_flat)
        elif do_grad_r:
            (minus_force,) = torch.autograd.grad(
                outputs=[energy_sum],
                inputs=[coord],
                create_graph=create_graph,
                retain_graph=True,
            )
        elif do_grad_c:
            (edge_grad,) = torch.autograd.grad(
                outputs=[energy_sum],
                inputs=[edge_vec],
                create_graph=create_graph,
                retain_graph=True,
            )
            edge_keep = edge_mask.to(dtype=edge_grad.dtype).unsqueeze(-1)
            edge_grad = edge_grad * edge_keep
            edge_vec_val = edge_vec.detach()
            edge_virial = -torch.einsum("Ei,Ej->Eij", edge_grad, edge_vec_val)
            edge_virial = edge_virial * edge_keep.unsqueeze(-1)
            edge_virial_flat = edge_virial.view(-1, 9)
            dst = edge_index[1].to(dtype=torch.long)
            atom_virial_flat = torch.zeros(
                n_node,
                9,
                dtype=edge_virial_flat.dtype,
                device=edge_virial_flat.device,
            ).index_add(0, dst, edge_virial_flat)

        # === Step 7. Concatenate local outputs ===
        return torch.cat(
            [
                atom_energy,
                minus_force,
                atom_virial_flat.view(nf, nloc, 9),
            ],
            dim=-1,
        )

    def core_compute_dens(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        *,
        force_input: torch.Tensor,
        noise_mask: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute SeZM `dens` energy/direct-force tensors from sparse edges.

        Parameters
        ----------
        coord
            Local coordinates with shape `(nf, nloc, 3)` in Angstroms.
        atype
            Local atom types with shape `(nf, nloc)`.
        edge_index
            Edge source/destination indices with shape `(2, E)`.
        edge_vec
            Edge vectors with shape `(E, 3)` in Angstroms.
        edge_mask
            Edge validity mask with shape `(E,)`.
        force_input
            Atom-wise force input tensor with shape `(nf, nloc, 3)`.
        noise_mask
            Atom-wise corruption mask with shape `(nf, nloc)`.
        fparam
            Frame parameters with shape `(nf, ndf)`, or None.
        aparam
            Atomic parameters with shape `(nf, nloc, nda)`, or None.

        Returns
        -------
        torch.Tensor
            Concatenated local tensor with shape `(nf, nloc, 7)` and layout
            `[atom_energy_norm | clean_dforce_norm | denoising_dforce_norm]`.
        """
        if self.inter_potential is not None:
            raise NotImplementedError(
                "SeZM `dens` path does not support analytical bridging potentials."
            )

        nf, nloc = atype.shape[:2]
        descriptor_model = self.atomic_model.descriptor
        dens_fitting = self.atomic_model.get_dens_fitting_net()
        force_embedding = dens_fitting.build_force_embedding(
            force_input,
            noise_mask=noise_mask,
        )

        with nvtx_range("SeZM/descriptor_dens"):
            descriptor, latent = descriptor_model.forward_with_edges(
                extended_coord=coord,
                extended_atype=atype,
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_mask=edge_mask,
                force_embedding=force_embedding,
            )
        if self.atomic_model.enable_eval_descriptor_hook:
            self.atomic_model.eval_descriptor_list.append(descriptor.detach())

        with nvtx_range("SeZM/dens_fitting_net"):
            fit_ret = dens_fitting(
                descriptor,
                latent,
                atype,
                noise_mask=noise_mask,
                fparam=fparam,
                aparam=aparam,
                return_components=True,
            )
        if self.atomic_model.enable_eval_fitting_last_layer_hook:
            assert "middle_output" in fit_ret, (
                "eval_fitting_last_layer not supported for this fitting net!"
            )
            self.atomic_model.eval_fitting_last_layer_list.append(
                fit_ret.pop("middle_output").detach()
            )
        return torch.cat(
            [
                fit_ret["energy"],
                fit_ret["clean_dforce"],
                fit_ret["denoising_dforce"],
            ],
            dim=-1,
        )

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: Float[Tensor, "nf nall_x3"] | Float[Tensor, "nf nall 3"],
        extended_atype: Int[Tensor, "nf nall"],
        nlist: Int[Tensor, "nf nloc nsel"],
        mapping: Int[Tensor, "nf nall"] | None = None,
        fparam: Float[Tensor, "nf ndf"] | None = None,
        aparam: Float[Tensor, "nf nall nda"] | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Lower-level public forward using the DeePMD lower-interface contract.

        Parameters
        ----------
        extended_coord
            Extended coordinates with shape (nf, nall*3) or (nf, nall, 3) in Å.
        extended_atype
            Extended atom types with shape (nf, nall).
        nlist
            Neighbor list with shape (nf, nloc, nsel).
        mapping
            Mapping indices with shape (nf, nall), or None.
        fparam
            Frame parameters with shape (nf, ndf) or None.
        aparam
            Atomic parameters with shape (nf, nall, nda) or None.
        do_atomic_virial
            Whether to compute atomic virial.
        comm_dict
            Communication dict forwarded to `forward_common_lower()`.

        Returns
        -------
        dict[str, torch.Tensor]
            Lower-interface outputs.
            When a fitting net is present, this always includes:
            - `atom_energy`: atomic energy on local atoms with shape (nf, nloc, 1)
            - `energy`: reduced energy with shape (nf, 1)
            It additionally includes:
            - `extended_force`: force on extended coordinates with shape (nf, nall, 3)
              when `self.do_grad_r("energy")` is true
            - `dforce`: fitting-net direct force output when energy is not coordinate differentiable
            - `virial`: reduced virial with shape (nf, 9) when `self.do_grad_c("energy")` is true
            - `extended_virial`: per-extended-atom virial with shape (nf, nall, 9)
              only when both `self.do_grad_c("energy")` and `do_atomic_virial` are true
            If no fitting net is present, the raw result of `forward_common_lower()` is returned.
        """
        if self.get_active_mode() == "dens":
            raise NotImplementedError(
                "SeZM `forward_lower` only supports the conservative `ener` mode."
            )
        cc_ext, _, fp, ap, input_prec = self._input_type_cast(
            extended_coord, fparam=fparam, aparam=aparam
        )
        model_ret = self.forward_common_lower(
            cc_ext,
            extended_atype,
            nlist,
            mapping,
            fparam=fp,
            aparam=ap,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
        )
        model_ret = self._output_type_cast(model_ret, input_prec)
        if self.get_fitting_net() is not None:
            model_predict: dict[str, torch.Tensor] = {}

            # === Step 1. Energy ===
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]

            # === Step 2. Force (independent branch) ===
            if self.do_grad_r("energy"):
                model_predict["extended_force"] = rearrange(
                    model_ret["energy_derv_r"],
                    "nf nall 1 three -> nf nall three",
                    three=3,
                )
            else:
                assert model_ret["dforce"] is not None
                model_predict["dforce"] = model_ret["dforce"]

            # === Step 3. Virial ===
            if self.do_grad_c("energy"):
                model_predict["virial"] = rearrange(
                    model_ret["energy_derv_c_redu"], "nf 1 nine -> nf nine", nine=9
                )
                if do_atomic_virial:
                    model_predict["extended_virial"] = rearrange(
                        model_ret["energy_derv_c"],
                        "nf nall 1 nine -> nf nall nine",
                        nine=9,
                    )
        else:
            model_predict = model_ret
        return model_predict

    def forward_common_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        extra_nlist_sort: bool = False,
        extended_coord_corr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Common lower forward on extended inputs before `forward_lower()` renames outputs.

        Parameters
        ----------
        extended_coord
            Extended coordinates with shape (nf, nall*3) or (nf, nall, 3) in Å.
        extended_atype
            Extended atom types with shape (nf, nall).
        nlist
            Neighbor list with shape (nf, nloc, nsel).
        mapping
            Maps extended indices to local indices with shape (nf, nall), or None.
        fparam
            Frame parameters with shape (nf, ndf), or None.
        aparam
            Atomic parameters with shape (nf, nall, nda), or None.
        do_atomic_virial
            Whether to compute atomic virial.
        comm_dict
            Communication data for parallel inference. Currently unused in SeZM.
        extra_nlist_sort
            Whether to forcibly sort the nlist.
        extended_coord_corr
            Coordinates correction for virial with shape (nf, nall*3) or None.

        Returns
        -------
        dict[str, torch.Tensor]
            DeePMD lower-style outputs before the `forward_lower()` key translation.
            For the energy task this includes:
            - `energy`: atomic energy on local atoms with shape (nf, nloc, 1)
            - `energy_redu`: reduced energy with shape (nf, 1)
            - `energy_derv_r`: derivative with respect to extended coordinates with shape
              (nf, nall, 1, 3) when enabled
            - `energy_derv_c`: per-extended-atom virial contribution with shape
              (nf, nall, 1, 9) when enabled and `do_atomic_virial=True`
            - `energy_derv_c_redu`: reduced virial with shape (nf, 1, 9) when enabled
            - `mask`: local-atom mask with shape (nf, nloc)
        """
        cc_ext, _, fp, ap, input_prec = self._input_type_cast(
            extended_coord, fparam=fparam, aparam=aparam
        )
        cc_ext = cc_ext.reshape(extended_atype.shape[0], -1, 3)
        if extended_coord_corr is not None and extended_coord_corr.ndim == 2:
            extended_coord_corr = extended_coord_corr.reshape(
                extended_atype.shape[0], -1, 3
            )
        del comm_dict
        nlist = self.format_nlist(
            cc_ext,
            extended_atype,
            nlist,
            extra_nlist_sort=extra_nlist_sort,
        )
        _, nloc, _ = nlist.shape
        atype = extended_atype[:, :nloc]
        descriptor_model = self.atomic_model.descriptor

        # === Step 1. Enable coordinate gradients on the extended coordinates ===
        if self.do_grad_r() or self.do_grad_c():
            cc_ext.requires_grad_(True)

        # === Step 2. Build compact sparse edges from the DeePMD nlist ===
        edge_index, edge_vec, edge_mask = self.build_edge_list_from_nlist(
            extended_coord=cc_ext,
            nlist=nlist,
            mapping=mapping,
        )

        # === Step 3. Descriptor forward through the sparse-edge entry ===
        descriptor, _ = descriptor_model.forward_with_edges(
            extended_coord=cc_ext[:, :nloc, :],
            extended_atype=atype,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
        )
        if self.atomic_model.enable_eval_descriptor_hook:
            self.atomic_model.eval_descriptor_list.append(descriptor.detach())

        # === Step 4. Fitting net + output statistics ===
        fit_ret = self.atomic_model.fitting_net(
            descriptor,
            atype,
            fparam=fparam,
            aparam=aparam,
        )
        if self.atomic_model.enable_eval_fitting_last_layer_hook:
            assert "middle_output" in fit_ret, (
                "eval_fitting_last_layer not supported for this fitting net!"
            )
            self.atomic_model.eval_fitting_last_layer_list.append(
                fit_ret.pop("middle_output").detach()
            )
        fit_ret = self.atomic_model.apply_out_stat(fit_ret, atype)

        # === Step 5. Apply the same atom masking contract as BaseAtomicModel ===
        ext_atom_mask = self.atomic_model.make_atom_mask(extended_atype)
        atom_mask = ext_atom_mask[:, :nloc].to(torch.int32)
        if self.atomic_model.atom_excl is not None:
            atom_mask *= self.atomic_model.atom_excl(atype)
        for key in fit_ret.keys():
            out_shape = fit_ret[key].shape
            flat_dim = 1
            for axis_size in out_shape[2:]:
                flat_dim *= axis_size
            fit_ret[key] = (
                fit_ret[key].reshape([out_shape[0], out_shape[1], flat_dim])
                * atom_mask[:, :, None]
            ).view(out_shape)
        fit_ret["mask"] = atom_mask

        if self.inter_potential is not None:
            fit_ret["energy"] = fit_ret["energy"] + self.inter_potential(
                cc_ext,
                extended_atype,
                nlist,
                nloc,
            )

        model_predict = fit_output_to_model_output(
            fit_ret,
            self.atomic_output_def(),
            cc_ext,
            do_atomic_virial=do_atomic_virial,
            create_graph=self.training,
            mask=fit_ret["mask"] if "mask" in fit_ret else None,
            extended_coord_corr=extended_coord_corr,
        )
        return self._output_type_cast(model_predict, input_prec)

    # =========================================================================
    # Compile Utilities
    # =========================================================================

    def trace_and_compile(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        fp: torch.Tensor,
        ap: torch.Tensor,
        do_atomic_virial: bool,
    ) -> None:
        """Trace the sparse-edge `core_compute()` and cache the compiled callable."""
        from torch._inductor import config as inductor_config

        inductor_config.max_autotune_report_choices_stats = False
        inductor_config.autotune_num_choices_displayed = 0

        def compute_fn(
            coord: torch.Tensor,
            atype: torch.Tensor,
            edge_index: torch.Tensor,
            edge_vec: torch.Tensor,
            edge_mask: torch.Tensor,
            fp: torch.Tensor,
            ap: torch.Tensor,
        ) -> torch.Tensor:
            return self.core_compute(
                coord,
                atype,
                edge_index,
                edge_vec,
                edge_mask,
                fparam=fp,
                aparam=ap,
            )

        trace_coord = coord
        trace_atype = atype
        trace_edge_index = edge_index
        trace_edge_vec = edge_vec
        trace_edge_mask = edge_mask
        trace_fp = fp
        trace_ap = ap
        # Avoid specializing the traced batch dimension to the concrete value 1,
        # although it is not necessary for the current implementation,
        # it is a good practice to avoid specializing the traced batch dimension to 1.
        if coord.shape[0] == 1:
            trace_coord = torch.cat([trace_coord, trace_coord], dim=0)
            trace_atype = torch.cat([trace_atype, trace_atype], dim=0)
            trace_edge_index = torch.cat(
                [trace_edge_index, trace_edge_index + coord.shape[1]],
                dim=1,
            )
            trace_edge_vec = torch.cat([trace_edge_vec, trace_edge_vec], dim=0)
            trace_edge_mask = torch.cat([trace_edge_mask, trace_edge_mask], dim=0)
            if trace_fp is not None:
                trace_fp = torch.cat([trace_fp, trace_fp], dim=0)
            if trace_ap is not None:
                trace_ap = torch.cat([trace_ap, trace_ap], dim=0)

        # Decompose silu_backward into primitive ops (sigmoid + mul + ...)
        # so that inductor can compile the eval graph without requiring a
        # higher-order derivative that PyTorch does not register for silu.
        from torch._decomp import (
            get_decompositions,
        )

        decomp_table = get_decompositions([torch.ops.aten.silu_backward.default])

        traced = make_fx(
            compute_fn,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            decomposition_table=decomp_table,
        )(
            trace_coord,
            trace_atype,
            trace_edge_index,
            trace_edge_vec,
            trace_edge_mask,
            trace_fp,
            trace_ap,
        )

        # When training with force/virial loss, make_fx wraps saved forward
        # activations in detach nodes that block second-order gradient flow.
        # In eval mode (create_graph=False) those detach nodes are correct
        # and must be kept — removing them causes inductor to attempt
        # higher-order differentiation of remaining backward ops.
        if self.training:
            _strip_saved_tensor_detach(traced)

        object.__setattr__(
            self,
            "compiled_core_compute",
            torch.compile(
                traced,
                backend="inductor",
                dynamic=True,
                options={
                    "max_autotune": False,
                    "epilogue_fusion": False,
                    "triton.cudagraphs": False,
                    "shape_padding": True,
                    "max_fusion_size": 8,
                },
            ),
        )
        self._core_compute_compiled_train = self.training
        self._core_compute_compiled_atomic_virial = do_atomic_virial

    def compile_dens(self) -> None:
        """Compile the direct-force `dens` path."""
        from torch._inductor import config as inductor_config

        inductor_config.max_autotune_report_choices_stats = False
        inductor_config.autotune_num_choices_displayed = 0

        object.__setattr__(
            self,
            "compiled_dens_compute",
            torch.compile(
                self.core_compute_dens,
                backend="inductor",
                dynamic=True,
                options={
                    "max_autotune": False,
                    "epilogue_fusion": False,
                    "triton.cudagraphs": False,
                    "shape_padding": True,
                    "max_fusion_size": 8,
                },
            ),
        )
        self._dens_compiled = True

    def attach_edge_vec_grad(
        self,
        coord_flat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attach local-coordinate gradients to edge vectors without changing values.

        Parameters
        ----------
        coord_flat
            Flattened local coordinates with shape (N, 3) in Å.
        edge_index
            Edge source/destination indices with shape (2, E).
        edge_vec
            Edge vectors with shape (E, 3) in Å.

        Returns
        -------
        torch.Tensor
            Edge vectors with attached gradient path and shape (E, 3).
        """
        # === Step 1. Detach the geometric values at the gradient boundary ===
        # This helper treats the incoming edge vectors as pure numeric values,
        # cuts any existing history, and then reattaches the clean
        # local-coordinate gradient path used by force / virial.
        edge_vec = edge_vec.detach()

        # === Step 2. Gather source and destination coordinates ===
        src = edge_index[0].to(dtype=torch.long)
        dst = edge_index[1].to(dtype=torch.long)
        coord_src = coord_flat.index_select(0, src)
        coord_dst = coord_flat.index_select(0, dst)

        # === Step 3. Inject the clean coordinate gradient path ===
        return (
            edge_vec
            + (coord_src - coord_src.detach())
            - (coord_dst - coord_dst.detach())
        )

    def should_use_compile(self) -> bool:
        """Return whether the current forward should use the compile path."""
        if self.training:
            return self.use_compile
        return bool(self._env_use_compile_infer)

    # =========================================================================
    # Neighbor List Construction
    # =========================================================================

    def build_neighbor_list(
        self,
        coord: Float[Tensor, "nf nloc 3"] | Float[Tensor, "nf nloc_x3"],
        atype: Int[Tensor, "nf nloc"],
        box: Float[Tensor, "nf 9"] | None,
    ) -> tuple[
        Float[Tensor, "nf nall 3"],
        Int[Tensor, "nf nall"],
        Int[Tensor, "nf nall"],
        Int[Tensor, "nf nloc nsel"],
    ]:
        """
        Build extended inputs and neighbor list (traditional path).

        Parameters
        ----------
        coord
            Coordinates with shape (nf, nloc, 3) in Å.
        atype
            Atom types with shape (nf, nloc).
        box
            Box tensor with shape (nf, 9) in Å, or None.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Extended coordinates, extended atom types, neighbor list, and mapping.
        """
        return extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.get_rcut(),
            self.get_sel(),
            mixed_types=True,
            box=box,
        )

    def build_edge_list_from_nlist(
        self,
        *,
        extended_coord: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build a compact edge list from DeePMD padded neighbor list.

        Parameters
        ----------
        extended_coord
            Extended coordinates with shape (nf, nall, 3).
        nlist
            DeePMD padded neighbor list with shape (nf, nloc, nsel).
        mapping
            Extended-to-local mapping with shape (nf, nall), or `None`.

        Returns
        -------
        edge_index
            Edge indices with shape (2, E).
        edge_vec
            Edge vectors with shape (E, 3).
        edge_mask
            Boolean mask with shape (E,).
        """
        nf, nloc, nsel = nlist.shape
        n_actual = nf * nloc
        device = extended_coord.device
        nall = extended_coord.shape[1]
        descriptor_model = self.atomic_model.descriptor
        coord_for_diff = extended_coord.to(dtype=descriptor_model.compute_dtype)

        # === Step 1. Build per-edge geometry in descriptor compute dtype ===
        valid_nlist = nlist >= 0
        dst_actual = torch.arange(
            n_actual, device=device, dtype=torch.long
        ).repeat_interleave(nsel)
        f_idx = dst_actual // nloc
        dst_local = dst_actual % nloc
        neighbor_flat = nlist.reshape(-1)
        valid_flat = neighbor_flat >= 0
        neighbor_safe = torch.where(
            valid_flat, neighbor_flat, torch.zeros_like(neighbor_flat)
        )
        coord_flat = coord_for_diff.flatten(0, 1)
        dst_ext = f_idx * nall + dst_local
        src_ext = f_idx * nall + neighbor_safe.to(dtype=torch.long)
        diff = coord_flat.index_select(0, src_ext) - coord_flat.index_select(0, dst_ext)
        edge_len2 = torch.sum(diff * diff, dim=-1)  # (n_actual * nsel,)
        edge_vec_actual = diff

        # === Step 2. Build compact src/dst ===
        if mapping is None:
            src_local = neighbor_safe.to(dtype=torch.long)
        else:
            mapping_flat = mapping.reshape(-1)
            src_local = mapping_flat.index_select(0, f_idx * nall + neighbor_safe)
        src_actual = f_idx * nloc + src_local.to(dtype=torch.long)

        # Filter: valid nlist entry AND src_local in valid range [0, nloc)
        # AND edge length > 0 (exclude true self-edge i==i where len=0).
        # Note: PBC self-image has len > 0 (different positions) and is kept.
        src_local_valid = (src_local >= 0) & (src_local < nloc)
        len_positive = edge_len2 > 1e-10
        edge_mask_actual = valid_flat & src_local_valid & len_positive

        valid_idx = torch.nonzero(edge_mask_actual, as_tuple=False).flatten()

        if valid_idx.numel() == 0:
            edge_index = torch.zeros((2, 1), dtype=torch.long, device=device)
            edge_vec = torch.tensor(
                [[0.0, 0.0, 1.0]], dtype=edge_vec_actual.dtype, device=device
            )
            edge_mask = torch.zeros(1, dtype=torch.bool, device=device)
            return edge_index, edge_vec, edge_mask

        src_sel = src_actual.index_select(0, valid_idx)
        dst_sel = dst_actual.index_select(0, valid_idx)
        edge_vec_sel = edge_vec_actual.index_select(0, valid_idx)
        edge_index = torch.stack([src_sel, dst_sel], dim=0)
        edge_mask = torch.ones(valid_idx.shape[0], dtype=torch.bool, device=device)
        return edge_index, edge_vec_sel, edge_mask

    # =========================================================================
    # Output Definitions
    # =========================================================================

    @torch.jit.export
    def get_observed_type_list(self) -> list[str]:
        """
        Get observed types (elements) of the model during data statistics.

        Returns
        -------
        list[str]
            A list of the observed types in this model.
        """
        type_map = self.get_type_map()
        out_bias = self.atomic_model.get_out_bias()[0]

        assert out_bias is not None, "No out_bias found in the model."
        assert out_bias.dim() == 2, "The supported out_bias should be a 2D tensor."
        assert out_bias.size(0) == len(type_map), (
            "The out_bias shape does not match the type_map length."
        )
        bias_mask = (
            torch.gt(torch.abs(out_bias), 1e-6).any(dim=-1).detach().cpu()
        )  # 1e-6 for stability

        # TorchScript does not support list comprehension with if clause
        result: list[str] = []
        for t, m in zip(type_map, bias_mask.tolist()):
            if m:
                result.append(t)
        return result

    def post_process_output(
        self,
        compute_ret: torch.Tensor,
        atype: torch.Tensor,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Convert the concatenated sparse-edge output to DeePMD model outputs.

        Parameters
        ----------
        compute_ret
            Concatenated tensor with shape (nf, nloc, 13).
        atype
            Local atom types with shape (nf, nloc).
        do_atomic_virial
            Whether to output per-atom virial.

        Returns
        -------
        dict[str, torch.Tensor]
            Standard DeePMD model predictions.
        """
        # === Step 1. Split concatenated tensor ===
        atom_energy = compute_ret[:, :, 0:1]
        minus_force = compute_ret[:, :, 1:4]
        atom_virial = compute_ret[:, :, 4:13]

        # === Step 2. Energy ===
        model_predict: dict[str, torch.Tensor] = {
            "energy": atom_energy,
            "energy_redu": torch.sum(atom_energy.to(self.redu_prec), dim=1),
        }

        # === Step 3. Force ===
        if self.do_grad_r("energy"):
            model_predict["energy_derv_r"] = (-minus_force).unsqueeze(-2)

        # === Step 4. Virial ===
        if self.do_grad_c("energy"):
            if do_atomic_virial:
                model_predict["energy_derv_c"] = atom_virial.unsqueeze(-2)
            model_predict["energy_derv_c_redu"] = torch.sum(
                atom_virial.to(self.redu_prec),
                dim=1,
                keepdim=True,
            )

        # === Step 5. Atom mask ===
        atom_mask = self.atomic_model.make_atom_mask(atype).to(torch.int32)
        if self.atomic_model.atom_excl is not None:
            atom_mask *= self.atomic_model.atom_excl(atype)
        model_predict["mask"] = atom_mask
        return model_predict

    def post_process_output_dens(
        self,
        compute_ret: torch.Tensor,
        atype: torch.Tensor,
        *,
        noise_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Convert the concatenated `dens` output to DeePMD model outputs.

        Parameters
        ----------
        compute_ret
            Concatenated tensor with shape `(nf, nloc, 7)` or `(1, n_node, 7)`.
        atype
            Local atom types with shape `(nf, nloc)`.
        noise_mask
            Corruption mask with shape `(nf, nloc)`.

        Returns
        -------
        dict[str, torch.Tensor]
            Standard DeePMD model predictions for `dens` mode.
        """
        nf, nloc = atype.shape[:2]
        n_actual = nf * nloc
        dens_ret = {
            "energy": compute_ret[:, :n_actual, 0:1].view(nf, nloc, 1),
            "clean_dforce": compute_ret[:, :n_actual, 1:4].view(nf, nloc, 3),
            "denoising_dforce": compute_ret[:, :n_actual, 4:7].view(nf, nloc, 3),
        }
        return self.atomic_model.apply_out_stat_dens(
            dens_ret,
            atype,
            noise_mask=noise_mask,
            energy_redu_dtype=self.redu_prec,
        )

    def convert_fp_ap(
        self,
        fp: torch.Tensor | None,
        ap: torch.Tensor | None,
        nf: int,
        nloc: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert optional fitting inputs to tensor-only compile inputs."""
        dim_fparam = self.get_dim_fparam()
        dim_aparam = self.get_dim_aparam()

        # === Step 1. Canonicalize frame parameters ===
        if fp is None:
            if dim_fparam > 0:
                raise ValueError(
                    "fparam is required because fitting net dim_fparam > 0"
                )
            fp = torch.empty((nf, 0), dtype=dtype, device=device)
        elif fp.shape[-1] != dim_fparam:
            raise ValueError(
                f"fparam last dim ({fp.shape[-1]}) != dim_fparam ({dim_fparam})"
            )

        # === Step 2. Canonicalize atomic parameters ===
        if ap is None:
            if dim_aparam > 0:
                raise ValueError(
                    "aparam is required because fitting net dim_aparam > 0"
                )
            ap = torch.empty((nf, nloc, 0), dtype=dtype, device=device)
        elif ap.shape[-1] != dim_aparam:
            raise ValueError(
                f"aparam last dim ({ap.shape[-1]}) != dim_aparam ({dim_aparam})"
            )

        return fp, ap

    def canonicalize_dens_inputs(
        self,
        force_input: torch.Tensor | None,
        noise_mask: torch.Tensor | None,
        nf: int,
        nloc: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Canonicalize optional public `dens` inputs to concrete tensors.

        Parameters
        ----------
        force_input
            Optional atom-wise force input tensor.
        noise_mask
            Optional atom-wise corruption mask.
        nf
            Number of frames.
        nloc
            Number of local atoms per frame.
        dtype
            Target floating-point dtype.
        device
            Target device.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Canonicalized force tensor with shape `(nf, nloc, 3)` and mask with
            shape `(nf, nloc)`.

        Notes
        -----
        `force_input` and `noise_mask` remain optional only at the outer model
        API. Internal `dens` compute functions always receive concrete tensors.
        """
        if force_input is None:
            force_input = torch.zeros((nf, nloc, 3), dtype=dtype, device=device)
        else:
            if force_input.ndim == 2:
                force_input = force_input.view(nf, nloc, 3)
            elif force_input.ndim != 3:
                raise ValueError(
                    "`force_input` must have shape (nf, nloc, 3) or (nf, nloc*3)."
                )
            force_input = force_input.to(device=device, dtype=dtype)

        if noise_mask is None:
            noise_mask = torch.zeros((nf, nloc), dtype=torch.bool, device=device)
        else:
            if noise_mask.ndim != 2:
                raise ValueError("`noise_mask` must have shape (nf, nloc).")
            noise_mask = noise_mask.to(device=device, dtype=torch.bool)

        return force_input, noise_mask

    def get_active_mode(self) -> str:
        """Return the current SeZM execution mode."""
        return self.atomic_model.get_active_mode()

    def set_active_mode(self, mode: str) -> None:
        """
        Switch the active SeZM execution mode.

        Parameters
        ----------
        mode
            Target mode. Must be `ener` or `dens`.
        """
        self.atomic_model.set_active_mode(mode)

    def set_active_mode_from_loss(self, loss_type: str) -> None:
        """
        Select the active SeZM path from `loss.type`.

        Parameters
        ----------
        loss_type
            Loss type name.
        """
        normalized = str(loss_type).lower()
        if normalized in {"ener", "dens"}:
            self.set_active_mode(normalized)

    def reset_head_for_mode(self, mode: str) -> None:
        """
        Reinitialize one SeZM fitting head and reset mode-specific compile state.

        Parameters
        ----------
        mode
            Target mode to reset.
        """
        self.atomic_model.reset_head_for_mode(mode)
        if mode == "dens":
            self._dens_compiled = False
            object.__setattr__(self, "compiled_dens_compute", None)
        else:
            self._core_compute_compiled_train = None
            self._core_compute_compiled_atomic_virial = None
            object.__setattr__(self, "compiled_core_compute", None)

    def translated_output_def(self) -> dict[str, Any]:
        """
        Translate model output definition to a dictionary format.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping output names to their corresponding output definitions.
        """
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": out_def_data["energy"],
            "energy": out_def_data["energy_redu"],
        }
        if "dforce" in out_def_data:
            output_def["force"] = out_def_data["dforce"]
        elif self.do_grad_r("energy"):
            output_def["force"] = out_def_data["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = out_def_data["energy_derv_c_redu"].squeeze(-2)
            output_def["atom_virial"] = out_def_data["energy_derv_c"].squeeze(-2)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]

        return output_def

    def serialize(self) -> dict[str, Any]:
        """
        Serialize the SeZM model including model-level bridging state.

        Returns
        -------
        dict[str, Any]
            Serialized SeZM model data.
        """
        return {
            "@class": "Model",
            "@version": 1,
            "type": self.model_type,
            "atomic_model": self.atomic_model.serialize(),
            "use_compile": self.use_compile,
            "enable_tf32": self.enable_tf32,
            "bridging_method": self.bridging_method,
            "bridging_r_inner": self.bridging_r_inner,
            "bridging_r_outer": self.bridging_r_outer,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SeZMModel:
        """
        Deserialize the SeZM model including model-level bridging state.

        Parameters
        ----------
        data
            Serialized SeZM model data.

        Returns
        -------
        SeZMModel
            Deserialized SeZM model.
        """
        data = data.copy()
        version = int(data.pop("@version", 1))
        if version != 1:
            raise ValueError(f"Unsupported SeZM version: {version}")
        data.pop("@class", None)
        data.pop("type", None)
        atomic_model = SeZMAtomicModel.deserialize(data.pop("atomic_model"))
        return cls(atomic_model_=atomic_model, **data)

    # =========================================================================
    # Context Managers
    # =========================================================================

    @contextmanager
    def tf32_precision_ctx(self) -> Generator[None, None, None]:
        """Context manager to temporarily set TF32 matmul precision."""
        if not self.should_use_compile() or not torch.cuda.is_available():
            yield
            return
        prev_precision = torch.get_float32_matmul_precision()
        try:
            if self.enable_tf32:
                torch.set_float32_matmul_precision("high")
            else:
                torch.set_float32_matmul_precision("highest")
            yield
        finally:
            torch.set_float32_matmul_precision(prev_precision)


# =============================================================================
# InterPotential: analytical pair potentials for bridging
# =============================================================================

# fmt: off
ELEMENT_TO_Z: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
    "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99,
    "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,
    "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111,
    "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117,
    "Og": 118,
}
# fmt: on

# ZBL screening function coefficients
_ZBL_A_COEFF = (0.18175, 0.50986, 0.28022, 0.028171)
_ZBL_B_COEFF = (3.1998, 0.94229, 0.4029, 0.20162)

# Physical constants
_KE_EV_A = 14.3996  # Coulomb constant in eV·Å
_A_BOHR = 0.5291772109  # Bohr radius in Å


class InterPotential(torch.nn.Module):
    """
    Analytical pair potential module for Zone bridging.

    Supports the Ziegler-Biersack-Littmark (ZBL) screened nuclear repulsion
    potential. Designed to be extensible to other analytical forms (LJ, Morse,
    etc.) through the ``mode`` parameter.

    Each pair (i, j) contributes ``V_ZBL(r_ij) / 2`` to both atom i and atom j,
    avoiding double-counting from the symmetric neighbor list.

    Parameters
    ----------
    type_map : list[str]
        Element symbols (e.g. ``["O", "H"]``). Index in this list corresponds
        to the ``atype`` integer values.
    mode : str
        Potential formula. Currently only ``"zbl"`` is supported.

    Raises
    ------
    ValueError
        If ``mode`` is not recognized, or if any element in ``type_map`` is
        not found in the periodic table.
    """

    def __init__(self, type_map: list[str], mode: str = "zbl") -> None:
        super().__init__()
        mode = mode.upper()
        if mode != "ZBL":
            raise ValueError(f"Unknown InterPotential mode: {mode}")
        self.mode = mode

        atomic_numbers = []
        for elem in type_map:
            z = ELEMENT_TO_Z.get(elem)
            if z is None:
                raise ValueError(f"Unknown element symbol: {elem}")
            atomic_numbers.append(z)
        self.register_buffer(
            "atomic_numbers",
            torch.tensor(atomic_numbers, dtype=torch.float64, device=env.DEVICE),
        )

    def _zbl_pair_energy(
        self,
        r: torch.Tensor,
        zi: torch.Tensor,
        zj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ZBL pair energy for given distances and nuclear charges.

        Parameters
        ----------
        r : torch.Tensor
            Pair distances with shape (...) in Å.
        zi : torch.Tensor
            Nuclear charge of atom i with shape (...).
        zj : torch.Tensor
            Nuclear charge of atom j with shape (...).

        Returns
        -------
        torch.Tensor
            Pair energies with shape (...) in eV.
        """
        a_screen = 0.88534 * _A_BOHR / (zi.pow(0.23) + zj.pow(0.23))
        x = r / a_screen
        phi = sum(a * torch.exp(-b * x) for a, b in zip(_ZBL_A_COEFF, _ZBL_B_COEFF))
        return _KE_EV_A * zi * zj / r * phi

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        nloc: int,
    ) -> torch.Tensor:
        """
        Compute per-atom pair energy from the standard neighbor list path.

        Parameters
        ----------
        extended_coord
            Coordinates in extended region with shape (nf, nall, 3) in Å.
        extended_atype
            Atom types in extended region with shape (nf, nall).
        nlist
            Neighbor list with shape (nf, nloc, nsel).
        nloc : int
            Number of local atoms.

        Returns
        -------
        torch.Tensor
            Per-atom pair energy with shape (nf, nloc, 1) in eV.
        """
        nf = extended_coord.shape[0]
        coord64 = extended_coord.to(dtype=torch.float64)
        z_all = self.atomic_numbers[extended_atype.clamp(min=0)]  # (nf, nall)

        # === Step 1. Gather neighbor coordinates and types ===
        nsel = nlist.shape[2]
        nlist_clamp = nlist.clamp(min=0)  # (nf, nloc, nsel)
        nei_coord = torch.gather(
            coord64, 1, nlist_clamp.unsqueeze(-1).expand(-1, -1, -1, 3).view(nf, -1, 3)
        ).view(nf, nloc, nsel, 3)
        atom_coord = coord64[:, :nloc].unsqueeze(2)  # (nf, nloc, 1, 3)
        diff = nei_coord - atom_coord  # (nf, nloc, nsel, 3)
        r = diff.norm(dim=-1).clamp(min=1e-10)  # (nf, nloc, nsel)

        zi = z_all[:, :nloc].unsqueeze(2).expand_as(r)  # (nf, nloc, nsel)
        zj_idx = nlist_clamp
        zj = torch.gather(z_all, 1, zj_idx.view(nf, -1)).view(nf, nloc, nsel)

        # === Step 2. Compute pair energies ===
        pair_e = self._zbl_pair_energy(r, zi, zj)  # (nf, nloc, nsel)

        # Mask padding entries (nlist == -1)
        valid = (nlist >= 0).to(dtype=pair_e.dtype)
        pair_e = pair_e * valid

        # Half contribution to avoid double-counting
        atom_pair_energy = (pair_e * 0.5).sum(dim=-1, keepdim=True)  # (nf, nloc, 1)
        return atom_pair_energy.to(dtype=extended_coord.dtype)

    def forward_from_edges(
        self,
        edge_vec: torch.Tensor,
        edge_index: torch.Tensor,
        atype_flat: torch.Tensor,
        edge_mask: torch.Tensor,
        n_node: int,
    ) -> torch.Tensor:
        """
        Compute per-atom pair energy from the compile-path edge list.

        Parameters
        ----------
        edge_vec
            Edge vectors with shape (E, 3) in Å.
        edge_index
            Edge source/destination indices with shape (2, E).
        atype_flat
            Flat atom types with shape (N,).
        edge_mask
            Boolean mask with shape (E,). True means valid edge.
        n_node : int
            Number of flattened local nodes.

        Returns
        -------
        torch.Tensor
            Per-atom pair energy with shape (1, N, 1) in eV.
        """
        src = edge_index[0].to(dtype=torch.long)
        dst = edge_index[1].to(dtype=torch.long)

        r = edge_vec.to(dtype=torch.float64).norm(dim=-1).clamp(min=1e-10)  # (E,)
        z_all = self.atomic_numbers[atype_flat.clamp(min=0)]  # (N,)
        zi = z_all[src]  # (E,)
        zj = z_all[dst]  # (E,)

        pair_e = self._zbl_pair_energy(r, zi, zj)  # (E,)
        pair_e = pair_e * edge_mask.to(dtype=pair_e.dtype)

        # Half contribution to each destination atom
        atom_energy = torch.zeros(n_node, dtype=pair_e.dtype, device=pair_e.device)
        atom_energy.index_add_(0, dst, pair_e * 0.5)

        return atom_energy.to(dtype=edge_vec.dtype).view(1, n_node, 1)
