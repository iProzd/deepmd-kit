# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM-Net: Smooth equivariant ZBL Message-passing Network."""

from __future__ import (
    annotations,
)

from contextlib import (
    contextmanager,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.nn.functional as F
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

from deepmd.pt.model.atomic_model.sezm_net_atomic_model import (
    SeZMNetAtomicModel,
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
    communicate_extended_output,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.nvtx import (
    nvtx_range,
)

SeZMNetModel_ = make_model(SeZMNetAtomicModel)


@BaseModel.register("SeZM-Net")
@BaseModel.register("se_zm_net")
@BaseModel.register("se_zm-net")
@BaseModel.register("sezm-net")
class SeZMNetModel(DPModelCommon, SeZMNetModel_):
    """
    SeZM-Net energy model with optional fixed-shape compile path.

    By default it uses the traditional DeePMD neighbor list path with ghost atoms
    and padded neighbor matrix, compatible with LAMMPS and other MD engines.
    When `use_compile=True`, it builds fixed-shape sparse edges from the standard
    neighbor list and routes the descriptor through a pure-tensor graph for
    torch.compile.
    """

    model_type = "SeZM-Net"

    def __init__(
        self,
        *args: Any,
        use_compile: bool = False,
        use_tf32: bool = False,
        n_node: int | None = None,
        n_edge: int = 0,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        SeZMNetModel_.__init__(self, *args, **kwargs)
        self.redu_prec = env.GLOBAL_PT_ENER_FLOAT_PRECISION
        self.use_compile = bool(use_compile)
        self.use_tf32 = bool(use_tf32)
        self.n_node = int(n_node) if n_node is not None else None
        self.n_edge = int(n_edge)
        self._compiled = False
        # Store compiled_compute outside the nn.Module tree so that
        # FSDP2 / DDP do not shard or sync its duplicated parameters.
        object.__setattr__(self, "compiled_compute", None)
        if self.use_compile:
            if self.n_node is None or self.n_node <= 0:
                raise ValueError("n_node must be positive when use_compile=True")
            if self.n_edge < 0:
                raise ValueError("n_edge must be non-negative")
            if self.n_edge > 0 and self.n_edge > self.n_node * self.get_nsel():
                raise ValueError("n_edge must be <= n_node * nsel when set")

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

        Returns
        -------
        dict[str, torch.Tensor]
            Model predictions including energy, forces, etc.
        """
        with nvtx_range("SeZMNet/forward_common"):
            if self.use_compile:
                return self.forward_common_compile(
                    coord,
                    atype,
                    box,
                    fparam=fparam,
                    aparam=aparam,
                    do_atomic_virial=do_atomic_virial,
                )

            # === Step 1. Cast inputs to correct dtype ===
            with nvtx_range("SeZMNet/input_type_cast"):
                cc, bb, fp, ap, input_prec = self.input_type_cast(
                    coord, box=box, fparam=fparam, aparam=aparam
                )
                del coord, box, fparam, aparam

            # === Step 2. Build neighbor list ===
            with nvtx_range("SeZMNet/build_neighbor_list"):
                # extended_coord: (nf, nall, 3), extended_atype: (nf, nall)
                # mapping: (nf, nall), nlist: (nf, nloc, nsel)
                extended_coord, extended_atype, mapping, nlist = (
                    self.build_neighbor_list(cc, atype, bb)
                )

            # === Step 3. Lower Forward + Communication ===
            with nvtx_range("SeZMNet/forward_lower"):
                model_predict_lower = self.forward_common_lower(
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping,
                    do_atomic_virial=do_atomic_virial,
                    fparam=fp,
                    aparam=ap,
                )

            with nvtx_range("SeZMNet/communicate_output"):
                model_predict = communicate_extended_output(
                    model_predict_lower,
                    self.model_output_def(),
                    mapping,
                    do_atomic_virial=do_atomic_virial,
                )

            with nvtx_range("SeZMNet/output_type_cast"):
                model_predict = self.output_type_cast(model_predict, input_prec)
                return model_predict

    def forward_common_compile(
        self,
        coord: Float[Tensor, "nf nloc 3"] | Float[Tensor, "nf nloc_x3"],
        atype: Int[Tensor, "nf nloc"],
        box: Float[Tensor, "nf 9"] | None = None,
        fparam: Float[Tensor, "nf ndf"] | None = None,
        aparam: Float[Tensor, "nf nloc nda"] | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass using fixed-shape sparse edges and torch.compile.

        This path uses DeePMD neighbor list to build fixed-shape edges,
        then traces/compiles the compute graph.
        """
        with nvtx_range("SeZMNet/forward_common_compile"):
            # === Step 1. Cast inputs to correct dtype ===
            with nvtx_range("SeZMNet/input_type_cast"):
                cc, bb, fp, ap, input_prec = self.input_type_cast(
                    coord, box=box, fparam=fparam, aparam=aparam
                )
                del coord, box, fparam, aparam

                nf, nloc = atype.shape[:2]
                if cc.ndim == 2:
                    cc = cc.view(nf, nloc, 3)

            # === Step 2. Prepare compile fitting inputs ===
            # Make compile inputs tensor-only to keep inductor/DDP runtime
            # on a single stable calling convention.
            fp, ap = self.prepare_compile_fitting_inputs(
                fp,
                ap,
                nf=nf,
                nloc=nloc,
                dtype=cc.dtype,
                device=cc.device,
            )

            # === Step 3. Enable gradient early ===
            need_grad = self.do_grad_r() or self.do_grad_c()
            if need_grad:
                cc.requires_grad_(True)

            # === Step 4. Build neighbor list (standard DeePMD path) ===
            with nvtx_range("SeZMNet/build_neighbor_list"):
                extended_coord, extended_atype, mapping, nlist = (
                    self.build_neighbor_list(cc, atype, bb)
                )

            # === Step 5. Flatten and pad to fixed n_node ===
            n_actual = nf * nloc
            if self.n_node is None:
                raise ValueError("n_node must be set when use_compile=True")
            n_node = self.n_node
            if n_actual > n_node:
                raise ValueError(
                    f"Actual atoms ({n_actual} = {nf}x{nloc}) exceed n_node ({n_node})"
                )
            cc_flat = cc.view(n_actual, 3)
            atype_flat = atype.view(n_actual)

            pad_size = n_node - n_actual
            if pad_size > 0:
                cc_flat = F.pad(
                    cc_flat, (0, 0, 0, pad_size), mode="constant", value=0.0
                )
                atype_flat = torch.cat(
                    [
                        atype_flat,
                        torch.full(
                            (pad_size,),
                            -1,
                            dtype=atype_flat.dtype,
                            device=atype_flat.device,
                        ),
                    ],
                    dim=0,
                )

            atom_mask = torch.zeros(n_node, dtype=cc_flat.dtype, device=cc_flat.device)
            atom_mask[:n_actual] = 1.0

            atype_valid = torch.where(atype_flat >= 0, atype_flat, 0)

            # === Step 6. Build fixed-shape edges from nlist ===
            edge_index, edge_vec, edge_mask = self.build_fixed_edge_list_from_nlist(
                extended_coord=extended_coord,
                nlist=nlist,
                mapping=mapping,
                n_node=n_node,
                n_edge=self.n_edge,
            )

            # === Step 7. Trace and compile on first forward ===
            with self.tf32_precision_ctx():
                if not self._compiled:
                    self.trace_and_compile(
                        cc_flat,
                        atype_valid,
                        edge_index,
                        edge_vec,
                        edge_mask,
                        atom_mask,
                        fp,
                        ap,
                    )

                # === Step 8. Forward through compiled compute path ===
                with nvtx_range("SeZMNet/forward_compute"):
                    compute_ret = self.compiled_compute(
                        cc_flat,
                        atype_valid,
                        edge_index,
                        edge_vec,
                        edge_mask,
                        atom_mask,
                        fp,
                        ap,
                    )

            # === Step 9. Post-process outputs ===
            with nvtx_range("SeZMNet/post_process"):
                model_predict = self.post_process_output(
                    compute_ret, atype, nf, nloc, do_atomic_virial
                )

            # === Step 10. Output type cast ===
            with nvtx_range("SeZMNet/output_type_cast"):
                model_predict = self.output_type_cast(model_predict, input_prec)
                return model_predict

    def trace_and_compile(
        self,
        cc_flat: torch.Tensor,
        atype_valid: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        atom_mask: torch.Tensor,
        fp: torch.Tensor,
        ap: torch.Tensor,
    ) -> None:
        """Trace computation graph with make_fx and compile."""

        def compute_fn(
            cc_flat: torch.Tensor,
            atype_valid: torch.Tensor,
            edge_index: torch.Tensor,
            edge_vec: torch.Tensor,
            edge_mask: torch.Tensor,
            atom_mask: torch.Tensor,
            fp: torch.Tensor,
            ap: torch.Tensor,
        ) -> torch.Tensor:
            return self.compile_compute_func(
                cc_flat,
                atype_valid,
                edge_index,
                edge_vec,
                edge_mask,
                atom_mask,
                fp,
                ap,
            )

        traced = make_fx(
            compute_fn,
            tracing_mode="real",
            _allow_non_fake_inputs=True,
        )(
            cc_flat,
            atype_valid,
            edge_index,
            edge_vec,
            edge_mask,
            atom_mask,
            fp,
            ap,
        )

        if not torch.cuda.is_available():
            # CPU fallback: use aot_eager for compatibility
            object.__setattr__(
                self, "compiled_compute", torch.compile(traced, backend="aot_eager")
            )
        else:
            # GPU: use inductor with full performance options
            # These options enable aggressive optimizations:
            # - max_autotune: profile to pick best matmul config
            # - epilogue_fusion: fuse pointwise ops into templates (requires max_autotune)
            # - triton.cudagraphs: reduce CPU overhead via CUDA graphs
            # - shape_padding: pad matrices for better GPU alignment (tensor cores)
            # - max_fusion_size: limit fusion size limit for complex graphs
            import torch.distributed as dist

            is_multi_gpu = (
                dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            )
            # Single-GPU: enable all optimizations
            # Multi-GPU: disable autotune/cudagraphs to avoid DDP sync issues
            compile_options = {
                "max_autotune": not is_multi_gpu,
                "epilogue_fusion": not is_multi_gpu,
                "triton.cudagraphs": not is_multi_gpu,
                "shape_padding": not is_multi_gpu,
                "max_fusion_size": 16,
            }
            object.__setattr__(
                self, "compiled_compute", torch.compile(traced, options=compile_options)
            )
        self._compiled = True

    def compile_compute_func(
        self,
        cc_flat: torch.Tensor,
        atype_valid: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_mask: torch.Tensor,
        atom_mask: torch.Tensor,
        fp: torch.Tensor,
        ap: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute-only forward pass (make_fx compatible) with fixed shapes.

        Returns a single concatenated tensor.

        Layout along last dim (always padded to ``n_node``):
            - ``[:, :, 0:1]``  — atom_energy  (always present)
            - ``[:, :, 1:4]``  — minus_force  (zeros when force is off)
            - ``[:, :, 4:13]`` — atom_virial  (zeros when virial is off)
        """
        if self.n_node is None:
            raise ValueError("n_node must be set when use_compile=True")

        n_node = self.n_node

        # === Step 1. Attach coord gradients to edge vectors ===
        edge_vec = self.attach_edge_vec_grad(cc_flat, edge_index, edge_vec)

        # === Step 2. Descriptor (edge path only) ===
        with nvtx_range("SeZMNet/descriptor"):
            descriptor_model = self.atomic_model.descriptor
            descriptor, rot_mat, g2, h2, _ = descriptor_model.forward_with_edges(
                extended_coord=cc_flat.view(1, n_node, 3),
                extended_atype=atype_valid.view(1, n_node),
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_mask=edge_mask,
            )
        assert descriptor is not None

        if descriptor.ndim == 2:
            descriptor_for_fit = descriptor.unsqueeze(0)
        else:
            descriptor_for_fit = descriptor
        atype_for_fit = atype_valid.unsqueeze(0)

        # === Step 3. Fitting net ===
        with nvtx_range("SeZMNet/fitting_net"):
            fit_ret = self.atomic_model.fitting_net(
                descriptor_for_fit,
                atype_for_fit,
                gr=rot_mat,
                g2=g2,
                h2=h2,
                fparam=fp,
                aparam=ap,
            )

        with nvtx_range("SeZMNet/apply_out_stat"):
            fit_ret = self.atomic_model.apply_out_stat(fit_ret, atype_for_fit)

        atom_energy = fit_ret["energy"]  # (1, n_node, 1)
        redu_prec = self.redu_prec
        atom_mask_view = atom_mask.to(dtype=redu_prec).view(1, n_node, 1)
        atom_energy_masked = atom_energy * atom_mask_view
        energy_sum = torch.sum(atom_energy_masked.to(redu_prec))

        do_grad_r = self.do_grad_r("energy")
        do_grad_c = self.do_grad_c("energy")

        # === Step 4. Force / virial via autograd ===
        # Always use create_graph=True for compile path compatibility.
        force_flat = torch.zeros(
            n_node, 3, dtype=atom_energy.dtype, device=atom_energy.device
        )
        virial_flat = torch.zeros(
            n_node, 9, dtype=atom_energy.dtype, device=atom_energy.device
        )

        # Force and virial
        if do_grad_r and do_grad_c:
            minus_force_flat, edge_grad = torch.autograd.grad(
                outputs=[energy_sum],
                inputs=[cc_flat, edge_vec],
                create_graph=True,
                retain_graph=True,
            )
            force_flat = minus_force_flat

            edge_keep = edge_mask.to(dtype=edge_grad.dtype).unsqueeze(-1)
            edge_grad = edge_grad * edge_keep
            edge_vec_val = edge_vec.detach()
            edge_virial = -torch.einsum("Ei,Ej->Eij", edge_grad, edge_vec_val)
            edge_virial = edge_virial * edge_keep.unsqueeze(-1)
            edge_virial_flat = edge_virial.view(-1, 9)
            dst = edge_index[1].to(dtype=torch.long)
            atom_virial = torch.zeros(
                n_node,
                9,
                dtype=edge_virial_flat.dtype,
                device=edge_virial_flat.device,
            )
            virial_flat = atom_virial.index_add(0, dst, edge_virial_flat)
        # Force only
        elif do_grad_r:
            (minus_force_flat,) = torch.autograd.grad(
                outputs=[energy_sum],
                inputs=[cc_flat],
                create_graph=True,
                retain_graph=True,
            )
            force_flat = minus_force_flat
        # Virial only
        elif do_grad_c:
            (edge_grad,) = torch.autograd.grad(
                outputs=[energy_sum],
                inputs=[edge_vec],
                create_graph=True,
                retain_graph=True,
            )
            edge_keep = edge_mask.to(dtype=edge_grad.dtype).unsqueeze(-1)
            edge_grad = edge_grad * edge_keep
            edge_vec_val = edge_vec.detach()
            edge_virial = -torch.einsum("Ei,Ej->Eij", edge_grad, edge_vec_val)
            edge_virial = edge_virial * edge_keep.unsqueeze(-1)
            edge_virial_flat = edge_virial.view(-1, 9)
            dst = edge_index[1].to(dtype=torch.long)
            atom_virial = torch.zeros(
                n_node,
                9,
                dtype=edge_virial_flat.dtype,
                device=edge_virial_flat.device,
            )
            virial_flat = atom_virial.index_add(0, dst, edge_virial_flat)

        # === Step 5. Concatenate into single output tensor ===
        # Layout: [energy(1) | force(3) | virial(9)] along last dim.
        return torch.cat(
            [
                atom_energy_masked.view(1, n_node, 1),
                force_flat.view(1, n_node, 3),
                virial_flat.view(1, n_node, 9),
            ],
            dim=-1,
        )  # (1, n_node, 13)

    def attach_edge_vec_grad(
        self,
        cc_flat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attach coordinate gradients to edge vectors without changing values.

        Parameters
        ----------
        cc_flat : torch.Tensor
            Flattened coordinates with shape (N, 3) in Å.
        edge_index : torch.Tensor
            Edge indices with shape (2, E).
        edge_vec : torch.Tensor
            Edge vectors with shape (E, 3) in Å.

        Returns
        -------
        torch.Tensor
            Edge vectors with attached gradient path and shape (E, 3).
        """
        # === Step 1. Detach values to avoid duplicate gradient paths ===
        edge_vec = edge_vec.detach()

        # === Step 2. Gather src/dst coordinates ===
        src = edge_index[0].to(dtype=torch.long)
        dst = edge_index[1].to(dtype=torch.long)
        cc_src = cc_flat.index_select(0, src)
        cc_dst = cc_flat.index_select(0, dst)

        # === Step 3. Inject gradient path via stop-gradient trick ===
        return edge_vec + (cc_src - cc_src.detach()) - (cc_dst - cc_dst.detach())

    def post_process_output(
        self,
        compute_ret: torch.Tensor,
        atype: torch.Tensor,
        nf: int,
        nloc: int,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Post-process the single concatenated tensor from compile_compute_func.

        Parameters
        ----------
        compute_ret : torch.Tensor
            Concatenated output with shape (1, n_node, 13).
            Layout: [energy(1) | force(3) | virial(9)].
        atype : torch.Tensor
            Atom types with shape (nf, nloc).
        nf : int
            Number of frames.
        nloc : int
            Number of local atoms per frame.
        do_atomic_virial : bool
            Whether to output per-atom virial.

        Returns
        -------
        dict[str, torch.Tensor]
            Standard DeePMD model predictions.
        """
        if self.n_node is None:
            raise ValueError("n_node must be set when use_compile=True")

        n_actual = nf * nloc
        redu_prec = self.redu_prec

        # === Step 1. Split concatenated tensor ===
        atom_energy_masked = compute_ret[:, :, 0:1]  # (1, n_node, 1)
        minus_force_flat = compute_ret[:, :, 1:4]  # (1, n_node, 3)
        atom_virial_flat = compute_ret[:, :, 4:13]  # (1, n_node, 9)

        # === Step 2. Energy ===
        # atom_energy is already masked in compile_compute_func.
        # The compile path flattens all frames into one graph, so we must slice to
        # actual atoms and reshape back to (nf, nloc, 1) before reducing frame energies.
        atom_energy = atom_energy_masked[:, :n_actual, :].view(nf, nloc, 1)
        energy = torch.sum(atom_energy.to(redu_prec), dim=1)

        model_predict: dict[str, torch.Tensor] = {
            "energy": atom_energy,
            "energy_redu": energy,
        }

        # === Step 3. Force ===
        if self.do_grad_r("energy"):
            force = -minus_force_flat[0, :n_actual, :].view(nf, nloc, 3)
            model_predict["energy_derv_r"] = force.unsqueeze(-2)

        # === Step 4. Virial ===
        if self.do_grad_c("energy"):
            atom_virial = atom_virial_flat[0, :n_actual, :].view(nf, nloc, 9)
            if do_atomic_virial:
                model_predict["energy_derv_c"] = atom_virial.unsqueeze(-2)
            model_predict["energy_derv_c_redu"] = torch.sum(
                atom_virial.to(redu_prec), dim=1, keepdim=True
            )

        # === Step 5. Atom mask ===
        ext_atom_mask = self.atomic_model.make_atom_mask(atype)
        atom_mask_out = ext_atom_mask.to(torch.int32)
        if self.atomic_model.atom_excl is not None:
            atom_mask_out = atom_mask_out * self.atomic_model.atom_excl(atype)
        model_predict["mask"] = atom_mask_out

        return model_predict

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
        Lower-level forward using traditional neighbor list (LAMMPS compatible).

        Parameters
        ----------
        extended_coord
            Extended coordinates with shape (nf, nall*3) in Å.
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
            Communication dict (unused).

        Returns
        -------
        dict[str, torch.Tensor]
            Model predictions including atom_energy, energy, extended_force,
            virial, extended_virial, and dforce.
        """
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
        )
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

    def build_fixed_edge_list_from_nlist(
        self,
        *,
        extended_coord: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None,
        n_node: int,
        n_edge: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build fixed-shape edge list from DeePMD padded neighbor list.

        Returns
        -------
        edge_index
            Edge indices with shape (2, n_edge).
        edge_vec
            Edge vectors with shape (n_edge, 3).
        edge_mask
            Boolean mask with shape (n_edge,).
        """
        nf, nloc, nsel = nlist.shape
        n_actual = nf * nloc
        device = extended_coord.device
        nall = extended_coord.shape[1]

        # === Step 1. Build per-edge geometry ===
        valid_nlist = nlist >= 0
        gather_index = torch.where(valid_nlist, nlist, torch.zeros_like(nlist))
        index = rearrange(gather_index, "nf nloc nnei -> nf (nloc nnei) 1").expand(
            -1, -1, 3
        )
        nei_pos = torch.gather(extended_coord, 1, index).view(nf, nloc, nsel, 3)
        atom_pos = extended_coord[:, :nloc].unsqueeze(2)
        diff = nei_pos - atom_pos
        edge_len2 = torch.sum(diff * diff, dim=-1).reshape(-1)  # (n_actual * nsel,)
        edge_vec_actual = diff.reshape(n_actual * nsel, 3)

        # === Step 2. Build fixed-shape src/dst ===
        dst_actual = torch.arange(
            n_actual, device=device, dtype=torch.long
        ).repeat_interleave(nsel)
        f_idx = dst_actual // nloc
        neighbor_flat = nlist.reshape(-1)
        valid_flat = neighbor_flat >= 0
        neighbor_safe = torch.where(
            valid_flat, neighbor_flat, torch.zeros_like(neighbor_flat)
        )

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

        n_edge = n_node * nsel if n_edge == 0 else n_edge
        src_full = torch.zeros(n_edge, dtype=torch.long, device=device)
        dst_full = torch.zeros(n_edge, dtype=torch.long, device=device)
        edge_mask_full = torch.zeros(n_edge, dtype=torch.bool, device=device)
        edge_vec_full = torch.zeros(
            n_edge, 3, dtype=edge_vec_actual.dtype, device=device
        )

        fill_count = n_actual * nsel
        if n_edge == n_node * nsel:
            dst_full = torch.arange(
                n_node, device=device, dtype=torch.long
            ).repeat_interleave(nsel)
            src_full[:fill_count] = src_actual
            dst_full[:fill_count] = dst_actual
            edge_mask_full[:fill_count] = edge_mask_actual
            edge_vec_full[:fill_count] = edge_vec_actual
        else:
            # === Step 3. Global edge compaction by distance ===
            valid_idx = torch.nonzero(edge_mask_actual).squeeze(-1)
            num_valid = int(valid_idx.numel())
            if num_valid > 0:
                if n_edge >= num_valid:
                    # Preserve original nlist traversal order when all valid
                    # edges fit in the fixed buffer. Reordering can create
                    # accumulation drift under AMP/bfloat16.
                    sel_idx = valid_idx
                else:
                    edge_len2_valid = edge_len2.index_select(0, valid_idx)
                    _, topk_rel = torch.topk(
                        edge_len2_valid,
                        n_edge,
                        largest=False,
                        sorted=False,
                    )
                    sel_idx = valid_idx.index_select(0, topk_rel)

                src_sel = src_actual.index_select(0, sel_idx)
                dst_sel = dst_actual.index_select(0, sel_idx)
                edge_mask_sel = edge_mask_actual.index_select(0, sel_idx)
                edge_vec_sel = edge_vec_actual.index_select(0, sel_idx)
                fill_count = int(src_sel.numel())

                src_full[:fill_count] = src_sel
                dst_full[:fill_count] = dst_sel
                edge_mask_full[:fill_count] = edge_mask_sel
                edge_vec_full[:fill_count] = edge_vec_sel

        edge_index = torch.stack([src_full, dst_full], dim=0)
        return edge_index, edge_vec_full, edge_mask_full

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
        if self.do_grad_r("energy"):
            output_def["force"] = out_def_data["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = out_def_data["energy_derv_c_redu"].squeeze(-2)
            output_def["atom_virial"] = out_def_data["energy_derv_c"].squeeze(-3)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]

        return output_def

    # =========================================================================
    # Context Managers
    # =========================================================================

    @contextmanager
    def tf32_precision_ctx(self) -> Generator[None, None, None]:
        """Context manager to temporarily set TF32 matmul precision."""
        if not self.use_compile or not torch.cuda.is_available():
            yield
            return
        prev_precision = torch.get_float32_matmul_precision()
        try:
            if self.use_tf32:
                torch.set_float32_matmul_precision("medium")
            else:
                torch.set_float32_matmul_precision("highest")
            yield
        finally:
            torch.set_float32_matmul_precision(prev_precision)

    # =========================================================================
    # Compile Input Helpers
    # =========================================================================

    def prepare_compile_fitting_inputs(
        self,
        fp: torch.Tensor | None,
        ap: torch.Tensor | None,
        nf: int,
        nloc: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert optional fitting inputs to tensor-only compile inputs.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tensor-only frame/atomic parameters for compiled runtime.
        """
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
