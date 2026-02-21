# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM-Net: Smooth equivariant ZBL Message-passing Network."""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
from einops import (
    rearrange,
)

if TYPE_CHECKING:
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
    SeZM-Net energy model using standard DeePMD neighbor list.

    This model uses the traditional DeePMD neighbor list path with ghost atoms
    and padded neighbor matrix, compatible with LAMMPS and other MD engines.
    """

    model_type = "SeZM-Net"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DPModelCommon.__init__(self)
        SeZMNetModel_.__init__(self, *args, **kwargs)

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
