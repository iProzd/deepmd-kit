# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    GLOBAL_PT_FLOAT_PRECISION,
)
from deepmd.utils.data import (
    DataRequirementItem,
)


class DeNSLoss(TaskLoss):
    def __init__(
        self,
        starter_learning_rate=1.0,
        start_pref_e=0.0,
        limit_pref_e=0.0,
        start_pref_n=0.0,
        limit_pref_n=0.0,
        noise_std=0.1,
        corrupt_ratio=1.0,
        inference=False,
        **kwargs,
    ) -> None:
        r"""Construct a layer to compute loss on energy, force and virial.

        Parameters
        ----------
        starter_learning_rate : float
            The learning rate at the start of the training.
        start_pref_e : float
            The prefactor of energy loss at the start of the training.
        limit_pref_e : float
            The prefactor of energy loss at the end of the training.
        start_pref_n : float
            The prefactor of noise loss at the start of the training.
        limit_pref_n : float
            The prefactor of noise loss at the end of the training.
        inference : bool
            If true, it will output all losses found in output, ignoring the pre-factors.
        **kwargs
            Other keyword arguments.
        """
        super().__init__()
        self.starter_learning_rate = starter_learning_rate
        self.has_e = (start_pref_e != 0.0 and limit_pref_e != 0.0) or inference
        self.has_n = (start_pref_n != 0.0 and limit_pref_n != 0.0) or inference

        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_n = start_pref_n
        self.limit_pref_n = limit_pref_n
        self.noise_std = noise_std
        self.corrupt_ratio = corrupt_ratio
        self.inference = inference

    def forward(self, input_dict, model, label, natoms, learning_rate, mae=False):
        """Return loss on energy and noise.

        Parameters
        ----------
        input_dict : dict[str, torch.Tensor]
            Model inputs.
        model : torch.nn.Module
            Model to be used to output the predictions.
        label : dict[str, torch.Tensor]
            Labels.
        natoms : int
            The local atom number.

        Returns
        -------
        model_pred: dict[str, torch.Tensor]
            Model predictions.
        loss: torch.Tensor
            Loss for model to minimize.
        more_loss: dict[str, torch.Tensor]
            Other losses for display.
        """
        coord_clean = input_dict["coord"]
        noise_vec = torch.zeros_like(coord_clean)
        noise_vec = noise_vec.normal_(mean=0.0, std=self.noise_std)
        if self.corrupt_ratio < 1.0:
            noise_mask = torch.rand(
                (coord_clean.shape[:2]),
                dtype=coord_clean.dtype,
                device=coord_clean.device,
            )
            noise_mask = noise_mask < self.corrupt_ratio
            noise_vec[(~noise_mask)] *= 0
        else:
            noise_mask = torch.ones(
                (coord_clean.shape[:2]),
                dtype=torch.bool,
                device=coord_clean.device,
            )
        noised_coord = input_dict["coord"] + noise_vec
        input_dict["coord"] = noised_coord

        # add force embedding
        input_dict["force_embedding_input"] = label["force"].detach().clone()
        model_pred = model(**input_dict)

        coef = learning_rate / self.starter_learning_rate
        pref_e = self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * coef
        pref_n = self.limit_pref_n + (self.start_pref_n - self.limit_pref_n) * coef

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}
        # more_loss['log_keys'] = []  # showed when validation on the fly
        # more_loss['test_keys'] = []  # showed when doing dp test
        atom_norm = 1.0 / natoms
        if self.has_e and "energy" in model_pred and "energy" in label:
            energy_pred = model_pred["energy"]
            energy_label = label["energy"]
            find_energy = label.get("find_energy", 0.0)
            pref_e = pref_e * find_energy
            l2_ener_loss = torch.mean(torch.square(energy_pred - energy_label))
            if not self.inference:
                more_loss["l2_ener_loss"] = self.display_if_exist(
                    l2_ener_loss.detach(), find_energy
                )
            loss += atom_norm * (pref_e * l2_ener_loss)
            rmse_e = l2_ener_loss.sqrt() * atom_norm
            more_loss["rmse_e"] = self.display_if_exist(rmse_e.detach(), find_energy)
            # more_loss['log_keys'].append('rmse_e')

        if self.has_n:
            noise_predict = model_pred["dforce"]
            diff_n = (noise_vec - noise_predict)[noise_mask].reshape(-1)
            l2_noise_loss = torch.mean(torch.square(diff_n))
            if not self.inference:
                more_loss["l2_noise_loss"] = self.display_if_exist(
                    l2_noise_loss.detach(), 1
                )
            loss += (pref_n * l2_noise_loss).to(GLOBAL_PT_FLOAT_PRECISION)
            rmse_n = l2_noise_loss.sqrt()
            more_loss["rmse_n"] = self.display_if_exist(rmse_n.detach(), 1)

        if not self.inference:
            more_loss["rmse"] = torch.sqrt(loss.detach())
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        if self.has_e:
            label_requirement.append(
                DataRequirementItem(
                    "energy",
                    ndof=1,
                    atomic=False,
                    must=False,
                    high_prec=True,
                )
            )
        if self.has_n:
            label_requirement.append(
                DataRequirementItem(
                    "force",
                    ndof=3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        return label_requirement
