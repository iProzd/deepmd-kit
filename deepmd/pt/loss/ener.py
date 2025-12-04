# SPDX-License-Identifier: LGPL-3.0-or-later
import math
from typing import (
    Any,
    Optional,
    Union,
)

import torch
import torch.nn.functional as F

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
from deepmd.utils.version import (
    check_version_compatibility,
)


def custom_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0,
    delta2: float = 1.0,
    k1: float = 0.0,
    k2: float = 0.0,
    use_root: bool = False,
) -> torch.Tensor:
    error = targets - predictions
    abs_error = torch.abs(error)
    quadratic_loss = 0.5 * torch.pow(error, 2)
    linear_loss = delta * (abs_error - 0.5 * delta)
    loss = torch.where(abs_error <= delta, quadratic_loss, linear_loss)
    if delta2 is not None and delta2 > delta:
        if not use_root:
            stage2_loss = 0.5 * k1 * torch.log1p(abs_error**2 / k1) + k2
        else:
            stage2_loss = k1 * torch.sqrt(abs_error) + k2
        loss = torch.where(abs_error > delta2, stage2_loss, loss)
    return torch.mean(loss)


def cauchy_params_from_huber(
    delta1: float,
    delta2: float,
) -> tuple[float, float]:
    assert delta2 >= delta1
    # --- 保证 L1 -> Cauchy 连续性所需的参数计算 ---
    # 1. 根据 C1 连续性 (梯度匹配) 求解 Cauchy 尺度 c^2
    # 目标: dL/dr (L1) = delta1; dL/dr (Cauchy) = r / (1 + r^2/c^2)
    # 匹配点 r=delta2: delta1 = delta2 / (1 + delta2^2/c^2)
    c_sq = delta2**2 / (delta2 / delta1 - 1.0)
    # 2. 计算 L2/L1 在 delta2 处的值 (C_match)
    # C_match = delta1 * delta2 - 0.5 * delta1^2
    L_L1_at_delta2 = delta1 * delta2 - 0.5 * delta1**2
    # 3. 计算 Cauchy 在 delta2 处的值 (L_Cauchy)
    L_Cauchy_at_delta2 = 0.5 * c_sq * math.log(1.0 + delta2**2 / c_sq)
    # 4. 计算偏移量 K (K = C_match - L_Cauchy)
    # 保证 L3(r) = L_Cauchy(r) + K 在 delta2 处值连续
    k = L_L1_at_delta2 - L_Cauchy_at_delta2
    return c_sq, k


def root_params_from_huber(
    delta1: float,
    delta2: float,
) -> tuple[float, float]:
    assert delta2 >= delta1
    # 1. 计算 K1 (缩放系数) - 保证 C1 连续性 (梯度匹配)
    # 目标梯度 dL/dr (L1) = delta1
    # L0.5 梯度 dL/dr (L0.5) = 0.5 * K1 * r^(-0.5)
    # 在 r=delta2 匹配: delta1 = 0.5 * K1 * delta2^(-0.5)
    # 解得 K1 = 2 * delta1 * sqrt(delta2)
    K1 = 2.0 * delta1 * math.sqrt(delta2)
    # 2. 计算 L1 在 delta2 处的值 (L_L1)
    L_L1_at_delta2 = delta1 * delta2 - 0.5 * delta1**2
    # 3. 计算 K2 (偏移量) - 保证 C0 连续性 (值匹配)
    # L_L1(delta2) = K1 * sqrt(delta2) + K2
    # 解得 K2 = L_L1(delta2) - K1 * sqrt(delta2)
    K2 = L_L1_at_delta2 - K1 * math.sqrt(delta2)
    # 简化后: K2 = - delta1 * delta2 - 0.5 * delta1^2
    return K1, K2


class EnergyStdLoss(TaskLoss):
    def __init__(
        self,
        starter_learning_rate: float = 1.0,
        start_pref_e: float = 0.0,
        limit_pref_e: float = 0.0,
        start_pref_f: float = 0.0,
        limit_pref_f: float = 0.0,
        start_pref_v: float = 0.0,
        limit_pref_v: float = 0.0,
        start_pref_ae: float = 0.0,
        limit_pref_ae: float = 0.0,
        start_pref_pf: float = 0.0,
        limit_pref_pf: float = 0.0,
        relative_f: Optional[float] = None,
        enable_atom_ener_coeff: bool = False,
        start_pref_gf: float = 0.0,
        limit_pref_gf: float = 0.0,
        numb_generalized_coord: int = 0,
        use_l1_all: bool = False,
        inference: bool = False,
        use_huber: bool = False,
        huber_delta: Union[float, list[float]] = 0.01,
        huber_two_stage_delta: Optional[Union[float, list[float]]] = None,
        trimmed_factor: float = 0.0,
        adaptive_loss: bool = False,
        learnable_pref: bool = False,
        huber_two_stage_use_root: bool = False,
        **kwargs: Any,
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
        start_pref_f : float
            The prefactor of force loss at the start of the training.
        limit_pref_f : float
            The prefactor of force loss at the end of the training.
        start_pref_v : float
            The prefactor of virial loss at the start of the training.
        limit_pref_v : float
            The prefactor of virial loss at the end of the training.
        start_pref_ae : float
            The prefactor of atomic energy loss at the start of the training.
        limit_pref_ae : float
            The prefactor of atomic energy loss at the end of the training.
        start_pref_pf : float
            The prefactor of atomic prefactor force loss at the start of the training.
        limit_pref_pf : float
            The prefactor of atomic prefactor force loss at the end of the training.
        relative_f : float
            If provided, relative force error will be used in the loss. The difference
            of force will be normalized by the magnitude of the force in the label with
            a shift given by relative_f
        enable_atom_ener_coeff : bool
            if true, the energy will be computed as \sum_i c_i E_i
        start_pref_gf : float
            The prefactor of generalized force loss at the start of the training.
        limit_pref_gf : float
            The prefactor of generalized force loss at the end of the training.
        numb_generalized_coord : int
            The dimension of generalized coordinates.
        use_l1_all : bool
            Whether to use L1 loss, if False (default), it will use L2 loss.
        inference : bool
            If true, it will output all losses found in output, ignoring the pre-factors.
        use_huber : bool
            Enables Huber loss calculation for energy/force/virial terms with user-defined threshold delta (D).
            The loss function smoothly transitions between L2 and L1 loss:
            - For absolute prediction errors within D: quadratic loss (0.5 * (error**2))
            - For absolute errors exceeding D: linear loss (D * |error| - 0.5 * D)
            Formula: loss = 0.5 * (error**2) if |error| <= D else D * (|error| - 0.5 * D).
        huber_delta : float
            The threshold delta (D) used for Huber loss, controlling transition between L2 and L1 loss.
        **kwargs
            Other keyword arguments.
        """
        super().__init__()
        self.starter_learning_rate = starter_learning_rate
        self.has_e = (start_pref_e != 0.0 and limit_pref_e != 0.0) or inference
        self.has_f = (start_pref_f != 0.0 and limit_pref_f != 0.0) or inference
        self.has_v = (start_pref_v != 0.0 and limit_pref_v != 0.0) or inference
        self.has_ae = (start_pref_ae != 0.0 and limit_pref_ae != 0.0) or inference
        self.has_pf = (start_pref_pf != 0.0 and limit_pref_pf != 0.0) or inference
        self.has_gf = start_pref_gf != 0.0 and limit_pref_gf != 0.0

        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f
        self.start_pref_v = start_pref_v
        self.limit_pref_v = limit_pref_v
        self.start_pref_ae = start_pref_ae
        self.limit_pref_ae = limit_pref_ae
        self.start_pref_pf = start_pref_pf
        self.limit_pref_pf = limit_pref_pf
        self.start_pref_gf = start_pref_gf
        self.limit_pref_gf = limit_pref_gf
        self.relative_f = relative_f
        self.enable_atom_ener_coeff = enable_atom_ener_coeff
        self.numb_generalized_coord = numb_generalized_coord
        if self.has_gf and self.numb_generalized_coord < 1:
            raise RuntimeError(
                "When generalized force loss is used, the dimension of generalized coordinates should be larger than 0"
            )
        self.use_l1_all = use_l1_all
        self.inference = inference
        self.use_huber = use_huber
        self.huber_delta = (
            [huber_delta] if isinstance(huber_delta, float) else huber_delta
        )
        self.huber_two_stage_delta = (
            [huber_two_stage_delta]
            if isinstance(huber_two_stage_delta, float)
            else huber_two_stage_delta
        )
        self.huber_two_stage_use_root = huber_two_stage_use_root
        if self.use_huber and self.huber_two_stage_delta is not None:
            assert len(self.huber_two_stage_delta) == len(self.huber_delta), (
                "When using two-stage Huber loss, the length of huber_two_stage_delta must match that of huber_delta."
            )
            self.k1 = []
            self.k2 = []
            for i, delta in enumerate(self.huber_delta):
                two_stage_delta = self.huber_two_stage_delta[i]
                k1, k2 = (
                    cauchy_params_from_huber(delta, two_stage_delta)
                    if not self.huber_two_stage_use_root
                    else root_params_from_huber(delta, two_stage_delta)
                )
                self.k1.append(k1)
                self.k2.append(k2)
        else:
            self.k1 = [0.0 for _ in self.huber_delta]
            self.k2 = [0.0 for _ in self.huber_delta]
        if self.use_huber and (
            self.has_pf or self.has_gf or self.relative_f is not None
        ):
            raise RuntimeError(
                "Huber loss is not implemented for force with atom_pref, generalized force and relative force. "
            )
        self.trimmed_factor = trimmed_factor
        self.adaptive_loss = adaptive_loss
        if self.adaptive_loss:
            self.alpha_raw = torch.nn.Parameter(
                torch.tensor(2.0, dtype=torch.float32, device=env.DEVICE)
            )
            self.scale_raw = torch.nn.Parameter(
                torch.tensor(1.0, dtype=torch.float32, device=env.DEVICE)
            )
            self.epsilon = 1e-6
        else:
            self.alpha_raw = None
            self.scale_raw = None
            self.epsilon = None
        self.learnable_pref = learnable_pref
        if self.learnable_pref:
            self.pref_linear = torch.nn.Linear(
                1, 1, dtype=torch.float64, device=env.DEVICE
            )
            initial_weight = -0.1
            initial_bias = 5.0
            # 1. 权重 W 必须是负值: 让 r^2 越大, logit (W*r^2 + b) 越小
            torch.nn.init.constant_(self.pref_linear.weight, initial_weight)
            # 2. 偏差 b 必须是正值: 确保 r^2 小时, logit > 0, sigmoid 接近 1
            torch.nn.init.constant_(self.pref_linear.bias, initial_bias)

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """Return loss on energy and force.

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
        model_pred = model(**input_dict)

        if "force" not in model_pred and "dforce" in model_pred:
            model_pred["force"] = model_pred["dforce"]
        coef = learning_rate / self.starter_learning_rate
        pref_e = self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * coef
        pref_f = self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * coef
        pref_v = self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * coef
        pref_ae = self.limit_pref_ae + (self.start_pref_ae - self.limit_pref_ae) * coef
        pref_pf = self.limit_pref_pf + (self.start_pref_pf - self.limit_pref_pf) * coef
        pref_gf = self.limit_pref_gf + (self.start_pref_gf - self.limit_pref_gf) * coef

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}
        huber_index = 0
        # more_loss['log_keys'] = []  # showed when validation on the fly
        # more_loss['test_keys'] = []  # showed when doing dp test
        atom_norm = 1.0 / natoms
        if self.has_e and "energy" in model_pred and "energy" in label:
            energy_pred = model_pred["energy"]
            energy_label = label["energy"]
            if self.enable_atom_ener_coeff and "atom_energy" in model_pred:
                atom_ener_pred = model_pred["atom_energy"]
                # when ener_coeff (\nu) is defined, the energy is defined as
                # E = \sum_i \nu_i E_i
                # instead of the sum of atomic energies.
                #
                # A case is that we want to train reaction energy
                # A + B -> C + D
                # E = - E(A) - E(B) + E(C) + E(D)
                # A, B, C, D could be put far away from each other
                atom_ener_coeff = label["atom_ener_coeff"]
                atom_ener_coeff = atom_ener_coeff.reshape(atom_ener_pred.shape)
                energy_pred = torch.sum(atom_ener_coeff * atom_ener_pred, dim=1)
            find_energy = label.get("find_energy", 0.0)
            pref_e = pref_e * find_energy
            if not self.use_l1_all:
                l2_ener_loss = torch.mean(torch.square(energy_pred - energy_label))
                if not self.inference:
                    more_loss["l2_ener_loss"] = self.display_if_exist(
                        l2_ener_loss.detach(), find_energy
                    )
                if self.adaptive_loss:
                    assert (
                        self.alpha_raw is not None
                        and self.scale_raw is not None
                        and self.epsilon is not None
                    )
                    alpha = 2.0 * torch.sigmoid(self.alpha_raw)
                    scale = torch.nn.functional.softplus(self.scale_raw) + self.epsilon
                    residual = (energy_pred - energy_label).reshape(-1)
                    # 标准化残差 (x / c)
                    scaled_x = residual / scale
                    # 平方项 (x/c)^2
                    squared_scaled_x = scaled_x**2
                    b = torch.abs(alpha - 2.0) + self.epsilon
                    d = alpha + self.epsilon  # 避免 alpha=0
                    loss_val = (b / d) * (
                        torch.pow((squared_scaled_x / b) + 1, d / 2.0) - 1.0
                    )
                    nll_loss = loss_val + torch.log(scale)
                    nll_loss = torch.mean(nll_loss)
                    loss += atom_norm * (pref_e * nll_loss)
                elif self.learnable_pref:
                    assert self.pref_linear is not None
                    l2_ener = torch.square(energy_pred - energy_label)
                    r_sq_input = l2_ener.view(-1, 1)
                    logit = self.pref_linear(r_sq_input)
                    # 计算权重因子 (0 < weight_factor < 1)
                    # 由于初始化时 W < 0, r^2 越大, logit 越小, weight_factor 越小, 实现了降权
                    weight_factor = torch.sigmoid(logit / 1000) + 0.1  # 0.1防止都学为0
                    weighted_err = weight_factor.view_as(l2_ener) * l2_ener
                    weighted_l2 = torch.mean(weighted_err) + torch.sum(
                        torch.abs(self.pref_linear.weight**2)
                    )
                    loss += atom_norm * (pref_e * weighted_l2)
                else:
                    if not self.use_huber:
                        loss += atom_norm * (pref_e * l2_ener_loss)
                    else:
                        used_index = min(huber_index, len(self.huber_delta) - 1)
                        l_huber_loss = custom_huber_loss(
                            atom_norm * model_pred["energy"],
                            atom_norm * label["energy"],
                            delta=self.huber_delta[used_index],
                            delta2=self.huber_two_stage_delta[used_index]
                            if self.huber_two_stage_delta is not None
                            else None,
                            k1=self.k1[used_index],
                            k2=self.k2[used_index],
                            use_root=self.huber_two_stage_use_root,
                        )
                        huber_index += 1
                        loss += pref_e * l_huber_loss
                rmse_e = l2_ener_loss.sqrt() * atom_norm
                more_loss["rmse_e"] = self.display_if_exist(
                    rmse_e.detach(), find_energy
                )
                # more_loss['log_keys'].append('rmse_e')
            else:  # use l1 and for all atoms
                l1_ener_loss = F.l1_loss(
                    energy_pred.reshape(-1),
                    energy_label.reshape(-1),
                    reduction="sum",
                )
                loss += pref_e * l1_ener_loss
                more_loss["mae_e"] = self.display_if_exist(
                    F.l1_loss(
                        energy_pred.reshape(-1),
                        energy_label.reshape(-1),
                        reduction="mean",
                    ).detach(),
                    find_energy,
                )
                # more_loss['log_keys'].append('rmse_e')
            if mae:
                mae_e = torch.mean(torch.abs(energy_pred - energy_label)) * atom_norm
                more_loss["mae_e"] = self.display_if_exist(mae_e.detach(), find_energy)
                mae_e_all = torch.mean(torch.abs(energy_pred - energy_label))
                more_loss["mae_e_all"] = self.display_if_exist(
                    mae_e_all.detach(), find_energy
                )

        if (
            (self.has_f or self.has_pf or self.relative_f or self.has_gf)
            and "force" in model_pred
            and "force" in label
        ):
            find_force = label.get("find_force", 0.0)
            pref_f = pref_f * find_force
            force_pred = model_pred["force"]
            force_label = label["force"]
            diff_f = (force_label - force_pred).reshape(-1)
            force_pred_reshape = force_pred.reshape(-1)
            force_label_reshape = force_label.reshape(-1)

            if self.trimmed_factor > 0.0:
                num_samples = diff_f.numel()
                num_keep = int(num_samples * (1 - self.trimmed_factor))
                keep_values, mask = torch.topk(diff_f.abs(), k=num_keep, largest=False)
                diff_f = diff_f[mask]
                force_pred_reshape = force_pred_reshape[mask]
                force_label_reshape = force_label_reshape[mask]

            if self.relative_f is not None:
                force_label_3 = force_label.reshape(-1, 3)
                norm_f = force_label_3.norm(dim=1, keepdim=True) + self.relative_f
                diff_f_3 = diff_f.reshape(-1, 3)
                diff_f_3 = diff_f_3 / norm_f
                diff_f = diff_f_3.reshape(-1)

            if self.has_f:
                if not self.use_l1_all:
                    l2_force_loss = torch.mean(torch.square(diff_f))
                    if not self.inference:
                        more_loss["l2_force_loss"] = self.display_if_exist(
                            l2_force_loss.detach(), find_force
                        )
                    if self.adaptive_loss:
                        assert (
                            self.alpha_raw is not None
                            and self.scale_raw is not None
                            and self.epsilon is not None
                        )
                        alpha = 2.0 * torch.sigmoid(self.alpha_raw)
                        scale = (
                            torch.nn.functional.softplus(self.scale_raw) + self.epsilon
                        )
                        residual = diff_f
                        # 标准化残差 (x / c)
                        scaled_x = residual / scale
                        # 平方项 (x/c)^2
                        squared_scaled_x = scaled_x**2
                        b = torch.abs(alpha - 2.0) + self.epsilon
                        d = alpha + self.epsilon  # 避免 alpha=0
                        loss_val = (b / d) * (
                            torch.pow((squared_scaled_x / b) + 1, d / 2.0) - 1.0
                        )
                        nll_loss = loss_val + torch.log(scale)
                        nll_loss = torch.mean(nll_loss)
                        loss += (pref_f * nll_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                    elif self.learnable_pref:
                        assert self.pref_linear is not None
                        l2_force = torch.square(diff_f)
                        r_sq_input = l2_force.view(-1, 1)
                        logit = self.pref_linear(r_sq_input)
                        # 计算权重因子 (0 < weight_factor < 1)
                        # 由于初始化时 W < 0, r^2 越大, logit 越小, weight_factor 越小, 实现了降权
                        weight_factor = torch.sigmoid(logit) + 0.1  # 0.1防止都学为0
                        weighted_err = weight_factor.view_as(l2_force) * l2_force
                        weighted_l2 = torch.mean(weighted_err) + torch.sum(
                            torch.abs(self.pref_linear.weight**2)
                        )
                        loss += (pref_f * weighted_l2).to(GLOBAL_PT_FLOAT_PRECISION)
                    else:
                        if not self.use_huber:
                            loss += (pref_f * l2_force_loss).to(
                                GLOBAL_PT_FLOAT_PRECISION
                            )
                        else:
                            used_index = min(huber_index, len(self.huber_delta) - 1)
                            l_huber_loss = custom_huber_loss(
                                force_pred_reshape,
                                force_label_reshape,
                                delta=self.huber_delta[used_index],
                                delta2=self.huber_two_stage_delta[used_index]
                                if self.huber_two_stage_delta is not None
                                else None,
                                k1=self.k1[used_index],
                                k2=self.k2[used_index],
                                use_root=self.huber_two_stage_use_root,
                            )
                            huber_index += 1
                            loss += pref_f * l_huber_loss
                    rmse_f = l2_force_loss.sqrt()
                    more_loss["rmse_f"] = self.display_if_exist(
                        rmse_f.detach(), find_force
                    )
                else:
                    l1_force_loss = F.l1_loss(
                        force_label_reshape, force_pred_reshape, reduction="none"
                    )
                    more_loss["mae_f"] = self.display_if_exist(
                        l1_force_loss.mean().detach(), find_force
                    )
                    l1_force_loss = l1_force_loss.sum(-1).mean(-1).sum()
                    loss += (pref_f * l1_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                if mae:
                    mae_f = torch.mean(torch.abs(diff_f))
                    more_loss["mae_f"] = self.display_if_exist(
                        mae_f.detach(), find_force
                    )

            if self.has_pf and "atom_pref" in label:
                atom_pref = label["atom_pref"]
                find_atom_pref = label.get("find_atom_pref", 0.0)
                pref_pf = pref_pf * find_atom_pref
                atom_pref_reshape = atom_pref.reshape(-1)
                l2_pref_force_loss = (torch.square(diff_f) * atom_pref_reshape).mean()
                if not self.inference:
                    more_loss["l2_pref_force_loss"] = self.display_if_exist(
                        l2_pref_force_loss.detach(), find_atom_pref
                    )
                loss += (pref_pf * l2_pref_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                rmse_pf = l2_pref_force_loss.sqrt()
                more_loss["rmse_pf"] = self.display_if_exist(
                    rmse_pf.detach(), find_atom_pref
                )

            if self.has_gf and "drdq" in label:
                drdq = label["drdq"]
                find_drdq = label.get("find_drdq", 0.0)
                pref_gf = pref_gf * find_drdq
                force_reshape_nframes = force_pred.reshape(-1, natoms * 3)
                force_label_reshape_nframes = force_label.reshape(-1, natoms * 3)
                drdq_reshape = drdq.reshape(-1, natoms * 3, self.numb_generalized_coord)
                gen_force_label = torch.einsum(
                    "bij,bi->bj", drdq_reshape, force_label_reshape_nframes
                )
                gen_force = torch.einsum(
                    "bij,bi->bj", drdq_reshape, force_reshape_nframes
                )
                diff_gen_force = gen_force_label - gen_force
                l2_gen_force_loss = torch.square(diff_gen_force).mean()
                if not self.inference:
                    more_loss["l2_gen_force_loss"] = self.display_if_exist(
                        l2_gen_force_loss.detach(), find_drdq
                    )
                loss += (pref_gf * l2_gen_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                rmse_gf = l2_gen_force_loss.sqrt()
                more_loss["rmse_gf"] = self.display_if_exist(
                    rmse_gf.detach(), find_drdq
                )

        if self.has_v and "virial" in model_pred and "virial" in label:
            find_virial = label.get("find_virial", 0.0)
            pref_v = pref_v * find_virial
            diff_v = label["virial"] - model_pred["virial"].reshape(-1, 9)
            l2_virial_loss = torch.mean(torch.square(diff_v))
            if not self.inference:
                more_loss["l2_virial_loss"] = self.display_if_exist(
                    l2_virial_loss.detach(), find_virial
                )
            if not self.use_huber:
                loss += atom_norm * (pref_v * l2_virial_loss)
            else:
                used_index = min(huber_index, len(self.huber_delta) - 1)
                l_huber_loss = custom_huber_loss(
                    atom_norm * model_pred["virial"].reshape(-1),
                    atom_norm * label["virial"].reshape(-1),
                    delta=self.huber_delta[used_index],
                    delta2=self.huber_two_stage_delta[used_index]
                    if self.huber_two_stage_delta is not None
                    else None,
                    k1=self.k1[used_index],
                    k2=self.k2[used_index],
                    use_root=self.huber_two_stage_use_root,
                )
                huber_index += 1
                loss += pref_v * l_huber_loss
            rmse_v = l2_virial_loss.sqrt() * atom_norm
            more_loss["rmse_v"] = self.display_if_exist(rmse_v.detach(), find_virial)
            if mae:
                mae_v = torch.mean(torch.abs(diff_v)) * atom_norm
                more_loss["mae_v"] = self.display_if_exist(mae_v.detach(), find_virial)

        if self.has_ae and "atom_energy" in model_pred and "atom_ener" in label:
            atom_ener = model_pred["atom_energy"]
            atom_ener_label = label["atom_ener"]
            find_atom_ener = label.get("find_atom_ener", 0.0)
            pref_ae = pref_ae * find_atom_ener
            atom_ener_reshape = atom_ener.reshape(-1)
            atom_ener_label_reshape = atom_ener_label.reshape(-1)
            l2_atom_ener_loss = torch.square(
                atom_ener_label_reshape - atom_ener_reshape
            ).mean()
            if not self.inference:
                more_loss["l2_atom_ener_loss"] = self.display_if_exist(
                    l2_atom_ener_loss.detach(), find_atom_ener
                )
            if not self.use_huber:
                loss += (pref_ae * l2_atom_ener_loss).to(GLOBAL_PT_FLOAT_PRECISION)
            else:
                used_index = min(huber_index, len(self.huber_delta) - 1)
                l_huber_loss = custom_huber_loss(
                    atom_ener_reshape,
                    atom_ener_label_reshape,
                    delta=self.huber_delta[used_index],
                    delta2=self.huber_two_stage_delta[used_index]
                    if self.huber_two_stage_delta is not None
                    else None,
                    k1=self.k1[used_index],
                    k2=self.k2[used_index],
                    use_root=self.huber_two_stage_use_root,
                )
                huber_index += 1
                loss += pref_ae * l_huber_loss
            rmse_ae = l2_atom_ener_loss.sqrt()
            more_loss["rmse_ae"] = self.display_if_exist(
                rmse_ae.detach(), find_atom_ener
            )

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
        if self.has_f:
            label_requirement.append(
                DataRequirementItem(
                    "force",
                    ndof=3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_v:
            label_requirement.append(
                DataRequirementItem(
                    "virial",
                    ndof=9,
                    atomic=False,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_ae:
            label_requirement.append(
                DataRequirementItem(
                    "atom_ener",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_pf:
            label_requirement.append(
                DataRequirementItem(
                    "atom_pref",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    repeat=3,
                )
            )
        if self.has_gf > 0:
            label_requirement.append(
                DataRequirementItem(
                    "drdq",
                    ndof=self.numb_generalized_coord * 3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.enable_atom_ener_coeff:
            label_requirement.append(
                DataRequirementItem(
                    "atom_ener_coeff",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    default=1.0,
                )
            )
        return label_requirement

    def serialize(self) -> dict:
        """Serialize the loss module.

        Returns
        -------
        dict
            The serialized loss module
        """
        return {
            "@class": "EnergyLoss",
            "@version": 2,
            "starter_learning_rate": self.starter_learning_rate,
            "start_pref_e": self.start_pref_e,
            "limit_pref_e": self.limit_pref_e,
            "start_pref_f": self.start_pref_f,
            "limit_pref_f": self.limit_pref_f,
            "start_pref_v": self.start_pref_v,
            "limit_pref_v": self.limit_pref_v,
            "start_pref_ae": self.start_pref_ae,
            "limit_pref_ae": self.limit_pref_ae,
            "start_pref_pf": self.start_pref_pf,
            "limit_pref_pf": self.limit_pref_pf,
            "relative_f": self.relative_f,
            "enable_atom_ener_coeff": self.enable_atom_ener_coeff,
            "start_pref_gf": self.start_pref_gf,
            "limit_pref_gf": self.limit_pref_gf,
            "numb_generalized_coord": self.numb_generalized_coord,
            "use_huber": self.use_huber,
            "huber_delta": self.huber_delta,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "TaskLoss":
        """Deserialize the loss module.

        Parameters
        ----------
        data : dict
            The serialized loss module

        Returns
        -------
        Loss
            The deserialized loss module
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 2, 1)
        data.pop("@class")
        return cls(**data)


class EnergyHessianStdLoss(EnergyStdLoss):
    def __init__(
        self,
        start_pref_h: float = 0.0,
        limit_pref_h: float = 0.0,
        **kwargs: Any,
    ) -> None:
        r"""Enable the layer to compute loss on hessian.

        Parameters
        ----------
        start_pref_h : float
            The prefactor of hessian loss at the start of the training.
        limit_pref_h : float
            The prefactor of hessian loss at the end of the training.
        **kwargs
            Other keyword arguments.
        """
        super().__init__(**kwargs)
        self.has_h = (start_pref_h != 0.0 and limit_pref_h != 0.0) or self.inference

        self.start_pref_h = start_pref_h
        self.limit_pref_h = limit_pref_h

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        model_pred, loss, more_loss = super().forward(
            input_dict, model, label, natoms, learning_rate, mae=mae
        )
        coef = learning_rate / self.starter_learning_rate
        pref_h = self.limit_pref_h + (self.start_pref_h - self.limit_pref_h) * coef

        if self.has_h and "hessian" in model_pred and "hessian" in label:
            find_hessian = label.get("find_hessian", 0.0)
            pref_h = pref_h * find_hessian
            diff_h = label["hessian"].reshape(
                -1,
            ) - model_pred["hessian"].reshape(
                -1,
            )
            l2_hessian_loss = torch.mean(torch.square(diff_h))
            if not self.inference:
                more_loss["l2_hessian_loss"] = self.display_if_exist(
                    l2_hessian_loss.detach(), find_hessian
                )
            loss += pref_h * l2_hessian_loss
            rmse_h = l2_hessian_loss.sqrt()
            more_loss["rmse_h"] = self.display_if_exist(rmse_h.detach(), find_hessian)
            if mae:
                mae_h = torch.mean(torch.abs(diff_h))
                more_loss["mae_h"] = self.display_if_exist(mae_h.detach(), find_hessian)

        if not self.inference:
            more_loss["rmse"] = torch.sqrt(loss.detach())
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Add hessian label requirement needed for this loss calculation."""
        label_requirement = super().label_requirement
        if self.has_h:
            label_requirement.append(
                DataRequirementItem(
                    "hessian",
                    ndof=1,  # 9=3*3 --> 3N*3N=ndof*natoms*natoms
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        return label_requirement
