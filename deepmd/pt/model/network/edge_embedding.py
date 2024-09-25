# SPDX-License-Identifier: LGPL-3.0-or-later
import math

import torch
import torch.nn as nn


class BesselBasis(nn.Module):
    """f : (*, 1) -> (*, bessel_basis_num)."""

    def __init__(
        self,
        cutoff_length: float,
        bessel_basis_num: int = 8,
        trainable_coeff: bool = True,
    ):
        super().__init__()
        self.num_basis = bessel_basis_num
        self.prefactor = 2.0 / cutoff_length
        self.coeffs = torch.FloatTensor(
            [n * math.pi / cutoff_length for n in range(1, bessel_basis_num + 1)]
        )
        if trainable_coeff:
            self.coeffs = nn.Parameter(self.coeffs)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        ur = r.unsqueeze(-1)  # to fit dimension
        return self.prefactor * torch.sin(self.coeffs * ur) / (ur + 1e-8)


class PolynomialCutoff(nn.Module):
    """f : (*, 1) -> (*, 1). https://arxiv.org/pdf/2003.03123.pdf."""

    def __init__(
        self,
        cutoff_length: float,
        poly_cut_p_value: int = 6,
    ):
        super().__init__()
        p = poly_cut_p_value
        self.cutoff_length = cutoff_length
        self.p = p
        self.coeff_p0 = (p + 1.0) * (p + 2.0) / 2.0
        self.coeff_p1 = p * (p + 2.0)
        self.coeff_p2 = p * (p + 1.0) / 2.0

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r = r / self.cutoff_length
        return (
            1
            - self.coeff_p0 * torch.pow(r, self.p)
            + self.coeff_p1 * torch.pow(r, self.p + 1.0)
            - self.coeff_p2 * torch.pow(r, self.p + 2.0)
        )


class XPLORCutoff(nn.Module):
    """https://hoomd-blue.readthedocs.io/en/latest/module-md-pair.html."""

    def __init__(
        self,
        cutoff_length: float,
        cutoff_on: float,
    ):
        super().__init__()
        self.r_on = cutoff_on
        self.r_cut = cutoff_length
        assert self.r_on < self.r_cut

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r_sq = r * r
        r_on_sq = self.r_on * self.r_on
        r_cut_sq = self.r_cut * self.r_cut
        return torch.where(
            r < self.r_on,
            1.0,
            (r_cut_sq - r_sq) ** 2
            * (r_cut_sq + 2 * r_sq - 3 * r_on_sq)
            / (r_cut_sq - r_on_sq) ** 3,
        )
