# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

import torch

log = logging.getLogger(__name__)


def compute_smooth_weight(distance, rmin: float, rmax: float):
    """Compute smooth weight for descriptor elements."""
    if rmin >= rmax:
        raise ValueError("rmin should be less than rmax.")
    distance = torch.clamp(distance, min=rmin, max=rmax)
    uu = (distance - rmin) / (rmax - rmin)
    uu2 = uu * uu
    vv = uu2 * uu * (-6 * uu2 + 15 * uu - 10) + 1
    return vv


def compute_envelope(distance, rmin: float, rmax: float):
    if rmin >= rmax:
        raise ValueError("rmin should be less than rmax.")
    distance = torch.clamp(distance, min=0.0, max=rmax)
    cutoff = rmax
    p = rmax - rmin
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2

    r_scaled = distance / cutoff
    env_val = 1 + a * r_scaled**p + b * r_scaled ** (p + 1) + c * r_scaled ** (p + 2)
    return env_val
