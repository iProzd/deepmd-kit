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
    C = 20
    a = C / rmin
    b = rmin
    env_val = torch.exp(-torch.exp(a * (distance - b)))
    return env_val


def compute_new_weight(distance, rmin: float, rmax: float):
    """Compute smooth weight for descriptor elements."""
    if rmin >= rmax:
        raise ValueError("rmin should be less than rmax.")
    distance = torch.clamp(distance, min=rmin, max=rmax)
    uu = (distance - rmin) / (rmax - rmin)
    uu2 = uu * uu
    uu3 = uu * uu2
    vv = uu3 * uu * (20 * uu3 - 70 * uu2 + 84 * uu - 35) + 1
    return vv
