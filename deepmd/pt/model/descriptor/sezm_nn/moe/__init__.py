# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM MoE (Mixture-of-Experts) modules for Expert Parallelism + Data Parallelism.

This package implements MoE components for the SeZM descriptor:
- Communication primitives (A2A with second-order derivatives)
- Router (top-k gating)
- Expert collections (routing + shared experts)
- MoE convolution layer (replaces SO2 linear stack)
"""

from __future__ import (
    annotations,
)

__all__ = []
