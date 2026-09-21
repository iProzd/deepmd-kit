# SPDX-License-Identifier: LGPL-3.0-or-later
"""vesin's pair-capacity error should carry the descriptor context it lacks.

``vesin`` caps the neighbor pairs its CUDA kernel stores per point, and its own
message already names ``VESIN_CUDA_MAX_PAIRS_PER_POINT`` and suggests reducing
the cutoff. What it cannot say is which value to use: the bound that matters is
the descriptor's ``sel``, which is a deepmd concept. These tests pin that the
annotation adds that context and leaves every other error alone.
"""

import unittest

from deepmd.pt_expt.utils.vesin_graph_builder import (
    _annotate_max_pairs_overflow,
)

# the wording vesin itself emits, abridged
VESIN_OVERFLOW = (
    "The number of neighbor pairs exceeds the maximum capacity of 216. "
    "Consider reducing the cutoff distance, or explicitly setting "
    "VESIN_CUDA_MAX_PAIRS_PER_POINT as an environment variable."
)


class TestVesinMaxPairsMessage(unittest.TestCase):
    def test_overflow_gains_descriptor_context(self) -> None:
        out = _annotate_max_pairs_overflow(RuntimeError(VESIN_OVERFLOW), 1, 6.0)
        text = str(out)
        # vesin's own wording is preserved, not replaced
        self.assertIn("exceeds the maximum capacity", text)
        self.assertIn("VESIN_CUDA_MAX_PAIRS_PER_POINT", text)
        # and the part only deepmd can supply
        self.assertIn("sel", text)
        self.assertIn("rcut=6", text)
        self.assertIn("1-atom", text)

    def test_other_runtime_errors_pass_through_unchanged(self) -> None:
        """Only the capacity error is annotated; nothing else is touched."""
        original = RuntimeError("CUDA out of memory")
        self.assertIs(_annotate_max_pairs_overflow(original, 64, 6.0), original)

    def test_annotation_does_not_change_the_exception_type(self) -> None:
        out = _annotate_max_pairs_overflow(RuntimeError(VESIN_OVERFLOW), 8, 5.0)
        self.assertIsInstance(out, RuntimeError)


if __name__ == "__main__":
    unittest.main()
