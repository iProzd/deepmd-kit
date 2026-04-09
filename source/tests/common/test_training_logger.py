# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.loggers.training import (
    format_grad_norm_message,
)


class TestFormatGradNormMessage(unittest.TestCase):
    """Test cases for the format_grad_norm_message function."""

    def test_single_task_format(self):
        """Test gradient norm message for single-task training."""
        result = format_grad_norm_message(batch=100, grad_norm=1.5e-3)
        self.assertEqual(result, "Batch     100: grad_norm = 1.50e-03")

    def test_single_task_small_batch(self):
        """Test gradient norm message with small batch number."""
        result = format_grad_norm_message(batch=1, grad_norm=2.5e-2)
        self.assertEqual(result, "Batch       1: grad_norm = 2.50e-02")

    def test_single_task_large_batch(self):
        """Test gradient norm message with large batch number."""
        result = format_grad_norm_message(batch=1000000, grad_norm=1.0e-5)
        self.assertEqual(result, "Batch 1000000: grad_norm = 1.00e-05")

    def test_multitask_format(self):
        """Test gradient norm message for multitask training."""
        result = format_grad_norm_message(
            batch=100, grad_norm={"task1": 1.5e-3, "task2": 2.0e-3}
        )
        self.assertEqual(
            result, "Batch     100: grad_norm: task1 = 1.50e-03, task2 = 2.00e-03"
        )

    def test_multitask_sorted_order(self):
        """Test that multitask gradient norms are sorted alphabetically."""
        result = format_grad_norm_message(
            batch=200, grad_norm={"zebra": 3.0e-4, "alpha": 1.0e-4, "beta": 2.0e-4}
        )
        # Should be sorted alphabetically
        self.assertEqual(
            result,
            "Batch     200: grad_norm: alpha = 1.00e-04, beta = 2.00e-04, zebra = 3.00e-04",
        )

    def test_multitask_single_task(self):
        """Test gradient norm message with single task in dict."""
        result = format_grad_norm_message(batch=50, grad_norm={"only_task": 1.2e-3})
        self.assertEqual(result, "Batch      50: grad_norm: only_task = 1.20e-03")

    def test_scientific_notation_very_small(self):
        """Test gradient norm message with very small values."""
        result = format_grad_norm_message(batch=100, grad_norm=1.0e-10)
        self.assertEqual(result, "Batch     100: grad_norm = 1.00e-10")

    def test_scientific_notation_large(self):
        """Test gradient norm message with large values."""
        result = format_grad_norm_message(batch=100, grad_norm=1.5e5)
        self.assertEqual(result, "Batch     100: grad_norm = 1.50e+05")

    def test_zero_grad_norm(self):
        """Test gradient norm message with zero gradient norm."""
        result = format_grad_norm_message(batch=100, grad_norm=0.0)
        self.assertEqual(result, "Batch     100: grad_norm = 0.00e+00")


if __name__ == "__main__":
    unittest.main()
