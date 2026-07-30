from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from c3d4_constraints import (  # noqa: E402
    chebyshev_fit_d4_constraint,
    fixed_c3_scan_constraint,
    format_d4_constraint,
)


class FixedC3ScanConstraintTests(unittest.TestCase):
    def test_log_linear_slice_finds_bounded_allowed_interval(self):
        rows = [
            {"c3": 0.0, "d4": -2.0, "xsec_fb": 2.0, "shape_sigma95_fb": 1.0},
            {"c3": 0.0, "d4": 0.0, "xsec_fb": 0.5, "shape_sigma95_fb": 1.0},
            {"c3": 0.0, "d4": 2.0, "xsec_fb": 2.0, "shape_sigma95_fb": 1.0},
            {"c3": 1.0, "d4": 0.0, "xsec_fb": 100.0, "shape_sigma95_fb": 1.0},
        ]

        result = fixed_c3_scan_constraint(
            rows,
            limit_key="shape_sigma95_fb",
            strategy="sm-crossfit-v2",
            limit_kind="pyhf CLs shape",
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["constraint_type"], "bounded_interval")
        self.assertFalse(result["scan_boundary_limited"])
        self.assertEqual(len(result["allowed_intervals"]), 1)
        self.assertAlmostEqual(result["lower_95cl"], -1.0)
        self.assertAlmostEqual(result["upper_95cl"], 1.0)
        self.assertEqual(result["crossings_d4"], [-1.0, 1.0])

    def test_multiple_allowed_intervals_preserve_scan_limited_edges(self):
        rows = [
            {"c3": 0.0, "d4": -2.0, "xsec_fb": 0.5, "cut_sigma95_fb": 1.0},
            {"c3": 0.0, "d4": 0.0, "xsec_fb": 2.0, "cut_sigma95_fb": 1.0},
            {"c3": 0.0, "d4": 2.0, "xsec_fb": 0.5, "cut_sigma95_fb": 1.0},
        ]

        result = fixed_c3_scan_constraint(
            rows,
            limit_key="cut_sigma95_fb",
            limit_kind="exact CLs cut",
        )

        self.assertEqual(result["constraint_type"], "multiple_intervals")
        self.assertTrue(result["scan_boundary_limited"])
        self.assertEqual(
            [
                (interval["lower"], interval["upper"])
                for interval in result["allowed_intervals"]
            ],
            [(-2.0, -1.0), (1.0, 2.0)],
        )

    def test_missing_fixed_slice_limit_is_reported_instead_of_skipped(self):
        rows = [
            {"c3": 0.0, "d4": -1.0, "xsec_fb": 1.0, "shape_sigma95_fb": 2.0},
            {"c3": 0.0, "d4": 1.0, "xsec_fb": 1.0, "shape_sigma95_fb": None},
        ]

        result = fixed_c3_scan_constraint(
            rows, limit_key="shape_sigma95_fb"
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("missing or non-positive", result["reason"])


class ChebyshevConstraintTests(unittest.TestCase):
    def test_chebyshev_slice_solves_exact_crossings(self):
        # On this domain x4=d4/2 and 0.5*T0+0.5*T2=x4^2.
        fit = {
            "status": "ok",
            "terms": [[0, 0], [0, 2]],
            "coefficients": [0.5, 0.5],
            "k3_range": [-1.0, 3.0],
            "k4_range": [-1.0, 3.0],
        }

        result = chebyshev_fit_d4_constraint(
            fit,
            target_value=0.25,
            strategy="legacy Chebyshev fit",
            limit_kind="Poisson CLs",
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["constraint_type"], "bounded_interval")
        self.assertAlmostEqual(result["lower_95cl"], -1.0)
        self.assertAlmostEqual(result["upper_95cl"], 1.0)
        self.assertAlmostEqual(result["target_effective_sigma_eff_fb"], 0.25)

    def test_terminal_formatter_labels_interval_and_method(self):
        result = {
            "status": "ok",
            "confidence_level": 0.95,
            "c3": 0.0,
            "strategy": "pooled-crossfit-v2",
            "limit_kind": "pyhf CLs shape",
            "allowed_intervals": [{"lower": -3.25, "upper": 4.5}],
            "scan_boundary_limited": False,
        }

        text = format_d4_constraint(result)

        self.assertIn("Expected 95% C.L. constraint on d4 for c3 = 0", text)
        self.assertIn("pooled-crossfit-v2, pyhf CLs shape", text)
        self.assertIn("allowed d4 = [-3.25, 4.5]", text)


if __name__ == "__main__":
    unittest.main()
