from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import resonance_one_bin_poisson as onebin


SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


class OneBinPoissonTests(unittest.TestCase):
    def test_q_zero_and_declared_crossing(self) -> None:
        self.assertEqual(onebin.poisson_q_one_bin(12.0, 3.0, 0.0), 0.0)
        limit = onebin.solve_sigma95_one_bin(4310.92654648924, 20.4534905669)
        self.assertAlmostEqual(
            onebin.poisson_q_one_bin(4310.92654648924, 20.4534905669, limit),
            onebin.Q95,
            places=10,
        )

    def test_limit_scales_inverse_to_signal_yield(self) -> None:
        nominal = onebin.solve_sigma95_one_bin(100.0, 4.0)
        doubled = onebin.solve_sigma95_one_bin(100.0, 8.0)
        self.assertAlmostEqual(doubled, nominal / 2.0, places=12)

    def test_zero_asimov_count_is_supported(self) -> None:
        limit = onebin.solve_sigma95_one_bin(0.0, 2.0)
        self.assertAlmostEqual(limit, onebin.Q95 / 4.0, places=12)

    def test_build_rows_fixes_exactly_one_bin_and_is_conservative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "points.csv"
            fields = [
                "point_id",
                "topology",
                "MS_GeV",
                "M2_GeV",
                "M3_GeV",
                "status",
                "selected_bins",
                "sigma95_fb",
                "signal_1fb_yield",
                "asimov_yield",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "point_id": "MS_0500",
                            "topology": "direct",
                            "MS_GeV": 500,
                            "M2_GeV": "",
                            "M3_GeV": "",
                            "status": "ok",
                            "selected_bins": 2,
                            "sigma95_fb": 1.0,
                            "signal_1fb_yield": 10.0,
                            "asimov_yield": 100.0,
                        },
                        {
                            "point_id": "MS_0600",
                            "topology": "direct",
                            "MS_GeV": 600,
                            "M2_GeV": "",
                            "M3_GeV": "",
                            "status": "ok",
                            "selected_bins": 1,
                            "sigma95_fb": 2.0,
                            "signal_1fb_yield": 8.0,
                            "asimov_yield": 100.0,
                        },
                    ]
                )
            with patch.dict(onebin.EXPECTED_POINTS, {"direct": 2}):
                rows = onebin.build_one_bin_rows("direct", path)
            self.assertEqual([row["fixed_bins"] for row in rows], [1, 1])
            self.assertTrue(
                all(
                    row["sigma95_fb"] >= row["source_scorefit_sigma95_fb"]
                    for row in rows
                )
            )

    def test_reference_yields_are_collapsed_to_total(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "yields.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["process", "kind", "score_bin_1", "score_bin_2", "total"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "process": "signal",
                        "kind": "signal",
                        "score_bin_1": 2.0,
                        "score_bin_2": 3.0,
                        "total": 5.0,
                    }
                )
            rows = onebin.collapse_reference_yields(path)
            self.assertEqual(rows[0]["inclusive_yield"], 5.0)
            self.assertNotIn("score_bin_1", rows[0])


@unittest.skipUnless(SCIPY_AVAILABLE, "scipy is not installed")
class DisplayTests(unittest.TestCase):
    def test_direct_curve_is_smooth_monotone_and_conservative(self) -> None:
        rows = [
            {"MS_GeV": 500.0, "sigma95_fb": 10.0},
            {"MS_GeV": 600.0, "sigma95_fb": 30.0},
            {"MS_GeV": 700.0, "sigma95_fb": 20.0},
            {"MS_GeV": 800.0, "sigma95_fb": 22.0},
            {"MS_GeV": 900.0, "sigma95_fb": 12.0},
        ]
        _, curve, audit = onebin.conservative_direct_curve(rows, grid_size=100)
        self.assertTrue(np.all(np.diff(curve) <= 1.0e-10))
        self.assertTrue(audit["conservative_at_every_generated_point"])
        self.assertGreater(audit["maximum_lift_log10"], 0.0)

    def test_smooth_surface_is_conservative_at_generated_points(self) -> None:
        coordinates = [
            (100.0, 300.0),
            (100.0, 500.0),
            (200.0, 500.0),
            (200.0, 700.0),
            (300.0, 700.0),
            (300.0, 900.0),
        ]
        rows = [
            {
                "M2_GeV": m2,
                "M3_GeV": m3,
                "sigma95_fb": 10.0 ** (0.001 * m2 + 0.0002 * m3),
            }
            for m2, m3 in coordinates
        ]
        _, _, surface, audit = onebin.conservative_cascade_surface(
            rows, smoothing_strength=1.0, grid_size=30
        )
        self.assertGreater(surface.count(), 0)
        self.assertTrue(audit["optimizer_success"])
        self.assertTrue(audit["conservative_at_every_generated_point"])
        self.assertLessEqual(audit["maximum_constraint_violation_log10"], 2.0e-9)


if __name__ == "__main__":
    unittest.main()
