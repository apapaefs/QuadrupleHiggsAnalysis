from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "resonance_xgboost_analysis.py"
SPEC = importlib.util.spec_from_file_location("resonance_xgboost_analysis_bounded", MODULE_PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def bisection(function, lower: float, upper: float) -> float:
    lower_value = float(function(lower))
    upper_value = float(function(upper))
    if lower_value == 0.0:
        return lower
    if upper_value == 0.0:
        return upper
    if lower_value * upper_value > 0.0:
        raise ValueError("test root is not bracketed")
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        midpoint_value = float(function(midpoint))
        if abs(midpoint_value) < 1.0e-12:
            return midpoint
        if lower_value * midpoint_value > 0.0:
            lower = midpoint
            lower_value = midpoint_value
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


class BoundedCurveRootTests(unittest.TestCase):
    def test_optional_non_crossing_band_returns_partial_and_keeps_median(self) -> None:
        calls: list[float] = []
        slopes = np.asarray([0.10, 0.20, 0.125, 0.10, 0.0625, 0.04])

        def evaluate(poi: float) -> np.ndarray:
            calls.append(float(poi))
            return 0.10 - slopes * poi

        result = analysis._bounded_cls_roots(
            evaluate,
            bisection,
            model_poi_bounds=(0.0, 1.0),
            hard_cap=1.0,
        )

        self.assertEqual(result["status"], "partial", result)
        self.assertAlmostEqual(result["curve_roots"]["observed_asimov"], 0.5)
        self.assertAlmostEqual(result["curve_roots"]["expected_median"], 0.5)
        self.assertIsNone(result["curve_roots"]["expected_plus2sigma"])
        self.assertEqual(
            result["curve_diagnostics"]["expected_plus2sigma"]["status"],
            "unbounded",
        )
        self.assertEqual(calls[0], 1.0, "the hard-cap preflight must run first")
        self.assertTrue(all(0.0 <= poi <= 1.0 for poi in calls))
        self.assertEqual(len(calls), len(set(calls)), "hypotest evaluations must be cached")

    def test_non_crossing_median_is_structured_unbounded(self) -> None:
        slopes = np.asarray([0.10, 0.20, 0.125, 0.04, 0.0625, 0.03])
        result = analysis._bounded_cls_roots(
            lambda poi: 0.10 - slopes * poi,
            bisection,
            model_poi_bounds=(0.0, 1.0),
            hard_cap=1.0,
        )

        self.assertEqual(result["status"], "unbounded", result)
        self.assertEqual(
            result["diagnostic_type"], "RequiredLimitCurveNotBracketed"
        )
        self.assertIsNone(result["curve_roots"]["expected_median"])
        self.assertEqual(
            result["curve_diagnostics"]["expected_median"]["status"],
            "unbounded",
        )

    def test_non_crossing_observed_curve_is_also_structured_unbounded(self) -> None:
        slopes = np.asarray([0.04, 0.20, 0.125, 0.10, 0.0625, 0.03])
        result = analysis._bounded_cls_roots(
            lambda poi: 0.10 - slopes * poi,
            bisection,
            model_poi_bounds=(0.0, 1.0),
            hard_cap=1.0,
        )

        self.assertEqual(result["status"], "unbounded", result)
        self.assertIsNone(result["curve_roots"]["observed_asimov"])
        self.assertAlmostEqual(result["curve_roots"]["expected_median"], 0.5)
        self.assertIn("observed_asimov", result["reason"])

    def test_out_of_bound_solver_probe_is_refused_without_calling_evaluator(self) -> None:
        calls: list[float] = []

        def evaluate(poi: float) -> np.ndarray:
            calls.append(float(poi))
            return np.full(6, 0.10 - 0.10 * poi)

        def bad_solver(function, lower: float, upper: float) -> float:
            return float(function(upper + 1.0))

        result = analysis._bounded_cls_roots(
            evaluate,
            bad_solver,
            model_poi_bounds=(0.0, 1.0),
            hard_cap=1.0,
        )

        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(result["diagnostic_type"], "RequiredLimitCurveFailure")
        self.assertTrue(all(0.0 <= poi <= 1.0 for poi in calls))
        self.assertIn(
            "outside",
            result["curve_diagnostics"]["expected_median"]["cause"]["message"],
        )

    def test_model_bound_takes_precedence_over_a_larger_configured_cap(self) -> None:
        calls: list[float] = []

        def evaluate(poi: float) -> np.ndarray:
            calls.append(float(poi))
            return np.full(6, 0.10 - 0.10 * poi)

        result = analysis._bounded_cls_roots(
            evaluate,
            bisection,
            model_poi_bounds=(0.0, 0.75),
            hard_cap=2.0,
        )

        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["effective_poi_hard_cap"], 0.75)
        self.assertEqual(calls[0], 0.75)
        self.assertTrue(all(poi <= 0.75 for poi in calls))

    def test_cli_exposes_configurable_positive_hard_cap(self) -> None:
        args = analysis.build_parser().parse_args(
            [
                "--topology",
                "direct",
                "--output-dir",
                "results",
                "--pyhf-poi-hard-cap",
                "6.5",
                "--pyhf-physical-limit-hard-cap-fb",
                "2500",
                "--pyhf-reference-max-attempts",
                "3",
            ]
        )
        self.assertEqual(args.pyhf_poi_hard_cap, 6.5)
        self.assertEqual(args.pyhf_physical_limit_hard_cap_fb, 2500.0)
        self.assertEqual(args.pyhf_reference_max_attempts, 3)


@unittest.skipUnless(
    os.environ.get("RUN_PYHF_INTEGRATION") == "1"
    and importlib.util.find_spec("pyhf") is not None,
    "set RUN_PYHF_INTEGRATION=1 with pyhf installed",
)
class PyhfBoundedIntegrationTests(unittest.TestCase):
    @staticmethod
    def channels() -> list[dict[str, object]]:
        return [
            {
                "name": "resolved_fold0",
                "signal": np.asarray([1.0, 2.0]),
                "background": np.asarray([10.0, 5.0]),
                "signal_staterror": np.asarray([0.1, 0.2]),
                "background_staterror": np.asarray([1.0, 0.7]),
            }
        ]

    def test_plus2_band_above_cap_preserves_observed_and_median(self) -> None:
        fit = analysis._pyhf_limit(self.channels(), poi_hard_cap=1.0)

        self.assertEqual(fit["status"], "partial", fit)
        self.assertTrue(math.isfinite(fit["observed_asimov"]))
        self.assertTrue(math.isfinite(fit["expected_median"]))
        self.assertIsNone(fit["expected_plus2sigma"])
        self.assertEqual(fit["mu_fit_bounds"], [0.0, 1.0])
        self.assertTrue(all(0.0 <= poi <= 1.0 for poi in fit["tested_poi_values"]))
        self.assertEqual(
            fit["curve_diagnostics"]["expected_plus2sigma"]["status"],
            "unbounded",
        )

    def test_median_above_cap_returns_unbounded_not_value_error(self) -> None:
        fit = analysis._pyhf_limit(
            self.channels(),
            poi_hard_cap=0.5,
            max_reference_attempts=1,
        )

        self.assertEqual(fit["status"], "unbounded", fit)
        self.assertEqual(
            fit["diagnostic_type"], "RequiredLimitCurveNotBracketed"
        )
        self.assertIsNone(fit["expected_median"])
        self.assertNotEqual(fit.get("error_type"), "ValueError")
        self.assertTrue(all(0.0 <= poi <= 0.5 for poi in fit["tested_poi_values"]))

    def test_unbounded_median_triggers_bounded_reference_rescaling(self) -> None:
        fit = analysis._pyhf_limit(self.channels(), poi_hard_cap=0.5)

        self.assertIn(fit["status"], {"ok", "partial"}, fit)
        self.assertGreater(len(fit["fit_attempts"]), 1)
        self.assertTrue(math.isfinite(fit["expected_median"]))
        for attempt in fit["fit_attempts"]:
            self.assertTrue(
                all(0.0 <= poi <= 0.5 for poi in attempt["tested_poi_values"])
            )

    def test_physical_cap_stops_rescaling_with_structured_diagnostic(self) -> None:
        fit = analysis._pyhf_limit(
            self.channels(),
            poi_hard_cap=0.5,
            physical_limit_hard_cap_fb=2.2,
            max_reference_attempts=8,
        )

        self.assertEqual(fit["status"], "unbounded", fit)
        self.assertEqual(fit["rescaling_stop_reason"], "physical_limit_hard_cap_reached")
        self.assertLessEqual(fit["physical_limit_cap_fb"], 2.2)


if __name__ == "__main__":
    unittest.main()
