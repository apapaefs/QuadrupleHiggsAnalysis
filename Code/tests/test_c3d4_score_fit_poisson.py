from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np

import c3d4_score_fit_poisson as scorefit


def partition(sample_id: str, scores, weights, source_ids=None):
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if source_ids is None:
        source_ids = np.arange(len(scores), dtype=np.int64)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    sample = SimpleNamespace(
        sample_id=sample_id,
        source_event_ids=source_ids,
        event_indices=source_ids,
    )
    return {
        sample_id: {
            "sample": sample,
            "mask": np.ones(len(scores), dtype=bool),
            "scores": scores,
            "physical_weights": weights,
        }
    }


def toy_hh4b_fit():
    return {
        "schema": "sm-hh4b-c3-cross-section-fit-v1",
        "coefficient_order": ["constant", "linear", "quadratic"],
        "coefficients_pb": [1.0e-5, -1.0e-6, 2.0e-7],
        "covariance_pb2": [
            [1.0e-16, 0.0, 0.0],
            [0.0, 1.0e-16, 0.0],
            [0.0, 0.0, 1.0e-16],
        ],
        "evaluation_range_c3": [-20.0, 20.0],
        "points": [
            {"c3": -1.0, "cross_section_pb": 1.12e-5, "integration_error_pb": 1.0e-7},
            {"c3": 0.0, "cross_section_pb": 1.00e-5, "integration_error_pb": 1.0e-7},
            {"c3": 1.0, "cross_section_pb": 9.20e-6, "integration_error_pb": 1.0e-7},
        ],
    }


class ScoreFitPoissonTests(unittest.TestCase):
    def test_feature_profile_cli_defaults_to_core52_and_accepts_full91(self):
        required = [
            "--study-dir",
            "/tmp/study",
            "--repository",
            "/tmp/repository",
            "--output-dir",
            "/tmp/output",
        ]
        parser = scorefit.build_parser()
        self.assertEqual(parser.parse_args(required).feature_profile, "core52")
        self.assertEqual(
            parser.parse_args([*required, "--feature-profile", "full91"]).feature_profile,
            "full91",
        )

    def test_feature_profile_contract_widths_are_explicit(self):
        for profile, expected in scorefit.FEATURE_PROFILES.items():
            indices = scorefit._profile_indices("extended-91-v2", profile)
            self.assertEqual(len(indices), expected)

    def test_poisson_deviance_closes_and_handles_zero_observation(self):
        observed = np.asarray([0.0, 2.0, 5.0])
        self.assertAlmostEqual(
            scorefit.poisson_asimov_deviance(observed, observed), 0.0, places=13
        )
        expected = np.asarray([3.0, 2.0, 5.0])
        self.assertAlmostEqual(
            scorefit.poisson_asimov_deviance(observed, expected), 6.0, places=13
        )
        self.assertTrue(
            np.isinf(scorefit.poisson_asimov_deviance([1.0], [0.0]))
        )

    def test_binned_discovery_q_is_positive(self):
        value = scorefit.asimov_discovery_q([2.0, 4.0], [20.0, 5.0])
        self.assertGreater(value, 0.0)
        with self.assertRaisesRegex(ValueError, "positive background"):
            scorefit.asimov_discovery_q([1.0], [0.0])

    def test_fixed_coupling_interval_uses_one_parameter_95_percent_level(self):
        self.assertAlmostEqual(scorefit.FIXED_COUPLING_95_LEVEL, 3.841458820694124)
        crossings = scorefit._level_crossings_1d(
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [4.0, 1.0, 0.0, 1.0, 4.0],
            1.5,
        )
        np.testing.assert_allclose(crossings, [-7.0 / 6.0, 7.0 / 6.0])

    def test_optional_background_nuisance_is_disabled_exactly_at_zero(self):
        tested = np.asarray([5.0, 1.0])
        sm = np.asarray([2.0, 3.0])
        background = np.asarray([30.0, 10.0])
        direct = scorefit.poisson_asimov_deviance(sm + background, tested + background)
        q, eta = scorefit.profiled_poisson_q(
            tested, sm, background, background_norm_fraction=0.0
        )
        self.assertAlmostEqual(q, direct, places=13)
        self.assertEqual(eta, 0.0)

    def test_correlated_background_nuisance_can_only_improve_tested_fit(self):
        tested = np.asarray([1.0, 1.0])
        sm = np.asarray([4.0, 4.0])
        background = np.asarray([20.0, 20.0])
        fixed, _ = scorefit.profiled_poisson_q(
            tested, sm, background, background_norm_fraction=0.0
        )
        profiled, eta = scorefit.profiled_poisson_q(
            tested, sm, background, background_norm_fraction=0.20
        )
        self.assertLessEqual(profiled, fixed + 1.0e-10)
        self.assertNotEqual(eta, 0.0)
        sm_q, sm_eta = scorefit.profiled_poisson_q(
            sm, sm, background, background_norm_fraction=0.20
        )
        self.assertAlmostEqual(sm_q, 0.0, places=12)
        self.assertAlmostEqual(sm_eta, 0.0, places=12)

    def test_background_tail_bin_is_merged_downward_for_all_folds(self):
        folds = [
            partition("b0", [0.1, 0.2, 0.6, 0.95], [1.0, 1.0, 1.0, 1.0]),
            partition("b1", [0.1, 0.3, 0.7, 0.97], [1.0, 1.0, 1.0, 1.0]),
        ]
        initial = [np.asarray([0.0, 0.5, 0.9, 1.0])] * 2
        edges, history, summary = scorefit.merge_background_quantile_edges(
            folds,
            initial,
            min_source_events=3,
            min_neff=1.0,
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["failed_bin"], 2)
        np.testing.assert_allclose(edges[0], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(edges[1], [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(summary.source_events, [4, 4])

    def test_signed_background_weights_are_absolute_only_for_quantile_edges(self):
        scores, weights = scorefit._partition_scores_and_weights(
            partition("signed", [0.1, 0.9], [2.0, -0.5])
        )
        np.testing.assert_allclose(scores, [0.1, 0.9])
        np.testing.assert_allclose(weights, [2.0, 0.5])
        signed_histogram = scorefit._histogram_partition(
            partition("signed", [0.1, 0.9], [2.0, -0.5]),
            [0.0, 0.5, 1.0],
        )
        np.testing.assert_allclose(signed_histogram.yields, [2.0, -0.5])

    def test_scan_selection_retains_baseline_within_one_percent(self):
        def result(name, mean_q, q4, q5):
            return {
                "name": name,
                "mean_binning_q": mean_q,
                "schemes": {
                    "background_quantile_4bin": {"validation_discovery_q": q4},
                    "background_quantile_5bin": {"validation_discovery_q": q5},
                },
            }

        rows = [
            result("baseline", 99.2, 99.0, 99.4),
            result("depth2", 100.0, 99.6, 100.4),
        ]
        configuration, scheme, audit = scorefit.select_scan_result(rows)
        self.assertEqual(configuration, "baseline")
        self.assertEqual(scheme, "background_quantile_4bin")
        self.assertTrue(audit["baseline_retained_within_one_percent"])

    def test_scan_selection_uses_clear_improvement_and_five_bins(self):
        rows = [
            {
                "name": "baseline",
                "mean_binning_q": 90.0,
                "schemes": {
                    "background_quantile_4bin": {"validation_discovery_q": 89.0},
                    "background_quantile_5bin": {"validation_discovery_q": 91.0},
                },
            },
            {
                "name": "depth2",
                "mean_binning_q": 100.0,
                "schemes": {
                    "background_quantile_4bin": {"validation_discovery_q": 96.0},
                    "background_quantile_5bin": {"validation_discovery_q": 104.0},
                },
            },
        ]
        configuration, scheme, _ = scorefit.select_scan_result(rows)
        self.assertEqual(configuration, "depth2")
        self.assertEqual(scheme, "background_quantile_5bin")

    def test_hh4b_fixed_shape_rate_is_c3_only_and_sm_q_is_zero(self):
        fit = toy_hh4b_fit()
        points = np.asarray(
            [
                [0.0, 0.0],
                [1.0, -10.0],
                [1.0, 10.0],
                [-1.0, 0.0],
            ]
        )
        templates = {
            "points": points,
            "hhhh": np.asarray([[2.0, 1.0], [3.0, 1.0], [3.5, 1.0], [1.0, 2.0]]),
            "hhhbb": np.asarray([[0.5, 0.2], [0.6, 0.2], [0.7, 0.2], [0.4, 0.3]]),
            "hh4b": np.asarray([0.3, 0.1]),
            "background": np.asarray([20.0, 8.0]),
        }
        result = scorefit.evaluate_likelihood_points(
            templates, fit, background_norm_fraction=0.0
        )
        self.assertAlmostEqual(result["hh4b_rate_scales"][1], result["hh4b_rate_scales"][2])
        self.assertAlmostEqual(result["q"]["background_x1"][0], 0.0, places=13)
        self.assertTrue(np.all(result["q"]["background_x1"] >= 0.0))

    def test_selected_sm_template_table_prints_score_bins_and_closes(self):
        def sample(sample_id, weights, xsec, *, c3=None, d4=None, description=None):
            weights = np.asarray(weights, dtype=float)
            return SimpleNamespace(
                sample_id=sample_id,
                c3=c3,
                d4=d4,
                xsec_fb=float(xsec),
                physical_weights=weights,
                entries=len(weights),
                metadata={} if description is None else {"description": description},
            )

        hhhh = sample("hhhh_sm_grid", [0.2, 0.1], 0.01, c3=0.0, d4=0.0)
        hhhbb = sample("hhhbb_sm_grid", [0.04, 0.01], 0.002, c3=0.0, d4=0.0)
        hh4b = sample("hh4b_sm", [0.3, 0.1], 0.02, c3=0.0, d4=0.0)
        background = sample(
            "background_a", [12.0, 8.0], 1.0, description="example background"
        )
        templates = {
            "points": np.asarray([[0.0, 0.0]]),
            "hhhh": np.asarray([[0.2, 0.1]]),
            "hhhbb": np.asarray([[0.04, 0.01]]),
            "hh4b": np.asarray([0.3, 0.1]),
            "background": np.asarray([12.0, 8.0]),
            "background_source_events": np.asarray([120, 80]),
            "background_neff": np.asarray([100.0, 60.0]),
            "background_processes": {
                "background_a": {
                    "yields": np.asarray([12.0, 8.0]),
                    "source_events": np.asarray([120, 80]),
                    "neff": np.asarray([100.0, 60.0]),
                }
            },
        }
        rows = scorefit.selected_sm_template_table_rows(
            templates=templates,
            grid_samples=[hhhh],
            hhhbb_samples=[hhhbb],
            hh4b_sample=hh4b,
            background_samples=[background],
            luminosity=3000.0,
        )
        self.assertEqual(len(rows), 7)
        self.assertAlmostEqual(rows[-1]["input_events"], 20.75)
        self.assertAlmostEqual(rows[-1]["binned_events"], 20.75)
        rendered = scorefit.terminal_selected_sm_template_table(
            rows,
            templates=templates,
            luminosity=3000.0,
            selected_configuration="baseline",
            selected_scheme="background_quantile_2bin",
        )
        self.assertIn("There is no XGBoost threshold", rendered)
        self.assertIn("example background", rendered)
        self.assertIn("background_quantile_2bin", rendered)
        self.assertIn("Background N_eff by bin", rendered)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy is not installed locally")
    def test_clough_interpolation_reproduces_a_finite_supported_surface(self):
        points = np.asarray(
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ]
        )
        q = np.sum(points * points, axis=1)
        likelihood = {
            "points": points,
            "valid_mask": np.ones(len(points), dtype=bool),
            "q": {
                "background_x0.25": q,
                "background_x1": q,
                "background_x4": q,
            },
        }
        result = scorefit.interpolate_q_surfaces(
            likelihood, c3_range=(-1.0, 1.0), d4_range=(-1.0, 1.0), grid_bins=101
        )
        center = result["fields"]["background_x1"]["clough"][50, 50]
        self.assertTrue(np.isfinite(center))
        self.assertAlmostEqual(center, 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
