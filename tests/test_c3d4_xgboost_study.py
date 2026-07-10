import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_DIR / "Code"
sys.path.insert(0, str(CODE_DIR))
try:
    import c3d4_xgboost_study as study
finally:
    sys.path.remove(str(CODE_DIR))


class FoldTests(unittest.TestCase):
    def test_source_local_folds_are_balanced_and_order_independent(self):
        sources = np.asarray(["a"] * 13 + ["b"] * 7)
        entries = np.asarray(list(range(13)) + list(range(100, 107)))
        folds = study.deterministic_folds(sources, entries)

        for source in ("a", "b"):
            counts = np.bincount(folds[sources == source], minlength=5)
            self.assertLessEqual(int(np.max(counts) - np.min(counts)), 1)

        permutation = np.asarray([7, 18, 3, 12, 0, 19, 6, 13, 2, 16, 1, 15, 8, 10, 5, 17, 4, 14, 9, 11])
        permuted = study.deterministic_folds(sources[permutation], entries[permutation])
        expected = {
            (source, int(entry)): int(fold)
            for source, entry, fold in zip(sources, entries, folds)
        }
        actual = {
            (source, int(entry)): int(fold)
            for source, entry, fold in zip(
                sources[permutation], entries[permutation], permuted
            )
        }
        self.assertEqual(expected, actual)

    def test_duplicate_source_entry_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate event_indices"):
            study.deterministic_folds(["a", "a"], [3, 3])

    def test_rotations_test_each_event_exactly_once(self):
        folds = study.deterministic_folds(["a"] * 23, range(23))
        rotations = study.crossfit_rotations(folds)
        np.testing.assert_array_equal(
            np.sum([rotation["test"] for rotation in rotations], axis=0),
            np.ones(23, dtype=int),
        )
        for index, rotation in enumerate(rotations):
            self.assertEqual(rotation["test_fold"], index)
            self.assertEqual(rotation["validation_fold"], (index + 1) % 5)
            self.assertFalse(np.any(rotation["train"] & rotation["validation"]))
            self.assertFalse(np.any(rotation["train"] & rotation["test"]))

    def test_fold_union_closes_signed_yield(self):
        weights = np.asarray([1.0, -0.25, 2.0, 3.5, -0.5, 1.25, 4.0])
        folds = study.deterministic_folds(["x"] * len(weights), range(len(weights)))
        heldout = sum(float(np.sum(weights[folds == fold])) for fold in range(5))
        self.assertTrue(
            math.isclose(heldout, float(np.sum(weights)), rel_tol=1e-15, abs_tol=1e-15)
        )


class TrainingWeightTests(unittest.TestCase):
    def test_equal_point_weights_use_absolute_intrapoint_ratios(self):
        physical = [1.0, -3.0, 10.0, 2.0, 2.0]
        points = ["p0", "p0", "p1", "p2", "p2"]
        weights = study.pooled_equal_point_weights(physical, points, expected_points=3)
        for point in ("p0", "p1", "p2"):
            self.assertAlmostEqual(float(np.sum(weights[np.asarray(points) == point])), 1.0 / 3.0)
        self.assertAlmostEqual(weights[1] / weights[0], 3.0)
        self.assertTrue(np.all(weights >= 0.0))

    def test_pooled_classifier_weights_balance_classes_and_preserve_background(self):
        labels = np.asarray([1, 1, 1, 0, 0])
        physical = np.asarray([1.0, 9.0, 4.0, 2.0, -6.0])
        points = np.asarray(["a", "a", "b"])
        weights = study.pooled_classifier_training_weights(
            labels, physical, points, expected_points=2
        )
        self.assertAlmostEqual(float(np.sum(weights[labels == 1])), 1.0)
        self.assertAlmostEqual(float(np.sum(weights[labels == 0])), 1.0)
        self.assertAlmostEqual(weights[4] / weights[3], 3.0)
        self.assertAlmostEqual(float(np.sum(weights[:2])), float(weights[2]))

    def test_zero_weight_point_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no non-zero physical training weight"):
            study.pooled_equal_point_weights([0.0, 1.0], ["zero", "nonzero"])

    def test_weighted_yield_keeps_sign_and_sumw2(self):
        result = study.weighted_yield([2.0, -1.0, 3.0])
        self.assertEqual(result["yield"], 4.0)
        self.assertEqual(result["sumw2"], 14.0)
        self.assertAlmostEqual(result["effective_entries"], 16.0 / 14.0)


class ExactLimitAndThresholdTests(unittest.TestCase):
    def test_exact_cls_reference_values(self):
        self.assertAlmostEqual(
            study.exact_cls_signal_upper_limit(0.0), -math.log(0.05), places=10
        )
        self.assertAlmostEqual(
            study.exact_cls_signal_upper_limit(2.9206089), 5.43543775563513, places=10
        )

    def test_background_scan_matches_direct_signed_sums(self):
        scores = np.asarray([0.1, 0.5, 0.5, 0.9])
        weights = np.asarray([2.0, -0.25, 1.25, 3.0])
        thresholds = np.asarray([0.0, 0.5, 0.8, 1.0])
        scan = study.background_threshold_scan(scores, weights, thresholds)
        for index, threshold in enumerate(thresholds):
            selected = scores >= threshold
            self.assertAlmostEqual(scan["yield"][index], float(np.sum(weights[selected])))
            self.assertAlmostEqual(
                scan["sumw2"][index], float(np.sum(np.square(weights[selected])))
            )
            self.assertEqual(scan["raw_entries"][index], int(np.sum(selected)))

    def test_unit_cross_section_signal_weights_set_sigma_limit_directly(self):
        signal_scores = np.asarray([0.8, 0.9])
        signal_unit_yields = np.asarray([2.0, 3.0])
        background_scores = np.full(30, 0.7)
        background_weights = np.full(30, 0.1)
        result = study.optimize_point_threshold(
            signal_scores,
            signal_unit_yields,
            background_scores,
            background_weights,
            thresholds=[0.5],
            min_background_raw=25,
            min_background_neff=10,
        )
        expected = study.exact_cls_signal_upper_limit(3.0) / 5.0
        self.assertAlmostEqual(result["sigma95_fb"], expected)
        self.assertEqual(result["signal_weight_convention"], "unit_cross_section_expected_yield")

    def test_efficiency_weight_convention_matches_formula(self):
        result = study.optimize_point_threshold(
            [0.8, 0.9],
            [2.0, 3.0],
            np.full(30, 0.7),
            np.full(30, 0.1),
            luminosity=100.0,
            signal_rate_factor=0.2,
            thresholds=[0.5],
        )
        expected = study.exact_cls_signal_upper_limit(3.0) / 20.0
        self.assertAlmostEqual(result["sigma95_fb"], expected)
        self.assertEqual(result["signal_weight_convention"], "efficiency_weights")

    def test_threshold_ties_prefer_larger_neff_then_lower_threshold(self):
        custom_scan = {
            "thresholds": np.asarray([0.0, 0.5]),
            "yield": np.asarray([10.0, 5.0]),
            "sumw2": np.asarray([10.0, 1.0]),
            "raw_entries": np.asarray([30, 25]),
            "effective_entries": np.asarray([10.0, 25.0]),
            "s95_events": np.asarray([5.0, 5.0]),
            "confidence_level": np.asarray([0.95]),
        }
        result = study.optimize_point_threshold(
            [0.9, 0.9],
            [1.0, 1.0],
            np.full(30, 0.9),
            np.ones(30),
            thresholds=[0.0, 0.5],
            background_scan=custom_scan,
        )
        self.assertEqual(result["threshold"], 0.5)

        custom_scan["effective_entries"] = np.asarray([25.0, 25.0])
        result = study.optimize_point_threshold(
            [0.9, 0.9],
            [1.0, 1.0],
            np.full(30, 0.9),
            np.ones(30),
            thresholds=[0.0, 0.5],
            background_scan=custom_scan,
        )
        self.assertEqual(result["threshold"], 0.0)

    def test_threshold_rejects_insufficient_background_mc(self):
        with self.assertRaises(study.NoValidThresholdError):
            study.optimize_point_threshold(
                [0.9], [1.0], [0.9] * 9, [1.0] * 9, thresholds=[0.5]
            )

    def test_limit_objective(self):
        values = np.asarray([1.0, 2.0, 4.0, 8.0])
        expected = 0.75 * np.median(np.log(values)) + 0.25 * np.quantile(
            np.log(values), 0.9
        )
        self.assertAlmostEqual(study.limit_objective(values), expected)


class BinningTests(unittest.TestCase):
    def test_score_bins_assign_internal_edge_to_upper_bin(self):
        np.testing.assert_array_equal(
            study.score_bin_indices([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]),
            [0, 1, 1],
        )

    def test_enumerates_all_subsets_for_distinct_quantile_edges(self):
        scores = np.linspace(0.0, 1.0, 1001)
        candidates = study.enumerate_score_binnings(scores, np.ones_like(scores))
        self.assertEqual(len(candidates), 15)  # C(4,1)+...+C(4,4)
        self.assertEqual({item["n_bins"] for item in candidates}, {2, 3, 4, 5})

    def test_validation_selection_prefers_fewer_bins_within_one_percent(self):
        scores = np.linspace(0.0, 1.0, 1001)

        def evaluator(edges):
            n_bins = len(edges) - 1
            return {2: 1.005, 3: 1.0, 4: 1.1, 5: 1.2}[n_bins]

        result = study.validation_binning(
            scores,
            np.ones_like(scores),
            limit_evaluator=evaluator,
            min_background_raw=0,
            min_background_neff=0,
        )
        self.assertEqual(result["selected"]["n_bins"], 2)
        self.assertAlmostEqual(result["minimum_expected_limit_fb"], 1.0)

    def test_test_binning_coarsens_instead_of_clipping_negative_yield(self):
        validation = {
            "fallback_hierarchy": [
                {"edges": [0.0, 0.5, 1.0]},
                {"edges": [0.0, 1.0]},
            ]
        }
        result = study.select_test_binning(
            validation,
            background_scores=[0.2, 0.8],
            background_weights=[-1.0, 2.0],
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["n_bins"], 1)
        self.assertEqual(result["background_yield"], [1.0])

    def test_test_binning_reports_failure_if_even_total_is_nonpositive(self):
        validation = {"fallback_hierarchy": [{"edges": [0.0, 1.0]}]}
        result = study.select_test_binning(
            validation, [0.2, 0.8], [-2.0, 1.0]
        )
        self.assertEqual(result["status"], "failed")


class PyhfTests(unittest.TestCase):
    def test_workspace_has_shared_poi_and_independent_mcstat_modifiers(self):
        channels = [
            {
                "name": "fold0",
                "signal": [1.0, 2.0],
                "background": [3.0, 4.0],
                "signal_staterror": [0.1, 0.2],
                "background_staterror": [0.3, 0.4],
            },
            {
                "name": "fold1",
                "signal": [0.5],
                "background": [2.0],
                "signal_staterror": [0.05],
                "background_staterror": [0.2],
            },
        ]
        spec = study.pyhf_workspace_spec(channels, include_staterror=True)
        signal_modifiers = [
            channel["samples"][0]["modifiers"] for channel in spec["channels"]
        ]
        self.assertTrue(all(items[0]["name"] == "sigma_fb" for items in signal_modifiers))
        self.assertNotEqual(signal_modifiers[0][1]["name"], signal_modifiers[1][1]["name"])
        self.assertEqual(spec["observations"][0]["data"], [3.0, 4.0])

    def test_workspace_refuses_negative_templates(self):
        with self.assertRaisesRegex(ValueError, "negative signal"):
            study.pyhf_workspace_spec(
                [{"name": "bad", "signal": [-1.0], "background": [2.0]}]
            )
        with self.assertRaisesRegex(ValueError, "nonpositive background"):
            study.pyhf_workspace_spec(
                [{"name": "bad", "signal": [1.0], "background": [0.0]}]
            )

    @unittest.skipUnless(importlib.util.find_spec("pyhf"), "pyhf is optional locally")
    def test_pyhf_one_bin_control_matches_declared_reference(self):
        result = study.pyhf_one_bin_limit([1.0], [2.9206089], include_staterror=False)
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertAlmostEqual(result["expected_median"], 4.73532, delta=0.03)

    @unittest.skipUnless(importlib.util.find_spec("pyhf"), "pyhf is optional locally")
    def test_informative_shape_is_not_worse_than_one_bin_without_nuisances(self):
        one_bin = study.pyhf_combined_limit(
            [{"name": "one", "signal": [2.0], "background": [20.0]}],
            include_staterror=False,
        )
        shape = study.pyhf_combined_limit(
            [{"name": "shape", "signal": [0.1, 1.9], "background": [19.0, 1.0]}],
            include_staterror=False,
        )
        self.assertEqual(one_bin["status"], "ok")
        self.assertEqual(shape["status"], "ok")
        self.assertLess(shape["expected_median"], one_bin["expected_median"])


class ParameterizedTests(unittest.TestCase):
    def test_parameterized_gate_passes_only_when_all_requirements_hold(self):
        sm = np.ones(5)
        pooled = np.asarray([0.8, 0.85, 0.9, 0.8, 0.9])
        rotation_sm = np.ones((5, 5))
        rotation_pooled = np.asarray(
            [pooled, pooled * 0.98, pooled * 1.01, pooled * 0.99, pooled * 1.02]
        )
        result = study.parameterized_gate(
            pooled, sm, 2, rotation_pooled, rotation_sm
        )
        self.assertTrue(result["passed"])

        failed = study.parameterized_gate(
            np.asarray([0.8, 0.85, 1.06, 0.8, 0.9]),
            sm,
            2,
            rotation_pooled,
            rotation_sm,
        )
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["criteria"]["sm_point"])

    def test_background_replicas_are_distinct_deterministic_and_weight_preserving(self):
        features = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        weights = np.asarray([0.6, 1.2])
        folds = np.asarray([1, 4])
        grid = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        kwargs = dict(
            features=features,
            training_weights=weights,
            folds=folds,
            source_ids=["a", "b"],
            event_indices=[10, 20],
            grid_points=grid,
        )
        first = study.make_background_parameter_replicas(**kwargs)
        second = study.make_background_parameter_replicas(**kwargs)
        np.testing.assert_array_equal(first["features"], second["features"])
        np.testing.assert_array_equal(first["folds"], np.repeat(folds, 3))
        self.assertAlmostEqual(float(np.sum(first["training_weights"][:3])), weights[0])
        self.assertAlmostEqual(float(np.sum(first["training_weights"][3:])), weights[1])
        for row in first["grid_points"].reshape(2, 3, 2):
            self.assertEqual(len(np.unique(row, axis=0)), 3)


class SerializationAndOptunaTests(unittest.TestCase):
    def test_json_csv_and_manifest_serialization(self):
        manifest = study.build_method_manifest(
            observable_set="extended-91-v2",
            feature_profile="core52",
            feature_names=["a", "b"],
            training_strategy="pooled-crossfit-v2",
            source_commit="abc123",
            normalization_inputs={"luminosity": np.float64(3000.0)},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = study.write_study_results(
                temporary_directory,
                manifest,
                [{"point": "SM", "limit": np.float64(2.0), "bad": math.inf}],
                [{"fold": 0, "edges": [0.0, 1.0]}],
            )
            loaded = json.loads(Path(paths["manifest"]).read_text())
            self.assertEqual(loaded["feature_count"], 2)
            points = json.loads(Path(paths["points_json"]).read_text())
            self.assertIsNone(points[0]["bad"])
            self.assertTrue(Path(paths["points_csv"]).exists())

    def test_xgboost_search_space_uses_declared_ranges(self):
        class FakeTrial:
            def suggest_int(self, name, low, high, step=1):
                self.last_int = (name, low, high, step)
                return low

            def suggest_float(self, name, low, high, log=False):
                return low

            def suggest_categorical(self, name, choices):
                return choices[0]

        params = study.xgboost_search_params(FakeTrial())
        self.assertEqual(params["n_estimators"], 200)
        self.assertEqual(params["max_depth"], 2)
        self.assertEqual(params["gamma"], 0.0)
        self.assertEqual(params["reg_lambda"], 0.1)

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Optuna is optional locally")
    def test_optuna_study_resumes_to_fixed_total_trial_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "study.sqlite3"

            def objective(trial):
                value = trial.suggest_float("x", 0.0, 1.0)
                return (value - 0.3) ** 2

            first = study.run_optuna_tuning(
                objective,
                database,
                "smoke",
                n_trials=3,
                enqueue_params={"x": 0.3},
            )
            self.assertEqual(len(first.trials), 3)
            resumed = study.run_optuna_tuning(
                objective,
                database,
                "smoke",
                n_trials=3,
                enqueue_params={"x": 0.3},
            )
            self.assertEqual(len(resumed.trials), 3)
            summary = study.summarize_optuna_study(resumed)
            self.assertEqual(summary["n_trials"], 3)
            self.assertIsNotNone(summary["best_trial"])


if __name__ == "__main__":
    unittest.main()
