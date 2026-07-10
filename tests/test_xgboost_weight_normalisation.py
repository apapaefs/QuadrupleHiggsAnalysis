import importlib.util
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_DIR / "Code"
MODULE_PATH = CODE_DIR / "xgboost_root_varfiles_module.py"


def _placeholder(*args, **kwargs):
    raise AssertionError("placeholder should not be called by this test")


def _load_module_with_stubs():
    sklearn = types.ModuleType("sklearn")
    model_selection = types.ModuleType("sklearn.model_selection")
    metrics = types.ModuleType("sklearn.metrics")
    xgboost = types.ModuleType("xgboost")
    read_root = types.ModuleType("read_root_varfiles")
    tqdm_module = types.ModuleType("tqdm")
    tqdm_auto = types.ModuleType("tqdm.auto")

    model_selection.train_test_split = _placeholder
    metrics.accuracy_score = _placeholder
    metrics.confusion_matrix = _placeholder
    metrics.RocCurveDisplay = object
    metrics.roc_auc_score = _placeholder
    metrics.roc_curve = _placeholder
    xgboost.XGBClassifier = object
    read_root.FEATURE_NAMES = [f"feature_{index}" for index in range(28)]
    read_root.read_ROOT_varfile = _placeholder
    tqdm_auto.tqdm = lambda iterable=None, *args, **kwargs: iterable

    modules = {
        "sklearn": sklearn,
        "sklearn.model_selection": model_selection,
        "sklearn.metrics": metrics,
        "xgboost": xgboost,
        "read_root_varfiles": read_root,
        "tqdm": tqdm_module,
        "tqdm.auto": tqdm_auto,
    }
    sys.path.insert(0, str(CODE_DIR))
    try:
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location("xgb_weight_module_under_test", MODULE_PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    finally:
        sys.path.remove(str(CODE_DIR))


class XGBoostWeightNormalisationTests(unittest.TestCase):
    def test_training_weights_preserve_physical_ratios_and_balance_classes(self):
        module = _load_module_with_stubs()
        labels = np.asarray([0, 0, 0, 1], dtype=int)
        physical = np.asarray([1.0, -2.0, 3.0, 20.0], dtype=float)

        weights = module._balanced_training_weights(labels, physical)

        self.assertAlmostEqual(float(np.sum(weights[labels == 0])), 2.0)
        self.assertAlmostEqual(float(np.sum(weights[labels == 1])), 2.0)
        np.testing.assert_allclose(weights[:3] / weights[0], [1.0, 2.0, 3.0])
        self.assertTrue(np.all(weights >= 0.0))

    def test_training_weights_reject_a_zero_weight_class(self):
        module = _load_module_with_stubs()
        with self.assertRaisesRegex(ValueError, "no non-zero physical training weight"):
            module._balanced_training_weights([0, 0, 1], [1.0, 2.0, 0.0])

    def test_split_weights_are_rescaled_to_each_full_source_yield(self):
        module = _load_module_with_stubs()
        scaled, metadata = module._rescale_source_physical_weights(
            full_physical_weights=[2.0, 3.0, 10.0, 20.0],
            full_sources=["signal", "signal", "background", "background"],
            split_physical_weights=[2.0, 20.0],
            split_sources=["signal", "background"],
        )

        np.testing.assert_allclose(scaled, [5.0, 30.0])
        self.assertAlmostEqual(metadata["signal"]["scale_factor"], 2.5)
        self.assertAlmostEqual(metadata["background"]["scale_factor"], 1.5)
        self.assertAlmostEqual(metadata["signal"]["split_preselected_events_after_rescaling"], 5.0)

    def test_split_weight_rescaling_requires_every_source(self):
        module = _load_module_with_stubs()
        with self.assertRaisesRegex(ValueError, "contains no events from source background"):
            module._rescale_source_physical_weights(
                full_physical_weights=[2.0, 10.0],
                full_sources=["signal", "background"],
                split_physical_weights=[2.0],
                split_sources=["signal"],
            )

    def test_normalisation_falls_back_to_generated_events_only_for_unit_weights(self):
        module = _load_module_with_stubs()
        denominator, source = module._normalisation_denominator(100, [1.0, 1.0], None)
        self.assertEqual(denominator, 100.0)
        self.assertEqual(source, "generated_events")

        with self.assertRaisesRegex(ValueError, "non-unit-weight sample"):
            module._normalisation_denominator(100, [1.0, 0.5], None)
        with self.assertRaisesRegex(ValueError, "Missing total input-weight metadata"):
            module._normalisation_denominator(None, [1.0, 1.0], None)

    def test_explicit_total_input_weight_normalises_weighted_samples(self):
        module = _load_module_with_stubs()
        denominator, source = module._normalisation_denominator(100, [2.0, 3.0], 250.0)
        self.assertEqual(denominator, 250.0)
        self.assertEqual(source, "input_weight_sum")

    def test_signal_and_background_sigma_eff_include_full_selection_efficiency(self):
        module = _load_module_with_stubs()

        class ScoreModel:
            def predict_proba(self, features):
                scores = np.asarray([0.9, 0.4, 0.8], dtype=float)
                return np.column_stack([1.0 - scores, scores])

        module.load_model = lambda path: ScoreModel()
        module.read_ROOT_varfile = lambda path, label, xsec, max_events=None: (
            np.zeros((3, 28), dtype=float).tolist(),
            [label, label, label],
            [2.0, 3.0, 5.0],
        )

        common = {
            "model_file": "unused.json",
            "threshold": 0.5,
            "luminosity": 100.0,
            "max_events": None,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            signal_rows = module.score_signal_files(
                signal_files=[Path(temporary_directory) / "signal.root"],
                output_dir=Path(temporary_directory) / "signal_scores",
                signal_xsecs_fb=[10.0],
                signal_rate_factors=[2.0],
                signal_generated_events=[100],
                signal_normalisation_weights=[20.0],
                **common,
            )
            background_result = module.score_background_files(
                background_files=[Path(temporary_directory) / "background.root"],
                output_dir=Path(temporary_directory) / "background_scores",
                background_xsecs_fb=[10.0],
                background_rate_factors=[2.0],
                background_generated_events=[100],
                background_normalisation_weights=[20.0],
                **common,
            )

        for row in [signal_rows[0], background_result["backgrounds"][0]]:
            self.assertAlmostEqual(row["expected_preselected_events"], 1000.0)
            self.assertAlmostEqual(row["expected_selected_events"], 700.0)
            self.assertAlmostEqual(row["analysis_efficiency"], 0.5)
            self.assertAlmostEqual(row["xgboost_efficiency"], 0.7)
            self.assertAlmostEqual(row["final_efficiency"], 0.35)
            self.assertAlmostEqual(row["raw_sigma_eff_fb"], 3.5)
            self.assertAlmostEqual(row["effective_sigma_eff_fb"], 7.0)
            self.assertAlmostEqual(
                row["effective_sigma_eff_fb"],
                row["expected_selected_events"] / 100.0,
            )

    def test_analysis_rescales_train_and_test_sources_before_weighted_consumers(self):
        module = _load_module_with_stubs()
        captures = {}

        signal_rows = np.zeros((2, 28), dtype=float)
        signal_rows[:, 0] = [1.0, 2.0]
        background_rows = np.zeros((2, 28), dtype=float)
        background_rows[:, 0] = [3.0, 4.0]

        def load_group(files, label, *args, **kwargs):
            if label == 1:
                return (
                    signal_rows.tolist(),
                    [1, 1],
                    [1.0, 1.0],
                    [10.0, 20.0],
                    ["signal", "signal"],
                    [{"entries": 2, "generated_events": 2}],
                )
            return (
                background_rows.tolist(),
                [0, 0],
                [1.0, 1.0],
                [100.0, 300.0],
                ["background", "background"],
                [{"entries": 2, "generated_events": 2}],
            )

        def deterministic_split(*arrays, **kwargs):
            train_indices = np.asarray([0, 2])
            test_indices = np.asarray([1, 3])
            result = []
            for array in arrays:
                array = np.asarray(array)
                result.extend([array[train_indices], array[test_indices]])
            return tuple(result)

        class FakeClassifier:
            def __init__(self, **params):
                self.feature_importances_ = np.zeros(28, dtype=float)

            def fit(self, features, labels, sample_weight=None):
                captures["training_weights"] = np.asarray(sample_weight, dtype=float)
                return self

            def predict_proba(self, features):
                scores = np.asarray([0.8, 0.2], dtype=float)
                return np.column_stack([1.0 - scores, scores])

        def capture_auc(labels, scores, weights):
            captures["auc_weights"] = np.asarray(weights, dtype=float)
            return 0.75

        def capture_threshold(scores, labels, weights, *args, **kwargs):
            captures["threshold_weights"] = np.asarray(weights, dtype=float)
            return {
                "threshold": 0.1,
                "signal_events": 30.0,
                "background_events": 400.0,
                "significance": 1.5,
                "signal_efficiency": 1.0,
                "background_efficiency": 1.0,
                "selected_background_entries": 1,
                "selected_background_effective_entries": 1.0,
                "selected_background_source_entries": {"background": 1},
                "mc_stat_requirement_satisfied": True,
            }

        module._load_signal_background_group = load_group
        module.train_test_split = deterministic_split
        module.xgb.XGBClassifier = FakeClassifier
        module.accuracy_score = lambda labels, predictions: 1.0
        module.roc_auc_score = lambda labels, scores: 0.75
        module.confusion_matrix = lambda *args, **kwargs: np.asarray([[1, 0], [0, 1]])
        module._weighted_auc_for_diagnostics = capture_auc
        module._best_significance_threshold = capture_threshold
        module.save_model = lambda *args, **kwargs: None
        module._write_feature_importance_plot = lambda *args, **kwargs: None
        module._write_scores_csv = lambda path, labels, scores, weights, sources: captures.update(
            scores_weights=np.asarray(weights, dtype=float)
        )
        module._write_roc_plot = lambda path, labels, scores, weights: captures.update(
            roc_weights=np.asarray(weights, dtype=float)
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = module.run_signal_background_analysis(
                signal_files=["signal.root"],
                background_files=["background.root"],
                output_dir=temporary_directory,
                signal_xsecs_fb=[1.0],
                background_xsecs_fb=[1.0],
                signal_generated_events=[2],
                background_generated_events=[2],
            )

        np.testing.assert_allclose(captures["training_weights"], [1.0, 1.0])
        for key in ("auc_weights", "threshold_weights", "scores_weights", "roc_weights"):
            np.testing.assert_allclose(captures[key], [30.0, 400.0])
        metrics = result["metrics"]
        self.assertTrue(math.isclose(metrics["training_source_scale_factors"]["signal"]["scale_factor"], 3.0))
        self.assertTrue(
            math.isclose(metrics["training_source_scale_factors"]["background"]["scale_factor"], 4.0)
        )
        self.assertTrue(math.isclose(metrics["heldout_source_scale_factors"]["signal"]["scale_factor"], 1.5))
        self.assertTrue(
            math.isclose(metrics["heldout_source_scale_factors"]["background"]["scale_factor"], 4.0 / 3.0)
        )


if __name__ == "__main__":
    unittest.main()
