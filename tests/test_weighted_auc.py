import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_DIR / "Code" / "xgboost_root_varfiles_module.py"


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
    read_root.FEATURE_NAMES = [f"feature_{idx}" for idx in range(28)]
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
    sys.path.insert(0, str(REPO_DIR / "Code"))
    try:
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location("xgb_module_under_test", MODULE_PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    finally:
        try:
            sys.path.remove(str(REPO_DIR / "Code"))
        except ValueError:
            pass


class WeightedAucTests(unittest.TestCase):
    def test_weighted_auc_uses_rank_definition_with_ties(self):
        module = _load_module_with_stubs()
        labels = np.asarray([1, 0, 1, 0, 0], dtype=int)
        scores = np.asarray([0.9, 0.8, 0.8, 0.3, 0.8], dtype=float)
        weights = np.asarray([2.0, 5.0, 3.0, 7.0, 11.0], dtype=float)

        # Positive 0.9 beats all negative weight. Positive 0.8 ties two
        # negative entries and beats the 0.3 entry.
        expected = (2.0 * (5.0 + 7.0 + 11.0) + 3.0 * (7.0 + 0.5 * (5.0 + 11.0))) / (
            (2.0 + 3.0) * (5.0 + 7.0 + 11.0)
        )

        self.assertAlmostEqual(module._weighted_binary_auc(labels, scores, weights), expected)

    def test_weighted_auc_handles_extreme_positive_weights(self):
        module = _load_module_with_stubs()
        labels = np.asarray([0, 1] * 1000, dtype=int)
        scores = np.linspace(0.0, 1.0, labels.size)
        weights = np.geomspace(1e-18, 1e18, labels.size)

        auc = module._weighted_binary_auc(labels, scores, weights)

        self.assertTrue(np.isfinite(auc))
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)

    def test_metric_weights_use_absolute_values_for_signed_mc_weights(self):
        module = _load_module_with_stubs()
        signed = np.asarray([2.0, -3.5, 0.0, -1.0], dtype=float)

        metric_weights = module._nonnegative_metric_weights(signed)

        np.testing.assert_allclose(metric_weights, [2.0, 3.5, 0.0, 1.0])
        self.assertTrue(np.all(metric_weights >= 0.0))

    def test_weighted_auc_diagnostic_accepts_signed_mc_weights(self):
        module = _load_module_with_stubs()
        labels = np.asarray([1, 0, 1, 0], dtype=int)
        scores = np.asarray([0.9, 0.7, 0.2, 0.1], dtype=float)
        signed_weights = np.asarray([2.0, -5.0, -3.0, 7.0], dtype=float)

        auc = module._weighted_auc_for_diagnostics(labels, scores, signed_weights)

        self.assertAlmostEqual(auc, module._weighted_binary_auc(labels, scores, np.abs(signed_weights)))


if __name__ == "__main__":
    unittest.main()
