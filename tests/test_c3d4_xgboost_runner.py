from __future__ import annotations

import sys
import csv
import subprocess
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


CODE = Path(__file__).resolve().parents[1] / "Code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import c3d4_xgboost_runner as runner  # noqa: E402
from observable_schemas import (  # noqa: E402
    EXTENDED_FEATURE_NAMES,
    PARAMETERIZED_ML_FEATURES,
    ModelContractError,
    validate_model_contract,
)


def sample(sample_id, kind, raw, physical, c3=None, d4=None):
    raw = np.asarray(raw, dtype=float)
    physical = np.asarray(physical, dtype=float)
    entries = len(raw)
    return runner.EventSample(
        path=Path(f"/{sample_id}.root"),
        sample_id=sample_id,
        kind=kind,
        features=np.arange(entries * 91, dtype=float).reshape(entries, 91),
        raw_weights=raw,
        physical_weights=physical,
        unit_xsec_weights=physical,
        event_indices=np.arange(entries),
        source_entry_indices=np.arange(entries),
        folds=np.arange(entries) % 5,
        xsec_fb=1.0,
        rate_factor=1.0,
        normalisation_weight=float(np.sum(raw)),
        normalisation_source="test",
        generated_events=entries,
        c3=c3,
        d4=d4,
        metadata={},
    )


class C3D4XGBoostRunnerTests(unittest.TestCase):
    def test_analysis_document_lists_the_complete_ordered_schema(self):
        document = (CODE.parent / "docs" / "c3d4_xgboost_analysis.md").read_text()
        for index, name in enumerate(EXTENDED_FEATURE_NAMES):
            self.assertIn(f"| {index} | `{name}` |", document)

    def test_legacy_schema_rejects_extended_profiles(self):
        with self.assertRaisesRegex(ValueError, "supports only corrected28"):
            runner.run_c3d4_study(
                sm_signal_specs=[],
                grid_signal_specs=[],
                background_specs=[],
                output_dir="/tmp/unused-qha-test",
                observable_set="legacy-28-v1",
                feature_profile="full91",
            )

    def test_pooled_training_arrays_equalize_points_and_classes(self):
        grid = [
            sample("p0", "grid_signal", [1, 2, 3, 4, 5], [10, 20, 30, 40, 50], 0, 0),
            sample("p1", "grid_signal", [5, 4, 3, 2, 1], [500, 400, 300, 200, 100], 1, 0),
        ]
        background = [
            sample("b0", "background", [1] * 5, [1, 2, 3, 4, 5]),
            sample("b1", "background", [1] * 5, [10, 20, 30, 40, 50]),
        ]

        _, labels, weights = runner._training_arrays(
            [],
            grid,
            background,
            strategy="pooled-crossfit-v2",
            profile_indices=np.arange(28),
            rotation=0,
            n_folds=5,
        )

        signal_weights = weights[labels == 1]
        background_weights = weights[labels == 0]
        # Six nonzero signal and six original background rows give a common
        # class total of (6 + 6) / 2 = 6 and mean effective-row weight one.
        self.assertAlmostEqual(float(np.sum(signal_weights)), 6.0)
        self.assertAlmostEqual(float(np.sum(background_weights)), 6.0)
        # Three training folds give three rows from each point, in sample order.
        self.assertAlmostEqual(float(np.sum(signal_weights[:3])), 3.0)
        self.assertAlmostEqual(float(np.sum(signal_weights[3:])), 3.0)
        # Physical process ratios are preserved within the background class.
        self.assertAlmostEqual(float(np.sum(background_weights[3:])), 60.0 / 11.0)

    def test_parameterized_training_replicates_background_and_preserves_class_totals(self):
        grid = [
            sample(f"p{index}", "grid_signal", [1] * 5, [1] * 5, index, 10 * index)
            for index in range(3)
        ]
        background = [sample("b0", "background", [1] * 5, [1, 2, 3, 4, 5])]

        features, labels, weights = runner._training_arrays(
            [],
            grid,
            background,
            strategy="parameterized-crossfit-v1",
            profile_indices=np.arange(28),
            rotation=0,
            n_folds=5,
        )

        self.assertEqual(features.shape[1], 30)
        self.assertEqual(int(np.sum(labels == 1)), 9)
        self.assertEqual(int(np.sum(labels == 0)), 9)
        # Nine signal rows plus three original background rows give class
        # totals of six.  Replication does not inflate the normalization.
        self.assertAlmostEqual(float(np.sum(weights[labels == 1])), 6.0)
        self.assertAlmostEqual(float(np.sum(weights[labels == 0])), 6.0)
        self.assertAlmostEqual(float(np.sum(weights)), 12.0)
        background_parameters = features[labels == 0, -2:]
        for start in range(0, len(background_parameters), 3):
            self.assertEqual(len(np.unique(background_parameters[start:start + 3], axis=0)), 3)

    def test_parameterized_model_metadata_requires_the_appended_coordinates(self):
        class FakeBooster:
            def __init__(self):
                self.feature_names = None
                self.attributes = {}
                self.count = 0

            def num_features(self):
                return self.count

            def set_attr(self, **attributes):
                self.attributes.update(attributes)

            def attr(self, name):
                return self.attributes.get(name)

        class FakeClassifier:
            def __init__(self, **parameters):
                self.parameters = parameters
                self.booster = FakeBooster()

            def fit(self, features, labels, sample_weight=None, verbose=False):
                self.n_features_in_ = features.shape[1]
                self.booster.count = features.shape[1]
                return self

            def get_booster(self):
                return self.booster

        features = np.arange(8 * 30, dtype=float).reshape(8, 30)
        labels = np.asarray([0, 1] * 4)
        weights = np.ones(8)
        with mock.patch.dict(
            sys.modules,
            {"xgboost": types.SimpleNamespace(XGBClassifier=FakeClassifier)},
        ):
            model, metadata, _ = runner._train_model(
                features,
                labels,
                weights,
                params={"n_estimators": 1, "max_depth": 1},
                seed=12345,
                observable_set="extended-91-v2",
                profile="corrected28",
                strategy="parameterized-crossfit-v1",
                rotation=0,
                source_commit="test",
            )
        self.assertEqual(metadata["feature_count"], 30)
        self.assertEqual(tuple(metadata["feature_names"][-2:]), tuple(
            name for name, _ in PARAMETERIZED_ML_FEATURES
        ))
        validate_model_contract(
            model,
            "extended-91-v2",
            "corrected28",
            ml_parameter_features=PARAMETERIZED_ML_FEATURES,
        )
        with self.assertRaises(ModelContractError):
            validate_model_contract(model, "extended-91-v2", "corrected28")
        self.assertEqual(
            metadata["classifier_weight_scale_version"],
            runner.CLASSIFIER_WEIGHT_SCALE_VERSION,
        )
        self.assertEqual(metadata["classifier_signal_weight_total"], 4.0)
        self.assertEqual(metadata["classifier_background_weight_total"], 4.0)
        self.assertEqual(metadata["classifier_effective_row_count"], 8)

    def test_prefit_guard_rejects_the_old_unit_class_totals(self):
        labels = np.asarray([1, 1, 0, 0])
        old_unit_class_weights = np.asarray([0.5, 0.5, 0.5, 0.5])
        with self.assertRaisesRegex(
            runner.ZeroSplitModelError, "every binary split impossible"
        ):
            runner._classifier_weight_diagnostics(
                labels,
                old_unit_class_weights,
                min_child_weight=1.0,
            )

    def test_real_xgboost_training_produces_nonconstant_split_model(self):
        script = f"""
import sys
sys.path.insert(0, {str(CODE)!r})
try:
    import xgboost
except ImportError:
    raise SystemExit(77)
import numpy as np
import c3d4_xgboost_runner as runner
rng = np.random.default_rng(12345)
background = rng.normal(0.0, 0.2, size=(100, 28))
signal = rng.normal(0.0, 0.2, size=(100, 28))
signal[:, 0] += 3.0
X = np.concatenate([signal, background])
y = np.concatenate([np.ones(100, dtype=np.int8), np.zeros(100, dtype=np.int8)])
signal_w, background_w = runner._balanced_weights(np.ones(100), np.ones(100))
weights = np.concatenate([signal_w, background_w])
model, metadata, _ = runner._train_model(
    X, y, weights,
    params={{"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1}},
    seed=12345,
    observable_set="extended-91-v2",
    profile="corrected28",
    strategy="sm-crossfit-v2",
    rotation=0,
    source_commit="test",
)
assert metadata["xgboost_split_nodes"] > 0
assert metadata["training_score_std"] > 0.0
assert np.ptp(model.predict_proba(X)[:, 1]) > 0.0
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
        )
        if completed.returncode == 77:
            raise unittest.SkipTest("xgboost is not installed in this Python environment")
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_crossfit_test_aggregation_uses_each_fold_without_rescaling(self):
        point = sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0)
        rotations = []
        for fold in range(5):
            rotations.append(
                {
                    "points": {
                        point.point_id: {
                            "threshold": 0.1 * fold,
                            "signal_unit_yield": 0.2,
                            "signal_sumw2_unit": 0.04,
                            "signal_raw_entries": 1,
                            "signal_feature_unit_yield": 1.0,
                            "background_yield": 0.4,
                            "background_sumw2": 0.16,
                            "background_raw_entries": 1,
                            "background_effective_entries": 1.0,
                            "cut_sigma95_fb": runner.exact_cls_signal_upper_limit(0.4) / 0.2,
                        }
                    }
                }
            )

        result = runner._aggregate_cut_results([point], rotations)[0]

        self.assertAlmostEqual(result["selected_signal_yield_per_fb"], 1.0)
        self.assertAlmostEqual(result["background_yield"], 2.0)
        self.assertEqual(result["background_raw_entries"], 5)
        self.assertAlmostEqual(result["threshold_mean"], 0.2)
        self.assertAlmostEqual(
            result["cut_sigma95_fb"],
            runner.exact_cls_signal_upper_limit(2.0),
        )

    def test_parameterized_validation_aggregation_uses_pointwise_background_scores(self):
        point = sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0)
        rotations = [
            {
                "validation": {
                    "parameterized": True,
                    "points": {point.point_id: {"threshold": 0.5, "sigma95_fb": 1.0}},
                }
            }
            for _ in range(5)
        ]
        arrays = {
            "signal_scores": np.asarray([0.6]),
            "signal_weights": np.asarray([0.2]),
            "background_scores": np.asarray([0.6]),
            "background_weights": np.asarray([0.4]),
        }
        with mock.patch.object(runner, "_validation_fold_arrays", return_value=arrays):
            result, _ = runner._aggregate_validation_crossfit([point], rotations)
        self.assertAlmostEqual(result[0]["validation_signal_yield_per_fb"], 1.0)
        self.assertAlmostEqual(result[0]["validation_background_yield"], 2.0)
        self.assertAlmostEqual(
            result[0]["validation_cut_sigma95_fb"],
            runner.exact_cls_signal_upper_limit(2.0),
        )

    def test_validation_scaling_is_fixed_before_test_fold_is_read(self):
        self.assertEqual(runner._partition_scale(5), 5.0)
        with self.assertRaises(ValueError):
            runner._partition_scale(1)

    def test_optuna_fingerprint_binds_schema_and_inputs(self):
        common = {
            "profile": "corrected28",
            "strategy": "pooled-crossfit-v2",
            "rotation": 0,
            "n_folds": 5,
            "seed": 12345,
            "source_commit": "abc",
            "fold_digest": "folds",
            "normalization_inputs": {"luminosity_fb_inverse": 3000.0},
            "input_hashes": {"sample.root": "hash-a"},
            "package_versions": {"xgboost": "3.0.2", "optuna": "4.9.0"},
        }
        extended = runner._run_fingerprint(observable_set="extended-91-v2", **common)
        legacy = runner._run_fingerprint(observable_set="legacy-28-v1", **common)
        changed_input = runner._run_fingerprint(
            observable_set="extended-91-v2",
            **{**common, "input_hashes": {"sample.root": "hash-b"}},
        )
        self.assertNotEqual(extended, legacy)
        self.assertNotEqual(extended, changed_input)
        changed_runtime = runner._run_fingerprint(
            observable_set="extended-91-v2",
            **{**common, "package_versions": {"xgboost": "3.1.0", "optuna": "4.9.0"}},
        )
        self.assertNotEqual(extended, changed_runtime)
        with mock.patch.object(
            runner, "CLASSIFIER_WEIGHT_SCALE_VERSION", "different-weight-scale"
        ):
            changed_weight_scale = runner._run_fingerprint(
                observable_set="extended-91-v2", **common
            )
        self.assertNotEqual(extended, changed_weight_scale)

    def test_optuna_attempt_budget_counts_pruned_and_running_trials(self):
        self.assertEqual(runner._remaining_optuna_attempts(["WAITING"], 40), 40)
        self.assertEqual(runner._remaining_optuna_attempts(["COMPLETE"] * 10, 40), 30)
        self.assertEqual(
            runner._remaining_optuna_attempts(
                ["COMPLETE"] * 9 + ["RUNNING"], 40
            ),
            30,
        )
        self.assertEqual(
            runner._remaining_optuna_attempts(
                ["COMPLETE"] * 35 + ["PRUNED"] * 5, 40
            ),
            0,
        )

    def test_csv_writer_keeps_nested_bin_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.csv"
            runner._write_rows(
                output,
                [{"point_id": "p", "fold_bin_edges": [[0.0, 0.5, 1.0]]}],
            )
            with output.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["fold_bin_edges"], "[[0.0,0.5,1.0]]")

    def test_one_bin_control_is_reported_when_shape_selection_fails(self):
        point = sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0)
        records = []
        for rotation in range(5):
            signal_row = {
                "scores": np.asarray([0.5]),
                "unit_xsec_weights": np.asarray([0.2]),
            }
            background_row = {
                "scores": np.asarray([0.5]),
                "physical_weights": np.asarray([0.4]),
            }
            records.append(
                {
                    "rotation": rotation,
                    "validation": {"signal_rows": {}, "background_rows": {}},
                    "test": {
                        "signal_rows": {point.sample_id: signal_row},
                        "background_rows": {"b": background_row},
                    },
                }
            )
        with mock.patch.object(
            runner, "_candidate_maps_for_validation", return_value=([], [])
        ), mock.patch.object(
            runner,
            "_select_shape_candidate",
            return_value={"status": "failed", "error": "no valid shape"},
        ), mock.patch.object(
            runner,
            "pyhf_one_bin_limit",
            return_value={"status": "ok", "expected_median": 7.0},
        ):
            result = runner._shape_results([point], records)[0]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["pyhf_one_bin_sigma95_fb"], 7.0)

    def test_negative_signal_bin_uses_validation_defined_coarser_fallback(self):
        point = sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0)
        records = []
        for rotation in range(5):
            records.append(
                {
                    "rotation": rotation,
                    "validation": {"signal_rows": {}, "background_rows": {}},
                    "test": {
                        "signal_rows": {
                            point.sample_id: {
                                "scores": np.asarray([0.25, 0.75]),
                                "unit_xsec_weights": np.asarray([-1.0, 2.0]),
                            }
                        },
                        "background_rows": {
                            "b": {
                                "scores": np.asarray([0.25, 0.75]),
                                "physical_weights": np.asarray([1.0, 1.0]),
                            }
                        },
                    },
                }
            )
        fine = {
            "n_bins": 2,
            "fold_edges": [[0.0, 0.5, 1.0] for _ in range(5)],
        }
        coarse = {
            "n_bins": 1,
            "fold_edges": [[0.0, 1.0] for _ in range(5)],
        }
        selection = {
            "status": "ok",
            "selected": fine,
            "fallback_hierarchy": [fine, coarse],
        }
        successful_fit = {"status": "ok", "expected_median": 4.0}
        with mock.patch.object(
            runner, "_candidate_maps_for_validation", return_value=([], [])
        ), mock.patch.object(
            runner, "_select_shape_candidate", return_value=selection
        ), mock.patch.object(
            runner, "pyhf_one_bin_limit", return_value=successful_fit
        ), mock.patch.object(
            runner, "pyhf_combined_limit", return_value=successful_fit
        ):
            result = runner._shape_results([point], records)[0]
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["bin_count"], 1)
        self.assertFalse(result["test_binning_attempts"][0]["positive_test_signal"])


if __name__ == "__main__":
    unittest.main()
