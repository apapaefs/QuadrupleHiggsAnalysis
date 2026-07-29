from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from contextlib import ExitStack, contextmanager, redirect_stdout
from unittest import mock
from pathlib import Path

import numpy as np


CODE = Path(__file__).resolve().parents[1] / "Code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import c3d4_xgboost_runner as runner  # noqa: E402
import c3d4_plot_style as plot_style  # noqa: E402
from hh4b_c3_xsec import fit_hh4b_c3_cross_section  # noqa: E402
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


def legacy_contour_rows(*, include_shape=True, ratio_offset=0.0):
    rows = []
    for c3 in (-20.0, 0.0, 20.0):
        for d4 in (-300.0, 0.0, 300.0):
            xsec = 10.0 ** (1.0 + 0.01 * c3 + 0.0005 * d4)
            ratio = 10.0 ** (ratio_offset + 0.03 * c3 + 0.001 * d4)
            cut_limit = xsec / ratio
            row = {
                "point_id": f"c3={c3:g},d4={d4:g}",
                "c3": c3,
                "d4": d4,
                "xsec_fb": xsec,
                "cut_sigma95_fb": cut_limit,
                "cut_sigma95_background_x0p25_fb": 0.8 * cut_limit,
                "cut_sigma95_background_x4_fb": 1.25 * cut_limit,
            }
            if include_shape:
                shape_limit = cut_limit / 1.1
                row.update(
                    {
                        "shape_sigma95_fb": shape_limit,
                        "shape_sigma95_background_x0p25_fb": 0.8
                        * shape_limit,
                        "shape_sigma95_background_x4_fb": 1.25 * shape_limit,
                    }
                )
            rows.append(row)
    return rows


def write_reusable_sm_optuna_study(path: Path, *, profile="full91"):
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method_version": "resolved-8b-c3d4-xgboost-v2.1",
        "status": "complete",
        "observable_set": "extended-91-v2",
        "selected_feature_profile": profile,
        "cv_folds": 5,
        "seed": 12345,
        "strategies_completed": ["sm-crossfit-v2", "pooled-crossfit-v2"],
    }
    (path / "method_manifest.json").write_text(json.dumps(manifest))
    fold_params = []
    for fold in range(5):
        params = {
            "n_estimators": 200 + 100 * fold,
            "max_depth": 2 + fold % 5,
            "learning_rate": 0.05,
            "min_child_weight": 1.0 + fold,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
        fold_params.append(params)
        history_dir = path / "sm-crossfit-v2" / "optuna"
        history_dir.mkdir(parents=True, exist_ok=True)
        (history_dir / f"fold_{fold}_history.json").write_text(
            json.dumps(
                {
                    "best_trial": 10 + fold,
                    "best_value": -0.5 - 0.01 * fold,
                    "trials": [
                        {
                            "number": 10 + fold,
                            "state": "COMPLETE",
                            "value": -0.5 - 0.01 * fold,
                            "params": params,
                            "user_attrs": {},
                        }
                    ],
                }
            )
        )
    return fold_params


def empty_shape_records():
    return [
        {
            "rotation": rotation,
            "validation": {
                "parameterized": False,
                "signal_rows": {},
                "background_rows": {},
            },
            "test": {
                "parameterized": False,
                "signal_rows": {},
                "background_rows": {},
            },
        }
        for rotation in range(5)
    ]


def fake_shape_payload(point, records, *, shared_candidates):
    del records, shared_candidates
    return {
        "kind": "result",
        "row": {
            "point_id": point.point_id,
            "c3": point.c3,
            "d4": point.d4,
            "status": "ok",
            "bin_count": 2,
            "shape_sigma95_fb": 10.0 + float(point.c3),
        },
        "warnings": [],
        "elapsed_seconds": 0.01,
    }


def successful_pyhf_limit(*args, **kwargs):
    """Deterministic pyhf stand-in that remains callable after POSIX fork."""

    del args, kwargs
    return {"status": "ok", "expected_median": 4.25}


def populated_shape_records(points, *, invalid_validation_signal=False):
    """Construct compact, statistically populated score records for shape tests."""

    background_scores = np.linspace(0.001, 0.999, 200, dtype=float)
    background_weights = np.full(200, 0.02, dtype=float)
    positive_signal_scores = np.linspace(0.01, 0.99, 80, dtype=float)
    positive_signal_weights = np.full(80, 0.025, dtype=float)
    invalid_signal_scores = np.concatenate(
        [np.full(25, 0.01, dtype=float), np.full(25, 0.99, dtype=float)]
    )
    # The total signed yield remains positive, but every multi-bin candidate
    # contains a negative low-score signal bin and is invalid for pyhf.
    invalid_signal_weights = np.concatenate(
        [np.full(25, -1.0, dtype=float), np.full(25, 2.0, dtype=float)]
    )

    records = []
    for rotation in range(5):
        validation_signal_rows = {}
        test_signal_rows = {}
        for point in points:
            validation_signal_rows[point.sample_id] = {
                "scores": (
                    invalid_signal_scores.copy()
                    if invalid_validation_signal
                    else positive_signal_scores.copy()
                ),
                "unit_xsec_weights": (
                    invalid_signal_weights.copy()
                    if invalid_validation_signal
                    else positive_signal_weights.copy()
                ),
                "scale": 1.0,
            }
            test_signal_rows[point.sample_id] = {
                "scores": positive_signal_scores.copy(),
                "unit_xsec_weights": positive_signal_weights.copy(),
                "scale": 1.0,
            }
        records.append(
            {
                "rotation": rotation,
                "validation": {
                    "parameterized": False,
                    "signal_rows": validation_signal_rows,
                    "background_rows": {
                        "background": {
                            "scores": background_scores.copy(),
                            "physical_weights": background_weights.copy(),
                            "scale": 1.0,
                        }
                    },
                },
                "test": {
                    "parameterized": False,
                    "signal_rows": test_signal_rows,
                    "background_rows": {
                        "background": {
                            "scores": background_scores.copy(),
                            "physical_weights": background_weights.copy(),
                            "scale": 1.0,
                        }
                    },
                },
            }
        )
    return records


@contextmanager
def mocked_mode_study_pipeline(*, shape_error=None, point_count=57):
    """Provide a fast variable-size pipeline while retaining real publication."""

    sm_samples = [sample("sm", "sm_signal", [1.0] * 5, [1.0] * 5)]
    grid_samples = [
        sample(f"grid-{index}", "grid_signal", [1.0] * 5, [1.0] * 5, index, 0)
        for index in range(point_count)
    ]
    background_samples = [
        sample("background", "background", [1.0] * 5, [1.0] * 5)
    ]
    # Smoke mode records size/mtime instead of hashing inputs.  Point every
    # synthetic sample at one real, read-only file so that metadata path is
    # exercised without creating dozens of test files.
    for synthetic_sample in [*sm_samples, *grid_samples, *background_samples]:
        synthetic_sample.path = Path(__file__)
        synthetic_sample.metadata = {
            "feature_source_completion": {
                "verified": True,
                "method": "test-fixture",
                "observed_events": 5,
                "expected_events": 5,
            }
        }
    samples_by_kind = {
        "sm_signal": sm_samples,
        "grid_signal": grid_samples,
        "background": background_samples,
    }

    class FakeModel:
        def save_model(self, path):
            Path(path).write_text("{}")

    def fake_load_samples(specs, *, kind, progress=None, **kwargs):
        del specs, kwargs
        loaded = samples_by_kind[kind]
        if progress is not None:
            for index, loaded_sample in enumerate(loaded, start=1):
                progress(index, len(loaded), loaded_sample)
        return loaded

    def point_payloads(limit_key="sigma95_fb"):
        return {
            point.point_id: {
                "point_id": point.point_id,
                "c3": point.c3,
                "d4": point.d4,
                limit_key: 5.0,
            }
            for point in grid_samples
        }

    def fake_fit_rotation(*args, rotation, params, **kwargs):
        del args, kwargs
        validation = {
            "rotation": rotation,
            "objective": 0.0,
            "points": point_payloads(),
            "signal_rows": {},
            "background_rows": {},
        }
        return FakeModel(), validation, {}, dict(params)

    def fake_test_rotation(*args, rotation, **kwargs):
        del args, kwargs
        return {
            "rotation": rotation,
            "points": point_payloads(),
            "signal_rows": {},
            "background_rows": {},
            "parameterized": False,
        }

    def fake_cut_results(*args, **kwargs):
        del args, kwargs
        return [
            {
                "point_id": point.point_id,
                "c3": point.c3,
                "d4": point.d4,
                "xsec_fb": 10.0,
                "feature_tree_efficiency": 0.8,
                "xgboost_efficiency": 0.6,
                "threshold_mean": 0.5,
                "background_yield": 1.0,
                "cut_sigma95_fb": 5.0,
            }
            for point in grid_samples
        ]

    def fake_validation_aggregate(*args, **kwargs):
        del args, kwargs
        rows = [
            {
                "point_id": point.point_id,
                "c3": point.c3,
                "d4": point.d4,
                "validation_cut_sigma95_fb": 5.0,
            }
            for point in grid_samples
        ]
        return rows, [5.0] * 5

    def fake_maps(rows, output_dir, prefix, **kwargs):
        del rows, kwargs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{prefix}_cut_exclusion_contour.pdf").write_text("map")
        return {
            "output_dir": str(output_dir),
            "prefix": prefix,
            "legacy_style_contours": {
                "cut": {"status": "ok"},
                "shape": {"status": "ok"},
            },
        }

    def fake_shape_results(*args, **kwargs):
        del args
        if shape_error is not None:
            raise shape_error
        rows = [
            {
                "point_id": point.point_id,
                "c3": point.c3,
                "d4": point.d4,
                "status": "ok",
                "bin_count": 2,
                "shape_sigma95_fb": 4.0,
                "pyhf_one_bin_sigma95_fb": 5.0,
            }
            for point in grid_samples
        ]
        metadata = {
            "strategy": kwargs.get("strategy"),
            "profile": kwargs.get("profile"),
            "shape_jobs": kwargs.get("shape_jobs", 1),
            "checkpoint_fingerprint": kwargs.get("checkpoint_fingerprint"),
            "checkpoint_dir": str(kwargs.get("checkpoint_dir")),
            "resumed_points": 0,
            "submitted_points": len(rows),
            "completed_points": len(rows),
            "retryable_points": [],
            "status": "complete",
        }
        return (rows, metadata) if kwargs.get("return_metadata") else rows

    def fake_coupling_holdout(*args, **kwargs):
        del args, kwargs
        rows = [
            {
                "point_id": point.point_id,
                "c3": point.c3,
                "d4": point.d4,
                "coupling_holdout_fold": index % 5,
                "holdout_to_event_crossfit_ratio": 1.0,
                "postfit_hhhbb_included": False,
            }
            for index, point in enumerate(grid_samples)
        ]
        return {
            "summary": {
                "status": "complete",
                "version": runner.COUPLING_HOLDOUT_VERSION,
                "point_count": len(rows),
                "median_holdout_to_event_crossfit_ratio": 1.0,
                "postfit_hhhbb_included": False,
            },
            "rows": rows,
        }

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(runner, "_load_samples", fake_load_samples))
        stack.enter_context(mock.patch.object(runner, "_source_commit", return_value="commit"))
        stack.enter_context(mock.patch.object(runner, "_sha256", return_value="hash"))
        stack.enter_context(
            mock.patch.object(runner, "_package_versions", return_value={"pyhf": "0.7.6"})
        )
        stack.enter_context(mock.patch.object(runner, "attach_model_metadata"))
        stack.enter_context(mock.patch.object(runner, "_fit_rotation", fake_fit_rotation))
        stack.enter_context(
            mock.patch.object(runner, "_evaluate_test_rotation", fake_test_rotation)
        )
        stack.enter_context(
            mock.patch.object(runner, "_aggregate_cut_results", fake_cut_results)
        )
        stack.enter_context(
            mock.patch.object(
                runner, "_aggregate_validation_crossfit", fake_validation_aggregate
            )
        )
        stack.enter_context(
            mock.patch.object(
                runner,
                "_sm_background_cutflow_rows",
                return_value=(
                    [
                        {
                            "sample_id": "background",
                            "sample_role": "background",
                            "is_signal": False,
                            "process_id": "background",
                            "description": "background",
                            "input_xsec_fb": 1.0,
                            "input_events": 5.0,
                            "xgboost_xsec_fb": 0.2,
                            "xgboost_events": 1.0,
                            "xgboost_events_error": 1.0,
                            "entries": 5,
                            "selected_entries": 1,
                        }
                    ],
                    [0.5] * 5,
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(
                runner,
                "_sm_signal_cutflow_rows",
                return_value=[
                    {
                        "sample_id": "grid-0",
                        "sample_role": "signal",
                        "is_signal": True,
                        "signal_component": "hhhh",
                        "point_id": "c3=0,d4=0",
                        "point_class": "standard-model-reference",
                        "representative_category": "SM reference",
                        "is_limit_representative": False,
                        "cut_signal_strength95": 10.0,
                        "theory_to_limit_ratio": 0.1,
                        "excluded_cut": False,
                        "c3": 0.0,
                        "d4": 0.0,
                        "process_id": "sm_hhhh",
                        "description": "SM gg -> hhhh -> 8b",
                        "input_xsec_fb": 1.0,
                        "input_events": 5.0,
                        "xgboost_xsec_fb": 0.1,
                        "xgboost_events": 0.5,
                        "xgboost_events_error": 0.25,
                        "entries": 5,
                        "selected_entries": 1,
                    }
                ],
            )
        )
        stack.enter_context(mock.patch.object(runner, "_write_standard_maps", fake_maps))
        stack.enter_context(
            mock.patch.object(runner, "_shape_fingerprint", return_value="fingerprint")
        )
        stack.enter_context(mock.patch.object(runner, "_shape_results", fake_shape_results))
        stack.enter_context(
            mock.patch.object(
                runner,
                "_parameterized_coupling_holdout_diagnostic",
                fake_coupling_holdout,
            )
        )
        yield grid_samples


def resolve_study_mode(study_mode, **overrides):
    arguments = {
        "study_mode": study_mode,
        "observable_set": "extended-91-v2",
        "feature_profile": None,
        "training_strategy": None,
        "optuna_trials": None,
        "max_events": None,
        "smoke_max_events": 2000,
        "run_shape": None,
        "hash_inputs": True,
    }
    arguments.update(overrides)
    return runner._resolve_study_mode(**arguments)


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

    def test_shape_runtime_options_are_validated_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            common = {
                "sm_signal_specs": [],
                "grid_signal_specs": [],
                "background_specs": [],
                "output_dir": directory,
            }
            with self.assertRaisesRegex(ValueError, "shape_jobs"):
                runner.run_c3d4_study(**common, shape_jobs=0)
            with self.assertRaisesRegex(ValueError, "progress_interval"):
                runner.run_c3d4_study(**common, progress_interval=0.0)

    def test_preview_mode_defaults_to_complete_core52_cut_study(self):
        policy = resolve_study_mode("preview")

        self.assertEqual(policy.name, "preview")
        self.assertEqual(policy.feature_profile, "core52")
        self.assertEqual(policy.training_strategy, "pooled-crossfit-v2")
        self.assertEqual(policy.optuna_trials, 0)
        self.assertIsNone(policy.max_events)
        self.assertFalse(policy.run_shape)
        self.assertFalse(policy.run_profile_ablation)
        self.assertFalse(policy.run_parameterized_gate)
        self.assertFalse(policy.run_coupling_holdout)
        self.assertTrue(policy.hash_inputs)
        self.assertEqual(policy.result_level, "preliminary-cut-only")
        self.assertTrue(policy.physics_result_valid)
        self.assertFalse(policy.paper_ready)
        self.assertEqual(
            policy.plot_watermark, "PRELIMINARY - SINGLE-BIN CUT RESULT"
        )

    def test_smoke_mode_defaults_to_truncated_nonphysics_contract(self):
        policy = resolve_study_mode("smoke")

        self.assertEqual(policy.name, "smoke")
        self.assertEqual(policy.feature_profile, "corrected28")
        self.assertEqual(policy.training_strategy, "sm-crossfit-v2")
        self.assertEqual(policy.optuna_trials, 0)
        self.assertEqual(policy.max_events, 2000)
        self.assertFalse(policy.run_shape)
        self.assertFalse(policy.run_profile_ablation)
        self.assertFalse(policy.run_parameterized_gate)
        self.assertFalse(policy.run_coupling_holdout)
        self.assertFalse(policy.hash_inputs)
        self.assertEqual(policy.result_level, "non-physics-smoke")
        self.assertFalse(policy.physics_result_valid)
        self.assertFalse(policy.paper_ready)
        self.assertEqual(policy.plot_watermark, "NON-PHYSICS SMOKE TEST")

    def test_fast_sm_mode_uses_full91_fixed_sm_crossfit_and_shape_likelihood(self):
        policy = resolve_study_mode("fast-sm")

        self.assertEqual(policy.name, "fast-sm")
        self.assertEqual(policy.feature_profile, "full91")
        self.assertEqual(policy.training_strategy, "sm-crossfit-v2")
        self.assertEqual(policy.optuna_trials, 0)
        self.assertIsNone(policy.max_events)
        self.assertTrue(policy.run_shape)
        self.assertFalse(policy.run_profile_ablation)
        self.assertFalse(policy.run_parameterized_gate)
        self.assertFalse(policy.run_coupling_holdout)
        self.assertTrue(policy.hash_inputs)
        self.assertEqual(policy.result_level, "fixed-parameter-full")
        self.assertTrue(policy.physics_result_valid)
        self.assertTrue(policy.paper_ready)
        self.assertIsNone(policy.plot_watermark)

        with self.assertRaisesRegex(ValueError, "requires sm-crossfit-v2"):
            resolve_study_mode(
                "fast-sm", training_strategy="pooled-crossfit-v2"
            )
        with self.assertRaisesRegex(ValueError, "fixed XGBoost parameters"):
            resolve_study_mode("fast-sm", optuna_trials=1)

    def test_fast_pooled_mode_uses_only_fixed_full91_pooled_crossfit(self):
        policy = resolve_study_mode("fast-pooled")

        self.assertEqual(policy.name, "fast-pooled")
        self.assertEqual(policy.feature_profile, "full91")
        self.assertEqual(policy.training_strategy, "pooled-crossfit-v2")
        self.assertEqual(policy.optuna_trials, 0)
        self.assertIsNone(policy.max_events)
        self.assertTrue(policy.run_shape)
        self.assertFalse(policy.run_profile_ablation)
        self.assertFalse(policy.run_parameterized_gate)
        self.assertFalse(policy.run_coupling_holdout)
        self.assertTrue(policy.hash_inputs)
        self.assertEqual(policy.result_level, "fixed-parameter-full")
        self.assertTrue(policy.physics_result_valid)
        self.assertTrue(policy.paper_ready)
        self.assertIsNone(policy.plot_watermark)

        with self.assertRaisesRegex(ValueError, "requires pooled-crossfit-v2"):
            resolve_study_mode(
                "fast-pooled", training_strategy="sm-crossfit-v2"
            )
        with self.assertRaisesRegex(ValueError, "fixed XGBoost parameters"):
            resolve_study_mode("fast-pooled", optuna_trials=1)

    def test_fast_parameterized_uses_fixed_full91_and_coupling_holdout(self):
        policy = resolve_study_mode("fast-parameterized")

        self.assertEqual(policy.name, "fast-parameterized")
        self.assertEqual(policy.feature_profile, "full91")
        self.assertEqual(
            policy.training_strategy,
            "parameterized-crossfit-v1",
        )
        self.assertEqual(policy.optuna_trials, 0)
        self.assertIsNone(policy.max_events)
        self.assertTrue(policy.run_shape)
        self.assertFalse(policy.run_profile_ablation)
        self.assertFalse(policy.run_parameterized_gate)
        self.assertTrue(policy.run_coupling_holdout)
        self.assertTrue(policy.hash_inputs)
        self.assertEqual(policy.result_level, "fixed-parameter-full")
        self.assertTrue(policy.physics_result_valid)
        self.assertTrue(policy.paper_ready)
        self.assertIsNone(policy.plot_watermark)

        with self.assertRaisesRegex(
            ValueError, "requires parameterized-crossfit-v1"
        ):
            resolve_study_mode(
                "fast-parameterized",
                training_strategy="pooled-crossfit-v2",
            )
        with self.assertRaisesRegex(ValueError, "fixed XGBoost parameters"):
            resolve_study_mode("fast-parameterized", optuna_trials=1)

    def test_full_mode_retains_the_current_complete_study_defaults(self):
        policy = resolve_study_mode("full")

        self.assertEqual(policy.name, "full")
        self.assertIsNone(policy.feature_profile)
        self.assertEqual(policy.training_strategy, "pooled-crossfit-v2")
        self.assertEqual(policy.optuna_trials, 40)
        self.assertIsNone(policy.max_events)
        self.assertTrue(policy.run_shape)
        self.assertTrue(policy.run_profile_ablation)
        self.assertTrue(policy.run_parameterized_gate)
        self.assertFalse(policy.run_coupling_holdout)
        self.assertTrue(policy.hash_inputs)
        self.assertEqual(policy.result_level, "full")
        self.assertTrue(policy.physics_result_valid)
        self.assertTrue(policy.paper_ready)
        self.assertIsNone(policy.plot_watermark)

    def test_final_manifest_earns_paper_ready_only_after_successful_full_run(self):
        with tempfile.TemporaryDirectory() as directory, mocked_mode_study_pipeline():
            output = Path(directory)
            with redirect_stdout(io.StringIO()):
                runner.run_c3d4_study(
                    sm_signal_specs=[{}],
                    grid_signal_specs=[{}] * 57,
                    background_specs=[{}],
                    output_dir=output,
                    feature_profile="corrected28",
                    training_strategy="sm-crossfit-v2",
                    optuna_trials=0,
                    study_mode="full",
                )

            manifest = json.loads((output / "method_manifest.json").read_text())
            summary = json.loads((output / "study_summary.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["result_level"], "full")
            self.assertTrue(manifest["physics_result_valid"])
            self.assertTrue(manifest["paper_ready"])
            self.assertTrue(manifest["feature_source_completion_verified"])
            self.assertTrue(manifest["uses_complete_event_samples"])
            self.assertTrue(summary["paper_ready"])
            self.assertTrue(summary["manifest"]["paper_ready"])

    def test_fast_sm_run_accepts_a_dynamic_grid_and_skips_tuning(self):
        with tempfile.TemporaryDirectory() as directory, mocked_mode_study_pipeline(
            point_count=4
        ):
            output = Path(directory)
            terminal = io.StringIO()
            with redirect_stdout(terminal):
                runner.run_c3d4_study(
                    sm_signal_specs=[{}],
                    grid_signal_specs=[{}] * 4,
                    background_specs=[{}],
                    output_dir=output,
                    study_mode="fast-sm",
                )

            manifest = json.loads((output / "method_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["study_mode"], "fast-sm")
            self.assertEqual(manifest["grid_signal_point_count"], 4)
            self.assertEqual(manifest["optuna_trials_per_fold"], 0)
            self.assertEqual(manifest["requested_training_strategy"], None)
            self.assertIn("SM XGBoost background cutflow / rates", terminal.getvalue())
            cutflow_dir = output / "sm-crossfit-v2"
            self.assertTrue((cutflow_dir / "sm_background_cutflow.csv").exists())
            self.assertTrue(
                (cutflow_dir / "sm_background_only_cutflow.csv").exists()
            )
            self.assertTrue((cutflow_dir / "sm_signal_cutflow.csv").exists())
            cutflow = json.loads(
                (cutflow_dir / "sm_background_cutflow.json").read_text()
            )
            self.assertEqual(cutflow["thresholds_by_fold"], [0.5] * 5)
            self.assertEqual(cutflow["rows"][0]["input_events"], 5.0)
            self.assertEqual(cutflow["rows"][0]["sample_role"], "signal")
            self.assertEqual(len(cutflow["signal_rows"]), 1)
            self.assertEqual(len(cutflow["background_rows"]), 1)
            self.assertTrue(
                cutflow["signal_rows_are_excluded_from_background_total"]
            )
            self.assertTrue(
                cutflow["signal_rows_are_alternative_coupling_hypotheses"]
            )
            self.assertEqual(len(cutflow["signal_totals_by_point"]), 1)
            self.assertFalse(
                cutflow["totals_by_role"]["signal"][
                    "additive_across_coupling_points"
                ]
            )
            self.assertAlmostEqual(
                cutflow["totals_by_role"]["background"]["xgboost_events"],
                1.0,
            )
            self.assertEqual(
                manifest["mode_policy"]["training_strategy"], "sm-crossfit-v2"
            )
            self.assertTrue(manifest["score_shape_enabled"])
            self.assertTrue((output / "sm-crossfit-v2").is_dir())
            self.assertFalse((output / "pooled-crossfit-v2").exists())

    def test_fast_pooled_run_builds_only_pooled_strategy_and_cutflow(self):
        with tempfile.TemporaryDirectory() as directory, mocked_mode_study_pipeline(
            point_count=4
        ):
            output = Path(directory)
            terminal = io.StringIO()
            with redirect_stdout(terminal):
                summary = runner.run_c3d4_study(
                    sm_signal_specs=[{}],
                    grid_signal_specs=[{}] * 4,
                    background_specs=[{}],
                    output_dir=output,
                    study_mode="fast-pooled",
                )

            manifest = json.loads((output / "method_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["study_mode"], "fast-pooled")
            self.assertEqual(
                manifest["mode_policy"]["training_strategy"],
                "pooled-crossfit-v2",
            )
            self.assertEqual(
                manifest["strategies_requested"], ["pooled-crossfit-v2"]
            )
            self.assertEqual(
                manifest["strategies_completed"], ["pooled-crossfit-v2"]
            )
            self.assertEqual(
                list(summary["strategy_results"]), ["pooled-crossfit-v2"]
            )
            self.assertEqual(
                runner._manifest_expected_strategies(manifest),
                ["pooled-crossfit-v2"],
            )
            self.assertFalse((output / "sm-crossfit-v2").exists())
            pooled_dir = output / "pooled-crossfit-v2"
            self.assertTrue((pooled_dir / "shape_results.json").exists())
            self.assertTrue((pooled_dir / "cut_results.json").exists())
            self.assertTrue((pooled_dir / "sm_background_cutflow.csv").exists())
            cutflow = json.loads(
                (pooled_dir / "sm_background_cutflow.json").read_text()
            )
            self.assertEqual(
                cutflow["classifier_strategy"], "pooled-crossfit-v2"
            )
            self.assertEqual(
                manifest["outputs"]["sm_background_cutflow"][
                    "classifier_strategy"
                ],
                "pooled-crossfit-v2",
            )
            self.assertIn(
                "Classifier strategy: pooled-crossfit-v2",
                terminal.getvalue(),
            )

    def test_fast_parameterized_runs_directly_with_holdout_and_no_gate(self):
        with tempfile.TemporaryDirectory() as directory, mocked_mode_study_pipeline(
            point_count=7
        ):
            output = Path(directory)
            terminal = io.StringIO()
            with redirect_stdout(terminal):
                summary = runner.run_c3d4_study(
                    sm_signal_specs=[{}],
                    grid_signal_specs=[{}] * 7,
                    background_specs=[{}],
                    output_dir=output,
                    study_mode="fast-parameterized",
                )

            manifest = json.loads((output / "method_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["study_mode"], "fast-parameterized")
            self.assertEqual(
                manifest["mode_policy"]["training_strategy"],
                "parameterized-crossfit-v1",
            )
            self.assertTrue(
                manifest["mode_policy"]["coupling_holdout_enabled"]
            )
            self.assertFalse(
                manifest["mode_policy"]["parameterized_gate_enabled"]
            )
            self.assertEqual(
                manifest["strategies_requested"],
                ["parameterized-crossfit-v1"],
            )
            self.assertEqual(
                manifest["strategies_completed"],
                ["parameterized-crossfit-v1"],
            )
            self.assertEqual(
                list(summary["strategy_results"]),
                ["parameterized-crossfit-v1"],
            )
            self.assertEqual(
                runner._manifest_expected_strategies(manifest),
                ["parameterized-crossfit-v1"],
            )
            parameterized = manifest["parameterized_classifier"]
            self.assertEqual(parameterized["status"], "complete")
            self.assertFalse(parameterized["gate_applied"])
            self.assertEqual(parameterized["optuna_trials_per_fold"], 0)
            self.assertEqual(
                parameterized["coupling_holdout"]["point_count"],
                7,
            )
            strategy_dir = output / "parameterized-crossfit-v1"
            self.assertTrue((strategy_dir / "shape_results.json").exists())
            self.assertTrue((strategy_dir / "cut_results.json").exists())
            self.assertTrue(
                (strategy_dir / "coupling_holdout" / "point_results.csv").exists()
            )
            self.assertTrue(
                (strategy_dir / "coupling_holdout" / "summary.json").exists()
            )
            self.assertTrue(
                (strategy_dir / "sm_background_cutflow.csv").exists()
            )
            self.assertFalse((output / "parameterized_classifier_gate.json").exists())
            self.assertIn(
                "Classifier strategy: parameterized-crossfit-v1",
                terminal.getvalue(),
            )

    def test_fast_sm_reuses_completed_fold_specific_optuna_parameters(self):
        with tempfile.TemporaryDirectory() as directory, mocked_mode_study_pipeline(
            point_count=4
        ):
            directory = Path(directory)
            source = directory / "source"
            expected_params = write_reusable_sm_optuna_study(source)
            output = directory / "output"

            with redirect_stdout(io.StringIO()):
                runner.run_c3d4_study(
                    sm_signal_specs=[{}],
                    grid_signal_specs=[{}] * 4,
                    background_specs=[{}],
                    output_dir=output,
                    study_mode="fast-sm",
                    reuse_sm_optuna_from=source,
                )

            manifest = json.loads((output / "method_manifest.json").read_text())
            reuse = manifest["reused_sm_optuna"]
            self.assertEqual(reuse["status"], "reused")
            self.assertEqual(Path(reuse["source_study"]), source.resolve())
            self.assertEqual(len(reuse["folds"]), 5)
            for fold, expected in enumerate(expected_params):
                history = json.loads(
                    (
                        output
                        / "sm-crossfit-v2"
                        / "optuna"
                        / f"fold_{fold}_history.json"
                    ).read_text()
                )
                self.assertEqual(history["status"], "reused")
                self.assertEqual(history["best_params"], expected)
                self.assertEqual(history["trials"], [])

    def test_reused_optuna_requires_matching_profile_and_fast_sm_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            write_reusable_sm_optuna_study(source, profile="core52")
            with self.assertRaisesRegex(ValueError, "different selected feature profile"):
                runner._load_reused_sm_optuna(
                    source,
                    observable_set="extended-91-v2",
                    profile="full91",
                    n_folds=5,
                    seed=12345,
                )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "only in fast-sm"):
                runner.run_c3d4_study(
                    sm_signal_specs=[],
                    grid_signal_specs=[],
                    background_specs=[],
                    output_dir=directory,
                    study_mode="preview",
                    reuse_sm_optuna_from="old-study",
                )

    def test_physics_modes_reject_unverified_extended_feature_sources(self):
        for mode in (
            "preview",
            "fast-sm",
            "fast-pooled",
            "fast-parameterized",
            "full",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, mocked_mode_study_pipeline() as grid_samples:
                grid_samples[0].metadata = {}
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(
                        ValueError, "requires verified complete extended-v2"
                    ):
                        runner.run_c3d4_study(
                            sm_signal_specs=[{}],
                            grid_signal_specs=[{}] * 57,
                            background_specs=[{}],
                            output_dir=directory,
                            feature_profile="corrected28",
                            training_strategy=(
                                "parameterized-crossfit-v1"
                                if mode == "fast-parameterized"
                                else (
                                    "pooled-crossfit-v2"
                                    if mode == "fast-pooled"
                                    else "sm-crossfit-v2"
                                )
                            ),
                            optuna_trials=0,
                            study_mode=mode,
                        )

    def test_quick_mode_final_manifests_remain_nonfinal(self):
        for mode in ("preview", "smoke"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, mocked_mode_study_pipeline():
                output = Path(directory)
                with redirect_stdout(io.StringIO()):
                    runner.run_c3d4_study(
                        sm_signal_specs=[{}],
                        grid_signal_specs=[{}] * 57,
                        background_specs=[{}],
                        output_dir=output,
                        feature_profile="corrected28",
                        training_strategy="sm-crossfit-v2",
                        optuna_trials=0,
                        study_mode=mode,
                        smoke_max_events=5,
                    )

                manifest = json.loads((output / "method_manifest.json").read_text())
                self.assertEqual(manifest["status"], "complete")
                self.assertFalse(manifest["paper_ready"])
                self.assertTrue(manifest["feature_source_completion_verified"])
                self.assertEqual(
                    manifest["uses_complete_event_samples"], mode == "preview"
                )
                self.assertFalse(
                    json.loads((output / "study_summary.json").read_text())[
                        "paper_ready"
                    ]
                )

    def test_shape_failure_preserves_nonfinal_cut_preview_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory, mocked_mode_study_pipeline(
            shape_error=RuntimeError("deliberate shape failure")
        ):
            output = Path(directory)
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "deliberate shape failure"):
                    runner.run_c3d4_study(
                        sm_signal_specs=[{}],
                        grid_signal_specs=[{}] * 57,
                        background_specs=[{}],
                        output_dir=output,
                        feature_profile="corrected28",
                        training_strategy="sm-crossfit-v2",
                        optuna_trials=0,
                        study_mode="full",
                    )

            manifest = json.loads((output / "method_manifest.json").read_text())
            preview = json.loads(
                (
                    output
                    / "sm-crossfit-v2"
                    / "cut_preview"
                    / "status.json"
                ).read_text()
            )
            self.assertEqual(manifest["status"], "incomplete")
            self.assertFalse(manifest["paper_ready"])
            self.assertEqual(preview["status"], "complete")
            self.assertEqual(preview["result_level"], "preliminary-cut-only")
            self.assertFalse(preview["paper_ready"])
            self.assertTrue(Path(preview["cut_results_json"]).exists())
            self.assertTrue(Path(preview["cut_exclusion_map"]).exists())

    def test_full_rerun_quarantines_stale_parameterized_results_before_gate(self):
        with tempfile.TemporaryDirectory() as directory, mocked_mode_study_pipeline():
            output = Path(directory)
            parameterized = output / "parameterized-crossfit-v1"
            maps = parameterized / "maps"
            preview = parameterized / "cut_preview"
            maps.mkdir(parents=True)
            preview.mkdir()
            (parameterized / "cut_results.csv").write_text("old-cut")
            (parameterized / "shape_results_status.json").write_text(
                '{"status":"complete"}'
            )
            (maps / "old.pdf").write_text("old-map")
            (preview / "status.json").write_text('{"status":"complete"}')
            (output / "parameterized_classifier_gate.json").write_text(
                '{"passed":true}'
            )

            original_fit = runner._fit_rotation
            fit_started = []

            def fit_after_quarantine(*args, **kwargs):
                fit_started.append(True)
                self.assertFalse((parameterized / "cut_results.csv").exists())
                self.assertFalse((parameterized / "maps").exists())
                self.assertFalse(
                    (output / "parameterized_classifier_gate.json").exists()
                )
                return original_fit(*args, **kwargs)

            with mock.patch.object(
                runner, "_fit_rotation", side_effect=fit_after_quarantine
            ), redirect_stdout(io.StringIO()):
                runner.run_c3d4_study(
                    sm_signal_specs=[{}],
                    grid_signal_specs=[{}] * 57,
                    background_specs=[{}],
                    output_dir=output,
                    feature_profile="corrected28",
                    training_strategy="sm-crossfit-v2",
                    optuna_trials=0,
                    study_mode="full",
                )

            manifest = json.loads((output / "method_manifest.json").read_text())
            archive = Path(
                manifest["previous_output_archives"][
                    "parameterized-crossfit-v1"
                ]
            )
            self.assertFalse((parameterized / "cut_results.csv").exists())
            self.assertFalse(
                (parameterized / "shape_results_status.json").exists()
            )
            self.assertFalse((parameterized / "maps").exists())
            self.assertFalse((parameterized / "cut_preview").exists())
            self.assertTrue(fit_started)
            self.assertEqual((archive / "cut_results.csv").read_text(), "old-cut")
            self.assertTrue((archive / "maps" / "old.pdf").exists())
            self.assertTrue((archive / "cut_preview" / "status.json").exists())
            gate_archive = Path(
                manifest["previous_parameterized_gate_archive"]
            )
            self.assertEqual(gate_archive.read_text(), '{"passed":true}')

    def test_study_modes_reject_ambiguous_truncation_and_expensive_stages(self):
        with self.assertRaisesRegex(ValueError, "preview mode requires complete"):
            resolve_study_mode("preview", max_events=100)
        with self.assertRaisesRegex(ValueError, "full mode requires complete"):
            resolve_study_mode("full", max_events=100)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            resolve_study_mode("full", optuna_trials=-1)

        for mode in ("preview", "smoke"):
            with self.subTest(mode=mode, option="optuna"):
                with self.assertRaisesRegex(ValueError, "fixed XGBoost parameters"):
                    resolve_study_mode(mode, optuna_trials=1)
            with self.subTest(mode=mode, option="shape"):
                with self.assertRaisesRegex(ValueError, "does not run the pyhf"):
                    resolve_study_mode(mode, run_shape=True)
            with self.subTest(mode=mode, option="parameterized"):
                with self.assertRaisesRegex(ValueError, "parameterized training"):
                    resolve_study_mode(
                        mode, training_strategy="parameterized-crossfit-v1"
                    )

    def test_postfit_hhhbb_shape_modes_reach_postfit_input_loading(self):
        for mode in ("fast-sm", "fast-pooled", "fast-parameterized"):
            loaded_kinds = []

            def stop_at_postfit(*args, kind, **kwargs):
                del args, kwargs
                loaded_kinds.append(kind)
                if kind == "postfit_hhhbb_signal":
                    raise RuntimeError("postfit-shape-input-reached")
                return []

            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                with mock.patch.object(
                    runner, "_load_samples", side_effect=stop_at_postfit
                ), self.assertRaisesRegex(
                    RuntimeError, "postfit-shape-input-reached"
                ):
                    runner.run_c3d4_study(
                        sm_signal_specs=[],
                        grid_signal_specs=[],
                        hhhbb_signal_specs=[{"path": "not-loaded.root"}],
                        background_specs=[],
                        output_dir=directory,
                        study_mode=mode,
                        run_shape=True,
                    )
            self.assertIn("postfit_hhhbb_signal", loaded_kinds)

    def test_sm_hh4b_reaches_postfit_loading_in_every_active_c3d4_mode(self):
        for mode in (
            "fast-sm",
            "fast-pooled",
            "fast-parameterized",
            "full",
        ):
            loaded_kinds = []

            def stop_at_sm_hh4b(*args, kind, **kwargs):
                del args, kwargs
                loaded_kinds.append(kind)
                if kind == "postfit_sm_hh4b_signal":
                    raise RuntimeError("sm-hh4b-input-reached")
                return []

            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                with mock.patch.object(
                    runner, "_load_samples", side_effect=stop_at_sm_hh4b
                ), self.assertRaisesRegex(
                    RuntimeError, "sm-hh4b-input-reached"
                ):
                    runner.run_c3d4_study(
                        sm_signal_specs=[],
                        grid_signal_specs=[],
                        sm_hh4b_signal_specs=[{"path": "not-loaded.root"}],
                        background_specs=[],
                        output_dir=directory,
                        study_mode=mode,
                        run_shape=False,
                    )
            self.assertIn("postfit_sm_hh4b_signal", loaded_kinds)

    def test_study_output_directory_rejects_cross_mode_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "method_manifest.json").write_text(
                json.dumps({"study_mode": "preview"})
            )
            runner._validate_study_output_mode(output, "preview")
            with self.assertRaisesRegex(ValueError, "belongs to 'preview' mode"):
                runner._validate_study_output_mode(output, "full")

    def test_study_output_directory_fails_closed_on_malformed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "method_manifest.json").write_text("{not-json")

            with self.assertRaisesRegex(ValueError, "unreadable or empty"):
                runner._validate_study_output_mode(output, "preview")

            runner._record_study_failure(
                output,
                "failed",
                ValueError("mode ownership unknown"),
                study_mode="preview",
            )
            self.assertEqual(
                (output / "method_manifest.json").read_text(), "{not-json"
            )
            self.assertEqual(len(list((output / "failed_attempts").glob("*.json"))), 1)

    def test_rejected_cross_mode_attempt_preserves_running_owner_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = {
                "study_mode": "full",
                "status": "running",
                "selected_feature_profile": "core52",
            }
            progress = {
                "study_mode": "full",
                "status": "running",
                "phase": "feature-profiles",
            }
            (output / "method_manifest.json").write_text(json.dumps(manifest))
            (output / "study_progress.json").write_text(json.dumps(progress))

            with self.assertRaisesRegex(ValueError, "belongs to 'full' mode"):
                runner._validate_study_output_mode(output, "preview")
            runner._record_study_failure(
                output,
                "failed",
                ValueError("cross-mode output directory"),
                study_mode="preview",
            )

            self.assertEqual(
                json.loads((output / "method_manifest.json").read_text()),
                manifest,
            )
            self.assertEqual(
                json.loads((output / "study_progress.json").read_text()),
                progress,
            )
            attempt_files = list((output / "failed_attempts").glob("*.json"))
            self.assertEqual(len(attempt_files), 1)
            attempt = json.loads(attempt_files[0].read_text())
            self.assertEqual(attempt["attempted_study_mode"], "preview")
            self.assertEqual(attempt["existing_manifest_study_mode"], "full")

    def test_study_mode_is_part_of_the_training_run_fingerprint(self):
        common = {
            "observable_set": "extended-91-v2",
            "profile": "core52",
            "strategy": "pooled-crossfit-v2",
            "rotation": 0,
            "n_folds": 5,
            "seed": 12345,
            "source_commit": "abc",
            "fold_digest": "folds",
            "normalization_inputs": {"luminosity_fb_inverse": 3000.0},
            "input_hashes": {"sample.root": "hash"},
            "package_versions": {"xgboost": "3.0.2"},
        }
        full = runner._run_fingerprint(**common, study_mode="full")
        preview = runner._run_fingerprint(**common, study_mode="preview")
        smoke = runner._run_fingerprint(**common, study_mode="smoke")

        self.assertEqual(len({full, preview, smoke}), 3)

    def test_cut_preview_publishes_annotated_rows_and_watermark(self):
        policy = resolve_study_mode("preview")
        rows = [{"point_id": "c3=0,d4=0", "c3": 0.0, "d4": 0.0}]
        runner._annotate_result_rows(rows, policy)

        self.assertEqual(rows[0]["study_mode"], "preview")
        self.assertEqual(rows[0]["result_level"], "preliminary-cut-only")
        self.assertTrue(rows[0]["physics_result_valid"])
        self.assertFalse(rows[0]["paper_ready"])
        self.assertTrue(rows[0]["uses_complete_event_samples"])
        self.assertFalse(rows[0]["score_shape_included"])

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "_write_json_atomic"
        ) as write_status, mock.patch.object(
            runner, "_write_rows"
        ) as write_rows, mock.patch.object(
            runner, "_write_json"
        ) as write_json, mock.patch.object(
            runner, "_write_standard_maps"
        ) as write_maps:
            write_maps.return_value = {
                "legacy_style_contours": {
                    "cut": {"status": "ok"},
                    "shape": {"status": "skipped"},
                }
            }
            strategy_dir = Path(directory) / "pooled-crossfit-v2"
            watermark = policy.plot_watermark
            payload = runner._publish_cut_preview(
                rows,
                strategy_dir,
                strategy="pooled-crossfit-v2",
                policy=policy,
                watermark=watermark,
            )

        self.assertEqual(write_status.call_count, 2)
        running = write_status.call_args_list[0].args[1]
        complete = write_status.call_args_list[1].args[1]
        self.assertEqual(running["status"], "running")
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(running["watermark"], watermark)
        self.assertEqual(complete["watermark"], watermark)
        self.assertEqual(payload, complete)
        write_rows.assert_called_once_with(
            strategy_dir / "cut_preview" / "cut_results.csv", rows
        )
        write_json.assert_called_once_with(
            strategy_dir / "cut_preview" / "cut_results.json", rows
        )
        write_maps.assert_called_once_with(
            rows,
            strategy_dir / "cut_preview" / "maps",
            "pooled-crossfit-v2_preview",
            watermark=watermark,
            legacy_contours=True,
            luminosity=3000.0,
            contour_c3_range=(-20.0, 20.0),
            contour_d4_range=(-500.0, 500.0),
            contour_grid_bins=301,
            contour_interpolation="linear",
            xsec_source_dir=runner.DEFAULT_HHHH_XSEC_SOURCE_DIR,
            xsec_overlay=True,
        )
        self.assertEqual(
            payload["legacy_style_contours"]["cut"]["status"], "ok"
        )

    def test_legacy_contour_spec_uses_pointwise_physical_exclusion_ratios(self):
        rows = legacy_contour_rows()
        cut = runner._legacy_contour_spec(rows, "cut")
        shape = runner._legacy_contour_spec(rows, "shape")

        self.assertEqual(cut["status"], "ok")
        self.assertEqual(shape["status"], "ok")
        expected_cut = {
            (row["c3"], row["d4"]): row["xsec_fb"]
            / row["cut_sigma95_fb"]
            for row in rows
        }
        expected_shape = {
            (row["c3"], row["d4"]): row["xsec_fb"]
            / row["shape_sigma95_fb"]
            for row in rows
        }
        for c3, d4, cut_ratio, shape_ratio in zip(
            cut["c3"], cut["d4"], cut["central_ratio"], shape["central_ratio"]
        ):
            self.assertAlmostEqual(cut_ratio, expected_cut[(c3, d4)])
            self.assertAlmostEqual(shape_ratio, expected_shape[(c3, d4)])
        self.assertTrue(cut["band_ordering_valid"])

    def test_legacy_contour_rejects_an_incomplete_manifest_point_set(self):
        rows = legacy_contour_rows()
        expected = [(row["c3"], row["d4"]) for row in rows]
        result = runner._legacy_contour_spec(
            rows[:-1],
            "shape",
            expected_coordinates=expected,
        )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["expected_point_count"], 9)
        self.assertEqual(result["usable_point_count"], 8)
        self.assertEqual(result["expectation_source"], "study-manifest")
        self.assertEqual(len(result["missing_coordinates"]), 1)

    def test_legacy_contour_rejects_an_incomplete_background_envelope(self):
        rows = legacy_contour_rows()
        rows[-1].pop("shape_sigma95_background_x4_fb")

        result = runner._legacy_contour_spec(rows, "shape")

        self.assertEqual(result["status"], "skipped")
        self.assertIn("incomplete shape background envelope", result["reason"])
        self.assertEqual(len(result["band_missing_coordinates"]), 1)

    def test_legacy_contour_rejects_reversed_background_envelope_limits(self):
        rows = legacy_contour_rows()
        rows[-1]["shape_sigma95_background_x0p25_fb"] = (
            1.2 * rows[-1]["shape_sigma95_fb"]
        )

        result = runner._legacy_contour_spec(rows, "shape")

        self.assertEqual(result["status"], "skipped")
        self.assertFalse(result["band_ordering_valid"])
        self.assertIn("invalid shape background-envelope ordering", result["reason"])

    def test_legacy_contour_rejects_duplicate_coupling_rows(self):
        rows = legacy_contour_rows()

        result = runner._legacy_contour_spec([*rows, dict(rows[0])], "cut")

        self.assertEqual(result["status"], "skipped")
        self.assertIn("duplicate cut rows", result["reason"])
        self.assertEqual(len(result["duplicate_coordinates"]), 1)

    def test_legacy_contour_binds_point_cross_sections_to_manifest(self):
        rows = legacy_contour_rows()
        expected_xsecs = {
            (row["c3"], row["d4"]): row["xsec_fb"] for row in rows
        }
        rows[0]["xsec_fb"] = 1.0

        result = runner._legacy_contour_spec(
            rows,
            "cut",
            expected_coordinates=sorted(expected_xsecs),
            expected_xsecs=expected_xsecs,
        )

        self.assertEqual(result["status"], "skipped")
        self.assertIn("do not match the study manifest", result["reason"])
        self.assertEqual(len(result["xsec_mismatches"]), 1)

    def test_constant_cross_section_surface_has_strictly_increasing_levels(self):
        levels = plot_style._make_hhhh_xsec_log_levels(np.ones((5, 5)))

        self.assertGreater(len(levels), 2)
        self.assertTrue(np.all(np.diff(levels) > 0.0))
        self.assertLess(levels[0], 1.0)
        self.assertGreater(levels[-1], 1.0)

    def test_cross_section_fit_rejects_rank_deficient_points(self):
        rows = [
            {
                "c3": 0.0,
                "d4": 0.0,
                "xsec_pb": 1.0,
                "xsec_error_pb": 0.1,
            }
            for _ in plot_style.DEFAULT_C3D4_CHEBYSHEV_TERMS
        ]

        fit = plot_style._fit_c3d4_chebyshev(
            rows,
            "xsec_pb",
            "xsec_error_pb",
            plot_style.DEFAULT_C3D4_CHEBYSHEV_TERMS,
            (-29.0, 31.0),
            (-699.0, 701.0),
        )

        self.assertEqual(fit["status"], "skipped")
        self.assertLess(fit["rank"], fit["n_terms"])
        self.assertIn("rank-deficient", fit["reason"])

    def test_legacy_style_contour_writes_png_pdf_band_and_watermark(self):
        rows = legacy_contour_rows()
        with tempfile.TemporaryDirectory() as directory:
            metadata = runner._write_legacy_style_exclusion_contours(
                rows,
                Path(directory),
                "pooled",
                limit_kind="cut",
                watermark="PRELIMINARY - SINGLE-BIN CUT RESULT",
                grid_bins=31,
                xsec_overlay=False,
            )

            self.assertEqual(metadata["status"], "ok")
            self.assertEqual(metadata["outputs"]["xsec"]["status"], "disabled")
            plot = metadata["outputs"]["no_xsec_atlas"]
            self.assertEqual(plot["status"], "ok")
            self.assertTrue(plot["limit_contour_drawn"])
            self.assertTrue(plot["background_band_drawn"])
            self.assertEqual(plot["background_boundary_count"], 2)
            self.assertEqual(
                plot["watermark"], "PRELIMINARY - SINGLE-BIN CUT RESULT"
            )
            self.assertTrue(plot["include_atlas"])
            self.assertGreater(Path(plot["png"]).stat().st_size, 0)
            self.assertGreater(Path(plot["pdf"]).stat().st_size, 0)

    def test_legacy_style_contour_labels_postfit_hhhbb_as_signal(self):
        rows = legacy_contour_rows()
        for row in rows:
            row["signal_components"] = "hhhh,hhhbb"
        with tempfile.TemporaryDirectory() as directory:
            metadata = runner._write_legacy_style_exclusion_contours(
                rows,
                Path(directory),
                "combined",
                limit_kind="cut",
                grid_bins=21,
                xsec_overlay=False,
            )

            plot = metadata["outputs"]["no_xsec_atlas"]
            self.assertEqual(plot["status"], "ok")
            self.assertIn("hhhg", plot["process_title"])
            self.assertIn("hhhg", plot["limit_label"])

    def test_legacy_style_contour_writes_both_cross_section_variants(self):
        def constant_surface(spec, c3_grid, d4_grid, **kwargs):
            del spec, d4_grid, kwargs
            return np.ma.asarray(np.ones_like(c3_grid)), {
                "status": "ok",
                "method": "test-constant-surface",
            }

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "_legacy_xsec_ratio_grid",
            side_effect=constant_surface,
        ):
            metadata = runner._write_legacy_style_exclusion_contours(
                legacy_contour_rows(),
                Path(directory),
                "pooled",
                limit_kind="cut",
                grid_bins=21,
                xsec_overlay=True,
            )

            for variant in ("xsec", "xsec_atlas", "no_xsec_atlas"):
                product = metadata["outputs"][variant]
                self.assertEqual(product["status"], "ok")
                self.assertGreater(Path(product["png"]).stat().st_size, 0)
                self.assertGreater(Path(product["pdf"]).stat().st_size, 0)

    def test_legacy_style_contour_skips_missing_shape_and_removes_stale_files(self):
        rows = legacy_contour_rows(include_shape=False)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            stale_paths = runner._legacy_contour_paths(
                directory, "preview", "shape"
            )
            for path in stale_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("stale")
                path.with_suffix(".pdf").write_text("stale")

            metadata = runner._write_legacy_style_exclusion_contours(
                rows,
                directory,
                "preview",
                limit_kind="shape",
                grid_bins=21,
                xsec_overlay=False,
            )

            self.assertEqual(metadata["status"], "skipped")
            self.assertIn("incomplete shape point set", metadata["reason"])
            for path in stale_paths.values():
                self.assertFalse(path.exists())
                self.assertFalse(path.with_suffix(".pdf").exists())

    def test_legacy_style_plot_omits_level_one_when_outside_ratio_range(self):
        rows = legacy_contour_rows(ratio_offset=2.0)
        with tempfile.TemporaryDirectory() as directory:
            metadata = runner._write_legacy_style_exclusion_contours(
                rows,
                Path(directory),
                "all_excluded",
                limit_kind="cut",
                grid_bins=21,
                xsec_overlay=False,
            )

            plot = metadata["outputs"]["no_xsec_atlas"]
            self.assertEqual(plot["status"], "ok")
            self.assertFalse(plot["limit_contour_drawn"])
            self.assertGreater(Path(plot["png"]).stat().st_size, 0)

    def test_standard_map_writer_records_the_legacy_contour_manifest(self):
        contour_metadata = {
            "style_version": runner.LEGACY_CONTOUR_STYLE_VERSION,
            "cut": {"status": "ok"},
            "shape": {"status": "skipped"},
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "_write_map"
        ), mock.patch.object(
            runner,
            "_write_legacy_style_contour_set",
            return_value=contour_metadata,
        ):
            output = Path(directory)
            runner._write_standard_maps(
                legacy_contour_rows(include_shape=False),
                output,
                "preview",
                legacy_contours=True,
            )

            saved = json.loads(
                (output / "legacy_contour_manifest.json").read_text()
            )
            self.assertEqual(saved, contour_metadata)

    def test_replot_existing_preview_tables_without_retraining(self):
        rows = legacy_contour_rows(include_shape=False)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "_write_legacy_style_contour_set",
            return_value={
                "style_version": runner.LEGACY_CONTOUR_STYLE_VERSION,
                "cut": {"status": "ok"},
                "shape": {"status": "skipped"},
            },
        ) as write_contours, mock.patch.object(
            runner, "_contour_product_count", return_value=1
        ):
            output = Path(directory)
            strategy = output / "sm-crossfit-v2"
            preview = strategy / "cut_preview"
            preview.mkdir(parents=True)
            (output / "method_manifest.json").write_text(
                json.dumps(
                    {
                        "study_mode": "preview",
                        "status": "complete",
                        "paper_ready": False,
                        "luminosity_fb_inverse": 3000.0,
                    }
                )
            )
            (strategy / "cut_results.json").write_text(json.dumps(rows))
            (strategy / "cut_results_status.json").write_text(
                json.dumps({"status": "complete"})
            )
            (preview / "cut_results.json").write_text(json.dumps(rows))
            (preview / "status.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "watermark": "PRELIMINARY - SINGLE-BIN CUT RESULT",
                    }
                )
            )

            payload = runner.replot_c3d4_study_contours(
                output,
                contour_grid_bins=21,
                xsec_overlay=False,
            )

            self.assertEqual(payload["status"], "complete")
            self.assertEqual(write_contours.call_count, 2)
            for call in write_contours.call_args_list:
                self.assertEqual(
                    call.kwargs["watermark"],
                    "PRELIMINARY - SINGLE-BIN CUT RESULT",
                )
            self.assertTrue((output / "contour_replot_manifest.json").exists())
            status = json.loads(
                (strategy / "cut_results_status.json").read_text()
            )
            self.assertEqual(
                status["legacy_style_contours"]["cut"]["status"], "ok"
            )

    def test_replot_rejects_a_luminosity_that_differs_from_the_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "method_manifest.json").write_text(
                json.dumps(
                    {
                        "study_mode": "preview",
                        "status": "complete",
                        "luminosity_fb_inverse": 139.0,
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                runner.replot_c3d4_study_contours(output, luminosity=3000.0)

    def test_replot_reuses_saved_plot_configuration_and_binds_null_source(self):
        manifest = {
            "luminosity_fb_inverse": 139.0,
            "legacy_contour_plots": {
                "c3_range": [-10.0, 12.0],
                "d4_range": [-150.0, 175.0],
                "grid_bins": 77,
                "xsec_overlay": False,
                "xsec_source_dir": None,
            },
        }

        config = runner._resolved_replot_config(
            manifest,
            luminosity=None,
            contour_c3_range=None,
            contour_d4_range=None,
            contour_grid_bins=None,
            xsec_source_dir=None,
            xsec_overlay=None,
        )

        self.assertEqual(config["luminosity"], 139.0)
        self.assertEqual(config["c3_range"], (-10.0, 12.0))
        self.assertEqual(config["d4_range"], (-150.0, 175.0))
        self.assertEqual(config["grid_bins"], 77)
        self.assertFalse(config["xsec_overlay"])
        self.assertIsNone(config["xsec_source_dir"])
        with self.assertRaisesRegex(ValueError, "source differs"):
            runner._resolved_replot_config(
                manifest,
                luminosity=None,
                contour_c3_range=None,
                contour_d4_range=None,
                contour_grid_bins=None,
                xsec_source_dir=Path("/different/source"),
                xsec_overlay=True,
            )

    def test_current_v2_manifest_requires_all_57_grid_points(self):
        manifest = {
            "method_version": runner.METHOD_VERSION,
            "inputs": [
                {"kind": "grid_signal", "c3": 0.0, "d4": 0.0},
            ],
        }

        with self.assertRaisesRegex(ValueError, "57-point"):
            runner._manifest_grid_coordinates(manifest)

    def test_current_manifest_accepts_a_declared_dynamic_grid_size(self):
        manifest = {
            "method_version": runner.METHOD_VERSION,
            "grid_signal_point_count": 4,
            "inputs": [
                {"kind": "grid_signal", "c3": 0.0, "d4": 0.0, "xsec_fb": 1.0},
                {"kind": "grid_signal", "c3": 1.0, "d4": 0.0, "xsec_fb": 2.0},
                {"kind": "grid_signal", "c3": 0.0, "d4": 1.0, "xsec_fb": 3.0},
                {"kind": "grid_signal", "c3": 1.0, "d4": 1.0, "xsec_fb": 4.0},
            ],
        }

        self.assertEqual(len(runner._manifest_grid_coordinates(manifest)), 4)
        self.assertEqual(len(runner._manifest_grid_point_xsecs(manifest)), 4)

    def test_clough_tocher_interpolation_reproduces_sampled_log_values(self):
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("SciPy is not installed in the lightweight test environment")
        rows = legacy_contour_rows()
        c3 = np.asarray([row["c3"] for row in rows], dtype=float)
        d4 = np.asarray([row["d4"] for row in rows], dtype=float)
        ratios = np.asarray(
            [row["xsec_fb"] / row["cut_sigma95_fb"] for row in rows],
            dtype=float,
        )
        c3_grid = c3.reshape(3, 3)
        d4_grid = d4.reshape(3, 3)

        result = runner._interpolate_log_point_values(
            c3,
            d4,
            ratios,
            c3_grid,
            d4_grid,
            method="clough-tocher",
        )

        self.assertIsNotNone(result)
        np.testing.assert_allclose(
            np.asarray(result), np.log10(ratios).reshape(3, 3), atol=1.0e-10
        )

    def test_replot_malformed_table_writes_failed_manifest_and_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            strategy = output / "sm-crossfit-v2"
            strategy.mkdir(parents=True)
            (output / "method_manifest.json").write_text(
                json.dumps(
                    {
                        "study_mode": "preview",
                        "status": "complete",
                        "luminosity_fb_inverse": 3000.0,
                        "requested_training_strategy": "sm-crossfit-v2",
                    }
                )
            )
            (strategy / "cut_results.json").write_text("{not-json")

            with self.assertRaisesRegex(ValueError, "No valid v2 contour"):
                runner.replot_c3d4_study_contours(
                    output,
                    contour_grid_bins=21,
                    xsec_overlay=False,
                )

            payload = json.loads(
                (output / "contour_replot_manifest.json").read_text()
            )
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["successful_plot_pairs"], 0)
            self.assertTrue(
                any("malformed" in issue for issue in payload["issues"])
            )

    def test_replot_valid_canonical_with_malformed_preview_is_partial(self):
        contour_metadata = {
            "style_version": runner.LEGACY_CONTOUR_STYLE_VERSION,
            "cut": {"status": "ok"},
            "shape": {"status": "skipped"},
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "_write_legacy_style_contour_set",
            return_value=contour_metadata,
        ), mock.patch.object(
            runner, "_contour_product_count", return_value=1
        ):
            output = Path(directory)
            strategy = output / "sm-crossfit-v2"
            preview = strategy / "cut_preview"
            preview.mkdir(parents=True)
            (output / "method_manifest.json").write_text(
                json.dumps(
                    {
                        "study_mode": "preview",
                        "status": "complete",
                        "paper_ready": False,
                        "luminosity_fb_inverse": 3000.0,
                        "requested_training_strategy": "sm-crossfit-v2",
                    }
                )
            )
            (strategy / "cut_results.json").write_text(
                json.dumps(legacy_contour_rows(include_shape=False))
            )
            (preview / "cut_results.json").write_text("not-json")

            payload = runner.replot_c3d4_study_contours(
                output,
                contour_grid_bins=21,
                xsec_overlay=False,
            )

            self.assertEqual(payload["status"], "partial")
            self.assertTrue(
                any("cut-preview table is malformed" in issue for issue in payload["issues"])
            )

    def test_fold_diagnostics_carry_an_explicit_nonfinal_mode_envelope(self):
        policy = resolve_study_mode("smoke")
        metadata = {
            **runner._result_labels(
                policy,
                paper_ready=False,
                score_shape_included=False,
            ),
            "result_role": "cross-fit-diagnostic",
        }
        validation = {
            "rotation": 0,
            "points": {
                "c3=0,d4=0": {
                    "point_id": "c3=0,d4=0",
                    "c3": 0.0,
                    "d4": 0.0,
                    "sigma95_fb": 1.0,
                }
            },
            "signal_rows": [],
            "background_rows": [],
        }
        rotations = [{"validation": validation, "test": validation}]

        compact = runner._compact_validation(
            validation, result_metadata=metadata
        )
        rows = runner._flatten_fold_points(
            rotations, "test", result_metadata=metadata
        )

        self.assertEqual(compact["result_metadata"], metadata)
        self.assertEqual(rows[0]["study_mode"], "smoke")
        self.assertEqual(rows[0]["result_level"], "non-physics-smoke")
        self.assertFalse(rows[0]["physics_result_valid"])
        self.assertFalse(rows[0]["paper_ready"])
        self.assertEqual(rows[0]["result_role"], "cross-fit-diagnostic")

    def test_top_level_failure_writes_terminal_progress_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "shape_jobs"):
                runner.run_c3d4_study(
                    sm_signal_specs=[],
                    grid_signal_specs=[],
                    background_specs=[],
                    output_dir=directory,
                    shape_jobs=0,
                )

            output = Path(directory)
            progress = json.loads((output / "study_progress.json").read_text())
            manifest = json.loads((output / "method_manifest.json").read_text())
            self.assertEqual(progress["status"], "failed")
            self.assertEqual(progress["last_error"]["error_type"], "ValueError")
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["last_error"]["error_type"], "ValueError")

    def test_pre_run_validation_failure_preserves_completed_campaign_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            old_progress = {
                "status": "complete",
                "phase": "complete",
                "current": {"selected_profile": "core52"},
            }
            old_manifest = {
                "status": "complete",
                "selected_feature_profile": "core52",
                "outputs": {"shape": "previous-shape-results.json"},
            }
            (output / "study_progress.json").write_text(json.dumps(old_progress))
            (output / "method_manifest.json").write_text(json.dumps(old_manifest))

            with self.assertRaisesRegex(ValueError, "shape_jobs"):
                runner.run_c3d4_study(
                    sm_signal_specs=[],
                    grid_signal_specs=[],
                    background_specs=[],
                    output_dir=output,
                    shape_jobs=0,
                )

            self.assertEqual(
                json.loads((output / "study_progress.json").read_text()),
                old_progress,
            )
            self.assertEqual(
                json.loads((output / "method_manifest.json").read_text()),
                old_manifest,
            )
            attempts = list((output / "failed_attempts").glob("attempt-*.json"))
            self.assertEqual(len(attempts), 1)
            attempt = json.loads(attempts[0].read_text())
            self.assertEqual(attempt["status"], "failed")
            self.assertEqual(attempt["error_type"], "ValueError")

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

    def test_coupling_holdout_assignment_is_deterministic_and_balanced(self):
        grid = [
            sample(
                f"p{index}",
                "grid_signal",
                [1] * 5,
                [1] * 5,
                index,
                10 * index,
            )
            for index in range(13)
        ]

        first = runner._coupling_holdout_assignments(
            grid,
            n_folds=5,
            seed=12345,
        )
        second = runner._coupling_holdout_assignments(
            list(reversed(grid)),
            n_folds=5,
            seed=12345,
        )

        self.assertEqual(first, second)
        self.assertEqual(set(first), {point.point_id for point in grid})
        counts = [sum(fold == index for fold in first.values()) for index in range(5)]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_parameterized_coupling_holdout_excludes_each_coordinate_once(self):
        grid = [
            sample(
                f"p{index}",
                "grid_signal",
                [1] * 5,
                [1] * 5,
                index,
                10 * index,
            )
            for index in range(7)
        ]
        assignments = {
            point.point_id: index % 5
            for index, point in enumerate(grid)
        }
        reference_records = [
            {
                "rotation": fold,
                "test": {
                    "points": {
                        point.point_id: {
                            "threshold": 0.4,
                            "cut_sigma95_fb": 5.0,
                        }
                        for point in grid
                    }
                },
            }
            for fold in range(5)
        ]
        training_sets = {}

        def fake_training_arrays(
            sm_samples,
            training_points,
            background_samples,
            *,
            rotation,
            **kwargs,
        ):
            del sm_samples, background_samples, kwargs
            training_sets[rotation] = {
                point.point_id for point in training_points
            }
            return (
                np.zeros((4, 93), dtype=float),
                np.asarray([1, 1, 0, 0], dtype=np.int8),
                np.ones(4, dtype=float),
            )

        def fake_validation(
            model,
            heldout,
            background_samples,
            *,
            rotation,
            **kwargs,
        ):
            del model, background_samples, kwargs
            return {
                "rotation": rotation,
                "objective": 0.0,
                "parameterized": True,
                "points": {
                    point.point_id: {
                        "threshold": 0.5,
                        "sigma95_fb": 6.0,
                    }
                    for point in heldout
                },
            }

        def fake_test(
            model,
            validation,
            heldout,
            background_samples,
            *,
            rotation,
            **kwargs,
        ):
            del model, validation, background_samples, kwargs
            return {
                "rotation": rotation,
                "parameterized": True,
                "points": {
                    point.point_id: {
                        "threshold": 0.5,
                        "cut_sigma95_fb": 6.0,
                    }
                    for point in heldout
                },
            }

        with mock.patch.object(
            runner,
            "_coupling_holdout_assignments",
            return_value=assignments,
        ), mock.patch.object(
            runner,
            "_training_arrays",
            side_effect=fake_training_arrays,
        ), mock.patch.object(
            runner,
            "_train_model",
            return_value=(object(), {"xgboost_split_nodes": 3}, {}),
        ), mock.patch.object(
            runner,
            "_validation_limits",
            side_effect=fake_validation,
        ), mock.patch.object(
            runner,
            "_evaluate_test_rotation",
            side_effect=fake_test,
        ):
            result = runner._parameterized_coupling_holdout_diagnostic(
                [],
                grid,
                [],
                reference_records,
                observable_set="extended-91-v2",
                profile="full91",
                n_folds=5,
                seed=12345,
                source_commit="test",
            )

        self.assertEqual(len(result["rows"]), len(grid))
        self.assertEqual(
            result["summary"]["median_holdout_to_event_crossfit_ratio"],
            1.2,
        )
        self.assertFalse(result["summary"]["postfit_hhhbb_included"])
        for point in grid:
            fold = assignments[point.point_id]
            self.assertNotIn(point.point_id, training_sets[fold])
            row = next(
                item
                for item in result["rows"]
                if item["point_id"] == point.point_id
            )
            self.assertEqual(row["coupling_holdout_fold"], fold)
            self.assertFalse(row["postfit_hhhbb_included"])

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

    def test_hhhg_point_name_maps_to_the_same_c3d4_coordinate(self):
        point = runner._parse_point(
            "run_gg_hhhg_4_-7.5_50.0_hhhbb_stage2"
            "-extended-v2-uniform-smear-v1_var.smearCMS.root"
        )

        self.assertEqual(point, (-7.5, 50.0))

    def test_sm_hh4b_point_name_maps_to_the_sm_coordinate(self):
        point = runner._parse_point(
            "HW-run_gg_hhbbbb_heft_4_0.0_0.0"
            "-extended-v2-uniform-smear-v1_var.smearCMS.root"
        )

        self.assertEqual(point, (0.0, 0.0))

    def test_parameterized_sm_hh4b_uses_sm_coordinates(self):
        sm_hh4b = sample(
            "sm-hh4b",
            "postfit_sm_hh4b_signal",
            [1] * 5,
            [0.2] * 5,
            0.0,
            0.0,
        )
        validation = {
            "points": {
                sm_hh4b.point_id: {
                    "threshold": 0.5,
                }
            }
        }
        with mock.patch.object(
            runner,
            "_predict",
            return_value=np.asarray([0.75]),
        ) as predict:
            result = runner._evaluate_postfit_signal_rotation(
                object(),
                validation,
                [sm_hh4b],
                rotation=0,
                n_folds=5,
                profile_indices=np.arange(91),
                parameterized=True,
            )

        self.assertEqual(
            result["points"][sm_hh4b.point_id]["signal_raw_entries"],
            1,
        )
        self.assertEqual(predict.call_args.args[-1], (0.0, 0.0))

    def test_sm_hh4b_aggregate_is_one_standalone_nonlimit_result(self):
        sm_hh4b = sample(
            "sm-hh4b",
            "postfit_sm_hh4b_signal",
            [1] * 5,
            [0.02] * 5,
            0.0,
            0.0,
        )
        sm_hh4b.unit_xsec_weights = np.full(5, 2.0)
        sm_hh4b.xsec_fb = 0.01
        sm_hh4b.rate_factor = 10.0
        rotations = []
        for fold in range(5):
            rotations.append(
                {
                    "points": {
                        sm_hh4b.point_id: {
                            "threshold": 0.4,
                            "signal_unit_yield": 2.0,
                            "signal_sumw2_unit": 4.0,
                            "signal_physical_yield": 0.02,
                            "signal_sumw2_physical": 0.0004,
                            "signal_raw_entries": 1,
                        }
                    }
                }
            )

        result = runner._aggregate_postfit_sm_hh4b_result(
            sm_hh4b,
            rotations,
            luminosity=1.0,
            strategy="fast-sm",
        )

        self.assertEqual(result["point_id"], "c3=0,d4=0")
        self.assertEqual(result["selected_raw_entries"], 5)
        self.assertAlmostEqual(result["xgboost_efficiency"], 1.0)
        self.assertAlmostEqual(result["nominal_selected_signal_yield"], 0.1)
        self.assertFalse(result["included_in_training"])
        self.assertFalse(result["included_in_background"])
        self.assertFalse(result["included_in_limits"])
        self.assertFalse(result["cross_section_fit_applied"])
        self.assertNotIn("cut_sigma95_fb", result)

    def test_sm_hh4b_cutflow_row_is_a_nonlimit_signal_reference(self):
        result = {
            "component": "sm_hh4b",
            "classifier_strategy": "sm-crossfit-v2",
            "point_id": "c3=0,d4=0",
            "c3": 0.0,
            "d4": 0.0,
            "file": "/sm-hh4b.root",
            "process_id": "sm_hh4b_heft",
            "xsec_fb": 0.01,
            "rate_factor": 0.2,
            "generated_events": 1000,
            "normalisation_weight": 1000.0,
            "entries": 100,
            "analysis_efficiency": 0.5,
            "xgboost_efficiency": 0.5,
            "final_efficiency": 0.25,
            "effective_feature_xsec_fb": 0.001,
            "effective_selected_xsec_fb": 0.0005,
            "nominal_feature_signal_yield": 3.0,
            "nominal_selected_signal_yield": 1.5,
            "nominal_selected_signal_staterror": 0.1,
            "selected_raw_entries": 25,
            "included_in_training": False,
            "included_in_threshold_optimization": False,
            "included_in_shape_binning_optimization": False,
            "included_in_background": False,
            "included_in_limits": False,
        }

        row = runner._sm_hh4b_signal_cutflow_row(
            result,
            luminosity=3000.0,
        )

        self.assertEqual(row["sample_role"], "signal")
        self.assertEqual(row["signal_component"], "sm_hh4b")
        self.assertIsNone(row["cut_signal_strength95"])
        self.assertAlmostEqual(row["input_xsec_fb"], 0.001)
        self.assertAlmostEqual(row["xgboost_xsec_fb"], 0.0005)
        self.assertEqual(row["selected_entries"], 25)
        self.assertFalse(row["included_in_training"])
        self.assertFalse(row["included_in_background"])
        self.assertFalse(row["included_in_limits"])
        rendered = runner.terminal_sm_background_cutflow_table(
            [row],
            luminosity=3000.0,
            thresholds=[0.5] * 5,
        )
        self.assertIn("SM HEFT gg->hh + b bbar b bbar", rendered)
        self.assertIn("post-training signal diagnostic only", rendered)

    def test_sm_hh4b_fit_rescales_only_cross_section_at_target_point(self):
        fit = fit_hh4b_c3_cross_section(
            [
                {
                    "c3": c3,
                    "cross_section_pb": (
                        1.0e-5 + 2.0e-6 * c3 + 3.0e-7 * c3 * c3
                    ),
                    "integration_error_pb": 1.0e-8,
                }
                for c3 in (-20.0, -2.0, -1.0, 0.0, 20.0)
            ]
        )
        result = {
            "component": "sm_hh4b",
            "point_id": "c3=0,d4=0",
            "c3": 0.0,
            "d4": 0.0,
            "file": "/sm-hh4b.root",
            "process_id": "sm_hh4b_heft",
            "xsec_fb": 0.01,
            "rate_factor": 0.2,
            "generated_events": 1000,
            "normalisation_weight": 1000.0,
            "entries": 100,
            "analysis_efficiency": 0.5,
            "xgboost_efficiency": 0.5,
            "effective_feature_xsec_fb": 0.001,
            "effective_selected_xsec_fb": 0.0005,
            "nominal_feature_signal_yield": 3.0,
            "nominal_selected_signal_yield": 1.5,
            "nominal_selected_signal_staterror": 0.1,
            "selected_raw_entries": 25,
            "c3_cross_section_fit": fit,
            "included_in_training": False,
            "included_in_threshold_optimization": False,
            "included_in_shape_binning_optimization": False,
            "included_in_background": False,
            "included_in_limits": False,
        }

        row = runner._sm_hh4b_signal_cutflow_row(
            result,
            luminosity=3000.0,
            point_id="c3=3,d4=200",
            c3=3.0,
            d4=200.0,
            category="positive diagonal",
            is_limit_representative=True,
        )

        self.assertAlmostEqual(row["production_xsec_fb"], 0.0187)
        self.assertAlmostEqual(row["cross_section_rescale_factor"], 1.87)
        self.assertAlmostEqual(row["input_xsec_fb"], 0.00187)
        self.assertAlmostEqual(row["xgboost_xsec_fb"], 0.000935)
        self.assertAlmostEqual(row["xgboost_efficiency"], 0.5)
        self.assertAlmostEqual(row["rate_factor"], 0.2)
        self.assertEqual(row["point_id"], "c3=3,d4=200")
        self.assertEqual(row["c3"], 3.0)
        self.assertEqual(row["d4"], 200.0)
        self.assertTrue(row["cross_section_fit_applied"])
        self.assertTrue(row["is_limit_representative"])
        self.assertFalse(row["included_in_limits"])

    def test_parameterized_postfit_hhhbb_is_scored_at_its_true_coordinate(self):
        hhhbb = sample(
            "hhhbb",
            "postfit_hhhbb_signal",
            [1] * 5,
            [0.2] * 5,
            -7.5,
            50.0,
        )
        validation = {
            "points": {
                hhhbb.point_id: {
                    "threshold": 0.5,
                }
            }
        }
        with mock.patch.object(
            runner,
            "_predict",
            return_value=np.asarray([0.75]),
        ) as predict:
            result = runner._evaluate_postfit_signal_rotation(
                object(),
                validation,
                [hhhbb],
                rotation=0,
                n_folds=5,
                profile_indices=np.arange(91),
                parameterized=True,
            )

        self.assertTrue(result["parameterized"])
        self.assertEqual(
            result["points"][hhhbb.point_id]["signal_raw_entries"],
            1,
        )
        self.assertEqual(
            predict.call_args.args[-1],
            (-7.5, 50.0),
        )

    def test_postfit_hhhbb_changes_only_the_final_signal_limit(self):
        hhhh = sample("hhhh", "grid_signal", [1] * 5, [1] * 5, 0, 0)
        hhhh.xsec_fb = 2.0
        hhhbb = sample(
            "hhhbb", "postfit_hhhbb_signal", [1] * 5, [0.5] * 5, 0, 0
        )
        hhhbb.xsec_fb = 0.5
        hhhbb.rate_factor = 0.4
        hhhh_rotations = []
        hhhbb_rotations = []
        for fold in range(5):
            hhhh_rotations.append(
                {
                    "points": {
                        hhhh.point_id: {
                            "threshold": 0.4,
                            "signal_unit_yield": 0.2,
                            "signal_sumw2_unit": 0.04,
                            "signal_raw_entries": 1,
                            "signal_feature_unit_yield": 0.2,
                            "background_yield": 0.4,
                            "background_sumw2": 0.16,
                            "background_raw_entries": 1,
                            "background_effective_entries": 1.0,
                            "s95_exact_events": runner.exact_cls_signal_upper_limit(
                                0.4
                            ),
                            "cut_sigma95_fb": (
                                runner.exact_cls_signal_upper_limit(0.4) / 0.2
                            ),
                        }
                    }
                }
            )
            hhhbb_rotations.append(
                {
                    "points": {
                        hhhbb.point_id: {
                            "threshold": 0.4,
                            "signal_unit_yield": 0.2,
                            "signal_sumw2_unit": 0.04,
                            "signal_physical_yield": 0.1,
                            "signal_sumw2_physical": 0.01,
                            "signal_raw_entries": 1,
                            "signal_feature_unit_yield": 0.2,
                            "signal_feature_physical_yield": 0.1,
                            "xgboost_efficiency": 1.0,
                        }
                    }
                }
            )

        aggregate = runner._aggregate_cut_results(
            [hhhh], hhhh_rotations
        )
        original_background = aggregate[0]["background_yield"]
        original_threshold = aggregate[0]["threshold_mean"]
        runner._add_postfit_hhhbb_cut_contribution(
            aggregate,
            [hhhh],
            [hhhbb],
            hhhbb_rotations,
        )
        result = aggregate[0]
        s95 = runner.exact_cls_signal_upper_limit(2.0)

        self.assertEqual(result["signal_components"], "hhhh,hhhbb")
        self.assertEqual(result["limit_parameter"], "common-signal-strength")
        self.assertAlmostEqual(result["hhhh_nominal_selected_signal_yield"], 2.0)
        self.assertAlmostEqual(result["hhhbb_nominal_selected_signal_yield"], 0.5)
        self.assertAlmostEqual(
            result["combined_nominal_selected_signal_yield"], 2.5
        )
        self.assertAlmostEqual(result["selected_signal_yield_per_fb"], 1.25)
        self.assertAlmostEqual(result["cut_sigma95_fb"], s95 / 1.25)
        self.assertAlmostEqual(result["cut_signal_strength95"], s95 / 2.5)
        self.assertAlmostEqual(
            result["cut_sigma95_fb"] / result["hhhh_xsec_fb"],
            result["cut_signal_strength95"],
        )
        self.assertEqual(result["background_yield"], original_background)
        self.assertEqual(result["threshold_mean"], original_threshold)
        self.assertTrue(
            all(
                math.isclose(fold["signal_unit_yield"], 0.25)
                for fold in result["folds"]
            )
        )

    def test_postfit_hhhbb_requires_an_exact_point_match(self):
        hhhh = sample("hhhh", "grid_signal", [1] * 5, [1] * 5, 0, 0)
        hhhbb = sample(
            "hhhbb", "postfit_hhhbb_signal", [1] * 5, [1] * 5, 1, 0
        )

        with self.assertRaisesRegex(ValueError, "exactly match"):
            runner._add_postfit_hhhbb_cut_contribution(
                [],
                [hhhh],
                [hhhbb],
                [],
            )

    def test_sm_background_cutflow_uses_exact_held_out_union_and_fold_thresholds(self):
        background = sample(
            "background",
            "background",
            [1.0] * 5,
            [10.0, 20.0, 30.0, 40.0, 50.0],
        )
        background.metadata = {
            "process_id": "gg_to_6b_2c",
            "description": "gg -> 6b + 2c",
        }
        records = []
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        scores = [0.2, 0.2, 0.6, 0.8, 0.85]
        for fold, (threshold, score) in enumerate(zip(thresholds, scores)):
            records.append(
                {
                    "test": {
                        "points": {
                            "c3=0,d4=0": {
                                "c3": 0.0,
                                "d4": 0.0,
                                "threshold": threshold,
                            }
                        },
                        "background_rows": {
                            background.sample_id: {
                                "scores": np.asarray([score]),
                                "physical_weights": np.asarray(
                                    [background.physical_weights[fold]]
                                ),
                                "event_indices": np.asarray(
                                    [background.event_indices[fold]]
                                ),
                            }
                        },
                    }
                }
            )

        rows, observed_thresholds = runner._sm_background_cutflow_rows(
            [background], records, luminosity=10.0
        )

        self.assertEqual(observed_thresholds, thresholds)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row["input_events"], 150.0)
        self.assertAlmostEqual(row["input_xsec_fb"], 15.0)
        # Folds 0, 2, and 3 pass their independently selected thresholds.
        self.assertAlmostEqual(row["xgboost_events"], 80.0)
        self.assertAlmostEqual(row["xgboost_xsec_fb"], 8.0)
        self.assertEqual(row["selected_entries"], 3)
        self.assertEqual(row["entries"], 5)

    def test_parameterized_sm_background_cutflow_rescores_at_sm_point(self):
        background = sample(
            "background",
            "background",
            [1.0] * 5,
            [1.0] * 5,
        )
        records = [
            {
                "rotation": fold,
                "_model_object": object(),
                "_background_samples": [background],
                "_profile_indices": np.arange(91),
                "_n_folds": 5,
                "test": {
                    "parameterized": True,
                    "points": {
                        "c3=0,d4=0": {
                            "c3": 0.0,
                            "d4": 0.0,
                            "threshold": 0.5,
                        }
                    },
                    "background_rows": {},
                },
            }
            for fold in range(5)
        ]

        def score_background(
            model,
            samples,
            *,
            rotation,
            parameter_point,
            **kwargs,
        ):
            del model, samples, kwargs
            self.assertEqual(parameter_point, (0.0, 0.0))
            return {
                background.sample_id: {
                    "scores": np.asarray([0.75]),
                    "physical_weights": np.asarray([1.0]),
                    "event_indices": np.asarray([rotation]),
                }
            }

        with mock.patch.object(
            runner,
            "_score_partition",
            side_effect=score_background,
        ) as scoring:
            rows, thresholds = runner._sm_background_cutflow_rows(
                [background],
                records,
                luminosity=10.0,
            )

        self.assertEqual(scoring.call_count, 5)
        self.assertEqual(thresholds, [0.5] * 5)
        self.assertEqual(rows[0]["selected_entries"], 5)
        self.assertEqual(rows[0]["xgboost_events"], 5.0)

    def test_sm_background_cutflow_rejects_overlapping_test_folds(self):
        background = sample("background", "background", [1.0] * 5, [1.0] * 5)
        records = []
        for _fold in range(5):
            records.append(
                {
                    "test": {
                        "points": {
                            "c3=0,d4=0": {
                                "c3": 0.0,
                                "d4": 0.0,
                                "threshold": 0.5,
                            }
                        },
                        "background_rows": {
                            background.sample_id: {
                                "scores": np.asarray([1.0]),
                                "physical_weights": np.asarray([1.0]),
                                "event_indices": np.asarray([0]),
                            }
                        },
                    }
                }
            )

        with self.assertRaisesRegex(ValueError, "exact, non-overlapping union"):
            runner._sm_background_cutflow_rows(
                [background], records, luminosity=3000.0
            )

    def test_sm_signal_cutflow_reports_all_components_as_separate_signals(self):
        hhhh = sample(
            "run_gg_4h_test_0_0",
            "grid_signal",
            [1.0] * 5,
            [2.0] * 5,
            0,
            0,
        )
        hhhh.xsec_fb = 2.0
        hhhh.rate_factor = 0.5
        hhhbb = sample(
            "run_gg_hhhg_test_0_0",
            "postfit_hhhbb_signal",
            [1.0] * 5,
            [0.2] * 5,
            0,
            0,
        )
        hhhbb.xsec_fb = 0.4
        hhhbb.rate_factor = 0.25
        aggregate = [
            {
                "point_id": hhhh.point_id,
                "c3": 0.0,
                "d4": 0.0,
                "cut_signal_strength95": 10.0,
                "excluded_cut": False,
                "hhhh_selected_signal_yield_per_fb": 1.5,
                "hhhh_selected_signal_staterror_per_fb": 0.4,
                "hhhbb_nominal_selected_signal_yield": 0.6,
                "hhhbb_nominal_selected_signal_staterror": 0.2,
                "hhhbb_selected_raw_entries": 2,
                "folds": [
                    {"signal_raw_entries": selected}
                    for selected in (1, 0, 1, 0, 1)
                ],
            }
        ]
        sm_hh4b_result = {
            "component": "sm_hh4b",
            "point_id": "c3=0,d4=0",
            "c3": 0.0,
            "d4": 0.0,
            "file": "/sm-hh4b.root",
            "process_id": "sm_hh4b_heft",
            "xsec_fb": 0.01,
            "rate_factor": 0.2,
            "generated_events": 1000,
            "normalisation_weight": 1000.0,
            "entries": 100,
            "analysis_efficiency": 0.5,
            "xgboost_efficiency": 0.5,
            "effective_feature_xsec_fb": 0.001,
            "effective_selected_xsec_fb": 0.0005,
            "nominal_feature_signal_yield": 0.01,
            "nominal_selected_signal_yield": 0.005,
            "nominal_selected_signal_staterror": 0.001,
            "selected_raw_entries": 25,
            "included_in_training": False,
            "included_in_threshold_optimization": False,
            "included_in_shape_binning_optimization": False,
            "included_in_background": False,
            "included_in_limits": False,
        }

        rows = runner._sm_signal_cutflow_rows(
            [hhhh],
            [hhhbb],
            aggregate,
            luminosity=10.0,
            sm_hh4b_result=sm_hh4b_result,
        )

        self.assertEqual(
            [row["signal_component"] for row in rows],
            ["hhhh", "hhhbb", "sm_hh4b"],
        )
        self.assertTrue(all(row["sample_role"] == "signal" for row in rows))
        self.assertTrue(all(row["is_signal"] for row in rows))
        self.assertAlmostEqual(rows[0]["effective_inclusive_xsec_fb"], 1.0)
        self.assertAlmostEqual(rows[0]["input_events"], 10.0)
        self.assertAlmostEqual(rows[0]["xgboost_events"], 3.0)
        self.assertAlmostEqual(rows[0]["xgboost_events_error"], 0.8)
        self.assertEqual(rows[0]["selected_entries"], 3)
        self.assertAlmostEqual(rows[1]["effective_inclusive_xsec_fb"], 0.1)
        self.assertAlmostEqual(rows[1]["input_events"], 1.0)
        self.assertAlmostEqual(rows[1]["xgboost_events"], 0.6)
        self.assertAlmostEqual(rows[1]["xgboost_events_error"], 0.2)
        self.assertEqual(rows[1]["selected_entries"], 2)
        self.assertAlmostEqual(rows[2]["input_events"], 0.01)
        self.assertAlmostEqual(rows[2]["xgboost_events"], 0.005)
        self.assertIsNone(rows[2]["cut_signal_strength95"])
        self.assertFalse(rows[2]["included_in_background"])
        self.assertFalse(rows[2]["included_in_limits"])

        background = {
            "sample_id": "background",
            "sample_role": "background",
            "is_signal": False,
            "description": "QCD background",
            "input_xsec_fb": 4.0,
            "input_events": 40.0,
            "xgboost_xsec_fb": 0.2,
            "xgboost_events": 2.0,
            "xgboost_events_error": 0.5,
            "entries": 20,
            "selected_entries": 2,
        }
        rendered = runner.terminal_sm_background_cutflow_table(
            [background, *rows],
            luminosity=10.0,
            thresholds=[0.5] * 5,
        )
        self.assertIn("signal references", rendered)
        self.assertLess(rendered.index("SM gg->hhhh->8b"), rendered.index("QCD background"))
        self.assertIn("SM HEFT gg->hh + b bbar b bbar", rendered)
        self.assertIn("post-training signal diagnostic only", rendered)
        self.assertIn("do not enter the background total", rendered)

    def test_hh4b_fit_rows_follow_the_same_representative_points_as_hhhbb(self):
        fit = fit_hh4b_c3_cross_section(
            [
                {
                    "c3": c3,
                    "cross_section_pb": 1.0e-5 + 1.0e-7 * c3 * c3,
                    "integration_error_pb": 1.0e-8,
                }
                for c3 in (-20.0, -2.0, -1.0, 0.0, 20.0)
            ]
        )
        hhhh_samples = [
            sample(
                f"hhhh_{c3:g}_{d4:g}",
                "grid_signal",
                [1.0] * 5,
                [0.2] * 5,
                c3,
                d4,
            )
            for c3, d4 in ((0.0, 0.0), (3.0, 200.0))
        ]
        hhhbb_samples = [
            sample(
                f"hhhbb_{c3:g}_{d4:g}",
                "postfit_hhhbb_signal",
                [1.0] * 5,
                [0.02] * 5,
                c3,
                d4,
            )
            for c3, d4 in ((0.0, 0.0), (3.0, 200.0))
        ]
        aggregate = []
        for hhhh in hhhh_samples:
            aggregate.append(
                {
                    "point_id": hhhh.point_id,
                    "c3": hhhh.c3,
                    "d4": hhhh.d4,
                    "cut_signal_strength95": 1.0,
                    "excluded_cut": False,
                    "hhhh_selected_signal_yield_per_fb": 0.5,
                    "hhhh_selected_signal_staterror_per_fb": 0.1,
                    "hhhbb_nominal_selected_signal_yield": 0.2,
                    "hhhbb_nominal_selected_signal_staterror": 0.04,
                    "hhhbb_selected_raw_entries": 2,
                    "folds": [{"signal_raw_entries": 1}] * 5,
                }
            )
        sm_hh4b_result = {
            "component": "sm_hh4b",
            "point_id": "c3=0,d4=0",
            "c3": 0.0,
            "d4": 0.0,
            "file": "/sm-hh4b.root",
            "process_id": "sm_hh4b_heft",
            "xsec_fb": 0.01,
            "rate_factor": 0.2,
            "generated_events": 1000,
            "normalisation_weight": 1000.0,
            "entries": 100,
            "analysis_efficiency": 0.5,
            "xgboost_efficiency": 0.5,
            "effective_feature_xsec_fb": 0.001,
            "effective_selected_xsec_fb": 0.0005,
            "nominal_feature_signal_yield": 3.0,
            "nominal_selected_signal_yield": 1.5,
            "nominal_selected_signal_staterror": 0.1,
            "selected_raw_entries": 25,
            "c3_cross_section_fit": fit,
            "included_in_training": False,
            "included_in_threshold_optimization": False,
            "included_in_shape_binning_optimization": False,
            "included_in_background": False,
            "included_in_limits": False,
        }
        representative = {
            "representative_category": "positive diagonal",
            "result": aggregate[1],
            "cut_signal_strength95": 1.0,
            "limit_proximity_log_mu95": 0.0,
        }

        with mock.patch.object(
            runner,
            "_select_limit_representative_points",
            return_value=[representative],
        ):
            rows = runner._sm_signal_cutflow_rows(
                hhhh_samples,
                hhhbb_samples,
                aggregate,
                luminosity=3000.0,
                include_limit_representatives=True,
                sm_hh4b_result=sm_hh4b_result,
            )

        coordinates_by_component = {}
        for row in rows:
            coordinates_by_component.setdefault(
                row["signal_component"], []
            ).append((row["c3"], row["d4"]))
        expected = [(0.0, 0.0), (3.0, 200.0)]
        self.assertEqual(coordinates_by_component["hhhh"], expected)
        self.assertEqual(coordinates_by_component["hhhbb"], expected)
        self.assertEqual(coordinates_by_component["sm_hh4b"], expected)
        fitted_hh4b = [
            row for row in rows if row["signal_component"] == "sm_hh4b"
        ]
        self.assertTrue(all(row["cut_signal_strength95"] is None for row in fitted_hh4b))
        self.assertTrue(all(not row["included_in_limits"] for row in fitted_hh4b))

    def test_pyhf_shape_cutflow_rows_replace_only_likelihood_signal_mu95(self):
        cutflow = [
            {
                "sample_id": "hhhh",
                "point_id": "c3=0,d4=0",
                "is_signal": True,
                "signal_component": "hhhh",
                "cut_signal_strength95": 8.0,
            },
            {
                "sample_id": "hhhbb",
                "point_id": "c3=0,d4=0",
                "is_signal": True,
                "signal_component": "hhhbb",
                "cut_signal_strength95": 8.0,
            },
            {
                "sample_id": "hh4b",
                "point_id": "c3=0,d4=0",
                "is_signal": True,
                "signal_component": "sm_hh4b",
                "included_in_limits": False,
                "cut_signal_strength95": None,
            },
            {
                "sample_id": "background",
                "is_signal": False,
            },
        ]
        rows = runner._pyhf_shape_cutflow_rows(
            cutflow,
            [
                {
                    "point_id": "c3=0,d4=0",
                    "xsec_fb": 0.25,
                    "shape_sigma95_fb": 1.0,
                }
            ],
        )

        self.assertEqual(rows[0]["shape_signal_strength95"], 4.0)
        self.assertEqual(rows[1]["shape_signal_strength95"], 4.0)
        self.assertIsNone(rows[2]["shape_signal_strength95"])
        self.assertIsNone(rows[3]["shape_signal_strength95"])
        self.assertEqual(rows[0]["cut_signal_strength95"], 8.0)

    def test_fast_sm_prints_final_pyhf_table_after_shape_maps(self):
        with tempfile.TemporaryDirectory() as directory, mocked_mode_study_pipeline(
            point_count=4
        ):
            terminal = io.StringIO()
            with redirect_stdout(terminal):
                runner.run_c3d4_study(
                    sm_signal_specs=[{}],
                    grid_signal_specs=[{}] * 4,
                    background_specs=[{}],
                    output_dir=directory,
                    study_mode="fast-sm",
                )

        rendered = terminal.getvalue()
        self.assertIn(
            "Classifier strategy: sm-crossfit-v2 (full pyhf score-shape fit)",
            rendered,
        )
        self.assertIn("SM pyhf score-shape expected limits", rendered)
        self.assertIn("mu95 (pyhf)", rendered)
        self.assertLess(
            rendered.index("[v2 maps] Completed strategy maps"),
            rendered.index("[v2 shape-publish] Publishing final pyhf"),
        )

    def test_limit_representatives_cover_axes_and_four_diagonal_quadrants(self):
        coordinates = [
            (0.0, -200.0, 1.05),
            (0.0, -250.0, 0.50),
            (0.0, 250.0, 0.82),
            (-9.0, 0.0, 1.09),
            (-12.0, 0.0, 0.20),
            (12.0, 0.0, 0.63),
            (7.5, 50.0, 1.14),
            (-1.5, 150.0, 1.52),
            (-6.0, -100.0, 1.01),
            (1.5, -150.0, 1.35),
        ]
        aggregate = [
            {
                "point_id": f"c3={c3:.12g},d4={d4:.12g}",
                "c3": c3,
                "d4": d4,
                "cut_signal_strength95": mu95,
            }
            for c3, d4, mu95 in coordinates
        ]

        selected = runner._select_limit_representative_points(aggregate)

        self.assertEqual(
            [
                (
                    row["representative_category"],
                    row["result"]["c3"],
                    row["result"]["d4"],
                )
                for row in selected
            ],
            [
                ("c3~0, d4<0", 0.0, -200.0),
                ("c3~0, d4>0", 0.0, 250.0),
                ("d4~0, c3<0", -9.0, 0.0),
                ("d4~0, c3>0", 12.0, 0.0),
                ("diagonal Q1", 7.5, 50.0),
                ("diagonal Q2", -1.5, 150.0),
                ("diagonal Q3", -6.0, -100.0),
                ("diagonal Q4", 1.5, -150.0),
            ],
        )

    def test_representative_signal_cutflow_has_two_components_per_point(self):
        coordinates = [
            (0.0, 0.0, 1000.0),
            (0.0, -200.0, 1.05),
            (0.0, 250.0, 0.82),
            (-9.0, 0.0, 1.09),
            (12.0, 0.0, 0.63),
            (7.5, 50.0, 1.14),
            (-1.5, 150.0, 1.52),
            (-6.0, -100.0, 1.01),
            (1.5, -150.0, 1.35),
        ]
        grid_samples = []
        hhhbb_samples = []
        aggregate = []
        for c3, d4, mu95 in coordinates:
            hhhh = sample(
                f"hhhh_{c3:g}_{d4:g}",
                "grid_signal",
                [1.0] * 5,
                [2.0] * 5,
                c3,
                d4,
            )
            hhhbb = sample(
                f"hhhbb_{c3:g}_{d4:g}",
                "postfit_hhhbb_signal",
                [1.0] * 5,
                [0.2] * 5,
                c3,
                d4,
            )
            grid_samples.append(hhhh)
            hhhbb_samples.append(hhhbb)
            aggregate.append(
                {
                    "point_id": hhhh.point_id,
                    "c3": c3,
                    "d4": d4,
                    "cut_signal_strength95": mu95,
                    "excluded_cut": mu95 <= 1.0,
                    "hhhh_selected_signal_yield_per_fb": 1.5,
                    "hhhh_selected_signal_staterror_per_fb": 0.4,
                    "hhhbb_nominal_selected_signal_yield": 0.6,
                    "hhhbb_nominal_selected_signal_staterror": 0.2,
                    "hhhbb_selected_raw_entries": 2,
                    "folds": [
                        {"signal_raw_entries": selected}
                        for selected in (1, 0, 1, 0, 1)
                    ],
                }
            )

        rows = runner._sm_signal_cutflow_rows(
            grid_samples,
            hhhbb_samples,
            aggregate,
            luminosity=10.0,
            include_limit_representatives=True,
        )
        point_totals = runner._cutflow_signal_totals_by_point(rows)

        self.assertEqual(len(rows), 18)
        self.assertEqual(
            sum(bool(row["is_limit_representative"]) for row in rows),
            16,
        )
        self.assertEqual(len(point_totals), 9)
        self.assertTrue(
            all(
                total["signal_components"] == ["hhhh", "hhhbb"]
                for total in point_totals
            )
        )
        self.assertEqual(
            [
                (total["c3"], total["d4"])
                for total in point_totals
                if total["is_limit_representative"]
            ],
            [
                (0.0, -200.0),
                (0.0, 250.0),
                (-9.0, 0.0),
                (12.0, 0.0),
                (7.5, 50.0),
                (-1.5, 150.0),
                (-6.0, -100.0),
                (1.5, -150.0),
            ],
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

    def test_postfit_hhhbb_enters_only_the_frozen_test_shape(self):
        point = sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0)
        point.xsec_fb = 2.0
        records = populated_shape_records([point])
        hhhbb_scores = np.linspace(0.01, 0.99, 80, dtype=float)
        hhhbb_physical_weights = np.full(80, 0.01, dtype=float)
        for record in records:
            record["postfit_hhhbb_test"] = {
                "rotation": record["rotation"],
                "points": {
                    point.point_id: {
                        "sample_id": "hhhbb-p0",
                        "c3": point.c3,
                        "d4": point.d4,
                    }
                },
                "signal_rows": {
                    "hhhbb-p0": {
                        "scores": hhhbb_scores.copy(),
                        "physical_weights": hhhbb_physical_weights.copy(),
                        "scale": 1.0,
                    }
                },
                "role": "postfit-signal-only",
            }

        validation = runner._validation_fold_arrays(records[0], point)
        test = runner._test_fold_arrays(records[0], point)
        self.assertEqual(len(validation["signal_scores"]), 80)
        self.assertEqual(len(test["signal_scores"]), 160)
        # The hhhbb physical weights are converted to equivalent hhhh-fb
        # weights by dividing by the point's hhhh theory cross section.
        np.testing.assert_allclose(test["signal_weights"][-80:], 0.005)
        self.assertAlmostEqual(float(np.sum(test["signal_weights"])), 2.4)

        compact = runner._compact_shape_records(
            records,
            observable_set="extended-91-v2",
            profile="full91",
            n_folds=5,
        )
        descriptor = runner._shape_point_descriptors([point])[0]
        compact_test = runner._test_fold_arrays(compact[0], descriptor)
        np.testing.assert_allclose(
            compact_test["signal_weights"], test["signal_weights"]
        )
        self.assertEqual(descriptor.xsec_fb, 2.0)

        with mock.patch.object(
            runner, "pyhf_one_bin_limit", new=successful_pyhf_limit
        ), mock.patch.object(
            runner, "pyhf_combined_limit", new=successful_pyhf_limit
        ):
            result = runner._shape_results([point], records, shape_jobs=1)[0]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["signal_components"], "hhhh,hhhbb")
        self.assertEqual(
            result["postfit_hhhbb_role"],
            "held-out-test-template-after-frozen-hhhh-validation-binning",
        )
        self.assertFalse(result["postfit_hhhbb_in_training"])
        self.assertFalse(result["postfit_hhhbb_in_threshold_optimization"])
        self.assertFalse(result["postfit_hhhbb_in_shape_binning_optimization"])
        np.testing.assert_allclose(result["one_bin_signal_sumw2"], 0.052)

    def test_pyhf_poi_bounds_follow_the_expected_limit_scale(self):
        channels = [
            {
                "signal": np.asarray([1.0, 3.0]),
                "background": np.asarray([2.0, 8.0]),
            }
        ]
        with mock.patch.object(
            runner, "exact_cls_signal_upper_limit", return_value=2.0
        ):
            bounds = runner._poi_bounds_for_channels(channels)

        self.assertEqual(bounds[0], 0.0)
        self.assertEqual(bounds[1], 5.0)
        self.assertLess(bounds[1], 100.0)
        self.assertEqual(
            runner._poi_bounds_from_estimate(0.25),
            (0.0, 2.5),
        )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            runner._poi_bounds_from_estimate(0.0)

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

    def test_invalid_validation_signal_is_terminal_and_checkpoint_reusable(self):
        point = sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0)
        records = populated_shape_records(
            [point], invalid_validation_signal=True
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "pyhf_one_bin_limit", new=successful_pyhf_limit
        ), mock.patch.object(
            runner,
            "pyhf_combined_limit",
            return_value={"status": "ok", "expected_median": 4.25},
        ) as combined_limit:
            checkpoint_dir = Path(directory)
            rows, metadata = runner._shape_results(
                [point],
                records,
                shape_jobs=1,
                checkpoint_dir=checkpoint_dir,
                checkpoint_fingerprint="invalid-signal-fingerprint",
                return_metadata=True,
            )

            self.assertEqual(rows[0]["status"], "invalid_signal")
            self.assertFalse(runner._shape_row_is_retryable(rows[0]))
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["retryable_points"], [])
            # Invalid signed templates are rejected before any shape fit.
            combined_limit.assert_not_called()

            descriptor = runner._shape_point_descriptors([point])[0]
            cached, checkpoint_status = runner._read_shape_checkpoint(
                runner._shape_checkpoint_path(checkpoint_dir, descriptor),
                fingerprint="invalid-signal-fingerprint",
                point=descriptor,
            )
            self.assertEqual(checkpoint_status, "reused")
            self.assertEqual(cached["status"], "invalid_signal")

    def test_nonpositive_one_bin_signal_is_terminal_not_retryable(self):
        point = sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0)
        records = populated_shape_records([point])
        for record in records:
            record["test"]["signal_rows"][point.sample_id][
                "unit_xsec_weights"
            ] = np.full(80, -0.025, dtype=float)
        with mock.patch.object(
            runner,
            "_select_shape_candidate",
            return_value={"status": "failed", "error": "not evaluated"},
        ), mock.patch.object(runner, "pyhf_one_bin_limit") as one_bin_limit:
            rows, metadata = runner._shape_results(
                [point], records, shape_jobs=1, return_metadata=True
            )

        one_bin_limit.assert_not_called()
        self.assertEqual(rows[0]["status"], "invalid_signal")
        self.assertEqual(rows[0]["terminal_reason"], "invalid_one_bin_signal")
        self.assertFalse(runner._shape_row_is_retryable(rows[0]))
        self.assertEqual(metadata["status"], "complete")

    def test_validation_pyhf_failure_is_retryable_and_checkpoint_incomplete(self):
        point = sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0)
        records = populated_shape_records([point])
        failed_fit = {
            "status": "optimizer_failed",
            "expected_median": None,
            "error": "deliberate numerical failure",
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "pyhf_one_bin_limit", new=successful_pyhf_limit
        ), mock.patch.object(
            runner, "pyhf_combined_limit", return_value=failed_fit
        ) as combined_limit:
            checkpoint_dir = Path(directory)
            rows, metadata = runner._shape_results(
                [point],
                records,
                shape_jobs=1,
                checkpoint_dir=checkpoint_dir,
                checkpoint_fingerprint="pyhf-failure-fingerprint",
                return_metadata=True,
            )

            self.assertGreater(combined_limit.call_count, 0)
            self.assertEqual(rows[0]["status"], "pyhf_failed")
            self.assertTrue(runner._shape_row_is_retryable(rows[0]))
            self.assertEqual(metadata["status"], "incomplete")
            self.assertEqual(metadata["retryable_points"], [point.point_id])

            descriptor = runner._shape_point_descriptors([point])[0]
            cached, checkpoint_status = runner._read_shape_checkpoint(
                runner._shape_checkpoint_path(checkpoint_dir, descriptor),
                fingerprint="pyhf-failure-fingerprint",
                point=descriptor,
            )
            self.assertIsNone(cached)
            self.assertEqual(checkpoint_status, "incomplete")

    @unittest.skipUnless(
        os.name == "posix" and "fork" in runner.multiprocessing.get_all_start_methods(),
        "parallel shape evaluation requires POSIX fork",
    )
    def test_serial_and_parallel_shape_rows_are_identical_and_sorted(self):
        points = [
            sample("p1", "grid_signal", [1] * 5, [1] * 5, 1, 0),
            sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0),
        ]
        records = empty_shape_records()
        with mock.patch.object(
            runner, "_candidate_maps_for_validation", return_value=([], [])
        ), mock.patch.object(
            runner, "_evaluate_shape_point_payload", side_effect=fake_shape_payload
        ):
            serial = runner._shape_results(points, records, shape_jobs=1)
            try:
                parallel = runner._shape_results(points, records, shape_jobs=2)
            except RuntimeError as error:
                if "Unable to start forked pyhf workers" in str(error):
                    self.skipTest(str(error))
                raise
        self.assertEqual(serial, parallel)
        self.assertEqual([row["point_id"] for row in parallel], [points[1].point_id, points[0].point_id])

    @unittest.skipUnless(
        os.name == "posix" and "fork" in runner.multiprocessing.get_all_start_methods(),
        "parallel shape evaluation requires POSIX fork",
    )
    def test_real_shape_evaluator_serial_and_parallel_rows_are_identical(self):
        points = [
            sample("p1", "grid_signal", [1] * 5, [1] * 5, 1, 0),
            sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0),
        ]
        records = populated_shape_records(points)
        # Patch only the expensive statistical backend. Candidate construction,
        # compact process records, test-bin fallback, and row assembly are real.
        with mock.patch.object(
            runner, "pyhf_one_bin_limit", new=successful_pyhf_limit
        ), mock.patch.object(
            runner, "pyhf_combined_limit", new=successful_pyhf_limit
        ):
            serial = runner._shape_results(points, records, shape_jobs=1)
            try:
                parallel = runner._shape_results(points, records, shape_jobs=2)
            except RuntimeError as error:
                if "Unable to start forked pyhf workers" in str(error):
                    self.skipTest(str(error))
                raise

        self.assertEqual(serial, parallel)
        self.assertEqual(
            [row["point_id"] for row in parallel],
            [points[1].point_id, points[0].point_id],
        )

    def test_shape_checkpoint_roundtrip_rejects_mismatch_and_corruption(self):
        point = runner.ShapePoint("p0", "c3=0,d4=0", 0.0, 0.0)
        payload = {
            "kind": "result",
            "row": {
                "point_id": point.point_id,
                "c3": point.c3,
                "d4": point.d4,
                "status": "ok",
            },
            "warnings": [],
            "elapsed_seconds": 1.25,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point.json"
            runner._write_shape_checkpoint(
                path,
                fingerprint="fingerprint-a",
                point=point,
                payload=payload,
                complete=True,
            )
            row, status = runner._read_shape_checkpoint(
                path, fingerprint="fingerprint-a", point=point
            )
            self.assertEqual(status, "reused")
            self.assertEqual(row["status"], "ok")

            row, status = runner._read_shape_checkpoint(
                path, fingerprint="fingerprint-b", point=point
            )
            self.assertIsNone(row)
            self.assertEqual(status, "incompatible")

            path.write_text("{not valid JSON")
            row, status = runner._read_shape_checkpoint(
                path, fingerprint="fingerprint-a", point=point
            )
            self.assertIsNone(row)
            self.assertEqual(status, "malformed")

    def test_shape_resume_reuses_success_and_retries_pyhf_failure_only(self):
        points = [
            sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0),
            sample("p1", "grid_signal", [1] * 5, [1] * 5, 1, 0),
        ]
        records = empty_shape_records()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "_candidate_maps_for_validation", return_value=([], [])
        ), mock.patch.object(
            runner, "_evaluate_shape_point_payload", side_effect=fake_shape_payload
        ):
            checkpoint_dir = Path(directory)
            _, first = runner._shape_results(
                points,
                records,
                shape_jobs=1,
                checkpoint_dir=checkpoint_dir,
                checkpoint_fingerprint="fingerprint",
                return_metadata=True,
            )
            self.assertEqual(first["resumed_points"], 0)

            failed_point = runner._shape_point_descriptors(points)[1]
            failed_payload = {
                "kind": "result",
                "row": {
                    "point_id": failed_point.point_id,
                    "c3": failed_point.c3,
                    "d4": failed_point.d4,
                    "status": "pyhf_failed",
                },
                "warnings": [],
                "elapsed_seconds": 0.5,
            }
            runner._write_shape_checkpoint(
                runner._shape_checkpoint_path(checkpoint_dir, failed_point),
                fingerprint="fingerprint",
                point=failed_point,
                payload=failed_payload,
                complete=False,
            )
            evaluator = mock.Mock(side_effect=fake_shape_payload)
            with mock.patch.object(runner, "_evaluate_shape_point_payload", evaluator):
                rows, resumed = runner._shape_results(
                    points,
                    records,
                    shape_jobs=1,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_fingerprint="fingerprint",
                    return_metadata=True,
                )
            self.assertEqual(evaluator.call_count, 1)
            self.assertEqual(resumed["resumed_points"], 1)
            self.assertEqual(resumed["submitted_points"], 1)
            self.assertEqual([row["status"] for row in rows], ["ok", "ok"])

    def test_worker_error_is_checkpointed_as_retryable_and_marks_stage_incomplete(self):
        point = sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0)

        def failed_payload(sample_point, records, *, shared_candidates):
            del sample_point, records, shared_candidates
            return {
                "kind": "worker_error",
                "error_type": "RuntimeError",
                "error": "deliberate test failure",
                "traceback": "traceback",
                "warnings": [],
                "elapsed_seconds": 0.1,
            }

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "_candidate_maps_for_validation", return_value=([], [])
        ), mock.patch.object(
            runner, "_evaluate_shape_point_payload", side_effect=failed_payload
        ):
            checkpoint_dir = Path(directory)
            rows, metadata = runner._shape_results(
                [point],
                empty_shape_records(),
                shape_jobs=1,
                checkpoint_dir=checkpoint_dir,
                checkpoint_fingerprint="fingerprint",
                return_metadata=True,
            )
            self.assertEqual(rows[0]["status"], "worker_error")
            self.assertEqual(metadata["status"], "incomplete")
            self.assertEqual(metadata["retryable_points"], [point.point_id])
            descriptor = runner._shape_point_descriptors([point])[0]
            cached, status = runner._read_shape_checkpoint(
                runner._shape_checkpoint_path(checkpoint_dir, descriptor),
                fingerprint="fingerprint",
                point=descriptor,
            )
            self.assertIsNone(cached)
            self.assertEqual(status, "incomplete")

    def test_shape_fingerprint_binds_model_and_pyhf_version(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "fold.json"
            model.write_text("model-a")
            records = [
                {
                    "rotation": 0,
                    "model": str(model),
                    "parameters": {"max_depth": 3},
                    "validation": {"parameterized": False},
                }
            ]
            common = {
                "strategy": "sm-crossfit-v2",
                "profile": "corrected28",
                "observable_set": "extended-91-v2",
                "n_folds": 5,
                "seed": 12345,
                "source_commit": "abc",
                "fold_digest": "folds",
                "normalization_inputs": {"luminosity": 3000.0},
                "input_hashes": {"sample.root": "input"},
                "records": records,
            }
            first = runner._shape_fingerprint(
                **common, package_versions={"pyhf": "0.7.6"}
            )
            model.write_text("model-b")
            changed_model = runner._shape_fingerprint(
                **common, package_versions={"pyhf": "0.7.6"}
            )
            changed_pyhf = runner._shape_fingerprint(
                **common, package_versions={"pyhf": "0.7.7"}
            )
            changed_mode = runner._shape_fingerprint(
                **common,
                package_versions={"pyhf": "0.7.7"},
                study_mode="preview",
            )
            self.assertNotEqual(first, changed_model)
            self.assertNotEqual(changed_model, changed_pyhf)
            self.assertNotEqual(changed_pyhf, changed_mode)

    def test_terminate_shape_executor_uses_pre_python314_process_fallback(self):
        class FakeProcess:
            def __init__(self):
                self.alive = True
                self.terminated = False
                self.killed = False
                self.join_timeouts = []

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.terminated = True
                # Deliberately remain alive to exercise the kill fallback.

            def join(self, timeout=None):
                self.join_timeouts.append(timeout)

            def kill(self):
                self.killed = True
                self.alive = False

        class FakeExecutor:
            def __init__(self, process):
                self._processes = {123: process}
                self.shutdown_arguments = None

            def shutdown(self, *, wait, cancel_futures):
                self.shutdown_arguments = {
                    "wait": wait,
                    "cancel_futures": cancel_futures,
                }

        process = FakeProcess()
        executor = FakeExecutor(process)
        runner._terminate_shape_executor(executor)

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(process.join_timeouts, [5.0, 1.0])
        self.assertEqual(
            executor.shutdown_arguments,
            {"wait": False, "cancel_futures": True},
        )

    def test_shape_executor_shutdown_supports_python38_signature(self):
        class LegacyExecutor:
            def __init__(self):
                self.wait = None

            def shutdown(self, *, wait):
                self.wait = wait

        executor = LegacyExecutor()
        runner._shutdown_shape_executor(
            executor,
            wait_for_workers=True,
            cancel_futures=False,
        )
        self.assertTrue(executor.wait)

    def test_quarantine_strategy_outputs_moves_only_canonical_products(self):
        with tempfile.TemporaryDirectory() as directory:
            strategy_dir = Path(directory) / "pooled-crossfit-v2"
            strategy_dir.mkdir(parents=True)
            expected_files = {
                "per_fold_validation.csv": "validation",
                "per_fold_test.csv": "test",
                "cut_results.csv": "cut-csv",
                "cut_results.json": "cut-json",
                "cut_results_status.json": "cut-status",
                "sm_background_cutflow.csv": "background-cutflow-csv",
                "sm_background_cutflow.json": "background-cutflow-json",
                "sm_background_only_cutflow.csv": "background-only-cutflow-csv",
                "sm_signal_cutflow.csv": "signal-cutflow-csv",
                "shape_results.csv": "shape-csv",
                "shape_results.json": "shape-json",
                "shape_results_status.json": "shape-status",
                "shape_results.partial.csv": "partial-shape-csv",
                "shape_results.partial.json": "partial-shape-json",
            }
            for relative, contents in expected_files.items():
                (strategy_dir / relative).write_text(contents)
            maps = strategy_dir / "maps"
            maps.mkdir()
            (maps / "limit.pdf").write_text("map")
            preserved = strategy_dir / "models"
            preserved.mkdir()
            (preserved / "fold0.json").write_text("model")

            archive = runner._quarantine_strategy_outputs(
                strategy_dir, "abcdef0123456789"
            )

            self.assertIsNotNone(archive)
            for relative, contents in expected_files.items():
                self.assertFalse((strategy_dir / relative).exists())
                self.assertEqual((archive / relative).read_text(), contents)
            self.assertFalse(maps.exists())
            self.assertEqual((archive / "maps" / "limit.pdf").read_text(), "map")
            self.assertEqual((preserved / "fold0.json").read_text(), "model")
            self.assertIsNone(
                runner._quarantine_strategy_outputs(
                    strategy_dir, "abcdef0123456789"
                )
            )

    def test_quick_mode_quarantine_includes_stale_cut_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            strategy_dir = Path(directory) / "sm-crossfit-v2"
            preview = strategy_dir / "cut_preview"
            preview.mkdir(parents=True)
            (preview / "status.json").write_text('{"status":"complete"}')

            archive = runner._quarantine_strategy_outputs(
                strategy_dir,
                "preview-pretraining",
                include_cut_preview=True,
            )

            self.assertFalse(preview.exists())
            self.assertEqual(
                (archive / "cut_preview" / "status.json").read_text(),
                '{"status":"complete"}',
            )

    def test_study_progress_is_flushed_to_terminal_and_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = runner.StudyProgress(Path(directory), interval_seconds=30.0)
            terminal = io.StringIO()
            with redirect_stdout(terminal):
                progress.emit(
                    "shape:sm-crossfit-v2",
                    "Completed shape point",
                    completed=1,
                    total=57,
                    point_id="c3=0,d4=0",
                )
            payload = json.loads((Path(directory) / "study_progress.json").read_text())
            self.assertEqual(payload["phase"], "shape:sm-crossfit-v2")
            self.assertEqual(payload["current"]["completed"], 1)
            self.assertIn("Completed shape point", terminal.getvalue())
            self.assertFalse(any(Path(directory).glob(".study_progress.json.tmp-*")))

    def test_serial_shape_evaluation_emits_heartbeat_during_slow_point(self):
        point = sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0)

        def slow_payload(sample_point, records, *, shared_candidates):
            time.sleep(0.04)
            return fake_shape_payload(
                sample_point, records, shared_candidates=shared_candidates
            )

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "_candidate_maps_for_validation", return_value=([], [])
        ), mock.patch.object(
            runner, "_evaluate_shape_point_payload", side_effect=slow_payload
        ):
            progress = runner.StudyProgress(Path(directory), interval_seconds=0.01)
            terminal = io.StringIO()
            with redirect_stdout(terminal):
                runner._shape_results(
                    [point], empty_shape_records(), shape_jobs=1, progress=progress
                )
            self.assertIn("Waiting for serial pyhf point", terminal.getvalue())

    @unittest.skipUnless(
        importlib.util.find_spec("pyhf")
        and os.name == "posix"
        and "fork" in runner.multiprocessing.get_all_start_methods(),
        "pyhf and POSIX fork are required",
    )
    def test_two_worker_pyhf_smoke(self):
        points = [
            sample("p0", "grid_signal", [1] * 5, [1] * 5, 0, 0),
            sample("p1", "grid_signal", [1] * 5, [1] * 5, 1, 0),
        ]

        def pyhf_payload(point, records, *, shared_candidates):
            del records, shared_candidates
            fit = runner.pyhf_one_bin_limit([1.0], [2.0], include_staterror=False)
            return {
                "kind": "result",
                "row": {
                    "point_id": point.point_id,
                    "c3": point.c3,
                    "d4": point.d4,
                    "status": fit["status"],
                    "shape_sigma95_fb": fit["expected_median"],
                },
                "warnings": [],
                "elapsed_seconds": 0.0,
            }

        with mock.patch.object(
            runner, "_candidate_maps_for_validation", return_value=([], [])
        ), mock.patch.object(
            runner, "_evaluate_shape_point_payload", side_effect=pyhf_payload
        ):
            try:
                rows = runner._shape_results(points, empty_shape_records(), shape_jobs=2)
            except RuntimeError as error:
                if "Unable to start forked pyhf workers" in str(error):
                    self.skipTest(str(error))
                raise
        self.assertEqual([row["status"] for row in rows], ["ok", "ok"])

    def test_v2_input_report_writes_normalized_and_stacked_full91_gallery(self):
        signal = sample("sm", "sm_signal", [1] * 5, [3, 6, 9, 12, 15])
        hhhbb = sample(
            "sm_hhhbb",
            "postfit_hhhbb_signal",
            [1] * 5,
            [1, 2, 3, 4, 5],
            0,
            0,
        )
        hhhbb.metadata = {
            "process_id": "sm_hhhbb",
            "postfit_signal_component": "hhhbb",
        }
        background = sample("bkg", "background", [1] * 5, [2, 4, 6, 8, 10])
        background.metadata = {"process_id": "gg_to_8b"}
        stacked_metadata = {
            "kind": "stacked_input_xsec",
            "signal_scale": 1000.0,
        }

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "write_observable_shape_plot"
        ) as normalized_plot, mock.patch.object(
            runner,
            "write_stacked_input_cross_section_plot",
            return_value=stacked_metadata,
        ) as stacked_plot:
            report = runner.write_v2_input_observable_report(
                [signal],
                [background],
                directory,
                observable_set="extended-91-v2",
                feature_profile="full91",
                luminosity=3000.0,
                comparison_signal_samples=[hhhbb],
            )

            self.assertEqual(normalized_plot.call_count, 91)
            self.assertEqual(stacked_plot.call_count, 91)
            self.assertEqual(report["plot_count"], 182)
            self.assertEqual(report["comparison_signal_count"], 1)
            self.assertTrue(Path(report["index"]).is_file())
            self.assertTrue(Path(report["metadata"]).is_file())
            first_samples = stacked_plot.call_args_list[0].args[2]
            np.testing.assert_array_equal(first_samples[0]["weights"], signal.physical_weights)
            np.testing.assert_array_equal(
                first_samples[1]["weights"], hhhbb.physical_weights
            )
            np.testing.assert_array_equal(
                first_samples[2]["weights"], background.physical_weights
            )
            self.assertEqual(first_samples[0]["signal_component"], "hhhh")
            self.assertEqual(first_samples[1]["signal_component"], "hhhbb")
            self.assertFalse(first_samples[1]["included_in_training"])
            self.assertEqual(
                first_samples[1]["analysis_role"],
                "post-training-signal-comparison",
            )
            self.assertAlmostEqual(
                first_samples[0]["input_xsec_fb"],
                float(np.sum(signal.physical_weights)) / 3000.0,
            )

    def test_backfill_report_uses_sm_hhhbb_only_at_sm_point(self):
        manifest = {
            "status": "complete",
            "observable_set": "extended-91-v2",
            "selected_feature_profile": "full91",
            "luminosity_fb_inverse": 3000.0,
            "cv_folds": 5,
            "seed": 12345,
            "mode_policy": {"max_events_per_source": None},
            "inputs": [
                {"kind": "sm_signal", "path": "/sm.root", "xsec_fb": 1.0},
                {"kind": "grid_signal", "path": "/grid.root", "xsec_fb": 1.0},
                {
                    "kind": "postfit_hhhbb_signal",
                    "path": "/sm_hhhbb.root",
                    "xsec_fb": 0.1,
                    "c3": 0.0,
                    "d4": 0.0,
                    "metadata": {"postfit_signal_component": "hhhbb"},
                },
                {
                    "kind": "postfit_hhhbb_signal",
                    "path": "/non_sm_hhhbb.root",
                    "xsec_fb": 0.2,
                    "c3": 1.0,
                    "d4": 0.0,
                    "metadata": {"postfit_signal_component": "hhhbb"},
                },
                {"kind": "background", "path": "/bkg.root", "xsec_fb": 2.0},
            ],
            "outputs": {},
        }
        loaded_signal = sample("sm", "sm_signal", [1] * 5, [1] * 5)
        loaded_hhhbb = sample(
            "sm_hhhbb",
            "postfit_hhhbb_signal",
            [1] * 5,
            [1] * 5,
            0,
            0,
        )
        loaded_background = sample("bkg", "background", [1] * 5, [1] * 5)
        report = {"status": "complete", "plot_count": 182, "index": "/index.html"}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "method_manifest.json").write_text(json.dumps(manifest))
            with mock.patch.object(
                runner,
                "_load_samples",
                side_effect=[[loaded_signal], [loaded_hhhbb], [loaded_background]],
            ) as load_samples, mock.patch.object(
                runner, "write_v2_input_observable_report", return_value=report
            ) as write_report:
                result = runner.write_c3d4_input_report_from_manifest(output)

            self.assertEqual(result, report)
            self.assertEqual(load_samples.call_count, 3)
            self.assertEqual(load_samples.call_args_list[0].kwargs["kind"], "sm_signal")
            self.assertEqual(
                load_samples.call_args_list[1].kwargs["kind"],
                "postfit_hhhbb_signal",
            )
            self.assertEqual(load_samples.call_args_list[2].kwargs["kind"], "background")
            self.assertEqual(
                [str(spec["path"]) for spec in load_samples.call_args_list[1].args[0]],
                ["/sm_hhhbb.root"],
            )
            self.assertNotIn("grid_signal", repr(load_samples.call_args_list))
            write_report.assert_called_once()
            self.assertEqual(
                write_report.call_args.kwargs["comparison_signal_samples"],
                [loaded_hhhbb],
            )
            updated = json.loads((output / "method_manifest.json").read_text())
            self.assertEqual(updated["outputs"]["input_observable_report"], report)


if __name__ == "__main__":
    unittest.main()
