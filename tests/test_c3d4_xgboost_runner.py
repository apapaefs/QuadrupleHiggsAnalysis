from __future__ import annotations

import csv
import importlib.util
import io
import json
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
        stack.enter_context(mock.patch.object(runner, "_write_standard_maps", fake_maps))
        stack.enter_context(
            mock.patch.object(runner, "_shape_fingerprint", return_value="fingerprint")
        )
        stack.enter_context(mock.patch.object(runner, "_shape_results", fake_shape_results))
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
            cutflow = json.loads(
                (cutflow_dir / "sm_background_cutflow.json").read_text()
            )
            self.assertEqual(cutflow["thresholds_by_fold"], [0.5] * 5)
            self.assertEqual(cutflow["rows"][0]["input_events"], 5.0)
            self.assertEqual(
                manifest["mode_policy"]["training_strategy"], "sm-crossfit-v2"
            )
            self.assertTrue(manifest["score_shape_enabled"])
            self.assertTrue((output / "sm-crossfit-v2").is_dir())
            self.assertFalse((output / "pooled-crossfit-v2").exists())

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
        for mode in ("preview", "fast-sm", "full"):
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
                            training_strategy="sm-crossfit-v2",
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


if __name__ == "__main__":
    unittest.main()
