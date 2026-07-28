"""End-to-end resolved-8b c3/d4 XGBoost v2 study orchestration.

This module deliberately lives alongside, rather than inside, the historical
``xgboost_root_varfiles_module``.  The old hold-out analysis and its files are
therefore not changed by the cross-fitted study implemented here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from c3d4_xgboost_study import (
    NoValidThresholdError,
    background_threshold_scan,
    binned_weight_summary,
    build_pyhf_channel,
    deterministic_folds,
    enumerate_score_binnings,
    exact_cls_signal_upper_limit,
    limit_objective,
    make_background_parameter_replicas,
    optimize_point_threshold,
    parameterize_features,
    parameterized_gate,
    pyhf_combined_limit,
    pyhf_one_bin_limit,
    rotation_masks,
    run_optuna_tuning,
    select_test_binning,
    validation_binning,
)
from observable_schemas import (
    EXTENDED_SCHEMA_ID,
    LEGACY_SCHEMA_ID,
    PARAMETERIZED_ML_FEATURES,
    attach_model_metadata,
    canonical_sample_id,
    get_feature_contract,
    validate_model_contract,
)
from read_root_varfiles import read_ROOT_varfile
from sample_report import (
    safe_feature_filename,
    sample_latex_label,
    sample_style,
    terminal_sm_background_cutflow_table,
    write_observable_shape_plot,
    write_report_index,
    write_stacked_input_cross_section_plot,
)


METHOD_VERSION = "resolved-8b-c3d4-xgboost-v2.5"
CLASSIFIER_WEIGHT_SCALE_VERSION = "equal-class-mean-effective-row-weight-1-v1"
BASE_SEED = 12345
DEFAULT_PROFILES = ("corrected28", "core52", "full91")
SHAPE_CHECKPOINT_VERSION = 2
SHAPE_ORCHESTRATION_VERSION = "parallel-checkpoint-postfit-signal-v2"
PYHF_POI_BRACKET_MULTIPLIER = 10.0
COUPLING_HOLDOUT_VERSION = "balanced-hash-fivefold-v1"
LEGACY_CONTOUR_STYLE_VERSION = "legacy-c3d4-overlay-v1"
DEFAULT_CONTOUR_C3_RANGE = (-20.0, 20.0)
DEFAULT_CONTOUR_D4_RANGE = (-500.0, 500.0)
DEFAULT_CONTOUR_GRID_BINS = 301
EXPECTED_C3D4_SIGNAL_POINT_COUNT = 57
CONTOUR_INTERPOLATION_METHODS = ("linear", "clough-tocher")
OPTUNA_TUNED_PARAMETER_NAMES = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "gamma",
    "reg_alpha",
    "reg_lambda",
)
DEFAULT_HHHH_XSEC_SOURCE_DIR = Path(
    "/mnt/ssd2/Projects/4H/MG5_aMC_v3_5_15/gg_4h_c3d4"
)
_LEGACY_XSEC_SURFACE_CACHE: dict[
    tuple[Any, ...], tuple[np.ma.MaskedArray, dict[str, Any]]
] = {}
_LEGACY_PERTURBATIVITY_CACHE: dict[tuple[Any, ...], np.ndarray] = {}
SHAPE_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
)
FIXED_XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "n_jobs": 1,
}

_POINT_PATTERN = re.compile(
    r"run_gg_(?:4h|hhhg)_[^_/]+_"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)_"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


@dataclass
class EventSample:
    path: Path
    sample_id: str
    kind: str
    features: np.ndarray
    raw_weights: np.ndarray
    physical_weights: np.ndarray
    unit_xsec_weights: np.ndarray
    event_indices: np.ndarray
    source_entry_indices: np.ndarray
    folds: np.ndarray
    xsec_fb: float
    rate_factor: float
    normalisation_weight: float
    normalisation_source: str
    generated_events: int | None
    c3: float | None = None
    d4: float | None = None
    metadata: dict[str, Any] | None = None

    @property
    def point_id(self) -> str | None:
        if self.c3 is None or self.d4 is None:
            return None
        return f"c3={self.c3:.12g},d4={self.d4:.12g}"

    @property
    def entries(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class ShapePoint:
    """Minimal point descriptor needed by a per-point shape evaluation."""

    sample_id: str
    point_id: str
    c3: float
    d4: float
    xsec_fb: float = 1.0


class ShapeEvaluationIncompleteError(RuntimeError):
    """Raised after checkpointing when a shape stage has retryable failures."""


@dataclass(frozen=True)
class StudyModePolicy:
    """Resolved, machine-readable execution contract for one study mode."""

    name: str
    feature_profile: str | None
    training_strategy: str
    optuna_trials: int
    max_events: int | None
    run_shape: bool
    run_profile_ablation: bool
    run_parameterized_gate: bool
    run_coupling_holdout: bool
    hash_inputs: bool
    result_level: str
    physics_result_valid: bool
    paper_ready: bool
    plot_watermark: str | None


class ZeroSplitModelError(RuntimeError):
    """Raised when an XGBoost configuration cannot produce a useful split."""


def _resolve_study_mode(
    *,
    study_mode: str,
    observable_set: str,
    feature_profile: str | None,
    training_strategy: str | None,
    optuna_trials: int | None,
    max_events: int | None,
    smoke_max_events: int,
    run_shape: bool | None,
    hash_inputs: bool,
) -> StudyModePolicy:
    """Resolve mode defaults and reject combinations with ambiguous physics status."""

    mode = str(study_mode).strip().lower()
    if mode not in {
        "smoke",
        "preview",
        "fast-sm",
        "fast-pooled",
        "fast-parameterized",
        "full",
    }:
        raise ValueError(f"Unknown study mode {study_mode!r}")
    default_strategy = {
        "smoke": "sm-crossfit-v2",
        "preview": "pooled-crossfit-v2",
        "fast-sm": "sm-crossfit-v2",
        "fast-pooled": "pooled-crossfit-v2",
        "fast-parameterized": "parameterized-crossfit-v1",
        "full": "pooled-crossfit-v2",
    }[mode]
    strategy = training_strategy or default_strategy
    if strategy not in {
        "sm-crossfit-v2",
        "pooled-crossfit-v2",
        "parameterized-crossfit-v1",
    }:
        raise ValueError(f"Unknown training strategy {strategy!r}")
    if (
        mode not in {"full", "fast-parameterized"}
        and strategy == "parameterized-crossfit-v1"
    ):
        raise ValueError(
            f"{mode} mode supports only sm-crossfit-v2 or pooled-crossfit-v2; "
            "parameterized training requires full or fast-parameterized mode"
        )
    if mode == "fast-sm" and strategy != "sm-crossfit-v2":
        raise ValueError("fast-sm mode requires sm-crossfit-v2 training")
    if mode == "fast-pooled" and strategy != "pooled-crossfit-v2":
        raise ValueError("fast-pooled mode requires pooled-crossfit-v2 training")
    if mode == "fast-parameterized" and strategy != "parameterized-crossfit-v1":
        raise ValueError(
            "fast-parameterized mode requires parameterized-crossfit-v1 training"
        )

    if mode == "full":
        if max_events is not None:
            raise ValueError(
                "full mode requires complete event samples; use smoke mode for --max-events"
            )
        trials = 40 if optuna_trials is None else int(optuna_trials)
        if trials < 0:
            raise ValueError("full-mode optuna_trials must be non-negative")
        shape = True if run_shape is None else bool(run_shape)
        profile = feature_profile
        result_level = "full" if shape else "preliminary-cut-only"
        return StudyModePolicy(
            name=mode,
            feature_profile=profile,
            training_strategy=strategy,
            optuna_trials=trials,
            max_events=None,
            run_shape=shape,
            run_profile_ablation=True,
            run_parameterized_gate=True,
            run_coupling_holdout=False,
            hash_inputs=bool(hash_inputs),
            result_level=result_level,
            physics_result_valid=True,
            paper_ready=bool(shape),
            plot_watermark=(
                None if shape else "PRELIMINARY - SINGLE-BIN CUT RESULT"
            ),
        )

    if mode in {"fast-sm", "fast-pooled", "fast-parameterized"}:
        if max_events is not None:
            raise ValueError(
                f"{mode} mode requires complete event samples; "
                "use smoke mode for --max-events"
            )
        if optuna_trials not in (None, 0):
            raise ValueError(
                f"{mode} mode uses fixed XGBoost parameters; "
                "omit --optuna-trials or set it to 0"
            )
        shape = True if run_shape is None else bool(run_shape)
        profile = feature_profile or (
            "corrected28" if observable_set == LEGACY_SCHEMA_ID else "full91"
        )
        return StudyModePolicy(
            name=mode,
            feature_profile=profile,
            training_strategy=(
                "sm-crossfit-v2"
                if mode == "fast-sm"
                else (
                    "pooled-crossfit-v2"
                    if mode == "fast-pooled"
                    else "parameterized-crossfit-v1"
                )
            ),
            optuna_trials=0,
            max_events=None,
            run_shape=shape,
            run_profile_ablation=False,
            run_parameterized_gate=False,
            run_coupling_holdout=mode == "fast-parameterized",
            hash_inputs=bool(hash_inputs),
            result_level=("fixed-parameter-full" if shape else "preliminary-cut-only"),
            physics_result_valid=True,
            paper_ready=bool(shape),
            plot_watermark=(
                None if shape else "PRELIMINARY - SINGLE-BIN CUT RESULT"
            ),
        )

    if optuna_trials not in (None, 0):
        raise ValueError(
            f"{mode} mode uses fixed XGBoost parameters; omit --optuna-trials or set it to 0"
        )
    if run_shape not in (None, False):
        raise ValueError(f"{mode} mode does not run the pyhf score-shape stage")
    if mode == "preview":
        if max_events is not None:
            raise ValueError(
                "preview mode requires complete event samples; use smoke mode for --max-events"
            )
        profile = feature_profile or (
            "corrected28" if observable_set == LEGACY_SCHEMA_ID else "core52"
        )
        return StudyModePolicy(
            name=mode,
            feature_profile=profile,
            training_strategy=strategy,
            optuna_trials=0,
            max_events=None,
            run_shape=False,
            run_profile_ablation=False,
            run_parameterized_gate=False,
            run_coupling_holdout=False,
            hash_inputs=bool(hash_inputs),
            result_level="preliminary-cut-only",
            physics_result_valid=True,
            paper_ready=False,
            plot_watermark="PRELIMINARY - SINGLE-BIN CUT RESULT",
        )

    smoke_limit = int(smoke_max_events if max_events is None else max_events)
    if smoke_limit < 1:
        raise ValueError("smoke-mode max_events must be at least one")
    profile = feature_profile or "corrected28"
    return StudyModePolicy(
        name=mode,
        feature_profile=profile,
        training_strategy=strategy,
        optuna_trials=0,
        max_events=smoke_limit,
        run_shape=False,
        run_profile_ablation=False,
        run_parameterized_gate=False,
        run_coupling_holdout=False,
        hash_inputs=False,
        result_level="non-physics-smoke",
        physics_result_valid=False,
        paper_ready=False,
        plot_watermark="NON-PHYSICS SMOKE TEST",
    )


def _finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _parse_point(path: str | Path) -> tuple[float, float]:
    match = _POINT_PATTERN.search(canonical_sample_id(path))
    if not match:
        raise ValueError(f"Cannot extract c3,d4 coordinates from {path}")
    return float(match.group(1)), float(match.group(2))


def _normalisation_denominator(
    generated_events: int | None,
    raw_weights: np.ndarray,
    normalisation_weight: float | None,
) -> tuple[float, str]:
    if normalisation_weight is not None:
        denominator = _finite_float(normalisation_weight, "normalisation_weight")
        if denominator <= 0.0:
            raise ValueError("normalisation_weight must be positive")
        return denominator, "analysis_total_weight_in"

    if generated_events is None or int(generated_events) <= 0:
        raise ValueError(
            "Missing total input-weight metadata; provide analysis total_weight_in or "
            "a generated-event count for a unit-weight sample"
        )
    if raw_weights.size and not np.allclose(raw_weights, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "A non-unit-weight sample is missing analysis total_weight_in; refusing to "
            "normalise it with a selected-event count"
        )
    return float(generated_events), "generated_events_unit_weights"


def _load_sample(
    spec: Mapping[str, Any],
    *,
    kind: str,
    observable_set: str,
    luminosity: float,
    n_folds: int,
    seed: int,
    max_events: int | None,
) -> EventSample:
    path = Path(spec["path"])
    sample_id = canonical_sample_id(path)
    load_profile = "full91" if observable_set == EXTENDED_SCHEMA_ID else "corrected28"
    features, _, _, root_metadata = read_ROOT_varfile(
        path,
        1 if kind != "background" else 0,
        xsec=1.0,
        max_events=max_events,
        observable_set=observable_set,
        feature_profile=load_profile,
        return_metadata=True,
    )
    feature_array = np.asarray(features, dtype=float)
    raw_weights = np.asarray(root_metadata["raw_weights"], dtype=float)
    event_indices = np.asarray(root_metadata["event_indices"], dtype=np.int64)
    source_entries = np.asarray(root_metadata["source_entry_indices"], dtype=np.int64)
    if feature_array.ndim != 2 or feature_array.shape[0] != raw_weights.size:
        raise ValueError(f"{path}: feature and weight arrays do not have matching rows")
    if raw_weights.size < n_folds:
        raise ValueError(f"{path}: only {raw_weights.size} valid entries for {n_folds} folds")

    xsec_fb = _finite_float(spec["xsec_fb"], f"{path} xsec_fb")
    rate_factor = _finite_float(spec.get("rate_factor", 1.0), f"{path} rate_factor")
    generated = spec.get("generated_events")
    generated = None if generated is None else int(generated)
    denominator, denominator_source = _normalisation_denominator(
        generated,
        raw_weights,
        spec.get("normalisation_weight"),
    )
    unit_xsec_weights = float(luminosity) * rate_factor * raw_weights / denominator
    physical_weights = xsec_fb * unit_xsec_weights
    source_ids = np.full(raw_weights.size, sample_id, dtype=object)
    folds = deterministic_folds(source_ids, event_indices, n_folds=n_folds, seed=seed)
    c3 = d4 = None
    if kind in {"grid_signal", "postfit_hhhbb_signal"}:
        c3, d4 = _parse_point(path)

    return EventSample(
        path=path,
        sample_id=sample_id,
        kind=kind,
        features=feature_array,
        raw_weights=raw_weights,
        physical_weights=physical_weights,
        unit_xsec_weights=unit_xsec_weights,
        event_indices=event_indices,
        source_entry_indices=source_entries,
        folds=np.asarray(folds, dtype=np.int16),
        xsec_fb=xsec_fb,
        rate_factor=rate_factor,
        normalisation_weight=denominator,
        normalisation_source=denominator_source,
        generated_events=generated,
        c3=c3,
        d4=d4,
        metadata=dict(spec.get("metadata") or {}),
    )


def _load_samples(
    specs: Sequence[Mapping[str, Any]],
    progress: Callable[[int, int, EventSample], None] | None = None,
    **kwargs: Any,
) -> list[EventSample]:
    samples = []
    total = len(specs)
    for index, spec in enumerate(specs, start=1):
        sample = _load_sample(spec, **kwargs)
        samples.append(sample)
        if progress is not None:
            progress(index, total, sample)
    return samples


def _profile_indices(observable_set: str, profile: str) -> np.ndarray:
    contract = get_feature_contract(observable_set, profile)
    return np.asarray(contract.feature_indices, dtype=int)


def _fold_mask(sample: EventSample, rotation: int, split: str, n_folds: int) -> np.ndarray:
    masks = rotation_masks(sample.folds, rotation, n_folds=n_folds)
    return np.asarray(masks[split], dtype=bool)


def _balanced_weights(
    signal_weights: np.ndarray,
    background_weights: np.ndarray,
    *,
    effective_row_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance the classes while keeping the mean effective-row weight at one."""

    signal_weights = np.asarray(signal_weights, dtype=float)
    background_weights = np.asarray(background_weights, dtype=float)
    if (
        signal_weights.ndim != 1
        or background_weights.ndim != 1
        or np.any(~np.isfinite(signal_weights))
        or np.any(~np.isfinite(background_weights))
        or np.any(signal_weights < 0.0)
        or np.any(background_weights < 0.0)
    ):
        raise ValueError(
            "Classifier weights must be finite, nonnegative one-dimensional arrays"
        )
    signal_total = float(np.sum(signal_weights))
    background_total = float(np.sum(background_weights))
    if signal_total <= 0.0 or background_total <= 0.0:
        raise ValueError("Both classifier classes require positive absolute training weight")
    if effective_row_count is None:
        effective_row_count = int(
            np.count_nonzero(signal_weights) + np.count_nonzero(background_weights)
        )
    effective_row_count = int(effective_row_count)
    if effective_row_count <= 0:
        raise ValueError("effective_row_count must be positive")
    class_total = 0.5 * float(effective_row_count)
    return (
        signal_weights * (class_total / signal_total),
        background_weights * (class_total / background_total),
    )


def _training_arrays(
    sm_samples: Sequence[EventSample],
    grid_samples: Sequence[EventSample],
    background_samples: Sequence[EventSample],
    *,
    strategy: str,
    profile_indices: np.ndarray,
    rotation: int,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if strategy == "sm-crossfit-v2":
        signal_samples = list(sm_samples)
    elif strategy in {"pooled-crossfit-v2", "parameterized-crossfit-v1"}:
        signal_samples = list(grid_samples)
    else:
        raise ValueError(f"Unknown training strategy {strategy!r}")
    if not signal_samples:
        raise ValueError(f"{strategy} has no signal samples")

    signal_features = []
    signal_weights = []
    point_count = len(signal_samples)
    for sample in signal_samples:
        mask = _fold_mask(sample, rotation, "train", n_folds)
        raw = np.abs(sample.raw_weights[mask])
        total = float(np.sum(raw))
        if total <= 0.0:
            raise ValueError(f"{sample.sample_id}: zero absolute signal training weight")
        features = sample.features[mask][:, profile_indices]
        if strategy == "parameterized-crossfit-v1":
            if sample.c3 is None or sample.d4 is None:
                raise ValueError("parameterized signal training requires c3,d4 coordinates")
            features = parameterize_features(features, sample.c3, sample.d4)
        signal_features.append(features)
        signal_weights.append(raw / total / point_count)

    background_features = []
    background_weights = []
    background_effective_rows = 0
    grid_points = np.asarray(
        [(sample.c3, sample.d4) for sample in grid_samples], dtype=float
    )
    for sample in background_samples:
        mask = _fold_mask(sample, rotation, "train", n_folds)
        # Absolute weights are required by XGBoost.  Their physical factors
        # retain the relative process mixture within the background class.
        weights = np.abs(sample.physical_weights[mask])
        background_effective_rows += int(np.count_nonzero(weights))
        features = sample.features[mask][:, profile_indices]
        if strategy == "parameterized-crossfit-v1":
            replicas = make_background_parameter_replicas(
                features,
                weights,
                sample.folds[mask],
                np.full(np.sum(mask), sample.sample_id, dtype=object),
                sample.event_indices[mask],
                grid_points,
                replicas_per_event=3,
                seed=BASE_SEED,
            )
            background_features.append(replicas["features"])
            background_weights.append(replicas["training_weights"])
        else:
            background_features.append(features)
            background_weights.append(weights)

    signal_X = np.concatenate(signal_features, axis=0)
    background_X = np.concatenate(background_features, axis=0)
    signal_w = np.concatenate(signal_weights)
    background_w = np.concatenate(background_weights)
    effective_row_count = int(np.count_nonzero(signal_w) + background_effective_rows)
    signal_w, background_w = _balanced_weights(
        signal_w,
        background_w,
        effective_row_count=effective_row_count,
    )
    X = np.concatenate([signal_X, background_X], axis=0)
    y = np.concatenate(
        [np.ones(signal_X.shape[0], dtype=np.int8), np.zeros(background_X.shape[0], dtype=np.int8)]
    )
    weights = np.concatenate([signal_w, background_w])
    return X, y, weights


def _classifier_weight_diagnostics(
    y: np.ndarray,
    weights: np.ndarray,
    min_child_weight: float,
) -> dict[str, Any]:
    labels = np.asarray(y)
    classifier_weights = np.asarray(weights, dtype=float)
    if (
        labels.ndim != 1
        or classifier_weights.ndim != 1
        or len(labels) != len(classifier_weights)
    ):
        raise ValueError("Classifier labels and weights must be matching one-dimensional arrays")
    if np.any(~np.isfinite(classifier_weights)) or np.any(classifier_weights < 0.0):
        raise ValueError("Classifier weights must be finite and nonnegative")
    signal_total = float(np.sum(classifier_weights[labels == 1]))
    background_total = float(np.sum(classifier_weights[labels == 0]))
    if signal_total <= 0.0 or background_total <= 0.0:
        raise ValueError("Both classifier classes require positive training weight")
    if not math.isclose(signal_total, background_total, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError("Signal and background classifier totals are not balanced")
    total = signal_total + background_total
    maximum_root_hessian = 0.25 * total
    minimum_split_hessian = 2.0 * float(min_child_weight)
    diagnostics = {
        "classifier_weight_scale_version": CLASSIFIER_WEIGHT_SCALE_VERSION,
        "classifier_effective_row_count": int(round(total)),
        "classifier_equal_class_total": signal_total,
        "classifier_signal_weight_total": signal_total,
        "classifier_background_weight_total": background_total,
        "classifier_total_weight": total,
        "maximum_initial_root_hessian": maximum_root_hessian,
        "minimum_hessian_for_binary_split": minimum_split_hessian,
    }
    if maximum_root_hessian < minimum_split_hessian:
        raise ZeroSplitModelError(
            "The classifier weight scale makes every binary split impossible: "
            f"maximum initial root Hessian {maximum_root_hessian:g} is below "
            f"2*min_child_weight={minimum_split_hessian:g}"
        )
    return diagnostics


def _xgboost_split_count(booster: Any) -> int | None:
    if not hasattr(booster, "get_dump"):
        return None
    return int(
        sum(tree.count('"split"') for tree in booster.get_dump(dump_format="json"))
    )


def _train_model(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    params: Mapping[str, Any],
    seed: int,
    observable_set: str,
    profile: str,
    strategy: str,
    rotation: int,
    source_commit: str,
):
    model_params = dict(FIXED_XGBOOST_PARAMS)
    model_params.update(dict(params))
    model_params["random_state"] = int(seed)
    model_params["n_jobs"] = 1
    weight_diagnostics = _classifier_weight_diagnostics(
        y,
        weights,
        model_params["min_child_weight"],
    )

    import xgboost as xgb

    model = xgb.XGBClassifier(**model_params)
    model.fit(X, y, sample_weight=weights, verbose=False)
    contract = get_feature_contract(observable_set, profile)
    ml_parameters = (
        PARAMETERIZED_ML_FEATURES
        if strategy == "parameterized-crossfit-v1"
        else None
    )
    model_feature_names = list(contract.feature_names)
    if ml_parameters:
        model_feature_names.extend(name for name, _ in ml_parameters)
    booster = model.get_booster()
    booster.feature_names = model_feature_names
    split_count = _xgboost_split_count(booster)
    if split_count == 0:
        raise ZeroSplitModelError(
            "XGBoost trained only constant leaves and produced zero split nodes"
        )
    score_diagnostics: dict[str, Any] = {}
    if hasattr(model, "predict_proba"):
        training_scores = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
        score_diagnostics = {
            "training_score_min": float(np.min(training_scores)),
            "training_score_max": float(np.max(training_scores)),
            "training_score_std": float(np.std(training_scores)),
        }
        if float(np.ptp(training_scores)) <= 1.0e-12:
            raise ZeroSplitModelError(
                "XGBoost training scores are numerically constant despite fitted split nodes"
            )
    metadata = attach_model_metadata(
        model,
        observable_set=observable_set,
        feature_profile=profile,
        ml_parameter_features=ml_parameters,
        training_strategy=strategy,
        method_version=METHOD_VERSION,
        fold=int(rotation),
        parameters=model_params,
        seed=int(seed),
        source_commit=source_commit,
        xgboost_split_nodes=split_count,
        **weight_diagnostics,
        **score_diagnostics,
    )
    validate_model_contract(
        model,
        observable_set,
        profile,
        ml_parameter_features=ml_parameters,
    )
    return model, metadata, model_params


def _predict(
    model: Any,
    sample: EventSample,
    mask: np.ndarray,
    profile_indices: np.ndarray,
    parameter_point: tuple[float, float] | None = None,
) -> np.ndarray:
    if not np.any(mask):
        return np.asarray([], dtype=float)
    features = sample.features[mask][:, profile_indices]
    if parameter_point is not None:
        features = parameterize_features(features, parameter_point[0], parameter_point[1])
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def _partition_scale(n_folds: int) -> float:
    """Return the predeclared validation-to-full-luminosity scale.

    Round-robin assignment makes every validation fold an unbiased one-in-K
    subsample.  Using K is independent of all held-out test-fold weights,
    unlike closing a validation source to its subsequently observed full sum.
    """

    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    return float(n_folds)


def _score_partition(
    model: Any,
    samples: Sequence[EventSample],
    *,
    rotation: int,
    split: str,
    n_folds: int,
    profile_indices: np.ndarray,
    scale_validation_to_full: bool,
    parameterized: bool = False,
    parameter_point: tuple[float, float] | None = None,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for sample in samples:
        mask = _fold_mask(sample, rotation, split, n_folds)
        scale = _partition_scale(n_folds) if scale_validation_to_full else 1.0
        scoring_point = parameter_point
        if parameterized and scoring_point is None:
            if sample.c3 is None or sample.d4 is None:
                raise ValueError(
                    "parameterized scoring requires either a point or signal coordinates"
                )
            scoring_point = (float(sample.c3), float(sample.d4))
        output[sample.sample_id] = {
            "sample": sample,
            "mask": mask,
            "scores": _predict(
                model,
                sample,
                mask,
                profile_indices,
                scoring_point if parameterized else None,
            ),
            "physical_weights": sample.physical_weights[mask] * scale,
            "unit_xsec_weights": sample.unit_xsec_weights[mask] * scale,
            "raw_weights": sample.raw_weights[mask],
            "event_indices": sample.event_indices[mask],
            "scale": float(scale),
        }
    return output


def _concatenate_partition(rows: Mapping[str, Mapping[str, Any]], key: str) -> np.ndarray:
    arrays = [np.asarray(row[key]) for row in rows.values()]
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=float)


def _effective_entries(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights * weights))
    return 0.0 if denominator <= 0.0 else float(np.sum(weights)) ** 2 / denominator


def _validation_limits(
    model: Any,
    grid_samples: Sequence[EventSample],
    background_samples: Sequence[EventSample],
    *,
    rotation: int,
    n_folds: int,
    profile_indices: np.ndarray,
    min_background_raw: int = 25,
    min_background_neff: float = 10.0,
    parameterized: bool = False,
) -> dict[str, Any]:
    signal_rows = _score_partition(
        model,
        grid_samples,
        rotation=rotation,
        split="validation",
        n_folds=n_folds,
        profile_indices=profile_indices,
        scale_validation_to_full=True,
        parameterized=parameterized,
    )
    thresholds = np.linspace(0.0, 1.0, 1001)
    background_rows = None
    reusable_background_scan = None
    if not parameterized:
        background_rows = _score_partition(
            model,
            background_samples,
            rotation=rotation,
            split="validation",
            n_folds=n_folds,
            profile_indices=profile_indices,
            scale_validation_to_full=True,
        )
        background_scores = _concatenate_partition(background_rows, "scores")
        background_weights = _concatenate_partition(background_rows, "physical_weights")
        reusable_background_scan = background_threshold_scan(
            background_scores,
            background_weights,
            thresholds=thresholds,
        )

    point_results: dict[str, dict[str, Any]] = {}
    sigma_values = []
    for sample in grid_samples:
        row = signal_rows[sample.sample_id]
        if parameterized:
            point_background_rows = _score_partition(
                model,
                background_samples,
                rotation=rotation,
                split="validation",
                n_folds=n_folds,
                profile_indices=profile_indices,
                scale_validation_to_full=True,
                parameterized=True,
                parameter_point=(float(sample.c3), float(sample.d4)),
            )
            background_scores = _concatenate_partition(point_background_rows, "scores")
            background_weights = _concatenate_partition(
                point_background_rows, "physical_weights"
            )
            point_background_scan = background_threshold_scan(
                background_scores,
                background_weights,
                thresholds=thresholds,
            )
        else:
            point_background_scan = reusable_background_scan
        try:
            optimum = optimize_point_threshold(
                row["scores"],
                row["unit_xsec_weights"],
                background_scores,
                background_weights,
                luminosity=None,
                thresholds=thresholds,
                min_background_raw=min_background_raw,
                min_background_neff=min_background_neff,
                background_scan=point_background_scan,
            )
        except NoValidThresholdError as exc:
            raise NoValidThresholdError(
                f"rotation {rotation}, point {sample.point_id}: {exc}"
            ) from exc
        compact = {key: value for key, value in optimum.items() if key != "candidates"}
        compact.update(
            {
                "sample_id": sample.sample_id,
                "c3": sample.c3,
                "d4": sample.d4,
                "validation_scale": row["scale"],
            }
        )
        point_results[sample.point_id] = compact
        sigma_values.append(float(optimum["sigma95_fb"]))
    return {
        "rotation": int(rotation),
        "objective": limit_objective(sigma_values),
        "points": point_results,
        "signal_rows": signal_rows,
        "background_rows": background_rows or {},
        "parameterized": bool(parameterized),
    }


def _fit_rotation(
    sm_samples: Sequence[EventSample],
    grid_samples: Sequence[EventSample],
    background_samples: Sequence[EventSample],
    *,
    strategy: str,
    observable_set: str,
    profile: str,
    rotation: int,
    n_folds: int,
    params: Mapping[str, Any],
    seed: int,
    source_commit: str,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    indices = _profile_indices(observable_set, profile)
    X, y, weights = _training_arrays(
        sm_samples,
        grid_samples,
        background_samples,
        strategy=strategy,
        profile_indices=indices,
        rotation=rotation,
        n_folds=n_folds,
    )
    model, model_metadata, model_params = _train_model(
        X,
        y,
        weights,
        params=params,
        seed=seed,
        observable_set=observable_set,
        profile=profile,
        strategy=strategy,
        rotation=rotation,
        source_commit=source_commit,
    )
    validation = _validation_limits(
        model,
        grid_samples,
        background_samples,
        rotation=rotation,
        n_folds=n_folds,
        profile_indices=indices,
        parameterized=strategy == "parameterized-crossfit-v1",
    )
    validation["n_train"] = int(X.shape[0])
    return model, validation, model_metadata, model_params


def _suggest_xgboost_params(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_categorical("gamma", [0.0, 0.01, 0.1, 1.0]),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [0.0, 0.001, 0.01, 0.1, 1.0]),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
    }


def _remaining_optuna_attempts(states: Sequence[str], target_trials: int) -> int:
    """Return executions needed without replacing pruned or stale-running trials."""

    names = [str(state) for state in states]
    finished = sum(state in {"COMPLETE", "PRUNED", "FAIL"} for state in names)
    running = sum(state == "RUNNING" for state in names)
    waiting = sum(state == "WAITING" for state in names)
    allocated = finished + running + waiting
    new_slots = max(0, int(target_trials) - allocated)
    return waiting + new_slots


def _tune_rotation(
    sm_samples: Sequence[EventSample],
    grid_samples: Sequence[EventSample],
    background_samples: Sequence[EventSample],
    *,
    strategy: str,
    observable_set: str,
    profile: str,
    rotation: int,
    n_folds: int,
    trials: int,
    output_dir: Path,
    seed: int,
    source_commit: str,
    run_fingerprint: str,
    progress: StudyProgress | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import optuna

    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = (
        output_dir / f"fold_{rotation}_{run_fingerprint[:12]}.sqlite3"
    ).resolve()
    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{storage_path}",
        study_name=f"{strategy}-{observable_set}-{profile}-fold-{rotation}-{run_fingerprint[:12]}",
        load_if_exists=True,
    )
    stored_fingerprint = study.user_attrs.get("run_fingerprint")
    if stored_fingerprint not in (None, run_fingerprint):
        raise RuntimeError(
            "Refusing to resume an Optuna study with an incompatible run fingerprint"
        )
    study.set_user_attr("run_fingerprint", run_fingerprint)
    if len(study.trials) == 0:
        study.enqueue_trial({key: FIXED_XGBOOST_PARAMS[key] for key in (
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        )})

    def objective(trial: Any) -> float:
        params = _suggest_xgboost_params(trial)
        try:
            _, validation, model_metadata, _ = _fit_rotation(
                sm_samples,
                grid_samples,
                background_samples,
                strategy=strategy,
                observable_set=observable_set,
                profile=profile,
                rotation=rotation,
                n_folds=n_folds,
                params=params,
                seed=seed,
                source_commit=source_commit,
            )
        except ZeroSplitModelError as exc:
            trial.set_user_attr("zero_split_reason", str(exc))
            raise optuna.TrialPruned(str(exc)) from exc
        trial.set_user_attr(
            "xgboost_split_nodes", model_metadata.get("xgboost_split_nodes")
        )
        trial.set_user_attr(
            "training_score_std", model_metadata.get("training_score_std")
        )
        trial.set_user_attr("median_sigma95_fb", float(np.median([
            point["sigma95_fb"] for point in validation["points"].values()
        ])))
        return float(validation["objective"])

    attempts_to_execute = _remaining_optuna_attempts(
        [trial.state.name for trial in study.trials],
        int(trials),
    )
    if attempts_to_execute:
        tuning_started = time.monotonic()

        def report_trial(study_object: Any, trial: Any) -> None:
            if progress is None:
                return
            completed = sum(
                item.state == optuna.trial.TrialState.COMPLETE
                for item in study_object.trials
            )
            best_value = (
                float(study_object.best_value)
                if completed
                else None
            )
            progress.emit(
                "tuning",
                "Completed Optuna trial",
                strategy=strategy,
                profile=profile,
                fold=rotation + 1,
                trial=int(trial.number) + 1,
                completed_trials=completed,
                total_trials=int(trials),
                best_objective=best_value,
                tuning_elapsed_seconds=float(time.monotonic() - tuning_started),
            )

        study.optimize(
            objective,
            n_trials=attempts_to_execute,
            n_jobs=1,
            gc_after_trial=True,
            callbacks=[report_trial],
        )
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        pruned_reasons = sorted({
            str(trial.user_attrs.get("zero_split_reason"))
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.PRUNED
            and trial.user_attrs.get("zero_split_reason")
        })
        raise RuntimeError(
            f"Optuna produced no completed trial for fold {rotation}; "
            f"zero-split reasons: {pruned_reasons or ['none recorded']}"
        )
    history = []
    for trial in study.trials:
        history.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
        )
    return dict(study.best_params), {
        "storage": str(storage_path),
        "study_name": study.study_name,
        "best_trial": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "trials": history,
    }


def _validated_reused_xgboost_params(
    params: Mapping[str, Any], *, fold: int
) -> dict[str, Any]:
    """Validate and normalize one previously completed Optuna best trial."""

    expected = set(OPTUNA_TUNED_PARAMETER_NAMES)
    actual = set(params)
    if actual != expected:
        raise ValueError(
            f"Reused SM Optuna fold {fold} has incompatible parameter keys "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )
    result = dict(params)
    for name in ("n_estimators", "max_depth"):
        try:
            value = int(result[name])
        except (TypeError, ValueError):
            raise ValueError(
                f"Reused SM Optuna fold {fold} has invalid {name}"
            ) from None
        if float(result[name]) != float(value):
            raise ValueError(
                f"Reused SM Optuna fold {fold} has non-integral {name}"
            )
        result[name] = value
    for name in expected - {"n_estimators", "max_depth"}:
        try:
            value = float(result[name])
        except (TypeError, ValueError):
            raise ValueError(
                f"Reused SM Optuna fold {fold} has invalid {name}"
            ) from None
        if not math.isfinite(value):
            raise ValueError(
                f"Reused SM Optuna fold {fold} has non-finite {name}"
            )
        result[name] = value
    if not (200 <= result["n_estimators"] <= 800 and result["n_estimators"] % 100 == 0):
        raise ValueError(f"Reused SM Optuna fold {fold} has out-of-range n_estimators")
    if not 2 <= result["max_depth"] <= 6:
        raise ValueError(f"Reused SM Optuna fold {fold} has out-of-range max_depth")
    if not 0.01 <= result["learning_rate"] <= 0.15:
        raise ValueError(f"Reused SM Optuna fold {fold} has out-of-range learning_rate")
    if not 1.0 <= result["min_child_weight"] <= 50.0:
        raise ValueError(f"Reused SM Optuna fold {fold} has out-of-range min_child_weight")
    for name in ("subsample", "colsample_bytree"):
        if not 0.6 <= result[name] <= 1.0:
            raise ValueError(f"Reused SM Optuna fold {fold} has out-of-range {name}")
    if result["gamma"] not in {0.0, 0.01, 0.1, 1.0}:
        raise ValueError(f"Reused SM Optuna fold {fold} has unsupported gamma")
    if result["reg_alpha"] not in {0.0, 0.001, 0.01, 0.1, 1.0}:
        raise ValueError(f"Reused SM Optuna fold {fold} has unsupported reg_alpha")
    if not 0.1 <= result["reg_lambda"] <= 30.0:
        raise ValueError(f"Reused SM Optuna fold {fold} has out-of-range reg_lambda")
    return result


def _load_reused_sm_optuna(
    source_dir: str | Path,
    *,
    observable_set: str,
    profile: str,
    n_folds: int,
    seed: int,
) -> dict[str, Any]:
    """Load compatible per-fold SM parameters from a completed v2 study."""

    source_dir = Path(source_dir).expanduser().resolve()
    manifest_path = source_dir / "method_manifest.json"
    manifest = _read_json_mapping(manifest_path)
    if not manifest:
        raise ValueError(
            f"No readable method_manifest.json found in reused Optuna study {source_dir}"
        )
    if manifest.get("status") != "complete":
        raise ValueError("The reused SM Optuna source study is not complete")
    if manifest.get("observable_set") != observable_set:
        raise ValueError(
            "The reused SM Optuna source has a different observable schema"
        )
    if manifest.get("selected_feature_profile") != profile:
        raise ValueError(
            "The reused SM Optuna source has a different selected feature profile: "
            f"{manifest.get('selected_feature_profile')!r} versus {profile!r}"
        )
    if int(manifest.get("cv_folds", -1)) != int(n_folds):
        raise ValueError("The reused SM Optuna source has a different fold count")
    if int(manifest.get("seed", -1)) != int(seed):
        raise ValueError("The reused SM Optuna source has a different base seed")
    completed = manifest.get("strategies_completed", [])
    if "sm-crossfit-v2" not in completed:
        raise ValueError("The reused Optuna source did not complete sm-crossfit-v2")

    folds = []
    for fold in range(int(n_folds)):
        history_path = (
            source_dir
            / "sm-crossfit-v2"
            / "optuna"
            / f"fold_{fold}_history.json"
        )
        history = _read_json_mapping(history_path)
        if not history:
            raise ValueError(f"Missing readable reused SM Optuna history {history_path}")
        try:
            best_trial = int(history["best_trial"])
            best_value = float(history["best_value"])
            trials = history["trials"]
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Malformed reused SM Optuna history {history_path}") from None
        if not math.isfinite(best_value) or not isinstance(trials, Sequence):
            raise ValueError(f"Malformed reused SM Optuna best result {history_path}")
        matches = [
            trial
            for trial in trials
            if isinstance(trial, Mapping)
            and int(trial.get("number", -1)) == best_trial
            and trial.get("state", "COMPLETE") == "COMPLETE"
        ]
        if len(matches) != 1 or not isinstance(matches[0].get("params"), Mapping):
            raise ValueError(
                f"Could not identify one completed best trial in {history_path}"
            )
        params = _validated_reused_xgboost_params(
            matches[0]["params"], fold=fold
        )
        folds.append(
            {
                "fold": fold,
                "best_trial": best_trial,
                "best_value": best_value,
                "parameters": params,
                "source_history": str(history_path),
                "source_history_sha256": _sha256(history_path),
            }
        )
    return {
        "status": "reused",
        "source_study": str(source_dir),
        "source_method_version": manifest.get("method_version"),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "observable_set": observable_set,
        "feature_profile": profile,
        "cv_folds": int(n_folds),
        "seed": int(seed),
        "folds": folds,
    }


def _evaluate_test_rotation(
    model: Any,
    validation: Mapping[str, Any],
    grid_samples: Sequence[EventSample],
    background_samples: Sequence[EventSample],
    *,
    rotation: int,
    n_folds: int,
    profile_indices: np.ndarray,
    parameterized: bool = False,
) -> dict[str, Any]:
    signal_rows = _score_partition(
        model,
        grid_samples,
        rotation=rotation,
        split="test",
        n_folds=n_folds,
        profile_indices=profile_indices,
        scale_validation_to_full=False,
        parameterized=parameterized,
    )
    background_rows = None
    if not parameterized:
        background_rows = _score_partition(
            model,
            background_samples,
            rotation=rotation,
            split="test",
            n_folds=n_folds,
            profile_indices=profile_indices,
            scale_validation_to_full=False,
        )
        background_scores = _concatenate_partition(background_rows, "scores")
        background_weights = _concatenate_partition(background_rows, "physical_weights")
    point_rows = {}
    for sample in grid_samples:
        if parameterized:
            point_background_rows = _score_partition(
                model,
                background_samples,
                rotation=rotation,
                split="test",
                n_folds=n_folds,
                profile_indices=profile_indices,
                scale_validation_to_full=False,
                parameterized=True,
                parameter_point=(float(sample.c3), float(sample.d4)),
            )
            background_scores = _concatenate_partition(point_background_rows, "scores")
            background_weights = _concatenate_partition(
                point_background_rows, "physical_weights"
            )
        threshold = float(validation["points"][sample.point_id]["threshold"])
        signal = signal_rows[sample.sample_id]
        signal_selected = signal["scores"] >= threshold
        background_selected = background_scores >= threshold
        selected_background = background_weights[background_selected]
        selected_signal_unit = signal["unit_xsec_weights"][signal_selected]
        signal_unit_yield = float(np.sum(selected_signal_unit))
        background_yield = float(np.sum(selected_background))
        fold_s95 = (
            exact_cls_signal_upper_limit(background_yield)
            if background_yield >= 0.0
            else math.inf
        )
        point_rows[sample.point_id] = {
            "rotation": int(rotation),
            "sample_id": sample.sample_id,
            "c3": sample.c3,
            "d4": sample.d4,
            "threshold": threshold,
            "signal_unit_yield": signal_unit_yield,
            "signal_sumw2_unit": float(np.sum(selected_signal_unit ** 2)),
            "signal_raw_entries": int(np.sum(signal_selected)),
            "signal_feature_unit_yield": float(np.sum(signal["unit_xsec_weights"])),
            "xgboost_efficiency": (
                signal_unit_yield / float(np.sum(signal["unit_xsec_weights"]))
                if float(np.sum(signal["unit_xsec_weights"])) != 0.0
                else 0.0
            ),
            "background_yield": background_yield,
            "background_sumw2": float(np.sum(selected_background ** 2)),
            "background_raw_entries": int(np.sum(background_selected)),
            "background_effective_entries": _effective_entries(selected_background),
            "s95_exact_events": fold_s95,
            "cut_sigma95_fb": (
                fold_s95 / signal_unit_yield if signal_unit_yield > 0.0 else math.inf
            ),
        }
    return {
        "rotation": int(rotation),
        "points": point_rows,
        "signal_rows": signal_rows,
        "background_rows": background_rows or {},
        "parameterized": bool(parameterized),
    }


def _aggregate_cut_results(
    grid_samples: Sequence[EventSample],
    rotations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    by_point = {sample.point_id: sample for sample in grid_samples}
    for point_id, sample in by_point.items():
        folds = [rotation["points"][point_id] for rotation in rotations]
        signal_yield = float(sum(row["signal_unit_yield"] for row in folds))
        signal_sumw2 = float(sum(row["signal_sumw2_unit"] for row in folds))
        background_yield = float(sum(row["background_yield"] for row in folds))
        background_sumw2 = float(sum(row["background_sumw2"] for row in folds))
        if signal_yield <= 0.0 or background_yield < 0.0:
            sigma95 = math.inf
            s95 = math.inf
        else:
            s95 = exact_cls_signal_upper_limit(background_yield)
            sigma95 = s95 / signal_yield
        cut_sigma95_down = (
            exact_cls_signal_upper_limit(background_yield * 0.25) / signal_yield
            if signal_yield > 0.0 and background_yield >= 0.0
            else math.inf
        )
        cut_sigma95_up = (
            exact_cls_signal_upper_limit(background_yield * 4.0) / signal_yield
            if signal_yield > 0.0 and background_yield >= 0.0
            else math.inf
        )
        total_feature_unit_yield = float(np.sum(sample.unit_xsec_weights))
        output.append(
            {
                "point_id": point_id,
                "c3": sample.c3,
                "d4": sample.d4,
                "xsec_fb": sample.xsec_fb,
                "feature_tree_efficiency": float(np.sum(sample.raw_weights) / sample.normalisation_weight),
                "xgboost_efficiency": (
                    signal_yield / total_feature_unit_yield if total_feature_unit_yield != 0.0 else 0.0
                ),
                "selected_signal_yield_per_fb": signal_yield,
                "selected_signal_staterror_per_fb": math.sqrt(signal_sumw2),
                "background_yield": background_yield,
                "background_staterror": math.sqrt(background_sumw2),
                "background_effective_entries": (
                    background_yield * background_yield / background_sumw2
                    if background_sumw2 > 0.0
                    else 0.0
                ),
                "background_raw_entries": int(sum(row["background_raw_entries"] for row in folds)),
                "s95_exact_events": s95,
                "cut_sigma95_fb": sigma95,
                "cut_sigma95_background_x0p25_fb": cut_sigma95_down,
                "cut_sigma95_background_x4_fb": cut_sigma95_up,
                "excluded_cut": bool(sample.xsec_fb >= sigma95),
                "threshold_mean": float(np.mean([row["threshold"] for row in folds])),
                "threshold_std": float(np.std([row["threshold"] for row in folds])),
                "fold_signal_yield_std": float(
                    np.std([row["signal_unit_yield"] for row in folds])
                ),
                "fold_background_yield_std": float(
                    np.std([row["background_yield"] for row in folds])
                ),
                "fold_cut_sigma95_std": float(
                    np.std([row["cut_sigma95_fb"] for row in folds])
                ),
                "folds": folds,
            }
        )
    return sorted(output, key=lambda row: (float(row["c3"]), float(row["d4"])))


def _evaluate_postfit_signal_rotation(
    model: Any,
    validation: Mapping[str, Any],
    component_samples: Sequence[EventSample],
    *,
    rotation: int,
    n_folds: int,
    profile_indices: np.ndarray,
    parameterized: bool = False,
) -> dict[str, Any]:
    """Score a signal component after the model and thresholds are fixed."""

    signal_rows = _score_partition(
        model,
        component_samples,
        rotation=rotation,
        split="test",
        n_folds=n_folds,
        profile_indices=profile_indices,
        scale_validation_to_full=False,
        parameterized=parameterized,
    )
    point_rows: dict[str, dict[str, Any]] = {}
    for sample in component_samples:
        threshold = float(validation["points"][sample.point_id]["threshold"])
        signal = signal_rows[sample.sample_id]
        selected = signal["scores"] >= threshold
        selected_unit = signal["unit_xsec_weights"][selected]
        selected_physical = signal["physical_weights"][selected]
        feature_unit_yield = float(np.sum(signal["unit_xsec_weights"]))
        feature_physical_yield = float(np.sum(signal["physical_weights"]))
        point_rows[sample.point_id] = {
            "rotation": int(rotation),
            "sample_id": sample.sample_id,
            "c3": sample.c3,
            "d4": sample.d4,
            "threshold": threshold,
            "signal_unit_yield": float(np.sum(selected_unit)),
            "signal_sumw2_unit": float(np.sum(selected_unit ** 2)),
            "signal_physical_yield": float(np.sum(selected_physical)),
            "signal_sumw2_physical": float(np.sum(selected_physical ** 2)),
            "signal_raw_entries": int(np.sum(selected)),
            "signal_feature_unit_yield": feature_unit_yield,
            "signal_feature_physical_yield": feature_physical_yield,
            "xgboost_efficiency": (
                float(np.sum(selected_unit)) / feature_unit_yield
                if feature_unit_yield != 0.0
                else 0.0
            ),
        }
    return {
        "rotation": int(rotation),
        "points": point_rows,
        "signal_rows": signal_rows,
        "role": "postfit-signal-only",
        "parameterized": bool(parameterized),
    }


def _coupling_holdout_assignments(
    grid_samples: Sequence[EventSample],
    *,
    n_folds: int = 5,
    seed: int = BASE_SEED,
) -> dict[str, int]:
    """Assign every coupling point once to a balanced deterministic holdout."""

    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError("coupling holdout requires at least two folds")
    point_ids = [str(sample.point_id) for sample in grid_samples]
    if not point_ids or any(sample.point_id is None for sample in grid_samples):
        raise ValueError("coupling holdout requires named c3/d4 signal points")
    if len(set(point_ids)) != len(point_ids):
        raise ValueError("coupling holdout signal points must be unique")

    ordered = sorted(
        point_ids,
        key=lambda point_id: hashlib.sha256(
            f"{int(seed)}\0{point_id}".encode("utf-8")
        ).hexdigest(),
    )
    return {
        point_id: index % n_folds
        for index, point_id in enumerate(ordered)
    }


def _parameterized_coupling_holdout_diagnostic(
    sm_samples: Sequence[EventSample],
    grid_samples: Sequence[EventSample],
    background_samples: Sequence[EventSample],
    reference_records: Sequence[Mapping[str, Any]],
    *,
    observable_set: str,
    profile: str,
    n_folds: int,
    seed: int,
    source_commit: str,
    progress: StudyProgress | None = None,
) -> dict[str, Any]:
    """Test parameter interpolation with each coupling coordinate held out once.

    One balanced, hash-defined subset of coupling points is removed from signal
    training in each fold.  The same event-level rotation is then used to choose
    thresholds on validation events and evaluate disjoint test events for those
    unseen coordinates.  No post-fit signal component enters this diagnostic.
    """

    n_folds = int(n_folds)
    if len(grid_samples) < 3:
        raise ValueError("coupling holdout requires at least three signal points")
    assignments = _coupling_holdout_assignments(
        grid_samples,
        n_folds=n_folds,
        seed=seed,
    )
    records_by_rotation = {
        int(record["rotation"]): record for record in reference_records
    }
    if set(records_by_rotation) != set(range(n_folds)):
        raise ValueError(
            "coupling holdout requires one reference event-crossfit record per fold"
        )

    profile_indices = _profile_indices(observable_set, profile)
    rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold in range(n_folds):
        heldout = [
            sample
            for sample in grid_samples
            if assignments[str(sample.point_id)] == fold
        ]
        if not heldout:
            continue
        training = [
            sample
            for sample in grid_samples
            if assignments[str(sample.point_id)] != fold
        ]
        if not training:
            raise ValueError(
                f"coupling holdout fold {fold} has no remaining training points"
            )
        if progress is not None:
            progress.emit(
                "coupling-holdout",
                "Training parameterized coupling-point holdout fold",
                fold=fold + 1,
                total_folds=n_folds,
                training_points=len(training),
                heldout_points=len(heldout),
            )

        X, y, weights = _training_arrays(
            sm_samples,
            training,
            background_samples,
            strategy="parameterized-crossfit-v1",
            profile_indices=profile_indices,
            rotation=fold,
            n_folds=n_folds,
        )
        model, model_metadata, _ = _train_model(
            X,
            y,
            weights,
            params=FIXED_XGBOOST_PARAMS,
            seed=int(seed) + fold,
            observable_set=observable_set,
            profile=profile,
            strategy="parameterized-crossfit-v1",
            rotation=fold,
            source_commit=source_commit,
        )
        validation = _validation_limits(
            model,
            heldout,
            background_samples,
            rotation=fold,
            n_folds=n_folds,
            profile_indices=profile_indices,
            parameterized=True,
        )
        test = _evaluate_test_rotation(
            model,
            validation,
            heldout,
            background_samples,
            rotation=fold,
            n_folds=n_folds,
            profile_indices=profile_indices,
            parameterized=True,
        )
        reference_points = records_by_rotation[fold]["test"]["points"]
        for sample in heldout:
            point_id = str(sample.point_id)
            holdout_validation = validation["points"][point_id]
            holdout_test = test["points"][point_id]
            if point_id not in reference_points:
                raise ValueError(
                    f"coupling holdout reference is missing point {point_id}"
                )
            reference = reference_points[point_id]
            holdout_limit = float(holdout_test["cut_sigma95_fb"])
            reference_limit = float(reference["cut_sigma95_fb"])
            ratio = (
                holdout_limit / reference_limit
                if (
                    math.isfinite(holdout_limit)
                    and math.isfinite(reference_limit)
                    and reference_limit > 0.0
                )
                else None
            )
            rows.append(
                {
                    "point_id": point_id,
                    "c3": float(sample.c3),
                    "d4": float(sample.d4),
                    "coupling_holdout_fold": fold,
                    "training_point_count": len(training),
                    "heldout_point_count": len(heldout),
                    "holdout_validation_threshold": float(
                        holdout_validation["threshold"]
                    ),
                    "holdout_validation_cut_sigma95_fb": float(
                        holdout_validation["sigma95_fb"]
                    ),
                    "holdout_test_cut_sigma95_fb": holdout_limit,
                    "event_crossfit_threshold": float(reference["threshold"]),
                    "event_crossfit_test_cut_sigma95_fb": reference_limit,
                    "holdout_to_event_crossfit_ratio": ratio,
                    "classifier_training_role": (
                        "entire-coupling-coordinate-held-out"
                    ),
                    "postfit_hhhbb_included": False,
                }
            )
        fold_summaries.append(
            {
                "fold": fold,
                "training_point_count": len(training),
                "heldout_point_count": len(heldout),
                "training_rows": int(X.shape[0]),
                "validation_objective": float(validation["objective"]),
                "xgboost_split_nodes": model_metadata.get(
                    "xgboost_split_nodes"
                ),
            }
        )

    rows.sort(key=lambda row: (float(row["c3"]), float(row["d4"])))
    if len(rows) != len(grid_samples):
        raise RuntimeError(
            "coupling holdout did not evaluate every c3/d4 point exactly once"
        )
    ratios = np.asarray(
        [
            float(row["holdout_to_event_crossfit_ratio"])
            for row in rows
            if row["holdout_to_event_crossfit_ratio"] is not None
        ],
        dtype=float,
    )
    assignment_payload = json.dumps(
        sorted(assignments.items()),
        separators=(",", ":"),
    )
    summary = {
        "status": "complete",
        "version": COUPLING_HOLDOUT_VERSION,
        "point_count": len(rows),
        "finite_ratio_count": int(len(ratios)),
        "fold_count": n_folds,
        "assignment_sha256": hashlib.sha256(
            assignment_payload.encode("utf-8")
        ).hexdigest(),
        "assignment_rule": (
            "sha256(seed,point_id) ordering followed by balanced round-robin folds"
        ),
        "event_split_rule": (
            "holdout fold k uses event rotation k: train on three event folds, "
            "threshold on validation=(k+1)%5, evaluate test=k"
        ),
        "heldout_coordinates_absent_from_signal_training": True,
        "heldout_coordinates_absent_from_background_parameter_replicas": True,
        "postfit_hhhbb_included": False,
        "median_holdout_to_event_crossfit_ratio": (
            float(np.median(ratios)) if len(ratios) else None
        ),
        "q90_holdout_to_event_crossfit_ratio": (
            float(np.quantile(ratios, 0.90)) if len(ratios) else None
        ),
        "points_favoring_holdout_model": int(np.sum(ratios < 1.0)),
        "folds": fold_summaries,
    }
    return {"summary": summary, "rows": rows}


def _add_postfit_hhhbb_cut_contribution(
    aggregate: list[dict[str, Any]],
    grid_samples: Sequence[EventSample],
    hhhbb_samples: Sequence[EventSample],
    rotations: Sequence[Mapping[str, Any]],
) -> None:
    """Add hhhbb to the nominal signal only after cut optimization.

    The classifier, validation thresholds, and background yields are unchanged.
    Limits remain expressed as an equivalent hhhh cross section by scaling the
    fixed hhhh and hhhbb theory predictions with one common signal strength.
    """

    if not hhhbb_samples:
        return
    grid_by_point = {sample.point_id: sample for sample in grid_samples}
    hhhbb_by_point = {sample.point_id: sample for sample in hhhbb_samples}
    if set(grid_by_point) != set(hhhbb_by_point):
        raise ValueError(
            "Post-fit hhhbb signal coordinates must exactly match the c3/d4 grid"
        )
    if len(rotations) == 0:
        raise ValueError("Post-fit hhhbb scoring requires cross-fit rotations")

    for row in aggregate:
        point_id = str(row["point_id"])
        hhhh_sample = grid_by_point[point_id]
        hhhbb_sample = hhhbb_by_point[point_id]
        component_folds = [
            rotation["points"][point_id] for rotation in rotations
        ]
        if len(component_folds) != len(row["folds"]):
            raise ValueError(
                f"{point_id}: hhhh and hhhbb fold counts do not match"
            )

        hhhh_xsec_fb = float(hhhh_sample.xsec_fb)
        hhhbb_xsec_fb = float(hhhbb_sample.xsec_fb)
        if hhhh_xsec_fb <= 0.0 or hhhbb_xsec_fb < 0.0:
            raise ValueError(
                f"{point_id}: post-fit signal cross sections must be nonnegative "
                "with a positive hhhh reference"
            )

        hhhh_unit_yield = float(row["selected_signal_yield_per_fb"])
        hhhh_staterror_unit = float(row["selected_signal_staterror_per_fb"])
        hhhh_nominal_yield = hhhh_xsec_fb * hhhh_unit_yield
        hhhh_nominal_sumw2 = (hhhh_xsec_fb * hhhh_staterror_unit) ** 2
        hhhbb_unit_yield = float(
            sum(fold["signal_unit_yield"] for fold in component_folds)
        )
        hhhbb_sumw2_unit = float(
            sum(fold["signal_sumw2_unit"] for fold in component_folds)
        )
        hhhbb_nominal_yield = float(
            sum(fold["signal_physical_yield"] for fold in component_folds)
        )
        hhhbb_nominal_sumw2 = float(
            sum(fold["signal_sumw2_physical"] for fold in component_folds)
        )
        combined_nominal_yield = hhhh_nominal_yield + hhhbb_nominal_yield
        combined_nominal_sumw2 = hhhh_nominal_sumw2 + hhhbb_nominal_sumw2
        equivalent_unit_yield = combined_nominal_yield / hhhh_xsec_fb
        equivalent_sumw2_unit = combined_nominal_sumw2 / (hhhh_xsec_fb ** 2)

        row.update(
            {
                "signal_components": "hhhh,hhhbb",
                "limit_parameter": "common-signal-strength",
                "limit_cross_section_basis": "equivalent-hhhh-fb",
                "hhhh_xsec_fb": hhhh_xsec_fb,
                "hhhh_selected_signal_yield_per_fb": hhhh_unit_yield,
                "hhhh_selected_signal_staterror_per_fb": hhhh_staterror_unit,
                "hhhh_nominal_selected_signal_yield": hhhh_nominal_yield,
                "hhhbb_file": str(hhhbb_sample.path),
                "hhhbb_xsec_fb": hhhbb_xsec_fb,
                "hhhbb_rate_factor": float(hhhbb_sample.rate_factor),
                "hhhbb_generated_events": hhhbb_sample.generated_events,
                "hhhbb_normalisation_weight": float(
                    hhhbb_sample.normalisation_weight
                ),
                "hhhbb_feature_tree_efficiency": float(
                    np.sum(hhhbb_sample.raw_weights)
                    / hhhbb_sample.normalisation_weight
                ),
                "hhhbb_xgboost_efficiency": (
                    hhhbb_unit_yield / float(np.sum(hhhbb_sample.unit_xsec_weights))
                    if float(np.sum(hhhbb_sample.unit_xsec_weights)) != 0.0
                    else 0.0
                ),
                "hhhbb_selected_signal_yield_per_fb": hhhbb_unit_yield,
                "hhhbb_selected_signal_staterror_per_fb": math.sqrt(
                    hhhbb_sumw2_unit
                ),
                "hhhbb_nominal_selected_signal_yield": hhhbb_nominal_yield,
                "hhhbb_nominal_selected_signal_staterror": math.sqrt(
                    hhhbb_nominal_sumw2
                ),
                "hhhbb_selected_raw_entries": int(
                    sum(fold["signal_raw_entries"] for fold in component_folds)
                ),
                "combined_nominal_selected_signal_yield": combined_nominal_yield,
                "combined_nominal_selected_signal_staterror": math.sqrt(
                    combined_nominal_sumw2
                ),
                "selected_signal_yield_per_fb": equivalent_unit_yield,
                "selected_signal_staterror_per_fb": math.sqrt(
                    equivalent_sumw2_unit
                ),
            }
        )

        for hhhh_fold, hhhbb_fold in zip(row["folds"], component_folds):
            hhhh_fold_unit = float(hhhh_fold["signal_unit_yield"])
            hhhh_fold_sumw2 = float(hhhh_fold["signal_sumw2_unit"])
            hhhh_fold_nominal = hhhh_xsec_fb * hhhh_fold_unit
            combined_fold_nominal = (
                hhhh_fold_nominal + float(hhhbb_fold["signal_physical_yield"])
            )
            combined_fold_sumw2 = (
                (hhhh_xsec_fb ** 2) * hhhh_fold_sumw2
                + float(hhhbb_fold["signal_sumw2_physical"])
            )
            combined_fold_unit = combined_fold_nominal / hhhh_xsec_fb
            hhhh_fold.update(
                {
                    "hhhh_signal_unit_yield": hhhh_fold_unit,
                    "hhhh_signal_sumw2_unit": hhhh_fold_sumw2,
                    "hhhbb_signal_unit_yield": float(
                        hhhbb_fold["signal_unit_yield"]
                    ),
                    "hhhbb_signal_sumw2_unit": float(
                        hhhbb_fold["signal_sumw2_unit"]
                    ),
                    "hhhbb_signal_physical_yield": float(
                        hhhbb_fold["signal_physical_yield"]
                    ),
                    "hhhbb_signal_sumw2_physical": float(
                        hhhbb_fold["signal_sumw2_physical"]
                    ),
                    "hhhbb_signal_raw_entries": int(
                        hhhbb_fold["signal_raw_entries"]
                    ),
                    "combined_signal_nominal_yield": combined_fold_nominal,
                    "signal_unit_yield": combined_fold_unit,
                    "signal_sumw2_unit": (
                        combined_fold_sumw2 / (hhhh_xsec_fb ** 2)
                    ),
                    "cut_sigma95_fb": (
                        float(hhhh_fold["s95_exact_events"])
                        / combined_fold_unit
                        if combined_fold_unit > 0.0
                        else math.inf
                    ),
                }
            )

        background_yield = float(row["background_yield"])
        s95 = exact_cls_signal_upper_limit(background_yield)
        s95_down = exact_cls_signal_upper_limit(background_yield * 0.25)
        s95_up = exact_cls_signal_upper_limit(background_yield * 4.0)
        row.update(
            {
                "s95_exact_events": s95,
                "cut_signal_strength95": (
                    s95 / combined_nominal_yield
                    if combined_nominal_yield > 0.0
                    else math.inf
                ),
                "cut_signal_strength95_background_x0p25": (
                    s95_down / combined_nominal_yield
                    if combined_nominal_yield > 0.0
                    else math.inf
                ),
                "cut_signal_strength95_background_x4": (
                    s95_up / combined_nominal_yield
                    if combined_nominal_yield > 0.0
                    else math.inf
                ),
                "cut_sigma95_fb": (
                    s95 / equivalent_unit_yield
                    if equivalent_unit_yield > 0.0
                    else math.inf
                ),
                "cut_sigma95_background_x0p25_fb": (
                    s95_down / equivalent_unit_yield
                    if equivalent_unit_yield > 0.0
                    else math.inf
                ),
                "cut_sigma95_background_x4_fb": (
                    s95_up / equivalent_unit_yield
                    if equivalent_unit_yield > 0.0
                    else math.inf
                ),
                "excluded_cut": bool(
                    combined_nominal_yield >= s95
                    if combined_nominal_yield > 0.0
                    else False
                ),
                "fold_signal_yield_std": float(
                    np.std(
                        [fold["signal_unit_yield"] for fold in row["folds"]]
                    )
                ),
                "fold_cut_sigma95_std": float(
                    np.std([fold["cut_sigma95_fb"] for fold in row["folds"]])
                ),
            }
        )


def _sm_background_cutflow_rows(
    background_samples: Sequence[EventSample],
    records: Sequence[Mapping[str, Any]],
    *,
    luminosity: float,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Aggregate per-background SM rates from disjoint cross-fit test folds.

    The feature-tree input rate comes directly from the full physical event
    weights.  The XGBoost rate is a union of held-out test folds, with the
    validation-selected SM threshold from the corresponding rotation.
    """

    luminosity = float(luminosity)
    if not math.isfinite(luminosity) or luminosity <= 0.0:
        raise ValueError("luminosity must be finite and positive")
    if not records:
        raise ValueError("SM background cutflow requires cross-fit records")

    thresholds: list[float] = []
    point_ids: list[str] = []
    for record in records:
        point_rows = record["test"]["points"]
        matches = [
            (str(point_id), point_row)
            for point_id, point_row in point_rows.items()
            if abs(float(point_row["c3"])) < 1.0e-12
            and abs(float(point_row["d4"])) < 1.0e-12
        ]
        if len(matches) != 1:
            raise ValueError(
                "SM background cutflow requires exactly one (c3,d4)=(0,0) "
                f"test point per fold; found {len(matches)}"
            )
        point_id, point_row = matches[0]
        threshold = float(point_row["threshold"])
        if not math.isfinite(threshold):
            raise ValueError("SM background cutflow encountered a non-finite threshold")
        point_ids.append(point_id)
        thresholds.append(threshold)

    rows: list[dict[str, Any]] = []
    for sample in background_samples:
        selected_weights: list[np.ndarray] = []
        held_out_indices: list[np.ndarray] = []
        selected_entries = 0
        for record, threshold, point_id in zip(records, thresholds, point_ids):
            if record["test"].get("parameterized"):
                cache = record.setdefault("_test_parameter_cache", {})
                if point_id not in cache:
                    cache[point_id] = _score_partition(
                        _shape_model_for_record(record),
                        record["_background_samples"],
                        rotation=int(record["rotation"]),
                        split="test",
                        n_folds=int(record["_n_folds"]),
                        profile_indices=record["_profile_indices"],
                        scale_validation_to_full=False,
                        parameterized=True,
                        parameter_point=(0.0, 0.0),
                    )
                background_rows = cache[point_id]
            else:
                background_rows = record["test"]["background_rows"]
            if sample.sample_id not in background_rows:
                raise ValueError(
                    f"SM background cutflow is missing {sample.sample_id!r} in a test fold"
                )
            fold_row = background_rows[sample.sample_id]
            scores = np.asarray(fold_row["scores"], dtype=float)
            weights = np.asarray(fold_row["physical_weights"], dtype=float)
            event_indices = np.asarray(fold_row["event_indices"], dtype=np.int64)
            if scores.shape != weights.shape or scores.shape != event_indices.shape:
                raise ValueError(
                    f"{sample.sample_id}: scores, weights, and event indices do not align"
                )
            selected = scores >= threshold
            selected_weights.append(weights[selected])
            held_out_indices.append(event_indices)
            selected_entries += int(np.sum(selected))

        all_indices = np.concatenate(held_out_indices)
        expected_indices = np.asarray(sample.event_indices, dtype=np.int64)
        if (
            all_indices.size != expected_indices.size
            or np.unique(all_indices).size != all_indices.size
            or not np.array_equal(np.sort(all_indices), np.sort(expected_indices))
        ):
            raise ValueError(
                f"{sample.sample_id}: held-out test folds do not form an exact, "
                "non-overlapping union of the feature-tree events"
            )

        selected = (
            np.concatenate(selected_weights)
            if selected_weights
            else np.asarray([], dtype=float)
        )
        input_events = float(np.sum(sample.physical_weights))
        xgboost_events = float(np.sum(selected))
        xgboost_error = float(math.sqrt(np.sum(selected * selected)))
        metadata = dict(sample.metadata or {})
        rows.append(
            {
                "sample_id": sample.sample_id,
                "sample_role": "background",
                "is_signal": False,
                "signal_component": None,
                "file": str(sample.path),
                "process_id": metadata.get("process_id", sample.sample_id),
                "description": metadata.get("description", sample.sample_id),
                "production_xsec_fb": float(sample.xsec_fb),
                "rate_factor": float(sample.rate_factor),
                "effective_inclusive_xsec_fb": (
                    float(sample.xsec_fb) * float(sample.rate_factor)
                ),
                "input_xsec_fb": input_events / luminosity,
                "input_events": input_events,
                "feature_tree_efficiency": (
                    input_events
                    / (
                        luminosity
                        * float(sample.xsec_fb)
                        * float(sample.rate_factor)
                    )
                    if float(sample.xsec_fb) * float(sample.rate_factor) != 0.0
                    else 0.0
                ),
                "xgboost_xsec_fb": xgboost_events / luminosity,
                "xgboost_events": xgboost_events,
                "xgboost_events_error": xgboost_error,
                "xgboost_xsec_error_fb": xgboost_error / luminosity,
                "xgboost_efficiency": (
                    xgboost_events / input_events if input_events != 0.0 else 0.0
                ),
                "entries": sample.entries,
                "selected_entries": selected_entries,
                "generated_events": sample.generated_events,
                "normalisation_weight": float(sample.normalisation_weight),
                "c3": sample.c3,
                "d4": sample.d4,
            }
        )

    return rows, thresholds


def _cut_signal_strength95(row: Mapping[str, Any]) -> float:
    """Return the common signal-strength limit for one cut-result row."""

    if row.get("cut_signal_strength95") is not None:
        value = float(row["cut_signal_strength95"])
    else:
        theory_xsec = float(row["xsec_fb"])
        value = (
            float(row["cut_sigma95_fb"]) / theory_xsec
            if theory_xsec > 0.0
            else math.inf
        )
    return value if math.isfinite(value) and value > 0.0 else math.inf


def _select_limit_representative_points(
    aggregate: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select eight grid points nearest mu95=1 in fixed geometric regions."""

    tolerance = 1.0e-12
    regions = (
        (
            "c3~0, d4<0",
            lambda c3, d4: abs(c3) < tolerance and d4 < -tolerance,
        ),
        (
            "c3~0, d4>0",
            lambda c3, d4: abs(c3) < tolerance and d4 > tolerance,
        ),
        (
            "d4~0, c3<0",
            lambda c3, d4: abs(d4) < tolerance and c3 < -tolerance,
        ),
        (
            "d4~0, c3>0",
            lambda c3, d4: abs(d4) < tolerance and c3 > tolerance,
        ),
        (
            "diagonal Q1",
            lambda c3, d4: c3 > tolerance and d4 > tolerance,
        ),
        (
            "diagonal Q2",
            lambda c3, d4: c3 < -tolerance and d4 > tolerance,
        ),
        (
            "diagonal Q3",
            lambda c3, d4: c3 < -tolerance and d4 < -tolerance,
        ),
        (
            "diagonal Q4",
            lambda c3, d4: c3 > tolerance and d4 < -tolerance,
        ),
    )
    selected: list[dict[str, Any]] = []
    for category, predicate in regions:
        candidates = []
        for row in aggregate:
            c3 = float(row["c3"])
            d4 = float(row["d4"])
            mu95 = _cut_signal_strength95(row)
            if predicate(c3, d4) and math.isfinite(mu95):
                candidates.append(
                    (
                        abs(math.log(mu95)),
                        c3,
                        d4,
                        row,
                        mu95,
                    )
                )
        if not candidates:
            raise ValueError(
                "Cannot select a 95% CL representative point for region "
                f"{category!r}"
            )
        distance, _c3, _d4, result, mu95 = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        selected.append(
            {
                "representative_category": category,
                "result": result,
                "cut_signal_strength95": mu95,
                "limit_proximity_log_mu95": float(distance),
            }
        )

    point_ids = [str(item["result"]["point_id"]) for item in selected]
    if len(set(point_ids)) != len(regions):
        raise ValueError("The 95% CL representative regions selected duplicate points")
    return selected


def _sm_signal_cutflow_rows(
    grid_samples: Sequence[EventSample],
    hhhbb_samples: Sequence[EventSample],
    aggregate: Sequence[Mapping[str, Any]],
    *,
    luminosity: float,
    include_limit_representatives: bool = False,
) -> list[dict[str, Any]]:
    """Build role-labelled SM and representative-point signal cutflow rows."""

    luminosity = float(luminosity)
    if not math.isfinite(luminosity) or luminosity <= 0.0:
        raise ValueError("luminosity must be finite and positive")

    grid_by_point = {str(sample.point_id): sample for sample in grid_samples}
    if len(grid_by_point) != len(grid_samples):
        raise ValueError("Signal cutflow grid contains duplicate coupling points")
    hhhbb_by_point = {str(sample.point_id): sample for sample in hhhbb_samples}
    if len(hhhbb_by_point) != len(hhhbb_samples):
        raise ValueError("Signal cutflow hhhbb grid contains duplicate coupling points")

    sm_results = [
        row
        for row in aggregate
        if abs(float(row["c3"])) < 1.0e-12
        and abs(float(row["d4"])) < 1.0e-12
    ]
    if len(sm_results) != 1:
        raise ValueError(
            "SM signal cutflow requires exactly one result at (c3,d4)=(0,0)"
        )
    sm_result = sm_results[0]
    selected_points = [
        {
            "representative_category": "SM reference",
            "result": sm_result,
            "cut_signal_strength95": _cut_signal_strength95(sm_result),
            "limit_proximity_log_mu95": None,
            "is_limit_representative": False,
        }
    ]
    if include_limit_representatives:
        selected_points.extend(
            {
                **item,
                "is_limit_representative": True,
            }
            for item in _select_limit_representative_points(aggregate)
        )

    def make_row(
        sample: EventSample,
        *,
        result: Mapping[str, Any],
        category: str,
        is_limit_representative: bool,
        component: str,
        process_id: str,
        description: str,
        selected_events: float,
        selected_error: float,
        selected_entries: int,
    ) -> dict[str, Any]:
        input_events = float(np.sum(sample.physical_weights))
        effective_inclusive_xsec = (
            float(sample.xsec_fb) * float(sample.rate_factor)
        )
        mu95 = _cut_signal_strength95(result)
        return {
            "sample_id": sample.sample_id,
            "sample_role": "signal",
            "is_signal": True,
            "signal_component": component,
            "point_id": sample.point_id,
            "point_class": (
                "limit-representative"
                if is_limit_representative
                else "standard-model-reference"
            ),
            "representative_category": category,
            "is_limit_representative": bool(is_limit_representative),
            "representative_selection": (
                "minimum-abs-log-mu95-in-region"
                if is_limit_representative
                else None
            ),
            "cut_signal_strength95": mu95,
            "theory_to_limit_ratio": (
                1.0 / mu95 if math.isfinite(mu95) and mu95 > 0.0 else 0.0
            ),
            "limit_proximity_log_mu95": (
                abs(math.log(mu95))
                if is_limit_representative
                and math.isfinite(mu95)
                and mu95 > 0.0
                else None
            ),
            "excluded_cut": bool(result.get("excluded_cut", False)),
            "file": str(sample.path),
            "process_id": process_id,
            "description": description,
            "production_xsec_fb": float(sample.xsec_fb),
            "rate_factor": float(sample.rate_factor),
            "effective_inclusive_xsec_fb": effective_inclusive_xsec,
            "input_xsec_fb": input_events / luminosity,
            "input_events": input_events,
            "xgboost_xsec_fb": selected_events / luminosity,
            "xgboost_events": selected_events,
            "xgboost_events_error": selected_error,
            "xgboost_xsec_error_fb": selected_error / luminosity,
            "xgboost_efficiency": (
                selected_events / input_events if input_events != 0.0 else 0.0
            ),
            "feature_tree_efficiency": (
                input_events / (luminosity * effective_inclusive_xsec)
                if effective_inclusive_xsec != 0.0
                else 0.0
            ),
            "entries": sample.entries,
            "selected_entries": int(selected_entries),
            "generated_events": sample.generated_events,
            "normalisation_weight": float(sample.normalisation_weight),
            "c3": sample.c3,
            "d4": sample.d4,
        }

    rows: list[dict[str, Any]] = []
    for point in selected_points:
        result = point["result"]
        point_id = str(result["point_id"])
        if point_id not in grid_by_point:
            raise ValueError(
                f"Signal cutflow result {point_id!r} has no matching hhhh sample"
            )
        hhhh_sample = grid_by_point[point_id]
        is_representative = bool(point["is_limit_representative"])
        category = str(point["representative_category"])
        hhhh_xsec_fb = float(hhhh_sample.xsec_fb)
        hhhh_unit_yield = float(
            result["hhhh_selected_signal_yield_per_fb"]
            if "hhhh_selected_signal_yield_per_fb" in result
            else result["selected_signal_yield_per_fb"]
        )
        hhhh_unit_error = float(
            result["hhhh_selected_signal_staterror_per_fb"]
            if "hhhh_selected_signal_staterror_per_fb" in result
            else result["selected_signal_staterror_per_fb"]
        )
        hhhh_selected_entries = sum(
            int(
                fold.get(
                    "hhhh_signal_raw_entries",
                    fold.get("signal_raw_entries", 0),
                )
            )
            for fold in result["folds"]
        )
        rows.append(
            make_row(
                hhhh_sample,
                result=result,
                category=category,
                is_limit_representative=is_representative,
                component="hhhh",
                process_id="hhhh" if is_representative else "sm_hhhh",
                description=(
                    f"hhhh [{category}]"
                    if is_representative
                    else "SM gg -> hhhh -> 8b"
                ),
                selected_events=hhhh_xsec_fb * hhhh_unit_yield,
                selected_error=hhhh_xsec_fb * hhhh_unit_error,
                selected_entries=hhhh_selected_entries,
            )
        )

        if hhhbb_samples:
            if point_id not in hhhbb_by_point:
                raise ValueError(
                    f"Signal cutflow result {point_id!r} has no matching hhhbb sample"
                )
            required = (
                "hhhbb_nominal_selected_signal_yield",
                "hhhbb_nominal_selected_signal_staterror",
                "hhhbb_selected_raw_entries",
            )
            missing = [key for key in required if key not in result]
            if missing:
                raise ValueError(
                    f"{point_id}: post-fit hhhbb cutflow fields are missing: "
                    + ", ".join(missing)
                )
            rows.append(
                make_row(
                    hhhbb_by_point[point_id],
                    result=result,
                    category=category,
                    is_limit_representative=is_representative,
                    component="hhhbb",
                    process_id="hhhbb" if is_representative else "sm_hhhbb",
                    description=(
                        f"hhh+bb [{category}]"
                        if is_representative
                        else (
                            "SM gg -> hhhg, forced g -> b bbar "
                            "(hhh + bb -> 8b)"
                        )
                    ),
                    selected_events=float(
                        result["hhhbb_nominal_selected_signal_yield"]
                    ),
                    selected_error=float(
                        result["hhhbb_nominal_selected_signal_staterror"]
                    ),
                    selected_entries=int(result["hhhbb_selected_raw_entries"]),
                )
            )
    return rows


def _cutflow_role_totals(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Return additive cutflow totals for one non-overlapping sample role."""

    rows = list(rows)
    return {
        "samples": len(rows),
        "entries": int(sum(int(row.get("entries", 0)) for row in rows)),
        "selected_entries": int(
            sum(int(row.get("selected_entries", 0)) for row in rows)
        ),
        "input_xsec_fb": float(
            sum(float(row.get("input_xsec_fb", 0.0)) for row in rows)
        ),
        "input_events": float(
            sum(float(row.get("input_events", 0.0)) for row in rows)
        ),
        "xgboost_xsec_fb": float(
            sum(float(row.get("xgboost_xsec_fb", 0.0)) for row in rows)
        ),
        "xgboost_events": float(
            sum(float(row.get("xgboost_events", 0.0)) for row in rows)
        ),
        "xgboost_events_error": float(
            math.sqrt(
                sum(
                    float(row.get("xgboost_events_error", 0.0)) ** 2
                    for row in rows
                )
            )
        ),
    }


def _cutflow_signal_totals_by_point(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add signal components only within, never across, coupling hypotheses."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        point_id = str(row["point_id"])
        grouped.setdefault(point_id, []).append(row)

    totals = []
    for point_rows in grouped.values():
        first = point_rows[0]
        total = _cutflow_role_totals(point_rows)
        total.update(
            {
                "point_id": first["point_id"],
                "c3": first["c3"],
                "d4": first["d4"],
                "point_class": first["point_class"],
                "representative_category": first["representative_category"],
                "is_limit_representative": first["is_limit_representative"],
                "cut_signal_strength95": first["cut_signal_strength95"],
                "theory_to_limit_ratio": first["theory_to_limit_ratio"],
                "excluded_cut": first["excluded_cut"],
                "signal_components": [
                    str(row["signal_component"]) for row in point_rows
                ],
            }
        )
        totals.append(total)
    return totals


def _aggregate_validation_crossfit(
    grid_samples: Sequence[EventSample],
    rotations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    by_point = {sample.point_id: sample for sample in grid_samples}
    point_totals = {
        point_id: {"signal": 0.0, "background": 0.0}
        for point_id in by_point
    }
    rotation_limits = []
    for rotation in rotations:
        fold_limits = []
        for point_id, sample in by_point.items():
            selected = rotation["validation"]["points"][point_id]
            threshold = float(selected["threshold"])
            arrays = _validation_fold_arrays(rotation, sample)
            signal_mask = arrays["signal_scores"] >= threshold
            background_mask = arrays["background_scores"] >= threshold
            signal_yield = float(np.sum(arrays["signal_weights"][signal_mask]))
            background_yield = float(
                np.sum(arrays["background_weights"][background_mask])
            )
            point_totals[point_id]["signal"] += signal_yield
            point_totals[point_id]["background"] += background_yield
            fold_limits.append(float(selected["sigma95_fb"]))
            if rotation["validation"].get("parameterized"):
                rotation.get("_validation_parameter_cache", {}).pop(point_id, None)
        rotation_limits.append(fold_limits)

    rows = []
    for point_id, sample in by_point.items():
        signal = point_totals[point_id]["signal"]
        background = point_totals[point_id]["background"]
        sigma95 = math.inf if signal <= 0.0 or background < 0.0 else exact_cls_signal_upper_limit(background) / signal
        rows.append(
            {
                "point_id": point_id,
                "c3": sample.c3,
                "d4": sample.d4,
                "validation_signal_yield_per_fb": signal,
                "validation_background_yield": background,
                "validation_cut_sigma95_fb": sigma95,
            }
        )
    rows.sort(key=lambda row: (float(row["c3"]), float(row["d4"])))
    # Rotation columns must follow the same sorted point order used in rows.
    original_order = list(by_point)
    order = [original_order.index(row["point_id"]) for row in rows]
    return rows, np.asarray(rotation_limits, dtype=float)[:, order]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON via an atomic same-directory replacement.

    Checkpoints and progress files must remain valid if the campaign is
    interrupted while they are being updated.  Historical output writers keep
    their original behaviour; this helper is reserved for restart-critical
    state.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "--"
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


@dataclass
class StudyProgress:
    """Parent-owned, flushed terminal and JSON progress reporter."""

    output_dir: Path
    interval_seconds: float
    started_monotonic: float = field(default_factory=time.monotonic)
    status: str = "running"
    phase: str = "startup"
    last_heartbeat_monotonic: float = field(default_factory=time.monotonic)

    @property
    def path(self) -> Path:
        return self.output_dir / "study_progress.json"

    def emit(
        self,
        phase: str,
        message: str,
        *,
        status: str | None = None,
        eta_seconds: float | None = None,
        **current: Any,
    ) -> None:
        self.phase = str(phase)
        if status is not None:
            self.status = str(status)
        now = time.monotonic()
        self.last_heartbeat_monotonic = now
        elapsed = now - self.started_monotonic
        payload = {
            "version": 1,
            "status": self.status,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - elapsed)),
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": float(elapsed),
            "phase": self.phase,
            "message": str(message),
            "current": current,
            "eta_seconds": None if eta_seconds is None else float(eta_seconds),
        }
        _write_json_atomic(self.path, payload)
        details = ", ".join(
            f"{key}={value}" for key, value in current.items() if value is not None
        )
        suffix = f" ({details})" if details else ""
        eta = "" if eta_seconds is None else f", ETA {_format_duration(eta_seconds)}"
        print(
            f"[v2 {self.phase}] {message}{suffix}; elapsed {_format_duration(elapsed)}{eta}",
            flush=True,
        )

    def heartbeat_due(self) -> bool:
        return time.monotonic() - self.last_heartbeat_monotonic >= self.interval_seconds


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key in fields:
                value = _json_safe(row.get(key))
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True, separators=(",", ":"))
                encoded[key] = value
            writer.writerow(encoded)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for package in ("xgboost", "pyhf", "optuna", "scikit-learn", "ROOT"):
        try:
            if package == "ROOT":
                import ROOT

                versions[package] = str(ROOT.gROOT.GetVersion())
            else:
                from importlib.metadata import version

                versions[package] = version(package)
        except Exception:
            versions[package] = "unavailable"
    return versions


def _sample_manifest(sample: EventSample, input_hash: str) -> dict[str, Any]:
    return {
        "path": str(sample.path),
        "sha256": input_hash,
        "sample_id": sample.sample_id,
        "kind": sample.kind,
        "entries": sample.entries,
        "sum_raw_weight": float(np.sum(sample.raw_weights)),
        "sum_physical_weight": float(np.sum(sample.physical_weights)),
        "xsec_fb": sample.xsec_fb,
        "rate_factor": sample.rate_factor,
        "normalisation_weight": sample.normalisation_weight,
        "normalisation_source": sample.normalisation_source,
        "generated_events": sample.generated_events,
        "c3": sample.c3,
        "d4": sample.d4,
        "fold_counts": np.bincount(sample.folds).tolist(),
        "metadata": sample.metadata or {},
    }


def _normalization_metadata(
    luminosity: float,
    sm_samples: Sequence[EventSample],
    grid_samples: Sequence[EventSample],
    background_samples: Sequence[EventSample],
    postfit_signal_samples: Sequence[EventSample] = (),
) -> dict[str, Any]:
    return {
        "luminosity_fb_inverse": float(luminosity),
        "formula": "L*xsec_fb*rate_factor*event_weight/total_weight_in",
        "sources": [
            {
                "sample_id": sample.sample_id,
                "kind": sample.kind,
                "xsec_fb": sample.xsec_fb,
                "rate_factor": sample.rate_factor,
                "normalisation_weight": sample.normalisation_weight,
                "normalisation_source": sample.normalisation_source,
                "generated_events": sample.generated_events,
            }
            for sample in [
                *sm_samples,
                *grid_samples,
                *background_samples,
                *postfit_signal_samples,
            ]
        ],
    }


def write_v2_input_observable_report(
    sm_samples: Sequence[EventSample],
    background_samples: Sequence[EventSample],
    output_dir: str | Path,
    *,
    observable_set: str,
    feature_profile: str,
    luminosity: float,
) -> dict[str, Any]:
    """Write v2 input-shape and legacy-style stacked cross-section galleries."""

    if not sm_samples:
        raise ValueError("The input-observable report requires an SM signal sample")
    if not background_samples:
        raise ValueError("The input-observable report requires background samples")
    luminosity = _finite_float(luminosity, "luminosity")
    if luminosity <= 0.0:
        raise ValueError("luminosity must be positive")

    report_dir = Path(output_dir) / "sample_report"
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    contract = get_feature_contract(observable_set, feature_profile)
    feature_names = list(contract.feature_names)
    feature_indices = list(contract.feature_indices)

    report_samples = []
    sample_rows = []
    for index, event_sample in enumerate([*sm_samples, *background_samples]):
        is_signal = event_sample.kind == "sm_signal"
        metadata = dict(event_sample.metadata or {})
        label = sample_latex_label(metadata, is_signal=is_signal)
        effective_xsec_fb = float(np.sum(event_sample.physical_weights)) / luminosity
        report_samples.append(
            {
                "sample_id": event_sample.sample_id,
                "label": label,
                "file": str(event_sample.path),
                "features": event_sample.features,
                "weights": event_sample.physical_weights,
                "input_xsec_fb": effective_xsec_fb,
                "normalisation_weight_sum": float(
                    np.sum(event_sample.physical_weights)
                ),
                "is_signal": is_signal,
                "process_id": metadata.get("process_id"),
                "metadata": metadata,
                "style": sample_style(index),
            }
        )
        sample_rows.append(
            {
                "sample_id": event_sample.sample_id,
                "kind": event_sample.kind,
                "label": label,
                "file": str(event_sample.path),
                "entries": event_sample.entries,
                "input_xsec_fb": effective_xsec_fb,
                "process_id": metadata.get("process_id", ""),
            }
        )

    plot_rows = []
    plot_metadata = []
    for feature_name, feature_index in zip(feature_names, feature_indices):
        feature_samples = [
            {**sample, "values": sample["features"][:, feature_index]}
            for sample in report_samples
        ]
        stem = safe_feature_filename(feature_name)
        normalized_path = plots_dir / f"{stem}.png"
        stacked_path = plots_dir / f"{stem}_stacked_input_xsec.png"
        write_observable_shape_plot(
            normalized_path,
            feature_name,
            feature_samples,
            title=f"Input shape: {feature_name}",
        )
        stacked_metadata = write_stacked_input_cross_section_plot(
            stacked_path,
            feature_name,
            feature_samples,
            signal_scale=1000.0,
        )
        plot_rows.extend(
            [
                {
                    "feature": feature_name,
                    "path": str(normalized_path),
                    "detail": "normalized input shapes",
                },
                {
                    "feature": feature_name,
                    "path": str(stacked_path),
                    "detail": "stacked physical input cross section; SM shown x1000",
                },
            ]
        )
        plot_metadata.append(
            {
                "feature": feature_name,
                "normalized_shape": str(normalized_path),
                "stacked_input_xsec": stacked_metadata,
            }
        )

    samples_csv = report_dir / "input_samples.csv"
    _write_rows(samples_csv, sample_rows)
    metadata = {
        "observable_set": observable_set,
        "feature_profile": feature_profile,
        "feature_names": feature_names,
        "luminosity_fb_inverse": luminosity,
        "normalization": (
            "Physical event weights from the v2 study; stacked bin contents sum "
            "to the effective input cross section after rate factors."
        ),
        "signal_display_scale": 1000.0,
        "samples": [
            {"label": row["label"], "file": row["file"]} for row in sample_rows
        ],
        "plots": plot_metadata,
        "report_line": (
            f"{len(feature_names)} observables; normalized comparisons and stacked "
            "input cross sections. The SM stack is enlarged by 1000 for visibility."
        ),
    }
    metadata_path = report_dir / "report_metadata.json"
    _write_json(metadata_path, metadata)
    index_path = report_dir / "index.html"
    write_report_index(
        index_path,
        plot_rows,
        samples_csv,
        metadata,
        title="4H XGBoost v2 Input Observables",
        table_label="Input-sample summary",
    )
    return {
        "status": "complete",
        "directory": str(report_dir),
        "index": str(index_path),
        "metadata": str(metadata_path),
        "input_samples": str(samples_csv),
        "observable_count": len(feature_names),
        "plot_count": len(plot_rows),
        "signal_display_scale": 1000.0,
    }


def _run_fingerprint(
    *,
    observable_set: str,
    profile: str,
    strategy: str,
    rotation: int,
    n_folds: int,
    seed: int,
    source_commit: str,
    fold_digest: str,
    normalization_inputs: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    package_versions: Mapping[str, Any],
    study_mode: str = "full",
) -> str:
    payload = {
        "method_version": METHOD_VERSION,
        "study_mode": str(study_mode),
        "classifier_weight_scale_version": CLASSIFIER_WEIGHT_SCALE_VERSION,
        "observable_set": observable_set,
        "profile": profile,
        "strategy": strategy,
        "rotation": int(rotation),
        "n_folds": int(n_folds),
        "seed": int(seed),
        "source_commit": source_commit,
        "fold_assignment_sha256": fold_digest,
        "normalization_inputs": normalization_inputs,
        "input_hashes": dict(sorted(input_hashes.items())),
        "package_versions": dict(sorted(package_versions.items())),
        "search_space": {
            "n_estimators": [200, 800, 100],
            "max_depth": [2, 6],
            "learning_rate": [0.01, 0.15, "log"],
            "min_child_weight": [1.0, 50.0, "log"],
            "subsample": [0.6, 1.0],
            "colsample_bytree": [0.6, 1.0],
            "gamma": [0.0, 0.01, 0.1, 1.0],
            "reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
            "reg_lambda": [0.1, 30.0, "log"],
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_fold_assignments(path: Path, samples: Sequence[EventSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_id", "kind", "source_entry_index", "event_index", "fold"]
        )
        for sample in samples:
            for source_entry, event_index, fold in zip(
                sample.source_entry_indices,
                sample.event_indices,
                sample.folds,
            ):
                writer.writerow(
                    [
                        sample.sample_id,
                        sample.kind,
                        int(source_entry),
                        int(event_index),
                        int(fold),
                    ]
                )


def _fold_assignment_digest(samples: Sequence[EventSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.sample_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(sample.source_entry_indices, dtype="<i8").tobytes())
        digest.update(np.asarray(sample.event_indices, dtype="<i8").tobytes())
        digest.update(np.asarray(sample.folds, dtype="<i2").tobytes())
    return digest.hexdigest()


def _load_legacy_baseline(path: Path | None) -> dict[tuple[float, float], float]:
    if path is None or not path.exists():
        return {}
    baseline = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                baseline[(float(row["c3"]), float(row["d4"]))] = float(row["xsec_95cl_fb"])
            except (KeyError, TypeError, ValueError):
                continue
    return baseline


def _add_baseline_ratios(
    rows: list[dict[str, Any]],
    legacy: Mapping[tuple[float, float], float],
    reference: Mapping[tuple[float, float], float] | None = None,
) -> None:
    for row in rows:
        key = (float(row["c3"]), float(row["d4"]))
        cut_limit = float(row["cut_sigma95_fb"])
        row["cut_exclusion_ratio"] = (
            float(row["xsec_fb"]) / cut_limit if cut_limit > 0.0 else None
        )
        shape_limit = row.get("shape_sigma95_fb")
        row["shape_exclusion_ratio"] = (
            float(row["xsec_fb"]) / float(shape_limit)
            if shape_limit is not None and float(shape_limit) > 0.0
            else None
        )
        legacy_value = legacy.get(key)
        row["legacy_exact_sigma95_fb"] = legacy_value
        row["cut_ratio_to_legacy"] = (
            float(row["cut_sigma95_fb"]) / legacy_value
            if legacy_value is not None and legacy_value > 0.0
            else None
        )
        if reference is not None:
            reference_value = reference.get(key)
            row["cut_ratio_to_reference"] = (
                float(row["cut_sigma95_fb"]) / reference_value
                if reference_value is not None and reference_value > 0.0
                else None
            )


def _draw_plot_watermark(axis: Any, watermark: str | None) -> None:
    if not watermark:
        return
    axis.text(
        0.5,
        0.04,
        str(watermark),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="crimson",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "crimson"},
        zorder=30,
    )


def _legacy_contour_fields(limit_kind: str) -> dict[str, str]:
    if limit_kind == "cut":
        return {
            "limit": "cut_sigma95_fb",
            "background_x0p25": "cut_sigma95_background_x0p25_fb",
            "background_x4": "cut_sigma95_background_x4_fb",
        }
    if limit_kind == "shape":
        return {
            "limit": "shape_sigma95_fb",
            "background_x0p25": "shape_sigma95_background_x0p25_fb",
            "background_x4": "shape_sigma95_background_x4_fb",
        }
    raise ValueError(f"Unknown exclusion-contour kind {limit_kind!r}")


def _legacy_contour_spec(
    rows: Sequence[Mapping[str, Any]],
    limit_kind: str,
    *,
    expected_coordinates: Sequence[tuple[float, float]] | None = None,
    expected_xsecs: Mapping[tuple[float, float], float] | None = None,
) -> dict[str, Any]:
    """Prepare pointwise v2 exclusion ratios without introducing a fit."""

    fields = _legacy_contour_fields(limit_kind)
    table_coordinates: set[tuple[float, float]] = set()
    for row in rows:
        try:
            coordinate = (float(row["c3"]), float(row["d4"]))
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in coordinate):
            table_coordinates.add(coordinate)
    if expected_coordinates is None and expected_xsecs is not None:
        expected = {
            (float(c3), float(d4)) for c3, d4 in expected_xsecs
        }
        expectation_source = "study-manifest-cross-sections"
    elif expected_coordinates is None:
        expected = table_coordinates
        expectation_source = "result-table"
    else:
        expected = {
            (float(c3), float(d4)) for c3, d4 in expected_coordinates
        }
        expectation_source = "study-manifest"

    grouped: dict[tuple[float, float], dict[str, list[float]]] = {}
    for row in rows:
        try:
            c3 = float(row["c3"])
            d4 = float(row["d4"])
            xsec = float(row["xsec_fb"])
            limit = float(row[fields["limit"]])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (c3, d4, xsec, limit)):
            continue
        if xsec <= 0.0 or limit <= 0.0:
            continue
        bucket = grouped.setdefault(
            (c3, d4),
            {"xsec": [], "limit": [], "down": [], "up": []},
        )
        bucket["xsec"].append(xsec)
        bucket["limit"].append(limit)
        for bucket_key, field_key in (
            ("down", "background_x0p25"),
            ("up", "background_x4"),
        ):
            try:
                value = float(row[fields[field_key]])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                bucket[bucket_key].append(value)

    usable_coordinates = set(grouped)
    missing_coordinates = sorted(expected - usable_coordinates)
    unexpected_coordinates = sorted(usable_coordinates - expected)
    if missing_coordinates or unexpected_coordinates:
        return {
            "status": "skipped",
            "reason": (
                f"incomplete {limit_kind} point set: "
                f"{len(usable_coordinates)} usable, {len(expected)} expected"
            ),
            "limit_kind": limit_kind,
            "fields": fields,
            "expected_point_count": len(expected),
            "usable_point_count": len(usable_coordinates),
            "expectation_source": expectation_source,
            "missing_coordinates": [list(point) for point in missing_coordinates],
            "unexpected_coordinates": [
                list(point) for point in unexpected_coordinates
            ],
        }
    duplicate_coordinates = sorted(
        point
        for point, values in grouped.items()
        if len(values["xsec"]) != 1
        or len(values["limit"]) != 1
        or len(values["down"]) > 1
        or len(values["up"]) > 1
    )
    if duplicate_coordinates:
        return {
            "status": "skipped",
            "reason": (
                f"duplicate {limit_kind} rows for "
                f"{len(duplicate_coordinates)} c3/d4 points"
            ),
            "limit_kind": limit_kind,
            "fields": fields,
            "expected_point_count": len(expected),
            "usable_point_count": len(usable_coordinates),
            "expectation_source": expectation_source,
            "duplicate_coordinates": [
                list(point) for point in duplicate_coordinates
            ],
        }
    if expected_xsecs is not None:
        manifest_xsec_coordinates = {
            (float(c3), float(d4)) for c3, d4 in expected_xsecs
        }
        if manifest_xsec_coordinates != expected:
            return {
                "status": "skipped",
                "reason": (
                    f"{limit_kind} manifest cross-section coordinates do not "
                    "match the expected point set"
                ),
                "limit_kind": limit_kind,
                "fields": fields,
                "expected_point_count": len(expected),
                "usable_point_count": len(usable_coordinates),
                "expectation_source": expectation_source,
            }
        mismatches = []
        for point in sorted(expected):
            expected_xsec = float(expected_xsecs[point])
            actual_xsec = float(grouped[point]["xsec"][0])
            if not math.isclose(
                actual_xsec,
                expected_xsec,
                rel_tol=1.0e-10,
                abs_tol=1.0e-15,
            ):
                mismatches.append(
                    {
                        "c3": point[0],
                        "d4": point[1],
                        "table_xsec_fb": actual_xsec,
                        "manifest_xsec_fb": expected_xsec,
                    }
                )
        if mismatches:
            return {
                "status": "skipped",
                "reason": (
                    f"{len(mismatches)} {limit_kind} production cross sections "
                    "do not match the study manifest"
                ),
                "limit_kind": limit_kind,
                "fields": fields,
                "expected_point_count": len(expected),
                "usable_point_count": len(usable_coordinates),
                "expectation_source": expectation_source,
                "xsec_mismatches": mismatches,
            }
    band_missing_coordinates = sorted(
        point
        for point in expected
        if not grouped[point]["down"] or not grouped[point]["up"]
    )
    if band_missing_coordinates:
        return {
            "status": "skipped",
            "reason": (
                f"incomplete {limit_kind} background envelope: "
                f"{len(band_missing_coordinates)} of {len(expected)} points missing"
            ),
            "limit_kind": limit_kind,
            "fields": fields,
            "expected_point_count": len(expected),
            "usable_point_count": len(usable_coordinates),
            "expectation_source": expectation_source,
            "band_missing_coordinates": [
                list(point) for point in band_missing_coordinates
            ],
        }

    if len(grouped) < 3:
        return {
            "status": "skipped",
            "reason": f"fewer than three finite {limit_kind} exclusion points",
            "limit_kind": limit_kind,
            "fields": fields,
            "expected_point_count": len(expected),
            "usable_point_count": len(usable_coordinates),
            "expectation_source": expectation_source,
        }

    coordinates = sorted(grouped)
    c3 = np.asarray([point[0] for point in coordinates], dtype=float)
    d4 = np.asarray([point[1] for point in coordinates], dtype=float)
    centered = np.column_stack((c3 - c3[0], d4 - d4[0]))
    if np.linalg.matrix_rank(centered) < 2:
        return {
            "status": "skipped",
            "reason": f"{limit_kind} exclusion points are collinear",
            "limit_kind": limit_kind,
            "fields": fields,
        }

    xsec = np.asarray(
        [float(np.mean(grouped[point]["xsec"])) for point in coordinates]
    )
    limit = np.asarray(
        [float(np.mean(grouped[point]["limit"])) for point in coordinates]
    )
    down_limit = np.asarray(
        [
            float(np.mean(grouped[point]["down"]))
            if grouped[point]["down"]
            else np.nan
            for point in coordinates
        ]
    )
    up_limit = np.asarray(
        [
            float(np.mean(grouped[point]["up"]))
            if grouped[point]["up"]
            else np.nan
            for point in coordinates
        ]
    )
    central_ratio = xsec / limit
    down_ratio = np.divide(
        xsec,
        down_limit,
        out=np.full_like(xsec, np.nan),
        where=np.isfinite(down_limit) & (down_limit > 0.0),
    )
    up_ratio = np.divide(
        xsec,
        up_limit,
        out=np.full_like(xsec, np.nan),
        where=np.isfinite(up_limit) & (up_limit > 0.0),
    )
    sm_indices = np.flatnonzero(
        np.isclose(c3, 0.0, rtol=0.0, atol=1.0e-12)
        & np.isclose(d4, 0.0, rtol=0.0, atol=1.0e-12)
    )
    sm_xsec = float(xsec[sm_indices[0]]) if sm_indices.size else None
    band_valid = np.isfinite(down_limit) & np.isfinite(up_limit)
    band_missing_coordinates = []
    band_complete = True
    band_ordering_valid = bool(
        np.all(
            (down_limit[band_valid] <= limit[band_valid] + 1.0e-12)
            & (limit[band_valid] <= up_limit[band_valid] + 1.0e-12)
        )
    ) if np.any(band_valid) else None
    if band_ordering_valid is not True:
        return {
            "status": "skipped",
            "reason": (
                f"invalid {limit_kind} background-envelope ordering; expected "
                "sigma95(Bx0.25) <= sigma95(B) <= sigma95(Bx4) at every point"
            ),
            "limit_kind": limit_kind,
            "fields": fields,
            "expected_point_count": len(expected),
            "usable_point_count": len(usable_coordinates),
            "expectation_source": expectation_source,
            "band_ordering_valid": False,
        }
    return {
        "status": "ok",
        "limit_kind": limit_kind,
        "fields": fields,
        "c3": c3,
        "d4": d4,
        "xsec_fb": xsec,
        "limit_fb": limit,
        "central_ratio": central_ratio,
        "background_x0p25_ratio": down_ratio,
        "background_x4_ratio": up_ratio,
        "sm_xsec_fb": sm_xsec,
        "point_count": int(c3.size),
        "expected_point_count": len(expected),
        "expectation_source": expectation_source,
        "duplicate_rows_merged": int(sum(len(value["limit"]) for value in grouped.values()) - len(grouped)),
        "central_ratio_min": float(np.min(central_ratio)),
        "central_ratio_max": float(np.max(central_ratio)),
        "band_complete": band_complete,
        "band_missing_coordinates": band_missing_coordinates,
        "band_ordering_valid": band_ordering_valid,
    }


def _interpolate_log_point_values(
    c3: np.ndarray,
    d4: np.ndarray,
    values: np.ndarray,
    c3_grid: np.ndarray,
    d4_grid: np.ndarray,
    *,
    method: str = "linear",
) -> np.ma.MaskedArray | None:
    method = str(method).strip().lower()
    if method not in CONTOUR_INTERPOLATION_METHODS:
        raise ValueError(
            f"Unknown contour interpolation {method!r}; choose from "
            + ", ".join(CONTOUR_INTERPOLATION_METHODS)
        )
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values > 0.0)
    if np.count_nonzero(valid) < 3:
        return None
    centered = np.column_stack(
        (c3[valid] - c3[valid][0], d4[valid] - d4[valid][0])
    )
    if np.linalg.matrix_rank(centered) < 2:
        return None
    clough_tocher = None
    if method == "clough-tocher":
        try:
            from scipy.interpolate import CloughTocher2DInterpolator
        except ImportError as error:
            raise RuntimeError(
                "Clough-Tocher contour interpolation requires SciPy"
            ) from error
        clough_tocher = CloughTocher2DInterpolator
    try:
        if method == "linear":
            import matplotlib.tri as mtri

            triangulation = mtri.Triangulation(c3[valid], d4[valid])
            interpolator = mtri.LinearTriInterpolator(
                triangulation, np.log10(values[valid])
            )
            interpolated = interpolator(c3_grid, d4_grid)
        else:
            points = np.column_stack((c3[valid], d4[valid]))
            interpolator = clough_tocher(
                points,
                np.log10(values[valid]),
                fill_value=np.nan,
                rescale=True,
            )
            interpolated = interpolator(c3_grid, d4_grid)
        return np.ma.masked_invalid(interpolated)
    except (RuntimeError, ValueError):
        return None


def _masked_range(values: np.ma.MaskedArray | None) -> tuple[float, float] | None:
    if values is None:
        return None
    compressed = np.ma.asarray(values).compressed()
    compressed = compressed[np.isfinite(compressed)]
    if not compressed.size:
        return None
    return float(np.min(compressed)), float(np.max(compressed))


def _remove_plot_pair(path: Path) -> None:
    for candidate in (path, path.with_suffix(".pdf")):
        if candidate.exists():
            candidate.unlink()


def _legacy_contour_paths(
    output_dir: Path,
    prefix: str,
    limit_kind: str,
) -> dict[str, Path]:
    stem = f"{prefix}_{limit_kind}_c3d4_hhhh_xsec_with_95cl"
    return {
        "xsec": output_dir / f"{stem}.png",
        "xsec_atlas": output_dir / f"{stem}_atl_phys_pub_2025_003.png",
        "no_xsec_atlas": output_dir
        / f"{stem}_atl_phys_pub_2025_003_no_ratio_contours.png",
    }


def _legacy_xsec_ratio_grid(
    spec: Mapping[str, Any],
    c3_grid: np.ndarray,
    d4_grid: np.ndarray,
    *,
    c3_range: tuple[float, float],
    d4_range: tuple[float, float],
    grid_bins: int,
    source_dir: Path | None,
    enabled: bool,
) -> tuple[np.ma.MaskedArray | None, dict[str, Any]]:
    del spec
    if not enabled:
        return None, {"status": "disabled"}
    if source_dir is None:
        return None, {
            "status": "skipped",
            "reason": "no hhhh cross-section source directory was configured",
        }
    cache_key = (
        str(Path(source_dir).expanduser().resolve()),
        tuple(float(value) for value in c3_range),
        tuple(float(value) for value in d4_range),
        int(grid_bins),
    )
    cached = _LEGACY_XSEC_SURFACE_CACHE.get(cache_key)
    if cached is not None:
        ratio, metadata = cached
        return ratio, {**metadata, "cache_hit": True}
    try:
        import c3d4_plot_style as legacy_style

        points, counts = legacy_style._read_hhhh_xsec_points(Path(source_dir))
        fit = legacy_style._fit_c3d4_chebyshev(
            points,
            "xsec_pb",
            "xsec_error_pb",
            legacy_style.DEFAULT_C3D4_CHEBYSHEV_TERMS,
            (-29.0, 31.0),
            (-699.0, 701.0),
        )
        if fit.get("status") != "ok":
            raise ValueError(fit.get("reason", "cross-section fit failed"))
        _, _, xsec_grid = legacy_style._evaluate_c3d4_chebyshev_grid(
            fit,
            c3_range,
            d4_range,
            grid_bins,
            grid_bins,
        )
        sm_xsec = legacy_style._evaluate_c3d4_chebyshev(0.0, 0.0, fit)
        if not math.isfinite(sm_xsec) or sm_xsec <= 0.0:
            raise ValueError("fitted SM cross section is not positive")
        raw_ratio = np.asarray(xsec_grid, dtype=float) / sm_xsec
        ratio = np.ma.masked_where(
            (~np.isfinite(raw_ratio)) | (raw_ratio <= 0.0), raw_ratio
        )
        metadata = {
            "status": "ok",
            "method": "legacy-chebyshev-xsec-fit",
            "source_dir": str(source_dir),
            "source_point_count": len(points),
            "source_run_counts": counts,
            "sm_xsec_pb": float(sm_xsec),
            "nonpositive_grid_cells": int(
                np.count_nonzero(np.isfinite(raw_ratio) & (raw_ratio <= 0.0))
            ),
            "chebyshev_fit": fit,
            "cache_hit": False,
        }
        wide_count = counts.get(legacy_style.DEFAULT_HHHH_XSEC_WIDE_RUNNUM, 0)
        if wide_count < legacy_style.DEFAULT_HHHH_XSEC_EXPECTED_WIDE_RUNS:
            metadata["wide_run_warning"] = (
                f"run {legacy_style.DEFAULT_HHHH_XSEC_WIDE_RUNNUM} has "
                f"{wide_count} completed points; expected "
                f"{legacy_style.DEFAULT_HHHH_XSEC_EXPECTED_WIDE_RUNS}"
            )
        _LEGACY_XSEC_SURFACE_CACHE[cache_key] = (ratio, metadata)
        return ratio, metadata
    except Exception as error:
        return None, {
            "status": "skipped",
            "reason": str(error),
            "method": "legacy-chebyshev-xsec-fit",
            "source_dir": str(source_dir),
        }


def _draw_legacy_style_exclusion_plot(
    path: Path,
    *,
    c3_grid: np.ndarray,
    d4_grid: np.ndarray,
    central_log_ratio: np.ma.MaskedArray,
    background_down_log_ratio: np.ma.MaskedArray | None,
    background_up_log_ratio: np.ma.MaskedArray | None,
    xsec_ratio_grid: np.ma.MaskedArray | None,
    perturbativity_grid: np.ndarray,
    include_xsec: bool,
    include_atlas: bool,
    limit_label: str,
    process_title: str,
    luminosity: float,
    c3_range: tuple[float, float],
    d4_range: tuple[float, float],
    watermark: str | None,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import c3d4_plot_style as legacy_style

    if include_xsec and xsec_ratio_grid is None:
        _remove_plot_pair(path)
        return {
            "status": "skipped",
            "reason": "cross-section ratio surface is unavailable",
        }

    central_range = _masked_range(central_log_ratio)
    if central_range is None:
        _remove_plot_pair(path)
        return {"status": "skipped", "reason": "no finite exclusion surface"}
    contour_drawn = central_range[0] <= 0.0 <= central_range[1] and central_range[0] != central_range[1]
    fig, axis = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    colorbar = None
    nonpositive_xsec_region_drawn = False
    if include_xsec:
        positive = np.ma.asarray(xsec_ratio_grid).compressed()
        positive = positive[np.isfinite(positive) & (positive > 0.0)]
        if not positive.size:
            plt.close(fig)
            _remove_plot_pair(path)
            return {"status": "skipped", "reason": "no positive cross-section ratios"}
        levels = legacy_style._make_hhhh_xsec_log_levels(positive)
        contour = axis.contourf(
            c3_grid,
            d4_grid,
            xsec_ratio_grid,
            levels=levels,
            norm=colors.LogNorm(vmin=levels[0], vmax=levels[-1]),
            cmap="viridis",
            extend="both",
        )
        raw_xsec_ratio = np.asarray(np.ma.getdata(xsec_ratio_grid), dtype=float)
        nonpositive_xsec = np.isfinite(raw_xsec_ratio) & (raw_xsec_ratio <= 0.0)
        if np.any(nonpositive_xsec):
            axis.contourf(
                c3_grid,
                d4_grid,
                nonpositive_xsec,
                levels=[0.5, 1.5],
                colors=["0.75"],
                alpha=0.8,
            )
            nonpositive_xsec_region_drawn = True
        line_levels = legacy_style._make_hhhh_xsec_line_levels(levels)
        if line_levels:
            lines = axis.contour(
                c3_grid,
                d4_grid,
                xsec_ratio_grid,
                levels=line_levels,
                colors="white",
                linewidths=0.55,
            )
            axis.clabel(
                lines,
                fmt=legacy_style._format_hhhh_xsec_level,
                inline=True,
                fontsize=10,
            )
        colorbar = fig.colorbar(contour, ax=axis)
        colorbar.set_label(
            r"$\sigma(c_3,d_4)/\sigma(0,0)$", fontsize=18
        )
        colorbar.ax.tick_params(labelsize=15)
    else:
        axis.set_facecolor("white")
        axis.grid(alpha=0.2, linewidth=0.5)

    band_drawn = False
    band_boundary_count = 0
    if background_down_log_ratio is not None and background_up_log_ratio is not None:
        down = np.ma.asarray(background_down_log_ratio)
        up = np.ma.asarray(background_up_log_ratio)
        valid = ~(
            np.ma.getmaskarray(down)
            | np.ma.getmaskarray(up)
            | ~np.isfinite(np.asarray(down.filled(np.nan)))
            | ~np.isfinite(np.asarray(up.filled(np.nan)))
        )
        lower = np.minimum(down.filled(np.nan), up.filled(np.nan))
        upper = np.maximum(down.filled(np.nan), up.filled(np.nan))
        band = valid & (lower <= 0.0) & (upper >= 0.0)
        if np.any(band):
            band_surface = np.ma.masked_where(~valid, band.astype(float))
            axis.contourf(
                c3_grid,
                d4_grid,
                band_surface,
                levels=[0.5, 1.5],
                colors=["crimson"],
                alpha=0.32,
            )
            axis.plot(
                [],
                [],
                color="crimson",
                linewidth=8.0,
                alpha=0.32,
                label=r"Background $\times[0.25,4]$",
            )
            band_drawn = True
        for boundary in (down, up):
            boundary_range = _masked_range(boundary)
            if (
                boundary_range is not None
                and boundary_range[0] <= 0.0 <= boundary_range[1]
                and boundary_range[0] != boundary_range[1]
            ):
                axis.contour(
                    c3_grid,
                    d4_grid,
                    boundary,
                    levels=[0.0],
                    colors=["crimson"],
                    linewidths=0.75,
                    linestyles="--",
                )
                band_boundary_count += 1

    perturbativity_min = float(np.nanmin(perturbativity_grid))
    perturbativity_max = float(np.nanmax(perturbativity_grid))
    perturbativity_drawn = (
        perturbativity_min
        <= legacy_style.DEFAULT_HHHH_PERTURBATIVITY_LEVEL
        <= perturbativity_max
    )
    if perturbativity_drawn:
        axis.contour(
            c3_grid,
            d4_grid,
            perturbativity_grid,
            levels=[legacy_style.DEFAULT_HHHH_PERTURBATIVITY_LEVEL],
            colors=["black"],
            linestyles="--",
            linewidths=1.7,
        )
        axis.plot(
            [],
            [],
            color="black",
            linestyle="--",
            linewidth=1.7,
            label=r"Perturbative unitarity, $hh \rightarrow hh$",
        )

    if contour_drawn:
        axis.contour(
            c3_grid,
            d4_grid,
            central_log_ratio,
            levels=[0.0],
            colors=["crimson"],
            linewidths=2.0,
        )
        axis.plot([], [], color="crimson", linewidth=2.0, label=limit_label)

    atlas_metadata = None
    if include_atlas:
        atlas_metadata = legacy_style._plot_atlas_phys_pub_2025_003_curve(axis)
    legacy_style._plot_sm_marker(axis)
    axis.set_xlim(c3_range)
    axis.set_ylim(d4_range)
    axis.set_xlabel(
        r"$c_3$", fontsize=legacy_style.DEFAULT_C3D4_OVERLAY_AXIS_LABEL_FONTSIZE
    )
    axis.set_ylabel(
        r"$d_4$", fontsize=legacy_style.DEFAULT_C3D4_OVERLAY_AXIS_LABEL_FONTSIZE
    )
    axis.set_title(
        process_title
        + " at 14 TeV, "
        + rf"$L={float(luminosity):g}\,\mathrm{{fb}}^{{-1}}$",
        fontsize=20,
    )
    axis.tick_params(axis="both", labelsize=15)
    _draw_plot_watermark(axis, watermark)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="best", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return {
        "status": "ok",
        "png": str(path),
        "pdf": str(path.with_suffix(".pdf")),
        "include_xsec": include_xsec,
        "include_atlas": include_atlas,
        "nonpositive_xsec_region_drawn": nonpositive_xsec_region_drawn,
        "limit_contour_drawn": contour_drawn,
        "background_band_drawn": band_drawn,
        "background_boundary_count": band_boundary_count,
        "perturbativity_contour_drawn": perturbativity_drawn,
        "atlas_reference_curve": atlas_metadata,
        "process_title": process_title,
        "limit_label": limit_label,
        "watermark": watermark,
    }


def _write_legacy_style_exclusion_contours(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    prefix: str,
    *,
    limit_kind: str,
    watermark: str | None = None,
    luminosity: float = 3000.0,
    c3_range: tuple[float, float] = DEFAULT_CONTOUR_C3_RANGE,
    d4_range: tuple[float, float] = DEFAULT_CONTOUR_D4_RANGE,
    grid_bins: int = DEFAULT_CONTOUR_GRID_BINS,
    xsec_source_dir: Path | None = DEFAULT_HHHH_XSEC_SOURCE_DIR,
    xsec_overlay: bool = True,
    expected_coordinates: Sequence[tuple[float, float]] | None = None,
    expected_xsecs: Mapping[tuple[float, float], float] | None = None,
    interpolation: str = "linear",
) -> dict[str, Any]:
    paths = _legacy_contour_paths(Path(output_dir), prefix, limit_kind)
    spec = _legacy_contour_spec(
        rows,
        limit_kind,
        expected_coordinates=expected_coordinates,
        expected_xsecs=expected_xsecs,
    )
    if spec.get("status") != "ok":
        for path in paths.values():
            _remove_plot_pair(path)
        return {
            **spec,
            "style_version": LEGACY_CONTOUR_STYLE_VERSION,
            "outputs": {},
            "watermark": watermark,
        }
    if int(grid_bins) < 3:
        raise ValueError("legacy contour grid_bins must be at least three")
    interpolation = str(interpolation).strip().lower()
    if interpolation not in CONTOUR_INTERPOLATION_METHODS:
        raise ValueError(
            f"Unknown contour interpolation {interpolation!r}; choose from "
            + ", ".join(CONTOUR_INTERPOLATION_METHODS)
        )
    c3_values = np.linspace(float(c3_range[0]), float(c3_range[1]), int(grid_bins))
    d4_values = np.linspace(float(d4_range[0]), float(d4_range[1]), int(grid_bins))
    c3_grid, d4_grid = np.meshgrid(c3_values, d4_values)
    central_log = _interpolate_log_point_values(
        spec["c3"],
        spec["d4"],
        spec["central_ratio"],
        c3_grid,
        d4_grid,
        method=interpolation,
    )
    if central_log is None or _masked_range(central_log) is None:
        for path in paths.values():
            _remove_plot_pair(path)
        return {
            "status": "skipped",
            "reason": "exclusion-ratio interpolation failed",
            "style_version": LEGACY_CONTOUR_STYLE_VERSION,
            "limit_kind": limit_kind,
            "outputs": {},
            "watermark": watermark,
        }
    background_down_log = _interpolate_log_point_values(
        spec["c3"],
        spec["d4"],
        spec["background_x0p25_ratio"],
        c3_grid,
        d4_grid,
        method=interpolation,
    )
    background_up_log = _interpolate_log_point_values(
        spec["c3"],
        spec["d4"],
        spec["background_x4_ratio"],
        c3_grid,
        d4_grid,
        method=interpolation,
    )
    xsec_ratio, xsec_metadata = _legacy_xsec_ratio_grid(
        spec,
        c3_grid,
        d4_grid,
        c3_range=c3_range,
        d4_range=d4_range,
        grid_bins=int(grid_bins),
        source_dir=None if xsec_source_dir is None else Path(xsec_source_dir),
        enabled=bool(xsec_overlay),
    )
    import c3d4_plot_style as legacy_style

    perturbativity_key = (
        tuple(float(value) for value in c3_range),
        tuple(float(value) for value in d4_range),
        int(grid_bins),
    )
    perturbativity = _LEGACY_PERTURBATIVITY_CACHE.get(perturbativity_key)
    if perturbativity is None:
        perturbativity = legacy_style._hhhh_perturbativity_grid(c3_grid, d4_grid)
        _LEGACY_PERTURBATIVITY_CACHE[perturbativity_key] = perturbativity
    includes_postfit_hhhbb = any(
        "hhhbb" in str(row.get("signal_components", "")).split(",")
        for row in rows
    )
    if includes_postfit_hhhbb:
        process_title = r"$hhhh + hhhg\,(g\to b\bar b)$ signal"
        label = (
            r"$hhhh + hhhg\,(g\to b\bar b)$, exact "
            r"$\mathrm{CL}_{s}$ 95% (cut)"
            if limit_kind == "cut"
            else r"$hhhh + hhhg\,(g\to b\bar b)$, pyhf "
            r"$\mathrm{CL}_{s}$ 95% (shape)"
        )
    else:
        process_title = r"$gg \to hhhh$"
        label = (
            r"$gg \rightarrow hhhh \rightarrow 8b$, exact "
            r"$\mathrm{CL}_{s}$ 95% (cut)"
            if limit_kind == "cut"
            else r"$gg \rightarrow hhhh \rightarrow 8b$, pyhf "
            r"$\mathrm{CL}_{s}$ 95% (shape)"
        )
    variants = {
        "xsec": {"include_xsec": True, "include_atlas": False},
        "xsec_atlas": {"include_xsec": True, "include_atlas": True},
        "no_xsec_atlas": {"include_xsec": False, "include_atlas": True},
    }
    outputs = {}
    for name, settings in variants.items():
        if settings["include_xsec"] and not xsec_overlay:
            _remove_plot_pair(paths[name])
            outputs[name] = {"status": "disabled"}
            continue
        try:
            outputs[name] = _draw_legacy_style_exclusion_plot(
                paths[name],
                c3_grid=c3_grid,
                d4_grid=d4_grid,
                central_log_ratio=central_log,
                background_down_log_ratio=background_down_log,
                background_up_log_ratio=background_up_log,
                xsec_ratio_grid=xsec_ratio,
                perturbativity_grid=perturbativity,
                include_xsec=settings["include_xsec"],
                include_atlas=settings["include_atlas"],
                limit_label=label,
                process_title=process_title,
                luminosity=luminosity,
                c3_range=c3_range,
                d4_range=d4_range,
                watermark=watermark,
            )
        except Exception as error:
            # Plotting is an output stage: preserve the numerical study and
            # publish an auditable skipped product instead of aborting it.
            import matplotlib.pyplot as plt

            plt.close("all")
            _remove_plot_pair(paths[name])
            outputs[name] = {
                "status": "skipped",
                "reason": f"{type(error).__name__}: {error}",
            }
    successful_variants = sum(
        output.get("status") == "ok" for output in outputs.values()
    )
    expected_variants = 3 if xsec_overlay else 1
    if successful_variants == expected_variants:
        contour_status = "ok"
        contour_reason = None
    elif successful_variants:
        contour_status = "partial"
        contour_reason = (
            f"wrote {successful_variants}/{expected_variants} expected plot variants"
        )
    else:
        contour_status = "skipped"
        contour_reason = "no plot variant was written successfully"
    return {
        "status": contour_status,
        "reason": contour_reason,
        "style_version": LEGACY_CONTOUR_STYLE_VERSION,
        "limit_kind": limit_kind,
        "interpolation": (
            "piecewise-linear triangulation of log10(xsec_fb/sigma95_fb)"
            if interpolation == "linear"
            else "Clough-Tocher piecewise-cubic interpolation of log10(xsec_fb/sigma95_fb)"
        ),
        "interpolation_method": interpolation,
        "point_count": spec["point_count"],
        "expected_point_count": spec["expected_point_count"],
        "expectation_source": spec["expectation_source"],
        "duplicate_rows_merged": spec["duplicate_rows_merged"],
        "central_ratio_min": spec["central_ratio_min"],
        "central_ratio_max": spec["central_ratio_max"],
        "band_complete": spec["band_complete"],
        "band_missing_coordinates": spec["band_missing_coordinates"],
        "band_ordering_valid": spec["band_ordering_valid"],
        "viewport": {"c3": list(c3_range), "d4": list(d4_range)},
        "grid_bins": int(grid_bins),
        "xsec_surface": xsec_metadata,
        "outputs": outputs,
        "successful_variant_count": int(successful_variants),
        "expected_variant_count": int(expected_variants),
        "watermark": watermark,
    }


def _write_legacy_style_contour_set(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    prefix: str,
    *,
    watermark: str | None = None,
    luminosity: float = 3000.0,
    c3_range: tuple[float, float] = DEFAULT_CONTOUR_C3_RANGE,
    d4_range: tuple[float, float] = DEFAULT_CONTOUR_D4_RANGE,
    grid_bins: int = DEFAULT_CONTOUR_GRID_BINS,
    xsec_source_dir: Path | None = DEFAULT_HHHH_XSEC_SOURCE_DIR,
    xsec_overlay: bool = True,
    expected_coordinates: Sequence[tuple[float, float]] | None = None,
    expected_xsecs: Mapping[tuple[float, float], float] | None = None,
    interpolation: str = "linear",
) -> dict[str, Any]:
    return {
        "style_version": LEGACY_CONTOUR_STYLE_VERSION,
        "cut": _write_legacy_style_exclusion_contours(
            rows,
            output_dir,
            prefix,
            limit_kind="cut",
            watermark=watermark,
            luminosity=luminosity,
            c3_range=c3_range,
            d4_range=d4_range,
            grid_bins=grid_bins,
            xsec_source_dir=xsec_source_dir,
            xsec_overlay=xsec_overlay,
            expected_coordinates=expected_coordinates,
            expected_xsecs=expected_xsecs,
            interpolation=interpolation,
        ),
        "shape": _write_legacy_style_exclusion_contours(
            rows,
            output_dir,
            prefix,
            limit_kind="shape",
            watermark=watermark,
            luminosity=luminosity,
            c3_range=c3_range,
            d4_range=d4_range,
            grid_bins=grid_bins,
            xsec_source_dir=xsec_source_dir,
            xsec_overlay=xsec_overlay,
            expected_coordinates=expected_coordinates,
            expected_xsecs=expected_xsecs,
            interpolation=interpolation,
        ),
    }


def _write_map(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    output: Path,
    *,
    title: str,
    logarithmic: bool = False,
    contour_level: float | None = None,
    watermark: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    x = np.asarray([float(row["c3"]) for row in rows])
    y = np.asarray([float(row["d4"]) for row in rows])
    values = []
    for row in rows:
        try:
            values.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            values.append(float("nan"))
    z = np.asarray(values)
    valid = np.isfinite(z) & (z > 0.0 if logarithmic else True)
    if np.sum(valid) < 3:
        return
    triangulation = mtri.Triangulation(x[valid], y[valid])
    plot_values = np.log10(z[valid]) if logarithmic else z[valid]
    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    filled = axis.tricontourf(triangulation, plot_values, levels=30, cmap="viridis")
    colorbar = figure.colorbar(filled, ax=axis)
    colorbar.set_label(f"log10({field})" if logarithmic else field)
    axis.scatter(x[valid], y[valid], s=12, facecolors="none", edgecolors="black", linewidths=0.5)
    if contour_level is not None:
        level = math.log10(contour_level) if logarithmic else contour_level
        try:
            axis.tricontour(triangulation, plot_values, levels=[level], colors="white", linewidths=1.2)
        except ValueError:
            pass
    axis.set_xlabel(r"$c_3$")
    axis.set_ylabel(r"$d_4$")
    axis.set_title(title)
    _draw_plot_watermark(axis, watermark)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def _write_standard_maps(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    prefix: str,
    *,
    watermark: str | None = None,
    legacy_contours: bool = False,
    luminosity: float = 3000.0,
    contour_c3_range: tuple[float, float] = DEFAULT_CONTOUR_C3_RANGE,
    contour_d4_range: tuple[float, float] = DEFAULT_CONTOUR_D4_RANGE,
    contour_grid_bins: int = DEFAULT_CONTOUR_GRID_BINS,
    contour_interpolation: str = "linear",
    xsec_source_dir: Path | None = DEFAULT_HHHH_XSEC_SOURCE_DIR,
    xsec_overlay: bool = True,
) -> dict[str, Any]:
    map_specs = (
        ("feature_tree_efficiency", "Feature-tree efficiency", False),
        ("xgboost_efficiency", "Cross-fitted XGBoost efficiency", False),
        ("threshold_mean", "Mean validation-selected threshold", False),
        ("cut_sigma95_fb", "Exact single-bin expected cross-section limit", True),
        ("bin_count", "Validation-selected score-bin count", False),
        ("pyhf_one_bin_sigma95_fb", "pyhf asymptotic one-bin control", True),
        ("shape_sigma95_fb", "pyhf score-shape limit with MC statistics", True),
        ("shape_ratio_to_pyhf_one_bin", "Shape / pyhf one-bin limit ratio", False),
        ("cut_ratio_to_legacy", "Cut limit / legacy exact limit", False),
        ("cut_ratio_to_reference", "Cut limit / tuned SM-model limit", False),
    )
    for field, title, logarithmic in map_specs:
        _write_map(
            rows,
            field,
            output_dir / f"{prefix}_{field}.png",
            title=title,
            logarithmic=logarithmic,
            watermark=watermark,
        )
    _write_map(
        rows,
        "cut_exclusion_ratio",
        output_dir / f"{prefix}_cut_exclusion_contour.png",
        title="Production cross section / exact cut limit",
        contour_level=1.0,
        watermark=watermark,
    )
    _write_map(
        rows,
        "shape_exclusion_ratio",
        output_dir / f"{prefix}_shape_exclusion_contour.png",
        title="Production cross section / pyhf shape limit",
        contour_level=1.0,
        watermark=watermark,
    )
    legacy_outputs = None
    if legacy_contours:
        legacy_outputs = _write_legacy_style_contour_set(
            rows,
            output_dir,
            prefix,
            watermark=watermark,
            luminosity=luminosity,
            c3_range=contour_c3_range,
            d4_range=contour_d4_range,
            grid_bins=contour_grid_bins,
            xsec_source_dir=xsec_source_dir,
            xsec_overlay=xsec_overlay,
            interpolation=contour_interpolation,
        )
        _write_json_atomic(
            Path(output_dir) / "legacy_contour_manifest.json",
            legacy_outputs,
        )
    return {
        "output_dir": str(output_dir),
        "prefix": prefix,
        "legacy_style_contours": legacy_outputs,
    }


def _result_labels(
    policy: StudyModePolicy,
    *,
    paper_ready: bool | None = None,
    score_shape_included: bool | None = None,
    uses_complete_event_samples: bool | None = None,
) -> dict[str, Any]:
    return {
        "study_mode": policy.name,
        "result_level": policy.result_level,
        "physics_result_valid": policy.physics_result_valid,
        "paper_ready": policy.paper_ready if paper_ready is None else bool(paper_ready),
        "uses_complete_event_samples": (
            policy.max_events is None
            if uses_complete_event_samples is None
            else bool(uses_complete_event_samples)
        ),
        "score_shape_included": (
            policy.run_shape
            if score_shape_included is None
            else bool(score_shape_included)
        ),
    }


def _annotate_result_rows(
    rows: Sequence[dict[str, Any]],
    policy: StudyModePolicy,
    *,
    uses_complete_event_samples: bool | None = None,
) -> None:
    """Attach an explicit publication-status label to every point result."""

    labels = _result_labels(
        policy,
        uses_complete_event_samples=uses_complete_event_samples,
    )
    for row in rows:
        row.update(labels)


def _publish_cut_preview(
    rows: Sequence[Mapping[str, Any]],
    strategy_dir: Path,
    *,
    strategy: str,
    policy: StudyModePolicy,
    watermark: str,
    luminosity: float = 3000.0,
    contour_c3_range: tuple[float, float] = DEFAULT_CONTOUR_C3_RANGE,
    contour_d4_range: tuple[float, float] = DEFAULT_CONTOUR_D4_RANGE,
    contour_grid_bins: int = DEFAULT_CONTOUR_GRID_BINS,
    contour_interpolation: str = "linear",
    xsec_source_dir: Path | None = DEFAULT_HHHH_XSEC_SOURCE_DIR,
    xsec_overlay: bool = True,
) -> dict[str, Any]:
    """Atomically advertise a usable cut-limit map before any shape fits."""

    preview_dir = strategy_dir / "cut_preview"
    status_file = preview_dir / "status.json"
    preview_level = (
        "preliminary-cut-only"
        if policy.name in {
            "full",
            "fast-sm",
            "fast-pooled",
            "fast-parameterized",
        }
        else policy.result_level
    )
    preview_rows = []
    for source in rows:
        row = dict(source)
        row.update(
            {
                "result_level": preview_level,
                "paper_ready": False,
                "score_shape_included": False,
            }
        )
        preview_rows.append(row)
    _write_json_atomic(
        status_file,
        {
            "status": "running",
            "strategy": strategy,
            "study_mode": policy.name,
            "result_level": preview_level,
            "paper_ready": False,
            "watermark": watermark,
        },
    )
    cut_results_csv = preview_dir / "cut_results.csv"
    cut_results_json = preview_dir / "cut_results.json"
    _write_rows(cut_results_csv, preview_rows)
    _write_json(cut_results_json, preview_rows)
    map_outputs = _write_standard_maps(
        preview_rows,
        preview_dir / "maps",
        f"{strategy}_preview",
        watermark=watermark,
        legacy_contours=True,
        luminosity=luminosity,
        contour_c3_range=contour_c3_range,
        contour_d4_range=contour_d4_range,
        contour_grid_bins=contour_grid_bins,
        contour_interpolation=contour_interpolation,
        xsec_source_dir=xsec_source_dir,
        xsec_overlay=xsec_overlay,
    )
    cut_exclusion_map = (
        preview_dir
        / "maps"
        / f"{strategy}_preview_cut_exclusion_contour.pdf"
    )
    payload = {
        "status": "complete",
        "strategy": strategy,
        "study_mode": policy.name,
        "result_level": preview_level,
        "physics_result_valid": policy.physics_result_valid,
        "paper_ready": False,
        "watermark": watermark,
        "status_file": str(status_file),
        "cut_results_csv": str(cut_results_csv),
        "cut_results_json": str(cut_results_json),
        "cut_results_sha256": (
            _sha256(cut_results_json) if cut_results_json.exists() else None
        ),
        "cut_exclusion_map": str(cut_exclusion_map) if cut_exclusion_map.exists() else None,
        "legacy_style_contours": (map_outputs or {}).get(
            "legacy_style_contours"
        ),
    }
    _write_json_atomic(status_file, payload)
    return payload


def _compact_validation(
    validation: Mapping[str, Any],
    *,
    result_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in validation.items()
        if key not in {"signal_rows", "background_rows", "background_rows_by_point"}
    }
    if result_metadata is not None:
        compact["result_metadata"] = dict(result_metadata)
    return compact


def _flatten_fold_points(
    rotations: Sequence[Mapping[str, Any]],
    container: str,
    *,
    result_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for rotation in rotations:
        payload = rotation[container]
        for point in payload["points"].values():
            row = {
                key: value
                for key, value in point.items()
                if not isinstance(value, (dict, list, tuple, np.ndarray))
            }
            if result_metadata is not None:
                row.update(result_metadata)
            rows.append(row)
    return rows


def _shape_model_for_record(record: Mapping[str, Any]) -> Any:
    """Return a parameterized shape-scoring model, loading it once per worker."""

    model = record.get("_model_object")
    if model is not None:
        return model
    model_path = record.get("_model_path") or record.get("model")
    if not model_path:
        raise RuntimeError("Parameterized shape record is missing a saved model path")
    import xgboost as xgb

    model = xgb.XGBClassifier(n_jobs=1)
    model.load_model(str(model_path))
    observable_set = record.get("_observable_set")
    profile = record.get("_feature_profile")
    if observable_set is not None and profile is not None:
        validate_model_contract(
            model,
            str(observable_set),
            str(profile),
            ml_parameter_features=PARAMETERIZED_ML_FEATURES,
        )
    # Shape records are ordinary dictionaries.  Caching here is intentionally
    # local to the worker process, never written back to the parent run state.
    record["_model_object"] = model
    return model


def _validation_fold_arrays(
    record: Mapping[str, Any],
    sample: EventSample | ShapePoint,
) -> dict[str, np.ndarray]:
    validation = record["validation"]
    signal = validation["signal_rows"][sample.sample_id]
    if validation.get("parameterized"):
        cache = record.setdefault("_validation_parameter_cache", {})
        if sample.point_id not in cache:
            rows = _score_partition(
                _shape_model_for_record(record),
                record["_background_samples"],
                rotation=int(record["rotation"]),
                split="validation",
                n_folds=int(record["_n_folds"]),
                profile_indices=record["_profile_indices"],
                scale_validation_to_full=True,
                parameterized=True,
                parameter_point=(float(sample.c3), float(sample.d4)),
            )
            cache[sample.point_id] = rows
        background_rows = cache[sample.point_id]
    else:
        background_rows = validation["background_rows"]
    background_scores = []
    background_weights = []
    for row in background_rows.values():
        background_scores.append(np.asarray(row["scores"], dtype=float))
        background_weights.append(
            np.asarray(row["physical_weights"], dtype=float) / float(row["scale"])
        )
    return {
        "signal_scores": np.asarray(signal["scores"], dtype=float),
        "signal_weights": np.asarray(signal["unit_xsec_weights"], dtype=float)
        / float(signal["scale"]),
        "background_scores": np.concatenate(background_scores),
        "background_weights": np.concatenate(background_weights),
    }


def _test_fold_arrays(
    record: Mapping[str, Any], sample: EventSample | ShapePoint
) -> dict[str, np.ndarray]:
    test = record["test"]
    signal = test["signal_rows"][sample.sample_id]
    signal_scores = np.asarray(signal["scores"], dtype=float)
    signal_weights = np.asarray(signal["unit_xsec_weights"], dtype=float)
    postfit = record.get("postfit_hhhbb_test")
    if postfit is not None:
        point = postfit.get("points", {}).get(sample.point_id)
        if point is None:
            raise ValueError(
                f"{sample.point_id}: post-fit hhhbb test template is missing"
            )
        sample_id = str(point["sample_id"])
        try:
            postfit_signal = postfit["signal_rows"][sample_id]
        except KeyError as error:
            raise ValueError(
                f"{sample.point_id}: post-fit hhhbb score row {sample_id!r} is missing"
            ) from error
        hhhh_xsec_fb = float(sample.xsec_fb)
        if not math.isfinite(hhhh_xsec_fb) or hhhh_xsec_fb <= 0.0:
            raise ValueError(
                f"{sample.point_id}: a positive finite hhhh cross section is required "
                "to express the post-fit hhhbb shape on the equivalent-hhhh-fb basis"
            )
        postfit_scores = np.asarray(postfit_signal["scores"], dtype=float)
        postfit_weights = (
            np.asarray(postfit_signal["physical_weights"], dtype=float)
            / hhhh_xsec_fb
        )
        if postfit_scores.shape != postfit_weights.shape:
            raise ValueError(
                f"{sample.point_id}: post-fit hhhbb scores and weights do not match"
            )
        signal_scores = np.concatenate((signal_scores, postfit_scores))
        signal_weights = np.concatenate((signal_weights, postfit_weights))
    if test.get("parameterized"):
        cache = record.setdefault("_test_parameter_cache", {})
        if sample.point_id not in cache:
            cache[sample.point_id] = _score_partition(
                _shape_model_for_record(record),
                record["_background_samples"],
                rotation=int(record["rotation"]),
                split="test",
                n_folds=int(record["_n_folds"]),
                profile_indices=record["_profile_indices"],
                scale_validation_to_full=False,
                parameterized=True,
                parameter_point=(float(sample.c3), float(sample.d4)),
            )
        background_rows = cache[sample.point_id]
    else:
        background_rows = test["background_rows"]
    return {
        "signal_scores": signal_scores,
        "signal_weights": signal_weights,
        "background_scores": _concatenate_partition(background_rows, "scores"),
        "background_weights": _concatenate_partition(
            background_rows, "physical_weights"
        ),
    }


def _compact_shape_partition_rows(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    signal: bool,
) -> dict[str, dict[str, Any]]:
    """Retain only arrays used by shape likelihood construction."""

    weight_key = "unit_xsec_weights" if signal else "physical_weights"
    output: dict[str, dict[str, Any]] = {}
    for sample_id, row in rows.items():
        output[str(sample_id)] = {
            "scores": np.asarray(row["scores"], dtype=float),
            weight_key: np.asarray(row[weight_key], dtype=float),
            "scale": float(row.get("scale", 1.0)),
        }
    return output


def _compact_shape_records(
    records: Sequence[Mapping[str, Any]],
    *,
    observable_set: str,
    profile: str,
    n_folds: int,
) -> list[dict[str, Any]]:
    """Build process-safe, score-only records for the shape executor.

    In particular, this strips live XGBoost objects from parameterized records.
    Those models are loaded lazily from their saved JSON files in each worker.
    """

    compact: list[dict[str, Any]] = []
    for record in records:
        validation = record["validation"]
        test = record["test"]
        parameterized = bool(validation.get("parameterized") or test.get("parameterized"))
        item: dict[str, Any] = {
            "rotation": int(record["rotation"]),
            "validation": {
                "parameterized": parameterized,
                "signal_rows": _compact_shape_partition_rows(
                    validation["signal_rows"], signal=True
                ),
                "background_rows": _compact_shape_partition_rows(
                    validation.get("background_rows", {}), signal=False
                ),
            },
            "test": {
                "parameterized": parameterized,
                "signal_rows": _compact_shape_partition_rows(
                    test["signal_rows"], signal=True
                ),
                "background_rows": _compact_shape_partition_rows(
                    test.get("background_rows", {}), signal=False
                ),
            },
        }
        postfit = record.get("postfit_hhhbb_test")
        if postfit is not None:
            item["postfit_hhhbb_test"] = {
                "points": {
                    str(point_id): {"sample_id": str(point["sample_id"])}
                    for point_id, point in postfit["points"].items()
                },
                "signal_rows": _compact_shape_partition_rows(
                    postfit["signal_rows"], signal=False
                ),
                "role": "postfit-signal-only",
            }
        if parameterized:
            model_path = record.get("model")
            if not model_path:
                raise RuntimeError("Parameterized shape record is missing its saved model")
            item.update(
                {
                    "_model_path": str(model_path),
                    "_background_samples": record["_background_samples"],
                    "_profile_indices": np.asarray(record["_profile_indices"], dtype=int),
                    "_n_folds": int(record.get("_n_folds", n_folds)),
                    "_observable_set": str(observable_set),
                    "_feature_profile": str(profile),
                }
            )
        compact.append(item)
    return compact


def _candidate_maps_for_validation(
    records: Sequence[Mapping[str, Any]],
    sample: EventSample | ShapePoint,
) -> tuple[list[dict[tuple[int, ...], Mapping[str, Any]]], list[tuple[int, ...]]]:
    maps = []
    common: set[tuple[int, ...]] | None = None
    # The background score distribution alone defines the quantile edges.
    for record in records:
        arrays = _validation_fold_arrays(record, sample)
        candidates = enumerate_score_binnings(
            arrays["background_scores"],
            arrays["background_weights"],
            min_bins=1,
            max_bins=5,
        )
        mapping = {
            tuple(int(index) for index in candidate["base_edge_indices"]): candidate
            for candidate in candidates
        }
        maps.append(mapping)
        keys = set(mapping)
        common = keys if common is None else common.intersection(keys)
    ordered = sorted(common or set(), key=lambda key: (len(key), key))
    return maps, ordered


def _valid_background_channel(channel: Mapping[str, Any]) -> bool:
    background = np.asarray(channel["background"], dtype=float)
    raw = np.asarray(channel["background_raw_entries"], dtype=int)
    neff = np.asarray(channel["background_effective_entries"], dtype=float)
    return bool(np.all(background > 0.0) and np.all(raw >= 25) and np.all(neff >= 10.0))


def _valid_signal_channel(channel: Mapping[str, Any]) -> bool:
    """Return whether pyhf can interpret a signed validation signal template."""

    signal = np.asarray(channel["signal"], dtype=float)
    return bool(
        signal.size
        and np.all(np.isfinite(signal))
        and np.all(signal >= 0.0)
        and float(np.sum(signal)) > 0.0
    )


def _poi_bounds_for_channels(channels: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    signal = float(
        sum(np.sum(np.asarray(channel["signal"], dtype=float)) for channel in channels)
    )
    background = float(
        sum(np.sum(np.asarray(channel["background"], dtype=float)) for channel in channels)
    )
    if signal <= 0.0 or background < 0.0:
        return (0.0, 1.0e4)
    estimate = exact_cls_signal_upper_limit(background) / signal
    return _poi_bounds_from_estimate(estimate)


def _poi_bounds_from_estimate(estimate: float) -> tuple[float, float]:
    """Return a pyhf search bracket tied to the expected limit scale.

    ``upper_limit`` probes the configured POI upper bound before starting its
    root search.  A large absolute floor therefore forces fits at physically
    irrelevant signal strengths and can make SLSQP drive MC-stat nuisance
    parameters to their bounds.  Ten times the analytic one-bin estimate
    safely brackets the expected bands while avoiding that numerical regime.
    """

    value = float(estimate)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("the pyhf POI estimate must be finite and positive")
    return (0.0, PYHF_POI_BRACKET_MULTIPLIER * value)


def _select_shape_candidate(
    records: Sequence[Mapping[str, Any]],
    sample: EventSample | ShapePoint,
    candidate_maps: Sequence[Mapping[tuple[int, ...], Mapping[str, Any]]],
    common_candidates: Sequence[tuple[int, ...]],
) -> dict[str, Any]:
    evaluated = []
    for key in common_candidates:
        if len(key) - 1 < 2:
            continue
        channels = []
        fold_edges = []
        valid_background = True
        valid_signal = True
        for rotation, (record, mapping) in enumerate(zip(records, candidate_maps)):
            arrays = _validation_fold_arrays(record, sample)
            edges = mapping[key]["edges"]
            channel = build_pyhf_channel(
                f"validation_fold{rotation}",
                arrays["signal_scores"],
                arrays["signal_weights"],
                arrays["background_scores"],
                arrays["background_weights"],
                edges,
            )
            if not _valid_background_channel(channel):
                valid_background = False
                break
            if not _valid_signal_channel(channel):
                valid_signal = False
                break
            channels.append(channel)
            fold_edges.append(list(map(float, edges)))
        if not valid_background:
            fit = {"status": "invalid_background", "expected_median": None}
        elif not valid_signal:
            fit = {"status": "invalid_signal", "expected_median": None}
        else:
            fit = pyhf_combined_limit(
                channels,
                include_staterror=True,
                poi_bounds=_poi_bounds_for_channels(channels),
            )
        evaluated.append(
            {
                "base_edge_indices": list(key),
                "n_bins": len(key) - 1,
                "fold_edges": fold_edges,
                "valid": bool(
                    valid_background
                    and valid_signal
                    and fit.get("status") == "ok"
                ),
                "expected_limit_fb": fit.get("expected_median"),
                "fit_status": fit.get("status"),
                "fit_error": fit.get("error"),
            }
        )
    valid = [
        row
        for row in evaluated
        if row["valid"]
        and row["expected_limit_fb"] is not None
        and math.isfinite(float(row["expected_limit_fb"]))
    ]
    if not valid:
        numerical_failures = [
            row
            for row in evaluated
            if row.get("fit_status")
            not in {"ok", "invalid_background", "invalid_signal"}
        ]
        if numerical_failures:
            return {
                "status": "pyhf_failed",
                "error": "one or more validation-binning pyhf fits failed numerically",
                "failed_candidates": numerical_failures,
                "candidates": evaluated,
            }
        invalid_signal = [
            row for row in evaluated if row.get("fit_status") == "invalid_signal"
        ]
        if invalid_signal:
            return {
                "status": "invalid_signal",
                "error": (
                    "no 2--5-bin candidate has non-negative validation signal "
                    "templates with positive sensitivity in every fold"
                ),
                "invalid_signal_candidates": invalid_signal,
                "candidates": evaluated,
            }
        return {
            "status": "failed",
            "error": "no 2--5-bin candidate satisfies every validation-fold MC constraint",
            "candidates": evaluated,
        }
    minimum = min(float(row["expected_limit_fb"]) for row in valid)
    near = [row for row in valid if float(row["expected_limit_fb"]) <= minimum * 1.01]
    selected = min(
        near,
        key=lambda row: (
            int(row["n_bins"]),
            float(row["expected_limit_fb"]),
            tuple(row["base_edge_indices"]),
        ),
    )
    # Build one deterministic, nested validation-only merge at each coarser
    # bin count.  The test set may decide only how far down this predeclared
    # hierarchy to move, never which alternative edges to use.
    fallback = [selected]
    current_indices = set(selected["base_edge_indices"])
    for target_bins in range(int(selected["n_bins"]) - 1, 1, -1):
        nested = [
            row
            for row in valid
            if int(row["n_bins"]) == target_bins
            and set(row["base_edge_indices"]).issubset(current_indices)
        ]
        if not nested:
            continue
        chosen = min(
            nested,
            key=lambda row: (
                float(row["expected_limit_fb"]),
                tuple(row["base_edge_indices"]),
            ),
        )
        fallback.append(chosen)
        current_indices = set(chosen["base_edge_indices"])
    one_bin_keys = [key for key in common_candidates if len(key) - 1 == 1]
    if one_bin_keys:
        nested_one_bin_keys = [key for key in one_bin_keys if set(key).issubset(current_indices)]
        key = (nested_one_bin_keys or one_bin_keys)[0]
        fold_edges = []
        valid_one_bin = True
        for rotation, (record, mapping) in enumerate(zip(records, candidate_maps)):
            arrays = _validation_fold_arrays(record, sample)
            edges = mapping[key]["edges"]
            channel = build_pyhf_channel(
                f"validation_fold{rotation}_onebin",
                arrays["signal_scores"],
                arrays["signal_weights"],
                arrays["background_scores"],
                arrays["background_weights"],
                edges,
            )
            valid_one_bin = valid_one_bin and _valid_background_channel(channel)
            fold_edges.append(list(map(float, edges)))
        if valid_one_bin:
            fallback.append(
                {
                    "base_edge_indices": list(key),
                    "n_bins": 1,
                    "fold_edges": fold_edges,
                    "valid": True,
                    "expected_limit_fb": None,
                    "fit_status": "fallback_only",
                    "fit_error": None,
                }
            )
    return {
        "status": "ok",
        "selected": selected,
        "minimum_expected_limit_fb": minimum,
        "fallback_hierarchy": fallback,
        "candidates": evaluated,
    }


def _scaled_background_channels(
    channels: Sequence[Mapping[str, Any]], factor: float
) -> list[dict[str, Any]]:
    scaled = []
    for channel in channels:
        item = dict(channel)
        item["background"] = np.asarray(channel["background"], dtype=float) * factor
        item["background_staterror"] = (
            np.asarray(channel["background_staterror"], dtype=float) * factor
        )
        scaled.append(item)
    return scaled


def _clear_parameter_caches(
    records: Sequence[Mapping[str, Any]], point_id: str | None = None
) -> None:
    """Discard point-local parameterized background scores after use."""

    for record in records:
        for cache_name in ("_validation_parameter_cache", "_test_parameter_cache"):
            cache = record.get(cache_name)
            if not isinstance(cache, dict):
                continue
            if point_id is None:
                cache.clear()
            else:
                cache.pop(point_id, None)


def _evaluate_shape_point(
    sample: EventSample | ShapePoint,
    records: Sequence[Mapping[str, Any]],
    *,
    shared_candidates: tuple[
        list[dict[tuple[int, ...], Mapping[str, Any]]], list[tuple[int, ...]]
    ]
    | None,
) -> dict[str, Any]:
    """Evaluate the frozen-validation pyhf prescription for one grid point."""

    parameterized = any(record["validation"].get("parameterized") for record in records)
    _clear_parameter_caches(records)
    try:
        if parameterized:
            candidate_maps, common_candidates = _candidate_maps_for_validation(records, sample)
        else:
            if shared_candidates is None:
                raise RuntimeError("Non-parameterized shape evaluation lacks shared candidates")
            candidate_maps, common_candidates = shared_candidates
        test_arrays = [_test_fold_arrays(record, sample) for record in records]
        one_bin_signal = [float(np.sum(arrays["signal_weights"])) for arrays in test_arrays]
        one_bin_background = [
            float(np.sum(arrays["background_weights"])) for arrays in test_arrays
        ]
        one_bin_signal_sumw2 = [
            float(np.sum(np.square(arrays["signal_weights"]))) for arrays in test_arrays
        ]
        one_bin_background_sumw2 = [
            float(np.sum(np.square(arrays["background_weights"]))) for arrays in test_arrays
        ]
        total_signal = float(sum(one_bin_signal))
        total_background = float(sum(one_bin_background))
        one_bin_estimate = (
            exact_cls_signal_upper_limit(total_background) / total_signal
            if total_signal > 0.0 and total_background >= 0.0
            else 100.0
        )
        valid_one_bin_signal = bool(
            np.all(np.isfinite(one_bin_signal))
            and np.all(np.asarray(one_bin_signal, dtype=float) > 0.0)
        )
        valid_one_bin_background = bool(
            np.all(np.isfinite(one_bin_background))
            and np.all(np.asarray(one_bin_background, dtype=float) > 0.0)
        )
        if not valid_one_bin_signal:
            one_bin = {
                "status": "invalid_signal",
                "expected_median": None,
                "error": (
                    "one or more held-out folds has non-positive signed signal "
                    "sensitivity"
                ),
            }
        elif not valid_one_bin_background:
            one_bin = {
                "status": "invalid_background",
                "expected_median": None,
                "error": (
                    "one or more held-out folds has non-positive signed background yield"
                ),
            }
        else:
            one_bin = pyhf_one_bin_limit(
                one_bin_signal,
                one_bin_background,
                include_staterror=False,
                poi_bounds=_poi_bounds_from_estimate(one_bin_estimate),
            )
        selection = _select_shape_candidate(
            records,
            sample,
            candidate_maps,
            common_candidates,
        )
        includes_postfit_hhhbb = any(
            record.get("postfit_hhhbb_test") is not None for record in records
        )
        base = {
            "point_id": sample.point_id,
            "c3": sample.c3,
            "d4": sample.d4,
            "status": selection["status"],
            "validation_binning": selection,
            "pyhf_one_bin_control": one_bin,
            "pyhf_one_bin_sigma95_fb": one_bin.get("expected_median"),
            "one_bin_signal_sumw2": one_bin_signal_sumw2,
            "one_bin_background_sumw2": one_bin_background_sumw2,
        }
        if includes_postfit_hhhbb:
            base.update(
                {
                    "signal_components": "hhhh,hhhbb",
                    "limit_parameter": "common-signal-strength",
                    "limit_cross_section_basis": "equivalent-hhhh-fb",
                    "hhhh_xsec_fb": float(sample.xsec_fb),
                    "postfit_hhhbb_in_training": False,
                    "postfit_hhhbb_in_threshold_optimization": False,
                    "postfit_hhhbb_in_shape_binning_optimization": False,
                    "postfit_hhhbb_role": (
                        "held-out-test-template-after-frozen-hhhh-validation-binning"
                    ),
                }
            )
        if one_bin.get("status") == "invalid_signal":
            return {
                **base,
                "status": "invalid_signal",
                "terminal_reason": "invalid_one_bin_signal",
            }
        if one_bin.get("status") == "invalid_background":
            return {
                **base,
                "status": "failed_nonpositive_test_bin",
                "terminal_reason": "invalid_one_bin_background",
            }
        if one_bin.get("status") != "ok":
            return {
                **base,
                "status": "pyhf_failed",
                "terminal_reason": "one_bin_control_failed",
            }
        if selection["status"] == "pyhf_failed":
            return {
                **base,
                "status": "pyhf_failed",
                "terminal_reason": "validation_shape_fit_failed",
            }
        if selection["status"] != "ok":
            # This is a valid physics-terminal result: the validation MC does
            # not support a statistically admissible score shape.
            return {**base, "terminal_reason": "invalid_binning"}

        chosen_channels = None
        chosen = None
        attempts = []
        for fallback_level, candidate in enumerate(selection["fallback_hierarchy"]):
            channels = []
            positive_background = True
            positive_signal = True
            for rotation, record in enumerate(records):
                arrays = _test_fold_arrays(record, sample)
                edges = candidate["fold_edges"][rotation]
                channel = build_pyhf_channel(
                    f"test_fold{rotation}",
                    arrays["signal_scores"],
                    arrays["signal_weights"],
                    arrays["background_scores"],
                    arrays["background_weights"],
                    edges,
                )
                if np.any(np.asarray(channel["background"], dtype=float) <= 0.0):
                    positive_background = False
                if np.any(np.asarray(channel["signal"], dtype=float) <= 0.0):
                    positive_signal = False
                channels.append(channel)
            attempts.append(
                {
                    "fallback_level": fallback_level,
                    "n_bins": candidate["n_bins"],
                    "positive_test_background": positive_background,
                    "positive_test_signal": positive_signal,
                }
            )
            if positive_background and positive_signal:
                chosen_channels = channels
                chosen = candidate
                break
        if chosen_channels is None:
            return {
                **base,
                "status": "failed_nonpositive_test_bin",
                "terminal_reason": "nonpositive_test_bin",
                "test_binning_attempts": attempts,
            }

        poi_bounds = _poi_bounds_for_channels(chosen_channels)
        shape_no_stat = pyhf_combined_limit(
            chosen_channels, include_staterror=False, poi_bounds=poi_bounds
        )
        shape_with_stat = pyhf_combined_limit(
            chosen_channels, include_staterror=True, poi_bounds=poi_bounds
        )
        channels_down = _scaled_background_channels(chosen_channels, 0.25)
        channels_up = _scaled_background_channels(chosen_channels, 4.0)
        background_down = pyhf_combined_limit(
            channels_down,
            include_staterror=True,
            poi_bounds=_poi_bounds_for_channels(channels_down),
        )
        background_up = pyhf_combined_limit(
            channels_up,
            include_staterror=True,
            poi_bounds=_poi_bounds_for_channels(channels_up),
        )
        return {
            **base,
            "status": (
                "ok"
                if all(
                    fit.get("status") == "ok"
                    for fit in (
                        one_bin,
                        shape_no_stat,
                        shape_with_stat,
                        background_down,
                        background_up,
                    )
                )
                else "pyhf_failed"
            ),
            "bin_count": int(chosen["n_bins"]),
            "used_fallback": bool(attempts[-1]["fallback_level"] > 0),
            "fallback_level": int(attempts[-1]["fallback_level"]),
            "fold_bin_edges": chosen["fold_edges"],
            "test_binning_attempts": attempts,
            "pyhf_one_bin_control": one_bin,
            "pyhf_shape_no_mcstat": shape_no_stat,
            "pyhf_shape_with_mcstat": shape_with_stat,
            "pyhf_shape_background_x0p25": background_down,
            "pyhf_shape_background_x4": background_up,
            "pyhf_one_bin_sigma95_fb": one_bin.get("expected_median"),
            "shape_sigma95_no_mcstat_fb": shape_no_stat.get("expected_median"),
            "shape_sigma95_fb": shape_with_stat.get("expected_median"),
            "shape_sigma95_background_x0p25_fb": background_down.get("expected_median"),
            "shape_sigma95_background_x4_fb": background_up.get("expected_median"),
            "one_bin_signal_sumw2": one_bin_signal_sumw2,
            "one_bin_background_sumw2": one_bin_background_sumw2,
        }
    finally:
        _clear_parameter_caches(records, sample.point_id)


def _shape_warning_records(caught: Sequence[warnings.WarningMessage]) -> list[dict[str, Any]]:
    """Return a compact, deterministic representation of worker warnings."""

    result = []
    seen = set()
    for warning in caught:
        item = {
            "category": warning.category.__name__,
            "message": str(warning.message),
            "filename": str(warning.filename),
            "lineno": int(warning.lineno),
        }
        key = tuple(item.items())
        if key not in seen:
            seen.add(key)
            result.append(item)
        if len(result) >= 20:
            break
    return result


def _evaluate_shape_point_payload(
    sample: EventSample | ShapePoint,
    records: Sequence[Mapping[str, Any]],
    *,
    shared_candidates: tuple[
        list[dict[tuple[int, ...], Mapping[str, Any]]], list[tuple[int, ...]]
    ]
    | None,
) -> dict[str, Any]:
    """Evaluate one point and return diagnostics without printing or writing."""

    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            row = _evaluate_shape_point(
                sample, records, shared_candidates=shared_candidates
            )
            return {
                "kind": "result",
                "row": row,
                "warnings": _shape_warning_records(caught),
                "elapsed_seconds": float(time.monotonic() - started),
            }
        except Exception as error:
            return {
                "kind": "worker_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "warnings": _shape_warning_records(caught),
                "elapsed_seconds": float(time.monotonic() - started),
            }


_SHAPE_WORKER_STATE: dict[str, Any] | None = None


def _shape_worker(point_index: int) -> dict[str, Any]:
    """Fork-worker entry point; all I/O remains in the parent process."""

    state = _SHAPE_WORKER_STATE
    if state is None:
        raise RuntimeError("Shape worker was started without its evaluation state")
    for variable in SHAPE_THREAD_ENVIRONMENT:
        os.environ[variable] = "1"
    return _evaluate_shape_point_payload(
        state["points"][int(point_index)],
        state["records"],
        shared_candidates=state["shared_candidates"],
    )


def _shutdown_shape_executor(
    executor: ProcessPoolExecutor,
    *,
    wait_for_workers: bool,
    cancel_futures: bool,
) -> None:
    """Shut down an executor on Python versions with or without cancel_futures."""

    try:
        executor.shutdown(
            wait=wait_for_workers,
            cancel_futures=cancel_futures,
        )
    except TypeError as error:
        if "cancel_futures" not in str(error):
            raise
        executor.shutdown(wait=wait_for_workers)


def _terminate_shape_executor(executor: ProcessPoolExecutor) -> None:
    """Stop active fork workers promptly after an interrupted shape stage."""

    terminate_workers = getattr(executor, "terminate_workers", None)
    if callable(terminate_workers):
        terminate_workers()
        return
    # Python <=3.13 has no public immediate-termination API.  Capture the
    # worker objects before shutdown clears the executor's internal mapping,
    # terminate them, then join so interpreter exit cannot wait on long fits.
    processes = list((getattr(executor, "_processes", None) or {}).values())
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except (OSError, ValueError):
            pass
    _shutdown_shape_executor(
        executor,
        wait_for_workers=False,
        cancel_futures=True,
    )
    for process in processes:
        try:
            process.join(timeout=5.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1.0)
        except (OSError, ValueError):
            pass


def _shape_point_descriptors(grid_samples: Sequence[EventSample]) -> list[ShapePoint]:
    points = []
    for sample in grid_samples:
        if sample.point_id is None or sample.c3 is None or sample.d4 is None:
            raise ValueError("Shape likelihoods require named c3/d4 grid samples")
        points.append(
            ShapePoint(
                sample_id=sample.sample_id,
                point_id=sample.point_id,
                c3=float(sample.c3),
                d4=float(sample.d4),
                xsec_fb=float(sample.xsec_fb),
            )
        )
    return points


def _shape_fingerprint(
    *,
    strategy: str,
    profile: str,
    observable_set: str,
    n_folds: int,
    seed: int,
    source_commit: str,
    fold_digest: str,
    normalization_inputs: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    package_versions: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    study_mode: str = "full",
) -> str:
    """Bind reusable shape checkpoints to every physics-relevant input."""

    models = []
    for record in records:
        path = Path(str(record["model"])).resolve()
        models.append(
            {
                "fold": int(record["rotation"]),
                "path": str(path),
                "sha256": _sha256(path),
                "parameters": record.get("parameters", {}),
            }
        )
    payload = {
        "checkpoint_version": SHAPE_CHECKPOINT_VERSION,
        "shape_orchestration_version": SHAPE_ORCHESTRATION_VERSION,
        "method_version": METHOD_VERSION,
        "study_mode": str(study_mode),
        "shape_algorithm": {
            "backend": "pyhf-asymptotic-numpy",
            "confidence_level": 0.95,
            "score_quantiles": [0.0, 0.50, 0.75, 0.90, 0.97, 1.0],
            "min_bins": 1,
            "max_bins": 5,
            "min_background_raw": 25,
            "min_background_neff": 10.0,
            "include_staterror": True,
            "background_envelope": [0.25, 4.0],
            "postfit_signal_policy": (
                "hhhbb-held-out-test-template-after-frozen-hhhh-validation-binning"
            ),
            "postfit_signal_weight_basis": (
                "physical-hhhbb-weight-divided-by-point-hhhh-xsec-fb"
            ),
        },
        "strategy": str(strategy),
        "profile": str(profile),
        "observable_set": str(observable_set),
        "parameterized": any(
            record["validation"].get("parameterized") for record in records
        ),
        "n_folds": int(n_folds),
        "seed": int(seed),
        "source_commit": str(source_commit),
        "fold_assignment_sha256": str(fold_digest),
        "normalization_inputs": normalization_inputs,
        "input_hashes": dict(sorted(input_hashes.items())),
        "package_versions": dict(sorted(package_versions.items())),
        "models": models,
    }
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _shape_checkpoint_path(directory: Path, point: ShapePoint) -> Path:
    token = hashlib.sha256(point.point_id.encode("utf-8")).hexdigest()[:16]
    return directory / f"point-{token}.json"


def _shape_row_is_retryable(row: Mapping[str, Any]) -> bool:
    return str(row.get("status")) in {"pyhf_failed", "worker_error"}


def _read_shape_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    point: ShapePoint,
) -> tuple[dict[str, Any] | None, str]:
    if not path.exists():
        return None, "missing"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return None, "malformed"
    if not isinstance(payload, Mapping):
        return None, "malformed"
    point_payload = payload.get("point")
    try:
        checkpoint_c3 = float(point_payload.get("c3", math.nan)) if isinstance(point_payload, Mapping) else math.nan
        checkpoint_d4 = float(point_payload.get("d4", math.nan)) if isinstance(point_payload, Mapping) else math.nan
    except (TypeError, ValueError):
        return None, "incompatible"
    if (
        payload.get("checkpoint_version") != SHAPE_CHECKPOINT_VERSION
        or payload.get("fingerprint") != fingerprint
        or not isinstance(point_payload, Mapping)
        or point_payload.get("point_id") != point.point_id
        or checkpoint_c3 != point.c3
        or checkpoint_d4 != point.d4
    ):
        return None, "incompatible"
    row = payload.get("row")
    if not isinstance(row, Mapping) or not bool(payload.get("complete")):
        return None, "incomplete"
    result = dict(row)
    if _shape_row_is_retryable(result):
        return None, "retryable"
    return result, "reused"


def _write_shape_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    point: ShapePoint,
    payload: Mapping[str, Any],
    complete: bool,
    strategy: str = "unknown",
) -> None:
    _write_json_atomic(
        path,
        {
            "checkpoint_version": SHAPE_CHECKPOINT_VERSION,
            "fingerprint": str(fingerprint),
            "strategy": str(strategy),
            "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "point": {
                "point_id": point.point_id,
                "c3": point.c3,
                "d4": point.d4,
            },
            "complete": bool(complete),
            "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
            "warnings": payload.get("warnings", []),
            "worker_diagnostics": {
                key: payload[key]
                for key in ("kind", "error_type", "error", "traceback")
                if key in payload
            },
            "row": payload["row"],
        },
    )


def _quarantine_strategy_outputs(
    strategy_dir: Path,
    fingerprint: str,
    *,
    include_cut_preview: bool = False,
) -> Path | None:
    """Move canonical outputs from an earlier strategy run out of the way."""

    candidates = [
        strategy_dir / "per_fold_validation.csv",
        strategy_dir / "per_fold_test.csv",
        strategy_dir / "cut_results.csv",
        strategy_dir / "cut_results.json",
        strategy_dir / "cut_results_status.json",
        strategy_dir / "sm_background_cutflow.csv",
        strategy_dir / "sm_background_cutflow.json",
        strategy_dir / "sm_background_only_cutflow.csv",
        strategy_dir / "sm_signal_cutflow.csv",
        strategy_dir / "coupling_holdout",
        strategy_dir / "shape_results.csv",
        strategy_dir / "shape_results.json",
        strategy_dir / "shape_results_status.json",
        strategy_dir / "shape_results.partial.csv",
        strategy_dir / "shape_results.partial.json",
        strategy_dir / "maps",
    ]
    if include_cut_preview:
        candidates.append(strategy_dir / "cut_preview")
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = (
        strategy_dir
        / "previous_outputs"
        / f"{str(fingerprint)[:12]}-{timestamp}-{os.getpid()}-{time.time_ns()}"
    )
    for path in existing:
        relative = path.relative_to(strategy_dir)
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
    return archive


def _shape_results(
    grid_samples: Sequence[EventSample],
    records: Sequence[Mapping[str, Any]],
    *,
    strategy: str = "unknown",
    profile: str = "unknown",
    observable_set: str = EXTENDED_SCHEMA_ID,
    n_folds: int = 5,
    shape_jobs: int = 1,
    checkpoint_dir: Path | None = None,
    checkpoint_fingerprint: str | None = None,
    progress: StudyProgress | None = None,
    return_metadata: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate shape limits serially or independently per point in processes."""

    shape_jobs = int(shape_jobs)
    if shape_jobs < 1:
        raise ValueError("shape_jobs must be at least one")
    if shape_jobs > 1 and (
        os.name != "posix" or "fork" not in multiprocessing.get_all_start_methods()
    ):
        raise RuntimeError(
            "Parallel shape evaluation requires the POSIX 'fork' multiprocessing method; "
            "use --shape-jobs 1 on this platform"
        )
    if checkpoint_dir is not None and not checkpoint_fingerprint:
        raise ValueError("checkpoint_dir requires a shape checkpoint fingerprint")

    points = _shape_point_descriptors(grid_samples)
    parameterized = any(record["validation"].get("parameterized") for record in records)
    rows: list[dict[str, Any]] = []
    pending: list[int] = []
    reused = 0
    checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for index, point in enumerate(points):
        if checkpoint_dir is None:
            pending.append(index)
            continue
        cached, _ = _read_shape_checkpoint(
            _shape_checkpoint_path(checkpoint_dir, point),
            fingerprint=str(checkpoint_fingerprint),
            point=point,
        )
        if cached is None:
            pending.append(index)
        else:
            rows.append(cached)
            reused += 1

    if progress is not None:
        progress.emit(
            f"shape:{strategy}",
            "Shape queue prepared",
            strategy=strategy,
            completed=len(rows),
            total=len(points),
            resumed=reused,
            queued=len(pending),
            active_workers=0,
            shape_jobs=shape_jobs,
        )

    durations: list[float] = []
    retryable_points: list[str] = []

    def finish(index: int, payload: dict[str, Any]) -> None:
        point = points[index]
        if payload.get("kind") == "result":
            row = dict(payload["row"])
        else:
            row = {
                "point_id": point.point_id,
                "c3": point.c3,
                "d4": point.d4,
                "status": "worker_error",
                "error_type": payload.get("error_type", "WorkerError"),
                "error": payload.get("error", "shape worker failed without an error message"),
            }
            payload = dict(payload)
            payload["row"] = row
            payload["kind"] = "worker_error"
        retryable = _shape_row_is_retryable(row)
        if retryable:
            retryable_points.append(point.point_id)
        elapsed = float(payload.get("elapsed_seconds", 0.0))
        durations.append(elapsed)
        if checkpoint_dir is not None:
            _write_shape_checkpoint(
                _shape_checkpoint_path(checkpoint_dir, point),
                fingerprint=str(checkpoint_fingerprint),
                point=point,
                payload=payload,
                complete=not retryable,
                strategy=strategy,
            )
        rows.append(row)
        worker_warnings = list(payload.get("warnings", []))
        warning_summary = "; ".join(
            f"{item.get('category', 'Warning')}: {item.get('message', '')}"
            for item in worker_warnings[:2]
            if isinstance(item, Mapping)
        ) or None
        remaining = len(points) - len(rows)
        active = min(shape_jobs, max(0, remaining))
        eta = (
            float(np.mean(durations)) * remaining / max(1, shape_jobs)
            if durations and remaining
            else 0.0 if remaining == 0 else None
        )
        if progress is not None:
            progress.emit(
                f"shape:{strategy}",
                "Completed shape point",
                strategy=strategy,
                point_id=point.point_id,
                completed=len(rows),
                total=len(points),
                resumed=reused,
                active_workers=active,
                retryable=len(retryable_points),
                point_status=row.get("status"),
                bin_count=row.get("bin_count"),
                sigma95_fb=row.get("shape_sigma95_fb"),
                warnings=len(worker_warnings),
                warning_summary=warning_summary,
                eta_seconds=eta,
            )

    if shape_jobs == 1:
        shared_candidates = (
            None
            if parameterized
            else _candidate_maps_for_validation(records, grid_samples[0])
        )
        try:
            for index in pending:
                point = points[index]
                stop_heartbeat = threading.Event()
                heartbeat: threading.Thread | None = None
                if progress is not None:
                    progress.emit(
                        f"shape:{strategy}",
                        "Starting serial pyhf point",
                        strategy=strategy,
                        point_id=point.point_id,
                        completed=len(rows),
                        total=len(points),
                        resumed=reused,
                        active_workers=1,
                        queued=max(0, len(points) - len(rows) - 1),
                    )

                    def report_serial_heartbeat() -> None:
                        while not stop_heartbeat.wait(progress.interval_seconds):
                            progress.emit(
                                f"shape:{strategy}",
                                "Waiting for serial pyhf point",
                                strategy=strategy,
                                point_id=point.point_id,
                                completed=len(rows),
                                total=len(points),
                                resumed=reused,
                                active_workers=1,
                                queued=max(0, len(points) - len(rows) - 1),
                                retryable=len(retryable_points),
                            )

                    heartbeat = threading.Thread(
                        target=report_serial_heartbeat,
                        name="pyhf-shape-progress",
                        daemon=True,
                    )
                    heartbeat.start()
                try:
                    payload = _evaluate_shape_point_payload(
                        grid_samples[index], records, shared_candidates=shared_candidates
                    )
                finally:
                    stop_heartbeat.set()
                    if heartbeat is not None:
                        heartbeat.join()
                finish(index, payload)
        except KeyboardInterrupt:
            if progress is not None:
                progress.emit(
                    f"shape:{strategy}",
                    "Shape evaluation interrupted; completed checkpoints are safe to resume",
                    status="interrupted",
                    completed=len(rows),
                    total=len(points),
                    resumed=reused,
                )
            raise
    elif pending:
        # Fork once after the classifiers and held-out scores are fixed.  Tasks
        # carry only an integer point index; the immutable state is inherited
        # copy-on-write rather than pickled for every submitted point.
        worker_records = _compact_shape_records(
            records,
            observable_set=observable_set,
            profile=profile,
            n_folds=n_folds,
        )
        worker_candidates = (
            None
            if parameterized
            else _candidate_maps_for_validation(worker_records, points[0])
        )
        global _SHAPE_WORKER_STATE
        _SHAPE_WORKER_STATE = {
            "points": points,
            "records": worker_records,
            "shared_candidates": worker_candidates,
        }
        executor: ProcessPoolExecutor | None = None
        try:
            try:
                executor = ProcessPoolExecutor(
                    max_workers=shape_jobs,
                    mp_context=multiprocessing.get_context("fork"),
                )
            except (OSError, PermissionError) as error:
                raise RuntimeError(
                    "Unable to start forked pyhf workers in this execution environment; "
                    "use --shape-jobs 1 or run on a POSIX host with process semaphores"
                ) from error
            futures = {executor.submit(_shape_worker, index): index for index in pending}
            outstanding = set(futures)
            while outstanding:
                timeout = progress.interval_seconds if progress is not None else None
                done, outstanding = wait(
                    outstanding, timeout=timeout, return_when=FIRST_COMPLETED
                )
                if not done:
                    if progress is not None:
                        remaining = len(points) - len(rows)
                        eta = (
                            float(np.mean(durations)) * remaining / max(1, shape_jobs)
                            if durations
                            else None
                        )
                        progress.emit(
                            f"shape:{strategy}",
                            "Waiting for pyhf workers",
                            strategy=strategy,
                            completed=len(rows),
                            total=len(points),
                            resumed=reused,
                            active_workers=min(shape_jobs, len(outstanding)),
                            queued=max(0, len(outstanding) - shape_jobs),
                            retryable=len(retryable_points),
                            eta_seconds=eta,
                        )
                    continue
                for future in done:
                    index = futures[future]
                    try:
                        payload = future.result()
                    except Exception as error:
                        payload = {
                            "kind": "worker_error",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                            "warnings": [],
                            "elapsed_seconds": 0.0,
                        }
                    finish(index, payload)
        except KeyboardInterrupt:
            if progress is not None:
                progress.emit(
                    f"shape:{strategy}",
                    "Shape evaluation interrupted; completed checkpoints are safe to resume",
                    status="interrupted",
                    completed=len(rows),
                    total=len(points),
                    resumed=reused,
                )
            if executor is not None:
                _terminate_shape_executor(executor)
                executor = None
            raise
        finally:
            if executor is not None:
                _shutdown_shape_executor(
                    executor,
                    wait_for_workers=True,
                    cancel_futures=False,
                )
            _SHAPE_WORKER_STATE = None

    rows.sort(key=lambda row: (float(row["c3"]), float(row["d4"])))
    metadata = {
        "strategy": strategy,
        "profile": profile,
        "shape_jobs": shape_jobs,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
        "resumed_points": reused,
        "submitted_points": len(pending),
        "completed_points": len(rows),
        "retryable_points": sorted(set(retryable_points)),
        "status": "complete" if not retryable_points else "incomplete",
    }
    return (rows, metadata) if return_metadata else rows


def _run_c3d4_study_impl(
    *,
    sm_signal_specs: Sequence[Mapping[str, Any]],
    grid_signal_specs: Sequence[Mapping[str, Any]],
    background_specs: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    observable_set: str = EXTENDED_SCHEMA_ID,
    feature_profile: str | None = None,
    training_strategy: str | None = None,
    cv_folds: int = 5,
    optuna_trials: int | None = None,
    luminosity: float = 3000.0,
    seed: int = BASE_SEED,
    max_events: int | None = None,
    legacy_scan_csv: str | Path | None = None,
    repo_dir: str | Path | None = None,
    run_shape: bool | None = None,
    hash_inputs: bool = True,
    shape_jobs: int = 1,
    progress_interval: float = 30.0,
    study_mode: str = "full",
    smoke_max_events: int = 2000,
    reuse_sm_optuna_from: str | Path | None = None,
    contour_c3_range: tuple[float, float] = DEFAULT_CONTOUR_C3_RANGE,
    contour_d4_range: tuple[float, float] = DEFAULT_CONTOUR_D4_RANGE,
    contour_grid_bins: int = DEFAULT_CONTOUR_GRID_BINS,
    contour_interpolation: str = "linear",
    xsec_source_dir: str | Path | None = DEFAULT_HHHH_XSEC_SOURCE_DIR,
    xsec_overlay: bool = True,
    write_input_report: bool = False,
    hhhbb_signal_specs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Run the complete versioned study and return its machine-readable summary."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parents[1]
    source_commit = _source_commit(repo_dir)
    cv_folds = int(cv_folds)
    shape_jobs = int(shape_jobs)
    progress_interval = float(progress_interval)
    contour_c3_range = tuple(float(value) for value in contour_c3_range)
    contour_d4_range = tuple(float(value) for value in contour_d4_range)
    contour_grid_bins = int(contour_grid_bins)
    contour_interpolation = str(contour_interpolation).strip().lower()
    reuse_sm_optuna_from = (
        None
        if reuse_sm_optuna_from is None
        else Path(reuse_sm_optuna_from).expanduser().resolve()
    )
    xsec_source_dir = None if xsec_source_dir is None else Path(xsec_source_dir)
    if cv_folds != 5:
        raise ValueError("resolved-8b v2 uses exactly five rotating folds")
    if shape_jobs < 1:
        raise ValueError("shape_jobs must be at least one")
    if not math.isfinite(progress_interval) or progress_interval <= 0.0:
        raise ValueError("progress_interval must be finite and positive")
    if (
        len(contour_c3_range) != 2
        or len(contour_d4_range) != 2
        or not all(math.isfinite(value) for value in (*contour_c3_range, *contour_d4_range))
        or contour_c3_range[0] >= contour_c3_range[1]
        or contour_d4_range[0] >= contour_d4_range[1]
    ):
        raise ValueError("contour plot ranges must be finite increasing pairs")
    if contour_grid_bins < 3:
        raise ValueError("contour_grid_bins must be at least three")
    if contour_interpolation not in CONTOUR_INTERPOLATION_METHODS:
        raise ValueError(
            f"Unknown contour interpolation {contour_interpolation!r}; choose from "
            + ", ".join(CONTOUR_INTERPOLATION_METHODS)
        )
    if shape_jobs > 1 and (
        os.name != "posix" or "fork" not in multiprocessing.get_all_start_methods()
    ):
        raise ValueError(
            "shape_jobs > 1 requires the POSIX 'fork' multiprocessing method"
        )
    if shape_jobs > 1:
        # The command-line driver applies these caps before importing NumPy,
        # SciPy, pyhf or XGBoost.  Repeat the assignment here so direct Python
        # callers also pass a bounded environment to forked workers.
        for variable in SHAPE_THREAD_ENVIRONMENT:
            os.environ[variable] = "1"
    requested_mode_inputs = {
        "study_mode": study_mode,
        "feature_profile": feature_profile,
        "training_strategy": training_strategy,
        "optuna_trials": optuna_trials,
        "max_events": max_events,
        "run_shape": run_shape,
        "hash_inputs": hash_inputs,
        "smoke_max_events": smoke_max_events,
        "reuse_sm_optuna_from": (
            None if reuse_sm_optuna_from is None else str(reuse_sm_optuna_from)
        ),
        "postfit_hhhbb_signal_point_count": len(hhhbb_signal_specs or ()),
    }
    mode_policy = _resolve_study_mode(
        study_mode=study_mode,
        observable_set=observable_set,
        feature_profile=feature_profile,
        training_strategy=training_strategy,
        optuna_trials=optuna_trials,
        max_events=max_events,
        smoke_max_events=smoke_max_events,
        run_shape=run_shape,
        hash_inputs=hash_inputs,
    )
    _validate_study_output_mode(output_dir, mode_policy.name)
    feature_profile = mode_policy.feature_profile
    training_strategy = mode_policy.training_strategy
    optuna_trials = mode_policy.optuna_trials
    max_events = mode_policy.max_events
    run_shape = mode_policy.run_shape
    hash_inputs = mode_policy.hash_inputs
    hhhbb_signal_specs = tuple(hhhbb_signal_specs or ())
    if hhhbb_signal_specs and mode_policy.run_parameterized_gate:
        raise ValueError(
            "Post-fit hhhbb is not yet supported for the full parameterized-gate "
            "workflow; use fast-sm, fast-pooled, fast-parameterized, or preview mode"
        )
    if reuse_sm_optuna_from is not None and mode_policy.name != "fast-sm":
        raise ValueError("--reuse-sm-optuna-from is supported only in fast-sm mode")
    if reuse_sm_optuna_from is not None and reuse_sm_optuna_from == output_dir.resolve():
        raise ValueError("The reused Optuna source and study output directory must differ")
    if optuna_trials < 0:
        raise ValueError("optuna_trials must be non-negative")
    if observable_set == LEGACY_SCHEMA_ID and feature_profile not in (None, "corrected28"):
        raise ValueError(
            f"{observable_set} supports only corrected28, not {feature_profile}"
        )

    progress = StudyProgress(output_dir, interval_seconds=progress_interval)
    progress.emit(
        "startup",
        "Starting resolved-8b c3/d4 XGBoost v2 study",
        observable_set=observable_set,
        training_strategy=training_strategy,
        study_mode=mode_policy.name,
        result_level=mode_policy.result_level,
        shape_jobs=shape_jobs,
        progress_interval_seconds=progress_interval,
    )

    common = {
        "observable_set": observable_set,
        "luminosity": float(luminosity),
        "n_folds": cv_folds,
        "seed": int(seed),
        "max_events": max_events,
    }
    def load_progress(label: str) -> Callable[[int, int, EventSample], None]:
        def report(index: int, total: int, sample: EventSample) -> None:
            progress.emit(
                "input-loading",
                f"Loaded {label} sample",
                sample_kind=label,
                sample_id=sample.sample_id,
                completed=index,
                total=total,
            )

        return report

    sm_samples = _load_samples(
        sm_signal_specs, kind="sm_signal", progress=load_progress("SM signal"), **common
    )
    grid_samples = _load_samples(
        grid_signal_specs,
        kind="grid_signal",
        progress=load_progress("c3/d4 signal"),
        **common,
    )
    hhhbb_samples = (
        _load_samples(
            hhhbb_signal_specs,
            kind="postfit_hhhbb_signal",
            progress=load_progress("post-fit hhhbb signal"),
            **common,
        )
        if hhhbb_signal_specs
        else []
    )
    background_samples = _load_samples(
        background_specs,
        kind="background",
        progress=load_progress("background"),
        **common,
    )
    if not sm_samples:
        raise ValueError("The SM cross-fit baseline requires a dedicated SM signal sample")
    if len(grid_samples) < 3:
        raise ValueError(
            "The c3/d4 study requires at least three signal points, "
            f"found {len(grid_samples)}"
        )
    point_ids = [sample.point_id for sample in grid_samples]
    if len(set(point_ids)) != len(point_ids):
        raise ValueError(
            "The c3/d4 inputs contain duplicate coupling coordinates; provide "
            "exactly one signal sample per point"
        )
    point_coordinates = np.asarray(
        [(sample.c3, sample.d4) for sample in grid_samples], dtype=float
    )
    if point_coordinates.shape != (len(grid_samples), 2) or np.any(
        ~np.isfinite(point_coordinates)
    ):
        raise ValueError(
            "The c3/d4 signal coordinates must be finite"
        )
    if hhhbb_samples:
        hhhbb_point_ids = [sample.point_id for sample in hhhbb_samples]
        if len(set(hhhbb_point_ids)) != len(hhhbb_point_ids):
            raise ValueError(
                "The post-fit hhhbb inputs contain duplicate coupling coordinates"
            )
        missing_hhhbb = sorted(set(point_ids) - set(hhhbb_point_ids))
        extra_hhhbb = sorted(set(hhhbb_point_ids) - set(point_ids))
        if missing_hhhbb or extra_hhhbb:
            raise ValueError(
                "Post-fit hhhbb coordinates must exactly match the c3/d4 grid; "
                f"missing={missing_hhhbb}, extra={extra_hhhbb}"
            )
    if not background_samples:
        raise ValueError("The study requires at least one background source")
    all_loaded_samples = [
        *sm_samples,
        *grid_samples,
        *background_samples,
        *hhhbb_samples,
    ]
    completion_records = [
        sample.metadata.get("feature_source_completion")
        for sample in all_loaded_samples
    ]
    feature_source_completion_verified = (
        observable_set != EXTENDED_SCHEMA_ID
        or (
            bool(completion_records)
            and all(
                isinstance(record, Mapping) and record.get("verified") is True
                for record in completion_records
            )
        )
    )
    if (
        observable_set == EXTENDED_SCHEMA_ID
        and mode_policy.name
        in {
            "preview",
            "fast-sm",
            "fast-pooled",
            "fast-parameterized",
            "full",
        }
        and not feature_source_completion_verified
    ):
        raise ValueError(
            f"{mode_policy.name} mode requires verified complete extended-v2 feature "
            "sources for every sample"
        )
    uses_complete_event_samples = bool(
        mode_policy.max_events is None and feature_source_completion_verified
    )
    normalization_inputs = _normalization_metadata(
        luminosity,
        sm_samples,
        grid_samples,
        background_samples,
        hhhbb_samples,
    )
    fold_digest = _fold_assignment_digest(all_loaded_samples)
    input_hashes = {}
    all_samples = all_loaded_samples
    progress.emit("input-hashing", "Hashing study inputs", completed=0, total=len(all_samples))
    for index, sample in enumerate(all_samples, start=1):
        input_hashes[str(sample.path)] = (
            _sha256(sample.path)
            if hash_inputs
            else f"not-computed:size={sample.path.stat().st_size}:mtime_ns={sample.path.stat().st_mtime_ns}"
        )
        progress.emit(
            "input-hashing",
            "Hashed study input",
            sample_id=sample.sample_id,
            completed=index,
            total=len(all_samples),
        )

    if not mode_policy.run_profile_ablation:
        profiles: tuple[str, ...] = ()
        get_feature_contract(observable_set, str(feature_profile))
    elif observable_set == LEGACY_SCHEMA_ID:
        profiles = ("corrected28",)
    elif feature_profile is None:
        profiles = DEFAULT_PROFILES
    else:
        profiles = (feature_profile,)
    for profile in profiles:
        get_feature_contract(observable_set, profile)

    primary_base_strategy = (
        "sm-crossfit-v2"
        if training_strategy == "sm-crossfit-v2"
        else "pooled-crossfit-v2"
    )
    if mode_policy.name == "fast-pooled":
        strategies = ["pooled-crossfit-v2"]
    elif mode_policy.name == "fast-parameterized":
        strategies = ["parameterized-crossfit-v1"]
    elif training_strategy == "sm-crossfit-v2":
        strategies = ["sm-crossfit-v2"]
    else:
        strategies = ["sm-crossfit-v2", "pooled-crossfit-v2"]
    legacy = _load_legacy_baseline(None if legacy_scan_csv is None else Path(legacy_scan_csv))
    runtime_versions = _package_versions()
    shape_thread_environment = {
        variable: os.environ.get(variable) for variable in SHAPE_THREAD_ENVIRONMENT
    }
    manifest = {
        "method_version": METHOD_VERSION,
        "study_mode": mode_policy.name,
        "result_level": mode_policy.result_level,
        "physics_result_valid": mode_policy.physics_result_valid,
        # Publication readiness is earned only after every requested stage
        # completes successfully; the static mode policy alone is insufficient.
        "paper_ready": False,
        "uses_complete_event_samples": uses_complete_event_samples,
        "feature_source_completion_verified": feature_source_completion_verified,
        "mode_policy": {
            "feature_profile": mode_policy.feature_profile,
            "training_strategy": mode_policy.training_strategy,
            "optuna_trials_per_fold": mode_policy.optuna_trials,
            "max_events_per_source": mode_policy.max_events,
            "profile_ablation_enabled": mode_policy.run_profile_ablation,
            "score_shape_enabled": mode_policy.run_shape,
            "parameterized_gate_enabled": mode_policy.run_parameterized_gate,
            "coupling_holdout_enabled": mode_policy.run_coupling_holdout,
            "input_hashing_enabled": mode_policy.hash_inputs,
            "plot_watermark": mode_policy.plot_watermark,
        },
        "strategies_requested": list(strategies),
        "classifier_weight_scale_version": CLASSIFIER_WEIGHT_SCALE_VERSION,
        "classifier_weight_normalization": (
            "equal signal/background totals; combined total equals the number of "
            "nonzero-weight signal plus original background training rows"
        ),
        "status": "running",
        "source_commit": source_commit,
        "observable_set": observable_set,
        "requested_mode_inputs": requested_mode_inputs,
        "requested_feature_profile": requested_mode_inputs["feature_profile"],
        "requested_training_strategy": requested_mode_inputs["training_strategy"],
        "cv_folds": cv_folds,
        "fold_rule": "test=f, validation=(f+1)%5, train=remaining three",
        "seed": int(seed),
        "luminosity_fb_inverse": float(luminosity),
        "normalization_inputs": normalization_inputs,
        "grid_signal_point_count": len(grid_samples),
        "postfit_signal_components": {
            "hhhbb": {
                "enabled": bool(hhhbb_samples),
                "point_count": len(hhhbb_samples),
                "role": "scored only after classifier and validation thresholds are fixed",
                "included_in_training": False,
                "included_in_threshold_optimization": False,
                "limit_parameter": (
                    "common-signal-strength" if hhhbb_samples else None
                ),
            }
        },
        "fold_assignment_sha256": fold_digest,
        "inputs": [
            _sample_manifest(sample, input_hash=input_hashes[str(sample.path)])
            for sample in all_samples
        ],
        "optuna_trials_per_fold": int(optuna_trials),
        "score_shape_enabled": bool(run_shape),
        "shape_evaluation": {
            "jobs": shape_jobs,
            "progress_interval_seconds": progress_interval,
            "executor": "serial" if shape_jobs == 1 else "process-fork",
            "checkpoint_version": SHAPE_CHECKPOINT_VERSION,
            "orchestration_version": SHAPE_ORCHESTRATION_VERSION,
            "thread_environment": shape_thread_environment,
            "strategies": {},
        },
        "fixed_xgboost_parameters": FIXED_XGBOOST_PARAMS,
        "legacy_contour_plots": {
            "style_version": LEGACY_CONTOUR_STYLE_VERSION,
            "enabled": True,
            "c3_range": list(contour_c3_range),
            "d4_range": list(contour_d4_range),
            "grid_bins": contour_grid_bins,
            "interpolation": contour_interpolation,
            "xsec_overlay": bool(xsec_overlay),
            "xsec_source_dir": (
                None if xsec_source_dir is None else str(xsec_source_dir)
            ),
            "exclusion_interpolation": (
                "piecewise-linear triangulation of log10(xsec_fb/sigma95_fb)"
                if contour_interpolation == "linear"
                else "Clough-Tocher piecewise-cubic interpolation of log10(xsec_fb/sigma95_fb)"
            ),
        },
        "package_versions": runtime_versions,
        "outputs": {},
        "input_observable_report": {
            "status": "pending" if write_input_report else "disabled"
        },
    }
    _write_json_atomic(output_dir / "method_manifest.json", manifest)
    quarantine_strategies = list(strategies)
    if mode_policy.run_parameterized_gate:
        quarantine_strategies.append("parameterized-crossfit-v1")
    for stale_strategy in quarantine_strategies:
        strategy_archive = _quarantine_strategy_outputs(
            output_dir / stale_strategy,
            f"{mode_policy.name}-startup-{source_commit}",
            include_cut_preview=True,
        )
        manifest.setdefault("previous_output_archives", {})[
            stale_strategy
        ] = (
            None if strategy_archive is None else str(strategy_archive)
        )
    if mode_policy.run_parameterized_gate:
        stale_gate = output_dir / "parameterized_classifier_gate.json"
        if stale_gate.exists():
            gate_archive = (
                output_dir
                / "previous_outputs"
                / (
                    "parameterized-gate-"
                    f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
                    f"{os.getpid()}-{time.time_ns()}.json"
                )
            )
            gate_archive.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stale_gate, gate_archive)
            manifest["previous_parameterized_gate_archive"] = str(gate_archive)
        else:
            manifest["previous_parameterized_gate_archive"] = None
    _write_json_atomic(output_dir / "method_manifest.json", manifest)
    progress.emit(
        "feature-profiles",
        (
            "Starting feature-profile comparison"
            if mode_policy.run_profile_ablation
            else "Using mode-fixed feature profile without ablation"
        ),
        profiles=list(profiles),
        selected_profile=feature_profile if not profiles else None,
        study_mode=mode_policy.name,
    )

    profile_results: dict[str, Any] = {}
    if profiles:
        print(f"Comparing feature profiles: {', '.join(profiles)}")
    else:
        print(f"Using fixed {mode_policy.name} profile: {feature_profile}")
    for profile in profiles:
        progress.emit(
            "feature-profiles",
            "Starting fixed-parameter profile",
            profile=profile,
            completed_profiles=len(profile_results),
            total_profiles=len(profiles),
        )
        profile_dir = output_dir / "feature_profile_ablation" / profile
        indices = _profile_indices(observable_set, profile)
        records = []
        validation_sigmas = []
        for rotation in range(cv_folds):
            print(f"  fixed profile {profile}, fold {rotation + 1}/{cv_folds}")
            progress.emit(
                "feature-profiles",
                "Fitting and scoring profile fold",
                profile=profile,
                fold=rotation + 1,
                total_folds=cv_folds,
            )
            model, validation, metadata, params = _fit_rotation(
                sm_samples,
                grid_samples,
                background_samples,
                strategy=primary_base_strategy,
                observable_set=observable_set,
                profile=profile,
                rotation=rotation,
                n_folds=cv_folds,
                params=FIXED_XGBOOST_PARAMS,
                seed=seed + rotation,
                source_commit=source_commit,
            )
            model_path = profile_dir / "models" / f"fold_{rotation}.json"
            metadata = dict(metadata)
            metadata["normalization_inputs"] = normalization_inputs
            metadata["package_versions"] = runtime_versions
            metadata["fold_assignment_sha256"] = fold_digest
            metadata["study_mode"] = mode_policy.name
            metadata["result_level"] = mode_policy.result_level
            metadata["physics_result_valid"] = mode_policy.physics_result_valid
            attach_model_metadata(model, metadata=metadata)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(model_path)
            test = _evaluate_test_rotation(
                model,
                validation,
                grid_samples,
                background_samples,
                rotation=rotation,
                n_folds=cv_folds,
                profile_indices=indices,
            )
            records.append(
                {
                    "rotation": rotation,
                    "model": str(model_path),
                    "model_metadata": metadata,
                    "parameters": params,
                    "validation": validation,
                    "test": test,
                }
            )
            validation_sigmas.extend(
                point["sigma95_fb"] for point in validation["points"].values()
            )
            _write_json(
                profile_dir / "validation" / f"fold_{rotation}.json",
                _compact_validation(validation),
            )
            progress.emit(
                "feature-profiles",
                "Completed profile fold",
                profile=profile,
                fold=rotation + 1,
                total_folds=cv_folds,
                validation_objective=validation.get("objective"),
            )
        progress.emit(
            "aggregation",
            "Aggregating cross-fitted profile results",
            profile=profile,
            completed_folds=cv_folds,
        )
        aggregate = _aggregate_cut_results(grid_samples, [record["test"] for record in records])
        _add_baseline_ratios(aggregate, legacy)
        objective = limit_objective(validation_sigmas)
        profile_results[profile] = {
            "objective": objective,
            "records": records,
            "aggregate": aggregate,
        }
        _write_rows(profile_dir / "test_results.csv", aggregate)
        _write_json(profile_dir / "test_results.json", aggregate)
        progress.emit("maps", "Writing profile-ablation maps", profile=profile)
        _write_standard_maps(aggregate, profile_dir / "maps", profile)
        progress.emit(
            "feature-profiles",
            "Completed fixed-parameter profile",
            profile=profile,
            completed_profiles=len(profile_results),
            total_profiles=len(profiles),
            validation_objective=objective,
        )

    if profile_results:
        best_objective = min(result["objective"] for result in profile_results.values())
        near_best = [
            profile
            for profile, result in profile_results.items()
            if math.exp(result["objective"]) <= math.exp(best_objective) * 1.01
        ]
        selected_profile = min(
            near_best,
            key=lambda profile: get_feature_contract(observable_set, profile).feature_count,
        )
        selection = {
            "selection_method": "validation-ablation",
            "selected_profile": selected_profile,
            "within_one_percent_candidates": near_best,
            "validation_objectives": {
                profile: result["objective"] for profile, result in profile_results.items()
            },
        }
    else:
        selected_profile = str(feature_profile)
        near_best = [selected_profile]
        selection = {
            "selection_method": f"fixed-by-{mode_policy.name}-mode",
            "selected_profile": selected_profile,
            "within_one_percent_candidates": [],
            "validation_objectives": {},
        }
    _write_json(output_dir / "feature_profile_selection.json", selection)
    print("Selected global profile:", selected_profile)
    manifest.update(
        {
            "selected_feature_profile": selected_profile,
            "feature_names": list(
                get_feature_contract(observable_set, selected_profile).feature_names
            ),
            "feature_profile_selection": selection,
        }
    )
    _write_json_atomic(output_dir / "method_manifest.json", manifest)
    input_observable_report = None
    if write_input_report:
        progress.emit(
            "input-report",
            "Writing normalized and stacked input-observable plots",
            observable_count=len(
                get_feature_contract(observable_set, selected_profile).feature_names
            ),
        )
        input_observable_report = write_v2_input_observable_report(
            sm_samples,
            background_samples,
            output_dir,
            observable_set=observable_set,
            feature_profile=selected_profile,
            luminosity=luminosity,
        )
        manifest["input_observable_report"] = input_observable_report
        _write_json_atomic(output_dir / "method_manifest.json", manifest)
        progress.emit(
            "input-report",
            "Completed normalized and stacked input-observable plots",
            plot_count=input_observable_report["plot_count"],
        )
    reused_sm_optuna = None
    if reuse_sm_optuna_from is not None:
        reused_sm_optuna = _load_reused_sm_optuna(
            reuse_sm_optuna_from,
            observable_set=observable_set,
            profile=selected_profile,
            n_folds=cv_folds,
            seed=seed,
        )
        manifest["reused_sm_optuna"] = reused_sm_optuna
        _write_json_atomic(output_dir / "method_manifest.json", manifest)
        progress.emit(
            "tuning",
            "Loaded frozen per-fold SM Optuna parameters",
            source_study=reused_sm_optuna["source_study"],
            profile=selected_profile,
            folds=cv_folds,
        )
    progress.emit(
        "feature-profiles",
        "Selected global feature profile",
        selected_profile=selected_profile,
        within_one_percent_candidates=near_best,
    )
    fold_result_metadata = {
        **_result_labels(
            mode_policy,
            paper_ready=False,
            score_shape_included=False,
            uses_complete_event_samples=uses_complete_event_samples,
        ),
        "result_role": "cross-fit-diagnostic",
    }

    def run_shape_stage(
        strategy: str,
        strategy_dir: Path,
        records: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        fingerprint = _shape_fingerprint(
            strategy=strategy,
            profile=selected_profile,
            observable_set=observable_set,
            n_folds=cv_folds,
            seed=seed,
            source_commit=source_commit,
            fold_digest=fold_digest,
            normalization_inputs=normalization_inputs,
            input_hashes=input_hashes,
            package_versions=runtime_versions,
            records=records,
            study_mode=mode_policy.name,
        )
        checkpoint_dir = strategy_dir / "shape_checkpoints" / fingerprint
        archived_outputs = _quarantine_strategy_outputs(strategy_dir, fingerprint)
        shape_status_file = strategy_dir / "shape_results_status.json"
        manifest["shape_evaluation"]["strategies"][strategy] = {
            "status": "running",
            "shape_jobs": shape_jobs,
            "checkpoint_fingerprint": fingerprint,
            "checkpoint_dir": str(checkpoint_dir),
            "resumed_points": 0,
            "thread_environment": shape_thread_environment,
        }
        _write_json_atomic(output_dir / "method_manifest.json", manifest)
        _write_json_atomic(
            shape_status_file,
            {
                "status": "running",
                "strategy": strategy,
                "checkpoint_fingerprint": fingerprint,
                "checkpoint_dir": str(checkpoint_dir),
                "canonical_results_published": False,
                "archived_previous_outputs": (
                    None if archived_outputs is None else str(archived_outputs)
                ),
            },
        )
        print(f"Selecting pyhf score shapes for {strategy}")
        progress.emit(
            f"shape:{strategy}",
            "Starting checkpointed pyhf score-shape evaluation",
            strategy=strategy,
            shape_jobs=shape_jobs,
            checkpoint_dir=str(checkpoint_dir),
        )
        try:
            shape, metadata = _shape_results(
                grid_samples,
                records,
                strategy=strategy,
                profile=selected_profile,
                observable_set=observable_set,
                n_folds=cv_folds,
                shape_jobs=shape_jobs,
                checkpoint_dir=checkpoint_dir,
                checkpoint_fingerprint=fingerprint,
                progress=progress,
                return_metadata=True,
            )
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            manifest["shape_evaluation"]["strategies"][strategy] = {
                "status": "interrupted",
                "shape_jobs": shape_jobs,
                "checkpoint_fingerprint": fingerprint,
                "checkpoint_dir": str(checkpoint_dir),
            }
            _write_json_atomic(
                shape_status_file,
                {
                    "status": "interrupted",
                    "strategy": strategy,
                    "checkpoint_fingerprint": fingerprint,
                    "checkpoint_dir": str(checkpoint_dir),
                    "canonical_results_published": False,
                },
            )
            _write_json_atomic(output_dir / "method_manifest.json", manifest)
            raise
        except Exception as error:
            manifest["status"] = "incomplete"
            manifest["shape_evaluation"]["strategies"][strategy] = {
                "status": "error",
                "shape_jobs": shape_jobs,
                "checkpoint_fingerprint": fingerprint,
                "checkpoint_dir": str(checkpoint_dir),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write_json_atomic(
                shape_status_file,
                {
                    "status": "error",
                    "strategy": strategy,
                    "checkpoint_fingerprint": fingerprint,
                    "checkpoint_dir": str(checkpoint_dir),
                    "canonical_results_published": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            _write_json_atomic(output_dir / "method_manifest.json", manifest)
            progress.emit(
                f"shape:{strategy}",
                "Shape evaluation stopped with an error; completed checkpoints are retained",
                status="incomplete",
                strategy=strategy,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        metadata = {
            **metadata,
            "thread_environment": shape_thread_environment,
        }
        manifest["shape_evaluation"]["strategies"][strategy] = metadata
        _write_json_atomic(output_dir / "method_manifest.json", manifest)
        if metadata["status"] != "complete":
            _write_rows(strategy_dir / "shape_results.partial.csv", shape)
            _write_json(strategy_dir / "shape_results.partial.json", shape)
            manifest["status"] = "incomplete"
            _write_json_atomic(
                shape_status_file,
                {
                    "status": "incomplete",
                    "strategy": strategy,
                    "checkpoint_fingerprint": fingerprint,
                    "checkpoint_dir": str(checkpoint_dir),
                    "canonical_results_published": False,
                    "retryable_points": metadata["retryable_points"],
                },
            )
            _write_json_atomic(output_dir / "method_manifest.json", manifest)
            progress.emit(
                f"shape:{strategy}",
                "Shape evaluation is incomplete; retryable checkpoints were retained",
                status="incomplete",
                strategy=strategy,
                retryable_points=metadata["retryable_points"],
                completed=metadata["completed_points"],
                total=len(grid_samples),
            )
            raise ShapeEvaluationIncompleteError(
                f"{strategy}: retryable pyhf/worker failures for "
                f"{', '.join(metadata['retryable_points'])}"
            )
        for partial in (
            strategy_dir / "shape_results.partial.csv",
            strategy_dir / "shape_results.partial.json",
        ):
            if partial.exists():
                partial.unlink()
        _write_json_atomic(
            shape_status_file,
            {
                "status": "ready_to_publish",
                "strategy": strategy,
                "checkpoint_fingerprint": fingerprint,
                "checkpoint_dir": str(checkpoint_dir),
                "canonical_results_published": False,
                "resumed_points": metadata["resumed_points"],
            },
        )
        return shape

    def mark_shape_results_published(strategy: str, strategy_dir: Path) -> None:
        metadata = manifest["shape_evaluation"]["strategies"][strategy]
        _write_json_atomic(
            strategy_dir / "shape_results_status.json",
            {
                "status": "complete",
                "strategy": strategy,
                "checkpoint_fingerprint": metadata.get("checkpoint_fingerprint"),
                "checkpoint_dir": metadata.get("checkpoint_dir"),
                "canonical_results_published": True,
                "resumed_points": metadata.get("resumed_points", 0),
                "completed_points": metadata.get("completed_points", 0),
            },
        )

    strategy_results: dict[str, Any] = {}
    manifest["cut_previews"] = {}
    sm_cut_preview_reference: dict[tuple[float, float], float] | None = None
    indices = _profile_indices(observable_set, selected_profile)
    for strategy in strategies:
        parameterized_strategy = strategy == "parameterized-crossfit-v1"
        strategy_dir = output_dir / strategy
        archived_before_training_text = manifest.get(
            "previous_output_archives", {}
        ).get(strategy)
        archived_before_training = (
            None
            if archived_before_training_text is None
            else Path(archived_before_training_text)
        )
        records = []
        print(f"Tuning {strategy} with profile {selected_profile}")
        progress.emit(
            "strategy",
            "Starting cross-fitted training strategy",
            strategy=strategy,
            profile=selected_profile,
            total_folds=cv_folds,
        )
        for rotation in range(cv_folds):
            fold_seed = seed + rotation
            progress.emit(
                "strategy",
                "Starting strategy fold",
                strategy=strategy,
                profile=selected_profile,
                fold=rotation + 1,
                total_folds=cv_folds,
                optuna_trials=optuna_trials,
            )
            if optuna_trials:
                best_params, tuning = _tune_rotation(
                    sm_samples,
                    grid_samples,
                    background_samples,
                    strategy=strategy,
                    observable_set=observable_set,
                    profile=selected_profile,
                    rotation=rotation,
                    n_folds=cv_folds,
                    trials=optuna_trials,
                    output_dir=strategy_dir / "optuna",
                    seed=fold_seed,
                    source_commit=source_commit,
                    run_fingerprint=_run_fingerprint(
                        observable_set=observable_set,
                        profile=selected_profile,
                        strategy=strategy,
                        rotation=rotation,
                        n_folds=cv_folds,
                        seed=fold_seed,
                        source_commit=source_commit,
                        fold_digest=fold_digest,
                        normalization_inputs=normalization_inputs,
                        input_hashes=input_hashes,
                        package_versions=runtime_versions,
                        study_mode=mode_policy.name,
                    ),
                    progress=progress,
                )
            elif strategy == "sm-crossfit-v2" and reused_sm_optuna is not None:
                reused_fold = reused_sm_optuna["folds"][rotation]
                best_params = dict(reused_fold["parameters"])
                tuning = {
                    "status": "reused",
                    "source_study": reused_sm_optuna["source_study"],
                    "source_history": reused_fold["source_history"],
                    "source_history_sha256": reused_fold[
                        "source_history_sha256"
                    ],
                    "best_trial": reused_fold["best_trial"],
                    "best_value": reused_fold["best_value"],
                    "best_params": best_params,
                    "trials": [],
                }
            else:
                best_params = dict(FIXED_XGBOOST_PARAMS)
                tuning = {"status": "disabled", "best_value": None, "trials": []}
            model, validation, metadata, fitted_params = _fit_rotation(
                sm_samples,
                grid_samples,
                background_samples,
                strategy=strategy,
                observable_set=observable_set,
                profile=selected_profile,
                rotation=rotation,
                n_folds=cv_folds,
                params=best_params,
                seed=fold_seed,
                source_commit=source_commit,
            )
            model_path = strategy_dir / "models" / f"fold_{rotation}.json"
            metadata = dict(metadata)
            metadata["normalization_inputs"] = normalization_inputs
            metadata["package_versions"] = runtime_versions
            metadata["fold_assignment_sha256"] = fold_digest
            metadata["study_mode"] = mode_policy.name
            metadata["result_level"] = mode_policy.result_level
            metadata["physics_result_valid"] = mode_policy.physics_result_valid
            metadata["hyperparameter_source"] = tuning.get("status", "optimized")
            if tuning.get("status") == "reused":
                metadata["reused_optuna_source_history"] = tuning[
                    "source_history"
                ]
                metadata["reused_optuna_source_history_sha256"] = tuning[
                    "source_history_sha256"
                ]
            attach_model_metadata(model, metadata=metadata)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(model_path)
            test = _evaluate_test_rotation(
                model,
                validation,
                grid_samples,
                background_samples,
                rotation=rotation,
                n_folds=cv_folds,
                profile_indices=indices,
                parameterized=parameterized_strategy,
            )
            postfit_hhhbb_test = (
                _evaluate_postfit_signal_rotation(
                    model,
                    validation,
                    hhhbb_samples,
                    rotation=rotation,
                    n_folds=cv_folds,
                    profile_indices=indices,
                    parameterized=parameterized_strategy,
                )
                if hhhbb_samples
                else None
            )
            record = {
                "rotation": rotation,
                "model": str(model_path),
                "model_metadata": metadata,
                "parameters": fitted_params,
                "tuning": tuning,
                "validation": validation,
                "test": test,
                "postfit_hhhbb_test": postfit_hhhbb_test,
            }
            if parameterized_strategy:
                record.update(
                    {
                        # Point-dependent background scoring is intentionally
                        # delayed and cached for cutflow/shape construction.
                        "_model_object": model,
                        "_background_samples": background_samples,
                        "_profile_indices": indices,
                        "_n_folds": cv_folds,
                    }
                )
            records.append(record)
            _write_json(strategy_dir / "optuna" / f"fold_{rotation}_history.json", tuning)
            _write_json(
                strategy_dir / "validation" / f"fold_{rotation}.json",
                _compact_validation(
                    validation, result_metadata=fold_result_metadata
                ),
            )
            _write_json(
                strategy_dir / "test" / f"fold_{rotation}.json",
                {
                    "rotation": test["rotation"],
                    "points": test["points"],
                    "result_metadata": fold_result_metadata,
                },
            )
            if postfit_hhhbb_test is not None:
                _write_json(
                    strategy_dir
                    / "postfit_hhhbb"
                    / "test"
                    / f"fold_{rotation}.json",
                    {
                        "rotation": rotation,
                        "role": "postfit-signal-only",
                        "parameterized": bool(
                            postfit_hhhbb_test.get("parameterized")
                        ),
                        "included_in_training": False,
                        "included_in_threshold_optimization": False,
                        "points": postfit_hhhbb_test["points"],
                        "result_metadata": fold_result_metadata,
                    },
                )
            progress.emit(
                "strategy",
                "Completed strategy fold",
                strategy=strategy,
                fold=rotation + 1,
                total_folds=cv_folds,
                validation_objective=validation.get("objective"),
                optuna_best_value=tuning.get("best_value"),
            )
        progress.emit(
            "aggregation",
            "Aggregating cross-fitted strategy results",
            strategy=strategy,
            completed_folds=cv_folds,
        )
        aggregate = _aggregate_cut_results(grid_samples, [record["test"] for record in records])
        if hhhbb_samples:
            progress.emit(
                "postfit-signal",
                "Adding hhhbb only after classifier and threshold optimization",
                strategy=strategy,
                component="hhhbb",
                point_count=len(hhhbb_samples),
            )
            _add_postfit_hhhbb_cut_contribution(
                aggregate,
                grid_samples,
                hhhbb_samples,
                [record["postfit_hhhbb_test"] for record in records],
            )
        validation_aggregate, rotation_validation_limits = _aggregate_validation_crossfit(
            grid_samples, records
        )
        sm_background_only_cutflow = None
        sm_signal_cutflow = None
        sm_background_cutflow = None
        sm_thresholds = None
        publish_sm_point_cutflow = (
            strategy == "sm-crossfit-v2"
            or (
                mode_policy.name in {"fast-pooled", "fast-parameterized"}
                and strategy
                in {"pooled-crossfit-v2", "parameterized-crossfit-v1"}
            )
        )
        if publish_sm_point_cutflow:
            sm_background_only_cutflow, sm_thresholds = _sm_background_cutflow_rows(
                background_samples,
                records,
                luminosity=luminosity,
            )
            sm_signal_cutflow = _sm_signal_cutflow_rows(
                grid_samples,
                hhhbb_samples,
                aggregate,
                luminosity=luminosity,
                include_limit_representatives=bool(hhhbb_samples),
            )
            sm_background_cutflow = [
                *sm_signal_cutflow,
                *sm_background_only_cutflow,
            ]
            sm_aggregate_rows = [
                row
                for row in aggregate
                if abs(float(row["c3"])) < 1.0e-12
                and abs(float(row["d4"])) < 1.0e-12
            ]
            if len(sm_aggregate_rows) != 1:
                raise ValueError(
                    "SM background cutflow requires exactly one aggregate "
                    "(c3,d4)=(0,0) result"
                )
            table_background_yield = float(
                sum(row["xgboost_events"] for row in sm_background_only_cutflow)
            )
            canonical_background_yield = float(
                sm_aggregate_rows[0]["background_yield"]
            )
            if not math.isclose(
                table_background_yield,
                canonical_background_yield,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "SM per-background XGBoost yield does not close to the "
                    "canonical aggregate background yield: "
                    f"{table_background_yield} versus {canonical_background_yield}"
                )
            print()
            print(f"Classifier strategy: {strategy}")
            print(
                terminal_sm_background_cutflow_table(
                    sm_background_cutflow,
                    luminosity=luminosity,
                    thresholds=sm_thresholds,
                )
            )
        _annotate_result_rows(
            aggregate,
            mode_policy,
            uses_complete_event_samples=uses_complete_event_samples,
        )
        preview_reference = (
            None if strategy == "sm-crossfit-v2" else sm_cut_preview_reference
        )
        _add_baseline_ratios(aggregate, legacy, preview_reference)
        preview_watermark = mode_policy.plot_watermark or (
            "PRELIMINARY - FULL RUN SHAPE FIT PENDING"
            if run_shape
            else "PRELIMINARY - SINGLE-BIN CUT RESULT"
        )
        progress.emit(
            "cut-preview",
            "Publishing exact single-bin cut preview before score-shape evaluation",
            strategy=strategy,
            study_mode=mode_policy.name,
        )
        cut_preview = _publish_cut_preview(
            aggregate,
            strategy_dir,
            strategy=strategy,
            policy=mode_policy,
            watermark=preview_watermark,
            luminosity=luminosity,
            contour_c3_range=contour_c3_range,
            contour_d4_range=contour_d4_range,
            contour_grid_bins=contour_grid_bins,
            contour_interpolation=contour_interpolation,
            xsec_source_dir=xsec_source_dir,
            xsec_overlay=xsec_overlay,
        )
        cut_preview["archived_previous_outputs"] = (
            None if archived_before_training is None else str(archived_before_training)
        )
        manifest["cut_previews"][strategy] = cut_preview
        _write_json_atomic(output_dir / "method_manifest.json", manifest)
        progress.emit(
            "cut-preview",
            "Cut-limit preview is available",
            strategy=strategy,
            study_mode=mode_policy.name,
            plot=cut_preview.get("cut_exclusion_map"),
            status_file=cut_preview.get("status_file"),
        )
        if strategy == "sm-crossfit-v2":
            sm_cut_preview_reference = {
                (float(row["c3"]), float(row["d4"])): float(row["cut_sigma95_fb"])
                for row in aggregate
            }
        if run_shape:
            shape = run_shape_stage(strategy, strategy_dir, records)
        else:
            shape = [
                {
                    "point_id": sample.point_id,
                    "c3": sample.c3,
                    "d4": sample.d4,
                    "status": "disabled",
                }
                for sample in grid_samples
            ]
            manifest["shape_evaluation"]["strategies"][strategy] = {
                "status": "disabled",
                "shape_jobs": shape_jobs,
            }
        _annotate_result_rows(
            shape,
            mode_policy,
            uses_complete_event_samples=uses_complete_event_samples,
        )
        shape_by_point = {row["point_id"]: row for row in shape}
        for row in aggregate:
            shape_row = shape_by_point[row["point_id"]]
            for key in (
                "status",
                "bin_count",
                "used_fallback",
                "fallback_level",
                "pyhf_one_bin_sigma95_fb",
                "shape_sigma95_no_mcstat_fb",
                "shape_sigma95_fb",
                "shape_sigma95_background_x0p25_fb",
                "shape_sigma95_background_x4_fb",
            ):
                row[f"shape_{key}" if key == "status" else key] = shape_row.get(key)
            one_bin = shape_row.get("pyhf_one_bin_sigma95_fb")
            shape_limit = shape_row.get("shape_sigma95_fb")
            row["shape_ratio_to_pyhf_one_bin"] = (
                float(shape_limit) / float(one_bin)
                if shape_limit is not None and one_bin is not None and float(one_bin) > 0.0
                else None
            )
        coupling_holdout = None
        if mode_policy.run_coupling_holdout and parameterized_strategy:
            progress.emit(
                "coupling-holdout",
                "Starting coupling-point holdout interpolation diagnostic",
                strategy=strategy,
                point_count=len(grid_samples),
                folds=cv_folds,
            )
            coupling_holdout = _parameterized_coupling_holdout_diagnostic(
                sm_samples,
                grid_samples,
                background_samples,
                records,
                observable_set=observable_set,
                profile=selected_profile,
                n_folds=cv_folds,
                seed=seed,
                source_commit=source_commit,
                progress=progress,
            )
            coupling_holdout_dir = strategy_dir / "coupling_holdout"
            _write_rows(
                coupling_holdout_dir / "point_results.csv",
                coupling_holdout["rows"],
            )
            _write_json(
                coupling_holdout_dir / "point_results.json",
                coupling_holdout["rows"],
            )
            _write_json_atomic(
                coupling_holdout_dir / "summary.json",
                coupling_holdout["summary"],
            )
            progress.emit(
                "coupling-holdout",
                "Completed coupling-point holdout interpolation diagnostic",
                strategy=strategy,
                point_count=len(coupling_holdout["rows"]),
                median_ratio=coupling_holdout["summary"].get(
                    "median_holdout_to_event_crossfit_ratio"
                ),
            )
        strategy_results[strategy] = {
            "records": records,
            "aggregate": aggregate,
            "validation_aggregate": validation_aggregate,
            "rotation_validation_limits": rotation_validation_limits,
            "shape": shape,
            "sm_background_cutflow": sm_background_cutflow,
            "sm_background_only_cutflow": sm_background_only_cutflow,
            "sm_signal_cutflow": sm_signal_cutflow,
            "sm_thresholds": sm_thresholds,
            "coupling_holdout": coupling_holdout,
        }
        _write_rows(
            strategy_dir / "per_fold_validation.csv",
            _flatten_fold_points(
                records,
                "validation",
                result_metadata=fold_result_metadata,
            ),
        )
        _write_rows(
            strategy_dir / "per_fold_test.csv",
            _flatten_fold_points(
                records,
                "test",
                result_metadata=fold_result_metadata,
            ),
        )
        _write_rows(strategy_dir / "cut_results.csv", aggregate)
        _write_json(strategy_dir / "cut_results.json", aggregate)
        if hhhbb_samples:
            hhhbb_contribution_rows = [
                {
                    key: row.get(key)
                    for key in (
                        "point_id",
                        "c3",
                        "d4",
                        "hhhh_xsec_fb",
                        "hhhh_nominal_selected_signal_yield",
                        "hhhbb_file",
                        "hhhbb_xsec_fb",
                        "hhhbb_rate_factor",
                        "hhhbb_generated_events",
                        "hhhbb_normalisation_weight",
                        "hhhbb_feature_tree_efficiency",
                        "hhhbb_xgboost_efficiency",
                        "hhhbb_selected_signal_yield_per_fb",
                        "hhhbb_selected_signal_staterror_per_fb",
                        "hhhbb_nominal_selected_signal_yield",
                        "hhhbb_nominal_selected_signal_staterror",
                        "hhhbb_selected_raw_entries",
                        "combined_nominal_selected_signal_yield",
                        "combined_nominal_selected_signal_staterror",
                        "cut_signal_strength95",
                        "cut_sigma95_fb",
                        "excluded_cut",
                    )
                }
                for row in aggregate
            ]
            _write_rows(
                strategy_dir / "postfit_hhhbb" / "contribution_results.csv",
                hhhbb_contribution_rows,
            )
            _write_json(
                strategy_dir / "postfit_hhhbb" / "contribution_results.json",
                hhhbb_contribution_rows,
            )
        _write_rows(strategy_dir / "shape_results.csv", shape)
        _write_json(strategy_dir / "shape_results.json", shape)
        if sm_background_cutflow is not None:
            _write_rows(
                strategy_dir / "sm_background_cutflow.csv",
                sm_background_cutflow,
            )
            _write_rows(
                strategy_dir / "sm_background_only_cutflow.csv",
                sm_background_only_cutflow,
            )
            _write_rows(
                strategy_dir / "sm_signal_cutflow.csv",
                sm_signal_cutflow,
            )
            _write_json(
                strategy_dir / "sm_background_cutflow.json",
                {
                    "classifier_strategy": strategy,
                    "luminosity_fb_inverse": float(luminosity),
                    "thresholds_by_fold": sm_thresholds,
                    "signal_rows_are_excluded_from_background_total": True,
                    "signal_rows_are_alternative_coupling_hypotheses": True,
                    "rows": sm_background_cutflow,
                    "signal_rows": sm_signal_cutflow,
                    "background_rows": sm_background_only_cutflow,
                    "signal_totals_by_point": _cutflow_signal_totals_by_point(
                        sm_signal_cutflow
                    ),
                    "totals_by_role": {
                        "signal": {
                            "rows": len(sm_signal_cutflow),
                            "coupling_points": len(
                                {
                                    str(row["point_id"])
                                    for row in sm_signal_cutflow
                                }
                            ),
                            "additive_across_coupling_points": False,
                        },
                        "background": _cutflow_role_totals(
                            sm_background_only_cutflow
                        ),
                    },
                },
            )
        if run_shape:
            mark_shape_results_published(strategy, strategy_dir)
        progress.emit(
            "strategy",
            "Completed strategy outputs",
            strategy=strategy,
            completed_folds=cv_folds,
        )

    sm_result = strategy_results.get("sm-crossfit-v2")
    sm_reference = (
        {
            (float(row["c3"]), float(row["d4"])): float(
                row["cut_sigma95_fb"]
            )
            for row in sm_result["aggregate"]
        }
        if sm_result is not None
        else None
    )
    for strategy, result in strategy_results.items():
        _add_baseline_ratios(
            result["aggregate"],
            legacy,
            (
                None
                if strategy == "sm-crossfit-v2" or sm_reference is None
                else sm_reference
            ),
        )
        strategy_dir = output_dir / strategy
        _write_rows(strategy_dir / "cut_results.csv", result["aggregate"])
        _write_json(strategy_dir / "cut_results.json", result["aggregate"])
        progress.emit("maps", "Writing strategy maps", strategy=strategy)
        map_outputs = _write_standard_maps(
            result["aggregate"],
            strategy_dir / "maps",
            strategy,
            watermark=mode_policy.plot_watermark,
            legacy_contours=True,
            luminosity=luminosity,
            contour_c3_range=contour_c3_range,
            contour_d4_range=contour_d4_range,
            contour_grid_bins=contour_grid_bins,
            contour_interpolation=contour_interpolation,
            xsec_source_dir=xsec_source_dir,
            xsec_overlay=xsec_overlay,
        )
        result["map_outputs"] = map_outputs
        _write_json_atomic(
            strategy_dir / "cut_results_status.json",
            {
                "status": "complete",
                "strategy": strategy,
                "study_mode": mode_policy.name,
                "result_level": mode_policy.result_level,
                "physics_result_valid": mode_policy.physics_result_valid,
                "paper_ready": mode_policy.paper_ready,
                "cut_results_csv": str(strategy_dir / "cut_results.csv"),
                "cut_results_json": str(strategy_dir / "cut_results.json"),
                "cut_results_sha256": _sha256(
                    strategy_dir / "cut_results.json"
                ),
                "legacy_style_contours": (map_outputs or {}).get(
                    "legacy_style_contours"
                ),
            },
        )
        progress.emit("maps", "Completed strategy maps", strategy=strategy)

    gate = None
    if (
        "pooled-crossfit-v2" in strategy_results
        and mode_policy.run_parameterized_gate
    ):
        sm_validation = strategy_results["sm-crossfit-v2"]["validation_aggregate"]
        pooled_validation = strategy_results["pooled-crossfit-v2"]["validation_aggregate"]
        sm_limits = np.asarray([row["validation_cut_sigma95_fb"] for row in sm_validation])
        pooled_limits = np.asarray([row["validation_cut_sigma95_fb"] for row in pooled_validation])
        sm_indices = [
            index
            for index, row in enumerate(sm_validation)
            if abs(float(row["c3"])) < 1.0e-12 and abs(float(row["d4"])) < 1.0e-12
        ]
        if len(sm_indices) != 1:
            raise ValueError("Exactly one (c3,d4)=(0,0) grid point is required for the gate")
        gate = parameterized_gate(
            pooled_limits,
            sm_limits,
            sm_indices[0],
            strategy_results["pooled-crossfit-v2"]["rotation_validation_limits"],
            strategy_results["sm-crossfit-v2"]["rotation_validation_limits"],
        )
        _write_json(output_dir / "parameterized_classifier_gate.json", gate)
        progress.emit(
            "parameterized-gate",
            "Evaluated parameterized-classifier gate",
            passed=gate.get("passed"),
            median_ratio=gate.get("median_ratio"),
        )

    run_parameterized = bool(
        gate
        and gate.get("passed")
        and mode_policy.run_parameterized_gate
        and training_strategy in {"pooled-crossfit-v2", "parameterized-crossfit-v1"}
    )
    if run_parameterized:
        strategy = "parameterized-crossfit-v1"
        strategy_dir = output_dir / strategy
        records = []
        parameterized_trials = 30
        print(
            f"Pooled gate passed; tuning {strategy} with {parameterized_trials} trials per fold"
        )
        progress.emit(
            "strategy",
            "Parameterized gate passed; starting parameterized strategy",
            strategy=strategy,
            profile=selected_profile,
            total_folds=cv_folds,
            optuna_trials=parameterized_trials,
        )
        for rotation in range(cv_folds):
            fold_seed = seed + rotation
            progress.emit(
                "strategy",
                "Starting parameterized strategy fold",
                strategy=strategy,
                fold=rotation + 1,
                total_folds=cv_folds,
                optuna_trials=parameterized_trials,
            )
            best_params, tuning = _tune_rotation(
                sm_samples,
                grid_samples,
                background_samples,
                strategy=strategy,
                observable_set=observable_set,
                profile=selected_profile,
                rotation=rotation,
                n_folds=cv_folds,
                trials=parameterized_trials,
                output_dir=strategy_dir / "optuna",
                seed=fold_seed,
                source_commit=source_commit,
                run_fingerprint=_run_fingerprint(
                    observable_set=observable_set,
                    profile=selected_profile,
                    strategy=strategy,
                    rotation=rotation,
                    n_folds=cv_folds,
                    seed=fold_seed,
                    source_commit=source_commit,
                    fold_digest=fold_digest,
                    normalization_inputs=normalization_inputs,
                    input_hashes=input_hashes,
                    package_versions=runtime_versions,
                    study_mode=mode_policy.name,
                ),
                progress=progress,
            )
            model, validation, metadata, fitted_params = _fit_rotation(
                sm_samples,
                grid_samples,
                background_samples,
                strategy=strategy,
                observable_set=observable_set,
                profile=selected_profile,
                rotation=rotation,
                n_folds=cv_folds,
                params=best_params,
                seed=fold_seed,
                source_commit=source_commit,
            )
            model_path = strategy_dir / "models" / f"fold_{rotation}.json"
            metadata = dict(metadata)
            metadata["normalization_inputs"] = normalization_inputs
            metadata["package_versions"] = runtime_versions
            metadata["fold_assignment_sha256"] = fold_digest
            metadata["study_mode"] = mode_policy.name
            metadata["result_level"] = mode_policy.result_level
            metadata["physics_result_valid"] = mode_policy.physics_result_valid
            attach_model_metadata(model, metadata=metadata)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(model_path)
            test = _evaluate_test_rotation(
                model,
                validation,
                grid_samples,
                background_samples,
                rotation=rotation,
                n_folds=cv_folds,
                profile_indices=indices,
                parameterized=True,
            )
            record = {
                "rotation": rotation,
                "model": str(model_path),
                "model_metadata": metadata,
                "parameters": fitted_params,
                "tuning": tuning,
                "validation": validation,
                "test": test,
                # In-memory-only objects used to rescore background test and
                # validation events at each evaluated parameter point.
                "_model_object": model,
                "_background_samples": background_samples,
                "_profile_indices": indices,
                "_n_folds": cv_folds,
            }
            records.append(record)
            _write_json(strategy_dir / "optuna" / f"fold_{rotation}_history.json", tuning)
            _write_json(
                strategy_dir / "validation" / f"fold_{rotation}.json",
                _compact_validation(
                    validation, result_metadata=fold_result_metadata
                ),
            )
            _write_json(
                strategy_dir / "test" / f"fold_{rotation}.json",
                {
                    "rotation": test["rotation"],
                    "points": test["points"],
                    "result_metadata": fold_result_metadata,
                },
            )
            progress.emit(
                "strategy",
                "Completed parameterized strategy fold",
                strategy=strategy,
                fold=rotation + 1,
                total_folds=cv_folds,
                validation_objective=validation.get("objective"),
                optuna_best_value=tuning.get("best_value"),
            )

        progress.emit(
            "aggregation",
            "Aggregating cross-fitted parameterized results",
            strategy=strategy,
            completed_folds=cv_folds,
        )
        aggregate = _aggregate_cut_results(
            grid_samples, [record["test"] for record in records]
        )
        validation_aggregate, rotation_validation_limits = _aggregate_validation_crossfit(
            grid_samples, records
        )
        _annotate_result_rows(
            aggregate,
            mode_policy,
            uses_complete_event_samples=uses_complete_event_samples,
        )
        pooled_cut_reference = {
            (float(row["c3"]), float(row["d4"])): float(row["cut_sigma95_fb"])
            for row in strategy_results["pooled-crossfit-v2"]["aggregate"]
        }
        _add_baseline_ratios(aggregate, legacy, pooled_cut_reference)
        progress.emit(
            "cut-preview",
            "Publishing parameterized exact single-bin cut preview",
            strategy=strategy,
            study_mode=mode_policy.name,
        )
        cut_preview = _publish_cut_preview(
            aggregate,
            strategy_dir,
            strategy=strategy,
            policy=mode_policy,
            watermark=(
                "PRELIMINARY - FULL RUN SHAPE FIT PENDING"
                if run_shape
                else "PRELIMINARY - SINGLE-BIN CUT RESULT"
            ),
            luminosity=luminosity,
            contour_c3_range=contour_c3_range,
            contour_d4_range=contour_d4_range,
            contour_grid_bins=contour_grid_bins,
            contour_interpolation=contour_interpolation,
            xsec_source_dir=xsec_source_dir,
            xsec_overlay=xsec_overlay,
        )
        cut_preview["archived_previous_outputs"] = (
            None if archived_before_training is None else str(archived_before_training)
        )
        manifest["cut_previews"][strategy] = cut_preview
        _write_json_atomic(output_dir / "method_manifest.json", manifest)
        progress.emit(
            "cut-preview",
            "Parameterized cut-limit preview is available",
            strategy=strategy,
            plot=cut_preview.get("cut_exclusion_map"),
            status_file=cut_preview.get("status_file"),
        )
        shape = run_shape_stage(strategy, strategy_dir, records) if run_shape else [
            {
                "point_id": sample.point_id,
                "c3": sample.c3,
                "d4": sample.d4,
                "status": "disabled",
            }
            for sample in grid_samples
        ]
        if not run_shape:
            manifest["shape_evaluation"]["strategies"][strategy] = {
                "status": "disabled",
                "shape_jobs": shape_jobs,
            }
        _annotate_result_rows(
            shape,
            mode_policy,
            uses_complete_event_samples=uses_complete_event_samples,
        )
        shape_by_point = {row["point_id"]: row for row in shape}
        for row in aggregate:
            shape_row = shape_by_point[row["point_id"]]
            for key in (
                "status",
                "bin_count",
                "used_fallback",
                "fallback_level",
                "pyhf_one_bin_sigma95_fb",
                "shape_sigma95_no_mcstat_fb",
                "shape_sigma95_fb",
                "shape_sigma95_background_x0p25_fb",
                "shape_sigma95_background_x4_fb",
            ):
                row[f"shape_{key}" if key == "status" else key] = shape_row.get(key)
            one_bin = shape_row.get("pyhf_one_bin_sigma95_fb")
            shape_limit = shape_row.get("shape_sigma95_fb")
            row["shape_ratio_to_pyhf_one_bin"] = (
                float(shape_limit) / float(one_bin)
                if shape_limit is not None and one_bin is not None and float(one_bin) > 0.0
                else None
            )
        sm_parameter_reference = {
            (float(row["c3"]), float(row["d4"])): float(row["cut_sigma95_fb"])
            for row in strategy_results["sm-crossfit-v2"]["aggregate"]
        }
        pooled_reference = {
            (float(row["c3"]), float(row["d4"])): float(row["cut_sigma95_fb"])
            for row in strategy_results["pooled-crossfit-v2"]["aggregate"]
        }
        _add_baseline_ratios(aggregate, legacy, pooled_reference)
        for row in aggregate:
            key = (float(row["c3"]), float(row["d4"]))
            row["cut_ratio_to_sm"] = (
                float(row["cut_sigma95_fb"]) / sm_parameter_reference[key]
                if sm_parameter_reference[key] > 0.0
                else None
            )
            row["cut_ratio_to_pooled"] = row.get("cut_ratio_to_reference")
        strategy_results[strategy] = {
            "records": records,
            "aggregate": aggregate,
            "validation_aggregate": validation_aggregate,
            "rotation_validation_limits": rotation_validation_limits,
            "shape": shape,
        }
        _write_rows(
            strategy_dir / "per_fold_validation.csv",
            _flatten_fold_points(
                records,
                "validation",
                result_metadata=fold_result_metadata,
            ),
        )
        _write_rows(
            strategy_dir / "per_fold_test.csv",
            _flatten_fold_points(
                records,
                "test",
                result_metadata=fold_result_metadata,
            ),
        )
        _write_rows(strategy_dir / "cut_results.csv", aggregate)
        _write_json(strategy_dir / "cut_results.json", aggregate)
        _write_rows(strategy_dir / "shape_results.csv", shape)
        _write_json(strategy_dir / "shape_results.json", shape)
        if run_shape:
            mark_shape_results_published(strategy, strategy_dir)
        progress.emit(
            "strategy",
            "Completed parameterized strategy outputs",
            strategy=strategy,
            completed_folds=cv_folds,
        )
        progress.emit("maps", "Writing parameterized strategy maps", strategy=strategy)
        map_outputs = _write_standard_maps(
            aggregate,
            strategy_dir / "maps",
            strategy,
            watermark=mode_policy.plot_watermark,
            legacy_contours=True,
            luminosity=luminosity,
            contour_c3_range=contour_c3_range,
            contour_d4_range=contour_d4_range,
            contour_grid_bins=contour_grid_bins,
            contour_interpolation=contour_interpolation,
            xsec_source_dir=xsec_source_dir,
            xsec_overlay=xsec_overlay,
        )
        strategy_results[strategy]["map_outputs"] = map_outputs
        _write_json_atomic(
            strategy_dir / "cut_results_status.json",
            {
                "status": "complete",
                "strategy": strategy,
                "study_mode": mode_policy.name,
                "result_level": mode_policy.result_level,
                "physics_result_valid": mode_policy.physics_result_valid,
                "paper_ready": mode_policy.paper_ready,
                "cut_results_csv": str(strategy_dir / "cut_results.csv"),
                "cut_results_json": str(strategy_dir / "cut_results.json"),
                "cut_results_sha256": _sha256(
                    strategy_dir / "cut_results.json"
                ),
                "legacy_style_contours": (map_outputs or {}).get(
                    "legacy_style_contours"
                ),
            },
        )
        manifest["parameterized_classifier"] = {
            "status": "complete",
            "paper_ready": mode_policy.paper_ready,
            "gate": gate,
            "optuna_trials_per_fold": parameterized_trials,
            "background_replicas_per_event": 3,
            "parameter_features": [
                {"name": name, "unit": unit}
                for name, unit in PARAMETERIZED_ML_FEATURES
            ],
        }
    elif gate is not None:
        manifest["parameterized_classifier"] = {
            "status": "gate_failed" if not gate.get("passed") else "not_requested",
            "gate": gate,
        }
        progress.emit(
            "parameterized-gate",
            "Parameterized classifier was not run",
            gate_status=manifest["parameterized_classifier"]["status"],
        )
    elif (
        mode_policy.run_coupling_holdout
        and "parameterized-crossfit-v1" in strategy_results
    ):
        coupling_holdout = strategy_results["parameterized-crossfit-v1"].get(
            "coupling_holdout"
        )
        manifest["parameterized_classifier"] = {
            "status": "complete",
            "paper_ready": mode_policy.paper_ready,
            "execution": "direct-fixed-parameter-crossfit",
            "gate": None,
            "gate_applied": False,
            "optuna_trials_per_fold": 0,
            "background_replicas_per_event": 3,
            "parameter_features": [
                {"name": name, "unit": unit}
                for name, unit in PARAMETERIZED_ML_FEATURES
            ],
            "coupling_holdout": (
                None
                if coupling_holdout is None
                else coupling_holdout["summary"]
            ),
            "postfit_hhhbb_role": (
                "held-out signal contribution after classifier, threshold, "
                "and shape-binning optimization"
                if hhhbb_samples
                else "not supplied"
            ),
        }
        progress.emit(
            "parameterized-gate",
            "Completed direct fixed-parameter parameterized classifier",
            study_mode=mode_policy.name,
            coupling_holdout_status=(
                None
                if coupling_holdout is None
                else coupling_holdout["summary"].get("status")
            ),
        )
    elif not mode_policy.run_parameterized_gate:
        manifest["parameterized_classifier"] = {
            "status": "skipped_by_study_mode",
            "study_mode": mode_policy.name,
            "reason": "parameterized gate and training are full-mode stages",
        }
        progress.emit(
            "parameterized-gate",
            "Skipped parameterized classifier for quick study mode",
            study_mode=mode_policy.name,
        )

    sample_manifests = []
    for sample in [
        *sm_samples,
        *grid_samples,
        *background_samples,
        *hhhbb_samples,
    ]:
        item = _sample_manifest(sample, input_hash=input_hashes[str(sample.path)])
        sample_manifests.append(item)
    fold_assignment_file = output_dir / "fold_assignments.csv"
    _write_fold_assignments(
        fold_assignment_file,
        [*sm_samples, *grid_samples, *background_samples, *hhhbb_samples],
    )
    output_manifest = {
        "feature_profile_selection": str(output_dir / "feature_profile_selection.json"),
        "fold_assignments": str(fold_assignment_file),
        "study_progress": str(progress.path),
        "cut_previews": manifest.get("cut_previews", {}),
        "strategy_maps": {
            strategy: result.get("map_outputs")
            for strategy, result in strategy_results.items()
        },
    }
    if hhhbb_samples:
        output_manifest["postfit_hhhbb"] = {
            strategy: {
                "csv": str(
                    output_dir
                    / strategy
                    / "postfit_hhhbb"
                    / "contribution_results.csv"
                ),
                "json": str(
                    output_dir
                    / strategy
                    / "postfit_hhhbb"
                    / "contribution_results.json"
                ),
            }
            for strategy in strategy_results
        }
    if input_observable_report is not None:
        output_manifest["input_observable_report"] = input_observable_report
    coupling_holdout_strategies = {
        strategy: result["coupling_holdout"]
        for strategy, result in strategy_results.items()
        if result.get("coupling_holdout") is not None
    }
    if coupling_holdout_strategies:
        output_manifest["coupling_holdout"] = {
            strategy: {
                "csv": str(
                    output_dir
                    / strategy
                    / "coupling_holdout"
                    / "point_results.csv"
                ),
                "json": str(
                    output_dir
                    / strategy
                    / "coupling_holdout"
                    / "point_results.json"
                ),
                "summary": str(
                    output_dir
                    / strategy
                    / "coupling_holdout"
                    / "summary.json"
                ),
            }
            for strategy in coupling_holdout_strategies
        }
    cutflow_strategy = next(
        (
            strategy
            for strategy, result in strategy_results.items()
            if result.get("sm_background_cutflow") is not None
        ),
        None,
    )
    if cutflow_strategy is not None:
        cutflow_result = strategy_results[cutflow_strategy]
        sm_strategy_dir = output_dir / cutflow_strategy
        output_manifest["sm_background_cutflow"] = {
            "csv": str(sm_strategy_dir / "sm_background_cutflow.csv"),
            "json": str(sm_strategy_dir / "sm_background_cutflow.json"),
            "classifier_strategy": cutflow_strategy,
            "contents": "role-labelled-signal-and-background",
            "background_only_csv": str(
                sm_strategy_dir / "sm_background_only_cutflow.csv"
            ),
            "signal_only_csv": str(sm_strategy_dir / "sm_signal_cutflow.csv"),
            "signal_rows_are_excluded_from_background_total": True,
            "signal_rows_are_alternative_coupling_hypotheses": True,
            "limit_representative_point_count": len(
                {
                    str(row["point_id"])
                    for row in cutflow_result.get(
                        "sm_signal_cutflow", []
                    )
                    if row.get("is_limit_representative")
                }
            ),
        }
    if gate is not None:
        output_manifest["parameterized_gate"] = str(
            output_dir / "parameterized_classifier_gate.json"
        )
    manifest.update(
        {
            "status": "complete",
            "result_level": mode_policy.result_level,
            "physics_result_valid": mode_policy.physics_result_valid,
            "paper_ready": mode_policy.paper_ready,
            "selected_feature_profile": selected_profile,
            "feature_names": list(
                get_feature_contract(observable_set, selected_profile).feature_names
            ),
            "feature_profile_selection": selection,
            "strategies_completed": list(strategy_results),
            "parameterized_gate": gate,
            "inputs": sample_manifests,
            "fold_assignment_sha256": fold_digest,
            "outputs": output_manifest,
        }
    )
    _write_json_atomic(output_dir / "method_manifest.json", manifest)
    summary = {
        "manifest": manifest,
        "study_mode": mode_policy.name,
        "result_level": mode_policy.result_level,
        "paper_ready": mode_policy.paper_ready,
        "uses_complete_event_samples": uses_complete_event_samples,
        "feature_source_completion_verified": feature_source_completion_verified,
        "selected_feature_profile": selected_profile,
        "strategy_results": {
            strategy: {
                "cut_results": result["aggregate"],
                "validation_results": result["validation_aggregate"],
                "sm_background_cutflow": result.get("sm_background_cutflow"),
                "sm_background_only_cutflow": result.get(
                    "sm_background_only_cutflow"
                ),
                "sm_signal_cutflow": result.get("sm_signal_cutflow"),
                "sm_thresholds": result.get("sm_thresholds"),
                "coupling_holdout": result.get("coupling_holdout"),
            }
            for strategy, result in strategy_results.items()
        },
        "parameterized_gate": gate,
    }
    _write_json(output_dir / "study_summary.json", summary)
    progress.emit(
        "complete",
        "Resolved-8b c3/d4 XGBoost v2 study complete",
        status="complete",
        study_mode=mode_policy.name,
        result_level=mode_policy.result_level,
        paper_ready=mode_policy.paper_ready,
        selected_profile=selected_profile,
        strategies=list(strategy_results),
    )
    return summary


def _read_json_mapping(path: Path) -> dict[str, Any]:
    """Read restart metadata defensively, returning an empty mapping on damage."""

    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def write_c3d4_input_report_from_manifest(
    output_dir: str | Path,
) -> dict[str, Any]:
    """Add the input-observable gallery to an already completed v2 study."""

    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = output_dir / "method_manifest.json"
    manifest = _read_json_mapping(manifest_path)
    if not manifest:
        raise ValueError(f"No readable method_manifest.json found in {output_dir}")
    if manifest.get("status") != "complete":
        raise ValueError(
            "The input-observable gallery can be backfilled only for a completed study"
        )
    observable_set = str(manifest.get("observable_set") or "")
    feature_profile = str(manifest.get("selected_feature_profile") or "")
    get_feature_contract(observable_set, feature_profile)
    luminosity = _finite_float(
        manifest.get("luminosity_fb_inverse"), "luminosity_fb_inverse"
    )
    n_folds = int(manifest.get("cv_folds", 5))
    seed = int(manifest.get("seed", BASE_SEED))
    max_events = (manifest.get("mode_policy") or {}).get("max_events_per_source")

    specs_by_kind: dict[str, list[dict[str, Any]]] = {
        "sm_signal": [],
        "background": [],
    }
    for item in manifest.get("inputs", []):
        if not isinstance(item, Mapping) or item.get("kind") not in specs_by_kind:
            continue
        path = Path(str(item.get("path", ""))).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[1] / path).resolve()
        specs_by_kind[str(item["kind"])].append(
            {
                "path": path,
                "xsec_fb": item.get("xsec_fb"),
                "rate_factor": item.get("rate_factor", 1.0),
                "normalisation_weight": item.get("normalisation_weight"),
                "generated_events": item.get("generated_events"),
                "metadata": dict(item.get("metadata") or {}),
            }
        )
    if not specs_by_kind["sm_signal"] or not specs_by_kind["background"]:
        raise ValueError(
            "The study manifest does not contain both SM signal and background inputs"
        )

    common = {
        "observable_set": observable_set,
        "luminosity": luminosity,
        "n_folds": n_folds,
        "seed": seed,
        "max_events": max_events,
    }
    sm_samples = _load_samples(
        specs_by_kind["sm_signal"], kind="sm_signal", **common
    )
    background_samples = _load_samples(
        specs_by_kind["background"], kind="background", **common
    )
    report = write_v2_input_observable_report(
        sm_samples,
        background_samples,
        output_dir,
        observable_set=observable_set,
        feature_profile=feature_profile,
        luminosity=luminosity,
    )
    manifest["input_observable_report"] = report
    outputs = dict(manifest.get("outputs") or {})
    outputs["input_observable_report"] = report
    manifest["outputs"] = outputs
    _write_json_atomic(manifest_path, manifest)

    summary_path = output_dir / "study_summary.json"
    summary = _read_json_mapping(summary_path)
    if summary:
        summary["manifest"] = manifest
        summary["input_observable_report"] = report
        _write_json_atomic(summary_path, summary)
    return report


def _validate_study_output_mode(output_dir: Path, study_mode: str) -> None:
    """Prevent one mode from overwriting another mode's models and results."""

    manifest_path = Path(output_dir) / "method_manifest.json"
    if not manifest_path.exists():
        return
    manifest = _read_json_mapping(manifest_path)
    if not manifest:
        raise ValueError(
            f"Study output directory {output_dir} contains an unreadable or empty "
            "method_manifest.json; repair it or choose a separate --study-outdir"
        )
    # Manifests written before study modes were introduced are full studies.
    existing_mode = str(manifest.get("study_mode", "full"))
    if existing_mode != str(study_mode):
        raise ValueError(
            f"Study output directory {output_dir} belongs to {existing_mode!r} mode, "
            f"not {study_mode!r}; choose a separate --study-outdir"
        )


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def _read_replot_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.exists():
        return [], "missing"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return [], "malformed"
    if not isinstance(payload, list):
        return [], "malformed"
    if not payload:
        return [], "empty"
    if not all(isinstance(row, Mapping) for row in payload):
        return [], "malformed"
    return [dict(row) for row in payload], "ok"


def _manifest_luminosity(manifest: Mapping[str, Any]) -> float:
    normalization = manifest.get("normalization_inputs", {})
    if not isinstance(normalization, Mapping):
        normalization = {}
    for candidate in (
        manifest.get("luminosity_fb_inverse"),
        normalization.get("luminosity_fb_inverse"),
        normalization.get("luminosity"),
    ):
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    raise ValueError(
        "The study manifest does not record a positive integrated luminosity"
    )


def _manifest_grid_point_count(manifest: Mapping[str, Any]) -> int | None:
    """Return the declared grid size, retaining compatibility with v2.1 studies."""

    declared = manifest.get("grid_signal_point_count")
    if declared is None:
        return (
            EXPECTED_C3D4_SIGNAL_POINT_COUNT
            if manifest.get("method_version") is not None
            else None
        )
    try:
        count = int(declared)
    except (TypeError, ValueError):
        raise ValueError("The v2 study manifest has an invalid grid point count") from None
    if count < 3:
        raise ValueError("The v2 study manifest must declare at least three grid points")
    return count


def _manifest_grid_coordinates(
    manifest: Mapping[str, Any],
) -> list[tuple[float, float]] | None:
    coordinates = set()
    grid_records = 0
    inputs = manifest.get("inputs", [])
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        inputs = []
    for sample in inputs:
        if not isinstance(sample, Mapping) or sample.get("kind") != "grid_signal":
            continue
        grid_records += 1
        try:
            point = (float(sample["c3"]), float(sample["d4"]))
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in point):
            coordinates.add(point)
    expected_count = _manifest_grid_point_count(manifest)
    if expected_count is not None and (
        len(coordinates) != expected_count or grid_records != expected_count
    ):
        raise ValueError(
            "The v2 study manifest does not contain the required complete "
            f"{expected_count}-point c3/d4 grid "
            f"({len(coordinates)} unique usable points in {grid_records} records)"
        )
    return sorted(coordinates) if coordinates else None


def _manifest_grid_point_xsecs(
    manifest: Mapping[str, Any],
) -> dict[tuple[float, float], float] | None:
    point_xsecs: dict[tuple[float, float], float] = {}
    invalid_points = []
    inputs = manifest.get("inputs", [])
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        inputs = []
    for sample in inputs:
        if not isinstance(sample, Mapping) or sample.get("kind") != "grid_signal":
            continue
        try:
            point = (float(sample["c3"]), float(sample["d4"]))
            xsec = float(sample["xsec_fb"])
        except (KeyError, TypeError, ValueError):
            invalid_points.append(sample.get("sample_id", "unknown"))
            continue
        if (
            not all(math.isfinite(value) for value in point)
            or not math.isfinite(xsec)
            or xsec <= 0.0
            or point in point_xsecs
        ):
            invalid_points.append(sample.get("sample_id", str(point)))
            continue
        point_xsecs[point] = xsec
    expected_count = _manifest_grid_point_count(manifest)
    if expected_count is not None and (
        len(point_xsecs) != expected_count or invalid_points
    ):
        raise ValueError(
            "The v2 study manifest does not contain one positive production "
            f"cross section for each of its {expected_count} "
            f"c3/d4 points ({len(point_xsecs)} usable; invalid={invalid_points})"
        )
    return point_xsecs or None


def _manifest_expected_strategies(manifest: Mapping[str, Any]) -> list[str]:
    requested = manifest.get("strategies_requested")
    if (
        isinstance(requested, Sequence)
        and not isinstance(requested, (str, bytes))
        and requested
    ):
        expected = [
            str(strategy)
            for strategy in requested
            if str(strategy)
            in {
                "sm-crossfit-v2",
                "pooled-crossfit-v2",
                "parameterized-crossfit-v1",
            }
        ]
        if len(expected) != len(requested):
            expected = []
    else:
        expected = []
    if expected:
        parameterized = manifest.get("parameterized_classifier", {})
        if (
            isinstance(parameterized, Mapping)
            and parameterized.get("status") == "complete"
        ):
            if "parameterized-crossfit-v1" not in expected:
                expected.append("parameterized-crossfit-v1")
        return expected

    policy = manifest.get("mode_policy", {})
    strategy = (
        policy.get("training_strategy")
        if isinstance(policy, Mapping)
        else None
    )
    strategy = strategy or manifest.get("requested_training_strategy")
    if strategy == "parameterized-crossfit-v1":
        expected = ["parameterized-crossfit-v1"]
    elif strategy in (None, "sm-crossfit-v2"):
        expected = ["sm-crossfit-v2"]
    else:
        expected = ["sm-crossfit-v2", "pooled-crossfit-v2"]
    parameterized = manifest.get("parameterized_classifier", {})
    if (
        isinstance(parameterized, Mapping)
        and parameterized.get("status") == "complete"
    ):
        if "parameterized-crossfit-v1" not in expected:
            expected.append("parameterized-crossfit-v1")
    return expected


def _contour_product_count(
    contour_set: Mapping[str, Any], *, include_shape: bool
) -> int:
    count = 0
    kinds = ("cut", "shape") if include_shape else ("cut",)
    for kind in kinds:
        result = contour_set.get(kind, {})
        if not isinstance(result, Mapping):
            continue
        outputs = result.get("outputs", {})
        if not isinstance(outputs, Mapping):
            continue
        for output in outputs.values():
            if not isinstance(output, Mapping) or output.get("status") != "ok":
                continue
            png = output.get("png")
            pdf = output.get("pdf")
            if png and pdf and Path(png).exists() and Path(pdf).exists():
                count += 1
    return count


def _resolved_replot_config(
    manifest: Mapping[str, Any],
    *,
    luminosity: float | None,
    contour_c3_range: tuple[float, float] | None,
    contour_d4_range: tuple[float, float] | None,
    contour_grid_bins: int | None,
    contour_interpolation: str | None = None,
    xsec_source_dir: str | Path | None,
    xsec_overlay: bool | None,
) -> dict[str, Any]:
    saved = manifest.get("legacy_contour_plots", {})
    if not isinstance(saved, Mapping):
        saved = {}
    saved_luminosity = _manifest_luminosity(manifest)
    if luminosity is not None and not math.isclose(
        float(luminosity), saved_luminosity, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            "Requested replot luminosity does not match the study manifest: "
            f"{float(luminosity):g} versus {saved_luminosity:g} fb^-1"
        )

    def resolved_range(
        requested: tuple[float, float] | None,
        key: str,
        default: tuple[float, float],
    ) -> tuple[float, float]:
        candidate = requested if requested is not None else saved.get(key, default)
        if not isinstance(candidate, Sequence) or len(candidate) != 2:
            raise ValueError(f"Invalid saved {key} contour range")
        result = (float(candidate[0]), float(candidate[1]))
        if not all(math.isfinite(value) for value in result) or result[0] >= result[1]:
            raise ValueError(f"Contour {key} must be finite and increasing")
        return result

    c3_range = resolved_range(
        contour_c3_range, "c3_range", DEFAULT_CONTOUR_C3_RANGE
    )
    d4_range = resolved_range(
        contour_d4_range, "d4_range", DEFAULT_CONTOUR_D4_RANGE
    )
    bins = int(
        contour_grid_bins
        if contour_grid_bins is not None
        else saved.get("grid_bins", DEFAULT_CONTOUR_GRID_BINS)
    )
    if bins < 3:
        raise ValueError("Contour grid must contain at least three bins per axis")
    interpolation = str(
        contour_interpolation
        if contour_interpolation is not None
        else saved.get("interpolation", "linear")
    ).strip().lower()
    if interpolation not in CONTOUR_INTERPOLATION_METHODS:
        raise ValueError(
            f"Unknown contour interpolation {interpolation!r}; choose from "
            + ", ".join(CONTOUR_INTERPOLATION_METHODS)
        )

    saved_source = saved.get("xsec_source_dir", DEFAULT_HHHH_XSEC_SOURCE_DIR)
    if xsec_source_dir is None:
        source = None if saved_source is None else Path(saved_source)
    else:
        source = Path(xsec_source_dir)
        saved_source_mismatch = (
            "xsec_source_dir" in saved
            and (
                saved_source is None
                or Path(saved_source).expanduser().resolve()
                != source.expanduser().resolve()
            )
        )
        if saved_source_mismatch:
            raise ValueError(
                "Requested hhhh cross-section source differs from the study manifest: "
                f"{source} versus {saved_source}"
            )
    overlay = bool(saved.get("xsec_overlay", True)) if xsec_overlay is None else bool(xsec_overlay)
    return {
        "luminosity": saved_luminosity,
        "c3_range": c3_range,
        "d4_range": d4_range,
        "grid_bins": bins,
        "interpolation": interpolation,
        "xsec_source_dir": source,
        "xsec_overlay": overlay,
    }


def replot_c3d4_study_contours(
    output_dir: str | Path,
    *,
    luminosity: float | None = None,
    contour_c3_range: tuple[float, float] | None = None,
    contour_d4_range: tuple[float, float] | None = None,
    contour_grid_bins: int | None = None,
    contour_interpolation: str | None = None,
    xsec_source_dir: str | Path | None = None,
    xsec_overlay: bool | None = None,
) -> dict[str, Any]:
    """Add paper-style contours to existing v2 tables without rerunning ML."""

    output_dir = Path(output_dir)
    manifest = _read_json_mapping(output_dir / "method_manifest.json")
    if not manifest:
        raise ValueError(f"No readable v2 method manifest found under {output_dir}")
    config = _resolved_replot_config(
        manifest,
        luminosity=luminosity,
        contour_c3_range=contour_c3_range,
        contour_d4_range=contour_d4_range,
        contour_grid_bins=contour_grid_bins,
        contour_interpolation=contour_interpolation,
        xsec_source_dir=xsec_source_dir,
        xsec_overlay=xsec_overlay,
    )
    mode = str(manifest.get("study_mode", "full"))
    status = str(manifest.get("status", "unknown"))
    paper_ready = bool(manifest.get("paper_ready", False))
    shape_expected = bool(manifest.get("score_shape_enabled", mode == "full"))
    expected_coordinates = _manifest_grid_coordinates(manifest)
    expected_xsecs = _manifest_grid_point_xsecs(manifest)
    expected_strategies = _manifest_expected_strategies(manifest)
    strategy_names = list(expected_strategies)

    table_inputs: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for strategy in strategy_names:
        strategy_dir = output_dir / strategy
        canonical_path = strategy_dir / "cut_results.json"
        preview_path = strategy_dir / "cut_preview" / "cut_results.json"
        canonical_rows, canonical_status = _read_replot_rows(canonical_path)
        preview_rows, preview_status = _read_replot_rows(preview_path)
        canonical_status_payload = _read_json_mapping(
            strategy_dir / "cut_results_status.json"
        )
        preview_status_payload = _read_json_mapping(
            strategy_dir / "cut_preview" / "status.json"
        )
        canonical_expected_hash = canonical_status_payload.get(
            "cut_results_sha256"
        )
        preview_expected_hash = preview_status_payload.get("cut_results_sha256")
        if (
            canonical_status == "ok"
            and canonical_expected_hash
            and _sha256(canonical_path) != canonical_expected_hash
        ):
            canonical_rows = []
            canonical_status = "hash-mismatch"
        if (
            preview_status == "ok"
            and preview_expected_hash
            and _sha256(preview_path) != preview_expected_hash
        ):
            preview_rows = []
            preview_status = "hash-mismatch"
        table_inputs[strategy] = {
            "canonical_path": canonical_path,
            "canonical_rows": canonical_rows,
            "canonical_status": canonical_status,
            "preview_path": preview_path,
            "preview_rows": preview_rows,
            "preview_status": preview_status,
            "canonical_expected_sha256": canonical_expected_hash,
            "preview_expected_sha256": preview_expected_hash,
        }
        if preview_status in {"malformed", "empty", "hash-mismatch"} or (
            manifest.get("method_version") is not None
            and preview_status != "ok"
        ):
            issues.append(
                f"{strategy}: cut-preview table is {preview_status}"
            )
        if canonical_status != "ok":
            issues.append(
                f"{strategy}: canonical table is {canonical_status}"
            )
            continue
        cut_precheck = _legacy_contour_spec(
            canonical_rows,
            "cut",
            expected_coordinates=expected_coordinates,
            expected_xsecs=expected_xsecs,
        )
        if cut_precheck.get("status") != "ok":
            issues.append(
                f"{strategy}: canonical cut points are incomplete "
                f"({cut_precheck.get('reason', 'unknown reason')})"
            )
        elif not cut_precheck.get("band_complete", False):
            issues.append(
                f"{strategy}: canonical cut background envelope is incomplete"
            )
        if shape_expected:
            shape_precheck = _legacy_contour_spec(
                canonical_rows,
                "shape",
                expected_coordinates=expected_coordinates,
                expected_xsecs=expected_xsecs,
            )
            if shape_precheck.get("status") != "ok":
                issues.append(
                    f"{strategy}: canonical shape points are incomplete "
                    f"({shape_precheck.get('reason', 'unknown reason')})"
                )
            elif not shape_precheck.get("band_complete", False):
                issues.append(
                    f"{strategy}: canonical shape background envelope is incomplete"
                )

    if mode == "smoke":
        canonical_watermark = "NON-PHYSICS SMOKE TEST"
    elif mode == "preview":
        canonical_watermark = "PRELIMINARY - SINGLE-BIN CUT RESULT"
    elif status == "complete" and paper_ready and not issues:
        canonical_watermark = None
    else:
        canonical_watermark = "INCOMPLETE FULL RUN - RESULTS NOT FINAL"

    results: dict[str, Any] = {}
    products = 0
    for strategy in strategy_names:
        strategy_dir = output_dir / strategy
        table_input = table_inputs[strategy]
        strategy_result: dict[str, Any] = {
            "canonical_table": {
                "path": str(table_input["canonical_path"]),
                "status": table_input["canonical_status"],
                "expected_sha256": table_input["canonical_expected_sha256"],
            },
            "cut_preview_table": {
                "path": str(table_input["preview_path"]),
                "status": table_input["preview_status"],
                "expected_sha256": table_input["preview_expected_sha256"],
            },
        }
        canonical_rows = table_input["canonical_rows"]
        if table_input["canonical_status"] == "ok":
            print(
                f"Writing legacy-style canonical contours for {strategy}",
                flush=True,
            )
            contour_metadata = _write_legacy_style_contour_set(
                canonical_rows,
                strategy_dir / "maps",
                strategy,
                watermark=canonical_watermark,
                luminosity=config["luminosity"],
                c3_range=config["c3_range"],
                d4_range=config["d4_range"],
                grid_bins=config["grid_bins"],
                xsec_source_dir=config["xsec_source_dir"],
                xsec_overlay=config["xsec_overlay"],
                expected_coordinates=expected_coordinates,
                expected_xsecs=expected_xsecs,
                interpolation=config["interpolation"],
            )
            contour_manifest = strategy_dir / "maps" / "legacy_contour_manifest.json"
            _write_json_atomic(contour_manifest, contour_metadata)
            status_file = strategy_dir / "cut_results_status.json"
            status_payload = _read_json_mapping(status_file)
            if status_payload:
                status_payload["legacy_style_contours"] = contour_metadata
                status_payload.setdefault(
                    "cut_results_sha256", _sha256(table_input["canonical_path"])
                )
                _write_json_atomic(status_file, status_payload)
            strategy_result["canonical"] = contour_metadata
            canonical_products = _contour_product_count(
                contour_metadata,
                include_shape=shape_expected,
            )
            products += canonical_products
            strategy_result["canonical_product_count"] = canonical_products
            expected_canonical_products = (
                3 if config["xsec_overlay"] else 1
            ) * (2 if shape_expected else 1)
            if canonical_products < expected_canonical_products:
                issues.append(
                    f"{strategy}: canonical contour set produced "
                    f"{canonical_products}/{expected_canonical_products} plot pairs"
                )

        preview_dir = strategy_dir / "cut_preview"
        preview_rows = table_input["preview_rows"]
        if table_input["preview_status"] == "ok":
            print(
                f"Writing legacy-style cut-preview contours for {strategy}",
                flush=True,
            )
            preview_status_file = preview_dir / "status.json"
            preview_status = _read_json_mapping(preview_status_file)
            preview_watermark = str(
                preview_status.get(
                    "watermark", "PRELIMINARY - SINGLE-BIN CUT RESULT"
                )
            )
            contour_metadata = _write_legacy_style_contour_set(
                preview_rows,
                preview_dir / "maps",
                f"{strategy}_preview",
                watermark=preview_watermark,
                luminosity=config["luminosity"],
                c3_range=config["c3_range"],
                d4_range=config["d4_range"],
                grid_bins=config["grid_bins"],
                xsec_source_dir=config["xsec_source_dir"],
                xsec_overlay=config["xsec_overlay"],
                expected_coordinates=expected_coordinates,
                expected_xsecs=expected_xsecs,
                interpolation=config["interpolation"],
            )
            contour_manifest = preview_dir / "maps" / "legacy_contour_manifest.json"
            _write_json_atomic(contour_manifest, contour_metadata)
            if preview_status:
                preview_status["legacy_style_contours"] = contour_metadata
                preview_status.setdefault(
                    "cut_results_sha256", _sha256(table_input["preview_path"])
                )
                _write_json_atomic(preview_status_file, preview_status)
            strategy_result["cut_preview"] = contour_metadata
            preview_products = _contour_product_count(
                contour_metadata,
                include_shape=False,
            )
            products += preview_products
            strategy_result["cut_preview_product_count"] = preview_products
            expected_preview_products = 3 if config["xsec_overlay"] else 1
            if preview_products < expected_preview_products:
                issues.append(
                    f"{strategy}: cut-preview contour set produced "
                    f"{preview_products}/{expected_preview_products} plot pairs"
                )
        if (
            table_input["canonical_status"] != "missing"
            or table_input["preview_status"] != "missing"
            or strategy in expected_strategies
        ):
            results[strategy] = strategy_result

    replot_status = "failed" if products == 0 else ("partial" if issues else "complete")
    payload = {
        "status": replot_status,
        "style_version": LEGACY_CONTOUR_STYLE_VERSION,
        "study_mode": mode,
        "study_status": status,
        "study_paper_ready": paper_ready,
        "paper_ready": bool(paper_ready and replot_status == "complete"),
        "luminosity_fb_inverse": float(config["luminosity"]),
        "c3_range": list(config["c3_range"]),
        "d4_range": list(config["d4_range"]),
        "grid_bins": int(config["grid_bins"]),
        "interpolation": config["interpolation"],
        "xsec_overlay": bool(config["xsec_overlay"]),
        "xsec_source_dir": (
            None
            if config["xsec_source_dir"] is None
            else str(config["xsec_source_dir"])
        ),
        "expected_strategies": expected_strategies,
        "expected_point_count": (
            None if expected_coordinates is None else len(expected_coordinates)
        ),
        "point_cross_sections_bound_to_manifest": expected_xsecs is not None,
        "successful_plot_pairs": products,
        "issues": issues,
        "strategies": results,
    }
    _write_json_atomic(output_dir / "contour_replot_manifest.json", payload)
    if products == 0:
        raise ValueError(
            f"No valid v2 contour products could be written under {output_dir}; "
            "see contour_replot_manifest.json"
        )
    return payload


def _record_study_failure(
    output_dir: Path,
    requested_status: str,
    error: BaseException,
    *,
    study_mode: str | None = None,
) -> None:
    """Publish a terminal top-level state without hiding the original error."""

    progress_path = output_dir / "study_progress.json"
    manifest_path = output_dir / "method_manifest.json"
    progress_payload = _read_json_mapping(progress_path)
    manifest_payload = _read_json_mapping(manifest_path)
    unreadable_existing_manifest = manifest_path.exists() and not manifest_payload
    progress_status_before = str(progress_payload.get("status", ""))
    manifest_status_before = str(manifest_payload.get("status", ""))
    manifest_mode_before = (
        str(manifest_payload.get("study_mode", "full"))
        if manifest_payload
        else None
    )
    progress_mode_before = (
        str(progress_payload["study_mode"])
        if progress_payload.get("study_mode") is not None
        else None
    )
    foreign_output_mode = bool(
        study_mode is not None
        and any(
            existing is not None and existing != str(study_mode)
            for existing in (manifest_mode_before, progress_mode_before)
        )
    )
    terminal_statuses = {"complete", "failed", "incomplete", "interrupted"}

    status = str(requested_status)
    if status == "failed":
        # The shape stage deliberately records retryable failures as
        # ``incomplete`` before raising.  Preserve that more precise state in
        # the top-level wrapper; ordinary setup/training failures remain
        # ``failed``.
        prior_statuses = {progress_status_before}
        if "interrupted" in prior_statuses:
            status = "interrupted"
        elif "incomplete" in prior_statuses:
            status = "incomplete"

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    diagnostic = {
        "error_type": type(error).__name__,
        "error": str(error),
    }
    # A command-line typo or preprocessing failure can happen before the new
    # attempt has created its own running manifest.  Never relabel an earlier
    # successful/incomplete campaign in that case.  Keep a separate audit
    # record and update only non-terminal files owned by the current attempt.
    preserves_terminal_progress = (
        progress_status_before in terminal_statuses or foreign_output_mode
    )
    preserves_terminal_manifest = (
        manifest_status_before in terminal_statuses
        or unreadable_existing_manifest
        or foreign_output_mode
    )
    if preserves_terminal_progress or preserves_terminal_manifest:
        attempt_id = (
            f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
            f"{os.getpid()}-{time.time_ns()}"
        )
        _write_json_atomic(
            output_dir / "failed_attempts" / f"attempt-{attempt_id}.json",
            {
                "method_version": METHOD_VERSION,
                "attempted_study_mode": study_mode,
                "existing_manifest_study_mode": manifest_mode_before,
                "existing_progress_study_mode": progress_mode_before,
                "status": status,
                "recorded_at_utc": now,
                "prior_progress_status": progress_status_before or None,
                "prior_manifest_status": manifest_status_before or None,
                **diagnostic,
            },
        )
    current = progress_payload.get("current")
    current = dict(current) if isinstance(current, Mapping) else {}
    current.update(diagnostic)
    progress_payload.update(
        {
            "version": 1,
            "status": status,
            "started_at_utc": progress_payload.get("started_at_utc", now),
            "updated_at_utc": now,
            "phase": progress_payload.get("phase", "startup"),
            "message": (
                "Resolved-8b c3/d4 XGBoost v2 study interrupted"
                if status == "interrupted"
                else "Resolved-8b c3/d4 XGBoost v2 study did not complete"
            ),
            "current": current,
            "last_error": diagnostic,
            "eta_seconds": None,
        }
    )
    if study_mode is not None:
        progress_payload["study_mode"] = str(study_mode)
    manifest_payload.update(
        {
            "method_version": manifest_payload.get("method_version", METHOD_VERSION),
            "status": status,
            "last_error": {**diagnostic, "recorded_at_utc": now},
        }
    )
    if study_mode is not None:
        manifest_payload["study_mode"] = str(study_mode)
    if not preserves_terminal_progress:
        _write_json_atomic(progress_path, progress_payload)
    if not preserves_terminal_manifest:
        _write_json_atomic(manifest_path, manifest_payload)


def run_c3d4_study(
    *,
    sm_signal_specs: Sequence[Mapping[str, Any]],
    grid_signal_specs: Sequence[Mapping[str, Any]],
    background_specs: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    observable_set: str = EXTENDED_SCHEMA_ID,
    feature_profile: str | None = None,
    training_strategy: str | None = None,
    cv_folds: int = 5,
    optuna_trials: int | None = None,
    luminosity: float = 3000.0,
    seed: int = BASE_SEED,
    max_events: int | None = None,
    legacy_scan_csv: str | Path | None = None,
    repo_dir: str | Path | None = None,
    run_shape: bool | None = None,
    hash_inputs: bool = True,
    shape_jobs: int = 1,
    progress_interval: float = 30.0,
    study_mode: str = "full",
    smoke_max_events: int = 2000,
    reuse_sm_optuna_from: str | Path | None = None,
    contour_c3_range: tuple[float, float] = DEFAULT_CONTOUR_C3_RANGE,
    contour_d4_range: tuple[float, float] = DEFAULT_CONTOUR_D4_RANGE,
    contour_grid_bins: int = DEFAULT_CONTOUR_GRID_BINS,
    contour_interpolation: str = "linear",
    xsec_source_dir: str | Path | None = DEFAULT_HHHH_XSEC_SOURCE_DIR,
    xsec_overlay: bool = True,
    write_input_report: bool = False,
    hhhbb_signal_specs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Run the v2 study and always leave truthful terminal restart metadata."""

    target = Path(output_dir)
    arguments = {
        "sm_signal_specs": sm_signal_specs,
        "grid_signal_specs": grid_signal_specs,
        "background_specs": background_specs,
        "output_dir": target,
        "observable_set": observable_set,
        "feature_profile": feature_profile,
        "training_strategy": training_strategy,
        "cv_folds": cv_folds,
        "optuna_trials": optuna_trials,
        "luminosity": luminosity,
        "seed": seed,
        "max_events": max_events,
        "legacy_scan_csv": legacy_scan_csv,
        "repo_dir": repo_dir,
        "run_shape": run_shape,
        "hash_inputs": hash_inputs,
        "shape_jobs": shape_jobs,
        "progress_interval": progress_interval,
        "study_mode": study_mode,
        "smoke_max_events": smoke_max_events,
        "reuse_sm_optuna_from": (
            None if reuse_sm_optuna_from is None else str(reuse_sm_optuna_from)
        ),
        "contour_c3_range": contour_c3_range,
        "contour_d4_range": contour_d4_range,
        "contour_grid_bins": contour_grid_bins,
        "contour_interpolation": contour_interpolation,
        "xsec_source_dir": xsec_source_dir,
        "xsec_overlay": xsec_overlay,
        "write_input_report": write_input_report,
        "hhhbb_signal_specs": hhhbb_signal_specs,
    }
    try:
        return _run_c3d4_study_impl(**arguments)
    except KeyboardInterrupt as error:
        try:
            _record_study_failure(
                target, "interrupted", error, study_mode=study_mode
            )
        except Exception:
            pass
        raise
    except ShapeEvaluationIncompleteError as error:
        try:
            _record_study_failure(
                target, "incomplete", error, study_mode=study_mode
            )
        except Exception:
            pass
        raise
    except Exception as error:
        try:
            _record_study_failure(target, "failed", error, study_mode=study_mode)
        except Exception:
            pass
        raise


__all__ = [
    "CLASSIFIER_WEIGHT_SCALE_VERSION",
    "LEGACY_CONTOUR_STYLE_VERSION",
    "METHOD_VERSION",
    "EventSample",
    "ShapeEvaluationIncompleteError",
    "StudyModePolicy",
    "StudyProgress",
    "ZeroSplitModelError",
    "replot_c3d4_study_contours",
    "run_c3d4_study",
    "write_c3d4_input_report_from_manifest",
    "write_v2_input_observable_report",
]
