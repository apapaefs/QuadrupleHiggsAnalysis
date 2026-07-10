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
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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


METHOD_VERSION = "resolved-8b-c3d4-xgboost-v2"
BASE_SEED = 12345
DEFAULT_PROFILES = ("corrected28", "core52", "full91")
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
    r"run_gg_4h_[^_/]+_"
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
    if kind == "grid_signal":
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
    **kwargs: Any,
) -> list[EventSample]:
    return [_load_sample(spec, **kwargs) for spec in specs]


def _profile_indices(observable_set: str, profile: str) -> np.ndarray:
    contract = get_feature_contract(observable_set, profile)
    return np.asarray(contract.feature_indices, dtype=int)


def _fold_mask(sample: EventSample, rotation: int, split: str, n_folds: int) -> np.ndarray:
    masks = rotation_masks(sample.folds, rotation, n_folds=n_folds)
    return np.asarray(masks[split], dtype=bool)


def _balanced_weights(signal_weights: np.ndarray, background_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal_weights = np.asarray(signal_weights, dtype=float)
    background_weights = np.asarray(background_weights, dtype=float)
    signal_total = float(np.sum(signal_weights))
    background_total = float(np.sum(background_weights))
    if signal_total <= 0.0 or background_total <= 0.0:
        raise ValueError("Both classifier classes require positive absolute training weight")
    return signal_weights / signal_total, background_weights / background_total


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
    grid_points = np.asarray(
        [(sample.c3, sample.d4) for sample in grid_samples], dtype=float
    )
    for sample in background_samples:
        mask = _fold_mask(sample, rotation, "train", n_folds)
        # Absolute weights are required by XGBoost.  Their physical factors
        # retain the relative process mixture within the background class.
        weights = np.abs(sample.physical_weights[mask])
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
    signal_w, background_w = _balanced_weights(signal_w, background_w)
    X = np.concatenate([signal_X, background_X], axis=0)
    y = np.concatenate(
        [np.ones(signal_X.shape[0], dtype=np.int8), np.zeros(background_X.shape[0], dtype=np.int8)]
    )
    weights = np.concatenate([signal_w, background_w])
    return X, y, weights


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
    import xgboost as xgb

    model_params = dict(FIXED_XGBOOST_PARAMS)
    model_params.update(dict(params))
    model_params["random_state"] = int(seed)
    model_params["n_jobs"] = 1
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
    model.get_booster().feature_names = model_feature_names
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
        _, validation, _, _ = _fit_rotation(
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
        trial.set_user_attr("median_sigma95_fb", float(np.median([
            point["sigma95_fb"] for point in validation["points"].values()
        ])))
        return float(validation["objective"])

    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, int(trials) - completed)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=True)
    if not study.best_trial:
        raise RuntimeError(f"Optuna produced no valid trial for fold {rotation}")
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
            for sample in [*sm_samples, *grid_samples, *background_samples]
        ],
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
) -> str:
    payload = {
        "method_version": METHOD_VERSION,
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


def _write_map(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    output: Path,
    *,
    title: str,
    logarithmic: bool = False,
    contour_level: float | None = None,
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
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def _write_standard_maps(rows: Sequence[Mapping[str, Any]], output_dir: Path, prefix: str) -> None:
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
        )
    _write_map(
        rows,
        "cut_exclusion_ratio",
        output_dir / f"{prefix}_cut_exclusion_contour.png",
        title="Production cross section / exact cut limit",
        contour_level=1.0,
    )
    _write_map(
        rows,
        "shape_exclusion_ratio",
        output_dir / f"{prefix}_shape_exclusion_contour.png",
        title="Production cross section / pyhf shape limit",
        contour_level=1.0,
    )


def _compact_validation(validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in validation.items()
        if key not in {"signal_rows", "background_rows", "background_rows_by_point"}
    }


def _flatten_fold_points(
    rotations: Sequence[Mapping[str, Any]],
    container: str,
) -> list[dict[str, Any]]:
    rows = []
    for rotation in rotations:
        payload = rotation[container]
        for point in payload["points"].values():
            rows.append(
                {
                    key: value
                    for key, value in point.items()
                    if not isinstance(value, (dict, list, tuple, np.ndarray))
                }
            )
    return rows


def _validation_fold_arrays(
    record: Mapping[str, Any],
    sample: EventSample,
) -> dict[str, np.ndarray]:
    validation = record["validation"]
    signal = validation["signal_rows"][sample.sample_id]
    if validation.get("parameterized"):
        cache = record.setdefault("_validation_parameter_cache", {})
        if sample.point_id not in cache:
            rows = _score_partition(
                record["_model_object"],
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


def _test_fold_arrays(record: Mapping[str, Any], sample: EventSample) -> dict[str, np.ndarray]:
    test = record["test"]
    signal = test["signal_rows"][sample.sample_id]
    if test.get("parameterized"):
        cache = record.setdefault("_test_parameter_cache", {})
        if sample.point_id not in cache:
            cache[sample.point_id] = _score_partition(
                record["_model_object"],
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
        "signal_scores": np.asarray(signal["scores"], dtype=float),
        "signal_weights": np.asarray(signal["unit_xsec_weights"], dtype=float),
        "background_scores": _concatenate_partition(background_rows, "scores"),
        "background_weights": _concatenate_partition(
            background_rows, "physical_weights"
        ),
    }


def _candidate_maps_for_validation(
    records: Sequence[Mapping[str, Any]],
    sample: EventSample,
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
    return (0.0, max(100.0, 100.0 * estimate))


def _select_shape_candidate(
    records: Sequence[Mapping[str, Any]],
    sample: EventSample,
    candidate_maps: Sequence[Mapping[tuple[int, ...], Mapping[str, Any]]],
    common_candidates: Sequence[tuple[int, ...]],
) -> dict[str, Any]:
    evaluated = []
    for key in common_candidates:
        if len(key) - 1 < 2:
            continue
        channels = []
        fold_edges = []
        valid = True
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
                valid = False
                break
            channels.append(channel)
            fold_edges.append(list(map(float, edges)))
        fit = (
            pyhf_combined_limit(
                channels,
                include_staterror=True,
                poi_bounds=_poi_bounds_for_channels(channels),
            )
            if valid
            else {"status": "invalid_background", "expected_median": None}
        )
        evaluated.append(
            {
                "base_edge_indices": list(key),
                "n_bins": len(key) - 1,
                "fold_edges": fold_edges,
                "valid": bool(valid and fit.get("status") == "ok"),
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


def _shape_results(
    grid_samples: Sequence[EventSample],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    parameterized = any(record["validation"].get("parameterized") for record in records)
    shared_candidates = (
        None
        if parameterized
        else _candidate_maps_for_validation(records, grid_samples[0])
    )
    rows = []

    def clear_parameter_caches(point_id: str | None = None) -> None:
        if not parameterized:
            return
        for record in records:
            for cache_name in ("_validation_parameter_cache", "_test_parameter_cache"):
                cache = record.setdefault(cache_name, {})
                if point_id is None:
                    cache.clear()
                else:
                    cache.pop(point_id, None)

    for sample in grid_samples:
        clear_parameter_caches()
        candidate_maps, common_candidates = (
            _candidate_maps_for_validation(records, sample)
            if parameterized
            else shared_candidates
        )
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
        one_bin = pyhf_one_bin_limit(
            one_bin_signal,
            one_bin_background,
            include_staterror=False,
            poi_bounds=(0.0, max(100.0, 100.0 * one_bin_estimate)),
        )
        selection = _select_shape_candidate(
            records,
            sample,
            candidate_maps,
            common_candidates,
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
        if selection["status"] != "ok":
            rows.append(base)
            clear_parameter_caches(sample.point_id)
            continue

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
            rows.append(
                {
                    **base,
                    "status": "failed_nonpositive_test_bin",
                    "test_binning_attempts": attempts,
                }
            )
            clear_parameter_caches(sample.point_id)
            continue

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
        rows.append(
            {
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
        )
        clear_parameter_caches(sample.point_id)
    rows.sort(key=lambda row: (float(row["c3"]), float(row["d4"])))
    return rows


def run_c3d4_study(
    *,
    sm_signal_specs: Sequence[Mapping[str, Any]],
    grid_signal_specs: Sequence[Mapping[str, Any]],
    background_specs: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    observable_set: str = EXTENDED_SCHEMA_ID,
    feature_profile: str | None = None,
    training_strategy: str = "pooled-crossfit-v2",
    cv_folds: int = 5,
    optuna_trials: int = 40,
    luminosity: float = 3000.0,
    seed: int = BASE_SEED,
    max_events: int | None = None,
    legacy_scan_csv: str | Path | None = None,
    repo_dir: str | Path | None = None,
    run_shape: bool = True,
    hash_inputs: bool = True,
) -> dict[str, Any]:
    """Run the complete versioned study and return its machine-readable summary."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parents[1]
    source_commit = _source_commit(repo_dir)
    cv_folds = int(cv_folds)
    if cv_folds != 5:
        raise ValueError("resolved-8b v2 uses exactly five rotating folds")
    if int(optuna_trials) < 0:
        raise ValueError("optuna_trials must be non-negative")
    if training_strategy not in {
        "sm-crossfit-v2",
        "pooled-crossfit-v2",
        "parameterized-crossfit-v1",
    }:
        raise ValueError(f"Unknown training strategy {training_strategy!r}")
    if observable_set == LEGACY_SCHEMA_ID and feature_profile not in (None, "corrected28"):
        raise ValueError(
            f"{observable_set} supports only corrected28, not {feature_profile}"
        )

    common = {
        "observable_set": observable_set,
        "luminosity": float(luminosity),
        "n_folds": cv_folds,
        "seed": int(seed),
        "max_events": max_events,
    }
    sm_samples = _load_samples(sm_signal_specs, kind="sm_signal", **common)
    grid_samples = _load_samples(grid_signal_specs, kind="grid_signal", **common)
    background_samples = _load_samples(background_specs, kind="background", **common)
    if not sm_samples:
        raise ValueError("The SM cross-fit baseline requires a dedicated SM signal sample")
    if len(grid_samples) != 57:
        raise ValueError(f"The pooled study requires exactly 57 c3/d4 samples, found {len(grid_samples)}")
    point_ids = [sample.point_id for sample in grid_samples]
    if len(set(point_ids)) != 57:
        raise ValueError("The c3/d4 inputs do not contain 57 unique points")
    if not background_samples:
        raise ValueError("The study requires at least one background source")
    normalization_inputs = _normalization_metadata(
        luminosity,
        sm_samples,
        grid_samples,
        background_samples,
    )
    fold_digest = _fold_assignment_digest(
        [*sm_samples, *grid_samples, *background_samples]
    )
    input_hashes = {
        str(sample.path): (
            _sha256(sample.path)
            if hash_inputs
            else f"not-computed:size={sample.path.stat().st_size}:mtime_ns={sample.path.stat().st_mtime_ns}"
        )
        for sample in [*sm_samples, *grid_samples, *background_samples]
    }

    if observable_set == LEGACY_SCHEMA_ID:
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
    legacy = _load_legacy_baseline(None if legacy_scan_csv is None else Path(legacy_scan_csv))
    runtime_versions = _package_versions()
    manifest = {
        "method_version": METHOD_VERSION,
        "status": "running",
        "source_commit": source_commit,
        "observable_set": observable_set,
        "requested_feature_profile": feature_profile,
        "requested_training_strategy": training_strategy,
        "cv_folds": cv_folds,
        "fold_rule": "test=f, validation=(f+1)%5, train=remaining three",
        "seed": int(seed),
        "luminosity_fb_inverse": float(luminosity),
        "optuna_trials_per_fold": int(optuna_trials),
        "score_shape_enabled": bool(run_shape),
        "fixed_xgboost_parameters": FIXED_XGBOOST_PARAMS,
        "package_versions": runtime_versions,
        "outputs": {},
    }
    _write_json(output_dir / "method_manifest.json", manifest)

    profile_results: dict[str, Any] = {}
    print(f"Comparing feature profiles: {', '.join(profiles)}")
    for profile in profiles:
        profile_dir = output_dir / "feature_profile_ablation" / profile
        indices = _profile_indices(observable_set, profile)
        records = []
        validation_sigmas = []
        for rotation in range(cv_folds):
            print(f"  fixed profile {profile}, fold {rotation + 1}/{cv_folds}")
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
        _write_standard_maps(aggregate, profile_dir / "maps", profile)

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
        "selected_profile": selected_profile,
        "within_one_percent_candidates": near_best,
        "validation_objectives": {
            profile: result["objective"] for profile, result in profile_results.items()
        },
    }
    _write_json(output_dir / "feature_profile_selection.json", selection)
    print("Selected global profile:", selected_profile)

    strategies = (
        ["sm-crossfit-v2"]
        if training_strategy == "sm-crossfit-v2"
        else ["sm-crossfit-v2", "pooled-crossfit-v2"]
    )
    strategy_results: dict[str, Any] = {}
    indices = _profile_indices(observable_set, selected_profile)
    for strategy in strategies:
        strategy_dir = output_dir / strategy
        records = []
        print(f"Tuning {strategy} with profile {selected_profile}")
        for rotation in range(cv_folds):
            fold_seed = seed + rotation
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
                    ),
                )
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
            record = {
                "rotation": rotation,
                "model": str(model_path),
                "model_metadata": metadata,
                "parameters": fitted_params,
                "tuning": tuning,
                "validation": validation,
                "test": test,
            }
            records.append(record)
            _write_json(strategy_dir / "optuna" / f"fold_{rotation}_history.json", tuning)
            _write_json(
                strategy_dir / "validation" / f"fold_{rotation}.json",
                _compact_validation(validation),
            )
            _write_json(
                strategy_dir / "test" / f"fold_{rotation}.json",
                {
                    "rotation": test["rotation"],
                    "points": test["points"],
                },
            )
        aggregate = _aggregate_cut_results(grid_samples, [record["test"] for record in records])
        validation_aggregate, rotation_validation_limits = _aggregate_validation_crossfit(
            grid_samples, records
        )
        if run_shape:
            print(f"Selecting pyhf score shapes for {strategy}")
            shape = _shape_results(grid_samples, records)
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
        strategy_results[strategy] = {
            "records": records,
            "aggregate": aggregate,
            "validation_aggregate": validation_aggregate,
            "rotation_validation_limits": rotation_validation_limits,
            "shape": shape,
        }
        _write_rows(strategy_dir / "per_fold_validation.csv", _flatten_fold_points(records, "validation"))
        _write_rows(strategy_dir / "per_fold_test.csv", _flatten_fold_points(records, "test"))
        _write_rows(strategy_dir / "cut_results.csv", aggregate)
        _write_json(strategy_dir / "cut_results.json", aggregate)
        _write_rows(strategy_dir / "shape_results.csv", shape)
        _write_json(strategy_dir / "shape_results.json", shape)

    sm_reference = {
        (float(row["c3"]), float(row["d4"])): float(row["cut_sigma95_fb"])
        for row in strategy_results["sm-crossfit-v2"]["aggregate"]
    }
    for strategy, result in strategy_results.items():
        _add_baseline_ratios(
            result["aggregate"],
            legacy,
            None if strategy == "sm-crossfit-v2" else sm_reference,
        )
        strategy_dir = output_dir / strategy
        _write_rows(strategy_dir / "cut_results.csv", result["aggregate"])
        _write_json(strategy_dir / "cut_results.json", result["aggregate"])
        _write_standard_maps(result["aggregate"], strategy_dir / "maps", strategy)

    gate = None
    if "pooled-crossfit-v2" in strategy_results:
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

    run_parameterized = bool(
        gate
        and gate.get("passed")
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
        for rotation in range(cv_folds):
            fold_seed = seed + rotation
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
                ),
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
                _compact_validation(validation),
            )
            _write_json(
                strategy_dir / "test" / f"fold_{rotation}.json",
                {"rotation": test["rotation"], "points": test["points"]},
            )

        aggregate = _aggregate_cut_results(
            grid_samples, [record["test"] for record in records]
        )
        validation_aggregate, rotation_validation_limits = _aggregate_validation_crossfit(
            grid_samples, records
        )
        print(f"Selecting pyhf score shapes for {strategy}")
        shape = _shape_results(grid_samples, records) if run_shape else [
            {
                "point_id": sample.point_id,
                "c3": sample.c3,
                "d4": sample.d4,
                "status": "disabled",
            }
            for sample in grid_samples
        ]
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
        _write_rows(strategy_dir / "per_fold_validation.csv", _flatten_fold_points(records, "validation"))
        _write_rows(strategy_dir / "per_fold_test.csv", _flatten_fold_points(records, "test"))
        _write_rows(strategy_dir / "cut_results.csv", aggregate)
        _write_json(strategy_dir / "cut_results.json", aggregate)
        _write_rows(strategy_dir / "shape_results.csv", shape)
        _write_json(strategy_dir / "shape_results.json", shape)
        _write_standard_maps(aggregate, strategy_dir / "maps", strategy)
        manifest["parameterized_classifier"] = {
            "status": "complete",
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

    sample_manifests = []
    for sample in [*sm_samples, *grid_samples, *background_samples]:
        item = _sample_manifest(sample, input_hash=input_hashes[str(sample.path)])
        sample_manifests.append(item)
    fold_assignment_file = output_dir / "fold_assignments.csv"
    _write_fold_assignments(
        fold_assignment_file,
        [*sm_samples, *grid_samples, *background_samples],
    )
    output_manifest = {
        "feature_profile_selection": str(output_dir / "feature_profile_selection.json"),
        "fold_assignments": str(fold_assignment_file),
    }
    if gate is not None:
        output_manifest["parameterized_gate"] = str(
            output_dir / "parameterized_classifier_gate.json"
        )
    manifest.update(
        {
            "status": "complete",
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
    _write_json(output_dir / "method_manifest.json", manifest)
    summary = {
        "manifest": manifest,
        "selected_feature_profile": selected_profile,
        "strategy_results": {
            strategy: {
                "cut_results": result["aggregate"],
                "validation_results": result["validation_aggregate"],
            }
            for strategy, result in strategy_results.items()
        },
        "parameterized_gate": gate,
    }
    _write_json(output_dir / "study_summary.json", summary)
    return summary


__all__ = ["METHOD_VERSION", "EventSample", "run_c3d4_study"]
