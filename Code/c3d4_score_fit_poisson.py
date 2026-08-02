#!/usr/bin/env python3
"""SM-trained XGBoost score-fit limits in the (c3, d4) plane.

This is an intentionally small alternative to the historical threshold and
pyhf paths.  It trains only on the dedicated SM hhhh sample, freezes the five
cross-fit classifiers, and then builds physical score templates for hhhh,
hhhbb, hh+4b, and the registered backgrounds.  The inference model is the
exact binned Poisson likelihood ratio to an SM signal-plus-background Asimov
data set.  There is no common signal-strength parameter and no interpolation
of signal yields before the likelihood is evaluated.

The only optional nuisance is one fully correlated background normalization.
It is disabled by default.  B/4 and 4B are always retained as explicitly
labelled stress tests rather than interpreted as an uncertainty interval.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from c3d4_plot_style import (
    ATL_PHYS_PUB_2025_003_FIGURE,
    ATL_PHYS_PUB_2025_003_LABEL,
    ATL_PHYS_PUB_2025_003_SOURCE_URL,
    DEFAULT_HHHH_PERTURBATIVITY_LEVEL,
    _hhhh_perturbativity_grid,
    _plot_atlas_phys_pub_2025_003_curve,
    _plot_sm_marker,
)
from c3d4_profile_from_study import (
    HistogramSummary,
    ProfileBuildError,
    _histogram_partition,
    _histogram_sample,
    _point_map,
    _read_json,
    _sha256,
    _specs_from_manifest,
    _validate_dedicated_sm_signal,
    _weighted_quantiles,
)
from c3d4_xgboost_runner import (
    FIXED_XGBOOST_PARAMS,
    _load_samples,
    _profile_indices,
    _score_partition,
    _train_model,
    _training_arrays,
)
from hh4b_c3_xsec import evaluate_hh4b_c3_fit
from observable_schemas import validate_model_contract
from sample_report import _terminal_table, terminal_label, terminal_number


SCHEMA_VERSION = "c3d4-score-fit-poisson-v1"
BUILDER_VERSION = "c3d4-score-fit-poisson-v1.3"
N_FOLDS = 5
SM_POINT = (0.0, 0.0)
SIMULTANEOUS_LEVELS = {"68": 2.30, "95": 5.991}
FIXED_COUPLING_95_LEVEL = 3.841458820694124
FEATURE_PROFILES = {"core52": 52, "full91": 91}
BACKGROUND_STRESS_FACTORS = (0.25, 1.0, 4.0)
BINNING_SCHEMES = {
    "background_quantile_4bin": (0.0, 0.50, 0.80, 0.95, 1.0),
    "background_quantile_5bin": (0.0, 0.50, 0.75, 0.90, 0.97, 1.0),
}
DEFAULT_MIN_BACKGROUND_SOURCE_EVENTS = 25
DEFAULT_MIN_BACKGROUND_NEFF = 10.0


class ScoreFitError(RuntimeError):
    """The score-fit analysis contract could not be satisfied."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _elapsed(start: float) -> str:
    seconds = max(0, int(time.monotonic() - start))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _progress(start: float, stage: str, message: str) -> None:
    print(f"[score-fit {stage}] {message}; elapsed {_elapsed(start)}", flush=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                _jsonable(payload),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ScoreFitError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            fieldnames: list[str] = []
            for row in rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(str(key))
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_array(value: Any, label: str, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{label} must be {ndim}-dimensional")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{label} contains a non-finite value")
    return array


def poisson_asimov_deviance(
    observed: Sequence[float] | np.ndarray,
    expected: Sequence[float] | np.ndarray,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> float | np.ndarray:
    """Return twice the exact Poisson log-likelihood ratio to the observation."""

    observation, expectation = np.broadcast_arrays(
        _finite_array(observed, "observed counts"),
        _finite_array(expected, "expected counts"),
    )
    if np.any(observation < 0.0) or np.any(expectation < 0.0):
        raise ValueError("Poisson counts and means must be non-negative")
    impossible = (observation > 0.0) & (expectation <= 0.0)
    if np.any(impossible):
        if axis is None:
            return math.inf
        terms = np.full_like(observation, np.inf, dtype=float)
        safe = ~impossible
        terms[safe] = expectation[safe] - observation[safe]
        positive = safe & (observation > 0.0)
        terms[positive] += observation[positive] * np.log(
            observation[positive] / expectation[positive]
        )
        return 2.0 * np.sum(terms, axis=axis)
    terms = expectation - observation
    positive = observation > 0.0
    terms = np.asarray(terms, dtype=float)
    terms[positive] += observation[positive] * np.log(
        observation[positive] / expectation[positive]
    )
    result = np.maximum(2.0 * np.sum(terms, axis=axis), 0.0)
    return float(result) if np.ndim(result) == 0 else result


def asimov_discovery_q(signal: Sequence[float], background: Sequence[float]) -> float:
    """Expected binned discovery q for an SM signal over background."""

    signal_array = _finite_array(signal, "signal", ndim=1)
    background_array = _finite_array(background, "background", ndim=1)
    if signal_array.shape != background_array.shape:
        raise ValueError("Signal and background bins do not align")
    if np.any(signal_array < 0.0) or np.any(background_array <= 0.0):
        raise ValueError("Discovery templates require non-negative signal and positive background")
    with np.errstate(divide="raise", invalid="raise"):
        terms = (signal_array + background_array) * np.log1p(
            signal_array / background_array
        ) - signal_array
    return float(2.0 * np.sum(terms))


def _golden_minimum(
    function: Any,
    lower: float = -8.0,
    upper: float = 8.0,
    *,
    iterations: int = 128,
) -> tuple[float, float]:
    """Small deterministic bounded scalar minimizer for the one nuisance."""

    ratio = 0.5 * (math.sqrt(5.0) - 1.0)
    a = float(lower)
    b = float(upper)
    c = b - ratio * (b - a)
    d = a + ratio * (b - a)
    fc = float(function(c))
    fd = float(function(d))
    for _ in range(int(iterations)):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = float(function(c))
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = float(function(d))
    candidates = [(a, function(a)), (b, function(b)), (c, fc), (d, fd), (0.0, function(0.0))]
    eta, value = min(candidates, key=lambda item: float(item[1]))
    return float(eta), float(value)


def profiled_poisson_q(
    tested_signal: Sequence[float],
    sm_signal: Sequence[float],
    background: Sequence[float],
    *,
    background_norm_fraction: float = 0.0,
) -> tuple[float, float]:
    """Poisson q and fitted correlated background-normalization nuisance."""

    tested = _finite_array(tested_signal, "tested signal", ndim=1)
    sm = _finite_array(sm_signal, "SM signal", ndim=1)
    bkg = _finite_array(background, "background", ndim=1)
    if not (tested.shape == sm.shape == bkg.shape):
        raise ValueError("Likelihood templates do not share one binning")
    if np.any(tested < 0.0) or np.any(sm < 0.0) or np.any(bkg <= 0.0):
        raise ValueError("Likelihood templates require non-negative signal and positive background")
    fraction = float(background_norm_fraction)
    if not math.isfinite(fraction) or fraction < 0.0:
        raise ValueError("background_norm_fraction must be finite and non-negative")
    observation = sm + bkg
    if fraction == 0.0:
        return float(poisson_asimov_deviance(observation, tested + bkg)), 0.0
    log_width = math.log1p(fraction)

    def objective(eta: float) -> float:
        expected = tested + bkg * math.exp(log_width * float(eta))
        return float(poisson_asimov_deviance(observation, expected)) + float(eta) ** 2

    eta, value = _golden_minimum(objective)
    return max(0.0, float(value)), eta


def _combine_histograms(summaries: Sequence[HistogramSummary]) -> HistogramSummary:
    if not summaries:
        raise ValueError("At least one histogram summary is required")
    yields = np.sum([np.asarray(item.yields, dtype=float) for item in summaries], axis=0)
    covariance = np.sum(
        [np.asarray(item.covariance, dtype=float) for item in summaries], axis=0
    )
    sources = np.sum(
        [np.asarray(item.source_events, dtype=np.int64) for item in summaries], axis=0
    )
    variances = np.diag(covariance).copy()
    effective = np.divide(
        yields * yields,
        variances,
        out=np.zeros_like(yields),
        where=variances > 0.0,
    )
    return HistogramSummary(yields, variances, covariance, sources, effective)


def _partition_scores_and_weights(
    partition: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    scores = [np.asarray(item["scores"], dtype=float) for item in partition.values()]
    weights = [np.asarray(item["physical_weights"], dtype=float) for item in partition.values()]
    if not scores:
        raise ScoreFitError("A background validation partition is empty")
    score_array = np.concatenate(scores)
    # Signed event weights remain signed in every physical template.  A
    # monotonic quantile CDF, however, requires a non-negative measure, so the
    # same absolute physical weights used by the classifier define score-bin
    # boundaries.  This choice affects boundaries only, never event yields.
    weight_array = np.abs(np.concatenate(weights))
    if np.any(~np.isfinite(score_array)) or np.any(~np.isfinite(weight_array)):
        raise ScoreFitError("Validation scores or physical weights are non-finite")
    if np.any(weight_array < 0.0) or float(np.sum(weight_array)) <= 0.0:
        raise ScoreFitError("Background quantiles require positive physical weight")
    return score_array, weight_array


def _background_failures(
    summary: HistogramSummary,
    *,
    min_source_events: int,
    min_neff: float,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for index, (value, sources, neff) in enumerate(
        zip(summary.yields, summary.source_events, summary.effective_events)
    ):
        reason = None
        if float(value) <= 0.0:
            reason = "nonpositive_background"
        elif int(sources) < int(min_source_events):
            reason = "background_source_events"
        elif float(neff) < float(min_neff):
            reason = "background_neff"
        if reason is not None:
            failures.append(
                {
                    "bin": int(index),
                    "reason": reason,
                    "yield": float(value),
                    "source_events": int(sources),
                    "neff": float(neff),
                }
            )
    return failures


def merge_background_quantile_edges(
    background_partitions: Sequence[Mapping[str, Mapping[str, Any]]],
    initial_edges_by_fold: Sequence[Sequence[float]],
    *,
    min_source_events: int = DEFAULT_MIN_BACKGROUND_SOURCE_EVENTS,
    min_neff: float = DEFAULT_MIN_BACKGROUND_NEFF,
) -> tuple[list[np.ndarray], list[dict[str, Any]], HistogramSummary]:
    """Apply the same deterministic ordinal-bin merge to every fold."""

    edges_by_fold = [np.asarray(edges, dtype=float) for edges in initial_edges_by_fold]
    if len(background_partitions) != len(edges_by_fold) or not edges_by_fold:
        raise ValueError("Background partitions and fold edges do not align")
    bin_counts = {len(edges) - 1 for edges in edges_by_fold}
    if len(bin_counts) != 1:
        raise ValueError("Every fold must begin with the same number of ordinal bins")
    history: list[dict[str, Any]] = []
    while True:
        combined = _combine_histograms(
            [
                _histogram_partition(partition, edges)
                for partition, edges in zip(background_partitions, edges_by_fold)
            ]
        )
        failures = _background_failures(
            combined,
            min_source_events=min_source_events,
            min_neff=min_neff,
        )
        if not failures:
            return edges_by_fold, history, combined
        n_bins = len(edges_by_fold[0]) - 1
        if n_bins <= 1:
            raise ScoreFitError(
                "Background effective-statistics gate fails even after merging to one bin: "
                + json.dumps(failures, sort_keys=True)
            )
        # The highest-score failing tail is handled first.  It is merged into
        # the adjacent lower-score bin.  The first bin, if ever failing, is
        # merged upward because it has no lower neighbour.
        failure = max(failures, key=lambda item: int(item["bin"]))
        failed_bin = int(failure["bin"])
        remove_index = failed_bin if failed_bin > 0 else 1
        before = [edges.tolist() for edges in edges_by_fold]
        edges_by_fold = [np.delete(edges, remove_index) for edges in edges_by_fold]
        history.append(
            {
                **failure,
                "failed_bin": int(failed_bin),
                "removed_boundary_index": int(remove_index),
                "before": before,
                "after": [edges.tolist() for edges in edges_by_fold],
            }
        )


def _scan_configurations() -> list[dict[str, Any]]:
    baseline = dict(FIXED_XGBOOST_PARAMS)
    baseline.pop("n_jobs", None)
    variations = [
        ("baseline", {}),
        ("depth2", {"max_depth": 2}),
        ("depth4", {"max_depth": 4}),
        ("min_child_weight5", {"min_child_weight": 5.0}),
        ("reg_lambda5", {"reg_lambda": 5.0}),
        ("slow500", {"n_estimators": 500, "learning_rate": 0.03}),
    ]
    output = []
    for name, changes in variations:
        params = {**baseline, **changes}
        output.append({"name": name, "parameters": params})
    return output


def select_scan_result(
    scan_results: Sequence[Mapping[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    """Apply the declared 1% baseline and 2% four-bin preference rules."""

    if not scan_results:
        raise ValueError("No scan results were supplied")
    by_name = {str(row["name"]): row for row in scan_results}
    if "baseline" not in by_name:
        raise ValueError("The deterministic scan has no baseline")
    best = max(scan_results, key=lambda row: float(row["mean_binning_q"]))
    baseline = by_name["baseline"]
    best_q = float(best["mean_binning_q"])
    baseline_q = float(baseline["mean_binning_q"])
    selected = baseline if baseline_q >= 0.99 * best_q else best
    metrics = dict(selected["schemes"])
    q4 = float(metrics["background_quantile_4bin"]["validation_discovery_q"])
    q5 = float(metrics["background_quantile_5bin"]["validation_discovery_q"])
    scheme = "background_quantile_4bin" if q4 >= 0.98 * q5 else "background_quantile_5bin"
    audit = {
        "best_raw_configuration": str(best["name"]),
        "best_raw_mean_q": best_q,
        "baseline_mean_q": baseline_q,
        "baseline_retained_within_one_percent": str(selected["name"]) == "baseline",
        "selected_configuration": str(selected["name"]),
        "selected_scheme": scheme,
        "selected_scheme_q4": q4,
        "selected_scheme_q5": q5,
        "four_bin_preference_within_two_percent": scheme == "background_quantile_4bin",
    }
    return str(selected["name"]), scheme, audit


def _model_cache_paths(output_dir: Path, config_name: str, fold: int) -> tuple[Path, Path]:
    directory = output_dir / "scan" / "models" / config_name
    return directory / f"fold_{fold}.json", directory / f"fold_{fold}.cache.json"


def _load_cached_model(
    model_path: Path,
    cache_path: Path,
    *,
    fingerprint: str,
    observable_set: str,
    profile: str,
    threads: int,
) -> Any | None:
    if not model_path.is_file() or not cache_path.is_file():
        return None
    record = _read_json(cache_path)
    if not isinstance(record, Mapping) or record.get("fingerprint") != fingerprint:
        raise ScoreFitError(
            f"Cached model fingerprint differs at {model_path}; use a new output directory"
        )
    import xgboost as xgb

    model = xgb.XGBClassifier(n_jobs=int(threads))
    model.load_model(str(model_path))
    model.set_params(n_jobs=int(threads))
    model.get_booster().set_param({"nthread": int(threads)})
    validate_model_contract(model, observable_set, profile)
    return model


def _save_cached_model(
    model: Any,
    model_path: Path,
    cache_path: Path,
    record: Mapping[str, Any],
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{model_path.stem}.", suffix=".json", dir=str(model_path.parent)
    )
    os.close(descriptor)
    try:
        model.save_model(temporary_name)
        os.replace(temporary_name, model_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    _atomic_json(cache_path, record)


def _fit_and_validate_one(
    *,
    config: Mapping[str, Any],
    fold: int,
    training_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    sm_samples: Sequence[Any],
    background_samples: Sequence[Any],
    profile_indices: np.ndarray,
    observable_set: str,
    profile: str,
    source_commit: str,
    seed: int,
    xgboost_threads: int,
    output_dir: Path,
    run_fingerprint: str,
) -> dict[str, Any]:
    name = str(config["name"])
    parameters = dict(config["parameters"])
    model_path, cache_path = _model_cache_paths(output_dir, name, fold)
    fingerprint = _sha256_payload(
        {
            "run": run_fingerprint,
            "configuration": name,
            "parameters": parameters,
            "fold": int(fold),
            "seed": int(seed + fold),
        }
    )
    model = _load_cached_model(
        model_path,
        cache_path,
        fingerprint=fingerprint,
        observable_set=observable_set,
        profile=profile,
        threads=xgboost_threads,
    )
    cached = model is not None
    metadata: Mapping[str, Any] | None = None
    fitted_parameters = parameters
    if model is None:
        X, labels, weights = training_arrays
        model, metadata, fitted_parameters = _train_model(
            X,
            labels,
            weights,
            params=parameters,
            seed=seed + fold,
            observable_set=observable_set,
            profile=profile,
            strategy="sm-crossfit-v2",
            rotation=fold,
            source_commit=source_commit,
            xgboost_jobs=xgboost_threads,
        )
        _save_cached_model(
            model,
            model_path,
            cache_path,
            {
                "schema": SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "configuration": name,
                "fold": int(fold),
                "parameters": fitted_parameters,
                "model_metadata": metadata,
                "created_utc": _utc_now(),
            },
        )
    sm_validation = _score_partition(
        model,
        sm_samples,
        rotation=fold,
        split="validation",
        n_folds=N_FOLDS,
        profile_indices=profile_indices,
        scale_validation_to_full=False,
        parameterized=False,
    )
    background_validation = _score_partition(
        model,
        background_samples,
        rotation=fold,
        split="validation",
        n_folds=N_FOLDS,
        profile_indices=profile_indices,
        scale_validation_to_full=False,
        parameterized=False,
    )
    return {
        "configuration": name,
        "fold": int(fold),
        "model": model,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "cached": cached,
        "parameters": fitted_parameters,
        "sm_validation": sm_validation,
        "background_validation": background_validation,
    }


def _evaluate_configuration(
    config: Mapping[str, Any],
    fold_records: Sequence[Mapping[str, Any]],
    *,
    min_source_events: int,
    min_neff: float,
) -> dict[str, Any]:
    ordered = sorted(fold_records, key=lambda row: int(row["fold"]))
    if [int(row["fold"]) for row in ordered] != list(range(N_FOLDS)):
        raise ScoreFitError(f"{config['name']}: validation folds are incomplete")
    background_partitions = [row["background_validation"] for row in ordered]
    signal_partitions = [row["sm_validation"] for row in ordered]
    scheme_results: dict[str, Any] = {}
    for scheme_name, quantiles in BINNING_SCHEMES.items():
        initial_edges = []
        for partition in background_partitions:
            scores, weights = _partition_scores_and_weights(partition)
            edges = _weighted_quantiles(scores, weights, quantiles)
            if len(edges) != len(quantiles):
                raise ScoreFitError(
                    f"{config['name']} {scheme_name}: repeated XGBoost scores collapsed bins"
                )
            initial_edges.append(edges)
        final_edges, merges, background = merge_background_quantile_edges(
            background_partitions,
            initial_edges,
            min_source_events=min_source_events,
            min_neff=min_neff,
        )
        signal = _combine_histograms(
            [
                _histogram_partition(partition, edges)
                for partition, edges in zip(signal_partitions, final_edges)
            ]
        )
        q_value = asimov_discovery_q(signal.yields, background.yields)
        scheme_results[scheme_name] = {
            "requested_quantiles": list(quantiles),
            "initial_edges_by_fold": [edges.tolist() for edges in initial_edges],
            "final_edges_by_fold": [edges.tolist() for edges in final_edges],
            "merge_history": merges,
            "final_bin_count": int(len(final_edges[0]) - 1),
            "validation_signal_yields": signal.yields.tolist(),
            "validation_background_yields": background.yields.tolist(),
            "validation_background_source_events": background.source_events.tolist(),
            "validation_background_neff": background.effective_events.tolist(),
            "validation_discovery_q": q_value,
            "validation_discovery_z": math.sqrt(max(0.0, q_value)),
        }
    mean_q = float(
        np.mean(
            [float(item["validation_discovery_q"]) for item in scheme_results.values()]
        )
    )
    return {
        "name": str(config["name"]),
        "parameters": dict(config["parameters"]),
        "mean_binning_q": mean_q,
        "schemes": scheme_results,
        "fold_models": [
            {
                "fold": int(row["fold"]),
                "path": str(row["model_path"]),
                "sha256": str(row["model_sha256"]),
                "cache_hit": bool(row["cached"]),
            }
            for row in ordered
        ],
    }


def _score_test_fold(
    *,
    fold: int,
    model: Any,
    grid_samples: Sequence[Any],
    hhhbb_samples: Sequence[Any],
    hh4b_samples: Sequence[Any],
    background_samples: Sequence[Any],
    profile_indices: np.ndarray,
    prediction_threads: int,
) -> dict[str, Any]:
    model.set_params(n_jobs=int(prediction_threads))
    model.get_booster().set_param({"nthread": int(prediction_threads)})
    kwargs = {
        "rotation": fold,
        "split": "test",
        "n_folds": N_FOLDS,
        "profile_indices": profile_indices,
        "scale_validation_to_full": False,
        "parameterized": False,
    }
    return {
        "fold": int(fold),
        "hhhh": _score_partition(model, grid_samples, **kwargs),
        "hhhbb": _score_partition(model, hhhbb_samples, **kwargs),
        "hh4b": _score_partition(model, hh4b_samples, **kwargs),
        "background": _score_partition(model, background_samples, **kwargs),
    }


def _template_for_sample(
    fold_partitions: Sequence[Mapping[str, Mapping[str, Any]]],
    sample: Any,
    edges_by_fold: Sequence[Sequence[float]],
) -> HistogramSummary:
    return _combine_histograms(
        [
            _histogram_sample(partition[sample.sample_id], edges)
            for partition, edges in zip(fold_partitions, edges_by_fold)
        ]
    )


def _assert_yield_closure(label: str, expected: float, observed: float) -> None:
    tolerance = max(1.0e-12, 2.0e-10 * max(abs(expected), abs(observed), 1.0))
    if not math.isclose(expected, observed, rel_tol=2.0e-10, abs_tol=tolerance):
        raise ScoreFitError(
            f"{label}: binned yield {observed:.16g} does not close to {expected:.16g}"
        )


def _sample_table_label(sample: Any) -> str:
    metadata = dict(getattr(sample, "metadata", None) or {})
    for key in ("description", "label", "process_id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return str(sample.sample_id)


def selected_sm_template_table_rows(
    *,
    templates: Mapping[str, Any],
    grid_samples: Sequence[Any],
    hhhbb_samples: Sequence[Any],
    hh4b_sample: Any,
    background_samples: Sequence[Any],
    luminosity: float,
) -> list[dict[str, Any]]:
    """Build the terminal/CSV rate table for the selected score binning.

    The score fit has no selected-event threshold.  Its analogue of the old
    ``N_XGB`` column is therefore the sum of all fitted score bins, which must
    close to the feature-tree input yield for every physical sample.
    """

    luminosity = float(luminosity)
    if not math.isfinite(luminosity) or luminosity <= 0.0:
        raise ValueError("luminosity must be finite and positive")
    points = _finite_array(templates["points"], "template-table points", ndim=2)
    hhhh = _finite_array(templates["hhhh"], "template-table hhhh", ndim=2)
    hhhbb = _finite_array(templates["hhhbb"], "template-table hhhbb", ndim=2)
    hh4b = _finite_array(templates["hh4b"], "template-table hh4b", ndim=1)
    background = _finite_array(
        templates["background"], "template-table background", ndim=1
    )
    if hhhh.shape != hhhbb.shape or hhhh.shape[0] != len(points):
        raise ScoreFitError("Template-table coupling arrays have inconsistent shapes")
    if hhhh.shape[1] != len(hh4b) or len(hh4b) != len(background):
        raise ScoreFitError("Template-table processes do not share one score binning")
    sm_indices = np.flatnonzero(
        np.isclose(points[:, 0], SM_POINT[0], atol=1.0e-12)
        & np.isclose(points[:, 1], SM_POINT[1], atol=1.0e-12)
    )
    if len(sm_indices) != 1:
        raise ScoreFitError(
            f"Template table requires one physical SM point, found {len(sm_indices)}"
        )
    sm_index = int(sm_indices[0])
    hhhh_by_point = _point_map(grid_samples, "template-table hhhh")
    hhhbb_by_point = _point_map(hhhbb_samples, "template-table hhhbb")
    if SM_POINT not in hhhh_by_point or SM_POINT not in hhhbb_by_point:
        raise ScoreFitError("Template table cannot find both physical SM signal samples")
    n_bins = len(background)

    def sample_row(
        *,
        role: str,
        label: str,
        sample: Any,
        yields: Sequence[float],
        c3: float | None,
        d4: float | None,
    ) -> dict[str, Any]:
        bin_yields = _finite_array(yields, f"template-table {label}", ndim=1)
        if len(bin_yields) != n_bins:
            raise ScoreFitError(f"Template-table row {label!r} has the wrong bin count")
        input_events = float(np.sum(np.asarray(sample.physical_weights, dtype=float)))
        binned_events = float(np.sum(bin_yields))
        _assert_yield_closure(f"template table {label}", input_events, binned_events)
        row: dict[str, Any] = {
            "role": role,
            "sample": label,
            "sample_id": str(sample.sample_id),
            "c3": c3,
            "d4": d4,
            "production_xsec_fb": float(sample.xsec_fb),
            "input_xsec_fb": input_events / luminosity,
            "input_events": input_events,
            "binned_events": binned_events,
            "mc_rows": int(sample.entries),
        }
        row.update(
            {
                f"score_bin_{index + 1}_events": float(value)
                for index, value in enumerate(bin_yields)
            }
        )
        return row

    sm_hhhh_sample = hhhh_by_point[SM_POINT]
    sm_hhhbb_sample = hhhbb_by_point[SM_POINT]
    rows = [
        sample_row(
            role="signal",
            label="SM hhhh",
            sample=sm_hhhh_sample,
            yields=hhhh[sm_index],
            c3=0.0,
            d4=0.0,
        ),
        sample_row(
            role="signal",
            label="SM hhh+bb",
            sample=sm_hhhbb_sample,
            yields=hhhbb[sm_index],
            c3=0.0,
            d4=0.0,
        ),
        sample_row(
            role="signal",
            label="SM hh+4b",
            sample=hh4b_sample,
            yields=hh4b,
            c3=0.0,
            d4=0.0,
        ),
    ]
    process_templates = templates["background_processes"]
    background_rows = []
    for sample in background_samples:
        if sample.sample_id not in process_templates:
            raise ScoreFitError(
                f"Template table is missing background process {sample.sample_id!r}"
            )
        background_rows.append(
            sample_row(
                role="background",
                label=_sample_table_label(sample),
                sample=sample,
                yields=process_templates[sample.sample_id]["yields"],
                c3=None,
                d4=None,
            )
        )
    background_rows.sort(key=lambda row: -abs(float(row["input_events"])))
    rows.extend(background_rows)

    signal_bins = hhhh[sm_index] + hhhbb[sm_index] + hh4b
    total_specs = (
        ("total", "Total background", background, None, None),
        ("total", "Total SM signal", signal_bins, 0.0, 0.0),
        ("Asimov", "SM signal + background", signal_bins + background, 0.0, 0.0),
    )
    for role, label, bin_yields, c3, d4 in total_specs:
        total = float(np.sum(bin_yields))
        row = {
            "role": role,
            "sample": label,
            "sample_id": "--",
            "c3": c3,
            "d4": d4,
            "production_xsec_fb": None,
            "input_xsec_fb": total / luminosity,
            "input_events": total,
            "binned_events": total,
            "mc_rows": None,
        }
        row.update(
            {
                f"score_bin_{index + 1}_events": float(value)
                for index, value in enumerate(bin_yields)
            }
        )
        rows.append(row)
    return rows


def terminal_selected_sm_template_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    templates: Mapping[str, Any],
    luminosity: float,
    selected_configuration: str,
    selected_scheme: str,
) -> str:
    """Render the selected physical SM score templates like the legacy table."""

    if not rows:
        raise ValueError("selected SM template table has no rows")
    n_bins = len(np.asarray(templates["background"], dtype=float))

    def optional_number(value: Any) -> str:
        if value is None:
            return "--"
        return terminal_number(value)

    headers = [
        "Role",
        "Sample",
        "c3",
        "d4",
        "sigma_prod [fb]",
        "sigma_input [fb]",
        "N_input",
        "N_binned",
        "MC rows",
        "Score-bin yields (low -> high)",
    ]
    table_rows = []
    for row in rows:
        bins = [
            float(row[f"score_bin_{index + 1}_events"])
            for index in range(n_bins)
        ]
        table_rows.append(
            [
                str(row["role"]),
                terminal_label(str(row["sample"])),
                optional_number(row.get("c3")),
                optional_number(row.get("d4")),
                optional_number(row.get("production_xsec_fb")),
                terminal_number(row["input_xsec_fb"]),
                terminal_number(row["input_events"]),
                terminal_number(row["binned_events"]),
                "--" if row.get("mc_rows") is None else str(int(row["mc_rows"])),
                "[" + ", ".join(terminal_number(value) for value in bins) + "]",
            ]
        )
    source_events = np.asarray(templates["background_source_events"], dtype=int)
    effective_events = np.asarray(templates["background_neff"], dtype=float)
    lines = [
        (
            "SM XGBoost score-fit templates / rates "
            f"(L = {float(luminosity):g} fb^-1)"
        ),
        f"Classifier configuration: {selected_configuration}; score binning: {selected_scheme}.",
        "There is no XGBoost threshold: every held-out event enters exactly one fitted score bin.",
        "N_binned must equal N_input; score-bin yields run from low to high classifier score.",
        "Input and binned yields include cross sections, K/BR factors, and tag/mistag factors.",
        (
            "Background independent source events by bin: ["
            + ", ".join(str(int(value)) for value in source_events)
            + "]"
        ),
        (
            "Background N_eff by bin: ["
            + ", ".join(terminal_number(value) for value in effective_events)
            + "]"
        ),
        _terminal_table(headers, table_rows, right_aligned=set(range(2, 9))),
    ]
    return "\n".join(lines)


def build_physical_templates(
    *,
    scheme_name: str,
    edges_by_fold: Sequence[Sequence[float]],
    scored_folds: Sequence[Mapping[str, Any]],
    grid_samples: Sequence[Any],
    hhhbb_samples: Sequence[Any],
    hh4b_sample: Any,
    background_samples: Sequence[Any],
    min_source_events: int,
    min_neff: float,
) -> dict[str, Any]:
    ordered = sorted(scored_folds, key=lambda row: int(row["fold"]))
    hhhh_partitions = [row["hhhh"] for row in ordered]
    hhhbb_partitions = [row["hhhbb"] for row in ordered]
    hh4b_partitions = [row["hh4b"] for row in ordered]
    background_partitions = [row["background"] for row in ordered]
    hhhh_map = _point_map(grid_samples, "hhhh")
    hhhbb_map = _point_map(hhhbb_samples, "hhhbb")
    points = sorted(hhhh_map)
    if set(points) != set(hhhbb_map):
        missing_hhhbb = sorted(set(points) - set(hhhbb_map))
        missing_hhhh = sorted(set(hhhbb_map) - set(points))
        raise ScoreFitError(
            f"hhhh/hhhbb coupling grids differ: missing hhhbb={missing_hhhbb}, "
            f"missing hhhh={missing_hhhh}"
        )
    if len(points) != 153:
        raise ScoreFitError(f"Expected 153 physical coupling points, found {len(points)}")
    hhhh_rows = []
    hhhbb_rows = []
    hhhh_neff = []
    hhhbb_neff = []
    for point in points:
        hhhh_sample = hhhh_map[point]
        hhhbb_sample = hhhbb_map[point]
        hhhh = _template_for_sample(hhhh_partitions, hhhh_sample, edges_by_fold)
        hhhbb = _template_for_sample(hhhbb_partitions, hhhbb_sample, edges_by_fold)
        _assert_yield_closure(
            f"{scheme_name} hhhh {point}",
            float(np.sum(hhhh_sample.physical_weights)),
            float(np.sum(hhhh.yields)),
        )
        _assert_yield_closure(
            f"{scheme_name} hhhbb {point}",
            float(np.sum(hhhbb_sample.physical_weights)),
            float(np.sum(hhhbb.yields)),
        )
        hhhh_rows.append(hhhh.yields)
        hhhbb_rows.append(hhhbb.yields)
        hhhh_neff.append(hhhh.effective_events)
        hhhbb_neff.append(hhhbb.effective_events)
    hh4b = _template_for_sample(hh4b_partitions, hh4b_sample, edges_by_fold)
    _assert_yield_closure(
        f"{scheme_name} hh4b",
        float(np.sum(hh4b_sample.physical_weights)),
        float(np.sum(hh4b.yields)),
    )
    background_fold_summaries = [
        _histogram_partition(partition, edges)
        for partition, edges in zip(background_partitions, edges_by_fold)
    ]
    background = _combine_histograms(background_fold_summaries)
    expected_background = float(
        sum(float(np.sum(sample.physical_weights)) for sample in background_samples)
    )
    _assert_yield_closure(
        f"{scheme_name} background", expected_background, float(np.sum(background.yields))
    )
    failures = _background_failures(
        background,
        min_source_events=min_source_events,
        min_neff=min_neff,
    )
    process_templates: dict[str, Any] = {}
    for sample in background_samples:
        summary = _template_for_sample(background_partitions, sample, edges_by_fold)
        _assert_yield_closure(
            f"{scheme_name} background {sample.sample_id}",
            float(np.sum(sample.physical_weights)),
            float(np.sum(summary.yields)),
        )
        process_templates[sample.sample_id] = {
            "yields": summary.yields,
            "source_events": summary.source_events,
            "neff": summary.effective_events,
        }
    return {
        "scheme": scheme_name,
        "points": np.asarray(points, dtype=float),
        "edges_by_fold": [np.asarray(edges, dtype=float) for edges in edges_by_fold],
        "hhhh": np.asarray(hhhh_rows, dtype=float),
        "hhhbb": np.asarray(hhhbb_rows, dtype=float),
        "hh4b": np.asarray(hh4b.yields, dtype=float),
        "background": np.asarray(background.yields, dtype=float),
        "hhhh_neff": np.asarray(hhhh_neff, dtype=float),
        "hhhbb_neff": np.asarray(hhhbb_neff, dtype=float),
        "hh4b_neff": np.asarray(hh4b.effective_events, dtype=float),
        "background_source_events": np.asarray(background.source_events, dtype=np.int64),
        "background_neff": np.asarray(background.effective_events, dtype=float),
        "background_gate_failures": failures,
        "background_processes": process_templates,
        "yield_closure": {
            "background_expected": expected_background,
            "background_binned": float(np.sum(background.yields)),
            "hh4b_expected": float(np.sum(hh4b_sample.physical_weights)),
            "hh4b_binned": float(np.sum(hh4b.yields)),
        },
    }


def evaluate_likelihood_points(
    templates: Mapping[str, Any],
    hh4b_fit: Mapping[str, Any],
    *,
    background_norm_fraction: float,
) -> dict[str, Any]:
    points = _finite_array(templates["points"], "coupling points", ndim=2)
    hhhh = _finite_array(templates["hhhh"], "hhhh templates", ndim=2)
    hhhbb = _finite_array(templates["hhhbb"], "hhhbb templates", ndim=2)
    hh4b = _finite_array(templates["hh4b"], "hh4b template", ndim=1)
    background = _finite_array(templates["background"], "background template", ndim=1)
    if hhhh.shape != hhhbb.shape or hhhh.shape[0] != len(points):
        raise ScoreFitError("Physical coupling templates have inconsistent shapes")
    if hhhh.shape[1] != len(hh4b) or len(hh4b) != len(background):
        raise ScoreFitError("Physical process templates do not share one binning")
    sm_indices = np.flatnonzero(
        np.isclose(points[:, 0], 0.0, atol=1.0e-12)
        & np.isclose(points[:, 1], 0.0, atol=1.0e-12)
    )
    if len(sm_indices) != 1:
        raise ScoreFitError(f"Expected one physical SM grid point, found {len(sm_indices)}")
    sm_index = int(sm_indices[0])
    reference = evaluate_hh4b_c3_fit(hh4b_fit, 0.0)
    reference_xsec = float(reference["cross_section_fb"])
    valid_c3 = [float(value) for value in hh4b_fit.get("evaluation_range_c3", (-20.0, 20.0))]
    if len(valid_c3) != 2 or valid_c3[0] >= valid_c3[1]:
        raise ScoreFitError("The hh+4b fit has no valid c3 interval")
    rate_scales = np.asarray(
        [
            float(evaluate_hh4b_c3_fit(hh4b_fit, c3)["cross_section_fb"])
            / reference_xsec
            for c3 in points[:, 0]
        ],
        dtype=float,
    )
    combined = hhhh + hhhbb + rate_scales[:, np.newaxis] * hh4b[np.newaxis, :]
    sm_signal = np.asarray(combined[sm_index], dtype=float)
    q_values: dict[str, np.ndarray] = {}
    eta_values: dict[str, np.ndarray] = {}
    for factor in BACKGROUND_STRESS_FACTORS:
        key = f"background_x{factor:g}"
        scaled_background = background * float(factor)
        q = np.zeros(len(points), dtype=float)
        eta = np.zeros(len(points), dtype=float)
        for index, tested_signal in enumerate(combined):
            q[index], eta[index] = profiled_poisson_q(
                tested_signal,
                sm_signal,
                scaled_background,
                background_norm_fraction=background_norm_fraction,
            )
        q_values[key] = q
        eta_values[key] = eta
    if abs(float(q_values["background_x1"][sm_index])) > 1.0e-9:
        raise ScoreFitError(
            f"The SM Asimov likelihood does not close: q={q_values['background_x1'][sm_index]:g}"
        )
    valid_mask = (
        (points[:, 0] >= valid_c3[0] - 1.0e-12)
        & (points[:, 0] <= valid_c3[1] + 1.0e-12)
    )
    return {
        "points": points,
        "hhhh": hhhh,
        "hhhbb": hhhbb,
        "hh4b_reference": hh4b,
        "hh4b_rate_scales": rate_scales,
        "combined": combined,
        "sm_index": sm_index,
        "sm_signal": sm_signal,
        "background": background,
        "q": q_values,
        "eta_hat": eta_values,
        "valid_c3": valid_c3,
        "valid_mask": valid_mask,
        "valid_point_count": int(np.count_nonzero(valid_mask)),
        "diagnostic_extrapolated_point_count": int(np.count_nonzero(~valid_mask)),
        "background_norm_fraction": float(background_norm_fraction),
    }


def interpolate_q_surfaces(
    likelihood: Mapping[str, Any],
    *,
    c3_range: Sequence[float],
    d4_range: Sequence[float],
    grid_bins: int,
) -> dict[str, Any]:
    """Interpolate already evaluated pointwise q values, never physical yields."""

    from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator

    points = np.asarray(likelihood["points"], dtype=float)
    valid = np.asarray(likelihood["valid_mask"], dtype=bool)
    valid_points = points[valid]
    if len(valid_points) < 3:
        raise ScoreFitError("At least three valid coupling points are required")
    c3_axis = np.linspace(float(c3_range[0]), float(c3_range[1]), int(grid_bins))
    d4_axis = np.linspace(float(d4_range[0]), float(d4_range[1]), int(grid_bins))
    c3_grid, d4_grid = np.meshgrid(c3_axis, d4_axis)
    c3_scale = max(abs(float(c3_range[0])), abs(float(c3_range[1])), 1.0)
    d4_scale = max(abs(float(d4_range[0])), abs(float(d4_range[1])), 1.0)
    standardized = np.column_stack(
        [valid_points[:, 0] / c3_scale, valid_points[:, 1] / d4_scale]
    )
    evaluation = np.column_stack(
        [(c3_grid / c3_scale).ravel(), (d4_grid / d4_scale).ravel()]
    )
    output: dict[str, Any] = {
        "c3_axis": c3_axis,
        "d4_axis": d4_axis,
        "coordinate_scales": [c3_scale, d4_scale],
        "valid_points": valid_points,
        "fields": {},
        "audit": {},
    }
    for key, values in likelihood["q"].items():
        q_points = np.asarray(values, dtype=float)[valid]
        clough_raw = np.asarray(
            CloughTocher2DInterpolator(standardized, q_points, fill_value=np.nan)(evaluation),
            dtype=float,
        ).reshape(c3_grid.shape)
        linear_raw = np.asarray(
            LinearNDInterpolator(standardized, q_points, fill_value=np.nan)(evaluation),
            dtype=float,
        ).reshape(c3_grid.shape)
        finite_clough = np.isfinite(clough_raw)
        finite_linear = np.isfinite(linear_raw)
        clough_negative = int(np.count_nonzero(finite_clough & (clough_raw < 0.0)))
        linear_negative = int(np.count_nonzero(finite_linear & (linear_raw < 0.0)))
        clough = np.where(finite_clough, np.maximum(clough_raw, 0.0), np.nan)
        linear = np.where(finite_linear, np.maximum(linear_raw, 0.0), np.nan)
        output["fields"][key] = {"clough": clough, "linear": linear}
        output["audit"][key] = {
            "point_q_min": float(np.min(q_points)),
            "point_q_max": float(np.max(q_points)),
            "clough_negative_cells_clipped": clough_negative,
            "linear_negative_cells_clipped": linear_negative,
            "clough_finite_cells": int(np.count_nonzero(finite_clough)),
            "linear_finite_cells": int(np.count_nonzero(finite_linear)),
            "clough_raw_min": (
                None if not np.any(finite_clough) else float(np.min(clough_raw[finite_clough]))
            ),
        }
    return output


def _surface_mesh(surface: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(
        np.asarray(surface["c3_axis"], dtype=float),
        np.asarray(surface["d4_axis"], dtype=float),
    )


def _level_crossings_1d(
    axis: Sequence[float], values: Sequence[float], level: float
) -> list[float]:
    """Return finite level crossings using local linear interpolation."""

    coordinates = np.asarray(axis, dtype=float)
    statistic = np.asarray(values, dtype=float)
    if coordinates.ndim != 1 or statistic.ndim != 1 or coordinates.shape != statistic.shape:
        raise ValueError("axis and values must be one-dimensional arrays of equal length")
    if len(coordinates) < 2 or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("axis must contain at least two strictly increasing values")
    target = float(level)
    crossings: list[float] = []
    for left, right, q_left, q_right in zip(
        coordinates[:-1], coordinates[1:], statistic[:-1], statistic[1:]
    ):
        if not (np.isfinite(q_left) and np.isfinite(q_right)):
            continue
        delta_left = float(q_left) - target
        delta_right = float(q_right) - target
        if delta_left == 0.0:
            crossings.append(float(left))
        if delta_left * delta_right < 0.0:
            fraction = -delta_left / (delta_right - delta_left)
            crossings.append(float(left + fraction * (right - left)))
    if np.isfinite(statistic[-1]) and float(statistic[-1]) == target:
        crossings.append(float(coordinates[-1]))
    unique: list[float] = []
    for crossing in crossings:
        if not unique or abs(crossing - unique[-1]) > 1.0e-10:
            unique.append(crossing)
    return unique


def fixed_coupling_95_intervals(
    surface: Mapping[str, Any],
    *,
    factor_key: str = "background_x1",
    interpolation: str = "clough",
) -> dict[str, Any]:
    """Extract one-parameter 95% intervals with the other coupling fixed to zero."""

    c3_axis = np.asarray(surface["c3_axis"], dtype=float)
    d4_axis = np.asarray(surface["d4_axis"], dtype=float)
    values = np.asarray(surface["fields"][factor_key][interpolation], dtype=float)
    if values.shape != (len(d4_axis), len(c3_axis)):
        raise ValueError("surface shape does not match its c3 and d4 axes")
    c3_zero = int(np.argmin(np.abs(c3_axis)))
    d4_zero = int(np.argmin(np.abs(d4_axis)))
    if abs(float(c3_axis[c3_zero])) > 1.0e-10 or abs(float(d4_axis[d4_zero])) > 1.0e-10:
        raise ScoreFitError("fixed-coupling intervals require zero on both surface axes")
    d4_crossings = _level_crossings_1d(
        d4_axis, values[:, c3_zero], FIXED_COUPLING_95_LEVEL
    )
    c3_crossings = _level_crossings_1d(
        c3_axis, values[d4_zero, :], FIXED_COUPLING_95_LEVEL
    )
    if len(d4_crossings) != 2 or len(c3_crossings) != 2:
        raise ScoreFitError(
            "expected exactly two 95% crossings for each fixed-coupling scan; "
            f"found d4|c3=0: {d4_crossings}, c3|d4=0: {c3_crossings}"
        )
    return {
        "confidence_level": 0.95,
        "degrees_of_freedom": 1,
        "q_threshold": FIXED_COUPLING_95_LEVEL,
        "interpolation": interpolation,
        "d4_at_c3_0": d4_crossings,
        "c3_at_d4_0": c3_crossings,
    }


def _contour_segments(
    ax: Any,
    c3_grid: np.ndarray,
    d4_grid: np.ndarray,
    values: np.ndarray,
    level: float,
    **kwargs: Any,
) -> tuple[Any | None, list[np.ndarray]]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return None, []
    minimum = float(np.min(values[finite]))
    maximum = float(np.max(values[finite]))
    if not (minimum <= float(level) <= maximum):
        return None, []
    contour = ax.contour(
        c3_grid,
        d4_grid,
        np.ma.masked_invalid(values),
        levels=[float(level)],
        **kwargs,
    )
    segments = [
        np.asarray(segment, dtype=float)
        for segment in contour.allsegs[0]
        if len(segment) >= 2
    ]
    return contour, segments


def _segment_audit(segments: Sequence[np.ndarray], tolerance: float = 1.0e-6) -> dict[str, Any]:
    closed = []
    for segment in segments:
        closed.append(bool(np.linalg.norm(segment[0] - segment[-1]) <= tolerance))
    return {
        "component_count": int(len(segments)),
        "closed_component_count": int(sum(closed)),
        "open_component_count": int(len(closed) - sum(closed)),
        "closed": closed,
    }


def _configure_axis(ax: Any, *, c3_range: Sequence[float], d4_range: Sequence[float]) -> None:
    ax.set_xlim(float(c3_range[0]), float(c3_range[1]))
    ax.set_ylim(float(d4_range[0]), float(d4_range[1]))
    ax.set_xlabel(r"$c_3$", fontsize=20)
    ax.set_ylabel(r"$d_4$", fontsize=20)
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=15,
    )
    ax.minorticks_on()
    ax.grid(alpha=0.15, linewidth=0.6)


def _draw_physics_overlays(
    ax: Any,
    c3_grid: np.ndarray,
    d4_grid: np.ndarray,
    unitarity: np.ndarray,
    *,
    draw_atlas: bool = True,
    draw_sm: bool = True,
) -> None:
    ax.contour(
        c3_grid,
        d4_grid,
        unitarity,
        levels=[DEFAULT_HHHH_PERTURBATIVITY_LEVEL],
        colors=["black"],
        linestyles=["--"],
        linewidths=[1.25],
        zorder=3,
    )
    if draw_atlas:
        _plot_atlas_phys_pub_2025_003_curve(ax)
    if draw_sm:
        _plot_sm_marker(ax)


def _save_figure(fig: Any, base: Path, title: str) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Title": title, "Creator": BUILDER_VERSION},
    )
    fig.savefig(png, bbox_inches="tight", dpi=260)
    return [pdf, png]


def make_plots(
    *,
    output_dir: Path,
    scan_results: Sequence[Mapping[str, Any]],
    selected_configuration: str,
    selected_scheme: str,
    templates_by_scheme: Mapping[str, Mapping[str, Any]],
    likelihood_by_scheme: Mapping[str, Mapping[str, Any]],
    surfaces_by_scheme: Mapping[str, Mapping[str, Any]],
    c3_range: Sequence[float],
    d4_range: Sequence[float],
    luminosity: float,
    sqrt_s_tev: float,
    feature_profile: str,
) -> tuple[list[Path], dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    selected_surface = surfaces_by_scheme[selected_scheme]
    c3_grid, d4_grid = _surface_mesh(selected_surface)
    unitarity = _hhhh_perturbativity_grid(c3_grid, d4_grid)
    fields = selected_surface["fields"]
    nominal_clough = np.asarray(fields["background_x1"]["clough"], dtype=float)
    low_clough = np.asarray(fields["background_x0.25"]["clough"], dtype=float)
    high_clough = np.asarray(fields["background_x4"]["clough"], dtype=float)
    nominal_linear = np.asarray(fields["background_x1"]["linear"], dtype=float)
    outputs: list[Path] = []
    audit: dict[str, Any] = {
        "selected_scheme": selected_scheme,
        "fixed_coupling_95": fixed_coupling_95_intervals(selected_surface),
    }
    red = "#c9272c"
    pale_red = "#f5c6c8"
    level95 = SIMULTANEOUS_LEVELS["95"]
    level68 = SIMULTANEOUS_LEVELS["68"]

    # Main reference-style paper panel: only the 95% nominal contour.
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    finite_stress = np.isfinite(low_clough) & np.isfinite(high_clough)
    stress_difference = finite_stress & ((low_clough <= level95) != (high_clough <= level95))
    if np.any(stress_difference):
        ax.contourf(
            c3_grid,
            d4_grid,
            stress_difference.astype(float),
            levels=[0.5, 1.5],
            colors=[pale_red],
            alpha=0.72,
            zorder=1,
        )
    _, low_segments = _contour_segments(
        ax,
        c3_grid,
        d4_grid,
        low_clough,
        level95,
        colors=[red],
        linestyles=[":"],
        linewidths=[1.15],
        zorder=4,
    )
    _, high_segments = _contour_segments(
        ax,
        c3_grid,
        d4_grid,
        high_clough,
        level95,
        colors=[red],
        linestyles=["--"],
        linewidths=[1.15],
        zorder=4,
    )
    _, nominal_segments = _contour_segments(
        ax,
        c3_grid,
        d4_grid,
        nominal_clough,
        level95,
        colors=[red],
        linestyles=["-"],
        linewidths=[2.45],
        zorder=6,
    )
    _draw_physics_overlays(ax, c3_grid, d4_grid, unitarity)
    _configure_axis(ax, c3_range=c3_range, d4_range=d4_range)
    ax.set_title(
        rf"Resolved $8b$ multi-Higgs boson analysis, $\sqrt{{s}}={sqrt_s_tev:g}$ TeV, "
        rf"$\mathcal{{L}}={luminosity / 1000.0:g}\,\mathrm{{ab}}^{{-1}}$",
        fontsize=18.0,
        pad=10.0,
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=red,
                lw=2.45,
                label=r"This analysis: $hhhh+hhh b\bar b+hh+4b$ (95% CL)",
            ),
            Patch(
                facecolor=pale_red,
                edgecolor=red,
                alpha=0.72,
                label=r"$\frac{1}{4}\times B\;-\;4\times B$ variation",
            ),
            Line2D(
                [0],
                [0],
                color="blue",
                lw=2.0,
                label=r"ATLAS $hhh\to6b$ (95% CL, no syst.)",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                lw=1.25,
                ls="--",
                label=r"Perturbative unitarity, $hh\to hh$",
            ),
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.88,
        fontsize=8.5,
        borderpad=0.45,
        labelspacing=0.35,
        handlelength=2.5,
        handletextpad=0.65,
    )
    fig.tight_layout()
    outputs.extend(
        _save_figure(
            fig,
            output_dir
            / "paper"
            / (
                "c3d4_scorefit_95"
                if feature_profile == "core52"
                else f"c3d4_{feature_profile}_scorefit_95"
            ),
            "XGBoost score-fit expected 95 percent c3/d4 contour",
        )
    )
    plt.close(fig)
    paper_audit = {
        "nominal_clough_95": _segment_audit(nominal_segments),
        "background_quarter_clough_95": _segment_audit(low_segments),
        "background_times4_clough_95": _segment_audit(high_segments),
    }

    # Both simultaneous confidence levels for the selected analysis.
    fig, ax = plt.subplots(figsize=(8.0, 6.3))
    _, selected_68 = _contour_segments(
        ax,
        c3_grid,
        d4_grid,
        nominal_clough,
        level68,
        colors=[red],
        linestyles=["--"],
        linewidths=[1.7],
        zorder=5,
    )
    _, selected_95 = _contour_segments(
        ax,
        c3_grid,
        d4_grid,
        nominal_clough,
        level95,
        colors=[red],
        linestyles=["-"],
        linewidths=[2.3],
        zorder=5,
    )
    _draw_physics_overlays(ax, c3_grid, d4_grid, unitarity)
    _configure_axis(ax, c3_range=c3_range, d4_range=d4_range)
    ax.legend(
        handles=[
            Line2D([0], [0], color=red, lw=1.7, ls="--", label="Simultaneous 68%"),
            Line2D([0], [0], color=red, lw=2.3, label="Simultaneous 95%"),
            Line2D([0], [0], color="blue", lw=2.0, label=ATL_PHYS_PUB_2025_003_LABEL),
        ],
        frameon=False,
        fontsize=8.5,
        loc="best",
    )
    fig.tight_layout()
    outputs.extend(
        _save_figure(
            fig,
            output_dir / "diagnostics" / f"c3d4_{feature_profile}_scorefit_68_95",
            f"{feature_profile} score-fit simultaneous 68 and 95 percent contours",
        )
    )
    plt.close(fig)
    paper_audit["selected_clough_68"] = _segment_audit(selected_68)
    paper_audit["selected_clough_95"] = _segment_audit(selected_95)

    # Direct four-bin/five-bin comparison, with identical physical overlays.
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.35), sharex=True, sharey=True)
    comparison_audit: dict[str, Any] = {}
    for ax, scheme_name in zip(axes, BINNING_SCHEMES):
        scheme_surface = surfaces_by_scheme[scheme_name]
        scheme_grid_x, scheme_grid_y = _surface_mesh(scheme_surface)
        values = np.asarray(
            scheme_surface["fields"]["background_x1"]["clough"], dtype=float
        )
        _, segments68 = _contour_segments(
            ax,
            scheme_grid_x,
            scheme_grid_y,
            values,
            level68,
            colors=[red],
            linestyles=["--"],
            linewidths=[1.55],
        )
        _, segments95 = _contour_segments(
            ax,
            scheme_grid_x,
            scheme_grid_y,
            values,
            level95,
            colors=[red],
            linestyles=["-"],
            linewidths=[2.15],
        )
        _draw_physics_overlays(ax, scheme_grid_x, scheme_grid_y, unitarity)
        _configure_axis(ax, c3_range=c3_range, d4_range=d4_range)
        final_bins = len(templates_by_scheme[scheme_name]["background"])
        ax.set_title(f"{scheme_name.replace('_', ' ')} ({final_bins} final bins)")
        comparison_audit[scheme_name] = {
            "68": _segment_audit(segments68),
            "95": _segment_audit(segments95),
        }
    axes[1].set_ylabel("")
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=red, lw=1.55, ls="--", label="68%"),
            Line2D([0], [0], color=red, lw=2.15, label="95%"),
            Line2D([0], [0], color="blue", lw=2.0, label="ATLAS no syst."),
        ],
        frameon=False,
        fontsize=8.2,
        loc="best",
    )
    fig.tight_layout()
    outputs.extend(
        _save_figure(
            fig,
            output_dir / "diagnostics" / "c3d4_binning_comparison",
            "Four-bin and five-bin score-fit contour comparison",
        )
    )
    plt.close(fig)

    # Interpolation closure: the final Clough result beside a linear diagnostic.
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.35), sharex=True, sharey=True)
    interpolation_audit: dict[str, Any] = {}
    for ax, method, values in (
        (axes[0], "Clough--Tocher (paper)", nominal_clough),
        (axes[1], "Piecewise linear (diagnostic)", nominal_linear),
    ):
        _, segments68 = _contour_segments(
            ax,
            c3_grid,
            d4_grid,
            values,
            level68,
            colors=[red],
            linestyles=["--"],
            linewidths=[1.55],
        )
        _, segments95 = _contour_segments(
            ax,
            c3_grid,
            d4_grid,
            values,
            level95,
            colors=[red],
            linestyles=["-"],
            linewidths=[2.15],
        )
        _draw_physics_overlays(ax, c3_grid, d4_grid, unitarity, draw_atlas=False)
        _configure_axis(ax, c3_range=c3_range, d4_range=d4_range)
        ax.set_title(method)
        interpolation_audit[method] = {
            "68": _segment_audit(segments68),
            "95": _segment_audit(segments95),
        }
    axes[1].set_ylabel("")
    fig.tight_layout()
    outputs.extend(
        _save_figure(
            fig,
            output_dir / "diagnostics" / "c3d4_interpolation_comparison",
            "Clough-Tocher and linear interpolation comparison",
        )
    )
    plt.close(fig)

    # The selected SM score templates, shown as ordinal score bins.
    selected_templates = templates_by_scheme[selected_scheme]
    selected_likelihood = likelihood_by_scheme[selected_scheme]
    sm_index = int(selected_likelihood["sm_index"])
    hhhh_sm = np.asarray(selected_templates["hhhh"], dtype=float)[sm_index]
    hhhbb_sm = np.asarray(selected_templates["hhhbb"], dtype=float)[sm_index]
    hh4b_sm = np.asarray(selected_templates["hh4b"], dtype=float)
    background = np.asarray(selected_templates["background"], dtype=float)
    bins = np.arange(len(background))
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.bar(bins, background, color="#b8b8b8", edgecolor="black", linewidth=0.5, label="Background")
    ax.step(bins, hhhh_sm, where="mid", color="#1f77b4", lw=2.0, label=r"SM $hhhh$")
    ax.step(bins, hhhbb_sm, where="mid", color="#9467bd", lw=2.0, label=r"SM $hhh b\bar b$")
    ax.step(bins, hh4b_sm, where="mid", color="#ff7f0e", lw=2.0, label=r"SM $hh+4b$")
    ax.set_yscale("log")
    ax.set_xlabel("Ordinal background-quantile score bin (low to high score)")
    ax.set_ylabel("Expected events")
    ax.set_xticks(bins)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    outputs.extend(
        _save_figure(
            fig,
            output_dir / "diagnostics" / "selected_sm_score_templates",
            "Selected binned SM score templates",
        )
    )
    plt.close(fig)

    # Compact deterministic scan summary.
    names = [str(row["name"]) for row in scan_results]
    q4 = [float(row["schemes"]["background_quantile_4bin"]["validation_discovery_q"]) for row in scan_results]
    q5 = [float(row["schemes"]["background_quantile_5bin"]["validation_discovery_q"]) for row in scan_results]
    x = np.arange(len(names), dtype=float)
    width = 0.37
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(x - width / 2, q4, width, color="#4c78a8", label="Requested four-bin scheme")
    ax.bar(x + width / 2, q5, width, color="#f58518", label="Requested five-bin scheme")
    selected_index = names.index(selected_configuration)
    ax.axvspan(selected_index - 0.48, selected_index + 0.48, color="#54a24b", alpha=0.13)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel(r"Validation $q_{0,A}$")
    ax.set_title(f"Selected classifier: {selected_configuration}; selected binning: {selected_scheme}")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    outputs.extend(
        _save_figure(
            fig,
            output_dir / "diagnostics" / "xgboost_scan_summary",
            "Deterministic XGBoost and score-binning scan",
        )
    )
    plt.close(fig)

    clough95 = interpolation_audit["Clough--Tocher (paper)"]["95"]
    linear95 = interpolation_audit["Piecewise linear (diagnostic)"]["95"]
    topology_supported = (
        int(clough95["closed_component_count"]) == int(linear95["closed_component_count"])
        and int(clough95["open_component_count"]) == 0
    )
    audit.update(
        {
            "paper": paper_audit,
            "binning_comparison": comparison_audit,
            "interpolation_comparison": interpolation_audit,
            "clough_linear_closed_component_agreement_95": topology_supported,
            "atlas_overlay": {
                "label": ATL_PHYS_PUB_2025_003_LABEL,
                "source": ATL_PHYS_PUB_2025_003_SOURCE_URL,
                "figure": ATL_PHYS_PUB_2025_003_FIGURE,
            },
        }
    )
    return outputs, audit


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for name in ("scipy", "xgboost", "sklearn", "matplotlib"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as error:
            versions[name] = f"unavailable: {error}"
    try:
        import ROOT

        versions["ROOT"] = str(ROOT.gROOT.GetVersion())
    except Exception as error:
        versions["ROOT"] = f"unavailable: {error}"
    return versions


def _write_analysis_artifacts(
    *,
    output_dir: Path,
    scan_results: Sequence[Mapping[str, Any]],
    selection_audit: Mapping[str, Any],
    templates_by_scheme: Mapping[str, Mapping[str, Any]],
    likelihood_by_scheme: Mapping[str, Mapping[str, Any]],
    surfaces_by_scheme: Mapping[str, Mapping[str, Any]],
) -> list[Path]:
    outputs: list[Path] = []
    scan_json = output_dir / "scan" / "scan_results.json"
    _atomic_json(scan_json, {"schema": SCHEMA_VERSION, "results": scan_results, "selection": selection_audit})
    outputs.append(scan_json)
    scan_rows = []
    for row in scan_results:
        scan_rows.append(
            {
                "configuration": row["name"],
                "mean_binning_q": row["mean_binning_q"],
                "four_bin_q": row["schemes"]["background_quantile_4bin"]["validation_discovery_q"],
                "five_bin_q": row["schemes"]["background_quantile_5bin"]["validation_discovery_q"],
                "four_final_bins": row["schemes"]["background_quantile_4bin"]["final_bin_count"],
                "five_final_bins": row["schemes"]["background_quantile_5bin"]["final_bin_count"],
                "selected": row["name"] == selection_audit["selected_configuration"],
                "parameters": json.dumps(row["parameters"], sort_keys=True, separators=(",", ":")),
            }
        )
    scan_csv = output_dir / "scan" / "scan_results.csv"
    _atomic_csv(scan_csv, scan_rows)
    outputs.append(scan_csv)

    template_arrays: dict[str, Any] = {}
    template_metadata: dict[str, Any] = {"schema": SCHEMA_VERSION, "schemes": {}}
    point_rows: list[dict[str, Any]] = []
    surface_arrays: dict[str, Any] = {}
    surface_metadata: dict[str, Any] = {"schema": SCHEMA_VERSION, "schemes": {}}
    for scheme_name, templates in templates_by_scheme.items():
        prefix = scheme_name
        for key in (
            "points",
            "hhhh",
            "hhhbb",
            "hh4b",
            "background",
            "hhhh_neff",
            "hhhbb_neff",
            "hh4b_neff",
            "background_source_events",
            "background_neff",
        ):
            template_arrays[f"{prefix}__{key}"] = np.asarray(templates[key])
        for fold, edges in enumerate(templates["edges_by_fold"]):
            template_arrays[f"{prefix}__fold_{fold}_edges"] = np.asarray(edges, dtype=float)
        template_metadata["schemes"][scheme_name] = {
            "bin_count": int(len(templates["background"])),
            "edges_by_fold": [np.asarray(edges).tolist() for edges in templates["edges_by_fold"]],
            "background_gate_failures": templates["background_gate_failures"],
            "background_processes": templates["background_processes"],
            "yield_closure": templates["yield_closure"],
        }
        likelihood = likelihood_by_scheme[scheme_name]
        points = np.asarray(likelihood["points"], dtype=float)
        for index, (c3, d4) in enumerate(points):
            row: dict[str, Any] = {
                "scheme": scheme_name,
                "point_index": index,
                "c3": float(c3),
                "d4": float(d4),
                "inside_hh4b_valid_c3": bool(likelihood["valid_mask"][index]),
                "hh4b_rate_scale": float(likelihood["hh4b_rate_scales"][index]),
                "hhhh_events": float(np.sum(likelihood["hhhh"][index])),
                "hhhbb_events": float(np.sum(likelihood["hhhbb"][index])),
                "hh4b_events": float(
                    np.sum(likelihood["hh4b_reference"] * likelihood["hh4b_rate_scales"][index])
                ),
                "combined_signal_events": float(np.sum(likelihood["combined"][index])),
            }
            for factor in BACKGROUND_STRESS_FACTORS:
                key = f"background_x{factor:g}"
                row[f"q_{key}"] = float(likelihood["q"][key][index])
                row[f"eta_hat_{key}"] = float(likelihood["eta_hat"][key][index])
            point_rows.append(row)
        surface = surfaces_by_scheme[scheme_name]
        surface_arrays[f"{prefix}__c3_axis"] = np.asarray(surface["c3_axis"])
        surface_arrays[f"{prefix}__d4_axis"] = np.asarray(surface["d4_axis"])
        for factor_key, fields in surface["fields"].items():
            surface_arrays[f"{prefix}__{factor_key}__clough"] = np.asarray(fields["clough"])
            surface_arrays[f"{prefix}__{factor_key}__linear"] = np.asarray(fields["linear"])
        surface_metadata["schemes"][scheme_name] = {
            "coordinate_scales": surface["coordinate_scales"],
            "valid_points": surface["valid_points"],
            "audit": surface["audit"],
        }
    template_npz = output_dir / "templates" / "physical_score_templates.npz"
    template_json = output_dir / "templates" / "physical_score_templates.json"
    _atomic_npz(template_npz, template_arrays)
    _atomic_json(template_json, template_metadata)
    outputs.extend([template_npz, template_json])
    points_csv = output_dir / "likelihood" / "pointwise_q.csv"
    points_json = output_dir / "likelihood" / "pointwise_q.json"
    _atomic_csv(points_csv, point_rows)
    _atomic_json(points_json, {"schema": SCHEMA_VERSION, "rows": point_rows})
    outputs.extend([points_csv, points_json])
    surface_npz = output_dir / "surfaces" / "q_surfaces.npz"
    surface_json = output_dir / "surfaces" / "q_surfaces.json"
    _atomic_npz(surface_npz, surface_arrays)
    _atomic_json(surface_json, surface_metadata)
    outputs.extend([surface_npz, surface_json])
    return outputs


def _method_readme(
    *,
    feature_profile: str,
    selected_configuration: str,
    selected_scheme: str,
    background_norm_fraction: float,
) -> str:
    nuisance = (
        "disabled (statistics-only nominal result)"
        if background_norm_fraction == 0.0
        else f"enabled with fractional width {background_norm_fraction:g}"
    )
    return f"""# {feature_profile} XGBoost score-fit c3/d4 limits

This directory contains an expected SM+B Asimov analysis at 14 TeV and
3 ab^-1.  Five group-aware cross-fit classifiers are trained only on the
dedicated SM hhhh sample versus the registered backgrounds.  The selected
classifier configuration is `{selected_configuration}` and the selected score
binning is `{selected_scheme}`.

For each physical coupling point and score bin i, the expectation is

    nu_i = B_i + S_i(hhhh) + S_i(hhhbb) + r_hh4b(c3) S_i(hh4b, SM).

The observation is the SM signal-plus-background expectation and the reported
pointwise statistic is

    q = 2 sum_i [nu_i - n_i + n_i log(n_i / nu_i)].

The simultaneous two-parameter levels are q=2.30 (68%) and q=5.991 (95%).
When one coupling is fixed to its SM value, the 95% one-parameter interval
instead uses q=3.8414588.
The paper 95% curve uses Clough--Tocher interpolation of q after q has been
evaluated at every physical point.  Physical yields are never interpolated
before evaluating the likelihood.

The correlated background-normalization nuisance is {nuisance}.  The B/4 and
4B curves are stress tests, not a systematic-uncertainty interval.  No pyhf,
CLs construction, MC-statistical nuisance, signal-rate nuisance, or ATLAS-like
5b control region is used.  Absolute physical weights define the monotonic
background-quantile edges, while signed physical weights are retained in all
templates; every final Poisson background bin is required to have positive net
yield.

At completion the selected SM signal/background score-template table is
printed to the terminal and saved as `templates/selected_sm_score_fit_table.csv`.
"""


def _load_all_samples(
    *,
    manifest: Mapping[str, Any],
    repository: Path,
    luminosity: float,
    seed: int,
    load_jobs: int,
    start_time: float,
) -> tuple[dict[str, list[Any]], dict[str, list[dict[str, Any]]]]:
    observable_set = str(manifest.get("observable_set"))
    kinds = (
        "sm_signal",
        "grid_signal",
        "background",
        "postfit_hhhbb_signal",
        "postfit_sm_hh4b_signal",
    )
    specs_by_kind: dict[str, list[dict[str, Any]]] = {}
    records_by_kind: dict[str, list[dict[str, Any]]] = {}
    for kind in kinds:
        _progress(start_time, "inputs", f"verifying recorded SHA-256 values for {kind}")
        specs, records = _specs_from_manifest(manifest, kind, repository)
        specs_by_kind[kind] = specs
        records_by_kind[kind] = records
    _validate_dedicated_sm_signal(records_by_kind)
    samples: dict[str, list[Any]] = {}
    for kind in kinds:
        total = len(specs_by_kind[kind])

        def report(completed: int, expected: int, sample: Any, *, label: str = kind) -> None:
            if completed == expected or completed == 1 or completed % 20 == 0:
                _progress(
                    start_time,
                    "inputs",
                    f"loaded {completed}/{expected} {label} files (latest {sample.sample_id})",
                )

        samples[kind] = _load_samples(
            specs_by_kind[kind],
            progress=report,
            load_jobs=min(int(load_jobs), total),
            kind=kind,
            observable_set=observable_set,
            luminosity=luminosity,
            n_folds=N_FOLDS,
            seed=seed,
            max_events=None,
        )
    expected_counts = {
        "sm_signal": 1,
        "grid_signal": 153,
        "postfit_hhhbb_signal": 153,
        "postfit_sm_hh4b_signal": 1,
    }
    for kind, expected in expected_counts.items():
        if len(samples[kind]) != expected:
            raise ScoreFitError(f"Expected {expected} {kind} inputs, found {len(samples[kind])}")
    if not samples["background"]:
        raise ScoreFitError("No background inputs were loaded")
    for kind, items in samples.items():
        for sample in items:
            for label, values in (
                ("raw weights", sample.raw_weights),
                ("physical weights", sample.physical_weights),
            ):
                array = np.asarray(values, dtype=float)
                if np.any(~np.isfinite(array)):
                    raise ScoreFitError(f"{sample.path}: {label} must be finite")
    return samples, records_by_kind


def _signed_weight_diagnostics(samples: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for kind, items in samples.items():
        rows = []
        for sample in items:
            raw = np.asarray(sample.raw_weights, dtype=float)
            physical = np.asarray(sample.physical_weights, dtype=float)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "entries": int(len(raw)),
                    "negative_raw_entries": int(np.count_nonzero(raw < 0.0)),
                    "negative_physical_entries": int(np.count_nonzero(physical < 0.0)),
                    "raw_weight_sum": float(np.sum(raw)),
                    "absolute_raw_weight_sum": float(np.sum(np.abs(raw))),
                    "physical_weight_sum": float(np.sum(physical)),
                    "absolute_physical_weight_sum": float(np.sum(np.abs(physical))),
                }
            )
        output[kind] = rows
    return output


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.monotonic()
    study_dir = Path(args.study_dir).expanduser().resolve()
    repository = Path(args.repository).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_failure = output_dir / "failure.json"
    if stale_failure.is_file():
        archive = (
            output_dir
            / "previous_failures"
            / f"failure-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stale_failure, archive)
    manifest_path = study_dir / "method_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing completed study manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping) or manifest.get("status") != "complete":
        raise ScoreFitError("The source study manifest is not complete")
    if str(manifest.get("observable_set")) != "extended-91-v2":
        raise ScoreFitError("The score fit requires the extended-91-v2 observable source")
    if int(manifest.get("cv_folds", -1)) != N_FOLDS:
        raise ScoreFitError("The source manifest does not use the required five folds")
    if int(args.scan_jobs) * int(args.xgboost_threads) > int(args.cpu_budget):
        raise ScoreFitError("scan_jobs*xgboost_threads exceeds cpu_budget")
    if int(args.score_jobs) * int(args.prediction_threads) > int(args.cpu_budget):
        raise ScoreFitError("score_jobs*prediction_threads exceeds cpu_budget")
    if int(args.grid_bins) < 101:
        raise ScoreFitError("grid_bins must be at least 101")
    feature_profile = str(args.feature_profile)
    expected_feature_count = FEATURE_PROFILES[feature_profile]
    luminosity = float(manifest.get("luminosity_fb_inverse", args.luminosity))
    if not math.isfinite(luminosity) or luminosity <= 0.0:
        raise ScoreFitError("The source manifest has no positive luminosity")
    driver_path = Path(__file__).resolve()
    manifest_sha = _sha256(manifest_path)
    driver_sha = _sha256(driver_path)
    configurations = _scan_configurations()
    run_contract = {
        "schema": SCHEMA_VERSION,
        "builder": BUILDER_VERSION,
        "source_manifest_sha256": manifest_sha,
        "driver_sha256": driver_sha,
        "observable_set": "extended-91-v2",
        "feature_profile": feature_profile,
        "training_strategy": "sm-crossfit-v2",
        "seed": int(args.seed),
        "folds": N_FOLDS,
        "configurations": configurations,
        "binning_schemes": BINNING_SCHEMES,
        "min_background_source_events": int(args.min_background_source_events),
        "min_background_neff": float(args.min_background_neff),
        "background_norm_fraction": float(args.background_norm_fraction),
        "luminosity_fb_inverse": luminosity,
    }
    run_fingerprint = _sha256_payload(run_contract)
    initial_manifest = {
        **run_contract,
        "status": "running",
        "run_fingerprint": run_fingerprint,
        "started_utc": _utc_now(),
        "study_dir": str(study_dir),
        "repository": str(repository),
        "output_dir": str(output_dir),
        "parallelism": {
            "cpu_budget": int(args.cpu_budget),
            "load_jobs": int(args.load_jobs),
            "scan_jobs": int(args.scan_jobs),
            "xgboost_threads": int(args.xgboost_threads),
            "score_jobs": int(args.score_jobs),
            "prediction_threads": int(args.prediction_threads),
        },
    }
    _atomic_json(output_dir / "run_manifest.json", initial_manifest)
    _progress(start_time, "start", f"run fingerprint {run_fingerprint}")

    samples, input_records = _load_all_samples(
        manifest=manifest,
        repository=repository,
        luminosity=luminosity,
        seed=int(args.seed),
        load_jobs=int(args.load_jobs),
        start_time=start_time,
    )
    _atomic_json(
        output_dir / "inputs" / "verified_inputs.json",
        {
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": manifest_sha,
            "inputs": input_records,
            "signed_weight_diagnostics": _signed_weight_diagnostics(samples),
            "bin_edge_weight_policy": "absolute physical weights",
            "template_weight_policy": "signed physical weights",
        },
    )
    sm_samples = samples["sm_signal"]
    grid_samples = samples["grid_signal"]
    background_samples = samples["background"]
    hhhbb_samples = samples["postfit_hhhbb_signal"]
    hh4b_samples = samples["postfit_sm_hh4b_signal"]
    hh4b_sample = hh4b_samples[0]
    hh4b_fit = dict((hh4b_sample.metadata or {}).get("c3_cross_section_fit") or {})
    reference_hh4b = evaluate_hh4b_c3_fit(hh4b_fit, 0.0)
    archived_hh4b_xsec = float(hh4b_sample.xsec_fb)
    hh4b_sample.physical_weights = (
        float(reference_hh4b["cross_section_fb"]) * hh4b_sample.unit_xsec_weights
    )
    hh4b_sample.xsec_fb = float(reference_hh4b["cross_section_fb"])
    profile_indices = _profile_indices("extended-91-v2", feature_profile)
    if len(profile_indices) != expected_feature_count:
        raise ScoreFitError(
            f"The {feature_profile} contract exposes {len(profile_indices)} features; "
            f"expected {expected_feature_count}"
        )

    _progress(start_time, "training", "building the five immutable training arrays")
    arrays_by_fold = [
        _training_arrays(
            sm_samples,
            grid_samples,
            background_samples,
            strategy="sm-crossfit-v2",
            profile_indices=profile_indices,
            rotation=fold,
            n_folds=N_FOLDS,
        )
        for fold in range(N_FOLDS)
    ]
    trained: dict[tuple[str, int], dict[str, Any]] = {}
    futures = {}
    source_commit = str(manifest.get("source_commit") or "source-manifest-uncommitted")
    with ThreadPoolExecutor(max_workers=int(args.scan_jobs)) as executor:
        for configuration in configurations:
            for fold in range(N_FOLDS):
                future = executor.submit(
                    _fit_and_validate_one,
                    config=configuration,
                    fold=fold,
                    training_arrays=arrays_by_fold[fold],
                    sm_samples=sm_samples,
                    background_samples=background_samples,
                    profile_indices=profile_indices,
                    observable_set="extended-91-v2",
                    profile=feature_profile,
                    source_commit=source_commit,
                    seed=int(args.seed),
                    xgboost_threads=int(args.xgboost_threads),
                    output_dir=output_dir,
                    run_fingerprint=run_fingerprint,
                )
                futures[future] = (str(configuration["name"]), fold)
        completed = 0
        for future in as_completed(futures):
            key = futures[future]
            trained[key] = future.result()
            completed += 1
            _progress(start_time, "training", f"completed {completed}/{len(futures)} fold/configuration fits")
    scan_results = []
    for configuration in configurations:
        name = str(configuration["name"])
        scan_results.append(
            _evaluate_configuration(
                configuration,
                [trained[(name, fold)] for fold in range(N_FOLDS)],
                min_source_events=int(args.min_background_source_events),
                min_neff=float(args.min_background_neff),
            )
        )
    selected_configuration, selected_scheme, selection_audit = select_scan_result(scan_results)
    _progress(
        start_time,
        "selection",
        f"selected {selected_configuration} with {selected_scheme}",
    )
    selected_scan = next(row for row in scan_results if row["name"] == selected_configuration)
    selected_models = [trained[(selected_configuration, fold)]["model"] for fold in range(N_FOLDS)]

    _progress(start_time, "scoring", "scoring all physical samples in five held-out folds")
    scored: list[dict[str, Any] | None] = [None] * N_FOLDS
    with ThreadPoolExecutor(max_workers=int(args.score_jobs)) as executor:
        score_futures = {
            executor.submit(
                _score_test_fold,
                fold=fold,
                model=selected_models[fold],
                grid_samples=grid_samples,
                hhhbb_samples=hhhbb_samples,
                hh4b_samples=hh4b_samples,
                background_samples=background_samples,
                profile_indices=profile_indices,
                prediction_threads=int(args.prediction_threads),
            ): fold
            for fold in range(N_FOLDS)
        }
        for future in as_completed(score_futures):
            fold = score_futures[future]
            scored[fold] = future.result()
            _progress(start_time, "scoring", f"completed held-out fold {fold + 1}/{N_FOLDS}")
    if any(row is None for row in scored):
        raise ScoreFitError("Held-out scoring did not return every fold")
    scored_folds = [row for row in scored if row is not None]

    templates_by_scheme: dict[str, Any] = {}
    likelihood_by_scheme: dict[str, Any] = {}
    surfaces_by_scheme: dict[str, Any] = {}
    for scheme_name in BINNING_SCHEMES:
        edges = selected_scan["schemes"][scheme_name]["final_edges_by_fold"]
        templates = build_physical_templates(
            scheme_name=scheme_name,
            edges_by_fold=edges,
            scored_folds=scored_folds,
            grid_samples=grid_samples,
            hhhbb_samples=hhhbb_samples,
            hh4b_sample=hh4b_sample,
            background_samples=background_samples,
            min_source_events=int(args.min_background_source_events),
            min_neff=float(args.min_background_neff),
        )
        likelihood = evaluate_likelihood_points(
            templates,
            hh4b_fit,
            background_norm_fraction=float(args.background_norm_fraction),
        )
        surface = interpolate_q_surfaces(
            likelihood,
            c3_range=(float(args.c3_min), float(args.c3_max)),
            d4_range=(float(args.d4_min), float(args.d4_max)),
            grid_bins=int(args.grid_bins),
        )
        templates_by_scheme[scheme_name] = templates
        likelihood_by_scheme[scheme_name] = likelihood
        surfaces_by_scheme[scheme_name] = surface
        _progress(start_time, "likelihood", f"built {scheme_name} physical q surface")

    artifact_outputs = _write_analysis_artifacts(
        output_dir=output_dir,
        scan_results=scan_results,
        selection_audit=selection_audit,
        templates_by_scheme=templates_by_scheme,
        likelihood_by_scheme=likelihood_by_scheme,
        surfaces_by_scheme=surfaces_by_scheme,
    )
    plot_outputs, plot_audit = make_plots(
        output_dir=output_dir,
        scan_results=scan_results,
        selected_configuration=selected_configuration,
        selected_scheme=selected_scheme,
        templates_by_scheme=templates_by_scheme,
        likelihood_by_scheme=likelihood_by_scheme,
        surfaces_by_scheme=surfaces_by_scheme,
        c3_range=(float(args.c3_min), float(args.c3_max)),
        d4_range=(float(args.d4_min), float(args.d4_max)),
        luminosity=luminosity,
        sqrt_s_tev=float(args.sqrt_s_tev),
        feature_profile=feature_profile,
    )
    readme_path = output_dir / "README.md"
    _atomic_text(
        readme_path,
        _method_readme(
            feature_profile=feature_profile,
            selected_configuration=selected_configuration,
            selected_scheme=selected_scheme,
            background_norm_fraction=float(args.background_norm_fraction),
        ),
    )
    chosen_templates = templates_by_scheme[selected_scheme]
    template_table_rows = selected_sm_template_table_rows(
        templates=chosen_templates,
        grid_samples=grid_samples,
        hhhbb_samples=hhhbb_samples,
        hh4b_sample=hh4b_sample,
        background_samples=background_samples,
        luminosity=luminosity,
    )
    template_table_path = output_dir / "templates" / "selected_sm_score_fit_table.csv"
    _atomic_csv(template_table_path, template_table_rows)
    template_table_text = terminal_selected_sm_template_table(
        template_table_rows,
        templates=chosen_templates,
        luminosity=luminosity,
        selected_configuration=selected_configuration,
        selected_scheme=selected_scheme,
    )
    topology_ok = bool(plot_audit["clough_linear_closed_component_agreement_95"])
    occupancy_ok = not bool(chosen_templates["background_gate_failures"])
    paper_contour = plot_audit["paper"]["nominal_clough_95"]
    contour_ok = int(paper_contour["component_count"]) > 0 and int(paper_contour["open_component_count"]) == 0
    paper_ready = bool(topology_ok and occupancy_ok and contour_ok)
    all_outputs = [
        output_dir / "inputs" / "verified_inputs.json",
        *artifact_outputs,
        *plot_outputs,
        template_table_path,
        readme_path,
    ]
    output_records = [
        {"path": str(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)}
        for path in all_outputs
    ]
    final_manifest = {
        **initial_manifest,
        "status": "complete",
        "completed_utc": _utc_now(),
        "elapsed_seconds": float(time.monotonic() - start_time),
        "package_versions": _package_versions(),
        "input_counts": {key: len(value) for key, value in samples.items()},
        "signed_weight_policy": {
            "classifier_and_quantile_edges": "absolute weights",
            "physical_templates": "signed weights",
            "poisson_gate": "every final background bin must have positive net yield",
        },
        "selected_configuration": selected_configuration,
        "selected_scheme": selected_scheme,
        "selected_sm_template_table": str(template_table_path),
        "selection": selection_audit,
        "scan_results_path": str(output_dir / "scan" / "scan_results.json"),
        "hh4b": {
            "archived_sample_cross_section_fb": archived_hh4b_xsec,
            "fitted_sm_cross_section_fb": float(reference_hh4b["cross_section_fb"]),
            "valid_c3": likelihood_by_scheme[selected_scheme]["valid_c3"],
            "valid_point_count": likelihood_by_scheme[selected_scheme]["valid_point_count"],
            "diagnostic_extrapolated_point_count": likelihood_by_scheme[selected_scheme]["diagnostic_extrapolated_point_count"],
            "shape_policy": "fixed SM score shape and efficiency; quadratic c3 rate; d4 independent",
        },
        "likelihood": {
            "observation": "SM signal plus background Asimov",
            "signals": ["hhhh", "hhhbb", "hh4b"],
            "simultaneous_levels": SIMULTANEOUS_LEVELS,
            "fixed_coupling_95_level": FIXED_COUPLING_95_LEVEL,
            "background_stress_factors": BACKGROUND_STRESS_FACTORS,
            "background_norm_fraction": float(args.background_norm_fraction),
            "pyhf": False,
            "cls": False,
            "mc_stat_nuisance": False,
        },
        "plot_audit": plot_audit,
        "paper_ready": paper_ready,
        "paper_ready_checks": {
            "selected_background_occupancy": occupancy_ok,
            "nominal_95_contour_closed_and_present": contour_ok,
            "clough_linear_topology_agreement": topology_ok,
        },
        "outputs": output_records,
    }
    _atomic_json(output_dir / "run_manifest.json", final_manifest)
    print()
    print(template_table_text, flush=True)
    _progress(start_time, "complete", f"paper_ready={paper_ready}; outputs at {output_dir}")
    return final_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SM cross-fit classifiers and build transparent binned Poisson c3/d4 contours."
    )
    parser.add_argument("--study-dir", required=True, type=Path, help="Completed source study containing method_manifest.json.")
    parser.add_argument("--repository", required=True, type=Path, help="Repository against which relative manifest paths resolve.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New isolated output directory.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--luminosity", type=float, default=3000.0, help="Fallback only; the source manifest value is authoritative.")
    parser.add_argument("--sqrt-s-tev", type=float, default=14.0)
    parser.add_argument(
        "--feature-profile",
        choices=tuple(FEATURE_PROFILES),
        default="core52",
        help="Observable feature contract; core52 remains the default publication setup.",
    )
    parser.add_argument("--load-jobs", type=int, default=48)
    parser.add_argument("--scan-jobs", type=int, default=30)
    parser.add_argument("--xgboost-threads", type=int, default=6)
    parser.add_argument("--score-jobs", type=int, default=5)
    parser.add_argument("--prediction-threads", type=int, default=36)
    parser.add_argument("--cpu-budget", type=int, default=180)
    parser.add_argument("--min-background-source-events", type=int, default=DEFAULT_MIN_BACKGROUND_SOURCE_EVENTS)
    parser.add_argument("--min-background-neff", type=float, default=DEFAULT_MIN_BACKGROUND_NEFF)
    parser.add_argument("--background-norm-fraction", type=float, default=0.0)
    parser.add_argument("--c3-min", type=float, default=-20.0)
    parser.add_argument("--c3-max", type=float, default=20.0)
    parser.add_argument("--d4-min", type=float, default=-300.0)
    parser.add_argument("--d4-max", type=float, default=300.0)
    parser.add_argument("--grid-bins", type=int, default=501)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_analysis(args)
    except Exception as error:
        output_dir = Path(args.output_dir).expanduser().resolve()
        try:
            _atomic_json(
                output_dir / "failure.json",
                {
                    "schema": SCHEMA_VERSION,
                    "status": "failed",
                    "recorded_utc": _utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        except Exception:
            pass
        print(f"score-fit failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
