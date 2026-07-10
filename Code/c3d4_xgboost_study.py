"""Numerical building blocks for the resolved-8b c3/d4 XGBoost v2 study.

This module deliberately does not know how ROOT files are laid out and does not
train a particular classifier.  The driver supplies NumPy arrays, while this
module owns the reproducible cross-fitting, training-weight, threshold, binning,
and statistical conventions of the v2 study.

There are three distinct kinds of weights throughout the public API:

* ``physical_weights`` are signed expected-event weights and are used for all
  efficiencies, yields, and limits;
* classifier weights are non-negative and are constructed from the absolute
  physical weights; and
* a signal template passed to :func:`pyhf_combined_limit` is the expected yield
  for a production cross section of exactly 1 fb, making the pyhf POI a cross
  section in fb.

Optional dependencies (pyhf and Optuna) are imported only by the functions that
need them.  This keeps schema/weight/fold tests runnable in the lightweight
analysis environment.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


METHOD_VERSION = "c3d4-xgboost-study-v2"
DEFAULT_SEED = 12345
DEFAULT_N_FOLDS = 5
DEFAULT_THRESHOLDS = np.linspace(0.0, 1.0, 1001)
DEFAULT_BIN_QUANTILES = (0.0, 0.50, 0.75, 0.90, 0.97, 1.0)

CURRENT_XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


class NoValidThresholdError(ValueError):
    """Raised when no threshold meets the signal/background requirements."""


class NoValidBinningError(ValueError):
    """Raised when no validation-defined score binning is statistically valid."""


def _array_1d(values: Sequence[Any], name: str, dtype: Any = float) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return array


def _finite_array(values: Sequence[Any], name: str) -> np.ndarray:
    array = _array_1d(values, name, float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _matching_arrays(*named_arrays: tuple[str, Sequence[Any]]) -> list[np.ndarray]:
    arrays = [_array_1d(values, name, None) for name, values in named_arrays]
    lengths = {len(array) for array in arrays}
    if len(lengths) > 1:
        names = ", ".join(name for name, _ in named_arrays)
        raise ValueError(f"{names} must have matching lengths")
    return arrays


def stable_seed(identifier: Any, seed: int = DEFAULT_SEED) -> int:
    """Return a process-independent uint32 seed for ``identifier``.

    Python's built-in hash is intentionally randomized between processes, so a
    SHA-256 digest is used to make fold assignments reproducible across hosts.
    """

    payload = f"{int(seed)}\0{identifier}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def deterministic_folds(
    source_ids: Sequence[Any],
    event_indices: Sequence[int] | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Assign source-local events to deterministic, balanced folds.

    Events from each source are sorted by their original entry index, shuffled
    with a source-derived seed, then assigned round-robin.  Consequently every
    source's fold populations differ by at most one.  The result is independent
    of the input row order as long as ``(source_id, event_index)`` is unique.
    """

    if int(n_folds) != n_folds or int(n_folds) < 2:
        raise ValueError("n_folds must be an integer greater than one")
    n_folds = int(n_folds)
    sources = _array_1d(source_ids, "source_ids", None)
    if event_indices is None:
        entries = np.arange(len(sources), dtype=np.int64)
    else:
        entries = _array_1d(event_indices, "event_indices", None)
        if len(entries) != len(sources):
            raise ValueError("source_ids and event_indices must have matching lengths")
        try:
            entries = entries.astype(np.int64)
        except (TypeError, ValueError) as error:
            raise ValueError("event_indices must be integers") from error

    folds = np.empty(len(sources), dtype=np.int16)
    source_keys = np.asarray([str(item) for item in sources], dtype=object)
    for source in sorted(set(source_keys.tolist())):
        positions = np.flatnonzero(source_keys == source)
        local_entries = entries[positions]
        if len(np.unique(local_entries)) != len(local_entries):
            raise ValueError(f"source {source!r} contains duplicate event_indices")
        canonical_positions = positions[np.argsort(local_entries, kind="mergesort")]
        rng = np.random.default_rng(stable_seed(source, seed))
        shuffled = canonical_positions[rng.permutation(len(canonical_positions))]
        folds[shuffled] = np.arange(len(shuffled), dtype=np.int64) % n_folds
    return folds


def rotation_masks(
    folds: Sequence[int], rotation: int, n_folds: int = DEFAULT_N_FOLDS
) -> dict[str, np.ndarray | int]:
    """Return train/validation/test masks for one rotating cross-fit split."""

    fold_array = _array_1d(folds, "folds", int)
    n_folds = int(n_folds)
    rotation = int(rotation)
    if n_folds < 3:
        raise ValueError("rotating train/validation/test evaluation needs at least 3 folds")
    if rotation < 0 or rotation >= n_folds:
        raise ValueError(f"rotation must be in [0, {n_folds - 1}]")
    if np.any((fold_array < 0) | (fold_array >= n_folds)):
        raise ValueError("fold labels are outside the requested fold range")

    test_fold = rotation
    validation_fold = (rotation + 1) % n_folds
    test = fold_array == test_fold
    validation = fold_array == validation_fold
    train = ~(test | validation)
    return {
        "rotation": rotation,
        "test_fold": test_fold,
        "validation_fold": validation_fold,
        "train": train,
        "validation": validation,
        "test": test,
    }


def crossfit_rotations(
    folds: Sequence[int], n_folds: int = DEFAULT_N_FOLDS
) -> list[dict[str, np.ndarray | int]]:
    """Return all rotations and verify that every event is tested once."""

    rotations = [rotation_masks(folds, index, n_folds) for index in range(int(n_folds))]
    test_multiplicity = np.sum([item["test"] for item in rotations], axis=0)
    if not np.all(test_multiplicity == 1):
        raise AssertionError("cross-fitting invariant failed: every event must be tested once")
    return rotations


def weighted_yield(weights: Sequence[float], mask: Sequence[bool] | None = None) -> dict[str, float | int]:
    """Summarize a signed physical yield and its Monte Carlo precision."""

    values = _finite_array(weights, "weights")
    if mask is not None:
        selected = _array_1d(mask, "mask", bool)
        if len(selected) != len(values):
            raise ValueError("weights and mask must have matching lengths")
        values = values[selected]
    sumw = float(np.sum(values, dtype=np.float64))
    sumw2 = float(np.sum(np.square(values), dtype=np.float64))
    neff = float(sumw * sumw / sumw2) if sumw2 > 0.0 else 0.0
    return {
        "yield": sumw,
        "sumw2": sumw2,
        "uncertainty": math.sqrt(sumw2),
        "raw_entries": int(len(values)),
        "effective_entries": neff,
    }


def pooled_equal_point_weights(
    physical_weights: Sequence[float],
    point_ids: Sequence[Any],
    expected_points: int | Sequence[Any] | None = None,
) -> np.ndarray:
    """Construct non-negative signal weights with equal total per c3/d4 point.

    Within a point, absolute physical-weight ratios are preserved.  Each point
    receives total classifier weight ``1 / n_points``.
    """

    physical = _finite_array(physical_weights, "physical_weights")
    points = _array_1d(point_ids, "point_ids", None)
    if len(points) != len(physical):
        raise ValueError("physical_weights and point_ids must have matching lengths")
    point_keys = np.asarray([str(item) for item in points], dtype=object)
    unique = sorted(set(point_keys.tolist()))
    if not unique:
        return np.asarray([], dtype=float)

    if expected_points is not None:
        if isinstance(expected_points, (int, np.integer)):
            if len(unique) != int(expected_points):
                raise ValueError(f"expected {int(expected_points)} signal points, found {len(unique)}")
        else:
            expected_keys = {str(item) for item in expected_points}
            actual_keys = set(unique)
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                extra = sorted(actual_keys - expected_keys)
                raise ValueError(f"signal point mismatch (missing={missing}, extra={extra})")

    weights = np.zeros(len(physical), dtype=float)
    point_total = 1.0 / len(unique)
    for point in unique:
        selected = point_keys == point
        magnitude = np.abs(physical[selected])
        denominator = float(np.sum(magnitude))
        if denominator <= 0.0:
            raise ValueError(f"signal point {point!r} has no non-zero physical training weight")
        weights[selected] = magnitude * (point_total / denominator)
    return weights


def balanced_binary_training_weights(
    labels: Sequence[int],
    base_weights: Sequence[float],
    class_total: float = 1.0,
) -> np.ndarray:
    """Normalize non-negative base weights to equal signal/background totals."""

    labels_array = _array_1d(labels, "labels", int)
    weights = _finite_array(base_weights, "base_weights")
    if len(labels_array) != len(weights):
        raise ValueError("labels and base_weights must have matching lengths")
    if np.any(weights < 0.0):
        raise ValueError("classifier base_weights must be non-negative")
    classes = np.unique(labels_array)
    if not np.array_equal(classes, np.asarray([0, 1])):
        raise ValueError("labels must contain both binary classes 0 and 1")
    class_total = float(class_total)
    if not math.isfinite(class_total) or class_total <= 0.0:
        raise ValueError("class_total must be positive and finite")

    result = np.zeros_like(weights)
    for label in classes:
        selected = labels_array == label
        denominator = float(np.sum(weights[selected]))
        if denominator <= 0.0:
            raise ValueError(f"class {label} has no non-zero classifier weight")
        result[selected] = weights[selected] * (class_total / denominator)
    return result


def pooled_classifier_training_weights(
    labels: Sequence[int],
    physical_weights: Sequence[float],
    signal_point_ids: Sequence[Any],
    expected_points: int | Sequence[Any] | None = 57,
    signal_label: int = 1,
) -> np.ndarray:
    """Build the complete pooled-training weights used by XGBoost.

    Signal points first receive equal total weight.  Background weights retain
    their absolute physical ratios.  The final normalization gives the signal
    and background classes equal total classifier weight.
    """

    labels_array = _array_1d(labels, "labels", int)
    physical = _finite_array(physical_weights, "physical_weights")
    if len(labels_array) != len(physical):
        raise ValueError("labels and physical_weights must have matching lengths")
    signal = labels_array == int(signal_label)
    background = ~signal
    if not np.any(signal) or not np.any(background):
        raise ValueError("training data must contain signal and background")

    point_ids = _array_1d(signal_point_ids, "signal_point_ids", None)
    if len(point_ids) == len(labels_array):
        point_ids = point_ids[signal]
    elif len(point_ids) != int(np.sum(signal)):
        raise ValueError("signal_point_ids must describe signal rows or all rows")

    base = np.zeros_like(physical)
    base[signal] = pooled_equal_point_weights(
        physical[signal], point_ids, expected_points=expected_points
    )
    base[background] = np.abs(physical[background])
    binary_labels = signal.astype(int)
    return balanced_binary_training_weights(binary_labels, base)


@functools.lru_cache(maxsize=1)
def _scipy_poisson_functions() -> tuple[Any, Any] | None:
    """Load exact incomplete-gamma helpers when SciPy is available."""

    try:
        from scipy.special import gammaincc, gammainccinv  # type: ignore
    except ImportError:
        return None
    return gammaincc, gammainccinv


def _poisson_cdf(observed_events: int, mean: float) -> float:
    observed_events = int(observed_events)
    mean = float(mean)
    if observed_events < 0:
        return 0.0
    if mean < 0.0 or not math.isfinite(mean):
        raise ValueError("Poisson mean must be finite and non-negative")
    if mean == 0.0:
        return 1.0
    scipy_functions = _scipy_poisson_functions()
    if scipy_functions is not None:
        gammaincc, _ = scipy_functions
        return float(gammaincc(observed_events + 1.0, mean))
    logs = [
        -mean + k * math.log(mean) - math.lgamma(k + 1.0)
        for k in range(observed_events + 1)
    ]
    maximum = max(logs)
    if maximum < math.log(np.finfo(float).tiny):
        return 0.0
    value = math.exp(maximum) * sum(math.exp(item - maximum) for item in logs)
    return float(min(max(value, 0.0), 1.0))


def poisson_median_observed(background_yield: float) -> int:
    """Return the lower integer median of a background-only Poisson model."""

    background_yield = float(background_yield)
    if not math.isfinite(background_yield) or background_yield < 0.0:
        raise ValueError("background_yield must be finite and non-negative")
    if background_yield == 0.0:
        return 0
    # Start near the median, then verify the discrete quantile.  With SciPy the
    # checks use the regularized incomplete gamma function; the dependency-free
    # fallback needs only the one or two nearby CDF evaluations.
    if background_yield > 100.0:
        observed = max(
            0,
            int(math.floor(background_yield + 1.0 / 3.0 - 0.02 / background_yield)),
        )
    else:
        observed = 0
    while _poisson_cdf(observed, background_yield) < 0.5:
        observed += 1
    while observed > 0 and _poisson_cdf(observed - 1, background_yield) >= 0.5:
        observed -= 1
    return observed


@functools.lru_cache(maxsize=250_000)
def _cached_exact_cls_signal_upper_limit(
    background_yield: float,
    confidence_level: float,
    observed_events: int | None,
) -> float:
    if observed_events is None:
        observed_events = poisson_median_observed(background_yield)
    observed_events = int(observed_events)

    background_cdf = _poisson_cdf(observed_events, background_yield)
    if background_cdf <= 0.0:
        raise ValueError("cannot compute CLs with a zero background-only CDF")
    alpha = 1.0 - confidence_level

    scipy_functions = _scipy_poisson_functions()
    if scipy_functions is not None:
        _, gammainccinv = scipy_functions
        total_mean = float(
            gammainccinv(observed_events + 1.0, alpha * background_cdf)
        )
        signal_yield = total_mean - background_yield
        if not math.isfinite(signal_yield) or signal_yield < 0.0:
            raise RuntimeError("incomplete-gamma inversion returned an invalid CLs limit")
        return signal_yield

    def cls(signal_yield: float) -> float:
        return _poisson_cdf(observed_events, background_yield + signal_yield) / background_cdf

    low = 0.0
    high = max(1.0, math.sqrt(background_yield + observed_events + 1.0) + 3.0)
    while cls(high) > alpha:
        high *= 2.0
        if high > 1.0e12:
            raise RuntimeError("failed to bracket the exact Poisson CLs upper limit")
    for _ in range(120):
        middle = 0.5 * (low + high)
        if cls(middle) > alpha:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))


def exact_cls_signal_upper_limit(
    background_yield: float,
    confidence_level: float = 0.95,
    observed_events: int | None = None,
) -> float:
    """Return the exact one-bin Poisson CLs upper limit on signal events.

    With ``observed_events=None`` this is the expected limit for the integer
    median background-only observation, matching the retained legacy method.
    Results are cached by exact background yield so the 57 signal points can
    reuse the common background scan in every model/fold evaluation.
    """

    background_yield = float(background_yield)
    confidence_level = float(confidence_level)
    if not math.isfinite(background_yield) or background_yield < 0.0:
        raise ValueError("background_yield must be finite and non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if observed_events is not None:
        if int(observed_events) != observed_events or int(observed_events) < 0:
            raise ValueError("observed_events must be a non-negative integer")
        observed_events = int(observed_events)
    return _cached_exact_cls_signal_upper_limit(
        background_yield, confidence_level, observed_events
    )


def threshold_weight_scan(
    scores: Sequence[float], weights: Sequence[float], thresholds: Sequence[float]
) -> dict[str, np.ndarray]:
    """Vectorized signed yield/sumw2/raw/Neff summaries above thresholds."""

    score_array = _finite_array(scores, "scores")
    weight_array = _finite_array(weights, "weights")
    scan = _finite_array(thresholds, "thresholds")
    if len(score_array) != len(weight_array):
        raise ValueError("scores and weights must have matching lengths")
    order = np.argsort(score_array, kind="mergesort")
    ordered_scores = score_array[order]
    ordered_weights = weight_array[order]
    n_events = len(ordered_scores)
    suffix_sumw = np.concatenate(
        [np.cumsum(ordered_weights[::-1], dtype=np.float64)[::-1], [0.0]]
    )
    suffix_sumw2 = np.concatenate(
        [np.cumsum(np.square(ordered_weights)[::-1], dtype=np.float64)[::-1], [0.0]]
    )
    indices = np.searchsorted(ordered_scores, scan, side="left")
    sumw = suffix_sumw[indices]
    sumw2 = suffix_sumw2[indices]
    raw = n_events - indices
    neff = np.divide(np.square(sumw), sumw2, out=np.zeros_like(sumw), where=sumw2 > 0.0)
    return {
        "thresholds": scan.copy(),
        "yield": sumw,
        "sumw2": sumw2,
        "raw_entries": raw.astype(int),
        "effective_entries": neff,
    }


def background_threshold_scan(
    background_scores: Sequence[float],
    background_weights: Sequence[float],
    thresholds: Sequence[float] | None = None,
    confidence_level: float = 0.95,
) -> dict[str, np.ndarray]:
    """Precompute a reusable background scan, including exact S95 values."""

    scan = DEFAULT_THRESHOLDS.copy() if thresholds is None else _finite_array(
        thresholds, "thresholds"
    )
    summary = threshold_weight_scan(background_scores, background_weights, scan)
    s95 = np.full(len(scan), np.nan, dtype=float)
    for index, background_yield in enumerate(summary["yield"]):
        if background_yield >= 0.0:
            s95[index] = exact_cls_signal_upper_limit(
                float(background_yield), confidence_level=confidence_level
            )
    summary["s95_events"] = s95
    summary["confidence_level"] = np.asarray([float(confidence_level)])
    return summary


def optimize_point_threshold(
    signal_scores: Sequence[float],
    signal_weights: Sequence[float],
    background_scores: Sequence[float],
    background_weights: Sequence[float],
    luminosity: float | None = None,
    signal_rate_factor: float = 1.0,
    thresholds: Sequence[float] | None = None,
    min_background_raw: int = 25,
    min_background_neff: float = 10.0,
    total_signal_weight: float | None = None,
    confidence_level: float = 0.95,
    within_fraction: float = 0.01,
    background_scan: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Select the pointwise threshold that minimizes the exact-CLs limit.

    ``background_weights`` must already be expected event yields at the target
    luminosity.  With the default ``luminosity=None``, ``signal_weights`` must
    be expected signal-event yields for a production cross section of 1 fb and
    the limit is simply ``S95 / selected_signal_weight``.  Alternatively, pass
    ``luminosity`` when signal weights are used only for their conditional
    efficiency; then ``luminosity * signal_rate_factor`` is the total expected
    feature-tree population per fb.  All yield sums are signed.
    """

    s_scores = _finite_array(signal_scores, "signal_scores")
    s_weights = _finite_array(signal_weights, "signal_weights")
    b_scores = _finite_array(background_scores, "background_scores")
    b_weights = _finite_array(background_weights, "background_weights")
    if len(s_scores) != len(s_weights):
        raise ValueError("signal_scores and signal_weights must have matching lengths")
    if len(b_scores) != len(b_weights):
        raise ValueError("background_scores and background_weights must have matching lengths")
    if luminosity is not None:
        luminosity = float(luminosity)
    signal_rate_factor = float(signal_rate_factor)
    if luminosity is not None and (not math.isfinite(luminosity) or luminosity <= 0.0):
        raise ValueError("luminosity must be positive and finite")
    if not math.isfinite(signal_rate_factor) or signal_rate_factor <= 0.0:
        raise ValueError("signal_rate_factor must be positive and finite")
    if int(min_background_raw) != min_background_raw or int(min_background_raw) < 0:
        raise ValueError("min_background_raw must be a non-negative integer")
    min_background_raw = int(min_background_raw)
    min_background_neff = float(min_background_neff)
    if not math.isfinite(min_background_neff) or min_background_neff < 0.0:
        raise ValueError("min_background_neff must be finite and non-negative")
    within_fraction = float(within_fraction)
    if not math.isfinite(within_fraction) or within_fraction < 0.0:
        raise ValueError("within_fraction must be finite and non-negative")

    if thresholds is None:
        scan = DEFAULT_THRESHOLDS.copy()
    else:
        scan = _finite_array(thresholds, "thresholds")
    if len(scan) == 0:
        raise ValueError("thresholds must not be empty")
    if np.any((scan < 0.0) | (scan > 1.0)):
        raise ValueError("thresholds must lie in [0, 1]")
    scan = np.unique(scan)

    signal_total = float(np.sum(s_weights)) if total_signal_weight is None else float(total_signal_weight)
    if not math.isfinite(signal_total) or signal_total <= 0.0:
        raise ValueError("total_signal_weight must be positive and finite")

    signal_scan = threshold_weight_scan(s_scores, s_weights, scan)
    if background_scan is None:
        b_scan = background_threshold_scan(
            b_scores, b_weights, scan, confidence_level=confidence_level
        )
    else:
        b_scan = {
            key: np.asarray(value)
            for key, value in background_scan.items()
            if key in {
                "thresholds",
                "yield",
                "sumw2",
                "raw_entries",
                "effective_entries",
                "s95_events",
                "confidence_level",
            }
        }
        required = {
            "thresholds",
            "yield",
            "sumw2",
            "raw_entries",
            "effective_entries",
            "s95_events",
        }
        if not required.issubset(b_scan):
            raise ValueError(f"background_scan is missing {sorted(required - set(b_scan))}")
        if not np.array_equal(np.asarray(b_scan["thresholds"], dtype=float), scan):
            raise ValueError("background_scan thresholds do not match the requested scan")
        stored_cl = np.asarray(b_scan.get("confidence_level", [confidence_level]), dtype=float)
        if len(stored_cl) != 1 or not math.isclose(
            float(stored_cl[0]), confidence_level, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("background_scan confidence level does not match")

    candidates: list[dict[str, Any]] = []
    rejection_counts = {
        "nonpositive_signal": 0,
        "nonpositive_background": 0,
        "background_raw": 0,
        "background_neff": 0,
    }
    for index, threshold in enumerate(scan):
        selected_signal = float(signal_scan["yield"][index])
        if selected_signal <= 0.0:
            rejection_counts["nonpositive_signal"] += 1
            continue
        efficiency = selected_signal / signal_total
        if efficiency <= 0.0:
            rejection_counts["nonpositive_signal"] += 1
            continue

        background_yield = float(b_scan["yield"][index])
        background_sumw2 = float(b_scan["sumw2"][index])
        background_raw = int(b_scan["raw_entries"][index])
        background_neff = float(b_scan["effective_entries"][index])
        if background_yield < 0.0:
            rejection_counts["nonpositive_background"] += 1
            continue
        if background_raw < min_background_raw:
            rejection_counts["background_raw"] += 1
            continue
        if background_neff < min_background_neff:
            rejection_counts["background_neff"] += 1
            continue
        s95 = float(b_scan["s95_events"][index])
        signal_yield_per_fb = (
            selected_signal
            if luminosity is None
            else luminosity * signal_rate_factor * efficiency
        )
        sigma95 = s95 / signal_yield_per_fb
        candidates.append(
            {
                "threshold": float(threshold),
                "sigma95_fb": float(sigma95),
                "s95_events": float(s95),
                "signal_efficiency": float(efficiency),
                "selected_signal_weight": selected_signal,
                "total_signal_weight": signal_total,
                "selected_signal_yield_per_fb": float(signal_yield_per_fb),
                "signal_weight_convention": (
                    "unit_cross_section_expected_yield"
                    if luminosity is None
                    else "efficiency_weights"
                ),
                "background_yield": background_yield,
                "background_sumw2": background_sumw2,
                "background_uncertainty": math.sqrt(background_sumw2),
                "background_raw_entries": background_raw,
                "background_effective_entries": background_neff,
            }
        )

    if not candidates:
        raise NoValidThresholdError(
            "no threshold satisfies positive signed yields and the requested "
            f"background MC-statistics constraints: {rejection_counts}"
        )
    minimum = min(item["sigma95_fb"] for item in candidates)
    near_best = [
        item for item in candidates if item["sigma95_fb"] <= minimum * (1.0 + within_fraction)
    ]
    # Primary tie-breaker: the largest effective background count.  The lower
    # threshold is deterministic and usually retains more signal if Neff ties.
    selected = min(
        near_best,
        key=lambda item: (-item["background_effective_entries"], item["threshold"]),
    )
    result = dict(selected)
    result.update(
        {
            "status": "ok",
            "minimum_sigma95_fb": float(minimum),
            "within_fraction": within_fraction,
            "n_thresholds_scanned": int(len(scan)),
            "n_valid_thresholds": int(len(candidates)),
            "rejection_counts": rejection_counts,
            "candidates": candidates,
        }
    )
    return result


def limit_objective(sigmas: Sequence[float]) -> float:
    """Return ``0.75 median(log sigma95) + 0.25 Q90(log sigma95)``."""

    values = _finite_array(sigmas, "sigmas")
    if len(values) == 0 or np.any(values <= 0.0):
        raise ValueError("sigmas must be a non-empty array of positive finite limits")
    logs = np.log(values)
    return float(0.75 * np.median(logs) + 0.25 * np.quantile(logs, 0.90))


def parameterized_gate(
    pooled_limits: Sequence[float],
    sm_limits: Sequence[float],
    sm_point_index: int,
    rotation_pooled_limits: Sequence[Sequence[float]],
    rotation_sm_limits: Sequence[Sequence[float]],
    median_max: float = 0.90,
    q90_max: float = 1.10,
    sm_max: float = 1.05,
    required_rotations: int = 4,
) -> dict[str, Any]:
    """Evaluate the pre-declared gate for a parameterized classifier."""

    pooled = _finite_array(pooled_limits, "pooled_limits")
    baseline = _finite_array(sm_limits, "sm_limits")
    if pooled.shape != baseline.shape or len(pooled) == 0:
        raise ValueError("pooled_limits and sm_limits must be non-empty and have matching shapes")
    if np.any(pooled <= 0.0) or np.any(baseline <= 0.0):
        raise ValueError("all pointwise limits must be positive")
    sm_point_index = int(sm_point_index)
    if sm_point_index < 0 or sm_point_index >= len(pooled):
        raise ValueError("sm_point_index is outside the limit arrays")

    rotation_pooled = np.asarray(rotation_pooled_limits, dtype=float)
    rotation_sm = np.asarray(rotation_sm_limits, dtype=float)
    if rotation_pooled.ndim != 2 or rotation_pooled.shape != rotation_sm.shape:
        raise ValueError("rotation limit arrays must be matching two-dimensional arrays")
    if rotation_pooled.shape[1] != len(pooled):
        raise ValueError("rotation limit arrays must have one column per c3/d4 point")
    if not np.all(np.isfinite(rotation_pooled)) or not np.all(np.isfinite(rotation_sm)):
        raise ValueError("rotation limits must be finite")
    if np.any(rotation_pooled <= 0.0) or np.any(rotation_sm <= 0.0):
        raise ValueError("rotation limits must be positive")

    ratios = pooled / baseline
    rotation_ratios = rotation_pooled / rotation_sm
    rotation_medians = np.median(rotation_ratios, axis=1)
    rotation_objective_ratios = np.asarray(
        [
            math.exp(limit_objective(pooled_row) - limit_objective(sm_row))
            for pooled_row, sm_row in zip(rotation_pooled, rotation_sm)
        ],
        dtype=float,
    )
    rotations_favoring_pooled = int(np.sum(rotation_objective_ratios < 1.0))
    metrics = {
        "median_ratio": float(np.median(ratios)),
        "q90_ratio": float(np.quantile(ratios, 0.90)),
        "sm_point_ratio": float(ratios[sm_point_index]),
        "rotations_favoring_pooled": rotations_favoring_pooled,
        "n_rotations": int(rotation_ratios.shape[0]),
        "rotation_median_ratios": rotation_medians.tolist(),
        "rotation_objective_ratios": rotation_objective_ratios.tolist(),
    }
    criteria = {
        "median": metrics["median_ratio"] <= float(median_max),
        "q90": metrics["q90_ratio"] <= float(q90_max),
        "sm_point": metrics["sm_point_ratio"] <= float(sm_max),
        "rotations": rotations_favoring_pooled >= int(required_rotations),
    }
    return {
        "passed": bool(all(criteria.values())),
        "criteria": criteria,
        "metrics": metrics,
        "thresholds": {
            "median_max": float(median_max),
            "q90_max": float(q90_max),
            "sm_max": float(sm_max),
            "required_rotations": int(required_rotations),
        },
        "pointwise_ratios": ratios.tolist(),
    }


def weighted_quantile(
    values: Sequence[float], quantiles: Sequence[float], weights: Sequence[float] | None = None
) -> np.ndarray:
    """Return deterministic quantiles, optionally weighted by non-negative weights."""

    data = _finite_array(values, "values")
    requested = _finite_array(quantiles, "quantiles")
    if len(data) == 0:
        raise ValueError("values must not be empty")
    if np.any((requested < 0.0) | (requested > 1.0)):
        raise ValueError("quantiles must lie in [0, 1]")
    if weights is None:
        return np.asarray(np.quantile(data, requested), dtype=float)

    weight_array = _finite_array(weights, "weights")
    if len(weight_array) != len(data):
        raise ValueError("values and weights must have matching lengths")
    if np.any(weight_array < 0.0):
        raise ValueError("quantile weights must be non-negative")
    if float(np.sum(weight_array)) <= 0.0:
        raise ValueError("quantile weights must have a positive sum")

    order = np.argsort(data, kind="mergesort")
    sorted_data = data[order]
    sorted_weights = weight_array[order]
    # Midpoint positions make the interpolation invariant under splitting one
    # weighted event into identical subevents.
    positions = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    positions /= float(np.sum(sorted_weights))
    return np.interp(requested, positions, sorted_data, left=sorted_data[0], right=sorted_data[-1])


def score_bin_indices(scores: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    """Map scores to bins; internal-edge ties enter the bin on their right."""

    score_array = _finite_array(scores, "scores")
    edge_array = _finite_array(edges, "edges")
    if len(edge_array) < 2 or np.any(np.diff(edge_array) <= 0.0):
        raise ValueError("edges must be strictly increasing and contain at least two values")
    return np.searchsorted(edge_array[1:-1], score_array, side="right").astype(np.int16)


def binned_weight_summary(
    scores: Sequence[float], weights: Sequence[float], edges: Sequence[float]
) -> dict[str, Any]:
    """Return signed yields, sumw2, raw counts, and Neff in score bins."""

    score_array = _finite_array(scores, "scores")
    weight_array = _finite_array(weights, "weights")
    if len(score_array) != len(weight_array):
        raise ValueError("scores and weights must have matching lengths")
    edge_array = _finite_array(edges, "edges")
    assignments = score_bin_indices(score_array, edge_array)
    n_bins = len(edge_array) - 1
    yields = np.bincount(assignments, weights=weight_array, minlength=n_bins).astype(float)
    sumw2 = np.bincount(assignments, weights=np.square(weight_array), minlength=n_bins).astype(float)
    raw = np.bincount(assignments, minlength=n_bins).astype(int)
    neff = np.divide(
        np.square(yields), sumw2, out=np.zeros_like(yields), where=sumw2 > 0.0
    )
    return {
        "edges": edge_array.tolist(),
        "yield": yields,
        "sumw2": sumw2,
        "uncertainty": np.sqrt(sumw2),
        "raw_entries": raw,
        "effective_entries": neff,
        "bin_indices": assignments,
    }


def enumerate_score_binnings(
    background_scores: Sequence[float],
    background_weights: Sequence[float],
    quantiles: Sequence[float] = DEFAULT_BIN_QUANTILES,
    min_bins: int = 2,
    max_bins: int = 5,
) -> list[dict[str, Any]]:
    """Enumerate unique 2--5-bin subsets of background-score quantile edges.

    Absolute physical weights define the quantiles because a signed cumulative
    distribution is not monotonic.  Signed weights are nevertheless retained
    for every yield and validity decision.
    """

    scores = _finite_array(background_scores, "background_scores")
    weights = _finite_array(background_weights, "background_weights")
    if len(scores) != len(weights):
        raise ValueError("background_scores and background_weights must have matching lengths")
    if len(scores) == 0:
        raise ValueError("background score sample must not be empty")
    quantile_values = _finite_array(quantiles, "quantiles")
    if len(quantile_values) < 3 or quantile_values[0] != 0.0 or quantile_values[-1] != 1.0:
        raise ValueError("quantiles must start at 0, end at 1, and contain an interior value")
    if np.any(np.diff(quantile_values) <= 0.0):
        raise ValueError("quantiles must be strictly increasing")
    min_bins = int(min_bins)
    max_bins = min(int(max_bins), len(quantile_values) - 1)
    if min_bins < 1 or max_bins < min_bins:
        raise ValueError("invalid min_bins/max_bins range")

    base_edges = weighted_quantile(scores, quantile_values, np.abs(weights))
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    interior_indices = range(1, len(base_edges) - 1)
    for n_bins in range(min_bins, max_bins + 1):
        for chosen in itertools.combinations(interior_indices, n_bins - 1):
            indices = (0, *chosen, len(base_edges) - 1)
            edges = np.asarray([base_edges[index] for index in indices], dtype=float)
            if np.any(np.diff(edges) <= 0.0):
                continue
            key = tuple(float(value) for value in edges)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "n_bins": n_bins,
                    "edges": list(key),
                    "base_edge_indices": list(indices),
                }
            )
    return candidates


def _valid_binned_background(
    summary: Mapping[str, Any], min_raw: int, min_neff: float
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    yields = np.asarray(summary["yield"], dtype=float)
    raw = np.asarray(summary["raw_entries"], dtype=int)
    neff = np.asarray(summary["effective_entries"], dtype=float)
    if np.any(yields <= 0.0):
        reasons.append("nonpositive_signed_background")
    if np.any(raw < int(min_raw)):
        reasons.append("raw_background_entries")
    if np.any(neff < float(min_neff)):
        reasons.append("effective_background_entries")
    return not reasons, reasons


def build_pyhf_channel(
    name: str,
    signal_scores: Sequence[float],
    signal_unit_weights: Sequence[float],
    background_scores: Sequence[float],
    background_weights: Sequence[float],
    edges: Sequence[float],
) -> dict[str, Any]:
    """Build one held-out-fold channel from signed score arrays."""

    signal = binned_weight_summary(signal_scores, signal_unit_weights, edges)
    background = binned_weight_summary(background_scores, background_weights, edges)
    return {
        "name": str(name),
        "edges": list(map(float, edges)),
        "signal": np.asarray(signal["yield"], dtype=float),
        "background": np.asarray(background["yield"], dtype=float),
        "signal_staterror": np.asarray(signal["uncertainty"], dtype=float),
        "background_staterror": np.asarray(background["uncertainty"], dtype=float),
        "signal_raw_entries": np.asarray(signal["raw_entries"], dtype=int),
        "background_raw_entries": np.asarray(background["raw_entries"], dtype=int),
        "signal_effective_entries": np.asarray(signal["effective_entries"], dtype=float),
        "background_effective_entries": np.asarray(
            background["effective_entries"], dtype=float
        ),
    }


def validation_binning(
    background_scores: Sequence[float],
    background_weights: Sequence[float],
    signal_scores: Sequence[float] | None = None,
    signal_unit_weights: Sequence[float] | None = None,
    limit_evaluator: Callable[[Sequence[float]], float] | None = None,
    quantiles: Sequence[float] = DEFAULT_BIN_QUANTILES,
    min_bins: int = 2,
    max_bins: int = 5,
    min_background_raw: int = 25,
    min_background_neff: float = 10.0,
    within_fraction: float = 0.01,
    include_staterror: bool = True,
) -> dict[str, Any]:
    """Select validation score bins and construct a nested fallback hierarchy.

    A caller may provide ``limit_evaluator(edges)`` (useful when combining
    several folds or caching fits).  Otherwise signal arrays are required and a
    one-channel pyhf expected limit is evaluated for every valid candidate.
    """

    if int(min_background_raw) != min_background_raw or int(min_background_raw) < 0:
        raise ValueError("min_background_raw must be a non-negative integer")
    min_background_raw = int(min_background_raw)
    min_background_neff = float(min_background_neff)
    within_fraction = float(within_fraction)
    if min_background_neff < 0.0 or within_fraction < 0.0:
        raise ValueError("min_background_neff and within_fraction must be non-negative")
    if limit_evaluator is None and (signal_scores is None or signal_unit_weights is None):
        raise ValueError("provide either limit_evaluator or signal score/weight arrays")

    candidates = enumerate_score_binnings(
        background_scores,
        background_weights,
        quantiles=quantiles,
        min_bins=min_bins,
        max_bins=max_bins,
    )
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        record = dict(candidate)
        summary = binned_weight_summary(
            background_scores, background_weights, candidate["edges"]
        )
        valid, reasons = _valid_binned_background(
            summary, min_background_raw, min_background_neff
        )
        record.update(
            {
                "background_yield": np.asarray(summary["yield"]).tolist(),
                "background_sumw2": np.asarray(summary["sumw2"]).tolist(),
                "background_raw_entries": np.asarray(summary["raw_entries"]).tolist(),
                "background_effective_entries": np.asarray(
                    summary["effective_entries"]
                ).tolist(),
                "valid": bool(valid),
                "invalid_reasons": reasons,
                "expected_limit_fb": None,
            }
        )
        if valid:
            if limit_evaluator is not None:
                try:
                    limit_value = float(limit_evaluator(candidate["edges"]))
                except Exception as error:  # fit failure is data, not a clipped yield
                    record["valid"] = False
                    record["invalid_reasons"] = [
                        f"limit_evaluator_failed:{type(error).__name__}:{error}"
                    ]
                else:
                    if math.isfinite(limit_value) and limit_value > 0.0:
                        record["expected_limit_fb"] = limit_value
                    else:
                        record["valid"] = False
                        record["invalid_reasons"] = ["nonfinite_or_nonpositive_limit"]
            else:
                channel = build_pyhf_channel(
                    "validation",
                    signal_scores,
                    signal_unit_weights,
                    background_scores,
                    background_weights,
                    candidate["edges"],
                )
                fit = pyhf_combined_limit([channel], include_staterror=include_staterror)
                if fit["status"] == "ok":
                    record["expected_limit_fb"] = fit["expected_median"]
                else:
                    record["valid"] = False
                    record["invalid_reasons"] = [f"pyhf_failed:{fit['error']}"]
        evaluated.append(record)

    valid_records = [
        item
        for item in evaluated
        if item["valid"] and item["expected_limit_fb"] is not None
    ]
    if not valid_records:
        counts: dict[str, int] = {}
        for item in evaluated:
            for reason in item["invalid_reasons"]:
                counts[reason] = counts.get(reason, 0) + 1
        raise NoValidBinningError(f"no validation score binning is valid: {counts}")

    minimum = min(item["expected_limit_fb"] for item in valid_records)
    near_best = [
        item
        for item in valid_records
        if item["expected_limit_fb"] <= minimum * (1.0 + within_fraction)
    ]
    # The declared tie-breaker is fewer bins.  Remaining ties are resolved by
    # limit and lexicographic edge order, independent of enumeration details.
    selected = min(
        near_best,
        key=lambda item: (
            item["n_bins"],
            item["expected_limit_fb"],
            tuple(item["edges"]),
        ),
    )

    valid_by_edges = {tuple(item["edges"]): item for item in valid_records}
    selected_edges = np.asarray(selected["edges"], dtype=float)
    fallback: list[dict[str, Any]] = [dict(selected)]
    interior = range(1, len(selected_edges) - 1)
    for n_bins in range(selected["n_bins"] - 1, 0, -1):
        possibilities: list[dict[str, Any]] = []
        for chosen in itertools.combinations(interior, n_bins - 1):
            edges = [
                float(selected_edges[index])
                for index in (0, *chosen, len(selected_edges) - 1)
            ]
            summary = binned_weight_summary(background_scores, background_weights, edges)
            valid, reasons = _valid_binned_background(
                summary, min_background_raw, min_background_neff
            )
            if not valid:
                continue
            stored = valid_by_edges.get(tuple(edges))
            possibilities.append(
                {
                    "n_bins": n_bins,
                    "edges": edges,
                    "expected_limit_fb": (
                        stored["expected_limit_fb"] if stored is not None else None
                    ),
                    "background_yield": np.asarray(summary["yield"]).tolist(),
                    "background_sumw2": np.asarray(summary["sumw2"]).tolist(),
                    "background_raw_entries": np.asarray(summary["raw_entries"]).tolist(),
                    "background_effective_entries": np.asarray(
                        summary["effective_entries"]
                    ).tolist(),
                    "valid": True,
                    "invalid_reasons": reasons,
                }
            )
        if possibilities:
            fallback.append(
                min(
                    possibilities,
                    key=lambda item: (
                        math.inf
                        if item["expected_limit_fb"] is None
                        else item["expected_limit_fb"],
                        tuple(item["edges"]),
                    ),
                )
            )

    return {
        "status": "ok",
        "selected": dict(selected),
        "minimum_expected_limit_fb": float(minimum),
        "within_fraction": within_fraction,
        "fallback_hierarchy": fallback,
        "candidates": evaluated,
        "quantiles": list(map(float, quantiles)),
        "constraints": {
            "min_background_raw": min_background_raw,
            "min_background_neff": min_background_neff,
        },
    }


def select_test_binning(
    validation_result: Mapping[str, Any],
    background_scores: Sequence[float],
    background_weights: Sequence[float],
) -> dict[str, Any]:
    """Apply frozen validation edges, coarsening only for nonpositive test bins."""

    hierarchy = validation_result.get("fallback_hierarchy", [])
    if not hierarchy:
        raise ValueError("validation_result contains no fallback_hierarchy")
    attempted: list[dict[str, Any]] = []
    for level, candidate in enumerate(hierarchy):
        summary = binned_weight_summary(
            background_scores, background_weights, candidate["edges"]
        )
        yields = np.asarray(summary["yield"], dtype=float)
        attempt = {
            "level": level,
            "n_bins": len(candidate["edges"]) - 1,
            "edges": list(map(float, candidate["edges"])),
            "background_yield": yields.tolist(),
            "background_sumw2": np.asarray(summary["sumw2"]).tolist(),
            "background_raw_entries": np.asarray(summary["raw_entries"]).tolist(),
            "background_effective_entries": np.asarray(
                summary["effective_entries"]
            ).tolist(),
            "positive": bool(np.all(yields > 0.0)),
        }
        attempted.append(attempt)
        if attempt["positive"]:
            return {
                "status": "ok",
                "fallback_level": level,
                "used_fallback": bool(level > 0),
                **{key: value for key, value in attempt.items() if key != "level"},
                "attempted": attempted,
            }
    return {
        "status": "failed",
        "error": "all validation-defined binnings contain a nonpositive signed test-background bin",
        "attempted": attempted,
    }


def _channel_arrays(channel: Mapping[str, Any]) -> dict[str, Any]:
    name = str(channel.get("name", "channel"))
    signal = _finite_array(channel.get("signal", []), f"{name}.signal")
    background = _finite_array(channel.get("background", []), f"{name}.background")
    if len(signal) == 0 or len(signal) != len(background):
        raise ValueError(f"{name} signal/background templates must be non-empty and match")
    if np.any(signal < 0.0):
        raise ValueError(f"{name} contains a negative signal-template bin")
    if np.any(background <= 0.0):
        raise ValueError(f"{name} contains a nonpositive background-template bin")
    result = {"name": name, "signal": signal, "background": background}
    for key, nominal in (
        ("signal_staterror", signal),
        ("background_staterror", background),
    ):
        if key in channel and channel[key] is not None:
            error = _finite_array(channel[key], f"{name}.{key}")
            if len(error) != len(nominal) or np.any(error < 0.0):
                raise ValueError(f"{name}.{key} must be non-negative and match its template")
        else:
            error = np.zeros_like(nominal)
        result[key] = error
    if float(np.sum(signal)) <= 0.0:
        raise ValueError(f"{name} has zero total signal sensitivity")
    return result


def pyhf_workspace_spec(
    channels: Sequence[Mapping[str, Any]],
    include_staterror: bool = True,
    poi_bounds: tuple[float, float] = (0.0, 1.0e7),
) -> dict[str, Any]:
    """Build a pyhf workspace whose shared POI is the signal cross section in fb."""

    if not channels:
        raise ValueError("at least one pyhf channel is required")
    lower, upper = map(float, poi_bounds)
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 <= lower < upper):
        raise ValueError("poi_bounds must be finite, increasing, and non-negative")

    normalized = [_channel_arrays(channel) for channel in channels]
    names = [item["name"] for item in normalized]
    if len(names) != len(set(names)):
        raise ValueError("pyhf channel names must be unique")

    channel_specs: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for index, channel in enumerate(normalized):
        safe_name = "".join(character if character.isalnum() else "_" for character in channel["name"])
        signal_modifiers: list[dict[str, Any]] = [
            {"name": "sigma_fb", "type": "normfactor", "data": None}
        ]
        background_modifiers: list[dict[str, Any]] = []
        if include_staterror:
            if np.any(channel["signal_staterror"] > 0.0):
                signal_modifiers.append(
                    {
                        "name": f"signal_mcstat_{index}_{safe_name}",
                        "type": "staterror",
                        "data": channel["signal_staterror"].tolist(),
                    }
                )
            if np.any(channel["background_staterror"] > 0.0):
                background_modifiers.append(
                    {
                        "name": f"background_mcstat_{index}_{safe_name}",
                        "type": "staterror",
                        "data": channel["background_staterror"].tolist(),
                    }
                )
        channel_specs.append(
            {
                "name": channel["name"],
                "samples": [
                    {
                        "name": "signal",
                        "data": channel["signal"].tolist(),
                        "modifiers": signal_modifiers,
                    },
                    {
                        "name": "background",
                        "data": channel["background"].tolist(),
                        "modifiers": background_modifiers,
                    },
                ],
            }
        )
        # Expected limits use the background-only Asimov observation.
        observations.append(
            {"name": channel["name"], "data": channel["background"].tolist()}
        )

    return {
        "channels": channel_specs,
        "observations": observations,
        "measurements": [
            {
                "name": "expected_limit",
                "config": {
                    "poi": "sigma_fb",
                    "parameters": [
                        {
                            "name": "sigma_fb",
                            "bounds": [[lower, upper]],
                            "inits": [min(max(1.0, lower), upper)],
                        }
                    ],
                },
            }
        ],
        "version": "1.0.0",
    }


def pyhf_combined_limit(
    channels: Sequence[Mapping[str, Any]],
    include_staterror: bool = True,
    confidence_level: float = 0.95,
    poi_bounds: tuple[float, float] = (0.0, 1.0e7),
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Compute a five-channel (or arbitrary-channel) asymptotic pyhf limit.

    Failures are returned as structured results by default.  No negative yield
    is clipped; invalid templates fail before pyhf is called.
    """

    confidence_level = float(confidence_level)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    try:
        import pyhf  # type: ignore

        pyhf.set_backend("numpy")
        spec = pyhf_workspace_spec(
            channels, include_staterror=include_staterror, poi_bounds=poi_bounds
        )
        workspace = pyhf.Workspace(spec)
        model = workspace.model(measurement_name="expected_limit")
        data = workspace.data(model)
        observed, expected = pyhf.infer.intervals.upper_limits.upper_limit(
            data, model, level=1.0 - confidence_level
        )
        observed_value = float(np.asarray(observed))
        expected_values = np.asarray(expected, dtype=float).reshape(-1)
        if len(expected_values) != 5:
            raise RuntimeError(
                f"pyhf returned {len(expected_values)} expected bands instead of five"
            )
        if not math.isfinite(observed_value) or not np.all(np.isfinite(expected_values)):
            raise RuntimeError("pyhf returned a non-finite upper limit")
        return {
            "status": "ok",
            "backend": "pyhf-asymptotic-numpy",
            "pyhf_version": getattr(pyhf, "__version__", None),
            "confidence_level": confidence_level,
            "include_staterror": bool(include_staterror),
            "observed": observed_value,
            "expected": expected_values.tolist(),
            "expected_minus2sigma": float(expected_values[0]),
            "expected_minus1sigma": float(expected_values[1]),
            "expected_median": float(expected_values[2]),
            "expected_plus1sigma": float(expected_values[3]),
            "expected_plus2sigma": float(expected_values[4]),
            "n_channels": len(channels),
            "workspace_spec": spec,
            "error": None,
        }
    except Exception as error:
        if raise_on_failure:
            raise
        return {
            "status": "failed",
            "backend": "pyhf-asymptotic-numpy",
            "confidence_level": confidence_level,
            "include_staterror": bool(include_staterror),
            "observed": None,
            "expected": None,
            "expected_median": None,
            "n_channels": len(channels),
            "error_type": type(error).__name__,
            "error": str(error),
        }


def pyhf_one_bin_limit(
    signal_yields: Sequence[float],
    background_yields: Sequence[float],
    signal_sumw2: Sequence[float] | None = None,
    background_sumw2: Sequence[float] | None = None,
    include_staterror: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience control calculation with one bin per held-out fold."""

    signal = _finite_array(signal_yields, "signal_yields")
    background = _finite_array(background_yields, "background_yields")
    if len(signal) != len(background) or len(signal) == 0:
        raise ValueError("signal_yields and background_yields must match and be non-empty")
    signal_variance = (
        np.zeros_like(signal)
        if signal_sumw2 is None
        else _finite_array(signal_sumw2, "signal_sumw2")
    )
    background_variance = (
        np.zeros_like(background)
        if background_sumw2 is None
        else _finite_array(background_sumw2, "background_sumw2")
    )
    if len(signal_variance) != len(signal) or len(background_variance) != len(background):
        raise ValueError("sumw2 arrays must match their yield arrays")
    if np.any(signal_variance < 0.0) or np.any(background_variance < 0.0):
        raise ValueError("sumw2 arrays must be non-negative")
    channels = [
        {
            "name": f"fold{index}",
            "signal": [signal[index]],
            "background": [background[index]],
            "signal_staterror": [math.sqrt(signal_variance[index])],
            "background_staterror": [math.sqrt(background_variance[index])],
        }
        for index in range(len(signal))
    ]
    return pyhf_combined_limit(
        channels, include_staterror=include_staterror, **kwargs
    )


def xgboost_search_params(trial: Any) -> dict[str, Any]:
    """Materialize the pre-declared v2 XGBoost search space for an Optuna trial."""

    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_categorical("gamma", [0.0, 0.01, 0.1, 1.0]),
        "reg_alpha": trial.suggest_categorical(
            "reg_alpha", [0.0, 0.001, 0.01, 0.1, 1.0]
        ),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
    }


def run_optuna_tuning(
    objective: Callable[[Any], float],
    storage_path: str | Path,
    study_name: str,
    n_trials: int = 40,
    seed: int = DEFAULT_SEED,
    enqueue_params: Mapping[str, Any] | None = CURRENT_XGBOOST_PARAMS,
    direction: str = "minimize",
    timeout: float | None = None,
    catch: tuple[type[Exception], ...] = (),
) -> Any:
    """Run or resume a seeded, sequential Optuna study up to ``n_trials`` total.

    The SQLite database is persistent.  A resumed study runs only the remaining
    trials, and the current XGBoost point is enqueued only for a new study.
    """

    try:
        import optuna  # type: ignore
    except ImportError as error:
        raise ImportError(
            "Optuna is required for tuning; use the reproducible ~/xgb-py310 environment"
        ) from error
    n_trials = int(n_trials)
    if n_trials < 0:
        raise ValueError("n_trials must be non-negative")
    if direction not in {"minimize", "maximize"}:
        raise ValueError("direction must be 'minimize' or 'maximize'")
    database = Path(storage_path).expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(
        study_name=str(study_name),
        storage=f"sqlite:///{database}",
        sampler=sampler,
        direction=direction,
        load_if_exists=True,
    )
    existing_trials = len(study.trials)
    if existing_trials == 0 and enqueue_params is not None:
        study.enqueue_trial(dict(enqueue_params))
    # ``study.optimize(n_trials=N)`` counts a WAITING enqueued trial among the
    # N executions.  Compute the remaining budget before enqueueing so a new
    # 40-trial study finishes with 40 trials rather than 39.
    remaining = max(0, n_trials - existing_trials)
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=timeout,
            n_jobs=1,
            gc_after_trial=True,
            catch=catch,
        )
    return study


def summarize_optuna_study(study: Any) -> dict[str, Any]:
    """Return a JSON-safe Optuna history and best-trial summary."""

    trials: list[dict[str, Any]] = []
    for trial in study.trials:
        trials.append(
            {
                "number": int(trial.number),
                "state": getattr(trial.state, "name", str(trial.state)),
                "value": None if trial.value is None else float(trial.value),
                "params": dict(trial.params),
                "user_attrs": dict(trial.user_attrs),
                "system_attrs": dict(trial.system_attrs),
            }
        )
    complete = [item for item in trials if item["state"] == "COMPLETE"]
    result = {
        "study_name": str(study.study_name),
        "direction": getattr(study.direction, "name", str(study.direction)),
        "n_trials": len(trials),
        "n_complete": len(complete),
        "trials": trials,
        "best_trial": None,
    }
    if complete:
        best = study.best_trial
        result["best_trial"] = {
            "number": int(best.number),
            "value": float(best.value),
            "params": dict(best.params),
        }
    return result


def parameterize_features(
    features: Sequence[Sequence[float]],
    c3: float | Sequence[float],
    d4: float | Sequence[float],
    c3_scale: float = 30.0,
    d4_scale: float = 700.0,
) -> np.ndarray:
    """Append normalized c3 and d4 coordinates to a feature matrix."""

    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("features must be a finite two-dimensional array")
    c3_scale = float(c3_scale)
    d4_scale = float(d4_scale)
    if c3_scale <= 0.0 or d4_scale <= 0.0:
        raise ValueError("parameter scales must be positive")

    def expand(value: float | Sequence[float], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(len(matrix), float(array))
        if array.ndim != 1 or len(array) != len(matrix) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite and scalar or one value per feature row")
        return array

    c3_values = expand(c3, "c3")
    d4_values = expand(d4, "d4")
    return np.column_stack([matrix, c3_values / c3_scale, d4_values / d4_scale])


def assign_background_parameter_points(
    source_ids: Sequence[Any],
    event_indices: Sequence[int],
    grid_points: Sequence[Sequence[float]],
    replicas_per_event: int = 3,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Assign deterministic, distinct, uniformly sampled grid points per event."""

    sources = _array_1d(source_ids, "source_ids", None)
    entries = _array_1d(event_indices, "event_indices", None)
    if len(sources) != len(entries):
        raise ValueError("source_ids and event_indices must have matching lengths")
    grid = np.asarray(grid_points, dtype=float)
    if grid.ndim != 2 or grid.shape[1] != 2 or not np.all(np.isfinite(grid)):
        raise ValueError("grid_points must be a finite N-by-2 array")
    if len(np.unique(grid, axis=0)) != len(grid):
        raise ValueError("grid_points must be distinct")
    replicas_per_event = int(replicas_per_event)
    if replicas_per_event < 1 or replicas_per_event > len(grid):
        raise ValueError("replicas_per_event must lie between one and the grid size")

    assignments = np.empty((len(sources), replicas_per_event, 2), dtype=float)
    for row, (source, entry) in enumerate(zip(sources, entries)):
        event_seed = stable_seed(f"background-parameters\0{source}\0{entry}", seed)
        rng = np.random.default_rng(event_seed)
        indices = rng.choice(len(grid), size=replicas_per_event, replace=False)
        assignments[row] = grid[indices]
    return assignments


def make_background_parameter_replicas(
    features: Sequence[Sequence[float]],
    training_weights: Sequence[float],
    folds: Sequence[int],
    source_ids: Sequence[Any],
    event_indices: Sequence[int],
    grid_points: Sequence[Sequence[float]],
    replicas_per_event: int = 3,
    seed: int = DEFAULT_SEED,
    c3_scale: float = 30.0,
    d4_scale: float = 700.0,
) -> dict[str, np.ndarray]:
    """Replicate background rows at parameter points without changing total weight."""

    matrix = np.asarray(features, dtype=float)
    weights = _finite_array(training_weights, "training_weights")
    fold_array = _array_1d(folds, "folds", int)
    sources = _array_1d(source_ids, "source_ids", None)
    entries = _array_1d(event_indices, "event_indices", None)
    n_rows = len(matrix)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("features must be a finite two-dimensional array")
    if any(len(item) != n_rows for item in (weights, fold_array, sources, entries)):
        raise ValueError("all background arrays must have matching row counts")
    if np.any(weights < 0.0):
        raise ValueError("training_weights must be non-negative")

    points = assign_background_parameter_points(
        sources,
        entries,
        grid_points,
        replicas_per_event=replicas_per_event,
        seed=seed,
    )
    repeated_features = np.repeat(matrix, replicas_per_event, axis=0)
    flattened_points = points.reshape(-1, 2)
    parameterized = parameterize_features(
        repeated_features,
        flattened_points[:, 0],
        flattened_points[:, 1],
        c3_scale=c3_scale,
        d4_scale=d4_scale,
    )
    return {
        "features": parameterized,
        "training_weights": np.repeat(weights / replicas_per_event, replicas_per_event),
        "folds": np.repeat(fold_array, replicas_per_event),
        "source_ids": np.repeat(sources, replicas_per_event),
        "event_indices": np.repeat(entries, replicas_per_event),
        "grid_points": flattened_points,
        "replica_index": np.tile(np.arange(replicas_per_event), n_rows),
    }


def json_safe(value: Any) -> Any:
    """Convert common scientific-Python objects to strict JSON values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    """Atomically write strict, sorted, human-readable JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def write_csv_records(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write flat result records, JSON-encoding nested cells deterministically."""

    rows = [dict(row) for row in records]
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                key = str(key)
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    fields = [str(item) for item in fieldnames]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                serialized: dict[str, Any] = {}
                for field in fields:
                    value = json_safe(row.get(field))
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
                    serialized[field] = value
                writer.writerow(serialized)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def package_versions(
    packages: Sequence[str] = ("numpy", "xgboost", "scikit-learn", "pyhf", "optuna")
) -> dict[str, str | None]:
    """Return installed package versions without importing heavy dependencies."""

    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[str(package)] = importlib.metadata.version(str(package))
        except importlib.metadata.PackageNotFoundError:
            versions[str(package)] = None
    return versions


def build_method_manifest(
    observable_set: str,
    feature_profile: str,
    feature_names: Sequence[str],
    training_strategy: str,
    source_commit: str | None,
    normalization_inputs: Mapping[str, Any],
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
    model_parameters: Mapping[str, Any] | None = None,
    input_files: Sequence[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the reproducibility manifest shared by v2 models and results."""

    names = [str(name) for name in feature_names]
    if not names or len(names) != len(set(names)):
        raise ValueError("feature_names must be non-empty and unique")
    if int(n_folds) < 3:
        raise ValueError("n_folds must be at least three")
    manifest = {
        "method_version": METHOD_VERSION,
        "observable_set": str(observable_set),
        "feature_profile": str(feature_profile),
        "feature_names": names,
        "feature_count": len(names),
        "training_strategy": str(training_strategy),
        "n_folds": int(n_folds),
        "seed": int(seed),
        "source_commit": source_commit,
        "package_versions": package_versions(),
        "normalization_inputs": json_safe(normalization_inputs),
        "model_parameters": json_safe(model_parameters or {}),
        "input_files": json_safe(input_files or []),
    }
    if extra:
        overlap = set(manifest) & set(extra)
        if overlap:
            raise ValueError(f"extra manifest fields would overwrite {sorted(overlap)}")
        manifest.update(json_safe(extra))
    return manifest


def write_study_results(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    point_rows: Iterable[Mapping[str, Any]],
    fold_rows: Iterable[Mapping[str, Any]] = (),
    extra_json: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Write the standard manifest and per-point/fold JSON+CSV result bundle."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    point_records = [dict(row) for row in point_rows]
    fold_records = [dict(row) for row in fold_rows]
    paths = {
        "manifest": str(write_json(directory / "method_manifest.json", manifest)),
        "points_json": str(write_json(directory / "point_results.json", point_records)),
        "points_csv": str(write_csv_records(directory / "point_results.csv", point_records)),
        "folds_json": str(write_json(directory / "fold_results.json", fold_records)),
        "folds_csv": str(write_csv_records(directory / "fold_results.csv", fold_records)),
    }
    if extra_json:
        for name, payload in extra_json.items():
            safe_name = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in str(name)
            )
            paths[f"extra:{name}"] = str(
                write_json(directory / f"{safe_name}.json", payload)
            )
    return paths


__all__ = [
    "METHOD_VERSION",
    "DEFAULT_SEED",
    "DEFAULT_N_FOLDS",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_BIN_QUANTILES",
    "CURRENT_XGBOOST_PARAMS",
    "NoValidThresholdError",
    "NoValidBinningError",
    "stable_seed",
    "deterministic_folds",
    "rotation_masks",
    "crossfit_rotations",
    "weighted_yield",
    "pooled_equal_point_weights",
    "balanced_binary_training_weights",
    "pooled_classifier_training_weights",
    "poisson_median_observed",
    "exact_cls_signal_upper_limit",
    "threshold_weight_scan",
    "background_threshold_scan",
    "optimize_point_threshold",
    "limit_objective",
    "weighted_quantile",
    "score_bin_indices",
    "binned_weight_summary",
    "enumerate_score_binnings",
    "build_pyhf_channel",
    "validation_binning",
    "select_test_binning",
    "pyhf_workspace_spec",
    "pyhf_combined_limit",
    "pyhf_one_bin_limit",
    "xgboost_search_params",
    "run_optuna_tuning",
    "summarize_optuna_study",
    "parameterized_gate",
    "parameterize_features",
    "assign_background_parameter_points",
    "make_background_parameter_replicas",
    "json_safe",
    "write_json",
    "write_csv_records",
    "package_versions",
    "build_method_manifest",
    "write_study_results",
]
