#!/usr/bin/env python3
"""AK4+AK8 XGBoost score-fit limits for the resonant four-Higgs scans.

The resolved, mixed, and boosted reconstruction hypotheses are described by a
single mass-conditioned classifier.  At each generated mass point the held-out
classifier score is divided into a few background-quantile bins and all of
their event counts enter a transparent Poisson likelihood.  The reported
quantity is the expected 95% upper limit on the resonant cross section before
the four Higgs-boson decays.  No pyhf model or nuisance parameter is used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

for _variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import numpy as np

import resonance_fatjet_xgboost_analysis as fat
import resonance_xgboost_analysis as resolved
from c3d4_xgboost_study import weighted_quantile


FEATURE_SET = fat.FEATURE_SET
METHOD_VERSION = "resonance-ak4ak8-score-fit-poisson-v3"
N_FOLDS = 5
SEED = 12345
COLLIDER_ENERGY_TEV = 14.0
LUMINOSITY_FB = 3000.0
HBB_BRANCHING_RATIO = 0.5824
EPS_B = 0.85
EPS_C = 0.10
EPS_LIGHT = 0.01
EPS_BB = EPS_B**2
FAKE_BB = 0.10
SIGNAL_REFERENCE_XSEC_FB = 1.0
BACKGROUND_REPLICAS = 3
MIN_BACKGROUND_SOURCE_EVENTS = 25
MIN_BACKGROUND_NEFF = 5.0
Q95 = 3.841458820694124
EXPECTED_SIGNAL_POINTS = {"direct": 42, "cascade": 441}
SM_ROLES = ("sm_hhhh", "sm_hhhbb", "sm_hh4b")
CATEGORY_NAMES = ("resolved", "mixed", "boosted")
BINNING_SCHEMES: dict[str, tuple[float, ...]] = {
    "background_quantile_4bin": (0.0, 0.50, 0.80, 0.95, 1.0),
    "background_quantile_5bin": (0.0, 0.50, 0.75, 0.90, 0.97, 1.0),
}
REFERENCE_POINTS = {
    "direct": resolved.MassPoint("direct", ms=1500.0),
    "cascade": resolved.MassPoint("cascade", m2=625.0, m3=1500.0),
}
CATEGORY_DIAGNOSTIC_POINTS = {
    "direct": (
        resolved.MassPoint("direct", ms=600.0),
        resolved.MassPoint("direct", ms=1500.0),
        resolved.MassPoint("direct", ms=4000.0),
    ),
    "cascade": (
        resolved.MassPoint("cascade", m2=275.0, m3=600.0),
        resolved.MassPoint("cascade", m2=625.0, m3=1500.0),
        resolved.MassPoint("cascade", m2=1500.0, m3=3500.0),
    ),
}


class ScoreFitError(resolved.AnalysisInputError):
    """Raised when the immutable score-fit contract cannot be satisfied."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScoreFitError(f"cannot read checkpoint {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ScoreFitError(f"checkpoint {path} is not a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def background_normalization_provenance(manifest: Path) -> dict[str, Any]:
    """Require the matching immutable LHE-header normalization audit."""

    audit_path = manifest.with_suffix(".normalization_audit.json")
    result: dict[str, Any] = {
        "status": "invalid",
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "audit": str(audit_path),
        "adopted_samples": [],
    }
    if not audit_path.is_file():
        result["reason"] = "missing immutable normalization audit sidecar"
        return result
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        result["reason"] = f"unreadable normalization audit: {error}"
        return result
    if not isinstance(audit, Mapping) or audit.get("schema") != (
        "resonance-background-normalization-audit-v1"
    ):
        result["reason"] = "unsupported normalization audit schema"
        return result
    if audit.get("output_manifest_sha256") != result["manifest_sha256"]:
        result["reason"] = "normalization audit does not match the manifest"
        return result
    adopted = audit.get("adopted_samples")
    if not isinstance(adopted, list) or not adopted:
        result["reason"] = "normalization audit contains no adopted samples"
        return result
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = {row.get("sample_id", "").strip(): row for row in csv.DictReader(handle)}
    verified: list[dict[str, Any]] = []
    for item in adopted:
        if not isinstance(item, Mapping):
            result["reason"] = "malformed adopted-sample audit entry"
            return result
        sample_id = str(item.get("sample_id", "")).strip()
        row = rows.get(sample_id)
        if row is None:
            result["reason"] = f"audited sample {sample_id!r} is absent"
            return result
        try:
            manifest_xsec = float(row["cross_section_fb"])
            adopted_xsec = float(item["adopted_cross_section_fb"])
            relative_uncertainty = float(item["relative_uncertainty"])
        except (KeyError, TypeError, ValueError) as error:
            result["reason"] = f"invalid audit entry for {sample_id}: {error}"
            return result
        if (
            row.get("normalization_source", "").strip() != "source_lhe_init"
            or not math.isclose(
                manifest_xsec, adopted_xsec, rel_tol=1.0e-10, abs_tol=1.0e-9
            )
            or not math.isfinite(relative_uncertainty)
            or relative_uncertainty < 0.0
        ):
            result["reason"] = f"manifest does not reproduce the audit for {sample_id}"
            return result
        verified.append(
            {
                "sample_id": sample_id,
                "cross_section_fb": manifest_xsec,
                "relative_uncertainty": relative_uncertainty,
                "source_lhe": item.get("source_lhe"),
                "source_lhe_sha256": item.get("source_lhe_sha256"),
            }
        )
    result.update(
        {
            "status": "accepted",
            "reason": (
                "audited LHE-init normalization verified; its integration uncertainty "
                "is recorded but not propagated in this statistics-only limit"
            ),
            "adopted_samples": verified,
        }
    )
    return result


def fold_role_masks(folds: Sequence[int], rotation: int) -> dict[str, np.ndarray]:
    """Return the declared three-train, one-validation, one-test split."""

    array = np.asarray(folds, dtype=int)
    if array.ndim != 1 or np.any((array < 0) | (array >= N_FOLDS)):
        raise ValueError("fold labels must be a one-dimensional array in [0,4]")
    if rotation < 0 or rotation >= N_FOLDS:
        raise ValueError("rotation must lie in [0,4]")
    test = array == rotation
    validation = array == ((rotation + 1) % N_FOLDS)
    return {"test": test, "validation": validation, "train": ~(test | validation)}


def poisson_asimov_deviance(
    observed: Sequence[float], expected: Sequence[float]
) -> float:
    """Twice the exact binned Poisson log-likelihood ratio."""

    observation, expectation = np.broadcast_arrays(
        np.asarray(observed, dtype=float), np.asarray(expected, dtype=float)
    )
    if (
        np.any(~np.isfinite(observation))
        or np.any(~np.isfinite(expectation))
        or np.any(observation < 0.0)
        or np.any(expectation < 0.0)
    ):
        raise ValueError("Poisson counts and means must be finite and non-negative")
    if np.any((observation > 0.0) & (expectation <= 0.0)):
        return math.inf
    terms = expectation - observation
    positive = observation > 0.0
    terms[positive] += observation[positive] * np.log(
        observation[positive] / expectation[positive]
    )
    return max(0.0, float(2.0 * np.sum(terms)))


def poisson_q(
    asimov_counts: Sequence[float], signal_per_fb: Sequence[float], cross_section_fb: float
) -> float:
    """Expected Poisson likelihood-ratio statistic at one cross section."""

    counts = np.asarray(asimov_counts, dtype=float)
    signal = np.asarray(signal_per_fb, dtype=float)
    if counts.ndim != 1 or signal.shape != counts.shape:
        raise ValueError("Asimov and signal templates must be aligned vectors")
    if (
        np.any(~np.isfinite(counts))
        or np.any(~np.isfinite(signal))
        or np.any(counts < 0.0)
        or np.any(signal < 0.0)
    ):
        raise ValueError("event-count templates must be finite and non-negative")
    if not math.isfinite(cross_section_fb) or cross_section_fb < 0.0:
        raise ValueError("cross section must be finite and non-negative")
    expectation = counts + cross_section_fb * signal / SIGNAL_REFERENCE_XSEC_FB
    return float(poisson_asimov_deviance(counts, expectation))


def solve_sigma95(
    asimov_counts: Sequence[float],
    signal_per_fb: Sequence[float],
    *,
    target_q: float = Q95,
) -> float:
    """Solve the monotonic Asimov crossing q(sigma)=3.841 by bisection."""

    counts = np.asarray(asimov_counts, dtype=float)
    signal = np.asarray(signal_per_fb, dtype=float)
    if not math.isfinite(target_q) or target_q <= 0.0:
        raise ValueError("target q must be positive and finite")
    poisson_q(counts, signal, 0.0)
    if not np.any(signal > 0.0):
        raise ScoreFitError("a zero signal template has no finite cross-section limit")
    low = 0.0
    high = 1.0
    while poisson_q(counts, signal, high) < target_q:
        high *= 2.0
        if high > 1.0e15:
            raise ScoreFitError("could not bracket the 95% cross-section limit")
    for _ in range(100):
        middle = 0.5 * (low + high)
        if poisson_q(counts, signal, middle) < target_q:
            low = middle
        else:
            high = middle
    result = 0.5 * (low + high)
    if not math.isfinite(result) or result <= 0.0:
        raise ScoreFitError("the 95% cross-section solution is invalid")
    return result


def occupancy_failures(
    summary: Mapping[str, Sequence[float]],
    *,
    min_source_events: int = MIN_BACKGROUND_SOURCE_EVENTS,
    min_neff: float = MIN_BACKGROUND_NEFF,
) -> list[dict[str, Any]]:
    """List score bins that lack the declared independent MC support."""

    yields = np.asarray(summary["yield"], dtype=float)
    raw = np.asarray(summary["raw"], dtype=int)
    neff = np.asarray(summary["neff"], dtype=float)
    failures: list[dict[str, Any]] = []
    for index, (value, sources, effective) in enumerate(zip(yields, raw, neff)):
        reason = None
        if not math.isfinite(float(value)) or value <= 0.0:
            reason = "nonpositive_background"
        elif int(sources) < int(min_source_events):
            reason = "background_source_events"
        elif not math.isfinite(float(effective)) or effective < float(min_neff):
            reason = "background_neff"
        if reason is not None:
            failures.append(
                {
                    "bin": int(index),
                    "reason": reason,
                    "yield": float(value),
                    "source_events": int(sources),
                    "neff": float(effective),
                }
            )
    return failures


def merge_failing_tail_edges(
    initial_edges_by_fold: Sequence[Sequence[float]],
    summarize: Any,
    *,
    min_source_events: int = MIN_BACKGROUND_SOURCE_EVENTS,
    min_neff: float = MIN_BACKGROUND_NEFF,
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Merge the highest failing ordinal bin into its lower neighbour."""

    edges_by_fold = [np.asarray(item, dtype=float) for item in initial_edges_by_fold]
    if len(edges_by_fold) != N_FOLDS:
        raise ValueError(f"exactly {N_FOLDS} fold-edge arrays are required")
    if len({len(item) for item in edges_by_fold}) != 1:
        raise ValueError("all folds must begin with the same bin count")
    history: list[dict[str, Any]] = []
    while True:
        combined = summarize(edges_by_fold)
        failures = occupancy_failures(
            combined, min_source_events=min_source_events, min_neff=min_neff
        )
        if not failures:
            return edges_by_fold, history, combined
        n_bins = len(edges_by_fold[0]) - 1
        if n_bins <= 1:
            raise ScoreFitError(
                "background occupancy fails after merging to one bin: "
                + json.dumps(failures, sort_keys=True)
            )
        failure = max(failures, key=lambda item: int(item["bin"]))
        failed_bin = int(failure["bin"])
        remove_index = failed_bin if failed_bin > 0 else 1
        before = [item.tolist() for item in edges_by_fold]
        edges_by_fold = [np.delete(item, remove_index) for item in edges_by_fold]
        history.append(
            {
                **failure,
                "failed_bin": failed_bin,
                "removed_boundary_index": remove_index,
                "before": before,
                "after": [item.tolist() for item in edges_by_fold],
            }
        )


def merge_partition_background_edges(
    initial_edges_by_fold: Sequence[Sequence[float]],
    summarizers: Mapping[str, Any],
    *,
    min_source_events: int = MIN_BACKGROUND_SOURCE_EVENTS,
    min_neff: float = MIN_BACKGROUND_NEFF,
) -> tuple[
    list[np.ndarray], list[dict[str, Any]], dict[str, dict[str, np.ndarray]]
]:
    """Coarsen bins until validation and held-out background MC both pass."""

    if set(summarizers) != {"validation", "test"}:
        raise ValueError("validation and test background summarizers are required")
    edges_by_fold = [np.asarray(item, dtype=float) for item in initial_edges_by_fold]
    if len(edges_by_fold) != N_FOLDS or len({len(item) for item in edges_by_fold}) != 1:
        raise ValueError("five aligned fold-edge arrays are required")
    history: list[dict[str, Any]] = []
    while True:
        summaries = {
            partition: summarize(edges_by_fold)
            for partition, summarize in summarizers.items()
        }
        failures = [
            {**failure, "partition": partition}
            for partition in ("validation", "test")
            for failure in occupancy_failures(
                summaries[partition],
                min_source_events=min_source_events,
                min_neff=min_neff,
            )
        ]
        if not failures:
            return edges_by_fold, history, summaries
        n_bins = len(edges_by_fold[0]) - 1
        if n_bins <= 1:
            raise ScoreFitError(
                "background occupancy fails after merging to one bin: "
                + json.dumps(failures, sort_keys=True)
            )
        # Address the highest-score failure first; for an exact tie, the test
        # partition is handled first.  Only background MC support enters this
        # deterministic coarsening -- never signal scores or a fitted limit.
        failure = max(
            failures,
            key=lambda item: (
                int(item["bin"]),
                1 if item["partition"] == "test" else 0,
            ),
        )
        failed_bin = int(failure["bin"])
        remove_index = failed_bin if failed_bin > 0 else 1
        before = [item.tolist() for item in edges_by_fold]
        edges_by_fold = [np.delete(item, remove_index) for item in edges_by_fold]
        history.append(
            {
                **failure,
                "failed_bin": failed_bin,
                "removed_boundary_index": remove_index,
                "before": before,
                "after": [item.tolist() for item in edges_by_fold],
            }
        )


def select_binning_scheme(
    point_results: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Choose one topology-wide scheme using median validation sensitivity."""

    ratios: list[float] = []
    for point in point_results:
        four = point["schemes"]["background_quantile_4bin"]
        five = point["schemes"]["background_quantile_5bin"]
        if four.get("status") != "ok" or five.get("status") != "ok":
            continue
        ratio = float(four["validation_sigma95_fb"]) / float(
            five["validation_sigma95_fb"]
        )
        if math.isfinite(ratio) and ratio > 0.0:
            ratios.append(ratio)
    if not ratios:
        raise ScoreFitError("no mass point supports a four-versus-five-bin comparison")
    median_ratio = float(np.median(np.asarray(ratios, dtype=float)))
    selected = (
        "background_quantile_4bin"
        if median_ratio <= 1.02
        else "background_quantile_5bin"
    )
    return selected, {
        "selected_scheme": selected,
        "median_validation_limit_ratio_four_over_five": median_ratio,
        "points_compared": len(ratios),
        "four_bin_preference_threshold": 1.02,
        "four_bins_preferred_within_two_percent": selected
        == "background_quantile_4bin",
    }


def _combine_summaries(
    parts: Sequence[Mapping[str, np.ndarray]], n_bins: int | None = None
) -> dict[str, np.ndarray]:
    if not parts:
        if n_bins is None:
            raise ValueError("empty summary collection needs an explicit bin count")
        return {
            "yield": np.zeros(n_bins),
            "sumw2": np.zeros(n_bins),
            "raw": np.zeros(n_bins, dtype=int),
            "neff": np.zeros(n_bins),
        }
    return fat.combine_summaries(parts)


def _feature_input_record(spec: resolved.SampleSpec) -> dict[str, Any]:
    if not spec.root_file.is_file() or not spec.summary_file.is_file():
        raise ScoreFitError(f"missing feature input for {spec.sample_id}")
    stat = spec.root_file.stat()
    return {
        "sample_id": spec.sample_id,
        "role": spec.role,
        "point": spec.point.as_dict() if spec.point else None,
        "root_file": str(spec.root_file.resolve()),
        "root_size_bytes": int(stat.st_size),
        "root_mtime_ns": int(stat.st_mtime_ns),
        "root_sha256": _sha256_file(spec.root_file),
        "summary_file": str(spec.summary_file.resolve()),
        "summary_sha256": _sha256_file(spec.summary_file),
    }


def _load_samples(
    topology: str,
    analysis_root: Path,
    signal_manifest: Path,
    background_manifest: Path,
    load_jobs: int,
) -> tuple[
    list[resolved.LoadedSample],
    list[resolved.LoadedSample],
    list[resolved.MassPoint],
    list[dict[str, Any]],
]:
    signal_specs = resolved.load_signal_specs(
        signal_manifest,
        topology,
        analysis_root / "ResonanceAnalysis/features/ak8-v1",
        "{scenario}/{run_name}_fatjet.root",
    )
    signal_specs = sorted(signal_specs, key=lambda item: item.point.sort_key)
    expected = EXPECTED_SIGNAL_POINTS[topology]
    if len(signal_specs) != expected:
        raise ScoreFitError(
            f"{topology} requires exactly {expected} AK4/AK8 signal pairs; "
            f"found {len(signal_specs)}"
        )
    point_ids = [item.point.point_id for item in signal_specs]
    if len(set(point_ids)) != expected:
        raise ScoreFitError(f"{topology} signal manifest contains duplicate mass points")
    background_specs, missing = resolved.load_background_specs(
        background_manifest, analysis_root, default_k_factor=2.0
    )
    if missing or len(background_specs) != 14:
        raise ScoreFitError(
            "the immutable resonance background manifest must provide all 14 feature pairs"
        )
    roles = [item.role for item in background_specs]
    for role in SM_ROLES:
        if roles.count(role) != 1:
            raise ScoreFitError(f"background manifest requires exactly one {role} sample")
    fat.TAGGING_SCENARIOS = {
        "nominal": {"eps_bb": EPS_BB, "fake_bb": FAKE_BB}
    }
    load_args = argparse.Namespace(
        tree_name="ResonanceFeatures",
        mode="simple",
        seed=SEED,
        tagging_scenarios=fat.TAGGING_SCENARIOS,
        luminosity=LUMINOSITY_FB,
        hbb_branching_ratio=HBB_BRANCHING_RATIO,
        eps_b=EPS_B,
        eps_c=EPS_C,
        eps_light=EPS_LIGHT,
    )
    all_specs = [*signal_specs, *background_specs]
    loaded_by_id: dict[str, resolved.LoadedSample] = {}
    # Tiresias deliberately uses process workers here: the established feature
    # environment falls back to PyROOT, whose global state is not thread-safe.
    with ProcessPoolExecutor(max_workers=max(1, int(load_jobs))) as executor:
        futures = {
            executor.submit(fat.load_sample, item, topology, load_args): item
            for item in all_specs
        }
        for future in as_completed(futures):
            item = futures[future]
            loaded_by_id[item.sample_id] = future.result()
    signals = [loaded_by_id[item.sample_id] for item in signal_specs]
    backgrounds = [loaded_by_id[item.sample_id] for item in background_specs]
    points = [item.spec.point for item in signals]
    return signals, backgrounds, points, missing


def _background_mass_features(
    sample: resolved.LoadedSample,
    topology: str,
    points: Sequence[resolved.MassPoint],
    indices: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    assigned = [points[int(index)] for index in indices]
    if topology == "direct":
        return resolved.engineer_features(
            sample.base_features,
            sample.base_feature_names,
            sample.table,
            topology,
            ms=np.asarray([point.ms for point in assigned], dtype=float),
        )
    return resolved.engineer_features(
        sample.base_features,
        sample.base_feature_names,
        sample.table,
        topology,
        m2=np.asarray([point.m2 for point in assigned], dtype=float),
        m3=np.asarray([point.m3 for point in assigned], dtype=float),
    )


def _train_one_fold(
    rotation: int,
    topology: str,
    signal_matrix: np.ndarray,
    signal_weights: np.ndarray,
    signal_folds: np.ndarray,
    signal_point_ids: np.ndarray,
    backgrounds: Sequence[resolved.LoadedSample],
    points: Sequence[resolved.MassPoint],
    assignments: Mapping[str, np.ndarray],
    feature_names: tuple[str, ...],
    output_path: Path,
    threads: int,
) -> Path:
    import xgboost as xgb  # type: ignore

    masks = fold_role_masks(signal_folds, rotation)
    signal_train = masks["train"]
    sx = signal_matrix[signal_train]
    sw = resolved._equal_signal_point_weights(
        signal_weights, signal_point_ids, signal_train
    )[signal_train]
    bx_parts: list[np.ndarray] = []
    bw_parts: list[np.ndarray] = []
    for sample in backgrounds:
        train = fold_role_masks(sample.folds, rotation)["train"]
        physical = np.abs(sample.scenario_weights["nominal"])
        for replica in range(BACKGROUND_REPLICAS):
            features, names = _background_mass_features(
                sample,
                topology,
                points,
                assignments[sample.spec.sample_id][:, replica],
            )
            if names != feature_names:
                raise ScoreFitError("background and signal feature contracts differ")
            bx_parts.append(features[train])
            bw_parts.append(physical[train] / BACKGROUND_REPLICAS)
    bx = np.concatenate(bx_parts)
    bw = np.concatenate(bw_parts)
    if len(sx) == 0 or len(bx) == 0 or float(np.sum(bw)) <= 0.0:
        raise ScoreFitError(f"fold {rotation} has an empty classifier class")
    common_total = 0.5 * float(len(sx) + len(bx))
    sw *= common_total
    bw *= common_total / float(np.sum(bw))
    features = np.concatenate([sx, bx])
    labels = np.concatenate(
        [np.ones(len(sx), dtype=np.int8), np.zeros(len(bx), dtype=np.int8)]
    )
    weights = np.concatenate([sw, bw])
    params = dict(resolved.FIXED_XGBOOST_PARAMS)
    params.update(random_state=SEED + rotation, n_jobs=int(threads))
    model = xgb.XGBClassifier(**params)
    model.fit(features, labels, sample_weight=weights, verbose=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # XGBoost selects the serialization format from the suffix.  Preserve the
    # final suffix on the atomic temporary path so a JSON model cannot be
    # written as UBJSON and subsequently mislabelled.
    temporary = output_path.with_name(
        f".{output_path.stem}.tmp-{os.getpid()}-{rotation}{output_path.suffix}"
    )
    model.save_model(str(temporary))
    os.replace(temporary, output_path)
    return output_path


def load_or_train_models(
    topology: str,
    signals: Sequence[resolved.LoadedSample],
    backgrounds: Sequence[resolved.LoadedSample],
    points: Sequence[resolved.MassPoint],
    model_dir: Path,
    input_fingerprint: str,
    *,
    model_jobs: int,
    xgboost_threads: int,
) -> tuple[list[Any], tuple[str, ...], dict[str, str], bool]:
    """Load a verified five-model cache or train the five folds concurrently."""

    import xgboost as xgb  # type: ignore

    model_dir.mkdir(parents=True, exist_ok=True)
    paths = [model_dir / f"{topology}_fold{fold}.json" for fold in range(N_FOLDS)]
    manifest_path = model_dir / "model_manifest.json"
    expected_names = fat.point_features(signals[0], signals[0].spec.point)[1]
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("input_fingerprint") != input_fingerprint:
            raise ScoreFitError("cached classifiers do not match the immutable inputs")
        if tuple(manifest.get("feature_names", ())) != expected_names:
            raise ScoreFitError("cached classifier feature contract has changed")
        models: list[Any] = []
        hashes: dict[str, str] = {}
        for fold, path in enumerate(paths):
            digest = _sha256_file(path)
            if manifest.get("model_sha256", {}).get(path.name) != digest:
                raise ScoreFitError(f"cached classifier hash mismatch for {path}")
            params = dict(resolved.FIXED_XGBOOST_PARAMS)
            params.update(random_state=SEED + fold, n_jobs=1)
            model = xgb.XGBClassifier(**params)
            model.load_model(str(path))
            models.append(model)
            hashes[path.name] = digest
        return models, expected_names, hashes, True
    if any(path.exists() for path in paths):
        raise ScoreFitError("unmanifested model files would contaminate this run")

    signal_features: list[np.ndarray] = []
    signal_weights: list[np.ndarray] = []
    signal_folds: list[np.ndarray] = []
    signal_point_ids: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    for sample in signals:
        features, names = fat.point_features(sample, sample.spec.point)
        feature_names = names if feature_names is None else feature_names
        if names != feature_names:
            raise ScoreFitError("signal feature contracts differ")
        signal_features.append(features)
        signal_weights.append(np.abs(sample.scenario_weights["nominal"]))
        signal_folds.append(sample.folds)
        signal_point_ids.append(
            np.full(sample.table.entries, sample.spec.point.point_id, dtype=object)
        )
    sx = np.concatenate(signal_features)
    sw = np.concatenate(signal_weights)
    sf = np.concatenate(signal_folds)
    sp = np.concatenate(signal_point_ids)
    assignments = {
        sample.spec.sample_id: fat._event_point_assignments(
            sample, points, BACKGROUND_REPLICAS, SEED
        )
        for sample in backgrounds
    }
    workers = min(N_FOLDS, int(model_jobs))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _train_one_fold,
                fold,
                topology,
                sx,
                sw,
                sf,
                sp,
                backgrounds,
                points,
                assignments,
                tuple(feature_names or ()),
                paths[fold],
                xgboost_threads,
            ): fold
            for fold in range(N_FOLDS)
        }
        for future in as_completed(futures):
            future.result()
    hashes = {path.name: _sha256_file(path) for path in paths}
    _atomic_json(
        manifest_path,
        {
            "method_version": METHOD_VERSION,
            "input_fingerprint": input_fingerprint,
            "topology": topology,
            "folds": N_FOLDS,
            "seed": SEED,
            "split": "test f, validation f+1, remaining three folds train",
            "background_mass_replicas": BACKGROUND_REPLICAS,
            "feature_names": list(feature_names or ()),
            "fixed_xgboost_parameters": {
                **resolved.FIXED_XGBOOST_PARAMS,
                "n_jobs": "runtime parallelism only",
            },
            "model_sha256": hashes,
        },
    )
    models = []
    for fold, path in enumerate(paths):
        params = dict(resolved.FIXED_XGBOOST_PARAMS)
        params.update(random_state=SEED + fold, n_jobs=1)
        model = xgb.XGBClassifier(**params)
        model.load_model(str(path))
        models.append(model)
    return models, tuple(feature_names or ()), hashes, False


def predict_point_crossfit(
    sample: resolved.LoadedSample,
    point: resolved.MassPoint,
    models: Sequence[Any],
) -> resolved.CrossfitScores:
    """Predict each row once in its test role and once in its validation role."""

    if len(models) != N_FOLDS:
        raise ValueError(f"expected {N_FOLDS} cross-fit models")
    features, _ = fat.point_features(sample, point)
    test = np.full(sample.table.entries, np.nan, dtype=float)
    validation = np.full(sample.table.entries, np.nan, dtype=float)
    for rotation, model in enumerate(models):
        masks = fold_role_masks(sample.folds, rotation)
        positions = np.flatnonzero(masks["test"] | masks["validation"])
        prediction = np.asarray(model.predict_proba(features[positions]), dtype=float)[:, 1]
        local_test = masks["test"][positions]
        test[positions[local_test]] = prediction[local_test]
        validation[positions[~local_test]] = prediction[~local_test]
    if (
        np.any(~np.isfinite(test))
        or np.any(~np.isfinite(validation))
        or np.any((test < 0.0) | (test > 1.0))
        or np.any((validation < 0.0) | (validation > 1.0))
    ):
        raise ScoreFitError(f"{sample.spec.sample_id}: incomplete cross-fit scores")
    return resolved.CrossfitScores(test=test, validation=validation)


def _score_cache_path(base: Path, point_id: str, sample_id: str) -> Path:
    safe = sample_id.replace("/", "_")
    return base / point_id / f"{safe}.npz"


def load_or_predict_scores(
    sample: resolved.LoadedSample,
    point: resolved.MassPoint,
    models: Sequence[Any],
    cache_base: Path,
    core_fingerprint: str,
) -> resolved.CrossfitScores:
    path = _score_cache_path(cache_base, point.point_id, sample.spec.sample_id)
    if path.is_file():
        with np.load(path, allow_pickle=False) as payload:
            fingerprint = str(np.asarray(payload["core_fingerprint"]).item())
            test = np.asarray(payload["test"], dtype=float)
            validation = np.asarray(payload["validation"], dtype=float)
        if (
            fingerprint != core_fingerprint
            or len(test) != sample.table.entries
            or len(validation) != sample.table.entries
        ):
            raise ScoreFitError(f"{path}: stale score checkpoint")
        return resolved.CrossfitScores(test=test, validation=validation)
    scores = predict_point_crossfit(sample, point, models)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        core_fingerprint=np.asarray(core_fingerprint),
        test=scores.test,
        validation=scores.validation,
    )
    os.replace(temporary, path)
    return scores


def _derive_fold_edges(
    samples: Sequence[resolved.LoadedSample],
    scores: Mapping[str, resolved.CrossfitScores],
    quantiles: Sequence[float],
) -> list[np.ndarray]:
    edges_by_fold: list[np.ndarray] = []
    for rotation in range(N_FOLDS):
        score_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        for sample in samples:
            validation = fold_role_masks(sample.folds, rotation)["validation"]
            score_parts.append(scores[sample.spec.sample_id].validation[validation])
            weight_parts.append(
                np.abs(sample.scenario_weights["nominal"][validation])
            )
        values = np.concatenate(score_parts)
        weights = np.concatenate(weight_parts)
        edges = weighted_quantile(values, quantiles, weights)
        edges = np.clip(np.asarray(edges, dtype=float), 0.0, 1.0)
        edges[0] = 0.0
        edges[-1] = 1.0
        if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) < 0.0):
            raise ScoreFitError(f"fold {rotation}: invalid background quantile edges")
        edges_by_fold.append(edges)
    return edges_by_fold


def _sample_partition_summary(
    sample: resolved.LoadedSample,
    scores: resolved.CrossfitScores,
    edges_by_fold: Sequence[Sequence[float]],
    split: str,
) -> dict[str, np.ndarray]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    parts: list[dict[str, np.ndarray]] = []
    event_indices = np.asarray(sample.table.arrays["event_index"], dtype=np.int64)
    physical = sample.scenario_weights["nominal"]
    score_values = scores.validation if split == "validation" else scores.test
    for rotation, edges in enumerate(edges_by_fold):
        mask = fold_role_masks(sample.folds, rotation)[split]
        parts.append(
            fat.grouped_binned_summary(
                score_values,
                physical,
                event_indices,
                edges,
                mask,
            )
        )
    return _combine_summaries(parts)


def _all_sample_summaries(
    samples: Sequence[resolved.LoadedSample],
    scores: Mapping[str, resolved.CrossfitScores],
    edges_by_fold: Sequence[Sequence[float]],
    split: str,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        sample.spec.sample_id: _sample_partition_summary(
            sample,
            scores[sample.spec.sample_id],
            edges_by_fold,
            split,
        )
        for sample in samples
    }


def _sum_sample_summaries(
    summaries: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return _combine_summaries(list(summaries.values()))


def build_asimov_counts(
    backgrounds: Sequence[resolved.LoadedSample],
    summaries: Mapping[str, Mapping[str, np.ndarray]],
) -> np.ndarray:
    """Build B+SM hhhh+SM hhhbb+SM hh4b and enforce all three SM inputs."""

    present_roles = [sample.spec.role for sample in backgrounds]
    for role in SM_ROLES:
        if present_roles.count(role) != 1:
            raise ScoreFitError(f"Asimov construction requires exactly one {role} sample")
    expected_ids = {sample.spec.sample_id for sample in backgrounds}
    if set(summaries) != expected_ids:
        missing = sorted(expected_ids.difference(summaries))
        extra = sorted(set(summaries).difference(expected_ids))
        raise ScoreFitError(f"Asimov sample mismatch; missing={missing}, extra={extra}")
    return np.asarray(_sum_sample_summaries(summaries)["yield"], dtype=float)


def _check_yield_closure(
    samples: Sequence[resolved.LoadedSample],
    summaries: Mapping[str, Mapping[str, np.ndarray]],
    split: str,
) -> None:
    for sample in samples:
        expected = float(np.sum(sample.scenario_weights["nominal"]))
        observed = float(np.sum(summaries[sample.spec.sample_id]["yield"]))
        tolerance = 1.0e-9 * max(1.0, abs(expected))
        if not math.isclose(observed, expected, rel_tol=1.0e-9, abs_tol=tolerance):
            raise ScoreFitError(
                f"{sample.spec.sample_id}: {split} score bins give {observed:g}, "
                f"not the physical yield {expected:g}"
            )


def _templates_are_physical(
    summaries: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for sample_id, summary in summaries.items():
        yields = np.asarray(summary["yield"], dtype=float)
        if np.any(~np.isfinite(yields)) or np.any(yields < -1.0e-12):
            failures.append(sample_id)
    return not failures, failures


def _scheme_result(
    scheme_name: str,
    quantiles: Sequence[float],
    signal: resolved.LoadedSample,
    backgrounds: Sequence[resolved.LoadedSample],
    scores: Mapping[str, resolved.CrossfitScores],
) -> dict[str, Any]:
    initial_edges = _derive_fold_edges(backgrounds, scores, quantiles)

    def validation_total(edges: Sequence[Sequence[float]]) -> dict[str, np.ndarray]:
        summaries = _all_sample_summaries(backgrounds, scores, edges, "validation")
        return _sum_sample_summaries(summaries)

    def test_total(edges: Sequence[Sequence[float]]) -> dict[str, np.ndarray]:
        summaries = _all_sample_summaries(backgrounds, scores, edges, "test")
        return _sum_sample_summaries(summaries)

    try:
        edges, merges, partition_backgrounds = merge_partition_background_edges(
            initial_edges,
            {"validation": validation_total, "test": test_total},
        )
    except ScoreFitError as error:
        return {
            "status": "invalid",
            "reason": str(error),
            "scheme": scheme_name,
            "requested_quantiles": list(quantiles),
            "initial_edges_by_fold": initial_edges,
        }
    validation_background = partition_backgrounds["validation"]
    held_out_background = partition_backgrounds["test"]

    validation_samples = _all_sample_summaries(backgrounds, scores, edges, "validation")
    validation_signal = _sample_partition_summary(
        signal, scores[signal.spec.sample_id], edges, "validation"
    )
    test_samples = _all_sample_summaries(backgrounds, scores, edges, "test")
    test_signal = _sample_partition_summary(
        signal, scores[signal.spec.sample_id], edges, "test"
    )
    _check_yield_closure(backgrounds, validation_samples, "validation")
    _check_yield_closure(backgrounds, test_samples, "test")
    _check_yield_closure(
        [signal], {signal.spec.sample_id: validation_signal}, "validation"
    )
    _check_yield_closure([signal], {signal.spec.sample_id: test_signal}, "test")

    held_out_failures = occupancy_failures(held_out_background)
    validation_physical, validation_bad = _templates_are_physical(validation_samples)
    test_physical, test_bad = _templates_are_physical(test_samples)
    signal_physical = bool(
        np.all(np.isfinite(validation_signal["yield"]))
        and np.all(validation_signal["yield"] >= -1.0e-12)
        and np.all(np.isfinite(test_signal["yield"]))
        and np.all(test_signal["yield"] >= -1.0e-12)
    )
    common: dict[str, Any] = {
        "scheme": scheme_name,
        "requested_quantiles": list(quantiles),
        "initial_edges_by_fold": initial_edges,
        "final_edges_by_fold": edges,
        "merges": merges,
        "merge_partitions": ["validation", "test"],
        "final_bin_count": len(edges[0]) - 1,
        "validation_background_occupancy": validation_background,
        "test_background_occupancy": held_out_background,
        "test_occupancy_failures": held_out_failures,
        "validation_nonphysical_samples": validation_bad,
        "test_nonphysical_samples": test_bad,
    }
    if held_out_failures or not validation_physical or not test_physical or not signal_physical:
        return {
            **common,
            "status": "invalid",
            "reason": "held-out occupancy or non-negative-template validation failed",
        }

    validation_asimov = build_asimov_counts(backgrounds, validation_samples)
    test_asimov = build_asimov_counts(backgrounds, test_samples)
    validation_signal_yield = np.maximum(
        np.asarray(validation_signal["yield"], dtype=float), 0.0
    )
    test_signal_yield = np.maximum(np.asarray(test_signal["yield"], dtype=float), 0.0)
    validation_limit = solve_sigma95(validation_asimov, validation_signal_yield)
    test_limit = solve_sigma95(test_asimov, test_signal_yield)
    return {
        **common,
        "status": "ok",
        "validation_sigma95_fb": validation_limit,
        "test_sigma95_fb": test_limit,
        "q_at_test_sigma95": poisson_q(test_asimov, test_signal_yield, test_limit),
        "validation_signal": validation_signal,
        "test_signal": test_signal,
        "validation_samples": validation_samples,
        "test_samples": test_samples,
        "validation_asimov": validation_asimov,
        "test_asimov": test_asimov,
    }


def _point_checkpoint_path(base: Path, point: resolved.MassPoint) -> Path:
    return base / f"{point.point_id}.json"


def analyze_point(
    point: resolved.MassPoint,
    signal: resolved.LoadedSample,
    backgrounds: Sequence[resolved.LoadedSample],
    models: Sequence[Any],
    score_cache: Path,
    point_cache: Path,
    core_fingerprint: str,
) -> dict[str, Any]:
    checkpoint = _point_checkpoint_path(point_cache, point)
    if checkpoint.is_file():
        payload = _read_json(checkpoint)
        if payload.get("core_fingerprint") != core_fingerprint:
            raise ScoreFitError(f"{checkpoint}: stale point checkpoint")
        return payload
    samples = [signal, *backgrounds]
    scores = {
        sample.spec.sample_id: load_or_predict_scores(
            sample, point, models, score_cache, core_fingerprint
        )
        for sample in samples
    }
    schemes = {
        name: _scheme_result(name, quantiles, signal, backgrounds, scores)
        for name, quantiles in BINNING_SCHEMES.items()
    }
    payload = {
        "method_version": METHOD_VERSION,
        "core_fingerprint": core_fingerprint,
        "point": point.as_dict(),
        "signal_sample_id": signal.spec.sample_id,
        "schemes": schemes,
    }
    _atomic_json(checkpoint, payload)
    return _read_json(checkpoint)


def _selected_scheme_payload(
    point_result: Mapping[str, Any], selected_scheme: str
) -> Mapping[str, Any]:
    payload = point_result["schemes"][selected_scheme]
    if payload.get("status") != "ok":
        raise ScoreFitError(
            f"{point_result['point']['point_id']}: selected score binning is invalid"
        )
    return payload


def _point_limit_rows(
    point_results: Sequence[Mapping[str, Any]], selected_scheme: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in sorted(
        point_results,
        key=lambda item: (
            float(item["point"].get("M3_GeV") or item["point"].get("MS_GeV") or 0.0),
            float(item["point"].get("M2_GeV") or 0.0),
        ),
    ):
        selected = point["schemes"][selected_scheme]
        four = point["schemes"]["background_quantile_4bin"]
        five = point["schemes"]["background_quantile_5bin"]
        occupancy = selected.get("test_background_occupancy", {})
        raw = np.asarray(occupancy.get("raw", []), dtype=float)
        neff = np.asarray(occupancy.get("neff", []), dtype=float)
        asimov = np.asarray(selected.get("test_asimov", []), dtype=float)
        signal = np.asarray(selected.get("test_signal", {}).get("yield", []), dtype=float)
        rows.append(
            {
                "point_id": point["point"]["point_id"],
                "topology": point["point"]["topology"],
                "MS_GeV": point["point"].get("MS_GeV"),
                "M2_GeV": point["point"].get("M2_GeV"),
                "M3_GeV": point["point"].get("M3_GeV"),
                "selected_binning": selected_scheme,
                "selected_bins": selected.get("final_bin_count"),
                "status": selected.get("status"),
                "sigma95_fb": selected.get("test_sigma95_fb"),
                "validation_sigma95_fb": selected.get("validation_sigma95_fb"),
                "four_bin_sigma95_fb": four.get("test_sigma95_fb"),
                "five_bin_sigma95_fb": five.get("test_sigma95_fb"),
                "signal_1fb_yield": float(np.sum(signal)) if len(signal) else None,
                "asimov_yield": float(np.sum(asimov)) if len(asimov) else None,
                "minimum_background_source_events": int(np.min(raw)) if len(raw) else None,
                "minimum_background_neff": float(np.min(neff)) if len(neff) else None,
                "fold_edges_json": json.dumps(
                    selected.get("final_edges_by_fold", []), separators=(",", ":")
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])
    os.replace(temporary, path)


def _write_selected_template_table(
    output_dir: Path,
    point_results: Sequence[Mapping[str, Any]],
    selected_scheme: str,
) -> None:
    rows: list[dict[str, Any]] = []
    for point in point_results:
        selected = point["schemes"][selected_scheme]
        if selected.get("status") != "ok":
            continue
        point_fields = {
            key: point["point"].get(key)
            for key in ("point_id", "topology", "MS_GeV", "M2_GeV", "M3_GeV")
        }
        samples = dict(selected["test_samples"])
        samples[point["signal_sample_id"]] = selected["test_signal"]
        for sample_id, template in samples.items():
            for index, value in enumerate(template["yield"]):
                rows.append(
                    {
                        **point_fields,
                        "sample_id": sample_id,
                        "ordinal_score_bin": index + 1,
                        "yield": value,
                        "sumw2": template["sumw2"][index],
                        "source_events": template["raw"][index],
                        "neff": template["neff"][index],
                    }
                )
    _write_csv(output_dir / "selected_score_templates.csv", rows)


def _process_label(sample_id: str, role: str) -> str:
    if role == "sm_hhhh":
        return "SM hhhh"
    if role == "sm_hhhbb":
        return "SM hhhbb"
    if role == "sm_hh4b":
        return "SM hh+4b"
    replacements = {
        "HW-": "",
        "gg_to_": "",
        "pp_to_": "",
        "_SM_HEFT": "",
        "_SM": "",
        "_": " ",
    }
    label = sample_id
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def _yield_table_records(
    point_result: Mapping[str, Any],
    selected_scheme: str,
    backgrounds: Sequence[resolved.LoadedSample],
) -> list[dict[str, Any]]:
    selected = _selected_scheme_payload(point_result, selected_scheme)
    n_bins = int(selected["final_bin_count"])
    sample_by_id = {sample.spec.sample_id: sample for sample in backgrounds}
    rows: list[dict[str, Any]] = []

    def append(label: str, kind: str, values: Sequence[float]) -> None:
        array = np.asarray(values, dtype=float)
        row: dict[str, Any] = {"process": label, "kind": kind}
        row.update({f"score_bin_{index + 1}": value for index, value in enumerate(array)})
        row["total"] = float(np.sum(array))
        rows.append(row)

    append("Resonant signal (1 fb)", "signal", selected["test_signal"]["yield"])
    conventional: list[np.ndarray] = []
    sm: list[np.ndarray] = []
    ordered = sorted(
        selected["test_samples"],
        key=lambda sample_id: (
            0 if sample_by_id[sample_id].spec.role in SM_ROLES else 1,
            SM_ROLES.index(sample_by_id[sample_id].spec.role)
            if sample_by_id[sample_id].spec.role in SM_ROLES
            else sample_id,
        ),
    )
    for sample_id in ordered:
        sample = sample_by_id[sample_id]
        values = np.asarray(selected["test_samples"][sample_id]["yield"], dtype=float)
        kind = "SM multi-Higgs" if sample.spec.role in SM_ROLES else "background"
        append(_process_label(sample_id, sample.spec.role), kind, values)
        (sm if sample.spec.role in SM_ROLES else conventional).append(values)
    conventional_total = (
        np.sum(conventional, axis=0) if conventional else np.zeros(n_bins, dtype=float)
    )
    sm_total = np.sum(sm, axis=0) if sm else np.zeros(n_bins, dtype=float)
    append("Total conventional background", "total", conventional_total)
    append("Total SM multi-Higgs", "total", sm_total)
    append("Asimov total", "total", conventional_total + sm_total)
    return rows


def _tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
    )


def _write_yield_table(
    output_dir: Path,
    topology: str,
    point_results: Sequence[Mapping[str, Any]],
    selected_scheme: str,
    backgrounds: Sequence[resolved.LoadedSample],
) -> list[dict[str, Any]]:
    reference = REFERENCE_POINTS[topology].point_id
    point_result = next(
        (item for item in point_results if item["point"]["point_id"] == reference), None
    )
    if point_result is None:
        raise ScoreFitError(f"missing requested yield-table point {reference}")
    rows = _yield_table_records(point_result, selected_scheme, backgrounds)
    _write_csv(output_dir / f"score_yields_{reference}.csv", rows)
    _atomic_json(output_dir / f"score_yields_{reference}.json", {"rows": rows})
    bin_fields = [field for field in rows[0] if field.startswith("score_bin_")]
    lines = [
        r"\begin{tabular}{l" + "r" * (len(bin_fields) + 1) + "}",
        r"\hline",
        "Process & "
        + " & ".join(f"Bin {index + 1}" for index in range(len(bin_fields)))
        + r" & Total \\",
        r"\hline",
    ]
    for row in rows:
        values = [float(row[field]) for field in bin_fields]
        lines.append(
            _tex_escape(str(row["process"]))
            + " & "
            + " & ".join(f"{value:.3g}" for value in values)
            + f" & {float(row['total']):.3g} "
            + r"\\"
        )
        if str(row["process"]) in {
            "Resonant signal (1 fb)",
            "Total conventional background",
            "Total SM multi-Higgs",
        }:
            lines.append(r"\hline")
    lines.extend([r"\hline", r"\end{tabular}"])
    (output_dir / f"score_yields_{reference}.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    widths = [max(len(str(row["process"])) for row in rows)] + [12] * (
        len(bin_fields) + 1
    )
    header = ["Process", *[f"Bin {i + 1}" for i in range(len(bin_fields))], "Total"]
    print("\n" + " | ".join(value.ljust(width) for value, width in zip(header, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        values = [str(row["process"])] + [
            f"{float(row[field]):.4g}" for field in bin_fields
        ] + [f"{float(row['total']):.4g}"]
        print(" | ".join(value.ljust(width) for value, width in zip(values, widths)))
    return rows


def _category_diagnostics(
    output_dir: Path,
    topology: str,
    signals: Sequence[resolved.LoadedSample],
) -> list[dict[str, Any]]:
    wanted = {point.point_id for point in CATEGORY_DIAGNOSTIC_POINTS[topology]}
    rows: list[dict[str, Any]] = []
    for sample in signals:
        point = sample.spec.point
        if point.point_id not in wanted:
            continue
        events = np.asarray(sample.table.arrays["event_index"], dtype=np.int64)
        categories = np.asarray(sample.table.arrays["category"], dtype=int)
        weights = sample.scenario_weights["nominal"]
        total = float(np.sum(weights))
        for index, name in enumerate(CATEGORY_NAMES):
            summary = fat.grouped_binned_summary(
                np.full(sample.table.entries, 0.5),
                weights,
                events,
                (0.0, 1.0),
                categories == index,
            )
            value = float(summary["yield"][0])
            rows.append(
                {
                    **point.as_dict(),
                    "category": name,
                    "yield_for_1fb": value,
                    "fraction": value / total if total > 0.0 else 0.0,
                    "source_events": int(summary["raw"][0]),
                    "neff": float(summary["neff"][0]),
                }
            )
    expected_rows = len(CATEGORY_DIAGNOSTIC_POINTS[topology]) * len(CATEGORY_NAMES)
    if len(rows) != expected_rows:
        raise ScoreFitError("one or more category-diagnostic mass points are absent")
    _write_csv(output_dir / "category_yield_diagnostics.csv", rows)
    _atomic_json(output_dir / "category_yield_diagnostics.json", {"rows": rows})
    return rows


def _plot_category_diagnostics(
    output_dir: Path, topology: str, rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[str, dict[str, float]] = {}
    labels: dict[str, str] = {}
    for row in rows:
        point_id = str(row["point_id"])
        grouped.setdefault(point_id, {})[str(row["category"])] = float(row["fraction"])
        if topology == "direct":
            labels[point_id] = rf"${float(row['MS_GeV']) / 1000.0:g}$"
        else:
            labels[point_id] = (
                rf"$({float(row['M2_GeV']):g},\,{float(row['M3_GeV']):g})$"
            )
    order = [point.point_id for point in CATEGORY_DIAGNOSTIC_POINTS[topology]]
    x = np.arange(len(order), dtype=float)
    bottom = np.zeros(len(order), dtype=float)
    colors = {"resolved": "#4477AA", "mixed": "#EEAA33", "boosted": "#228833"}
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for category in CATEGORY_NAMES:
        values = np.asarray([grouped[item][category] for item in order], dtype=float)
        ax.bar(x, values, bottom=bottom, label=category.capitalize(), color=colors[category])
        bottom += values
    ax.set_xticks(x, [labels[item] for item in order])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of the 1-fb signal yield", fontsize=14)
    ax.set_xlabel(
        r"$M_S$ [TeV]" if topology == "direct" else r"$(M_2,M_3)$ [GeV]",
        fontsize=14,
    )
    ax.tick_params(labelsize=12)
    ax.legend(frameon=False, fontsize=11, ncol=3, loc="upper center")
    ax.set_title("AK4/AK8 reconstruction composition", fontsize=15)
    fig.tight_layout()
    outputs: list[str] = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"{topology}_category_yields.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _plot_direct_limit(
    output_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda item: float(item["MS_GeV"]))
    masses = np.asarray([float(item["MS_GeV"]) for item in ordered]) / 1000.0
    limits = np.asarray([float(item["sigma95_fb"]) for item in ordered])
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.plot(masses, limits, color="#C51B29", marker="o", markersize=4.5, linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_S$ [TeV]", fontsize=15)
    ax.set_ylabel(r"Expected 95% upper limit on $\sigma_{\rm dir}$ [fb]", fontsize=15)
    ax.tick_params(axis="both", which="both", labelsize=13)
    ax.grid(True, which="both", alpha=0.22)
    ax.text(
        0.03,
        0.96,
        rf"$\sqrt{{s}}={COLLIDER_ENERGY_TEV:g}$ TeV, "
        rf"${LUMINOSITY_FB / 1000.0:g}$ ab$^{{-1}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )
    ax.set_title("Direct resonant four-Higgs production", fontsize=16)
    fig.tight_layout()
    outputs: list[str] = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"direct_expected_cross_section_limit.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _cascade_interpolation(
    rows: Sequence[Mapping[str, Any]], grid_size: int = 420
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ma.MaskedArray,
    np.ma.MaskedArray,
    dict[str, Any],
]:
    from scipy.interpolate import (  # type: ignore
        CloughTocher2DInterpolator,
        LinearNDInterpolator,
    )
    from scipy.spatial import Delaunay  # type: ignore

    m2 = np.asarray([float(item["M2_GeV"]) for item in rows], dtype=float)
    m3 = np.asarray([float(item["M3_GeV"]) for item in rows], dtype=float)
    log_limit = np.log10(np.asarray([float(item["sigma95_fb"]) for item in rows]))
    low = np.asarray([np.min(m2), np.min(m3)], dtype=float)
    span = np.asarray([np.ptp(m2), np.ptp(m3)], dtype=float)
    if np.any(span <= 0.0):
        raise ScoreFitError("cascade mass coordinates cannot be rescaled")
    coordinates = np.column_stack([(m2 - low[0]) / span[0], (m3 - low[1]) / span[1]])
    triangulation = Delaunay(coordinates)
    clough_interpolator = CloughTocher2DInterpolator(coordinates, log_limit)
    linear_interpolator = LinearNDInterpolator(coordinates, log_limit)
    gx = np.linspace(np.min(m2), np.max(m2), grid_size)
    gy = np.linspace(np.min(m3), np.max(m3), grid_size)
    xx, yy = np.meshgrid(gx, gy)
    scaled = np.column_stack(
        [((xx.ravel() - low[0]) / span[0]), ((yy.ravel() - low[1]) / span[1])]
    )
    simplex = triangulation.find_simplex(scaled)
    inside = (simplex >= 0) & (yy.ravel() > 2.0 * xx.ravel())
    clough_values = np.full(len(scaled), np.nan, dtype=float)
    linear_values = np.full(len(scaled), np.nan, dtype=float)
    clough_values[inside] = np.asarray(
        clough_interpolator(scaled[inside]), dtype=float
    )
    linear_values[inside] = np.asarray(
        linear_interpolator(scaled[inside]), dtype=float
    )
    clough_finite_inside = inside & np.isfinite(clough_values)
    linear_finite_inside = inside & np.isfinite(linear_values)
    unsupported = np.zeros(len(scaled), dtype=bool)
    if np.any(clough_finite_inside):
        vertices = triangulation.simplices[simplex[clough_finite_inside]]
        local_values = log_limit[vertices]
        local_min = np.min(local_values, axis=1)
        local_max = np.max(local_values, axis=1)
        tolerance = np.maximum(0.10, 0.25 * (local_max - local_min))
        candidate = clough_values[clough_finite_inside]
        unsupported_values = (candidate < local_min - tolerance) | (
            candidate > local_max + tolerance
        )
        unsupported[np.flatnonzero(clough_finite_inside)] = unsupported_values
    clough_exact = np.asarray(clough_interpolator(coordinates), dtype=float)
    linear_exact = np.asarray(linear_interpolator(coordinates), dtype=float)
    clough_exact_residual = float(np.max(np.abs(clough_exact - log_limit)))
    linear_exact_residual = float(np.max(np.abs(linear_exact - log_limit)))
    clough_ready = bool(
        clough_exact_residual < 1.0e-8
        and np.sum(clough_finite_inside) == np.sum(inside)
        and not np.any(unsupported)
    )
    linear_ready = bool(
        linear_exact_residual < 1.0e-8
        and np.sum(linear_finite_inside) == np.sum(inside)
    )
    display_values = clough_values if clough_ready else linear_values
    display_method = (
        "coordinate-rescaled CloughTocher2DInterpolator on log10(sigma95)"
        if clough_ready
        else "coordinate-rescaled LinearNDInterpolator on log10(sigma95)"
    )

    def masked(values: np.ndarray) -> np.ma.MaskedArray:
        result = np.ma.masked_invalid(values.reshape(xx.shape))
        result.mask = np.ma.getmaskarray(result) | (~inside.reshape(xx.shape))
        return result

    audit = {
        "requested_method": (
            "coordinate-rescaled CloughTocher2DInterpolator on log10(sigma95)"
        ),
        "display_method": display_method,
        "fallback_applied": not clough_ready,
        "fallback_reason": (
            None
            if clough_ready
            else "Clough-Tocher produced values unsupported by enclosing physical points"
        ),
        "grid_size": grid_size,
        "physical_points": len(rows),
        "inside_grid_nodes": int(np.sum(inside)),
        "clough_tocher": {
            "exact_point_max_abs_log10_residual": clough_exact_residual,
            "finite_inside_grid_nodes": int(np.sum(clough_finite_inside)),
            "unsupported_overshoot_nodes": int(np.sum(unsupported)),
            "accepted_for_display": clough_ready,
        },
        "linear": {
            "exact_point_max_abs_log10_residual": linear_exact_residual,
            "finite_inside_grid_nodes": int(np.sum(linear_finite_inside)),
            "accepted_for_display": linear_ready,
        },
        "paper_ready": clough_ready or linear_ready,
    }
    return xx, yy, masked(display_values), masked(clough_values), audit


def _plot_cascade_limit(
    output_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> tuple[list[str], dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    xx, yy, log_limits, clough_log_limits, audit = _cascade_interpolation(rows)
    point_m2 = np.asarray([float(item["M2_GeV"]) for item in rows], dtype=float)
    point_m3 = np.asarray([float(item["M3_GeV"]) for item in rows], dtype=float)
    point_limits = np.asarray([float(item["sigma95_fb"]) for item in rows], dtype=float)
    outputs: list[str] = []

    def draw_map(
        interpolated: np.ma.MaskedArray,
        stem: str,
        title: str,
    ) -> None:
        values = np.ma.power(10.0, interpolated)
        fig, ax = plt.subplots(figsize=(9.2, 7.0))
        image = ax.pcolormesh(
            xx,
            yy,
            values,
            shading="auto",
            cmap="viridis_r",
            norm=LogNorm(
                vmin=float(np.min(point_limits)), vmax=float(np.max(point_limits))
            ),
            rasterized=True,
        )
        ax.scatter(
            point_m2,
            point_m3,
            s=9,
            facecolors="none",
            edgecolors="black",
            linewidths=0.35,
            alpha=0.75,
            label="Generated mass points",
        )
        boundary_x = np.linspace(np.min(point_m2), np.max(point_m2), 500)
        boundary_y = 2.0 * boundary_x
        boundary_mask = (boundary_y >= np.min(point_m3)) & (
            boundary_y <= np.max(point_m3)
        )
        ax.plot(
            boundary_x[boundary_mask],
            boundary_y[boundary_mask],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=r"$M_3=2M_2$",
        )
        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label(
            r"Expected 95% upper limit on $\sigma_{\rm cas}$ [fb]", fontsize=14
        )
        colorbar.ax.tick_params(labelsize=12)
        ax.set_xlabel(r"$M_2$ [GeV]", fontsize=15)
        ax.set_ylabel(r"$M_3$ [GeV]", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.legend(frameon=False, fontsize=11, loc="upper left")
        ax.text(
            0.03,
            0.30,
            rf"$\sqrt{{s}}={COLLIDER_ENERGY_TEV:g}$ TeV, "
            rf"${LUMINOSITY_FB / 1000.0:g}$ ab$^{{-1}}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            color="white",
        )
        ax.set_title(title, fontsize=16)
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=240 if suffix == "png" else None)
            outputs.append(str(path))
        plt.close(fig)

    draw_map(
        log_limits,
        "cascade_expected_cross_section_limit",
        "Cascade resonant four-Higgs production",
    )
    if audit["fallback_applied"]:
        draw_map(
            clough_log_limits,
            "cascade_clough_tocher_diagnostic",
            "Clough--Tocher interpolation diagnostic (rejected)",
        )
    return outputs, audit


def _plot_binning_comparison(
    output_dir: Path,
    topology: str,
    point_results: Sequence[Mapping[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x: list[float] = []
    ratio: list[float] = []
    for point in point_results:
        four = point["schemes"]["background_quantile_4bin"]
        five = point["schemes"]["background_quantile_5bin"]
        if four.get("status") != "ok" or five.get("status") != "ok":
            continue
        coordinate = (
            float(point["point"]["MS_GeV"])
            if topology == "direct"
            else float(point["point"]["M3_GeV"])
        )
        x.append(coordinate)
        ratio.append(float(four["test_sigma95_fb"]) / float(five["test_sigma95_fb"]))
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    ax.scatter(x, ratio, s=18, color="#4477AA", alpha=0.75)
    ax.axhline(1.0, color="black", linewidth=1.2)
    ax.axhline(1.02, color="#C51B29", linewidth=1.2, linestyle="--")
    ax.set_xlabel(r"$M_S$ [GeV]" if topology == "direct" else r"$M_3$ [GeV]", fontsize=14)
    ax.set_ylabel(r"$\sigma_{95}^{\rm 4bin}/\sigma_{95}^{\rm 5bin}$", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    outputs: list[str] = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"{topology}_binning_comparison.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("xgboost", "scipy", "matplotlib", "uproot"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "not installed"
    return versions


def _write_method_readme(
    output_dir: Path,
    topology: str,
    selected_scheme: str,
    paper_ready: bool,
) -> None:
    text = f"""# Resonant four-Higgs score-fit projection

This directory contains the {topology} result from `{METHOD_VERSION}`.  One
mass-conditioned XGBoost classifier is trained with five source-event-grouped
cross-fit folds.  The resolved AK4, mixed AK4/AK8 and boosted AK8 hypotheses
enter one classifier score.  Genuine double-b AK8 tags use epsilon_b squared
({EPS_BB:g}); false double tags use {FAKE_BB:g}.

At each generated mass point, background-weighted score quantiles are fixed on
the validation fold and applied to the disjoint held-out fold.  The globally
selected scheme is `{selected_scheme}`.  Every fitted score-bin count enters
the statistics-only Poisson likelihood.  The tabulated limit solves
q(sigma_95)={Q95:.12g} for a signal template normalized to 1 fb before the four
Higgs decays.  SM hhhh, hhhbb and hh+4b production are included in the Asimov
event counts together with the conventional backgrounds.

No pyhf model, nuisance parameter, expected band, MC-statistical term or
interpolated template is used.  Cascade interpolation is visualization only;
all limits are calculated at generated points first.

Paper-ready status: **{str(paper_ready).lower()}**.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _parallel_input_records(
    samples: Sequence[resolved.LoadedSample], jobs: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as executor:
        futures = {
            executor.submit(_feature_input_record, sample.spec): sample.spec.sample_id
            for sample in samples
        }
        for future in as_completed(futures):
            records.append(future.result())
    return sorted(records, key=lambda item: str(item["sample_id"]))


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, choices=("direct", "cascade"))
    parser.add_argument("--analysis-root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--load-jobs", type=int, default=48)
    parser.add_argument("--model-jobs", type=int, default=5)
    parser.add_argument("--xgboost-threads", type=int, default=36)
    parser.add_argument("--point-jobs", type=int, default=10)
    parser.add_argument("--prediction-threads", type=int, default=18)
    parser.add_argument("--thread-budget", type=int, default=180)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (
        "load_jobs",
        "model_jobs",
        "xgboost_threads",
        "point_jobs",
        "prediction_threads",
        "thread_budget",
    ):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.model_jobs > N_FOLDS:
        raise SystemExit(f"--model-jobs cannot exceed {N_FOLDS}")
    if args.model_jobs * args.xgboost_threads > args.thread_budget:
        raise SystemExit("model-jobs times xgboost-threads exceeds the thread budget")
    if args.point_jobs * args.prediction_threads > args.thread_budget:
        raise SystemExit("point-jobs times prediction-threads exceeds the thread budget")

    start = time.perf_counter()
    timings: dict[str, float] = {}
    analysis_root = args.analysis_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_manifest = (
        analysis_root / "HerwigSignalPoints/mass_scan_10k_ak8-v1/manifest.csv"
    ).resolve()
    background_manifest = (
        analysis_root
        / "ResonanceAnalysis/background_manifest_ak8-v1-full14_scorefit-v3.csv"
    ).resolve()
    for path in (signal_manifest, background_manifest):
        if not path.is_file():
            raise ScoreFitError(f"missing immutable input manifest {path}")
    normalization = background_normalization_provenance(background_manifest)
    if normalization.get("status") != "accepted":
        raise ScoreFitError(
            "the 14-sample background manifest lacks a matching accepted normalization audit: "
            + str(normalization.get("reason"))
        )

    stage = time.perf_counter()
    signals, backgrounds, points, missing = _load_samples(
        args.topology,
        analysis_root,
        signal_manifest,
        background_manifest,
        args.load_jobs,
    )
    timings["load_feature_tables_seconds"] = time.perf_counter() - stage
    if missing:
        raise ScoreFitError("optional feature inputs are not allowed in the 14-sample analysis")

    stage = time.perf_counter()
    input_records = _parallel_input_records([*signals, *backgrounds], args.load_jobs)
    source_files = [
        Path(__file__).resolve(),
        Path(resolved.__file__).resolve(),
        Path(fat.__file__).resolve(),
        Path(poisson_asimov_deviance.__code__.co_filename).resolve(),
    ]
    source_hashes = {str(path): _sha256_file(path) for path in sorted(set(source_files))}
    input_contract = {
        "method_version": METHOD_VERSION,
        "topology": args.topology,
        "seed": SEED,
        "feature_set": FEATURE_SET,
        "signal_manifest": str(signal_manifest),
        "signal_manifest_sha256": _sha256_file(signal_manifest),
        "background_manifest": str(background_manifest),
        "background_manifest_sha256": _sha256_file(background_manifest),
        "normalization_audit": normalization,
        "signal_reference_cross_section_fb": SIGNAL_REFERENCE_XSEC_FB,
        "collider_energy_TeV": COLLIDER_ENERGY_TEV,
        "luminosity_fb": LUMINOSITY_FB,
        "hbb_branching_ratio": HBB_BRANCHING_RATIO,
        "tagging": {
            "eps_b": EPS_B,
            "eps_c": EPS_C,
            "eps_light": EPS_LIGHT,
            "eps_bb": EPS_BB,
            "fake_bb": FAKE_BB,
        },
        "input_features": input_records,
        "source_hashes": source_hashes,
    }
    input_fingerprint = _fingerprint(input_contract)
    _atomic_json(output_dir / "input_hashes.json", input_contract)
    timings["hash_inputs_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    models, feature_names, model_hashes, models_resumed = load_or_train_models(
        args.topology,
        signals,
        backgrounds,
        points,
        output_dir / "models",
        input_fingerprint,
        model_jobs=args.model_jobs,
        xgboost_threads=args.xgboost_threads,
    )
    timings["load_or_train_models_seconds"] = time.perf_counter() - stage
    for model in models:
        model.set_params(n_jobs=int(args.prediction_threads))
    core_fingerprint = _fingerprint(
        {
            "input_fingerprint": input_fingerprint,
            "model_hashes": model_hashes,
            "binning_schemes": BINNING_SCHEMES,
            "minimum_background_source_events": MIN_BACKGROUND_SOURCE_EVENTS,
            "minimum_background_neff": MIN_BACKGROUND_NEFF,
            "q95": Q95,
        }
    )

    signal_by_point = {sample.spec.point.point_id: sample for sample in signals}
    stage = time.perf_counter()
    point_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.point_jobs) as executor:
        futures = {
            executor.submit(
                analyze_point,
                point,
                signal_by_point[point.point_id],
                backgrounds,
                models,
                output_dir / "scores",
                output_dir / "points",
                core_fingerprint,
            ): point
            for point in points
        }
        completed = 0
        for future in as_completed(futures):
            point_results.append(future.result())
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(points):
                print(f"[{args.topology}] completed {completed}/{len(points)} mass points", flush=True)
    timings["score_and_limit_points_seconds"] = time.perf_counter() - stage
    point_results = sorted(
        point_results,
        key=lambda item: signal_by_point[item["point"]["point_id"]].spec.point.sort_key,
    )

    selected_scheme, selection_audit = select_binning_scheme(point_results)
    both_schemes_complete = all(
        all(point["schemes"][name].get("status") == "ok" for name in BINNING_SCHEMES)
        for point in point_results
    )
    selected_complete = all(
        point["schemes"][selected_scheme].get("status") == "ok"
        for point in point_results
    )
    if not selected_complete:
        raise ScoreFitError("the selected score binning is invalid at one or more mass points")

    limit_rows = _point_limit_rows(point_results, selected_scheme)
    _write_csv(output_dir / "pointwise_limits.csv", limit_rows)
    _atomic_json(
        output_dir / "pointwise_limits.json",
        {
            "method_version": METHOD_VERSION,
            "selected_scheme": selected_scheme,
            "rows": limit_rows,
        },
    )
    _atomic_json(
        output_dir / "pointwise_score_templates.json",
        {
            "method_version": METHOD_VERSION,
            "selected_scheme": selected_scheme,
            "points": point_results,
        },
    )
    _write_selected_template_table(output_dir, point_results, selected_scheme)
    _atomic_json(
        output_dir / "binning_audit.json",
        {
            "selection": selection_audit,
            "points": [
                {
                    "point": point["point"],
                    "schemes": {
                        name: {
                            key: point["schemes"][name].get(key)
                            for key in (
                                "status",
                                "reason",
                                "requested_quantiles",
                                "initial_edges_by_fold",
                                "final_edges_by_fold",
                                "merges",
                                "final_bin_count",
                                "validation_sigma95_fb",
                                "test_sigma95_fb",
                                "test_occupancy_failures",
                            )
                        }
                        for name in BINNING_SCHEMES
                    },
                }
                for point in point_results
            ],
        },
    )
    yield_rows = _write_yield_table(
        output_dir, args.topology, point_results, selected_scheme, backgrounds
    )
    category_rows = _category_diagnostics(output_dir, args.topology, signals)

    stage = time.perf_counter()
    plot_outputs = _plot_binning_comparison(output_dir, args.topology, point_results)
    plot_outputs.extend(
        _plot_category_diagnostics(output_dir, args.topology, category_rows)
    )
    interpolation_audit: dict[str, Any] = {
        "applied": False,
        "paper_ready": True,
        "reason": "direct limits are joined only through physical mass values",
    }
    if args.topology == "direct":
        plot_outputs.extend(_plot_direct_limit(output_dir, limit_rows))
    else:
        cascade_outputs, interpolation_audit = _plot_cascade_limit(
            output_dir, limit_rows
        )
        interpolation_audit["applied"] = True
        plot_outputs.extend(cascade_outputs)
    _atomic_json(output_dir / "interpolation_audit.json", interpolation_audit)
    timings["tables_and_plots_seconds"] = time.perf_counter() - stage

    reasons: list[str] = []
    if len(point_results) != EXPECTED_SIGNAL_POINTS[args.topology]:
        reasons.append("physical point count is incomplete")
    if not both_schemes_complete:
        reasons.append("a validation or held-out occupancy gate failed")
    if not interpolation_audit.get("paper_ready", False):
        reasons.append("cascade interpolation has unsupported structure")
    if any(
        not math.isfinite(float(row["sigma95_fb"])) or float(row["sigma95_fb"]) < 0.0
        for row in limit_rows
    ):
        reasons.append("a pointwise limit is non-finite or negative")
    paper_ready = not reasons
    timings["total_seconds"] = time.perf_counter() - start
    versions = _package_versions()
    _atomic_json(output_dir / "timings.json", timings)
    _atomic_json(output_dir / "package_versions.json", versions)
    _write_method_readme(output_dir, args.topology, selected_scheme, paper_ready)
    method_manifest = {
        "method_version": METHOD_VERSION,
        "status": "complete",
        "paper_ready": paper_ready,
        "paper_ready_failures": reasons,
        "topology": args.topology,
        "command": shlex.join(sys.argv),
        "analysis_root": str(analysis_root),
        "output_dir": str(output_dir),
        "input_fingerprint": input_fingerprint,
        "core_fingerprint": core_fingerprint,
        "signal_points": len(point_results),
        "expected_signal_points": EXPECTED_SIGNAL_POINTS[args.topology],
        "background_samples": len(backgrounds),
        "sm_multi_higgs_roles": list(SM_ROLES),
        "normalization_provenance": normalization,
        "five_fold_split": "test f, validation f+1, remaining three folds train",
        "mass_conditioned_classifier": True,
        "equal_total_signal_training_weight_per_mass_point": True,
        "background_mass_replicas": BACKGROUND_REPLICAS,
        "fixed_xgboost_parameters": resolved.FIXED_XGBOOST_PARAMS,
        "feature_names": feature_names,
        "model_sha256": model_hashes,
        "models_resumed": models_resumed,
        "selected_binning": selected_scheme,
        "binning_selection": selection_audit,
        "score_bins_are_ordinal_across_folds": True,
        "minimum_background_source_events_per_bin": MIN_BACKGROUND_SOURCE_EVENTS,
        "minimum_background_neff_per_bin": MIN_BACKGROUND_NEFF,
        "signal_reference_cross_section_fb_before_decays": SIGNAL_REFERENCE_XSEC_FB,
        "collider_energy_TeV": COLLIDER_ENERGY_TEV,
        "asimov_components": ["conventional backgrounds", *SM_ROLES],
        "statistic": "direct binned Poisson Asimov likelihood ratio",
        "q95": Q95,
        "fractional_asimov_counts": True,
        "pyhf_used": False,
        "nuisance_parameters": 0,
        "mc_statistical_terms": False,
        "expected_bands": False,
        "template_interpolation": False,
        "cascade_display_interpolation": interpolation_audit,
        "tagging": input_contract["tagging"],
        "parallelism": {
            "load_jobs": args.load_jobs,
            "model_jobs": args.model_jobs,
            "xgboost_threads": args.xgboost_threads,
            "point_jobs": args.point_jobs,
            "prediction_threads": args.prediction_threads,
            "thread_budget": args.thread_budget,
        },
        "plot_outputs": plot_outputs,
        "yield_table_rows": len(yield_rows),
        "category_diagnostic_rows": len(category_rows),
        "timings": timings,
        "package_versions": versions,
    }
    _atomic_json(output_dir / "method_manifest.json", method_manifest)
    print(
        f"[{args.topology}] selected {selected_scheme}; "
        f"paper_ready={str(paper_ready).lower()}; output={output_dir}",
        flush=True,
    )
    return 0 if paper_ready else 3


if __name__ == "__main__":
    raise SystemExit(main())
