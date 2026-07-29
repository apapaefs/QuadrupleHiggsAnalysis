"""Reconstruct and plot the SM score templates from a completed fast-sm study.

The final pyhf workspaces in ``shape_results.json`` are authoritative: they
contain the exact signal and background bin contents used for the published
limit.  This module extracts those templates and, unless ``--skip-rescore`` is
requested, independently reloads the recorded ROOT inputs and five saved
XGBoost models.  Every event is then scored by the model for its held-out test
fold.  The independently reconstructed binned yields and MC-statistical
uncertainties must close to the saved workspace before any report is written.

The report is deliberately a post-processing step.  It never trains XGBoost,
chooses score-bin boundaries, or reruns pyhf.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from c3d4_xgboost_runner import (
    _load_samples,
    _profile_indices,
    _score_partition,
)
from c3d4_xgboost_study import binned_weight_summary, build_pyhf_channel
from observable_schemas import get_feature_contract, validate_model_contract


REPORT_VERSION = "fast-sm-sm-score-template-report-v1"
DEFAULT_STRATEGY = "sm-crossfit-v2"
DEFAULT_DISPLAY_BINS = 40
_FOLD_PATTERN = re.compile(r"(\d+)$")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Could not read JSON from {path}: {error}") from error


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _finite_array(values: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or np.any(~np.isfinite(result)):
        raise ValueError(f"{label} must be a finite one-dimensional array")
    return result


def _fold_number(name: str) -> int:
    match = _FOLD_PATTERN.search(str(name))
    if match is None:
        raise ValueError(f"Cannot determine the fold number from channel {name!r}")
    return int(match.group(1))


def _find_sm_shape_row(shape_results: Any) -> dict[str, Any]:
    if not isinstance(shape_results, list):
        raise ValueError("shape_results.json must contain a list of point results")
    matches = [
        dict(row)
        for row in shape_results
        if isinstance(row, Mapping)
        and math.isclose(float(row.get("c3", math.nan)), 0.0, abs_tol=1.0e-12)
        and math.isclose(float(row.get("d4", math.nan)), 0.0, abs_tol=1.0e-12)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one c3=d4=0 row in shape_results.json, "
            f"found {len(matches)}"
        )
    row = matches[0]
    if row.get("status") != "ok":
        raise ValueError(
            f"The SM shape result is not usable: status={row.get('status')!r}"
        )
    if row.get("pyhf_shape_with_mcstat", {}).get("status") != "ok":
        raise ValueError("The SM pyhf shape result with MC statistics is not usable")
    return row


def _staterror(sample: Mapping[str, Any]) -> np.ndarray:
    modifiers = [
        modifier
        for modifier in sample.get("modifiers", [])
        if isinstance(modifier, Mapping) and modifier.get("type") == "staterror"
    ]
    if len(modifiers) != 1:
        raise ValueError(
            f"Sample {sample.get('name')!r} must have exactly one staterror modifier"
        )
    return _finite_array(modifiers[0].get("data", []), "staterror data")


def extract_workspace_templates(sm_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract the exact nominal templates and staterrors used by pyhf."""

    fit = sm_row.get("pyhf_shape_with_mcstat")
    workspace = fit.get("workspace_spec") if isinstance(fit, Mapping) else None
    if not isinstance(workspace, Mapping):
        raise ValueError(
            "The SM result does not contain a pyhf workspace specification"
        )

    fold_edges = sm_row.get("fold_bin_edges")
    if not isinstance(fold_edges, list):
        raise ValueError("The SM result does not contain per-fold score-bin edges")

    observations = {
        str(item["name"]): _finite_array(item["data"], "observation data")
        for item in workspace.get("observations", [])
    }
    channels: list[dict[str, Any]] = []
    for channel_spec in workspace.get("channels", []):
        name = str(channel_spec.get("name", ""))
        fold = _fold_number(name)
        if fold >= len(fold_edges):
            raise ValueError(f"{name}: no corresponding score-bin edges")
        samples = {
            str(sample.get("name")): sample
            for sample in channel_spec.get("samples", [])
            if isinstance(sample, Mapping)
        }
        if set(samples) != {"signal", "background"}:
            raise ValueError(
                f"{name}: expected exactly signal and background samples, got "
                f"{sorted(samples)}"
            )
        signal = _finite_array(samples["signal"].get("data", []), f"{name} signal")
        background = _finite_array(
            samples["background"].get("data", []), f"{name} background"
        )
        signal_staterror = _staterror(samples["signal"])
        background_staterror = _staterror(samples["background"])
        edges = _finite_array(fold_edges[fold], f"{name} edges")
        expected_shape = (len(edges) - 1,)
        for label, values in (
            ("signal", signal),
            ("background", background),
            ("signal staterror", signal_staterror),
            ("background staterror", background_staterror),
        ):
            if values.shape != expected_shape:
                raise ValueError(
                    f"{name} {label} has shape {values.shape}, "
                    f"expected {expected_shape}"
                )
        if name not in observations:
            raise ValueError(f"{name}: missing workspace observation")
        if not np.allclose(observations[name], background, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"{name}: the saved observation is not the background-only Asimov data"
            )
        channels.append(
            {
                "name": name,
                "fold": fold,
                "edges": edges,
                "signal": signal,
                "signal_staterror": signal_staterror,
                "background": background,
                "background_staterror": background_staterror,
                "observation": observations[name],
            }
        )

    channels.sort(key=lambda channel: int(channel["fold"]))
    expected_folds = list(range(len(fold_edges)))
    if [channel["fold"] for channel in channels] != expected_folds:
        raise ValueError(
            "Workspace channels do not form the expected consecutive fold sequence"
        )
    return channels


def _manifest_specs(
    manifest: Mapping[str, Any],
    *,
    kind: str,
    repo_root: Path,
    sm_point_only: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for item in manifest.get("inputs", []):
        if not isinstance(item, Mapping) or item.get("kind") != kind:
            continue
        if sm_point_only and not (
            item.get("c3") is not None
            and item.get("d4") is not None
            and math.isclose(float(item["c3"]), 0.0, abs_tol=1.0e-12)
            and math.isclose(float(item["d4"]), 0.0, abs_tol=1.0e-12)
        ):
            continue
        path = Path(str(item.get("path", ""))).expanduser()
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        record = dict(item)
        record["resolved_path"] = str(path)
        specs.append(
            {
                "path": path,
                "xsec_fb": item.get("xsec_fb"),
                "rate_factor": item.get("rate_factor", 1.0),
                "normalisation_weight": item.get("normalisation_weight"),
                "generated_events": item.get("generated_events"),
                "metadata": dict(item.get("metadata") or {}),
            }
        )
        records.append(record)
    return specs, records


def _verify_input_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for item in records:
        path = Path(str(item["resolved_path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Recorded input does not exist: {path}")
        expected = str(item.get("sha256") or "")
        if not expected:
            raise ValueError(f"{path}: the study manifest has no SHA-256 digest")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"{path}: SHA-256 mismatch (manifest {expected}, current {actual})"
            )
        verified.append(
            {
                "sample_id": item.get("sample_id"),
                "kind": item.get("kind"),
                "path": str(path),
                "sha256": actual,
                "verified": True,
                "entries": item.get("entries"),
                "sum_raw_weight": item.get("sum_raw_weight"),
                "sum_physical_weight": item.get("sum_physical_weight"),
                "xsec_fb": item.get("xsec_fb"),
                "rate_factor": item.get("rate_factor"),
                "normalisation_weight": item.get("normalisation_weight"),
                "normalisation_source": item.get("normalisation_source"),
                "generated_events": item.get("generated_events"),
                "c3": item.get("c3"),
                "d4": item.get("d4"),
            }
        )
    return verified


def _load_models(
    strategy_dir: Path,
    *,
    n_folds: int,
    observable_set: str,
    profile: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    import xgboost as xgb

    models: list[Any] = []
    records: list[dict[str, Any]] = []
    for fold in range(n_folds):
        path = strategy_dir / "models" / f"fold_{fold}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved fold model: {path}")
        model = xgb.XGBClassifier(n_jobs=1)
        model.load_model(str(path))
        metadata = validate_model_contract(model, observable_set, profile)
        models.append(model)
        records.append(
            {
                "fold": fold,
                "path": str(path),
                "sha256": _sha256(path),
                "model_metadata": metadata,
            }
        )
    return models, records


def _event_rows(
    partition: Mapping[str, Mapping[str, Any]],
    *,
    fold: int,
    role: str,
    component: str,
    hhhh_xsec_fb: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, item in partition.items():
        sample = item["sample"]
        scores = _finite_array(item["scores"], f"{sample_id} scores")
        raw = _finite_array(item["raw_weights"], f"{sample_id} raw weights")
        unit_xsec = _finite_array(
            item["unit_xsec_weights"], f"{sample_id} unit-cross-section weights"
        )
        physical = _finite_array(
            item["physical_weights"], f"{sample_id} physical weights"
        )
        if role == "signal" and component == "hhhh":
            template = unit_xsec
        elif role == "signal" and component == "hhhbb":
            template = physical / float(hhhh_xsec_fb)
        else:
            template = physical
        mask = np.asarray(item["mask"], dtype=bool)
        source_entries = np.asarray(sample.source_entry_indices[mask], dtype=np.int64)
        event_indices = np.asarray(item["event_indices"], dtype=np.int64)
        if not (
            scores.shape
            == raw.shape
            == unit_xsec.shape
            == physical.shape
            == template.shape
            == event_indices.shape
            == source_entries.shape
        ):
            raise ValueError(
                f"{sample_id}: scored event arrays do not have matching shapes"
            )
        for index in range(len(scores)):
            rows.append(
                {
                    "role": role,
                    "component": component,
                    "sample_id": str(sample_id),
                    "test_fold": int(fold),
                    "event_index": int(event_indices[index]),
                    "source_entry_index": int(source_entries[index]),
                    "xgboost_score": float(scores[index]),
                    "raw_weight": float(raw[index]),
                    "unit_xsec_weight": float(unit_xsec[index]),
                    "physical_weight": float(physical[index]),
                    "template_weight_per_equivalent_hhhh_fb": float(template[index]),
                }
            )
    return rows


def _partition_arrays(
    partition: Mapping[str, Mapping[str, Any]], key: str
) -> np.ndarray:
    arrays = [np.asarray(item[key], dtype=float) for item in partition.values()]
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=float)


def reconstruct_oof_scores(
    *,
    models: Sequence[Any],
    hhhh_samples: Sequence[Any],
    hhhbb_samples: Sequence[Any],
    background_samples: Sequence[Any],
    n_folds: int,
    profile_indices: np.ndarray,
    hhhh_xsec_fb: float,
) -> dict[str, Any]:
    """Score every loaded event once and return fold-wise arrays."""

    if len(models) != n_folds:
        raise ValueError(f"Expected {n_folds} models, got {len(models)}")
    if len(hhhh_samples) != 1:
        raise ValueError("Exactly one SM grid hhhh sample is required")
    if len(hhhbb_samples) > 1:
        raise ValueError("At most one SM hhhbb sample is supported")

    folds: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for fold, model in enumerate(models):
        common = {
            "rotation": fold,
            "split": "test",
            "n_folds": n_folds,
            "profile_indices": profile_indices,
            "scale_validation_to_full": False,
            "parameterized": False,
        }
        hhhh = _score_partition(model, hhhh_samples, **common)
        hhhbb = _score_partition(model, hhhbb_samples, **common)
        background = _score_partition(model, background_samples, **common)

        hhhh_scores = _partition_arrays(hhhh, "scores")
        hhhh_unit = _partition_arrays(hhhh, "unit_xsec_weights")
        hhhh_physical = _partition_arrays(hhhh, "physical_weights")
        hhhbb_scores = _partition_arrays(hhhbb, "scores")
        hhhbb_physical = _partition_arrays(hhhbb, "physical_weights")
        background_scores = _partition_arrays(background, "scores")
        background_weights = _partition_arrays(background, "physical_weights")
        signal_scores = np.concatenate((hhhh_scores, hhhbb_scores))
        signal_template_weights = np.concatenate(
            (hhhh_unit, hhhbb_physical / float(hhhh_xsec_fb))
        )
        signal_physical_weights = np.concatenate((hhhh_physical, hhhbb_physical))
        folds.append(
            {
                "fold": fold,
                "hhhh_scores": hhhh_scores,
                "hhhh_unit_weights": hhhh_unit,
                "hhhh_physical_weights": hhhh_physical,
                "hhhbb_scores": hhhbb_scores,
                "hhhbb_physical_weights": hhhbb_physical,
                "signal_scores": signal_scores,
                "signal_template_weights": signal_template_weights,
                "signal_physical_weights": signal_physical_weights,
                "background_scores": background_scores,
                "background_weights": background_weights,
                "background_partitions": background,
            }
        )
        rows.extend(
            _event_rows(
                hhhh,
                fold=fold,
                role="signal",
                component="hhhh",
                hhhh_xsec_fb=hhhh_xsec_fb,
            )
        )
        rows.extend(
            _event_rows(
                hhhbb,
                fold=fold,
                role="signal",
                component="hhhbb",
                hhhh_xsec_fb=hhhh_xsec_fb,
            )
        )
        rows.extend(
            _event_rows(
                background,
                fold=fold,
                role="background",
                component="",
                hhhh_xsec_fb=hhhh_xsec_fb,
            )
        )

    expected_rows = sum(
        int(sample.entries)
        for sample in [*hhhh_samples, *hhhbb_samples, *background_samples]
    )
    if len(rows) != expected_rows:
        raise AssertionError(
            "Out-of-fold scoring is not exactly-once: "
            f"wrote {len(rows)} rows for {expected_rows} loaded events"
        )
    identities = [
        (row["sample_id"], int(row["source_entry_index"])) for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise AssertionError("Out-of-fold score table contains duplicate source events")
    return {"folds": folds, "event_rows": rows}


def validate_template_closure(
    folds: Sequence[Mapping[str, Any]],
    workspace_channels: Sequence[Mapping[str, Any]],
    *,
    rtol: float = 2.0e-10,
    atol: float = 2.0e-12,
) -> dict[str, Any]:
    """Require reconstructed score templates to match the stored workspace."""

    if len(folds) != len(workspace_channels):
        raise ValueError("Reconstructed folds and workspace channels do not match")
    comparisons: list[dict[str, Any]] = []
    largest_absolute = 0.0
    largest_relative = 0.0
    for reconstructed, saved in zip(folds, workspace_channels):
        if int(reconstructed["fold"]) != int(saved["fold"]):
            raise ValueError("Reconstructed and saved fold order does not match")
        channel = build_pyhf_channel(
            str(saved["name"]),
            reconstructed["signal_scores"],
            reconstructed["signal_template_weights"],
            reconstructed["background_scores"],
            reconstructed["background_weights"],
            saved["edges"],
        )
        fields = (
            ("signal", channel["signal"], saved["signal"]),
            (
                "signal_staterror",
                channel["signal_staterror"],
                saved["signal_staterror"],
            ),
            ("background", channel["background"], saved["background"]),
            (
                "background_staterror",
                channel["background_staterror"],
                saved["background_staterror"],
            ),
        )
        fold_result: dict[str, Any] = {
            "fold": int(saved["fold"]),
            "channel": str(saved["name"]),
            "fields": {},
        }
        for label, actual, expected in fields:
            actual = np.asarray(actual, dtype=float)
            expected = np.asarray(expected, dtype=float)
            difference = np.abs(actual - expected)
            relative = np.divide(
                difference,
                np.abs(expected),
                out=np.zeros_like(difference),
                where=np.abs(expected) > 0.0,
            )
            field_absolute = float(np.max(difference)) if len(difference) else 0.0
            field_relative = float(np.max(relative)) if len(relative) else 0.0
            largest_absolute = max(largest_absolute, field_absolute)
            largest_relative = max(largest_relative, field_relative)
            if not np.allclose(actual, expected, rtol=rtol, atol=atol):
                raise AssertionError(
                    f"{saved['name']} {label} does not close to the saved pyhf "
                    f"workspace (max absolute difference {field_absolute:.6g}, "
                    f"max relative difference {field_relative:.6g})"
                )
            fold_result["fields"][label] = {
                "max_absolute_difference": field_absolute,
                "max_relative_difference": field_relative,
            }
        comparisons.append(fold_result)
    return {
        "status": "passed",
        "relative_tolerance": float(rtol),
        "absolute_tolerance": float(atol),
        "max_absolute_difference": largest_absolute,
        "max_relative_difference": largest_relative,
        "folds": comparisons,
    }


def _sum_histograms(
    summaries: Sequence[Mapping[str, Any]], edges: np.ndarray
) -> dict[str, Any]:
    if not summaries:
        zeros = np.zeros(len(edges) - 1, dtype=float)
        return {
            "edges": edges,
            "yield": zeros,
            "sumw2": zeros.copy(),
            "uncertainty": zeros.copy(),
            "raw_entries": np.zeros(len(edges) - 1, dtype=int),
            "effective_entries": zeros.copy(),
        }
    yields = np.sum(
        [np.asarray(item["yield"], dtype=float) for item in summaries], axis=0
    )
    sumw2 = np.sum(
        [np.asarray(item["sumw2"], dtype=float) for item in summaries], axis=0
    )
    raw = np.sum(
        [np.asarray(item["raw_entries"], dtype=int) for item in summaries], axis=0
    )
    neff = np.divide(
        np.square(yields), sumw2, out=np.zeros_like(yields), where=sumw2 > 0.0
    )
    return {
        "edges": edges,
        "yield": yields,
        "sumw2": sumw2,
        "uncertainty": np.sqrt(sumw2),
        "raw_entries": raw,
        "effective_entries": neff,
    }


def make_display_histograms(
    folds: Sequence[Mapping[str, Any]], *, n_bins: int
) -> dict[str, Any]:
    if int(n_bins) < 2:
        raise ValueError("The display histogram requires at least two bins")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    signal_template = []
    signal_physical = []
    hhhh_physical = []
    hhhbb_physical = []
    background = []
    for fold in folds:
        signal_template.append(
            binned_weight_summary(
                fold["signal_scores"], fold["signal_template_weights"], edges
            )
        )
        signal_physical.append(
            binned_weight_summary(
                fold["signal_scores"], fold["signal_physical_weights"], edges
            )
        )
        hhhh_physical.append(
            binned_weight_summary(
                fold["hhhh_scores"], fold["hhhh_physical_weights"], edges
            )
        )
        hhhbb_physical.append(
            binned_weight_summary(
                fold["hhhbb_scores"], fold["hhhbb_physical_weights"], edges
            )
        )
        background.append(
            binned_weight_summary(
                fold["background_scores"], fold["background_weights"], edges
            )
        )
    return {
        "edges": edges,
        "signal_template_per_equivalent_hhhh_fb": _sum_histograms(
            signal_template, edges
        ),
        "signal_sm_physical": _sum_histograms(signal_physical, edges),
        "hhhh_sm_physical": _sum_histograms(hhhh_physical, edges),
        "hhhbb_sm_physical": _sum_histograms(hhhbb_physical, edges),
        "background": _sum_histograms(background, edges),
    }


def _write_event_scores(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "role",
        "component",
        "sample_id",
        "test_fold",
        "event_index",
        "source_entry_index",
        "xgboost_score",
        "raw_weight",
        "unit_xsec_weight",
        "physical_weight",
        "template_weight_per_equivalent_hhhh_fb",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as gzip_handle:
            with io.TextIOWrapper(
                gzip_handle, newline="", encoding="utf-8"
            ) as text_handle:
                writer = csv.DictWriter(text_handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field) for field in fields})


def _write_template_csv(
    path: Path,
    channels: Sequence[Mapping[str, Any]],
    *,
    hhhh_xsec_fb: float,
    sigma95_fb: float,
) -> None:
    fields = [
        "channel",
        "test_fold",
        "score_bin",
        "score_low",
        "score_high",
        "signal_per_equivalent_hhhh_fb",
        "signal_mcstat_per_equivalent_hhhh_fb",
        "signal_at_sm",
        "signal_at_expected_95cl_limit",
        "background",
        "background_mcstat",
        "signal_over_background_per_equivalent_hhhh_fb",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for channel in channels:
            edges = np.asarray(channel["edges"], dtype=float)
            for index in range(len(edges) - 1):
                signal = float(channel["signal"][index])
                background = float(channel["background"][index])
                writer.writerow(
                    {
                        "channel": channel["name"],
                        "test_fold": int(channel["fold"]),
                        "score_bin": index,
                        "score_low": float(edges[index]),
                        "score_high": float(edges[index + 1]),
                        "signal_per_equivalent_hhhh_fb": signal,
                        "signal_mcstat_per_equivalent_hhhh_fb": float(
                            channel["signal_staterror"][index]
                        ),
                        "signal_at_sm": hhhh_xsec_fb * signal,
                        "signal_at_expected_95cl_limit": sigma95_fb * signal,
                        "background": background,
                        "background_mcstat": float(
                            channel["background_staterror"][index]
                        ),
                        "signal_over_background_per_equivalent_hhhh_fb": (
                            signal / background if background > 0.0 else math.nan
                        ),
                    }
                )


def _apply_plot_style() -> None:
    import matplotlib as mpl

    try:
        import mplhep as hep

        hep.style.use("ROOT")
    except ImportError:
        pass
    mpl.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.1,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: Any, stem: Path, *, dpi: int) -> dict[str, str]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fixed_time = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
    fig.savefig(
        pdf,
        metadata={
            "Creator": REPORT_VERSION,
            "Producer": REPORT_VERSION,
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(png, dpi=int(dpi), metadata={"Software": REPORT_VERSION})
    return {"pdf": str(pdf), "png": str(png)}


def plot_oof_score_distribution(
    histograms: Mapping[str, Any],
    output_stem: Path,
    *,
    luminosity_fb_inverse: float,
    hhhh_xsec_fb: float,
    sigma95_fb: float,
    dpi: int,
) -> dict[str, str]:
    """Plot normalized and expected-yield out-of-fold score distributions."""

    import matplotlib.pyplot as plt

    _apply_plot_style()
    edges = np.asarray(histograms["edges"], dtype=float)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    background = histograms["background"]
    signal_template = histograms["signal_template_per_equivalent_hhhh_fb"]
    b = np.asarray(background["yield"], dtype=float)
    b_err = np.asarray(background["uncertainty"], dtype=float)
    s_template = np.asarray(signal_template["yield"], dtype=float)
    s_sm = np.asarray(histograms["signal_sm_physical"]["yield"], dtype=float)
    if not np.allclose(
        s_sm, hhhh_xsec_fb * s_template, rtol=2.0e-10, atol=2.0e-12
    ):
        raise AssertionError(
            "The physical SM signal histogram does not close to the equivalent-"
            "hhhh-fb template multiplied by the recorded SM hhhh cross section"
        )
    s_limit = sigma95_fb * s_template

    b_norm = b / float(np.sum(b))
    s_norm = s_template / float(np.sum(s_template))
    b_norm_err = b_err / float(np.sum(b))

    fig, (ax_shape, ax_yield) = plt.subplots(
        2,
        1,
        figsize=(8.0, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.08},
    )
    blue = "#4477AA"
    red = "#CC6677"
    amber = "#AA7722"

    ax_shape.stairs(
        b_norm / widths,
        edges,
        color=blue,
        linewidth=2.0,
        fill=True,
        alpha=0.24,
        label="All backgrounds",
    )
    ax_shape.fill_between(
        centers,
        np.maximum(0.0, (b_norm - b_norm_err) / widths),
        (b_norm + b_norm_err) / widths,
        step="mid",
        color=blue,
        alpha=0.25,
        linewidth=0.0,
        label="Background MC stat.",
    )
    ax_shape.stairs(
        s_norm / widths,
        edges,
        color=red,
        linewidth=2.2,
        label=r"SM-point signal shape: $hhhh+hhhbb$",
    )
    ax_shape.set_ylabel("Unit-normalized density")
    ax_shape.set_ylim(bottom=0.0)
    ax_shape.legend(loc="upper left", frameon=False, fontsize=10)
    ax_shape.text(
        0.98,
        0.96,
        r"$\sqrt{s}=14$ TeV"
        + "\n"
        + rf"$\mathcal{{L}}={luminosity_fb_inverse / 1000:g}$ ab$^{{-1}}$",
        ha="right",
        va="top",
        transform=ax_shape.transAxes,
    )

    ax_yield.stairs(
        b,
        edges,
        color=blue,
        linewidth=2.0,
        fill=True,
        alpha=0.24,
        label="All backgrounds",
    )
    ax_yield.fill_between(
        centers,
        np.maximum(np.finfo(float).tiny, b - b_err),
        b + b_err,
        step="mid",
        color=blue,
        alpha=0.30,
        linewidth=0.0,
        label="_nolegend_",
    )
    ax_yield.stairs(
        s_sm,
        edges,
        color=red,
        linewidth=2.1,
        label=rf"SM prediction, $\sigma_{{hhhh}}={hhhh_xsec_fb:.3g}$ fb",
    )
    ax_yield.stairs(
        s_limit,
        edges,
        color=amber,
        linewidth=1.9,
        linestyle="--",
        label=rf"Median 95% CL limit, $\sigma_{{hhhh}}={sigma95_fb:.3g}$ fb",
    )
    positive = np.concatenate((b[b > 0.0], s_sm[s_sm > 0.0], s_limit[s_limit > 0.0]))
    if len(positive):
        ax_yield.set_ylim(
            bottom=max(float(np.min(positive)) * 0.35, 1.0e-7),
            top=float(np.max(positive)) * 4.0,
        )
    ax_yield.set_yscale("log")
    ax_yield.set_xlabel("Out-of-fold XGBoost score  (more signal-like $\\rightarrow$)")
    ax_yield.set_ylabel("Expected events / score bin")
    ax_yield.set_xlim(0.0, 1.0)
    ax_yield.legend(loc="upper right", frameon=False, ncol=1, fontsize=9)
    ax_yield.text(
        0.02,
        0.04,
        r"$c_3=d_4=0$; every event is scored once by a model that did not train on it",
        transform=ax_yield.transAxes,
        fontsize=10,
        va="bottom",
    )
    return _save_figure(fig, output_stem, dpi=dpi)


def plot_unrolled_pyhf_templates(
    channels: Sequence[Mapping[str, Any]],
    output_stem: Path,
    *,
    luminosity_fb_inverse: float,
    hhhh_xsec_fb: float,
    sigma95_fb: float,
    dpi: int,
) -> dict[str, str]:
    """Plot the exact saved pyhf bins, unrolled across the five channels."""

    import matplotlib.pyplot as plt

    _apply_plot_style()
    signal = np.concatenate(
        [np.asarray(channel["signal"], dtype=float) for channel in channels]
    )
    signal_error = np.concatenate(
        [np.asarray(channel["signal_staterror"], dtype=float) for channel in channels]
    )
    background = np.concatenate(
        [np.asarray(channel["background"], dtype=float) for channel in channels]
    )
    background_error = np.concatenate(
        [
            np.asarray(channel["background_staterror"], dtype=float)
            for channel in channels
        ]
    )
    n_bins_per_fold = [len(channel["signal"]) for channel in channels]
    boundaries = np.concatenate(([0], np.cumsum(n_bins_per_fold)))
    x = np.arange(len(signal), dtype=float)

    fig, (ax, ratio) = plt.subplots(
        2,
        1,
        figsize=(10.2, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06},
    )
    blue = "#4477AA"
    red = "#CC6677"

    ax.bar(
        x,
        background,
        width=0.84,
        color=blue,
        alpha=0.60,
        edgecolor=blue,
        linewidth=0.8,
        label="Background",
    )
    ax.fill_between(
        x,
        np.maximum(np.finfo(float).tiny, background - background_error),
        background + background_error,
        step="mid",
        color=blue,
        alpha=0.24,
        linewidth=0.0,
        label="Background MC stat.",
    )
    ax.errorbar(
        x,
        signal,
        yerr=signal_error,
        fmt="o",
        markersize=4.0,
        color=red,
        ecolor=red,
        elinewidth=1.0,
        capsize=1.8,
        label=r"Signal, 1 fb equivalent $\sigma_{hhhh}$ (MC stat.)",
        zorder=5,
    )
    ax.set_yscale("log")
    positive = np.concatenate((signal[signal > 0.0], background[background > 0.0]))
    if len(positive):
        ax.set_ylim(float(np.min(positive)) * 0.35, float(np.max(positive)) * 4.0)
    ax.set_ylabel("Expected events")
    ax.legend(loc="upper left", frameon=False, ncol=1, fontsize=9)
    ax.text(
        0.99,
        0.97,
        r"$\sqrt{s}=14$ TeV"
        + "\n"
        + rf"$\mathcal{{L}}={luminosity_fb_inverse / 1000:g}$ ab$^{{-1}}$"
        + "\n"
        + rf"$c_3=d_4=0,\ \sigma^{{\rm exp}}_{{95}}={sigma95_fb:.3g}$ fb",
        ha="right",
        va="top",
        transform=ax.transAxes,
    )

    signal_over_background = np.divide(
        signal,
        background,
        out=np.full_like(signal, np.nan),
        where=background > 0.0,
    )
    ratio.plot(x, signal_over_background, "o", color=red, markersize=4.0)
    ratio.axhline(1.0, color="0.35", linewidth=1.0, linestyle=":")
    ratio.set_yscale("log")
    ratio.set_ylabel(r"$S(1\,{\rm fb})/B$")
    ratio.set_xlabel(
        "Frozen score-bin number within each evaluation fold "
        "(more signal-like $\\rightarrow$)"
    )

    tick_positions = []
    tick_labels = []
    for fold_index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        tick_positions.extend(np.arange(left, right, dtype=float))
        tick_labels.extend([str(index + 1) for index in range(right - left)])
        center = 0.5 * (left + right - 1)
        ax.text(
            center,
            0.015,
            f"fold {fold_index + 1}",
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
            fontsize=10,
        )
        if fold_index:
            for axis in (ax, ratio):
                axis.axvline(left - 0.5, color="0.55", linewidth=0.8)
    ratio.set_xticks(tick_positions)
    ratio.set_xticklabels(tick_labels)
    ratio.set_xlim(-0.6, len(signal) - 0.4)
    return _save_figure(fig, output_stem, dpi=dpi)


def _write_output_readme(
    path: Path,
    *,
    study_dir: Path,
    sigma95_fb: float,
    hhhh_xsec_fb: float,
    rescored: bool,
) -> None:
    closure_text = (
        "The ROOT files and five saved models were reloaded, every event was "
        "scored exactly once out of fold, and the reconstructed bins were required "
        "to match the stored workspace."
        if rescored
        else "This report was made from the exact templates stored in the completed "
        "workspace; the optional ROOT/model closure test was skipped."
    )
    path.write_text(
        f"""# SM fast-sm XGBoost-score templates

These files describe the Standard-Model coupling point (`c3=d4=0`) in the
completed study `{study_dir}`.

The important distinction is:

1. XGBoost was trained earlier and assigns one frozen score to each event.
2. Validation-background quantiles fixed separate score boundaries for each
   fold.
3. The saved pyhf workspace contains Poisson event-count templates in those
   fixed bins. pyhf does not train XGBoost or fit a smooth score curve.

`sm_oof_xgboost_score_distribution` shows a common 0--1 display histogram.
It is useful for seeing classifier separation, but its display bins are not
the pyhf bins. `sm_pyhf_unrolled_templates` shows the exact five-channel,
20-bin templates passed to pyhf. Each fold has its own boundaries, recorded in
`sm_pyhf_templates.csv` and `workspace_sm.json`.

The signal template is expressed per 1 fb of *equivalent* hhhh production
cross section. At the SM point the hhhh and post-training hhhbb predictions
are scaled by one common signal strength. The saved median expected limit is
`{sigma95_fb:.12g} fb`, while the hhhh theory cross section used as the
equivalent basis is `{hhhh_xsec_fb:.12g} fb`.

The workspace observations are background-only Asimov counts: the nominal
background expectation is used in place of real observed data. {closure_text}

`report.json` records input and model hashes, normalization metadata, closure
diagnostics, and hashes of every generated artifact. `sm_oof_scores.csv.gz`
is the auditable event-level score table when re-scoring was enabled.
""",
        encoding="utf-8",
    )


def write_sm_score_template_report(
    study_dir: str | Path,
    *,
    strategy: str = DEFAULT_STRATEGY,
    output_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    display_bins: int = DEFAULT_DISPLAY_BINS,
    dpi: int = 300,
    rescore: bool = True,
    verify_input_hashes: bool = True,
) -> dict[str, Any]:
    """Write exact SM template tables and publication-quality plots."""

    study_dir = Path(study_dir).expanduser().resolve()
    if output_dir is None:
        output_dir = study_dir / strategy / "score_template_report" / "sm"
    output_dir = Path(output_dir).expanduser().resolve()
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    repo_root = Path(repo_root).expanduser().resolve()

    manifest_path = study_dir / "method_manifest.json"
    shape_path = study_dir / strategy / "shape_results.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping) or manifest.get("status") != "complete":
        raise ValueError(f"{manifest_path} is not a completed study manifest")
    if strategy not in manifest.get("strategies_completed", []):
        raise ValueError(f"Strategy {strategy!r} is not complete in the study manifest")
    if not bool(manifest.get("physics_result_valid")):
        raise ValueError("The recorded study is not marked as a valid physics result")
    shape_results = _read_json(shape_path)
    sm_row = _find_sm_shape_row(shape_results)
    channels = extract_workspace_templates(sm_row)

    observable_set = str(manifest.get("observable_set") or "")
    profile = str(manifest.get("selected_feature_profile") or "")
    get_feature_contract(observable_set, profile)
    luminosity = float(manifest.get("luminosity_fb_inverse"))
    n_folds = int(manifest.get("cv_folds"))
    seed = int(manifest.get("seed"))
    max_events = (manifest.get("mode_policy") or {}).get("max_events_per_source")
    hhhh_xsec_fb = float(sm_row["hhhh_xsec_fb"])
    sigma95_fb = float(sm_row["shape_sigma95_fb"])
    workspace = sm_row["pyhf_shape_with_mcstat"]["workspace_spec"]

    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_path = output_dir / "workspace_sm.json"
    template_csv = output_dir / "sm_pyhf_templates.csv"
    _write_json_atomic(workspace_path, workspace)
    _write_template_csv(
        template_csv,
        channels,
        hhhh_xsec_fb=hhhh_xsec_fb,
        sigma95_fb=sigma95_fb,
    )

    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "status": "running",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "study_dir": str(study_dir),
        "strategy": strategy,
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "method_version": manifest.get("method_version"),
            "source_commit": manifest.get("source_commit"),
            "study_mode": manifest.get("study_mode"),
            "paper_ready": manifest.get("paper_ready"),
            "package_versions": manifest.get("package_versions"),
        },
        "shape_results": {"path": str(shape_path), "sha256": _sha256(shape_path)},
        "observable_set": observable_set,
        "feature_profile": profile,
        "luminosity_fb_inverse": luminosity,
        "seed": seed,
        "cv_folds": n_folds,
        "point": {
            "point_id": sm_row.get("point_id"),
            "c3": float(sm_row["c3"]),
            "d4": float(sm_row["d4"]),
            "signal_components": sm_row.get("signal_components"),
            "hhhh_xsec_fb": hhhh_xsec_fb,
            "limit_cross_section_basis": sm_row.get("limit_cross_section_basis"),
            "shape_sigma95_fb": sigma95_fb,
            "bin_count_per_fold": int(sm_row["bin_count"]),
            "fold_bin_edges": sm_row["fold_bin_edges"],
        },
        "workspace": {
            "path": str(workspace_path),
            "poi": workspace["measurements"][0]["config"]["poi"],
            "observation_kind": "background-only Asimov",
            "channels": channels,
        },
        "rescore": {"enabled": bool(rescore)},
    }

    histograms = None
    event_score_path = output_dir / "sm_oof_scores.csv.gz"
    if rescore:
        spec_groups: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {
            "grid_signal": _manifest_specs(
                manifest,
                kind="grid_signal",
                repo_root=repo_root,
                sm_point_only=True,
            ),
            "postfit_hhhbb_signal": _manifest_specs(
                manifest,
                kind="postfit_hhhbb_signal",
                repo_root=repo_root,
                sm_point_only=True,
            ),
            "background": _manifest_specs(
                manifest,
                kind="background",
                repo_root=repo_root,
                sm_point_only=False,
            ),
        }
        if len(spec_groups["grid_signal"][0]) != 1:
            raise ValueError("The manifest must contain exactly one SM grid hhhh input")
        if len(spec_groups["postfit_hhhbb_signal"][0]) != 1:
            raise ValueError("The manifest must contain exactly one SM hhhbb input")
        if not spec_groups["background"][0]:
            raise ValueError("The manifest contains no background inputs")

        input_records = [
            item
            for _, records in spec_groups.values()
            for item in records
        ]
        if verify_input_hashes:
            verified_inputs = _verify_input_records(input_records)
        else:
            verified_inputs = [
                {
                    "sample_id": item.get("sample_id"),
                    "kind": item.get("kind"),
                    "path": item["resolved_path"],
                    "sha256": item.get("sha256"),
                    "verified": False,
                    "entries": item.get("entries"),
                    "sum_raw_weight": item.get("sum_raw_weight"),
                    "sum_physical_weight": item.get("sum_physical_weight"),
                    "xsec_fb": item.get("xsec_fb"),
                    "rate_factor": item.get("rate_factor"),
                    "normalisation_weight": item.get("normalisation_weight"),
                    "normalisation_source": item.get("normalisation_source"),
                    "generated_events": item.get("generated_events"),
                    "c3": item.get("c3"),
                    "d4": item.get("d4"),
                }
                for item in input_records
            ]
        common = {
            "observable_set": observable_set,
            "luminosity": luminosity,
            "n_folds": n_folds,
            "seed": seed,
            "max_events": max_events,
        }
        hhhh_samples = _load_samples(
            spec_groups["grid_signal"][0], kind="grid_signal", **common
        )
        hhhbb_samples = _load_samples(
            spec_groups["postfit_hhhbb_signal"][0],
            kind="postfit_hhhbb_signal",
            **common,
        )
        background_samples = _load_samples(
            spec_groups["background"][0], kind="background", **common
        )
        models, model_records = _load_models(
            study_dir / strategy,
            n_folds=n_folds,
            observable_set=observable_set,
            profile=profile,
        )
        reconstructed = reconstruct_oof_scores(
            models=models,
            hhhh_samples=hhhh_samples,
            hhhbb_samples=hhhbb_samples,
            background_samples=background_samples,
            n_folds=n_folds,
            profile_indices=_profile_indices(observable_set, profile),
            hhhh_xsec_fb=hhhh_xsec_fb,
        )
        closure = validate_template_closure(reconstructed["folds"], channels)
        histograms = make_display_histograms(
            reconstructed["folds"], n_bins=int(display_bins)
        )
        _write_event_scores(event_score_path, reconstructed["event_rows"])
        report["rescore"].update(
            {
                "status": "passed",
                "repo_root": str(repo_root),
                "input_hashes_checked": bool(verify_input_hashes),
                "inputs": verified_inputs,
                "models": model_records,
                "loaded_event_count": len(reconstructed["event_rows"]),
                "exactly_once_out_of_fold": True,
                "template_closure": closure,
                "event_score_table": str(event_score_path),
            }
        )
    else:
        report["rescore"]["status"] = "skipped"

    if histograms is not None:
        display_histogram_path = output_dir / "sm_oof_display_histograms.json"
        _write_json_atomic(display_histogram_path, histograms)
        score_plots = plot_oof_score_distribution(
            histograms,
            output_dir / "sm_oof_xgboost_score_distribution",
            luminosity_fb_inverse=luminosity,
            hhhh_xsec_fb=hhhh_xsec_fb,
            sigma95_fb=sigma95_fb,
            dpi=dpi,
        )
        report["display_histograms"] = {
            "path": str(display_histogram_path),
            "binning_role": "visualization only; not the pyhf binning",
            "plots": score_plots,
        }
    unrolled_plots = plot_unrolled_pyhf_templates(
        channels,
        output_dir / "sm_pyhf_unrolled_templates",
        luminosity_fb_inverse=luminosity,
        hhhh_xsec_fb=hhhh_xsec_fb,
        sigma95_fb=sigma95_fb,
        dpi=dpi,
    )
    report["unrolled_pyhf_templates"] = {
        "table": str(template_csv),
        "plots": unrolled_plots,
        "binning_role": "exact bins supplied to pyhf",
    }
    readme_path = output_dir / "README.md"
    _write_output_readme(
        readme_path,
        study_dir=study_dir,
        sigma95_fb=sigma95_fb,
        hhhh_xsec_fb=hhhh_xsec_fb,
        rescored=rescore,
    )

    artifact_paths = [
        workspace_path,
        template_csv,
        readme_path,
        *[Path(path) for path in unrolled_plots.values()],
    ]
    if rescore:
        artifact_paths.extend(
            [
                event_score_path,
                Path(report["display_histograms"]["path"]),
                *[
                    Path(path)
                    for path in report["display_histograms"]["plots"].values()
                ],
            ]
        )
    report["artifacts"] = [
        {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in artifact_paths
    ]
    report["status"] = "complete"
    report_path = output_dir / "report.json"
    _write_json_atomic(report_path, report)
    return _read_json(report_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write exact SM pyhf-template and out-of-fold XGBoost-score plots "
            "from a completed c3/d4 fast-sm study."
        )
    )
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Repository root used to resolve relative ROOT paths in the manifest.",
    )
    parser.add_argument("--display-bins", type=int, default=DEFAULT_DISPLAY_BINS)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--skip-rescore",
        action="store_true",
        help=(
            "Extract and plot saved workspace templates without reloading ROOT/models."
        ),
    )
    parser.add_argument(
        "--skip-input-hash-check",
        action="store_true",
        help="Do not recompute SHA-256 digests for the ROOT inputs used in re-scoring.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = write_sm_score_template_report(
        args.study_dir,
        strategy=args.strategy,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        display_bins=args.display_bins,
        dpi=args.dpi,
        rescore=not args.skip_rescore,
        verify_input_hashes=not args.skip_input_hash_check,
    )
    print(
        json.dumps(
            {"status": report["status"], "artifacts": report["artifacts"]}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
