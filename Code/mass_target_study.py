#!/usr/bin/env python3
"""Small held-out scan of four-Higgs reconstruction mass targets.

The study reruns the exact resolved and hybrid C++ reconstructions for a
compact, predeclared set of mass-target tuples.  It chooses a tuple using a
deterministic tuning subset and reports its signal-versus-background
mass-score AUC on a disjoint validation subset.  The objective is deliberately
limited to the reconstruction mass score; shortlisted tuples must still be
confirmed with the complete classifier and expected-limit workflows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


WORKFLOWS = ("resonant", "nonresonant")
STUDY_VERSION = "four-higgs-mass-target-study-v1"
BASELINE_TARGETS = {
    "resonant": (125.0, 125.0, 125.0, 125.0),
    "nonresonant": (120.0, 115.0, 110.0, 105.0),
}


class StudyInputError(RuntimeError):
    """Raised when a study input or generated artifact violates the contract."""


@dataclass(frozen=True)
class TargetPoint:
    values: tuple[float, float, float, float]

    @property
    def target_id(self) -> str:
        return "m" + "_".join(_format_id_number(value) for value in self.values)

    @property
    def cli_value(self) -> str:
        return ",".join(_format_number(value) for value in self.values)


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    label: str
    workflow: str
    raw_root: Path
    c_mistags: int
    light_mistags: int
    max_reco_jets: int
    class_weight: float

    def supports(self, workflow: str) -> bool:
        return self.workflow in {"both", workflow}


@dataclass(frozen=True)
class ExtractionJob:
    workflow: str
    target: TargetPoint
    sample: SampleSpec
    command: tuple[str, ...]
    output_root: Path
    summary_json: Path
    log_file: Path
    input_list: Path | None
    record_json: Path


@dataclass(frozen=True)
class Observations:
    sample_id: str
    label: str
    event_index: np.ndarray
    score: np.ndarray
    weight: np.ndarray
    candidate_masses: np.ndarray


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else repr(value)


def _format_id_number(value: float) -> str:
    text = _format_number(value)
    return text.replace("-", "neg").replace(".", "p")


def parse_target_point(text: str) -> TargetPoint:
    pieces = [piece.strip() for piece in text.split(",")]
    if len(pieces) != 4:
        raise StudyInputError(
            f"mass target {text!r} must contain four comma-separated values"
        )
    try:
        values = tuple(float(piece) for piece in pieces)
    except ValueError as error:
        raise StudyInputError(f"invalid mass target {text!r}") from error
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise StudyInputError(f"mass targets must be positive and finite: {text!r}")
    if any(values[index] > values[index - 1] for index in range(1, 4)):
        raise StudyInputError(
            "mass targets must be non-increasing in descending candidate-pT order: "
            f"{text!r}"
        )
    return TargetPoint(values=values)  # type: ignore[arg-type]


def small_target_points() -> list[TargetPoint]:
    """Return the frozen compact scan around the two current prescriptions."""

    raw: list[tuple[float, float, float, float]] = []
    for common in (105.0, 110.0, 115.0, 120.0, 125.0, 130.0):
        raw.append((common, common, common, common))

    staggered = np.asarray(BASELINE_TARGETS["nonresonant"], dtype=float)
    for shift in (-5.0, -2.5, 0.0, 2.5, 5.0):
        raw.append(tuple(float(value) for value in staggered + shift))
    for rank in range(4):
        for shift in (-2.5, 2.5):
            varied = staggered.copy()
            varied[rank] += shift
            raw.append(tuple(float(value) for value in varied))

    result: list[TargetPoint] = []
    seen: set[tuple[float, float, float, float]] = set()
    for values in raw:
        point = parse_target_point(",".join(_format_number(value) for value in values))
        if point.values not in seen:
            seen.add(point.values)
            result.append(point)
    return result


def resolve_target_points(
    preset: str,
    custom_values: Sequence[str],
) -> list[TargetPoint]:
    points = small_target_points() if preset == "small" else []
    points.extend(
        TargetPoint(values=values) for values in BASELINE_TARGETS.values()
    )
    points.extend(parse_target_point(value) for value in custom_values)
    unique: list[TargetPoint] = []
    seen: set[tuple[float, float, float, float]] = set()
    for point in points:
        if point.values not in seen:
            seen.add(point.values)
            unique.append(point)
    if not unique:
        raise StudyInputError("no mass-target points were selected")
    return unique


def _nonnegative_int(value: str, field: str, sample_id: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise StudyInputError(f"{sample_id}: {field} must be an integer") from error
    if parsed < 0:
        raise StudyInputError(f"{sample_id}: {field} must be non-negative")
    return parsed


def load_sample_manifest(
    path: Path,
    analysis_root: Path,
    requested_workflows: Sequence[str],
) -> list[SampleSpec]:
    if not path.is_file():
        raise StudyInputError(f"sample manifest does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "class", "workflow", "raw_root"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise StudyInputError(
                f"{path}: missing required columns {sorted(missing)}"
            )
        rows = list(reader)

    samples: list[SampleSpec] = []
    for row in rows:
        sample_id = row["sample_id"].strip()
        label = row["class"].strip().lower()
        workflow = row["workflow"].strip().lower()
        if not sample_id:
            raise StudyInputError(f"{path}: empty sample_id")
        if label not in {"signal", "background"}:
            raise StudyInputError(
                f"{sample_id}: class must be signal or background"
            )
        if workflow not in {"both", *WORKFLOWS}:
            raise StudyInputError(
                f"{sample_id}: workflow must be both, resonant, or nonresonant"
            )
        raw_text = row["raw_root"].strip()
        if not raw_text:
            raise StudyInputError(f"{sample_id}: raw_root is required")
        raw_root = Path(raw_text).expanduser()
        if not raw_root.is_absolute():
            raw_root = (analysis_root / raw_root).resolve()
        c_mistags = _nonnegative_int(
            row.get("c_mistags", "0").strip() or "0",
            "c_mistags",
            sample_id,
        )
        light_mistags = _nonnegative_int(
            row.get("light_mistags", "0").strip() or "0",
            "light_mistags",
            sample_id,
        )
        if c_mistags + light_mistags > 8:
            raise StudyInputError(
                f"{sample_id}: c_mistags + light_mistags exceeds eight"
            )
        max_reco_jets = _nonnegative_int(
            row.get("max_reco_jets", "10").strip() or "10",
            "max_reco_jets",
            sample_id,
        )
        if not 4 <= max_reco_jets <= 10:
            raise StudyInputError(
                f"{sample_id}: max_reco_jets must lie in [4, 10]"
            )
        try:
            class_weight = float(
                row.get("class_weight", "1").strip() or "1"
            )
        except ValueError as error:
            raise StudyInputError(
                f"{sample_id}: class_weight must be numeric"
            ) from error
        if not math.isfinite(class_weight) or class_weight <= 0.0:
            raise StudyInputError(
                f"{sample_id}: class_weight must be positive and finite"
            )
        samples.append(
            SampleSpec(
                sample_id=sample_id,
                label=label,
                workflow=workflow,
                raw_root=raw_root,
                c_mistags=c_mistags,
                light_mistags=light_mistags,
                max_reco_jets=max_reco_jets,
                class_weight=class_weight,
            )
        )

    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = sorted(
            sample_id
            for sample_id in set(sample_ids)
            if sample_ids.count(sample_id) > 1
        )
        raise StudyInputError(f"duplicate sample IDs: {duplicates}")
    for workflow in requested_workflows:
        labels = {
            sample.label for sample in samples if sample.supports(workflow)
        }
        if labels != {"signal", "background"}:
            raise StudyInputError(
                f"{workflow} requires at least one signal and one background; "
                f"found {sorted(labels)}"
            )
    return samples


def requested_workflows(value: str) -> tuple[str, ...]:
    return WORKFLOWS if value == "both" else (value,)


def _nonresonant_paths(
    output_dir: Path,
    target: TargetPoint,
    sample: SampleSpec,
) -> tuple[Path, Path, Path, str]:
    directory = output_dir / "extractions" / "nonresonant" / target.target_id
    input_list = directory / f"{sample.sample_id}.input"
    tag = (
        "extended-v2-uniform-smear-v1-target-study-"
        f"{target.target_id}"
    )
    output_root = directory / (
        f"{sample.sample_id}-{tag}_var.smearCMS.root"
    )
    summary = directory / f"{sample.sample_id}-{tag}.analysis_summary.json"
    return input_list, output_root, summary, tag


def build_extraction_jobs(
    *,
    output_dir: Path,
    targets: Sequence[TargetPoint],
    samples: Sequence[SampleSpec],
    workflows: Sequence[str],
    resonant_executable: Path,
    nonresonant_executable: Path,
    max_events: int,
) -> list[ExtractionJob]:
    jobs: list[ExtractionJob] = []
    for workflow in workflows:
        for target in targets:
            for sample in samples:
                if not sample.supports(workflow):
                    continue
                log_file = (
                    output_dir
                    / "logs"
                    / workflow
                    / target.target_id
                    / f"{sample.sample_id}.log"
                )
                input_list: Path | None = None
                if workflow == "resonant":
                    output_root = (
                        output_dir
                        / "extractions"
                        / workflow
                        / target.target_id
                        / f"{sample.sample_id}.root"
                    )
                    summary = output_root.with_suffix(
                        ".analysis_summary.json"
                    )
                    command = [
                        str(resonant_executable),
                        str(sample.raw_root),
                        "--output",
                        str(output_root),
                        "--max-events",
                        str(max_events),
                        "--max-reco-jets",
                        str(sample.max_reco_jets),
                        "--higgs-mass-targets",
                        target.cli_value,
                    ]
                else:
                    input_list, output_root, summary, tag = _nonresonant_paths(
                        output_dir, target, sample
                    )
                    command = [
                        str(nonresonant_executable),
                        str(input_list),
                        "-t",
                        tag,
                        "-n",
                        str(max_events),
                        "--higgs-mass-targets",
                        target.cli_value,
                    ]
                if sample.c_mistags:
                    command.extend(["--c-mistags", str(sample.c_mistags)])
                if sample.light_mistags:
                    command.extend(
                        ["--light-mistags", str(sample.light_mistags)]
                    )
                jobs.append(
                    ExtractionJob(
                        workflow=workflow,
                        target=target,
                        sample=sample,
                        command=tuple(command),
                        output_root=output_root,
                        summary_json=summary,
                        log_file=log_file,
                        input_list=input_list,
                        record_json=output_root.with_suffix(".job.json"),
                    )
                )
    return jobs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _job_contract(
    job: ExtractionJob,
    source_files: Sequence[Path],
) -> dict[str, Any]:
    raw_stat = job.sample.raw_root.stat()
    executable = Path(job.command[0])
    return {
        "study_version": STUDY_VERSION,
        "workflow": job.workflow,
        "target_id": job.target.target_id,
        "targets_gev": list(job.target.values),
        "sample_id": job.sample.sample_id,
        "raw_root": str(job.sample.raw_root),
        "raw_size": raw_stat.st_size,
        "raw_mtime_ns": raw_stat.st_mtime_ns,
        "command": list(job.command),
        "executable": str(executable),
        "executable_sha256": _sha256(executable),
        "sources": [
            {"path": str(source), "sha256": _sha256(source)}
            for source in source_files
        ],
    }


def _job_is_current(
    job: ExtractionJob,
    source_files: Sequence[Path],
) -> bool:
    if not (
        job.output_root.is_file()
        and job.summary_json.is_file()
        and job.record_json.is_file()
    ):
        return False
    try:
        recorded = json.loads(job.record_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return recorded == _job_contract(job, source_files)


def _run_extraction_job(
    job: ExtractionJob,
    source_files: Sequence[Path],
    force: bool,
) -> dict[str, Any]:
    if not job.sample.raw_root.is_file():
        raise StudyInputError(
            f"{job.sample.sample_id}: raw ROOT file is missing: "
            f"{job.sample.raw_root}"
        )
    if not force and _job_is_current(job, source_files):
        return {
            "workflow": job.workflow,
            "target_id": job.target.target_id,
            "sample_id": job.sample.sample_id,
            "status": "reused",
            "output_root": str(job.output_root),
        }

    job.output_root.parent.mkdir(parents=True, exist_ok=True)
    job.log_file.parent.mkdir(parents=True, exist_ok=True)
    if job.input_list is not None:
        job.input_list.write_text(
            f"{job.sample.raw_root}\n", encoding="utf-8"
        )
    with job.log_file.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            list(job.command),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{job.workflow}/{job.target.target_id}/{job.sample.sample_id} "
            f"failed; see {job.log_file}"
        )
    if not job.output_root.is_file() or not job.summary_json.is_file():
        raise RuntimeError(
            f"extractor did not create the expected ROOT/JSON pair for "
            f"{job.sample.sample_id}; see {job.log_file}"
        )
    contract = _job_contract(job, source_files)
    job.record_json.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "workflow": job.workflow,
        "target_id": job.target.target_id,
        "sample_id": job.sample.sample_id,
        "status": "complete",
        "output_root": str(job.output_root),
    }


def run_extractions(
    jobs: Sequence[ExtractionJob],
    *,
    workers: int,
    force: bool,
    resonant_source: Path,
    nonresonant_source: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    sources_by_workflow = {
        "resonant": (resonant_source,),
        "nonresonant": (
            nonresonant_source,
            nonresonant_source.with_name("Extended91Observables.h"),
        ),
    }
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        future_jobs = {
            pool.submit(
                _run_extraction_job,
                job,
                sources_by_workflow[job.workflow],
                force,
            ): job
            for job in jobs
        }
        for future in as_completed(future_jobs):
            job = future_jobs[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - aggregate all jobs
                failures.append(str(error))
                continue
            records.append(record)
            print(
                f"[{record['status']}] {job.workflow} "
                f"{job.target.target_id} {job.sample.sample_id}"
            )
    records.sort(
        key=lambda row: (
            row["workflow"],
            row["target_id"],
            row["sample_id"],
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {"jobs": records, "failures": failures},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} extraction job(s) failed:\n- "
            + "\n- ".join(failures)
        )
    return records


def weighted_auc(
    signal_scores: np.ndarray,
    background_scores: np.ndarray,
    signal_weights: np.ndarray,
    background_weights: np.ndarray,
) -> float:
    """Weighted probability that a random signal score exceeds background."""

    signal_scores = np.asarray(signal_scores, dtype=float)
    background_scores = np.asarray(background_scores, dtype=float)
    signal_weights = np.asarray(signal_weights, dtype=float)
    background_weights = np.asarray(background_weights, dtype=float)
    if not (
        len(signal_scores)
        and len(background_scores)
        and len(signal_scores) == len(signal_weights)
        and len(background_scores) == len(background_weights)
    ):
        raise StudyInputError("AUC inputs are empty or have inconsistent sizes")
    if (
        np.any(~np.isfinite(signal_scores))
        or np.any(~np.isfinite(background_scores))
        or np.any(~np.isfinite(signal_weights))
        or np.any(~np.isfinite(background_weights))
        or np.any(signal_weights < 0.0)
        or np.any(background_weights < 0.0)
    ):
        raise StudyInputError("AUC inputs must be finite with non-negative weights")
    signal_total = float(np.sum(signal_weights))
    background_total = float(np.sum(background_weights))
    if signal_total <= 0.0 or background_total <= 0.0:
        raise StudyInputError("AUC class weights must each have positive sum")

    scores = np.concatenate((signal_scores, background_scores))
    labels = np.concatenate(
        (
            np.ones(len(signal_scores), dtype=np.int8),
            np.zeros(len(background_scores), dtype=np.int8),
        )
    )
    weights = np.concatenate((signal_weights, background_weights))
    order = np.argsort(scores, kind="mergesort")
    scores = scores[order]
    labels = labels[order]
    weights = weights[order]

    numerator = 0.0
    background_before = 0.0
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[stop] == scores[start]:
            stop += 1
        group_labels = labels[start:stop]
        group_weights = weights[start:stop]
        signal_group = float(np.sum(group_weights[group_labels == 1]))
        background_group = float(np.sum(group_weights[group_labels == 0]))
        numerator += signal_group * (
            background_before + 0.5 * background_group
        )
        background_before += background_group
        start = stop
    return numerator / (signal_total * background_total)


def _is_validation_event(
    sample_id: str,
    event_index: int,
    *,
    fraction: float,
    seed: int,
) -> bool:
    payload = f"{seed}:{sample_id}:{event_index}".encode("utf-8")
    value = int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "big"
    )
    return value / float(2**64) < fraction


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(values, kind="mergesort")
    values = np.asarray(values, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    cumulative = np.cumsum(weights)
    index = int(np.searchsorted(cumulative, probability * total, side="left"))
    return float(values[min(index, len(values) - 1)])


def _root_named_title(root_file: Any, name: str) -> str:
    item = root_file.Get(name)
    if not item:
        raise StudyInputError(f"ROOT metadata {name!r} is missing")
    return str(item.GetTitle())


def load_root_observations(job: ExtractionJob, root_module: Any) -> Observations:
    root_file = root_module.TFile.Open(str(job.output_root), "READ")
    if not root_file or root_file.IsZombie():
        raise StudyInputError(f"cannot open {job.output_root}")
    try:
        if job.workflow == "resonant":
            tree = root_file.Get("ResonanceFeatures")
            metadata = "higgs_mass_targets_gev_json"
        else:
            tree = root_file.Get("Data3")
            metadata = "Data3_higgs_mass_targets_gev_json"
        if not tree:
            raise StudyInputError(
                f"{job.output_root}: expected feature tree is missing"
            )
        observed_targets = tuple(
            float(value)
            for value in json.loads(_root_named_title(root_file, metadata))
        )
        if observed_targets != job.target.values:
            raise StudyInputError(
                f"{job.output_root}: embedded targets {observed_targets} do not "
                f"match requested {job.target.values}"
            )

        event_indices: list[int] = []
        scores: list[float] = []
        weights: list[float] = []
        masses: list[list[float]] = []
        for entry in tree:
            event_indices.append(int(entry.event_index))
            weights.append(float(entry.weight))
            if job.workflow == "resonant":
                scores.append(-float(entry.best_score))
                masses.append(
                    [float(entry.higgs_mass[index]) for index in range(4)]
                )
            else:
                scores.append(-float(entry.features[9]))
                masses.append(
                    [float(entry.features[28 + index]) for index in range(4)]
                )
    finally:
        root_file.Close()

    arrays = Observations(
        sample_id=job.sample.sample_id,
        label=job.sample.label,
        event_index=np.asarray(event_indices, dtype=np.int64),
        score=np.asarray(scores, dtype=float),
        weight=np.asarray(weights, dtype=float),
        candidate_masses=np.asarray(masses, dtype=float).reshape((-1, 4)),
    )
    if (
        np.any(~np.isfinite(arrays.score))
        or np.any(~np.isfinite(arrays.weight))
        or np.any(~np.isfinite(arrays.candidate_masses))
        or np.any(arrays.weight < 0.0)
    ):
        raise StudyInputError(
            f"{job.output_root}: non-finite values or negative event weights "
            "are unsupported by this compact AUC study"
        )
    if len(np.unique(arrays.event_index)) != len(arrays.event_index):
        raise StudyInputError(f"{job.output_root}: duplicate event_index values")
    return arrays


def _combine_partition(
    observations: Sequence[tuple[Observations, SampleSpec]],
    *,
    validation: bool,
    validation_fraction: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    for sample_observations, sample in observations:
        mask = np.asarray(
            [
                _is_validation_event(
                    sample.sample_id,
                    int(event_index),
                    fraction=validation_fraction,
                    seed=split_seed,
                )
                == validation
                for event_index in sample_observations.event_index
            ],
            dtype=bool,
        )
        total_weight = float(np.sum(sample_observations.weight))
        if total_weight <= 0.0:
            raise StudyInputError(
                f"{sample.sample_id}: event weights have non-positive sum"
            )
        scaled_weight = (
            sample_observations.weight[mask]
            * sample.class_weight
            / total_weight
        )
        scores.append(sample_observations.score[mask])
        weights.append(scaled_weight)
        masses.append(sample_observations.candidate_masses[mask])
    if not scores:
        raise StudyInputError("no observations are available for one class")
    return (
        np.concatenate(scores),
        np.concatenate(weights),
        np.concatenate(masses, axis=0),
    )


def _bootstrap_auc_interval(
    signal_scores: np.ndarray,
    background_scores: np.ndarray,
    signal_weights: np.ndarray,
    background_weights: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    if repetitions <= 0:
        return float("nan"), float("nan")
    random = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        signal_indices = random.integers(
            0, len(signal_scores), len(signal_scores)
        )
        background_indices = random.integers(
            0, len(background_scores), len(background_scores)
        )
        values[index] = weighted_auc(
            signal_scores[signal_indices],
            background_scores[background_indices],
            signal_weights[signal_indices],
            background_weights[background_indices],
        )
    low, high = np.quantile(values, (0.16, 0.84))
    return float(low), float(high)


def evaluate_jobs(
    jobs: Sequence[ExtractionJob],
    *,
    output_dir: Path,
    validation_fraction: float,
    split_seed: int,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    make_plots: bool,
    root_module: Any | None = None,
) -> dict[str, Any]:
    if not 0.0 < validation_fraction < 1.0:
        raise StudyInputError("--validation-fraction must lie strictly in (0, 1)")
    missing = [
        str(job.output_root)
        for job in jobs
        if not job.output_root.is_file()
    ]
    if missing:
        raise StudyInputError(
            f"{len(missing)} extraction output(s) are missing; run mode first. "
            f"First missing file: {missing[0]}"
        )
    if root_module is None:
        try:
            import ROOT as root_module  # type: ignore[no-redef]
        except ImportError as error:
            raise StudyInputError(
                "PyROOT is required to evaluate the generated feature trees"
            ) from error
    root_module.gROOT.SetBatch(True)

    cached: dict[tuple[str, str, str], Observations] = {}
    jobs_by_group: dict[tuple[str, str], list[ExtractionJob]] = {}
    for job in jobs:
        cached[(job.workflow, job.target.target_id, job.sample.sample_id)] = (
            load_root_observations(job, root_module)
        )
        jobs_by_group.setdefault(
            (job.workflow, job.target.target_id), []
        ).append(job)

    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    validation_arrays: dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    for (workflow, target_id), group_jobs in sorted(jobs_by_group.items()):
        target = group_jobs[0].target
        by_label: dict[str, list[tuple[Observations, SampleSpec]]] = {
            "signal": [],
            "background": [],
        }
        for job in group_jobs:
            by_label[job.sample.label].append(
                (
                    cached[
                        (
                            job.workflow,
                            job.target.target_id,
                            job.sample.sample_id,
                        )
                    ],
                    job.sample,
                )
            )
        signal_tune = _combine_partition(
            by_label["signal"],
            validation=False,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
        background_tune = _combine_partition(
            by_label["background"],
            validation=False,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
        signal_validation = _combine_partition(
            by_label["signal"],
            validation=True,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
        background_validation = _combine_partition(
            by_label["background"],
            validation=True,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
        tune_auc = weighted_auc(
            signal_tune[0],
            background_tune[0],
            signal_tune[1],
            background_tune[1],
        )
        validation_auc = weighted_auc(
            signal_validation[0],
            background_validation[0],
            signal_validation[1],
            background_validation[1],
        )
        for signal_observations, signal_sample in by_label["signal"]:
            sample_tune = _combine_partition(
                [(signal_observations, signal_sample)],
                validation=False,
                validation_fraction=validation_fraction,
                split_seed=split_seed,
            )
            sample_validation = _combine_partition(
                [(signal_observations, signal_sample)],
                validation=True,
                validation_fraction=validation_fraction,
                split_seed=split_seed,
            )
            sample_rows.append(
                {
                    "workflow": workflow,
                    "target_id": target_id,
                    "signal_sample_id": signal_sample.sample_id,
                    "tune_auc": weighted_auc(
                        sample_tune[0],
                        background_tune[0],
                        sample_tune[1],
                        background_tune[1],
                    ),
                    "validation_auc": weighted_auc(
                        sample_validation[0],
                        background_validation[0],
                        sample_validation[1],
                        background_validation[1],
                    ),
                    "tune_signal_events": len(sample_tune[0]),
                    "validation_signal_events": len(sample_validation[0]),
                }
            )
        validation_arrays[(workflow, target_id)] = (
            signal_validation[0],
            background_validation[0],
            signal_validation[1],
            background_validation[1],
        )
        row: dict[str, Any] = {
            "workflow": workflow,
            "target_id": target_id,
            "target_h1_gev": target.values[0],
            "target_h2_gev": target.values[1],
            "target_h3_gev": target.values[2],
            "target_h4_gev": target.values[3],
            "tune_auc": tune_auc,
            "validation_auc": validation_auc,
            "tune_signal_events": len(signal_tune[0]),
            "tune_background_events": len(background_tune[0]),
            "validation_signal_events": len(signal_validation[0]),
            "validation_background_events": len(background_validation[0]),
        }
        for rank in range(4):
            values = signal_validation[2][:, rank]
            weights = signal_validation[1]
            row[f"signal_mass_h{rank + 1}_q16_gev"] = _weighted_quantile(
                values, weights, 0.16
            )
            row[f"signal_mass_h{rank + 1}_median_gev"] = _weighted_quantile(
                values, weights, 0.50
            )
            row[f"signal_mass_h{rank + 1}_q84_gev"] = _weighted_quantile(
                values, weights, 0.84
            )
        rows.append(row)

    recommendations: dict[str, Any] = {}
    for workflow in sorted({row["workflow"] for row in rows}):
        workflow_rows = [row for row in rows if row["workflow"] == workflow]
        baseline_values = BASELINE_TARGETS[workflow]
        baseline_id = TargetPoint(baseline_values).target_id

        def selection_key(row: dict[str, Any]) -> tuple[float, float, str]:
            distance = sum(
                (
                    float(row[f"target_h{index}_gev"])
                    - baseline_values[index - 1]
                )
                ** 2
                for index in range(1, 5)
            )
            return float(row["tune_auc"]), -distance, str(row["target_id"])

        selected = max(
            workflow_rows,
            key=selection_key,
        )
        baseline = next(
            (row for row in workflow_rows if row["target_id"] == baseline_id),
            None,
        )
        if baseline is None:
            raise StudyInputError(
                f"{workflow}: baseline target {baseline_id} is absent"
            )
        arrays = validation_arrays[(workflow, selected["target_id"])]
        low, high = _bootstrap_auc_interval(
            arrays[0],
            arrays[1],
            arrays[2],
            arrays[3],
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        )
        validation_difference = (
            selected["validation_auc"] - baseline["validation_auc"]
        )
        if selected["target_id"] == baseline_id:
            decision = "retain_baseline"
        elif validation_difference <= 0.0:
            decision = "retain_baseline_no_held_out_gain"
        else:
            decision = "shortlist_for_full_analysis"
        recommendations[workflow] = {
            "selection_rule": (
                "maximum tuning-split mass-score AUC; exact ties prefer the "
                "tuple closest to the current baseline"
            ),
            "selected_target_id": selected["target_id"],
            "selected_targets_gev": [
                selected[f"target_h{index}_gev"] for index in range(1, 5)
            ],
            "tune_auc": selected["tune_auc"],
            "validation_auc": selected["validation_auc"],
            "validation_auc_bootstrap_68_interval": [low, high],
            "baseline_target_id": baseline_id,
            "baseline_validation_auc": baseline["validation_auc"],
            "validation_auc_difference_from_baseline": validation_difference,
            "decision": decision,
            "selected_validation_auc_by_signal_sample": {
                row["signal_sample_id"]: row["validation_auc"]
                for row in sample_rows
                if row["workflow"] == workflow
                and row["target_id"] == selected["target_id"]
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sample_metrics_path = output_dir / "sample_metrics.csv"
    with sample_metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    summary = {
        "study_version": STUDY_VERSION,
        "method": "held-out reconstruction-mass-score scan",
        "score_definition": (
            "negative best_score for resonance-hybrid-v1; negative chi8 for "
            "extended-91-v2"
        ),
        "candidate_order": "descending pT",
        "sample_weighting": (
            "each sample is normalized to its manifest class_weight before "
            "combination"
        ),
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "bootstrap_repetitions": bootstrap_repetitions,
        "recommendations": recommendations,
        "metrics_csv": str(metrics_path),
        "sample_metrics_csv": str(sample_metrics_path),
        "limitations": [
            "The optimized objective is the one-dimensional reconstruction mass-score AUC, not the final expected limit.",
            "The raw HwSim trees do not provide a validated truth-Higgs parent label for each reconstructed jet.",
            "Any selected tuple must be confirmed with the full XGBoost and expected-limit workflows on independent samples.",
        ],
    }
    (output_dir / "study_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown_report(output_dir / "REPORT.md", summary)
    if make_plots:
        _plot_metrics(rows, recommendations, output_dir)
    return summary


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Four-Higgs mass-target study",
        "",
        "Targets were selected on a deterministic tuning split and evaluated on "
        "a disjoint validation split using only the reconstruction mass score.",
        "",
    ]
    for workflow, result in summary["recommendations"].items():
        targets = ", ".join(
            _format_number(value) for value in result["selected_targets_gev"]
        )
        interval = result["validation_auc_bootstrap_68_interval"]
        sample_aucs = list(
            result["selected_validation_auc_by_signal_sample"].values()
        )
        lines.extend(
            [
                f"## {workflow.capitalize()}",
                "",
                f"- Tune-selected targets: `({targets})` GeV",
                f"- Tuning AUC: `{result['tune_auc']:.6f}`",
                f"- Validation AUC: `{result['validation_auc']:.6f}`",
                f"- Event-bootstrap 68% interval: "
                f"`[{interval[0]:.6f}, {interval[1]:.6f}]`",
                f"- Baseline validation AUC: "
                f"`{result['baseline_validation_auc']:.6f}`",
                f"- Validation difference from baseline: "
                f"`{result['validation_auc_difference_from_baseline']:+.6f}`",
                f"- Validation AUC range across signal samples: "
                f"`[{min(sample_aucs):.6f}, {max(sample_aucs):.6f}]`",
                f"- Decision: `{result['decision']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "This is a reconstruction diagnostic, not a final sensitivity "
            "optimization. The selected tuples must be confirmed by rerunning "
            "the complete classifier and expected-limit analyses on independent "
            "samples before changing either nominal reconstruction.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_metrics(
    rows: Sequence[dict[str, Any]],
    recommendations: dict[str, Any],
    output_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is unavailable; skipping study plots")
        return

    workflows = sorted({row["workflow"] for row in rows})
    figure, axes = plt.subplots(
        len(workflows),
        1,
        figsize=(max(10.0, 0.52 * len(rows) / len(workflows)), 4.2 * len(workflows)),
        squeeze=False,
    )
    for axis, workflow in zip(axes[:, 0], workflows):
        selected_rows = [row for row in rows if row["workflow"] == workflow]
        selected_rows.sort(key=lambda row: row["target_id"])
        positions = np.arange(len(selected_rows))
        tune = [row["tune_auc"] for row in selected_rows]
        validation = [row["validation_auc"] for row in selected_rows]
        axis.plot(positions, tune, "o-", label="tuning")
        axis.plot(positions, validation, "s-", label="validation")
        selected_id = recommendations[workflow]["selected_target_id"]
        selected_index = next(
            index
            for index, row in enumerate(selected_rows)
            if row["target_id"] == selected_id
        )
        axis.axvline(
            selected_index,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="selected on tuning split",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [
                "("
                + ",".join(
                    _format_number(row[f"target_h{index}_gev"])
                    for index in range(1, 5)
                )
                + ")"
                for row in selected_rows
            ],
            rotation=60,
            ha="right",
        )
        axis.set_ylabel("mass-score AUC")
        axis.set_title(workflow.capitalize())
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "target_auc.png", dpi=180)
    figure.savefig(output_dir / "target_auc.pdf")
    plt.close(figure)


def _build_executables(
    analysis_root: Path,
    workflows: Sequence[str],
) -> None:
    targets = []
    if "resonant" in workflows:
        targets.append("FourHiggsResonanceAnalysis")
    if "nonresonant" in workflows:
        targets.append("FourHiggs8bAnalysis_smear_CMS")
    for target in targets:
        subprocess.run(
            ["make", "-C", str(analysis_root / "Code"), target],
            check=True,
        )


def _validate_executable(
    executable: Path,
    source_files: Sequence[Path],
    workflow: str,
) -> None:
    if not executable.is_file():
        raise SystemExit(
            f"{workflow} executable is missing: {executable}; "
            "rerun with --build"
        )
    missing_sources = [source for source in source_files if not source.is_file()]
    if missing_sources:
        raise SystemExit(
            f"{workflow} source dependency is missing: {missing_sources[0]}"
        )
    executable_mtime = executable.stat().st_mtime_ns
    newer_sources = [
        source
        for source in source_files
        if source.stat().st_mtime_ns > executable_mtime
    ]
    if newer_sources:
        raise SystemExit(
            f"{workflow} executable is older than {newer_sources[0]}; "
            "rerun with --build"
        )


def _write_plan(
    output_dir: Path,
    jobs: Sequence[ExtractionJob],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "study_version": STUDY_VERSION,
        "jobs": [
            {
                "workflow": job.workflow,
                "target_id": job.target.target_id,
                "targets_gev": list(job.target.values),
                "sample_id": job.sample.sample_id,
                "command": list(job.command),
                "output_root": str(job.output_root),
            }
            for job in jobs
        ]
    }
    (output_dir / "study_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    code_dir = Path(__file__).resolve().parent
    analysis_root = code_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "evaluate", "all"))
    parser.add_argument("--analysis-root", type=Path, default=analysis_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("MassTargetStudy/sample_manifest_tiresias.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("MassTargetStudy/results/small_scan"),
    )
    parser.add_argument(
        "--workflow",
        choices=("both", *WORKFLOWS),
        default="both",
    )
    parser.add_argument("--preset", choices=("small", "none"), default="small")
    parser.add_argument(
        "--targets",
        action="append",
        default=[],
        help="Four comma-separated targets; repeat to extend the preset",
    )
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resonant-executable",
        type=Path,
        default=Path("Code/FourHiggsResonanceAnalysis"),
    )
    parser.add_argument(
        "--nonresonant-executable",
        type=Path,
        default=Path("Code/FourHiggs8bAnalysis_smear_CMS"),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.35,
    )
    parser.add_argument("--split-seed", type=int, default=8128)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=271828)
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def _resolve_from_root(path: Path, root: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_events <= 0:
        raise SystemExit("--max-events must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.bootstrap_repetitions < 0:
        raise SystemExit("--bootstrap-repetitions must be non-negative")

    root = args.analysis_root.expanduser().resolve()
    workflows = requested_workflows(args.workflow)
    manifest = _resolve_from_root(args.manifest, root)
    output_dir = _resolve_from_root(args.output_dir, root)
    resonant_executable = _resolve_from_root(args.resonant_executable, root)
    nonresonant_executable = _resolve_from_root(
        args.nonresonant_executable, root
    )
    targets = resolve_target_points(args.preset, args.targets)
    samples = load_sample_manifest(manifest, root, workflows)
    jobs = build_extraction_jobs(
        output_dir=output_dir,
        targets=targets,
        samples=samples,
        workflows=workflows,
        resonant_executable=resonant_executable,
        nonresonant_executable=nonresonant_executable,
        max_events=args.max_events,
    )
    _write_plan(output_dir, jobs)
    print(
        f"Prepared {len(jobs)} jobs for {len(targets)} target tuples; "
        f"plan: {output_dir / 'study_plan.json'}"
    )
    if args.dry_run:
        for job in jobs:
            print(" ".join(job.command))
        return 0

    if args.mode in {"run", "all"}:
        if args.build:
            _build_executables(root, workflows)
        executables = {
            "resonant": resonant_executable,
            "nonresonant": nonresonant_executable,
        }
        sources = {
            "resonant": (
                root / "Code" / "FourHiggsResonanceAnalysis.cc",
            ),
            "nonresonant": (
                root / "Code" / "FourHiggs8bAnalysis_smear_CMS.cc",
                root / "Code" / "Extended91Observables.h",
            ),
        }
        for workflow in workflows:
            _validate_executable(
                executables[workflow], sources[workflow], workflow
            )
        run_extractions(
            jobs,
            workers=args.workers,
            force=args.force,
            resonant_source=root / "Code" / "FourHiggsResonanceAnalysis.cc",
            nonresonant_source=root
            / "Code"
            / "FourHiggs8bAnalysis_smear_CMS.cc",
            output_dir=output_dir,
        )

    if args.mode in {"evaluate", "all"}:
        summary = evaluate_jobs(
            jobs,
            output_dir=output_dir,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
            bootstrap_repetitions=args.bootstrap_repetitions,
            bootstrap_seed=args.bootstrap_seed,
            make_plots=not args.skip_plots,
        )
        for workflow, result in summary["recommendations"].items():
            print(
                f"{workflow}: selected "
                f"{tuple(result['selected_targets_gev'])} GeV; "
                f"validation AUC={result['validation_auc']:.6f}; "
                f"delta baseline="
                f"{result['validation_auc_difference_from_baseline']:+.6f}; "
                f"decision={result['decision']}"
            )
        print(f"Report: {output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
