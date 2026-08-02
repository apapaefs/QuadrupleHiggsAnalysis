#!/usr/bin/env python3
"""Extract versioned AK8/Soft-Drop resonance hypothesis features."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
from typing import Sequence


FEATURE_SET = "fatjet-ak8-softdrop-v1"
METHOD_VERSION = FEATURE_SET
PREPROCESSING_VERSION = "fatjet-ak8-preprocessing-v1"
SMEARING_MODEL_ID = "cms-energy-uniform-fourvector-v1"
DEFAULT_FEATURE_BASE = Path("ResonanceAnalysis/features/ak8-v1")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _seed(label: str) -> int:
    digest = hashlib.sha256(f"{FEATURE_SET}\0{label}".encode()).hexdigest()
    return 14101983 + int(digest[:8], 16) % 700000000


def _summary_path(output: Path) -> Path:
    return output.with_suffix(".analysis_summary.json")


def _validate_feature_pair(
    sample_id: str,
    input_path: Path,
    output: Path,
    c_mistags: int,
    light_mistags: int,
    max_events: int | None,
    max_reco_jets: int,
    no_smear: bool,
) -> dict[str, object]:
    summary_path = _summary_path(output)
    if not output.is_file() or output.stat().st_size == 0 or not summary_path.is_file():
        raise RuntimeError(f"{sample_id}: incomplete ROOT/JSON feature pair")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"{sample_id}: unreadable {summary_path}: {error}") from error
    events_available = int(summary.get("events_available", -1))
    expected_events = min(events_available, max_events) if max_events else events_available
    smearing = summary.get("smearing", {})
    fatjet = summary.get("fatjet_definition", {})
    diagnostics = summary.get("diagnostics", {})
    checks = {
        "schema": summary.get("schema") == FEATURE_SET,
        "method": summary.get("method_version") == METHOD_VERSION,
        "preprocessing": summary.get("preprocessing_version") == PREPROCESSING_VERSION,
        "input": Path(str(summary.get("input", ""))).resolve() == input_path.resolve(),
        "events": int(summary.get("events_requested", -1)) == expected_events,
        "c_mistags": int(summary.get("c_mistags", -1)) == c_mistags,
        "light_mistags": int(summary.get("light_mistags", -1)) == light_mistags,
        "max_reco_jets": int(summary.get("max_reco_true_bjets", -1)) == max_reco_jets,
        "no_preapplied_tags": summary.get("tag_efficiencies_applied") is False,
        "smearing": bool(smearing.get("enabled")) == (not no_smear),
        "seed": int(smearing.get("seed", -1)) == _seed(sample_id),
        "smearing_model": smearing.get("model_id") == SMEARING_MODEL_ID,
        "coherent_fourvector": smearing.get("fourvector_scaling") == "uniform_correlated",
        "coherent_grooming": smearing.get("correlated_groomed_ungroomed_scaling") is True,
        "one_draw": int(smearing.get("gaussian_draws_per_physical_jet", -1)) == 1,
        "ak8_threshold": math.isclose(
            float(smearing.get("ak8_pt_threshold_gev", math.nan)), 300.0
        ),
        "ak8_definition": (
            fatjet.get("algorithm") == "anti-kt"
            and math.isclose(float(fatjet.get("R", math.nan)), 0.8)
            and math.isclose(float(fatjet.get("softdrop_beta", math.nan)), 0.0)
            and math.isclose(float(fatjet.get("softdrop_zcut", math.nan)), 0.1)
            and int(fatjet.get("max_retained_candidates", -1)) == 4
            and fatjet.get("true_double_b_multiplicities") == [2, 3]
        ),
        "nominal_probability_closure": abs(
            float(diagnostics.get("max_pattern_probability_residual_nominal", math.inf))
        ) <= 1.0e-12,
        "conservative_probability_closure": abs(
            float(diagnostics.get("max_pattern_probability_residual_conservative", math.inf))
        ) <= 1.0e-12,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    input_counter = summary.get("input_counter", {})
    reconstructable = summary.get("reconstructable_counter", {})
    hypotheses = summary.get("hypothesis_row_counter", {})
    try:
        counters = [
            (int(counter["events"]), float(counter["sumw"]), float(counter["sumw2"]))
            for counter in (input_counter, reconstructable, hypotheses)
        ]
    except (KeyError, TypeError, ValueError):
        counters = []
    if len(counters) != 3 or any(
        events < 0 or not math.isfinite(sumw) or not math.isfinite(sumw2) or sumw2 < 0.0
        for events, sumw, sumw2 in counters
    ):
        failed.append("finite_counters")
    if counters and counters[0][0] != expected_events:
        failed.append("input_event_counter")
    if failed:
        raise RuntimeError(
            f"{sample_id}: existing feature pair violates the {FEATURE_SET} contract: "
            + ", ".join(sorted(set(failed)))
        )
    return summary


def _signal_jobs(
    root: Path,
    manifest: Path,
    output_base: Path,
    topology: str,
    only: set[str],
    higgs_mass: float,
) -> list[dict[str, object]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"scenario", "miota_GeV", "meta_GeV", "run_name", "output_root"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{manifest} is missing columns: {', '.join(sorted(missing))}")
    jobs: list[dict[str, object]] = []
    for row in rows:
        scenario = row["scenario"].strip().lower()
        run_name = row["run_name"].strip()
        if (topology != "all" and scenario != topology) or (only and run_name not in only):
            continue
        m3 = float(row["miota_GeV"])
        if m3 <= 4.0 * higgs_mass:
            raise SystemExit(f"{run_name}: M3={m3:g} must satisfy M3 > 4 mh")
        if scenario == "cascade":
            m2 = float(row["meta_GeV"])
            if m2 <= 2.0 * higgs_mass or m3 <= 2.0 * m2:
                raise SystemExit(f"{run_name}: cascade hierarchy is not satisfied")
        elif scenario != "direct":
            raise SystemExit(f"unsupported signal scenario {scenario!r}")
        jobs.append(
            {
                "id": run_name,
                "kind": "signal",
                "input": _resolve(root, row["output_root"]),
                "output": (output_base / scenario / f"{run_name}_fatjet.root").resolve(),
                "c_mistags": 0,
                "light_mistags": 0,
            }
        )
    return jobs


def _background_jobs(
    root: Path, manifest: Path, output_base: Path, only: set[str]
) -> list[dict[str, object]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "raw_root", "c_mistags", "light_mistags"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{manifest} is missing columns: {', '.join(sorted(missing))}")
    jobs: list[dict[str, object]] = []
    for row in rows:
        sample_id = row["sample_id"].strip()
        if only and sample_id not in only:
            continue
        if str(row.get("optional", "false")).strip().lower() in {"1", "true", "yes"}:
            raw = _resolve(root, row["raw_root"])
            if not raw.is_file():
                continue
        jobs.append(
            {
                "id": sample_id,
                "kind": row.get("role", "background"),
                "input": _resolve(root, row["raw_root"]),
                "output": (output_base / f"{sample_id}_fatjet.root").resolve(),
                "c_mistags": int(row.get("c_mistags", 0) or 0),
                "light_mistags": int(row.get("light_mistags", 0) or 0),
            }
        )
    return jobs


def _versioned_manifest_text(
    source: Path, analysis_root: Path, output_base: Path
) -> str:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "root_file" not in fields:
        raise RuntimeError(f"{source} has no root_file column")
    for row in rows:
        target = (output_base / f"{row['sample_id'].strip()}_fatjet.root").resolve()
        try:
            row["root_file"] = str(target.relative_to(analysis_root))
        except ValueError:
            row["root_file"] = str(target)
        # A feature manifest describes the persisted feature pair, rather than
        # the potentially larger raw campaign from which it was extracted.
        # Keep the strict entry check aligned with the immutable extractor
        # summary (some legacy backgrounds intentionally used a validated raw
        # subset).  The source manifest itself is never modified.
        summary_path = target.with_suffix(".analysis_summary.json")
        if "generated_events" in fields and target.is_file() and summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                row["generated_events"] = str(int(summary["input_counter"]["events"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"{summary_path}: cannot resolve the feature input event count"
                ) from error
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_immutable(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"refusing to replace incompatible file {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_job(
    job: dict[str, object],
    executable: Path,
    log_dir: Path,
    max_events: int | None,
    max_reco_jets: int,
    no_smear: bool,
    dry_run: bool,
) -> dict[str, object]:
    sample_id = str(job["id"])
    input_path = Path(job["input"])
    output = Path(job["output"])
    record: dict[str, object] = {
        "id": sample_id,
        "kind": job["kind"],
        "input": str(input_path),
        "output": str(output),
    }
    if output.exists() or _summary_path(output).exists():
        _validate_feature_pair(
            sample_id,
            input_path,
            output,
            int(job["c_mistags"]),
            int(job["light_mistags"]),
            max_events,
            max_reco_jets,
            no_smear,
        )
        record["status"] = "kept_existing"
        return record
    if not input_path.is_file():
        raise FileNotFoundError(f"{sample_id}: missing AK8 raw ROOT file {input_path}")
    command = [
        str(executable),
        str(input_path),
        "--output",
        str(output),
        "--max-reco-jets",
        str(max_reco_jets),
        "--c-mistags",
        str(job["c_mistags"]),
        "--light-mistags",
        str(job["light_mistags"]),
        "--seed",
        str(_seed(sample_id)),
    ]
    if max_events is not None:
        command.extend(["--max-events", str(max_events)])
    if no_smear:
        command.append("--no-smear")
    record["command"] = command
    if dry_run:
        record["status"] = "would_run"
        return record
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{sample_id}.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False, text=True
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{sample_id}: extractor failed; see {log_path}")
    _validate_feature_pair(
        sample_id,
        input_path,
        output,
        int(job["c_mistags"]),
        int(job["light_mistags"]),
        max_events,
        max_reco_jets,
        no_smear,
    )
    record.update(status="complete", log=str(log_path), bytes=output.stat().st_size)
    return record


def build_parser() -> argparse.ArgumentParser:
    code_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", choices=(FEATURE_SET,), default=FEATURE_SET)
    parser.add_argument("--analysis-root", type=Path, default=code_dir.parent)
    parser.add_argument("--kind", choices=("all", "signals", "backgrounds"), default="all")
    parser.add_argument("--topology", choices=("all", "direct", "cascade"), default="all")
    parser.add_argument(
        "--signal-manifest",
        type=Path,
        default=Path("HerwigSignalPoints/mass_scan_10k_ak8-v1/manifest.csv"),
    )
    parser.add_argument(
        "--background-manifest",
        type=Path,
        default=Path("ResonanceAnalysis/background_manifest_ak8-v1.csv"),
    )
    parser.add_argument("--signal-output-dir", type=Path, default=DEFAULT_FEATURE_BASE)
    parser.add_argument(
        "--background-output-dir",
        type=Path,
        default=DEFAULT_FEATURE_BASE / "backgrounds",
    )
    parser.add_argument(
        "--resolved-background-manifest",
        type=Path,
        default=Path("ResonanceAnalysis/background_manifest_ak8-v1_features.csv"),
    )
    parser.add_argument(
        "--executable", type=Path, default=Path("Code/FourHiggsFatJetAnalysis")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--max-reco-jets", type=int, default=10)
    parser.add_argument("--higgs-mass", type=float, default=125.0)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--no-smear", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1 or not 4 <= args.max_reco_jets <= 10:
        raise SystemExit("--workers must be positive and --max-reco-jets must be in [4,10]")
    root = args.analysis_root.expanduser().resolve()
    only = set(args.only)
    jobs: list[dict[str, object]] = []
    background_manifest: Path | None = None
    background_output = _resolve(root, args.background_output_dir)
    if args.kind in {"all", "signals"}:
        jobs.extend(
            _signal_jobs(
                root,
                _resolve(root, args.signal_manifest),
                _resolve(root, args.signal_output_dir),
                args.topology,
                only,
                args.higgs_mass,
            )
        )
    if args.kind in {"all", "backgrounds"}:
        background_manifest = _resolve(root, args.background_manifest)
        jobs.extend(_background_jobs(root, background_manifest, background_output, only))
    if not jobs:
        raise SystemExit("no AK8 feature jobs selected")
    executable = _resolve(root, args.executable)
    if not args.dry_run and not executable.is_file():
        raise SystemExit(
            f"extractor not found: {executable}; build with make -C Code FourHiggsFatJetAnalysis"
        )
    records: list[dict[str, object]] = []
    failures: list[str] = []
    log_dir = root / "ResonanceAnalysis/logs/features/ak8-v1"
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = {
            pool.submit(
                _run_job,
                job,
                executable,
                log_dir,
                args.max_events,
                args.max_reco_jets,
                args.no_smear,
                args.dry_run,
            ): str(job["id"])
            for job in jobs
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                record = future.result()
                records.append(record)
                print(f"[{record['status']}] {sample_id}", flush=True)
            except Exception as error:
                failures.append(f"{sample_id}: {error}")
                print(f"[failed] {sample_id}: {error}", flush=True)
    records.sort(key=lambda item: (str(item["kind"]), str(item["id"])))
    if not args.dry_run:
        status = root / "ResonanceAnalysis/feature_campaign_status_ak8-v1.json"
        status.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_set": FEATURE_SET,
            "method_version": METHOD_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "smearing_model_id": SMEARING_MODEL_ID,
            "samples": records,
            "last_run_failures": failures,
        }
        status.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Status: {status}")
        if background_manifest is not None:
            target = _resolve(root, args.resolved_background_manifest)
            _write_immutable(
                target,
                _versioned_manifest_text(background_manifest, root, background_output),
            )
            print(f"AK8 feature background manifest: {target}")
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
