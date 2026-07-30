#!/usr/bin/env python3
"""Build hybrid four-Higgs feature trees for the mass scan and backgrounds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


METHOD_VERSION = "resonance-hybrid-v1.3-baseline-mass-targets"
PREPROCESSING_VERSION = "resonance-preprocessing-v2"
SMEARING_MODEL_ID = "cms-energy-uniform-fourvector-v1"
DEFAULT_HIGGS_MASS_TARGETS_GEV = (120.0, 115.0, 110.0, 105.0)
MASS_TARGET_PROFILE_ID = "baseline-120-115-110-105"
DEFAULT_FEATURE_BASE = (
    Path("ResonanceAnalysis/features")
    / SMEARING_MODEL_ID
    / MASS_TARGET_PROFILE_ID
)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _seed(label: str) -> int:
    return 14101983 + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) % 700000000


def _summary_path(output: Path) -> Path:
    return output.with_suffix(".analysis_summary.json")


def _validate_feature_pair(
    sample_id: str,
    input_path: Path,
    output: Path,
    summary_path: Path,
    c_mistags: int,
    light_mistags: int,
    max_events: int | None,
    max_reco_jets: int,
    no_smear: bool,
) -> dict[str, object]:
    if not output.is_file() or output.stat().st_size == 0 or not summary_path.is_file():
        raise RuntimeError(f"{sample_id}: incomplete ROOT/JSON feature pair")
    try:
        summary = json.loads(summary_path.read_text())
    except Exception as error:
        raise RuntimeError(f"{sample_id}: unreadable summary {summary_path}: {error}") from error
    expected_seed = _seed(sample_id)
    expected_events = int(summary.get("events_available", -1))
    if max_events is not None:
        expected_events = min(expected_events, max_events)
    try:
        mass_targets = tuple(
            float(value) for value in summary.get("higgs_mass_targets_gev", ())
        )
    except (TypeError, ValueError):
        mass_targets = ()
    checks = {
        "schema": summary.get("schema") == "resonance-hybrid-v1",
        "method_version": summary.get("method_version") == METHOD_VERSION,
        "preprocessing_version": summary.get("preprocessing_version") == PREPROCESSING_VERSION,
        "higgs_mass_targets": mass_targets == DEFAULT_HIGGS_MASS_TARGETS_GEV,
        "higgs_mass_target_assignment": (
            summary.get("higgs_mass_target_assignment")
            == "candidate_pt_rank_descending"
        ),
        "input": Path(str(summary.get("input", ""))).resolve() == input_path.resolve(),
        "events_requested": int(summary.get("events_requested", -1)) == expected_events,
        "c_mistags": int(summary.get("c_mistags", -1)) == c_mistags,
        "light_mistags": int(summary.get("light_mistags", -1)) == light_mistags,
        "max_reco_jets": int(summary.get("max_reco_true_bjets", -1)) == max_reco_jets,
        "smearing": bool(summary.get("smearing", {}).get("enabled")) == (not no_smear),
        "seed": int(summary.get("smearing", {}).get("seed", -1)) == expected_seed,
        "smearing_preprocessing_version": (
            summary.get("smearing", {}).get("preprocessing_version") == PREPROCESSING_VERSION
        ),
        "smearing_model_id": summary.get("smearing", {}).get("model_id") == SMEARING_MODEL_ID,
        "fourvector_scaling": (
            summary.get("smearing", {}).get("fourvector_scaling") == "uniform_correlated"
        ),
        "correlated_mass_scaling": (
            summary.get("smearing", {}).get("correlated_mass_scaling") is True
        ),
        "mass_not_fixed": summary.get("smearing", {}).get("preserves_jet_mass") is False,
        "one_gaussian_draw": (
            int(summary.get("smearing", {}).get("gaussian_draws_per_jet", -1)) == 1
        ),
        "energy_floor": math.isclose(
            float(summary.get("smearing", {}).get("energy_floor_gev", math.nan)),
            1.0e-6,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "eta_preselection": (
            summary.get("smearing", {}).get("eta_preselection")
            == "finite |eta|<2.5 before smearing"
        ),
        "pt_threshold": (
            summary.get("smearing", {}).get("pt_threshold") == "smeared pT>20 GeV"
        ),
        "smear_before_pt_threshold": (
            summary.get("smearing", {}).get("smear_before_pt_threshold") is True
        ),
        "acceptance_order": (
            summary.get("smearing", {}).get("acceptance_order")
            == "raw_abs_eta_then_smear_then_smeared_pt"
        ),
        "migration_diagnostics": all(
            isinstance(summary.get("diagnostics", {}).get(name), int)
            and summary["diagnostics"][name] >= 0
            for name in (
                "true_b_upward_pt_migrations",
                "true_b_downward_pt_migrations",
                "non_b_upward_pt_migrations",
                "non_b_downward_pt_migrations",
            )
        ),
        "upward_migration_pt_bins": all(
            isinstance(summary.get("diagnostics", {}).get(name), dict)
            and set(summary["diagnostics"][name])
            == {"[10,12)", "[12,15)", "[15,20]"}
            and all(
                isinstance(value, int) and value >= 0
                for value in summary["diagnostics"][name].values()
            )
            for name in (
                "true_b_upward_pt_migrations_by_raw_pt_gev",
                "non_b_upward_pt_migrations_by_raw_pt_gev",
            )
        ),
        "upward_migration_pt_bin_sums": all(
            isinstance(summary.get("diagnostics", {}).get(total_name), int)
            and isinstance(summary.get("diagnostics", {}).get(bins_name), dict)
            and all(
                isinstance(value, int)
                for value in summary["diagnostics"][bins_name].values()
            )
            and sum(summary["diagnostics"][bins_name].values())
            == summary["diagnostics"][total_name]
            for total_name, bins_name in (
                (
                    "true_b_upward_pt_migrations",
                    "true_b_upward_pt_migrations_by_raw_pt_gev",
                ),
                (
                    "non_b_upward_pt_migrations",
                    "non_b_upward_pt_migrations_by_raw_pt_gev",
                ),
            )
        ),
        "mass_scaling_diagnostic": math.isfinite(
            float(
                summary.get("diagnostics", {}).get(
                    "max_smearing_mass_scaling_residual_gev", math.nan
                )
            )
        ),
        "tag_efficiencies": summary.get("tag_efficiencies_applied") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    reco = summary.get("reconstructable_counter", {})
    categories = summary.get("categories", {})
    nmerged = summary.get("n_merged", {})
    reco_events = int(reco.get("events", -1))
    category_events = sum(int(item.get("events", 0)) for item in categories.values())
    nmerged_events = sum(int(item.get("events", 0)) for item in nmerged.values())
    if reco_events < 0 or reco_events != category_events or reco_events != nmerged_events:
        failed.append("category_closure")
    for counter in (summary.get("input_counter", {}), reco):
        if not all(math.isfinite(float(counter.get(key, math.nan))) for key in ("sumw", "sumw2")):
            failed.append("finite_weights")
            break
    if failed:
        raise RuntimeError(
            f"{sample_id}: existing feature pair does not match this campaign: "
            + ", ".join(sorted(set(failed)))
        )
    return summary


def _signal_jobs(
    analysis_root: Path,
    manifest: Path,
    output_base: Path,
    topology: str,
    only: set[str],
    higgs_mass: float,
) -> list[dict[str, object]]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"scenario", "miota_GeV", "meta_GeV", "run_name", "output_root"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{manifest} is missing columns: {', '.join(sorted(missing))}")

    jobs: list[dict[str, object]] = []
    for row in rows:
        scenario = row["scenario"].strip().lower()
        if topology != "all" and scenario != topology:
            continue
        run_name = row["run_name"].strip()
        if only and run_name not in only:
            continue
        m3 = float(row["miota_GeV"])
        if m3 <= 4.0 * higgs_mass:
            raise SystemExit(f"{run_name}: M3={m3:g} does not satisfy M3 > 4 mh")
        if scenario == "cascade":
            m2 = float(row["meta_GeV"])
            if m2 <= 2.0 * higgs_mass:
                raise SystemExit(f"{run_name}: M2={m2:g} does not satisfy M2 > 2 mh")
            if m3 <= 2.0 * m2:
                raise SystemExit(f"{run_name}: M3={m3:g} does not satisfy M3 > 2 M2")
        elif scenario != "direct":
            raise SystemExit(f"unsupported signal scenario {scenario!r}")
        jobs.append(
            {
                "id": run_name,
                "kind": "signal",
                "input": _resolve(analysis_root, row["output_root"]).resolve(),
                "output": (output_base / scenario / f"{run_name}_resonance.root").resolve(),
                "c_mistags": 0,
                "light_mistags": 0,
            }
        )
    return jobs


def _background_jobs(
    analysis_root: Path,
    manifest: Path,
    output_base: Path,
    only: set[str],
) -> list[dict[str, object]]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "root_file", "raw_root", "c_mistags", "light_mistags"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{manifest} is missing columns: {', '.join(sorted(missing))}")
    jobs: list[dict[str, object]] = []
    for row in rows:
        sample_id = row["sample_id"].strip()
        if only and sample_id not in only:
            continue
        jobs.append(
            {
                "id": sample_id,
                "kind": row.get("role", "background"),
                "input": _resolve(analysis_root, row["raw_root"]).resolve(),
                "output": (output_base / f"{sample_id}_resonance.root").resolve(),
                "c_mistags": int(row.get("c_mistags", 0) or 0),
                "light_mistags": int(row.get("light_mistags", 0) or 0),
            }
        )
    return jobs


def _versioned_background_manifest_text(
    source_manifest: Path,
    analysis_root: Path,
    output_base: Path,
) -> str:
    analysis_root = analysis_root.resolve()
    with source_manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "root_file" not in fieldnames:
        raise RuntimeError(f"{source_manifest} has no root_file column")
    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        if not sample_id:
            raise RuntimeError(f"{source_manifest} contains a row without sample_id")
        output = (output_base / f"{sample_id}_resonance.root").resolve()
        try:
            row["root_file"] = str(output.relative_to(analysis_root))
        except ValueError:
            row["root_file"] = str(output)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_versioned_background_manifest(
    source_manifest: Path,
    target_manifest: Path,
    analysis_root: Path,
    output_base: Path,
) -> None:
    content = _versioned_background_manifest_text(
        source_manifest, analysis_root, output_base
    )
    if target_manifest.exists():
        if target_manifest.read_text() != content:
            raise RuntimeError(
                f"refusing to replace incompatible resolved manifest {target_manifest}"
            )
        return
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(content, encoding="utf-8")


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
    summary = _summary_path(output)
    record = {
        "id": sample_id,
        "kind": job["kind"],
        "input": str(input_path),
        "output": str(output),
        "summary": str(summary),
    }
    if output.exists() or summary.exists():
        _validate_feature_pair(
            sample_id,
            input_path,
            output,
            summary,
            int(job["c_mistags"]),
            int(job["light_mistags"]),
            max_events,
            max_reco_jets,
            no_smear,
        )
        record["status"] = "kept_existing"
        return record
    if not input_path.is_file():
        raise FileNotFoundError(f"{sample_id}: missing raw ROOT file {input_path}")
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
        "--higgs-mass-targets",
        ",".join(f"{target:g}" for target in DEFAULT_HIGGS_MASS_TARGETS_GEV),
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
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    record.update(returncode=completed.returncode, log=str(log_path))
    if completed.returncode != 0:
        raise RuntimeError(f"{sample_id}: extractor failed; see {log_path}")
    _validate_feature_pair(
        sample_id,
        input_path,
        output,
        summary,
        int(job["c_mistags"]),
        int(job["light_mistags"]),
        max_events,
        max_reco_jets,
        no_smear,
    )
    record.update(status="complete", bytes=output.stat().st_size)
    return record


def main() -> int:
    code_dir = Path(__file__).resolve().parent
    default_root = code_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=default_root)
    parser.add_argument("--kind", choices=("all", "signals", "backgrounds"), default="all")
    parser.add_argument("--topology", choices=("all", "direct", "cascade"), default="all")
    parser.add_argument(
        "--signal-manifest",
        type=Path,
        default=Path("HerwigSignalPoints/mass_scan_10k/manifest.csv"),
    )
    parser.add_argument(
        "--background-manifest",
        type=Path,
        default=Path("ResonanceAnalysis/background_manifest.csv"),
    )
    parser.add_argument(
        "--signal-output-dir",
        type=Path,
        default=DEFAULT_FEATURE_BASE,
    )
    parser.add_argument(
        "--background-output-dir",
        type=Path,
        default=DEFAULT_FEATURE_BASE / "backgrounds",
    )
    parser.add_argument(
        "--resolved-background-manifest",
        type=Path,
        help=(
            "Non-overwriting manifest with root_file paths redirected to the versioned "
            "background feature directory; defaults beside the input manifest"
        ),
    )
    parser.add_argument("--executable", type=Path, default=Path("Code/FourHiggsResonanceAnalysis"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--max-reco-jets", type=int, default=10)
    parser.add_argument("--higgs-mass", type=float, default=125.0)
    parser.add_argument("--only", action="append", default=[], help="Run/sample id; repeat as needed")
    parser.add_argument("--no-smear", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or not 4 <= args.max_reco_jets <= 10:
        raise SystemExit("--workers must be positive and --max-reco-jets must be in [4, 10]")

    root = args.analysis_root.expanduser().resolve()
    resolve_arg = lambda path: path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()
    jobs: list[dict[str, object]] = []
    only = set(args.only)
    background_manifest: Path | None = None
    resolved_background_manifest: Path | None = None
    background_output_dir = resolve_arg(args.background_output_dir)
    if args.kind in {"all", "signals"}:
        jobs.extend(
            _signal_jobs(
                root,
                resolve_arg(args.signal_manifest),
                resolve_arg(args.signal_output_dir),
                args.topology,
                only,
                args.higgs_mass,
            )
        )
    if args.kind in {"all", "backgrounds"}:
        background_manifest = resolve_arg(args.background_manifest)
        if args.resolved_background_manifest is None:
            resolved_background_manifest = background_manifest.with_name(
                f"{background_manifest.stem}_{SMEARING_MODEL_ID}_"
                f"{MASS_TARGET_PROFILE_ID}.csv"
            )
        else:
            resolved_background_manifest = resolve_arg(args.resolved_background_manifest)
        jobs.extend(
            _background_jobs(root, background_manifest, background_output_dir, only)
        )
    if not jobs:
        raise SystemExit("no feature jobs selected")

    executable = resolve_arg(args.executable)
    if not args.dry_run and not executable.is_file():
        raise SystemExit(f"extractor not found: {executable}; build it with make -C Code FourHiggsResonanceAnalysis")
    records: list[dict[str, object]] = []
    failures: list[str] = []
    log_dir = (
        root
        / "ResonanceAnalysis"
        / "logs"
        / "features"
        / SMEARING_MODEL_ID
        / MASS_TARGET_PROFILE_ID
    )
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
            except Exception as error:  # independent samples continue
                failures.append(f"{sample_id}: {error}")
                print(f"[failed] {sample_id}: {error}", flush=True)

    records.sort(key=lambda item: (str(item["kind"]), str(item["id"])))
    if not args.dry_run:
        status = (
            root
            / "ResonanceAnalysis"
            / (
                f"feature_campaign_status_{SMEARING_MODEL_ID}_"
                f"{MASS_TARGET_PROFILE_ID}.json"
            )
        )
        status.parent.mkdir(parents=True, exist_ok=True)
        previous_records: list[dict[str, object]] = []
        if status.is_file():
            try:
                previous_records = list(json.loads(status.read_text()).get("samples", []))
            except Exception:
                previous_records = []
        merged = {
            (str(item.get("kind", "")), str(item.get("id", ""))): item
            for item in previous_records
        }
        for item in records:
            merged[(str(item["kind"]), str(item["id"]))] = item
        merged_records = sorted(
            merged.values(), key=lambda item: (str(item.get("kind", "")), str(item.get("id", "")))
        )
        status_payload = {
            "method_version": METHOD_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "smearing_model_id": SMEARING_MODEL_ID,
            "mass_target_profile_id": MASS_TARGET_PROFILE_ID,
            "higgs_mass_targets_gev": list(DEFAULT_HIGGS_MASS_TARGETS_GEV),
            "samples": merged_records,
            "last_run_failures": failures,
        }
        if resolved_background_manifest is not None:
            status_payload["resolved_background_manifest"] = str(
                resolved_background_manifest
            )
        status.write_text(json.dumps(status_payload, indent=2) + "\n")
        print(f"Status: {status}")
    if failures:
        raise SystemExit("; ".join(failures))
    if background_manifest is not None and resolved_background_manifest is not None:
        if args.dry_run:
            print(f"Resolved background manifest would be: {resolved_background_manifest}")
        else:
            _write_versioned_background_manifest(
                background_manifest,
                resolved_background_manifest,
                root,
                background_output_dir,
            )
            print(f"Resolved background manifest: {resolved_background_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
