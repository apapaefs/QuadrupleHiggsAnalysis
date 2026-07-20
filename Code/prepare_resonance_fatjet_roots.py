#!/usr/bin/env python3
"""Regenerate non-overwriting HwSim AK8 raw samples for resonance analyses."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Sequence


CAMPAIGN = "ak8-v1"
FATJET_SETTINGS = {
    "FatJets": "Yes",
    "RFatParameter": "0.8",
    "PTCutFatJets": "150.0",
    "EtaCutFatJets": "6.0",
    "FatSoftDropBeta": "0.0",
    "FatSoftDropZCut": "0.1",
}
AK8_BRANCHES = (
    "numFatJets",
    "theFatJets",
    "theSoftDropFatJets",
    "tau21FatJets",
    "bHadronMultiplicityFatJets",
    "cHadronMultiplicityFatJets",
)


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _seed(sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return 271000 + int(digest[:8], 16) % 700000000


def _load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return fields, rows


def _write_immutable(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"refusing to replace incompatible file {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _manifest_text(fields: list[str], rows: list[dict[str, str]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _validate_ak8_root(path: Path, expected_events: int, audit_events: int = 1000) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty ROOT output {path}")
    try:
        import ROOT  # type: ignore
    except ImportError as error:
        raise RuntimeError("PyROOT is required to validate AK8 raw samples") from error
    root_file = ROOT.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"unreadable ROOT output {path}")
    tree = root_file.Get("Data")
    entries = int(tree.GetEntries()) if tree else -1
    missing = [name for name in AK8_BRANCHES if not tree or not tree.GetBranch(name)]
    if entries != expected_events or missing:
        root_file.Close()
        if entries != expected_events:
            raise RuntimeError(f"{path}: expected {expected_events} entries, found {entries}")
        raise RuntimeError(f"{path}: missing AK8 branches {', '.join(missing)}")
    checked = min(entries, max(0, audit_events))
    for entry in range(checked):
        tree.GetEntry(entry)
        count = int(tree.numFatJets)
        if count < 0 or count > 100:
            root_file.Close()
            raise RuntimeError(f"{path}: invalid numFatJets={count} at entry {entry}")
        previous_pt = math.inf
        for index in range(count):
            energy = float(tree.theFatJets[index])
            px = float(tree.theFatJets[100 + index])
            py = float(tree.theFatJets[200 + index])
            pz = float(tree.theFatJets[300 + index])
            sd = [
                float(tree.theSoftDropFatJets[axis * 100 + index])
                for axis in range(4)
            ]
            tau21 = float(tree.tau21FatJets[index])
            multiplicities = (
                int(tree.bHadronMultiplicityFatJets[index]),
                int(tree.cHadronMultiplicityFatJets[index]),
            )
            values = [energy, px, py, pz, tau21, *sd]
            pt = math.hypot(px, py)
            if not all(math.isfinite(value) for value in values):
                root_file.Close()
                raise RuntimeError(f"{path}: non-finite AK8 value at entry {entry}")
            if pt > previous_pt + 1.0e-10:
                root_file.Close()
                raise RuntimeError(f"{path}: AK8 branches are not pT ordered at entry {entry}")
            if any(value < 0 for value in multiplicities):
                root_file.Close()
                raise RuntimeError(f"{path}: negative flavour multiplicity at entry {entry}")
            previous_pt = pt
    root_file.Close()


def _setup_text(work: Path) -> str:
    lines = [f"set /Herwig/Analysis/HwSim:OutputLocation {work.as_posix()}/"]
    lines.extend(
        f"set /Herwig/Analysis/HwSim:{name} {value}"
        for name, value in FATJET_SETTINGS.items()
    )
    return "\n".join(lines) + "\n"


def _signal_records(
    root: Path, manifest: Path, topology: str, only: set[str]
) -> tuple[list[dict[str, object]], str]:
    fields, rows = _load_rows(manifest)
    if not {"scenario", "run_name", "card", "lhe", "events", "output_root"}.issubset(fields):
        raise RuntimeError(f"{manifest}: incomplete signal manifest")
    target_base = root / "HerwigSignalPoints/mass_scan_10k_ak8-v1"
    records: list[dict[str, object]] = []
    updated: list[dict[str, str]] = []
    for original in rows:
        row = dict(original)
        scenario = row["scenario"].strip().lower()
        sample_id = row["run_name"].strip()
        target = target_base / scenario / "events" / f"{sample_id}.root"
        row["output_root"] = str(target.relative_to(root))
        updated.append(row)
        if (topology != "all" and scenario != topology) or (only and sample_id not in only):
            continue
        source_run = _resolve(root, Path(row["card"]).with_suffix(".run"))
        records.append(
            {
                "sample_id": sample_id,
                "kind": "signal",
                "source_run": source_run,
                "source_lhe": _resolve(root, row["lhe"]),
                "target": target,
                "events": int(float(row["events"])),
                "seed": int(row.get("seed") or _seed(sample_id)),
            }
        )
    return records, _manifest_text(fields, updated)


def _background_records(
    root: Path, manifest: Path, only: set[str]
) -> tuple[list[dict[str, object]], str]:
    fields, rows = _load_rows(manifest)
    required = {"sample_id", "source_run", "source_lhe", "generated_events", "raw_root"}
    if not required.issubset(fields):
        raise RuntimeError(f"{manifest}: incomplete background manifest")
    updated: list[dict[str, str]] = []
    records: list[dict[str, object]] = []
    for original in rows:
        row = dict(original)
        sample_id = row["sample_id"].strip()
        target = root / "ResonanceAnalysis/raw_backgrounds_ak8-v1" / f"{sample_id}-ak8-v1.root"
        row["raw_root"] = str(target.relative_to(root))
        updated.append(row)
        if only and sample_id not in only:
            continue
        source_run = _resolve(root, row["source_run"])
        source_lhe = _resolve(root, row["source_lhe"])
        if _as_bool(row.get("optional", False)) and (
            not source_run.is_file() or not source_lhe.is_file()
        ):
            continue
        records.append(
            {
                "sample_id": sample_id,
                "kind": row.get("role", "background"),
                "source_run": source_run,
                "source_lhe": source_lhe,
                "target": target,
                "events": int(float(row["generated_events"])),
                "seed": _seed(sample_id),
            }
        )
    return records, _manifest_text(fields, updated)


def _run_one(
    record: dict[str, object],
    work_root: Path,
    herwig: str,
    dry_run: bool,
    audit_events: int,
) -> dict[str, object]:
    sample_id = str(record["sample_id"])
    source_run = Path(record["source_run"])
    source_lhe = Path(record["source_lhe"])
    target = Path(record["target"])
    events = int(record["events"])
    result = dict(record)
    result.update(
        source_run=str(source_run), source_lhe=str(source_lhe), target=str(target)
    )
    if target.exists():
        if not dry_run:
            _validate_ak8_root(target, events, audit_events)
        result["status"] = "kept_existing"
        return result
    for source in (source_run, source_lhe):
        if not source.is_file():
            raise FileNotFoundError(f"{sample_id}: missing source {source}")
    if dry_run:
        result["status"] = "would_run"
        return result
    target.parent.mkdir(parents=True, exist_ok=True)
    logs = target.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{sample_id}-", dir=work_root))
    keep_work = True
    try:
        local_run = work / source_run.name
        local_lhe = work / source_lhe.name
        shutil.copy2(source_run, local_run)
        shutil.copy2(source_lhe, local_lhe)
        setup = work / f"resonance-{CAMPAIGN}.setup"
        setup.write_text(_setup_text(work), encoding="utf-8")
        prefix = source_run.stem
        before = set(work.glob(f"{prefix}*.root"))
        command = [
            herwig,
            "run",
            f"--numevents={events}",
            f"--seed={int(record['seed'])}",
            f"--tag={CAMPAIGN}",
            f"--setupfile={setup.name}",
            local_run.name,
        ]
        log_path = logs / f"{sample_id}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=work,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        result.update(command=command, log=str(log_path), returncode=completed.returncode)
        if completed.returncode != 0:
            raise RuntimeError(f"{sample_id}: Herwig failed; see {log_path}")
        produced = sorted(set(work.glob(f"{prefix}*.root")) - before)
        if len(produced) != 1:
            raise RuntimeError(f"{sample_id}: expected one new ROOT output, found {len(produced)}")
        _validate_ak8_root(produced[0], events, audit_events)
        if target.exists():
            raise FileExistsError(f"refusing to replace {target}")
        shutil.move(str(produced[0]), target)
        _validate_ak8_root(target, events, audit_events)
        result.update(status="complete", bytes=target.stat().st_size)
        keep_work = False
        return result
    finally:
        result["work_directory"] = str(work)
        if not keep_work:
            shutil.rmtree(work)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=root)
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
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--work-root", type=Path, default=Path(os.getenv("TMPDIR", "/tmp")) / "4h-ak8-v1")
    parser.add_argument("--herwig", default="Herwig")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--audit-events", type=int, default=1000)
    parser.add_argument(
        "--smoke-events",
        type=int,
        help="Run only N events per sample into the separate smoke directory",
    )
    parser.add_argument(
        "--smoke-output-dir",
        type=Path,
        default=Path("ResonanceAnalysis/smoke/ak8-v1/raw"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.workers < 1
        or args.audit_events < 0
        or (args.smoke_events is not None and args.smoke_events < 1)
    ):
        raise SystemExit("--workers must be positive and --audit-events non-negative")
    if not args.dry_run and shutil.which(args.herwig) is None:
        raise SystemExit(f"Herwig executable not found: {args.herwig}")
    root = args.analysis_root.expanduser().resolve()
    only = set(args.only)
    records: list[dict[str, object]] = []
    signal_manifest_text: str | None = None
    background_manifest_text: str | None = None
    if args.kind in {"all", "signals"}:
        selected, signal_manifest_text = _signal_records(
            root, _resolve(root, args.signal_manifest), args.topology, only
        )
        records.extend(selected)
    if args.kind in {"all", "backgrounds"}:
        selected, background_manifest_text = _background_records(
            root, _resolve(root, args.background_manifest), only
        )
        records.extend(selected)
    if not records:
        raise SystemExit("no AK8 regeneration jobs selected")
    if args.smoke_events is not None:
        smoke_base = _resolve(root, args.smoke_output_dir)
        for record in records:
            record["events"] = min(int(record["events"]), args.smoke_events)
            record["target"] = (
                smoke_base
                / str(record["kind"])
                / f"{record['sample_id']}-ak8-v1-smoke.root"
            )
    completed_records: list[dict[str, object]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(records))) as pool:
        futures = {
            pool.submit(
                _run_one,
                record,
                args.work_root.expanduser().resolve(),
                args.herwig,
                args.dry_run,
                args.audit_events,
            ): str(record["sample_id"])
            for record in records
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                result = future.result()
                completed_records.append(result)
                print(f"[{result['status']}] {sample_id}", flush=True)
            except Exception as error:
                failures.append(f"{sample_id}: {error}")
                print(f"[failed] {sample_id}: {error}", flush=True)
    completed_records.sort(key=lambda item: (str(item["kind"]), str(item["sample_id"])))
    if not args.dry_run:
        if signal_manifest_text is not None and args.smoke_events is None:
            _write_immutable(
                root / "HerwigSignalPoints/mass_scan_10k_ak8-v1/manifest.csv",
                signal_manifest_text,
            )
        if background_manifest_text is not None and args.smoke_events is None:
            _write_immutable(
                root / "ResonanceAnalysis/background_manifest_ak8-v1.csv",
                background_manifest_text,
            )
        if args.smoke_events is not None:
            smoke_output = _resolve(root, args.smoke_output_dir)
            status = smoke_output.with_name(f"{smoke_output.name}_status.json")
        else:
            status = root / "ResonanceAnalysis/raw_ak8-v1_status.json"
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(
            json.dumps(
                {
                    "campaign": CAMPAIGN,
                    "smoke_events": args.smoke_events,
                    "fatjet_settings": FATJET_SETTINGS,
                    "samples": completed_records,
                    "last_run_failures": failures,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Status: {status}")
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
