#!/usr/bin/env python3
"""Regenerate branch-complete HwSim background ROOT files without overwriting samples.

The legacy background ROOT files predate ``bHadronMultiplicity``.  The hybrid
resolved/merged analysis deliberately refuses to guess that information, so
this helper reruns the existing, serialized Herwig runs into a new campaign
directory.  Existing outputs are always kept and treated as completed work.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _seed(sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return 271000 + int(digest[:8], 16) % 700000000


def _count_lhe_events(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "rb") as handle:
        for line in handle:
            if b"<event" in line:
                count += 1
    return count


def _validate_bhadron_root(path: Path, expected_events: int) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty ROOT output {path}")
    try:
        import ROOT  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "PyROOT is required to validate regenerated backgrounds; source the ROOT environment"
        ) from error
    root_file = ROOT.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"unreadable ROOT output {path}")
    tree = root_file.Get("Data")
    entries = int(tree.GetEntries()) if tree else -1
    has_branch = bool(tree and tree.GetBranch("bHadronMultiplicity"))
    root_file.Close()
    if entries != expected_events:
        raise RuntimeError(f"{path}: expected {expected_events} Data entries, found {entries}")
    if not has_branch:
        raise RuntimeError(f"{path}: Data tree is missing bHadronMultiplicity")


def _read_rows(manifest: Path, only: set[str]) -> list[dict[str, str]]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "raw_root", "source_run", "source_lhe", "generated_events"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{manifest} is missing columns: {', '.join(sorted(missing))}")
    return [
        row
        for row in rows
        if _as_bool(row.get("regenerate", "true"))
        and (not only or row["sample_id"] in only)
    ]


def _run_one(
    row: dict[str, str],
    analysis_root: Path,
    work_root: Path,
    herwig: str,
    dry_run: bool,
) -> dict[str, object]:
    sample_id = row["sample_id"]
    target = _resolve(analysis_root, row["raw_root"]).resolve()
    source_run = _resolve(analysis_root, row["source_run"]).resolve()
    source_lhe = _resolve(analysis_root, row["source_lhe"]).resolve()
    events = int(float(row["generated_events"]))
    seed = _seed(sample_id)

    record: dict[str, object] = {
        "sample_id": sample_id,
        "target": str(target),
        "source_run": str(source_run),
        "source_lhe": str(source_lhe),
        "events": events,
        "seed": seed,
    }
    if target.exists():
        if not dry_run:
            _validate_bhadron_root(target, events)
        record["status"] = "kept_existing"
        return record
    for source in (source_run, source_lhe):
        if not source.is_file():
            raise FileNotFoundError(f"{sample_id}: missing input {source}")
    observed_lhe_events = _count_lhe_events(source_lhe)
    declared_lhe_events = int(float(row.get("lhe_event_count") or observed_lhe_events))
    if observed_lhe_events != declared_lhe_events:
        raise RuntimeError(
            f"{sample_id}: LHE contains {observed_lhe_events} events, manifest declares "
            f"{declared_lhe_events}"
        )
    hard_event_policy = row.get("hard_event_policy", "no_hard_event_reuse").strip()
    # Keep one statistically independent shower history per hard LHE event.
    # Recycling would let the same hard event leak across XGBoost folds and
    # would make the per-entry MC statistical uncertainty too optimistic.
    if events > observed_lhe_events:
        raise RuntimeError(
            f"{sample_id}: requested {events} showers from {observed_lhe_events} hard events "
            "but hard-event recycling is forbidden for the resonance analysis"
        )
    record.update(
        lhe_events=observed_lhe_events,
        hard_event_policy=hard_event_policy,
        hard_event_reuse_factor=events / observed_lhe_events,
    )
    if dry_run:
        record["status"] = "would_run"
        return record

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

        # This setup changes only HwSim's output directory.  The run itself and
        # the original ROOT sample remain untouched.
        setup = work / "resonance-bhadmult-v1.setup"
        setup.write_text(
            "set /Herwig/Analysis/HwSim:OutputLocation "
            f"{work.as_posix()}/\n",
            encoding="utf-8",
        )
        prefix = source_run.stem
        before = set(work.glob(f"{prefix}*.root"))
        command = [
            herwig,
            "run",
            f"--numevents={events}",
            f"--seed={seed}",
            "--tag=bhadmult-v1",
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
                check=False,
                text=True,
            )
        record.update(command=command, log=str(log_path), returncode=completed.returncode)
        if completed.returncode != 0:
            raise RuntimeError(f"{sample_id}: Herwig failed; see {log_path}")

        after = set(work.glob(f"{prefix}*.root"))
        produced = sorted(after - before)
        if len(produced) != 1:
            raise RuntimeError(
                f"{sample_id}: expected one new {prefix}*.root, found {len(produced)}"
            )
        if target.exists():
            raise FileExistsError(f"refusing to replace {target}")
        _validate_bhadron_root(produced[0], events)
        shutil.move(str(produced[0]), target)
        _validate_bhadron_root(target, events)
        record.update(status="complete", bytes=target.stat().st_size)
        keep_work = False
        return record
    finally:
        record["work_directory"] = str(work)
        if not keep_work:
            shutil.rmtree(work)


def main() -> int:
    code_dir = Path(__file__).resolve().parent
    default_root = code_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=default_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_root / "ResonanceAnalysis" / "background_manifest.csv",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--work-root", type=Path, default=Path(os.getenv("TMPDIR", "/tmp")) / "4h-resonance-backgrounds")
    parser.add_argument("--herwig", default="Herwig")
    parser.add_argument("--only", action="append", default=[], help="Sample id to run; repeat as needed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    analysis_root = args.analysis_root.expanduser().resolve()
    manifest = args.manifest.expanduser()
    if not manifest.is_absolute():
        manifest = analysis_root / manifest
    rows = _read_rows(manifest.resolve(), set(args.only))
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if not rows:
        raise SystemExit("no regeneratable background rows selected")
    if not args.dry_run and shutil.which(args.herwig) is None:
        raise SystemExit(f"Herwig executable not found: {args.herwig}")

    records: list[dict[str, object]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(rows))) as pool:
        futures = {
            pool.submit(
                _run_one,
                row,
                analysis_root,
                args.work_root.expanduser().resolve(),
                args.herwig,
                args.dry_run,
            ): row["sample_id"]
            for row in rows
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                record = future.result()
                records.append(record)
                print(f"[{record['status']}] {sample_id}", flush=True)
            except Exception as error:  # retain other independent jobs
                failures.append(f"{sample_id}: {error}")
                print(f"[failed] {sample_id}: {error}", flush=True)

    records.sort(key=lambda item: str(item["sample_id"]))
    status_file = analysis_root / "ResonanceAnalysis" / "background_regeneration_status.json"
    if not args.dry_run:
        status_file.parent.mkdir(parents=True, exist_ok=True)
        previous_records: list[dict[str, object]] = []
        if status_file.is_file():
            try:
                previous_records = list(json.loads(status_file.read_text()).get("samples", []))
            except Exception:
                previous_records = []
        merged = {str(item.get("sample_id", "")): item for item in previous_records}
        for item in records:
            merged[str(item["sample_id"])] = item
        status_file.write_text(
            json.dumps(
                {
                    "samples": [merged[key] for key in sorted(merged)],
                    "last_run_failures": failures,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Status: {status_file}")
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
