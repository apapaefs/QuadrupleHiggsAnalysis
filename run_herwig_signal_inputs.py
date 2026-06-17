#!/usr/bin/env python3
"""Run Herwig over a list of signal input files."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path


COMPLETE_MARKER = "Number of events that pass basic cuts"


def timestamp() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def read_inputs(list_path: Path, limit: int | None) -> list[Path]:
    base = Path.cwd()
    inputs: list[Path] = []
    with list_path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            if not path.is_absolute():
                path = base / path
            inputs.append(path.resolve())
            if limit is not None and len(inputs) >= limit:
                break
    return inputs


def tagged_name(run_name: str, tag: str) -> str:
    return f"{run_name}-{tag}" if tag else run_name


def is_complete(output_root: Path, run_log: Path, input_path: Path | None = None) -> bool:
    if not output_root.exists() or output_root.stat().st_size == 0:
        return False
    if not run_log.exists():
        return False
    if input_path is not None:
        input_mtime = input_path.stat().st_mtime
        if output_root.stat().st_mtime < input_mtime or run_log.stat().st_mtime < input_mtime:
            return False
    return COMPLETE_MARKER in run_log.read_text(errors="replace")


def append_header(log_path: Path, command: list[str], cwd: Path) -> None:
    with log_path.open("a") as log:
        log.write(f"\n[{timestamp()}] cwd: {cwd}\n")
        log.write(f"[{timestamp()}] command: {' '.join(command)}\n")
        log.flush()


def run_command(command: list[str], cwd: Path, log_path: Path) -> int:
    append_header(log_path, command, cwd)
    with log_path.open("a") as log:
        proc = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"[{timestamp()}] exit_code: {proc.returncode}\n")
        return proc.returncode


def run_one(input_path: Path, args: argparse.Namespace) -> tuple[str, str, float]:
    start = time.monotonic()
    if not input_path.exists():
        return (input_path.name, "missing_input", 0.0)

    workdir = input_path.parent
    run_name = input_path.stem
    log_dir = workdir / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    run_file = workdir / f"{run_name}.run"
    output_root = workdir / "events" / f"{tagged_name(run_name, args.tag)}.root"
    read_log = log_dir / f"{run_name}.read.log"
    run_log = log_dir / f"{tagged_name(run_name, args.tag)}.run.log"

    if not args.force and is_complete(output_root, run_log, input_path):
        return (run_name, "skipped_complete", time.monotonic() - start)

    read_needed = args.force_read or not run_file.exists()
    if run_file.exists() and not args.force_read:
        read_needed = run_file.stat().st_mtime < input_path.stat().st_mtime

    if read_needed:
        read_code = run_command([args.herwig, "read", input_path.name], workdir, read_log)
        if read_code != 0:
            return (run_name, f"read_failed:{read_code}", time.monotonic() - start)

    command = [args.herwig, "run"]
    if args.numevents is not None:
        command.extend(["-N", str(args.numevents)])
    if args.tag:
        command.extend(["-t", args.tag])
    command.append(run_file.name)

    run_code = run_command(command, workdir, run_log)
    elapsed = time.monotonic() - start
    if run_code != 0:
        return (run_name, f"run_failed:{run_code}", elapsed)
    if not is_complete(output_root, run_log):
        return (run_name, "run_finished_without_complete_marker", elapsed)
    return (run_name, "complete", elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        type=Path,
        default=Path("HerwigSignalPoints/c3d4_10k/herwig_inputs_to_run.txt"),
        help="Text file with one Herwig .in path per line.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Number of Herwig inputs to run concurrently.")
    parser.add_argument("--limit", type=int, help="Run only the first N inputs from the list.")
    parser.add_argument("--numevents", type=int, help="Override the number of events passed to Herwig run.")
    parser.add_argument("--tag", default="", help="Optional Herwig output tag, useful for smoke tests.")
    parser.add_argument("--herwig", default="Herwig", help="Herwig executable.")
    parser.add_argument("--log-dir", default="logs", help="Per-input log directory, relative to each input dir.")
    parser.add_argument("--force", action="store_true", help="Run even when the output appears complete.")
    parser.add_argument("--force-read", action="store_true", help="Recreate .run files even when they are current.")
    parser.add_argument("--dry-run", action="store_true", help="Print the inputs that would run and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = read_inputs(args.list, args.limit)
    if not inputs:
        print(f"No inputs found in {args.list}", file=sys.stderr)
        return 1

    print(f"[{timestamp()}] loaded {len(inputs)} input(s) from {args.list}", flush=True)
    if args.dry_run:
        for path in inputs:
            print(path)
        return 0

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_input = {executor.submit(run_one, path, args): path for path in inputs}
        for index, future in enumerate(concurrent.futures.as_completed(future_to_input), start=1):
            run_name, status, elapsed = future.result()
            if status not in {"complete", "skipped_complete"}:
                failures += 1
            print(
                f"[{timestamp()}] {index}/{len(inputs)} {run_name}: {status} ({elapsed:.1f}s)",
                flush=True,
            )

    if failures:
        print(f"[{timestamp()}] finished with {failures} failure(s)", file=sys.stderr, flush=True)
        return 1
    print(f"[{timestamp()}] finished successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
