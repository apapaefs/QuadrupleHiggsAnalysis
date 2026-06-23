#!/usr/bin/env python3
"""Prepare a Sherpa example run directory with MPI-aware total event counts."""

import argparse
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "gg4b2c2j": ROOT / "Examples" / "GluonFusion_GG_2bbbar_ccbar_2j_LHE",
    "gg4b4c": ROOT / "Examples" / "GluonFusion_GG_2bbbar_2ccbar_LHE",
    "gg4b4j": ROOT / "Examples" / "GluonFusion_GG_2bbbar_4j_LHE",
    "gg6b2j": ROOT / "Examples" / "GluonFusion_GG_3bbbar_2j_LHE",
    "gg6bcc": ROOT / "Examples" / "GluonFusion_GG_3bbbar_ccbar_LHE",
    "gg8b": ROOT / "Examples" / "GluonFusion_GG_4bbbar_LHE",
    "z6b": ROOT / "Examples" / "PP_Z_6bbbar_Zbb_DecayOS_LHE",
}


def replace_card_value(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*.*$", re.MULTILINE)
    replacement = f"{key}: {value}"
    if not pattern.search(text):
        raise SystemExit(f"Could not find '{key}:' in Sherpa.yaml")
    return pattern.sub(replacement, text, count=1)


def read_card_value(text: str, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(.*)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise SystemExit(f"Could not find '{key}:' in Sherpa.yaml")
    return match.group(1).strip()


def upsert_card_value(text: str, key: str, value: str, after_key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*.*$", re.MULTILINE)
    replacement = f"{key}: {value}"
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1)

    anchor = re.compile(rf"^({re.escape(after_key)}:\s*.*)$", re.MULTILINE)
    if not anchor.search(text):
        raise SystemExit(f"Could not find '{after_key}:' in Sherpa.yaml")
    return anchor.sub(rf"\1\n{replacement}", text, count=1)


def enforce_mpi_run_defaults(text: str) -> str:
    text = upsert_card_value(text, "MPI_EVENT_MODE", "1", "EVENTS")
    text = upsert_card_value(text, "MPI_SEED_MODE", "1", "MPI_EVENT_MODE")
    text = upsert_card_value(text, "BATCH_MODE", "5", "MPI_SEED_MODE")
    return upsert_card_value(text, "EVENT_DISPLAY_INTERVAL", "100", "BATCH_MODE")


def replace_event_output(text: str, prefix: str) -> str:
    pattern = re.compile(r"^EVENT_OUTPUT:\s*LHEF\[[^\]]+\]\s*$", re.MULTILINE)
    replacement = f"EVENT_OUTPUT: LHEF[{prefix}]"
    if not pattern.search(text):
        raise SystemExit("Could not find 'EVENT_OUTPUT: LHEF[...]' in Sherpa.yaml")
    return pattern.sub(replacement, text, count=1)


def read_event_output_prefix(text: str) -> str:
    pattern = re.compile(r"^EVENT_OUTPUT:\s*LHEF\[([^\]]+)\]\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise SystemExit("Could not find 'EVENT_OUTPUT: LHEF[...]' in Sherpa.yaml")
    return match.group(1)


def copy_example(example_dir: Path, run_dir: Path, force: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        if not force:
            raise SystemExit(f"{run_dir} exists and is not empty; pass --force to overwrite Sherpa.yaml")
        shutil.copy2(example_dir / "Sherpa.yaml", run_dir / "Sherpa.yaml")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    for path in example_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, run_dir / path.name)


def bash_default(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )


def write_seeded_generation_runner(
    run_dir: Path,
    total_events: int,
    jobs: int,
    output_prefix: str,
    result_directory: str,
) -> None:
    runner = run_dir / "run_seeded_generation.sh"
    runner.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

TOTAL_EVENTS="${{1:-{total_events}}}"
JOBS="${{2:-{jobs}}}"
CARD="${{CARD:-Sherpa.yaml}}"
OUTBASE="${{OUTBASE:-events}}"
OUTPUT_STEM="${{OUTPUT_STEM:-{bash_default(output_prefix)}}}"
BASE_SEED="${{BASE_SEED:-1234}}"
RESULT_DIRECTORY="${{RESULT_DIRECTORY:-{bash_default(result_directory)}}}"

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$TOTAL_EVENTS" -lt 1 ]; then
  echo "TOTAL_EVENTS must be positive" >&2
  exit 2
fi

if [ "$JOBS" -lt 1 ]; then
  echo "JOBS must be positive" >&2
  exit 2
fi

if [ ! -e "$CARD" ]; then
  echo "Missing $CARD" >&2
  exit 2
fi

if [ ! -e Process ] && [ ! -e Process.zip ]; then
  echo "Missing Process/ or Process.zip. Run the integration in this directory first." >&2
  exit 2
fi

if [ ! -e "$RESULT_DIRECTORY" ] && [ ! -e "${{RESULT_DIRECTORY}}.zip" ]; then
  echo "Missing $RESULT_DIRECTORY or $RESULT_DIRECTORY.zip. Run the integration in this directory first." >&2
  exit 2
fi

mkdir -p "$OUTBASE"
pids=()

base_events=$((TOTAL_EVENTS / JOBS))
remainder=$((TOTAL_EVENTS % JOBS))

copy_if_missing() {{
  local src="$1"
  local dst="$workdir/$(basename "$src")"
  if [ -e "$src" ] && [ ! -e "$dst" ]; then
    cp -a "$src" "$workdir/"
  fi
}}

for idx in $(seq 1 "$JOBS"); do
  events_this="$base_events"
  if [ "$idx" -le "$remainder" ]; then
    events_this=$((events_this + 1))
  fi
  if [ "$events_this" -eq 0 ]; then
    continue
  fi

  seed=$((BASE_SEED * idx))
  workdir=$(printf "%s/job_%04d" "$OUTBASE" "$idx")
  mkdir -p "$workdir"
  cp "$CARD" "$workdir/Sherpa.yaml"
  copy_if_missing Process
  copy_if_missing Process.zip
  copy_if_missing "$RESULT_DIRECTORY"
  copy_if_missing "${{RESULT_DIRECTORY}}.zip"

  (
    cd "$workdir"
    Sherpa Sherpa.yaml \\
      "EVENTS: $events_this" \\
      "MPI_EVENT_MODE: 0" \\
      "MPI_SEED_MODE: 0" \\
      "RANDOM_SEED: $seed" \\
      "EVENT_OUTPUT: LHEF[${{OUTPUT_STEM}}_${{seed}}]" \\
      > "sherpa_${{seed}}.log" 2>&1
  ) &
  pids+=("$!")
done

status=0
for pid in "${{pids[@]}}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  echo "At least one Sherpa shard failed. Check $OUTBASE/job_*/sherpa_*.log" >&2
  exit "$status"
fi

echo "Requested $TOTAL_EVENTS events across $JOBS single-rank Sherpa jobs under $OUTBASE"
"""
    )
    runner.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("example", choices=sorted(EXAMPLES))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--total-events", type=int, help="desired total events over all MPI ranks")
    parser.add_argument("--np", type=int, default=1, help="MPI rank count")
    parser.add_argument("--output-prefix", help="LHEF output prefix")
    parser.add_argument(
        "--seeded-jobs",
        type=int,
        help="write run_seeded_generation.sh with this default number of single-rank jobs",
    )
    parser.add_argument("--round-up", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true", help="overwrite Sherpa.yaml in an existing run directory")
    args = parser.parse_args()

    if args.np < 1:
        raise SystemExit("--np must be positive")
    if args.total_events is not None and args.total_events < 1:
        raise SystemExit("--total-events must be positive")
    if args.seeded_jobs is not None and args.seeded_jobs < 1:
        raise SystemExit("--seeded-jobs must be positive")

    example_dir = EXAMPLES[args.example]
    copy_example(example_dir, args.run_dir, args.force)

    card = args.run_dir / "Sherpa.yaml"
    text = card.read_text()
    text = enforce_mpi_run_defaults(text)

    if args.total_events is not None:
        text = replace_card_value(text, "EVENTS", str(args.total_events))

    if args.output_prefix:
        text = replace_event_output(text, args.output_prefix)

    card.write_text(text)

    card_total_events = int(read_card_value(text, "EVENTS"))
    output_prefix = read_event_output_prefix(text)
    result_directory = read_card_value(text, "RESULT_DIRECTORY")
    if args.seeded_jobs is not None:
        write_seeded_generation_runner(
            args.run_dir,
            card_total_events,
            args.seeded_jobs,
            output_prefix,
            result_directory,
        )

    print(f"Prepared {args.example} run in {args.run_dir}")
    if args.total_events is not None:
        print(f"EVENTS total: {args.total_events}")
        print(f"MPI_EVENT_MODE: 1, so Sherpa distributes this total over -np {args.np}")
    print("MPI_SEED_MODE: 1")
    print("BATCH_MODE: 5")
    print("EVENT_DISPLAY_INTERVAL: 100")
    if args.seeded_jobs is not None:
        print("Integration command before seeded generation:")
        print(f"  cd {args.run_dir}")
        print("  Sherpa -I Sherpa.yaml")
        print(f"  mpirun --use-hwthread-cpus -np {args.np} --bind-to hwthread --map-by hwthread Sherpa -e 0 Sherpa.yaml")
        print("Seeded single-rank generation command:")
        print(f"  ./run_seeded_generation.sh {card_total_events} {args.seeded_jobs}")
    else:
        print("Run command:")
        print(f"  mpirun --use-hwthread-cpus -np {args.np} --bind-to hwthread --map-by hwthread Sherpa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
