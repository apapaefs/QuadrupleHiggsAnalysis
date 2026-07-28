#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: run_herwig_c3d4_40k.sh [options]

Prepare the repository-standard HwSim Herwig cards for the 153-point c3/d4
40k LHE campaign and run them in parallel.

Options:
  --jobs N            Concurrent Herwig jobs (default: 32, or
                      HERWIG_C3D4_JOBS).
  --prepare-only      Validate the LHE transfer and prepare cards, then stop.
  --dry-run           Prepare cards and print the 153 inputs without running.
  --limit N           Run only the first N prepared inputs.
  --tag TAG           Add a Herwig output tag (recommended for smoke tests).
  --numevents N       Override Herwig's event count. A non-40000 value requires
                      --tag so a smoke test cannot replace production output.
  --force             Rerun inputs whose output is already complete.
  --force-read        Rebuild existing Herwig .run files from their .in cards.
  --skip-checksums    Skip the transferred-file SHA-256 check.
  -h, --help          Show this help.

Examples:
  ./Signals/c3d4_40k/run_herwig_parallel.sh --prepare-only
  ./Signals/c3d4_40k/run_herwig_parallel.sh --jobs 32
  ./Signals/c3d4_40k/run_herwig_parallel.sh --jobs 1 --limit 1 \
    --numevents 10 --tag smoke
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
POINTS_FILE="$SCRIPT_DIR/metadata/points_153.csv"
CHECKSUM_FILE="$SCRIPT_DIR/metadata/source_sha256.txt"
HERWIG_OUTDIR="$REPO_DIR/HerwigSignalPoints/c3d4_40k"
HERWIG_MANIFEST="$HERWIG_OUTDIR/herwig_inputs_manifest.csv"
HERWIG_INPUT_LIST="$HERWIG_OUTDIR/herwig_inputs_to_run.txt"
HERWIG_MODULE="${HERWIG_MODULE:-herwig/stable-full-py3-rivet4}"
JOBS="${HERWIG_C3D4_JOBS:-32}"
PREPARE_ONLY=0
DRY_RUN=0
SKIP_CHECKSUMS=0
LIMIT=""
TAG=""
NUMEVENTS=""
FORCE=0
FORCE_READ=0

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    printf 'Missing value for %s\n' "$1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      require_value "$@"
      JOBS="$2"
      shift 2
      ;;
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-checksums)
      SKIP_CHECKSUMS=1
      shift
      ;;
    --limit)
      require_value "$@"
      LIMIT="$2"
      shift 2
      ;;
    --tag)
      require_value "$@"
      TAG="$2"
      shift 2
      ;;
    --numevents)
      require_value "$@"
      NUMEVENTS="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --force-read)
      FORCE_READ=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for numeric_value in "$JOBS" ${LIMIT:+"$LIMIT"} ${NUMEVENTS:+"$NUMEVENTS"}; do
  if [[ ! "$numeric_value" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Expected a positive integer, got: %s\n' "$numeric_value" >&2
    exit 2
  fi
done

ONLINE_CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
if (( JOBS > ONLINE_CPUS )); then
  printf 'Refusing --jobs %d on a host with %d online CPUs.\n' "$JOBS" "$ONLINE_CPUS" >&2
  exit 2
fi
if [[ -n "$NUMEVENTS" && "$NUMEVENTS" != "40000" && -z "$TAG" ]]; then
  printf 'A non-production --numevents value requires --tag.\n' >&2
  exit 2
fi
if (( PREPARE_ONLY && DRY_RUN )); then
  printf -- '--prepare-only and --dry-run are mutually exclusive.\n' >&2
  exit 2
fi

if [[ ! -f "$POINTS_FILE" ]]; then
  printf 'Missing point manifest: %s\n' "$POINTS_FILE" >&2
  exit 1
fi
if [[ ! -f "$REPO_DIR/4h_analyzer.py" || ! -f "$REPO_DIR/run_herwig_signal_inputs.py" ]]; then
  printf 'Repository Herwig helpers are missing under %s\n' "$REPO_DIR" >&2
  exit 1
fi

python3 - "$SCRIPT_DIR" "$POINTS_FILE" <<'PY'
import csv
import sys
from pathlib import Path

signal_dir = Path(sys.argv[1])
points_file = Path(sys.argv[2])
with points_file.open(newline="") as handle:
    rows = list(csv.DictReader(handle))

required = {"run_name", "target_events", "c3", "d4"}
missing_columns = required - set(rows[0] if rows else ())
errors = []
if missing_columns:
    errors.append("point manifest missing columns: " + ", ".join(sorted(missing_columns)))
expected_names = [row.get("run_name", "") for row in rows]
if len(rows) != 153:
    errors.append(f"point manifest has {len(rows)} rows, expected 153")
if len(set(expected_names)) != len(expected_names):
    errors.append("point manifest contains duplicate run names")
for row in rows:
    if row.get("target_events") != "40000":
        errors.append(f"{row.get('run_name')}: target_events={row.get('target_events')}")

events_dir = signal_dir / "Events"
actual_dirs = {
    path.name: path
    for path in events_dir.glob("run_gg_4h_*")
    if path.is_dir()
}
expected_set = set(expected_names)
missing = sorted(expected_set - set(actual_dirs))
extra = sorted(set(actual_dirs) - expected_set)
if missing:
    errors.append("missing run directories: " + ", ".join(missing))
if extra:
    errors.append("unexpected run directories: " + ", ".join(extra))

for name in sorted(expected_set & set(actual_dirs)):
    run_dir = actual_dirs[name]
    lhe = run_dir / "unweighted_events.lhe.gz"
    banners = list(run_dir.glob("*_banner.txt"))
    if not lhe.is_file() or lhe.stat().st_size == 0:
        errors.append(f"{name}: missing or empty unweighted_events.lhe.gz")
    if len(banners) != 1:
        errors.append(f"{name}: expected one banner, found {len(banners)}")

if errors:
    for error in errors:
        print("ERROR:", error, file=sys.stderr)
    raise SystemExit(1)
print("LHE inventory preflight: 153/153 expected run directories present")
PY

if (( ! SKIP_CHECKSUMS )); then
  if [[ ! -f "$CHECKSUM_FILE" ]]; then
    printf 'Missing checksum manifest: %s\n' "$CHECKSUM_FILE" >&2
    exit 1
  fi
  printf 'Checking transferred LHE and banner SHA-256 values...\n'
  (
    cd "$SCRIPT_DIR"
    sha256sum --check --strict --quiet "$CHECKSUM_FILE"
  )
  printf 'Transferred-file checksums: OK\n'
fi

if [[ -f /etc/profile.d/modules.sh ]]; then
  # Always refresh the function and MODULEPATH: exported shell functions from
  # an SSH client can otherwise leave a callable but unusable `module`.
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
elif [[ -f /usr/share/Modules/init/bash ]]; then
  # shellcheck disable=SC1091
  source /usr/share/Modules/init/bash
elif ! type module >/dev/null 2>&1; then
  printf 'The environment-modules initialization script was not found.\n' >&2
  exit 1
fi
if [[ -d /usr/share/modules/modulefiles ]]; then
  module use /usr/share/modules/modulefiles
fi
module load "$HERWIG_MODULE"
command -v Herwig >/dev/null
printf 'Using %s (%s)\n' "$(command -v Herwig)" "$(Herwig --version 2>&1 | head -1)"

mkdir -p "$HERWIG_OUTDIR"
cd "$REPO_DIR"
python3 "$REPO_DIR/4h_analyzer.py" \
  --prepare-herwig-inputs "$SCRIPT_DIR" \
  --herwig-template "$REPO_DIR/Signals/HW-gg_hhhh_SM.in" \
  --herwig-outdir "$HERWIG_OUTDIR" \
  --herwig-manifest "$HERWIG_MANIFEST" \
  --herwig-nevents 40000 \
  --herwig-required-generated-events 40000 \
  --herwig-output-location events/ \
  --herwig-run-prefix HW

python3 - "$HERWIG_MANIFEST" "$HERWIG_INPUT_LIST" <<'PY'
import csv
import re
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
input_list = Path(sys.argv[2])
with manifest.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
inputs = [
    Path(line.strip())
    for line in input_list.read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
errors = []
if len(rows) != 153:
    errors.append(f"Herwig manifest has {len(rows)} rows, expected 153")
if len(inputs) != 153 or len(set(inputs)) != 153:
    errors.append(f"Herwig input list has {len(inputs)} entries ({len(set(inputs))} unique), expected 153")
allowed_statuses = {"written", "overwritten", "skipped_existing"}
for row in rows:
    if row["status"] not in allowed_statuses:
        errors.append(f"{row['run_name']}: preparation status={row['status']} ({row['reason']})")
        continue
    card = Path(row["herwig_input"])
    if not card.is_file():
        errors.append(f"{row['run_name']}: missing card {card}")
        continue
    text = card.read_text(errors="replace")
    required_lines = (
        rf"^set\s+theLHReader:FileName\s+{re.escape(row['lhe_file'])}\s*$",
        r"^set\s+theGenerator:NumberOfEvents\s+40000\s*$",
        r"^set\s+/Herwig/Analysis/HwSim:OutputLocation\s+events/\s*$",
        rf"^saverun\s+{re.escape(row['run_name'])}\s+theGenerator\s*$",
    )
    for pattern in required_lines:
        if not re.search(pattern, text, flags=re.MULTILINE):
            errors.append(f"{row['run_name']}: card does not match {pattern}")
if errors:
    for error in errors:
        print("ERROR:", error, file=sys.stderr)
    raise SystemExit(1)
print("Herwig card preflight: 153/153 cards ready")
PY

if (( PREPARE_ONLY )); then
  printf 'Preparation complete; Herwig was not launched.\n'
  exit 0
fi

runner=(
  python3 "$REPO_DIR/run_herwig_signal_inputs.py"
  --list "$HERWIG_INPUT_LIST"
  --jobs "$JOBS"
)
[[ -n "$LIMIT" ]] && runner+=(--limit "$LIMIT")
[[ -n "$TAG" ]] && runner+=(--tag "$TAG")
[[ -n "$NUMEVENTS" ]] && runner+=(--numevents "$NUMEVENTS")
(( FORCE )) && runner+=(--force)
(( FORCE_READ )) && runner+=(--force-read)

if (( DRY_RUN )); then
  runner+=(--dry-run)
  "${runner[@]}"
  exit 0
fi

if command -v pgrep >/dev/null 2>&1; then
  ACTIVE_HERWIG="$(pgrep -fc '^Herwig (read|run) ' || true)"
  ACTIVE_HERWIG="${ACTIVE_HERWIG:-0}"
  if (( ACTIVE_HERWIG + JOBS > ONLINE_CPUS )); then
    printf 'WARNING: %d Herwig process(es) are already active; %d more jobs exceed %d online CPUs.\n' \
      "$ACTIVE_HERWIG" "$JOBS" "$ONLINE_CPUS" >&2
  fi
fi

mkdir -p "$HERWIG_OUTDIR/logs"
exec 9>"$HERWIG_OUTDIR/.herwig_parallel.lock"
if ! flock -n 9; then
  printf 'Another c3/d4 40k Herwig launcher holds %s\n' "$HERWIG_OUTDIR/.herwig_parallel.lock" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
CONTROLLER_LOG="$HERWIG_OUTDIR/logs/controller_$(date -u +%Y%m%dT%H%M%SZ).log"
printf 'Launching Herwig with jobs=%d; controller log: %s\n' "$JOBS" "$CONTROLLER_LOG"
"${runner[@]}" 2>&1 | tee -a "$CONTROLLER_LOG"
