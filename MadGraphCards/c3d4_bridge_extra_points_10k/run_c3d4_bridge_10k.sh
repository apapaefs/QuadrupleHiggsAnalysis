#!/usr/bin/env bash

# Generate the 60 new points in the regular c3/d4 bridge grid for gg -> hhhh.
# The three existing points (0,-100), (0,0), and (0,100) are intentionally
# omitted. Runs are sequential because a single MG5 process directory is not
# safe for concurrent card edits; each run can use MG5's multicore mode.

set -Eeuo pipefail

readonly SCRIPT_NAME="$(basename -- "$0")"
readonly PROCESS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly MG5_ROOT="$(cd -- "$PROCESS_DIR/.." && pwd)"
readonly HEPTOOLS_LIB="$MG5_ROOT/HEPTools/lib"
readonly PARAM_CARD="$PROCESS_DIR/Cards/param_card.dat"
readonly RUN_CARD="$PROCESS_DIR/Cards/run_card.dat"
readonly LOG_DIR="$PROCESS_DIR/bridge_scan_logs"
readonly MANIFEST="$PROCESS_DIR/bridge_scan_manifest.csv"

events=10000
cores=32
limit=0
dry_run=0
verify_only=0

usage() {
    cat <<EOF
Usage: ./$SCRIPT_NAME [options]

Generate the 60 missing points in
  c3 = {-12,-9,-6,-3,0,3,6,9,12}
  d4 = {-300,-200,-100,0,100,200,300}
with (0,-100), (0,0), and (0,100) omitted as already existing.

Options:
  --cores N       MG5 multicore workers for one point (default: 32)
  --events N      Unweighted events per point (default: 10000)
  --limit N       Process only the first N points in the balanced ordering
                  (use --limit 30 for an initial refinement stage)
  --dry-run       Print the selected points and commands without changing files
  --verify-only   Validate existing LHE files and rewrite the manifest
  -h, --help      Show this help

The script is resumable: an existing LHE file is skipped only after its event
count and embedded c3/d4 values have been validated. An incomplete or invalid
existing run is never overwritten automatically.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

while (($#)); do
    case "$1" in
        --cores)
            (($# >= 2)) || die "--cores requires an argument"
            cores="$2"
            shift 2
            ;;
        --events)
            (($# >= 2)) || die "--events requires an argument"
            events="$2"
            shift 2
            ;;
        --limit)
            (($# >= 2)) || die "--limit requires an argument"
            limit="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --verify-only)
            verify_only=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

is_positive_integer "$cores" || die "--cores must be a positive integer"
is_positive_integer "$events" || die "--events must be a positive integer"
[[ "$limit" =~ ^[0-9]+$ ]] || die "--limit must be a non-negative integer"

[[ -x "$PROCESS_DIR/bin/generate_events" ]] || \
    die "run this copy of the script from the gg_4h_c3d4 process directory"
[[ -f "$PARAM_CARD" ]] || die "missing $PARAM_CARD"
[[ -f "$RUN_CARD" ]] || die "missing $RUN_CARD"
command -v python3 >/dev/null || die "python3 is required"
[[ -d "$HEPTOOLS_LIB" ]] || die "missing MG5 HEPTools library directory: $HEPTOOLS_LIB"

# MadLoop links dynamically to the MG5-managed Ninja and COLLIER libraries.
# This generated process has no embedded RUNPATH, so make their installation
# directory available to both the initialization check and all event workers.
export LD_LIBRARY_PATH="$HEPTOOLS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# When a MadLoop check executable already exists, fail early with the actual
# unresolved libraries instead of MG5's generic "Failed initialization" error.
madloop_check="$PROCESS_DIR/SubProcesses/PV0_0_1_gg_hhhh/check"
if [[ -x "$madloop_check" ]] && command -v ldd >/dev/null; then
    missing_libraries=$(ldd "$madloop_check" 2>/dev/null | awk '/=> not found/ {print $1}')
    [[ -z "$missing_libraries" ]] || \
        die "MadLoop runtime libraries are unresolved: ${missing_libraries//$'\n'/, }"
fi

# Balanced order: the first 30 points span both signs and all three non-zero
# |d4| layers, making --limit 30 useful as an initial refinement pass. The
# remaining points then complete the regular bridge grid.
declare -a points=()
add_point() {
    local point="$1,$2"
    local existing
    for existing in "${points[@]-}"; do
        [[ "$existing" == "$point" ]] && return
    done
    points+=("$point")
}

# Initial 30-point refinement: 16 at |d4|=100, eight at |d4|=200,
# four at |d4|=300, and two on the c3=0 bridge.
for c3 in -3 3 -6 6 -9 9 -12 12; do
    for d4 in -100 100; do
        add_point "$c3" "$d4"
    done
done
for c3 in -6 6 -12 12; do
    for d4 in -200 200; do
        add_point "$c3" "$d4"
    done
done
for c3 in -9 9; do
    for d4 in -300 300; do
        add_point "$c3" "$d4"
    done
done
for d4 in -200 200; do
    add_point 0 "$d4"
done
((${#points[@]} == 30)) || die "internal initial-point-list error"

# Fill in every remaining position in the 63-point regular grid, omitting the
# three legacy points on c3=0 at d4=-100,0,100.
for c3 in -12 -9 -6 -3 0 3 6 9 12; do
    for d4 in -300 -200 -100 0 100 200 300; do
        if [[ "$c3" == 0 && ("$d4" == -100 || "$d4" == 0 || "$d4" == 100) ]]; then
            continue
        fi
        add_point "$c3" "$d4"
    done
done

((${#points[@]} == 60)) || die "internal point-list error: expected 60 points"
if ((limit > 0 && limit > ${#points[@]})); then
    die "--limit cannot exceed ${#points[@]}"
fi
selected_count=${#points[@]}
if ((limit > 0)); then
    selected_count=$limit
fi

format_value() {
    printf '%.1f' "$1"
}

run_name_for() {
    local c3="$1" d4="$2"
    printf 'run_gg_4h_4_%s_%s' "$(format_value "$c3")" "$(format_value "$d4")"
}

set_cards() {
    local c3="$1" d4="$2"
    python3 - "$PARAM_CARD" "$RUN_CARD" "$c3" "$d4" "$events" <<'PY'
import pathlib
import re
import sys

param_path = pathlib.Path(sys.argv[1])
run_path = pathlib.Path(sys.argv[2])
c3 = float(sys.argv[3])
d4 = float(sys.argv[4])
events = int(sys.argv[5])

lines = param_path.read_text().splitlines(keepends=True)
targets = {("TRIPCOUP", "4"): c3, ("QUARTCOUP", "6"): d4}
seen = set()
block = None
for i, line in enumerate(lines):
    match = re.match(r"\s*BLOCK\s+(\S+)", line, flags=re.IGNORECASE)
    if match:
        block = match.group(1).upper()
        continue
    data, separator, comment = line.partition("#")
    fields = data.split()
    key = (block, fields[0]) if fields else None
    if key not in targets:
        continue
    indent = re.match(r"\s*", line).group(0)
    suffix = f" #{comment.rstrip()}" if separator else ""
    lines[i] = f"{indent}{fields[0]} {targets[key]:.8e}{suffix}\n"
    seen.add(key)

if seen != set(targets):
    missing = sorted(set(targets) - seen)
    raise SystemExit(f"could not update param-card entries: {missing}")
param_path.write_text("".join(lines))

lines = run_path.read_text().splitlines(keepends=True)
count = 0
pattern = re.compile(r"^(\s*)\S+(\s*=\s*nevents\b.*)$", flags=re.IGNORECASE)
for i, line in enumerate(lines):
    match = pattern.match(line)
    if match:
        newline = "\n" if line.endswith("\n") else ""
        lines[i] = f"{match.group(1)}{events}{match.group(2).rstrip()}{newline}"
        count += 1
if count != 1:
    raise SystemExit(f"expected one nevents entry in run card, found {count}")
run_path.write_text("".join(lines))
PY
}

# Validate the event count and the parameter values embedded in the LHE header.
# On success, print "EVENT_COUNT,CROSS_SECTION_PB".
inspect_lhe() {
    local lhe="$1" expected_c3="$2" expected_d4="$3"
    python3 - "$lhe" "$expected_c3" "$expected_d4" "$events" <<'PY'
import gzip
import math
import re
import sys

path = sys.argv[1]
expected_c3 = float(sys.argv[2])
expected_d4 = float(sys.argv[3])
expected_events = int(sys.argv[4])

opener = gzip.open if path.endswith(".gz") else open
event_count = 0
block = None
values = {}
cross_section = None
number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
xsec_pattern = re.compile(r"Integrated weight \(pb\)\s*:\s*(" + number + r")")

with opener(path, "rt", errors="replace") as stream:
    for line in stream:
        stripped = line.strip()
        if stripped == "<event>":
            event_count += 1
        match = re.match(r"\s*BLOCK\s+(\S+)", line, flags=re.IGNORECASE)
        if match:
            block = match.group(1).upper()
            continue
        data = line.partition("#")[0].split()
        if block == "TRIPCOUP" and len(data) >= 2 and data[0] == "4":
            values["c3"] = float(data[1].replace("D", "E").replace("d", "e"))
        elif block == "QUARTCOUP" and len(data) >= 2 and data[0] == "6":
            values["d4"] = float(data[1].replace("D", "E").replace("d", "e"))
        match = xsec_pattern.search(line)
        if match:
            cross_section = float(match.group(1).replace("D", "E").replace("d", "e"))

errors = []
if event_count != expected_events:
    errors.append(f"events={event_count}, expected={expected_events}")
for name, expected in (("c3", expected_c3), ("d4", expected_d4)):
    actual = values.get(name)
    if actual is None or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"{name}={actual}, expected={expected}")
if cross_section is None:
    errors.append("missing integrated weight")
if errors:
    raise SystemExit("; ".join(errors))

print(f"{event_count},{cross_section:.12g}")
PY
}

write_manifest() {
    local tmp_manifest="$MANIFEST.tmp.$$"
    local complete=0 missing=0 invalid=0
    printf 'index,c3,d4,run_name,status,event_count,cross_section_pb,lhe_file,log_file\n' >"$tmp_manifest"

    local i point c3 d4 run_name lhe log details status event_count xsec rc
    for ((i = 0; i < selected_count; ++i)); do
        point=${points[$i]}
        IFS=, read -r c3 d4 <<<"$point"
        run_name=$(run_name_for "$c3" "$d4")
        lhe="$PROCESS_DIR/Events/$run_name/unweighted_events.lhe.gz"
        log="$LOG_DIR/$run_name.log"
        status=missing
        event_count=
        xsec=
        if [[ -f "$lhe" ]]; then
            set +e
            details=$(inspect_lhe "$lhe" "$c3" "$d4" 2>/dev/null)
            rc=$?
            set -e
            if ((rc == 0)); then
                status=complete
                IFS=, read -r event_count xsec <<<"$details"
            else
                status=invalid
            fi
        fi
        case "$status" in
            complete) ((++complete)) ;;
            missing) ((++missing)) ;;
            invalid) ((++invalid)) ;;
        esac
        printf '%d,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$((i + 1))" "$(format_value "$c3")" "$(format_value "$d4")" \
            "$run_name" "$status" "$event_count" "$xsec" "$lhe" "$log" \
            >>"$tmp_manifest"
    done
    mv -- "$tmp_manifest" "$MANIFEST"
    printf 'Manifest: %s (%d complete, %d missing, %d invalid)\n' \
        "$MANIFEST" "$complete" "$missing" "$invalid"
    ((missing == 0 && invalid == 0))
}

if ((dry_run)); then
    printf 'Process directory: %s\nEvents per point: %d\nCores per point: %d\nSelected points: %d/%d\n' \
        "$PROCESS_DIR" "$events" "$cores" "$selected_count" "${#points[@]}"
    for ((i = 0; i < selected_count; ++i)); do
        IFS=, read -r c3 d4 <<<"${points[$i]}"
        run_name=$(run_name_for "$c3" "$d4")
        printf '%2d  c3=%6s  d4=%7s  %q %q -f --multicore %q\n' \
            "$((i + 1))" "$(format_value "$c3")" "$(format_value "$d4")" \
            "$PROCESS_DIR/bin/generate_events" "$run_name" "--nb_core=$cores"
    done
    exit 0
fi

mkdir -p -- "$LOG_DIR"

# Prevent simultaneous runs from racing while editing the shared Cards files.
exec 9>"$PROCESS_DIR/.run_c3d4_bridge_10k.lock"
if command -v flock >/dev/null; then
    flock -n 9 || die "another bridge-scan runner holds the process-directory lock"
fi

if ((verify_only)); then
    write_manifest || exit 1
    exit 0
fi

backup_dir=$(mktemp -d "${TMPDIR:-/tmp}/c3d4-bridge-cards.XXXXXX")
cp -p -- "$PARAM_CARD" "$backup_dir/param_card.dat"
cp -p -- "$RUN_CARD" "$backup_dir/run_card.dat"
cards_restored=0
restore_cards() {
    if ((cards_restored == 0)); then
        cp -p -- "$backup_dir/param_card.dat" "$PARAM_CARD"
        cp -p -- "$backup_dir/run_card.dat" "$RUN_CARD"
        cards_restored=1
    fi
    rm -rf -- "$backup_dir"
}
trap restore_cards EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Starting c3/d4 bridge scan: %d selected points, %d events each, %d cores\n' \
    "$selected_count" "$events" "$cores"

for ((i = 0; i < selected_count; ++i)); do
    IFS=, read -r c3 d4 <<<"${points[$i]}"
    run_name=$(run_name_for "$c3" "$d4")
    run_dir="$PROCESS_DIR/Events/$run_name"
    lhe="$run_dir/unweighted_events.lhe.gz"
    log="$LOG_DIR/$run_name.log"

    if [[ -f "$lhe" ]]; then
        if details=$(inspect_lhe "$lhe" "$c3" "$d4" 2>&1); then
            printf '[%02d/%02d] SKIP %s (%s)\n' \
                "$((i + 1))" "$selected_count" "$run_name" "$details"
            continue
        fi
        die "$run_name has an invalid existing LHE file: $details"
    fi
    [[ ! -e "$run_dir" ]] || \
        die "$run_name has an existing incomplete run directory; inspect or move it before resuming"

    printf '[%02d/%02d] RUN  %s (c3=%s, d4=%s)\n' \
        "$((i + 1))" "$selected_count" "$run_name" \
        "$(format_value "$c3")" "$(format_value "$d4")"
    set_cards "$c3" "$d4"

    set +e
    "$PROCESS_DIR/bin/generate_events" "$run_name" -f --multicore \
        "--nb_core=$cores" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    ((rc == 0)) || die "MG5 failed for $run_name (exit $rc); see $log"
    [[ -f "$lhe" ]] || die "MG5 returned success but did not create $lhe"
    details=$(inspect_lhe "$lhe" "$c3" "$d4") || \
        die "post-run validation failed for $run_name"
    printf '[%02d/%02d] DONE %s (%s)\n' \
        "$((i + 1))" "$selected_count" "$run_name" "$details"
done

restore_cards
trap - EXIT INT TERM

if write_manifest; then
    printf 'All %d selected bridge points are complete and validated.\n' "$selected_count"
else
    die "scan finished with missing or invalid selected points"
fi
