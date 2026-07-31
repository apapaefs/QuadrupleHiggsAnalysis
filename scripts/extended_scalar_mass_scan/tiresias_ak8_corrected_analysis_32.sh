#!/usr/bin/env bash
# Run the corrected AK8 direct+cascade statistical campaign on Tiresias.
#
# This controller consumes existing ROOT feature files and an existing LHE
# header normalization.  It contains no event-generation or feature-extraction
# stage.  The default campaign is the statistically supported resolved-only
# likelihood.  The all-category inclusive fallback is opt-in, diagnostic, and
# is never promoted as an analysis default.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SELF="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"

# All paths can be overridden explicitly when the Tiresias layout differs.
ANALYSIS_ROOT=${AK8_ANALYSIS_ROOT:-/home/apapaefs/Projects/QuadrupleHiggsAnalysis}
SOURCE_REPO=${AK8_SOURCE_REPO:-${ANALYSIS_ROOT}}
CODE_REF=${AK8_CODE_REF:-HEAD}
RUN_ROOT=${AK8_RUN_ROOT:-/mnt/ssd2/Projects/4H/QuadrupleHiggsAnalysis_runs/AK8ResolvedOnlyXsecFix/ak8-v1-xsecfix-20260731}
CODE_DIR=${AK8_CODE_DIR:-${RUN_ROOT}/code_checkout}
CONTROL_ROOT=${AK8_CONTROL_ROOT:-${RUN_ROOT}/control}

PYTHON_BIN=${AK8_PYTHON_BIN:-/home/apapaefs/xgb-py310/bin/python3}
ROOT_ENV=${AK8_ROOT_ENV:-/home/apapaefs/root310install/bin/thisroot.sh}
XGB_ENV=${AK8_XGB_ENV:-/home/apapaefs/xgb-py310/bin/activate}
SIGNAL_MANIFEST=${AK8_SIGNAL_MANIFEST:-${ANALYSIS_ROOT}/HerwigSignalPoints/mass_scan_10k_ak8-v1/manifest.csv}
SIGNAL_ROOT_DIR=${AK8_SIGNAL_ROOT_DIR:-${ANALYSIS_ROOT}/ResonanceAnalysis/features/ak8-v1-cascade-r04-10k-20260731}
INPUT_BACKGROUND_MANIFEST=${AK8_BACKGROUND_MANIFEST:-${ANALYSIS_ROOT}/ResonanceAnalysis/background_manifest_ak8-v1-cascade-r04-10k-20260731_features.csv}
CORRECTED_BACKGROUND_MANIFEST=${AK8_CORRECTED_BACKGROUND_MANIFEST:-${RUN_ROOT}/inputs/background_manifest_ak8-v1-xsecfix.csv}
CORRECTED_BACKGROUND_AUDIT=${AK8_CORRECTED_BACKGROUND_AUDIT:-${CORRECTED_BACKGROUND_MANIFEST%.*}.normalization_audit.json}
BACKGROUND_PROVENANCE=${AK8_BACKGROUND_PROVENANCE:-}

SUPPORTED_ROOT=${AK8_SUPPORTED_ROOT:-${RUN_ROOT}/results/resolved-only}
DIAGNOSTIC_ROOT=${AK8_DIAGNOSTIC_ROOT:-${RUN_ROOT}/results/inclusive-diagnostic}
CURRENT_LINK=${AK8_CURRENT_LINK:-${ANALYSIS_ROOT}/ResonanceAnalysis/results/ak8-v1-resolved-only-current}

WORKERS_PER_ANALYSIS=${AK8_WORKERS_PER_ANALYSIS:-16}
MIN_BACKGROUND_RAW=${AK8_MIN_BACKGROUND_RAW:-25}
MIN_BACKGROUND_NEFF=${AK8_MIN_BACKGROUND_NEFF:-10}
FALLBACK_BACKGROUND_NEFF=${AK8_FALLBACK_BACKGROUND_NEFF:-5}
RUN_DIAGNOSTIC=${AK8_RUN_DIAGNOSTIC:-0}

ADOPT_SAMPLE=${AK8_ADOPT_SAMPLE:-HW-gg_to_4b_2c_2j}
EXPECTED_ADOPTED_XSEC_FB=${AK8_EXPECTED_ADOPTED_XSEC_FB:-2751.78}
EXPECTED_DIRECT_POINTS=${AK8_EXPECTED_DIRECT_POINTS:-42}
EXPECTED_CASCADE_POINTS=${AK8_EXPECTED_CASCADE_POINTS:-441}

CODE_COMMIT=${AK8_CODE_COMMIT:-}
CONTROLLER_RUN=${CONTROLLER_RUN:-}
LAST_PHASE=not_started
LAST_DETAIL="controller has not started"

usage() {
    cat <<EOF
Usage: $(basename -- "$SELF") COMMAND [OPTION]

Commands:
  start [--with-diagnostic]  Start/resume the corrected campaign in background.
  status                     Show controller state and checkpoint counts.
  follow                     Follow the latest controller log.
  validate                   Strictly validate completed supported results.
                             Also validates diagnostic results if requested.
  help                       Show this help.

The two topologies run concurrently with ${WORKERS_PER_ANALYSIS} workers each
(${WORKERS_PER_ANALYSIS} x 2, never more than 32 total workers).  "start" runs
the supported --pyhf-low-mc-policy exclude campaign first.  The optional
--with-diagnostic stage subsequently runs inclusive-diagnostic, but it is
never promoted.  This script never generates background events.

Common overrides (set as environment variables before "start"):
  AK8_CODE_REF               Exact commit/ref to snapshot (default: HEAD)
  AK8_RUN_ROOT               Immutable campaign/output root
  AK8_ANALYSIS_ROOT          Production data tree (read-only inputs)
  AK8_BACKGROUND_MANIFEST    Existing AK8 feature manifest
  AK8_SIGNAL_MANIFEST        Existing signal manifest
  AK8_SIGNAL_ROOT_DIR        Existing AK8 ROOT feature directory
  AK8_PYTHON_BIN             Python with uproot, xgboost and pyhf
  AK8_ROOT_ENV               ROOT environment setup sourced by the controller
  AK8_XGB_ENV                XGBoost/pyhf virtual-environment activation script

Example:
  AK8_CODE_REF=codex/c3d4-parameterized-campaign \\
    $SELF start --with-diagnostic
  watch -n 32 '$SELF status'
  $SELF follow
  $SELF validate
EOF
}

die() {
    echo "error: $*" >&2
    return 1
}

require_absolute() {
    local label=$1
    local value=$2
    [[ $value == /* ]] || die "$label must be an absolute path: $value"
}

is_positive_integer() {
    [[ $1 =~ ^[1-9][0-9]*$ ]]
}

git_in() {
    local repository=$1
    shift
    git -c safe.directory="$repository" -C "$repository" "$@"
}

activate_runtime() {
    [[ -f $ROOT_ENV ]] || die "missing ROOT environment setup: $ROOT_ENV"
    [[ -f $XGB_ENV ]] || die "missing XGBoost environment setup: $XGB_ENV"
    # shellcheck disable=SC1090
    source "$ROOT_ENV"
    # shellcheck disable=SC1090
    source "$XGB_ENV"
    "$PYTHON_BIN" -c \
        'import ROOT, pyhf, scipy, xgboost; print("runtime=" + ROOT.gROOT.GetVersion() + ",pyhf=" + pyhf.__version__ + ",xgboost=" + xgboost.__version__)' \
        >/dev/null || die "the configured runtime cannot import ROOT, pyhf, scipy and xgboost"
}

validate_config() {
    local path_value
    for path_value in \
        "$ANALYSIS_ROOT" "$SOURCE_REPO" "$RUN_ROOT" "$CODE_DIR" \
        "$CONTROL_ROOT" "$PYTHON_BIN" "$SIGNAL_MANIFEST" \
        "$ROOT_ENV" "$XGB_ENV" \
        "$SIGNAL_ROOT_DIR" "$INPUT_BACKGROUND_MANIFEST" \
        "$CORRECTED_BACKGROUND_MANIFEST" "$SUPPORTED_ROOT" \
        "$DIAGNOSTIC_ROOT" "$CURRENT_LINK"; do
        require_absolute "configured path" "$path_value"
    done
    [[ $RUN_ROOT != / && $RUN_ROOT != "$ANALYSIS_ROOT" ]] || \
        die "AK8_RUN_ROOT must be a dedicated campaign directory"
    [[ -d $ANALYSIS_ROOT ]] || die "missing analysis root: $ANALYSIS_ROOT"
    git_in "$SOURCE_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
        die "not a Git repository: $SOURCE_REPO"
    [[ -x $PYTHON_BIN ]] || die "Python is not executable: $PYTHON_BIN"
    activate_runtime
    command -v git >/dev/null || die "git is required"
    command -v flock >/dev/null || die "flock is required on Tiresias"
    is_positive_integer "$WORKERS_PER_ANALYSIS" || \
        die "AK8_WORKERS_PER_ANALYSIS must be a positive integer"
    (( WORKERS_PER_ANALYSIS <= 16 )) || \
        die "at most 16 workers per topology are allowed"
    (( 2 * WORKERS_PER_ANALYSIS <= 32 )) || \
        die "direct+cascade may not exceed 32 workers"
    is_positive_integer "$EXPECTED_DIRECT_POINTS" || \
        die "AK8_EXPECTED_DIRECT_POINTS must be positive"
    is_positive_integer "$EXPECTED_CASCADE_POINTS" || \
        die "AK8_EXPECTED_CASCADE_POINTS must be positive"
    [[ $RUN_DIAGNOSTIC == 0 || $RUN_DIAGNOSTIC == 1 ]] || \
        die "AK8_RUN_DIAGNOSTIC must be 0 or 1"
    if [[ -z $CODE_COMMIT ]]; then
        CODE_COMMIT=$(git_in "$SOURCE_REPO" rev-parse --verify "${CODE_REF}^{commit}") || \
            die "cannot resolve AK8_CODE_REF=$CODE_REF in $SOURCE_REPO"
    fi
}

write_config() {
    local destination=$1
    {
        printf 'ANALYSIS_ROOT=%q\n' "$ANALYSIS_ROOT"
        printf 'SOURCE_REPO=%q\n' "$SOURCE_REPO"
        printf 'CODE_REF=%q\n' "$CODE_REF"
        printf 'CODE_COMMIT=%q\n' "$CODE_COMMIT"
        printf 'RUN_ROOT=%q\n' "$RUN_ROOT"
        printf 'CODE_DIR=%q\n' "$CODE_DIR"
        printf 'CONTROL_ROOT=%q\n' "$CONTROL_ROOT"
        printf 'PYTHON_BIN=%q\n' "$PYTHON_BIN"
        printf 'ROOT_ENV=%q\n' "$ROOT_ENV"
        printf 'XGB_ENV=%q\n' "$XGB_ENV"
        printf 'SIGNAL_MANIFEST=%q\n' "$SIGNAL_MANIFEST"
        printf 'SIGNAL_ROOT_DIR=%q\n' "$SIGNAL_ROOT_DIR"
        printf 'INPUT_BACKGROUND_MANIFEST=%q\n' "$INPUT_BACKGROUND_MANIFEST"
        printf 'CORRECTED_BACKGROUND_MANIFEST=%q\n' "$CORRECTED_BACKGROUND_MANIFEST"
        printf 'CORRECTED_BACKGROUND_AUDIT=%q\n' "$CORRECTED_BACKGROUND_AUDIT"
        printf 'BACKGROUND_PROVENANCE=%q\n' "$BACKGROUND_PROVENANCE"
        printf 'SUPPORTED_ROOT=%q\n' "$SUPPORTED_ROOT"
        printf 'DIAGNOSTIC_ROOT=%q\n' "$DIAGNOSTIC_ROOT"
        printf 'CURRENT_LINK=%q\n' "$CURRENT_LINK"
        printf 'WORKERS_PER_ANALYSIS=%q\n' "$WORKERS_PER_ANALYSIS"
        printf 'MIN_BACKGROUND_RAW=%q\n' "$MIN_BACKGROUND_RAW"
        printf 'MIN_BACKGROUND_NEFF=%q\n' "$MIN_BACKGROUND_NEFF"
        printf 'FALLBACK_BACKGROUND_NEFF=%q\n' "$FALLBACK_BACKGROUND_NEFF"
        printf 'RUN_DIAGNOSTIC=%q\n' "$RUN_DIAGNOSTIC"
        printf 'ADOPT_SAMPLE=%q\n' "$ADOPT_SAMPLE"
        printf 'EXPECTED_ADOPTED_XSEC_FB=%q\n' "$EXPECTED_ADOPTED_XSEC_FB"
        printf 'EXPECTED_DIRECT_POINTS=%q\n' "$EXPECTED_DIRECT_POINTS"
        printf 'EXPECTED_CASCADE_POINTS=%q\n' "$EXPECTED_CASCADE_POINTS"
    } > "$destination"
}

latest_run_dir() {
    local link_path=${CONTROL_ROOT}/latest
    [[ -L $link_path ]] || die "no campaign has been started under $CONTROL_ROOT"
    readlink -- "$link_path"
}

write_state() {
    local phase=$1
    local detail=${2//$'\n'/ }
    local state_tmp=${CONTROLLER_RUN}/state.env.tmp.$$
    LAST_PHASE=$phase
    LAST_DETAIL=$detail
    {
        printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'phase=%s\n' "$phase"
        printf 'detail=%s\n' "$detail"
        printf 'controller_pid=%s\n' "$$"
        printf 'code_commit=%s\n' "$CODE_COMMIT"
        printf 'analysis_root=%s\n' "$ANALYSIS_ROOT"
        printf 'run_root=%s\n' "$RUN_ROOT"
        printf 'supported_results=%s\n' "$SUPPORTED_ROOT"
        printf 'diagnostic_results=%s\n' "$DIAGNOSTIC_ROOT"
        printf 'resolved_only_current=%s\n' "$CURRENT_LINK"
        printf 'diagnostic_requested=%s\n' "$RUN_DIAGNOSTIC"
        printf 'controller_log=%s\n' "${CONTROLLER_RUN}/controller.log"
    } > "$state_tmp"
    mv -- "$state_tmp" "${CONTROLLER_RUN}/state.env"
}

controller_exit() {
    local rc=$1
    trap - EXIT
    if (( rc != 0 )) && [[ $LAST_PHASE != failed ]]; then
        write_state failed "${LAST_PHASE}: ${LAST_DETAIL}; controller exited with status ${rc}" || true
    fi
    exit "$rc"
}

resolve_isolated_code() {
    write_state preparing_code "snapshotting commit $CODE_COMMIT into isolated code tree"
    if [[ ! -e $CODE_DIR ]]; then
        mkdir -p -- "$(dirname -- "$CODE_DIR")"
        git -c safe.directory="$SOURCE_REPO" clone --shared --no-checkout \
            "$SOURCE_REPO" "$CODE_DIR"
        git_in "$CODE_DIR" checkout --detach "$CODE_COMMIT"
    fi
    [[ -d $CODE_DIR/.git ]] || die "$CODE_DIR exists but is not an isolated Git checkout"
    local isolated_commit
    isolated_commit=$(git_in "$CODE_DIR" rev-parse HEAD)
    [[ $isolated_commit == "$CODE_COMMIT" ]] || \
        die "$CODE_DIR is pinned to $isolated_commit, expected $CODE_COMMIT; choose a new AK8_RUN_ROOT"
    [[ -z $(git_in "$CODE_DIR" status --porcelain) ]] || \
        die "$CODE_DIR is not clean; choose a new AK8_RUN_ROOT"
    [[ -f $CODE_DIR/Code/resonance_fatjet_xgboost_analysis.py ]] || \
        die "isolated commit lacks the AK8 analysis"
    [[ -f $CODE_DIR/Code/prepare_resonance_background_normalization_manifest.py ]] || \
        die "isolated commit lacks the normalization-manifest helper"
    grep -q -- 'inclusive-diagnostic' "$CODE_DIR/Code/resonance_fatjet_xgboost_analysis.py" || \
        die "isolated AK8 code lacks the low-MC policy implementation"
}

copy_optional_provenance() {
    [[ -n $BACKGROUND_PROVENANCE ]] || return 0
    require_absolute "AK8_BACKGROUND_PROVENANCE" "$BACKGROUND_PROVENANCE"
    [[ -f $BACKGROUND_PROVENANCE ]] || \
        die "missing supplied background provenance: $BACKGROUND_PROVENANCE"
    local destination=${RUN_ROOT}/inputs/$(basename -- "$BACKGROUND_PROVENANCE")
    if [[ -e $destination ]]; then
        cmp -s -- "$BACKGROUND_PROVENANCE" "$destination" || \
            die "$destination exists with different provenance content"
    else
        cp -- "$BACKGROUND_PROVENANCE" "$destination"
    fi
}

prepare_existing_inputs() {
    write_state preparing_inputs "checking existing ROOT/LHE inputs and correcting one manifest normalization"
    [[ -f $SIGNAL_MANIFEST ]] || die "missing signal manifest: $SIGNAL_MANIFEST"
    [[ -d $SIGNAL_ROOT_DIR ]] || die "missing signal feature directory: $SIGNAL_ROOT_DIR"
    [[ -f $INPUT_BACKGROUND_MANIFEST ]] || \
        die "missing existing background feature manifest: $INPUT_BACKGROUND_MANIFEST"
    mkdir -p -- "$RUN_ROOT/inputs" "$RUN_ROOT/logs"
    copy_optional_provenance
    "$PYTHON_BIN" "$CODE_DIR/Code/prepare_resonance_background_normalization_manifest.py" \
        --analysis-root "$ANALYSIS_ROOT" \
        --input-manifest "$INPUT_BACKGROUND_MANIFEST" \
        --output-manifest "$CORRECTED_BACKGROUND_MANIFEST" \
        --adopt-lhe-xsec "$ADOPT_SAMPLE"
    [[ -f $CORRECTED_BACKGROUND_MANIFEST && -f $CORRECTED_BACKGROUND_AUDIT ]] || \
        die "normalization helper did not create its immutable manifest and audit"
}

analysis_command() {
    local topology=$1
    local mode=$2
    local policy=$3
    local output_root=$4
    local -n command_ref=$5
    command_ref=(
        "$PYTHON_BIN" -u "$CODE_DIR/Code/resonance_fatjet_xgboost_analysis.py"
        --analysis-root "$ANALYSIS_ROOT"
        --feature-set fatjet-ak8-softdrop-v1
        --topology "$topology"
        --mode "$mode"
        --signal-manifest "$SIGNAL_MANIFEST"
        --signal-root-dir "$SIGNAL_ROOT_DIR"
        --background-manifest "$CORRECTED_BACKGROUND_MANIFEST"
        --output-dir "$output_root/$topology"
        --min-background-raw "$MIN_BACKGROUND_RAW"
        --min-background-neff "$MIN_BACKGROUND_NEFF"
        --pyhf-low-mc-policy "$policy"
    )
    if [[ -n $FALLBACK_BACKGROUND_NEFF ]]; then
        command_ref+=(--fallback-background-neff "$FALLBACK_BACKGROUND_NEFF")
    fi
    if [[ $mode == fast ]]; then
        command_ref+=(--point-jobs "$WORKERS_PER_ANALYSIS" --pyhf-jobs 1)
    else
        command_ref+=(--point-jobs 1 --pyhf-jobs "$WORKERS_PER_ANALYSIS")
    fi
}

run_analysis_job() {
    local topology=$1
    local mode=$2
    local policy=$3
    local output_root=$4
    local log_path=$5
    local -a command=()
    analysis_command "$topology" "$mode" "$policy" "$output_root" command
    (
        export PYTHONDONTWRITEBYTECODE=1
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export MPLCONFIGDIR=${RUN_ROOT}/mplconfig
        printf 'started_utc=%s\ncommand=' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '%q ' "${command[@]}"
        printf '\n'
        exec "${command[@]}"
    ) > "$log_path" 2>&1
}

run_pair() {
    local phase=$1
    local mode=$2
    local policy=$3
    local output_root=$4
    local direct_log=${CONTROLLER_RUN}/${policy}-${mode}-direct.log
    local cascade_log=${CONTROLLER_RUN}/${policy}-${mode}-cascade.log
    write_state "$phase" "$policy $mode: direct+cascade on $((2 * WORKERS_PER_ANALYSIS)) workers; logs in $CONTROLLER_RUN"
    mkdir -p -- "$output_root/direct" "$output_root/cascade" "$RUN_ROOT/mplconfig"
    run_analysis_job direct "$mode" "$policy" "$output_root" "$direct_log" &
    local direct_pid=$!
    run_analysis_job cascade "$mode" "$policy" "$output_root" "$cascade_log" &
    local cascade_pid=$!
    local jobs_tmp=${CONTROLLER_RUN}/active_jobs.env.tmp.$$
    {
        printf 'job_phase=%s\n' "$phase"
        printf 'direct_pid=%s\n' "$direct_pid"
        printf 'cascade_pid=%s\n' "$cascade_pid"
        printf 'direct_log=%s\n' "$direct_log"
        printf 'cascade_log=%s\n' "$cascade_log"
    } > "$jobs_tmp"
    mv -- "$jobs_tmp" "$CONTROLLER_RUN/active_jobs.env"
    local direct_rc cascade_rc
    set +e
    wait "$direct_pid"
    direct_rc=$?
    wait "$cascade_pid"
    cascade_rc=$?
    set -e
    {
        printf 'job_phase=%s\n' "$phase"
        printf 'jobs_complete=yes\n'
        printf 'direct_pid=%s\n' "$direct_pid"
        printf 'direct_exit=%s\n' "$direct_rc"
        printf 'cascade_pid=%s\n' "$cascade_pid"
        printf 'cascade_exit=%s\n' "$cascade_rc"
        printf 'direct_log=%s\n' "$direct_log"
        printf 'cascade_log=%s\n' "$cascade_log"
    } > "$jobs_tmp"
    mv -- "$jobs_tmp" "$CONTROLLER_RUN/active_jobs.env"
    if (( direct_rc != 0 || cascade_rc != 0 )); then
        echo "$policy $mode failed: direct=$direct_rc cascade=$cascade_rc" >&2
        echo "direct log: $direct_log" >&2
        echo "cascade log: $cascade_log" >&2
        return 1
    fi
}

validate_result_set() {
    local result_root=$1
    local policy=$2
    local expected_scope=$3
    local validity=$4
    "$PYTHON_BIN" - "$result_root" "$policy" "$expected_scope" "$validity" \
        "$EXPECTED_DIRECT_POINTS" "$EXPECTED_CASCADE_POINTS" \
        "$CORRECTED_BACKGROUND_MANIFEST" "$CORRECTED_BACKGROUND_AUDIT" \
        "$ADOPT_SAMPLE" "$EXPECTED_ADOPTED_XSEC_FB" <<'PY'
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

(
    result_text,
    policy,
    expected_scope,
    validity_text,
    expected_direct_text,
    expected_cascade_text,
    corrected_manifest_text,
    audit_text,
    adopted_sample,
    expected_xsec_text,
) = sys.argv[1:]
result_root = Path(result_text)
corrected_manifest = Path(corrected_manifest_text)
audit_path = Path(audit_text)
expected_counts = {
    "direct": int(expected_direct_text),
    "cascade": int(expected_cascade_text),
}
expected_validity = validity_text == "true"
expected_xsec = float(expected_xsec_text)
errors = []

def load_json(path):
    if not path.is_file():
        errors.append(f"missing {path}")
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        errors.append(f"cannot read {path}: {exc}")
        return None

audit = load_json(audit_path)
if audit is not None:
    adopted = {row.get("sample_id"): row for row in audit.get("adopted_samples", [])}
    row = adopted.get(adopted_sample)
    if row is None:
        errors.append(f"normalization audit does not adopt {adopted_sample}")
    else:
        actual = float(row.get("adopted_cross_section_fb", float("nan")))
        if not math.isclose(actual, expected_xsec, rel_tol=1e-7, abs_tol=1e-5):
            errors.append(
                f"{adopted_sample} xsec is {actual:g} fb, expected {expected_xsec:g} fb"
            )
        if not math.isfinite(float(row.get("relative_uncertainty", float("nan")))):
            errors.append(f"{adopted_sample} lacks a finite integration uncertainty")

if corrected_manifest.is_file():
    with corrected_manifest.open(newline="") as handle:
        rows = {row.get("sample_id"): row for row in csv.DictReader(handle)}
    row = rows.get(adopted_sample)
    if row is None:
        errors.append(f"corrected manifest lacks {adopted_sample}")
    elif row.get("normalization_source") != "source_lhe_init":
        errors.append(f"{adopted_sample} normalization is not sourced from the LHE init block")
else:
    errors.append(f"missing {corrected_manifest}")

for topology, expected_points in expected_counts.items():
    output = result_root / topology
    manifest = load_json(output / "method_manifest.json")
    limits = load_json(output / "point_limits.json")
    stats = load_json(output / "template_statistics_summary.json")
    if manifest is None or limits is None or stats is None:
        continue
    if manifest.get("mode") != "full":
        errors.append(f"{topology}: method manifest is not full mode")
    if manifest.get("status") != "complete":
        errors.append(f"{topology}: method manifest status={manifest.get('status')!r}")
    completion = manifest.get("limit_completion", {})
    if not completion.get("computationally_complete"):
        errors.append(f"{topology}: pyhf checkpoint set is not computationally complete")
    if not completion.get("median_limits_complete"):
        errors.append(f"{topology}: one or more median limits are missing")
    if manifest.get("pyhf_low_mc_policy") != policy:
        errors.append(f"{topology}: wrong low-MC policy")
    if manifest.get("analysis_scope") != expected_scope:
        errors.append(
            f"{topology}: scope={manifest.get('analysis_scope')!r}, expected {expected_scope!r}"
        )
    if bool(manifest.get("physics_result_valid")) != expected_validity:
        errors.append(
            f"{topology}: physics_result_valid={manifest.get('physics_result_valid')!r}"
        )
    if stats.get("pyhf_low_mc_policy") != policy:
        errors.append(f"{topology}: template statistics policy mismatch")
    if stats.get("physics_scope") != expected_scope:
        errors.append(f"{topology}: template statistics scope mismatch")
    if policy == "exclude":
        if not stats.get("retained_template_statistics_satisfied"):
            errors.append(f"{topology}: retained resolved templates fail primary statistics")
        missing = {"mixed", "boosted"} - set(
            stats.get("categories_failing_primary_template_support", [])
        )
        if missing:
            errors.append(f"{topology}: low-MC exclusions not audited for {sorted(missing)}")
        expected_channels = 5
    else:
        if stats.get("primary_template_statistics_satisfied"):
            errors.append(f"{topology}: diagnostic unexpectedly claims primary MC support")
        if not stats.get("all_categories_retained"):
            errors.append(f"{topology}: diagnostic did not retain all categories")
        if not stats.get("channel_bins_below_primary_requirements", 0):
            errors.append(f"{topology}: diagnostic lacks a low-MC warning")
        expected_channels = 15

    expected_rows = 2 * expected_points
    if len(limits) != expected_rows:
        errors.append(f"{topology}: {len(limits)} limit rows, expected {expected_rows}")
    point_scenarios = defaultdict(list)
    status_counts = Counter()
    for row in limits:
        point_id = row.get("point_id")
        point_scenarios[point_id].append(row.get("tagging_scenario"))
        status_counts[row.get("status")] += 1
        value = row.get("expected_median_limit_fb", row.get("expected_median"))
        try:
            finite_positive = math.isfinite(float(value)) and float(value) > 0.0
        except (TypeError, ValueError):
            finite_positive = False
        if not finite_positive:
            errors.append(f"{topology}/{point_id}: non-finite expected median limit")
        if row.get("n_channels") != expected_channels:
            errors.append(
                f"{topology}/{point_id}/{row.get('tagging_scenario')}: "
                f"n_channels={row.get('n_channels')}, expected {expected_channels}"
            )
    if sum(status_counts.values()) != expected_rows or set(status_counts) - {"ok", "partial"}:
        errors.append(
            f"{topology}: expected only finite ok/partial limits, found {dict(status_counts)}"
        )
    malformed = {
        point: scenarios
        for point, scenarios in point_scenarios.items()
        if sorted(scenarios) != ["conservative", "nominal"]
    }
    if len(point_scenarios) != expected_points or malformed:
        errors.append(
            f"{topology}: expected {expected_points} unique points with both scenarios; "
            f"found {len(point_scenarios)} points and {len(malformed)} malformed"
        )
    if topology == "cascade":
        anomaly = [
            row for row in limits
            if float(row.get("M2_GeV", -1)) == 1800.0
            and float(row.get("M3_GeV", -1)) == 4000.0
        ]
        if len(anomaly) != 2:
            errors.append("cascade: missing the two (1800,4000) anomaly checks")
        else:
            summary = ", ".join(
                f"{row['tagging_scenario']}={float(row.get('expected_median_limit_fb', row.get('expected_median'))):.6g} fb"
                for row in sorted(anomaly, key=lambda item: item["tagging_scenario"])
            )
            print(f"cascade anomaly replacement: {summary}")

if errors:
    print("VALIDATION FAILED", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)
print(
    f"VALIDATION PASSED: policy={policy}, scope={expected_scope}, "
    f"direct={expected_counts['direct']} points, cascade={expected_counts['cascade']} points"
)
PY
}

validate_supported() {
    validate_result_set "$SUPPORTED_ROOT" exclude resolved_only true
}

validate_diagnostic() {
    validate_result_set \
        "$DIAGNOSTIC_ROOT" inclusive-diagnostic diagnostic_existing_mc_only false
}

seed_diagnostic_caches() {
    local topology source destination component
    write_state preparing_diagnostic "hard-linking immutable trained-model and score caches into diagnostic output"
    for topology in direct cascade; do
        source=${SUPPORTED_ROOT}/${topology}/checkpoints
        destination=${DIAGNOSTIC_ROOT}/${topology}/checkpoints
        mkdir -p -- "$destination"
        for component in models scores; do
            [[ -d $source/$component ]] || die "missing supported cache: $source/$component"
            if [[ ! -e $destination/$component ]]; then
                cp -al -- "$source/$component" "$destination/$component"
            fi
        done
    done
}

promote_supported() {
    local link_parent previous candidate validation_dir validation_file
    link_parent=$(dirname -- "$CURRENT_LINK")
    mkdir -p -- "$link_parent"
    if [[ -e $CURRENT_LINK || -L $CURRENT_LINK ]]; then
        [[ -L $CURRENT_LINK ]] || \
            die "refusing to replace non-symlink default path: $CURRENT_LINK"
        previous=$(readlink -- "$CURRENT_LINK")
    else
        previous=none
    fi
    candidate=${CURRENT_LINK}.next.$$
    [[ ! -e $candidate && ! -L $candidate ]] || die "promotion candidate already exists: $candidate"
    ln -s -- "$SUPPORTED_ROOT" "$candidate"
    mv -Tf -- "$candidate" "$CURRENT_LINK"
    validation_dir=${RUN_ROOT}/validation
    mkdir -p -- "$validation_dir"
    validation_file=${validation_dir}/resolved-only-validated-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt
    {
        printf 'validated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'code_commit=%s\n' "$CODE_COMMIT"
        printf 'policy=exclude\n'
        printf 'scope=resolved_only\n'
        printf 'result_root=%s\n' "$SUPPORTED_ROOT"
        printf 'current_link=%s\n' "$CURRENT_LINK"
        printf 'previous_target=%s\n' "$previous"
    } > "$validation_file"
}

controller_main() {
    trap 'controller_exit $?' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    validate_config
    resolve_isolated_code
    prepare_existing_inputs

    if ! run_pair supported_fast fast exclude "$SUPPORTED_ROOT"; then
        write_state failed "supported fast direct/cascade stage failed; completed checkpoints were retained"
        return 1
    fi
    if ! run_pair supported_full full exclude "$SUPPORTED_ROOT"; then
        write_state failed "supported full pyhf stage failed; completed checkpoints were retained"
        return 1
    fi
    write_state validating_supported "strict validation of normalization, point coverage, templates and pyhf limits"
    if ! validate_supported; then
        write_state failed "resolved-only validation failed; no default symlink was changed"
        return 1
    fi
    promote_supported

    if [[ $RUN_DIAGNOSTIC == 1 ]]; then
        seed_diagnostic_caches
        if ! run_pair diagnostic_fast fast inclusive-diagnostic "$DIAGNOSTIC_ROOT"; then
            write_state complete_with_diagnostic_failure \
                "resolved-only result validated/promoted; optional diagnostic fast stage failed"
            return 0
        fi
        if ! run_pair diagnostic_full full inclusive-diagnostic "$DIAGNOSTIC_ROOT"; then
            write_state complete_with_diagnostic_failure \
                "resolved-only result validated/promoted; optional diagnostic pyhf stage failed"
            return 0
        fi
        write_state validating_diagnostic "checking diagnostic labels, complete coverage and low-MC warning"
        if ! validate_diagnostic; then
            write_state complete_with_diagnostic_failure \
                "resolved-only result validated/promoted; optional diagnostic validation failed"
            return 0
        fi
        write_state complete \
            "resolved-only result validated/promoted; inclusive all-category diagnostic also completed"
    else
        write_state complete "resolved-only result validated and promoted; diagnostic was not requested"
    fi
}

start_campaign() {
    local option=${1:-}
    if [[ -n $option ]]; then
        [[ $option == --with-diagnostic ]] || die "unknown start option: $option"
        RUN_DIAGNOSTIC=1
    fi
    validate_config
    mkdir -p -- "$CONTROL_ROOT/runs"
    exec 9>"${CONTROL_ROOT}/start.lock"
    flock -n 9 || die "another start operation is in progress"
    local existing_run existing_pid child_pid timestamp
    if [[ -L $CONTROL_ROOT/latest ]]; then
        existing_run=$(readlink -- "$CONTROL_ROOT/latest")
        if [[ -f $existing_run/controller.pid ]]; then
            existing_pid=$(<"$existing_run/controller.pid")
            if [[ $existing_pid =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
                die "controller PID $existing_pid is already running; use status/follow"
            fi
        fi
        if [[ -f $existing_run/active_jobs.env ]] && \
            ! grep -q '^jobs_complete=yes$' "$existing_run/active_jobs.env"; then
            while IFS='=' read -r _ child_pid; do
                [[ $child_pid =~ ^[0-9]+$ ]] || continue
                if kill -0 "$child_pid" 2>/dev/null; then
                    die "analysis worker parent PID $child_pid is still running; do not start a duplicate campaign"
                fi
            done < <(grep -E '^(direct|cascade)_pid=' "$existing_run/active_jobs.env" || true)
        fi
    elif [[ -e $CONTROL_ROOT/latest ]]; then
        die "refusing to replace non-symlink control path: $CONTROL_ROOT/latest"
    fi
    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    CONTROLLER_RUN=${CONTROL_ROOT}/runs/${timestamp}-$$
    mkdir -- "$CONTROLLER_RUN"
    write_config "$CONTROLLER_RUN/config.env"
    ln -sfn -- "$CONTROLLER_RUN" "$CONTROL_ROOT/latest"
    nohup "$SELF" _controller "$CONTROLLER_RUN" \
        9>&- > "$CONTROLLER_RUN/controller.log" 2>&1 &
    local controller_pid=$!
    printf '%s\n' "$controller_pid" > "$CONTROLLER_RUN/controller.pid"
    disown "$controller_pid" 2>/dev/null || true
    echo "started controller_pid=$controller_pid"
    echo "code_commit=$CODE_COMMIT"
    echo "controller_log=$CONTROLLER_RUN/controller.log"
    echo "supported_results=$SUPPORTED_ROOT"
    echo "diagnostic_requested=$RUN_DIAGNOSTIC"
    echo "monitor: $SELF status"
    echo "follow:  $SELF follow"
}

status_campaign() {
    local run_dir config_path pid alive=no child_key child_pid child_alive
    run_dir=$(latest_run_dir)
    config_path=$run_dir/config.env
    [[ -f $config_path ]] || die "missing controller config: $config_path"
    # shellcheck disable=SC1090
    source "$config_path"
    if [[ -f $run_dir/controller.pid ]]; then
        pid=$(<"$run_dir/controller.pid")
        if [[ $pid =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            alive=yes
        fi
    else
        pid=unknown
    fi
    if [[ -f $run_dir/state.env ]]; then
        cat "$run_dir/state.env"
    else
        echo "phase=starting"
        echo "detail=controller has not written its first state snapshot"
    fi
    echo "controller_alive=$alive"
    echo "controller_pid=$pid"
    if [[ -f $run_dir/active_jobs.env ]]; then
        while IFS='=' read -r child_key child_pid; do
            [[ $child_pid =~ ^[0-9]+$ ]] || continue
            child_alive=no
            kill -0 "$child_pid" 2>/dev/null && child_alive=yes
            echo "${child_key}_alive=$child_alive"
            echo "${child_key}=$child_pid"
        done < <(grep -E '^(direct|cascade)_pid=' "$run_dir/active_jobs.env" || true)
    fi
    if [[ -L $CURRENT_LINK ]]; then
        echo "resolved_only_current_target=$(readlink -- "$CURRENT_LINK")"
    else
        echo "resolved_only_current_target=not_promoted"
    fi
    local root topology template_count pyhf_count manifest
    for root in "$SUPPORTED_ROOT" "$DIAGNOSTIC_ROOT"; do
        for topology in direct cascade; do
            template_count=0
            pyhf_count=0
            if [[ -d $root/$topology/checkpoints/templates ]]; then
                template_count=$(find "$root/$topology/checkpoints/templates" -maxdepth 1 -type f -name '*.json' | wc -l)
            fi
            if [[ -d $root/$topology/checkpoints/pyhf ]]; then
                pyhf_count=$(find "$root/$topology/checkpoints/pyhf" -maxdepth 1 -type f -name '*.json' | wc -l)
            fi
            echo "$(basename -- "$root")_${topology}_templates=$template_count"
            echo "$(basename -- "$root")_${topology}_pyhf=$pyhf_count"
            manifest=$root/$topology/method_manifest.json
            if [[ -f $manifest ]]; then
                "$PYTHON_BIN" -c \
                    'import json,sys; p=json.load(open(sys.argv[1])); print("manifest="+sys.argv[1]+" status="+str(p.get("status"))+" scope="+str(p.get("analysis_scope"))+" physics_valid="+str(p.get("physics_result_valid")))' \
                    "$manifest" || true
            fi
        done
    done
}

follow_campaign() {
    local run_dir
    run_dir=$(latest_run_dir)
    [[ -f $run_dir/controller.log ]] || die "missing controller log: $run_dir/controller.log"
    tail -n 120 -F "$run_dir/controller.log"
}

validate_campaign() {
    local run_dir config_path
    run_dir=$(latest_run_dir)
    config_path=$run_dir/config.env
    [[ -f $config_path ]] || die "missing controller config: $config_path"
    # shellcheck disable=SC1090
    source "$config_path"
    validate_config
    validate_supported
    if [[ $RUN_DIAGNOSTIC == 1 ]]; then
        validate_diagnostic
    fi
}

command_name=${1:-help}
case $command_name in
    start)
        shift
        (( $# <= 1 )) || { usage >&2; exit 2; }
        start_campaign "${1:-}"
        ;;
    status)
        (( $# == 1 )) || { usage >&2; exit 2; }
        status_campaign
        ;;
    follow)
        (( $# == 1 )) || { usage >&2; exit 2; }
        follow_campaign
        ;;
    validate)
        (( $# == 1 )) || { usage >&2; exit 2; }
        validate_campaign
        ;;
    _controller)
        (( $# == 2 )) || exit 2
        CONTROLLER_RUN=$2
        [[ -f $CONTROLLER_RUN/config.env ]] || die "missing internal controller config"
        # shellcheck disable=SC1090
        source "$CONTROLLER_RUN/config.env"
        controller_main
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
