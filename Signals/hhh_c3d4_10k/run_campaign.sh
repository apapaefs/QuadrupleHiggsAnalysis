#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CAMPAIGN_PY="$SCRIPT_DIR/campaign.py"
ANALYSIS_PY="$SCRIPT_DIR/c3d4_bjet_ratio_scan.py"
ANALYZER_SOURCE="$REPO_DIR/Code/BJetMultiplicityAnalysis.cc"
ANALYZER="$REPO_DIR/Code/BJetMultiplicityAnalysis"

DEFAULT_CPUS="${HHH_SCAN_CPUS:-64}"
HERWIG_ACTIVATE="${HERWIG_ACTIVATE:-/home/apapaefs/Projects/Herwig/Herwig-730-full-python3-rivet4/bin/activate}"
ROOT_THISROOT="${ROOT_THISROOT:-/home/apapaefs/root310install/bin/thisroot.sh}"
XGB_ACTIVATE="${XGB_ACTIVATE:-/home/apapaefs/xgb-py310/bin/activate}"
ROOT_BIN="${ROOT_BIN:-/home/apapaefs/root310install/bin}"
PYTHON_BIN="${HHH_SCAN_PYTHON:-/home/apapaefs/xgb-py310/bin/python3}"

usage() {
  cat <<'EOF'
Usage: Signals/hhh_c3d4_10k/run_campaign.sh COMMAND [options]

Commands:
  run-mg5         Generate missing 10k-event inclusive HHH LHE points.
  status-mg5      Report HHH MG5 completion (--deep validates every LHE).
  run-herwig      Prepare and shower all completed HHH LHE points.
  status-herwig   Report HHH Herwig completion.
  analyze         Calibrate the 90% HHH pairing cut, then analyze all samples.
  plot            Write baseline and paired >=6/exactly-6 ratio contours.
  validate        Validate the completed tables, normalizations, and plots.
  status          Report MG5, Herwig, HHHbb, cache, and output status.
  smoke           Run the isolated 100-event HHH and three-process smoke test.

Options:
  --cpus N        Aggregate logical-CPU budget (default: 64).
  --source-repo P Canonical production-data repository.
  --mg5-process P Existing MG5 gg_hhh process directory.
  --herwig-dir P  HHH Herwig output directory.
  --results-dir P Analysis result directory (analysis commands only).
  --force          Re-run cached analysis outputs (analyze only).
  --deep           Deep LHE validation (status-mg5/status only).
  -h, --help       Show this help.

The production commands are resumable. OMP_NUM_THREADS is fixed to one for
parallel Herwig and analysis workers.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

COMMAND="$1"
shift
CPUS="$DEFAULT_CPUS"
CAMPAIGN_FORWARD=()
ANALYSIS_FORWARD=()
SOURCE_REPO_VALUE="${QHA_SOURCE_REPO:-/home/apapaefs/Projects/QuadrupleHiggsAnalysis}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpus)
      if [[ $# -lt 2 ]]; then
        printf 'Missing value for --cpus\n' >&2
        exit 2
      fi
      CPUS="$2"
      shift 2
      ;;
    --source-repo)
      if [[ $# -lt 2 ]]; then
        printf 'Missing value for --source-repo\n' >&2
        exit 2
      fi
      SOURCE_REPO_VALUE="$2"
      CAMPAIGN_FORWARD+=("$1" "$2")
      ANALYSIS_FORWARD+=("$1" "$2")
      shift 2
      ;;
    --mg5-process|--herwig-dir)
      if [[ $# -lt 2 ]]; then
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
      fi
      CAMPAIGN_FORWARD+=("$1" "$2")
      ANALYSIS_FORWARD+=("$1" "$2")
      shift 2
      ;;
    --results-dir|--analyzer)
      if [[ $# -lt 2 ]]; then
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
      fi
      ANALYSIS_FORWARD+=("$1" "$2")
      shift 2
      ;;
    --deep|--details)
      CAMPAIGN_FORWARD+=("$1")
      shift
      ;;
    --force)
      ANALYSIS_FORWARD+=("$1")
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

if [[ ! "$CPUS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Expected a positive --cpus value, got %s\n' "$CPUS" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

activate_herwig() {
  if [[ ! -f "$HERWIG_ACTIVATE" ]]; then
    printf 'Missing Herwig activation script: %s\n' "$HERWIG_ACTIVATE" >&2
    exit 1
  fi
  set +u
  # shellcheck disable=SC1090
  source "$HERWIG_ACTIVATE"
  set -u
  command -v Herwig >/dev/null
  printf 'Using %s\n' "$(Herwig --version 2>&1 | head -1)"
}

activate_analysis() {
  if [[ ! -f "$ROOT_THISROOT" || ! -f "$XGB_ACTIVATE" ]]; then
    printf 'Missing ROOT or Python analysis activation script.\n' >&2
    exit 1
  fi
  set +u
  # shellcheck disable=SC1090
  source "$ROOT_THISROOT"
  # shellcheck disable=SC1090
  source "$XGB_ACTIVATE"
  set -u
  export PATH="$ROOT_BIN:$PATH"
  command -v root-config >/dev/null
  "$PYTHON_BIN" -c 'import matplotlib, numpy'
}

build_analyzer() {
  if [[ ! -x "$ANALYZER" || "$ANALYZER_SOURCE" -nt "$ANALYZER" ]]; then
    printf 'Building %s\n' "$ANALYZER"
    # ROOT deliberately supplies a whitespace-separated flag list.
    # shellcheck disable=SC2046
    g++ -std=c++17 -O2 -Wall -Wextra \
      -D_GLIBCXX_USE_CXX11_ABI=1 \
      $(root-config --cflags) \
      "$ANALYZER_SOURCE" \
      -o "$ANALYZER" \
      $(root-config --libs)
  fi
}

export OMP_NUM_THREADS=1
cd "$REPO_DIR"

case "$COMMAND" in
  run-mg5)
    exec "$PYTHON_BIN" "$CAMPAIGN_PY" run-mg5 --cpus "$CPUS" "${CAMPAIGN_FORWARD[@]}"
    ;;
  status-mg5)
    exec "$PYTHON_BIN" "$CAMPAIGN_PY" status-mg5 --cpus "$CPUS" "${CAMPAIGN_FORWARD[@]}"
    ;;
  run-herwig)
    activate_herwig
    exec "$PYTHON_BIN" "$CAMPAIGN_PY" run-herwig --cpus "$CPUS" "${CAMPAIGN_FORWARD[@]}"
    ;;
  status-herwig)
    exec "$PYTHON_BIN" "$CAMPAIGN_PY" status-herwig --cpus "$CPUS" "${CAMPAIGN_FORWARD[@]}"
    ;;
  analyze)
    activate_analysis
    build_analyzer
    exec "$PYTHON_BIN" "$ANALYSIS_PY" analyze --cpus "$CPUS" "${ANALYSIS_FORWARD[@]}"
    ;;
  plot)
    activate_analysis
    exec "$PYTHON_BIN" "$ANALYSIS_PY" plot --cpus "$CPUS" "${ANALYSIS_FORWARD[@]}"
    ;;
  validate)
    activate_analysis
    "$PYTHON_BIN" "$REPO_DIR/Signals/c3d4_40k/validate_hhhbb_inputs.py" \
      --campaign-dir "$REPO_DIR/Signals/c3d4_40k" \
      --workdir "$SOURCE_REPO_VALUE/HerwigForcedSplitting/gg_hhhg_c3d4_10k_hhhbb_153"
    exec "$PYTHON_BIN" "$ANALYSIS_PY" validate --cpus "$CPUS" "${ANALYSIS_FORWARD[@]}"
    ;;
  status)
    "$PYTHON_BIN" "$CAMPAIGN_PY" status --cpus "$CPUS" "${CAMPAIGN_FORWARD[@]}"
    exec "$PYTHON_BIN" "$ANALYSIS_PY" status --cpus "$CPUS" "${ANALYSIS_FORWARD[@]}"
    ;;
  smoke)
    activate_herwig
    "$PYTHON_BIN" "$CAMPAIGN_PY" smoke-samples --cpus "$CPUS" "${CAMPAIGN_FORWARD[@]}"
    activate_analysis
    build_analyzer
    exec "$PYTHON_BIN" "$ANALYSIS_PY" smoke-analysis --cpus "$CPUS" "${ANALYSIS_FORWARD[@]}"
    ;;
  *)
    printf 'Unknown command: %s\n' "$COMMAND" >&2
    usage >&2
    exit 2
    ;;
esac
