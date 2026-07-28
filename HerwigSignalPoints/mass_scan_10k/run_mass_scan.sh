#!/usr/bin/env bash
set -euo pipefail

campaign_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$campaign_dir/../.." && pwd)

mode=${1:-all}
jobs=${2:-8}
if (($# >= 1)); then shift; fi
if (($# >= 1)); then shift; fi

case "$mode" in
  all|direct|cascade) ;;
  *)
    echo "Usage: $0 [all|direct|cascade] [jobs] [run_herwig_signal_inputs.py options]" >&2
    exit 2
    ;;
esac
if ! [[ "$jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "jobs must be a positive integer, got: $jobs" >&2
  exit 2
fi

if ! type module >/dev/null 2>&1; then
  for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    if [[ -r "$init" ]]; then
      # shellcheck disable=SC1090
      source "$init"
      break
    fi
  done
fi
module load herwig/stable-full-py3-rivet4

# Each worker is one independent Herwig instance; prevent numerical libraries
# from adding nested threads inside those instances.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$repo_dir"
exec python3 run_herwig_signal_inputs.py \
  --list "HerwigSignalPoints/mass_scan_10k/herwig_inputs_${mode}.txt" \
  --jobs "$jobs" \
  "$@"
