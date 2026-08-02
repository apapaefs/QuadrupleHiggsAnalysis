#!/usr/bin/env bash
set -euo pipefail

analysis_root=/home/apapaefs/Projects/QuadrupleHiggsAnalysis
run_root=${1:-/mnt/ssd2/Projects/4H/QuadrupleHiggsAnalysis_runs/ResonanceScoreFit/ak4ak8-scorefit-v1}
python_executable=/home/apapaefs/xgb-py310/bin/python

source /home/apapaefs/root310install/bin/thisroot.sh >/dev/null 2>&1
mkdir -p "${run_root}/logs" "${run_root}/mplconfig"
export MPLCONFIGDIR="${run_root}/mplconfig"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

common=(
  --analysis-root "${analysis_root}"
  --load-jobs 24
  --model-jobs 5
  --xgboost-threads 18
  --point-jobs 10
  --prediction-threads 9
  --thread-budget 90
)

numactl --interleave=all "${python_executable}" \
  "${analysis_root}/Code/resonance_score_fit_poisson.py" \
  --topology direct \
  --output-dir "${run_root}/direct" \
  "${common[@]}" \
  >"${run_root}/logs/direct.log" 2>&1 &
direct_pid=$!

numactl --interleave=all "${python_executable}" \
  "${analysis_root}/Code/resonance_score_fit_poisson.py" \
  --topology cascade \
  --output-dir "${run_root}/cascade" \
  "${common[@]}" \
  >"${run_root}/logs/cascade.log" 2>&1 &
cascade_pid=$!

set +e
wait "${direct_pid}"
direct_status=$?
wait "${cascade_pid}"
cascade_status=$?
set -e

status_file="${run_root}/run_status.txt"
{
  echo "direct_status=${direct_status}"
  echo "cascade_status=${cascade_status}"
} >"${status_file}"

if (( direct_status != 0 || cascade_status != 0 )); then
  exit 3
fi
