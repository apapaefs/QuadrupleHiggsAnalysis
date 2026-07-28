#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd)"
analysis_jobs="${C3D4_FAST_ANALYSIS_JOBS:-8}"
study_outdir="${C3D4_FAST_OUTDIR:-${repo_dir}/xgboost_c3d4_study_v2_uniform-smear-v1_fast-sm_153-hhhh-plus-hhhbb-cut-only}"
hhhbb_dir="${repo_dir}/HerwigForcedSplitting/gg_hhhg_c3d4_10k_hhhbb_153/events"

python3 "${script_dir}/validate_analysis_inputs.py" \
  --write-csv "${script_dir}/metadata/cross_section_audit.csv" \
  --write-json "${script_dir}/metadata/cross_section_audit.json"
python3 "${script_dir}/validate_hhhbb_inputs.py" \
  --write-csv "${script_dir}/metadata/hhhbb_inputs_153.csv" \
  --write-json "${script_dir}/metadata/hhhbb_input_audit.json"

set +u
source ~/Projects/Herwig/Herwig-730-full-python3-rivet4/bin/activate
source ~/root310install/bin/thisroot.sh
source ~/xgb-py310/bin/activate
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${repo_dir}"
exec python 4h_analyzer.py \
  --run-c3d4-xgboost-study \
  --study-mode fast-sm \
  --observable-set extended-91-v2 \
  --feature-profile full91 \
  --training-strategy sm-crossfit-v2 \
  --cv-folds 5 \
  --optuna-trials 0 \
  --no-pyhf \
  --c3d4-signal-dir "${repo_dir}/HerwigSignalPoints/c3d4_40k/events" \
  --c3d4-default-generated-events 40000 \
  --hhhbb-signal-dir "${hhhbb_dir}" \
  --hhhbb-default-generated-events 10000 \
  --c3d4-xsec-source-dir "${script_dir}" \
  --analysis-jobs "${analysis_jobs}" \
  --shape-jobs 1 \
  --progress-interval 30 \
  --c3d4-contour-interpolation clough-tocher \
  --study-outdir "${study_outdir}"
