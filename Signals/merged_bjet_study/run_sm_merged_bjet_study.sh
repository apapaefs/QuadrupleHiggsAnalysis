#!/usr/bin/env bash
set -euo pipefail

analysis_root="$HOME/Projects/QuadrupleHiggsAnalysis"
herwig_prefix="$HOME/Projects/Herwig/Herwig-730-full-python3-rivet4"
signal_dir="$analysis_root/Signals"
output_dir="$signal_dir/merged_bjet_study"

run_id="${RUN_ID:-sm-merged-bjet-$(date +%Y%m%d-%H%M%S)}"
seed="${SEED:-260713}"
nevents="${NEVENTS:-10000}"
work_dir="${TMPDIR:-/tmp}/hwsim-${run_id}"

mkdir -p "$output_dir" "$work_dir"

# Keep this study isolated: neither the original SM ROOT file nor its analysis
# products in Signals/events are read, replaced, or removed.
cp "$signal_dir/HW-gg_hhhh_SM.run" "$work_dir/"
cp "$signal_dir/gg_hhhh_SM.lhe.gz" "$work_dir/"
cp "$output_dir/merged-bjet-study.setup" "$work_dir/"

set +u
source "$HOME/root310install/bin/thisroot.sh"
source "$herwig_prefix/bin/activate"
set -u

root_output="$output_dir/HW-gg_hhhh_SM-S${seed}-merged-bjet-study.setup-${run_id}.root"
result_prefix="$output_dir/${run_id}"

if [[ -e "$root_output" || -e "${result_prefix}.root" || -e "${result_prefix}.json" ]]; then
  echo "Refusing to overwrite an existing merged-b-jet study for $run_id" >&2
  exit 2
fi

cd "$work_dir"
Herwig run \
  --numevents="$nevents" \
  --seed="$seed" \
  --tag="$run_id" \
  --setupfile=merged-bjet-study.setup \
  HW-gg_hhhh_SM.run

test -s "$root_output"
"$analysis_root/Code/FourHiggsMergedBJetStudy" \
  "$root_output" \
  "$result_prefix"

echo "SM merged-b-jet study complete"
echo "HwSim ROOT: $root_output"
echo "Study ROOT: ${result_prefix}.root"
echo "Summary: ${result_prefix}.json"
