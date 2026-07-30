# Four-Higgs reconstruction mass-target study

This is a compact, reproducible calibration study for the four mass targets
used while reconstructing \(hhhh\to8b\). It applies the same target tuples to
both existing object definitions:

- the resonant hybrid resolved/merged reconstruction, whose nominal tuple is
  \((125,125,125,125)\) GeV;
- the non-resonant resolved-eight-jet reconstruction, whose nominal tuple is
  \((120,115,110,105)\) GeV.

The four entries always correspond to reconstructed Higgs candidates ordered
by decreasing candidate \(p_T\). They are reconstruction anchors, not four
different physical Higgs masses.

## What the small scan measures

`Code/mass_target_study.py` reruns the actual C++ pairing algorithms for a
frozen 19-tuple grid:

- common targets of 105, 110, 115, 120, 125, and 130 GeV;
- global shifts of \(-5,-2.5,0,+2.5,+5\) GeV around
  \((120,115,110,105)\) GeV;
- independent \(\pm2.5\) GeV shifts of each entry in that staggered tuple.

The nominal tuple of each workflow is always present, including when
`--preset none` is used with custom points.

For every tuple and workflow, the study uses the negative reconstruction
residual as a one-dimensional signal score: `-best_score` in the resonant
analysis and `-chi8` in the non-resonant analysis. Events are assigned
deterministically to a 65% tuning subset and a disjoint 35% validation subset.
The tuple with the largest signal-versus-background weighted AUC on the
tuning subset is selected; only then is its held-out validation AUC reported.
An exact tuning-AUC tie is resolved in favor of the tuple closest to the
workflow's current baseline.
Each manifest sample is normalized to its `class_weight`, so the supplied
manifest gives equal weight to each representative signal point.

This is intentionally a reconstruction diagnostic, not an optimization of the
final expected limit. The raw HwSim trees do not retain a validated
Higgs-parent label for every reconstructed jet, so a direct truth-matched
pairing-efficiency calibration is not available. Any shortlisted tuple must
be confirmed with the complete XGBoost and expected-limit workflows on
independent samples before either nominal reconstruction is changed.

## Representative samples

`sample_manifest_tiresias.csv` uses:

- three direct resonance masses spanning 600--3000 GeV;
- three cascade points spanning 800--4000 GeV in the parent mass;
- the non-resonant SM point and two widely separated \(c_3,d_4\) points;
- the same branch-complete `gg -> 8b` sample as background in both workflows.

These samples are deliberately small and broad in kinematics. Add more
background components or signal points by extending the manifest. For a
physics-rate mixture, replace the equal `class_weight=1` entries with the
desired relative rates; the AUC itself is invariant under a common
normalization of either class.

## Run on Tiresias

From the repository root:

```bash
cd ~/Projects/QuadrupleHiggsAnalysis

source /etc/profile.d/modules.sh
module load herwig/stable-full-py3-rivet4
source ~/xgb-py310/bin/activate

set +u
source ~/root310install/bin/thisroot.sh
set -u

export MPLCONFIGDIR=/tmp/mass-target-matplotlib

python3 Code/mass_target_study.py all \
  --build \
  --max-events 1000 \
  --workers 4
```

The default plan contains 209 extraction jobs: 19 target tuples times seven
resonant samples, plus 19 times four non-resonant samples. Completed jobs are
reused only when the command, raw-file size and modification time, executable
digest, and relevant C++ source/header digests still match. Use `--force` to
rerun them. The runner also refuses an executable older than a relevant
source/header and asks for `--build`, preventing a scan with a stale binary.

To inspect the exact commands without requiring the raw files or executables:

```bash
python3 Code/mass_target_study.py all \
  --max-events 1000 \
  --dry-run \
  > /tmp/mass-target-study-plan.txt
```

Run only one workflow with `--workflow resonant` or
`--workflow nonresonant`. Add a target tuple with a repeatable option such as
`--targets 122.5,117.5,112.5,107.5`. Target tuples must be non-increasing
because their entries are assigned in descending candidate-\(p_T\) order.

## Outputs and decision rule

Generated products are kept below
`MassTargetStudy/results/small_scan/` and are ignored by git. The principal
outputs are:

- `study_plan.json`: exact extraction commands;
- `run_manifest.json`: completed/reused jobs and failures;
- `metrics.csv`: tuning and validation AUC for every tuple, plus held-out
  signal candidate-mass quantiles;
- `sample_metrics.csv`: the same AUC check separately for every representative
  signal point, against the combined background sample(s);
- `study_summary.json` and `REPORT.md`: the tune-selected tuple, its validation
  result, and comparison with the workflow's current baseline;
- `target_auc.png` and `target_auc.pdf`: tuning/validation scan plots.

The report returns `retain_baseline` when the baseline itself is selected,
`retain_baseline_no_held_out_gain` when a different tune-selected tuple fails
to improve validation AUC, and otherwise only
`shortlist_for_full_analysis`. A target change is supported only if its
held-out gain is stable when the manifest is enlarged and it also improves
the downstream classifier and expected-limit results. A negligible or
unstable validation gain is evidence for retaining the current nominal
target, not for choosing the largest tuning AUC.
