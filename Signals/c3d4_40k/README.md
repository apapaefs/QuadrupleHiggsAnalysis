# c3/d4 40k signal LHE campaign

This directory contains the 153 unique `gg -> hhhh` c3/d4 signal points
generated on Odysseus:

- 117 established signal-plus-bridge points;
- 36 refinement points;
- 40,000 events per point (6,120,000 events total);
- MadGraph run group `5`.

The transferred run directories are under `Events/`.  Each contains the
compressed LHE file and its MadGraph banner.  `metadata/points_153.csv` is the
authoritative point list, and `metadata/source_sha256.txt` records the
Odysseus SHA-256 values for every transferred LHE and banner.

## Herwig

From the repository root, first prepare and inspect all 153 Herwig cards:

```bash
./Signals/c3d4_40k/run_herwig_parallel.sh --prepare-only
```

Run the campaign with 32 concurrent single-process Herwig jobs:

```bash
./Signals/c3d4_40k/run_herwig_parallel.sh --jobs 32
```

The script loads `herwig/stable-full-py3-rivet4`, regenerates the input
manifest with the repository's `4h_analyzer.py`, and launches the inputs with
`run_herwig_signal_inputs.py`.  Cards, `.run` files, logs, and ROOT output are
written beneath `HerwigSignalPoints/c3d4_40k/`.  A restart skips outputs that
already have both a nonempty ROOT file and the repository's completion marker.
An exclusive lock prevents two production launchers from running at once.

For a ten-event smoke test that cannot overwrite production output:

```bash
./Signals/c3d4_40k/run_herwig_parallel.sh \
  --jobs 1 --limit 1 --numevents 10 --tag smoke
```

Use `--dry-run` to list all prepared inputs.  See `--help` for the remaining
restart and validation options.

## Fast c3/d4 limits without pyhf

The campaign-specific preflight requires an exact match between
`metadata/points_153.csv` and the production Herwig ROOT files, checks 40,000
generated events per point, and compares each Herwig `Total:` cross section
with the corresponding MG5 LHE integrated weight:

```bash
./Signals/c3d4_40k/validate_analysis_inputs.py
```

The companion forced-splitting preflight requires exactly one completed
`gg -> hhhg`, forced `g -> b bbar` (`hhhbb`) Stage-2 ROOT file at every one of
the same 153 coordinates.  It reads the full-precision accepted cross section
from each point's `merge_summary.json`, rather than the rounded Stage-2 Herwig
table.  All points are exposed through the single campaign-style directory
`HerwigForcedSplitting/gg_hhhg_c3d4_10k_hhhbb_153/`.  Its event and point
entries are symlinks to the five immutable production shards, so the ROOT
files, weighted merged LHE files, logs, and normalization metadata are not
duplicated:

```bash
./Signals/c3d4_40k/validate_hhhbb_inputs.py
```

Run the complete-sample, fixed-parameter SM cross-fit and exact single-bin cut
limits with no pyhf score-shape stage using

```bash
./Signals/c3d4_40k/run_fast_limits_no_pyhf.sh
```

The wrapper runs both preflights first and pins
`HerwigSignalPoints/c3d4_40k/events`, the consolidated
`HerwigForcedSplitting/gg_hhhg_c3d4_10k_hhhbb_153/events` directory, and
`Signals/c3d4_40k` as the cross-section surface source.
The `hhhh` sample alone defines the classifier and validation-selected
thresholds.  The `hhhbb` sample is scored only afterwards and is added to the
final nominal signal yield with
`K_signal * BR(h->bb)^3 * btag^8`; it is never treated as background.

The combined limit is a common signal-strength limit for the fixed `hhhh` and
`hhhbb` predictions at each point, also reported as an equivalent `hhhh`
cross-section limit.  The wrapper writes to
`xgboost_c3d4_study_v2_uniform-smear-v1_fast-sm_153-hhhh-plus-hhhbb-cut-only/`
by default.
Set `C3D4_FAST_ANALYSIS_JOBS` or `C3D4_FAST_OUTDIR` to override the analysis
parallelism or output directory.
