# Forced g -> b bbar Splitting Smoke Tests

This directory contains the local steering helpers for the approximate
`gg -> hhh + 2b` and `gg -> hh + 4b` samples.

Stage 1 reads the full-loop MG5 hard process and writes an intermediate LHE
after showering only final-state `g -> b,bbar`:

- `gg_hhhg`: `MinB = 2`, `MinSplitPairs = 1`, `LimitEmissions = OneFinalStateEmission`.
- `gg_hhgg`: `MinB = 4`, `MinSplitPairs = 2`, `RequireDistinctHardGluons = Yes`, `LimitEmissions = NoLimit`.

Both cards set `SplitMinBPt = 15*GeV`, `SplitMaxBEta = 3.0`,
`SplitMinDeltaR = 0.3`, and `SplitMinDeltaRToOtherB = 0.3`.  Higgs decays
are intentionally absent in Stage 1: the card disables the Higgs entry in
`ShowerHandler:DecayInShower`, uses no hadronization, and writes stable Higgs
bosons plus shower-created b quarks to LHE.

Stage 2 reads the split LHE, forces `h0 -> b,bbar`, and runs the usual
HwSim-style analysis chain.

`fixtures/toy_hhgg.lhe` is a synthetic `gg -> hhgg` file for software testing
only.  It has momentum conservation and balanced LHE colour tags, but no
physics interpretation and must not be included in production manifests or
rate summaries.

Example card generation:

```sh
python3 -m ForcedSplitting.herwig_cards stage1 gg_hhhg \
  --input-lhe /mnt/ssd2/Projects/4H/MG5_aMC_v3_5_15/gg_hhhg/Events/run_01/unweighted_events.lhe.gz \
  --output-prefix gg_hhhg_stage1 \
  --events 100 \
  --card-out gg_hhhg_stage1.in

python3 -m ForcedSplitting.herwig_cards stage1 gg_hhgg \
  --input-lhe toy_hhgg.lhe \
  --output-prefix toy_hhgg_stage1 \
  --events 2 \
  --card-out toy_hhgg_stage1.in

python3 -m ForcedSplitting.herwig_cards stage2 \
  --input-lhe toy_hhgg_stage1.lhe \
  --output-location events/ \
  --events 1 \
  --run-name toy_hhgg_stage2 \
  --card-out toy_hhgg_stage2.in
```

Herwig's LHEWriter may write a single-process split LHE with `LPRUP = 0` in
the `<init>` block while the events use `IDPRUP = 1`.  Normalize the split LHE
before Stage 2 to avoid the Herwig `found undeclared processes` warning:

```sh
python3 -m ForcedSplitting.lhe_validation normalize-process-ids toy_hhgg_stage1.lhe
```

For final production with nonzero `ProbeTrials`, apply the forced-splitting
acceptance sidecar before Stage 2:

```sh
python3 -m ForcedSplitting.lhe_weights \
  toy_hhgg_stage1.lhe \
  toy_hhgg_stage1.force_split.weights \
  toy_hhgg_stage1.weighted.lhe
```

This multiplies each event weight by the event's `p_hat` estimate from the
sidecar and updates the LHE `<init>` cross section.  With equal unit input
event weights, the weighted LHE satisfies
`mean(XWGTUP) = XSECUP * mean(p_hat)`.

Create the `HwSim:OutputLocation` directory before running Stage 2.  For
nested sample-specific directories, use the exact directory name produced by
the card or keep `--output-location events/` and separate outputs by run name.

## Single-Command Chain

From the directory containing the MG LHE file, run the full chain with:

```sh
PYTHONPATH=/mnt/ssd2/Projects/4H/QuadrupleHiggsAnalysis \
python3 -m ForcedSplitting.run_chain gg_hhhg \
  --input-lhe unweighted_events.lhe.gz \
  --workdir /mnt/ssd2/Projects/4H/QuadrupleHiggsAnalysis/HerwigForcedSplitting/gg_hhhg_run01_p90k \
  --run-name gg_hhhg_run01 \
  --events 1000 \
  --probe-trials 90000
```

For `hh+gg`, replace `gg_hhhg` with `gg_hhgg` and choose a matching workdir
and run name.  The command writes Stage-1 and Stage-2 Herwig cards, runs
`Herwig read/run` for both stages, normalizes the Stage-1 LHE process ids,
applies the sidecar `p_hat` weights when `--probe-trials` is nonzero, verifies
the weighted LHE, and writes a JSON summary such as
`gg_hhhg_run01_summary.json`.

The runner counts `<event>` blocks in the input LHE first.  By default it
aborts if `--events` is larger than the input event count, because Herwig can
reopen the Les Houches file and reuse hard events.  For production, generate
at least as many MG hard events as the requested Stage-1 events, or lower
`--events`.  For diagnostic plumbing tests only, pass
`--allow-input-oversampling`.

By default the command aborts if any sidecar row has `probe_successes = 0`,
because Herwig skips zero-weight LHE events under `VarNegWeight`.  For tiny
diagnostic runs only, pass `--allow-zero-probe-successes`.

## MG5 Signal-Point Pipeline

For an MG5 process directory with an `Events/` subdirectory, prepare the paired
Stage-1 and Stage-2 Herwig cards with:

```sh
python3 -m ForcedSplitting.signal_pipeline gg_hhhg \
  --mg5-dir /mnt/ssd2/Projects/4H/MG5_aMC_v3_5_15/gg_hhhg_c3d4 \
  --outdir HerwigForcedSplitting/gg_hhhg_c3d4_1k \
  --events 1000 \
  --reference-grid-manifest HerwigSignalPoints/c3d4_10k/herwig_inputs_manifest.csv
```

For `hh+gg`, use:

```sh
python3 -m ForcedSplitting.signal_pipeline gg_hhgg \
  --mg5-dir /mnt/ssd2/Projects/4H/MG5_aMC_v3_5_15/gg_hhgg_c3d4 \
  --outdir HerwigForcedSplitting/gg_hhgg_c3d4_1k \
  --events 1000 \
  --reference-grid-manifest HerwigSignalPoints/c3d4_10k/herwig_inputs_manifest.csv
```

The pipeline writes:

- `forced_splitting_manifest.csv`;
- `stage1_inputs_to_run.txt`;
- `stage1_outputs_to_normalize.txt`;
- `stage1_outputs_to_reweight.txt`;
- `stage2_inputs_to_run.txt`;
- one Stage-1 LHEWriter card and one Stage-2 HwSim card per selected MG5 run.

The reference-grid manifest is optional, but should be used for production so
that the selected `(c3,d4)` points match the existing hhhh grid.  Runs outside
that grid are marked `skipped_not_in_reference_grid` in the manifest.

Run Stage 1 first:

```sh
cd HerwigForcedSplitting/gg_hhhg_c3d4_1k
while read card; do Herwig read "$card"; Herwig run "${card%.in}.run"; done < stage1_inputs_to_run.txt
```

Normalize the Stage-1 split LHE process ids:

```sh
while read lhe; do python3 -m ForcedSplitting.lhe_validation normalize-process-ids "$lhe"; done < stage1_outputs_to_normalize.txt
```

For final production runs prepared with nonzero `--probe-trials`, apply the
sidecar acceptance factors:

```sh
while read inlhe weights outlhe; do python3 -m ForcedSplitting.lhe_weights "$inlhe" "$weights" "$outlhe"; done < stage1_outputs_to_reweight.txt
```

Check the sidecars before Stage 2:

```sh
awk 'NF && $1 !~ /^#/ {n++; if ($3 == 0) z++; sum += $4} END {print "rows", n, "zero_success_rows", z+0, "mean_p_hat", sum/n}' *.force_split.weights
```

Rows with `probe_successes = 0` become zero-weight events in the weighted LHE.
Herwig skips zero-weight LHE events under `VarNegWeight`, so final production
should use enough `ProbeTrials` that zero-success rows are absent or negligible.

Then run Stage 2 after the split LHE files exist:

```sh
while read card; do Herwig read "$card"; Herwig run "${card%.in}.run"; done < stage2_inputs_to_run.txt
```

For final production with split-filter corrections, repeat the preparation with
a nonzero `--probe-trials` value.  In that mode the pipeline writes Stage-2
cards that read the `.weighted.lhe` files.
