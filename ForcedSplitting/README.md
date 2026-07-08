# Forced g -> b bbar Splitting Smoke Tests

This directory contains the local steering helpers for the approximate
`gg -> hhh + 2b`, `gg -> hh + 4b`, and `gg -> h + 6b` samples.

Stage 1 reads the full-loop MG5 hard process and writes an intermediate LHE
after showering only final-state `g -> b,bbar`:

- `gg_hhhg`: `MinB = 2`, `MinSplitPairs = 1`, `LimitEmissions = OneFinalStateEmission`.
- `gg_hhgg`: `MinB = 4`, `MinSplitPairs = 2`, `RequireDistinctHardGluons = Yes`, `LimitEmissions = NoLimit`.
- `gg_hggg`: `MinB = 6`, `MinSplitPairs = 3`, `RequireDistinctHardGluons = Yes`, `LimitEmissions = NoLimit`.

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

python3 -m ForcedSplitting.herwig_cards stage1 gg_hggg \
  --input-lhe /mnt/ssd2/Projects/4H/MG5_aMC_v3_5_15/gg_hggg/Events/run_01/unweighted_events.lhe.gz \
  --output-prefix gg_hggg_stage1 \
  --events 100 \
  --card-out gg_hggg_stage1.in

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

When `ProbeTrials` is nonzero, the Stage-1 cards reserve additional accepted
shower attempts after the fixed probes.  By default the card writer sets
`ShowerHandler:MaxTry` and `ForceSplitVeto:ResetAfterAttempts` to Herwig's
`100000` shower-attempt ceiling, and caps the effective `ProbeTrials` value if
needed to leave 10000 post-probe attempts.  This matters for high-statistics
probe runs: the fixed probes measure the split acceptance, but the shower
still needs real post-probe attempts to produce an accepted event.

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

For `hh+gg` or `h+ggg`, replace `gg_hhhg` with `gg_hhgg` or `gg_hggg` and
choose a matching workdir and run name.  The command writes Stage-1 and Stage-2 Herwig cards, runs
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

First generate the MG5 hard samples on the same production `(c3,d4)` points
as the `hhhh` signal manifest.  The MG5 launcher reads the signal run-card
settings from `gg_4h_c3d4/Cards/run_card.dat`, keeps only production manifest
rows (`written` or `skipped_existing`), writes a MadEvent command deck, skips
points with an existing `unweighted_events.lhe(.gz)`, and launches
`bin/madevent` unless `--dry-run` is passed.

Dry-run the deck first:

```sh
PYTHONPATH=/mnt/ssd2/Projects/4H/QuadrupleHiggsAnalysis \
python3 -m ForcedSplitting.mg5_grid gg_hhhg \
  --mg5-root /mnt/ssd2/Projects/4H/MG5_aMC_v3_5_15 \
  --reference-grid-manifest /mnt/ssd2/Projects/4H/QuadrupleHiggsAnalysis/HerwigSignalPoints/c3d4_10k/herwig_inputs_manifest.csv \
  --events 1000 \
  --cores 324 \
  --dry-run
```

Then launch the missing MG5 points by dropping `--dry-run`.  The launcher
defaults to `--cores 324`, which writes `set run_mode 2` and `set nb_core 324`
to each launch block.  For `hh+gg` or `h+ggg`, replace `gg_hhhg` with
`gg_hhgg` or `gg_hggg`.  The generated runs are named like
`Events/run_gg_hhhg_4_<c3>_<d4>/`, `Events/run_gg_hhgg_4_<c3>_<d4>/`, or
`Events/run_gg_hggg_4_<c3>_<d4>/`.

## hhhbb c3/d4 Production Campaign

Use `hhhbb_campaign` for the production approximation
`gg -> hhhg, g -> b bbar`.  It uses the same 57-point reference grid as the
`hhhh` signal, splits each MG5 LHE into independent chunks, applies
`ProbeTrials` sidecar weights per chunk, merges the weighted split LHE files
once with `XSECUP = mean(XWGTUP)`, and then runs one Stage-2 HwSim card per
point with forced `h0 -> b,bbar`.

On odysseus, first prepare/check the MG5 hard samples:

```sh
cd ~/Projects/QuadrupleHiggsAnalysis
git pull origin main
module load herwig/730

python3 -m ForcedSplitting.hhhbb_campaign prepare-mg5 \
  --mg5-root ~/Projects/QuadrupleHiggsAnalysis/MG5_aMC_v3_5_15 \
  --reference-grid-manifest HerwigSignalPoints/c3d4_10k/herwig_inputs_manifest.csv \
  --events 10000 \
  --cores 324 \
  --dry-run

python3 -m ForcedSplitting.hhhbb_campaign prepare-mg5 \
  --mg5-root ~/Projects/QuadrupleHiggsAnalysis/MG5_aMC_v3_5_15 \
  --reference-grid-manifest HerwigSignalPoints/c3d4_10k/herwig_inputs_manifest.csv \
  --events 10000 \
  --cores 324
```

Then run the forced-splitting production campaign:

```sh
python3 -m ForcedSplitting.hhhbb_campaign run \
  --mg5-dir ~/Projects/QuadrupleHiggsAnalysis/MG5_aMC_v3_5_15/gg_hhhg \
  --reference-grid-manifest HerwigSignalPoints/c3d4_10k/herwig_inputs_manifest.csv \
  --workdir HerwigForcedSplitting/gg_hhhg_c3d4_10k_hhhbb \
  --events 10000 \
  --jobs 32 \
  --probe-trials 99999 \
  --allow-zero-probe-successes
```

While the MG5 grid is running, monitor the hard-process generation with:

```sh
python3 -m ForcedSplitting.hhhbb_campaign monitor-mg5 \
  --mg5-dir ~/Projects/QuadrupleHiggsAnalysis/MG5_aMC_v3_5_15/gg_hhhg \
  --reference-grid-manifest HerwigSignalPoints/c3d4_10k/herwig_inputs_manifest.csv \
  --tail 30
```

This reports completed LHEs, incomplete run directories, pending points,
recent MG5 debug-log errors, matching processes, and the tail of
`ForcedSplittingDecks/mg5_grid.log`.  Add `--count-events` for a slower check
that opens completed LHE files and counts their `<event>` blocks.

The MG5 deck writer defaults to multicore mode with `nb_core = 324` for this
campaign.  The generated `.mg5cmd` contains `set run_mode 2` and
`set nb_core 324` in every point launch block, and `monitor-mg5` reports the
configured core count from the latest deck.

The Stage-1 card writer caps the effective `ProbeTrials` value under Herwig's
legal `ShowerHandler:MaxTry` limit while leaving post-probe attempts to produce
accepted events.  Each point writes:

- `run_gg_hhhg_4_<c3>_<d4>/jobs/job*/` with chunk inputs, Stage-1 cards,
  sidecars, and weighted split LHE chunks;
- `run_gg_hhhg_4_<c3>_<d4>_split.weighted.merged.lhe.gz`;
- `run_gg_hhhg_4_<c3>_<d4>/merge_summary.json`;
- `events/run_gg_hhhg_4_<c3>_<d4>_hhhbb_stage2.root`;
- top-level `hhhbb_campaign_manifest.csv` and `hhhbb_campaign_summary.json`.

Check a completed or partially completed campaign with:

```sh
python3 -m ForcedSplitting.hhhbb_campaign check \
  --workdir HerwigForcedSplitting/gg_hhhg_c3d4_10k_hhhbb
```

After copying `HerwigForcedSplitting/gg_hhhg_c3d4_10k_hhhbb/events/` to the
analysis machine, score hhhh as before and add hhhbb only to the final
Poisson 95% CL signal yield:

```sh
python3 4h_analyzer.py --run-c3d4-limit-scan \
  --background-csv Backgrounds/processes.csv \
  --analysis-jobs 6 \
  --hhhbb-signal-dir HerwigForcedSplitting/gg_hhhg_c3d4_10k_hhhbb/events
```

The SM-trained XGBoost model and threshold are still optimized with the hhhh
SM signal plus backgrounds only.  The hhhbb grid is scored separately in
`xgboost_c3d4_scan/hhhbb_signal_scores/` and is combined with hhhh only in the
rows passed to `write_c3d4_limit_scan`.  The hhhh rate factor remains
`K_signal * BR(h->bb)^4 * btag^8`; the hhhbb rate factor is
`K_signal * BR(h->bb)^3 * btag^8`.

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

For `h+ggg`, use:

```sh
python3 -m ForcedSplitting.signal_pipeline gg_hggg \
  --mg5-dir /mnt/ssd2/Projects/4H/MG5_aMC_v3_5_15/gg_hggg \
  --outdir HerwigForcedSplitting/gg_hggg_c3d4_1k \
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

## HEFT hhh+bb 8b Validation

The `validation_hhhbb` helper compares `gg_hhhg_heft` plus one forced
final-state `g -> b,bbar` split with the direct `gg_hhhbb_heft` HEFT sample.
Both branches are then passed through the validation-only LHEWriter Higgs
decay card, forcing `h0 -> b,bbar` with BR set to one, and the final LHE-level
8b samples are plotted with the sample-report webpage style.

Run the baseline validation with explicit MG5 LHE paths:

```sh
cd /mnt/ssd2/Projects/4H/QuadrupleHiggsAnalysis
module load herwig/730

SPLIT_LHE=/home/apapaefs/Projects/QuadrupleHiggsAnalysis/MG5_aMC_v3_5_15/gg_hhhg_heft/Events/run_02/unweighted_events.lhe.gz
DIRECT_LHE=/home/apapaefs/Projects/QuadrupleHiggsAnalysis/MG5_aMC_v3_5_15/gg_hhhbb_heft/Events/run_01/unweighted_events.lhe.gz

python3 -m ForcedSplitting.validation_hhhbb run \
  --split-lhe "$SPLIT_LHE" \
  --direct-lhe "$DIRECT_LHE" \
  --workdir HerwigForcedSplitting/hhhbb_heft_validation_baseline \
  --events 10000 \
  --run-name hhhbb_heft_validation \
  --overwrite
```

For the split-filter weighted version, add probe trials:

```sh
python3 -m ForcedSplitting.validation_hhhbb run \
  --split-lhe "$SPLIT_LHE" \
  --direct-lhe "$DIRECT_LHE" \
  --workdir HerwigForcedSplitting/hhhbb_heft_validation_baseline_weighted \
  --events 10000 \
  --probe-trials 10000 \
  --run-name hhhbb_heft_validation \
  --overwrite \
  --allow-zero-probe-successes
```

If the default `NNPDF23_nlo_as_0119` LHAPDF set is not available on a
machine, either install it with LHAPDF or pass a local set explicitly, for
example `--pdf-name CT10nlo_as_0119`.

The report lands in `WORKDIR/report/index.html`, with
`validation_table.txt` and `report_metadata.json` beside it.  Regenerate only
the webpage from existing final 8b LHE files with:

```sh
python3 -m ForcedSplitting.validation_hhhbb compare \
  --split-lhe HerwigForcedSplitting/hhhbb_heft_validation_baseline/hhhbb_heft_validation_split_final8b.lhe \
  --direct-lhe HerwigForcedSplitting/hhhbb_heft_validation_baseline/hhhbb_heft_validation_direct_final8b.lhe \
  --split-source-lhe HerwigForcedSplitting/hhhbb_heft_validation_baseline/hhhbb_heft_validation_split_stage1.lhe \
  --direct-source-lhe "$DIRECT_LHE" \
  --output-dir HerwigForcedSplitting/hhhbb_heft_validation_baseline/report
```
