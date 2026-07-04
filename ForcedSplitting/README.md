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

Create the `HwSim:OutputLocation` directory before running Stage 2.  For
nested sample-specific directories, use the exact directory name produced by
the card or keep `--output-location events/` and separate outputs by run name.
