# HHH versus HHHH >=6 b-tag campaign

This campaign compares the basic six-or-more b-tag rate for

- inclusive full-loop `gg -> hhh`;
- the existing full-loop `gg -> hhhg`, forced `g -> b bbar` approximation;
- the existing full-loop `gg -> hhhh` signal.

It uses the exact 153 `(c3,d4)` coordinates in
`../c3d4_40k/metadata/points_153.csv`.  The new inclusive HHH sample contains
10,000 events per point at 14 TeV.  The HHHbb and HHHH inputs are the existing
10k and 40k Herwig campaigns, respectively.

The HHHbb sample definition is preserved: full-loop `gg -> hhhg` followed by
weighted forced `g -> b bbar`, with generation-level `pT_b > 15 GeV`,
`|eta_b| < 3`, and `DeltaR_bb > 0.3`.  The raw
`*_hhhbb_stage2.root` files are analyzed directly; earlier derived ROOT files
with an eight-b-jet preselection are not used.

The analysis applies the repository's
`cms-energy-uniform-fourvector-v1` response with seed `14101983`, selects
truth-b jets with smeared `pT > 20 GeV` and raw `|eta| < 2.5`, and evaluates
the probabilities for exactly 6, exactly 7, and at least 8 tags analytically
with a per-jet efficiency of 0.85.  There are no mistags, jet-pair separation
cuts, Higgs reconstruction cuts, or classifiers.

## Normalization

Herwig forces each Higgs to `b bbar` with branching ratio one.  The reported
physical cross sections therefore apply

```text
HHH:    sigma_prod * BR(h->bb)^3 * acceptance
HHHbb:  sigma_forced_split * BR(h->bb)^3 * weighted_acceptance
HHHH:   sigma_prod * BR(h->bb)^4 * acceptance
```

with `BR(h->bb) = 0.5824`.  HHHbb uses the exact probe-trial-corrected
`merged_xsec_pb` and its uncertainty from each `merge_summary.json`.

The primary ratio is

```text
sigma(HHHH, >=6 tags) /
  [sigma(HHH, >=6 tags) + sigma(HHHbb, >=6 tags)] .
```

This is deliberately labelled `additive_unmatched`: inclusive showered HHH
can itself contain `g -> b bbar`, so the sum is an estimate rather than an
overlap-safe merged prediction.  An HHH-only denominator is written as a
diagnostic.

## Commands

All commands are resumable and default to an aggregate budget of 64 logical
CPUs:

```bash
cd ~/Projects/QuadrupleHiggsAnalysis-hhh-ge6b

./Signals/hhh_c3d4_10k/run_campaign.sh status
./Signals/hhh_c3d4_10k/run_campaign.sh run-mg5
./Signals/hhh_c3d4_10k/run_campaign.sh status-mg5 --deep
./Signals/hhh_c3d4_10k/run_campaign.sh run-herwig
./Signals/hhh_c3d4_10k/run_campaign.sh status-herwig
./Signals/hhh_c3d4_10k/run_campaign.sh analyze
./Signals/hhh_c3d4_10k/run_campaign.sh plot
./Signals/hhh_c3d4_10k/run_campaign.sh validate
```

Use `--cpus N` to override the shared budget.  MG5 uses one point at a time
with `nb_core=N`; Herwig and the ROOT analysis use up to `N` single-threaded
workers with `OMP_NUM_THREADS=1`.

Run the isolated 100-event SM chain with:

```bash
./Signals/hhh_c3d4_10k/run_campaign.sh smoke
```

The smoke command never uses a production HHH run name and does not launch
the 153-point scan.

## Outputs

Results are written beneath `Signals/hhh_c3d4_10k/results/`, including:

- per-point HHH, HHHbb, HHH+HHHbb, and HHHH CSV/JSON cross-section tables;
- the all-in-one `ratio_points.csv` plus separately named pointwise CSV/JSON
  tables for the primary and HHH-only ratios;
- `c3d4_hhhh_ge6btag_over_hhh_plus_hhhbb_ge6btag_ratio_contours.pdf`;
- `c3d4_hhhh_ge6btag_over_hhh_ge6btag_ratio_contours.pdf`;
- matching PNG files, plot metadata, and validation metadata.

The contour plots use a piecewise-linear triangulation of the measured
pointwise ratios and do not extrapolate beyond the scan's convex hull.
