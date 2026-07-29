# SM hh+4b HEFT signal

This singleton campaign showers the normalized Sherpa
`gg -> hh + b bbar b bbar` SM HEFT snapshot. The LHE contains 9,515 complete
events from 64 closed shards and has
`sum(XWGTUP) = 9.62241e-06 pb`. Both stable Higgs bosons are forced to
`h0 -> b,bbar` in Herwig.

The source production requested 20,000 events, but all 64 Sherpa processes
exited with code 2 after writing valid, closed LHE shards. The recovered 9,515
events pass the repository mass-shell and color-flow validator; this campaign
therefore uses that complete recovered set rather than claiming 20,000 events.

Run on Tiresias from the repository root:

```bash
module load herwig/stable-full-py3-rivet4
python3 run_herwig_signal_inputs.py \
  --list HerwigSignalPoints/sm_hh4b_heft/herwig_sm_hh4b_inputs_to_run.txt \
  --jobs 1
```

Add this option to any v2 c3/d4 analyzer mode:

```bash
--sm-hh4b-signal-dir HerwigSignalPoints/sm_hh4b_heft/events
```

The sample is scored only after each classifier and SM threshold are fixed. It
is excluded from training, threshold and score-binning optimization,
background totals, and c3/d4 limits. Each strategy writes exactly one row to
`postfit_sm_hh4b/result.{csv,json}`. The default signal K factor of 2,
`BR(h -> bb)^2`, and the common eight-tag factor are applied by the analyzer.

## c3-dependent cross section

The five Sherpa integrations at
`c3 = {-20, -2, -1, 0, 20}` are fitted to the HEFT quadratic

```text
sigma_pb(c3) = constant + linear*c3 + quadratic*c3^2
```

with the integration errors used as independent weights. Copy the completed
campaign to
`SherpaColorFlow/runs/gg_hh4b_c3_fit_14tev`, then produce the tracked fit with:

```bash
python3 SherpaColorFlow/scripts/fit_hh4b_c3_cross_section.py \
  --campaign-dir SherpaColorFlow/runs/gg_hh4b_c3_fit_14tev \
  --output Signals/sm_hh4b_heft/c3_xsec_fit.json
```

The fitter requires all five points and rejects any point above 1.5% relative
integration error. The standard fit file is discovered automatically by
`4h_analyzer.py`; `--sm-hh4b-c3-xsec-fit PATH` selects another validated fit.

When the fit is present, the final signal-reference table adds an hh+4b row at
every coupling coordinate already selected for hhhh and hhh+bb. These rows use
the cross section evaluated at that row's `c3` and the single SM hh+4b
analysis/XGBoost efficiency. They remain absent from classifier training,
optimization, backgrounds, and limits. The fit contains raw Sherpa generator
cross sections, so the analyzer applies the configured signal K-factor exactly
once through the existing hh+4b rate factor.
