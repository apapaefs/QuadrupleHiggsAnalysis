# SM hh+4b HEFT signal

This singleton campaign showers the normalized Sherpa
`gg -> hh + b bbar b bbar` SM HEFT snapshot. The LHE contains 9,514 complete
events from 64 still-running shards and has
`sum(XWGTUP) = 9.62241e-06 pb`. Both stable Higgs bosons are forced to
`h0 -> b,bbar` in Herwig.

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
