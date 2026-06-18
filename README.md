# Quadruple Higgs Analysis

Utilities for the 4-Higgs Herwig/HwSim workflow, SM-trained XGBoost optimization,
and c3/d4 limit plotting.

## Main Entry Points

- `4h_analyzer.py`: prepares Herwig inputs, runs missing C++ analysis outputs,
  trains/scores XGBoost, and writes c3/d4 limit plots.
- `run_herwig_signal_inputs.py`: launches prepared Herwig signal input files.
- `Code/FourHiggs8bAnalysis_smear_CMS.cc`: HwSim ROOT analysis that produces
  `*_var.smearCMS.root` variable trees.
- `Code/xgboost_root_varfiles_module.py`: XGBoost training, scoring, and limit
  plotting helpers.

## Background Metadata

`Backgrounds/processes.csv` is the source of truth for the local background LHE
files and cross sections. The generated `HW-*.in` files are included for the
current background samples.

## Sherpa Colour-Flow Generation

`SherpaColorFlow/` contains a vendored patched Sherpa source tree, corrected
8b and `Z+6b, Z -> b bbar` LHE cards, MPI build/run helpers, and a general LHE
colour-flow validator.

## Data Policy

Generated ROOT files, Herwig logs/outputs, build products, and temporary test
samples are ignored. The current small LHE inputs needed for the documented
background/signal templates are tracked directly; larger generated campaigns
should stay outside git or use external storage.
