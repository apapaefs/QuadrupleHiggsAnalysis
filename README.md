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

## Sherpa Colour-Flow OpenMPI Runs

`SherpaColorFlow/` contains a vendored patched Sherpa source tree, corrected
8b and `Z+6b, Z -> b bbar` LHE cards, MPI build/run helpers, and a general LHE
colour-flow validator.

From a built local Sherpa MPI install, prepare a run directory and launch with
OpenMPI like this:

```bash
cd /mnt/ssd2/Projects/4H/QuadrupleHiggsAnalysis/SherpaColorFlow

export SHERPA_PREFIX=$PWD/install/sherpa-mpi
export PATH=$SHERPA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$SHERPA_PREFIX/lib/SHERPA-MC:$SHERPA_PREFIX/lib:$SHERPA_PREFIX/lib64:${LD_LIBRARY_PATH:-}
export LHAPDF_DATA_PATH=$SHERPA_PREFIX/share/SHERPA-MC/LHAPDF
export LHAPATH=$LHAPDF_DATA_PATH

./scripts/prepare_sherpa_run.py gg8b runs/gg8b_1000evt_np192 \
  --total-events 1000 \
  --np 192 \
  --output-prefix gg_4bbbar_1000evt_np192

cd runs/gg8b_1000evt_np192
/usr/bin/mpirun.openmpi \
  --use-hwthread-cpus \
  -np 192 \
  --bind-to hwthread \
  --map-by hwthread \
  Sherpa
```

`prepare_sherpa_run.py` writes `MPI_EVENT_MODE: 1`, so `EVENTS` is the total
requested over the MPI job, not the number per rank.

Available Sherpa process keys:

| Key | Process | Card |
| --- | --- | --- |
| `gg8b` | `g g -> b bbar b bbar b bbar b bbar` | `SherpaColorFlow/Examples/GluonFusion_GG_4bbbar_LHE/Sherpa.yaml` |
| `gg6bcc` | `g g -> b bbar b bbar b bbar c cbar` | `SherpaColorFlow/Examples/GluonFusion_GG_3bbbar_ccbar_LHE/Sherpa.yaml` |
| `gg6b2j` | `g g -> b bbar b bbar b bbar j j` | `SherpaColorFlow/Examples/GluonFusion_GG_3bbbar_2j_LHE/Sherpa.yaml` |
| `gg4b4c` | `g g -> b bbar b bbar c cbar c cbar` | `SherpaColorFlow/Examples/GluonFusion_GG_2bbbar_2ccbar_LHE/Sherpa.yaml` |
| `gg4b2c2j` | `g g -> b bbar b bbar c cbar j j` | `SherpaColorFlow/Examples/GluonFusion_GG_2bbbar_ccbar_2j_LHE/Sherpa.yaml` |
| `gg4b4j` | `g g -> b bbar b bbar j j j j` | `SherpaColorFlow/Examples/GluonFusion_GG_2bbbar_4j_LHE/Sherpa.yaml` |
| `z6b` | `p p -> Z + 6b`, `Z -> b bbar` | `SherpaColorFlow/Examples/PP_Z_6bbbar_Zbb_DecayOS_LHE/Sherpa.yaml` |

See `SherpaColorFlow/README.md` for build commands, validation commands, and
event-count monitoring.

## Data Policy

Generated ROOT files, Herwig logs/outputs, build products, and temporary test
samples are ignored. The current small LHE inputs needed for the documented
background/signal templates are tracked directly; larger generated campaigns
should stay outside git or use external storage.
