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

## Herwig and Limit Analysis Pipeline

Start from the repository root and load the Herwig environment used for the
HwSim plugin:

```bash
cd /mnt/ssd2/Projects/4H/QuadrupleHiggsAnalysis
module load herwig/stable-full-py3-rivet4
```

The c3/d4 signal grid is driven by
`HerwigSignalPoints/c3d4_10k/herwig_inputs_to_run.txt`. If the corresponding
ROOT files are missing or need to be regenerated, launch the prepared signal
inputs with:

```bash
python3 run_herwig_signal_inputs.py \
  --list HerwigSignalPoints/c3d4_10k/herwig_inputs_to_run.txt \
  --jobs 57
```

The SM signal is trained separately by the analyzer. If its Herwig ROOT file is
missing, run:

```bash
printf '%s\n' "$PWD/Signals/HW-gg_hhhh_SM.in" > /tmp/herwig_sm_signal_inputs.txt

python3 run_herwig_signal_inputs.py \
  --list /tmp/herwig_sm_signal_inputs.txt \
  --jobs 1
```

For the current single-background workflow, use the local deduplicated
`g g -> 8b` LHE file as the input to `Backgrounds/HW-gg_to_8b.in`:

```text
Backgrounds/merged_gg8b_colorflow_sherpa_runs_plus_gilberto_20260626_dedup.lhe
```

Large LHE inputs are local campaign products and should not be committed to git.
Keep `Backgrounds/processes.csv` aligned with the local LHE filename and cross
section, then regenerate the background Herwig manifest:

```bash
python3 4h_analyzer.py \
  --prepare-background-herwig-inputs \
  --background-csv Backgrounds/processes.csv
```

Run Herwig over the selected background inputs:

```bash
python3 run_herwig_signal_inputs.py \
  --list Backgrounds/herwig_background_inputs_to_run.txt \
  --jobs 1
```

Finally run the SM-trained XGBoost optimization, score the c3/d4 signal grid,
run any missing C++ `*_var.smearCMS.root` analysis outputs, compute efficiencies,
and write the cross-section and 95% CL limit plots:

```bash
python3 4h_analyzer.py --run-c3d4-limit-scan \
  --background-csv Backgrounds/processes.csv \
  --analysis-jobs 6
```

The main physics defaults are:

```text
L = 3000 fb^-1
signal K-factor = 2
background K-factor = 2
b-tag efficiency = 0.85
c -> b mistag rate = 0.1
j -> b mistag rate = 0.01
```

Override them with `--luminosity`, `--signal-k-factor`,
`--background-k-factor`, `--btagging-rate`, `--c-mistag-rate`, and
`--light-mistag-rate` as needed.

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

For the seeded single-rank shard workflow, include `--seeded-jobs N` when
preparing the run. That option creates an executable
`run_seeded_generation.sh` inside the run directory, for example
`SherpaColorFlow/runs/gg8b_template/run_seeded_generation.sh`. Run it from that
same directory after the integration artifacts are present:

```bash
./scripts/prepare_sherpa_run.py gg8b runs/gg8b_template \
  --total-events 10000 \
  --np 32 \
  --output-prefix gg_4bbbar_10k \
  --seeded-jobs 64

cd runs/gg8b_template
Sherpa -I Sherpa.yaml
/usr/bin/mpirun.openmpi --use-hwthread-cpus -np 32 --bind-to hwthread --map-by hwthread Sherpa -e 0 Sherpa.yaml
./run_seeded_generation.sh 10000 64
```

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
