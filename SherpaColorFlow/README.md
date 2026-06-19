# Sherpa Color-Flow Bundle

This directory vendors a patched Sherpa source tree for matrix-element-only
LHE generation with sampled Comix colour flows written into `ICOLUP`.

The patch is intended for high-multiplicity heavy-flavour samples where the
standard Sherpa LHEF output lacks useful colour-flow tags. It is an opt-in
large-`N_c` shower handoff approximation, not a full colour-density-matrix
export.

## Layout

- `sherpa/`: patched Sherpa source, based on official Sherpa commit
  `a7ba2c8b98da1fbc5e9fc290f1b8e6584afa71fe`.
- `patches/sherpa-lhef-color-flow-hack.patch`: portable source patch.
- `Examples/GluonFusion_GG_4bbbar_LHE/Sherpa.yaml`: corrected `gg -> 8b`
  card following the tiresias setup.
- `Examples/GluonFusion_GG_3bbbar_ccbar_LHE/Sherpa.yaml`: `gg -> 6b + c cbar`.
- `Examples/GluonFusion_GG_3bbbar_2j_LHE/Sherpa.yaml`: `gg -> 6b + 2j`.
- `Examples/GluonFusion_GG_2bbbar_2ccbar_LHE/Sherpa.yaml`: `gg -> 4b + 4c`.
- `Examples/GluonFusion_GG_2bbbar_ccbar_2j_LHE/Sherpa.yaml`: `gg -> 4b + 2c + 2j`.
- `Examples/GluonFusion_GG_2bbbar_4j_LHE/Sherpa.yaml`: `gg -> 4b + 4j`.
- `Examples/PP_Z_6bbbar_Zbb_DecayOS_LHE/Sherpa.yaml`: corrected
  `p p -> Z + 6b`, `Z -> b bbar` card.
- `sherpa/Examples/QuadrupleHiggs/`: mirrored copies of the `gg` example
  cards inside the patched Sherpa source tree.
- `scripts/validate_lhe_color.py`: generic LHE mass-shell and colour-flow
  validator.
- `scripts/build_sherpa_mpi.sh`: MPI build helper.
- `scripts/prepare_sherpa_run.py`: copies an example into a run directory,
  keeps `EVENTS` as the requested total with `MPI_EVENT_MODE: 1`, and applies
  the MPI progress settings used for long high-multiplicity runs.

## Build on physres1

```bash
cd QuadrupleHiggsAnalysis/SherpaColorFlow
MPI_HOME=/home/apapaefs/Projects/4H/sherpa-deps/openmpi-4.1.6 \
PREFIX=$HOME/Projects/4H/sherpa-colorflow-mpi \
BUILD_DIR=$HOME/Projects/4H/sherpa-colorflow-build \
./scripts/build_sherpa_mpi.sh
```

Then activate the installation:

```bash
export PATH=$HOME/Projects/4H/sherpa-colorflow-mpi/bin:$PATH
export LD_LIBRARY_PATH=$HOME/Projects/4H/sherpa-colorflow-mpi/lib:$HOME/Projects/4H/sherpa-colorflow-mpi/lib64:${LD_LIBRARY_PATH:-}
```

## Prepare and run examples

For exactly 100 total validation events over 20 MPI ranks:

```bash
cd QuadrupleHiggsAnalysis/SherpaColorFlow
./scripts/prepare_sherpa_run.py z6b runs/z6b_100evt \
  --total-events 100 --np 20 \
  --output-prefix pp_z_3bb_zbb_decayos_colorhack_100evt
cd runs/z6b_100evt
mpirun --use-hwthread-cpus -np 20 --bind-to hwthread --map-by hwthread Sherpa
```

The example cards and `prepare_sherpa_run.py` set:

```yaml
MPI_EVENT_MODE: 1
BATCH_MODE: 5
EVENT_DISPLAY_INTERVAL: 1000000
```

With these settings `EVENTS` is the requested total over the MPI job, and
Sherpa avoids the frequent progress-print cross-section synchronization that
can make high-rank, low-efficiency unweighting runs wait for the slowest rank
after every accepted event.

For a larger 64-rank production run:

```bash
cd QuadrupleHiggsAnalysis/SherpaColorFlow
./scripts/prepare_sherpa_run.py z6b runs/z6b_40000evt \
  --total-events 40000 --np 64 \
  --output-prefix pp_z_3bb_zbb_decayos_colorhack_40000evt
cd runs/z6b_40000evt
mpirun --use-hwthread-cpus -np 64 --bind-to hwthread --map-by hwthread Sherpa
```

Available example keys:

- `gg8b`: `gg -> 8b`
- `gg6bcc`: `gg -> 6b + c cbar`
- `gg6b2j`: `gg -> 6b + 2j`
- `gg4b4c`: `gg -> 4b + 4c`
- `gg4b2c2j`: `gg -> 4b + 2c + 2j`
- `gg4b4j`: `gg -> 4b + 4j`
- `z6b`: `pp -> Z + 6b`, `Z -> b bbar`

## Validate LHE output

Z-decay sample:

```bash
python3 ../../scripts/validate_lhe_color.py . \
  --prefix pp_z_3bb_zbb_decayos_colorhack_100evt \
  --expected-events 100 \
  --expect-final-abs-pdg 5 \
  --expect-final-count 8 \
  --forbid-final-pdg 23 \
  --require-first-qqbar-singlet 5
```

Pure 8b sample:

```bash
python3 ../../scripts/validate_lhe_color.py . \
  --prefix gg_4bbbar \
  --expect-final-abs-pdg 5 \
  --expect-final-count 8
```

Generated LHE files and build products are intentionally ignored by git.
