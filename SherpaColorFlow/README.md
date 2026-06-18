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
- `Examples/PP_Z_6bbbar_Zbb_DecayOS_LHE/Sherpa.yaml`: corrected
  `p p -> Z + 6b`, `Z -> b bbar` card.
- `scripts/validate_lhe_color.py`: generic LHE mass-shell and colour-flow
  validator.
- `scripts/build_sherpa_mpi.sh`: MPI build helper.
- `scripts/prepare_sherpa_run.py`: copies an example into a run directory and
  sets exact total event counts for MPI runs.

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

For a larger 64-rank production run:

```bash
cd QuadrupleHiggsAnalysis/SherpaColorFlow
./scripts/prepare_sherpa_run.py z6b runs/z6b_40000evt \
  --total-events 40000 --np 64 \
  --output-prefix pp_z_3bb_zbb_decayos_colorhack_40000evt
cd runs/z6b_40000evt
mpirun --use-hwthread-cpus -np 64 --bind-to hwthread --map-by hwthread Sherpa
```

For the pure 8b example, replace `z6b` with `gg8b`.

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
