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
- `scripts/merge_lhe_shards.py`: combines sharded LHE output into one closed
  LHE file while preserving the physical cross section.
- `scripts/build_sherpa_mpi.sh`: MPI build helper.
- `scripts/prepare_sherpa_run.py`: copies an example into a run directory,
  keeps `EVENTS` as the requested total with `MPI_EVENT_MODE: 1`, and applies
  the MPI seed/progress settings used for long high-multiplicity runs. With
  `--seeded-jobs`, it also writes an executable
  `<run_dir>/run_seeded_generation.sh`; this is generated per run directory and
  is not stored under `scripts/`.

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

The OpenMPI executable on this machine is `/usr/bin/mpirun.openmpi`. Use it
explicitly so the run does not accidentally pick up another MPI implementation
from the environment.

Activate the local Sherpa MPI install before preparing or launching runs:

```bash
cd /mnt/ssd2/Projects/4H/QuadrupleHiggsAnalysis/SherpaColorFlow

export SHERPA_PREFIX=$PWD/install/sherpa-mpi
export PATH=$SHERPA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$SHERPA_PREFIX/lib/SHERPA-MC:$SHERPA_PREFIX/lib:$SHERPA_PREFIX/lib64:${LD_LIBRARY_PATH:-}
export LHAPDF_DATA_PATH=$SHERPA_PREFIX/share/SHERPA-MC/LHAPDF
export LHAPATH=$LHAPDF_DATA_PATH
```

For exactly 1000 total `gg -> 8b` events over 192 MPI ranks:

```bash
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

`prepare_sherpa_run.py` prints the follow-up commands after it writes the run
directory. The printed MPI command uses `mpirun`; on physres1 use
`/usr/bin/mpirun.openmpi` in its place.

To save a log:

```bash
/usr/bin/mpirun.openmpi --use-hwthread-cpus -np 192 --bind-to hwthread --map-by hwthread Sherpa > sherpa_np192.log 2>&1
```

For exactly 100 total validation events over 20 MPI ranks:

```bash
./scripts/prepare_sherpa_run.py z6b runs/z6b_100evt \
  --total-events 100 --np 20 \
  --output-prefix pp_z_3bb_zbb_decayos_colorhack_100evt
cd runs/z6b_100evt
/usr/bin/mpirun.openmpi --use-hwthread-cpus -np 20 --bind-to hwthread --map-by hwthread Sherpa
```

The example cards and `prepare_sherpa_run.py` set:

```yaml
MPI_EVENT_MODE: 1
MPI_SEED_MODE: 1
BATCH_MODE: 5
EVENT_DISPLAY_INTERVAL: 100
```

With these settings `EVENTS` is the requested total over the MPI job, and
Sherpa uses additive per-rank seeds instead of the default multiplicative
seeding. This avoids rank classes with systematically poor random streams, and
the progress settings avoid frequent cross-section synchronization that can
make high-rank, low-efficiency unweighting runs wait for the slowest rank after
every accepted event.

For a larger 64-rank production run:

```bash
./scripts/prepare_sherpa_run.py z6b runs/z6b_40000evt \
  --total-events 40000 --np 64 \
  --output-prefix pp_z_3bb_zbb_decayos_colorhack_40000evt
cd runs/z6b_40000evt
/usr/bin/mpirun.openmpi --use-hwthread-cpus -np 64 --bind-to hwthread --map-by hwthread Sherpa
```

To reuse one integration for many single-rank generation shards, ask the setup
script to write a seeded runner. The runner is created inside the run
directory named in the second argument:

```bash
./scripts/prepare_sherpa_run.py gg8b runs/gg8b_template \
  --total-events 10000 \
  --np 32 \
  --output-prefix gg_4bbbar_10k \
  --seeded-jobs 64
```

This creates `runs/gg8b_template/run_seeded_generation.sh`. If that file is
missing, the run directory was prepared without `--seeded-jobs` or you are not
inside the run directory.

Run the integration once in `runs/gg8b_template`. Then launch the single-rank
event shards from that same run directory:

```bash
cd runs/gg8b_template
Sherpa -I Sherpa.yaml
/usr/bin/mpirun.openmpi --use-hwthread-cpus -np 32 --bind-to hwthread --map-by hwthread Sherpa -e 0 Sherpa.yaml
./run_seeded_generation.sh 10000 64
```

The two trailing numbers are adjustable: total requested events first, number
of single-rank Sherpa jobs second. Each job gets its own `events/job_XXXX`
working directory with copied `Process/` and `Results_PartiallyUnweighted*`
artifacts, a unique seed, and a unique LHE prefix. The runner refuses to use a
non-empty `OUTBASE`, so use a fresh output directory when adding more events:

```bash
OUTBASE=events_more_20k BASE_SEED=4321 ./run_seeded_generation.sh 20000 164
```

Monitor completed LHE events by counting closed event blocks:

```bash
rg -c '^</event>' runs/gg8b_1000evt_np192/gg_4bbbar_1000evt_np192_*.lhe 2>/dev/null \
  | awk -F: '{s += $2} END {print s+0 " / 1000 events"}'
```

## Merge sharded LHE output

After all single-rank shards finish, merge them from the parent run directory:

```bash
python3 ../../scripts/merge_lhe_shards.py events \
  --prefix gg_4bbbar_10k_ \
  --output gg_4bbbar_10k_merged.lhe \
  --expected-events 10000
```

The merge script writes one header, one init block, all complete event blocks,
and one final `</LesHouchesEvents>` footer. If an input shard is missing the
final LHE footer, the script prints a warning for that file. The merged output
is still closed correctly, but incomplete trailing `<event>` blocks are skipped
unless `--strict` is used.

To repair unclosed input shards after the jobs have definitely stopped, add:

```bash
python3 ../../scripts/merge_lhe_shards.py events \
  --prefix gg_4bbbar_10k_ \
  --output gg_4bbbar_10k_merged.lhe \
  --expected-events 10000 \
  --fix-unclosed-inputs
```

`--fix-unclosed-inputs` truncates each affected input after its last complete
`</event>` block and appends the final LHE footer. It creates `.bak` backups by
default; use `--no-backup` only for disposable test data.

The script validates that all shard `<init>` blocks agree. When sibling
`sherpa_*.log` files are present, it also reads the Sherpa-reported physical
cross section and writes that into the merged `<init>` process line. This is
needed for the seeded single-rank workflow, where the shard LHE files may carry
placeholder init process lines such as `1 1 1 1`. The cross section is not
summed over shards.

Available process keys:

| Key | Process | Output prefix in card |
| --- | --- | --- |
| `gg8b` | `g g -> b bbar b bbar b bbar b bbar` | `gg_4bbbar` |
| `gg6bcc` | `g g -> b bbar b bbar b bbar c cbar` | `gg_3bbbar_ccbar` |
| `gg6b2j` | `g g -> b bbar b bbar b bbar j j` | `gg_3bbbar_2j` |
| `gg4b4c` | `g g -> b bbar b bbar c cbar c cbar` | `gg_2bbbar_2ccbar` |
| `gg4b2c2j` | `g g -> b bbar b bbar c cbar j j` | `gg_2bbbar_ccbar_2j` |
| `gg4b4j` | `g g -> b bbar b bbar j j j j` | `gg_2bbbar_4j` |
| `z6b` | `p p -> Z + 6b`, `Z -> b bbar` | `pp_z_3bb_zbb_decayos_colorhack` |

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
