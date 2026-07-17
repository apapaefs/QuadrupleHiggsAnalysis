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
- `patches/sherpa-comix-dangling-current-cleanup.patch`: removes dangling
  Comix phase-space currents from every lookup registry before deletion. This
  prevents a first-point use-after-free in high-multiplicity processes such as
  `gg -> hh + b bbar b bbar`.
- `Examples/GluonFusion_GG_4bbbar_LHE/Sherpa.yaml`: corrected `gg -> 8b`
  card following the tiresias setup.
- `Examples/GluonFusion_GG_3bbbar_ccbar_LHE/Sherpa.yaml`: `gg -> 6b + c cbar`.
- `Examples/GluonFusion_GG_3bbbar_2j_LHE/Sherpa.yaml`: `gg -> 6b + 2j`.
- `Examples/GluonFusion_HEFT_GG_H_3bbbar_Hbb_LHE/Sherpa.yaml`: HEFT
  `gg -> h + 6b`, with `h -> b bbar` forced using the Herwig signal-card
  BR=1 convention.
- `Examples/GluonFusion_UFO_HEFT_GG_HH_LHE/`: stable-Higgs
  `gg -> hh` integration card for the Sherpa-adapted `heft_c3d4` UFO.
- `Examples/GluonFusion_UFO_HEFT_GG_HH_2bbbar_LHE/`: inclusive stable-Higgs
  `gg -> hh + b bbar b bbar` card retaining both the bottom-Yukawa and HEFT
  amplitudes.
- `Examples/GluonFusion_GG_2bbbar_2ccbar_LHE/Sherpa.yaml`: `gg -> 4b + 4c`.
- `Examples/GluonFusion_GG_2bbbar_ccbar_2j_LHE/Sherpa.yaml`: `gg -> 4b + 2c + 2j`.
- `Examples/GluonFusion_GG_2bbbar_4j_LHE/Sherpa.yaml`: `gg -> 4b + 4j`.
- `Examples/PP_Z_6bbbar_Zbb_DecayOS_LHE/Sherpa.yaml`: corrected
  `p p -> Z + 6b`, `Z -> b bbar` card.
- `Examples/GluonFusion_GG_TTbar_4b_AllHad_0c4j_DecayOS_LHE/Sherpa.yaml`: `gg -> ttbar + 4b`
  with all-hadronic top decays and no W-decay charm quarks.
- `Examples/GluonFusion_GG_TTbar_4b_AllHad_1c3j_DecayOS_LHE/Sherpa.yaml`: `gg -> ttbar + 4b`
  with all-hadronic top decays and exactly one W-decay charm quark.
- `Examples/GluonFusion_GG_TTbar_4b_AllHad_2c2j_DecayOS_LHE/Sherpa.yaml`: `gg -> ttbar + 4b`
  with all-hadronic top decays and two W-decay charm quarks.
- `sherpa/Examples/QuadrupleHiggs/`: mirrored copies of the `gg` example
  cards inside the patched Sherpa source tree.
- `scripts/validate_lhe_color.py`: generic LHE mass-shell and colour-flow
  validator.
- `scripts/merge_lhe_shards.py`: combines sharded LHE output into one closed
  LHE file while preserving the physical cross section.
- `scripts/merge_lhe_normalized_weights.py`: combines independently produced
  LHE source groups and rescales partially-unweighted `XWGTUP` values to a
  requested total cross section.
- `scripts/build_sherpa_mpi.sh`: MPI build helper.
- `scripts/prepare_heft_c3d4_sherpa_ufo.py`: validates and copies the tracked
  `heft_c3d4` UFO, adds the two embedded QCD powers to its `GH`/`Gphi`
  coupling metadata, and records source/adapted hashes.
- `scripts/run_gg_hh_c3_fit.py`: resumable sequential `c3=-2,-1,0` integration
  and exact quadratic fit for the 14 TeV `gg -> hh` card. It integrates with
  `-e 0` and does not generate production events.
- `scripts/prepare_sherpa_run.py`: copies an example into a run directory,
  keeps `EVENTS` as the requested total with `MPI_EVENT_MODE: 1`, and applies
  the MPI seed/progress settings used for long high-multiplicity runs. With
  `--seeded-jobs`, it also writes an executable
  `<run_dir>/run_seeded_generation.sh`; this is generated per run directory and
  is not stored under `scripts/`.

## Build on physres1 or physres2

The build helper enables Sherpa's UFO interface by default while leaving the
Sherpa Python add-on disabled. The interpreter selected at configure time is
embedded in `Sherpa-generate-model`, so use a persistent virtual environment
rather than a temporary Python installation. The following Python 3.9-compatible
environment is the reference setup on `physres2`:

```bash
export UFO_VENV=$HOME/Projects/4H/sherpa-ufo-venv
python3 -m venv "$UFO_VENV"
"$UFO_VENV/bin/python" -m pip install --upgrade pip
"$UFO_VENV/bin/python" -m pip install \
  numpy==1.26.4 sympy==1.13.3 opt-einsum==3.4.0
export PYTHON_EXECUTABLE=$UFO_VENV/bin/python
```

During conversion, Sherpa's bundled, source-pinned `opt_einsum` must precede
the pinned PyPI `opt-einsum` in `PYTHONPATH`. The PyPI implementation cannot
contract the symbolic scalar used by this high-multiplicity Lorentz model.

The wrapper first looks for the project OpenMPI installation and, when that is
not present, falls back to `/usr/lib64/openmpi`. Set `MPI_HOME` explicitly to
override that selection. For example, use the project installation on
`physres1`:

```bash
cd ~/Projects/QuadrupleHiggsAnalysis/SherpaColorFlow

export MPI_HOME=/home/apapaefs/Projects/4H/sherpa-deps/openmpi-4.1.6
export SHERPA_PREFIX=$HOME/Projects/4H/sherpa-colorflow-mpi
export BUILD_DIR=$HOME/Projects/4H/sherpa-colorflow-build

MPI_HOME=$MPI_HOME \
PYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
PREFIX=$SHERPA_PREFIX \
BUILD_DIR=$BUILD_DIR \
./scripts/build_sherpa_mpi.sh
```

On `physres2`, where the project OpenMPI prefix is absent, omit an inherited
override and let the wrapper select the system installation:

```bash
unset MPI_HOME
PYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
PREFIX=$SHERPA_PREFIX \
BUILD_DIR=$BUILD_DIR \
JOBS=32 \
./scripts/build_sherpa_mpi.sh
```

Set `SHERPA_ENABLE_UFO=OFF` on the wrapper command only when an internal-model
build without UFO conversion support is intentionally required.

Then activate the installation. Read `CMAKE_INSTALL_LIBDIR` from the build
cache instead of assuming that the platform installed libraries below `lib`
rather than `lib64` (or another GNU install directory):

```bash
export SHERPA_PREFIX=$HOME/Projects/4H/sherpa-colorflow-mpi
export BUILD_DIR=$HOME/Projects/4H/sherpa-colorflow-build

if [[ -z "${MPI_HOME:-}" ]]; then
  if [[ -x /home/apapaefs/Projects/4H/sherpa-deps/openmpi-4.1.6/bin/mpirun ]]; then
    export MPI_HOME=/home/apapaefs/Projects/4H/sherpa-deps/openmpi-4.1.6
  else
    export MPI_HOME=/usr/lib64/openmpi
  fi
fi

SHERPA_LIBDIR_REL=$(awk '/^CMAKE_INSTALL_LIBDIR:/ {
  sub(/^[^=]*=/, ""); value=$0
} END { print value }' "$BUILD_DIR/CMakeCache.txt")
test -n "$SHERPA_LIBDIR_REL"
case "$SHERPA_LIBDIR_REL" in
  /*) export SHERPA_LIBDIR=$SHERPA_LIBDIR_REL ;;
  *)  export SHERPA_LIBDIR=$SHERPA_PREFIX/$SHERPA_LIBDIR_REL ;;
esac

export PATH=$SHERPA_PREFIX/bin:$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SHERPA_LIBDIR/SHERPA-MC:$SHERPA_LIBDIR:$MPI_HOME/lib:$MPI_HOME/lib64:${LD_LIBRARY_PATH:-}
export LHAPDF_DATA_PATH=$SHERPA_PREFIX/share/SHERPA-MC/LHAPDF
export LHAPATH=$LHAPDF_DATA_PATH

Sherpa --version
```

Confirm that the installed model generator is enabled and tied to the intended
interpreter before converting a UFO model:

```bash
grep '^SHERPA_ENABLE_UFO:BOOL=ON$' "$BUILD_DIR/CMakeCache.txt"
grep '^SHERPA_ENABLE_PYTHON:BOOL=OFF$' "$BUILD_DIR/CMakeCache.txt"
test -x "$SHERPA_PREFIX/bin/Sherpa-generate-model"
test "$(head -n 1 "$SHERPA_PREFIX/bin/Sherpa-generate-model")" = \
  "#!$PYTHON_EXECUTABLE"
"$SHERPA_PREFIX/bin/Sherpa-generate-model" --help
```

The UFO interface generates and compiles a Sherpa C++ model library. UFO event
generation is supported by Comix, not Amegic. The Python environment and the
source UFO directory are only needed during conversion; the resulting model
library is sufficient at run time.

## Adapt and convert `heft_c3d4`

The original MadGraph UFO leaves the two powers of the strong coupling inside
`GH` and `Gphi` out of its coupling-order metadata. This is useful in some
MadGraph workflows but prevents Sherpa from assigning the intended running
`alpha_s` power. Never edit that tracked model in place. Prepare a validated
copy named `heft_c3d4_sherpa` instead:

```bash
cd ~/Projects/QuadrupleHiggsAnalysis
export UFO_SCRATCH=$HOME/Projects/4H/sherpa-ufo-models/heft_c3d4-sherpa-v1
mkdir -p "$UFO_SCRATCH"
python3 SherpaColorFlow/scripts/prepare_heft_c3d4_sherpa_ufo.py \
  MadGraphModels/heft_c3d4 \
  "$UFO_SCRATCH/heft_c3d4_sherpa"
```

The adapter requires the exact reviewed metadata for all eleven `GH`/`Gphi`
couplings before writing the copy. Its `sherpa_ufo_provenance.json` records the
source tree, adapter, adapted tree, and individual transformation hashes.

This UFO uses legacy absolute Python imports, so expose the scratch model
directory explicitly during conversion. Put Sherpa's installed Python
site-packages first: the generated launcher appends that directory too late to
override a conflicting package already imported from the virtual environment.
Use the same OpenMPI compiler wrappers as the Sherpa build. Do not pass
`--auto_convert`: the source is already Python 3 compatible, and automatic
conversion would create a different model name.

```bash
export UFO_MODEL=$UFO_SCRATCH/heft_c3d4_sherpa
export UFO_GENERATED=$UFO_SCRATCH/generated
SHERPA_PYTHON_SITE=$(awk -F"'" '/^sys.path.append/ { print $2 }' \
  "$SHERPA_PREFIX/bin/Sherpa-generate-model")
test -d "$SHERPA_PYTHON_SITE/ufo_interface"
test -x "$MPI_HOME/bin/mpicc"
test -x "$MPI_HOME/bin/mpicxx"
cd "$UFO_SCRATCH"
PYTHONPATH="$SHERPA_PYTHON_SITE:$UFO_MODEL${PYTHONPATH:+:$PYTHONPATH}" \
CC="$MPI_HOME/bin/mpicc" \
CXX="$MPI_HOME/bin/mpicxx" \
  "$SHERPA_PREFIX/bin/Sherpa-generate-model" "$UFO_MODEL" \
  --nmax 7 --ncore 32 --output_dir "$UFO_GENERATED"
```

The converter currently does not propagate failures from every CMake build or
install subprocess. A zero converter exit status is therefore insufficient.
Verify the fresh library, its dynamic dependencies, and representative two-,
three-, and four-gluon Higgs-pair source before using it:

```bash
MODEL_LIBRARY=$SHERPA_LIBDIR/SHERPA-MC/libSherpaheft_c3d4_sherpa.so
test -s "$MODEL_LIBRARY"
ldd "$MODEL_LIBRARY"
rg -n 'GGHH|GGGHH|GGGGHH' "$UFO_GENERATED"
```

## Integrate and fit `gg -> hh` at 14 TeV

The `gg_hh_ufo` and `gg_hh4b_ufo` aliases copy complete example bundles,
including their tagged UFO parameter card. Both Higgs bosons remain undecayed;
the cards disable Sherpa showers, fragmentation, MPI, beam remnants, and hard
decays so Herwig can perform the later Higgs decays.

The two-body card uses Sherpa's dedicated process-level `SChannel` map for the
massive 2-to-2 phase space. This is needed at `c3=-1`: the full matrix element
then reduces to the four-point `ggHH` contact interaction, for which the
graph-derived Comix phase-space channel has no propagator graph. `SChannel`
retains the requested full amplitude-order bounds and agrees with both the
default graph-derived result away from the contact-only point and the
independent `TChannel` 2-to-2 map.

The fit driver initializes and integrates the three symmetric basis points
sequentially using 32 ranks. It writes per-attempt logs, point JSON markers, a
CSV table, and the coefficient covariance under
`runs/gg_hh_c3_fit_14tev/`. Completed points with matching card, parameter-card,
model, and execution metadata are reused; mismatches are refused. The three
points use deterministic, widely separated base random seeds so their MPI rank
streams are distinct when propagating the point errors into the coefficient
covariance.

```bash
cd ~/Projects/QuadrupleHiggsAnalysis/SherpaColorFlow
./scripts/run_gg_hh_c3_fit.py \
  --np 32 \
  --mpirun "$MPI_HOME/bin/mpirun" \
  --model-library "$MODEL_LIBRARY" \
  --model-provenance "$UFO_MODEL/sherpa_ufo_provenance.json"
```

The fitted convention is
`sigma(kappa_lambda) = A*kappa_lambda^2 + B*kappa_lambda + C`, with
`kappa_lambda = 1+c3`. The driver uses `kappa_lambda=-1,0,+1` and propagates
the independent integration errors exactly into the `(A,B,C)` covariance.
It always calls the integration as `Sherpa -e 0 Sherpa.yaml`, so this step
does not create LHE events.

The inclusive high-multiplicity card is intended only for initialization in
the initial validation pass:

```bash
./scripts/prepare_sherpa_run.py gg_hh4b_ufo runs/gg_hh4b_ufo_init --np 32
cd runs/gg_hh4b_ufo_init
Sherpa -I Sherpa.yaml
```

If CMake prints `SHERPA: GIT IS NOT AVAILABLE!`, that is expected for this
vendored source tree and does not stop the build. It only means the binary will
not be stamped with a Git branch and revision.

If configuration already succeeded and only the compile or install step needs
to be resumed, run:

```bash
cmake --build "$BUILD_DIR" --parallel "$(getconf _NPROCESSORS_ONLN)"
cmake --install "$BUILD_DIR"
```

## Install LHAPDF sets

The example cards use Sherpa's LHAPDF interface:

```yaml
PDF_LIBRARY: LHAPDFSherpa
PDF_SET: NNPDF23_nlo_as_0119
MPI_PDF_LIBRARY: LHAPDFSherpa
MPI_PDF_SET: NNPDF23_nlo_as_0119
```

If Sherpa reports that `NNPDF23_nlo_as_0119` does not exist in any loaded
library, install the PDF grid into the Sherpa installation's LHAPDF data
directory:

```bash
export SHERPA_PREFIX=$HOME/Projects/4H/sherpa-colorflow-mpi
export PDFDIR=$SHERPA_PREFIX/share/SHERPA-MC/LHAPDF
export LHAPDF=$HOME/Projects/4H/sherpa-colorflow-build/EXTERNALSRC/src/downloadedlhapdf/bin/lhapdf

mkdir -p "$PDFDIR"
"$LHAPDF" --pdfdir "$PDFDIR" install NNPDF23_nlo_as_0119

test -f "$PDFDIR/NNPDF23_nlo_as_0119/NNPDF23_nlo_as_0119.info" && echo OK
```

If the LHAPDF installer cannot fetch the grid, download and unpack the set
directly:

```bash
cd "$PDFDIR"
curl -L -O https://lhapdfsets.web.cern.ch/lhapdfsets/current/NNPDF23_nlo_as_0119.tar.gz
tar -xzf NNPDF23_nlo_as_0119.tar.gz
```

Before running Sherpa, point the process at the same PDF directory:

```bash
export LHAPDF_DATA_PATH=$PDFDIR
export LHAPATH=$PDFDIR
```

## Prepare and run examples

Use `$MPI_HOME/bin/mpirun` explicitly so the run uses the same MPI
implementation selected for the build. This is the project OpenMPI prefix on
`physres1` and normally `/usr/lib64/openmpi` on `physres2`.

Activate the local Sherpa MPI install before preparing or launching runs:

```bash
cd ~/Projects/QuadrupleHiggsAnalysis/SherpaColorFlow

export SHERPA_PREFIX=$HOME/Projects/4H/sherpa-colorflow-mpi
export BUILD_DIR=$HOME/Projects/4H/sherpa-colorflow-build

if [[ -x /home/apapaefs/Projects/4H/sherpa-deps/openmpi-4.1.6/bin/mpirun ]]; then
  export MPI_HOME=/home/apapaefs/Projects/4H/sherpa-deps/openmpi-4.1.6
else
  export MPI_HOME=/usr/lib64/openmpi
fi

SHERPA_LIBDIR_REL=$(awk '/^CMAKE_INSTALL_LIBDIR:/ {
  sub(/^[^=]*=/, ""); value=$0
} END { print value }' "$BUILD_DIR/CMakeCache.txt")
case "$SHERPA_LIBDIR_REL" in
  /*) export SHERPA_LIBDIR=$SHERPA_LIBDIR_REL ;;
  *)  export SHERPA_LIBDIR=$SHERPA_PREFIX/$SHERPA_LIBDIR_REL ;;
esac

export PATH=$SHERPA_PREFIX/bin:$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SHERPA_LIBDIR/SHERPA-MC:$SHERPA_LIBDIR:$MPI_HOME/lib:$MPI_HOME/lib64:${LD_LIBRARY_PATH:-}
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
$MPI_HOME/bin/mpirun \
  --use-hwthread-cpus \
  -np 192 \
  --bind-to hwthread \
  --map-by hwthread \
  Sherpa
```

`prepare_sherpa_run.py` prints the follow-up commands after it writes the run
directory. Replace its unqualified `mpirun` with `$MPI_HOME/bin/mpirun`.

To save a log:

```bash
$MPI_HOME/bin/mpirun --use-hwthread-cpus -np 192 --bind-to hwthread --map-by hwthread Sherpa > sherpa_np192.log 2>&1
```

For exactly 100 total validation events over 20 MPI ranks:

```bash
./scripts/prepare_sherpa_run.py z6b runs/z6b_100evt \
  --total-events 100 --np 20 \
  --output-prefix pp_z_3bb_zbb_decayos_colorhack_100evt
cd runs/z6b_100evt
$MPI_HOME/bin/mpirun --use-hwthread-cpus -np 20 --bind-to hwthread --map-by hwthread Sherpa
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
$MPI_HOME/bin/mpirun --use-hwthread-cpus -np 64 --bind-to hwthread --map-by hwthread Sherpa
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
$MPI_HOME/bin/mpirun --use-hwthread-cpus -np 32 --bind-to hwthread --map-by hwthread Sherpa -e 0 Sherpa.yaml
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

## Merge independent weighted samples

Use `merge_lhe_shards.py` for shards from one homogeneous production. If the
final sample combines separate productions whose raw partially-unweighted
`XWGTUP` values are on different scales, use
`merge_lhe_normalized_weights.py` instead and pass the trusted total cross
section explicitly.

Each positional input is treated as one normalization source group. The script
rescales events inside each group as

```text
new XWGTUP_i = sigma_total * f_source * old XWGTUP_i / sum_source(old XWGTUP)
```

with `sum_source(f_source) = 1`. By default
`f_source = N_source / N_total`, so the merged file satisfies
`sum_i XWGTUP_i = sigma_total`.

Example for the `gg -> 8b` sample normalized to the shared integration result:

```bash
cd /mnt/ssd2/Projects/4H/QuadrupleHiggsAnalysis/SherpaColorFlow

scripts/merge_lhe_normalized_weights.py \
  runs/gg8b_template/events_from_shared_integration_3 \
  runs/gg8b_template/events_from_shared_integration_3_topup_6352 \
  runs/gg8b_template/events_from_shared_integration_4_10k \
  /path/to/gilberto_extracted_lhe_files \
  --total-xsec 0.00106993 \
  --total-xerr 3.10012e-05 \
  --output /path/to/merged_gg8b_colorflow_normalized.lhe
```

Do not feed an already-merged file back into this script unless it is meant to
be a single source group. To apportion the cross section by weighted effective
statistics instead of raw event counts, add
`--fraction-mode effective-events`. The script writes a JSON manifest next to
the output by default, including the event counts, raw weight sums, source
fractions, and scale factors used for each group.

Available process keys:

| Key | Process | Output prefix in card |
| --- | --- | --- |
| `gg8b` | `g g -> b bbar b bbar b bbar b bbar` | `gg_4bbbar` |
| `gg6bcc` | `g g -> b bbar b bbar b bbar c cbar` | `gg_3bbbar_ccbar` |
| `gg6b2j` | `g g -> b bbar b bbar b bbar j j` | `gg_3bbbar_2j` |
| `gg_h6b_heft` | HEFT `g g -> h + 6b`, `h -> b bbar` forced | `gg_heft_h_3bbbar_hbb` |
| `gg_hh_ufo` | UFO HEFT `g g -> h h`, stable Higgs bosons | `gg_hh_ufo_heft` |
| `gg_hh4b_ufo` | UFO HEFT `g g -> h h + b bbar b bbar`, stable Higgs bosons | `gg_hh_2bbbar_ufo_heft` |
| `gg4b4c` | `g g -> b bbar b bbar c cbar c cbar` | `gg_2bbbar_2ccbar` |
| `gg4b2c2j` | `g g -> b bbar b bbar c cbar j j` | `gg_2bbbar_ccbar_2j` |
| `gg4b4j` | `g g -> b bbar b bbar j j j j` | `gg_2bbbar_4j` |
| `ttbar4b_0c4j` | `g g -> ttbar + 4b`, all-hadronic, 0 W-charm | `ttbar_4b_allhad_0c4j_decayos` |
| `ttbar4b_1c3j` | `g g -> ttbar + 4b`, all-hadronic, 1 W-charm | `ttbar_4b_allhad_1c3j_decayos` |
| `ttbar4b_2c2j` | `g g -> ttbar + 4b`, all-hadronic, 2 W-charm | `ttbar_4b_allhad_2c2j_decayos` |
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
