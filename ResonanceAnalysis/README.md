# Resonant four-Higgs analysis

This workflow adds two mass-aware, resolved/merged XGBoost analyses alongside
the existing `c3,d4` analysis. It does not replace or modify that campaign.

- **Direct:** `gg_iota0_hhhh`, interpreted as
  \(pp\to\iota_0\to4h\), with a 42-point \(M_S\) scan.
- **Cascade:** `gg_iota0_eta0eta0_hhhh`, interpreted as
  \(pp\to\iota_0\to\eta_0\eta_0\to4h\), with a 441-point
  \((M_2,M_3)\) scan.

The scan manifest enforces \(M_S=M_3>4m_h\) for the direct topology and
\(M_2>2m_h\), \(M_3>4m_h\), and \(M_3>2M_2\) for the cascade. The generated
scalar widths are 1 GeV. Each signal point contains 10,000 Herwig events.

## Current publication workflow (2026-08-02)

- All **42 direct** and **441 cascade** AK4/AK8 signal pairs are complete.
- The immutable background manifest contains 14 AK4/AK8 feature pairs: the
  three SM multi-Higgs samples and all 11 conventional background samples.
- The `gg -> 4b+4j` row uses the complete compatible 29,616-event feature
  sample. The `gg -> 4b+2c+2j` normalization is taken from its audited LHE
  header. Neither input is silently substituted by the score-fit driver.
- The paper result is produced by `Code/resonance_score_fit_poisson.py` and
  contains no pyhf model or classifier threshold cut. Older smoke, threshold,
  and one-bin products below are retained only as provenance.

The publication driver trains one five-fold mass-conditioned classifier for
each topology. It compares four and five background-quantile score divisions,
chooses one division globally for the topology, and uses every score bin that
has sufficient background Monte Carlo support. A failing high-score bin is
merged downward. The minimum per retained bin is 25 independent background
source events and an effective count of five. This coarsening depends only on
background support, not on the signal or the resulting limit.

The SM Asimov counts contain the conventional backgrounds and the physical
SM `hhhh`, `hhh+bb`, and `hh+4b` yields. The expected upper limit is obtained
from the direct binned Poisson statistic at `q = 3.841`. The resonant template
is normalized to 1 fb before the four Higgs decays. No nuisance parameter,
expected band, or finite-Monte-Carlo term enters the likelihood.

On Tiresias, launch both topologies within a combined 180-thread ceiling with:

```bash
cd /home/apapaefs/Projects/QuadrupleHiggsAnalysis
bash Code/run_resonance_score_fit_tiresias.sh \
  /mnt/ssd2/Projects/4H/QuadrupleHiggsAnalysis_runs/ResonanceScoreFit/ak4ak8-scorefit-v1
```

The wrapper uses `/home/apapaefs/xgb-py310/bin/python`, sources
`/home/apapaefs/root310install`, applies NUMA interleaving, runs the direct and
cascade jobs concurrently, and writes restartable models and point caches.
Each topology produces the exact pointwise CSV/JSON limits, score-yield table,
binning audit, category diagnostics, PDF/PNG figures, input hashes, package
versions, timings, and a method manifest.

## Environment and build on Tiresias

Run from the project root:

```bash
cd ~/Projects/QuadrupleHiggsAnalysis

set +u
source ~/Projects/Herwig/Herwig-730-full-python3-rivet4/bin/activate
source ~/root310install/bin/thisroot.sh
source ~/xgb-py310/bin/activate
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

make -C Code FourHiggsResonanceAnalysis
python3 -m unittest discover -s Code/tests -p 'test_resonance_xgboost_analysis.py'
python3 -m unittest discover -s Code/tests -p 'test_resonance_fatjet_analysis.py'
```

The extractor reads the raw HwSim `Data` tree and requires
`bHadronMultiplicity`; it refuses branchless inputs. Jets first satisfy the
finite \(|\eta|<2.5\) preselection. Their energies are then fluctuated with the
deterministic CMS-style resolution, using exactly one Gaussian draw per
eta-accepted stored jet, before the analysis requirement on the smeared
\(p_T>20\) GeV is applied. The complete four-vector is multiplied by
\(E_{\rm smear}/E\), so the direction is fixed and the jet mass scales in
correlation with its energy. The preprocessing contract is
`cms-energy-uniform-fourvector-v1`. No b-, double-b-, c-, or light-tag
efficiency is applied in C++. The JSON diagnostics record upward and downward
20 GeV threshold migrations separately for true-b and non-b candidates, as
well as the maximum residual in the expected correlated mass scaling. Upward
migrations are additionally split by raw jet transverse momentum into
`[10,12)`, `[12,15)`, and `[15,20]` GeV bins to expose sensitivity to the
10 GeV HwSim storage threshold.

## Categories and normalization

A jet with capped B-hadron multiplicity two is one merged Higgs candidate. A
resolved Higgs is a pair of single-tag objects. For the fixed charm- and
light-mistag composition of a sample, the extractor enumerates all admissible
resolved/merged assignments, including every allowed choice of double-b jets
and every pairing of the selected single-tag objects into four Higgs
candidates. Each assignment is ranked with

\[
{\cal S}_h=\sum_{i=1}^{4}
\left(\frac{m_{h_i}-m_h}{m_h}\right)^2,
\qquad m_h=125\ {\rm GeV},
\]

and the assignment with the smallest value is retained. The feature tree
records this value as `best_score`, together with `second_score` for the
next-best assignment and `score_gap = second_score - best_score`; the latter
two are set to `-1` when no second assignment exists. The four reconstructed
Higgs candidates define three exclusive categories:

| Category | Definition |
| --- | --- |
| resolved | `n_merged = 0` |
| mixed | `n_merged = 1 or 2` |
| boosted | `n_merged = 3 or 4` |

Every feature-tree entry satisfies

\[
2n_{bb}+n_b+n_c+n_j=8,
\]

and the analysis applies the tagging factor exactly once,

\[
\epsilon_{bb}^{n_{bb}}\epsilon_b^{n_b}
\epsilon_c^{n_c}\epsilon_j^{n_j}.
\]

The complete event weight is

\[
w_i^{\rm phys}=\frac{w_i^{\rm raw}}{\sum_{\rm generated}w^{\rm raw}}
\,\mathcal L\,\sigma\,K\,r\,
{\rm BR}(h\to b\bar b)^{n_h}
\epsilon_{bb}^{n_{bb}}\epsilon_b^{n_b}
\epsilon_c^{n_c}\epsilon_j^{n_j}.
\]

The denominator is always `input_counter.sumw` from the extractor JSON
sidecar, never a manifest event count or the number of reconstructed entries.
Signed weights and `sumw2` are retained.

Defaults are

```text
luminosity                  3000 fb^-1
signal cross section        1 fb
signal K-factor             1
BR(h -> bb)                 0.5824
epsilon_b                   0.85
epsilon_bb nominal          0.85^2 = 0.7225
epsilon_bb conservative     0.30
epsilon_c                   0.10
epsilon_light               0.01
```

For the 1 fb signal hypothesis these give the exact pre-reconstruction
closures

```text
produced 4h events                    3000
after BR(h -> bb)^4                   345.1490798665728
after nominal eight-tag efficiency   94.0498539896
```

The 1 fb hypothesis means the complete direct or cascade production rate into
\(4h\), including the scalar cascade branching fractions but before the four
Higgs decays. Generated MG5/LHE signal cross sections are reported separately
as diagnostics and never replace this hypothesis.

## Classifier and limit definition

Direct and cascade use separate mass-aware classifiers. The direct model is
conditioned on \(M_S\); the cascade model is conditioned on \(M_2\), \(M_3\),
their hierarchy variables, the reconstructed \(m_{4h}\), and the three
possible \(hh+hh\) assignments. Backgrounds are evaluated again at every
physical signal mass point.

The analysis uses five deterministic source-local folds and a fixed XGBoost
configuration: 300 trees, depth 3, learning rate 0.05, 0.9 row/column
subsampling, and one XGBoost worker. There is **no Optuna import, trial, or
hyperparameter tuning**. Out-of-fold scores are divided by source event into a
40% threshold-selection partition and a disjoint 60% evaluation partition.
At each mass point, one score requirement is chosen with the nominal tagging
working point by minimizing the median expected exact one-bin Poisson
\(CL_s\) limit. The selection requires at least 25 unique background events and
\(N_{\rm eff}\geq10\); there is no relaxed fallback. The same requirement is
applied to both tagging scenarios, and the resolved, mixed, and boosted yields
are summed into one count.

The nominal result is a statistics-only projection with a known background.
It uses no pyhf model, score-shape likelihood, nuisance parameter, or fitted
Monte Carlo uncertainty. Finite-simulation information is retained in the
threshold and final-count support gates and in the output audits.

## Short technical smoke test

First create complete features for the three direct and three cascade points
selected by `--smoke-points 3`, followed by the three currently available
backgrounds. These products use the versioned, non-overwriting production
directory and are retained when the full extraction is resumed. The second
command also writes the resolved smoke-background manifest:

```bash
python3 Code/prepare_resonance_features.py \
  --analysis-root . \
  --kind signals \
  --only HW-gg_iota0_hhhh-miota_0525 \
  --only HW-gg_iota0_hhhh-miota_1100 \
  --only HW-gg_iota0_hhhh-miota_5000 \
  --only HW-gg_iota0_eta0eta0_hhhh-miota_0575-meta_0275 \
  --only HW-gg_iota0_eta0eta0_hhhh-miota_3000-meta_0625 \
  --only HW-gg_iota0_eta0eta0_hhhh-miota_5000-meta_2400 \
  --workers 6

python3 Code/prepare_resonance_features.py \
  --analysis-root . \
  --kind backgrounds \
  --background-manifest ResonanceAnalysis/background_manifest_smoke.csv \
  --workers 3
```

Then run three representative mass points and at most 250 reconstructed rows
per input:

```bash
python3 Code/resonance_xgboost_analysis.py \
  --analysis-root . \
  --topology direct \
  --mode smoke \
  --signal-root-dir ResonanceAnalysis/features/cms-energy-uniform-fourvector-v1 \
  --background-manifest ResonanceAnalysis/background_manifest_smoke_cms-energy-uniform-fourvector-v1.csv \
  --smoke-points 3 \
  --smoke-max-events 250 \
  --output-dir ResonanceAnalysis/results/smoke/cms-energy-uniform-fourvector-v1/direct

python3 Code/resonance_xgboost_analysis.py \
  --analysis-root . \
  --topology cascade \
  --mode smoke \
  --signal-root-dir ResonanceAnalysis/features/cms-energy-uniform-fourvector-v1 \
  --background-manifest ResonanceAnalysis/background_manifest_smoke_cms-energy-uniform-fourvector-v1.csv \
  --smoke-points 3 \
  --smoke-max-events 250 \
  --output-dir ResonanceAnalysis/results/smoke/cms-energy-uniform-fourvector-v1/cascade
```

Smoke mode checks loading, mass-aware features, folds, tagging closure, output
tables, and plotting. It is not an exclusion result and uses a separate output
directory from production.

## Complete background preparation

Preview the missing branch-complete background production:

```bash
python3 Code/prepare_resonance_background_roots.py \
  --analysis-root . \
  --manifest ResonanceAnalysis/background_manifest.csv \
  --workers 4 \
  --dry-run
```

Launch the long Herwig step:

```bash
mkdir -p ResonanceAnalysis/logs

nohup python3 -u Code/prepare_resonance_background_roots.py \
  --analysis-root . \
  --manifest ResonanceAnalysis/background_manifest.csv \
  --workers 4 \
  --herwig "$HOME/Projects/Herwig/Herwig-730-full-python3-rivet4/bin/Herwig" \
  > ResonanceAnalysis/logs/background-regeneration.log 2>&1 &
```

Here `--workers 4` means four independent concurrent Herwig instances. The
helper counts unique LHE `<event>` records and refuses hard-event recycling, so
copies of one hard event cannot leak between XGBoost folds. It writes new
`*-bhadmult-v1.root` samples under `ResonanceAnalysis/raw_backgrounds/`; the
legacy ROOT files are untouched.

Monitor with

```bash
tail -f ResonanceAnalysis/logs/background-regeneration.log
python3 -m json.tool ResonanceAnalysis/background_regeneration_status.json | less
```

Once regeneration is complete, extract new features for all 483 signal points
and all backgrounds. The old fixed-mass feature pairs are neither overwritten
nor reused:

```bash
nohup python3 -u Code/prepare_resonance_features.py \
  --analysis-root . \
  --kind all \
  --workers 8 \
  > ResonanceAnalysis/logs/feature-extraction-cms-energy-uniform-fourvector-v1.log 2>&1 &
```

`--workers 8` means eight independent extractor processes. Monitor
`ResonanceAnalysis/feature_campaign_status_cms-energy-uniform-fourvector-v1.json`
and the per-sample logs under
`ResonanceAnalysis/logs/features/cms-energy-uniform-fourvector-v1/`. The new
ROOT/JSON pairs are written below
`ResonanceAnalysis/features/cms-energy-uniform-fourvector-v1/`, and the helper
writes
`ResonanceAnalysis/background_manifest_cms-energy-uniform-fourvector-v1.csv`
with the matching background paths.

## Archived resolved-only shape analysis

The commands in this section reproduce the earlier AK4 resolved-only pyhf
study. They are retained for provenance and are not the nominal resonance
limit prescription. The current AK4+AK8 simple counting analysis is documented
below under `ak8-v1`.

Do not start these until the feature-extraction command exits successfully,
`ResonanceAnalysis/background_manifest_cms-energy-uniform-fourvector-v1.csv`
exists, and every non-optional row in that resolved manifest has a compatible
feature ROOT/JSON pair. The versioned feature status must also report an empty
`last_run_failures` list. The two topologies can then run in parallel:

```bash
nohup python3 -u Code/resonance_xgboost_analysis.py \
  --analysis-root . \
  --topology direct \
  --mode full \
  --signal-root-dir ResonanceAnalysis/features/cms-energy-uniform-fourvector-v1 \
  --background-manifest ResonanceAnalysis/background_manifest_cms-energy-uniform-fourvector-v1.csv \
  --output-dir ResonanceAnalysis/results/cms-energy-uniform-fourvector-v1/direct \
  > ResonanceAnalysis/logs/xgboost-direct-cms-energy-uniform-fourvector-v1.log 2>&1 &

nohup python3 -u Code/resonance_xgboost_analysis.py \
  --analysis-root . \
  --topology cascade \
  --mode full \
  --signal-root-dir ResonanceAnalysis/features/cms-energy-uniform-fourvector-v1 \
  --background-manifest ResonanceAnalysis/background_manifest_cms-energy-uniform-fourvector-v1.csv \
  --output-dir ResonanceAnalysis/results/cms-energy-uniform-fourvector-v1/cascade \
  > ResonanceAnalysis/logs/xgboost-cascade-cms-energy-uniform-fourvector-v1.log 2>&1 &
```

Full mode requires pyhf before training and exits nonzero if any required
expected-limit fit fails. Re-running the identical command resumes safely:
model hashes are checked and atomic `checkpoints/<point_id>.json` files skip
completed mass points. `run_config.json` fingerprints the manifests, feature
ROOT/summary inputs, extractor source, and analysis settings. If any of these
changes, reuse is refused; choose a new output directory rather than deleting
or overwriting the old result.

Background regeneration and feature extraction are also resumable. Existing
targets are validated and kept. A lone/incompatible ROOT or JSON feature file
is treated as an error, not overwritten silently.

## Archived resolved-only outputs

Each direct/cascade result directory contains:

- `input_cross_sections.csv/json`: generated cross-section diagnostics,
  physical background inputs, and the 1 fb signal normalization.
- `point_category_yields.csv/json`: generated, post-branching,
  reconstructed, tagged, and in-limit yields for resolved, mixed, boosted,
  and combined categories.
- `score_bin_yields.csv/json`: fold/bin yields, raw and effective entries,
  every background component, `TOTAL_BACKGROUND`, and exact pyhf aggregates.
- `point_limits.csv/json`: pointwise median expected 95% CL limits and bands.
- `normalization_audit.json` and `binning_audit.json`: closure and template
  validation details.
- `method_manifest.json`, `run_config.json`, `models/`, and `checkpoints/`:
  complete provenance and resume state.
- `input_report.html` and `direct_sigma95.{png,pdf}` or
  `cascade_sigma95_contour.{png,pdf}`.

The direct plot is the expected \(\sigma_{95}(M_S)\) line. The cascade plot is
the expected \(\sigma_{95}(M_2,M_3)\) plane over the sampled physical region;
no result is extrapolated outside the mass grid.

## Adding SM `hhh+bb` and `hh+4b`

When these samples are available, append rows to `background_manifest.csv`
with roles `sm_hhhbb` and `sm_hh4b`. Use `hbb_power=3` and `hbb_power=2`,
respectively, because only the Higgs decays receive physical
\({\rm BR}(h\to b\bar b)\) factors. Supply the physical cross section,
K-factor, unique generated/LHE event counts, source run/LHE, raw and feature
ROOT paths, and the appropriate exact c/light candidate composition.

Rows may be marked `optional=true` while their feature ROOT files are absent;
the omission is then recorded in `method_manifest.json`. Once added, regenerate
and extract them with the same helpers and run the analysis into a **new**
result directory. The changed manifest/input fingerprint deliberately prevents
stale models or limits from being reused.

## AK8/Soft-Drop analysis (`ak8-v1`)

The AK8 path is a separate, non-overwriting complement to the resolved
workflow above. It uses anti-
\(k_T\), \(R=0.8\) jets with Soft Drop \((\beta,z_{\rm cut})=(0,0.1)\)
and ungroomed \(\tau_{21}\). After the same deterministic CMS energy
smearing used for AK4 jets, eligible fat jets satisfy \(p_T>300\) GeV and
\(|\eta|<2.5\). No hard groomed-mass window is imposed.

An AK8 jet with exactly two or three ghost-associated B hadrons is a genuine
double-b candidate. A jet with fewer than two B hadrons uses the flat fake
probability, irrespective of its light/charm/single-b composition. Jets with
four or more B hadrons are recorded as `hh`-like diagnostics and are excluded
from the single-Higgs AK8 candidates. For every source event, all pass/fail
patterns of the four leading eligible candidates are retained. The physical
hypothesis weight is

\[
w_{e,h,s}=w_e^0\,
\epsilon_{bb,s}^{n_{tp}}(1-\epsilon_{bb,s})^{n_{tf}}
f_{bb,s}^{n_{fp}}(1-f_{bb,s})^{n_{ff}}
\epsilon_b^{n_b}\epsilon_c^{n_c}\epsilon_j^{n_j}.
\]

No random tag decision is made. All hypotheses from one generator event share
one fold. In every histogram bin their weights are first summed by generator
event and only then squared for MC statistics. The generator-level input
weight sum, never the repeated hypothesis-row sum, is the normalization
denominator.

### Build and install HwSim

The current HwSim plugin links the installed FastJet-contrib SoftDrop and
Nsubjettiness libraries. Build it in the Herwig environment:

```bash
source ~/Projects/Herwig/Herwig-730-full-python3-rivet4/bin/activate
source ~/root310install/bin/thisroot.sh

cd ~/Projects/Herwig/Herwig-730-full-python3-rivet4/src/Herwig-7.3.0/Contrib
make

cd hwsim
make -j8
make install
```

The new interfaces default to `FatJets No`; therefore existing cards and AK4
output remain unchanged. The AK8 regeneration helper enables the following
settings at run time without rewriting a serialized `.run` file:

```text
FatJets Yes
RFatParameter 0.8
PTCutFatJets 150 GeV
EtaCutFatJets 6.0
FatSoftDropBeta 0
FatSoftDropZCut 0.1
```

For the Python orchestration and analysis stages in a fresh Tiresias shell,
combine the Herwig runtime, Python-3.10 ROOT bindings, and XGBoost environment
in this order:

```bash
source ~/Projects/Herwig/Herwig-730-full-python3-rivet4/bin/activate
source ~/root310install/bin/thisroot.sh
source ~/xgb-py310/bin/activate
```

The regeneration commands below pass the absolute Herwig executable because
activating the Python virtual environment changes `PATH`.

### 1. Non-overwriting Herwig regeneration

Preview the selected signal and background jobs first:

```bash
cd ~/Projects/QuadrupleHiggsAnalysis

python3 Code/prepare_resonance_fatjet_roots.py \
  --analysis-root . \
  --kind all \
  --workers 8 \
  --dry-run
```

For a branch/kinematics smoke test without creating any production target or
manifest, use `--smoke-events`. The outputs go only to
`ResonanceAnalysis/smoke/ak8-v1/raw/`:

```bash
python3 Code/prepare_resonance_fatjet_roots.py \
  --analysis-root . \
  --kind all \
  --workers 3 \
  --herwig "$HOME/Projects/Herwig/Herwig-730-full-python3-rivet4/bin/Herwig" \
  --smoke-events 10 \
  --only HW-gg_iota0_hhhh-miota_0525 \
  --only HW-gg_iota0_hhhh-miota_1100 \
  --only HW-gg_iota0_hhhh-miota_5000 \
  --only HW-gg_iota0_eta0eta0_hhhh-miota_0575-meta_0275 \
  --only HW-gg_iota0_eta0eta0_hhhh-miota_3000-meta_0625 \
  --only HW-gg_iota0_eta0eta0_hhhh-miota_5000-meta_2400 \
  --only HW-gg_hhhh_SM \
  --only HW-gg_to_6b_2j \
  --only HW-gg_to_4b_4j
```

The long production command is intentionally left for an interactive Tiresias
session. Here `--workers 8` means eight independent Herwig instances:

```bash
mkdir -p ResonanceAnalysis/logs

nohup python3 -u Code/prepare_resonance_fatjet_roots.py \
  --analysis-root . \
  --kind all \
  --workers 8 \
  --herwig "$HOME/Projects/Herwig/Herwig-730-full-python3-rivet4/bin/Herwig" \
  > ResonanceAnalysis/logs/raw-ak8-v1.log 2>&1 &
```

Outputs are written only below
`HerwigSignalPoints/mass_scan_10k_ak8-v1/` and
`ResonanceAnalysis/raw_backgrounds_ak8-v1/`. Existing targets are validated
and kept; incompatible or partial targets cause an error rather than being
overwritten. The helper writes the matching immutable signal and background
manifests.

### 2. AK8 hypothesis feature extraction

Build the separate extractor and launch the long feature stage:

```bash
# Re-expose root-config and fastjet-config while compiling.  Activating the
# XGBoost environment last then selects its Python without losing ROOT's
# PYTHONPATH and library settings.
source ~/Projects/Herwig/Herwig-730-full-python3-rivet4/bin/activate
source ~/root310install/bin/thisroot.sh
make -C Code FourHiggsFatJetAnalysis
source ~/xgb-py310/bin/activate

nohup python3 -u Code/prepare_resonance_features.py \
  --analysis-root . \
  --feature-set fatjet-ak8-softdrop-v1 \
  --kind all \
  --workers 8 \
  > ResonanceAnalysis/logs/features-ak8-v1.log 2>&1 &
```

The feature contract is `fatjet-ak8-softdrop-v1`; products live below
`ResonanceAnalysis/features/ak8-v1/`. The extractor records the source
`event_index`, pass bitmask, tag exponents, exact B/C multiplicities, and all
normalization/probability-closure diagnostics. Hadron multiplicities, tag
exponents, and hypothesis identifiers are audit-only and cannot enter the
classifier.

### 3. Historical one-bin direct and cascade limits

This section documents the superseded threshold analysis and is not the
publication workflow.

Simple mode trains five fixed-configuration, mass-parameterized XGBoost
models and caches one out-of-fold score for every hypothesis row. Generator
events are then assigned by a deterministic hash to a 40% threshold-selection
partition or a disjoint 60% evaluation partition. At each physical mass point,
the nominal tagging working point selects one combined-category score
requirement in steps of 0.001 by minimizing the median expected exact one-bin
Poisson \(CL_s\) cross-section limit. Requirements within 1% of the optimum
prefer the largest background \(N_{\rm eff}\).

The selected region must contain at least 25 unique background source events
and \(N_{\rm eff}\geq10\) in both the validation and evaluation samples. There
is no low-statistics fallback: an unsupported point is reported as invalid and
has no quoted limit. The one nominally selected threshold is reused for the
conservative tagging scenario. Resolved, mixed, and boosted hypotheses all
enter the same final count; their separate yields remain available as
diagnostics.

The expected observation is the lower integer median of a background-only
Poisson distribution. The signal-event limit is the solution of

\[
\frac{P(N\leq n_{\rm med}\mid s_{95}+B)}
     {P(N\leq n_{\rm med}\mid B)}=0.05,
\qquad
\sigma_{95}=\frac{s_{95}}{S_{1\,{\rm fb}}}\,{\rm fb}.
\]

The background is treated as known. No pyhf model, score-shape fit, nuisance
parameter, or systematic uncertainty enters the limit; it is explicitly a
statistics-only phenomenological projection. The audited
`HW-gg_to_4b_2c_2j` LHE-header normalization (2751.78 fb with 7.47%
integration uncertainty) is retained in the provenance record, but that
uncertainty is not propagated through the counting model.

Run the two topologies independently or concurrently with:

```bash
nohup python3 -u Code/resonance_fatjet_xgboost_analysis.py \
  --analysis-root . \
  --feature-set fatjet-ak8-softdrop-v1 \
  --topology direct \
  --mode simple \
  --point-jobs 8 \
  --min-background-raw 25 \
  --min-background-neff 10 \
  --output-dir ResonanceAnalysis/results/ak8-v1-simple-poisson/direct \
  > ResonanceAnalysis/logs/xgboost-ak8-v1-simple-poisson-direct.log 2>&1 &

nohup python3 -u Code/resonance_fatjet_xgboost_analysis.py \
  --analysis-root . \
  --feature-set fatjet-ak8-softdrop-v1 \
  --topology cascade \
  --mode simple \
  --point-jobs 8 \
  --min-background-raw 25 \
  --min-background-neff 10 \
  --output-dir ResonanceAnalysis/results/ak8-v1-simple-poisson/cascade \
  > ResonanceAnalysis/logs/xgboost-ak8-v1-simple-poisson-cascade.log 2>&1 &
```

There is no second fit stage. Each result directory contains
`point_limits.csv/json`, `point_category_yields.csv/json`,
`threshold_audit.json`, `simple_poisson_limits.pdf/png`, the cached models and
scores, and `method_manifest.json`. Re-running the identical command verifies
the fingerprints and reuses completed point checkpoints.
