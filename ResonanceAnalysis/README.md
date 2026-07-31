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

## Current state (2026-07-16)

- All **42 direct** and **441 cascade** raw Herwig signal samples are ready.
  Their earlier fixed-mass feature files remain under
  `ResonanceAnalysis/features/` as a non-overwritten legacy campaign and are
  rejected by the current feature validator.
- Three branch-complete background raw samples are ready:
  `HW-gg_hhhh_SM`, `HW-gg_to_6b_2j`, and `HW-gg_to_4b_4j`.
- The other nine required QCD/reducible backgrounds still need the
  non-overwriting Herwig regeneration and feature-extraction steps below.
- `background_manifest_smoke.csv` selects the three ready backgrounds for
  technical smoke tests. A smoke result is deliberately marked non-physical.
- `background_manifest.csv` is the 11-component QCD/reducible background model
  plus SM \(hhhh\), and is required for the full result.

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
hyperparameter tuning**. Each rotation selects score bins from its validation
fold and applies them only to the disjoint test fold. The pyhf likelihood uses
category-by-fold channels, MC statistical errors, and a shared 10% background
normalization nuisance.

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

## Full direct and cascade analyses

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

## Outputs

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

### 3. Fast direct and cascade validation

Fast mode trains the same five fixed-configuration, mass-parameterized
XGBoost models used by the final analysis. It caches pointwise out-of-fold
scores and grouped multi-bin templates, scans one combined-category threshold
in steps of 0.001, and evaluates an exact one-bin Poisson \(CL_s\) limit on a
disjoint event-level test partition. It does not import or require pyhf.
The threshold scan first requires at least 25 unique validation-background
events and \(N_{\rm eff}\geq10\). With
`--fallback-background-neff 5`, only a point and tagging scenario for which no
threshold satisfies the primary requirement is retried at
\(N_{\rm eff}\geq5\); the unique-event requirement is never relaxed. The
result tables record the selected tier and both audit counts, and open squares
identify fallback limits in the fast plots. This fallback affects only the
fast cut-and-count validation. The pyhf template construction first requires
the configured unique-event and \(N_{\rm eff}\) thresholds in every bin and
both tagging scenarios for a 2--5-bin score shape. If no shape candidate
passes across the current scan, the default `--pyhf-low-mc-policy exclude`
uses a consistent resolved-only likelihood at every mass point; mixed and
boosted channels are not switched on at isolated points. An explicit
`--pyhf-low-mc-policy inclusive-diagnostic` run instead uses one inclusive
\([0,1]\) score bin when both scenarios still have positive background yield,
at least one unique event, and positive \(N_{\rm eff}\). This relaxed run is
always labelled as diagnostic and is never marked physics-valid. Each template
checkpoint records the policy, whether the inclusive fallback was attempted
and used, its relaxed requirements, and the per-scenario validation yields,
unique counts, and effective counts. `template_statistics_summary.json`
separately records the retained scope, excluded categories, and all
test-template bins that fail the primary MC-statistics requirements.

For the current 14-sample Tiresias background set, no additional events are
assumed: only the resolved templates meet the primary requirements. The
default result is therefore labelled `resolved_only`; mixed and boosted AK8
categories appear only in the existing-MC diagnostic run. Before retraining,
`prepare_resonance_background_normalization_manifest.py` creates a new,
immutable manifest that adopts the audited `HW-gg_to_4b_2c_2j` LHE-header
normalization (2751.78 fb with 7.47% integration uncertainty). The accompanying
audit sidecar is required for a result to be marked physics-valid; the existing
10% background-normalization nuisance covers that integration uncertainty.

On Tiresias,
`scripts/extended_scalar_mass_scan/tiresias_ak8_corrected_analysis_32.sh`
runs direct and cascade together with 16 workers each, validates the supported
result before promoting it, and can subsequently produce the labelled
all-category diagnostic with `start --with-diagnostic`. It consumes only the
existing ROOT/LHE inputs. The equivalent individual commands are:

```bash
nohup python3 -u Code/resonance_xgboost_analysis.py \
  --analysis-root . \
  --feature-set fatjet-ak8-softdrop-v1 \
  --topology direct \
  --mode fast \
  --point-jobs 8 \
  --min-background-neff 10 \
  --fallback-background-neff 5 \
  --pyhf-low-mc-policy exclude \
  --output-dir ResonanceAnalysis/results/ak8-v1-neff10-fallback5/direct \
  > ResonanceAnalysis/logs/xgboost-ak8-v1-neff10-fallback5-direct-fast.log 2>&1 &

nohup python3 -u Code/resonance_xgboost_analysis.py \
  --analysis-root . \
  --feature-set fatjet-ak8-softdrop-v1 \
  --topology cascade \
  --mode fast \
  --point-jobs 8 \
  --min-background-neff 10 \
  --fallback-background-neff 5 \
  --pyhf-low-mc-policy exclude \
  --output-dir ResonanceAnalysis/results/ak8-v1-neff10-fallback5/cascade \
  > ResonanceAnalysis/logs/xgboost-ak8-v1-neff10-fallback5-cascade-fast.log 2>&1 &
```

Fast plots are visibly labelled `FAST CUT-AND-COUNT VALIDATION — NOT FINAL
PYHF`. They are validation products, not the final exclusion result.

### 4. Cached full pyhf limits

After inspecting the fast normalization, category yields, grouped `sumw2`,
threshold audits, and direct/cascade plots, run only the missing pyhf fits:

```bash
nohup python3 -u Code/resonance_xgboost_analysis.py \
  --analysis-root . \
  --feature-set fatjet-ak8-softdrop-v1 \
  --topology direct \
  --mode full \
  --pyhf-jobs 8 \
  --min-background-neff 10 \
  --fallback-background-neff 5 \
  --pyhf-low-mc-policy exclude \
  --output-dir ResonanceAnalysis/results/ak8-v1-neff10-fallback5/direct \
  > ResonanceAnalysis/logs/xgboost-ak8-v1-neff10-fallback5-direct-full.log 2>&1 &

nohup python3 -u Code/resonance_xgboost_analysis.py \
  --analysis-root . \
  --feature-set fatjet-ak8-softdrop-v1 \
  --topology cascade \
  --mode full \
  --pyhf-jobs 8 \
  --min-background-neff 10 \
  --fallback-background-neff 5 \
  --pyhf-low-mc-policy exclude \
  --output-dir ResonanceAnalysis/results/ak8-v1-neff10-fallback5/cascade \
  > ResonanceAnalysis/logs/xgboost-ak8-v1-neff10-fallback5-cascade-full.log 2>&1 &
```

Full mode refuses to train or rescore. It verifies the mode-independent core
fingerprint, reuses the cached category-by-fold templates, and writes only
missing pyhf checkpoints. The pyhf fingerprint additionally binds the pyhf
version and fit settings. Any mismatch fails explicitly; a changed campaign
must use a new output directory.
