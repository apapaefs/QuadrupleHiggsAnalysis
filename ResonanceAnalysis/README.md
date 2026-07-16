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

- All **42 direct** and **441 cascade** feature ROOT files are complete under
  `ResonanceAnalysis/features/`.
- Three branch-complete background feature samples are ready:
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

source /etc/profile.d/modules.sh
module load herwig/stable-full-py3-rivet4
source ~/xgb-py310/bin/activate

set +u
source ~/root310install/bin/thisroot.sh
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

make -C Code FourHiggsResonanceAnalysis
python3 -m unittest discover -s Code/tests -p 'test_resonance_xgboost_analysis.py'
```

The extractor reads the raw HwSim `Data` tree and requires
`bHadronMultiplicity`; it refuses branchless inputs. It applies the raw
\(p_T>20\) GeV and \(|\eta|<2.5\) acceptance, deterministic CMS-style energy
smearing that preserves the jet mass, and writes a `ResonanceFeatures` tree.
No b-, double-b-, c-, or light-tag efficiency is applied in C++.

## Categories and normalization

A jet with capped B-hadron multiplicity two is one merged Higgs candidate. A
resolved Higgs is a pair of single-tag objects. The four reconstructed Higgs
candidates define three exclusive categories:

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

First validate or create the three currently available background feature
pairs. Existing compatible ROOT/JSON pairs are kept:

```bash
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
  --background-manifest ResonanceAnalysis/background_manifest_smoke.csv \
  --smoke-points 3 \
  --smoke-max-events 250 \
  --output-dir ResonanceAnalysis/results/smoke/direct

python3 Code/resonance_xgboost_analysis.py \
  --analysis-root . \
  --topology cascade \
  --mode smoke \
  --background-manifest ResonanceAnalysis/background_manifest_smoke.csv \
  --smoke-points 3 \
  --smoke-max-events 250 \
  --output-dir ResonanceAnalysis/results/smoke/cascade
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

Once regeneration is complete, validate the 483 existing signal feature pairs
and produce all missing background features:

```bash
nohup python3 -u Code/prepare_resonance_features.py \
  --analysis-root . \
  --kind all \
  --workers 8 \
  > ResonanceAnalysis/logs/feature-extraction.log 2>&1 &
```

`--workers 8` means eight independent extractor processes. Monitor
`ResonanceAnalysis/feature_campaign_status.json` and the per-sample logs under
`ResonanceAnalysis/logs/features/`.

## Full direct and cascade analyses

Do not start these until all non-optional rows in `background_manifest.csv`
have compatible feature ROOT/JSON pairs. The two topologies can then run in
parallel:

```bash
nohup python3 -u Code/resonance_xgboost_analysis.py \
  --analysis-root . \
  --topology direct \
  --mode full \
  --output-dir ResonanceAnalysis/results/direct \
  > ResonanceAnalysis/logs/xgboost-direct.log 2>&1 &

nohup python3 -u Code/resonance_xgboost_analysis.py \
  --analysis-root . \
  --topology cascade \
  --mode full \
  --output-dir ResonanceAnalysis/results/cascade \
  > ResonanceAnalysis/logs/xgboost-cascade.log 2>&1 &
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
