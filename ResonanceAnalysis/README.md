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

## Current state (2026-07-30)

- All **42 direct** and **441 cascade** raw Herwig signal samples are ready.
  Their complete earlier fixed-mass feature campaign remains under
  `ResonanceAnalysis/features/` as a non-overwritten legacy data set and is
  rejected by the current feature validator.
- All 14 raw background inputs are branch-complete. Resonance feature pairs
  already exist for `HW-gg_hhhh_SM`, `HW-gg_to_6b_2j`, and
  `HW-gg_to_4b_4j` under the legacy preprocessing contract; the two newly
  added SM multihiggs inputs and the other nine QCD/reducible backgrounds
  still need extraction with the current versioned implementation.
- `background_manifest_smoke.csv` selects these five backgrounds for technical
  smoke tests; the preparation helper creates compatible versioned feature
  pairs as needed. A smoke result is deliberately marked non-physical.
- `background_manifest.csv` is the 11-component QCD/reducible background model
  plus SM \(hhhh\), \(hhh+b\bar b\), and \(hh+4b\). A full result requires
  exactly one available row for each of the three SM multihiggs roles.

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

The \(125\) GeV value is a common reconstruction target for **every** signal
and background event in the resonant workflow; it is not a special mass
assigned to the resonant backgrounds. It represents the nominal on-shell
Higgs mass used in the simulation and makes the resolved/merged assignment
symmetric under interchange of the four Higgs candidates. This is particularly
useful when a candidate can be either one double-B jet or a resolved jet pair
and there is no natural candidate-\(p_T\) rank to which a different mass target
should be attached.

The resolved non-resonant self-coupling workflow instead orders four resolved
candidates by decreasing \(p_T\) and uses the frozen staggered pairing targets
\((120,115,110,105)\) GeV. Those numbers can be interpreted as heuristic
reconstruction anchors for the downward and rank-dependent response after
detector smearing, not four different physical Higgs masses; the repository
does not contain a calibration record establishing that the exact values are
optimal. The two choices are nevertheless internally consistent because each
workflow applies its own pairing prescription identically to all of its
signals and backgrounds, but their reconstruction efficiencies should not be
compared as though the object definitions were identical.

The common-\(125\)-GeV choice is physics-motivated rather than an empirical
claim that it is optimal after detector response. A dedicated reconstruction
systematic should repeat the resonant extraction with response-calibrated
common targets, and with a staggered prescription in the fully resolved
category, before interpreting small sensitivity differences between the two
workflows.

The reproducible implementation of that check is documented in
[the mass-target study](../MassTargetStudy/README.md). It applies the common
and staggered prescriptions, plus nearby variations, to both workflows and
selects a tuple on a tuning split before reporting its performance on
held-out events.

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
selected by `--smoke-points 3`, followed by the five smoke backgrounds.
These products use the versioned, non-overwriting production directory and
are retained when the full extraction is resumed. The second command also
writes the resolved smoke-background manifest:

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
  --workers 5
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

## SM `hhh+bb` and `hh+4b` backgrounds

The two components are now non-optional entries in both background manifests:

| Role | Production sample | Cross section before Higgs decays | Unique events | `hbb_power` |
| --- | --- | ---: | ---: | ---: |
| `sm_hhhbb` | full-loop \(gg\to hhhg\), followed by forced \(g\to b\bar b\) | \(4.31371657238\times10^{-4}\) fb | 10,000 | 3 |
| `sm_hh4b` | direct HEFT \(gg\to hh+b\bar b b\bar b\) | \(9.62241\times10^{-3}\) fb | 9,515 | 2 |

The manifest K-factor is applied separately. Only the three or two physical
Higgs decays receive \({\rm BR}(h\to b\bar b)\); the associated bottom quarks
do not. Both samples contain eight true bottom-quark candidates, so their
charm- and light-mistag multiplicities are zero. Their existing HwSim ROOT
files contain `bHadronMultiplicity`, and `regenerate=false` prevents the
background-regeneration helper from producing duplicate shower histories.

On Tiresias, the `hhh+bb` row uses the consolidated campaign symlink
`HerwigForcedSplitting/gg_hhhg_c3d4_10k_hhhbb_parallel2_w1`. The `hh+4b`
row uses `HerwigSignalPoints/sm_hh4b_heft/` and the normalized LHE under
`Signals/sm_hh4b_heft/lhe/`. Produce just the two new resonance feature pairs
with:

```bash
python3 Code/prepare_resonance_features.py \
  --analysis-root . \
  --kind backgrounds \
  --only HW-gg_hhhbb_SM \
  --only HW-gg_hh4b_SM_HEFT \
  --workers 2
```

These are irreducible SM backgrounds to a heavy-resonance search, even though
the non-resonant self-coupling study treats the corresponding coupling-dependent
channels as signal components. The present samples also carry important
modelling qualifications: forced \(g\to b\bar b\) is not the complete direct
\(gg\to hhhb\bar b\) matrix element and is not matched against inclusive
heavy-flavour shower production, while the \(hh+4b\) sample uses HEFT rather
than the finite-top-mass loop amplitude. The common 10% background nuisance
does not by itself establish that these approximations are covered. Results
including them should therefore be described as conditional on these nominal
models, with dedicated normalization and shape variations added before a
precision interpretation.

After feature extraction, run the direct and cascade analyses into **new**
result directories. The changed manifest and feature-input fingerprints
deliberately prevent stale models or limits from being reused.
