# SM \(hhhh\to 8b\) vs. \(gg\to 8b\): ROOT → XGBoost → pyhf

This executable tutorial follows two Monte Carlo samples from schema-checked
ROOT trees, through a deterministic cross-fitted classifier, to a binned
HistFactory likelihood. Its central point is simple:

> XGBoost is trained first. **pyhf neither fits nor retrains the classifier.**
> The frozen out-of-fold score is histogrammed, and pyhf fits the event counts
> in those bins while profiling nuisance parameters.

The numerical result is deliberately a teaching example. It contains only the
SM \(hhhh\to 8b\) signal and pure \(gg\to8b\) background, background-only
Asimov observations, finite-MC uncertainties, and an explicitly illustrative
10% background normalization nuisance. It is not a publication-ready
exclusion, even though the figures use publication-quality construction.

## Contents

- `sm_gg8b_xgboost_pyhf.ipynb` — the annotated, executable lesson;
- `BEGINNER_GUIDE.md` — a standalone explanation of weights, folds, score
  templates, Asimov data, profiling, \(CL_s\), and the relation to `fast-sm`;
- `tutorial_helpers.py` — tested loading, cross-fitting, likelihood and plotting
  helpers;
- `config.json` — all sample, physics, random-seed, hash and binning inputs;
- `requirements.txt` — the Python layer to install on top of an existing
  ROOT/PyROOT environment;
- `tutorial_outputs/` — generated models, tables, JSON and figures (ignored by
  Git).

Run commands below from the `QuadrupleHiggsAnalysis-c3d4-v2` repository root.

## 1. Obtain and verify the inputs

The tutorial uses the internally consistent `extended-91-v2` files from the
current Tiresias analysis tree. It does not use the stale `AlpGen/.../4H`
snapshot or any historical XGBoost artifact.

```bash
mkdir -p Signals/events Backgrounds/events

scp tiresias:/home/apapaefs/Projects/QuadrupleHiggsAnalysis/Signals/events/HW-gg_hhhh_SM-extended-v2-uniform-smear-v1_var.smearCMS.root Signals/events/
scp tiresias:/home/apapaefs/Projects/QuadrupleHiggsAnalysis/Signals/events/HW-gg_hhhh_SM-extended-v2-uniform-smear-v1.analysis_summary.json Signals/events/
scp tiresias:/home/apapaefs/Projects/QuadrupleHiggsAnalysis/Backgrounds/events/HW-gg_to_8b-extended-v2-uniform-smear-v1_var.smearCMS.root Backgrounds/events/
scp tiresias:/home/apapaefs/Projects/QuadrupleHiggsAnalysis/Backgrounds/events/HW-gg_to_8b-extended-v2-uniform-smear-v1.analysis_summary.json Backgrounds/events/

shasum -a 256 \
  Signals/events/HW-gg_hhhh_SM-extended-v2-uniform-smear-v1_var.smearCMS.root \
  Signals/events/HW-gg_hhhh_SM-extended-v2-uniform-smear-v1.analysis_summary.json \
  Backgrounds/events/HW-gg_to_8b-extended-v2-uniform-smear-v1_var.smearCMS.root \
  Backgrounds/events/HW-gg_to_8b-extended-v2-uniform-smear-v1.analysis_summary.json
```

The expected hashes are:

| input | bytes | SHA-256 |
| --- | ---: | --- |
| signal ROOT | 912792 | `70688a574bb175e7a4a319209aa13b0335536417ebd8aa53a6ecc60b5fd9c6e1` |
| signal sidecar | 720 | `948f5048afaa2a3d6595b63203c54fb4a84eae0f190acc9490918eebae1f5247` |
| background ROOT | 8213249 | `b5514870553f792045465d3f893efe48ac9c1752d4f5de5000c416149766bef1` |
| background sidecar | 753 | `6e60f04e87e9566e2675019e44e76e92906c0b4c4f5e6c9fe4bfb99dbaed7acd` |

The loader checks these hashes, the `Data3` metadata, the
`extended-91-v2`/`corrected28` feature contract, and the matching sidecars
before training. ROOT products and generated tutorial outputs are ignored by
Git.

## 2. Prepare the environment

Activate the ROOT environment already used for this repository, then create or
activate a Python environment in which PyROOT remains importable:

```bash
source ~/root310install/bin/thisroot.sh
python3 -m venv --system-site-packages .venv-sm-gg8b-tutorial
source .venv-sm-gg8b-tutorial/bin/activate
python -m pip install -r tutorials/sm_gg8b_xgboost_pyhf/requirements.txt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

`pyhf==0.7.6` is pinned because the notebook teaches that version's public
[`fit`](https://pyhf.readthedocs.io/en/v0.7.6/_generated/pyhf.infer.mle.fit.html),
[`hypotest`](https://pyhf.readthedocs.io/en/v0.7.6/_generated/pyhf.infer.hypotest.html),
and
[`upper_limit`](https://pyhf.readthedocs.io/en/v0.7.6/_generated/pyhf.infer.intervals.upper_limits.upper_limit.html)
interfaces. ROOT itself stays external because its installation is
platform-specific.

## 3. Execute the notebook

Interactively:

```bash
jupyter lab tutorials/sm_gg8b_xgboost_pyhf/sm_gg8b_xgboost_pyhf.ipynb
```

Reproducibly and non-interactively:

```bash
jupyter nbconvert \
  --to notebook \
  --execute tutorials/sm_gg8b_xgboost_pyhf/sm_gg8b_xgboost_pyhf.ipynb \
  --output sm_gg8b_xgboost_pyhf.executed.ipynb \
  --ExecutePreprocessor.timeout=1800
```

The run uses five deterministic rotations. In rotation \(r\), fold \(r\) is
used exactly once for inference, fold \((r+1)\bmod 5\) selects the binning, and
the remaining three folds train XGBoost. Training uses class-balanced
\(\lvert w\rvert\); all yields retain their signed physical weights.

For a ROOT event with stored weight \(w_i\),

\[
 w_i^{\mathrm{phys}}
 = \mathcal L\,\sigma\,F\,\frac{w_i}{W_{\mathrm{input}}},
\]

with

\[
 F_s=K_s\,\mathrm{BR}(h\to b\bar b)^4\epsilon_b^8,\qquad
 F_b=K_b\,\epsilon_b^8,
\]

where the configured phenomenological factors are \(K_s=K_b=2\). The signal
POI is the production cross section *before* applying this fixed \(K_s\);
the \(K\)-factor is part of the declared yield convention, not a fitted
nuisance.

The signal histogram is separately normalized to a production cross section
of 1 fb. Consequently the shared pyhf `normfactor`,
`sigma_hhhh_fb`, is a cross section in fb; the notebook also reports
\(\mu=\sigma/\sigma_{\mathrm{SM}}\).

## 4. What pyhf fits

After bin edges are frozen on validation events, each held-out fold becomes a
channel. In score bin \(i\), the templates store

\[
s_i=\sum_{\mathrm{signal}} w_i^{(1\,\mathrm{fb})},\quad
b_i=\sum_{\mathrm{background}} w_i^{\mathrm{phys}},\quad
\delta s_i=\sqrt{\sum w_i^2},\quad
\delta b_i=\sqrt{\sum w_i^2}.
\]

The likelihood has the schematic form

\[
\mathcal L(\sigma,\boldsymbol\theta)=
\prod_i\operatorname{Pois}\!\left(
n_i\mid \sigma s_i(\boldsymbol\theta)+b_i(\boldsymbol\theta)
\right)\prod_k\pi_k(\theta_k).
\]

The workspace contains:

- a shared signal `normfactor` POI named `sigma_hhhh_fb`;
- independent signal and background `staterror` terms for every channel;
- a shared pedagogical `gg8b_norm` `normsys` with `lo=0.90`, `hi=1.10`;
- background-only Asimov observations for the headline expected limit.

The generation integration error of 2.54% is recorded in provenance but is not
promoted to a QCD theory nuisance.

An Asimov dataset is the deterministic model expectation at specified
parameter values, without a random Poisson fluctuation. It may therefore have
fractional bin counts. The “observed-on-Asimov” limit written by pyhf is a
calculation on that synthetic expectation and is not an observed-data result.

The notebook also constructs a deterministic injected-Asimov diagnostic with
\(\sigma_{\mathrm{inj}}=0.5\,\sigma_{95}^{\mathrm{expected}}\) and a
\(+0.5\sigma\) background-normalization displacement. This sample exists only
to make fit evolution and pulls visible. It is labelled **NOT DATA** in text,
JSON and figures and is never used as the headline result.

## 5. Auditable outputs

`tutorial_outputs/` contains:

- `event_scores.csv` with event identifiers, folds, signed weights and one
  out-of-fold score per event;
- `fold_models/fold_*.json`;
- `binning.json`, the primary `workspace.json`, inclusive
  `workspace_one_bin.json`, MC-stat-only `workspace_mcstat_only.json`, and
  `fit_results.json`;
- `versions.json`, seeds and `input_hashes.json`;
- PDF and 300-dpi PNG pairs for the ROC, feature importance, normalized and
  expected-yield scores, unrolled prefit/postfit fit, likelihood scan,
  expected \(CL_s\) bands, nuisance pulls and reduced correlation matrix.

No invalid signed yield is clipped. If a frozen test-fold bin has nonpositive
background, the code walks the validation-defined coarsening hierarchy; if
that hierarchy fails, the run stops.

The validation-fold weights are multiplied by five when ranking candidate
binnings so their yield represents the full sample. This leaves the relative
MC uncertainty at its one-fold value; it is conservative by \(\sqrt5\)
relative to five independent folds at the same total yield and can favor
coarser binning. Also, while an event never selects the edges for its own
channel, each validation fold later appears as a different test channel in
the combined likelihood. Exact independence would require an extra selection
sample or nested cross-validation; the notebook states this mild
selection-induced-optimism caveat explicitly.

The result table compares the primary five-channel model with the same
finite-MC workspace after removing the illustrative `gg8b_norm` term, as well
as with the inclusive one-bin control. The headline remains the five-channel
model containing both MC-statistical terms and the declared 10% pedagogical
normalization uncertainty.

For pyhf 0.7.6, the no-scan `upper_limit` convenience path does not forward a
non-default `level` to its internal root finder. The precise tutorial result
therefore calls `toms748_scan` with
`level = 1 - confidence_level`; the notebook also demonstrates
`upper_limit` with an explicit scan. A unit test verifies that 90% and 95%
limits differ.

## Relation to production `fast-sm`

This is a faithful miniature of the `fast-sm` data flow: complete-event
five-fold SM cross-fitting, frozen scores, signed yields, validation-defined
score regions, and a pyhf shape likelihood. It is not numerically identical
to production. In particular, this tutorial uses the 28-variable
`corrected28` projection and only two processes, whereas `fast-sm` uses
`full91` for `extended-91-v2` inputs and the configured production background
composition. The tutorial’s injected Asimov sample and 10% `gg8b_norm`
modifier are teaching devices.

Read `BEGINNER_GUIDE.md` before the notebook for a slower explanation of every
stage and the statistical interpretation.

## Tests

```bash
python -m pytest -q \
  tests/test_sm_gg8b_pyhf_tutorial.py \
  tests/test_c3d4_xgboost_study.py
```

The tests cover normalization, fold separation, validation-only binning,
template closure, workspace structure, Asimov fits, the repository's
approximately 4.735 reference limit, an informative two-bin toy, and the
required artifact inventory.
