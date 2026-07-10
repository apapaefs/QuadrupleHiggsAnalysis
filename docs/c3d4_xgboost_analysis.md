# Resolved-​8b c3/d4 XGBoost analysis

## Scope and versioning

We investigate non-resonant quadruple-Higgs production in the resolved
\(h h h h\to 8b\) final state at a 14 TeV proton--proton collider.  The v2
analysis starts from events containing eight reconstructed candidate jets and
does not introduce a six- or seven-tag category.  The luminosity, branching
fractions, tagging and mistagging probabilities, process-dependent cross
sections and K-factors are external normalization inputs; none of them is used
to determine the relative importance of the c3/d4 signal points during
classifier training.

The historical workflow is retained as `legacy-28-v1`.  It continues to use
the `Data2` tree, `variables[29]`, the existing SM-trained model, the exact
single-bin Poisson \(CL_s\) construction, `--run-c3d4-limit-scan`, untagged
ROOT files and `xgboost_c3d4_scan/`.  The new `extended-91-v2` schema is stored
in a separate `Data3` tree in files named
`*-extended-v2_var.smearCMS.root`; all study products are written below
`xgboost_c3d4_study_v2/`.

## Event population and reconstruction

Both `Data2` and `Data3` are filled for exactly the same event population:
events for which eight analysis candidate jets can be selected.  This is the
feature-tree population and precedes the legacy \(\chi_8\), candidate-mass,
Higgs-\(p_T\) and angular requirements.  The tagged analysis summary therefore
reports, separately,

- `feature_tree_mc_events_out`, `feature_tree_weight_out` and
  `feature_tree_efficiency`;
- the existing preselection quantities; and
- the existing final-cut quantities.

The v2 reconstruction enumerates all 105 perfect matchings of eight jets.
Within each matching, the two constituent indices of every pair are
canonicalized and the four candidate momenta are ordered by decreasing
\(p_T\), with the canonical constituent indices providing the deterministic
tie-break.  The mass targets \((120,115,110,105)\) GeV are assigned only after
this ordering.  We then define

\[
\chi_8=\left[\sum_{a=1}^{4}
  \left(m_{bb,a}-m_a^{\rm target}\right)^2\right]^{1/2}.
\]

The best and second-best distinct matchings are ranked by \(\chi_8\), followed
by the flattened canonical pairing as a tie-break.  Candidate masses,
constituents, \(p_T\), \(\Delta R_{bb}\) and all two- and three-candidate
subsystems therefore refer to the same ordered objects.  The symmetric
common-125-GeV pairing is not used in v2.

The `Data3` non-feature branches are `weight`, `event_index`, `cut_mask` and
`passes_legacy_full_selection`.  The last two describe the retained legacy
selection, while the 91 classifier inputs use the corrected v2 reconstruction.
ROOT metadata records the schema ID, ordered names and units, feature count and
the expected 105 pairings.  A model is scored only if its embedded schema,
profile, count, order, names and contract digest match the input tree.
Metadata-free 28-input models are accepted only through an explicit
`legacy-28-v1` compatibility warning.

## Observable definitions

Jets are ordered by decreasing \(p_T\); Higgs candidates are ordered as
described above.  Pair indices follow the order 12, 13, 14, 23, 24, 34, and
triple indices follow 123, 124, 134, 234.  Angles are wrapped into
\([0,\pi]\).  The full immutable feature contract is:

| Index | Name | Unit | Definition |
|---:|---|---|---|
| 0 | `bjet1_pt` | GeV | Leading selected-jet transverse momentum. |
| 1 | `bjet2_pt` | GeV | Second selected-jet transverse momentum. |
| 2 | `bjet3_pt` | GeV | Third selected-jet transverse momentum. |
| 3 | `bjet4_pt` | GeV | Fourth selected-jet transverse momentum. |
| 4 | `bjet5_pt` | GeV | Fifth selected-jet transverse momentum. |
| 5 | `bjet6_pt` | GeV | Sixth selected-jet transverse momentum. |
| 6 | `bjet7_pt` | GeV | Seventh selected-jet transverse momentum. |
| 7 | `bjet8_pt` | GeV | Eighth selected-jet transverse momentum. |
| 8 | `m8b` | GeV | Invariant mass of the eight selected jets. |
| 9 | `chi8` | GeV | Best coherent staggered-target pairing distance. |
| 10 | `delta_m_h1` | GeV | \(|m_{bb,1}-120\ {\rm GeV}|\), aligned with candidate 1. |
| 11 | `delta_m_h2` | GeV | \(|m_{bb,2}-115\ {\rm GeV}|\), aligned with candidate 2. |
| 12 | `delta_m_h3` | GeV | \(|m_{bb,3}-110\ {\rm GeV}|\), aligned with candidate 3. |
| 13 | `delta_m_h4` | GeV | \(|m_{bb,4}-105\ {\rm GeV}|\), aligned with candidate 4. |
| 14 | `higgs1_pt` | GeV | Transverse momentum of candidate 1. |
| 15 | `higgs2_pt` | GeV | Transverse momentum of candidate 2. |
| 16 | `higgs3_pt` | GeV | Transverse momentum of candidate 3. |
| 17 | `higgs4_pt` | GeV | Transverse momentum of candidate 4. |
| 18 | `dr_hh_12` | dimensionless | \(\Delta R(H_1,H_2)\). |
| 19 | `dr_hh_13` | dimensionless | \(\Delta R(H_1,H_3)\). |
| 20 | `dr_hh_14` | dimensionless | \(\Delta R(H_1,H_4)\). |
| 21 | `dr_hh_23` | dimensionless | \(\Delta R(H_2,H_3)\). |
| 22 | `dr_hh_24` | dimensionless | \(\Delta R(H_2,H_4)\). |
| 23 | `dr_hh_34` | dimensionless | \(\Delta R(H_3,H_4)\). |
| 24 | `dr_bb_h1` | dimensionless | Constituent \(\Delta R_{bb}\) of candidate 1. |
| 25 | `dr_bb_h2` | dimensionless | Constituent \(\Delta R_{bb}\) of candidate 2. |
| 26 | `dr_bb_h3` | dimensionless | Constituent \(\Delta R_{bb}\) of candidate 3. |
| 27 | `dr_bb_h4` | dimensionless | Constituent \(\Delta R_{bb}\) of candidate 4. |
| 28 | `m_bb_h1` | GeV | Invariant mass of candidate 1. |
| 29 | `m_bb_h2` | GeV | Invariant mass of candidate 2. |
| 30 | `m_bb_h3` | GeV | Invariant mass of candidate 3. |
| 31 | `m_bb_h4` | GeV | Invariant mass of candidate 4. |
| 32 | `chi8_second` | GeV | \(\chi_8\) of the deterministic second-best matching. |
| 33 | `delta_chi8` | GeV | `chi8_second - chi8`. |
| 34 | `n_pairings_chi8_lt60` | count | Number of the 105 matchings with \(\chi_8<60\) GeV. |
| 35 | `m_hh_12` | GeV | \(m(H_1,H_2)\). |
| 36 | `m_hh_13` | GeV | \(m(H_1,H_3)\). |
| 37 | `m_hh_14` | GeV | \(m(H_1,H_4)\). |
| 38 | `m_hh_23` | GeV | \(m(H_2,H_3)\). |
| 39 | `m_hh_24` | GeV | \(m(H_2,H_4)\). |
| 40 | `m_hh_34` | GeV | \(m(H_3,H_4)\). |
| 41 | `m_hhh_123` | GeV | \(m(H_1,H_2,H_3)\). |
| 42 | `m_hhh_124` | GeV | \(m(H_1,H_2,H_4)\). |
| 43 | `m_hhh_134` | GeV | \(m(H_1,H_3,H_4)\). |
| 44 | `m_hhh_234` | GeV | \(m(H_2,H_3,H_4)\). |
| 45 | `z_bb_h1` | dimensionless | \(\min(p_{T,b_1},p_{T,b_2})/(p_{T,b_1}+p_{T,b_2})\) for candidate 1. |
| 46 | `z_bb_h2` | dimensionless | Constituent \(p_T\)-sharing fraction for candidate 2. |
| 47 | `z_bb_h3` | dimensionless | Constituent \(p_T\)-sharing fraction for candidate 3. |
| 48 | `z_bb_h4` | dimensionless | Constituent \(p_T\)-sharing fraction for candidate 4. |
| 49 | `pt_4h_over_m_4h` | dimensionless | \(p_T(H_1+H_2+H_3+H_4)/m_{4H}\). |
| 50 | `abs_y_4h` | dimensionless | Absolute rapidity of the reconstructed four-candidate system. |
| 51 | `ht_8b` | GeV | Scalar sum \(\sum_{j=1}^{8}p_{T,j}\). |
| 52 | `mean_m_bb` | GeV | Mean of the four candidate masses. |
| 53 | `std_m_bb` | GeV | Population standard deviation of the four candidate masses. |
| 54 | `max_abs_m_bb_minus_125` | GeV | \(\max_a|m_{bb,a}-125\ {\rm GeV}|\). |
| 55 | `abs_cos_theta_star_h1` | dimensionless | Absolute helicity angle for candidate 1. |
| 56 | `abs_cos_theta_star_h2` | dimensionless | Absolute helicity angle for candidate 2. |
| 57 | `abs_cos_theta_star_h3` | dimensionless | Absolute helicity angle for candidate 3. |
| 58 | `abs_cos_theta_star_h4` | dimensionless | Absolute helicity angle for candidate 4. |
| 59 | `abs_deta_bb_h1` | dimensionless | Constituent \(|\Delta\eta_{bb}|\) of candidate 1. |
| 60 | `abs_deta_bb_h2` | dimensionless | Constituent \(|\Delta\eta_{bb}|\) of candidate 2. |
| 61 | `abs_deta_bb_h3` | dimensionless | Constituent \(|\Delta\eta_{bb}|\) of candidate 3. |
| 62 | `abs_deta_bb_h4` | dimensionless | Constituent \(|\Delta\eta_{bb}|\) of candidate 4. |
| 63 | `abs_dphi_bb_h1` | rad | Constituent wrapped \(|\Delta\phi_{bb}|\) of candidate 1. |
| 64 | `abs_dphi_bb_h2` | rad | Constituent wrapped \(|\Delta\phi_{bb}|\) of candidate 2. |
| 65 | `abs_dphi_bb_h3` | rad | Constituent wrapped \(|\Delta\phi_{bb}|\) of candidate 3. |
| 66 | `abs_dphi_bb_h4` | rad | Constituent wrapped \(|\Delta\phi_{bb}|\) of candidate 4. |
| 67 | `min_dr_bpair_1` | dimensionless | Smallest of all 28 selected-jet pair \(\Delta R\) values. |
| 68 | `min_dr_bpair_2` | dimensionless | Second-smallest all-pair \(\Delta R\). |
| 69 | `min_dr_bpair_3` | dimensionless | Third-smallest all-pair \(\Delta R\). |
| 70 | `min_dr_bpair_4` | dimensionless | Fourth-smallest all-pair \(\Delta R\). |
| 71 | `min_m_bpair_1` | GeV | Smallest of all 28 selected-jet pair masses. |
| 72 | `min_m_bpair_2` | GeV | Second-smallest all-pair mass. |
| 73 | `min_m_bpair_3` | GeV | Third-smallest all-pair mass. |
| 74 | `min_m_bpair_4` | GeV | Fourth-smallest all-pair mass. |
| 75 | `higgs_rapidity_span` | dimensionless | \(\max_a y(H_a)-\min_a y(H_a)\). |
| 76 | `abs_dy_hh_12` | dimensionless | \(|y(H_1)-y(H_2)|\). |
| 77 | `abs_dy_hh_13` | dimensionless | \(|y(H_1)-y(H_3)|\). |
| 78 | `abs_dy_hh_14` | dimensionless | \(|y(H_1)-y(H_4)|\). |
| 79 | `abs_dy_hh_23` | dimensionless | \(|y(H_2)-y(H_3)|\). |
| 80 | `abs_dy_hh_24` | dimensionless | \(|y(H_2)-y(H_4)|\). |
| 81 | `abs_dy_hh_34` | dimensionless | \(|y(H_3)-y(H_4)|\). |
| 82 | `abs_dphi_hh_12` | rad | Wrapped \(|\Delta\phi(H_1,H_2)|\). |
| 83 | `abs_dphi_hh_13` | rad | Wrapped \(|\Delta\phi(H_1,H_3)|\). |
| 84 | `abs_dphi_hh_14` | rad | Wrapped \(|\Delta\phi(H_1,H_4)|\). |
| 85 | `abs_dphi_hh_23` | rad | Wrapped \(|\Delta\phi(H_2,H_3)|\). |
| 86 | `abs_dphi_hh_24` | rad | Wrapped \(|\Delta\phi(H_2,H_4)|\). |
| 87 | `abs_dphi_hh_34` | rad | Wrapped \(|\Delta\phi(H_3,H_4)|\). |
| 88 | `centrality` | dimensionless | \(\sum_jp_{T,j}/\sum_jE_j\) for the eight selected jets. |
| 89 | `transverse_sphericity` | dimensionless | \(2\lambda_2/(\lambda_1+\lambda_2)\) of the \(2\times2\) transverse momentum tensor. |
| 90 | `zness` | GeV | \(\min_{i<j}|m(b_i,b_j)-m_Z|\) over all 28 jet pairs. |

The helicity variable is

\[
|\cos\theta^*_{bb,a}|=
\left|\widehat{\mathbf p}_{b}^{\,(H_a\ \mathrm{rest})}\cdot
\widehat{\mathbf p}_{H_a}^{\,(4H\ \mathrm{rest})}\right|.
\]

An undefined numerical axis is stored as zero and counted in the analysis
summary.  The implementation also counts any non-finite feature that would
have to be sanitized; production acceptance requires this count to vanish.

The named profiles are `corrected28` (indices 0--27), `core52` (indices
0--51) and `full91` (indices 0--90).

## Physical and classifier weights

For source \(p\), an event with signed generator/analyzer weight \(w_{i,p}\)
has physical expected-event weight

\[
w^{\rm phys}_{i,p}=\mathcal L\,\sigma_p\,F_p\,
\frac{w_{i,p}}{W_p^{\rm input}},
\]

where \(W_p^{\rm input}\) is `total_weight_in`, not the number of entries in
the feature tree.  For signal, \(F_p\) contains the signal K-factor,
\({\rm BR}(h\to b\bar b)^4\) and the eight-tag factor.  The background factor
is process dependent and contains its K-factor, any decay branching fraction
not already generated, and the appropriate b-tag, c-mistag and light-mistag
probabilities.  Signed physical weights are retained for all efficiencies,
yields, \(\sum w^2\) errors and limits.

XGBoost requires non-negative weights.  The pooled signal training weight is

\[
w^{\rm train}_{i,p}=\frac{|w_{i,p}|}{\sum_{j\in p}|w_{j,p}|}\frac{1}{57}.
\]

Thus, every c3/d4 point has equal total importance.  Background training uses
\(|w^{\rm phys}|\), which preserves the physical process mixture, and the two
classifier classes are finally normalized to equal totals.  Their common
absolute scale is chosen so that the combined classifier weight equals the
number of nonzero-weight signal rows plus original (pre-replication)
background rows in the training partition.  The mean effective-row weight is
therefore one, making XGBoost's `min_child_weight` and regularization ranges
meaningful.  Parameterized background replicas divide an original event's
weight without increasing this normalization count.  Physical signal cross
sections are used only when producing yields, limits and exclusion ratios.

Before fitting, the implementation rejects a configuration if

\[
\frac14\sum_iw_i^{\rm train}<2\,{\tt min\_child\_weight},
\]

because even the initial root node then lacks enough Hessian weight to form
two valid children.  Final and fixed-profile models must contain at least one
split and nonconstant training scores; zero-split Optuna trials are pruned.

## Cross-fitting, profile selection and tuning

Within every source, valid original event indices are sorted, shuffled with a
SHA-256-derived seed based on the canonical source ID and 12345, and assigned
round-robin to five folds.  For rotation \(f\), fold \(f\) is the test set,
fold \((f+1)\bmod5\) is the validation set, and the remaining three folds are
used for training.  Every event is therefore scored once by a model that saw
it in neither training nor validation.  Test-fold weights are never rescaled;
the union of the five test partitions is already the full physical yield.
Validation weights are multiplied by the predeclared factor of five, since
each validation fold is a deterministic one-in-five subsample.  This estimate
uses no test-fold event or weight while choosing thresholds or hyperparameters.

The fixed current XGBoost configuration is first run for `corrected28`,
`core52` and `full91` on identical folds.  One global profile is selected from
validation data; if its limit objective is within 1% of the optimum, the
smaller profile is preferred.  All three test results are retained as
ablations.

The chosen profile is tuned independently in each rotation with a seeded,
sequential Optuna TPE study.  The SQLite studies resume to a total of 40
trials, and the fixed current configuration is enqueued first.  The search is

- 200--800 trees in steps of 100;
- depth 2--6;
- learning rate 0.01--0.15 (logarithmic);
- minimum child weight 1--50 (logarithmic);
- row and column subsampling 0.6--1;
- \(\gamma\in\{0,0.01,0.1,1\}\);
- \(\alpha\in\{0,10^{-3},10^{-2},0.1,1\}\); and
- \(\lambda\in[0.1,30]\) (logarithmic).

The minimized objective gives equal importance to all points,

\[
J=0.75\,\operatorname{median}_p(\ln\sigma_{95,p})+
0.25\,Q_{90,p}(\ln\sigma_{95,p}).
\]

`sm-crossfit-v2` uses only the dedicated SM signal for classifier training;
`pooled-crossfit-v2` uses the 57 grid samples and excludes the separate SM
file.

## Exact cut-and-count limit

For every validation rotation and c3/d4 point, 1001 thresholds in \([0,1]\)
are scanned.  A threshold must retain at least 25 raw and 10 effective
background entries.  We minimize

\[
\sigma_{95}(t)=\frac{S_{95}^{\rm exact}[B(t)]}
{\mathcal L F_s\epsilon_s(t)}.
\]

The numerator is the exact one-bin Poisson \(CL_s\) upper limit for the median
integer background-only observation.  Among thresholds within 1% of the
minimum, the largest effective background count is preferred, followed by the
lower threshold.  Thresholds are frozen before the test fold is read.  The
five selected test partitions are then added without rescaling and the exact
limit is recomputed from their total signed signal and background yields.

At \(B=2.9206089\), this construction gives
\(S_{95}=5.4354378\) events.  The pyhf asymptotic one-bin control gives a
different value (approximately 4.735 events without nuisances); this backend
difference is reported explicitly and is not called an ML improvement.

## Score-shape likelihood

For each held-out channel, candidate score edges start from validation
background quantiles \([0,0.50,0.75,0.90,0.97,1]\).  All subsets giving two
to five bins are considered, with positive signed background, at least 25 raw
entries and at least 10 effective entries required in every validation bin.
The validation pyhf limit including the signal and background MC-statistical
`staterror` terms selects the shape; the explicit raw/effective-entry
requirements provide an additional guard against poorly populated bins.
Candidates within 1% prefer fewer bins.

The fold-specific numerical edges are frozen and applied to the corresponding
test fold.  If a signed test signal or background bin is non-positive, the
next coarser validation-valid candidate is used.  Negative yields are never clipped.  Each
signal template is normalized to a production cross section of 1 fb, so the
shared pyhf POI is directly \(\sigma\) in fb.

The output contains:

- the exact cut-and-count result;
- a nuisance-free pyhf one-bin control;
- a nuisance-free pyhf score-shape result;
- the primary score-shape result with independent signal and background
  `staterror` modifiers derived from \(\sqrt{\sum_iw_i^2}\); and
- independent reruns with the total background scaled by 0.25 and 4.

The background envelope is not a fitted nuisance.  Process cross-section
uncertainties from `Backgrounds/processes.csv` are retained in the manifest as
diagnostics but are not profiled in the v2 likelihood.

## Parameterized-classifier gate

A parameterized \(f(x,c_3,d_4)\) classifier is considered only if the pooled
validation limits, relative to the tuned extended-SM classifier, satisfy all
of the following: median ratio at most 0.90, 90th-percentile ratio at most
1.10, SM-point ratio at most 1.05, and a better limit objective in at least
four of five rotations.  The gate and every component metric are written to
`parameterized_classifier_gate.json`.  If it fails, parameterized training is
not performed.

If the gate passes, the runner appends \(c_3/30\) and \(d_4/700\) at ML time
and performs 30 sequential Optuna trials per fold.  Signal receives its true
coordinates; every background event is
replicated at three deterministic, distinct grid coordinates within its
original fold, with its classifier weight divided by three.  Background test
events are rescored at every evaluated point.  Parameterized models are
materialized only after the data-driven gate passes; a failed gate stops the
study after the pooled result.

## Commands and outputs

The retained legacy command is, for example,

```bash
python 4h_analyzer.py --run-c3d4-limit-scan
```

The full v2 study is run in the ROOT/Python environment containing ROOT,
XGBoost, pyhf 0.7.6 and Optuna 4.9.0:

```bash
source ~/root310install/bin/thisroot.sh
source ~/xgb-py310/bin/activate
export PATH="$HOME/Projects/Herwig/HerwigPol2/bin:$PATH"
python 4h_analyzer.py \
  --run-c3d4-xgboost-study \
  --observable-set extended-91-v2 \
  --training-strategy pooled-crossfit-v2 \
  --cv-folds 5 \
  --optuna-trials 40 \
  --analysis-jobs 8 \
  --study-outdir xgboost_c3d4_study_v2
```

Passing `--feature-profile corrected28|core52|full91` forces one profile;
omitting it performs the validation-only global comparison.  Models, fold
thresholds, bin edges, efficiencies, yields, raw and effective MC statistics,
\(S_{95}\), cut and shape \(\sigma_{95}\), baseline ratios, maps, contours,
Optuna histories and the gate decision are written only under the study
directory.  `method_manifest.json` records package versions, source commit,
input hashes, normalization inputs, fold populations, schema, feature names,
strategy, parameters and seeds.

## Limitations and deferred studies

The result is conditional on the resolved eight-candidate definition and on
the adopted detector/tagging approximation.  It does not profile scale, PDF,
shower, hadronization, MPI, tune, tagging or background cross-section
systematics.  Common-125-GeV re-pairing, topness, extra-jet activity, tagging
discriminants, boosted substructure and colour-flow observables are deferred
to dedicated systematic studies.  An improvement is measured against the
frozen baselines; it is not assumed by construction.
