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
`*-extended-v2-uniform-smear-v1_var.smearCMS.root`; all study products are
written below `xgboost_c3d4_study_v2_uniform-smear-v1/`.  The former
`extended-v2` tag is rejected, so files produced with different four-vector
smearing prescriptions cannot be mixed or overwritten.

## Jet-energy smearing

The detector response follows the CMS-like energy resolution used in
[our triple-Higgs analysis](https://arxiv.org/abs/2509.16364).  For a jet with
energy \(E\), one Gaussian fluctuation is drawn,

\[
E'=\max(E_{\min},E+\Delta E),\qquad
\Delta E\sim {\cal N}(0,\sigma_E),\qquad E_{\min}=10^{-6}\ {\rm GeV},
\]

with

\[
\sigma_E =
\begin{cases}
\sqrt{(0.05E)^2+(1.5)^2E}, & |\eta|\leq 3,\\
\sqrt{(0.13E)^2+(2.7)^2E}, & 3<|\eta|\leq 5.
\end{cases}
\]

All energies in this expression are in GeV.  The analysis requires
\(|\eta|<2.5\), so accepted jets always use the first branch.  The cited
paper keeps the jet angles fixed and maps the fluctuated energy to the
massless four-vector \(p'_T=E'/\cosh\eta\).  This remains the exact untagged
`legacy-28-v1` behaviour: the raw \(p_T>20\) GeV and pseudorapidity cuts are
applied before smearing.

For `extended-91-v2`, the same energy fluctuation is instead propagated to a
possibly massive reconstructed jet with one correlated scale factor,

\[
s=E'/E,\qquad p'^{\mu}=s\,p^{\mu},\qquad m'=s\,m.
\]

This massive-jet extension is specific to the versioned v2 workflow; it is
not the massless mapping quoted from the cited paper.  It leaves \(\eta\) and
\(\phi\) unchanged, scales all four components consistently and does not
hold the original jet mass fixed.  Finite raw \(\eta\) and \(p_T\) are
required, then the \(|\eta|<2.5\) requirement is applied, the jet is smeared
exactly once, and the reconstructed requirement \(p'_T>20\) GeV is imposed.
Thus the v2 population includes upward threshold migrations and excludes downward
migrations.  Both true-\(b\) and non-\(b\) jets are audited before the
event-level true-\(b\) multiplicity requirement; upward migrations are
reported in the disjoint raw-\(p_T\) intervals 10--12, 12--15 and 15--20 GeV.

The pseudo-random seed is fixed to `14101983`.  ROOT and JSON metadata record
the smearing model, acceptance order, four-vector scaling, seed, energy floor,
number of Gaussian draws and the maximum numerical residual in
\(|m'-sm|\).  Existing tagged output is reused only when these values, the
migration-bin closure and the reported efficiency ratios all satisfy the v2
contract.

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

The nominal `extended-91-v2` output remains fixed to the staggered tuple.
For calibration only, the extractor also accepts `--higgs-mass-targets`
together with an `extended-v2-uniform-smear-v1-target-study-*` tag; this
writes isolated feature trees and records the four targets in ROOT and JSON
metadata without changing the legacy `Data2` reconstruction. The compact
held-out comparison with the resonant prescription is documented in
[the mass-target study](../MassTargetStudy/README.md).

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
w^{\rm train}_{i,p}=\frac{|w_{i,p}|}{\sum_{j\in p}|w_{j,p}|}\frac{1}{N_{\rm points}}.
\]

Thus, every one of the (N_{\rm points}) sampled c3/d4 points has equal total
importance.  Background training uses
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
fold \((f+1)\bmod5\) is the inner validation set, and the remaining three folds
are used for training.  Fold \(f\) is the outer test set for that rotation.
Every event is therefore scored once by a model that saw it in neither
training nor validation, and the selection applied to the event uses only the
paired inner validation fold.  Test-fold weights are never rescaled; the union
of the five outer test partitions is already the full physical yield.
Validation weights are multiplied by the predeclared factor of five when a
single inner fold is used to project rates to the complete sample.  This
estimate uses no event or weight from its paired outer test fold.

An event may occur in the training or inner validation partition of a
different rotation.  This is the standard nested-cross-fit reuse: the five
development procedures are correlated, but every event's own outer-test score
and selection are constructed without that event.  Only the five disjoint
outer test partitions enter the final templates.

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
`pooled-crossfit-v2` uses all unique grid samples and excludes the separate SM
file.  The original production grid contains 57 points, while denser campaigns
may contain any larger number of unique coordinates.

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

For each outer test channel, candidate score edges start from the paired inner
validation fold's background quantiles
\([0,0.50,0.75,0.90,0.97,1]\).  All subsets giving two to five bins are
considered independently in each rotation, with positive signed background,
at least 25 raw entries and at least 10 effective entries required in every
validation bin.  The single-channel validation weights are multiplied by five
to project their yields to the complete sample; their `staterror` terms are
scaled consistently and therefore retain the relative MC precision of the
one-fold validation estimate.  The validation pyhf limit including the signal
and background MC-statistical terms selects the shape.  Candidates within 1%
prefer fewer bins.

There is no common bin-count or edge-structure decision across validation
folds.  The fold-specific numerical edges and bin count are frozen and applied
only to the corresponding outer test fold.  If a signed test signal or
background bin is non-positive, the next coarser candidate from that fold's
validation-defined hierarchy is used.  Negative yields are never clipped.
Thus an event never influences the binning used for its own channel, although
the development samples of different rotations overlap.  Each signal template
is normalized to a production cross section of 1 fb, so the shared pyhf POI is
directly \(\sigma\) in fb.

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

### Parallel evaluation, checkpoints and progress

After the five classifiers, held-out scores and validation-selected candidate
binnings have been fixed, the pyhf calculation for one coupling point is
independent of every other point.  The runner can therefore evaluate points in
separate POSIX processes with `--shape-jobs N`.  This changes only the order and
wall-clock time of the fits: the score arrays, signed event weights, bin edges,
workspace construction and statistical prescription are identical to the
serial calculation.  The default remains one worker.

Each completed point is written atomically below

```text
<study-outdir>/<strategy>/shape_checkpoints/<fingerprint>/point-<id>.json
```

The fingerprint binds the source and method versions, schema and profile,
strategy, fold assignment, normalization and input hashes, pyhf version and
the five saved model hashes.  A repeated command reuses only complete matching
checkpoints.  Missing, malformed, incompatible, interrupted, worker-error or
numerically failed pyhf points are evaluated again.  Invalid validation
binning, invalid signed signal templates and non-positive signed test bins are
retained as terminal analysis outcomes rather than repeatedly retried.  The
normal `shape_results.csv` and `shape_results.json` files are written only after
every point has reached a terminal state.  Earlier canonical tables and maps
are first moved below `<strategy>/previous_outputs/`, so an interrupted rerun
cannot look like a newly completed result.  An interruption leaves restartable
checkpoints; `shape_results.partial.*` is also written when all submitted tasks
return but retryable failures remain.  The publication state is recorded in
`<strategy>/shape_results_status.json`.

A command-line or preprocessing failure that occurs before a new attempt owns
the manifest never relabels an earlier completed campaign.  Such failures are
recorded separately below `<study-outdir>/failed_attempts/`.

Progress is printed with immediate flushing throughout input discovery and
tagged ROOT regeneration, input loading, hashing, profile comparison, Optuna
tuning, fold training/scoring, aggregation, shape evaluation and plot
production.  The latest state is also atomically recorded in
`<study-outdir>/study_progress.json`.  During the shape stage it contains the
completed, resumed, active and queued point counts, the latest point and fit
status, elapsed time and an estimated remaining time.  For example, from a
second shell one can monitor it with

```bash
watch -n 10 'jq . xgboost_c3d4_study_v2_uniform-smear-v1/study_progress.json'
```

Checkpointing resumes the pointwise shape calculation once a repeated full
command reaches that stage.  It does not currently skip the preceding profile
comparison and model fitting, although completed Optuna studies continue to
resume from their persistent SQLite databases.

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

### Smoke, preview, fast-SM and full modes

The v2 driver provides four explicitly labelled execution levels:

| Mode | Events | Classifier setup | Statistical output | Intended use |
|---|---|---|---|---|
| `smoke` | At most 2000 feature-tree entries per source by default | `corrected28`, fixed parameters and the SM cross-fit by default | exact single-bin cut limit only | code and file-flow checks; **not a physics result** |
| `preview` | all events and all supplied unique coupling points | `core52`, fixed parameters and pooled plus SM cross-fits by default | exact single-bin cut limit only | rapid, physically normalized but preliminary exclusion map |
| `fast-sm` | all events and all supplied unique coupling points | `full91`, fixed parameters and the SM cross-fit only | exact cut and pyhf score-shape limits | fast, physically valid SM-trained result without Optuna or pooled training |
| `full` | all events and all supplied unique coupling points | validation profile comparison, Optuna tuning and the parameterized gate | exact cut and pyhf score-shape limits | complete optimized workflow |

Preview, fast-SM and full modes reject `--max-events`.  The v2 workflow also rejects
`--analysis-max-events` in every mode: truncating the C++ analysis could place
a partial sample under the shared `*-extended-v2-uniform-smear-v1_var.smearCMS.root` filename
and silently contaminate a later production run.  Smoke mode truncates only
the Python read of an existing feature tree.  Its tables carry
`physics_result_valid=false` and its plots are watermarked
`NON-PHYSICS SMOKE TEST`.

Before any mode reuses a tagged file, the driver checks the immutable Data3
schema, the analysis-summary metadata, the requested background mistag
composition, and that `mc_events_in` equals the number of entries in the raw
ROOT `Data` tree (or an independently recorded generated-event count when the
raw file is unavailable).  A failed check forces regeneration from the raw
ROOT file.  With `--no-run-missing-analysis`, the command stops rather than
silently using the invalid file.  The completion evidence is stored for every
sample in `method_manifest.json`; `uses_complete_event_samples=true` is only
reported when the source check succeeds and no Python event cap is active.

A smoke test can be launched with

```bash
python 4h_analyzer.py \
  --run-c3d4-xgboost-study \
  --study-mode smoke
```

The default output is `xgboost_c3d4_study_v2_uniform-smear-v1_smoke/`.  Use
`--smoke-max-events N` or `--max-events N` to change the Python-side cap.

A physics-faithful quick preview is

```bash
python 4h_analyzer.py \
  --run-c3d4-xgboost-study \
  --study-mode preview \
  --feature-profile core52 \
  --training-strategy pooled-crossfit-v2
```

The default output is `xgboost_c3d4_study_v2_uniform-smear-v1_preview/`.  No Optuna or pyhf
shape calculation is run.  The preliminary exclusion map is written to

```text
<study-outdir>/<strategy>/cut_preview/maps/
  <strategy>_preview_cut_exclusion_contour.pdf
```

The same cut-preview map is published during a full run immediately after a
strategy completes its five cross-fit rotations and before that strategy
enters the expensive pyhf shape stage.  Its status is recorded atomically in
`<strategy>/cut_preview/status.json`, so the map can be inspected while the
full job continues.  Preview products are watermarked and are not the results
to quote in the paper.

Additional unique coupling points are supplied by repeating
`--c3d4-signal-dir`; they must not be merged with the original points:

```bash
python 4h_analyzer.py \
  --run-c3d4-xgboost-study \
  --study-mode preview \
  --training-strategy sm-crossfit-v2 \
  --c3d4-signal-dir HerwigSignalPoints/c3d4_10k/events \
  --c3d4-signal-dir HerwigSignalPoints/c3d4_additional/events \
  --study-outdir xgboost_c3d4_study_v2_uniform-smear-v1_dense_preview
```

At least three unique points are required.  Duplicate coordinates are
rejected, and the manifest records the complete dynamic point count.  For
pooled training each of the (N_{\rm points}) coordinates retains equal total
classifier weight.

The fixed-parameter SM-only shape study is

```bash
python 4h_analyzer.py \
  --run-c3d4-xgboost-study \
  --study-mode fast-sm \
  --optuna-trials 0 \
  --shape-jobs 8 \
  --c3d4-contour-interpolation clough-tocher
```

It uses the complete event samples, `full91`, five-fold SM training, the exact
cut limit and the same pyhf score-shape likelihood as full mode.  It skips the
feature-profile comparison, every Optuna study, pooled training and the
parameterized-classifier gate.  Its default output is
`xgboost_c3d4_study_v2_uniform-smear-v1_fast-sm/`.

No previous Optuna history is reused in this command.  In particular, tuning
results from a legacy or pre-uniform-smearing study must not be imported into
the new detector-response contract.

### Legacy-style exclusion contours

Each preview and final strategy now also writes the three paper-style contour
variants used by `xgboost_c3d4_scan/`, separately for the exact single-bin cut
limit and, when available, the pyhf score-shape limit:

```text
<prefix>_cut_c3d4_hhhh_xsec_with_95cl.{png,pdf}
<prefix>_cut_c3d4_hhhh_xsec_with_95cl_atl_phys_pub_2025_003.{png,pdf}
<prefix>_cut_c3d4_hhhh_xsec_with_95cl_atl_phys_pub_2025_003_no_ratio_contours.{png,pdf}
```

The analogous shape files replace `_cut_` by `_shape_`.  They retain the
legacy viridis cross-section-ratio background, crimson central contour and
background-normalization band, perturbative-unitarity boundary, SM marker,
and optional ATL-PHYS-PUB-2025-003 curve.  Smoke and preview contours keep the
same non-final watermark as their other maps.  The default paper-style
viewport is $c_3\in[-20,20]$ and $d_4\in[-500,500]$.

The v2 exclusion boundary is evaluated from the point-dependent quantity

\[
R(c_3,d_4)=\frac{\sigma_{hhhh}(c_3,d_4)}
                  {\sigma_{95}(c_3,d_4)}
\]

and the 95% CL contour is (R=1).  By default, the logarithm of this ratio is
interpolated piecewise linearly on the sampled c3/d4 triangulation.  Passing
`--c3d4-contour-interpolation clough-tocher` instead uses SciPy's smooth
piecewise-cubic Clough--Tocher interpolator, with coordinate rescaling and no
extrapolation outside the sampled convex hull.  This deliberately does
not reuse the legacy common-(S_{95}) fitted event surface: in v2, the selected
threshold and background yield, and hence (S_{95}), can differ at every
coupling point.  The (B\times[0.25,4]) band is built from the corresponding
pointwise alternative limits.

For a final, unwatermarked contour, the complete coupling-point set recorded
in `method_manifest.json` must have finite cut or shape limits (and both
background-envelope reruns).  A missing or failed point is not bridged by the
triangulation: that contour is marked incomplete and skipped.  This prevents a
small surviving subset of points from being mistaken for a paper-ready result.
Exactly one result row is required for every manifest point, and its
production cross section must agree with the corresponding manifest input.
This check specifically prevents stale tables containing the historical
1-fb fallback from being replotted.  New result status files also record a
SHA-256 digest of their JSON table; a later hash mismatch is rejected.

Contours can be added to a completed or partially completed study without
retraining XGBoost or rerunning pyhf:

```bash
python 4h_analyzer.py \
  --replot-c3d4-study-contours \
  --study-outdir xgboost_c3d4_study_v2_uniform-smear-v1
```

Use the preview or smoke output directory explicitly when replotting those
modes.  By default the replot command inherits the luminosity, viewport,
resolution, cross-section source and overlay choice saved by the original
study.  The viewport, resolution and interpolation can be changed explicitly with the
existing `--c3d4-plot-*` options.  A requested luminosity or cross-section
source that disagrees with the manifest is rejected, so an old numerical
result cannot silently acquire a new label or heat map.

For example, an existing result can be compared with the smooth interpolation
without rerunning XGBoost or pyhf:

```bash
python 4h_analyzer.py \
  --replot-c3d4-study-contours \
  --study-outdir xgboost_c3d4_study_v2_uniform-smear-v1_fast-sm \
  --c3d4-contour-interpolation clough-tocher \
  --c3d4-plot-nbins 801
```

The interpolation method is recorded in each contour manifest.  Linear
interpolation remains the assumption-minimal reference; Clough--Tocher is a
smooth presentation and robustness comparison rather than additional physics
information.

`--no-c3d4-xsec-overlay` still writes the white-background contour with the
SM, unitarity and ATLAS overlays.  The colored variants require the same MG5
Chebyshev cross-section surface as the legacy analysis; if that source is
unavailable they are reported as skipped rather than silently replaced by a
different interpolation.  Every set is described in
`legacy_contour_manifest.json`, and the replot-only command writes the
top-level `contour_replot_manifest.json`.  Its status is `complete`, `partial`
or `failed`, with malformed/missing tables and skipped products listed under
`issues`.  `study_paper_ready` retains the state of the numerical campaign,
whereas the replot manifest's own `paper_ready` flag is true only when the
contour replot is also complete.

### Commands

The retained legacy command is, for example,

```bash
python 4h_analyzer.py --run-c3d4-limit-scan
```

The full v2 study is run in the ROOT/Python environment containing ROOT,
XGBoost, pyhf 0.7.6 and Optuna 4.9.0:

```bash
set +u
source ~/Projects/Herwig/Herwig-730-full-python3-rivet4/bin/activate
source ~/root310install/bin/thisroot.sh
source ~/xgb-py310/bin/activate
set -u
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python 4h_analyzer.py \
  --run-c3d4-xgboost-study \
  --study-mode full \
  --observable-set extended-91-v2 \
  --training-strategy pooled-crossfit-v2 \
  --cv-folds 5 \
  --optuna-trials 40 \
  --analysis-jobs 8 \
  --shape-jobs 4 \
  --progress-interval 30 \
  --study-outdir xgboost_c3d4_study_v2_uniform-smear-v1
```

In full mode, passing `--feature-profile corrected28|core52|full91` forces one
profile; omitting it performs the validation-only global comparison.  Fast-SM,
preview and smoke modes skip that comparison and use their mode defaults unless a
profile is supplied explicitly.  Models, fold
thresholds, bin edges, efficiencies, yields, raw and effective MC statistics,
\(S_{95}\), cut and shape \(\sigma_{95}\), baseline ratios, maps, contours,
Optuna histories and the gate decision are written only under the study
directory.  `method_manifest.json` records package versions, source commit,
input hashes, normalization inputs, fold populations, schema, feature names,
strategy, parameters and seeds, together with the shape worker count, thread
caps, checkpoint fingerprints and per-strategy resume counts.

The command also writes `<study-outdir>/sample_report/index.html`.  For every
observable in the selected profile this contains a normalized SM/background
comparison and a legacy-style stacked input-cross-section histogram.  The
stack uses the same physical event normalization as the v2 study and enlarges
the SM contribution by 1000 only for display.  Pass `--no-sample-report` to
disable this output.

The same gallery can be added to a completed study without retraining or
rerunning the statistical calculation:

```bash
python 4h_analyzer.py \
  --write-c3d4-v2-input-report \
  --study-outdir xgboost_c3d4_study_v2_uniform-smear-v1_fast-sm
```

## Limitations and deferred studies

The result is conditional on the resolved eight-candidate definition and on
the adopted detector/tagging approximation.  It does not profile scale, PDF,
shower, hadronization, MPI, tune, tagging or background cross-section
systematics.  Common-125-GeV re-pairing, topness, extra-jet activity, tagging
discriminants, boosted substructure and colour-flow observables are deferred
to dedicated systematic studies.  An improvement is measured against the
frozen baselines; it is not assumed by construction.
