# A beginner’s guide to the SM \(hhhh\to8b\) XGBoost–pyhf tutorial

This note explains what the executable notebook is doing and, just as
importantly, why each step is needed. It assumes familiarity with event
samples and histograms, but no previous use of cross-fitting, HistFactory, or
pyhf.

The analysis can be summarized as

```text
ROOT events
  -> validated 28-variable event table and several kinds of weight
  -> five train/validation/test rotations
  -> one frozen, out-of-fold XGBoost score per event
  -> validation-selected score bins
  -> signal/background count templates in five channels
  -> HistFactory JSON workspace
  -> pyhf likelihood fits, profile scans, CLs, and expected upper limits
```

There are two separate inference problems in that chain:

1. XGBoost learns how to rank events as signal-like or background-like.
2. pyhf uses the resulting binned event counts to infer or constrain a signal
   cross section.

pyhf never sees the 28 input variables, does not retrain XGBoost, and does not
fit a smooth function to the score. The phrase “fit the XGBoost score” means
“fit the Poisson counts in fixed bins of a previously trained XGBoost score.”

## 1. What enters from ROOT

The tutorial reads the `Data3` tree from one SM \(hhhh\to8b\) file and one
pure \(gg\to8b\) file. It requires the `extended-91-v2` schema and projects it
onto the named `corrected28` feature profile. Thus each accepted event has:

- 28 classifier inputs;
- a signed Monte Carlo weight;
- a stable source and event identifier;
- a deterministic fold number from 0 to 4.

The analysis-summary JSON beside each ROOT file records, among other
provenance, \(W_{\rm input}\), the normalization sum before analysis. The
loader checks the ROOT and JSON hashes and verifies that their metadata agree.
The weight is kept in a separate array and is explicitly excluded from the
classifier features.

This matters because allowing an event weight into the feature matrix would
let the classifier learn generator bookkeeping rather than event kinematics.

## 2. Why there are several weights

The stored event weight \(w_i\) is not yet the number of events expected at
the HL-LHC. The signed physical weight is

\[
w_i^{\rm phys}
=\mathcal L\,\sigma\,F\,\frac{w_i}{W_{\rm input}}.
\]

Here \(\mathcal L=3000~{\rm fb}^{-1}\), \(\sigma\) is the production cross
section, and \(F\) collects the fixed factors used to map the generated process
to the selected \(8b\) final state:

\[
F_s=K_s\,{\rm BR}(h\to b\bar b)^4\epsilon_b^8,\qquad
F_b=K_b\,\epsilon_b^8.
\]

The tutorial declares \(K_s=K_b=2\), \({\rm BR}(h\to b\bar b)=0.5824\), and
\(\epsilon_b=0.85\). The factor of two is therefore the configured
phenomenological \(K\)-factor; it is not a combinatorial factor. These
\(K\)-factors are a repository-wide analysis convention, not uncertainties.

Four weight concepts appear:

- **Raw weight** \(w_i\): the signed value stored for the event.
- **Physical weight** \(w_i^{\rm phys}\): its expected contribution at the
  declared luminosity and production cross section.
- **Unit-cross-section signal weight**: the signal physical weight with
  \(\sigma\) set to \(1~{\rm fb}\). The sum in a score bin is therefore the
  expected signal yield per fb.
- **Classifier training weight**: a non-negative weight proportional to
  \(\lvert w_i^{\rm phys}\rvert\), rescaled so the signal and background
  classes have equal total weight in a training rotation.

XGBoost requires a non-negative loss weight, which is why the absolute value
is used only for training. Histogram yields always retain the sign:

\[
N=\sum_i w_i^{\rm phys},\qquad
\delta N_{\rm MC}=\sqrt{\sum_i (w_i^{\rm phys})^2}.
\]

Negative-weight events can cancel positive ones in \(N\), but not in the
Monte Carlo variance. The useful diagnostic

\[
N_{\rm eff}=\frac{(\sum_i w_i)^2}{\sum_iw_i^2}
\]

describes the precision of a weighted sample. It can be much smaller than the
raw number of events.

The signal template is expressed per \(1~{\rm fb}\), so the pyhf parameter of
interest is a production cross section before the fixed \(K_s\) factor:

\[
N_{s,i}(\sigma_{hhhh})=\sigma_{hhhh}\,s_i^{(1\,{\rm fb})}.
\]

Consequently the workspace POI `sigma_hhhh_fb` is in fb and
\(\mu=\sigma_{hhhh}/\sigma_{hhhh}^{\rm SM}\). Changing the fixed \(K_s\) would
change the mapping between the POI and yield. The background \(K_b=2\)
doubles the nominal background; in a simple background-dominated counting
experiment this would weaken a limit by roughly \(\sqrt{2}\), although the
actual multi-bin result also depends on shapes and nuisance parameters.

## 3. Why five-fold cross-fitting is used

If a classifier is scored on events used to train it, statistical
fluctuations can look like genuine separation. The resulting template can be
too optimistic. Cross-fitting avoids this by ensuring that every likelihood
event is scored by a model that did not train on that event.

The fixed rotation is:

| rotation | test/inference fold | bin-selection fold | training folds |
| ---: | ---: | ---: | --- |
| 0 | 0 | 1 | 2, 3, 4 |
| 1 | 1 | 2 | 0, 3, 4 |
| 2 | 2 | 3 | 0, 1, 4 |
| 3 | 3 | 4 | 0, 1, 2 |
| 4 | 4 | 0 | 1, 2, 3 |

Every event is:

- in a training set three times;
- in a bin-selection set once;
- in a test/inference set once.

Only its test prediction becomes its out-of-fold score. The five test folds
are disjoint and their union is the complete event sample, so the total signed
yield closes exactly.

The validation score is also produced by a model that did not train on that
validation event. It is used to choose bin edges, never to fill the test
template for the same rotation.

There is one subtlety. Fold \(f+1\) chooses the edges for channel \(f\), and in
the next rotation it becomes the test sample for channel \(f+1\). Thus no event
selects the binning of its own channel, but the combined five-channel
likelihood eventually contains all five validation folds as other channels.
This can cause mild selection-induced optimism. Exact independence would
require an additional bin-selection sample or nested cross-validation. The
tutorial records this limitation instead of claiming stronger independence.

## 4. What the XGBoost score means

XGBoost minimizes a class-balanced weighted binary log loss on the three
training folds. Its output lies between zero and one and ranks events:
larger values are more signal-like for this training problem.

It need not be a calibrated posterior probability. For this analysis only its
ordering and shape are used. The weighted ROC curve answers how the
signal-efficiency/background-rejection trade-off changes with a score
threshold. Feature importance reports which inputs the trees used most often
or most profitably, but does not establish a causal physics interpretation.

## 5. How score bins are selected

For each rotation, candidate edges start from validation-background score
quantiles

\[
[0,\ 0.50,\ 0.75,\ 0.90,\ 0.97,\ 1].
\]

Subsets give candidate histograms with two to five bins. A candidate is kept
only if each validation-background bin has:

- positive signed background yield;
- at least 25 raw events;
- \(N_{\rm eff}\ge10\).

The candidate with the best expected limit is selected, with fewer bins
preferred when candidates lie within 1% of the optimum. These rules prevent a
small, unstable high-score tail from masquerading as strong sensitivity.

One validation fold is multiplied by five to represent the full expected
yield during this choice. Multiplying every weight by five multiplies
\(\sum w\) by five and \(\sum w^2\) by 25, so its *relative* MC uncertainty is
still that of one fold. A truly five-times-larger independent sample would
have a relative uncertainty smaller by \(\sqrt5\). The tutorial convention is
therefore conservative and can favor coarser bins.

After selection, the edges are frozen and applied to the disjoint test fold.
If a test bin has nonpositive signed background, its value is not clipped.
The code tries only the coarser nested alternatives already defined using the
validation fold and otherwise stops.

## 6. From histograms to a HistFactory workspace

Each test fold becomes one pyhf **channel**. A channel contains a signal
sample and a background sample. For bin \(i\):

\[
s_i=\sum_{\rm signal} w_i^{(1\,{\rm fb})},\qquad
b_i=\sum_{\rm background}w_i^{\rm phys},
\]

\[
\delta s_i=\sqrt{\sum_{\rm signal}(w_i^{(1\,{\rm fb})})^2},\qquad
\delta b_i=\sqrt{\sum_{\rm background}(w_i^{\rm phys})^2}.
\]

The workspace is a JSON representation of a HistFactory likelihood. Its main
objects are:

- **channel**: a disjoint set of score bins, here one held-out fold;
- **sample**: a process template inside a channel, here signal or \(gg8b\);
- **observation**: the bin values being fitted;
- **modifier**: a rule describing how a parameter changes a sample;
- **measurement**: the declaration of the POI, parameter bounds, and initial
  values.

The modifiers are:

- shared signal `normfactor` `sigma_hhhh_fb`: an unconstrained multiplicative
  POI;
- channel-local signal and background `staterror`: constrained nuisance
  parameters representing the supplied \(\sqrt{\sum w^2}\);
- shared `gg8b_norm` `normsys`: one nuisance coherently scaling background in
  all channels between the declared 0.90 and 1.10 variations.

The `gg8b_norm` term is deliberately pedagogical. It is not a measured 10%
uncertainty. The recorded 2.54% event-generation integration error is
provenance and is not used as a theory nuisance.

Schematically,

\[
\mathcal L(\sigma,\boldsymbol\theta)
=\prod_i{\rm Pois}\!\left(
n_i\mid \sigma s_i(\boldsymbol\theta)+b_i(\boldsymbol\theta)
\right)
\prod_k\pi_k(\theta_k).
\]

The Poisson factors use the score-bin counts. The constraint factors
\(\pi_k\) encode auxiliary information about nuisance parameters. Accordingly
`workspace.data(model)` contains both the main bin observations and auxiliary
data required by those constraints.

## 7. What an Asimov dataset is

An Asimov dataset replaces random observations by their exact model
expectation at specified parameter values. It is deterministic, may contain
fractional bin values, and has no Poisson fluctuation.

For the headline result the specified truth is background only:

\[
n_i^{A}=b_i,\qquad \sigma_{hhhh}=0,
\]

with nuisance parameters at their nominal values. Fitting this dataset should
give \(\hat\sigma_{hhhh}\simeq0\). It answers: “What limit would a typical
background-only experiment be expected to set under this model?”

An “observed limit evaluated on the background-only Asimov dataset” is not an
observed collision-data result. It normally coincides with the median expected
limit by construction.

The tutorial also creates a separate injected Asimov dataset with

\[
\sigma_{\rm inj}=0.5\,\sigma_{95}^{\rm expected}
\]

and a \(+0.5\sigma\) displacement of `gg8b_norm`. Its auxiliary data are
shifted consistently. It exists to make the post-fit movement, nuisance
pulls, and correlations visible. It is labelled **NOT DATA** everywhere and
is never used for the headline expected limit. With a converged optimizer,
fitting an Asimov dataset generated by the same model should recover its
generating parameters to numerical precision.

## 8. Global fits and profile-likelihood scans

`pyhf.infer.mle.fit(data, model)` varies the POI and all nuisance parameters
to find the global maximum-likelihood estimate
\((\hat\sigma,\hat{\boldsymbol\theta})\).

`pyhf.infer.mle.fixed_poi_fit(sigma, data, model)` holds the cross section at a
chosen value and re-optimizes every nuisance parameter. These re-optimized
values are the conditional estimates
\(\hat{\hat{\boldsymbol\theta}}(\sigma)\).

The profile-likelihood curve compares the conditional and global fits:

\[
\Delta[-\ln\mathcal L](\sigma)
=-\ln\mathcal L(\sigma,\hat{\hat{\boldsymbol\theta}}(\sigma))
 \ln\mathcal L(\hat\sigma,\hat{\boldsymbol\theta}).
\]

Profiling is crucial: a proposed signal is judged after allowing each nuisance
to take the value that best accommodates that signal, subject to its
constraint. A pull reports a fitted nuisance displacement in units of its
constraint width. A correlation matrix shows which fitted parameters can
partly compensate each other.

## 9. \(q_\mu\), \(\widetilde q_\mu\), \(CL_s\), and the upper limit

For each proposed signal cross section \(\mu\equiv\sigma_{hhhh}\), pyhf builds
a profile-likelihood-ratio test statistic. The tutorial requests
`test_stat="qtilde"`, the bounded \(\widetilde q_\mu\) convention appropriate
when the physical signal strength cannot be negative.

`pyhf.infer.hypotest` uses the asymptotic distributions of this statistic to
calculate \(CL_s\). In words, \(CL_s\) compares the incompatibility of the
signal-plus-background hypothesis with the data while dividing by a
background-sensitivity factor. This protects against excluding signals in
regions where the experiment has little power.

For a 95% CL upper limit, scan the cross section upward and find the crossing

\[
CL_s(\sigma_{95})=0.05.
\]

Cross sections above that crossing are excluded by this convention. pyhf
returns five expected crossings corresponding to background-only outcomes at
\(-2\sigma,-1\sigma,\) median, \(+1\sigma,+2\sigma\). They describe the
expected spread over hypothetical background-only experiments, not an
uncertainty on one fitted number.

In pyhf 0.7.6 there is a version-specific trap:
`upper_limit(data, model, scan=None, level=...)` fails to forward a non-default
`level` to its automatic root finder. The tutorial therefore:

- uses `upper_limit` with an explicit scan when demonstrating that public
  convenience API; and
- calls `toms748_scan` directly for the precise reported result, passing
  `level=1-confidence_level`.

A regression test verifies that the 90% and 95% limits differ and that the
requested confidence level is recorded.

## 10. How this relates to the production `fast-sm` analysis

The tutorial reproduces the central architecture of `fast-sm`:

- complete-event deterministic five-fold SM cross-fitting;
- fixed/frozen classifier models;
- signed physical yields and unit-cross-section signal templates;
- validation-defined score regions;
- a pyhf score-shape likelihood and expected limits.

It is intentionally smaller and is not numerically identical:

- this lesson uses `corrected28`; production `fast-sm` on
  `extended-91-v2` uses `full91`;
- this lesson has only SM \(hhhh\to8b\) and pure \(gg\to8b\);
- production can contain the configured full background composition and
  coupling-point machinery;
- this lesson adds an illustrative 10% normalization nuisance and an injected
  Asimov diagnostic specifically for teaching;
- its one-fold validation-yield scaling keeps conservative one-fold relative
  MC statistics.

Thus the notebook is a faithful miniature of the data flow and statistical
logic, not a replacement for the production driver or a reproduction of its
final numerical limit.

## 11. What the result does and does not mean

The headline number is a background-only median expected 95% CL limit for the
two-sample model encoded in `workspace.json`. It can be converted to
\(\mu_{95}\) by dividing by the configured SM cross section.

It is not publication-ready physics because the model omits reducible and
mistag backgrounds, observed collision data, and genuine detector and theory
shape variations. Omitting positive background components will generally
make the expected sensitivity look too strong, although retraining and
different score shapes prevent a universal numerical rescaling.

The plots are nevertheless constructed to publication standards—fixed
styling, explicit labels, vector PDF, 300-dpi PNG, uncertainty bands, ratio
panel, and provenance—so that the plotting patterns can be reused once the
physics model is complete.
