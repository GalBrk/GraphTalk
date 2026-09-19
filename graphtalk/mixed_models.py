"""Population-averaged (GEE) cross-check for the primer-significance question.

`graphtalk/significance.py` deliberately stays hand-rolled and
scipy/statsmodels-free (see its own module docstring). This module is a
scoped exception, not a reversal of that policy: it exists for a question a
paired permutation test structurally can't answer cheaply -- "pooling all
six conditions and every instance into a single model per model_family,
does the primer still move accuracy" -- without fragmenting the data into
36 separate n=30 cells the way `scripts/check_significance.py`'s per-cell
tests do. `docs/sweep-findings.md`'s "Pool the cells" suggestion, actually
implemented.

**GEE, not a full mixed-effects (subject-specific) GLMM.** A Generalized
Estimating Equation, grouped on the graph, is the standard
population-averaged analogue of what the clustered permutation test already
targets: repeated measures (seven conditions x six tasks) per graph,
correlated with each other, with no claim about a random-intercept
distribution the way a true GLMM would need.

**Grouped on the graph index, not `instance_id`.** An `instance_id` is
`"<task>/<index>"`, and the six tasks sharing an index are the *same
graph* -- identical nodes and edges, byte-identical encoding in
`prompts.jsonl` -- asked six different questions. Grouping on the full
string put the seven conditions on one task in a group and treated the same
graph's other five tasks as unrelated observations, which is both wrong on
its own terms and a different cluster granularity from
`scripts/check_significance.py`'s. Since being comparable to that script
cell-for-cell is this module's entire reason to exist, the two have to
cluster on the same thing; `check_significance._graph_index` is the shared
definition. It's far more tractable to fit reliably at this
data's scale (30-180 clusters, binary outcome) than a subject-specific
GLMM, which needs numerical integration or a variational approximation to
fit at all. The subject-specific, partial-pooling-across-models-and-conditions
question is `graphtalk/hierarchical_model.py`'s job instead (PyMC), not this
module's -- the two are deliberately not overlapping.

**Independence working correlation, not exchangeable.** An exchangeable
working correlation (the more natural first choice for "repeated conditions
on the same graph") was tried first and reliably fails to converge under
`statsmodels`' default `ctol`/`maxiter` on this identity-link binomial model
-- a known numerical fragility of identity-link binomial GEE, not specific
to this data (raising `maxiter` to 200 doesn't fix it). Switching to an
independence working correlation converges cleanly, and lands on point
estimates matching the exchangeable fit's within thousandths (both close to
the raw per-condition mean difference, as expected for a saturated
categorical predictor). This is not a compromise on validity: GEE's whole
point (Liang & Zeger 1986) is that the robust "sandwich" standard errors
stay asymptotically valid under a *misspecified* working correlation, as
long as the mean model itself (`condition` as the only predictor) is
correct -- independence only gives up some statistical *efficiency*
(tighter CIs) relative to a correctly-specified correlation structure,
never correctness. `fit_gee_one_model` reports `converged` on every row so
a caller can see this for themselves rather than trust the module's word.

**Identity link, not the GEE default logit.** `sm.families.Binomial()`
defaults to a logit link, whose coefficients are log-odds -- not directly
comparable to `significance.paired_permutation_test_clustered`'s
`observed_diff` (a raw probability difference) without a marginal-effects
conversion. Fitting with an identity link instead makes each condition's
coefficient *be* the same quantity `check_significance.py` calls `delta`:
the average difference in P(correct) against `none`, holding the GEE's
population-averaged structure fixed. This is the standard "risk difference"
regression trick (identity-link binomial GEE), not a novel choice, and is
what makes this module's output directly diffable against
`analysis/significance_report.csv` column-for-column
(`condition, delta, ci_low, ci_high, p_value`).

**No multiplicity correction is applied here, and that is a limitation of
the output, not a property of the fit.** This module used to claim that
fitting all six conditions simultaneously "already adjusts each condition's
estimate for the others (it is not six independent tests fished from the
same pile)". That is arithmetically false: `condition` is a saturated
categorical predictor with no other covariates, so each coefficient is just
`mean(condition) - mean(none)` and refitting any one condition on its own
reproduces the joint fit's coefficient to ~13 decimal places. These *are*
six separate comparisons and carry the full multiplicity burden.

So a caller citing these p-values must apply
`graphtalk.significance.benjamini_hochberg` to the `p_value` column first,
in the same family `check_significance.py` would use. Raw p-values are
returned because the correction depends on which family the caller is
asking about, not because none is needed.

**And this is a consistency check, not independent corroboration.** With an
identity link and a saturated predictor, `delta` here is algebraically the
same raw mean difference `paired_permutation_test_clustered` reports -- the
two agree to ~1e-14 on every cell, by construction rather than by
confirmation. The GEE's p-value is also uniformly smaller (48 of 48 cells
on the last full comparison), since a sandwich-SE Wald test is more
permissive here than a permutation test. Read agreement as "the pipeline
computed the effect size correctly", never as a second, independent line of
evidence for an effect.
"""

import warnings

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import DomainWarning

from graphtalk.analysis import graph_index as _graph_index

CONTROL = "none"

# The GEE-side name for one condition's coefficient, from
# `C(condition, Treatment(reference='none'))`'s patsy-generated term names,
# e.g. "C(condition, Treatment(reference='none'))[T.all]" -> "all".
_TERM_PREFIX = "C(condition, Treatment(reference='none'))[T."


def _condition_from_term(term: str) -> str | None:
  """`None` for the intercept (the `none` reference level itself) or any
  other non-condition term; the bare condition name otherwise."""
  if not term.startswith(_TERM_PREFIX) or not term.endswith("]"):
    return None
  return term[len(_TERM_PREFIX):-1]


def _main_sweep_scope(frame: pd.DataFrame, model: str) -> pd.DataFrame:
  """One model's main-sweep (non-thinking-arm) rows -- the same scope as
  `check_significance.py`'s main-sweep report, so this module's `delta` is
  comparable to `significance_report.csv`'s rows cell for cell.
  Non-terminating rows are **not** filtered out here (nor is their `exact`
  overridden) -- `graphtalk.analysis.build_frame` already forces them to
  0.0 upstream, so they enter this fit as ordinary, trustworthy rows,
  exactly like `check_significance.py`'s own main-sweep frame. This
  function used to drop them (matching that pipeline's old `excluded`
  bound); it no longer needs to, now that bound is gone -- see
  `scripts/check_significance.py`'s module docstring for the full
  reasoning. Renamed from `_main_sweep_excluded`, which became actively
  misleading once it stopped excluding anything."""
  return frame[(frame["model"] == model) & (~frame["is_think"])]


def fit_gee_one_model(frame: pd.DataFrame, metric: str = "exact") -> pd.DataFrame:
  """Fits one identity-link binomial GEE of `metric` on `condition`
  (`none` reference), clustered on the graph via an independence
  working correlation (see the module docstring's "Independence working
  correlation, not exchangeable" section for why).

  `frame` must already be scoped to exactly one model and the rows that
  should enter the fit (see `_main_sweep_scope`) -- this function does not
  filter by model itself, so a caller iterating over several models
  doesn't have to re-derive that scope from this module.

  Returns one row per non-control condition actually present in `frame`:
  `condition`, `delta` (the identity-link coefficient -- directly a
  probability difference against `none`), `std_err`, `ci_low`, `ci_high`,
  `p_value`, `n_obs`, `n_groups` (clusters, i.e. distinct graphs), and
  `converged`. Raises `ValueError` if `frame` has no `none` rows (the
  reference level `patsy` needs) or is otherwise too small to fit.

  `n_groups` is read back off the fitted model, not counted in pandas
  beforehand: counted separately it reports what the grouping *should* have
  been even when the fit never received it, which is exactly the failure a
  reader would use this column to rule out.
  """
  if frame.empty:
    raise ValueError("fit_gee_one_model got an empty frame")
  if CONTROL not in frame["condition"].unique():
    raise ValueError(
        f"frame has no {CONTROL!r} rows -- GEE needs the reference level present"
    )
  # The identity link deliberately doesn't respect the Binomial family's
  # [0, 1] domain during fitting (see the module docstring's "risk
  # difference regression" note) -- statsmodels warns about exactly that
  # known, accepted characteristic on every call; filtered here by class
  # rather than blanket-suppressed, so an unrelated warning still surfaces.
  # The graph, not the "<task>/<index>" instance id -- see the module
  # docstring. Shares `analysis.graph_index` rather than re-splitting the
  # string here, so this and `check_significance.py` can't drift apart into
  # different cluster granularities again.
  frame = frame.assign(_graph=frame["instance_id"].map(_graph_index))
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DomainWarning)
    model = smf.gee(
        f"{metric} ~ C(condition, Treatment(reference={CONTROL!r}))",
        groups="_graph",
        data=frame,
        family=sm.families.Binomial(link=sm.families.links.Identity()),
        cov_struct=sm.cov_struct.Independence(),
    )
    result = model.fit(maxiter=200)
  conf_int = result.conf_int()
  rows = []
  for term in result.params.index:
    condition = _condition_from_term(term)
    if condition is None:
      continue
    rows.append({
        "condition": condition,
        "delta": result.params[term],
        "std_err": result.bse[term],
        "ci_low": conf_int.loc[term, 0],
        "ci_high": conf_int.loc[term, 1],
        "p_value": result.pvalues[term],
        "n_obs": int(result.nobs),
        # Off the fit, not counted in pandas: a separately-counted value
        # reports the intended grouping even when the model never received
        # it, so it can't witness that the clustering actually happened.
        "n_groups": len(result.model.group_labels),
        "converged": bool(result.converged),
    })
  return pd.DataFrame(rows)


def fit_gee_all_models(frame: pd.DataFrame, metric: str = "exact") -> pd.DataFrame:
  """`fit_gee_one_model`, once per distinct `model` in `frame`
  (main-sweep-only -- see `_main_sweep_scope`), concatenated with a
  `model`/`model_family` column so the result lines up against
  `analysis/significance_report.csv`'s `group` column for the same
  main-sweep rows.
  """
  frames = []
  for model in sorted(frame["model"].unique()):
    scoped = _main_sweep_scope(frame, model)
    if scoped.empty:
      continue
    fit = fit_gee_one_model(scoped, metric=metric)
    fit.insert(0, "model", model)
    frames.append(fit)
  if not frames:
    return pd.DataFrame(columns=[
        "model", "condition", "delta", "std_err", "ci_low", "ci_high",
        "p_value", "n_obs", "n_groups", "converged",
    ])
  return pd.concat(frames, ignore_index=True)
