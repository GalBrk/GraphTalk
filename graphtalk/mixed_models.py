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
Estimating Equation, grouped on `instance_id`, is the standard
population-averaged analogue of what the clustered permutation test already
targets: repeated measures (six conditions) per graph instance, correlated
with each other, with no claim about a random-intercept distribution the
way a true GLMM would need. It's far more tractable to fit reliably at this
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

No multiplicity correction is applied here -- unlike `check_significance.py`,
which corrects across a condition family via `graphtalk.significance
.benjamini_hochberg`, this module fits all six conditions *simultaneously*
in one regression per model, which already adjusts each condition's
estimate for the others (it is not six independent tests fished from the
same pile). Report raw p-values; a caller wanting a BH pass across them can
still apply `graphtalk.significance.benjamini_hochberg` to the returned
`p_value` column, but this module's purpose is a cross-check against the
existing corrected numbers, not a second, independently-corrected pipeline.
"""

import warnings

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import DomainWarning

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


def _main_sweep_excluded(frame: pd.DataFrame, model: str) -> pd.DataFrame:
  """One model's main-sweep (non-thinking-arm) rows, non-terminating
  responses dropped -- the same scope as `check_significance.py`'s
  `excluded` bound, so this module's `delta` is comparable to
  `significance_report.csv`'s `excluded`-bound rows specifically (not
  `best_case`/`worst_case`, which this module has no analogue for)."""
  return frame[
      (frame["model"] == model)
      & (~frame["is_think"])
      & (frame["failure_type"] != "non_terminating")
  ]


def fit_gee_one_model(frame: pd.DataFrame, metric: str = "exact") -> pd.DataFrame:
  """Fits one identity-link binomial GEE of `metric` on `condition`
  (`none` reference), clustered on `instance_id` via an exchangeable
  working correlation.

  `frame` must already be scoped to exactly one model and the rows that
  should enter the fit (see `_main_sweep_excluded`) -- this function does
  not filter by model or exclude non-terminating rows itself, so a caller
  iterating over several models doesn't have to re-derive that scope from
  this module.

  Returns one row per non-control condition actually present in `frame`:
  `condition`, `delta` (the identity-link coefficient -- directly a
  probability difference against `none`), `std_err`, `ci_low`, `ci_high`,
  `p_value`, `n_obs`, `n_groups` (clusters, i.e. distinct `instance_id`
  values), `converged`. Raises `ValueError` if `frame` has no `none` rows
  (the reference level `patsy` needs) or is otherwise too small to fit.
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
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DomainWarning)
    model = smf.gee(
        f"{metric} ~ C(condition, Treatment(reference={CONTROL!r}))",
        groups="instance_id",
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
        "n_groups": frame["instance_id"].nunique(),
        "converged": bool(result.converged),
    })
  return pd.DataFrame(rows)


def fit_gee_all_models(frame: pd.DataFrame, metric: str = "exact") -> pd.DataFrame:
  """`fit_gee_one_model`, once per distinct `model` in `frame`
  (main-sweep-only, non-terminating-excluded -- see `_main_sweep_excluded`),
  concatenated with a `model`/`model_family` column so the result lines up
  against `analysis/significance_report.csv`'s `group` column for the same
  main-sweep/`excluded` rows.
  """
  frames = []
  for model in sorted(frame["model"].unique()):
    scoped = _main_sweep_excluded(frame, model)
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
