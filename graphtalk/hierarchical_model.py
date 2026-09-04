"""Bayesian hierarchical partial-pooling cross-check for the primer-
significance question.

`graphtalk/mixed_models.py`'s GEE fits one population-averaged model per
model_family, already pooling six conditions into one fit instead of six
n=30 permutation tests. This module goes one step further: partial pooling
*across models too*, so a noisy per-(model, condition) estimate shrinks
toward the pooled mean it's evidence for, rather than being read in
isolation. This is `docs/sweep-findings.md`'s "Pool the cells" idea taken
to its natural conclusion -- GEE pools within a model, this pools across
models and conditions jointly.

**Model** (log-odds/logit scale throughout -- see "Why logit, not identity
link" below):

    logit(P(correct)) = alpha_model[m] + beta_condition[c] + gamma[m, c]
                         + u_instance[i]           (c != "none" only)
    logit(P(correct)) = alpha_model[m] + u_instance[i]         (c == "none")

`alpha_model` is each model's own baseline (control-condition) log-odds of
being correct -- deliberately unpooled (a flat `Normal(0, 2)` prior, not
hierarchical): the four models are different architectures at different
scales, not exchangeable draws from one population, so there is no
population to partially pool their baselines toward.

`beta_condition[c]` is the *population-level* effect of condition `c`,
shared across models -- `Normal(0, sigma_condition)`, `sigma_condition ~
HalfNormal(1)`. `gamma[m, c]` is model `m`'s own *deviation* from that
population effect for condition `c` -- `Normal(0, sigma_gamma)`, its own
hierarchical scale. Both use a non-centered parameterization
(`raw ~ Normal(0, 1)`, `param = raw * sigma`) -- the standard fix for the
funnel geometry that centered hierarchical parameterizations give NUTS
trouble sampling, especially when `sigma` itself is close to zero (which is
exactly the regime a "no real effect" condition lives in).

`u_instance[i]` is one graph instance's own difficulty, `Normal(0,
sigma_instance)` -- the Bayesian analogue of `paired_permutation_test
_clustered`'s `cluster_ids`: a hard graph is hard across every condition
and every model, and this term is what prevents the six conditions applied
to the same graph from being read as six independent pieces of evidence.
Unlike the permutation test's `cluster_ids` (which key on `(model,
instance_id)` -- see `graphtalk/significance.py`'s module docstring, "the
cluster id carries model"), `u_instance` here is shared *across* models on
purpose: the same 180 graphs underlie every model's rows, and "this graph
is structurally easy/hard" is a property of the graph, not of any one
model. Model-specific idiosyncrasy on top of that shared difficulty is what
`alpha_model`/`gamma` already capture separately.

**Why logit, not the identity link `graphtalk/mixed_models.py` uses.** The
identity link there is a deliberate trick to make GEE's coefficients
directly read as `check_significance.py`'s raw probability-difference
`delta`. Full MCMC has no analogous convergence fragility to work around
(NUTS samples the exact posterior; there's no iterative reweighting step
that can fail to converge the way identity-link GEE's did), so there's no
forcing reason to leave the natural link for a binary outcome. Report
`beta_condition`/`gamma` on the log-odds scale they're fit on: sign and
`P(> 0)` are directly interpretable (positive means the condition helps),
magnitude is not a percentage-point delta the way `mixed_models.py`'s is --
read this module's numbers for direction and credible-interval width, not
as a plug-in replacement for `significance_report.csv`'s `delta` column.

**Convergence diagnostics are not optional here.** Unlike a permutation
test (exact by construction below `_EXACT_CLUSTER_THRESHOLD`, or a
well-understood Monte Carlo approximation above it), an unconverged MCMC
chain can look plausible while being wrong -- `fit_summary`'s `r_hat`/
`ess_bulk` columns and `n_divergences` are not decoration; a caller must
check them, not just read the posterior mean. `graphtalk.significance`'s
whole design philosophy (hand-rolled and *provably* correct for what it
claims) is why this module lives apart from it rather than replacing it --
this is a genuinely different kind of trust, not a strictly stronger one.
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

CONTROL = "none"


def build_model(
    frame: pd.DataFrame, metric: str = "exact",
) -> tuple[pm.Model, dict]:
  """Builds (but does not fit) the hierarchical model described in the
  module docstring against `frame`.

  `frame` must carry `model`, `condition`, `instance_id`, and `metric`
  (0/1) columns -- already scoped to whatever rows should enter the fit
  (main sweep, non-terminating excluded, etc.; this function does no
  filtering itself, matching `graphtalk.mixed_models.fit_gee_one_model`'s
  convention).

  Returns `(model, index)`, where `index` carries the label lists
  (`models`, `conditions`, `instances`) needed to map the fitted
  `beta_condition`/`gamma`/`alpha_model` arrays' positions back to names.
  """
  if frame.empty:
    raise ValueError("build_model got an empty frame")
  if CONTROL not in frame["condition"].unique():
    raise ValueError(
        f"frame has no {CONTROL!r} rows -- the model needs the reference "
        "level present"
    )

  models = sorted(frame["model"].unique())
  conditions = sorted(c for c in frame["condition"].unique() if c != CONTROL)
  instances = sorted(frame["instance_id"].unique())

  model_pos = {m: i for i, m in enumerate(models)}
  condition_pos = {c: i for i, c in enumerate(conditions)}
  instance_pos = {ins: i for i, ins in enumerate(instances)}

  model_idx = frame["model"].map(model_pos).to_numpy()
  instance_idx = frame["instance_id"].map(instance_pos).to_numpy()
  is_control = (frame["condition"] == CONTROL).to_numpy()
  # Control rows get condition index 0 arbitrarily -- `is_control` masks
  # their condition/interaction contribution to exactly 0 regardless, so
  # which valid index they carry never affects the likelihood.
  condition_idx = frame["condition"].map(condition_pos).fillna(0).astype(int).to_numpy()
  y = frame[metric].to_numpy()

  n_models, n_conditions, n_instances = len(models), len(conditions), len(instances)

  with pm.Model() as model:
    alpha_model = pm.Normal("alpha_model", 0.0, 2.0, shape=n_models)

    sigma_condition = pm.HalfNormal("sigma_condition", 1.0)
    beta_condition_raw = pm.Normal("beta_condition_raw", 0.0, 1.0, shape=n_conditions)
    beta_condition = pm.Deterministic(
        "beta_condition", beta_condition_raw * sigma_condition
    )

    sigma_gamma = pm.HalfNormal("sigma_gamma", 1.0)
    gamma_raw = pm.Normal("gamma_raw", 0.0, 1.0, shape=(n_models, n_conditions))
    gamma = pm.Deterministic("gamma", gamma_raw * sigma_gamma)

    sigma_instance = pm.HalfNormal("sigma_instance", 1.0)
    u_instance_raw = pm.Normal("u_instance_raw", 0.0, 1.0, shape=n_instances)
    u_instance = pm.Deterministic("u_instance", u_instance_raw * sigma_instance)

    condition_effect = pt.where(
        is_control,
        0.0,
        beta_condition[condition_idx] + gamma[model_idx, condition_idx],
    )
    logit_p = alpha_model[model_idx] + u_instance[instance_idx] + condition_effect
    pm.Bernoulli("y_obs", logit_p=logit_p, observed=y)

  index = {"models": models, "conditions": conditions, "instances": instances}
  return model, index


def fit(
    frame: pd.DataFrame, metric: str = "exact", draws: int = 1000, tune: int = 1000,
    chains: int = 2, target_accept: float = 0.9, seed: int = 1234,
    progressbar: bool = False,
) -> tuple[az.InferenceData, dict]:
  """Builds and fits the hierarchical model, returning `(trace, index)` --
  see `build_model` for `index` and the module docstring for the model
  itself. `draws`/`tune` default to a real, publishable-quality run (not
  the fast settings a quick check would use); pass smaller values for a
  fast smoke test, larger for a final reported number.
  """
  model, index = build_model(frame, metric=metric)
  with model:
    trace = pm.sample(
        draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        random_seed=seed, progressbar=progressbar,
    )
  return trace, index


def fit_summary(trace: az.InferenceData, index: dict) -> pd.DataFrame:
  """One row per (model, condition): the population-level `beta_condition`
  contribution, model `m`'s own `gamma` deviation, and their sum (`total`,
  the fully shrunk model-and-condition-specific estimate) -- posterior
  mean, 94% HDI, and `prob_positive` (`P(total > 0)`, the Bayesian
  analogue of a significance flag) for each, plus `r_hat`/`ess_bulk`/
  `n_divergences` so a caller can check convergence before trusting any of
  it (see the module docstring's closing paragraph).
  """
  models, conditions = index["models"], index["conditions"]
  beta = trace.posterior["beta_condition"]  # dims: chain, draw, beta_condition_dim_0
  gamma = trace.posterior["gamma"]          # dims: chain, draw, gamma_dim_0, gamma_dim_1
  summary = az.summary(
      trace, var_names=["beta_condition", "gamma"], hdi_prob=0.94, kind="diagnostics"
  )
  n_divergences = int(trace.sample_stats["diverging"].sum())

  def _hdi_bounds(data_array) -> tuple[float, float]:
    """`(lower, upper)` from `az.hdi`, which returns a `Dataset` with one
    data variable (named after `data_array.name`, or the fallback `"x"`
    when the array is unnamed -- e.g. after a `DataArray + DataArray`
    between two differently-named arrays, which drops the name) holding a
    2-element `hdi` dim ordered `("lower", "higher")`."""
    hdi = az.hdi(data_array, hdi_prob=0.94)
    values = np.asarray(next(iter(hdi.data_vars.values())))
    return float(values[0]), float(values[1])

  rows = []
  for ci, condition in enumerate(conditions):
    beta_c = beta.isel({beta.dims[-1]: ci})
    beta_mean = float(beta_c.mean())
    beta_hdi_low, beta_hdi_high = _hdi_bounds(beta_c)
    beta_r_hat = summary.loc[f"beta_condition[{ci}]", "r_hat"]
    beta_ess = summary.loc[f"beta_condition[{ci}]", "ess_bulk"]
    for mi, model_name in enumerate(models):
      total = beta_c + gamma.isel({gamma.dims[-2]: mi, gamma.dims[-1]: ci})
      total_mean = float(total.mean())
      total_hdi_low, total_hdi_high = _hdi_bounds(total)
      prob_positive = float((total > 0).mean())
      gamma_r_hat = summary.loc[f"gamma[{mi}, {ci}]", "r_hat"]
      gamma_ess = summary.loc[f"gamma[{mi}, {ci}]", "ess_bulk"]
      rows.append({
          "model": model_name,
          "condition": condition,
          "beta_condition_mean": beta_mean,
          "beta_condition_hdi_low": beta_hdi_low,
          "beta_condition_hdi_high": beta_hdi_high,
          "total_mean": total_mean,
          "total_hdi_low": total_hdi_low,
          "total_hdi_high": total_hdi_high,
          "prob_positive": prob_positive,
          "r_hat": max(beta_r_hat, gamma_r_hat),
          "ess_bulk": min(beta_ess, gamma_ess),
          "n_divergences": n_divergences,
      })
  return pd.DataFrame(rows)
