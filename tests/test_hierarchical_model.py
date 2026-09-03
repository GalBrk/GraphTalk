"""Tests for graphtalk/hierarchical_model.py.

Split deliberately into two speed tiers, mirroring this project's existing
convention for expensive checks (`scripts/check_significance.py`'s `--mde`
flag: real by default, off unless asked for):

- **Fast, deterministic** (`Test*` below without "real_sampling" in the
  name): `build_model`'s structural correctness (shapes, error handling)
  needs no MCMC at all. `fit_summary`'s numeric extraction (means, HDI
  bounds, `prob_positive`, `r_hat`/`ess_bulk` wiring) is tested against a
  hand-built `arviz.InferenceData` with known posterior values instead of a
  real `pm.sample` trace -- this is what actually needs unit testing (the
  aggregation math and the `az.hdi`/`az.summary` indexing), and testing it
  this way is both instant and exact, rather than at the mercy of MCMC's
  own randomness.
- **One real, bounded MCMC integration test** (`test_fit_end_to_end_*`):
  actually calls `fit()` + `fit_summary()` on tiny synthetic data, with
  small `draws`/`tune` (not the real-run defaults) so it stays bounded --
  its job is "does the whole pipeline run and land somewhere sensible",
  not statistical rigor. The fuller validation the Phase 1.2.2 plan calls
  for (prior/posterior-predictive checks, a real shrinkage demonstration,
  convergence diagnostics at real settings) lives in
  `scripts/validate_hierarchical_model.py` instead, run explicitly rather
  than on every `pytest` invocation -- real MCMC at a trustworthy sample
  size is minutes, not seconds, and does not belong in the default suite.
"""

import random

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from graphtalk import hierarchical_model as hm


def _synthetic_frame(control_rate, treatment_rate, n=20, seed=1, model="m"):
  rng = random.Random(seed)
  rows = []
  for i in range(n):
    rows.append({
        "model": model, "instance_id": f"task/{i}", "condition": "none",
        "exact": 1.0 if rng.random() < control_rate else 0.0,
    })
    rows.append({
        "model": model, "instance_id": f"task/{i}", "condition": "treat",
        "exact": 1.0 if rng.random() < treatment_rate else 0.0,
    })
  return pd.DataFrame(rows)


# --- build_model: structural, no sampling ------------------------------------


def test_build_model_raises_on_empty_frame():
  with pytest.raises(ValueError, match="empty"):
    hm.build_model(pd.DataFrame(columns=["model", "condition", "instance_id", "exact"]))


def test_build_model_raises_on_missing_control():
  frame = _synthetic_frame(0.5, 0.5, n=5)
  frame = frame[frame["condition"] != "none"]
  with pytest.raises(ValueError, match="none"):
    hm.build_model(frame)


def test_build_model_index_matches_data():
  frame = pd.concat([
      _synthetic_frame(0.5, 0.5, n=5, model="m1"),
      _synthetic_frame(0.5, 0.5, n=5, model="m2"),
  ], ignore_index=True)
  _, index = hm.build_model(frame)
  assert index["models"] == ["m1", "m2"]
  assert index["conditions"] == ["treat"]
  assert len(index["instances"]) == 5


def test_build_model_free_rvs_match_the_documented_model():
  frame = _synthetic_frame(0.5, 0.5, n=5)
  model, _ = hm.build_model(frame)
  names = {rv.name for rv in model.free_RVs}
  assert names == {
      "alpha_model", "beta_condition_raw", "gamma_raw", "u_instance_raw",
      "sigma_condition", "sigma_gamma", "sigma_instance",
  }


# --- fit_summary: exact extraction math, against a hand-built trace --------


def _synthetic_trace(
    beta_values, gamma_values, n_chains=4, n_draws=200, noise=0.05, seed=1,
):
  """A minimal `arviz.InferenceData` shaped like a real
  `hierarchical_model.fit()` trace, with independent per-chain-per-draw
  Gaussian noise around known `beta_values`/`gamma_values` -- real enough
  for `r_hat`/`ess_bulk` to be well-defined (unlike a zero-variance
  constant posterior, which makes `r_hat` a 0/0 `NaN`), but with a known
  ground truth `fit_summary`'s extracted means/HDI can be checked against.
  `beta_values`: one float per condition. `gamma_values`: one float per
  (model, condition), shape `(n_models, n_conditions)`.
  """
  rng = np.random.RandomState(seed)
  n_conditions = len(beta_values)
  n_models = len(gamma_values)
  beta = np.array(beta_values)[None, None, :] + rng.normal(
      0, noise, size=(n_chains, n_draws, n_conditions)
  )
  gamma = np.array(gamma_values)[None, None, :, :] + rng.normal(
      0, noise, size=(n_chains, n_draws, n_models, n_conditions)
  )
  posterior = xr.Dataset({
      "beta_condition": (["chain", "draw", "beta_condition_dim_0"], beta),
      "gamma": (["chain", "draw", "gamma_dim_0", "gamma_dim_1"], gamma),
  })
  sample_stats = xr.Dataset({
      "diverging": (["chain", "draw"], np.zeros((n_chains, n_draws), dtype=bool)),
  })
  return az.InferenceData(posterior=posterior, sample_stats=sample_stats)


def test_fit_summary_recovers_known_beta_and_gamma_means():
  trace = _synthetic_trace(
      beta_values=[0.5], gamma_values=[[0.2], [-0.3]], seed=2,
  )
  index = {"models": ["m1", "m2"], "conditions": ["treat"]}
  result = hm.fit_summary(trace, index).set_index("model")
  assert result.loc["m1", "beta_condition_mean"] == pytest.approx(0.5, abs=0.02)
  assert result.loc["m1", "total_mean"] == pytest.approx(0.7, abs=0.03)
  assert result.loc["m2", "total_mean"] == pytest.approx(0.2, abs=0.03)


def test_fit_summary_hdi_brackets_the_true_value():
  trace = _synthetic_trace(beta_values=[1.0], gamma_values=[[0.0]], seed=3)
  index = {"models": ["m1"], "conditions": ["treat"]}
  row = hm.fit_summary(trace, index).iloc[0]
  assert row["beta_condition_hdi_low"] < 1.0 < row["beta_condition_hdi_high"]
  assert row["total_hdi_low"] < 1.0 < row["total_hdi_high"]


def test_fit_summary_prob_positive_is_near_one_for_a_clear_positive_effect():
  trace = _synthetic_trace(beta_values=[2.0], gamma_values=[[0.0]], noise=0.1, seed=4)
  index = {"models": ["m1"], "conditions": ["treat"]}
  row = hm.fit_summary(trace, index).iloc[0]
  assert row["prob_positive"] > 0.99


def test_fit_summary_prob_positive_is_near_zero_for_a_clear_negative_effect():
  trace = _synthetic_trace(beta_values=[-2.0], gamma_values=[[0.0]], noise=0.1, seed=5)
  index = {"models": ["m1"], "conditions": ["treat"]}
  row = hm.fit_summary(trace, index).iloc[0]
  assert row["prob_positive"] < 0.01


def test_fit_summary_reports_r_hat_close_to_one_for_well_mixed_chains():
  """Independent per-chain noise around the same true value is exactly
  what `r_hat` should read as "converged" (close to 1.0)."""
  trace = _synthetic_trace(beta_values=[0.3], gamma_values=[[0.0]], seed=6)
  index = {"models": ["m1"], "conditions": ["treat"]}
  row = hm.fit_summary(trace, index).iloc[0]
  assert row["r_hat"] < 1.05


def test_fit_summary_surfaces_divergences():
  trace = _synthetic_trace(beta_values=[0.3], gamma_values=[[0.0]], seed=7)
  trace.sample_stats["diverging"][0, 0] = True
  trace.sample_stats["diverging"][1, 5] = True
  index = {"models": ["m1"], "conditions": ["treat"]}
  result = hm.fit_summary(trace, index)
  assert (result["n_divergences"] == 2).all()


def test_fit_summary_multiple_conditions_and_models_shape():
  trace = _synthetic_trace(
      beta_values=[0.5, -0.2, 0.0], gamma_values=[[0.1, 0.1, 0.1], [0.0, 0.0, 0.0]],
      seed=8,
  )
  index = {"models": ["m1", "m2"], "conditions": ["degree", "filler", "rwse"]}
  result = hm.fit_summary(trace, index)
  assert len(result) == 6  # 2 models x 3 conditions
  assert set(result["condition"]) == {"degree", "filler", "rwse"}
  assert set(result["model"]) == {"m1", "m2"}


# --- one real, bounded end-to-end integration test --------------------------


def test_fit_end_to_end_recovers_a_strong_effect_direction():
  """The only test in this file that actually runs `pm.sample`. Small
  `draws`/`tune` (not the real-run defaults `fit`'s signature uses) and a
  large, unambiguous injected effect -- this is a smoke test for "does the
  pipeline run end to end and land in the right place", not a statistical
  power or convergence-quality check (see `scripts/validate_hierarchical
  _model.py` for that)."""
  frame = pd.concat([
      _synthetic_frame(0.2, 0.9, n=15, seed=10, model="m1"),
      _synthetic_frame(0.2, 0.9, n=15, seed=11, model="m2"),
  ], ignore_index=True)
  trace, index = hm.fit(
      frame, draws=60, tune=60, chains=2, target_accept=0.9, seed=1234,
      progressbar=False,
  )
  result = hm.fit_summary(trace, index)
  assert (result["total_mean"] > 0).all()
  assert (result["prob_positive"] > 0.9).all()
