"""Tests for graphtalk/mixed_models.py -- the GEE cross-check.

Mirrors tests/test_significance.py's convention: small synthetic frames,
no real sweep data, deterministic seeds. Validates the three things
Phase 1.2.1's plan called for: a known-injected-effect recovery test, a
cross-check that the identity-link/independence-correlation choice actually
reduces to the simple mean difference (the same quantity
`graphtalk.significance.paired_permutation_test_clustered` reports as
`observed_diff`) for a saturated categorical design, and basic fit
diagnostics (`converged`, `n_groups`) behaving sanely.
"""

import random

import pandas as pd
import pytest

from graphtalk import mixed_models


def _synthetic_frame(control_rate, treatment_rate, n=40, seed=1, extra_conditions=()):
  """`n` graph instances, a `none` control and a `treat` condition (plus
  any `extra_conditions`, at the same rate as control -- true nulls, useful
  for multi-condition tests), each instance's outcome drawn independently
  per condition at the given Bernoulli rate. `instance_id` is the cluster
  key `fit_gee_one_model` groups on."""
  rng = random.Random(seed)
  rows = []
  for i in range(n):
    rows.append({
        "model": "m", "instance_id": f"task/{i}", "condition": "none",
        "exact": 1.0 if rng.random() < control_rate else 0.0,
    })
    rows.append({
        "model": "m", "instance_id": f"task/{i}", "condition": "treat",
        "exact": 1.0 if rng.random() < treatment_rate else 0.0,
    })
    for extra in extra_conditions:
      rows.append({
          "model": "m", "instance_id": f"task/{i}", "condition": extra,
          "exact": 1.0 if rng.random() < control_rate else 0.0,
      })
  return pd.DataFrame(rows)


# --- recovers a known injected effect ---------------------------------------


def test_recovers_a_known_positive_effect():
  frame = _synthetic_frame(control_rate=0.3, treatment_rate=0.8, n=60, seed=1)
  result = mixed_models.fit_gee_one_model(frame)
  row = result.iloc[0]
  assert row["condition"] == "treat"
  assert row["delta"] > 0.3
  assert row["p_value"] < 0.01
  assert bool(row["converged"])


def test_recovers_a_known_negative_effect():
  frame = _synthetic_frame(control_rate=0.8, treatment_rate=0.3, n=60, seed=2)
  result = mixed_models.fit_gee_one_model(frame)
  row = result.iloc[0]
  assert row["delta"] < -0.3
  assert row["p_value"] < 0.01


def test_true_null_is_not_significant():
  """Same rate in both conditions -- the fit should not manufacture a
  significant effect out of noise at a reasonable sample size."""
  frame = _synthetic_frame(control_rate=0.5, treatment_rate=0.5, n=60, seed=3)
  result = mixed_models.fit_gee_one_model(frame)
  row = result.iloc[0]
  assert abs(row["delta"]) < 0.2
  assert row["p_value"] > 0.05


# --- delta matches the simple mean difference for a saturated design -------


def test_delta_matches_naive_mean_difference():
  """For a saturated categorical predictor (one dummy per condition, no
  other covariates) with an identity link and independence working
  correlation, the point estimate must equal the simple difference in
  condition means -- the same quantity
  `paired_permutation_test_clustered` calls `observed_diff`. This is what
  makes the two methods' `delta` columns directly comparable; if this ever
  stops holding (e.g. from a link/family change), every downstream
  comparison in `scripts/check_significance_glmm.py` silently breaks."""
  frame = _synthetic_frame(control_rate=0.4, treatment_rate=0.65, n=50, seed=4)
  result = mixed_models.fit_gee_one_model(frame)
  naive = (
      frame[frame["condition"] == "treat"]["exact"].mean()
      - frame[frame["condition"] == "none"]["exact"].mean()
  )
  assert result.iloc[0]["delta"] == pytest.approx(naive, abs=1e-9)


def test_delta_matches_naive_mean_difference_multi_condition():
  """Same property, holding with more than one non-control condition in
  the same fit -- each condition's coefficient is still exactly its own
  simple mean difference against `none`, unaffected by the other
  conditions being in the model (a saturated categorical design has no
  shared slope to interfere)."""
  frame = _synthetic_frame(
      control_rate=0.4, treatment_rate=0.7, n=50, seed=5,
      extra_conditions=("degree", "filler"),
  )
  result = mixed_models.fit_gee_one_model(frame).set_index("condition")
  for condition in ("treat", "degree", "filler"):
    naive = (
        frame[frame["condition"] == condition]["exact"].mean()
        - frame[frame["condition"] == "none"]["exact"].mean()
    )
    assert result.loc[condition, "delta"] == pytest.approx(naive, abs=1e-9)


# --- fit diagnostics ----------------------------------------------------------


def test_n_groups_counts_distinct_instances_not_rows():
  frame = _synthetic_frame(control_rate=0.5, treatment_rate=0.5, n=25, seed=6)
  result = mixed_models.fit_gee_one_model(frame)
  assert result.iloc[0]["n_groups"] == 25
  assert result.iloc[0]["n_obs"] == 50


def test_converged_is_reported_true_on_well_behaved_data():
  frame = _synthetic_frame(control_rate=0.5, treatment_rate=0.6, n=60, seed=7)
  result = mixed_models.fit_gee_one_model(frame)
  assert bool(result.iloc[0]["converged"])


# --- input validation ---------------------------------------------------------


def test_empty_frame_raises():
  with pytest.raises(ValueError, match="empty"):
    mixed_models.fit_gee_one_model(pd.DataFrame(columns=["condition", "instance_id", "exact"]))


def test_missing_control_condition_raises():
  frame = _synthetic_frame(control_rate=0.5, treatment_rate=0.5, n=10, seed=8)
  frame = frame[frame["condition"] != "none"]
  with pytest.raises(ValueError, match="none"):
    mixed_models.fit_gee_one_model(frame)


# --- fit_gee_all_models: per-model iteration and scope ------------------------


def test_fit_gee_all_models_iterates_per_model():
  a = _synthetic_frame(control_rate=0.3, treatment_rate=0.8, n=30, seed=9)
  a["model"] = "model_a"
  a["is_think"] = False
  a["failure_type"] = "correct"
  b = _synthetic_frame(control_rate=0.5, treatment_rate=0.5, n=30, seed=10)
  b["model"] = "model_b"
  b["is_think"] = False
  b["failure_type"] = "correct"
  frame = pd.concat([a, b], ignore_index=True)
  result = mixed_models.fit_gee_all_models(frame)
  assert set(result["model"].unique()) == {"model_a", "model_b"}
  assert len(result) == 2  # one non-control condition ("treat") per model


def test_fit_gee_all_models_excludes_thinking_arm_and_non_terminating():
  frame = _synthetic_frame(control_rate=0.4, treatment_rate=0.9, n=30, seed=11)
  frame["model"] = "m"
  frame["is_think"] = False
  frame["failure_type"] = "correct"
  # A thinking-arm copy and a non-terminating copy, both with an opposite
  # (near-zero) effect -- if either leaked into the fit, delta would move
  # sharply toward zero.
  think = frame.copy()
  think["is_think"] = True
  think["exact"] = 0.5
  non_terminating = frame.copy()
  non_terminating["instance_id"] = non_terminating["instance_id"] + "-nt"
  non_terminating["failure_type"] = "non_terminating"
  non_terminating["exact"] = 0.5
  combined = pd.concat([frame, think, non_terminating], ignore_index=True)

  clean_result = mixed_models.fit_gee_one_model(frame)
  combined_result = mixed_models.fit_gee_all_models(combined)
  assert combined_result.iloc[0]["delta"] == pytest.approx(
      clean_result.iloc[0]["delta"], abs=1e-9
  )
