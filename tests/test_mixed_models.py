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


def test_n_groups_counts_distinct_graphs_not_rows():
  frame = _synthetic_frame(control_rate=0.5, treatment_rate=0.5, n=25, seed=6)
  result = mixed_models.fit_gee_one_model(frame)
  assert result.iloc[0]["n_groups"] == 25
  assert result.iloc[0]["n_obs"] == 50


def test_the_six_tasks_on_one_graph_are_one_group():
  """`node_count/7` and `edge_count/7` are the same graph asked two
  questions, so they belong in one GEE group. Grouping on the whole
  `instance_id` split them, which both misstates the correlation structure
  and puts this module at a different cluster granularity from
  `check_significance.py` -- the one thing it exists to be comparable to.
  """
  rng = random.Random(23)
  rows = []
  for task in ("node_count", "edge_count", "cycle_check"):
    for i in range(10):
      for condition in ("none", "treat"):
        rows.append({"model": "m", "instance_id": f"{task}/{i}",
                     "condition": condition,
                     "exact": 1.0 if rng.random() < 0.5 else 0.0})
  result = mixed_models.fit_gee_one_model(pd.DataFrame(rows))
  # 30 distinct instance_ids, but only 10 distinct graphs.
  assert result.iloc[0]["n_obs"] == 60
  assert result.iloc[0]["n_groups"] == 10


def _within_graph_correlated_frame(n_graphs=40, n_tasks=6, seed=17):
  """Rows with *genuine* within-graph correlation: each graph is either
  easy (both conditions almost always right) or hard (almost always
  wrong), and every task on that graph inherits it.

  Every other fixture in this file draws each row independently, so there
  is no correlation for the grouping to account for and the module's tests
  pass whether or not it accounts for any. This is the fixture that can
  tell the difference.
  """
  rng = random.Random(seed)
  rows = []
  for graph in range(n_graphs):
    easy = rng.random() < 0.5
    rate = 0.95 if easy else 0.05
    for task in range(n_tasks):
      for condition in ("none", "treat"):
        rows.append({
            "model": "m", "instance_id": f"task{task}/{graph}",
            "condition": condition,
            "exact": 1.0 if rng.random() < rate else 0.0,
        })
  return pd.DataFrame(rows)


def test_grouping_actually_changes_the_standard_error_on_correlated_data():
  """The regression guard the audit asked for. Deleting the grouping used
  to leave all 12 tests green, because no fixture had any correlation to
  lose. Here it must matter: with rows sharing a graph moving together, a
  fit that treats each row as its own cluster reports a materially
  different (over-confident) sandwich standard error than one that
  clusters correctly. If this test ever passes with the grouping removed,
  the cross-check has stopped cross-checking.
  """
  frame = _within_graph_correlated_frame()
  clustered = mixed_models.fit_gee_one_model(frame).iloc[0]

  ungrouped = frame.assign(instance_id=[f"task/{i}" for i in range(len(frame))])
  unclustered = mixed_models.fit_gee_one_model(ungrouped).iloc[0]

  assert clustered["n_groups"] == 40
  assert unclustered["n_groups"] == len(frame)
  # Same point estimate either way (saturated design), different variance.
  assert clustered["delta"] == pytest.approx(unclustered["delta"], abs=1e-9)
  assert clustered["std_err"] != pytest.approx(unclustered["std_err"], rel=0.05)


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


def test_fit_gee_all_models_excludes_thinking_arm():
  frame = _synthetic_frame(control_rate=0.4, treatment_rate=0.9, n=30, seed=11)
  frame["model"] = "m"
  frame["is_think"] = False
  frame["failure_type"] = "correct"
  # A thinking-arm copy with an opposite (near-zero) effect -- if it leaked
  # into the fit, delta would move sharply toward zero. The thinking arm is
  # a genuinely different question (`check_significance.py`'s own
  # non-termination-rate test), unaffected by the non-terminating-row
  # refactor -- unlike the case below, this exclusion still applies.
  think = frame.copy()
  think["is_think"] = True
  think["exact"] = 0.5
  combined = pd.concat([frame, think], ignore_index=True)

  clean_result = mixed_models.fit_gee_one_model(frame)
  combined_result = mixed_models.fit_gee_all_models(combined)
  assert combined_result.iloc[0]["delta"] == pytest.approx(
      clean_result.iloc[0]["delta"], abs=1e-9
  )


def test_fit_gee_all_models_includes_non_terminating():
  """Non-terminating rows are no longer excluded from this fit -- see
  `_main_sweep_scope`'s docstring. Real data reaches this module through
  `graphtalk.analysis.build_frame`, which already forces a non-terminating
  row's `exact` to 0.0 before it gets here; this test bypasses build_frame
  (a hand-built synthetic frame) specifically to isolate *this module's own
  scoping* from that upstream forcing -- a non-terminating row with a
  distinct `exact` value must still visibly move the fit, proving it was
  not silently dropped the way it used to be.
  """
  frame = _synthetic_frame(control_rate=0.4, treatment_rate=0.9, n=30, seed=11)
  frame["model"] = "m"
  frame["is_think"] = False
  frame["failure_type"] = "correct"
  non_terminating = frame.copy()
  non_terminating["instance_id"] = non_terminating["instance_id"] + "-nt"
  non_terminating["failure_type"] = "non_terminating"
  non_terminating["exact"] = 0.5
  combined = pd.concat([frame, non_terminating], ignore_index=True)

  clean_result = mixed_models.fit_gee_one_model(frame)
  combined_result = mixed_models.fit_gee_all_models(combined)
  assert combined_result.iloc[0]["delta"] != pytest.approx(
      clean_result.iloc[0]["delta"], abs=1e-9
  )
  # And it's not a fluke of exclusion elsewhere -- n_obs must actually
  # reflect the doubled row count.
  assert combined_result.iloc[0]["n_obs"] == 2 * clean_result.iloc[0]["n_obs"]
