"""Tests for graphtalk/significance.py and its guards.

No test file existed for this module before -- these are the checks that
justify trusting `scripts/check_significance.py`'s numbers, added alongside
the fixes they validate: exact agreement with `scoring.mcnemar` (the
project's already-trusted test, on the same data), and a synthetic
correlated dataset demonstrating why the *_clustered functions exist at all
-- the naive functions read as confidently significant on data that's really
just a handful of independent instances duplicated, and the clustered ones
correctly don't.
"""

import random
from types import SimpleNamespace

import pandas as pd
import pytest

from graphtalk import analysis
from graphtalk import scoring
from graphtalk import significance
from scripts import check_significance as cs


def _default_args(**overrides):
  """The `args` namespace `_report` needs, with every field it reads --
  kept in one place so a new CLI flag only has to be added here once rather
  than in every direct `_report` call in this file."""
  base = dict(
      n_perm=200, n_boot=200, alpha=0.05, q=0.05, seed=1,
      low_power_threshold=0.15, near_ceiling_threshold=0.95,
      mde=False, mde_power_target=0.8, mde_replicates=50, mde_n_perm=100,
  )
  base.update(overrides)
  return SimpleNamespace(**base)


# --- benjamini_hochberg ------------------------------------------------------


def test_benjamini_hochberg_textbook_example():
  # Sorted ascending: 0.005(1) 0.01(2) 0.03(3) 0.04(4) 0.07(5) 0.09(6) 0.15(7)
  # 0.20(8), thresholds rank*0.05/8: only ranks 1-2 satisfy p <= threshold,
  # and the largest such rank is 2 -- reject ranks 1-2, i.e. p in {0.005, 0.01}.
  p_values = [0.01, 0.04, 0.03, 0.005, 0.20, 0.15, 0.09, 0.07]
  reject = significance.benjamini_hochberg(p_values, q=0.05)
  assert reject == [True, False, False, True, False, False, False, False]


def test_benjamini_hochberg_empty_input():
  assert significance.benjamini_hochberg([]) == []


def test_benjamini_hochberg_nothing_survives():
  assert significance.benjamini_hochberg([0.5, 0.6, 0.9], q=0.05) == [False, False, False]


# --- paired_permutation_test: edge cases ------------------------------------


def test_all_concordant_pairs_give_p_one():
  control = treatment = [1, 0, 1, 1, 0, 0, 1]
  result = significance.paired_permutation_test(control, treatment)
  assert result["observed_diff"] == 0.0
  assert result["p_value"] == 1.0


def test_all_discordant_one_direction_gives_a_tiny_p_value():
  control = [0] * 12
  treatment = [1] * 12
  result = significance.paired_permutation_test(control, treatment, n_perm=20_000, seed=1)
  assert result["p_value"] < 0.001


def test_mismatched_lengths_raise():
  with pytest.raises(ValueError, match="equal lengths"):
    significance.paired_permutation_test([1, 0], [1, 0, 1])


# --- cross-validation against the project's already-trusted exact test -----


def test_clustered_exact_path_matches_mcnemar_on_a_small_p_value():
  """`paired_permutation_test_clustered` with one pair per cluster and few
  enough clusters to enumerate exactly must agree bit-for-bit with
  `scoring.mcnemar` -- both answer the same combinatorial question about a
  paired binary outcome, so this is a direct regression tie between the new
  pooled test and the one this project already trusts.
  """
  control =   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]
  treatment = [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]
  mc = scoring.mcnemar(control, treatment)
  cluster_ids = list(range(len(control)))  # unique -> exact enumeration (n=14)
  perm = significance.paired_permutation_test_clustered(control, treatment, cluster_ids)
  assert perm["p_value"] == pytest.approx(mc["p_value"], abs=1e-9)


def test_clustered_reduces_to_unclustered_on_unique_cluster_ids():
  """Above the exact-enumeration threshold, so both paths use Monte Carlo --
  with the same seed and data, unique cluster_ids must give identical
  results to the unclustered function (each cluster holds exactly one pair).
  """
  control = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0] * 5    # n=50
  treatment = [1, 1, 0, 1, 0, 1, 1, 0, 0, 1] * 5
  cluster_ids = list(range(len(control)))
  a = significance.paired_permutation_test(control, treatment, n_perm=5000, seed=42)
  b = significance.paired_permutation_test_clustered(
      control, treatment, cluster_ids, n_perm=5000, seed=42
  )
  assert a["observed_diff"] == b["observed_diff"]
  assert a["p_value"] == b["p_value"]

  ci_a = significance.cluster_bootstrap_ci(control, treatment, n_boot=5000, seed=42)
  ci_b = significance.cluster_bootstrap_ci_clustered(
      control, treatment, cluster_ids, n_boot=5000, seed=42
  )
  assert ci_a["ci_low"] == ci_b["ci_low"]
  assert ci_a["ci_high"] == ci_b["ci_high"]


# --- the actual bug this session found: naive pooling is anti-conservative -


def _correlated_dataset(n_instances=30, replicas=4, seed=1):
  """`n_instances` independent (control, treatment) draws, each duplicated
  `replicas` times under the same cluster_id -- simulating e.g. 4 models
  that agree perfectly on each graph. A real, if modest, effect is present
  (treatment ~5 points better on average).
  """
  import random
  rng = random.Random(seed)
  control, treatment, cluster_ids = [], [], []
  for i in range(n_instances):
    c = 1 if rng.random() < 0.60 else 0
    t = 1 if rng.random() < 0.65 else 0
    for _ in range(replicas):
      control.append(c)
      treatment.append(t)
      cluster_ids.append(i)
  return control, treatment, cluster_ids


def test_clustering_is_more_conservative_than_naive_pooling():
  """The load-bearing test for this whole fix: naive pooling on data that is
  really only `n_instances` independent draws, duplicated `replicas` times,
  must not be allowed to look more significant than it is. Confirmed
  concretely during planning: the naive test reported p < 0.001 on exactly
  this shape of data (30 real clusters read as 120 independent pairs);
  the clustered test correctly reports the true, much larger p-value.
  """
  control, treatment, cluster_ids = _correlated_dataset()
  naive = significance.paired_permutation_test(control, treatment, n_perm=20_000, seed=7)
  clustered = significance.paired_permutation_test_clustered(
      control, treatment, cluster_ids, n_perm=20_000, seed=7
  )
  assert clustered["n_clusters"] == 30
  assert naive["n_pairs"] == 120
  assert naive["observed_diff"] == clustered["observed_diff"]
  assert clustered["p_value"] > naive["p_value"]


def test_clustered_bootstrap_ci_is_wider_than_naive():
  control, treatment, cluster_ids = _correlated_dataset()
  naive = significance.cluster_bootstrap_ci(control, treatment, n_boot=20_000, seed=7)
  clustered = significance.cluster_bootstrap_ci_clustered(
      control, treatment, cluster_ids, n_boot=20_000, seed=7
  )
  naive_width = naive["ci_high"] - naive["ci_low"]
  clustered_width = clustered["ci_high"] - clustered["ci_low"]
  assert clustered_width > naive_width
  assert clustered["point_estimate"] == naive["point_estimate"]


# --- assert_unique_pairing_key ----------------------------------------------


def test_assert_unique_pairing_key_passes_on_a_clean_frame():
  frame = pd.DataFrame({
      "model": ["gemma4-12b", "gemma4-12b"],
      "instance_id": ["node_count/0", "node_count/1"],
      "style": ["zero_shot", "zero_shot"],
      "node_naming": ["integer", "integer"],
  })
  analysis.assert_unique_pairing_key(frame, ["model", "instance_id", "style", "node_naming"])


def test_assert_unique_pairing_key_raises_on_a_duplicate():
  """The exact silent-corruption scenario this guards against: without it,
  `pd.concat`'s join in `_paired_values` cross-joins the duplicated key
  instead of erroring."""
  frame = pd.DataFrame({
      "model": ["gemma4-12b", "gemma4-12b"],
      "instance_id": ["node_count/0", "node_count/0"],
      "style": ["zero_shot", "zero_shot"],
      "node_naming": ["integer", "integer"],
  })
  with pytest.raises(ValueError, match="duplicate"):
    analysis.assert_unique_pairing_key(
        frame, ["model", "instance_id", "style", "node_naming"]
    )


def test_assert_unique_pairing_key_needs_condition_for_a_whole_frame():
  """Regression: a real `sweep_frame.csv`-shaped frame has one row per
  (model, instance_id, style, node_naming) *per condition* -- seven of
  them, by design. Checking uniqueness on `check_significance.py`'s
  *pairing* key (which omits `condition`, correct only after
  `_paired_values` has already filtered to one condition) against the
  *whole*, all-conditions frame would flag every single row as a duplicate
  of its six sibling conditions. `main()` must check the full row identity
  (`condition` included), not the pairing key, against the unfiltered frame.
  """
  frame = pd.DataFrame({
      "model": ["gemma4-12b"] * 3,
      "instance_id": ["node_count/0"] * 3,
      "style": ["zero_shot"] * 3,
      "node_naming": ["integer"] * 3,
      "condition": ["none", "degree", "clustering"],
  })
  # The pairing key alone (no condition) looks like 3 duplicates of the same row...
  with pytest.raises(ValueError, match="duplicate"):
    analysis.assert_unique_pairing_key(frame, ["model", "instance_id", "style", "node_naming"])
  # ...but the full row identity, condition included, is genuinely unique.
  analysis.assert_unique_pairing_key(
      frame, ["model", "instance_id", "condition", "style", "node_naming"]
  )


# --- scripts/check_significance.py: the non_terminating exclusion ----------


def test_count_excluded_non_terminating():
  raw = pd.DataFrame({
      "condition": ["none", "degree", "degree"],
      "failure_type": ["correct", "non_terminating", "correct"],
  })
  assert cs._count_excluded_non_terminating(raw, "degree") == 1
  assert cs._count_excluded_non_terminating(raw, "clustering") == 0


def test_non_terminating_rows_excluded_before_pairing():
  """A non_terminating row that happens to score exact=1.0 must not be
  paired in -- confirming the exclusion `main()` applies before
  `_paired_values` runs actually removes the row, not just makes it visible
  in the count above.
  """
  raw = pd.DataFrame({
      "model": ["gemma4-12b"] * 4,
      "instance_id": ["node_count/0", "node_count/0", "node_count/1", "node_count/1"],
      "style": ["zero_shot"] * 4,
      "node_naming": ["integer"] * 4,
      "condition": ["none", "degree", "none", "degree"],
      "failure_type": ["correct", "non_terminating", "correct", "correct"],
      "exact": [1.0, 1.0, 0.0, 1.0],
  })
  filtered = raw[raw["failure_type"] != "non_terminating"]
  control, treatment, cluster_ids = cs._paired_values(filtered, "degree", "exact")
  # node_count/0's degree row was non_terminating and dropped -- its control
  # row (none) now has no treatment partner, so the pair disappears entirely
  # rather than pairing a real control against a truncated treatment.
  assert cluster_ids == [("gemma4-12b", "node_count/1")]
  assert control == [0.0]
  assert treatment == [1.0]


# --- Fix 2: cluster id carries model, not just instance_id -----------------


def test_cluster_id_carries_model_preventing_cross_model_merge():
  """The actual bug fixed: two different models sharing an `instance_id`
  must produce two clusters, not one, when their rows are pooled together
  (e.g. a "pooled across all models" call). Before this fix, `cluster_ids`
  was bare `instance_id`, so this pair would have collapsed into a single
  cluster and understated the true variance.
  """
  frame = pd.DataFrame({
      "model": ["gemma4-12b", "gemma4-12b", "qwen3-8b", "qwen3-8b"],
      "instance_id": ["node_count/0"] * 4,
      "style": ["zero_shot"] * 4,
      "node_naming": ["integer"] * 4,
      "condition": ["none", "degree", "none", "degree"],
      "exact": [1.0, 0.0, 1.0, 1.0],
  })
  control, treatment, cluster_ids = cs._paired_values(frame, "degree", "exact")
  assert len(control) == 2
  assert cluster_ids == [
      ("gemma4-12b", "node_count/0"), ("qwen3-8b", "node_count/0"),
  ]
  assert len(set(cluster_ids)) == 2


# --- Fix 1: the best_case/worst_case bracket --------------------------------


def test_bracket_frame_overrides_only_non_terminating_rows():
  raw = pd.DataFrame({
      "failure_type": ["correct", "non_terminating", "wrong"],
      "exact": [1.0, 0.3, 0.0],
  })
  best = cs._bracket_frame(raw, 1.0)
  worst = cs._bracket_frame(raw, 0.0)
  assert best["exact"].tolist() == [1.0, 1.0, 0.0]
  assert worst["exact"].tolist() == [1.0, 0.0, 0.0]
  # The input frame itself is untouched -- `_bracket_frame` must copy, not
  # mutate in place, since `main()` builds both bounds from the same
  # `main_sweep_raw`.
  assert raw["exact"].tolist() == [1.0, 0.3, 0.0]


def test_report_n_instances_missing_is_computed_from_data_not_hardcoded():
  """Regression: `n_instances_missing` must be read from the frame's own
  instance count, not a hardcoded 180 -- checked on a fixture with a
  deliberately non-default instance count (3), one of which drops out of
  the `degree` pairing entirely.
  """
  raw = pd.DataFrame({
      "model": ["gemma4-12b"] * 5,
      "instance_id": ["a", "a", "b", "b", "c"],
      "style": ["zero_shot"] * 5,
      "node_naming": ["integer"] * 5,
      "condition": ["none", "degree", "none", "degree", "none"],
      "failure_type": ["correct"] * 5,
      "exact": [1.0, 1.0, 0.0, 1.0, 1.0],
      # `_report` assumes this column exists -- `main()` guarantees it via
      # its predates-the-column backward-compat fallback before any
      # `_report` call, so a direct-call fixture must supply it too.
      "looped_on_correct_answer": [None] * 5,
  })
  records = []
  args = _default_args()
  cs._report(raw, raw, "exact", "gemma4-12b", args, "main_sweep", records,
             bound="excluded")
  row = next(r for r in records if r["condition"] == "degree")
  assert row["n_clusters"] == 2       # "c" has no `degree` row to pair with
  assert row["n_instances_missing"] == 1   # 3 total instances - 2 clusters


def test_report_n_instances_missing_uses_model_instance_pairs_for_pooled_data():
  """Regression for the bug caught running this against the real sweep:
  with Fix 2's composite `(model, instance_id)` clusters, a pooled-across-
  models call can have *more* clusters than there are distinct
  `instance_id` values (up to one per model sharing that graph number).
  `n_instances_missing` must be baselined against distinct
  `(model, instance_id)` pairs, not bare `instance_id`, or it goes
  negative.
  """
  raw = pd.DataFrame({
      "model": ["gemma4-12b", "gemma4-12b", "qwen3-8b", "qwen3-8b"],
      "instance_id": ["a", "a", "a", "a"],
      "style": ["zero_shot"] * 4,
      "node_naming": ["integer"] * 4,
      "condition": ["none", "degree", "none", "degree"],
      "failure_type": ["correct"] * 4,
      "exact": [1.0, 1.0, 0.0, 1.0],
      "looped_on_correct_answer": [None] * 4,
  })
  records = []
  args = _default_args()
  cs._report(raw, raw, "exact", "pooled across all models", args,
             "main_sweep", records, bound="excluded")
  row = next(r for r in records if r["condition"] == "degree")
  assert row["n_clusters"] == 2       # one per model, same instance_id
  assert row["n_instances_missing"] == 0   # both (model, instance_id) pairs present


# --- Fix 4: the "looped on the correct answer" diagnostic -------------------


def test_count_looped_on_correct_answer():
  raw = pd.DataFrame({
      "condition": ["none", "degree", "degree"],
      "looped_on_correct_answer": [False, True, False],
  })
  assert cs._count_looped_on_correct_answer(raw, "degree") == 1
  assert cs._count_looped_on_correct_answer(raw, "clustering") == 0


# --- Fix 3: the whole-table BH pass -----------------------------------------


def test_apply_global_bh_excludes_bracket_and_pooled_rows():
  records = [
      {"group": "gemma4-12b", "bound": "excluded", "p_value": 0.001, "is_derived_condition": False},
      {"group": "gemma4-12b", "bound": "best_case", "p_value": 0.001, "is_derived_condition": False},
      {"group": "gemma4-12b", "bound": "worst_case", "p_value": 0.001, "is_derived_condition": False},
      {"group": "pooled across all models", "bound": "excluded", "p_value": 0.001, "is_derived_condition": False},
      {"group": "gemma4-e4b", "bound": "not_applicable", "p_value": 0.9, "is_derived_condition": False},
  ]
  cs._apply_global_bh(records, q=0.05)
  by_key = {(r["group"], r["bound"]): r["bh_significant_global"] for r in records}
  assert by_key[("gemma4-12b", "excluded")] is True
  assert by_key[("gemma4-e4b", "not_applicable")] is False
  # Bracket and pooled rows are excluded from the family entirely -- `None`
  # ("not tested"), not `False` ("tested, not significant").
  assert by_key[("gemma4-12b", "best_case")] is None
  assert by_key[("gemma4-12b", "worst_case")] is None
  assert by_key[("pooled across all models", "excluded")] is None


def test_apply_global_bh_excludes_derived_condition_rows():
  """`all` (the union of degree/clustering/rwse) must not be pooled into
  the same multiple-comparison family as the independent conditions --
  it's mechanically correlated with them, not a fifth independent test."""
  records = [
      {"group": "gemma4-12b", "bound": "excluded", "p_value": 0.001, "is_derived_condition": False},
      {"group": "gemma4-12b", "bound": "excluded", "p_value": 0.001, "is_derived_condition": True},
  ]
  cs._apply_global_bh(records, q=0.05)
  assert records[0]["bh_significant_global"] is True
  assert records[1]["bh_significant_global"] is None


def test_global_bh_can_be_stricter_than_per_family_bh():
  """Not a universal law -- BH's rejection region depends on the whole
  family, so it is not guaranteed in general that adding more tests can
  only remove significant results, never add them. But on report-shaped
  data (mostly-null p-values from many models/conditions), embedding a
  pair that's significant within its own small family into the much larger
  whole-table family typically raises the bar enough to flip it. Shown
  concretely here rather than asserted as a theorem.
  """
  small_family = significance.benjamini_hochberg([0.01, 0.02], q=0.05)
  assert small_family == [True, True]

  p_values = [0.01, 0.02] + [0.5 + 0.02 * i for i in range(18)]
  records = [
      {"group": f"model_{i}", "bound": "excluded", "p_value": p, "is_derived_condition": False}
      for i, p in enumerate(p_values)
  ]
  cs._apply_global_bh(records, q=0.05)
  assert records[0]["bh_significant_global"] is False
  assert records[1]["bh_significant_global"] is False


# --- near_ceiling ------------------------------------------------------------


def _near_ceiling_frame(control_rate: float, n: int = 40):
  """`n` paired rows where the CONTROL condition's mean `exact` is exactly
  `control_rate` (alternating 1s/0s to hit it precisely) and the TREATMENT
  condition is unrelated noise -- `near_ceiling` only reads the control
  side, so treatment's exact shape doesn't matter here."""
  n_ones = round(control_rate * n)
  control_vals = [1.0] * n_ones + [0.0] * (n - n_ones)
  return pd.DataFrame({
      "model": ["gemma4-12b"] * (2 * n),
      "instance_id": [f"node_count/{i}" for i in range(n)] * 2,
      "style": ["zero_shot"] * (2 * n),
      "node_naming": ["integer"] * (2 * n),
      "condition": ["none"] * n + ["degree"] * n,
      "failure_type": ["correct"] * (2 * n),
      "exact": control_vals + [0.5] * n,
      "looped_on_correct_answer": [None] * (2 * n),
  })


def test_near_ceiling_true_above_threshold():
  raw = _near_ceiling_frame(0.98)
  records = []
  cs._report(raw, raw, "exact", "gemma4-12b", _default_args(),
             "main_sweep", records, bound="excluded")
  assert all(r["near_ceiling"] is True for r in records)


def test_near_ceiling_true_below_complement_a_floor_case():
  raw = _near_ceiling_frame(0.02)
  records = []
  cs._report(raw, raw, "exact", "gemma4-12b", _default_args(),
             "main_sweep", records, bound="excluded")
  assert all(r["near_ceiling"] is True for r in records)


def test_near_ceiling_false_mid_range():
  raw = _near_ceiling_frame(0.5)
  records = []
  cs._report(raw, raw, "exact", "gemma4-12b", _default_args(),
             "main_sweep", records, bound="excluded")
  assert all(r["near_ceiling"] is False for r in records)


def test_near_ceiling_not_populated_for_bracket_bounds():
  """A best_case/worst_case bracket forces non-terminating rows to fixed
  extremes -- reading `near_ceiling` off that would be measuring the
  bracket's own construction, not the model."""
  raw = _near_ceiling_frame(0.98)
  records = []
  cs._report(raw, raw, "exact", "gemma4-12b", _default_args(),
             "main_sweep", records, bound="best_case")
  assert all(r["near_ceiling"] is None for r in records)


# --- headroom -----------------------------------------------------------------


def test_headroom_matches_hand_computed_value():
  # n=50 so 0.98 * 50 = 49 lands exactly (no rounding) -- headroom = 0.02.
  raw = _near_ceiling_frame(0.98, n=50)
  records = []
  cs._report(raw, raw, "exact", "gemma4-12b", _default_args(),
             "main_sweep", records, bound="excluded")
  assert all(r["headroom"] == pytest.approx(0.02) for r in records)


def test_headroom_matches_hand_computed_value_below_half():
  raw = _near_ceiling_frame(0.3)
  records = []
  cs._report(raw, raw, "exact", "gemma4-12b", _default_args(),
             "main_sweep", records, bound="excluded")
  # min(0.3, 1 - 0.3) = 0.3 -- below the midpoint, headroom is the control
  # rate itself, not its complement.
  assert all(r["headroom"] == pytest.approx(0.3) for r in records)


def test_headroom_not_populated_for_bracket_bounds():
  """Same populated scope as `near_ceiling` -- a best_case/worst_case
  bracket forces non-terminating rows to fixed extremes, so headroom read
  off it would measure the bracket's own construction, not the model."""
  raw = _near_ceiling_frame(0.98)
  records = []
  cs._report(raw, raw, "exact", "gemma4-12b", _default_args(),
             "main_sweep", records, bound="best_case")
  assert all(r["headroom"] is None for r in records)


# --- `all` is a derived condition, corrected as its own family -------------


def _multi_condition_frame(n: int = 30):
  """One `instance_id` set repeated under `none` plus every non-control
  primer condition (the five independent ones and `all`) -- for testing
  that `all`, the union of degree/clustering/rwse, is excluded from the
  independent conditions' BH-correction family."""
  conditions = ["none", "degree", "clustering", "rwse", "components", "filler", "all"]
  cols = {"model": [], "instance_id": [], "style": [], "node_naming": [],
          "condition": [], "failure_type": [], "exact": [],
          "looped_on_correct_answer": []}
  for condition in conditions:
    for i in range(n):
      cols["model"].append("gemma4-12b")
      cols["instance_id"].append(f"node_count/{i}")
      cols["style"].append("zero_shot")
      cols["node_naming"].append("integer")
      cols["condition"].append(condition)
      cols["failure_type"].append("correct")
      cols["exact"].append(1.0 if i % 2 == 0 else 0.0)
      cols["looped_on_correct_answer"].append(None)
  return pd.DataFrame(cols)


def test_all_condition_is_marked_derived_others_are_not():
  raw = _multi_condition_frame()
  records = []
  cs._report(raw, raw, "exact", "gemma4-12b", _default_args(),
             "main_sweep", records, bound="excluded")
  by_condition = {r["condition"]: r for r in records}
  assert by_condition["all"]["is_derived_condition"] is True
  assert all(
      by_condition[c]["is_derived_condition"] is False
      for c in ("degree", "clustering", "rwse", "components", "filler")
  )


def test_all_condition_gets_its_own_bh_family():
  """The five independent conditions share one `bh_family`; `all` must get
  a distinct one, so it is corrected as a single-hypothesis family instead
  of inflating (or diluting) the real one."""
  raw = _multi_condition_frame()
  records = []
  cs._report(raw, raw, "exact", "gemma4-12b", _default_args(),
             "main_sweep", records, bound="excluded")
  by_condition = {r["condition"]: r for r in records}
  independent_families = {
      by_condition[c]["bh_family"]
      for c in ("degree", "clustering", "rwse", "components", "filler")
  }
  assert len(independent_families) == 1
  assert by_condition["all"]["bh_family"] != independent_families.pop()
  assert by_condition["all"]["bh_family"].endswith("/derived")


# --- task_delta_min/max -------------------------------------------------------


def test_task_delta_range_surfaces_heterogeneity_a_pooled_delta_hides():
  """Two tasks with opposite-sign effects of equal size pool to ~0 --
  `task_delta_min`/`max` must still show the real spread."""
  raw = pd.DataFrame({
      "model": ["gemma4-12b"] * 8,
      "instance_id": (
          ["node_count/0", "node_count/1"] * 2
          + ["edge_count/0", "edge_count/1"] * 2
      ),
      "style": ["zero_shot"] * 8,
      "node_naming": ["integer"] * 8,
      "condition": ["none", "none", "degree", "degree"] * 2,
      "failure_type": ["correct"] * 8,
      # node_count: control=0, treatment=1 (helps). edge_count: control=1,
      # treatment=0 (hurts). Pooled delta = (1 + 1 - 1 - 1) / 4 = 0.
      "exact": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
      "looped_on_correct_answer": [None] * 8,
  })
  records = []
  cs._report(raw, raw, "exact", "gemma4-12b", _default_args(),
             "main_sweep", records, bound="excluded")
  row = records[0]
  assert row["delta"] == pytest.approx(0.0)
  assert row["task_delta_min"] == pytest.approx(-1.0)
  assert row["task_delta_max"] == pytest.approx(1.0)


# --- MDE wiring ----------------------------------------------------------


def test_mde_computed_only_for_non_significant_excluded_bound_rows():
  """`--mde` (here `args.mde=True`) must only run the MDE search for rows
  that came back not significant, on a primary bound -- significant rows
  have nothing to explain, and a bracket bound isn't primary. Checks that
  the search *ran* (`mde_power_target` populated) rather than that it
  *converged* to a finite delta -- with only 10 clusters and the fast,
  low-replicate settings used here, the search can legitimately report
  "MDE exceeds 1.0" (`mde_delta=None`) instead of a number, which is
  correct behavior for a genuinely underpowered fixture, not a wiring bug.
  """
  raw = pd.DataFrame({
      "model": ["gemma4-12b"] * 20,
      "instance_id": [f"node_count/{i}" for i in range(10)] * 2,
      "style": ["zero_shot"] * 20,
      "node_naming": ["integer"] * 20,
      "condition": ["none"] * 10 + ["degree"] * 10,
      "failure_type": ["correct"] * 20,
      # No real effect: control and treatment identical -> not significant.
      "exact": [1.0, 0.0] * 5 + [1.0, 0.0] * 5,
      "looped_on_correct_answer": [None] * 20,
  })
  records = []
  args = _default_args(mde=True, mde_replicates=20, mde_n_perm=50)
  cs._report(raw, raw, "exact", "gemma4-12b", args, "main_sweep", records,
             bound="excluded")
  row = records[0]
  assert row["bh_significant"] is False
  assert row["mde_power_target"] == args.mde_power_target
  assert row["mde_delta"] is None or 0.0 < row["mde_delta"] <= 1.0


def test_mde_not_computed_when_flag_is_off():
  raw = pd.DataFrame({
      "model": ["gemma4-12b"] * 20,
      "instance_id": [f"node_count/{i}" for i in range(10)] * 2,
      "style": ["zero_shot"] * 20,
      "node_naming": ["integer"] * 20,
      "condition": ["none"] * 10 + ["degree"] * 10,
      "failure_type": ["correct"] * 20,
      "exact": [1.0, 0.0] * 5 + [1.0, 0.0] * 5,
      "looped_on_correct_answer": [None] * 20,
  })
  records = []
  cs._report(raw, raw, "exact", "gemma4-12b", _default_args(mde=False),
             "main_sweep", records, bound="excluded")
  assert records[0]["mde_delta"] is None


# --- graphtalk.significance: _resample_clusters / minimum_detectable_effect --


def test_resample_clusters_preserves_pair_structure():
  """Each drawn cluster's items travel together (a 2-item cluster is never
  split across two draws), and `draw_ids` gives each of the `m` draws its
  own label even if the same original cluster is drawn twice -- two draws
  of the same cluster must not collapse into one `draw_id`."""
  cluster0, cluster1 = [("c0", "t0")], [("c1a", "t1a"), ("c1b", "t1b")]
  clusters = [cluster0, cluster1]
  rng = random.Random(0)
  items, draw_ids = significance._resample_clusters(clusters, rng)
  assert len(items) == len(draw_ids)
  by_draw: dict = {}
  for item, did in zip(items, draw_ids):
    by_draw.setdefault(did, []).append(item)
  assert len(by_draw) == 2   # m=2 draws, regardless of which clusters they hit
  for group in by_draw.values():
    assert group == cluster0 or group == cluster1


def test_mde_large_effect_converges_to_a_small_delta():
  rng = random.Random(1)
  n = 50
  control, treatment, cluster_ids = [], [], []
  for i in range(n):
    control.append(1.0 if rng.random() < 0.5 else 0.0)
    treatment.append(1.0 if rng.random() < 0.85 else 0.0)
    cluster_ids.append(i)
  boot = significance.cluster_bootstrap_ci_clustered(
      control, treatment, cluster_ids, n_boot=500, seed=1
  )
  width = boot["ci_high"] - boot["ci_low"]
  mde = significance.minimum_detectable_effect_clustered(
      control, treatment, cluster_ids, initial_hi=max(0.05, width),
      n_replicates=100, n_perm=200, seed=1,
  )
  assert mde["delta"] is not None
  # A ~50-pair, ~50% base-rate design's MDE is on the order of a few tenths,
  # not near-zero (the deterministic-injection bug this test guards against
  # made it converge to ~0.001) and not near 1.0 either.
  assert 0.05 < mde["delta"] < 0.6
  assert mde["realized_diff"] <= mde["delta"] + 1e-9


def test_mde_near_ceiling_base_rate_needs_a_larger_delta():
  """MDE depends on the row's own noise structure (cluster count, control's
  base rate) -- not on the observed treatment effect, and not on whether
  the *observed* effect happens to be large or small (see the plan's note
  on why "tiny observed effect -> large MDE" was dropped as a test claim;
  it isn't true). What *is* true: a near-ceiling control has little room
  for a Bernoulli draw to move (`clip(1 + delta) = 1`, no signal possible
  from those pairs at all), so reaching the same power needs a larger
  nominal delta than a mid-range control at the same cluster count.
  """
  n = 60
  rng_mid = random.Random(3)
  control_mid = [1.0 if rng_mid.random() < 0.5 else 0.0 for _ in range(n)]
  treatment_mid = [1.0 if rng_mid.random() < 0.5 else 0.0 for _ in range(n)]
  cluster_ids = list(range(n))

  rng_ceiling = random.Random(4)
  control_ceiling = [1.0 if rng_ceiling.random() < 0.8 else 0.0 for _ in range(n)]
  treatment_ceiling = [1.0 if rng_ceiling.random() < 0.8 else 0.0 for _ in range(n)]

  mde_mid = significance.minimum_detectable_effect_clustered(
      control_mid, treatment_mid, cluster_ids, initial_hi=0.1,
      n_replicates=100, n_perm=200, seed=3,
  )
  mde_ceiling = significance.minimum_detectable_effect_clustered(
      control_ceiling, treatment_ceiling, cluster_ids, initial_hi=0.1,
      n_replicates=100, n_perm=200, seed=4,
  )
  assert mde_mid["delta"] is not None
  # "exceeds 1.0" (None) is itself consistent with "harder to detect" --
  # only a *smaller or equal* ceiling delta would contradict the claim.
  if mde_ceiling["delta"] is not None:
    assert mde_ceiling["delta"] > mde_mid["delta"]


def test_mde_no_pairs_returns_none_with_a_note():
  mde = significance.minimum_detectable_effect_clustered(
      [], [], [], initial_hi=0.1
  )
  assert mde["delta"] is None
  assert mde["note"] is not None


# --- --metric mae ------------------------------------------------------------


def test_mae_eligible_frame_filters_correctly():
  raw = pd.DataFrame({
      "task": ["node_count", "node_count", "cycle_check", "node_count"],
      "failure_type": ["correct", "non_terminating", "correct", "unparsed"],
      "absolute_error": [2.0, 3.0, None, None],
  })
  eligible = cs._mae_eligible_frame(raw)
  assert len(eligible) == 1
  assert eligible.iloc[0]["absolute_error"] == 2.0


def test_report_mae_sign_convention_positive_means_helped():
  """`mae_delta = control_mae - treatment_mae`, so a condition that
  reduces error (helps) must show a *positive* `mae_delta` -- the opposite
  raw sign from what `paired_permutation_test_clustered` itself returns
  (`treatment - control`), which is why `_report_mae` flips it."""
  raw = pd.DataFrame({
      "model": ["gemma4-12b"] * 20,
      "instance_id": [f"node_count/{i}" for i in range(10)] * 2,
      "style": ["zero_shot"] * 20,
      "node_naming": ["integer"] * 20,
      "condition": ["none"] * 10 + ["degree"] * 10,
      "failure_type": ["correct"] * 20,
      # control error consistently higher than treatment -> primer helps.
      "absolute_error": [5.0] * 10 + [1.0] * 10,
  })
  records = []
  cs._report_mae(raw, raw, "gemma4-12b", "node_count", _default_args(), records)
  row = records[0]
  assert row["mae_delta"] == pytest.approx(4.0)
  assert row["bh_significant"] is True


def test_report_mae_keeps_tasks_separate_not_pooled():
  """Two tasks with opposite-sign MAE effects must produce two separate
  records, not get averaged into one pooled number the way `exact`'s
  metric pools across all 6 tasks."""
  raw = pd.DataFrame({
      "model": ["gemma4-12b"] * 8,
      "instance_id": (
          ["node_count/0", "node_count/1"] * 2
          + ["edge_count/0", "edge_count/1"] * 2
      ),
      "style": ["zero_shot"] * 8,
      "node_naming": ["integer"] * 8,
      "task": ["node_count"] * 4 + ["edge_count"] * 4,
      "condition": ["none", "none", "degree", "degree"] * 2,
      "failure_type": ["correct"] * 8,
      # node_count: control error > treatment error -> helps.
      # edge_count: control error < treatment error -> hurts.
      "absolute_error": [5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0],
  })
  records = []
  args = _default_args()
  for task in ("node_count", "edge_count"):
    subset = raw[raw["task"] == task]
    cs._report_mae(subset, subset, "gemma4-12b", task, args, records)
  by_task = {r["task"]: r["mae_delta"] for r in records}
  assert by_task["node_count"] == pytest.approx(4.0)
  assert by_task["edge_count"] == pytest.approx(-4.0)


def test_report_mae_also_excludes_all_from_the_bh_family():
  """Same fix as `_report`'s: `all` must not be pooled into the same
  multiple-comparison family as the independent conditions in `--metric
  mae` mode either."""
  n = 30
  conditions = ["none", "degree", "clustering", "rwse", "components", "filler", "all"]
  cols = {"model": [], "instance_id": [], "style": [], "node_naming": [],
          "condition": [], "failure_type": [], "absolute_error": []}
  for condition in conditions:
    for i in range(n):
      cols["model"].append("gemma4-12b")
      cols["instance_id"].append(f"node_count/{i}")
      cols["style"].append("zero_shot")
      cols["node_naming"].append("integer")
      cols["condition"].append(condition)
      cols["failure_type"].append("correct")
      cols["absolute_error"].append(1.0 if i % 2 == 0 else 3.0)
  raw = pd.DataFrame(cols)
  records = []
  cs._report_mae(raw, raw, "gemma4-12b", "node_count", _default_args(), records)
  by_condition = {r["condition"]: r for r in records}
  assert by_condition["all"]["is_derived_condition"] is True
  independent_families = {
      by_condition[c]["bh_family"]
      for c in ("degree", "clustering", "rwse", "components", "filler")
  }
  assert len(independent_families) == 1
  assert by_condition["all"]["bh_family"].endswith("/derived")


def test_report_mae_tags_records_with_metric():
  raw = pd.DataFrame({
      "model": ["gemma4-12b"] * 20,
      "instance_id": [f"node_count/{i}" for i in range(10)] * 2,
      "style": ["zero_shot"] * 20,
      "node_naming": ["integer"] * 20,
      "condition": ["none"] * 10 + ["degree"] * 10,
      "failure_type": ["correct"] * 20,
      "absolute_error": [5.0] * 10 + [1.0] * 10,
  })
  records = []
  cs._report_mae(raw, raw, "gemma4-12b", "node_count", _default_args(), records)
  assert all(r["metric"] == "mae" for r in records)


def test_report_tags_records_with_metric():
  raw = _near_ceiling_frame(0.5)
  records = []
  cs._report(raw, raw, "exact", "gemma4-12b", _default_args(),
             "main_sweep", records, bound="excluded")
  assert all(r["metric"] == "exact" for r in records)


def test_apply_global_bh_pools_exact_and_mae_into_one_family():
  """1.1.2: exact and mae are two lenses on the same hypotheses and must
  share one multiplicity budget, not two separate ones."""
  exact_records = [
      {"group": f"exact_{i}", "bound": "excluded", "p_value": 0.5 + 0.01 * i,
       "is_derived_condition": False, "metric": "exact"}
      for i in range(10)
  ]
  mae_records = [
      {"group": f"mae_{i}", "bound": "excluded", "p_value": 0.001,
       "is_derived_condition": False, "metric": "mae"}
      for i in range(2)
  ]
  records = exact_records + mae_records
  cs._apply_global_bh(records, q=0.05)
  # The union has 12 rows; the two mae rows' small p-values are the ones
  # that survive BH at that family size -- confirms both metrics were
  # actually pooled into one correction, not run as two separate passes.
  assert all(r["bh_significant_global"] is True for r in mae_records)
  assert all(r["bh_significant_global"] is False for r in exact_records)
