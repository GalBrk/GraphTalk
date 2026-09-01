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

import pandas as pd
import pytest

from graphtalk import analysis
from graphtalk import scoring
from graphtalk import significance
from scripts import check_significance as cs


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
  assert cluster_ids == ["node_count/1"]
  assert control == [0.0]
  assert treatment == [1.0]
