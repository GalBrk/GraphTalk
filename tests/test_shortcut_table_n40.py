"""Tests for scripts/shortcut_table_n40.py's pure `flatten` function."""

from scripts.shortcut_table_n40 import flatten


def test_flatten_averages_across_densities():
  per_density = {
      "0.1": {"node_degree/none": 0.2, "node_degree/degree": 1.0},
      "0.5": {"node_degree/none": 0.0, "node_degree/degree": 1.0},
  }
  out = flatten(per_density)
  assert out == {"node_degree/none": 0.1, "node_degree/degree": 1.0}


def test_flatten_handles_a_key_missing_from_one_density():
  per_density = {
      "0.1": {"a": 1.0, "b": 0.5},
      "0.2": {"a": 0.0},
  }
  # `b` is only present at one density; averaging over len(per_density) (2)
  # rather than over the count of densities that have `b` intentionally
  # treats a missing cell as diluting the mean, not as absent from it --
  # every task/condition pair is scored at every density in this corpus, so
  # this only matters if that invariant is ever broken.
  out = flatten(per_density)
  assert out["a"] == 0.5
  assert out["b"] == 0.25
