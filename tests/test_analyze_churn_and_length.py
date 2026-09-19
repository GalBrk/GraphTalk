"""Tests for scripts/analyze_churn_and_length.py's `cells` function."""

import importlib.util
import pathlib

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "analyze_churn_and_length.py")
_spec = importlib.util.spec_from_file_location("analyze_churn_and_length", _SCRIPT)
acl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acl)


def test_churn_counts_helped_and_hurt_separately_not_just_net():
  # 2 helped, 2 hurt: net delta is 0, but churn (discordant pairs) is 4, not 0.
  data = {
      ("node_degree", 0.5, "none", "a"): (0, 5),
      ("node_degree", 0.5, "none", "b"): (1, 5),
      ("node_degree", 0.5, "none", "c"): (0, 5),
      ("node_degree", 0.5, "none", "d"): (1, 5),
      ("node_degree", 0.5, "degree", "a"): (1, 5),   # helped
      ("node_degree", 0.5, "degree", "b"): (0, 5),   # hurt
      ("node_degree", 0.5, "degree", "c"): (1, 5),   # helped
      ("node_degree", 0.5, "degree", "d"): (0, 5),   # hurt
  }
  # pad to clear min_pairs=10 by duplicating with distinct ids in a real run;
  # here min_pairs is lowered directly instead.
  cell, = acl.cells(data, min_pairs=4)
  assert cell["helped"] == 2
  assert cell["hurt"] == 2
  assert cell["churn"] == 4
  assert cell["delta"] == 0.0
  assert cell["sig_over_churn"] == 0.0


def test_hit_cap_pairs_are_excluded_from_both_churn_and_tokens():
  data = {
      ("node_degree", 0.5, "none", "a"): (None, None),      # none capped
      ("node_degree", 0.5, "degree", "a"): (1, 500),
      ("node_degree", 0.5, "none", "b"): (0, 100),
      ("node_degree", 0.5, "degree", "b"): (None, None),    # degree capped
      ("node_degree", 0.5, "none", "c"): (0, 100),
      ("node_degree", 0.5, "degree", "c"): (1, 300),
  }
  cell, = acl.cells(data, min_pairs=1)
  assert cell["n"] == 1  # only pair c has both sides uncapped
  assert cell["dtokens"] == 200.0


def test_mean_token_delta_is_paired_not_pooled_across_arms():
  data = {
      ("node_degree", 0.5, "none", "a"): (0, 100),
      ("node_degree", 0.5, "none", "b"): (0, 200),
      ("node_degree", 0.5, "degree", "a"): (0, 150),   # +50
      ("node_degree", 0.5, "degree", "b"): (0, 260),   # +60
  }
  cell, = acl.cells(data, min_pairs=2)
  assert cell["dtokens"] == 55.0


def test_cells_below_min_pairs_are_dropped():
  data = {
      ("node_degree", 0.5, "none", "a"): (0, 100),
      ("node_degree", 0.5, "degree", "a"): (1, 150),
  }
  assert acl.cells(data, min_pairs=10) == []
