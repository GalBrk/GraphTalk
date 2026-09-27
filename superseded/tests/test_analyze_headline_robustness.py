"""Tests for the pure pairing logic in superseded/scripts/analyze_headline_robustness.py."""

import importlib.util
import pathlib

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "analyze_headline_robustness.py")
_spec = importlib.util.spec_from_file_location("analyze_headline_robustness", _SCRIPT)
ahr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ahr)


def test_paired_delta_only_uses_ids_present_under_both_conditions():
  scores = {
      ("a", "clustering"): 1.0, ("a", "none"): 0.0,   # +1
      ("b", "clustering"): 0.0, ("b", "none"): 1.0,   # -1
      ("c", "clustering"): 1.0,                        # no `none` row: excluded
      ("d", "none"): 1.0,                              # no `clustering` row: excluded
  }
  delta, p, n = ahr.paired_delta(scores, {"a", "b", "c", "d"}, "clustering", "none")
  assert n == 2
  assert delta == 0.0


def test_paired_delta_sign_matches_which_condition_wins():
  scores = {("a", "x"): 1.0, ("a", "y"): 0.0,
           ("b", "x"): 1.0, ("b", "y"): 0.0}
  delta, p, n = ahr.paired_delta(scores, {"a", "b"}, "x", "y")
  assert delta == 100.0
  assert n == 2
