"""Tests for `scripts/test_vs_controls.py`'s cell-aggregation logic.

The comparisons themselves reuse `scoring.score_one`/`significance.
paired_permutation_test`/`benjamini_hochberg`, already covered elsewhere
(matching ci_all.py's own precedent of no dedicated test file for the load +
score loop). `cells_beating_both_controls` is the one piece of new branching
logic worth pinning directly: it decides what "beats both none and filler"
means once a global BH correction is in the picture.
"""

import importlib.util
import pathlib

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "test_vs_controls.py")
_spec = importlib.util.spec_from_file_location("test_vs_controls", _SCRIPT)
tvc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tvc)


def _row(arm, task, cond, control, delta, reject):
  return {"arm": arm, "task": task, "condition": cond, "control": control,
          "n": 100, "delta": delta, "p_perm": 0.0, "bh_global_reject": reject}


def test_cell_must_beat_both_controls_and_survive_bh():
  results = [
      _row("a", "node_degree", "clustering", "none", +10.0, True),
      _row("a", "node_degree", "clustering", "filler", +8.0, True),
  ]
  beats, _ = tvc.cells_beating_both_controls(results)
  assert len(beats) == 1


def test_positive_vs_none_but_not_significant_vs_filler_is_excluded():
  results = [
      _row("a", "node_degree", "clustering", "none", +10.0, True),
      _row("a", "node_degree", "clustering", "filler", +8.0, False),  # fails BH
  ]
  beats, _ = tvc.cells_beating_both_controls(results)
  assert beats == []


def test_negative_delta_vs_filler_is_excluded_even_if_bh_rejects():
  # bh_global_reject only says "not by chance", not "in the helpful direction".
  results = [
      _row("a", "node_degree", "clustering", "none", +10.0, True),
      _row("a", "node_degree", "clustering", "filler", -5.0, True),
  ]
  beats, _ = tvc.cells_beating_both_controls(results)
  assert beats == []


def test_missing_one_control_excludes_the_cell():
  results = [_row("a", "node_degree", "clustering", "none", +10.0, True)]
  beats, by_cell = tvc.cells_beating_both_controls(results)
  assert beats == []
  assert len(by_cell) == 1  # the cell exists, just incomplete
