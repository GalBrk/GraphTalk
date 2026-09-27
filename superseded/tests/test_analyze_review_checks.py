"""Tests for the pure helpers in `superseded/scripts/analyze_review_checks.py`."""

import importlib.util
import math
import pathlib
import sys

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # live scripts
_spec = importlib.util.spec_from_file_location(
    "analyze_review_checks", _SCRIPTS / "analyze_review_checks.py")
arc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arc)


def test_sdt_separates_bias_from_sensitivity():
  d, c = arc.sdt(50, 100, 50, 100)
  assert abs(d) < 1e-9 and abs(c) < 1e-9
  # Same sensitivity, more "yes": d' unchanged, criterion drops (liberal).
  d1, c1 = arc.sdt(69, 100, 31, 100)
  d2, c2 = arc.sdt(89, 100, 58, 100)
  assert abs(d1 - d2) < 0.1 and c2 < c1 - 0.5
  # A 100% hit rate stays finite under the log-linear correction.
  assert math.isfinite(arc.sdt(100, 100, 40, 100)[0])


def test_interaction_recovers_slope_difference():
  pts = [(b / 10, -30 * b / 10, True) for b in range(11)]
  pts += [(b / 10, 5.0 + (b % 2), False) for b in range(11)]
  assert abs(arc.interaction_coef(pts) + 30) < 1e-6


def test_crossfit_point_uses_disjoint_folds():
  pairs = [(f"id{i}", i % 2, 1) for i in range(40)]
  folds = {i: arc.abl._fold(i) for i, _, _ in pairs}
  base, eff = arc.cell_point(pairs, fold=0)
  fold0 = [b for i, b, _ in pairs if folds[i] == 0]
  fold1 = [b for i, b, _ in pairs if folds[i] == 1]
  assert base == sum(fold0) / len(fold0)
  assert abs(eff - 100 * (1 - sum(fold1) / len(fold1))) < 1e-9
