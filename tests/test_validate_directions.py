"""Checks for scripts/validate_directions.py's text helpers and scoring."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import validate_directions as vd  # noqa: E402


def test_window_marks_the_span_and_collapses_whitespace():
  text = "a\n\nb  I count 12 nodes here c"
  s = text.index("12")
  assert vd.window(text, s, s + 2, before=8, after=6) == "… I count «12» nodes …"
  assert vd.window("12", 0, 2) == "«12»"


def test_spread_keeps_first_and_last():
  assert vd.spread(list(range(10)), 4) == [0, 3, 6, 9]
  assert vd.spread([1, 2], 6) == [1, 2]


def test_normalize_reads_what_a_labeller_types():
  assert [vd.normalize(v) for v in ["Y", " yes", "0", "N ", "b", "?", None, ""]] == [
      "y", "y", "n", "n", "b", "", "", ""]


def test_agreement_counts_only_yes_no_labels_against_the_claim():
  x = pd.DataFrame({"label": ["y", "n", "", "y"], "expected": ["y", "y", "y", "n"]})
  assert vd.agreement(x) == (1, 3)


def test_clopper_pearson_matches_the_known_bound():
  lo, hi = vd.clopper_pearson(20, 20)
  assert lo == pytest.approx(0.8316, abs=1e-4) and hi == 1.0
  assert vd.clopper_pearson(0, 10)[0] == 0.0
