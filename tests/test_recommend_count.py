"""`scripts/recommend_count.py`: the closed-form `--count` extrapolation,
and the on-demand MDE path for family-significant-but-not-globally-
significant cells (see the module docstring for why that case needs a
fresh simulation rather than reusing anything `check_significance.py`
already computed).
"""

import random

import pandas as pd
import pytest

from scripts import recommend_count as rc


def _report_row(**overrides) -> dict:
  base = {
      "arm": "main_sweep", "metric": "exact", "bound": "excluded",
      "is_derived_condition": False, "group": "model_a", "condition": "degree",
      "n_clusters": 30, "delta": 0.1, "mde_delta": None,
      "mde_delta_negative": None, "bh_significant": False,
      "bh_significant_global": False,
  }
  base.update(overrides)
  return base


def test_already_globally_significant_is_skipped():
  report = pd.DataFrame([_report_row(
      bh_significant=True, bh_significant_global=True,
  )])
  result = rc.recommend(report)
  assert result.iloc[0]["skip_reason"] == "already globally significant"
  assert pd.isna(result.iloc[0]["recommended_count"])


def test_zero_delta_is_skipped():
  report = pd.DataFrame([_report_row(delta=0.0)])
  result = rc.recommend(report)
  assert result.iloc[0]["skip_reason"] == "observed delta is exactly zero"


def test_non_significant_cell_uses_the_reports_own_mde_unchanged():
  # Existing behavior: bh_significant=False entirely -> use mde_delta from
  # the report directly, no frame needed.
  report = pd.DataFrame([_report_row(
      delta=0.1, mde_delta=0.2, bh_significant=False, bh_significant_global=False,
  )])
  result = rc.recommend(report)
  row = result.iloc[0]
  assert row["skip_reason"] is None
  assert row["mde_used"] == pytest.approx(0.2)
  ratio = (0.2 / 0.1) ** 2
  assert row["n_clusters_needed"] == pytest.approx(30 * ratio)


def test_family_significant_not_global_is_skipped_without_a_frame():
  report = pd.DataFrame([_report_row(
      bh_significant=True, bh_significant_global=False,
  )])
  result = rc.recommend(report)   # no frame passed
  assert "pass --frame" in result.iloc[0]["skip_reason"]
  assert pd.isna(result.iloc[0]["recommended_count"])


def _synthetic_frame(control_rate: float, treatment_rate: float, n: int, seed: int):
  """Mirrors `tests/test_significance.py`'s own `_near_ceiling_frame`-style
  fixtures: `n` paired instances, control/treatment exact outcomes drawn
  from fixed rates so the resulting delta and MDE are both real, non-
  degenerate numbers."""
  rng = random.Random(seed)
  rows = []
  for i in range(n):
    instance_id = f"node_count/{i}"
    for condition, rate in (("none", control_rate), ("degree", treatment_rate)):
      rows.append({
          "model": "model_a", "model_family": "model_a", "is_think": False,
          "instance_id": instance_id, "style": "zero_shot",
          "node_naming": "integer", "condition": condition,
          "exact": 1.0 if rng.random() < rate else 0.0,
      })
  return pd.DataFrame(rows)


def test_family_significant_not_global_gets_a_real_mde_with_a_frame():
  frame = _synthetic_frame(control_rate=0.5, treatment_rate=0.8, n=30, seed=7)
  observed_delta = (
      frame.loc[frame["condition"] == "degree", "exact"].mean()
      - frame.loc[frame["condition"] == "none", "exact"].mean()
  )
  report = pd.DataFrame([_report_row(
      group="model_a", condition="degree", n_clusters=30, delta=observed_delta,
      bh_significant=True, bh_significant_global=False,
  )])
  result = rc.recommend(report, frame=frame)
  row = result.iloc[0]
  assert row["skip_reason"] is None
  assert row["mde_used"] is not None and not pd.isna(row["mde_used"])
  # A real MDE was simulated (not just copied from a blank report column)
  # and drives a real, positive recommendation.
  assert row["recommended_count"] > 0


def test_mde_for_family_significant_cell_matches_the_observed_deltas_sign():
  frame = _synthetic_frame(control_rate=0.5, treatment_rate=0.8, n=30, seed=11)
  mde = rc._mde_for_family_significant_cell(frame, "model_a", "degree", delta=0.1)
  assert mde is None or mde > 0   # positive-direction search, per delta > 0


def test_mde_for_family_significant_cell_returns_none_for_no_paired_rows():
  frame = _synthetic_frame(control_rate=0.5, treatment_rate=0.8, n=5, seed=3)
  # A condition that doesn't exist in this frame at all -- no pairs.
  assert rc._mde_for_family_significant_cell(frame, "model_a", "rwse", delta=0.1) is None
