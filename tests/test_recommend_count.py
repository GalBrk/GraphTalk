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
      "mde_realized_diff": None, "mde_delta_negative": None,
      "mde_realized_diff_negative": None, "bh_significant": False,
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


def test_non_significant_cell_uses_the_reports_own_realized_mde():
  """`bh_significant=False` -> take the MDE from the report, no frame
  needed. It must be the **realized** column: `delta` is an observed
  accuracy difference, and only `mde_realized_diff` is on that scale.
  """
  report = pd.DataFrame([_report_row(
      delta=0.1, mde_delta=0.6, mde_realized_diff=0.2,
      bh_significant=False, bh_significant_global=False,
  )])
  result = rc.recommend(report)
  row = result.iloc[0]
  assert row["skip_reason"] is None
  assert row["mde_used"] == pytest.approx(0.2)
  ratio = (0.2 / 0.1) ** 2
  assert row["n_clusters_needed"] == pytest.approx(30 * ratio)


def test_the_swept_parameter_is_never_divided_by_an_observed_delta():
  """The units bug, pinned so it cannot come back. `mde_delta` is the
  parameter the MDE search swept; `mde_realized_diff` is what that
  parameter actually produced on the row's data. Near the ceiling they
  differ by an order of magnitude -- only ~11% of rows can move at 89%
  control accuracy, so a swept 0.40 realizes ~0.044. Using the parameter
  inflates the extrapolation by `(1 / headroom) ** 2`, which measured
  53-163x on the real positive-`delta` cells and turned a cell already
  significant at 30 graphs into a 515-graph cluster booking.
  """
  report = pd.DataFrame([_report_row(
      delta=0.0444, mde_delta=0.40, mde_realized_diff=0.044,
      bh_significant=False, bh_significant_global=False,
  )])
  row = rc.recommend(report).iloc[0]
  inflated = 30 * (0.40 / 0.0444) ** 2       # ~2,435 graphs
  correct = 30 * (0.044 / 0.0444) ** 2       # ~29 graphs
  assert row["recommended_count"] == pytest.approx(correct, rel=1e-6)
  assert row["recommended_count"] < inflated / 50


def test_a_non_converged_search_is_skipped_not_extrapolated():
  """"MDE exceeds 1.0" leaves `mde_delta` null but still reports the
  realized value the search reached. Extrapolating from that would invent
  a recommendation out of a bound the data never actually attained, so
  convergence is still read off the parameter column."""
  report = pd.DataFrame([_report_row(
      delta=0.1, mde_delta=None, mde_realized_diff=0.013,
      bh_significant=False, bh_significant_global=False,
  )])
  row = rc.recommend(report).iloc[0]
  assert row["skip_reason"].startswith("MDE did not converge")
  assert pd.isna(row["recommended_count"])


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
