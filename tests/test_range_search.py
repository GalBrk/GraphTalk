"""Tests for graphtalk/range_search.py.

Pins the two classification bars (Stage 2's relaxed `scout_decision` and
Stage 3's stricter `zone_decision`/`classify_cell`) against hand-picked
fixtures, including the boundary values themselves, and checks that nothing
here ever excludes a row -- only labels it.
"""

import pytest

from graphtalk import range_search as rs


# --- wilson_interval ---------------------------------------------------------


def test_wilson_interval_zero_n_is_maximally_wide():
  lo, hi = rs.wilson_interval(0, 0)
  assert (lo, hi) == (0.0, 1.0)


def test_wilson_interval_narrows_with_more_samples():
  lo30, hi30 = rs.wilson_interval(27, 30)  # 90% observed
  lo100, hi100 = rs.wilson_interval(90, 100)  # same 90% observed
  assert (hi100 - lo100) < (hi30 - lo30)


def test_wilson_interval_unsupported_confidence_raises():
  with pytest.raises(ValueError, match="confidence"):
    rs.wilson_interval(10, 20, confidence=0.99)


# --- scout_decision: relaxed thresholds, boundary values --------------------


def test_scout_decision_survives_between_relaxed_and_strict_ceiling():
  # 95% would already be dropped by zone_decision's 0.90 ceiling, but
  # survives the scout's relaxed 0.98 -- proving the scout is strictly
  # more permissive, not just differently tuned.
  result = rs.scout_decision(
      none_accuracy=0.95, majority_baseline=0.10, truncation_rate=0.05,
  )
  assert result["decision"] == "survive"
  assert rs.zone_decision(0.95, majority_baseline=0.10) == "ceiling"


def test_scout_decision_drops_above_relaxed_ceiling():
  result = rs.scout_decision(
      none_accuracy=0.99, majority_baseline=0.10, truncation_rate=0.0,
  )
  assert result["decision"] == "drop"
  assert result["zone"] == "ceiling"
  assert "0.99" in result["reason"] or "99.0%" in result["reason"]


def test_scout_decision_exactly_at_ceiling_boundary_survives():
  # Strictly greater-than, so the boundary value itself survives.
  result = rs.scout_decision(
      none_accuracy=0.98, majority_baseline=0.10, truncation_rate=0.0,
  )
  assert result["decision"] == "survive"


def test_scout_decision_floor_rule_is_not_relaxed():
  # Standard floor margin (0.02), same as zone_decision -- no rescue here.
  result = rs.scout_decision(
      none_accuracy=0.11, majority_baseline=0.10, truncation_rate=0.0,
  )
  assert result["decision"] == "drop"
  assert result["zone"] == "floor"


def test_scout_decision_survives_moderate_truncation():
  # 20% truncation would fail zone_decision's implicit 10% bar but
  # survives the scout's relaxed 30%.
  result = rs.scout_decision(
      none_accuracy=0.5, majority_baseline=0.10, truncation_rate=0.20,
  )
  assert result["decision"] == "survive"


def test_scout_decision_drops_extreme_truncation():
  result = rs.scout_decision(
      none_accuracy=0.5, majority_baseline=0.10, truncation_rate=0.35,
  )
  assert result["decision"] == "drop"
  assert result["zone"] is None
  assert "truncation" in result["reason"]


# --- zone_decision / classify_cell -------------------------------------------


def test_zone_decision_node_count_style_cell_hits_ceiling_automatically():
  # A --node-count-fixed cell drives none-accuracy and majority_baseline
  # both toward 1.0 -- no special case needed, the ordinary ceiling branch
  # (checked before floor) already excludes it.
  assert rs.zone_decision(none_accuracy=1.0, majority_baseline=1.0) == "ceiling"


def test_zone_decision_informative_between_ceiling_and_floor():
  assert rs.zone_decision(none_accuracy=0.5, majority_baseline=0.10) == "informative"


def test_classify_cell_passes_only_when_all_three_checks_clear():
  clean = rs.classify_cell(
      none_accuracy=0.5, majority_baseline=0.10,
      shortcut_score=0.5, truncation_rate=0.05,
  )
  assert clean["passes"] is True
  assert clean["reason"] == "clean"

  contaminated = rs.classify_cell(
      none_accuracy=0.5, majority_baseline=0.10,
      shortcut_score=0.95, truncation_rate=0.05,
  )
  assert contaminated["passes"] is False
  assert "shortcut" in contaminated["reason"]

  truncated = rs.classify_cell(
      none_accuracy=0.5, majority_baseline=0.10,
      shortcut_score=0.5, truncation_rate=0.5,
  )
  assert truncated["passes"] is False
  assert "truncation" in truncated["reason"]


def test_classify_cell_missing_shortcut_score_is_never_clean():
  result = rs.classify_cell(
      none_accuracy=0.5, majority_baseline=0.10,
      shortcut_score=None, truncation_rate=0.05,
  )
  assert result["shortcut_clean"] is False
  assert result["passes"] is False


# --- arm_divergence -----------------------------------------------------------


def test_arm_divergence_false_when_arms_agree():
  plain = rs.classify_cell(0.5, 0.10, 0.5, 0.05)
  think = rs.classify_cell(0.5, 0.10, 0.5, 0.05)
  result = rs.arm_divergence(plain, think)
  assert result["diverges"] is False
  assert result["reason"] == ""


def test_arm_divergence_true_when_only_think_truncates():
  plain = rs.classify_cell(0.5, 0.10, 0.5, truncation_rate=0.02)
  think = rs.classify_cell(0.5, 0.10, 0.5, truncation_rate=0.18)
  result = rs.arm_divergence(plain, think)
  assert result["diverges"] is True
  assert "truncation_clean" in result["reason"]
  assert "passes" in result["reason"]


def test_arm_divergence_true_when_zones_disagree():
  # Plain is floor (no headroom), think is informative -- exactly the
  # "one arm raises a risk the other doesn't" case this exists to surface.
  plain = rs.classify_cell(0.11, 0.10, 0.5, 0.02)
  think = rs.classify_cell(0.61, 0.10, 0.5, 0.02)
  result = rs.arm_divergence(plain, think)
  assert result["diverges"] is True
  assert "zone" in result["reason"]
