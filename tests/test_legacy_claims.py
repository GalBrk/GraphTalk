"""Checks for scripts/legacy_claims.py's two pure helpers."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import legacy_claims as lc  # noqa: E402


def test_mde_points_is_floored_at_one_discordant_pair():
  assert lc.mde_points(0.0, 400) == pytest.approx(lc.mde_points(1 / 400, 400))
  assert lc.mde_points(0.08, 100) == pytest.approx(100 * lc.Z80 * (0.08 / 100) ** 0.5)


def test_route_gains_are_measured_against_none():
  bars = {"node_degree/none": 0.1, "node_degree/rwse": 0.2, "node_degree/degree": 1.0}
  g = lc.route_gains(bars, tasks=("node_degree",), primers=("rwse", "degree"))
  assert g == pytest.approx({"node_degree/rwse": 0.1, "node_degree/degree": 0.9})
