"""Tests for `scripts/analyze_baseline_law.py`.

Two of these guard decisions that would fail silently rather than loudly.

`test_degenerate_tasks_report_no_route_gain` pins the n=40 correction: the
shortcut bars in `shortcuts.json` were fitted on the published split's small
graphs, where a per-node primer gives `node_count` away. At n=40 the gold
answer is the constant 40, so the blind bar is ~1.00 under `none` too and the
primer adds nothing. Leaving those tasks in flips the split's headline
correlation from $+0.04$ to $-0.41$.

`test_size40_is_not_read_as_a_replication_seed` pins the seed filter. The
seed-offset replication corpus exists so that it can never be pooled with the
default-seed graphs, and its marker is `/s<digits>/` -- but every instance id
in the density sweep also contains `size40`, so a filter that matches on
`"/s"` discards the entire corpus and reports nothing at all.
"""

import importlib.util
import math
import pathlib

import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "analyze_baseline_law.py")
_spec = importlib.util.spec_from_file_location("analyze_baseline_law", _SCRIPT)
abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abl)

BARS = {
    "node_degree/none": 0.082, "node_degree/degree": 1.000,
    "node_degree/clustering": 0.082, "node_degree/rwse": 0.616,
    "node_count/none": 0.064, "node_count/clustering": 1.000,
    "cycle_check/none": 0.832, "cycle_check/components": 1.000,
}


# --- the route split -------------------------------------------------------

def test_route_gain_is_measured_against_none_not_against_zero():
  assert abl.route_gain(BARS, "node_degree", "degree") == pytest.approx(0.918)
  assert abl.route_gain(BARS, "node_degree", "clustering") == pytest.approx(0.0)


def test_degenerate_tasks_report_no_route_gain():
  # Both bars claim a large gain on the published split and neither survives
  # at n=40, where the answer is constant.
  assert BARS["node_count/clustering"] - BARS["node_count/none"] > 0.9
  assert abl.route_gain(BARS, "node_count", "clustering") == 0.0
  assert abl.route_gain(BARS, "cycle_check", "components") == 0.0
  assert not abl.offers_route(BARS, "node_count", "clustering")


def test_offers_route_threshold_sits_in_the_empty_band():
  # Every gain in the real table is <= 0.012 or >= 0.114, so the threshold
  # only has to land between them.
  assert abl.offers_route(BARS, "node_degree", "rwse")
  assert not abl.offers_route(BARS, "node_degree", "clustering")


# --- instance-id parsing ---------------------------------------------------

@pytest.mark.parametrize("instance_id, expected", [
    ("node_degree/size40/p0.35/12", 0.35),
    ("node_degree/size40/p0.1/0", 0.1),
    ("node_degree/7", None),
])
def test_density_of(instance_id, expected):
  assert abl.density_of(instance_id) == expected


def test_size40_is_not_read_as_a_replication_seed():
  assert not abl.is_replication_seed("node_degree/size40/p0.35/12")
  assert abl.is_replication_seed("node_degree/size40/s500000/p0.35/12")


# --- statistics ------------------------------------------------------------
#
# scipy is not installed in the envs this project runs in (see CLAUDE.md), so
# `pearson` is hand-rolled and needs pinning against known values.

def test_pearson_on_an_exact_relation():
  r, p = abl.pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
  assert r == pytest.approx(1.0, abs=1e-6)
  assert p < 1e-6


def test_pearson_matches_a_known_value():
  # scipy.stats.pearsonr([1,2,3,4,5,6], [2,1,4,3,6,5]) -> (0.8285714, 0.04156)
  r, p = abl.pearson([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5])
  assert r == pytest.approx(0.8285714, abs=1e-6)
  assert p == pytest.approx(0.041563, abs=1e-5)


def test_pearson_is_nan_without_variance():
  r, _ = abl.pearson([1, 1, 1, 1], [1, 2, 3, 4])
  assert math.isnan(r)
