"""Tests for `superseded/scripts/analyze_baseline_law.py`.

Three of these guard decisions the paper's §5.3 rests on, and would fail
silently rather than loudly.

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

`test_pairs_drop_capped_rows_on_both_sides` pins the `hit_cap` policy, which
matches `superseded/scripts/score_density_sweep.py`: a truncated generation is a budget
failure, and a pair is only usable when neither side truncated.
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


# --- pairing ---------------------------------------------------------------

def test_pairs_drop_capped_rows_on_both_sides():
  scores = {
      ("node_degree", 0.5, "a", "none"): 1.0,
      ("node_degree", 0.5, "a", "degree"): 0.0,
      ("node_degree", 0.5, "b", "none"): None,      # control truncated
      ("node_degree", 0.5, "b", "degree"): 1.0,
      ("node_degree", 0.5, "c", "none"): 1.0,
      ("node_degree", 0.5, "c", "degree"): None,    # treatment truncated
  }
  cell, = abl.cells_from_scores(scores, min_pairs=1)
  assert cell["n"] == 1
  assert cell["baseline"] == pytest.approx(1.0)
  assert cell["delta"] == pytest.approx(-100.0)


def test_cells_are_keyed_by_density_so_levels_are_not_pooled():
  scores = {}
  for density, treated in ((0.1, 1.0), (0.5, 0.0)):
    scores[("node_degree", density, "a", "none")] = 0.0
    scores[("node_degree", density, "a", "degree")] = treated
  cells = abl.cells_from_scores(scores, min_pairs=1)
  assert {c["density"] for c in cells} == {0.1, 0.5}
  assert {c["delta"] for c in cells} == {100.0, 0.0}


def test_cells_below_min_pairs_are_dropped():
  scores = {("node_degree", 0.5, "a", "none"): 1.0,
            ("node_degree", 0.5, "a", "degree"): 0.0}
  assert abl.cells_from_scores(scores, min_pairs=10) == []


# --- statistics ------------------------------------------------------------
#
# scipy is not installed in the envs this project runs in (see CLAUDE.md), so
# `pearson` and `ols` are hand-rolled and need pinning against known values.

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


def test_fit_line_recovers_slope_and_intercept():
  slope, intercept = abl.fit_line([0, 1, 2, 3], [1, 3, 5, 7])
  assert slope == pytest.approx(2.0)
  assert intercept == pytest.approx(1.0)


def test_ols_recovers_an_exact_plane():
  rows = [([x, z], 3.0 + 2.0 * x - 1.0 * z)
          for x in (0, 1, 2, 3) for z in (0, 1, 2)]
  terms = dict((name, coef) for name, coef, _, _ in
               abl.ols(rows, ["x", "z"]))
  assert terms["intercept"] == pytest.approx(3.0, abs=1e-9)
  assert terms["x"] == pytest.approx(2.0, abs=1e-9)
  assert terms["z"] == pytest.approx(-1.0, abs=1e-9)


def test_ols_zeroes_the_coefficient_of_an_irrelevant_term():
  # y depends on x only, so z's coefficient must be 0 even though z varies.
  # Its t is not asserted: the fit is exact here, so the residual variance and
  # the standard error are both ~0 and their ratio is numerical noise. That is
  # the reason the paper's regression is read off the coefficients and their
  # p-values on real, noisy cells rather than on a constructed case.
  rows = [([x, z], 2.0 * x) for x in (0, 1, 2, 3, 4) for z in (0, 1, 2, 3)]
  terms = {name: coef for name, coef, _, _ in abl.ols(rows, ["x", "z"])}
  assert terms["x"] == pytest.approx(2.0, abs=1e-9)
  assert terms["z"] == pytest.approx(0.0, abs=1e-9)


# --- cross-fitted cells and cluster bootstrap (TEST 5) ----------------------

def _synthetic_scores(n=40, seed=0):
  """40 paired instances, `none` and `degree`, with real per-pair noise so
  the two folds are not byte-identical to each other.
  """
  import random
  rng = random.Random(seed)
  scores = {}
  for i in range(n):
    iid = f"g{i}"
    b = 1.0 if rng.random() < 0.5 else 0.0
    v = 1.0 if rng.random() < 0.8 else 0.0
    scores[("node_degree", 0.5, iid, "none")] = b
    scores[("node_degree", 0.5, iid, "degree")] = v
  return scores


def test_crossfit_produces_two_disjoint_fold_assignments_per_cell():
  scores = _synthetic_scores()
  cells = abl.cells_from_scores_crossfit(scores, min_pairs=4)
  assert {c["fold"] for c in cells} == {"a>b", "b>a"}
  assert len(cells) == 2
  a_over_b, b_over_a = (c for c in cells if c["fold"] == "a>b"), \
                       (c for c in cells if c["fold"] == "b>a")
  # every pair used for the *_"a>b"_ cell's baseline is <n> the OTHER cell's
  # delta fold; checked indirectly via the fold sizes summing to n.
  for cell in cells:
    assert cell["n_baseline_fold"] + cell["n_delta_fold"] == 40


def test_crossfit_is_deterministic_across_calls():
  scores = _synthetic_scores()
  first = abl.cells_from_scores_crossfit(scores, min_pairs=4)
  second = abl.cells_from_scores_crossfit(scores, min_pairs=4)
  assert first == second


def test_crossfit_drops_cells_with_a_too_small_fold():
  # 5 pairs total is enough for cells_from_scores's min_pairs=4, but with a
  # coin-flip 2-way split neither fold reliably clears min_pairs // 2 = 2.
  scores = {}
  for i in range(5):
    scores[("node_degree", 0.5, f"g{i}", "none")] = 0.0
    scores[("node_degree", 0.5, f"g{i}", "degree")] = 1.0
  # min_pairs=100 forces both folds below the len(fold) < min_pairs // 2 gate.
  assert abl.cells_from_scores_crossfit(scores, min_pairs=100) == []
