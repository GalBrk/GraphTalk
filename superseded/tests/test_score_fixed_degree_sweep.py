"""Tests for `superseded/scripts/score_fixed_degree_sweep.py`.

The one thing this script exists to get right that `score_density_sweep.py`
gets wrong on this design: two cells sharing a density value must not be pooled
together just because `density_of` would collapse them. `test_cell_of_disambiguates_shared_density`
and `test_shared_density_cells_stay_separate` pin that directly.
"""

import importlib.util
import pathlib

import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "score_fixed_degree_sweep.py")
_spec = importlib.util.spec_from_file_location(
    "score_fixed_degree_sweep", _SCRIPT)
score_fixed_degree_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_fixed_degree_sweep)


def _row(instance_id, condition, gold, response, hit_cap=False):
  return {"instance_id": instance_id, "task": "node_degree",
          "condition": condition, "gold": gold, "response": response,
          "hit_cap": hit_cap}


@pytest.mark.parametrize("instance_id, expected", [
    ("node_degree/size80/p0.101/17", (80, 0.101)),
    ("node_degree/size160/p0.101/17", (160, 0.101)),
    ("node_degree/size40/p0.1/0", (40, 0.1)),
    ("node_degree/size40/0", None),          # plain size sweep, no density
    ("node_degree/size40/path/3", None),     # a 'p' segment that is not a level
])
def test_cell_of(instance_id, expected):
  assert score_fixed_degree_sweep.cell_of(instance_id) == expected


def test_cell_of_disambiguates_shared_density():
  """size=80/p=0.101 and size=160/p=0.101 are distinct cells in the real
  fixdeg design (mean degree 8 and 16 respectively) despite sharing a density."""
  a = score_fixed_degree_sweep.cell_of("node_degree/size80/p0.101/3")
  b = score_fixed_degree_sweep.cell_of("node_degree/size160/p0.101/3")
  assert a != b
  assert a[1] == b[1] == 0.101


def test_mean_degree_recovers_the_target_block():
  assert score_fixed_degree_sweep.mean_degree((20, 0.421)) == 8
  assert score_fixed_degree_sweep.mean_degree((160, 0.101)) == 16


def test_blind_bar_is_the_modal_gold():
  answer, bar = score_fixed_degree_sweep.blind_bar(["4", "4", "4", "7", "9"])
  assert answer == "4"
  assert bar == pytest.approx(0.6)


def test_shared_density_cells_stay_separate():
  """The collision score_density_sweep.py would hit: two cells at p=0.101 must
  not be pooled into one row."""
  rows = [
      _row("node_degree/size80/p0.101/0", "none", "8", "the degree is 8"),
      _row("node_degree/size160/p0.101/0", "none", "16", "the degree is 3"),
  ]
  cells = score_fixed_degree_sweep.summarize(rows)["cells"]
  assert cells[((80, 0.101), "none")]["total"] == pytest.approx(1.0)
  assert cells[((160, 0.101), "none")]["total"] == pytest.approx(0.0)


def test_capped_rows_are_dropped_not_scored_zero():
  rows = [
      _row("node_degree/size40/p0.2/0", "none", "7", "the degree is 7"),
      _row("node_degree/size40/p0.2/1", "none", "7", "the degree is 7"),
      _row("node_degree/size40/p0.2/2", "none", "7", "the degree is 1",
           hit_cap=True),
  ]
  cell = score_fixed_degree_sweep.summarize(rows)["cells"][((40, 0.2), "none")]
  assert cell["capped"] == 1
  assert cell["kept"] == 2
  assert cell["total"] / cell["kept"] == pytest.approx(1.0)


def test_pairs_only_on_shared_instances():
  rows = [
      _row("node_degree/size40/p0.2/0", "none", "7", "the degree is 7"),
      _row("node_degree/size40/p0.2/0", "clustering", "7", "the degree is 7"),
      _row("node_degree/size40/p0.2/1", "none", "5", "the degree is 5"),
      _row("node_degree/size40/p0.2/2", "clustering", "5", "the degree is 5"),
  ]
  summary = score_fixed_degree_sweep.summarize(rows)
  control, treatment = score_fixed_degree_sweep.paired_arms(
      summary["paired"], {(40, 0.2)}, "clustering")
  assert len(control) == len(treatment) == 1


def test_paired_arms_pools_a_mean_degree_block():
  """A block-level pool must combine every size at that mean degree and no
  other, which is what the 'pooled per mean-degree block' report line needs."""
  rows = [
      _row("node_degree/size20/p0.421/0", "none", "8", "the degree is 8"),
      _row("node_degree/size20/p0.421/0", "filler", "8", "the degree is 8"),
      _row("node_degree/size40/p0.205/0", "none", "8", "the degree is 8"),
      _row("node_degree/size40/p0.205/0", "filler", "8", "the degree is 1"),
      # A d~16 cell that must NOT enter the d~8 block pool.
      _row("node_degree/size20/p0.842/0", "none", "16", "the degree is 16"),
      _row("node_degree/size20/p0.842/0", "filler", "16", "the degree is 1"),
  ]
  summary = score_fixed_degree_sweep.summarize(rows)
  d8_block = {c for c in ((20, 0.421), (40, 0.205))}
  control, treatment = score_fixed_degree_sweep.paired_arms(
      summary["paired"], d8_block, "filler")
  assert len(control) == len(treatment) == 2


def test_paired_arms_none_pools_every_cell():
  rows = [
      _row("node_degree/size20/p0.421/0", "none", "8", "the degree is 8"),
      _row("node_degree/size20/p0.421/0", "filler", "8", "the degree is 8"),
      _row("node_degree/size160/p0.101/0", "none", "16", "the degree is 16"),
      _row("node_degree/size160/p0.101/0", "filler", "16", "the degree is 1"),
  ]
  summary = score_fixed_degree_sweep.summarize(rows)
  control, treatment = score_fixed_degree_sweep.paired_arms(
      summary["paired"], None, "filler")
  assert len(control) == len(treatment) == 2
