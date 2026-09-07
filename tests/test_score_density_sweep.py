"""Tests for `scripts/score_density_sweep.py`.

Two of these guard decisions that are easy to get wrong silently rather than
loudly. `test_capped_rows_are_dropped_not_scored_zero` pins the `hit_cap`
policy: a truncated generation is a budget failure, and scoring it zero would
mix the non-termination rate into accuracy, which on `ec500` was worth a full
percentage point of apparent effect. `test_pairs_only_on_shared_instances`
pins the pairing: McNemar on misaligned arms is not a weaker test, it is a
different and meaningless one.
"""

import importlib.util
import pathlib

import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "score_density_sweep.py")
_spec = importlib.util.spec_from_file_location("score_density_sweep", _SCRIPT)
score_density_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_density_sweep)


def _row(instance_id, condition, gold, response, hit_cap=False):
  return {"instance_id": instance_id, "task": "node_degree",
          "condition": condition, "gold": gold, "response": response,
          "hit_cap": hit_cap}


@pytest.mark.parametrize("instance_id, expected", [
    ("node_degree/size40/p0.35/17", 0.35),
    ("node_degree/size40/p0.1/0", 0.1),
    ("node_degree/size40/p1/0", 1.0),
    ("node_degree/size40/0", None),          # plain size sweep, no density
    ("node_degree/size40/path/3", None),     # a 'p' segment that is not a level
])
def test_density_of(instance_id, expected):
  assert score_density_sweep.density_of(instance_id) == expected


def test_blind_bar_is_the_modal_gold():
  answer, bar = score_density_sweep.blind_bar(["4", "4", "4", "7", "9"])
  assert answer == "4"
  assert bar == pytest.approx(0.6)


def test_capped_rows_are_dropped_not_scored_zero():
  """A truncated row must not depress the mean, and must stay countable."""
  rows = [
      _row("node_degree/size40/p0.2/0", "none", "7", "the degree is 7"),
      _row("node_degree/size40/p0.2/1", "none", "7", "the degree is 7"),
      _row("node_degree/size40/p0.2/2", "none", "7", "the degree is 1",
           hit_cap=True),
  ]
  cell = score_density_sweep.summarize(rows)["cells"][(0.2, "none")]
  assert cell["capped"] == 1
  assert cell["kept"] == 2
  # 1.0, not the 0.667 that scoring the capped row zero would give.
  assert cell["total"] / cell["kept"] == pytest.approx(1.0)


def test_capped_rows_do_not_enter_the_paired_arms():
  """Dropping a row must drop its *pair*, or the arms silently misalign."""
  rows = [
      _row("node_degree/size40/p0.2/0", "none", "7", "the degree is 7"),
      _row("node_degree/size40/p0.2/0", "clustering", "7", "the degree is 3",
           hit_cap=True),
      _row("node_degree/size40/p0.2/1", "none", "5", "the degree is 5"),
      _row("node_degree/size40/p0.2/1", "clustering", "5", "the degree is 5"),
  ]
  summary = score_density_sweep.summarize(rows)
  control, treatment = score_density_sweep.paired_arms(
      summary["paired"], 0.2, "clustering")
  assert (control, treatment) == ([1.0], [1.0])


def test_pairs_only_on_shared_instances():
  """An instance seen under one condition only contributes to neither arm."""
  rows = [
      _row("node_degree/size40/p0.2/0", "none", "7", "the degree is 7"),
      _row("node_degree/size40/p0.2/0", "clustering", "7", "the degree is 7"),
      _row("node_degree/size40/p0.2/1", "none", "5", "the degree is 5"),
      _row("node_degree/size40/p0.2/2", "clustering", "5", "the degree is 5"),
  ]
  summary = score_density_sweep.summarize(rows)
  control, treatment = score_density_sweep.paired_arms(
      summary["paired"], 0.2, "clustering")
  assert len(control) == len(treatment) == 1


def test_levels_do_not_pool_unless_asked():
  """Per-level arms hold only that level; `density=None` pools every level."""
  rows = []
  for level in ("0.2", "0.5"):
    for index in range(3):
      for condition in ("none", "clustering"):
        rows.append(_row(f"node_degree/size40/p{level}/{index}", condition,
                         "7", "the degree is 7"))
  paired = score_density_sweep.summarize(rows)["paired"]
  assert len(score_density_sweep.paired_arms(paired, 0.2, "clustering")[0]) == 3
  assert len(score_density_sweep.paired_arms(paired, None, "clustering")[0]) == 6
