"""Tests of the two analyze_baseline_law.py helpers that only the superseded
cross-fit and ceiling tests used; they run against the archived full copy."""
import importlib.util
import pathlib

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "analyze_baseline_law.py")
_spec = importlib.util.spec_from_file_location("analyze_baseline_law", _SCRIPT)
abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abl)


def test_cluster_bootstrap_ci_brackets_a_strong_positive_slope():
  # baseline and delta move together within each block by construction, so
  # the bootstrap CI on the slope must sit entirely above zero.
  cells = []
  for block in range(6):
    for i in range(5):
      b = 0.1 * i
      cells.append({"arm": "x", "task": "t", "density": block,
                    "baseline": b, "delta": 50.0 * b})
  boot = abl.cluster_bootstrap_r_slope(
      cells, block_key=lambda c: (c["arm"], c["task"], c["density"]),
      n_boot=500, seed=0,
  )
  assert boot["slope_ci"][0] > 0
  assert boot["r_ci"][0] > 0


def test_ceiling_by_arm_averages_only_the_none_condition():
  scores_by_arm = {
      "armA": {
          ("t", None, "g1", "none"): 1.0,
          ("t", None, "g2", "none"): 0.0,
          ("t", None, "g1", "degree"): 1.0,   # not `none`; must be ignored
          ("t", None, "g3", "none"): None,    # hit_cap; must be dropped
      },
      "armB": {
          ("t", None, "g1", "degree"): 1.0,   # no `none` rows at all
      },
  }
  table = abl.ceiling_by_arm(scores_by_arm)
  assert table == {"armA": (0.5, 2)}
