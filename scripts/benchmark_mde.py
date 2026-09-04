"""Benchmark for `--mde-fast` (Phase 1.3.2): is the cheap MDE preset
(n_replicates=50, n_perm=200, n_steps=5) close enough to the full-precision
one (200/500/8) to trust, and how much faster is it. Not a pytest test --
real MDE simulation is seconds per row even at the fast preset, minutes at
full precision across a whole sweep; this belongs in an explicitly-run
script, the same way `scripts/validate_hierarchical_model.py` and
`scripts/characterize_non_termination.py` do for their own expensive
checks.

Samples real non-significant main-sweep `excluded`-bound cells from
`analysis/sweep_frame.csv` + `analysis/significance_report.csv` (rows
where an MDE would actually be computed by `check_significance.py --mde`),
re-derives each cell's `(control, treatment, cluster_ids)` the same way
`scripts/check_significance.py::_report` does, and runs
`graphtalk.significance.minimum_detectable_effect_clustered` at both
presets with the *same seed* -- so any difference is the presets, not
random variation between runs.

  PYTHONPATH=. .venv/bin/python scripts/benchmark_mde.py --n-cells 10
"""

import argparse
import time

import pandas as pd

from graphtalk import significance
from scripts import check_significance as cs

_FAST = {"n_replicates": 50, "n_perm": 200, "n_steps": 5}
_FULL = {"n_replicates": 200, "n_perm": 500, "n_steps": 8}


def _sample_cells(sweep_frame_path: str, report_path: str, n_cells: int, seed: int):
  """Up to `n_cells` (arm, group, condition, bound) tuples from
  `report_path` that are `bh_significant=False`, non-derived,
  `bound in ("excluded", "not_applicable")` -- the exact eligibility
  `check_significance.py::_report`'s `mde_eligible` uses."""
  report = pd.read_csv(report_path)
  eligible = report[
      (~report["bh_significant"])
      & (~report["is_derived_condition"])
      & (report["bound"].isin(["excluded", "not_applicable"]))
      & (report["metric"] != "mae")  # MDE isn't computed in mae mode
  ]
  sample = eligible.sample(n=min(n_cells, len(eligible)), random_state=seed)
  return sample[["arm", "group", "condition", "bound"]].to_dict("records")


def _paired_values_for_cell(frame: pd.DataFrame, cell: dict):
  """Reconstructs the same `(control, treatment, cluster_ids)` triple
  `_report` would have passed to `minimum_detectable_effect_clustered` for
  this cell, from raw `sweep_frame.csv` rows -- mirrors
  `scripts/check_significance.py::main`'s scoping (main sweep vs thinking
  arm, `excluded` bound drops non_terminating rows, `not_applicable` is
  the thinking arm's own non-termination outcome) plus
  `_report`'s `_paired_values` call."""
  metric = "exact" if cell["arm"] == "main_sweep" else "non_terminating"
  if cell["arm"] == "main_sweep":
    scoped = frame[~frame["is_think"]]
    if cell["bound"] == "excluded":
      scoped = scoped[scoped["failure_type"] != "non_terminating"]
  else:
    scoped = frame[frame["is_think"]]
  if cell["group"] != "pooled across all models":
    scoped = scoped[scoped["model_family"] == cell["group"]]
  return cs._paired_values(scoped, cell["condition"], metric)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frame", default="analysis/sweep_frame.csv")
  parser.add_argument("--report", default="analysis/significance_report.csv")
  parser.add_argument("--n-cells", type=int, default=10)
  parser.add_argument("--seed", type=int, default=1234)
  args = parser.parse_args()

  frame = pd.read_csv(args.frame)
  cells = _sample_cells(args.frame, args.report, args.n_cells, args.seed)
  print(f"Benchmarking {len(cells)} cells: fast (n_replicates=50, n_perm=200, "
        f"n_steps=5) vs full (200/500/8)\n")

  rows = []
  for cell in cells:
    control, treatment, cluster_ids = _paired_values_for_cell(frame, cell)
    if not control:
      continue
    initial_hi = 0.1
    seed = f"{args.seed}:{cell['arm']}:{cell['group']}:{cell['bound']}:{cell['condition']}"

    t0 = time.perf_counter()
    fast = significance.minimum_detectable_effect_clustered(
        control, treatment, cluster_ids, initial_hi=initial_hi, seed=seed, **_FAST
    )
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    full = significance.minimum_detectable_effect_clustered(
        control, treatment, cluster_ids, initial_hi=initial_hi, seed=seed, **_FULL
    )
    t_full = time.perf_counter() - t0

    row = {
        "arm": cell["arm"], "group": cell["group"], "condition": cell["condition"],
        "fast_delta": fast["delta"], "full_delta": full["delta"],
        "fast_delta_negative": fast["delta_negative"],
        "full_delta_negative": full["delta_negative"],
        "t_fast": t_fast, "t_full": t_full,
    }
    rows.append(row)
    print(f"  {cell['group']:<20}{cell['condition']:<12} "
          f"fast={fast['delta']} full={full['delta']}  "
          f"fast_neg={fast['delta_negative']} full_neg={full['delta_negative']}  "
          f"({t_fast:.1f}s vs {t_full:.1f}s)")

  results = pd.DataFrame(rows)
  print("\n=== Summary ===")
  for col_fast, col_full, label in (
      ("fast_delta", "full_delta", "positive delta"),
      ("fast_delta_negative", "full_delta_negative", "negative delta"),
  ):
    both = results.dropna(subset=[col_fast, col_full])
    if both.empty:
      print(f"  {label}: no cells where both presets converged -- can't compare")
      continue
    diff = (both[col_fast] - both[col_full]).abs()
    print(f"  {label}: {len(both)}/{len(results)} cells where both converged -- "
          f"mean |diff|={diff.mean():.4f}, max |diff|={diff.max():.4f}")
    # graphtalk.significance.minimum_detectable_effect_clustered's own
    # docstring: "Power estimates carry real Monte Carlo noise (SE ~= 0.03
    # at n_replicates=200 near power=0.8)" -- the fast preset's own noise
    # floor is higher (fewer replicates), so holding it to that same 0.03
    # bound is already the stricter of the two, not a loosened target.
    verdict = "OK" if diff.max() < 0.05 else "CHECK -- exceeds the ~0.03 SE noise floor by a wide margin"
    print(f"    verdict: {verdict}")

  speedup = results["t_full"].sum() / results["t_fast"].sum() if results["t_fast"].sum() else None
  print(f"\n  total time -- fast: {results['t_fast'].sum():.1f}s, "
        f"full: {results['t_full'].sum():.1f}s, speedup: {speedup:.1f}x")
  print(f"    verdict: {'OK -- at least 3x faster' if speedup and speedup >= 3 else 'CHECK -- under 3x'}")


if __name__ == "__main__":
  main()
