"""Refits the shortcut bars at n=40, per density level.

`shortcuts.json` (from `shortcut_table.py`) is fit and scored on the vendored
generator's 5-19 node graphs, which the paper's `_long.tex` itself says "do
not transfer" to the n=40 corpus the models are actually evaluated on -- see
superseded/docs/paper-revision-handoff.md. This refits the same rung-3 solver (theorem +
heuristic + fitted rules, disjoint fit/test graph sets) on n=40 ER graphs at
each density densfull40/densfull40hi actually used, so route-vs-no-route
classification and any bar-relative Δ can be computed honestly.

  PYTHONPATH=. python scripts/shortcut_table_n40.py --json shortcuts_n40.json
"""

import argparse
import json

from graphtalk import primers
from graphtalk import shortcuts

# densfull40's 4 levels + densfull40hi's 3.
DENSITIES = (0.10, 0.20, 0.35, 0.50, 0.65, 0.75, 0.85)


def flatten(per_density: dict) -> dict:
  """Mean bar per "task/condition" across densities, for callers (route
  classification in analyze_baseline_law.py) that key on one flat bar per
  cell rather than per (task, condition, density). Averaging rather than
  taking a single density keeps the classification density-independent,
  the same way the published-split shortcuts.json is graph-size-independent
  within its own corpus.
  """
  keys = {k for bars in per_density.values() for k in bars}
  return {k: sum(bars.get(k, 0.0) for bars in per_density.values()) / len(per_density)
          for k in keys}


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--graphs", type=int, default=300)
  parser.add_argument("--fit-seed", type=int, default=555_555)
  parser.add_argument("--test-seed", type=int, default=777_777)
  parser.add_argument("--json", default="shortcuts_n40.json")
  parser.add_argument("--json-rung", type=int, default=3, choices=shortcuts.RUNGS)
  parser.add_argument("--flat-json", default=None,
                      help="also write the density-averaged flat table "
                           "(one bar per task/condition) here, for callers "
                           "like analyze_baseline_law.py's route split that "
                           "key on a single bar per cell")
  args = parser.parse_args(argv)

  out = {}  # density (str) -> "task/condition" -> shortcut
  print(f"{'density':>8}  {'task':<16} {'condition':<12} {'base':>7} {'r3':>8}   gap")
  for density in DENSITIES:
    fit_graphs = shortcuts.generate_n40_corpus(args.graphs, args.fit_seed, density)
    test_graphs = shortcuts.generate_n40_corpus(args.graphs, args.test_seed, density)
    bars = {}
    for condition in primers.CONDITIONS:
      fit_parsed = shortcuts.parse_corpus(fit_graphs, condition)
      test_parsed = shortcuts.parse_corpus(test_graphs, condition)
      for task in shortcuts.TASKS:
        cell = shortcuts.score_cell(
            condition, task, args.json_rung, fit_graphs, test_graphs,
            fit_parsed=fit_parsed, test_parsed=test_parsed,
        )
        bars[f"{task}/{condition}"] = cell.shortcut
        gap = 100 * (cell.shortcut - cell.baseline)
        print(f"{density:>8.2f}  {task:<16} {condition:<12} {cell.baseline:>6.1%} "
              f"{cell.shortcut:>7.1%}   {gap:>+6.1f}pp")
    out[f"{density:g}"] = bars

  with open(args.json, "w") as fh:
    json.dump(out, fh, indent=1)
  print(f"\nwrote {sum(len(v) for v in out.values())} cells "
        f"across {len(out)} densities to {args.json}")

  if args.flat_json:
    with open(args.flat_json, "w") as fh:
      json.dump(flatten(out), fh, indent=1, sort_keys=True)
    print(f"wrote the density-averaged flat table to {args.flat_json}")


if __name__ == "__main__":
  main()
