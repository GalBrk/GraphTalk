"""Computes the shortcut-ceiling table: what a primer-only solver scores.

7 conditions x 6 tasks x 3 rungs, plus theorem coverage, the in-sample inflation
measurement, and the exact enumeration bound on small graphs. No model queries --
this is plain Python over graphs the vendored generator produces, and the point
of it is to say which cells are worth spending cluster time on.

Read the table before committing any cluster time. Cells where the shortcut
already scores ~100% are decided before a model runs; cells with real headroom
between baseline and shortcut are where a model run buys information.

  PYTHONPATH=. .venv/bin/python scripts/shortcut_table.py --graphs 500
"""

import argparse
import sys

from graphtalk import primers
from graphtalk import shortcuts

BAR = "-" * 78


def print_cells(cells, split):
  """One block per task: conditions down the side, rungs across."""
  print(f"\n{BAR}\nSHORTCUT SCORES")
  print(f"fitted on seed {split.fit_seed}, scored on seed {split.test_seed}, "
        f"{split.size} graphs each")
  print("base = majority-class baseline; r1/r2/r3 = shortcut score by rung")
  print("'decided' means the shortcut already scores >= 99%: a model run there")
  print("adds nothing, because the comparison is settled before it starts.")

  for task in shortcuts.TASKS:
    print(f"\n{task}")
    print(f"  {'condition':<12} {'base':>7} {'r1':>8} {'r2':>8} {'r3':>8}   gap(r3)")
    for condition in primers.CONDITIONS:
      row = [cells[(condition, task, rung)] for rung in shortcuts.RUNGS]
      base = row[0].baseline
      decided = " decided" if row[-1].shortcut >= 0.99 else ""
      gap = 100 * (row[-1].shortcut - base)
      print(
          f"  {condition:<12} {base:>6.1%} "
          + " ".join(f"{cell.shortcut:>7.1%}" for cell in row)
          + f"   {gap:>+6.1f}pp{decided}"
      )


def print_theorems(cells):
  print(f"\n{BAR}\nTHEOREM COVERAGE  (precision is 1.0 by construction, asserted in tests)")
  seen = {}
  for (condition, task, rung), cell in cells.items():
    if rung != 3:
      continue
    for stat in cell.theorems:
      if stat["fired"]:
        key = (stat["rule"], condition)
        seen[key] = max(seen.get(key, 0.0), stat["coverage"])
  print(f"  {'rule':<28} {'condition':<12} {'coverage':>9}")
  for (rule, condition), coverage in sorted(
      seen.items(), key=lambda kv: (-kv[1], kv[0])
  ):
    print(f"  {rule:<28} {condition:<12} {coverage:>8.1%}")


def print_rules(cells):
  print(f"\n{BAR}\nHEURISTIC AND FITTED RULES  (rung 3, where they fire at all)")
  print(f"  {'rule':<34} {'condition':<12} {'cover':>7} {'acc':>7}")
  rows = []
  for (condition, task, rung), cell in cells.items():
    if rung != 3:
      continue
    for stat in cell.rules:
      if stat["coverage"] > 0 and stat["accuracy"] is not None:
        rows.append((stat["rule"], condition, stat["coverage"], stat["accuracy"]))
  for rule, condition, coverage, accuracy in sorted(
      set(rows), key=lambda r: (r[0], -r[3])
  ):
    print(f"  {rule:<34} {condition:<12} {coverage:>6.1%} {accuracy:>6.1%}")


def print_inflation(split, fit_graphs, test_graphs):
  print(f"\n{BAR}\nWHY THE TRAIN/TEST SPLIT IS NOT PEDANTRY")
  print("  A fitted rule scored on the graphs it was fitted on reports a bar that")
  print("  is too high, which makes a model look like it failed to reach a")
  print("  threshold that was never really there.\n")
  print(f"  {'rule':<26} {'rows':>6} {'in-sample':>11} {'honest':>9} {'inflation':>11}")
  for name in ("cycle_lookup_c", "cycle_lookup_c_n"):
    rule = next(r for r in shortcuts.FITTED if r.name == name)
    result = shortcuts.inflation(
        "components", rule.task, 3, rule, fit_graphs, test_graphs
    )
    table_rows = result["table_rows"]
    delta = 100 * (result["in_sample"] - result["out_of_sample"])
    print(
        f"  {name:<26} {table_rows if table_rows else '-':>6} "
        f"{result['in_sample']:>10.1%} {result['out_of_sample']:>8.1%} "
        f"{delta:>+9.1f}pp"
    )
  print("\n  More table rows relative to the data means more memorising. The")
  print("  direction of the error is the dangerous one.")


def best_on_small(task, small, fit_graphs):
  """Our strongest solver, scored on the same small graphs the bound covers.

  Scoring on the whole corpus would compare against a bound computed elsewhere.
  The query draw lines up because `exact_island` advances its generator only on
  the graphs it keeps, in the same order.
  """
  best = 0.0
  for condition in ("degree", "all"):
    for rung in shortcuts.RUNGS:
      fit_rows = shortcuts.build_rows(fit_graphs, condition, task, rung, seed=5)
      rows = shortcuts.build_rows(small, condition, task, rung, seed=5)
      fitted = shortcuts.rank_rules(
          shortcuts.fit_rules(
              shortcuts.HEURISTICS + shortcuts.FITTED, fit_rows, task
          ),
          fit_rows,
          task,
      )
      fallback = shortcuts.majority_answer([gold for _, gold in fit_rows])
      best = max(best, shortcuts.score_solver(rows, task, fitted, fallback))
  return best


def worst_posterior_excess(task, test_graphs, fit_graphs):
  """How far our solver ever beats the Bayes rule in posterior mass.

  This is the soundness check, and it is deliberately not the accuracy
  comparison printed alongside it. Both `bayes` and `our best` below are scored
  against the single graph that happened to be drawn, and at these row counts
  that is noisy enough for a correct rule to come out above the bound by luck --
  it did exactly that for the two reconstruction theorems. Bayes maximises the
  posterior share pointwise, so an excess here cannot be luck.
  """
  worst = -1.0
  for condition in ("degree", "all"):
    for rung in shortcuts.RUNGS:
      fit_rows = shortcuts.build_rows(fit_graphs, condition, task, rung, seed=5)
      fitted = shortcuts.rank_rules(
          shortcuts.fit_rules(
              shortcuts.HEURISTICS + shortcuts.FITTED, fit_rows, task
          ),
          fit_rows,
          task,
      )
      fallback = shortcuts.majority_answer([gold for _, gold in fit_rows])
      result = shortcuts.island_posterior(
          test_graphs, condition, task,
          lambda ctx: shortcuts.solve_context(ctx, task, fitted, fallback),
          rung=rung,
      )
      if result["worst_excess"] is not None:
        worst = max(worst, result["worst_excess"])
  return worst


def print_island(test_graphs, fit_graphs):
  print(f"\n{BAR}\nEXACT BOUND ON SMALL GRAPHS  (n <= 6, every consistent graph enumerated)")
  print("  Conditions on the stated degrees, so it bounds the degree/all arms.")
  print("  determined = answer identical across all consistent graphs")
  print("  bayes      = exact ceiling for ANY primer-only solver")
  print("  our best   = strongest hand-written solver, on these same rows")
  print("  excess     = worst pointwise posterior excess over Bayes; > 0 is a bug\n")
  print(f"  {'task':<17} {'rows':>5} {'determined':>11} {'bayes':>7} "
        f"{'our best':>9} {'excess':>8}  verdict")
  for task in shortcuts.TASKS:
    result = shortcuts.exact_island(test_graphs, task)
    if not result["rows"]:
      continue
    small = [g for g in test_graphs if g.number_of_nodes() <= 6]
    ours = best_on_small(task, small, fit_graphs)
    bayes = result["bayes_optimal"]
    excess = worst_posterior_excess(task, test_graphs, fit_graphs)
    if excess > 1e-9:
      verdict = "ABOVE BOUND -- rule reads something unstated"
    elif bayes - ours > 0.05:
      verdict = f"gap {100*(bayes-ours):.0f}pp: rules may remain unfound"
    else:
      verdict = "no gap worth chasing"
    print(
        f"  {task:<17} {result['rows']:>5} {result['determined']:>10.1%} "
        f"{bayes:>6.1%} {ours:>8.1%} {excess:>+8.4f}  {verdict}"
    )
  print(f"\n  (small graphs are {result['coverage']:.1%} of rows)")
  print("  The gap column is a single query draw over a few dozen rows and carries")
  print("  a 3-6 point standard deviation, so read it as a hint about where to look")
  print("  for rules, never as a verdict. Only the excess column is exact.")


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--graphs", type=int, default=500)
  parser.add_argument("--fit-seed", type=int, default=999)
  parser.add_argument("--test-seed", type=int, default=1234)
  args = parser.parse_args(argv)

  split = shortcuts.Split(
      fit_seed=args.fit_seed, test_seed=args.test_seed, size=args.graphs
  )
  fit_graphs, test_graphs = split.fit_graphs(), split.test_graphs()

  cells = {}
  for condition in primers.CONDITIONS:
    # The primer depends only on the condition, so render once and reuse it
    # across all six tasks and all three rungs.
    fit_parsed = shortcuts.parse_corpus(fit_graphs, condition)
    test_parsed = shortcuts.parse_corpus(test_graphs, condition)
    for task in shortcuts.TASKS:
      for rung in shortcuts.RUNGS:
        cells[(condition, task, rung)] = shortcuts.score_cell(
            condition, task, rung, fit_graphs, test_graphs,
            fit_parsed=fit_parsed, test_parsed=test_parsed,
        )
    print(f"  ... {condition}", file=sys.stderr)

  print_cells(cells, split)
  print_theorems(cells)
  print_rules(cells)
  print_inflation(split, fit_graphs, test_graphs)
  print_island(test_graphs, fit_graphs)
  print(f"\n{BAR}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
