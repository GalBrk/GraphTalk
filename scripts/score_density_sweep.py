"""Score a fixed-size density sweep: accuracy and paired tests per density level.

`scripts/score_sweep.py` groups by (task, style), which is the right grouping for
the tracked corpus but collapses exactly the variable a density sweep manipulates.
This script groups by the density level encoded in the `instance_id`
(`node_degree/size40/p0.35/17` -> 0.35), which `build_size_sweep.py --densities`
writes there so that two levels can never collide into one resume key.

What it reports, per (density, condition):

  * mean score under the metric `scoring.score_one` names for the task;
  * the **blind bar** for that level -- the score of always answering the modal
    gold, i.e. what a solver that never reads the graph gets. This is computed
    per level rather than taken from `shortcuts.json` because the gold
    distribution moves with density: at n=40 the modal-degree baseline runs 0.23
    at p=0.10 down to 0.15 at p=0.50, so a single corpus-wide bar would flatter
    the sparse levels and penalise the dense ones;
  * mean absolute error on the integer tasks, which separates "drifting off by
    one" from "collapsed";
  * exact McNemar against the control, per level and pooled across levels, on
    rows paired by `instance_id`.

`hit_cap` rows are dropped rather than scored zero -- a truncated generation is a
budget failure, not a wrong answer, and counting it as one confounds the
non-termination rate with accuracy. The count dropped is reported per cell so
the choice stays visible; see `docs/primer-effects-and-power.md`, "If you only
read one thing", rule 2.

  PYTHONPATH=. python scripts/score_density_sweep.py \
      --responses "runs/qwen3-1.7b.degdens40.shard*of5.jsonl"
"""

import argparse
import collections
import csv
import glob
import json
import random
import statistics

from graphtalk import scoring
from graphtalk import significance

CONTROL = "none"

# Same two presets `check_significance.py` offers: a fast approximate default
# (benchmarked there against full precision, staying within the MDE search's
# own Monte Carlo noise floor) and a slower one for a number meant to be
# quoted, via --mde.
_MDE_FAST = {"n_replicates": 50, "n_perm": 200, "n_steps": 5}
_MDE_FULL = {"n_replicates": 200, "n_perm": 500, "n_steps": 8}


def density_of(instance_id: str) -> float | None:
  """The density level `build_size_sweep.py` encoded in an instance id.

  Returns None for ids with no density segment (a plain size sweep), so a mixed
  set of files degrades to a single unlabelled group rather than raising.
  """
  for part in instance_id.split("/"):
    if len(part) > 1 and part[0] == "p":
      try:
        return float(part[1:])
      except ValueError:
        continue
  return None


def load(patterns) -> list[dict]:
  records = []
  for pattern in patterns:
    paths = sorted(glob.glob(pattern)) or [pattern]
    for path in paths:
      with open(path) as handle:
        for line in handle:
          if line.strip():
            records.append(json.loads(line))
  return records


def blind_bar(golds) -> tuple[str, float]:
  """Best constant answer over these rows, and what it scores.

  Exact-match on the gold string, which is the right comparison for the integer
  tasks a density sweep can use. It is a *lower* bound on a clever blind solver
  and an upper bound on chance, which is the interval a result has to clear.
  """
  counts = collections.Counter(golds)
  answer, hits = counts.most_common(1)[0]
  return answer, hits / len(golds)


def summarize(records) -> dict:
  """Per-cell aggregates and per-pair scores, hit_cap rows dropped."""
  cells = collections.defaultdict(
      lambda: {"kept": 0, "capped": 0, "total": 0.0,
               "abs_error": 0.0, "n_error": 0, "parsed": 0}
  )
  paired = collections.defaultdict(dict)   # (density, instance_id) -> cond -> score
  golds = collections.defaultdict(list)    # density -> golds
  for record in records:
    density = density_of(record["instance_id"])
    cell = cells[(density, record["condition"])]
    if record.get("hit_cap"):
      cell["capped"] += 1
      continue
    result = scoring.score_one(
        scoring.extract_answer(record["response"], record["task"]),
        record["gold"], record["task"],
    )
    cell["kept"] += 1
    cell["total"] += result["primary"]
    cell["parsed"] += int(result["parsed"])
    if result["absolute_error"] is not None:
      cell["abs_error"] += result["absolute_error"]
      cell["n_error"] += 1
    paired[(density, record["instance_id"])][record["condition"]] = result["primary"]
    golds[density].append(record["gold"])
  return {"cells": cells, "paired": paired, "golds": golds}


def paired_arms(paired, density, condition, control=CONTROL):
  """Aligned (control, treatment) hit vectors over instances holding both.

  `density=None` pools every level. Pairing is on `instance_id`, which
  `build_size_sweep.py` keeps identical across conditions -- same graph, same
  queried node, same gold -- so this really is a within-graph comparison.
  """
  control_hits, treatment_hits = [], []
  for (level, _), by_condition in paired.items():
    if density is not None and level != density:
      continue
    if control in by_condition and condition in by_condition:
      control_hits.append(by_condition[control])
      treatment_hits.append(by_condition[condition])
  return control_hits, treatment_hits


def _slope(xs, ys) -> float:
  """OLS slope of `ys` on `xs`, which for a two-point-or-more design is the
  per-unit-density change in the paired difference."""
  mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
  numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
  denominator = sum((x - mean_x) ** 2 for x in xs)
  return numerator / denominator if denominator else float("nan")


def trend_test(paired, condition, levels=None, control=CONTROL,
               draws=20000, seed=0) -> dict:
  """Does the primer's benefit change with density? Permutation test on a slope.

  Per-cell McNemar answers "is any single level significant", which is not the
  hypothesis `docs/difficulty-scaling.md` states: it claims the benefit *grows*
  with density, and that is a statement about a slope. Fitting it directly also
  uses every graph at once instead of splitting them into four underpowered
  cells.

  The null is that density carries no information about the paired difference,
  so density labels are permuted across graphs. Under permutation `mean(x)`,
  `var(x)` and `mean(y)` are all invariant, so the slope is a monotone function
  of `sum(x*y)` and only that sum has to be recomputed per draw -- which is what
  makes 20,000 draws cheap enough to run by default.
  """
  xs, ys = [], []
  for (density, _), by_condition in paired.items():
    if levels is not None and density not in levels:
      continue
    if control in by_condition and condition in by_condition:
      xs.append(density)
      ys.append(by_condition[condition] - by_condition[control])
  if len(set(xs)) < 2:
    return {"slope": float("nan"), "p_value": 1.0, "n": len(xs)}

  observed = _slope(xs, ys)
  n = len(xs)
  centre = n * statistics.mean(xs) * statistics.mean(ys)
  target = abs(sum(x * y for x, y in zip(xs, ys)) - centre)
  rng = random.Random(seed)
  shuffled = list(xs)
  at_least_as_extreme = 0
  for _ in range(draws):
    rng.shuffle(shuffled)
    if abs(sum(x * y for x, y in zip(shuffled, ys)) - centre) >= target:
      at_least_as_extreme += 1
  # +1 to both parts: the observed arrangement is itself a valid permutation, so
  # a p-value of exactly 0 is not attainable and should not be reported.
  return {"slope": observed,
          "p_value": (at_least_as_extreme + 1) / (draws + 1),
          "n": n}


def _mde_for_row(control_hits, treatment_hits, mde_settings, seed) -> dict:
  """MDE for one pooled/per-level McNemar row, on the same paired hit
  vectors the test itself used.

  No natural clustering unit exists here the way it does in
  `check_significance.py`'s main sweep (where six tasks share one graph):
  a density-sweep pair is one graph at one density, contributing exactly one
  row to this comparison, never repeated -- so each pair is its own cluster,
  and `paired_permutation_test_clustered`'s machinery correctly reduces to
  the unclustered case underneath.
  """
  cluster_ids = list(range(len(control_hits)))
  return significance.minimum_detectable_effect_clustered(
      control_hits, treatment_hits, cluster_ids, initial_hi=0.05,
      seed=seed, **mde_settings,
  )


def report(summary, control=CONTROL, trend_max=None, draws=20000,
           mde_eligible=True, mde_settings=_MDE_FAST, mde_seed=0,
           csv_prefix=None) -> None:
  cells, paired, golds = summary["cells"], summary["paired"], summary["golds"]
  densities = sorted({d for d, _ in cells}, key=lambda d: (d is None, d))
  conditions = sorted({c for _, c in cells})
  header = "  p      " + "".join(f"{c:>15}" for c in conditions)

  print("blind bar (best constant answer, per level):")
  bars, blind_answers = {}, {}
  for density in densities:
    blind_answers[density], bars[density] = blind_bar(golds[density])
    print(f"  p={density!s:<6} modal gold {blind_answers[density]!r:>6}  "
          f"bar {bars[density]:.3f}  (n={len(golds[density])})")

  for label, value in (
      ("mean score", lambda c: c["total"] / c["kept"] if c["kept"] else float("nan")),
      ("mean |error|",
       lambda c: c["abs_error"] / c["n_error"] if c["n_error"] else float("nan")),
  ):
    print(f"\n{label} by density x condition (hit_cap dropped):")
    print(header)
    for density in densities:
      line = f"  {density!s:<7}"
      for condition in conditions:
        line += f"{value(cells[(density, condition)]):>15.3f}"
      print(line)

  for label, key in (("capped rows dropped", "capped"),
                     ("unparsable rows (scored 0, not dropped)", None)):
    print(f"\n{label} by density x condition:")
    print(header)
    for density in densities:
      line = f"  {density!s:<7}"
      for condition in conditions:
        cell = cells[(density, condition)]
        line += f"{cell['capped'] if key else cell['kept'] - cell['parsed']:>15}"
      print(line)

  cell_rows = []
  if csv_prefix:
    for density in densities:
      for condition in conditions:
        cell = cells[(density, condition)]
        cell_rows.append({
            "density": density, "condition": condition,
            "kept": cell["kept"], "capped": cell["capped"],
            "parsed": cell["parsed"], "unparsable": cell["kept"] - cell["parsed"],
            "mean_score": cell["total"] / cell["kept"] if cell["kept"] else None,
            "mean_abs_error": (
                cell["abs_error"] / cell["n_error"] if cell["n_error"] else None
            ),
            "blind_bar_answer": blind_answers.get(density),
            "blind_bar_score": bars.get(density),
        })

  def run(density):
    out = []
    for condition in conditions:
      if condition == control:
        continue
      control_hits, treatment_hits = paired_arms(paired, density, condition, control)
      if not control_hits:
        continue
      test = scoring.mcnemar(control_hits, treatment_hits)
      delta = (test["c"] - test["b"]) / len(control_hits)
      out.append((density, condition, len(control_hits), test, delta,
                  control_hits, treatment_hits))
    return out

  test_rows = []

  def show(rows, reject, family: str) -> None:
    for (density, condition, n, test, delta, control_hits,
         treatment_hits), keep in zip(rows, reject):
      level = "POOLED" if density is None else f"p={density:g}"
      mde = None
      if mde_eligible and not keep:
        row_seed = f"{mde_seed}:{family}:{level}:{condition}"
        mde = _mde_for_row(control_hits, treatment_hits, mde_settings, row_seed)
      print(f"  {level:<8} {condition:>11} - {control}: n={n:>5} "
            f"win {test['c']:>4} lose {test['b']:>4} delta {delta:+.4f} "
            f"p={test['p_value']:.4f}{'  *' if keep else ''}")
      if mde is not None:
        print(f"    MDE: delta={mde['delta']} realized={mde['realized_diff']} "
              f"({mde['note'] or 'ok'})  "
              f"delta_negative={mde['delta_negative']} "
              f"realized_negative={mde['realized_diff_negative']} "
              f"({mde['note_negative'] or 'ok'})")
      if csv_prefix:
        test_rows.append({
            "family": family, "density": density, "condition": condition,
            "n": n, "win": test["c"], "lose": test["b"], "delta": delta,
            "p_value": test["p_value"], "bh_significant": keep,
            "mde_delta": mde["delta"] if mde else None,
            "mde_realized_diff": mde["realized_diff"] if mde else None,
            "mde_note": mde["note"] if mde else None,
            "mde_delta_negative": mde["delta_negative"] if mde else None,
            "mde_realized_diff_negative":
                mde["realized_diff_negative"] if mde else None,
            "mde_note_negative": mde["note_negative"] if mde else None,
        })

  print(f"\npooled across levels, paired vs {control!r} "
        "(exact McNemar on rows sharing an instance_id):")
  pooled = run(None) if len(densities) > 1 else []
  show(pooled,
       significance.benjamini_hochberg([row[3]["p_value"] for row in pooled]),
       "pooled")

  print("\nper level (descriptive -- these are a family, correct before "
        "quoting any one):")
  per_level = [row for density in densities for row in run(density)]
  # The pooled tests above are deliberately *not* in this family: they reuse the
  # same rows, so folding them in would let the pooled p-value drag a per-level
  # p-value under the threshold on the strength of its own data. The pooled test
  # is the primary and stands on its own; the per-level rows are exploratory.
  show(per_level,
       significance.benjamini_hochberg([row[3]["p_value"] for row in per_level]),
       "per_level")
  if per_level:
    print("  (* = survives Benjamini-Hochberg at q=0.05 within its own family)")
  if mde_eligible:
    print("  (MDE printed under any row that did not survive BH -- the "
          "smallest true effect, in each direction, this row's data could "
          "reliably have detected; fast preset unless --mde was passed)")

  if len(densities) >= 2:
    print(f"\ntrend in the paired difference vs density "
          f"({draws:,} label permutations):")
    ranges = [("all levels", None)]
    if trend_max is not None:
      ranges.append((f"p <= {trend_max:g}",
                     {d for d in densities if d <= trend_max}))
    trends = []
    for name, levels in ranges:
      for condition in conditions:
        if condition == control:
          continue
        test = trend_test(paired, condition, levels, control, draws)
        # A condition absent from this range (`degree` was only run at the high
        # densities) has no slope. Printing it as nan would be noise; letting it
        # into the BH family would be worse, since a placeholder p=1.0 inflates
        # the family size and weakens every real test in it.
        if test["n"]:
          trends.append((name, condition, test))
    keep = significance.benjamini_hochberg([t["p_value"] for *_, t in trends])
    trend_rows = []
    for (name, condition, test), survives in zip(trends, keep):
      print(f"  {name:<12} {condition:>11}: slope {test['slope']:+.3f} "
            f"per unit density  p={test['p_value']:.3f}  n={test['n']:>5}"
            f"{'  *' if survives else ''}")
      if csv_prefix:
        trend_rows.append({
            "range": name, "condition": condition, "slope": test["slope"],
            "p_value": test["p_value"], "n": test["n"],
            "bh_significant": survives,
        })
    print("  (slope is the change in `primer - control` per unit of density;"
          " * = survives BH at q=0.05)")
  else:
    trend_rows = []

  if csv_prefix:
    _write_csv(f"{csv_prefix}.cells.csv", cell_rows)
    _write_csv(f"{csv_prefix}.tests.csv", test_rows)
    _write_csv(f"{csv_prefix}.trend.csv", trend_rows)
    print(f"\nwrote {csv_prefix}.cells.csv, {csv_prefix}.tests.csv, "
          f"{csv_prefix}.trend.csv")


def _write_csv(path: str, rows: list) -> None:
  """Writes `rows` (a list of same-shaped dicts) to `path`, header from the
  first row's keys. Writes nothing (not even a header) for an empty list --
  a family that reduces to nothing on this data (e.g. no trend rows for a
  single-density input) is a valid, silent outcome, not an error to paper
  over with a headerless file."""
  if not rows:
    return
  with open(path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--responses", nargs="+", required=True,
                      help="response jsonl paths or globs")
  parser.add_argument("--control", default=CONTROL)
  parser.add_argument("--trend-max", type=float, default=None,
                      help="also fit the density trend restricted to levels at "
                           "or below this density. Use it to exclude levels "
                           "where the model is at floor: averaging a dead "
                           "instrument into a live one drags any slope to zero.")
  parser.add_argument("--draws", type=int, default=20000,
                      help="label permutations for the trend test")
  parser.add_argument("--mde", action="store_true",
                      help="full-precision MDE (200/500/8 replicates/perm/"
                           "steps) instead of the default fast preset "
                           "(50/200/5) -- use for a number meant to be quoted")
  parser.add_argument("--no-mde", action="store_true",
                      help="skip MDE entirely (it only runs on rows that "
                           "don't survive BH, but the search itself is not "
                           "free)")
  parser.add_argument("--mde-seed", default=0)
  parser.add_argument("--csv", default=None,
                      help="prefix to write <prefix>.cells.csv, "
                           "<prefix>.tests.csv and <prefix>.trend.csv -- the "
                           "same numbers printed to stdout, for loading "
                           "elsewhere instead of re-parsing the printout")
  args = parser.parse_args()
  if args.mde and args.no_mde:
    parser.error("--mde and --no-mde are mutually exclusive")

  records = load(args.responses)
  print(f"{len(records)} rows loaded\n")
  report(summarize(records), control=args.control,
         trend_max=args.trend_max, draws=args.draws,
         mde_eligible=not args.no_mde,
         mde_settings=_MDE_FULL if args.mde else _MDE_FAST,
         mde_seed=args.mde_seed, csv_prefix=args.csv)


if __name__ == "__main__":
  main()
