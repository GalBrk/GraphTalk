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
import glob
import json

from graphtalk import scoring
from graphtalk import significance

CONTROL = "none"


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


def report(summary, control=CONTROL) -> None:
  cells, paired, golds = summary["cells"], summary["paired"], summary["golds"]
  densities = sorted({d for d, _ in cells}, key=lambda d: (d is None, d))
  conditions = sorted({c for _, c in cells})
  header = "  p      " + "".join(f"{c:>15}" for c in conditions)

  print("blind bar (best constant answer, per level):")
  bars = {}
  for density in densities:
    answer, bars[density] = blind_bar(golds[density])
    print(f"  p={density!s:<6} modal gold {answer!r:>6}  bar {bars[density]:.3f}"
          f"  (n={len(golds[density])})")

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
      out.append((density, condition, len(control_hits), test, delta))
    return out

  def show(rows, reject) -> None:
    for (density, condition, n, test, delta), keep in zip(rows, reject):
      level = "POOLED" if density is None else f"p={density:g}"
      print(f"  {level:<8} {condition:>11} - {control}: n={n:>5} "
            f"win {test['c']:>4} lose {test['b']:>4} delta {delta:+.4f} "
            f"p={test['p_value']:.4f}{'  *' if keep else ''}")

  print(f"\npooled across levels, paired vs {control!r} "
        "(exact McNemar on rows sharing an instance_id):")
  pooled = run(None) if len(densities) > 1 else []
  show(pooled, significance.benjamini_hochberg([t["p_value"] for *_, t, _ in pooled]))

  print("\nper level (descriptive -- these are a family, correct before "
        "quoting any one):")
  per_level = [row for density in densities for row in run(density)]
  # The pooled tests above are deliberately *not* in this family: they reuse the
  # same rows, so folding them in would let the pooled p-value drag a per-level
  # p-value under the threshold on the strength of its own data. The pooled test
  # is the primary and stands on its own; the per-level rows are exploratory.
  show(per_level,
       significance.benjamini_hochberg([t["p_value"] for *_, t, _ in per_level]))
  if per_level:
    print("  (* = survives Benjamini-Hochberg at q=0.05 within its own family)")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--responses", nargs="+", required=True,
                      help="response jsonl paths or globs")
  parser.add_argument("--control", default=CONTROL)
  args = parser.parse_args()

  records = load(args.responses)
  print(f"{len(records)} rows loaded\n")
  report(summarize(records), control=args.control)


if __name__ == "__main__":
  main()
