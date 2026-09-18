"""Paired analysis of the rewiring experiment.

The design is within-instance: for each base graph the corpus holds a `low`,
`base` and `high` triangle-structure variant that are the *same graph* rewired,
with the same degree sequence, the same question and the same gold answer, and a
prompt of identical length. So every comparison here pairs a graph against
itself, and the graph is the clustering unit for every test.

Two questions are asked, and the order matters:

  1. **Does `none` move across rewiring levels?** This is asked FIRST and
     reported FIRST, because it decides what the second question means. If
     `none` is flat while the `clustering` effect grows, that is evidence about
     primer *content*. If `none` also moves, the finding becomes "clustering
     changes the task" -- real, but a different claim. Deciding this after
     seeing the primer result would be choosing the interpretation to fit the
     answer.
  2. **Does `clustering` beat `none`, and does that gap grow with structure?**
     Reported against `filler` as well as `none`, since `filler` is the
     length-matched placebo and a `none`-delta is content *minus* length.

Everything is stdlib: `significance.py` is scipy-free by project convention, and
the cluster env has neither scipy nor statsmodels installed.
"""

import argparse
import glob
import json
import re
from collections import defaultdict

from graphtalk import significance
from graphtalk import scoring

_TRAILING = re.compile(r"[\s.]+$")
_norm = lambda value: _TRAILING.sub("", str(value).strip())
_truthy = lambda value: str(value).lower() == "true"

LEVELS = ("low", "base", "high")


def load(patterns, task):
  """scores[(model, rung, level, condition)][base_index] = 0/1, plus cap counts."""
  scores = defaultdict(dict)
  capped = defaultdict(int)
  for pattern in patterns:
    for path in glob.glob(pattern):
      if "/archive/" in path or ".got." in path:
        continue
      with open(path, encoding="utf-8") as handle:
        for line in handle:
          try:
            row = json.loads(line)
          except json.JSONDecodeError:
            continue
          if row.get("task") != task:
            continue
          parts = str(row.get("instance_id", "")).split("/")
          if len(parts) != 4:
            continue
          _, rung, level, index = parts
          key = (row.get("model"), rung, level, row.get("condition"))
          if _truthy(row.get("hit_cap")) or _truthy(row.get("overflow")):
            capped[key] += 1
            continue
          predicted = scoring.extract_answer(row.get("response") or "", task)
          scores[key][index] = int(
              predicted is not None and _norm(predicted) == _norm(row.get("gold")))
  return scores, capped


def paired(scores, control_key, treatment_key):
  """Aligned control/treatment vectors over the base graphs both share."""
  control = scores.get(control_key, {})
  treatment = scores.get(treatment_key, {})
  shared = sorted(set(control) & set(treatment))
  return ([control[i] for i in shared],
          [treatment[i] for i in shared],
          shared)


def test(control, treatment, clusters, seed=0):
  if not control:
    return None
  result = significance.paired_permutation_test_clustered(
      control, treatment, clusters, seed=seed)
  interval = significance.cluster_bootstrap_ci_clustered(
      control, treatment, clusters, seed=seed)
  return {
      "n": len(control),
      "n_clusters": result["n_clusters"],
      "control": sum(control) / len(control),
      "treatment": sum(treatment) / len(treatment),
      "delta": result["observed_diff"],
      "p": result["p_value"],
      "ci_low": interval["ci_low"],
      "ci_high": interval["ci_high"],
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--responses", nargs="+", required=True)
  parser.add_argument("--task", default="node_degree")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  scores, capped = load(args.responses, args.task)
  if not scores:
    raise SystemExit("no matching rows found")
  models = sorted({k[0] for k in scores})
  rungs = sorted({k[1] for k in scores})

  p_values, labels = [], []

  for model in models:
    print(f"\n{'='*74}\n{model}\n{'='*74}")
    for rung in rungs:
      have = [lv for lv in LEVELS if (model, rung, lv, "none") in scores]
      if len(have) < 2:
        continue
      print(f"\n  rung {rung}")

      # --- Question 1, asked and reported first --------------------------
      print("   [1] does `none` move with rewiring? (decides what [2] means)")
      control, treatment, shared = paired(
          scores, (model, rung, "low", "none"), (model, rung, "high", "none"))
      outcome = test(control, treatment, shared, args.seed)
      if outcome:
        verdict = ("NONE MOVES -- read [2] as 'clustering changes the task'"
                   if outcome["p"] is not None and outcome["p"] < 0.05
                   else "none is flat -- [2] is about primer content")
        print(f"       none low={outcome['control']:.3f} high={outcome['treatment']:.3f} "
              f"delta={outcome['delta']:+.3f} p={outcome['p']:.4f}  -> {verdict}")
        p_values.append(outcome["p"]); labels.append(f"{model}/{rung}/none-moves")

      # --- Question 2 ----------------------------------------------------
      print("   [2] primer effect, per rewiring level")
      for level in have:
        for condition in ("clustering", "filler"):
          control, treatment, shared = paired(
              scores, (model, rung, level, "none"),
              (model, rung, level, condition))
          outcome = test(control, treatment, shared, args.seed)
          if not outcome:
            continue
          print(f"       {level:<5} {condition:<11} none={outcome['control']:.3f} "
                f"{condition[:4]}={outcome['treatment']:.3f} "
                f"delta={outcome['delta']:+.3f} "
                f"CI[{outcome['ci_low']:+.3f},{outcome['ci_high']:+.3f}] "
                f"p={outcome['p']:.4f} n={outcome['n']}")
          p_values.append(outcome["p"])
          labels.append(f"{model}/{rung}/{level}/{condition}")

      dropped = sum(v for k, v in capped.items() if k[0] == model and k[1] == rung)
      if dropped:
        print(f"       ({dropped} capped/overflow rows excluded from the above)")

  if p_values:
    flags = significance.benjamini_hochberg(p_values)
    survivors = [lab for lab, keep in zip(labels, flags) if keep]
    print(f"\n{'='*74}\nBenjamini-Hochberg over all {len(p_values)} tests, q=0.05")
    print(f"  {len(survivors)} survive:")
    for label in survivors:
      print(f"    {label}")


if __name__ == "__main__":
  main()
