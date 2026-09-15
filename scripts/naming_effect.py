"""Stage 4b: does renaming the nodes change what the models get right?

`--node-naming got` renames every node from an integer to a Game-of-Thrones
character throughout the primer, the encoding and the question. Everything else
is held fixed: the same 30 graphs per task, the same queries, the same seven
primer conditions, the same models. So the two sweeps pair row for row on
`(model, instance_id, task, condition, style)`, and the difference between them
is the naming and nothing else.

That pairing is the whole point. Fatemi et al.'s headline claim is that how a
graph is *phrased* moves accuracy by tens of points; node naming is one axis of
that phrasing, isolated here from every other axis.

Three things this does that a plain groupby would get wrong:

  * **Pairs rather than compares means.** Two independent means over the same 30
    graphs waste the pairing and inflate the variance; the paired difference is
    the estimator with the right standard error.
  * **Clusters on `instance_id`.** The same graph appears under all seven
    conditions, so rows are not independent. `graphtalk/significance.py`'s
    clustered permutation and bootstrap flip and resample whole graphs, which is
    what keeps seven correlated rows from counting as seven pieces of evidence.
  * **Refuses to compare an incomplete arm.** An arm still generating has fewer
    GoT rows than integer ones, and silently intersecting them would report a
    difference measured on whichever rows happened to finish first -- biased
    toward the early tasks in the prompt file's ordering. Incomplete arms are
    named and skipped.

Non-terminating rows are dropped before pairing, matching
`scripts/check_significance.py`: a truncated response's score reflects abandoned
working, and truncation itself responds to condition.

  PYTHONPATH=. python scripts/naming_effect.py --runs runs
"""

import argparse
import collections
import glob
import json
import os

from graphtalk import significance
from scripts import score_sweep

ARMS = ("gemma4-e4b", "gemma4-12b", "qwen3-8b", "qwen3-14b",
        "gemma4-e4b-think", "gemma4-12b-think", "qwen3-8b-think",
        "qwen3-14b-think")
_KEY = ("instance_id", "task", "condition", "style")


def _load(paths: list[str]) -> list[dict]:
  """Scored records, with GoT responses converted back to integers first."""
  rows = [json.loads(line) for p in paths for line in open(p) if line.strip()]
  if not rows:
    return []
  return score_sweep.score_records(score_sweep.desubstitute_named_responses(rows))


def arm_paths(runs: str, model: str) -> tuple[list[str], list[str]]:
  """(integer paths, got paths) for one arm, excluding archived rows."""
  every = glob.glob(os.path.join(runs, f"{model}.*jsonl")) + \
          glob.glob(os.path.join(runs, f"{model}.jsonl"))
  every = [p for p in set(every) if "archive" not in p and ".redo." not in p]
  got = sorted(p for p in every if ".got" in p)
  integer = sorted(p for p in every if ".got" not in p)
  return integer, got


def paired(integer: list[dict], got: list[dict], drop_capped: bool = True) -> tuple[list, list, list]:
  """Aligned (integer score, got score, cluster id) over shared keys."""
  # `hit_cap`, not `non_terminating`: the latter is a column
  # `graphtalk.analysis.build_frame` derives, and these are raw scored records
  # from `scripts.score_sweep`, which carry the generator's own flag instead.
  # Keying off the wrong name silently excluded nothing at all.
  def keep(r):
    return r["style"] == "zero_shot" and not (drop_capped and r.get("hit_cap"))
  gi = {tuple(r[k] for k in _KEY): r for r in integer if keep(r)}
  gg = {tuple(r[k] for k in _KEY): r for r in got if keep(r)}
  shared = sorted(gi.keys() & gg.keys())
  return ([gi[k]["score"]["primary"] for k in shared],
          [gg[k]["score"]["primary"] for k in shared],
          [k[0] for k in shared])


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--runs", default="runs")
  parser.add_argument("--n-perm", type=int, default=10_000)
  parser.add_argument("--n-boot", type=int, default=10_000)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--q", type=float, default=0.05)
  parser.add_argument("--keep-capped", action="store_true",
                      help="include responses that hit the token cap; by default "
                           "they are dropped, since a truncated response's score "
                           "reflects abandoned working and GoT names cost more "
                           "tokens, which would confound naming with truncation")
  args = parser.parse_args()

  results, skipped = [], []
  for model in ARMS:
    ipaths, gpaths = arm_paths(args.runs, model)
    if not gpaths:
      skipped.append((model, "no GoT rows"))
      continue
    integer, got = _load(ipaths), _load(gpaths)
    n_got = sum(1 for r in got if r["style"] == "zero_shot")
    n_int = sum(1 for r in integer if r["style"] == "zero_shot")
    if n_got != n_int:
      skipped.append((model, f"incomplete: {n_got} GoT vs {n_int} integer rows"))
      continue
    ctrl, treat, clusters = paired(integer, got, drop_capped=not args.keep_capped)
    if not ctrl:
      skipped.append((model, "no shared pairs"))
      continue
    perm = significance.paired_permutation_test_clustered(
        ctrl, treat, clusters, n_perm=args.n_perm, seed=args.seed)
    boot = significance.cluster_bootstrap_ci_clustered(
        ctrl, treat, clusters, n_boot=args.n_boot, seed=args.seed)
    results.append((model, perm["n_pairs"], perm["n_clusters"],
                    sum(ctrl) / len(ctrl), sum(treat) / len(treat),
                    perm["observed_diff"], boot["ci_low"], boot["ci_high"],
                    perm["p_value"]))

  print(f"{'arm':<20}{'pairs':>7}{'graphs':>7}{'integer':>9}{'got':>8}"
        f"{'delta':>8}{'95% CI':>20}{'p':>8}{'':>4}")
  qs = significance.benjamini_hochberg([r[8] for r in results], q=args.q) \
      if results else []
  for (model, n, m, ci, cg, d, lo, hi, p), sig in zip(results, qs):
    # `lo`/`hi` are `None` where too few pairs disagreed for a percentile
    # bootstrap to mean anything -- say so rather than printing a
    # confident-looking interval; see cluster_bootstrap_ci_clustered.
    interval = "n/a" if lo is None else f"[{lo:+.1%}, {hi:+.1%}]"
    print(f"{model:<20}{n:>7}{m:>7}{ci:>9.1%}{cg:>8.1%}{d:>+8.1%}"
          f"{interval:>20}{p:>8.4f}{'  *' if sig else '':>4}")
  if skipped:
    print("\nskipped:")
    for model, why in skipped:
      print(f"  {model:<20}{why}")
  print("\n* = significant after Benjamini-Hochberg across the arms tested.")


if __name__ == "__main__":
  main()
