"""Diagnostic: is the qwen3-8b/degree (GOT) global-significance flip at
--count 500 just more power on a stable effect, or did the *added* 470
instances per task behave differently from the original 30?

`build_prompts.py::load_rows` takes the first `count` rows of the published
split, and the published split is already a shuffle of the generator's
emission order -- so instances 0-29 in the --count 500 frame are byte-
identical (same instance_ids, same prompts) to the tracked --count 30 GOT
sweep, and instances 30-499 are the newly-generated ones. This script
re-derives that split from the frame itself (by instance_id's numeric
suffix, matching build_prompts.py's `f"{task}/{index}"` naming) rather than
assuming a separate file for each half, and runs the exact same clustered
permutation test and cluster-bootstrap CI check_significance.py itself uses
-- via check_significance.py's own `_paired_values`, the same reuse pattern
scripts/benchmark_mde.py already uses -- on the "original" slice, the "new"
slice, and the full 500 for reference.

Read the three lines side by side:
  - If "original" and "new" have overlapping CIs and similar deltas, the
    effect looks stable across the whole sample -- the global-significance
    flip is most likely explained by `MDE ~ 1/sqrt(N)` finally catching up
    to a real, constant effect (pure power), not by a change in what's
    being measured.
  - If "new" diverges from what "original" alone would have predicted,
    something about the added instances (graph composition, task mix, or a
    generation-environment difference between the two runs) is worth
    chasing before trusting the pooled number at face value.

This is a diagnostic split, not a new formal hypothesis test: no BH
correction is applied here, and the result is not meant to replace
`bh_significant_global` in analysis/significance_report.count500.got.csv --
only to explain it.

    PYTHONPATH=. .venv/bin/python scripts/check_old_vs_new_subsample.py \
        --frame analysis/sweep_frame.count500.got.csv \
        --model qwen3-8b --condition degree
"""

import argparse
import re

import pandas as pd

from graphtalk import significance
from scripts import check_significance as cs

_INSTANCE_INDEX_RE = re.compile(r"/(\d+)$")


def _instance_index(instance_id: str) -> int:
  """The numeric suffix of an instance_id (e.g. "edge_count/27" -> 27).

  Matches `build_prompts.py`'s `f"{task}/{index}"` naming exactly -- this
  is asserting a format check_significance.py and build_prompts.py already
  assume everywhere else, not adding a new one.
  """
  match = _INSTANCE_INDEX_RE.search(str(instance_id))
  if not match:
    raise ValueError(f"instance_id {instance_id!r} doesn't end in /<index>")
  return int(match.group(1))


def _run(label: str, frame: pd.DataFrame, condition: str, metric: str,
         seed: str, n_perm: int, n_boot: int) -> dict:
  control, treatment, cluster_ids = cs._paired_values(frame, condition, metric)
  if not control:
    print(f"  {label:<22} -- no paired rows found")
    return {}
  perm = significance.paired_permutation_test_clustered(
      control, treatment, cluster_ids, n_perm=n_perm, seed=f"{seed}:{label}"
  )
  boot = significance.cluster_bootstrap_ci_clustered(
      control, treatment, cluster_ids, n_boot=n_boot, seed=f"{seed}:{label}"
  )
  task_lo, task_hi = cs._task_delta_range(control, treatment, cluster_ids)
  result = {
      "label": label,
      "n_clusters": perm["n_clusters"],
      "delta": boot["point_estimate"],
      "ci_low": boot["ci_low"],
      "ci_high": boot["ci_high"],
      "p_value": perm["p_value"],
      "task_delta_min": task_lo,
      "task_delta_max": task_hi,
  }
  print(
      f"  {label:<22} n_clusters={result['n_clusters']:>5}  "
      f"delta={result['delta']:+.4f}  "
      f"95% CI=[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]  "
      f"p={result['p_value']:.4f}  "
      f"per-task range=[{task_lo:+.4f}, {task_hi:+.4f}]"
  )
  return result


def _ci_overlap(a: dict, b: dict) -> bool:
  return a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frame", default="analysis/sweep_frame.count500.got.csv")
  parser.add_argument("--model", default="qwen3-8b")
  parser.add_argument("--condition", default="degree")
  parser.add_argument("--metric", default="exact")
  parser.add_argument("--node-naming", default="got")
  parser.add_argument(
      "--split-at", type=int, default=30,
      help="instances 0..split-at-1 per task are 'original' (identical to "
           "the tracked --count 30 sweep); instances >= split-at are 'new'",
  )
  parser.add_argument("--n-perm", type=int, default=10_000)
  parser.add_argument("--n-boot", type=int, default=10_000)
  parser.add_argument("--seed", default="old-vs-new")
  args = parser.parse_args()

  frame = pd.read_csv(args.frame)
  frame = frame[(frame["model"] == args.model) & (~frame["is_think"])]
  if "node_naming" in frame.columns:
    frame = frame[frame["node_naming"] == args.node_naming]
  if frame.empty:
    raise ValueError(
        f"no rows for model={args.model!r}, node_naming={args.node_naming!r} "
        f"in {args.frame}"
    )

  frame = frame.copy()
  frame["_index"] = frame["instance_id"].map(_instance_index)

  original = frame[frame["_index"] < args.split_at]
  new = frame[frame["_index"] >= args.split_at]

  print(
      f"{args.model}/{args.condition}, metric={args.metric}, "
      f"node_naming={args.node_naming}, split at index {args.split_at}\n"
  )
  full_result = _run(
      f"full (0-{frame['_index'].max()})", frame, args.condition,
      args.metric, args.seed, args.n_perm, args.n_boot,
  )
  print()
  original_result = _run(
      f"original (0-{args.split_at - 1})", original, args.condition,
      args.metric, args.seed, args.n_perm, args.n_boot,
  )
  new_result = _run(
      f"new ({args.split_at}-{frame['_index'].max()})", new, args.condition,
      args.metric, args.seed, args.n_perm, args.n_boot,
  )

  if original_result and new_result:
    print("\n  interpretation:")
    overlap = _ci_overlap(original_result, new_result)
    print(f"    original vs new 95% CIs overlap: {overlap}")
    if overlap:
      print(
          "    -> consistent with one stable effect across the full sample;"
          "\n       the global-significance flip looks like pure added"
          "\n       power (MDE shrinking as n_clusters grows), not a change"
          "\n       in what's being measured."
      )
    else:
      print(
          "    -> the 'new' slice's estimate is not what the 'original'"
          "\n       30 alone would predict -- worth chasing structurally:"
          "\n       compare graph size/density between the two slices"
          "\n       (prompts_got.count500.jsonl's nodes/edges fields), check"
          "\n       whether 'new' rows were generated in the same run/"
          "\n       environment as 'original', and check the per-task"
          "\n       breakdown (per-task range above, or a full per-task"
          "\n       rerun) for which task(s) the divergence lives in."
      )
    # A quick, non-rigorous consistency check: does the "original" slice
    # here reproduce the number the standalone --count 30 sweep already
    # reported? If not, the two runs' first 30 instances aren't actually
    # measuring the same thing (e.g. regenerated responses differ from the
    # original run's), independent of anything about the new 470.
    print(
        f"\n  sanity check: compare 'original' above ({original_result['delta']:+.4f}, "
        f"n={original_result['n_clusters']}) against the tracked --count 30 "
        f"GOT report's own delta for {args.model}/{args.condition} "
        f"(analysis/significance_report.got.csv). These should match closely "
        f"if the shared first-{args.split_at} instances' responses weren't "
        f"regenerated differently between the two runs."
    )


if __name__ == "__main__":
  main()
