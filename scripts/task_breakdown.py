"""Per-task breakdown for qwen3-8b/degree (GOT): is the --count 500 effect
still concentrated on edge_count alone, the way it was at --count 30, or
has it spread to other tasks -- most notably node_degree, which the
`--metric mae` cross-check already hinted at (analysis/significance_report
.count500.got.csv's MAE rows: node_degree cleared its own bar, edge_count
only borderline)?

`check_significance.py --metric exact` pools across all 6 tasks by design
(see its own module docstring on why: that's what gives the McNemar-
underpowered per-cell tests any power at all). This script is the per-task
view that pooling deliberately discards -- now affordable, since --count
500 gives each task ~500 pairs instead of 30 -- using the exact same
clustered-permutation/bootstrap primitives check_significance.py itself
uses, so the numbers here are directly comparable to (not a different
methodology from) the pooled report.

`--metric mae` reuses check_significance.py's own imputation machinery
(`_mae_imputation_table`/`_mae_eligible_frame`) for the 3 integer tasks it
is defined on (node_count, edge_count, node_degree), computed from the
*whole* frame (every model), matching that module's own scoping -- not
recomputed per model, which `_mae_imputation_table`'s docstring explicitly
warns against (too few wrong rows per model/condition for a stable
median).

Pass --old-frame to print the tracked --count 30 report's own frame's
breakdown alongside the new one, so the two sample sizes' per-task
patterns sit side by side rather than requiring a separate run to compare
by eye.

Descriptive only, like check_old_vs_new_subsample.py -- no BH correction
is applied across these per-task cells; this is meant to locate where an
already-established pooled effect lives, not to run a new independent
hypothesis test at a finer grain.

    PYTHONPATH=. .venv/bin/python scripts/task_breakdown.py \
        --frame analysis/sweep_frame.count500.got.csv \
        --old-frame analysis/sweep_frame.got.csv \
        --model qwen3-8b --condition degree
"""

import argparse

import pandas as pd

from graphtalk import significance
from scripts import check_significance as cs

_MAE_TASKS = ("node_count", "edge_count", "node_degree")


def _exact_breakdown(model_scope: pd.DataFrame, condition: str, n_perm: int,
                      n_boot: int, seed: str) -> list[dict]:
  rows = []
  for task in sorted(model_scope["task"].unique()):
    task_frame = model_scope[model_scope["task"] == task]
    control, treatment, cluster_ids = cs._paired_values(task_frame, condition, "exact")
    if not control:
      continue
    perm = significance.paired_permutation_test_clustered(
        control, treatment, cluster_ids, n_perm=n_perm, seed=f"{seed}:exact:{task}"
    )
    boot = significance.cluster_bootstrap_ci_clustered(
        control, treatment, cluster_ids, n_boot=n_boot, seed=f"{seed}:exact:{task}"
    )
    rows.append({
        "task": task, "metric": "exact",
        "n_clusters": perm["n_clusters"],
        "delta": boot["point_estimate"],
        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        "p_value": perm["p_value"],
    })
  return rows


def _mae_breakdown(full_scope: pd.DataFrame, model: str, condition: str,
                    n_perm: int, n_boot: int, seed: str) -> list[dict]:
  # Imputation table from the WHOLE main-sweep frame (every model), matching
  # _mae_imputation_table's own docstring -- a per-model slice is usually
  # too few wrong rows for a stable median.
  imputation_table = cs._mae_imputation_table(full_scope)
  eligible = cs._mae_eligible_frame(full_scope, imputation_table)
  eligible = eligible[eligible["model"] == model]
  rows = []
  for task in _MAE_TASKS:
    task_frame = eligible[eligible["task"] == task]
    control, treatment, cluster_ids = cs._paired_values(
        task_frame, condition, "absolute_error"
    )
    if not control:
      continue
    perm = significance.paired_permutation_test_clustered(
        control, treatment, cluster_ids, n_perm=n_perm, seed=f"{seed}:mae:{task}"
    )
    boot = significance.cluster_bootstrap_ci_clustered(
        control, treatment, cluster_ids, n_boot=n_boot, seed=f"{seed}:mae:{task}"
    )
    # mae_delta = control - treatment (lower error is better); flip the
    # underlying treatment-control sign so positive means "helped" here
    # too, matching _report_mae's own convention.
    rows.append({
        "task": task, "metric": "mae",
        "n_clusters": perm["n_clusters"],
        "delta": -boot["point_estimate"],
        "ci_low": -boot["ci_high"], "ci_high": -boot["ci_low"],
        "p_value": perm["p_value"],
    })
  return rows


def _print_table(label: str, rows: list[dict]) -> None:
  print(f"\n  {label}")
  if not rows:
    print("    (no eligible rows)")
    return
  print(f"    {'task':<16}{'metric':<8}{'n':>6}{'delta':>10}{'95% CI':>24}{'p':>9}")
  for r in rows:
    flag = "  *" if r["p_value"] < 0.05 else ""
    print(f"    {r['task']:<16}{r['metric']:<8}{r['n_clusters']:>6}"
          f"{r['delta']:>+10.4f}   [{r['ci_low']:>+.4f}, {r['ci_high']:>+.4f}]"
          f"{r['p_value']:>9.4f}{flag}")


def _run(path: str, label: str, model: str, condition: str, node_naming: str,
          n_perm: int, n_boot: int, seed: str) -> None:
  raw = pd.read_csv(path)
  full_scope = raw[~raw["is_think"]]
  if "node_naming" in full_scope.columns:
    full_scope = full_scope[full_scope["node_naming"] == node_naming]
  model_scope = full_scope[full_scope["model"] == model]
  if model_scope.empty:
    print(f"\n  {label}: no rows for model={model!r} in {path}")
    return
  exact_rows = _exact_breakdown(model_scope, condition, n_perm, n_boot, seed)
  mae_rows = _mae_breakdown(full_scope, model, condition, n_perm, n_boot, seed)
  _print_table(f"{label} -- exact (* = p<0.05, uncorrected, descriptive)", exact_rows)
  _print_table(f"{label} -- mae", mae_rows)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frame", default="analysis/sweep_frame.count500.got.csv")
  parser.add_argument("--old-frame", default=None,
                      help="optional: the tracked --count 30 report's frame, "
                           "for a side-by-side comparison of the per-task "
                           "pattern at the two sample sizes")
  parser.add_argument("--model", default="qwen3-8b")
  parser.add_argument("--condition", default="degree")
  parser.add_argument("--node-naming", default="got")
  parser.add_argument("--n-perm", type=int, default=10_000)
  parser.add_argument("--n-boot", type=int, default=10_000)
  parser.add_argument("--seed", default="task-breakdown")
  args = parser.parse_args()

  _run(args.frame, f"{args.model}/{args.condition} ({args.frame})",
       args.model, args.condition, args.node_naming,
       args.n_perm, args.n_boot, args.seed)
  if args.old_frame:
    _run(args.old_frame, f"{args.model}/{args.condition} ({args.old_frame})",
         args.model, args.condition, args.node_naming,
         args.n_perm, args.n_boot, f"{args.seed}:old")


if __name__ == "__main__":
  main()
