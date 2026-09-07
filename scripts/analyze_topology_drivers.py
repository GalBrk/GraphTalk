"""Does the qwen3-8b/degree (GOT) primer effect concentrate on particular
structural features, and which task actually drives it at n=500 --
`edge_count` (as originally reported at n=30, +35.7pp) or `node_degree` (as
the n=500 MAE breakdown in `analysis/significance_report.count500.got.csv`
suggests: node_degree MAE p=0.0144 clears the bar, edge_count MAE p=0.0504
narrowly misses)?

Reuses the exact functions the headline result itself was computed with --
`scripts.check_significance._paired_values` for the control/treatment/
cluster_ids extraction and `graphtalk.significance
.paired_permutation_test_clustered`/`cluster_bootstrap_ci_clustered` for
every stratum's effect size -- so "driver analysis" numbers are directly
comparable, unit for unit, to `analysis/significance_report.count500.got.csv`
rather than a bespoke metric. Structural features come from
`scripts/extract_graph_topology.py`'s output, joined on the numeric suffix of
`instance_id` (every task shares the same graph at a given index -- see that
script's module docstring).

This step is inherently exploratory multi-feature dredging (one task-level
split plus several structural splits, tested against the same data that
produced the headline result), so `graphtalk.significance.benjamini_hochberg`
is applied across every stratified test run here, and any single stratum that
survives should be read as a new hypothesis for a follow-up confirmatory run
(mirroring `analysis/confirmatory_got_degree.json`'s pre-registration
discipline), not asserted as a second confirmed finding from the same data
that produced it.

    PYTHONPATH=. .venv/Scripts/python.exe scripts/analyze_topology_drivers.py \
        --frame analysis/sweep_frame.count500.got.csv \
        --features analysis/topology_features.csv \
        --model qwen3-8b --condition degree
"""

import argparse
import re

import pandas as pd

from graphtalk import significance
from scripts import check_significance as cs

_INSTANCE_INDEX_RE = re.compile(r"/(\d+)$")


def _instance_index(instance_id: str) -> int:
  """Matches `build_prompts.py`'s `f"{task}/{index}"` naming, the same
  pattern `scripts/check_old_vs_new_subsample.py` and
  `scripts/diff_shared_instances.py` already use."""
  match = _INSTANCE_INDEX_RE.search(str(instance_id))
  if not match:
    raise ValueError(f"instance_id {instance_id!r} doesn't end in /<index>")
  return int(match.group(1))


def _task(instance_id: str) -> str:
  return str(instance_id).split("/", 1)[0]


def _run(label, control, treatment, cluster_ids, n_perm, n_boot, seed) -> dict | None:
  if not control:
    print(f"  {label:<26} -- no paired rows")
    return None
  perm = significance.paired_permutation_test_clustered(
      control, treatment, cluster_ids, n_perm=n_perm, seed=f"{seed}:{label}"
  )
  boot = significance.cluster_bootstrap_ci_clustered(
      control, treatment, cluster_ids, n_boot=n_boot, seed=f"{seed}:{label}"
  )
  result = {
      "label": label,
      "n_clusters": perm["n_clusters"],
      "delta": boot["point_estimate"],
      "ci_low": boot["ci_low"],
      "ci_high": boot["ci_high"],
      "p_value": perm["p_value"],
  }
  print(
      f"  {label:<26} n={result['n_clusters']:>5}  delta={result['delta']:+.4f}  "
      f"95% CI=[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]  "
      f"p={result['p_value']:.4f}"
  )
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frame", default="analysis/sweep_frame.count500.got.csv")
  parser.add_argument("--features", default="analysis/topology_features.csv")
  parser.add_argument("--model", default="qwen3-8b")
  parser.add_argument("--condition", default="degree")
  parser.add_argument("--metric", default="exact")
  parser.add_argument("--node-naming", default="got")
  parser.add_argument(
      "--task", default=None,
      help="restrict to one task before stratifying, e.g. 'edge_count' -- "
           "the pooled (default, all tasks) strata below are confounded by "
           "task composition (a structural feature correlated with which "
           "task dominates a stratum reads as a 'driver' even when the "
           "within-task pattern reverses; verified this actually happens "
           "here: --task edge_count flips the density/is_forest/"
           "is_bipartite/has_isolated_node direction found in the pooled "
           "run). Pass this to check whether a pooled 'driver' survives "
           "within the one task that actually carries the effect.")
  parser.add_argument("--n-perm", type=int, default=10_000)
  parser.add_argument("--n-boot", type=int, default=10_000)
  parser.add_argument("--seed", default="topology-drivers")
  parser.add_argument("--out", default=None)
  args = parser.parse_args()

  frame = pd.read_csv(args.frame)
  frame = frame[(frame["model"] == args.model) & (~frame["is_think"])]
  if "node_naming" in frame.columns:
    frame = frame[frame["node_naming"] == args.node_naming]
  if args.task:
    frame = frame[frame["task"] == args.task]
  if frame.empty:
    raise SystemExit(
        f"no rows for model={args.model!r}, node_naming={args.node_naming!r}"
        + (f", task={args.task!r}" if args.task else "")
    )

  control, treatment, cluster_ids = cs._paired_values(frame, args.condition, args.metric)
  if not control:
    raise SystemExit("no paired rows found")

  topo = pd.read_csv(args.features).set_index("index")
  instance_ids = [iid for _, iid in cluster_ids]
  indices = [_instance_index(iid) for iid in instance_ids]
  tasks = [_task(iid) for iid in instance_ids]

  # Tercile bins computed once on the 500-graph population (not on the
  # 3000-pair sample, which would just repeat each graph's value 6x).
  degree_std_bins = pd.qcut(topo["degree_std"], 3, labels=["low", "mid", "high"],
                            duplicates="drop")
  density_bins = pd.qcut(topo["density"], 3, labels=["low", "mid", "high"],
                         duplicates="drop")

  def stratum(label: str, mask: list) -> dict | None:
    c = [v for v, m in zip(control, mask) if m]
    t = [v for v, m in zip(treatment, mask) if m]
    ids = [v for v, m in zip(cluster_ids, mask) if m]
    return _run(label, c, t, ids, args.n_perm, args.n_boot, args.seed)

  print(f"{args.model}/{args.condition}, metric={args.metric}, "
        f"node_naming={args.node_naming}, n_pairs={len(control)}\n")

  results = []
  results.append(("ALL", stratum("ALL", [True] * len(control))))

  print("\nby task (resolves the edge_count vs node_degree attribution question):")
  for task in sorted(set(tasks)):
    mask = [t == task for t in tasks]
    results.append((f"task={task}", stratum(f"task={task}", mask)))

  print("\nby has_isolated_node:")
  for value in (False, True):
    mask = [bool(topo.loc[i, "has_isolated_node"]) == value for i in indices]
    label = f"has_isolated_node={value}"
    results.append((label, stratum(label, mask)))

  print("\nby is_forest (no cycle anywhere in the graph):")
  for value in (False, True):
    mask = [bool(topo.loc[i, "is_forest"]) == value for i in indices]
    label = f"is_forest={value}"
    results.append((label, stratum(label, mask)))

  print("\nby is_bipartite:")
  for value in (False, True):
    mask = [bool(topo.loc[i, "is_bipartite"]) == value for i in indices]
    label = f"is_bipartite={value}"
    results.append((label, stratum(label, mask)))

  print("\nby size_bucket:")
  for bucket in ("small", "medium", "large"):
    mask = [topo.loc[i, "size_bucket"] == bucket for i in indices]
    label = f"size_bucket={bucket}"
    results.append((label, stratum(label, mask)))

  print("\nby degree_std tercile:")
  for level in ("low", "mid", "high"):
    mask = [degree_std_bins.loc[i] == level for i in indices]
    label = f"degree_std={level}"
    results.append((label, stratum(label, mask)))

  print("\nby density tercile:")
  for level in ("low", "mid", "high"):
    mask = [density_bins.loc[i] == level for i in indices]
    label = f"density={level}"
    results.append((label, stratum(label, mask)))

  report = pd.DataFrame([r for _, r in results if r is not None])
  non_all = report[report["label"] != "ALL"].reset_index(drop=True)
  reject = significance.benjamini_hochberg(non_all["p_value"].tolist(), q=0.05)
  non_all["bh_significant"] = reject
  all_row = report[report["label"] == "ALL"].copy()
  all_row["bh_significant"] = None
  report = pd.concat([all_row, non_all], ignore_index=True)

  n_survivors = sum(reject)
  print(
      f"\n{n_survivors}/{len(reject)} stratified/task tests survive BH "
      "correction (q=0.05) across this exploratory sweep -- treat any "
      "survivor as a new hypothesis for a follow-up confirmatory run, not a "
      "second confirmed finding from the same data that produced the "
      "headline result."
  )
  if n_survivors:
    print("survivors:")
    for _, row in non_all[non_all["bh_significant"]].iterrows():
      print(f"  {row['label']}: delta={row['delta']:+.4f}, p={row['p_value']:.4f}")

  if args.out:
    report.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
  main()
