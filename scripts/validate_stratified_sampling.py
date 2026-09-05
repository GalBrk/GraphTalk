"""Track 2.2's required dry run: before spending any GPU time on
`build_prompts.py --graph-source stratified` (which oversamples the
largest graphs per task, on the theory that bigger graphs are harder and
so yield more discordant pairs per graph collected -- see that function's
docstring in `scripts/build_prompts.py`), check that theory against
*already-collected* responses instead of assuming it.

**Method.** Joins `analysis/sweep_frame.csv` (which has `exact` per
(model, instance_id, condition) but no graph-size columns) against
`prompts.jsonl` (which has `nodes`/`edges` per `instance_id` but no
response columns) to attach node/edge counts to every response row. For
each near-ceiling model (control-condition accuracy above
`--near-ceiling-threshold`, the same population Track 2.2 targets --
low-ceiling models already have plenty of headroom and don't need
stratified sampling), and each non-control condition, splits that cell's
paired (control, treatment) instances into two node-count strata (small
vs large, split at the per-task median so the comparison isn't confounded
by task difficulty varying with typical graph size) and compares:

  - discordant-pair rate: fraction of paired instances where the model's
    `exact` correctness differs between the control and treatment
    condition -- the quantity that actually carries information for a
    paired test (a concordant pair, correct-correct or wrong-wrong,
    contributes nothing to detecting a treatment effect).
  - the point-estimate delta and permutation p-value in each stratum
    separately, via the same `paired_permutation_test_clustered` the main
    pipeline uses.

If large graphs really do yield more discordant pairs than small graphs
for near-ceiling models, this shows up directly as a higher discordant
rate in the "large" stratum -- the evidence Track 2.2's strategy needs
before any new data is collected. If the two strata don't differ (or
large graphs are *less* discordant, e.g. because a near-ceiling model
also degrades on control for large graphs, leaving no headroom to
improve), that's equally worth knowing now rather than after paying for
it.

  PYTHONPATH=. .venv/bin/python scripts/validate_stratified_sampling.py
"""

import argparse
import json

import pandas as pd

from graphtalk import significance
from scripts import check_significance as cs

CONTROL = cs.CONTROL


def _load_node_counts(prompts_path: str) -> pd.DataFrame:
  """One row per `instance_id` with its `nodes`/`edges` -- both are
  properties of the underlying graph, identical across every
  condition/style row `build_prompts.py` emits for that instance_id, so
  the first occurrence is as good as any other."""
  seen: dict = {}
  with open(prompts_path, encoding="utf-8") as handle:
    for line in handle:
      row = json.loads(line)
      instance_id = row["instance_id"]
      if instance_id not in seen:
        seen[instance_id] = {"instance_id": instance_id, "nodes": row["nodes"],
                              "edges": row["edges"]}
  return pd.DataFrame(seen.values())


def _main_sweep_rows(frame: pd.DataFrame) -> pd.DataFrame:
  """Same scope as `check_significance.py`'s main-sweep report, taken from
  that script rather than restated.

  It used to restate it as "non-`-think` rows, non-terminating failures
  dropped", which stopped being true when Phase 2 started forcing
  non-terminating rows to score as wrong and keeping them. At the correct
  scope this script's own headline changes materially -- 7 of 12 cells
  becomes 11 of 12, and the mean discordant-rate gap widens from
  0.0185/0.0217 to 0.0226/0.0516 -- so the published version was
  understating the effect it was built to detect."""
  return cs.main_sweep_scope(frame)


def _near_ceiling_models(frame: pd.DataFrame, threshold: float) -> list:
  """Models whose control-condition accuracy is above `threshold` --
  Track 2.2's actual target population, mirroring
  `check_significance.py`'s own near-ceiling test (`control_mean >
  args.near_ceiling_threshold`, high side only: this dry run is about
  *improving* a model that already looks near-perfect, not a
  near-floor one)."""
  control = frame[frame["condition"] == CONTROL]
  means = control.groupby("model_family")["exact"].mean()
  return sorted(means[means > threshold].index.tolist())


def _discordant_rate(control: list, treatment: list) -> float:
  n = len(control)
  if n == 0:
    return float("nan")
  return sum(1 for c, t in zip(control, treatment) if c != t) / n


def compare_strata(frame: pd.DataFrame, model: str, condition: str,
                    n_perm: int = 500, seed: int = 0) -> dict | None:
  """For one (model, condition) cell, splits paired instances into
  small/large node-count strata at the per-instance median and returns
  discordant-pair rate + permutation-test results for each stratum.
  Returns `None` if either stratum ends up with fewer than 2 clusters
  (too small to run a meaningful test on)."""
  cell = frame[frame["model_family"] == model]
  control, treatment, cluster_ids = cs._paired_values(cell, condition, "exact")
  if not control:
    return None
  # Keyed by graph index, matching `cs._paired_values`' cluster id -- which
  # is `(model, graph_index)`, not `(model, instance_id)`. `nodes` is a
  # property of the graph, and the six tasks sharing an index share that
  # graph exactly, so collapsing the task prefix here is a re-keying, not an
  # approximation. Keying by `instance_id` would now raise `KeyError` on
  # every lookup.
  cell_graphs = cell.assign(_graph=cell["instance_id"].map(cs._graph_index))
  nodes_by_graph = cell_graphs.drop_duplicates("_graph").set_index("_graph")["nodes"]
  node_counts = [nodes_by_graph[graph_index] for _model, graph_index in cluster_ids]
  median = pd.Series(node_counts).median()

  strata = {"small": {"control": [], "treatment": [], "cluster_ids": []},
            "large": {"control": [], "treatment": [], "cluster_ids": []}}
  for c, t, cid, n_nodes in zip(control, treatment, cluster_ids, node_counts):
    bucket = "large" if n_nodes > median else "small"
    strata[bucket]["control"].append(c)
    strata[bucket]["treatment"].append(t)
    strata[bucket]["cluster_ids"].append(cid)

  if len(strata["small"]["control"]) < 2 or len(strata["large"]["control"]) < 2:
    return None

  result = {"model": model, "condition": condition, "median_nodes": median,
            "n_small": len(strata["small"]["control"]),
            "n_large": len(strata["large"]["control"])}
  for bucket, data in strata.items():
    result[f"{bucket}_discordant_rate"] = _discordant_rate(data["control"], data["treatment"])
    result[f"{bucket}_delta"] = (
        sum(data["treatment"]) / len(data["treatment"])
        - sum(data["control"]) / len(data["control"])
    )
    test = significance.paired_permutation_test_clustered(
        data["control"], data["treatment"], data["cluster_ids"],
        n_perm=n_perm, seed=seed,
    )
    result[f"{bucket}_p_value"] = test["p_value"]
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--frame", default="analysis/sweep_frame.csv")
  parser.add_argument("--prompts", default="prompts.jsonl")
  parser.add_argument("--near-ceiling-threshold", type=float, default=0.95)
  parser.add_argument("--n-perm", type=int, default=500)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out", default=None)
  args = parser.parse_args()

  frame = pd.read_csv(args.frame)
  node_counts = _load_node_counts(args.prompts)
  frame = frame.merge(node_counts, on="instance_id", how="inner")
  frame = _main_sweep_rows(frame)

  models = _near_ceiling_models(frame, args.near_ceiling_threshold)
  if not models:
    print(f"No models exceed the near-ceiling threshold ({args.near_ceiling_threshold}) "
          f"on the control condition in {args.frame} -- nothing for stratified "
          f"sampling to target; skipping.")
    return

  conditions = sorted(c for c in frame["condition"].unique() if c != CONTROL)
  print(f"Near-ceiling models ({args.near_ceiling_threshold} threshold): {models}\n")
  print(f"{'model':<14}{'condition':<12}{'median_n':>9}{'n_small':>8}{'n_large':>8}  "
        f"{'small_discord':>14}{'large_discord':>14}  {'small_p':>9}{'large_p':>9}")

  rows = []
  for model in models:
    for condition in conditions:
      result = compare_strata(frame, model, condition, n_perm=args.n_perm, seed=args.seed)
      if result is None:
        continue
      rows.append(result)
      print(f"{result['model']:<14}{result['condition']:<12}{result['median_nodes']:>9.1f}"
            f"{result['n_small']:>8}{result['n_large']:>8}  "
            f"{result['small_discordant_rate']:>14.4f}{result['large_discordant_rate']:>14.4f}  "
            f"{result['small_p_value']:>9.4f}{result['large_p_value']:>9.4f}")

  if not rows:
    print("\nNo cell had enough paired instances in both strata to compare.")
    return

  table = pd.DataFrame(rows)
  more_discordant_large = int((table["large_discordant_rate"] > table["small_discordant_rate"]).sum())
  print(f"\n{more_discordant_large}/{len(table)} cells show a higher discordant-pair rate "
        f"in the large-graph stratum than the small-graph stratum.")
  mean_small = table["small_discordant_rate"].mean()
  mean_large = table["large_discordant_rate"].mean()
  print(f"Mean discordant rate: small={mean_small:.4f}  large={mean_large:.4f}")

  if args.out:
    table.to_csv(args.out, index=False)
    print(f"\nwrote {len(table)} rows to {args.out}")


if __name__ == "__main__":
  main()
