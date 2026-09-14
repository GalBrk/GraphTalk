"""One-off diagnostics for three open questions `scripts/check_significance.py`
itself can't answer -- read the printed output and fold whatever's worth
keeping into `docs/sweep-findings.md` as prose. These are investigations, not
a permanent pipeline stage: nothing here writes a new column into
`analysis/significance_report.csv`.

Three sections:

1. **Which graph instances go missing.** `n_instances_missing` in
   `analysis/significance_report.csv` counts, per (model, condition), how
   many `(model, instance_id)` clusters contributed zero surviving pairs to
   the `excluded` bound (every style's row for that instance was
   `non_terminating` on the control or treatment side). This re-derives
   *which* instances those are, via the same `check_significance._paired_values`
   join, and joins their graph size (`nodes`) from `prompts*.jsonl` to see
   whether missingness skews toward larger graphs or particular tasks.
2. **Why `gemma4-e4b` truncates so much more than the other three models.**
   A quantitative pass first (`response_len_chars`, `task`, `style`,
   `condition`, `length_outlier` -- already in the canonical frame, no join
   needed), then a small stratified qualitative sample of raw `gemma4-e4b`
   non-terminating responses, re-joined from `runs/*.jsonl` the same way
   `scripts/sample_failures.py` does, for a manual read of what the
   abandoned generation is actually doing.
3. **Whether the `filler` measurement-instrument confound is still present**
   in the current, regenerated `sweep_frame.csv` -- `non_terminating_source`
   should now read `recorded` (the `hit_cap`-backed instrument) uniformly
   across every condition, `filler` included, rather than a mix with the
   older `ground_truth_file` fallback.

  PYTHONPATH=. .venv/bin/python scripts/characterize_non_termination.py \
      --frame analysis/sweep_frame.csv \
      --significance analysis/significance_report.csv \
      --responses runs/*.jsonl \
      --prompts prompts.jsonl prompts_zero_shot.jsonl \
      --out analysis/non_termination_sample.csv
"""

import argparse
import glob
import json

import pandas as pd

from graphtalk import analysis
from scripts import check_significance as cs
from scripts import score_sweep

_PREVIEW_CHARS = 400


def _load_paths(patterns: list[str]) -> list[str]:
  paths = sorted(p for pattern in patterns for p in glob.glob(pattern))
  return [p for p in paths if not analysis.is_excluded(p)]


def _load_prompt_sizes(paths: list[str]) -> dict[tuple, int]:
  """`{(instance_id, condition, style, node_naming): nodes}` -- same join
  key `scripts/sample_failures.py` uses, but only `nodes` is needed here."""
  sizes = {}
  for path in paths:
    with open(path) as handle:
      for line in handle:
        if not line.strip():
          continue
        row = json.loads(line)
        key = (row["instance_id"], row["condition"], row["style"],
               row.get("node_naming", "integer"))
        sizes[key] = row["nodes"]
  return sizes


def _missing_instances(frame: pd.DataFrame, sig: pd.DataFrame,
                        node_sizes: dict) -> pd.DataFrame:
  """One row per `(model, condition, instance_id)` that dropped out of the
  `excluded` bound's pairing entirely (see module docstring, section 1).

  Re-derives membership via `check_significance._paired_values` -- the same
  join `_report` uses -- rather than re-deriving the join logic separately,
  so this can never disagree with what `n_instances_missing` actually
  counted.
  """
  main_sweep_raw = cs.main_sweep_scope(frame)
  # Same rows: the pipeline no longer drops non-terminating ones (Phase 2
  # forces them to score as wrong and keeps them), so there is no second,
  # narrower scope left for this to compare against. Kept as two names
  # because the callers below read differently, not because they differ.
  main_sweep = main_sweep_raw
  targets = sig[(sig["bound"] == "excluded") & (sig["n_instances_missing"] > 0)
                & (sig["group"] != "pooled across all models")]
  rows = []
  for _, sig_row in targets.iterrows():
    model, condition = sig_row["group"], sig_row["condition"]
    group = main_sweep[main_sweep["model"] == model]
    raw_group = main_sweep_raw[main_sweep_raw["model"] == model]
    _, _, present = cs._paired_values(group, condition, "exact")
    # `_paired_values`' cluster id is `(model, graph_index)`, so membership
    # has to be compared at the graph level -- the unit `n_instances_missing`
    # itself now counts. Comparing these graph indices against full
    # `"<task>/<index>"` ids would make the set difference report *every*
    # instance as missing, silently, with no error to notice.
    present_graphs = {graph_index for _model, graph_index in present}
    all_graphs = set(raw_group["instance_id"].map(cs._graph_index).unique())
    missing_graphs = all_graphs - present_graphs
    missing_ids = sorted(
        instance_id for instance_id in raw_group["instance_id"].unique()
        if cs._graph_index(instance_id) in missing_graphs
    )
    for instance_id in missing_ids:
      task = instance_id.partition("/")[0]
      style_rows = raw_group[raw_group["instance_id"] == instance_id]
      nodes_seen = {
          node_sizes.get((instance_id, cond, style, "integer"))
          for cond, style in zip(style_rows["condition"], style_rows["style"])
      }
      nodes_seen.discard(None)
      rows.append({
          "model": model, "condition": condition, "instance_id": instance_id,
          "task": task, "nodes": next(iter(nodes_seen), None),
      })
  return pd.DataFrame(rows)


def _non_termination_quant_breakdown(frame: pd.DataFrame) -> None:
  non_term = frame[frame["failure_type"] == "non_terminating"]
  if non_term.empty:
    print("no non_terminating rows in this frame")
    return
  print("\n-- non-terminating rate and mean response_len_chars, by model --")
  by_model = frame.groupby("model").agg(
      rows=("failure_type", "size"),
      non_terminating=("failure_type", lambda s: (s == "non_terminating").sum()),
  )
  by_model["rate"] = by_model["non_terminating"] / by_model["rows"]
  by_model["mean_len_non_terminating"] = non_term.groupby("model")[
      "response_len_chars"
  ].mean()
  print(by_model.to_string())

  print("\n-- gemma4-e4b non-terminating rows, by task --")
  e4b = non_term[non_term["model"] == "gemma4-e4b"]
  if not e4b.empty:
    print(e4b["task"].value_counts().to_string())
    print("\n-- gemma4-e4b non-terminating rows, by condition --")
    print(e4b["condition"].value_counts().to_string())
    print("\n-- gemma4-e4b non-terminating rows, by style --")
    print(e4b["style"].value_counts().to_string())
  else:
    print("gemma4-e4b has no non_terminating rows in this frame")


def _qualitative_sample(frame: pd.DataFrame, model: str, n: int,
                         seed: int, responses: dict) -> pd.DataFrame:
  """`n` raw `model` non-terminating responses, stratified by task, for a
  manual read of what the abandoned generation is doing -- same
  full/preview/tail shape as `scripts/sample_failures.py`'s CSV."""
  pool = frame[(frame["model"] == model)
               & (frame["failure_type"] == "non_terminating")]
  if pool.empty:
    return pool
  sampled = [
      group.sample(min(len(group), n), random_state=seed)
      for _, group in pool.groupby("task", sort=False)
  ]
  sample = pd.concat(sampled, ignore_index=True)
  full, previews, tails = [], [], []
  for _, row in sample.iterrows():
    naming = row["node_naming"] if "node_naming" in row else "integer"
    text = responses.get(
        (row["model"], row["instance_id"], row["condition"], row["style"], naming),
        "",
    )
    full.append(text)
    previews.append(text[:_PREVIEW_CHARS])
    tails.append(text[-_PREVIEW_CHARS:])
  return sample.assign(
      response_full=full, response_preview=previews, response_tail_preview=tails,
  )


def _instrument_check(frame: pd.DataFrame) -> None:
  print("\n-- non_terminating_source by condition (filler instrument check) --")
  print(pd.crosstab(frame["condition"], frame["non_terminating_source"]).to_string())


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frame", required=True)
  parser.add_argument("--significance", required=True,
                       help="analysis/significance_report.csv, for section 1")
  parser.add_argument("--responses", nargs="+", required=True)
  parser.add_argument("--prompts", nargs="+", default=[])
  parser.add_argument("--qual-model", default="gemma4-e4b",
                       help="model to pull the qualitative sample from")
  parser.add_argument("--qual-n-per-task", type=int, default=2)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--out", default=None,
                       help="optional path to write the qualitative sample "
                            "(section 2) as CSV")
  args = parser.parse_args()

  frame = pd.read_csv(args.frame)
  scheme = analysis.frame_node_naming(frame)
  sig = pd.read_csv(args.significance)

  print("=" * 78)
  print("Section 1: which graph instances go missing")
  node_sizes = _load_prompt_sizes(args.prompts) if args.prompts else {}
  missing = _missing_instances(frame, sig, node_sizes)
  if missing.empty:
    print("no rows with n_instances_missing > 0 in the excluded bound")
  else:
    print(f"{len(missing)} missing (model, condition, instance_id) rows")
    print("\n-- by task --")
    print(missing["task"].value_counts().to_string())
    if missing["nodes"].notna().any():
      print("\n-- graph size (nodes) of missing vs. all instances --")
      print(f"missing: mean={missing['nodes'].mean():.1f}, "
            f"n={missing['nodes'].notna().sum()}")
      all_sizes = [
          node_sizes[k] for k in node_sizes
          if k[3] == "integer"
      ]
      if all_sizes:
        print(f"all (any condition/style): mean="
              f"{sum(all_sizes) / len(all_sizes):.1f}, n={len(all_sizes)}")
    else:
      print("(no --prompts given, or no sizes matched -- graph size unknown)")

  print(f"\n{'=' * 78}")
  print("Section 2: why gemma4-e4b truncates more")
  _non_termination_quant_breakdown(frame)

  responses = {}
  qual_sample = pd.DataFrame()
  if args.responses:
    records = score_sweep.load(_load_paths(args.responses))
    responses = {
        (r["model"], r["instance_id"], r["condition"], r["style"],
         r.get("node_naming", "integer")): r["response"]
        for r in records
    }
    qual_sample = _qualitative_sample(
        frame, args.qual_model, args.qual_n_per_task, args.seed, responses
    )
    print(f"\n-- {len(qual_sample)}-row qualitative sample of "
          f"{args.qual_model} non-terminating responses (read the tail, "
          "not just the head -- the diagnostic content is usually there) --")
    for _, row in qual_sample.iterrows():
      print(f"\n  [{row['task']}/{row['condition']}/{row['style']}] "
            f"{row['instance_id']}")
      print(f"  ...{row['response_tail_preview']!r}")

  print(f"\n{'=' * 78}")
  print("Section 3: filler instrument check")
  _instrument_check(frame)

  if args.out and not qual_sample.empty:
    out = analysis.tagged_path(args.out, scheme)
    qual_sample.to_csv(out, index=False)
    print(f"\nwrote {len(qual_sample)} qualitative-sample rows to {out}")


if __name__ == "__main__":
  main()
