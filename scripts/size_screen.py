"""Per-(model, size, task, condition) significance screen for the qwen3-1.7b
node-size sweep (`cluster/run_size_sweep.sh`'s `--node-count` runs).

Structurally the same screen as `scripts/task_scoped_screen.py` (reuses its
pairing/permutation/bootstrap machinery directly), extended with two axes
that script doesn't have: graph size, and the thinking arm compared
side-by-side with the plain one instead of filtered out.

Response rows carry no node-count field -- only the `prompts_..._n<size>
.jsonl` file(s) used to build them do (via `build_prompts.py
--node-count`). Since `--node-count N` fixes every graph in a
`--graph-source diverse` pool at exactly N nodes (verified: a built pool
of 49 prompts at `--node-count 80` had `nodes == 80` on all 49), size is
trusted from which response file(s) a row came from -- both
`--responses-pattern` and `--prompts-pattern` match on a `n<size>`
filename fragment, tolerant of the task tag `cluster/run_size_sweep.sh`
puts in front of it (`runs/<model>.<task(s)>_n<size>.jsonl`,
`prompts_<task(s)>_n<size>.jsonl` -- a task-qualified tag so rerunning
with a different `--tasks`/`TASKS` value never overwrites another task's
prompts or output, and multiple task-tagged files at the same size are
all picked up and merged rather than requiring one file per size).
Verification against the prompts file(s) only happens when at least one
matches -- its absence is not an error, since the prompts files don't
necessarily travel with an uploaded `runs/` directory. Primer condition
is never a filename axis: every condition already lives as a column
within one file, for every task/model, matching how every other script
in this project (`score_sweep.py`, `build_sweep_frame.py`, ...) already
treats it.

  PYTHONPATH=. .venv/bin/python scripts/size_screen.py \
      --model-family qwen3-1.7b --sizes 20 40 80 --shortcuts shortcuts.json
"""

import argparse
import glob
import json

import pandas as pd

from graphtalk import analysis
from graphtalk import significance
from scripts import check_significance as cs
from scripts import score_sweep


def _load_prompt_sizes(paths: list[str]) -> dict:
  """instance_id -> nodes, merged across every matching
  `build_prompts.py --node-count` file (one per task tag, per the
  `prompts_<task(s)>_n<size>.jsonl` convention). Different tasks never
  collide on `instance_id` (it's prefixed by task name), so a plain merge
  is safe; a real collision -- the same instance_id claiming two
  different node counts -- means two of the matched files disagree about
  size and is a build/naming mistake worth failing loudly on rather than
  silently picking one.
  """
  sizes = {}
  for path in paths:
    with open(path) as handle:
      for line in handle:
        if not line.strip():
          continue
        record = json.loads(line)
        instance_id, nodes = record["instance_id"], record["nodes"]
        if instance_id in sizes and sizes[instance_id] != nodes:
          raise ValueError(
              f"{instance_id!r} claims {sizes[instance_id]} nodes in one "
              f"matched prompts file and {nodes} in {path} -- files may "
              f"be mismatched"
          )
        sizes[instance_id] = nodes
  return sizes


def load_size_frame(
    model: str, size: int, responses_pattern: str, prompts_pattern: str,
    shortcuts_by_cell: dict,
) -> pd.DataFrame | None:
  """Scored frame for one (model, size), tagged with a `nodes` column, or
  `None` if no response file exists yet for this combination.
  """
  paths = sorted(
      p for p in glob.glob(responses_pattern.format(model=model, size=size))
      if not analysis.is_excluded(p)
  )
  if not paths:
    return None

  records = score_sweep.load(paths)
  score_sweep.desubstitute_named_responses(records)
  score_sweep.score_records(records)
  frame = analysis.build_frame(records, set(), shortcuts_by_cell)

  prompts_paths = sorted(glob.glob(prompts_pattern.format(size=size)))
  expected = _load_prompt_sizes(prompts_paths) if prompts_paths else None
  if expected is not None:
    actual = frame["instance_id"].map(expected)
    missing = frame[actual.isna()]
    if not missing.empty:
      raise ValueError(
          f"{len(missing)} row(s) in {paths} have no matching instance_id "
          f"in {prompts_paths}"
      )
    mismatched = frame[actual != size]
    if not mismatched.empty:
      raise ValueError(
          f"{len(mismatched)} row(s) in {paths} report a node count other "
          f"than {size} per {prompts_paths} -- files may be mismatched"
      )

  frame["nodes"] = size
  return frame


def screen(
    frame: pd.DataFrame, n_perm: int = 10_000, n_boot: int = 10_000,
    alpha: float = 0.05, seed: int = 1234, metric: str = "primary",
) -> list[dict]:
  """One row per (model, size, task, condition) cell in `frame`,
  `condition != "none"` and not a derived condition (`all`). `model` is the
  raw key (e.g. `qwen3-1.7b` vs. `qwen3-1.7b-think`), kept separate rather
  than collapsed to `model_family`, so the thinking arm is a normal row
  here instead of being filtered out the way `task_scoped_screen.screen`
  does.
  """
  rows = []
  for model in sorted(frame["model"].unique()):
    model_frame = frame[frame["model"] == model]
    for nodes in sorted(model_frame["nodes"].unique()):
      size_frame = model_frame[model_frame["nodes"] == nodes]
      for task in sorted(size_frame["task"].unique()):
        task_frame = size_frame[size_frame["task"] == task]
        conditions = sorted(
            c for c in task_frame["condition"].unique()
            if c != cs.CONTROL and not cs._is_derived_condition(c)
        )
        for condition in conditions:
          control, treatment, cluster_ids = cs._paired_values(
              task_frame, condition, metric
          )
          if not control:
            continue
          cell_seed = f"{seed}:{model}:{nodes}:{task}:{condition}"
          perm = significance.paired_permutation_test_clustered(
              control, treatment, cluster_ids, n_perm=n_perm, seed=cell_seed
          )
          boot = significance.cluster_bootstrap_ci_clustered(
              control, treatment, cluster_ids, n_boot=n_boot, seed=cell_seed,
              alpha=alpha,
          )
          rows.append({
              "model": model,
              "is_think": model.endswith("-think"),
              "nodes": nodes,
              "task": task,
              "condition": condition,
              "n_clusters": perm["n_clusters"],
              "success_rate_none": sum(control) / len(control),
              "success_rate_condition": sum(treatment) / len(treatment),
              "delta": perm["observed_diff"],
              "ci_low": boot["ci_low"],
              "ci_high": boot["ci_high"],
              "p_value": perm["p_value"],
          })
  return rows


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-family", default="qwen3-1.7b")
  parser.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 80])
  parser.add_argument("--responses-pattern", default="runs/{model}.*n{size}*.jsonl",
                      help="{model} and {size} are filled in per (model, size); "
                           "the leading '*' tolerates a task tag before "
                           "'n{size}' (e.g. runs/qwen3-1.7b.node_degree_n20"
                           ".jsonl), and every matching file is merged")
  parser.add_argument("--prompts-pattern", default="prompts_*n{size}.jsonl",
                      help="{size} is filled in per size; every matching file "
                           "(one per task tag, if more than one task has been "
                           "built at this size) is merged to verify node "
                           "counts, skipped entirely if none exist")
  parser.add_argument("--shortcuts", default="shortcuts.json",
                      help="pass '' to skip shortcut-score enrichment")
  parser.add_argument("--metric", default="primary", choices=["exact", "primary"])
  parser.add_argument("--n-perm", type=int, default=10_000)
  parser.add_argument("--n-boot", type=int, default=10_000)
  parser.add_argument("--alpha", type=float, default=0.05)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--out", default=None,
                      help="default analysis/size_screen.<model-family>.csv")
  args = parser.parse_args()

  shortcuts_by_cell = analysis.load_shortcuts(args.shortcuts) if args.shortcuts else {}

  frames = []
  for model in (args.model_family, f"{args.model_family}-think"):
    for size in args.sizes:
      frame = load_size_frame(
          model, size, args.responses_pattern, args.prompts_pattern,
          shortcuts_by_cell,
      )
      if frame is None:
        print(f"skipping {model} @ {size} nodes -- no response file found")
        continue
      frames.append(frame)

  if not frames:
    raise SystemExit("no response files found for any (model, size) combination")

  combined = pd.concat(frames, ignore_index=True)
  rows = screen(
      combined, n_perm=args.n_perm, n_boot=args.n_boot, alpha=args.alpha,
      seed=args.seed, metric=args.metric,
  )
  result = pd.DataFrame(rows).sort_values(
      ["task", "condition", "nodes", "is_think"]
  )

  out = args.out or f"analysis/size_screen.{args.model_family}.csv"
  result.to_csv(out, index=False)
  print(f"wrote {len(result)} rows to {out}")


if __name__ == "__main__":
  main()
