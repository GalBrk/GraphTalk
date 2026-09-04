"""Stage 1: materialise every prompt in the sweep to a JSONL file.

Split out from the model run on purpose. The prompt set is the experiment's
independent variable, so writing it to a file first means it can be read,
diffed and checked on a laptop, and that every model in the sweep is handed the
*identical* file rather than re-deriving prompts from a code path that might have
changed between runs. It also keeps this stage free of `torch`.

The design is paired, which is what the proposal's McNemar test requires: the
same graph and the same query appear under every primer condition and both prompt
styles, differing only in the primer. `instance_id` is the pairing key.

  PYTHONPATH=. .venv/bin/python scripts/build_prompts.py --count 30
"""

import argparse
import collections
import json
import os
import random

from graphtalk import graphqa
from graphtalk import node_naming
from graphtalk import primers
from graphtalk import prompts
from graphtalk import scoring

# The published split ships one particular query draw per row, and that draw is
# what a model gets scored against. Re-sampling queries here would score the model
# on questions the dataset never asked -- see the provenance section of
# docs/plans/primer-computation.md, where the two draws differ by five points on
# `edge_existence`.
SPLIT = "zero_shot_test"


def load_rows(config: str, count: int, split: str, cache: str) -> list[dict]:
  """Fetches (and caches) the first `count` rows of one task's split.

  The first N rather than a random sample: the published rows are already a
  shuffle of the generator's emission order -- only 2 of 500 sit at the same index
  -- so a prefix is an unbiased draw and is reproducible without a seed.
  """
  os.makedirs(cache, exist_ok=True)
  path = os.path.join(cache, f"{config}.{split}.{count}.json")
  if os.path.exists(path):
    with open(path) as handle:
      return json.load(handle)
  rows = graphqa.fetch_rows(config, split, 0, count)
  with open(path, "w") as handle:
    json.dump(rows, handle)
  return rows


def build(count: int, conditions, styles, split: str, cache: str,
          k_min: int, k_max: int) -> list[dict]:
  records = []
  for task in scoring.TASKS:
    rows = load_rows(task, count, split, cache)
    for index, row in enumerate(rows):
      graph = graphqa.parse_graph(row["question"])
      gold = row["answer"]
      # The gold answer ships with the row; recomputing it from the parsed graph
      # is a check that parsing recovered the right graph. A mismatch means the
      # prompt would describe a different graph than the answer was written for,
      # which is unrecoverable and must stop the build rather than be scored.
      recomputed = graphqa.expected_answer(graph, task, row["task_description"])
      if scoring.normalize(recomputed) != scoring.normalize(gold):
        raise ValueError(
            f"{task} row {index}: parsed graph disagrees with the shipped "
            f"answer ({recomputed!r} vs {gold!r})"
        )
      # Reworded after the check above, not before: `expected_answer` reads
      # query node ids straight out of this string, and keeping the check on
      # the dataset's pristine wording means it never has to assume anything
      # about the reworded text's digit layout.
      task_description = row["task_description"]
      if task == "edge_existence":
        task_description = graphqa.reword_edge_existence(task_description)
      for condition in conditions:
        for style in styles:
          records.append({
              "instance_id": f"{task}/{index}",
              "task": task,
              "condition": condition,
              "style": style,
              "prompt": prompts.build_prompt(
                  graph, condition, task_description,
                  style=style, k_min=k_min, k_max=k_max,
              ),
              "gold": gold,
              "nodes": graph.number_of_nodes(),
              "edges": graph.number_of_edges(),
          })
  return records


def build_diverse(count: int, conditions, styles, k_min: int, k_max: int,
                   seed: int = 1234) -> list[dict]:
  """Like `build`, but sources graphs from a balanced multi-algorithm pool
  (`diverse_corpus`) instead of the published (ER-only) zero_shot_test split.

  Builds one pool of `count` graphs and reuses it across every task, unlike
  `build`, which fetches a separately-shuffled `count` rows per task -- sharing
  the pool is what lets a later analysis compare per-algorithm success rate
  against a consistent graph set across tasks.

  `graphtalk.diverse_corpus` is imported here, not at module level, so
  that everything else in this script stays usable even when that module
  is missing (as of this writing, `graphtalk/diverse_corpus.py` doesn't
  exist in this checkout at all -- only `tests/test_diverse_corpus.py`
  was ever committed, a pre-existing gap unrelated to `--graph-source
  diverse` specifically; see CLAUDE.md/session notes). Only a caller who
  actually asks for `--graph-source diverse` pays for that gap.
  """
  from graphtalk import diverse_corpus
  pool = diverse_corpus.build_pool(count, seed=seed)
  records = []
  for task in scoring.TASKS:
    rng = random.Random(seed)
    seen = collections.Counter()
    for algorithm, graph in pool:
      index = seen[algorithm]
      seen[algorithm] += 1
      row = diverse_corpus.make_row(graph, task, rng)
      for condition in conditions:
        for style in styles:
          records.append({
              "instance_id": f"{task}/diverse/{algorithm}/{index}",
              "task": task,
              "condition": condition,
              "style": style,
              "prompt": prompts.build_prompt(
                  graph, condition, row["task_description"],
                  style=style, k_min=k_min, k_max=k_max,
              ),
              "gold": row["gold"],
              "nodes": graph.number_of_nodes(),
              "edges": graph.number_of_edges(),
              "algorithm": algorithm,
          })
  return records


def build_stratified(count: int, conditions, styles, split: str, cache: str,
                      k_min: int, k_max: int, pool_size: int = 500) -> list[dict]:
  """Like `build`, but selects the `count` *largest* graphs (by node
  count) out of a `pool_size`-row candidate pool per task, instead of
  simply the first `count` rows in split order.

  Track 2.2: near-ceiling models (`gemma4-12b`/`gemma4-e4b` in the main
  sweep, per `analysis/README.md`'s "Current significance results")
  barely produce any discordant pairs at `--count 30` -- there is almost
  nothing left to flip. Uniformly scaling `--count` raises the discordant
  count roughly proportionally, but `docs/sweep-findings.md`'s "Missing
  instances skew toward larger graphs" already establishes, on already-
  collected data, that larger graphs are where these models' errors (and,
  by the same logic, a primer's chance to change the verdict) concentrate.
  Oversampling large graphs raises the discordant-pair *yield per graph
  collected* instead of just the total graph count -- see
  `scripts/validate_stratified_sampling.py` for the data-driven check of
  that assumption against real, already-collected responses (Track 2.2's
  required validation) before spending any GPU time generating with this
  mode.

  `pool_size` (default 500, the published split's per-task cap -- see
  `SPLIT`) is fetched and parsed to rank by size; only the `count` largest
  survive to have prompts actually built for them, so this costs more
  `graphqa.fetch_rows`/`parse_graph` calls than `build` for the same
  `count`, but no more prompt-building. Ties (equal node count) break on
  original split index, for a deterministic selection at a fixed `split`.

  `instance_id`s are tagged `<task>/stratified/<original index>`, not
  bare `<task>/<index>` -- kept clearly distinguishable from `build`'s
  main-sweep corpus (mirrors `build_diverse`'s `<task>/diverse/...`
  convention) so a stratified run can never silently merge with, or be
  mistaken for, the main sweep's historical, comparable corpus in a
  downstream frame or analysis.
  """
  records = []
  for task in scoring.TASKS:
    candidates = load_rows(task, pool_size, split, cache)
    sized = []
    for index, row in enumerate(candidates):
      graph = graphqa.parse_graph(row["question"])
      sized.append((graph.number_of_nodes(), index, row, graph))
    sized.sort(key=lambda item: (-item[0], item[1]))
    for _, index, row, graph in sized[:count]:
      gold = row["answer"]
      recomputed = graphqa.expected_answer(graph, task, row["task_description"])
      if scoring.normalize(recomputed) != scoring.normalize(gold):
        raise ValueError(
            f"{task} row {index}: parsed graph disagrees with the shipped "
            f"answer ({recomputed!r} vs {gold!r})"
        )
      task_description = row["task_description"]
      if task == "edge_existence":
        task_description = graphqa.reword_edge_existence(task_description)
      for condition in conditions:
        for style in styles:
          records.append({
              "instance_id": f"{task}/stratified/{index}",
              "task": task,
              "condition": condition,
              "style": style,
              "prompt": prompts.build_prompt(
                  graph, condition, task_description,
                  style=style, k_min=k_min, k_max=k_max,
              ),
              "gold": gold,
              "nodes": graph.number_of_nodes(),
              "edges": graph.number_of_edges(),
          })
  return records


def build_named(count: int, conditions, styles, split: str, cache: str,
                 k_min: int, k_max: int, node_naming_scheme: str) -> list[dict]:
  """Like `build`, but every prompt uses `node_naming_scheme`'s node names.

  Reuses every existing building block unchanged (`load_rows`,
  `graphqa.parse_graph`, `graphqa.expected_answer`,
  `graphqa.reword_edge_existence`); only prompt construction and the extra
  `node_naming` record field differ from `build`.
  """
  records = []
  for task in scoring.TASKS:
    rows = load_rows(task, count, split, cache)
    for index, row in enumerate(rows):
      graph = graphqa.parse_graph(row["question"])
      name_map = node_naming.build_name_map(graph, node_naming_scheme)
      gold = row["answer"]
      recomputed = graphqa.expected_answer(graph, task, row["task_description"])
      if scoring.normalize(recomputed) != scoring.normalize(gold):
        raise ValueError(
            f"{task} row {index}: parsed graph disagrees with the shipped "
            f"answer ({recomputed!r} vs {gold!r})"
        )
      task_description = row["task_description"]
      if task == "edge_existence":
        task_description = graphqa.reword_edge_existence(task_description)
      for condition in conditions:
        for style in styles:
          records.append({
              "instance_id": f"{task}/{index}",
              "task": task,
              "condition": condition,
              "style": style,
              "prompt": node_naming.build_named_prompt(
                  graph, condition, task_description, name_map,
                  style=style, k_min=k_min, k_max=k_max,
              ),
              "gold": gold,
              "nodes": graph.number_of_nodes(),
              "edges": graph.number_of_edges(),
              "node_naming": node_naming_scheme,
          })
  return records


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--count", type=int, default=30,
                      help="rows per task for --graph-source published; total "
                           "graphs in the shared pool for --graph-source diverse "
                           "(the proposal's starting budget is 30)")
  parser.add_argument("--split", default=SPLIT)
  parser.add_argument("--out", default="prompts.jsonl")
  parser.add_argument("--cache", default=".cache/sweep_rows")
  parser.add_argument("--conditions", nargs="+", default=sorted(primers.CONDITIONS))
  parser.add_argument("--styles", nargs="+", default=list(prompts.PROMPT_STYLES))
  parser.add_argument("--k-min", type=int, default=2)
  parser.add_argument("--k-max", type=int, default=3)
  parser.add_argument("--node-naming", default="integer", choices=node_naming.NAMINGS)
  parser.add_argument("--graph-source", default="published",
                      choices=["published", "diverse", "stratified"],
                      help="published: fetch from the HF zero_shot_test split "
                           "(ER only, today's default, unchanged). diverse: "
                           "generate a pool balanced across er/ba/sbm/sfn/"
                           "complete/star/path locally (graphtalk.diverse_corpus). "
                           "stratified (Track 2.2): the --count LARGEST graphs "
                           "(by node count) out of a --pool-size candidate pool "
                           "from the published split, instead of the first "
                           "--count in split order -- for near-ceiling models, "
                           "where docs/sweep-findings.md's 'Missing instances "
                           "skew toward larger graphs' finding suggests larger "
                           "graphs yield more discordant pairs per graph "
                           "collected; see scripts/validate_stratified_sampling.py "
                           "before spending GPU time on this. diverse/stratified "
                           "are only supported with --node-naming integer.")
  parser.add_argument("--pool-size", type=int, default=500,
                      help="--graph-source stratified only: candidate pool size "
                           "per task to rank by graph size before taking the "
                           "--count largest (default 500, the published split's "
                           "per-task cap)")
  args = parser.parse_args()

  if args.graph_source in ("diverse", "stratified") and args.node_naming != "integer":
    raise NotImplementedError(
        f"--graph-source {args.graph_source} only supports --node-naming "
        f"integer for now"
    )
  if args.graph_source == "diverse":
    records = build_diverse(args.count, args.conditions, args.styles,
                            args.k_min, args.k_max)
  elif args.graph_source == "stratified":
    records = build_stratified(args.count, args.conditions, args.styles,
                               args.split, args.cache, args.k_min, args.k_max,
                               pool_size=args.pool_size)
  elif args.node_naming == "integer":
    records = build(args.count, args.conditions, args.styles, args.split,
                    args.cache, args.k_min, args.k_max)
  else:
    records = build_named(args.count, args.conditions, args.styles, args.split,
                          args.cache, args.k_min, args.k_max, args.node_naming)
  with open(args.out, "w") as handle:
    for record in records:
      handle.write(json.dumps(record) + "\n")

  instances = len({r["instance_id"] for r in records})
  print(f"wrote {len(records)} prompts to {args.out}")
  print(f"  {instances} instances x {len(args.conditions)} conditions "
        f"x {len(args.styles)} styles")
  print(f"  conditions: {', '.join(args.conditions)}")
  print(f"  styles:     {', '.join(args.styles)}")
  lengths = [len(r["prompt"]) for r in records]
  print(f"  prompt chars: min {min(lengths)}, mean {sum(lengths)//len(lengths)}, "
        f"max {max(lengths)}")


if __name__ == "__main__":
  main()
