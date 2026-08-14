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
import json
import os

from graphtalk import graphqa
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
      for condition in conditions:
        for style in styles:
          records.append({
              "instance_id": f"{task}/{index}",
              "task": task,
              "condition": condition,
              "style": style,
              "prompt": prompts.build_prompt(
                  graph, condition, row["task_description"],
                  style=style, k_min=k_min, k_max=k_max,
              ),
              "gold": gold,
              "nodes": graph.number_of_nodes(),
              "edges": graph.number_of_edges(),
          })
  return records


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--count", type=int, default=30,
                      help="rows per task; the proposal's starting budget is 30")
  parser.add_argument("--split", default=SPLIT)
  parser.add_argument("--out", default="prompts.jsonl")
  parser.add_argument("--cache", default=".cache/sweep_rows")
  parser.add_argument("--conditions", nargs="+", default=sorted(primers.CONDITIONS))
  parser.add_argument("--styles", nargs="+", default=list(prompts.PROMPT_STYLES))
  parser.add_argument("--k-min", type=int, default=2)
  parser.add_argument("--k-max", type=int, default=3)
  args = parser.parse_args()

  records = build(args.count, args.conditions, args.styles, args.split,
                  args.cache, args.k_min, args.k_max)
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
