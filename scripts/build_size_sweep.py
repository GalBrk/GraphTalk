"""Build a size-scaling prompt set: ER graphs well beyond the published range.

The published `zero_shot_test` split, and the vendored generator behind it, cap
node counts at 19 (`talk_like_a_graph/graph_generators._NUMBER_OF_NODES_RANGE`
is small 5-9 / medium 10-14 / large 15-19). Nothing in the tracked corpus
answers "does this model degrade as graphs get big?", because the corpus has no
big graphs.

This script generates its own graphs at chosen sizes, keeping the published
corpus's density policy -- Erdos-Renyi with sparsity ~ U(0, 1) -- so a size
class differs from the tracked corpus in size and nothing else.

  PYTHONPATH=. python scripts/build_size_sweep.py --sizes 20 40 80 --count 50

**Why sizes stop at ~80.** Under U(0, 1) sparsity, edges grow as O(n^2) and the
`incident` encoding lists every one of them, so prompt length does too. Measured
against the Qwen3 tokenizer (both 1.7B and 8B have a 40,960-token context):

    n=20    723 /   948 /  1,242 tokens (p10/med/p90)
    n=40  2,480 / 3,367 /  4,932
    n=80  9,503 / 13,379 / 19,763   <- worst observed 22,093, still fits
    n=160 40,445 / 57,217 / 87,413  <- even p10 exceeds the context

n=160 is not a budgeting problem, it is impossible: the median graph needs 57k
tokens of a 41k window. Going bigger requires abandoning U(0, 1) sparsity for a
fixed average degree (edges linear in n, ~13k tokens at n=320) -- a different
density regime, and therefore a different experiment, not a bigger version of
this one.

Task set is deliberately four of the six. `edge_count` is excluded because a
large graph has hundreds of edges and the model enumerates them, so it hits the
generation budget and fails by *truncation* rather than by inability -- the
measurement would be of `max_new_tokens`, not of the model. `cycle_check` is
excluded because every graph this dense has a cycle, so gold is "yes" for every
instance and the task degenerates.

Gold answers come from `graphqa.gold_answer` via `diverse_corpus.make_row`,
never re-derived here, so a bug in this script cannot corrupt its own answer key.
"""

import argparse
import json
import random

import networkx as nx

from graphtalk import diverse_corpus
from graphtalk import graphqa
from graphtalk import prompts
from graphtalk import scoring

# Four of six -- see the module docstring for why edge_count and cycle_check
# are not size-scaling questions.
TASKS = ("node_count", "node_degree", "connected_nodes", "edge_existence")


def build(sizes, count, conditions, seed, style="zero_shot", densities=None,
          tasks=TASKS):
  """Prompts for one graph per (density level, size, index).

  `densities=None` is the original size-sweep behaviour: sparsity is drawn
  per graph from U(0, 1), the instance_id carries only the size class, and
  no density keys are emitted -- so re-running this script with no new flags
  reproduces `prompts.sizesweep.jsonl` byte for byte. Passing explicit
  levels pins sparsity to each one instead (`random.uniform(p, p) == p`),
  which is what turns density from corpus noise into a controlled variable.
  """
  records = []
  density_levels = list(densities) if densities else [None]
  for density in density_levels:
    for size in sizes:
      for index in range(count):
        if density is None:
          # One seed per (size, index) so a graph is reproducible from its
          # instance_id alone, and so two size classes never share a draw.
          s = seed + 1000 * size + index
        else:
          # A distinct block per density level, keyed on the level's *value*
          # and not its position in `densities`. Keying on position would mean
          # p=0.5 drew one graph when passed alone and a different one when
          # passed second -- while keeping the same instance_id. run_sweep.py
          # resumes by instance_id, so a re-run over a subset of levels would
          # then pair a fresh graph with an already-generated key and skip it,
          # silently mixing two corpora inside one cell. Levels are therefore
          # distinguished to three decimal places.
          s = (seed + 1000000 * (1 + int(round(density * 1000)))
               + 1000 * size + index)
        rng = random.Random(s)
        sparsity = rng.uniform(0.0, 1.0) if density is None else density
        graph = graphqa.canonical(
            nx.erdos_renyi_graph(size, sparsity, seed=s)
        )
        for task in tasks:
          row = diverse_corpus.make_row(graph, task, random.Random(s))
          for condition in conditions:
            # The density level joins the instance_id for the same reason the
            # size class does: two cells that differ only in sparsity must not
            # collide into one key, or run_sweep.py would treat the second as
            # already generated and skip it in silence.
            suffix = "" if density is None else f"/p{density:g}"
            record = {
                # Tagged with the size class so no downstream frame can pool a
                # size sweep row with a tracked-corpus row of the same index.
                "instance_id": f"{task}/size{size}{suffix}/{index}",
                "task": task,
                "condition": condition,
                "style": style,
                "prompt": prompts.build_prompt(
                    graph, condition, row["task_description"], style=style
                ),
                "gold": row["gold"],
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "size_class": size,
            }
            if density is not None:
              record["density_class"] = density
              record["density"] = sparsity
            records.append(record)
  return records


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 80])
  parser.add_argument("--count", type=int, default=50,
                      help="graphs per size class")
  parser.add_argument("--conditions", nargs="+", default=["none"])
  parser.add_argument("--densities", type=float, nargs="+", default=None,
                      help="pin ER sparsity to each of these levels instead "
                           "of drawing it from U(0, 1); one cell per "
                           "(density, size). Omit for the original behaviour.")
  parser.add_argument("--tasks", nargs="+", default=list(TASKS),
                      help=f"default: {' '.join(TASKS)}")
  parser.add_argument("--seed", type=int, default=20260906)
  parser.add_argument("--out", default="prompts.sizesweep.jsonl")
  args = parser.parse_args()

  for task in args.tasks:
    if task not in scoring.TASKS:
      parser.error(f"unknown task {task!r}; known: {' '.join(scoring.TASKS)}")

  records = build(args.sizes, args.count, args.conditions, args.seed,
                  densities=args.densities, tasks=tuple(args.tasks))
  with open(args.out, "w") as handle:
    for record in records:
      handle.write(json.dumps(record) + "\n")

  print(f"wrote {len(records)} prompts to {args.out}")
  for density in (args.densities or [None]):
    for size in args.sizes:
      rows = [r for r in records if r["size_class"] == size
              and r.get("density_class") == density]
      if not rows:
        continue
      edges = sorted(r["edges"] for r in rows)
      chars = sorted(len(r["prompt"]) for r in rows)
      label = f"size {size:>4}" + ("" if density is None else f" p={density:<5g}")
      print(f"  {label}: {len(rows):>5} prompts, "
            f"edges med {edges[len(edges)//2]:>5}, "
            f"prompt chars med {chars[len(chars)//2]:>7,} max {chars[-1]:>7,}")


if __name__ == "__main__":
  main()
