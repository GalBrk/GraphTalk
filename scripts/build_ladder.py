"""Build prompt files for the shared difficulty ladder.

Two modes, because the ladder is run in two passes with very different costs:

  * `--stage screen` -- `none` only, every rung, a handful of graphs each. This
    is the cheap pass whose output is the model x rung matrix: where each model
    sits on each rung, and therefore which rungs are valid primer tests for it.
    One condition, no primer, short answers.
  * `--stage rewire` -- the experiment. For each base graph, three rewiring
    levels (low / base / high) x the requested conditions. Every level of a base
    graph is the *same graph* with the same degree sequence, so the rows form
    within-instance matched triples and the gold answer is shared across them.

`instance_id` encodes the rung and, in rewire mode, the level and base graph:
`node_degree/n80k8/base/17` and `node_degree/n80k8/high/17` are the same graph.
Scoring and pairing both key off this, so it is built from the parameters rather
than a counter -- resuming a partial run has to land on the same ids.

Graph generation goes through `graphqa.canonical` exactly as `diverse_corpus`
does, so re-encoding the same graph is reproducible; rewiring is applied
*before* canonicalisation so the canonical form describes what the model sees.
"""

import argparse
import json
import random

import networkx as nx

from graphtalk import graphqa
from graphtalk import ladder
from graphtalk import primers
from graphtalk import prompts
from graphtalk import rewiring

LEVELS = ("low", "base", "high")


def _rung_tag(n: int, degree: int) -> str:
  return f"n{n}k{degree}"


def _rows_for_graph(graph, task, rng, rung, level, index, conditions, style):
  """Every (condition) row for one graph, sharing one question and gold answer.

  The question is drawn once and reused across conditions and rewiring levels:
  asking a different node per condition would make the paired comparison a
  comparison of different questions.
  """
  from graphtalk import diverse_corpus
  canonical = graphqa.canonical(graph)
  row = diverse_corpus.make_row(canonical, task, rng)
  out = []
  for condition in conditions:
    prompt = prompts.build_prompt(canonical, condition, row["task_description"],
                                  style=style)
    out.append({
        "instance_id": f"{task}/{rung}/{level}/{index}",
        "task": task,
        "condition": condition,
        "style": style,
        "prompt": prompt,
        "gold": row["gold"],
        "nodes": canonical.number_of_nodes(),
        "edges": canonical.number_of_edges(),
        "algorithm": "er_rewired",
        "rung": rung,
        "level": level,
    })
  return out


def build(stage, task, count, conditions, style, seed, mult, rungs=None,
          index_offset=0):
  rows = []
  selected = 0
  for n, degree, _tokens in ladder.RUNGS:
    rung = _rung_tag(n, degree)
    if rungs is not None and rung not in rungs:
      continue
    selected += 1
    edges = int(round(degree * n / 2))
    for index in range(index_offset, index_offset + count):
      # index doubles as the graph seed offset AND as the instance_id suffix,
      # so topping up an existing run means shifting BOTH together -- a top-up
      # that reused index 0.. would collide with the first batch's ids while
      # silently drawing different graphs.
      graph_seed = seed + index
      base = nx.gnm_random_graph(n, edges, seed=graph_seed)
      # One rng per (rung, index) so the question drawn for a graph does not
      # depend on how many conditions or levels were requested.
      rng = random.Random(graph_seed)
      if stage == "screen":
        rows.extend(_rows_for_graph(base, task, rng, rung, "base", index,
                                    conditions, style))
      else:
        low, high = rewiring.rewired_pair(base, mult=mult, seed=graph_seed)
        for level, graph in (("low", low), ("base", base), ("high", high)):
          # Same rng seed per level: the question must be identical across the
          # triple, or the levels are not paired.
          rows.extend(_rows_for_graph(graph, task, random.Random(graph_seed),
                                      rung, level, index, conditions, style))
  if rungs is not None and selected != len(rungs):
    # A typo in --rungs must not silently shrink the experiment.
    known = {_rung_tag(n, d) for n, d, _t in ladder.RUNGS}
    raise SystemExit(f"unknown rung(s): {sorted(set(rungs) - known)}; "
                     f"known: {sorted(known)}")
  return rows


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage", choices=("screen", "rewire"), required=True)
  parser.add_argument("--task", default="node_degree")
  parser.add_argument("--count", type=int, default=50,
                      help="base graphs per rung")
  parser.add_argument("--conditions", default=None,
                      help="comma-separated; defaults to 'none' for screen and "
                           "'none,clustering,filler' for rewire")
  parser.add_argument("--style", default="zero_shot")
  parser.add_argument("--seed", type=int, default=20260910)
  parser.add_argument("--mult", type=float, default=rewiring.DEFAULT_MULT)
  parser.add_argument("--index-offset", type=int, default=0,
                      help="first graph index; set it to the previous run's "
                           "--count to draw a disjoint top-up batch whose "
                           "instance_ids do not collide with it")
  parser.add_argument("--rungs", default=None,
                      help="comma-separated rung tags (e.g. 'n40k12,n60k16'); "
                           "defaults to every rung. Stage 3 should pass only "
                           "the rungs that cleared BOTH gates for the model "
                           "being run -- see analysis/ladder_matrix.limited.csv")
  parser.add_argument("--out", required=True)
  args = parser.parse_args()

  if args.conditions:
    conditions = tuple(c.strip() for c in args.conditions.split(","))
  else:
    conditions = ("none",) if args.stage == "screen" else (
        "none", "clustering", "filler")
  unknown = [c for c in conditions if c not in primers.CONDITIONS]
  if unknown:
    raise SystemExit(f"unknown condition(s): {unknown}; "
                     f"known: {sorted(primers.CONDITIONS)}")

  rungs = (frozenset(r.strip() for r in args.rungs.split(",")) if args.rungs
           else None)
  rows = build(args.stage, args.task, args.count, conditions, args.style,
               args.seed, args.mult, rungs, args.index_offset)
  with open(args.out, "w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row) + "\n")

  levels = len(LEVELS) if args.stage == "rewire" else 1
  print(f"{len(rows)} rows -> {args.out}")
  print(f"  {len(rungs or ladder.RUNGS)} rungs x {args.count} graphs x {levels} level(s) "
        f"x {len(conditions)} condition(s)")
  print(f"  conditions: {', '.join(conditions)}")


if __name__ == "__main__":
  main()
