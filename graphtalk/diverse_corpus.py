"""A small, structurally-diverse graph pool, generated locally rather than
fetched from the published (ER-only) GraphQA split.

`scripts/measure_real_rows.py` established that the published `zero_shot_test`
split is exactly `graph_generators.generate_graphs(500, "er", ...)` -- every row
comes from one algorithm. This module builds an alternative pool spanning all
seven algorithms `talk_like_a_graph/graph_generators.py` implements, balanced so
each gets roughly equal representation, for comparing per-structure accuracy the
way the original paper's robustness experiments do.

`make_row` reproduces -- deliberately, not by importing -- the task wording and
uniform target-sampling used by `talk_like_a_graph/graph_tasks.py`'s
`prepare_examples_dict` methods (`node_degree`, `connected_nodes`,
`edge_existence` each sample `random.sample(graph.nodes(), k=...)`; the other
three tasks take no target). Reusing that module directly would pull in its
`graph_text_encoders` dependency for a one-line node-name lookup that, for this
project's plain-integer naming, is just `str(node)`.
"""

import random

import networkx as nx

from talk_like_a_graph import graph_generators

from graphtalk import graphqa

ALGORITHMS = ("er", "ba", "sbm", "sfn", "complete", "star", "path")


def build_pool(count: int, seed: int = 1234) -> list[tuple[str, nx.Graph]]:
  """Generates `count` graphs split as evenly as possible across `ALGORITHMS`.

  The first `count % len(ALGORITHMS)` algorithms (in `ALGORITHMS` order) get one
  extra graph, so counts differ by at most one -- "similar representation" per
  algorithm. Every graph is canonicalized, matching the invariant `primers.py`
  requires. Deterministic in `seed`.
  """
  base, extra = divmod(count, len(ALGORITHMS))
  pool = []
  for index, algorithm in enumerate(ALGORITHMS):
    n = base + (1 if index < extra else 0)
    graphs = graph_generators.generate_graphs(
        n, algorithm, directed=False, random_seed=seed
    )
    pool.extend((algorithm, graphqa.canonical(g)) for g in graphs)
  return pool


# Verbatim from talk_like_a_graph/graph_tasks.py's `prepare_examples_dict`
# methods -- kept as literals rather than imported so a wording change there is
# a visible diff here too, the same discipline `shortcuts.py` uses for
# `primers.py`'s render/parse round trip.
_NO_TARGET_TEMPLATES = {
    "node_count": "Q: How many nodes are in this graph?\nA: ",
    "edge_count": "Q: How many edges are in this graph?\nA: ",
    "cycle_check": "Q: Is there a cycle in this graph?\nA: ",
}


def make_row(graph: nx.Graph, task: str, rng: random.Random) -> dict:
  """Builds one task row (task_description, gold, targets) for `graph`.

  Targets are sampled with the given `rng` rather than the global `random`
  module, so graph generation (which does reseed the global module) and target
  sampling can be seeded independently and reproducibly.
  """
  if task in _NO_TARGET_TEMPLATES:
    targets: tuple[int, ...] = ()
    task_description = _NO_TARGET_TEMPLATES[task]
  elif task == "node_degree":
    node = rng.sample(sorted(graph.nodes()), k=1)[0]
    targets = (node,)
    task_description = "Q: What is the degree of node %s?\nA: " % node
  elif task == "connected_nodes":
    node = rng.sample(sorted(graph.nodes()), k=1)[0]
    targets = (node,)
    task_description = (
        "Q: List all the nodes connected to %s in alphabetical order.\nA: "
        % node
    )
  elif task == "edge_existence":
    source, target = rng.sample(sorted(graph.nodes()), k=2)
    targets = (source, target)
    task_description = (
        "Q: Does an edge exist between Node %s and Node %s?\nA: "
        % (source, target)
    )
  else:
    raise ValueError(f"no row template defined for task: {task}")

  gold = graphqa.gold_answer(graph, task, targets)
  return {"task_description": task_description, "gold": gold, "targets": targets}
