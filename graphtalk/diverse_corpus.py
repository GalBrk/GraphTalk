"""A graph pool balanced across the vendored generator's seven algorithms.

`build_prompts.py --graph-source diverse` exists because the published
`zero_shot_test` split is Erdos-Renyi only (see `graphqa.py`'s module
docstring) -- everything this project measures about a primer's effect is
therefore implicitly scoped to one graph family. `talk_like_a_graph
/graph_generators.py` (vendored, unmodified) already implements seven --
`er`, `ba`, `sbm`, `sfn`, `complete`, `star`, `path` -- and
`graphtalk/shortcuts.py::generate_corpus` already leans on it for its own
(ER-only) fitted-rule corpus. This module is the same idea generalized to
all seven, so a diverse run can ask whether a primer's effect is an
artifact of ER's specific degree distribution.

Reconstructed from its own usage contract (`tests/test_diverse_corpus.py`
and `scripts/build_prompts.py::build_diverse`) after discovering this file
was never actually committed -- see the quarantine note at the top of
`tests/test_diverse_corpus.py` for how that was confirmed. Every wording
and gold-answer choice below is pinned by that existing test file, not
invented fresh.
"""

import random

import networkx as nx

from graphtalk import graphqa

ALGORITHMS = ("er", "ba", "sbm", "sfn", "complete", "star", "path")


def build_pool(
    count: int,
    seed: int = 1234,
    node_size_ranges: dict | None = None,
    er_min_sparsity: float = 0.0,
    er_max_sparsity: float = 1.0,
    algorithms: tuple[str, ...] | None = None,
) -> list[tuple[str, nx.Graph]]:
  """`count` canonical graphs, spread as evenly as possible across
  `algorithms` (default: all of `ALGORITHMS`) -- a `count` not a multiple
  of `len(algorithms)` puts the remainder on the first few algorithms in
  `algorithms` order, and a `count` smaller than `len(algorithms)` leaves
  the rest at zero -- both are fine, not edge cases to avoid.

  One `graph_generators.generate_graphs` call per algorithm rather than
  one shared call, since the vendored function takes a single `algorithm`
  string, not a mix. Each call gets its own seed derived from `seed` (not
  the same seed reused seven times) so per-algorithm draws don't
  coincidentally share the same size/sparsity sequence -- determinism
  only requires `build_pool(count, seed)` itself to be a pure function of
  its inputs, not that sub-calls be independently seeded in any special
  way. The seed offset is `ALGORITHMS.index(algorithm)`, not the filtered
  list's position, so restricting `algorithms` never changes the seed a
  surviving algorithm draws from -- `build_pool(count, algorithms=("er",))`
  reproduces exactly the `"er"` rows `build_pool(count)` would have drawn.

  `node_size_ranges` and `er_min_sparsity`/`er_max_sparsity` pass straight
  through to `graph_generators.generate_graphs` (same defaults, so omitting
  them reproduces today's behavior exactly). `er_min_sparsity`/
  `er_max_sparsity` only affect the `"er"` algorithm -- the vendored
  generator consumes them nowhere else, so the other six algorithms in the
  pool are unaffected by them. `algorithms` is the filter that makes this
  matter in practice: density is only really controllable for `"er"`, so a
  density sweep wants `algorithms=("er",)` rather than spending 6/7 of
  `count` on algorithms `er_min_sparsity`/`er_max_sparsity` can't touch.

  Canonicalized the same way `shortcuts.generate_corpus` canonicalizes its
  own generated graphs, for the same reason: the vendored encoder's output
  depends on iteration order, so every graph handed downstream needs to
  already be in sorted node/edge order.
  """
  # Imported here, not at module scope, so that importing `diverse_corpus`
  # for its constants (e.g. `ALGORITHMS`) doesn't require the vendored
  # `talk_like_a_graph` package to be importable -- mirrors
  # `shortcuts.generate_corpus`'s own lazy import, same rationale.
  from talk_like_a_graph import graph_generators  # pylint: disable=g-import-not-at-top

  algorithms = algorithms or ALGORITHMS
  invalid = set(algorithms) - set(ALGORITHMS)
  if invalid:
    raise ValueError(
        f"unknown algorithm(s): {sorted(invalid)}; known: {list(ALGORITHMS)}"
    )
  if not algorithms:
    raise ValueError("algorithms must be non-empty")

  base, remainder = divmod(count, len(algorithms))
  pool = []
  for index, algorithm in enumerate(algorithms):
    n = base + (1 if index < remainder else 0)
    if n == 0:
      continue
    graphs = graph_generators.generate_graphs(
        n, algorithm, directed=False,
        random_seed=seed + ALGORITHMS.index(algorithm),
        node_size_ranges=node_size_ranges,
        er_min_sparsity=er_min_sparsity,
        er_max_sparsity=er_max_sparsity,
    )
    for graph in graphs:
      pool.append((algorithm, graphqa.canonical(graph)))
  return pool


def make_row(graph: nx.Graph, task: str, rng: random.Random) -> dict:
  """One `(task_description, targets, gold)` row for `graph`, wording
  matched verbatim to `talk_like_a_graph/graph_tasks.py`'s own templates
  (grep-checked, not retyped from memory) so a diverse-sourced prompt reads
  identically to a published-split one for the same task. `gold` is always
  `graphqa.gold_answer` applied directly to `graph`/`targets` -- never
  re-derived -- so a bug in this function's own wording can't also corrupt
  its answer key.
  """
  nodes = list(graph.nodes())
  if task == "node_count":
    task_description = "Q: How many nodes are in this graph?\nA: "
    targets: tuple = ()
  elif task == "edge_count":
    task_description = "Q: How many edges are in this graph?\nA: "
    targets = ()
  elif task == "cycle_check":
    task_description = "Q: Is there a cycle in this graph?\nA: "
    targets = ()
  elif task == "node_degree":
    node = rng.choice(nodes)
    task_description = f"Q: What is the degree of node {node}?\nA: "
    targets = (node,)
  elif task == "connected_nodes":
    node = rng.choice(nodes)
    task_description = (
        f"Q: List all the nodes connected to {node} in alphabetical order.\nA: "
    )
    targets = (node,)
  elif task == "edge_existence":
    source, target = rng.sample(nodes, 2)
    task_description = (
        f"Q: Does an edge exist between Node {source} and Node {target}?\nA: "
    )
    targets = (source, target)
  elif task == "reachability":
    source, target = rng.sample(nodes, 2)
    task_description = (
        f"Q: Is there a path from node {source} to node {target}?\nA: "
    )
    targets = (source, target)
  else:
    raise ValueError(f"no row template defined for task: {task!r}")
  return {
      "task_description": task_description,
      "targets": targets,
      "gold": graphqa.gold_answer(graph, task, targets),
  }
