"""The multi-algorithm graph pool used by `build_prompts.py --graph-source diverse`.

`build_pool`'s per-algorithm split and `make_row`'s wording/target-sampling are
the two places a bug would silently produce a skewed or mis-graded diverse run,
so both get pinned here: the wording literals are checked verbatim against
`talk_like_a_graph/graph_tasks.py`'s templates (the same discipline
`test_primers.py`'s round-trip test uses for `render_primer`/`parse_primer`),
and gold answers are cross-checked against `graphqa.gold_answer` directly rather
than re-derived.
"""

import random

from graphtalk import diverse_corpus
from graphtalk import graphqa
from graphtalk import scoring
from scripts import build_prompts


# --- build_pool -------------------------------------------------------------


def test_pool_counts_sum_to_requested_total():
  pool = diverse_corpus.build_pool(30)
  assert len(pool) == 30


def test_pool_counts_are_balanced_across_algorithms():
  pool = diverse_corpus.build_pool(30)
  counts = {alg: 0 for alg in diverse_corpus.ALGORITHMS}
  for algorithm, _ in pool:
    counts[algorithm] += 1
  assert set(counts) == set(diverse_corpus.ALGORITHMS)
  assert max(counts.values()) - min(counts.values()) <= 1


def test_pool_graphs_are_canonical():
  pool = diverse_corpus.build_pool(30)
  for _, graph in pool:
    assert list(graph.nodes()) == list(graphqa.canonical(graph).nodes())
    assert list(graph.edges()) == list(graphqa.canonical(graph).edges())


def test_pool_is_deterministic_given_a_seed():
  first = diverse_corpus.build_pool(30, seed=42)
  second = diverse_corpus.build_pool(30, seed=42)
  assert [(alg, sorted(g.edges())) for alg, g in first] == (
      [(alg, sorted(g.edges())) for alg, g in second]
  )


def test_small_count_still_splits_without_error():
  # Fewer graphs than algorithms: some algorithms legitimately get zero.
  pool = diverse_corpus.build_pool(3)
  assert len(pool) == 3


# --- make_row -----------------------------------------------------------


_EXPECTED_NO_TARGET_WORDING = {
    "node_count": "Q: How many nodes are in this graph?\nA: ",
    "edge_count": "Q: How many edges are in this graph?\nA: ",
    "cycle_check": "Q: Is there a cycle in this graph?\nA: ",
}


def _one_graph():
  return diverse_corpus.build_pool(1)[0][1]


def test_no_target_tasks_match_vendored_wording_verbatim():
  graph = _one_graph()
  rng = random.Random(0)
  for task, expected in _EXPECTED_NO_TARGET_WORDING.items():
    row = diverse_corpus.make_row(graph, task, rng)
    assert row["task_description"] == expected
    assert row["targets"] == ()


def test_node_degree_wording_and_gold():
  graph = _one_graph()
  rng = random.Random(0)
  row = diverse_corpus.make_row(graph, "node_degree", rng)
  (node,) = row["targets"]
  assert row["task_description"] == f"Q: What is the degree of node {node}?\nA: "
  assert row["gold"] == graphqa.gold_answer(graph, "node_degree", (node,))


def test_connected_nodes_wording_and_gold():
  graph = _one_graph()
  rng = random.Random(0)
  row = diverse_corpus.make_row(graph, "connected_nodes", rng)
  (node,) = row["targets"]
  assert row["task_description"] == (
      f"Q: List all the nodes connected to {node} in alphabetical order.\nA: "
  )
  assert row["gold"] == graphqa.gold_answer(graph, "connected_nodes", (node,))


def test_edge_existence_samples_two_distinct_nodes():
  graph = _one_graph()
  rng = random.Random(0)
  row = diverse_corpus.make_row(graph, "edge_existence", rng)
  source, target = row["targets"]
  assert source != target
  assert row["task_description"] == (
      f"Q: Does an edge exist between Node {source} and Node {target}?\nA: "
  )
  assert row["gold"] == graphqa.gold_answer(
      graph, "edge_existence", (source, target)
  )


def test_unknown_task_raises():
  graph = _one_graph()
  try:
    diverse_corpus.make_row(graph, "not_a_task", random.Random(0))
    assert False, "expected ValueError"
  except ValueError:
    pass


# --- build_prompts.build_diverse end-to-end ---------------------------------


def test_build_diverse_smoke():
  records = build_prompts.build_diverse(
      7, conditions=["none"], styles=["zero_shot"], k_min=2, k_max=3,
  )
  assert records
  for record in records:
    assert record["algorithm"] in diverse_corpus.ALGORITHMS
    assert record["prompt"]
    assert record["gold"]
    assert record["task"] in scoring.TASKS

  # 7 graphs (1 per algorithm) x 6 tasks x 1 condition x 1 style.
  assert len(records) == 7 * len(scoring.TASKS)
