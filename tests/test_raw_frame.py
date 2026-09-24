"""One runnable check behind `scripts/build_raw_frame.py`.

The frame is the single input to every number in `docs/results/n40-sweep.md`,
and its two load-bearing claims are (a) the corpus can be regenerated exactly from
an `instance_id`, and (b) one graph serves all six tasks, so `graph_id` -- not
`instance_id` -- is the graph identity. Both are asserted here, plus the strategy
marker that a Q5 claim actually rests on.

  uv run --no-sync pytest -q tests/test_raw_frame.py
"""

import random

import networkx as nx

from graphtalk import diverse_corpus
from graphtalk import graphqa
from scripts import build_raw_frame as brf


def _graph(density, index):
  return graphqa.canonical(
      nx.erdos_renyi_graph(brf.SIZE, density, seed=brf.seed_of(density, index)))


def test_seed_formula_has_no_task_term():
  """One graph per (density, index), shared by all six tasks.

  If a task term ever enters the seed, `graph_id` stops being a graph identity
  and the clustered bootstrap silently resamples the wrong unit.
  """
  assert brf.seed_of(0.35, 7) == 20260906 + 1000000 * 351 + 1000 * 40 + 7
  # Distinct densities and indices never collide.
  seeds = {brf.seed_of(d, i) for d in (0.1, 0.2, 0.35, 0.5, 0.65, 0.75, 0.85)
           for i in range(100)}
  assert len(seeds) == 700


def test_regenerates_a_known_instance():
  """A spot check with values read off the regenerated graph, not from a run."""
  g = _graph(0.35, 7)
  assert g.number_of_nodes() == 40
  assert g.number_of_edges() == 276
  row = diverse_corpus.make_row(g, "node_degree", random.Random(brf.seed_of(0.35, 7)))
  assert row["targets"] == (9,)
  assert str(row["gold"]) == "13"
  assert g.degree(9) == 13


def test_three_tasks_share_the_queried_node():
  """node_degree, connected_nodes and edge_existence query the same node.

  This is what makes the output-operation contrast in Q3 a controlled comparison
  rather than three unrelated samples -- and what stops the six tasks being
  treated as six independent draws.
  """
  for density in (0.1, 0.5, 0.85):
    for index in (0, 13, 41):
      g = _graph(density, index)
      seed = brf.seed_of(density, index)
      nd = diverse_corpus.make_row(g, "node_degree", random.Random(seed))
      cn = diverse_corpus.make_row(g, "connected_nodes", random.Random(seed))
      ee = diverse_corpus.make_row(g, "edge_existence", random.Random(seed))
      assert nd["targets"][0] == cn["targets"][0] == ee["targets"][0]


def test_target_features_match_the_graph():
  g = _graph(0.5, 3)
  clustering = nx.clustering(g)
  row = diverse_corpus.make_row(g, "edge_existence", random.Random(brf.seed_of(0.5, 3)))
  a, b = row["targets"]
  feats = brf.target_features(g, clustering, "edge_existence", row["targets"],
                              row["gold"])
  assert feats["target_degree"] == g.degree(a)
  assert feats["pair_common_neighbors"] == len(list(nx.common_neighbors(g, a, b)))
  assert feats["gold_is_yes"] == int(g.has_edge(a, b))


def test_degree_sum_marker_fires_only_on_the_strategy():
  """`uses_degree_sum` is the one text marker a reported claim depends on.

  Its sibling `narrates_neighbors` was renamed after hand-validation showed it
  matches citations of the *edge list*, not the primer -- see the module comment
  in build_raw_frame.py. Keep this asymmetry: a marker earns a claim only by
  passing a check like this one.
  """
  rx = brf._MARKERS["uses_degree_sum"]
  assert rx.search("sum the degrees and divide by 2")
  assert rx.search("By the Handshaking Lemma, 2|E| = sum of degrees")
  assert not rx.search("Node 7 is connected to nodes 1, 2, 3. The degree is 3.")
  assert not rx.search("I counted 14 edges incident to node 0.")
