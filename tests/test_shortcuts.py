"""Tests for the primer parser. No network.

The round trip is the point of this file: render a primer, parse it back, and the
recovered values must equal the rounded originals exactly. That checks
`primers.render_primer` and `shortcuts.parse_primer` against each other, and it
is the only check in the repo that would catch a renderer change nobody meant to
make -- the primer tests assert arithmetic and golden strings, neither of which
notices if the two sides drift together.

The corpus is generated rather than hand-built so that the round trip sees real
distributions, and `test_corpus_is_not_vacuous` asserts it actually contains the
awkward cases (isolated nodes, triangle-free graphs, multi-component graphs)
rather than trusting the seed.
"""

import networkx as nx
import pytest

from graphtalk import graphqa
from graphtalk import primers
from graphtalk import shortcuts
from talk_like_a_graph import graph_generators

CORPUS_SIZE = 40
CORPUS_SEED = 1234


def corpus() -> list[nx.Graph]:
  graphs = graph_generators.generate_graphs(
      CORPUS_SIZE, "er", False, random_seed=CORPUS_SEED
  )
  return [graphqa.canonical(g) for g in graphs]


CORPUS = corpus()

EDGE_CASES = {
    "empty": nx.Graph(),
    "single": nx.empty_graph(1),
    "pair": nx.Graph([(0, 1)]),
    "triangle": nx.Graph([(0, 1), (1, 2), (0, 2)]),
    "edgeless": nx.empty_graph(4),
    "isolated": nx.Graph([(0, 1), (1, 2)]),
    "star_high": nx.Graph([(7, 0), (7, 1), (7, 2)]),
    "two_triangles": nx.Graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]),
}
EDGE_CASES["isolated"].add_node(9)
EDGE_CASES = {k: graphqa.canonical(v) for k, v in EDGE_CASES.items()}


def check_round_trip(graph, condition, k_min=2, k_max=3, target_chars=None):
  """Renders, parses, and asserts every recovered value matches the original."""
  text = primers.build_primer(
      graph, condition, k_min=k_min, k_max=k_max, target_chars=target_chars
  )
  parsed = shortcuts.parse_primer(text)
  parts = primers.CONDITIONS[condition]
  padded = target_chars is not None and bool(text)
  nodes = sorted(graph.nodes())

  if "components" in parts:
    assert parsed.components == primers.component_count(graph)
  else:
    assert parsed.components is None

  if "degree" in parts:
    assert parsed.degree == primers.degrees(graph)
  else:
    assert parsed.degree == {}

  if "clustering" in parts:
    expected = primers.clustering(graph)
    assert set(parsed.clustering) == set(expected)
    for node, value in expected.items():
      assert primers._fmt(parsed.clustering[node]) == primers._fmt(value)
  else:
    assert parsed.clustering == {}

  if "rwse" in parts:
    expected = primers.rwse(graph, k_min=k_min, k_max=k_max)
    assert set(parsed.rwse) == set(expected)
    for node, table in expected.items():
      assert set(parsed.rwse[node]) == set(table)
      for k, value in table.items():
        assert primers._fmt(parsed.rwse[node][k]) == primers._fmt(value)
  else:
    assert parsed.rwse == {}

  others = graph.number_of_nodes() - 1
  if "filler" in parts:
    assert parsed.filler == {node: others for node in nodes}
  elif padded:
    # Padding repeats the length-control sentence, so filler may appear for a
    # prefix of the nodes even when it is not one of the condition's parts.
    assert all(value == others for value in parsed.filler.values())
  else:
    assert parsed.filler == {}

  if any(part in primers.NODE_PARTS for part in parts):
    assert parsed.nodes == tuple(nodes)
  elif padded:
    assert set(parsed.nodes) <= set(nodes)
  else:
    assert parsed.nodes == ()

  return text, parsed


# --- the round trip -------------------------------------------------------


@pytest.mark.parametrize("condition", sorted(primers.CONDITIONS))
def test_round_trip_on_corpus(condition):
  for graph in CORPUS:
    check_round_trip(graph, condition)


@pytest.mark.parametrize("condition", sorted(primers.CONDITIONS))
def test_round_trip_on_edge_cases(condition):
  for name, graph in EDGE_CASES.items():
    check_round_trip(graph, condition), name


@pytest.mark.parametrize("k_min,k_max", [(1, 3), (2, 2), (2, 5), (1, 6)])
def test_round_trip_across_k_ranges(k_min, k_max):
  # k=1 exercises the singular "after 1 step", and a range of three or more
  # exercises the Oxford comma *inside* the RWSE phrase, which is the case the
  # step splitter has to get right.
  for graph in CORPUS[:10]:
    for condition in ("rwse", "all"):
      check_round_trip(graph, condition, k_min=k_min, k_max=k_max)


@pytest.mark.parametrize("condition", sorted(primers.CONDITIONS))
@pytest.mark.parametrize("target_chars", [200, 1500])
def test_round_trip_when_padded(condition, target_chars):
  for graph in CORPUS[:10]:
    text, _ = check_round_trip(graph, condition, target_chars=target_chars)
    if condition != "none" and graph.number_of_nodes():
      assert len(text) >= target_chars


def test_corpus_is_not_vacuous():
  """The round trip is only meaningful if the corpus contains the hard cases."""
  assert any(min(dict(g.degree).values()) == 0 for g in CORPUS), "no isolated node"
  assert any(nx.number_connected_components(g) > 1 for g in CORPUS), "all connected"
  assert any(
      sum(nx.triangles(g).values()) == 0 for g in CORPUS
  ), "no triangle-free graph"
  assert any(g.number_of_edges() == 0 for g in CORPUS) or any(
      g.number_of_edges() > 50 for g in CORPUS
  ), "no extreme density"


# --- what the parser recovers --------------------------------------------


def test_empty_primer_parses_to_empty():
  parsed = shortcuts.parse_primer("")
  assert parsed.is_empty()
  assert parsed.components is None
  assert parsed.nodes == ()


def test_none_condition_is_empty_for_every_graph():
  for graph in CORPUS[:5]:
    assert shortcuts.parse_primer(primers.build_primer(graph, "none")).is_empty()


def test_recovers_only_two_decimal_precision():
  """The parser must yield what the model sees, not the underlying float."""
  graph = graphqa.canonical(nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (2, 4)]))
  exact = primers.clustering(graph)[2]
  parsed = shortcuts.parse_primer(primers.build_primer(graph, "clustering"))
  assert exact != parsed.clustering[2], "test graph no longer has a rounded value"
  assert parsed.clustering[2] == float(primers._fmt(exact))


def test_node_count_is_recoverable_from_a_node_level_primer():
  # Rung 1 already yields n for every node-level condition, because the primer
  # emits one sentence per node including the isolated ones the encoding omits.
  graph = EDGE_CASES["isolated"]
  for condition in ("degree", "clustering", "rwse", "filler", "all"):
    parsed = shortcuts.parse_primer(primers.build_primer(graph, condition))
    assert len(parsed.nodes) == graph.number_of_nodes()


def test_components_primer_alone_yields_no_nodes():
  parsed = shortcuts.parse_primer(
      primers.build_primer(EDGE_CASES["two_triangles"], "components")
  )
  assert parsed.components == 2
  assert parsed.nodes == ()


def test_triangle_closed_forms():
  parsed = shortcuts.parse_primer(
      primers.build_primer(EDGE_CASES["triangle"], "all")
  )
  for node in (0, 1, 2):
    assert parsed.degree[node] == 2
    assert parsed.clustering[node] == 1.0
    assert parsed.rwse[node] == {2: 0.5, 3: 0.25}


def test_isolated_node_rwse_is_all_zero():
  parsed = shortcuts.parse_primer(
      primers.build_primer(EDGE_CASES["isolated"], "all")
  )
  assert parsed.degree[9] == 0
  assert parsed.rwse[9] == {2: 0.0, 3: 0.0}


# --- strictness -----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Node 0 is lonely.",
        "The graph has 3 connected components.",
        "Node 0 has degree 4",  # no terminating period
        "Node 0 has ",
        "Node 0 has degree.",
        "Node 0 has degree 4 plus clustering coefficient 0.50.",
        "Node 0 has clustering coefficient 0.5.",  # one decimal
        "Node 0 has return probability 0.50 after 2 steps and.",
        "Node 0 has degree 4. And node 1 has degree 2.",
        "This graph has three connected components.",
        "Node 0 has 4 other vertices in this graph.",
    ],
)
def test_rejects_malformed_text(text):
  with pytest.raises(ValueError):
    shortcuts.parse_primer(text)


@pytest.mark.parametrize(
    "text",
    [
        "This graph has 1 connected components.",
        "This graph has 2 connected component.",
        "Node 0 has 1 other nodes in this graph.",
        "Node 0 has 3 other node in this graph.",
        "Node 0 has return probability 0.00 after 1 steps.",
    ],
)
def test_rejects_wrong_plural(text):
  with pytest.raises(ValueError):
    shortcuts.parse_primer(text)


def test_accepts_correct_singulars():
  parsed = shortcuts.parse_primer("This graph has 1 connected component.")
  assert parsed.components == 1
  parsed = shortcuts.parse_primer("Node 0 has 1 other node in this graph.")
  assert parsed.filler == {0: 1}
  parsed = shortcuts.parse_primer("Node 0 has return probability 0.00 after 1 step.")
  assert parsed.rwse == {0: {1: 0.0}}


def test_rejects_conflicting_repeat():
  with pytest.raises(ValueError, match="conflicting"):
    shortcuts.parse_primer("Node 0 has degree 4. Node 0 has degree 5.")
  with pytest.raises(ValueError, match="conflicting"):
    shortcuts.parse_primer(
        "This graph has 2 connected components. This graph has 3 connected components."
    )


def test_accepts_consistent_repeat():
  """Padding legitimately repeats a node's filler sentence."""
  parsed = shortcuts.parse_primer(
      "Node 0 has 3 other nodes in this graph."
      " Node 0 has 3 other nodes in this graph."
  )
  assert parsed.filler == {0: 3}
  assert parsed.nodes == (0,)


def test_rejects_duplicate_step():
  with pytest.raises(ValueError, match="twice"):
    shortcuts.parse_primer(
        "Node 0 has return probability 0.50 after 2 steps and 0.25 after 2 steps."
    )


def test_rejects_bad_step_join():
  with pytest.raises(ValueError):
    shortcuts.parse_primer(
        "Node 0 has return probability 0.50 after 2 steps, 0.25 after 3 steps."
    )


@pytest.mark.parametrize(
    "text",
    [
        # Oxford comma missing at three phrases.
        "Node 0 has degree 4, clustering coefficient 0.50 and return probability"
        " 0.10 after 2 steps and 0.20 after 3 steps.",
        # Oxford comma present at two phrases.
        "Node 0 has degree 4, and 3 other nodes in this graph.",
        # Comma instead of a bare "and" at two phrases.
        "Node 0 has degree 4, 3 other nodes in this graph.",
        # Same three cases inside the RWSE step list.
        "Node 0 has return probability 0.10 after 1 step, 0.20 after 2 steps and"
        " 0.30 after 3 steps.",
        "Node 0 has return probability 0.10 after 2 steps, and 0.20 after 3 steps.",
    ],
)
def test_rejects_wrong_join(text):
  """The renderer emits exactly one join style; anything else is a drift signal.

  Without this the parser accepts strings `render_primer` never produces, and the
  round trip stops noticing a change to the join rule.
  """
  with pytest.raises(ValueError, match="do not match the rule"):
    shortcuts.parse_primer(text)
