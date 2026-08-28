"""Analytic tests for the primer statistics and renderer. No network.

RWSE is implemented three times here, with distinct jobs:

  * `primers.rwse`      -- the production code, repeated matrix multiplication.
  * `enumerate_weighted` -- every walk of length k, each weighted by the product
    of 1/degree at every step it takes. Exact, so it is asserted to 1e-12.
  * `simulate`          -- a walker stepped at random with a fixed seed.
    Approximate, so it is asserted within 0.03. This is the only one of the
    three that can catch a *conceptual* error, because it never expresses the
    weighting as code: the weighting emerges from the walker's choices.

The load-bearing graph for the cross-check is lopsided, not regular. Return
fraction and return probability coincide whenever every walk carries equal
weight, which is guaranteed on regular graphs -- on the triangle, K4 and the star
they agree at every node and every k, so none of those can serve as the check.

Regenerate the golden fixture with `.venv/bin/python tests/test_primers.py` after
an intentional wording change, and read the diff before committing it.
"""

import itertools
import json
import random
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from graphtalk import graphqa
from graphtalk import primers
from talk_like_a_graph import graph_generators
from talk_like_a_graph import graph_text_encoders

GOLDEN_PATH = Path(__file__).parent / "golden" / "primers.json"

# Simulation tolerance. Worst observed disagreement at 5000 walks is 0.013, so
# this has a fivefold margin, and the fixed seed makes the test non-flaky.
SIMULATION_TOLERANCE = 0.03
SIMULATION_WALKS = 5000

LOPSIDED_EDGES = [(0, 1), (0, 2), (2, 3), (2, 4)]
# Node 4 at k=4 has the exact value 53/200, a non-dyadic two-decimal tie: the
# case where `M @ P` and `P @ M` accumulation disagree in the last bit. Node 6
# is isolated, so this also exercises the degree-0 clamp.
TIE_EDGES = [
    (0, 2), (0, 3), (0, 5), (1, 4), (1, 7),
    (2, 3), (2, 5), (3, 4), (3, 5), (3, 7),
]


def _graph(edges, nodes=()) -> nx.Graph:
  graph = nx.Graph()
  graph.add_nodes_from(nodes)
  graph.add_edges_from(edges)
  return graphqa.canonical(graph)


TRIANGLE = _graph([(0, 1), (1, 2), (0, 2)])
K4 = graphqa.canonical(nx.complete_graph(4))
STAR = graphqa.canonical(nx.star_graph(4))
PATH = graphqa.canonical(nx.path_graph(5))
LOPSIDED = _graph(LOPSIDED_EDGES)
TIE_GRAPH = _graph(TIE_EDGES, nodes=range(8))
WITH_ISOLATED = _graph([(0, 1), (1, 2), (0, 2)], nodes=[0, 1, 2, 5])
TWO_TRIANGLES = _graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
# A star centred on a high-numbered node: every node's values differ from its
# neighbours', so a node-mapping error cannot hide behind symmetry.
HIGH_STAR_EDGES = [(9, 0), (9, 1), (9, 2), (9, 3)]
HIGH_STAR = _graph(HIGH_STAR_EDGES)
EDGELESS = _graph([], nodes=range(4))

GRAPHS = {
    "triangle": TRIANGLE,
    "k4": K4,
    "star": STAR,
    "path": PATH,
    "lopsided": LOPSIDED,
    "tie": TIE_GRAPH,
    "with_isolated": WITH_ISOLATED,
    "two_triangles": TWO_TRIANGLES,
    "high_star": HIGH_STAR,
    "edgeless": EDGELESS,
}

NODE_LEVEL_CONDITIONS = ("degree", "clustering", "rwse", "filler", "all")


# --- reference implementations of RWSE ---------------------------------------


def enumerate_weighted(graph: nx.Graph, node: int, k: int) -> float:
  """Sums the probability of every length-k walk from `node` back to `node`.

  Each walk is weighted by the product of 1/degree at every step it takes, which
  is what makes this the return *probability* and not the return *fraction*.
  A walker on a degree-0 node contributes no walk, which is the same convention
  the production clamp produces.
  """
  total = 0.0
  stack = [(node, 1.0, 0)]
  while stack:
    current, weight, steps = stack.pop()
    if steps == k:
      total += weight if current == node else 0.0
      continue
    degree = graph.degree(current)
    if degree == 0:
      continue
    for neighbor in graph.neighbors(current):
      stack.append((neighbor, weight / degree, steps + 1))
  return total


def return_fraction(graph: nx.Graph, node: int, k: int) -> float:
  """The share of length-k walks that return, counting every walk equally.

  This is the quantity RWSE is *not*. It is here so that nobody replaces the
  weighted version with it: they differ on any graph whose walks carry unequal
  weight, which is most of the corpus.
  """
  walks, returns = 0, 0
  stack = [(node, 0)]
  while stack:
    current, steps = stack.pop()
    if steps == k:
      walks += 1
      returns += current == node
      continue
    for neighbor in graph.neighbors(current):
      stack.append((neighbor, steps + 1))
  return returns / walks if walks else 0.0


def simulate(
    graph: nx.Graph, node: int, k: int, walks: int = SIMULATION_WALKS, seed: int = 0
) -> float:
  """Steps a walker at random and counts returns. Never weights anything.

  A walker starting on an isolated node cannot move, so it is trivially still at
  its starting node and this returns 1.0 -- where the production code returns
  0.0 by convention. See test_isolated_node_conventions.
  """
  neighbors = {n: sorted(graph.neighbors(n)) for n in graph.nodes()}
  if not neighbors[node]:
    return 1.0
  rng = random.Random(seed)
  returns = 0
  for _ in range(walks):
    current = node
    for _ in range(k):
      current = rng.choice(neighbors[current])
    returns += current == node
  return returns / walks


# --- RWSE: three implementations ---------------------------------------------


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_matrix_rwse_matches_weighted_enumeration(name):
  graph = GRAPHS[name]
  for k in (1, 2, 3, 4):
    table = primers.rwse(graph, k_min=k, k_max=k)
    for node in graph.nodes():
      assert table[node][k] == pytest.approx(
          enumerate_weighted(graph, node, k), abs=1e-12
      ), f"{name}: node {node}, k={k}"


@pytest.mark.parametrize("name", ["lopsided", "path", "tie", "high_star", "k4"])
def test_matrix_rwse_matches_random_walk_simulation(name):
  graph = GRAPHS[name]
  table = primers.rwse(graph, k_min=2, k_max=3)
  for node in graph.nodes():
    if graph.degree(node) == 0:
      continue  # the convention, not the definition; asserted separately
    for k in (2, 3):
      assert table[node][k] == pytest.approx(
          simulate(graph, node, k), abs=SIMULATION_TOLERANCE
      ), f"{name}: node {node}, k={k}"


def test_return_probability_is_not_return_fraction():
  # The lopsided graph is the one that can tell the two quantities apart.
  assert primers.rwse(LOPSIDED, 2, 2)[0][2] == pytest.approx(2 / 3)
  assert return_fraction(LOPSIDED, 0, 2) == pytest.approx(0.5)
  # The path disagrees too, at interior nodes and even k only.
  assert primers.rwse(PATH, 2, 2)[1][2] == pytest.approx(0.75)
  assert return_fraction(PATH, 1, 2) == pytest.approx(2 / 3)


@pytest.mark.parametrize("name", ["triangle", "k4", "star"])
def test_regular_and_star_graphs_cannot_tell_the_two_apart(name):
  # Recorded so nobody "simplifies" the cross-check onto one of these.
  graph = GRAPHS[name]
  for node in graph.nodes():
    for k in (2, 3):
      assert primers.rwse(graph, k, k)[node][k] == pytest.approx(
          return_fraction(graph, node, k)
      )


# --- ordering and node mapping ------------------------------------------------


def test_primer_text_is_insertion_order_invariant():
  rng = random.Random(11)
  texts = set()
  for _ in range(25):
    nodes = list(range(5))
    edges = [tuple(edge) for edge in LOPSIDED_EDGES]
    rng.shuffle(nodes)
    rng.shuffle(edges)
    edges = [edge if rng.random() < 0.5 else edge[::-1] for edge in edges]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    texts.add(primers.build_primer(graphqa.canonical(graph), "all"))
    # Also without canonicalising: primers.py passes an explicit nodelist so it
    # is correct for any graph it is handed, and this is what pins that. A
    # missing nodelist, or a diagonal paired against the wrong list, only shows
    # up on a graph whose insertion order is not sorted order.
    texts.add(primers.build_primer(graph, "all"))
  assert len(texts) == 1
  assert texts == {primers.build_primer(LOPSIDED, "all")}


def test_canonical_collapses_encoder_orderings():
  raw, canonical = set(), set()
  for permutation in itertools.permutations(LOPSIDED_EDGES):
    graph = nx.Graph()
    graph.add_edges_from(permutation)
    raw.add(graph_text_encoders.encode_graph(graph, "incident"))
    canonical.add(
        graph_text_encoders.encode_graph(
            graphqa.canonical(graph), "incident"
        )
    )
  assert len(raw) > 1  # the encoder follows insertion order
  assert len(canonical) == 1


def test_each_node_receives_its_own_values():
  # Star centred on node 9: the centre returns with probability 1 after two
  # steps, a leaf with probability 1/4. A node-mapping error swaps them.
  assert primers.degrees(HIGH_STAR) == {n: 1 for n in range(4)} | {9: 4}
  table = primers.rwse(HIGH_STAR, 2, 3)
  assert table[9][2] == pytest.approx(1.0)
  assert all(table[leaf][2] == pytest.approx(0.25) for leaf in range(4))
  assert primers.build_primer(HIGH_STAR, "all").endswith(
      "Node 9 has degree 4, clustering coefficient 0.00, and return probability"
      " 1.00 after 2 steps and 0.00 after 3 steps."
  )
  assert primers.build_primer(HIGH_STAR, "all").startswith(
      "Node 0 has degree 1, clustering coefficient 0.00, and return probability"
      " 0.25 after 2 steps and 0.00 after 3 steps."
  )
  # The same star as networkx builds it from the edge list, so node 9 is
  # inserted first and insertion order is not sorted order. Any graph handed
  # straight to primers.py has to come out the same way.
  raw = nx.Graph(HIGH_STAR_EDGES)
  assert list(raw.nodes()) != sorted(raw.nodes())
  assert primers.rwse(raw, 2, 3) == table
  assert primers.build_primer(raw, "all") == primers.build_primer(
      HIGH_STAR, "all"
  )


# --- rendering stability ------------------------------------------------------


def _accumulate(graph: nx.Graph, k: int, reverse: bool) -> np.ndarray:
  nodes = sorted(graph.nodes())
  adjacency = nx.to_numpy_array(graph, nodelist=nodes)
  degree = adjacency.sum(axis=1)
  transition = adjacency / np.where(degree > 0.0, degree, 1.0)[:, np.newaxis]
  walk = np.eye(len(nodes))
  for _ in range(k):
    walk = transition @ walk if reverse else walk @ transition
  return np.diagonal(walk)


def test_multiply_order_renders_identically():
  # Non-dyadic tie: node 4 at k=4 is exactly 53/200. On this machine the raw
  # two-decimal format of the two accumulation orders differs (0.26 vs 0.27);
  # _fmt is what removes that.
  nodes = sorted(TIE_GRAPH.nodes())
  for k in (2, 3, 4):
    forward = _accumulate(TIE_GRAPH, k, reverse=False)
    backward = _accumulate(TIE_GRAPH, k, reverse=True)
    for index, node in enumerate(nodes):
      assert primers._fmt(forward[index]) == primers._fmt(backward[index]), (
          f"node {node}, k={k}"
      )
  index4 = nodes.index(4)
  assert _accumulate(TIE_GRAPH, 4, reverse=False)[index4] == pytest.approx(
      53 / 200, abs=1e-12
  )
  assert primers._fmt(_accumulate(TIE_GRAPH, 4, reverse=False)[index4]) == "0.27"
  assert primers._fmt(_accumulate(TIE_GRAPH, 4, reverse=True)[index4]) == "0.27"


def test_fmt_rounds_exact_halves_to_even():
  # Deterministic on every machine, and not a bug. Recorded so nobody chases it.
  assert primers._fmt(0.125) == "0.12"
  assert primers._fmt(0.135) == "0.14"


# --- golden fixture -----------------------------------------------------------


def _golden_cases() -> list[dict]:
  return json.loads(GOLDEN_PATH.read_text())


def test_golden_primers_match():
  for case in _golden_cases():
    graph = _graph(
        [tuple(edge) for edge in case["edges"]], nodes=case["nodes"]
    )
    for condition, expected in case["primers"].items():
      assert primers.build_primer(graph, condition) == expected, (
          f"{case['name']} / {condition}"
      )


def test_golden_fixture_covers_every_condition():
  for case in _golden_cases():
    assert sorted(case["primers"]) == sorted(primers.CONDITIONS)


# --- closed forms -------------------------------------------------------------


def test_triangle_closed_form():
  assert primers.degrees(TRIANGLE) == {0: 2, 1: 2, 2: 2}
  assert primers.clustering(TRIANGLE) == {0: 1.0, 1: 1.0, 2: 1.0}
  table = primers.rwse(TRIANGLE, 2, 3)
  for node in TRIANGLE.nodes():
    assert table[node][2] == pytest.approx(0.5)
    assert table[node][3] == pytest.approx(0.25)


def test_k4_closed_form():
  assert primers.degrees(K4) == {node: 3 for node in range(4)}
  assert primers.clustering(K4) == {node: 1.0 for node in range(4)}


@pytest.mark.parametrize("name", ["star", "path", "lopsided"])
def test_bipartite_graphs_have_zero_odd_return(name):
  graph = GRAPHS[name]
  assert set(primers.clustering(graph).values()) == {0.0}
  table = primers.rwse(graph, 3, 3)
  for node in graph.nodes():
    assert table[node][3] == 0.0  # exactly, not approximately


# --- conventions --------------------------------------------------------------


def test_isolated_node_conventions():
  assert primers.degrees(WITH_ISOLATED)[5] == 0
  assert primers.clustering(WITH_ISOLATED)[5] == 0.0
  table = primers.rwse(WITH_ISOLATED, 1, 4)
  assert set(table[5].values()) == {0.0}
  # This is a convention, not the definition. A walker on an isolated node
  # cannot move, so it is trivially still at its starting node and the
  # definition-as-simulated gives 1.0. 0.0 is chosen because it reads as "no
  # walk structure here".
  assert simulate(WITH_ISOLATED, 5, 2) == 1.0


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_k1_is_zero_for_every_node(name):
  graph = GRAPHS[name]
  table = primers.rwse(graph, k_min=1, k_max=1)
  assert set(table[node][1] for node in graph.nodes()) == {0.0}


def test_rwse_rejects_a_k_below_one():
  with pytest.raises(ValueError):
    primers.rwse(TRIANGLE, k_min=0, k_max=3)
  with pytest.raises(ValueError):
    primers.rwse(TRIANGLE, k_min=3, k_max=2)


# --- components and the circuit rank ------------------------------------------


def test_component_count():
  assert primers.component_count(TRIANGLE) == 1
  assert primers.component_count(TWO_TRIANGLES) == 2
  assert primers.component_count(WITH_ISOLATED) == 2  # the lone node counts
  assert primers.component_count(EDGELESS) == 4


def _circuit_rank_graphs() -> list[nx.Graph]:
  graphs = [
      PATH,                                        # tree
      STAR,                                        # tree
      LOPSIDED,                                    # tree
      _graph([(0, 1), (2, 3)]),                    # forest
      _graph([(0, 1), (2, 3)], nodes=[0, 1, 2, 3, 4]),  # forest + isolate
      graphqa.canonical(nx.cycle_graph(5)),        # cycle
      _graph([(0, 1), (1, 2), (0, 2)], nodes=range(6)),  # cycle + isolates
      TRIANGLE,
      TWO_TRIANGLES,                               # disjoint union
      K4,
      TIE_GRAPH,
      EDGELESS,
  ]
  graphs += [
      graphqa.canonical(g)
      for g in graph_generators.generate_graphs(50, "er", False, random_seed=1234)
  ]
  return graphs


@pytest.mark.parametrize("graph", _circuit_rank_graphs())
def test_circuit_rank_identity(graph):
  try:
    nx.find_cycle(graph)
    has_cycle = True
  except nx.NetworkXNoCycle:
    has_cycle = False
  circuit_rank = (
      graph.number_of_edges()
      - graph.number_of_nodes()
      + primers.component_count(graph)
  )
  assert (circuit_rank > 0) == has_cycle


# --- renderer -----------------------------------------------------------------


def _sentences(text: str) -> list[str]:
  """Splits a primer back into sentences.

  Safe against the decimals in the text: they are never followed by a space.
  """
  if not text:
    return []
  return [piece + "." for piece in text.rstrip(".").split(". ")]


def test_none_condition_is_empty():
  assert primers.build_primer(TRIANGLE, "none") == ""
  assert primers.render_primer(TRIANGLE, ()) == ""


def test_there_are_seven_conditions():
  assert len(primers.CONDITIONS) == 7


@pytest.mark.parametrize("condition", NODE_LEVEL_CONDITIONS)
def test_one_sentence_per_node_in_sorted_order(condition):
  graph = TIE_GRAPH
  sentences = _sentences(primers.build_primer(graph, condition))
  assert len(sentences) == graph.number_of_nodes()
  # filler is the deliberate exception to the shared "has" verb -- see
  # _filler_phrase for why.
  verb = "is" if condition == "filler" else "has"
  for node, sentence in zip(sorted(graph.nodes()), sentences):
    assert sentence.startswith(f"Node {node} {verb} ")


@pytest.mark.parametrize(
    "condition", [c for c in NODE_LEVEL_CONDITIONS if c != "filler"]
)
def test_node_level_conditions_share_one_sentence_shape(condition):
  # One sentence, `Node N has ...`, whose only interior periods are decimals.
  # filler is deliberately exempt -- see test_filler_sentence_shape below.
  shape = re.compile(r"^Node \d+ has (?:[^.]|\.\d)+\.$")
  for sentence in _sentences(primers.build_primer(TIE_GRAPH, condition)):
    assert shape.match(sentence), sentence


def test_filler_sentence_shape_is_the_deliberate_exception():
  # Deliberately exempt from the shared "has" shape above -- see
  # _filler_phrase for why.
  shape = re.compile(r"^Node \d+ is simply present within the graph G\.$")
  for sentence in _sentences(primers.build_primer(TIE_GRAPH, "filler")):
    assert shape.match(sentence), sentence


def test_components_sentence_shape_is_graph_level():
  # Deliberately exempt from the node-level shape above.
  shape = re.compile(r"^This graph has \d+ connected components?\.$")
  assert shape.match(primers.build_primer(TWO_TRIANGLES, "components"))


def test_graph_level_sentences_come_first():
  text = primers.render_primer(TWO_TRIANGLES, ("degree", "components"))
  assert text.startswith("This graph has 2 connected components. Node 0 has")


def test_two_phrases_join_with_and_and_no_comma():
  text = primers.render_primer(TRIANGLE, ("degree", "clustering"))
  assert text.startswith("Node 0 has degree 2 and clustering coefficient 1.00.")


def test_three_phrases_take_the_oxford_comma():
  text = primers.build_primer(TRIANGLE, "all")
  assert text.startswith(
      "Node 0 has degree 2, clustering coefficient 1.00, and return probability"
      " 0.50 after 2 steps and 0.25 after 3 steps."
  )


def test_rwse_phrase_labels_its_step_counts():
  text = primers.build_primer(TRIANGLE, "rwse")
  assert text.startswith(
      "Node 0 has return probability 0.50 after 2 steps and 0.25 after 3 steps."
  )
  single = primers.render_primer(TRIANGLE, ("rwse",), k_min=1, k_max=1)
  assert single.startswith("Node 0 has return probability 0.00 after 1 step.")


@pytest.mark.parametrize("condition", sorted(primers.CONDITIONS))
def test_every_float_is_two_decimals(condition):
  text = primers.build_primer(TIE_GRAPH, condition)
  for number in re.findall(r"\d+\.\d+", text):
    assert re.fullmatch(r"\d+\.\d\d", number), number


def test_singular_forms():
  # filler has no singular/plural form to test: "is in G" carries no count.
  pair = _graph([(0, 1)])
  assert primers.build_primer(pair, "components") == (
      "This graph has 1 connected component."
  )
  assert primers.build_primer(TWO_TRIANGLES, "components") == (
      "This graph has 2 connected components."
  )


def test_part_order_is_fixed_and_unknown_parts_are_rejected():
  assert primers.render_primer(
      TRIANGLE, ("rwse", "degree")
  ) == primers.render_primer(TRIANGLE, ("degree", "rwse"))
  with pytest.raises(ValueError):
    primers.render_primer(TRIANGLE, ("components", "eigenvector"))
  with pytest.raises(ValueError):
    primers.build_primer(TRIANGLE, "degrees")


# --- length control -----------------------------------------------------------


def test_filler_states_nothing_structural():
  text = primers.build_primer(TIE_GRAPH, "filler")
  for word in ("degree", "clustering", "return probability", "component"):
    assert word not in text
  # Identical for any two graphs with the same node count.
  three_edges = _graph([(0, 1), (1, 2), (0, 2)])
  one_edge = _graph([(0, 1)], nodes=[0, 1, 2])
  assert primers.build_primer(three_edges, "filler") == primers.build_primer(
      one_edge, "filler"
  )
  # The only numerals it introduces are the node ids themselves -- no numeral
  # states a relationship, unlike the old "N other nodes" wording.
  numerals = {int(tok) for tok in re.findall(r"\d+", text)}
  assert numerals <= set(TIE_GRAPH.nodes())


def test_filler_mentions_isolated_nodes_without_describing_them():
  # The salience control: `filler` names node 5 without saying it is isolated.
  text = primers.build_primer(WITH_ISOLATED, "filler")
  assert "Node 5 is simply present within the graph G." in text


def test_target_chars_pads_with_inert_filler():
  padded = primers.build_primer(TIE_GRAPH, "filler", target_chars=1500)
  assert len(padded) >= 1500
  shape = re.compile(r"^Node \d+ is simply present within the graph G\.$")
  for sentence in _sentences(padded):
    assert shape.match(sentence), sentence


def test_target_chars_never_truncates_and_never_pads_none():
  unpadded = primers.build_primer(TIE_GRAPH, "all")
  assert primers.build_primer(TIE_GRAPH, "all", target_chars=10) == unpadded
  assert primers.build_primer(TIE_GRAPH, "none", target_chars=500) == ""


# --- diagnostic ---------------------------------------------------------------


def test_correlation_is_none_when_either_vector_is_constant():
  # Triangle: degree is constant. Path: k=3 is constant zero (triangle-free).
  assert primers.rwse_degree_correlation(TRIANGLE) == {2: None, 3: None}
  assert primers.rwse_degree_correlation(PATH)[3] is None


def test_correlation_is_defined_and_bounded_on_an_asymmetric_graph():
  values = primers.rwse_degree_correlation(TIE_GRAPH)
  assert set(values) == {2, 3}
  for k, value in values.items():
    assert value is not None and -1.0 <= value <= 1.0, k


def _write_golden() -> None:
  """Regenerates tests/golden/primers.json. Read the diff before committing."""
  cases = []
  for name in ("pair", "lopsided", "with_isolated", "two_triangles", "tie"):
    graph = {
        "pair": _graph([(0, 1)]),
        "lopsided": LOPSIDED,
        "with_isolated": WITH_ISOLATED,
        "two_triangles": TWO_TRIANGLES,
        "tie": TIE_GRAPH,
    }[name]
    cases.append({
        "name": name,
        "nodes": sorted(graph.nodes()),
        "edges": [list(edge) for edge in sorted(graph.edges())],
        "primers": {
            condition: primers.build_primer(graph, condition)
            for condition in primers.CONDITIONS
        },
    })
  GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
  GOLDEN_PATH.write_text(json.dumps(cases, indent=2) + "\n")
  print(f"wrote {GOLDEN_PATH}")


if __name__ == "__main__":
  _write_golden()
