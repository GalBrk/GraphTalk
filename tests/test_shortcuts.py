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

import dataclasses
import inspect
import random

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


# --- theorem rules --------------------------------------------------------
#
# The load-bearing assertion in this section is that every theorem has precision
# exactly 1.0. A theorem that is ever wrong is not a theorem, and a bar built on
# it would be wrong in the direction that makes a model look like it failed to
# reach something that was never there.

FIT_SEED = 999
SWEEP_SIZE = 120

SWEEP = [
    graphqa.canonical(g)
    for g in graph_generators.generate_graphs(
        SWEEP_SIZE, "er", False, random_seed=CORPUS_SEED
    )
]

def adversarial_graphs() -> list[nx.Graph]:
  """Structures the ER generator effectively never produces.

  Precision 1.0 is only as strong as the corpus it is asserted over, and the ER
  sweep turned out to contain no tree at all: every forest in it sits at
  m <= n-2, and both graphs at m = n-1 have a cycle. So a false rule keyed on
  that boundary -- "m >= n-1 implies a cycle", wrong on exactly the trees --
  scored a clean 1.0 over 120 ER graphs. This corpus sits on those boundaries
  deliberately.
  """
  out = [
      nx.complete_bipartite_graph(3, 3),  # cycles but no triangle
      nx.complete_bipartite_graph(2, 4),
      nx.empty_graph(5),
      nx.disjoint_union(nx.path_graph(4), nx.path_graph(3)),  # forest, c = 2
      nx.disjoint_union(nx.star_graph(3), nx.path_graph(5)),
      nx.disjoint_union(nx.cycle_graph(4), nx.path_graph(3)),  # cycle plus tree
  ]
  for size in range(2, 10):
    out.append(nx.path_graph(size))  # tree: m = n-1, no cycle
    out.append(nx.star_graph(size))  # tree
    out.append(nx.complete_graph(size))
    if size >= 3:
      out.append(nx.cycle_graph(size))  # unicyclic: m = n
  for seed in range(6):
    out.append(nx.random_labeled_tree(8, seed=seed))

  # Trees and cycles with isolated nodes bolted on, which moves c without moving
  # m and so separates circuit rank from the edge-count rules.
  for base in (nx.path_graph(5), nx.cycle_graph(5), nx.complete_graph(4)):
    graph = base.copy()
    graph.add_nodes_from([90, 91])
    out.append(graph)
  return [graphqa.canonical(g) for g in out]


ADVERSARIAL = adversarial_graphs()
CORPORA = {"er": SWEEP, "adversarial": ADVERSARIAL}

_PARSED: dict[tuple[str, int, str], shortcuts.ParsedPrimer] = {}


def parsed(corpus: str, index: int, condition: str) -> shortcuts.ParsedPrimer:
  """Renders and parses once per (graph, condition); the sweep reuses it."""
  key = (corpus, index, condition)
  if key not in _PARSED:
    _PARSED[key] = shortcuts.parse_primer(
        primers.build_primer(CORPORA[corpus][index], condition)
    )
  return _PARSED[key]


def rows_for(corpus: str, condition: str, task: str, rung: int, seed: int = 5):
  rng = random.Random(seed)
  rows = []
  for index, graph in enumerate(CORPORA[corpus]):
    if graph.number_of_nodes() < shortcuts.QUERY_ARITY[task]:
      continue
    targets = shortcuts.sample_query(graph, task, rng)
    rows.append((
        shortcuts.Context(
            primer=parsed(corpus, index, condition),
            targets=targets,
            n=graph.number_of_nodes() if rung >= 2 else None,
            m=graph.number_of_edges() if rung >= 3 else None,
        ),
        graphqa.gold_answer(graph, task, targets),
    ))
  return rows


def sweep_rows(condition: str, task: str, rung: int, seed: int = 5):
  return rows_for("er", condition, task, rung, seed)


@pytest.mark.parametrize("corpus", sorted(CORPORA))
@pytest.mark.parametrize("rule", shortcuts.THEOREMS, ids=lambda r: r.name)
def test_theorem_precision_is_exactly_one(rule, corpus):
  for condition in sorted(primers.CONDITIONS):
    for rung in shortcuts.RUNGS:
      result = shortcuts.score_theorem(
          rule, rows_for(corpus, condition, rule.task, rung)
      )
      assert result["precision"] in (None, 1.0), (corpus, condition, rung, result)


def test_adversarial_corpus_covers_the_boundaries():
  """Without these structures the precision assertion has little power."""
  trees = [g for g in ADVERSARIAL if nx.is_tree(g)]
  assert trees, "no tree: the m = n-1 boundary is untested"
  assert any(
      g.number_of_edges() == g.number_of_nodes() - 1 and nx.is_forest(g)
      for g in ADVERSARIAL
  ), "no graph at m = n-1 without a cycle"
  assert any(
      g.number_of_edges() == g.number_of_nodes() for g in ADVERSARIAL
  ), "no unicyclic graph at m = n"
  assert any(
      sum(nx.triangles(g).values()) == 0 and not nx.is_forest(g)
      for g in ADVERSARIAL
  ), "no triangle-free graph that still has a cycle"
  assert any(
      nx.number_connected_components(g) > 1 for g in ADVERSARIAL
  ), "no disconnected graph"


def test_every_theorem_fires_somewhere():
  """A rule that never fires anywhere is dead code, not a zero-coverage result."""
  for rule in shortcuts.THEOREMS:
    fired = max(
        shortcuts.score_theorem(rule, sweep_rows(condition, rule.task, rung))["fired"]
        for condition in primers.CONDITIONS
        for rung in shortcuts.RUNGS
    )
    assert fired > 0, f"{rule.name} never fires on any condition or rung"


def test_no_theorem_fires_on_the_none_arm_at_rung_one():
  """The plan's `none` sanity check, which holds only at rung 1."""
  for rule in shortcuts.THEOREMS:
    result = shortcuts.score_theorem(rule, sweep_rows("none", rule.task, 1))
    assert result["fired"] == 0, (rule.name, result)


def test_the_none_arm_is_not_inert_above_rung_one():
  """Above rung 1 the `none` arm is an encoding-only bar, not the baseline.

  Granting n and m is independent of the primer, so `m >= n` fires on the empty
  primer at rung 3. That is not a leak -- it is what n and m alone are worth, and
  it means the plan's "`none` must equal the majority baseline" check has to be
  read as a rung-1 statement.
  """
  rule = next(r for r in shortcuts.THEOREMS if r.name == "edges_at_least_nodes")
  assert shortcuts.score_theorem(rule, sweep_rows("none", "cycle_check", 1))["fired"] == 0
  assert shortcuts.score_theorem(rule, sweep_rows("none", "cycle_check", 3))["fired"] > 0


def test_coverage_is_monotone_in_rung():
  """Each rung grants strictly more, so a rule cannot fire less often."""
  for rule in shortcuts.THEOREMS:
    for condition in sorted(primers.CONDITIONS):
      coverage = [
          shortcuts.score_theorem(rule, sweep_rows(condition, rule.task, rung))[
              "coverage"
          ]
          for rung in shortcuts.RUNGS
      ]
      assert coverage[0] <= coverage[1] <= coverage[2], (rule.name, condition, coverage)


def test_solver_cannot_read_the_graph():
  """Structural, so the guarantee survives refactoring rather than review."""
  names = {field.name for field in dataclasses.fields(shortcuts.Context)}
  assert names == {"primer", "targets", "n", "m"}, names
  for rule in shortcuts.THEOREMS:
    params = list(inspect.signature(rule.solve).parameters)
    assert params == ["ctx"], (rule.name, params)


# --- individual theorems, on graphs whose answers are known by hand -------


def theorem(name: str) -> shortcuts.Rule:
  return next(r for r in shortcuts.THEOREMS if r.name == name)


def test_circuit_rank_answers_in_both_directions():
  path = graphqa.canonical(nx.Graph([(0, 1), (1, 2), (2, 3)]))
  triangle = EDGE_CASES["triangle"]
  rule = theorem("circuit_rank")
  assert rule.solve(shortcuts.make_context(path, "components", (), 3)) == (
      "No, there is no cycle"
  )
  assert rule.solve(shortcuts.make_context(triangle, "components", (), 3)) == (
      "Yes, there is a cycle"
  )
  # It is the only cycle_check theorem that needs m, so rung 1 cannot run it.
  assert rule.solve(shortcuts.make_context(path, "components", (), 1)) is None


def test_triangle_theorems_agree_and_abstain_on_forests():
  path = graphqa.canonical(nx.Graph([(0, 1), (1, 2), (2, 3)]))
  triangle = EDGE_CASES["triangle"]
  for name, condition in (("clustering_triangle", "clustering"),
                          ("rwse_triangle", "rwse")):
    rule = theorem(name)
    assert rule.solve(shortcuts.make_context(triangle, condition, (), 1)) == (
        "Yes, there is a cycle"
    )
    # One-directional: no triangle means no answer, not "no cycle".
    assert rule.solve(shortcuts.make_context(path, condition, (), 1)) is None


def test_degree_theorems_on_edge_existence():
  star = graphqa.canonical(nx.Graph([(0, 1), (0, 2), (0, 3)]))
  star.add_node(4)
  zero = theorem("degree_zero_no_edge")
  full = theorem("degree_full_edge")
  # Node 4 is isolated, so no edge touches it.
  assert zero.solve(shortcuts.make_context(star, "degree", (1, 4), 1)) == "No"
  assert zero.solve(shortcuts.make_context(star, "degree", (1, 2), 1)) is None
  # Node 0 has degree 3 out of 5 nodes, so it is not adjacent to everything.
  assert full.solve(shortcuts.make_context(star, "degree", (0, 4), 1)) is None
  clique = graphqa.canonical(nx.complete_graph(4))
  assert full.solve(shortcuts.make_context(clique, "degree", (0, 1), 1)) == "Yes"


def test_rwse_zero_needs_k_two_in_range():
  """Odd k alone is 0 for every node of a triangle-free graph, 19% of them."""
  path = graphqa.canonical(nx.Graph([(0, 1), (1, 2), (2, 3)]))
  rule = theorem("rwse_zero_no_nodes")
  assert rule.solve(
      shortcuts.make_context(path, "rwse", (1,), 1, k_min=3, k_max=3)
  ) is None
  assert rule.solve(shortcuts.make_context(path, "rwse", (1,), 1)) is None
  isolated = EDGE_CASES["isolated"]
  assert rule.solve(shortcuts.make_context(isolated, "rwse", (9,), 1)) == "No nodes"


def test_degree_sum_recovers_edge_count_without_a_grant():
  for graph in SWEEP[:20]:
    context = shortcuts.make_context(graph, "degree", (), 1)
    assert context.m_from_degrees == graph.number_of_edges()
    assert theorem("degree_sum").solve(context) == str(graph.number_of_edges())


# --- baselines ------------------------------------------------------------


def test_majority_answer_is_deterministic_under_ties():
  assert shortcuts.majority_answer(["b", "a"]) == "b"
  assert shortcuts.majority_answer(["a", "b"]) == "b"
  assert shortcuts.majority_answer(["a", "a", "b"]) == "a"


def test_baseline_is_fitted_and_scored_on_disjoint_seeds():
  """The split has to be structural; this asserts the two corpora differ."""
  fit = [
      graphqa.canonical(g)
      for g in graph_generators.generate_graphs(
          SWEEP_SIZE, "er", False, random_seed=FIT_SEED
      )
  ]
  assert FIT_SEED != CORPUS_SEED

  def signature(g):
    return (tuple(sorted(g.nodes())), tuple(sorted(map(tuple, map(sorted, g.edges())))))

  overlap = set(map(signature, fit)) & set(map(signature, SWEEP))
  # Tiny edgeless graphs can coincide across seeds; anything more would mean the
  # seeds are not actually independent.
  assert len(overlap) <= 2, len(overlap)
  assert all(len(edges) == 0 for _, edges in overlap), overlap


def test_cycle_check_baseline_is_the_yes_rate():
  fit = [
      graphqa.canonical(g)
      for g in graph_generators.generate_graphs(
          SWEEP_SIZE, "er", False, random_seed=FIT_SEED
      )
  ]
  fit_golds = [graphqa.gold_answer(g, "cycle_check", ()) for g in fit]
  test_golds = [graphqa.gold_answer(g, "cycle_check", ()) for g in SWEEP]
  answer, accuracy = shortcuts.baseline_accuracy(fit_golds, test_golds)
  assert answer == "Yes, there is a cycle"
  # The plan records 83.2% on 500 graphs; a 120-graph sample sits near it.
  assert 0.70 <= accuracy <= 0.95, accuracy


# --- fitted rules, the solver, and the exact island -----------------------

SPLIT = shortcuts.Split(fit_seed=FIT_SEED, test_seed=CORPUS_SEED, size=SWEEP_SIZE)
FIT_GRAPHS = SPLIT.fit_graphs()


def test_split_refuses_to_fit_and_score_on_the_same_seed():
  """Structural, because this is the mistake that reverses the finding's sign."""
  with pytest.raises(ValueError, match="different seeds"):
    shortcuts.Split(fit_seed=7, test_seed=7)
  assert SPLIT.fit_seed != SPLIT.test_seed


def test_split_corpora_actually_differ():
  def signature(graph):
    return (
        graph.number_of_nodes(),
        tuple(sorted(tuple(sorted(e)) for e in graph.edges())),
    )

  overlap = set(map(signature, FIT_GRAPHS)) & set(map(signature, SWEEP))
  # Small edgeless graphs can coincide across seeds; anything beyond that would
  # mean the two seeds are not independent.
  assert all(edges == () for _, edges in overlap), overlap


def test_solve_has_no_graph_parameter():
  params = list(inspect.signature(shortcuts.solve).parameters)
  assert "graph" not in params
  assert params[:2] == ["primer_text", "task"]


def test_solve_reads_text_not_statistics():
  """The public entry point takes the rendered string, so it sees what the model sees."""
  graph = EDGE_CASES["triangle"]
  text = primers.build_primer(graph, "components")
  assert shortcuts.solve(text, "cycle_check", (), n=3, m=3) == "Yes, there is a cycle"
  path = graphqa.canonical(nx.Graph([(0, 1), (1, 2)]))
  assert shortcuts.solve(
      primers.build_primer(path, "components"), "cycle_check", (), n=3, m=2
  ) == "No, there is no cycle"


def test_solver_always_answers():
  """A cell needs a score, so the solver falls back rather than abstaining."""
  for task in shortcuts.TASKS:
    fit_rows = shortcuts.build_rows(FIT_GRAPHS, "none", task, 1, seed=5)
    fallback = shortcuts.majority_answer([gold for _, gold in fit_rows])
    for context, _ in shortcuts.build_rows(SWEEP, "none", task, 1, seed=6)[:20]:
      assert shortcuts.solve_context(context, task, (), fallback) != ""


@pytest.mark.parametrize("task", shortcuts.TASKS)
def test_shortcut_never_falls_below_the_baseline(task):
  """The solver may not be worse than guessing the majority class.

  It falls back to exactly that answer, so a shortcut below the baseline would
  mean a rule fired and was wrong more often than abstaining would have been.
  """
  for condition in ("none", "degree", "components", "all"):
    cell = shortcuts.score_cell(condition, task, 3, FIT_GRAPHS, SWEEP)
    assert cell.shortcut >= cell.baseline - 1e-9, cell


def test_none_arm_at_rung_one_is_exactly_the_baseline():
  """The plan's `none` sanity check: an empty primer buys nothing at rung 1."""
  for task in shortcuts.TASKS:
    cell = shortcuts.score_cell("none", task, 1, FIT_GRAPHS, SWEEP)
    assert cell.shortcut == pytest.approx(cell.baseline), (task, cell)


def test_shortcut_is_monotone_in_rung():
  """Each rung grants strictly more, so the score cannot fall."""
  for condition in ("components", "degree", "rwse"):
    for task in shortcuts.TASKS:
      scores = [
          shortcuts.score_cell(condition, task, rung, FIT_GRAPHS, SWEEP).shortcut
          for rung in shortcuts.RUNGS
      ]
      assert scores[0] <= scores[1] + 1e-9 <= scores[2] + 1e-9, (
          condition, task, scores
      )


def test_components_cycle_ladder_matches_the_plan():
  """The anchor the whole rung design is built on."""
  rung1 = shortcuts.score_cell("components", "cycle_check", 1, FIT_GRAPHS, SWEEP)
  rung3 = shortcuts.score_cell("components", "cycle_check", 3, FIT_GRAPHS, SWEEP)
  assert rung1.shortcut > rung1.baseline, "rung 1 should already beat the baseline"
  assert rung3.shortcut == 1.0, "circuit rank is exact"


def test_fitted_rules_do_not_see_their_evaluation_set():
  """In-sample fitting inflates the bar, and more table rows inflate it more.

  Uses its own larger split: at 120 graphs the effect is inside the noise, and
  this is the measurement the whole design rests on.
  """
  split = shortcuts.Split(fit_seed=FIT_SEED, test_seed=CORPUS_SEED, size=400)
  fit_graphs, test_graphs = split.fit_graphs(), split.test_graphs()

  results = {}
  for name in ("cycle_lookup_c", "cycle_lookup_c_n"):
    rule = next(r for r in shortcuts.FITTED if r.name == name)
    results[name] = shortcuts.inflation(
        "components", "cycle_check", 3, rule, fit_graphs, test_graphs
    )

  for name, result in results.items():
    assert result["in_sample"] > result["out_of_sample"], (name, result)

  small, large = results["cycle_lookup_c"], results["cycle_lookup_c_n"]
  assert large["table_rows"] > small["table_rows"]
  assert (large["in_sample"] - large["out_of_sample"]) > (
      small["in_sample"] - small["out_of_sample"]
  ), results


# --- the exact island -----------------------------------------------------


def test_exact_island_bounds_the_hand_written_rules():
  """No rule may beat the information-theoretic ceiling.

  A rule above it is reading something the primer does not state, which is a bug
  rather than a discovery.
  """
  small = [g for g in SWEEP if g.number_of_nodes() <= 6]
  assert small, "no small graphs in the corpus to bound"

  for task in ("edge_existence", "cycle_check", "connected_nodes"):
    island = shortcuts.exact_island(SWEEP, task)
    # Answering with the majority over consistent graphs is at least as good as
    # only answering where they all agree, so the ceiling dominates the
    # determined fraction. A violation means the enumeration is wrong.
    assert island["determined"] <= island["bayes_optimal"] + 1e-9, island

    # And no hand-written rule may beat the ceiling. Scored on the same small
    # graphs the island covers, with the same query draw.
    for condition in ("degree", "all"):
      for rung in shortcuts.RUNGS:
        rows = shortcuts.build_rows(small, condition, task, rung, seed=5)
        fit_rows = shortcuts.build_rows(FIT_GRAPHS, condition, task, rung, seed=5)
        fitted = shortcuts.rank_rules(
            shortcuts.fit_rules(
                shortcuts.HEURISTICS + shortcuts.FITTED, fit_rows, task
            ),
            fit_rows,
            task,
        )
        fallback = shortcuts.majority_answer([g for _, g in fit_rows])
        ours = shortcuts.score_solver(rows, task, fitted, fallback)
        assert ours <= island["bayes_optimal"] + 1e-9, (
            task, condition, rung, ours, island
        )


def test_exact_island_is_exact_on_a_hand_checked_case():
  """A 4-cycle: degrees are all 2, and the consistent set is the three 4-cycles."""
  square = graphqa.canonical(nx.cycle_graph(4))
  island = shortcuts.exact_island([square], "cycle_check")
  # Every graph on 4 nodes with all degrees 2 is a 4-cycle, so a cycle is certain.
  assert island["rows"] == 1
  assert island["determined"] == 1.0
  assert island["bayes_optimal"] == 1.0

  # Degrees (1,1,1,1) admit three different perfect matchings, so which node is
  # adjacent to which is genuinely undetermined.
  matching = graphqa.canonical(nx.Graph([(0, 1), (2, 3)]))
  island = shortcuts.exact_island([matching], "connected_nodes")
  assert island["determined"] == 0.0


def test_exact_island_skips_graphs_it_cannot_enumerate():
  big = [g for g in SWEEP if g.number_of_nodes() > 6]
  island = shortcuts.exact_island(big, "cycle_check")
  assert island["rows"] == 0
  assert island["skipped"] == len(big)
