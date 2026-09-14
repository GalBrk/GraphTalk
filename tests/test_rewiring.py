"""Degree-preserving rewiring, whose four invariants *are* the experimental design.

The whole point of rewiring rather than swapping generator families is that
everything except triangle structure is held fixed -- including the rendered
prompt. If any invariant silently broke, the paired comparison would quietly
become a comparison between two different tasks with two different gold answers,
and nothing downstream would notice. So each one is pinned here, and the
prompt-identity check goes through `prompts.build_prompt` rather than
re-deriving the encoding, so a change to the encoder fails this test too.
"""

import networkx as nx
import pytest

from graphtalk import graphqa
from graphtalk import prompts
from graphtalk import rewiring


QUESTION = "What is the degree of node 0?"


def _graph(n=60, m=180, seed=4):
  return nx.gnm_random_graph(n, m, seed=seed)


# --- invariants -------------------------------------------------------------

@pytest.mark.parametrize("direction", ["low", "high"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_rewiring_preserves_degree_sequence(direction, seed):
  base = _graph(seed=seed)
  out = rewiring.rewire(base, direction, seed=seed)
  assert (sorted(d for _, d in base.degree())
          == sorted(d for _, d in out.degree()))


@pytest.mark.parametrize("direction", ["low", "high"])
def test_rewiring_preserves_node_set_and_edge_count(direction):
  base = _graph()
  out = rewiring.rewire(base, direction, seed=7)
  assert sorted(base.nodes()) == sorted(out.nodes())
  assert base.number_of_edges() == out.number_of_edges()


def test_rewiring_introduces_no_self_loops():
  out = rewiring.rewire(_graph(), "high", seed=7)
  assert not any(u == v for u, v in out.edges())


def test_assert_invariants_catches_a_degree_change():
  base = _graph()
  broken = base.copy()
  u, v = next(iter(broken.edges()))
  broken.remove_edge(u, v)
  with pytest.raises(AssertionError):
    rewiring.assert_invariants(base, broken)


# --- the load-bearing claim -------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rendered_prompt_length_is_unchanged(seed):
  """The reason this design defuses the length confound by construction.

  `filler` prices prompt length at -11.7pp on dense graphs, which is larger than
  most primer effects on record. Rewiring has to leave length untouched or the
  comparison inherits that confound.
  """
  base = _graph(seed=seed)
  low, high = rewiring.rewired_pair(base, seed=seed)
  rendered = [
      prompts.build_prompt(graphqa.canonical(g), "none", QUESTION)
      for g in (base, low, high)
  ]
  assert len(rendered[0]) == len(rendered[1]) == len(rendered[2])


def test_gold_answers_are_unchanged():
  """Degree preservation means every `node_degree` answer survives rewiring."""
  base = _graph()
  low, high = rewiring.rewired_pair(base, seed=1)
  for node in base:
    assert base.degree(node) == low.degree(node) == high.degree(node)


# --- direction --------------------------------------------------------------

def test_rewiring_moves_transitivity_in_the_requested_direction():
  base = _graph()
  low, high = rewiring.rewired_pair(base, seed=2)
  assert nx.transitivity(low) < nx.transitivity(base) < nx.transitivity(high)


def test_high_direction_opens_a_wide_clustering_gap():
  """The gap is the experiment's independent variable, so a narrow one is a bug.

  Measured at this size the swing is roughly 0.00 -> 0.6; 0.15 is a loose floor
  that catches a broken accept rule without pinning an exact stochastic value.
  """
  base = _graph()
  low, high = rewiring.rewired_pair(base, seed=2)
  assert nx.transitivity(high) - nx.transitivity(low) > 0.15


# --- guards -----------------------------------------------------------------

def test_unknown_direction_raises():
  with pytest.raises(ValueError, match="direction"):
    rewiring.rewire(_graph(), "sideways")


def test_tiny_graph_is_returned_unchanged():
  tiny = nx.Graph([(0, 1)])
  out = rewiring.rewire(tiny, "high", seed=0)
  assert sorted(out.edges()) == [(0, 1)]
