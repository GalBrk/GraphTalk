"""Checks for scripts/primer_directions.py's parsers and statistics."""
import os
import sys

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import primer_directions as pd_  # noqa: E402

from graphtalk import graphqa, primers  # noqa: E402

VALUES = {0: ("2", "1.00", "0.50", "0.25"), 1: ("2", "1.00", "0.50", "0.25"),
          2: ("1", "0.00", "0.40", "0.00")}


def test_primer_values_reads_every_node_the_renderer_prints():
  g = graphqa.canonical(nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)]))
  g.add_node(4)                                            # isolated: still has a line
  values = pd_.primer_values(primers.build_primer(g, "all"))
  assert sorted(values) == [0, 1, 2, 3, 4]
  assert values[2][0] == "3" and values[4] == ("0", "0.00", "0.00", "0.00")
  assert values[0][1] == "1.00"                             # a triangle corner


def test_source_tells_the_primer_value_from_a_computed_one():
  text = "The primer says 12, but I count 11 connections. Let me recount... 13 nodes."
  assert pd_.stated_values(text) >= {11, 13}
  assert pd_.source(12, 12, text) == "stated"
  assert pd_.source(11, 12, text) == "own"
  assert pd_.source(9, 12, text) == "other"
  assert pd_.source(None, 12, text) is None
  assert 216 in pd_.stated_values(r"so $\frac{432}{2} = 216$")


def test_stated_values_do_not_take_a_node_id_for_a_count():
  assert pd_.stated_values("Let me count node 7's neighbours. Counting for node 12:") == set()


def test_quotes_are_distinct_in_the_primers_phrasing_and_classified():
  text = ("**Node 1** has clustering coefficient 0.00, and node 2's clustering coefficient "
          "is 0.00. Node 0 has degree 2, clustering coefficient 1.00. Node 1 has clustering "
          "coefficient 0.00. Node 2 has clustering coefficient: 0.00.")
  col = pd_.FEATURE["clustering"]
  assert pd_.quotes(text, "clustering") == {(0, "1.00", col), (1, "0.00", col), (2, "0.00", col)}
  assert pd_.classify_quote(0, "1.00", VALUES, col) == "right"
  assert pd_.classify_quote(1, "0.00", VALUES, col) == "near"      # node 2's value
  assert pd_.classify_quote(0, "0.33", VALUES, col) == "other"
  assert pd_.classify_quote(9, "0.00", VALUES, col) is None


def test_rwse_quotes_are_checked_against_the_step_they_name():
  q = pd_.quotes("Node 0 has return probability 0.25 after 3 steps. The return probability "
                 "of node 2 after 2 steps is 0.40.", "rwse")
  assert q == {(0, "0.25", 3), (2, "0.40", 2)}
  assert all(pd_.classify_quote(k, v, VALUES, c) == "right" for k, v, c in q)


def test_near_chance_counts_how_often_other_nodes_print_the_value():
  values = {k: (str(k % 2),) for k in range(10)}       # degrees 0,1,0,1,...
  # Value "0" for node 5: nodes 4 and 6 print "0"; among the rest 3 of 7 do.
  assert pd_.near_chance(5, "0", values, 0) == pytest.approx(1 - (1 - 3 / 7) ** 2)


def test_candidates_follow_the_queried_node_and_the_final_halving():
  t = "Node 7 has degree 5? No. The degree of node 7 is 6. Node 3 has degree 9. That's 4."
  assert [v for _, v in pd_.candidates("node_degree", t, 7)] == [5, 6]
  e = ("Node 0: 12 edges. The number of edges is 160 divided by 2, which is 80. Wait, "
       "42 / 2 = 21, so the graph has 21 edges.")
  assert [v for _, v in pd_.candidates("edge_count", e, None)] == [80, 21, 21]


def test_consistent_compares_the_answer_with_the_models_own_list():
  assert pd_.consistent({3, 5}, 5, True) is True
  assert pd_.consistent({3, 5}, 5, False) is False
  assert pd_.consistent(set(), 5, False) is True
  assert pd_.consistent(None, 5, True) is None
  assert pd_.is_yes("Yes.") is True and pd_.is_yes("no") is False and pd_.is_yes(None) is None


def test_informativeness_is_one_when_the_feature_pins_the_degree():
  assert pd_.informativeness(VALUES, "rwse") == 1.0
  flat = {k: (str(k), "0.10", "0.05", "0.01") for k in range(4)}  # one class, four degrees
  assert pd_.informativeness(flat, "rwse") == 0.25
  assert pd_.informativeness_gain(flat, "rwse") == 0.0
  assert pd_.informativeness_gain(VALUES, "rwse") == 1.0


def test_degree_tercile_ranks_within_the_graph():
  values = {k: (str(k),) for k in range(9)}
  assert [pd_.degree_tercile(values, k) for k in (0, 4, 8)] == [0, 1, 2]


def test_boot_gap_brackets_the_gap_and_ignores_rows_outside_the_groups():
  rng = np.random.default_rng(0)
  x = rng.integers(0, 2, 400).astype(float)
  g = np.arange(400) % 2 == 0
  strata = np.repeat([0.1, 0.2, 0.35, 0.5], 100)
  lo, hi = pd_.boot_gap(x, g, ~g, strata)
  assert lo <= pd_.gap(x, g, ~g) <= hi
  # A missing value outside both groups changes neither the gap nor its interval.
  x2, g1, g2 = np.append(x, np.nan), np.append(g, False), np.append(~g, False)
  s2 = np.append(strata, 0.5)
  assert pd_.gap(x2, g1, g2) == pd_.gap(x, g, ~g)
  assert np.allclose(pd_.boot_gap(x2, g1, g2, s2), [lo, hi])
