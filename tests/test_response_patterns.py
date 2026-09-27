"""Checks for scripts/response_patterns.py's parsers and its discovery rule."""
import collections
import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import primer_findings as pf  # noqa: E402
import response_patterns as rp  # noqa: E402

# A triangle 0-1-2 plus an isolated node 3: degrees 2, 2, 2, 0; 3 edges.
DEG = {0: 2, 1: 2, 2: 2, 3: 0}
TABLE = ("- **Node 0**: 2 connections\n- Node 1: 2\n| Node | Neighbours | Degree |\n"
         "| 2 | 0, 1 | 2 |\nNode 3 has 0 neighbors\n")


def test_degree_table_reads_every_layout_and_keeps_the_last_value():
  text = TABLE + "Wait, recount: Node 0: 5\n"
  assert rp.degree_table(text, 4) == {0: 5, 1: 2, 2: 2, 3: 0}
  assert rp.degree_table("1. Node 0: 5.\n2. Node 1: degree 3.\n| Node | Degree |\n"
                         "| Node 2 | 4 |", 40) == {0: 5, 1: 3, 2: 4}


def test_degree_table_skips_what_is_not_a_degree():
  assert rp.degree_table("Node 0: 1, 2\nNode 1: 0.33\nNode 9: 4", 4) == {}
  assert rp.degree_table("the total up to Node 18: 48, through Node 19: 50", 40) == {}
  # A running-sum column, a degree column after a neighbour list, an edge list
  # and a headerless table: only the degree columns count.
  assert rp.degree_table("| Node | Degree | Running sum |\n| 0 | 2 | 2 |\n| 1 | 3 | 5 |\n"
                         "\n| Node | Connected nodes | Count |\n| 2 | 7 | 1 |\n"
                         "\n| Node A | Node B |\n| 3 | 7 |\n\n| 4 | 9 |", 40) == {0: 2, 1: 3, 2: 1}


def test_degree_table_does_not_read_a_neighbour_list_column():
  # Node 1 has one neighbour, 32: the list column shows a bare 32, which is not its degree.
  assert rp.degree_table("| Node | Connections | Degree |\n|---|---|---|\n| 0 | 3, 7 | 2 |\n"
                         "| 1 | 32 | 1 |", 40) == {0: 2, 1: 1}
  # A connections column of counts, with no neighbour list in it, is read.
  assert rp.degree_table("| Node | Connections |\n| 0 | 12 |\n| 1 | 14 |", 40) == {0: 12, 1: 14}


def test_degree_table_does_not_crash_on_a_superscript_digit():
  assert rp.degree_table("| Node | Degree |\n| 3 | ² |", 40) == {}


def test_edge_chain_names_the_first_step_that_fails():
  ok = TABLE + "Sum = 6, so 6 / 2 = 3."
  assert rp.edge_chain(ok, DEG, 3, False) == "right"
  assert rp.edge_chain(ok, DEG, 4, False) == "answer"
  assert rp.edge_chain(ok, DEG, None, True) == "cut"
  assert rp.edge_chain(ok.replace("Node 1: 2", "Node 1: 3"), DEG, 3, False) == "values"
  assert rp.edge_chain(ok.replace("Node 3 has 0 neighbors\n", ""), DEG, 3, False) == "nodes"
  assert rp.edge_chain(TABLE + r"$\frac{8}{2} = \boxed{4}$", DEG, 4, False) == "sum"
  assert rp.edge_chain(TABLE + "6 / 2 = 4", DEG, 4, False) == "halving"
  assert rp.edge_chain(TABLE, DEG, None, True) == "cut"
  assert rp.edge_chain(TABLE, DEG, 3, False) == "unparsed"
  assert rp.edge_chain("There are 3 edges.", DEG, 3, False) == "no table"


def test_edge_chain_calls_a_truncated_incomplete_table_cut():
  partial = TABLE.replace("Node 3 has 0 neighbors\n", "")
  assert rp.edge_chain(partial, DEG, None, True) == "cut"


def test_named_cycles_are_checked_against_the_graph():
  g = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
  walks = rp.named_cycles(r"**0 → 1 → 2 → 0**; also 0 -> 2 -> 0, 0-1-3-0, 1 \to 2 \to 0 \to 1")
  assert walks == [[0, 1, 2, 0], [0, 2, 0], [0, 1, 3, 0], [1, 2, 0, 1]]
  assert [rp.is_cycle(w, g) for w in walks] == [True, False, False, True]
  assert rp.named_cycles("39 - 0 + 1 = 40, and 0 → 1 → 2 is open") == []
  assert rp.named_cycles("0 - 1 - 2 - 0 and Node 1 → Node 2 → Node 0 → Node 1") == [
      [0, 1, 2, 0], [1, 2, 0, 1]]


def test_conflict_points_at_the_match_reports_discrepancy_accepts():
  text = "Let me check for any inconsistency." + " " * 400 + "There is a discrepancy here."
  assert pf.reports_discrepancy(text)
  assert rp._conflict(text, [0]) == text.index("discrepancy")
  assert rp._conflict("Let me check for any inconsistency.", [0]) is None


def test_dismisses_needs_a_primer_term_and_reads_contractions():
  find = rp.TEXT["dismisses"][2]
  assert find("The clustering coefficient isn't relevant here.", [0]) is not None
  assert find("The return probabilities aren’t needed.", [0]) is not None
  assert find("Checking the degrees is not necessary.", [0]) is None


def test_tree_bound_needs_an_edge_count_argument():
  find = rp.TEXT["tree_bound"][2]
  assert find("A tree has n - 1 edges; this graph has far more.", [0]) is not None
  assert find("It has many more edges than a tree with 40 nodes.", [0]) is not None
  assert find("If the graph is a tree, it has no cycles.", [0]) is None


def test_lists_neighbours_spots_a_restated_list_in_any_words():
  nbrs = {1, 6, 10, 14}
  assert rp.lists_neighbours("the connections are explicitly listed as: 1, 6, 10, 14", nbrs)
  assert not rp.lists_neighbours("degrees 3, 5, 7 and the answer is 4", nbrs)


def _counts(docs):
  return {k: (collections.Counter(w for s in v for w in s), len(v)) for k, v in docs.items()}


def test_phrase_shifts_needs_the_same_direction_in_three_arms():
  arms = ["a", "b", "c", "d"]
  docs = {}
  for arm in arms:
    docs[(arm, "none")] = [set()] * 10
    # "x y" rises in three arms, "z" in two; "x" only ever appears inside "x y".
    moved = arm != "d"
    docs[(arm, "p")] = [{"x", "x y"} if moved else set()] * 8 + [set()] * 2
    if arm in ("a", "b"):
      docs[(arm, "p")] = [s | {"z"} for s in docs[(arm, "p")]]
  out = rp.phrase_shifts(_counts(docs), arms, "p")
  assert [w for w, _ in out] == ["x y"]
  assert out[0][1] == [0.8, 0.8, 0.8, 0.0]


def test_phrase_shifts_treats_exactly_fifteen_points_alike():
  # 0.35 - 0.20 is 0.1499999... in floating point; it must still count as 15 points.
  arms = ["a", "b", "c"]
  docs = {}
  for arm in arms:
    docs[(arm, "none")] = [{"w"}] * 20 + [set()] * 80
    docs[(arm, "p")] = [{"w"}] * 35 + [set()] * 65
  assert [w for w, _ in rp.phrase_shifts(_counts(docs), arms, "p", min_arms=3)] == ["w"]


def test_relation_class():
  assert rp.relation_class("node_degree", "degree", 1.0, 0.86) == "answer-carrying"
  assert rp.relation_class("edge_existence", "rwse", 0.55, -0.18) == "informative side"
  assert rp.relation_class("edge_count", "rwse", 0.03, 0.0) == "uninformative side"
  assert rp.relation_class("edge_count", "filler", 0.03, 0.0) == "control"
  assert rp.relation_class("cycle_check", "degree", 1.0, 0.0) == "constant task"
