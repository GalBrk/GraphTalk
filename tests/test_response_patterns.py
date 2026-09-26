"""Checks for scripts/response_patterns.py's parsers and its discovery rule."""
import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import response_patterns as rp  # noqa: E402

# A triangle 0-1-2 plus an isolated node 3: degrees 2, 2, 2, 0; 3 edges.
DEG = {0: 2, 1: 2, 2: 2, 3: 0}
TABLE = "- **Node 0**: 2 connections\n- Node 1: 2\n| 2 | 0, 1 | 2 |\nNode 3 has 0 neighbors\n"


def test_degree_table_reads_every_layout_and_keeps_the_last_value():
  text = "| Node | Degree |\n|---|---|\n" + TABLE + "Wait, recount: Node 0: 5\n"
  assert rp.degree_table(text, 4) == {0: 5, 1: 2, 2: 2, 3: 0}


def test_degree_table_skips_neighbour_lists_decimals_and_foreign_ids():
  assert rp.degree_table("Node 0: 1, 2\nNode 1: 0.33\nNode 9: 4", 4) == {}


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


def test_named_cycles_are_checked_against_the_graph():
  g = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
  walks = rp.named_cycles(r"**0 → 1 → 2 → 0**; also 0 -> 2 -> 0, 0-1-3-0, 1 \to 2 \to 0 \to 1")
  assert walks == [[0, 1, 2, 0], [0, 2, 0], [0, 1, 3, 0], [1, 2, 0, 1]]
  assert [rp.is_cycle(w, g) for w in walks] == [True, False, False, True]
  assert rp.named_cycles("39 - 0 + 1 = 40, and 0 → 1 → 2 is open") == []


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
  out = rp.phrase_shifts(docs, arms, "p")
  assert [w for w, _ in out] == ["x y"]
  assert out[0][1] == [0.8, 0.8, 0.8, 0.0]


def test_relation_class():
  assert rp.relation_class("node_degree", "degree", 1.0, 0.86) == "answer-carrying"
  assert rp.relation_class("edge_existence", "rwse", 0.55, -0.18) == "informative side"
  assert rp.relation_class("edge_count", "rwse", 0.03, 0.0) == "uninformative side"
  assert rp.relation_class("edge_count", "filler", 0.03, 0.0) == "control"
  assert rp.relation_class("cycle_check", "degree", 1.0, 0.0) == "constant task"
