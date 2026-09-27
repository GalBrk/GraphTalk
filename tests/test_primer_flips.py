"""Checks for scripts/primer_flips.py's text helpers, on real rendered primers."""
import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import primer_flips as pfl  # noqa: E402

from graphtalk import graphqa, primers  # noqa: E402

# A 12-node path: nodes 1 and 10 both exist, so "Node 1 " must not match "Node 10".
G = graphqa.canonical(nx.path_graph(12))


def test_primer_says_keeps_the_queried_nodes_only():
  assert (pfl.primer_says(primers.build_primer(G, "degree"), (1, 10))
          == "Node 1 has degree 2. Node 10 has degree 2.")
  says = pfl.primer_says(primers.build_primer(G, "all"), (1,))
  assert says.startswith("Node 1 has degree 2,") and says.count("Node ") == 1
  assert pfl.primer_says(primers.build_primer(G, "rwse"), ()) == ""


def test_primer_says_keeps_a_whole_graph_sentence():
  p = primers.build_primer(G, "components")
  assert p.startswith("This graph ") and pfl.primer_says(p, (1,)) == p


def test_start_and_end():
  think = "<think>\nOkay, let's see.\n</think>\n\nThe answer is **7**."
  assert pfl.start(think) == "Okay, let's see. </think> The answer is **7**."
  assert pfl.end(think) == "The answer is **7**."
  cut = "<think>\n" + "x " * 400          # truncated: no </think>
  assert pfl.end(cut) == pfl.flat(cut)[-pfl.END:]
