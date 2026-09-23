"""Checks for scripts/primer_findings.py: the route classifier, the pairing
rule every paper number rests on, and the Benjamini-Hochberg helper."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import primer_findings as pf  # noqa: E402


def test_route_retrieve_assert_enumerate():
  assert pf.route("Node 7 has degree 5, so the answer is 5.", 7) == "retrieve"
  assert pf.route("Node 7 is connected to nodes 1, 2, 3. That is 3.", 7) == "assert"
  listing = "Node 7 is connected to nodes 1, 2, 3, 4, 5.\n" + "\n".join(
      f"{i}. {i}" for i in range(1, 6)) + "\nSo 5."
  assert pf.route(listing, 7) == "enumerate"
  # A different node's neighbour list is not the queried node's.
  assert pf.route("Node 8 is connected to node 7.", 7) == "retrieve"


def test_pairs_drop_a_pair_when_either_side_truncates():
  rows = []
  for g, (ex_a, cap_a, ex_b, cap_b) in enumerate(
      [(1, 0, 0, 0), (0, 0, 1, 0), (1, 1, 1, 0), (0, 0, 1, 1)]):
    rows += [dict(arm="a", task="t", density_class=0.1, graph_id=g,
                  condition="none", exact=ex_a, hit_cap=cap_a),
             dict(arm="a", task="t", density_class=0.1, graph_id=g,
                  condition="degree", exact=ex_b, hit_cap=cap_b)]
  j = pf.pairs(pd.DataFrame(rows), "a", "t", "none", "degree", [0.1])
  assert sorted(j.index.get_level_values("graph_id")) == [0, 1]


def test_bh_matches_hand_computation_and_is_monotone():
  # ranks 1, 3, 2, 4: raw q = .04, .0533, .06, .5; the monotone pass lowers .06
  q = pf.bh([0.01, 0.04, 0.03, 0.5])
  assert np.allclose(q, [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])
  order = np.argsort([0.01, 0.04, 0.03, 0.5])
  assert np.all(np.diff(q[order]) >= 0)
