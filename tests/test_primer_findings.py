"""Checks for scripts/primer_findings.py: the route classifier, the pairing
rule every paper number rests on, and the Benjamini-Hochberg helper."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import primer_findings as pf  # noqa: E402
import analyze_primer_window as apw  # noqa: E402


def test_route_retrieve_assert_enumerate():
  assert pf.route("Node 7 has degree 5, so the answer is 5.", 7) == "retrieve"
  assert pf.route("Node 7 is connected to nodes 1, 2, 3. That is 3.", 7) == "assert"
  listing = "Node 7 is connected to nodes 1, 2, 3, 4, 5.\n" + "\n".join(
      f"{i}. {i}" for i in range(1, 6)) + "\nSo 5."
  assert pf.route(listing, 7) == "enumerate"
  # A different node's neighbour list is not the queried node's.
  assert pf.route("Node 8 is connected to node 7.", 7) == "retrieve"
  # Opening with an answer and then restating the list is a recount, not retrieval.
  first = "The degree of node 7 is **6**.\n\nNode 7 is connected to nodes 1, 2, 3, 4, 5."
  assert pf.route(first, 7) == "assert"
  assert pf.opens_with(first) == 6
  assert pf.route("**The degree of node 7 is 5**", 7) == "retrieve"
  # Naming the neighbours inline also restates the list.
  assert pf.route("counting the nodes directly connected to node 7, which are: 1, 2.",
                  7) == "assert"
  assert pf.opens_with("Node 7 is connected to nodes 1, 2.") is None


def test_discrepancy_is_reported_not_looked_for():
  assert pf.reports_discrepancy("I count 5, but the primer says 6. That is a discrepancy.")
  assert pf.reports_discrepancy("Wait, that's a contradiction.")
  assert not pf.reports_discrepancy("Let me check if there's any conflicting information.")
  assert not pf.reports_discrepancy("I want to see if there's any inconsistency here.")


def _four_graphs():
  rows = []
  # (exact_none, cap_none, exact_degree, cap_degree) per graph
  for g, (ex_a, cap_a, ex_b, cap_b) in enumerate(
      [(1, 0, 0, 0), (0, 0, 1, 0), (1, 1, 1, 0), (0, 0, 1, 1)]):
    rows += [dict(arm="a", task="t", density_class=0.1, graph_id=g,
                  condition="none", exact=ex_a, hit_cap=cap_a),
             dict(arm="a", task="t", density_class=0.1, graph_id=g,
                  condition="degree", exact=ex_b, hit_cap=cap_b)]
  return pf.with_outcomes(pd.DataFrame(rows))


def test_pairs_keep_every_pair_and_mark_truncation():
  j = pf.pairs(_four_graphs(), "a", "t", "none", "degree", [0.1])
  assert sorted(j.index.get_level_values("graph_id")) == [0, 1, 2, 3]
  g2 = j.xs(2, level="graph_id").iloc[0]
  # The control hit the budget on a text that matched the gold: truncated, not correct.
  assert g2.correct_a == 0 and g2.truncated_a == 1
  assert len(pf.finished(j)) == 2


def test_effect_is_on_the_correct_share_with_the_truncated_change_beside_it():
  j = pf.pairs(_four_graphs(), "a", "t", "none", "degree", [0.1])
  d, lo, hi, broke, fixed, p, n, dt = pf.effect(j)
  assert d == pytest.approx(25.0)      # correct 1/4 -> 2/4
  assert dt == pytest.approx(0.0)      # truncated 1/4 -> 1/4
  assert (broke, fixed, n) == (1, 2, 4)
  assert "truncated +0.0, wrong -25.0" in pf.fmt(pf.effect(j))


def test_cells_keep_a_mostly_truncated_cell_and_flag_it():
  rows = []
  for g in range(10):
    rows.append(dict(arm="qwen3-4b", task="edge_count", density_class=0.1,
                     graph_id=g, condition="none", exact=1, hit_cap=0))
    rows.append(dict(arm="qwen3-4b", task="edge_count", density_class=0.1,
                     graph_id=g, condition="degree", exact=1, hit_cap=int(g < 9)))
  t = apw.cells(pd.DataFrame(rows), {"edge_count/degree": 1.0})
  row = t[t.condition == "degree"].iloc[0]
  assert row.n == 10 and row.baseline == 1.0
  assert row.delta == pytest.approx(-90.0)
  assert row.trunc_b == pytest.approx(0.9) and bool(row.flagged)


def test_boot_interval_does_not_depend_on_earlier_draws():
  j = pd.DataFrame({"x": np.arange(40) % 3},
                   index=pd.MultiIndex.from_product([[0.1, 0.2], range(20)]))
  # With one shared generator the second call would draw different resamples.
  first = pf.boot(j, lambda s: s.x.mean())
  assert np.array_equal(first, pf.boot(j, lambda s: s.x.mean()))


def test_bh_matches_hand_computation_and_is_monotone():
  # ranks 1, 3, 2, 4: raw q = .04, .0533, .06, .5; the monotone pass lowers .06
  q = pf.bh([0.01, 0.04, 0.03, 0.5])
  assert np.allclose(q, [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])
  order = np.argsort([0.01, 0.04, 0.03, 0.5])
  assert np.all(np.diff(q[order]) >= 0)


def test_neighbour_set_reads_lists_and_no_nodes():
  assert pf.neighbour_set("15, 39") == {15, 39}
  assert pf.neighbour_set("No nodes") == set()
  assert pf.neighbour_set("") is None
  assert pf.neighbour_set(float("nan")) is None


def test_rwse_pairs_needs_all_forty_sentences():
  s = " ".join(f"Node {i} has return probability 0.2{i % 3} after 2 steps "
               f"and 0.0{i % 2} after 3 steps." for i in range(40))
  assert len(set(pf.rwse_pairs(s, "x"))) == 6
  with pytest.raises(ValueError, match="node_degree/size40/p0.5/7"):
    pf.rwse_pairs(s.split(" Node 39")[0], "node_degree/size40/p0.5/7")


def _boot_reference(j, stat):
  """The original bootstrap: pd.concat of per-density resamples."""
  groups = [g for _, g in j.groupby(level=0)]
  rng = np.random.default_rng(pf.SEED)
  vals = [stat(pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups]))
          for _ in range(pf.B)]
  return np.percentile(vals, [2.5, 97.5])


def test_boot_draws_the_same_resamples_as_the_concat_version():
  rng = np.random.default_rng(3)
  dens = [0.5] * 7 + [0.1] * 5 + [0.35] * 9          # unsorted, unequal groups
  j = pd.DataFrame({"a": rng.integers(0, 2, 21), "b": rng.random(21)},
                   index=pd.MultiIndex.from_arrays([dens, range(21)]))
  stat = lambda s: 100 * (s.b.mean() - s.a.mean()) + s.index.get_level_values(1)[0]
  assert pf.boot_positions(j) is not None
  assert np.array_equal(pf.boot(j, stat), _boot_reference(j, stat))


def test_record_correct_counts_a_truncated_match_as_not_correct():
  rec = {"response": "The degree of node 3 is 5.", "gold": "5", "task": "node_degree"}
  assert pf.record_correct({**rec, "hit_cap": False}) == 1
  assert pf.record_correct({**rec, "hit_cap": True}) == 0
  assert pf.record_correct({**rec, "response": "It is 4.", "hit_cap": False}) == 0


def test_fmt_rounds_equal_ties_the_same_way_regardless_of_float_noise():
  up = pf.fmt((9.25, 1.0, 2.0, 1, 2, 0.5, 4, 0.0))
  down = pf.fmt((-9.250000000000002, -2.0, -1.0, 1, 2, 0.5, 4, 0.0))
  assert up.startswith("+9.2 ") and down.startswith("-9.2 ")


def test_edge_existence_runs_over_every_arm(capsys):
  rows = []
  for arm in pf.ARMS:
    for dens in pf.DENS4 + pf.DENSHI:
      for g in range(6):
        for c in ["none", "filler"] + pf.PRIMERS:
          yes = g % 2
          rows.append(dict(arm=arm, task="edge_existence", condition=c,
                           density_class=dens, graph_id=f"p{dens}/{g}", exact=int(g < 4),
                           hit_cap=0, gold_is_yes=yes, pred="Yes" if (g < 4) == bool(yes) else "No",
                           n_new_tokens=100))
  pf.B = 20
  pf.edge_existence(pf.with_outcomes(pd.DataFrame(rows)))
  out = capsys.readouterr().out
  assert out.count("[collapse]") == len(pf.ARMS) and "FA removed" in out


def test_one_decimal_rounds_ties_on_the_decimal_value():
  # 2.55 is stored as 2.5499..., -5.75 exactly; both are ties and round to even.
  assert pf.f1(51 / 20) == "+2.6"
  assert pf.f1(-5.75) == "-5.8"
  assert pf.f1(9.25) == "+9.2" and pf.f1(-9.250000000000002) == "-9.2"
  assert pf.f1(0.0) == "+0.0" and pf.f1(3.14159, sign=False) == "3.1"


def test_nodes_listed_counts_distinct_node_lines():
  body = "Node 0 is connected to nodes 1.\nNode 1 is connected to nodes 0, 2.\nNode 2 is connected to nodes 1."
  assert pf.nodes_listed(body) == 3


def test_relative_error_uses_finished_parsed_answers():
  d = pd.DataFrame({"pred": ["90", "110", "", "50"], "gold": ["100", "100", "100", "100"],
                    "hit_cap": [0, 0, 0, 1]})
  assert pf.median_relative_error(d) == pytest.approx(0.10)
