"""Tests for the response parser in `scripts/analyze_rq3_leads.py`."""

import importlib.util
import pathlib
import sys

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "analyze_rq3_leads", _SCRIPTS / "analyze_rq3_leads.py")
rq3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq3)


def test_transcribed_reads_inline_and_bulleted_lists():
  rq3._check_transcribed()


def test_transcribed_stops_at_prose():
  resp = "Node 7 is connected to nodes 1, 2 and 9.\n\nLet's count: 1, 2, 9 -> 3."
  assert rq3.transcribed(resp, 7) == [1, 2, 9]


def test_pairs_keep_truncated_pairs_as_not_correct():
  rows = {("node_degree/size40/p0.35/0", "clustering"): dict(correct=0, cap=True),
          ("node_degree/size40/p0.35/0", "none"): dict(correct=1, cap=False),
          ("node_degree/size40/p0.35/1", "clustering"): dict(correct=1, cap=False),
          ("node_degree/size40/p0.35/1", "none"): dict(correct=0, cap=False)}
  ps = rq3.pairs(rows, "clustering", "none")
  assert sorted(ps) == [("node_degree/size40/p0.35/0", 0, 1, 1, 0),
                        ("node_degree/size40/p0.35/1", 1, 0, 0, 0)]
  s = rq3.summary(ps)
  assert s["delta"] == 0.0 and s["truncated_change"] == 50.0 and s["n"] == 2


def test_tidy_rounds_p_values_to_two_significant_digits_only():
  got = rq3.tidy({"p": 0.000551385, "delta": 4.25, "x": {"holm": 0.107200898, "n": 800},
                  "replication_bonferroni_over_screen": 0.0220554, "replication_p": 1.0})
  assert got == {"p": 0.00055, "delta": 4.25, "x": {"holm": 0.11, "n": 800},
                 "replication_bonferroni_over_screen": 0.022, "replication_p": 1.0}


def test_record_scores_a_truncated_exact_answer_as_truncated_not_correct():
  row = rq3.record({"response": "The answer is 5.", "gold": "5", "hit_cap": True,
                    "task": "node_degree", "n_new_tokens": 10})
  assert (row["exact"], row["correct"], row["cap"], row["pred"]) == (1, 0, True, 5)


def test_summary_is_the_density_scorers_effect():
  import score_density_sweep as sds
  ps = [(f"node_degree/size40/p{d}/{i}", int(i % 3 == 0), int(i % 4 == 0), 0, int(i == 5))
        for d in (0.35, 0.5) for i in range(12)]
  e = sds.effect([(float(p[0].split("/p")[1].split("/")[0]), (p[2], p[4]), (p[1], p[3])) for p in ps])
  s = rq3.summary(ps)
  assert s["delta"] == round(e["d"], 2) and s["ci"] == [round(e["lo"], 2), round(e["hi"], 2)]
  assert s["p"] == e["p"] and s["truncated_change"] == round(e["dt"], 2)
