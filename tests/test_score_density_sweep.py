"""Checks for scripts/score_density_sweep.py: grouping keys, the truncation rule
(R1), exact-match scoring (R2), pairing, flagged levels, the trend test, and the
between-run-set comparison."""
import importlib.util
import io
import json
import pathlib
import re
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "score_density_sweep.py"
_spec = importlib.util.spec_from_file_location("score_density_sweep", _SCRIPT)
sds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sds)


def _row(iid, cond, response, gold="5", task="node_degree", cap=False, tokens=100):
  return {"instance_id": iid, "condition": cond, "response": response, "gold": gold,
          "task": task, "hit_cap": cap, "n_new_tokens": tokens}


@pytest.mark.parametrize("iid,group,expected", [
    ("node_degree/size40/p0.35/17", "density", 0.35),
    ("node_degree/size80/p0.101/3", "cell", (80, 0.101)),
    ("connected_nodes/size40/p0.2/3", "task", ("connected_nodes", 0.2)),
])
def test_level_of(iid, group, expected):
  assert sds.level_of(iid, group, iid.split("/")[0]) == expected


def test_shared_density_cells_stay_separate():
  rows = [_row("node_degree/size80/p0.101/0", "none", "5"),
          _row("node_degree/size160/p0.101/0", "none", "5")]
  s = sds.summarize(rows, "cell")
  assert {lvl for lvl, _ in s["cells"]} == {(80, 0.101), (160, 0.101)}


def test_mean_degree_recovers_the_target_block():
  assert sds.mean_degree((80, 0.101)) == 8 and sds.mean_degree((160, 0.101)) == 16


def test_a_truncated_row_is_kept_counted_and_not_correct():
  rows = [_row("node_degree/size40/p0.1/0", "none", "The answer is 5.", cap=True),
          _row("node_degree/size40/p0.1/0", "degree", "The answer is 5.")]
  s = sds.summarize(rows, "density")
  cell = s["cells"][(0.1, "none")]
  assert cell["n"] == 1 and cell["truncated"] == 1 and cell["correct"] == 0
  pairs = sds.pairs_for(s["paired"], "degree", None, "none")
  assert [(a, b) for _, a, b in pairs] == [((0, 1), (1, 0))]


def test_connected_nodes_is_scored_by_exact_set_match():
  rows = [_row("connected_nodes/size40/p0.1/0", "none", "1, 2", gold="1, 2, 3",
               task="connected_nodes")]
  s = sds.summarize(rows, "task")
  assert s["cells"][(("connected_nodes", 0.1), "none")]["correct"] == 0


def test_pairs_only_on_shared_instances():
  rows = [_row("node_degree/size40/p0.1/0", "none", "5"),
          _row("node_degree/size40/p0.1/1", "degree", "5")]
  assert sds.pairs_for(sds.summarize(rows, "density")["paired"], "degree", None, "none") == []


def test_effect_reports_the_truncated_change():
  rows = [(0.1, (1, 0), (0, 1)), (0.1, (0, 0), (1, 0)), (0.1, (0, 0), (1, 0)), (0.1, (1, 0), (1, 0))]
  e = sds.effect(rows)
  assert e["d"] == pytest.approx(25.0) and e["dt"] == pytest.approx(25.0)
  assert (e["fixed"], e["broke"], e["n"]) == (2, 1, 4)


def test_flagged_levels_are_left_out_of_the_pool():
  rows = []
  for i in range(10):
    rows.append(_row(f"node_degree/size40/p0.1/{i}", "none", "5"))
    rows.append(_row(f"node_degree/size40/p0.1/{i}", "degree", "5", cap=i < 2))
    rows.append(_row(f"node_degree/size40/p0.2/{i}", "none", "5"))
    rows.append(_row(f"node_degree/size40/p0.2/{i}", "degree", "5"))
  s = sds.summarize(rows, "density")
  assert sds.flagged_levels(s["cells"], "degree", "none") == {0.1}


def test_trend_recovers_a_constructed_positive_slope():
  paired = {}
  for i, d in enumerate([0.1, 0.2, 0.35, 0.5] * 25):
    gain = 1 if d >= 0.35 else 0
    paired[(d, f"g{i}")] = {"none": (0, 0), "degree": (gain, 0)}
  t = sds.trend_test(paired, "degree", draws=2000)
  assert t["slope"] > 0 and t["p_value"] < 0.01


def test_trend_p_value_is_never_zero():
  paired = {(d, f"g{d}{i}"): {"none": (0, 0), "degree": (int(d > 0.3), 0)}
            for d in (0.1, 0.5) for i in range(30)}
  assert sds.trend_test(paired, "degree", draws=200)["p_value"] > 0


def test_versus_pairs_two_run_sets_on_instance_and_condition():
  a = [_row("node_degree/size40/p0.1/0", "none", "4"), _row("node_degree/size40/p0.1/1", "none", "5")]
  b = [_row("node_degree/size40/p0.1/0", "none", "5"), _row("node_degree/size40/p0.1/1", "none", "5")]
  rows = sds.versus_pairs(sds.summarize(a, "density")["paired"],
                          sds.summarize(b, "density")["paired"], "none")
  assert sds.effect(rows)["d"] == pytest.approx(50.0)


def test_report_handles_a_primer_run_at_only_some_levels():
  import io
  rows = []
  for d in (0.1, 0.2, 0.35, 0.65):
    for i in range(6):
      rows.append(_row(f"node_degree/size40/p{d}/{i}", "none", "5" if i % 2 else "4"))
      rows.append(_row(f"node_degree/size40/p{d}/{i}", "clustering", "5"))
      if d >= 0.35:
        rows.append(_row(f"node_degree/size40/p{d}/{i}", "degree", "5"))
  buf = io.StringIO()
  sds.report(sds.summarize(rows, "density"), "t", "title", trend=True, continuum=True, out=buf)
  assert "[t] title" in buf.getvalue() and "degree:" in buf.getvalue()


def test_headroom_can_be_limited_to_some_levels():
  import io
  rows = []
  for d in (0.1, 0.65):
    for i in range(4):
      for cond, ans in (("none", "4"), ("clustering", "5" if i < 2 else "4"), ("degree", "5")):
        rows.append(_row(f"node_degree/size40/p{d}/{i}", cond, ans))
  buf = io.StringIO()
  sds.report(sds.summarize(rows, "density"), "t", "title", headroom=[0.1], out=buf)
  text = buf.getvalue()
  assert "p=0.1: none 0.0 clustering 50.0 degree 100.0 captured +50.0%" in text
  assert "p=0.65: none" not in text


def test_trend_is_printed_for_each_pooled_range():
  import io
  rows = []
  for d in (0.1, 0.2, 0.65, 0.75):
    for i in range(6):
      rows.append(_row(f"node_degree/size40/p{d}/{i}", "none", "4"))
      rows.append(_row(f"node_degree/size40/p{d}/{i}", "clustering", "5" if d < 0.5 and i < 3 else "4"))
  buf = io.StringIO()
  sds.report(sds.summarize(rows, "density"), "t", "title", pools=[("all", None), ("low", [0.1, 0.2])],
             trend=True, out=buf)
  text = buf.getvalue()
  assert "trend over all" in text and "trend over low" in text


def test_versus_leaves_a_flagged_level_out_of_the_pool():
  import io
  first, second = [], []
  for d in (0.1, 0.2):
    for i in range(10):
      first.append(_row(f"node_degree/size40/p{d}/{i}", "none", "4"))
      second.append(_row(f"node_degree/size40/p{d}/{i}", "none", "5", cap=(d == 0.1 and i < 2)))
  buf = io.StringIO()
  sds.report_versus(sds.summarize(first, "density"), sds.summarize(second, "density"), "g", "t", out=buf)
  text = buf.getvalue()
  assert "p=0.1 none" in text and "FLAGGED" in text
  assert "pooled none: +100.0" in text and "n=10 " in text.split("pooled none")[1]


def test_cell_line_reports_the_share_of_finished_answers_saying_n_minus_1():
  import io
  rows = [_row(f"node_degree/size40/p0.85/{i}", "none", "The answer is 39." if i < 3 else "35")
          for i in range(4)]
  rows.append(_row("node_degree/size40/p0.85/4", "none", "39", cap=True))
  buf = io.StringIO()
  sds.report(sds.summarize(rows, "density"), "t", "title", out=buf)
  assert "answers n-1 75.0%" in buf.getvalue()


def test_versus_lines_carry_median_tokens_of_both_sides():
  import io
  first = [_row(f"node_degree/size40/p0.1/{i}", "none", "4", tokens=100) for i in range(3)]
  second = [_row(f"node_degree/size40/p0.1/{i}", "none", "5", tokens=1500) for i in range(3)]
  buf = io.StringIO()
  sds.report_versus(sds.summarize(first, "density"), sds.summarize(second, "density"), "g", "t", out=buf)
  assert "tokens 100 -> 1500 (x15.0)" in buf.getvalue()


def test_prompt_stats_give_median_characters_added_and_mean_edges():
  import io
  recs = [{"instance_id": f"node_degree/size40/p0.1/{i}", "task": "node_degree", "condition": c,
           "prompt": "x" * (100 + i + (50 if c == "clustering" else 0)), "edges": 78 + i}
          for i in range(3) for c in ("none", "clustering")]
  buf = io.StringIO()
  sds.report_prompts(recs, "density", "s", "title", out=buf)
  text = buf.getvalue()
  assert "[s] title" in text
  assert "p=0.1: edges mean 79.0; none 101 chars; clustering 151 chars (+50)" in text


def _flagged_rows():
  """p=0.1 has the primer 20% truncated (flagged); p=0.2 and p=0.35 are clean."""
  rows = []
  for d, gain in ((0.1, 10), (0.2, 3), (0.35, 6)):
    for i in range(10):
      rows.append(_row(f"node_degree/size40/p{d}/{i}", "none", "4"))
      rows.append(_row(f"node_degree/size40/p{d}/{i}", "degree", "5" if i < gain else "4",
                       cap=(d == 0.1 and i < 2)))
  return rows


def test_a_flagged_level_is_left_out_of_pools_and_trends_and_listed():
  buf = io.StringIO()
  sds.report(sds.summarize(_flagged_rows(), "density"), "t", "title", trend=True, out=buf)
  text = buf.getvalue()
  pooled = [l for l in text.splitlines() if l.startswith("  all levels degree:")][0]
  assert "n=20 " in pooled and "(left out: p=0.1)" in pooled
  trend = [l for l in text.split("trend over all levels")[1].splitlines() if "degree:" in l][0]
  assert "n=20" in trend and "(left out: p=0.1)" in trend


def test_versus_pooled_line_lists_the_level_it_leaves_out():
  first = [_row(f"node_degree/size40/p{d}/{i}", "none", "4") for d in (0.1, 0.2) for i in range(10)]
  second = [_row(f"node_degree/size40/p{d}/{i}", "none", "5", cap=(d == 0.1 and i < 2))
            for d in (0.1, 0.2) for i in range(10)]
  buf = io.StringIO()
  sds.report_versus(sds.summarize(first, "density"), sds.summarize(second, "density"), "g", "t", out=buf)
  assert "(left out: p=0.1)" in [l for l in buf.getvalue().splitlines() if "pooled none" in l][0]


def test_prompt_added_characters_are_the_median_of_per_graph_differences():
  recs = []
  for i, (base, extra) in enumerate([(100, 10), (200, 30), (300, 20)]):
    for c, n in (("none", base), ("clustering", base + extra)):
      recs.append({"instance_id": f"node_degree/size40/p0.1/{i}", "task": "node_degree",
                   "condition": c, "prompt": "x" * n, "edges": 1})
  buf = io.StringIO()
  sds.report_prompts(recs, "density", "s", "t", out=buf)
  assert "clustering 230 chars (+20)" in buf.getvalue()


@pytest.mark.parametrize("argv", [["--group", "cell", "--trend"], ["--group", "task", "--continuum"],
                                  ["--group", "density", "--blocks"], ["--group", "cell", "--levels", "0.1"]])
def test_main_rejects_options_its_grouping_cannot_use(argv, monkeypatch):
  monkeypatch.setattr(sys, "argv", ["score_density_sweep.py", "--responses", "missing.jsonl"] + argv)
  with pytest.raises(SystemExit):
    sds.main()


def test_per_level_q_values_are_corrected_within_each_task():
  rows = []
  for task, gold, wins in (("node_degree", "5", 12), ("connected_nodes", "1, 2", 1)):
    for i in range(20):
      iid = f"{task}/size40/p0.1/{i}"
      rows.append(_row(iid, "none", "0", gold=gold, task=task))
      rows.append(_row(iid, "degree", gold if i < wins else "0", gold=gold, task=task))
  buf = io.StringIO()
  sds.report(sds.summarize(rows, "task"), "t", "title", out=buf)
  line = [l for l in buf.getvalue().splitlines() if l.startswith("  node_degree p=0.1 degree:")][0]
  assert re.search(r"\] p=(\S+)", line).group(1) == re.search(r" q=(\S+)", line).group(1)


def test_load_rejects_duplicate_rows_that_differ(tmp_path):
  a = _row("node_degree/size40/p0.1/0", "none", "5")
  f = tmp_path / "r.jsonl"
  f.write_text(json.dumps(a) + "\n" + json.dumps(dict(a, response="4")) + "\n")
  with pytest.raises(ValueError):
    sds.load([str(f)])


def test_load_collapses_identical_duplicates(tmp_path):
  a = _row("node_degree/size40/p0.1/0", "none", "5")
  f = tmp_path / "r.jsonl"
  f.write_text((json.dumps(a) + "\n") * 2)
  assert len(sds.load([str(f)])) == 1


def test_cell_line_states_the_finished_count_when_some_are_truncated():
  rows = [_row(f"node_degree/size40/p0.1/{i}", "none", "5", cap=i == 0) for i in range(4)]
  buf = io.StringIO()
  sds.report(sds.summarize(rows, "density"), "t", "title", out=buf)
  assert "finished 3" in buf.getvalue()
