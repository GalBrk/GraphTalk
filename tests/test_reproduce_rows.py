"""Checks for scripts/reproduce_rows.py, run offline on the committed data.

The subset it picks must cover every task, condition and density with prompts
whose committed responses finished, copied byte for byte from the prompt file;
the comparison must score committed rows against themselves as a perfect
reproduction and catch a changed answer. Also pins the main sweep's default
budgets to the cap their committed rows reach, which a reproduction relies on.

  uv run --no-sync pytest -q tests/test_reproduce_rows.py
"""

import collections
import glob
import json
import pathlib
import sys

import pytest

from graphtalk import models
from scripts import build_raw_frame as brf
from scripts import reproduce_rows as rr

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = str(ROOT / "data" / "runs")
PROMPTS = str(ROOT / rr.PROMPTS)
MODEL = "qwen3-1.7b"


def _run(monkeypatch, *argv):
  monkeypatch.setattr(sys, "argv", ["reproduce_rows.py", *argv])
  rr.main()


@pytest.fixture(scope="module")
def committed():
  return rr.load_rows(rr.committed_paths(MODEL, runs_dir=RUNS))


@pytest.fixture(scope="module")
def prompt_lines():
  return {rr.key_of(r): line for line, r in rr.load_lines(PROMPTS)}


@pytest.fixture
def subset(tmp_path, monkeypatch, capsys):
  out = tmp_path / "subset.jsonl"
  _run(monkeypatch, "subset", "--model", MODEL, "--rows", "48", "--max-tokens", "2000",
       "--out", str(out), "--prompts", PROMPTS, "--runs-dir", RUNS)
  return rr.load_lines(str(out)), capsys.readouterr().out


def test_run_sets_are_the_frames():
  assert rr.RUN_SETS == brf.TAGS


def test_subset_covers_every_task_condition_and_density(subset, prompt_lines):
  lines, printed = subset
  records = [r for _, r in lines]
  assert len(records) == 48
  every = [json.loads(line) for line in prompt_lines.values()]
  assert {r["task"] for r in records} == {r["task"] for r in every}
  assert {r["condition"] for r in records} == {r["condition"] for r in every}
  assert ({rr.density_of(r["instance_id"]) for r in records}
          == {rr.density_of(r["instance_id"]) for r in every})
  cells = collections.Counter((r["task"], rr.density_of(r["instance_id"])) for r in records)
  assert len(cells) == 24 and set(cells.values()) == {2}
  assert "WARNING" not in printed


def test_subset_rows_are_prompt_lines_whose_committed_rows_finished(subset, committed,
                                                                     prompt_lines):
  lines, _ = subset
  assert len({rr.key_of(r) for _, r in lines}) == len(lines)
  for line, r in lines:
    assert line == prompt_lines[rr.key_of(r)]
    row = committed[rr.key_of(r)]
    assert not row["hit_cap"] and row["n_new_tokens"] <= 2000


def test_subset_is_deterministic(tmp_path, monkeypatch, capsys, subset):
  again = tmp_path / "again.jsonl"
  _run(monkeypatch, "subset", "--model", MODEL, "--rows", "48", "--max-tokens", "2000",
       "--out", str(again), "--prompts", PROMPTS, "--runs-dir", RUNS)
  assert [line for line, _ in rr.load_lines(str(again))] == [line for line, _ in subset[0]]


def test_subset_refuses_an_output_under_data_runs(monkeypatch):
  with pytest.raises(SystemExit, match="inside"):
    _run(monkeypatch, "subset", "--model", MODEL, "--rows", "4", "--max-tokens", "2000",
         "--out", RUNS + "/archive/subset.jsonl", "--prompts", PROMPTS, "--runs-dir", RUNS)


def test_pick_spreads_and_refuses_too_few_eligible():
  records, committed = [], {}
  for index in range(3):
    for density in ("p0.1", "p0.5"):
      for task in ("node_degree", "edge_count"):
        for condition in ("none", "degree", "filler"):
          r = {"instance_id": f"{task}/size40/{density}/{index}", "task": task,
               "condition": condition, "style": "zero_shot"}
          records.append(r)
          long = task == "edge_count" and density == "p0.5"
          committed[rr.key_of(r)] = {"hit_cap": index == 0,
                                     "n_new_tokens": 5000 if long else 100}
  chosen = [records[i] for i in rr.pick(records, committed, 6, 1000)]
  assert {r["task"] for r in chosen} == {"node_degree", "edge_count"}
  assert {r["condition"] for r in chosen} == {"none", "degree", "filler"}
  assert {rr.density_of(r["instance_id"]) for r in chosen} == {"p0.1", "p0.5"}
  assert all(rr.finished_below(committed[rr.key_of(r)], 1000) for r in chosen)
  with pytest.raises(ValueError, match="only 18"):
    rr.pick(records, committed, 19, 1000)


def _regenerated(subset, committed):
  rows = []
  for _, r in subset[0]:
    row = dict(committed[rr.key_of(r)])
    row["overflow"] = False             # run_sweep.py writes it; committed rows lack it
    rows.append(row)
  return rows


def test_committed_rows_against_themselves_reproduce_exactly(subset, committed, tmp_path,
                                                              monkeypatch, capsys):
  rows = _regenerated(subset, committed)
  results = rr.compare(rows, committed)
  assert all(r["same_text"] and r["same_answer"] and r["same_outcome"] for r in results)
  path = tmp_path / "regenerated.jsonl"
  path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
  subset_path = tmp_path / "subset.jsonl"
  subset_path.write_text("".join(line for line, _ in subset[0]), encoding="utf-8")
  _run(monkeypatch, "compare", "--model", MODEL, "--regenerated", str(path),
       "--subset", str(subset_path), "--runs-dir", RUNS)
  printed = capsys.readouterr().out
  assert "exact text             48/48 (100.0%)" in printed
  assert "same extracted answer  48/48 (100.0%)" in printed
  assert "same outcome           48/48 (100.0%)" in printed
  assert "subset rows with no regenerated row: 0" in printed
  assert "rows that differ" not in printed


def test_compare_catches_a_changed_answer_and_a_missing_row(subset, committed, tmp_path,
                                                             monkeypatch, capsys):
  rows = _regenerated(subset, committed)
  target = next(r for r in rows if r["task"] == "node_count")
  target["response"] = target["response"][:30] + "\n\nThe answer is 7."
  results = {(r["instance_id"], r["condition"]): r for r in rr.compare(rows, committed)}
  hit = results[(target["instance_id"], target["condition"])]
  assert not hit["same_text"] and not hit["same_answer"] and hit["answer"][1] == "7"
  assert hit["prefix"] == 30
  assert sum(r["same_text"] for r in results.values()) == len(rows) - 1

  path = tmp_path / "regenerated.jsonl"
  kept = [r for r in rows if r["task"] != "cycle_check"] + [
      r for r in rows if r["task"] == "cycle_check"][1:]
  path.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
  subset_path = tmp_path / "subset.jsonl"
  subset_path.write_text("".join(line for line, _ in subset[0]), encoding="utf-8")
  _run(monkeypatch, "compare", "--model", MODEL, "--regenerated", str(path),
       "--subset", str(subset_path), "--runs-dir", RUNS)
  printed = capsys.readouterr().out
  assert "subset rows with no regenerated row: 1" in printed
  assert target["instance_id"] in printed and "'7'" in printed


def test_compare_refuses_rows_it_cannot_pair(committed):
  row = dict(next(iter(committed.values())))
  with pytest.raises(ValueError, match="gold/model"):
    rr.compare([dict(row, gold="not the gold")], committed)
  with pytest.raises(ValueError, match="no committed row"):
    rr.compare([dict(row, instance_id="node_count/size40/p0.1/999")], committed)


@pytest.mark.parametrize("arm", brf.ARMS)
def test_main_sweep_default_budget_is_the_committed_cap(arm):
  """Every capped densfull40 row of an arm stops at the registry's budget, so
  `run_sweep.py` with no override regenerates the main sweep at its own cap."""
  caps = set()
  for path in glob.glob(f"{RUNS}/{arm}.densfull40.shard*.jsonl"):
    with open(path, encoding="utf-8") as handle:
      for line in handle:
        row = json.loads(line)
        if row["hit_cap"]:
          caps.add(row["n_new_tokens"])
  assert caps == {models.budget(models.MODELS[arm], "zero_shot")}
