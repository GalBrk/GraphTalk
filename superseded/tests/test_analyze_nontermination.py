"""Tests for superseded/scripts/analyze_nontermination.py against small run-file fixtures."""

import importlib.util
import json
import pathlib

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "analyze_nontermination.py")
_spec = importlib.util.spec_from_file_location("analyze_nontermination", _SCRIPT)
ant = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ant)


def _write_runs(tmp_path, arm, rows):
  runs_dir = tmp_path / "runs"
  runs_dir.mkdir(exist_ok=True)
  path = runs_dir / f"{arm}.densfull40.shard0of1.jsonl"
  with open(path, "w", encoding="utf-8") as fh:
    for row in rows:
      fh.write(json.dumps(row) + "\n")


def test_new_failures_only_counts_instances_control_did_not_also_cap(
    tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  _write_runs(tmp_path, "arm1", [
      {"instance_id": "a", "condition": "none", "task": "node_degree",
       "hit_cap": False, "gold": "1", "response": "1"},
      {"instance_id": "a", "condition": "degree", "task": "node_degree",
       "hit_cap": True, "gold": "1", "response": "..."},   # new failure
      {"instance_id": "b", "condition": "none", "task": "node_degree",
       "hit_cap": True, "gold": "1", "response": "..."},
      {"instance_id": "b", "condition": "degree", "task": "node_degree",
       "hit_cap": True, "gold": "1", "response": "..."},   # capped both sides
      {"instance_id": "c", "condition": "none", "task": "edge_count",
       "hit_cap": False, "gold": "5", "response": "5"},
      {"instance_id": "c", "condition": "degree", "task": "edge_count",
       "hit_cap": True, "gold": "5", "response": "..."},   # excluded task
  ])
  cond_cap, both_cap = ant.new_failures("arm1", exclude_task="edge_count")
  assert cond_cap == 2   # a and b, not c
  assert both_cap == 1   # only b


def test_nontermination_rate_delta_and_pairing(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  _write_runs(tmp_path, "arm1", [
      {"instance_id": "a", "condition": "none", "task": "node_degree",
       "hit_cap": False, "gold": "1", "response": "1"},
      {"instance_id": "a", "condition": "degree", "task": "node_degree",
       "hit_cap": True, "gold": "1", "response": "..."},
      {"instance_id": "b", "condition": "none", "task": "node_degree",
       "hit_cap": False, "gold": "1", "response": "1"},
      {"instance_id": "b", "condition": "degree", "task": "node_degree",
       "hit_cap": False, "gold": "1", "response": "1"},
  ])
  result = ant.nontermination_rate("arm1")
  assert result["n"] == 2
  assert result["rate_condition"] == 0.5
  assert result["rate_control"] == 0.0
  assert result["delta_pp"] == 50.0


def test_recovery_by_thinking_only_checks_the_broken_arms_wrong_instances(
    tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  _write_runs(tmp_path, "base", [
      {"instance_id": "a", "condition": "degree", "task": "node_degree",
       "hit_cap": False, "gold": "3", "response": "2"},   # wrong
      {"instance_id": "b", "condition": "degree", "task": "node_degree",
       "hit_cap": False, "gold": "3", "response": "3"},   # correct, not in set
  ])
  _write_runs(tmp_path, "think", [
      {"instance_id": "a", "condition": "degree", "task": "node_degree",
       "hit_cap": False, "gold": "3", "response": "3"},   # recovered
      {"instance_id": "b", "condition": "degree", "task": "node_degree",
       "hit_cap": False, "gold": "3", "response": "1"},   # would be wrong but excluded
  ])
  result = ant.recovery_by_thinking("base", "think", "node_degree")
  assert result["n_broken"] == 1
  assert result["n_recovered"] == 1
  assert result["n_missing"] == 0
