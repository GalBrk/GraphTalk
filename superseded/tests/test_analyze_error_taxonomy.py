"""Tests for superseded/scripts/analyze_error_taxonomy.py.

Each function is I/O-heavy (globs runs/ and reads a prompts file), so these
write small temp fixtures rather than mocking internals -- the point is to
pin the actual classification logic (off-by-k bucketing, neighbour-copy
detection, hit/false-alarm counting, the "39" rate) against known inputs.
"""

import importlib.util
import json
import pathlib

import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "analyze_error_taxonomy.py")
_spec = importlib.util.spec_from_file_location("analyze_error_taxonomy", _SCRIPT)
aet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aet)


def _write_jsonl(path, rows):
  with open(path, "w", encoding="utf-8") as fh:
    for row in rows:
      fh.write(json.dumps(row) + "\n")


@pytest.fixture
def tmp_prompts(tmp_path):
  # A 4-node star: node 0 connects to 1, 2, 3 (degree 3); leaves have degree 1.
  prompt = ("Node 0 is connected to nodes 1, 2, 3. Node 1 is connected to "
            "node 0. Node 2 is connected to node 0. Node 3 is connected to "
            "node 0. What is the degree of node 1?")
  path = tmp_path / "prompts.jsonl"
  _write_jsonl(path, [{"task": "node_degree", "instance_id": "g0",
                       "prompt": prompt}])
  return str(path)


def test_neighbour_copy_is_detected_when_wrong_answer_equals_a_neighbours_degree(
    tmp_prompts, tmp_path):
  # gold is 1 (node 1's true degree); the model says 3, which is node 1's
  # only neighbour's (node 0's) degree -- a copy, not a random miss.
  runs = tmp_path / "qwen3-1.7b.densfull40.shard0of1.jsonl"
  _write_jsonl(runs, [{"task": "node_degree", "instance_id": "g0",
                       "condition": "none", "gold": "1",
                       "response": "The degree of node 1 is 3.",
                       "hit_cap": False}])
  errors, _ = aet.node_degree_taxonomy(
      tmp_prompts, str(tmp_path / "{arm}.densfull40.shard*.jsonl"))
  key = ("qwen3-1.7b", "none")
  assert errors[key]["n_wrong"] == 1
  assert errors[key]["neighbour_copy"] == 1
  assert errors[key]["off_by"] == {"2": 1}


def test_correct_answers_are_not_counted_as_errors(tmp_prompts, tmp_path):
  runs = tmp_path / "qwen3-1.7b.densfull40.shard0of1.jsonl"
  _write_jsonl(runs, [{"task": "node_degree", "instance_id": "g0",
                       "condition": "none", "gold": "1",
                       "response": "The degree of node 1 is 1.",
                       "hit_cap": False}])
  errors, _ = aet.node_degree_taxonomy(
      tmp_prompts, str(tmp_path / "{arm}.densfull40.shard*.jsonl"))
  assert errors == {}


def test_edge_existence_hit_and_false_alarm_rates(tmp_path):
  runs = tmp_path / "qwen3-1.7b.densfull40.shard0of1.jsonl"
  _write_jsonl(runs, [
      {"task": "edge_existence", "instance_id": "g0", "condition": "none",
       "gold": "Yes", "response": "Yes.", "hit_cap": False},   # hit
      {"task": "edge_existence", "instance_id": "g1", "condition": "none",
       "gold": "Yes", "response": "No.", "hit_cap": False},    # miss
      {"task": "edge_existence", "instance_id": "g2", "condition": "none",
       "gold": "No", "response": "Yes.", "hit_cap": False},    # false alarm
      {"task": "edge_existence", "instance_id": "g3", "condition": "none",
       "gold": "No", "response": "No.", "hit_cap": False},     # correct reject
  ])
  out = aet.edge_existence_taxonomy(str(tmp_path / "{arm}.densfull40.shard*.jsonl"))
  v = out[("qwen3-1.7b", "none")]
  assert v["hit_rate"] == pytest.approx(0.5)
  assert v["false_alarm_rate"] == pytest.approx(0.5)


def test_node_count_39_rate(tmp_path):
  runs = tmp_path / "qwen3-1.7b.densfull40.shard0of1.jsonl"
  _write_jsonl(runs, [
      {"task": "node_count", "instance_id": "g0", "condition": "filler",
       "gold": "40", "response": "39.", "hit_cap": False},
      {"task": "node_count", "instance_id": "g1", "condition": "filler",
       "gold": "40", "response": "40.", "hit_cap": False},
  ])
  out = aet.node_count_taxonomy(str(tmp_path / "{arm}.densfull40.shard*.jsonl"))
  v = out[("qwen3-1.7b", "filler")]
  assert v["rate_39"] == pytest.approx(0.5)
  assert v["rate_correct"] == pytest.approx(0.5)


def test_hit_cap_rows_are_excluded_everywhere(tmp_prompts, tmp_path):
  runs = tmp_path / "qwen3-1.7b.densfull40.shard0of1.jsonl"
  _write_jsonl(runs, [{"task": "node_degree", "instance_id": "g0",
                       "condition": "none", "gold": "1",
                       "response": "3.", "hit_cap": True}])
  errors, position = aet.node_degree_taxonomy(
      tmp_prompts, str(tmp_path / "{arm}.densfull40.shard*.jsonl"))
  assert errors == {}
  assert position == {}
