"""Tests for graphtalk/analysis.py.

The ground-truth regression is the point of this file: `analysis.build_frame`
must reproduce `analysis/truncated_keys.json`'s exact per-model non-
terminating counts on real thinking-arm data. If a future change to the
detection logic disagrees with any of those 350 labelled rows, that is a bug
in the new code, not a new finding -- see `docs/sweep-findings.md`, which
treats those counts as established.
"""

import glob
import os

import pytest

from graphtalk import analysis
from scripts import score_sweep

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRUNCATED_KEYS = os.path.join(_REPO_ROOT, "analysis", "truncated_keys.json")

_EXPECTED_NON_TERMINATING = {
    "gemma4-e4b-think": 0,
    "gemma4-12b-think": 282,
    "qwen3-8b-think": 49,
    "qwen3-14b-think": 19,
}


def _think_run_paths() -> list[str]:
  paths = glob.glob(os.path.join(_REPO_ROOT, "runs", "*-think.shard*.jsonl"))
  return [p for p in paths if not analysis.is_excluded(p)]


@pytest.mark.skipif(
    not os.path.exists(_TRUNCATED_KEYS), reason="ground truth file not present"
)
def test_non_terminating_matches_ground_truth():
  paths = _think_run_paths()
  if not paths:
    pytest.skip("thinking-arm run files not present")

  records = score_sweep.score_records(score_sweep.load(paths))
  truncated = analysis.load_truncated_keys(_TRUNCATED_KEYS)
  frame = analysis.build_frame(records, truncated, {})

  counts = frame[frame["non_terminating"]].groupby("model").size().to_dict()
  for model, expected in _EXPECTED_NON_TERMINATING.items():
    assert counts.get(model, 0) == expected, (
        f"{model}: expected {expected} non-terminating rows, got "
        f"{counts.get(model, 0)}"
    )


def test_is_excluded():
  assert analysis.is_excluded("runs/smoke-gemma4-e4b.jsonl")
  assert analysis.is_excluded("runs/gemma4-12b-think.redo.shard0of12.jsonl")
  assert not analysis.is_excluded("runs/gemma4-12b.jsonl")
  assert not analysis.is_excluded("runs/gemma4-12b-think.shard0of4.jsonl")


def _record(instance_id, model, response, gold=" 5.", condition="none"):
  return {
      "instance_id": instance_id,
      "task": "node_count",
      "condition": condition,
      "style": "zero_shot",
      "gold": gold,
      "model": model,
      "response": response,
  }


def test_failure_type_classification():
  records = [
      _record("node_count/0", "gemma4-12b", "A: 5"),   # correct
      _record("node_count/1", "gemma4-12b", "A: 9"),   # wrong
      _record("node_count/2", "gemma4-12b", ""),        # unparsed
  ]
  scored = score_sweep.score_records(records)
  frame = analysis.build_frame(scored, set(), {})
  types = dict(zip(frame["instance_id"], frame["failure_type"]))
  assert types["node_count/0"] == "correct"
  assert types["node_count/1"] == "wrong"
  assert types["node_count/2"] == "unparsed"


def test_non_terminating_takes_precedence_even_when_parsed():
  # A row can be in truncated_keys.json (ground truth: cut off mid-thought)
  # and still `parse`, because the extractor finds an integer in the
  # abandoned working -- non_terminating must win over wrong/correct.
  record = _record("node_count/0", "gemma4-12b-think", "A: 5")
  scored = score_sweep.score_records([record])
  truncated = {("gemma4-12b-think", "node_count/0", "none", "zero_shot")}
  frame = analysis.build_frame(scored, truncated, {})
  assert frame.iloc[0]["failure_type"] == "non_terminating"


def test_shortcut_score_joins_on_task_and_condition():
  record = _record("node_count/0", "gemma4-12b", "A: 5", condition="degree")
  scored = score_sweep.score_records([record])
  frame = analysis.build_frame(scored, set(), {("node_count", "degree"): 0.4})
  assert frame.iloc[0]["shortcut_score"] == 0.4


def test_sample_failures_excludes_correct_and_respects_stratum_cap():
  records = [
      _record(f"node_count/{i}", "gemma4-12b", "A: 9" if i % 2 else "A: 5")
      for i in range(10)
  ]
  scored = score_sweep.score_records(records)
  frame = analysis.build_frame(scored, set(), {})
  sample = analysis.sample_failures(frame, n_per_stratum=2, seed=1234)
  assert (sample["failure_type"] != "correct").all()
  assert len(sample) <= 2
