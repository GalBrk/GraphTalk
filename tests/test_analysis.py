"""Tests for graphtalk/analysis.py.

The ground-truth regression is the point of this file: `analysis.build_frame`
must reproduce `analysis/truncated_keys.json`'s exact per-model non-
terminating counts on real thinking-arm data. If a future change to the
detection logic disagrees with any of those labelled rows, that is a bug
in the new code, not a new finding -- see `docs/sweep-findings.md`, which
treats those counts as established.
"""

import collections
import glob
import os

import pandas as pd
import pytest

from graphtalk import analysis
from scripts import score_sweep

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRUNCATED_KEYS = os.path.join(_REPO_ROOT, "analysis", "truncated_keys.json")

# The historical figures, over the complete pre-rewording thinking arm. Kept as
# documentation, not as the assertion: the prompt-rewording re-run regenerates 79
# of the originally 350 labelled rows, and those are judged by their own `hit_cap`
# rather than by this file, so a hardcoded total is wrong for the duration of the
# re-run and again afterwards. The test below pins the *invariant* instead, which
# holds before, during and after -- and still reproduces these numbers when the
# full pre-rewording dataset is what is on disk.
_HISTORICAL_NON_TERMINATING = {
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

  # What the labels imply for the rows actually on disk: a labelled row counts
  # only while it is still present and still lacks its own `hit_cap`; once
  # regenerated it carries the measurement and the file no longer governs it.
  expected = collections.Counter()
  for record in records:
    key = (record["model"], record["instance_id"], record["condition"],
           record["style"])
    recorded = record.get("hit_cap")
    if recorded if recorded is not None else key in truncated:
      expected[record["model"]] += 1

  counts = collections.Counter(
      frame[frame["non_terminating"]].groupby("model").size().to_dict()
  )
  assert counts == expected, f"got {dict(counts)}, expected {dict(expected)}"

  # Not vacuous: every labelled row still on disk and not regenerated must be
  # flagged. This is what would break if `load`/`is_excluded` started dropping a
  # shard, or if the key tuple were built in a different order.
  on_disk = {(r["model"], r["instance_id"], r["condition"], r["style"])
             for r in records if r.get("hit_cap") is None}
  still_governed = truncated & on_disk
  flagged = {(r.model, r.instance_id, r.condition, r.style)
             for r in frame[frame["non_terminating"]].itertuples()}
  assert still_governed <= flagged, (
      f"{len(still_governed - flagged)} labelled rows on disk went unflagged"
  )


def test_is_excluded():
  # Anything under runs/archive/ is out, whatever it is called.
  assert analysis.is_excluded("runs/archive/smoke-gemma4-e4b.jsonl")
  assert analysis.is_excluded("runs/archive/gemma4-12b-think.redo.shard0of12.jsonl")
  assert analysis.is_excluded("runs/archive/anything.jsonl")
  # The tracked sweep is in, including the regeneration files.
  assert not analysis.is_excluded("runs/gemma4-12b.jsonl")
  assert not analysis.is_excluded("runs/gemma4-12b-think.shard0of4.jsonl")
  assert not analysis.is_excluded("runs/gemma4-12b.rerun.jsonl")
  assert not analysis.is_excluded("runs/qwen3-8b-think.rerun.shard0of2.jsonl")
  # Legacy names still excluded wherever they sit, as a safety net.
  assert analysis.is_excluded("runs/smoke-gemma4-e4b.jsonl")


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


# --- recorded hit_cap, and the wording split ---------------------------------


def test_wording_separates_the_two_filler_primers():
  """`condition: filler` means two different primers depending on style.

  Only the `zero_shot` rows were regenerated after the rewording; the obsolete
  `zero_cot` style keeps the original `Node N has <n-1> other nodes` text. A
  mean over both is a mean over two independent variables.
  """
  assert analysis.wording("node_count", "filler", "zero_shot") == "revised"
  assert analysis.wording("node_count", "filler", "zero_cot") == "original"
  assert analysis.wording("edge_existence", "none", "zero_shot") == "revised"
  assert analysis.wording("edge_existence", "none", "zero_cot") == "original"
  # Untouched by either rewording: both wordings are byte-identical here, so the
  # distinction does not arise and must not be invented.
  assert analysis.wording("node_count", "degree", "zero_shot") == "unaffected"
  assert analysis.wording("node_count", "none", "zero_cot") == "unaffected"


def test_backfilled_rows_are_labelled_as_such():
  """A re-tokenized flag must not be reported as the generator's own count."""
  record = _record("node_count/0", "gemma4-12b-think", "A: 5")
  record["n_new_tokens"] = 8190
  record["hit_cap"] = True
  record["token_count_source"] = "retokenized"
  scored = score_sweep.score_records([record])
  frame = analysis.build_frame(scored, set(), {})
  assert bool(frame.iloc[0]["non_terminating"])
  assert frame.iloc[0]["non_terminating_source"] == "retokenized"


def test_recorded_hit_cap_beats_the_ground_truth_file():
  """A row that states how it ended is believed over the hand-made file."""
  record = _record("node_count/0", "gemma4-12b-think", "A: 5")
  record["n_new_tokens"] = 4096
  record["hit_cap"] = True
  scored = score_sweep.score_records([record])
  frame = analysis.build_frame(scored, set(), {})   # not in truncated_keys
  assert bool(frame.iloc[0]["non_terminating"])
  assert frame.iloc[0]["non_terminating_source"] == "generator"
  assert frame.iloc[0]["n_new_tokens"] == 4096


def test_recorded_hit_cap_false_is_not_treated_as_missing():
  """`hit_cap: false` is a measurement, not an absent field.

  Reading it as missing would send the row to `truncated_keys.json` and let a
  stale label override what the generator actually observed.
  """
  record = _record("node_count/0", "gemma4-12b-think", "A: 5")
  record["n_new_tokens"] = 120
  record["hit_cap"] = False
  scored = score_sweep.score_records([record])
  stale = {("gemma4-12b-think", "node_count/0", "none", "zero_shot")}
  frame = analysis.build_frame(scored, stale, {})
  assert not bool(frame.iloc[0]["non_terminating"])
  assert frame.iloc[0]["non_terminating_source"] == "generator"


def test_rows_without_hit_cap_still_use_the_ground_truth_file():
  """The 12,240 rows generated before the instrumentation must not regress."""
  record = _record("node_count/0", "gemma4-12b-think", "A: 5")
  scored = score_sweep.score_records([record])
  truncated = {("gemma4-12b-think", "node_count/0", "none", "zero_shot")}
  frame = analysis.build_frame(scored, truncated, {})
  assert bool(frame.iloc[0]["non_terminating"])
  assert frame.iloc[0]["non_terminating_source"] == "ground_truth_file"
  assert pd.isna(frame.iloc[0]["n_new_tokens"])
