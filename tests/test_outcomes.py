"""R1: a response that hit the token budget is its own outcome."""
import pandas as pd
import pytest

from graphtalk import outcomes as oc


def test_truncated_wins_over_the_text_it_stopped_on():
  assert list(oc.outcome([1, 0, 1, 0], [True, True, False, False])) == [
      "truncated", "truncated", "correct", "wrong"]


def test_outcome_accepts_pandas_columns():
  f = pd.DataFrame({"exact": [1.0, 0.0], "hit_cap": [0, 1]})
  assert list(oc.outcome(f.exact, f.hit_cap)) == ["correct", "truncated"]


def test_shares_are_of_all_responses_and_sum_to_one():
  s = oc.shares(["correct", "wrong", "truncated", "correct"])
  assert s == {"correct": 0.5, "wrong": 0.25, "truncated": 0.25}


def test_shares_of_no_responses_is_an_error():
  with pytest.raises(ValueError):
    oc.shares([])


def test_flag_is_inclusive_at_fifteen_percent():
  assert oc.flagged(["truncated"] * 3 + ["correct"] * 17)
  assert not oc.flagged(["truncated"] * 2 + ["correct"] * 18)
