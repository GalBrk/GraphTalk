"""Tests for `graphtalk.scoring.extract_answer_first`.

No test file existed for `graphtalk/scoring.py` before this session; this one
stays scoped to `extract_answer_first`, added alongside the non-termination
"looped on the correct answer" diagnostic in `graphtalk/analysis.py`'s
`build_frame` (see `scripts/check_significance.py`'s `n_looped_on_correct_answer`
column). It does not attempt to cover `extract_answer` itself, which has no
existing regression suite to extend here.
"""

import pytest

from graphtalk import scoring


@pytest.mark.parametrize("task,text,expected", [
    (
        "node_degree",
        "The degree of node 7 is 2. Answer: 2. Let me double check... "
        "Answer: 2. Final answer: 2.",
        "2",
    ),
    (
        "edge_existence",
        "Answer: Yes. Repeating the conclusion: Answer: Yes. Answer: Yes.",
        "Yes",
    ),
    (
        "connected_nodes",
        "Node 3 is connected to nodes 1, 2, 3.\n"
        "Node 3 is connected to nodes 1, 2, 3.",
        "1, 2, 3",
    ),
])
def test_extract_answer_first_agrees_with_extract_answer_on_a_stable_loop(
    task, text, expected
):
  """A response that settles on one answer and then loops/repeats it must
  have `extract_answer_first` and `extract_answer` agree -- this is the
  "looped on the correct answer" signal `graphtalk.analysis.build_frame`
  relies on: first == last means stable, not drifting.
  """
  assert scoring.extract_answer(text, task) == expected
  assert scoring.extract_answer_first(text, task) == expected


def test_extract_answer_first_returns_an_earlier_value_on_a_drifting_integer_response():
  # Two lines, not one: `_MARKER`'s capture is greedy and can't cross a
  # newline, so a single-line "Answer: 5. ... Final answer: 7." would only
  # ever produce one marker match (spanning to the end of the line) and
  # `extract_answer`/`extract_answer_first` would agree on it -- exactly the
  # stable-loop shape the parametrized test above covers, not a drift.
  text = "Answer: 5.\nWait, let me recompute that.\nFinal answer: 7."
  assert scoring.extract_answer(text, "node_count") == "7"
  assert scoring.extract_answer_first(text, "node_count") == "5"


def test_extract_answer_first_returns_an_earlier_value_on_a_drifting_boolean_response():
  text = "Answer: No. Actually, on reflection, Answer: Yes."
  assert scoring.extract_answer(text, "cycle_check") == "Yes"
  assert scoring.extract_answer_first(text, "cycle_check") == "No"


def test_extract_answer_first_returns_an_earlier_value_on_a_drifting_node_list():
  text = (
      "Node 3 is connected to nodes 1, 2.\n"
      "Actually, node 3 is connected to nodes 1, 2, 3."
  )
  assert scoring.extract_answer(text, "connected_nodes") == "1, 2, 3"
  assert scoring.extract_answer_first(text, "connected_nodes") == "1, 2"


def test_extract_answer_first_on_empty_text_returns_none():
  assert scoring.extract_answer_first("", "node_count") is None
  assert scoring.extract_answer_first("   ", "node_count") is None


def test_extract_answer_first_raises_on_unknown_task():
  with pytest.raises(ValueError, match="unknown task"):
    scoring.extract_answer_first("Answer: 3.", "not_a_real_task")
