"""Tests for `scripts/audit_extraction.py`.

Pins the finding step 0 of the paper-revision plan rests on: extraction never
silently reads a stale value from inside a `<think>` block on a row that
actually has a stated conclusion after it. `test_no_close_and_empty_after_close`
guard the two classify() buckets that mean "there is nothing to compare".
"""

import importlib.util
import pathlib

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "audit_extraction.py")
_spec = importlib.util.spec_from_file_location("audit_extraction", _SCRIPT)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def test_clean_response_has_no_risk_flag():
  text = "<think>\nmaybe 7? no, let me recount, 9.\n</think>\nThe degree of node 3 is **9**."
  assert audit.classify(text) == "clean"


def test_no_close_when_think_never_closes():
  text = "<think>\nstill reasoning about node 3, its neighbours are 1, 4"
  assert audit.classify(text) == "no_close"


def test_empty_after_close_when_generation_stops_at_the_tag():
  text = "<think>\nreasoning...\n</think>\n   "
  assert audit.classify(text) == "empty_after_close"


def test_full_text_and_post_close_extraction_agree_when_answer_is_after_close():
  # The extractor must land on the stated conclusion (9), not a stray digit
  # considered and discarded while thinking (7).
  import sys
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
  from graphtalk import scoring

  text = ("<think>\nmaybe 7? no wait, let me recount... actually 9.\n</think>\n"
          "The degree of node 3 is **9**.")
  close = text.rfind("</think>")
  after = text[close + len("</think>"):].strip()
  full = scoring.extract_answer(text, "node_degree")
  post = scoring.extract_answer(after, "node_degree")
  assert full == post == "9"
