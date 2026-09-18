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
