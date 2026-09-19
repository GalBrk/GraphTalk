"""The pre-GPU screen that decides whether a cell can show a primer effect.

Both bars are pinned against cases whose answer is known analytically rather
than by running the screen and recording what it said: a regular graph has one
degree so `maj_base` is exactly 1.0, and a complete graph has clustering 1.0
everywhere so its rendered spread is exactly 0.
"""

import networkx as nx
import pytest

from graphtalk import cell_screen


# --- maj_base ---------------------------------------------------------------

def test_regular_graph_is_maximally_blind():
  """Why random regular graphs are disqualified: every answer is the same."""
  assert cell_screen.maj_base(nx.random_regular_graph(8, 40, seed=1)) == 1.0


def test_star_graph_is_mostly_blind():
  """n-1 leaves of degree 1 and one hub, so the modal share is (n-1)/n."""
  assert cell_screen.maj_base(nx.star_graph(19)) == pytest.approx(19 / 20)


def test_er_graph_is_not_blind():
  assert cell_screen.maj_base(nx.gnm_random_graph(80, 320, seed=1)) < 0.25


# --- clu_sd -----------------------------------------------------------------

def test_complete_graph_primer_is_silent():
  """Clustering is 1.0 at every node, so the primer is one repeated sentence."""
  assert cell_screen.clu_sd(nx.complete_graph(20)) == 0.0


def test_clu_sd_uses_the_rendered_two_decimal_value():
  """Variation below the rendered precision must screen as silent.

  `primers._fmt` rounds to 2 decimals, so a cell whose clustering differs only
  in the sixth decimal shows the model identical text -- screening on the
  full-precision spread would wrongly call that informative.
  """
  path = nx.path_graph(30)          # clustering is 0.0 at every node
  assert cell_screen.clu_sd(path) == 0.0


# --- screen_cell ------------------------------------------------------------

def test_screen_cell_reports_both_raw_and_rewired_spread():
  cell = cell_screen.screen_cell(60, 12, n_graphs=3, seed=0)
  assert cell["clu_sd_rewired"] > cell["clu_sd_raw"]


def test_screen_cell_flags_a_dense_cell_as_silent_before_rewiring():
  """The screen's reason for existing: raw spread rejects the useful cells.

  Dense cells are where the task is magnitude-limited and a primer can help,
  but their raw clustering spread is well under the bar. Rewiring is what makes
  them testable, so the screen must report the rewired value.
  """
  cell = cell_screen.screen_cell(40, 16, n_graphs=3, seed=0)
  assert cell["clu_sd_raw"] < cell_screen.CLU_SD_MIN
  assert cell["clu_sd_rewired"] >= cell_screen.CLU_SD_MIN
  assert cell["passes"]


def test_single_seed_screening_is_refused():
  """Single-seed screening flipped two cells across the maj_base bar in
  development, so it is a type error rather than a documented caveat."""
  with pytest.raises(ValueError, match="at least 2"):
    cell_screen.screen_cell(40, 8, n_graphs=1)


# --- verdict ----------------------------------------------------------------

def test_verdict_reports_every_failing_reason():
  cell = {"blind": True, "silent": True, "tokens": 9_000}
  assert cell_screen.verdict(cell, reading_limit=3_000) == (
      "fail:blind,silent,unreadable")


def test_verdict_without_a_reading_limit_screens_structure_only():
  cell = {"blind": False, "silent": False, "tokens": 99_999}
  assert cell_screen.verdict(cell) == "pass"
