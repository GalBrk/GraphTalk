"""The shared ladder: its rungs, and the arithmetic that decides which are usable.

`RUNGS` carries token counts measured with the real tokenizer. They are pinned
the way `tests/golden/primers.json` pins rendered primer text -- an unintended
drift would silently reshape every downstream run -- but the structural screens
are re-derived here rather than copied, so this file cross-checks `cell_screen`
instead of restating it.
"""

import pytest

from graphtalk import cell_screen
from graphtalk import ladder


# --- rung table -------------------------------------------------------------

def test_rungs_are_sorted_by_token_count():
  tokens = [t for _, _, t in ladder.RUNGS]
  assert tokens == sorted(tokens)


def test_rungs_span_an_order_of_magnitude():
  """A ladder that does not span cannot place both a 0.6B and a 14B."""
  tokens = [t for _, _, t in ladder.RUNGS]
  assert tokens[-1] / tokens[0] > 8


def test_no_rung_uses_the_dead_n20_row():
  """n=20 screened blind at every degree (maj_base 0.287-0.325)."""
  assert all(n > 20 for n, _, _ in ladder.RUNGS)


@pytest.mark.parametrize("n,degree,tokens", ladder.RUNGS)
def test_every_rung_still_passes_the_structural_screen(n, degree, tokens):
  cell = cell_screen.screen_cell(n, degree, n_graphs=2, seed=11)
  assert cell["passes"], f"rung ({n}, {degree}) no longer passes: {cell}"


# --- context arithmetic -----------------------------------------------------

def test_context_headroom_goes_negative_when_a_prompt_would_overflow():
  """The guard that produced 35 mis-read `overflow` rows in an earlier sweep."""
  assert ladder.context_headroom(25_000, 32_768, 8_192) < 0
  assert ladder.context_headroom(10_000, 32_768, 8_192) > 0


def test_think_arm_fits_fewer_rungs_than_plain():
  """Think spends half the window on generation, so it reaches fewer rungs."""
  plain = ladder.feasible_rungs(32_768, 3_072)
  think = ladder.feasible_rungs(32_768, 16_384)
  assert len(think) < len(plain)


def test_primer_multiplier_is_applied():
  """The `all` primer roughly doubles prompt length and must be counted."""
  assert (len(ladder.feasible_rungs(32_768, 16_384, primer_multiplier=1.0))
          >= len(ladder.feasible_rungs(32_768, 16_384, primer_multiplier=2.0)))


# --- banding ----------------------------------------------------------------

@pytest.mark.parametrize("accuracy,expected", [
    (0.99, "ceiling"), (0.90, "ceiling"),
    (0.55, "informative"), (0.21, "informative"),
    (0.20, "floor"), (0.05, "floor"),
])
def test_classify_band(accuracy, expected):
  assert ladder.classify_band(accuracy) == expected


def test_model_band_separates_valid_from_unreadable():
  """An informative-but-unreadable rung measures reading, not primers.

  It is reported separately rather than dropped, because a model with many of
  them and no valid rung is a finding about the design's reach.
  """
  accuracies = {(n, d): 0.5 for n, d, _ in ladder.RUNGS}
  out = ladder.model_band(accuracies, reading_limit=3_000)
  assert out["valid"]
  assert out["informative_but_unreadable"]
  assert all(r["tokens"] <= 3_000 for r in out["valid"])
  assert all(r["tokens"] > 3_000 for r in out["informative_but_unreadable"])


def test_model_band_skips_rungs_with_no_measurement():
  out = ladder.model_band({}, reading_limit=None)
  assert out["rungs"] == []


def test_ceiling_rungs_are_never_valid():
  accuracies = {(n, d): 0.99 for n, d, _ in ladder.RUNGS}
  assert ladder.model_band(accuracies)["valid"] == []


def test_reading_zone_names_the_measured_brackets():
  assert ladder.reading_zone(1_000) == "clean"
  assert ladder.reading_zone(2_500) == "bracket-1"
  assert ladder.reading_zone(5_000) == "bracket-2"
  assert ladder.reading_zone(12_000) == "collapsed"
