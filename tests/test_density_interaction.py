"""Checks for scripts/density_interaction.py's statistics."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import density_interaction as di  # noqa: E402

P = np.repeat([0.1, 0.2, 0.35, 0.5], 100)


def test_between_ss_is_zero_for_equal_group_means_and_grows_with_spread():
  flat = np.tile([1.0, -1.0], 200)
  assert di.between_ss(flat, P) == pytest.approx(0.0)
  step = np.where(P > 0.3, 1.0, 0.0)
  assert di.between_ss(step, P) == pytest.approx(400 * 0.25)


def test_slope_is_per_tenth_of_density():
  assert di.slope(10 * P, P) == pytest.approx(1.0)       # x = 10p rises 1 per +0.1


def test_permutation_tests_find_a_real_density_effect_and_not_noise():
  rng = np.random.default_rng(1)
  noise = rng.choice([-100.0, 0.0, 100.0], 400)
  ph, ps = di.perm_pvalues(noise, P, draws=400)
  assert ph > 0.05 and ps > 0.05
  trend = np.where(rng.random(400) < 0.2 + P, 100.0, 0.0)
  ph, ps = di.perm_pvalues(trend, P, draws=400)
  assert ph < 0.01 and ps < 0.01


def test_density_coefficient_is_zero_when_the_band_explains_everything():
  band = np.array(["low"] * 4 + ["high"] * 4)
  p = np.array([0.1, 0.2, 0.35, 0.5] * 2)
  delta = np.where(band == "low", 10.0, -5.0)
  assert di.density_coefficient(delta, band, p) == pytest.approx(0.0, abs=1e-9)
  assert di.density_coefficient(delta + 20 * p, band, p) == pytest.approx(2.0)


def test_within_band_shuffle_detects_density_beyond_the_band():
  rng = np.random.default_rng(2)
  band = np.repeat(["a", "b", "c"], 40)
  p = np.tile(np.repeat([0.1, 0.2, 0.35, 0.5], 10), 3)
  base = {"a": 10.0, "b": 0.0, "c": -10.0}
  delta = np.array([base[b] for b in band]) + rng.normal(0, 1, 120)
  assert di.within_band_pvalue(delta, band, p, draws=300) > 0.05
  assert di.within_band_pvalue(delta + 40 * p, band, p, draws=300) < 0.01


def test_band_of_uses_the_repos_bands():
  lo, hi = di.apw.BANDS[0]
  assert di.band_of(lo) == f"{lo:.2f}-{min(hi, 1.0):.2f}"
  assert di.band_of(1.0).endswith("1.00")
