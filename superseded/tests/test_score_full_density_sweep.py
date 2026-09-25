"""Tests for the MDE/CSV additions to `superseded/scripts/score_full_density_sweep.py`.

`mde_for_arms` pins that each (control, treatment) pair here is its own
cluster -- a full-task density sweep pair is one graph at one density,
contributing exactly one row, never repeated, so there is no natural
clustering unit the way six tasks sharing one graph gives the main sweep
one. `write_csv` pins the empty-input contract: a family that reduces to
nothing on this data is a valid, silent outcome, not an error.
"""

import csv
import importlib.util
import pathlib

from graphtalk import significance

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "score_full_density_sweep.py")
_spec = importlib.util.spec_from_file_location(
    "score_full_density_sweep", _SCRIPT)
score_full_density_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_full_density_sweep)

_FAST = {"n_replicates": 20, "n_perm": 50, "n_steps": 3}


def test_mde_for_arms_clusters_each_pair_independently():
  control = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
  treatment = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  result = score_full_density_sweep.mde_for_arms(
      control, treatment, seed=0, settings=_FAST)
  expected = significance.minimum_detectable_effect_clustered(
      control, treatment, list(range(len(control))), initial_hi=0.05,
      seed=0, **_FAST)
  assert result == expected


def test_mde_for_arms_reports_no_paired_rows_when_empty():
  result = score_full_density_sweep.mde_for_arms(
      [], [], seed=0, settings=_FAST)
  assert result["delta"] is None
  assert result["note"] == "no paired rows"


def test_write_csv_writes_header_and_rows(tmp_path):
  path = tmp_path / "out.csv"
  rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
  score_full_density_sweep.write_csv(str(path), rows)
  with open(path, newline="") as handle:
    read_back = list(csv.DictReader(handle))
  assert read_back == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


def test_write_csv_skips_empty_list(tmp_path):
  path = tmp_path / "out.csv"
  score_full_density_sweep.write_csv(str(path), [])
  assert not path.exists()
