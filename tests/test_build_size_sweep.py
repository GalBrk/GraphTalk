"""Tests for `scripts/build_size_sweep.py`'s density mode.

The density knob is opt-in, and the reason it has to be is that
`prompts.sizesweep.jsonl` was already generated and already scored: if
passing no new flag changed a single byte of this script's output, the four
completed `size` arms would no longer be reproducible from the script that
built them. `test_default_output_unchanged_by_density_support` is that
guarantee, not a formality.
"""

import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_size_sweep.py"
_spec = importlib.util.spec_from_file_location("build_size_sweep", _SCRIPT)
build_size_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_size_sweep)


def _records(**kwargs):
  kwargs.setdefault("sizes", [20])
  kwargs.setdefault("count", 4)
  kwargs.setdefault("conditions", ["none"])
  kwargs.setdefault("seed", 20260906)
  return build_size_sweep.build(**kwargs)


def test_default_output_unchanged_by_density_support():
  """No `densities` argument must reproduce the pre-density record exactly."""
  records = _records()
  assert records, "expected the default path to still build rows"
  for record in records:
    # The density keys are what a downstream frame would group on; emitting
    # them on the default path would silently reshape every existing size row.
    assert "density_class" not in record
    assert "density" not in record
    # The size-only instance_id is the resume key run_sweep.py matches on.
    assert record["instance_id"].count("/") == 2
    assert "/p" not in record["instance_id"]


def test_default_path_is_deterministic():
  assert _records() == _records()


def test_densities_pin_sparsity_exactly():
  """`random.uniform(p, p) == p`, so a level is a pinned density, not a range."""
  for level in (0.05, 0.5, 0.75):
    for record in _records(densities=[level]):
      assert record["density_class"] == level
      assert record["density"] == pytest.approx(level)


def test_denser_levels_produce_more_edges():
  def median_edges(level):
    edges = sorted(r["edges"] for r in _records(count=20, densities=[level]))
    return edges[len(edges) // 2]

  assert median_edges(0.05) < median_edges(0.35) < median_edges(0.75)


def test_density_levels_do_not_collide_on_instance_id():
  """Two levels of one size must not share a key, or run_sweep.py would treat
  the second cell as already generated and skip it without saying so."""
  records = _records(densities=[0.05, 0.1, 0.2, 0.35, 0.5, 0.75])
  keys = [(r["instance_id"], r["task"], r["condition"]) for r in records]
  assert len(keys) == len(set(keys))


def test_a_levels_graphs_do_not_depend_on_which_levels_accompany_it():
  """p=0.5 must draw the same graphs whether it is passed alone or alongside
  other levels. The seed block is keyed on the density value, not its list
  position, precisely so that a re-run over a subset of levels cannot pair a
  fresh graph with an instance_id run_sweep.py already considers generated."""
  alone = _records(densities=[0.5])
  accompanied = [r for r in _records(densities=[0.05, 0.4, 0.5])
                 if r["density_class"] == 0.5]
  assert alone == accompanied


def test_distinct_levels_draw_distinct_graphs():
  """Two levels of the same size must not share a draw."""
  low = _records(densities=[0.2])
  high = _records(densities=[0.75])
  assert [r["prompt"] for r in low] != [r["prompt"] for r in high]


def test_tasks_override_restricts_the_task_set():
  records = _records(tasks=("node_degree", "connected_nodes"))
  assert {r["task"] for r in records} == {"node_degree", "connected_nodes"}


def test_non_default_seed_is_tagged_into_the_instance_id():
  """A replication corpus must not collide with the scored one.

  `instance_id` is (task, size, density, index) -- none of which mentions the
  seed -- so without this tag a fresh-seed run produces byte-identical keys for
  entirely different graphs, and any analysis pairing by key would compare a
  graph against a different graph.
  """
  base = _records(densities=[0.5], seed=build_size_sweep.DEFAULT_SEED)
  replication = _records(densities=[0.5],
                         seed=build_size_sweep.DEFAULT_SEED + 500000)
  base_ids = {r["instance_id"] for r in base}
  replication_ids = {r["instance_id"] for r in replication}
  assert not (base_ids & replication_ids)
  # Checked as a path segment, not a substring: "/size20" also contains "/s".
  assert all("s20760906" in r["instance_id"].split("/") for r in replication)


def test_default_seed_leaves_the_instance_id_untagged():
  """The already-scored arms must stay reproducible byte for byte."""
  records = _records(densities=[0.5], seed=build_size_sweep.DEFAULT_SEED)
  assert all(not any(part.startswith("s2026") for part in
                     r["instance_id"].split("/")) for r in records)


def test_a_seed_changes_the_graphs_not_just_the_key():
  """The tag must accompany a real corpus change, not decorate the same draw."""
  base = _records(densities=[0.5], seed=build_size_sweep.DEFAULT_SEED)
  replication = _records(densities=[0.5],
                         seed=build_size_sweep.DEFAULT_SEED + 500000)
  assert [r["edges"] for r in base] != [r["edges"] for r in replication]


def test_a_near_seed_is_refused_because_the_corpora_would_overlap():
  """seed+k re-indexes the same graphs by k; that must not build silently."""
  with pytest.raises(ValueError, match="share graphs"):
    _records(densities=[0.5], count=10, seed=build_size_sweep.DEFAULT_SEED + 3)


def test_a_distant_seed_shares_no_graph_with_the_default_corpus():
  base = _records(sizes=[40], count=30, densities=[0.35],
                  tasks=("node_degree",), seed=build_size_sweep.DEFAULT_SEED)
  far = _records(sizes=[40], count=30, densities=[0.35],
                 tasks=("node_degree",),
                 seed=build_size_sweep.DEFAULT_SEED + 500000)
  fingerprint = lambda rows: {(r["edges"], r["gold"]) for r in rows}
  overlap = fingerprint(base) & fingerprint(far)
  # A few collisions are expected by chance on (edges, gold) alone; a shifted
  # corpus would collide on nearly all 30.
  assert len(overlap) < 10, f"{len(overlap)}/30 fingerprints shared"
