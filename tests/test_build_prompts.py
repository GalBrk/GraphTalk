"""`scripts/build_prompts.py::build_stratified` (Track 2.2): the size-based
selection logic is new, untested-elsewhere code that a future real sweep
will rely on to pick which graphs to spend GPU time on, so its ranking,
tie-breaking, and instance_id-namespacing get pinned here directly rather
than only checked ad hoc against live network data (as this feature
already was once, by hand, during development).

No network access: `load_rows` is monkeypatched to return synthetic rows
built from vendored-generator graphs, matching the project's existing
"tests use the vendored generator for graphs" convention (see
`tests/test_diverse_corpus.py`, `tests/test_primers.py`). The synthetic
`question` text only needs to satisfy `graphqa.parse_graph`'s two regexes
(a node-list header and an edge-list body) -- it need not resemble the
published dataset's full prose.
"""

import random

import networkx as nx
import pytest

from graphtalk import graphqa
from scripts import build_prompts


def _fake_question(graph: nx.Graph) -> str:
  """Minimal text `graphqa.parse_graph` can recover `graph` from."""
  nodes = ", ".join(str(n) for n in graph.nodes())
  edges = " ".join(f"({a}, {b})" for a, b in graph.edges())
  return f"There is a graph among nodes {nodes}.\nThe edges in G are:\n{edges}"


def _fake_row(graph: nx.Graph, task: str, rng: random.Random) -> dict:
  """A `load_rows`-shaped row for `task`, in the *pre-reword* wording
  `graphqa.reword_edge_existence` expects as input (mirrors the real
  published split, which `build_stratified` rewords after its own
  gold-answer cross-check -- same order as `build`/`build_named`)."""
  nodes = list(graph.nodes())
  if task == "node_count":
    task_description = "Q: How many nodes are in this graph?\nA: "
    targets = ()
  elif task == "edge_count":
    task_description = "Q: How many edges are in this graph?\nA: "
    targets = ()
  elif task == "cycle_check":
    task_description = "Q: Is there a cycle in this graph?\nA: "
    targets = ()
  elif task == "node_degree":
    node = rng.choice(nodes)
    task_description = f"Q: What is the degree of node {node}?\nA: "
    targets = (node,)
  elif task == "connected_nodes":
    node = rng.choice(nodes)
    task_description = f"Q: List all the nodes connected to {node} in alphabetical order.\nA: "
    targets = (node,)
  elif task == "edge_existence":
    source, target = rng.sample(nodes, 2)
    task_description = f"Q: Is node {source} connected to node {target}?\nA: "
    targets = (source, target)
  else:
    raise ValueError(f"unhandled task: {task}")
  return {
      "question": _fake_question(graph),
      "task_description": task_description,
      "answer": graphqa.gold_answer(graph, task, targets),
  }


def _graph_of_size(n: int, seed: int) -> nx.Graph:
  """A connected graph with exactly `n` nodes -- a path plus a handful of
  extra edges, so `cycle_check`/`connected_nodes` see nontrivial structure
  rather than a bare path every time."""
  graph = nx.path_graph(n)
  rng = random.Random(seed)
  extra = min(n, 3)
  nodes = list(graph.nodes())
  for _ in range(extra):
    if len(nodes) < 2:
      break
    a, b = rng.sample(nodes, 2)
    graph.add_edge(a, b)
  return graphqa.canonical(graph)


def _graphs_by_size(sizes: list[int]) -> list[nx.Graph]:
  """One graph per size in `sizes`, index order preserved -- the
  `(size, original_index)` tuple `build_stratified` sorts on. Shared
  across every task's stubbed `load_rows` call so a given original index
  always names the same graph regardless of which task is being built."""
  return [_graph_of_size(size, seed=index) for index, size in enumerate(sizes)]


@pytest.fixture(autouse=True)
def _stub_load_rows(monkeypatch):
  """Replaced per-test via `monkeypatch.setattr(build_prompts, "load_rows",
  ...)` in each test body; this fixture only exists so a test that forgets
  to stub it fails loudly (network) rather than hanging -- placeholder
  that always raises."""
  def _unstubbed(*_args, **_kwargs):
    raise AssertionError("load_rows was not stubbed for this test")
  monkeypatch.setattr(build_prompts, "load_rows", _unstubbed)


def _stub_pool(monkeypatch, sizes: list[int]):
  """Every task sees a pool over the *same* size-tagged graphs (so a given
  original index always means the same size no matter which task) --
  `build_stratified` calls `load_rows` once per task, and each call gets
  freshly built rows using `config` (the task name) for correct
  task_description/answer wording, the way real per-task published rows
  would differ from each other while still sharing question difficulty."""
  graphs = _graphs_by_size(sizes)
  calls = []

  def _fake_load_rows(config, count, split, cache):
    calls.append((config, count, split, cache))
    rng = random.Random(0)
    return [_fake_row(graph, config, rng) for graph in graphs]

  monkeypatch.setattr(build_prompts, "load_rows", _fake_load_rows)
  return graphs, calls


def test_selects_the_count_largest_graphs_by_node_count(monkeypatch):
  sizes = [5, 9, 3, 12, 7, 4, 20, 6]
  _stub_pool(monkeypatch, sizes)
  records = build_prompts.build_stratified(
      count=3, conditions=["none"], styles=["zero_shot"], split="x",
      cache="unused", k_min=2, k_max=3, pool_size=len(sizes),
  )
  node_count_records = [r for r in records if r["task"] == "node_count"]
  selected_sizes = sorted(r["nodes"] for r in node_count_records)
  # The 3 largest sizes in `sizes`: 20, 12, 9.
  assert selected_sizes == [9, 12, 20]


def test_ties_break_on_original_split_index(monkeypatch):
  sizes = [8, 8, 8, 5]
  _stub_pool(monkeypatch, sizes)
  records = build_prompts.build_stratified(
      count=2, conditions=["none"], styles=["zero_shot"], split="x",
      cache="unused", k_min=2, k_max=3, pool_size=len(sizes),
  )
  node_count_records = [r for r in records if r["task"] == "node_count"]
  selected_indices = sorted(
      int(r["instance_id"].rsplit("/", 1)[1]) for r in node_count_records
  )
  # All three size-8 rows tie; the two lowest original indices (0, 1) win
  # deterministically, not an arbitrary two of the three.
  assert selected_indices == [0, 1]


def test_instance_id_uses_the_stratified_namespace_and_original_index(monkeypatch):
  sizes = [5, 9, 3]
  _stub_pool(monkeypatch, sizes)
  records = build_prompts.build_stratified(
      count=1, conditions=["none"], styles=["zero_shot"], split="x",
      cache="unused", k_min=2, k_max=3, pool_size=len(sizes),
  )
  node_count_records = [r for r in records if r["task"] == "node_count"]
  (record,) = node_count_records
  # Index 1 (size 9) is the single largest -- confirms both the namespace
  # and that the *original* pool index survives selection, not a
  # re-numbered rank.
  assert record["instance_id"] == "node_count/stratified/1"


def test_load_rows_is_called_with_pool_size_not_count(monkeypatch):
  sizes = [5, 9, 3, 12, 7]
  _pool, calls = _stub_pool(monkeypatch, sizes)
  build_prompts.build_stratified(
      count=2, conditions=["none"], styles=["zero_shot"], split="a_split",
      cache="a_cache", k_min=2, k_max=3, pool_size=len(sizes),
  )
  assert len(calls) == len(build_prompts.scoring.TASKS)
  for _config, count, split, cache in calls:
    assert count == len(sizes)  # pool_size, not the requested count=2
    assert split == "a_split"
    assert cache == "a_cache"


def test_record_count_matches_count_times_conditions_times_styles_times_tasks(monkeypatch):
  sizes = list(range(3, 15))
  _stub_pool(monkeypatch, sizes)
  conditions = ["none", "degree"]
  styles = ["zero_shot"]
  count = 4
  records = build_prompts.build_stratified(
      count=count, conditions=conditions, styles=styles, split="x",
      cache="unused", k_min=2, k_max=3, pool_size=len(sizes),
  )
  expected = count * len(conditions) * len(styles) * len(build_prompts.scoring.TASKS)
  assert len(records) == expected


def test_selected_graphs_carry_correct_gold_answers(monkeypatch):
  sizes = [4, 10, 6]
  _stub_pool(monkeypatch, sizes)
  records = build_prompts.build_stratified(
      count=2, conditions=["none"], styles=["zero_shot"], split="x",
      cache="unused", k_min=2, k_max=3, pool_size=len(sizes),
  )
  for record in records:
    if record["task"] != "node_count":
      continue
    assert record["gold"] == str(record["nodes"])


def test_raises_when_shipped_answer_disagrees_with_the_parsed_graph(monkeypatch):
  bad_row = {
      "question": _fake_question(_graph_of_size(5, seed=0)),
      "task_description": "Q: How many nodes are in this graph?\nA: ",
      "answer": "999",  # Deliberately wrong -- doesn't match the parsed graph.
  }

  def _fake_load_rows(config, count, split, cache):
    return [bad_row]

  monkeypatch.setattr(build_prompts, "load_rows", _fake_load_rows)
  with pytest.raises(ValueError):
    build_prompts.build_stratified(
        count=1, conditions=["none"], styles=["zero_shot"], split="x",
        cache="unused", k_min=2, k_max=3, pool_size=1,
    )
