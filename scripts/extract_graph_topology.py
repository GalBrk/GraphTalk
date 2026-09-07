"""Per-instance structural/topology features for the published GraphQA
zero_shot_test split, joinable onto any sweep frame via `instance_id`.

Motivation: `qwen3-8b`/`degree` (GOT) flipped from family-significant-only at
--count 30 to globally BH-significant at --count 500 (commit `30fa9ed`).
`scripts/check_old_vs_new_subsample.py` (run separately) already shows the
old-30 and new-470 slices have overlapping CIs and similar deltas -- pure
power, not a different effect at scale. This script supplies the other half
of that investigation: the actual per-graph structural features needed to
check whether the *old 30* instances were a compositionally atypical draw of
the fixed 500-graph population, and whether the `degree` primer's advantage
concentrates on particular structural features (see
`scripts/compare_old_vs_new_topology.py` and
`scripts/analyze_topology_drivers.py`, which consume this script's output).

Reuses `scripts/measure_real_rows.py`'s fetch/parse/cache plumbing
(`load_rows`, `parse_rows`) rather than duplicating it, and
`graphtalk.primers.degrees`/`component_count` for the two statistics that
module already exposes -- everything else here (density, tree/forest/
bipartite checks, triangle count, circuit rank) has no existing bundled
"graph-level stats" function anywhere in the repo (confirmed: `primers.py`
only exposes per-node/per-graph pieces used for primer rendering, and
`shortcuts.py`'s structural predicates are inlined closures over *parsed
primer text*, not the graph, so they aren't reusable here) and is computed
directly from the parsed `networkx.Graph` with plain networkx calls.

`verify_alignment` checks the one load-bearing assumption this script's
join depends on: that row i of every task config is the same graph
(established for the first `--rows` rows by `measure_real_rows.py`'s own
`verify_corpus`, re-checked here against a second config specifically
because this script is run against 500 rows, more than any prior run of
that check covered).

    PYTHONPATH=. .venv/Scripts/python.exe scripts/extract_graph_topology.py \
        --count 500 --out analysis/topology_features.csv
"""

import argparse

import networkx as nx
import numpy as np

from graphtalk import primers
from scripts import measure_real_rows as mrr

DEFAULT_SPLIT = "zero_shot_test"

# Matches talk_like_a_graph/graph_generators.py's _NUMBER_OF_NODES_RANGE
# exactly (np.arange(5, 10) / (10, 15) / (15, 20), i.e. inclusive 5-9/10-14/
# 15-19) -- the only quasi-categorical "graph type" label available, since
# the whole published split is single-algorithm Erdos-Renyi (no generator/
# algorithm field ships in the data; see measure_real_rows.py's docstring).
_SIZE_BUCKETS = (("small", 5, 9), ("medium", 10, 14), ("large", 15, 19))


def size_bucket(num_nodes: int) -> str | None:
  """The generator's size bucket for `num_nodes`, or `None` outside 5-19
  (every published-split graph falls in range; `None` only matters for
  hand-built test graphs smaller than the generator ever produces)."""
  for name, low, high in _SIZE_BUCKETS:
    if low <= num_nodes <= high:
      return name
  return None


def graph_topology(graph: nx.Graph) -> dict:
  """Structural features of one graph, as a flat dict of plain Python
  values (safe to write straight to CSV).

  Pure function of the graph -- no fetching, no I/O -- so it's testable on
  hand-built graphs without network access.
  """
  num_nodes = graph.number_of_nodes()
  num_edges = graph.number_of_edges()
  component_count = primers.component_count(graph)
  degree_values = list(primers.degrees(graph).values())
  clustering_values = list(primers.clustering(graph).values())
  triangle_count = sum(nx.triangles(graph).values()) // 3
  return {
      "num_nodes": num_nodes,
      "num_edges": num_edges,
      "density": float(nx.density(graph)),
      "degree_mean": float(np.mean(degree_values)),
      "degree_std": float(np.std(degree_values)),
      "degree_min": int(min(degree_values)),
      "degree_max": int(max(degree_values)),
      "component_count": component_count,
      # Circuit rank (cyclomatic number) m - n + c: 0 iff the graph is a
      # forest, the same identity `primers.component_count`'s docstring
      # cites and `shortcuts.py`'s inlined `_circuit_rank` rule tests.
      "circuit_rank": num_edges - num_nodes + component_count,
      "is_tree": bool(nx.is_tree(graph)),
      "is_forest": bool(nx.is_forest(graph)),
      "is_bipartite": bool(nx.is_bipartite(graph)),
      "has_isolated_node": min(degree_values) == 0,
      "triangle_count": triangle_count,
      "is_triangle_free": triangle_count == 0,
      "clustering_mean": float(np.mean(clustering_values)),
      "size_bucket": size_bucket(num_nodes),
  }


def verify_alignment(reference_graphs: list[nx.Graph], other_graphs: list[nx.Graph]) -> int:
  """How many indices `reference_graphs`/`other_graphs` disagree on
  (structurally), the same node/edge-set comparison
  `measure_real_rows.py::verify_corpus` uses for its own alignment check --
  re-run here because this script asks for 500 rows, more than any prior
  run of that check has covered."""
  mismatches = 0
  for a, b in zip(reference_graphs, other_graphs):
    if sorted(a.nodes()) != sorted(b.nodes()) or sorted(a.edges()) != sorted(b.edges()):
      mismatches += 1
  return mismatches


def extract(config: str, split: str, count: int, cache_dir: str) -> list[dict]:
  """One row per index 0..count-1: `graph_topology`'s features plus
  `index`. Does not include `task`/`instance_id` -- callers join this onto
  a sweep frame by extracting the numeric suffix of `instance_id`
  (`f"{task}/{index}"`, `build_prompts.py`'s own convention) since every
  task shares the same graph at a given index (see module docstring)."""
  rows = mrr.load_rows(config, split, count, cache_dir)
  graphs = mrr.parse_rows(rows)
  out = []
  for index, graph in enumerate(graphs):
    features = graph_topology(graph)
    features["index"] = index
    out.append(features)
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--split", default=DEFAULT_SPLIT)
  parser.add_argument("--count", type=int, default=500)
  parser.add_argument("--config", default="node_count",
                      help="task config to fetch graphs from -- any task "
                           "works, since every task shares the same graph "
                           "at a given row index")
  parser.add_argument("--verify-config", default="edge_count",
                      help="a second config to fetch and check for "
                           "structural alignment against --config, before "
                           "trusting the one-fetch-covers-every-task "
                           "assumption this script's join depends on; pass "
                           "'' to skip")
  parser.add_argument("--cache-dir", default=mrr.DEFAULT_CACHE)
  parser.add_argument("--out", default="analysis/topology_features.csv")
  args = parser.parse_args()

  rows = mrr.load_rows(args.config, args.split, args.count, args.cache_dir)
  graphs = mrr.parse_rows(rows)
  print(f"fetched/parsed {len(graphs)} graphs from {args.config}/{args.split}")

  if args.verify_config:
    other_rows = mrr.load_rows(args.verify_config, args.split, args.count, args.cache_dir)
    other_graphs = mrr.parse_rows(other_rows)
    mismatches = verify_alignment(graphs, other_graphs)
    print(f"alignment check against {args.verify_config}: "
          f"{mismatches}/{len(graphs)} indices structurally differ")
    if mismatches:
      raise SystemExit(
          f"{args.config} and {args.verify_config} disagree on "
          f"{mismatches} graphs -- the per-task-shares-one-graph assumption "
          f"this script's join depends on does not hold at --count "
          f"{args.count}; do not trust a downstream join on `index` alone."
      )

  records = []
  for index, graph in enumerate(graphs):
    features = graph_topology(graph)
    features["index"] = index
    records.append(features)

  fieldnames = ["index"] + [key for key in records[0] if key != "index"]
  import csv
  with open(args.out, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
  print(f"wrote {len(records)} rows to {args.out}")


if __name__ == "__main__":
  main()
