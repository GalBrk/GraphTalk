"""Tests for scripts/extract_graph_topology.py's pure feature computation.

New glue code with no prior regression coverage -- an off-by-one in circuit
rank or the wrong isolated-node convention would silently poison every
downstream comparison (scripts/compare_old_vs_new_topology.py,
scripts/analyze_topology_drivers.py) built on top of `graph_topology`'s
output, so each feature is checked against a hand-computed value on small,
fully-worked graphs rather than only smoke-tested end to end.
"""

import networkx as nx

from scripts import extract_graph_topology as egt


def _triangle():
  graph = nx.Graph()
  graph.add_nodes_from([0, 1, 2])
  graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
  return graph


def _path():
  graph = nx.Graph()
  graph.add_nodes_from([0, 1, 2])
  graph.add_edges_from([(0, 1), (1, 2)])
  return graph


def _disconnected_with_isolates():
  # {0, 1} an edge, {2} and {3} isolated -- 3 components, not 2, since each
  # isolated node is its own component (primers.component_count's own
  # documented convention).
  graph = nx.Graph()
  graph.add_nodes_from([0, 1, 2, 3])
  graph.add_edges_from([(0, 1)])
  return graph


def test_triangle():
  features = egt.graph_topology(_triangle())
  assert features["num_nodes"] == 3
  assert features["num_edges"] == 3
  assert features["density"] == 1.0
  assert features["degree_mean"] == 2.0
  assert features["degree_min"] == 2
  assert features["degree_max"] == 2
  assert features["component_count"] == 1
  assert features["circuit_rank"] == 1  # m - n + c = 3 - 3 + 1
  assert features["is_tree"] is False
  assert features["is_forest"] is False
  assert features["is_bipartite"] is False  # odd cycle
  assert features["has_isolated_node"] is False
  assert features["triangle_count"] == 1
  assert features["is_triangle_free"] is False
  assert features["clustering_mean"] == 1.0
  assert features["size_bucket"] is None  # below the generator's 5-19 range


def test_path():
  features = egt.graph_topology(_path())
  assert features["num_edges"] == 2
  assert features["component_count"] == 1
  assert features["circuit_rank"] == 0  # m - n + c = 2 - 3 + 1
  assert features["is_tree"] is True
  assert features["is_forest"] is True
  assert features["is_bipartite"] is True
  assert features["has_isolated_node"] is False
  assert features["triangle_count"] == 0
  assert features["is_triangle_free"] is True
  assert features["clustering_mean"] == 0.0


def test_disconnected_with_isolates():
  features = egt.graph_topology(_disconnected_with_isolates())
  assert features["component_count"] == 3
  assert features["circuit_rank"] == 0  # m - n + c = 1 - 4 + 3
  assert features["is_tree"] is False  # more than one component
  assert features["is_forest"] is True
  assert features["is_bipartite"] is True
  assert features["has_isolated_node"] is True
  assert features["triangle_count"] == 0
  assert features["degree_min"] == 0


def test_size_bucket_boundaries():
  assert egt.size_bucket(4) is None
  assert egt.size_bucket(5) == "small"
  assert egt.size_bucket(9) == "small"
  assert egt.size_bucket(10) == "medium"
  assert egt.size_bucket(14) == "medium"
  assert egt.size_bucket(15) == "large"
  assert egt.size_bucket(19) == "large"
  assert egt.size_bucket(20) is None


def test_verify_alignment_counts_structural_mismatches():
  reference = [_triangle(), _path()]
  identical = [_triangle(), _path()]
  different = [_triangle(), _triangle()]
  assert egt.verify_alignment(reference, identical) == 0
  assert egt.verify_alignment(reference, different) == 1
