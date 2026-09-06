import numpy as np
from absl.testing import parameterized
from . import graph_generators
from absl.testing import absltest


class GraphGenerationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='er_undirected_1',
          algorithm='er',
          directed=False,
          k=1,
      ),
      dict(
          testcase_name='er_directed_1',
          algorithm='er',
          directed=True,
          k=1,
      ),
      dict(
          testcase_name='ba_undirected_5',
          algorithm='ba',
          directed=False,
          k=5,
      ),
      dict(
          testcase_name='ba_directed_5',
          algorithm='ba',
          directed=True,
          k=5,
      ),
  )
  def test_number_of_graphs(self, algorithm, directed, k):
    generated_graph = graph_generators.generate_graphs(k, algorithm, directed)
    self.assertLen(generated_graph, k)

  @parameterized.named_parameters(
      dict(
          testcase_name='er_undirected',
          algorithm='er',
          directed=False,
      ),
      dict(
          testcase_name='er_directed',
          algorithm='er',
          directed=True,
      ),
      dict(
          testcase_name='ba_undirected',
          algorithm='ba',
          directed=False,
      ),
      dict(
          testcase_name='ba_directed',
          algorithm='ba',
          directed=True,
      ),
      dict(
          testcase_name='sbm_undirected',
          algorithm='sbm',
          directed=False,
      ),
      dict(
          testcase_name='sbm_directed',
          algorithm='sbm',
          directed=True,
      ),
      dict(
          testcase_name='sfn_undirected',
          algorithm='sfn',
          directed=False,
      ),
      dict(
          testcase_name='sfn_directed',
          algorithm='sfn',
          directed=True,
      ),
      dict(
          testcase_name='complete_undirected',
          algorithm='complete',
          directed=False,
      ),
      dict(
          testcase_name='complete_directed',
          algorithm='complete',
          directed=True,
      ),
      dict(
          testcase_name='star_undirected',
          algorithm='star',
          directed=False,
      ),
      dict(
          testcase_name='star_directed',
          algorithm='star',
          directed=True,
      ),
      dict(
          testcase_name='path_undirected',
          algorithm='path',
          directed=False,
      ),
      dict(
          testcase_name='path_directed',
          algorithm='path',
          directed=True,
      ),
  )
  def test_directions(self, algorithm, directed):
    generated_graph = graph_generators.generate_graphs(1, algorithm, directed)
    self.assertEqual(generated_graph[0].is_directed(), directed)

  def test_node_size_ranges_override_produces_larger_graphs(self):
    xlarge_ranges = {'xlarge': np.arange(20, 40)}
    graphs = graph_generators.generate_graphs(
        20, 'er', False, node_size_ranges=xlarge_ranges
    )
    for graph in graphs:
      self.assertGreaterEqual(graph.number_of_nodes(), 20)
      self.assertLess(graph.number_of_nodes(), 40)

  def test_node_size_ranges_default_unchanged(self):
    # Same seed, same call shape as today: must match byte-for-byte, since
    # every existing caller omits this parameter and must see no change.
    with_default = graph_generators.generate_graphs(10, 'er', False, random_seed=7)
    explicit_none = graph_generators.generate_graphs(
        10, 'er', False, random_seed=7, node_size_ranges=None
    )
    self.assertEqual(
        [sorted(g.edges()) for g in with_default],
        [sorted(g.edges()) for g in explicit_none],
    )

  def test_node_size_ranges_override_supports_sbm(self):
    # sbm additionally indexes _NUMBER_OF_COMMUNITIES_RANGE by bucket name;
    # an unrecognized bucket name (e.g. "xlarge") must fall back rather than
    # KeyError.
    xlarge_ranges = {'xlarge': np.arange(20, 25)}
    graphs = graph_generators.generate_graphs(
        5, 'sbm', False, node_size_ranges=xlarge_ranges
    )
    self.assertLen(graphs, 5)


if __name__ == '__main__':
  absltest.main()
