# Vendored code

This directory is a copy of Google Research's reference implementation for
[Talk like a Graph: Encoding Graphs for Large Language Models](https://arxiv.org/abs/2310.04560)
and [Let Your Graph Do the Talking](https://arxiv.org/abs/2402.05862).

| | |
|---|---|
| Source | https://github.com/google-research/talk-like-a-graph |
| Commit | `36af51e19ef7a44049d64306e3cae56c07067e81` (2024-08-25) |
| Vendored on | 2026-08-08 |
| License | Apache 2.0, see `LICENSE` |

Upstream's `talk_like_a_graph/` package and `tutorial/` directory were copied
here verbatim. Any local modifications made after this point are ours and are
not reflected upstream.

## Local modifications

- `graph_generators_test.py`, `graph_text_encoders_test.py`: test classes
  declared `(absltest.TestCase, parameterized.TestCase)`, which is an invalid
  MRO because `parameterized.TestCase` already subclasses `absltest.TestCase`.
  Both now inherit from `parameterized.TestCase` alone. Without this the test
  files fail at collection with `TypeError: Cannot create a consistent method
  resolution order`.
- `graph_tasks.py`: `EdgeExistence`'s question text changed from "Is node A
  connected to node B?" to "Does an edge exist between Node A and Node B?" --
  the task is graded on a single edge (`graph.has_edge`), and "connected to"
  reads as general graph connectivity, which is a different and broader
  question. This class is not on `graphtalk`'s live prompt-building path (see
  `graphtalk/graphqa.py:reword_edge_existence`, which applies the same
  rewording to the published dataset's frozen `task_description` field); the
  edit here is for consistency with the vendored source, not functional.

Original disclaimer: this is not an official Google product.
