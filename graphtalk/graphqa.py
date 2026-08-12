"""Shared GraphQA plumbing: row fetching, graph recovery, gold answers.

The published GraphQA dataset ships no edge lists -- every field is a string and
the graph exists only as rendered English inside the `question` field, which is
why `parse_graph` exists. Rows are fetched over the HTTP rows API rather than
the `datasets` library on purpose: `datasets` brings its own pyarrow/fsspec pins
and this venv holds a hand-tuned TensorFlow / tf-keras / tensorflow-gnn
combination that is easy to disturb.
"""

import json
import re
import urllib.parse
import urllib.request

import networkx as nx

ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET = "baharef/GraphQA"

_EDGE_RE = re.compile(r"\((\d+),\s*(\d+)\)")
_NODES_RE = re.compile(r"among nodes ([^.]+)\.")
_EDGES_HEADER = "The edges in G are:"


def canonical(graph: nx.Graph) -> nx.Graph:
  """Rebuilds so insertion order is sorted order, for nodes and adjacency.

  The vendored encoder iterates graph.nodes() and graph.neighbors() in insertion
  order, so identical graphs built differently render differently. Normalising
  here means the encoder inherits sorted order without being modified.
  """
  out = nx.Graph()
  out.add_nodes_from(sorted(graph.nodes()))
  out.add_edges_from(sorted(tuple(sorted(e)) for e in graph.edges()))
  return out


def parse_graph(question: str) -> nx.Graph:
  """Recovers a canonical networkx graph from a GraphQA question string.

  Nodes come from the header rather than the edge list, so isolated nodes
  survive. Integer labels are preserved, which the vendored encoders in
  talk_like_a_graph require. The result is passed through `canonical` so that
  re-encoding it is reproducible regardless of the order the prose listed
  things in.
  """
  header = _NODES_RE.search(question)
  if not header:
    raise ValueError("could not find the node list in the question")

  graph = nx.Graph()
  graph.add_nodes_from(int(tok) for tok in re.findall(r"\d+", header.group(1)))

  if _EDGES_HEADER in question:
    body = question.split(_EDGES_HEADER, 1)[1]
    graph.add_edges_from(
        (int(a), int(b)) for a, b in _EDGE_RE.findall(body)
    )
  return canonical(graph)


def _target_nodes(task_description: str) -> list[int]:
  return [int(tok) for tok in re.findall(r"\d+", task_description)]


def gold_answer(graph: nx.Graph, config: str, targets: tuple[int, ...] = ()) -> str:
  """Recomputes a task's ground-truth answer from the graph and its query nodes.

  Separate from `expected_answer` because the shortcut table samples its own
  queries against generated graphs and so has targets in hand, with no question
  string to recover them from.
  """
  if config == "node_degree":
    return str(graph.degree(targets[0]))
  if config == "node_count":
    return str(graph.number_of_nodes())
  if config == "edge_count":
    return str(graph.number_of_edges())
  if config == "connected_nodes":
    neighbors = sorted(graph.neighbors(targets[0]))
    # The dataset spells an isolated target as "No nodes." rather than "".
    return ", ".join(str(n) for n in neighbors) if neighbors else "No nodes"
  if config == "cycle_check":
    try:
      nx.find_cycle(graph)
      return "Yes, there is a cycle"
    except nx.NetworkXNoCycle:
      return "No, there is no cycle"
  if config == "edge_existence":
    a, b = targets[:2]
    return "Yes" if graph.has_edge(a, b) else "No"
  raise ValueError(f"no answer check defined for config: {config}")


def expected_answer(graph: nx.Graph, config: str, task_description: str) -> str:
  """Recomputes a task's ground-truth answer straight from the parsed graph."""
  return gold_answer(graph, config, tuple(_target_nodes(task_description)))


def normalize(answer: str) -> str:
  return answer.strip().rstrip(".").strip()


def fetch_rows(config: str, split: str, offset: int, length: int) -> list[dict]:
  query = urllib.parse.urlencode(
      {
          "dataset": DATASET,
          "config": config,
          "split": split,
          "offset": offset,
          "length": length,
      }
  )
  with urllib.request.urlopen(f"{ROWS_API}?{query}") as response:
    payload = json.load(response)
  return [r["row"] for r in payload["rows"]]
