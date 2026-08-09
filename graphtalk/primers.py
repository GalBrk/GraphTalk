"""Primer statistics and the single renderer that turns them into English.

A primer is the project's independent variable: a short preamble of factual
sentences about each node's local structure, prepended before the graph encoding
and the question. Every condition comes out of `render_primer`, so the seven arms
differ in content and never in format -- which is what makes a difference between
them interpretable at all, given that Fatemi et al.'s central result is that
phrasing alone moves accuracy by tens of points.

Every function here is a pure function of the graph. Two things are load-bearing
for that: graphs arrive canonicalised (see `graphqa.canonical`), and every
rendered float goes through `_fmt`.

The pieces a primer is built from are called *parts*, not "components" --
connected components are one of the features and the collision would be
confusing in code.
"""

import networkx as nx
import numpy as np

# A degree-0 row of the adjacency matrix would divide by zero, so the divisor is
# clamped to this, which makes an isolated node's whole RWSE vector 0.0. That is
# a convention which deliberately departs from the definition: a walker on an
# isolated node cannot move, so it is trivially still at its starting node and a
# simulation of the definition returns 1.0. 0.0 is chosen because it reads as
# "no walk structure here". tests/test_primers.py asserts it as a convention
# rather than letting it fall out of the arithmetic.
_ISOLATED_DIVISOR = 1.0

# Parts in the order they are rendered: graph-level sentences first, then the
# node-level phrases inside each node's sentence. The order is fixed here rather
# than taken from the caller's `parts` argument, so a given set of parts always
# renders to exactly one string.
GRAPH_PARTS = ("components",)
NODE_PARTS = ("degree", "clustering", "rwse", "filler")
PARTS = GRAPH_PARTS + NODE_PARTS

# The seven experimental conditions, as part tuples. `all` deliberately excludes
# `components`: it is the union of the three per-node features, and components is
# its own arm because it costs one sentence rather than one per node.
CONDITIONS = {
    "none": (),
    "components": ("components",),
    "degree": ("degree",),
    "clustering": ("clustering",),
    "rwse": ("rwse",),
    "filler": ("filler",),
    "all": ("degree", "clustering", "rwse"),
}


def degrees(graph: nx.Graph) -> dict[int, int]:
  """Degree per node, in sorted node order."""
  return {node: int(graph.degree(node)) for node in sorted(graph.nodes())}


def clustering(graph: nx.Graph) -> dict[int, float]:
  """Local clustering coefficient per node, in sorted node order."""
  values = nx.clustering(graph)
  return {node: float(values[node]) for node in sorted(graph.nodes())}


def rwse(
    graph: nx.Graph, k_min: int = 2, k_max: int = 3
) -> dict[int, dict[int, float]]:
  """Random-walk structural encoding: the diagonal of P^k for P = D^-1 A.

  Returned as {node: {k: return probability}} so callers never index by
  position. k=1 is 0 for every node of every simple graph (A has a zero
  diagonal) and is outside the default range for that reason.

  Values are full precision; rounding happens only at render time.
  """
  if k_min < 1:
    raise ValueError(f"k_min must be at least 1, got {k_min}")
  if k_max < k_min:
    raise ValueError(f"k_max {k_max} is below k_min {k_min}")

  # Row i of the matrix is the i-th *inserted* node, not node i. Pass nodelist
  # explicitly and pair the diagonal back against the same list: getting this
  # wrong attaches every node's values to a different node and produces numbers
  # that are all individually plausible.
  nodes = sorted(graph.nodes())
  adjacency = nx.to_numpy_array(graph, nodelist=nodes)
  degree = adjacency.sum(axis=1)
  divisor = np.where(degree > 0.0, degree, _ISOLATED_DIVISOR)
  transition = adjacency / divisor[:, np.newaxis]

  out: dict[int, dict[int, float]] = {node: {} for node in nodes}
  walk = np.eye(len(nodes))
  for k in range(1, k_max + 1):
    # Accumulate as M @ P, never P @ M. Both satisfy "repeated matrix multiply"
    # and they disagree in the last bit, which used to be visible in the
    # rendered text; _fmt now absorbs that, and pinning the order here keeps the
    # two defences independent.
    walk = walk @ transition
    if k < k_min:
      continue
    diagonal = np.diagonal(walk)
    for index, node in enumerate(nodes):
      out[node][k] = float(diagonal[index])
  return out


def component_count(graph: nx.Graph) -> int:
  """Number of connected components.

  Isolated nodes each count as their own component, which is what networkx
  already does and what the circuit-rank identity m - n + c > 0 requires.
  """
  return nx.number_connected_components(graph)


def _fmt(value: float) -> str:
  """Formats to two decimals, stably across BLAS builds and multiply orders.

  Values sitting exactly on a two-decimal boundary are odd/200, whose reduced
  denominator is 8, 40 or 200. Only 8 is a power of two, so only those are
  exactly representable; the rest land one bit either side of the boundary
  depending on the summation order, which differs across BLAS builds.
  Pre-rounding to six decimals removes that: a two-decimal tie needs at most
  2**3 in its denominator and a six-decimal tie needs 2**7, so no value can be
  both, and the pre-round can never itself flip a digit.

  Note that _fmt(0.125) is "0.12", not "0.13" -- Python rounds exact halves to
  even. That is deterministic on every machine and is not a bug.
  """
  return format(round(float(value), 6), ".2f")


def _join(phrases: list[str]) -> str:
  """Joins phrases: `and` with no comma at two, Oxford comma at three or more."""
  if len(phrases) == 1:
    return phrases[0]
  if len(phrases) == 2:
    return "%s and %s" % (phrases[0], phrases[1])
  return "%s, and %s" % (", ".join(phrases[:-1]), phrases[-1])


def _degree_phrase(degree: int) -> str:
  return "degree %d" % degree


def _clustering_phrase(coefficient: float) -> str:
  return "clustering coefficient %s" % _fmt(coefficient)


def _rwse_phrase(returns: dict[int, float]) -> str:
  steps = [
      "%s after %d step%s" % (_fmt(returns[k]), k, "" if k == 1 else "s")
      for k in sorted(returns)
  ]
  return "return probability %s" % _join(steps)


def _filler_phrase(graph: nx.Graph) -> str:
  """The length control: true, and structurally vacuous.

  The encoding's own first line already ends `and <n-1>.`, so this introduces no
  numeral the `none` arm lacks, and it contradicts nothing in the encoding.
  """
  others = graph.number_of_nodes() - 1
  return "%d other node%s in this graph" % (others, "" if others == 1 else "s")


def _components_sentence(graph: nx.Graph) -> str:
  count = component_count(graph)
  return "This graph has %d connected component%s." % (
      count,
      "" if count == 1 else "s",
  )


def _pad(text: str, graph: nx.Graph, target_chars: int) -> str:
  """Extends a primer with inert filler sentences up to `target_chars`.

  This exists because `rwse` (905 chars) and `all` (1441) are longer than the
  `filler` control (507), so the control has to be paddable up past them. The
  padding repeats the length-control sentence in sorted node order, cycling if
  one pass is not enough, and so states no fact the `filler` part does not.

  It never truncates: a primer already at or over the target comes back
  unchanged, which is why achieved character counts must be reported rather than
  assumed.
  """
  nodes = sorted(graph.nodes())
  if not nodes:
    return text
  pieces = [text] if text else []
  index = 0
  while len(" ".join(pieces)) < target_chars:
    node = nodes[index % len(nodes)]
    pieces.append("Node %d has %s." % (node, _filler_phrase(graph)))
    index += 1
  return " ".join(pieces)


def render_primer(
    graph: nx.Graph,
    parts: tuple[str, ...] = (),
    k_min: int = 2,
    k_max: int = 3,
    target_chars: int | None = None,
) -> str:
  """Renders the primer text for a set of parts. The only renderer there is.

  Graph-level sentences come first, then one sentence per node in sorted node
  order, joining that node's requested phrases under a shared verb. Everything
  is joined with a single space.

  `parts=()` yields the empty string -- that is the `none` condition, not a
  special case, and it is never padded.
  """
  unknown = [part for part in parts if part not in PARTS]
  if unknown:
    raise ValueError(f"unknown primer part(s): {unknown}; known: {list(PARTS)}")

  selected = [part for part in PARTS if part in parts]
  if not selected:
    return ""

  sentences = []
  if "components" in selected:
    sentences.append(_components_sentence(graph))

  node_parts = [part for part in selected if part in NODE_PARTS]
  if node_parts:
    degree_by_node = degrees(graph) if "degree" in node_parts else {}
    clustering_by_node = clustering(graph) if "clustering" in node_parts else {}
    rwse_by_node = (
        rwse(graph, k_min=k_min, k_max=k_max) if "rwse" in node_parts else {}
    )
    for node in sorted(graph.nodes()):
      phrases = []
      for part in node_parts:
        if part == "degree":
          phrases.append(_degree_phrase(degree_by_node[node]))
        elif part == "clustering":
          phrases.append(_clustering_phrase(clustering_by_node[node]))
        elif part == "rwse":
          phrases.append(_rwse_phrase(rwse_by_node[node]))
        else:
          phrases.append(_filler_phrase(graph))
      sentences.append("Node %d has %s." % (node, _join(phrases)))

  text = " ".join(sentences)
  if target_chars is not None:
    text = _pad(text, graph, target_chars)
  return text


def build_primer(
    graph: nx.Graph,
    condition: str,
    k_min: int = 2,
    k_max: int = 3,
    target_chars: int | None = None,
) -> str:
  """Renders one of the seven named conditions."""
  if condition not in CONDITIONS:
    raise ValueError(
        f"unknown condition: {condition}; known: {list(CONDITIONS)}"
    )
  return render_primer(
      graph,
      CONDITIONS[condition],
      k_min=k_min,
      k_max=k_max,
      target_chars=target_chars,
  )


def rwse_degree_correlation(
    graph: nx.Graph, k_min: int = 2, k_max: int = 3
) -> dict[int, float | None]:
  """Pearson r between degree and RWSE, per k, on unrounded values.

  A descriptive statistic, not an implementation check: a low or negative r is
  also the symptom of a node-mapping bug in `rwse`, so the two are
  indistinguishable here. The ordering test in tests/test_primers.py does that
  job instead.

  Returns None for a k where either vector is constant -- which is every k odd
  on a triangle-free graph, 19% of generated graphs -- rather than a NaN. Any
  corpus-level summary must therefore name its aggregation (use the mean of
  per-graph r; pooling all nodes measures graph size instead of node degree) and
  note that different k are averaged over different populations.
  """
  nodes = sorted(graph.nodes())
  degree_vector = np.array([graph.degree(node) for node in nodes], dtype=float)
  table = rwse(graph, k_min=k_min, k_max=k_max)

  out: dict[int, float | None] = {}
  for k in range(k_min, k_max + 1):
    values = np.array([table[node][k] for node in nodes])
    if len(nodes) < 2 or degree_vector.std() == 0.0 or values.std() == 0.0:
      out[k] = None
    else:
      out[k] = float(np.corrcoef(degree_vector, values)[0, 1])
  return out
