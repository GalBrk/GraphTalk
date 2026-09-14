"""Degree-preserving rewiring: move triangle structure while holding everything
else -- including the prompt -- byte-identical.

Why this exists. Every earlier attempt to vary graph structure varied it by
swapping the *generator* (Erdos-Renyi vs Barabasi-Albert vs regular vs ...),
which moves three things at once: the shape, the degree distribution, and
therefore both the `node_degree` gold answers and `maj_base` (what a blind
guesser scores). Measured across five families at matched `(n, m)`, `maj_base`
ranged 0.15 to 1.000 -- a 6x swing, larger than any primer effect the project
has recorded. A family contrast is therefore not a structure contrast.

A double-edge swap removes edges `(a,b)` and `(c,d)` and adds `(a,d)` and
`(c,b)`. Every one of the four endpoints keeps its degree, so:

  * the degree sequence is preserved *exactly*, hence every `node_degree` gold
    answer and `maj_base` are unchanged;
  * `n` and `m` are unchanged, and the `incident` encoder emits one line per
    node listing its neighbours, so the rendered prompt is the same length --
    in practice the same string length to the character;
  * what does move is the triangle count, which is exactly what the
    `clustering` primer reports.

That makes a *within-instance paired* design available: the same graph, rewired,
asked the same question, with the same answer. The only thing that differs
between the arms is the quantity the primer talks about.

Acceptance is on the LOCAL triangle delta, not a global recount. A global
`nx.triangles` per proposed swap is O(m*k) and made a 480-edge graph take
minutes; the common-neighbour count of the two removed pairs and the two added
pairs is O(deg) and gives the identical accept/reject decision. This is not an
optimisation detail -- with a global recount the swap budget has to be capped so
low that large graphs come out barely rewired, which is how an earlier screen
wrongly concluded that dense cells could not carry an informative primer.

Swap budget is a multiple of `m`, never a flat cap, for the same reason: the
work needed to restructure a graph scales with its edge count. Measured,
`DEFAULT_MULT = 3` is where clustering spread plateaus; x1 undershoots on large
graphs and x8+ buys nothing.
"""

import random

import networkx as nx

# Multiple of |E| to use as the swap budget. At x1 a 640-edge graph reaches a
# rendered-clustering spread of 0.130; at x3, 0.202; at x8 and x20, 0.205 and
# 0.211 -- i.e. the curve is flat past x3, so x3 is the cheapest point that is
# not undershooting.
DEFAULT_MULT = 3

# Proposals rejected for degeneracy (shared endpoints, or an edge that already
# exists) are cheap but not free, so the attempt loop needs a ceiling. 40x the
# target is generous: even on dense graphs the observed accept rate leaves this
# untouched, and it turns a pathological input into a return rather than a hang.
_MAX_ATTEMPT_MULT = 40


def rewire(graph, direction: str, mult: float = DEFAULT_MULT, seed=0):
  """Return a copy of `graph` with triangle structure pushed up or down.

  `direction` is "high" (accept swaps that do not lose triangles) or "low"
  (accept swaps that do not gain them). Ties are accepted in both directions:
  a strict inequality stalls almost immediately, because most swaps on a sparse
  graph are triangle-neutral.

  The degree sequence, node set and edge count of the result are guaranteed
  identical to the input's -- `assert_invariants` checks exactly that, and the
  test suite calls it on every generated pair.
  """
  if direction not in ("high", "low"):
    raise ValueError(f"direction must be 'high' or 'low', got {direction!r}")
  if graph.number_of_edges() < 2:
    return graph.copy()

  want_high = direction == "high"
  rng = random.Random(seed)
  adjacency = {node: set(graph[node]) for node in graph}
  edges = [tuple(sorted(edge)) for edge in graph.edges()]
  present = set(edges)

  target = int(mult * len(edges))
  accepted = 0
  for _ in range(target * _MAX_ATTEMPT_MULT):
    if accepted >= target:
      break
    i = rng.randrange(len(edges))
    j = rng.randrange(len(edges))
    if i == j:
      continue
    a, b = edges[i]
    c, d = edges[j]
    # Without this coin flip the swap always pairs the lower-numbered endpoints,
    # which biases which rewirings are reachable at all.
    if rng.random() < 0.5:
      c, d = d, c
    if len({a, b, c, d}) < 4:
      continue
    new_i = tuple(sorted((a, d)))
    new_j = tuple(sorted((c, b)))
    if new_i in present or new_j in present:
      continue

    # Local triangle delta: a triangle on edge (u,v) is a common neighbour of u
    # and v, so the change is (common neighbours gained) - (common neighbours
    # lost). The removals have to be applied before counting the additions,
    # or a path through one of the removed edges is double-counted.
    before = len(adjacency[a] & adjacency[b]) + len(adjacency[c] & adjacency[d])
    adjacency[a].discard(b)
    adjacency[b].discard(a)
    adjacency[c].discard(d)
    adjacency[d].discard(c)
    after = len(adjacency[a] & adjacency[d]) + len(adjacency[c] & adjacency[b])
    gain = after - before

    if gain >= 0 if want_high else gain <= 0:
      adjacency[a].add(d)
      adjacency[d].add(a)
      adjacency[c].add(b)
      adjacency[b].add(c)
      present.discard(tuple(sorted((a, b))))
      present.discard(tuple(sorted((c, d))))
      present.add(new_i)
      present.add(new_j)
      edges[i] = new_i
      edges[j] = new_j
      accepted += 1
    else:
      adjacency[a].add(b)
      adjacency[b].add(a)
      adjacency[c].add(d)
      adjacency[d].add(c)

  rewired = nx.Graph()
  rewired.add_nodes_from(graph.nodes())
  rewired.add_edges_from(present)
  return rewired


def assert_invariants(original, rewired) -> None:
  """Raise if `rewired` is not a legal degree-preserving rewiring of `original`.

  These four properties are the entire reason the design works, so they are
  asserted rather than assumed: if any one of them silently broke, the paired
  comparison would quietly become a comparison of two different tasks.
  """
  if sorted(original.nodes()) != sorted(rewired.nodes()):
    raise AssertionError("node set changed")
  if original.number_of_edges() != rewired.number_of_edges():
    raise AssertionError(
        f"edge count changed: {original.number_of_edges()} -> "
        f"{rewired.number_of_edges()}")
  before = sorted(degree for _, degree in original.degree())
  after = sorted(degree for _, degree in rewired.degree())
  if before != after:
    raise AssertionError("degree sequence changed")
  if any(u == v for u, v in rewired.edges()):
    raise AssertionError("self-loop introduced")


def rewired_pair(graph, mult: float = DEFAULT_MULT, seed=0):
  """The (low, high) triangle-structure pair for one base graph.

  Returned together because they are only meaningful as a pair -- the
  experiment compares a graph against itself at two levels of clustering, and
  an absolute triangle count means nothing on its own.
  """
  low = rewire(graph, "low", mult=mult, seed=seed)
  high = rewire(graph, "high", mult=mult, seed=seed + 1)
  assert_invariants(graph, low)
  assert_invariants(graph, high)
  return low, high
