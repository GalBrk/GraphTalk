"""Pre-GPU screen: decide whether a `(n, mean_degree)` cell can show a primer
effect at all, before spending any generation on it.

Three independent ways a cell can be worthless, all computable from graph
samples alone:

  * **blind** -- `maj_base`, the frequency of the modal degree, is what a
    guesser scores on `node_degree` without reading anything. Random regular
    graphs are the extreme case: every answer is the same integer, so
    `maj_base` is 1.000 and the cell measures nothing. Anything above ~0.25 is
    a task a guesser wins too often for a primer effect to be legible.
  * **silent** -- `clu_sd`, the spread of the *rendered, 2-decimal* clustering
    values. This is literally the text the `clustering` primer emits. If it is
    ~0 the primer is a constant string across nodes, and no sample size can
    make a constant informative. Dense graphs fail here: clustering saturates
    toward a single value.
  * **unreadable** -- prompt tokens beyond the model's measured reading limit.
    A hard cell above that limit measures reading, not primers: at the density
    where the plain arm collapses, the `degree` control -- which writes the
    answer verbatim into the prompt -- was worth +0.7pp. A primer cannot beat a
    reading failure.

`clu_sd` is reported both raw and after degree-preserving rewiring, because
rewiring is part of the design rather than a post-hoc fix: it roughly triples
the rendered clustering spread at fixed `(n, m)` and is what makes the dense,
magnitude-limited cells -- the ones where primers can actually help -- usable.
Screening on the raw value alone rejects exactly the cells worth running.

Every statistic is averaged over several seeds and reported with its spread.
Single-seed screening flipped two cells across the `maj_base` bar during this
module's own development, so `n_graphs=1` is not offered.
"""

import statistics

import networkx as nx

from graphtalk import primers, rewiring

# A guesser scoring above this on `node_degree` leaves too little signal for a
# primer effect to be separable from the majority baseline.
MAJ_BASE_MAX = 0.25

# Below this, the rendered clustering primer is effectively one repeated
# sentence. The bar is on the *rendered* (2-dp) value, not the full-precision
# one, because the rendered text is what the model actually sees.
CLU_SD_MIN = 0.10


def maj_base(graph) -> float:
  """Fraction of nodes sharing the modal degree = the blind baseline."""
  degrees = [degree for _, degree in graph.degree()]
  if not degrees:
    return 1.0
  counts = {}
  for degree in degrees:
    counts[degree] = counts.get(degree, 0) + 1
  return max(counts.values()) / len(degrees)


def clu_sd(graph) -> float:
  """Spread of the rendered clustering values -- what the primer actually says.

  Rounded to 2 decimals first, matching `primers._fmt`, so a cell whose
  clustering varies only in the sixth decimal correctly screens as silent.
  """
  values = [round(value, 2) for value in primers.clustering(graph).values()]
  if len(values) < 2:
    return 0.0
  return statistics.pstdev(values)


def screen_cell(n: int, mean_degree: float, n_graphs: int = 8, seed: int = 0,
                token_counter=None) -> dict:
  """Screen one `(n, mean_degree)` cell over `n_graphs` samples.

  `token_counter` is an optional callable taking a graph and returning a prompt
  token count; it is injected rather than imported so this module stays free of
  the transformers dependency and can run anywhere.
  """
  if n_graphs < 2:
    raise ValueError("screening needs at least 2 graphs; single-seed screening "
                     "is unreliable near the thresholds")
  edges = int(round(mean_degree * n / 2))
  majs, raws, rewireds, tokens = [], [], [], []
  for index in range(n_graphs):
    graph = nx.gnm_random_graph(n, edges, seed=seed + index)
    majs.append(maj_base(graph))
    raws.append(clu_sd(graph))
    _, high = rewiring.rewired_pair(graph, seed=seed + index)
    rewireds.append(clu_sd(high))
    if token_counter is not None:
      tokens.append(token_counter(graph))

  result = {
      "n": n,
      "mean_degree": mean_degree,
      "edges": edges,
      "maj_base": statistics.fmean(majs),
      "maj_base_sd": statistics.pstdev(majs),
      "clu_sd_raw": statistics.fmean(raws),
      "clu_sd_rewired": statistics.fmean(rewireds),
      "clu_sd_rewired_sd": statistics.pstdev(rewireds),
      "tokens": statistics.fmean(tokens) if tokens else None,
  }
  result["blind"] = result["maj_base"] > MAJ_BASE_MAX
  result["silent"] = result["clu_sd_rewired"] < CLU_SD_MIN
  result["passes"] = not (result["blind"] or result["silent"])
  return result


def verdict(cell: dict, reading_limit: int | None = None) -> str:
  """One-word status, or the reasons a cell fails.

  `reading_limit` is per model and per arm -- there is no global value, which
  is why it is a parameter and not a constant. Omitting it screens structure
  only and says nothing about readability.
  """
  reasons = []
  if cell["blind"]:
    reasons.append("blind")
  if cell["silent"]:
    reasons.append("silent")
  if reading_limit is not None and cell.get("tokens") is not None:
    if cell["tokens"] > reading_limit:
      reasons.append("unreadable")
  return "pass" if not reasons else "fail:" + ",".join(reasons)
