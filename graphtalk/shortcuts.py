"""Primer-only solvers: what is answerable by reading the primer and nothing else.

A *shortcut score* is what a small deterministic program scores when it reads the
rendered primer text and never sees the graph, judged by the same scorer used on
model output. That number is the bar the hypothesis has to clear: a model landing
at or below it did the primer arithmetic and nothing more, and only a model above
it combined the primer with the encoding.

This module starts at the bottom, with the parser. Everything else here will take
a primer *string* rather than statistics, which buys two guarantees structurally
rather than by discipline:

  * a rule can only use the two-decimal values the model actually sees, so it
    cannot beat the model by reading full precision;
  * a rule cannot touch the graph, because there is no graph to touch.

The parser's second job is a round-trip check on the renderer: render, parse, and
the recovered values must equal the rounded originals exactly. That test checks
`primers.render_primer` and `parse_primer` against each other, and it is the only
check that would catch a renderer change nobody meant to make. For it to be a
cross-check rather than a tautology, **the parser and the rules call nothing in
`primers`** -- the join rule is restated here in `_expected_separators`. The
evaluation harness at the foot of this file does render primers, which is the
point of it; nothing above that section does.

Parsing is deliberately **strict**. Every sentence must be fully accounted for,
and a conflicting repeat of a value raises. A parser that silently skipped text
it did not recognise would let a renderer change through and quietly shrink the
bar, which is the one direction of error that reverses the sign of the finding.

One consequence worth stating, because it partly collapses the rung ladder: a
node-level primer emits exactly one sentence per node, including isolated ones,
so counting sentences recovers n from the primer alone. `ParsedPrimer.nodes` is
therefore rung-1 information for every condition except `none` and an unpadded
`components`.
"""

import collections
import dataclasses
import functools
import itertools
import random
import re

import networkx as nx
import numpy as np

from graphtalk import graphqa
from graphtalk import primers

# A sentence ends at a period followed by whitespace or by end of string. Decimal
# points are always followed by a digit, so they are never a split point.
_SENTENCE_SPLIT = re.compile(r"(?<=\.)\s+")

_COMPONENTS_RE = re.compile(r"This graph has (\d+) connected components?\.$")
_NODE_RE = re.compile(r"Node (\d+) has (.+)\.$")

# Each node-level phrase has a distinctive opening, and appears at most once in a
# sentence. Locating those openings partitions the sentence body without having
# to disambiguate the joins: the RWSE phrase contains its own " and ", so
# splitting on separators first would be ambiguous, while no separator can be
# mistaken for the start of a phrase.
_PHRASE_STARTS = (
    ("degree", re.compile(r"degree \d")),
    ("clustering", re.compile(r"clustering coefficient \d")),
    ("rwse", re.compile(r"return probability \d")),
    ("filler", re.compile(r"\d+ other nodes? in this graph")),
)

_DEGREE_RE = re.compile(r"degree (\d+)$")
_CLUSTERING_RE = re.compile(r"clustering coefficient (\d+\.\d{2})$")
_RWSE_RE = re.compile(r"return probability (.+)$")
_RWSE_STEP_RE = re.compile(r"(\d+\.\d{2}) after (\d+) steps?$")
_FILLER_RE = re.compile(r"(\d+) other nodes? in this graph$")

# Longest first: ", and " also ends with " and ", so testing the short one first
# would strip five characters and leave a stray comma behind.
_SEPARATORS = (", and ", " and ", ", ")


@dataclasses.dataclass(frozen=True)
class ParsedPrimer:
  """Everything recoverable from rendered primer text, and nothing else.

  Floats are the two-decimal values as rendered, not the full-precision
  statistics they came from. That is the point: a rule reading more precision
  than the prompt shows would measure a bar nobody was ever given.

  `nodes` lists the nodes that received a sentence, which for a node-level primer
  is every node of the graph -- including isolated ones the encoding body omits.
  """

  components: int | None
  degree: dict[int, int]
  clustering: dict[int, float]
  rwse: dict[int, dict[int, float]]
  filler: dict[int, int]
  nodes: tuple[int, ...]

  def is_empty(self) -> bool:
    """True for the `none` condition, whose primer is the empty string."""
    return self.components is None and not self.nodes


def _check_plural(count: int, noun: str, text: str) -> None:
  """Rejects `1 components` and `2 component`.

  The parser is where format truth is centralised, so agreement is checked on the
  way back in rather than trusted. A grammatical slip in a condition that appears
  in every prompt of its arm is exactly the kind of thing that could surface as a
  spurious effect.
  """
  match = re.search(rf"\b{count} {re.escape(noun)}(s?)\b", text)
  if not match:
    raise ValueError(f"expected {count} to be followed by {noun!r}: {text!r}")
  # Substring testing would not do: "step" is a prefix of "steps", so a singular
  # check would pass on "1 steps".
  if (match.group(1) == "s") != (count != 1):
    raise ValueError(f"wrong plural for {count} {noun!r}: {text!r}")


def _expected_separators(count: int) -> list[str]:
  """The separator sequence the renderer's join rule produces for `count` items.

  Restated here rather than imported from `primers`, deliberately. A parser that
  validated the join by calling the renderer's own `_join` would agree with it
  under any mutation, and the round-trip test would pass while both sides drifted
  together. Nothing in this module shares code with the renderer, which is what
  makes the round trip a cross-check rather than a tautology.
  """
  if count <= 1:
    return []
  if count == 2:
    return [" and "]
  return [", "] * (count - 2) + [", and "]


def _check_separators(found: list[str], count: int, context: str) -> None:
  expected = _expected_separators(count)
  if found != expected:
    raise ValueError(
        f"joins {found!r} do not match the rule {expected!r} for {count} "
        f"items: {context!r}"
    )


def _strip_separator(segment: str) -> tuple[str, str]:
  for separator in _SEPARATORS:
    if segment.endswith(separator):
      return segment[: -len(separator)], separator
  raise ValueError(f"phrase does not end in a known separator: {segment!r}")


def _split_phrases(body: str) -> list[tuple[str, str]]:
  """Cuts a node sentence body into (part name, phrase) pairs, in render order."""
  marks = []
  for name, pattern in _PHRASE_STARTS:
    match = pattern.search(body)
    if match:
      marks.append((match.start(), name))
  marks.sort()

  if not marks:
    raise ValueError(f"no known phrase in node sentence: {body!r}")
  if marks[0][0] != 0:
    raise ValueError(f"unrecognised text before the first phrase: {body!r}")

  out = []
  separators = []
  for index, (start, name) in enumerate(marks):
    last = index + 1 == len(marks)
    end = len(body) if last else marks[index + 1][0]
    segment = body[start:end]
    if last:
      out.append((name, segment))
    else:
      phrase, separator = _strip_separator(segment)
      out.append((name, phrase))
      separators.append(separator)

  _check_separators(separators, len(out), body)
  return out


def _parse_rwse(phrase: str) -> dict[int, float]:
  """Reads `return probability 0.27 after 2 steps and 0.15 after 3 steps`."""
  match = _RWSE_RE.fullmatch(phrase)
  if not match:
    raise ValueError(f"malformed RWSE phrase: {phrase!r}")

  content = match.group(1)
  pieces, separators = _split_steps(content)
  _check_separators(separators, len(pieces), content)

  steps: dict[int, float] = {}
  for piece in pieces:
    step_match = _RWSE_STEP_RE.fullmatch(piece)
    if not step_match:
      raise ValueError(f"malformed RWSE step: {piece!r}")
    value, k = step_match.group(1), int(step_match.group(2))
    if k in steps:
      raise ValueError(f"step {k} appears twice in {phrase!r}")
    _check_plural(k, "step", piece)
    steps[k] = float(value)
  return steps


def _split_steps(content: str) -> tuple[list[str], list[str]]:
  """Splits the RWSE step list, returning the pieces and the joins between them.

  The step list uses the same join rule as the outer sentence, so the RWSE phrase
  can contain both a comma and an "and" of its own. Splitting only where a
  separator is followed by the start of another step keeps that unambiguous.
  """
  pieces = []
  separators = []
  rest = content
  while True:
    match = re.search(r"(, and | and |, )(?=\d+\.\d{2} after )", rest)
    if not match:
      pieces.append(rest)
      return pieces, separators
    pieces.append(rest[: match.start()])
    separators.append(match.group(1))
    rest = rest[match.end() :]


def _merge(store: dict, node: int, value, part: str) -> None:
  """Records a value, rejecting a repeat that disagrees.

  Repeats are expected: `primers._pad` appends filler sentences for nodes that
  already have one, so the same node legitimately appears more than once. A
  repeat with a *different* value is a bug, not padding.
  """
  if node in store and store[node] != value:
    raise ValueError(
        f"node {node} has conflicting {part}: {store[node]!r} then {value!r}"
    )
  store[node] = value


def parse_primer(text: str) -> ParsedPrimer:
  """Recovers the stated facts from rendered primer text.

  Raises ValueError on anything it cannot fully account for. The empty string is
  the `none` condition and parses to an empty result.
  """
  components: int | None = None
  degree: dict[int, int] = {}
  clustering: dict[int, float] = {}
  rwse: dict[int, dict[int, float]] = {}
  filler: dict[int, int] = {}
  nodes: list[int] = []

  stripped = text.strip()
  if not stripped:
    return ParsedPrimer(None, degree, clustering, rwse, filler, ())

  for sentence in _SENTENCE_SPLIT.split(stripped):
    if not sentence:
      continue

    graph_level = _COMPONENTS_RE.fullmatch(sentence)
    if graph_level:
      count = int(graph_level.group(1))
      _check_plural(count, "connected component", sentence)
      if components is not None and components != count:
        raise ValueError(f"conflicting component counts: {components} and {count}")
      components = count
      continue

    node_level = _NODE_RE.fullmatch(sentence)
    if not node_level:
      raise ValueError(f"unrecognised primer sentence: {sentence!r}")

    node = int(node_level.group(1))
    if node not in nodes:
      nodes.append(node)

    for name, phrase in _split_phrases(node_level.group(2)):
      if name == "degree":
        match = _DEGREE_RE.fullmatch(phrase)
        if not match:
          raise ValueError(f"malformed degree phrase: {phrase!r}")
        _merge(degree, node, int(match.group(1)), "degree")
      elif name == "clustering":
        match = _CLUSTERING_RE.fullmatch(phrase)
        if not match:
          raise ValueError(f"malformed clustering phrase: {phrase!r}")
        _merge(clustering, node, float(match.group(1)), "clustering")
      elif name == "rwse":
        _merge(rwse, node, _parse_rwse(phrase), "rwse")
      else:
        match = _FILLER_RE.fullmatch(phrase)
        if not match:
          raise ValueError(f"malformed filler phrase: {phrase!r}")
        others = int(match.group(1))
        _check_plural(others, "other node", phrase)
        _merge(filler, node, others, "filler")

  return ParsedPrimer(
      components, degree, clustering, rwse, filler, tuple(sorted(nodes))
  )


# --- tasks, queries, gold answers ----------------------------------------

TASKS = (
    "node_count",
    "edge_count",
    "node_degree",
    "connected_nodes",
    "edge_existence",
    "cycle_check",
)

# How many query nodes each task samples. The vendored generator draws a uniform
# pair for `edge_existence` (graph_tasks.py:136) and one uniform node for
# `connected_nodes` (:393) and `node_degree` (:264); the other three emit one row
# per graph with no query at all. Any rate quoted per row has to say which of
# these it used, because they count different things.
QUERY_ARITY = {
    "node_count": 0,
    "edge_count": 0,
    "cycle_check": 0,
    "node_degree": 1,
    "connected_nodes": 1,
    "edge_existence": 2,
}


def sample_query(graph, task: str, rng) -> tuple[int, ...]:
  """Draws a task's query nodes, without replacement, in the vendored style."""
  arity = QUERY_ARITY[task]
  if arity == 0:
    return ()
  return tuple(rng.sample(sorted(graph.nodes()), k=arity))


# --- what a solver is allowed to see -------------------------------------


@dataclasses.dataclass(frozen=True)
class Context:
  """Everything a primer-only solver may read.

  There is deliberately no graph field. That is the guarantee the whole design
  rests on, and making it structural means it cannot be lost by forgetting: a
  rule cannot consult the graph because the graph is not here to consult.

  `n` and `m` are what the rung grants from the encoding, None meaning withheld.
  Rules that are entitled to a granted value read `known_n` / `known_m`; rules
  measuring what the *primer* contributes read `n_from_sentences` /
  `m_from_degrees`, which never fall back to a grant.
  """

  primer: ParsedPrimer
  targets: tuple[int, ...] = ()
  n: int | None = None
  m: int | None = None

  @property
  def n_from_sentences(self) -> int | None:
    """n recovered from the primer alone, by counting node sentences.

    The renderer emits exactly one sentence per node, including the isolated ones
    the encoding body omits, so this is exact for every node-level condition --
    which is why the rung ladder does not separate on those arms. See the rung
    section of docs/plans/shortcut-ceilings.md.

    It assumes the renderer's one-sentence-per-node contract, which `_pad` breaks
    for a *graph-level* primer padded with per-node filler. The table is computed
    on unpadded primers for that reason.
    """
    return len(self.primer.nodes) or None

  @property
  def m_from_degrees(self) -> int | None:
    """m recovered from the primer alone, as sum(degrees) / 2."""
    if self.primer.degree and set(self.primer.degree) == set(self.primer.nodes):
      return sum(self.primer.degree.values()) // 2
    return None

  @property
  def known_n(self) -> int | None:
    return self.n if self.n is not None else self.n_from_sentences

  @property
  def known_m(self) -> int | None:
    return self.m if self.m is not None else self.m_from_degrees


@dataclasses.dataclass(frozen=True)
class Rule:
  """A named deterministic route from primer text to an answer.

  `solve` returns an answer string, or None to abstain. Abstention is how a rule
  says the primer it was handed does not carry what it needs -- which is also how
  a rule confines itself to the conditions it applies to, without anyone having
  to maintain a list of them.
  """

  name: str
  task: str
  kind: str
  why: str
  solve: object


# --- theorems -------------------------------------------------------------
#
# Every rule below is exact: when it fires, it is right, and the phase-2 test
# asserts precision is 1.0 over a corpus rather than taking that on trust. What
# varies between them is coverage.
#
# Rounding cuts one way only. A statistic that is exactly zero renders as "0.00",
# so a rendered positive can never come from a true zero and the "> 0" tests keep
# their precision. A true positive small enough to round to "0.00" costs coverage
# instead, which is the harmless direction.


def _all_zero_rwse(ctx: Context, node: int) -> bool | None:
  """Whether the primer's RWSE vector for `node` is all zeros.

  Equivalent to degree 0 under the clamp convention, but only when k=2 is in the
  rendered range: the k=2 return probability of a node with a neighbour is at
  least 1/(n-1), which is 0.06 at n=19 and so can never round to 0.00, while odd
  k alone is exactly 0 for every node of a triangle-free graph -- 19% of them.
  """
  table = ctx.primer.rwse.get(node)
  if not table or 2 not in table:
    return None
  return all(value == 0.0 for value in table.values())


def _degree_zero_no_edge(ctx):
  a, b = ctx.targets
  if ctx.primer.degree.get(a) == 0 or ctx.primer.degree.get(b) == 0:
    return "No"
  return None


def _degree_full_edge(ctx):
  n = ctx.known_n
  if n is None:
    return None
  # Query nodes are drawn without replacement, so a node of degree n-1 is
  # adjacent to every other node including its partner.
  if any(ctx.primer.degree.get(t) == n - 1 for t in ctx.targets):
    return "Yes"
  return None


def _rwse_zero_no_edge(ctx):
  if any(_all_zero_rwse(ctx, t) for t in ctx.targets):
    return "No"
  return None


def _degree_zero_no_nodes(ctx):
  if ctx.primer.degree.get(ctx.targets[0]) == 0:
    return "No nodes"
  return None


def _rwse_zero_no_nodes(ctx):
  if _all_zero_rwse(ctx, ctx.targets[0]):
    return "No nodes"
  return None


def _clustering_triangle(ctx):
  if any(value > 0.0 for value in ctx.primer.clustering.values()):
    return "Yes, there is a cycle"
  return None


def _rwse_triangle(ctx):
  if any(table.get(3, 0.0) > 0.0 for table in ctx.primer.rwse.values()):
    return "Yes, there is a cycle"
  return None


def _circuit_rank(ctx):
  c, n, m = ctx.primer.components, ctx.known_n, ctx.known_m
  if c is None or n is None or m is None:
    return None
  # Exact in both directions, which makes this the only cycle_check theorem that
  # can answer No. It is the entire justification for the components condition.
  return "Yes, there is a cycle" if m - n + c > 0 else "No, there is no cycle"


def _edges_at_least_nodes(ctx):
  n, m = ctx.known_n, ctx.known_m
  if n is None or m is None or m < n:
    return None
  # A forest has at most n-1 edges, so m >= n cannot be a forest.
  return "Yes, there is a cycle"


def _count_sentences(ctx):
  n = ctx.n_from_sentences
  return None if n is None else str(n)


def _components_edgeless(ctx):
  c, m = ctx.primer.components, ctx.known_m
  if c is None or m is None or m != 0:
    return None
  # c = n exactly when m = 0. Dominated by simply knowing n, so it only ever
  # fires at a rung that already grants n -- kept because the *information* is
  # what the components arm's node_count caveat is about.
  return str(c)


def _degree_sum(ctx):
  m = ctx.m_from_degrees
  return None if m is None else str(m)


def _stated_degree(ctx):
  degree = ctx.primer.degree.get(ctx.targets[0])
  return None if degree is None else str(degree)


THEOREMS = (
    Rule("degree_zero_no_edge", "edge_existence", "theorem",
         "degree 0 forces No", _degree_zero_no_edge),
    Rule("degree_full_edge", "edge_existence", "theorem",
         "degree n-1 forces Yes", _degree_full_edge),
    Rule("rwse_zero_no_edge", "edge_existence", "theorem",
         "all-zero RWSE means degree 0", _rwse_zero_no_edge),
    Rule("degree_zero_no_nodes", "connected_nodes", "theorem",
         "degree 0 means No nodes", _degree_zero_no_nodes),
    Rule("rwse_zero_no_nodes", "connected_nodes", "theorem",
         "all-zero RWSE means No nodes", _rwse_zero_no_nodes),
    Rule("clustering_triangle", "cycle_check", "theorem",
         "clustering > 0 means a triangle", _clustering_triangle),
    Rule("rwse_triangle", "cycle_check", "theorem",
         "RWSE(k=3) > 0 means a triangle", _rwse_triangle),
    Rule("circuit_rank", "cycle_check", "theorem",
         "m - n + c > 0 iff a cycle", _circuit_rank),
    Rule("edges_at_least_nodes", "cycle_check", "theorem",
         "m >= n cannot be a forest", _edges_at_least_nodes),
    Rule("count_sentences", "node_count", "theorem",
         "one sentence per node", _count_sentences),
    Rule("components_edgeless", "node_count", "theorem",
         "c = n when m = 0", _components_edgeless),
    Rule("degree_sum", "edge_count", "theorem",
         "sum(degrees) / 2 = m", _degree_sum),
    Rule("stated_degree", "node_degree", "theorem",
         "the primer states the degree", _stated_degree),
)


# --- evaluation harness ---------------------------------------------------
#
# Everything below renders primers, which is why it sits under the rules rather
# than among them: the parser and the rules above import nothing from `primers`,
# and the round trip is only a cross-check for as long as that stays true.

RUNGS = (1, 2, 3)


def make_context(
    graph, condition: str, targets: tuple[int, ...], rung: int,
    k_min: int = 2, k_max: int = 3,
) -> Context:
  """Renders a primer, parses it back, and grants what the rung allows.

  The grants come off the graph because they are facts the *encoding* shows: n is
  one token at the end of its first line, and m is a count of neighbour-mentions
  halved. Reading them here rather than making a rule find them keeps the rule
  honest about which rung it needs.
  """
  if rung not in RUNGS:
    raise ValueError(f"unknown rung: {rung}; known: {list(RUNGS)}")
  parsed = parse_primer(
      primers.build_primer(graph, condition, k_min=k_min, k_max=k_max)
  )
  return Context(
      primer=parsed,
      targets=targets,
      n=graph.number_of_nodes() if rung >= 2 else None,
      m=graph.number_of_edges() if rung >= 3 else None,
  )


def parse_corpus(graphs, condition: str, k_min: int = 2, k_max: int = 3):
  """Renders and parses one primer per graph, so the table can reuse them.

  The full table is 7 conditions x 6 tasks x 3 rungs, and the primer depends only
  on the condition -- rendering per cell would repeat the same work 18 times.
  """
  return [
      parse_primer(primers.build_primer(g, condition, k_min=k_min, k_max=k_max))
      for g in graphs
  ]


def build_rows(
    graphs, condition: str, task: str, rung: int, seed: int = 0,
    k_min: int = 2, k_max: int = 3, parsed=None,
) -> list[tuple[Context, str]]:
  """Pairs a solver context with its gold answer, one row per graph.

  `parsed` optionally supplies primers already rendered by `parse_corpus`, index
  aligned with `graphs`.

  Graphs too small to supply the task's query are dropped rather than raising:
  the generator's minimum is 5 nodes, so this only ever bites on hand-built test
  graphs, and dropping them is better than a corpus that cannot include them.
  """
  rng = random.Random(seed)
  rows = []
  for index, graph in enumerate(graphs):
    if graph.number_of_nodes() < QUERY_ARITY[task]:
      continue
    targets = sample_query(graph, task, rng)
    if parsed is None:
      context = make_context(graph, condition, targets, rung, k_min, k_max)
    else:
      context = Context(
          primer=parsed[index],
          targets=targets,
          n=graph.number_of_nodes() if rung >= 2 else None,
          m=graph.number_of_edges() if rung >= 3 else None,
      )
    rows.append((context, graphqa.gold_answer(graph, task, targets)))
  return rows


def score_theorem(rule: Rule, rows) -> dict:
  """Coverage and precision for a theorem rule.

  Precision is 1 by construction for a correct theorem, so it is reported in
  order to be *asserted*: a value below 1 is a bug in the rule or in
  `graphqa.gold_answer`, and either way the run should fail rather than quietly
  publish a number. Coverage -- the share of rows where the rule fires -- is the
  quantity that actually varies.
  """
  fired = correct = 0
  for context, gold in rows:
    answer = rule.solve(context)
    if answer is None:
      continue
    fired += 1
    correct += graphqa.normalize(answer) == graphqa.normalize(gold)
  return {
      "rule": rule.name,
      "task": rule.task,
      "fired": fired,
      "total": len(rows),
      "coverage": fired / len(rows) if rows else 0.0,
      "precision": correct / fired if fired else None,
  }


def majority_answer(golds) -> str:
  """The most common gold answer, ties broken lexicographically for determinism."""
  counts = collections.Counter(golds)
  return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def baseline_accuracy(fit_golds, test_golds) -> tuple[str, float]:
  """Majority-class accuracy, fitted and evaluated on disjoint graph sets.

  One bit of fitted information, so the inflation from fitting in-sample would be
  negligible -- but the split costs nothing here and the whole design rests on
  the bar being honest, so it is done the same way everywhere.
  """
  answer = majority_answer(fit_golds)
  hits = sum(gold == answer for gold in test_golds)
  return answer, hits / len(test_golds) if test_golds else 0.0


# --- heuristics and fitted rules ------------------------------------------
#
# Three kinds of rule, reported separately because mixing them produces a number
# that means nothing:
#
#   theorem    always correct; report coverage, precision is 1 by construction
#   heuristic  can be wrong, but nothing was fitted, so no split is needed
#   fitted     contains numbers derived from data, so it MUST be fitted on graphs
#              disjoint from the ones it is scored on -- see `Split`
#
# Fitted rules are the dangerous kind. Fitting and scoring on the same graphs
# inflates the bar, and an inflated bar makes a model look like it failed to
# reach something that was never there, which reverses the sign of the finding.


@dataclasses.dataclass(frozen=True)
class FittedRule:
  """A rule whose parameters come from data, or a heuristic with none.

  `fit` maps fitting rows to parameters; a heuristic sets it to None and its
  `apply` ignores the parameter argument. `apply` returns an answer or None.
  """

  name: str
  task: str
  kind: str
  why: str
  fit: object
  apply: object


@dataclasses.dataclass(frozen=True)
class Split:
  """Disjoint graph sets for fitting and for evaluation.

  The plan calls for this to be structural rather than a comment, because it is
  the single mistake that would reverse the sign of the headline finding. Equal
  seeds raise here, so a rule cannot be fitted on its own evaluation set without
  someone deliberately defeating the type.
  """

  fit_seed: int = 999
  test_seed: int = 1234
  size: int = 500

  def __post_init__(self):
    if self.fit_seed == self.test_seed:
      raise ValueError(
          f"fitting and evaluation must use different seeds, got "
          f"{self.fit_seed} for both"
      )

  def fit_graphs(self) -> list:
    return generate_corpus(self.size, self.fit_seed)

  def test_graphs(self) -> list:
    return generate_corpus(self.size, self.test_seed)


def generate_corpus(count: int, seed: int) -> list:
  """Canonical ER graphs from the vendored generator that produced GraphQA."""
  # Imported here rather than at module scope so that importing `shortcuts` does
  # not pull in the vendored package for callers that only want the parser.
  from talk_like_a_graph import graph_generators  # pylint: disable=g-import-not-at-top

  return [
      graphqa.canonical(g)
      for g in graph_generators.generate_graphs(
          count, "er", False, random_seed=seed
      )
  ]


def _fit_threshold(rows, feature, high: str, low: str):
  """Best split point on a scalar feature, chosen to maximise fitting accuracy."""
  scored = []
  for context, gold in rows:
    value = feature(context)
    if value is not None:
      scored.append((value, gold))
  if not scored:
    return None

  seen = sorted({value for value, _ in scored})
  candidates = (
      [(seen[i] + seen[i + 1]) / 2 for i in range(len(seen) - 1)] or [seen[0]]
  )
  best_hits, best = -1, candidates[0]
  for threshold in candidates:
    hits = sum(
        (high if value > threshold else low) == gold for value, gold in scored
    )
    if hits > best_hits:
      best_hits, best = hits, threshold
  return best


def _bucket_majority(rows, key):
  """Majority gold answer per bucket -- the lookup table, fitted by counting."""
  buckets = collections.defaultdict(list)
  for context, gold in rows:
    bucket = key(context)
    if bucket is not None:
      buckets[bucket].append(gold)
  return {bucket: majority_answer(golds) for bucket, golds in buckets.items()}


def _lookup(context, params, key):
  if not params:
    return None
  bucket = key(context)
  return params.get(bucket)


# cycle_check, keyed on the component count the primer states.
def _key_c(context):
  return context.primer.components


def _key_cn(context):
  c, n = context.primer.components, context.known_n
  return None if c is None or n is None else (c, n)


# node_count, keyed on the largest stated degree.
def _key_max_degree(context):
  return max(context.primer.degree.values()) if context.primer.degree else None


def _mean_clustering(context):
  values = context.primer.clustering
  return sum(values.values()) / len(values) if values else None


def _mean_rwse(context, k: int = 2):
  values = [t[k] for t in context.primer.rwse.values() if k in t]
  return sum(values) / len(values) if values else None


def _chung_lu(context):
  """Configuration-model edge probability d_a*d_b / 2m for the queried pair."""
  m = context.known_m
  if m is None or m == 0 or len(context.targets) < 2:
    return None
  a, b = context.targets[:2]
  da, db = context.primer.degree.get(a), context.primer.degree.get(b)
  if da is None or db is None:
    return None
  return (da * db) / (2.0 * m)


def _degree_sum_margin(context):
  """d_a + d_b - (n-1): positive is the plan's parameter-free Yes rule."""
  n = context.known_n
  if n is None or len(context.targets) < 2:
    return None
  a, b = context.targets[:2]
  da, db = context.primer.degree.get(a), context.primer.degree.get(b)
  if da is None or db is None:
    return None
  return (da + db) - (n - 1)


def _apply_degree_sum(context, params):
  del params  # parameter-free: the threshold is 0 by definition, not by fitting
  margin = _degree_sum_margin(context)
  return None if margin is None else ("Yes" if margin > 0 else "No")


def _threshold_applier(feature, high: str, low: str):
  def apply(context, params):
    if params is None:
      return None
    value = feature(context)
    return None if value is None else (high if value > params else low)

  return apply


def _rwse_features(context, node: int):
  """2m * RWSE_k, the stationary inversion of d_i = 2m * pi_i."""
  table = context.primer.rwse.get(node)
  m = context.known_m
  if not table or m is None:
    return None
  return [2.0 * m * table.get(2, 0.0), 2.0 * m * table.get(3, 0.0), 1.0]


def _fit_stationary(rows):
  features, targets = [], []
  for context, gold in rows:
    if not context.targets:
      continue
    row = _rwse_features(context, context.targets[0])
    if row is not None:
      features.append(row)
      targets.append(float(gold))
  if len(features) < 8:
    return None
  coefficients, *_ = np.linalg.lstsq(
      np.array(features), np.array(targets), rcond=None
  )
  return coefficients


def _apply_stationary(context, params):
  if params is None or not context.targets:
    return None
  row = _rwse_features(context, context.targets[0])
  if row is None:
    return None
  predicted = float(np.dot(params, row))
  ceiling = (context.known_n - 1) if context.known_n else 100
  return str(int(round(min(max(predicted, 0.0), ceiling))))


def _pairs(n: int | None) -> float | None:
  return None if n is None else n * (n - 1) / 2.0


def _fit_edge_count_density(rows):
  """m from mean clustering, which estimates p in an ER graph."""
  features, targets = [], []
  for context, gold in rows:
    mean, pairs = _mean_clustering(context), _pairs(context.known_n)
    if mean is None or pairs is None:
      continue
    features.append([mean * pairs, pairs, 1.0])
    targets.append(float(gold))
  if len(features) < 8:
    return None
  coefficients, *_ = np.linalg.lstsq(
      np.array(features), np.array(targets), rcond=None
  )
  return coefficients


def _apply_edge_count_density(context, params):
  if params is None:
    return None
  mean, pairs = _mean_clustering(context), _pairs(context.known_n)
  if mean is None or pairs is None:
    return None
  predicted = float(np.dot(params, [mean * pairs, pairs, 1.0]))
  return str(int(round(min(max(predicted, 0.0), pairs))))


HEURISTICS = (
    FittedRule(
        "degree_sum_exceeds_n", "edge_existence", "heuristic",
        "Yes iff d_a + d_b > n-1, one comparison on two stated numbers",
        None, _apply_degree_sum,
    ),
)

FITTED = (
    FittedRule(
        "cycle_lookup_c", "cycle_check", "fitted",
        "majority answer per component count",
        lambda rows: _bucket_majority(rows, _key_c),
        lambda ctx, p: _lookup(ctx, p, _key_c),
    ),
    FittedRule(
        "cycle_lookup_c_n", "cycle_check", "fitted",
        "majority answer per (c, n) -- more table rows, more inflation",
        lambda rows: _bucket_majority(rows, _key_cn),
        lambda ctx, p: _lookup(ctx, p, _key_cn),
    ),
    FittedRule(
        "node_count_from_max_degree", "node_count", "fitted",
        "majority n per largest stated degree",
        lambda rows: _bucket_majority(rows, _key_max_degree),
        lambda ctx, p: _lookup(ctx, p, _key_max_degree),
    ),
    FittedRule(
        "edge_existence_clustering_density", "edge_existence", "fitted",
        "mean clustering estimates ER density, thresholded",
        lambda rows: _fit_threshold(rows, _mean_clustering, "Yes", "No"),
        _threshold_applier(_mean_clustering, "Yes", "No"),
    ),
    FittedRule(
        "edge_existence_rwse_density", "edge_existence", "fitted",
        "mean RWSE(k=2) estimates density, thresholded",
        lambda rows: _fit_threshold(rows, _mean_rwse, "Yes", "No"),
        _threshold_applier(_mean_rwse, "Yes", "No"),
    ),
    FittedRule(
        "edge_existence_chung_lu", "edge_existence", "fitted",
        "configuration-model edge probability d_a*d_b/2m, thresholded",
        lambda rows: _fit_threshold(rows, _chung_lu, "Yes", "No"),
        _threshold_applier(_chung_lu, "Yes", "No"),
    ),
    FittedRule(
        "node_degree_stationary", "node_degree", "fitted",
        "d_i = 2m * pi_i, inverted from RWSE by least squares",
        _fit_stationary, _apply_stationary,
    ),
    FittedRule(
        "edge_count_from_clustering", "edge_count", "fitted",
        "m from mean clustering as a density estimate",
        _fit_edge_count_density, _apply_edge_count_density,
    ),
)

ALL_RULES = THEOREMS + HEURISTICS + FITTED


# --- the solver -----------------------------------------------------------


def fit_rules(rules, fit_rows, task: str) -> list:
  """Fits the rules for one task on its fitting rows.

  Filtering by task here rather than at use time is not tidiness: the numeric
  fitters call float() on the gold answer, so handing a least-squares rule the
  rows of `connected_nodes` -- whose answers are neighbour lists -- raises.
  """
  return [
      (rule, rule.fit(fit_rows) if rule.fit is not None else None)
      for rule in rules
      if rule.task == task
  ]


def rank_rules(fitted, fit_rows, task: str) -> list:
  """Orders applicable rules by their accuracy on the *fitting* rows.

  The solver tries rules in order and takes the first that fires, so the order is
  itself a fitted choice. Ranking on the fitting rows keeps that choice out of
  the evaluation set, which is the same discipline the parameters get.
  """
  scored = []
  for rule, params in fitted:
    if rule.task != task:
      continue
    fired = hits = 0
    for context, gold in fit_rows:
      answer = rule.apply(context, params)
      if answer is None:
        continue
      fired += 1
      hits += graphqa.normalize(answer) == graphqa.normalize(gold)
    accuracy = hits / fired if fired else -1.0
    scored.append((accuracy, rule.name, rule, params))
  scored.sort(key=lambda item: (-item[0], item[1]))
  return [(rule, params) for _, _, rule, params in scored]


def solve_context(context: Context, task: str, fitted=(), fallback: str = "") -> str:
  """Answers from a parsed primer: theorems first, then ranked rules, then guess.

  Theorems go first unconditionally because they are exact -- there is never a
  reason to prefer a fallible rule when one of them fires.
  """
  for rule in THEOREMS:
    if rule.task != task:
      continue
    answer = rule.solve(context)
    if answer is not None:
      return answer
  for rule, params in fitted:
    if rule.task != task:
      continue
    answer = rule.apply(context, params)
    if answer is not None:
      return answer
  return fallback


def solve(
    primer_text: str, task: str, targets=(), n: int | None = None,
    m: int | None = None, fitted=(), fallback: str = "",
) -> str:
  """The primer-only solver, taking the rendered primer as a string.

  Note what is absent from this signature: there is no graph parameter, and so no
  way for a rule to consult the graph. tests/test_shortcuts.py asserts that
  absence, so the guarantee survives refactoring rather than resting on review.
  Taking text rather than statistics is the second half of it -- a rule can only
  read the two-decimal values the prompt actually showed the model.
  """
  context = Context(parse_primer(primer_text), tuple(targets), n, m)
  return solve_context(context, task, fitted, fallback)


def score_solver(rows, task: str, fitted=(), fallback: str = "") -> float:
  hits = sum(
      graphqa.normalize(solve_context(context, task, fitted, fallback))
      == graphqa.normalize(gold)
      for context, gold in rows
  )
  return hits / len(rows) if rows else 0.0


# --- one cell of the table ------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Cell:
  """Baseline, shortcut score and per-rule detail for one (condition, task, rung)."""

  condition: str
  task: str
  rung: int
  baseline: float
  baseline_answer: str
  shortcut: float
  rows: int
  theorems: tuple
  rules: tuple


def score_cell(
    condition: str, task: str, rung: int, fit_graphs, test_graphs,
    fit_parsed=None, test_parsed=None, seed: int = 5,
) -> Cell:
  """Everything the table needs for one cell, with fitting kept off the test set."""
  fit_rows = build_rows(
      fit_graphs, condition, task, rung, seed=seed, parsed=fit_parsed
  )
  test_rows = build_rows(
      test_graphs, condition, task, rung, seed=seed + 1, parsed=test_parsed
  )

  fitted = rank_rules(
      fit_rules(HEURISTICS + FITTED, fit_rows, task), fit_rows, task
  )
  fallback = majority_answer([gold for _, gold in fit_rows]) if fit_rows else ""

  baseline_answer, baseline = baseline_accuracy(
      [gold for _, gold in fit_rows], [gold for _, gold in test_rows]
  )
  shortcut = score_solver(test_rows, task, fitted, fallback)

  theorems = tuple(
      score_theorem(rule, test_rows) for rule in THEOREMS if rule.task == task
  )
  detail = []
  for rule, params in fitted:
    fired = hits = 0
    for context, gold in test_rows:
      answer = rule.apply(context, params)
      if answer is None:
        continue
      fired += 1
      hits += graphqa.normalize(answer) == graphqa.normalize(gold)
    detail.append({
        "rule": rule.name,
        "kind": rule.kind,
        "coverage": fired / len(test_rows) if test_rows else 0.0,
        "accuracy": hits / fired if fired else None,
    })

  return Cell(
      condition=condition,
      task=task,
      rung=rung,
      baseline=baseline,
      baseline_answer=baseline_answer,
      shortcut=shortcut,
      rows=len(test_rows),
      theorems=theorems,
      rules=tuple(detail),
  )


def inflation(condition: str, task: str, rung: int, rule: FittedRule,
              fit_graphs, test_graphs, seed: int = 5) -> dict:
  """How much a fitted rule gains by being scored on the graphs it was fitted on.

  This is the measurement the whole design rests on: an inflated bar makes a
  model look like it failed to reach a threshold that was never really there.
  """
  fit_rows = build_rows(fit_graphs, condition, task, rung, seed=seed)
  test_rows = build_rows(test_graphs, condition, task, rung, seed=seed + 1)

  # Both sides are scored over *all* test rows, with the same fallback standing in
  # where the rule abstains. Scoring only the rows a rule fires on would compare
  # different denominators: a lookup fitted out of sample abstains on buckets it
  # never saw, and those are disproportionately the rare, hard ones -- which can
  # make the honest number come out higher than the cheating one.
  fallback = majority_answer([gold for _, gold in fit_rows]) if fit_rows else ""

  def accuracy(params, rows):
    hits = 0
    for context, gold in rows:
      answer = rule.apply(context, params)
      if answer is None:
        answer = fallback
      hits += graphqa.normalize(answer) == graphqa.normalize(gold)
    return hits / len(rows) if rows else None

  honest = rule.fit(fit_rows)
  cheating = rule.fit(test_rows)
  return {
      "rule": rule.name,
      "table_rows": len(honest) if isinstance(honest, dict) else None,
      "in_sample": accuracy(cheating, test_rows),
      "out_of_sample": accuracy(honest, test_rows),
  }


# --- the exact island -----------------------------------------------------
#
# The shortcut score covers only the rules we thought of, so it is a lower bound
# on primer-only performance. For small graphs that bound can be replaced by an
# exact one, computed without any rule at all.
#
# Enumerate every labelled graph whose per-node degrees match what the primer
# states. Two quantities fall out:
#
#   determined       the share of queries whose answer is identical across every
#                    consistent graph -- the information-theoretic bound at 100%
#                    precision, which no rule can exceed
#   bayes_optimal    accuracy of answering with the majority over consistent
#                    graphs -- the exact ceiling for ANY primer-only solver
#
# `bayes_optimal` is exact rather than approximate because ER conditioned on a
# degree sequence is uniform over the graphs with that sequence: ER(n,p) gives
# equal probability to all graphs with the same edge count, and fixing the
# degrees fixes the edge count. So the consistent set, weighted uniformly, is the
# true posterior and its majority is the Bayes rule.
#
# If a hand-written rule sits far below this, there are rules we have not found
# and the floor is loose. If one ever sits above it, that rule is reading
# something the primer does not state.

_MAX_ISLAND_NODES = 6


@functools.lru_cache(maxsize=4)
def _labelled_index(n: int) -> dict:
  """Every labelled graph on n nodes, indexed by its per-node degree tuple.

  2**15 = 32768 graphs at n=6 and 1024 at n=5, so exhaustive enumeration is
  cheaper than being clever. Edge tuples are stored rather than graph objects to
  keep the index small; the graph is rebuilt only for the handful that match.
  """
  pairs = list(itertools.combinations(range(n), 2))
  index = collections.defaultdict(list)
  for mask in range(1 << len(pairs)):
    edges = tuple(pair for i, pair in enumerate(pairs) if mask >> i & 1)
    degrees = [0] * n
    for a, b in edges:
      degrees[a] += 1
      degrees[b] += 1
    index[tuple(degrees)].append(edges)
  return dict(index)


def _rebuild(n: int, edges) -> "nx.Graph":
  graph = nx.Graph()
  graph.add_nodes_from(range(n))
  graph.add_edges_from(edges)
  return graph


def exact_island(graphs, task: str, seed: int = 5,
                 max_nodes: int = _MAX_ISLAND_NODES) -> dict:
  """The exact primer-only ceiling on graphs small enough to enumerate.

  Conditions on the stated degrees, so it bounds the `degree` and `all` arms.
  Graphs above `max_nodes` are skipped and counted, since the bound only applies
  to the rows it covers.
  """
  rng = random.Random(seed)
  determined = optimal = total = skipped = 0
  for graph in graphs:
    n = graph.number_of_nodes()
    if n > max_nodes or sorted(graph.nodes()) != list(range(n)):
      skipped += 1
      continue
    if n < QUERY_ARITY[task]:
      skipped += 1
      continue
    targets = sample_query(graph, task, rng)
    key = tuple(graph.degree(node) for node in range(n))
    candidates = _labelled_index(n)[key]
    answers = [
        graphqa.gold_answer(_rebuild(n, edges), task, targets)
        for edges in candidates
    ]
    truth = graphqa.gold_answer(graph, task, targets)
    total += 1
    determined += len(set(answers)) == 1
    optimal += majority_answer(answers) == truth

  return {
      "task": task,
      "rows": total,
      "skipped": skipped,
      "coverage": total / (total + skipped) if total + skipped else 0.0,
      "determined": determined / total if total else None,
      "bayes_optimal": optimal / total if total else None,
  }
