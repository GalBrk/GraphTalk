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
import random
import re

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


def build_rows(
    graphs, condition: str, task: str, rung: int, seed: int = 0,
    k_min: int = 2, k_max: int = 3,
) -> list[tuple[Context, str]]:
  """Pairs a solver context with its gold answer, one row per graph.

  Graphs too small to supply the task's query are dropped rather than raising:
  the generator's minimum is 5 nodes, so this only ever bites on hand-built test
  graphs, and dropping them is better than a corpus that cannot include them.
  """
  rng = random.Random(seed)
  rows = []
  for graph in graphs:
    if graph.number_of_nodes() < QUERY_ARITY[task]:
      continue
    targets = sample_query(graph, task, rng)
    rows.append((
        make_context(graph, condition, targets, rung, k_min, k_max),
        graphqa.gold_answer(graph, task, targets),
    ))
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
