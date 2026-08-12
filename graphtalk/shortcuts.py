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
check that would catch a renderer change nobody meant to make.

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

import dataclasses
import re

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
