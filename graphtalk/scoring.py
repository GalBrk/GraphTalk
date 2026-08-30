"""Answer extraction and the proposal's metrics.

Two jobs, kept apart because they fail differently.

**Extraction** turns free model text into an answer. It is the quiet source of
measurement error in a study like this: an extractor that misses a correct answer
scores it as wrong, and since CoT responses are longer and messier than zero-shot
ones, a naive extractor makes CoT look worse than it is -- which is exactly one
of the comparisons the proposal wants to draw. So extraction is written per task,
documented, and tested against the shapes models actually emit rather than being
a single regex.

**Scoring** applies the metric the proposal names for each task:

  node_degree, node_count, edge_count   exact match, with MAE as a secondary
  connected_nodes                       set-level F1
  cycle_check, edge_existence           accuracy, beside a majority baseline

Note that `connected_nodes` is scored by F1 here and by exact match in
`shortcuts.py`. That is deliberate but it means the two numbers are not
comparable: a shortcut score of 35.2% is an exact-match rate, and comparing a
model's F1 against it would be comparing different quantities. Use
`score_predictions` with the same metric on both sides before putting a model
number next to a shortcut number.
"""

import math
import re

TASKS = (
    "node_count", "edge_count", "node_degree",
    "connected_nodes", "edge_existence", "cycle_check",
)

# Tasks whose answer is a single integer.
_INTEGER_TASKS = ("node_count", "edge_count", "node_degree")
# Tasks whose answer is Yes or No.
_BOOLEAN_TASKS = ("edge_existence", "cycle_check")

_INTEGER = re.compile(r"-?\d+")
# "the answer is 12", "Answer: 12", "final answer: 12" -- checked before falling
# back to positional heuristics, since an explicit marker is unambiguous.
_MARKER = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:)?\s*[:\-]?\s*(.+)", re.IGNORECASE
)
_YES = re.compile(r"\byes\b", re.IGNORECASE)
_NO = re.compile(r"\bno\b", re.IGNORECASE)
_NO_NODES = re.compile(r"\bno\s+nodes?\b", re.IGNORECASE)
# A model answering "None"/"None." in place of the dataset's "No nodes"
# spelling for an isolated node. Anchored to the end of the scope rather than
# searched anywhere in free text, so a reasoning sentence like "None of the
# nodes are directly reachable, but node 5 is adjacent" is not misread as the
# answer -- that end anchor is what actually provides the safety (checked
# against every non-empty-gold connected_nodes row in the tracked sweep: zero
# false positives), not what precedes "none". An earlier version also
# required "none" be preceded by ":" or start-of-line, to additionally guard
# against a false positive that never actually occurred, and that extra
# requirement missed real answers glued onto a sentence with no separator at
# all, e.g. "...the list is empty.None" -- a likely generation artifact, but
# still the correct answer. Dropped.
_NONE_ANSWER = re.compile(r"\bnone\.?\s*$", re.IGNORECASE)
# "[]"/"[ ]" as an alternative empty-list spelling. Checked against the same
# non-empty-gold rows for "[]" appearing anywhere (not just at line end):
# zero matches, so this is not observed to collide with a real answer.
_EMPTY_BRACKETS = re.compile(r"\[\s*\]\s*$")
# Strips one trailing markdown-emphasis marker, sentence-ending period, or
# parenthetical aside, so a decorated empty-set answer -- "**None**", "[] (an
# empty list)" -- can still be recognised once the wrapping around it is
# removed. Applied repeatedly (bounded by the string shrinking each time)
# since a real row combines several: "...**[]** (empty list)." strips the
# period, then "(empty list)", then "**", in that order. The period matters
# on its own -- regression found against a real response ending "...is:
# **[]** (empty list).": without it, `_EMPTY_BRACKETS` never matched this
# line (the string doesn't end in "]", it ends in "."), so extraction fell
# through to a stray "0" digit from "connected to 0" earlier on the same
# line -- the same "queried node's own id" confound `_best_list` already
# guards against for non-empty answers, here defeating the empty-set check
# instead. Only feeds the empty-set check below; `_best_list`'s digit-list
# matching runs on the raw line/tail, so this cannot change any
# already-tested non-empty-answer extraction.
_TRAILING_DECORATION = re.compile(r"\s*(?:\*+|`+|\.|\([^()]*\))\s*$")


def _strip_trailing_decoration(text: str) -> str:
  while True:
    stripped = _TRAILING_DECORATION.sub("", text)
    if stripped == text:
      return text
    text = stripped
# A run of integers separated by commas and/or "and" -- the connected_nodes
# shape. The `,\s*and\s+` branch must come first: an Oxford comma makes the
# separator ", and ", and a plain `,` branch would consume the comma, fail to
# find a digit after it, and silently truncate the list one element early.
_NODE_LIST = re.compile(r"\d+(?:\s*(?:,\s*and\s+|,\s*|\s+and\s+)\d+)*")
# `edge_existence`-only paraphrases of the boolean answer that never say the
# bare word "yes"/"no". Not applied to `cycle_check`, whose CoT reasoning
# routinely mentions edges and connectivity without that being the task's
# actual yes/no answer -- see `_extract_boolean`'s `task` gate. Both are
# consulted only as a fallback when no bare yes/no exists anywhere in the
# response (see `_extract_boolean`) -- an earlier version fed them into the
# same position-based yes/no comparison as the bare tokens and regressed 44
# real rows: a mid-response restatement of the question ("...to determine if
# an edge exists between X and Y, we need to...") sometimes sat later in the
# text than the response's own correct bare "No" and won the comparison.
_EDGE_EXISTS = re.compile(r"\bedge\s+exists?\b", re.IGNORECASE)
# "is not connected to"/"isn't connected to" need no special handling: `\s+`
# between "is"/"are" and "connected" only matches whitespace, and "isn't"/
# "aren't" aren't the literal word "is"/"are" -- grammar excludes those false
# positives without extra machinery.
_CONNECTED_YES = re.compile(r"\b(?:is|are)\s+connected\s+to\b", re.IGNORECASE)
# Disqualifies a match that is restating the question or hedging rather than
# stating the model's conclusion -- e.g. "To determine if an edge exists
# between X and Y, we need to..." (the live question is itself "Does an edge
# exist between Node A and Node B?", so CoT responses very commonly restate
# it as a preamble) or "we cannot determine if an edge exists ... from the
# given data" (an explicit refusal, which must stay `unparsed` rather than
# become a guessed "Yes" -- this exact phrasing was a second real regression
# found during development). Checked against the text before a match via a
# plain regex on the prefix slice rather than a `re` lookbehind, since the
# disqualifying word can be separated from "edge"/"connected" by "an"/"the"/
# "node" and Python's `re` module requires lookbehind to be fixed-width.
_QUESTION_OR_HEDGE_LEADIN = re.compile(
    r"\b(?:if|whether|does|do|did|can|could|would|should|check|determine|"
    r"examine|verify|confirm)\b[\w\s]{0,12}$",
    re.IGNORECASE,
)


def _stated_conclusion_matches(pattern: "re.Pattern[str]", scope: str) -> list[int]:
  """Start positions of `pattern` in `scope`, excluding question/hedge lead-ins."""
  return [m.start() for m in pattern.finditer(scope)
          if not _QUESTION_OR_HEDGE_LEADIN.search(scope[:m.start()])]


def _last_nonempty_line(text: str) -> str | None:
  for line in reversed(text.splitlines()):
    line = line.strip()
    if line:
      return line
  return None


def _marker_tail(text: str) -> str | None:
  """The text after the last explicit answer marker, if there is one."""
  found = list(_MARKER.finditer(text))
  return found[-1].group(1).strip() if found else None


def has_answer_marker(text: str) -> bool:
  """Whether `text` contains an explicit "answer is/answer:" marker.

  A diagnostic signal only. Most responses in this sweep state their answer as
  a bare `A: <value>` line rather than through this marker, so its absence
  does not by itself imply a truncated or non-terminating response -- see
  `graphtalk/analysis.py` for the length-outlier heuristic actually used to
  flag suspected non-termination beyond the labelled ground truth.
  """
  return _marker_tail(text) is not None


def _extract_integer(text: str) -> str | None:
  """The answered integer.

  Prefers an explicit marker; otherwise takes the *last* integer in the response.
  Last rather than first because a CoT response ends with its conclusion while
  its middle is full of intermediate arithmetic. On a zero-shot response there is
  usually only one integer, so the two rules agree.
  """
  tail = _marker_tail(text)
  for scope in (tail, text):
    if scope:
      found = _INTEGER.findall(scope)
      if found:
        return found[0] if scope is tail else found[-1]
  return None


def _extract_boolean(text: str, task: str) -> str | None:
  """Yes or No.

  Scans for the last standalone yes/no token, for the same reason as the integer
  rule: CoT states its conclusion at the end. `No nodes` is excluded first so a
  `connected_nodes`-style phrase can never be read as a boolean No.

  Prefers the marker tail but falls back to the full text when the tail holds
  neither token, mirroring `_extract_integer` -- a marker can fire on real
  answer-introducing text ("Answer: ") whose immediate continuation still isn't
  the stated conclusion, and the real Yes/No a few sentences later must not be
  missed just because a tail was found at all.

  For `edge_existence` only, a response with no bare yes/no anywhere falls
  back to paraphrases that state the conclusion without that word -- "An edge
  exists between..." or "...is connected to...". The fallback runs only after
  both scopes have been checked for a bare token, so it can turn an `unparsed`
  row into a parsed one but can never override an already-stated answer. It
  is also checked only against each scope's *last non-empty line*, not
  searched anywhere in it: the graph encoding's own vocabulary for describing
  real (but unrelated) edges is this same phrase -- e.g. "Node 8 is connected
  to nodes 4, 5." stated early while summarising the graph, in a response that
  goes on to correctly refuse to answer about the queried pair. Restricting to
  the last line is the same "conclusion is stated last, don't pool across the
  whole response" rule `_extract_node_list` already applies for `No nodes`.
  """
  tail = _marker_tail(text)
  for scope in (tail, text):
    if not scope:
      continue
    scope = _NO_NODES.sub(" ", scope)
    last_yes = max((m.start() for m in _YES.finditer(scope)), default=-1)
    last_no = max((m.start() for m in _NO.finditer(scope)), default=-1)
    if last_yes >= 0 or last_no >= 0:
      return "Yes" if last_yes > last_no else "No"
  if task == "edge_existence":
    for scope in (tail, text):
      if not scope:
        continue
      line = _last_nonempty_line(scope) or scope
      if (_stated_conclusion_matches(_EDGE_EXISTS, line)
          or _stated_conclusion_matches(_CONNECTED_YES, line)):
        return "Yes"
  return None


def _best_list(line: str) -> str | None:
  """The most list-like run of integers in one line.

  Chosen by integer count first, then by position. Both rules are load-bearing.
  Count, because a sentence like `Node 3 is connected to nodes 0, 2, and 9.`
  contains two runs and the answer is the longer one -- taking the first match
  returns the queried node's own id, which is wrong on every such response and
  wrong in a way that looks plausible. Position breaks the tie for
  `Node 12 is connected to node 5.`, where both runs hold one integer and the
  answer is the later one.
  """
  best = None
  for match in _NODE_LIST.finditer(line):
    key = (len(_INTEGER.findall(match.group(0))), match.start())
    if best is None or key > best[0]:
      best = (key, match.group(0))
  return _clean_list(best[1]) if best else None


def _extract_node_list(text: str) -> str | None:
  """A neighbour list, or the dataset's `No nodes` spelling for an isolated node.

  Scanned line by line from the end and never across the whole response: a CoT
  answer names many nodes while reasoning, so pooling every integer in the text
  would return the union of everything it considered. The last line that parses
  as a list is the stated answer.

  A model's own "None"/"None."/"[]" (`_NONE_ANSWER`/`_EMPTY_BRACKETS`, checked
  after stripping trailing markdown decoration) is treated the same as the
  dataset's "No nodes" (`_NO_NODES`) -- all canonicalize to the same string
  so nothing downstream needs to know which spelling produced it.

  The per-line scan runs *before* the marker tail, not after: `_marker_tail`
  returns text after the *last* "answer"-labeled mention anywhere in the
  response, which can be a mid-reasoning heading rather than the true
  conclusion -- regression found against a real response where "3. **Determine
  the Answer:** ... node 0 has no listed neighbors." preceded the true final
  "A: []" line, and a stray digit in that heading ("node 0") was returned
  under tail-first priority instead of the real answer one line later. The
  per-line scan already implements "the last stated line wins"; the marker
  tail is now a fallback for when the true last lines have no content at all,
  not an override that can shadow them.
  """
  for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
    core = _strip_trailing_decoration(line)
    if _NO_NODES.search(core) or _NONE_ANSWER.search(core) or _EMPTY_BRACKETS.search(core):
      return "No nodes"
    found = _best_list(line)
    if found:
      return found

  tail = _marker_tail(text)
  if tail is not None:
    core = _strip_trailing_decoration(tail)
    if _NO_NODES.search(core) or _NONE_ANSWER.search(core) or _EMPTY_BRACKETS.search(core):
      return "No nodes"
    found = _best_list(tail)
    if found:
      return found
  return None


def _clean_list(raw: str) -> str:
  """Normalises a matched run of integers to the dataset's `0, 1, 2` spelling."""
  return ", ".join(_INTEGER.findall(raw))


def extract_answer(text: str, task: str) -> str | None:
  """The model's answer to `task`, or None when nothing could be read.

  None is distinct from a wrong answer and is counted separately: a run with many
  unparseable responses is an extraction bug or a truncated generation, not a
  model that got things wrong, and collapsing the two hides that.
  """
  if task not in TASKS:
    raise ValueError(f"unknown task: {task}; known: {list(TASKS)}")
  if not text or not text.strip():
    return None
  if task in _INTEGER_TASKS:
    return _extract_integer(text)
  if task in _BOOLEAN_TASKS:
    return _extract_boolean(text, task)
  return _extract_node_list(text)


def normalize(answer: str) -> str:
  """Trims the dataset's leading space and trailing period."""
  return answer.strip().rstrip(".").strip()


def _node_set(answer: str) -> set[int]:
  if _NO_NODES.search(answer):
    return set()
  return {int(tok) for tok in _INTEGER.findall(answer)}


def _boolean(answer: str) -> str | None:
  cleaned = _NO_NODES.sub(" ", answer)
  if _YES.search(cleaned):
    return "Yes"
  if _NO.search(cleaned):
    return "No"
  return None


def set_f1(predicted: str | None, gold: str) -> float:
  """Set-level F1 between two neighbour lists.

  Both empty scores 1.0 -- an isolated node's gold answer is `No nodes`, and a
  model that says so is exactly right, so the degenerate case has to be a hit
  rather than a 0/0. One empty and one not scores 0.0.
  """
  if predicted is None:
    return 0.0
  want, got = _node_set(gold), _node_set(predicted)
  if not want and not got:
    return 1.0
  if not want or not got:
    return 0.0
  overlap = len(want & got)
  if not overlap:
    return 0.0
  precision, recall = overlap / len(got), overlap / len(want)
  return 2 * precision * recall / (precision + recall)


def score_one(predicted: str | None, gold: str, task: str) -> dict:
  """Scores a single response under the metric the proposal names for the task.

  `primary` is the number the task is reported on -- accuracy for five tasks, F1
  for `connected_nodes`. `exact` is exact match for every task, kept alongside so
  that `connected_nodes` can also be compared against the shortcut table, which is
  an exact-match figure.
  """
  if task not in TASKS:
    raise ValueError(f"unknown task: {task}; known: {list(TASKS)}")
  result = {"parsed": predicted is not None, "exact": 0.0, "primary": 0.0,
            "absolute_error": None}
  if predicted is None:
    return result

  if task in _BOOLEAN_TASKS:
    hit = float(_boolean(predicted) == _boolean(gold) and _boolean(gold) is not None)
    result["exact"] = result["primary"] = hit
    return result

  if task == "connected_nodes":
    result["exact"] = float(_node_set(predicted) == _node_set(gold))
    result["primary"] = set_f1(predicted, gold)
    return result

  hit = float(normalize(predicted) == normalize(gold))
  result["exact"] = result["primary"] = hit
  # Mean absolute error is the proposal's secondary signal on the integer tasks:
  # it separates "off by one" from "wildly wrong", which accuracy cannot.
  try:
    result["absolute_error"] = abs(int(normalize(predicted)) - int(normalize(gold)))
  except ValueError:
    result["absolute_error"] = None
  return result


def aggregate(scores) -> dict:
  """Means over a list of `score_one` results.

  `absolute_error` averages only over the rows where it is defined, and reports
  how many those were, so a mean over three of thirty rows cannot be read as a
  mean over thirty.
  """
  scores = list(scores)
  if not scores:
    return {"rows": 0}
  errors = [s["absolute_error"] for s in scores if s["absolute_error"] is not None]
  return {
      "rows": len(scores),
      "parsed": sum(s["parsed"] for s in scores) / len(scores),
      "exact": sum(s["exact"] for s in scores) / len(scores),
      "primary": sum(s["primary"] for s in scores) / len(scores),
      "mae": sum(errors) / len(errors) if errors else None,
      "mae_rows": len(errors),
  }


def majority_baseline(golds) -> tuple[str, float]:
  """The most common gold answer and its share.

  The proposal asks for this on `cycle_check` and `edge_existence` specifically,
  because both carry a known class imbalance -- `cycle_check` is 83.2% Yes on the
  published test split, so an accuracy below that is worse than a constant.
  """
  golds = [normalize(g) for g in golds]
  if not golds:
    return "", 0.0
  counts = {}
  for gold in golds:
    counts[gold] = counts.get(gold, 0) + 1
  best = max(sorted(counts), key=lambda answer: counts[answer])
  return best, counts[best] / len(golds)


def mcnemar(control_hits, treatment_hits) -> dict:
  """Exact McNemar test on paired binary outcomes.

  The proposal's test for one primer condition against the no-primer control, on
  the same graphs. Only the discordant pairs carry information: b is the count
  where the control was right and the treatment wrong, c the reverse, and under
  the null each discordant pair is a fair coin. The exact binomial version is used
  rather than the chi-square approximation because at 30 rows the discordant count
  is often single digits, where the approximation is not trustworthy.

  Takes sequences of 0/1 (or bool) aligned pairwise; raises if they are not the
  same length, since a silent misalignment would invert the pairing.
  """
  control_hits, treatment_hits = list(control_hits), list(treatment_hits)
  if len(control_hits) != len(treatment_hits):
    raise ValueError(
        f"paired test needs equal lengths, got {len(control_hits)} "
        f"and {len(treatment_hits)}"
    )
  b = sum(1 for x, y in zip(control_hits, treatment_hits) if x and not y)
  c = sum(1 for x, y in zip(control_hits, treatment_hits) if y and not x)
  discordant = b + c
  if discordant == 0:
    # No pair disagrees, so there is nothing to test; p = 1 rather than 0/0.
    return {"b": b, "c": c, "discordant": 0, "p_value": 1.0}
  smaller = min(b, c)
  tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
  p_value = min(1.0, 2.0 * tail / (2 ** discordant))
  return {"b": b, "c": c, "discordant": discordant, "p_value": p_value}
