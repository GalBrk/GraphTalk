"""What the models claim about the graph on the way to an answer, checked against the graph.

cycle_check first. At n=40 its gold is "yes" on every graph, so accuracy only
says a model answered yes. For every correct, finished response of the main
sweep (runs/qwen3-{1.7b,4b}[-think].densfull40*) this checks each cycle the final
answer names -- for the thinking arms, the text after </think> -- against the
edges of its own prompt. It extends [rpcycle] in scripts/response_patterns.py,
which reads arrow chains only and counts a response as real when any chain it
writes is.

A named cycle is an arrow chain in any notation ("0 → 3 → 8 → 0", "0-3-8-0",
"0 \\rightarrow 3 ..."), a node list ("nodes 0, 3 and 8 form a triangle", "a
cycle through nodes 0, 3, 8"), or a closing chain of "a is connected to b"
statements. A chain that runs into a loop without returning to its start
("0 → 2 → 5 → 7 → 2", a lasso) is judged on its loop. A named cycle is rejected
when the text after it -- up to the next one or the next section break, WINDOW
characters at most -- says so ("not a cycle", "repeats node 2", "invalid", ...).

A named cycle is real (every step an edge, and its distinct edges close a
cycle), invented (some step is not an edge), length-2 (one edge walked there and
back) or retraced (every step an edge, but it only goes back the way it came).
An answer rests on a real cycle if it names one and does not reject it, wherever
it appears; otherwise on the last cycle it does not reject (invented, or not a
cycle); otherwise on none, where an edge-count argument ("more than n - 1
edges") is counted.

The same reading runs on the thinking traces (the text before </think>), and on
the cycle_check clean-condition runs (runs/qwen3-0.6b[-think].cc500*: 500
published graphs of 5-19 nodes, 84 of them acyclic), where a cycle named on an
acyclic graph is necessarily invented and the answer that rests on it is wrong.
The pilot's published split has 2 acyclic graphs of 30, too few to use.

The other tasks of the main sweep, finished responses. "Node x is connected to
(nodes) a, b, c" -- or "node x's neighbours are ..." -- states those edges; the
same words after "check if" / "whether" / "maybe" / "imply that", or before a
"?" or "via" / "through" in the same sentence, are not a claim and are skipped. The
thinking arms write these lists in their trace, so edge_existence and
node_degree read the whole response. A response repeated across shards counts
once, as in pf.load_runs.
  edge_existence   whether the last list stated for either endpoint contains the
                   other when the pair is not an edge (a fabricated edge), or
                   omits it when it is one (a dropped edge); and whether any list
                   the response states contains a non-neighbour
  node_degree      the longest list stated for the queried node (its reading of
                   the node's line) against its neighbours: a wrong answer is
                   misread (the list is wrong), miscounted (the list is right)
                   or has no list
  connected_nodes  the answer's nodes against the true neighbours
  edge_count       the first broken step of the degree-sum chain,
                   response_patterns.edge_chain ([rpchain]); all responses
  node_count       the gold is 40 on every graph: the answer given (40, 39 or
                   other), and whether the answer lists the ids one per line
                   (30 or more lines holding one id each), the path that ends
                   by reporting the last id, 39

The high-density extension (runs/qwen3-*.densfull40hi*, p = .65-.85) has
edge_existence and node_degree only; it is read the same way and reported in
its own blocks, so every other block stays p <= .50.

Why answers are wrong, beyond the claims above:
  edge_existence   a false yes may claim the other node is listed ("Node 12 is
                   listed among these"), a miss that it is not ("Node 4 is not in
                   this list"); the same words after "check if" / "whether" are
                   skipped. A miss may also rule that both lists must agree
                   ("discrepancy", "inconsistent", "mutual", ...). Shared
                   neighbours are counted on the graph, not read from the text.
  cycle_check      a "no" may mention an odd degree sum or the handshaking
                   lemma, return probabilities / self-loops, or call the graph
                   directed
  node_degree      whether a wrong answer is below the true degree
  connected_nodes  where each invented neighbour sits (a neighbour's
                   neighbour, the line of node q-1 or q+1, one off a true
                   neighbour's id, q itself), against the share of q's
                   non-neighbours there; which neighbours are dropped
The primer-against-none counts, the filler table and the cycle_check outcomes
come from frame.csv (rule R1), not from the text.

Each primer is tested against none on the same graphs: exact McNemar, BH over
the primers within each arm, as in primer_findings.

  [ccanswer]   per arm and primer: % of correct answers by what they rest on,
               and % that rejected a real cycle and rest on something else
  [ccinvent]   per arm and primer: % asserting an invented cycle, and % naming
               one and rejecting it
  [cctest]     paired tests: asserts an invented cycle / rests on an invented
               cycle / rests on a real cycle
  [ccwhere]    per arm: how often the closing step vs an inner step is the
               invented one; the fake neighbour taken from the line of node
               a-1 or a+1, or one off a real neighbour's id, against chance
  [ccdensity]  per arm: % asserting an invented cycle, by density
  [ccthink]    thinking traces of the correct answers: % naming a real cycle /
               asserting an invented one / rejecting an invented one; tested
  [ccacyclic]  cc500: on acyclic graphs, the wrong "yes" answers by what they
               rest on; on cyclic graphs, correct answers resting on a real
               cycle; wrong "yes" on acyclic graphs tested
  [eeclaims]   edge_existence: fabricated and dropped queried edges, and how
               many false alarms / misses they account for; tested
  [ndlist]     node_degree: stated lists and why the wrong answers are wrong;
               tested
  [cnset]      connected_nodes: answers with invented / missed neighbours;
               tested
  [ecchain]    edge_count: first broken step; misread degree values tested
  [ncanswer]   node_count: answers of 40 / 39 / other, the listing path and
               accuracy on each path; answering 39 tested
  [ncdensity]  node_count: % answering 39, by density
  [eehigh]     [eeclaims] at p = .65-.85
  [ndhigh]     [ndlist] at p = .65-.85
  [pvn]        every primer against none on the 24 (arm, task) pairings: counts
               by kind of change, the largest changes, each significant one
  [fillerdens] filler against none by density, 100 graphs per cell,
               uncorrected exact McNemar
  [ccoutcome]  cycle_check: % correct / answering no / truncated
  [ccno]       cycle_check "no" answers by the reason cited; answering no tested
  [eeshared]   edge_existence false yes by shared neighbours; a within-density
               permutation test without a primer; primers tested on pairs
               sharing 4 or more
  [eewhy]      edge_existence false yes: states the pair as an edge / claims the
               other node is listed / neither; claiming tested
  [eemiss]     edge_existence misses: "not listed" claims, a stated list that
               holds the other node, "both lists must agree" reasoning
  [ndsign]     node_degree: % of wrong answers below the true degree;
               undercounting tested
  [cnsource]   connected_nodes: invented neighbours by source against chance,
               missed neighbours by place on the line; q itself, dropping the
               first and dropping the last tested
  [ectable]    edge_count: % with a wrong degree value, among answers that list
               a table
  [ccsample]   random examples of each cycle rule, to check by eye
  [clsample]   random examples of the edge_existence and node_degree readings
  [ncsample]   node_count answers of 39
  [eesample]   edge_existence membership claims and misses

  PYTHONPATH=. python scripts/check_cycle_claims.py --csv-dir csv2/raw-trends \\
      > csv2/raw-trends/check_cycle_claims.txt
"""
import argparse
import collections
import csv
import glob
import json
import os
import random
import re
import sys

import numpy as np
import pandas as pd

from graphtalk import scoring

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import primer_findings as pf  # noqa: E402  (FRAME, ARMS, PRIMERS, SEED, with_outcomes, bh, neighbour_set)
import response_patterns as rp  # noqa: E402  (edge_chain: the [rpchain] reading of edge_count)

PROMPTS = "prompts.densfull40.jsonl"
PROMPTS_HI = "prompts.densfull40hi.jsonl"
CC500 = "prompts.cyclecheck500.clean.jsonl"
CC500_ARMS = ["qwen3-0.6b", "qwen3-0.6b-think"]
THINK = [a for a in pf.ARMS if a.endswith("-think")]
SHORT = {"qwen3-1.7b": "1.7B", "qwen3-1.7b-think": "1.7B-T", "qwen3-4b": "4B",
         "qwen3-4b-think": "4B-T", "qwen3-0.6b": "0.6B", "qwen3-0.6b-think": "0.6B-T"}
CONDS = ["none", "filler"] + pf.PRIMERS
CC500_CONDS = ["none", "components", "clustering", "rwse"]
DENS = ["p0.1", "p0.2", "p0.35", "p0.5"]
HI = ["0.65", "0.75", "0.85"]
CHAIN = ["right", "values", "nodes", "sum", "halving", "answer", "no table", "unparsed", "cut"]
WINDOW = 250   # characters after a named cycle searched for a rejection
TASKS6 = ["node_count", "cycle_check", "node_degree", "edge_count", "connected_nodes", "edge_existence"]
SHARED = ["0", "1", "2-3", "4+"]   # neighbours a queried pair shares
B_PERM = 2000
PVN = ["gain, states the answer", "gain, does not state it", "gain, truncation-driven", "gain, not significant",
       "neutral (within 1 point)", "loss, not significant", "loss, truncation-driven", "loss, other"]

_ARROW = re.compile(r"\s*(?:→|->|⟶|➝|⇒|=>|\\?(?:long)?rightarrow|\\to\b|—|–)\s*"
                    r"|(?<=\d)\s*-\s*(?=\d)")
_CHAIN = re.compile(r"\d+(?:\s*→\s*(?:node\s+)?\d+)+", re.I)
_BACK = re.compile(r"\s*,?\s*(?:and\s+)?(?:then\s+)?(?:back|returns?)\s+to\s+(?:node\s+)?(\d+)",
                   re.I)
_LIST = r"((?:\d+\s*,\s*)+(?:and\s+)?\d+|\d+\s+and\s+\d+)"
_FORMS = [re.compile(r"(?:nodes?\s+)?" + _LIST + r"\s*,?\s*(?:form|forms|forming|create|creates"
                     r"|make|makes|constitute)\s+(?:a|an)\s+(?:[\w-]+\s+){0,3}?(?:triangle|cycle|loop)",
                     re.I),
          re.compile(r"(?:triangle|cycle|loop)\s*(?:\([^)]*\)\s*)?(?:involving|between|among|through"
                     r"|formed by|consisting of|with|on)\s+(?:the\s+)?(?:nodes?\s+)?" + _LIST, re.I)]
_PAIR = re.compile(r"(\d+)\s+is\s+(?:also\s+|directly\s+)?connected\s+(?:back\s+)?to\s+(?:node\s+)?(\d+)"
                   r"|(\d+)\s*\(\s*connected\s+(?:back\s+)?to\s+(?:node\s+)?(\d+)", re.I)
# A length-2 claim made in words, with no walk written out.
_TWO = re.compile(r"[^.\n]*(?:length[- ]2\b|2-cycle|cycle of two|two-node cycle"
                  r"|(?:both directions|bidirectional)[^.\n]*form\w*\s+(?:a\s+)?(?:\w+\s+)?cycle)"
                  r"[^.\n]*", re.I)
_EDGEARG = re.compile(r"n\s*[-−–→]\s*1\b|\btree\b|more edges than|number of edges", re.I)
# Verb forms only: "(with/no) repeated nodes (except the start)" is praise, not a rejection.
# "not a simple cycle" is left out: it is mostly a concession ("..., but it shows a loop").
_REJECT = re.compile(r"not a (?:valid |proper )?cycle|isn\x27?t a (?:valid )?cycle|not valid"
                     r"|invalid|(?<!not )(?<!no )(?<!without )(?:repeats|repeating|revisits|revisiting)"
                     r" (?:node|nodes|a node"
                     r"|the node|vertex|\d+)|doesn\x27?t (?:work|form|close|lead|return|connect)"
                     r"|does not (?:work|form|close|lead|return|connect)|is not (?:directly )?connected"
                     r"|isn\x27?t (?:directly )?connected|\bno edge (?:between|from|to|connect)|dead end|✗|❌"
                     r"|\bwrong\b|incorrect"
                     r"|backtrack|not possible|\bfails?\b", re.I)
# Edge lists a response states. (?!\d) stops "connected to 12 other nodes" from
# backtracking to "1"; the count itself is refused by the lookahead after it. An
# item followed by "is"/"has"/"'s" opens the next clause ("..., and Node 31 is
# connected to ..."), so the list stops before it.
_CLAUSE = r"(?!\d)(?!\s*(?:is|has|are|was|\x27s)\b)"
_NUMS = r"(?:node\s+)?\d+" + _CLAUSE + r"(?:\s*(?:,\s*and|,|and)\s*(?:node\s+)?\d+" + _CLAUSE + ")*"
_NOT_A_COUNT = r"(?!\d)(?!\s*(?:other\s+|different\s+|distinct\s+)?(?:nodes|neighbou?rs|edges|connections)\b)"
_STATE = [re.compile(r"\bnode\s+(\d+)\s+is\s+(?:also\s+|directly\s+|only\s+)?connected\s+to\s*:?\s*"
                     r"(?:the\s+following\s+)?(?:nodes?\s*:?\s*)?(" + _NUMS + ")" + _NOT_A_COUNT, re.I),
          re.compile(r"\bnode\s+(\d+)(?:\x27s)?\s+(?:neighbou?rs|connections|adjacent nodes)\s*"
                     r"(?:are|is|include|:)\s*:?\s*(" + _NUMS + ")" + _NOT_A_COUNT, re.I),
          re.compile(r"\bneighbou?rs\s+of\s+node\s+(\d+)\s*(?:are|:)\s*:?\s*(" + _NUMS + ")" + _NOT_A_COUNT,
                     re.I)]
_QUESTION = re.compile(r"(?:\bif|whether|check|determine|see|verify|maybe|perhaps|suppose|assume|assuming"
                       r"|imply|implies|mean)\s+(?:\w+\s+){0,2}$", re.I)
# "... is connected to Node 21 via Node 26" / "... through another node?" is not an edge claim.
_INDIRECT = re.compile(r"^[^.\n!]*?(?:\?|\bvia\b|\bthrough\b|\bindirect)", re.I)
# node_count: a line holding one id ("- 7"); 30 or more of them list the ids one per line.
_IDLINE = re.compile(r"^\s*[-*]\s*\d+\s*$", re.M)
# edge_existence: "Node 12 is listed among these", "Node 4 is not in this list".
_MEMBER = re.compile(r"\b(?:node\s+)?(\d+)\s+(?:is|appears)\s+(not\s+)?(?:also\s+)?(?:explicitly\s+)?"
                     r"(?:listed|present|included|mentioned|found|among|in (?:the|this|its|that|these|both)\b)",
                     re.I)
_CHECKING = re.compile(r"(?:check|whether|\bif)\b[^.\n]*$", re.I)
_BOTHLISTS = re.compile(r"discrepanc|inconsisten|contradict|mutual|bidirectional", re.I)
_ODDSUM = re.compile(r"\bodd\b|handshak", re.I)
_RETURN = re.compile(r"return probab|self[- ]?(?:loop|edge)", re.I)
_DIRECTED = re.compile(r"(?<!un)directed graph|graph is directed|not (?:necessarily )?undirected", re.I)
# Primers whose text states the answer ([bars] >= 0.85 in primer_findings).
CARRY = {("node_degree", "degree"), ("node_degree", "all"), ("edge_count", "degree"), ("edge_count", "all")}
_TARGET = {"node_degree": re.compile(r"degree of node (\d+)\?"),
           "connected_nodes": re.compile(r"connected to (\d+) in alphabetical"),
           "edge_existence": re.compile(r"between Node (\d+) and Node (\d+)\?")}


def parse(prompt):
  """(edge set, neighbour sets, the prompt's line per node) of the graph a prompt encodes."""
  es, nb, lines = set(), collections.defaultdict(set), {}
  for m in re.finditer(r"^Node (\d+) is connected to nodes? ([\d, and]+)\.", prompt, re.M):
    a = int(m[1])
    lines[a] = m[0]
    for b in map(int, re.findall(r"\d+", m[2])):
      es.add(frozenset((a, b)))
      nb[a].add(b)
  return es, nb, lines


def graphs(path):
  """instance_id -> (edges, neighbours, lines, queried node ids, gold, node count) for one prompt file."""
  out = {}
  for line in open(path, encoding="utf-8"):
    r = json.loads(line)
    if r["instance_id"] in out:
      continue
    es, nb, lines = parse(r["prompt"])
    assert len(es) == r.get("edges", len(es)), r["instance_id"]
    n = int(re.search(r"among nodes ([\d, and]+)\.", r["prompt"])[1].split()[-1]) + 1
    q = _TARGET.get(r["task"])
    targets = tuple(map(int, q.search(r["prompt"]).groups())) if q else ()
    out[r["instance_id"]] = (es, nb, lines, targets, r["gold"], n)
  return out


def outcomes_by_key():
  """(arm, instance_id, condition) -> (correct, truncated, pred), rule R1."""
  f = pf.with_outcomes(pd.read_csv(pf.FRAME, dtype={"pred": str}, usecols=[
      "arm", "task", "condition", "instance_id", "exact", "hit_cap", "pred"]))
  return {(r.arm, r.instance_id, r.condition): (bool(r.correct), bool(r.truncated), r.pred)
          for r in f.itertuples()}


def norm(text):
  """Markdown stripped, every arrow notation turned into ' → '."""
  return _ARROW.sub(" → ", re.sub(r"\*\*|`|\$|\\\(|\\\)|\\\[|\\\]", "", text))


def answer_part(text):
  return text.split("</think>")[-1]


def trace_part(text):
  return text.split("</think>")[0] if "</think>" in text else ""


def _dedup(walk):
  """'8–10 → 10–11' writes each node twice; keep one."""
  out = walk[:1]
  for x in walk[1:]:
    if x != out[-1]:
      out.append(x)
  return out


def named(t):
  """(start, end, walk, quoted, kind) of every cycle t names, in order."""
  out = []
  for m in _CHAIN.finditer(t):
    w, end, kind = _dedup([int(x) for x in re.findall(r"\d+", m[0])]), m.end(), "closed"
    if w[0] != w[-1]:
      b = _BACK.match(t, end)
      if b:
        w, end = w + [int(b[1])], b.end()
      if w[0] != w[-1]:
        if w[-1] not in w[:-1]:
          continue
        w, kind = w[w.index(w[-1]):], "lasso"
    out.append((m.start(), end, w, t[m.start():end], kind))
  for rx in _FORMS:
    for m in rx.finditer(t):
      w = [int(x) for x in re.findall(r"\d+", m[1])]
      out.append((m.start(), m.end(), _dedup(w + w[:1]), m[0], "closed"))
  for para in re.finditer(r"(?:(?!\n\s*\n).)+", t, re.S):
    ps = [(m.start() + para.start(), m.end() + para.start(), int(m[1] or m[3]), int(m[2] or m[4]))
          for m in _PAIR.finditer(para[0])]
    for i in range(len(ps)):
      j = i
      while j + 1 < len(ps) and ps[j][3] == ps[j + 1][2]:
        j += 1
      if j - i >= 2 and ps[j][3] == ps[i][2]:
        out.append((ps[i][0], ps[j][1], [ps[i][2]] + [p[3] for p in ps[i:j + 1]],
                    t[ps[i][0]:ps[j][1]], "closed"))
  return sorted((x for x in out if max(x[2]) < 40 and len(set(x[2])) >= 2), key=lambda x: x[0])


def verdict(w, es):
  nodes = set(w[:-1])
  if len(nodes) == 2:
    return "length-2"
  if any(frozenset(s) not in es for s in zip(w, w[1:])):
    return "invented"
  # A closed walk contains a cycle iff its distinct edges are at least its nodes.
  return "real" if len({frozenset(s) for s in zip(w, w[1:])}) >= len(nodes) else "retraced"


def read(text, es):
  """What a text rests on, one dict per cycle it names, and whether it counts edges."""
  t = norm(text)
  cl = named(t)
  info = []
  for i, (s, e, w, quoted, kind) in enumerate(cl):
    after = t[e:min(cl[i + 1][0] if i + 1 < len(cl) else len(t), e + WINDOW)]
    after = re.split(r"\n\s*(?:---|#)", after)[0]   # a new section is not a verdict on this one
    info.append(dict(walk=w, quoted=quoted, kind=kind, after=after,
                     verdict=verdict(w, es), rejected=bool(_REJECT.search(after))))
  kept = [x for x in info if not x["rejected"]]
  if any(x["verdict"] == "real" for x in kept):
    rests = "real"
  elif kept:
    rests = "invented" if kept[-1]["verdict"] == "invented" else "not a cycle"
  elif _TWO.search(t):
    rests = "not a cycle"
  else:
    rests = "none"
  return rests, info, bool(_EDGEARG.search(t))


def statements(text):
  """(x, [y, ...], quoted) for every edge list a text states, in order; questions skipped."""
  t = text.replace("*", "")
  out = []
  for rx in _STATE:
    for m in rx.finditer(t):
      # "check if node 3 is connected to 7" and "node 3 is connected to 7? No." ask, not claim.
      x = int(m[1])
      ys = [int(y) for y in re.findall(r"\d+", m[2]) if int(y) != x]   # a node is not its own neighbour
      if ys and not _QUESTION.search(t[max(0, m.start() - 40):m.start()]) \
          and not _INDIRECT.match(t[m.end():m.end() + 80]):
        out.append((m.start(), x, ys, m[0]))
  return [x[1:] for x in sorted(out)]


def member(text, nodes, negated):
  """The first claim that one of nodes is listed (negated: is not listed), or None; checks skipped."""
  for m in _MEMBER.finditer(text):
    before = text[max(0, m.start() - 60):m.start()]
    if int(m[1]) in nodes and bool(m[2]) == negated and not _CHECKING.search(before) \
        and not _QUESTION.search(before[-40:]):
      return m
  return None


def _selfcheck():
  es = {frozenset(p) for p in [(0, 1), (1, 2), (0, 2), (2, 3)]}
  r = lambda s: read(s, es)[0]
  assert r("Yes: 0 → 1 → 2 → 0 is a cycle.") == "real"
  assert r("0 - 1 - 2 - 0 (no repeated nodes).") == "real"
  assert r("0 → 1 → 2 → 0: a closed path with repeated nodes (except the start).") == "real"
  assert r("0 → 1 → 2 → 1 → 0 repeats node 1.\n\n0 → 1 → 3 → 0 is a cycle.") == "invented"
  assert r("0 → 1 → 2 → 0, a closed loop without repeating nodes.") == "real"
  assert r("0 → 1 → 2 → 0. All nodes are distinct, and no edges are repeated.") == "real"
  assert r("0 → 1 → 3 → 0, but there is no edge between 1 and 3.") == "none"
  assert r("0 → 1 → 2 → 0 → 1 → 2 → 0 (though not a simple cycle, it shows a loop).") == "real"
  assert r("Try 0 → 1 → 3 → 0 → not a cycle.\n\nNodes 0, 1, and 2 form a triangle.") == "real"
  assert r("0 → 1 → 2 → 0\nThis is not a cycle.\n\n0 → 1 → 3 → 0 is a cycle.") == "invented"
  assert r("3 → 2 → 0 → 1 → 2 is a cycle.") == "real"                      # lasso, real loop
  assert r("Nodes 0, 1, 2, 0 form a cycle.") == "real"                     # start repeated
  assert r("0 → 1 → 2 → 0 is a cycle.\n\n---\n### Next\nThat idea fails.") == "real"
  assert r("0 → 1 → 0 is a cycle.") == "not a cycle"
  assert r("0 → 1 → 2 → 1 → 0 is a cycle.") == "not a cycle"               # retraced
  assert r("Nodes 0 and 1 are connected in both directions, forming a cycle of length 2.") == "not a cycle"
  assert r("It has more than n - 1 edges, so it is not a tree.") == "none"
  assert read("more than n - 1 edges", es)[2]
  s = lambda x: [(a, b) for a, b, _ in statements(x)]
  assert s("Node 3 is connected to nodes 0, 4, and 7.") == [(3, [0, 4, 7])]
  assert s("**Node 3** is connected to node 5.") == [(3, [5])]
  assert s("Node 3 is connected to 12 other nodes.") == []
  assert s("We check if Node 3 is connected to Node 7.") == []
  assert s("Node 3\x27s neighbors are 1, 2") == [(3, [1, 2])]
  assert s("Node 3 is not connected to node 7.") == []
  assert s("So, Node 3 is connected to Node 7? No.") == []
  assert s("Node 2 is connected to nodes 1 and 3, and Node 4 is connected to node 5.") == [(2, [1, 3]), (4, [5])]
  assert s("Node 36\x27s connections:\n  Node 8 is connected to node 2.") == [(8, [2])]
  assert s("Node 1 is connected to the following nodes:\n\nNode 1") == []
  assert s("This does not imply that Node 15 is connected to Node 24.") == []
  assert s("So Node 10 is connected to Node 21 via Node 26.") == []
  assert s("Maybe Node 32 is connected to Node 37 through another node?") == []
  assert s("Node 5 is connected to nodes 1, 2. Is that all?") == [(5, [1, 2])]
  assert member("Node 12 is listed among these connected nodes.", (12, 3), False)
  assert not member("### Step 1: Check if 12 is in the list of neighbors of 3", (12, 3), False)
  assert member("Node 4 is not in this list, so there is no edge.", (4, 7), True)
  assert not member("Node 4 is not in this list, so there is no edge.", (4, 7), False)
  assert not member("Node 9 is listed among these.", (4, 7), False)


def paired(df, col, conds, arms):
  """Per arm, each primer against none on the same graphs: [primer, % none, % primer, n, q]."""
  out = {}
  for arm in arms:
    x = df[df.arm == arm].set_index(["instance_id", "condition"])[col].unstack()
    rows = []
    for c in conds[1:]:
      if c not in x or "none" not in x:
        continue
      j = x[["none", c]].dropna().astype(bool)
      p = scoring.mcnemar(j["none"].to_numpy(), j[c].to_numpy())["p_value"]
      rows.append([c, 100 * j["none"].mean(), 100 * j[c].mean(), len(j), p])
    for row, q in zip(rows, pf.bh([row[4] for row in rows])):
      row[4] = q
    out[arm] = rows
  return out


def print_paired(label, res):
  print(f"  {label}: % none -> % primer (change, * BH q < .05), pairs")
  for arm, rows in res.items():
    print(f"    {SHORT[arm]:6s} " + " | ".join(
        f"{c} {a:.1f}->{b:.1f} ({b - a:+.1f}{'*' if q < .05 else ''}, n={n})" for c, a, b, n, q in rows))


def pc(x):
  """A share in %, or '-' on no rows."""
  return f"{100 * x.mean():5.1f}" if len(x) else "    -"


def perm_within(x, y, groups, rng):
  """Mean x where y minus mean x where not y, and its two-sided p with y permuted within each group."""
  y = y.astype(bool)
  if y.all() or not y.any():
    return float("nan"), float("nan")
  stat = lambda z: x[z].mean() - x[~z].mean()
  d, idx, hits = stat(y), [np.flatnonzero(groups == g) for g in np.unique(groups)], 0
  for _ in range(B_PERM):
    z = y.copy()
    for i in idx:
      z[i] = rng.permutation(y[i])
    hits += abs(stat(z)) >= abs(d)
  return d, (1 + hits) / (B_PERM + 1)


def against_none(x, c):
  """One (graph, condition) table's pairs of c and none: correct and truncated, none's columns suffixed _n."""
  return pd.concat([x.xs(c, axis=1, level=1), x.xs("none", axis=1, level=1).add_suffix("_n")], axis=1).dropna()


def kind(r):
  """[pvn]'s kind of change for one primer-against-none pairing."""
  sig, e = r["q"] < .05, r["e"]
  driven = sig and -r["dt"] * np.sign(e) >= abs(e) / 2   # the truncated share moves against it by half or more
  if sig and e > 0:
    return PVN[2] if driven else PVN[0] if (r["task"], r["c"]) in CARRY else PVN[1]
  if sig:
    return PVN[6] if driven else PVN[7]
  return PVN[4] if abs(e) <= 1 else PVN[3] if e > 0 else PVN[5]


def primer_vs_none(f):
  """[pvn]: every primer against none on each (arm, task), p <= .50, BH over the six within each, as [main]."""
  rows = []
  for (arm, task), d in f[f.density_class <= .5].groupby(["arm", "task"]):
    x = d.set_index(["graph_id", "condition"])[["correct", "truncated"]].unstack()
    rr = []
    for c in CONDS[1:]:
      j = against_none(x, c)
      rr.append(dict(arm=arm, task=task, c=c, e=100 * (j.correct.mean() - j.correct_n.mean()),
                     dt=100 * (j.truncated.mean() - j.truncated_n.mean()),
                     p=scoring.mcnemar(j.correct_n.astype(bool).to_numpy(),
                                       j.correct.astype(bool).to_numpy())["p_value"]))
    for r, q in zip(rr, pf.bh([r["p"] for r in rr])):
      r["q"] = q
      r["kind"] = kind(r)
    rows += rr
  name = lambda r: f"{r['c']} {pf.f1(r['e'])} {SHORT[r['arm']]} {r['task']}"
  print(f"[pvn] every primer against none, p <= .50, {len(rows) // 6} (arm, task) pairings: correct share (R1), "
        "exact McNemar, BH over the six within each pairing as in [main]; truncation-driven when the truncated "
        "share moves against the change by at least half of it; states the answer as in [bars]")
  print(f"  {'':26s}" + "".join(f"{c:>11s}" for c in CONDS[1:]))
  for k in PVN:
    print(f"  {k:26s}" + "".join(f"{sum(r['kind'] == k and r['c'] == c for r in rows):11d}" for c in CONDS[1:]))
  for label, pick, best in (("largest gain that does not state the answer", lambda r: r["kind"] == PVN[1], max),
                            ("same, outside node_count", lambda r: r["kind"] == PVN[1] and r["task"] != "node_count",
                             max),
                            ("largest loss, not truncation-driven", lambda r: r["kind"] == PVN[7], min)):
    print(f"  {label}: " + " | ".join(
        name(best(rs, key=lambda r: r["e"])) if rs else f"{c} none"
        for c in CONDS[1:] for rs in [[r for r in rows if r["c"] == c and pick(r)]]))
  for k in (PVN[0], PVN[1], PVN[2], PVN[6], PVN[7]):
    print(f"  {k}: " + "; ".join(f"{name(r)} (truncated {pf.f1(r['dt'])})" for r in rows if r["kind"] == k))


def filler_by_density(f):
  """[fillerdens]: filler against none per (arm, task, density), exact McNemar, uncorrected."""
  print("[fillerdens] filler against none per density, 100 graphs per cell: change in % correct (* p < .05, "
        "exact McNemar, uncorrected), the truncated change when 5 points or more, then ·% correct without a primer")
  for task in TASKS6:
    for arm in pf.ARMS:
      d, cells = f[(f.arm == arm) & (f.task == task)], []
      for p in sorted(d.density_class.unique()):
        j = against_none(d[d.density_class == p].set_index(["graph_id", "condition"])[
            ["correct", "truncated"]].unstack(), "filler")
        e, dt = 100 * (j.correct.mean() - j.correct_n.mean()), 100 * (j.truncated.mean() - j.truncated_n.mean())
        pv = scoring.mcnemar(j.correct_n.astype(bool).to_numpy(), j.correct.astype(bool).to_numpy())["p_value"]
        cells.append(f"p={p:g} {'0' if round(e) == 0 else f'{e:+.0f}'}{'*' if pv < .05 else ''}"
                     + (f" (tr {dt:+.0f})" if abs(dt) >= 5 else "") + f" ·{100 * j.correct_n.mean():.0f}")
      print(f"  {task:15s} {SHORT[arm]:6s} " + " | ".join(cells))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--csv-dir", help="write cycle_claims.csv (one row per named cycle, and per "
                  "answer naming none) and response_claims.csv (one row per response of the other tasks)")
  args = ap.parse_args()
  _selfcheck()
  g, ok = {**graphs(PROMPTS), **graphs(PROMPTS_HI)}, outcomes_by_key()
  cyc, trace, ee, nd, cn, ec, nc, where = [], [], [], [], [], [], [], []
  ccall, cninv, cnmiss = [], [], []
  claim_rows, samples = [], collections.defaultdict(list)

  def claim_log(base, info, es, nb, n, part):
    for i, x in enumerate(info, 1):
      x["bad"] = []
      steps = list(zip(x["walk"], x["walk"][1:]))
      for k, (a, b) in enumerate(steps):
        if frozenset((a, b)) in es:
          continue
        non = [y for y in range(n) if y != a and frozenset((a, y)) not in es]
        adj, off = nb.get(a - 1, set()) | nb.get(a + 1, set()), {y + d for y in nb.get(a, ()) for d in (-1, 1)}
        x["bad"].append(dict(step=(a, b), closing=k == len(steps) - 1, adj=b in adj, off=b in off,
                             adj_chance=sum(y in adj for y in non) / len(non),
                             off_chance=sum(y in off for y in non) / len(non)))
      claim_rows.append(dict(base, part=part, index=i, kind=x["kind"], verdict=x["verdict"],
                             rejected=int(x["rejected"]), walk=" → ".join(map(str, x["walk"])),
                             quoted=x["quoted"][:300], text_after=x["after"][:200],
                             invented_steps=";".join(f"{a}-{b}" for a, b in (s["step"] for s in x["bad"])),
                             closing_invented=sum(s["closing"] for s in x["bad"]),
                             from_adjacent_line=sum(s["adj"] for s in x["bad"]),
                             off_by_one=sum(s["off"] for s in x["bad"])))
    if not info:
      claim_rows.append(dict(base, part=part, index=0))

  for arm in pf.ARMS:
    seen = set()   # a response repeated across shards counts once, the first, as in pf.load_runs
    for path in sorted(glob.glob(f"runs/{arm}.densfull40*.shard*.jsonl")):
      for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        iid, cond, task = r.get("instance_id"), r.get("condition"), r.get("task")
        if (arm, iid, cond) not in ok or (iid, cond) in seen:
          continue
        seen.add((iid, cond))
        correct, truncated, pred = ok[(arm, iid, cond)]
        es, nb, lines, targets, gold, n = g[iid]
        p = iid.split("/")[2]
        main_sweep = p in DENS   # samples are drawn from the main sweep only
        key = dict(arm=arm, condition=cond, instance_id=iid, density=p[1:])
        text = r["response"] or ""

        if task == "cycle_check" and not truncated:
          a = answer_part(text)
          said_no = isinstance(pred, str) and pred.strip().lower().startswith("no")
          ccall.append(dict(key, said_no=said_no, odd=bool(_ODDSUM.search(a)), ret=bool(_RETURN.search(a)),
                            directed=bool(_DIRECTED.search(a))))
          if said_no:
            samples["ccno"].append((arm, cond, iid, a))

        if task == "cycle_check" and correct:
          rests, info, edgearg = read(answer_part(text), es)
          asserted = [x for x in info if x["verdict"] == "invented" and not x["rejected"]]
          rejected_real = any(x["verdict"] == "real" and x["rejected"] for x in info) and rests != "real"
          cyc.append(dict(key, rests=rests, edgearg=rests == "none" and edgearg,
                          rejected_real=rejected_real, asserted=bool(asserted),
                          caught=any(x["verdict"] == "invented" and x["rejected"] for x in info)))
          base = dict(key, source="densfull40", rests_on=rests, edge_count_argument=int(rests == "none" and edgearg),
                      rejected_real=int(rejected_real), n_named=len(info))
          claim_log(base, info, es, nb, n, "answer")
          for x in asserted:
            for s in x["bad"]:
              where.append(dict(arm=arm, **{k: s[k] for k in ("closing", "adj", "off", "adj_chance",
                                                               "off_chance")}))
              if s["adj"]:
                samples["adj"].append((arm, cond, iid, s["step"], lines))
            where.append(dict(arm=arm, steps=len(x["walk"]) - 1, closing_bad=any(s["closing"] for s in x["bad"]),
                              inner_bad=sum(not s["closing"] for s in x["bad"])))
          for x in info:
            if x["rejected"]:
              samples["rejected"].append((arm, cond, iid, x))
          if rejected_real and rests == "invented":
            samples["rejected_real"].append((arm, cond, iid, info))
          if rests == "real" and not (info[-1]["verdict"] == "real" and not info[-1]["rejected"]):
            samples["real_not_last"].append((arm, cond, iid, info))
          if arm in THINK:
            _, tinfo, _ = read(trace_part(text), es)
            trace.append(dict(key, named=bool(tinfo),
                              real=any(x["verdict"] == "real" and not x["rejected"] for x in tinfo),
                              asserted=any(x["verdict"] == "invented" and not x["rejected"] for x in tinfo),
                              caught=any(x["verdict"] == "invented" and x["rejected"] for x in tinfo)))
            claim_log(dict(base, rests_on="", n_named=len(tinfo)), tinfo, es, nb, n, "trace")

        # The thinking arms write their lists in the trace, so these two read the whole
        # response; for the queried node or pair, the last list stated is the one used.
        elif task == "edge_existence" and not truncated:
          u, v = targets
          st = statements(text)
          edge = v in nb.get(u, ())
          query = [(x, ys, qt) for x, ys, qt in st if x in (u, v)]
          last = query[-1] if query else None
          other = (v if last[0] == u else u) if last else None
          said_yes = isinstance(pred, str) and pred.strip().lower().startswith("yes")
          a = answer_part(text).replace("*", "").replace("`", "")
          claim_in, claim_out = member(a, (u, v), False), member(a, (u, v), True)
          rec = dict(key, task=task, correct=int(correct), pred=pred, edge=edge, said_yes=said_yes,
                     n_statements=len(st),
                     fabricated=any(y not in nb.get(x, ()) and y != x for x, ys, _ in st for y in ys),
                     states_nonedge=bool(last) and other in last[1] and not edge,
                     drops_edge=bool(last) and len(last[1]) >= 2 and other not in last[1] and edge,
                     states_pair=bool(last) and other in last[1],
                     shared=len(nb.get(u, set()) & nb.get(v, set())),
                     claims_listed=bool(claim_in), claims_unlisted=bool(claim_out),
                     both_lists=bool(_BOTHLISTS.search(text)))
          ee.append(rec)
          if said_yes and not edge and claim_in and main_sweep:
            samples["member"].append((arm, cond, iid, (u, v), a, claim_in, lines.get(u, f"Node {u}: no line")))
          if edge and not said_yes:
            samples["miss"].append((arm, cond, iid, (u, v), a, claim_out, lines.get(u, f"Node {u}: no line")))
          if rec["states_nonedge"] and main_sweep:
            samples["ee" if said_yes else "ee_no"].append(
                (arm, cond, iid, (u, v), last[2], lines.get(last[0], f"Node {last[0]}: no line")))
          fab = [(x, qt) for x, ys, qt in st if any(y not in nb.get(x, ()) and y != x for y in ys)]
          if arm in THINK and fab and main_sweep:
            x, qt = fab[0]
            samples["ee_fab"].append((arm, cond, iid, (u, v), qt, lines.get(x, f"Node {x}: no line")))

        elif task == "node_degree" and not truncated:
          t = targets[0]
          lists = [(ys, qt) for x, ys, qt in statements(text) if x == t]
          lists = sorted(lists, key=lambda l: len(l[0]))   # the longest: its reading of the node's line
          true = nb.get(t, set())
          ys = set(lists[-1][0]) if lists else None
          kind = "no list" if ys is None else ("miscounted" if ys == true else "misread")
          rec = dict(key, task=task, correct=int(correct), pred=pred, stated=ys is not None,
                     invented=len(ys - true) if ys is not None else 0,
                     missed=len(true - ys) if ys is not None else 0, kind=kind,
                     under=isinstance(pred, str) and float(pred) < len(true))
          nd.append(rec)
          if kind == "misread" and main_sweep:
            samples["nd_ok" if correct else "nd"].append(
                (arm, cond, iid, t, lists[-1][1], sorted(ys - true), sorted(true - ys),
                 lines.get(t, f"Node {t}: no line")))

        elif task == "connected_nodes" and not truncated:
          s = pf.neighbour_set(pred)
          if s is not None:
            q = targets[0]
            true = nb.get(q, set())
            line = sorted(true)   # the node's line lists its neighbours in ascending order
            cn.append(dict(key, task=task, correct=int(correct), pred=pred,
                           invented=len(s - true), missed=len(true - s), has_self=q in s,
                           drops_first=bool(line) and line[0] not in s, drops_last=bool(line) and line[-1] not in s))
            two = set().union(*(nb.get(y, set()) for y in true)) - true - {q}
            adj = (nb.get(q - 1, set()) | nb.get(q + 1, set())) - true - {q}
            off = {y + d for y in true for d in (-1, 1)} - true - {q}
            non = [y for y in range(n) if y != q and y not in true]
            share = lambda cls: sum(z in cls for z in non) / len(non)
            for y in s - true:
              cninv.append(dict(arm=arm, self=y == q, two=y in two, adj=y in adj, off=y in off,
                                two_chance=share(two), adj_chance=share(adj), off_chance=share(off)))
            for y in true - s:
              cnmiss.append(dict(arm=arm, first=y == line[0], last=y == line[-1], below=y < q,
                                 below_chance=sum(z < q for z in line) / len(line)))

        elif task == "edge_count":
          degrees = {i: len(nb.get(i, ())) for i in range(n)}
          p_int = None if truncated or not isinstance(pred, str) else int(float(pred))
          ec.append(dict(key, task=task, correct=int(correct), pred=pred,
                         chain=rp.edge_chain(text, degrees, p_int, truncated)))

        elif task == "node_count" and not truncated:
          listing = len(_IDLINE.findall(answer_part(text))) >= 30
          nc.append(dict(key, task=task, correct=int(correct), pred=pred, listing=listing))
          if pred == "39":
            samples["nc39"].append((arm, cond, iid, listing, answer_part(text)))

  # cc500: the published graphs, some acyclic.
  cg, cc = graphs(CC500), []
  for arm in CC500_ARMS:
    for path in sorted(glob.glob(f"runs/{arm}.cc500.shard*.jsonl")):
      for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        iid, cond, text = r["instance_id"], r["condition"], r["response"] or ""
        es, nb, lines, _, gold, n = cg[iid]
        if r["hit_cap"]:
          continue
        pred = scoring.extract_answer(text, "cycle_check")
        correct = scoring.score_one(pred, gold, "cycle_check")["exact"] == 1
        acyclic = not gold.strip().lower().startswith("yes")
        rests, info, edgearg = read(answer_part(text), es)
        assert not (acyclic and rests == "real"), (arm, iid, cond)
        cc.append(dict(arm=arm, condition=cond, instance_id=iid, acyclic=acyclic, correct=correct,
                       said_yes=pred is not None and pred.strip().lower().startswith("yes"), rests=rests,
                       asserted=any(x["verdict"] == "invented" and not x["rejected"] for x in info)))
        claim_log(dict(arm=arm, condition=cond, instance_id=iid, density="", source="cc500",
                       rests_on=rests, edge_count_argument=int(rests == "none" and edgearg),
                       rejected_real=0, n_named=len(info)), info, es, nb, n, "answer")

  cyc, trace, ee, nd, cn, ec, nc, cc = map(pd.DataFrame, (cyc, trace, ee, nd, cn, ec, nc, cc))
  ccall, cninv, cnmiss = map(pd.DataFrame, (ccall, cninv, cnmiss))
  cell = lambda d, arm, c: d[(d.arm == arm) & (d.condition == c)]
  main_part, high_part = (lambda d: d[~d.density.isin(HI)]), (lambda d: d[d.density.isin(HI)])

  print("[ccanswer] correct finished answers, % resting on: a real cycle / an invented cycle / "
        "not a cycle (length-2, retraced) / none named (of which an edge-count argument) | "
        "% that rejected a real cycle and rest on something else")
  for arm in pf.ARMS:
    for c in CONDS:
      x = cell(cyc, arm, c)
      print(f"  {SHORT[arm]:6s} {c:10s} n={len(x):3d}  {pc(x.rests == 'real')} / {pc(x.rests == 'invented')} / "
            f"{pc(x.rests == 'not a cycle')} / {pc(x.rests == 'none')} ({pc(x.edgearg)}) | {pc(x.rejected_real)}")
  print("[ccinvent] correct finished answers, % asserting an invented cycle / % naming one and rejecting it")
  for arm in pf.ARMS:
    print(f"  {SHORT[arm]:6s} " + " | ".join(
        f"{c} {pc(cell(cyc, arm, c).asserted).strip()}/{pc(cell(cyc, arm, c).caught).strip()}" for c in CONDS))
  print("[cctest] cycle_check, correct finished answers, each primer against none on the same graphs")
  cyc["on_invented"], cyc["on_real"] = cyc.rests == "invented", cyc.rests == "real"
  print_paired("asserts an invented cycle", paired(cyc, "asserted", CONDS, pf.ARMS))
  print_paired("rests on an invented cycle", paired(cyc, "on_invented", CONDS, pf.ARMS))
  print_paired("rests on a real cycle", paired(cyc, "on_real", CONDS, pf.ARMS))

  w = pd.DataFrame(where)
  print("[ccwhere] asserted invented cycles, all primers: invented steps; % of closing steps invented / "
        "% of inner steps invented; fake neighbour in the line of node a-1 or a+1, % (chance %); "
        "one off a real neighbour's id, % (chance %)")
  for arm in pf.ARMS:
    s, c = w[(w.arm == arm) & w.steps.isna()], w[(w.arm == arm) & w.steps.notna()]
    print(f"  {SHORT[arm]:6s} {len(s):5d} steps; closing {pc(c.closing_bad.astype(bool))} / "
          f"inner {100 * c.inner_bad.sum() / max((c.steps - 1).sum(), 1):5.1f}; adjacent line "
          f"{pc(s.adj.astype(bool))} ({100 * s.adj_chance.mean():4.1f}); off by one {pc(s.off.astype(bool))} "
          f"({100 * s.off_chance.mean():4.1f})")
  print("[ccdensity] % of correct finished answers asserting an invented cycle, all primers")
  for arm in pf.ARMS:
    print(f"  {SHORT[arm]:6s} " + "  ".join(
        f"p={p[1:]} {pc(cyc[(cyc.arm == arm) & (cyc.density == p[1:])].asserted).strip()}" for p in DENS))

  print("[ccthink] thinking traces (text before </think>) of the correct finished answers: % naming any "
        "cycle / % naming a real cycle, kept / % asserting an invented cycle / % naming one and rejecting it")
  for arm in THINK:
    for c in CONDS:
      x = cell(trace, arm, c)
      print(f"  {SHORT[arm]:6s} {c:10s} n={len(x):3d}  {pc(x.named)} / {pc(x.real)} / {pc(x.asserted)} / "
            f"{pc(x.caught)}")
  print_paired("trace asserts an invented cycle", paired(trace, "asserted", CONDS, THINK))

  print("[ccacyclic] cc500, finished answers. Acyclic graphs: n, % answering yes (wrong); of those, % resting "
        "on an invented cycle / not a cycle / none named. Cyclic graphs: n correct, % resting on a real cycle")
  for arm in CC500_ARMS:
    for c in CC500_CONDS:
      x = cell(cc, arm, c)
      a, y = x[x.acyclic], x[~x.acyclic & x.correct]
      wy = a[a.said_yes]
      print(f"  {SHORT[arm]:6s} {c:10s} acyclic n={len(a):3d} yes {pc(a.said_yes)}; of those {pc(wy.rests == 'invented')} / "
            f"{pc(wy.rests == 'not a cycle')} / {pc(wy.rests == 'none')} | cyclic correct n={len(y):3d} "
            f"real {pc(y.rests == 'real')}")
  print_paired("answers yes on an acyclic graph", paired(cc[cc.acyclic], "said_yes", CC500_CONDS, CC500_ARMS))

  def ee_block(d):
    for arm in pf.ARMS:
      for c in CONDS:
        x = cell(d, arm, c)
        ne, e = x[~x.edge], x[x.edge]
        fa, miss = ne[ne.said_yes], e[~e.said_yes]
        print(f"  {SHORT[arm]:6s} {c:10s} non-edges n={len(ne):3d} yes {pc(ne.said_yes)} states {pc(ne.states_nonedge)} "
              f"of yes {pc(fa.states_nonedge)} | edges n={len(e):3d} no {pc(~e.said_yes)} of no "
              f"{pc(miss.drops_edge)} | any fabricated {pc(x.fabricated)}")
    print_paired("states the queried non-edge as an edge", paired(d[~d.edge], "states_nonedge", CONDS, pf.ARMS))
    print_paired("states any edge that is not one", paired(d, "fabricated", CONDS, pf.ARMS))

  def nd_block(d):
    for arm in pf.ARMS:
      for c in CONDS:
        x = cell(d, arm, c)
        s, wr = x[x.stated], x[x.correct == 0]
        print(f"  {SHORT[arm]:6s} {c:10s} n={len(x):3d} states {pc(x.stated)}; invented {pc(s.invented > 0)} / "
              f"missed {pc(s.missed > 0)} | wrong n={len(wr):3d} {pc(wr.kind == 'misread')} / "
              f"{pc(wr.kind == 'miscounted')} / {pc(wr.kind == 'no list')}")
    print_paired("states a list with an invented neighbour", paired(d, "invents", CONDS, pf.ARMS))

  print("[eeclaims] edge_existence, finished answers. Non-edges: n, % answering yes (false alarm), % stating "
        "the pair as an edge, % of false alarms that state it. Edges: n, % answering no (miss), % of misses "
        "restating an endpoint's list without the other. All: % stating any edge that is not one")
  ee_block(main_part(ee))

  print("[ndlist] node_degree, finished answers: % stating the node's list; of those, % with an invented "
        "neighbour / % with a missed one | wrong answers: n, % misread / miscounted / no list")
  nd["invents"] = nd.invented > 0
  nd_block(main_part(nd))

  print("[cnset] connected_nodes, finished answers: % with an invented neighbour / % with a missed one / "
        "mean invented per answer")
  for arm in pf.ARMS:
    print(f"  {SHORT[arm]:6s} " + " | ".join(
        f"{c} {pc(cell(cn, arm, c).invented > 0).strip()}/{pc(cell(cn, arm, c).missed > 0).strip()}/"
        f"{cell(cn, arm, c).invented.mean():.2f}" for c in CONDS))
  cn["invents"] = cn.invented > 0
  print_paired("answer has an invented neighbour", paired(cn, "invents", CONDS, pf.ARMS))

  print("[ecchain] edge_count, all responses: % by first broken step (" + " / ".join(CHAIN) + ")")
  for arm in pf.ARMS:
    for c in CONDS:
      x = cell(ec, arm, c)
      print(f"  {SHORT[arm]:6s} {c:10s} n={len(x):3d} " + " / ".join(pc(x.chain == k).strip() for k in CHAIN))
  ec["misread"] = ec.chain == "values"
  print_paired("lists a wrong degree value", paired(ec, "misread", CONDS, pf.ARMS))

  print("[ncanswer] node_count, finished answers: n, % answering 40 / 39 / other | % listing the ids one per "
        "line; % correct when listing (n) / when not (n)")
  for arm in pf.ARMS:
    for c in CONDS:
      x = cell(nc, arm, c)
      li, di = x[x.listing], x[~x.listing]
      print(f"  {SHORT[arm]:6s} {c:10s} n={len(x):3d}  {pc(x.pred == '40')} / {pc(x.pred == '39')} / "
            f"{pc(~x.pred.isin(['40', '39']))} | listing {pc(x.listing)}; correct {pc(li.correct == 1)} "
            f"(n={len(li):3d}) / {pc(di.correct == 1)} (n={len(di):3d})")
  nc["says39"] = nc.pred == "39"
  print_paired("answers 39", paired(nc, "says39", CONDS, pf.ARMS))
  print("[ncdensity] node_count, % of finished answers that answer 39, by density")
  for arm in pf.ARMS:
    for c in CONDS:
      x = cell(nc, arm, c)
      print(f"  {SHORT[arm]:6s} {c:10s} " + "  ".join(
          f"p={p[1:]} {pc(x[x.density == p[1:]].says39).strip()}" for p in DENS))

  print("[eehigh] edge_existence at p = .65-.85, as [eeclaims]")
  ee_block(high_part(ee))
  print("[ndhigh] node_degree at p = .65-.85, as [ndlist]")
  nd_block(high_part(nd))

  f = pf.with_outcomes(pd.read_csv(pf.FRAME, dtype={"pred": str}, usecols=[
      "arm", "task", "condition", "graph_id", "density_class", "exact", "hit_cap", "pred"]))
  primer_vs_none(f)
  filler_by_density(f)

  print("[ccoutcome] cycle_check, all responses, p <= .50: % correct / answering no / truncated (n)")
  x = f[(f.task == "cycle_check") & (f.density_class <= .5)]
  x = x.assign(said_no=(x.truncated == 0) & x.pred.str.lower().str.startswith("no", na=False))
  for arm in pf.ARMS:
    print(f"  {SHORT[arm]:6s} " + " | ".join(
        f"{c} {pc(y.correct == 1).strip()} / {pc(y.said_no).strip()} / {pc(y.truncated == 1).strip()} ({len(y)})"
        for c in CONDS for y in [x[(x.arm == arm) & (x.condition == c)]]))

  print("[ccno] cycle_check finished answers: % answering no; of those, n, % mentioning an odd degree sum or "
        "the handshaking lemma / % mentioning return probabilities or self-loops / % calling the graph directed")
  for arm in pf.ARMS:
    print(f"  {SHORT[arm]:6s} " + " | ".join(
        f"{c} {pc(y.said_no).strip()}; {len(z)}: {pc(z.odd).strip()} / {pc(z.ret).strip()} / "
        f"{pc(z.directed).strip()}" for c in CONDS for y in [cell(ccall, arm, c)] for z in [y[y.said_no]]))
  print_paired("answers no", paired(ccall, "said_no", CONDS, pf.ARMS))

  e0 = main_part(ee)
  ne = e0[~e0.edge].assign(bin=lambda d: pd.cut(d.shared, [-1, 0, 1, 3, 10 ** 3], labels=SHARED))
  print("[eeshared] edge_existence non-edges, finished answers, p <= .50: % answered yes by the neighbours the "
        "pair shares (" + " / ".join(SHARED) + "), n in brackets; all primers pooled, then p = .20 alone")
  for arm in pf.ARMS:
    x = ne[ne.arm == arm]
    by = lambda d: " / ".join(f"{pc(d[d.bin == b].said_yes).strip()} ({(d.bin == b).sum()})" for b in SHARED)
    print(f"  {SHORT[arm]:6s} pooled {by(x)} | p=0.2 {by(x[x.density == '0.2'])}")
  print("  shared neighbours of false alarms minus those of correct no's, no primer, permuted within density "
        f"({B_PERM} permutations)")
  rng_p = np.random.default_rng(pf.SEED)
  for arm in pf.ARMS:
    x = ne[(ne.arm == arm) & (ne.condition == "none")]
    d, p = perm_within(x.shared.to_numpy(), x.said_yes.to_numpy(), x.density.to_numpy(), rng_p)
    print(f"    {SHORT[arm]:6s} {d:+.2f} neighbours, p={p:.2g}, {int(x.said_yes.sum())} false alarms of {len(x)}")
  print("  % answered yes on pairs sharing 4 or more, by primer")
  for arm in pf.ARMS:
    x = ne[(ne.arm == arm) & (ne.shared >= 4)]
    print(f"    {SHORT[arm]:6s} " + " | ".join(f"{c} {pc(cell(x, arm, c).said_yes).strip()}" for c in CONDS))
  print_paired("answers yes to a non-edge pair sharing 4 or more", paired(ne[ne.shared >= 4], "said_yes", CONDS,
                                                                           pf.ARMS))

  print("[eewhy] edge_existence false alarms, % by the first reason given: states the pair as an edge (as "
        "[eeclaims]) / claims the other node is listed / neither")
  fa = ee[~ee.edge & ee.said_yes].assign(
      reason=lambda d: np.where(d.states_nonedge, "states", np.where(d.claims_listed, "listed", "neither")))
  for arm in pf.ARMS:
    print(f"  {SHORT[arm]:6s} " + " | ".join(
        f"{label} n={len(y)} {pc(y.reason == 'states').strip()} / {pc(y.reason == 'listed').strip()} / "
        f"{pc(y.reason == 'neither').strip()}" for label, y in (("p<=.50", main_part(fa[fa.arm == arm])),
                                                                ("p>=.65", high_part(fa[fa.arm == arm])))))
  print("  p <= .50, % of false alarms claiming the other node is listed, by primer")
  for arm in pf.ARMS:
    x = main_part(fa)
    print(f"    {SHORT[arm]:6s} " + " | ".join(f"{c} {pc(cell(x, arm, c).reason == 'listed').strip()} "
                                           f"({len(cell(x, arm, c))})" for c in CONDS))
  print_paired("p <= .50, non-edges: claims the other node is listed",
               paired(main_part(ee[~ee.edge]), "claims_listed", CONDS, pf.ARMS))

  print("[eemiss] edge_existence misses (true edges answered no), all densities: n (at p >= .65), % claiming the "
        "other node is not listed / % stating a list for an endpoint that holds the other / % ruling that both "
        "lists must agree")
  ms = ee[ee.edge & ~ee.said_yes]
  for arm in pf.ARMS:
    x = ms[ms.arm == arm]
    print(f"  {SHORT[arm]:6s} n={len(x)} ({len(high_part(x))}) {pc(x.claims_unlisted).strip()} / "
          f"{pc(x.states_pair).strip()} / {pc(x.both_lists).strip()}")

  print("[ndsign] node_degree, wrong finished answers, p <= .50: % below the true degree (n)")
  n0 = main_part(nd)
  for arm in pf.ARMS:
    print(f"  {SHORT[arm]:6s} " + " | ".join(
        f"{c} {pc(y.under).strip()} ({len(y)})" for c in CONDS
        for y in [cell(n0, arm, c)[cell(n0, arm, c).correct == 0]]))
  print_paired("answers below the true degree", paired(n0, "under", CONDS, pf.ARMS))

  print("[cnsource] connected_nodes, finished answers, all primers. Invented neighbours other than q: n, % a "
        "neighbour's neighbour / in the line of node q-1 or q+1 / one off a true neighbour's id (chance %); % of "
        "all invented that are q itself")
  for arm in pf.ARMS:
    x = cninv[cninv.arm == arm]
    o = x[~x.self]
    print(f"  {SHORT[arm]:6s} n={len(o):4d} " + " / ".join(
        f"{pc(o[k]).strip()} ({100 * o[k + '_chance'].mean():.0f})" for k in ("two", "adj", "off"))
        + f"; q itself {pc(x.self).strip()} of {len(x)}")
  print("  missed neighbours: n, % the first on the node's line / the last / below q (chance %)")
  for arm in pf.ARMS:
    x = cnmiss[cnmiss.arm == arm]
    print(f"    {SHORT[arm]:6s} n={len(x):4d} {pc(x['first']).strip()} / {pc(x['last']).strip()} / "
          f"{pc(x.below).strip()} ({100 * x.below_chance.mean():.0f})")
  print_paired("answer includes q itself", paired(cn, "has_self", CONDS, pf.ARMS))
  print_paired("drops the first neighbour on the line", paired(cn, "drops_first", CONDS, pf.ARMS))
  print_paired("drops the last neighbour on the line", paired(cn, "drops_last", CONDS, pf.ARMS))

  print("[ectable] edge_count, answers that list a degree table: % with a wrong degree value (n)")
  t = ec[ec.chain != "no table"]
  for arm in pf.ARMS:
    print(f"  {SHORT[arm]:6s} " + " | ".join(
        f"{c} {pc(cell(t, arm, c).chain == 'values').strip()} ({len(cell(t, arm, c))})" for c in CONDS))

  rng = random.Random(pf.SEED)
  steps = lambda wk, es: "  ".join(f"{a}–{b} {'✓' if frozenset((a, b)) in es else '✗'}"
                                   for a, b in zip(wk, wk[1:]))
  flat = lambda s, n: s[:n].replace("\n", " | ")
  print("[ccsample] rejected named cycles: the text after each")
  for arm, cond, iid, x in rng.sample(samples["rejected"], 5):
    print(f"  {SHORT[arm]} {cond} {iid} [{x['verdict']}] \"{flat(x['quoted'], 80)}\" -> \"{flat(x['after'], 120)}\"")
  for key, label in (("rejected_real", "rejected a real cycle, rests on an invented one"),
                     ("real_not_last", "rests on a real cycle that is not the last one named")):
    print(f"[ccsample] {label}")
    for arm, cond, iid, info in rng.sample(samples[key], min(3, len(samples[key]))):
      print(f"  {SHORT[arm]} {cond} {iid}")
      for x in info:
        tag = "rejected" if x["rejected"] else "kept"
        print(f"    [{x['verdict']}, {tag}] \"{flat(x['quoted'], 80)}\"  {steps(x['walk'], g[iid][0])}")
        print(f"      after: \"{flat(x['after'], 100)}\"")
  print("[ccsample] invented steps whose fake neighbour is in the line of node a-1 or a+1")
  for arm, cond, iid, (a, b), lines in rng.sample(samples["adj"], 3):
    print(f"  {SHORT[arm]} {cond} {iid}: claimed {a}–{b}")
    for n in (a - 1, a, a + 1):
      if n in lines:
        print(f"    {lines[n][:150]}")
  for key, label in (("ee", "edge_existence false alarms that state the queried non-edge"),
                     ("ee_no", "edge_existence answers of no that state the queried non-edge"),
                     ("ee_fab", "edge_existence, thinking arms: a stated list with a non-neighbour")):
    print(f"[clsample] {label}: the statement, then the graph")
    for arm, cond, iid, pair, qt, true in rng.sample(samples[key], min(4, len(samples[key]))):
      print(f"  {SHORT[arm]} {cond} {iid} asked {pair}: \"{flat(qt, 120)}\"\n    graph: {true[:150]}")
  for key, label in (("nd", "node_degree wrong answers with a misread list"),
                     ("nd_ok", "node_degree correct answers with a misread list")):
    print(f"[clsample] {label}: the statement, invented, missed, the graph")
    for arm, cond, iid, t, qt, inv, mis, true in rng.sample(samples[key], min(4, len(samples[key]))):
      print(f"  {SHORT[arm]} {cond} {iid}: \"{flat(qt, 120)}\" invented {inv} missed {mis}\n    graph: {true[:150]}")

  around = lambda a, m, k: flat(a[max(0, m.start() - 100):m.end() + 80] if m else a[-k:], 220)
  print("[ccsample] cycle_check answers of no: the text around the reason cited (or the answer's end)")
  for arm, cond, iid, a in rng.sample(samples["ccno"], min(5, len(samples["ccno"]))):
    print(f"  {SHORT[arm]} {cond} {iid}: \"...{around(a, _ODDSUM.search(a) or _RETURN.search(a), 200)}...\"")
  print("[ncsample] node_count answers of 39: plain 4B, then plain 1.7B listing the ids (start ... end)")
  for arm, listing, k in (("qwen3-4b", False, 3), ("qwen3-1.7b", True, 2)):
    pool = [s for s in samples["nc39"] if s[0] == arm and s[3] == listing]
    for a, cond, iid, _, text in rng.sample(pool, min(k, len(pool))):
      text = text.strip()
      print(f"  {SHORT[a]} {cond} {iid}: \"{flat(text, 140)} ... {flat(text[-100:], 100)}\"")
  for key, label in (("member", "false alarms claiming the other node is listed"),
                     ("miss", "misses: the text around a 'not listed' claim (or the answer's end)")):
    print(f"[eesample] {label}, then the line of the first node asked")
    for arm, cond, iid, pair, a, m, line in rng.sample(samples[key], min(5, len(samples[key]))):
      print(f"  {SHORT[arm]} {cond} {iid} asked {pair}: \"...{around(a, m, 200)}...\"\n    graph: {line[:150]}")

  if args.csv_dir:
    cols = ["source", "part", "arm", "condition", "instance_id", "density", "rests_on",
            "edge_count_argument", "rejected_real", "n_named", "index", "kind", "verdict", "rejected",
            "walk", "quoted", "text_after", "invented_steps", "closing_invented", "from_adjacent_line",
            "off_by_one"]
    path = os.path.join(args.csv_dir, "cycle_claims.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
      wr = csv.DictWriter(fh, fieldnames=cols)
      wr.writeheader()
      wr.writerows(claim_rows)
    other = pd.concat([ee, nd, cn, ec, nc], ignore_index=True)
    other.to_csv(os.path.join(args.csv_dir, "response_claims.csv"), index=False)
    print(f"wrote {len(claim_rows)} rows to {path} and {len(other)} rows to "
          f"{os.path.join(args.csv_dir, 'response_claims.csv')}")


if __name__ == "__main__":
  main()
