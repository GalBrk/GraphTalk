"""What the primers change in how the models answer, and whether accuracy moves
with what a primer says about the task. Read from the raw responses of the
40-node sweep (runs/qwen3-{1.7b,4b}[-think].densfull40*), all 84,000 of them.

Truncated responses are kept everywhere: a response the budget cut off still
shows whether the model was counting or copying. Accuracy follows rule R1
(graphtalk/outcomes.py): a truncated response is never correct, and every
shift is printed with the change in the truncated share beside it ("t").

  [rpcontrast]  discovery: phrases whose share of responses (first 12,000
                characters) moves by 15 points or more, the same way, in three
                or more arms. Not validated; the named patterns were read off it
  [rpvalid]     precision of each text pattern on the hand-labelled sample
                (VALIDATION); a pattern under 0.9 is left out of every block
                after this one. Numeric patterns are checked against the graph
                and need no labels
  [rpchain]     edge_count: the per-node values a response lists, against the
                true degrees, and the first step at which its chain goes wrong
  [rpcycle]     cycle_check: the cycles responses write out, checked against
                the graph
  [rpcopy]      wrong node_degree / connected_nodes answers that belong to
                another node
  [rp<task>]    per task, each pattern's share under none and its shift under
                every primer, per arm, paired on the graphs; exact McNemar, BH
                over every (task, density group, pattern, arm, primer)
  [rptrend]     shifts that go the same way at q < .05 in three or more arms
  [rpassoc]     for each trend: accuracy of finished responses with and without
                the pattern (observational -- the model picks its procedure)
  [rprelation]  per arm, the accuracy effect of each primer against how much it
                tells the graph-blind solver about the task (bar(primer) -
                bar(none), shortcuts_n40_flat.json), and the patterns that moved
                in the cells whose accuracy moved

  PYTHONPATH=. python scripts/response_patterns.py --sample
  PYTHONPATH=. python scripts/response_patterns.py --csv-dir csv2/raw-trends \\
      > csv2/raw-trends/response_patterns.txt
"""
import argparse
import collections
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_raw_frame as brf  # noqa: E402  (Corpus: the true graphs)
import primer_findings as pf  # noqa: E402

from graphtalk import primers, scoring  # noqa: E402

ARMS = pf.ARMS
SHORT = {"qwen3-1.7b": "1.7B", "qwen3-1.7b-think": "1.7B-T", "qwen3-4b": "4B",
         "qwen3-4b-think": "4B-T"}
CONDS = ["filler"] + pf.PRIMERS           # each compared against none
TASKS = ["node_count", "node_degree", "connected_nodes", "edge_count",
         "edge_existence", "cycle_check"]
CONSTANT = ("node_count", "cycle_check")   # gold is 40 / "yes" on every graph
HI_TASKS = ("node_degree", "edge_existence")
VALIDATION = "csv2/raw-trends/response_pattern_validation.csv"
MIN_PRECISION = 0.9
PER_ARM, N_SAMPLE = 5, 20
CAP = 12000                                # characters read by the discovery pass

# ------------------------------------------------------------------ text patterns
# Each finder returns the position of its evidence in the text, or None.


def _rx(pattern):
  rx = re.compile(pattern, re.I)

  def find(text, targets):
    m = rx.search(text)
    return m.start() if m else None
  return find


def _line(t):
  return rf"[Nn]ode {t}\**\s+is connected to"


def _target(template):
  def find(text, targets):
    m = re.search(template(targets[0]), text)
    return m.start() if m else None
  return find


def _lookup(both):
  def find(text, targets):
    hits = [m for m in (re.search(_line(t), text) for t in targets) if m]
    return min(m.start() for m in hits) if hits and (not both or len(hits) == len(targets)) else None
  return find


def _route(kind):
  def find(text, targets):
    if pf.route(text, targets[0]) != kind:
      return None
    m = pf._ENUM_LINE.search(text) if kind == "enumerate" else None
    return m.start() if m else 0
  return find


def _conflict(text, targets):
  if not pf.reports_discrepancy(text):
    return None
  return pf._DISCREPANCY.search(text).start()


_PAIR = re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)")


def _edge_pairs(text, targets):
  found = list(_PAIR.finditer(text))
  return found[0].start() if len(found) >= 10 else None


_DISMISS = (r"\b(?:not|n't)\s+(?:directly\s+|really\s+|actually\s+|necessarily\s+|"
            r"seem\s+(?:to\s+be\s+)?)?(?:relevant|needed|necessary|required|useful|"
            r"helpful|related)\b|\birrelevant\b|red herring|distract")

ALL = tuple(TASKS)
TEXT = {  # name: (tasks, meaning, finder)
    "dismisses": (ALL, "says some given information is irrelevant or not needed",
                  _rx(_DISMISS)),
    "names_clustering": (ALL, "mentions clustering coefficients",
                         _rx(r"clustering coefficient")),
    "names_rwse": (ALL, "mentions return probabilities, random walks or Markov chains",
                   _rx(r"return probabilit|random[- ]walk|markov")),
    "names_components": (ALL, "mentions connected components",
                         _rx(r"\bcomponents?\b")),
    "names_degree": (tuple(t for t in TASKS if t != "node_degree"),
                     "reasons with node degrees", _rx(r"\bdegrees?\b")),
    "degree_sum": (("edge_count", "cycle_check"),
                   "sums the degrees and halves the sum (handshake)",
                   _rx(brf._MARKERS["uses_degree_sum"].pattern)),
    "edge_pairs": (("edge_count",), "lists edges as node pairs (ten or more)",
                   _edge_pairs),
    "reports_conflict": (("node_degree", "edge_count", "connected_nodes"),
                         "reports a conflict between values", _conflict),
    "retrieve": (("node_degree",), "answers without restating the queried node's line",
                 _route("retrieve")),
    "enumerate": (("node_degree",), "lists the queried node's neighbours one per line",
                  _route("enumerate")),
    "cites_degree": (("node_degree",),
                     "states the queried node's degree in the primer's words",
                     _target(lambda t: rf"[Nn]ode {t}\**\s+has degree")),
    "restates_line": (("connected_nodes",), "restates the queried node's line",
                      _target(_line)),
    "lookup_any": (("edge_existence",), "restates the line of at least one endpoint",
                   _lookup(False)),
    "lookup_both": (("edge_existence",), "restates the lines of both endpoints",
                    _lookup(True)),
    "hedges": (("edge_existence",), "judges the edge by likelihood",
               _rx(r"\b(?:likely|unlikely|probably)\b")),
    "tree_bound": (("cycle_check",), "argues from the edge count against n - 1 (a tree)",
                   _rx(r"\bn\s*-\s*1\b|\b40\s*-\s*1\b|\btrees?\b|more edges than|"
                       r"at least as many edges")),
    "id_range": (("node_count",), "reads the count off the range of node ids",
                 _rx(r"\b0\s*(?:to|through|-|–|—)\s*39\b|\b39\s*-\s*0\s*\+\s*1\b|"
                     r"(?:highest|largest|maximum) (?:node )?(?:id|number|index|label)")),
}

NUMERIC = {  # name: (tasks, meaning); checked against the graph, not labelled
    "quotes_primer_value": (ALL, "quotes a decimal the clustering or rwse primer prints "
                                 "for this graph (under any condition)"),
    "degree_table": (("edge_count",), "lists a value for 30 or more of the 40 nodes"),
    "degree_table_right": (("edge_count",), "lists all 40 nodes, every value the true degree"),
    "cycle_named": (("cycle_check",), "writes out a closed walk"),
    "cycle_real": (("cycle_check",), "writes out a closed walk that is a cycle of the graph"),
}

# ------------------------------------------------------------------ numeric detectors
_TABLE = [
    re.compile(r"\bnode\s+(\d+)\s*(?:\([^)\n]*\))?\s*[:=\-–]\s*(\d+)\b(?!\s*[,.]\s*\d)", re.I),
    re.compile(r"\bnode\s+(\d+)\s+has\s+(?:a\s+)?(?:degree\s+(?:of\s+)?)?(\d+)\b"
               r"(?!\s*[,.]\s*\d)", re.I),
]
_HALF = re.compile(r"\\d?frac\{\s*(\d+)\s*\}\{\s*2\s*\}\s*=\s*(?:\\boxed\{)?\s*(\d+)"
                   r"|(\d+)\s*(?:/|÷|divided by)\s*2\s*=\s*(?:\\boxed\{)?\s*(\d+)", re.I)
_WALK = re.compile(r"\d+(?:\s*(?:→|->|⟶|=>|–|—|\\to\b|\\rightarrow)\s*\d+|-\d+){2,}")
_DEC = re.compile(r"(?<![\d.])\d\.\d\d(?!\d)")


def degree_table(text, n):
  """The per-node values a response lists ('Node 3: 12', 'Node 3 has 12
  connections', a '| 3 | ... | 12 |' table row) as {node: value}. A node listed
  more than once keeps its last value; ids outside 0..n-1 are ignored."""
  text = text.replace("*", "")
  found = [(m.start(), int(m.group(1)), int(m.group(2)))
           for rx in _TABLE for m in rx.finditer(text)]
  pos = 0
  for line in text.splitlines(keepends=True):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if line.lstrip().startswith("|") and len(cells) >= 2 and cells[0].isdigit() \
        and cells[-1].isdigit():
      found.append((pos, int(cells[0]), int(cells[-1])))
    pos += len(line)
  return {k: v for _, k, v in sorted(found) if k < n}


def edge_chain(text, degrees, pred, truncated):
  """Where an edge_count response's handshake chain first goes wrong: 'no table'
  (values for under 3/4 of the nodes), 'values', 'nodes' (some missing), 'sum'
  (the stated sum is not the sum of the listed values), 'halving', 'answer'; else
  'cut' (truncated, right as far as it got), 'unparsed' (finished, no halving
  written as S/2 = H) or 'right'."""
  table = degree_table(text, len(degrees))
  if len(table) < 0.75 * len(degrees):
    return "no table"
  if any(degrees[k] != v for k, v in table.items()):
    return "values"
  if len(table) < len(degrees):
    return "nodes"
  halves = _HALF.findall(text.replace("*", ""))
  if not halves:
    return "cut" if truncated else "unparsed"
  a, b, c, d = halves[-1]
  s, h = int(a or c), int(b or d)
  if s != sum(table.values()):
    return "sum"
  if 2 * h != s:
    return "halving"
  if truncated:
    return "cut"
  return "right" if pred == h else "answer"


def named_cycles(text):
  """The closed walks a response writes out ('0 → 5 → 7 → 0'), as node lists."""
  walks = ([int(x) for x in re.findall(r"\d+", m.group(0))]
           for m in _WALK.finditer(text.replace("*", "")))
  return [w for w in walks if w[0] == w[-1]]


def is_cycle(walk, g):
  """A closed walk through three or more distinct nodes, every step an edge."""
  inner = walk[:-1]
  return (walk[0] == walk[-1] and len(inner) >= 3 and len(set(inner)) == len(inner)
          and all(g.has_edge(a, b) for a, b in zip(walk, walk[1:])))


# ------------------------------------------------------------------ discovery
_TOK = re.compile(r"<dec>|<num>|[a-z]+|[+=→|/×*√]")


def grams(text, nmax=3):
  """The set of 1..nmax-grams in text, numbers normalized to <num>/<dec>."""
  t = re.sub(r"\d+", "<num>", re.sub(r"\d+\.\d+", "<dec>", text[:CAP].lower()))
  w = _TOK.findall(t)
  return {" ".join(w[i:i + n]) for n in range(1, nmax + 1) for i in range(len(w) - n + 1)}


def phrase_shifts(docs, arms, cond, min_shift=0.15, min_arms=3):
  """Phrases whose share of responses moves by min_shift or more the same way in
  min_arms arms, cond against none; docs[(arm, condition)] is a list of phrase
  sets. Ranked by the min_arms-th largest shift, longer phrases first on ties; a
  phrase inside a higher-ranked one with a shift within .05 of it is dropped.
  Returns [(phrase, shifts)]."""
  share = {}
  for arm in arms:
    for c in ("none", cond):
      ds = docs[(arm, c)]
      share[(arm, c)] = {w: k / len(ds) for w, k in
                         collections.Counter(w for s in ds for w in s).items()}
  found = []
  for w in set().union(*share.values()):
    d = [share[(a, cond)].get(w, 0) - share[(a, "none")].get(w, 0) for a in arms]
    for sign in (1, -1):
      s = sorted((sign * x for x in d), reverse=True)[min_arms - 1]
      if s >= min_shift:
        found.append((sign * s, w, d))
  found.sort(key=lambda x: (-abs(x[0]), -len(x[1].split()), x[1]))
  keep = []
  for score, w, d in found:
    if not any(f" {w} " in f" {k} " and abs(score - ks) < .05 for ks, k, _ in keep):
      keep.append((score, w, d))
  return [(w, d) for _, w, d in keep]


def contrast(d, out_rows):
  print("[rpcontrast] discovery, main sweep: phrases (numbers as <num>/<dec>) whose share "
        "of responses moves 15+ points the same way in 3+ arms, primer vs none; shifts "
        "per arm " + " / ".join(SHORT[a] for a in ARMS) + ", top 8")
  for task in TASKS:
    m = d[(d.task == task) & d.density_class.isin(pf.DENS4)]
    docs = {(a, c): [grams(t) for t in g.text] for (a, c), g in m.groupby(["arm", "condition"])}
    for c in CONDS:
      found = phrase_shifts(docs, ARMS, c)
      print(f"  {task} / {c}: {len(found)} phrases; " + " | ".join(
          f"'{w}' " + " ".join(f"{x:+.2f}" for x in s) for w, s in found[:8]))
      out_rows += [dict(task=task, condition=c, phrase=w, **{a: x for a, x in zip(ARMS, s)})
                   for w, s in found]


# ------------------------------------------------------------------ loading
def load():
  """The frame with outcomes, text, targets and every pattern column."""
  f = pf.with_outcomes(pd.read_csv(pf.FRAME, dtype={"pred": str, "gold": str}))
  runs = {a: pf.load_runs(f"runs/{a}.densfull40*.shard*.jsonl") for a in ARMS}
  corpus, graphs = brf.Corpus(), {}
  rows = []
  for r in f.itertuples():
    text = runs[r.arm][(r.instance_id, r.condition)]["response"] or ""
    key = (r.density_class, r.index)
    if key not in graphs:
      g = corpus.graph(*key)[0]
      dec = set()
      for c in ("clustering", "rwse"):
        dec |= set(_DEC.findall(primers.build_primer(g, c)))
      graphs[key] = (g, dict(g.degree()), dec)
    g, degrees, dec = graphs[key]
    targets = [int(t) for t in corpus.row(*key, r.task)["targets"]]
    row = {"text": text, "targets": targets,
           "quotes_primer_value": int(bool(set(_DEC.findall(text)) & dec))}
    for name, (tasks, _, find) in TEXT.items():
      if r.task in tasks:
        row[name] = int(find(text, targets) is not None)
    if r.task == "edge_count":
      t = degree_table(text, len(degrees))
      right = [degrees[k] == v for k, v in t.items()]
      row["degree_table"] = int(len(t) >= 30)
      row["degree_table_right"] = int(len(t) == len(degrees) and all(right))
      row["node_acc"] = np.mean(right) if len(t) >= 30 else np.nan
      pred = None if r.hit_cap or pd.isna(r.pred) else int(float(r.pred))
      row["chain"] = edge_chain(text, degrees, pred, bool(r.hit_cap))
    if r.task == "cycle_check":
      walks = named_cycles(text)
      row["cycle_named"] = int(bool(walks))
      row["cycle_real"] = int(any(is_cycle(w, g) for w in walks))
    if r.task in ("node_degree", "connected_nodes"):
      row["copied"], row["copy_chance"] = copy_kind(r, g, degrees, targets[0])
    rows.append(row)
  return f.join(pd.DataFrame(rows, index=f.index))


def copy_kind(r, g, degrees, t):
  """For a finished wrong answer, (kind, chance). node_degree: 'near' when it
  equals the degree of t-1 or t+1, else 'other', and the chance of that match
  given how often the other nodes share the value (as primer_findings.leakage).
  connected_nodes: the offset u - t of another node u whose neighbour set it
  equals, else 'other'. (NaN, NaN) for any other response."""
  if r.hit_cap or r.exact or pd.isna(r.pred):
    return np.nan, np.nan
  if r.task == "node_degree":
    v = int(float(r.pred))
    near = [u for u in (t - 1, t + 1) if u in degrees]
    q = np.mean([degrees[u] == v for u in degrees if u != t and u not in near])
    return ("near" if any(degrees[u] == v for u in near) else "other",
            1 - (1 - q) ** len(near))
  s = pf.neighbour_set(r.pred)
  hits = [u - t for u in g if u != t and s == set(g[u])]
  return (min(hits, key=abs) if hits else "other"), np.nan


# ------------------------------------------------------------------ validation
def write_sample(d):
  """N_SAMPLE seeded positives per text pattern, PER_ARM from each arm first."""
  rng = np.random.default_rng(pf.SEED)
  rows = []
  for name, (tasks, meaning, find) in TEXT.items():
    pos = d[d.task.isin(tasks) & (d[name] == 1)]
    picks = []
    for arm in ARMS:
      a = pos[pos.arm == arm].index
      picks += list(rng.choice(a, min(PER_ARM, len(a)), replace=False))
    rest = pos.index.difference(picks)
    picks += list(rng.choice(rest, min(N_SAMPLE - len(picks), len(rest)), replace=False))
    for i in picks:
      r = d.loc[i]
      if name == "retrieve":        # evidence is an absence: show the answer itself
        ex = r.text[:200] + " [...] " + r.text.split("</think>")[-1][:700]
      else:
        at = find(r.text, r.targets)
        ex = r.text[max(0, at - 200):at + 300]
      rows.append(dict(pattern=name, meaning=meaning, arm=r.arm, task=r.task,
                       condition=r.condition, instance_id=r.instance_id,
                       truncated=int(r.hit_cap), targets=" ".join(map(str, r.targets)),
                       excerpt=" ".join(ex.split()), label=""))
  pd.DataFrame(rows).to_csv(VALIDATION, index=False)
  print(f"wrote {len(rows)} samples to {VALIDATION}; fill in label (1 = the excerpt "
        f"shows the meaning, 0 = it does not)")


def validation():
  v = (pd.read_csv(VALIDATION) if os.path.exists(VALIDATION)
       else pd.DataFrame(columns=["pattern", "label"]))
  print(f"[rpvalid] text patterns: labelled precision on {VALIDATION} (seeded, "
        f"{PER_ARM} per arm first); kept at {MIN_PRECISION} or more")
  keep = []
  for name, (_, meaning, _) in TEXT.items():
    s = v[(v.pattern == name) & v.label.notna()]
    if not len(s):
      print(f"  {name:18s} not labelled -- left out ({meaning})")
      continue
    p = s.label.mean()
    print(f"  {name:18s} {int(s.label.sum())}/{len(s)} = {p:.2f} "
          f"{'kept' if p >= MIN_PRECISION else 'LEFT OUT'} ({meaning})")
    if p >= MIN_PRECISION:
      keep.append(name)
  return keep


# ------------------------------------------------------------------ statistics
def boot(x, strata):
  """95% interval of 100 * mean(x), resampling within each stratum."""
  rng = np.random.default_rng(pf.SEED)
  groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
  idx = np.concatenate([g[rng.integers(0, len(g), (pf.B, len(g)))] for g in groups], axis=1)
  return np.percentile(100 * x[idx].mean(1), [2.5, 97.5])


def paired(j, col):
  a, b = j[col + "_a"].to_numpy(float), j[col + "_b"].to_numpy(float)
  m = scoring.mcnemar(a.astype(bool), b.astype(bool))
  lo, hi = boot(b - a, j.index.get_level_values(0).to_numpy())
  return {"n": len(j), "share_none": 100 * a.mean(), "share": 100 * b.mean(),
          "shift": 100 * (b.mean() - a.mean()), "lo": lo, "hi": hi,
          "a_only": m["b"], "b_only": m["c"], "p": m["p_value"],
          "d_truncated": 100 * (j.truncated_b.mean() - j.truncated_a.mean()),
          "flag": max(j.truncated_a.mean(), j.truncated_b.mean()) >= .15}


def groups(task):
  return [("main", pf.DENS4)] + ([("hi", pf.DENSHI)] if task in HI_TASKS else [])


def spec(name):
  return TEXT[name][:2] if name in TEXT else NUMERIC[name]


def pattern_shifts(d, names):
  rows = []
  for task in TASKS:
    for arm in ARMS:
      dt = d[(d.arm == arm) & (d.task == task)]
      for name in (n for n in names if task in spec(n)[0]):
        for grp, dens in groups(task):
          for c in CONDS:
            rows.append(dict(task=task, group=grp, pattern=name, arm=arm, condition=c,
                             **paired(pf.pairs(dt, arm, task, "none", c, dens), name)))
  r = pd.DataFrame(rows)
  r["q"] = pf.bh(r.p.to_numpy())
  return r


def star(q):
  return "*" if q < .05 else ""


def print_tasks(r, trend_keys):
  for task in TASKS:
    for grp, dens in groups(task):
      tag = "rp" + task.replace("_", "") + ("hi" if grp == "hi" else "")
      where = ("main sweep, p<=.50" if grp == "main" else
               "high-density extension, p>=.65 (plain arms at 2048 tokens)")
      print(f"[{tag}] {task}, {where}: share of all responses (truncated included) "
            f"with each pattern under none, per arm; then the shift in points under each "
            f"primer, * q<.05, (t ...) the change in the truncated share, TREND = 3+ arms "
            f"the same way at q<.05")
      x = r[(r.task == task) & (r.group == grp)]
      for name in x.pattern.unique():
        y = x[x.pattern == name]
        none = y[y.condition == CONDS[0]].set_index("arm").share_none
        print(f"  {name} ({spec(name)[1]}): none " + " / ".join(
            f"{SHORT[a]} {none[a]:.0f}%" for a in ARMS))
        for c in CONDS:
          z = y[y.condition == c].set_index("arm")
          mark = "  TREND" if (task, grp, c, name) in trend_keys else ""
          print(f"    {c:10s} " + " | ".join(
              f"{pf.f1(z['shift'][a])}{star(z.q[a])} (t {pf.f1(z.d_truncated[a])})"
              for a in ARMS) + mark)


def find_trends(r):
  """{(task, group, primer, pattern): (sign, arms not moving that way)} for shifts
  that go the same way at q < .05 in three or more arms."""
  out = {}
  for (task, grp, c, name), g in r.groupby(["task", "group", "condition", "pattern"]):
    g = g.set_index("arm")
    for sign in (1, -1):
      with_it = [a for a in ARMS if g.q[a] < .05 and sign * g["shift"][a] > 0]
      if len(with_it) >= 3:
        out[(task, grp, c, name)] = (sign, [a for a in ARMS if a not in with_it])
  return out


def print_trends(r, trend_keys):
  print("[rptrend] pattern shifts that go the same way at q<.05 in 3+ arms (task, "
        "density group, primer, pattern: shift per arm " + " / ".join(SHORT[a] for a in ARMS)
        + "; arms not moving that way)")
  for (task, grp, c, name), (sign, rest) in sorted(trend_keys.items()):
    g = r[(r.task == task) & (r.group == grp) & (r.condition == c)
          & (r.pattern == name)].set_index("arm")
    print(f"  {task} {grp} {c}: {name} {'up' if sign > 0 else 'down'} " + " / ".join(
        f"{pf.f1(g['shift'][a])}{star(g.q[a])}" for a in ARMS)
        + (f"; not: {', '.join(SHORT[a] for a in rest)}" if rest else ""))


def associations(d, trend_keys):
  print("[rpassoc] for each trend, finished responses under that primer: accuracy with "
        "the pattern vs without (n), where both have 10+; observational")
  for (task, grp, c, name) in sorted(trend_keys):
    dens = dict(groups(task))[grp]
    cells = []
    for a in ARMS:
      x = d[(d.arm == a) & (d.task == task) & (d.condition == c) & (d.hit_cap == 0)
            & d.density_class.isin(dens)]
      on, off = x[x[name] == 1], x[x[name] == 0]
      if len(on) >= 10 and len(off) >= 10:
        cells.append(f"{SHORT[a]} {100 * on.exact.mean():.0f}% ({len(on)}) vs "
                     f"{100 * off.exact.mean():.0f}% ({len(off)})")
    if cells:
      print(f"  {task} {grp} {c} {name}: " + "; ".join(cells))


# ------------------------------------------------------------------ numeric blocks
def chain_block(d):
  order = ["right", "values", "nodes", "sum", "halving", "answer", "cut", "unparsed"]
  print("[rpchain] edge_count, main sweep, all responses: share listing values for 30+ "
        "nodes; among those, per-node accuracy against the true degrees and the share "
        "with all 40 right; then where the chain first fails, % of responses with a "
        "table: " + " / ".join(order))
  for a in ARMS:
    for c in ["none"] + CONDS:
      x = d[(d.arm == a) & (d.task == "edge_count") & (d.condition == c)
            & d.density_class.isin(pf.DENS4)]
      t = x[x.degree_table == 1]
      share = t.chain.value_counts(normalize=True)
      print(f"  {SHORT[a]:6s} {c:10s} table {100 * len(t) / len(x):3.0f}% | per-node right "
            f"{100 * t.node_acc.mean():5.1f}% | all 40 right "
            f"{100 * t.degree_table_right.mean():3.0f}% | "
            + " / ".join(f"{100 * share.get(k, 0):.0f}" for k in order))


def cycle_block(d):
  print("[rpcycle] cycle_check, main sweep, all responses, per primer: % writing out a "
        "closed walk / % whose walk is a cycle of the graph / % whose walks are none")
  for a in ARMS:
    x = d[(d.arm == a) & (d.task == "cycle_check") & d.density_class.isin(pf.DENS4)]
    parts = []
    for c in ["none"] + CONDS:
      y = x[x.condition == c]
      named, real = y.cycle_named == 1, y.cycle_real == 1
      parts.append(f"{c} {100 * named.mean():.0f}/{100 * real.mean():.0f}/"
                   f"{100 * (named & ~real).mean():.0f}")
    print(f"  {SHORT[a]:6s} " + " | ".join(parts))


def copy_block(d):
  print("[rpcopy] finished wrong answers. node_degree, all densities: equal to the degree "
        "of node t-1 or t+1 (the neighbouring primer lines), against the count expected "
        "from how often other nodes share that degree. connected_nodes, main sweep: "
        "equal to another node's neighbour set (of which t-1 or t+1)")
  for a in ARMS:
    x = d[(d.arm == a) & (d.task == "node_degree") & d.copied.notna()]
    print(f"  node_degree {SHORT[a]:6s} " + " | ".join(
        f"{c} {int((y.copied == 'near').sum())}/{len(y)} vs {y.copy_chance.sum():.1f}"
        for c in ["none"] + CONDS for y in [x[x.condition == c]]))
  for a in ARMS:
    x = d[(d.arm == a) & (d.task == "connected_nodes") & d.copied.notna()
          & d.density_class.isin(pf.DENS4)]
    print(f"  connected_nodes {SHORT[a]:6s} " + " | ".join(
        f"{c} {int((y.copied != 'other').sum())} ({int(y.copied.isin([-1, 1]).sum())})/{len(y)}"
        for c in ["none"] + CONDS for y in [x[x.condition == c]]))


# ------------------------------------------------------------------ primer-task relation
CLASSES = ["answer-carrying", "informative side", "uninformative side", "control",
           "constant task"]


def relation_class(task, cond, bar, dbar):
  """What the primer gives a graph-blind solver for this task: the answer (it
  scores 1.0 from the primer, n40-sweep.md section 2), some information (moves it
  by .05 or more either way), or none."""
  if task in CONSTANT:
    return "constant task"
  if cond == "filler":
    return "control"
  if bar >= 0.999:
    return "answer-carrying"
  return "informative side" if abs(dbar) >= 0.05 else "uninformative side"


def relation(d, r):
  bars = json.load(open(pf.BARS))
  rows = []
  for arm in ARMS:
    for task in TASKS:
      dt = d[(d.arm == arm) & (d.task == task)]
      for c in CONDS:
        e = paired(pf.pairs(dt, arm, task, "none", c, pf.DENS4), "correct")
        vf = (paired(pf.pairs(dt, arm, task, "filler", c, pf.DENS4), "correct")["shift"]
              if c != "filler" else np.nan)
        dbar = bars[f"{task}/{c}"] - bars[f"{task}/none"]
        rows.append(dict(arm=arm, task=task, condition=c, dbar=dbar,
                         relation=relation_class(task, c, bars[f"{task}/{c}"], dbar),
                         vs_filler=vf, **e))
  x = pd.DataFrame(rows)
  x["q"] = pf.bh(x.p.to_numpy())
  print("[rprelation] accuracy (correct share of all responses; truncated is never "
        "correct) against none, main sweep, 400 paired graphs, by what the primer gives "
        "the graph-blind solver for the task: dbar = bar(primer) - bar(none). Class means "
        "leave out cells with 15%+ truncated on either side (~); * q<.05, BH over all "
        f"{len(x)} (arm, task, primer)")
  for arm in ARMS:
    y = x[x.arm == arm]
    parts = []
    for cls in CLASSES:
      z = y[(y.relation == cls) & ~y.flag]
      if len(z):
        parts.append(f"{cls} {len(z)} cells, mean {pf.f1(z['shift'].mean())} (up "
                     f"{int(((z.q < .05) & (z['shift'] > 0)).sum())}, down "
                     f"{int(((z.q < .05) & (z['shift'] < 0)).sum())})"
                     + (f", vs filler {pf.f1(z.vs_filler.mean())}" if cls != "control" else ""))
    print(f"  {SHORT[arm]}: " + "; ".join(parts) + f"; flagged {int(y.flag.sum())}")
  print("  per task and primer (dbar, class): effect per arm; then, where accuracy moved, "
        "the three largest pattern shifts at q<.05 in the same cell")
  for task in TASKS:
    for c in CONDS:
      z = x[(x.task == task) & (x.condition == c)].set_index("arm")
      print(f"  {task}/{c} ({z.dbar.iloc[0]:+.2f}, {z.relation.iloc[0]}): " + " | ".join(
          f"{SHORT[a]} {pf.f1(z['shift'][a])}{star(z.q[a])}{'~' if z.flag[a] else ''}"
          f" (t {pf.f1(z.d_truncated[a])})" for a in ARMS))
      moved = []
      for a in ARMS:
        if z.q[a] < .05:
          m = r[(r.arm == a) & (r.task == task) & (r.condition == c) & (r.group == "main")
                & (r.q < .05)]
          m = m.loc[m["shift"].abs().sort_values(ascending=False).index[:3]]
          if len(m):
            moved.append(f"{SHORT[a]}: " + ", ".join(
                f"{p} {pf.f1(s)}" for p, s in zip(m.pattern, m["shift"])))
      if moved:
        print("      patterns: " + "; ".join(moved))
  return x


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--csv-dir", help="also write response_pattern_cells.csv, "
                  "response_contrasts.csv and response_relation.csv here")
  ap.add_argument("--sample", action="store_true",
                  help=f"write the validation sample to {VALIDATION} and stop")
  args = ap.parse_args()
  sys.stdout.reconfigure(encoding="utf-8")
  d = load()
  if args.sample:
    write_sample(d)
    return
  phrases = []
  contrast(d, phrases)
  names = validation() + list(NUMERIC)
  chain_block(d)
  cycle_block(d)
  copy_block(d)
  lean = d.drop(columns=["text", "targets"])
  r = pattern_shifts(lean, names)
  keys = find_trends(r)
  print_tasks(r, keys)
  print_trends(r, keys)
  associations(d, keys)
  rel = relation(lean, r)
  if args.csv_dir:
    r.to_csv(os.path.join(args.csv_dir, "response_pattern_cells.csv"), index=False)
    pd.DataFrame(phrases).to_csv(os.path.join(args.csv_dir, "response_contrasts.csv"),
                                 index=False)
    rel.to_csv(os.path.join(args.csv_dir, "response_relation.csv"), index=False)


if __name__ == "__main__":
  main()
