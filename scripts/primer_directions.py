"""Nine further questions about what the primers do to the 40-node sweep's
responses (runs/qwen3-{1.7b,4b}[-think].densfull40*), past accuracy and the
reasoning patterns of scripts/response_patterns.py. Exploratory: the text
measures written here (stated values, quotes, candidate answers, revision words)
are not hand-validated; the validated response_patterns markers they build on
are reports_conflict and the numeric detectors. Accuracy follows rule R1
(graphtalk/outcomes.py): a truncated response is never correct. Every primer
value is read from the primer text itself, exactly as the model saw it.

  [pdsource]    A1 when a response reports a conflict under degree/all, which
                value its final answer takes: the primer's, one it computed
                itself, or neither; and how often a conflict ends truncated
  [pdmisread]   A2 clustering coefficients and return probabilities quoted in
                responses, against the primer: right, the value of the
                neighbouring line (k-1, k+1), or another value; with the
                neighbouring-line count expected by chance. Degree is left out
                (models state degrees they counted in the same words); its copying
                is measured by response_patterns' [rpchain] and [rpcopy]
  [pdcommit]    A3 thinking arms: where the first candidate answer appears in the
                trace, whether it is the final answer, how many distinct values
                are tried, and revision words per 1,000 characters
  [pdposition]  A4 node_degree: the primer's effect on queried nodes in the first
                quarter of the lines minus the last quarter (difference in
                differences against none, since the encoding is in node order too)
  [pdconsist]   B5 on the same queried node t, is the edge_existence answer for
                (t, b) consistent with b being in the model's own neighbour list
                for t (connected_nodes)?
  [pdheur]      B6 edge_existence false alarms and misses on pairs whose endpoints
                print a high vs low degree / clustering / return probability: does
                a primer that shows the feature widen the gap against none?
  [pddensity]   B7 each primer's node_degree effect per density, against how far
                the printed rwse or clustering values pin a node's degree
  [pdnode]      B8 effects by the queried node's degree rank within its graph
  [pderror]     B9 direction of wrong node_degree / edge_count answers

  PYTHONPATH=. python scripts/primer_directions.py > csv2/raw-trends/primer_directions.txt
"""
import collections
import functools
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_raw_frame as brf  # noqa: E402  (Corpus: the true graphs)
import primer_findings as pf  # noqa: E402
import response_patterns as rp  # noqa: E402

from graphtalk import primers, scoring, shortcuts  # noqa: E402

ARMS, SHORT, CONDS = rp.ARMS, rp.SHORT, rp.CONDS
THINK = ["qwen3-1.7b-think", "qwen3-4b-think"]
MAIN, HI = pf.DENS4, pf.DENSHI
DENS = {"node_degree": MAIN + HI, "edge_existence": MAIN + HI}   # other tasks: MAIN

# ------------------------------------------------------------------ primer values
FEATURE = {"degree": 0, "clustering": 1, "rwse": 2}      # index into primer_values' tuples
SHOWN = {"degree": ("degree", "all"), "clustering": ("clustering", "all"),
         "rwse": ("rwse", "all")}


def primer_values(all_text):
  """{node: (degree, clustering, rw2, rw3)} as the primer prints them (strings),
  read by graphtalk.shortcuts.parse_primer, which raises on any sentence it cannot
  account for. The `all` primer prints the same values as each single-feature
  primer (one renderer)."""
  p = shortcuts.parse_primer(all_text)
  return {k: (str(p.degree[k]), f"{p.clustering[k]:.2f}", f"{p.rwse[k][2]:.2f}",
              f"{p.rwse[k][3]:.2f}") for k in p.degree}


@functools.lru_cache(maxsize=None)
def _corpus():
  return brf.Corpus()      # built on first use, not at import


@functools.lru_cache(maxsize=None)
def facts(dens, idx):
  """(graph, primer values) for one graph; every node has a primer line."""
  g = _corpus().graph(dens, idx)[0]
  values = primer_values(primers.build_primer(g, "all"))
  assert len(values) == g.number_of_nodes(), (dens, idx)
  return g, values


def to_int(p):
  try:
    return int(float(p))
  except (TypeError, ValueError):
    return None


def boot_gap(x, g1, g2, strata):
  """95% interval, in points, of mean(x[g1]) - mean(x[g2]), resampling rows within
  each stratum (pf.SEED, pf.B draws). x is a per-graph paired difference; rows in
  neither group or with a missing x are dropped first, as gap() ignores them.
  Resamples that leave a group empty are skipped."""
  x, g1, g2 = (np.asarray(v, float) for v in (x, g1, g2))
  keep = ~np.isnan(x) & ((g1 == 1) | (g2 == 1))
  x, g1, g2, strata = x[keep], g1[keep], g2[keep], np.asarray(strata)[keep]
  rng = np.random.default_rng(pf.SEED)
  groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
  idx = np.concatenate([g[rng.integers(0, len(g), (pf.B, len(g)))] for g in groups], axis=1)
  xs, a, b = x[idx], g1[idx], g2[idx]
  with np.errstate(invalid="ignore", divide="ignore"):
    v = 100 * ((xs * a).sum(1) / a.sum(1) - (xs * b).sum(1) / b.sum(1))
  return np.nanpercentile(v, [2.5, 97.5])


def gap(x, g1, g2):
  x, g1, g2 = (np.asarray(v, float) for v in (x, g1, g2))
  return 100 * (np.nanmean(x[g1 == 1]) - np.nanmean(x[g2 == 1]))


def fmt_gap(x, g1, g2, strata):
  lo, hi = boot_gap(x, g1, g2, strata)
  return f"{pf.f1(gap(x, g1, g2))} [{pf.f1(lo)}, {pf.f1(hi)}]"


# ------------------------------------------------------------------ A1 source in a conflict
# The number must follow the cue directly: "count node 7's neighbours" states no count.
_COUNTED = re.compile(r"\b(?:I count(?:ed)?|count(?:ed)? (?:is|of|=)|that(?:'s| is)|"
                      r"total(?: of)?|there are|I (?:see|get|find|found)|comes to)\s*(\d+)"
                      r"\b(?![.,]\d)|\b(\d+)\s+(?:nodes|neighbou?rs|connections|edges)\b", re.I)


def stated_values(text):
  """Integers a response states as a count or a result ('I count 22', '22
  connections', 'S / 2 = H'), whatever it finally answers."""
  t = text.replace("*", "")
  out = {int(a or b) for a, b in _COUNTED.findall(t)}
  return out | {int(b or d) for a, b, c, d in rp._HALF.findall(t)}


def source(pred, gold, text):
  """For a finished, parsed answer: 'stated' when it is the primer's value (the
  gold, under degree/all), 'own' when it is another value the response computed,
  else 'other'. None when there is no parsed answer."""
  if pred is None:
    return None
  if pred == gold:
    return "stated"
  return "own" if pred in stated_values(text) else "other"


def source_block(d):
  print("[pdsource] A1, degree and all: share of responses reporting a conflict "
        "(response_patterns.reports_conflict, validated 19/20; under none, where no "
        "degree is stated, for scale); truncated share with vs without a conflict; "
        "finished answers of responses with a conflict: the primer's value / a value "
        "the response computed itself / other")
  for task in ("node_degree", "edge_count"):
    for arm in ARMS:
      x = d[(d.arm == arm) & (d.task == task) & d.density_class.isin(DENS.get(task, MAIN))]
      floor = 100 * x[x.condition == "none"].reports_conflict.mean()
      for c in ("degree", "all"):
        y = x[x.condition == c]
        conf = y.reports_conflict == 1
        fin = y[conf & (y.hit_cap == 0)]
        kinds = collections.Counter(
            source(to_int(p), to_int(g), t) for p, g, t in zip(fin.pred, fin.gold, fin.text))
        kinds.pop(None, None)                 # finished without a parsed answer
        n = sum(kinds.values())
        share = " / ".join(f"{100 * kinds[k] / n:.0f}%" if n else "-"
                           for k in ("stated", "own", "other"))
        with_c = f"{100 * y[conf].hit_cap.mean():3.0f}%" if conf.any() else "  -"
        print(f"  {task:11s} {SHORT[arm]:6s} {c:6s} conflict {100 * conf.mean():4.1f}% "
              f"(none {floor:.1f}%) | truncated {with_c} with vs "
              f"{100 * y[~conf].hit_cap.mean():3.0f}% without | finished with a conflict "
              f"n={n}: {share}")


# ------------------------------------------------------------------ A2 misreading the primer
# Degree is left out: a model states degrees it counted itself in the same words
# ("Node 5 has degree 3"), so a wrong one is not a misread. Degree copying is
# measured on edge_count tables ([rpchain]) and node_degree answers ([rpcopy]).
# Clustering coefficients and return probabilities are not computed by the models,
# and the none rows print how often such quotes appear with no primer.
_OF = r"\s*(?:of\s+|is\s+|=\s*|:\s*)?"
_STEP = r"(?:\s+after\s+(?P<s>\d)\s+steps?)?"
_QUOTE = {
    "clustering": [
        re.compile(r"\bnode (?P<k>\d+)\s+has (?:degree \d+,\s*)?clustering coefficient"
                   + _OF + r"(?P<v>\d\.\d\d)", re.I),
        re.compile(r"\bclustering coefficient of node (?P<k>\d+)\s*(?:is|=|:)\s*(?P<v>\d\.\d\d)", re.I),
        re.compile(r"\bnode (?P<k>\d+)'s clustering coefficient" + _OF + r"(?P<v>\d\.\d\d)", re.I)],
    "rwse": [
        re.compile(r"\bnode (?P<k>\d+)\s+has (?:degree \d+,\s*)?(?:clustering coefficient "
                   r"\d\.\d\d,\s*(?:and\s+)?)?return probabilit(?:y|ies)" + _OF
                   + r"(?P<v>\d\.\d\d)" + _STEP, re.I),
        re.compile(r"\breturn probabilit(?:y|ies) of node (?P<k>\d+)" + _STEP
                   + r"\s*(?:is|=|:)\s*(?P<v>\d\.\d\d)", re.I)],
}


def quotes(text, feature):
  """The distinct (node, value, column) a response states for one primer feature,
  in the primer's own phrasing ('Node 5 has clustering coefficient 0.43'); for
  rwse the column is the 3-step value when the quote says 'after 3 steps'."""
  t = text.replace("*", "")
  out = set()
  for rx in _QUOTE[feature]:
    for m in rx.finditer(t):
      col = FEATURE[feature] + (m.groupdict().get("s") == "3")
      out.add((int(m["k"]), m["v"], col))
  return out


def classify_quote(k, v, values, col):
  """'right', 'near' (not k's value but k-1's or k+1's) or 'other'; None for an
  id that is not a node."""
  if k not in values:
    return None
  if values[k][col] == v:
    return "right"
  near = [u for u in (k - 1, k + 1) if u in values]
  return "near" if any(values[u][col] == v for u in near) else "other"


def near_chance(k, v, values, col):
  """Chance that a wrong value equals k-1's or k+1's, given how often the other
  nodes print it (as primer_findings.leakage)."""
  near = [u for u in (k - 1, k + 1) if u in values]
  rest = [u for u in values if u != k and u not in near]
  q = np.mean([values[u][col] == v for u in rest]) if rest else 0.0
  return 1 - (1 - q) ** len(near)


def misread_block(d):
  print("[pdmisread] A2, all tasks and densities: responses quoting a node's clustering "
        "coefficient or return probability in the primer's phrasing, each distinct "
        "(node, value) counted once per response; under none (no primer: how often such "
        "quotes arise anyway) and under the primers that print the feature: distinct "
        "quotes right / the neighbouring line's value, observed vs expected by chance "
        "(responses with one) / other")
  for feature in ("clustering", "rwse"):
    for arm in ARMS:
      for c in ("none",) + SHOWN[feature]:
        x = d[(d.arm == arm) & (d.condition == c)]
        kinds, chance, responses, near_resp = collections.Counter(), 0.0, 0, 0
        for text, dens, idx in zip(x.text, x.density_class, x["index"]):
          found = quotes(text, feature)
          if not found:
            continue
          responses += 1
          if c == "none":
            continue
          values = facts(dens, idx)[1]
          near_here = False
          for k, v, col in found:
            kind = classify_quote(k, v, values, col)
            if kind:
              kinds[kind] += 1
              near_here |= kind == "near"
              if kind != "right":
                chance += near_chance(k, v, values, col)
          near_resp += near_here
        n = sum(kinds.values())
        if c == "none":
          print(f"  {feature:10s} {SHORT[arm]:6s} {c:10s} {responses} of {len(x)} responses quote a value")
        elif n:
          print(f"  {feature:10s} {SHORT[arm]:6s} {c:10s} {responses} responses, {n} quotes: "
                f"right {100 * kinds['right'] / n:.1f}% | near {kinds['near']} vs "
                f"{chance:.1f} by chance ({near_resp} responses) | other {kinds['other']}")


# ------------------------------------------------------------------ A3 commitment and revision
_REVISE = re.compile(r"\bwait\b|\blet me (?:re-?check|re-?count|double[- ]check|verify|"
                     r"check (?:that )?again)|\brecount|\bmistake\b|\bmiscount", re.I)
# A value about to be divided is a degree sum, not an edge count ("the number of
# edges is 160 divided by 2, which is 80" states 80).
_AFTER = (r"\s*(?:is|would be|should be|=|:)\s*(\d+)\b(?![.,]\d)"
          r"(?!\s*(?:/|÷|divided|over)\b)")


def candidates(task, text, t):
  """(position, value) of each answer a text states on the way: for node_degree
  only statements about the queried node t, for edge_count only whole-graph
  statements (a bare "12 edges" is usually one node's count)."""
  s = text.replace("*", "")
  if task == "node_degree":
    pats = [rf"degree of node {t}\b" + _AFTER,
            rf"\bnode {t}\s+has (?:a )?degree (?:of )?(\d+)\b(?![.,]\d)",
            rf"\bnode {t}\s+(?:is connected to|has) (\d+)\s+(?:nodes|neighbou?rs|connections|edges)\b"]
  else:
    pats = [r"(?:number of edges|total (?:number of )?edges|answer)" + _AFTER,
            r"/\s*2\s*=\s*(\d+)\b(?![.,]\d)",
            r"divided by (?:2|two)[,\s]*(?:which\s+)?(?:is|=|gives|equals)\s*(\d+)\b(?![.,]\d)",
            r"(?:graph|there) (?:has|are) (\d+) (?:unique |total |distinct )?edges\b"]
  found = [(m.start(), int(m.group(1))) for p in pats for m in re.finditer(p, s, re.I)]
  return sorted(found)


def trace_of(text):
  """The reasoning before </think> (the whole text when a truncated response never
  closed it)."""
  return text.split("</think>")[0]


def commit_block(d):
  print("[pdcommit] A3, thinking arms, main sweep: finished responses -- share with a "
        "candidate answer in the trace, median position of the first one (% of the "
        "trace), share whose first candidate is the final answer, mean distinct "
        "candidates; revision words per 1,000 characters, finished | truncated")
  for arm in THINK:
    for task in ("node_degree", "edge_count"):
      x = d[(d.arm == arm) & (d.task == task) & d.density_class.isin(MAIN)]
      for c in ["none"] + CONDS:
        y = x[x.condition == c]
        rows = []
        for text, trunc, targets, pred in zip(y.text, y.hit_cap, y.targets, y.pred):
          tr = trace_of(text)
          cand = candidates(task, tr, targets[0] if targets else None)
          rows.append(dict(trunc=trunc, has=float(bool(cand)),
                           first=cand[0][0] / max(len(tr), 1) if cand else np.nan,
                           same=float(cand[0][1] == to_int(pred)) if cand and not trunc else np.nan,
                           distinct=len({v for _, v in cand}),
                           rev=1000 * len(_REVISE.findall(tr)) / max(len(tr), 1)))
        r = pd.DataFrame(rows)
        f, t = r[r.trunc == 0], r[r.trunc == 1]
        print(f"  {SHORT[arm]:6s} {task:11s} {c:10s} finished n={len(f)}: candidate "
              f"{100 * f.has.mean():.0f}%, first at {100 * f['first'].median():.0f}%, first=final "
              f"{100 * f.same.mean():.0f}%, distinct {f.distinct.mean():.1f} | revisions/1k "
              f"{f.rev.mean():.2f} | truncated n={len(t)}"
              + (f": distinct {t.distinct.mean():.1f}, revisions/1k {t.rev.mean():.2f}" if len(t) else ""))


# ------------------------------------------------------------------ A4 position in the primer
def position_block(d):
  print("[pdposition] A4, node_degree, all seven densities, R1 correct share: accuracy "
        "by the queried node's quarter of the lines (ids 0-9 / 10-19 / 20-29 / 30-39) "
        "under none -> the primer; then the primer's effect on the first quarter minus "
        "its effect on the last, 95% interval over graphs within density")
  for arm in ARMS:
    dt = d[(d.arm == arm) & (d.task == "node_degree")]
    for c in ("degree", "all", "clustering", "rwse", "filler"):
      j = pf.pairs(dt, arm, "node_degree", "none", c, MAIN + HI)
      q = (j.target_id_a.astype(int) // 10).to_numpy()
      x = (j.correct_b - j.correct_a).to_numpy()
      strata = j.index.get_level_values(0).to_numpy()
      acc = " / ".join(f"{100 * j.correct_a[q == k].mean():.0f}->{100 * j.correct_b[q == k].mean():.0f}"
                       for k in range(4))
      print(f"  {SHORT[arm]:6s} {c:10s} {acc} | first minus last quarter "
            f"{fmt_gap(x, q == 0, q == 3, strata)}")


# ------------------------------------------------------------------ B5 consistency across tasks
def is_yes(p):
  return None if pd.isna(p) else str(p).strip().lower().startswith("y")


def consistent(nbrs, other, yes):
  """Whether an edge_existence answer agrees with the model's neighbour list for
  the queried node: 'yes' exactly when the other endpoint is in the list. None
  when either answer is missing."""
  if nbrs is None or yes is None:
    return None
  return (other in nbrs) == yes


def consistency_frame(d):
  """One row per (arm, condition, graph) where connected_nodes and edge_existence
  both finished with a parsed answer, main sweep."""
  cols = ["arm", "condition", "graph_id", "density_class", "pred", "exact", "targets"]
  m = d[d.density_class.isin(MAIN) & (d.hit_cap == 0)]
  cn = m[m.task == "connected_nodes"][cols]
  ee = m[m.task == "edge_existence"][cols[:3] + ["pred", "exact", "targets"]]
  j = cn.merge(ee, on=["arm", "condition", "graph_id"], suffixes=("_cn", "_ee"))
  t = j.targets_cn.str[0]
  other = [b if a == tt else a for (a, b), tt in zip(j.targets_ee, t)]
  j["consistent"] = [consistent(pf.neighbour_set(p), o, is_yes(y))
                     for p, o, y in zip(j.pred_cn, other, j.pred_ee)]
  j = j[j.consistent.notna()].copy()
  j["consistent"] = j.consistent.astype(int)
  j["both"] = ((j.exact_cn == 1) & (j.exact_ee == 1)).astype(int)
  return j


def consistency_block(d):
  j = consistency_frame(d)
  print("[pdconsist] B5, main sweep, graphs where connected_nodes and edge_existence both "
        "finished: consistent (the yes/no answer for (t, b) agrees with b being in the "
        "model's own list for t) / both correct / consistent but not both correct; then "
        "the change in consistency against none on the graphs both conditions share, "
        "exact McNemar p")
  for arm in ARMS:
    x = j[j.arm == arm]
    base = x[x.condition == "none"].set_index("graph_id")
    for c in ["none"] + CONDS:
      y = x[x.condition == c]
      line = (f"  {SHORT[arm]:6s} {c:10s} n={len(y)}: consistent {100 * y.consistent.mean():.1f}% "
              f"| both correct {100 * y.both.mean():.1f}% | consistent, not both "
              f"{100 * (y.consistent - y.both).clip(lower=0).mean():.1f}%")
      if c != "none":
        p = y.set_index("graph_id").join(base, lsuffix="_b", rsuffix="_a", how="inner")
        m = scoring.mcnemar(p.consistent_a.astype(bool), p.consistent_b.astype(bool))
        line += (f" | vs none {pf.f1(100 * (p.consistent_b.mean() - p.consistent_a.mean()))} "
                 f"(n={len(p)}, p={m['p_value']:.2g})")
      print(line)


# ------------------------------------------------------------------ B6 primer values as heuristics
def pair_features(values, a, b):
  """Endpoint features as the primer prints them: degree sum, mean clustering,
  mean 2-step return probability."""
  va, vb = values[a], values[b]
  return (int(va[0]) + int(vb[0]), (float(va[1]) + float(vb[1])) / 2,
          (float(va[2]) + float(vb[2])) / 2)


def heuristic_block(d):
  ee = d[(d.task == "edge_existence") & d.density_class.isin(MAIN)].copy()
  feats = [pair_features(facts(dn, i)[1], *t) for dn, i, t in zip(ee.density_class, ee["index"], ee.targets)]
  ee = ee.join(pd.DataFrame(feats, index=ee.index, columns=["f_degree", "f_clustering", "f_rwse"]))
  for f in ("degree", "clustering", "rwse"):      # above the median of its density
    ee[f"hi_{f}"] = (ee[f"f_{f}"] > ee.groupby("density_class")[f"f_{f}"].transform("median")).astype(int)
  ee["said_yes"] = [np.nan if h or is_yes(p) is None else float(is_yes(p))
                    for p, h in zip(ee.pred, ee.hit_cap)]
  print("[pdheur] B6, edge_existence, main sweep, pairs where both responses finished: "
        "false-alarm rate (yes on a non-edge) and miss rate (no on an edge) for pairs "
        "whose endpoints print a high vs low value (above/below the density's median), "
        "none -> primer; then the primer's change in the false-alarm (miss) rate on high "
        "pairs minus on low pairs, 95% interval")
  for f in ("degree", "clustering", "rwse"):
    for arm in ARMS:
      dt = ee[ee.arm == arm]
      for c in SHOWN[f] + ("filler",):
        j = pf.pairs(dt, arm, "edge_existence", "none", c, MAIN)
        j = j[j.said_yes_a.notna() & j.said_yes_b.notna()]
        strata = j.index.get_level_values(0).to_numpy()
        hi = j[f"hi_{f}_a"].to_numpy()
        parts = []
        for label, gold in (("false alarm", 0), ("miss", 1)):
          m = (j.gold_is_yes_a == gold).to_numpy()
          said = j.said_yes_a if gold == 0 else 1 - j.said_yes_a
          said_b = j.said_yes_b if gold == 0 else 1 - j.said_yes_b
          x = (said_b - said).to_numpy()
          rates = (f"high {100 * said[m & (hi == 1)].mean():.0f}->{100 * said_b[m & (hi == 1)].mean():.0f}, "
                   f"low {100 * said[m & (hi == 0)].mean():.0f}->{100 * said_b[m & (hi == 0)].mean():.0f}")
          parts.append(f"{label} {rates}, high minus low "
                       f"{fmt_gap(x[m], hi[m] == 1, hi[m] == 0, strata[m])}")
        print(f"  {f:10s} {SHORT[arm]:6s} {c:10s} " + " | ".join(parts))


# ------------------------------------------------------------------ B7 relation by density
def informativeness(values, feature):
  """Share of nodes whose degree is the most common degree among the nodes that
  print the same value of the feature (rwse: the 2- and 3-step pair): 1.0 when
  the printed feature pins every node's degree, low when it tells nothing."""
  key = (lambda v: (v[2], v[3])) if feature == "rwse" else (lambda v: v[FEATURE[feature]])
  groups = collections.defaultdict(list)
  for v in values.values():
    groups[key(v)].append(v[0])
  return sum(collections.Counter(ds).most_common(1)[0][1] for ds in groups.values()) / len(values)


def informativeness_gain(values, feature):
  """informativeness() above what guessing the graph's most common degree for
  every node already scores (degrees bunch up in dense graphs): 0 = the feature
  adds nothing, 1 = it pins every degree."""
  base = collections.Counter(v[0] for v in values.values()).most_common(1)[0][1] / len(values)
  return (informativeness(values, feature) - base) / (1 - base) if base < 1 else 0.0


def density_block(d):
  raw = {f: {dn: np.mean([informativeness(facts(dn, i)[1], f) for i in range(100)])
             for dn in MAIN + HI} for f in ("rwse", "clustering")}
  info = {f: {dn: np.mean([informativeness_gain(facts(dn, i)[1], f) for i in range(100)])
              for dn in MAIN + HI} for f in ("rwse", "clustering")}
  print("[pddensity] B7, node_degree: how far the printed feature pins a node's degree "
        "(share of nodes whose degree is the most common one among nodes printing the same "
        "value; then its gain over guessing the graph's most common degree, 0 = adds "
        "nothing, 1 = pins every degree), then each primer's effect per density (points, "
        "R1), p = .10 .20 .35 .50 .65 .75 .85; Spearman rho over the seven densities "
        "between the gain and the effect")
  for f in ("rwse", "clustering"):
    print(f"  informativeness {f:10s} " + " ".join(f"{raw[f][dn]:.2f}" for dn in MAIN + HI))
    print(f"  gain            {f:10s} " + " ".join(f"{info[f][dn]:.2f}" for dn in MAIN + HI))
  for arm in ARMS:
    dt = d[(d.arm == arm) & (d.task == "node_degree")]
    for c in ("rwse", "clustering", "degree", "filler"):
      eff = [rp.diff(pf.pairs(dt, arm, "node_degree", "none", c, [dn]), "correct")
             for dn in MAIN + HI]
      line = f"  {SHORT[arm]:6s} {c:10s} " + " ".join(f"{e:+5.0f}" for e in eff)
      if c in info:
        rho, p = stats.spearmanr([info[c][dn] for dn in MAIN + HI], eff)
        line += f" | rho {rho:+.2f} (p={p:.2g})"
      print(line)


# ------------------------------------------------------------------ B8 which nodes a primer helps
def degree_tercile(values, t):
  """0/1/2: the queried node's degree in the bottom, middle or top third of its
  graph (average rank among the graph's nodes)."""
  degs = np.array([int(v[0]) for v in values.values()])
  rank = (stats.rankdata(degs)[list(values).index(t)] - 0.5) / len(degs)
  return int(min(rank * 3, 2))


@functools.lru_cache(maxsize=None)
def queried_tercile(dens, idx, t):
  return degree_tercile(facts(dens, idx)[1], t)


def node_block(d):
  print("[pdnode] B8, the queried node's degree tercile within its graph (low / mid / "
        "high): each primer's effect on R1 correct share per tercile, then high minus "
        "low, 95% interval; node_degree and edge_existence over all seven densities, "
        "connected_nodes over the main sweep")
  for task in ("node_degree", "connected_nodes", "edge_existence"):
    for arm in ARMS:
      dt = d[(d.arm == arm) & (d.task == task)]
      for c in CONDS:
        j = pf.pairs(dt, arm, task, "none", c, DENS.get(task, MAIN))
        tt = np.array([queried_tercile(dn, int(i), int(t)) for dn, i, t in
                       zip(j.index.get_level_values(0), j.index_a, j.target_id_a)])
        x = (j.correct_b - j.correct_a).to_numpy()
        strata = j.index.get_level_values(0).to_numpy()
        per = " / ".join(f"{pf.f1(100 * x[tt == k].mean())}" for k in range(3))
        print(f"  {task:15s} {SHORT[arm]:6s} {c:10s} {per} | high minus low "
              f"{fmt_gap(x, tt == 2, tt == 0, strata)}")


# ------------------------------------------------------------------ B9 direction of errors
def error_block(d):
  print("[pderror] B9, finished responses with a numeric answer: n, share exact / under "
        "/ over, median signed error of the wrong ones, median relative error of the "
        "wrong ones; node_degree over all seven densities, edge_count over the main sweep")
  for task in ("node_degree", "edge_count"):
    for arm in ARMS:
      x = d[(d.arm == arm) & (d.task == task) & (d.hit_cap == 0)
            & d.density_class.isin(DENS.get(task, MAIN))]
      for c in ["none"] + CONDS:
        y = x[x.condition == c]
        p = np.array([to_int(v) for v in y.pred], float)
        g = np.array([to_int(v) for v in y.gold], float)
        ok = ~np.isnan(p)
        e = p[ok] - g[ok]
        wrong = e != 0
        rel = np.abs(e[wrong]) / np.maximum(g[ok][wrong], 1)
        print(f"  {task:11s} {SHORT[arm]:6s} {c:10s} n={ok.sum()}: exact {100 * (e == 0).mean():.0f}% "
              f"| under {100 * (e < 0).mean():.0f}% | over {100 * (e > 0).mean():.0f}% | median "
              f"wrong error {np.median(e[wrong]) if wrong.any() else float('nan'):+.0f} | median "
              f"relative {100 * np.median(rel) if wrong.any() else float('nan'):.0f}%")


def main():
  sys.stdout.reconfigure(encoding="utf-8")
  d = rp.load()
  for block in (source_block, misread_block, commit_block, position_block,
                consistency_block, heuristic_block, density_block, node_block, error_block):
    block(d)


if __name__ == "__main__":
  main()
