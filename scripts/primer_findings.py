"""Numbers behind the v3 paper's findings, recomputed from the raw generations.

Every body number that a generated float does not already carry is printed
here, under a tag that paper/NUMBERS.md points to. Two figure-data CSVs are
written for paper/make_v3_figures.py.

Sources: csv2/raw-trends/frame.csv (the 84,000-row frame that
scripts/build_raw_frame.py rebuilds from runs/ and checks gold-for-gold), the
raw responses in runs/ for the two text measures (route and discrepancy), and
runs/qwen3-1.7b.{degdens40,degdensrep,degfixdeg}.* for the replication.

Conventions are the paper's: pair two conditions on the shared graph within
(arm, task, density); drop the pair if either generation hit the budget; exact
match; exact McNemar; 95% intervals from a bootstrap over graphs, stratified by
density.

  PYTHONPATH=. python scripts/primer_findings.py
  PYTHONPATH=. python scripts/primer_findings.py --csv-dir csv2/raw-trends
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from graphtalk import scoring

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_primer_window as apw  # noqa: E402  (cells(): the window table's rule)

FRAME = "csv2/raw-trends/frame.csv"
BARS = "shortcuts_n40_flat.json"
ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"]
PRIMERS = ["components", "clustering", "rwse", "degree", "all"]
TASKS4 = ["edge_existence", "node_degree", "connected_nodes", "edge_count"]
DENS4 = [0.10, 0.20, 0.35, 0.50]
DENSHI = [0.65, 0.75, 0.85]
BAND = (0.25, 0.75)          # where both kinds of primer gain (Table 2's bins)
# Every bootstrap restarts from SEED, so an interval does not depend on which
# analyses ran before it (a shared generator shifted them whenever one was added).
SEED = 20260923
B = 2000

# A response retrieves when it answers without restating the queried node's
# neighbour list ("Node 7 is connected to ..."); otherwise it enumerates when it
# lists at least five neighbours one per line, and asserts a count when it does
# not. A response that opens with an answer and then restates the list is not a
# retrieval: most such qwen3-4b responses from p=.50 open with a value other than
# the stated degree and then recount, so opens_with() reports them separately.
_ENUM_LINE = re.compile(r"^\s*(?:\d+\.|[-*])\s*\**\d+\**\s*$", re.M)
_ANSWER_FIRST = re.compile(r"^\s*(?:\*\*)?The degree of node \d+ is \**(\d+)", re.I)
_DISCREPANCY = re.compile(r"discrepanc|contradict|conflict|inconsisten", re.I)
# "to see if there's any inconsistency" checks for a discrepancy without reporting one.
_HYPOTHETICAL = re.compile(r"\b(?:any|no|not|without|whether|if there|check for|"
                           r"check if|to see if)\b", re.I)


def route(text, target):
  text = text or ""
  lists = re.search(rf"[Nn]ode {target}\**\s+is connected to"
                    r"|connected to (?:the following )?nodes"
                    rf"|(?:nodes|edges) (?:directly )?connected to (?:\*\*)?node {target}\b,?"
                    r" (?:which|that) are", text)
  if not lists:
    return "retrieve"
  return "enumerate" if len(_ENUM_LINE.findall(text)) >= 5 else "assert"


def opens_with(text):
  """The value a response states before anything else, or None."""
  m = _ANSWER_FIRST.search(text or "")
  return int(m.group(1)) if m else None


def reports_discrepancy(text):
  """True when a discrepancy word is used to report one, not to look for one: the
  60 characters before it, within its sentence, carry no hypothetical cue."""
  text = text or ""
  for m in _DISCREPANCY.finditer(text):
    before = re.split(r"[.!?\n]", text[max(0, m.start() - 60):m.start()])[-1]
    if not _HYPOTHETICAL.search(before):
      return True
  return False


# ------------------------------------------------------------------ helpers
def pairs(f, arm, task, a, b, dens):
  d = f[(f.arm == arm) & (f.task == task) & f.density_class.isin(dens)]
  x = d[d.condition == a].set_index(["density_class", "graph_id"])
  y = d[d.condition == b].set_index(["density_class", "graph_id"])
  j = x.join(y, lsuffix="_a", rsuffix="_b", how="inner")
  return j[(j.hit_cap_a == 0) & (j.hit_cap_b == 0)]


def pairs_as_error(f, arm, task, a, b, dens):
  """All pairs, with a generation that reaches the budget scored as wrong."""
  d = f[(f.arm == arm) & (f.task == task) & f.density_class.isin(dens)].copy()
  d["exact"] = np.where(d.hit_cap == 1, 0, d.exact)
  x = d[d.condition == a].set_index(["density_class", "graph_id"])
  y = d[d.condition == b].set_index(["density_class", "graph_id"])
  return x.join(y, lsuffix="_a", rsuffix="_b", how="inner")


def boot(j, stat):
  """95% interval of stat(j) over graphs resampled within each density."""
  groups = [g for _, g in j.groupby(level=0)]
  rng = np.random.default_rng(SEED)
  vals = []
  for _ in range(B):
    vals.append(stat(pd.concat([g.iloc[rng.integers(0, len(g), len(g))]
                                for g in groups])))
  return np.percentile(vals, [2.5, 97.5])


def effect(j, col="exact"):
  """Paired effect in points, its interval, broke/fixed and exact McNemar p."""
  a, b = j[col + "_a"].astype(bool), j[col + "_b"].astype(bool)
  m = scoring.mcnemar(a.to_numpy(), b.to_numpy())
  d = 100 * (b.mean() - a.mean())
  lo, hi = boot(j, lambda s: 100 * (s[col + "_b"].mean() - s[col + "_a"].mean()))
  return d, lo, hi, m["b"], m["c"], m["p_value"], len(j)


def fmt(e):
  d, lo, hi, broke, fixed, p, n = e
  return f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}] broke {broke} fixed {fixed} p={p:.2g} n={n}"


def bh(p):
  p = np.asarray(p, float)
  o = np.argsort(p)
  r = p[o] * len(p) / np.arange(1, len(p) + 1)
  r = np.minimum.accumulate(r[::-1])[::-1]
  q = np.empty(len(p))
  q[o] = np.clip(r, 0, 1)
  return q


def load_runs(pattern, tasks=None, conds=None):
  rows = {}
  for path in sorted(glob.glob(pattern)):
    for line in open(path, encoding="utf-8"):
      if not line.strip():
        continue
      r = json.loads(line)
      if tasks and r["task"] not in tasks or conds and r["condition"] not in conds:
        continue
      rows.setdefault((r["instance_id"], r["condition"]), r)
  return rows


# ------------------------------------------------------------------ analyses
def band_table(f, bars):
  t = apw.cells(f, bars)
  print(f"[cells] {len(t)} cells with >=50 untruncated pairs; "
        f"{int(t.carries.sum())} answer-carrying, {int((~t.carries).sum())} side")
  print("[bands] mean effect by baseline band (answer-carrying | side information)")
  # round(., 9) first: a band mean that is exactly a half (-23/4) arrives as
  # -5.7499999... and would otherwise print as -5.7.
  print(apw.window(t).round(9).to_string(index=False, float_format=lambda x: f"{x:+.1f}"))
  side = t[~t.carries]
  print(f"[bands] side-information cells at or above 0.90: "
        f"{int((side.baseline >= 0.90).sum())} of {len(side)}; at 1.00: "
        f"{int((side.baseline >= 0.999).sum())}")
  top4b = t[t.carries & (t.baseline >= 0.90) & (t.arm == "qwen3-4b")]
  print(f"[bands] qwen3-4b answer-carrying cells >=0.90: {len(top4b)}, effects "
        + ", ".join(f"{v:+.0f}" for v in sorted(top4b.delta)))
  lo, hi = BAND
  s = t[(~t.carries) & (t.baseline >= lo) & (t.baseline < hi)]
  g = s.groupby("condition").delta.agg(["mean", "size"])
  print(f"[bands] side information by primer, baseline {lo}-{hi}: "
        + ", ".join(f"{c} {r['mean']:+.1f} ({int(r['size'])})" for c, r in g.iterrows()))
  deg = s.condition.isin(["degree", "all"])
  print(f"[bands] side information, baseline {lo}-{hi}: degree and all "
        f"{s[deg].delta.mean():+.1f} ({int(deg.sum())} cells, "
        f"{', '.join(sorted(set(s[deg].task)))}); components, clustering and rwse "
        f"{s[~deg].delta.mean():+.1f} ({int((~deg).sum())} cells)")
  w = t[t.carries & (t.baseline >= lo) & (t.baseline < hi)]
  print(f"[window] answer-carrying cells, baseline {lo}-{hi}: {len(w)} "
        f"({', '.join(sorted(set(w.task)))}), effects {w.delta.min():+.0f} to "
        f"{w.delta.max():+.0f}; by arm " + ", ".join(
            f"{a} {int(r['size'])} at {r['mean']:+.1f}"
            for a, r in w.groupby("arm").delta.agg(["size", "mean"]).iterrows()))

  # Regression to the mean: bin each cell on half its graphs, measure on the other.
  cells = []
  for arm in ARMS:
    for task in TASKS4:
      for dens in sorted(f.density_class.unique()):
        for c in PRIMERS:
          j = pairs(f, arm, task, "none", c, [dens])
          if len(j) >= 50:
            cells.append((bars.get(f"{task}/{c}", 0) >= apw.CARRIES,
                          j.exact_a.to_numpy(float), j.exact_b.to_numpy(float)))
  rng = np.random.default_rng(7)
  acc = {k: [] for k in range(len(apw.BANDS))}
  for _ in range(400):
    b1, dx, car = [], [], []
    for carries, x, y in cells:
      perm = rng.permutation(len(x))
      h1, h2 = perm[: len(x) // 2], perm[len(x) // 2:]
      b1.append(x[h1].mean())
      dx.append(100 * (y[h2].mean() - x[h2].mean()))
      car.append(carries)
    b1, dx, car = np.array(b1), np.array(dx), np.array(car)
    for k, (blo, bhi) in enumerate(apw.BANDS):
      m = (b1 >= blo) & (b1 < bhi)
      acc[k].append((dx[m & car].mean() if (m & car).any() else np.nan,
                     dx[m & ~car].mean() if (m & ~car).any() else np.nan))
  print("[splithalf] binned on half the graphs, effect on the other half:")
  for k, (blo, bhi) in enumerate(apw.BANDS):
    a = np.nanmean(np.array(acc[k]), axis=0)
    print(f"  {blo:.2f}-{min(bhi, 1):.2f}: carrying {a[0]:+.1f} side {a[1]:+.1f}")
  return t


def per_arm(t):
  lo, hi = BAND
  print(f"[arms] median baseline and cells with baseline in [{lo}, {hi})")
  for arm in ARMS:
    a = t[t.arm == arm]
    w = a[(a.baseline >= lo) & (a.baseline < hi)]
    # The thinking arms keep no edge_count cells (they mostly truncate), so the
    # like-for-like comparison of arms is on the other tasks.
    m = a[a.task != "edge_count"].baseline.median()
    print(f"  {arm:17s} median {100 * a.baseline.median():5.1f}  in band "
          f"{len(w)}/{len(a)}  below {lo}: {int((a.baseline < lo).sum())}  "
          f"median outside edge_count {100 * m:5.1f}")


def four_b_think_null(f):
  worst = 0.0
  for task in ["node_degree", "edge_existence", "connected_nodes"]:
    for c in PRIMERS:
      d, lo, hi, *_ = effect(pairs(f, "qwen3-4b-think", task, "none", c, DENS4))
      worst = max(worst, abs(d))
  print(f"[null4bt] qwen3-4b-think, largest |effect| over five primers on "
        f"node_degree/edge_existence/connected_nodes (main sweep): {worst:.1f}")


def sign_flip(f):
  print("[flip] qwen3-4b node_degree, degree and all vs none, by density")
  for c in ["degree", "all"]:
    for dens in DENS4 + DENSHI:
      j = pairs(f, "qwen3-4b", "node_degree", "none", c, [dens])
      m = scoring.mcnemar(j.exact_a.astype(bool).to_numpy(),
                          j.exact_b.astype(bool).to_numpy())
      print(f"  {c:6s} p={dens:.2f} base {100 * j.exact_a.mean():5.1f} "
            f"delta {100 * (j.exact_b.mean() - j.exact_a.mean()):+5.1f} "
            f"broke {m['b']} fixed {m['c']} p={m['p_value']:.2g}")
  d = f[(f.arm == "qwen3-4b") & (f.task == "node_degree") & (f.density_class == 0.5)
        & (f.hit_cap == 0)]
  print("  median new tokens at p=.50: "
        + ", ".join(f"{c} {d[d.condition == c].n_new_tokens.median():.0f}"
                    for c in ["none", "degree", "filler"]))


def procedure(f):
  runs = {}
  for arm in ["qwen3-4b", "qwen3-1.7b-think"]:
    runs[arm] = load_runs(f"runs/{arm}.densfull40*.shard*.jsonl",
                          tasks={"node_degree"}, conds={"none", "degree"})
  nd = f[(f.task == "node_degree") & f.condition.isin(["none", "degree"])
         & f.arm.isin(runs)].copy()
  nd["text"] = [runs[a][(i, c)]["response"] for a, i, c in
                zip(nd.arm, nd.instance_id, nd.condition)]
  nd["route"] = [route(x, int(t)) for x, t in zip(nd.text, nd.target_id)]
  nd["flag"] = nd.text.apply(reports_discrepancy)
  nd["opens"] = nd.text.apply(opens_with)
  capped = nd[nd.hit_cap == 1]
  nd = nd[nd.hit_cap == 0]

  s = nd[nd.arm == "qwen3-4b"]
  print("[route] qwen3-4b node_degree: share of responses by route (accuracy)")
  for c in ["none", "degree"]:
    x = s[(s.condition == c) & (s.density_class == 0.5)]
    print(f"  {c:6s} p=.50: " + ", ".join(
        f"{r} {100 * (x.route == r).mean():.0f}% ({100 * x[x.route == r].exact.mean():.0f}%)"
        for r in ["retrieve", "assert", "enumerate"] if (x.route == r).any()))
  print("  by density: accuracy without a primer | under degree: retrieves, its "
        "accuracy, enumerates | the rest: accuracy, share opening with an answer, "
        "share opening with a wrong value")
  for dens in DENS4 + DENSHI:
    n = s[(s.condition == "none") & (s.density_class == dens)]
    g = s[(s.condition == "degree") & (s.density_class == dens)]
    r, o = g[g.route == "retrieve"], g[g.route != "retrieve"]
    wrong = o.opens.notna() & (o.opens != o.target_degree)
    print(f"   p={dens:.2f}: none {100 * n.exact.mean():5.1f} (enumerates "
          f"{100 * (n.route == 'enumerate').mean():.0f}%) | retrieves "
          f"{100 * len(r) / len(g):.0f}% at {100 * r.exact.mean():.1f}% | enumerates "
          f"{100 * (g.route == 'enumerate').mean():.0f}% | rest {100 * len(o) / len(g):.0f}% "
          f"at {100 * o.exact.mean():.0f}%, opens {100 * o.opens.notna().mean():.0f}%, "
          f"wrong opening {100 * wrong.mean():.0f}%")

  s = nd[nd.arm == "qwen3-1.7b-think"]
  print(f"[verify] qwen3-1.7b-think under degree: retrieves at most "
        f"{100 * max((s[(s.condition == 'degree') & (s.density_class == d)].route == 'retrieve').mean() for d in DENS4 + DENSHI):.0f}%")
  x = s[(s.condition == "degree") & (s.density_class >= 0.35)]
  print(f"[verify] qwen3-1.7b-think under degree, p>=.35: enumerate "
        f"{100 * (x.route == 'enumerate').mean():.0f}%; by density "
        + ", ".join(f"{100 * (x[x.density_class == d].route == 'enumerate').mean():.0f}"
                    for d in sorted(x.density_class.unique())))
  for band, dens in (("p<=.50", DENS4), ("p>=.65", DENSHI)):
    for c in ["none", "degree"]:
      y = s[(s.condition == c) & s.density_class.isin(dens)]
      print(f"  {c:6s} {band}: discrepancy reported in {100 * y.flag.mean():.1f}% "
            f"(n={len(y)}); correct when stated {100 * y[y.flag].exact.mean():.0f}%")
  print("  degree vs none, 7 densities: "
        + fmt(effect(pairs(f, "qwen3-1.7b-think", "node_degree", "none", "degree",
                           DENS4 + DENSHI))))
  print("  degree vs none, p>=.65:     "
        + fmt(effect(pairs(f, "qwen3-1.7b-think", "node_degree", "none", "degree",
                           DENSHI))))
  print("  degree vs none by density: " + ", ".join(
      f"{100 * (j.exact_b.mean() - j.exact_a.mean()):+.1f}"
      for j in (pairs(f, "qwen3-1.7b-think", "node_degree", "none", "degree", [p])
                for p in DENS4 + DENSHI)))

  # The pair rule drops truncated generations, and degree truncates more often.
  t = f[(f.arm == "qwen3-1.7b-think") & (f.task == "node_degree")]
  print("[trunc] qwen3-1.7b-think node_degree generations reaching the budget: " + ", ".join(
      f"{c} {int(t[t.condition == c].hit_cap.sum())}/{int((t.condition == c).sum())} "
      f"({100 * t[t.condition == c].hit_cap.mean():.1f}%)" for c in ["none", "degree", "all"]))
  cap = capped[(capped.arm == "qwen3-1.7b-think") & (capped.condition == "degree")]
  print(f"  truncated degree generations reporting a discrepancy: "
        f"{100 * cap.flag.mean():.0f}% (n={len(cap)})")
  print("  degree vs none, 7 densities, truncation counted as an error: "
        + fmt(effect(pairs_as_error(f, "qwen3-1.7b-think", "node_degree", "none",
                                    "degree", DENS4 + DENSHI))))


def plain_small(f):
  print("[plain17] qwen3-1.7b node_degree, degree vs none by density: " + ", ".join(
      f"p={p:.2f} {e[0]:+.0f} ({e[4]} fixed, {e[3]} broke, p={e[5]:.2g})"
      for p, e in ((p, effect(pairs(f, "qwen3-1.7b", "node_degree", "none", "degree", [p])))
                   for p in DENS4 + DENSHI)))


def recovery(f):
  print("[recover] accuracy under degree (none) where the solver scores 1.00")
  for task, dens in (("node_degree", DENS4 + DENSHI), ("edge_count", DENS4)):
    for arm in ARMS:
      j = pairs(f, arm, task, "none", "degree", dens)
      if len(j):
        print(f"  {task:11s} {arm:17s} {j.exact_b.mean():.2f} ({j.exact_a.mean():.2f}) n={len(j)}")


def edge_count(f):
  print("[edgecount] handshake wording (uses_degree_sum, terminated generations, as "
        "every procedure share) and exact match, main sweep")
  for arm in ["qwen3-1.7b", "qwen3-4b"]:
    d = f[(f.arm == arm) & (f.task == "edge_count") & f.density_class.isin(DENS4)]
    for c in ["none", "degree"]:
      x = d[d.condition == c]
      t = x[x.hit_cap == 0]
      print(f"  {arm:10s} {c:6s} wording by density "
            + ", ".join(f"{100 * t[t.density_class == p].uses_degree_sum.mean():.0f}"
                        for p in DENS4)
            + f"%; truncated {100 * x.hit_cap.mean():.0f}%; exact among terminated "
            f"{100 * x[x.hit_cap == 0].exact.mean():.1f}%")
    for p in DENS4:
      j = pairs(f, arm, "edge_count", "none", "degree", [p])
      print(f"    p={p:.2f}: degree-none {100 * (j.exact_b.mean() - j.exact_a.mean()):+.1f} "
            f"n={len(j)}")
    print("    pooled: " + fmt(effect(pairs(f, arm, "edge_count", "none", "degree", DENS4))))
  j = pairs(f, "qwen3-1.7b", "edge_count", "none", "degree", DENS4)
  print(f"  qwen3-1.7b mean absolute error none {j.abs_error_a.mean():.1f} -> degree "
        f"{j.abs_error_b.mean():.1f} (n={len(j)})")
  x = f[(f.arm == "qwen3-1.7b") & (f.task == "edge_count") & (f.condition == "degree")
        & (f.hit_cap == 0) & (f.uses_degree_sum == 1)]
  print(f"  qwen3-1.7b degree responses using the wording: {len(x)}, exact {100 * x.exact.mean():.1f}%")


def edge_existence(f):
  d = f[(f.task == "edge_existence") & (f.hit_cap == 0)].copy()
  d["yes"] = (d.pred == "Yes").astype(int)
  d["gy"] = d.gold_is_yes.astype(int)
  print("[fa] edge_existence, pooled over seven densities: hit rate / false-alarm rate")
  for arm in ["qwen3-1.7b", "qwen3-4b"]:
    s = d[d.arm == arm]
    hits = s[s.gy == 1].groupby("condition").yes.mean()
    fas = s[s.gy == 0].groupby("condition").yes.mean()
    err = s[s.exact == 0]
    print(f"  {arm}: hits {hits.min():.2f}-{hits.max():.2f}; errors that are false "
          f"alarms {100 * (err.gy == 0).mean():.0f}%")
    for c in ["filler"] + PRIMERS:
      j = pairs(f.assign(fa=(f.pred == "Yes").astype(int)), arm, "edge_existence",
                "none", c, DENS4 + DENSHI)
      j = j[j.gold_is_yes_a == 0]
      e = effect(j, "fa")
      a = effect(pairs(f, arm, "edge_existence", "none", c, DENS4 + DENSHI))
      print(f"    {c:10s} FA {fas['none']:.2f} -> {fas[c]:.2f}: dFA {fmt(e)} | accuracy "
            f"{a[0]:+.1f} [{a[1]:+.1f}, {a[2]:+.1f}]")

  s = d[d.arm == "qwen3-1.7b"]
  rows = []
  for (dens, c), g in s.groupby(["density_class", "condition"]):
    pos, neg = g[g.gy == 1], g[g.gy == 0]
    rows.append(dict(density=dens, condition=c, yes=g.yes.mean(),
                     bacc=0.5 * (pos.exact.mean() + neg.exact.mean()),
                     tokens=g.n_new_tokens.median()))
  cur = pd.DataFrame(rows)
  print("[collapse] qwen3-1.7b edge_existence by density (yes-rate / balanced acc / median tokens)")
  for c in ["none", "filler", "components", "clustering", "rwse", "degree", "all"]:
    x = cur[cur.condition == c].sort_values("density")
    print(f"  {c:10s} " + " ".join(f"{r.yes:.2f}/{r.bacc:.2f}/{r.tokens:.0f}"
                                   for r in x.itertuples()))
  for c in ["degree", "all", "clustering", "rwse", "filler"]:
    j = pairs(f, "qwen3-1.7b", "edge_existence", "none", c, DENSHI)

    def dba(jj):
      v = []
      for _, g in jj.groupby(level=0):
        p, n = g[g.gold_is_yes_a == 1], g[g.gold_is_yes_a == 0]
        v.append(0.5 * (p.exact_b.mean() + n.exact_b.mean())
                 - 0.5 * (p.exact_a.mean() + n.exact_a.mean()))
      return 100 * np.mean(v)
    lo, hi = boot(j, dba)
    print(f"  balanced accuracy, p>=.65, {c} vs none: {dba(j):+.1f} [{lo:+.1f}, {hi:+.1f}]")


def clustering_high(f):
  print("[clusthi] qwen3-4b node_degree vs none, p>=.65 pooled")
  for c in ["clustering", "rwse", "filler", "degree", "all", "components"]:
    print(f"  {c:10s} " + fmt(effect(pairs(f, "qwen3-4b", "node_degree", "none", c, DENSHI))))


def replication():
  print("[replic] qwen3-1.7b node_degree, clustering vs none, dedicated runs")

  def contrast(rows, dens_ok, label):
    diffs, strata = [], []
    for (iid, c), r in rows.items():
      if c != "clustering":
        continue
      b = rows.get((iid, "none"))
      if b is None or r.get("hit_cap") or b.get("hit_cap"):
        continue
      m = re.search(r"/size(\d+)/p([\d.]+)/", iid)
      if not dens_ok(m):
        continue
      ex = [scoring.score_one(scoring.extract_answer(x["response"] or "", "node_degree"),
                              x["gold"], "node_degree")["exact"] for x in (r, b)]
      diffs.append(int(ex[0]) - int(ex[1]))
      strata.append(m.group(0))
    d, st = np.array(diffs), np.array(strata)
    idx = [np.flatnonzero(st == s) for s in np.unique(st)]
    rng = np.random.default_rng(SEED)
    bs = [np.mean(np.concatenate([d[rng.choice(i, len(i))] for i in idx])) for _ in range(B)]
    fixed, broke = int((d == 1).sum()), int((d == -1).sum())
    m = scoring.mcnemar(np.array([0] * fixed + [1] * broke, bool),
                        np.array([1] * fixed + [0] * broke, bool))
    print(f"  {label:34s} {100 * d.mean():+.1f} [{100 * np.percentile(bs, 2.5):+.1f}, "
          f"{100 * np.percentile(bs, 97.5):+.1f}] fixed {fixed} broke {broke} "
          f"p={m['p_value']:.2g} n={len(d)}")

  main = load_runs("runs/qwen3-1.7b.degdens40.shard*.jsonl")
  contrast(main, lambda m: float(m.group(2)) <= 0.5, "400 graphs per density, p<=.50")
  # Indices 0-99 at each density are the main sweep's own graphs; the other 300
  # are new, so they are the part of this run that replicates independently.
  new = {k: v for k, v in main.items() if int(k[0].rsplit("/", 1)[1]) >= 100}
  contrast(new, lambda m: float(m.group(2)) <= 0.5, "the 300 new graphs per density")
  contrast(load_runs("runs/qwen3-1.7b.degdensrep.shard*.jsonl"), lambda m: True,
           "fresh seeds, 1,600 graphs")
  grid = load_runs("runs/qwen3-1.7b.degfixdeg*.jsonl")
  contrast(grid, lambda m: True, "fixed mean degree, n in {20..160}")
  sizes = sorted({int(re.search(r"/size(\d+)/", i).group(1)) for i, _ in grid})
  print(f"  grid sizes {sizes}")


def components(f):
  print("[comp] qwen3-4b connected_nodes, components vs none by density; "
        "graphs with >1 component")
  one = f[(f.arm == "qwen3-4b") & (f.task == "connected_nodes") & (f.condition == "none")]
  for dens in DENS4:
    j = pairs(f, "qwen3-4b", "connected_nodes", "none", "components", [dens])
    multi = int((one[one.density_class == dens].n_components > 1).sum())
    print(f"  p={dens:.2f}: {100 * (j.exact_b.mean() - j.exact_a.mean()):+.1f}; "
          f"multi-component graphs {multi}/100")
  print("  pooled: " + fmt(effect(pairs(f, "qwen3-4b", "connected_nodes", "none",
                                          "components", DENS4))))


def bundle(f):
  parts = ["degree", "clustering", "rwse"]
  rows = []
  for arm in ARMS:
    for task in TASKS4:
      for dens in DENS4:
        eff = {}
        for c in parts + ["all"]:
          j = pairs(f, arm, task, "none", c, [dens])
          eff[c] = 100 * (j.exact_b.mean() - j.exact_a.mean()) if len(j) >= 50 else np.nan
        rows.append(dict(arm=arm, task=task, dens=dens, **eff))
  t = pd.DataFrame(rows).dropna()
  e = t["all"] - t[parts].mean(axis=1)
  rng = np.random.default_rng(SEED)
  bs = [e.iloc[rng.integers(0, len(e), len(e))].mean() for _ in range(B)]
  print(f"[bundle] all minus the mean of its parts over {len(t)} (arm, task, "
        f"density) combinations: {e.mean():+.1f} "
        f"[{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]")
  print("  qwen3-4b node_degree p>=.65: " + ", ".join(
      f"{c} {effect(pairs(f, 'qwen3-4b', 'node_degree', 'none', c, DENSHI))[0]:+.1f}"
      for c in parts + ["all"]))


def length(f):
  kch = (f[f.density_class.isin(DENS4)].groupby("condition").primer_chars.mean() / 1000)
  print("[length] mean primer characters (main sweep): "
        + ", ".join(f"{c} {1000 * kch[c]:,.0f}" for c in ["components", "degree",
                                                          "clustering", "rwse", "all", "filler"]))
  rows = []
  x = np.array([kch[c] for c in PRIMERS])
  for arm in ARMS:
    for task in TASKS4:
      d = f[(f.arm == arm) & (f.task == task) & f.density_class.isin(DENS4)]
      W = d.pivot_table(index=["density_class", "graph_id"], columns="condition",
                        values="exact")
      C = d.pivot_table(index=["density_class", "graph_id"], columns="condition",
                        values="hit_cap")
      W = W.where(C == 0)
      # Table 2's rule: a column whose fewest untruncated pairs fall below 100 is
      # selected on termination, so it is not interpreted (two edge_count columns).
      if min((W["none"].notna() & W[c].notna()).sum() for c in PRIMERS + ["filler"]) < 100:
        continue

      def fit(w):
        ds = [100 * (w.loc[w["none"].notna() & w[c].notna(), c].mean()
                     - w.loc[w["none"].notna() & w[c].notna(), "none"].mean())
              for c in PRIMERS]
        slope, icpt = np.polyfit(x, ds, 1)
        ok = w["none"].notna() & w["filler"].notna()
        fil = 100 * (w.loc[ok, "filler"].mean() - w.loc[ok, "none"].mean())
        return slope, fil - (icpt + slope * kch["filler"]), np.polyfit(x[:4], ds[:4], 1)[0]
      s, resid, s4 = fit(W)
      rng = np.random.default_rng(SEED)
      bs = np.array([fit(W.iloc[rng.integers(0, len(W), len(W))])[0] for _ in range(400)])
      lo, hi = np.nanpercentile(bs, [2.5, 97.5])
      rows.append(dict(arm=arm, task=task, slope=s, lo=lo, hi=hi, resid=resid,
                       slope_no_all=s4))
  t = pd.DataFrame(rows)
  for size in ["1.7b", "4b"]:
    s = t[t.arm.str.startswith("qwen3-" + size)]
    print(f"  {size} (Table 2 columns with >=100 pairs): slope positive in "
          f"{(s.slope > 0).sum()} of {len(s)} "
          f"({(s.slope_no_all > 0).sum()} without all); range {s.slope.min():+.1f} to "
          f"{s.slope.max():+.1f} points per 1,000 chars")
  print(f"  all: mean {t.slope.mean():+.2f}; CI excludes 0 in "
        f"{int(((t.lo > 0) | (t.hi < 0)).sum())} of {len(t)}")
  big = t.nsmallest(2, "resid")
  print("  filler below the fitted line: " + ", ".join(
      f"{r.arm}/{r.task} {-r.resid:.1f}" for r in big.itertuples()))


def clustering_spread():
  """Within-graph spread of the clustering values the primer prints, by density."""
  pat = re.compile(r"Node \d+ has clustering coefficient (\d+\.\d+)")
  sd = {}
  for path in ("prompts.densfull40.jsonl", "prompts.densfull40hi.jsonl"):
    for line in open(path, encoding="utf-8"):
      r = json.loads(line)
      if r["task"] != "node_degree" or r["condition"] != "clustering":
        continue
      dens = float(re.search(r"/p([\d.]+)/", r["instance_id"]).group(1))
      vals = np.array([float(v) for v in pat.findall(r["prompt"])])
      sd.setdefault(dens, []).append(vals.std())
  print("[cluster] mean within-graph SD of printed clustering values: "
        + ", ".join(f"p={d:.2f} {np.mean(v):.3f}" for d, v in sorted(sd.items())))


def node_count(f):
  d = f[(f.arm == "qwen3-1.7b") & (f.task == "node_count") & (f.hit_cap == 0)]
  print("[nodecount] qwen3-1.7b share answering 39: " + ", ".join(
      f"{c} {100 * (d[d.condition == c].pred.astype(str) == '39').mean():.1f}%"
      for c in ["none", "rwse", "degree", "components", "filler", "all", "clustering"]))


def measurement(f):
  """Why exact match on connected_nodes, and how the edge share moves."""
  same, flipped = [], 0
  for arm in ARMS:
    for c in ["filler"] + PRIMERS:
      j = pairs(f, arm, "connected_nodes", "none", c, DENS4)
      ex = 100 * (j.exact_b.mean() - j.exact_a.mean())
      f1 = 100 * (j.f1_b.mean() - j.f1_a.mean())
      if abs(ex) >= 5:
        if ex * f1 > 0:
          same.append(f1 / ex)
        else:
          flipped += 1
      if arm == "qwen3-4b" and c == "filler":
        print(f"[f1] qwen3-4b filler on connected_nodes: exact {ex:+.1f}, F1 {f1:+.1f}")
  print(f"[f1] of {len(same) + flipped} contrasts with |exact| >= 5, F1 moves the same "
        f"way in {len(same)} (by {min(same):.2f} to {max(same):.2f} as much) and "
        f"reverses {flipped}")
  e = f[(f.task == "edge_existence") & (f.arm == "qwen3-1.7b") & (f.condition == "none")]
  print("[goldshare] share of queried pairs that are edges: " + ", ".join(
      f"p={d:.2f} {100 * g.gold_is_yes.mean():.0f}%" for d, g in e.groupby("density_class")))


def power(f, t):
  """Smallest paired effect one cell can detect (80% power, two-sided .05), over all
  cells and over the side-information cells at baselines 0.25-0.75, where primers act."""
  disc = []
  for arm in ARMS:
    for task in TASKS4:
      for dens in sorted(f.density_class.unique()):
        for c in ["filler"] + PRIMERS:
          j = pairs(f, arm, task, "none", c, [dens])
          if len(j) >= 100:
            disc.append((j.exact_a != j.exact_b).mean())
  z = 1.959964 + 0.841621
  med = float(np.median(disc))
  print(f"[power] median discordance {med:.3f} over {len(disc)} cells -> smallest "
        f"detectable effect {100 * z * np.sqrt(med / 100):.1f} points at 100 graphs")
  lo, hi = BAND
  s = t[(~t.carries) & (t.baseline >= lo) & (t.baseline < hi)]
  dm = []
  for r in s.itertuples():
    j = pairs(f, r.arm, r.task, "none", r.condition, [r.density])
    if len(j) >= 100:
      dm.append((j.exact_a != j.exact_b).mean())
  med = float(np.median(dm))
  print(f"[power] side-information cells at baselines {lo}-{hi}: median discordance "
        f"{med:.3f} over {len(dm)} cells -> {100 * z * np.sqrt(med / 100):.1f} points")


def position(f):
  """qwen3-4b retrieval accuracy by where the queried node's line sits in the primer
  (lines are in node order): nodes 0-9 against 10-39, within density."""
  runs = load_runs("runs/qwen3-4b.densfull40*.shard*.jsonl", tasks={"node_degree"},
                   conds={"degree", "all"})
  d = f[(f.arm == "qwen3-4b") & (f.task == "node_degree") & (f.hit_cap == 0)
        & f.condition.isin(["degree", "all"])].copy()
  d["route"] = [route(runs[(i, c)]["response"], int(t))
                for i, c, t in zip(d.instance_id, d.condition, d.target_id)]
  d = d[d.route == "retrieve"].copy()
  d["early"] = d.target_id < 10

  def gap(x, early):
    diffs, w = [], []
    for _, idx in x.groupby("density_class").groups.items():
      e = early.loc[idx]
      if e.any() and (~e).any():
        diffs.append(x.exact.loc[idx][e].mean() - x.exact.loc[idx][~e].mean())
        w.append(len(idx))
    return 100 * np.average(diffs, weights=w)

  rng = np.random.default_rng(SEED)
  for c in ["degree", "all"]:
    g = d[d.condition == c]
    obs = gap(g, g.early)
    perm = [gap(g, g.groupby("density_class").early.transform(
        lambda s: rng.permutation(s.to_numpy()))) for _ in range(B)]
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (B + 1)
    print(f"[position] qwen3-4b {c}, retrievals: accuracy for queried nodes 0-9 "
          f"{100 * g[g.early].exact.mean():.1f}% (n={int(g.early.sum())}) vs 10-39 "
          f"{100 * g[~g.early].exact.mean():.1f}% (n={int((~g.early).sum())}); within "
          f"density {obs:+.1f} points, permutation p={p:.2g}; by decade of the node id "
          + " / ".join(f"{100 * g[g.target_id // 10 == k].exact.mean():.1f}"
                       for k in range(4)) + "%")


def rerun():
  """Identical prompts generated twice: degdens40 re-ran the main sweep's graphs."""
  main = load_runs("runs/qwen3-1.7b.densfull40.shard*.jsonl", tasks={"node_degree"})
  again = load_runs("runs/qwen3-1.7b.degdens40.shard*.jsonl")
  n = same = changed = 0
  for key, r in again.items():
    b = main.get(key)
    if b is None:
      continue
    n += 1
    same += r["response"] == b["response"]
    ex = [scoring.score_one(scoring.extract_answer(x["response"] or "", "node_degree"),
                            x["gold"], "node_degree")["exact"] for x in (r, b)]
    changed += ex[0] != ex[1]
  print(f"[rerun] qwen3-1.7b node_degree prompts generated twice: {n}; identical "
        f"responses {same} ({100 * same / n:.0f}%); correctness changes on {changed} "
        f"({100 * changed / n:.1f}%)")


def other_procedures(f):
  """Two procedure checks outside the stated-answer account."""
  runs = load_runs("runs/qwen3-4b.densfull40.shard*.jsonl", tasks={"connected_nodes"},
                   conds={"none", "components", "filler"})
  d = f[(f.arm == "qwen3-4b") & (f.task == "connected_nodes") & (f.hit_cap == 0)
        & f.condition.isin(["none", "components", "filler"])].copy()
  d["restates"] = [bool(re.search(rf"[Nn]ode {int(t)}\**\s+is connected to",
                                  runs[(i, c)]["response"] or ""))
                   for i, c, t in zip(d.instance_id, d.condition, d.target_id)]
  print("[compproc] qwen3-4b connected_nodes, share restating the queried node's line "
        "(accuracy of those): " + ", ".join(
            f"{c} {100 * d[d.condition == c].restates.mean():.1f}% "
            f"({100 * d[(d.condition == c) & d.restates].exact.mean():.0f}%)"
            for c in ["none", "components", "filler"]))
  runs = load_runs("runs/qwen3-4b.densfull40hi.shard*.jsonl", tasks={"node_degree"},
                   conds={"none", "clustering"})
  d = f[(f.arm == "qwen3-4b") & (f.task == "node_degree") & (f.hit_cap == 0)
        & f.condition.isin(["none", "clustering"]) & f.density_class.isin(DENSHI)].copy()
  d["route"] = [route(runs[(i, c)]["response"], int(t))
                for i, c, t in zip(d.instance_id, d.condition, d.target_id)]
  print("[clustproc] qwen3-4b node_degree p>=.65: " + "; ".join(
      f"{c} retrieve/assert/enumerate "
      + "/".join(f"{100 * (g.route == r).mean():.0f}" for r in ["retrieve", "assert", "enumerate"])
      + f"%, median {g.n_new_tokens.median():.0f} tokens" for c, g in d.groupby("condition")))
  runs = load_runs("runs/qwen3-1.7b.densfull40.shard*.jsonl", tasks={"node_degree"},
                   conds={"degree"})
  j = pairs(f, "qwen3-1.7b", "node_degree", "none", "degree", [0.20, 0.35])
  fixed = j[(j.exact_a == 0) & (j.exact_b == 1)]
  cites = sum(bool(re.search(rf"[Nn]ode {int(r.target_id_b)}\** has degree", x))
              or reports_discrepancy(x)
              for r in fixed.itertuples()
              for x in [runs[(r.instance_id_b, "degree")]["response"] or ""])
  print(f"[plaincite] qwen3-1.7b items degree fixes at p=.20/.35: {len(fixed)}; citing the "
        f"stated degree or reporting a discrepancy: {cites}")


def extraction(f):
  term = f[f.hit_cap == 0]
  print(f"[extract] parsed {100 * f.parsed.mean():.2f}% of {len(f)}; unparsed "
        f"{int((f.parsed == 0).sum())}, of which truncated "
        f"{int(((f.parsed == 0) & (f.hit_cap == 1)).sum())}, cycle_check "
        f"{int(((f.parsed == 0) & (f.task == 'cycle_check')).sum())}; terminated "
        f"{len(term)}, unparsed among them {int((term.parsed == 0).sum())}")
  hi = f[f.density_class.isin(DENSHI)].groupby(["arm", "task"]).hit_cap.mean()
  print(f"[extract] high-density extension: largest truncation rate {100 * hi.max():.1f}% "
        f"({hi.idxmax()})")


def route_data(f):
  """Per (arm, condition, density) on node_degree: accuracy, and the share and
  accuracy of each route (retrieve / assert / enumerate), plus how often the
  response states a discrepancy with the primer. Terminated generations only."""
  rows = []
  for arm in ["qwen3-4b", "qwen3-1.7b-think", "qwen3-1.7b"]:
    conds = ["none", "degree", "all", "clustering", "rwse"]
    runs = load_runs(f"runs/{arm}.densfull40*.shard*.jsonl", tasks={"node_degree"},
                     conds=set(conds))
    d = f[(f.arm == arm) & (f.task == "node_degree") & (f.hit_cap == 0)
          & f.condition.isin(conds)].copy()
    text = [runs[(i, c)]["response"] for i, c in zip(d.instance_id, d.condition)]
    d["route"] = [route(x, int(t)) for x, t in zip(text, d.target_id)]
    d["flag"] = [reports_discrepancy(x) for x in text]
    for (c, dens), g in d.groupby(["condition", "density_class"]):
      r = dict(arm=arm, condition=c, density=dens, n=len(g), acc=g.exact.mean(),
               discrepancy=g.flag.mean())
      for k in ["retrieve", "assert", "enumerate"]:
        s = g[g.route == k]
        r["share_" + k] = len(s) / len(g)
        r["acc_" + k] = s.exact.mean() if len(s) else np.nan
      other = g[g.route != "retrieve"]
      r["acc_other"] = other.exact.mean() if len(other) else np.nan
      rows.append(r)
  return pd.DataFrame(rows)


def figure_data(f, bars, out):
  """The CSVs the paper's generated floats draw from (the build's fast path)."""
  t = apw.cells(f, bars)
  ps = []
  for _, r in t.iterrows():
    j = pairs(f, r.arm, r.task, "none", r.condition, [r.density])
    ps.append(scoring.mcnemar(j.exact_a.astype(bool).to_numpy(),
                              j.exact_b.astype(bool).to_numpy())["p_value"])
  t["p"] = ps
  t["q"] = t.groupby(["arm", "task"]).p.transform(lambda s: bh(s.to_numpy()))
  t.to_csv(os.path.join(out, "primer_cells.csv"), index=False)
  d = f[(f.task == "edge_existence") & (f.hit_cap == 0) & (f.arm == "qwen3-1.7b")]
  rows = []
  for (dens, c), g in d.groupby(["density_class", "condition"]):
    pos, neg = g[g.gold_is_yes == 1], g[g.gold_is_yes == 0]
    rows.append(dict(density=dens, condition=c, yes=(g.pred == "Yes").mean(),
                     bacc=0.5 * (pos.exact.mean() + neg.exact.mean()),
                     tokens=g.n_new_tokens.median()))
  pd.DataFrame(rows).to_csv(os.path.join(out, "edge_existence_collapse.csv"), index=False)
  route_data(f).to_csv(os.path.join(out, "node_degree_routes.csv"), index=False)
  print("wrote primer_cells.csv, edge_existence_collapse.csv and "
        "node_degree_routes.csv to", out)


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--frame", default=FRAME)
  ap.add_argument("--csv-dir")
  ap.add_argument("--figure-data", action="store_true",
                  help="write the two figure CSVs to --csv-dir and stop")
  args = ap.parse_args()
  f = pd.read_csv(args.frame)
  bars = json.load(open(BARS, encoding="utf-8"))
  if args.figure_data:
    figure_data(f, bars, args.csv_dir)
    return
  t = band_table(f, bars)
  per_arm(t)
  four_b_think_null(f)
  sign_flip(f)
  procedure(f)
  plain_small(f)
  recovery(f)
  edge_count(f)
  edge_existence(f)
  clustering_high(f)
  replication()
  components(f)
  bundle(f)
  length(f)
  node_count(f)
  clustering_spread()
  measurement(f)
  power(f, t)
  position(f)
  rerun()
  other_procedures(f)
  extraction(f)
  if args.csv_dir:
    figure_data(f, bars, args.csv_dir)


if __name__ == "__main__":
  main()
