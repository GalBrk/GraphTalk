"""The one analysis of the 40-node sweep, recomputed from the raw generations.

Every number in docs/results/n40-sweep.md is printed here, under a tag
("[flip]", "[main]", ...); tests/test_results_docs.py checks the doc against the
saved output, csv2/raw-trends/primer_findings.txt. --csv-dir also writes
primer_cells.csv, edge_existence_collapse.csv and node_degree_routes.csv.

Sources: csv2/raw-trends/frame.csv (the 84,000-row frame that
scripts/build_raw_frame.py rebuilds from runs/ and checks gold-for-gold), the
raw responses in runs/ for the two text measures (route and discrepancy), and
runs/qwen3-1.7b.{degdens40,degdens40hi,degdensrep,degfixdeg}.* for the replication.

Conventions: rule R1 (graphtalk/outcomes.py). Two conditions are paired on the
shared graph within (arm, task, density) and every pair is kept; a response
that hit the budget is truncated, never correct; an effect is the change in the
correct share, printed with the change in the truncated share. Answer
descriptions (MAE, yes-rate, false alarms, routes, wording) use finished
responses only. Exact McNemar; 95% intervals from a bootstrap over graphs,
stratified by density.

  PYTHONPATH=. python scripts/primer_findings.py
  PYTHONPATH=. python scripts/primer_findings.py --csv-dir csv2/raw-trends
"""
import argparse
import glob
from decimal import ROUND_HALF_EVEN, Decimal
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from graphtalk import outcomes, scoring

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_primer_window as apw  # noqa: E402  (cells(): the window table's rule)
import score_density_sweep as sds  # noqa: E402  ([replic]: one effect computation)

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
# [replic] is the exception: it runs through score_density_sweep.effect, with
# that module's own seed, so its lines equal the density follow-ups' output.
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
def with_outcomes(f):
  """Add R1's 0/1 columns: `correct` and `truncated`."""
  f = f.copy()
  o = outcomes.outcome(f.exact, f.hit_cap)
  f["correct"] = (o == outcomes.CORRECT).astype(int)
  f["truncated"] = (o == outcomes.TRUNCATED).astype(int)
  return f


def pairs(f, arm, task, a, b, dens):
  """Every graph that has both conditions, joined as *_a / *_b. Nothing is
  dropped; code that describes answers calls finished() itself."""
  d = f[(f.arm == arm) & (f.task == task) & f.density_class.isin(dens)]
  x = d[d.condition == a].set_index(["density_class", "graph_id"])
  y = d[d.condition == b].set_index(["density_class", "graph_id"])
  return x.join(y, lsuffix="_a", rsuffix="_b", how="inner")


def finished(j):
  """The pairs in which both responses finished within the budget."""
  return j[(j.hit_cap_a == 0) & (j.hit_cap_b == 0)]


def boot_positions(j):
  """Row positions of each density group, groups in sorted density order."""
  level = j.index.get_level_values(0)
  return [np.flatnonzero(level == d) for d in sorted(level.unique())]


def boot(j, stat):
  """95% interval of stat(j) over graphs resampled within each density. One
  iloc per resample over the same random draws, in the same order, as a
  per-group pd.concat (tests pin the two to identical intervals)."""
  groups = boot_positions(j)
  rng = np.random.default_rng(SEED)
  vals = []
  for _ in range(B):
    idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
    vals.append(stat(j.iloc[idx]))
  return np.percentile(vals, [2.5, 97.5])


def effect(j, col="correct"):
  """Paired effect in points, its interval, broke/fixed, exact McNemar p, n, and
  (for the correct share) the change in the truncated share in points."""
  a, b = j[col + "_a"].astype(bool), j[col + "_b"].astype(bool)
  m = scoring.mcnemar(a.to_numpy(), b.to_numpy())
  d = 100 * (b.mean() - a.mean())
  lo, hi = boot(j, lambda s: 100 * (s[col + "_b"].mean() - s[col + "_a"].mean()))
  dt = (100 * (j.truncated_b.mean() - j.truncated_a.mean())
        if col == "correct" else np.nan)
  return d, lo, hi, m["b"], m["c"], m["p_value"], len(j), dt


def f1(x, sign=True):
  """x to one decimal, ties to even on its decimal value: 2.55 (stored as
  2.5499...) and -5.75 round the same way, and float noise cannot tip a tie."""
  if x != x:
    return "nan"
  q = Decimal(repr(round(float(x), 9))).quantize(Decimal("0.1"), rounding=ROUND_HALF_EVEN)
  return f"{q:+}" if sign else f"{q}"


def fmt(e):
  d, lo, hi, broke, fixed, p, n, dt = e
  s = f"{f1(d)} [{f1(lo)}, {f1(hi)}] broke {broke} fixed {fixed} p={p:.2g} n={n}"
  if not np.isnan(dt):
    s += f" | truncated {f1(dt)}, wrong {f1(-d - dt)}"
  return s


def bh(p):
  p = np.asarray(p, float)
  o = np.argsort(p)
  r = p[o] * len(p) / np.arange(1, len(p) + 1)
  r = np.minimum.accumulate(r[::-1])[::-1]
  q = np.empty(len(p))
  q[o] = np.clip(r, 0, 1)
  return q


def record_correct(r):
  """R1 on one raw run record: 1 when it finished and its answer is exact."""
  if r.get("hit_cap"):
    return 0
  pred = scoring.extract_answer(r["response"] or "", r["task"])
  return int(scoring.score_one(pred, r["gold"], r["task"])["exact"])


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
  """Effects binned by each cell's own no-primer accuracy. The bands use the cells
  under the truncation flag, where that accuracy is accuracy: in a flagged cell
  the correct share mostly measures finishing within the budget. Flagged cells
  are listed on their own ([flagged]), and [bandsens] shows the bands at other
  thresholds, so every cell is reported once and the threshold is visible.
  Returns the cells under the flag."""
  t = apw.cells(f, bars)
  t["tmax"] = t[["trunc_a", "trunc_b"]].max(axis=1)
  u = t[~t.flagged]
  print(f"[cells] {len(t)} (arm, task, density, primer) cells over the four tasks whose "
        f"gold varies ({t.groupby(['arm', 'task', 'density']).ngroups} (arm, task, density) x "
        f"{t.condition.nunique()} primers); {int(t.flagged.sum())} flagged (truncated share >= 15%), "
        f"{len(u)} under the flag: {int(u.carries.sum())} answer-carrying, "
        f"{int((~u.carries).sum())} side")
  print("[bands] mean effect by baseline band, cells under the truncation flag "
        "(answer-carrying | side information)")
  # round(., 9) first: a band mean that is exactly a half (-23/4) arrives as
  # -5.7499999... and would otherwise print as -5.7.
  print(apw.window(u).to_string(index=False, float_format=f1))
  print("[bandsens] band means, answer-carrying (cells) / side (cells), keeping cells "
        "whose larger truncated share is below each threshold")
  for thr in (0.05, 0.10, 0.15, 0.30, 0.50, 1.01):
    w = apw.window(t[t.tmax < thr])
    label = "all cells" if thr > 1 else f"below {thr:.2f}"
    print(f"  {label}: " + " | ".join(
        f"{r.band} {f1(r.d_carries)} ({r.n_carries}) / {f1(r.d_side)} ({r.n_side})"
        for r in w.itertuples()))
  print("[flagged] cells at or above the truncation flag: none correct/wrong/truncated % "
        "-> primer correct/wrong/truncated %, change in correct share")
  for r in t[t.flagged].sort_values(["arm", "task", "condition", "density"]).itertuples():
    ca, ta, cb, tb = 100 * r.baseline, 100 * r.trunc_a, 100 * r.acc, 100 * r.trunc_b
    print(f"  {r.arm} {r.task} p={r.density:.2f} {r.condition}: "
          f"{ca:.0f}/{100 - ca - ta:.0f}/{ta:.0f} -> {cb:.0f}/{100 - cb - tb:.0f}/{tb:.0f}, "
          f"{r.delta:+.0f}")
  t = u
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
        + ", ".join(f"{c} {f1(r['mean'])} ({int(r['size'])})" for c, r in g.iterrows()))
  deg = s.condition.isin(["degree", "all"])
  print(f"[bands] side information, baseline {lo}-{hi}: degree and all "
        f"{f1(s[deg].delta.mean())} ({int(deg.sum())} cells, "
        f"{', '.join(sorted(set(s[deg].task)))}); components, clustering and rwse "
        f"{f1(s[~deg].delta.mean())} ({int((~deg).sum())} cells)")
  w = t[t.carries & (t.baseline >= lo) & (t.baseline < hi)]
  print(f"[window] answer-carrying cells, baseline {lo}-{hi}: {len(w)} "
        f"({', '.join(sorted(set(w.task)))}), effects {w.delta.min():+.0f} to "
        f"{w.delta.max():+.0f}; by arm " + ", ".join(
            f"{a} {int(r['size'])} at {f1(r['mean'])}"
            for a, r in w.groupby("arm").delta.agg(["size", "mean"]).iterrows()))

  # Regression to the mean: bin each cell on half its graphs, measure on the other.
  cells = []
  for arm in ARMS:
    for task in TASKS4:
      for dens in sorted(f.density_class.unique()):
        for c in PRIMERS:
          j = pairs(f, arm, task, "none", c, [dens])
          if max(j.truncated_a.mean(), j.truncated_b.mean()) < outcomes.FLAG:
            cells.append((bars.get(f"{task}/{c}", 0) >= apw.CARRIES,
                          j.correct_a.to_numpy(float), j.correct_b.to_numpy(float)))
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
  print("[splithalf] cells under the truncation flag, binned on half the graphs, effect on the other half:")
  for k, (blo, bhi) in enumerate(apw.BANDS):
    a = np.nanmean(np.array(acc[k]), axis=0)
    print(f"  {blo:.2f}-{min(bhi, 1):.2f}: carrying {f1(a[0])} side {f1(a[1])}")
  return t


def per_arm(t):
  lo, hi = BAND
  print(f"[arms] cells under the truncation flag: median baseline and cells with baseline in [{lo}, {hi})")
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
      m = scoring.mcnemar(j.correct_a.astype(bool).to_numpy(),
                          j.correct_b.astype(bool).to_numpy())
      print(f"  {c:6s} p={dens:.2f} base {100 * j.correct_a.mean():5.1f} "
            f"delta {100 * (j.correct_b.mean() - j.correct_a.mean()):+5.1f} "
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

  # The enumerate rule needs five list lines, so a node with fewer than five
  # neighbours that is listed in full counts as "assert".
  low = s[s.density_class.isin([0.10, 0.20]) & (s.route == "assert")]
  full = low[(low.target_degree < 5) & (low.text.apply(lambda x: len(_ENUM_LINE.findall(x or "")))
                                        == low.target_degree)]
  print(f"  p<=.20, none and degree: {len(low)} responses classed assert, of which "
        f"{len(full)} list every neighbour of a node with fewer than five, one per line")

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
      f"{f1(100 * (j.correct_b.mean() - j.correct_a.mean()))}"
      for j in (pairs(f, "qwen3-1.7b-think", "node_degree", "none", "degree", [p])
                for p in DENS4 + DENSHI)))

  # Truncation is its own outcome, and degree truncates more often.
  t = f[(f.arm == "qwen3-1.7b-think") & (f.task == "node_degree")]
  print("[trunc] qwen3-1.7b-think node_degree generations reaching the budget: " + ", ".join(
      f"{c} {int(t[t.condition == c].hit_cap.sum())}/{int((t.condition == c).sum())} "
      f"({100 * t[t.condition == c].hit_cap.mean():.1f}%)" for c in ["none", "degree", "all"]))
  cap = capped[(capped.arm == "qwen3-1.7b-think") & (capped.condition == "degree")]
  print(f"  truncated degree generations reporting a discrepancy: "
        f"{100 * cap.flag.mean():.0f}% (n={len(cap)})")


def plain_small(f):
  print("[plain17] qwen3-1.7b node_degree, degree vs none by density (no-primer correct "
        "share %): " + ", ".join(
      f"p={p:.2f} {e[0]:+.0f} (base {100 * j.correct_a.mean():.0f}, {e[4]} fixed, "
      f"{e[3]} broke, p={e[5]:.2g})"
      for p, j, e in ((p, j, effect(j)) for p, j in
                      ((p, pairs(f, "qwen3-1.7b", "node_degree", "none", "degree", [p]))
                       for p in DENS4 + DENSHI))))


def recovery(f):
  print("[recover] accuracy under degree (none) where the solver scores 1.00")
  for task, dens in (("node_degree", DENS4 + DENSHI), ("edge_count", DENS4)):
    for arm in ARMS:
      j = pairs(f, arm, task, "none", "degree", dens)
      if len(j):
        print(f"  {task:11s} {arm:17s} {j.correct_b.mean():.2f} ({j.correct_a.mean():.2f}) n={len(j)}")


def edge_count(f):
  print("[edgecount] edge_count, main sweep: correct / wrong / truncated shares of all "
        "responses; exact among finished; for none and degree, the handshake wording "
        "(uses_degree_sum) among finished, by density")
  for arm in ARMS:
    d = f[(f.arm == arm) & (f.task == "edge_count") & f.density_class.isin(DENS4)]
    for c in ["none", "filler"] + PRIMERS:
      x = d[d.condition == c]
      t = x[x.hit_cap == 0]
      words = ("; wording by density " + ", ".join(
          f"{100 * w.uses_degree_sum.mean():.0f}%" if len(w) else "n/a"
          for w in (t[t.density_class == p] for p in DENS4))
               if c in ("none", "degree") else "")
      rel = (f"; median relative error of finished {100 * median_relative_error(x):.1f}%"
             if c in ("none", "degree") and len(t) else "")
      print(f"  {arm:17s} {c:10s} correct {100 * x.correct.mean():.2f}% wrong "
            f"{100 * (1 - x.correct.mean() - x.truncated.mean()):.2f}% truncated "
            f"{100 * x.truncated.mean():.2f}%; exact among finished "
            f"{f'{100 * t.exact.mean():.1f}%' if len(t) else 'n/a'}{words}{rel}")
    for p in DENS4:
      j = pairs(f, arm, "edge_count", "none", "degree", [p])
      print(f"    p={p:.2f}: degree-none {f1(100 * (j.correct_b.mean() - j.correct_a.mean()))} "
            f"n={len(j)}")
    for c in ["filler"] + PRIMERS:
      print(f"    pooled {c} vs none: "
            + fmt(effect(pairs(f, arm, "edge_count", "none", c, DENS4))))
  j = finished(pairs(f, "qwen3-1.7b", "edge_count", "none", "degree", DENS4))
  print(f"  qwen3-1.7b mean absolute error, both finished, none {j.abs_error_a.mean():.1f} -> degree "
        f"{j.abs_error_b.mean():.1f} (n={len(j)})")
  x = f[(f.arm == "qwen3-1.7b") & (f.task == "edge_count") & (f.condition == "degree")
        & (f.hit_cap == 0) & (f.uses_degree_sum == 1)]
  print(f"  qwen3-1.7b degree responses using the wording: {len(x)}, exact {100 * x.exact.mean():.1f}%")


def collapse_rows(f):
  """edge_existence per (arm, density, condition): yes-rate and median tokens of
  finished responses, balanced accuracy over all responses (a truncated one is
  not correct), and the truncated share."""
  rows = []
  e = f[f.task == "edge_existence"]
  for (arm, dens, c), g in e.groupby(["arm", "density_class", "condition"]):
    fin = g[g.hit_cap == 0]
    pos, neg = g[g.gold_is_yes == 1], g[g.gold_is_yes == 0]
    rows.append(dict(arm=arm, density=dens, condition=c,
                     yes=(fin.pred == "Yes").mean(),
                     bacc=0.5 * (pos.correct.mean() + neg.correct.mean()),
                     truncated=g.truncated.mean(), tokens=fin.n_new_tokens.median()))
  return pd.DataFrame(rows)


def edge_existence(f):
  d = f[(f.task == "edge_existence") & (f.hit_cap == 0)].copy()
  d["yes"] = (d.pred == "Yes").astype(int)
  d["gy"] = d.gold_is_yes.astype(int)
  print("[fa] edge_existence, finished responses pooled over seven densities: hit "
        "rate / false-alarm rate")
  for arm in ARMS:
    s = d[d.arm == arm]
    hits = s[s.gy == 1].groupby("condition").yes.mean()
    fas = s[s.gy == 0].groupby("condition").yes.mean()
    err = s[s.exact == 0]
    print(f"  {arm}: hits {hits.min():.2f}-{hits.max():.2f}; errors that are false "
          f"alarms {100 * (err.gy == 0).mean():.0f}%")
    for c in ["filler"] + PRIMERS:
      j = finished(pairs(f.assign(fa=(f.pred == "Yes").astype(int)), arm,
                         "edge_existence", "none", c, DENS4 + DENSHI))
      j = j[j.gold_is_yes_a == 0]
      e = effect(j, "fa")
      a = effect(pairs(f, arm, "edge_existence", "none", c, DENS4 + DENSHI))
      dfa, lo, hi, removed, added, pv, n, _ = e
      print(f"    {c:10s} FA rate of all finished {fas['none']:.2f} -> {fas[c]:.2f}; "
            f"paired, both finished: dFA {f1(dfa)} [{f1(lo)}, {f1(hi)}] FA removed "
            f"{removed} added {added} p={pv:.2g} n={n} | accuracy "
            f"{f1(a[0])} [{f1(a[1])}, {f1(a[2])}]")

  cur = collapse_rows(f)
  for arm in ARMS:
    print(f"[collapse] {arm} edge_existence by density: yes-rate of finished / "
          "balanced accuracy (truncated not correct) / truncated share / median "
          "tokens of finished")
    for c in ["none", "filler", "components", "clustering", "rwse", "degree", "all"]:
      x = cur[(cur.arm == arm) & (cur.condition == c)].sort_values("density")
      print(f"  {c:10s} " + " ".join(f"{r.yes:.2f}/{r.bacc:.2f}/{r.truncated:.2f}/{r.tokens:.0f}"
                                     for r in x.itertuples()))
  for c in ["degree", "all", "clustering", "rwse", "filler"]:
    j = pairs(f, "qwen3-1.7b", "edge_existence", "none", c, DENSHI)

    def dba(jj):
      v = []
      for _, g in jj.groupby(level=0):
        p, n = g[g.gold_is_yes_a == 1], g[g.gold_is_yes_a == 0]
        v.append(0.5 * (p.correct_b.mean() + n.correct_b.mean())
                 - 0.5 * (p.correct_a.mean() + n.correct_a.mean()))
      return 100 * np.mean(v)
    lo, hi = boot(j, dba)
    print(f"  balanced accuracy, qwen3-1.7b, p>=.65, {c} vs none: {f1(dba(j))} [{f1(lo)}, {f1(hi)}]")


def clustering_high(f):
  print("[clusthi] qwen3-4b node_degree vs none, p>=.65 pooled")
  for c in ["clustering", "rwse", "filler", "degree", "all", "components"]:
    print(f"  {c:10s} " + fmt(effect(pairs(f, "qwen3-4b", "node_degree", "none", c, DENSHI))))
  for band, dens in (("p<=.50", DENS4), ("p>=.65", DENSHI)):
    print(f"  qwen3-1.7b clustering vs none, {band}: "
          + fmt(effect(pairs(f, "qwen3-1.7b", "node_degree", "none", "clustering", dens))))


def replication():
  """The dedicated runs' pooled clustering-vs-none effects, computed by the
  density scorer (score_density_sweep.effect), so these lines and the density
  follow-ups' pooled lines are one computation."""
  print("[replic] qwen3-1.7b node_degree, clustering vs none, dedicated runs")

  def contrast(records, label):
    s = sds.summarize(records, "cell")
    flagged = sds.flagged_levels(s["cells"], "clustering", "none")
    ok = {lvl for lvl, _ in s["cells"]} - flagged
    e = sds.effect(sds.pairs_for(s["paired"], "clustering", ok, "none"))
    print(f"  {label:34s} {f1(e['d'])} [{f1(e['lo'])}, {f1(e['hi'])}] fixed {e['fixed']} "
          f"broke {e['broke']} p={e['p']:.2g} n={e['n']} truncated {f1(e['dt'])}"
          + sds._left_out(flagged))

  main = sds.load(["runs/qwen3-1.7b.degdens40.shard*.jsonl"])
  contrast(main, "400 graphs per density, p<=.50")
  # Indices 0-99 at each density are the main sweep's own graphs; the other 300
  # are new, so they are the part of this run that replicates independently.
  contrast([r for r in main if int(r["instance_id"].rsplit("/", 1)[1]) >= 100],
           "the 300 new graphs per density")
  contrast(sds.load(["runs/qwen3-1.7b.degdensrep.shard*.jsonl"]), "fresh seeds, 1,600 graphs")
  grid = sds.load(["runs/qwen3-1.7b.degfixdeg.shard*.jsonl"])
  contrast(grid, "fixed mean degree, n in {20..160}")
  sizes = sorted({int(re.search(r"/size(\d+)/", r["instance_id"]).group(1)) for r in grid})
  print(f"  grid sizes {sizes}")
  contrast(sds.load(["runs/qwen3-1.7b.degdens40hi.shard*.jsonl"]),
           "400 graphs per density, p>=.65")


def components(f):
  print("[comp] qwen3-4b connected_nodes, components vs none by density; "
        "graphs with >1 component")
  one = f[(f.arm == "qwen3-4b") & (f.task == "connected_nodes") & (f.condition == "none")]
  for dens in DENS4:
    j = pairs(f, "qwen3-4b", "connected_nodes", "none", "components", [dens])
    multi = int((one[one.density_class == dens].n_components > 1).sum())
    print(f"  p={dens:.2f}: {f1(100 * (j.correct_b.mean() - j.correct_a.mean()))}; "
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
        flagged = False
        for c in parts + ["all"]:
          j = pairs(f, arm, task, "none", c, [dens])
          eff[c] = 100 * (j.correct_b.mean() - j.correct_a.mean())
          flagged |= max(j.truncated_a.mean(), j.truncated_b.mean()) >= outcomes.FLAG
        # R1: a combination with a flagged cell is left out, as in [bands] and [length].
        if not flagged:
          rows.append(dict(arm=arm, task=task, dens=dens, **eff))
  t = pd.DataFrame(rows)
  e = t["all"] - t[parts].mean(axis=1)
  rng = np.random.default_rng(SEED)
  bs = [e.iloc[rng.integers(0, len(e), len(e))].mean() for _ in range(B)]
  print(f"[bundle] all minus the mean of its parts over {len(t)} (arm, task, "
        f"density) combinations under the truncation flag: {f1(e.mean())} "
        f"[{f1(np.percentile(bs, 2.5))}, {f1(np.percentile(bs, 97.5))}]")
  print("  qwen3-4b node_degree p>=.65: " + ", ".join(
      f"{c} {f1(effect(pairs(f, 'qwen3-4b', 'node_degree', 'none', c, DENSHI))[0])}"
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
                        values="correct")
      T = d.pivot_table(index=["density_class", "graph_id"], columns="condition",
                        values="truncated")
      # R1: a combination where any condition reaches the truncation flag is not
      # interpreted (its correct share moves with the budget, not the primer).
      if (T[["none", "filler"] + PRIMERS].mean() >= outcomes.FLAG).any():
        continue

      def fit(w):
        ds = [100 * (w[c].mean() - w["none"].mean()) for c in PRIMERS]
        slope, icpt = np.polyfit(x, ds, 1)
        fil = 100 * (w["filler"].mean() - w["none"].mean())
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
    print(f"  {size} (combinations below the truncation flag): slope positive in "
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
  print("[nodecount] qwen3-1.7b share of finished responses answering 39: " + ", ".join(
      f"{c} {100 * (d[d.condition == c].pred.astype(str) == '39').mean():.1f}%"
      for c in ["none", "rwse", "degree", "components", "filler", "all", "clustering"]))
  listed = {}
  for line in open("prompts.densfull40.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["task"] == "node_count" and r["condition"] == "none":
      dens = float(re.search(r"/p([\d.]+)/", r["instance_id"]).group(1))
      listed.setdefault(dens, []).append(nodes_listed(r["prompt"]) == 40)
  n = d[d.condition == "none"]
  print("  without a primer, by density: answers 39 / graphs whose encoding gives all 40 "
        "nodes a line: " + ", ".join(
            f"p={p:.2f} {100 * (n[n.density_class == p].pred.astype(str) == '39').mean():.0f}% / "
            f"{sum(v)}/{len(v)}" for p, v in sorted(listed.items())))


def measurement(f):
  """Why exact match on connected_nodes, and how the edge share moves."""
  same, flipped = [], 0
  for arm in ARMS:
    for c in ["filler"] + PRIMERS:
      j = finished(pairs(f, arm, "connected_nodes", "none", c, DENS4))
      ex = 100 * (j.correct_b.mean() - j.correct_a.mean())
      df1 = 100 * (j.f1_b.mean() - j.f1_a.mean())
      if abs(ex) >= 5:
        if ex * df1 > 0:
          same.append(df1 / ex)
        else:
          flipped += 1
      if arm == "qwen3-4b" and c == "filler":
        print(f"[f1] qwen3-4b filler on connected_nodes: exact {f1(ex)}, F1 {f1(df1)}")
  print(f"[f1] (finished pairs) of {len(same) + flipped} contrasts with |exact| >= 5, F1 moves the same "
        f"way in {len(same)} (by {min(same):.2f} to {max(same):.2f} as much) and "
        f"reverses {flipped}")
  e = f[(f.task == "edge_existence") & (f.arm == "qwen3-1.7b") & (f.condition == "none")]
  print("[goldshare] share of queried pairs that are edges: " + ", ".join(
      f"p={d:.2f} {100 * g.gold_is_yes.mean():.0f}%" for d, g in e.groupby("density_class")))


def power(f, t):
  """Smallest paired effect one cell of 100 graphs can detect (80% power,
  two-sided .05), over all cells under the truncation flag and over the
  side-information cells at baselines 0.25-0.75, where primers act."""
  disc = []
  for arm in ARMS:
    for task in TASKS4:
      for dens in sorted(f.density_class.unique()):
        for c in ["filler"] + PRIMERS:
          j = pairs(f, arm, task, "none", c, [dens])
          if len(j) >= 100 and max(j.truncated_a.mean(), j.truncated_b.mean()) < outcomes.FLAG:
            disc.append((j.correct_a != j.correct_b).mean())
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
      dm.append((j.correct_a != j.correct_b).mean())
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


def leakage(f):
  """What the degree sentences of the saved prompts give away, and what plain
  qwen3-4b's wrong answers under them look like. Sentences are in node order, so
  the ones next to node k's are k-1's and k+1's; chance is the rate at which the
  other nodes' stated degrees equal the wrong answer."""
  deg = re.compile(r"Node (\d+) has degree (\d+)")
  stated, hits = {}, {}
  for path in ["prompts.densfull40.jsonl", "prompts.densfull40hi.jsonl"]:
    for line in open(path, encoding="utf-8"):
      r = json.loads(line)
      if r["condition"] not in ("degree", "all") or r["task"] not in ("node_degree", "edge_count"):
        continue
      d = {int(a): int(b) for a, b in deg.findall(r["prompt"])}
      stated[(r["instance_id"], r["condition"])] = d
      ans = (d[int(re.findall(r"degree of node (\d+)", r["prompt"])[-1])]
             if r["task"] == "node_degree" else sum(d.values()) // 2)
      h, n = hits.get((r["task"], r["condition"]), (0, 0))
      hits[(r["task"], r["condition"])] = (h + (str(ans) == r["gold"].strip()), n + 1)
  print("[leak] gold read off the saved prompts (the queried node's degree sentence; half "
        "the sum of all 40): " + ", ".join(f"{t}/{c} {h}/{n}" for (t, c), (h, n) in sorted(hits.items())))

  runs = load_runs("runs/qwen3-4b.densfull40*.shard*.jsonl", tasks={"node_degree"},
                   conds={"degree", "all"})
  d = f[(f.arm == "qwen3-4b") & (f.task == "node_degree") & (f.hit_cap == 0)
        & f.condition.isin(["degree", "all"]) & (f.exact == 0) & f.pred.notna()].copy()
  d["route"] = [route(runs[(i, c)]["response"], int(t))
                for i, c, t in zip(d.instance_id, d.condition, d.target_id)]
  rng = np.random.default_rng(SEED)
  for c in ["degree", "all"]:
    for label, s in [("retrievals", d[(d.condition == c) & (d.route == "retrieve")]),
                     ("all routes", d[d.condition == c])]:
      copied, chance, off = 0, [], []
      for i, k, v, g in zip(s.instance_id, s.target_id.astype(int),
                            s.pred.astype(float).astype(int), s.gold.astype(int)):
        sd = stated[(i, c)]
        near = [j for j in (k - 1, k + 1) if j in sd]
        copied += any(sd[j] == v for j in near)
        q = np.mean([sd[j] == v for j in sd if j != k and j not in near])
        chance.append(1 - (1 - q) ** len(near))
        off.append(abs(v - g))
      chance, off = np.array(chance), np.array(off)
      sims = (rng.random((B, len(chance))) < chance).sum(1)
      p = (np.sum(sims >= copied) + 1) / (B + 1)
      print(f"[copyerr] qwen3-4b {c}, wrong {label} (n={len(s)}): within 1 of gold "
            f"{100 * np.mean(off <= 1):.0f}%, within 3 {100 * np.mean(off <= 3):.0f}%; "
            f"equal to the degree stated for node k-1 or k+1 {copied} vs "
            f"{chance.sum():.1f} by chance, p={p:.2g}")


def nodes_listed(prompt):
  """How many nodes have their own line in an incident encoding (isolated nodes
  have none)."""
  return len(set(re.findall(r"Node (\d+) is connected to", prompt)))


def median_relative_error(d):
  """Median |pred - gold| / gold over finished responses with a numeric answer."""
  fin = d[d.hit_cap == 0]
  pred = pd.to_numeric(fin.pred, errors="coerce")
  gold = pd.to_numeric(fin.gold, errors="coerce")
  ok = pred.notna() & gold.notna() & (gold != 0)
  return float(((pred[ok] - gold[ok]).abs() / gold[ok]).median())


def neighbour_set(s):
  """A connected_nodes answer as a set of node ids; None when there is no answer."""
  if not isinstance(s, str) or not s.strip():
    return None
  return set() if "no nodes" in s.lower() else set(map(int, re.findall(r"-?\d+", s)))


def joint(f):
  """Degree and neighbour answers for the same queried node, main sweep (p<=.50).
  J: both correct (a truncated side is not correct). C: both finished and
  parsed, a valid neighbour set, and the stated degree equals its size."""
  cols = ["arm", "condition", "density_class", "graph_id"]
  nd = f[(f.task == "node_degree") & f.density_class.isin(DENS4)]
  cn = f[(f.task == "connected_nodes") & f.density_class.isin(DENS4)]
  j = nd.merge(cn, on=cols, suffixes=("_d", "_n"), validate="one_to_one")
  gold_sets = j.gold_n.map(neighbour_set)
  assert (j.gold_d.astype(int) == gold_sets.map(len)).all(), "tasks query different nodes"
  sets = j.pred_n.map(neighbour_set)
  valid = sets.map(lambda s: s is not None and all(0 <= v < 40 for v in s))
  size = sets.map(lambda s: len(s) if s is not None else -1)
  j["J"] = ((j.correct_d == 1) & (j.correct_n == 1)).astype(int)
  j["C"] = ((j.hit_cap_d == 0) & (j.hit_cap_n == 0) & (j.parsed_d == 1)
            & (j.parsed_n == 1) & valid
            & (pd.to_numeric(j.pred_d, errors="coerce") == size)).astype(int)
  print("[joint] degree and neighbours of the same node, p<=.50: J both correct, "
        "C consistent, C-J consistent but not both correct (J is inside C), % of items; "
        "then J against none, with BH q over the six primers")
  for arm in ARMS:
    s = j[j.arm == arm]
    print(f"  {arm:17s} " + ", ".join(
        f"{c} J {100 * s[s.condition == c].J.mean():.2f} C {100 * s[s.condition == c].C.mean():.2f}"
        f" C-J {100 * (s[s.condition == c].C.mean() - s[s.condition == c].J.mean()):.2f}"
        for c in ["none", "filler"] + PRIMERS))
    base = s[s.condition == "none"].set_index(["density_class", "graph_id"])
    effs = []
    for c in ["filler"] + PRIMERS:
      x = base.join(s[s.condition == c].set_index(["density_class", "graph_id"]),
                    lsuffix="_a", rsuffix="_b", how="inner")
      effs.append((c, effect(x, "J")))
    for (c, e), q in zip(effs, bh([e[5] for _, e in effs])):
      print(f"    J {c:10s} " + fmt(e) + f" q={q:.2g}")


def rwse_pairs(prompt, instance_id):
  """The (2-step, 3-step) return probabilities the rwse primer prints, one per node."""
  found = re.findall(r"Node (\d+) has return probability (\d+\.\d+) after 2 steps "
                     r"and (\d+\.\d+) after 3 steps", prompt)
  if len(found) != 40:
    raise ValueError(f"{instance_id}: {len(found)} rwse sentences, expected 40")
  return [(a, b) for _, a, b in found]


def rwse_resolution():
  """Distinct printed return-probability pairs per graph against distinct stated
  degrees, from the saved prompts (the text the model saw), by density."""
  deg = re.compile(r"Node (\d+) has degree (\d+)")
  seen = {}
  for path in ("prompts.densfull40.jsonl", "prompts.densfull40hi.jsonl"):
    for line in open(path, encoding="utf-8"):
      r = json.loads(line)
      if r["task"] != "node_degree" or r["condition"] not in ("rwse", "degree"):
        continue
      seen.setdefault(r["instance_id"], {})[r["condition"]] = r["prompt"]
  by = {}
  for iid, p in seen.items():
    pr = rwse_pairs(p["rwse"], iid)
    counts = pd.Series(pr).value_counts()
    degrees = {int(d) for _, d in deg.findall(p["degree"])}
    dens = float(re.search(r"/p([\d.]+)/", iid).group(1))
    by.setdefault(dens, []).append((len(counts), counts.iloc[0] / 40, len(degrees)))
  print("[rwse] per graph, by density: distinct printed (2-step, 3-step) pairs / "
        "modal pair's share of nodes / distinct stated degrees / graphs with fewer "
        "rwse classes than degrees")
  for dens, v in sorted(by.items()):
    a = np.array(v, dtype=float)
    print(f"  p={dens:.2f}: {a[:, 0].mean():.2f} / {100 * a[:, 1].mean():.2f}% / "
          f"{a[:, 2].mean():.2f} / {int((a[:, 0] < a[:, 2]).sum())}/{len(a)}")


def setup_facts(f):
  """The facts the setup section states: constant golds, budgets, truncation."""
  print(f"[setup] {len(f)} responses on {f.groupby(['density_class', 'graph_id']).ngroups} "
        f"graphs; {f.groupby(['arm', 'task', 'condition', 'density_class']).ngroups} "
        f"cells of {f.groupby(['arm', 'task', 'condition', 'density_class']).size().min()}"
        f"-{f.groupby(['arm', 'task', 'condition', 'density_class']).size().max()}")
  print("  distinct gold values: node_count "
        f"{sorted(f[f.task == 'node_count'].gold.astype(str).unique())}, cycle_check "
        f"{sorted(f[f.task == 'cycle_check'].gold.astype(str).unique())}")
  hi = f.density_class.isin(DENSHI)
  for label, part in (("main sweep", f[~hi]), ("high-density extension", f[hi])):
    cap = part[part.hit_cap == 1].groupby("arm").n_new_tokens.max()
    print(f"  budget reached, {label}: " + ", ".join(f"{a} {int(v)}" for a, v in cap.items()))
  t = f[~hi].groupby(["arm", "task"]).truncated.mean()
  print("  nonzero truncated share, main sweep (arm/task): " + ", ".join(
      f"{a}/{k} {f1(100 * v, sign=False)}%" for (a, k), v in t.items() if v > 0))
  e = f[(f.task == "edge_existence") & (f.arm == ARMS[0]) & (f.condition == "none")]
  share = e.groupby("density_class").gold_is_yes.mean()
  pooled = e[e.density_class.isin(DENS4)].gold_is_yes.mean()
  print("  majority-class accuracy, edge_existence (share of queried pairs that are "
        "edges / majority): " + ", ".join(
            f"p={d:.2f} {100 * v:.0f}%/{100 * max(v, 1 - v):.0f}%" for d, v in share.items())
        + f"; main sweep pooled {100 * max(pooled, 1 - pooled):.2f}%; cycle_check 100%")


def main_table(f):
  """Every primer against no primer and against the structure-free filler, per arm
  and task, main sweep (p<=.50): correct shares, paired effect, exact McNemar,
  Benjamini-Hochberg q within (arm, task, control), change in the truncated share;
  secondary, over finished pairs: MAE for the integer tasks, set-F1 for
  connected_nodes."""
  tasks = ["node_count", "node_degree", "connected_nodes", "edge_count",
           "edge_existence", "cycle_check"]
  print("[main] main sweep (p<=.50), per arm and task, condition vs control: correct "
        "share control -> condition, effect [95% CI], McNemar p, BH q, fixed/broke, n, "
        "truncated change; MAE or F1 of finished pairs; FLAGGED when either side's "
        "truncated share is 15% or more")
  for arm in ARMS:
    for task in tasks:
      for control, conds in (("none", ["filler"] + PRIMERS), ("filler", PRIMERS)):
        rows = []
        for c in conds:
          j = pairs(f, arm, task, control, c, DENS4)
          fin = finished(j)
          sec = ""
          if task in ("node_count", "node_degree", "edge_count"):
            sec = (f"; MAE {fin.abs_error_a.mean():.2f} -> {fin.abs_error_b.mean():.2f} "
                   f"({len(fin)} finished pairs)")
          elif task == "connected_nodes":
            sec = (f"; F1 {fin.f1_a.mean():.3f} -> {fin.f1_b.mean():.3f} "
                   f"({len(fin)} finished pairs)")
          if max(j.truncated_a.mean(), j.truncated_b.mean()) >= outcomes.FLAG:
            sec += " FLAGGED"
          rows.append((c, j, effect(j), sec))
        q = bh([e[5] for _, _, e, _ in rows])
        print(f"  {arm} {task} vs {control}:")
        for (c, j, e, sec), qq in zip(rows, q):
          d, lo, hi, broke, fixed, p, n, dt = e
          print(f"    {c:10s} {100 * j.correct_a.mean():.2f} -> {100 * j.correct_b.mean():.2f}: "
                f"{f1(d)} [{f1(lo)}, {f1(hi)}] p={p:.2g} q={qq:.2g} fixed {fixed} "
                f"broke {broke} n={n} truncated {f1(dt)}{sec}")


def rwse_fit():
  """The fitted rwse density rule for edge_existence against the majority answer,
  on the graphs it was fitted on and on held-out graphs (the solver's settings:
  300 graphs, fit seed 555555, test seed 777777, rung 3)."""
  from graphtalk import shortcuts
  rule = next(r for r in shortcuts.FITTED if r.name == "edge_existence_rwse_density")
  print("[rwsefit] edge_existence, rwse density rule vs the majority answer (%): "
        "in-sample / out-of-sample / majority")
  for dens in DENSHI:
    fit = shortcuts.generate_n40_corpus(300, 555_555, dens)
    test = shortcuts.generate_n40_corpus(300, 777_777, dens)
    res = shortcuts.inflation("rwse", "edge_existence", 3, rule, fit, test)
    golds = [g for _, g in shortcuts.build_rows(test, "rwse", "edge_existence", 3, seed=6)]
    maj = max(golds.count(g) for g in set(golds)) / len(golds)
    print(f"  p={dens:.2f}: {100 * res['in_sample']:.1f} / {100 * res['out_of_sample']:.1f} / "
          f"{100 * maj:.1f}")


def solver_bars(bars):
  """The graph-blind solver's accuracy per (task, primer), from shortcuts_n40_flat.json."""
  from graphtalk import shortcuts
  print(f"[bars] graph-blind solver ({len(shortcuts.THEOREMS)} exact rules, "
        f"{len(shortcuts.HEURISTICS)} heuristic, {len(shortcuts.FITTED)} fitted on "
        "disjoint graphs) accuracy (%), n=40: task: none, then each primer")
  for task in TASKS4:
    print(f"  {task}: none {100 * bars[f'{task}/none']:.1f}, " + ", ".join(
        f"{c} {100 * bars[f'{task}/{c}']:.1f}" for c in ["filler"] + PRIMERS))
  side = {k: v for k, v in bars.items()
          if k.split("/")[0] in TASKS4 and v < apw.CARRIES}
  top = max(side, key=side.get)
  print(f"  carrying (>= {apw.CARRIES}): " + ", ".join(
      sorted(k for k, v in bars.items() if k.split("/")[0] in TASKS4 and v >= apw.CARRIES))
      + f"; highest other: {top} {100 * side[top]:.1f}")


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
    changed += record_correct(r) != record_correct(b)
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
  fixed = j[(j.correct_a == 0) & (j.correct_b == 1)]
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
        f"({hi.idxmax()[0]} {hi.idxmax()[1]})")


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
    ps.append(scoring.mcnemar(j.correct_a.astype(bool).to_numpy(),
                              j.correct_b.astype(bool).to_numpy())["p_value"])
  t["p"] = ps
  t["q"] = t.groupby(["arm", "task"]).p.transform(lambda s: bh(s.to_numpy()))
  t.to_csv(os.path.join(out, "primer_cells.csv"), index=False)
  collapse_rows(f).to_csv(os.path.join(out, "edge_existence_collapse.csv"), index=False)
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
  f = with_outcomes(pd.read_csv(args.frame))
  bars = json.load(open(BARS, encoding="utf-8"))
  if args.figure_data:
    figure_data(f, bars, args.csv_dir)
    return
  t = band_table(f, bars)
  setup_facts(f)
  solver_bars(bars)
  rwse_fit()
  main_table(f)
  per_arm(t)
  four_b_think_null(f)
  sign_flip(f)
  procedure(f)
  plain_small(f)
  recovery(f)
  edge_count(f)
  edge_existence(f)
  joint(f)
  clustering_high(f)
  replication()
  components(f)
  bundle(f)
  length(f)
  node_count(f)
  clustering_spread()
  rwse_resolution()
  measurement(f)
  power(f, t)
  position(f)
  leakage(f)
  rerun()
  other_procedures(f)
  extraction(f)
  if args.csv_dir:
    figure_data(f, bars, args.csv_dir)


if __name__ == "__main__":
  main()
