"""Quantities earlier drafts of the 40-node paper reported, recomputed from the
raw runs under the current scoring rule (R1, graphtalk/outcomes.py: a truncated
response is its own outcome, never correct and never dropped).

Each block names the draft that reported the quantity and the definition used
here. Where a draft's definition cannot be applied as written, the block says
what replaces it. The values the drafts printed are recomputed from the
archived outputs by superseded/scripts/reproduce_cut_claims.py, so both the old
and the current value can be traced. docs/results/n40-sweep.md cites these only
where it says so; the analyses behind the results are in primer_findings.py.

  PYTHONPATH=. python scripts/legacy_claims.py > csv2/raw-trends/legacy_claims.txt
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import primer_findings as pf  # noqa: E402

from graphtalk import scoring  # noqa: E402

TASKS6 = ["node_count", "node_degree", "connected_nodes", "edge_count",
          "edge_existence", "cycle_check"]
Z80 = 1.959964 + 0.841621   # two-sided .05, 80% power
ROUTE_GAIN = 0.05           # the drafts' route threshold on bar(cond) - bar(none)


def mde_points(discordance, n):
  """Smallest paired effect (points) detectable at 80% power, two-sided .05,
  from the discordance; floored at one discordant pair, since a cell with none
  cannot be tested at all."""
  return 100 * Z80 * np.sqrt(max(discordance, 1 / n) / n)


def route_gains(bars, tasks=pf.TASKS4, primers=pf.PRIMERS):
  """bar(task/cond) - bar(task/none) for every (task, primer)."""
  return {f"{t}/{c}": bars[f"{t}/{c}"] - bars[f"{t}/none"] for t in tasks for c in primers}


def rows_analysed(f):
  counts = {"40-node sweep": len(f)}
  for label, pattern in (("degdens40", "runs/qwen3-1.7b.degdens40.shard*.jsonl"),
                         ("degdens40hi", "runs/qwen3-1.7b.degdens40hi.shard*.jsonl"),
                         ("degdensrep", "runs/qwen3-1.7b.degdensrep.shard*.jsonl"),
                         ("degfixdeg", "runs/qwen3-1.7b.degfixdeg.shard*.jsonl")):
    counts[label] = len(pf.load_runs(pattern))
  print("[lrows] v3 'analysed' count; here: distinct rows that primer_findings.py reads: "
        + ", ".join(f"{k} {v}" for k, v in counts.items()) + f"; total {sum(counts.values())}")


def copy_test(f):
  """v1: when plain qwen3-4b is wrong under `degree`, its answer equals the
  degree stated for node k-1 or k+1 (its neighbours in the primer's list) more
  often than chance; one-sided binomial. Here: finished wrong answers; the
  background is the mean per-answer chance that one of those two lines states
  the answer, from the rate at which the other nodes' stated degrees equal it
  (v1's own background, 17.3%, came from a script that was never committed)."""
  deg = re.compile(r"Node (\d+) has degree (\d+)")
  stated = {}
  for path in ("prompts.densfull40.jsonl", "prompts.densfull40hi.jsonl"):
    for line in open(path, encoding="utf-8"):
      r = json.loads(line)
      if r["task"] == "node_degree" and r["condition"] == "degree":
        stated[r["instance_id"]] = {int(a): int(b) for a, b in deg.findall(r["prompt"])}
  d = f[(f.arm == "qwen3-4b") & (f.task == "node_degree") & (f.condition == "degree")
        & (f.hit_cap == 0) & (f.exact == 0)].copy()
  d["v"] = pd.to_numeric(d.pred, errors="coerce")
  d = d[d.v.notna()]
  print("[lcopy] v1 copy test: plain qwen3-4b, node_degree under degree, finished wrong answers "
        "equal to the degree stated for node k-1 or k+1")
  for label, dens in (("main sweep, p<=.50", pf.DENS4), ("all seven densities", pf.DENS4 + pf.DENSHI)):
    s = d[d.density_class.isin(dens)]
    copied, chance = 0, []
    for iid, k, v in zip(s.instance_id, s.target_id.astype(int), s.v.astype(int)):
      sd = stated[iid]
      near = [j for j in (k - 1, k + 1) if j in sd]
      copied += any(sd[j] == v for j in near)
      q = np.mean([sd[j] == v for j in sd if j != k and j not in near])
      chance.append(1 - (1 - q) ** len(near))
    bg = float(np.mean(chance))
    p = stats.binomtest(copied, len(s), bg, alternative="greater").pvalue
    print(f"  {label}: {copied} of {len(s)}; background {100 * bg:.1f}%; one-sided binomial "
          f"p={p:.2g}")


def detectability(f):
  """v1/v2: among the null cells (every arm x task x primer against `none`,
  main sweep, 400 paired graphs each, Benjamini-Hochberg q >= .05 within arm
  and task; comparisons under the truncation flag only), the share where
  the no-primer correct share leaves less room in some direction than the
  smallest detectable effect, and the median detectable gain where a gain is
  detectable. The drafts simulated power per cell; here the detectable effect
  is the analytic value [power] uses (mde_points)."""
  rows = []
  for arm in pf.ARMS:
    for task in TASKS6:
      group = []
      for c in ["filler"] + pf.PRIMERS:
        j = pf.pairs(f, arm, task, "none", c, pf.DENS4)
        a, b = j.correct_a.astype(bool).to_numpy(), j.correct_b.astype(bool).to_numpy()
        group.append(dict(arm=arm, task=task, condition=c, n=len(j),
                          p=scoring.mcnemar(a, b)["p_value"], base=a.mean(),
                          disc=(a != b).mean(),
                          flagged=max(j.truncated_a.mean(), j.truncated_b.mean()) >= 0.15))
      for row, q in zip(group, pf.bh([g["p"] for g in group])):
        row["q"] = q
      rows += group
  t = pd.DataFrame(rows)
  null = t[(t.q >= 0.05) & ~t.flagged].copy()
  null["mde"] = [mde_points(d, n) for d, n in zip(null.disc, null.n)]
  null["no_gain"] = 100 * (1 - null.base) < null.mde
  null["no_harm"] = 100 * null.base < null.mde
  print("[lmde] v1/v2 null comparisons against none (400 paired graphs each, under the "
        "truncation flag): per arm, null comparisons, share where a gain or a harm cannot "
        "be detected (floor or ceiling), median detectable gain (points) where a gain can be")
  for arm in pf.ARMS:
    s = null[null.arm == arm]
    either = s.no_gain | s.no_harm
    print(f"  {arm:17s} {len(s)} of {int(((t.arm == arm) & ~t.flagged).sum())} null; floor or ceiling "
          f"{int(either.sum())}/{len(s)}; median detectable gain "
          f"{pf.f1(s[~s.no_gain].mde.median(), sign=False)}")
  for size in ("1.7b", "4b"):
    s = null[null.arm.str.startswith("qwen3-" + size)]
    print(f"  qwen3-{size} arms pooled: median detectable gain "
          f"{pf.f1(s[~s.no_gain].mde.median(), sign=False)} over {int((~s.no_gain).sum())} cells")


def thinking_baseline(f):
  """v2: qwen3-1.7b-think's no-primer accuracy on node_degree and
  connected_nodes, as the unweighted mean of its per-density correct shares."""
  out = []
  for task, dens in (("node_degree", pf.DENS4 + pf.DENSHI), ("connected_nodes", pf.DENS4)):
    d = f[(f.arm == "qwen3-1.7b-think") & (f.task == task) & (f.condition == "none")]
    per = [d[d.density_class == p].correct.mean() for p in dens]
    out.append(f"{task} {np.mean(per):.3f} over {len(dens)} densities")
  print("[lbase] v2 qwen3-1.7b-think no-primer accuracy, unweighted mean of per-density "
        "correct shares: " + "; ".join(out))


def length_cost(f):
  """v2: the filler cost (filler minus none, points) against the no-primer
  correct share, per arm, over the (task, density) cells under the truncation
  flag; and plain qwen3-1.7b's filler cost on edge_existence by density."""
  print("[llen] v2 filler cost vs no-primer accuracy: Pearson r per arm over (task, density) "
        "cells under the truncation flag")
  for arm in pf.ARMS:
    cost, base = [], []
    for task in pf.TASKS4:
      for dens in sorted(f[(f.arm == arm) & (f.task == task)].density_class.unique()):
        j = pf.pairs(f, arm, task, "none", "filler", [dens])
        if max(j.truncated_a.mean(), j.truncated_b.mean()) >= 0.15:
          continue
        cost.append(100 * (j.correct_b.mean() - j.correct_a.mean()))
        base.append(j.correct_a.mean())
    r, p = stats.pearsonr(base, cost)
    print(f"  {arm:17s} r={r:+.2f} (p={p:.2g}) over {len(cost)} cells")
  print("  qwen3-1.7b edge_existence, filler minus none by density: " + ", ".join(
      pf.f1(100 * (j.correct_b.mean() - j.correct_a.mean()))
      for j in (pf.pairs(f, "qwen3-1.7b", "edge_existence", "none", "filler", [p])
                for p in pf.DENS4 + pf.DENSHI)))


def crossfit(f, bars):
  """v2 and the short draft: route cells (the solver bar exceeds `none`'s by
  more than 0.05, the drafts' definition) on node_degree and edge_count. Each
  cell's graphs split into two folds by graph index parity; the baseline (the
  no-primer correct share) from one fold and the effect from the other, both
  ways, so each cell gives two points. Pearson r and OLS slope of the effect
  (points) on the baseline, per task and pooled, over cells under the
  truncation flag."""
  gains = route_gains(bars, tasks=("node_degree", "edge_count"))
  route = [k for k, g in gains.items() if g > ROUTE_GAIN]
  points = []
  for arm in pf.ARMS:
    for key in route:
      task, c = key.split("/")
      for dens in sorted(f[(f.arm == arm) & (f.task == task)].density_class.unique()):
        j = pf.pairs(f, arm, task, "none", c, [dens])
        if max(j.truncated_a.mean(), j.truncated_b.mean()) >= 0.15:
          continue
        for fold in (0, 1):
          a, b = j[j.index_a % 2 == fold], j[j.index_a % 2 != fold]
          points.append((task, a.correct_a.mean(), 100 * (b.correct_b.mean() - b.correct_a.mean())))
  t = pd.DataFrame(points, columns=["task", "base", "eff"])
  print(f"[lcross] v2/short cross-fitted baseline vs effect on route cells ({', '.join(route)}), "
        "cells under the truncation flag: points, r, p, slope (points per unit baseline)")
  for label, s in (("node_degree", t[t.task == "node_degree"]),
                   ("edge_count", t[t.task == "edge_count"]), ("pooled", t)):
    r, p = stats.pearsonr(s.base, s.eff)
    slope = np.polyfit(s.base, s.eff, 1)[0]
    print(f"  {label:11s} {len(s)} points, r={r:+.3f} (p={p:.2g}), slope {pf.f1(slope)}")


def route_threshold(bars):
  """The short draft: with route = bar gain over `none` above 0.05, the largest
  gain at or below the threshold and the smallest above it."""
  gains = route_gains(bars)
  below = max((g, k) for k, g in gains.items() if g <= ROUTE_GAIN)
  above = min((g, k) for k, g in gains.items() if g > ROUTE_GAIN)
  print(f"[lroute] short-draft route threshold 0.05 on bar(cond) - bar(none): largest gain "
        f"at or below {below[0]:.3f} ({below[1]}), smallest above {above[0]:.3f} ({above[1]})")


def main():
  f = pf.with_outcomes(pd.read_csv(pf.FRAME))
  bars = json.load(open(pf.BARS, encoding="utf-8"))
  rows_analysed(f)
  copy_test(f)
  detectability(f)
  thinking_baseline(f)
  length_cost(f)
  crossfit(f, bars)
  route_threshold(bars)


if __name__ == "__main__":
  main()
