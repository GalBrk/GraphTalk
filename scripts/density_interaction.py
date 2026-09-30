"""Does edge density change how a primer affects accuracy in the 40-node sweep,
and does it do so beyond how hard the task already is without a primer?

Read from outputs/n40-sweep/frame.csv (scripts/build_raw_frame.py). Accuracy
follows rule R1 (graphtalk/outcomes.py): a truncated response is never correct.
The four tasks whose answer varies; every primer against none on the same graphs.

  [dxhet]    for each arm x task x primer, the effect at p = .10 .20 .35 .50 and
             whether it differs across the four densities: a permutation test of
             the between-density spread of the paired effect (density labels
             shuffled across graphs, each graph's pair intact), BH over all
             tests. Also the effect's slope per +0.1 of density with its own
             permutation p. ~ marks a test where a density has 15%+ truncated
             responses on either side
  [dxvary]   the effects that vary with density at q < .05
  [dxhethi]  (and [dxvaryhi]) the same over all seven densities for node_degree and edge_existence
             (the plain arms ran p >= .65 with 2048 tokens, a quarter of the
             main sweep's budget, so density and budget move together there)
  [dxbase]   density beyond difficulty: each (arm, task, density, primer) cell's
             effect against its no-primer accuracy band and its density. The band
             comes from half of the cell's graphs and the effect from the other
             half (no regression to the mean); the density coefficient (points
             per +0.1 of density at a fixed band) is tested by shuffling density
             among cells of the same band. Answer-carrying and side-information
             primers apart, and per task

  PYTHONPATH=. python scripts/density_interaction.py > outputs/n40-sweep/density_interaction.txt
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_primer_window as apw  # noqa: E402  (BANDS, CARRIES)
import primer_findings as pf  # noqa: E402

from graphtalk import outcomes  # noqa: E402

ARMS, TASKS4, MAIN, HI = pf.ARMS, pf.TASKS4, pf.DENS4, pf.DENSHI
CONDS = ["filler"] + pf.PRIMERS
SHORT = {"qwen3-1.7b": "1.7B", "qwen3-1.7b-think": "1.7B-T", "qwen3-4b": "4B",
         "qwen3-4b-think": "4B-T"}
B = pf.B


# ------------------------------------------------------------------ statistics
def between_ss(x, groups):
  """sum over groups of n_g * (mean_g - overall mean)^2: how far the mean of x
  moves across the groups (0 when every group has the same mean)."""
  x, groups = np.asarray(x, float), np.asarray(groups)
  _, g = np.unique(groups, return_inverse=True)
  n, s = np.bincount(g), np.bincount(g, weights=x)
  return float((n * (s / n - x.mean()) ** 2).sum())


def slope(x, p):
  """Least-squares change in x per +0.1 of p, in the units of x."""
  x, p = np.asarray(x, float), np.asarray(p, float)
  pc = p - p.mean()
  return 0.1 * float((pc * (x - x.mean())).sum() / (pc ** 2).sum())


def perm_pvalues(x, p, seed=pf.SEED, draws=B):
  """Permutation p-values of the between-density spread and of |slope|, shuffling
  the density labels across rows (each row one graph's paired difference)."""
  x, p = np.asarray(x, float), np.asarray(p, float)
  rng = np.random.default_rng(seed)
  h0, s0 = between_ss(x, p), abs(slope(x, p))
  hits_h = hits_s = 0
  for _ in range(draws):
    q = rng.permutation(p)
    hits_h += between_ss(x, q) >= h0 - 1e-12
    hits_s += abs(slope(x, q)) >= s0 - 1e-12
  return (hits_h + 1) / (draws + 1), (hits_s + 1) / (draws + 1)


def density_coefficient(delta, band, p):
  """Coefficient of p in delta ~ band (one level per band) + p, per +0.1 of p."""
  delta, p = np.asarray(delta, float), np.asarray(p, float)
  levels = sorted(set(band))
  X = np.column_stack([np.asarray(band) == b for b in levels] + [p]).astype(float)
  coef = np.linalg.lstsq(X, delta, rcond=None)[0]
  return 0.1 * float(coef[-1])


def within_band_pvalue(delta, band, p, seed=pf.SEED, draws=B):
  """Permutation p-value of |density_coefficient| when density is shuffled among
  cells of the same band (bands with a single density carry no information)."""
  delta, band, p = np.asarray(delta, float), np.asarray(band), np.asarray(p, float)
  rng = np.random.default_rng(seed)
  c0 = abs(density_coefficient(delta, band, p))
  idx = [np.flatnonzero(band == b) for b in sorted(set(band))]
  hits = 0
  for _ in range(draws):
    q = p.copy()
    for i in idx:
      q[i] = p[rng.permutation(i)]
    hits += abs(density_coefficient(delta, band, q)) >= c0 - 1e-12
  return (hits + 1) / (draws + 1)


def col(d):
  """Column name for the effect at density d ('e10' for p = .10)."""
  return f"e{int(round(100 * d)):02d}"


def band_of(baseline):
  """The apw.BANDS label a no-primer accuracy falls in."""
  for lo, hi in apw.BANDS:
    if lo <= baseline < hi:
      return f"{lo:.2f}-{min(hi, 1.0):.2f}"
  return f"{apw.BANDS[-1][0]:.2f}-1.00"


# ------------------------------------------------------------------ [dxhet]
def heterogeneity(f, dens):
  rows = []
  for arm in ARMS:
    for task in TASKS4:
      if not set(dens) <= set(f[f.task == task].density_class.unique()):
        continue
      dt = f[(f.arm == arm) & (f.task == task)]
      for c in CONDS:
        j = pf.pairs(dt, arm, task, "none", c, dens)
        x = 100 * (j.correct_b - j.correct_a).to_numpy(float)
        p = j.index.get_level_values(0).to_numpy(float)
        per = {d: x[p == d].mean() for d in dens}
        trunc = max(max(j.truncated_a[p == d].mean(), j.truncated_b[p == d].mean())
                    for d in dens)
        ph, ps = perm_pvalues(x, p)
        rows.append(dict(arm=arm, task=task, condition=c, n=len(j),
                         **{col(d): per[d] for d in dens},
                         spread=max(per.values()) - min(per.values()),
                         slope=slope(x, p), p_het=ph, p_slope=ps,
                         flag=trunc >= outcomes.FLAG))
  r = pd.DataFrame(rows)
  r["q_het"] = pf.bh(r.p_het.to_numpy())
  return r


def print_het(r, dens, tag, what):
  cols = [col(d) for d in dens]
  print(f"[{tag}] {what}: the effect (points, R1) at p = "
        + " ".join(f"{d:.2f}" for d in dens) + "; its largest minus smallest; the "
        "between-density test (p, and q with BH over all "
        f"{len(r)} tests); the slope per +0.1 of density (permutation p); ~ = a density "
        "with 15%+ truncated on either side")
  for task in TASKS4:
    for arm in ARMS:
      for x in r[(r.task == task) & (r.arm == arm)].itertuples():
        per = " ".join(f"{getattr(x, c):+5.1f}" for c in cols)
        print(f"  {task:15s} {SHORT[arm]:6s} {x.condition:10s} {per} | spread "
              f"{x.spread:4.1f} | p={x.p_het:.3g} q={x.q_het:.2g} | slope "
              f"{x.slope:+.1f} (p={x.p_slope:.2g}){' ~' if x.flag else ''}")


def print_vary(r, dens, tag="dxvary"):
  v = r[r.q_het < .05].sort_values(["task", "arm", "condition"])
  clean, flagged = v[~v.flag], v[v.flag]
  print(f"[{tag}] effects that vary with density at q < .05: {len(v)} of {len(r)} "
        f"tests, {len(clean)} of them under the truncation flag, {len(flagged)} flagged")
  for x in v.itertuples():
    per = ", ".join(f"{getattr(x, col(d)):+.0f}" for d in dens)
    print(f"  {x.task} {SHORT[x.arm]} {x.condition}: {per} (slope {x.slope:+.1f} per +0.1; "
          f"q={x.q_het:.2g}){' ~' if x.flag else ''}")


# ------------------------------------------------------------------ [dxbase]
def split_cells(f, bars, dens, seed=pf.SEED):
  """Cells under the truncation flag, each with its baseline measured on half its
  graphs and its effect (points) on the other half, the halves fixed by seed."""
  rng = np.random.default_rng(seed)
  out = []
  for arm in ARMS:
    for task in TASKS4:
      dt = f[(f.arm == arm) & (f.task == task)]
      for d in dens:
        for c in CONDS:
          j = pf.pairs(dt, arm, task, "none", c, [d])
          if not len(j) or max(j.truncated_a.mean(), j.truncated_b.mean()) >= outcomes.FLAG:
            continue
          a, b = j.correct_a.to_numpy(float), j.correct_b.to_numpy(float)
          perm = rng.permutation(len(a))
          h1, h2 = perm[: len(a) // 2], perm[len(a) // 2:]
          out.append(dict(arm=arm, task=task, density=d, condition=c,
                          carries=bars.get(f"{task}/{c}", 0) >= apw.CARRIES,
                          baseline=a[h1].mean(), delta=100 * (b[h2].mean() - a[h2].mean())))
  t = pd.DataFrame(out)
  t["band"] = t.baseline.map(band_of)
  return t


def print_base(t, label):
  print(f"[dxbase] {label}: density coefficient at a fixed no-primer band (points of "
        "effect per +0.1 of density), permutation p with density shuffled within "
        "bands; cells under the truncation flag, band from half the graphs and effect "
        "from the other half")
  groups = [("answer-carrying", t.carries), ("side information", ~t.carries)]
  groups += [(f"side information, {task}", (~t.carries) & (t.task == task)) for task in TASKS4]
  for name, m in groups:
    s = t[m]
    if s.density.nunique() < 2:
      continue
    coef = density_coefficient(s.delta, s.band, s.density)
    p = within_band_pvalue(s.delta, s.band, s.density)
    print(f"  {name:32s} {len(s):3d} cells: {coef:+.2f} per +0.1 (p={p:.2g})")
  print("  mean effect by band (rows) and density (columns), side information "
        "(answer-carrying): cells")
  dens = sorted(t.density.unique())
  for band in sorted(t.band.unique()):
    s = t[t.band == band]
    cells = []
    for d in dens:
      side, car = s[(s.density == d) & ~s.carries], s[(s.density == d) & s.carries]
      cells.append((f"{side.delta.mean():+5.1f}({len(side)})" if len(side) else "    -   ")
                   + (f" [{car.delta.mean():+.0f}({len(car)})]" if len(car) else ""))
    print(f"    {band}: " + " | ".join(f"p{d:.2f} {c}" for d, c in zip(dens, cells)))


def main():
  argparse.ArgumentParser(description=__doc__,
                          formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
  sys.stdout.reconfigure(encoding="utf-8")
  f = pf.with_outcomes(pd.read_csv(pf.FRAME))
  with open(pf.BARS) as fh:
    bars = json.load(fh)
  r = heterogeneity(f, MAIN)
  print_het(r, MAIN, "dxhet", "main sweep, four tasks, 400 paired graphs per test")
  print_vary(r, MAIN)
  rh = heterogeneity(f, MAIN + HI)
  print_het(rh, MAIN + HI, "dxhethi", "node_degree and edge_existence over all seven "
            "densities (plain arms at 2048 tokens from p = .65)")
  print_vary(rh, MAIN + HI, "dxvaryhi")
  print_base(split_cells(f, bars, MAIN), "main sweep")
  print_base(split_cells(f, bars, MAIN + HI), "main sweep plus p >= .65 for "
             "node_degree and edge_existence (plain arms' budget changes there)")


if __name__ == "__main__":
  main()
