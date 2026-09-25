"""Score a run set whose instance ids carry its design level: outcome shares,
error size, and paired effects against a control, per level.

Levels come from the instance id (`node_degree/size80/p0.101/17`), grouped by
`--group`:

  density   0.101                  one graph size, several densities
  cell      (80, 0.101)            size and density: the fixed-mean-degree design
                                   has two cells that share a density
  task      ("node_degree", 0.101) runs that ask more than one task

Scoring follows rule R1 (graphtalk/outcomes.py): every response is correct,
wrong or truncated; a truncated response is never counted wrong and never
dropped. Exact match is the primary score for every task, so connected_nodes
is scored by exact set match. A primer's effect is the paired change, on the
same instances, in the share of all responses that is correct, with the change
in the truncated share beside it. A level whose control or primer cell has 15%
or more truncated responses is flagged: it is listed, and left out of pooled
effects and trends. Error size and response length use finished responses.

`--versus GLOB` pairs this run set with a second one on (instance id,
condition) and reports the second minus the first (e.g. a thinking arm minus
its plain arm on the same graphs).

Output is plain text under a `[tag]` line (`--tag`), so a results doc can cite
it; scripts/density_followups.py runs the density follow-up family through it.

  PYTHONPATH=. python scripts/score_density_sweep.py \\
      --responses "runs/qwen3-1.7b.degdens40.shard*of5.jsonl" --tag dd40
"""
import argparse
import collections
import glob
import json
import random
import statistics
import sys

import os

import numpy as np

from graphtalk import outcomes, scoring

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_baseline_law as abl  # noqa: E402  (pearson: scipy-free)

CONTROL = "none"
SEED = 20260925
B = 2000
Z80 = 1.959964 + 0.841621


def f1(x, sign=True):
  """One decimal, ties to even on the decimal value (as primer_findings.f1)."""
  from decimal import ROUND_HALF_EVEN, Decimal
  if x != x:
    return "nan"
  q = Decimal(repr(round(float(x), 9))).quantize(Decimal("0.1"), rounding=ROUND_HALF_EVEN)
  return f"{q:+}" if sign else f"{q}"


def level_of(instance_id, group, task):
  size = density = None
  for part in instance_id.split("/"):
    if part.startswith("size") and part[4:].isdigit():
      size = int(part[4:])
    elif len(part) > 1 and part[0] == "p":
      try:
        density = float(part[1:])
      except ValueError:
        continue
  if group == "density":
    return density
  if group == "cell":
    return (size, density)
  if group == "task":
    return (task, density)
  raise ValueError(f"unknown group {group!r}")


def size_of(instance_id):
  return next(int(part[4:]) for part in instance_id.split("/")
              if part.startswith("size") and part[4:].isdigit())


def mean_degree(cell):
  """Target mean degree of a (size, density) cell: p * (n - 1), rounded."""
  size, density = cell
  return round(density * (size - 1))


def load(patterns):
  """Rows of these files, one per (instance id, condition). Identical
  duplicates collapse; two different rows for one key are an error."""
  rows = {}
  for pattern in patterns:
    for path in sorted(glob.glob(pattern)) or [pattern]:
      with open(path, encoding="utf-8") as handle:
        for line in handle:
          if line.strip():
            r = json.loads(line)
            key = (r["instance_id"], r["condition"])
            if key in rows and rows[key] != r:
              raise ValueError(f"{path}: two different rows for {key}")
            rows.setdefault(key, r)
  return list(rows.values())


def summarize(records, group):
  cells = collections.defaultdict(
      lambda: {"n": 0, "correct": 0, "truncated": 0, "errors": [], "tokens": [], "golds": [],
               "task": None, "nminus1": 0})
  paired = collections.defaultdict(dict)   # (level, instance id) -> condition -> (correct, truncated)
  golds = collections.defaultdict(list)    # level -> gold per instance (control rows)
  for r in records:
    level = level_of(r["instance_id"], group, r["task"])
    pred = scoring.extract_answer(r["response"] or "", r["task"])
    res = scoring.score_one(pred, r["gold"], r["task"])
    o = outcomes.outcome(res["exact"], bool(r.get("hit_cap")))
    correct, truncated = int(o == outcomes.CORRECT), int(o == outcomes.TRUNCATED)
    cell = cells[(level, r["condition"])]
    cell["n"] += 1
    cell["correct"] += correct
    cell["truncated"] += truncated
    cell["golds"].append(r["gold"])
    cell["task"] = r["task"]
    if not truncated:
      # node_degree's saturated answer: the degree of a node joined to every
      # other node (39 at n=40).
      cell["nminus1"] += str(pred).strip() == str(size_of(r["instance_id"]) - 1)
      if res["absolute_error"] is not None:
        cell["errors"].append(res["absolute_error"])
      if r.get("n_new_tokens") is not None:
        cell["tokens"].append(r["n_new_tokens"])
    paired[(level, r["instance_id"])][r["condition"]] = (correct, truncated)
    if r["condition"] == CONTROL:
      golds[level].append(r["gold"])
  return {"cells": cells, "paired": paired, "golds": golds}


def blind_bar(golds):
  """The best constant answer and the share of golds it matches."""
  answer, hits = collections.Counter(golds).most_common(1)[0]
  return answer, hits / len(golds)


def pairs_for(paired, condition, levels, control=CONTROL):
  """(level, control outcome, primer outcome) for every instance with both."""
  return [(lvl, v[control], v[condition]) for (lvl, _), v in sorted(paired.items(), key=str)
          if (levels is None or lvl in levels) and control in v and condition in v]


def versus_pairs(first, second, condition):
  """(level, first set's outcome, second set's outcome) on shared instances."""
  return [(lvl, v[condition], second[key][condition])
          for key, v in sorted(first.items(), key=str)
          for lvl in [key[0]]
          if condition in v and key in second and condition in second[key]]


def effect(rows, seed=SEED, draws=B):
  """Paired change in the correct share (points) with a bootstrap interval over
  instances within level, exact McNemar, fixed/broke, and the change in the
  truncated share."""
  a = np.array([r[1][0] for r in rows])
  b = np.array([r[2][0] for r in rows])
  ta = np.array([r[1][1] for r in rows])
  tb = np.array([r[2][1] for r in rows])
  m = scoring.mcnemar(a.astype(bool), b.astype(bool))
  levels = [r[0] for r in rows]
  groups = [np.flatnonzero([lv == g for lv in levels]) for g in sorted(set(levels), key=str)]
  rng = np.random.default_rng(seed)
  diff = b - a
  vals = [diff[np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])].mean()
          for _ in range(draws)]
  lo, hi = np.percentile(vals, [2.5, 97.5])
  return {"n": len(rows), "d": 100 * diff.mean(), "lo": 100 * lo, "hi": 100 * hi,
          "fixed": m["c"], "broke": m["b"], "p": m["p_value"],
          "dt": 100 * (tb.mean() - ta.mean()), "disc": float((a != b).mean())}


def share(cell, key):
  return cell[key] / cell["n"] if cell["n"] else float("nan")


def flagged_levels(cells, condition, control=CONTROL):
  """Levels where the control or the primer cell has 15% or more truncated."""
  levels = {lvl for lvl, c in cells if c == condition}
  return {lvl for lvl in levels
          if max(share(cells[(lvl, control)], "truncated"),
                 share(cells[(lvl, condition)], "truncated")) >= outcomes.FLAG}


def trend_test(paired, condition, levels=None, control=CONTROL, draws=20000, seed=0):
  """Permutation test on the slope of the paired correct difference against
  density: density labels are permuted across instances."""
  xs, ys = [], []
  for (density, _), v in paired.items():
    if levels is not None and density not in levels:
      continue
    if control in v and condition in v:
      xs.append(density)
      ys.append(v[condition][0] - v[control][0])
  if len(set(xs)) < 2:
    return {"slope": float("nan"), "p_value": 1.0, "n": len(xs)}
  mx, my = statistics.mean(xs), statistics.mean(ys)
  slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
           / sum((x - mx) ** 2 for x in xs))
  # The statistic |n*sum(xy) - sum(x)*sum(y)| in integers (density in
  # thousandths), so a permutation that ties the observed value counts exactly.
  ix, n = [round(x * 1000) for x in xs], len(xs)
  centre = sum(ix) * sum(ys)
  target = abs(n * sum(x * y for x, y in zip(ix, ys)) - centre)
  rng, shuffled, hits = random.Random(seed), list(ix), 0
  for _ in range(draws):
    rng.shuffle(shuffled)
    hits += abs(n * sum(x * y for x, y in zip(shuffled, ys)) - centre) >= target
  return {"slope": slope, "p_value": (hits + 1) / (draws + 1), "n": n}


def _fmt_level(level):
  if isinstance(level, tuple) and isinstance(level[0], int):
    return f"size={level[0]} p={level[1]:g} (d~{mean_degree(level)})"
  if isinstance(level, tuple):
    return f"{level[0]} p={level[1]:g}"
  return f"p={level:g}"


def _effect_line(label, e, q=None):
  qs = f" q={q:.2g}" if q is not None else ""
  return (f"  {label}: {f1(e['d'])} [{f1(e['lo'])}, {f1(e['hi'])}] p={e['p']:.2g}{qs} "
          f"fixed {e['fixed']} broke {e['broke']} n={e['n']} truncated {f1(e['dt'])}")


def report(summary, tag, title, control=CONTROL, pools=None, trend=False,
           continuum=False, blocks=False, headroom=False, gold_means=False, out=sys.stdout):
  """Print one tagged block. `pools` is a list of (label, levels or None);
  each gets its own pooled effects and BH family (default: all levels)."""
  cells, paired, golds = summary["cells"], summary["paired"], summary["golds"]
  levels = sorted({lvl for lvl, _ in cells})
  conditions = sorted({c for _, c in cells})
  primers = [c for c in conditions if c != control]
  p = lambda s="": print(s, file=out)
  p(f"[{tag}] {title}")
  p(f"  outcome shares by level and condition (correct/wrong/truncated %, n; FLAGGED "
    f"when 15% or more truncated); MAE and median response tokens of finished responses")
  for lvl in levels:
    parts = []
    for c in conditions:
      cell = cells.get((lvl, c))
      if not cell or not cell["n"]:
        continue
      cr, tr = share(cell, "correct"), share(cell, "truncated")
      mae = f" MAE {statistics.mean(cell['errors']):.2f}" if cell["errors"] else ""
      tok = f" tokens {statistics.median(cell['tokens']):.0f}" if cell["tokens"] else ""
      finished = cell["n"] - cell["truncated"]
      if cell["task"] == "node_degree" and finished:
        tok += f" answers n-1 {100 * cell['nminus1'] / finished:.1f}%"
      flag = " FLAGGED" if tr >= outcomes.FLAG else ""
      fin = f" finished {cell['n'] - cell['truncated']}" if cell["truncated"] else ""
      parts.append(f"{c} {100 * cr:.1f}/{100 * (1 - cr - tr):.1f}/{100 * tr:.1f} "
                   f"n={cell['n']}{fin}{mae}{tok}{flag}")
    p(f"    {_fmt_level(lvl)}: " + "; ".join(parts))
  p("  blind bar (the best constant answer's share of golds, per level): " + ", ".join(
      f"{_fmt_level(lvl)} {blind_bar(golds[lvl])[1]:.3f}" for lvl in levels if golds.get(lvl)))
  if gold_means:
    p("  mean gold under the control, per level: " + ", ".join(
        f"{_fmt_level(lvl)} {np.mean([float(g) for g in golds[lvl]]):.2f}"
        for lvl in levels if golds.get(lvl)))
  if not primers:
    return
  use = set(levels)
  for label, pool in pools or [("all levels", None)]:
    pl = set(levels) if pool is None else set(pool) & set(levels)
    pooled, empty = [], []
    for c in primers:
      ok = pl - flagged_levels(cells, c, control)
      rows = pairs_for(paired, c, ok, control)
      if rows:
        pooled.append((c, effect(rows), pl - ok))
      elif pl - ok:
        empty.append((c, pl - ok))
    p(f"  pooled over {label}, flagged levels left out, vs {control} (BH q over the "
      f"{len(pooled)} primers):")
    for (c, e, dropped), keep in zip(pooled, bh_q([e["p"] for _, e, _ in pooled])):
      mde = f1(100 * Z80 * np.sqrt(max(e["disc"], 1 / e["n"]) / e["n"]), sign=False)
      p(_effect_line(f"{label} {c}", e, keep) + f" MDE {mde}" + _left_out(dropped))
    for c, dropped in empty:
      p(f"  {label} {c}: every level flagged" + _left_out(dropped))
  family = lambda lvl: lvl[0] if isinstance(lvl, tuple) and isinstance(lvl[0], str) else ""
  p(f"  per level vs {control} (BH q within this family"
    + (", one family per task):" if any(family(lvl) for lvl in levels) else "):"))
  per = [(lvl, c, effect(pairs_for(paired, c, {lvl}, control)))
         for lvl in levels for c in primers if pairs_for(paired, c, {lvl}, control)]
  qs = {}
  for fam in {family(lvl) for lvl, _, _ in per}:
    idx = [k for k, (lvl, _, _) in enumerate(per) if family(lvl) == fam]
    qs.update(zip(idx, bh_q([per[k][2]["p"] for k in idx])))
  for k, (lvl, c, e) in enumerate(per):
    flag = " FLAGGED" if lvl in flagged_levels(cells, c, control) else ""
    p(_effect_line(f"{_fmt_level(lvl)} {c}", e, qs[k]) + flag)
  if blocks:
    p("  pooled per mean-degree block, flagged cells left out:")
    for d in sorted({mean_degree(lvl) for lvl in levels}):
      block = {lvl for lvl in levels if mean_degree(lvl) == d}
      for c in primers:
        ok = block - flagged_levels(cells, c, control)
        rows = pairs_for(paired, c, ok, control)
        if rows:
          p(_effect_line(f"d~{d} {c}", effect(rows)) + _left_out(block - ok))
  if trend:
    for label, pool in pools or [("all levels", None)]:
      pl = use if pool is None else set(pool) & use
      p(f"  trend over {label}: slope of the paired correct difference against density, "
        "unflagged levels (20,000 label permutations):")
      for c in primers:
        dropped = pl & flagged_levels(cells, c, control)
        t = trend_test(paired, c, pl - dropped, control)
        if t["n"] and t["slope"] == t["slope"]:
          p(f"    {c}: slope {t['slope']:+.3f} per unit density, p={t['p_value']:.2g}, "
            f"n={t['n']}" + _left_out(dropped))
        elif dropped:
          p(f"    {c}: fewer than two unflagged levels" + _left_out(dropped))
  if continuum:
    p("  continuum: Pearson r between each level's effect and its control correct "
      "share, over unflagged levels")
    for c in primers:
      dropped = use & flagged_levels(cells, c, control)
      ok = [lvl for lvl in sorted(use - dropped) if pairs_for(paired, c, {lvl}, control)]
      xs = [share(cells[(lvl, control)], "correct") for lvl in ok]
      ys = [effect(pairs_for(paired, c, {lvl}, control), draws=1)["d"] for lvl in ok]
      if len(ok) >= 3:
        r, pv = abl.pearson(xs, ys)
        p(f"    {c}: r={r:+.2f} (p={pv:.2g}) over {len(ok)} levels" + _left_out(dropped))
  if headroom and "clustering" in conditions and "degree" in conditions:
    p("  headroom captured by clustering, (clustering - none) / (degree - none) on the "
      "correct share, per level:")
    for lvl in (levels if headroom is True else [lvl for lvl in levels if lvl in headroom]):
      n_, c_, d_ = (share(cells[(lvl, k)], "correct") for k in (control, "clustering", "degree"))
      gap = d_ - n_
      p(f"    {_fmt_level(lvl)}: none {100 * n_:.1f} clustering {100 * c_:.1f} "
        f"degree {100 * d_:.1f} captured {f1(100 * (c_ - n_) / gap) if gap else 'n/a'}%")


def _left_out(levels):
  return (" (left out: " + ", ".join(map(_fmt_level, sorted(levels))) + ")") if levels else ""


def bh_q(pvals):
  """Benjamini-Hochberg adjusted q-values."""
  p = np.asarray(pvals, float)
  if not len(p):
    return []
  o = np.argsort(p)
  r = p[o] * len(p) / np.arange(1, len(p) + 1)
  r = np.minimum.accumulate(r[::-1])[::-1]
  q = np.empty(len(p))
  q[o] = np.clip(r, 0, 1)
  return list(q)


def report_versus(first, second, tag, title, levels=None, out=sys.stdout):
  """Second run set minus the first, on shared (instance, condition). A level
  where either run set's cell is 15% or more truncated is FLAGGED and left out
  of the pooled line."""
  print(f"[{tag}] {title}", file=out)
  conditions = sorted({c for v in first["paired"].values() for c in v})
  rows_all = []
  for c in conditions:
    rows = [r for r in versus_pairs(first["paired"], second["paired"], c)
            if levels is None or r[0] in levels]
    flagged = {lvl for lvl in {r[0] for r in rows}
               if any(share(s["cells"][(lvl, c)], "truncated") >= outcomes.FLAG
                      for s in (first, second))}
    if rows:
      pooled = [r for r in rows if r[0] not in flagged]
      if pooled:
        rows_all.append((c, effect(pooled), flagged))
      for lvl in sorted({r[0] for r in rows}):
        t1, t2 = (statistics.median(s["cells"][(lvl, c)]["tokens"] or [float("nan")])
                  for s in (first, second))
        print(_effect_line(f"{_fmt_level(lvl)} {c}", effect([r for r in rows if r[0] == lvl]))
              + f" tokens {t1:.0f} -> {t2:.0f} (x{t2 / t1:.1f})"
              + (" FLAGGED" if lvl in flagged else ""), file=out)
  for c, e, flagged in rows_all:
    print(_effect_line(f"pooled {c}", e) + _left_out(flagged), file=out)


def report_prompts(records, group, tag, title, control=CONTROL, out=sys.stdout):
  """Prompt design per level: mean edges per graph, and median prompt
  characters per condition with the characters it adds over the control."""
  chars = collections.defaultdict(dict)   # (level, condition) -> instance -> characters
  edges = collections.defaultdict(dict)
  for r in records:
    level = level_of(r["instance_id"], group, r["task"])
    chars[(level, r["condition"])][r["instance_id"]] = len(r["prompt"])
    edges[level][r["instance_id"]] = r["edges"]
  print(f"[{tag}] {title}", file=out)
  for lvl in sorted(edges):
    conds = sorted({c for l, c in chars if l == lvl}, key=lambda c: (c != control, c))
    base = chars.get((lvl, control))
    parts = []
    for c in conds:
      m = statistics.median(chars[(lvl, c)].values())
      added = ""
      if base and c != control:
        diffs = [n - base[i] for i, n in chars[(lvl, c)].items() if i in base]
        added = f" ({statistics.median(diffs):+,.0f})"
      parts.append(f"{c} {m:,.0f} chars{added}")
    print(f"    {_fmt_level(lvl)}: edges mean {statistics.mean(edges[lvl].values()):.1f}; "
          + "; ".join(parts), file=out)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--responses", nargs="+", required=True)
  ap.add_argument("--group", choices=("density", "cell", "task"), default="density")
  ap.add_argument("--control", default=CONTROL)
  ap.add_argument("--tag", default="sweep")
  ap.add_argument("--title", default="")
  ap.add_argument("--levels", default=None,
                  help="comma-separated densities to pool over (group density)")
  ap.add_argument("--versus", nargs="+", default=None)
  ap.add_argument("--trend", action="store_true")
  ap.add_argument("--continuum", action="store_true")
  ap.add_argument("--blocks", action="store_true")
  ap.add_argument("--headroom", action="store_true")
  ap.add_argument("--gold-means", action="store_true")
  args = ap.parse_args()
  if args.group != "density" and (args.trend or args.continuum or args.levels):
    ap.error("--trend, --continuum and --levels need --group density")
  if args.group != "cell" and args.blocks:
    ap.error("--blocks needs --group cell")
  first = summarize(load(args.responses), args.group)
  levels = None if args.levels is None else [float(x) for x in args.levels.split(",")]
  if args.versus:
    report_versus(first, summarize(load(args.versus), args.group), args.tag,
                  args.title or f"{args.versus} minus {args.responses}", levels)
    return
  pools = None if levels is None else [("the given levels", levels)]
  report(first, args.tag, args.title or " ".join(args.responses), args.control, pools,
         args.trend, args.continuum, args.blocks, args.headroom, args.gold_means)


if __name__ == "__main__":
  main()
