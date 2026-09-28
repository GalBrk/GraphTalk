"""How much of a primer's effect is any change to the prompt, whether the models agree on which
questions a primer fixes, and checks on the responses themselves, for the 40-node
sweep (data/runs/qwen3-{1.7b,4b}[-think].densfull40*, p <= .50).

Outcomes follow rule R1 (graphtalk/outcomes.py) through frame.csv: a truncated
response is never correct. A question is one (task, graph); a condition flips a
question when its outcome (correct or not) differs from the outcome under none.
primer_findings' [main] prints each primer's fixed and broken counts; the checks
here compare those flips across conditions and arms.

  [churn]       per (arm, task): % of questions each condition flips (its net
                change in % correct); each primer's flips against filler's on the
                same questions, exact McNemar, BH over the five primers within the
                pairing. Pooled per arm: the flips that cancel (2 x the smaller of
                fixed and broken, per task), which filler's own effect cannot inflate
  [overlap]     per arm: % of the questions filler flips that a primer flips too,
                against % of the rest, and that difference within (task, density);
                permutation within (task, density), BH over the five primers
  [rerunflip]   qwen3-1.7b node_degree: the questions whose outcome under none
                changes on a second generation of the same prompt ([rerun])
                against the rest: % each condition flips, Fisher exact and a
                permutation within density, BH over the six conditions
  [fragile]     per (arm, task): % of questions right under all seven conditions,
                wrong under all, or mixed; mixed by density, and when none is
                truncated vs finished
  [crossarm]    per pair of arms: Cohen's kappa of correct under none per task,
                chance agreement taken within each density; among questions both
                arms get wrong under none, those a condition fixes in both against
                the count expected within (task, density); permutation within
                (task, density), BH over the six conditions within the pair
  [loop]        a truncated response is a loop when its last TAIL characters repeat
                with a fixed period: at least LOOP of them equal the character one
                period earlier, for some period up to MAX_PERIOD. % of truncations
                that are loops; per condition, % truncated in a loop / while
                working; each condition against none on both, exact McNemar, BH
                over the six conditions within (arm, task)
  [cyclecons]   same arm, condition and graph: cycle_check answers of no whose
                edge_count answer is 40 or more, which implies a cycle on 40 nodes
  [faithful]    thinking arms, finished responses: the answer the trace concludes
                (extract_answer on the text before </think>) against the scored
                answer; connected_nodes is left out, since in a trace the extractor
                reads partial lists. The extractor was built for final answers, so
                some disagreements are misreadings of the trace ("0 to 39" read
                as 39): the count is an upper bound, and [faithsample] prints
                every case to read
  [boxed]       a check on the scorer: finished node_count / node_degree /
                edge_count answers whose answer part holds a \\boxed{} integer,
                the last one against the scored answer
  [lenmed]      finished pairs against none: median change in new tokens over all
                pairs, the questions a condition fixes and those it breaks; fixed
                against broken, Mann-Whitney, BH over the arm's tested cells

Checks of alternative explanations for the findings above and in
docs/investigate_connections_and_cycles.md:
  [churnwhy]    of each condition's flips against none, % where either response
                hit the token budget (a flip at the budget, not in the answer)
  [fillerband]  the side-information cells of primer_findings' [bands] middle band
                (components, clustering, rwse; under the truncation flag): each
                primer's mean effect, and filler's on the same (arm, task,
                density) -- is the side gain any preamble's?
  [samequestion] questions with both a right and a wrong finished response across
                the seven conditions: on the same question, is the right response
                longer? ([lenmed]'s fixed/broken split would follow from it)
  [eedegree]    edge_existence non-edges, plain arms: shared neighbours against
                false yes with the pair's degree sum held fixed (quartile within
                density), and the degree sum with shared neighbours held fixed;
                permutation within strata
  [loopsample]  random truncated tails on each side of the loop threshold
  [faithsample] every [faithful] disagreement

  PYTHONPATH=. python scripts/primer_robustness.py --csv-dir outputs/n40-sweep \\
      > outputs/n40-sweep/primer_robustness.txt
"""
import argparse
import glob
import itertools
import json
import os
import random
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

from graphtalk import outcomes, scoring

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_primer_window as apw  # noqa: E402  (cells: the cells of primer_findings' [bands])
import check_cycle_claims as ccc  # noqa: E402  (graphs, perm_within, pc, answer/trace split)
import primer_findings as pf  # noqa: E402  (FRAME, ARMS, PRIMERS, SEED, with_outcomes, bh, f1)

SHORT = ccc.SHORT
CONDS = ["none", "filler"] + pf.PRIMERS
TASKS = ccc.TASKS6
DENS = pf.DENS4
THINK = ccc.THINK
pc = ccc.pc
TAIL, MAX_PERIOD, LOOP = 2000, 800, 0.98
FAITH_TASKS = ["node_count", "cycle_check", "node_degree", "edge_count", "edge_existence"]
INT_TASKS = ["node_count", "node_degree", "edge_count"]
_BOXED = re.compile(r"boxed\{(?:\s*\\text\{)?\s*(-?\d+)\s*\}")


def period_match(text):
  """(share, period): the best share of the last TAIL characters equal to the one a period earlier."""
  a = np.array([ord(ch) for ch in text[-TAIL:]])
  best = (0.0, 0)
  for p in range(1, min(MAX_PERIOD, len(a) // 2) + 1):
    s = float((a[p:] == a[:-p]).mean())
    if s > best[0]:
      best = (s, p)
      if s == 1.0:
        break
  return best


def concluded(text, task):
  """The answer a thinking trace concludes, by the scorer's own reading; None without a trace."""
  trace = ccc.trace_part(text)
  return scoring.extract_answer(trace, task) if trace.strip() else None


def same(a, b, task):
  return scoring.score_one(a, b, task)["exact"] == 1


def kappa(a, b, s):
  """Cohen's kappa of two 0/1 vectors, chance agreement taken within each stratum of s."""
  a, b, pe = np.asarray(a, float), np.asarray(b, float), 0.0
  for v in np.unique(s):
    m = s == v
    pa, pb = a[m].mean(), b[m].mean()
    pe += m.mean() * (pa * pb + (1 - pa) * (1 - pb))
  return (np.mean(a == b) - pe) / (1 - pe) if pe < 1 else float("nan")


def _selfcheck():
  assert period_match("x" * 50 + "abc, " * 600) == (1.0, 5)
  work = "".join(f"- {100 + 7 * i} + {i % 9} = {100 + 7 * i + i % 9}\n" for i in range(200))
  assert period_match(work)[0] < LOOP
  assert period_match("".join(f"Node {i}: {i * 7 % 13}\n\n" for i in range(300)))[0] < LOOP
  assert concluded("<think>So the answer is 5.</think>The answer is 5.", "node_degree") == "5"
  assert concluded("The answer is 5.", "node_degree") is None
  assert _BOXED.findall(r"$\boxed{400.5}$ ... $\boxed{400}$ and \boxed{\text{12}}") == ["400", "12"]
  a, s = np.array([1, 0, 1, 0, 1, 1, 0, 0]), np.zeros(8)
  assert kappa(a, a, s) == 1 and abs(kappa(a, 1 - a, s) + 1) < 1e-9


def load_frame():
  f = pf.with_outcomes(pd.read_csv(pf.FRAME, dtype={"pred": str, "gold": str}))
  return f[f.density_class <= .5]


def wide(f, col):
  """(arm, task, density, graph) x condition table of one column."""
  return f.set_index(["arm", "task", "density_class", "graph_id", "condition"])[col].unstack()


def strata(x):
  """Each row's (task, density) group, for a table indexed as wide() leaves it."""
  return np.array([f"{t}/{d}" for t, d in zip(x.index.get_level_values("task"),
                                              x.index.get_level_values("density_class"))])


def one(x, arm, task):
  return x.xs((arm, task), level=["arm", "task"])


# ------------------------------------------------------------------ frame checks
def churn(w):
  print("[churn] main sweep, 400 questions per (arm, task): % of questions each condition flips against none "
        "(net change in % correct); * the primer's flips differ from filler's on the same questions, exact "
        "McNemar, BH q < .05 over the five primers within the pairing")
  print("  pooled over the six tasks: % flipped (net change; % that cancel, 2 x the smaller of fixed and broken "
        "per task, averaged over the tasks)")
  for arm in pf.ARMS:
    x = w.xs(arm, level="arm")
    # flipped minus |net| = fixed + broken - |fixed - broken| = 2 x min(fixed, broken)
    cancel = lambda c: np.mean([100 * ((y[c] != y["none"]).mean() - abs((y[c] - y["none"]).mean()))
                                for task in TASKS for y in [one(w, arm, task)]])
    print(f"    {SHORT[arm]:6s} " + " | ".join(
        f"{c} {100 * (x[c] != x['none']).mean():.1f} ({pf.f1(100 * (x[c].mean() - x['none'].mean()))}; "
        f"{cancel(c):.1f})" for c in CONDS[1:]))
  for arm in pf.ARMS:
    for task in TASKS:
      x = one(w, arm, task)
      flip = {c: (x[c] != x["none"]).to_numpy() for c in CONDS[1:]}
      q = dict(zip(pf.PRIMERS, pf.bh([scoring.mcnemar(flip["filler"], flip[c])["p_value"] for c in pf.PRIMERS])))
      print(f"  {SHORT[arm]:6s} {task:15s} " + " | ".join(
          f"{c} {100 * flip[c].mean():.1f}{'*' if q.get(c, 1) < .05 else ''} "
          f"({pf.f1(100 * (x[c].mean() - x['none'].mean()))})" for c in CONDS[1:]))


def overlap(w):
  print(f"[overlap] per arm, pooled over the six tasks: % of the questions filler flips that the primer flips too "
        f"vs % of the others [the difference within (task, density), weighted by size, over the strata that have "
        f"both]; permutation within (task, density), {ccc.B_PERM} draws, BH over the five primers (* q < .05)")
  for arm in pf.ARMS:
    x = w.xs(arm, level="arm")
    g, y, rows = strata(x), (x["filler"] != x["none"]).to_numpy(), []
    both = [s for s in np.unique(g) if y[g == s].any() and not y[g == s].all()]
    for c in pf.PRIMERS:
      z = (x[c] != x["none"]).to_numpy().astype(float)
      _, p = ccc.perm_within(z, y, g, np.random.default_rng(pf.SEED))
      within = sum((g == s).sum() * (z[(g == s) & y].mean() - z[(g == s) & ~y].mean()) for s in both)
      rows.append((c, 100 * z[y].mean(), 100 * z[~y].mean(),
                   100 * within / sum((g == s).sum() for s in both), p))
    print(f"  {SHORT[arm]:6s} filler flips {y.sum()} of {len(y)}: " + " | ".join(
        f"{c} {a:.1f} vs {b:.1f} [{pf.f1(d)}]{'*' if q < .05 else ''} (p={p:.2g})"
        for (c, a, b, d, p), q in zip(rows, pf.bh([1.0 if r[4] != r[4] else r[4] for r in rows]))))


def rerun_flip(w):
  """Questions whose outcome under none changes when the same prompt is generated again (primer_findings'
  [rerun]: qwen3-1.7b node_degree, the only arm and task generated twice) against the rest: does each
  condition flip them more often? The part of a condition's flips that chance alone produces."""
  main = pf.load_runs("data/runs/qwen3-1.7b.densfull40.shard*.jsonl", tasks={"node_degree"}, conds={"none"})
  again = pf.load_runs("data/runs/qwen3-1.7b.degdens40.shard*.jsonl", conds={"none"})
  unstable = {k[0].split("/", 2)[2] for k, r in again.items()
              if k in main and pf.record_correct(r) != pf.record_correct(main[k])}
  x = one(w, "qwen3-1.7b", "node_degree")
  y = x.index.get_level_values("graph_id").isin(unstable)
  g = np.asarray(x.index.get_level_values("density_class"))
  print(f"[rerunflip] qwen3-1.7b node_degree, main sweep (p<=.50): questions whose outcome under none changes "
        f"when the same prompt is generated again ([rerun]) against the rest: % each condition flips against "
        f"none, flips on those questions / all flips, Fisher exact p, permutation within density "
        f"({ccc.B_PERM} draws), BH over the six conditions on the permutation p")
  print(f"  outcome changes on a second generation: {y.sum()} of {len(y)} questions")
  rows = []
  for c in CONDS[1:]:
    z = (x[c] != x["none"]).to_numpy()
    _, p_fisher = stats.fisher_exact([[(z & y).sum(), (~z & y).sum()], [(z & ~y).sum(), (~z & ~y).sum()]])
    _, p_perm = ccc.perm_within(z.astype(float), y, g, np.random.default_rng(pf.SEED))
    rows.append((c, 100 * z[y].mean(), 100 * z[~y].mean(), (z & y).sum(), z.sum(), p_fisher, p_perm))
  for (c, a, b, k, n, pfi, ppe), q in zip(rows, pf.bh([r[6] for r in rows])):
    print(f"  {c:10s} {a:.1f} vs {b:.1f} ({k} of {n} flips) Fisher p={pfi:.2g} perm p={ppe:.2g} q={q:.2g}")


def fragile(w, t):
  print("[fragile] main sweep, 400 questions per (arm, task): % right under all seven conditions / wrong under all / "
        "mixed | % mixed by density | % mixed where none is truncated (n) vs finished (n)")
  k = w[CONDS].sum(axis=1)
  for arm in pf.ARMS:
    for task in TASKS:
      kk = one(k, arm, task)
      tn = one(t, arm, task)["none"].reindex(kk.index).astype(bool)
      mixed, dens = (kk > 0) & (kk < 7), kk.index.get_level_values("density_class")
      print(f"  {SHORT[arm]:6s} {task:15s} {pc(kk == 7)} / {pc(kk == 0)} / {pc(mixed)} | "
            + " ".join(f"{d:g} {pc(mixed[dens == d]).strip()}" for d in DENS)
            + f" | {pc(mixed[tn]).strip()} ({tn.sum()}) vs {pc(mixed[~tn]).strip()} ({(~tn).sum()})")


def crossarm(w):
  pairs = list(itertools.combinations(pf.ARMS, 2))
  label = lambda a, b: f"{SHORT[a]} vs {SHORT[b]}"
  print("[crossarm] Cohen's kappa of correct under none, per task, chance agreement taken within each density "
        "(nan: no variation)")
  print(f"  {'':16s}" + "".join(f"{t:>16s}" for t in TASKS))
  for a, b in pairs:
    cells = []
    for task in TASKS:
      xa = one(w, a, task)["none"]
      xb = one(w, b, task)["none"].reindex(xa.index)
      cells.append(kappa(xa, xb, xa.index.get_level_values("density_class").to_numpy()))
    print(f"  {label(a, b):16s}" + "".join(f"{k:16.2f}" for k in cells))
  print("[crossarm] questions both arms get wrong under none, pooled over the six tasks: per condition, % fixed in "
        "the first arm / in the second; fixed in both, observed (expected within (task, density)); permutation "
        "within (task, density), BH over the six conditions (* q < .05)")
  for a, b in pairs:
    xa = w.xs(a, level="arm")
    xb = w.xs(b, level="arm").reindex(xa.index)
    m = ((xa["none"] == 0) & (xb["none"] == 0)).to_numpy()
    xa, xb = xa[m], xb[m]
    g, rows = strata(xa), []
    for c in CONDS[1:]:
      fa, fb = xa[c].to_numpy().astype(float), xb[c].to_numpy().astype(bool)
      exp = sum(fa[g == s].mean() * fb[g == s].mean() * (g == s).sum() for s in np.unique(g))
      _, p = ccc.perm_within(fa, fb, g, np.random.default_rng(pf.SEED))
      rows.append((c, 100 * fa.mean(), 100 * fb.mean(), int((fa.astype(bool) & fb).sum()), exp, p))
    qs = pf.bh([1.0 if r[5] != r[5] else r[5] for r in rows])
    print(f"  {label(a, b):16s} n={m.sum()}: " + " | ".join(
        f"{c} {x:.1f}/{y:.1f} both {o} ({e:.1f}){'*' if q < .05 else ''}"
        for (c, x, y, o, e, _), q in zip(rows, qs)))


def cyclecons(f):
  d = f[f.truncated == 0]
  key = ["arm", "condition", "graph_id"]
  j = d[d.task == "cycle_check"][key + ["pred"]].merge(d[d.task == "edge_count"][key + ["pred"]], on=key,
                                                        how="left", suffixes=("_cc", "_ec"))
  j = j[j.pred_cc.notna()]
  j["no"] = j.pred_cc.str.lower().str.startswith("no")
  j["m"] = pd.to_numeric(j.pred_ec, errors="coerce")
  print("[cyclecons] same arm, condition and graph, finished answers: cycle_check answers of no; of those, with a "
        "finished edge_count answer; of those, 40 or more edges (the model's own count implies a cycle) | the same "
        "share among answers of yes | per condition, no answers / of them with 40 or more")
  for arm in pf.ARMS:
    x = j[j.arm == arm]
    no, yes = x[x.no], x[~x.no]
    print(f"  {SHORT[arm]:6s} no {len(no)}; with a count {no.m.notna().sum()}; 40 or more {(no.m >= 40).sum()} "
          f"({pc(no.m.dropna() >= 40).strip()}%) | yes: {pc(yes.m.dropna() >= 40).strip()}% of "
          f"{yes.m.notna().sum()} | " + ", ".join(
              f"{c} {(no.condition == c).sum()}/{((no.condition == c) & (no.m >= 40)).sum()}" for c in CONDS))


def lenmed(f):
  tok, cor, tr = wide(f, "n_new_tokens"), wide(f, "correct"), wide(f, "truncated")
  med = lambda x: f"{x.median():+.0f}" if len(x) else "-"
  print("[lenmed] finished pairs against none: median change in new tokens, all pairs / questions the condition "
        "fixes (n, % shorter) / questions it breaks (n, % shorter); * fixed and broken differ, Mann-Whitney, BH "
        "q < .05 over the arm's cells with 10 or more of each")
  for arm in pf.ARMS:
    rows = []
    for task in TASKS:
      T, C, R = one(tok, arm, task), one(cor, arm, task), one(tr, arm, task)
      for c in CONDS[1:]:
        m = (R[c] == 0) & (R["none"] == 0)
        d = (T[c] - T["none"])[m]
        fx, br = d[(C["none"][m] == 0) & (C[c][m] == 1)], d[(C["none"][m] == 1) & (C[c][m] == 0)]
        p = stats.mannwhitneyu(fx, br).pvalue if len(fx) >= 10 and len(br) >= 10 else float("nan")
        rows.append((task, c, d, fx, br, p))
    tested = [i for i, r in enumerate(rows) if r[5] == r[5]]
    q = dict(zip(tested, pf.bh([rows[i][5] for i in tested]))) if tested else {}
    for task in TASKS:
      print(f"  {SHORT[arm]:6s} {task:15s} " + " | ".join(
          f"{c} {med(d)} / {med(fx)} ({len(fx)}, {pc(fx < 0).strip()}) / {med(br)} ({len(br)}, "
          f"{pc(br < 0).strip()}){'*' if q.get(i, 1) < .05 else ''}"
          for i, (t, c, d, fx, br, _) in enumerate(rows) if t == task))


# ------------------------------------------------------------------ alternative explanations
def churn_why(f):
  C, T = wide(f, "correct"), wide(f, "truncated")
  print("[churnwhy] flips against none, main sweep, all six tasks: n, and % where either response hit the token "
        "budget")
  for arm in pf.ARMS:
    c, t = C.xs(arm, level="arm"), T.xs(arm, level="arm")
    print(f"  {SHORT[arm]:6s} " + " | ".join(
        f"{k} {m.sum()}: {pc(((t[k] + t['none']) > 0)[m]).strip()}"
        for k in CONDS[1:] for m in [c[k] != c["none"]]))


def filler_band(full):
  """full: the whole frame with R1 columns (the bands use every density, as [bands] does)."""
  t = apw.cells(full, json.load(open(pf.BARS)))
  side = t[~t.flagged & (t.bar < apw.CARRIES) & (t.baseline >= pf.BAND[0]) & (t.baseline < pf.BAND[1])
           & t.condition.isin(["components", "clustering", "rwse"])]
  fill = []
  for arm, task, dens in sorted(set(zip(side.arm, side.task, side.density))):
    j = pf.pairs(full, arm, task, "none", "filler", [dens])
    if max(j.truncated_a.mean(), j.truncated_b.mean()) < outcomes.FLAG:
      fill.append(100 * (j.correct_b.mean() - j.correct_a.mean()))
  print(f"[fillerband] side-information cells of [bands], baseline {pf.BAND[0]}-{pf.BAND[1]}, under the truncation "
        "flag: mean effect against none (cells)")
  print("  " + ", ".join(f"{c} {pf.f1(x.delta.mean())} ({len(x)})" for c, x in side.groupby("condition"))
        + f"; the three {pf.f1(side.delta.mean())} ({len(side)}) | filler on the same (arm, task, density) "
        f"{pf.f1(np.mean(fill))} ({len(fill)})")


def same_question(f):
  d = f[f.truncated == 0]
  print("[samequestion] questions with both a right and a wrong finished response across the seven conditions: "
        "median over questions of (mean new tokens of the right ones - of the wrong ones), % of questions where the "
        "right ones are longer, n questions")
  for arm in pf.ARMS:
    cells = []
    for task in TASKS:
      g = (d[(d.arm == arm) & (d.task == task)].groupby(["graph_id", "correct"]).n_new_tokens.mean()
           .unstack().reindex(columns=[0, 1]).dropna())
      diff = g[1] - g[0]
      cells.append(f"{task} {diff.median():+.0f} ({pc(diff > 0).strip()}%, {len(g)})" if len(g) >= 10
                   else f"{task} - ({len(g)})")
    print(f"  {SHORT[arm]:6s} " + " | ".join(cells))


def ee_degree(f):
  g, rows = ccc.graphs(ccc.PROMPTS), []
  d = f[(f.task == "edge_existence") & (f.truncated == 0) & f.arm.isin(["qwen3-1.7b", "qwen3-4b"])]
  for r in d.itertuples():
    _, nb, _, (u, v), _, _ = g[r.instance_id]
    if v not in nb.get(u, ()):
      rows.append(dict(arm=r.arm, condition=r.condition, density=r.density_class,
                       yes=isinstance(r.pred, str) and r.pred.lower().startswith("yes"),
                       shared=len(nb.get(u, set()) & nb.get(v, set())), degsum=len(nb.get(u, ())) + len(nb.get(v, ()))))
  d = pd.DataFrame(rows)
  x0 = d[(d.arm == "qwen3-1.7b") & (d.condition == "none")]
  print("[eedegree] edge_existence non-edges, finished answers of the plain arms: correlation of shared neighbours "
        "and the pair's degree sum within density: " + ", ".join(
            f"p={p:g} {x[['shared', 'degsum']].corr().iloc[0, 1]:.2f}" for p, x in x0.groupby("density")))
  print("  false yes minus correct no: mean shared neighbours, p within density / within density x degree-sum "
        "quartile; mean degree sum, p within density x shared bin (0, 1, 2-3, 4+); permutation within strata "
        "(and condition)")
  for arm in ["qwen3-1.7b", "qwen3-4b"]:
    for conds, label in ((["none"], "none"), (CONDS, "all seven")):
      x = d[(d.arm == arm) & d.condition.isin(conds)].copy()
      x["dq"] = x.groupby(["density", "condition"]).degsum.transform(
          lambda s: pd.qcut(s.rank(method="first"), 4, labels=False))
      x["sb"] = pd.cut(x.shared, [-1, 0, 1, 3, 99], labels=False)
      test = lambda col, by: ccc.perm_within(x[col].to_numpy(float), x.yes.to_numpy(),
                                             x.groupby(by).ngroup().to_numpy(), np.random.default_rng(pf.SEED))
      ds, p1 = test("shared", ["density", "condition"])
      _, p2 = test("shared", ["density", "condition", "dq"])
      dg, p3 = test("degsum", ["density", "condition", "sb"])
      print(f"  {SHORT[arm]:4s} {label:9s} n={len(x)}, false yes {x.yes.sum()}: shared {ds:+.2f}, p={p1:.2g} / "
            f"{p2:.2g} | degree sum {dg:+.2f}, p={p3:.2g}")


# ------------------------------------------------------------------ text checks
def text_pass(ok):
  """Loop scores of the truncated responses, trace-vs-final answers of the thinking arms, and
  boxed-vs-scored answers."""
  loops, faith, boxed = [], [], []
  for arm in pf.ARMS:
    seen = set()   # a response repeated across shards counts once, the first, as in pf.load_runs
    for path in sorted(glob.glob(f"data/runs/{arm}.densfull40.shard*.jsonl")):
      for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        key = (arm, r["instance_id"], r["condition"])
        if key not in ok or key[1:] in seen:
          continue
        seen.add(key[1:])
        truncated, correct, pred, gold = ok[key]
        text, task = r["response"] or "", r["task"]
        base = dict(arm=arm, task=task, condition=r["condition"], instance_id=r["instance_id"])
        if truncated:
          s, p = period_match(text)
          loops.append(dict(base, match=s, period=p, loop=s >= LOOP, tail=text[-200:]))
          continue
        box = _BOXED.findall(ccc.answer_part(text)) if task in INT_TASKS and isinstance(pred, str) else []
        if box:
          boxed.append(dict(base, final=pred, boxed=box[-1], differs=not same(box[-1], pred, task),
                            final_correct=correct, boxed_correct=same(box[-1], gold, task)))
        if arm in THINK and task in FAITH_TASKS and isinstance(pred, str):
          b = concluded(text, task)
          differs = b is not None and not same(b, pred, task)
          faith.append(dict(base, final=pred, trace=b, has_trace=b is not None, differs=differs,
                            final_correct=correct, trace_correct=b is not None and same(b, gold, task),
                            trace_tail=ccc.trace_part(text)[-250:] if differs else "",
                            answer_head=ccc.answer_part(text)[:250] if differs else ""))
  return pd.DataFrame(loops), pd.DataFrame(faith), pd.DataFrame(boxed)


def loop_block(f, L):
  d = f.merge(L[["arm", "instance_id", "condition", "loop"]], how="left", on=["arm", "instance_id", "condition"])
  d["loop_tr"] = (d.truncated == 1) & d["loop"].eq(True)
  d["work_tr"] = (d.truncated == 1) & ~d["loop"].eq(True)
  print(f"[loop] truncated responses, main sweep: n, % whose last {TAIL} characters are a loop (at least "
        f"{LOOP:.0%} equal one period earlier, period up to {MAX_PERIOD}), median period of the loops")
  for arm in pf.ARMS:
    x = L[L.arm == arm]
    print(f"  {SHORT[arm]:6s} {len(x)}: loops {pc(x.loop).strip()}%, median period {x[x.loop].period.median():.0f}")
  edges = [0, .3, .5, .7, .9, LOOP, 1.0001]
  counts = np.histogram(L.match, bins=edges)[0]
  print("  the score (best share equal one period earlier) of all truncated responses: " + " | ".join(
      f"[{a:g}, {min(b, 1):g}{']' if b > 1 else ')'} {k}" for a, b, k in zip(edges, edges[1:], counts)))
  print("  per (arm, task), % of responses truncated in a loop / truncated while working, by condition")
  for arm in pf.ARMS:
    for task in TASKS:
      x = d[(d.arm == arm) & (d.task == task)]
      print(f"  {SHORT[arm]:6s} {task:15s} " + " | ".join(
          f"{c} {pc(y.loop_tr).strip()}/{pc(y.work_tr).strip()}" for c in CONDS for y in [x[x.condition == c]]))
  print("  changes against none, exact McNemar, BH q < .05 over the six conditions within (arm, task):")
  for col, label in (("loop_tr", "in a loop"), ("work_tr", "while working")):
    hits = []
    for arm in pf.ARMS:
      for task in TASKS:
        x = d[(d.arm == arm) & (d.task == task)].set_index(["graph_id", "condition"])[col].unstack()
        ps = [scoring.mcnemar(x["none"].to_numpy(), x[c].to_numpy())["p_value"] for c in CONDS[1:]]
        hits += [f"{SHORT[arm]} {task} {c} {100 * x['none'].mean():.1f}->{100 * x[c].mean():.1f}"
                 for c, q in zip(CONDS[1:], pf.bh(ps)) if q < .05]
    print(f"    truncated {label}: " + ("; ".join(hits) or "none"))


def faithful(F):
  print("[faithful] thinking arms, finished responses with a scored answer: n whose trace concludes an answer (of "
        "n); how many differ from the scored answer; of those, the scored answer right / the trace's right")
  for arm in THINK:
    for task in FAITH_TASKS:
      x = F[(F.arm == arm) & (F.task == task)]
      y = x[x.has_trace]
      z = y[y.differs]
      print(f"  {SHORT[arm]:6s} {task:15s} {len(y)} of {len(x)}; differ {len(z)} ({pc(y.differs).strip()}%); "
            f"right: final {z.final_correct.sum()} / trace {z.trace_correct.sum()}")
    x = F[(F.arm == arm) & F.has_trace]
    print(f"  {SHORT[arm]:6s} differ by condition: "
          + ", ".join(f"{c} {x[x.condition == c].differs.sum()}" for c in CONDS))


def boxed_block(X):
  print("[boxed] finished answers whose answer part holds a \\boxed{} integer: n; how many differ from the scored "
        "answer; of those, boxed right and scored wrong / scored right and boxed wrong / both wrong")
  for arm in pf.ARMS:
    print(f"  {SHORT[arm]:6s} " + " | ".join(
        f"{task} {len(x)}; differ {len(z)}: {(z.boxed_correct & ~z.final_correct).sum()} / "
        f"{(~z.boxed_correct & z.final_correct).sum()} / {(~z.boxed_correct & ~z.final_correct).sum()}"
        for task in INT_TASKS for x in [X[(X.arm == arm) & (X.task == task)]] for z in [x[x.differs]]))


def samples(L, F):
  rng = random.Random(pf.SEED)
  print("[loopsample] truncated tails (last 200 characters), at random: two loops and two others per arm")
  for arm in pf.ARMS:
    for is_loop in (True, False):
      x = list(L[(L.arm == arm) & (L.loop == is_loop)].itertuples())
      for r in rng.sample(x, min(2, len(x))):
        print(f"  -- {SHORT[arm]} {r.condition} {r.instance_id} match {r.match:.3f} period {r.period} "
              f"loop={is_loop}\n     {r.tail!r}")
  print("[faithsample] every [faithful] disagreement: scored answer / the trace's; the trace's last 250 characters "
        "| the answer's first 250")
  for r in F[F.differs].itertuples():
    mark = lambda ok: "right" if ok else "wrong"
    print(f"  -- {SHORT[r.arm]} {r.condition} {r.instance_id}: final {r.final} ({mark(r.final_correct)}) / trace "
          f"{r.trace} ({mark(r.trace_correct)})\n     {r.trace_tail!r}\n     | {r.answer_head!r}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--csv-dir", help="write robustness_responses.csv: one row per response the text checks read")
  args = ap.parse_args()
  _selfcheck()
  f = load_frame()
  w = wide(f, "correct")
  assert not w.isna().any().any()
  churn(w)
  overlap(w)
  rerun_flip(w)
  fragile(w, wide(f, "truncated"))
  crossarm(w)
  ok = {(r.arm, r.instance_id, r.condition): (bool(r.truncated), bool(r.correct), r.pred, r.gold)
        for r in f.itertuples()}
  L, F, X = text_pass(ok)
  assert len(L) == f.truncated.sum(), (len(L), f.truncated.sum())
  loop_block(f, L)
  cyclecons(f)
  faithful(F)
  boxed_block(X)
  lenmed(f)
  churn_why(f)
  filler_band(pf.with_outcomes(pd.read_csv(pf.FRAME)))
  same_question(f)
  ee_degree(f)
  samples(L, F)
  if args.csv_dir:
    out = os.path.join(args.csv_dir, "robustness_responses.csv")
    pd.concat([L.drop(columns="tail").assign(source="densfull40", check="loop"),
               F.drop(columns=["trace_tail", "answer_head"]).assign(source="densfull40", check="faithful"),
               X.assign(source="densfull40", check="boxed")]).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
  main()
