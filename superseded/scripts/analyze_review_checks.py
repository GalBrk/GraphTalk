"""CPU-only checks raised by the 2026-09-18 review of the paper.

Each test answers one reviewer objection from data already in `runs/`; none
needs a GPU.

  interaction  route x baseline interaction tested directly (cluster
               bootstrap over (arm, task, density) blocks), instead of
               contrasting a significant r with a non-significant one. Also
               the cross-fitted r on ONE fold direction, so every cell is one
               point -- the paper's 232/762 count both directions of a cell.
  logit        the same relation on the logit scale and inside a
               baseline-matched window, since accuracy is bounded in [0, 1]
               (a cell at 0.99 can only fall, one at 0.02 can only rise).
  headline     clustering-vs-filler split into clustering-vs-none and
               filler-vs-none per density: how much of the +4.3 is filler
               hurting rather than clustering helping.
  rewiring     mean clustering coefficient per rewiring level.
  tokens       primer length in Qwen3 tokens, not characters.
  prior        density-prior hypothesis: in ER graphs every node's clustering
               is ~p, so the clustering primer states the density. Do
               node_degree answers under `clustering` drift toward
               mean(C) * (n - 1)?
  sdt          d' (sensitivity) and c (criterion) for edge_existence.
  budget       thinking arms rescored as if capped at 2,048 new tokens (the
               plain arms' budget). Exact under greedy decoding: a trace that
               finished within 2,048 tokens is unchanged by the lower cap.
  mae          mean absolute error for the integer tasks (from superseded/ci_all.json).

  PYTHONPATH=. python superseded/scripts/analyze_review_checks.py \\
      --tokenizer path/to/Qwen3/tokenizer.json
"""

import argparse
import collections
import glob
import json
import math
import os
import random
import re
import statistics
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append("scripts")  # live scripts it still imports; run from the repo root
import analyze_baseline_law as abl  # noqa: E402
import analyze_headline_robustness as ahr  # noqa: E402

from graphtalk import scoring  # noqa: E402

CORPORA = ("densfull40", "densfull40hi")
MATCHED_WINDOW = (0.10, 0.90)
PLAIN_BUDGET = 2048
_ADJ = re.compile(r"Node (\d+) is connected to nodes? ([\d, ]+)\.")
_Z = statistics.NormalDist().inv_cdf


# --------------------------------------------------------------------------
# pure helpers (tested in superseded/tests/test_analyze_review_checks.py)
# --------------------------------------------------------------------------

def paired_cells(scores, min_pairs=10):
  """{(task, dens, iid, cond): score|None} -> cells holding their raw pairs,
  hit_cap (None) dropped from both sides, as in `abl.cells_from_scores`."""
  grouped = collections.defaultdict(dict)
  for (task, dens, iid, cond), value in scores.items():
    grouped[(task, dens, cond)][iid] = value
  out = []
  for (task, dens, cond), values in sorted(grouped.items(), key=repr):
    if cond == "none":
      continue
    base = grouped.get((task, dens, "none"), {})
    pairs = [(i, base[i], v) for i, v in values.items()
             if base.get(i) is not None and v is not None]
    if len(pairs) >= min_pairs:
      out.append(dict(task=task, density=dens, condition=cond, pairs=pairs))
  return out


def _logit(k, n):
  p = (k + 0.5) / (n + 1)
  return math.log(p / (1 - p))


def cell_point(pairs, fold=None, scale="pp"):
  """(baseline, effect) for one cell. fold=None: both from every pair (the
  naive estimate). fold=0/1: baseline from that fold, effect from the other,
  the same split `abl.cells_from_scores_crossfit` uses."""
  if fold is None:
    base_pairs = eff_pairs = pairs
  else:
    base_pairs = [p for p in pairs if abl._fold(p[0]) == fold]
    eff_pairs = [p for p in pairs if abl._fold(p[0]) != fold]
    if len(base_pairs) < 5 or len(eff_pairs) < 5:
      return None
  baseline = sum(b for _, b, _ in base_pairs) / len(base_pairs)
  n = len(eff_pairs)
  sb = sum(b for _, b, _ in eff_pairs)
  sv = sum(v for _, _, v in eff_pairs)
  if scale == "pp":
    return baseline, 100.0 * (sv - sb) / n
  return baseline, _logit(sv, n) - _logit(sb, n)


def interaction_coef(points):
  """points: [(baseline, effect, route)] -> slope(route) - slope(no route),
  the baseline:route coefficient of effect ~ baseline * route."""
  rows = [([b, float(r), b * float(r)], e) for b, e, r in points]
  return abl.ols(rows, ["baseline", "route", "baseline:route"])[3][1]


def cluster_bootstrap(points, blocks, stat, n_boot=2000, seed=0):
  """Percentile CI and a two-sided bootstrap p for `stat(points)`,
  resampling whole blocks so correlated cells stay together."""
  by_block = collections.defaultdict(list)
  for point, block in zip(points, blocks):
    by_block[block].append(point)
  keys = list(by_block)
  rng = random.Random(seed)
  draws = []
  for _ in range(n_boot):
    sample = [p for _ in keys for p in by_block[keys[rng.randrange(len(keys))]]]
    try:
      value = stat(sample)
    except (ValueError, ZeroDivisionError):
      continue
    if not math.isnan(value):
      draws.append(value)
  draws.sort()
  lo = draws[int(0.025 * (len(draws) - 1))]
  hi = draws[int(0.975 * (len(draws) - 1))]
  tail = min(sum(d <= 0 for d in draws), sum(d >= 0 for d in draws))
  return dict(estimate=stat(points), ci=[lo, hi],
              p_boot=min(1.0, 2 * tail / len(draws)), n_boot=len(draws),
              n_blocks=len(keys))


def fisher_z_diff(r1, n1, r2, n2):
  """z and two-sided p for r1 != r2 (independent samples)."""
  z = (math.atanh(r1) - math.atanh(r2)) / math.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
  return z, 2 * (1 - statistics.NormalDist().cdf(abs(z)))


def sdt(hits, n_yes, false_alarms, n_no):
  """(d', c) with the log-linear correction (Hautus 1995), so a 100% hit
  rate -- which this corpus has -- stays finite."""
  zh = _Z((hits + 0.5) / (n_yes + 1))
  zf = _Z((false_alarms + 0.5) / (n_no + 1))
  return zh - zf, -(zh + zf) / 2


def _r(points):
  r, p = abl.pearson([b for b, _, _ in points], [e for _, e, _ in points])
  return dict(r=r, p=p, k=len(points))


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

_MAIN = {}


def main_cells(runs, bars):
  """Every non-degenerate densfull40(+hi) cell, with arm, route and block."""
  if "cells" not in _MAIN:
    cells = []
    for arm in abl.DENSFULL_ARMS:
      for corpus in CORPORA:
        for cell in paired_cells(abl.score_run([f"{runs}/{arm}.{corpus}.shard*.jsonl"])):
          if cell["task"] in abl.DEGENERATE_TASKS:
            continue
          cell.update(arm=arm, corpus=corpus,
                      route=abl.offers_route(bars, cell["task"], cell["condition"]),
                      block=(arm, cell["task"], cell["density"]))
          cells.append(cell)
    _MAIN["cells"] = cells
  return _MAIN["cells"]


def _points(cells, fold=None, scale="pp", window=None):
  pts, blocks = [], []
  folds = (0, 1) if fold == "both" else (fold,)
  for cell in cells:
    for f in folds:
      point = cell_point(cell["pairs"], f, scale)
      if point is None:
        continue
      if window and not window[0] <= point[0] <= window[1]:
        continue
      pts.append((point[0], point[1], cell["route"]))
      blocks.append(cell["block"])
  return pts, blocks


def _relation(pts, blocks):
  route = [p for p in pts if p[2]]
  plain = [p for p in pts if not p[2]]
  out = dict(route=_r(route), no_route=_r(plain),
             interaction=cluster_bootstrap(pts, blocks, interaction_coef))
  z, p = fisher_z_diff(out["route"]["r"], len(route), out["no_route"]["r"], len(plain))
  out["fisher_z"] = dict(z=z, p=p, note="assumes independent cells; the bootstrap does not")
  out["mean_baseline"] = {
      "route": sum(p[0] for p in route) / len(route),
      "no_route": sum(p[0] for p in plain) / len(plain)}
  return out


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_interaction(args):
  cells = main_cells(args.runs, args.bars_n40)
  out = {"unique_cells": dict(route=sum(c["route"] for c in cells),
                              no_route=sum(not c["route"] for c in cells))}
  for label, fold in (("naive", None), ("crossfit_fold0", 0),
                      ("crossfit_fold1", 1), ("crossfit_both", "both")):
    out[label] = _relation(*_points(cells, fold))

  held = []
  for arm, patterns in abl.HELDOUT_GLOBS.items():
    scores = abl.score_run([f"{args.runs}/{os.path.basename(p)}" for p in patterns],
                           by_density=False)
    for cell in paired_cells(scores):
      if cell["task"] in abl.DEGENERATE_TASKS:
        continue
      b, e = cell_point(cell["pairs"])
      held.append(((b, e, abl.offers_route(args.bars_pub, cell["task"],
                                           cell["condition"])), (arm, cell["task"])))
  out["heldout_naive"] = _relation([p for p, _ in held], [b for _, b in held])
  return out


def test_logit(args):
  cells = main_cells(args.runs, args.bars_n40)
  return {
      "logit_crossfit_fold0": _relation(*_points(cells, 0, "logit")),
      "logit_crossfit_both": _relation(*_points(cells, "both", "logit")),
      "matched_window": list(MATCHED_WINDOW),
      "pp_crossfit_both_matched": _relation(*_points(cells, "both", "pp", MATCHED_WINDOW)),
      "logit_crossfit_both_matched": _relation(*_points(cells, "both", "logit",
                                                        MATCHED_WINDOW)),
  }


def test_headline(args):
  scores = {**ahr._load(ahr.MAIN_GLOBS), **ahr._load(ahr.FILLER_GLOB)}
  by_density = collections.defaultdict(set)
  for iid, _ in scores:
    by_density[abl.density_of(iid)].add(iid)
  levels = [("pooled", {i for s in by_density.values() for i in s})]
  levels += [(f"{d:.2f}", ids) for d, ids in sorted(by_density.items())]
  out = {}
  for label, ids in levels:
    row = {}
    for a, b in (("clustering", "none"), ("filler", "none"), ("clustering", "filler")):
      delta, p, n = ahr.paired_delta(scores, ids, a, b)
      row[f"{a}-{b}"] = dict(delta=delta, p=p, n=n)
    out[label] = row
  return out


def _graph(prompt, n):
  g = nx.Graph()
  g.add_nodes_from(range(n))
  for node, nbrs in _ADJ.findall(prompt):
    g.add_edges_from((int(node), int(x)) for x in nbrs.replace(" ", "").split(",") if x)
  return g


def _mean_clustering(prompt_rows):
  """instance_id -> (mean clustering over all n nodes, mean degree)."""
  out = {}
  for d in prompt_rows:
    if d["instance_id"] in out:
      continue
    n = d["nodes"] if isinstance(d["nodes"], int) else len(d["nodes"])
    g = _graph(d["prompt"], n)
    out[d["instance_id"]] = (nx.average_clustering(g), 2 * g.number_of_edges() / n, n)
  return out


def _prompts(path, task="node_degree"):
  with open(path, encoding="utf-8") as fh:
    return [d for d in map(json.loads, fh) if d["task"] == task]


def test_rewiring(args):
  rows = _prompts("prompts.rewire_shared.jsonl")
  stats = _mean_clustering(rows)
  level_of = {d["instance_id"]: (d["rung"], d["level"]) for d in rows}
  out = collections.defaultdict(list)
  for iid, (c, k, _) in stats.items():
    out[level_of[iid]].append((c, k))
  return {f"{r}/{lv}": dict(mean_clustering=sum(c for c, _ in v) / len(v),
                            mean_degree=sum(k for _, k in v) / len(v), n=len(v))
          for (r, lv), v in sorted(out.items())}


def test_tokens(args):
  if not args.tokenizer:
    return {"skipped": "pass --tokenizer path/to/tokenizer.json"}
  from tokenizers import Tokenizer
  tok = Tokenizer.from_file(args.tokenizer)
  by_iid = collections.defaultdict(dict)
  for d in _prompts("prompts.densfull40.jsonl"):
    by_iid[d["instance_id"]][d["condition"]] = d["prompt"]
  conds = sorted({c for v in by_iid.values() for c in v})
  flat = [(iid, c, p) for iid, v in by_iid.items() for c, p in v.items()]
  lens = {(iid, c): len(e.ids) for (iid, c, _), e in
          zip(flat, tok.encode_batch([p for _, _, p in flat]))}
  chars = {(iid, c): len(p) for iid, c, p in flat}
  out = {}
  for c in conds:
    ids = [i for i in by_iid if c in by_iid[i] and "none" in by_iid[i]]
    out[c] = dict(
        primer_tokens=sum(lens[(i, c)] - lens[(i, "none")] for i in ids) / len(ids),
        primer_chars=sum(chars[(i, c)] - chars[(i, "none")] for i in ids) / len(ids),
        prompt_tokens=sum(lens[(i, c)] for i in ids) / len(ids), n=len(ids))
  return out


def _integer_answers(pattern, conds):
  """(iid, cond) -> (answer, gold) for non-capped node_degree rows."""
  out = {}
  for path in sorted(glob.glob(pattern)):
    with open(path, encoding="utf-8") as fh:
      for line in fh:
        if not line.strip():
          continue
        r = json.loads(line)
        if r["task"] != "node_degree" or r["condition"] not in conds or r.get("hit_cap"):
          continue
        try:
          out.setdefault((r["instance_id"], r["condition"]),
                         (int(str(scoring.extract_answer(r["response"], "node_degree")).strip()),
                          int(r["gold"])))
        except (TypeError, ValueError):
          continue
  return out


def _paired_boot(values, n_boot=2000, seed=0):
  rng = random.Random(seed)
  means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                 for _ in range(n_boot))
  return dict(mean=sum(values) / len(values), ci=[means[int(0.025 * (n_boot - 1))],
                                                  means[int(0.975 * (n_boot - 1))]],
              n=len(values))


def test_prior(args):
  """Two readings of the density-prior hypothesis.

  densfull40(+hi): among WRONG answers on instances wrong under both
  conditions, does the error point toward mean(C)*(n-1) more often under
  `clustering` than under `none`?

  rewiring: gold and mean degree are identical across low/base/high while
  mean(C) moves, so if the clustering primer acts as a density prior, the
  paired signed-error shift (clustering - none) should rise with the level.
  """
  conds = ("none", "clustering", "filler")
  out = {"densfull40": {}, "rewiring": {}}
  stats = _mean_clustering(_prompts("prompts.densfull40.jsonl")
                           + _prompts("prompts.densfull40hi.jsonl"))
  for arm in abl.DENSFULL_ARMS:
    ans = _integer_answers(f"{args.runs}/{arm}.densfull40*.shard*.jsonl", conds)
    row = {}
    for cond in ("clustering", "filler"):
      diffs = []
      for iid, (c, _, n) in stats.items():
        a, b = ans.get((iid, cond)), ans.get((iid, "none"))
        if not a or not b or a[0] == a[1] or b[0] == b[1]:
          continue
        x = c * (n - 1) - a[1]
        if abs(x) < 0.5:
          continue
        toward = lambda e: float((e > 0) == (x > 0))  # noqa: E731
        diffs.append(toward(a[0] - a[1]) - toward(b[0] - b[1]))
      if len(diffs) >= 10:
        row[f"{cond}-none_toward_prior_rate"] = _paired_boot(diffs)
    out["densfull40"][arm] = row

  rows = _prompts("prompts.rewire_shared.jsonl")
  rstats = _mean_clustering(rows)
  level_of = {d["instance_id"]: d["level"] for d in rows}
  for path in sorted(glob.glob(f"{args.runs}/*.rewire_shared.jsonl")):
    arm = os.path.basename(path).split(".rewire_shared")[0]
    ans = _integer_answers(path, conds)
    row = {}
    for level in ("low", "base", "high"):
      shift = []
      for iid, lv in level_of.items():
        a, b = ans.get((iid, "clustering")), ans.get((iid, "none"))
        if lv == level and a and b:
          clip = lambda e: max(-20, min(20, e))  # noqa: E731
          shift.append(clip(a[0] - a[1]) - clip(b[0] - b[1]))
      cs = [rstats[i][0] for i, lv in level_of.items() if lv == level]
      row[level] = dict(mean_clustering=sum(cs) / len(cs),
                        prior=sum(cs) / len(cs) * 39,
                        signed_error_shift=_paired_boot(shift) if shift else None)
    out["rewiring"][arm] = row
  return out


def _yes_no(pattern):
  """density -> iid -> cond -> (gold_yes, pred_yes), hit_cap dropped."""
  out = collections.defaultdict(lambda: collections.defaultdict(dict))
  for path in sorted(glob.glob(pattern)):
    with open(path, encoding="utf-8") as fh:
      for line in fh:
        if not line.strip():
          continue
        r = json.loads(line)
        if r["task"] != "edge_existence" or r.get("hit_cap"):
          continue
        out[abl.density_of(r["instance_id"])][r["instance_id"]].setdefault(
            r["condition"],
            (str(r["gold"]).strip().lower().startswith("yes"),
             scoring.extract_answer(r["response"], "edge_existence") == "Yes"))
  return out


def _sdt_by_cond(by_iid):
  counts = collections.defaultdict(lambda: [0, 0, 0, 0])  # hits, n_yes, fa, n_no
  for conds in by_iid:
    for cond, (gold, pred) in conds.items():
      c = counts[cond]
      if gold:
        c[0] += pred
        c[1] += 1
      else:
        c[2] += pred
        c[3] += 1
  return {cond: sdt(*c) for cond, c in counts.items() if c[1] and c[3]}


def test_sdt(args):
  out = {}
  rng = random.Random(0)
  for arm in abl.DENSFULL_ARMS:
    for corpus in CORPORA:
      data = _yes_no(f"{args.runs}/{arm}.{corpus}.shard*.jsonl")
      if not data:
        continue
      levels = sorted(data)
      point = {d: _sdt_by_cond(list(data[d].values())) for d in levels}
      boots = []
      for _ in range(1000):
        draw = {}
        for d in levels:
          ids = list(data[d])
          draw[d] = _sdt_by_cond([data[d][ids[rng.randrange(len(ids))]] for _ in ids])
        boots.append(draw)
      conds = sorted(set.intersection(*(set(point[d]) for d in levels)) - {"none"})

      def mean_delta(est, cond, i):
        return sum(est[d][cond][i] - est[d]["none"][i] for d in levels) / len(levels)

      row = {"none": {f"{d:.2f}": dict(d_prime=point[d]["none"][0],
                                       c=point[d]["none"][1]) for d in levels}}
      for cond in conds:
        entry = {}
        for i, name in ((0, "delta_d_prime"), (1, "delta_c")):
          # A resample can lose every gold-yes item at a sparse level; drop
          # that draw rather than the whole condition.
          draws = sorted(mean_delta(b, cond, i) for b in boots
                         if all(cond in b[d] and "none" in b[d] for d in levels))
          entry[name] = dict(mean_over_densities=mean_delta(point, cond, i),
                             ci=[draws[int(0.025 * (len(draws) - 1))],
                                 draws[int(0.975 * (len(draws) - 1))]],
                             n_boot=len(draws))
        entry["per_density"] = {f"{d:.2f}": dict(
            d_prime=point[d][cond][0], c=point[d][cond][1]) for d in levels}
        row[cond] = entry
      out[f"{arm}|{corpus}"] = row
  return out


def _zeroed(pattern, budget=None):
  """(task, iid, cond) -> 0/1 with truncation scored wrong, optionally
  re-truncating at `budget` new tokens."""
  out = {}
  for path in sorted(glob.glob(pattern)):
    with open(path, encoding="utf-8") as fh:
      for line in fh:
        if not line.strip():
          continue
        r = json.loads(line)
        key = (r["task"], r["instance_id"], r["condition"])
        if key in out:
          continue
        if r.get("hit_cap") or (budget and r["n_new_tokens"] > budget):
          out[key] = 0.0
          continue
        out[key] = scoring.score_one(scoring.extract_answer(r["response"], r["task"]),
                                     r["gold"], r["task"])["exact"]
  return out


def test_budget(args):
  out = {}
  for arm in ("qwen3-1.7b", "qwen3-4b"):
    plain = _zeroed(f"{args.runs}/{arm}.densfull40.shard*.jsonl")
    think = _zeroed(f"{args.runs}/{arm}-think.densfull40.shard*.jsonl")
    capped = _zeroed(f"{args.runs}/{arm}-think.densfull40.shard*.jsonl", PLAIN_BUDGET)
    lengths = collections.defaultdict(list)
    for path in glob.glob(f"{args.runs}/{arm}-think.densfull40.shard*.jsonl"):
      with open(path, encoding="utf-8") as fh:
        for r in map(json.loads, filter(str.strip, fh)):
          lengths[r["task"]].append(not r.get("hit_cap")
                                    and r["n_new_tokens"] <= PLAIN_BUDGET)
    within = {t: sum(v) / len(v) for t, v in lengths.items()}
    tasks = sorted({t for t, _, _ in think})
    conds = sorted({c for _, _, c in think} - {"none"})
    row = {}
    for task in tasks:
      entry = {}
      for label, table in (("plain_2048", plain), ("think_8192", think),
                           ("think_at_2048", capped)):
        none = {i: v for (t, i, c), v in table.items() if t == task and c == "none"}
        e = {"acc_none": sum(none.values()) / len(none)}
        for cond in conds:
          pairs = [(none[i], v) for (t, i, c), v in table.items()
                   if t == task and c == cond and i in none]
          e[cond] = 100.0 * sum(v - b for b, v in pairs) / len(pairs)
        entry[label] = e
      entry["think_share_within_2048"] = within[task]
      row[task] = entry
    out[arm] = row
  return out


def test_mae(args):
  with open("superseded/ci_all.json", encoding="utf-8") as fh:
    cells = json.load(fh)
  out = {}
  for key, v in cells.items():
    arm, task, cond = key.split("|")
    if v.get("mae_none") is None:
      continue
    out.setdefault(f"{arm}|{task}", {"none": v["mae_none"]})[cond] = v["mae_cond"]
  return out


TESTS = {"interaction": test_interaction, "logit": test_logit,
         "headline": test_headline, "rewiring": test_rewiring,
         "tokens": test_tokens, "prior": test_prior, "sdt": test_sdt,
         "budget": test_budget, "mae": test_mae}


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--runs", default="runs")
  ap.add_argument("--shortcuts-n40", default="shortcuts_n40_flat.json")
  ap.add_argument("--shortcuts", default="shortcuts.json")
  ap.add_argument("--tokenizer", default=None, help="a Qwen3 tokenizer.json")
  ap.add_argument("--test", action="append", choices=sorted(TESTS))
  ap.add_argument("--json", default="superseded/review_checks.json")
  args = ap.parse_args()
  with open(args.shortcuts_n40, encoding="utf-8") as fh:
    args.bars_n40 = json.load(fh)
  with open(args.shortcuts, encoding="utf-8") as fh:
    args.bars_pub = json.load(fh)

  results = {}
  if os.path.exists(args.json):
    with open(args.json, encoding="utf-8") as fh:
      results = json.load(fh)
  for name in args.test or list(TESTS):
    print(f"== {name}", flush=True)
    results[name] = TESTS[name](args)
    print(json.dumps(results[name], indent=1, default=str)[:4000], flush=True)
    with open(args.json, "w", encoding="utf-8") as fh:  # per test: a crash keeps the rest
      json.dump(results, fh, indent=1, default=str)
  print(f"wrote {args.json}")


if __name__ == "__main__":
  main()
