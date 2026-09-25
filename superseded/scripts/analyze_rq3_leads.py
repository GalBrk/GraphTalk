"""CPU-only follow-ups on the one non-answer effect (RQ3): `clustering` on
`node_degree` for `qwen3-1.7b`, dedicated 400-graph-per-density sweep.

The run files carry no prompt, so each graph is rebuilt from its seed with
`build_size_sweep`'s exact recipe and checked against the stored gold
before any analysis uses it.

  selection  -- how many non-answer cells the main sweep screened, whether
                the dedicated sweep reuses main-sweep graphs, and the effect
                on graphs the screen never saw
  hetero     -- does the gain depend on the queried node (degree, local
                clustering, list position)?
  errors     -- shape of the errors (off-by-one vs. large misses)
  behaviour  -- response length, and whether errors are transcription
                (wrong neighbour list) or counting (right list, wrong count)

  PYTHONPATH=. python superseded/scripts/analyze_rq3_leads.py [--test NAME ...]
"""
import argparse
import glob
import json
import random
import re
import sys

import networkx as nx

from graphtalk import graphqa, prompts, scoring
from graphtalk import diverse_corpus

sys.path[:0] = ["superseded/scripts", "scripts"]
import analyze_baseline_law as abl  # noqa: E402
import build_size_sweep as bss  # noqa: E402

ARM, TASK, N = "qwen3-1.7b", "node_degree", 40
REP_SEED = 20760906
GLOBS = [f"runs/{ARM}.{c}.shard*of5.jsonl"
         for c in ("degdens40", "degdens40hi", "degdensfill", "degdensrep")]
CONDS = ("none", "clustering", "filler", "components")
OUT = "superseded/rq3_leads.json"


# ---------------------------------------------------------------- data

def rebuild(iid):
  """Graph and queried node for `node_degree/size40/p<p>[/s<seed>]/<i>`."""
  parts = iid.split("/")
  p = abl.density_of(iid)
  seed = int(parts[3][1:]) if parts[3].startswith("s") else bss.DEFAULT_SEED
  s = seed + 1000000 * (1 + int(round(p * 1000))) + 1000 * N + int(parts[-1])
  g = graphqa.canonical(nx.erdos_renyi_graph(N, p, seed=s))
  row = diverse_corpus.make_row(g, TASK, random.Random(s))
  return g, int(re.search(r"node (\d+)\?", row["task_description"]).group(1)), row["gold"]


def load():
  rows, seen = {}, set()
  for pattern in GLOBS:
    for path in sorted(glob.glob(pattern)):
      with open(path, encoding="utf-8") as fh:
        for line in fh:
          if not line.strip():
            continue
          r = json.loads(line)
          k = (r["instance_id"], r["condition"])
          if k in seen:
            continue
          seen.add(k)
          pred = scoring.extract_answer(r["response"], TASK)
          try:
            pred = int(str(pred).strip())
          except (TypeError, ValueError):
            pred = None
          rows[k] = dict(pred=pred, gold=int(r["gold"]), cap=bool(r.get("hit_cap")),
                         exact=int(pred == int(r["gold"])), resp=r["response"],
                         ntok=r.get("n_new_tokens"))
  graphs = {}
  for iid in {i for i, _ in rows}:
    g, v, gold = rebuild(iid)
    for c in CONDS:
      if (iid, c) in rows:
        assert int(gold) == rows[(iid, c)]["gold"], f"rebuild mismatch {iid}"
    graphs[iid] = (g, v)
  return rows, graphs


def pairs(rows, a, b, pred=lambda i: True):
  """[(iid, score_a, score_b)] over non-capped pairs."""
  out = []
  for (i, c) in rows:
    if c != a or (i, b) not in rows or not pred(i):
      continue
    ra, rb = rows[(i, a)], rows[(i, b)]
    if ra["cap"] or rb["cap"]:
      continue
    out.append((i, ra["exact"], rb["exact"]))
  return out


def summary(ps, n_boot=2000, seed=0):
  if not ps:
    return None
  d = [a - b for _, a, b in ps]
  rng = random.Random(seed)
  means = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
  return dict(delta=round(100 * sum(d) / len(d), 2),
              ci=[round(100 * means[int(0.025 * (n_boot - 1))], 2),
                  round(100 * means[int(0.975 * (n_boot - 1))], 2)],
              p=scoring.mcnemar([b for _, _, b in ps], [a for _, a, _ in ps])["p_value"],
              n=len(ps))


def is_rep(i):
  return abl.is_replication_seed(i)


# ---------------------------------------------------------------- tests

def test_selection(rows, graphs):
  """Screen size, graph overlap with the screen, and the unscreened effect."""
  # Candidate cells in the main-sweep screen: every (arm, non-degenerate
  # task, non-answer condition) the n=40 bars do not mark as a route.
  bars = json.load(open("shortcuts_n40_flat.json"))
  cand = [(arm, t, c) for arm in abl.DENSFULL_ARMS
          for t in ("edge_count", "node_degree", "connected_nodes", "edge_existence")
          for c in ("components", "clustering", "rwse")
          if not abl.offers_route(bars, t, c)]
  # Does the dedicated sweep reuse main-sweep graphs? Same builder, same
  # seed and density keys -> index i < 100 is the same graph iff prompts match.
  main = {}
  for f in ("prompts.densfull40.jsonl", "prompts.densfull40hi.jsonl"):
    for line in open(f, encoding="utf-8"):
      r = json.loads(line)
      if r["task"] == TASK and r["condition"] == "none":
        main[r["instance_id"]] = r["prompt"]
  shared = 0
  for iid, prompt in main.items():
    if iid in graphs:
      g, v = graphs[iid]
      rebuilt = prompts.build_prompt(g, "none", f"Q: What is the degree of node {v}?\nA: ")
      shared += rebuilt == prompt
  in_screen = lambda i: i in main
  fresh = lambda i: not is_rep(i) and i not in main
  out = dict(
      screen_cells=len(cand),
      screen_cells_by_density=len(cand) * 4,
      dedicated_ids_in_screen=sum(1 for i in graphs if in_screen(i)),
      identical_prompts=shared,
      screened_graphs=dict(clust_none=summary(pairs(rows, "clustering", "none", in_screen)),
                           clust_filler=summary(pairs(rows, "clustering", "filler", in_screen))),
      unscreened_graphs=dict(clust_none=summary(pairs(rows, "clustering", "none", fresh)),
                             clust_filler=summary(pairs(rows, "clustering", "filler", fresh))),
      replication=summary(pairs(rows, "clustering", "none", is_rep)),
  )
  rep_p = out["replication"]["p"]
  out["replication_bonferroni_over_screen"] = min(1.0, rep_p * len(cand))
  # The paper's pooled estimate at the intermediate densities. The range was
  # chosen from the default seed, so the replication seed is its
  # out-of-sample check.
  mid = lambda i: not is_rep(i) and abl.density_of(i) in (0.35, 0.5)
  rep_mid = lambda i: is_rep(i) and abl.density_of(i) in (0.35, 0.5)
  out["mid_pooled"] = dict(
      clust_none=summary(pairs(rows, "clustering", "none", mid)),
      clust_filler=summary(pairs(rows, "clustering", "filler", mid)),
      filler_none=summary(pairs(rows, "filler", "none", mid)),
      replication_clust_none=summary(pairs(rows, "clustering", "none", rep_mid)))
  # Holm across the 14 per-density tests behind "beats both controls at
  # p=.35/.50" (7 densities x {vs none, vs filler}).
  tests = []
  for p in sorted({abl.density_of(i) for i in graphs if not is_rep(i)}):
    for b in ("none", "filler"):
      s = summary(pairs(rows, "clustering", b,
                        lambda i, p=p: not is_rep(i) and abl.density_of(i) == p))
      tests.append((f"p{p:g}_vs_{b}", s["delta"], s["p"]))
  order = sorted(range(len(tests)), key=lambda k: tests[k][2])
  holm, run = {}, 0.0
  for rank, k in enumerate(order):
    run = max(run, min(1.0, (len(tests) - rank) * tests[k][2]))
    holm[tests[k][0]] = dict(delta=tests[k][1], p=tests[k][2], holm=run)
  out["per_density_holm"] = holm
  return out


def covariates(g, v):
  d = g.degree(v)
  p = nx.density(g)
  return dict(z=(d - p * (N - 1)) / ((N - 1) * p * (1 - p)) ** 0.5,
              c=nx.clustering(g, v) - nx.average_clustering(g),
              pos=v)


def _terciles(rows, graphs, a, b, key, pred):
  """Within-density terciles of a covariate, so density cannot drive them."""
  groups = {0: [], 1: [], 2: []}
  by_p = {}
  for i, x, y in pairs(rows, a, b, pred):
    by_p.setdefault(abl.density_of(i), []).append((covariates(*graphs[i])[key], i, x, y))
  for vals in by_p.values():
    vals.sort()
    for k, (_, i, x, y) in enumerate(vals):
      groups[min(2, 3 * k // len(vals))].append((i, x, y))
  return {["low", "mid", "high"][t]: summary(ps) for t, ps in groups.items()}


def _ols(rows, graphs, a, b, pred):
  ps = pairs(rows, a, b, pred)
  levels = sorted({abl.density_of(i) for i, _, _ in ps})
  data, names = [], ["z", "c", "pos"] + [f"p{p:g}" for p in levels]
  for i, x, y in ps:
    cv = covariates(*graphs[i])
    data.append([100 * (x - y), cv["z"], 10 * cv["c"], cv["pos"] / 10]
                + [1.0 * (abl.density_of(i) == p) for p in levels])
  import numpy as np
  import statsmodels.api as sm
  arr = np.array(data)
  fit = sm.OLS(arr[:, 0], arr[:, 1:]).fit(cov_type="HC1")
  return {n: dict(coef=round(float(fit.params[k]), 2), p=float(fit.pvalues[k]))
          for k, n in enumerate(names[:3])}


def test_hetero(rows, graphs):
  out = {}
  scopes = {"all": lambda i: not is_rep(i),
            "mid": lambda i: not is_rep(i) and abl.density_of(i) in (0.35, 0.5)}
  for b in ("none", "filler"):
    for sname, pred in scopes.items():
      key = f"clust_vs_{b}/{sname}"
      out[key] = {cov: _terciles(rows, graphs, "clustering", b, cov, pred)
                  for cov in ("z", "c", "pos")}
      out[key]["ols"] = _ols(rows, graphs, "clustering", b, pred)
  # Is the position pattern specific to clustering, and is it just where
  # `none` is weakest? The replication seed was not used above, so it is an
  # out-of-sample check on whatever pattern the default seed shows.
  for c in ("filler", "components"):
    for sname, pred in scopes.items():
      out[f"{c}_vs_none/{sname}/pos"] = _terciles(rows, graphs, c, "none", "pos", pred)
  out["replication/clust_vs_none/pos"] = _terciles(rows, graphs, "clustering", "none",
                                                   "pos", is_rep)
  for sname, pred in scopes.items():
    acc = {}
    for label, lo, hi in (("low", 0, 13), ("mid", 14, 26), ("high", 27, 39)):
      rs = [r for (i, c), r in rows.items() if c == "none" and pred(i)
            and not r["cap"] and lo <= graphs[i][1] <= hi]
      acc[label] = round(100 * sum(r["exact"] for r in rs) / len(rs), 1)
    out[f"none_accuracy_by_pos/{sname}"] = acc
  return out


def _band(p):
  return "low" if p <= 0.2 else "mid" if p <= 0.5 else "high"


def test_errors(rows, graphs):
  out = {}
  for band in ("low", "mid", "high"):
    for c in CONDS:
      rs = [r for (i, cc), r in rows.items() if cc == c and not r["cap"]
            and not is_rep(i) and _band(abl.density_of(i)) == band]
      if not rs:
        continue
      errs = [r["pred"] - r["gold"] for r in rs if r["pred"] is not None]
      n = len(rs)
      out[f"{band}/{c}"] = dict(
          n=n, acc=round(100 * sum(r["exact"] for r in rs) / n, 1),
          unparsed=round(100 * (n - len(errs)) / n, 1),
          off_by_1=round(100 * sum(abs(e) == 1 for e in errs) / n, 1),
          off_by_2=round(100 * sum(abs(e) == 2 for e in errs) / n, 1),
          off_3plus=round(100 * sum(abs(e) >= 3 for e in errs) / n, 1),
          under=round(100 * sum(e < 0 for e in errs) / n, 1),
          over=round(100 * sum(e > 0 for e in errs) / n, 1),
          mean_signed=round(sum(max(-20, min(20, e)) for e in errs) / max(1, len(errs)), 2))
  return out


_FILLER_WORDS = {"and", "the", "following", "nodes", "node"}


def transcribed(resp, v):
  """The neighbour list the response first writes for the queried node, or
  None when it writes none: the numbers after the first 'connected to'
  that follows a mention of node v, inline or as a bulleted list, up to the
  first other word (e.g. "Let's count")."""
  at = re.search(rf"\b{v}\b", resp)
  for m in re.finditer(r"connected to", resp[at.start() if at else 0:], re.I):
    nums = []
    for tok in re.finditer(r"\d+|[A-Za-z']+",
                           resp[(at.start() if at else 0) + m.end():]):
      t = tok.group()
      if t.isdigit():
        nums.append(int(t))
      elif t.lower() not in _FILLER_WORDS or nums:
        if t.lower() == "and" and nums:
          continue
        break
    if nums:
      return nums
  return None


def _check_transcribed():
  assert transcribed("Node 5 is connected to nodes 2, 25, 33.\nSo", 5) == [2, 25, 33]
  assert transcribed("**Node 24** is connected to the following nodes:\n\n  - 0\n  - 2\n\nLet's", 24) == [0, 2]
  assert transcribed("**Node 38 is connected to:**\n\n- 0\n- 3 and 7\n\nLet's", 38) == [0, 3, 7]
  assert transcribed("I count the edges of node 4.", 4) is None
  assert transcribed("degree of node 25, edges connected to it.\n- **Node 25** is "
                     "connected to the following nodes:\n  - 0\n  - 2\n", 25) == [0, 2]


def classify(r, g, v):
  if r["exact"]:
    return "correct"
  lst = transcribed(r["resp"], v)
  if lst is None:
    return "no_list"
  true = set(g.neighbors(v))
  if set(lst) != true or len(lst) != len(true):
    if set(lst) == set(g.neighbors(v - 1)) if v > 0 else False:
      return "wrong_line"
    if v < N - 1 and set(lst) == set(g.neighbors(v + 1)):
      return "wrong_line"
    return "bad_transcription"
  return "miscount" if r["pred"] != len(lst) else "other"


def test_behaviour(rows, graphs):
  out = {}
  for band in ("low", "mid", "high"):
    for c in CONDS:
      keys = [(i, cc) for (i, cc) in rows if cc == c and not is_rep(i)
              and _band(abl.density_of(i)) == band and not rows[(i, cc)]["cap"]]
      if not keys:
        continue
      kinds = {}
      for k in keys:
        kind = classify(rows[k], *graphs[k[0]])
        kinds[kind] = kinds.get(kind, 0) + 1
      toks = sorted(rows[k]["ntok"] for k in keys if rows[k]["ntok"] is not None)
      out[f"{band}/{c}"] = dict(
          n=len(keys), median_tokens=toks[len(toks) // 2] if toks else None,
          mentions_clustering=round(100 * sum("clustering" in rows[k]["resp"].lower()
                                              for k in keys) / len(keys), 1),
          **{f"pct_{k}": round(100 * n / len(keys), 1) for k, n in sorted(kinds.items())})
  # Paired: among instances wrong under none and right under clustering,
  # what kind of error did none make? (what clustering fixes)
  fixed, broke = {}, {}
  for i, a, b in pairs(rows, "clustering", "none", lambda i: not is_rep(i)):
    if a > b:
      k = classify(rows[(i, "none")], *graphs[i])
      fixed[k] = fixed.get(k, 0) + 1
    elif b > a:
      k = classify(rows[(i, "clustering")], *graphs[i])
      broke[k] = broke.get(k, 0) + 1
  out["none_error_fixed_by_clustering"] = fixed
  out["clustering_error_where_none_right"] = broke
  # What goes wrong in a bad transcription, at the densities where the
  # effect lives: dropped neighbours (omission) or invented ones (insertion)?
  for c in CONDS:
    kinds, missing, n_all = {}, 0, 0
    for (i, cc), r in rows.items():
      if cc != c or is_rep(i) or abl.density_of(i) not in (0.35, 0.5) or r["cap"]:
        continue
      g, v = graphs[i]
      lst, true = transcribed(r["resp"], v), set(g.neighbors(v))
      n_all += 1
      missing += len(true - set(lst or []))
      if r["exact"] or lst is None:
        continue
      s = set(lst)
      k = ("list_ok" if s == true and len(lst) == len(true) else
           "omission" if s < true else "insertion" if s > true else "mixed")
      kinds[k] = kinds.get(k, 0) + 1
    n_err = sum(kinds.values())
    out[f"mid_transcription/{c}"] = dict(
        errors=n_err, mean_missing_neighbours=round(missing / n_all, 2),
        **{f"pct_{k}": round(100 * v / n_err, 1) for k, v in sorted(kinds.items())})
  return out


TESTS = dict(selection=test_selection, hetero=test_hetero, errors=test_errors,
             behaviour=test_behaviour)


def main():
  _check_transcribed()
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--test", action="append", choices=list(TESTS))
  ap.add_argument("--out", default=OUT)
  args = ap.parse_args()
  rows, graphs = load()
  print(f"loaded {len(rows)} rows, {len(graphs)} graphs (all rebuilt golds match)")
  results = {}
  for name in args.test or TESTS:
    results[name] = TESTS[name](rows, graphs)
    print(f"\n== {name}\n" + json.dumps(results[name], indent=1))
    with open(args.out, "w") as fh:
      json.dump(results, fh, indent=1)


if __name__ == "__main__":
  main()
