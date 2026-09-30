"""A learned graph-blind bar for the clustering and rwse primers of the 40-node sweep.

The solver bars (data/shortcuts_n40.json) come from hand-written rules. This
fits a lookup from the printed values alone to the answer: node_degree from the
queried node's printed value (rwse: its 2- and 3-step pair), edge_existence from
the two endpoints' values. Each density's lookup is fit on generated graphs
(seed FIT) and kept only if it beats the majority answer on held-out graphs
(seed TEST); it is then scored on the sweep's saved prompts. Every feature is
read from the primer text by graphtalk.shortcuts.parse_primer: the solver never
sees a graph. Graphs are used only to label the fit and test primers and to
check that the saved primers re-render byte for byte.

  PYTHONPATH=. python scripts/blind_bars.py > outputs/n40-sweep/blind_bars.txt

Rule R1 for the models' accuracy: a truncated response is never correct.
"""
import argparse
import collections
import json
import random
import re

import numpy as np
import pandas as pd

from graphtalk import outcomes, primers, shortcuts

DENS = (0.10, 0.20, 0.35, 0.50, 0.65, 0.75, 0.85)
SWEEP_SEED = 20260906
FIT, TEST = 555_555, 777_777          # the n40 solver bars' fit and test seeds
N_FIT, N_TEST, PAIRS = 1000, 300, 100
CONDS = ("clustering", "rwse")
TASKS = ("node_degree", "edge_existence")
PROMPTS = ("data/prompts/prompts.densfull40.jsonl", "data/prompts/prompts.densfull40hi.jsonl")
ARMS = ("qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think")
Q = {"node_degree": re.compile(r"Q: What is the degree of node (\d+)\?"),
     "edge_existence": re.compile(r"Q: Does an edge exist between Node (\d+) and Node (\d+)\?")}


def printed(primer, cond):
  """{node: printed value}, read from the primer text only."""
  p = shortcuts.parse_primer(primer)
  if cond == "clustering":
    return p.clustering
  return {k: (v[2], v[3]) for k, v in p.rwse.items()}


def key(v, task, nodes):
  if task == "node_degree":
    return v[nodes[0]]
  return tuple(sorted((v[nodes[0]], v[nodes[1]]), key=str))


def labelled(graphs, cond, task, seed):
  """(key, answer) for every node (node_degree) or PAIRS random pairs per graph."""
  rng = random.Random(seed)
  out = []
  for g in graphs:
    v = printed(primers.build_primer(g, cond), cond)
    if task == "node_degree":
      out += [(key(v, task, (t,)), str(g.degree(t))) for t in g.nodes]
    else:
      for a, b in (rng.sample(sorted(g.nodes), 2) for _ in range(PAIRS)):
        out.append((key(v, task, (a, b)), "Yes" if g.has_edge(a, b) else "No"))
  return out


class Lookup:
  """The most common answer per printed key; the majority answer for a key never seen."""

  def __init__(self, rows):
    self.table = collections.defaultdict(collections.Counter)
    for k, y in rows:
      self.table[k][y] += 1
    self.majority = collections.Counter(y for _, y in rows).most_common(1)[0][0]

  def __call__(self, k):
    return self.table[k].most_common(1)[0][0] if k in self.table else self.majority


def main():
  argparse.ArgumentParser(description=__doc__,
                          formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
  rows = [json.loads(line) for p in PROMPTS for line in open(p, encoding="utf-8")]
  rows = [r for r in rows if r["task"] in TASKS and r["condition"] in CONDS]
  sweep = {d: shortcuts.generate_n40_corpus(100, SWEEP_SEED, d) for d in DENS}
  differ = sum(primers.build_primer(sweep[r["density_class"]][int(r["instance_id"].rsplit("/", 1)[1])],
                                    r["condition"]) != r["prompt"].split("\n\n", 1)[0] for r in rows)
  shortcuts.Split(fit_seed=FIT, test_seed=TEST)   # raises if the seeds are equal
  print(f"[blindcheck] {len(rows)} saved clustering/rwse primers (node_degree, edge_existence) "
        f"re-rendered from their graphs: {differ} differ. Fit seed {FIT} ({N_FIT} graphs per "
        f"density), test seed {TEST} ({N_TEST}), sweep seed {SWEEP_SEED} (100)")
  assert differ == 0

  bars = json.load(open("data/shortcuts_n40.json", encoding="utf-8"))
  frame = pd.read_csv("outputs/n40-sweep/frame.csv")
  frame = frame[frame.task.isin(TASKS) & frame.condition.isin(("none",) + CONDS)]
  frame["correct"] = (outcomes.outcome(frame.exact, frame.hit_cap) == outcomes.CORRECT).astype(int)
  blind_right = {}
  print("[blindbar] graph-blind accuracy (%) on the sweep's saved prompts, by density: "
        "rule bar (data/shortcuts_n40.json) / learned lookup / majority answer; "
        "`majority` marks a density whose lookup did not beat the majority answer on "
        "held-out graphs and falls back to it")
  unique = {}
  fit = {d: shortcuts.generate_n40_corpus(N_FIT, FIT, d) for d in DENS}
  test = {d: shortcuts.generate_n40_corpus(N_TEST, TEST, d) for d in DENS}
  for cond in CONDS:
    unique[cond] = [np.mean([sum(c == 1 for c in collections.Counter(
        printed(primers.build_primer(g, cond), cond).values()).values()) / len(g)
        for g in test[d]]) for d in DENS]
    for task in TASKS:
      cells, learned, major = [], [], []
      for d in DENS:
        solver = Lookup(labelled(fit[d], cond, task, 1))
        held = labelled(test[d], cond, task, 2)
        use_lookup = (np.mean([solver(k) == y for k, y in held])
                      > np.mean([solver.majority == y for _, y in held]))
        s = [r for r in rows if r["task"] == task and r["condition"] == cond
             and r["density_class"] == d]
        hits = []
        for r in s:
          nodes = tuple(int(x) for x in Q[task].search(r["prompt"]).groups())
          guess = solver(key(printed(r["prompt"].split("\n\n", 1)[0], cond), task, nodes)) \
              if use_lookup else solver.majority
          hits.append(guess == r["gold"])
          blind_right[(task, cond, r["instance_id"])] = guess == r["gold"]
        learned.append(100 * np.mean(hits))
        major.append(100 * np.mean([solver.majority == r["gold"] for r in s]))
        cells.append(f"p={d:.2f} {100 * bars[f'{d:g}'][f'{task}/{cond}']:.1f}/{learned[-1]:.1f}/"
                     f"{major[-1]:.1f}{'' if use_lookup else ' majority'}")
      print(f"  {task} x {cond}: " + ", ".join(cells))
      print(f"    mean over densities: rule {100 * np.mean([bars[f'{d:g}'][f'{task}/{cond}'] for d in DENS]):.1f}, "
            f"learned {np.mean(learned):.1f}, majority {np.mean(major):.1f}")

  print("[blindunique] share of nodes (%) whose printed value is unique within its graph, "
        "held-out graphs, p = .10 to .85")
  for cond in CONDS:
    print(f"  {cond}: " + ", ".join(f"{100 * u:.0f}" for u in unique[cond]))

  print("[blindmodels] per arm, correct (%) under the primer on the prompts the learned "
        "lookup answers correctly, against the same graphs without a primer (n)")
  for task in TASKS:
    for cond in CONDS:
      parts = []
      for arm in ARMS:
        a = frame[(frame.arm == arm) & (frame.task == task)]
        y = a[a.condition == cond]
        y = y[[blind_right[(task, cond, i)] for i in y.instance_id]]
        x = a[a.condition == "none"].set_index("graph_id").correct.reindex(y.graph_id)
        parts.append(f"{arm} {100 * y.correct.mean():.1f} ({100 * x.mean():.1f}) n={len(y)}")
      print(f"  {task} x {cond}: " + "; ".join(parts))


if __name__ == "__main__":
  main()
