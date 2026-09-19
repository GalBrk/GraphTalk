"""Error taxonomy for the three tasks the plan singles out (step 4.6).

  node_degree      off-by-k distribution, neighbour-degree-copy rate (promotes
                   scripts/candidates/b_copying.py into a tested script), and
                   error rate as a function of the queried node's position in
                   the primer -- which for this corpus's canonical (sorted
                   node id) rendering is just the node's own id, so no offset
                   parsing is needed to answer that question.
  edge_existence   hit rate (P(Yes | gold Yes)) and false-alarm rate
                   (P(Yes | gold No)) per condition, to separate "the model
                   reads the graph better" from "the model says Yes more/less
                   often regardless of the graph" (a bias shift).
  node_count       the "39" rate per condition -- gold is always 40 at n=40,
                   so an off-by-one answer is a specific, checkable failure
                   mode, not just "wrong".

  PYTHONPATH=. python scripts/analyze_error_taxonomy.py --json error_taxonomy.json
"""

import argparse
import collections
import glob
import json
import re

from graphtalk import scoring

ARMS = ("qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think")
CONDS = ("none", "components", "clustering", "rwse", "degree", "filler", "all")

_ADJ = re.compile(r"Node (\d+) is connected to nodes? ([\d, ]+)\.")
_QRY_DEGREE = re.compile(r"What is the degree of node (\d+)\?")


def _responses(pattern, task):
  seen = {}
  for path in sorted(glob.glob(pattern)):
    with open(path, encoding="utf-8") as fh:
      for line in fh:
        if not line.strip():
          continue
        r = json.loads(line)
        if r["task"] != task or r.get("hit_cap"):
          continue
        seen.setdefault((r["instance_id"], r["condition"]), r)
  return seen


def _node_degree_graphs(prompts_path):
  """instance_id -> (degree-by-node, queried node, its neighbours)."""
  out = {}
  with open(prompts_path, encoding="utf-8") as fh:
    for line in fh:
      d = json.loads(line)
      if d["task"] != "node_degree" or d["instance_id"] in out:
        continue
      adj = {int(n): [int(x) for x in nb.replace(" ", "").split(",") if x]
             for n, nb in _ADJ.findall(d["prompt"])}
      q = _QRY_DEGREE.search(d["prompt"])
      if not q:
        continue
      k = int(q.group(1))
      deg = {n: len(v) for n, v in adj.items()}
      out[d["instance_id"]] = (deg, k, adj.get(k, []))
  return out


def node_degree_taxonomy(prompts_path, runs_glob_tmpl):
  """-> {(arm, cond): {n_wrong, off_by, neighbour_copy, position_bins}}"""
  graphs = _node_degree_graphs(prompts_path)
  out = {}
  position_rows = []  # (arm, cond, k, correct) for the position analysis
  for arm in ARMS:
    responses = _responses(runs_glob_tmpl.format(arm=arm), "node_degree")
    for (iid, cond), r in responses.items():
      if cond not in CONDS or iid not in graphs:
        continue
      deg, k, nbrs = graphs[iid]
      gold = int(r["gold"])
      got = scoring.extract_answer(r["response"], "node_degree")
      try:
        v = int(str(got).strip())
      except (TypeError, ValueError):
        continue
      correct = v == gold
      position_rows.append((arm, cond, k, correct))
      if correct:
        continue
      key = (arm, cond)
      bucket = out.setdefault(key, {
          "n_wrong": 0, "off_by": collections.Counter(),
          "neighbour_copy": 0, "adjacent_id_copy": 0,
      })
      bucket["n_wrong"] += 1
      off = v - gold
      bucket["off_by"][str(off) if abs(off) <= 3 else ("+" if off > 0 else "-")] += 1
      if any(deg.get(n) == v for n in nbrs):
        bucket["neighbour_copy"] += 1
      if deg.get(k - 1) == v or deg.get(k + 1) == v:
        bucket["adjacent_id_copy"] += 1

  for bucket in out.values():
    bucket["off_by"] = dict(bucket["off_by"])

  # Position (= queried node id, since primers render nodes in sorted order)
  # analysis: accuracy in four id quartiles, pooled per condition across arms.
  position = collections.defaultdict(lambda: collections.Counter())
  for arm, cond, k, correct in position_rows:
    quartile = min(3, k // 10)  # n=40 -> quartiles of 10 ids each
    position[cond][(quartile, "n")] += 1
    position[cond][(quartile, "correct")] += int(correct)
  position_table = {}
  for cond, counts in position.items():
    row = {}
    for q in range(4):
      n, c = counts[(q, "n")], counts[(q, "correct")]
      row[f"q{q}"] = (c / n, n) if n else (None, 0)
    position_table[cond] = row

  return out, position_table


def edge_existence_taxonomy(runs_glob_tmpl):
  """-> {(arm, cond): {hit_rate, false_alarm_rate, n_yes, n_no}}"""
  out = {}
  for arm in ARMS:
    responses = _responses(runs_glob_tmpl.format(arm=arm), "edge_existence")
    counts = collections.defaultdict(lambda: collections.Counter())
    for (_, cond), r in responses.items():
      if cond not in CONDS:
        continue
      gold_yes = str(r["gold"]).strip().lower().startswith("yes")
      predicted = scoring.extract_answer(r["response"], "edge_existence")
      pred_yes = predicted == "Yes"
      counts[cond]["n_gold_yes" if gold_yes else "n_gold_no"] += 1
      if gold_yes and pred_yes:
        counts[cond]["hit"] += 1
      if not gold_yes and pred_yes:
        counts[cond]["false_alarm"] += 1
    for cond, c in counts.items():
      ny, nn = c["n_gold_yes"], c["n_gold_no"]
      out[(arm, cond)] = {
          "hit_rate": (c["hit"] / ny) if ny else None,
          "false_alarm_rate": (c["false_alarm"] / nn) if nn else None,
          "n_gold_yes": ny, "n_gold_no": nn,
      }
  return out


def node_count_taxonomy(runs_glob_tmpl):
  """-> {(arm, cond): {n, rate_39, rate_correct}}"""
  out = {}
  for arm in ARMS:
    responses = _responses(runs_glob_tmpl.format(arm=arm), "node_count")
    counts = collections.defaultdict(lambda: collections.Counter())
    for (_, cond), r in responses.items():
      if cond not in CONDS:
        continue
      got = scoring.extract_answer(r["response"], "node_count")
      counts[cond]["n"] += 1
      if str(got).strip() == "39":
        counts[cond]["n_39"] += 1
      if str(got).strip() == str(r["gold"]).strip():
        counts[cond]["n_correct"] += 1
    for cond, c in counts.items():
      out[(arm, cond)] = {
          "n": c["n"], "rate_39": c["n_39"] / c["n"] if c["n"] else None,
          "rate_correct": c["n_correct"] / c["n"] if c["n"] else None,
      }
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--prompts", default="prompts.densfull40.jsonl")
  ap.add_argument("--runs", default="runs/{arm}.densfull40.shard*.jsonl")
  ap.add_argument("--json", default="error_taxonomy.json")
  args = ap.parse_args()

  nd_errors, nd_position = node_degree_taxonomy(args.prompts, args.runs)
  ee = edge_existence_taxonomy(args.runs)
  nc = node_count_taxonomy(args.runs)

  print("=== node_degree: how wrong answers are wrong ===")
  print(f"{'arm':<17}{'cond':<12}{'n_wrong':>8}{'nbr-copy':>10}{'id+-1':>8}")
  for (arm, cond), b in sorted(nd_errors.items()):
    print(f"{arm:<17}{cond:<12}{b['n_wrong']:>8}"
          f"{100*b['neighbour_copy']/b['n_wrong']:>9.1f}%"
          f"{100*b['adjacent_id_copy']/b['n_wrong']:>7.1f}%")

  print("\n=== node_degree: accuracy by queried-node-id quartile"
        " (0-9/10-19/20-29/30-39), pooled across arms ===")
  print(f"{'cond':<12}{'q0':>14}{'q1':>14}{'q2':>14}{'q3':>14}")
  for cond, row in sorted(nd_position.items()):
    cells = []
    for q in range(4):
      acc, n = row.get(f"q{q}", (None, 0))
      cells.append(f"{acc:.1%} (n={n})" if acc is not None else "-")
    print(f"{cond:<12}" + "".join(f"{c:>14}" for c in cells))

  print("\n=== edge_existence: hit rate vs false-alarm rate ===")
  print(f"{'arm':<17}{'cond':<12}{'hit':>8}{'false-alarm':>13}")
  for (arm, cond), v in sorted(ee.items()):
    if v["hit_rate"] is None:
      continue
    print(f"{arm:<17}{cond:<12}{v['hit_rate']:>7.1%}{v['false_alarm_rate']:>13.1%}")

  print("\n=== node_count: rate of answering exactly 39 (gold is always 40) ===")
  print(f"{'arm':<17}{'cond':<12}{'rate_39':>9}{'rate_correct':>13}")
  for (arm, cond), v in sorted(nc.items()):
    print(f"{arm:<17}{cond:<12}{v['rate_39']:>8.1%}{v['rate_correct']:>13.1%}")

  json.dump({
      "node_degree_errors": {f"{a}|{c}": v for (a, c), v in nd_errors.items()},
      "node_degree_position": nd_position,
      "edge_existence": {f"{a}|{c}": v for (a, c), v in ee.items()},
      "node_count": {f"{a}|{c}": v for (a, c), v in nc.items()},
  }, open(args.json, "w"), indent=1)
  print(f"\nwrote {args.json}")


if __name__ == "__main__":
  main()
