"""Promotes candidates (d) and (e) from docs/candidate-analyses.md: the
discordant-pair (churn) decomposition and the paired generated-token delta,
one pass over densfull40.

(d) A raw net delta hides how much of it is two roughly-equal, cancelling
    groups of instances (helped vs. hurt) rather than a uniform shift --
    `sig/churn` = |helped-hurt| / (helped+hurt) is near 0 for a
    coin-flip-like effect and near 1 for a one-directional one.
(e) Primers change how long the model generates before answering, which is
    a measurable process effect distinct from accuracy -- e.g. `filler`
    shortens generation in every arm even though it isn't the shortest
    primer text, and `degree` adds ~670 tokens for qwen3-1.7b-think on
    node_degree alone.

  PYTHONPATH=. python scripts/analyze_churn_and_length.py --json churn_len_densfull40.json
"""

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, "scripts")
import analyze_baseline_law as abl  # noqa: E402  (DENSFULL_ARMS, density_of)
from graphtalk import scoring  # noqa: E402


def load(patterns):
  """-> {(task, density, condition, instance_id): (score|None, n_new_tokens)}"""
  out, seen = {}, set()
  for pattern in patterns:
    for path in sorted(glob.glob(pattern)):
      if ".got." in os.path.basename(path):
        continue
      with open(path, encoding="utf-8") as fh:
        for line in fh:
          if not line.strip():
            continue
          r = json.loads(line)
          key = (r["task"], abl.density_of(r["instance_id"]),
                 r["condition"], r["instance_id"])
          if key in seen:
            continue
          seen.add(key)
          score = None if r.get("hit_cap") else scoring.score_one(
              scoring.extract_answer(r["response"], r["task"]),
              r["gold"], r["task"])["primary"]
          out[key] = (score, r.get("n_new_tokens"))
  return out


def cells(data, control="none", min_pairs=10):
  """Per (task, density, condition) churn + mean paired token delta.

  `hit_cap` pairs are dropped from both the churn and the length numbers --
  an 8192-token response is a censored length observation, not a real one.
  """
  grouped = collections.defaultdict(dict)
  for (task, dens, cond, iid), value in data.items():
    grouped[(task, dens, cond)][iid] = value
  rows = []
  for (task, dens, cond), values in sorted(grouped.items(), key=repr):
    if cond == control:
      continue
    base = grouped.get((task, dens, control), {})
    helped = hurt = n = 0
    token_delta_sum, token_delta_n = 0.0, 0
    for iid, (score, tokens) in values.items():
      ctrl = base.get(iid)
      if ctrl is None:
        continue
      ctrl_score, ctrl_tokens = ctrl
      if ctrl_score is None or score is None:
        continue
      if ctrl_tokens is not None and tokens is not None:
        token_delta_sum += tokens - ctrl_tokens
        token_delta_n += 1
      n += 1
      if score > ctrl_score:
        helped += 1
      elif score < ctrl_score:
        hurt += 1
    if n < min_pairs:
      continue
    rows.append(dict(
        task=task, density=dens, condition=cond, n=n,
        helped=helped, hurt=hurt, churn=helped + hurt,
        delta=100.0 * (helped - hurt) / n,
        sig_over_churn=(abs(helped - hurt) / (helped + hurt)
                        if helped + hurt else None),
        dtokens=(token_delta_sum / token_delta_n if token_delta_n else None),
    ))
  return rows


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--runs", default="runs/{arm}.densfull40.shard*.jsonl")
  ap.add_argument("--json", default="churn_len_densfull40.json")
  args = ap.parse_args()

  all_rows = {}
  for arm in abl.DENSFULL_ARMS:
    rows = cells(load([args.runs.format(arm=arm)]))
    for r in rows:
      r["arm"] = arm
    all_rows[arm] = rows
    print(f"loaded {arm}: {len(rows)} cells", flush=True)

  flat = [r for rs in all_rows.values() for r in rs]
  print(f"\n=== churn summary, all {len(flat)} cells ===")
  tot_n = sum(r["n"] for r in flat)
  tot_ch = sum(r["churn"] for r in flat)
  tot_net = sum(abs(r["helped"] - r["hurt"]) for r in flat)
  print(f"  pairs {tot_n}, discordant {tot_ch} ({100.0*tot_ch/tot_n:.1f}%), "
        f"|net| {tot_net} = {100.0*tot_net/tot_ch:.1f}% of discordant pairs")

  # Equal weight per (task, density) cell, matching the original candidate
  # script -- not weighted by pair count, so a task with more density levels
  # doesn't dominate the mean.
  print("\n=== mean paired change in generated tokens vs none"
        " (equal weight per task/density cell) ===")
  conds = ["components", "clustering", "rwse", "degree", "filler", "all"]
  print(f"{'arm':<17}" + "".join(f"{c:>12}" for c in conds))
  for arm in abl.DENSFULL_ARMS:
    line = f"{arm:<17}"
    for cond in conds:
      sel = [r["dtokens"] for r in all_rows[arm]
             if r["condition"] == cond and r["dtokens"] is not None]
      line += f"{(sum(sel)/len(sel) if sel else float('nan')):>+12.1f}"
    print(line)

  with open(args.json, "w") as fh:
    json.dump(all_rows, fh, indent=0)
  print(f"\nwrote {args.json}")


if __name__ == "__main__":
  main()
