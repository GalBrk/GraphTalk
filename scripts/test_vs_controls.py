"""Paired tests against BOTH `filler` and `components`, not just `none`.

Every existing significance table (`ci_all.py`, `check_significance.py`) tests
each condition against `none` only. The paper claims some cells "beat both
`none` and `filler` significantly" (docs/paper-revision-handoff.md's error
list), but no script anywhere actually runs the vs-`filler` comparison -- it
doesn't exist until this script. `components` is added as a second control
because it is the length-cheap, near-content-free condition (37 chars mean at
n=40, vs. filler's 1829): a condition that only beats `none` but not
`components` is failing against a much weaker bar than it looks like it's
clearing.

Two Benjamini-Hochberg passes are reported side by side:
  - per (arm, task) family, matching score_full_density_sweep.py's scope;
  - one GLOBAL correction across every comparison this script runs, so a
    "significant" cell in the per-family view that would not survive
    correcting for testing hundreds of cells at once is visible as such.

  PYTHONPATH=. python scripts/test_vs_controls.py --corpus densfull40 --json vs_controls.json
"""

import argparse
import collections
import glob
import json

from graphtalk import scoring, significance

ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"]
CONDS = ["components", "clustering", "rwse", "degree", "filler", "all"]
CONTROLS = ["none", "filler", "components"]
TASKS = ["connected_nodes", "cycle_check", "edge_count", "edge_existence",
         "node_count", "node_degree"]


def load(arm, corpus):
  seen, rows = set(), []
  for path in sorted(glob.glob(f"runs/{arm}.{corpus}.shard*.jsonl")):
    with open(path, encoding="utf-8") as fh:
      for line in fh:
        if not line.strip():
          continue
        r = json.loads(line)
        k = (r["instance_id"], r["condition"], r["style"])
        if k in seen:
          continue
        seen.add(k)
        rows.append(r)
  return rows


def cells_beating_both_controls(results):
  """Cells whose delta vs `none` AND vs `filler` are both positive and
  survive `bh_global_reject`. `results` already carries per-row BH flags.
  """
  by_cell = collections.defaultdict(dict)
  for r in results:
    by_cell[(r["arm"], r["task"], r["condition"])][r["control"]] = r
  return [
      cell for cell in by_cell.values()
      if "none" in cell and "filler" in cell
      and cell["none"]["delta"] > 0 and cell["none"]["bh_global_reject"]
      and cell["filler"]["delta"] > 0 and cell["filler"]["bh_global_reject"]
  ], by_cell


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--corpus", default="densfull40")
  ap.add_argument("--json", default=None)
  ap.add_argument("--n-perm", type=int, default=10000)
  args = ap.parse_args()
  out_path = args.json or f"vs_controls_{args.corpus}.json"

  results = []  # list of dicts, filled in with p-values then BH flags
  for arm in ARMS:
    rows = load(arm, args.corpus)
    if not rows:
      continue
    # (task, iid) -> condition -> (capped, exact)
    scored = collections.defaultdict(dict)
    for r in rows:
      t = r["task"]
      capped = bool(r.get("hit_cap"))
      if capped:
        scored[(t, r["instance_id"])][r["condition"]] = (True, 0.0)
        continue
      res = scoring.score_one(
          scoring.extract_answer(r["response"], t), r["gold"], t)
      scored[(t, r["instance_id"])][r["condition"]] = (False, res["exact"])

    for t in TASKS:
      for cond in CONDS:
        for control in CONTROLS:
          if control == cond:
            continue
          ctrl_vals, treat_vals = [], []
          for key, v in scored.items():
            if key[0] != t or control not in v or cond not in v:
              continue
            ccap, ce = v[control]
            tcap, te = v[cond]
            if ccap or tcap:
              continue
            ctrl_vals.append(ce)
            treat_vals.append(te)
          if len(ctrl_vals) < 10:
            continue
          perm = significance.paired_permutation_test(
              ctrl_vals, treat_vals, n_perm=args.n_perm, seed=0)
          results.append({
              "arm": arm, "task": t, "condition": cond, "control": control,
              "n": len(ctrl_vals),
              "delta": 100.0 * (sum(treat_vals) - sum(ctrl_vals)) / len(ctrl_vals),
              "p_perm": perm["p_value"],
          })
      print(f"done {arm} {t}", flush=True)

  # Per (arm, task) family BH.
  families = collections.defaultdict(list)
  for i, r in enumerate(results):
    families[(r["arm"], r["task"])].append(i)
  for idxs in families.values():
    flags = significance.benjamini_hochberg([results[i]["p_perm"] for i in idxs])
    for i, flag in zip(idxs, flags):
      results[i]["bh_family_reject"] = flag

  # Global BH across every comparison in this run.
  global_flags = significance.benjamini_hochberg([r["p_perm"] for r in results])
  for r, flag in zip(results, global_flags):
    r["bh_global_reject"] = flag

  with open(out_path, "w") as fh:
    json.dump(results, fh, indent=1)
  print(f"\nwrote {len(results)} comparisons to {out_path}")

  # Headline: cells that beat BOTH none and filler, surviving the global BH.
  beats_both, by_cell = cells_beating_both_controls(results)
  print(f"\ncells that beat BOTH none and filler, surviving the GLOBAL BH: "
        f"{len(beats_both)} of {len(by_cell)}")
  for cell in beats_both:
    r = cell["none"]
    print(f"  {r['arm']:<18} {r['task']:<16} {r['condition']:<12} "
          f"vs none: {cell['none']['delta']:+.1f} (p={cell['none']['p_perm']:.3g})  "
          f"vs filler: {cell['filler']['delta']:+.1f} (p={cell['filler']['p_perm']:.3g})")


if __name__ == "__main__":
  main()
