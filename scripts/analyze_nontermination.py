"""Reproduces two specific process-effect claims flagged as unsourced in
docs/paper-revision-handoff.md, both on densfull40:

  1. "qwen3-1.7b-think's degree primer produces 137 non-terminating
     generations (outside edge_count) of which only 7 also fail to
     terminate under none" -- i.e. of the paired instances where `degree`
     hits the token cap, how many are NEW failures (none did not cap on
     that same instance) vs. failures that would have happened anyway.

  2. "qwen3-4b-think answers correctly on all 29 of the instances the
     degree primer breaks for qwen3-4b" -- takes the exact instance ids
     qwen3-4b gets wrong under `degree` and checks qwen3-4b-think's
     accuracy on that SAME set, under the SAME primer.

Neither number is looked up from a doc; both are recomputed from runs/ with
the current scorer every time this runs.

  PYTHONPATH=. python scripts/analyze_nontermination.py
"""

import glob
import json

from graphtalk import scoring


def _load(arm, corpus="densfull40"):
  seen, rows = {}, []
  for path in sorted(glob.glob(f"runs/{arm}.{corpus}.shard*.jsonl")):
    with open(path, encoding="utf-8") as fh:
      for line in fh:
        if not line.strip():
          continue
        r = json.loads(line)
        k = (r["instance_id"], r["condition"])
        if k in seen:
          continue
        seen[k] = r
  return seen


def new_failures(arm, condition="degree", control="none", exclude_task=None):
  """(n_capped_under_condition, n_also_capped_under_control) outside exclude_task."""
  rows = _load(arm)
  capped_cond = capped_both = 0
  for (iid, cond), r in rows.items():
    if cond != condition:
      continue
    if exclude_task and r["task"] == exclude_task:
      continue
    if not r.get("hit_cap"):
      continue
    capped_cond += 1
    ctrl = rows.get((iid, control))
    if ctrl is not None and ctrl.get("hit_cap"):
      capped_both += 1
  return capped_cond, capped_both


def nontermination_rate(arm, condition="degree", control="none", exclude_task=None):
  """Paired non-termination rate delta (condition - control), with a sign test."""
  rows = _load(arm)
  cond_cap, ctrl_cap, n = [], [], 0
  for (iid, cond), r in rows.items():
    if cond != condition:
      continue
    if exclude_task and r["task"] == exclude_task:
      continue
    ctrl = rows.get((iid, control))
    if ctrl is None:
      continue
    cond_cap.append(int(bool(r.get("hit_cap"))))
    ctrl_cap.append(int(bool(ctrl.get("hit_cap"))))
    n += 1
  mc = scoring.mcnemar(ctrl_cap, cond_cap)
  rate_cond = sum(cond_cap) / n if n else float("nan")
  rate_ctrl = sum(ctrl_cap) / n if n else float("nan")
  return {
      "n": n, "rate_condition": rate_cond, "rate_control": rate_ctrl,
      "delta_pp": 100 * (rate_cond - rate_ctrl), "p_mcnemar": mc["p_value"],
  }


def recovery_by_thinking(broken_arm, recovering_arm, task, condition="degree"):
  """Instance ids `broken_arm` gets wrong under `condition`; accuracy of
  `recovering_arm` on that exact set, under the same condition.
  """
  broken_rows = _load(broken_arm)
  recovering_rows = _load(recovering_arm)
  wrong_ids = []
  for (iid, cond), r in broken_rows.items():
    if cond != condition or r["task"] != task or r.get("hit_cap"):
      continue
    got = scoring.extract_answer(r["response"], task)
    if scoring.score_one(got, r["gold"], task)["exact"] != 1.0:
      wrong_ids.append(iid)

  correct = 0
  missing = []
  for iid in wrong_ids:
    r = recovering_rows.get((iid, condition))
    if r is None:
      missing.append(iid)
      continue
    if r.get("hit_cap"):
      continue
    got = scoring.extract_answer(r["response"], task)
    correct += scoring.score_one(got, r["gold"], task)["exact"] == 1.0
  return {
      "n_broken": len(wrong_ids), "n_recovered": correct, "n_missing": len(missing),
  }


def main():
  print("=== non-termination: degree vs none, qwen3-1.7b-think (all tasks) ===")
  rate = nontermination_rate("qwen3-1.7b-think")
  print(f"  n={rate['n']}  rate under degree={rate['rate_condition']:.1%}  "
        f"rate under none={rate['rate_control']:.1%}  "
        f"delta={rate['delta_pp']:+.1f}pp  p={rate['p_mcnemar']:.3g}")

  print("\n=== non-termination: degree vs none, qwen3-1.7b-think"
        " (outside edge_count) ===")
  rate = nontermination_rate("qwen3-1.7b-think", exclude_task="edge_count")
  print(f"  n={rate['n']}  rate under degree={rate['rate_condition']:.1%}  "
        f"rate under none={rate['rate_control']:.1%}  "
        f"delta={rate['delta_pp']:+.1f}pp  p={rate['p_mcnemar']:.3g}")

  cond_cap, both_cap = new_failures("qwen3-1.7b-think", exclude_task="edge_count")
  print(f"\n  {cond_cap} non-terminating generations under degree"
        f" (outside edge_count), of which {both_cap} also fail under none"
        f" ({cond_cap - both_cap} new failures attributable to the primer)")

  print("\n=== recovery under thinking: qwen3-4b degree failures,"
        " re-scored for qwen3-4b-think ===")
  rec = recovery_by_thinking("qwen3-4b", "qwen3-4b-think", "node_degree")
  print(f"  qwen3-4b gets {rec['n_broken']} instances wrong under degree/node_degree")
  print(f"  qwen3-4b-think (same primer, same instances): "
        f"{rec['n_recovered']} of {rec['n_broken']} correct"
        f"{f' ({rec['n_missing']} missing from the think run)' if rec['n_missing'] else ''}")


if __name__ == "__main__":
  main()
