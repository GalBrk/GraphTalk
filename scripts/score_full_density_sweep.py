"""Score a fixed-size density sweep that spans multiple tasks.

`score_density_sweep.py` groups by (density, condition) only, which is correct
for a single-task sweep (its original use, `node_degree` alone) but silently
pools every task's scores into one number for a sweep like this one --
`prompts.densfull40.jsonl`, all 6 tasks x all 7 conditions x 4 densities.
Averaging `node_count` exact-match with `connected_nodes` F1 into one cell
would answer a question nobody asked. This script adds `task` as a third
grouping key and otherwise keeps the same rules: `hit_cap` rows dropped (not
scored zero), effects read against `shortcuts.json`'s bar rather than against
zero, and per-task Benjamini-Hochberg across conditions rather than one
family across all 42 (task, condition) cells.

  PYTHONPATH=. python scripts/score_full_density_sweep.py \
      --responses "runs/qwen3-1.7b.densfull40.shard*of25.jsonl" \
      --shortcuts shortcuts.json
"""

import argparse
import collections
import glob
import json

from graphtalk import scoring
from graphtalk import significance

CONTROL = "none"


def density_of(instance_id: str) -> float | None:
  for part in instance_id.split("/"):
    if len(part) > 1 and part[0] == "p":
      try:
        return float(part[1:])
      except ValueError:
        continue
  return None


def load(patterns) -> list[dict]:
  records = []
  seen = set()
  for pattern in patterns:
    for path in sorted(glob.glob(pattern)) or [pattern]:
      with open(path) as handle:
        for line in handle:
          if not line.strip():
            continue
          r = json.loads(line)
          key = (r["instance_id"], r["condition"], r["style"])
          if key in seen:
            continue  # a handful of duplicate rows survive preemption/resume
          seen.add(key)
          records.append(r)
  return records


def summarize(records):
  cells = collections.defaultdict(
      lambda: {"kept": 0, "capped": 0, "total": 0.0, "parsed": 0,
               "abs_error": 0.0, "n_error": 0}
  )
  paired = collections.defaultdict(dict)  # (task, density, instance_id) -> cond -> score
  golds = collections.defaultdict(list)   # (task, density) -> golds
  for r in records:
    density = density_of(r["instance_id"])
    task = r["task"]
    cell = cells[(task, density, r["condition"])]
    if r.get("hit_cap"):
      cell["capped"] += 1
      continue
    result = scoring.score_one(
        scoring.extract_answer(r["response"], task), r["gold"], task)
    cell["kept"] += 1
    cell["total"] += result["primary"]
    cell["parsed"] += int(result["parsed"])
    if result["absolute_error"] is not None:
      cell["abs_error"] += result["absolute_error"]
      cell["n_error"] += 1
    paired[(task, density, r["instance_id"])][r["condition"]] = result["primary"]
    golds[(task, density)].append(r["gold"])
  return cells, paired, golds


def blind_bar(golds):
  counts = collections.Counter(golds)
  answer, hits = counts.most_common(1)[0]
  return answer, hits / len(golds)


def paired_arms(paired, task, condition, control=CONTROL, density=None):
  control_hits, treatment_hits = [], []
  for (t, level, _), by_condition in paired.items():
    if t != task:
      continue
    if density is not None and level != density:
      continue
    if control in by_condition and condition in by_condition:
      control_hits.append(by_condition[control])
      treatment_hits.append(by_condition[condition])
  return control_hits, treatment_hits


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--responses", nargs="+", required=True)
  parser.add_argument("--shortcuts", default=None)
  parser.add_argument("--control", default=CONTROL)
  parser.add_argument("--tasks", nargs="+", default=None)
  args = parser.parse_args()

  bars = {}
  if args.shortcuts:
    with open(args.shortcuts) as handle:
      raw = json.load(handle)
    # shortcuts.json is flat: {"task/condition": bar}
    for key, bar in raw.items():
      task, _, condition = key.partition("/")
      bars.setdefault(task, {})[condition] = bar

  records = load(args.responses)
  print(f"{len(records)} unique rows loaded from {len(args.responses)} pattern(s)")
  cells, paired, golds = summarize(records)

  tasks = args.tasks or sorted({t for t, _, _ in cells})
  densities = sorted({d for _, d, _ in cells})
  conditions = sorted({c for _, _, c in cells})

  for task in tasks:
    print(f"\n{'='*90}\nTASK: {task}\n{'='*90}")
    header = "  p      " + "".join(f"{c:>12}" for c in conditions)

    print("blind bar (best constant answer):")
    for d in densities:
      key = (task, d)
      if key not in golds:
        continue
      answer, bar = blind_bar(golds[key])
      print(f"  p={d!s:<6} modal {answer!r:>6}  bar {bar:.3f}  (n={len(golds[key])})")

    print("\nmean score by density x condition (hit_cap dropped):")
    print(header)
    for d in densities:
      line = f"  {d!s:<7}"
      for c in conditions:
        cell = cells.get((task, d, c))
        v = cell["total"] / cell["kept"] if cell and cell["kept"] else float("nan")
        line += f"{v:>12.3f}"
      print(line)

    if bars.get(task):
      print("\nshortcut bar (primer-only solver, from shortcuts.json):")
      line = "  all p  "
      for c in conditions:
        b = bars[task].get(c)
        line += f"{b:>12.3f}" if b is not None else f"{'--':>12}"
      print(line)

    print("\ncapped rows dropped by density x condition:")
    print(header)
    for d in densities:
      line = f"  {d!s:<7}"
      for c in conditions:
        cell = cells.get((task, d, c))
        line += f"{(cell['capped'] if cell else 0):>12}"
      print(line)

    print("\nunparsable rows (scored 0, not dropped) by density x condition:")
    print(header)
    for d in densities:
      line = f"  {d!s:<7}"
      for c in conditions:
        cell = cells.get((task, d, c))
        v = (cell["kept"] - cell["parsed"]) if cell else 0
        line += f"{v:>12}"
      print(line)

    print(f"\npooled across densities, paired vs {args.control!r} (exact McNemar):")
    rows = []
    for c in conditions:
      if c == args.control:
        continue
      ctrl_hits, treat_hits = paired_arms(paired, task, c, args.control)
      if not ctrl_hits:
        continue
      test = scoring.mcnemar(ctrl_hits, treat_hits)
      delta = (test["c"] - test["b"]) / len(ctrl_hits)
      mean_ctrl = sum(ctrl_hits) / len(ctrl_hits)
      mean_treat = sum(treat_hits) / len(treat_hits)
      rows.append((c, len(ctrl_hits), test, delta, mean_treat - mean_ctrl))
    reject = significance.benjamini_hochberg([t["p_value"] for *_, t, _, _ in rows])
    for (c, n, test, delta, raw_diff), keep in zip(rows, reject):
      bar_delta = ""
      if bars.get(task) and bars[task].get(c) is not None and bars[task].get(args.control) is not None:
        bar_delta = f"  bar-adj {raw_diff - (bars[task][c] - bars[task][args.control]):+.4f}"
      print(f"  {c:>11} - {args.control}: n={n:>5} win {test['c']:>4} lose {test['b']:>4} "
            f"delta {delta:+.4f} p={test['p_value']:.4f}{'  *sig(BH)' if keep else ''}{bar_delta}")


if __name__ == "__main__":
  main()
