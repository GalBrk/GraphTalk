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

  PYTHONPATH=. python superseded/scripts/score_full_density_sweep.py \
      --responses "runs/qwen3-1.7b.densfull40.shard*of25.jsonl" \
      --shortcuts shortcuts.json
"""

import argparse
import collections
import csv
import glob
import json

from graphtalk import scoring
from graphtalk import significance

CONTROL = "none"

# Same two presets `check_significance.py` offers: a fast approximate default
# for routine runs, and a slower one for a number meant to be quoted, via
# --mde.
_MDE_FAST = {"n_replicates": 50, "n_perm": 200, "n_steps": 5}
_MDE_FULL = {"n_replicates": 200, "n_perm": 500, "n_steps": 8}


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


def mde_for_arms(control_hits, treatment_hits, seed, settings) -> dict:
  """MDE for one pooled McNemar row, on the paired hit vectors the test
  itself used.

  A full-task density sweep pair is one graph at one density under one
  task, contributing exactly one row here, never repeated -- so each pair
  is its own cluster, unlike the main sweep's six-tasks-per-graph rows.

  Graded scores are binarized to exact-match first. `connected_nodes` is the
  only task here scored with set-F1, and the MDE simulator is Bernoulli end
  to end (see `minimum_detectable_effect_clustered`, which now refuses a
  non-binary vector rather than returning the floor artifact it used to).
  The threshold is `>= 0.9999` -- "recovered the whole set", the same one
  `superseded/scripts/analyze_primer_survival.py` applies for the same reason -- and it
  makes the MDE a *stricter* question than the McNemar above it, which still
  scores partial credit. The returned number therefore answers "how large an
  effect on fully-correct answers could this cell have seen", not "on F1".
  """
  control_hits = [1.0 if v >= 0.9999 else 0.0 for v in control_hits]
  treatment_hits = [1.0 if v >= 0.9999 else 0.0 for v in treatment_hits]
  cluster_ids = list(range(len(control_hits)))
  return significance.minimum_detectable_effect_clustered(
      control_hits, treatment_hits, cluster_ids, initial_hi=0.05,
      seed=seed, **settings,
  )


def write_csv(path: str, rows: list) -> None:
  """Writes `rows` to `path`, header from the first row's keys.

  Writes nothing (not even a header) for an empty list -- a family that
  reduces to nothing on this data is a valid, silent outcome, not an error
  to paper over with a headerless file.
  """
  if not rows:
    return
  with open(path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--responses", nargs="+", required=True)
  parser.add_argument("--shortcuts", default=None)
  parser.add_argument("--control", default=CONTROL)
  parser.add_argument("--tasks", nargs="+", default=None)
  parser.add_argument("--mde", action="store_true",
                      help="full-precision MDE (200/500/8 replicates/perm/"
                           "steps) instead of the default fast preset "
                           "(50/200/5) -- use for a number meant to be quoted")
  parser.add_argument("--no-mde", action="store_true",
                      help="skip MDE entirely (it only runs on conditions "
                           "that don't survive BH, but the search itself is "
                           "not free)")
  parser.add_argument("--csv", default=None,
                      help="write one row per (task, condition) pooled test "
                           "to this CSV path")
  args = parser.parse_args()
  mde_settings = _MDE_FULL if args.mde else _MDE_FAST

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
  csv_rows = []

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
      rows.append((c, len(ctrl_hits), test, delta, mean_treat - mean_ctrl,
                   ctrl_hits, treat_hits))
    reject = significance.benjamini_hochberg(
        [row[2]["p_value"] for row in rows])
    if reject and not args.no_mde:
      print("  (MDE printed under any row that did not survive BH -- the "
            "smallest true effect, in each direction, this row's data could "
            "reliably have detected; fast preset unless --mde was passed)")
    for (c, n, test, delta, raw_diff, ctrl_hits, treat_hits), keep in zip(
        rows, reject):
      bar_delta = ""
      if bars.get(task) and bars[task].get(c) is not None and bars[task].get(args.control) is not None:
        bar_delta = f"  bar-adj {raw_diff - (bars[task][c] - bars[task][args.control]):+.4f}"
      print(f"  {c:>11} - {args.control}: n={n:>5} win {test['c']:>4} lose {test['b']:>4} "
            f"delta {delta:+.4f} p={test['p_value']:.4f}{'  *sig(BH)' if keep else ''}{bar_delta}")
      mde = None
      if not keep and not args.no_mde:
        mde_seed = f"{task}:{c}:mde"
        mde = mde_for_arms(ctrl_hits, treat_hits, mde_seed, mde_settings)
        print(f"    MDE: delta={mde['delta']} realized={mde['realized_diff']} "
              f"({mde['note'] or 'ok'})  "
              f"delta_negative={mde['delta_negative']} "
              f"realized_negative={mde['realized_diff_negative']} "
              f"({mde['note_negative'] or 'ok'})")
      if args.csv:
        csv_rows.append({
            "task": task, "condition": c, "n": n, "win": test["c"],
            "lose": test["b"], "delta": delta, "p_value": test["p_value"],
            "bh_significant": keep,
            "mde_delta": mde["delta"] if mde else None,
            "mde_realized_diff": mde["realized_diff"] if mde else None,
            "mde_note": mde["note"] if mde else None,
            "mde_delta_negative": mde["delta_negative"] if mde else None,
            "mde_realized_diff_negative":
                mde["realized_diff_negative"] if mde else None,
            "mde_note_negative": mde["note_negative"] if mde else None,
        })

  if args.csv:
    write_csv(args.csv, csv_rows)
    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
  main()
