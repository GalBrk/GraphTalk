"""Does a primer effect survive on both model sizes, one, or neither?

The canonical n=40 experiment (`densfull40` + `densfull40hi`, see
`docs/repo-scope.md`) runs four arms: `qwen3-1.7b`, `qwen3-1.7b-think`,
`qwen3-4b`, `qwen3-4b-think`. `score_full_density_sweep.py` scores one arm at a
time and prints it; nothing joins the arms back together to ask the question the
paper actually asks -- *did this effect replicate across model sizes?* -- so that
join has been redone by hand in several places, each time with a slightly
different rule for what "replicated" means.

This script fixes the rule and writes it down. Four outputs, all in
`analysis/tables/`:

  primer_effects_by_density.csv  every (arm, task, density, condition): `none`
                                 accuracy, condition accuracy, delta, the
                                 McNemar discordant counts b/c, p, BH flag.
                                 The master table -- everything else derives
                                 from it.
  primer_effects_pooled.csv      the same pooled over densities within a design
                                 (`lo` = p 0.10-0.50, `hi` = p 0.65-0.85).
  primer_vs_filler.csv           uncontaminated primers tested against `filler`
                                 rather than `none`, on mid-range cells only.
                                 This is the length-controlled comparison
                                 `docs/primer-effects-and-power.md` says to use
                                 and which no other script computes.
  primer_survival.csv            the verdict per (task, condition): BOTH, ONE,
                                 NEITHER, CONFLICT, or UNTESTABLE, with the
                                 reason.

**Headroom is the precondition for every verdict here.** A cell where `none`
already scores ~1.00 cannot show a primer effect, and reporting its null as a
failure to replicate is wrong. `qwen3-4b-think` has *no* mid-range cell anywhere
in 0.10-0.85; it is an untested arm, not a negative result. So a condition that
is significant on one model and null on another is only counted ONE (a genuine
non-replication) when the other model had headroom to show it; otherwise it is
UNTESTABLE there. `MID_RANGE` sets the band, and the choice is made on `none`
accuracy alone -- never on the treatment -- so it cannot select for an effect.

`hit_cap` rows are dropped rather than scored zero, matching
`score_full_density_sweep.py`. That choice changes the verdict on `cycle_check`
and `edge_count`; see `docs/full-task-density-sweep.md`'s Caveats.

`connected_nodes` is binarized as exact match, not F1. Its F1 sits at 96-100%
under every condition and hides everything the task has to show.

  PYTHONPATH=. python scripts/analyze_primer_survival.py

Writes a provenance manifest next to the tables (`primer_survival_manifest.json`:
git sha, input files, row counts, the thresholds above) so a number in these
tables can be traced back to the run files it came from without re-running
anything.
"""

import argparse
import collections
import csv
import glob
import json
import os
import subprocess
import sys

from graphtalk import scoring
from graphtalk import significance

CONTROL = "none"
LENGTH_CONTROL = "filler"

# The seven primer conditions, minus the control itself.
CONDITIONS = ["components", "clustering", "rwse", "degree", "filler", "all"]

# Conditions whose primer does not state or trivially imply the answer to any
# task. `degree`/`all` state the `node_degree` answer verbatim and sum to the
# `edge_count` answer; `filler` is the length control, not a treatment.
UNCONTAMINATED = ["components", "clustering", "rwse"]

# A cell can only show an effect if the control leaves room to move. Chosen on
# `none` accuracy alone, independent of any treatment.
MID_RANGE = (0.30, 0.90)

# The two density bands, which are separate prompt files and separate runs.
DESIGNS = {"lo": "densfull40", "hi": "densfull40hi"}

ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"]

# Which arms are compared to each other for a replication verdict. Plain and
# thinking are different models for this purpose, not two readings of one.
FAMILIES = {"plain": ("qwen3-1.7b", "qwen3-4b"),
            "think": ("qwen3-1.7b-think", "qwen3-4b-think")}

MIN_CELL = 50  # fewer paired rows than this and the cell is not reported


def density_of(instance_id):
  """The pinned ER density `build_size_sweep.py --densities` wrote into the id.

  Same parse as `score_full_density_sweep.py` -- response rows carry no
  `density_class` field, only the id encodes it.
  """
  for part in instance_id.split("/"):
    if len(part) > 1 and part[0] == "p":
      try:
        return float(part[1:])
      except ValueError:
        continue
  return None


def binarize(task, value):
  """`connected_nodes` scores as F1; binarize it to exact match.

  Everything else is already 0/1 under its own primary metric.
  """
  if task == "connected_nodes":
    return 1.0 if value >= 0.9999 else 0.0
  return value


def load(runs_dir):
  """(arm, design, task, condition) -> {instance_id: score}, capped rows dropped.

  Scores straight from the run files rather than from
  `analysis/*.densfull40.rows.csv`, which only exist for the `lo` design and
  were written by a different scorer -- reading both would mean two scoring
  paths behind one table.
  """
  hits = collections.defaultdict(dict)
  inputs, kept, capped = [], 0, 0
  for arm in ARMS:
    for design, tag in DESIGNS.items():
      paths = sorted(glob.glob(os.path.join(runs_dir, f"{arm}.{tag}.shard*.jsonl")))
      inputs.extend(paths)
      for path in paths:
        with open(path, encoding="utf-8") as handle:
          for line in handle:
            if not line.strip():
              continue
            row = json.loads(line)
            if row.get("hit_cap"):
              capped += 1
              continue
            result = scoring.score_one(
                scoring.extract_answer(row["response"], row["task"]),
                row["gold"], row["task"])
            # A duplicate (instance_id, condition) from a resume overlap
            # overwrites rather than double-counting; the rows are identical.
            hits[(arm, design, row["task"], row["condition"])][row["instance_id"]] = (
                result["primary"])
            kept += 1
  return hits, inputs, kept, capped


def compare(hits, arm, design, task, control, treatment, density=None):
  """One paired McNemar of `treatment` against `control` on shared instances."""
  ctl = hits.get((arm, design, task, control), {})
  trt = hits.get((arm, design, task, treatment), {})
  ids = sorted(set(ctl) & set(trt))
  if density is not None:
    ids = [i for i in ids if density_of(i) == density]
  if len(ids) < MIN_CELL:
    return None
  control_hits = [binarize(task, ctl[i]) for i in ids]
  treatment_hits = [binarize(task, trt[i]) for i in ids]
  test = scoring.mcnemar(control_hits, treatment_hits)
  n = len(ids)
  return {
      "arm": arm, "design": design, "task": task, "condition": treatment,
      "control": control, "n": n,
      "control_acc": round(sum(control_hits) / n, 4),
      "condition_acc": round(sum(treatment_hits) / n, 4),
      "delta_pp": round(100 * (sum(treatment_hits) - sum(control_hits)) / n, 2),
      "b_helped_to_hurt": test["b"], "c_hurt_to_helped": test["c"],
      "p_value": test["p_value"],
  }


def with_bh(rows, key):
  """Benjamini-Hochberg within each `key(row)` family, in place."""
  families = collections.defaultdict(list)
  for i, row in enumerate(rows):
    families[key(row)].append(i)
  for indices in families.values():
    flags = significance.benjamini_hochberg([rows[i]["p_value"] for i in indices])
    for i, flag in zip(indices, flags):
      rows[i]["bh_significant"] = bool(flag)
  return rows


def has_headroom(rows, arm, task, condition):
  """Did this arm ever have room to show an effect on this task?

  True when any density's `none` accuracy sits inside MID_RANGE. Read off the
  control column, so it says nothing about whether a primer helped.
  """
  return any(MID_RANGE[0] <= r["control_acc"] <= MID_RANGE[1]
             for r in rows
             if r["arm"] == arm and r["task"] == task and r["condition"] == condition)


def verdict(per_density, pooled):
  """Per (family, task, condition): did the effect replicate across model sizes?

  BOTH       significant and same sign on both sizes
  CONFLICT   significant on both, opposite signs -- worse than a non-replication
  ONE        significant on one, null on the other *which had headroom*
  UNTESTABLE significant on one, the other never had headroom to show it
  NEITHER    significant on neither, both had headroom
  NO_HEADROOM  neither size could be tested at all
  """
  out = []
  by_key = {(r["arm"], r["design"], r["task"], r["condition"]): r for r in pooled}
  tasks = sorted({r["task"] for r in pooled})
  for family, (small, large) in FAMILIES.items():
    for task in tasks:
      for condition in CONDITIONS:
        for design in DESIGNS:
          s = by_key.get((small, design, task, condition))
          l = by_key.get((large, design, task, condition))
          if s is None and l is None:
            continue
          room = {a: has_headroom(per_density, a, task, condition)
                  for a in (small, large)}
          sig = {a: bool(r and r["bh_significant"])
                 for a, r in ((small, s), (large, l))}
          delta = {a: (r["delta_pp"] if r else None)
                   for a, r in ((small, s), (large, l))}
          if sig[small] and sig[large]:
            same = (delta[small] > 0) == (delta[large] > 0)
            call, why = (("BOTH", "significant and same sign on both sizes")
                         if same else
                         ("CONFLICT", "significant on both sizes, opposite signs"))
          elif sig[small] or sig[large]:
            hit = small if sig[small] else large
            miss = large if sig[small] else small
            if room[miss]:
              call, why = "ONE", f"significant on {hit}; {miss} had headroom and did not show it"
            else:
              call, why = "UNTESTABLE", f"significant on {hit}; {miss} never left ceiling/floor"
          elif room[small] or room[large]:
            call, why = "NEITHER", "headroom on at least one size, significant on neither"
          else:
            call, why = "NO_HEADROOM", "no mid-range cell on either size"
          out.append({
              "family": family, "design": design, "task": task,
              "condition": condition, "verdict": call, "reason": why,
              "contaminated": condition not in UNCONTAMINATED,
              "small_arm": small, "small_delta_pp": delta[small],
              "small_significant": sig[small], "small_had_headroom": room[small],
              "large_arm": large, "large_delta_pp": delta[large],
              "large_significant": sig[large], "large_had_headroom": room[large],
          })
  return out


def write_csv(path, rows):
  if not rows:
    print(f"  (nothing to write to {path})")
    return
  with open(path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"  wrote {path} ({len(rows)} rows)")


def git_sha():
  try:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
  except Exception:
    return None


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--runs", default="runs")
  parser.add_argument("--out-dir", default="analysis/tables")
  args = parser.parse_args(argv)
  os.makedirs(args.out_dir, exist_ok=True)

  print("loading and scoring run files ...")
  hits, inputs, kept, capped = load(args.runs)
  print(f"  {kept} rows scored, {capped} hit_cap rows dropped, "
        f"{len(inputs)} shard files")

  densities = sorted({d for m in hits.values() for d in map(density_of, m)
                      if d is not None})
  tasks = sorted({t for (_, _, t, _) in hits})

  per_density, pooled, vs_filler = [], [], []
  for arm in ARMS:
    for design in DESIGNS:
      for task in tasks:
        for condition in CONDITIONS:
          row = compare(hits, arm, design, task, CONTROL, condition)
          if row:
            pooled.append(row)
          for density in densities:
            row = compare(hits, arm, design, task, CONTROL, condition, density)
            if row:
              row["density"] = density
              per_density.append(row)

  with_bh(per_density, lambda r: (r["arm"], r["design"], r["task"], r.get("density")))
  with_bh(pooled, lambda r: (r["arm"], r["design"], r["task"]))

  # The length-controlled comparison: clean primers against `filler`, on cells
  # where `none` left room. Selection is on `none`, never on the treatment.
  control_acc = {(r["arm"], r["design"], r["task"], r["density"]): r["control_acc"]
                 for r in per_density}
  for arm in ARMS:
    for design in DESIGNS:
      for task in tasks:
        for condition in UNCONTAMINATED:
          for density in densities:
            base = control_acc.get((arm, design, task, density))
            if base is None or not MID_RANGE[0] <= base <= MID_RANGE[1]:
              continue
            row = compare(hits, arm, design, task, LENGTH_CONTROL, condition, density)
            if row:
              row["density"] = density
              row["none_acc"] = base
              vs_filler.append(row)
  with_bh(vs_filler, lambda r: "all")  # one family: this is a single question

  print("writing tables ...")
  write_csv(os.path.join(args.out_dir, "primer_effects_by_density.csv"), per_density)
  write_csv(os.path.join(args.out_dir, "primer_effects_pooled.csv"), pooled)
  write_csv(os.path.join(args.out_dir, "primer_vs_filler.csv"), vs_filler)
  survival = verdict(per_density, pooled)
  write_csv(os.path.join(args.out_dir, "primer_survival.csv"), survival)

  manifest = {
      "git_sha": git_sha(),
      "script": "scripts/analyze_primer_survival.py",
      "arms": ARMS, "designs": DESIGNS, "densities": densities, "tasks": tasks,
      "mid_range": list(MID_RANGE), "min_cell": MIN_CELL,
      "capped_rows": "dropped", "connected_nodes_metric": "exact match (F1 binarized at 1.0)",
      "rows_scored": kept, "rows_dropped_hit_cap": capped,
      "input_files": sorted(inputs),
  }
  path = os.path.join(args.out_dir, "primer_survival_manifest.json")
  with open(path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=1)
  print(f"  wrote {path}")

  counts = collections.Counter((r["family"], r["verdict"]) for r in survival)
  print("\nverdicts (all conditions, both density bands):")
  for (family, call), n in sorted(counts.items()):
    print(f"  {family:6} {call:12} {n:>4}")
  clean = collections.Counter(
      (r["family"], r["verdict"]) for r in survival if not r["contaminated"])
  print("\nverdicts, uncontaminated primers only:")
  for (family, call), n in sorted(clean.items()):
    print(f"  {family:6} {call:12} {n:>4}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
