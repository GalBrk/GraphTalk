"""The "Smart Hybrid" size/density grid scorer -- two stages, no GPU needed.

`--stage scout`: reads the cheap, plain-arm, `none`-only pilot
(`cluster/run_density_size_sweep.sh --stage scout`) across the full
node-count x density x task grid, and applies `graphtalk.range_search
.scout_decision`'s relaxed screen (drop only >98% ceiling, >30% truncation,
or the standard floor rule). Writes three files, never fewer:

  * `analysis/density_size_grid.scout.csv` -- every cell, survive or drop.
  * `analysis/density_size_grid.archive_filtered.csv` -- the dropped subset
    only, so a cell can be handed back in with `--force-include` on the
    `full` stage without re-deriving which ones were cut.
  * `analysis/density_size_grid.survivors.json` -- the `(task, nodes,
    density)` keys that survived, for `cluster/run_density_size_sweep.sh
    --stage full` to build prompts for.

`--stage full`: reads the deep-dive reporter (both arms, all 7 conditions,
100 graphs/cell) for whichever cells were actually built and generated
(scout survivors plus anything force-included), and reports every check on
both arms for every row -- `graphtalk.range_search.classify_cell` labels a
row, it never drops one. Writes `analysis/density_size_grid.full.csv` (one
row per (task, condition, nodes, density, arm), every generated row scored
and kept) and a narrative `docs/density-size-range-search.md` that presents
the grid without naming a "recommended range" -- that call is the user's.

  PYTHONPATH=. .venv/bin/python scripts/score_density_size_grid.py \\
      --stage scout --sizes 10 20 40 60 80 \\
      --densities 0.10 0.20 0.35 0.50 0.65 0.75 0.85

  PYTHONPATH=. .venv/bin/python scripts/score_density_size_grid.py \\
      --stage full --sizes 10 20 40 60 80 \\
      --densities 0.10 0.20 0.35 0.50 0.65 0.75 0.85 --shortcuts shortcuts.json
"""

import argparse
import collections
import glob
import json
import math

import pandas as pd

from graphtalk import analysis
from graphtalk import range_search
from graphtalk import scoring
from graphtalk import significance
from scripts import check_significance as cs
from scripts import score_sweep

DEFAULT_SIZES = (10, 20, 40, 60, 80)
DEFAULT_DENSITIES = (0.10, 0.20, 0.35, 0.50, 0.65, 0.75, 0.85)
DEFAULT_TASKS = tuple(scoring.TASKS)


def _fmt_density(p: float) -> str:
  return f"{p:.2f}"


def verify_density(prompts_path: str, expected_density: float,
                    tolerance: float = 0.05) -> float:
  """Mean empirical ER edge probability over one prompts file's
  `none`-condition rows for its (alphabetically) first task, checked
  against `expected_density`.

  Density can't be checked for exact equality the way node count can:
  `--er-min-sparsity P --er-max-sparsity P` pins the edge *probability*,
  not the edge *count*, which is stochastic per graph. Every task in one
  `build_diverse` call shares the same underlying pool
  (`build_prompts.build_diverse`'s own docstring), so any one task's
  `none` rows are enough to check the whole file's pool.
  """
  with open(prompts_path) as handle:
    records = [json.loads(line) for line in handle if line.strip()]
  if not records:
    return expected_density
  first_task = sorted({r["task"] for r in records})[0]
  rows = [r for r in records
          if r["condition"] == "none" and r["task"] == first_task]
  densities = [
      2 * r["edges"] / (r["nodes"] * (r["nodes"] - 1))
      for r in rows if r["nodes"] > 1
  ]
  if not densities:
    return expected_density
  mean_density = sum(densities) / len(densities)
  if abs(mean_density - expected_density) > tolerance:
    raise ValueError(
        f"{prompts_path}: mean empirical density {mean_density:.3f} differs "
        f"from requested {expected_density:.3f} by more than {tolerance}"
    )
  return mean_density


def load_grid_cell_frame(
    model: str, n: int, p: float, responses_pattern: str, prompts_pattern: str,
    shortcuts_by_cell: dict, density_tolerance: float,
) -> pd.DataFrame | None:
  """Scored frame for one (model, n, density), tagged with `nodes`/
  `density` columns, or `None` if no response file exists yet -- absence
  is not an error here, it just means that cell hasn't been generated
  (yet, or ever, for `--stage full` on a task the scout dropped).
  """
  density_str = _fmt_density(p)
  paths = sorted(
      path for path in glob.glob(
          responses_pattern.format(model=model, size=n, density=density_str)
      )
      if not analysis.is_excluded(path)
  )
  if not paths:
    return None
  records = score_sweep.load(paths)
  score_sweep.desubstitute_named_responses(records)
  score_sweep.score_records(records)
  frame = analysis.build_frame(records, set(), shortcuts_by_cell)
  if frame.empty:
    return None
  prompts_paths = sorted(
      glob.glob(prompts_pattern.format(size=n, density=density_str))
  )
  for prompts_path in prompts_paths:
    verify_density(prompts_path, p, tolerance=density_tolerance)
  frame["nodes"] = n
  frame["density"] = p
  return frame


def load_all_frames(
    models, sizes, densities, responses_pattern: str, prompts_pattern: str,
    shortcuts_by_cell: dict, density_tolerance: float,
) -> dict:
  """`{(model, n, p): frame}` for every combination with a response file."""
  frames = {}
  for model in models:
    for n in sizes:
      for p in densities:
        frame = load_grid_cell_frame(
            model, n, p, responses_pattern, prompts_pattern,
            shortcuts_by_cell, density_tolerance,
        )
        if frame is not None:
          frames[(model, n, p)] = frame
  return frames


# --- Stage 2: scout scoring ---------------------------------------------------


def scout_cells(
    frame_by_size_density: dict, tasks, ceiling: float, floor_margin: float,
    truncation_drop: float,
) -> list[dict]:
  """One row per (task, nodes, density) -- `graphtalk.range_search
  .scout_decision`'s relaxed screen, applied to the plain-arm `none`-only
  pilot. `decision`/`zone`/`reason` are the columns a caller filters on;
  every cell that was actually scored appears here, survive or drop.
  """
  rows = []
  for (n, p), frame in sorted(frame_by_size_density.items()):
    for task in tasks:
      task_frame = frame[(frame["task"] == task) & (frame["condition"] == "none")]
      if task_frame.empty:
        continue
      none_accuracy = task_frame["primary"].mean()
      _, majority_baseline = scoring.majority_baseline(list(task_frame["gold"]))
      truncation_rate = task_frame["non_terminating"].mean()
      decision = range_search.scout_decision(
          none_accuracy, majority_baseline, truncation_rate,
          ceiling=ceiling, floor_margin=floor_margin,
          truncation_drop=truncation_drop,
      )
      rows.append({
          "task": task, "nodes": n, "density": p,
          "n_graphs": len(task_frame),
          "none_accuracy": none_accuracy,
          "majority_baseline": majority_baseline,
          "truncation_rate": truncation_rate,
          "decision": decision["decision"],
          "zone": decision["zone"],
          "reason": decision["reason"],
      })
  return rows


# --- Stage 3: full reporter ---------------------------------------------------


def full_cells(
    frame_by_arm_size_density: dict, tasks, shortcuts_by_cell: dict,
    ceiling: float, floor_margin: float, shortcut_bar: float,
    truncation_threshold: float, min_effect_of_interest: float,
    power_target: float, n_perm: int, n_boot: int, n_replicates: int,
    seed: int,
) -> list[dict]:
  """One row per (task, condition, nodes, density, arm) -- every generated
  row is scored and kept. `none` rows carry zone/truncation only (there is
  no primer to compare against itself); non-`none` rows additionally carry
  delta/CI/shortcut/required-N and `arm_divergence`, computed once per
  (task, condition, nodes, density) and copied onto both arms' rows for
  that cell so a reader sees the disagreement from either row.
  """
  by_np = collections.defaultdict(dict)
  for (arm, n, p), frame in frame_by_arm_size_density.items():
    by_np[(n, p)][arm] = frame

  rows = []
  for (n, p), arm_frames in sorted(by_np.items()):
    for task in tasks:
      none_stats = {}
      for arm, frame in arm_frames.items():
        task_frame = frame[frame["task"] == task]
        none_rows = task_frame[task_frame["condition"] == cs.CONTROL]
        if none_rows.empty:
          continue
        _, majority_baseline = scoring.majority_baseline(list(none_rows["gold"]))
        none_stats[arm] = {
            "none_accuracy": none_rows["primary"].mean(),
            "majority_baseline": majority_baseline,
            "truncation_rate": none_rows["non_terminating"].mean(),
            "n_graphs": len(none_rows),
        }
      if not none_stats:
        continue

      for arm, stats in none_stats.items():
        zone = range_search.zone_decision(
            stats["none_accuracy"], stats["majority_baseline"],
            ceiling, floor_margin,
        )
        rows.append({
            "task": task, "condition": cs.CONTROL, "nodes": n, "density": p,
            "arm": arm, "n_graphs": stats["n_graphs"],
            "accuracy": stats["none_accuracy"],
            "majority_baseline": stats["majority_baseline"],
            "truncation_rate": stats["truncation_rate"],
            "zone": zone, "delta": None, "ci_low": None, "ci_high": None,
            "shortcut_score": None, "shortcut_clean": None,
            "truncation_clean": None, "passes": None,
            "required_n_closed_form": None, "required_n_clustered": None,
            "achieved_power": None, "pilot_n_clusters": None,
            "arm_divergence": None, "arm_divergence_reason": None,
        })

      conditions = set()
      for arm, frame in arm_frames.items():
        task_frame = frame[frame["task"] == task]
        conditions.update(
            c for c in task_frame["condition"].unique() if c != cs.CONTROL
        )

      for condition in sorted(conditions):
        classify_by_arm = {}
        row_by_arm = {}
        for arm, stats in none_stats.items():
          frame = arm_frames[arm]
          task_frame = frame[frame["task"] == task]
          control, treatment, cluster_ids = cs._paired_values(
              task_frame, condition, "primary"
          )
          shortcut_score = shortcuts_by_cell.get((task, condition))

          if not control:
            accuracy = truncation_rate = delta = ci_low = ci_high = None
            required_closed = required_clustered = None
            achieved_power = pilot_n_clusters = None
            n_graphs = 0
          else:
            cond_rows = task_frame[task_frame["condition"] == condition]
            truncation_rate = cond_rows["non_terminating"].mean()
            accuracy = sum(treatment) / len(treatment)
            delta = sum(t - c for c, t in zip(control, treatment)) / len(control)
            boot = significance.cluster_bootstrap_ci_clustered(
                control, treatment, cluster_ids, n_boot=n_boot, seed=seed,
            )
            ci_low, ci_high = boot["ci_low"], boot["ci_high"]
            target_delta = (
                delta if abs(delta) >= min_effect_of_interest
                else math.copysign(min_effect_of_interest, delta or 1.0)
            )
            required_closed = significance.required_n_closed_form(target_delta)
            clustered = significance.required_sample_size_clustered(
                control, treatment, cluster_ids, target_delta,
                power_target=power_target, n_replicates=n_replicates,
                n_perm=n_perm, seed=seed,
            )
            required_clustered = clustered["required_n_clusters"]
            achieved_power = clustered["achieved_power"]
            pilot_n_clusters = clustered["pilot_n_clusters"]
            n_graphs = len(control)

          classification = range_search.classify_cell(
              stats["none_accuracy"], stats["majority_baseline"],
              shortcut_score, truncation_rate, ceiling=ceiling,
              floor_margin=floor_margin, shortcut_bar=shortcut_bar,
              truncation_threshold=truncation_threshold,
          )
          classify_by_arm[arm] = classification
          row_by_arm[arm] = {
              "task": task, "condition": condition, "nodes": n, "density": p,
              "arm": arm, "n_graphs": n_graphs, "accuracy": accuracy,
              "majority_baseline": stats["majority_baseline"],
              "truncation_rate": truncation_rate,
              "zone": classification["zone"],
              "delta": delta, "ci_low": ci_low, "ci_high": ci_high,
              "shortcut_score": shortcut_score,
              "shortcut_clean": classification["shortcut_clean"],
              "truncation_clean": classification["truncation_clean"],
              "passes": classification["passes"],
              "required_n_closed_form": required_closed,
              "required_n_clustered": required_clustered,
              "achieved_power": achieved_power,
              "pilot_n_clusters": pilot_n_clusters,
          }

        diverges, reason = None, None
        arms_present = list(classify_by_arm)
        plain_arms = [a for a in arms_present if not a.endswith("-think")]
        think_arms = [a for a in arms_present if a.endswith("-think")]
        if plain_arms and think_arms:
          divergence = range_search.arm_divergence(
              classify_by_arm[plain_arms[0]], classify_by_arm[think_arms[0]]
          )
          diverges, reason = divergence["diverges"], divergence["reason"]
        for row in row_by_arm.values():
          row["arm_divergence"] = diverges
          row["arm_divergence_reason"] = reason
          rows.append(row)
  return rows


def write_report(result: pd.DataFrame, path: str) -> None:
  """Narrative summary alongside the full CSV -- per task, how many rows
  pass all three checks and how many show arm divergence, plus the
  passing/diverging rows themselves. Deliberately does not name a
  "recommended range": that decision belongs to whoever reads this.
  """
  lines = [
      "# Size/density grid -- full reporter\n\n",
      "Every `(task, condition, nodes, density, arm)` cell generated in "
      "Stage 3 is scored here -- nothing is excluded. `passes` means the "
      "cell cleared all three checks (informative zone, >=10pp shortcut "
      "headroom, <10% truncation) on that arm alone; it is a label, not a "
      "recommendation. The full numbers are in the CSV next to this file; "
      "this document summarizes them per task so a human can decide the "
      "range from here.\n\n",
      "## Method\n\n",
      "- none-baseline zone: ceiling >=90%, floor <= majority baseline + "
      "2pp, else informative.\n",
      "- shortcut headroom: `1 - shortcut_score` >= 10pp.\n",
      "- truncation: combined `hit_cap`/`overflow` rate < 10%, per arm.\n",
      "- `arm_divergence`: the two arms disagree on `passes`, or on any "
      "individual check, for the same (task, condition, nodes, density) "
      "cell.\n",
      "- `required_n_closed_form`/`required_n_clustered`: how many paired "
      "graphs a follow-up would need to confirm this cell's own observed "
      "effect (floored at a 3pp minimum effect of interest) at 80% "
      "power -- informational, not a bar this report filters on.\n",
  ]

  for task in sorted(result["task"].dropna().unique()):
    task_rows = result[result["task"] == task]
    non_none = task_rows[task_rows["condition"] != cs.CONTROL]
    passing = non_none[non_none["passes"] == True]  # noqa: E712
    diverging = non_none[non_none["arm_divergence"] == True]  # noqa: E712
    lines.append(f"\n## {task}\n\n")
    lines.append(
        f"{len(non_none)} (condition, nodes, density, arm) rows checked, "
        f"{len(passing)} pass all three checks, {len(diverging)} show arm "
        f"divergence.\n"
    )
    if not passing.empty:
      lines.append(
          "\n### Passing cells\n\n"
          "| condition | n | density | arm | accuracy | delta | zone | "
          "required N (closed form / clustered) |\n"
          "|---|---|---|---|---|---|---|---|\n"
      )
      for _, row in passing.sort_values(
          ["nodes", "density", "condition", "arm"]
      ).iterrows():
        lines.append(
            f"| {row['condition']} | {row['nodes']} | {row['density']:.2f} "
            f"| {row['arm']} | {row['accuracy']:.1%} | {row['delta']:+.1%} "
            f"| {row['zone']} | {row['required_n_closed_form']} / "
            f"{row['required_n_clustered']} |\n"
        )
    if not diverging.empty:
      lines.append("\n### Arm divergence\n\n| condition | n | density | reason |\n|---|---|---|---|\n")
      seen = set()
      for _, row in diverging.sort_values(["nodes", "density", "condition"]).iterrows():
        key = (row["condition"], row["nodes"], row["density"])
        if key in seen:
          continue
        seen.add(key)
        lines.append(
            f"| {row['condition']} | {row['nodes']} | {row['density']:.2f} "
            f"| {row['arm_divergence_reason']} |\n"
        )

  with open(path, "w") as handle:
    handle.write("".join(lines))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage", required=True, choices=["scout", "full"])
  parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
  parser.add_argument("--densities", type=float, nargs="+",
                      default=list(DEFAULT_DENSITIES))
  parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS),
                      choices=scoring.ALL_TASKS)
  parser.add_argument("--model-family", default="qwen3-1.7b")
  parser.add_argument("--responses-pattern", default=None,
                      help="{model}/{size}/{density} are filled in; default "
                           "depends on --stage (see module docstring)")
  parser.add_argument("--prompts-pattern", default=None,
                      help="{size}/{density} are filled in; default depends "
                           "on --stage")
  parser.add_argument("--density-tolerance", type=float, default=0.05)
  parser.add_argument("--shortcuts", default="shortcuts.json",
                      help="--stage full only; pass '' to skip shortcut "
                           "enrichment")
  parser.add_argument("--ceiling", type=float, default=None,
                      help="default 0.98 for --stage scout, 0.90 for "
                           "--stage full")
  parser.add_argument("--floor-margin", type=float, default=0.02)
  parser.add_argument("--truncation-drop", type=float, default=0.30,
                      help="--stage scout only")
  parser.add_argument("--shortcut-bar", type=float, default=0.10,
                      help="--stage full only")
  parser.add_argument("--truncation-threshold", type=float, default=0.10,
                      help="--stage full only")
  parser.add_argument("--min-effect-of-interest", type=float, default=0.03,
                      help="--stage full only")
  parser.add_argument("--power-target", type=float, default=0.8,
                      help="--stage full only")
  parser.add_argument("--n-perm", type=int, default=500)
  parser.add_argument("--n-boot", type=int, default=10_000)
  parser.add_argument("--n-replicates", type=int, default=200)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--out", default=None)
  parser.add_argument("--archive-out", default=None, help="--stage scout only")
  parser.add_argument("--survivors-out", default=None, help="--stage scout only")
  parser.add_argument("--report", default=None, help="--stage full only")
  args = parser.parse_args()

  ceiling = args.ceiling
  if ceiling is None:
    ceiling = 0.98 if args.stage == "scout" else 0.90

  if args.stage == "scout":
    responses_pattern = (
        args.responses_pattern
        or "runs/{model}.grid_scout_n{size}_p{density}.jsonl"
    )
    prompts_pattern = (
        args.prompts_pattern or "prompts_grid_scout_n{size}_p{density}.jsonl"
    )
    frames = load_all_frames(
        [args.model_family], args.sizes, args.densities, responses_pattern,
        prompts_pattern, {}, args.density_tolerance,
    )
    if not frames:
      raise SystemExit("no scout response files found")
    by_np = {(n, p): frame for (_model, n, p), frame in frames.items()}
    rows = scout_cells(
        by_np, args.tasks, ceiling, args.floor_margin, args.truncation_drop,
    )
    result = pd.DataFrame(rows).sort_values(["task", "nodes", "density"])

    out = args.out or "analysis/density_size_grid.scout.csv"
    result.to_csv(out, index=False)
    dropped = result[result["decision"] == "drop"]
    archive_out = (
        args.archive_out or "analysis/density_size_grid.archive_filtered.csv"
    )
    dropped.to_csv(archive_out, index=False)
    survivors = result[result["decision"] == "survive"][
        ["task", "nodes", "density"]
    ]
    survivors_out = (
        args.survivors_out or "analysis/density_size_grid.survivors.json"
    )
    with open(survivors_out, "w") as handle:
      json.dump(survivors.to_dict("records"), handle, indent=2)

    print(f"wrote {len(result)} rows to {out}")
    print(f"  {len(dropped)} dropped -> {archive_out}")
    print(f"  {len(survivors)} survivors -> {survivors_out}")

  else:
    shortcuts_by_cell = (
        analysis.load_shortcuts(args.shortcuts) if args.shortcuts else {}
    )
    responses_pattern = (
        args.responses_pattern
        or "runs/{model}.grid_full_n{size}_p{density}.jsonl"
    )
    prompts_pattern = (
        args.prompts_pattern or "prompts_grid_full_n{size}_p{density}.jsonl"
    )
    models = [args.model_family, f"{args.model_family}-think"]
    frames = load_all_frames(
        models, args.sizes, args.densities, responses_pattern,
        prompts_pattern, shortcuts_by_cell, args.density_tolerance,
    )
    if not frames:
      raise SystemExit("no full-reporter response files found")
    rows = full_cells(
        frames, args.tasks, shortcuts_by_cell, ceiling, args.floor_margin,
        args.shortcut_bar, args.truncation_threshold,
        args.min_effect_of_interest, args.power_target, args.n_perm,
        args.n_boot, args.n_replicates, args.seed,
    )
    result = pd.DataFrame(rows).sort_values(
        ["task", "nodes", "density", "condition", "arm"]
    )

    out = args.out or "analysis/density_size_grid.full.csv"
    result.to_csv(out, index=False)
    report_path = args.report or "docs/density-size-range-search.md"
    write_report(result, report_path)

    print(f"wrote {len(result)} rows to {out}")
    print(f"wrote narrative report to {report_path}")


if __name__ == "__main__":
  main()
