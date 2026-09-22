"""Phase 1 (`docs/plans/run_improved_tests.md`): a repeatable, per-(model,
condition, task) significance screen for the `exact` metric, run directly
against already-collected data -- no new GPU time.

This turns what was originally a one-off manual check (`degree` vs `none`
restricted to `edge_count` alone, spot-checked by hand and found
near-significant or significant in 3 of 4 models at n=30) into a script
that screens every model/condition/task cell in a frame at once, so a
concentrated task-specific effect the pooled `--metric exact` view dilutes
away is never missed just because nobody thought to look at that one
cell.

**On "shortcut-ceiling-bound tasks."** The plan that requested this script
originally assumed a fixed list of *tasks* a graph-blind primer-only
solver already solves near 100% (see `docs/plans/shortcut-ceilings.md`,
`graphtalk/shortcuts.py`), to be excluded from this screen as an
uninteresting confound. Checking the real, current `shortcuts.json`
(embedded in every row of the frame as `shortcut_score`) shows this isn't
actually true *per task*: shortcut ceiling is a property of a
`(task, condition)` **pair**, not a task alone -- no task is anywhere near
ceiling-bound under the `none` control condition (the highest is
`cycle_check`/`none` at 0.832, not ~1.0), so there is no fixed task list
to exclude on that basis. The plan's current revision replaces that
exclusion with a full audit instead:
`analysis/primer_task_shortcut_audit.md` traces, per `(condition, task)`
pair, whether the primer text mechanically determines the answer
(`shortcut`), gives real but partial leverage via a one-directional
theorem or a moderate correlation (`partial`), or gives nothing detectable
(`none`) -- grounded in `graphtalk/primers.py`'s renderer and
`graphtalk/shortcuts.py`'s actual rule implementations, not the ceiling
number alone. `_SHORTCUT_AUDIT` below is that audit's classification,
hardcoded from the doc (not recomputed from `shortcut_score` at runtime,
so a code change to the audit doc and a code change to this table can
never silently drift apart without a diff showing both). Every one of
the 4 candidates the manual scan and Phase A1 found is `shortcut`-flagged
-- see the audit doc's closing section for what that implies for Phase 5.

`condition == "all"` (the derived union of degree/clustering/rwse) is
excluded outright as a screened cell, per the plan -- it is mechanically
correlated with its components and not independent evidence, matching
`check_significance.py`'s own `_is_derived_condition` policy.

  PYTHONPATH=. .venv/bin/python scripts/task_scoped_screen.py \
      --frame csv2/sweep-small-graph/sweep_frame.got.csv
"""

import argparse
import json

import pandas as pd

from graphtalk import analysis
from graphtalk import significance
from scripts import check_significance as cs

# analysis/primer_task_shortcut_audit.md's classification, hardcoded here
# (see module docstring for why this isn't derived from shortcut_score at
# runtime). `all` is omitted -- excluded from the screen entirely, so it
# never needs a lookup here.
_SHORTCUT_AUDIT = {
    ("degree", "node_count"): "shortcut",
    ("clustering", "node_count"): "shortcut",
    ("rwse", "node_count"): "shortcut",
    ("filler", "node_count"): "shortcut",
    ("components", "node_count"): "none",

    ("degree", "edge_count"): "shortcut",
    ("clustering", "edge_count"): "partial",
    ("rwse", "edge_count"): "none",
    ("filler", "edge_count"): "none",
    ("components", "edge_count"): "none",

    ("degree", "node_degree"): "shortcut",
    ("clustering", "node_degree"): "none",
    ("rwse", "node_degree"): "partial",
    ("filler", "node_degree"): "none",
    ("components", "node_degree"): "none",

    ("degree", "cycle_check"): "partial",
    ("clustering", "cycle_check"): "partial",
    ("rwse", "cycle_check"): "partial",
    ("filler", "cycle_check"): "none",
    ("components", "cycle_check"): "shortcut",

    ("degree", "edge_existence"): "partial",
    ("clustering", "edge_existence"): "partial",
    ("rwse", "edge_existence"): "partial",
    ("filler", "edge_existence"): "none",
    ("components", "edge_existence"): "none",

    ("degree", "connected_nodes"): "partial",
    ("clustering", "connected_nodes"): "none",
    ("rwse", "connected_nodes"): "none",
    ("filler", "connected_nodes"): "none",
    ("components", "connected_nodes"): "none",
}

# Bars at or above this are "the primer text already determines the answer".
_SHORTCUT = 0.99
# A treatment bar this far above the control's is real but partial leverage.
_PARTIAL = 0.05


def load_bars(path):
  """A flat `"<task>/<condition>": bar` file (shortcut_table*.py's --json)."""
  with open(path, encoding="utf-8") as handle:
    raw = json.load(handle)
  bars = {}
  for key, bar in raw.items():
    task, _, condition = key.partition("/")
    bars.setdefault(task, {})[condition] = bar
  return bars


def flag_from_bars(control_bar, treatment_bar):
  """Classify a cell from the bars actually fit on the corpus being screened.

  `_SHORTCUT_AUDIT` is a lookup table built for the 5-19 node published
  split, and it is wrong in both directions on the n=40 corpus:

    - It calls `components`/`node_count` clean. At n=40 *every* `node_count`
      bar is 1.00, the `none` control included, because the answer is always
      "40" -- a graph-blind program scores 100% with no primer at all. That
      is `degenerate`: the cell cannot carry evidence for or against a primer,
      which is a different (and more disqualifying) thing than the primer
      leaking the answer.
    - It calls `clustering`/`edge_existence` and `clustering`/`node_degree`
      contaminated. On the 5-19 split `clustering` really does add +22 pp of
      shortcut information on `edge_existence`. At n=40 the two bars are
      0.727 vs 0.735 and 0.144 vs 0.144 -- bar deltas of -0.008 and exactly
      0. Those cells are clean here, and the audit table would wrongly demote
      the two strongest results in the sweep.

  So when the caller supplies bars refit on the corpus under screen, the flag
  is derived from those bars instead of looked up.
  """
  if control_bar is None or treatment_bar is None:
    return "unknown"
  if control_bar >= _SHORTCUT:
    return "degenerate"
  if treatment_bar >= _SHORTCUT:
    return "shortcut"
  if treatment_bar - control_bar >= _PARTIAL:
    return "partial"
  return "none"


def screen(
    frame: pd.DataFrame, n_perm: int = 10_000, n_boot: int = 10_000,
    alpha: float = 0.05, seed: int = 1234, bars: dict | None = None,
) -> list[dict]:
  """One row per (model, condition, task) cell in `frame`'s main sweep
  (non-`is_think`), `condition != "none"` and not a derived condition
  (`all`). Reuses `check_significance._paired_values` for pairing --
  the plan's own explicit instruction, so this script's pairing can never
  drift from the pairing the rest of the pipeline relies on.
  """
  main_sweep = frame[~frame["is_think"]]
  rows = []
  for model in sorted(main_sweep["model_family"].unique()):
    model_frame = main_sweep[main_sweep["model_family"] == model]
    for task in sorted(model_frame["task"].unique()):
      task_frame = model_frame[model_frame["task"] == task]
      conditions = sorted(
          c for c in task_frame["condition"].unique()
          if c != cs.CONTROL and not cs._is_derived_condition(c)
      )
      control_ceiling_rows = task_frame.loc[
          task_frame["condition"] == cs.CONTROL, "shortcut_score"
      ]
      control_ceiling = (
          control_ceiling_rows.iloc[0] if not control_ceiling_rows.empty else None
      )
      if bars is not None:
        control_ceiling = bars.get(task, {}).get(cs.CONTROL, control_ceiling)
      for condition in conditions:
        control, treatment, cluster_ids = cs._paired_values(
            task_frame, condition, "exact"
        )
        if not control:
          continue
        cell_seed = f"{seed}:{model}:{task}:{condition}"
        perm = significance.paired_permutation_test_clustered(
            control, treatment, cluster_ids, n_perm=n_perm, seed=cell_seed
        )
        boot = significance.cluster_bootstrap_ci_clustered(
            control, treatment, cluster_ids, n_boot=n_boot, seed=cell_seed,
            alpha=alpha,
        )
        treatment_ceiling_rows = task_frame.loc[
            task_frame["condition"] == condition, "shortcut_score"
        ]
        treatment_ceiling = (
            treatment_ceiling_rows.iloc[0] if not treatment_ceiling_rows.empty
            else None
        )
        if bars is not None:
          treatment_ceiling = bars.get(task, {}).get(condition, treatment_ceiling)
        rows.append({
            "model": model,
            "condition": condition,
            "task": task,
            "n_clusters": perm["n_clusters"],
            "delta": perm["observed_diff"],
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "p_value": perm["p_value"],
            "shortcut_ceiling_treatment": treatment_ceiling,
            "shortcut_ceiling_control": control_ceiling,
            # With `bars` (refit on the corpus under screen), derive the flag
            # from those bars -- see `flag_from_bars` for why the audit table
            # is wrong in both directions at n=40. Without them, fall back to
            # the audit: every (condition, task) pair the screen can produce
            # is a key in _SHORTCUT_AUDIT, so a KeyError means a new pair was
            # added without extending the audit, which should fail loudly
            # rather than silently label it "none".
            "shortcut_flag": (
                flag_from_bars(control_ceiling, treatment_ceiling)
                if bars is not None else _SHORTCUT_AUDIT[(condition, task)]
            ),
            "bar_delta": (
                None if control_ceiling is None or treatment_ceiling is None
                else round(treatment_ceiling - control_ceiling, 4)
            ),
            "bar_source": "refit" if bars is not None else "audit(5-19)",
        })
  return rows


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frame", default="csv2/sweep-small-graph/sweep_frame.got.csv")
  parser.add_argument("--out", default=None,
                      help="default analysis/task_scoped_screen.<scheme>.csv")
  parser.add_argument("--n-perm", type=int, default=10_000)
  parser.add_argument("--n-boot", type=int, default=10_000)
  parser.add_argument("--alpha", type=float, default=0.05)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--shortcuts", default=None,
                      help="flat \"<task>/<condition>\": bar JSON to classify "
                           "cells from, instead of _SHORTCUT_AUDIT and the "
                           "frame's shortcut_score. Defaults to "
                           "shortcuts_n40_flat.json when the frame's "
                           "instance_ids say n=40, because the audit table "
                           "is wrong in both directions there (see "
                           "flag_from_bars). Pass \"none\" to force the "
                           "audit.")
  parser.add_argument("--screen-p", type=float, default=0.10,
                      help="cells with p <= this are printed as candidates "
                           "(a looser screening threshold than 0.05 -- the "
                           "point is to surface follow-up candidates, not "
                           "declare findings)")
  args = parser.parse_args()

  frame = pd.read_csv(args.frame)
  scheme = analysis.frame_node_naming(frame)

  # The n=40 corpus encodes its size in the instance id (`.../size40/p0.35/17`).
  # Its bars are not the published split's, so the audit table must not be used
  # on it -- see `flag_from_bars`.
  is_n40 = frame["instance_id"].astype(str).str.contains("/size40/").any()
  shortcuts = args.shortcuts
  if shortcuts is None and is_n40:
    shortcuts = "shortcuts_n40_flat.json"
  bars = None if shortcuts in (None, "none") else load_bars(shortcuts)
  if bars is not None:
    print(f"classifying cells from refit bars: {shortcuts}")
  elif is_n40:
    print("WARNING: n=40 frame classified with the 5-19 audit table; "
          "shortcut_flag is unreliable here (see flag_from_bars)")

  rows = screen(
      frame, n_perm=args.n_perm, n_boot=args.n_boot, alpha=args.alpha,
      seed=args.seed, bars=bars,
  )
  result = pd.DataFrame(rows).sort_values("p_value")

  out = args.out or analysis.tagged_path(
      "csv2/sweep-small-graph/task_scoped_screen.csv", scheme
  )
  result.to_csv(out, index=False)
  print(f"wrote {len(result)} rows to {out}")

  candidates = result[result["p_value"] <= args.screen_p]

  def _print_section(rows_df: pd.DataFrame, title: str) -> None:
    print(f"\n{title} ({len(rows_df)}):\n")
    if rows_df.empty:
      return
    print(f"  {'model':<14}{'condition':<12}{'task':<17}{'n':>5}{'delta':>9}"
          f"{'p':>9}  shortcut_flag  ceiling(treat/none)")
    for _, row in rows_df.iterrows():
      print(f"  {row['model']:<14}{row['condition']:<12}{row['task']:<17}"
            f"{row['n_clusters']:>5}{row['delta']:>+9.3f}{row['p_value']:>9.4f}"
            f"  {row['shortcut_flag']:<13}  {row['shortcut_ceiling_treatment']:.2f}/"
            f"{row['shortcut_ceiling_control']:.2f}")

  # Split per the plan: "none" cells (real graph-reasoning candidates)
  # reported separately from shortcut/partial cells (arithmetic/lookup
  # execution-reliability candidates) -- see analysis/
  # primer_task_shortcut_audit.md's closing section for why this split
  # matters for Phase 5 prioritization.
  none_candidates = candidates[candidates["shortcut_flag"] == "none"]
  degenerate = candidates[candidates["shortcut_flag"] == "degenerate"]
  other_candidates = candidates[
      ~candidates["shortcut_flag"].isin(("none", "degenerate"))
  ]
  print(f"\n{len(candidates)} cell(s) with p <= {args.screen_p}, split by "
        f"shortcut_flag (bars: "
        f"{shortcuts if bars is not None else 'audit table, 5-19 split'}):")
  _print_section(
      none_candidates,
      "shortcut_flag == 'none' -- real graph-reasoning candidates",
  )
  _print_section(
      other_candidates,
      "shortcut_flag in ('shortcut', 'partial') -- arithmetic/lookup "
      "execution-reliability candidates, NOT graph-reasoning evidence",
  )
  # Kept separate from shortcut/partial: there the primer leaks the answer, so
  # the model result is uninterpretable *as primer evidence*. Here the control
  # itself is already at 1.00, so the cell carries no information at all and
  # the delta is measuring something other than graph reasoning -- at n=40 that
  # is the `node_count` off-by-one artifact, which every condition including
  # `filler` moves by the same amount.
  _print_section(
      degenerate,
      "shortcut_flag == 'degenerate' -- the CONTROL's own bar is >= 0.99, so "
      "a graph-blind program already scores ~100% with no primer; not "
      "evidence either way",
  )


if __name__ == "__main__":
  main()
