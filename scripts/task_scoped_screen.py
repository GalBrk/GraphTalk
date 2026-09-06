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
      --frame analysis/sweep_frame.got.csv
"""

import argparse

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


def screen(
    frame: pd.DataFrame, n_perm: int = 10_000, n_boot: int = 10_000,
    alpha: float = 0.05, seed: int = 1234,
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
            # Audit-derived classification (analysis/
            # primer_task_shortcut_audit.md), not recomputed from the
            # ceiling numbers above -- see module docstring. Every
            # (condition, task) pair the screen can produce is a key in
            # _SHORTCUT_AUDIT; a KeyError here means a new condition/task
            # was added without extending the audit, which should fail
            # loudly rather than silently label it "none".
            "shortcut_flag": _SHORTCUT_AUDIT[(condition, task)],
        })
  return rows


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frame", default="analysis/sweep_frame.got.csv")
  parser.add_argument("--out", default=None,
                      help="default analysis/task_scoped_screen.<scheme>.csv")
  parser.add_argument("--n-perm", type=int, default=10_000)
  parser.add_argument("--n-boot", type=int, default=10_000)
  parser.add_argument("--alpha", type=float, default=0.05)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--screen-p", type=float, default=0.10,
                      help="cells with p <= this are printed as candidates "
                           "(a looser screening threshold than 0.05 -- the "
                           "point is to surface follow-up candidates, not "
                           "declare findings)")
  args = parser.parse_args()

  frame = pd.read_csv(args.frame)
  scheme = analysis.frame_node_naming(frame)

  rows = screen(
      frame, n_perm=args.n_perm, n_boot=args.n_boot, alpha=args.alpha,
      seed=args.seed,
  )
  result = pd.DataFrame(rows).sort_values("p_value")

  out = args.out or analysis.tagged_path(
      "analysis/task_scoped_screen.csv", scheme
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
  other_candidates = candidates[candidates["shortcut_flag"] != "none"]
  print(f"\n{len(candidates)} cell(s) with p <= {args.screen_p}, split by "
        f"shortcut_flag (see analysis/primer_task_shortcut_audit.md):")
  _print_section(
      none_candidates,
      "shortcut_flag == 'none' -- real graph-reasoning candidates",
  )
  _print_section(
      other_candidates,
      "shortcut_flag in ('shortcut', 'partial') -- arithmetic/lookup "
      "execution-reliability candidates, NOT graph-reasoning evidence",
  )


if __name__ == "__main__":
  main()
