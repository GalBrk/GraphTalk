"""Stage 4: pooled significance tests over the tracked sweep. No GPU needed.

`scripts/score_sweep.py` runs exact McNemar per (task, style, condition)
cell -- 288 cells across 4 models, most with fewer than 10 discordant pairs
out of 30, and no correction for testing 288 of them at once (see
docs/sweep-findings.md, "The McNemar analysis is underpowered"). This script
pools pairs across task and style instead, per `graphtalk/significance.py`:
a permutation p-value, a bootstrap CI on the effect size, and a
Benjamini-Hochberg correction across the *independent* conditions tested for
one model -- `all` (the union of `degree`/`clustering`/`rwse`) is corrected
as its own single-hypothesis family instead of being pooled with the
conditions it's derived from, see `_is_derived_condition`. It reuses the
same pooling for the thinking arm's non-termination rate, replacing the ad
hoc p-values quoted in prose there.

Reads the already-scored, already-joined table `scripts/build_sweep_frame.py`
writes -- no re-scoring, no re-reading raw runs/*.jsonl. Raises if `--frame`
carries more than one `node_naming` scheme (a pooled significance number
across schemes would be meaningless, not just mislabeled) or a duplicated
`(model, instance_id, condition, style, node_naming)` key (silently corrupts
every pairing downstream, via `pandas`'s cross-join on a non-unique index).

Pooling across style (and, for "pooled across all models" rows, across
model too) means the same graph instance recurs many times in one pooled
sample, so `(model, instance_id)` is threaded through as a *cluster* id: the
permutation test flips and the bootstrap resamples whole clusters, not
individual rows, so correlated rows sharing a graph don't get counted as
independent evidence (see `graphtalk/significance.py`'s module docstring).
The cluster id carries `model`, not just `instance_id` -- otherwise a
"pooled across all models" row would merge the same graph number from four
different model families into one cluster, assuming those families'
errors on that graph correlate as strongly as one model's own repeated
answers to it do. They may not, so only same-model rows sharing an instance
are treated as correlated.

Main-sweep `exact` rows are reported three times per condition, tagged by
`bound`, since a non-terminating response's true (untruncated) outcome is
unknown:

- `excluded` drops `non_terminating` responses before pairing -- a
  truncated response's `exact` score reflects abandoned working, not
  reasoning quality.
- `best_case`/`worst_case` keep every row, but override a non-terminating
  response's `exact` score to 1.0/0.0 rather than trusting whatever the
  truncated-text answer-extractor happened to land on -- a real bracket on
  the unknown true outcome, not a second point estimate.

Dropping non-terminating rows (`excluded`) is not free of bias either:
non-termination itself responds to the primer condition
(docs/sweep-findings.md), so a condition that happens to truncate on
instances it would have gotten wrong anyway would look artificially better
under `excluded` alone. Reporting the bracket alongside it makes that range
visible instead of picking one estimate silently. `n_excluded_non_terminating`
reports how many rows `excluded` dropped per condition (always 0 for
`best_case`/`worst_case`, which drop nothing, only override). The thinking
arm's own `non_terminating` rows are the outcome being tested there --
never excluded or overridden, and `bound` is `not_applicable` for those
rows (not the literal text `"n/a"` -- that string is one of pandas'
default NA tokens, so `pd.read_csv` would silently turn it into `NaN` on
every re-read, which is exactly the bug this sentinel avoids).

`n_looped_on_correct_answer` and `low_power` are further diagnostics,
populated only on `bound == "excluded"` main-sweep rows (blank elsewhere):
the former counts non-terminating rows whose response settled on the
correct answer before getting cut off rather than genuinely drifting (see
`graphtalk.analysis.build_frame`'s `looped_on_correct_answer` column); the
latter flags a row where the exclusion rate is high enough that a
"not significant" result may mean "not enough clean data" rather than
"no effect" (default threshold 15%, `--low-power-threshold`).
`n_instances_missing` (every row, every bound) is the count of graph
instances with zero surviving pairs -- computed from the data, not a
hardcoded total, since the sweep's instance count changes with `--count`.

`bh_significant` corrects across the ~5 *independent* conditions within one
`(arm, group, bound)` family (`all` gets its own `.../derived` family, see
above). `is_derived_condition` marks which rows that is, so a reader can
tell the two families apart without parsing the `bh_family` string.
`bh_significant_global` is a second, additional correction across every
`excluded`/`not_applicable`-bound, non-derived row from the whole run
(**now spanning both `--metric exact` and `--metric mae` in one pass** --
see `--metric` below), excluding `best_case`/`worst_case` rows (a
sensitivity bracket, not an independent hypothesis), excluding
`pooled across all models` rows in both arms (built from the same
underlying pairs as their sibling per-model rows, so not an independent
test either), and excluding derived-condition rows for the same reason as
`bh_significant`. The first column answers "does this condition matter for
this model"; the second answers "how many of the whole table's findings
survive testing it all at once". Both are kept; neither replaces the other.

Four further diagnostics distinguish "no effect" from "couldn't have seen
one even if it were there":

- `near_ceiling` -- the CONTROL condition's own mean `metric` is above
  `--near-ceiling-threshold` (default 95%) or below its complement (a
  floor, not just a ceiling -- e.g. `gemma4-e4b-think`'s 0% non-termination
  baseline). Populated on `bound in ("excluded", "not_applicable")` rows
  only; a
  best_case/worst_case bracket forces non-terminating rows to fixed
  extremes and would distort the read.
- `headroom` -- `min(control_mean, 1 - control_mean)`, the theoretical max
  fraction of rows that could still flip. Alongside `near_ceiling` and
  `low_power`, this is what lets a reader tell "low_power *because of*
  near-ceiling headroom" apart from "low_power despite real headroom left"
  -- the two flags alone can't distinguish those, and on the current data
  they happen to co-occur completely for one model. Same populated scope
  as `near_ceiling`.
- `task_delta_min`/`task_delta_max` -- per-task point-estimate deltas
  (`instance_id` already encodes its task as a `/`-prefix, e.g.
  `edge_count/27`, so no extra join is needed), purely descriptive, no new
  hypothesis test and no added multiple-comparison burden. A pooled `delta`
  near zero with a wide `[task_delta_min, task_delta_max]` range means the
  condition helps on some tasks and hurts on others rather than doing
  nothing everywhere.
- `mde_delta`/`mde_realized_diff`/`mde_power_target` -- opt-in via `--mde`
  (off by default: roughly 6-7x's the run's total time), computed only for
  rows where `bh_significant is False` and
  `bound in ("excluded", "not_applicable")`.
  `graphtalk.significance.minimum_detectable_effect_clustered` simulates
  the smallest true effect this row's data could have reliably detected;
  `mde_realized_diff` can come in below `mde_delta` on a `near_ceiling` row,
  which is itself a second, independent confirmation that a ceiling effect
  is suppressing detectability there. Blank wherever not computed --
  `bh_significant is True` (nothing to explain), a bracket row (not a
  primary interpretive bound), or `--mde` wasn't passed.

  PYTHONPATH=. .venv/bin/python scripts/check_significance.py \
      --frame analysis/sweep_frame.csv

**`--metric`**: `both` (default) runs `exact` (accuracy vs. `none`, pooled
across tasks, main sweep + thinking arm) and `mae` (mean absolute error
instead of exact-match accuracy, reused clustering/permutation/bootstrap/BH
machinery) in **one pass**, so `_apply_global_bh` corrects across every
record either metric produced -- `exact` and `mae` are two lenses on the
same underlying (model, condition) hypotheses, not two independent
questions, so they share one multiplicity budget rather than each getting
its own (which would understate how many comparisons were actually made).
Pass `--metric exact` or `--metric mae` to run just one, e.g. for a faster
iteration loop during development; its BH correction is then scoped to
only that metric's rows, matching the old (pre-unification) behavior.

`mae` is scoped to the 3 integer tasks `absolute_error` is defined for
(`node_count`, `edge_count`, `node_degree`; the frame carries `NaN` for
the other three, where the metric doesn't apply). Reported **per task**,
not pooled across them: a node-count error of 2 and a degree error of 2
aren't the same size of mistake, so averaging them the way `exact` pools
across all 6 tasks would conflate different quantities -- this mode's
per-task split incidentally gives real per-task significance for these
three tasks, unlike `exact`'s pooled-only view.

Excludes `non_terminating` rows before pairing (same reasoning as the
`exact` metric's `excluded` bound -- a truncated response's extracted
number isn't a meaningful "how close" signal) and any row where
`absolute_error` itself is undefined (`unparsed` responses). No
`best_case`/`worst_case` bracket (unbounded metric, no principled "worst
case" value the way 0/1 has), no `near_ceiling` (a 0-1-accuracy concept),
no `task_delta_min`/`max` (redundant -- this mode doesn't pool across
tasks to begin with). Reports `mae_delta = control_mae - treatment_mae`,
not the raw `treatment - control` the underlying functions return --
lower error is better, the opposite sign convention from `exact`'s
"higher is better", so this flip keeps a positive number meaning "helped"
in both modes.

  PYTHONPATH=. .venv/bin/python scripts/check_significance.py \
      --frame analysis/sweep_frame.csv --metric mae
"""

import argparse

import pandas as pd

from graphtalk import analysis
from graphtalk import primers
from graphtalk import significance

CONTROL = "none"
_KEYS = ["model", "instance_id", "style", "node_naming"]
_BRACKET_BOUNDS = ("best_case", "worst_case")
_MAE_TASKS = ("node_count", "edge_count", "node_degree")


def _is_derived_condition(condition: str) -> bool:
  """Whether `condition` is a combination of other conditions (currently
  just `all`, the union of `degree`/`clustering`/`rwse`) rather than an
  independent manipulation.

  Derived from `graphtalk.primers.CONDITIONS` rather than hardcoding the
  string `"all"`, so a future multi-component condition is caught the same
  way without a second place to remember to update. A derived condition is
  mechanically correlated with its components -- pooling it into the same
  multiple-comparison family as `degree`/`clustering`/`rwse` overstates how
  many independent hypotheses are being tested, so it is corrected as its
  own single-hypothesis family instead (see `_report`/`_report_mae`).
  """
  return len(primers.CONDITIONS[condition]) > 1


def _paired_values(frame: pd.DataFrame, condition: str, metric: str):
  """Aligned (control, treatment, cluster_ids) for `condition` vs `CONTROL`.

  Paired on `_KEYS`: instance_id alone repeats across styles for the same
  task, so style has to be part of the key or a control row of one style could
  pair against a treatment row of another; model has to be part of it too,
  since the same (instance_id, style) key recurs once per model when `frame`
  pools rows across models. `node_naming` is part of it for the same reason
  `model` is -- without it, `main()`'s own guard aside, a frame that ever did
  carry both schemes for one model would pair a `got` control row against an
  `integer` treatment row (or vice versa) via `set_index`, which raises on
  nothing; it just silently keeps one of the duplicates. `cluster_ids` is
  `(model, instance_id)` pulled back out of the joined index -- the unit
  `paired_permutation_test_clustered`/`cluster_bootstrap_ci_clustered` need,
  since the *pairing* key has to include style/node_naming to be unique but
  the *correlation* those two functions correct for lives at the
  per-model-per-instance level. `model` stays part of the cluster id (not
  just `instance_id`) so a pooled-across-models call doesn't merge the same
  graph number from different model families into one cluster.
  """
  control = frame[frame["condition"] == CONTROL].set_index(_KEYS)[metric]
  treatment = frame[frame["condition"] == condition].set_index(_KEYS)[metric]
  joined = pd.concat(
      [control.rename("control"), treatment.rename("treatment")],
      axis=1, join="inner",
  )
  cluster_ids = list(zip(
      joined.index.get_level_values("model"),
      joined.index.get_level_values("instance_id"),
  ))
  return joined["control"].tolist(), joined["treatment"].tolist(), cluster_ids


def _count_excluded_non_terminating(raw_frame: pd.DataFrame, condition: str) -> int:
  """How many `condition`-or-`CONTROL` rows in `raw_frame` were dropped for
  being `non_terminating` -- the size of the confound the `excluded` bound
  removes, kept visible rather than silently disappearing."""
  relevant = raw_frame[raw_frame["condition"].isin([CONTROL, condition])]
  return int((relevant["failure_type"] == "non_terminating").sum())


def _count_looped_on_correct_answer(raw_frame: pd.DataFrame, condition: str) -> int:
  """How many of those same `non_terminating` rows settled on the correct
  answer before getting cut off, rather than genuinely drifting -- same
  scope as `_count_excluded_non_terminating`, see its docstring."""
  relevant = raw_frame[raw_frame["condition"].isin([CONTROL, condition])]
  return int((relevant["looped_on_correct_answer"] == True).sum())  # noqa: E712


def _task_delta_range(control, treatment, cluster_ids):
  """Per-task point-estimate deltas (`mean(treatment) - mean(control)`),
  descriptive only -- no new hypothesis test, no new multiple-comparison
  burden. `instance_id` (the second element of each `cluster_ids` tuple)
  already encodes its task as a `/`-prefix (e.g. `edge_count/27`), so no
  extra join or `_paired_values` change is needed. Returns `(min, max)`
  across tasks -- surfaces whether a near-zero pooled delta is hiding tasks
  that actually disagree in direction, or genuinely reflects "nothing much
  happening on any task."
  """
  by_task: dict = {}
  for (_model, instance_id), c, t in zip(cluster_ids, control, treatment):
    task = instance_id.split("/")[0]
    by_task.setdefault(task, []).append(t - c)
  if not by_task:
    return None, None
  task_means = [sum(diffs) / len(diffs) for diffs in by_task.values()]
  return min(task_means), max(task_means)


def _bracket_frame(raw_frame: pd.DataFrame, value: float) -> pd.DataFrame:
  """`raw_frame` with `exact` overridden to `value` on `non_terminating`
  rows, every other row's actual observed value left untouched -- the
  best_case (`value=1.0`) / worst_case (`value=0.0`) bracket: a
  non-terminating response's true outcome is unknown, so this asks "what if
  it had gone the best/worst possible way" instead of trusting the
  truncated-text extractor's guess."""
  frame = raw_frame.copy()
  frame.loc[frame["failure_type"] == "non_terminating", "exact"] = value
  return frame


def _report(
    frame: pd.DataFrame, raw_frame: pd.DataFrame, metric: str, label: str,
    args, arm: str, records: list, bound: str = "not_applicable",
) -> None:
  print(f"\n  {label} [{bound}]")
  conditions = sorted(c for c in frame["condition"].unique() if c != CONTROL)
  rows = []
  # The number of *clusters* -- (model, instance_id) pairs, matching Fix 2's
  # cluster granularity, not bare instance_id -- available to this group
  # before any exclusion. Read from the data, not hardcoded, so
  # `n_instances_missing` stays correct if the sweep's `--count` ever
  # changes (see module docstring). Using bare instance_id here would
  # undercount the baseline for a pooled-across-models call (up to 4
  # clusters can share one instance_id) and make `n_instances_missing`
  # go negative -- caught by running this against the real sweep data.
  total_clusters_possible = int(
      raw_frame[["model", "instance_id"]].drop_duplicates().shape[0]
  )
  near_ceiling = None
  headroom = None
  if bound not in _BRACKET_BOUNDS:
    control_mean = frame.loc[frame["condition"] == CONTROL, metric].mean()
    if pd.notna(control_mean):
      near_ceiling = bool(
          control_mean > args.near_ceiling_threshold
          or control_mean < 1 - args.near_ceiling_threshold
      )
      # The theoretical max fraction of rows that could still flip --
      # `near_ceiling` alone can't say whether a `low_power` row is
      # low_power *because* of that headroom limit or despite having real
      # headroom left; this makes the two separable.
      headroom = min(control_mean, 1 - control_mean)
  for condition in conditions:
    control, treatment, cluster_ids = _paired_values(frame, condition, metric)
    if not control:
      print(f"    {condition:<12} -- no paired rows found")
      continue
    # A distinct seed per test, not one value reused everywhere: otherwise
    # the Monte Carlo estimation noise across the p-values fed into the same
    # benjamini_hochberg call below is correlated rather than independent.
    # random.Random only accepts None/int/float/str/bytes/bytearray, not an
    # arbitrary tuple, hence the string join.
    seed = f"{args.seed}:{arm}:{label}:{bound}:{condition}"
    perm = significance.paired_permutation_test_clustered(
        control, treatment, cluster_ids, n_perm=args.n_perm, seed=seed
    )
    boot = significance.cluster_bootstrap_ci_clustered(
        control, treatment, cluster_ids, n_boot=args.n_boot, seed=seed,
        alpha=args.alpha,
    )
    task_delta_min, task_delta_max = _task_delta_range(
        control, treatment, cluster_ids
    )
    if bound in _BRACKET_BOUNDS:
      # Nothing was excluded here -- non_terminating rows were overridden,
      # not dropped -- so both diagnostics are inapplicable, not just zero.
      n_excluded, n_looped, low_power = 0, None, None
    else:
      n_excluded = _count_excluded_non_terminating(raw_frame, condition)
      if bound == "excluded":
        n_looped = _count_looped_on_correct_answer(raw_frame, condition)
        denom = perm["n_pairs"] + n_excluded
        low_power = (
            (n_excluded / denom) > args.low_power_threshold if denom else False
        )
      else:
        # Thinking arm (`bound == "not_applicable"`): `n_excluded` here is really the
        # non-termination count itself, the outcome being tested, not a
        # confound to flag power against -- so these stay inapplicable too.
        n_looped, low_power = None, None
    n_instances_missing = total_clusters_possible - perm["n_clusters"]
    rows.append((
        condition, perm, boot, n_excluded, n_looped, low_power,
        n_instances_missing, task_delta_min, task_delta_max,
        control, treatment, cluster_ids, _is_derived_condition(condition),
    ))

  if not rows:
    return
  mde_eligible = args.mde and bound not in _BRACKET_BOUNDS
  print(f"    {'condition':<12}{'n_clusters':>11}{'delta':>10}"
        f"{'95% CI':>22}{'p (perm)':>10}  BH-sig")

  # Independent conditions (degree/clustering/rwse/components/filler) are
  # corrected together; `all` -- mechanically the union of three of those --
  # is corrected as its own single-hypothesis family instead of inflating
  # (or diluting) the real one. See `_is_derived_condition`.
  independent_rows = [r for r in rows if not r[-1]]
  derived_rows = [r for r in rows if r[-1]]

  def _emit(group_rows: list, family_suffix: str) -> None:
    if not group_rows:
      return
    # What "corrected together" means for the bh_significant flags below:
    # this one (arm, label, bound[, derived]) family, not the whole report
    # -- see the module docstring's note on pooling, and `main()`'s
    # separate global BH pass.
    bh_family = f"{arm}/{label}/{bound}{family_suffix}"
    reject = significance.benjamini_hochberg(
        [row[1]["p_value"] for row in group_rows], q=args.q
    )
    for (condition, perm, boot, n_excluded, n_looped, low_power, n_missing,
         task_delta_min, task_delta_max, control, treatment, cluster_ids,
         is_derived), sig in zip(group_rows, reject):
      ci = f"[{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}]"
      print(f"    {condition:<12}{perm['n_clusters']:>11}"
            f"{perm['observed_diff']:>+10.3f}{ci:>22}"
            f"{perm['p_value']:>10.4f}  {'yes' if sig else 'no'}")
      mde_delta = mde_realized = mde_power_target = None
      if mde_eligible and not sig:
        ci_width = boot["ci_high"] - boot["ci_low"]
        mde_seed = f"{args.seed}:{arm}:{label}:{bound}:{condition}:mde"
        mde = significance.minimum_detectable_effect_clustered(
            control, treatment, cluster_ids, initial_hi=max(0.05, ci_width),
            alpha=args.alpha, power_target=args.mde_power_target,
            n_replicates=args.mde_replicates, n_perm=args.mde_n_perm,
            seed=mde_seed,
        )
        mde_delta, mde_realized = mde["delta"], mde["realized_diff"]
        mde_power_target = mde["power_target"]
        print(f"      MDE: delta={mde_delta} realized={mde_realized} "
              f"({mde['note'] or 'ok'})")
      records.append({
          "arm": arm,
          "group": label,
          "metric": metric,
          "condition": condition,
          "is_derived_condition": is_derived,
          "bound": bound,
          "bh_family": bh_family,
          "n_pairs": perm["n_pairs"],
          "n_clusters": perm["n_clusters"],
          "n_instances_missing": n_missing,
          "n_excluded_non_terminating": n_excluded,
          "n_looped_on_correct_answer": n_looped,
          "low_power": low_power,
          "near_ceiling": near_ceiling,
          "headroom": headroom,
          "task_delta_min": task_delta_min,
          "task_delta_max": task_delta_max,
          "delta": perm["observed_diff"],
          "ci_low": boot["ci_low"],
          "ci_high": boot["ci_high"],
          "p_value": perm["p_value"],
          "bh_significant": sig,
          "mde_delta": mde_delta,
          "mde_realized_diff": mde_realized,
          "mde_power_target": mde_power_target,
      })

  _emit(independent_rows, "")
  _emit(derived_rows, "/derived")


def _apply_global_bh(records: list, q: float) -> None:
  """The whole-table BH pass, mutating `records` in place.

  Eligible rows are every `bound not in _BRACKET_BOUNDS` row (excludes the
  best_case/worst_case sensitivity bracket) whose `group` is not
  `"pooled across all models"` (excludes a row built from the same
  underlying pairs as its sibling per-model rows in the same arm -- not an
  independent hypothesis) and whose condition is not derived (excludes
  `all`, mechanically the union of degree/clustering/rwse -- see
  `_is_derived_condition` -- also not an independent hypothesis). Called
  once per `main()` run, over every record collected regardless of which
  metric produced it (`exact`, `non_terminating`, `mae`) -- exact and mae
  are two lenses on the same underlying (model, condition) hypotheses, so
  they share one multiplicity budget rather than two separate ones.
  Ineligible rows get `bh_significant_global = None`, not `False`, so "not
  significant" and "not tested" stay distinguishable.
  """
  eligible = [
      r for r in records
      if r["bound"] not in _BRACKET_BOUNDS
      and r["group"] != "pooled across all models"
      and not r["is_derived_condition"]
  ]
  reject_global = significance.benjamini_hochberg(
      [r["p_value"] for r in eligible], q=q
  )
  for r, sig in zip(eligible, reject_global):
    r["bh_significant_global"] = sig
  for r in records:
    r.setdefault("bh_significant_global", None)


def _mae_eligible_frame(main_sweep_raw: pd.DataFrame) -> pd.DataFrame:
  """Rows `--metric mae` can pair: one of `_MAE_TASKS`, `non_terminating`
  excluded, and `absolute_error` actually defined (an `unparsed` row has
  none) -- see the module docstring's `mae` mode section for why each of
  these is excluded."""
  return main_sweep_raw[
      main_sweep_raw["task"].isin(_MAE_TASKS)
      & (main_sweep_raw["failure_type"] != "non_terminating")
      & main_sweep_raw["absolute_error"].notna()
  ]


def _report_mae(
    frame: pd.DataFrame, raw_frame: pd.DataFrame, label: str, task: str,
    args, records: list,
) -> None:
  """`--metric mae`'s counterpart to `_report`, scoped to one
  `(label, task)` pair.

  Deliberately does not call `_report`: its bracket/`near_ceiling`/
  `task_delta`/MDE logic is all specific to the 0-1 `exact` metric and
  doesn't translate to an unbounded error metric (see the module
  docstring). The underlying clustering/permutation/bootstrap/BH
  primitives are metric-agnostic and are reused as-is; only the
  orchestration and the `mae_delta` sign flip are new.
  """
  print(f"\n  {label} / {task}")
  conditions = sorted(c for c in frame["condition"].unique() if c != CONTROL)
  rows = []
  for condition in conditions:
    control, treatment, cluster_ids = _paired_values(
        frame, condition, "absolute_error"
    )
    if not control:
      print(f"    {condition:<12} -- no paired rows found")
      continue
    seed = f"{args.seed}:mae:{label}:{task}:{condition}"
    perm = significance.paired_permutation_test_clustered(
        control, treatment, cluster_ids, n_perm=args.n_perm, seed=seed
    )
    boot = significance.cluster_bootstrap_ci_clustered(
        control, treatment, cluster_ids, n_boot=args.n_boot, seed=seed,
        alpha=args.alpha,
    )
    n_excluded = _count_excluded_non_terminating(raw_frame, condition)
    rows.append((condition, perm, boot, n_excluded, _is_derived_condition(condition)))

  if not rows:
    return
  print(f"    {'condition':<12}{'n_clusters':>11}{'mae_delta':>10}"
        f"{'95% CI':>22}{'p (perm)':>10}  BH-sig")

  # Same split as `_report`: `all` is a derived condition (the union of
  # degree/clustering/rwse), corrected as its own single-hypothesis family
  # rather than pooled with the independent conditions -- see
  # `_is_derived_condition`.
  independent_rows = [r for r in rows if not r[-1]]
  derived_rows = [r for r in rows if r[-1]]

  def _emit(group_rows: list, family_suffix: str) -> None:
    if not group_rows:
      return
    bh_family = f"mae/{label}/{task}{family_suffix}"
    reject = significance.benjamini_hochberg(
        [perm["p_value"] for _, perm, _, _, _ in group_rows], q=args.q
    )
    for (condition, perm, boot, n_excluded, is_derived), sig in zip(group_rows, reject):
      # Flip sign: the underlying functions return treatment - control (raw
      # error, higher = worse); mae_delta = control - treatment so positive
      # still means "the condition helped", matching `exact`'s convention.
      mae_delta = -perm["observed_diff"]
      ci_low, ci_high = -boot["ci_high"], -boot["ci_low"]
      ci = f"[{ci_low:+.3f}, {ci_high:+.3f}]"
      print(f"    {condition:<12}{perm['n_clusters']:>11}"
            f"{mae_delta:>+10.3f}{ci:>22}"
            f"{perm['p_value']:>10.4f}  {'yes' if sig else 'no'}")
      records.append({
          "arm": "main_sweep",
          "group": label,
          "metric": "mae",
          "task": task,
          "condition": condition,
          "is_derived_condition": is_derived,
          "bound": "excluded",
          "bh_family": bh_family,
          "n_pairs": perm["n_pairs"],
          "n_clusters": perm["n_clusters"],
          "n_excluded_non_terminating": n_excluded,
          "mae_delta": mae_delta,
          "ci_low": ci_low,
          "ci_high": ci_high,
          "p_value": perm["p_value"],
          "bh_significant": sig,
      })

  _emit(independent_rows, "")
  _emit(derived_rows, "/derived")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frame", default="analysis/sweep_frame.csv")
  parser.add_argument("--metric", choices=("exact", "mae", "both"), default="both",
                       help="'both' (default): runs 'exact' (accuracy-vs-none, "
                            "pooled across tasks, main sweep + thinking arm) "
                            "and 'mae' (per-task mean absolute error, 3 "
                            "integer tasks) in one pass, sharing one "
                            "multiplicity-correction budget -- see the module "
                            "docstring. 'exact' or 'mae' alone runs only that "
                            "metric, kept for faster single-metric runs during "
                            "development; its BH correction is then scoped to "
                            "just that metric's rows, not the union")
  parser.add_argument("--n-perm", type=int, default=10_000,
                       help="permutations for the pooled p-value")
  parser.add_argument("--n-boot", type=int, default=10_000,
                       help="resamples for the pooled bootstrap CI")
  parser.add_argument("--alpha", type=float, default=0.05,
                       help="bootstrap CI level (default 95%% CI)")
  parser.add_argument("--q", type=float, default=0.05,
                       help="Benjamini-Hochberg FDR level")
  parser.add_argument("--low-power-threshold", type=float, default=0.15,
                       help="flag a row's `low_power` column when its "
                            "non-terminating exclusion rate exceeds this "
                            "fraction of its would-be sample")
  parser.add_argument("--near-ceiling-threshold", type=float, default=0.95,
                       help="flag a row's `near_ceiling` column when its "
                            "control condition's mean metric is above this "
                            "(or below 1 minus this)")
  parser.add_argument("--mde", action="store_true",
                       help="compute a simulated minimum-detectable-effect "
                            "for every non-significant excluded/n-a-bound "
                            "row -- roughly 6-7x's total runtime, off by "
                            "default")
  parser.add_argument("--mde-power-target", type=float, default=0.8)
  parser.add_argument("--mde-replicates", type=int, default=200,
                       help="simulated trials per candidate effect size")
  parser.add_argument("--mde-n-perm", type=int, default=500,
                       help="permutations per simulated trial's inner test "
                            "(lower than --n-perm: averaged over many "
                            "trials, so per-trial precision matters less)")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--out", default=None,
                       help="optional path to write every row printed above "
                            "as CSV -- analysis.tagged_path suffixes it for a "
                            "non-integer node_naming scheme automatically")
  args = parser.parse_args()

  frame = pd.read_csv(args.frame)
  # Raises on a genuine mix; a frame predating this column has none at all,
  # normalized to "integer" so `_paired_values`'s key can always rely on it
  # being present.
  scheme = analysis.frame_node_naming(frame)
  if "node_naming" not in frame.columns:
    frame = frame.assign(node_naming="integer")
  # A frame predating `graphtalk.analysis.build_frame`'s
  # `looped_on_correct_answer` column carries neither it nor
  # `predicted_first` -- default both to a column of Nones so
  # `_count_looped_on_correct_answer` still has something to compare
  # against (it will just count 0 everywhere, same as "not computed yet").
  if "looped_on_correct_answer" not in frame.columns:
    frame = frame.assign(looped_on_correct_answer=None, predicted_first=None)
  # The FULL row identity, not `_KEYS` -- `_KEYS` is the *pairing* key,
  # correct only once `_paired_values` has already filtered down to a single
  # `condition`. Checking it against the whole (all-conditions) frame would
  # flag every row as a "duplicate" of its six sibling conditions.
  analysis.assert_unique_pairing_key(frame, ["model", "instance_id", "condition", "style", "node_naming"])
  records = []

  main_sweep_raw = frame[~frame["is_think"]]

  if args.metric in ("exact", "both"):
    # main_sweep_raw is what n_excluded_non_terminating/n_instances_missing
    # count against; main_sweep (non_terminating dropped) is the `excluded`
    # bound; main_sweep_best/worst (non_terminating overridden, not dropped)
    # are the bracket -- see the module docstring on why all three are kept.
    main_sweep = main_sweep_raw[main_sweep_raw["failure_type"] != "non_terminating"]
    main_sweep_best = _bracket_frame(main_sweep_raw, 1.0)
    main_sweep_worst = _bracket_frame(main_sweep_raw, 0.0)
    think = frame[frame["is_think"]]

    print("=" * 78)
    print("Main sweep: accuracy (exact) vs `none`, pooled across task + style")
    for model_family, group in main_sweep.groupby("model_family"):
      raw_group = main_sweep_raw[main_sweep_raw["model_family"] == model_family]
      best_group = main_sweep_best[main_sweep_best["model_family"] == model_family]
      worst_group = main_sweep_worst[main_sweep_worst["model_family"] == model_family]
      _report(group, raw_group, "exact", model_family, args, "main_sweep",
              records, bound="excluded")
      _report(best_group, raw_group, "exact", model_family, args, "main_sweep",
              records, bound="best_case")
      _report(worst_group, raw_group, "exact", model_family, args, "main_sweep",
              records, bound="worst_case")
    _report(main_sweep, main_sweep_raw, "exact", "pooled across all models",
            args, "main_sweep", records, bound="excluded")
    _report(main_sweep_best, main_sweep_raw, "exact", "pooled across all models",
            args, "main_sweep", records, bound="best_case")
    _report(main_sweep_worst, main_sweep_raw, "exact", "pooled across all models",
            args, "main_sweep", records, bound="worst_case")

    if not think.empty:
      print(f"\n{'=' * 78}")
      print("Thinking arm: non-termination rate vs `none`, pooled across task")
      for model_family, group in think.groupby("model_family"):
        _report(group, group, "non_terminating", model_family, args,
                "thinking_arm", records)
      _report(think, think, "non_terminating", "pooled across all models",
              args, "thinking_arm", records)

  if args.metric in ("mae", "both"):
    print("=" * 78)
    print("Main sweep: mean absolute error vs `none`, per task (not pooled)")
    for model_family, raw_group in main_sweep_raw.groupby("model_family"):
      eligible = _mae_eligible_frame(raw_group)
      for task in _MAE_TASKS:
        _report_mae(
            eligible[eligible["task"] == task], raw_group[raw_group["task"] == task],
            model_family, task, args, records,
        )

  # A second, whole-table BH pass, over every record collected above
  # regardless of which metric produced it -- see `_apply_global_bh`.
  _apply_global_bh(records, args.q)

  if args.out:
    out = analysis.tagged_path(args.out, scheme)
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"\nwrote {len(records)} rows to {out}")


if __name__ == "__main__":
  main()
