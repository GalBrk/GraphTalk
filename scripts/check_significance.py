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

Main-sweep `exact` rows: one row per condition, `bound="excluded"` (kept as
the literal value for schema stability, even though nothing is excluded
any more -- see below). A non-terminating response's `exact`/`primary` are
forced to 0.0 in `graphtalk.analysis.build_frame` itself, unconditionally
-- never trusting whatever the truncated-text answer-extractor happened to
land on, and never dropped from the pairing either. This replaced an
earlier three-way `excluded`/`best_case`/`worst_case` bound system that
bracketed the uncertainty about a truncated response's true outcome
instead of resolving it; that bracket is gone because there is no longer
any uncertainty left to bracket -- every non-terminating row is now
scored, deterministically, as wrong, by design, everywhere downstream.
`graphtalk.analysis.build_frame`'s `truncated_but_correct` column is the
one place that discarded information survives: `True` when the forced
row's raw, pre-override `exact` was actually a hit.

Forcing every non-terminating row to wrong is not free of bias: non-
termination itself responds to the primer condition (docs/sweep-
findings.md), so a condition that induces more truncation will now
mechanically look worse here in a way that partly reflects generation
length rather than reasoning quality. This is the flip side of the old
`excluded` bound's own bias (a condition that truncates on instances it
would have gotten wrong anyway looked artificially *better* there) --
resolved in the conservative direction rather than left open, per this
project's own choice, not a claim that the bias has vanished.
`n_forced_wrong_non_terminating` reports how many of a condition's rows
were forced this way, so that bias stays visible rather than silent. The
thinking arm's own `non_terminating` rows are a different question
entirely -- the outcome being tested there, never forced or dropped, and
`bound` is `not_applicable` for those rows (not the literal text `"n/a"`
-- that string is one of pandas' default NA tokens, so `pd.read_csv`
would silently turn it into `NaN` on every re-read, which is exactly the
bug this sentinel avoids).

`n_looped_on_correct_answer` and `high_non_termination_rate` are further
diagnostics, populated only on `bound == "excluded"` main-sweep rows
(blank elsewhere): the former counts non-terminating rows whose response
settled on the correct answer before getting cut off rather than
genuinely drifting (see `graphtalk.analysis.build_frame`'s
`looped_on_correct_answer` column -- closely related to, but not the same
computation as, `truncated_but_correct`, since one reads the *first*
stated value and the other the score of the *last*; they usually agree).
`high_non_termination_rate` flags a row where the forced-wrong share is
high enough that its accuracy number should be read with real caution --
not "not enough clean data" (nothing is dropped any more), but "a
meaningful part of what moved this number is truncation rate, not
reasoning" (default threshold 15%, `--low-power-threshold`).
`n_instances_missing` (every row) is the count of graph instances with
zero surviving pairs -- computed from the data, not a hardcoded total,
since the sweep's instance count changes with `--count`.

`bh_significant` corrects across the ~5 *independent* conditions within one
`(arm, group, bound)` family (`all` gets its own `.../derived` family, see
above). `is_derived_condition` marks which rows that is, so a reader can
tell the two families apart without parsing the `bh_family` string.
`bh_significant_global` is a second, additional correction across every
`excluded`/`not_applicable`-bound, non-derived row from the whole run
(**now spanning both `--metric exact` and `--metric mae` in one pass** --
see `--metric` below), excluding
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
  baseline). Now computed from every row, forced-wrong non-terminating
  ones included -- a model whose apparent near-ceiling accuracy partly
  depended on truncated rows being invisible will show a truer, slightly
  lower rate here than before this refactor, which is a correct
  consequence of no longer hiding them, not a regression.
- `headroom` -- `min(control_mean, 1 - control_mean)`, the theoretical max
  fraction of rows that could still flip. Alongside `near_ceiling` and
  `high_non_termination_rate`, this is what lets a reader tell
  "high_non_termination_rate *because of* near-ceiling headroom" apart
  from "despite real headroom left" -- the two flags alone can't
  distinguish those. Same populated scope as `near_ceiling`.
- `task_delta_min`/`task_delta_max` -- per-task point-estimate deltas
  (`instance_id` already encodes its task as a `/`-prefix, e.g.
  `edge_count/27`, so no extra join is needed), purely descriptive, no new
  hypothesis test and no added multiple-comparison burden. A pooled `delta`
  near zero with a wide `[task_delta_min, task_delta_max]` range means the
  condition helps on some tasks and hurts on others rather than doing
  nothing everywhere.
- `mde_delta`/`mde_realized_diff`/`mde_delta_negative`/
  `mde_realized_diff_negative`/`mde_power_target` -- computed automatically
  for every row where `bh_significant is False` and `bound in ("excluded",
  "not_applicable")`, at a fast-approximate preset (`n_replicates=50,
  n_perm=200, n_steps=5`) unless `--mde` asks for full precision
  (`200/500/8`, roughly 6-7x slower -- use for a final reported number, not
  routine runs) or `--no-mde` turns the whole thing off. The fast preset is
  benchmarked against full precision in `scripts/benchmark_mde.py`: mean/max
  `|delta|` difference stays within `graphtalk.significance
  .minimum_detectable_effect_clustered`'s own documented Monte Carlo noise
  floor (SE ~= 0.03) at several times the speed -- see that script and
  `--mde-fast`'s help text.
  `graphtalk.significance.minimum_detectable_effect_clustered` simulates
  the smallest true effect this row's data could have reliably detected,
  **in both directions** -- a candidate improvement (`mde_delta`, positive)
  and a candidate harm (`mde_delta_negative`, negative) are searched
  independently, since a near-ceiling control has very different headroom
  in each: almost no room to improve (often `mde_delta = None`, "exceeds
  1.0") but plenty of room to get worse (`mde_delta_negative` usually
  converges to a real value there). `mde_realized_diff`/
  `mde_realized_diff_negative` can come in on the far side of their
  respective `mde_delta`/`mde_delta_negative` on a `near_ceiling` row,
  which is itself a second, independent confirmation that a ceiling effect
  is suppressing detectability there. Blank wherever not computed --
  `bh_significant is True` (nothing to explain), or `--no-mde` was passed.

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

Non-terminating rows are **included**, matching the `exact` metric's own
treatment -- but a truncated response's extracted number is not a
meaningful "how close" signal any more than its extracted exact-match
answer is (see `graphtalk.analysis.build_frame`'s reasoning for
`exact`/`primary`), so its own `absolute_error`, real or missing, is never
used. Instead `_mae_imputation_table` substitutes the **median
`absolute_error` among genuinely-wrong (parsed, terminating) rows for the
same task**, computed once from the whole frame -- a principled "typical
wrong answer's error" stand-in, not a guess, and not the arbitrary
worst-case constant the old `best_case`/`worst_case` bracket would have
suggested (rejected for the same reason that bracket was retired for
`exact`: it isn't grounded in this data). Median rather than mean, since a
badly-wrong integer guess is heavy-tailed and a few large misses
shouldn't dominate the "typical" value. `n_mae_imputed` reports how many
of a cell's rows got this substitution.

Genuinely-*unparsed*-but-terminated rows (a complete response that never
stated a number) are unaffected by this refactor and remain excluded, via
`absolute_error.notna()` -- out of scope for the non-terminating-row
question this exists to answer.

No `near_ceiling` (a 0-1-accuracy concept), no `task_delta_min`/`max`
(redundant -- this mode doesn't pool across tasks to begin with). Reports
`mae_delta = control_mae - treatment_mae`, not the raw `treatment -
control` the underlying functions return -- lower error is better, the
opposite sign convention from `exact`'s "higher is better", so this flip
keeps a positive number meaning "helped" in both modes.

  PYTHONPATH=. .venv/bin/python scripts/check_significance.py \
      --frame analysis/sweep_frame.csv --metric mae

**`--confirmatory-config`**: an optional JSON file naming the (arm, model,
condition, metric) cells decided *before a sweep's results are seen* to be
the ones a claim will actually rest on -- everything else is exploratory,
labelled as such rather than silently sharing a multiplicity budget with
cells chosen with the benefit of hindsight. Format:

```json
{"confirmatory": [
    {"model": "gemma4-12b", "condition": "degree"},
    {"model": "qwen3-8b", "condition": "filler", "metric": "mae"}
]}
```

`arm` and `metric` are optional per entry (default: any arm, `"exact"`);
`model` matches the `group` column, so `"pooled across all models"` is a
valid value too. Every record gets a `hypothesis_type` column
(`"confirmatory"`/`"exploratory"`) and `_apply_global_bh` corrects the two
groups *separately* -- a smaller, stricter confirmatory family, and an
exploratory family that's still reported (not suppressed) but explicitly
labelled hypothesis-generating rather than confirmed. Omit `--confirmatory-
config` and nothing changes: every record's `hypothesis_type` is blank and
`_apply_global_bh` corrects everything together in one family, exactly as
it always has. Committing the config file *before* running the sweep it
applies to is a discipline this flag makes possible to follow, not
something the code itself can enforce -- there is no way to check "was
this file's git history older than the run" from inside a Python process
reading it after the fact.
"""

import argparse
import json

import pandas as pd

from graphtalk import analysis
from graphtalk import primers
from graphtalk import significance

CONTROL = "none"
_KEYS = ["model", "instance_id", "style", "node_naming"]
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


def _load_confirmatory_config(path: str | None) -> set | None:
  """Loads `--confirmatory-config`'s JSON file (see the module docstring
  for its format) into a set of `(arm, group, condition, metric)` tuples,
  `arm`/`metric` normalized to `"*"` (any) when an entry omits them.
  `None` -- not an empty set -- when `path` is falsy: distinguishes "no
  config, every row is unlabelled" from "a config was loaded and happens
  to list nothing confirmatory", which would otherwise both look like
  "nothing is confirmatory" to `_hypothesis_type` but mean very different
  things (the second should still label every row `"exploratory"`, the
  first should leave `hypothesis_type` blank).
  """
  if not path:
    return None
  with open(path) as handle:
    raw = json.load(handle)
  return {
      (entry.get("arm", "*"), entry["model"], entry["condition"],
       entry.get("metric", "exact"))
      for entry in raw["confirmatory"]
  }


def _hypothesis_type(
    confirmatory: set | None, arm: str, group: str, condition: str, metric: str,
) -> str | None:
  """`"confirmatory"` if `(arm, group, condition, metric)` (or its
  any-arm form, `("*", group, condition, metric)`) is in `confirmatory`,
  `"exploratory"` if a config was loaded but this cell isn't in it, `None`
  if `confirmatory is None` (no `--confirmatory-config` was given --
  today's fully-exploratory-by-omission behavior, left blank rather than
  spelled out as `"exploratory"` for every row, so an unlabelled run and a
  run with an empty confirmatory list remain visibly different in the
  output)."""
  if confirmatory is None:
    return None
  if (arm, group, condition, metric) in confirmatory:
    return "confirmatory"
  if ("*", group, condition, metric) in confirmatory:
    return "confirmatory"
  return "exploratory"


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


def _count_forced_wrong_non_terminating(raw_frame: pd.DataFrame, condition: str) -> int:
  """How many `condition`-or-`CONTROL` rows in `raw_frame` are
  `non_terminating` -- the size of the confound `graphtalk.analysis
  .build_frame` resolves by forcing these rows to score as wrong, kept
  visible rather than silently absorbed into the pooled `delta`."""
  relevant = raw_frame[raw_frame["condition"].isin([CONTROL, condition])]
  return int((relevant["failure_type"] == "non_terminating").sum())


def _count_looped_on_correct_answer(raw_frame: pd.DataFrame, condition: str) -> int:
  """How many of those same `non_terminating` rows settled on the correct
  answer before getting cut off, rather than genuinely drifting -- same
  scope as `_count_forced_wrong_non_terminating`, see its docstring."""
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
  control_mean = frame.loc[frame["condition"] == CONTROL, metric].mean()
  if pd.notna(control_mean):
    near_ceiling = bool(
        control_mean > args.near_ceiling_threshold
        or control_mean < 1 - args.near_ceiling_threshold
    )
    # The theoretical max fraction of rows that could still flip --
    # `near_ceiling` alone can't say whether a `high_non_termination_rate`
    # row is that *because* of that headroom limit or despite having real
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
    n_forced_wrong = _count_forced_wrong_non_terminating(raw_frame, condition)
    if bound == "excluded":
      n_looped = _count_looped_on_correct_answer(raw_frame, condition)
      denom = perm["n_pairs"]  # every row is paired now; nothing is dropped
      high_non_termination_rate = (
          (n_forced_wrong / denom) > args.high_non_termination_threshold
          if denom else False
      )
    else:
      # Thinking arm (`bound == "not_applicable"`): `n_forced_wrong` here is
      # really the non-termination count itself, the outcome being tested,
      # not a confound to flag against -- so these stay inapplicable.
      n_looped, high_non_termination_rate = None, None
    n_instances_missing = total_clusters_possible - perm["n_clusters"]
    rows.append((
        condition, perm, boot, n_forced_wrong, n_looped,
        high_non_termination_rate, n_instances_missing, task_delta_min,
        task_delta_max, control, treatment, cluster_ids,
        _is_derived_condition(condition),
    ))

  if not rows:
    return
  mde_eligible = args.mde
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
    for (condition, perm, boot, n_forced_wrong, n_looped,
         high_non_termination_rate, n_missing, task_delta_min, task_delta_max,
         control, treatment, cluster_ids, is_derived), sig in zip(
             group_rows, reject):
      ci = f"[{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}]"
      print(f"    {condition:<12}{perm['n_clusters']:>11}"
            f"{perm['observed_diff']:>+10.3f}{ci:>22}"
            f"{perm['p_value']:>10.4f}  {'yes' if sig else 'no'}")
      mde_delta = mde_realized = mde_power_target = None
      mde_delta_negative = mde_realized_negative = None
      if mde_eligible and not sig:
        ci_width = boot["ci_high"] - boot["ci_low"]
        mde_seed = f"{args.seed}:{arm}:{label}:{bound}:{condition}:mde"
        mde = significance.minimum_detectable_effect_clustered(
            control, treatment, cluster_ids, initial_hi=max(0.05, ci_width),
            alpha=args.alpha, power_target=args.mde_power_target,
            n_replicates=args.mde_replicates, n_perm=args.mde_n_perm,
            n_steps=args.mde_n_steps, seed=mde_seed,
        )
        mde_delta, mde_realized = mde["delta"], mde["realized_diff"]
        mde_power_target = mde["power_target"]
        mde_delta_negative = mde["delta_negative"]
        mde_realized_negative = mde["realized_diff_negative"]
        print(f"      MDE: delta={mde_delta} realized={mde_realized} "
              f"({mde['note'] or 'ok'})  "
              f"delta_negative={mde_delta_negative} "
              f"realized_negative={mde_realized_negative} "
              f"({mde['note_negative'] or 'ok'})")
      records.append({
          "arm": arm,
          "group": label,
          "metric": metric,
          "condition": condition,
          "hypothesis_type": _hypothesis_type(
              args.confirmatory, arm, label, condition, metric
          ),
          "is_derived_condition": is_derived,
          "bound": bound,
          "bh_family": bh_family,
          "n_pairs": perm["n_pairs"],
          "n_clusters": perm["n_clusters"],
          "n_instances_missing": n_missing,
          "n_forced_wrong_non_terminating": n_forced_wrong,
          "n_looped_on_correct_answer": n_looped,
          "high_non_termination_rate": high_non_termination_rate,
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
          "mde_delta_negative": mde_delta_negative,
          "mde_realized_diff_negative": mde_realized_negative,
          "mde_power_target": mde_power_target,
      })

  _emit(independent_rows, "")
  _emit(derived_rows, "/derived")


def _apply_global_bh(records: list, q: float) -> None:
  """The whole-table BH pass, mutating `records` in place.

  Eligible rows are every row whose `group` is not `"pooled across all
  models"` (excludes a row built from the same underlying pairs as its
  sibling per-model rows in the same arm -- not an independent hypothesis)
  and whose condition is not derived (excludes `all`, mechanically the
  union of degree/clustering/rwse -- see `_is_derived_condition` -- also
  not an independent hypothesis). `bound` no longer needs its own
  eligibility check here -- `_report` now only ever produces
  `bound="excluded"` (main sweep) or `bound="not_applicable"` (thinking
  arm) rows, both real, independently-interpretable results now that the
  best_case/worst_case sensitivity bracket has been retired (see the
  module docstring); there is no bracket row left to exclude. Called once
  per `main()` run, over every record collected regardless of which
  metric produced it (`exact`, `non_terminating`, `mae`) -- exact and mae
  are two lenses on the same underlying (model, condition) hypotheses, so
  they share one multiplicity budget rather than two separate ones.
  Ineligible rows get `bh_significant_global = None`, not `False`, so "not
  significant" and "not tested" stay distinguishable.

  Eligible rows are further split by `hypothesis_type` (see
  `--confirmatory-config` in the module docstring) before correcting --
  confirmatory and exploratory cells get their own, separately-corrected
  family, never pooled into one. When no config was given every record's
  `hypothesis_type` is `None`, which is then the *only* group, so this
  reduces to exactly one whole-table family -- today's behavior,
  unchanged.
  """
  eligible = [
      r for r in records
      if r["group"] != "pooled across all models"
      and not r["is_derived_condition"]
  ]
  by_hypothesis_type: dict = {}
  for r in eligible:
    by_hypothesis_type.setdefault(r["hypothesis_type"], []).append(r)
  for group_rows in by_hypothesis_type.values():
    reject_global = significance.benjamini_hochberg(
        [r["p_value"] for r in group_rows], q=q
    )
    for r, sig in zip(group_rows, reject_global):
      r["bh_significant_global"] = sig
  for r in records:
    r.setdefault("bh_significant_global", None)


def _mae_imputation_table(raw_frame: pd.DataFrame) -> dict[str, float]:
  """Per-task median `absolute_error` among genuinely-wrong (parsed,
  terminating) rows -- the value substituted for a non-terminating row's
  own `absolute_error` in `_mae_eligible_frame`.

  There is no trustworthy `absolute_error` to read directly off a
  non-terminating row: an unparsed truncation has none at all, and a
  parsed-but-truncated one carries the same "the abandoned text
  coincidentally looks informative" risk `graphtalk.analysis.build_frame`
  already rejects for `exact` -- so neither case uses the row's own value,
  even when one exists. Median, not mean: a wrong integer guess's error is
  heavy-tailed (a badly wrong `edge_count` guess can be off by dozens), and
  the median is the "typical wrong answer", not one a few large misses can
  dominate. Computed once from `raw_frame` as a whole, not per (model,
  condition) -- those cells are often under 30 wrong rows, too few for a
  stable estimate, and there is no evidence a condition changes the *size*
  of a wrong guess's error, only whether the model is right at all.
  """
  wrong = raw_frame[raw_frame["failure_type"] == "wrong"]
  return {
      task: wrong.loc[wrong["task"] == task, "absolute_error"].median()
      for task in _MAE_TASKS
  }


def _mae_eligible_frame(
    main_sweep_raw: pd.DataFrame, imputation_table: dict[str, float],
) -> pd.DataFrame:
  """Rows `--metric mae` can pair: one of `_MAE_TASKS`, and either a real
  `absolute_error` (a terminated, parsed row) or `non_terminating` (in
  which case `absolute_error` is overwritten with `imputation_table`'s
  value for that task -- see `_mae_imputation_table`). A genuinely
  *unparsed*-but-terminated row (a complete response that never stated a
  number) is excluded exactly as before this refactor -- out of scope for
  the non-terminating-row question `_mae_imputation_table` exists to
  answer.
  """
  eligible = main_sweep_raw[
      main_sweep_raw["task"].isin(_MAE_TASKS)
      & (main_sweep_raw["absolute_error"].notna()
         | (main_sweep_raw["failure_type"] == "non_terminating"))
  ].copy()
  is_non_terminating = eligible["failure_type"] == "non_terminating"
  eligible.loc[is_non_terminating, "absolute_error"] = (
      eligible.loc[is_non_terminating, "task"].map(imputation_table)
  )
  return eligible


def _report_mae(
    frame: pd.DataFrame, raw_frame: pd.DataFrame, label: str, task: str,
    args, records: list,
) -> None:
  """`--metric mae`'s counterpart to `_report`, scoped to one
  `(label, task)` pair.

  Deliberately does not call `_report`: its `near_ceiling`/`task_delta`/MDE
  logic is all specific to the 0-1 `exact` metric and doesn't translate to
  an unbounded error metric (see the module docstring). The underlying
  clustering/permutation/bootstrap/BH primitives are metric-agnostic and
  are reused as-is; only the orchestration and the `mae_delta` sign flip
  are new. `frame` is expected to already have non-terminating rows'
  `absolute_error` imputed (see `_mae_eligible_frame`) -- this function
  does not know or care which rows were imputed, only how many
  (`n_mae_imputed`, for transparency in the output).
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
    n_imputed = _count_forced_wrong_non_terminating(raw_frame, condition)
    rows.append((condition, perm, boot, n_imputed, _is_derived_condition(condition)))

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
    for (condition, perm, boot, n_imputed, is_derived), sig in zip(group_rows, reject):
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
          "hypothesis_type": _hypothesis_type(
              args.confirmatory, "main_sweep", label, condition, "mae"
          ),
          "is_derived_condition": is_derived,
          "bound": "excluded",
          "bh_family": bh_family,
          "n_pairs": perm["n_pairs"],
          "n_clusters": perm["n_clusters"],
          "n_mae_imputed": n_imputed,
          "mae_delta": mae_delta,
          "ci_low": ci_low,
          "ci_high": ci_high,
          "p_value": perm["p_value"],
          "bh_significant": sig,
      })

  _emit(independent_rows, "")
  _emit(derived_rows, "/derived")


def _resolve_mde_settings(
    mde: bool, no_mde: bool, mde_replicates, mde_n_perm, mde_n_steps,
) -> dict:
  """Phase 1.3.3: MDE runs by default; `no_mde` is the only way to turn it
  off. `mde` (the parsed `--mde` flag) only chooses full precision
  (`n_replicates=200, n_perm=500, n_steps=8`) over the default fast preset
  (`50/200/5`) -- it does not itself enable MDE, which is already on. An
  explicit `mde_replicates`/`mde_n_perm`/`mde_n_steps` (not `None`) always
  wins over either preset. Returns the four resolved values as a dict
  (`mde` here means "should MDE run at all", the trigger `_report` reads --
  not "was full precision requested", which is `mde`'s *input* meaning;
  the rename happens across this call on purpose, so the caller can't
  confuse the two)."""
  full_precision = mde
  return {
      "mde": not no_mde,
      "mde_replicates": mde_replicates if mde_replicates is not None
                         else (200 if full_precision else 50),
      "mde_n_perm": mde_n_perm if mde_n_perm is not None
                    else (500 if full_precision else 200),
      "mde_n_steps": mde_n_steps if mde_n_steps is not None
                     else (8 if full_precision else 5),
  }


def _apply_filter(frame: pd.DataFrame, filter_expr: str | None) -> pd.DataFrame:
  """`--filter`'s pandas `DataFrame.query()` expression applied to
  `frame`, or `frame` unchanged when no filter was given. Formalizes what
  used to be an undocumented manual pre-filter step (see the module
  docstring's `--filter` help text and `analysis/README.md`) into one
  reproducible, testable operation -- a `--filter "style == 'zero_shot'"`
  run and a manually-pre-filtered `frame.query(...)` call passed straight
  into the rest of `main()` are now provably the same thing, not two
  routes that happen to usually agree.
  """
  if not filter_expr:
    return frame
  return frame.query(filter_expr)


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
  parser.add_argument("--high-non-termination-threshold", type=float, default=0.15,
                       help="flag a row's `high_non_termination_rate` "
                            "column when the fraction of its pairs that "
                            "are non-terminating (and so forced to score "
                            "as wrong -- see the module docstring) "
                            "exceeds this")
  parser.add_argument("--near-ceiling-threshold", type=float, default=0.95,
                       help="flag a row's `near_ceiling` column when its "
                            "control condition's mean metric is above this "
                            "(or below 1 minus this)")
  parser.add_argument("--mde", action="store_true",
                       help="upgrade the automatic MDE (see --no-mde) from "
                            "the fast-approximate default to full precision "
                            "(n_replicates=200, n_perm=500, n_steps=8 "
                            "unless overridden below) -- roughly 6-7x the "
                            "fast preset's time. Use for a final reported "
                            "number, not routine runs")
  parser.add_argument("--no-mde", action="store_true",
                       help="disable MDE entirely. Since Phase 1.3.3, MDE "
                            "is computed automatically (fast-approximate "
                            "preset) for every non-significant "
                            "excluded/n-a-bound row -- pass this to skip "
                            "it, e.g. for a quicker exploratory run")
  parser.add_argument("--mde-fast", action="store_true",
                       help="explicit spelling of the default fast-"
                            "approximate MDE preset (n_replicates=50, "
                            "n_perm=200, n_steps=5, benchmarked in "
                            "scripts/benchmark_mde.py against --mde's full "
                            "precision: mean/max |delta| difference stays "
                            "within graphtalk.significance"
                            ".minimum_detectable_effect_clustered's own "
                            "documented Monte Carlo noise floor, SE ~= "
                            "0.03, at several times the speed). Doesn't "
                            "change anything on its own since it's already "
                            "the default; kept for scripts that want to be "
                            "explicit rather than rely on it")
  parser.add_argument("--mde-power-target", type=float, default=0.8)
  parser.add_argument("--mde-replicates", type=int, default=None,
                       help="simulated trials per candidate effect size -- "
                            "default depends on --mde (200 vs the fast "
                            "preset's 50) unless set explicitly here")
  parser.add_argument("--mde-n-perm", type=int, default=None,
                       help="permutations per simulated trial's inner test "
                            "(lower than --n-perm: averaged over many "
                            "trials, so per-trial precision matters less) "
                            "-- default depends on --mde (500 vs the fast "
                            "preset's 200) unless set explicitly here")
  parser.add_argument("--mde-n-steps", type=int, default=None,
                       help="bisection steps in the MDE search -- default "
                            "depends on --mde (8 vs the fast preset's 5) "
                            "unless set explicitly here")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--filter", default=None,
                       help="optional pandas `DataFrame.query()` expression "
                            "applied to `--frame` before anything else runs "
                            "-- e.g. --filter \"model == 'gemma4-12b'\" or "
                            "--filter \"style == 'zero_shot'\". Formalizes "
                            "what used to be an undocumented manual "
                            "pre-filter step for any 'what holds up under "
                            "subset X' question (see analysis/README.md); "
                            "one flag is now the whole reproduction "
                            "recipe instead of a remembered pandas snippet. "
                            "`bh_family`/`bh_significant_global`'s "
                            "multiplicity correction is computed only over "
                            "the filtered rows -- a `--filter`'d run is a "
                            "genuinely different, smaller-family question "
                            "than the unfiltered one, not a display-only "
                            "slice of it")
  parser.add_argument("--confirmatory-config", default=None,
                       help="optional JSON file naming (arm, model, "
                            "condition, metric) cells pre-registered as "
                            "confirmatory before a sweep's results were "
                            "seen -- see the module docstring for the "
                            "format. Every record gets a `hypothesis_type` "
                            "column and confirmatory/exploratory cells are "
                            "BH-corrected in separate families; omit this "
                            "flag and nothing changes from today's "
                            "single-family behavior")
  parser.add_argument("--out", default=None,
                       help="optional path to write every row printed above "
                            "as CSV -- analysis.tagged_path suffixes it for a "
                            "non-integer node_naming scheme automatically")
  args = parser.parse_args()

  args.confirmatory = _load_confirmatory_config(args.confirmatory_config)

  if args.no_mde and args.mde:
    parser.error("--mde and --no-mde are mutually exclusive")
  resolved = _resolve_mde_settings(
      mde=args.mde, no_mde=args.no_mde, mde_replicates=args.mde_replicates,
      mde_n_perm=args.mde_n_perm, mde_n_steps=args.mde_n_steps,
  )
  args.mde = resolved["mde"]
  args.mde_replicates = resolved["mde_replicates"]
  args.mde_n_perm = resolved["mde_n_perm"]
  args.mde_n_steps = resolved["mde_n_steps"]

  frame = pd.read_csv(args.frame)
  frame = _apply_filter(frame, args.filter)
  if frame.empty and args.filter:
    print(f"--filter {args.filter!r} matched no rows")
    return
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
    # main_sweep_raw is what n_forced_wrong_non_terminating/
    # n_instances_missing count against; main_sweep IS main_sweep_raw now --
    # `graphtalk.analysis.build_frame` already forces every non_terminating
    # row's `exact`/`primary` to 0.0, so there is nothing left to drop or
    # bracket here (see the module docstring; this used to be three bounds,
    # `excluded`/`best_case`/`worst_case`).
    main_sweep = main_sweep_raw
    think = frame[frame["is_think"]]

    print("=" * 78)
    print("Main sweep: accuracy (exact) vs `none`, pooled across task + style")
    for model_family, group in main_sweep.groupby("model_family"):
      _report(group, group, "exact", model_family, args, "main_sweep",
              records, bound="excluded")
    _report(main_sweep, main_sweep_raw, "exact", "pooled across all models",
            args, "main_sweep", records, bound="excluded")

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
    # Computed once, from every model pooled together -- a per-(model, task)
    # slice is often too small (<30 wrong rows) for a stable median; see
    # `_mae_imputation_table`'s own docstring.
    imputation_table = _mae_imputation_table(main_sweep_raw)
    for model_family, raw_group in main_sweep_raw.groupby("model_family"):
      eligible = _mae_eligible_frame(raw_group, imputation_table)
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
