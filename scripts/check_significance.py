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

Pooling across task (and, for "pooled across all models" rows, across
model too) means the same graph recurs many times in one pooled sample, so
`(model, graph_index)` is threaded through as a *cluster* id: the
permutation test flips and the bootstrap resamples whole clusters, not
individual rows, so correlated rows sharing a graph don't get counted as
independent evidence (see `graphtalk/significance.py`'s module docstring).

The cluster id is the **graph number**, not the whole `instance_id`. An
`instance_id` is `"<task>/<index>"`, and `node_count/7` and `edge_count/7`
are the same graph -- same nodes, same edges, byte-identical encoding in
`prompts.jsonl`, differing only in the question appended after it. Keying
clusters on the full string therefore gave every cluster exactly one member
(`n_clusters == n_pairs` on every row of the report this replaced), so the
clustering corrected for nothing at all. That went unnoticed because this
docstring used to justify clustering by repetition *across prompt styles*,
which was true until the `zero_cot` purge left `zero_shot` as the only
style; nothing was then updated to point at the six-tasks-per-graph
repetition that remained. See `_paired_values` and `_within_graph_icc`.

The cluster id carries `model`, not just the graph index -- otherwise a
"pooled across all models" row would merge the same graph number from four
different model families into one cluster, assuming those families'
errors on that graph correlate as strongly as one model's own repeated
answers to it do. They may not, so only same-model rows sharing a graph
are treated as correlated.

`within_graph_icc` reports, per row, how much correlation this is actually
correcting for, so the choice of unit is answerable from the data. On the
current sweep it averages about -0.01 -- the six tasks on one graph move
essentially independently -- and is clearly positive only on `rwse`, which
is also the condition whose p-value moves most under the corrected key.
The cluster-robust test is the right default because which cells carry
correlation isn't knowable in advance, not because every cell does.

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
`high_non_termination_rate` flags a row where the forced-wrong share of
its *pairs* (either side non-terminating, matching `_paired_values`'s own
join -- not raw rows summed across both condition sides, which would
double-count a pair with both sides non-terminating and isn't a fraction
of `n_pairs`; see `_count_forced_wrong_pairs`) is high enough that its
accuracy number should be read with real caution -- not "not enough clean
data" (nothing is dropped any more), but "a meaningful part of what moved
this number is truncation rate, not reasoning" (default threshold 15%,
`--high-non-termination-threshold`).
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


def _graph_index(instance_id: str) -> str:
  """The graph number out of an `"<task>/<index>"` `instance_id`.

  The same index under different tasks is the *same graph* -- same nodes,
  same edges, byte-identical encoding in `prompts.jsonl` -- asked a
  different question. So this, not the whole `instance_id`, is the unit
  rows can be correlated within; see `_paired_values`. Raises rather than
  guessing on an id that doesn't carry a task prefix, since silently
  treating the whole string as an index would reintroduce exactly the
  one-member-per-cluster no-op this function exists to prevent.
  """
  task, sep, index = instance_id.partition("/")
  if not sep or not index:
    raise ValueError(
        f"instance_id {instance_id!r} has no '<task>/<index>' shape -- "
        f"cannot derive the graph index the cluster id needs"
    )
  return index


def _within_graph_icc(control, treatment, cluster_ids) -> float | None:
  """One-way random-effects ICC of the paired differences within a cluster.

  Reports how much correlation the clustering in `_paired_values` is
  actually correcting for, so the choice of cluster unit is answerable from
  the data rather than from a docstring. Roughly: the share of the paired
  differences' variance that lives *between* graphs rather than within one.
  ~0 means the six tasks on a graph move independently and clustering
  changes nothing; clearly positive means they move together and an
  unclustered test would overstate its evidence.

  `None` when every cluster holds one pair (nothing to measure) or the
  variance components are degenerate (a cell where no pair disagrees).
  Descriptive only -- no hypothesis test, no multiple-comparison burden,
  exactly like `task_delta_min`/`task_delta_max`.
  """
  by_cluster: dict = {}
  for cluster_id, c, t in zip(cluster_ids, control, treatment):
    by_cluster.setdefault(cluster_id, []).append(t - c)
  sizes = [len(v) for v in by_cluster.values()]
  if len(by_cluster) < 2 or max(sizes) < 2:
    return None
  k = sum(sizes) / len(sizes)
  means = [sum(v) / len(v) for v in by_cluster.values()]
  grand = sum(means) / len(means)
  ms_between = k * sum((m - grand) ** 2 for m in means) / (len(means) - 1)
  within = [
      sum((d - m) ** 2 for d in v) / (len(v) - 1)
      for v, m in zip(by_cluster.values(), means) if len(v) > 1
  ]
  ms_within = sum(within) / len(within)
  denominator = ms_between + (k - 1) * ms_within
  if denominator <= 0:
    return None
  return (ms_between - ms_within) / denominator


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
  `(model, graph_index)` pulled back out of the joined index -- the unit
  `paired_permutation_test_clustered`/`cluster_bootstrap_ci_clustered` need,
  since the *pairing* key has to include style/node_naming to be unique but
  the *correlation* those two functions correct for lives at the
  per-model-per-graph level. `model` stays part of the cluster id so a
  pooled-across-models call doesn't merge the same graph number from
  different model families into one cluster -- a much stronger correlation
  assumption than intended, and one this script has deliberately declined to
  make since the second-pass audit.

  **`graph_index`, not `instance_id`.** `instance_id` is `"<task>/<index>"`,
  so clustering on it treats `node_count/7` and `edge_count/7` as unrelated.
  They are the same graph: identical node set, identical edge list,
  byte-identical encoding in `prompts.jsonl`, differing only in the question
  appended at the end. Keying on the whole string gave every cluster exactly
  one member on the current data (`n_clusters == n_pairs` on every committed
  row), which made the clustering machinery a no-op -- it corrected for
  nothing at all. That went unnoticed because the docstrings justified
  clustering by repetition *across prompt styles*, which was real until the
  `zero_cot` purge left `zero_shot` as the only style; the six-tasks-share-a-
  graph repetition was never the one being corrected for. Splitting the task
  prefix off restores the intended unit: 30 clusters per model, 120 pooled.

  This is a conservative choice rather than a claim of strong dependence.
  `_within_graph_icc` measures the actual correlation per row (mean ICC on
  this data is about -0.01, i.e. essentially none, and positive only on
  `rwse`) -- exactly the cell whose p-value moves most under the fix. The
  cluster-robust test is the right default because which cells carry
  correlation is not knowable in advance, not because every cell does.
  """
  control = frame[frame["condition"] == CONTROL].set_index(_KEYS)[metric]
  treatment = frame[frame["condition"] == condition].set_index(_KEYS)[metric]
  joined = pd.concat(
      [control.rename("control"), treatment.rename("treatment")],
      axis=1, join="inner",
  )
  cluster_ids = list(zip(
      joined.index.get_level_values("model"),
      [_graph_index(i) for i in joined.index.get_level_values("instance_id")],
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


def _count_forced_wrong_pairs(frame: pd.DataFrame, condition: str) -> int:
  """How many of `condition`'s *paired* rows (control-and-treatment both
  present, same join `_paired_values` performs) are `non_terminating` on
  either side -- the numerator `high_non_termination_rate` actually needs,
  at the same granularity as its denominator (`n_pairs`).

  `_count_forced_wrong_non_terminating` (the `n_forced_wrong_non_terminating`
  diagnostic) counts raw rows summed across *both* condition sides
  independently -- a pair where both sides are non-terminating counts
  twice there, against a denominator (`n_pairs`) that counts it once, so
  the resulting ratio isn't a true fraction of pairs (and could exceed 1.0
  in the extreme). This function mirrors `_paired_values`'s own
  `set_index(_KEYS)` inner join exactly, so its count is guaranteed to
  match `n_pairs`' own unit -- one increment per pair, never per side.

  Reads `failure_type == "non_terminating"`, not a separate
  `non_terminating` boolean column -- `graphtalk.analysis.build_frame`
  guarantees the two agree exactly (`_failure_type` returns
  `"non_terminating"` if and only if that boolean is True), and every
  other helper in this module (`_count_forced_wrong_non_terminating`,
  `_count_looped_on_correct_answer`) already reads `failure_type` for the
  same reason, so a hand-built frame that carries one but not the other
  (every direct-`_report()`-call test fixture in `tests/test_significance
  .py` predates the `non_terminating` column existing at all) still works.
  """
  control = frame[frame["condition"] == CONTROL].set_index(_KEYS)
  treatment = frame[frame["condition"] == condition].set_index(_KEYS)
  control_nt = control["failure_type"] == "non_terminating"
  treatment_nt = treatment["failure_type"] == "non_terminating"
  joined = pd.concat(
      [control_nt.rename("control"), treatment_nt.rename("treatment")],
      axis=1, join="inner",
  )
  return int((joined["control"] | joined["treatment"]).sum())


def _paired_tasks(frame: pd.DataFrame, condition: str, metric: str) -> list:
  """The task label of each pair `_paired_values` returns, in the same
  order.

  Mirrors `_paired_values`'s own `set_index(_KEYS)` inner join exactly (the
  same guarantee `_count_forced_wrong_pairs` gives), so the result lines up
  element-for-element with its `control`/`treatment`/`cluster_ids`.

  A separate join rather than a fourth return value: six callers outside
  this function unpack `_paired_values` as a 3-tuple, and only
  `_task_delta_range` needs the task. It used to read the task off the
  cluster id's `instance_id` prefix, which stopped working when the cluster
  id became `(model, graph_index)` -- and would have failed *silently*,
  grouping by graph number instead of task, since `"7".split("/")[0]` is
  just `"7"`.
  """
  control = frame[frame["condition"] == CONTROL].set_index(_KEYS)[metric]
  treatment = frame[frame["condition"] == condition].set_index(_KEYS)[metric]
  joined = pd.concat(
      [control.rename("control"), treatment.rename("treatment")],
      axis=1, join="inner",
  )
  return [
      instance_id.partition("/")[0]
      for instance_id in joined.index.get_level_values("instance_id")
  ]


def _task_delta_range(control, treatment, tasks):
  """Per-task point-estimate deltas (`mean(treatment) - mean(control)`),
  descriptive only -- no new hypothesis test, no new multiple-comparison
  burden. `tasks` is `_paired_tasks`'s aligned task label per pair. Returns
  `(min, max)` across tasks -- surfaces whether a near-zero pooled delta is
  hiding tasks that actually disagree in direction, or genuinely reflects
  "nothing much happening on any task."
  """
  by_task: dict = {}
  for task, c, t in zip(tasks, control, treatment):
    by_task.setdefault(task, []).append(t - c)
  if not by_task:
    return None, None
  task_means = [sum(diffs) / len(diffs) for diffs in by_task.values()]
  return min(task_means), max(task_means)


def _near_threshold(p_value: float, group_rows: list, reject: list, args) -> bool:
  """Whether this p-value sits close enough to its own BH threshold that
  Monte Carlo noise, not the data, decides which side of it the row lands.

  A permutation p-value is `(hits + 1) / (n_perm + 1)`, so it can only take
  values on a grid of step `1 / (n_perm + 1)` and carries binomial sampling
  noise of roughly `sqrt(p (1 - p) / n_perm)` around its true value. The
  thresholds this project actually decides on are small -- the whole-table
  BH pass puts rank 1 of 100 at 0.0005, ten grid steps from zero at the old
  `n_perm=10_000` -- so a verdict can turn on a handful of individual random
  draws. That happened: the report's only whole-table-significant row
  cleared its threshold by 5e-8 against a grid resolution of 1e-4, and
  reversed under most other seeds.

  Flagged, not corrected: the honest reading of such a row is "on the
  boundary, unresolved at this `n_perm`", and the fix is more permutations
  (`--n-perm`), not a different verdict. Uses a 3-sigma band, so a flag
  means the noise plausibly spans the threshold rather than merely touching
  it.
  """
  if not group_rows or p_value is None or not (0.0 < p_value < 1.0):
    return False
  # This row's own BH threshold: (rank / m) * q at its rank in the family.
  ordered = sorted(row[1]["p_value"] for row in group_rows)
  m = len(ordered)
  rank = ordered.index(p_value) + 1
  threshold = (rank / m) * args.q
  standard_error = (p_value * (1 - p_value) / args.n_perm) ** 0.5
  grid_step = 1.0 / (args.n_perm + 1)
  return abs(p_value - threshold) < max(3 * standard_error, grid_step)


def _report(
    frame: pd.DataFrame, raw_frame: pd.DataFrame, metric: str, label: str,
    args, arm: str, records: list, bound: str = "not_applicable",
) -> None:
  print(f"\n  {label} [{bound}]")
  conditions = sorted(c for c in frame["condition"].unique() if c != CONTROL)
  rows = []
  # The number of *clusters* -- (model, graph_index) pairs, matching
  # `_paired_values`' own cluster granularity -- available to this group
  # before any exclusion. Read from the data, not hardcoded, so
  # `n_instances_missing` stays correct if the sweep's `--count` ever
  # changes (see module docstring). This has to track `_paired_values`
  # exactly: counting bare `instance_id` would undercount the baseline for
  # a pooled-across-models call (up to 4 clusters share one instance_id)
  # and make `n_instances_missing` go negative, while counting
  # `(model, instance_id)` -- correct before the cluster id dropped the
  # task prefix -- now overcounts it 6x (once per task on the same graph)
  # and would report 150 of 180 instances "missing" on a complete cell.
  total_clusters_possible = int(
      raw_frame.assign(_graph=raw_frame["instance_id"].map(_graph_index))
      [["model", "_graph"]].drop_duplicates().shape[0]
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
        control, treatment, _paired_tasks(frame, condition, metric)
    )
    within_graph_icc = _within_graph_icc(control, treatment, cluster_ids)
    n_forced_wrong = _count_forced_wrong_non_terminating(raw_frame, condition)
    if bound == "excluded":
      n_looped = _count_looped_on_correct_answer(raw_frame, condition)
      # Pair-level count, not `n_forced_wrong` -- that sums non-terminating
      # rows across both condition sides independently (double-counting a
      # pair where both sides are non-terminating), which isn't a true
      # fraction of `n_pairs`; see `_count_forced_wrong_pairs`'s docstring.
      n_forced_wrong_pairs = _count_forced_wrong_pairs(frame, condition)
      denom = perm["n_pairs"]  # every row is paired now; nothing is dropped
      high_non_termination_rate = (
          (n_forced_wrong_pairs / denom) > args.high_non_termination_threshold
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
        task_delta_max, within_graph_icc, control, treatment, cluster_ids,
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
         within_graph_icc, control, treatment, cluster_ids,
         is_derived), sig in zip(group_rows, reject):
      ci = f"[{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}]"
      near_threshold = _near_threshold(
          perm["p_value"], group_rows, reject, args
      )
      # Recomputed, not carried through `rows`: identical by construction to
      # the string the permutation/bootstrap calls above were seeded with
      # (same f-string, same inputs), and recording it is what lets a reader
      # reproduce this exact p-value rather than a differently-seeded one.
      seed_for_record = f"{args.seed}:{arm}:{label}:{bound}:{condition}"
      print(f"    {condition:<12}{perm['n_clusters']:>11}"
            f"{perm['observed_diff']:>+10.3f}{ci:>22}"
            f"{perm['p_value']:>10.4f}  {'yes' if sig else 'no'}"
            f"{'  <-- within MC noise of its threshold' if near_threshold else ''}")
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
          "within_graph_icc": within_graph_icc,
          "delta": perm["observed_diff"],
          "ci_low": boot["ci_low"],
          "ci_high": boot["ci_high"],
          "p_value": perm["p_value"],
          "n_perm": args.n_perm,
          "seed": seed_for_record,
          "bh_significant": sig,
          "near_threshold": near_threshold,
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
    p_values = [r["p_value"] for r in group_rows]
    reject_global = significance.benjamini_hochberg(p_values, q=q)
    # The whole-table thresholds are the small ones -- rank 1 of 100 lands
    # at 0.0005 -- so this, not the per-family pass, is where a verdict can
    # turn on a handful of random draws. It is exactly where it did: the
    # report this replaced had its only globally-significant row clear
    # 0.00050000 with p=0.00049995, a margin of 5e-8 against a p-value grid
    # of 1e-4. `near_threshold` is OR-ed rather than overwritten, so a row
    # already flagged against its per-family threshold stays flagged.
    ordered = sorted(p_values)
    m = len(ordered)
    for r, sig in zip(group_rows, reject_global):
      r["bh_significant_global"] = sig
      p_value = r["p_value"]
      # `n_perm` is absent on a hand-built record (several tests construct
      # these directly). The flag is a diagnostic, not part of the
      # correction, so skip it rather than making every caller supply the
      # field -- `bh_significant_global` above is unaffected either way.
      n_perm = r.get("n_perm")
      if not n_perm or not (0.0 < p_value < 1.0):
        continue
      threshold = ((ordered.index(p_value) + 1) / m) * q
      standard_error = (p_value * (1 - p_value) / n_perm) ** 0.5
      grid_step = 1.0 / (n_perm + 1)
      r["near_threshold"] = bool(
          r.get("near_threshold")
          or abs(p_value - threshold) < max(3 * standard_error, grid_step)
      )
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

  Raises if any `_MAE_TASKS` task has zero genuinely-wrong rows to impute
  from (e.g. an aggressive `--filter`, or a hypothetical future model
  that's simply never wrong on some task). `pandas.Series.median()` on an
  empty selection is silently `NaN`, not an error -- left unguarded, that
  `NaN` would propagate into every non-terminating row's `absolute_error`
  for that task and corrupt the permutation test (a `NaN` participates in
  `sum()`/`mean()` calls as `NaN`, poisoning the whole cell) with nothing
  visibly wrong until someone notices garbage output far downstream.
  Raising here instead is loud and immediate, at the one place the actual
  cause is knowable.
  """
  wrong = raw_frame[raw_frame["failure_type"] == "wrong"]
  table = {}
  for task in _MAE_TASKS:
    values = wrong.loc[wrong["task"] == task, "absolute_error"]
    if values.empty:
      raise ValueError(
          f"no wrong rows for task {task!r} to impute a non-terminating "
          f"row's absolute_error from -- see _mae_imputation_table's "
          f"docstring"
      )
    table[task] = values.median()
  return table


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
      near_threshold = _near_threshold(
          perm["p_value"], group_rows, reject, args
      )
      print(f"    {condition:<12}{perm['n_clusters']:>11}"
            f"{mae_delta:>+10.3f}{ci:>22}"
            f"{perm['p_value']:>10.4f}  {'yes' if sig else 'no'}"
            f"{'  <-- within MC noise of its threshold' if near_threshold else ''}")
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
          "n_perm": args.n_perm,
          "seed": f"{args.seed}:mae:{label}:{task}:{condition}",
          "bh_significant": sig,
          "near_threshold": near_threshold,
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
  parser.add_argument("--n-perm", type=int, default=200_000,
                       help="permutations for the pooled p-value. A "
                            "permutation p-value lands on a grid of step "
                            "1/(n_perm+1), so at the old 10,000 default the "
                            "whole-table BH threshold (0.0005 at rank 1 of "
                            "100) sat five grid steps from zero and verdicts "
                            "turned on single random draws -- the report's "
                            "only globally-significant row cleared its "
                            "threshold by 5e-8 and reversed under most other "
                            "seeds. 200,000 puts the grid an order of "
                            "magnitude below the smallest threshold in use; "
                            "`near_threshold` flags any row still inside the "
                            "noise band. Lower it for a fast development run, "
                            "not for a reported number")
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
