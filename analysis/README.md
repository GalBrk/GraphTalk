# Measurement artefacts

Small, expensive-to-reproduce measurements that decisions in this project rest
on. They live in git because the reasoning in `docs/sweep-findings.md` and
`cluster/README.md` cites them, and because each cost GPU time that a reader
should not have to spend again to check the claim.

| file | what it is |
|---|---|
| `budget-gemma4-e4b.jsonl` | 24 `zero_shot` prompts spanning all six tasks, generated at a 2048 cap with exact `n_new_tokens` per row. The measurement that set `MAX_NEW_TOKENS["zero_shot"]`: at 64 tokens this model scored 3/24, at 2048 it scored 24/24 with no row reaching the cap. |
| `budget-qwen3-8b.jsonl` | the same 24 prompts on Qwen3-8B with `enable_thinking=False` |
| `budget-qwen3-8b-THINKING.jsonl` | three of those rows with thinking **on**, kept as the before-picture: 1179 mean tokens on a `node_count` question the same model answers in 105 without |
| `truncated_keys.json` | historical record of the **271** thinking-arm rows that were hand-labelled non-terminating. No longer consulted: every tracked row now carries `hit_cap`. Kept because it is the provenance of a claim, and because two of its rows turn out to have terminated. |
| `sweep_frame.csv` | one row per scored response over the whole tracked sweep, 10,080 rows. |
| `failure_sample.csv` | the stratified manual-inspection sample, with full response text. **Stale — do not cite; see below.** |
| `significance_report.csv`/`.txt` | `scripts/check_significance.py --metric both` (the default)'s pooled permutation/bootstrap/BH-corrected results, one row per (arm, group, metric, condition) -- `exact`, `mae`, and the thinking arm's `non_terminating` all in one file, sharing **one** multiplicity-correction budget. See "Phase 1.1: multiplicity & scope fixes" below. |
| `significance_report_mae.csv` | `check_significance.py --metric mae`'s own, **separately**-corrected reading of the same `mae` rows -- its `bh_significant_global` answers "does this survive correction among just the 72 `mae` tests", a different (weaker-family, easier-to-clear) question than `significance_report.csv`'s "does this survive correction among all 192 `exact`+`mae`+`non_terminating` tests". The two can and do disagree on the same cell (see "Phase 1.1" below) -- read whichever question you're actually asking. Costs no GPU time to regenerate; reuses `sweep_frame.csv`. |
| `significance_report_exact_by_style.csv` | accuracy significance re-scoped to one style/arm at a time -- the main sweep's own accuracy plus the thinking arm's own accuracy. Now that `zero_shot` is the only prompt style, its "by style" scoping is a historical name; kept because it also carries the thinking-arm breakdown. **Stale — predates the `all`-exclusion/unified-correction fix below; do not cite until regenerated (planned for Phase 1.4's checked-in scoped-comparison script).** |
| `significance_report_mae_by_style.csv` | the `mae` metric re-scoped to the main sweep and the thinking arm. **Stale — same caveat as above.** |
| `significance_report_exact_zeroshot_and_thinking_mde.csv` | simulated minimum-detectable-effect for every non-significant main-sweep and thinking-arm accuracy cell. **Stale — same caveat as above.** |

## The two CSVs, and what is in them now

Both were regenerated on 2026-08-29, after the prompt-rewording re-run and with the
current `scoring._extract_boolean`. They supersede the versions committed at
`3545662`, which were written before both changes and understated parse rates
badly -- `unparsed` alone fell from 342 rows to 70.

`sweep_frame.csv` carries three columns the earlier version did not:

- `wording` -- `revised` / `unaffected`, marking which text produced the
  row: `filler` and `edge_existence` were reworded and every tracked row was
  regenerated against the new text, so `wording` now only distinguishes which
  cells that rewording touched.
- `non_terminating_source` -- `recorded` where the row carries its own `hit_cap`,
  `ground_truth_file` where `truncated_keys.json` is still the only record.
- `n_new_tokens` -- present only on regenerated rows; empty, not zero, elsewhere.

A fourth was added since, once `runs/<model>.got.jsonl` files exist:

- `node_naming` -- `integer` (the default, and the only value in the frame
  today) or `got`. `scripts/build_sweep_frame.py` refuses to build a frame
  mixing the two -- it raises rather than silently pooling `(instance_id,
  condition, style)` rows that only look like duplicates because the column
  distinguishing them didn't exist yet. Score each scheme separately and let
  each land at its own `.got.`-tagged CSV; see `README.md#node-naming`.

## The significance report

Regenerated with a corrected methodology, not just fresher `sweep_frame.csv`
input -- an audit of `scripts/check_significance.py` found it was pooling
correlated rows (the same graph instance recurs up to 12x per condition,
across styles and models) as if they were independent, which is anti-
conservative, and including `non_terminating` rows in the main-sweep
accuracy test, confounding "does this primer help reasoning" with "does
this primer change truncation rate" (`docs/DATA.md`/`runs/README.md`
already establish that non-terminating rows must be filtered before
reporting accuracy; this script wasn't).

The fix: `graphtalk/significance.py` gained `paired_permutation_test_clustered`/
`cluster_bootstrap_ci_clustered`, which flip/resample whole graph instances
rather than individual rows, and `check_significance.py` excludes
non-terminating rows from the main-sweep test (`n_excluded_non_terminating`
reports how many, per condition, so the confound's size stays visible)
and adds a uniqueness guard (`graphtalk.analysis.assert_unique_pairing_key`)
before pairing anything.

**The effect was not cosmetic**: three `bh_significant` flags flipped versus
the pre-fix regeneration, including `gemma4-12b`/`rwse` going from
"significant" (p=0.0072) to clearly not (p=1.0) -- exactly the kind of
false positive naive pooling produces. `n_pairs` (raw rows) and `n_clusters`
(real graph instances) are both reported now precisely so this gap stays
visible rather than being averaged away again.

### A second pass: the cluster id, the exclusion bias, and one whole-table view

A follow-up audit of the same script found the fix above still left three
things unaddressed, each now covered by a new column rather than a code
change alone:

**The cluster id was `instance_id` alone**, so a "pooled across all models"
row merged the same graph number from all four model families into one
cluster -- a much stronger correlation assumption than intended (one
model's errors on a graph correlating with a different model's errors on
the same graph, not just with its own repeated answers). Fixed by keying
clusters on `(model, instance_id)` instead. Concretely: pooled `n_clusters`
went from 180 to 709-720 (one cluster per model per instance rather than one
per instance, full-stop). No `bh_significant` flag flipped from this change
alone in the current data, but the CIs it feeds are now correctly narrower,
not artificially wide from an assumption the data doesn't support.

**Excluding `non_terminating` rows is not free of bias either.** The first
fix already noted non-termination responds to the primer condition; dropping
those rows before pairing (`bound == "excluded"`, unchanged) can still make
a condition that happens to truncate on instances it would have gotten
wrong anyway look artificially better. Rather than trust the truncated-text
extractor's guess as a second point estimate, `best_case`/`worst_case` rows
now bracket the true unknown outcome by forcing every non-terminating row's
score to 1.0/0.0 -- a real range, not another single number. On
`gemma4-e4b`/`filler`, the row with the most exclusions (67 of 382 rows,
`low_power = True`): `excluded` reports -0.038, and the bracket is
[-0.050, -0.031] -- `excluded` falls inside it, as it should, but the range
itself is the more honest thing to report. `n_looped_on_correct_answer`
(1 of those 67 rows) shows the extractor's guess would rarely have moved
that estimate much on this data -- most non-terminating rows really were
headed somewhere other than the right answer, not cut off one token short
of it.

**`bh_significant` never covered more than one `(arm, group, bound)`
family (~6 conditions) at a time.** A reader treating the whole table as one
5%-FDR-controlled set was getting a weaker guarantee than that reads as.
`bh_significant_global` adds one more correction across every `excluded`/
`not_applicable`-bound row in the whole run (excluding `best_case`/
`worst_case`, a sensitivity bracket rather than an independent hypothesis,
and excluding `pooled across all models` rows in both arms, which are built
from the same pairs as their sibling per-model rows rather than an
independent test). On the current data this flips three rows from
significant to not: `gemma4-e4b`/`filler`, `qwen3-14b`/`all`, and
`qwen3-14b`/`clustering` -- real findings under the narrower per-model
question, not under "does this matter anywhere in the whole study."

Both `bh_significant` and `bh_significant_global` are kept side by side --
neither replaces the other, since they answer different questions.

### A third pass: no-headroom rows, masked heterogeneity, and simulated power

`gemma4-12b` and `gemma4-e4b` sit at 96-99% main-sweep control accuracy --
almost no primer effect could show up there regardless of sample size, a
different problem from `low_power`'s "too much data excluded." **`near_ceiling`**
flags a row where the CONTROL condition's own mean is above 95% (or below
5%, a floor case -- e.g. `gemma4-e4b-think`'s 0% non-termination baseline),
on `excluded`/`not_applicable`-bound rows. Confirmed on the current data:
`True` for `gemma4-12b` and `gemma4-e4b` main-sweep, `False` for `qwen3-14b`
and `qwen3-8b` (86-90% control accuracy, real headroom left).

**A pooled `delta` can hide tasks that disagree in direction.**
`task_delta_min`/`task_delta_max` report each condition's per-task point
estimates (descriptive only -- no new hypothesis test, no added
multiple-comparison burden) alongside the pooled number. Pooled
`degree` (main sweep, all models): `delta=+0.033`, but
`[task_delta_min, task_delta_max] = [-0.004, +0.136]` -- one task is
essentially flat while another moves 13 points, which the pooled number
alone doesn't show.

**A `bh_significant=False` row could be a real null or just underpowered.**
`--mde` (opt-in, off by default -- took ~50 minutes on the full run)
runs `graphtalk.significance.minimum_detectable_effect_clustered`, a real
simulation (bootstrap-resample this row's own clusters, inject a candidate
effect as a Bernoulli draw at the shifted probability, rerun the paired
test, repeat) rather than a formula, for every non-significant
`excluded`/`not_applicable`-bound row (33 of 73 on the current data).
Reports `mde_delta` (the smallest effect this row's data could reliably
have detected) and `mde_realized_diff` (what that translated to on
average).

Confirmed on the real regenerated data: every extreme-`near_ceiling` row
(`gemma4-12b`/`gemma4-e4b` main sweep, 96-98% control accuracy) came back
`mde_delta=None` ("MDE exceeds 1.0") -- even the largest simulated effect
couldn't reliably reach 80% power there.

**That result is direction-specific, and reading it as "ceiling suppresses
all detectability" would be wrong.** `minimum_detectable_effect_clustered`
only searches positive `delta` (a candidate *improvement*), never negative
`delta` (a candidate *harm*). A near-ceiling control (e.g. 98% correct)
has almost no room for a case to flip from wrong to right, but plenty of
room -- 98% of cases -- to flip from right to wrong. So the `mde_delta=None`
result on `gemma4-12b`'s `all`/`clustering`/`degree`/`rwse` rows means only
"an improvement this large couldn't be confirmed"; it says nothing about
detecting harm, and indeed the same model's `components`/`filler` rows
(also `near_ceiling=True`) detected harm without difficulty
(`bh_significant=True`, delta -0.026 and -0.029). Where a `near_ceiling`
row comes back not significant, check its direction (the `delta` column's
sign) before citing the ceiling as the explanation: for a row already
moving in the harmful direction, the CI's harmful-side bound (`ci_low` for
a negative delta) is the more direct answer -- e.g. `gemma4-12b`/`clustering`
at `ci=[-0.009, 0.000]` bounds any hidden harm to under one accuracy point,
comparable in size to a genuine null rather than an underpowered test.
Extending the search to test both directions (or the observed row's own
direction) is possible but hasn't been done -- see "Not yet done" below.

Less-extreme `near_ceiling` rows (thinking-arm floor cases, ~0% baseline) converged to
small MDEs (~0.04) with `mde_realized_diff` tracking `mde_delta` closely.
Comfortably-powered rows (neither `near_ceiling` nor `low_power`) range
from `mde_delta=0.08` (`pooled across all models`/`rwse`, `n_clusters=709`)
to ~0.20 (single-model rows, `n_clusters≈180`) -- more data, smaller MDE,
as expected. One planned comparison didn't separate cleanly: every
`low_power=True` row in this data is *also* `near_ceiling=True` (all five
are `gemma4-e4b` main sweep), and the ceiling dominates completely enough
that none of them converged either -- `low_power` alone, holding ceiling
constant, isn't isolable in the current data, itself a real finding about
which constraint actually binds for `gemma4-e4b`'s main-sweep nulls.

One implementation bug caught before it reached the committed data: the
thinking arm's `bound` sentinel was originally the literal string `"n/a"`,
which is one of pandas' default NA tokens -- every re-read of the CSV
silently turned it into `NaN`, invisible to any `bound == "n/a"` filter
applied *after* loading the file (the file itself had the right text; only
`pd.read_csv` was wrong). Renamed to `"not_applicable"`, caught by a
sanity-check query on the regenerated file returning an empty result where
it shouldn't have.

**Not yet done**: `minimum_detectable_effect_clustered` searching both
directions (or the observed row's own direction) instead of positive
`delta` only, which would give a proper "smallest detectable harm" number
for a `near_ceiling` row moving in the harmful direction, rather than
requiring a manual CI read as above. Not implemented because the CI-based
read already answers the specific case that motivated it; worth doing if a
formal power-style number for harm detection is needed later.

### Phase 1.1: multiplicity & scope fixes

Two scope bugs, both about what counts as one multiple-comparison family.

**`all` was pooled into the same BH family as the conditions it's derived
from.** `all` is the union of `degree`/`clustering`/`rwse`
(`graphtalk/primers.CONDITIONS`), not a sixth independent manipulation --
correcting it alongside the five real conditions overstated how many
independent hypotheses were being tested (and let a mechanically-correlated
result dilute or inflate the real family's BH threshold). Every row now
carries `is_derived_condition`; `all` gets its own single-hypothesis
`bh_family` (suffixed `/derived`) and is excluded from `bh_significant_global`
by the same logic that already excluded `best_case`/`worst_case` and
`pooled across all models` rows. Consequence for the legend below: `all` can
only ever read • (significant at raw α, no family to be corrected against)
or `--`, never ✅ or ⚠️; a `pooled across all models` row can still be ⚠️
(it has its own 5-condition family) but, like `all`, is structurally
excluded from `bh_significant_global` and so can never be ✅ either -- an
existing rule this pass didn't change, but one an earlier version of this
table had marked inconsistently (`pooled across all models`/`filler` shown
✅; fixed below to ⚠️).

**`--metric exact` and `--metric mae` had two separate BH families when run
together informally.** They're two lenses on the same (model, condition)
hypotheses, not two independent questions, so treating them as separate
families understated the true number of comparisons made. `--metric both`
(now the default) runs both in one pass into one shared `records` list, and
`_apply_global_bh` corrects across the union once, written to
`significance_report.csv`. `--metric mae` alone still works and still
writes its own `significance_report_mae.csv` (kept, not retired -- see the
file table above), but that file's `bh_significant_global` answers a
narrower question (survives correction among just the 60 eligible `mae`
tests) than
`significance_report.csv`'s (survives correction among all 192 `exact`+
`mae`+`non_terminating` tests together). One concrete effect of the
difference: `qwen3-8b`/`edge_count`/`rwse` (`mae_delta=-12.8`, p=0.0010) is
`bh_significant_global=False` in the standalone `significance_report_mae.csv`
but `True` in the unified `significance_report.csv` -- rank 2 of 100 eligible
pooled tests (threshold 0.001) clears the line that rank 1 of 60 eligible
mae-only tests (threshold 0.00083) didn't, since p=0.0010 sits just above
the mae-only threshold but exactly at the pooled one. Both numbers are
correct; they're answers to different questions, and the unified one is the
one to cite going forward.

**Also added: `headroom`** (`min(control_mean, 1 - control_mean)`) alongside
`near_ceiling`/`low_power`, so a `low_power` row can be read as "low_power
*because of* near-ceiling headroom" versus "low_power despite real headroom
left" -- the two booleans alone couldn't distinguish those, and on the
current data they happen to co-occur completely for `gemma4-e4b`.

### Phase 1.4: reproducibility & process hardening

**`--filter`.** The `*_by_style.csv` artifacts (now deleted, see the
zero_cot purge) were produced by an undocumented manual pre-filter step --
someone hand-edited `sweep_frame.csv` down to one style before running
`check_significance.py`, a process this repo never checked in a script
for. `--filter "<pandas query expression>"` (e.g. `--filter "model ==
'gemma4-12b'"`) formalizes that into one flag applied to `--frame`
immediately after loading, before anything else runs -- any future "what
holds up under subset X" question is now one documented command instead
of a remembered snippet. The multiplicity correction is computed only over
the filtered rows: a `--filter`'d run answers a genuinely different,
smaller-family question than the unfiltered one, not a display slice of it.

**`--confirmatory-config`.** A JSON file naming which (arm, model,
condition, metric) cells were decided *before a sweep's results were
seen* to be the ones a claim will actually rest on -- see
`scripts/check_significance.py`'s module docstring for the exact format.
Every record gets a `hypothesis_type` column (`"confirmatory"`/
`"exploratory"`, blank when no config is given) and `bh_significant_global`
corrects the two groups in **separate** families: a small, strict
confirmatory family, and an exploratory family that's still reported (not
suppressed) but explicitly labelled hypothesis-generating rather than
confirmed. This is the direct answer to the pattern this project has hit
more than once (see "Retracted: 'Most primer findings depend on the
retired `zero_cot` style'" and the `filler`-primer episode in
`docs/sweep-findings.md`) -- a large exploratory sweep produces findings
that later turn out to be artifacts of pooling or of the specific format
tested, and there was previously no structural way to tell "this was
predicted in advance" apart from "this was noticed after the fact and
looks real." Committing the config file before running the sweep it
applies to is the discipline this flag makes possible, not something code
can verify from inside a process reading the file after the run -- that
part still depends on actually doing it in that order.

### The `mae` metric mode

Accuracy is not the only lens `check_significance.py` applies to
`sweep_frame.csv`. `--metric mae` re-analyzes the same already-scored data
-- no new GPU time -- using each response's `absolute_error` (present for
the 3 integer tasks: `node_count`, `edge_count`, `node_degree`) instead of
exact-match. Exact-match collapses "off by 1" and "off by 20" into the
same "wrong"; a primer that makes wrong answers *closer* to correct
without flipping enough of them to exact matches is invisible to that
metric at this sample size, but visible to a graded one. Reported per
`(model, task, condition)`, not pooled across tasks -- a node-count error
of 2 and a degree error of 2 aren't the same size of mistake.
`mae_delta = control_mae - treatment_mae`, so positive still means "the
condition helped", matching the `exact` metric's sign convention despite
the underlying quantity being inverted (lower error is better). This is
this project's own secondary-signal choice (`graphtalk/scoring.py`
attributes it to "the proposal"), not Fatemi et al.'s own methodology --
the paper's own headline metric is exact-match accuracy, the same one the
rest of this significance pipeline uses.

## Current significance results, at a glance

Three lenses on the same tracked sweep: does the primer change *whether*
the model is exactly right (main sweep), does it change *how often* the
model times out (thinking arm), and, for the three tasks with a numeric
answer, does it change *how far off* a wrong answer is (`mae`). ✅ =
significant and survives the whole-table correction
(`bh_significant_global`); ⚠️ = significant within its own model's (or
model-and-task's) comparison only; -- = not significant.

**`all`, the union of `degree`/`clustering`/`rwse`, is corrected as its own
single-hypothesis family** (`_is_derived_condition`, `scripts/check_significance.py`)
rather than pooled with the five independent conditions -- it's mechanically
correlated with them, not a sixth independent test. • marks a significant
`all` result: significant at raw α (p ≤ 0.05), but not eligible for
`bh_significant_global` (a single-hypothesis test needs no multiplicity
correction, but also can't be said to "survive" one). **`exact` and `mae`
now share one multiplicity-correction budget** (`--metric both`, the
default) rather than two separate ones -- a `mae` cell can newly read ✅
where it previously read ⚠️, or vice versa, purely from being pooled with
`exact`'s rows rather than any change in the underlying data.

### Main sweep -- accuracy vs. `none`

**No cell is significant.** `gemma4-12b` and `gemma4-e4b` are `near_ceiling`
on all six conditions (96-99% control accuracy, per "A third pass" above),
genuinely inconclusive rather than null; `qwen3-14b` and `qwen3-8b` have real
headroom (88-90% control accuracy) and still show no significant effect --
closer to a genuine null.

| Model | all | clustering | components | degree | filler | rwse |
|---|---|---|---|---|---|---|
| `gemma4-12b` [^ceiling] | -- | -- | -- | -- | -- | -- |
| `gemma4-e4b` [^ceiling] | -- | -- | -- | -- | -- | -- |
| `qwen3-14b` | -- | -- | -- | -- | -- | -- |
| `qwen3-8b` | -- | -- | -- | -- | -- | -- |
| pooled across all models | -- | -- | -- | -- | -- | -- |

[^ceiling]: `near_ceiling=True` on all six conditions (96-99% control
accuracy) -- see the CI-based bound in "A third pass" above before reading
any of these cells as a confirmed null.

Unchanged by excluding `all` from the independent-condition family (its own
raw p-value ranges 0.54-1.0 across all five groups -- nowhere close to
significant on its own either). An earlier version of this table, computed
while `zero_cot` rows were still pooled into the main sweep, showed several
✅/⚠️ cells (`gemma4-12b`/`components`, `gemma4-e4b`/`components`,
`qwen3-14b`/`degree`, `qwen3-8b`/`all`, and others). None of them survive on
`zero_shot`-only data -- see "Retracted: 'Most primer findings depend on the
retired `zero_cot` style'" in `docs/sweep-findings.md`.

### Thinking arm -- non-termination rate vs. `none`

"fewer timeouts" is the favorable direction:

| Model | all | clustering | components | degree | filler | rwse |
|---|---|---|---|---|---|---|
| `gemma4-12b` | • fewer timeouts | -- | -- | -- | ✅ fewer timeouts | -- |
| `gemma4-e4b` | -- | -- | -- | -- | -- | -- |
| `qwen3-14b` | -- | -- | -- | -- | -- | -- |
| `qwen3-8b` | -- | -- | -- | -- | -- | -- |
| pooled across all models | • fewer timeouts | -- | -- | -- | ⚠️ fewer timeouts | -- |

`components` no longer reads significant for `gemma4-12b` once `all` is
removed from its family: at the old family size of 6, `all`'s very small
p-value (0.0037) reshaped the BH rejection region enough to carry
`components` (own p-value unchanged) over the line too -- exactly the kind
of family-composition artifact excluding derived conditions is meant to
close (BH's rejection region depends on the whole family, not just each
p-value in isolation, so this can go either direction; see
`tests/test_significance.py::test_global_bh_can_be_stricter_than_per_family_bh`
for the general phenomenon). `all` itself is still a real, if narrower,
effect on `gemma4-12b` (p=0.0037) and pooled (p=0.048, barely) -- both •,
not ✅, since a single-hypothesis test isn't part of the global correction.

### Per-task error (MAE) vs. `none`

`node_degree` carries no signal at all (errors are small and rare on that
task regardless of condition) and is omitted; the other two tasks:

**`node_count`**

No significant cells.

| Model | all | clustering | components | degree | filler | rwse |
|---|---|---|---|---|---|---|
| `gemma4-12b` | -- | -- | -- | -- | -- | -- |
| `gemma4-e4b` | -- | -- | -- | -- | -- | -- |
| `qwen3-14b` | -- | -- | -- | -- | -- | -- |
| `qwen3-8b` | -- | -- | -- | -- | -- | -- |

**`edge_count`**

| Model | all | clustering | components | degree | filler | rwse |
|---|---|---|---|---|---|---|
| `gemma4-12b` | -- | -- | -- | -- | -- | -- |
| `gemma4-e4b` | -- | -- | -- | -- | -- | -- |
| `qwen3-14b` | -- | ⚠️ helps | -- | -- | -- | -- |
| `qwen3-8b` | -- | -- | -- | -- | -- | ✅ hurts |

`edge_count` is still where essentially all of the MAE-specific signal
lives, consistent with "Why `gemma4-e4b` truncates more"
(`docs/sweep-findings.md`): it is exactly the task where models fall into
exhaustive, error-prone manual counting, so it is the task where a bad
primer's damage shows up as *larger* miscounts before it shows up as *more
frequent* wrong answers. `qwen3-14b`/`clustering` is significant only within
its own model's five-condition comparison; `qwen3-8b`/`rwse` (p=0.0010) now
survives the whole-table correction (✅, upgraded from ⚠️) since `--metric
both` pools it with `exact`'s mostly-null p-values rather than testing `mae`
against its own, smaller, separately-corrected family -- rank 2 of 100 in
the unified family, comfortably under its BH threshold. An earlier,
`zero_cot`-pooled version of this table showed five significant cells;
three did not survive restricting to `zero_shot`-only data (see the
retraction note in `docs/sweep-findings.md`).

### Retracted: "What holds up without `zero_cot`"

This subsection used to compare `zero_shot`-only significance against
`zero_cot`-only to show how much of the tables above depended on the
retired style. `zero_cot` and its rows have since been fully removed from
the project, so that comparison is no longer reproducible; the tables above
are now `zero_shot`-only by construction; see the matching retraction note
in `docs/sweep-findings.md`.

## Track 2: data-collection planning (dry runs, no GPU time yet)

Track 1 is analysis of already-collected data; Track 2 is about what a
*future* sweep should look like. Both scripts below are dry runs against
already-collected data -- deliberately, so a `--count`/sampling decision
is checked before any new GPU time is spent, not after.

### 2.1: `scripts/recommend_count.py` -- MDE-targeted `--count`

Translates each non-significant `main_sweep`/`exact` cell's already-computed
`mde_delta`/`mde_delta_negative` into a recommended `--count`, via the
closed-form `MDE ~ 1/sqrt(N)` scaling
(`n_clusters_needed = n_clusters * (mde_delta / delta) ** 2`; see the
script's own docstring for the full derivation and skip conditions). Against
the current `significance_report.csv`: 16 of 20 non-significant cells get a
finite recommendation, and only 6 of those 16 are affordable within the
published split's 500-graph-per-task cap -- the rest (mostly `qwen3-14b`,
which needs 2,430-36,490 clusters depending on condition) would need a
larger or restructured corpus (see 2.2) rather than just a bigger `--count`.
4 cells are unrecommendable outright: 2 have an exactly-zero observed delta,
2 didn't converge (near-ceiling/near-floor, where headroom -- not sample
size -- is the binding constraint; see the third pass above).

`scripts/validate_recommend_count.py` checks that extrapolation against a
real bootstrap power simulation at the recommended size, for cells small
enough to simulate in reasonable time (`--max-n-clusters-target`, default
10,000 -- the largest real recommendations, 30,000+ clusters, are already
far past the published cap regardless of whether the closed-form estimate
is exact, so aren't worth the simulation cost). Result on the 6
under-3,000-cluster cells: 5/6 land at simulated power 0.87-1.00, one
right at 0.87 (near the 80% target), the rest well above it -- i.e. the
closed-form extrapolation is **conservative** here (recommends more
clusters than strictly needed for 80% power), never optimistic, in every
cell checked so far. That's the safe direction for a planning number to
err in, but it means `recommended_count` should be read as an upper
bound, not a precise minimum.

### 2.2: `scripts/validate_stratified_sampling.py` -- does graph size predict discordant pairs?

`build_prompts.py --graph-source stratified` (Track 2.2's proposed fix)
oversamples the largest graphs per task instead of scaling `--count`
uniformly, on the theory that larger graphs concentrate more of a
near-ceiling model's errors (and so more of a primer's chance to flip an
answer) per graph collected. Checked against already-collected responses
(`sweep_frame.csv` joined against `prompts.jsonl` for per-instance
`nodes`), splitting each near-ceiling model's (`gemma4-12b`, `gemma4-e4b`
at the default 0.95 threshold) paired instances into small/large node-count
strata at the per-cell median: 7 of 12 (model, condition) cells show a
higher discordant-pair rate in the large-graph stratum, and the mean
discordant rate across all cells is higher for large graphs (0.0217 vs.
0.0185) -- a real but modest effect, not a dramatic one. Individual
per-cell permutation p-values stay far from significance in both strata at
current sample sizes (as expected -- these are the same near-ceiling cells
Track 1 already flagged as underpowered), so this is a directional signal
supporting the stratified-sampling strategy, not yet a confirmed effect;
worth re-checking once a stratified run's own data exists.

## The batching baseline

`budget-gemma4-e4b.jsonl` and `budget-qwen3-8b.jsonl` are the reference for
verifying a batched implementation. Decoding is greedy, so a correct batch must
reproduce these responses near-identically. That check matters more than it
sounds: **Qwen3's tokenizer defaults to `padding_side='right'` while Gemma's
defaults to `'left'`**, and for a decoder-only model the wrong padding side
produces fluent, well-formed, entirely wrong text rather than an error.

## Not kept

The prompt subsets used to drive these runs (`diag_prompts`, `think_probe`,
`tail-probe`, `redo-*`) are regenerable from `prompts_zero_shot.jsonl` plus
`truncated_keys.json`, and are not tracked.
