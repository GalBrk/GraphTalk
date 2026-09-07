# Run plan: improving statistical power across GOT and integer sweeps

> **Status: partially executed.** Phases 1 and 2 have landed --
> `analysis/task_scoped_screen.csv`, `analysis/task_scoped_screen.got.csv` and
> `analysis/task_scoped_screen_comparison.md` are on disk, as are four
> `analysis/confirmatory_*.json` pre-registrations. Later phases have no
> artifacts, so treat them as outstanding. This document is still live: follow
> its instructions, unlike the executed plans beside it.

## How to use this document

This is an autonomous execution plan. Work through the phases in order.
For each phase: plan the implementation, implement it, test it, verify it
against the acceptance criteria given, and log what happened (including
anything that didn't match expectations -- don't silently paper over a
discrepancy, name it, the way prior phases of this investigation did with
the 5/180 decoding-noise flips and the Simpson's-paradox catch).

**Do not stop for confirmation except where explicitly marked `STOP`.**
Every `STOP` in this plan is at the point immediately before submitting a
SLURM job (`sbatch cluster/sweep.sbatch ...`) -- prepare everything up to
and including the exact command, print it, and end the run there. Nothing
else in this plan should pause for human input; use your own judgment,
proceed, and log the reasoning.

**The goal is not "more significant results."** It is well-powered tests
that would detect a real effect if one exists, and honest null results
where one doesn't. Do not chase significance by pooling data in whatever
way makes a p-value smaller, and do not treat a larger n as automatically
better -- `docs/plans/scale-vs-topology-investigation.md` and the
`qwen3-14b`/`edge_count` discovery both came from *narrowing* scope
correctly, not from more data. Keep that discipline through every phase
below.

**Every new script needs tests**, matching this project's existing
convention (`tests/test_significance.py`, `tests/test_recommend_count.py`,
etc.). After each phase, run `pytest -q` and report the pass count and any
failures/regressions before moving to the next phase (compare against the
project's known baseline: **603 passed** on the merged
`small-model-suite-and-primer-power` branch, with
`tests/test_hierarchical_model.py` and `tests/test_mixed_models.py` `--ignore`d
because neither graphtalk conda env has `statsmodels`/`pymc`. The 572 figure
this line used to quote predates both that merge and those ignores; compare
against 603, or a passing run will look like a regression).

**Effect-size discipline for any new sample-size calculation**: use a
conservative estimate (the bootstrap CI's lower bound, or an explicitly
discounted point estimate), never the raw observed delta from a cell that
was selected because it looked promising. The `qwen3-8b`/`degree`
replication (+7.8pp observed -> +6.5pp true) is direct, measured proof that
raw observed effects on selected cells are optimistic.

---

## Phase 1 — Task-scoped screen, GOT scheme (zero new GPU time)

**Rationale.** `recommend_count.py`'s sample-size recommendations pool
across all 6 tasks. Manually re-checking `degree` vs `none` restricted to
`edge_count` alone (data already on disk) found a near-significant or
significant effect in 3 of 4 models at existing n=30 -- a signal the pooled
view was diluting, not one that needs new data to see. This needs to be a
repeatable script, not a one-off manual check, and needs to run across
every model/condition/task combination, not just the one cell already
spot-checked.

**Implementation.**
1. **Audit which (condition, task) pairs let the primer answer the task
   directly, without the model needing the graph encoding at all.** This
   is a different, sharper confound than the shortcut-ceiling one below.
   `graphtalk/primers.py`'s `degree` condition renders one sentence per
   node, literally `"Node X has degree Y."`, for every node in the graph.
   That means, from primer text alone:
   - `degree` x `node_count`: the number of "Node ... has degree ..."
     sentences *is* the node count.
   - `degree` x `node_degree`: the queried node's degree is stated
     verbatim -- a direct lookup, not even arithmetic.
   - `degree` x `edge_count`: edge count = sum of the stated degrees / 2.
   - `all` inherits every one of these, since it is defined as `degree +
     clustering + rwse`.
   - `clustering` x `cycle_check` is weaker and one-directional: a nonzero
     clustering coefficient on any node is sufficient, not necessary,
     evidence a cycle exists -- it can only ever help confirm a cycle,
     never rule one out.
   Verify this against the actual code (`render_primer`/`_degree_phrase`
   in `graphtalk/primers.py`, and `graphtalk/shortcuts.py`'s existing
   primer-text parser -- reuse that parser rather than re-deriving
   extraction logic) and against `tests/test_primers.py`/
   `tests/test_shortcuts.py`, rather than trusting this list blindly --
   check for cases this pass missed (e.g. whether `components` gives
   partial information for any task via cross-component reasoning about
   the specific queried nodes). Write the result to
   `analysis/primer_task_shortcut_audit.md`: one row per (condition, task)
   pair used in the sweep, classified `shortcut` / `partial` / `none`,
   with the one-line mechanism for each `shortcut`/`partial` row.
2. Locate which of the 6 tasks are shortcut-ceiling-bound *independent of
   primer* (a graph-blind program solves them at ~100% even under `none`,
   per the "Shortcut-ceiling confound" note in the original
   significance-review plan) -- derive this list from the actual code/docs,
   don't assume one from memory. This is a distinct exclusion from step 1:
   step 1 is "this primer gives away this task," step 2 is "this task is
   trivial regardless of primer." Document which tasks were excluded and
   why.
3. Write `scripts/task_scoped_screen.py`: for every `(model, condition,
   task)` combination in `analysis/sweep_frame.got.csv` (excluding the
   `all` condition as a screened cell, since it isn't independent
   evidence, and the shortcut-ceiling tasks from step 2), run the same
   `paired_permutation_test_clustered` / `cluster_bootstrap_ci_clustered`
   primitives `check_significance.py` and `task_breakdown.py` already use
   (reuse `check_significance.py`'s `_paired_values`, don't reimplement
   pairing logic). Output one row per cell to
   `analysis/task_scoped_screen.got.csv`: model, condition, task,
   n_clusters, delta, ci_low, ci_high, p_value, and a `shortcut_flag`
   column populated from step 1's audit.
4. Sort/flag cells with p < 0.10 (screening threshold, deliberately looser
   than 0.05 -- the point is to surface candidates for follow-up, not to
   declare findings) for a human-readable summary printed at the end,
   **split into two sections**: `shortcut_flag == "none"` cells first --
   these are the ones that actually test whether the primer helps graph
   reasoning -- then `shortcut`/`partial` cells separately, explicitly
   labeled as testing arithmetic/lookup execution reliability rather than
   reasoning.

**Verification.** Write tests in `tests/test_task_scoped_screen.py`
covering the shortcut-task exclusion and the pairing logic on a small
synthetic frame. Then, as a **regression check against already-known
values**, confirm the script reproduces (within Monte Carlo tolerance --
permutation p-values vary by seed, so check the delta is within ~0.01 and
the p-value is on the same side of 0.05, not an exact match) the following
`degree`/`edge_count` cells already computed by hand this session:

| model | delta | p |
|---|---|---|
| `gemma4-12b` | +0.167 | 0.065 |
| `gemma4-e4b` | +0.067 | 0.77 |
| `qwen3-14b` | +0.300 | 0.0228 |
| `qwen3-8b` | +0.367 | ~0.0040 |

If any of these don't reproduce within tolerance, stop and debug before
trusting the rest of the script's output -- this table is the ground truth
this phase exists to reproduce systematically. **Note these are all
`shortcut_flag == "shortcut"` cells per step 1's audit** -- reproducing
them validates that the pairing/permutation machinery works correctly,
not that this is where the follow-up priority should go (see Phase 5).

**Note on `gemma4-e4b` specifically**: its edge_count effect (+0.067,
p=0.77) is roughly a third to a fifth the size of the other three models',
not just "the same effect, less significant." Don't fold it into "the
effect generalizes across models" -- flag it in the output as
structurally different, while noting n=30 can't fully distinguish "no
effect" from "small effect, underpowered."

---

## Phase 2 — Task-scoped screen, integer scheme (zero new GPU time)

**Rationale.** Every task-scoping and topology finding so far was done on
GOT naming only. Whether the `degree`/`edge_count` pattern replicates
under integer naming (or is somehow GOT-specific) is an open, unanswered
question, and it's answerable with data already on disk.

**Implementation.** Run `task_scoped_screen.py` (from Phase 1, it should
already support this via a `--node-naming` flag or by defaulting to
processing whichever frame it's pointed at) against
`analysis/sweep_frame.csv` (the integer-naming frame), producing
`analysis/task_scoped_screen.integer.csv`. Then write a small comparison
(script or a short markdown table appended to
`analysis/task_scoped_screen_comparison.md`) joining the two schemes' `p <
0.10` cells side by side.

**Verification.** Confirm `qwen3-8b`/`degree`/`edge_count`/integer is
present and check whether its delta/p is consistent with the GOT result --
report this explicitly either way (consistent, or a real scheme-dependent
difference worth flagging as its own finding, not something to smooth
over).

---

## Phase 3 — Decide and document the reporting/multiplicity policy

**Rationale.** Phase 1 and the `qwen3-14b` discovery both demonstrate that
per-task, not pooled-across-6-tasks, should be the default significance
granularity going forward. This needs to be a stated policy, not an
implicit habit, since it changes what counts as one hypothesis for BH
purposes.

**Implementation.** Add a section to `analysis/README.md` (matching its
existing style) stating explicitly:
- Per-task is the primary reported granularity; pooled-across-tasks is
  kept as a secondary summary column, not the significance decision.
- Confirmatory family (currently just `qwen3-8b`/`degree`/GOT/`edge_count`,
  plus whatever this plan's Phase 5/6 pre-registers) is corrected
  separately from the exploratory family (everything from Phase 1/2's
  screen).
- GOT and integer naming are corrected as separate families, never pooled
  together.

Label this section as a policy decision made as part of this plan, with a
one-line pointer back to this document and the `qwen3-14b` discovery that
motivated it.

---

## Phase 4 — Infrastructure: per-task sizing and GOT-aware stratification (zero new GPU time)

**4a. Extend `recommend_count.py` to accept `--task`.**

**Rationale.** Right now, sizing a follow-up restricted to one task
requires manual recomputation (as happened by hand for `qwen3-14b`). This
needs to be a supported flag, not a one-off.

**Implementation.** Add `--task` to `recommend_count.py`, scoping its MDE
calculation to the given task's paired data instead of pooling across all
6. Preserve the existing pooled behavior as the default when `--task` is
omitted, for backward compatibility with any existing callers/docs.

**Verification.** Add a test confirming `--task edge_count` on
`qwen3-14b`/`degree` recommends a substantially smaller `--count` than the
existing pooled recommendation (1,944) -- direction matters here, not an
exact number. Also confirm the no-`--task` behavior is byte-identical to
before this change (regression test against a recorded prior output).

**4b. Extend graph stratification to support GOT naming.**

**Rationale.** `--graph-source stratified` (oversampling large graphs)
currently only supports `--node-naming integer`. Any GOT-scheme follow-up
that wants to concentrate sampling on large/dense graphs needs this
extended first. Node ids are contiguous `0..n-1` regardless of naming
scheme, and GOT name assignment (`node_naming.py`'s `GOT_NAMES[i]`) is
purely positional -- this should be mechanical, but verify that assumption
rather than trusting it.

**Implementation.** Locate the actual stratified-sampling implementation
(search for `build_stratified` / `--graph-source stratified` -- likely in
`build_prompts.py` or a graph-sampling module it imports) and remove the
integer-only restriction, routing the selected large graphs through the
same GOT naming assignment the standard path uses.

**Verification.** Extend `scripts/validate_stratified_sampling.py`'s
existing pattern (it already does a median-node-count stratified check for
integer naming) to also run under `--node-naming got`, confirming the
selected graphs' node-naming assignment matches what the standard
(non-stratified) path would produce for the same underlying graph. Add a
unit test asserting `--node-naming got` no longer raises/rejects.

---

## Phase 5 — Prepare (not run) the top non-shortcut follow-up(s)

**Rationale.** Phase 1's audit means `degree`/`edge_count` -- and, by the
same mechanism, `degree`/`node_degree` and `degree`/`node_count` -- test
whether the model can correctly execute a handed arithmetic/lookup
operation, not whether the primer improved graph *reasoning*. That's a
legitimate, still-interesting question in its own right (and connects
directly to the difficulty-scaling hypothesis: does execution reliability
on a fully-handed shortcut degrade with graph size the same way manual
counting does?), but it is a different claim, and shouldn't become the
flagship "primers help reasoning" result by default just because it's the
easiest effect to detect.

**Implementation.**
1. From Phase 1/2's screen, identify the most promising cell(s) among
   `shortcut_flag == "none"` pairs -- the real candidates are `clustering`,
   `rwse`, or `components` paired with `edge_existence`, `connected_nodes`,
   or `cycle_check` (treat `clustering`/`cycle_check` as `partial`, not
   `none`, per the audit -- a nonzero clustering coefficient only ever
   helps confirm a cycle, never rule one out). Prepare a pre-registration
   and sized follow-up for the top one using steps 3 below.
2. Separately, and lower priority: if the arithmetic-execution-reliability
   question is still worth answering on its own terms, prepare the
   `degree`/`edge_count` (and/or `node_degree`) follow-up too -- but
   pre-register it explicitly as *"does the model reliably execute a
   fully-handed arithmetic shortcut, and does that reliability degrade
   with graph size"*, not as evidence the primer improves reasoning.
3. For each cell prepared: write a pre-registration file analogous to
   `analysis/confirmatory_got_degree.json` naming the exact cell and
   claimed direction, before any new data collection. Use the extended
   `recommend_count.py --task` (Phase 4a) with a conservative effect
   estimate (the CI lower bound, not the raw point estimate -- you have
   direct proof from the `qwen3-8b` replication that raw observed effects
   on selected cells run optimistic) to compute the required `--count`.
   Build the prompts file for that count (`build_prompts.py` /
   `cluster/submit_sweep.sh --node-naming got --count <N>`, same path
   used for the `qwen3-8b` replication) -- CPU-only, login-node step, safe
   to run.
4. Check whether Phase 1/2's screen surfaced any other `p < 0.10`,
   `shortcut_flag == "none"` cells cheap enough to prepare alongside the
   top pick in the same pass, and prepare those too.

**STOP HERE.** Print the exact `sbatch cluster/sweep.sbatch <model>
[...]` command(s) needed to actually generate the new data for each
prepared follow-up, and end the run. Do not submit them.

---

## Phase 6 — Conditional: stratified follow-up, only if Phase 5 remains underpowered

**Rationale.** Stratified sampling toward large/dense graphs is a real
lever (the difficulty-scaling hypothesis showed the `edge_count` effect
concentrates there), but it has a hard ceiling -- the published split has
only ~500 rows/task total, ~175 of them "large" -- and it's real
additional engineering (Phase 4b). Only worth spending once Phase 5's
plain follow-up size is checked against that ceiling.

**Implementation.** Only proceed with this phase if Phase 5's computed
`--count` for any cell would need more large graphs than the ~175/task
ceiling provides at the default (non-stratified) sampling rate. If so,
using Phase 4b's GOT-aware stratification, build a stratified prompt set
concentrating on large/dense graphs for that specific cell, sized with the
same conservative-estimate discipline as Phase 5. State explicitly in the
pre-registration file that this targets "graphs at these sizes/densities
in the published split," not a general large-graph claim.

**STOP HERE.** Print the exact `sbatch` command(s), and end the run. Do
not submit them.

---

## What "done" looks like

By the end of Phase 4, every model/condition/task/scheme combination
already on disk has been screened at the correct (task-scoped) granularity
in both naming schemes, the reporting policy this implies is documented,
and the infrastructure needed to size and build a proper follow-up exists
and is tested. Phases 5 and 6 end with one or more fully-prepared,
pre-registered, correctly-sized follow-up sweeps -- and the exact commands
needed to actually run them, waiting on a human to pull the trigger on GPU
time.
