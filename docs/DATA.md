# Data reference

Every dataset in this repository, its schema, and how the files join to each
other. `runs/README.md` covers day-to-day use; this file is the authority on
structure.

## Inventory

| path | rows | what |
|---|---|---|
| `prompts.jsonl` | 2,520 | every prompt in the main sweep |
| `prompts_zero_shot.jsonl` | 1,260 | the `zero_shot` half, byte-identical to those rows; the thinking arm's input |
| `runs/<model>.jsonl` | 2,520 × 4 | main sweep responses, one file per model |
| `runs/<model>-think.shard<i>of<n>.jsonl` | 1,260 × 4 | thinking-arm responses, split across shard files |
| `runs/<model>-think.redo.shard<i>of<n>.jsonl` | 67 | evidence only — see caveats |
| `runs/archive/smoke-gemma4-e4b.jsonl` | 20 | a smoke test; archived, excluded by directory |
| `shortcuts.json` | 42 entries | the primer-only solver bar per (task, condition) |
| `prompts_got.jsonl` | 2,520 | the same prompts with Game-of-Thrones node names; only its 1,260 `zero_shot` rows were generated |
| `runs/<model>.got.jsonl` | 1,260 × 8 | GoT-named responses, every arm, `zero_shot` only; `node_naming: "got"` on every row |
| `analysis/*.jsonl` | 51 | token-budget measurements; see `analysis/README.md` |

25,267 response rows in total: 10,080 main sweep, 5,040 thinking arm, **10,080
GoT-named**, 67 redo. The GoT rows pair one-to-one with the `zero_shot` half of
the other two — same graphs, same queries, same primers — and are scored
separately, never pooled (`graphtalk.analysis.infer_node_naming` raises on a
mix). See README.md#node-naming.

## The pairing key

`instance_id` has the form `"<task>/<index>"`, e.g. `node_count/7`. **The same
graph and the same query appear under all seven conditions and both styles**,
differing only in the primer. That is what makes the design paired, and it is the
key the McNemar test is computed over.

A row is uniquely identified by `(instance_id, condition, style)` within a model.
No file contains a duplicate of that triple; this is asserted after every run.

**That triple assumes one node-naming scheme.** `runs/<model>.jsonl` (the
`integer` scheme, unstated on the row) and `runs/<model>.got.jsonl` (the GoT
scheme, `node_naming: "got"` on every row) each satisfy the triple on their
own, but the two files together do not — the same `(instance_id, condition,
style)` recurs once per scheme. `analysis.build_frame`'s `node_naming`
column widens the true key to `(instance_id, condition, style, node_naming)`
within a model, and every scoring script (`build_sweep_frame.py`,
`sample_failures.py`, `check_significance.py`) raises rather than silently
pooling if its input carries more than one scheme — see
`README.md#node-naming`.

## `prompts.jsonl` / `prompts_zero_shot.jsonl`

| field | type | meaning |
|---|---|---|
| `instance_id` | str | pairing key, `"<task>/<index>"` |
| `task` | str | one of the six tasks below |
| `condition` | str | one of the seven primer conditions below |
| `style` | str | `zero_shot` or `zero_cot` |
| `prompt` | str | the exact text handed to the model, primer included |
| `gold` | str | the published answer, dataset formatting preserved |
| `nodes` | int | node count of the underlying graph, 5–19 |
| `edges` | int | edge count of the underlying graph |

`nodes` and `edges` describe the graph, not the answer — they are present on every
row regardless of task, and are what you group by to ask whether an effect depends
on graph size.

## `runs/*.jsonl`

| field | type | meaning |
|---|---|---|
| `instance_id`, `task`, `condition`, `style`, `gold` | | copied from the prompt row |
| `model` | str | model key, e.g. `gemma4-12b` or `gemma4-12b-think` |
| `response` | str | the generated text, prompt stripped |

**The schema is identical for both arms**; the arm is identified by the `model`
field carrying the `-think` suffix. There is no `prompt` field — join on
`(instance_id, condition, style)` against the prompt file to recover it.

Because `model` is on every row, **sharded files need no reassembly**:
`score_sweep.py` groups by that field, so pointing it at `runs/*.jsonl` pools the
shards correctly. Shard filenames are bookkeeping for resumable jobs, not
meaningful divisions of the data.

## `shortcuts.json`

A flat object keyed `"<task>/<condition>"` with a float score in [0, 1]: what a
deterministic program scoring only the rendered primer text achieves on that cell.
It is the bar a model result is read against — see
`docs/plans/shortcut-ceilings.md`.

All **42** cells are present (6 tasks x 7 conditions). Note that
`shortcut-ceilings.md` speaks of *36* cells: that is 42 minus the six `none`
cells, which are excluded there as trivial because an empty primer can only score
the majority-class baseline. They are still written to the file, and their values
are that baseline — `cycle_check/none` is 0.832, `node_count/none` 0.064 — which
makes them a useful sanity check rather than padding: if a `none` cell ever
differs from the baseline, the solver is reading something it should not.

## Vocabularies

**Tasks (6)** — `node_count`, `edge_count`, `node_degree`, `connected_nodes`,
`edge_existence`, `cycle_check`.

**Conditions (7)** — `none` (no primer, the control), `degree`, `clustering`,
`rwse`, `components` (one statistic each), `all` (every statistic), `filler` (a
primer of the same shape carrying no structural information, which separates
"primer present" from "primer informative").

**Styles (2)** — `zero_shot`, `zero_cot`. The thinking arm is `zero_shot` only.

> **`zero_cot` is retired and will not be used again in this project.** Chain-of-thought
> is measured by the thinking arm (native reasoning at `zero_shot`), which superseded
> it. The 5,040 `zero_cot` rows are kept because a response is only interpretable
> against the prompt that produced it, but they are a historical record: their `filler`
> and `edge_existence` cells answer prompts that no longer exist, and the style ran at
> half the token budget, so its accuracy is depressed by truncation as well as by the
> prompt. **Filter on `style == "zero_shot"` for anything current.** See
> `graphtalk/prompts.py`.

**`gold` formatting varies by task** and keeps the dataset's own punctuation, so
compare through `graphtalk.scoring`, never by string equality:

```
node_count       ' 18.'
edge_count       ' 115.'
node_degree      '13.'
connected_nodes  '0, 1, 2, 4, 5, 6, 10, 11, 12, 13, 15, 16, 17.'
edge_existence   'No.'
cycle_check      'Yes, there is a cycle.'
```

## "Connected" means adjacent, not reachable

`edge_existence` asks *"Is node A connected to node B?"* and `connected_nodes`
asks *"List all the nodes connected to A"*. In both, the upstream ground truth is
**direct adjacency**, not reachability. From `talk_like_a_graph/graph_tasks.py`:

```python
if ((source, target) in graph.edges()) or ((target, source) in graph.edges()):
    answer = 'Yes.'
else:
    answer = 'No.'
```

No path check, no traversal. `connected_nodes` likewise returns the neighbour
list, not the reachable set.

The wording is genuinely ambiguous and the distinction is not academic here.
Across the 30 `edge_existence` instances:

| | count |
|---|---|
| gold `Yes` — an edge exists | 12 |
| gold `No`, but a path exists | **14** |
| gold `No`, genuinely unreachable | 4 |

**47% of instances would flip under the reachability reading**, almost all at
shortest-path distance 2, mostly in single-component graphs. A model reading
"connected" as "reachable" would answer `Yes` on 26 of 30 and score 53.3%, below
the 60% majority baseline.

**The models do not read it that way.** Share answering `Yes` on the 14
path-only pairs, where the reachability reading predicts ~100%:

| model | edge (gold Y) | path-only (gold N) | unreachable (gold N) |
|---|---|---|---|
| gemma4-12b | 100% | 5% | 5% |
| gemma4-12b-think | 100% | 6% | 4% |
| gemma4-e4b | 92% | 20% | 2% |
| gemma4-e4b-think | 95% | 16% | 7% |
| qwen3-14b | 100% | 17% | 9% |
| qwen3-14b-think | 100% | 23% | 7% |
| qwen3-8b | 99% | 34% | 14% |
| qwen3-8b-think | 89% | 24% | 4% |

At 5-34% rather than ~100%, all eight resolve the ambiguity the same way the gold
does *most of the time*. The original conclusion drawn here was that the ambiguity
therefore does not drive the error rate and no correction was warranted.

### The prediction, staked before the re-run

Before any row was regenerated, this section's conclusion was written down as a
falsifiable claim: *"the rewording moves `edge_existence` accuracy very little. If
accuracy is flat, the ambiguity was not what drove error on this task and the
rewording bought comparability loss for nothing; if it moves, the measurement above
was wrong about what the models were doing. Either outcome is a result — but only
because the claim was staked first."*

### Outcome, recorded 2026-08-29

**That conclusion was wrong, and the re-run measured it.** The question was
reworded to *"Does an edge exist between Node A and Node B?"* and all affected rows
regenerated. Accuracy rose by a mean of **10.2 points**, and the path-only `Yes`
rate in the table above fell to **exactly 0.0% in every one of the eight arms**.
The 5-34% residual this section treated as negligible was the entire effect.

What the table got right is the **gradient**, and it turns out to be the whole
story. The path-only `Yes` rate runs inversely to model quality, from 5% on
gemma4-12b to 34% on qwen3-8b, and the accuracy gain from rewording tracks that
rate at **r = +0.95** — from +1.7 on gemma4-12b to +18.9 on qwen3-8b. So the
measurement here was sound and reproduces against the regenerated rows; the error
was inferential. gemma4-12b, the model this section leaned on hardest, is the one
model for which "the ambiguity is inert" holds: it gained 1.7 points, near enough
to nothing. Generalising from it to a sweep whose error mass sits in the weaker
models is what produced the wrong recommendation.

The narrower claim survives intact — the models do resolve "connected" as adjacency
far more often than not. It just turns out that "far more often than not" was
hiding about a tenth of this task's accuracy.

Full analysis in `docs/sweep-findings.md`; reproduce with
`scripts/rewording_effect.py`.

Caveat on precision: this rests on 30 instances, 14 of them ambiguous. The
per-model rates are over 14 x 7 conditions = 98 rows each and are stable enough,
but the instance count is small and a different query draw would move them.

## `connected_nodes`'s "No nodes" spelling was matched too narrowly

The dataset spells an isolated node's answer `"No nodes."`
(`graphtalk/graphqa.py`), and `graphtalk/scoring.py` only recognised that
literal phrase (`_NO_NODES = re.compile(r"\bno\s+nodes?\b")`). A model
answering the equally correct `"None"`/`"None."` instead matched neither that
regex nor the digit-list one, so extraction fell through to a stray digit
earlier in the response (typically a number mentioned in the question text
itself) and scored the row `wrong` rather than `correct`.

### The prediction, staked before the re-score

Only one of the 30 `connected_nodes` instances (`connected_nodes/2`) has gold
`No nodes.`, so the fix could only move rows tied to that single instance —
predicted before rescoring: a handful of rows across the 8 arms, not a
sweep-wide shift, since every other instance's gold is a non-empty list that
"None" recognition cannot touch.

### Outcome, recorded 2026-08-30

`graphtalk/scoring.py` gained a second, tightly-anchored regex,
`_NONE_ANSWER = re.compile(r"(?:^|:)\s*none\.?\s*$")`, wired into
`_extract_node_list` alongside `_NO_NODES` and canonicalising to the same
`"No nodes"` string, so nothing downstream (`_node_set`, `set_f1`,
`score_one`) needed to change. Anchored to end-of-line/end-of-tail (optionally
after a `"<label>:"` prefix like `"A:"`) rather than searched anywhere, so a
sentence like *"None of the nodes are directly connected, but node 5 is
adjacent"* is not misread as the empty-set answer.

Rescoring `runs/*.jsonl` with the fix and rebuilding
`analysis/sweep_frame.csv` flipped **8 rows** from `wrong` to `correct`, all
`connected_nodes/2`, spanning 7 (model, condition, style) combinations on
`gemma4-12b` plus one on `gemma4-e4b-think`. Confirmed against the previous
committed frame that no other task and no non-terminating row changed. The
prediction held: a single-instance bug produces a single-instance fix, not a
sweep-wide accuracy move.

**Known gap, left out of scope on purpose (round one):** one further
`connected_nodes/2` row (`qwen3-14b`, `condition: all`) answers `"A: []"` — a
different spelling this fix does not recognise — and a semicolon-prefixed
`"...; none."` shape is also not caught, since the regex requires `^` or `:`
immediately before `"none"`. Both were called precision-over-recall
tradeoffs, not oversights, at the time.

### Round two, recorded 2026-08-30

Both "known gaps" above turned out to be worth closing. `analysis/failure_sample.csv`
(this file now exists and was read directly, unlike the never-present
`false_examples.csv` earlier sessions were asked about) surfaced the `[]`
case again plus two more real shapes the original fix missed entirely:
`"None"` glued onto a sentence with **no separator at all** — `"...the list
is empty.None"`, a likely generation artifact — and the right token wrapped
in markdown emphasis or a trailing parenthetical aside — `"**A: None**"`,
`"A: [] (or None, depending on expected format for an empty list)"`.

Rescanning the tracked sweep found **51 of 84** gold-`No nodes.` rows still
not scoring `correct`. Before touching the code, the "; none." and `[]`
precision worry from round one was checked directly against evidence rather
than left as an assumption: across all 2,436 `connected_nodes` rows whose
gold is a *non-empty* list, zero have their last line end in `none`/`None.`,
and zero contain `[]` anywhere. Both gaps were safe to close.

The fix: `_NONE_ANSWER`'s anchor relaxed from `(?:^|:)\s*none\.?\s*$` to
`\bnone\.?\s*$` (the end-of-line requirement was already what provided the
safety, not the colon/start prefix); a new `_EMPTY_BRACKETS` pattern for
`"[]"`/`"[ ]"`; and a `_strip_trailing_decoration` helper that peels off
markdown emphasis, a trailing parenthetical, or a trailing period before
either check runs, so `"**[]** (empty list)."` is still read correctly.

**A second, unrelated bug was found and fixed in the same change:**
`_marker_tail` returns text after the *last* "answer"-labeled mention
anywhere in the response, which can be a mid-reasoning heading rather than
the true conclusion. A real response had `"3. **Determine the Answer:** ...
node 0 has no listed neighbors."` precede the true final `"A: []"` line; the
old tail-first priority returned a stray digit from the heading ("node 0")
instead of ever reaching the real answer. `_extract_node_list` now runs its
per-line scan of the full response *before* consulting the marker tail
(previously the reverse), making the tail a fallback rather than an override.

**Measured effect was larger than predicted, in a good way.** The prediction
was that this round would only move rows tied to the single empty-gold
instance (`connected_nodes/2`). Rescoring instead flipped **53 rows** across
**12 different instances** — the stale-marker fix turned out to also correct
several *non-empty*-gold rows that had the identical bug (e.g. gold
`"1, 2, 3, 4, 5, 6, 7, 8"` was previously extracted as the single stray digit
`"8"`; now the full, correct list). Confirmed precisely against the pre-fix
frame: every changed row moved `unparsed → correct` (1 row) or
`wrong → correct` (52 rows), and zero rows that were already `correct`
changed to anything else — the broader reach is a bonus from fixing a shared
bug, not a regression. One cosmetic-only difference (a `correct` row's `predicted`
string reordered from `"3, 7, 12"` to `"12, 3, 7"`, same set, same score) was
also observed and is not a behavior change worth chasing further.

**Still out of scope, on purpose:** genuinely wrong/refusal/hedge responses,
and `\boxed{}`-style LaTeX empty-box notation (2 rows) — not in the reported
examples and not evidenced beyond that count.

Reproduce with `tests/test_prompts.py::test_extracts_node_lists`,
`test_stale_answer_marker_does_not_shadow_the_final_line`,
`test_bracket_answer_survives_a_trailing_period_after_the_decoration`, and
`scripts/build_sweep_frame.py`.

## `edge_existence` responses that never say "yes"/"no" were unparsed

`graphtalk/scoring.py`'s `_extract_boolean` only recognised a standalone
`\byes\b`/`\bno\b` token. Some responses state their conclusion in prose that
never contains either word — most commonly `"An edge exists between Node A
and Node B."`, echoing the live question's own wording (*"Does an edge exist
between Node A and Node B?"*) — so extraction returned `None` and a
semantically-correct row scored `unparsed`.

### The prediction, staked before the re-score

Scanning all 2,520 tracked `edge_existence` rows found 17 `unparsed`, in four
disjoint groups: 5 live-`zero_shot` rows saying "An edge exists..."; 5
retired-`zero_cot` rows saying "...is connected to..."; 2 genuine refusals
("Cannot be determined..."); 5 truncated/non-terminating responses with no
stated conclusion at all. Only the first two groups are a paraphrase to fix —
predicted before rescoring: **10 rows**, all `unparsed → correct`, nothing
else in the frame touched.

### Outcome, recorded 2026-08-30 — and two regressions the naive version caused

The first implementation fed the new patterns into the *same* position-based
comparison already used for bare `yes`/`no` (whichever token's last occurrence
sits later in the text wins, since CoT states its conclusion at the end). That
version measured **62 changed rows, not 10** — it was reverted before being
recorded here, but the mechanism is worth keeping because it shaped the actual
fix:

1. **A mid-response restatement of the question outranked the real answer.**
   44 rows that already correctly resolved to `"No"` via a bare token flipped
   to `"Yes"`, because a later sentence like *"...to determine **if an edge
   exists** between X and Y, we need to..."* — the CoT restating the question
   itself as a preamble — sat later in the scanned text than the response's
   own bare `"No"`, and won the position comparison.
2. **A refusal was coerced into an answer.** A response that explicitly said
   *"we cannot determine if an edge exists between Node 0 and Node 1 from the
   given data"* — the same restatement phrase, this time inside a hedge — was
   read as a stated `"Yes"` even though the model declined to answer at all.

The fix that shipped addresses both by construction rather than by patching
around each case: the new patterns (`_EDGE_EXISTS`, `_CONNECTED_YES` in
`graphtalk/scoring.py`) are consulted **only as a fallback**, after both scopes
have already been checked for a bare token and found none — so they can turn
an `unparsed` row into a parsed one but can never override an
already-resolved answer, which is what (1) was. A `_QUESTION_OR_HEDGE_LEADIN`
check excludes a match immediately preceded by "if"/"whether"/"determine"/etc.
(checked against the text before the match, not a `re` lookbehind, since the
disqualifying word can be separated from "edge" by "an"/"the"), which handles
the direct form of (2).

That still wasn't sufficient on its own: a third real row showed the fallback
matching a true, but *irrelevant*, `"is connected to"` sentence — the graph
encoding's own vocabulary for describing a real, different edge, stated early
while the response summarises the graph, before it goes on to correctly refuse
to answer about the queried pair (`"* Node 8 is connected to nodes 4, 5. ...
A: Cannot be determined..."`). No lead-in word precedes it, so the hedge guard
doesn't catch it. The fallback is therefore also restricted to each scope's
**last non-empty line only**, not searched anywhere in it — the same "the
conclusion is stated last, don't pool across the whole response" rule
`_extract_node_list` already applies for `No nodes` (see the section above).

With both guards in place, rescoring `runs/*.jsonl` and rebuilding
`analysis/sweep_frame.csv` flipped exactly the predicted **10 rows** from
`unparsed` to `correct`, and confirmed against the previous frame that no
other task, no non-terminating row, and no row that already had a non-null
`predicted` value changed. The two refusal rows correctly remain `unparsed`.

**No symmetric "No" gap exists in the observed data** — every gold-`No`
`unparsed` row was a refusal or a truncation, not a "no edge"-style paraphrase,
because that phrasing already contains the bare word "no" and the existing
extractor already handles it. The fix is Yes-only, not a guess at symmetry.

Reproduce with `tests/test_prompts.py::test_extracts_edge_existence_paraphrases`
(and its three regression-specific siblings) and `scripts/build_sweep_frame.py`.

## `_extract_integer`'s tail scope picked the wrong number

`_extract_integer` (`node_count`/`edge_count`/`node_degree`) prefers an
explicit marker's tail over the full response, on the reasoning that "an
explicit marker is unambiguous." It wasn't: within the tail, it took the
*first* integer unconditionally, which is the queried node's own id whenever
a "node X is Y" sentence states the id before the value — "The degree of
node 7 is 2." read as `['7', '2']`, returning `7` instead of the gold `2`.

### The prediction, staked before the re-score

A sample of 23 `wrong`/`unparsed` rows (`analysis/failure_sample.csv`,
excluding `non_terminating`) found this bug in all 11 sampled `node_degree`
rows and both sampled `node_count` rows — 0 in `edge_count` (5 rows, all
genuinely wrong model arithmetic) or `cycle_check` (1 row, genuine reasoning
error). Predicted before rescoring: a fix scoped to `_extract_integer` would
move roughly this many rows, concentrated in `node_degree`.

### Outcome, recorded 2026-08-30 — three designs regressed before this one shipped

**Attempt 1: take the tail's *last* integer instead of the first** (matching
the full-text scope's existing rule, and also tightening `_MARKER` to
require `is`/`:`/`-` after "answer" — a bare verb use like "I can **answer**
this using..." was hijacking the tail on 2 of the sample rows). Rescoring
this version found **24 regressions**, in two independent mechanisms:

1. **A response with two "answer" mentions.** `_marker_tail` takes the
   *last* matching occurrence in the whole response, on the assumption any
   match is as good as another. One real response said "Here's a thinking
   process to arrive at the **answer**:" early (a preamble) and "**Answer
   the Question:**" later (a heading whose same-line tail held no digits and
   already, harmlessly, fell through to the full text). Tightening `_MARKER`
   stopped matching the harmless heading — exposing the harmful preamble as
   `found[-1]` instead, and regressing a `cycle_check` row too (a stray "no"
   in restated reasoning after the tightening exposed a different marker).
   **Reverted the `_MARKER` change entirely** — the 2-row payoff didn't
   justify a second failure mode discovered on top of the first.
2. **"Last integer" breaks exactly when the correct value comes *first* in
   the tail.** A real response said "18.The graph is described among the
   nodes: 0, 1, ..., 17." — the marker's own next token (`18`) was already
   correct, but a restated node list glued on with no separator (a likely
   generation artifact) made `17` the tail's last integer instead.

**Attempt 2: cut the tail at a "glued" continuation (no-space
period/paren-before-capital) or a ". If" hedge clause, blank out "node X"
references, then take the last remaining integer.** Fixed attempt 1's
regressions, but found **9 more** on rescoring: the same "continuation after
the real answer" shape recurs with plain sentences that are neither glued
nor hedges — "...which is 11. The sum of the degrees is ... = 64. ... = 32."
and "should be 6 nodes. Let me count again: 0,1,2,3,4,5." both continue past
a correct, already-stated answer with no signal attempt 2's two specific
detectors were built to catch.

**What shipped: restrict the tail to its first sentence** (up to its first
period, whole tail if none), *then* blank out a "node X" reference, *then*
take the last remaining integer. This one rule subsumes both of attempt 2's
special-case detectors (a glued or hedged continuation is, definitionally,
not part of the first sentence) and generalizes to the two new shapes found
against it. Rescoring this version found exactly **2** further changes, and
both are not a bug: `node_degree/2`'s `filler` condition primes the model
with "Node 0 has 8 other nodes in this graph" (a node-*count* filler
sentence, misleading by design for this condition), which the model
misreads as "every node's degree is 8" and says so outright — the old
extractor was reading `0` from "node 0" in that same sentence, not the
model's stated (wrong) conclusion, and happened to match gold `0` by
coincidence. The fix correctly reads what the model actually said; the row
newly (and correctly) counts as a model error, not an extraction bug.

Rescoring `runs/*.jsonl` with the shipped version and rebuilding
`analysis/sweep_frame.csv` flipped **507 rows** from `wrong`/`unparsed` to
`correct` (480 `node_degree`, 28 `node_count`, 1 `edge_count`) — far more
than the ~13-row sample predicted, because the bug wasn't specific to the
sampled rows; it affected this shape everywhere it occurred in the tracked
sweep. Confirmed against the pre-fix frame that every other change is in
these three tasks only, and the only two `correct → wrong` transitions are
the coincidental `node_degree/2` case above, not a regression.

Reproduce with `tests/test_prompts.py::test_extracts_integers` and
`scripts/build_sweep_frame.py`.

## Caveats that travel with specific rows

These are properties of the data, not of the analysis, so they belong here:

- **Anything under `runs/archive/` is not part of the sweep.** It holds the smoke
  test (20 rows carrying `model: gemma4-e4b`) and the 4x-cap regeneration probe.
  `runs/*.jsonl` no longer matches them, and `graphtalk/analysis.py` excludes the
  directory outright rather than matching on filenames.
- **679 rows never terminate** and are truncated at the token cap. Every row now
  carries `hit_cap`, measured on one instrument (`scripts/backfill_hit_cap.py`).
  Split by whether the prompt style is still in use: **309** in the thinking arm
  (`zero_shot`, live, 6.13%), **39** in the plain arms at `zero_shot` (live, 0.77%,
  of which 34 are `gemma4-e4b`), and **331** at `zero_cot` (obsolete, 6.57% — that
  style gets half the token budget). Only the first two affect anything current;
  the `zero_cot` figure mostly explains why that style looks worse than it is, and
  is not worth correcting for a format nothing uses.
  They still *parse*, because the extractor finds an integer in the abandoned
  working, so they score as confident wrong answers rather than as missing. Their
  Of those, 271 predate per-row token counts and their exact keys are in
  `analysis/truncated_keys.json`; the other 45 are recorded directly on the row as
  `hit_cap`, and `graphtalk/analysis.py` prefers that when present. Filter them
  before reporting accuracy: on `gemma4-12b-think` the difference is 81.2% against
  99.1%. The count fell from 350 because the reworded `filler` primer roughly
  halved non-termination in that condition -- see `docs/sweep-findings.md`.
- **`runs/archive/*.redo.shard*.jsonl` are evidence, not answers.** 67 of those
  rows regenerated at a 4x larger cap; 76% still hit it. They exist to show the cap
  was not the cause. Do not merge them into the arm.
- **955 main-sweep rows were generated on CPU** before a driver mismatch was
  found — 438 `gemma4-e4b`, 426 `qwen3-14b`, 91 `qwen3-8b`. Greedy decoding means
  they should match GPU output, but this is unverified.
- **The two arms used different torch builds**, cu130 for the main sweep and
  cu126 for the thinking arm. Same version, same transformers, greedy throughout.
- **2,880 rows are being regenerated under reworded prompts.** The `filler` primer
  and the `edge_existence` question were both reworded in `d7cdcf7..3545662`, so the
  360 affected `zero_shot` rows in each of the eight arms are generated from prompts
  the rest of the sweep never saw. The 1,440 affected rows in the obsolete `zero_cot`
  prompt style are **not** regenerated and keep the original wording — so
  `condition: filler` does not mean one thing across the whole sweep, and the frame
  carries a `wording` column to say which is which. Never pool the two.
- **`prompts.original-wording.jsonl` is what produced those un-regenerated rows.**
  `prompts.jsonl` now holds the revised wording throughout, including for the 360
  `zero_cot` rows whose responses were *not* regenerated — so for those rows the
  tracked prompt file no longer matches the tracked response. This file closes that
  gap: it is the 360 original-wording `zero_cot` prompts, verbatim, and it is the
  prompt of record for the 1,440 responses (360 x 4 plain models) that keep the old
  text. Rebuilding it from `build_prompts.py` is not possible; it exists because a
  response is only interpretable against the exact prompt that produced it.

## Reproducing

> Since the rewording, `build_prompts.py` reproduces `prompts.jsonl` exactly, but
> **not** the prompts behind every tracked response: the 1,440 un-regenerated
> `zero_cot` rows came from `prompts.original-wording.jsonl`. See the caveat above.

```bash
python scripts/build_prompts.py --count 30            # -> prompts.jsonl
python scripts/shortcut_table.py --graphs 500 --json shortcuts.json
python scripts/score_sweep.py --responses $(ls runs/*.jsonl | grep -v smoke) \
                             --shortcuts shortcuts.json
```

Beyond `score_sweep.py`'s per-cell McNemar (underpowered at 30 pairs/cell, see
`docs/sweep-findings.md`), `scripts/check_significance.py` pools pairs across task and
style per (model, condition) for a permutation p-value, a bootstrap CI, and a
Benjamini-Hochberg correction — over both main-sweep accuracy and thinking-arm
non-termination rate. It reads the joined table `build_sweep_frame.py` writes, and
needs the `analysis` extra (`pandas`) installed:

```bash
python scripts/build_sweep_frame.py --responses $(ls runs/*.jsonl | grep -v smoke) \
    --shortcuts shortcuts.json --truncated-keys analysis/truncated_keys.json
python scripts/check_significance.py --frame analysis/sweep_frame.csv --out significance.csv
```

`--count` takes a *prefix* of each split, so a larger value is a strict superset:
`--count 40` contains all 2,520 `--count 30` rows with byte-identical prompt text.
Existing responses therefore remain valid when the sweep is grown, and
`run_sweep.py` regenerates only what is missing.
