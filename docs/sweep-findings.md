# First full sweep: results, and what the design cannot yet answer

Four models over the 2,520-prompt file, 10,080 generations. Verified complete and
clean: every model has all 2,520 unique `(instance_id, condition, style)` keys,
with no duplicates, no gaps and no empty responses, across several preemptions and
resumes.

| model | runtime | notes |
|---|---|---|
| `gemma4-12b` | 12h49m | |
| `qwen3-8b` | 6h51m | |
| `qwen3-14b` | 8h26m | |
| `gemma4-e4b` | preempted and resumed | slowest despite being smallest; see below |

Scored with `scripts/score_sweep.py` against `shortcuts.json` from
`scripts/shortcut_table.py --graphs 500`.

## Headline accuracies

Exact match, pooled over all six tasks. Pooling mixes tasks of very different
difficulty and is shown here only for orientation — read the per-task table from
`score_sweep.py` before drawing conclusions.

> **Revised 2026-08-29, after the prompt-rewording re-run.** Two things changed under
> these numbers and both are now folded in: `scoring._extract_boolean` gained a
> fallback that rescues 276 boolean rows sweep-wide, and the `filler` primer and
> `edge_existence` question were reworded, with 2,880 `zero_shot` rows regenerated
> against the new text. The originally published table is preserved in git at
> `3545662`. One of the three patterns below did not survive -- it turned out to be the
> length control misbehaving rather than a property of primers; see `filler`.

| model | style | `none` | `degree` | `all` | `filler` |
|---|---|---|---|---|---|
| gemma4-e4b | zero_shot | 95.6% | 96.1% | 95.6% | 96.1% † |
| gemma4-e4b | zero_cot | 82.8% | 80.0% | 82.8% | 71.7% ‡ |
| gemma4-12b | zero_shot | 98.3% | **100.0%** | 99.4% | 96.7% † |
| gemma4-12b | zero_cot | 95.6% | 93.9% | 92.8% | 92.2% ‡ |
| qwen3-8b | zero_shot | 89.4% | 91.1% | 90.6% | 81.7% † |
| qwen3-8b | zero_cot | 66.1% | 78.3% | 80.0% | 65.0% ‡ |
| qwen3-14b | zero_shot | 83.3% | 91.7% | 89.4% | 82.2% † |
| qwen3-14b | zero_cot | 71.7% | 78.3% | 75.6% | 69.4% ‡ |

† revised `filler` wording  ‡ **original** `filler` wording.

**The `zero_cot` rows are historical.** That prompt style is retired — chain-of-thought
in this project is the thinking arm, which superseded it — and nothing new will be
generated in it. Two consequences for reading this table: its `filler` cells were not
regenerated, so the two `filler` values in each model's pair answer *different prompts*
and must not be read down the column (`graphtalk/analysis.py` exposes this as the
`wording` column); and `zero_cot` was given half the token budget, so its numbers are
depressed by truncation as well as by the prompt. The `zero_shot` rows are the live
result.

Three patterns were originally stated here as holding across every model. Two still
do. The third turned out to be an artefact of the primer's wording rather than a
property of primers, which is the main thing the 2026-08-29 re-run established:

- **Primers help, modestly.** `degree` or `all` beats `none` in six of eight
  model-style combinations (seven before the re-scoring). The largest gain is
  qwen3-8b under `zero_cot`, 66.1% to 80.0%.
- **`filler` is inert, as the design predicted — the earlier penalty was the
  control being broken.** The hypothesis for this arm was never that filler would
  hurt: `docs/plans/primer-computation.md` §"An inert length control" requires it
  to do nothing, or nothing much, precisely so that the length effect can be
  isolated. The measured penalty of up to 13 points was a surprise, and it was
  written up as a property of primers. It was a property of *that primer*.

  That plan also names the failure mode exactly: "an inert length control, **not a
  misinformation placebo** … a drop in accuracy could mean the model was misled, or
  merely confused by an inconsistent prompt — neither of which is the length effect
  the control exists to isolate." The old wording was cleared of that charge three
  separate times, on the argument that `Node N has <n-1> other nodes` introduces no
  numeral the `none` arm lacks. That argument was wrong.
  `analysis/failure_sample.csv` shows models reading the numeral as a degree claim
  and deriving a complete graph K_n from it in 8 of 9 sampled rows — the placebo the
  design warned against, produced by the safeguard that was supposed to prevent it.

  With a genuinely content-free primer the penalty collapses from a mean of −5.7
  points to +0.6 across the eight arms, and on `gemma4-e4b` `zero_shot` it lands at
  96.1% against 95.6% for `none` — the same thing, which is what the control was
  specified to be. A small residual may survive in the non-thinking arms
  (`qwen3-8b` is still 7.7 points below `none`) and is worth checking; the large
  uniform effect does not. See §"Non-termination responds to the primer", where the
  same reversal shows up on an independent measure.
- **`zero_cot` is worse than `zero_shot`, but a third of the gap is a token budget
  — and the format is obsolete anyway.** The direction holds everywhere. The size
  does not survive scrutiny: `zero_cot` is given **1024** new tokens against
  `zero_shot`'s **2048** while being asked to reason more, and hits that cap **8
  times as often** (331 rows against 39). Those rows parse 84.6% of the time, so
  they score as confident wrong answers rather than as gaps. Excluding them the
  mean gap falls from **+13.0 to +8.5 points**, and on `gemma4-12b` to **exactly
  zero** (98.3% either way) — its entire apparent CoT penalty was truncation.

  **This is not worth fixing, because `zero_cot` is retired.** Chain-of-thought
  in this project is the thinking arm, which superseded it; no further rows will
  be generated in this style. Quote the gap on terminated rows only, or drop the
  claim — but do not spend GPU time raising a budget for a prompt style the
  project no longer uses. The live CoT comparison is thinking arm against plain,
  both at `zero_shot`, and it is not affected by any of this.

These are far above the paper's numbers — Fatemi et al. report 18.8% for PaLM 2
on `node_count`, against 98-100% here. The task is not hard for current models,
which is the root of the problem in the next section.

## The McNemar analysis is underpowered, and n is only half the reason

The proposal's test is McNemar against the `none` control on paired instances.
There are **72 such cells per model** (6 tasks x 2 styles x 6 non-control
conditions), each with 30 pairs.

McNemar uses only **discordant** pairs — instances where the primer flips the
verdict. Concordant pairs contribute nothing. Measured:

| model | median discordant | mean | max | cells with <10 | p<0.05 (uncorrected) | concordance |
|---|---|---|---|---|---|---|
| gemma4-e4b | 5 | 5.1 | 14 | 60/72 | 4 | 82.9% |
| gemma4-12b | **0** | 1.2 | 8 | **72/72** | 3 | **96.0%** |
| qwen3-8b | 3 | 4.7 | 16 | 59/72 | 12 | 84.4% |
| qwen3-14b | 2 | 3.6 | 12 | 68/72 | 9 | 88.0% |

**259 of 288 cells have fewer than 10 discordant pairs**, below any threshold at
which McNemar is interpretable. Across all four models 28 of 288 cells reach
p<0.05 uncorrected, against ~14 expected by chance at that alpha; almost none
would survive a correction for 288 tests.

### Two different causes, needing two different responses

The instinct is to raise `--count`. That helps one cause and not the other.

**Cause 1: genuinely low discordance.** For gemma4-e4b, qwen3-8b and qwen3-14b
the discordance rate is 12-17%. Here more instances is exactly the right fix and
scales linearly: at `--count 500` those models would see 60-85 discordant pairs
per cell, comfortably enough.

**Cause 2: a ceiling.** gemma4-12b scores 98.3% under `none` and 100% under
`degree` on zero_shot. Its 96% concordance is not evidence that the primer does
nothing — it is that **there is almost nothing left to flip**. Only 1.7% of rows
are even available to be improved. No value of `--count` fixes this, because the
limit is headroom rather than sample size. At `--count 500` gemma4-12b would
still reach only ~20 discordant pairs per cell.

The ceiling is the more important finding. The proposal's question — does the
primer help the model reason? — presumes the tasks are hard enough for help to be
visible. For a 12B model on 30-row GraphQA prompts, mostly they are not.

### What `--count 500` would cost, and what it would buy

The published `zero_shot_test` split holds 500 rows per task, so 500 is the
ceiling on `--count`. That is **16.7x** the current sweep: 42,000 prompts per
model, an estimated 130-215 h each at measured single-stream rates, or six to
nine chained 24 h links per model.

`build_prompts.py` takes a *prefix* of each split, so a larger `--count` is a
strict superset of a smaller one: verified that `--count 40` contains all 2,520
`--count 30` keys with byte-identical prompt text. The existing 10,080 responses
are therefore reusable — `run_sweep.py` skips them and generates only the new
rows. Scaling up costs the difference, not the whole.

Even so, this is the strongest argument yet for **batched generation**, which
remains unimplemented. At 3-5x it turns a week per model into a couple of days.

### Cheaper alternatives that cost no GPU time

- **Pool the cells.** 72 tests of 30 pairs each is what destroys the power, both
  through small n and through the multiple-comparison penalty. A single
  mixed-effects logistic model over all 2,520 rows per model, with instance as a
  random effect, uses the same data without slicing it into 72 groups.
- **Report the effect sizes.** The three patterns above are consistent in
  direction across four models and eight model-style combinations. Consistency
  across independent models is evidence that per-cell significance testing at
  n=30 will not produce.
- **Target the headroom.** If the ceiling is the obstacle, the fix is harder
  instances — larger graphs, or the tasks where accuracy is not already near 100%
  — rather than more of the easy ones.

## Design caveats to carry into the write-up

**`zero_shot` now reasons.** The `zero_shot` budget was raised from 64 to 2,048
tokens after measurement showed 64 truncated ~90% of answers mid-sentence (see
`graphtalk/models.py`). With room, these instruction-tuned models narrate their
working before answering on the `zero_shot` prompt too. On a paired sample the
two styles produced near-identical text, opening with the same "### Step 1:
Understand the structure". The contrast between the styles is therefore about
prompt wording, not about whether reasoning happens — and `zero_cot` still scores
6-18 points *worse*, which is a finding in its own right but not the one the
design set out to make.

**955 rows were generated on CPU.** Before a driver mismatch was diagnosed (see
`cluster/README.md`), three jobs silently ran on the host: 438 rows of
gemma4-e4b, 426 of qwen3-14b, 91 of qwen3-8b. Greedy decoding and identical
arithmetic mean they should match GPU output exactly, but this has **not been
verified**. Regenerating a sample on GPU and diffing would retire the caveat.

**Thinking modes are off for both families, deliberately.** Gemma 4 defaults to
thinking off; Qwen3 defaults to thinking *on*. Left alone, Qwen would have
reasoned in a hidden `<think>` channel on both prompt styles, collapsing the
style contrast for one family only. `enable_thinking=False` on the Qwen specs
aligns them. Verified: zero `<think>` blocks across all 10,080 responses.

**`gemma4-e4b` is not a 4B model.** It generates 1,244 mean characters against
631-870 for the others, and its per-character throughput (45.6 chars/s) matches
the 12B and 14B models rather than beating them. The "E" is an *effective*
parameter count; the checkpoint is 15 GB, so roughly 7.5B parameters in bf16, and
it loads through the multimodal `AutoModelForImageTextToText` path. Slowest of
the four, despite the name.

## Next

1. Read the per-task output of `score_sweep.py`; the pooled table above hides
   which tasks are at ceiling and which have headroom.
2. Decide between pooled analysis and a larger `--count` on the basis of effect
   sizes, not on the per-cell p-values, which n=30 cannot deliver.
3. If `--count` grows, land batching first.

---

# The thinking arm

The same four checkpoints over the same 1,260 `zero_shot` prompts with the native
reasoning channel enabled -- so these rows pair against the main sweep row for
row, with the thinking channel as the only difference. Both families have such a
channel and their defaults are opposite (Gemma off, Qwen on), which is why the
main sweep pins both to off; this arm pins both to on. Qwen marks it with
`<think>` blocks, Gemma with a `thought` section.

## Some responses never terminate, and that is the headline

| model | rows | non-terminating | accuracy (terminating) | accuracy (naive) |
|---|---|---|---|---|
| `gemma4-e4b-think` | 1260 | **0** (0.0%) | 90.2% | 90.2% |
| `gemma4-12b-think` | 1260 | **282** (22.4%) | **99.1%** | 81.2% |
| `qwen3-8b-think` | 1260 | 49 (3.9%) | 85.7% | 84.8% |
| `qwen3-14b-think` | 1260 | 19 (1.5%) | 94.8% | 94.1% |

Read the naive column only to see how badly it misleads. A response cut off
mid-working still *parses*, because the answer extractor finds an integer in the
abandoned arithmetic, so a non-terminating row scores as a confident wrong answer
rather than as missing data. On `gemma4-12b` that drags a genuine 99.1% down to a
reported 81.2%.

**This is not a token budget that was set too low.** The rows were regenerated at
32,768 tokens, four times the original cap, and **76% of them still hit it**, with
the median landing exactly on the cap -- the distribution is censored wherever the
cap is put. Non-terminating rows by task: `edge_count` 203, `connected_nodes` 78,
`node_degree` 30, `cycle_check` 26, `edge_existence` 13.

Reading one in full shows why. The model enumerates every node's adjacency list
correctly, verifies all n(n-1)/2 pairs, and then begins re-verifying lists it has
already checked: *"Wait, let me re-check Node 14's connections one more time…
Wait, let me re-check Node 15's connections one more time…"*. The answer is
effectively in hand by the midpoint; what follows is unbounded self-doubt. Each
pass is fresh text, so a repetition penalty would not fire, and the model never
emits the `**Answer:**` marker a stop sequence could match. Graph size raises the
odds without explaining it -- non-terminating rows average 14.0 nodes against
12.5 for terminating ones, but both span the full 5-19 range.

## Non-termination responds to the primer

Rate by condition, pooled over all four models (n=720 per condition):

| `filler` | `all` | `degree` | `components` | `rwse` | `none` | `clustering` |
|---|---|---|---|---|---|---|
| **4.3%** † | 5.6% | 5.8% | 5.8% | 6.2% | 7.4% | **8.1%** |

† `filler` regenerated under the revised wording; the other six columns are the
original rows. **The two are measured by different instruments** — `filler` from the
`hit_cap` flag `scripts/run_sweep.py` now records per row, the rest from the
hand-labelled `analysis/truncated_keys.json`. Both are meant to mean "hit the token
cap", but they were produced by different routes and no row carries both, so they
cannot be cross-validated against each other. Read the `filler` column against the
others with that in mind.

**This reverses the finding previously stated here, in the direction the design
expected.** Under the original wording `filler` was the *highest* rate at 9.4%,
above the `none` control at 7.4%, and that was read as evidence that padding without
information harms the model — "given padding of the same shape it has more to verify
and nothing to verify it with." The control was specified to be inert; this was the
anomaly, not the prediction. Under a genuinely content-free primer `filler` is the
*lowest* rate at 4.3%, below every informative condition. On `gemma4-12b-think`, where the rates are large enough
to see clearly, `filler` is 15.0% against `none` at 24.4% and `clustering` at 26.7%.

The mechanism in that original explanation was wrong because its premise was wrong.
`Node N has <n-1> other nodes in this graph` is not padding — it is a false
statement about the graph, and `analysis/failure_sample.csv` catches models spending
their budget trying to reconcile it (*"If D_i = 12 for all 13 nodes, the graph must
be a complete graph K_13"*). What raised non-termination was the contradiction, not
the length. Remove the contradiction and the condition becomes the cheapest of the
seven.

So the two measures that were said to converge were not independent evidence of a
shared effect; they were two symptoms of one wording defect. The convergence
argument should be retired rather than restated with new numbers.

**It also gives `gemma4-12b` somewhere to move.** That model's accuracy is pinned
at 97.5-99.1% with almost no headroom, which is what made its McNemar cells
useless. Non-termination is an outcome variable that responds to the manipulation
on precisely the model whose accuracy cannot.

## Non-termination is not only a thinking-arm problem

`scripts/backfill_hit_cap.py` put every row in the sweep on the same instrument by
re-tokenizing responses against their own budget, validated at **100% agreement**
(45/45 capped, 2,835/2,835 not) against the 2,880 rows carrying the generator's own
count. That was done to remove an instrument confound from the `filler` comparison
above. It also turned up something no one was looking for.

**The main sweep has 370 truncated rows**, where non-termination was treated as a
property of thinking mode. But almost all of them are in `zero_cot`, the obsolete
prompt style, which gets half the token budget. Split by whether the format is
still in use:

| | capped | rate |
|---|---|---|
| plain arms, `zero_shot` (live) | 39 / 5,040 | **0.77%** |
| thinking arm, `zero_shot` (live) | 309 / 5,040 | 6.13% |
| plain arms, `zero_cot` (obsolete) | 331 / 5,040 | 6.57% |

**In the live format this is small.** 34 of those 39 rows are `gemma4-e4b`, which
gains 2.4 points when they are excluded (94.4% → 96.8%); every other plain arm
moves by 0.2 points or less, and `qwen3-14b` has none at all. So the practical
correction is one arm, and the headline `zero_shot` numbers stand.

It is worth knowing anyway, for two reasons. `gemma4-e4b` is the arm this document
already singles out as anomalous, and 2.4 points of that anomaly is truncation
rather than capability. And these rows average **9.5% accuracy while parsing 84.6%
of the time** — they surface as wrong answers, not as gaps, which is why nothing
caught them: `analysis/truncated_keys.json` never covered the main sweep at all.

The cross-check also found the hand-curated file itself is imperfect: two of its
271 labelled rows (`gemma4-12b-think` `edge_count/25` and `connected_nodes/19`)
re-tokenize well short of the budget and end with a complete `A: …` answer. They
terminated. Both are retained in the file as a record of what was labelled; the
frame is driven by the instrument, not the file.

## The `edge_existence` question was ambiguous, and it mattered

`edge_existence` is graded on a single edge (`graph.has_edge`), but the published
question asked *"Is node A connected to node B?"* — and "connected to" also means
reachable-by-any-path. Of the 30 instances, 12 have gold `Yes`, **14 have gold `No`
with a path present**, and 4 are genuinely unreachable, so 47% of instances flip
under a reachability reading. The question was reworded to *"Does an edge exist
between Node A and Node B?"* and all 2,880 affected `zero_shot` rows regenerated.

The effect is large, and it is concentrated exactly where the ambiguity lives:

| arm | prior "Yes" rate on path-only pairs | after | accuracy Δ |
|---|---|---|---|
| `gemma4-12b` | 3.6% | 0.0% | +1.7 |
| `gemma4-12b-think` | 8.3% | 0.0% | +3.9 |
| `gemma4-e4b-think` | 19.0% | 0.0% | +8.9 |
| `qwen3-14b` | 17.9% | 0.0% | +8.3 |
| `gemma4-e4b` | 21.4% | 0.0% | +9.4 |
| `qwen3-14b-think` | 27.4% | 0.0% | +12.8 |
| `qwen3-8b-think` | 39.3% | 0.0% | +17.8 |
| `qwen3-8b` | 38.1% | 0.0% | +18.9 |

Mean +10.2 points, and the gain per arm tracks that arm's prior rate of answering
`Yes` on path-only pairs at **r = +0.95**. The `edge` and `unreachable` instance
classes — the ones a reachability reading answers the same way — barely move. Seven
of the eight arms reach exactly 100%.

Two things are worth keeping from this beyond the number:

**It is a per-model effect, not a uniform one.** The gain runs from +1.7 on
`gemma4-12b` to +18.9 on `qwen3-8b`, because the stronger models were already
resolving "connected" as adjacency and the weaker ones were not. An analysis that
looked only at `gemma4-12b` would have concluded the ambiguity was inert — and one
did, in `docs/DATA.md`, before the re-run tested it.

**The rewording is not free.** On `qwen3-8b` the `edge (gold Yes)` class regressed
from 100.0% to 97.2%: two rows in 72 now answer `No` where the old wording got them
right. The trade is heavily favourable, but it is a trade.

Reproduce with `scripts/rewording_effect.py`.

## `connected_nodes`'s "None" answers were scored wrong, narrowly

A smaller sibling of the previous finding, same root cause -- extraction
matched the dataset's exact answer spelling and nothing else. `connected_nodes`
spells an isolated node's gold answer `"No nodes."`; `graphtalk/scoring.py`
only recognised that literal phrase, so a model answering `"None"`/`"None."`
fell through to a stray digit earlier in the response and scored `wrong`.

Only one of the 30 instances (`connected_nodes/2`) has gold `No nodes.`, so
the fix's reach was bounded before it was measured: rescoring `runs/*.jsonl`
after adding a `"None"`/`"None."` recognizer (anchored to end-of-line so a
sentence like "None of the nodes are directly connected" is not misread)
flipped exactly **8 rows** from `wrong` to `correct`, all `connected_nodes/2`,
7 of them on `gemma4-12b` and 1 on `gemma4-e4b-think`. No other task or
instance changed, and no non-terminating row was touched. Left out of scope:
one `connected_nodes/2` row spelled `"A: []"` instead, which the fix does not
recognise -- a deliberate precision-over-recall boundary, not a miss.

Full writeup in `docs/DATA.md`; reproduce with
`tests/test_prompts.py::test_extracts_node_lists` and
`scripts/build_sweep_frame.py`.

**Round two.** Both boundaries called out above turned out to be worth
closing, plus two more real shapes surfaced by reading
`analysis/failure_sample.csv` directly: `"None"` glued onto a sentence with
no separator (`"...the list is empty.None"`) and the token wrapped in
markdown emphasis or a trailing parenthetical (`"**A: None**"`, `"A: [] (or
None, depending on expected format for an empty list)"`). Checked for false
positives *before* changing anything: across all 2,436 non-empty-gold
`connected_nodes` rows, zero have a last line ending in `none`/`None.`, and
zero contain `[]` anywhere -- both boundaries were safe to relax.

A second, unrelated bug was found and fixed alongside it: `_marker_tail`
grabs text after the *last* "answer"-labeled mention anywhere in the
response, which can be a mid-reasoning heading rather than the true
conclusion. `_extract_node_list` now scans the full response's last lines
*before* consulting the marker tail (previously the reverse), so a stray
digit in a heading like `"3. **Determine the Answer:** ... node 0 has no
listed neighbors."` can no longer shadow the real answer one line later.

The predicted effect was "a handful more `connected_nodes/2` rows." The
measured effect was **53 rows across 12 instances**: the stale-marker fix
also corrected several *non-empty*-gold rows with the identical bug (gold
`"1, 2, 3, 4, 5, 6, 7, 8"` had been extracted as the stray digit `"8"`; now
the full list). Confirmed precisely against the pre-fix frame: every change
was `unparsed`/`wrong → correct`, zero were `correct → anything else` --
broader than planned because the underlying bug was shared, not because
anything was left unverified.

Full writeup, including the exact false-positive checks, in `docs/DATA.md`;
reproduce with `tests/test_prompts.py::test_extracts_node_lists` and its
stale-marker and trailing-decoration regression tests, and
`scripts/build_sweep_frame.py`.

## `edge_existence` conclusions stated without "yes"/"no" were unparsed

Same shape of bug, `_extract_boolean` this time: it only recognised a
standalone `yes`/`no` token, so a response concluding *"An edge exists between
Node A and Node B."* -- echoing the live question's own wording -- scored
`unparsed`. 17 of 2,520 tracked rows were affected; only 10 (5 live-`zero_shot`
"edge exists" rows, 5 retired-`zero_cot` "is connected to" rows) are a
paraphrase worth fixing -- the rest are genuine refusals or truncations, which
must stay `unparsed`.

**The obvious fix regressed 44 rows before it shipped.** Feeding the new
patterns into the same position-based "last occurrence wins" comparison
already used for bare `yes`/`no` let a mid-response restatement of the
question ("...to determine **if an edge exists** between X and Y...") outrank
the response's own correct bare `"No"` when the restatement happened to sit
later in the text, and separately coerced an explicit refusal ("we cannot
determine if an edge exists...") into a stated `"Yes"`. A third real row
defeated the first attempt at a guard too: a true but irrelevant "is connected
to" sentence about a *different*, real edge, stated early while the response
summarised the graph before correctly refusing to answer the queried pair.

The fix that shipped: the new patterns are a **fallback only**, consulted
after both scopes are confirmed to have no bare token anywhere (so they can
never override an already-resolved answer), gated to exclude a question/hedge
lead-in ("if", "whether", "determine", ...), and checked only against each
scope's **last non-empty line** rather than searched anywhere in it -- the
same rule `_extract_node_list` already uses for `No nodes`. Rescoring against
this version flipped exactly the predicted 10 rows, `unparsed → correct`, with
zero rows regressed and the two refusals still correctly `unparsed`.

Full writeup, including the two regressions in more detail, is in
`docs/DATA.md`; reproduce with
`tests/test_prompts.py::test_extracts_edge_existence_paraphrases` and its
three regression-specific siblings, and `scripts/build_sweep_frame.py`.

## `_extract_integer` was reading the queried node's own id as the answer

The largest fix this session, and the one that took the most attempts.
`_extract_integer` (`node_count`/`edge_count`/`node_degree`) preferred a
marker's tail over the full response, and inside that tail took the *first*
integer -- which is the queried node's own id whenever the response says
"The degree of node 7 is 2." (id before value). A 23-row sample of
`wrong`/`unparsed` rows found this in all 11 sampled `node_degree` rows and
both sampled `node_count` rows; `edge_count` (5 rows) and `cycle_check`
(1 row) in the same sample were genuine model errors, extraction already
correct.

Two intermediate designs shipped-then-reverted before the fix that stuck,
each caught by the mandatory full-sweep rescore, not by inspection:

- **Take the tail's last integer instead of the first**, plus tightening
  `_MARKER` so a bare verb "answer" doesn't count as a label: regressed
  **24 rows**. A response with two "answer" mentions -- a harmful early
  preamble and a harmless later heading -- had the tightened regex stop
  matching the harmless one, exposing the harmful one as the selected
  marker instead (plus a `cycle_check` regression from the same mechanism).
  The `_MARKER` change was reverted outright. Separately, "last integer"
  broke exactly when the correct value came *first* in the tail, followed
  by a glued-on restated node list.
- **Cut the tail at a glued continuation or a ". If" hedge, blank "node X"
  references, take the last remainder**: fixed those 24, but found **9
  more** -- the same "continuation past the real answer" shape recurs in
  plain sentences with no hedge word or glue artifact to detect.

**What shipped:** restrict the tail to its first sentence, blank a "node X"
reference within it, take the last remaining integer. One rule, not two
special-case detectors, and it generalizes to shapes neither detector
caught. Final rescore: **2** further changes, both a correctly-surfaced
model error (`node_degree/2`'s `filler` primer, "Node 0 has 8 other nodes,"
misreads as "every node has degree 8," and the model says so outright --
the old code was accidentally reading `0` from "node 0" in that sentence,
not the model's real, wrong answer), not a regression.

**Measured effect: 507 rows**, `node_degree` 480 / `node_count` 28 /
`edge_count` 1, `wrong`/`unparsed → correct` -- far beyond the ~13-row
sample's prediction, because the bug wasn't sample-specific; it hit this
tail shape everywhere it occurred in the tracked sweep. `edge_count`'s much
smaller share confirms the earlier finding that most of that task's error
is genuine arithmetic mistakes, not extraction.

Full writeup, including the three-attempt history, is in `docs/DATA.md`;
reproduce with `tests/test_prompts.py::test_extracts_integers` and
`scripts/build_sweep_frame.py`.

## Provenance

The thinking arm was generated with `graphtalk-cu126` (torch 2.13.0+cu126) while
the main sweep used the cu130 build; same torch version, same transformers,
greedy decoding throughout. The regeneration evidence is kept in
`runs/archive/*.redo.shard*.jsonl` -- 67 rows re-run at 32,768 tokens, retained because
they are the evidence that the cap was never the cause, not because they are
usable answers.

**The 2026-08-29 prompt-rewording re-run.** 2,880 rows -- the 360 affected
`zero_shot` rows in each of the eight arms -- were regenerated after the `filler`
primer and `edge_existence` question were reworded. Each arm's original build was
kept (cu130 for the four plain arms, cu126 for the four thinking arms), so the
re-run adds no new torch stratum. The 1,440 affected rows in the obsolete `zero_cot`
prompt style were **not** regenerated and retain the original wording; the prompts
that produced them are preserved in `prompts.original-wording.jsonl`, and the frame's
`wording` column marks which is which. Rows generated from this point carry
`n_new_tokens` and `hit_cap`, so non-termination is recorded per row rather than
depending on `analysis/truncated_keys.json`; older rows still depend on it.

Total non-terminating rows is now **316**, not the 350 previously reported: 271 from
the ground-truth file for rows not regenerated, plus 45 recorded directly. The drop
is concentrated in `filler`, for the reason given above.
