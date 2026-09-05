# First full sweep: results, and what the design cannot yet answer

Four models over the 1,260-prompt file, 5,040 generations. Verified complete and
clean: every model has all 1,260 unique `(instance_id, condition, style)` keys,
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

| model | `none` | `degree` | `all` | `filler` |
|---|---|---|---|---|
| gemma4-e4b | 95.6% | 96.1% | 95.6% | 96.1% |
| gemma4-12b | 98.3% | **100.0%** | 99.4% | 96.7% |
| qwen3-8b | 89.4% | 91.1% | 90.6% | 81.7% |
| qwen3-14b | 83.3% | 91.7% | 89.4% | 82.2% |

All at `zero_shot`, the only prompt style this project generates or scores.

Two patterns hold across every model:

- **Primers help, modestly.** `degree` or `all` beats `none` in every model.
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
  points to +0.6 across the four models, and on `gemma4-e4b` it lands at
  96.1% against 95.6% for `none` — the same thing, which is what the control was
  specified to be. A small residual may survive on `qwen3-8b` (still 7.7 points
  below `none`) and is worth checking; the large uniform effect does not. See
  §"Non-termination responds to the primer", where the same reversal shows up on
  an independent measure.

Chain-of-thought in this project is measured by the thinking arm against the
plain arms, both at `zero_shot` — see below.

These are far above the paper's numbers — Fatemi et al. report 18.8% for PaLM 2
on `node_count`, against 98-100% here. The task is not hard for current models,
which is the root of the problem in the next section.

## The McNemar analysis is underpowered, and n is only half the reason

The proposal's test is McNemar against the `none` control on paired instances.
There are **36 such cells per model** (6 tasks x 6 non-control conditions),
each with 30 pairs.

McNemar uses only **discordant** pairs — instances where the primer flips the
verdict. Concordant pairs contribute nothing. Measured:

| model | median discordant | mean | max | cells with <10 | p<0.05 (uncorrected) | concordance |
|---|---|---|---|---|---|---|
| gemma4-e4b | 0.5 | 1.6 | 8 | **36/36** | 0 | 94.7% |
| gemma4-12b | **0** | 0.4 | 3 | **36/36** | 0 | **98.6%** |
| qwen3-8b | 0.5 | 1.8 | 9 | **36/36** | 2 | 94.0% |
| qwen3-14b | 0.5 | 0.9 | 8 | **36/36** | 0 | 96.9% |

**Every one of the 144 cells has fewer than 10 discordant pairs**, below any
threshold at which McNemar is interpretable. Across all four models 2 of 144
cells reach p<0.05 uncorrected, against ~7 expected by chance at that alpha —
fewer false positives than chance alone predicts, not more; none would survive
a correction for 144 tests.

### Two different causes, needing two different responses

The instinct is to raise `--count`. That helps one cause and not the other.

**Cause 1: genuinely low discordance.** All four models now show single-digit
discordance rates (1.4-6.0%) at `zero_shot` — lower than the 12-17% measured
when `zero_cot` rows were still pooled in, which were driving much of the
earlier discordance with noisier output. More instances still helps
proportionally: at `--count 500`, even the highest rate here (qwen3-8b, 6.0%)
would yield only ~30 discordant pairs per cell — better, but still thin.

**Cause 2: a ceiling.** gemma4-12b scores 98.9% under `none` and 100% under
`degree`; gemma4-e4b is close behind at 96.7% under `none`. Both leave little
headroom to flip — gemma4-12b would reach only ~7 discordant pairs per cell
even at `--count 500`, because the limit is headroom, not sample size.
qwen3-8b and qwen3-14b (90.0% and 88.9% under `none`) have real headroom left
but still show low discordance, meaning the primer rarely changes the verdict
even on rows the model gets wrong at baseline — closer to Cause 1 than Cause 2,
but not cleanly either.

The ceiling is the more important finding. The proposal's question — does the
primer help the model reason? — presumes the tasks are hard enough for help to be
visible. For a 12B model on 30-row GraphQA prompts, mostly they are not.

### What `--count 500` would cost, and what it would buy

The published `zero_shot_test` split holds 500 rows per task, so 500 is the
ceiling on `--count`. That is **16.7x** the current sweep: 21,000 prompts per
model. The 130-215 h estimate measured against the original, larger
`zero_cot`-included sweep no longer applies directly -- re-measure single-stream
rate against the current `zero_shot`-only sweep before re-quoting a chain length.

`build_prompts.py` takes a *prefix* of each split, so a larger `--count` is a
strict superset of a smaller one: verified that `--count 40` contains all 1,260
`--count 30` keys with byte-identical prompt text. The existing 5,040 responses
are therefore reusable — `run_sweep.py` skips them and generates only the new
rows. Scaling up costs the difference, not the whole.

Even so, this is the strongest argument yet for **batched generation**, which
remains unimplemented. At 3-5x it turns a week per model into a couple of days.

### Cheaper alternatives that cost no GPU time

- **Pool the cells.** 36 tests of 30 pairs each is what destroys the power, both
  through small n and through the multiple-comparison penalty. A single
  mixed-effects logistic model over all 1,260 rows per model, with instance as a
  random effect, uses the same data without slicing it into 36 groups.
- **Report the effect sizes.** The two patterns above are consistent in
  direction across all four models. Consistency across independent models is
  evidence that per-cell significance testing at n=30 will not produce.
- **Target the headroom.** If the ceiling is the obstacle, the fix is harder
  instances — larger graphs, or the tasks where accuracy is not already near 100%
  — rather than more of the easy ones.

## Design caveats to carry into the write-up

**Truncation impersonates a finding, and has done so three times.** Filter
capped responses out before comparing anything, and say so when reporting.

A response cut off at the token budget still *parses* — the extractor pulls an
integer or a yes/no out of the abandoned working — so it scores as a confident
wrong answer rather than as missing data. It is invisible in a parse rate and
invisible in a gap; it shows up only as a condition or a model looking worse.
Two separate results in this document turned out to be that and nothing else:

- **"`filler` hurts"** — the length control scored below the no-primer control
  on accuracy *and* had the highest non-termination rate. Both measures were
  responding to the primer's false numeral, not to padding.
- **"GoT naming costs 4 points"** — GoT names are ~8% longer, on an arm already
  truncating 20% of responses. On terminated rows the effect is −0.1.

A third, "`zero_cot` is worse than `zero_shot`", was the same pattern (`zero_cot`
ran at half the token budget and truncated 8x as often) but is no longer
reproducible from tracked data -- `zero_cot` and its 5,040 rows were removed
from the project once the format was fully retired.

Every one of these looked statistically solid before the capped rows came out.
`hit_cap` is now recorded on every row (`scripts/backfill_hit_cap.py`), so there
is no longer an excuse for pooling them in; `scripts/check_significance.py` and
`scripts/naming_effect.py` both exclude them by default.


**`zero_shot` now reasons.** The `zero_shot` budget was raised from 64 to 2,048
tokens after measurement showed 64 truncated ~90% of answers mid-sentence (see
`graphtalk/models.py`). With room, these instruction-tuned models narrate their
working before answering, opening with headers like "### Step 1: Understand the
structure" unprompted. Chain-of-thought in this project is measured by the
thinking arm (native reasoning) against this same `zero_shot` prompt, not by a
separate prompt style.

**955 rows were generated on CPU.** Before a driver mismatch was diagnosed (see
`cluster/README.md`), three jobs silently ran on the host: 438 rows of
gemma4-e4b, 426 of qwen3-14b, 91 of qwen3-8b. Greedy decoding and identical
arithmetic mean they should match GPU output exactly, but this has **not been
verified**. Regenerating a sample on GPU and diffing would retire the caveat.

**Thinking modes are off for both families, deliberately, in the plain arms.**
Gemma 4 defaults to thinking off; Qwen3 defaults to thinking *on*. Left alone,
Qwen would have reasoned in a hidden `<think>` channel even in the plain arm,
collapsing the plain-vs-thinking-arm contrast for one family only.
`enable_thinking=False` on the Qwen specs aligns them. Verified: zero `<think>`
blocks across all 5,040 plain-arm responses.

**`gemma4-e4b` is not a 4B model.** It generates 1,244 mean characters against
631-870 for the others, and its per-character throughput (45.6 chars/s) matches
the 12B and 14B models rather than beating them. The "E" is an *effective*
parameter count; the checkpoint is 15 GB, so roughly 7.5B parameters in bf16, and
it loads through the multimodal `AutoModelForImageTextToText` path. Slowest of
the four, despite the name.

## Next

> **See also `docs/primer-effects-and-power.md` (2026-09-05).** It re-reads this
> data through the shortcut bar and locates the power failure precisely: five of
> six tasks are saturated, `edge_count` is the only one with headroom, and it is
> also the cell whose `degree` bar is 1.00 -- so the arm with power is the
> contaminated one, while `edge_count` x {`components`, `rwse`, `clustering`}
> (bars 0.02-0.15) sits at n=30. It proposes that as the experiment to run, and
> documents `qwen3-0.6b`'s off-by-one `node_count` artifact plus the GoT
> desubstitution trap that corrupts any fresh `connected_nodes` analysis.


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

**The plain (non-thinking) arms have 39 truncated rows**, where non-termination
was treated as a property of thinking mode. (An earlier version of this
finding measured 370 and attributed almost all of it to `zero_cot`, the
retired prompt style, which got half the token budget -- that comparison is
no longer reproducible now that `zero_cot` and its rows are gone; what
survives is the plain-arm count below.)

| | capped | rate |
|---|---|---|
| plain arms | 39 / 5,040 | **0.77%** |
| thinking arm | 309 / 5,040 | 6.13% |

**This is small.** 34 of those 39 rows are `gemma4-e4b`, which
gains 2.4 points when they are excluded (94.4% → 96.8%); every other plain arm
moves by 0.2 points or less, and `qwen3-14b` has none at all. So the practical
correction is one arm, and the headline `zero_shot` numbers stand.

It is worth knowing anyway, for two reasons. `gemma4-e4b` is the arm this document
already singles out as anomalous, and 2.4 points of that anomaly is truncation
rather than capability. And these rows average **7.7% accuracy while parsing 89.7%
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

## Renaming every node changes nothing

The second sweep renames nodes from `0, 1, 2, ...` to Game-of-Thrones characters
(`Ned`, `Catelyn`, `Daenerys`, ...) throughout the primer, the encoding and the
question, via `--node-naming got`. Everything else is held fixed: the same 30
graphs per task, the same queries, the same seven primer conditions, the same
eight arms, `zero_shot` only. So the two sweeps pair row for row on
`(model, instance_id, task, condition, style)` and differ in nothing but the
names. 10,080 generations.

The effect is nil, in every arm:

| arm | integer | got | delta | 95% CI | p |
|---|---|---|---|---|---|
| `gemma4-e4b` | 97.9% | 97.6% | −0.3 | [−1.4, +0.8] | 0.60 |
| `gemma4-12b` | 98.6% | 98.4% | −0.2 | [−0.6, +0.2] | 0.69 |
| `qwen3-8b` | 90.2% | 90.1% | −0.1 | [−1.6, +1.4] | 0.87 |
| `qwen3-14b` | 90.5% | 90.8% | +0.3 | [−1.0, +1.6] | 0.73 |
| `gemma4-e4b-think` | 98.4% | 97.3% | −1.1 | [−2.4, +0.1] | 0.07 |
| `gemma4-12b-think` | 100.0% | 99.9% | −0.1 | [−0.4, +0.0] | 0.25 |
| `qwen3-8b-think` | 99.8% | 99.7% | −0.2 | [−0.8, +0.3] | 0.75 |
| `qwen3-14b-think` | 99.8% | 100.0% | +0.2 | [+0.0, +0.6] | 0.50 |

Pooled: **−0.19 points** (sd 0.42, range −1.1 to +0.3). **Zero of eight**
significant after Benjamini-Hochberg. Paired permutation test and cluster
bootstrap from `graphtalk/significance.py`, clustered on `instance_id` because
one graph recurs under all seven conditions; truncated responses excluded.
Reproduce with `scripts/naming_effect.py`.

This is not an underpowered null. The intervals are roughly ±1.5 points over
1,200+ paired rows per arm, so it is a *precise* zero, which says more than
"no significant difference".

**What it bears on.** Fatemi et al.'s central result — quoted in
`graphtalk/primers.py` as the reason every condition must share a format — is
that phrasing alone moves accuracy by tens of points. Node naming is one axis of
phrasing, and it is bounded here at well under a point. That does not refute
them: they vary whole encoding schemes (`adjacency` vs `incident` vs
`friendship`), of which naming is one component, and this project holds the
`incident` edge encoding fixed throughout. What it does say is that the axis
often assumed to carry the effect — whether nodes look like integers or like
people — is not where it lives, at least for these models on these tasks.

### The one apparent effect was truncation, again

`gemma4-12b-think` initially measured −4.0 points, p=0.0005, comfortably
significant. It is an artifact, and worth recording because of how it arose.

GoT names cost that arm about **8% more response tokens**. It was already
truncating **20%** of its responses at the budget, so the extra length tips
another 4% over the cap: capped rises 20.0% → 24.0%, which is the apparent
accuracy drop to the decimal. On responses that actually terminate, both namings
score ~100%.

The reason it looked real for as long as it did is that the truncated rows were
not being excluded — the filter in `scripts/naming_effect.py` tested a field name
that does not exist on those records, so it silently passed everything through.
The script now drops capped rows by default and `--keep-capped` opts back in; the
difference between the two readings is the finding.

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
re-run adds no new torch stratum. The frame's `wording` column still marks which
cells the rewording touched (`revised` vs `unaffected`) — it no longer needs an
`original` value, since the only rows that kept the pre-rewording text were the
now-removed `zero_cot` ones. Rows generated from this point carry `n_new_tokens`
and `hit_cap`, so non-termination is recorded per row rather than depending on
`analysis/truncated_keys.json`; older rows still depend on it.

**The GoT-naming sweep, 2026-09-02.** A second 10,080-row sweep over the same
graphs and queries with Game-of-Thrones node names, `zero_shot` only across all
eight arms, generated entirely on `graphtalk-cu126` -- so unlike the integer
sweep it carries no build split at all. Its 1,819 first-attempt rows were
discarded rather than resumed, to keep that uniform; they are kept outside the
repo. Scored separately from the integer rows and never pooled with them.

Total non-terminating rows is now **316**, not the 350 previously reported: 271 from
the ground-truth file for rows not regenerated, plus 45 recorded directly. The drop
is concentrated in `filler`, for the reason given above.

## Missing instances skew toward larger graphs, and it's the same two tasks

`scripts/characterize_non_termination.py` re-derives, from the `excluded` bound's
own pairing join, exactly which `(model, condition, instance_id)` rows contribute
zero surviving pairs (`n_instances_missing` in `analysis/significance_report.csv`) --
52 across the whole main sweep. They are **not** spread evenly: 45 are `edge_count`,
7 are `cycle_check`, and every other task has zero. Their mean graph size is **16.8
nodes** against **12.8** for the corpus as a whole -- missingness concentrates on
the larger, harder end of the size range, on exactly the two tasks that require
exhaustive per-edge or per-node reasoning (counting every edge; walking the whole
graph for a cycle). This is a mild selection-bias risk `analysis/significance_report.csv`'s
`n_instances_missing` column makes visible but can't correct: the `excluded` bound's
main-sweep accuracy numbers are drawn from a corpus slightly biased toward smaller/
easier `edge_count`/`cycle_check` instances than the full 180.

## Why `gemma4-e4b` truncates more: frequency, not length, and the same two tasks again

Main-sweep non-termination rate by model: `gemma4-12b` 0.2%, `gemma4-e4b` **2.7%**,
`qwen3-14b` 0%, `qwen3-8b` 0.2%. The obvious guess -- that `gemma4-e4b` just writes
longer before getting cut off -- is wrong: its mean length on non-terminating rows
(4,020 chars) is in the same range as `gemma4-12b`'s (3,394) and `qwen3-8b`'s
(4,031). It truncates about as *late* as everyone else, just more *often*.

`edge_count` (27 of 34) and `cycle_check` (7 of 34) account for **all** of
`gemma4-e4b`'s non-terminating rows -- the identical two tasks driving the missing-
instances finding above.

A stratified read of raw `gemma4-e4b` non-terminating responses (in
`analysis/non_termination_sample.csv`) shows a consistent pattern on both tasks:
exhaustive, node-by-node or edge-by-edge manual re-verification rather than a
direct computation. On `edge_count`, it re-derives the adjacency list edge by edge
with running "already counted" bookkeeping instead of summing degrees and dividing
by two. On `cycle_check`, it narrates a full manual DFS, checking every neighbor of
every node against the current stack. Both scale with graph size in a way a more
direct method wouldn't -- consistent with missing instances skewing larger above.
(An earlier reading also attributed a handful of `edge_existence` non-terminating
rows to the same exhaustive-verification instinct; those were entirely `zero_cot`
rows and no longer exist -- `gemma4-e4b`'s live non-terminating rows are `edge_count`
and `cycle_check` only.) This reads as `gemma4-e4b` defaulting into exhaustive
self-verification on tasks whose
direct-computation shortcut it doesn't reliably take, not a token-budget or
formatting problem -- a generation-behavior question, not a scoring-pipeline one,
and out of scope for anything `check_significance.py` can fix.

`edge_count`'s fragility is not unique to `gemma4-e4b`, only most visible there
as non-termination. `check_significance.py --metric mae` (`analysis/README.md`,
"The `mae` metric mode") scores mean absolute error instead of exact-match on
the three integer tasks, and `edge_count` carries essentially all of that
signal too, on `zero_shot` data alone: `qwen3-8b`/`rwse` (mae_delta −12.8,
p=0.001) and `qwen3-14b`/`clustering` (mae_delta +0.93, p=0.008) both survive
significance, in cells where exact-match accuracy shows no effect at all --
the same task where a bad primer's damage tends to land, whether it surfaces
as never finishing (`gemma4-e4b`) or as a wrong answer landing further from
correct. The `filler`/`edge_count` pattern reported by an earlier, pooled
`zero_shot`+`zero_cot` reading of this data did not survive restricting to
`zero_shot` alone.

## The `filler` instrument confound is closed

`non_terminating_source` (`generator` = `scripts/run_sweep.py`'s own recorded
`hit_cap`; `retokenized` = `scripts/backfill_hit_cap.py`'s independent audit, see
"Non-termination is not only a thinking-arm problem" above) splits roughly
1:5 (`generator`:`retokenized`) for every condition **except** `filler`, which is
**100% `generator`** (1,440:0) -- every remaining `filler` row comes from the
2026-08-29 prompt-rewording re-run, which recorded `hit_cap` directly and so
never needed the retokenization backfill the other six conditions partly rely
on. Removing `zero_cot` removed exactly the older, pre-rewording `filler` rows
that depended on retokenization, so this is no longer a mixed-instrument
comparability concern -- `filler`'s non-termination numbers are now on one
instrument, not two.

## Retracted: "Most primer findings depend on the retired `zero_cot` style"

This project used to carry a finding here comparing `zero_shot`-only accuracy
significance against `zero_cot`-only, concluding that most of the pooled
main-sweep primer-helps/primer-hurts table traced back to the retired,
half-token-budget style rather than to the live prompt format. `zero_cot` and
its 5,040 rows have since been fully removed from the project (code, data, and
docs), so that comparison can no longer be reproduced and the finding is
retracted along with the data it depended on.

The underlying concern it raised -- that a pooled `zero_shot`+`zero_cot` table
could overstate what holds on live data -- is now moot by construction rather
than resolved by re-analysis: every accuracy table in this document is
`zero_shot`-only, because that is the only style the project generates or
scores. There is no pooled-vs-`zero_shot`-only distinction left to draw.

The one part of the retracted finding that was never about `zero_cot` at all
survives unchanged: the thinking arm's non-termination result ("Non-termination
responds to the primer" above) is a reliability effect measured entirely within
`zero_shot`, on both arms, and was never confounded with the retired style.
