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

| model | style | `none` | `degree` | `all` | `filler` |
|---|---|---|---|---|---|
| gemma4-e4b | zero_shot | 92.2% | 93.9% | 95.0% | 79.4% |
| gemma4-e4b | zero_cot | 78.3% | 76.1% | 80.0% | 68.9% |
| gemma4-12b | zero_shot | **98.3%** | **100.0%** | 99.4% | 93.9% |
| gemma4-12b | zero_cot | 95.6% | 93.9% | 91.7% | 91.1% |
| qwen3-8b | zero_shot | 82.2% | 88.9% | 90.0% | 81.7% |
| qwen3-8b | zero_cot | 64.4% | 76.7% | 78.9% | 63.9% |
| qwen3-14b | zero_shot | 80.0% | 89.4% | 89.4% | 74.4% |
| qwen3-14b | zero_cot | 71.7% | 78.3% | 75.0% | 69.4% |

Three patterns hold across every model, which is what makes them worth stating
before any significance testing:

- **Primers help, modestly.** `degree` and `all` beat `none` in seven of eight
  model-style combinations. The largest gain is qwen3-8b under `zero_cot`, 64.4%
  to 78.9%.
- **`filler` hurts.** An uninformative primer of the same shape scores *below*
  the no-primer control everywhere, by up to 13 points (gemma4-e4b zero_shot,
  92.2% to 79.4%). Whatever the informative primers do, part of it has to be
  netted against a penalty for primer presence as such. This is the control
  earning its place.
- **`zero_cot` is worse than `zero_shot`.** Everywhere, for every model, by 6 to
  18 points. Being told to think step by step makes these models *less* accurate
  on these tasks.

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

| `all` | `degree` | `components` | `rwse` | `none` | `clustering` | `filler` |
|---|---|---|---|---|---|---|
| **5.4%** | 6.0% | 6.2% | 6.4% | 7.4% | 7.8% | **9.4%** |

Informative primers sit below the no-primer control; the length-matched
uninformative `filler` sits above it. On `gemma4-12b`, where the rates are large
enough to see clearly, `all` is 15.6% against `filler` at 30.0%.

The endpoints separate: `filler` vs `all` is p=0.003 pooled and p=0.001 on
`gemma4-12b` alone. The individual steps against `none` do not -- `filler` vs
`none` is p=0.18, `none` vs `all` p=0.10 -- so the monotone ordering across all
seven conditions is suggestive rather than established, and should be reported
that way.

This converges with the main sweep's `filler` result, where a length-matched
uninformative primer scored *below* the no-primer control on accuracy. Two
independent measures now agree that a primer adding material without adding
information actively harms the model, and the mechanism is legible: given the
degree sequence a model can shortcut the pairwise enumeration and commit; given
padding of the same shape it has more to verify and nothing to verify it with.

**It also gives `gemma4-12b` somewhere to move.** That model's accuracy is pinned
at 97.5-99.1% with almost no headroom, which is what made its McNemar cells
useless. Non-termination is an outcome variable that responds to the manipulation
on precisely the model whose accuracy cannot.

## Provenance

The thinking arm was generated with `graphtalk-cu126` (torch 2.13.0+cu126) while
the main sweep used the cu130 build; same torch version, same transformers,
greedy decoding throughout. The regeneration evidence is kept in
`runs/*.redo.shard*.jsonl` -- 67 rows re-run at 32,768 tokens, retained because
they are the evidence that the cap was never the cause, not because they are
usable answers.
