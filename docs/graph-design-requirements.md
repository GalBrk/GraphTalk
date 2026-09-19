# What a graph needs to be a valid primer test

Four requirements. A graph corpus that misses any one of them measures
something other than primer content, usually without failing loudly.

Everything here is measured, not assumed; the source job or script is named for
each number. Screens come from `graphtalk/cell_screen.py`, accuracies from
`analysis/ladder_matrix.limited.csv`, reading limits from the retrieval probe
(jobs 873209-873246).

## 1. `n >= 40` -- enough distinct degree values

At n=20 a blind guesser that always answers the modal degree scores
`maj_base` 0.287-0.325. Twenty nodes do not produce enough distinct degree
values for the task to have an answer worth reasoning about, at any density.
Nothing else you do to a 20-node graph rescues it.

Bar: `cell_screen.MAJ_BASE_MAX = 0.25`.

## 2. `k_bar >= 12` -- mean degree is the difficulty knob, not node count

The two ways a graph can be hard have opposite implications, and only one of
them leaves room for a primer:

- **Length-limited** (low `k_bar`, long prompt). The model cannot read the
  graph, and cannot read a primer either. At p=0.85 the `degree` control --
  which writes the answer *verbatim* into the prompt -- was worth **+0.7 pp**
  (job 871263). A primer cannot repair a reading failure.
- **Magnitude-limited** (high `k_bar`). The model reads everything and
  miscounts. The same control was worth **+6.8 pp** at p=0.50. This is the
  only regime where a primer has anything to do.

Confirmed without being fitted to: `qwen35-2b`'s valid rungs came out as
exactly `k_bar in {12, 16}`, and it ceilinged on every `k_bar in {4, 8}` rung.
The split landed exactly where the mechanism predicts.

Mean degree, edge count and prompt length are easy to confuse here -- they
move together in most corpora. They are separated in "Which is the driver"
below; the short version is that mean degree and length drive *different*
failure modes and edge count drives neither.

**Generate with `nx.gnm_random_graph(n, m)`, `m = round(k_bar * n / 2)` -- not
`gnp`.** Prompt length is a function of `(n, m)` alone, so letting `m` drift
between graphs leaks length variation into the corpus, and length alone is
worth 6-12 pp here (the `filler` control).

## 3. Tokens below the model's *reading limit* -- not the context window

These are different limits and the binding one is invisible:

| model | reading limit | context window |
| --- | --- | --- |
| `qwen3-0.6b` | < 1,505 | 32,768 |
| `qwen3-1.7b` | ~2,505 | 32,768 |
| `qwen35-2b` | none found to 6,305 | 32,768 |
| `qwen3-8b` | none found to 6,305 | 32,768 |

The reading limit binds **5-13x earlier** than the window. A prompt that fits
comfortably can still be unreadable, and the failure is *positional*, not a
matter of total length: `qwen3-1.7b` holds 0.96 on a fact at position 0.1 in
the prompt while collapsing to **0.18** on the same fact at position 0.5.

So "does not truncate" is the wrong target. The target is "below this model's
reading limit", and that number comes only from the retrieval probe
(`scripts/build_retrieval_probe.py`), per model.

Note the think arm reserves 16,384 tokens for generation, so its usable prompt
budget is ~16k even though the window is 32,768. That has never been the
binding constraint for any model measured so far.

## 4. `clu_sd >= 0.10` after rewiring -- and plain ER cannot deliver it

This is the requirement that surprises, and it is in **direct tension** with
requirement 2. Raising `k_bar` makes the task harder and simultaneously
flattens every node's clustering coefficient toward the graph mean. Every rung
that satisfies requirement 2 fails the clustering bar as a plain ER graph:

| n | k_bar | m | tokens | maj_base | clu_sd raw | clu_sd rewired |
| --- | --- | --- | --- | --- | --- | --- |
| 40 | 12 | 240 | 2,290 | 0.188 | **0.064** X | 0.136 OK |
| 60 | 12 | 360 | 3,489 | 0.217 | **0.059** X | 0.161 OK |
| 40 | 16 | 320 | 2,893 | 0.162 | **0.042** X | 0.119 OK |
| 80 | 12 | 480 | 4,688 | 0.153 | **0.051** X | 0.179 OK |
| 80 | 16 | 640 | 5,931 | 0.153 | **0.036** X | 0.144 OK |
| 120 | 12 | 720 | 7,373 | 0.142 | **0.042** X | 0.181 OK |
| 120 | 16 | 960 | 9,344 | 0.129 | **0.034** X | 0.150 OK |

(`graphtalk/cell_screen.py::screen_cell`, 4 graphs per cell, seed 1000. The
bar is on the **rendered** 2-dp value, since that is what the model sees.)

**ER alone is therefore not a sufficient generator.** The corpus is ER *plus a
degree-preserving rewiring pass* (`graphtalk/rewiring.py`), which roughly
triples `clu_sd` while holding `n`, `m`, the degree sequence, every gold
answer, and the prompt length exactly fixed. Rewiring is not a nicety here; it
is what makes the useful densities testable at all.

## The shared configuration

`n = 40`, `k_bar = 12`, `m = 240` -> 2,290 tokens, `maj_base` 0.188, rewired
`clu_sd` 0.136.

It is the only configuration clearing all four gates for every model that has
a valid rung at all: under `qwen3-1.7b`'s 2,505-token reading limit, hard
enough that 1.7b sits at 0.480 and `qwen35-2b` at 0.860, and structurally
informative once rewired.

## Per-model status

| model | valid rungs | note |
| --- | --- | --- |
| `qwen3-0.6b` | **none** | cannot read even the smallest rung |
| `qwen3-0.6b-think` | **none** | same |
| `qwen3-1.7b` | n40k8, n80k4, **n40k12** | |
| `qwen3-1.7b-think` | **n40k12** | reasoning removes most of the headroom |
| `qwen35-2b` | n40k12, n40k16, n60k12, n60k16, n80k12, n80k16 | widest window |
| `qwen35-2b-think` | **none** | ceilings on all 18 rungs |
| `qwen3-8b` | screen pending | probe done: no limit to 6,305 |
| `gemma4-e4b` | screen pending | probe running |
| `gemma4-12b` | screen pending | probe running |
| `qwen3-14b` | screen pending | probe running |

"No valid rung" is a result about that model, not a gap in the corpus, and the
plan is structured so it is reportable as such.

## The large models, and a gap in the probe

`qwen3-8b`'s probe is complete and is the most encouraging result so far: flat
**0.967-1.000 across the entire probe range**, with no positional collapse at
all. Where 1.7b loses the middle of its prompt, 8b does not. If that holds for
the 12B/14B class, the large models are free to use the high rungs, and their
difficulty will be magnitude-limited -- the regime where primers can act.

But that is exactly where the current measurement runs out:

- **The probe tops out at 6,305 tokens; the ladder goes to 15,136.** For
  `qwen3-1.7b` this did not matter, because its limit (~2,505) is far inside
  the probed range. For a model with no limit below 6,305 it matters entirely:
  we cannot yet distinguish "no reading limit" from "no reading limit *that we
  looked for*".
- The large models may need the high rungs -- but **not for the reason the
  size sweep appears to say.** See "n is a proxy" below.

**Recommended follow-up, once the running probes land:** extend
`build_retrieval_probe.py` with `--statements 900 1200 1500`, which covers
roughly 9k / 12k / 15k tokens at ~10 tokens per statement (measured:
k=160 -> 1,505 and k=640 -> 6,305). Run it on the 8B/e4b/12B/14B arms only.
Without it, any high-rung result for a large model cannot be attributed to
reasoning rather than reading -- which is the single distinction this whole
design exists to make.

Do not submit that extension while the current probes for those same models
are still in flight; they write to the same run files.

## "n is a proxy": why the size sweep's n=80 is not a node-count finding

The size sweep reports `qwen3-8b` at 1.000 (size20), 0.940 (size40), 0.479
(size80), which invites the reading "8B needs 80-node graphs". It does not.

`build_size_sweep.py` draws sparsity ~ U(0, 1) *per graph*, so `n` and mean
degree are not separable in that corpus. Measured on the actual prompt files:

| corpus | n | median m | median k_bar | median chars |
| --- | --- | --- | --- | --- |
| size20 | 20 | 62 | 6.2 | 2,054 |
| size40 | 40 | 143 | 7.2 | 4,692 |
| size80 | 80 | 524 | **13.1** | 12,677 |

Going size40 -> size80 raises `k_bar` from 7.2 to 13.1 -- **it crosses the
`k_bar >= 12` threshold of requirement 2**, independently derived from the
ladder on a different corpus. The two easy cells are both in the
length-limited band where 8B ceilings; the hard one is the first to reach the
magnitude-limited band.

So the driver is mean degree, and `n=80` is where that particular corpus's
random density draw happened to deliver it. Two independent lines of evidence
now point at `k_bar ~ 12` as the boundary.

**This is worth stating as a prediction, because it is falsifiable and the
answer is already being computed.** If `k_bar` is the driver, `qwen3-8b`
should be informative at **n40k12 -- 40 nodes, 2,290 tokens** -- roughly a
fifth the prompt length of size80, and comfortably inside any plausible
reading limit. If `n` were the driver it should ceiling there instead. The
running `qwen3-8b` ladder screen (job 873237) decides it.

If the prediction holds, the large models do **not** need the expensive high
rungs, the probe gap above matters much less than it appears to, and the
shared n40k12 configuration extends to the whole suite.

## Which is the driver: mean degree, edge count, or prompt length?

These three are easy to confuse because they move together in most corpora --
`m = k_bar * n / 2`, and prompt length is a function of `(n, m)`. Earlier work
in this project attributed difficulty to edge count, and then to prompt
length. Both were reading the same effect through a proxy.

The ladder settles it, because it varies `n` and `k_bar` *independently* and
therefore contains rungs with identical edge counts but different mean degree.

### Test 1 -- hold edges fixed, vary mean degree

If `m` or length were the driver, the rows in each block would be equal.

| m=240 | tokens | 0.6b | 1.7b | 2b |
| --- | --- | --- | --- | --- |
| k_bar=4 (n=120) | 3,430 | 0.660 | 0.760 | 0.980 |
| k_bar=8 (n=60) | 2,569 | 0.760 | 0.720 | 0.980 |
| k_bar=12 (n=40) | 2,290 | **0.500** | **0.480** | **0.860** |

The higher-`k_bar` rung is harder *and shorter*: length runs opposite to
difficulty. This rules out length-alone, and rules out edge count (constant by
construction). The pattern holds in 6 of the 7 fixed-`m` blocks, across all
three models. Full table: `scripts/analyze_ladder.py` output grouped by `m`.

### Test 2 -- hold mean degree fixed, vary length

Accuracy still falls: `qwen3-1.7b` at `k_bar=8` goes 0.840 -> 0.408 as tokens
go 1,691 -> 15,136. So length is a real and independent driver too.

### Test 3 -- fit both

`accuracy = a + b * k_bar + c * log2(tokens)`, least squares over all 18 rungs:

| model | per +1 `k_bar` | per 2x length | probe reading limit |
| --- | --- | --- | --- |
| `qwen3-0.6b` | -0.037 | **-0.183** | < 1,505 |
| `qwen3-1.7b` | -0.041 | -0.095 | ~2,505 |
| `qwen35-2b` | -0.014 | **+0.018** | none to 6,305 |

**The length coefficient tracks the independently measured reading limit.**
`qwen35-2b`, the model whose probe found no reading limit, has a length
coefficient of essentially zero -- length genuinely does not hurt it.
`qwen3-0.6b`, with the lowest limit, pays the largest penalty. Two unrelated
measurements agreeing is the strongest evidence in this document.

### The answer

- **Mean degree drives magnitude-limited difficulty** -- the model reads the
  graph correctly and miscounts. Universal across models. On `node_degree`,
  `k_bar` *is* the mean answer magnitude.
- **Prompt length drives reading-limited difficulty**, but only past that
  model's reading limit. Model-specific, and ~0 for a model that has not
  reached its own.
- **Edge count is not a driver in its own right.** It is not separately
  identifiable from `k_bar` and `n`; hold it fixed and difficulty still tracks
  `k_bar`. The earlier edge-count attribution was a degree effect seen through
  an `m`-shaped proxy -- the same error as the `n`-shaped proxy above.

This is why requirements 2 and 3 are separate gates rather than one: they
constrain different failure modes, and a cell can fail either independently.

### The honest limit, and the test that removes it

At fixed `m`, `k_bar` and `n` are perfectly anti-correlated, so Test 1 alone
says "`k_bar` up **or** `n` down". Test 3 breaks that (all 18 rungs, `n` and
`k_bar` independent), and the direction argues for `k_bar` regardless: the
*smaller* graph being harder contradicts any "more material is harder" story.

The zero-confound test already exists and costs nothing. **Query a low-degree
vs a high-degree node on a byte-identical prompt** -- same graph, same length,
same edges, only the answer's magnitude differs. Job 871262 measured 0.624 vs
0.225 at matched edge counts. It is pre-registered as a stratifier over the
Stage 3 data rather than a separate run, so it settles the question with
nothing left to confound and no extra generation.

## Verification

- Screens: `PYTHONPATH=. python -c` over `cell_screen.screen_cell(n, k)`,
  >= 4 graphs. Single-seed screening flipped two cells during development and
  `screen_cell` now refuses `n_graphs < 2`.
- Rewiring invariants are asserted by `tests/test_rewiring.py`: node set, edge
  count, degree sequence, no self-loops, and prompt length unchanged.
- Gold answers carry a trailing period (`'13.'`). Normalise before comparing
  or four models will appear to score 0.000.
