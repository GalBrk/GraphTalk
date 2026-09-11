# Ladder screen and retrieval-probe results (2026-09-11)

Read `docs/ladder-and-rewiring.md` first — this is the design; this document is
the first results pass over it, run with two scripts:

- `scripts/analyze_ladder.py` → `analysis/ladder_matrix.csv` (215 rows), now
  covering all 12 model arms (it previously covered only 6 — `gemma4-12b`,
  `gemma4-e4b(-think)`, `qwen3-8b-think`, and `qwen3-14b` were missing).
- `scripts/analyze_retrieval.py` (new) → `analysis/retrieval_matrix.csv`
  (445 rows), scoring the `retrieval_locate` reading-limit probe. Nothing
  turned this into per-model reading limits before this pass; the only
  numbers that existed were `graphtalk/ladder.py`'s `READING_CLEAN`/
  `READING_DEGRADED`/`READING_MIDDLE_COLLAPSE` constants, measured by hand
  for `qwen3-1.7b` alone.

## Why this pair of files answers one question

`analyze_ladder.py` classifies every `(n, mean_degree)` rung a model was run
on into `ceiling`/`floor`/`informative`, on the two axes documented in
`ladder-and-rewiring.md`: `n` (length) and mean degree (answer magnitude).
"Informative" alone conflates two different failures with opposite
implications for a primer:

- **length-limited** — the model can't read the graph, so it can't use a
  primer either;
- **magnitude-limited** — the model reads everything fine and still gets it
  wrong, which is exactly the work a primer could short-circuit.

Telling these apart needs a per-model **reading limit**, which only the
retrieval probe can supply — that's what `analyze_retrieval.py` measures.
Once both exist, `ladder_matrix.csv`'s `valid` column (`informative AND
readable`) is the direct answer to "which graphs are hard but readable for
this model" — the only rungs where a primer test means something.

## Reading limits (`retrieval_locate`)

The probe tests `k` ∈ {160, 220, 260, 320, 420, 520, 640} distractor
statements (≈922–3,742 tokens, `chars // 4` heuristic — see caveat below),
at position 0.1/0.5/0.9 and magnitude small/large, no graph anywhere in the
prompt.

| model | reading limit located? | value | note |
|---|---|---|---|
| `qwen3-0.6b` | yes | **0** (never clean) | pooled accuracy never reaches 0.90, even at k=160 (~922 tok); lost-in-the-middle gap already >0.15 at the shortest prompt tested |
| `qwen3-0.6b-think` | yes | **0** (never clean) | same pattern |
| `qwen3-1.7b` | yes | **~1,509 tok** (k=260) | genuine crossing within tested range |
| `qwen3-1.7b-think` | yes | **~2,449 tok** (k=420) | genuine crossing within tested range |
| `gemma4-12b` | no — lower bound only | ≥3,742 tok | still ≥0.99 pooled accuracy at the longest tested prompt |
| `gemma4-e4b-think` | no — lower bound only | ≥3,742 tok | same |
| `qwen3-8b` | no — lower bound only | ≥3,742 tok | same |
| `qwen3-8b-think` | no — lower bound only | ≥3,742 tok | same |
| `qwen3-14b` | no — lower bound only | ≥3,742 tok | same |
| `qwen35-2b` | no — lower bound only | ≥3,742 tok | same |
| `qwen35-2b-think` | no — lower bound only | ≥3,742 tok | same |

**Only 4 of 11 models have a real, located reading limit.** For the other 7,
the probe's longest tested prompt (~3,742 tokens) is shorter than most of the
ladder's own rungs (which run to 15,136 tokens), so all we know is "reads
fine to at least 3,742 tokens" — not where it actually breaks. Passing that
untested ceiling into `analyze_ladder.py` as if it were a measured limit
would falsely mark every longer ladder rung as unreadable for those 7 models,
so the ladder run below only supplies `--reading-limits` for the 4 models
that have a real one; the other 7 are scored with readability unassessed
(treated as readable, i.e. "unknown," not "yes").

## Ladder: hard-but-readable rungs per model

```
PYTHONPATH=. python scripts/analyze_ladder.py --responses 'runs/*.ladder_screen.jsonl' \
    --reading-limits qwen3-0.6b=0 qwen3-0.6b-think=0 qwen3-1.7b=1509 qwen3-1.7b-think=2449 \
    --out analysis/ladder_matrix.csv
```

| model | valid (hard but readable) rungs | reading limit applied? |
|---|---|---|
| `gemma4-12b` | none — never leaves ceiling | — |
| `gemma4-e4b` | none — never leaves ceiling | — |
| `gemma4-e4b-think` | none — never leaves ceiling | — |
| `qwen35-2b-think` | none — never leaves ceiling | — |
| `qwen3-0.6b` | **none** — every informative rung is past its reading limit | yes (0) |
| `qwen3-1.7b` | **none** — every informative rung is past its reading limit | yes (1,509) |
| `qwen3-1.7b-think` | `n40k12` | yes (2,449) |
| `qwen35-2b` | `n40k12, n40k16, n60k12, n60k16, n80k12, n80k16` | no (unknown) |
| `qwen3-8b-think` | `n80k8, n60k12, n80k16, n120k16, n160k12` | no (unknown) |
| `qwen3-14b` | `n60k16, n80k16, n120k16, n200k12` | no (unknown) |
| `qwen3-8b` | `n60k16, n120k12, n120k16, n160k12, n200k12, n300k8` | no (unknown) |

Reading this table:

- **`qwen3-0.6b` and `qwen3-1.7b`'s difficulty on this ladder is entirely a
  reading problem, not a graph-reasoning one** — every rung where the plain
  task gets hard for them is a rung they've already been shown can't be read
  cleanly. A primer test at any of those rungs would measure reading, not
  primer content.
- **`gemma4-12b`, `gemma4-e4b(-think)`, and `qwen35-2b-think` never left
  ceiling anywhere on this 18-rung ladder** — no primer test is interpretable
  for them yet; the ladder would need to extend further to find their
  informative band.
- **The five rows with `reading limit applied? = no`** — `qwen35-2b`,
  `qwen3-8b(-think)`, `qwen3-14b` — have real informative rungs, but "valid"
  there currently just means "not shown to be unreadable," not "confirmed
  readable." Treat these as candidates, not settled — see "Not yet done"
  below.

## Isolating the magnitude effect (holding `n` fixed)

`ladder_screen` varies both axes at once, so a raw accuracy drop between two
rungs could be either axis. Holding `n` fixed and reading across `k̄` only
isolates the magnitude effect specifically — still only for `task=node_degree`,
the one task this ladder covers:

| model | where it drops off ceiling (holding `n` fixed) | confirmed magnitude, not reading? |
|---|---|---|
| `qwen3-1.7b-think` | `k̄=8→12` at `n=40`: 0.92 → 0.82 | **yes** — 2,290 tokens is under its measured 2,449-token reading limit, so this is a clean, isolated magnitude effect |
| `qwen3-14b` | `k̄=12→16`, consistently at `n=60/80/120`: ~0.92-0.96 → 0.74-0.84 | not yet — no confirmed reading limit for this model, so length can't be ruled out |
| `qwen3-8b` | `k̄=8→12` at `n=120`: 0.94 → 0.88 | same caveat |
| `qwen35-2b` | `k̄=8→12`, consistently at `n=40/60/80`: ~0.98 → 0.80-0.86 | same caveat |
| `qwen3-8b-think` | non-monotonic — informative/ceiling flip back and forth across `k̄=8/12/16` at the same `n` (e.g. `n=80`: 0.87→0.91→0.89) | no — looks like sampling noise at n=50/cell near the 0.90 boundary, not a clean cliff |
| `gemma4-12b`, `gemma4-e4b(-think)`, `qwen35-2b-think` | none found | stay at ~0.98-1.0 even at the highest tested `k̄=16` |
| `qwen3-0.6b`, `qwen3-1.7b` | very strong, monotonic: accuracy roughly halves for each `k̄` step up, at every `n` | **confounded, not confirmed** — every rung tested is already past these two models' reading limits (0 and ~1,509 tokens), so the collapse could be reading failure, magnitude failure, or both |

## Speculative: extrapolating accuracy past `k̄=16`

**This section is a guess, not a measurement** — nothing past `k̄=16` has been
run. It answers "if the trend measured so far continued, where would it go,"
which is useful for deciding where to extend the ladder next, not for citing
an accuracy number.

Method: for each model, an ordinary-least-squares fit of
`accuracy ~ b0 + b1*k̄ + b2*n` over its 16-18 usable ladder cells (`n_rows`
>= 20, to exclude the two unfinished jobs' near-empty cells), then evaluated
at `k̄` = 16 (in-range, as a sanity check against the real numbers above),
20, 24, 28, holding `n` at 80 (or 40 for `qwen3-0.6b(-think)`/`qwen3-1.7b(-think)`,
whose informative range sits at smaller `n`). Clipped to `[0, 1]`.

| model | slope / +1 degree | R² | acc. @ k̄=16 (real) | @ 20 (guess) | @ 24 (guess) | @ 28 (guess) |
|---|---|---|---|---|---|---|
| `qwen3-0.6b` | −0.057 | 0.89 | 0.23 | 0.00 | 0.00 | 0.00 |
| `qwen3-1.7b` | −0.051 | 0.93 | 0.31 | 0.11 | 0.00 | 0.00 |
| `qwen3-1.7b-think` | −0.045 | 0.84 | 0.50 | 0.32 | 0.14 | 0.00 |
| `qwen3-0.6b-think` | −0.040 | 0.88 | 0.60 | 0.44 | 0.28 | 0.12 |
| `qwen3-14b` | −0.017 | 0.58 | 0.85 | 0.78 | 0.71 | 0.65 |
| `qwen35-2b` | −0.012 | 0.54 | 0.85 | 0.81 | 0.76 | 0.71 |
| `qwen3-8b` | −0.006 | 0.67 | 0.92 | 0.90 | 0.87 | 0.85 |
| `qwen3-8b-think` | −0.006 | 0.25 | 0.90 | 0.88 | 0.86 | 0.83 |
| `qwen35-2b-think` | ~0.000 | 0.03 | 1.00 | 0.99 | 0.99 | 0.99 |
| `gemma4-12b`, `gemma4-e4b`, `gemma4-e4b-think` | ~0.000 | n/a (no variance — accuracy is 1.0 everywhere tested) | 1.00 | 1.00 | 1.00 | 1.00 |

Read with real caution, in decreasing order of how much:

- **The extrapolation is linear over a range where the real quantity is a
  bounded, almost certainly saturating accuracy curve.** A straight line is
  the crudest possible guess at that shape — it is why `qwen3-0.6b`/
  `qwen3-1.7b` are predicted to hit exactly 0 by `k̄≈20-24` rather than
  leveling off near their blind-guess floor (`maj_base`, not measured here).
  Treat every number in the last three columns as "roughly this direction,"
  not a real forecast.
- **This extrapolates 4-12 degree units past the last tested point (`k̄=16`)
  on a fit built from only 4 distinct `k̄` values.** That's a large
  extrapolation relative to the data actually collected.
- **The ceiling models' fit is degenerate, not informative.** `gemma4-12b`/
  `gemma4-e4b(-think)` have essentially zero accuracy variance in the tested
  range (R²=0 or undefined), so the flat-line prediction just restates "we
  have not yet found a magnitude that dents this model" — it says nothing
  about whether `k̄=28` or `k̄=80` would.
- **One real signal despite all that: a slope ranking.** The four smallest/
  weakest models (`qwen3-0.6b(-think)`, `qwen3-1.7b(-think)`) have slopes
  3-10x steeper than the mid-size models (`qwen3-14b`, `qwen35-2b`,
  `qwen3-8b(-think)`), which in turn are steeper than the three models with
  no measurable slope at all. That ordering is a genuine finding even though
  the absolute extrapolated values aren't — bigger/stronger models don't just
  have a higher ceiling, they degrade *more slowly per unit of added
  magnitude* once they do start to degrade.
- **A secondary, more speculative observation: thinking softens the slope for
  the two smallest models but not for the 8B one.** `qwen3-0.6b`
  (−0.057) → `qwen3-0.6b-think` (−0.040) and `qwen3-1.7b` (−0.051) →
  `qwen3-1.7b-think` (−0.045) both flatten with thinking on; `qwen3-8b`
  (−0.006) and `qwen3-8b-think` (−0.006) are identical. Consistent with
  reasoning tokens helping most where the base model is weakest at the
  underlying counting task, but this is one data point per model pair, not
  a tested hypothesis.

If a next ladder is built to chase this, the slope ranking says where a
useful next rung is likely to sit: `qwen3-14b`/`qwen35-2b`/`qwen3-8b(-think)`
would need `k̄` well past 28 (extrapolated ~0.65-0.85 there) to reach floor,
while `gemma4-12b`/`gemma4-e4b(-think)` have shown no slope at all yet and a
next rung for them is a bigger jump into the unknown rather than a
continuation of a trend.

## Token cost vs. density, and where the clean window closes

Mean degree can't be raised independently of prompt length: `k̄ = 2·edges/n`,
so reaching a target `k̄` at a *lower* density requires a *larger* `n`, which
is exactly what drives the token count up (`tokens ≈ -665 + 19.5·n +
8.16·edges`, fit on the 18 real ladder rungs, R²=0.998). Two constraints are
in tension for any target `k̄`:

- **Fit under a model's context window** — 32,768 tokens for the Qwen3/
  Qwen3.5 arms (`graphtalk/models.py`'s `max_context_tokens`) — which wants
  *higher* density (smaller `n` for the same `k̄`).
- **Stay below the density-collapse boundary** (~0.50, from
  `docs/difficulty-scaling.md`/`primer-effects-and-power.md` — above this,
  models start guessing "complete graph" and the task stops measuring
  counting difficulty at all) — which wants *lower* density.

The full curve, for five very different `k̄` targets:

**k̄=10**

| density | n | edges | tokens |
|---|---|---|---|
| 0.05 | 201 | 1,005 | 11,459 |
| 0.10 | 101 | 505 | 5,427 |
| 0.20 | 51 | 255 | 2,411 |
| 0.30 | 34 | 172 | 1,406 |
| 0.41 | 25 | 127 | 867 |
| **0.50** | 21 | 105 | 602 |
| 0.65 | 16 | 82 | 324 |
| 0.95 | 12 | 58 | 31 |

Crosses 32,768 tokens at density ~0.018 — every density from there up to 1.0
fits comfortably. **No tension at all**: the whole 0.05-0.50 clean range is
available and cheap (602-11,459 tokens).

**k̄=20**

| density | n | edges | tokens |
|---|---|---|---|
| 0.05 | 401 | 4,010 | 39,892 (over cap) |
| 0.10 | 201 | 2,010 | 19,664 |
| 0.20 | 101 | 1,010 | 9,550 |
| 0.30 | 68 | 677 | 6,179 |
| 0.41 | 50 | 498 | 4,370 |
| **0.50** | 41 | 410 | 3,482 |

Crosses 32,768 tokens at density ~0.061 — still a wide clean window
(0.061-0.50), just no longer the *whole* range.

**k̄=40**

| density | n | edges | tokens |
|---|---|---|---|
| 0.05 | 801 | 16,020 | 145,745 (over cap) |
| 0.20 | 201 | 4,020 | 36,075 (over cap) |
| 0.25 | 161 | 3,220 | 28,763 |
| 0.30 | 134 | 2,687 | 23,889 |
| 0.41 | 99 | 1,971 | 17,351 |
| **0.50** | 81 | 1,620 | 14,141 |

Crosses 32,768 tokens at density ~0.220. The clean window has shrunk to
0.220-0.50 — a third of what `k̄=20` had.

**k̄=80**

| density | n | edges | tokens |
|---|---|---|---|
| 0.30 | 268 | 10,707 | 91,967 (over cap) |
| 0.41 | 196 | 7,845 | 67,208 (over cap) |
| **0.50** | 161 | 6,440 | 55,053 (over cap) |
| 0.65 | 124 | 4,963 | 42,275 (over cap, and already past collapse) |
| 0.80 | 101 | 4,040 | 34,289 (over cap, and already past collapse) |
| 0.95 | 85 | 3,408 | 28,824 (fits, but density-collapse) |

Crosses 32,768 tokens at density ~0.837 — **past** the 0.50 collapse
boundary. **The window has closed**: there is no density at which this
prompt both fits in a Qwen-family context window and stays outside the
collapse regime.

**k̄=160**

| density | n | edges | tokens |
|---|---|---|---|
| 0.50 | 321 | 25,680 | 215,255 (over cap) |
| 0.80 | 201 | 16,080 | 134,537 (over cap) |
| 0.95 | 169 | 13,554 | 113,296 (over cap) |

Crosses 32,768 tokens at density ~0.990 — a graph that dense is essentially
complete. **No usable window at any density.**

**Where it bites: the window closes between `k̄=40` and `k̄=80`.** Solving
for the exact crossing (density-at-cap = 0.50) gives **`k̄≈61`**: below that,
some density keeps a prompt both under 32,768 tokens and below the collapse
boundary; above it, no density does either for the Qwen family. That number
is itself only as good as the linear token fit and the 0.50 collapse
threshold it's built on (both real, measured quantities, but the fit is
being evaluated well outside the `n`/`k̄` region it was fit on for the larger
targets), so read `k̄≈61` as "roughly where this stops being possible," not
an exact boundary.

### Connecting this to what's actually configured per model

- **Qwen3 / Qwen3.5 arms** (`qwen3-0.6b(-think)`, `qwen3-1.7b(-think)`,
  `qwen3-8b(-think)`, `qwen3-14b(-think)`, `qwen35-2b(-think)`): all specced
  at `max_context_tokens=32,768` in `graphtalk/models.py`. Every number above
  applies to them directly, and `hf_backend.py`'s overflow guard would refuse
  to generate rather than silently truncating.
- **Gemma 4 arms** (`gemma4-e4b(-think)`, `gemma4-12b(-think)`): **no
  restriction has been found or measured for these in this project.**
  `graphtalk/models.py` leaves `max_context_tokens=None` for all four —
  confirmed by reading the file, not inferred — so `hf_backend.py`'s overflow
  guard is a documented no-op for them (`docs/difficulty-scaling.md` flags
  this as a pre-existing gap, not something this session's data changes). I
  also don't have verified knowledge of Gemma 4's actual published context
  window to substitute here — it isn't something I'm confident predates my
  training data — so the honest state is: unknown until someone fills in
  `max_context_tokens` from the real model card (google/gemma-4-*-it) or
  measures it directly. Until then, a large prompt sent to a Gemma arm is
  genuinely untested territory, not merely "unchecked but probably fine."

## Recommended `k̄` per model, for a corpus design

The goal this section answers: pick a `(n, k̄)` cell per model where `none`
is informative (not ceiling, not floor) with enough margin that `filler` —
the length-matched, content-free placebo — is expected to stay informative
too rather than being dragged to floor by the length cost alone
(`docs/primer-effects-and-power.md`'s `filler` finding: −11.7pp on dense
graphs from length alone). **There is no single `k̄` that works for every
model** — that's the whole reason the ladder has 18 rungs instead of one
cell; the table below is per model, not a project-wide default.

| model | recommended rung | `none` accuracy | tokens | confidence |
|---|---|---|---|---|
| `qwen3-1.7b-think` | `n40k12` | 0.82 | 2,290 | only valid rung on this ladder — no alternative to compare against |
| `qwen3-14b` | `n60k16` (cheapest) or `n120k16` (more floor margin) | 0.74 / 0.84 | 4,407 / 9,344 | reading limit unconfirmed past 3,742 tok (see "Reading limits" above) |
| `qwen3-8b` | `n60k16` | 0.88 | 4,407 | same caveat |
| `qwen3-8b-think` | `n60k12` | 0.85 | 3,489 | same caveat, plus this model's ladder file is incomplete at the high end (see "Data completeness caveat" below) |
| `qwen35-2b` | `n60k12` | 0.80 | 3,489 | reading limit unconfirmed |
| `qwen3-0.6b(-think)`, `qwen3-1.7b` | **none usable** | — | — | every informative rung is already past their measured reading limit — a primer test there measures reading, not primer content |
| `gemma4-12b`, `gemma4-e4b(-think)`, `qwen35-2b-think` | **none exists yet** | — | — | never leave ceiling anywhere on this 18-rung ladder |

The cheaper/mid-band picks (over the highest-`k̄` valid rungs) are
deliberate: `none` sitting at 0.74-0.88 leaves room in both directions — if a
real primer helps, there's headroom below ceiling to show it; if `filler`
hurts, there's headroom above floor before the cell stops being
interpretable. That reasoning about `filler` is an inference from the
general finding above, though, not a measurement at these specific cells —
which is exactly the missing piece below.

### What's missing to actually answer this

1. **`filler` has never been run on the ladder at all.** `ladder_screen` is
   `condition=none` only (see `runs/README.md`). The only place `filler` has
   been measured is the single shared rung `n40k12` (`rewire_shared`), and
   only for `qwen35-2b` and `qwen3-1.7b-think` there. Whether `filler` stays
   informative at any rung in the table above is extrapolated from that one
   point plus the general length-cost finding, not measured at those cells.
   **This is the single highest-value gap to close** before committing to a
   corpus design around this goal.
2. **Reading limits are unconfirmed above ~3,742 tokens** for `qwen3-8b(-think)`,
   `qwen3-14b`, `qwen35-2b` (see "Reading limits" above) — their recommended
   rungs rest on "not yet shown unreadable," not "confirmed readable."
3. **No rung past `k̄=16` has ever been run**, so there is no data-backed
   option for the three ceiling models. "Token cost vs. density" above shows
   a clean, in-context rung stops being *possible at all* past `k̄≈61` for
   the Qwen family, but nothing between 16 and 61 has been tested, so even
   that range is unverified, not a ready menu.
4. **Generation-budget headroom hasn't been checked with a primer added on
   top.** `max_new_tokens` (2048 plain / 8192-16384 thinking, from
   `graphtalk/models.py`) was sized against the primer sweep's shorter
   prompts. `graphtalk.ladder.context_headroom` exists to check
   `max_context − max_new − prompt_tokens ≥ 0`, but nobody has run it for
   the candidate rungs above with a primer's extra length included — a real
   risk for the `-think` arms specifically, which need the most room to
   answer without truncating (see "enough place for thinking and
   answering" in the corpus-design goal this section is answering).
5. **Two of the candidate files are from incomplete generation jobs**
   (`qwen3-8b-think`, `qwen35-2b-think`, see "Data completeness caveat"
   below) — treat their rungs above as provisional until those finish.
6. **Gemma's context cap is unmeasured** (`max_context_tokens=None` in
   `graphtalk/models.py`) — we can't bound how far past `k̄=16` a next rung
   could safely go for the three ceiling models, or whether their true
   context window would even accommodate one.

## Data completeness caveat

Two of the newer ladder files are from apparently-unfinished generation jobs,
not real zero-accuracy results:

- `runs/qwen3-8b-think.ladder_screen.jsonl`: 804/900 rows — `n300k8`
  (the longest rung) has **0** rows, and `n200k12` has only 4/50.
- `runs/qwen35-2b-think.ladder_screen.jsonl`: 868/900 rows — `n300k8` has
  only 18/50.

Both models' longest-rung numbers in `ladder_matrix.csv` should be treated as
provisional until those jobs finish and the file is regenerated.

## Token-count caveat

`approx_tokens` in `retrieval_matrix.csv` is `chars // 4` on a reconstructed
prompt (`scripts/build_retrieval_probe.build`), the same heuristic
`scripts/build_prompts.py` already uses for its own overflow warning
(`_APPROX_CHARS_PER_TOKEN = 4`, documented in `docs/difficulty-scaling.md`).
It is **not** a real per-model tokenizer count — none of the models' actual
tokenizers are available without `transformers`/GPU in this environment.
`graphtalk/ladder.py`'s `RUNGS` token counts, by contrast, *are* real
(qwen3's tokenizer, noted in that module's own comment), so a reading limit
from this probe and a rung's token count aren't perfectly comparable units —
close enough for a bracket, not for a precise crossing.

This is a distinct question from each `ModelSpec.max_context_tokens` in
`graphtalk/models.py`, which is the checkpoint's *published* context window
(input+output combined), used only as a hard overflow guard before
generation. It's already filled in for the Qwen3/Qwen3.5 arms (32,768) but
still `None` for `gemma4-e4b`, `gemma4-12b`, `qwen3-8b`, `qwen3-14b` (and
their `-think` variants) — a pre-existing gap noted in
`docs/difficulty-scaling.md`, not something this session's data changes.
Filling those in from the model cards would close that gap, but it answers
"how much can this model take in before it errors," not "where does its
retrieval accuracy actually degrade" — the nominal window is far larger than
every measured reading limit above (e.g. `qwen3-1.7b`'s 32,768-token nominal
window vs. its measured ~1,509-token degradation point).

## Not yet done

- **Extend `build_retrieval_probe.py --statements` past 640** (e.g. toward
  4,000–8,000+ to overlap the ladder's longer rungs) and re-run for the 7
  models that only have a lower bound today. Without this, most of
  `ladder_matrix.csv`'s `valid` rungs for those models rest on "not shown
  unreadable," not "confirmed readable."
- **Regenerate `ladder_matrix.csv`** once `qwen3-8b-think.ladder_screen.jsonl`
  and `qwen35-2b-think.ladder_screen.jsonl` finish generating their missing
  `n300k8`/`n200k12` rows.
- **No primer has actually been run at any valid rung yet.** This document
  only locates *where* a primer test would be interpretable; running one
  (`--condition` other than `none`) at, e.g., `qwen3-8b`'s `n60k16` is the
  next step once the two points above are settled — see
  `docs/ladder-and-rewiring.md`'s rewiring stage for the mechanism that keeps
  such a test free of the length confound.
