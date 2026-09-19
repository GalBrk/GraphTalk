# Building graphs that leave room to measure primer impact — current status

This is the one doc to read for "what graph should I generate, for which model,
to get a fair primer test." It consolidates and supersedes three docs that are
being deleted alongside it (see "Superseded" at the end) because they reached a
conclusion later work overturned, or because their content is now duplicated
by a doc that's easier to trust.

Goal, verbatim from the plan that started this line of work
(`docs/plans/finding-graphs-that-make-primer-effects-measurable.md`): *"A
clean, well-powered, shortcut-free demonstration of whether a primer helps
[a model] answer GraphQA questions... Scope: only the graphs change."*

## The mechanism, corrected

Two independent things make a graph hard, and only one of them leaves room for
a primer to help:

- **Mean degree (`k_bar`) drives magnitude-limited difficulty.** The model
  reads the graph correctly and miscounts. This is the regime a primer can
  act in — job 871263 measured the `degree` control (states the answer
  verbatim) worth **+6.8pp** at p=0.50.
- **Prompt length, past a model-specific *reading limit*, drives
  length-limited difficulty.** The model cannot reliably use even a fact
  stated outright. No primer can repair this — the same `degree` control was
  worth only **+0.7pp** once the plain arm had collapsed on length. Job
  871909 measured this directly, with **no graph in the design at all**: a
  plain list of `"Node X has degree Y."` statements retrieves perfectly to
  ~1,500 tokens, then collapses — at 6,305 tokens the model finds a fact at
  the *middle* of the prompt only 19.5% of the time, against ~89% at the
  ends.
- **Edge count / density is not an independent driver.** Job 871262 held mean
  degree fixed (at 8 and at 16) while varying `n` 20→160, so density fell 8x
  and edges + prompt length grew 8x with the answer distribution unchanged.
  Accuracy fell *monotonically in both blocks* (0.912→0.624 at k=8,
  0.575→0.224 at k=16) — the opposite of what "density drives difficulty"
  predicts. Every "denser is harder" reading anywhere in this project's
  history was density standing in for mean degree or edge count seen through
  a correlated proxy.
- **Node count alone is also not an independent driver — it's usually a
  `k_bar` proxy.** `scripts/build_size_sweep.py` draws density ~U(0,1) per
  graph, so its n=80 class has median `k_bar`=13.1 against n=40's 7.2 — it
  crosses the `k_bar>=12` threshold above, which is why `qwen3-8b` looked
  like it "needed n=80" in the old size sweep. The ladder (below) confirmed
  this directly: `qwen3-8b` is still at ceiling on `n40k12` (2,290 tokens)
  and only turns informative once `k_bar` reaches 16, regardless of `n`.

Full derivation, per-model gates, and the ER-plane/rewiring trade-off:
**`docs/graph-design-requirements.md`** (four requirements: `n>=40`,
`k_bar>=12`, tokens under the model's reading limit, `clu_sd>=0.10` after
degree-preserving rewiring). That doc is current and this one does not repeat
its derivation — only its bottom-line status table, updated below with data
that landed after it was written.

## What actually builds the graphs

One shared 18-rung corpus (`graphtalk/ladder.py::RUNGS`), every model runs all
of it, each model's primer test is scored only on its own valid rungs. Two
axes — `n` (length) and `k_bar` (magnitude) — because they're the two
independent drivers above, and plain Erdős–Rényi cannot deliver both a hard
`node_degree` task *and* a legible `clustering` primer at once: raising
`k_bar` flattens every node's clustering toward the graph mean
(`docs/graph-design-requirements.md` §4). The fix is
**degree-preserving rewiring** (`graphtalk/rewiring.py`) — repeated
double-edge swaps that hold `n`, `m`, the degree sequence, every gold answer,
and prompt length *exactly* fixed, moving only triangle structure. Design
notes: `docs/ladder-and-rewiring.md`.

## Per-model status (current, 2026-09-10)

`qwen3-0.6b`, `qwen3-1.7b`, `qwen35-2b` and their `-think` arms are from
`docs/graph-design-requirements.md`, unchanged. `qwen3-8b`, `gemma4-e4b`,
`gemma4-12b` and their `-think` arms are freshly scored for this doc from
`runs/*.ladder_screen.jsonl`, which had already been generated but never
scored — this closes that gap.

| model | valid rungs (primer-testable) | note |
| --- | --- | --- |
| `qwen3-0.6b` | **none** | reading limit < 1,505 tokens; below the ladder's smallest rung |
| `qwen3-0.6b-think` | **none** | same |
| `qwen3-1.7b` | n40k8, n80k4, n40k12 | reading limit ~2,505 |
| `qwen3-1.7b-think` | n40k12 only | reasoning removes most of the headroom |
| `qwen35-2b` | n40k12, n40k16, n60k12, n60k16, n80k12, n80k16 | widest window of any model with a confirmed reading limit; no limit found to 6,305 |
| `qwen35-2b-think` | **none** | ceilings on all 18 rungs |
| `qwen3-8b` | **n60k16 confirmed** (4,407 tok, 0.880); n120k12/n120k16/n160k12/n200k12/n300k8 informative but **readability unconfirmed** (7,373–15,136 tok, past the 6,305-token probed range) | see "The qwen3-8b prediction," below |
| `qwen3-8b-think` | screen 63/900 rows — **incomplete**, not yet decidable | |
| `gemma4-e4b` | screen 445/900 rows, **ceiling on every rung scored so far** (through n60k16) | no reading-limit probe run at all yet |
| `gemma4-e4b-think` | screen 278/900 rows, ceiling on every rung scored so far (through n120k4) | no reading-limit probe |
| `gemma4-12b` | screen 210/900 rows, ceiling on every rung scored so far (through n40k16) | no reading-limit probe |
| `gemma4-12b-think` | no run files exist | not started |
| `qwen3-14b`, `qwen3-14b-think` | no run files exist | not started |

### The `qwen3-8b` prediction, resolved (partially)

`docs/graph-design-requirements.md` posed a falsifiable prediction: if
`k_bar` (not `n`) is the driver, `qwen3-8b` should already be informative at
**n40k12** — a fifth the length of the old size80 cell. It is not. `qwen3-8b`
ceilings straight through n40k12 (0.960) and every other rung up to n60k12
(0.960); it first turns informative at **n60k16 (0.880)**. So the *mechanism*
prediction (mean degree, not length, is what breaks it) is confirmed — but
the model needs a harder `k_bar` than 1.7b/2b did, not the cheap n40k12 cell
hoped for. That means the expensive, long rungs (n120–n300) are **not**
avoidable for this model — they're the only place it currently leaves
ceiling.

The catch: the reading-limit probe for `qwen3-8b` only ran to 6,305 tokens
and found no collapse in that range (flat 0.967–1.000). n60k16 (4,407
tokens) sits safely inside that, so it's a trustworthy primer test. Every
other informative rung for this model (n120k12 and up) sits **past the
probed range** — we cannot yet tell whether those are real primer-testable
cells or the model quietly hitting a reading limit nobody has measured.
`docs/graph-design-requirements.md`'s own recommended follow-up
(`build_retrieval_probe.py --statements 900 1200 1500`, ~9k/12k/15k tokens,
run on `qwen3-8b`/`gemma4-e4b`/`gemma4-12b`/`qwen3-14b` only) is what
resolves this, and it's cheap (no graphs, no model reasoning, just retrieval).

## Coverage beyond `node_degree`

The ladder is `node_degree`-only so far. For the other six tasks, the only
range data that exists is a `qwen3-1.7b`/`-think` node-count sweep in
**`docs/primer-impact-and-truncation.md`** (kept, not deleted — see below).
Read its node_degree columns as a `k_bar` proxy per the finding above, not as
an independent node-count effect; its `node_count`, `cycle_check`,
`connected_nodes`, `edge_existence`, `edge_count` and `reachability` columns
are not affected by that caveat and remain the best evidence for those tasks'
usable size ranges. Nothing yet re-derives them on a controlled `(n, k_bar)`
grid.

## Runs that are not worth new GPU time

- **The `components` primer on any plain Erdős–Rényi graph, at any density or
  size.** A single ER graph is connected with probability ≈1.00 at every
  density worth testing on any of the tasks here, so `components` cannot
  move — confirmed independently in `docs/primer-effects-and-power.md`
  (inert on `ec500`, +0.1pp pooled p=0.95 across the density sweep) and
  `docs/plans/finding-graphs-that-make-primer-effects-measurable.md` (plan
  §"secondary"). Don't re-run it on ER; it's structurally silent until
  `plant_components` (the disjoint-block generator in the plan, not yet
  built) exists.
- **New uncontrolled size sweeps** (`build_size_sweep.py`'s n20/40/60/80,
  density ~U(0,1) per graph) **for a new model or task.** This format
  conflates `n` and `k_bar` (see "n is a proxy," above) — a result from it
  can't tell you which one moved the needle. Existing data from it is still
  useful (it's the only source for six tasks — see "Coverage beyond
  node_degree"), but extending it to more models repeats a confound the
  ladder was built specifically to remove. Use the `(n, k_bar)` ladder
  instead.
- **New fixed-`n`, varying-density sweeps**, same reason — density at fixed
  `n` moves `k_bar` and length together; the ladder already separates them.
- **A full `rewire`-stage run (the expensive one — 3 rewiring levels x
  conditions) on `qwen3-0.6b` / `qwen3-0.6b-think` on this ladder.** Zero
  valid rungs; the `screen` stage already settled this at 1/18th the cost.
- **A full `rewire`-stage run on `qwen35-2b-think`** on this ladder as-is —
  it ceilings on all 18 rungs. It needs harder cells (`k_bar` pushed past 16,
  the already-screened-but-unbuilt range up to 24 — see
  `graphtalk/ladder.py`'s module docstring) before spending the expensive
  stage on it.
- **The 72-cell family-contrast grid** (regular/ER/WS/BA/SBM at matched
  `(n, m)`) proposed in the now-deleted `docs/handoff-structural-sweep.md`.
  Never built, and already decided against: swapping the *generator* moves
  `maj_base` by up to 6x on its own (0.15–1.000 across five families at
  matched `(n, m)`, per `docs/ladder-and-rewiring.md`), which swamps any
  primer effect. Degree-preserving rewiring (above) is what replaced this
  idea. Don't build it.
- **Extending `qwen3-8b`'s/`gemma4-*`'s `rewire`-stage runs onto the long
  rungs (n120+) before the reading-limit probe covers that range.** Per "The
  `qwen3-8b` prediction" above, those cells are the model's only informative
  ground on this ladder, but an unconfirmed-readable "informative" result
  can't be told apart from a reading artifact. Extend the probe first — it's
  far cheaper than the confirmatory `rewire` stage it would otherwise force a
  re-run of.

## Not yet resolved — worth doing, not yet done

- **Reading-limit probe gap for the big four models.** `qwen3-8b`'s probe
  covers only to 6,305 tokens; `gemma4-e4b`, `gemma4-12b`, `qwen3-14b` have
  no probe data at all. See "The `qwen3-8b` prediction" above.
- **`gemma4-e4b`/`gemma4-12b`/`qwen3-8b-think`'s ladder screens are
  incomplete** (partial rung coverage — see the status table). Finishing them
  (cheap `screen`-stage generation, `none` condition only) is what turns
  "ceiling on everything scored so far" into an actual verdict.
- **`_APPROX_CHARS_PER_TOKEN = 4` in `scripts/build_prompts.py:38` is still
  wrong.** Measured real ratio is 1.2–2.3 chars/token for this project's
  digit-dense graph encodings, never 4. It only feeds an overflow *warning*
  message today (not generation), so it's not urgent, but it will
  under-warn by 1.7–3.3x if anyone starts relying on it.
- **The plain-arm generation budget (`MAX_NEW_TOKENS = {"zero_shot": 2048}`
  in `graphtalk/models.py`) is still the default for `gemma4-e4b`,
  `gemma4-12b`, `qwen3-8b`, `qwen3-14b`.** For `qwen3-1.7b`/`node_degree` the
  measured tail ran up to 2,433 tokens against that same 2,048 default before
  it got its own 8,192 override; the four biggest models' plain-arm output
  tails haven't been measured on the harder ladder rungs. Worth a histogram
  before trusting a "no truncation" read on their `rewire`-stage runs.

## Superseded — deleted alongside this doc

- **`docs/node_degree-density-and-size.md`** — its "Combined takeaways"
  concluded *"both size and density are real, independent difficulty
  knobs."* They aren't, independently — see "The mechanism, corrected"
  above. Its underlying numbers were transcribed from
  `docs/primer-effects-and-power.md` ("Density at a fixed size"), which is
  still tracked and correct, so nothing is lost.
- **`docs/primer-impact-and-truncation-density-n40.md`** — its raw numbers
  are the same `node_degree`-at-n40-density data already in
  `docs/primer-effects-and-power.md` (its own header says as much: "cross-checked
  against their published tables... they agree to within rounding"). Its one
  original contribution, a ceiling/floor/informative recommended-range
  convention for density, is superseded by `graphtalk/ladder.py`'s
  `classify_band` + the per-model status table above, which does the same
  classification on the causally-correct axis (`k_bar`) and across models
  rather than one density sweep on one model.
- **`docs/handoff-structural-sweep.md`** — a design-session log. Its still-true
  findings (the chars/token bug, the overflow-vs-hit_cap distinction, the
  measured output budgets) are folded into this doc and
  `docs/graph-design-requirements.md`; its proposed 72-cell family grid was
  decided against (see "Runs that are not worth new GPU time," above); its
  environment notes (§9–11) were tied to one user's one cluster login session
  and are already stale there (test counts, missing-package list).

## References

- `docs/plans/finding-graphs-that-make-primer-effects-measurable.md` — the
  plan that motivated all of this; states the goal and the staged design.
- `docs/graph-design-requirements.md` — the four gates, per-model status as of
  its writing, and the mean-degree-vs-edge-count-vs-length derivation in full.
- `docs/ladder-and-rewiring.md` — why a ladder, why two axes, the
  degree-preserving rewiring mechanics, how to run it.
- `docs/primer-effects-and-power.md` — source of jobs 871262 (density vs edge
  count), 871263 (the `degree` ceiling), 871909 (the reading limit measured
  without a graph), and every other primer-effect number this doc cites.
- `docs/primer-impact-and-truncation.md` — per-task, per-size recommended
  ranges for the six tasks the ladder hasn't reached yet.
- `docs/sweep-findings.md` — "Truncation impersonates a finding," "Target the
  headroom," and the shortcut-bar reading convention (`bar(cond) -
  bar(none)`, never against zero) that every table above assumes.
