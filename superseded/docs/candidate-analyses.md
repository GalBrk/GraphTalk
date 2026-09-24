# Candidate analyses for the paper (collected 2026-09-17)

Six directions raised while reviewing `superseded/paper/talk_like_a_graph.tex`, all run
against existing `runs/` data — no new generation. **Nothing here has been
folded into the paper.** This file exists so we can decide what is worth
including.

Verdicts at a glance:

| # | Direction | Verdict |
|---|---|---|
| a | Route gap as a better predictor than baseline | **Reject** — worse on every measure |
| b | Are `degree`-primer errors copying errors? | **Adopt** — direct mechanism evidence |
| c | Is the density continuum just answer magnitude? | **Adopt as defence** — confound ruled out |
| d | Discordant-pair decomposition | **Adopt, with care** — it weakens the headline |
| e | Generated-token length by condition | **Adopt** — independent corroboration |
| f | Retrieval probe and GoT naming, both unused | **Adopt both** — strongest unused assets |
| + | `clustering` replicates across graph size | **Adopt** — widens the one surviving claim |

The scripts that produced every number here are in
[`scripts/candidates/`](../../scripts/candidates/README.md), with a table mapping
each one to its section below. They are candidate-quality: no tests, hardcoded
run globs. Anything adopted needs rewriting into `scripts/` proper with unit
tests, the way `analyze_baseline_law.py` was.

---

## (a) Route gap instead of baseline — REJECT

**Hypothesis.** Route substitution predicts the effect should track the
*difference in reliability between the two routes*, `bar(task, cond) -
baseline`, not baseline alone. That predictor has the crossover built in at
zero, so it should transfer across corpora where the raw baseline does not.

**Result.** It is worse on every measure.

| corpus | cells | r(delta, baseline) | r(delta, gap) |
|---|---|---|---|
| densfull40, route | 177 | **-0.390** (p=8e-8) | +0.313 (p=2e-5) |
| densfull40, no route | 177 | +0.041 (p=0.59) | -0.231 (p=0.002) |
| heldout, route | 73 | **-0.662** (p=2e-10) | +0.451 (p=6e-5) |
| heldout, no route | 54 | -0.206 (p=0.14) | +0.261 (p=0.056) |

The gap also destroys the control: it correlates with the effect in the
*no-route* group too (p=0.002), which baseline does not. Cross-corpus sign
transfer, fit on densfull40 and predicted on heldout route cells:

- baseline: 59/73 = 0.808 (majority 0.699)
- gap: 40/73 = 0.548 (majority 0.699)

**One incidental correction for the paper.** The paper says predicting the
sign on a new corpus "does no better than chance" (`talk_like_a_graph.tex:474`).
Restricted to route cells it is 0.808 against a 0.699 majority — better than
chance, though not impressively: the fitted line predicts positive for only
10/73 cells, and 63/73 held-out route cells sit above the densfull40 crossover
of 0.792, so it is close to "predict negative everywhere". The claim as
written is slightly too strong; "barely beats the majority class, because the
held-out corpus has almost no baseline spread (median 1.000)" is the accurate
version.

---

## (b) Are the `degree`-primer errors copying errors? — ADOPT

**Promoted.** `scripts/candidates/b_copying.py` is superseded by
`superseded/scripts/analyze_error_taxonomy.py` (node_degree_taxonomy), which covers all
4 arms x 7 conditions instead of the subset below, adds an off-by-k
breakdown and a primer-position analysis, and has tests. The file here is
kept for its original numbers below and is no longer run.

**Design.** For every `node_degree` instance in `densfull40`, parse the graph
and the queried node `k` out of the prompt. For each *wrong* answer, ask
whether the value the model produced is the degree the primer states for a
node *adjacent in id* to `k` — i.e. an off-by-one read of the primer list.

| arm | cond | wrong | = degree of k±1 |
|---|---|---|---|
| qwen3-4b | **degree** | 29 | **41.4%** |
| qwen3-4b | all | 35 | 5.7% |
| qwen3-4b | rwse | 12 | 8.3% |
| qwen3-4b | clustering | 5 | 20.0% |
| qwen3-4b | none / filler / components | 3 / 5 / 2 | 0.0% |
| qwen3-1.7b (all 7 conditions) | — | 1550 | 17.3% |

Exact binomial, 12/29 against the 17.3% background: **p = 0.002**.

**Reading.** In exactly the cell where the paper reports `degree` costing
qwen3-4b -6.5 points, its errors are misindexed reads of the primer list, at
2.4x the background rate — and no other condition on that arm does this. That
is route substitution observed directly rather than inferred from a slope, and
it is most of what the proposed corrupted-primer experiment would buy, for
free. The 1.7B arms show no elevation at all (17.2% under `degree` against
18.4% under `none`), so the failure mode is specific to the arm the primer
harms.

Two caveats: n=29, and `all` (which contains `degree`, and costs 4B -8.0)
does *not* show it — 5.7%, and its wrong answers fall in the degree multiset
*less* often than chance. The two harmful conditions fail differently, which
is worth a sentence but is not explained here.

---

## (c) Is Table 4's "difficulty" just answer magnitude? — ADOPT AS DEFENCE

**The threat.** In the density continuum, gold degrees run ~4 at p=0.10 and
~34 at p=0.85. The ramp could be large-integer arithmetic, not traversal. A
reviewer will ask.

**Test 1 — `runs/qwen3-1.7b.degfixdeg`,** which pins mean degree and sweeps
`n` from 20 to 160. Answer magnitude fixed, graph size 8x:

| mean deg | n | `none` acc | mean gold |
|---|---|---|---|
| 8 | 20 | 0.912 | 8.0 |
| 8 | 40 | 0.728 | 7.8 |
| 8 | 80 | 0.645 | 8.0 |
| 8 | 160 | 0.624 | 7.7 |
| 16 | 20 | 0.575 | 16.2 |
| 16 | 40 | 0.398 | 16.1 |
| 16 | 80 | 0.225 | 16.1 |
| 16 | 160 | 0.224 | 16.2 |

Accuracy falls by a third at constant answer magnitude, so size matters on its
own. Comparing rows at equal `n` shows magnitude matters too.

**Test 2 — within-density decomposition** on the continuum corpus
(`qwen3-1.7b-think`, `none`, 2784 instances). Regressing per-instance
correctness on gold degree and density jointly:

| term | coef | t |
|---|---|---|
| gold degree | -0.0022 | -0.71 |
| density | **-0.594** | **-4.67** |

Marginally the two look identical (r = -0.360 and -0.369 against correctness),
but **jointly only density survives.** The confound is ruled out.

Caveat: the two predictors are strongly collinear, so the separation rests on
limited overlap between gold-degree bands and density levels. The bands that
do overlap agree — e.g. gold degree 20-24 scores 0.62 at p=0.50 and 0.41 at
p=0.65.

**Where it goes.** Two sentences in Results or Limitations. It costs little
and closes an obvious line of attack on the strongest table in the paper.

---

## (d) Discordant-pair decomposition — ADOPT, WITH CARE

**Promoted.** `scripts/candidates/de_churn_len.py` + `de_report.py` are
superseded by `superseded/scripts/analyze_churn_and_length.py`, a single tested script
that reproduces the same numbers (verified: 50,876 pairs, 5,239 discordant,
52.1% of discordant pairs cancel in net). The files here are kept for
history and are no longer run.

Every reported delta is a net. McNemar already computes the two discordant
counts; printing them separates "helps a coherent subset" from "randomises
answers with a favourable net". `sig/churn` below is |help - hurt| / (help +
hurt): 1.00 is one-directional, 0.00 is pure churn.

| arm | task | cond | n | help | hurt | net | sig/churn |
|---|---|---|---|---|---|---|---|
| qwen3-4b | edge_count | degree | 393 | 114 | 5 | +27.7 | **0.92** |
| qwen3-4b | node_degree | degree | 400 | 3 | 29 | -6.5 | **0.81** |
| qwen3-4b | node_degree | all | 400 | 2 | 34 | -8.0 | **0.89** |
| qwen3-1.7b | node_degree | degree | 400 | 57 | 27 | +7.5 | 0.36 |
| qwen3-1.7b | node_degree | all | 400 | 77 | 36 | +10.2 | 0.36 |
| **qwen3-1.7b** | **node_degree** | **clustering** | 400 | 53 | 41 | +3.0 | **0.13** |
| qwen3-1.7b | edge_existence | filler | 400 | 10 | 73 | -15.8 | 0.76 |
| qwen3-1.7b | connected_nodes | filler | 400 | 44 | 85 | -10.2 | 0.32 |
| qwen3-4b | connected_nodes | filler | 400 | 24 | 81 | -14.2 | 0.54 |
| qwen3-1.7b | node_count | clustering | 400 | 269 | 0 | +67.2 | 1.00 |

Across all 546 densfull40 cells: 50,876 pairs, 5,239 discordant (10.3%), of
which only 52.1% are net — roughly half of all primer-induced answer changes
cancel out.

**This cuts against the paper.** The headline surviving cell,
`clustering`/`node_degree`/1.7B, is the churniest in the table: 94 answers
change to produce a net of 12. The route-substitution cells are the clean
ones (0.81-0.92, essentially one-directional), which corroborates (b) — but
the one effect the paper offers as a genuine content gain looks much more like
churn with a favourable remainder. We should report this ourselves rather than
have a reviewer find it.

---

## (e) Generated-token length by condition — ADOPT

`n_new_tokens` is in every run row and has never been analysed. Mean paired
change against `none`, truncated pairs excluded from both sides:

| arm | components | clustering | rwse | degree | filler | all |
|---|---|---|---|---|---|---|
| qwen3-1.7b | -0.0 | +42.3 | +216.1 | -21.4 | **-102.4** | +94.7 |
| qwen3-1.7b-think | -37.1 | -161.7 | +141.8 | **+670.4** | **-369.0** | -35.8 |
| qwen3-4b | -7.0 | -121.3 | -128.0 | +19.1 | **-161.7** | -144.8 |
| qwen3-4b-think | +43.3 | -302.1 | -76.0 | -76.3 | **-261.2** | -414.5 |

Per task, `qwen3-1.7b-think`: `degree` adds +859 tokens on `node_degree` and
+1690 on `cycle_check`, while `filler` removes -228 and -822 on the same two.

**Reading.** The content-free control *shortens* generation in all four arms —
added input text buys less reasoning, not more. The answer-stating primer
*lengthens* it sharply in the thinking arm. The two conditions the paper
contrasts on accuracy move generation length in opposite directions too, which
is an independent signature of the same dissociation and does not depend on
the baseline-accuracy argument at all. It also makes "length is an active
treatment, not a covariate" (Discussion, second regularity) concrete rather
than inferential.

Note: truncated (`hit_cap`) pairs must be dropped from both sides here, not
just from accuracy. Left in, they contribute a censored 8192 and swing
`qwen3-1.7b`/`edge_count`/`degree` from -1098 to -2181.

---

## (f) Two datasets we own and do not use — ADOPT BOTH

### f1. The retrieval probe: how reliable IS the primer route?

`runs/qwen3-1.7b.retrieval` — 3,600 instances, 0 truncated, `node_degree`,
varying the number of facts `k` in the preamble and the position of the needle.
The paper currently parks the retrieval probe as never run.

| k | acc | pos 0.1 | pos 0.5 | pos 0.9 |
|---|---|---|---|---|
| 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| 40 | 1.000 | 1.000 | 1.000 | 1.000 |
| 160 | 0.985 | 0.985 | 0.990 | 0.980 |
| 320 | 0.848 | 0.925 | 0.800 | 0.820 |
| 640 | 0.655 | 0.975 | **0.195** | 0.795 |
| 1280 | 0.503 | 0.585 | 0.470 | 0.455 |

r(k, correct) = -0.499 (p=8e-226); r(position, correct) = -0.076 (p=4e-6).

**Reading.** A textbook lost-in-the-middle curve, and it supplies the one
quantity the route-substitution account assumes but never measures: the
reliability of the primer route. At our n=40, k=40, so retrieval is perfect —
the primer route is not degraded by reading difficulty at all, and the whole
effect must come from whether the model *uses* it. That is a sharper statement
than the paper currently makes, and it retires the `liu2024lostinthemiddle` /
`kamradt2023needle` citations from decoration to evidence.

It also predicts the design where primers *would* fail for retrieval reasons:
k >= 320, i.e. graphs past ~300 nodes. Worth one sentence as a scope boundary.

### f2. GoT naming: is route substitution just integer-token copying?

Eight arms on the published split, run with Game-of-Thrones character names
instead of integer ids, scored through `node_naming.desubstitute_response`.
Paired delta against `none` under each naming:

| cond | mean INT | mean GOT | cells | r(INT, GOT) |
|---|---|---|---|---|
| **degree** | +1.59 | +2.69 | 39 | **0.933** |
| all | +0.90 | +1.51 | 39 | 0.333 |
| components | -0.24 | +0.45 | 39 | 0.345 |
| clustering | +0.92 | +0.92 | 39 | 0.079 |
| rwse | -0.86 | +0.61 | 38 | -0.381 |
| ALL | +0.47 | +1.24 | 194 | 0.542 (p=3e-16) |

Sample cells: `qwen3-8b`/`edge_count`/`degree` +23.3 (INT) vs +35.7 (GOT);
`qwen3-14b`/`edge_count`/`degree` +20.0 vs +30.0.

**Reading.** The answer-stating primer's effect is near-perfectly preserved
when node ids stop being integers (r = 0.933), and if anything is *larger*.
The non-route primers correlate at ~0 across namings, which is what you expect
of conditions with no real effect. So route substitution is not an artifact of
integer tokenisation — it survives a complete change of surface form. This is
the cleanest generalisation test in the whole dataset.

Note `filler` is absent: the published-split arms were run without it, so
there is no length control on this corpus.

---

## Bonus: the `clustering` headline replicates across graph SIZE

Found while running (c). `degfixdeg` pins mean degree and sweeps `n`, and it
carries `none` and `clustering` — so it is an untouched replication set for the
one cell that survives every control. `qwen3-1.7b`, `node_degree`:

| n | p | mean deg | pairs | help | hurt | delta | McNemar |
|---|---|---|---|---|---|---|---|
| 20 | 0.421 | 8 | 400 | 12 | 31 | -4.8 | 0.0054 |
| 20 | 0.842 | 17 | 400 | 105 | 39 | +16.5 | 3e-8 |
| 40 | 0.205 | 8 | 400 | 49 | 32 | +4.2 | 0.075 |
| 40 | 0.410 | 16 | 399 | 80 | 49 | +7.8 | 0.0080 |
| 80 | 0.101 | 8 | 400 | 61 | 19 | +10.5 | 4e-6 |
| 80 | 0.203 | 16 | 400 | 78 | 39 | +9.8 | 0.0004 |
| 160 | 0.050 | 8 | 394 | 79 | 52 | +6.9 | 0.0227 |
| 160 | 0.101 | 16 | 398 | 40 | 45 | -1.3 | 0.66 |
| **pooled** | | | **3191** | **504** | **306** | **+6.2** | **3.5e-12** |

Positive at 6/8 design points, n = 20 to 160. The paper's claim rests on n=40
plus one seed replication (+3.8, +4.3); this is a third, independent corpus
spanning an 8x range of graph size, agreeing at n=40 (+4.2, +7.8) and pooling
to +6.2.

It is not universal: significantly *negative* at n=20/deg8 (-4.8, p=0.005) and
null at n=160/deg16. Reporting the full grid rather than the pooled number is
the honest version, and the two failures are informative — the effect needs
both enough nodes and enough density to appear.

Note this sits awkwardly with (d): the same cell is the churniest in the
corpus. Both are true. The effect is real, replicated across three corpora,
and mostly churn with a consistently favourable remainder.

---

## What this would change in the paper

Ordered by how much it moves the argument:

1. **(f1) retrieval** and **(f2) GoT** are whole results sitting unused. Either
   would be a section; both are stronger than some of what is in the paper now.
   Both are currently parked material, so including them is a scope decision,
   not just an editing one.
2. **(b) copying** converts the central behavioural claim into a mechanism
   claim without any interpretability machinery — and makes the Future Work
   section's corrupted-primer proposal largely redundant.
3. **(bonus) clustering across size** widens the one positive claim from one
   corpus to three.
4. **(c) confound** and **(e) length** are cheap defensive additions.
5. **(d) churn** must be reported if we keep the clustering headline at its
   current strength. It is the one item here that makes the paper weaker, which
   is why it should not be left for a reviewer.
6. **(a)** should be recorded as tried and rejected, and the "no better than
   chance" sentence at `talk_like_a_graph.tex:474` softened to match what
   actually happens.
