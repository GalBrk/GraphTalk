# Primer effects on 40-node graph question answering — analysis and results

Draft prose and tables for the paper. Written to `docs/` so it does not disturb
`paper/`; move it when the surrounding structure exists. Numbers come from
`csv2/raw-trends/`, figures from `analysis/raw-trends/`, both produced by
`scripts/build_raw_frame.py` and `superseded/scripts/raw_trends.py`.

Claims here are kept to what was measured. Where a mechanism is proposed it is
labelled as a hypothesis, and where a candidate explanation was tested and
survived or failed, that is said explicitly.

---

## Data and experimental design

We evaluate four model arms — `qwen3-1.7b` and `qwen3-4b`, each with and without
extended reasoning ("think") — on six GraphQA tasks over 40-node Erdős–Rényi
graphs. Edge probability is pinned rather than sampled, at four levels
(p = 0.1, 0.2, 0.35, 0.5) for all six tasks, and extended to three further levels
(p = 0.65, 0.75, 0.85) for `node_degree` and `edge_existence`. Each prompt is
built in one of seven primer conditions: `none` (graph encoding and question
only), five informative primers (`degree`, `clustering`, `rwse`, `components`,
`all`), and `filler`, a content-free preamble that writes one sentence per node
(*"Node 0 is simply present within the graph G."*) and therefore matches the
informative primers in length and format while carrying no graph information.
Every cell holds exactly 100 graphs, giving 84,000 scored responses over 700
distinct graphs.

The corpus is reconstructed rather than trusted. The generator derives its
Erdős–Rényi seed from `(density, size, index)` with no task term, so every graph,
every queried node and every gold answer is recoverable from a response's
identifier alone. We regenerate all 84,000 and require the recomputed gold answer
to match the one recorded at generation time; all 84,000 match. A consequence of
the seed formula is that a single graph serves all six tasks at a given
`(density, index)`, and that `node_degree`, `connected_nodes` and
`edge_existence` query the *same* node of that graph. This leaves the paired
comparison between conditions intact — within a task, two conditions differ only
in the primer — and additionally supplies a controlled comparison across required
output formats, which we use below. The intra-graph correlation of correctness is
0.12 (1.7B) and 0.06 (4B); because the graph effect cancels in paired
differences, clustering the bootstrap on graph identity changes a pooled
confidence interval from width 7.0 to 6.7, so we report it as a guard rather than
as a correction.

## Measurement decisions

Three choices materially change what the results say, and we state them before
the tables.

First, **every task is scored on exact match**, including `connected_nodes`.
That task is conventionally reported with set-F1, which compresses nearly all of
its variation: in one cell `qwen3-4b` under `filler` scores 94.3% F1 and 72%
exact, and `qwen3-1.7b` at p = 0.5 scores roughly 95% F1 against 38–51% exact. We
retain F1 as a secondary column only.

Second, **every effect is reported against two baselines**: against `none`, which
measures the net consequence of adding a primer, and against `filler`, which
isolates the effect of the primer's *content* from the effect of prepending a
preamble at all. These frequently disagree in sign, and the disagreement is
itself a result.

Third, **two of the six tasks have a constant gold answer at n = 40.**
`node_count` is always 40 and `cycle_check` is always "yes", because every graph
in this density range contains a cycle. Trials remain independent, so a deficit
from 100% is real signal — but it measures whether the model is displaced from an
answer it could obtain trivially, not whether it can reason about the graph. We
report both tasks and label them accordingly rather than dropping them.

Responses that exhaust the generation budget (`hit_cap`) are scored as wrong and,
separately, the analysis is repeated with them dropped; the non-termination rate
is reported as an outcome in its own right. Response-length statistics use
non-capped rows only, since a truncated response is short by construction, and
are suppressed entirely in cells where the cap rate exceeds 20%.

---

## Results

### Overall effect of each primer, by model and task

Tables 1–6 give exact-match accuracy per condition, pooled over density, with the
paired difference against `none` and against `filler`. Asterisks mark cells
significant at q < 0.05 after Benjamini–Hochberg correction within each
(model, task) family; significance is computed per density cell and the pooled
column is marked if any constituent cell survives.

#### Table 1 — `node_count` (constant gold: 40)

| condition | 1.7B acc | vs none | vs filler | 4B acc | vs none | vs filler | 1.7B-think acc | vs none | vs filler | 4B-think acc | vs none | vs filler |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `none` | 31.5 | -- | -39.8* | 90.2 | -- | -9.2* | 92.2 | -- | -7.7* | 99.5 | -- | -0.5 |
| `filler` | 71.2 | +39.8* | -- | 99.5 | +9.2* | -- | 100.0 | +7.7* | -- | 100.0 | +0.5 | -- |
| `degree` | 38.2 | +6.7* | -33.0* | 98.2 | +8.0* | -1.2 | 98.5 | +6.2* | -1.5 | 100.0 | +0.5 | +0.0 |
| `clustering` | 98.8 | +67.3* | +27.5* | 99.0 | +8.7* | -0.5 | 100.0 | +7.7* | +0.0 | 100.0 | +0.5 | +0.0 |
| `rwse` | 33.5 | +2.0* | -37.7* | 99.5 | +9.2* | +0.0 | 100.0 | +7.7* | +0.0 | 99.8 | +0.2 | -0.2 |
| `components` | 52.8 | +21.2* | -18.5* | 99.2 | +9.0* | -0.2 | 95.5 | +3.2* | -4.5 | 99.5 | +0.0 | -0.5 |
| `all` | 78.5 | +47.0* | +7.2* | 100.0 | +9.7* | +0.5 | 100.0 | +7.7* | +0.0 | 100.0 | +0.5 | +0.0 |

#### Table 2 — `cycle_check` (constant gold: yes)

| condition | 1.7B acc | vs none | vs filler | 4B acc | vs none | vs filler | 1.7B-think acc | vs none | vs filler | 4B-think acc | vs none | vs filler |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `none` | 92.2 | -- | -1.8 | 99.8 | -- | +0.2 | 98.5 | -- | -1.5 | 99.2 | -- | -0.8 |
| `filler` | 94.0 | +1.8 | -- | 99.5 | -0.2 | -- | 100.0 | +1.5 | -- | 100.0 | +0.8 | -- |
| `degree` | 73.2 | -19.0* | -20.7* | 96.8 | -3.0 | -2.8 | 85.8 | -12.7* | -14.2* | 96.0 | -3.2 | -4.0 |
| `clustering` | 99.2 | +7.0* | +5.2* | 99.5 | -0.2 | +0.0 | 98.8 | +0.2 | -1.2 | 100.0 | +0.8 | +0.0 |
| `rwse` | 91.8 | -0.5 | -2.2 | 100.0 | +0.2 | +0.5 | 99.8 | +1.2 | -0.2 | 100.0 | +0.8 | +0.0 |
| `components` | 93.8 | +1.5 | -0.2 | 97.7 | -2.0 | -1.8 | 97.5 | -1.0 | -2.5 | 98.8 | -0.5 | -1.2 |
| `all` | 98.0 | +5.7* | +4.0* | 99.2 | -0.5 | -0.2 | 98.2 | -0.2 | -1.8 | 99.5 | +0.2 | -0.5 |

#### Table 3 — `edge_existence`

| condition | 1.7B acc | vs none | vs filler | 4B acc | vs none | vs filler | 1.7B-think acc | vs none | vs filler | 4B-think acc | vs none | vs filler |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `none` | 73.3 | -- | +9.6* | 90.3 | -- | +6.7* | 99.1 | -- | +0.7 | 99.7 | -- | -0.3 |
| `filler` | 63.7 | -9.6* | -- | 83.6 | -6.7* | -- | 98.4 | -0.7 | -- | 100.0 | +0.3 | -- |
| `degree` | 76.4 | +3.1* | +12.7* | 86.1 | -4.1* | +2.6 | 97.6 | -1.6 | -0.9 | 100.0 | +0.3 | +0.0 |
| `clustering` | 76.0 | +2.7* | +12.3* | 94.9 | +4.6 | +11.3* | 98.4 | -0.7 | +0.0 | 100.0 | +0.3 | +0.0 |
| `rwse` | 76.6 | +3.3* | +12.9* | 84.3 | -6.0* | +0.7* | 98.6 | -0.6 | +0.1 | 99.7 | +0.0 | -0.3 |
| `components` | 68.1 | -5.1* | +4.4* | 87.3 | -3.0 | +3.7* | 98.7 | -0.4 | +0.3 | 100.0 | +0.3 | +0.0 |
| `all` | 82.7 | +9.4* | +19.0* | 83.6 | -6.7* | +0.0* | 97.7 | -1.4 | -0.7 | 99.9 | +0.1 | -0.1 |

#### Table 4 — `node_degree`

| condition | 1.7B acc | vs none | vs filler | 4B acc | vs none | vs filler | 1.7B-think acc | vs none | vs filler | 4B-think acc | vs none | vs filler |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `none` | 38.7 | -- | +2.6* | 80.3 | -- | -0.6 | 63.4 | -- | +6.9* | 97.4 | -- | -0.9 |
| `filler` | 36.1 | -2.6 | -- | 80.9 | +0.6 | -- | 56.6 | -6.9* | -- | 98.3 | +0.9 | -- |
| `degree` | 43.1 | +4.4 | +7.0* | 80.6 | +0.3* | -0.3* | 83.6 | +20.1* | +27.0* | 98.6 | +1.1 | +0.3 |
| `clustering` | 39.0 | +0.3 | +2.9 | 84.9 | +4.6* | +4.0* | 60.4 | -3.0 | +3.9* | 98.0 | +0.6 | -0.3 |
| `rwse` | 38.7 | -0.0 | +2.6 | 82.3 | +2.0* | +1.4 | 63.1 | -0.3 | +6.6 | 96.3 | -1.1 | -2.0 |
| `components` | 39.9 | +1.1 | +3.7 | 80.0 | -0.3* | -0.9* | 62.3 | -1.1 | +5.7* | 96.4 | -1.0 | -1.9 |
| `all` | 45.7 | +7.0 | +9.6* | 66.6 | -13.7* | -14.3* | 80.3 | +16.9* | +23.7* | 97.9 | +0.4 | -0.4 |

#### Table 5 — `connected_nodes` (exact match)

| condition | 1.7B acc | vs none | vs filler | 4B acc | vs none | vs filler | 1.7B-think acc | vs none | vs filler | 4B-think acc | vs none | vs filler |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `none` | 72.5 | -- | +6.2 | 86.0 | -- | +12.5* | 78.8 | -- | -2.5 | 97.0 | -- | -1.0 |
| `filler` | 66.2 | -6.2 | -- | 73.5 | -12.5* | -- | 81.2 | +2.5 | -- | 98.0 | +1.0 | -- |
| `degree` | 76.0 | +3.5 | +9.7* | 83.0 | -3.0 | +9.5* | 85.0 | +6.3 | +3.8 | 97.2 | +0.2 | -0.8 |
| `clustering` | 73.8 | +1.2 | +7.5* | 87.0 | +1.0 | +13.5* | 81.0 | +2.2 | -0.2 | 96.5 | -0.5 | -1.5 |
| `rwse` | 73.5 | +1.0 | +7.2 | 86.0 | -0.0 | +12.5* | 79.0 | +0.2 | -2.2 | 96.8 | -0.2 | -1.2 |
| `components` | 73.2 | +0.8 | +7.0* | 95.2 | +9.2* | +21.7* | 81.5 | +2.7 | +0.2 | 97.2 | +0.2 | -0.8 |
| `all` | 74.5 | +2.0 | +8.2* | 83.2 | -2.8 | +9.7* | 85.0 | +6.2 | +3.8 | 95.5 | -1.5 | -2.5 |

#### Table 6 — `edge_count`

| condition | 1.7B acc | vs none | vs filler | 4B acc | vs none | vs filler | 1.7B-think acc | vs none | vs filler | 4B-think acc | vs none | vs filler |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `none` | 1.2 | -- | +0.5 | 2.0 | -- | -1.2 | 25.2 | -- | -2.0 | 34.8 | -- | -9.7 |
| `filler` | 0.8 | -0.5 | -- | 3.2 | +1.2 | -- | 27.3 | +2.0 | -- | 44.5 | +9.7 | -- |
| `degree` | 4.2 | +3.0 | +3.5 | 29.3 | +27.3* | +26.0* | 21.8 | -3.5* | -5.5* | 41.0 | +6.2* | -3.5* |
| `clustering` | 0.8 | -0.5 | +0.0 | 0.8 | -1.2 | -2.5* | 24.8 | -0.5 | -2.5 | 52.8 | +18.0* | +8.2 |
| `rwse` | 0.5 | -0.8 | -0.2 | 0.5 | -1.5 | -2.7* | 21.3 | -4.0 | -6.0 | 45.7 | +11.0* | +1.2 |
| `components` | 0.2 | -1.0 | -0.5 | 0.8 | -1.2 | -2.5* | 22.8 | -2.5 | -4.5 | 59.7 | +25.0* | +15.2* |
| `all` | 2.0 | +0.8 | +1.3 | 5.8 | +3.7* | +2.5* | 34.2 | +9.0* | +7.0* | 28.0 | -6.7* | -16.5* |

### Reading the tables

**The two baselines disagree in sign often enough that neither alone determines
whether a primer helped.** The same condition can be harmful against one and
beneficial against the other. On `edge_existence` for `qwen3-4b`, `rwse` measures
−6.0 against `none` and +0.7 against `filler`: adding it to a bare prompt costs
six points, but a content-free preamble of the same length costs nearly seven, so
against a length-matched control the content is not harmful at all. The reverse
appears on `connected_nodes`, where the same model and primer measure 0.0 against
`none` and +12.5 against `filler` — a condition that looks perfectly inert turns
out to recover the entire cost that `filler` imposes on that cell. Reversals of
this kind are not confined to small effects: on `node_degree`, `qwen3-1.7b-think`
`degree` measures +20.1 against `none` and +27.0 against `filler`, a third of the
effect being masked by the fact that `filler` itself costs that cell 6.9 points.

The disagreement arises because `filler` is not inert. It carries no graph
information, yet it moves accuracy by as much as 39.8 points (`qwen3-1.7b`,
`node_count`) and by 12.5 points on `qwen3-4b` `connected_nodes`, in both
directions depending on the task. Any quantity of the form "accuracy with primer
minus accuracy without primer" therefore sums two distinct effects: the
consequence of prepending a preamble at all, and the consequence of what that
preamble says. Separating them requires a length- and format-matched control
condition in the design, not a correction applied afterwards.

This has a direct consequence for how a primer's value should be stated. We treat
the `filler` column as the content effect and the `none` column as the deployment
effect, and they answer different questions: whether a graph statistic carries
usable information, and whether adding it to a prompt is worthwhile in practice.
For `qwen3-4b` on `edge_existence`, `all` measures −6.7 against `none` and 0.0
against `filler` — its content is neither helping nor hurting, and the entire net
cost is the preamble. Reporting either column alone would state one of those two
facts and silently discard the other.

The second observation is that **effect size tracks how much room the model has
to move.** `qwen3-4b-think` scores 95–100% on five of six tasks and shows no
effect larger than ±4 points anywhere in Tables 1–5; `qwen3-1.7b`, which has the
widest headroom, carries the largest effects. This is an association across
arms; headroom varies with the model rather than being set by the design.

The third observation concerns the two constant-gold tasks. On `node_count`,
`qwen3-1.7b` scores **31.5%** under `none` — on a question whose answer is stated
in the prompt's own node range — and every primer improves it, `clustering`
reaching 98.8%. Inspecting the responses, the failure is an off-by-one: the model
answers **39** in 98% of `none` cases, evidently reading "nodes 0 through 39" and
returning the last index. Some primers displace it from that failure mode and
others do not, and the split does not follow primer length or format: `degree`
and `clustering` both write one sentence per node, yet `degree` leaves the model
at 38.2% and `clustering` lifts it to 98.8%. We report the pattern without a
mechanism. On `cycle_check`, where the answer is always "yes", `degree` *lowers*
accuracy by 19.0 points on 1.7B and 12.7 on 1.7B-think — a primer displacing a
model from an answer it otherwise gets almost for free.

### Effects depend on density, and can reverse

Pooling over density conceals reversals. Table 7 resolves `node_degree` by
density against the `filler` baseline, across the full range p = 0.1 to 0.85.

#### Table 7 — `node_degree`, effect vs `filler` by density (points)

| arm | condition | p=0.1 | p=0.2 | p=0.35 | p=0.5 | p=0.65 | p=0.75 | p=0.85 |
|---|---|---|---|---|---|---|---|---|
| 1.7B | `degree` | -2.0 | +17.0* | +18.0* | -1.0 | +1.0 | +12.0* | +4.0 |
| 1.7B | `all` | +1.0 | +17.0* | +17.0 | +8.0 | +8.0 | +14.0* | +2.0 |
| 4B | `degree` | -6.0 | -1.0 | -7.0 | -10.0 | -10.0 | +8.0 | +24.0* |
| 4B | `clustering` | +0.0 | +1.0 | +0.0 | -1.0 | +3.0 | +10.0 | +15.0* |
| 4B | `all` | -7.0 | -3.0 | -7.0 | -13.0* | -41.0* | -18.0 | -11.0 |
| 1.7B-think | `degree` | +1.0 | +9.0 | +27.0* | +24.0* | +38.0* | +37.0* | +53.0* |
| 1.7B-think | `all` | +1.0 | +12.0* | +26.0* | +23.0* | +47.0* | +31.0* | +26.0* |
| 4B-think | `degree` | +2.0 | -1.0 | +0.0 | +1.0 | -3.0 | +1.0 | +2.0 |

(Full table including `none`, `rwse` and `components` in
`superseded/csv2/raw-trends/effects.csv`.)

Two patterns stand out. On `qwen3-1.7b-think` the `degree` effect **grows
monotonically with density**, from +1 at p = 0.1 to +53 at p = 0.85 — the primer
becomes more valuable exactly as the model's unaided accuracy falls (63.4% pooled
under `none`, and 42–52% at the highest densities). On `qwen3-4b` the same primer
**reverses sign**: it costs 6–10 points up to p = 0.65 and then gains 8 and 24
points at p = 0.75 and 0.85. This is notable because the `degree` primer states
the queried node's degree verbatim in every one of these cells; what changes with
density is not the information available but whether the model does better using
it than not.

The `all` condition on `qwen3-4b` runs the other way, reaching −41 points at
p = 0.65. `all` concatenates `degree`, `clustering` and `rwse` into a single
per-node sentence, so it contains strictly more information than `degree` and
performs substantially worse.

### Combining primers does not combine their effects

Because `all` is exactly the union of `degree`, `clustering` and `rwse`, its
effect can be compared against the sum of its parts. Over the 71 cells where the
summed part-effects exceed 5 points in magnitude, the **median ratio of the
bundled effect to the sum of its parts is 0.37**. The relationship holds for
harm as well as help: on `qwen3-4b` `edge_existence` at p = 0.5 the parts sum to
−35 points and `all` measures −17. Two sample cells: `qwen3-4b` `edge_count` at
p = 0.5, parts +33 and `all` +9; `qwen3-1.7b` `edge_existence` at p = 0.5, parts
+50 and `all` +26. We note that in cells where the summed parts exceed the
available headroom the ratio is bounded by the ceiling and is not informative, so
the figure should be read as a tendency rather than a coefficient.

### Effects are close to one-directional

Because conditions are paired on identical graphs, each effect decomposes into
the items a primer fixed and the items it broke. These are markedly asymmetric.
At p = 0.5 on `qwen3-4b`, `degree` on `edge_count` fixes 33 items and breaks 0;
`rwse` on `edge_existence` fixes **0** and breaks 23; `degree` on `node_degree`
fixes 1 and breaks 13. `clustering` on `edge_existence` is the exception, fixing
8 and breaking 4. A net effect of +4 arising from 8 fixes and 4 breaks is a
different phenomenon from a net effect of +4 arising from 4 fixes and 0 breaks,
and we report fixed and broken counts alongside every difference.

### Accuracy on `edge_existence` is confounded by the class prior

The proportion of queried node pairs that are genuinely adjacent equals the edge
probability, so gold answers shift from 10% "yes" at p = 0.1 to 85% at p = 0.85,
and the majority-class baseline falls from 90% to 52% and then rises again to
85%. Raw accuracy therefore traces a U-shape in density that does not reflect
model skill. For `qwen3-1.7b` under `none`, raw accuracy falls from 89% at
p = 0.1 to 54% at p = 0.5 and then *recovers* to 86% at p = 0.85. Balanced
accuracy — the mean of recall on adjacent and non-adjacent pairs — instead reads
94, 56, and 53: the apparent recovery is the class prior moving, not the model
improving.

The error structure is one-sided. Recall on genuinely adjacent pairs is at or
near 100% in nearly every cell, for both model sizes and all conditions: the
models effectively never miss a real edge, and every error is a false positive.
Primers modulate that false-positive rate. On `qwen3-4b` at p = 0.5, where 48% of
pairs are adjacent, the proportion of pairs answered "yes" is 58% under `none`,
69% under `filler`, 74% under `degree`, 75% under `all`, 81% under `rwse`, and
54% under `clustering`. `clustering` is the only condition that moves the rate
*toward* the true base rate, and it is also the only condition that helps this
task for `qwen3-4b` (+11.3 against `filler`). That a node with local clustering
0.00 is evidence against adjacency offers a plausible account of why.

We emphasise that `filler` shifts the rate from 58% to 69% while carrying no
graph information, so a substantial part of the shift is attributable to the
presence of a preamble rather than to its content.

### At high density the smaller model stops producing reasoning

The bias above is not a gradual drift. For `qwen3-1.7b` on `edge_existence` under
`none`, median response length is 238 tokens at p = 0.35 and 206 at p = 0.5, then
falls to **50 and 48 tokens** at p = 0.65 and 0.75, while the proportion of pairs
answered "yes" rises to 98–99% and balanced accuracy sits at chance. The model
ceases to produce intermediate reasoning and emits a bare answer. `qwen3-4b` does
not show this: its median length *rises* with density (100 to 202 tokens) and its
balanced accuracy remains 79–89% throughout.

Primer content covaries with whether this occurs. At p = 0.75 and 0.85 the median
lengths are 48/120 tokens under `none`, 66/41 under `filler`, and 54/47 under
`components`, against 358/350 under `degree`, 319/314 under `clustering`,
320/378 under `rwse` and 376/404 under `all`. The three conditions that shorten
are the three carrying no per-node numeric content (`components` is a single
sentence; `filler` is contentless). Continuing to generate is not sufficient for
accuracy, however: `clustering` sustains length and still sits at 53% balanced
accuracy at p = 0.85, while `all` reaches 66%.

Two candidate explanations can be tested against existing data, because a
separate sweep varies node count and mean degree independently. Holding mean
degree at 8 while growing the graph from 40 to 300 nodes lengthens the prompt
roughly 7.5-fold and costs `qwen3-1.7b` 44 points of accuracy (84% to 40%), but
median response length *rises* over that range (94 to 122 tokens) — no shortening
occurs. Holding prompt length fixed instead, at 640 edges, and comparing a graph
requiring roughly 16 neighbours to be counted against one requiring 8, costs 34
points (24% against 58%), and the harder condition produces *more* tokens, not
fewer (218 against 106). Neither prompt length nor task difficulty alone
reproduces the shortening.

What does covary with it is the required answer format. On the same graphs at the
same densities, `qwen3-1.7b` under `none` produces 50 tokens on `edge_existence`
at p = 0.65 and 287 tokens on `node_degree`, scoring 68% and 8% respectively; at
p = 0.85, 120 tokens and 268 tokens, scoring 86% and 5%. Where the required
answer is binary the model stops early and scores well; where it is an integer it
continues to generate and fails. We note that at p = 0.85 answering "yes"
uniformly would score 85%, so the observed behaviour is close to the best
available strategy for a model unable to solve the task. Answer format covaries
with the task here rather than being manipulated within one, so we report the
covariation.

### Accuracy on `node_degree` depends on the queried node's position in the primer

The `degree` primer lists nodes in index order, one sentence each, so the queried
node's index is also its line number. Under `degree`, accuracy on `node_degree`
declines with that position: pooled over p = 0.35–0.85, `qwen3-4b` scores 82.3%
for target indices 0–9 and 69.6% for indices 30–39 (Spearman ρ = −0.09,
p = 0.014); `qwen3-1.7b-think` scores 91.2% against 73.3% (ρ = −0.16,
p < 0.0001). Because index and position are confounded, we repeat the analysis on
conditions with no answer to locate. Under `none` the same bucketing is flat for
`qwen3-4b` (+0.4 points, p = 0.79), as it is under `clustering` (−4.7, p = 0.32),
which has identical length and one-sentence-per-node format but does not state
the degree. The decline appears under the conditions that state the answer and
not under matched conditions that do not.

The contrast is one of steepness and monotonicity rather than of slope against
no slope: `filler` on `qwen3-1.7b-think` also declines, by 12.1 points
(p = 0.006). Separating position from node index outright is a question for a
shuffled- or reversed-order primer, which renders the identical facts in a
different sequence.

### The models are not invariant to an unrelated primer

Restricting to cells where the `none` accuracy lies between 20% and 95%, so that
a ceiling cannot be mistaken for insensitivity, we measure the mean absolute
accuracy shift produced by a primer whose content does not bear on the task:
9.1 points for `qwen3-1.7b`, 10.4 for `qwen3-4b`, 6.4 for `qwen3-1.7b-think` and
14.9 for `qwen3-4b-think`. The corresponding shifts produced by `filler` are
10.1, 9.8, 7.4 and 9.2. With the exception of `qwen3-4b-think`, whose value rests
on 14 cells concentrated in `edge_count`, an irrelevant primer displaces accuracy
no more than a content-free preamble of matched length does.

This measure interacts with the preceding section and should not be read alone: a
model that has stopped producing reasoning is insensitive to its prompt by
construction. `qwen3-1.7b`'s mean displacement falls from 16.4 points at p = 0.35
to 2.4 at p = 0.85, over exactly the range where its response length collapses.

### Differences in required output format, with the graph held fixed

Because `node_degree`, `connected_nodes` and `edge_existence` query the same node
of the same graph, the difference between them isolates the required output. The
ordering is not consistent across models. Under `none`, `qwen3-1.7b` scores lower
on counting a neighbour set than on reproducing it (−11, −18 and −18 points at
p = 0.2, 0.35 and 0.5), while `qwen3-4b` scores *higher* (+11, +15, +20). We
report the reversal; characterising it calls for a wider range of model sizes.

### Error magnitude on `edge_count`

On `edge_count`, errors under `none` are systematically low: for `qwen3-4b` at
p = 0.5 the median signed error over incorrect responses is −64 with 95% of
errors below the true value, and at p = 0.35 it is −26 with 85% below. Under
`degree` the bias disappears (median signed error +3 at p = 0.5, 40% below) while
the magnitude does not shrink (median absolute error 65 against 115). Inspecting
responses, `qwen3-1.7b` at p = 0.5 states a "sum the degrees and divide by two"
procedure in 7% of `none` responses and 100% of `degree` responses, while
explicit per-node enumeration falls from a median of 40 node mentions to zero,
and the non-termination rate falls from 29.8% to 9.5%. Among responses within a
single condition, those stating the summation procedure score no better than
those that do not (median difference +0.0 points across 23 cells); we note that
in the condition where the primer helps most the procedure is stated in 100% of
responses, leaving no within-condition contrast available there.

---

## Summary of what these measurements support

The measurements support the following descriptive statements.

A primer's measured effect depends on the baseline it is measured against, and
the two baselines disagree in sign in cells spread across every task. A
content-free preamble of matched length is not inert — it moves accuracy by up to
39.8 points, in both directions depending on the task — so a difference taken
against a bare prompt sums the effect of the preamble with the effect of its
content. Separating the two requires the matched control to be present in the
design.

Beyond that, effects are large in some cells and absent in others, and their size
tracks the accuracy the model reaches without a primer. Their sign depends on
density and on the model, and reverses within a single (model, task, primer)
combination as density changes: `degree` on `node_degree` costs `qwen3-4b` 6 to
10 points up to p = 0.65 and gains it 8 and 24 points at p = 0.75 and 0.85.
Bundling three primers into one yields a median of 0.37 of the sum of their
separate effects. At the item level effects are close to one-directional, with
conditions that fix many items and break none, or break many and fix none.

On `edge_existence`, accuracy is confounded with a class prior that moves from
10% to 85% across the density range; all errors are false positives, and primers
modulate the false-positive rate, with `clustering` the only condition that moves
it toward the true base rate. At high density the smaller model stops producing
intermediate reasoning on that task — median length falls from 238 to 48 tokens —
while continuing to reason at length on a task over the same graphs that requires
a numeric answer. Prompt length and task difficulty, varied independently in a
separate sweep, each cost accuracy without producing the shortening.

On `node_degree`, accuracy under a primer that states the answer declines with
the answer's position within the primer, and does not decline under length- and
format-matched conditions that omit the answer.
