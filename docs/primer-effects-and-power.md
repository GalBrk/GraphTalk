# Where primers actually help, and why the design has not yet been able to tell

Written 2026-09-05, after adding `qwen3-0.6b`/`qwen3-1.7b`/`qwen35-2b` to the
suite and probing at `--count 100` on the `none`/`degree` conditions.

This document re-reads existing sweep data through the shortcut bar and reaches
a conclusion about the *design* rather than about primers.

> **Analysis pitfall this document was initially wrong about.** Scoring
> `runs/*.jsonl` by calling `scoring.extract_answer` directly is **incorrect for
> the 28 `.got.` files**: a GoT-named response answers in character names
> ("Maester, Catelyn") while `gold` stays integers, so it must go through
> `node_naming.desubstitute_response` first (see CLAUDE.md, "node_naming.py").
> Pooling them undesubstituted corrupts `connected_nodes` specifically -- the
> only task whose *answer* is a node list -- and made `qwen3-8b` appear to score
> 0.080 there when it actually scores 0.996. Every number below **excludes
> `.got.` files**. Anyone writing a fresh analysis script will hit this trap;
> `scripts/build_sweep_frame.py` is the supported path that handles it.

## Summary

Updated 2026-09-06, after running the powered experiments. The earlier version
of this document said the uncontaminated cells were "positive but unpowered" and
that no cell cleared every control. At power, both statements are wrong.

1. **Three cells now clear every control** -- powered, significant, above the
   relevant baseline, and not reproducible by the shortcut solver:

   | model | task | primer | delta | disc | p |
   |---|---|---|---|---|---|
   | qwen3-8b | edge_count | **rwse** | **-13.7 pp** | 108 | 0.0000 |
   | qwen3-8b | edge_count | **clustering** | **+5.0 pp** | 69 | 0.0076 |
   | qwen3-0.6b-think | cycle_check | **clustering** | **+4.3 pp** | 55 | 0.0065 |

2. **"Do primers help?" has no single answer -- it depends on the primer.** On
   one task, one model, at n=500: `clustering` helps by 5 points, `components`
   does nothing (+1.3, p=0.54), and `rwse` *hurts by 13.7 points*. The largest
   clean effect in the project is negative.

3. **`clustering` is the only primer that helps twice**, on two different tasks
   and two different models (`edge_count`/8B and `cycle_check`/0.6B-think). It
   is the first pattern here that repeats across cells rather than appearing
   once. **It did not survive replication on a third**: `qwen3-1.7b` on the
   same `ec500` cell scores -1.3 pp (p=0.64). See "The `qwen3-1.7b`
   replication".

3b. **The replication qualifies items 1 and 2 and should be read with them.**
   `rwse`'s harm reproduces in *direction* on `qwen3-1.7b` but at -4.0 pp
   (p=0.11) rather than -13.7, and `clustering`'s help does not reproduce at
   all. Neither headline is model-general on the evidence available; both are
   strongest on `qwen3-8b`.

4. **Five of six tasks remain saturated** for the larger models, and every large
   `degree` gain remains shortcut-explained. Those findings are unchanged.

5. **Saturation is an artifact of the corpus's 19-node cap, not of the tasks.**
   Regenerated at 80 nodes, `node_degree` falls to 0.143 (1.7B plain) and 0.479
   (8B plain) -- but `node_count` stays at 1.000. What breaks is aggregation
   over scattered mentions, which graph size multiplies; see "Does size break
   them?"

The honest status of the proposal's question is no longer "the experiment has
not been run". It has been run. The answer is that primers are not one
intervention: some carry usable structure, some are inert, and at least one is
actively misleading.

## The shortcut bar decides which cells are informative

`shortcuts.json` (`scripts/shortcut_table.py --graphs 500`) scores a primer-only
solver that never sees the graph:

| task | components | clustering | rwse | degree | all | filler | none |
|---|---|---|---|---|---|---|---|
| node_count | 0.08 | 1.00 | 1.00 | **1.00** | 1.00 | 1.00 | 0.06 |
| edge_count | **0.02** | **0.15** | **0.02** | **1.00** | 1.00 | 0.02 | 0.02 |
| node_degree | 0.08 | 0.08 | 0.62 | **1.00** | 1.00 | 0.08 | 0.08 |
| connected_nodes | 0.08 | 0.08 | 0.08 | 0.21 | 0.35 | 0.08 | 0.08 |
| edge_existence | 0.50 | 0.72 | 0.64 | 0.79 | 0.79 | 0.50 | 0.50 |
| cycle_check | 1.00 | 0.83 | 0.83 | 0.95 | 0.95 | 0.83 | 0.83 |

**`degree` is the worst condition for testing the hypothesis and the one the
sweep has most data on.** It gives the answer away on three of six tasks.

Cross that against headroom and only one row survives: **`edge_count`**, which
has both real headroom and three clean conditions.

## Every large primer gain is shortcut-explainable

`none` vs `degree`, GoT and capped rows excluded:

| model | task | none | degree | delta | n | disc | p | bar | beats bar? |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b-think | edge_count | 0.324 | 0.873 | +0.549 | 71 | 39 | 0.0000 | 1.00 | no |
| qwen3-1.7b | edge_count | 0.237 | 0.691 | +0.454 | 97 | 48 | 0.0000 | 1.00 | no |
| qwen3-8b | edge_count | 0.433 | 0.667 | +0.233 | 30 | 9 | 0.039 | 1.00 | no |
| qwen3-14b | edge_count | 0.400 | 0.600 | +0.200 | 30 | 8 | 0.070 | 1.00 | no |
| qwen3-1.7b | node_degree | 0.884 | 0.953 | +0.070 | 86 | 8 | 0.070 | 1.00 | no |
| gemma4-12b | edge_count | 0.933 | 1.000 | +0.067 | 30 | 2 | 0.500 | 1.00 | ties |
| qwen3-0.6b | connected_nodes | 0.834 | 0.875 | +0.041 | 100 | 8 | 1.000 | 0.21 | YES |

Highly significant, reproducible across five models spanning 0.6B-14B, and
uninterpretable as a reasoning claim: the primer contains the answer.

## The clean cells: positive, and never powered

Cells with bar < 0.35, GoT excluded:

| model | task | cond | none | primer | delta | n | disc | p | bar |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | edge_count | clustering | 0.433 | 0.567 | +0.133 | 30 | 6 | 0.219 | 0.15 |
| qwen3-14b | edge_count | components | 0.400 | 0.500 | +0.100 | 30 | 3 | 0.250 | 0.02 |
| qwen3-14b | edge_count | clustering | 0.400 | 0.467 | +0.067 | 30 | 2 | 0.500 | 0.15 |
| qwen3-0.6b | connected_nodes | degree | 0.834 | 0.875 | +0.041 | 100 | 8 | 1.000 | 0.21 |
| qwen3-8b | edge_count | components | 0.433 | 0.467 | +0.033 | 30 | 7 | 1.000 | 0.02 |
| qwen3-14b | edge_count | rwse | 0.400 | 0.433 | +0.033 | 30 | 1 | 1.000 | 0.02 |

Every top cell is `edge_count`. Effects are positive and clear their bars.
**Every one is at n=30 with 1-7 discordant pairs.** Power and clean cells are in
disjoint places: `degree` was scaled to n=500, the clean conditions never were.

## The experiment, and what it found

**`edge_count` x {`components`, `clustering`, `rwse`} vs `none`, n=500,
`qwen3-8b`** (`prompts.edgecount500.clean.jsonl`, tag `ec500`). 1,984/2,000 rows
-- one shard timed out 16 rows short after losing ~3 h to checkpoint warm-up
contention -- 56 capped (2.8%), **0 unparsed**, conditions balanced at n=477-489.

| condition | score | delta (pp) | paired | disc | p | bar | > bar |
|---|---|---|---|---|---|---|---|
| none | 0.520 | -- | 477 | | | 0.02 | -- |
| components | 0.536 | +1.3 | 463 | 66 | 0.539 | 0.02 | YES |
| **clustering** | **0.578** | **+5.0** | 460 | 69 | **0.0076** | 0.15 | YES |
| **rwse** | **0.380** | **-13.7** | 468 | 108 | **0.0000** | 0.02 | YES |

Discordance ran 14-23%, so n=500 delivered 66-108 discordant pairs per cell
against the 1-7 that every earlier cell had. This is the first cell in the
project with enough power to distinguish an effect from nothing.

**`rwse` hurting by 13.7 points is the largest clean effect measured anywhere in
this project.** Its bar is 0.02, so it cannot be dismissed as shortcut
interference -- a random-walk structural encoding genuinely degrades the model's
edge counting. Any framing of primers as "extra information, at worst neutral"
is refuted by this cell.

`components` behaving as a null (+1.3, p=0.54) is the right control result:
component count carries little about edge count, and the measurement says so.

### The `qwen3-1.7b` replication

**Same cell, same prompts, smaller model** (`prompts.edgecount500.clean.jsonl`,
tag `ec500`, job 858244, 2,000/2,000 rows, **0 unparsed**). Truncated rows are
dropped, not scored as failures -- the house rule, and a deliberate choice here
rather than a default; see the note below on why it is load-bearing for `rwse`.

| condition | score | delta (pp) | paired | disc | p | bar | replicates? |
|---|---|---|---|---|---|---|---|
| none | 0.240 | -- | 487 | | | 0.02 | -- |
| components | 0.224 | -2.1 | 469 | 108 | 0.382 | 0.02 | yes, still inert |
| clustering | 0.235 | -1.3 | 473 | 120 | 0.643 | 0.15 | **no** |
| **rwse** | **0.211** | **-4.0** | 426 | 101 | 0.111 | 0.02 | direction only |

The baseline lands where it was predicted to (0.240 against the 0.237 this model
scored on `edge_count` in the small-model suite), so the run is sound and the
differences below are about the primers, not the setup.

**`clustering` does not replicate.** +5.0 pp (p=0.0076) on the 8B becomes
-1.3 pp (p=0.64) here. Read against its bar this is worse than it looks: a
primer-only solver *gains 13 points* from the clustering primer
(`bar` 0.15 vs 0.02), and this model gained nothing. Both models are in fact
below that bar -- the 8B's celebrated +5.0 pp is itself less than a trivial rule
extracts from the same text -- so "clustering helps" was never as clean as item
1 of the Summary implies.

**`components` replicates as a null**, which is the control behaving correctly
for a second time.

**`rwse` hurts here too, but at roughly a third the magnitude and without
significance** (-4.0 pp, p=0.11). Under the doc's own pre-registered framing --
"if it replicates the claim is solid; if not it is a `qwen3-8b` property" -- the
answer is neither. The sign is reproducible; the 13.7-point magnitude is not.

**The truncation asymmetry is a finding in its own right, and it is where most
of `rwse`'s harm now lives.** Capped rows by condition:

| condition | capped |
|---|---|
| none | 13/500 (2.6%) |
| clustering | 15/500 (3.0%) |
| components | 19/500 (3.8%) |
| **rwse** | **63/500 (12.6%)** |

`rwse` makes this model fail to terminate roughly five times as often as no
primer at all. That is a real effect of the condition, not a nuisance -- which
makes dropping those rows a substantive decision rather than hygiene. Scoring
them as failures instead moves `rwse` to -5.0 pp at p=0.025, i.e. from
non-significant to significant. **This document reports the filtered number**,
on the grounds that the accuracy metric should measure edge counting rather than
`max_new_tokens`; the non-termination rate is reported separately, above, rather
than folded into it. Anyone re-analysing this cell should know the conclusion
turns on that choice and state which one they used.

### The `cycle_check` experiment

**`cycle_check` x {`components`, `clustering`, `rwse`} vs `none`, n=500,
`qwen3-0.6b` both arms** (`prompts.cyclecheck500.clean.jsonl`, tag `cc500`),
2,000/2,000 rows each.

Chosen because `cycle_check` is the only task where the *shortcut information a
primer adds* -- `bar(cond) - bar(none)`, a sharper criterion than the absolute
bar used earlier in this document -- is exactly zero for two conditions:

| task | components | clustering | rwse | degree |
|---|---|---|---|---|
| cycle_check | +0.17 | **+0.00** | **+0.00** | +0.11 |

`components` is therefore a built-in *within-task positive control*: it hands
over real shortcut content while `clustering`/`rwse` hand over none.

**PLAIN (`qwen3-0.6b`)** -- majority-class floor is 0.832:

| cond | score | delta (pp) | disc | p | vs floor |
|---|---|---|---|---|---|
| none | 0.792 | -- | | | -0.040 |
| components | 0.832 | +4.1 | 28 | 0.0002 | +0.000 |
| clustering | 0.811 | +2.1 | 48 | 0.193 | -0.021 |
| rwse | 0.826 | +3.0 | 37 | 0.020 | -0.006 |

Every condition lands at or below the floor. The plain 0.6B never escapes
majority-guessing on this task; the "gains" only lift it back to it.

**THINK (`qwen3-0.6b-think`):**

| cond | score | delta (pp) | disc | p | vs floor |
|---|---|---|---|---|---|
| none | 0.843 | -- | | | +0.011 |
| components | 0.864 | +1.5 | 15 | 0.119 | +0.032 |
| **clustering** | **0.885** | **+4.3** | 55 | **0.0065** | **+0.053** |
| rwse | 0.848 | +0.6 | 17 | 0.629 | +0.016 |

Survives Bonferroni across all six comparisons (0.0065 x 6 = 0.039), and the
shortcut-rich `components` control does *not* move -- so this is not generic
primer-presence and not shortcut exploitation.

**The mechanism is legible in the class split** (84 acyclic / 412 cyclic):

| cond | acyclic ("No") | cyclic ("Yes") |
|---|---|---|
| none | 0.083 | 0.998 |
| components | 0.179 | 0.998 |
| **clustering** | **0.506** | 0.963 |
| rwse | 0.131 | 0.995 |

Without a primer the model is doing pure majority-guessing: 99.8% on cyclic,
8.3% on acyclic. `clustering` takes acyclic detection **8.3% -> 50.6%**, which
is the entire effect.

The model is reading "every clustering coefficient is 0" as triangle-free and
inferring acyclic. **That is not the theorem `shortcuts.py` implements.** Its
`clustering_triangle` rule runs the sound direction -- clustering > 0 *proves* a
cycle -- which is worth nothing here because "yes" is already the majority
guess. The model uses the converse, which is a **heuristic, not a valid
inference**: a graph can be triangle-free and still hold a 4-cycle.

It visibly pays that price: cyclic accuracy drops 0.998 -> 0.963, i.e. it now
answers "no cycle" on ~3.5% of cyclic graphs -- exactly the triangle-free ones
with longer cycles. That error signature is what confirms the mechanism rather
than merely correlating with it.

### What was not run

`qwen3-14b` on `ec500` was cancelled at 206/2,000 rows. Its shard 0 spent ~10 of
its 12 h in the page-cache warm-up (see Operational notes) and would have needed
a full resubmission. **So `edge_count` has a single model behind it**, and the
`clustering`/`rwse` results are unreplicated. `qwen3-1.7b` -- which has the most
`edge_count` headroom in the suite at 0.237 -- is the cheapest replication and
was deliberately deferred until the 8B result existed.

`degree` should be dropped from the headline regardless. Keep it as a documented
positive control for shortcut exploitation, which it demonstrates unusually well.

## Retracted from the first draft of this document

- ~~"`connected_nodes` scoring is broken; a 0.6B beats an 8B tenfold"~~ -- an
  artifact of pooling GoT rows without desubstitution. `qwen3-8b` scores 0.996
  and `qwen3-14b`/`gemma4-*` score 1.000. The task is saturated, not broken.
- ~~"`connected_nodes` is the only uncontaminated task with headroom; target
  it"~~ -- it has no headroom. `edge_count` is the target.
- ~~"qwen3-8b connected_nodes +0.053 at n=500, p<0.0001"~~ and ~~"qwen3-8b-think
  connected_nodes + rwse +0.149"~~ -- both GoT artifacts; gone after exclusion.
- The `sweep-findings.md:368` note about "resolving connected as adjacency"
  concerns **`edge_existence`**, not `connected_nodes`. Mis-cited here initially.

## The new small models

Probe: `prompts.count100.none_degree.jsonl`, 6 tasks x 100 graphs x
{none, degree}, 1,200 rows per arm, tag `probe100`.

`qwen3-0.6b` plain (95.5 min generation, 0.2% unparsed, 3 capped):

| task | none | degree | delta | disc | p |
|---|---|---|---|---|---|
| node_count | 0.900 | 0.550 | -0.350 | 39 | 0.0000 |
| edge_count | 0.070 | 0.090 | +0.020 | 14 | 0.79 |
| node_degree | 0.890 | 0.810 | -0.080 | 22 | 0.13 |
| connected_nodes | 0.834 | 0.875 | +0.041 | 8 | 1.00 |
| edge_existence | 0.680 | 0.620 | -0.060 | 10 | 0.11 |
| cycle_check | 0.856 | 0.814 | -0.041 | 12 | 0.39 |

pooled none 0.705 / degree 0.625 (**-0.079**); excluding the `node_count`
artifact below, **-0.025**.

`qwen3-0.6b` think (6 shards, 3.0% capped): pooled none 0.843 / degree 0.920
(**+0.078**), driven by `edge_count` 0.324 -> 0.873.

**The same model, same prompts, same instances: the primer hurts by 7.9 points
without the reasoning channel and helps by 7.8 with it.**

**Retracted 2026-09-06 -- this does not replicate.** `qwen3-1.7b` runs +9.6 pp
plain and +1.3 pp think: same sign in both arms. The 0.6B's *negative* plain
number was the off-by-one artifact below, not a property of small models, so the
"sign flip" was an artifact comparison. Do not repeat the claim.

### The off-by-one artifact (do not report as a primer effect)

`qwen3-0.6b`'s -0.350 on `node_count` is not a reasoning failure. Of 100
`degree` rows: 55 exact, 45 off by exactly -1, **zero other error types**;
corrected for the convention it scores 1.000. The model identifies the node set
correctly every time and reports the highest node *label* (N-1) instead of |V|.

Confirmed two ways:

- The think arm verbalises the missing step on rows it gets right: *"The nodes
  are numbered from 0 to 17 ... so that's 18 nodes in total."*
- The rate scales with size: **0/31 (0%) below 10 nodes, 19/41 (46%) at 10-15,
  26/28 (93%) at 16+**. A pure labelling confusion would be flat; this is a
  counting-capacity limit, with the primer's per-node enumeration supplying the
  salient wrong answer.

**The existing sweep is not contaminated**: off-by-one rate on `node_count` is
0% for `gemma4-12b`, `gemma4-e4b`, `qwen3-8b` (560 rows) and `qwen3-14b`;
2-3% for `gemma4-e4b-think`. Same genre as the three cases in
`sweep-findings.md` where an artifact impersonated a finding.

`qwen3-0.6b` is genuinely too weak on `edge_count` in the plain arm (0.070/0.090,
diffuse errors), though its think arm reaches 0.324/0.873 on the same task.

### `qwen3-1.7b` clears the artifact and is the better suite member

Complete at 1,200 rows (0.4% capped, 0.3% unparsed):

| task | none | degree | delta | disc | p | bar |
|---|---|---|---|---|---|---|
| node_count | 0.950 | 0.960 | +0.010 | 7 | 1.000 | 1.00 |
| edge_count | **0.237** | 0.691 | +0.454 | 48 | 0.0000 | 1.00 |
| node_degree | 0.890 | 0.960 | +0.070 | 9 | 0.039 | 1.00 |
| connected_nodes | 0.963 | 0.984 | +0.021 | 3 | 1.000 | 0.21 |
| edge_existence | 0.890 | 0.890 | +0.000 | 8 | 1.000 | 0.79 |
| cycle_check | 0.888 | 0.918 | +0.031 | 5 | 0.375 | 0.95 |

pooled none 0.804 / degree 0.900 (**+0.096**).

**The off-by-one rate drops to 3% under `degree` (from 45%) and 5% under `none`
(from 10%).** This is direct confirmation of the counting-capacity explanation
above: 1.7B clears the limit the 0.6B falls off past ~15 nodes, and `node_count`
becomes an interpretable cell (+0.010) instead of an artifact reported as a
-0.350 primer effect.

**`qwen3-1.7b` has more `edge_count` headroom than `qwen3-8b`** -- 0.237 against
0.433 -- with a clean parse rate and no artifact. On the one task that is not
saturated it is the weakest baseline in the suite, which makes it the strongest
candidate for the n=500 clean-condition experiment, and roughly 4x cheaper per
row than `qwen3-8b`. Not yet submitted: the value of adding an arm depends on
whether the clean conditions produce a measurable effect at all, which
`ec500` will answer first.

## Does size break them? Yes -- but only on one task

Nothing in the tracked corpus could answer this: the published split and the
vendored generator both cap node counts at 19
(`graph_generators._NUMBER_OF_NODES_RANGE`). `scripts/build_size_sweep.py`
generates its own ER graphs at chosen sizes keeping the corpus's U(0, 1)
sparsity, so a size class differs from the tracked corpus in size and nothing
else.

**Design.** Sizes 20 / 40 / 80 x 50 graphs x 4 tasks x `none`, on `qwen3-1.7b`
and `qwen3-8b`, both arms each (`prompts.sizesweep.jsonl`, tag `size`, 600 rows
per arm, all four complete).

**Why it stops at 80.** Under U(0, 1) sparsity edges grow as O(n^2) and the
`incident` encoding lists every one. Measured prompt tokens (both models have a
40,960-token context):

| n | p10 | median | p90 | max |
|---|---|---|---|---|
| 20 | 723 | 948 | 1,242 | 1,403 |
| 40 | 2,480 | 3,367 | 4,932 | 5,538 |
| 80 | 9,503 | 13,379 | 19,763 | 22,093 |
| 160 | 40,445 | 57,217 | 87,413 | 95,852 |

n=160 is not a budgeting problem, it is impossible: even the 10th-percentile
graph exceeds the context. Going bigger needs a fixed average degree instead
(edges linear in n, ~13k tokens at n=320) -- a different density regime, so a
different experiment rather than a longer version of this one.

`edge_count` and `cycle_check` are excluded: the first would measure truncation
(hundreds of edges to enumerate against a 2,048-token budget), the second
degenerates (every graph this dense has a cycle).

### Pooled accuracy

| arm | n=20 | n=40 | n=80 | capped |
|---|---|---|---|---|
| 1.7B plain | 0.874 | 0.679 | 0.631 | 1/600 |
| 1.7B think | 0.994 | 0.878 | 0.731 | 21/600 |
| 8B plain | 0.985 | 0.985 | 0.847 | 2/600 |
| 8B think | 0.979 | 0.983 | 0.926 | 38/600 |

Degradation is monotonic in all four arms, and capacity buys graceful decline
rather than immunity: 8B think loses 5 pp from n=20 to n=80, 1.7B plain loses
24 pp.

### The pooled number is misleading -- it is one task

| task | 1.7B plain | 1.7B think | 8B plain | 8B think |
|---|---|---|---|---|
| node_count | 0.820 / 0.600 / 0.860 | 1.000 / 0.978 / **1.000** | 0.980 / 1.000 / 0.980 | 1.000 / 1.000 / **1.000** |
| **node_degree** | 0.860 / 0.440 / **0.143** | 0.980 / 0.580 / **0.205** | 1.000 / 0.940 / **0.479** | 0.977 / 0.977 / **0.727** |
| connected_nodes | 0.937 / 0.975 / 0.870 | 0.998 / 0.965 / 0.834 | 0.980 / 1.000 / 0.935 | 0.928 / 0.953 / 0.976 |
| edge_existence | 0.880 / 0.700 / 0.640 | 1.000 / 1.000 / 0.837 | 0.980 / 1.000 / 0.980 | 1.000 / 1.000 / 0.980 |

(cells are n=20 / n=40 / n=80)

**`node_count` does not degrade at all** -- 1.000 at every size for both think
arms. Counting 80 nodes is trivial because the node list is contiguous in the
prompt. This also shows the off-by-one artifact does not reappear at scale for
these models.

**`node_degree` collapses**, to 0.143 on the 1.7B plain arm -- worse than
1-in-5. The error profile confirms genuine miscounting rather than a parse
failure or a refusal (exact answers / median absolute error over non-exact):

| arm | n=20 | n=40 | n=80 |
|---|---|---|---|
| 1.7B plain | 43/50, m=1 | 22/50, m=2 | 7/49, **m=8** |
| 1.7B think | 49/50, m=1 | 29/50, m=1 | 9/44, **m=6** |
| 8B plain | 50/50, m=0 | 47/50, m=1 | 23/48, **m=4** |
| 8B think | 42/43, m=1 | 43/44, m=1 | 32/44, m=1 |

Errors grow from off-by-one to a median of 6-8 (max observed 79). A
representative 1.7B-think failure: gold 72, answered 66, after 3,282 tokens of
reasoning.

### What actually predicts degradation

`node_degree` is the only one of the four tasks whose work scales with **edges**
rather than nodes: answering it means counting every occurrence of one node
across ~1,958 edge mentions scattered through a 20k-token prompt. The other
three read a contiguous region (`node_count`), do one local lookup
(`connected_nodes`), or run a single membership test (`edge_existence`).

So the finding is not "models degrade on large graphs". It is **"models degrade
on tasks that require aggregating many scattered mentions, and graph size is
what multiplies the mentions."** Size is the independent variable; dispersed
aggregation is the mechanism.

Note `8B think` is the exception that supports this: it holds a median error of
1 at every size and only drops to 0.727, i.e. the reasoning channel is being
spent on exactly the bookkeeping the task needs.

## Where this stands, and what to do next

State as of 2026-09-06. Branch `small-model-suite-and-primer-power`, pushed to
`git@github.com:GalBrk/GraphTalk.git` (the remote moved from `ArnavShahor/`;
other clones still need `git remote set-url`).

### Data on disk, all complete

| tag | arms | rows/arm | what |
|---|---|---|---|
| `probe100` | qwen3-0.6b, -think, qwen3-1.7b, -think, qwen35-2b | 1,200 | 6 tasks x 100 graphs x {none, degree} |
| `cc500` | qwen3-0.6b, qwen3-0.6b-think | 2,000 | cycle_check x 500 x {none, components, clustering, rwse} |
| `ec500` | qwen3-8b | 1,984 | edge_count, same 4 conditions (16 rows short: one shard timed out) |
| `ec500` | qwen3-1.7b | 2,000 | the replication of the above; job 858244, complete 2026-09-07 |
| `density40` | qwen3-1.7b | 2,400 | n=40 x 6 pinned ER densities x {node_degree, connected_nodes} x {none, degree}; job 858671 |
| `size` | qwen3-1.7b, -think, qwen3-8b, -think | 600 | 20/40/80-node graphs x 4 tasks x none |

### Nothing is in flight

Jobs 858244 (`ec17-clean`) and 858671 (`dens40-q17b`) both completed on
2026-09-07 and are written up above and below respectively.

### Density at a fixed size

**What it tests.** `docs/difficulty-scaling.md` and the driver analysis in
`docs/plans/scale-vs-topology-investigation.md` claim the `degree` primer's
benefit on `edge_count` "grows monotonically and substantially with graph size,
degree-sequence variance, and density". That claim is observational, measured
over 5-19 node graphs where all three co-vary, and the same document names a
Simpson's-paradox confound in its own naive version. This run is the controlled
form of it: **size held fixed at n=40, density pinned per level**, so density
moves alone.

**Design.** n=40 x 6 density levels x 100 graphs x {`node_degree`,
`connected_nodes`} x {`none`, `degree`} = 2,400 prompts, `qwen3-1.7b`.

| p | mean degree | edges (med) | prompt chars (med / max) | `node_degree` majority baseline |
|---|---|---|---|---|
| 0.05 | 2.0 | 39 | 2,099 / 2,704 | 0.300 |
| 0.10 | 3.9 | 79 | 2,589 / 3,050 | 0.210 |
| 0.20 | 7.8 | 155 | 3,252 / 3,703 | 0.170 |
| 0.35 | 13.6 | 273 | 4,111 / 4,662 | 0.170 |
| 0.50 | 19.5 | 390 | 5,000 / 5,542 | 0.180 |
| 0.75 | 29.2 | 585 | 6,542 / 6,962 | 0.160 |

Every cell keeps real headroom (majority baseline 0.16-0.30), which is the
check that stops a level from being another `reachability`.

**Why the bounds are where they are.**

- **p = 1.00 is excluded** -- that is the complete graph. Every node has degree
  39, so `node_degree`'s majority baseline is **1.000** and answering "39"
  blind scores perfectly. Worth noting the tracked corpus's own `U(0, 1)`
  policy includes this degenerate endpoint as a rare draw.
- **p = 0.05 is the floor.** Below the connectivity threshold (~0.09 at n=40)
  about 16% of `connected_nodes` gold sets are empty. `scoring.set_f1` handles
  that deliberately and correctly (both-empty scores 1.0, since "No nodes" is
  the right answer for an isolated node), but an empty answer is near-free
  accuracy, so going sparser donates points rather than measuring anything.
- **Two tasks, not six.** `edge_count` is out for the same reason the size
  sweep excluded it: 585 edges at p=0.75 against a 2,048-token budget measures
  truncation, not ability. `cycle_check` degenerates -- every graph past the
  sparsest level has a cycle. `node_count` is already flat at 1.000 across
  every size measured, and density does not change the node list.
- **ER-only, built here rather than via `--graph-source diverse`.**
  `er_min_sparsity`/`er_max_sparsity` reach only the `er` algorithm; the other
  six in `diverse_corpus.ALGORITHMS` have no sparsity parameter at all
  (`complete`/`star`/`path` are at the density extremes by construction). At
  the `--count 30` that `difficulty-scaling.md` recommends, the knob would
  touch 5 graphs out of 30. `build_size_sweep.py` generates ER directly, so
  100% of the corpus receives the manipulation.
- **Levels are pinned, not sampled.** `--densities` sets
  `er_min == er_max`, and `random.uniform(p, p) == p`, so each level is an
  exact density rather than another `U`-draw. This is what makes density an
  experimental variable instead of corpus noise.

**Result (job 858671, 2,400/2,400 rows, 1 capped, 100% parsed).**

**First, a design error, because it changes how half of this reads.**
`node_degree` was the wrong task to pair with the `degree` primer. That primer
renders one sentence per node -- literally `"Node X has degree Y."` -- so for
`node_degree` it **states the answer verbatim**. Checked, not assumed:
600/600 `degree`-condition rows contain `"Node <queried> has degree <gold>."`.
That cell therefore has a shortcut ceiling of 1.0 by construction and cannot
speak to whether a primer aids reasoning. It is exactly the "(condition, task)
pairs that let the primer answer the task directly" confound that
`docs/plans/run_improved_tests.md` Phase 1 was written to catch, and it was
walked into anyway -- and worse, it was *readable off the bar table in this very
document*, which lists `node_degree`/`degree` at **1.00**. It did not need to be
discovered empirically afterwards.

`connected_nodes` is uncontaminated -- the `degree` primer gives the neighbour
*count*, not the identities -- but it was a poor choice for a different reason:
it had no headroom. The size sweep above already measured `qwen3-1.7b` on
`connected_nodes` at 0.937 / 0.975 / **0.870** for n=20/40/80, so at n=40 this
model was known to be at 0.975 before the job was submitted.

Both tasks were therefore disqualified in advance by numbers already in this
file. They were picked by elimination from the size sweep's exclusions
(`edge_count` for truncation, `cycle_check` for degeneracy, `node_count` for
being flat) rather than by the criterion this document states two sections
earlier: cross the shortcut bars against headroom, and "only one row survives:
`edge_count`".

**The clean cell: `connected_nodes` (set F1).**

| p | none | degree | gap |
|---|---|---|---|
| 0.05 | 0.999 | 0.909 | **-0.090** |
| 0.10 | 0.979 | 0.975 | -0.004 |
| 0.20 | 0.984 | 0.992 | +0.009 |
| 0.35 | 0.969 | 0.978 | +0.009 |
| 0.50 | 0.960 | 0.969 | +0.009 |
| 0.75 | 0.947 | 0.945 | -0.002 |

Read across all six levels this looks flat, and an earlier version of this
section called it a non-reproduction of the driver analysis's claim. **That was
wrong, and the reason is a range mismatch.**

The driver analysis binned `nx.density` into terciles over the tracked corpus,
which is 5-19 nodes. Density is a *ratio*, so the same density means very
different absolute edge counts at n=12 and n=40 -- and absolute edges is what
our own size-sweep finding says drives difficulty (aggregation over scattered
mentions). Measured against that corpus:

| this run's level | edges | share of the tracked corpus at or below it |
|---|---|---|
| p=0.05 | 38 | 64% |
| p=0.10 | 77 | 85% |
| p=0.20 | 156 | **99.5%** |
| p=0.75 | 583 | 100% (3.4x its maximum of 170) |

The claim being tested therefore lives almost entirely inside this run's two
sparsest levels. Restricted to that overlap the gap **rises** monotonically --
`connected_nodes` -0.090 -> -0.004 -> +0.009, and `node_degree` -0.030 ->
+0.030 -> +0.110 -- and only flattens beyond where the original corpus ends.

So the correct statement is: **this run does not refute the density claim, and
within the claim's own range it is directionally consistent with it.** The
caveats stay: the rise is tiny in absolute terms, `none` sits at 0.95-1.0 so
the cell is near ceiling, `node_degree` is shortcut-explained, and three points
is not a trend. Underpowered either way, but not a null.

Also note the design flaw this exposes: three of the six levels sit in the
tracked corpus's *low* density tercile and only one in its *high* tercile,
which is where the reported effect was strongest. The level spacing was chosen
because sparse graphs are cheap and non-degenerate, which was the wrong
criterion for testing this particular claim.

The one clean signal that survives all of this is the **-0.090 at the sparsest
level**: on sparse graphs the `degree` primer *hurts*, the same direction as
`rwse`.

**The shortcut-explained cell, read for what it does measure.** `node_degree`
cannot test the primer question, but it does measure retrieval of a stated fact
as the prompt grows:

| p | none | degree (answer present verbatim) |
|---|---|---|
| 0.05 | 0.980 | 0.950 |
| 0.10 | 0.910 | 0.940 |
| 0.20 | 0.810 | 0.920 |
| 0.35 | 0.430 | 0.630 |
| 0.50 | 0.310 | 0.310 |
| 0.75 | 0.120 | 0.140 |

At p=0.75 the prompt contains the sentence `"Node 12 has degree 30."` and the
model is still wrong 86% of the time; having the answer written down is worth
+0.02. The gap is an inverted U -- peaking at +0.200 (p=0.35) and gone by
p=0.50 -- which is a floor effect, not a density effect: past a certain prompt
length the model cannot retrieve a stated fact at all, so there is nothing left
for the primer to add. Note this is *not* the same claim as "size breaks them"
from the size sweep, which held density at U(0, 1) and varied n; here n is
fixed at 40 and only the edge count moves.

**What would actually settle the density question -- corrected.** An earlier
version of this section proposed `edge_count` with the `degree` primer, on the
reasoning that degrees sum to twice the edge count so the primer aids
computation without stating the answer. That reasoning is exactly why it does
not work: the same identity makes its bar **1.00** in the table above. It would
have been shortcut-explained in the same way `node_degree` was.

Read the bar table properly and the `degree` primer is shortcut-explainable on
nearly every task (1.00 on `node_count`, `edge_count` and `node_degree`; 0.79 on
`edge_existence`; 0.95 on `cycle_check`). The informative conditions are
`components`, `clustering` and `rwse` on `edge_count`, at bars 0.02 / 0.15 /
0.02 -- which is the `ec500` design, and why that cell is this project's
flagship.

**The density sweep worth running is `ec500`'s conditions across density
levels**, not the `degree` primer on any task. Truncation is the only real
obstacle, and it is smaller than assumed: outputs here peaked at 777 tokens
against a 2,048-token budget. Weight the levels toward the dense end this time,
and keep at least two inside the tracked corpus's range so the result can be
compared with the driver analysis rather than talking past it.

### Deliberately not run

- **`--xlarge` (20-39 nodes) from `docs/difficulty-scaling.md`** -- subsumed.
  The size sweep above already covers 20/40/80 at 50 graphs per class, so the
  bucket sits strictly inside measured ground at lower resolution.
- **`reachability`** -- degenerate on this corpus. Five of the seven generator
  families are connected by construction, `sbm` almost always is, and `er` is
  once density is raised; under `difficulty-scaling.md`'s own recommended
  command the gold answer is "Yes" **210 times out of 210**, so answering "Yes"
  blind scores 1.000. It also has no shortcut bar -- `shortcuts.py` keeps its
  own task list and was not extended -- so an effect there would be
  uninterpretable by this project's own standard even if the golds were
  balanced. Reviving it needs disconnected graphs in the pool (a forest family,
  or SBM with near-zero inter-community probability) plus a solver entry.
- **The input-overflow guard** -- `ModelSpec.max_context_tokens` is `None` for
  every model, so the check `hf_backend.py` performs is a no-op. Not a blocker
  here: the densest prompt in this run is 6,962 characters against a
  40,960-token context.
- **`qwen3-14b` on `ec500`** -- cancelled at 206/2,000 after its shard 0 spent
  ~10 of 12 h in the page-cache warm-up. Partial rows are in
  `runs/archive/cancelled-*`. `qwen3-1.7b` replaces it as the replication.
- **`ec8b`'s missing 16 rows** -- shard 0 timed out. `run_sweep.py` resumes by
  key, so resubmitting that one shard would collect only the gap. Not done:
  1,984/2,000 with balanced conditions changes nothing.
- **`qwen35-2b-think` `probe100`** -- cancelled at 73/1,200; it could only have
  confirmed a `degree` result that is shortcut-explained anyway.
- **Size sweep beyond n=80** -- impossible under U(0, 1) sparsity; needs a
  fixed-degree density regime, i.e. a different experiment.
- **`cc500` on `qwen3-1.7b`** -- the `cycle_check` finding is also single-model.
  Its plain `cycle_check` baseline is 0.888 against a 0.832 floor, so there is
  less headroom than the 0.6B had, but it is the natural second data point.

### If you only read one thing

Primers are not one intervention. `components` is inert on both models that
have tested it. `rwse` does real damage on `qwen3-8b` (-13.7 pp) and directional
but non-significant damage on `qwen3-1.7b` (-4.0 pp). `clustering` helped on two
cells and then **failed to replicate on a third** -- and it sits below its own
shortcut bar in every cell measured, so it was never as clean as it looked.

Two rules, both learned the hard way:

1. **Read every result against `bar(cond) - bar(none)` from `shortcuts.json`,
   not against zero**, and check whether the primer simply contains the answer
   before running anything -- the `density40` run lost half its design to
   exactly that (`degree` primer + `node_degree` task).
2. **Filter `hit_cap` rows before comparing anything**, but say so, because it
   is not neutral: on `ec500`/1.7B it is the difference between `rwse` at
   -4.0 pp (p=0.11) and -5.0 pp (p=0.025). Report the non-termination rate
   alongside accuracy rather than letting it hide inside it.

## Operational notes

- **`--mem=64G` in `sweep.sbatch` is sized for Qwen3-14B's 29.6 GB checkpoint.**
  For a 1.2 GB model it cut eligible nodes from 5 to 1 and cost ~45 min of queue
  time. Right-size per model (`--mem=16G --cpus-per-task=4` worked).
- **n-501 has a 535.x driver (CUDA 12.2); the default cu130 env cannot use it.**
  `sweep.sbatch`'s comments name only n-802/803/804. Three shards died there in
  94 s (the driver guard working). Use `GRAPHTALK_ENV=graphtalk-cu126` or
  `--exclude=n-501`.
- **Use an ODD `--array` shard count.** With 2 conditions the stride preserves
  parity, so an even count sends `none` to even shards and `degree` to odd ones;
  partial progress is then unpaired and mid-run comparisons are meaningless.
- **`--array=1,2,4` does NOT mean "shards 1, 2 and 4 of 5".**
  `SLURM_ARRAY_TASK_COUNT` is the number of tasks (3), so `sweep.sbatch` computes
  `NSHARDS=3` and each task strides `records[i::3]`. Resubmitting failed shards
  that way produced 28 wrongly-strided rows, 12 of them duplicating rows owned by
  the surviving `*of5` shards. Pass `SLURM_ARRAY_TASK_ID`/`SLURM_ARRAY_TASK_COUNT`
  explicitly on separate non-array jobs instead; see `cluster/README.md`.
- **Page-cache warm-up cost ~44 min for a 1.2 GB checkpoint on n-602** vs ~9 min
  on n-502. It exists for gemma4-e4b's 16 GB checkpoint over NFS; at this size
  it plausibly costs more than it saves.
- **Qwen3.5 needs `AutoModelForImageTextToText`** -- declares
  `Qwen3_5ForConditionalGeneration`, multimodal even at 2B, like Gemma 4.
  Supported by the env's `transformers` 5.15.0.

## Caveats

- The `--count 100` probe corpus is a strict superset of the tracked `--count
  30` corpus (same split, first-N prefix), so instance ids are comparable, but
  it covers only `none` and `degree`.
- p-values are per (model, task) pooled over the instances shown, not the
  proposal's 36-cell grid. Discordance *rates* compare across documents; cell
  counts do not.
- All five `probe100` arms are complete at 1,200 rows: `qwen3-0.6b`,
  `qwen3-0.6b-think`, `qwen3-1.7b`, `qwen3-1.7b-think`, `qwen35-2b`.
  `qwen3-1.7b-think` pooled +1.3 pp (none 0.965 / degree 0.978) with every cell
  0.92-0.99 and 1-7 discordant pairs -- reasoning removes the headroom at 1.7B,
  so its think arm measures nothing. Off-by-one on `node_count` is 5%/3% plain
  and 1%/1% think, against the 0.6B's 10%/45%.
  **`qwen35-2b-think` was cancelled at 73/1200 rows** -- it could only have
  confirmed a `degree`-condition result that is shortcut-explained anyway, and
  the plain-vs-think contrast it would have tested is covered by
  `qwen3-1.7b-think` at a size where the off-by-one artifact is absent. Its
  partial rows are in `runs/archive/cancelled-*` so they cannot reach a frame.
- The `edge_count` n=500 experiment (`ec500`) runs as jobs 853382+853499
  (`qwen3-8b`, 5 shards) and 853383 (`qwen3-14b`, 5 shards), all pinned to
  `--constraint=a6000` after three separate 535.x-driver failures.
- All GoT-named runs are excluded throughout. Re-including them requires
  desubstitution and would mainly affect `connected_nodes`.
