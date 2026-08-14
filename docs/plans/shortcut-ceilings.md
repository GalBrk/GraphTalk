# Shortcut ceilings

## Context

This plan depends on `docs/plans/primer-computation.md` and cannot start before it: it
consumes rendered primer text, and that text must be stable and correct first.

The primer plan's taxonomy audit found deterministic routes from primer text into every
one of the six tasks for the `degree` and `all` conditions, and into `node_count` and
`connected_nodes` for `components`. The sharpest single case: on `edge_existence`, the
rule "answer Yes iff d_a + d_b > n−1" — one comparison between two numbers the primer
states — scores 79.2% ± 1.7, against a 53.0% majority baseline and the 45.1% the paper
reports for PaLM on ER `edge_existence`.

That breaks the question the proposal set out to ask. "Does adding the primer improve
accuracy?" has a trivially affirmative answer wherever the primer contains the answer, and
a gain there supports nothing about graph reasoning — only that models can read.

Three ways to respond, two of which fail:

- **Remove the information.** Suppressing the degree-0 sentence makes the *absence* of a
  sentence the tell, and makes primer coverage a function of degree. Worse, not better.
- **Sample around it.** Restricting `edge_existence` to pairs with `1 ≤ degree ≤ n−2` does
  work and is distribution-neutral (density 0.502 → 0.501, mean components 2.12 → 1.99).
  But it only patches the cells we happen to have found, and at 30 rows per task the
  excluded stratum is too small to report.
- **Measure it and use it as the bar.** This plan.

## What a shortcut score is

A **primer-only solver** is a small deterministic program that reads the rendered primer
text and nothing else, and emits an answer in the task's answer format. Score it with the
same scorer used for model output. That number is the **shortcut score** for that
(condition, task) cell.

It is not a model, and there is no learning in any interesting sense. For `cycle_check`
under the `components` condition the entire solver is a lookup table:

```
   c | graphs | had a cycle | no cycle | so the rule answers
   1 |   2938 |        2906 |       32 | Yes
   2 |    275 |         215 |       60 | Yes
   3 |    173 |          90 |       83 | Yes
   4 |    124 |          40 |       84 | No
```

Read c from the primer, look up the row, output that answer.

## Why this construct avoids a judgement call

The primer plan had to argue about **depth** — whether a primer route makes an answer
*shallower* than the encoding already makes it. That argument is necessary for deciding
whether something counts as a leak, and it is genuinely contestable. Is summing fifteen
stated degrees shallower than counting lines in an edge list? Reasonable people differ,
and roughly half the audit's candidate routes died on exactly that question.

The shortcut score does not need the depth criterion at all. It asks a mechanical
question: **can a program that never reads the graph produce the right answer?** That has
an objective yes/no answer and a measurable rate.

Consequence: the audit's 34 unverified routes do not need re-verification for this
purpose. They need implementing and measuring, which is what this plan does.

## The three rungs

What the solver is allowed to read beyond the primer matters, because two numbers from
the encoding are not equally cheap:

| rung | granted | justification |
|---|---|---|
| 1 | primer text only | the strict version |
| 2 | + n | the encoding's first line always ends `and <n-1>.`, verified 500/500. One token. |
| 3 | + n and m | m requires counting neighbour-mentions in the body and halving, because the incident encoding prints every edge twice. Real work. |

### The ladder does not separate on every arm

Measured once the parser existed, and not anticipated when the rungs were written: **a
node-level primer emits one sentence per node, so counting sentences recovers n at rung
1.** That applies to `degree`, `clustering`, `rwse`, `filler` and `all`. For the `degree`
arm m comes free at rung 1 too, since `sum(degrees) / 2 = m` is one of the theorem rules
below — verified on a 14-node graph: 14 sentences give n, and the stated degrees sum to
2·9 for m = 9.

| condition | what actually separates |
|---|---|
| `degree`, `all` | nothing — n and m are both rung-1, so all three rungs coincide |
| `clustering`, `rwse`, `filler` | rung 1 = rung 2 (n is free); rung 3 still adds m |
| `components`, `none` | all three rungs distinct, as designed |

This is a property of the ladder as a measuring instrument, not a leak, and it is not
worth redesigning the primer over. n was never secret: the encoding's first line reads
`G describes a graph among nodes 0, 1, ..., and 13.`, which is the same fact and the
justification for granting it at rung 2 in the first place. Nor would repackaging help —
combining the three features into one sentence per node is exactly what `all` already
does, and any primer that states a fact about every node reveals how many nodes there
are regardless of how the sentences are cut. Suppressing per-node sentences to avoid it
was already rejected under "Remove the information" above.

The consequence is for reading the table, not for building it: **rung 1 must not be
described as "the solver does not know n."** Report the collapse per arm. It also
sharpens why `components` is the arm the ladder was built for — it is the only one where
the three rungs measure three different things.

Report all three per cell. The ladder is the most informative part of the design, because
the `components` condition is *built* to require rung 3 — circuit rank `m − n + c > 0` is
its entire justification. Measured on 500 graphs:

```
cycle_check under the components primer
  1. majority baseline (always Yes)                 : 83.2%
  2. rung 1, fitted lookup on c alone               : 93.2%
  3. rung 3, m - n + c > 0 (the intended route)     : 100.0%
```

So a model landing near 93% read the component count and stopped; a model reaching 100%
counted the edges and did the arithmetic. That distinguishes *which step failed*, which no
single accuracy number can.

It also shows the components condition is not the clean arm it was assumed to be: rung 1
already beats the baseline by ten points without any of the intended reasoning.

## The four regimes

Three numbers per cell — majority baseline, shortcut score, model accuracy — give an
interpretation that the original two-number design could not:

| where the model lands | reading |
|---|---|
| below the majority baseline | the primer is harming it |
| between baseline and shortcut | partial use of the stated facts |
| at the shortcut score | doing the primer arithmetic and nothing more |
| **above the shortcut score** | genuinely combined primer with graph |

Only the last regime supports the proposal's thesis. The shortcut score is therefore not
a caveat to report alongside the result — it is the threshold the hypothesis has to clear.

This matters for how a null gets read, too. If the primer hands over a rule worth 77% and
the model scores 50%, that is a sharp positive finding in its own right: models fail to
exploit explicitly stated structural facts even when doing so is trivially sufficient.
That is a stronger claim about black-box structural reasoning than "the primer helped by
four points."

## Rule taxonomy, and the methodology each kind needs

Three kinds, reported separately. Mixing them produces a number that means nothing.

**Theorems.** Always correct. Report **coverage** — the fraction of rows where the rule
applies — and note that precision is 1 by construction. No train/test split; there is
nothing fitted and nothing to overfit.

- `degree = 0` → no edge, and → `" No nodes."` for `connected_nodes`
- `degree = n−1` → edge exists
- all-zero RWSE vector ⟺ degree 0 (under the primer plan's clamp convention)
- `sum(degrees) / 2 = m`
- `clustering > 0` → a triangle exists → a cycle exists
- `RWSE(k=3) > 0` → the same triangle test
- `m − n + c > 0` ⟺ a cycle exists
- `c = n` ⟺ `m = 0`
- `m ≥ n` → a cycle exists (a forest has at most n−1 edges)

**Parameter-free heuristics.** Can be wrong, but nothing was fitted, so no split is
needed. Report accuracy on the test rows.

- `d_a + d_b > n−1` → Yes for `edge_existence` (79.2% ± 1.7 over 200 query resamples)

**Fitted rules.** Contain numbers or lookup tables derived from data. These **must** be
fitted on a disjoint set of graphs — a different generator seed — and evaluated on the
test-split rows.

- `c` → Yes/No lookup for `cycle_check`
- `max degree + 1 = n` for `node_count`
- density estimators from mean clustering or mean RWSE, for `edge_existence`
- stationary inversion `d_i ≈ 2m · RWSE_k` for `node_degree`

Why the split is not pedantry, measured on the `cycle_check` lookup:

```
rule keyed on c only          (19 table rows)
  fit on the SAME 500, scored on those 500 : 94.4%
  fit on 4000 others, scored on those 500  : 93.2%     inflation +1.2 points

rule keyed on (n, c)          (169 table rows)
  fit on the SAME 500, scored on those 500 : 97.6%
  fit on 4000 others, scored on those 500  : 93.6%     inflation +4.0 points
```

More table rows relative to the data means more memorising means more inflation. And the
direction of the error is the dangerous one: an inflated shortcut score makes the model
look like it *failed to reach* a bar that was never really there, reversing the sign of
the finding. The whole design rests on this one comparison, so the bar has to be honest.

## The solver reads text, not numbers

The solver's signature takes the **rendered primer string**, not statistics:

```python
def solve(primer_text: str, task: str, targets: list[int],
          n: int | None = None, m: int | None = None) -> str:
```

Two guarantees fall out of that signature rather than out of discipline:

- **It can only use the two-decimal values the model actually sees.** A rule reading
  full-precision RWSE could beat one reading `0.02`, and would measure a bar nobody was
  ever shown.
- **It cannot touch the graph.** There is no graph parameter to touch.

This is the same move as the primer plan's single renderer: make the property structural
so there is no way to violate it by forgetting.

Cost is a small parser that reads numbers back out of the primer text. That parser earns
its keep as a **round-trip test**: render, then parse, and the recovered values must equal
the rounded originals exactly. That test checks the renderer and the parser against each
other, and it is also the reason the primer plan's `_fmt` fix must land first — unstable
digits would make both the parse and the bar unstable.

## The score is a floor, with one exact island

The shortcut score covers only the rules we thought of, so it is a **lower bound** on
primer-only performance. Someone finds a better rule and the true bar is higher. Report it
as "at least this good", and list the rules used, so a reader can see what was tried.

One exception makes this checkable rather than merely hedged. For n ≤ 6, enumerate every
labelled graph consistent with the degree sequence the primer states, and compute the
**exact** information-theoretic bound — the fraction of queries whose answer is identical
across all consistent graphs. Measured in the audit: degrees alone determine 80.8% of
pairs at n=5 and 53.9% at n=6, and n ≤ 6 is about 9% of rows.

Use that island as a cross-check. If the heuristic ceiling on small graphs sits far below
the exact bound, there are rules we have not found, and the floor is loose.

## Scope

7 conditions × 6 tasks = 42 cells, minus the 6 `none` cells whose shortcut score is the
majority baseline by definition. **36 cells.**

Compute all of them. There are no model queries involved — it is plain Python over graphs
we already generate, seconds of CPU. The temptation is to compute only the cells that look
interesting, but the cheap ones are exactly the ones that justify skipping expensive work.

Then use the table to triage the sweep:

- Cells where the shortcut already scores ~100% (`degree` × `node_degree`, `degree` ×
  `edge_count`) tell you nothing a model run would add. The comparison is decided before
  you start. Skip them, or run them only as manipulation checks.
- Cells with real headroom between baseline and shortcut are where cluster time buys
  information.

That inverts the cost problem: the table is not extra work on top of the experiment, it is
what tells you which parts of the experiment not to run.

## Measured results

Computed by `scripts/shortcut_table.py --graphs 500`, fitted on seed 999 and scored on
seed 1234. Shortcut score at rung 3, against the majority baseline.

| task | baseline | best arm | shortcut | verdict |
|---|---|---|---|---|
| `node_count` | 6.4% | every node-level arm | **100%** | decided |
| `edge_count` | 1.8% | `degree`, `all` | **100%** | decided |
| `node_degree` | 8.2% | `degree`, `all` | **100%** | decided (manipulation check) |
| `cycle_check` | 83.2% | `components` | **100%** | decided |
| `edge_existence` | 49.8% | `degree`, `all` | 79.4% | real headroom |
| `connected_nodes` | 8.2% | none of them | 8.2% | wide open |

**Four of the six tasks are already decided by a program that never sees the graph.**
Running a model on those cells cannot inform the hypothesis: `node_count` because the
primer emits one sentence per node, `edge_count` because `sum(degrees)/2 = m`,
`node_degree` because the primer states the answer, and `cycle_check` on the `components`
arm because circuit rank is exact.

Two findings that were not anticipated:

- **`cycle_check` gains nothing on any per-node arm.** `clustering > 0` fires at 80.8%
  coverage and `m >= n` at 78.6%, both at precision 1, but both answer only *Yes* — and
  the majority baseline is already "always Yes" at 83.2%. A one-directional rule cannot
  beat a baseline that agrees with it. Only `components` can answer *No*, which is a
  sharper argument for that arm than the plan originally made.
- **`connected_nodes` has no shortcut at all.** The gold answer is a neighbour list, and a
  primer-only solver can produce nothing but `" No nodes."`, which is already the
  majority answer. Shortcut equals baseline on all seven arms, so every point a model
  scores above 8.2% is genuine. This is the cleanest cell in the design.

The `none` arm equals its baseline exactly at rung 1 on all six tasks, as the sanity check
requires.

### The ladder, and the fitted-rule inflation

`components` x `cycle_check` reproduces the anchor: 83.2% baseline, 92.6% at rung 1,
94.6% at rung 2, 100% at rung 3. It is the only cell where all three rungs separate.

In-sample fitting inflates as predicted, and more table rows inflate more:

| rule | table rows | fitted in-sample | fitted honestly | inflation |
|---|---|---|---|---|
| `c` -> Yes/No | 15 | 94.4% | 92.6% | +1.8pp |
| `(c, n)` -> Yes/No | 95 | 97.6% | 90.6% | +7.0pp |

Both in-sample figures match the values the plan recorded (94.4% and 97.6%). The honest
figures are lower here than the plan's because the fitting set is 500 graphs rather than
4000.

### The exact island says rules remain unfound

On n <= 6 (9.2% of rows), enumerating every labelled graph consistent with the stated
degrees gives the exact ceiling for *any* primer-only solver:

| task | determined | exact ceiling | our best | gap |
|---|---|---|---|---|
| `edge_existence` | 67.4% | 93.5% | 91.3% | tight |
| `cycle_check` | 97.8% | 100% | 65.2% | **35pp unfound** |
| `connected_nodes` | 56.5% | 69.6% | 10.9% | **59pp unfound** |

No rule exceeds its bound, which is the correctness check. But two gaps are large, and
both are diagnosable: the degree sequence determines the component count on small graphs,
so a rule that reconstructs `c` from the stated degrees would feed circuit rank and close
most of the `cycle_check` gap; and the degree sequence often determines the neighbour list
outright, which no rule here attempts. **The floor is loose, and the table should be read
as "at least this good" rather than as the ceiling.**

## Anchors already measured

Seeds for the table, all on `generate_graphs(500, "er", False, random_seed=1234)` with
each task's own query sampling:

| cell | baseline | shortcut | notes |
|---|---|---|---|
| `degree` × `edge_existence` | 53.0% | 27.9% coverage at 100% precision (theorem); 79.2% accuracy (heuristic) | paper reports 45.1% for a model |
| `components` × `cycle_check` | 83.2% | 93.2% at rung 1; 100.0% at rung 3 | the ladder in action |
| `components` × `node_count` | — | states the answer on 1.2% of rows | c = n ⟺ m = 0 |
| `components` × `connected_nodes` | — | excludes `" No nodes."` on 73.6% of rows | c = 1 ⟹ no isolated node |
| `clustering` × `cycle_check` | 83.2% | fires on 80.8% at 100% precision; 97.6% as a rule | clustering > 0 ⟹ triangle |
| `degree` × `connected_nodes` | — | 9.4% coverage at 100% precision | degree 0 ⟹ `" No nodes."` |

Every number here was generator-derived, and has since been checked against the published
rows by `scripts/measure_real_rows.py`. The graph-level ones are exact rather than
approximate: `generate_graphs(500, "er", False, random_seed=1234)` and the published
`zero_shot_test` split are the same multiset of graphs, merely shuffled. Only the rates
that depend on the per-row query draw moved — `edge_existence` from ≈51.6% to 53.0%, and
`connected_nodes` degree-0 coverage from 9.0% to 9.4%. See the provenance section of the
primer plan.

## Files

- `graphtalk/shortcuts.py` — **landed in full**: the primer parser, thirteen theorem
  rules, one parameter-free heuristic, eight fitted rules with the train/test split, the
  solver, and the exact enumeration bound.
- `scripts/shortcut_table.py` — **landed**; computes and prints the 36-cell table at all
  three rungs, with coverage for theorems and accuracy for the rest
- `tests/test_shortcuts.py` — **landed**, 124 tests covering all of the above
- depends on `graphtalk.graphqa.expected_answer` and `normalize` for gold answers, which
  is why the primer plan moves them into the package

## Verification

- **Theorem rules have precision exactly 1.0.** Assert it over a corpus. Any
  counterexample is a bug in the rule or in `expected_answer`, and either way the run
  should fail rather than report a number.
- **Fitted rules cannot see their evaluation set.** Make it structural: the fitting
  function takes a seed and the evaluation function takes a different one, and a test
  asserts the two graph sets are disjoint. A comment is not enough — this is the mistake
  that reverses the sign of the headline finding.
- **The solver cannot read the graph.** Guaranteed by the signature; assert that
  `shortcuts.solve` accepts no graph argument, so the guarantee survives refactoring.
- **Round trip.** Render a primer, parse it back, and assert the recovered values equal
  the rounded originals exactly, on a corpus. This checks renderer and parser against each
  other — **which requires that they share no code.** A first draft validated the RWSE
  step list by calling the renderer's own `_join`, so mutating that function mutated the
  check with it and the two agreed under any change. `shortcuts.py` now imports nothing
  from `primers.py` and restates the join rule independently in `_expected_separators`.
  The parser is also strict about which join style it accepts: tolerating both `a, b and
  c` and `a, b, and c` made a change to the join rule invisible.

  Established by mutation testing rather than by inspection. Seven deliberate renderer
  breakages — decimal places, both join-rule variants, dropped RWSE step labels, wrong
  plurals, a wrong noun, and misaligned node values — are all caught; the first pass
  caught five of seven, which is how both defects above were found. Re-run that check
  after any change to either side.
- **`none` sanity.** The shortcut score for the `none` condition must equal the
  majority-class baseline, since the primer is empty. If it does not, the solver is
  reading something it should not.
- **Exact island cross-check.** For n ≤ 6, the heuristic ceiling must not exceed the exact
  enumeration bound, and a large gap below it is a signal that rules are missing.
- **Rung monotonicity.** Rung 3 ≥ rung 2 ≥ rung 1 for every cell, since each rung grants
  strictly more. A violation means a fitted rule is overfitting or a rung is leaking.

```bash
uv run --no-sync pytest tests/ -q
.venv/bin/python scripts/shortcut_table.py --graphs 500
```

Read the table before committing any cluster time.

## What this does not settle

- **Whether the table is a paper result or internal triage.** It is being built to the
  standard needed for the former — train/test discipline throughout — because that costs
  almost nothing in code and is the difference between a number that can be published and
  one that cannot. Promoting it additionally needs the n ≤ 6 exact check done carefully,
  which is scoped here but should be treated as a follow-on.
- **Whether to also report a leak-free `edge_existence` stratum.** Reporting the shortcut
  score covers the interpretation problem, so the sampling filter is now optional. It
  would still buy one cell where the proposal's aligned/adjacent/agnostic framing survives
  intact, at the cost of a sampling constraint already verified as distribution-neutral.
  Decide when writing up, not now.
- ~~**Real-data confirmation.**~~ **Settled.** `scripts/measure_real_rows.py
  --shortcut-table` re-runs the table with the 500 published `zero_shot_test` graphs as
  the evaluation set, still fitting on generated seed-999 graphs. Every theorem keeps
  precision 1.0, all six verdicts hold, and the `components` × `cycle_check` ladder
  reproduces exactly at 83.2% / 92.6% / 94.6% / 100%. The cells that moved are the ones
  whose task draws query nodes, and they moved because the published draw differs from
  the resampled one, not because the data differs: `edge_existence` 49.8% → 48.2%
  baseline and 79.4% → 77.8% shortcut, `node_degree` and `connected_nodes` 8.2% → 9.6%
  baseline. The generator turned out to be not merely a good proxy but the same corpus,
  so there was less here to confirm than the plan assumed.

  One caveat the table cannot shed: `build_rows` resamples queries rather than reading
  the ones the dataset ships, so `edge_existence`, `connected_nodes` and `node_degree`
  cells are still scored on a query draw of our own. On the published draw the
  `edge_existence` baseline is 53.0%, four points above the 48.2% the table reports.
  Reading a model's score against the table means resampling the model's queries too, or
  re-deriving the baseline from the published `task_description` fields.

  The n ≤ 6 exact island is too small to confirm anything either way: at 46 rows a single
  query draw carries a 3–6 point standard deviation, measured over ten seeds, which is
  wider than the differences between any two draws. In particular the `edge_existence`
  row's "tight" verdict was never robust — it flips to a 13-point gap on another draw of
  the same graphs.
