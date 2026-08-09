# Shortcut ceilings

## Context

This plan depends on `docs/plans/primer-computation.md` and cannot start before it: it
consumes rendered primer text, and that text must be stable and correct first.

The primer plan's taxonomy audit found deterministic routes from primer text into every
one of the six tasks for the `degree` and `all` conditions, and into `node_count` and
`connected_nodes` for `components`. The sharpest single case: on `edge_existence`, the
rule "answer Yes iff d_a + d_b > n−1" — one comparison between two numbers the primer
states — scores 79.2% ± 1.7, against a ≈51.6% majority baseline and the 45.1% the paper
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

## Anchors already measured

Seeds for the table, all on `generate_graphs(500, "er", False, random_seed=1234)` with
each task's own query sampling:

| cell | baseline | shortcut | notes |
|---|---|---|---|
| `degree` × `edge_existence` | ≈51.6% | 27.9% coverage at 100% precision (theorem); 79.2% accuracy (heuristic) | paper reports 45.1% for a model |
| `components` × `cycle_check` | 83.2% | 93.2% at rung 1; 100.0% at rung 3 | the ladder in action |
| `components` × `node_count` | — | states the answer on 1.2% of rows | c = n ⟺ m = 0 |
| `components` × `connected_nodes` | — | excludes `" No nodes."` on 73.6% of rows | c = 1 ⟹ no isolated node |
| `clustering` × `cycle_check` | 83.2% | fires on 80.8% at 100% precision; 97.6% as a rule | clustering > 0 ⟹ triangle |
| `degree` × `connected_nodes` | — | 9.0% coverage at 100% precision | degree 0 ⟹ `" No nodes."` |

Every number here is generator-derived; network access to the rows API is blocked in the
current environment. See the provenance section of the primer plan.

## Files

- `graphtalk/shortcuts.py` — new; rule definitions, the solver, the primer parser
- `scripts/shortcut_table.py` — new; computes and prints the 36-cell table at all three
  rungs, with coverage for theorems and accuracy for the rest
- `tests/test_shortcuts.py` — new
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
  other.
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
- **Real-data confirmation.** Every rate here needs re-measuring on the published rows.
  The generator is the code that produced GraphQA, so it is a good proxy, but it is a
  proxy — and the plan it supersedes recorded two dataset statistics that turned out to be
  wrong.
