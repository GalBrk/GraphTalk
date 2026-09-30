# What else the primers do: nine directions on the 40-node sweep

**Status: exploratory; text detectors validated by hand.** The text measures A1,
A2 and A3 rest on (stated values, quotes, candidate answers, revision words,
`reports_conflict`) were checked on a blind, seeded sample of 226 items. One of
the authors labelled the sample by hand, and three independent LLM labellers
labelled it too; the author's labels agree with the final labels. Every pass/fail check passes (see
[Validation](#validation) and
[`primer-directions-validation.md`](primer-directions-validation.md)). A4 and
B5–B9 are numeric (answers and node ids against the graph and the primer text)
and need no labels. Intervals are unadjusted across the many comparisons below. Every number cited
here is checked against the outputs below by `tests/test_results_docs.py`;
the main results are in [`results/n40-sweep.md`](results/n40-sweep.md).

Source: `outputs/n40-sweep/directions_validation.txt`
Source: `outputs/n40-sweep/blind_bars.txt`
Source: `outputs/n40-sweep/primer_directions.txt`, printed by
`scripts/primer_directions.py`, which reuses `scripts/response_patterns.py`
(loading, `reports_conflict`, the degree tables) and `scripts/primer_findings.py`
(pairing on graphs). Reproduce:

    PYTHONPATH=. python scripts/primer_directions.py > outputs/n40-sweep/primer_directions.txt

Data: the main experiment's runs, Qwen3-1.7B and Qwen3-4B with and without
thinking (*1.7B*, *1.7B-T*, *4B*, *4B-T*), every primer against `none` on the
same graphs. Accuracy follows rule R1: a truncated response is never correct.
Every primer value is read from the primer text the model saw
(`graphtalk.shortcuts.parse_primer`). A number followed by a tag, `93% [pdsource]`,
is printed in that tag's block of the source file. Related:
[`investigate_connections_and_cycles.md`](investigate_connections_and_cycles.md)
checks whether the claims responses make about the graph are true, including why
`edge_existence` false alarms happen.

## Summary

1. **A stated degree that disagrees with the model's own count is detected, and
   resolved differently by each arm.** 4B-T ends on the primer's value in
   93% [pdsource] of such cases, 1.7B-T in 66% [pdsource], plain 4B in
   56% [pdsource]. On `edge_count`, a conflict goes with truncation: 93% [pdsource]
   of 1.7B-T responses with a conflict run out of budget, against 54% [pdsource]
   without one.
2. **Side information is read accurately and only by the thinking arms.** No
   response quotes a clustering coefficient or return probability without the
   primer (0 of 3000 [pdmisread] per arm), and the plain arms almost never quote
   one. The thinking arms' quotes are right 94.9% [pdmisread] to 98.7% [pdmisread]
   of the time. Under the long `all` primer, 4B-T's wrong quotes
   equal the neighbouring line's value about twice as often as chance.
3. **A primer that states the answer brings it into the reasoning earlier.**
   Under `degree`, 1.7B-T's first candidate for the node's degree comes at
   27% [pdcommit] of the trace, against 44% [pdcommit] without a primer, often
   as the model reads the stated value aloud before weighing it against its own
   count, and with fewer revision words.
4. **Answer-carrying primers help the queried nodes early in the list more than
   late ones**, for the weaker arms (1.7B `all`: +18.2 [pdposition] points more
   on the first quarter than on the last).
5. **Primers change how coherent a model's answers about one node are.** Plain
   1.7B's `edge_existence` and `connected_nodes` answers agree more under `all`
   (+11.5 [pdconsist]) and less under `filler` (−14.8 [pdconsist]); plain 4B's
   agree less under `rwse` and `all` (−9.5 [pdconsist] each). This follows the
   `edge_existence` accuracy effects in `n40-sweep.md` §3.
6. **No sign that printed values are used as an edge heuristic** beyond what a
   content-free primer does.
7. **The rwse and clustering effects on `node_degree` do not track how much those
   features reveal about degree** across densities.
8. **For plain 4B, `degree` helps high-degree queried nodes and hurts low-degree
   ones** (+22.5 [pdnode] points of difference): it trades a reliable count on
   small neighbour lists for a lookup that also helps on long ones.
9. **The degree primer removes the direction of the `edge_count` errors.** Plain
   1.7B over-counts and plain 4B under-counts without it; under `degree` both
   err in both directions, by less.

## A1. Which value a conflict ends on — `[pdsource]`

When a response reports a conflict (e.g. "the degree is given as 23 but I count
22"), which value does its final answer take: the primer's (which is always the
true value), another value the response computed, or neither?

- **Who sees a conflict.** Under `degree` on `node_degree`: 1.7B-T in
  38.4% [pdsource] of responses (2.1% [pdsource] without a primer, where no degree
  is stated), 4B-T 19.1% [pdsource], plain 4B 10.0% [pdsource], plain 1.7B
  0.0% [pdsource].
- **How it is resolved** (finished responses with a conflict, `node_degree` under
  `degree`): the primer's value / the model's own value / other.

  | Arm | n | Primer | Own | Other |
  |---|---|---|---|---|
  | 4B-T | 132 [pdsource] | 93% [pdsource] | 6% [pdsource] | 1% [pdsource] |
  | 1.7B-T | 218 [pdsource] | 66% [pdsource] | 30% [pdsource] | 4% [pdsource] |
  | 4B | 70 [pdsource] | 56% [pdsource] | 41% [pdsource] | 3% [pdsource] |

- **Conflicts and truncation.** On `node_degree`, 1.7B-T truncates in
  19% [pdsource] of responses with a conflict and 0% [pdsource] without. On
  `edge_count` the thinking arms report conflicts even without a primer
  (1.7B-T 11.8% [pdsource], 4B-T 33.0% [pdsource]). In a labelled sample, all
  20 [dvc2] such conflicts without a primer are between the model's own recounts
  or sums; under `degree`, 14 [dvc2] of 20 still are and 6 [dvc2] set a stated
  degree against the model's own count. Under `degree` this rises to
  69.8% [pdsource] and
  64.5% [pdsource], and responses with a conflict truncate in 93% [pdsource] and
  83% [pdsource] of cases against 54% [pdsource] and 15% [pdsource] without.
- **What it tells us.** The larger thinking model trusts the stated value; the
  smaller one keeps its own count a third of the time. On `edge_count` the
  stated degrees set off more checking. Most of it is the model re-checking its
  own sums, and part of it weighs a stated degree against a recount. That
  checking does not end within the budget, which is a mechanism for the
  truncation the degree primer causes there.

## A2. Misreading the primer — `[pdmisread]`

Clustering coefficients and return probabilities that a response quotes in the
primer's phrasing, checked against the primer: right, the neighbouring line's
value (k−1 or k+1), or another value. Degree is left out, because models state
degrees they counted in the same words; degree copying is measured on
`edge_count` tables (`[rpchain]`) and `node_degree` answers (`[rpcopy]`) in
`response_patterns.py`.

- No response quotes such a value without the primer: 0 of 3000 [pdmisread] in
  every arm. The quotes are reads of the primer.
- Only the thinking arms quote: under `all`, 855 [pdmisread] 1.7B-T responses
  quote clustering values, against 8 [pdmisread] plain 1.7B responses.
- Quotes are accurate: 97.6% [pdmisread] right for 1.7B-T under `clustering`,
  94.9% [pdmisread] for 4B-T under `all`.
- Under `all` (about 4,700 characters), 4B-T's wrong quotes take the
  neighbouring line's value more often than chance: 38 [pdmisread] against
  21.6 [pdmisread] for clustering, 30 [pdmisread] against 15.9 [pdmisread] for
  return probabilities. 1.7B-T is closer to chance (43 [pdmisread] against
  32.1 [pdmisread] under `rwse`).
- **What it tells us.** Side information is read, not ignored or garbled; its
  cost (n40-sweep §7) is not a reading error. The errors that do occur in the
  longest primer are slips to the adjacent line, as for degree.

## A3. Early commitment and revision — `[pdcommit]`

Thinking arms, main sweep: where in the reasoning the first candidate answer
appears, whether it is the final answer, and revision words ("wait", "let me
recount", "mistake") per 1,000 characters.

- `node_degree`: without a primer the first candidate is almost always final
  (1.7B-T 99% [pdcommit], 4B-T 100% [pdcommit]). Under `degree` it appears much
  earlier, at 27% [pdcommit] of 1.7B-T's trace against 44% [pdcommit] (4B-T:
  30% [pdcommit] against 48% [pdcommit]), is final slightly less often
  (93% [pdcommit], 98% [pdcommit]), and revision words fall (1.7B-T
  4.13 [pdcommit] to 2.98 [pdcommit] per 1,000 characters).
- `edge_count`: under `degree` the first candidate also comes earlier (1.7B-T
  37% [pdcommit] against 54% [pdcommit]; 4B-T 39% [pdcommit] against
  58% [pdcommit]), but revision words double (1.7B-T 1.01 [pdcommit] to
  2.19 [pdcommit]; truncated responses 1.11 [pdcommit] to 3.46 [pdcommit]).
  4B-T's first candidate is final in 76% [pdcommit] of finished responses without
  a primer and 69% [pdcommit] under `all`.
- **What it tells us.** The stated answer enters the reasoning early, often when
  the model reads the degree list aloud. The model then weighs it against its own
  count before deciding; the validation counts such a reading as a candidate only
  when that comparison follows. On `node_degree` the model then verifies briefly.
  On `edge_count`, the early value is followed by more revision, the checking of
  A1.

## A4. Where in the primer — `[pdposition]`

`node_degree` over all seven densities: the primer's effect on queried nodes in
the first quarter of the lines (ids 0–9) minus its effect on the last quarter
(ids 30–39). This is a difference in differences against `none`, because the
encoding also lists nodes in id order.

| Arm | `degree` | `all` | `rwse` | `filler` |
|---|---|---|---|---|
| 1.7B | +13.1 [pdposition] | +18.2 [pdposition] | +21.5 [pdposition] | +7.7 [pdposition] |
| 1.7B-T | +13.0 [pdposition] | +16.1 [pdposition] | −0.4 [pdposition] | +6.5 [pdposition] |
| 4B | +9.1 [pdposition] | +10.7 [pdposition] | −1.3 [pdposition] | −5.7 [pdposition] |
| 4B-T | −3.2 [pdposition] | −2.4 [pdposition] | +2.0 [pdposition] | −3.7 [pdposition] |

- Intervals exclude zero for 1.7B `degree` [+3.2, +22.9] [pdposition], `all`
  [+7.9, +28.6] [pdposition] and `rwse` [+12.1, +30.8] [pdposition], 1.7B-T
  `degree` and `all`, and 4B `all` [+1.8, +19.9] [pdposition]; `filler` does not
  for the 1.7B arms.
- **What it tells us.** For the arms that gain from a primer, the gain is
  concentrated in the first lines. The weaker models read the start of a long
  primer better than its end. 4B-T shows no gradient. Node ids 0–9 are both the
  first lines and single-digit, so the two cannot be separated (as in n40-sweep §4).

## B5. Consistency across tasks on the same node — `[pdconsist]`

`node_degree`, `connected_nodes` and `edge_existence` query the same node t. An
`edge_existence` answer for (t, b) is consistent when "yes" goes with b being in
the model's own `connected_nodes` list for t. Main sweep, graphs where both
responses finished.

- Without a primer: plain 1.7B 70.0% [pdconsist], plain 4B 90.0% [pdconsist],
  1.7B-T 99.0% [pdconsist], 4B-T 100.0% [pdconsist].
- Plain 1.7B: `all` +11.5 [pdconsist] (p = 9.1e-06 [pdconsist]), `rwse`
  +5.5 [pdconsist]; `filler` −14.8 [pdconsist] (p = 2.7e-10 [pdconsist]),
  `components` −8.0 [pdconsist].
- Plain 4B: `rwse` and `all` −9.5 [pdconsist] each, `filler` −6.8 [pdconsist],
  `degree` −4.5 [pdconsist]; `clustering` +4.5 [pdconsist].
- 1.7B-T: small losses (`degree` −3.1 [pdconsist]); 4B-T stays at 100% under
  every primer.
- The share consistent but not both correct (a coherent but wrong picture) falls
  for 1.7B-T from 20.1% [pdconsist] to 12.2% [pdconsist] under `degree`.
- **What it tells us.** A primer changes whether a plain model holds one picture
  of the node across tasks, in the same direction as its `edge_existence`
  accuracy effect (plain 1.7B `all` +11.5, plain 4B `all` −10.0 in n40-sweep §3).
  For plain 4B, most primers make the two answers less coherent.

## B6. Primer values as heuristics — `[pdheur]`

`edge_existence`, main sweep: do false alarms (yes on a non-edge) or misses rise
more for pairs whose endpoints print high values (above the density's median)
than for low ones, under a primer that shows the value, against `none`?

- Plain 4B's false alarms rise more on pairs of high-degree endpoints under
  `degree` (+7.1 [pdheur]) and `all` (+8.8 [pdheur]), but the intervals include
  zero, and the content-free `filler` does the same (+11.5 [pdheur],
  [+0.8, +22.2] [pdheur]).
- Under `rwse`, plain 4B's false alarms rise more on pairs with *low* printed
  return probability, that is high-degree endpoints: 14 → 34 against 10 → 17
  [pdheur], difference −12.4 [pdheur] [−21.8, −2.4] [pdheur]. `filler` points the
  same way (−6.9 [pdheur], interval including zero).
- The thinking arms make almost no false alarms under any condition.
- **What it tells us.** Nothing isolates a heuristic read from the printed values.
  The pattern that is there (plain 4B saying "yes" more for high-degree
  endpoints once any primer is added) appears with `filler` too, so it is a
  response to the presence of a preamble, not to its numbers.

## B7. Side-information effects by density — `[pddensity]`

How far do the printed rwse or clustering values pin a node's degree, and does
the primer's `node_degree` effect follow that across the seven densities? The
measure is the gain over guessing the graph's most common degree: 0 = adds
nothing, 1 = pins every degree. It works within one graph: each node gets the
most common degree among the nodes that print the same value in that graph, so
a value printed by one node only counts as pinned. It measures how finely the
values separate a graph's nodes, not what a reader without the graph recovers.

- rwse separates nodes well only in sparse graphs: gain 0.79 [pddensity] at p = .10,
  0.16 [pddensity] at p = .50. Clustering: 0.63 [pddensity] at p = .20,
  0.13 [pddensity] at p = .85. At p = .10, 55 [blindunique] per cent of nodes print
  an rwse pair no other node in their graph prints.
- Read without the graph, by a lookup from printed value to degree fit on 1,000
  other graphs per density (`scripts/blind_bars.py`), rwse gives the queried
  node's degree on 41.0 [blindbar] per cent of prompts at p = .10 and clustering
  on 54.0 [blindbar] at p = .20, against 17.0 [blindbar] and 16.0 [blindbar] from
  the majority answer.
- The rwse effect does not follow the gain. At p = .10, where rwse separates
  nodes best, the effect is +3, +6, +0 and −3 points [pddensity] in the four arms, and
  no arm's rank correlation over the seven densities is significant (1.7B-T
  rho +0.64 [pddensity], p = 0.12 [pddensity]).
- Clustering: 1.7B-T's effect does follow the gain (rho +0.85 [pddensity],
  p = 0.016 [pddensity]): small gains at low density, losses at p ≥ .65. Plain
  4B's goes the other way, growing with density to +18 [pddensity] at p = .85
  while the gain falls.
- **What it tells us.** The models do not turn rwse into a degree even in sparse
  graphs, where the printed values give a reader without the graph part of the
  answer; plain 4B's high-density clustering gain (n40-sweep §7) is not
  explained by what the feature reveals.

## B8. Which nodes a primer helps — `[pdnode]`

Effects by the queried node's degree tercile within its graph (low / mid / high).

- Plain 4B, `node_degree`, `degree`: −11.8 / +1.1 / +10.6 [pdnode]; high minus low
  +22.5 [pdnode], [+14.0, +30.7] [pdnode]. Under `all` every tercile loses
  (−19.4 / −12.5 / −9.7 [pdnode]).
- Plain 4B shows a smaller high-minus-low gradient under every primer, including
  `filler` (+6.9 [pdnode]) and `clustering` (+11.6 [pdnode]).
- No other arm or task shows a clear gradient; only `components` on
  `edge_existence` does (plain 1.7B +6.9 [pdnode], plain 4B +6.6 [pdnode]).
- **What it tells us.** For plain 4B, the stated degree helps where counting is
  hard (long neighbour lists) and hurts where counting was reliable (short ones).
  This is the within-graph form of the density sign flip in n40-sweep §4. Part of
  the gradient appears with any preamble.

## B9. Direction of errors — `[pderror]`

Finished responses with a numeric answer.

- `edge_count` without a primer: plain 1.7B over-counts (76% [pderror] of answers
  too high, median error of the wrong ones +48 [pderror]); plain 4B under-counts
  (80% [pderror] too low, median −14 [pderror]).
- Under `degree` both lose their direction: plain 1.7B 53% [pderror] too low and
  43% [pderror] too high, median −4 [pderror]; plain 4B 32% [pderror] and
  38% [pderror], median +2 [pderror]. The median relative error of plain 1.7B's
  wrong answers halves, from 31% [pderror] to 15% [pderror].
- `clustering` deepens plain 4B's under-count: 91% [pderror] too low, median
  −32 [pderror].
- `node_degree`: `all` doubles plain 4B's over-counts (17% [pderror] to
  30% [pderror] of answers); `degree` removes most of 1.7B-T's (12% [pderror] to
  2% [pderror]).
- **What it tells us.** Without a primer each plain model has a systematic bias
  on `edge_count` (the small one adds, the larger one drops). The stated degrees
  replace it with unbiased arithmetic slips. `clustering` pushes plain 4B further
  toward dropping.

## Validation

A blind, seeded sample of each text detector was drawn and scored by
`scripts/validate_directions.py`. One of the authors labelled it by hand, and
three independent LLM labellers labelled it too. The author decided the 8 items
where the LLM labellers disagreed (six C3, one C4, one C6), and the author's own
labels agree with the final labels. A check passes when the detector agrees with the
labels on 90% or more.

| Check | Detector | Agreement | Supports |
|---|---|---|---|
| C1 | answer reached by the model's own work, or not | 39 [dvc1] of 40 [dvc1], 98% [dvc1] | A1 own/other |
| C3 | first candidate answer in the reasoning | 37 [dvc3] of 40 [dvc3], 92% [dvc3] | A3 commitment |
| C4 | revision words | 27 [dvc4] of 30 [dvc4], 90% [dvc4] | A3 revision rate |
| C5 | quoted primer value and its node | 45 [dvc5] of 45 [dvc5] | A2 misreads |
| C6 | `reports_conflict` | 31 [dvc6] of 31 [dvc6] | all of A1 |

C4 passes at the bar itself; its 95% interval runs from 73 [dvc4] to
98 [dvc4] percent. C2 is not pass
or fail; it reads what the `edge_count` conflicts compare (A1).

## Limits

- The text measures are regular expressions checked on samples of 30 to 45 items
  each, by one human labeller and three LLM labellers. A second human labeller
  would allow an inter-rater agreement score. A2 and A3 were tightened after a
  code review found node ids read as counts and degree sums read as edge counts.
- Many comparisons, no multiplicity correction; the intervals are 95% bootstrap
  intervals over graphs within density.
- A4 cannot separate line position from single-digit ids.
- B7 has seven densities per correlation.
- Plain arms rarely write out their reasoning, so A2 and A3 describe the thinking
  arms only.
