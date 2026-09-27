# Primer robustness

How much of a primer's effect in the 40-node sweep is the prompt changing
rather than what the primer says, whether the models agree on which questions a
primer fixes, and five checks on the responses themselves: why responses run
out of budget, whether `cycle_check` answers agree with the same model's
`edge_count` answers, whether the final answer follows the thinking trace,
whether the scorer reads the boxed answer, and whether fixes come with longer or
shorter responses.

Runs: the 40-node sweep at p ≤ .50 (`runs/qwen3-{1.7b,4b}[-think].densfull40*`).
These are Erdős–Rényi graphs with 40 nodes, 100 per density at p = .10, .20, .35
and .50, with six tasks and seven conditions. The models are Qwen3-1.7B and
Qwen3-4B, each with thinking off (1.7B, 4B) and on (1.7B-T, 4B-T).

This file is not in `docs/results/`, and `tests/test_results_docs.py` does not
check it. Every number names its source:

| Tag | Source |
|---|---|
| **R** `[tag]` | `csv2/raw-trends/primer_robustness.txt` (`scripts/primer_robustness.py`); per-response rows in `csv2/raw-trends/robustness_responses.csv` |
| **C** `[tag]` | `csv2/raw-trends/check_cycle_claims.txt` (`scripts/check_cycle_claims.py`) |
| **P** `[tag]` | `csv2/raw-trends/primer_findings.txt` (`scripts/primer_findings.py`) |

Conventions:
- **Correctness.** Rule R1: a response that hits the token budget is
  truncated and never correct.
- **Question.** One (task, graph), 400 per (arm, task).
- **Flip.** A condition *flips* a question when the question's outcome
  (correct or not) differs from its outcome under no primer (none).
- **Significance.** Benjamini–Hochberg q < .05, over the family each section
  names.
- **Abbreviations.** ND `node_degree`, CN `connected_nodes`, EC `edge_count`,
  EE `edge_existence`, NC `node_count`, CC `cycle_check`.

## 1. Any change to the prompt flips many questions

P `[main]` gives each primer's fixed and broken counts against none. The table
below adds filler's flips, the control with no graph content. Each cell is:
- % of questions flipped against none,
- the net change in % correct,
- % that cancel (2 × the smaller of fixed and broken, per task, averaged over
  the six tasks).

The cancelling part is back-and-forth that a real effect of filler cannot
inflate. R `[churn]`:

| Arm | filler | components | clustering | rwse | degree | all |
|---|---|---|---|---|---|---|
| 1.7B | 18.6 (+3.3; 7.7) | 15.6 (+2.8; 9.8) | 23.6 (+13.9; 9.6) | 20.8 (+1.8; 18.6) | 18.6 (+1.2; 11.6) | 23.4 (+13.1; 10.3) |
| 1.7B-T | 10.4 (+1.6; 7.2) | 12.3 (−1.0; 10.0) | 11.8 (+1.6; 9.2) | 11.2 (+1.6; 8.2) | 17.5 (−1.2; 9.5) | 13.9 (+5.0; 8.1) |
| 4B | 9.0 (−1.7; 3.8) | 6.6 (+2.1; 2.5) | 7.0 (+2.1; 4.3) | 8.5 (−0.8; 4.7) | 13.7 (+3.1; 5.1) | 10.7 (−1.2; 4.9) |
| 4B-T | 6.2 (+2.8; 3.5) | 8.7 (+3.6; 3.8) | 8.5 (+4.2; 3.8) | 7.1 (+1.5; 4.8) | 9.1 (+1.0; 6.2) | 7.5 (+0.0; 6.6) |

- **Every condition, filler included, flips questions both ways.** Between
  2.5% and 18.6% of the questions flip and cancel out.
- **Filler's back-and-forth matches the primers'.** Filler's cancelling share
  (7.7 / 7.2 / 3.8 / 3.5) is of the same size as the primers':
  - 1.7B: 9.6–18.6
  - 1.7B-T: 8.1–10.0
  - 4B: 2.5–5.1
  - 4B-T: 3.8–6.6
- **Nets are small next to flips.** 1.7B's degree primer flips 18.6% of the
  questions for a net +1.2. rwse flips 20.8% for +1.8.

**What the flips are.** This splits the flips by whether either response hit
the token budget. R `[churnwhy]`, % of each condition's flips:

| Arm | filler | components | clustering | rwse | degree | all |
|---|---|---|---|---|---|---|
| 1.7B | 13.0 | 13.6 | 6.7 | 8.0 | 21.9 | 6.8 |
| 1.7B-T | 48.4 | 64.5 | 50.2 | 44.4 | 61.2 | 51.8 |
| 4B | 3.2 | 5.7 | 3.0 | 3.9 | 3.3 | 2.3 |
| 4B-T | 94.7 | 95.2 | 93.1 | 88.3 | 89.5 | 83.3 |

- **In the thinking arms, most flips happen at the token budget.** In 4B-T,
  83–95% of flips have one response that ran out of tokens; in 1.7B-T,
  44–65%. A condition flips these questions because it makes the response a
  little longer or shorter than 8,192 tokens, and filler does this as well as
  any primer. That is a concrete mechanism, not noise: questions whose
  response runs close to the budget.
- **In the plain arms, most flips are finished answers that change:** 78–93%
  for 1.7B, 94–98% for 4B. These data cannot say why. Three explanations fit:
  - **The model gets these questions right only some of the time,** and any
    edit re-draws the outcome. Some of this is pure chance: identical prompts
    regenerated change correctness on 5.3% of 1.7B `node_degree` questions
    (P `[rerun]`), the only arm and task with reruns.
  - **Adding any block of text has its own effect** on the same questions.
    Filler fixes plain 1.7B's `node_count` in one direction (+39.8 of its
    40.8% flips), so this effect is real where it is large.
  - **Some questions are harder than their density suggests,** so they flip
    under any edit. Stratifying by density cannot remove this.
- **What this means for the paired tests.** McNemar uses only the pairs that
  disagree, and flips in both directions cost it power without biasing it, so
  the tests are valid. But the number of pairs that disagree is not evidence
  of content. A significant effect against none is the primer's content plus
  whatever any added text does. The comparison against filler removes the
  second part, but filler is not neutral (C `[pvn]`, §3 of
  `investigate_connections_and_cycles.md`).

**Which primers flip more or fewer questions than filler.** Tested per
(arm, task) against filler's flips on the same questions: exact McNemar, BH
over the five primers. Only **31 of the 120** (arm, task, primer) cells differ
from filler (R `[churn]`, the starred cells). The rest flip as many questions as
filler does.

| Arm | Cells that flip a different number of questions from filler (% flipped; filler's) |
|---|---|
| 1.7B | NC clustering 67.2, degree 14.8, all 49.5 (40.8); CC clustering 9.5, degree 32.5, all 10.5 (14.5); ND clustering 23.5, rwse 22.5, all 28.2 (16.5); EC degree 5.5 (1.5); EE components 13.8 (20.8) |
| 1.7B-T | NC components 15.0 (9.0); CC components 13.5, clustering 5.5, degree 25.8, all 6.8 (3.0); EC degree 25.8, all 27.5 (18.0) |
| 4B | ND components 0.2, degree 8.0, all 9.0 (2.0); EC degree 29.8 (3.8); CN components 12.8, clustering 15.5, rwse 17.0 (23.0) |
| 4B-T | CC components 4.8, degree 7.0 (1.0); ND rwse 6.2 (3.0); EC components 35.2, clustering 36.8, degree 35.5 (27.8) |

Some cells flip *fewer* questions than filler: 4B CN components (12.8 vs
23.0), 1.7B EE components (13.8 vs 20.8), 4B ND components (0.2 vs 2.0). In
those cells filler itself does harm: its net change is −12.5 on 4B CN and
−15.8 on 1.7B EE (R `[churn]`). The components primer adds far less text than
filler, 37 characters against 1,829 (P `[length]`).

## 2. Primers flip the questions that filler flips

For each primer, the table compares two groups of questions: those filler
flips, and those it leaves alone. It shows the % of each group the primer also
flips, and in brackets the difference within (task, density). Permutation
within (task, density), 2000 draws. All 20 tests are at p = .0005, the floor
for 2000 draws. R `[overlap]`:

| Arm (filler flips) | components | clustering | rwse | degree | all |
|---|---|---|---|---|---|
| 1.7B (447 of 2400) | 49.9 vs 7.7 [+41.2] | 69.1 vs 13.2 [+46.6] | 47.0 vs 14.7 [+35.3] | 38.5 vs 14.1 [+34.2] | 59.3 vs 15.2 [+39.0] |
| 1.7B-T (250) | 57.2 vs 7.1 [+51.4] | 63.2 vs 5.8 [+56.4] | 61.6 vs 5.4 [+58.9] | 59.2 vs 12.7 [+41.9] | 57.6 vs 8.8 [+55.2] |
| 4B (216) | 42.1 vs 3.1 [+32.5] | 44.4 vs 3.3 [+40.1] | 54.6 vs 4.0 [+40.9] | 48.1 vs 10.3 [+36.3] | 55.6 vs 6.2 [+43.0] |
| 4B-T (150) | 52.7 vs 5.8 [+47.4] | 66.0 vs 4.6 [+61.7] | 51.3 vs 4.2 [+50.5] | 54.0 vs 6.1 [+59.9] | 46.7 vs 4.9 [+48.1] |

- **The same questions flip under any edit.** A question filler flips is
  flipped by any primer 38–69% of the time; any other question, 3–15% of the
  time.
- **The pattern holds within (task, density).** The gap is +32 to +62 points,
  so it isn't produced by mixing tasks of different difficulty.
- **Why they are the same questions.** In the thinking arms, the shared
  questions are mostly those whose response runs close to the token budget
  (§1, `[churnwhy]`). In the plain arms the three explanations of §1 remain:
  outcomes re-drawn by any edit, a common effect of adding text, or difficulty
  within density. With a right/wrong outcome, two conditions that flip a
  question always flip it the same way. So the overlap alone cannot tell a
  common effect from chance.
- **The degree primer stands out in three arms.** It has the highest flip rate
  on questions filler leaves alone in 1.7B-T (12.7), 4B (10.3) and 4B-T (6.1).
  In 1.7B, all (15.2) and rwse (14.7) are higher. This fits a primer that
  states the answer on ND and EC (C `[pvn]`) reaching questions that other
  edits leave alone.

## 3. How many questions can move at all

Across all seven conditions, a question is right under every condition, wrong
under every condition, or mixed. R `[fragile]` (% right always / wrong always /
mixed):

| Task | 1.7B | 1.7B-T | 4B | 4B-T |
|---|---|---|---|---|
| NC | 7.5 / 0.2 / 92.2 | 81.5 / 0.0 / 18.5 | 88.2 / 0.0 / 11.8 | 98.2 / 0.0 / 1.8 |
| CC | 51.2 / 0.0 / 48.8 | 60.5 / 0.0 / 39.5 | 89.8 / 0.0 / 10.2 | 86.2 / 0.0 / 13.8 |
| ND | 40.0 / 11.2 / 48.8 | 53.2 / 1.5 / 45.2 | 82.2 / 0.0 / 17.8 | 89.2 / 0.0 / 10.8 |
| EC | 0.0 / 92.0 / 8.0 | 0.0 / 44.0 / 56.0 | 0.0 / 63.7 / 36.2 | 0.2 / 21.2 / 78.5 |
| CN | 47.2 / 9.5 / 43.2 | 59.2 / 4.5 / 36.2 | 54.2 / 0.0 / 45.8 | 83.0 / 0.2 / 16.8 |
| EE | 45.2 / 4.2 / 50.5 | 89.0 / 0.0 / 11.0 | 66.2 / 1.2 / 32.5 | 96.5 / 0.0 / 3.5 |

- **The movable share bounds any effect.** An effect can only come from mixed
  questions: 4B-T has 1.8–16.8% outside EC, and 1.7B has 43–92% outside EC.
- **EC is the one task that is wrong everywhere.** No question is right under
  all seven conditions in any arm. Up to 92.0% (1.7B) are wrong under all of
  them.
- **Density.** Mixed questions grow with density on ND and CN for the 1.7B
  arms (% mixed at p = .10, .20, .35, .50):
  - 1.7B ND: 14, 43, 75, 63
  - 1.7B-T ND: 17, 29, 68, 67
  - 1.7B CN: 25, 33, 55, 60
  - 1.7B-T CN: 18, 23, 49, 55

  They shrink with density on the thinking arms' EC, where almost nothing is
  right under every condition, so the rest is wrong under all of them: 1.7B-T
  91, 69, 34, 30; 4B-T 98, 97, 78, 41.
- **Questions truncated without a primer are rescued by some condition, except
  on EC.** When none hits the budget, some condition answers the question
  correctly in every case outside EC:
  - 1.7B CC: 36 of 36
  - 1.7B-T NC, CC and ND: 31, 12 and 3
  - 4B-T ND: 11 of 11

  On EC the rescued share is 3.4% (1.7B), 47.4% (1.7B-T) and 73.3% (4B-T).

  This is what six further attempts would give even if no condition helped: a
  question answered right a fair share of the time is answered right at least
  once in six tries. So it shows only that running out of tokens under none is
  not a fixed property of the question. It does not show that the primers
  rescue it.

## 4. Where the arms fail, and are fixed on, the same questions

**Agreement without a primer.** Cohen's kappa of correct under none, with
chance agreement taken within each density. R `[crossarm]`:

| Pair | NC | CC | ND | EC | CN | EE |
|---|---|---|---|---|---|---|
| 1.7B vs 1.7B-T | 0.02 | −0.04 | 0.24 | −0.05 | **0.56** | 0.01 |
| 1.7B vs 4B | 0.00 | −0.02 | 0.01 | 0.13 | 0.05 | 0.16 |
| 1.7B vs 4B-T | 0.00 | 0.03 | 0.01 | 0.00 | −0.04 | −0.00 |
| 1.7B-T vs 4B | −0.03 | −0.01 | 0.00 | 0.01 | 0.06 | −0.03 |
| 1.7B-T vs 4B-T | −0.00 | −0.01 | −0.02 | 0.02 | 0.02 | −0.00 |
| 4B vs 4B-T | 0.09 | −0.01 | −0.01 | 0.03 | −0.07 | −0.00 |

- **Almost no agreement beyond density.** Once density is taken out, whether
  one arm gets a question right says almost nothing about whether another arm
  does: kappa −0.07 to 0.16 in five of the six pairs.
- **The one exception is 1.7B with 1.7B-T,** the same weights in two modes,
  on CN (0.56) and ND (0.24). 4B and 4B-T, also the same weights, agree at
  chance.

Kappa near 0 has three explanations, and the data favour the third where they
can tell:
- **Ceiling and floor.** Kappa means little where an arm is almost always
  right or wrong (4B ND 99%, most arms' EC near 0).
- **Similar questions.** Erdős–Rényi graphs at one density may differ too
  little in difficulty for any question to be hard for every model.
- **Different error mechanisms.** Agreement appears exactly where two arms
  share one:
  - On CN, the 1.7B arms both drop the *last* neighbour of a line (39.1% and
    21.0% of their misses), and plain 4B drops the *first* (61.3%)
    (C `[cnsource]`). Kappa is 0.56 for 1.7B with 1.7B-T and 0.05 for 1.7B
    with 4B.
  - On EE, both plain models call pairs with shared neighbours edges
    (C `[eeshared]`), and 1.7B with 4B has the highest cross-size kappa (0.16).

  So a model fails a question when *its* mechanism fails on it, and questions
  are shared only when mechanisms are.

**Fixes shared across arms.** This restricts to questions both arms get wrong
under none. For each condition it compares the questions fixed in both arms
against the count expected if fixes were independent within (task, density).
Permutation test, BH over the six conditions per pair. R `[crossarm]`:

| Pair (questions both wrong) | More shared fixes than chance |
|---|---|
| 1.7B vs 1.7B-T (496) | filler 33 vs 26.3; components 24 vs 18.6; clustering 40 vs 33.6; rwse 36 vs 25.0 |
| 4B vs 4B-T (318) | degree 37 vs 24.4 |
| the other four pairs | none |

- **The same weights share fixes, under any condition.** 1.7B and 1.7B-T
  share fixes beyond chance under filler too, so this is the same model's
  fragile questions, not the primer's content.
- **The degree primer fixes the same questions in 4B and 4B-T.** That is the
  one case where a primer's content fixes the same questions in two arms.
- **Across model sizes, fixed questions overlap at chance.** A primer does not
  pick out graphs that are "hard in a way the statistic helps with". What it
  fixes depends on the model.

## 5. Why responses run out of budget

A truncated response is a *loop* when its last 2,000 characters repeat with a
fixed period: at least 98% of characters equal the one a period earlier, for
some period up to 800 characters. The scores split cleanly, so the threshold
sits in an empty stretch (R `[loop]`, the score line). Of 5,390 truncated
responses, 1,118 score 0.98 or more and 12 fall between 0.90 and 0.98; most
score below 0.3 (3,371), and 695 score 0.3–0.5.

*Working* covers everything else, including runaway running sums such as
"4617 + 19 = 4636 …", which change their numbers and so never repeat exactly
(R `[loopsample]`). R `[loop]`:

| Arm | Truncated | Loops | Median period (characters) |
|---|---|---|---|
| 1.7B | 800 | 83.9% | 16 |
| 1.7B-T | 2,582 | 6.8% | 29 |
| 4B | 48 | 89.6% | 75 |
| 4B-T | 1,960 | 11.6% | 176 |

- **Plain arms run out of budget by looping; thinking arms by working.**
  Plain loops include "18 + 18 + 18 …", "0 → 18 → 10 → 18 → 10 …" and 4B
  restating "Node 10 → 6 → 10 is not valid" (R `[loopsample]`).
- **The thinking arms' EC truncations are work, not loops.** Without a primer,
  1.7B-T truncates 80.0% of EC questions while working (1.8% in a loop), and
  4B-T 78.2% (1.2%).

Changes against none (exact McNemar, BH over the six conditions within
(arm, task)). R `[loop]`, % of all responses:

| Arm, task | Truncated in a loop | Truncated while working |
|---|---|---|
| 1.7B CC | none 9.0 → clustering 0.8, rwse 1.8, all 0.8; **degree 16.8** | — |
| 1.7B EC | none 22.2 → clustering 13.8, degree 8.8 | none 7.5 → rwse 1.2, degree 0.8 |
| 1.7B-T NC | none 5.2 → 0.0 under filler, clustering, rwse and all; degree 0.8 | none 2.5 → 0.0 under filler, clustering, rwse and all |
| 1.7B-T CC | — | none 2.0 → filler 0.0; **components 9.2, degree 21.5** |
| 1.7B-T ND | — | **none 0.0 → degree 4.8** |
| 1.7B-T EC | **none 1.8 → all 6.8** | none 80.0 → all 62.7 |
| 4B-T CC | — | **none 0.5 → components 3.8, degree 5.2** |
| 4B-T ND | none 2.8 → degree 0.2 | — |
| 4B-T EC | — | none 78.2 → filler 65.0, components 52.5, clustering 51.2, rwse 65.0, degree 55.2, all 52.8 |

Bold marks increases.

- **4B-T EC: primers shorten work, not break loops.** The primers that reduce
  4B-T's EC truncation cut the truncations *while working*. Filler does too
  (78.2 → 65.0), so part of that is any edit to the prompt.
- **The degree primer makes the 1.7B arms run long on CC.** 1.7B loops more
  (9.0 → 16.8), and 1.7B-T works until the budget (2.0 → 21.5). In 1.7B, the
  degree condition is also where "no" answers cite an odd degree sum
  (C `[ccno]`).

## 6. `cycle_check` "no" answers against the same model's edge count

On 40 nodes, 40 or more edges imply a cycle. For each finished `cycle_check`
answer of no, this looks up the same arm's finished `edge_count` answer, under
the same condition, on the same graph. R `[cyclecons]`:

| Arm | "No" answers | With an edge count | Of those, 40 or more | Same share among "yes" answers |
|---|---|---|---|---|
| 1.7B | 77 | 69 | 69 (100%) | 99.6% of 1,964 |
| 1.7B-T | 2 | 1 | 1 | 99.4% of 508 |
| 4B | 18 | 18 | 17 (94.4%) | 99.1% of 2,737 |
| 4B-T | 3 | 1 | 1 | 100% of 1,041 |

- **The "no" answers contradict the model's own count.** When a model answers
  that the graph has no cycle, its own count of the same graph's edges almost
  always implies one: 88 of the 89 "no" answers that have a count.
- **The "no" does not come from judging the graph sparse.** The same counts go
  with "yes" answers.
- **What this does and does not show.** The edge count answers a different
  prompt about the same graph, so the "no" answer never saw that count. The
  comparison shows that a model that says "no cycle" still counts the graph as
  dense. It suggests the model does not reason from the edge count when asked
  about cycles; it does not show a contradiction inside one response.
- **Most "no" answers come from rwse and degree.** In 1.7B: rwse 31 (26 with
  40 or more), degree 37 (36). The reasons those answers give (C `[ccno]`: an
  odd degree sum under degree, "the graph is directed" under rwse) are
  unrelated to edges.

## 7. The final answer follows the trace

For the thinking arms, this compares the answer each finished trace concludes
with the scored answer. The trace's answer is the scorer's own extractor run
on the text before `</think>`. CN is left out, because in a trace the
extractor reads partial lists.

R `[faithful]`: 23,578 responses are compared; 32 differ, 24 of them in 1.7B-T
`node_count`. At most 0.9% of any (arm, task) differs.

Reading all 32 (R `[faithsample]`):
- **28 are the extractor misreading the trace**, not disagreements. The trace
  concludes the scored answer ("So 0 to 39 is 40 nodes. Therefore, the answer
  is 40."), and the extractor takes another number from it (39, 1 from
  "1 connected component", "no" from "no other cycles").
- **3 are real, and all 3 are under the degree primer.**
  - 1.7B-T `node_degree` p0.35/29: the trace concludes a correct 13, and the
    answer part switches to 12.
  - 1.7B-T `edge_count` p0.35/47: the trace concludes a correct 276, and the
    answer part recomputes 280.
  - 4B-T `node_degree` p0.5/40: the trace settles on the primer's 21, and the
    answer part resolves the conflict to the correct 26.
- **1 is the scorer**: 1.7B-T `node_count` p0.35/25 boxes 40, and is scored 39
  (§8).

So in all but three of the 23,578 responses, the final answer is the one the
trace reached, and primers have nothing to change there.

## 8. A check on the scorer: the boxed answer

This compares finished `node_count`, `node_degree` and `edge_count` answers
whose answer part holds a `\boxed{}` integer, taking the last one, with the
scored answer. R `[boxed]`: 10,448 such answers, 31 differ.
- **8 are scored wrong though the boxed value is right**: 1.7B `edge_count` 3,
  1.7B-T `node_count` 3, 4B-T `node_count` 2. For example, "The graph contains
  **40 nodes**, as all nodes from 0 to 39 are explicitly listed … \boxed{40}" is
  read as 39.
- **23 box two values and both are wrong.** 4B `edge_count` boxes 290.5 and
  then a rounded value; 1.7B `edge_count` 3.
- **None is scored right with a wrong boxed value.**

Among boxed answers, the scorer undercounts accuracy by 8 responses, at most 3
in any (arm, task). No main-sweep cell of 400 questions (arm, task, condition)
moves by more than 0.75 points. The scorer was left as it is.

## 9. Do fixes come with longer or shorter responses?

This takes pairs where both responses finished. For each, it reports the median
change in new tokens against none over all pairs, over the questions the
condition fixes, and over those it breaks. Mann–Whitney tests fixes against
breaks, BH within the arm over cells with 10 or more of each. R `[lenmed]`, the
clear patterns:

| Arm, task | What the fixes and breaks look like |
|---|---|
| 4B EE | Every condition, all six significant: fixes run **longer** (median +26 to +166 tokens, 12.5–33.3% shorter), breaks run **shorter** (−98 to −171, 73.6–87.5% shorter). |
| 4B CN | Breaks shorter under filler (−80), rwse (−85) and all (−70). Fixed against broken is significant for filler, clustering, rwse, degree and all. |
| 1.7B EE | Clustering, rwse, degree and all: fixes run longer (+78 to +135). Filler's breaks are shorter (−83, 98.6% shorter). Five of six significant. |
| 1.7B NC | Fixes much shorter (−175 to −200, 93–99% shorter). The breaks under components, rwse and degree run longer (+124 to +191), all three significant. |
| 1.7B-T ND | Degree: fixes +846, breaks +3,101 (significant). This fits the conflict-resolution traces P `[trunc]` reports for this arm and primer. |
| 1.7B-T CN | rwse: fixes +108, breaks +618 (significant). |
| 4B-T EC | Median over all pairs: all −3,133, degree −2,308. With every degree stated, the thinking model does far less counting. |

**The pattern comes from the questions, not the primers.** Across all
questions, 4B's wrong EE answers are no shorter than its right ones. The link
is *within* a question. This takes questions with at least one right and one
wrong finished response across the seven conditions, and compares the length
of the right responses with the wrong ones, on the same question. R
`[samequestion]`, median difference in tokens (% of questions where the right
ones are longer; questions):

| Arm | EE | CN | ND | EC | NC | CC |
|---|---|---|---|---|---|---|
| 1.7B | +71 (83.6%; 201) | +3 (54.9%; 173) | +2 (57.4%; 195) | −527 (18.8%; 32) | −132 (1.1%; 369) | −1,270 (4.1%; 74) |
| 4B | +95 (87.5%; 128) | +48 (79.8%; 183) | +46 (73.2%; 71) | +246 (79.3%; 145) | −7 (38.3%; 47) | −727 (43.8%; 16) |

- **On the same question, 4B's longer response is usually the right one.**
  EE 87.5%, CN 79.8%, EC 79.3%, ND 73.2%. 1.7B shows the same on EE (83.6%).
  When any edit re-draws such a question, a fix goes from a short wrong answer
  to a long right one, and a break the other way. So `[lenmed]`'s fixed/broken
  split describes the model, not what a primer does: checking more pays off
  on these tasks.
- **1.7B NC and CC go the other way.** There the long responses are the wrong
  ones: listing the ids one per line ends in 39 (C `[ncanswer]`), and a long
  CC response is a loop (§5).
- **What the lengths do say about primers.** Only the whole-pair medians:
  4B-T's EC responses shrink by 3,133 tokens under all and 2,308 under degree,
  because a primer that states every degree removes the counting.

## 10. Two checks of the other investigation's claims

These test alternative explanations for two findings of
`investigate_connections_and_cycles.md`, where they are discussed.

**Is the side-information gain any preamble's?** The table takes the cells of
P `[bands]`'s middle band (none-accuracy 0.25–0.75, under the truncation flag)
where components, clustering and rwse are measured. It compares each primer's
mean effect against none with filler's, on the same (arm, task, density).
R `[fillerband]`:

| components | clustering | rwse | the three | filler, same cells |
|---|---|---|---|---|
| −0.8 (17) | +2.4 (17) | +3.1 (17) | +1.6 (51) | **−6.9** (17) |

- **The +1.6 is not what any block of text does.** Filler, 1,829 characters of
  node names without structure, *loses* 6.9 points in the same cells.
- **The content effect lies between two bounds.** Against filler, the side
  statistics are about 8.5 points better. Filler costs graph reading
  (`investigate_connections_and_cycles.md` §3), so the side statistics' own
  effect lies between +1.6 (against none) and +8.5 (against filler).

**Shared neighbours, or long lines?** Pairs that share many neighbours also
have high degrees, and so long lines to read. Within density, shared
neighbours and the pair's degree sum correlate 0.40 (p = .10) to 0.87
(p = .50). R `[eedegree]` tests each with the other held fixed, on non-edges,
by permutation within strata:

| Arm, conditions | False yes | Shared neighbours, degree sum held fixed | Degree sum, shared neighbours held fixed |
|---|---|---|---|
| 1.7B, none | 118 | p = .0035 | p = .011 |
| 1.7B, all seven | 821 | p = .0005 | p = .0005 |
| 4B, none | 37 | p = .081 | p = .31 |
| 4B, all seven | 384 | p = .0005 | p = .0005 |

- **Both matter.** Shared neighbours predict a false yes beyond the pair's
  degrees, and the degrees predict it beyond shared neighbours.
- **The one exception is 4B without a primer,** where 37 false alarms are too
  few to separate the two.

## 11. What this adds

1. **Filler flips as many questions as most primers, and the same ones.**
   89 of 120 (arm, task, primer) cells flip no more or fewer questions than
   filler. A question filler flips is flipped by a primer 38–69% of the time,
   any other 3–15%.
2. **What the flips are depends on the arm.**
   - In the thinking arms, most are at the token budget (4B-T 83–95%, 1.7B-T
     44–65%).
   - In the plain arms, most are finished answers that change. The data
     cannot say whether they are chance, a common effect of adding text, or
     difficulty within density.
   - The paired tests are valid, but a primer-against-none effect includes
     what any added text does.
3. **The side statistics do beat an equally long text.** In the middle band,
   components, clustering and rwse gain +1.6 against none while filler loses
   6.9 on the same cells.
4. **Arms agree where their error mechanisms agree.** Kappa is near 0 across
   model sizes. It reaches 0.56 where both 1.7B arms drop the last neighbour
   of a line (CN), and 0.16 where both plain models are misled by shared
   neighbours (EE).
5. **Truncation has two causes.**
   - Plain models loop (84–90% of their truncations).
   - Thinking models run out mid-work (7–12% loops); on EC that is nearly all
     of it.
   - Primers that cut 4B-T's EC truncation cut the work, and filler does too.
6. **"No cycle" answers come from models that count the graph as dense.** 88
   of 89 such answers go with the same model's count of 40 or more edges on
   that graph, from a separate prompt.
7. **Final answers follow the traces.** There are 3 real departures in 23,578,
   all under the degree primer.
8. **The scorer undercounts 8 of 10,448 boxed answers.**
9. **On the same question, 4B's longer response is usually the right one.**
   Fixes therefore look longer and breaks shorter under any edit. This is a
   property of the model, not of a primer.

## 12. Limits

- **Kappa.** It is uninformative where an arm is near 0 or 100% correct
  (4B ND is 99%).
- **Chance.** Pure randomness is measured only for 1.7B `node_degree`
  (P `[rerun]`). No other arm or task has reruns, so the chance part of the
  plain arms' flips is unknown.
- **Filler.** It is the only content-free control, and it is not neutral. A
  second, different filler of the same length would separate a common effect
  of adding text from chance.
- **Loops.** A loop is exact repetition. Runaway arithmetic that changes its
  numbers counts as working.
- **Traces.** `[faithful]` runs the final-answer extractor on traces, so its
  counts are upper bounds. The 28 misreadings and 3 real cases come from
  reading every row of `[faithsample]`.
- **Response length.** `[lenmed]` conditions on both responses finishing, so
  fixes and breaks that come from truncation (§5) are outside it.

## 13. Reproduce

```bash
PYTHONPATH=. python scripts/primer_robustness.py --csv-dir csv2/raw-trends     > csv2/raw-trends/primer_robustness.txt
```

It reads `csv2/raw-trends/frame.csv` (`scripts/build_raw_frame.py`), the
40-node responses in `runs/`, and the prompts file for the graphs. It reuses:
- `check_cycle_claims.py`: graph parsing, the permutation test, splitting a
  response into trace and answer.
- `analyze_primer_window.cells`: the cells of `[bands]`.

A self-check covers:
- the loop score,
- the trace reading,
- the boxed-value pattern,
- kappa.

`robustness_responses.csv` has one row per response the text checks read,
with a `check` column: `loop`, `faithful` or `boxed`.
