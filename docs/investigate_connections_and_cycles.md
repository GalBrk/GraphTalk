# Investigating connections and cycles

What the primers change in the 40-node sweep, primer by primer, and whether the
models' correct answers rest on the graph or on claims about the graph that are
false. Research question: when a black-box language model reads a textual
graph, do explicit structural statistics improve execution of graph queries?

Runs: the 40-node sweep (`data/runs/qwen3-{1.7b,4b}[-think].densfull40*`), Erdős–Rényi
graphs with 40 nodes, p = .10–.50, 100 graphs per density. Also its
high-density extension, p = .65–.85. Models: Qwen3-1.7B and Qwen3-4B, each
with thinking off (plain) and on (-T).

This file is not in `docs/results/`, and `tests/test_results_docs.py` does not
check it. Every number names its source:

| Tag | Source |
|---|---|
| **P** `[tag]` | `outputs/n40-sweep/primer_findings.txt` (`scripts/primer_findings.py`), or `outputs/n40-sweep/primer_cells.csv` |
| **D** `[tag]` | `outputs/density-followups/density_followups.txt` (`scripts/density_followups.py`) |
| **C** `[tag]` | `outputs/n40-sweep/check_cycle_claims.txt` (`scripts/check_cycle_claims.py`); per-claim rows in `cycle_claims.csv` and `response_claims.csv` |
| **R** `[tag]` | `outputs/n40-sweep/primer_robustness.txt` (`scripts/primer_robustness.py`, see `primer-robustness.md`) |

Throughout: an *effect* is the paired change in % correct against no primer on
the same graphs, in percentage points; *significant* is Benjamini–Hochberg
q < 0.05 on exact McNemar tests; † marks a comparison where 15% or more of
responses hit the token budget. Task abbreviations: ND `node_degree`,
CN `connected_nodes`, EC `edge_count`, EE `edge_existence`, NC `node_count`,
CC `cycle_check`.

## Conclusions

This section is written for the paper. Each conclusion is stated in plain
words, followed by the numbers behind it, each with its source tag. Unless a
line says otherwise:
- **Data.** The main sweep, p ≤ .50, with 400 paired graphs per (model, task,
  primer).
- **Effect.** The change in % correct of *all* responses against no primer. A
  truncated response counts as not correct, and the truncated change is given
  beside the effect when it is 2 points or more.
- **Intervals and tests.** Brackets are 95% bootstrap intervals. q is
  Benjamini–Hochberg over the six conditions within each (model, task)
  (P `[main]`).
- **Finished shares.** A share marked *finished* counts finished responses
  only.

How much of an effect any change to the prompt makes, and where the models
agree, are the conclusions of `primer-robustness.md`.

### C1. A primer helps when it states an answer the model cannot compute; other statistics add little

On `node_degree`, `degree` and `all` print the queried node's degree, so a
solver that reads only the primer scores 100% (P `[bars]`). Effects
(P `[main]`):

| Model | % correct, none | `degree` | `all` |
|---|---|---|---|
| 1.7B | 60.50 | +7.5 [+3.0, +11.8], q = 0.0042 | +10.2 [+5.2, +15.2], q = 0.00086 |
| 1.7B-T | 76.25 | +8.8 [+4.2, +13.2], q = 0.0008 (truncated +4.5) | +12.2 [+8.0, +16.5], q = 3.4e-07 |
| 4B | 99.25 | −6.5 [−9.2, −4.0], q = 7.7e-06 | −8.0 [−11.0, −5.2], q = 1.2e-07 |
| 4B-T | 97.00 | +2.5 [+0.8, +4.2], q = 0.078 (direction only) | +0.5 [−1.5, +2.8], q = 0.98 |

- **The gain is where there is room.** In cells whose accuracy without a
  primer is 0.25–0.75, answer-stating primers gain:
  - +16.1 (10 cells) and +12.9 (8 cells), against +2.6 (20) and +3.9 (47) for
    the other primers (P `[bands]`).
  - +13.9 and +16.2 when the cells are sorted on half the graphs and measured
    on the other half, so this is not regression to the mean
    (P `[splithalf]`).
  - All 18 such cells are `node_degree`. 1.7B-T's 10 average +21.5
    (P `[window]`), and at p ≥ .65 `degree` gives 1.7B-T +24.8
    (D `[ddthink]`).
- **A model that already counts loses.** Plain 4B has 57 wrong answers under
  `degree` that take the degree without restating the node's list. 17 of them
  equal the degree printed for node k−1 or k+1, against 10.4 expected by chance
  (p = .017, P `[copyerr]`).
- **Statistics that do not state the answer add little, but it is not
  nothing.**
  - `components`, `clustering` and `rwse` average +1.6 over the 51 middle-band
    cells (P `[bands]`).
  - `filler`, 1,829 characters naming every node with no structure, averages
    −6.9 on the same 17 (model, task, density) (R `[fillerband]`). So their
    own effect lies between +1.6 (against none) and about +8.5 (against
    `filler`).
  - Outside `node_count`, five significant gains are not truncation-driven:
    `components` on 4B CN +9.2 [+6.0, +12.8]; `clustering` on 4B EE +4.0
    [+1.0, +7.0]; `degree` and `all` on 1.7B-T CN +6.2 [+2.8, +10.0] and
    [+2.5, +10.0]; `all` on 1.7B EE +11.5 [+6.2, +16.5] (C `[pvn]`,
    P `[main]`).
- **The largest gains are on `node_count`, and they are not counting.** They
  change which number the model writes (C3).

### C2. Correct answers often rest on false claims, and no primer fixes that

At 40 nodes every graph has a cycle, so `cycle_check`'s answer is always yes.
% of correct finished answers resting on a real or an invented cycle, no
primer (C `[ccanswer]`):

| Model | Correct answers | Real cycle | Invented cycle |
|---|---|---|---|
| 1.7B | 363 | 36.9 | 43.5 |
| 1.7B-T | 388 | 79.9 | 17.8 |
| 4B | 396 | 34.1 | 23.0 |
| 4B-T | 397 | 96.5 | 3.3 |

- **Thinking makes the reasoning real:** 36.9% → 79.9% (1.7B) and 34.1% →
  96.5% (4B).
- **Thinking traces still invent, and withdraw part of it.** 53.4% of
  1.7B-T's traces and 25.4% of 4B-T's assert an invented cycle, and part is
  withdrawn before the answer (C `[ccthink]`).
- **No primer fixes it.**
  - `degree` lowers invented cycles for the plain models (−25.1, −19.5) by
    replacing cycle-finding with an edge-count argument, and real cycles fall
    by about as much (−22.1, −20.6) (C `[ccinvent]`, `[cctest]`).
  - `clustering` and `all` *raise* invented cycles significantly for three
    models each, for example 1.7B-T `clustering` +13.0 and 4B `all` +11.3
    (C `[ccinvent]`).
- **The invented edge is usually the one that closes the walk:** 56.6% of
  1.7B's invented steps (C `[ccwhere]`).
- **The same happens on `edge_existence`.** A stated fake edge backs 33.9% of
  plain 1.7B's false alarms without a primer (finished, C `[eeclaims]`).

### C3. Why answers are wrong

- **`node_degree`: omission.** 24.6% of plain 1.7B's stated lists miss a
  neighbour and 3.3% invent one (C `[ndlist]`). 73.4% of its wrong answers are
  too low (finished, C `[ndsign]`).
- **`connected_nodes`: position.** The missed neighbour is the last on the
  line in 39.1% (1.7B) and 21.0% (1.7B-T) of misses, and the first in 61.3%
  for plain 4B (C `[cnsource]`).
- **`edge_existence` false yes: shared neighbours, long lines, and false "is
  listed" claims.**
  - % of non-edges answered yes (finished, all primers pooled) rises with
    shared neighbours: 1.7B from 1.7% (none shared) to 65.9% (4 or more), 4B
    from 0.2% to 38.8% (C `[eeshared]`).
  - Without a primer the link holds within density: p = .0005 (1.7B), .014
    (4B) (C `[eeshared]`).
  - With the pair's degree sum held fixed, it holds for 1.7B (p = .0035). For
    4B it does not (p = .081, 37 false alarms). The degree sum predicts false
    alarms on its own too (1.7B p = .011) (R `[eedegree]`).
  - Plain 4B's false alarms mostly claim the other node "is listed" in a list
    that does not contain it: 41.7% of 384 at p ≤ .50, 74.5% of 200 at
    p ≥ .65 (C `[eewhy]`).
- **`edge_existence` false no: rare, and a misread list.** There are 75
  misses in all, 68 at p ≥ .65. The thinking models read one list wrong and
  then rule that both lists must agree: 71% of 1.7B-T's 21 misses
  (C `[eemiss]`).
- **`node_count`: the last id, not a count.** Plain 1.7B answers 39 on 68.5%
  of finished answers (C `[ncanswer]`).
  - It lists the ids one per line in 59.5% of answers, and those are right
    0.8% of the time (238 answers); the others are right 76.5% of the time
    (162).
  - `clustering` removes the listing path (2.8%) and the error (1.2% answer
    39), for +67.2 [+64.5, +70.0] (P `[main]`).
  - Plain 4B's 9.8% disappears under every primer, `filler` included (+8.0 to
    +9.8, P `[main]`).
- **`cycle_check` "no": the same model counts a dense graph.** 88 of the 89
  "no" answers with an edge count go with that model's own count of 40 or
  more edges on the same graph, from a separate prompt (R `[cyclecons]`).

### C4. A statistic shown alone can change how the model reads the task

- **Plain 1.7B on `cycle_check`: each primer brings its own wrong reason.**
  - Its "no" answers rise from 0.3% to 7.8% under `rwse`, and from 0.0% to
    12.3% under `degree` (paired, finished; both q < .05) (C `[ccno]`).
  - 67.7% of the 31 `rwse` "no" answers call the graph directed. 97.3% of the
    37 `degree` "no" answers cite an odd degree sum or the handshaking lemma.
  - `all` prints both statistics, yet has 5 "no" answers and neither reason
    (0% and 0%) (C `[ccno]`).
  - Under `degree` it costs 1.7B −17.5 [−23.0, −12.2] of accuracy (truncated
    +8.2) (P `[main]`).
- **"Cycles of length 2" come with `rwse`, not with its words.** Correct
  answers resting on a walk to a neighbour and back rise under `rwse` to
  42.5% (4B-T, from 0.0) and 38.7% (4B, from 6.8). Under `all`, which prints
  the same "return probability … after 2 steps", they are 0.0% and 6.6%
  (C `[ccanswer]`). Under `rwse`, 4B-T's answers resting on a real cycle fall
  by 41.0 (C `[cctest]`).
- **`filler`, which names every node and states nothing, also changes the
  reading.**
  - Invented cycles +12.4 (1.7B, C `[ccinvent]`), stated fake edges +9.6
    (1.7B, C `[eeclaims]`), invented neighbours +14.8 (4B, C `[cnset]`).
  - 1.7B's `edge_existence` false-alarm rate goes from 0.51 to 0.69, with
    hits at 0.98–1.00 (finished, P `[fa]`).
  - Accuracy changes: 1.7B EE −15.8 [−20.0, −11.8], 4B CN −12.5
    [−17.2, −7.8] (P `[main]`).
  - Why is not tested. It adds 40 lines that open "Node X"; its "is simply
    present" matches the "is listed / present" claims above; and it moves the
    graph away from the question.

### C5. What is still open

These are open (§8):
- Why `clustering` fixes the `node_count` error and `rwse` does not.
- Whether `rwse`'s effects come from being the only statistic shown or from
  its wording.
- Acyclic 40-node graphs, without which a false "yes" on `cycle_check`
  cannot be measured.
- Missed edges, too rare to test.

## 1. The main question: statistics that state the answer, and statistics that do not

A primer *states the answer* when a solver that reads only the primer text
scores near-perfectly on the task. The solver's scores (P `[bars]`) split into
two groups with nothing between them:

| Task | none | filler | components | clustering | rwse | degree | all |
|---|---|---|---|---|---|---|---|
| ND | 14.4 | 14.4 | 14.4 | 14.4 | 21.0 | **100.0** | **100.0** |
| EC | 3.0 | 3.0 | 3.0 | 10.4 | 3.0 | **100.0** | **100.0** |
| CN | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.1 | 0.8 |
| EE | 73.5 | 73.5 | 73.5 | 72.7 | 55.4 | 74.6 | 74.1 |

Any cutoff in (0.746, 1.0] gives the same split; the code uses 0.85
(`scripts/analyze_primer_window.py`, `CARRIES`). A cutoff below 0.746 would mark
every `edge_existence` primer, `none` included, as stating the answer, because
73.5 is the majority-answer rate. Measured as the solver's gain over no primer
instead, the split is the same: +85.6 and +97.0 for the four answer-stating
cells, at most +7.4 (EC/`clustering`) elsewhere.

Mean effect by each cell's accuracy without a primer, cells under the truncation
flag, four tasks whose answer varies, all four models pooled (P `[bands]`):

| Accuracy without primer | States the answer (cells) | Does not (cells) |
|---|---|---|
| 0.00–0.25 | +8.3 (16) | −1.7 (23) |
| 0.25–0.50 | +16.1 (10) | +2.6 (20) |
| 0.50–0.75 | +12.9 (8) | +3.9 (47) |
| 0.75–0.90 | −5.8 (4) | +0.7 (51) |
| 0.90–1.00 | −0.9 (28) | −0.9 (177) |

- Grouping on half the graphs and measuring on the other half gives +13.9 and
  +16.2 for the middle bands (P `[splithalf]`); the pattern is not regression
  to the mean.
- The answer-stating cells in 0.25–0.75 are all `node_degree`: 18 cells, of
  which 10 are 1.7B-T at a mean of +21.5; plain 1.7B 4 at +10.5, plain 4B 4 at
  +1.8 (P `[window]`).
- The side-information gain in 0.25–0.75 is mostly degree information again:
  `degree` and `all` on CN and EE give +9.5 over 16 cells; `components`,
  `clustering` and `rwse` give +1.6 over 51 (P `[bands]`).
- **The +1.6 is not what any block of text does.** On the same (arm, task,
  density), `filler`, 1,829 characters without structure, *loses* 6.9 points
  (R `[fillerband]`). Filler costs graph reading (§3), so the side
  statistics' own effect lies between +1.6 (against none) and about +8.5
  (against `filler`).
- On the aligned task (ND under `degree`), the primer helps the models that count
  poorly and costs the one that already counts: 1.7B +7.5, 1.7B-T +8.8, 4B
  −6.5, 4B-T +2.5 (q 0.078) (P `[main]`).

**Answer.** Statistics that state the answer raise accuracy where the model
cannot compute it well itself. Statistics that do not state it add about +1.6
points where there is room, and about 0 at ceiling. That is more than an
equally long text without statistics does, which loses 6.9 in the same cells.

## 2. Every primer against no primer

Each primer against no primer on 24 pairings (4 models × 6 tasks), main sweep
(p ≤ .50, 400 graphs per pairing; effects as in P `[main]`). `node_count` and
`cycle_check` count like any task: the model does not know their answer is
constant. A significant change is *truncation-driven* when the share of
responses hitting the budget moves against it by at least half the effect
(C `[pvn]`).

| | filler | components | clustering | rwse | degree | all |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Sig. gain, primer states the answer | – | – | – | – | 3 | 3 |
| Sig. gain, primer does not state it | 2 | 3 | 3 | 1 | 3 | 4 |
| Sig. gain, truncation-driven | 3 | 1 | 3 | 3 | 2 | 2 |
| Gain, not significant | 5 | 4 | 7 | 5 | 4 | 3 |
| Neutral (within ±1) | 8 | 9 | 8 | 8 | 3 | 6 |
| Loss, not significant | 2 | 4 | 3 | 5 | 4 | 4 |
| Sig. loss, truncation-driven | 0 | 2 | 0 | 0 | 2 | 0 |
| Sig. loss, other | 4 | 1 | 0 | 2 | 3 | 2 |
| Largest sig. gain where it does not state the answer | +39.8 1.7B NC | +21.2 1.7B NC | +67.2 1.7B NC | +9.2 4B NC | +8.0 4B NC | +47.0 1.7B NC |
| Same, outside NC | none | +9.2 4B CN | +4.0 4B EE | none | +6.2 1.7B-T CN | +11.5 1.7B EE |
| Largest sig. loss, not truncation-driven | −15.8 1.7B EE | −8.2 1.7B EE | none | −10.0 4B EE | −17.5† 1.7B CC | −10.0 4B EE |

- 16 of the 128 pairings where the primer does not state the answer show a
  significant gain that is not truncation-driven; 11 of them are `node_count`
  in the two plain models, and for plain 4B every primer gives the same +8 to
  +10 there (§4). A further 14 significant gains are truncation-driven.
- `clustering` is the only primer with no significant loss.
- The five gains outside `node_count`: `components` on 4B CN (+9.2; the
  sentence is the same for every graph from p = .20 and works as a cue: the
  model restates the queried node's line in 98.5% of responses against 63.0%,
  P `[compproc]`), `clustering` on 4B EE (+4.0; false alarms 0.16 → 0.08,
  P `[fa]`), `degree` and `all` on 1.7B-T CN (+6.2 each), `all` on 1.7B EE
  (+11.5; false alarms 0.51 → 0.33, P `[fa]`).

Beyond the main sweep (`node_degree`):
- `clustering`: plain 4B at p ≥ .65 +11.3 (P `[clusthi]`); plain 1.7B at
  p ≤ .50 replicated at +3.8, +4.2 (fresh seeds) and +6.2 (fixed mean degree)
  (P `[replic]`); 1.7B-T at p ≥ .65 −5.4 (D `[ddthink]`).
- `rwse`: plain 4B at p ≥ .65 +7.7 (P `[clusthi]`), where it prints only 2.0–2.7
  distinct values per graph (P `[rwse]`).
- `degree`: 1.7B-T at p ≥ .65 +24.8 (D `[ddthink]`); plain 4B +9.3 (P `[clusthi]`).
- `all`: plain 4B at p ≥ .65 −21.3 (P `[clusthi]`).

## 3. `filler`

`filler` names every node ("Node 0 is simply present within the graph G.") and
states no structure. Against no primer, pooled over p ≤ .50 (P `[main]`):

| Task | 1.7B | 1.7B-T | 4B | 4B-T |
|---|---:|---:|---:|---:|
| NC (answer always 40) | **+39.8** | **+9.0** (truncated −7.8) | **+9.2** | +0.5 |
| CC (answer always yes) | +3.0 (truncated −2.8) | **+3.0** (truncated −3.0) | −0.5 | +0.5 |
| ND | −0.5 | −3.0 | −0.5 | +2.0 (q 0.12) |
| EC | −0.5† | −1.5† | +1.2 | **+12.2**† (truncated −13.0) |
| CN | **−6.2** | +2.5 | **−12.5** | +1.2 |
| EE | **−15.8** | −0.5 | **−7.0** | 0.0 |

By density, 100 graphs per cell, uncorrected exact McNemar; each cell is
"effect ·accuracy without a primer", with the truncation change shown when it
is 5 points or more (C `[fillerdens]`):

| Task | Model | p=.10 | p=.20 | p=.35 | p=.50 | p=.65 | p=.75 | p=.85 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NC | 1.7B | **+14** ·2 | **+70** ·0 | **+71** ·28 | +4 ·96 | | | |
| | 1.7B-T | +2 ·98 | +3 ·97 | **+17** (tr −17) ·83 | **+14** (tr −9) ·86 | | | |
| | 4B | **+6** ·92 | **+31** ·69 | 0 ·100 | 0 ·100 | | | |
| | 4B-T | +2 ·98 | 0 ·100 | 0 ·100 | 0 ·100 | | | |
| CC | 1.7B | **+13** (tr −12) ·82 | −1 ·91 | 0 ·91 | 0 ·99 | | | |
| | 1.7B-T | +1 ·99 | +2 ·98 | +5 (tr −5) ·95 | +4 ·96 | | | |
| | 4B | −4 ·99 | +3 ·97 | −1 ·100 | 0 ·100 | | | |
| | 4B-T | +1 ·98 | 0 ·100 | +1 ·99 | 0 ·100 | | | |
| ND | 1.7B | +5 ·91 | −7 ·80 | 0 ·41 | 0 ·30 | −1 ·8 | **−13** ·16 | −2 ·5 |
| | 1.7B-T | +7 ·90 | **−10** ·95 | −4 ·60 | −5 ·60 | **−17** ·52 | −9 ·45 | −10 ·42 |
| | 4B | 0 ·100 | 0 ·99 | 0 ·99 | −2 ·99 | +4 ·82 | −1 ·50 | +3 ·33 |
| | 4B-T | +1 ·96 | +5 (tr −5) ·95 | +3 ·97 | −1 ·100 | 0 ·100 | +1 ·96 | −1 ·96 |
| EC | 1.7B | −3 (tr −16) ·3 | +1 ·2 | 0 (tr −10) ·0 | 0 (tr +12) ·0 | | | |
| | 1.7B-T | −9 (tr +8) ·48 | +1 ·15 | +2 ·0 | 0 ·0 | | | |
| | 4B | +5 ·5 | 0 ·2 | 0 ·1 | 0 ·0 | | | |
| | 4B-T | **+21** (tr −20) ·39 | +12 (tr −16) ·34 | **+15** (tr −15) ·7 | +1 ·0 | | | |
| CN | 1.7B | 0 ·92 | −5 ·91 | −10 ·59 | −10 ·48 | | | |
| | 1.7B-T | +3 ·92 | +1 ·96 | +4 ·68 | +2 ·59 | | | |
| | 4B | **−21** ·93 | **−14** ·88 | −6 ·84 | −9 ·79 | | | |
| | 4B-T | +3 ·95 | −1 ·94 | +2 ·97 | +1 ·96 | | | |
| EE | 1.7B | −5 ·89 | **−20** ·73 | **−32** ·66 | **−6** ·54 | −2 ·68 | −1 ·77 | −1 ·86 |
| | 1.7B-T | −2 ·99 | −2 ·100 | +3 ·97 | −1 ·100 | 0 ·99 | +1 ·98 | −3 ·100 |
| | 4B | 0 ·96 | **−9** ·93 | −8 ·83 | **−11** ·90 | **−12** ·87 | −2 ·87 | −5 ·96 |
| | 4B-T | 0 ·99 | 0 ·100 | 0 ·100 | 0 ·100 | 0 ·100 | 0 ·100 | +2 ·98 |

- Its significant gains are on `node_count`, and on `cycle_check` and
  4B-T `edge_count` where fewer responses hit the budget.
- It raises false alarms on `edge_existence`: 0.51 → 0.69 for 1.7B, 0.16 →
  0.30 for 4B (P `[fa]`).
- On `node_degree` at 400 graphs per density: +3.0 at p = .10 for both 1.7B arms
  (q 0.068, 0.08), and −6.4 (plain) and −11.4 (thinking) at p ≥ .65 (D
  `[ddplain]`, `[ddthink]`), where plain 1.7B answers 39 ("every other node")
  on 85.5% of responses at p = .85 against 61.0% without a primer (D `[ddplain]`).
- Because `filler` itself costs graph reading, a primer-vs-`filler` contrast
  mixes the primer's effect with `filler`'s: `components` on 4B CN is +21.8
  against `filler` and +9.2 against no primer (P `[main]`).

## 4. `node_count`: an off-by-one, and one primer that fixes it

The answer is always 40. % of finished answers that answer 39, p ≤ .50, 400
per cell (369–400 for 1.7B-T); **bold** is a significant change from no primer
(C `[ncanswer]`):

| Primer | 1.7B | 4B | 1.7B-T | 4B-T |
|---|---:|---:|---:|---:|
| none | 68.5 | 9.8 | 0.0 | 0.5 |
| filler | **28.7** | **0.5** | 0.0 | 0.0 |
| components | **47.2** | **0.8** | 0.3 | 0.2 |
| clustering | **1.2** | **0.8** | 0.0 | 0.0 |
| rwse | 66.5 | **0.5** | 0.0 | 0.0 |
| degree | **61.8** | **1.8** | 0.8 | 0.0 |
| all | **21.5** | **0.0** | 0.0 | 0.0 |

By density, % answering 39 (C `[ncdensity]`):

| Model | Primer | p=.10 | p=.20 | p=.35 | p=.50 |
|---|---|---:|---:|---:|---:|
| 1.7B | none | 98 | 100 | 72 | 4 |
| | filler | 84 | 30 | 1 | 0 |
| | components | 81 | 59 | 30 | 19 |
| | clustering | 5 | 0 | 0 | 0 |
| | rwse | 68 | 83 | 45 | **70** |
| | degree | 97 | 95 | 55 | 0 |
| | all | 30 | 48 | 6 | 2 |
| 4B | none | 8 | 31 | 0 | 0 |
| | every primer | 0–6 | 0–1 | 0 | 0 |

Every model copies the id list 0–39 from the encoding's first line and never
counts it. Plain 1.7B then takes one of two paths: it lists the ids one per line
(30 or more lines holding one id each) and reports the last one, 39, or it
answers directly ("This is a total of N nodes") (C `[ncanswer]`):

| Primer | Lists ids one per line | Correct when listing (n) | Correct otherwise (n) | Correct overall |
|---|---:|---:|---:|---:|
| none | 59.5 | 0.8 (238) | 76.5 (162) | 31.5 |
| degree | 53.8 | 1.4 (215) | 81.1 (185) | 38.2 |
| rwse | 19.8 | 1.3 (79) | **41.4** (321) | 33.5 |
| components | 22.5 | 3.3 (90) | 67.1 (310) | 52.8 |
| filler | 12.2 | 0.0 (49) | 81.2 (351) | 71.2 |
| all | 3.5 | 7.1 (14) | 81.1 (386) | 78.5 |
| clustering | 2.8 | 54.5 (11) | **100.0** (389) | 98.8 |

- The listing path is almost never correct.
- `rwse` and `clustering` both cut that path. They split on the direct path:
  40 on 41.4% of answers under `rwse`, on all 389 under `clustering`. Primer
  length does not explain it (`all` contains the RWSE text and reaches 81.1%),
  nor does naming every node (all three do).
- Without a primer, the error fades with density (98% at p = .10, 4% at
  p = .50); under `rwse` it does not (70% at p = .50).
- Plain 4B never lists the ids. It errs on the direct path only ("numbered from
  0 to 39, which is a total of 39 nodes", C `[ncsample]`; most of its 39s are a
  bare "A: 39"), at p = .10 and .20, and every
  primer removes it, `filler` included (−8.0 to −9.8, all significant).
- The thinking models do not make the error (at most 0.8%).

## 5. `cycle_check`: do correct answers rest on real cycles?

The gold answer is "yes" on every 40-node graph. Outcomes, p ≤ .50, 400 per
cell, % correct / answering no / truncated (C `[ccoutcome]`):

| Model | none | filler | components | clustering | rwse | degree | all |
|---|---|---|---|---|---|---|---|
| 1.7B | 90.8 / 0.2 / 9.0 | 93.8 / 0.0 / 6.2 | 92.8 / 0.8 / 6.5 | 99.2 / 0.0 / 0.8 | 90.5 / 7.8 / 1.8 | 73.2 / 9.2 / 17.2 | 97.8 / 1.2 / 1.0 |
| 1.7B-T | 97.0 / 0.0 / 3.0 | 100 / 0.0 / 0.0 | 89.5 / 0.0 / 10.5 | 96.5 / 0.0 / 3.5 | 99.2 / 0.0 / 0.8 | 75.2 / 0.5 / 24.2 | 96.2 / 0.0 / 3.8 |
| 4B | 99.0 / 0.2 / 0.8 | 98.5 / 0.2 / 1.2 | 97.0 / 1.5 / 1.5 | 99.5 / 0.0 / 0.5 | 98.8 / 0.0 / 1.2 | 96.8 / 1.8 / 1.5 | 98.8 / 0.8 / 0.5 |
| 4B-T | 99.2 / 0.0 / 0.8 | 99.8 / 0.0 / 0.2 | 96.0 / 0.0 / 4.0 | 99.2 / 0.0 / 0.8 | 99.5 / 0.0 / 0.5 | 93.8 / 0.8 / 5.5 | 98.8 / 0.0 / 1.2 |

Plain 1.7B's "no" answers rise significantly under `rwse` (+7.5) and `degree`
(+12.3), and follow wrong reasoning (C `[ccno]`, `[ccsample]`):
- Under `degree` (37 finished "no" answers), 97% mention an odd degree sum or
  the handshaking lemma: it sums the 40 stated degrees, gets an odd total and
  misapplies the lemma.
- Under `rwse` (31), 68% call the graph *directed* ("The graph is a directed
  graph (not necessarily undirected)"). All mention return probabilities, but
  the samples set them aside as "not directly relevant"; none mention an odd
  sum.
- The other models answer no to at most 1.8% of graphs under any primer.

**It is not the sentences themselves.** `all` prints the same return
probabilities *and* every degree. Yet 1.7B answers no to only 1.3% of graphs
under it (5 answers), and none of those calls the graph directed or cites an
odd degree sum (0% and 0%, C `[ccno]`). Three explanations fit:
- **The statistic shown alone sets the frame.** Under `rwse`, return
  probabilities are the only thing said about each node, and random-walk
  probabilities describe a directed process. Under `degree`, 40 numbers to
  add invite the handshaking argument. In `all`, each node's line opens with
  its degree and mixes three statistics, so neither frame dominates.
- **Summing is harder in `all`.** The degrees sit inside longer sentences, so
  the model may not try to add them. That removes the odd sum but not the
  directed reading.
- **Chance.** 5 "no" answers under `all` is a small sample.

Only the first explains both errors at once.

### 5.1 What the correct answers rest on

Each cycle a correct, finished answer names (for the thinking arms, in the text
after `</think>`) is checked step by step against the edges of its prompt (§7).
An answer rests on a **real** cycle if it names one and does not reject it,
wherever it appears; if it rejects a real cycle and rests on an invented one,
it rests on the **invented** one. % of correct answers (C `[ccanswer]`):

| Model | Primer | n | Real | Invented | Not a cycle (length-2, retraced) | None named (edge-count argument) | Rejected a real cycle |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.7B | none | 363 | 36.9 | 43.5 | 16.0 | 3.6 (0.8) | 1.1 |
| | filler | 375 | 33.3 | 55.5 | 9.6 | 1.6 (0.0) | 0.3 |
| | components | 371 | 27.8 | 40.4 | 25.3 | 6.5 (4.3) | 0.3 |
| | clustering | 397 | 39.5 | 48.1 | 7.8 | 4.5 (1.0) | 0.0 |
| | rwse | 362 | 33.7 | 40.3 | 19.9 | 6.1 (1.7) | 0.0 |
| | degree | 293 | 13.0 | 20.5 | 1.4 | 65.2 (55.6) | 0.0 |
| | all | 391 | 38.4 | 50.4 | 7.9 | 3.3 (1.3) | 0.0 |
| 1.7B-T | none | 388 | 79.9 | 17.8 | 1.0 | 1.3 (0.5) | 0.0 |
| | filler | 400 | 77.2 | 19.8 | 2.5 | 0.5 (0.0) | 0.0 |
| | components | 358 | 74.9 | 22.6 | 0.0 | 2.5 (2.0) | 0.0 |
| | clustering | 386 | 80.6 | 19.2 | 0.0 | 0.3 (0.3) | 0.0 |
| | rwse | 397 | 68.0 | 20.7 | 10.6 | 0.8 (0.5) | 0.0 |
| | degree | 301 | 61.5 | 29.6 | 0.0 | 9.0 (9.0) | 0.0 |
| | all | 385 | 73.0 | 21.8 | 3.1 | 2.1 (1.8) | 0.0 |
| 4B | none | 396 | 34.1 | 23.0 | 6.8 | 36.1 (25.0) | 0.3 |
| | filler | 394 | 44.2 | 28.2 | 5.3 | 22.3 (2.3) | 0.5 |
| | components | 388 | 39.2 | 18.6 | 3.4 | 38.9 (26.5) | 0.3 |
| | clustering | 398 | 48.7 | 31.2 | 1.0 | 19.1 (5.8) | 0.3 |
| | rwse | 395 | 33.2 | 20.8 | 38.7 | 7.3 (0.5) | 0.0 |
| | degree | 387 | 13.4 | 3.9 | 2.1 | 80.6 (65.6) | 0.3 |
| | all | 395 | 45.1 | 33.4 | 6.6 | 14.9 (0.8) | 0.3 |
| 4B-T | none | 397 | 96.5 | 3.3 | 0.0 | 0.3 (0.0) | 0.0 |
| | filler | 399 | 94.2 | 5.0 | 0.5 | 0.3 (0.0) | 0.0 |
| | components | 384 | 92.2 | 4.4 | 0.0 | 3.4 (2.9) | 0.0 |
| | clustering | 397 | 95.5 | 4.5 | 0.0 | 0.0 (0.0) | 0.0 |
| | rwse | 398 | 55.0 | 2.5 | 42.5 | 0.0 (0.0) | 0.0 |
| | degree | 375 | 90.9 | 5.1 | 0.0 | 4.0 (2.7) | 0.0 |
| | all | 395 | 92.7 | 6.3 | 0.0 | 1.0 (1.0) | 0.0 |

The easiest table to read is the change in the share resting on a real cycle.
Paired on graphs where both answers are correct and finished; **bold** q < 0.05
(C `[cctest]`):

| Model | none | filler | components | clustering | rwse | degree | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.7B | 36.9 | −3.2 | **−9.1** | +2.2 | −4.3 | **−22.1** | +1.7 |
| 1.7B-T | 79.9 | −2.8 | −4.3 | +0.8 | **−11.4** | **−18.1** | −6.7 |
| 4B | 34.1 | **+9.7** | +4.7 | **+14.7** | −1.0 | **−20.6** | **+11.5** |
| 4B-T | 96.5 | −2.0 | **−4.5** | −1.0 | **−41.0** | **−5.6** | −3.6 |

% of correct answers asserting at least one invented cycle, and the paired
change (C `[ccinvent]`, `[cctest]`):

| Model | none | filler | components | clustering | rwse | degree | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.7B | 45.5 | **+12.4** | −4.1 | +5.8 | 0.0 | **−25.1** | **+10.7** |
| 1.7B-T | 39.2 | +7.7 | −1.7 | **+13.0** | −0.5 | +1.7 | **+12.6** |
| 4B | 23.5 | +5.6 | −3.1 | **+9.6** | −3.3 | **−19.5** | **+11.3** |
| 4B-T | 5.8 | +0.3 | +1.3 | +4.8 | −1.0 | +0.8 | +4.6 |

- Thinking is what makes the reasoning real: 36.9% → 79.9% for 1.7B and 34.1% →
  96.5% for 4B without a primer.
- No primer lowers fabrication while keeping real cycle-finding. `degree` lowers
  invented cycles for the plain models (−25.1, −19.5) by moving them to the
  edge-count argument (more than n − 1 edges, so a cycle exists); real cycles
  fall by the same amount (−22.1, −20.6). For 1.7B-T it raises the share resting
  on an invented cycle (+12.6, C `[cctest]`).
- `clustering` and `all` raise invented cycles significantly for three models
  each; for plain 4B they raise real cycles as well, moving it from naming no
  cycle (36.1% without a primer) to naming one.
- Rejecting a real cycle and resting on an invented one is rare (at most 1.1%);
  catching one's own invented cycle is too (at most 3.0%, C `[ccinvent]`).

### 5.2 Where the invented edges are

Asserted invented cycles, all primers pooled (C `[ccwhere]`):

| Model | Invented steps | Closing step invented | Inner step invented | Fake neighbour in the line of node a±1 (chance) | One off a real neighbour (chance) |
|---|---:|---:|---:|---:|---:|
| 1.7B | 5,398 | 56.6% | 21.6% | 38.1% (41.8%) | 32.4% (44.8%) |
| 1.7B-T | 2,838 | 56.8% | 16.1% | 38.9% (44.1%) | 35.8% (48.6%) |
| 4B | 1,439 | 62.7% | 14.4% | 44.2% (43.1%) | 36.8% (47.5%) |
| 4B-T | 333 | 43.9% | 23.4% | 55.6% (48.1%) | 49.8% (55.3%) |

- The invented edge is mostly the one that closes the walk back to its start:
  the models walk real edges, then assert the walk closes.
- The fake neighbour is not taken from the neighbouring line, nor off by one,
  more often than chance.
- The share asserting an invented cycle is flat across density: 1.7B 50.9 /
  52.3 / 43.4 / 39.3% at p = .10 / .20 / .35 / .50 (C `[ccdensity]`).

### 5.3 "Cycles of length 2"

A length-2 claim is a walk a → b → a: 2 nodes, and one edge walked twice. The
graphs are undirected and simple, and the benchmark's gold uses
`nx.find_cycle` on the undirected graph (`talk_like_a_graph/graph_tasks.py:53`),
so the shortest cycle is a triangle. The incident encoding lists every edge
under both endpoints, and the models read the two lines as two edges ("nodes 0
and 4 are directly connected in both directions (0 → 4 and 4 → 0), forming a
cycle of length 2").

`rwse` raises the share of correct answers resting on a non-cycle (a length-2
or retraced walk) for every model: 4B-T 0.0 → 42.5%, 4B 6.8 → 38.7%, 1.7B-T
1.0 → 10.6%, 1.7B 16.0 → 19.9% (C `[ccanswer]`).

The obvious link is the primer's wording, "return probability … after 2
steps": a walk that returns in 2 steps goes to a neighbour and back. But `all`
prints the same words and hardly produces these claims: 4B-T 0.0%, 4B 6.6%,
1.7B-T 3.1%, 1.7B 7.9% (C `[ccanswer]`). So the wording alone does not cause
them. As with the directed reading (§5), what differs is that under `rwse` the
return probabilities are the only statistic. A graph described only by how
often a walk returns may lead the model to treat "going and coming back" as
the cycle it is asked for.

### 5.4 Thinking traces

The same reading on the text before `</think>` of the correct answers
(C `[ccthink]`), % naming a real cycle and keeping it / asserting an invented
one / withdrawing an invented one:

| Model | none | filler | components | clustering | rwse | degree | all |
|---|---|---|---|---|---|---|---|
| 1.7B-T | 86.9 / 53.4 / 9.0 | 80.8 / 59.0 / 12.5 | 78.8 / 44.1 / 9.2 | 82.4 / 62.7 / 10.6 | 70.5 / 48.1 / 16.4 | 70.4 / 55.5 / 13.0 | 73.8 / 66.8 / 14.8 |
| 4B-T | 98.5 / 25.4 / 23.7 | 96.7 / 22.3 / 19.5 | 94.5 / 26.0 / 19.3 | 96.7 / 31.2 / 14.1 | 56.8 / 24.6 / 19.8 | 93.9 / 22.7 / 11.2 | 94.4 / 33.4 / 16.5 |

- Traces invent more than final answers do (53% against 39% for 1.7B-T, 25%
  against 6% for 4B-T), and part of it is withdrawn in the trace.
- Paired change in traces asserting an invented cycle: 1.7B-T `components`
  **−9.2**, `clustering` **+9.0**, `all` **+13.4**; nothing significant for 4B-T.

## 6. Fabricated edges in the other tasks

A statement "Node x is connected to (nodes) a, b, c" (or "node x's neighbours
are …") claims those edges (§7).

### 6.1 `edge_existence`

% of non-edge pairs where the model states the pair as an edge, and the paired
change (C `[eeclaims]`):

| Model | none | filler | components | clustering | rwse | degree | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.7B | 13.7 | **+9.6** | +3.8 | **−9.2** | **−11.0** | **−10.2** | **−9.6** |
| 1.7B-T | 0.0 | +0.7 | +0.3 | +0.7 | +0.7 | +0.7 | +0.3 |
| 4B | 4.1 | +1.4 | −1.0 | −2.4 | +1.4 | +0.3 | +1.7 |
| 4B-T | 0.0 | 0.0 | +0.3 | 0.0 | 0.0 | 0.0 | 0.0 |

False alarms (% of non-edge pairs answered yes) and the share of them that state
the queried non-edge (C `[eeclaims]`):

| Model | none | filler | components | clustering | rwse | degree | all |
|---|---|---|---|---|---|---|---|
| 1.7B | 40.3 / 33.9 | 61.8 / 37.6 | 51.5 / 33.8 | 35.2 / 11.7 | 32.5 / 8.4 | 34.5 / 9.9 | 24.6 / 15.3 |
| 4B | 12.6 / 21.6 | 22.5 / 24.2 | 16.7 / 18.4 | 7.5 / 22.7 | 26.6 / 20.5 | 18.9 / 21.8 | 26.4 / 22.1 |

- A stated fake edge backs a third of plain 1.7B's false alarms and a fifth of
  plain 4B's; the rest answer yes without naming the edge. Under `clustering`,
  `rwse`, `degree` and `all`, plain 1.7B stops stating the fake edge (8–15% of
  its false alarms) but still answers yes to 25–35% of non-edge pairs.
- The models almost never answer no to a true edge (at most 1.9% at p ≤ .50);
  misses are read at high density below.
- Any statement of an edge that does not exist: 1.7B 14.5% (`filler` **+8.0**;
  `clustering`, `degree`, `all` **−8**); 1.7B-T `rwse` **+7.0**; 4B-T `rwse`,
  `degree`, `all` **+2.3** from 0.5% (C `[eeclaims]`).

**A shared neighbour is read as an edge.** % of non-edge pairs answered yes, by
how many neighbours the two nodes share (counted on the graph), p ≤ .50, all
primers pooled; pairs in brackets (C `[eeshared]`):

| Model | 0 shared | 1 | 2–3 | 4 or more |
|---|---:|---:|---:|---:|
| 1.7B | 1.7 (524) | 34.3 (350) | 47.3 (448) | 65.9 (728) |
| 1.7B-T | 0.2 | 2.6 | 5.1 | 2.3 |
| 4B | 0.2 (524) | 4.6 (350) | 19.0 (447) | 38.8 (727) |
| 4B-T | 0.0 | 0.0 | 0.0 | 0.0 |

- It is not density. Within p = .20 alone, 1.7B goes 2 → 37 → 46 → 81% and 4B
  0 → 5 → 17 → 31%. Without a primer, the pairs a model calls edges share more
  neighbours than those it correctly calls non-edges, with the answers permuted
  within each density: +4.5 for 1.7B (p = 0.0005), +2.8 for 4B (p = 0.014).
- **It is not only the pair's degree.** Pairs that share many neighbours also
  have long lines to read: within density, shared neighbours and the pair's
  degree sum correlate 0.40 (p = .10) to 0.87 (p = .50). With the degree sum
  held fixed (quartile within density), shared neighbours still predict a false
  yes for 1.7B: p = .0035 without a primer, .0005 over all seven conditions.
  For 4B they do over all seven conditions (p = .0005), but not without a
  primer alone (p = .081, 37 false alarms). The degree sum also predicts a
  false yes with shared neighbours held fixed (1.7B p = .011 without a primer;
  both models p = .0005 over all seven) (R `[eedegree]`).
- **So both matter.** A pair is called an edge more often when the two nodes
  share neighbours, and also when their lines are long.
- A typical answer: "Node 18 is connected to Node 17, and Node 9 is connected to
  Node 17. Therefore, there is an edge between Node 18 and Node 9."

% answered yes on non-edge pairs sharing 4 or more neighbours, 104 pairs per
cell; **bold** q < 0.05 against no primer (C `[eeshared]`):

| Model | none | filler | components | clustering | rwse | degree | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.7B | 73.1 | **99.0** | **86.5** | 61.5 | **51.0** | **49.0** | **41.3** |
| 4B | 24.0 | **43.3** | **36.5** | 14.4 | **56.7** | **41.7** | **54.8** |

- `filler` and `components` raise the shared-neighbour false alarm for both
  plain models. `rwse`, `degree` and `all` lower it for 1.7B and raise it for
  4B. `clustering` lowers it for both, but not significantly (−11.5, −9.6).

**What the false alarms say.** Each false alarm is sorted by the first reason it
gives: it states the queried pair as an edge (as `[eeclaims]`); or it claims the
other node is listed in an endpoint's list ("Node 12 is listed among these
connected nodes", right after quoting the list correctly); or neither. All
primers pooled (C `[eewhy]`):

| Model | Densities | False alarms | States the edge | Claims membership | Neither |
|---|---|---:|---:|---:|---:|
| 1.7B | p ≤ .50 | 821 | 24.4 | 12.5 | 63.1 |
| 1.7B | p ≥ .65 | 445 | 48.5 | 11.0 | 40.4 |
| 4B | p ≤ .50 | 384 | 21.6 | 41.7 | 36.7 |
| 4B | p ≥ .65 | 200 | 11.5 | 74.5 | 14.0 |

The membership claim is plain 4B's main false alarm. Its share of 4B's false
alarms at p ≤ .50: none 62.2, `components` 63.3, `degree` 58.2, `rwse` 37.2,
`filler` 33.3, `clustering` 27.3, `all` 22.1%. As a share of all non-edge
pairs, `clustering` is the only primer that changes it significantly (4B,
8.5 → 2.4%) (C `[eewhy]`). 1.7B-T's false alarms are few (50 at p ≤ .50), and
92% of them give neither reason; 4B-T has none at p ≤ .50 (C `[eewhy]`).

**What the misses say.** 75 misses in all, 68 of them at p ≥ .65 (C `[eemiss]`,
`[eesample]`):
- 1.7B-T (21): 95% say the other node is not listed; 90% have written down a
  list for one endpoint that contains the other; 71% resolve the mismatch by a
  rule that both lists must agree ("the edge is only present if both nodes are
  mutually connected", "this discrepancy suggests an inconsistency in the
  data"). One list is misread, and the model trusts the misreading.
- 4B (40): 75% say the other node is not listed, 45% while quoting a list that
  contains it ("Node 37 is connected to: 0, 1, 3, 4, 6, … Node 4 is **not** in
  this list").
- 4B-T (3, all at p = .85): 2 of 3 rule that both lists must agree ("Node 15's
  connections do not include Node 33, while Node 33's connections do include
  Node 15. This discrepancy suggests a possible inconsistency").
- 1.7B (11): 36% say the other node is not listed; none invoke the rule.

**High density (p = .65–.85).** 73 non-edges and 227 edges per model and
primer. False alarms (%) / % stating the queried non-edge (C `[eehigh]`):

| Model | none | filler | components | clustering | rwse | degree | all |
|---|---|---|---|---|---|---|---|
| 1.7B | 94.5 / 69.9 | 100.0 / 68.5 | 98.6 / 68.5 | 86.3 / 20.5 | 90.4 / 23.3 | 75.3 / 27.4 | 64.4 / 17.8 |
| 4B | 30.1 / 4.1 | 57.5 / 15.1 | 45.8 / 4.2 | 11.0 / 0.0 | 34.2 / 4.1 | 50.0 / 4.2 | 47.2 / 4.2 |

- `clustering`, `rwse`, `degree` and `all` stop plain 1.7B from stating the fake
  edge (**−49.3**, **−46.6**, **−42.5**, **−52.1**), but it still answers yes to
  64–90% of non-edges.
- The thinking models answer yes to at most 4.1% of non-edges.
- Misses stay rare even with 76% of queried pairs being edges: at most 3.5% of
  true edges (4B without a primer).
- Any stated edge that does not exist: 1.7B-T 19.0% without a primer, `rwse`
  **+14.3**; 1.7B `all` **−19.3**, `degree` **−8.1**; 4B `filler` **+3.7**.

### 6.2 `node_degree`

The longest list the response states for the queried node (its reading of the
node's line). Why the wrong answers are wrong (C `[ndlist]`):

| Model | Primer | Wrong answers | Misread (list wrong) | Miscounted (list right) | No list |
|---|---|---:|---:|---:|---:|
| 1.7B | none | 158 | 43.0 | 6.3 | 50.6 |
| | filler | 160 | 35.0 | 4.4 | 60.6 |
| | components | 151 | 12.6 | 2.6 | 84.8 |
| | clustering | 146 | 63.0 | 15.8 | 21.2 |
| | rwse | 141 | 56.7 | 10.6 | 32.6 |
| | degree | 128 | 57.0 | 14.1 | 28.9 |
| | all | 117 | 44.4 | 21.4 | 34.2 |
| 1.7B-T | none, filler, components, clustering, rwse | 82–106 | 97.9–100 | 0–2.1 | 0 |
| | degree | 39 | 79.5 | 17.9 | 2.6 |
| | all | 45 | 86.7 | 8.9 | 4.4 |
| 4B | degree | 29 | 6.9 | 31.0 | 62.1 |
| | all | 35 | 28.6 | 37.1 | 34.3 |

- Misreading is mostly omission: of plain 1.7B's stated lists without a primer,
  24.6% miss a neighbour and 3.3% invent one (C `[ndlist]`).
- Stated lists with an invented neighbour, paired change: 1.7B `clustering`
  **+5.8**, `rwse` **+6.0**; 1.7B-T `rwse` **+4.8** (C `[ndlist]`).
- The other 4B and all 4B-T cells have 0–12 wrong answers.
- Wrong answers mostly undercount, as omission predicts: 1.7B 73.4% without a
  primer (55.6–73.4% across primers), 1.7B-T 91.3% (78.0–91.3%). Plain 4B under
  `degree` and `all` mostly overcounts (undercounts 37.9% of 29 and 34.3% of
  35), in line with copying a nearby node's stated degree (P `[copyerr]`)
  (C `[ndsign]`).
- As a share of all answers, `rwse`, `degree` and `all` cut undercounting for
  both 1.7B models: 1.7B **−8.3**, **−5.5**, **−12.7**; 1.7B-T **−5.1**,
  **−11.2**, **−11.9** (C `[ndsign]`).

**High density (p = .65–.85)**, 300 answers per cell (C `[ndhigh]`):

| Model | Primer | Wrong answers | States the node's list | Misread | Miscounted | No list |
|---|---|---:|---:|---:|---:|---:|
| 1.7B | none | 270 | 5.0 | 2.2 | 1.9 | 95.9 |
| | clustering | 281 | 25.3 | 17.8 | 6.8 | 75.4 |
| | rwse | 287 | 32.3 | 24.7 | 6.6 | 68.6 |
| | all | 263 | 25.3 | 14.1 | 9.9 | 76.0 |
| 1.7B-T | none | 156 | 100.0 | 99.4 | 0.6 | 0.0 |
| | degree | 52 | 99.6 | 96.2 | 3.8 | 0.0 |
| 4B | none | 135 | 28.3 | 0.7 | 23.7 | 75.6 |
| | filler | 129 | 80.7 | 4.7 | 74.4 | 20.9 |
| | clustering | 101 | 41.7 | 4.0 | 37.6 | 58.4 |
| | degree | 107 | 30.0 | 0.9 | 57.9 | 41.1 |
| | all | 199 | 54.7 | 1.5 | 68.3 | 30.2 |

- Plain 1.7B mostly answers without writing the list; 4B-T has 4–12 wrong
  answers per cell.
- Plain 4B reads the list right and counts it wrong: under `filler`, `degree`
  and `all`, most of its wrong answers state the correct list.
- 1.7B-T's lists invent a neighbour in 33.9% of answers without a primer
  (3.3% at p ≤ .50), and every primer raises it: `filler` **+20.5**,
  `components` **+11.0**, `clustering` **+19.0**, `rwse` **+23.1**, `degree`
  **+6.1**, `all` +5.5. Plain 1.7B, as a share of all its answers: `clustering`
  **+12.4**, `rwse` **+21.4**, `all` **+9.0** from 1.3%.

### 6.3 `connected_nodes`

% of answers containing a node that is not a neighbour, and the paired change
(C `[cnset]`):

| Model | none | filler | components | clustering | rwse | degree | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.7B | 9.8 | **+7.0** | **+3.2** | +3.2 | **+6.2** | −1.2 | +4.0 |
| 1.7B-T | 5.6 | 0.0 | +1.0 | +1.0 | **+4.8** | −0.8 | −0.3 |
| 4B | 6.5 | **+14.8** | **−3.5** | +3.5 | +2.8 | **+5.8** | **+7.3** |
| 4B-T | 0.5 | −0.3 | −0.3 | +0.3 | +0.3 | +0.3 | +0.3 |

Missed neighbours without a primer: 1.7B 22.8%, 1.7B-T 16.5%, 4B 9.0%, 4B-T
1.0% (C `[cnset]`).

**Where the errors come from**, p ≤ .50, all primers pooled (C `[cnsource]`).
Invented neighbours other than the queried node q, % in each class, with chance
in brackets: the share of q's non-neighbours in that class, averaged over the
invented neighbours (so weighted toward the dense graphs where they occur); and
the share of all invented neighbours that are q itself:

| Model | Invented, not q | Neighbour of a neighbour | In the line of node q±1 | One off a true neighbour's id | q itself |
|---|---:|---:|---:|---:|---:|
| 1.7B | 1,345 | 97.2 (97) | 75.8 (66) | 63.3 (70) | 2.9 |
| 1.7B-T | 505 | 95.6 (95) | 77.0 (65) | 63.6 (67) | 5.3 |
| 4B | 536 | 91.0 (85) | 54.5 (49) | 43.5 (52) | 22.7 |
| 4B-T | 27 | 96.3 (89) | 81.5 (52) | 59.3 (57) | 0.0 |

Missed neighbours, % by their place on the node's line (ids are in ascending
order):

| Model | Missed | The first one | The last one | Below q (chance) |
|---|---:|---:|---:|---:|
| 1.7B | 868 | 3.2 | 39.1 | 23.6 (45) |
| 1.7B-T | 846 | 2.5 | 21.0 | 34.0 (46) |
| 4B | 287 | 61.3 | 3.8 | 93.0 (43) |
| 4B-T | 53 | 3.8 | 17.0 | 62.3 (68) |

- Invented neighbours sit in the line of node q−1 or q+1 somewhat more often
  than chance (1.7B 75.8% against 66%); being a neighbour's neighbour is at
  chance, and one off a true neighbour's id is below it.
- The 1.7B models drop the end of the line; plain 4B drops the start, and puts
  q itself in its answer (2.8% without a primer; `filler` **+10.2**, `rwse`
  **+3.5**, `all` **+4.8**, `components` **−2.2**).
- `clustering` cuts plain 1.7B's dropped last neighbour (14.5 → 7.5%,
  **−7.0**); `components` cuts plain 4B's dropped first neighbour (8.2 → 2.0%,
  **−6.2**) (C `[cnsource]`).

### 6.4 `edge_count`

The per-node degree table a response lists, read by
`response_patterns.edge_chain` (`[rpchain]`; §7 on how the table is read). % of
all responses whose table has a wrong value, and the paired change
(C `[ecchain]`):

| Model | none | filler | components | clustering | rwse | degree | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.7B | 51.5 | **+12.2** | **−7.5** | **−24.8** | **−40.2** | **−24.0** | **−27.5** |
| 1.7B-T | 46.5 | +3.5 | −0.5 | **+8.8** | **−8.5** | **+19.5** | **+9.5** |
| 4B | 80.0 | **+10.0** | **−15.0** | **+13.8** | **+15.2** | **−43.5** | **+5.8** |
| 4B-T | 12.0 | +1.5 | +3.5 | +4.5 | +0.2 | **+44.2** | **+66.5** |

Primers also change whether a table is listed at all. Among the responses that
list one, % with a wrong value, not tested (C `[ectable]`):

| Model | none | filler | components | clustering | rwse | degree | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.7B | 98.6 | 87.9 | 96.2 | 99.1 | 88.2 | 31.0 | 44.4 |
| 1.7B-T | 61.8 | 61.0 | 60.7 | 64.8 | 54.1 | 66.2 | 56.4 |
| 4B | 81.2 | 90.2 | 70.8 | 95.7 | 96.5 | 40.9 | 85.8 |
| 4B-T | 14.3 | 14.6 | 17.4 | 17.9 | 13.8 | 56.4 | 78.7 |

Under `degree` and `all`, 4B-T miscopies degrees out of the primer: one response
lists "Node 27: degree 6" where the true degree, which the primer states, is 9.

## 7. Method

`scripts/check_cycle_claims.py` reads the responses and checks every claim
against the graph parsed from the response's own prompt (edge counts match the
prompt row's for every graph).

- **A named cycle:** an arrow chain in any notation ("0 → 3 → 8 → 0",
  "0-3-8-0", `\rightarrow`); a node list ("nodes 0, 3 and 8 form a triangle",
  "a cycle through nodes …"); or a closing chain of "a is connected to b"
  statements. A chain that runs into a loop without returning to its start (a
  lasso) is judged on its loop.
- **Its verdict:** real (every step an edge, and its distinct edges close a
  cycle), invented (some step is not an edge), length-2, or retraced (every step
  an edge, but it only goes back the way it came).
- **A rejection:** the text after the cycle, up to the next one or the next
  section break (250 characters at most), says so ("not a cycle", "repeats node
  2", "invalid"); "not a simple cycle" is a concession, not a rejection.
- **An edge statement:** "Node x is connected to (nodes) a, b, c" or "node x's
  neighbours are …". Not a claim when it follows "check if", "whether",
  "maybe" or "imply that", or is followed in the same sentence by "?", "via" or
  "through". A node is never counted as its own neighbour, and a list stops
  before an item that opens the next clause ("…, and Node 31 is …").
- **The `node_count` path:** an answer lists the ids one per line when 30 or
  more of its lines (after `</think>`) hold one id each ("- 7"); the answer
  given is the scored prediction.
- **Which responses:** correct finished answers for `cycle_check`; finished
  answers for `edge_existence`, `node_degree`, `connected_nodes` and
  `node_count`, with the thinking arms read including their trace for the first
  two; all responses for `edge_count`. A response repeated across shards counts
  once, the first, as in `primer_findings.load_runs`.
- **High density:** the p = .65–.85 runs (`edge_existence` and `node_degree`
  only) are read the same way and reported in their own blocks (`[eehigh]`,
  `[ndhigh]`); every other block is p ≤ .50.
- **Why an answer is wrong:** a *membership claim* is "(node) v is listed /
  among / included / present / in the (list)" for an endpoint, or the same with
  "not", in the final answer, skipped after "check if" or "whether"; *both lists
  must agree* is "discrepancy", "inconsistent", "contradict", "mutual" or
  "bidirectional" anywhere in the response. Shared neighbours are counted on the
  graph; a keyword reading of shared-neighbour arguments was dropped because it
  also matched bare claims such as "both nodes are connected". A `cycle_check`
  "no" is read for an odd degree sum or the handshaking lemma, return
  probabilities or self-loops, and a *directed* graph. These are mentions, not
  parsed arguments; `[ccsample]` and `[eesample]` print examples.
- **The degree table (`edge_count`):** `response_patterns.degree_table` reads
  "Node 3: 12", "Node 3: degree 12", "Node 3 has 12 …" and markdown table rows
  under a count column (degree, count, number of, connections), skipping
  running totals ("total up to Node 18: 48"), decimals, and any column that
  lists neighbours ("3, 7, 12"). A truncated response with correct but
  incomplete values is "cut". Against the reading of `87fbf80`, this changes
  216 of 11,200 responses, 132 of them truncated responses no longer counted as
  missing nodes. The reading of `795f5a1` also took a "Connections" column that
  lists neighbours as a count column, so a node with one neighbour got that
  neighbour's id as its degree; that flagged 168 correct tables as wrong and is
  fixed here.
- **Tests:** each primer against none on the same graphs, exact McNemar
  (`graphtalk.scoring.mcnemar`), Benjamini–Hochberg over the primers within
  each model (`primer_findings.bh`).

**Validation.** Each run prints random samples of every rule
(`[ccsample]`, `[clsample]`, `[ncsample]`, `[eesample]`); every sample was
checked against the prompt's lines. A 37-case self-check on a toy graph runs
first and covers each rule that sampling showed to need one; the degree-table
reading is tested in `tests/test_response_patterns.py`.

**Limits.**
- The readings are regex-based and spot-checked, not hand-labelled. Hedged or
  unusual phrasings can still pass as claims, and unparsed cycle formats fall
  into "none named".
- Thinking-trace statements include exploration the model later corrects.
- For `cycle_check`, only the final answer of the thinking arms is judged; the
  trace is reported separately (§5.4).
- The shares "of false alarms" and "among responses listing a table" are not
  tested; paired tests use graphs where both responses finished.
- At high density only 73 non-edges per model and primer are queried.

**Reproduce:**

    PYTHONPATH=. python scripts/check_cycle_claims.py --csv-dir outputs/n40-sweep \
        > outputs/n40-sweep/check_cycle_claims.txt

`scripts/response_patterns.py`, whose `degree_table` and `edge_chain` §6.4
uses, has its own output (pattern shifts per primer; not cited here):

    PYTHONPATH=. python scripts/response_patterns.py --csv-dir outputs/n40-sweep \
        > outputs/n40-sweep/response_patterns.txt

## 8. Open questions

- **Why `clustering` fixes plain 1.7B's `node_count` and `rwse` does not.** The
  choice between writing 40 and 39 on the direct path is a single token. A GPU
  probe of P("40") against P("39") after "This is a total of", under each primer
  and with the two primers' number formats swapped, would locate it.
- **Why `rwse` alone causes length-2 claims when `all` does not.** Both print
  "return probability … after 2 steps". Showing the return probabilities with
  a second statistic other than degree, or rewording them, would separate "the
  only statistic shown" from the wording.
- **Missed edges on `edge_existence`.** Even at p = .65–.85, where 76% of
  queried pairs are edges, no model answers no to more than 3.5% of true edges
  (75 misses in all), too few to test a primer against.
- **Acyclic graphs.** Every 40-node graph has a cycle, so a false "yes" on
  `cycle_check` cannot be measured here, and neither can whether a primer helps
  a model recognise a graph without one. That needs acyclic 40-node graphs,
  which means new runs.
- **Whether `rwse` makes the model read the graph as directed.** 68% of plain
  1.7B's `rwse` "no" answers on `cycle_check` call the graph directed, and none
  of its `all` answers do. Checking the same words in its answers to the other
  tasks, and rewording the primer, would show whether the "random walk"
  framing causes it.
