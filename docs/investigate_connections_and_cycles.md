# Investigating connections and cycles

What the primers change in the 40-node sweep, primer by primer, and whether the
models' correct answers rest on the graph or on claims about the graph that are
false. Research question: when a black-box language model reads a textual
graph, do explicit structural statistics improve execution of graph queries?

Runs: the 40-node sweep (`runs/qwen3-{1.7b,4b}[-think].densfull40*`, Erdős–Rényi
graphs with 40 nodes, p = .10–.50, 100 graphs per density) and its high-density
extension (p = .65–.85); the `cycle_check` clean-condition runs
(`runs/qwen3-0.6b[-think].cc500*`, 500 published graphs of 5–19 nodes, 84 of
them acyclic). Models: Qwen3-1.7B and Qwen3-4B with thinking off (plain) and on
(-T), and Qwen3-0.6B for `cc500`.

This file is not in `docs/results/`, and `tests/test_results_docs.py` does not
check it. Every number names its source:

| Tag | Source |
|---|---|
| **P** `[tag]` | `csv2/raw-trends/primer_findings.txt` (`scripts/primer_findings.py`), or `csv2/raw-trends/primer_cells.csv` |
| **D** `[tag]` | `csv2/density-followups/density_followups.txt` (`scripts/density_followups.py`) |
| **C** `[tag]` | `csv2/raw-trends/check_cycle_claims.txt` (`scripts/check_cycle_claims.py`); per-claim rows in `cycle_claims.csv` and `response_claims.csv` |
| **S** | computed in-session from `csv2/raw-trends/frame.csv`, `runs/` or the CSVs above; not saved by a script |

Throughout: an *effect* is the paired change in % correct against no primer on
the same graphs, in percentage points; *significant* is Benjamini–Hochberg
q < 0.05 on exact McNemar tests; † marks a comparison where 15% or more of
responses hit the token budget. Task abbreviations: ND `node_degree`,
CN `connected_nodes`, EC `edge_count`, EE `edge_existence`, NC `node_count`,
CC `cycle_check`.

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
- On the aligned task (ND under `degree`), the primer helps the models that count
  poorly and costs the one that already counts: 1.7B +7.5, 1.7B-T +8.8, 4B
  −6.5, 4B-T +2.5 (q 0.078) (P `[main]`).

**Answer.** Statistics that state the answer raise accuracy where the model
cannot compute it well itself; statistics that do not state it add about
+1.6 points where there is room, and about 0 at ceiling.

## 2. Every primer against no primer

Each primer against no primer on 24 pairings (4 models × 6 tasks), main sweep
(p ≤ .50, 400 graphs per pairing), from P `[main]`. `node_count` and
`cycle_check` count like any task: the model does not know their answer is
constant. A significant change is *truncation-driven* when the share of
responses hitting the budget moves by at least half the effect (S, from
`[main]`).

| | filler | components | clustering | rwse | degree | all |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Sig. gain, primer states the answer | – | – | – | – | 3 | 3 |
| Sig. gain, primer does not state it | 2 | 3 | 3 | 1 | 3 | 4 |
| Sig. gain, truncation-driven | 3 | 1 | 3 | 3 | 2 | 2 |
| Gain, not significant | 5 | 4 | 6 | 4 | 4 | 3 |
| Neutral (within ±1) | 8 | 10 | 10 | 9 | 3 | 7 |
| Loss, not significant | 2 | 3 | 2 | 5 | 4 | 3 |
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
is 5 points or more (S, from `frame.csv`):

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

The answer is always 40. Share of responses answering 39, p ≤ .35, 300 per
cell (S, from `runs/`):

| Primer | 1.7B | 4B | 1.7B-T | 4B-T |
|---|---:|---:|---:|---:|
| none | 90% | 13% | 0% | 1% |
| rwse | 65% | 1% | 0% | 0% |
| degree | 82% | 2% | 1% | 0% |
| components | 57% | 1% | 0% | 0% |
| filler | 38% | 1% | 0% | 0% |
| all | 28% | 0% | 0% | 0% |
| clustering | 2% | 1% | 0% | 0% |

Over all finished responses at p ≤ .50, plain 1.7B answers 39 on 68.5% without a
primer, 66.5% under `rwse` and 1.2% under `clustering` (P `[nodecount]`).

Every model copies the id list 0–39 from the encoding's first line and never
counts it. Plain 1.7B then takes one of two paths (S):

| Primer | Lists ids one per line | Correct when listing | Correct on the direct "total of N" path | Correct overall |
|---|---:|---:|---:|---:|
| none | 79% | 1% | 44% | 10% |
| degree | 72% | 1% | 59% | 18% |
| rwse | 25% | 1% | **46%** | 35% |
| components | 30% | 3% | 60% | 43% |
| filler | 16% | 0% | 74% | 62% |
| all | 5% | 7% | 75% | 72% |
| clustering | 4% | 55% (of 11) | **100%** | 98% |

- The one-per-line path ends by reporting the last id, 39.
- `rwse` and `clustering` both cut that path; they split on the direct path,
  where the model writes 40 on 46% of responses under `rwse` and 289 of 289
  under `clustering`. Primer length does not explain it (`all` contains the RWSE
  text and reaches 75%), nor does naming every node (all three do).
- Plain 4B makes the same error on the direct path only ("numbered from 0 to 39,
  which is a total of 39 nodes"), at p = .10 and .20, and any primer removes it,
  `filler` included. The thinking models do not make it.

## 5. `cycle_check`: do correct answers rest on real cycles?

The gold answer is "yes" on every 40-node graph. Outcomes, p ≤ .50, 400 per
cell, % correct / answering no / truncated (S, from `frame.csv`):

| Model | none | filler | components | clustering | rwse | degree | all |
|---|---|---|---|---|---|---|---|
| 1.7B | 90.8 / 0.2 / 9.0 | 93.8 / 0.0 / 6.2 | 92.8 / 0.8 / 6.5 | 99.2 / 0.0 / 0.8 | 90.5 / 7.8 / 1.8 | 73.2 / 9.5 / 17.2 | 97.8 / 1.2 / 1.0 |
| 1.7B-T | 97.0 / 0.0 / 3.0 | 100 / 0.0 / 0.0 | 89.5 / 0.0 / 10.5 | 96.5 / 0.0 / 3.5 | 99.2 / 0.0 / 0.8 | 75.2 / 0.5 / 24.2 | 96.2 / 0.0 / 3.8 |
| 4B | 99.0 / 0.2 / 0.8 | 98.5 / 0.2 / 1.2 | 97.0 / 1.5 / 1.5 | 99.5 / 0.0 / 0.5 | 98.8 / 0.0 / 1.2 | 96.8 / 1.8 / 1.5 | 98.8 / 0.8 / 0.5 |
| 4B-T | 99.2 / 0.0 / 0.8 | 99.8 / 0.0 / 0.2 | 96.0 / 0.0 / 4.0 | 99.2 / 0.0 / 0.8 | 99.5 / 0.0 / 0.5 | 93.8 / 0.8 / 5.5 | 98.8 / 0.0 / 1.2 |

The "no" answers of plain 1.7B follow wrong reasoning (S): under `degree`, it
sums the 40 stated degrees, gets an odd total and misapplies the handshaking
lemma (35 of 47 mention an odd sum); under `rwse`, 30 of 32 mention return
probability and conclude "no cycle" from finding no self-loops.

### 5.1 What the correct answers rest on

Each cycle a correct, finished answer names (for the thinking arms, in the text
after `</think>`) is checked step by step against the edges of its prompt (§8).
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
A plausible link, not tested: the primer's wording, "return probability … after
2 steps".

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

### 5.5 Graphs without a cycle (`cc500`, Qwen3-0.6B)

On an acyclic graph a named cycle is necessarily invented and a "yes" is wrong.
Finished answers (C `[ccacyclic]`):

| Model | Primer | Acyclic: says yes | Of those: invented / not a cycle / none named | Cyclic, correct: rests on a real cycle |
|---|---|---:|---|---:|
| 0.6B | none | 94.0 | 13.9 / 67.1 / 19.0 | 47.4 |
| | components | 97.6 | 13.8 / 66.2 / 20.0 | 36.4 |
| | clustering | 89.0 | 30.1 / 50.7 / 19.2 | 43.3 |
| | rwse | 95.1 | 11.5 / 69.2 / 19.2 | 45.5 |
| 0.6B-T | none | 91.7 | 20.8 / 68.8 / 10.4 | 70.6 |
| | components | 82.1 | 35.9 / 53.1 / 10.9 | 69.8 |
| | clustering | **49.4** (−42.2) | 41.5 / 56.1 / 2.4 | 65.1 |
| | rwse | 86.9 | 20.5 / 71.2 / 8.2 | 63.7 |

- The model says yes to nearly every acyclic graph, citing mostly a length-2 or
  retraced walk.
- `clustering` halves the false yes of 0.6B-T: on a forest every printed
  clustering coefficient is 0.
- No real cycle is ever found on an acyclic graph (the script asserts it).
- The pilot's published split has 2 acyclic graphs of 30, too few to use.

## 6. Fabricated edges in the other tasks

A statement "Node x is connected to (nodes) a, b, c" (or "node x's neighbours
are …") claims those edges (§8).

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
- Missed edges cannot be assessed: the models almost never answer no to a true
  edge (at most 1.9%).
- Any statement of an edge that does not exist: 1.7B 14.5% (`filler` **+8.0**;
  `clustering`, `degree`, `all` **−8**); 1.7B-T `rwse` **+7.0**; 4B-T `rwse`,
  `degree`, `all` **+2.3** from 0.5% (C `[eeclaims]`).

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

### 6.4 `edge_count`

The per-node degree table a response lists, read by
`response_patterns.edge_chain` (`[rpchain]`). % of all responses whose table has
a wrong value, and the paired change (C `[ecchain]`):

| Model | none | filler | components | clustering | rwse | degree | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.7B | 51.2 | **+12.8** | −5.7 | **−24.5** | **−39.7** | **−23.7** | **−29.0** |
| 1.7B-T | 46.5 | +2.2 | −1.5 | **+7.8** | **−8.0** | **+19.5** | **+9.5** |
| 4B | 80.2 | **+9.8** | **−15.2** | **+13.5** | **+15.0** | **−43.8** | +5.5 |
| 4B-T | 12.8 | +1.5 | +2.8 | +3.8 | 0.0 | **+43.5** | **+64.2** |

Primers also change whether a table is listed at all. Among the responses that
list one, % with a wrong value (S, from `response_claims.csv`, not tested):

| Model | none | degree | all |
|---|---:|---:|---:|
| 1.7B | 99 | 31 | 43 |
| 1.7B-T | 62 | 66 | 56 |
| 4B | 81 | 41 | 86 |
| 4B-T | 15 | 57 | 79 |

Under `degree` and `all`, 4B-T miscopies degrees out of the primer: one response
lists "Node 27: degree 6" where the true degree, which the primer states, is 9.

## 7. Findings

1. **Explicit statistics help where they state the answer and the model cannot
   compute it.** Degree statistics raise `node_degree` accuracy for the 1.7B
   models (1.7B-T +21.5 over its middle-band cells) and cost plain 4B, which
   already counts. Statistics that do not state the answer add about +1.6
   points where there is room.
2. **Correct answers often rest on claims that are false.** A made-up cycle
   backs 43.5% of plain 1.7B's correct `cycle_check` answers without a primer,
   and 3.3% of 4B-T's. Thinking separates the two: the traces still invent, and
   the models withdraw part of it before answering.
3. **No primer reduces fabrication while keeping real reasoning.** `degree`
   lowers invented cycles only by replacing cycle-finding with an edge-count
   argument; `clustering` and `all` raise invented cycles.
4. **`filler` increases fabrication for the plain models across tasks:**
   invented cycles +12.4 (1.7B), stated fake edges +9.6 (1.7B), invented
   neighbours +14.8 (4B). It is not a neutral length control.
5. **`rwse` increases invented claims for the thinking models:** length-2
   "cycles" (4B-T real cycles −41.0), invented edges and neighbours for 1.7B-T
   on `edge_existence` (+7.0), `node_degree` (+4.8) and `connected_nodes`
   (+4.8).
6. **Misreading is mostly omission.** Missed neighbours outnumber invented ones
   about 7 to 1 in plain 1.7B's `node_degree` lists; invented cycle edges are
   mostly the closing step.
7. **`node_count` gains measure which number the model writes, not counting.**
   No model counts; plain 1.7B's error is reporting the last id (39), and only
   `clustering` fixes its direct path completely.
8. **Where a primer carries the answer's key fact, it works even for a small
   model.** On acyclic graphs, `clustering` (all coefficients 0) halves 0.6B-T's
   false yes (−42.2).

## 8. Method

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
- **Which responses:** correct finished answers for `cycle_check`; finished
  answers for `edge_existence`, `node_degree` and `connected_nodes`, with the
  thinking arms read including their trace; all responses for `edge_count`. A
  response repeated across shards counts once, the first, as in
  `primer_findings.load_runs`.
- **Tests:** each primer against none on the same graphs, exact McNemar
  (`graphtalk.scoring.mcnemar`), Benjamini–Hochberg over the primers within
  each model (`primer_findings.bh`).

**Validation.** Each run prints random samples of every rule
(`[ccsample]`, `[clsample]`); every sample was checked against the prompt's
lines. A 32-case self-check on a toy graph runs first and covers each rule that
sampling showed to need one.

**Limits.**
- The readings are regex-based and spot-checked, not hand-labelled. Hedged or
  unusual phrasings can still pass as claims, and unparsed cycle formats fall
  into "none named".
- Thinking-trace statements include exploration the model later corrects.
- For `cycle_check`, only the final answer of the thinking arms is judged; the
  trace is reported separately (§5.4).
- The shares "of false alarms" and "among responses listing a table" are not
  tested; paired tests use graphs where both responses finished.
- `edge_count` depends on `scripts/response_patterns.py`, which is not tracked.

**Reproduce:**

    PYTHONPATH=. python scripts/check_cycle_claims.py --csv-dir csv2/raw-trends \
        > csv2/raw-trends/check_cycle_claims.txt

## 9. Open questions

- **Why `clustering` fixes plain 1.7B's `node_count` and `rwse` does not.** The
  choice between writing 40 and 39 on the direct path is a single token. A GPU
  probe of P("40") against P("39") after "This is a total of", under each primer
  and with the two primers' number formats swapped, would locate it.
- **Whether `rwse` causes length-2 claims through its "after 2 steps" wording.**
  Rewording the primer (for example "return probability after a 2-step walk")
  would test it.
- **Missed edges on `edge_existence`.** The models almost never answer no to a
  true edge, so dropped edges cannot be measured on these runs.
- **Acyclic graphs for the larger models.** Only Qwen3-0.6B ran `cc500`; the same
  check on 1.7B and 4B would need new runs.
