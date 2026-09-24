# The 40-node sweep

Source: `csv2/raw-trends/primer_findings.txt`
Source: `csv2/raw-trends/legacy_claims.txt`

The main experiment. Runs:
`runs/{qwen3-1.7b,qwen3-1.7b-think,qwen3-4b,qwen3-4b-think}.{densfull40,densfull40hi}.shard*.jsonl`,
with prompts in `prompts.densfull40.jsonl` and `prompts.densfull40hi.jsonl`. Reproduce:

    PYTHONPATH=. python scripts/build_raw_frame.py
    PYTHONPATH=. python scripts/primer_findings.py --csv-dir csv2/raw-trends > csv2/raw-trends/primer_findings.txt
    PYTHONPATH=. python scripts/legacy_claims.py > csv2/raw-trends/legacy_claims.txt

`build_raw_frame.py` regenerates every graph from its instance id and stops if
any regenerated gold answer differs from the recorded one. `legacy_claims.py`
computes quantities that other analyses of these runs used, under the same
rules; this document cites them only in §4 and §10.

**Scoring.** Every response is correct, wrong or truncated (it used the whole
token budget); a truncated response is never counted as wrong and never
dropped (`graphtalk/outcomes.py`). An effect is the paired change, on the same
graphs, in the share of all responses that is correct, in percentage points,
printed with the change in the truncated share beside it. Intervals are 95%
bootstrap intervals over graphs within density; tests are exact McNemar;
q-values are Benjamini–Hochberg within (arm, task, control). Descriptions of
answers (error size, yes-rate, false alarms, response text) use finished
responses only. A cell is *flagged* (†) when 15% or more of its responses on
either side are truncated; pooled analyses leave flagged cells out and list
them separately.

**Reading this document.** Every number is followed by the tag it is printed
under in one of the two sources, so `+27.2 [edgecount]` is found in the
`[edgecount]` block; `tests/test_results_docs.py` checks each one.

## 1. Setup and measurement

- **Design.** 84000 [setup] responses on 700 [setup] graphs, in 840 [setup]
  cells of 100 [setup] graphs each. Models: Qwen3-1.7B and Qwen3-4B, each with
  thinking off (*plain*) and on (*thinking*), zero-shot, incident encoding.
  Graphs: Erdős–Rényi with 40 nodes; all six tasks at edge probability
  p = .10, .20, .35, .50 (the *main sweep*), and `node_degree` and
  `edge_existence` also at p = .65, .75, .85 (the *high-density extension*).
  Conditions: `none`, `filler`, `components`, `clustering`, `rwse`, `degree`,
  `all` (degree, clustering and RWSE together).
- **Token budget.** 8192 [setup] tokens for every arm in the main sweep; in the
  high-density extension, 2048 [setup] for the two plain arms and 8192 [setup]
  for the thinking arms.
- **Truncation.** `edge_count` truncates on 22.4% [setup] of plain Qwen3-1.7B
  responses and on 81.0% [setup] and 62.2% [setup] of the two thinking arms'
  responses; every other (arm, task) truncates on at most 6.5% [setup].
- **Two tasks have a constant answer at 40 nodes.** `node_count` is always
  40 [setup] and `cycle_check` always "yes", so neither measures reading the
  graph.
- **`edge_existence` answers depend on density.** The share of queried pairs
  that are edges rises from 10% [goldshare] at p = .10 to 85% [goldshare] at
  p = .85; the majority-class accuracy over the main sweep is 73.25% [setup].
- **Primer lengths** (mean characters, main sweep): `components` 37 [length],
  `degree` 891 [length], `clustering` 1,629 [length], `rwse` 2,949 [length],
  `all` 4,691 [length], `filler` 1,829 [length]. `filler` states no structure
  and names every node; its length lies between those of the single-feature
  node-level primers.

## 2. Which primers state the answer

A graph-blind solver reads only the primer text: 16 [bars] exact rules,
1 [bars] heuristic and 8 [bars] rules fitted on graphs disjoint from the ones
scored.

- `degree` and `all` state the answer to `node_degree` and `edge_count`: the
  solver scores 100.0 [bars] on both.
- Among the four tasks whose answer varies, no other (task, primer) cell lets it
  do better than 74.6 [bars]
  (`edge_existence` under `degree`); on `node_degree` it scores 14.4 [bars] from
  no primer and 21.0 [bars] from `rwse`; on `connected_nodes` it never exceeds
  0.8 [bars].
- On `edge_existence`, on the solver's own graphs (300 per density), the fitted
  RWSE density rule is less accurate than the majority answer at p = .65, .75
  and .85, even when scored on the graphs it was fitted on: 38.0 [rwsefit]
  against 62.3 [rwsefit] at p = .65, and 12.0 [rwsefit] against 88.0 [rwsefit]
  at p = .85.
- The saved prompts confirm the leakage: reading the queried node's degree
  sentence gives the gold `node_degree` answer on 700/700 [leak] prompts, and
  half the sum of the stated degrees gives the gold `edge_count` on
  400/400 [leak].

Below, *answer-carrying* means `degree` or `all` on `node_degree` or
`edge_count`; every other (primer, task) is *side information*.

## 3. Every primer against no primer

`[main]` prints every arm × task × primer over the main sweep, against `none`
and against `filler`. The rows below answer the proposal's two questions.

### Task alignment

| Task (alignment) | Arm | Primer | Correct, `none` → primer (%) | Effect | q |
|---|---|---|---|---|---|
| `node_degree` (aligned) | 1.7B | `degree` | 60.50 [main] → 68.00 [main] | +7.5 [main] | 0.0042 [main] |
| | 1.7B-thinking | `degree` | 76.25 [main] → 85.00 [main] | +8.8 [main] | 0.0008 [main] |
| | 4B | `degree` | 99.25 [main] → 92.75 [main] | −6.5 [main] | 7.7e-06 [main] |
| | 4B-thinking | `degree` | 97.00 [main] → 99.50 [main] | +2.5 [main] | 0.078 [main] |
| `edge_count` (adjacent) | 4B | `degree` | 2.00 [main] → 29.25 [main] | +27.2 [main] | 3.4e-27 [main] |
| | 4B-thinking † | `components` | 20.00 [main] → 44.75 [main] | +24.8 [main] | 2.9e-17 [main] |
| `connected_nodes` (adjacent) | 4B | `components` | 86.00 [main] → 95.25 [main] | +9.2 [main] | 4.9e-07 [main] |
| | 1.7B-thinking | `degree` | 78.75 [main] → 85.00 [main] | +6.2 [main] | 0.0079 [main] |
| `node_count` (agnostic) | 1.7B | `clustering` | 31.50 [main] → 98.75 [main] | +67.2 [main] | 1.3e-80 [main] |
| | 1.7B | `filler` | 31.50 [main] → 71.25 [main] | +39.8 [main] | 4.6e-45 [main] |
| | 4B | `all` | 90.25 [main] → 100.00 [main] | +9.8 [main] | 2.2e-11 [main] |
| `edge_existence` (agnostic) | 1.7B | `all` | 70.50 [main] → 82.00 [main] | +11.5 [main] | 1.7e-05 [main] |
| | 1.7B | `filler` | 70.50 [main] → 54.75 [main] | −15.8 [main] | 3.5e-12 [main] |
| | 4B | `rwse` | 90.50 [main] → 80.50 [main] | −10.0 [main] | 4.3e-06 [main] |
| `cycle_check` (agnostic) | 1.7B † | `degree` | 90.75 [main] → 73.25 [main] | −17.5 [main] | 3.3e-09 [main] |
| | 1.7B-thinking † | `degree` | 97.00 [main] → 75.25 [main] | −21.8 [main] | 3.1e-19 [main] |

† flagged: 15% or more of the responses on one side are truncated.

- On the aligned task the answer-carrying primer helps the arms that count
  poorly and costs plain Qwen3-4B, which already counts correctly (§4).
- The largest gains on adjacent and agnostic tasks come from answer leakage,
  from finishing within the budget, or from no mechanism the output
  identifies: `edge_count` under `degree` is leaked (§2); the 4B-thinking
  `edge_count` gain is a change in truncation, from 79.50% [edgecount]
  truncated to 54.50% [edgecount] under `components` (the truncated change is
  −25.0 [main]).
- On `node_count`, plain Qwen3-1.7B answers 39 on 68.5% [nodecount] of
  finished responses without a primer, 1.2% [nodecount] under `clustering` and
  28.7% [nodecount] under `filler`. Without a primer it answers 39 on
  98% [nodecount] of items at p = .10 and 100% [nodecount] at p = .20, where
  every node has a line in the encoding (100/100 [nodecount] graphs), and on
  4% [nodecount] at p = .50.
- `degree` lowers `cycle_check` accuracy for both 1.7B arms. For 1.7B-thinking
  nearly all of it is truncation (the truncated share rises by +21.2 [main] of
  the −21.8 [main]); for plain 1.7B, +8.2 [main] of the −17.5 [main].

### Model capacity

- The median no-primer correct share over cells under the truncation flag is
  66.0 [arms] for plain 1.7B, 87.0 [arms] for plain 4B, 93.5 [arms] for
  1.7B-thinking and 97.5 [arms] for 4B-thinking.
- Qwen3-4B with thinking moves by at most 2.5 [null4bt] points under any
  primer on `node_degree`, `edge_existence` and `connected_nodes`.
- Primers act where the no-primer accuracy leaves room (§4): 40 [arms] of the
  plain 1.7B cells sit between 25% and 75%, against 0 [arms] for 4B-thinking.

### Against `filler`

- `components` beats `filler` by +21.8 [main] on plain 4B `connected_nodes`, and
  by +12.5 [main] on 4B-thinking `edge_count` (†).
- `filler` itself costs plain 4B −12.5 [main] on `connected_nodes` and plain
  1.7B −15.8 [main] on `edge_existence`.

## 4. A primer that states the answer changes the procedure

**Plain Qwen3-4B, `node_degree`.** Under `degree`, the effect by density is
−6.0 [flip], −1.0 [flip], −7.0 [flip], −12.0 [flip], −6.0 [flip],
+7.0 [flip], +27.0 [flip] (p = .10 to .85), against no-primer accuracy of
100.0 [flip], 99.0 [flip], 99.0 [flip], 99.0 [flip], 82.0 [flip], 50.0 [flip]
and 33.0 [flip]. At p = .50, 13 [flip] answers break and 1 [flip] is fixed
(p = 0.0018 [flip]).

- Without a primer it lists the queried node's neighbours: at p = .50 it
  enumerates in 100% [route] of responses, correct in 99% [route].
- Under `degree` it *retrieves*, answering without restating the neighbour
  list, in 44% [route] to 74% [route] of responses across densities, with
  retrieval accuracy between 73.4% [route] and 98.2% [route]. At p = .50:
  retrieve 62% [route] (92% [route] correct); restate the queried node's
  neighbour line without listing the neighbours one per line, 30% [route]
  (73% [route]); enumerate 8% [route]. Median response length at p = .50
  falls from 273 [flip] tokens to 34 [flip].
- Enumeration is counted when five or more neighbours are listed one per line;
  none of the 245 [route] responses at p ≤ .20 classed as asserting lists every
  neighbour, one per line, of a node with fewer than five.
- Some wrong retrievals are copy errors: of 57 [copyerr], 42% [copyerr] are
  within 1 of the gold and 68% [copyerr] within 3, and 17 [copyerr] equal the
  degree stated in the sentence just before or after the queried node's in the
  primer, against 10.4 [copyerr] expected by chance (p = 0.017 [copyerr]). Over
  every wrong answer in the main sweep, 12 [lcopy] of 29 [lcopy] do, against a
  19.4% [lcopy] chance (p = 0.0055 [lcopy]).
- Retrieval is more accurate for queried nodes 0–9 (95.7% [position]) than for
  10–39 (83.1% [position]); node ids 0–9 are both single-digit and first in the
  list, so the two cannot be separated here.

**Qwen3-1.7B with thinking.** It keeps counting: it retrieves in at most
2% [verify] of responses and enumerates in 90% [verify] at p ≥ .35. It reports a
conflict with the stated degree in 23.2% [verify] of responses at p ≤ .50 and
48.5% [verify] at p ≥ .65 (the same detector fires on 0.3% [verify] and
4.4% [verify] of responses without a primer, where no degree is stated), and
when it does it lands on the stated value, which is correct, in 61% [verify]
and 69% [verify] of them. Over all seven densities the effect is
+16.0 [verify] (truncated +6.4 [verify]); at p ≥ .65, +25.7 [verify]
(truncated +9.0 [verify]). `degree` raises its truncation from 8/700 [trunc]
to 53/700 [trunc] responses, 96% [trunc] of which report a conflict before
the budget runs out.

**Plain Qwen3-1.7B.** `degree` helps at p = .20, +10 [plain17] from a no-primer
correct share of 80 [plain17], and at p = .35, +18 [plain17] from 41 [plain17];
it does not at p = .50, −1 [plain17] from 30 [plain17], or above, between
−1 [plain17] and +2 [plain17].

**Effect by baseline.** Binning every (arm, task, density, primer) cell of the
four tasks whose answer varies, under the truncation flag, by its no-primer
correct share:

| No-primer correct share | Answer-carrying | Side information |
|---|---|---|
| 0.00–0.25 | +8.3 [bands] | −1.7 [bands] |
| 0.25–0.50 | +16.1 [bands] | +2.6 [bands] |
| 0.50–0.75 | +12.9 [bands] | +3.9 [bands] |
| 0.75–0.90 | −5.8 [bands] | +0.7 [bands] |
| 0.90–1.00 | −0.9 [bands] | −0.9 [bands] |

- Binning cells on half their graphs and measuring the effect on the other
  half gives +13.9 [splithalf] and +16.2 [splithalf] for the two middle
  answer-carrying bands, so the pattern is not regression to the mean.
  Estimating each cell's baseline and effect on disjoint halves of its graphs,
  for the primers that raise the solver's `node_degree` accuracy by more than
  0.05 (`rwse`, `degree`, `all`), gives r = −0.248 [lcross] between baseline
  and effect over 168 [lcross] points (p = 0.0012 [lcross]; the points come two
  per cell and cells share their baselines, so they are not independent).
- The result does not depend on the flag: keeping cells below 10% truncation
  gives +15.4 [bandsens] and +12.5 [bandsens]; below 50%, +16.1 [bandsens] and
  +12.9 [bandsens]. Only adding the cells where half or more of the responses
  truncate moves the 0.25–0.50 band, to +5.0 [bandsens]; every flagged cell is
  listed under `[flagged]`, and §9 describes them.
- In the middle bands the answer-carrying cells are all `node_degree`:
  18 [window] cells, of which 10 [window] are 1.7B-thinking at a mean of
  +21.5 [window].

## 5. A trade-off per-task accuracy hides

For plain Qwen3-4B, `degree` raises `edge_count` from 2.00% [edgecount] to
29.25% [edgecount] correct (+27.2 [edgecount]; 114 [edgecount] fixed,
5 [edgecount] broken). It already sums degrees without a primer, in
96% [edgecount] to 100% [edgecount] of finished responses, and its answers are
off by a median 10.9% [edgecount] of the true count.

On the same graphs, `node_degree` and `connected_nodes` query the same node, so
the two answers can be checked against each other. *J* is the share of items
where both are correct; *C* the share where both finished and the stated degree
equals the size of the stated neighbour set. Every item in J is in C.

- Plain 4B: `degree` lowers J from 85.75 [joint] to 76.50 [joint]
  (−9.2 [joint]; 28 [joint] fixed, 65 [joint] broken; q = 0.00032 [joint]),
  while `components` raises it to 95.00 [joint] (+9.2 [joint]).
- Plain 1.7B: `degree` raises J from 54.00 [joint] to 62.50 [joint] and C to
  73.00 [joint]; the answers that agree with each other without both being
  correct (C − J) rise from 8.25 [joint] to 10.50 [joint].
- 4B-thinking stays between 93.00 [joint] and 96.00 [joint] under every
  condition.

## 6. `edge_existence` measures a yes-bias

- The plain arms say "yes" to 0.98 [fa]–1.00 [fa] (1.7B) and 0.97 [fa]–0.99 [fa]
  (4B) of true edges under every condition; 99% [fa] and 94% [fa] of their
  errors are false alarms.
- Plain Qwen3-1.7B collapses as density rises: at p = .65, .75 and .85 its
  yes-rate is 0.98 [collapse], 0.99 [collapse] and 0.99 [collapse], its
  balanced accuracy 0.53 [collapse], 0.52 [collapse] and 0.53 [collapse], and
  its median response 48 [collapse] tokens at p = .75, against 206 [collapse]
  at p = .50.
- Primers act on false alarms. For plain 1.7B, `all` lowers the false-alarm
  rate from 0.51 [fa] to 0.33 [fa] (−18.6 [fa] on paired responses) and
  `filler` raises it to 0.69 [fa] (+18.3 [fa]); at p ≥ .65, balanced accuracy
  rises by +8.6 [collapse] under `degree` and +14.9 [collapse] under `all`. For
  plain 4B, `clustering` is the only primer that lowers false alarms, from
  0.16 [fa] to 0.08 [fa].
- The thinking arms show no bias: their false-alarm rate without a primer is
  0.01 [fa] (1.7B) and 0.00 [fa] (4B).

## 7. Side information is small and non-specific

- **`clustering`, plain 4B, `node_degree` at p ≥ .65:** +11.3 [clusthi]
  (`rwse` +7.7 [clusthi], `filler` +2.0 [clusthi], `all` −21.3 [clusthi]).
  The printed coefficients barely vary within a graph there (standard deviation
  0.021 [cluster] at p = .65, lower above), and the procedure is unchanged:
  69% [clustproc] of responses enumerate, against 74% [clustproc] without a
  primer.
- **`clustering`, plain 1.7B, `node_degree`:** +3.0 [clusthi] in the main sweep
  (p = 0.26 [clusthi]). Dedicated runs replicate a gain: at p ≤ .50,
  +4.0 [replic] on 300 new graphs per density (p = 0.0047 [replic]) and
  +4.2 [replic] on fresh seeds (p = 0.00055 [replic]); on a grid at fixed mean
  degree over n = 20 to 160, +6.2 [replic] (p = 2.2e-12 [replic]). At 40 nodes
  and p ≥ .65 it is absent: −0.7 [replic] on 1200 [replic] dedicated pairs,
  −3.3 [clusthi] in the main sweep.
- **`components`, plain 4B, `connected_nodes`:** +9.2 [comp] (44 [comp] fixed,
  7 [comp] broken). From p = .20 every graph is connected (0/100 [comp] with
  more than one component), so the primer is one fixed sentence; with it the
  model restates the queried node's line in 98.5% [compproc] of responses,
  against 63.0% [compproc] without.
- **Bundling:** `all` is no better than the mean of its three parts,
  +0.3 [bundle] over 53 [bundle] main-sweep (arm, task, density) combinations
  under the truncation flag.
- **Length:** the effect's slope in primer length is positive for plain 1.7B in
  5 of 6 [length] combinations and for plain 4B in 0 of 7 [length]; `filler`
  sits 18.2 [length] and 13.8 [length] points below the fitted line on its two
  largest costs, so its cost is not length alone.

## 8. Feature resolution

The `rwse` primer prints each node's return probability after 2 and 3 steps to
two decimals. The number of distinct printed pairs per 40-node graph is
30.14 [rwse] at p = .10, 6.74 [rwse] at p = .35 and 3.30 [rwse] at p = .50, and
between 2.00 [rwse] and 2.72 [rwse] at p ≥ .65; at p = .50 the most common pair
covers 62.68% [rwse] of the nodes, while the degree primer states 12.50 [rwse]
distinct degrees. From p = .35, every graph (100/100 [rwse]) has fewer RWSE
classes than degree classes.

## 9. Truncation as an outcome

56 [cells] of the 440 [cells] (arm, task, density, primer) cells of the four
tasks whose answer varies are flagged, all of them `edge_count`; the main-table
rows marked † are flagged on the same rule. Where a cell is flagged the
primer's main effect is on finishing within the budget, and its direction
depends on density:

- 4B-thinking without a primer is correct on 20.00% [edgecount], wrong on
  0.50% [edgecount] and truncated on 79.50% [edgecount]; it is correct on
  97.6% [edgecount] of the responses it finishes.
- At p = .10, `degree` raises 1.7B-thinking truncation from 50 [flagged] to
  80 [flagged] percent (correct share −37 [flagged]); at p = .50 it lowers it
  from 100 [flagged] to 84 [flagged] (+10 [flagged]).

## 10. Limits

- Generating the same 1200 [rerun] prompts twice changes the outcome of
  5.3% [rerun] of them.
- In one cell of 100 graphs, the median smallest detectable paired effect is
  7.4 [power] points (cells under the truncation flag). Pooled over the main
  sweep's 400 paired graphs, among the comparisons against `none` that are not
  significant and under the truncation flag, a gain or a harm cannot be
  detected in 0/16 [lmde] for plain
  1.7B and 18/28 [lmde] for 4B-thinking, because the no-primer accuracy is at
  floor or ceiling; where a gain can be detected, its median detectable size
  is 5.7 [lmde] points for the 1.7B arms and 3.2 [lmde] for the 4B arms.
- The answer extractor parses 99.84% [extract] of responses.
- For `connected_nodes`, set-F1 moves in the same direction as exact match in
  only 3 [f1] of 5 [f1] large contrasts, and by 0.11 [f1] to 0.17 [f1] as much;
  exact match is the primary metric.
- The plain arms ran the high-density extension with a 2048 [setup]-token
  budget, a quarter of the main sweep's.
