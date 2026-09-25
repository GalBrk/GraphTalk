# The density follow-ups

Source: `csv2/density-followups/density_followups.txt`

Dedicated `node_degree` runs that follow up the 40-node sweep
([n40-sweep.md](n40-sweep.md)): Qwen3-1.7B with thinking off (*plain*) and on
(*thinking*) at seven edge densities, a fresh-seed replication, a
fixed-mean-degree grid (Qwen3-1.7B and Qwen3-8B), and the `density40` pilot
that preceded them. Runs:

| Run set | Model | Graphs | Conditions |
|---|---|---|---|
| `degdens40` | 1.7B plain | n = 40, p = .10, .20, .35, .50 | `none`, `components`, `clustering` |
| `degceil` | 1.7B plain | the same graphs | `degree` |
| `degdens40hi` | 1.7B plain | n = 40, p = .65, .75, .85 | `none`, `components`, `clustering`, `degree` |
| `degdensfill` | 1.7B plain | all seven densities | `filler` |
| `degdensthink`, `degdensfillT` | 1.7B thinking | all seven densities, the same graphs | `none`, `components`, `clustering`, `degree`; `filler` |
| `degdensrep` | 1.7B plain | p = .10 to .50 at a fresh seed | `none`, `clustering` |
| `degfixdeg` | 1.7B plain, 8B plain | eight (n, p) cells at mean degree 8 and 16 | `none`, `clustering` |
| `density40` | 1.7B plain | n = 40, six densities; `node_degree` and `connected_nodes` | `none`, `degree` |

Files: `runs/qwen3-1.7b.{degdens40,degceil,degdens40hi,degdensfill,degdensrep,degfixdeg,density40}.shard*.jsonl`,
`runs/qwen3-1.7b-think.{degdensthink,degdensfillT}.shard*.jsonl`,
`runs/qwen3-8b.degfixdeg.shard*.jsonl`. Reproduce:

    PYTHONPATH=. python scripts/density_followups.py > csv2/density-followups/density_followups.txt

The forensics need `statsmodels` (the `analysis` extra). The driver rebuilds
the graphs and prompts of the plain, thinking and fixed-mean-degree runs with
`scripts/build_size_sweep.py`, checks the `density40` runs against their
prompt file, and stops if any gold answer differs from the recorded one; it
scores each run set with `scripts/score_density_sweep.py` and runs the
forensics in `scripts/analyze_rq3_leads.py`, which rebuilds the fresh-seed
graphs itself.

**Scoring.** As in [n40-sweep.md](n40-sweep.md): every response is correct,
wrong or truncated, and a truncated response is never counted as wrong and
never dropped. An effect is the paired change, on the same graphs, in the share
of all responses that is correct, in percentage points, with the change in the
truncated share beside it. Intervals are 95% bootstrap intervals over graphs
within density (or within cell); tests are exact McNemar; q-values are
Benjamini–Hochberg, over the primers for a pooled range and over every
(level, primer) of a block for per-level effects (within each task for the
pilot). Error size, token counts and
the share answering 39 use finished responses. A level where 15% or more of
either side's responses are truncated would be flagged and left out of pooled
effects and trends; none is here (the largest truncated share is
11.8 [ddthink] per cent, thinking with `degree` at p = .65). The *blind bar* is the share of golds matched by the
best constant answer at that level.

**Reading this document.** Every number is followed by the tag it is printed
under in the source, so `+3.8 [ddplain]` is found in the `[ddplain]` block;
`tests/test_results_docs.py` checks each one.

## 1. Design

- **Graphs.** Erdős–Rényi graphs with 40 nodes, 400 [ddplain] per density
  and condition. Mean edges rise from 78.5 [ddsetup] at p = .10 to
  663.3 [ddsetup] at p = .85, and the no-primer prompt from 2,002 [ddsetup] to
  6,407 [ddsetup] characters. 700 [rqselection] of the dedicated graphs are
  the main sweep's own, with identical prompts (700 [rqselection]).
- **What each primer adds** (median over graphs of the characters added
  to `none`): `components` +39 [ddsetup], `degree` +871 [ddsetup] at p = .10
  to +911 [ddsetup] at p = .85, `clustering` +1,631 [ddsetup], `filler`
  +1,831 [ddsetup]. `filler` is therefore roughly length-matched to
  `clustering`, not to `components` or `degree`.
- **Fixed mean degree.** Four sizes at mean degree 8 (n = 20, p = .421 to
  n = 160, p = .05) and four at 16 (n = 20, p = .842 to n = 160, p = .101). The
  mean gold stays between 7.67 [fixdeg17] and 8.03 [fixdeg17], and between
  16.13 [fixdeg17] and 16.22 [fixdeg17], while edges grow from 80.0 [fixsetup] to 635.8 [fixsetup]
  (and 160.6 [fixsetup] to 1286.4 [fixsetup]) and the no-primer prompt from
  1,308 [fixsetup] to 11,109 [fixsetup] characters (and 1,873 [fixsetup] to
  16,732 [fixsetup]).
- **The `density40` pilot.** 100 [d40] graphs per density at p = .05, .10, .20,
  .35, .50, .75; the `node_degree` prompt without a primer grows from
  1,550 [d40setup] to 5,816 [d40setup] characters.

## 2. `clustering` on `node_degree`, plain Qwen3-1.7B

- **At p ≤ .50:** +3.8 [ddplain] (+1.6 [ddplain] to +6.1 [ddplain]),
  p = 0.0017 [ddplain], 215 [ddplain] graphs fixed and 154 [ddplain] broken of
  1600 [ddplain]. On a fresh seed at the same densities: +4.2 [ddrep]
  (p = 0.00055 [ddrep], 1600 [ddrep] pairs). `primer_findings.py`'s `[replic]`
  block, cited in [n40-sweep.md §7](n40-sweep.md#7-side-information-is-small-and-non-specific),
  is the same computation.
- **By density:** +0.2 [ddplain] at p = .10, +3.2 [ddplain] at .20,
  +7.5 [ddplain] at .35 (q = 0.044 [ddplain], the only level that survives
  correction) and +4.2 [ddplain] at .50; on the fresh seed +2.0 [ddrep],
  +4.0 [ddrep], +4.5 [ddrep] and +6.5 [ddrep].
- **At p ≥ .65 it is absent:** −0.7 [ddplain] (p = 0.57 [ddplain]). Over all
  seven densities it is +1.9 [ddplain] (p = 0.022 [ddplain]).
- **Against the roughly length-matched `filler`:** +3.2 [ddplainfill] at p ≤ .50
  (p = 0.007 [ddplainfill]).
- **No trend within p ≤ .50:** slope +0.113 [ddplain] in the correct share
  per unit of density (p = 0.15 [ddplain]); across all seven levels the effect
  shows no significant correlation with the no-primer accuracy
  (r = +0.24 [ddplain], p = 0.6 [ddplain]).
- **Share of the answer-carrying primer's gain it recovers:**
  35.1% [ddplain] at p = .20, 34.5% [ddplain] at .35 and 63.0% [ddplain] at
  .50.
- It also helps at fixed mean degree, pooled over n = 20 to 160
  (+6.2 [fixdeg17], §6), though not in every cell; §8 locates the gain in the
  model's reading of the queried node's line.

## 3. The answer-carrying `degree` primer

The `degree` primer states every node's degree, so it states the answer.

- **Plain:** +10.2 [ddplain] at p ≤ .50, largest at p = .35
  (+21.8 [ddplain]) and +6.8 [ddplain] at .50, and +1.2 [ddplain] at p ≥ .65
  (p = 0.33 [ddplain]).
- **Why plain fails at high density:** without a primer the plain model
  answers 39, the degree of a node joined to every other node, on
  24.0% [ddplain], 46.2% [ddplain] and 61.0% [ddplain] of finished responses at
  p = .65, .75 and .85, and on 70.8% [ddplain] at .85 even with the degree
  stated. Its accuracy there, 14.0 [ddplain], 9.0 [ddplain] and
  5.2 [ddplain] per cent, is below the blind bar at .75 and .85, where the
  best constant answer matches a share of 0.165 [ddplain] and
  0.172 [ddplain] of golds.
- **Thinking:** +11.3 [ddthink] at p ≤ .50 and +24.8 [ddthink] at p ≥ .65,
  rising to +29.5 [ddthink] at p = .85; the thinking model answers 39 on
  1.3% [ddthink] of finished no-primer responses at .85. `degree` also adds
  truncation in this arm: +2.3 [ddthink] points at p ≤ .50, +9.0 [ddthink] at
  p ≥ .65 and +6.2 [ddthink] at p = .85.

## 4. Thinking against plain

Same graphs, same prompts; thinking accuracy minus plain accuracy, per
condition.

- **Without a primer:** +24.8 [ddgap] pooled, from +1.5 [ddgap] at p = .10
  (p = 0.46 [ddgap], not significant) through +10.8 [ddgap], +24.0 [ddgap],
  +29.0 [ddgap], +29.8 [ddgap] and +35.0 [ddgap] to +43.5 [ddgap] at p = .85.
- **With a primer**, pooled: `components` +24.6 [ddgap], `clustering`
  +22.0 [ddgap], `filler` +21.2 [ddgap], and `degree`, which states the answer,
  +35.5 [ddgap] (with +5.8 [ddgap] points more truncated).
- **Cost in tokens** (median, finished responses, no primer): 76 [ddgap] →
  1299 [ddgap] at p = .10 (×17.1 [ddgap]), 283 [ddgap] → 2009 [ddgap] at .65
  (×7.1 [ddgap]) and 282 [ddgap] → 3232 [ddgap] at .85 (×11.5 [ddgap]).
- **Primers in the thinking arm:**
  - `clustering`: +2.5 [ddthink] at p ≤ .50 (q = 0.043 [ddthink]) and
    −5.4 [ddthink] at p ≥ .65 (q = 0.0015 [ddthink]). Per level it helps at
    p = .20 (+5.2 [ddthink], q = 0.0097 [ddthink]) and hurts at p = .75
    (−8.5 [ddthink], q = 0.011 [ddthink]); the cost at p = .85 does not
    survive correction (−5.8 [ddthink], q = 0.085 [ddthink]). At .35 and .50,
    where the plain arm's gain peaks, it is +3.2 [ddthink] and
    −0.8 [ddthink]. The effect falls with density (slope −0.156 [ddthink] in
    the correct share per unit of density, p = 5e-05 [ddthink], the smallest
    20,000 [ddthink] permutations can give) and, across all seven levels,
    tracks the no-primer accuracy (r = +0.79 [ddthink], p = 0.033 [ddthink]).
  - `components`: +1.3 [ddthink] at p ≤ .50 (p = 0.15 [ddthink]) and
    −3.0 [ddthink] at p ≥ .65 (p = 0.035 [ddthink]).
  - `degree`: across all seven levels, the gain grows where the no-primer
    accuracy is lowest (r = −0.94 [ddthink], p = 0.0015 [ddthink]).

## 5. The `filler` control

`filler` names every node and states no structure.

- **Plain:** +0.6 [ddplain] at p ≤ .50 (p = 0.6 [ddplain]) and −6.4 [ddplain]
  at p ≥ .65; −9.5 [ddplain] at p = .65 alone.
- **Thinking:** −1.9 [ddthink] at p ≤ .50 (p = 0.072 [ddthink]) and
  −11.4 [ddthink] at p ≥ .65.
- So against `filler`, any primer at p ≥ .65 looks like a gain (plain
  `clustering` +5.8 [ddplainfill], `components` +5.3 [ddplainfill]; no primer
  at all +6.4 [ddplainfill]): that is `filler`'s cost, not the primers'
  content. `clustering` against `filler` is informative in the plain arm at
  p ≤ .50, where `filler` costs nothing: +3.2 [ddplainfill]. In the thinking
  arm it is +4.4 [ddthinkfill] at p ≤ .50, but there `filler` itself costs
  −9.5 [ddthink] at p = .50 (q = 0.00086 [ddthink]), so part of that is
  `filler`'s cost.

## 6. Fixed mean degree

- **Accuracy without a primer falls with n at fixed mean degree:** at mean
  degree 8, 91.2 [fixdeg17], 72.8 [fixdeg17], 64.5 [fixdeg17] and
  61.5 [fixdeg17] for n = 20, 40, 80, 160; at 16, 57.5 [fixdeg17],
  39.8 [fixdeg17], 22.5 [fixdeg17] and 22.2 [fixdeg17].
- **At matched edge counts, the larger answer is harder:** n = 160, p = .05
  (635.8 [fixsetup] edges) scores 61.5 [fixdeg17]; n = 80, p = .203
  (641.0 [fixsetup] edges) scores 22.5 [fixdeg17].
- **`clustering`, Qwen3-1.7B:** +6.2 [fixdeg17] pooled
  (p = 2.2e-12 [fixdeg17]); +4.2 [fixdeg17] at mean degree 8 and
  +8.2 [fixdeg17] at 16. The largest gain is at n = 20, p = .842
  (+16.5 [fixdeg17], p = 3.5e-08 [fixdeg17]), then n = 80, p = .101
  (+10.5 [fixdeg17]); the one significant cost is at n = 20, p = .421
  (−4.8 [fixdeg17], p = 0.0054 [fixdeg17]), where accuracy is highest, and
  n = 160, p = .101 shows nothing (−1.0 [fixdeg17], p = 0.75 [fixdeg17]).
- **`clustering`, Qwen3-8B:** −1.0 [fixdeg8] pooled (p = 0.05 [fixdeg8]).
  Without a primer Qwen3-8B is near ceiling at n ≤ 40 and at n = 80 with mean
  degree 8 (96.5 [fixdeg8] to 99.8 [fixdeg8]); it scores 89.5 [fixdeg8] at
  n = 80 with mean degree 16, and 89.8 [fixdeg8] and 80.2 [fixdeg8] at
  n = 160.

## 7. The `density40` pilot

- **`node_degree`:** `degree` +5.7 [d40] pooled (p = 0.0015 [d40]); per level
  +11.0 [d40] at p = .20 and +20.0 [d40] at .35 (q = 0.01 [d40] and
  0.0072 [d40]), nothing at
  .50 (+0.0 [d40]). Without a primer accuracy falls from 97.0 [d40] at p = .05
  to 12.0 [d40] at .75.
- **`connected_nodes`** (exact set match): `degree` +3.0 [d40] pooled
  (p = 0.085 [d40]); −9.0 [d40] at p = .05 (p = 0.012 [d40]), which does not
  survive correction within the task (q = 0.07 [d40]). Without a primer
  accuracy falls from 99.0 [d40] to 19.0 [d40].

## 8. Where the `clustering` gain comes from

From the plain runs at the default seed, with the fresh seed held out.

- **Selection.** The screen that picked `clustering` on `node_degree` covered
  40 [rqselection] cells. On the 700 [rqselection] dedicated graphs the screen
  also saw, the effect is 1.29 [rqselection] points (p = 0.47 [rqselection]); on the
  2100 [rqselection] it did not see, 2.1 [rqselection] points
  (p = 0.031 [rqselection]). The fresh-seed result survives a Bonferroni
  correction over the screen (p = 0.022 [rqselection]).
- **At p = .35 and .50,** 5.88 [rqselection] points (p = 0.0036 [rqselection]); the
  held-out seed at the same densities gives 5.5 [rqselection]
  (p = 0.0083 [rqselection]).
- **The queried node's position in the node list.** By thirds of the list
  (graphs ranked by the queried node's index within each density),
  early / middle / late, in points: over all seven densities
  5.01 [rqhetero] (p = 0.00063 [rqhetero]), 0.75 [rqhetero],
  −0.11 [rqhetero]; at p = .35 and .50, 11.57 [rqhetero],
  3.38 [rqhetero], 2.63 [rqhetero]; on the held-out seed (p ≤ .50), 7.46 [rqhetero]
  (p = 0.00046 [rqhetero]), 3.01 [rqhetero], 2.26 [rqhetero]. In an OLS fit
  adjusting for the node's degree (as a deviation from the expected degree),
  its local clustering and density, the effect falls by 1.74 [rqhetero] points
  per 10 list positions (coefficient −1.74 [rqhetero],
  p = 0.018 [rqhetero]). Without
  a primer accuracy does not depend on position (38.0 [rqhetero],
  37.3 [rqhetero], 38.6 [rqhetero]).
- **What goes wrong without it** (finished responses, p = .35 and .50): the
  model mostly under-counts (in per cent of finished responses,
  51.0 [rqerrors] under and 14.6 [rqerrors] over; 42.5 [rqerrors] and
  17.2 [rqerrors] with `clustering`). Most errors are in copying the node's
  neighbour list (54.9 [rqbehaviour] per cent of responses), not in counting
  it (10.8 [rqbehaviour]); of the wrong answers, 65.1 [rqbehaviour] per cent
  list too few neighbours. With `clustering`, a response
  misses 0.86 [rqbehaviour] neighbours on average, against 1.49 [rqbehaviour]
  without.
- **Fixed and broken** (all seven densities): of the responses `clustering` corrects, the
  no-primer error was a copying error in 224 [rqbehaviour] and a miscount in
  61 [rqbehaviour]; of those it breaks, 176 [rqbehaviour] and
  57 [rqbehaviour]. No response mentions clustering (0.0 [rqbehaviour]).
- **At p ≥ .65** the plain model over-counts (73.1 [rqerrors] per cent of
  finished no-primer responses, mean signed error 3.6 [rqerrors]), matching its
  answers of 39 (§3).

## 9. Limits

- One model family. The density runs use Qwen3-1.7B only; Qwen3-8B is run only
  at fixed mean degree, where it is near ceiling in five of the eight cells.
- The per-level effects are many tests; read them by their q-values.
- `prompts.degfixdegfill*.jsonl` (the fixed-mean-degree graphs with `filler`
  and `degree`) were built but never run.
