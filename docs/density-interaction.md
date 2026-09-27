# Does density change what a primer does?

**Status: exploratory.** A systematic test, over every arm, task and primer of
the 40-node sweep, of whether a primer's effect on accuracy changes with the
graph's edge density, and whether density matters beyond how hard the task
already is without a primer. Intervals and tests are unadjusted except where BH
is named. Nothing here is checked by `tests/test_results_docs.py`; the verified
results are in [`results/n40-sweep.md`](results/n40-sweep.md) and
[`results/density-followups.md`](results/density-followups.md).

Source: `outputs/n40-sweep/density_interaction.txt`, printed by
`scripts/density_interaction.py` from `outputs/n40-sweep/frame.csv`. Reproduce:

    PYTHONPATH=. python scripts/density_interaction.py > outputs/n40-sweep/density_interaction.txt

Data: the main experiment's runs, Qwen3-1.7B and Qwen3-4B with and without
thinking (*1.7B*, *1.7B-T*, *4B*, *4B-T*), the four tasks whose answer varies,
every primer (and `filler`) against `none` on the same graphs. Accuracy follows
rule R1: a truncated response is never correct. A number followed by a tag, as in
`+27.0 [dxhethi]`, is printed in that tag's block of the source.

## What was already known

Per-density effects were computed before, but the question was never asked for
every arm, task and primer at once.

- `n40-sweep.md` §4: the `degree` primer's effect on `node_degree` changes sign
  with density for plain 4B, and effects are organised by no-primer accuracy
  (the bands).
- §6: plain 1.7B's `edge_existence` collapses to "yes" at high density.
- `density-followups.md`: `node_degree` at 400 graphs per density, with a
  fixed-mean-degree grid that separates density from graph size.
- `outputs/n40-sweep/primer_cells.csv` holds every (arm, task, density, primer)
  effect, but not a test of whether it varies.

## Method

- **Does an effect vary with density? (`[dxhet]`, `[dxhethi]`)** For each arm ×
  task × primer, each graph gives a paired difference (primer − none, correct
  under R1). The statistic is how far its mean moves across the densities (the
  between-density sum of squares). Its p-value comes from 2,000 shuffles of the
  density labels across graphs, each graph's pair kept intact. BH is applied over
  all tests of a table. The main sweep has four densities (p = .10–.50) for all
  four tasks: 96 tests. `node_degree` and `edge_existence` also have p = .65–.85:
  48 tests over all seven densities. The plain arms ran those with a quarter of
  the token budget, so density and budget move together there.
- **Beyond difficulty? (`[dxbase]`)** Each (arm, task, density, primer) cell under
  the truncation flag gets its no-primer accuracy band (the bands of n40-sweep §4)
  from half of its graphs and its effect from the other half, which avoids
  regression to the mean. A model effect ~ band + density gives the density
  coefficient: points of effect per +0.1 of density at a fixed band. It is tested
  by shuffling density among cells of the same band.

## 1. Density changes a minority of effects

On the main sweep, 16 [dxvary] of 96 [dxvary] effects vary with density at
q < .05. 7 [dxvary] of them are `edge_count` cells where a density has 15% or
more truncated responses, so what changes there is how often the thinking arms
finish. The 9 [dxvary] under the truncation flag, effect at p = .10 / .20 / .35 / .50:

| Task | Arm | Primer | Effects | Slope per +0.1 |
|---|---|---|---|---|
| `edge_existence` | 1.7B | `all` | +0, +8, +12, +26 [dxvary] | +6.1 [dxvary] |
| `edge_existence` | 1.7B | `degree` | −6, −3, +1, +24 [dxvary] | +7.1 [dxvary] |
| `edge_existence` | 1.7B | `filler` | −5, −20, −32, −6 [dxvary] | −0.6 [dxvary] |
| `edge_existence` | 4B | `degree` | +3, −2, −4, −16 [dxvary] | −4.4 [dxvary] |
| `edge_existence` | 4B | `rwse` | +0, −4, −13, −23 [dxvary] | −5.8 [dxvary] |
| `connected_nodes` | 1.7B-T | `degree` | +0, −4, +13, +16 [dxvary] | +5.0 [dxvary] |
| `connected_nodes` | 4B | `rwse` | −10, −5, +5, +10 [dxvary] | +5.2 [dxvary] |
| `connected_nodes` | 1.7B | `all` | −2, −7, +15, +2 [dxvary] | +2.6 [dxvary] |
| `node_degree` | 1.7B-T | `all` | +8, +1, +22, +18 [dxvary] | +3.9 [dxvary] |

Over all seven densities, 15 [dxvaryhi] of the 48 [dxvaryhi] `node_degree` and
`edge_existence` effects vary with density, among them:

- plain 4B `degree` on `node_degree`, from −6.0 [dxhethi] at p = .10 to
  +27.0 [dxhethi] at p = .85 (the sign flip of n40-sweep §4);
- 1.7B-T `degree` on `node_degree`, rising to +40.0 [dxhethi] at p = .85;
- plain 4B `clustering` and `rwse` on `node_degree`, between −4 and +1 points up
  to p = .50 and then +18.0 [dxhethi] and +13.0 [dxhethi] at p = .85 and .75;
- plain 4B `all` on `node_degree`, which loses most at p = .65 (−37.0 [dxhethi]).

4B-T's effects do not vary with density outside the truncation-flagged
`edge_count` cells.

**What it tells us.** Where a primer's effect depends on density, it mostly grows
for the weaker arms as the graph gets denser and the task harder: gains appear at
p = .35–.50 on `connected_nodes` and `edge_existence`, and at p ≥ .65 on
`node_degree`. For plain 4B on `edge_existence`, `degree` and `rwse` lose more as
density rises. The strongest arm is unmoved at every density.

## 2. At a fixed difficulty, density adds almost nothing

Density coefficient at a fixed no-primer accuracy band (main sweep, cells under
the truncation flag):

| Cells | n | Per +0.1 of density | p |
|---|---|---|---|
| Side information, all tasks | 275 [dxbase] | +0.05 [dxbase] | 0.85 [dxbase] |
| Answer-carrying | 42 [dxbase] | −1.18 [dxbase] | 0.28 [dxbase] |
| Side information, `node_degree` | 64 [dxbase] | −0.93 [dxbase] | 0.006 [dxbase] |
| Side information, `edge_existence` | 96 [dxbase] | +0.18 [dxbase] | 0.74 [dxbase] |
| Side information, `connected_nodes` | 96 [dxbase] | +0.49 [dxbase] | 0.43 [dxbase] |

- Adding p = .65–.85 for `node_degree` and `edge_existence` does not change the
  picture overall (side information +0.04 [dxbase], p = 0.77 [dxbase];
  answer-carrying −1.30 [dxbase], p = 0.059 [dxbase]).
- The one per-task residual does not replicate. `node_degree`'s −0.93 falls to
  −0.13 [dxbase] (p = 0.58 [dxbase]) with the high densities added, while
  `edge_existence` rises to +0.43 [dxbase] (p = 0.018 [dxbase]), where the plain
  arms' budget also changes.
- **What it tells us.** Density changes a primer's effect by changing how hard the
  task is without it. Once the no-primer accuracy is held fixed, density itself
  adds nothing detectable. This extends to all four tasks what n40-sweep §4 shows
  with its bands and density-followups §6 shows for `node_degree` with a fixed
  mean degree.

## Limits

- 96 and 48 tests, corrected by BH within each table. The six per-task density
  coefficients are not corrected across one another; `node_degree`'s p = 0.006
  would survive a correction for six, `edge_existence`'s p = 0.018 would not.
- The density coefficient is linear in density within five coarse bands; a
  non-linear density effect at a fixed band would be missed.
- At p ≥ .65 the plain arms ran with 2048 tokens, so density and budget are
  confounded there.
- Truncation-flagged cells (`edge_count`, thinking arms) are tested in section 1
  but left out of section 2. Their effects are mostly about finishing within
  the budget.
