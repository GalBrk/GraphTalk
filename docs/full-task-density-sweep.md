# The full-task, full-condition density sweep, scored in full

Reproduces and extends the summary already in `docs/primer-effects-and-power.md`
("The full-task, full-condition density sweep (2026-09-13)") with the complete
per-task, per-density tables behind it, for the two arms that are complete:
`qwen3-1.7b` and `qwen3-1.7b-think`.

## Design (from `docs/primer-effects-and-power.md`)

Every earlier density-at-fixed-n run deliberately narrowed to one or two tasks
and 2-4 conditions, for the reasons given in that document's "Density at a
fixed size" (truncation, degeneracy, contamination). This run does the
opposite on purpose: **n=40, all 4 density levels {0.10, 0.20, 0.35, 0.50},
all 7 conditions, all 6 tasks, 100 graphs/cell** -- 16,800 prompts
(`prompts.densfull40.jsonl`), tag `densfull40`.

Scored with `scripts/score_full_density_sweep.py` (added alongside this run),
which extends `score_density_sweep.py` with `task` as a third grouping key --
pooling all 6 tasks into one (density, condition) cell, which the older
script does, mixes `node_count` exact-match with `connected_nodes` F1 into a
meaningless average. Bar read from `shortcuts.json`; `hit_cap` rows dropped,
not scored zero; per-task Benjamini-Hochberg across conditions.

**Status:** `qwen3-1.7b` and `qwen3-1.7b-think` complete, 16,800/16,800 rows
each (`runs/<model>.densfull40.shard*of25.jsonl`).

Reproduce:

```bash
PYTHONPATH=. python scripts/score_full_density_sweep.py \
    --responses "runs/qwen3-1.7b.densfull40.shard*of25.jsonl" \
    --shortcuts shortcuts.json

PYTHONPATH=. python scripts/score_full_density_sweep.py \
    --responses "runs/qwen3-1.7b-think.densfull40.shard*of25.jsonl" \
    --shortcuts shortcuts.json
```

Per-row scored CSVs for manual review (one row per response, with the
extracted answer and correctness alongside the raw model text) are in
`analysis/qwen3-1.7b.densfull40.rows.csv` and
`analysis/qwen3-1.7b-think.densfull40.rows.csv`.

## Read the hit_cap rate before anything else

| | non-think | think |
|---|---|---|
| overall | 4.8% | 15.4% |
| `edge_count` | **22.4%** | **81.0%** |
| `cycle_check` | 6.1% | 6.5% |
| everything else | ≤0.8% | ≤2.5% |

`edge_count` under thinking is mostly not data: several (density, condition)
cells are **100% capped** at p>=0.35 even at the raised 8192-token budget, and
those cells score `nan` in the scorer's output rather than a number. Treat
`edge_count`-think at p>=0.35 as absent, not as a result; p=0.10/0.20 are
usable but thin.

## `qwen3-1.7b` (plain arm), full per-task tables

```
TASK: connected_nodes
blind bar (best constant answer):
  p=0.1    modal 'No nodes'  bar 0.019  (n=699)
  p=0.2    modal '0, 9, 12, 29, 36, 38'  bar 0.010  (n=700)
  p=0.35   modal '4, 8, 10, 11, 18, 22, 24, 26, 27, 29, 30, 31, 35, 36'  bar 0.010  (n=700)
  p=0.5    modal '0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 25, 29, 30, 31, 33'  bar 0.010  (n=700)

mean score by density x condition (hit_cap dropped):
  p               all  clustering  components      degree      filler        none        rwse
  0.1           0.960       0.960       0.978       0.978       0.969       0.970       0.989
  0.2           0.978       0.990       0.982       0.993       0.982       0.987       0.985
  0.35          0.961       0.975       0.968       0.978       0.957       0.969       0.956
  0.5           0.961       0.965       0.959       0.969       0.942       0.963       0.950

shortcut bar (primer-only solver, from shortcuts.json):
  all p         0.352       0.082       0.082       0.208       0.082       0.082       0.082

pooled across densities, paired vs 'none' (exact McNemar):
          all - none: n=  399 win    1 lose    5 delta -0.0100 p=0.2188  bar-adj -0.2795
   clustering - none: n=  400 win    1 lose    1 delta +0.0000 p=1.0000  bar-adj +0.0003
   components - none: n=  400 win    1 lose    0 delta +0.0025 p=1.0000  bar-adj -0.0004
       degree - none: n=  400 win    1 lose    1 delta +0.0000 p=1.0000  bar-adj -0.1186
       filler - none: n=  400 win    1 lose    1 delta +0.0000 p=1.0000  bar-adj -0.0096
         rwse - none: n=  400 win    1 lose    0 delta +0.0025 p=1.0000  bar-adj -0.0018

TASK: cycle_check
blind bar: 1.000 at every density (gold is "Yes" almost everywhere)

mean score by density x condition (hit_cap dropped):
  p               all  clustering  components      degree      filler        none        rwse
  0.1           0.969       1.000       0.978       0.989       1.000       0.988       0.946
  0.2           1.000       1.000       1.000       0.857       1.000       1.000       0.940
  0.35          0.990       1.000       0.989       0.851       1.000       1.000       0.870
  0.5           0.990       1.000       1.000       0.831       1.000       1.000       0.930

shortcut bar: all p  0.946 (all) / 0.832 (clustering/filler/none/rwse) / 1.000 (components) / 0.946 (degree)

pooled across densities, paired vs 'none' (exact McNemar):
          all - none: n=  362 win    1 lose    5 delta -0.0110 p=0.2188  bar-adj -0.1250
   clustering - none: n=  362 win    1 lose    0 delta +0.0028 p=1.0000  bar-adj +0.0028
   components - none: n=  343 win    0 lose    2 delta -0.0058 p=0.5000  bar-adj -0.1738
       degree - none: n=  301 win    0 lose   38 delta -0.1262 p=0.0000  *sig(BH)  bar-adj -0.2402
       filler - none: n=  341 win    1 lose    0 delta +0.0029 p=1.0000  bar-adj +0.0029
         rwse - none: n=  358 win    1 lose   28 delta -0.0754 p=0.0000  *sig(BH)  bar-adj -0.0754

TASK: edge_count
blind bar: 0.046-0.083 across densities (real headroom)

mean score by density x condition (hit_cap dropped) -- heavily censored, see hit_cap table below:
  p               all  clustering  components      degree      filler        none        rwse
  0.1           0.000       0.000       0.000       0.067       0.000       0.043       0.000
  0.2           0.011       0.031       0.013       0.032       0.033       0.022       0.022
  0.35          0.060       0.000       0.000       0.010       0.000       0.000       0.000
  0.5           0.054       0.000       0.000       0.087       0.000       0.000       0.000

capped rows dropped by density x condition (out of 100/cell):
  p               all  clustering  components      degree      filler        none        rwse
  0.1              26          10          20          10          15          31          10
  0.2               9           3          20           6           9           9           9
  0.35             33          26          33           2          40          50          21
  0.5              44          33          36          20          41          29          32

pooled across densities, paired vs 'none' (exact McNemar):
          all - none: n=  211 win    8 lose    3 delta +0.0237 p=0.2266  bar-adj -0.9583
   clustering - none: n=  238 win    3 lose    4 delta -0.0042 p=1.0000  bar-adj -0.1342
   components - none: n=  211 win    1 lose    4 delta -0.0142 p=0.3750  bar-adj -0.0142
       degree - none: n=  253 win   13 lose    3 delta +0.0395 p=0.0213  bar-adj -0.9425
       filler - none: n=  224 win    2 lose    3 delta -0.0045 p=1.0000  bar-adj -0.0045
         rwse - none: n=  239 win    2 lose    4 delta -0.0084 p=0.6875  bar-adj -0.0084

TASK: edge_existence
blind bar: 0.900 (p=0.1) down to 0.520 (p=0.5)

mean score by density x condition (hit_cap dropped):
  p               all  clustering  components      degree      filler        none        rwse
  0.1           0.890       0.840       0.860       0.830       0.840       0.890       0.919
  0.2           0.810       0.740       0.620       0.700       0.530       0.730       0.680
  0.35          0.780       0.720       0.500       0.670       0.340       0.660       0.760
  0.5           0.800       0.670       0.510       0.780       0.480       0.540       0.670

pooled across densities, paired vs 'none' (exact McNemar):
          all - none: n=  400 win   75 lose   29 delta +0.1150 p=0.0000  *sig(BH)  bar-adj -0.1810
   clustering - none: n=  400 win   48 lose   33 delta +0.0375 p=0.1193  bar-adj -0.1845
   components - none: n=  400 win   11 lose   44 delta -0.0825 p=0.0000  *sig(BH)  bar-adj -0.0825
       degree - none: n=  400 win   49 lose   33 delta +0.0400 p=0.0970  bar-adj -0.2560
       filler - none: n=  400 win   10 lose   73 delta -0.1575 p=0.0000  *sig(BH)  bar-adj -0.1575
         rwse - none: n=  399 win   55 lose   34 delta +0.0526 p=0.0334  bar-adj -0.0914

TASK: node_count
blind bar: 1.000 at every density (gold is trivially 40, n fixed)

mean score by density x condition (hit_cap dropped):
  p               all  clustering  components      degree      filler        none        rwse
  0.1           0.700       0.950       0.190       0.030       0.160       0.020       0.320
  0.2           0.520       1.000       0.410       0.050       0.700       0.000       0.170
  0.35          0.940       1.000       0.700       0.450       0.990       0.280       0.550
  0.5           0.980       1.000       0.810       1.000       1.000       0.960       0.300

pooled across densities, paired vs 'none' (exact McNemar):
          all - none: n=  400 win  193 lose    5 delta +0.4700 p=0.0000  *sig(BH)  bar-adj -0.4660
   clustering - none: n=  400 win  269 lose    0 delta +0.6725 p=0.0000  *sig(BH)  bar-adj -0.2635
   components - none: n=  400 win  113 lose   28 delta +0.2125 p=0.0000  *sig(BH)  bar-adj +0.2005
       degree - none: n=  400 win   43 lose   16 delta +0.0675 p=0.0006  *sig(BH)  bar-adj -0.8685
       filler - none: n=  400 win  161 lose    2 delta +0.3975 p=0.0000  *sig(BH)  bar-adj -0.5385
         rwse - none: n=  400 win   87 lose   79 delta +0.0200 p=0.5871  bar-adj -0.9160

TASK: node_degree
blind bar: 0.170-0.210 (real headroom)

mean score by density x condition (hit_cap dropped):
  p               all  clustering  components      degree      filler        none        rwse
  0.1           0.970       0.930       0.920       0.940       0.960       0.910       0.940
  0.2           0.900       0.820       0.820       0.900       0.730       0.800       0.840
  0.35          0.580       0.470       0.440       0.590       0.410       0.410       0.475
  0.5           0.380       0.320       0.310       0.290       0.300       0.300       0.330

pooled across densities, paired vs 'none' (exact McNemar):
          all - none: n=  400 win   77 lose   36 delta +0.1025 p=0.0001  *sig(BH)  bar-adj -0.8155
   clustering - none: n=  400 win   53 lose   41 delta +0.0300 p=0.2564  bar-adj +0.0300
   components - none: n=  400 win   36 lose   29 delta +0.0175 p=0.4570  bar-adj +0.0175
       degree - none: n=  400 win   57 lose   27 delta +0.0750 p=0.0014  *sig(BH)  bar-adj -0.8430
       filler - none: n=  400 win   32 lose   34 delta -0.0050 p=0.9022  bar-adj -0.0050
         rwse - none: n=  399 win   53 lose   37 delta +0.0401 p=0.1133  bar-adj -0.4939
```

## `qwen3-1.7b-think`, full per-task tables

```
TASK: connected_nodes -- near ceiling throughout (0.93-1.00), nothing significant.
pooled across densities, paired vs 'none':
   all -0.0077 p=0.375 / clustering +0.0025 p=1.0 / components +0.0026 p=1.0 /
   degree -0.0051 p=0.5 / filler -0.0025 p=1.0 / rwse -0.0127 p=0.0625

TASK: cycle_check -- at ceiling (1.000) at nearly every density x condition cell.
None of the 6 non-control conditions differ from 'none' (all p>=0.5, most
p=1.0, zero discordant pairs in most cells). `degree`/`components` shed
12-34% of rows to hit_cap per level -- survivorship, not evidence.

TASK: edge_count -- mostly not data. capped rows dropped by density x condition
(out of ~50-100 rows/cell after dedup):
  p               all  clustering  components      degree      filler        none        rwse
  0.1              70          66          56          80          58          50          60
  0.2              78          68          73          77          76          77          87
  0.35             67          96         100          83          98         100         100
  0.5              63         100         100          84         100         100         100
Several cells are 100% capped (nan in mean score); the one "significant" McNemar
result (degree -61.5pp, p=0.0078) rests on n=13 surviving pairs and should not
be trusted.

TASK: edge_existence -- near ceiling (0.95-1.00), small negative trend for
degree/all that does not clear significance at the pooled level:
   all -0.0201 p=0.0386 (not BH-significant) / degree -0.0227 p=0.0225 (not BH-significant)
   clustering -0.0101 p=0.289 / components -0.0075 p=0.508 / filler -0.0075 p=0.508 / rwse -0.0126 p=0.227

TASK: node_count -- at or near ceiling (0.95-1.00) everywhere; no condition
differs from 'none' at a level that survives BH (best case p=0.0625, all four
of all/clustering/filler/degree tied on the same 5 discordant pairs).

TASK: node_degree
mean score by density x condition (hit_cap dropped):
  p               all  clustering  components      degree      filler        none        rwse
  0.1           0.980       0.949       0.939       0.980       0.980       0.928       0.960
  0.2           0.970       0.970       0.890       0.939       0.850       0.950       0.929
  0.35          0.820       0.610       0.620       0.837       0.560       0.600       0.650
  0.5           0.780       0.590       0.580       0.820       0.550       0.600       0.636

pooled across densities, paired vs 'none' (exact McNemar):
          all - none: n=  396 win   63 lose   17 delta +0.1162 p=0.0000  *sig(BH)  bar-adj -0.8018
   clustering - none: n=  396 win   31 lose   27 delta +0.0101 p=0.6940  bar-adj +0.0101
   components - none: n=  395 win   21 lose   25 delta -0.0101 p=0.6587  bar-adj -0.0101
       degree - none: n=  376 win   59 lose   15 delta +0.1170 p=0.0000  *sig(BH)  bar-adj -0.8010
       filler - none: n=  396 win   24 lose   38 delta -0.0354 p=0.0980  bar-adj -0.0354
         rwse - none: n=  394 win   41 lose   31 delta +0.0254 p=0.2888  bar-adj -0.5086
```

## Pooled accuracy by primer, per task (each model)

Same data as the per-task tables above, pooled across all four densities and
reshaped task x primer for a per-model read. `connected_nodes` is set-F1, not
exact match. Delta is percentage points vs `none` for that task; bold marks
the largest moves.

### qwen3-1.7b (plain)

| task | none | components | clustering | rwse | degree | filler | all |
|---|---|---|---|---|---|---|---|
| connected_nodes | 97.21% | 97.16% (−0.04) | 97.24% (+0.03) | 97.02% (−0.18) | 97.94% (+0.74) | 96.24% (−0.96) | 96.50% (−0.70) |
| cycle_check | 99.73% | 99.20% (−0.53) | 100.00% (+0.27) | 92.11% (**−7.61**) | 88.52% (**−11.21**) | 100.00% (+0.27) | 98.74% (−0.99) |
| edge_count | 1.78% | 0.34% (−1.44) | 0.91% (−0.86) | 0.61% (−1.17) | 4.70% (+2.92) | 1.02% (−0.76) | 2.78% (+1.00) |
| edge_existence | 70.50% | 62.25% (**−8.25**) | 74.25% (+3.75) | 75.69% (+5.19) | 74.50% (+4.00) | 54.75% (**−15.75**) | 82.00% (**+11.50**) |
| node_count | 31.50% | 52.75% (+21.25) | 98.75% (**+67.25**) | 33.50% (+2.00) | 38.25% (+6.75) | 71.25% (**+39.75**) | 78.50% (**+47.00**) |
| node_degree | 60.50% | 62.25% (+1.75) | 63.50% (+3.00) | 64.66% (+4.16) | 68.00% (+7.50) | 60.00% (−0.50) | 70.75% (**+10.25**) |

### qwen3-1.7b-think

| task | none | components | clustering | rwse | degree | filler | all |
|---|---|---|---|---|---|---|---|
| connected_nodes | 98.56% | 98.25% (−0.31) | 98.55% (−0.01) | 96.25% (−2.31) | 97.74% (−0.82) | 98.18% (−0.37) | 98.07% (−0.49) |
| cycle_check | 100.00% | 100.00% (+0.00) | 100.00% (+0.00) | 100.00% (+0.00) | 99.34% (−0.66) | 100.00% (+0.00) | 100.00% (+0.00) |
| edge_count | 86.30% | 84.51% (−1.79) | 82.86% (−3.44) | 94.34% (+8.04) | 65.79% (**−20.51**) | 83.82% (−2.48) | 68.03% (**−18.27**) |
| edge_existence | 99.25% | 98.50% (−0.75) | 98.25% (−1.00) | 97.99% (−1.25) | 96.98% (−2.26) | 98.50% (−0.75) | 97.25% (−2.00) |
| node_count | 98.64% | 99.73% (+1.09) | 100.00% (+1.36) | 100.00% (+1.36) | 99.23% (+0.58) | 100.00% (+1.36) | 100.00% (+1.36) |
| node_degree | 76.83% | 75.63% (−1.20) | 77.94% (+1.12) | 79.35% (+2.52) | 89.71% (**+12.88**) | 73.43% (−3.39) | 88.72% (**+11.90**) |

Thinking mostly flattens variance across primers -- `connected_nodes`,
`cycle_check`, `edge_existence` and `node_count` are all near ceiling for
every condition, so there is little room for a primer to move anything. The
two tasks where it doesn't flatten move in opposite directions: `node_degree`
opens up a *bigger* `degree`/`all` gap under thinking (+12.9/+11.9 pp vs
+7.5/+10.3 pp plain), while `edge_count` turns `degree`/`all` from small
plain-arm moves into large think-arm harm (−20.5/−18.3 pp) -- see the hit_cap
caveats above before reading that one as a reasoning effect rather than
truncation.

## Which results are statistically significant

Significance is read from the exact-McNemar tests already computed per task
above (pooled across densities, Benjamini-Hochberg corrected within each
task's 6 comparisons against `none`). A significant raw delta is not the same
as a real effect here -- the bar-adjusted delta (raw delta minus what the
shortcut-only solver gains from the same primer) is what tells the two apart.

**Real, bar-clean significant effects** (the shortcut bar does not explain
them):

| model | task | primer | effect | p |
|---|---|---|---|---|
| plain | cycle_check | degree | **−12.6 pp (harm)** | <0.0001 |
| plain | cycle_check | rwse | **−7.5 pp (harm)** | <0.0001 |
| plain | edge_existence | filler | **−15.8 pp (harm)** | <0.0001 |
| plain | edge_existence | components | **−8.3 pp (harm)** | <0.0001 |
| plain | node_count | components | **+21.3 pp raw / +20.1 pp bar-adjusted (real gain)** | <0.0001 |

`node_count`/`components` is the one genuinely good result in this run:
`components` measurably helps beyond what a primer-only solver could already
extract from the same text.

**Significant but shortcut-explained** (the primer states or implies the
answer, so the "effect" is retrieval, not reasoning):

- `node_count`: `all` (+47.0 pp), `clustering` (+67.3 pp), `filler`
  (+39.8 pp) -- all bar-adjusted strongly *negative* (−0.26 to −0.54), i.e.
  the shortcut solver gains even more than the model does from the same text.
- `node_degree`: `all` and `degree`, in both arms -- bar-adjusted −0.80 to
  −0.84 in both plain and think.
- `edge_existence`: `all` (+11.5 pp) -- bar-adjusted −0.18.
- `edge_count`-think: `degree` (−61.5 pp, p=0.0078) -- technically clears BH,
  but rests on only **13** surviving pairs after `hit_cap` filtering (this
  cell is over 80% capped); not trustworthy regardless of the p-value.

**Not significant anywhere:** `connected_nodes` in either arm, and
`cycle_check`/`edge_existence`/`node_count` under thinking -- all near
ceiling with nothing left to detect.

**Net:** 2 real harms (`cycle_check` under `degree`/`rwse`, plain arm only),
1 real gain (`node_count` under `components`, plain arm only), and every other
"significant" cell is either a shortcut ghost or too underpowered
(`edge_count`-think) to trust.

## Cross-checked against the original 5-19-node sweep's own significance pipeline

Everything above uses `score_full_density_sweep.py`'s own per-task exact
McNemar + per-task BH correction. The original 5-19-node sweep answers "is
this significant" with a different tool: `scripts/build_sweep_frame.py` +
`scripts/check_significance.py`, `graphtalk/significance.py`'s clustered
permutation test + cluster bootstrap + Benjamini-Hochberg, built specifically
because per-cell McNemar is underpowered (`docs/sweep-findings.md`, "The
McNemar analysis is underpowered"). Ran it here, unmodified, against the same
`densfull40` run files:

```bash
PYTHONPATH=. python scripts/build_sweep_frame.py \
    --responses "runs/qwen3-1.7b.densfull40.shard*of25.jsonl" \
                "runs/qwen3-1.7b-think.densfull40.shard*of25.jsonl" \
    --shortcuts shortcuts.json --out analysis/densfull40_frame.csv

PYTHONPATH=. python scripts/check_significance.py \
    --frame analysis/densfull40_frame.csv --metric exact --no-mde \
    --out analysis/densfull40_significance.csv
```

One fix was needed first: `build_sweep_frame.py`'s loader (unlike
`score_full_density_sweep.py`'s) doesn't dedupe, and this data carries the
same "a handful of duplicate rows survive preemption/resume" artifact that
script already guards against -- 4 exact-duplicate rows (same key, same
score) were dropped from the frame before testing.

**Pooled across all 6 tasks and all 4 densities** (the tracked sweep's own
"cross-condition comparison" view, `qwen3-1.7b` plain arm):

| condition | delta | 95% CI | p (perm) | BH-sig |
|---|---|---|---|---|
| clustering | +13.9 pp | [+12.0, +15.8] | 0.0001 | yes |
| all | +13.1 pp | [+11.3, +15.0] | 0.0001 | yes |
| filler | +3.3 pp | [+1.6, +5.0] | 0.0001 | yes |
| components | +2.8 pp | [+1.1, +4.3] | 0.0010 | yes |
| rwse | +1.8 pp | [+0.0, +3.7] | 0.0542 | no |
| degree | +1.2 pp | [−0.5, +3.0] | 0.1828 | no |

Every one of these pooled deltas is dominated by `node_count`, the task with
by far the largest raw gains -- exactly why `score_full_density_sweep.py`
tests per-task rather than pooling across tasks in the first place. Read this
table as "does this condition move the average at all," not as "is this a
real effect" -- that needs the bar-adjusted, per-task view.

**Per-task, same clustered-permutation machinery**: agrees with the per-task
table above on every cell in `edge_count`, `edge_existence`, `node_count`, and
`node_degree` -- same direction, same BH-significance call, all 24 pairs
across the two tables. `node_count`/`components`'s real gain reproduces
almost exactly (+21.3 pp there, +21.2 pp here, both bar-independent
significant).

`cycle_check` is the exception, and it disagrees on 3 of 6 conditions:

| condition | this doc (`hit_cap` dropped) | this pipeline (`hit_cap` forced wrong) |
|---|---|---|
| degree | −12.6 pp, sig | −17.5 pp, sig |
| rwse | **−7.5 pp, sig (harm)** | **−0.3 pp, not sig** |
| clustering | **+0.3 pp, not sig** | **+8.5 pp, sig (gain)** |
| all | **−1.1 pp, not sig** | **+7.0 pp, sig (gain)** |
| components | −0.6 pp, not sig | +2.0 pp, not sig |
| filler | +0.3 pp, not sig | +3.0 pp, not sig |

This is not the more rigorous test catching something the per-task table
missed -- it's the two pipelines applying opposite rules for what a truncated
response counts as (this document drops `hit_cap` rows; `check_significance.py`
forces them to score 0, its own deliberate choice for the tracked sweep, see
`graphtalk/analysis.py::build_frame`'s docstring), and `cycle_check` is where
that choice actually bites. Checking the frame directly confirms it --
`none`'s own non-termination rate on this task is far higher than most
conditions it's compared against:

| condition | non-terminating rate on `cycle_check` |
|---|---|
| `none` | 9.00% |
| `degree` | 17.25% |
| `components` | 6.50% |
| `filler` | 6.25% |
| `rwse` | 1.75% |
| `clustering` | 0.75% |
| `all` | 1.00% |

Forcing every non-terminating row to "wrong" punishes `none` (9.0% capped)
far more than `clustering`/`all` (under 1% capped each), manufacturing an
apparent `clustering`/`all` gain that is really just `none` losing extra rows
to truncation. The same mechanism erases `rwse`'s harm (`rwse` is capped less
than `none` too, so `none`'s own-goal closes most of the gap). `degree`, capped
even more than `none` (17.25%), gets *more* harmful under the forced-wrong
convention, not less -- consistent with the same story running the other way.

Net: `degree`'s harm on `cycle_check` is the one finding that survives both
conventions and both tools -- treat it as the robust result. `rwse`'s harm and
`clustering`/`all`'s apparent gains on this task are convention-dependent
artifacts of differential truncation rates, not established effects, under
either tool alone.

**A result this pipeline adds that has no counterpart above**: thinking-arm
non-termination rate vs `none`, pooled across all 6 tasks, tested the same way
`check_significance.py` tests it for the tracked sweep's own thinking arm:

| condition | delta (non-termination rate) | p (perm) | BH-sig |
|---|---|---|---|
| all | −3.4 pp (less truncation) | 0.0001 | yes |
| degree | +3.4 pp (more truncation) | 0.0001 | yes |
| filler | −1.8 pp (less truncation) | 0.0007 | yes |
| clustering | −1.3 pp (less truncation) | 0.0161 | yes |
| rwse | −1.0 pp | 0.0525 | no |
| components | +1.2 pp | 0.0665 | no |

`degree` reliably makes the thinking arm truncate more often; `clustering`,
`filler`, and `all` reliably make it truncate less. This backs the hit_cap-rate
story in "Read the hit_cap rate before anything else" with an actual
BH-corrected p-value instead of an eyeballed percentage.

**Bottom line:** yes, this is the same analysis as the original 5-19-node
sweep, mechanically -- same unmodified tool, same clustered-permutation +
bootstrap + BH machinery, run against this run data instead of the tracked
sweep's. It reproduces every real/shortcut-ghost call made above for 5 of the
6 tasks. `cycle_check` is the one place it disagrees, and the disagreement is
fully explained by the two pipelines' opposite, both-deliberate conventions
for a truncated response interacting with this task's unusually uneven
per-condition `hit_cap` rates -- not by this pipeline being more sensitive.

## Synthesis

- **The `clustering` -> `node_degree` headline (job 866467's clean +3.8 pp,
  p=0.0017) does not replicate at significance in either arm here**, once
  pooled across all four densities: +3.0 pp (p=0.256) plain, +1.0 pp (p=0.694)
  think. Same direction every time it has been measured, never negative, but
  underpowered at n=400 pooled pairs when spread this thin across densities
  and conditions -- consistent with the "power check, not independent
  replication" framing already in `primer-effects-and-power.md`.
- **`cycle_check`: `degree` and `rwse` cause real, bar-clean harm in the plain
  arm** (-12.6 pp and -7.5 pp, both p<0.0001, bar-adjusted delta equal to the
  raw delta since both conditions share `none`'s shortcut bar). **The thinking
  arm erases it completely** -- ceiling at 1.000 everywhere, though partly by
  survivorship (`degree`/`components` lose 12-34% of rows to `hit_cap`).
- **`edge_existence`: `components` (-8.3 pp) and `filler` (-15.8 pp) cause
  real, bar-clean harm in the plain arm** (both p<0.0001), replicating this
  project's length-cost finding on a task it hadn't been measured on before.
  Thinking again flattens the task to ceiling (0.95-1.00), removing the
  headroom needed to see it.
- **Most `node_count`/`node_degree` "wins" are shortcut ghosts, not
  reasoning gains.** `all`, `degree`, and `filler` all post large, nominally
  significant positive deltas on `node_count`, but bar-adjusted they are all
  strongly *negative* (-0.47 to -0.92) -- the shortcut-only solver gains even
  more from those primers than the model does. `components` is the one
  exception with a genuine bar-adjusted +0.20 on `node_count`.
- **`edge_count` is barely usable at n=40 with 100 graphs/level**, and
  essentially unusable under thinking past p=0.20 -- capped rows run 10-100
  per 100-row cell in the plain arm and reach 100% in several think-arm cells.
  Any headline drawn from this task in this run should be treated as absent,
  not measured, per the same reasoning `primer-effects-and-power.md` already
  applies to the size sweep's exclusion of `edge_count`.
- **`connected_nodes` has no headroom in either arm** (0.93-1.00 throughout),
  exactly as "Density at a fixed size" predicted from the size-sweep numbers --
  nothing significant, nothing to read into it.

## Caveats

- All numbers here exclude GoT-named rows (none exist in this tag) and drop
  `hit_cap` rows rather than scoring them zero, per this project's house rule
  -- see `primer-effects-and-power.md`'s `ec500` write-up for why that choice
  is load-bearing specifically for conditions with an elevated truncation
  rate (`rwse`, and here `edge_count` broadly). Several of the "significant"
  cells above rest on well under 50 discordant pairs and should be read as
  directional, not as independent confirmation.
- `qwen3-4b`/`qwen3-4b-think` on this same tag were still in flight as of the
  last update to `primer-effects-and-power.md`; this document covers only the
  two completed 1.7B arms and should be extended once those land.
