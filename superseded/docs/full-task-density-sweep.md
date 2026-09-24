# The full-task, full-condition density sweep, scored in full

Reproduces and extends the summary already in `docs/primer-effects-and-power.md`
("The full-task, full-condition density sweep (2026-09-13)") with the complete
per-task, per-density tables behind it, for all four completed arms:
`qwen3-1.7b`, `qwen3-1.7b-think`, `qwen3-4b`, and `qwen3-4b-think`.

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

**Status:** all four arms complete, 16,800/16,800 rows each
(`runs/<model>.densfull40.shard*of25.jsonl`). `qwen3-4b`/`qwen3-4b-think`
landed 2026-09-16 (`b202998`), after the two 1.7B arms.

Reproduce:

```bash
PYTHONPATH=. python scripts/score_full_density_sweep.py \
    --responses "runs/qwen3-1.7b.densfull40.shard*of25.jsonl" \
    --shortcuts shortcuts.json

PYTHONPATH=. python scripts/score_full_density_sweep.py \
    --responses "runs/qwen3-1.7b-think.densfull40.shard*of25.jsonl" \
    --shortcuts shortcuts.json

PYTHONPATH=. python scripts/score_full_density_sweep.py \
    --responses "runs/qwen3-4b.densfull40.shard*of25.jsonl" \
    --shortcuts shortcuts.json

PYTHONPATH=. python scripts/score_full_density_sweep.py \
    --responses "runs/qwen3-4b-think.densfull40.shard*of25.jsonl" \
    --shortcuts shortcuts.json
```

Per-row scored CSVs for manual review (one row per response, with the
extracted answer and correctness alongside the raw model text) are in
`analysis/<model>.densfull40.rows.csv` for all four arms.

## Read the hit_cap rate before anything else

| | qwen3-1.7b | qwen3-1.7b-think | qwen3-4b | qwen3-4b-think |
|---|---|---|---|---|
| overall | 4.8% | 15.4% | 0.3% | 11.7% |
| `edge_count` | **22.4%** | **81.0%** | 0.6% | **62.3%** |
| `cycle_check` | 6.1% | 6.5% | 1.0% | 1.9% |
| everything else | ≤0.8% | ≤2.5% | ≤0.1% | ≤3.5% |

`edge_count` under thinking is mostly not data for both model sizes: several
(density, condition) cells are **100% capped** even at the raised 8192-token
budget, and those cells score `nan` in the scorer's output rather than a
number. `qwen3-4b`'s plain arm is essentially untruncated everywhere (0.3%
overall) -- this matters below, since it means the two significance
pipelines' opposite non-terminating-row conventions barely have anything to
disagree about for that arm, unlike `qwen3-1.7b`'s plain arm.

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

## `qwen3-4b` (plain arm), full per-task tables

```
TASK: connected_nodes -- the pooled McNemar test here is degenerate: it pools
`primary` (F1, near-ceiling for every condition, 0.97-1.00) and treats a
score as "wrong" only when it's exactly 0.0, which almost never happens for
a partial-credit metric. It shows 0-1 discordant pairs everywhere and should
not be read as "no effect" -- see the exact-match numbers this task's own
cross-check section below.
mean primary (F1) score by density x condition (hit_cap dropped):
  p               all  clustering  components      degree      filler        none        rwse
  0.1           0.946       0.959       1.000       0.955       0.943       0.985       0.959
  0.2           0.977       0.998       0.999       0.982       0.961       0.986       0.962
  0.35          0.991       0.993       0.996       0.983       0.979       0.985       0.994
  0.5           0.983       0.989       0.995       0.988       0.981       0.993       0.996

TASK: cycle_check
blind bar: 1.000 at every density (gold is "Yes" almost everywhere)

mean score by density x condition (hit_cap dropped):
  p               all  clustering  components      degree      filler        none        rwse
  0.1           1.000       1.000       0.969       0.990       0.990       0.990       1.000
  0.2           1.000       1.000       0.990       0.990       1.000       1.000       1.000
  0.35          1.000       1.000       1.000       0.969       1.000       1.000       1.000
  0.5           0.970       1.000       0.980       0.980       1.000       1.000       1.000

Rescored after the 2026-09-16 extraction fix (see Caveats). Before it, `filler`
read 0.140 at p=0.2 and `none`/`all`/`rwse` 0.86/0.64/0.51 at p=0.5: the
responses said "Yes, there is a cycle" and then defined a cycle as a path
"with no repeated edges or nodes", and the extractor took that "no" as the
answer.

shortcut bar: all p  0.946 (all/degree) / 0.832 (clustering/filler/none/rwse)
/ 1.000 (components)

pooled across densities, paired vs 'none' (exact McNemar):
          all - none: n=  395 win    1 lose    3 delta -0.0051 p=0.6250  bar-adj -0.1191
   clustering - none: n=  395 win    1 lose    0 delta +0.0025 p=1.0000  bar-adj +0.0025
   components - none: n=  391 win    1 lose    6 delta -0.0128 p=0.1250  bar-adj -0.1808
       degree - none: n=  391 win    0 lose    6 delta -0.0153 p=0.0312  bar-adj -0.1293
       filler - none: n=  392 win    0 lose    1 delta -0.0026 p=1.0000  bar-adj -0.0026
         rwse - none: n=  392 win    1 lose    0 delta +0.0026 p=1.0000  bar-adj +0.0026

TASK: edge_count
blind bar: 0.050-0.079 across densities (real headroom)

mean score by density x condition (hit_cap dropped) -- much less censored
than qwen3-1.7b (0.3% overall hit_cap on this arm vs 4.8%):
  p               all  clustering  components      degree      filler        none        rwse
  0.1           0.030       0.000       0.000       0.278       0.100       0.050       0.010
  0.2           0.030       0.010       0.010       0.323       0.020       0.020       0.010
  0.35          0.081       0.020       0.020       0.253       0.010       0.010       0.000
  0.5           0.092       0.000       0.000       0.337       0.000       0.000       0.000

shortcut bar: 1.000 (all/degree) / 0.148 (clustering) / 0.018 (components/
filler/none/rwse)

pooled across densities, paired vs 'none' (exact McNemar):
          all - none: n=  397 win   21 lose    6 delta +0.0378 p=0.0059  *sig(BH)  bar-adj -0.9442
   clustering - none: n=  399 win    3 lose    8 delta -0.0125 p=0.2266  bar-adj -0.1425
   components - none: n=  397 win    2 lose    7 delta -0.0126 p=0.1797  bar-adj -0.0126
       degree - none: n=  393 win  114 lose    5 delta +0.2774 p=0.0000  *sig(BH)  bar-adj -0.7046
       filler - none: n=  400 win   10 lose    5 delta +0.0125 p=0.3018  bar-adj +0.0125
         rwse - none: n=  398 win    1 lose    7 delta -0.0151 p=0.0703  bar-adj -0.0151

TASK: edge_existence
blind bar: 0.900 (p=0.1) down to 0.519 (p=0.5)

mean score by density x condition (hit_cap dropped):
  p               all  clustering  components      degree      filler        none        rwse
  0.1           0.960       0.970       0.990       0.990       0.960       0.960       0.960
  0.2           0.840       0.980       0.940       0.919       0.840       0.930       0.890
  0.35          0.700       0.890       0.750       0.790       0.750       0.830       0.700
  0.5           0.730       0.940       0.830       0.747       0.790       0.900       0.670

shortcut bar: 0.794 (all/degree) / 0.720 (clustering) / 0.642 (rwse) / 0.498
(components/filler/none)

pooled across densities, paired vs 'none' (exact McNemar):
          all - none: n=  399 win   15 lose   54 delta -0.0977 p=0.0000  *sig(BH)  bar-adj -0.3937
   clustering - none: n=  400 win   28 lose   12 delta +0.0400 p=0.0166  *sig(BH)  bar-adj -0.1820
   components - none: n=  400 win   16 lose   27 delta -0.0275 p=0.1263  bar-adj -0.0275
       degree - none: n=  398 win   15 lose   32 delta -0.0427 p=0.0186  *sig(BH)  bar-adj -0.3387
       filler - none: n=  400 win   14 lose   42 delta -0.0700 p=0.0002  *sig(BH)  bar-adj -0.0700
         rwse - none: n=  400 win   13 lose   53 delta -0.1000 p=0.0000  *sig(BH)  bar-adj -0.2440

TASK: node_count
blind bar: 1.000 at every density

mean score by density x condition (hit_cap dropped):
  p               all  clustering  components      degree      filler        none        rwse
  0.1           1.000       0.970       0.970       0.940       0.980       0.920       0.980
  0.2           1.000       0.990       1.000       0.990       1.000       0.690       1.000
  0.35          1.000       1.000       1.000       1.000       1.000       1.000       1.000
  0.5           1.000       1.000       1.000       1.000       1.000       1.000       1.000

shortcut bar: 1.000 (all/clustering/degree/filler/rwse) / 0.076 (components)
/ 0.064 (none)

pooled across densities, paired vs 'none' (exact McNemar):
          all - none: n=  400 win   39 lose    0 delta +0.0975 p=0.0000  *sig(BH)  bar-adj -0.8385
   clustering - none: n=  400 win   38 lose    3 delta +0.0875 p=0.0000  *sig(BH)  bar-adj -0.8485
   components - none: n=  400 win   37 lose    1 delta +0.0900 p=0.0000  *sig(BH)  bar-adj +0.0780
       degree - none: n=  400 win   37 lose    5 delta +0.0800 p=0.0000  *sig(BH)  bar-adj -0.8560
       filler - none: n=  400 win   37 lose    0 delta +0.0925 p=0.0000  *sig(BH)  bar-adj -0.8435
         rwse - none: n=  400 win   39 lose    2 delta +0.0925 p=0.0000  *sig(BH)  bar-adj -0.8435

TASK: node_degree
blind bar: 0.170-0.210 (real headroom)

mean score by density x condition (hit_cap dropped):
  p               all  clustering  components      degree      filler        none        rwse
  0.1           0.880       1.000       1.000       0.940       1.000       1.000       1.000
  0.2           0.960       1.000       0.990       0.980       0.990       0.990       0.970
  0.35          0.920       0.990       0.990       0.920       0.990       0.990       0.960
  0.5           0.840       0.960       1.000       0.870       0.970       0.990       0.950

shortcut bar: 1.000 (all/degree) / 0.616 (rwse) / 0.082 (clustering/
components/filler/none)

pooled across densities, paired vs 'none' (exact McNemar):
          all - none: n=  400 win    2 lose   39 delta -0.0925 p=0.0000  *sig(BH)  bar-adj -1.0105
   clustering - none: n=  400 win    3 lose    5 delta -0.0050 p=0.7266  bar-adj -0.0050
   components - none: n=  400 win    1 lose    0 delta +0.0025 p=1.0000  bar-adj +0.0025
       degree - none: n=  400 win    3 lose   29 delta -0.0650 p=0.0000  *sig(BH)  bar-adj -0.9830
       filler - none: n=  400 win    3 lose    5 delta -0.0050 p=0.7266  bar-adj -0.0050
         rwse - none: n=  400 win    2 lose   11 delta -0.0225 p=0.0225  *sig(BH)  bar-adj -0.5565
```

## `qwen3-4b-think`, full per-task tables

```
TASK: connected_nodes -- near ceiling throughout (0.99-1.00 F1); same
degenerate-McNemar caveat as the plain arm above.
pooled across densities: all +0.0000 p=1.0 / clustering +0.0000 p=1.0 /
   components +0.0000 p=1.0 / degree +0.0000 p=1.0 / filler +0.0000 p=1.0 /
   rwse +0.0000 p=1.0  -- zero discordant pairs on the F1-booleanized test
   everywhere; see the cross-check section for the exact-match numbers.

TASK: cycle_check -- at ceiling (1.000) at nearly every density x condition
cell; `degree` is the one exception, dipping to 0.979-0.990 (n=375, delta
-0.0080 p=0.25, not significant). `components`/`degree` shed 1-8 rows/cell to
hit_cap at p>=0.35 -- survivorship, not evidence, at these small counts.

TASK: edge_count -- mostly not data. capped rows dropped by density x
condition (out of 100/cell):
  p               all  clustering  components      degree      filler        none        rwse
  0.1              50          20          27          64          40          60          31
  0.2              47          47          34          52          50          66          48
  0.35             63          51          65          59          77          92          91
  0.5              63          96          92          60          99         100          99
`none` is 100% capped at p=0.5 (mean score `nan`); the pooled test's n drops
to 34-60 pairs per condition as a result. `all` (-41.5 pp, p=0.0000) and
`degree` (-23.5 pp, p=0.0078) look like real harms but both bar-adjust to
below -1.2 (shortcut bar for `edge_count`/`all`/`degree` is already 1.000) --
and both rest on well under 60 surviving pairs. Treat every `edge_count`-think
number here as thin, same house rule as the `qwen3-1.7b` arm.

TASK: edge_existence -- at ceiling (1.000) at every density x condition cell;
0-4 rows/cell dropped to hit_cap. Every condition ties `none` exactly
(0 discordant pairs, p=1.0 for five of six conditions).

TASK: node_count -- at or near ceiling (0.99-1.00) everywhere; every
condition ties `none` within 2 discordant pairs (best case p=0.5, none
survives BH).

TASK: node_degree -- near ceiling (0.94-1.00) at every density x condition
cell; only `rwse` clears nominal significance (n=381, delta -0.0210,
p=0.0078, *sig(BH), bar-adj -0.5550 -- `rwse`'s own shortcut bar on this task
is 0.616, so more than half of the raw effect is shortcut-explainable even
before the harm direction is accounted for).
```

## Summary: accuracy vs `none`, pooled over densities

Exact match for every task, including `connected_nodes`: its F1 sits at 96-100% for every
condition and hides the differences exact match shows. `hit_cap` rows dropped. `none` column is
accuracy (%); the others are pp vs `none`. **Bold** = significant, from a paired exact McNemar
test on pairs where neither row capped, BH-corrected across the 6 primers within the task.
Re-verified 2026-09-16 from `runs/*.densfull40.*`.

| qwen3-1.7b | cap | none | components | clustering | rwse | degree | filler | all |
|---|---|---|---|---|---|---|---|---|
| connected_nodes | 0% | 72.5 | +0.8 | +1.2 | +1.0 | +3.5 | **−6.2** | +2.2 |
| cycle_check | 6% | 99.7 | −0.5 | +0.3 | **−7.6** | **−11.2** | +0.3 | −1.0 |
| edge_count | 22% | 1.8 | −1.4 | −0.9 | −1.2 | +2.9 | −0.8 | +1.0 |
| edge_existence | 0% | 70.5 | **−8.2** | +3.8 | +5.2 | +4.0 | **−15.8** | **+11.5** |
| node_count | 0% | 31.5 | **+21.2** | **+67.2** | +2.0 | **+6.8** | **+39.8** | **+47.0** |
| node_degree | 0% | 60.5 | +1.8 | +3.0 | +4.2 | **+7.5** | −0.5 | **+10.2** |

| qwen3-1.7b-think | cap | none | components | clustering | rwse | degree | filler | all |
|---|---|---|---|---|---|---|---|---|
| connected_nodes | 1% | 79.7 | +2.3 | +1.2 | −0.3 | **+6.5** | +1.9 | **+5.9** |
| cycle_check | 7% | 100.0 | +0.0 | +0.0 | +0.0 | −0.7 | +0.0 | +0.0 |
| edge_count | 81% | 86.3 | −1.8 | −3.4 | +8.0 | **−20.5** | −2.5 | −18.3 |
| edge_existence | 0% | 99.2 | −0.8 | −1.0 | −1.3 | −2.3 | −0.7 | −2.0 |
| node_count | 3% | 98.6 | +1.1 | +1.4 | +1.4 | +0.6 | +1.4 | +1.4 |
| node_degree | 1% | 76.8 | −1.2 | +1.1 | +2.5 | **+12.9** | −3.4 | **+11.9** |

| qwen3-4b | cap | none | components | clustering | rwse | degree | filler | all |
|---|---|---|---|---|---|---|---|---|
| connected_nodes | 0% | 86.0 | **+9.2** | +1.0 | +0.0 | −3.0 | **−12.5** | −2.8 |
| cycle_check | 1% | 99.7 | −1.3 | +0.3 | +0.3 | −1.5 | −0.0 | −0.5 |
| edge_count | 1% | 2.0 | −1.2 | −1.2 | −1.5 | **+27.8** | +1.2 | **+3.8** |
| edge_existence | 0% | 90.5 | −2.8 | **+4.0** | **−10.0** | **−4.3** | **−7.0** | **−9.8** |
| node_count | 0% | 90.2 | **+9.0** | **+8.8** | **+9.2** | **+8.0** | **+9.2** | **+9.8** |
| node_degree | 0% | 99.2 | +0.2 | −0.5 | **−2.2** | **−6.5** | −0.5 | **−9.2** |

| qwen3-4b-think | cap | none | components | clustering | rwse | degree | filler | all |
|---|---|---|---|---|---|---|---|---|
| connected_nodes | 3% | 98.7 | +0.8 | +0.5 | +0.3 | +0.3 | +0.8 | −0.3 |
| cycle_check | 2% | 100.0 | +0.0 | +0.0 | +0.0 | −0.8 | +0.0 | +0.0 |
| edge_count | 62% | 97.6 | +0.8 | −1.3 | −5.2 | **−28.5** | −1.3 | **−48.4** |
| edge_existence | 1% | 100.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |
| node_count | 0% | 99.5 | +0.0 | +0.5 | +0.5 | +0.5 | +0.5 | +0.5 |
| node_degree | 2% | 99.7 | −0.3 | −0.2 | **−1.8** | +0.0 | +0.3 | −0.5 |

## Conclusions

**Bottom line: no primer gain replicates across model sizes.** Several (task, primer) cells
beat both `none` and `filler` significantly in one plain arm, but none does so in both. The
closest, `degree`/`all` on `node_degree`, helps both 1.7B arms and hurts 4B. Three patterns
are robust: `degree` helps only where the primer states the answer, and only partly; extra
text of any kind moves the tasks whose answer is trivial, regardless of content; thinking
removes almost all headroom.

**Two rules for reading this run:**
- **`shortcuts.json` bars were fit on GraphQA's small ER graphs, not n=40.** Here always
  answering "40" scores 100% on `node_count` and always answering "Yes" scores 100% on
  `cycle_check`. Only bars backed by exact theorems carry over reliably: `degree`/`all` state
  the `node_degree` answer, and summing the stated degrees gives `edge_count`. Ignore the
  `bar-adj` column in the per-task logs above everywhere else.
- **`filler` is not neutral.** It significantly hurts 4B on `connected_nodes` (−12.5) and
  `edge_existence` (−7.0), and 1.7B on `edge_existence` (−15.8)
  and `connected_nodes` (−6.2). At n=40 it adds ~1,830 characters: close to `clustering`
  (1,631), twice `degree` (893), far above `components` (39), below `rwse` (2,951) and `all`
  (4,693).

**By task:**
- **`node_count`: no structural effect.** Nearly every wrong answer is "39", an off-by-one
  from the encoding's "nodes 0, …, 39" listing. In 4B every condition, `filler` included,
  fixes it equally (+8 to +10). In 1.7B the gains follow no content logic: `clustering` +67,
  `filler` +40, `components` +21, `degree` +7, `rwse` +2 (n.s.). The earlier claim that
  "`components` is a real gain that replicates" is retracted.
- **`node_degree`: `degree` is the only primer with a consistent content effect, and its sign
  depends on model size.** It helps the small model, far short of the 100% the primer makes
  available: +7.5 (1.7B), +12.9 (1.7B-think). It hurts 4B: `degree` −6.5 and `all` −9.2, worst
  at p=0.1 and p=0.5 (at p=0.5: `none` 99%, `degree` 87%, `all` 84%). 26 of 4B's 29 `degree`
  misses are off by 1-3. `clustering` → `node_degree`, the earlier headline, is not
  significant in any arm (+3.0 / +1.1 / −0.5 / −0.2).
- **`edge_count`: only `degree` moves plain arms**, where summing the stated degrees gives the
  answer: +27.8 for 4B, which still fails ~70% of the time. Under thinking the task is
  unreadable. 62-81% of rows cap, only easy graphs finish (`none` scores 86-98% on the ones
  that do), and the large `degree`/`all` drops rest on 13-41 pairs.
- **`cycle_check` (gold is always "Yes"): only 1.7B is hurt, by answering "No".** `degree`
  −11.2 holds under both ways of handling capped rows. `rwse` −7.6 does not: it vanishes when
  capped rows count as wrong. 4B scores 98-100% under every condition with no significant
  difference; its earlier `filler` −19.3 / `rwse` −6.6 "collapses" were an extraction bug
  (see Caveats).
- **`edge_existence`: the effects depend on model size.** 1.7B: `all` +11.5, `components`
  −8.2, `filler` −15.8. 4B: `rwse` −10.0, `all` −9.8, `filler` −7.0, `degree` −4.3,
  `clustering` +4.0. Only `filler` (harm) has the same sign and significance in both sizes;
  `all` flips sign.
- **`connected_nodes`:** F1 is at ceiling everywhere. On exact match: 4B `components` +9.2
  and `filler` −12.5; 1.7B-think `degree` +6.5 and `all` +5.9.

**Thinking** takes `cycle_check`, `edge_existence`, `node_count` and (for 4B)
`connected_nodes` to ceiling for every condition. The primers also change how often the
thinking arm truncates, compared with `none`, across all tasks (paired McNemar):
- 1.7B-think: `degree` truncates more (+3.4 pp, p<0.001). `all` (−3.4), `filler` (−1.8) and
  `clustering` (−1.3) truncate less.
- 4B-think: every primer truncates less, by 2.0-4.3 pp (all p<0.001).

## Caveats

- **Boolean extraction was fixed on 2026-09-16, and every number here uses the fix.**
  `scoring._extract_boolean` takes the last yes/no token. 4B plain answers "Yes, there is a
  cycle" and then defines a cycle as a path "with no repeated edges or nodes", so that "no"
  was scored as the answer. The extractor now ignores "no repeated" (`_NO_REPEATED`,
  regression test in `tests/test_scoring.py`). Rescoring changed only 4B `cycle_check` (225
  rows, all wrong→right); the other three arms are unchanged, and
  `superseded/csv2/sweep-large-graph/qwen3-4b.densfull40.rows.csv` was regenerated with the fix.
- **Capped rows are dropped.** On `cycle_check` and `edge_count` that choice changes the
  verdict. `check_significance.py` scores capped rows as 0. On 1.7B `cycle_check` that turns
  `rwse` null and makes `clustering` (+8.5) and `all` (+7.0) significant, because `none` caps
  more often than either (9% vs <1%). That explains the two pipelines' disagreement on this
  task; neither one is simply more sensitive.
- The McNemar test in `score_full_density_sweep.py` binarizes `connected_nodes` F1 as "> 0".
  That is almost always true, so the test is degenerate on that task; use exact match, as
  above.
- The accuracy deltas are unpaired differences of means. The significance marks come from the
  paired test, so the two can differ by a few tenths.
- `csv2/sweep-large-graph/*-think.densfull40.rows.csv` are 90-105 MB; consider Git LFS.
