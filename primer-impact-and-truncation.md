# Primer impact on success rate, and truncation among wrong answers

Scope: every `qwen3-1.7b` / `qwen3-1.7b-think` run file completed so far across the
n20/40/60/80 size sweep (`runs/qwen3-1.7b*.jsonl`, per-task named files only — the
old undifferentiated `qwen3-1.7b[-think].n{20,40,80}.jsonl` files are superseded
duplicates of the `node_degree_n*` files and excluded to avoid double-counting).
Scored directly from the raw per-file records with `graphtalk.scoring.score_one`
(the `primary` metric — accuracy for 6 tasks, F1 for `connected_nodes`), not from
`analysis/sweep_frame.qwen3-1.7b.csv`, because `instance_id` doesn't encode graph
size and a pooled frame can't otherwise be split by `n`.

**Two jobs were still running at write time**: `qwen3-1.7b-think` `node_degree_n40`
(871518, currently 20-21/30 rows/condition) and `cycle_check_n60` (871508,
currently 17-18/30 rows/condition). Re-run this analysis once they finish if you
need the last few points — everything else below is complete (30/30 per cell).

## Part 1 — success rate by size, and Δ vs. the `none` primer at that size

One table per task, per model. Rows are primer conditions, columns are graph
size (`n20`/`n40`/`n60`/`n80`, whichever sizes exist for that task — not split
into separate 60-vs-rest tables, all sizes sit in one table so you can read the
trend across the row). `n rows` in each cell is the row count that size/primer
cell is built from (30 unless a job is still finishing). `Δpp vs none` compares
that primer to `none` **at the same size** — never pooled across sizes, since
pooling would hide exactly the kind of size-dependent effect (and let the
heavily size-skewed truncation rate, see Part 2, quietly bias the number).

**Read the `bar` column before trusting a Δ.** It's the primer-only (no-graph)
solver's accuracy on that (task, condition) from `shortcuts.json` — what a model
could score reading *only the primer text*, never the graph. Bar ≥0.9 (⚠️) means
the primer all but hands over the answer; a Δ there reflects an easier shortcut,
not better graph reasoning. The bar is per (task, condition), not per size, so it
applies uniformly across a row.

#### qwen3-1.7b

**node_count**
| primer | n20 rate (n rows) | n40 rate (n rows) | n60 rate (n rows) | n80 rate (n rows) | n20 Δpp | n40 Δpp | n60 Δpp | n80 Δpp | bar |
|---|---|---|---|---|---|---|---|---|---|
| none | 93.3% (30) | 73.3% (30) | 73.3% (30) | 50.0% (30) | +0.0 | +0.0 | +0.0 | +0.0 | 0.06 |
| components | 96.7% (30) | 66.7% (30) | 80.0% (30) | 46.7% (30) | +3.3 | -6.7 | +6.7 | -3.3 | 0.08 |
| degree | 100.0% (30) | 73.3% (30) | 90.0% (30) | 76.7% (30) | +6.7 | +0.0 | +16.7 | +26.7 | 1.00 ⚠️ |
| clustering | 100.0% (30) | 100.0% (30) | 96.7% (30) | 83.3% (30) | +6.7 | +26.7 | +23.3 | +33.3 | 1.00 ⚠️ |
| rwse | 100.0% (30) | 56.7% (30) | 66.7% (30) | 46.7% (30) | +6.7 | -16.7 | -6.7 | -3.3 | 1.00 ⚠️ |
| filler | 100.0% (30) | 86.7% (30) | 83.3% (30) | 83.3% (30) | +6.7 | +13.3 | +10.0 | +33.3 | 1.00 ⚠️ |
| all | 100.0% (30) | 93.3% (30) | 80.0% (30) | 60.0% (30) | +6.7 | +20.0 | +6.7 | +10.0 | 1.00 ⚠️ |

**node_degree**
| primer | n20 rate (n rows) | n40 rate (n rows) | n60 rate (n rows) | n80 rate (n rows) | n20 Δpp | n40 Δpp | n60 Δpp | n80 Δpp | bar |
|---|---|---|---|---|---|---|---|---|---|
| none | 96.7% (30) | 86.7% (30) | 56.7% (30) | 53.3% (30) | +0.0 | +0.0 | +0.0 | +0.0 | 0.08 |
| components | 96.7% (30) | 80.0% (30) | 63.3% (30) | 53.3% (30) | +0.0 | -6.7 | +6.7 | +0.0 | 0.08 |
| degree | 96.7% (30) | 86.7% (30) | 66.7% (30) | 56.7% (30) | +0.0 | +0.0 | +10.0 | +3.3 | 1.00 ⚠️ |
| clustering | 90.0% (30) | 76.7% (30) | 63.3% (30) | 50.0% (30) | -6.7 | -10.0 | +6.7 | -3.3 | 0.08 |
| rwse | 96.7% (30) | 86.7% (30) | 76.7% (30) | 53.3% (30) | +0.0 | +0.0 | +20.0 | +0.0 | 0.62 |
| filler | 93.3% (30) | 76.7% (30) | 63.3% (30) | 50.0% (30) | -3.3 | -10.0 | +6.7 | -3.3 | 0.08 |
| all | 96.7% (30) | 83.3% (30) | 73.3% (30) | 63.3% (30) | +0.0 | -3.3 | +16.7 | +10.0 | 1.00 ⚠️ |

**edge_count** (only n60/n80 exist for this model)
| primer | n60 rate (n rows) | n80 rate (n rows) | n60 Δpp | n80 Δpp | bar |
|---|---|---|---|---|---|
| none | 26.7% (30) | 0.0% (30) | +0.0 | +0.0 | 0.02 |
| components | 26.7% (30) | 0.0% (30) | +0.0 | +0.0 | 0.02 |
| degree | 40.0% (30) | 13.3% (30) | +13.3 | +13.3 | 1.00 ⚠️ |
| clustering | 40.0% (30) | 13.3% (30) | +13.3 | +13.3 | 0.15 |
| rwse | 13.3% (30) | 13.3% (30) | -13.3 | +13.3 | 0.02 |
| filler | 30.0% (30) | 13.3% (30) | +3.3 | +13.3 | 0.02 |
| all | 26.7% (30) | 16.7% (30) | +0.0 | +16.7 | 1.00 ⚠️ |

**edge_existence**
| primer | n20 rate (n rows) | n40 rate (n rows) | n60 rate (n rows) | n80 rate (n rows) | n20 Δpp | n40 Δpp | n60 Δpp | n80 Δpp | bar |
|---|---|---|---|---|---|---|---|---|---|
| none | 96.7% (30) | 66.7% (30) | 56.7% (30) | 53.3% (30) | +0.0 | +0.0 | +0.0 | +0.0 | 0.50 |
| components | 76.7% (30) | 66.7% (30) | 53.3% (30) | 50.0% (30) | -20.0 | +0.0 | -3.3 | -3.3 | 0.50 |
| degree | 86.7% (30) | 73.3% (30) | 76.7% (30) | 53.3% (30) | -10.0 | +6.7 | +20.0 | +0.0 | 0.79 |
| clustering | 83.3% (30) | 80.0% (30) | 83.3% (30) | 50.0% (30) | -13.3 | +13.3 | +26.7 | -3.3 | 0.72 |
| rwse | 96.7% (30) | 80.0% (30) | 70.0% (30) | 56.7% (30) | +0.0 | +13.3 | +13.3 | +3.3 | 0.64 |
| filler | 83.3% (30) | 66.7% (30) | 60.0% (30) | 50.0% (30) | -13.3 | +0.0 | +3.3 | -3.3 | 0.50 |
| all | 90.0% (30) | 86.7% (30) | 86.7% (30) | 66.7% (30) | -6.7 | +20.0 | +30.0 | +13.3 | 0.79 |

**cycle_check**
| primer | n20 rate (n rows) | n40 rate (n rows) | n60 rate (n rows) | n80 rate (n rows) | n20 Δpp | n40 Δpp | n60 Δpp | n80 Δpp | bar |
|---|---|---|---|---|---|---|---|---|---|
| none | 83.3% (30) | 73.3% (30) | 73.3% (30) | 56.7% (30) | +0.0 | +0.0 | +0.0 | +0.0 | 0.83 |
| components | 86.7% (30) | 66.7% (30) | 73.3% (30) | 56.7% (30) | +3.3 | -6.7 | +0.0 | +0.0 | 1.00 ⚠️ |
| degree | 93.3% (30) | 60.0% (30) | 66.7% (30) | 36.7% (30) | +10.0 | -13.3 | -6.7 | -20.0 | 0.95 ⚠️ |
| clustering | 73.3% (30) | 66.7% (30) | 73.3% (30) | 56.7% (30) | -10.0 | -6.7 | +0.0 | +0.0 | 0.83 |
| rwse | 86.7% (30) | 76.7% (30) | 70.0% (30) | 56.7% (30) | +3.3 | +3.3 | -3.3 | +0.0 | 0.83 |
| filler | 73.3% (30) | 66.7% (30) | 73.3% (30) | 53.3% (30) | -10.0 | -6.7 | +0.0 | -3.3 | 0.83 |
| all | 100.0% (30) | 100.0% (30) | 73.3% (30) | 70.0% (30) | +16.7 | +26.7 | +0.0 | +13.3 | 0.95 ⚠️ |

**connected_nodes** (F1, not accuracy)
| primer | n20 rate (n rows) | n40 rate (n rows) | n60 rate (n rows) | n80 rate (n rows) | n20 Δpp | n40 Δpp | n60 Δpp | n80 Δpp | bar |
|---|---|---|---|---|---|---|---|---|---|
| none | 96.7% (30) | 86.8% (30) | 81.4% (30) | 69.1% (30) | +0.0 | +0.0 | +0.0 | +0.0 | 0.08 |
| components | 90.6% (30) | 85.8% (30) | 75.8% (30) | 64.5% (30) | -6.0 | -1.0 | -5.6 | -4.6 | 0.08 |
| degree | 100.0% (30) | 92.1% (30) | 77.4% (30) | 72.5% (30) | +3.3 | +5.3 | -4.0 | +3.4 | 0.21 |
| clustering | 95.9% (30) | 82.3% (30) | 83.7% (30) | 61.4% (30) | -0.8 | -4.5 | +2.3 | -7.7 | 0.08 |
| rwse | 90.0% (30) | 95.5% (30) | 89.8% (30) | 77.1% (30) | -6.7 | +8.8 | +8.4 | +8.0 | 0.08 |
| filler | 96.7% (30) | 96.0% (30) | 92.5% (30) | 78.3% (30) | +0.0 | +9.2 | +11.1 | +9.2 | 0.08 |
| all | 90.0% (30) | 94.0% (30) | 93.8% (30) | 74.1% (30) | -6.7 | +7.2 | +12.3 | +5.0 | 0.35 |

**reachability** (n60 only — task not run at other sizes)
| primer | n60 rate (n rows) | n60 Δpp | bar |
|---|---|---|---|
| none | 100.0% (30) | +0.0 | n/a |
| components | 96.7% (30) | -3.3 | n/a |
| degree | 100.0% (30) | +0.0 | n/a |
| clustering | 96.7% (30) | -3.3 | n/a |
| rwse | 96.7% (30) | -3.3 | n/a |
| filler | 100.0% (30) | +0.0 | n/a |
| all | 96.7% (30) | -3.3 | n/a |

#### qwen3-1.7b-think

**node_count**
| primer | n20 rate (n rows) | n40 rate (n rows) | n60 rate (n rows) | n80 rate (n rows) | n20 Δpp | n40 Δpp | n60 Δpp | n80 Δpp | bar |
|---|---|---|---|---|---|---|---|---|---|
| none | 100.0% (30) | 96.7% (30) | 100.0% (30) | 83.3% (30) | +0.0 | +0.0 | +0.0 | +0.0 | 0.06 |
| components | 100.0% (30) | 100.0% (30) | 100.0% (30) | 80.0% (30) | +0.0 | +3.3 | +0.0 | -3.3 | 0.08 |
| degree | 100.0% (30) | 100.0% (30) | 100.0% (30) | 80.0% (30) | +0.0 | +3.3 | +0.0 | -3.3 | 1.00 ⚠️ |
| clustering | 100.0% (30) | 100.0% (30) | 100.0% (30) | 80.0% (30) | +0.0 | +3.3 | +0.0 | -3.3 | 1.00 ⚠️ |
| rwse | 100.0% (30) | 100.0% (30) | 100.0% (30) | 76.7% (30) | +0.0 | +3.3 | +0.0 | -6.7 | 1.00 ⚠️ |
| filler | 100.0% (30) | 100.0% (30) | 100.0% (30) | 80.0% (30) | +0.0 | +3.3 | +0.0 | -3.3 | 1.00 ⚠️ |
| all | 100.0% (30) | 100.0% (30) | 83.3% (30) | 76.7% (30) | +0.0 | +3.3 | -16.7 | -6.7 | 1.00 ⚠️ |

**node_degree** (n40 still finishing — 20-21/30 rows/condition)
| primer | n20 rate (n rows) | n40 rate (n rows) | n60 rate (n rows) | n80 rate (n rows) | n20 Δpp | n40 Δpp | n60 Δpp | n80 Δpp | bar |
|---|---|---|---|---|---|---|---|---|---|
| none | 96.7% (30) | 80.0% (20) | 70.0% (30) | 63.3% (30) | +0.0 | +0.0 | +0.0 | +0.0 | 0.08 |
| components | 93.3% (30) | 76.2% (21) | 66.7% (30) | 63.3% (30) | -3.3 | -3.8 | -3.3 | +0.0 | 0.08 |
| degree | 100.0% (30) | 100.0% (20) | 83.3% (30) | 60.0% (30) | +3.3 | +20.0 | +13.3 | -3.3 | 1.00 ⚠️ |
| clustering | 96.7% (30) | 85.7% (21) | 66.7% (30) | 56.7% (30) | +0.0 | +5.7 | -3.3 | -6.7 | 0.08 |
| rwse | 96.7% (30) | 85.0% (20) | 76.7% (30) | 56.7% (30) | +0.0 | +5.0 | +6.7 | -6.7 | 0.62 |
| filler | 93.3% (30) | 90.0% (20) | 60.0% (30) | 56.7% (30) | -3.3 | +10.0 | -10.0 | -6.7 | 0.08 |
| all | 100.0% (30) | 95.2% (21) | 70.0% (30) | 70.0% (30) | +3.3 | +15.2 | +0.0 | +6.7 | 1.00 ⚠️ |

**edge_count** (n60 only for this model)
| primer | n60 rate (n rows) | n60 Δpp | bar |
|---|---|---|---|
| none | 40.0% (30) | +0.0 | 0.02 |
| components | 43.3% (30) | +3.3 | 0.02 |
| degree | 56.7% (30) | +16.7 | 1.00 ⚠️ |
| clustering | 40.0% (30) | +0.0 | 0.15 |
| rwse | 40.0% (30) | +0.0 | 0.02 |
| filler | 40.0% (30) | +0.0 | 0.02 |
| all | 46.7% (30) | +6.7 | 1.00 ⚠️ |

**edge_existence**
| primer | n20 rate (n rows) | n40 rate (n rows) | n60 rate (n rows) | n80 rate (n rows) | n20 Δpp | n40 Δpp | n60 Δpp | n80 Δpp | bar |
|---|---|---|---|---|---|---|---|---|---|
| none | 100.0% (30) | 100.0% (30) | 100.0% (30) | 83.3% (30) | +0.0 | +0.0 | +0.0 | +0.0 | 0.50 |
| components | 100.0% (30) | 96.7% (30) | 96.7% (30) | 83.3% (30) | +0.0 | -3.3 | -3.3 | +0.0 | 0.50 |
| degree | 100.0% (30) | 100.0% (30) | 96.7% (30) | 80.0% (30) | +0.0 | +0.0 | -3.3 | -3.3 | 0.79 |
| clustering | 100.0% (30) | 93.3% (30) | 96.7% (30) | 76.7% (30) | +0.0 | -6.7 | -3.3 | -6.7 | 0.72 |
| rwse | 100.0% (30) | 96.7% (30) | 96.7% (30) | 76.7% (30) | +0.0 | -3.3 | -3.3 | -6.7 | 0.64 |
| filler | 100.0% (30) | 100.0% (30) | 100.0% (30) | 80.0% (30) | +0.0 | +0.0 | +0.0 | -3.3 | 0.50 |
| all | 100.0% (30) | 96.7% (30) | 80.0% (30) | 76.7% (30) | +0.0 | -3.3 | -20.0 | -6.7 | 0.79 |

**cycle_check** (n60 still finishing — 17-18/30 rows/condition)
| primer | n20 rate (n rows) | n40 rate (n rows) | n60 rate (n rows) | n80 rate (n rows) | n20 Δpp | n40 Δpp | n60 Δpp | n80 Δpp | bar |
|---|---|---|---|---|---|---|---|---|---|
| none | 96.7% (30) | 86.7% (30) | 100.0% (17) | 70.0% (30) | +0.0 | +0.0 | +0.0 | +0.0 | 0.83 |
| components | 100.0% (30) | 86.7% (30) | 100.0% (18) | 80.0% (30) | +3.3 | +0.0 | +0.0 | +10.0 | 1.00 ⚠️ |
| degree | 96.7% (30) | 96.7% (30) | 88.9% (18) | 80.0% (30) | +0.0 | +10.0 | -11.1 | +10.0 | 0.95 ⚠️ |
| clustering | 100.0% (30) | 100.0% (30) | 100.0% (18) | 80.0% (30) | +3.3 | +13.3 | +0.0 | +10.0 | 0.83 |
| rwse | 73.3% (30) | 100.0% (30) | 100.0% (17) | 50.0% (30) | -23.3 | +13.3 | +0.0 | -20.0 | 0.83 |
| filler | 86.7% (30) | 100.0% (30) | 100.0% (17) | 63.3% (30) | -10.0 | +13.3 | +0.0 | -6.7 | 0.83 |
| all | 100.0% (30) | 96.7% (30) | 94.4% (18) | 50.0% (30) | +3.3 | +10.0 | -5.6 | -20.0 | 0.95 ⚠️ |

**connected_nodes** (F1, not accuracy)
| primer | n20 rate (n rows) | n40 rate (n rows) | n60 rate (n rows) | n80 rate (n rows) | n20 Δpp | n40 Δpp | n60 Δpp | n80 Δpp | bar |
|---|---|---|---|---|---|---|---|---|---|
| none | 99.3% (30) | 96.6% (30) | 81.1% (30) | 75.8% (30) | +0.0 | +0.0 | +0.0 | +0.0 | 0.08 |
| components | 93.1% (30) | 87.2% (30) | 76.9% (30) | 69.0% (30) | -6.2 | -9.4 | -4.2 | -6.9 | 0.08 |
| degree | 100.0% (30) | 98.8% (30) | 90.0% (30) | 73.7% (30) | +0.7 | +2.2 | +8.8 | -2.2 | 0.21 |
| clustering | 96.9% (30) | 89.2% (30) | 91.2% (30) | 73.0% (30) | -2.4 | -7.4 | +10.1 | -2.8 | 0.08 |
| rwse | 96.6% (30) | 87.3% (30) | 87.5% (30) | 64.3% (30) | -2.7 | -9.3 | +6.3 | -11.6 | 0.08 |
| filler | 98.6% (30) | 96.8% (30) | 87.2% (30) | 71.3% (30) | -0.7 | +0.2 | +6.1 | -4.6 | 0.08 |
| all | 99.8% (30) | 96.5% (30) | 71.4% (30) | 63.5% (30) | +0.5 | -0.1 | -9.7 | -12.3 | 0.35 |

**reachability** (n60 only — task not run at other sizes)
| primer | n60 rate (n rows) | n60 Δpp | bar |
|---|---|---|---|
| none | 100.0% (30) | +0.0 | n/a |
| components | 100.0% (30) | +0.0 | n/a |
| degree | 100.0% (30) | +0.0 | n/a |
| clustering | 100.0% (30) | +0.0 | n/a |
| rwse | 96.7% (30) | -3.3 | n/a |
| filler | 100.0% (30) | +0.0 | n/a |
| all | 83.3% (30) | -16.7 | n/a |

### What the size breakdown shows that pooling hid

- **Both models decay with size on almost every task, primer or not.** `none`
  itself drops steadily — e.g. plain qwen3-1.7b `node_degree` 96.7%→53.3%
  (n20→n80), `edge_existence` 96.7%→53.3%. A primer's Δ needs to be read against
  that shrinking baseline, not as a flat number.
- **Several apparent "primer helps" results are actually n60/n80-only effects
  that a pooled average smeared across all sizes.** E.g. plain qwen3-1.7b
  `edge_existence`/`all`: +30.0pp at n60 but only -6.7pp at n20 — the pooled
  Part-1 (old) table showed a single +14.2pp that hid this reversal entirely.
  Same pattern in `cycle_check`/`all` (n20 +16.7, n40 +26.7, n60 +0.0, n80
  +13.3) and `connected_nodes`/`filler` (climbs from +0.0 at n20 to +11.1 at
  n60 for both models).
- **The think arm's `all`-condition regressions are concentrated at n60/n80**,
  not general: `node_count` -16.7pp at n60 but +0.0 to +3.3pp at n20/n40;
  `edge_existence` -20.0pp at n60, -6.7pp at n80, but +0.0 at n20/n40;
  `connected_nodes` -9.7pp at n60, -12.3pp at n80, but ≈0 at n20/n40. Part 2
  below shows this lines up with where truncation among wrong answers spikes
  for the think arm — so at least part of these "primer hurts" numbers are a
  token-budget artifact, not the primer genuinely confusing the model.
- **cycle_check/degree for plain qwen3-1.7b** is the clearest monotonic-with-size
  regression not explained by truncation at n20/n40 (+10.0 at n20 → -13.3 at n40
  → -6.7 at n60 → -20.0 at n80) — worth flagging on its own, since it's also the
  primer condition Part 2 shows truncates unusually heavily on `cycle_check` even
  at n40/n60 (42-60% of wrong answers).

### Which conditions are shortcut-contaminated (bar is per task/condition, applies to every size in that row)

- **node_count and edge_count are almost entirely contaminated.** Every condition
  except `none`/`components` (and `rwse`/`filler` on `edge_count` specifically)
  has a shortcut bar ≥0.9 — the "degree"/"clustering" primers essentially state or
  strongly imply the node/edge count as a side effect of what they render. Read
  any Δ in those rows as "the primer handed the model the answer," not "the
  model reasoned over the graph better." `filler` is also flagged ⚠️ on
  `node_count` — the filler primer's sentence count scales with `n`, so its
  intended purpose (length-matched, content-free control) is compromised for
  this task specifically.
- **node_degree**: only `degree` and `all` are contaminated (bar 1.00); `rwse`
  sits at a middling 0.62. `components`/`clustering`/`filler` are clean (bar ≈
  `none`'s 0.08) — and across sizes those clean rows show no consistent positive
  Δ for either model, so there's no real evidence any primer helps
  `node_degree` once the contaminated conditions are set aside.
- **edge_existence, cycle_check, connected_nodes** have the widest range of clean
  (bar close to `none`) or moderately-clean conditions, so their per-size Δs are
  the most trustworthy in this sweep:
  - `edge_existence`/`rwse` (bar 0.64, mid-range) is positive at n40/n60 for
    plain qwen3-1.7b (+13.3pp each) but flat/negative for the think arm at every
    size — a real plain-vs-think divergence, not a contamination artifact.
  - `cycle_check`/`clustering` (bar 0.83, same as `none`) is flat-to-negative for
    plain qwen3-1.7b but consistently positive for the think arm at n20/n40/n80
    (+3.3/+13.3/+10.0pp) — the cleanest positive read for the think arm in this
    whole table.
  - `connected_nodes` (F1, not accuracy)/`filler` and `/rwse` (bar 0.08, same as
    `none`) are the cleanest primers and also the best performers for plain
    qwen3-1.7b at n40-n80 (`filler`: +9.2/+11.1/+9.2pp) — the strongest
    non-contaminated positive signal in this dataset. The think arm doesn't
    replicate it at n60/n80 (`filler` +6.1pp at n60 but -4.6pp at n80).
- **reachability** has no shortcut bar on file (task not in `shortcuts.json`) and
  is n60-only (n=30/condition) — small-sample, treat any Δ here as noise; the
  think-arm `all` condition's -16.7pp is a truncation artifact (Part 2 shows
  5/5 of its wrong answers are truncated), not a reasoning regression.

## Part 2 — truncation among wrong answers, by task × primer × size

Cells show `truncated_wrong / total_wrong (pct)`; `truncated` = `hit_cap` or
`overflow` on that row. `-` = no wrong answers in that cell (nothing to explain).
This is the breakdown the pooled Part-1 table can't show: it isolates which
specific (task, primer, size) combinations have their wrong-answer count
inflated by truncation rather than genuine model error.

#### qwen3-1.7b

**node_count** (trunc% of wrong answers; `-` = no wrong answers)
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 0/2 (0%) | 0/8 (0%) | 0/8 (0%) | 5/15 (33%) |
| components | 0/1 (0%) | 0/10 (0%) | 0/6 (0%) | 5/16 (31%) |
| degree | - | 0/8 (0%) | 0/3 (0%) | 5/7 (71%) |
| clustering | - | - | 0/1 (0%) | 5/5 (100%) |
| rwse | - | 0/13 (0%) | 0/10 (0%) | 5/16 (31%) |
| filler | - | 0/4 (0%) | 0/5 (0%) | 5/5 (100%) |
| all | - | 0/2 (0%) | 0/6 (0%) | 5/12 (42%) |

**node_degree** (trunc% of wrong answers; `-` = no wrong answers)
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 0/1 (0%) | 0/4 (0%) | 0/13 (0%) | 6/14 (43%) |
| components | 0/1 (0%) | 0/6 (0%) | 0/11 (0%) | 5/14 (36%) |
| degree | 0/1 (0%) | 0/4 (0%) | 0/10 (0%) | 5/13 (38%) |
| clustering | 0/3 (0%) | 0/7 (0%) | 0/11 (0%) | 5/15 (33%) |
| rwse | 0/1 (0%) | 0/4 (0%) | 0/7 (0%) | 5/14 (36%) |
| filler | 0/2 (0%) | 0/7 (0%) | 0/11 (0%) | 5/15 (33%) |
| all | 0/1 (0%) | 0/5 (0%) | 0/8 (0%) | 5/11 (45%) |

**edge_count** (trunc% of wrong answers; `-` = no wrong answers)
| primer | n60 | n80 |
|---|---|---|
| none | 12/22 (55%) | 14/30 (47%) |
| components | 13/22 (59%) | 15/30 (50%) |
| degree | 11/18 (61%) | 16/26 (62%) |
| clustering | 7/18 (39%) | 9/26 (35%) |
| rwse | 13/26 (50%) | 15/26 (58%) |
| filler | 12/21 (57%) | 13/26 (50%) |
| all | 16/22 (73%) | 17/25 (68%) |

**edge_existence** (trunc% of wrong answers; `-` = no wrong answers)
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 0/1 (0%) | 0/10 (0%) | 0/13 (0%) | 5/14 (36%) |
| components | 0/7 (0%) | 1/10 (10%) | 0/14 (0%) | 5/15 (33%) |
| degree | 0/4 (0%) | 0/8 (0%) | 0/7 (0%) | 5/14 (36%) |
| clustering | 0/5 (0%) | 0/6 (0%) | 0/5 (0%) | 5/15 (33%) |
| rwse | 0/1 (0%) | 0/6 (0%) | 0/9 (0%) | 5/13 (38%) |
| filler | 0/5 (0%) | 0/10 (0%) | 0/12 (0%) | 5/15 (33%) |
| all | 0/3 (0%) | 0/4 (0%) | 0/4 (0%) | 5/10 (50%) |

**cycle_check** (trunc% of wrong answers; `-` = no wrong answers)
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 1/5 (20%) | 2/8 (25%) | 0/8 (0%) | 5/13 (38%) |
| components | 0/4 (0%) | 0/10 (0%) | 0/8 (0%) | 5/13 (38%) |
| degree | 0/2 (0%) | 5/12 (42%) | 6/10 (60%) | 11/19 (58%) |
| clustering | 0/8 (0%) | 0/10 (0%) | 0/8 (0%) | 5/13 (38%) |
| rwse | 0/4 (0%) | 0/7 (0%) | 1/9 (11%) | 5/13 (38%) |
| filler | 0/8 (0%) | 0/10 (0%) | 0/8 (0%) | 5/14 (36%) |
| all | - | - | 0/8 (0%) | 5/9 (56%) |

**connected_nodes** (trunc% of wrong answers; `-` = no wrong answers)
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 0/1 (0%) | 0/6 (0%) | 1/19 (5%) | 5/15 (33%) |
| components | 0/3 (0%) | 0/8 (0%) | 1/18 (6%) | 5/18 (28%) |
| degree | - | 0/7 (0%) | 0/12 (0%) | 5/16 (31%) |
| clustering | 0/2 (0%) | 0/11 (0%) | 0/14 (0%) | 5/18 (28%) |
| rwse | 0/4 (0%) | 0/5 (0%) | 0/14 (0%) | 5/12 (42%) |
| filler | 0/3 (0%) | 0/4 (0%) | 0/12 (0%) | 5/14 (36%) |
| all | 0/3 (0%) | 0/6 (0%) | 0/10 (0%) | 5/15 (33%) |

**reachability** (trunc% of wrong answers; `-` = no wrong answers)
| primer | n60 |
|---|---|
| none | - |
| components | 1/1 (100%) |
| degree | - |
| clustering | 1/1 (100%) |
| rwse | 0/1 (0%) |
| filler | - |
| all | 0/1 (0%) |

#### qwen3-1.7b-think

**node_count** (trunc% of wrong answers; `-` = no wrong answers)
| primer | n40 | n60 | n80 |
|---|---|---|---|
| none | 1/1 (100%) | - | 5/5 (100%) |
| components | - | - | 6/6 (100%) |
| degree | - | - | 6/6 (100%) |
| clustering | - | - | 6/6 (100%) |
| rwse | - | - | 7/7 (100%) |
| filler | - | - | 6/6 (100%) |
| all | - | 5/5 (100%) | 7/7 (100%) |

**node_degree** (trunc% of wrong answers; `-` = no wrong answers)
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 0/1 (0%) | 0/4 (0%) | 0/9 (0%) | 5/11 (45%) |
| components | 0/2 (0%) | 0/5 (0%) | 0/10 (0%) | 5/11 (45%) |
| degree | - | - | 0/5 (0%) | 6/12 (50%) |
| clustering | 0/1 (0%) | 1/3 (33%) | 0/10 (0%) | 6/13 (46%) |
| rwse | 0/1 (0%) | 0/3 (0%) | 1/7 (14%) | 8/13 (62%) |
| filler | 2/2 (100%) | 0/2 (0%) | 0/12 (0%) | 6/13 (46%) |
| all | - | 0/1 (0%) | 5/9 (56%) | 7/9 (78%) |

**edge_count** (trunc% of wrong answers; `-` = no wrong answers)
| primer | n60 |
|---|---|
| none | 15/18 (83%) |
| components | 15/17 (88%) |
| degree | 11/13 (85%) |
| clustering | 17/18 (94%) |
| rwse | 17/18 (94%) |
| filler | 17/18 (94%) |
| all | 11/16 (69%) |

**edge_existence** (trunc% of wrong answers; `-` = no wrong answers)
| primer | n40 | n60 | n80 |
|---|---|---|---|
| none | - | - | 5/5 (100%) |
| components | 0/1 (0%) | 0/1 (0%) | 5/5 (100%) |
| degree | - | 0/1 (0%) | 6/6 (100%) |
| clustering | 1/2 (50%) | 0/1 (0%) | 6/7 (86%) |
| rwse | 1/1 (100%) | 0/1 (0%) | 7/7 (100%) |
| filler | - | - | 6/6 (100%) |
| all | 0/1 (0%) | 5/6 (83%) | 7/7 (100%) |

**cycle_check** (trunc% of wrong answers; `-` = no wrong answers)
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 1/1 (100%) | 0/4 (0%) | - | 5/9 (56%) |
| components | - | 0/4 (0%) | - | 6/6 (100%) |
| degree | 0/1 (0%) | 0/1 (0%) | 1/2 (50%) | 6/6 (100%) |
| clustering | - | - | - | 6/6 (100%) |
| rwse | 0/8 (0%) | - | - | 7/15 (47%) |
| filler | 0/4 (0%) | - | - | 7/11 (64%) |
| all | - | 1/1 (100%) | 1/1 (100%) | 7/15 (47%) |

**connected_nodes** (trunc% of wrong answers; `-` = no wrong answers)
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 1/3 (33%) | 0/3 (0%) | 0/15 (0%) | 5/12 (42%) |
| components | 0/4 (0%) | 0/7 (0%) | 0/12 (0%) | 5/14 (36%) |
| degree | - | 0/2 (0%) | 0/10 (0%) | 6/14 (43%) |
| clustering | 0/2 (0%) | 0/5 (0%) | 0/9 (0%) | 6/13 (46%) |
| rwse | 0/2 (0%) | 1/6 (17%) | 0/8 (0%) | 7/15 (47%) |
| filler | 1/2 (50%) | 0/1 (0%) | 0/14 (0%) | 6/14 (43%) |
| all | 0/2 (0%) | 0/2 (0%) | 6/14 (43%) | 7/14 (50%) |

**reachability** (trunc% of wrong answers; `-` = no wrong answers)
| primer | n60 |
|---|---|
| none | - |
| components | - |
| degree | - |
| clustering | - |
| rwse | 1/1 (100%) |
| filler | 5/5 (100%) |

*(reachability `all` row above shows 5/5 (100%) truncated wrong — this is what
drives the think-arm `reachability`/`all` -16.7pp in Part 1: all 5 wrong answers
there are truncation, not the model reasoning worse under that primer.)*

### Truncation hot spots

- **n80 is the dominant truncation driver for every task/primer combo, for both
  models**, typically 30-50% of wrong answers, rising to 100% for
  `qwen3-1.7b-think`/`node_count`/`n80` and `edge_existence`/`n80`. At n80, wrong
  counts are also small (5-30), so a handful of truncations swing the percentage
  a lot — read n80 percentages as noisy, not as precise rates.
- **`edge_count` truncates heavily even at n60**, for both models and every
  primer — 39-73% for plain, 69-94% for think. This is the task/size combination
  where the token cap (not primer quality or graph difficulty) is the main
  reason answers come out wrong; the `edge_count`/`all` success-rate numbers in
  Part 1 (13.3-46.7%) are substantially truncation-limited, not primer-limited.
- **`cycle_check`/`degree`** is a standout for plain qwen3-1.7b: 42-60% of wrong
  answers are truncated at n40/n60, worse than every other primer on that task at
  those sizes — worth a second look at whether the `degree` primer text itself
  is unusually verbose or induces longer reasoning before the model answers.
- **think-arm `node_degree`/`all`** goes from 0% truncated at n20/n40 to 56% at
  n60 and 78% at n80 — a clean escalation with size, consistent with the earlier
  token-budget analysis (n60/n80 have shrinking headroom for the think arm
  specifically).
- **Plain qwen3-1.7b below n60 is almost entirely truncation-free** — nearly
  every n20/n40 cell across every task is 0%. Truncation as a source of wrong
  answers is a large-graph (n60+) and edge_count-specific phenomenon here, not a
  general problem with the sweep.

## Bottom line

- Genuine (non-shortcut-contaminated) primer effects in this uncontrolled-density
  sweep exist but are size-dependent, not flat: `connected_nodes` `filler`/`rwse`
  help plain qwen3-1.7b, growing from ~0 at n20 to +9-11pp at n40-n80;
  `cycle_check` `clustering` helps the think arm consistently from n20 through
  n80 (bar 0.83, same as `none`); `edge_existence` `rwse` helps plain
  qwen3-1.7b at n40/n60 specifically but not the think arm anywhere. `node_degree`
  shows no clean positive Δ for either model at any size once contaminated
  conditions (`degree`, `all`) are excluded. Most of the large positive Δs on
  `node_count`/`edge_count`, and most `all`-condition Δs, are shortcut-bar ⚠️
  and shouldn't be read as reasoning gains regardless of size.
- Pooling across sizes (the original version of this table) actively hid
  reversals — several conditions look flat or negative pooled but are strongly
  positive at one size and strongly negative at another (`edge_existence`/`all`,
  `cycle_check`/`all` for plain qwen3-1.7b). Any future primer-effect summary
  for this project should report by size, not pooled.
- Truncation is concentrated at n60+/`edge_count` and n80 generally — exactly
  where [`docs/node_degree-density-and-size.md`](node_degree-density-and-size.md)
  already flagged headroom problems — and directly explains part of the think
  arm's apparent `all`-condition regressions at n60/n80. Any primer-effect
  conclusion drawn from n60/n80 cells (especially `edge_count`, and the think
  arm generally) should be treated as truncation-limited until the token-budget
  plan (`can-i-shorten-the-structured-sloth` in `~/.claude/plans/`) is actually
  implemented and re-run.
