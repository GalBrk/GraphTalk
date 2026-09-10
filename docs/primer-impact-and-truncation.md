# Primer impact on success rate, and truncation among all rows

Scope: every `qwen3-1.7b` / `qwen3-1.7b-think` run file completed so far across the
n20/40/60/80 size sweep (`runs/qwen3-1.7b*.jsonl`, per-task named files only — the
old undifferentiated `qwen3-1.7b[-think].n{20,40,80}.jsonl` files are superseded
duplicates of the `node_degree_n*` files and excluded to avoid double-counting).
Scored directly from the raw per-file records with `graphtalk.scoring.score_one`
(the `primary` metric — accuracy for 6 tasks, F1 for `connected_nodes`), not from
`analysis/sweep_frame.qwen3-1.7b.csv`, because `instance_id` doesn't encode graph
size and a pooled frame can't otherwise be split by `n`.

**Complete.** Every task/model/size cell below has `n=30` per condition — the two
cells that were still in flight in an earlier version of this document
(`qwen3-1.7b-think` `node_degree_n40` and `cycle_check_n60`) have both finished and
are recomputed here.

**Part 2's truncation metric changed from the first draft of this document.** It
previously reported truncation as a percentage *of wrong answers only* (isolating
whether wrong answers are truncation-driven). It now reports truncation as a
percentage *of all rows in the cell* — how much of the sweep's sample at a given
size is being burned by hitting the token cap at all, correct or not. This is the
number that matters for choosing a safe node-count range (the question this
document was written to help answer; see "Recommended node-count range" at the
end), whereas the wrong-answer-only framing only tells you whether an *already
low* accuracy number is partly an artifact.

## Part 1 — success rate by size, and Δ vs. the `none` primer at that size

One table per task, per model. Rows are primer conditions, columns are graph
size (`n20`/`n40`/`n60`/`n80`, whichever sizes exist for that task — not split
into separate 60-vs-rest tables, all sizes sit in one table so you can read the
trend across the row). `n rows` in each cell is 30 throughout. `Δpp vs none`
compares that primer to `none` **at the same size** — never pooled across sizes,
since pooling would hide exactly the kind of size-dependent effect (and let the
heavily size-skewed truncation rate, see Part 2, quietly bias the number).

**Read the `bar` column before trusting a Δ.** It's the primer-only (no-graph)
solver's accuracy on that (task, condition) from `shortcuts.json` — what a model
could score reading *only the primer text*, never the graph. Bar ≥0.9 (⚠️) means
the primer all but hands over the answer; a Δ there reflects an easier shortcut,
not better graph reasoning. The bar is per (task, condition), not per size, so it
applies uniformly across a row. **This is a real, structural limit, not a detail
to skip past when reading Δs below or the range recommendation at the end** — a
⚠️-flagged row's Δ stays contamination-explained at every size in this sweep, no
choice of node-count range fixes it.

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

**node_degree** (now complete — was partial at n40 in the previous draft)
| primer | n20 rate (n rows) | n40 rate (n rows) | n60 rate (n rows) | n80 rate (n rows) | n20 Δpp | n40 Δpp | n60 Δpp | n80 Δpp | bar |
|---|---|---|---|---|---|---|---|---|---|
| none | 96.7% (30) | 86.7% (30) | 70.0% (30) | 63.3% (30) | +0.0 | +0.0 | +0.0 | +0.0 | 0.08 |
| components | 93.3% (30) | 83.3% (30) | 66.7% (30) | 63.3% (30) | -3.3 | -3.3 | -3.3 | +0.0 | 0.08 |
| degree | 100.0% (30) | 100.0% (30) | 83.3% (30) | 60.0% (30) | +3.3 | +13.3 | +13.3 | -3.3 | 1.00 ⚠️ |
| clustering | 96.7% (30) | 90.0% (30) | 66.7% (30) | 56.7% (30) | +0.0 | +3.3 | -3.3 | -6.7 | 0.08 |
| rwse | 96.7% (30) | 90.0% (30) | 76.7% (30) | 56.7% (30) | +0.0 | +3.3 | +6.7 | -6.7 | 0.62 |
| filler | 93.3% (30) | 93.3% (30) | 60.0% (30) | 56.7% (30) | -3.3 | +6.7 | -10.0 | -6.7 | 0.08 |
| all | 100.0% (30) | 96.7% (30) | 70.0% (30) | 70.0% (30) | +3.3 | +10.0 | +0.0 | +6.7 | 1.00 ⚠️ |

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

**cycle_check** (now complete — was partial at n60 in the previous draft, and the
refreshed numbers move a lot: the 17-18-row partial sample had `none`/`components`/
`rwse`/`filler` all reading 100.0%, which the full 30-row sample corrects down to
86.7%/73.3%/73.3%/86.7%. Nothing after this table depends on the stale partial
numbers.)
| primer | n20 rate (n rows) | n40 rate (n rows) | n60 rate (n rows) | n80 rate (n rows) | n20 Δpp | n40 Δpp | n60 Δpp | n80 Δpp | bar |
|---|---|---|---|---|---|---|---|---|---|
| none | 96.7% (30) | 86.7% (30) | 86.7% (30) | 70.0% (30) | +0.0 | +0.0 | +0.0 | +0.0 | 0.83 |
| components | 100.0% (30) | 86.7% (30) | 73.3% (30) | 80.0% (30) | +3.3 | +0.0 | -13.3 | +10.0 | 1.00 ⚠️ |
| degree | 96.7% (30) | 96.7% (30) | 93.3% (30) | 80.0% (30) | +0.0 | +10.0 | +6.7 | +10.0 | 0.95 ⚠️ |
| clustering | 100.0% (30) | 100.0% (30) | 100.0% (30) | 80.0% (30) | +3.3 | +13.3 | +13.3 | +10.0 | 0.83 |
| rwse | 73.3% (30) | 100.0% (30) | 73.3% (30) | 50.0% (30) | -23.3 | +13.3 | -13.3 | -20.0 | 0.83 |
| filler | 86.7% (30) | 100.0% (30) | 86.7% (30) | 63.3% (30) | -10.0 | +13.3 | +0.0 | -6.7 | 0.83 |
| all | 100.0% (30) | 96.7% (30) | 83.3% (30) | 50.0% (30) | +3.3 | +10.0 | -3.3 | -20.0 | 0.95 ⚠️ |

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
  below shows this lines up with where truncation spikes for the think arm —
  so at least part of these "primer hurts" numbers are a token-budget artifact,
  not the primer genuinely confusing the model.
- **cycle_check/degree for plain qwen3-1.7b** is the clearest monotonic-with-size
  regression not explained by truncation at n20/n40 (+10.0 at n20 → -13.3 at n40
  → -6.7 at n60 → -20.0 at n80) — worth flagging on its own, since it's also the
  primer condition Part 2 shows truncates unusually heavily on `cycle_check`,
  starting as early as n40.
- **The refreshed think/`cycle_check` n60 table changes the earlier partial-data
  reading.** With the full 30 rows, `none` reads 86.7% (not the partial sample's
  100.0%), which puts `components` (-13.3pp) and `rwse` (-13.3pp) into a real
  negative Δ at n60 that the partial sample's ceiling effect had hidden.

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
    plain qwen3-1.7b but consistently positive for the think arm at n20/n40/n60/n80
    (+3.3/+13.3/+13.3/+10.0pp, now the full n60 row too) — the cleanest positive
    read for the think arm in this whole table.
  - `connected_nodes` (F1, not accuracy)/`filler` and `/rwse` (bar 0.08, same as
    `none`) are the cleanest primers and also the best performers for plain
    qwen3-1.7b at n40-n80 (`filler`: +9.2/+11.1/+9.2pp) — the strongest
    non-contaminated positive signal in this dataset. The think arm doesn't
    replicate it at n60/n80 (`filler` +6.1pp at n60 but -4.6pp at n80).
- **reachability** has no shortcut bar on file (task not in `shortcuts.json`) and
  is n60-only (n=30/condition) — small-sample, treat any Δ here as noise; the
  think-arm `all` condition's -16.7pp is a truncation artifact (Part 2 shows
  5/30 of its rows are truncated), not a reasoning regression.

## Part 2 — truncation among all rows, by task × primer × size

Cells show `pct% (trunc/n)`, `trunc` = `hit_cap` or `overflow` (`response is
None`) rows, `n` = 30 in every cell. Unlike the previous draft, this is a
fraction of **every row scored in the cell**, not only the wrong ones — it
directly answers "how much of this cell's sample got cut off by the token
budget," which is the number that should drive a node-count choice, independent
of whether the model would have gotten the answer right anyway.

#### qwen3-1.7b

**node_count**
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| components | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| degree | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| clustering | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| rwse | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| filler | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| all | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |

**node_degree**
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 0% (0/30) | 0% (0/30) | 0% (0/30) | 20% (6/30) |
| components | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| degree | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| clustering | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| rwse | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| filler | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| all | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |

**edge_count** (only n60/n80 exist for this model)
| primer | n60 | n80 |
|---|---|---|
| none | 40% (12/30) | 47% (14/30) |
| components | 43% (13/30) | 50% (15/30) |
| degree | 37% (11/30) | 53% (16/30) |
| clustering | 23% (7/30) | 30% (9/30) |
| rwse | 43% (13/30) | 50% (15/30) |
| filler | 40% (12/30) | 43% (13/30) |
| all | 53% (16/30) | 57% (17/30) |

**edge_existence**
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| components | 0% (0/30) | 3% (1/30) | 0% (0/30) | 17% (5/30) |
| degree | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| clustering | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| rwse | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| filler | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| all | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |

**cycle_check**
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 3% (1/30) | 7% (2/30) | 0% (0/30) | 17% (5/30) |
| components | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| degree | 0% (0/30) | 17% (5/30) | 20% (6/30) | 37% (11/30) |
| clustering | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| rwse | 0% (0/30) | 0% (0/30) | 3% (1/30) | 17% (5/30) |
| filler | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| all | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |

**connected_nodes**
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 0% (0/30) | 0% (0/30) | 3% (1/30) | 17% (5/30) |
| components | 0% (0/30) | 0% (0/30) | 3% (1/30) | 17% (5/30) |
| degree | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| clustering | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| rwse | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| filler | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| all | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |

**reachability** (n60 only)
| primer | n60 |
|---|---|
| none | 0% (0/30) |
| components | 3% (1/30) |
| degree | 0% (0/30) |
| clustering | 3% (1/30) |
| rwse | 0% (0/30) |
| filler | 0% (0/30) |
| all | 0% (0/30) |

#### qwen3-1.7b-think

**node_count**
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 0% (0/30) | 3% (1/30) | 0% (0/30) | 17% (5/30) |
| components | 0% (0/30) | 0% (0/30) | 0% (0/30) | 20% (6/30) |
| degree | 0% (0/30) | 0% (0/30) | 0% (0/30) | 20% (6/30) |
| clustering | 0% (0/30) | 0% (0/30) | 0% (0/30) | 20% (6/30) |
| rwse | 0% (0/30) | 0% (0/30) | 0% (0/30) | 23% (7/30) |
| filler | 0% (0/30) | 0% (0/30) | 0% (0/30) | 20% (6/30) |
| all | 0% (0/30) | 0% (0/30) | 17% (5/30) | 23% (7/30) |

**node_degree**
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| components | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| degree | 0% (0/30) | 0% (0/30) | 0% (0/30) | 20% (6/30) |
| clustering | 0% (0/30) | 3% (1/30) | 0% (0/30) | 20% (6/30) |
| rwse | 0% (0/30) | 0% (0/30) | 3% (1/30) | 27% (8/30) |
| filler | 7% (2/30) | 0% (0/30) | 0% (0/30) | 20% (6/30) |
| all | 0% (0/30) | 0% (0/30) | 17% (5/30) | 23% (7/30) |

**edge_count** (n60 only for this model)
| primer | n60 |
|---|---|
| none | 50% (15/30) |
| components | 53% (16/30) |
| degree | 43% (13/30) |
| clustering | 57% (17/30) |
| rwse | 57% (17/30) |
| filler | 57% (17/30) |
| all | 43% (13/30) |

**edge_existence**
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| components | 0% (0/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| degree | 0% (0/30) | 0% (0/30) | 0% (0/30) | 23% (7/30) |
| clustering | 0% (0/30) | 3% (1/30) | 0% (0/30) | 20% (6/30) |
| rwse | 0% (0/30) | 7% (2/30) | 0% (0/30) | 23% (7/30) |
| filler | 0% (0/30) | 0% (0/30) | 0% (0/30) | 20% (6/30) |
| all | 0% (0/30) | 0% (0/30) | 17% (5/30) | 23% (7/30) |

**cycle_check** (now complete at n60 — see the note under Part 1's table)
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 3% (1/30) | 0% (0/30) | 3% (1/30) | 20% (6/30) |
| components | 0% (0/30) | 0% (0/30) | 0% (0/30) | 20% (6/30) |
| degree | 0% (0/30) | 0% (0/30) | 10% (3/30) | 23% (7/30) |
| clustering | 0% (0/30) | 0% (0/30) | 0% (0/30) | 20% (6/30) |
| rwse | 0% (0/30) | 0% (0/30) | 0% (0/30) | 23% (7/30) |
| filler | 0% (0/30) | 13% (4/30) | 0% (0/30) | 23% (7/30) |
| all | 0% (0/30) | 3% (1/30) | 17% (5/30) | 23% (7/30) |

**connected_nodes**
| primer | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 3% (1/30) | 0% (0/30) | 0% (0/30) | 17% (5/30) |
| components | 0% (0/30) | 0% (0/30) | 0% (0/30) | 20% (6/30) |
| degree | 0% (0/30) | 0% (0/30) | 3% (1/30) | 20% (6/30) |
| clustering | 0% (0/30) | 0% (0/30) | 0% (0/30) | 20% (6/30) |
| rwse | 0% (0/30) | 3% (1/30) | 0% (0/30) | 23% (7/30) |
| filler | 3% (1/30) | 0% (0/30) | 0% (0/30) | 20% (6/30) |
| all | 0% (0/30) | 0% (0/30) | 20% (6/30) | 23% (7/30) |

**reachability** (n60 only)
| primer | n60 |
|---|---|
| none | 0% (0/30) |
| components | 0% (0/30) |
| degree | 0% (0/30) |
| clustering | 0% (0/30) |
| rwse | 3% (1/30) |
| filler | 0% (0/30) |
| all | 17% (5/30) |

### Truncation hot spots

- **n80 is a broad, roughly-uniform ~17-27% truncation floor across almost every
  task/primer/model.** At n80, essentially every cell in this document sits in
  that band regardless of primer — meaning at n80, somewhere between 1-in-6 and
  1-in-4 of the *entire* sample (not just the wrong answers) is a token-cap cutoff
  before this project's own scoring even runs. That's large enough to treat any
  n80 comparison as running on a smaller effective n than the nominal 30/cell.
- **`edge_count` truncates far harder than anything else, starting at its very
  first tested size (n60).** 23-57% of all rows for plain, 43-57% for think — at
  n60, not n80. No other task reaches that range even at n80. This confirms (with
  the all-rows metric, not just wrong-answer-only) that `edge_count`'s
  success-rate numbers in Part 1 are truncation-limited at every size this task
  was run at, not primer-limited.
- **`cycle_check`/`degree` (plain) is the standout non-`edge_count` cell**: 0% at
  n20, then 17% at n40, 20% at n60, 37% at n80 — a clean escalation that starts a
  full size class earlier than every other primer on that task. Worth a look at
  whether the `degree` primer text is unusually verbose on `cycle_check`
  specifically or induces longer reasoning before the model commits to an answer.
- **The think arm doesn't truncate uniformly earlier — it truncates in bursts
  tied to specific primers.** `cycle_check`/`filler` jumps to 13% at n40 for
  think while staying at 0% for plain at the same size; `cycle_check`/`degree`
  reaches 10% at n60 for think (vs. plain's 20%, i.e. plain is actually worse
  there). The pattern isn't "think always truncates first," it's "which primer's
  prompt happens to be long enough to matter" — consistent with truncation being
  driven by prompt+response length, not by the thinking arm being categorically
  more verbose at every size.
- **Below n60, truncation as a share of all rows is close to zero almost
  everywhere** — the only cells above 5% below n60 are `cycle_check`/`degree`
  (plain, 17% at n40) and think's `cycle_check`/`filler` (13% at n40). Every
  other task/primer/model cell at n20/n40 is 0-7%. Truncation as a driver of the
  sweep's sample loss is concentrated at n60+ and in `edge_count`/`cycle_check`
  specifically, not a general problem across the sweep.

## Bottom line

- Genuine (non-shortcut-contaminated) primer effects in this uncontrolled-density
  sweep exist but are size-dependent, not flat: `connected_nodes` `filler`/`rwse`
  help plain qwen3-1.7b, growing from ~0 at n20 to +9-11pp at n40-n80;
  `cycle_check` `clustering` helps the think arm consistently from n20 through
  n80 (bar 0.83, same as `none`, and now confirmed on the full n60 sample);
  `edge_existence` `rwse` helps plain qwen3-1.7b at n40/n60 specifically but not
  the think arm anywhere. `node_degree` shows no clean positive Δ for either
  model at any size once contaminated conditions (`degree`, `all`) are excluded.
  Most of the large positive Δs on `node_count`/`edge_count`, and most
  `all`-condition Δs, are shortcut-bar ⚠️ and shouldn't be read as reasoning
  gains regardless of size — this is a content limitation, not something a
  different node-count range can fix.
- Pooling across sizes (the original version of this table) actively hid
  reversals — several conditions look flat or negative pooled but are strongly
  positive at one size and strongly negative at another (`edge_existence`/`all`,
  `cycle_check`/`all` for plain qwen3-1.7b). Any future primer-effect summary
  for this project should report by size, not pooled.
- **Truncation, now measured as a share of all rows, is concentrated at
  n60+/`edge_count` and n80 generally** — roughly 17-27% of every n80 cell and
  23-57% of every `edge_count` cell from n60 up, regardless of primer. This
  directly explains part of the think arm's apparent `all`-condition regressions
  at n60/n80 (Part 1). Any primer-effect conclusion drawn from n60/n80 cells
  (especially `edge_count`, and the think arm's bursty per-primer truncation)
  should be treated as running on a reduced effective sample, not the nominal
  n=30, until the token-budget plan
  (`can-i-shorten-the-structured-sloth` in `~/.claude/plans/`) is implemented and
  re-run.

## Recommended node-count range

**Method.** For every (task, size) cell, classify `none`'s own accuracy (not any
other primer's — this avoids the shortcut-bar contamination discussed above,
since `none`'s bar is always low, 0.02-0.83) into a zone:
- **ceiling** if `none` ≥ 90% — the task is already saturated, no room for a
  primer to add measurable value;
- **floor** if `none` ≤ `graphtalk.scoring.majority_baseline` (+2pp) for that
  cell — the model is at or below what a graph-blind constant-answer guesser
  scores, so there's nothing a primer could add either;
- **informative** otherwise.

Separately, flag a cell **truncated** if any primer's truncated-rate-of-all-rows
(Part 2) is ≥10% in that cell — a comparison there risks confounding a primer
effect with a token-budget artifact regardless of whether `none` itself is in
the informative zone.

| task | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| node_count | ceiling (93%, bar=100%*) | floor (73%, bar=100%*) | floor (73%, bar=100%*) | floor + **trunc** (50%, bar=100%*) |
| node_degree | ceiling (97%) | informative | informative | informative + **trunc** |
| edge_count | n/a (not run) | n/a (not run) | floor + **trunc** | floor + **trunc** |
| edge_existence | ceiling (97%) | informative | floor (57%, bar≈70%) | floor + **trunc** |
| cycle_check | informative | informative + **trunc** (degree primer, 17%) | floor (73% == bar) + **trunc** | floor + **trunc** |
| connected_nodes | ceiling (97%) | informative | informative | informative + **trunc** |
| reachability | n/a (not run) | n/a (not run) | ceiling (100%) | n/a (not run) |

*`node_count`'s "bar" here is its majority baseline, not the shortcut bar — it
reads 100% at every size **by construction**: the size-sweep design fixes every
graph in a given `n` cell to exactly `n` nodes, so "always answer `n`" is a
perfect guesser regardless of anything the model does. This isn't fixable by
choosing a different size; `node_count` cannot serve as a primer-effect probe in
this design at all, at any `n`. (`edge_count`/`reachability` weren't run at
n20/40/80 in this sweep, so those cells are marked n/a rather than scored — see
the size-availability note under each task's Part 1 table.)

**Recommendation: use n40 and n60 as the core range; treat n20 as a ceiling
reference point only, and n80 with the token-budget caveat above (or don't use
it until the budget is raised).**

- **n40-n60 is the widest window that's `informative` (not ceiling, not floor)
  and truncation-clean for the most tasks**: `node_degree` and `connected_nodes`
  clear it at both sizes for both models; `edge_existence` clears it at n40 only
  (floors by n60, `none` there is barely above the class-imbalance baseline);
  `cycle_check` is the exception — it starts truncating on the `degree` primer
  specifically already at n40 (17%) and floors outright by n60 for the plain
  model, so it doesn't have a genuinely clean point past n20 in this sweep.
- **n20 is mostly a ceiling point** (`node_count`, `edge_existence`,
  `connected_nodes`, `node_degree` are all ≥90% `none` accuracy for at least one
  model) — useful as a "task hasn't broken yet" reference, not for detecting a
  primer's Δ. The one exception is `cycle_check`, where n20 is actually the
  *cleanest* point available (informative, 3% truncation) — everything past it
  either truncates on `degree` or floors.
- **n80 should be treated as truncation-compromised across the board**: every
  task/model cell there carries ≥17% truncated-of-all-rows, on top of several
  tasks already being at or past floor by then. A Δ measured at n80 is a Δ on a
  reduced, cap-selected sample, not the full 30/cell it looks like.
- **node_count, edge_count and reachability are not fixable by picking a
  different size at all**: `node_count`'s floor is structural (majority baseline
  100% by construction), `edge_count` truncates too heavily at every size it was
  run at (needs a raised token budget, not a different `n`), and `reachability`
  was only run at one size and sits at ceiling there. None of the three should
  be expected to produce a primer signal in a size sweep regardless of range.
- **Shortcut contamination is a separate, size-independent constraint on top of
  all this** (see Part 1's bar column): even within the recommended n40-n60
  window, `node_degree`/`degree`+`all`, `edge_existence`/`degree`, and
  `cycle_check`/`components`+`degree`+`all` stay bar ≥0.79-1.00 at every size, so
  their Δs in that window are still content-contamination, not a size problem —
  widening or narrowing the range doesn't rescue those specific (task, primer)
  pairs.
