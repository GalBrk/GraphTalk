# Superseded — do not cite

Produced by analysis code that has since been fixed. Kept only so a claim in
`docs/` can be traced to its source. Not regenerated here: the experiment is
being re-run, and the corrected scripts write back to `analysis/`.

Four defects in the code that wrote these files:

1. **The cluster key was a no-op.** `check_significance.py` clustered on
   `(model, instance_id)`, but `instance_id` is `"<task>/<index>"` and
   `node_count/7` and `edge_count/7` are the same graph. Every cluster held
   one pair, so the clustered permutation test corrected for nothing. Now
   keyed on `(model, graph_index)`. Changes four verdicts:

   | cell | published here | corrected |
   |---|---|---|
   | pooled `degree` | 0.0050 ✅ | 0.0043 ✅ |
   | pooled `rwse` | 0.0105 ✅ | 0.0327 ✗ |
   | `qwen3-14b`/`degree` | 0.0205 ✗ | 0.0079 ✅ |
   | pooled `all` (non-term.) | 0.0482 ✅ | 0.0704 ✗ |

2. **Verdicts inside Monte Carlo noise.** At `n_perm=10_000` a p-value moves
   in steps of 1e-4, against a whole-table threshold of 5e-4. The only
   whole-table-significant row here cleared it by **5e-8** and reverses at a
   higher count. Default is now 200,000, with `seed`/`n_perm` recorded and a
   `near_threshold` flag. **The corrected table has no whole-table-significant
   rows.**

3. **The MDE injector moved pairs one way only**, so every `mde_*` column
   here is too small and every power claim too high (~2x on a representative
   cell).

4. **Degenerate CIs.** Six rows publish a 95% CI of `[0.000, 0.000]`; others
   print intervals off 2 discordant pairs in 180. The corrected code returns
   no interval below 10 discordant pairs.

`significance_report.txt` is additionally pre-Phase-2: it still contains the
retired `best_case`/`worst_case` arms and its pooled numbers contradict the
CSV's. `significance_report_mae.csv` and the three `*_by_style`/`_mde` CSVs
predate the unified-correction fix. `failure_sample.csv` predates the
extraction fixes.
