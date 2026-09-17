# Candidate analyses

Throwaway-quality scripts that produced every number in
`docs/primer-effects-and-power.md`'s companion,
[`docs/candidate-analyses.md`](../../docs/candidate-analyses.md). None of them
feeds the paper. They are kept so the decision about what to fold in can be
revisited without re-deriving the numbers.

**Status: candidates, not pipeline.** They have no tests, hardcode their run
globs, and print rather than export. Anything adopted into the paper should be
rewritten into `scripts/` proper with unit tests, the way
`scripts/analyze_baseline_law.py` was -- these exist to preserve the working,
not to be maintained.

Run every one from the repo root with `PYTHONPATH=.`; they import
`scripts/analyze_baseline_law.py` as a library for its `pearson`/`ols`/
`fit_line`/`route_gain` helpers.

| script | section of `candidate-analyses.md` | notes |
|---|---|---|
| `a_routegap.py` | (a) route gap vs baseline | takes an output path for the cell dump |
| `a_transfer.py` | (a), cross-corpus sign transfer | reads `a_routegap.py`'s dump |
| `b_copying.py` | (b) are `degree` errors copying errors | parses graphs out of `prompts.densfull40.jsonl` |
| `c_confound.py` | (c) density vs answer magnitude | `degfixdeg` plus a within-density regression |
| `de_churn_len.py` | (d) and (e), the scoring pass | writes a JSON dump; slow, one pass over `densfull40` |
| `de_report.py` | (d) and (e), the tables | reads `de_churn_len.py`'s dump |
| `f_clustering_size.py` | bonus, `clustering` across graph size | |
| `g_retrieval.py` | (f1) the retrieval probe | |
| `h_got.py` | (f2) GoT naming vs integer ids | desubstitutes before scoring; see `graphtalk/node_naming.py` |

```bash
PYTHONPATH=. python scripts/candidates/a_routegap.py /tmp/a_cells.json
PYTHONPATH=. python scripts/candidates/a_transfer.py /tmp/a_cells.json
PYTHONPATH=. python scripts/candidates/b_copying.py /tmp/b_detail.json
PYTHONPATH=. python scripts/candidates/c_confound.py
PYTHONPATH=. python scripts/candidates/de_churn_len.py /tmp/de_rows.json
PYTHONPATH=. python scripts/candidates/de_report.py /tmp/de_rows.json
PYTHONPATH=. python scripts/candidates/f_clustering_size.py
PYTHONPATH=. python scripts/candidates/g_retrieval.py
PYTHONPATH=. python scripts/candidates/h_got.py
```

One trap worth keeping in mind, since it is the kind that fails silently:
`de_churn_len.py` drops `hit_cap` pairs from the *length* comparison as well as
the accuracy one. A truncated row reports a censored 8192 tokens, and leaving
those in swings `qwen3-1.7b`/`edge_count`/`degree` from -1098 to -2181.
