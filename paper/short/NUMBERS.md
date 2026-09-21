# Where every number in `short.tex` comes from

One line per figure cited in the 5-page paper. If a number is not here, it does
not belong in the paper.

## Build

```bash
cd paper/short
BIBINPUTS='..;.' BSTINPUTS='..;.' \
  pdflatex --include-directory=.. short.tex && bibtex short && \
  pdflatex --include-directory=.. short.tex && \
  pdflatex --include-directory=.. short.tex
```

`acl.sty`, `acl_natbib.bst`, `custom.bib` and `crossfit.pdf` are read from
`../`. **Nothing in `paper/` outside `paper/short/` is written to.** Verify with
`git status paper/` before committing.

Regenerate the floats first:

```bash
PYTHONPATH=. python paper/short/make_baseline_table.py   # Table 1 + macros.tex
PYTHONPATH=. python paper/short/make_degfixdeg_table.py  # Table 2
```

Both read `runs/*.jsonl` directly, so a table cannot drift from the
generations. `make_degfixdeg_table.py` asserts its pooled figure against
`scripts/score_fixed_degree_sweep.py`'s committed output and fails loudly if
they disagree.

## Section 2 — method

| Number | Source |
|---|---|
| `filler` matches `clustering` to within 7.8% in tokens | `prompts.densfull40.jsonl`; verified in `docs/paper-claim-audit.md` §1 |
| `components` adds 9 tokens, `filler` adds 457 | same, mean over tasks at p=0.35 |
| route split: largest sub-threshold 0.011, smallest supra-threshold 0.066 | `shortcuts_n40_flat.json`; audit §1 verified all 42 cells |
| 16,800 prompts per arm | `analysis/{arm}.densfull40.rows.csv`, 16,800 rows each |
| truncation under 6.5%, `edge_count` 22.4% on `qwen3-1.7b`, ≥62% on think arms | audit §3; corrects the 8-page paper's "fewer than 7%" |

## Section 3 — what the six tasks measure

| Number | Source |
|---|---|
| Table 1 in full (prior and `none` accuracy, 6 tasks × 7 densities × 2 arms) | `paper/short/make_baseline_table.py` → `baseline_table.tex` |
| 400/400 `cycle_check` graphs cyclic; prior 1.000 | gold distribution, `runs/*.densfull40.*`; same script |
| `node_count` 1.7b: 0.020 / 0.000 / 0.280 / 0.960 | Table 1 |
| "39" rate 98% / 100% / 72% / 4% | `make_baseline_table.py` → `macros.tex` (`\ncThirtyNine*`). The **pooled** per-condition version is `scripts/analyze_error_taxonomy.py` (`rate_39`), which is where the 8-page paper's 68.5% comes from; the per-density split is new. |
| at p=0.20 the body lists all 40 nodes, model still answers 39 on 100/100 | regex over `prompts.densfull40.jsonl` bodies (`Node (\d+) is connected`), min distinct = 40 |
| `edge_count` median relative error, 4b: 8.1 / 9.7 / 12.6 / 17.2% | `macros.tex` (`\ecRelFourb*`) |
| think arms exact 0.960 (1.7b-think p=.10), 0.975 / 1.000 / 0.875 (4b-think) | `runs/*-think.densfull40.*`, `scoring.score_one`; truncation 50–100% |
| `edge_existence` 1.7b hit 1.000, FA 0.122→0.933 across 7 densities; 4b FA 0.044–0.375 | recomputed with the same predicate as `analyze_error_taxonomy.py`. **Its pooled equivalent is `tab:booleanbias` in the 8-page paper: 40.3% (1.7b) / 12.6% (4b), which this reproduces exactly when pooled over the four main-sweep densities. That table's caption says "p=0.50"; `edge_existence_taxonomy()` applies no density filter, so the caption is wrong and the numbers are right.** |
| `node_degree` 1.7b below prior at p≥.65 (0.081/0.160/0.050 vs 0.170/0.160/0.200) | Table 1 |
| published split: `cycle_check` prior 83.2% (416/500), 0.6b `none` 78.8%, `components` 83.4% | `analysis/tables/published_split.cc500.txt` (`scripts/score_sweep.py`) |
| published split: `edge_count` prior 3.4%, 1.7b 23.4%, 8b 50.0% | `analysis/tables/published_split.ec500.txt` |

## Section 4 — route substitution

| Number | Source |
|---|---|
| interaction −11.0, 95% CI [−17.1, −5.7] | `scripts/analyze_baseline_law.py --test crossfit`; audit §1 verified against `review_checks.json["interaction"]` |
| relation disappears inside the 0.10–0.90 window | same, windowed row |
| per task: 232 cells r=−0.308 slope −10.5 [−16.6,−5.4]; 168 `node_degree` r=−0.235 p=0.0022 slope −9.1 [−21.6,−1.8]; 64 `edge_count` r=−0.664 slope −26.5 | audit §4; reproduce with `analyze_baseline_law.py`'s `arm_cells_crossfit` filtered by task (`score_run` accepts `task_filter`) |
| `node_degree` truncates on 0–1.6% | `hit_cap` counts, `densfull40` + `densfull40hi` |
| `degree` benefit +3.1 → +34.2 for 1.7b-think | audit §1 verified; strictly monotone |

## Section 5 — length

| Number | Source |
|---|---|
| 29 content / 21 length / 11 net BH-significant of 90 cells | `analysis/tables/primer_decomposition.csv` (`scripts/analyze_primer_survival.py`) |
| `filler` hurts `connected_nodes` and `edge_existence` in both plain arms | `paper/main_table.tex`; replication across sizes from `analysis/tables/primer_survival.csv` |
| `rwse` costs 8b 13.3 points of `edge_count` (36.7 vs 50.0, p<1e-3); 1.7b 4.6 points (p=0.036) | `analysis/tables/published_split.ec500.txt` |
| non-termination 15.8→19.2% (p=2.4e-6); outside `edge_count` 2.6→6.9% (p=9.2e-11), 130 new failures | `analysis/tables/nontermination.densfull40.txt` (`scripts/analyze_nontermination.py`) |

## Section 6 — what survives

| Number | Source |
|---|---|
| exactly two non-route primers beat both controls | exhaustive scan of `vs_controls_densfull40.json` over non-route conditions with `bh_global_reject` on both controls |
| `qwen3-4b`/`connected_nodes`/`components`: +9.2 vs `none`, +21.8 vs `filler`, n=400, p≈1e-4 | `vs_controls_densfull40.json`; **not reported in the 8-page paper**, and it falsifies that paper's "the only primer that…" claim |
| `clustering`/`node_degree` +5.9 [2.1, 9.8]; replication +5.5; main sweep +3.0 p=0.26 | audit §1 verified the whole block |
| Table 2 in full, pooled +6.2, n=3,191, 504 helped / 306 hurt, p<1e-4, 6/8 cells BH | `paper/short/make_degfixdeg_table.py`, asserted against `analysis/tables/degfixdeg.1.7b.txt` |
| +8.2 at mean degree ≈16 vs +4.2 at ≈8 | `analysis/tables/degfixdeg.1.7b.txt`, "pooled per mean-degree block" |
| `qwen3-8b` 0.998–0.802, shows nothing | `analysis/tables/degfixdeg.8b.txt`; Table 2 columns 6–7 |
| n 20→160 costs 29 points and saturates; doubling mean degree costs 33–42 at every size | Table 2, `none` column |

## Limitations / Ethics

| Number | Source |
|---|---|
| rewiring drives mean clustering 0.01→0.73, benefit flat | `scripts/analyze_rewiring_sweep.py`; audit §1 verified |
| density-prior predicts a shift up to 17, observed within 0.3 | `tab:prior` in the 8-page paper; audit §1 verified |
| 217,178 generations | `wc -l runs/*.jsonl` excluding `runs/archive/`; audit §3 (the 8-page paper says "approximately 200,000") |

## Deliberately not used

- Reading limits from `docs/ladder-and-retrieval-results.md`. The
  `retrieval_locate` probe measures single-fact retrieval with no graph in the
  prompt; the tasks here require aggregation over the whole encoding, so the
  limit is not a graph-reading budget and cannot serve as a token threshold.
- `analysis/topology_drivers_report.csv` (forests +10.1, bipartite +10.1).
  Exploratory multi-feature dredging on a superseded corpus and a non-main arm;
  its own docstring says to treat it as hypothesis-generating.
- `analysis/tables/size_baseline.csv` — the `size` runs use an unpinned density
  near p≈0.65, so they do not describe the n=40 sweep. Superseded by Table 1.
