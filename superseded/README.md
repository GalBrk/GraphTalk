# Superseded

Nothing in this directory is current. Every file here was replaced by a current
version, listed below; it is kept so that nothing is lost and every earlier
number can still be traced to the file and code that produced it.

- **Current results** live in [`docs/results/`](../docs/results/README.md).
- **Why the earlier versions disagreed:** [`paper/CLAIMS_LEDGER.md`](paper/CLAIMS_LEDGER.md)
  checks every claim of every earlier paper version against the raw runs.
- **Exact rebuild** of anything here, as it stood: `git checkout 7037749`, then run
  it from its original path (the "Original path" column).
- **Path-fixed runs:** the layout mirrors the original one (`superseded/<original
  path>`), and paths inside these files were rewritten to point at their new
  locations, so the scripts also run from here (from the repo root, with
  `PYTHONPATH=.`). Their outputs are written inside `superseded/`.
- Two scripts here, `paper/make_v3_tables.py` and `paper/make_v3_figures.py`,
  read files that the current pipeline regenerates (`primer_cells.csv`,
  `edge_existence_collapse.csv`, `node_degree_routes.csv`); the versions they
  were built on, with the `primer_findings.txt` that v3 and the ledger cite, are
  kept in `superseded/csv2/raw-trends/`, and the references point there.

## Paper versions

All analyse the 40-node sweep (`runs/*.densfull40*`). Replaced by
[`docs/results/n40-sweep.md`](../docs/results/n40-sweep.md).

| Original path | What it was |
|---|---|
| `paper/talk_like_a_graph.tex`, `.pdf`, `.NEW.pdf`, and its generated tables (`main_table.tex`, `ci_table.tex`, `appendix_tables.tex`, `connected_nodes_*.tex`, `degfixdeg_table.tex`, `length_cost_table.tex`, `published_ceiling.tex`) and figures (`effects.pdf`, `crossfit.pdf`, `rewiring.pdf`, `density.pdf`, `headline.pdf`, `continuum.pdf`) | v1, "Hints, Placebos and Shortcuts" (8-page body); built by `paper/make_all.sh` and `paper/make_*.py` |
| `paper/talk_like_a_graph.v2.tex`, `.v2.pdf` | v2, the same paper at 12 pages |
| `paper/talk_like_a_graph.v3.tex`, `.v3.pdf`, `v3_*.tex`, `v3_fig_*.pdf` | v3, "When Do Structural Primers Help LLMs Reason over Graphs?"; built by `paper/make_v3.sh`, `make_v3_tables.py`, `make_v3_figures.py` |
| `paper/short/` | a 5-page draft, "What GraphQA Measures at n=40" |
| `paper/compare/independent/` | an independent reanalysis (`analyze.py`, `results/`) and its paper, "Reliability, Consistency, and Feature Resolution" |
| `paper/compare/Structural_Primers_GraphTalk_ACL.pdf` | an earlier build of that paper |
| `paper/synthesis/` | a draft combining v3 and the reanalysis, "Procedural Shortcuts … Invariants and Feature Resolution", with `PROVENANCE.md` |
| `paper/structural_primers_acl2023.pdf` | a 5-page draft built outside this repo, no source |
| `paper/Structural_Primers_Graph_Reasoning_ACL2023.pdf`, `… (1).pdf` | a 6-page draft built outside this repo, no source, in two builds that differ in one table; a byte-identical copy of the second build was removed |
| `paper/CLAIMS_LEDGER.md` | the audit of all the versions above against the raw runs |
| `paper/cut_claims_reproduced.txt` | output of `scripts/reproduce_cut_claims.py` (new here): the values the drafts printed for the claims the ledger cut or found wrong, recomputed from the archived outputs; the same quantities under the current scoring rule are in `csv2/raw-trends/legacy_claims.txt` (`scripts/legacy_claims.py`) |
| `paper/NUMBERS.md`, `paper/cited2.txt` | v1/v3 number-to-source map; a citation list |

Every paper source here compiles from its own directory with `latexmk -pdf
<file>.tex` (the ACL style files and `custom.bib` sit next to each source;
`compare/independent/` also needs its parent directory on the TeX search path:
`TEXINPUTS=..:` on TeX Live, `--include-directory=..` on MiKTeX). v3, v1, the
short draft and the synthesis draft reproduce their committed page counts; v2
builds one page longer, and v1 carries one undefined `\ref`
(`sec:results-difficulty`, from the shared `main_table.tex`), because inputs
they share changed after they were built. The committed PDFs are the
originals.

## Claims that cannot be recomputed from the repo

Every other quantitative claim in the paper versions above is computed by a
committed script (current or archived). These are not:

- **Token counts.** The short draft's `components` +9 and `filler` +457 tokens at
  p=.35 need a Qwen3 tokenizer, which is not in the repo. The measured counts
  that are committed pool all densities (`review_checks.json`, `tokens`:
  `components` 8, `filler` 470) and stand.
- **Hand-labelled parser checks** in the independent reanalysis (its 28-response
  sample, the 24 singleton cases, and the sensitivity of C and J to them): the
  responses are in `paper/compare/independent/results/reviewer_*.jsonl`, but the
  verdicts were read by hand.
- **v1's MDE medians (5.6, 10.0)**: their per-cell inputs are only in the
  untracked `archive/pre-mde-fold/` holding pen.
- **v2's length-cost correlations (+0.55, -0.09, -0.44)**: v2 did not state its
  row set, and neither of the two natural row sets of `primer_decomposition.csv`
  (all rows, mid-range rows) reproduces all three
  (see `paper/cut_claims_reproduced.txt`).
- **v3's +19.9 for 1.7B-thinking over finished pairs** survives in
  `csv2/raw-trends/primer_findings.txt` here; the code that printed it is at
  commit `7037749`.

## Analysis scripts and their tests

Each is a second analysis of the 40-node runs, replaced by
`scripts/primer_findings.py` (output: `csv2/raw-trends/primer_findings.txt`).

| Original path | What it computed |
|---|---|
| `scripts/raw_trends.py`, `scripts/raw_trends_figures.py` | the effect tables and figures v3 was built from |
| `scripts/analyze_primer_survival.py`, `scripts/analyze_effect_drivers.py` | per-density effects and their correlates (v2) |
| `scripts/test_vs_controls.py`, `ci_all.py` | tests against `none`/`filler`/`components`, permutation p-values (v1, v2) |
| `scripts/analyze_review_checks.py` | reviewer checks behind v1, including the cross-fitted interaction the ledger cut; its `tokens` section needs `--tokenizer <Qwen3 tokenizer.json>`, which is not in the repo |
| `scripts/analyze_churn_and_length.py`, `scripts/analyze_error_taxonomy.py`, `scripts/analyze_nontermination.py` | churn, error shape, truncation (v1, v2) |
| `scripts/candidates/a_routegap.py`, `a_transfer.py` | candidate analyses, rejected |
| `scripts/reproduce_cut_claims.py` (new) | recomputes the drafts' values for the claims the ledger cut or found wrong, from the archived outputs; output in `paper/cut_claims_reproduced.txt` |
| `scripts/analyze_baseline_law.py` (copy) | the full version, with the `split`, `heldout`, `instrument`, `crossfit` and `ceiling` tests; the live copy keeps only `continuum` and the shared helpers |
| `scripts/analyze_primer_window.py` (copy) | the version v3 used, with its own report (`window`, `by_primer`, `gap`, `--tex`); the live copy keeps only `cells()` and `window()` |
| `tests/test_analyze_*.py`, `tests/test_test_vs_controls.py`, `tests/test_analyze_baseline_law_superseded.py` | tests of the scripts above; run with `python -m pytest superseded/tests` |

## Outputs

All computed from the 40-node runs by the scripts above; replaced by
`csv2/raw-trends/primer_findings.txt` and its three CSVs.

| Original path | Producer |
|---|---|
| `csv2/sweep-large-graph/` | `score_full_density_sweep.py`, `analyze_primer_survival.py`, `build_sweep_frame.py`, `check_significance.py`, `task_scoped_screen.py` |
| `csv2/raw-trends/{effects,effects_capdropped,edge_existence_balanced,difficulty,output_operation,behaviour,strategy_vs_accuracy,serial_position,error_shape,relevance,additivity,moderators,headroom,marker_validation,ladder_length_vs_difficulty}.csv` | `raw_trends.py` |
| `csv2/raw-trends/{primer_findings.txt,primer_cells.csv,edge_existence_collapse.csv,node_degree_routes.csv}` | `primer_findings.py` before truncation became its own outcome (the version v3 and the ledger cite) |
| `csv2/one-offs/{primer_window,serial_slope_diffs,truncation_flips}.csv` | `analyze_primer_window.py --csv`; the other two have no surviving producer |
| `analysis/raw-trends/` | `raw_trends_figures.py` |
| `analysis/rerun/*densfull40*`, `analysis/rerun/primer_survival*`, `analysis/tables/*densfull40*`, `analysis/tables/primer_survival_manifest.json` | saved console output of the scripts above |
| `ci_all.json`, `ci_all_hi.json`, `vs_controls_densfull40{,hi}.json`, `churn_len_densfull40.json`, `error_taxonomy.json`, `review_checks.json` | `ci_all.py`, `test_vs_controls.py`, `analyze_churn_and_length.py`, `analyze_error_taxonomy.py`, `analyze_review_checks.py` |

Rerunning each script regenerates its committed outputs byte-identically, with
two exceptions:

- `csv2/raw-trends/moderators.csv`, `analysis/raw-trends/fig_additivity.png` and
  `fig_serial_position.png` were generated at commit `b79e394`, and
  `raw_trends.py` changed afterwards (it now drops a pair when either side
  truncated, so a thinking-arm delta can be over 99 pairs instead of 100). The
  exact files come from `git checkout b79e394`.
- the `tokens` section of `review_checks.json` needs the Qwen3 tokenizer (above).

## Docs

Replaced by [`docs/results/n40-sweep.md`](../docs/results/n40-sweep.md).

| Original path | What it was |
|---|---|
| `docs/raw-trends-large-graph.md` | the first rebuild of the 40-node sweep from raw runs |
| `docs/full-task-density-sweep.md` | per-task, per-density tables of the main sweep (before the high-density extension) |
| `docs/primer-effects-paper-draft.md` | a draft of the results section |
| `docs/candidate-analyses.md` | six candidate analyses with adopt/reject verdicts |
| `docs/paper-claim-audit.md`, `docs/paper-revision-handoff.md`, `docs/paper-v2-consolidation.md`, `docs/paper-v3-review.md`, `docs/paper-v3-review-fixes.md` | review and revision logs of v1–v3 |
