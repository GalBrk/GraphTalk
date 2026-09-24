# One source of truth per family of runs — design

Status: **design, approved** (2026-09-24). Branch `results-sot`, based on
`7037749`. The implementation plan will be written next to this file once the
design is approved. Both files move to `superseded/docs/plans/` when executed.

## Goal

For every family of runs in `runs/`:

1. exactly **one current pipeline** (one script, or a chain of scripts in which
   every reported quantity is computed in exactly one place);
2. exactly **one current results doc**, `docs/results/<family>.md`, stating only
   what that pipeline's committed output shows, with every number traceable to it
   and checked by a test;
3. every other doc, script and output about those runs moved to `superseded/`,
   with nothing deleted and every path that pointed at a moved file fixed;
4. every number in the results doc **independently re-derived** from `runs/` by an
   agent that does not use the pipeline's code, repeated until a round finds
   nothing.

The paper is out of scope. It will be written later, from `docs/results/`.

## Why

The repo holds eight versions of the paper and about ten docs that analyse the
same 84,000 40-node responses, and they disagree (see
`paper/CLAIMS_LEDGER.md`). The cause is structural, not a single bug: numbers
were computed by three different pipelines with different truncation rules, typed
by hand into prose, and older documents were never updated when a later analysis
replaced them. The same pattern holds in other families: 12 places were found
where two docs state different numbers or conclusions for the same quantity on the
same runs (for example the `clustering` replication is +3.8, +1.9, +4.0 and +5.9
in four docs; the qwen3-1.7b reading limit is ~2,505 and ~1,509 tokens in two).

## Repo-wide rules

**R1 — truncation is its own outcome.** Every response has exactly one outcome:
`correct`, `wrong` or `truncated`. A response that hit the token budget is
`truncated`, whatever its abandoned text says; it is never labelled wrong and
never dropped. The same rule applies to every family:

- every cell reports the three outcomes as shares of all its responses; they sum
  to 100%;
- a primer's effect is the paired change in each share on the same graphs. The
  headline is the change in the correct share, with its bootstrap interval
  (graphs resampled within density) and exact McNemar test on correct versus not
  correct; the changes in the wrong and truncated shares are reported next to it,
  so a reader sees whether answers moved from wrong to correct or from correct to
  truncated;
- a `truncated_but_correct` flag records a truncated response whose abandoned
  text contained the gold answer;
- quantities that describe an answer (error magnitude/MAE, yes-rate, false
  alarms, off-by-k) and descriptions of response text (route, discrepancy
  reports, wording) use finished responses only, and state their n;
- a cell whose truncated share is 15% or more is flagged.

R1 is implemented once, in `graphtalk/`, and every current script calls it.
Scripts that drop truncated rows or force them to wrong today are switched,
including `graphtalk/analysis.py::build_frame` and `check_significance.py`
(which force them to wrong and impute their MAE).

**R2 — metrics.** Exact match is the primary metric for every task.
`connected_nodes` is scored by exact set match, with set-F1 as a secondary
column. `edge_existence` is reported with balanced accuracy and the "yes" rate
next to raw accuracy, because the share of true edges rises from 10% to 85%
with density.

**R3 — every number is tagged and tested.** Each current pipeline writes its full
output to a committed file. Each results doc names that file in its header and
cites every number exactly as the file prints it, followed by its tag, e.g.
`+27.3 [edgecount]`. `tests/test_results_docs.py` parses every doc in
`docs/results/`, and fails if a cited value does not appear under its tag in the
named output. A tag's block runs from the line that starts with `[tag]` to the
next line that starts with a tag. A doc may cite a quantity owned by another
family's doc by linking to it, never by restating the number.

**R4 — current claims only.** A results doc states what the current output shows.
It does not mention earlier versions, corrections or retracted numbers. A claim
the current pipeline cannot reproduce is left out of the current doc; it survives
only in `superseded/`.

**R5 — conflicts are resolved by rerunning, never by choosing.** Where two docs
disagree, the current pipeline is rerun from `runs/` under R1–R2, and the doc
states what it prints.

**R6 — `superseded/`.** One tracked top-level directory that mirrors the original
layout (`superseded/paper/…`, `superseded/docs/…`, `superseded/scripts/…`,
`superseded/csv2/…`, `superseded/analysis/…`, `superseded/tests/…`).

- Files are moved with `git mv`. Nothing is deleted except (a) regenerable build
  products that git already ignores (LaTeX `.aux/.log/.fls/.fdb_latexmk/.bbl/
  .blg/.out`, `__pycache__/`) and (b) the byte-identical duplicate
  `Structural_Primers_Graph_Reasoning_ACL2023 (2).pdf`.
- Untracked records (the two remaining draft PDFs under `paper/compare/`) are
  committed into `superseded/`.
- Inside moved files, paths to other moved files are rewritten to their
  `superseded/` location. A moved script's default output path points inside
  `superseded/`, so running it never writes into the live tree.
- A moved script whose inputs are about to change (v3's three `primer_findings`
  CSVs) gets a snapshot of the current inputs next to it, so it still
  reproduces its numbers exactly.
- `superseded/README.md` lists every moved item: original path, what replaced
  it, and how to reproduce it (the path-fixed command, or `git checkout 7037749`
  for an exact rebuild).
- Moved tests are not collected by the default `pytest` run (`testpaths` is
  unchanged); each is run once from `superseded/` during verification.
- The existing gitignored root `archive/` holding pen is not touched. `superseded/`
  is not matched by the `archive/` ignore rule.

## Layout after the work

- `docs/results/README.md`: the index, one row per family: doc, pipeline,
  reproduce command, output file.
- `docs/results/<family>.md`: one per family (below).
- `docs/`: only live design, data and plan docs.
- `superseded/`: everything else, with its README.
- `paper/`: the ACL style files, `custom.bib`, and a README saying there is no
  current paper and results live in `docs/results/`.

## Families, pipelines and decisions

Each phase follows the same steps: (1) align the pipeline to R1–R2 and rerun it
from `runs/`; (2) write the results doc from its output; (3) move everything else
about those runs to `superseded/` and fix pointers; (4) verify (below); (5)
commit.

### Phase 1 — 40-node sweep (`densfull40`, `densfull40hi`)

- **Pipeline:** `build_raw_frame.py` → `csv2/raw-trends/frame.csv` →
  `primer_findings.py` (with `analyze_primer_window.cells()`), plus
  `shortcut_table_n40.py` → `shortcuts_n40{,_flat}.json` for the solver bars.
- **R1 change:** `pairs()` and `cells()` keep every pair and report the three
  outcome shares; `pairs_as_error()` is deleted; the `cells()` rule of at least
  50 untruncated pairs is replaced by the 15% truncation flag; MAE, yes-rate and
  false-alarm claims use finished responses; `tests/test_primer_findings.py`'s
  pairing test is updated.
- **Additions still needed** (`[leak]`, `[copyerr]` and the high-density
  `[replic]` line landed in `7037749`): `[joint]` degree/neighbour joint
  correctness and consistency per arm and condition, with paired effects;
  `[balanced]` balanced accuracy and yes-rate per density for both plain arms;
  `[rwse]` distinct random-walk pairs per graph by density, counted from the
  rendered primer text, with degree classes for comparison.
- **Checkpoint:** after the rerun, compare every claim in
  `paper/CLAIMS_LEDGER.md` against the new output. Any claim whose verdict
  changes is brought to the user before the doc states it.
- **Doc:** `docs/results/n40-sweep.md`, organised like the ledger's §12. The
  `clustering` replication on the dedicated runs stays in `primer_findings.py`
  (`[replic]`); Phase 2's doc links to it instead of restating it.
- **Superseded:**
  - *paper/:* all versions (v1, v2, v3, `short/`, `compare/`, `synthesis/`,
    the three draft PDFs), their generated tables and figures, all their build
    scripts (`make_all.sh`, `make_v3.sh`, `make_*.py`), `NUMBERS.md`,
    `cited2.txt`, `v4.log`, and `CLAIMS_LEDGER.md` (as the record of why the
    versions differed). v3 gets a snapshot of `primer_cells.csv`,
    `edge_existence_collapse.csv` and `node_degree_routes.csv` as they are now.
  - *scripts:* `analyze_churn_and_length.py`, `analyze_effect_drivers.py`,
    `analyze_error_taxonomy.py`, `analyze_nontermination.py`,
    `analyze_primer_survival.py`, `test_vs_controls.py`, `raw_trends.py`,
    `raw_trends_figures.py`, root `ci_all.py`, `candidates/a_routegap.py`,
    `candidates/a_transfer.py`, `analyze_review_checks.py`, and the tests of
    each.
  - *`analyze_baseline_law.py`:* a full copy goes to `superseded/scripts/`; the
    live copy loses the densfull40 tests (`split`, `instrument`, `crossfit`,
    `heldout`, which produce claims the ledger cut) and keeps what live scripts
    import and what other phases use.
  - *outputs:* all of `csv2/sweep-large-graph/`;
    `csv2/one-offs/{primer_window,serial_slope_diffs,truncation_flips}.csv`;
    the 14 CSVs `raw_trends.py` writes into `csv2/raw-trends/` and
    `ladder_length_vs_difficulty.csv`; all of `analysis/raw-trends/`; the
    40-node files in `analysis/rerun/` and `analysis/tables/`; root
    `ci_all.json`, `ci_all_hi.json`, `vs_controls_densfull40{,hi}.json`,
    `churn_len_densfull40.json`, `error_taxonomy.json`, `review_checks.json`.
  - *docs:* `primer-effects-paper-draft.md`, `raw-trends-large-graph.md`,
    `full-task-density-sweep.md`, `candidate-analyses.md`,
    `paper-claim-audit.md`, `paper-revision-handoff.md`,
    `paper-v2-consolidation.md`, `paper-v3-review.md`,
    `paper-v3-review-fixes.md`.

### Phase 2 — density follow-ups (`degdens40`, `degdens40hi`, `degdensthink`, `degdensfill`, `degdensfillT`, `degdensrep`, `degceil`, `degfixdeg`, and the `density40` pilot)

- **Pipeline:** `score_density_sweep.py` (per-density scoring; switched to R1,
  since it drops truncated rows today), `score_fixed_degree_sweep.py`
  (`degfixdeg`; stale docstring glob fixed), and `analyze_rq3_leads.py` (the
  CPU forensics). Each quantity is computed in one of them.
  `analyze_headline_robustness.py`, `candidates/c_confound.py` and
  `candidates/f_clustering_size.py` are kept only for quantities no other
  script computes; otherwise they are superseded.
- **Decided by rerun (R5):** the headline density window for the `clustering`
  effect, the replication figure (4.2 or 4.3), and whether the early-node
  position mechanism in `rq3-leads.md` holds.
- **Doc:** `docs/results/density-followups.md`, with `density40` as a pilot
  section.
- **Superseded:** `rq3-leads.md` (its reproducible claims move into the doc),
  `primer-impact-and-truncation-density-n40.md`,
  `node_degree-density-and-size.md`, `handoff-structural-sweep.md`,
  `csv2/one-offs/qwen3-1.7b.density40.bytask.csv` if no current script
  produces it. `plans/rq3-gpu-tests.md` stays live (a plan, not results), with
  its "+5.9" replaced by a link to the doc.

### Phase 3 — 5–19-node sweep (main arms and `.rerun.` files), GoT naming (`got`, `got.count500`)

- **Pipeline:** `build_sweep_frame.py` → `check_significance.py`, both switched
  to R1 (`build_frame` stops forcing truncated responses to wrong;
  `check_significance.py` drops its forced-wrong columns and MAE imputation);
  `score_sweep.py` for per-cell accuracy, switched to R1; `naming_effect.py` for
  GoT; the `got.count500` investigation scripts (`check_old_vs_new_subsample.py`,
  `diff_shared_instances.py`, `extract_graph_topology.py`,
  `compare_old_vs_new_topology.py`, `analyze_topology_drivers.py`).
- **Docs:** `docs/results/small-graph-sweep.md`, `docs/results/got-naming.md`.
- **Superseded:** `sweep-findings.md`; the results sections of
  `analysis/README.md` (its regeneration notes fold into the doc, so the whole
  file moves); `analysis/primer_task_shortcut_audit.md`;
  `analysis/task_scoped_screen_comparison.md`; `analysis/superseded/`;
  `check_significance_glmm.py`; `rewording_effect.py` (needs a pre-rerun copy
  of `runs/` that is not in the repo); `candidates/h_got.py`;
  `plans/scale-vs-topology-investigation.md` (done; its reproducible claims move
  into `got-naming.md`). `plans/run_improved_tests.md` stays live: its later
  phases are still outstanding. Its pointers are fixed.
- **Also fixed:** `DATA.md`'s pre-purge counts and its "filter truncated rows"
  instruction (lines 96–108, 513–519); `runs/README.md`'s 96.7% (C1).

### Phase 4 — published-split probes (`probe100`, `cc500`, `ec500`), size sweep (`size`)

- **Pipeline:** `score_sweep.py` (R1) for the probes; the size sweep gets
  size-aware scoring by extending `score_density_sweep.py` to group by `n`
  (today it prints `p=None`), rather than a new script.
- **Docs:** `docs/results/published-split-probes.md` (including the
  undocumented `qwen3-4b` arm), `docs/results/size-sweep.md` (including its
  undocumented `qwen3-4b` arm).
- **Superseded:** `primer-impact-and-truncation.md` and `csv2/graph-size-test/`
  (both describe a size sweep whose runs were deleted in `b49ce3b`; exact
  reproduction is from that commit).

### Phase 5 — ladder, retrieval, rewiring (`ladder_screen`, `retrieval_locate`, `retrieval_extend`, `retrieval`, `retrieval_threshold`, `rewire_shared`, `rewire_extra`)

- **Pipeline:** `analyze_ladder.py`; `analyze_retrieval.py`, extended to cover
  the `retrieval` and `retrieval_threshold` families and the 8B runs, which no
  doc reports today; `analyze_rewiring_sweep.py` → `rewiring_*.json`.
- **Docs:** `ladder-and-retrieval-results.md` becomes
  `docs/results/ladder-retrieval.md` (`git mv`, then rewritten from the output);
  a new `docs/results/rewiring.md`.
- **Superseded:** `candidates/g_retrieval.py` (its glob mixes three families);
  `csv2/ladder-retrieval/retrieval_extend_matrix.csv` if nothing current
  produces it.
- **Fixed in place:** the stale per-model numbers and status table in
  `graph-design-requirements.md` and `graph-corpus-status.md` become links to
  the doc; `ladder-and-rewiring.md`'s rewiring command, whose glob matches
  nothing.

### Phase 6 — repo-wide

- `primer-effects-and-power.md` moves to `superseded/` once every family doc
  carries its reproducible content.
- `docs/results/README.md` (the index) is written.
- `CLAUDE.md`: the results-doc paragraph, the docs table (which today omits 12
  docs), the command list (`analyze_baseline_law.py`'s cut tests), and
  "content-free" → "length-matched, structure-free".
- Root `README.md`: its results summary becomes a pointer to the index.
- `repo-scope.md`, `collaborator-access.md`, `scripts/candidates/README.md`
  (which lists three scripts that never existed), `paper/README.md`.
- Executed or finished plans in `docs/plans/` move to `superseded/docs/plans/`:
  `primer-computation.md`, `shortcut-ceilings.md`,
  `finding-graphs-that-make-primer-effects-measurable.md`.
- The assistant's memory notes are updated to point at `docs/results/`.

## Verification

**Per phase.** After each phase's doc is written, a fresh agent that has not seen
the pipeline's code re-derives every number in the doc from `runs/` with its
own scoring code, under R1–R2, and checks every path in the files the phase
touched. I fix what it finds, then a *new* agent re-verifies. This repeats until
a round reports nothing. Only then is the phase committed.

**Final, whole repo.** After Phase 6, a final agent checks:

1. each family in `runs/` has exactly one results doc and one pipeline outside
   `superseded/` (by searching every live doc and script for each family's
   filename token);
2. no live or superseded file references a path that does not exist;
3. `pytest` passes (with the two `statsmodels` files ignored), including
   `tests/test_results_docs.py`, and each moved test passes when run from
   `superseded/`;
4. no claim the ledger cut appears outside `superseded/` (for example −11.0,
   "eleven held-out", +5.9, "content-free", "2,048 for plain", "both thinking
   arms at ceiling");
5. nothing was lost: `git diff --name-status --find-renames 7037749..HEAD` shows
   only renames, modifications and additions, apart from the duplicate PDF.

Fixes, then a new final agent, until a round is clean.

## Git

One commit per phase, plus one per verification-fix round. No push. The
branch is merged only when the user asks.

## Out of scope

- Generating new model responses; every number comes from existing `runs/`.
- Changing answer extraction in `graphtalk/scoring.py`.
- Writing the paper.
