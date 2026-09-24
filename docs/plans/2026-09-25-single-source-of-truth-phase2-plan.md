# One source of truth — Phase 2 implementation plan (density follow-ups)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (the user chose native execution). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give the density follow-up runs one pipeline and one results doc, under the same rules as Phase 1, and move every other analysis of these runs to `superseded/`.

**Architecture:** `scripts/score_density_sweep.py`, `scripts/score_fixed_degree_sweep.py` and `scripts/score_full_density_sweep.py` are one scorer written three times; only the grouping key differs (density; (size, density); (task, density)). They merge into `score_density_sweep.py` with a `--group` option, switched to R1 (`graphtalk/outcomes.py`) and R2 (exact match primary). `scripts/analyze_rq3_leads.py` keeps the forensics, switched to R1. A driver, `scripts/density_followups.py`, runs every run set of the family through them under a tag and writes one committed output, `csv2/density-followups/density_followups.txt`, that `docs/results/density-followups.md` cites.

**Spec:** `docs/plans/2026-09-24-single-source-of-truth.md` (R1–R6; "Phase 2"). **Phase 1** (committed) owns `[replic]` and `[rerun]` in `primer_findings.py`; the new doc links to `docs/results/n40-sweep.md` for them instead of restating them.

## Global Constraints

- R1: truncated is its own outcome; effects are paired changes in the correct share with the truncated change beside them; answer descriptions (MAE, token counts) use finished responses; a cell with ≥15% truncated on either side is flagged and left out of pooled analyses and trends, and listed.
- R2: exact match is primary for every task (`connected_nodes` exact set match; F1 secondary).
- R3: every number in the doc is written as printed and tagged; `tests/test_results_docs.py` checks it.
- R4: current claims only. R5: conflicts resolved by rerunning. R6: moves with `git mv` into `superseded/`, paths fixed, nothing lost.
- Rounding: one-decimal effects use `primer_findings.f1` (ties to even on the decimal value).
- Memory on this laptop is short: load one run set at a time; never hold all families in memory.
- Stage as you go; commit once after a clean verification round.

## The family

| Run set | Files | Model | Design |
|---|---|---|---|
| `degdens40` | 5 × 960 | qwen3-1.7b | p .10–.50 × {none, components, clustering}, 400 graphs |
| `degdens40hi` | 5 × 960 | qwen3-1.7b | p .65–.85 × {none, components, clustering, degree}, 400 |
| `degdensthink` | 7 × 1,600 | qwen3-1.7b-think | 7 p × {none, components, clustering, degree}, 400 |
| `degdensfill` | 5 × 560 | qwen3-1.7b | filler × 7 p × 400 |
| `degdensfillT` | 7 × 400 | qwen3-1.7b-think | filler × 7 p × 400 |
| `degdensrep` | 5 × 640 | qwen3-1.7b | p .10–.50, fresh seeds, {none, clustering} × 400 |
| `degceil` | 3 files, 1,600 | qwen3-1.7b | degree × p .10–.50 on degdens40's graphs |
| `degfixdeg` | 5 × 1,280 each | qwen3-1.7b, qwen3-8b | 8 (n, p) cells at fixed mean degree × {none, clustering} × 400 |
| `density40` | 3 × 800 | qwen3-1.7b | {node_degree, connected_nodes} × 6 p (.05–.75) × 100 × {none, degree} |

## Task 1: One scorer, `score_density_sweep.py --group`

**Files:** Modify `scripts/score_density_sweep.py`; tests `tests/test_score_density_sweep.py` (flip the two drop tests, add grouping and outcome tests; fold in `tests/test_score_fixed_degree_sweep.py`'s cell tests).

- [ ] Tests first: (a) a capped row is kept, counted truncated, and not correct in the paired arms; (b) `--group cell` keeps size80/p0.101 and size160/p0.101 apart; (c) `--group task` keeps node_degree and connected_nodes apart; (d) connected_nodes is scored by exact set match; (e) the effect line prints the truncated change; (f) a flagged cell is excluded from the pooled test and the trend and listed.
- [ ] Implement: `key_of(record, group)` → density | (size, density) | (task, density); `summarize()` stores per record `outcome.outcome(exact, hit_cap)`; cells report correct/wrong/truncated shares, MAE over finished; `paired[key][instance][condition] = (correct, truncated)`; pooled and per-level effects print `f1(Δcorrect)`, CI (bootstrap over instances within level), McNemar p, BH q, Δtruncated; flagged cells skipped in pooled/trend and printed under a "flagged" line; `--levels` restricts the pooled range (e.g. `--levels 0.1,0.2,0.35,0.5`); `--tag NAME` prints `[NAME] <description>` as the first line; `--versus GLOB` pairs this run set's rows with another run set's rows on (instance_id, condition) and reports the correct-share difference (for think − plain).
- [ ] Remove the MDE simulation (unused by any current claim; `[power]`-style analytic MDE is printed per pooled row instead).
- [ ] `score_fixed_degree_sweep.py`'s extra quantities (mean-degree blocks, the degceil "headroom captured" line, and `c_confound`'s mean gold per cell) move in as `--group cell` options.

## Task 2: `analyze_rq3_leads.py` under R1

**Files:** Modify `scripts/analyze_rq3_leads.py`, `tests/test_analyze_rq3_leads.py`.

- [ ] Keep capped pairs (L94-95) scored not correct; bootstrap stratified by density with the truncated change (L100-110); print text blocks under tags (`[rqsel]`, `[rqhet]`, `[rqerr]`, `[rqbeh]`) instead of only JSON; drop `replication` (owned by `[replic]`); rows filtered to finished only where they describe answers (L257, L330, L360), with n printed.

## Task 3: The driver and its output

**Files:** Create `scripts/density_followups.py`, `csv2/density-followups/density_followups.txt`.

- [ ] One function per tag, each calling the scorer on one run set (memory: one at a time): `[dd40]` degdens40; `[dd40hi]` degdens40hi; `[ddfill]` degdens40+hi vs `degdensfill` with `--control filler`; `[ddrep]` degdensrep; `[ddthink]` degdensthink, pooled p≤.50 and p≥.65; `[ddthinkfill]` degdensthink vs degdensfillT; `[ddgap]` think − plain on none and degree (degdensthink vs degdens40+hi); `[ddceil]` degceil headroom; `[fixdeg17]`, `[fixdeg8]` degfixdeg per cell; `[d40]` density40 by task; the rq3 tags; `[ddtokens]` median/mean/max response tokens of finished responses per run set and condition.
- [ ] Run it; commit the output.

## Task 4: Checkpoint against every doc claim

- [ ] For each claim listed in the Phase 2 mapping (docs/rq3-leads.md, primer-impact-and-truncation-density-n40.md, node_degree-density-and-size.md, handoff-structural-sweep.md, primer-effects-and-power.md's family sections, README.md, graph-corpus-status.md, graph-design-requirements.md, plans/rq3-gpu-tests.md), find the tag that prints it or mark it not carried; list moved and reversed values; resolve conflicts 1–13 by the rerun. Stop and show the user every reversed claim and every claim no code can reproduce.

## Task 5: `docs/results/density-followups.md` and the index

- [ ] Header (Source lines, runs, reproduce command), sections: design; `clustering` on node_degree (linking to n40-sweep.md for `[replic]`); `degree` at high density; the thinking arm; filler controls; fixed mean degree; density40 pilot; forensics; limits. Every number tagged. Index row added.

## Task 6: Supersede and fix pointers

- [ ] `git mv` to `superseded/`: `scripts/score_fixed_degree_sweep.py`, `scripts/score_full_density_sweep.py`, `scripts/analyze_headline_robustness.py`, `scripts/candidates/c_confound.py`, `scripts/candidates/f_clustering_size.py` and their tests; docs `rq3-leads.md`, `primer-impact-and-truncation-density-n40.md`, `node_degree-density-and-size.md`, `handoff-structural-sweep.md`; outputs `csv2/degdens-probes/*`, `analysis/tables/degfixdeg.*`, `analysis/rerun/degfixdeg.qwen3-8b.txt`, `csv2/one-offs/qwen3-1.7b.density40.bytask.csv`, `rq3_leads.json`. `analyze_baseline_law.py`'s `continuum` test: moves to the scorer (`[ddthink]` continuum r); the live file keeps only helpers.
- [ ] Rewrite references (the Phase 1 rewriter, excluding plan files and `superseded/README.md`); fix `plans/rq3-gpu-tests.md` (+5.9), `README.md` L11-20, `graph-corpus-status.md` L25, `graph-design-requirements.md` L26-28, `repo-scope.md`, `CLAUDE.md` (score_density_sweep description), `runs/README.md` (degceil, degfixdeg rows); add Phase 2 rows to `superseded/README.md`.
- [ ] Prove the moved material reproduces (as Phase 1 Task 9).

## Task 7: Verify until clean, commit

- [ ] Fresh independent agents re-derive every number in the new doc from `runs/` with their own code, check paths and nothing-lost, run the tests; fix and repeat until a round finds nothing; one commit.
