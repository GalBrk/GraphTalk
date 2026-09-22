# Paper claim audit — notes to check

Audit of `paper/talk_like_a_graph.tex` against the repo's own artifacts and
docs, done 2026-09-20/21. **Nothing in the paper was changed.** These are
findings to verify, not decisions — each one names the evidence and where to
re-check it, so you can disagree with any of them on the data.

Read §2 ("what turned out to be wrong") before acting on anything else: two
findings that looked important did not survive scrutiny.

---

## 1. Verified correct — don't re-audit these

Re-derived from artifacts, cell by cell. All match the paper to the printed
digit:

- The `-11.0` interaction and **all seven rows** of `tab:robust`
  (`review_checks.json["interaction"]`, `["logit"]`).
- **All 42 cells** of `tab:shortcut` and all 42 of `tab:shortcut-published`
  (`shortcuts_n40_flat.json`, `shortcuts.json`).
- The route threshold claim: largest sub-threshold gain is `+0.01048`
  (`edge_existence/degree`), smallest supra-threshold `+0.06619`
  (`node_degree/rwse`) — so "at most 0.011 or at least 0.066" is exact, and
  the route set is exactly {`node_degree`: rwse, degree, all} plus
  {`edge_count`: clustering, degree, all}.
- The whole clustering block: `+1.9` pooled over 7 densities (p=0.0224,
  n=2800); `+5.9 [2.1, 9.8]` pooled over p=.35+.50; the `+5.5` replication;
  `+3.0`/p=0.26 on the main sweep. **This looked inconsistent and is not** —
  `tab:headline`'s `+4.2` row is the four-density window and its own density
  column says so, so it coexists with the `+5.5` two-density figure.
- `tab:prior`, `tab:budget`, `tab:sdt`, `tab:mae`, `tab:booleanbias`,
  `tab:baseline`, `tab:hitcap` — every cell.
- The primer token table (8/372/470/510/1,070/1,612 against a 2,165-token
  `none` prompt) and the "within 8%" length match (7.84%).
- 16,800 prompts per arm (7×6×4×100; each
  `analysis/{arm}.densfull40.rows.csv` has exactly 16,800 rows).
- "+3.1 to +34.2 points" — in fact strictly monotone, so if anything
  understated.
- The `filler`/`edge_existence` rewording caveat in `DATA.md` was checked and
  **does not** undermine the length-control result: `prompts.densfull40.jsonl`
  uses the corrected filler ("Node 0 is simply present within the graph G.")
  and the revised question. The n=40 results are post-correction.

Pattern worth noting: **every defect found is in the prose, none in the
generated tables.** The tables are machine-written and clean; the prose
drifted from them.

---

## 2. What turned out to be WRONG in this audit

Recorded so a new session doesn't repeat them.

### 2a. "Reframe RQ3 from density to baseline difficulty" — failed

Proposed on the strength of `docs/primer-effects-and-power.md` §5a, then
checked by two agents given neutral prompts. Density **is** ruled out by the
fixed-mean-degree dissociation. But the proposed replacement is not
supported: the inverted-U in baseline holds in one arm (r=+0.78) and
**reverses in `qwen3-8b` (r=−0.94)**, and the repo's own 177-cell test over
no-route primers gives **r=+0.041, p=0.59**. There are two significant sign
flips at matched baseline (`qwen3-4b` p=.75, baseline 0.500 → +9.0;
`qwen3-1.7b-think` p=.75, baseline 0.469 → −8.8).

Also relevant: the paper's prose **already hedges correctly**
("per-density estimates are too imprecise to localise the effect more
finely"), so it was not overclaiming in the way first alleged.

### 2b. "Report the 8B reversal" — dropped

`runs/qwen3-8b.degfixdeg.*` gives −1.0 pooled where `qwen3-1.7b` gives +6.2
on the same grid. This looked damaging. It is not, per
`scripts/score_fixed_degree_sweep.py`:

| cell | 1.7b `none` | 8b `none` | 1.7b Δ | 8b Δ |
|---|---|---|---|---|
| n=20 d≈8 | 0.912 | 0.998 | −4.8 * | +0.3 |
| n=40 d≈8 | 0.728 | 0.995 | +4.3 | +0.0 |
| n=80 d≈8 | 0.645 | 0.965 | +10.5 * | −0.8 |
| n=160 d≈8 | 0.624 | 0.897 | +6.9 * | −2.8 |
| n=160 d≈16 | 0.224 | 0.802 | −1.3 | −4.0 |

1. **`qwen3-8b` is at ceiling** (0.998 / 0.995 / 0.965) in most cells —
   nothing for a primer to add. Its only two headroom cells go −2.8 and −4.0
   and **neither is significant** (p=.19, p=.14). That is the paper's own RQ1
   finding playing out, not a contradiction of RQ3.
2. **The corpus has no `filler` arm**, so it cannot test the paper's actual
   claim (that `clustering` beats a length control) at all.
3. **`degfixdeg` appears nowhere in the paper**, for any arm — so reporting
   it means importing a new corpus to rebut a claim the paper doesn't make.

`degfixdeg` design, for reference: holds mean degree fixed (≈8 and ≈16) and
varies n (20/40/80/160), so density falls while edge count and prompt length
rise. One task (`node_degree`), two conditions (`none`, `clustering`), 400
graphs per cell, 6,400 rows per arm. Built by
`build_size_sweep.py --densities`; arms `qwen3-1.7b` (Sept 10) and
`qwen3-8b` (Sept 19).

---

## 3. Claims contradicted by the repo's own data

Each verified. `tex` line numbers are approximate.

| Where | The claim | What the data say |
|---|---|---|
| §5.4, ~454 | "`clustering` on `node_degree` is **the only** primer that does not state the answer and still exceeds both controls" | `vs_controls_densfull40.json` (the paper's **own main sweep**): `qwen3-4b`/`connected_nodes`/`components` beats `none` +9.2 and `filler` +21.8, both `bh_global_reject: true`, and is non-route. Never mentioned anywhere in the paper. `qwen3-1.7b`/`edge_existence`/`all` also qualifies (+11.5/+27.2, replicating +6.7/+8.0 on the hi band) — §5.2's criterion-shift argument covers `edge_existence` but not `connected_nodes`. |
| §5.4, ~473 | "the thinking arms do not show it at all" | `docs/paper-revision-handoff.md` §E lists this exact wording under **Retractions**: *"'clustering does not survive thinking' corrected: think +2.3 (p=0.036) at p≤.5, and +5.0 length-controlled."* Same corpus. An earlier revision said it, it was corrected, and it is back. |
| §4 footnote | Eleven further arms "were used for **no claim reported here**" | Three body claims use them: the held-out row of `tab:robust`; `qwen35-2b` in the rewiring (§3.3, `tab:prior`, `fig:rewiring`); the Game-of-Thrones renaming check. `paper-revision-handoff.md` §2 suggests relabelling them "earlier-collected, not pre-registered". |
| §5.1, ~370 | "every task truncates on fewer than 7% of instances" outside the two flagged cells | `tab:hitcap` in the same paper: `qwen3-1.7b`/`edge_count` is **22.4%** and carries no flag (the flag is defined as ≥62%). Next-highest unflagged is 6.5%. |
| §5.2, ~378 | filler on `node_degree` "harmless up to p=.35, then costs between 9 and 14 points", r=+0.95, 6.3 pooled | These are **`qwen3-1.7b-think`** on the dedicated corpus. The sentence sits in a paragraph scoped to "both **plain** arms" on the main sweep, whose figures are −2.4 pooled. Also the densest level is −14.1 (outside "9 to 14") and p=.35 is −4.3 (called "harmless"). |
| §3.4, ~288 | "changes the verdict on **only** three `qwen3-1.7b` `cycle_check` cells" | Recomputing BH on `ci_all.json` `p_mcnemar` vs `p_mcnemar_zero`: **21 cells flip**, of which 3 are that arm/task. True as scoped, wrong as stated. |
| §5.3, ~447 | copy signature: "when **a model** errs despite a stated answer… (p=.002)" | Holds for `qwen3-4b`/`degree` only (12/29 = 41.4% vs 17.3% background). The sibling `all` on the same arm gives 2/35 = 5.7%; other arms sit at background. The generating script is not in the working tree but **is recoverable**: `git show 04895d9:scripts/candidates/b_copying.py`. |
| §5.4, ~478 | "for `qwen35-2b` `filler` matches or exceeds `clustering` at every level" | `rewiring_qwen35-2b.json`, rung `n40k12` (the only rung plotted): at `low`, clustering +5.56 vs filler +3.02. Both n.s., so "indistinguishable" holds; "matches or exceeds" does not. |
| Abstract | "**replicates** on eleven held-out arms" | That interaction's CI is [−109, −3.8]. §5.3 itself says "estimated far less precisely". Supports "same sign". |
| §5.2, ~387 | "hit rate stays at 99–100% under every condition" | `qwen3-1.7b`/`rwse` is 98.1%; `tab:booleanbias` omits that condition, so the table hides it. |
| §3.4, ~287 | "a manual check of 50 rows per arm" | `NUMBERS.md` traces this to `scripts/audit_extraction.py`, an **automated** check. The 50-row manual audit appears only as a planned step in `paper-revision-handoff.md` §4.0; no artifact records it. |
| Ethics | "approximately 200,000 generations" | Line count of `runs/*.jsonl` (archive excluded) = **217,178**. |
| §5.5, ~501 | "median detectable size is **5.6** points for the 1.7B arms" | Recomputed median over null rows with a detectable gain is **5.94** (n=30); no filter variant reproduces 5.6. The 4B figure (10.0) is exact. |

**Settled in discussion:** the first two rows (the "only primer" claim and
the reinstated retraction) were agreed as worth changing. The rest are
unreviewed.

---

## 4. New result: the RQ1 interaction holds per task

Run during the audit because the interaction pools only two tasks, one of
which (`edge_count`) truncates on 22–81% of generations — so a reviewer would
ask whether the headline is one task or a truncation artifact. It is neither.

| split | k | r | p | slope (95% CI) |
|---|---|---|---|---|
| all route cells (paper headline) | 232 | −0.308 | 1.8e−06 | −10.5 [−16.6, −5.4] |
| `node_degree` only | 168 | −0.235 | 0.0022 | −9.1 [−21.6, −1.8] |
| `edge_count` only | 64 | −0.664 | 2.2e−09 | −26.5 [−31.5, −20.5] |

`node_degree` truncates on 0–1.6% of generations and carries the relation on
its own, CI excluding zero. Pooled r reproduces the paper's −0.31 exactly, so
this is the paper's own pipeline partitioned, not a new analysis. Reproduce
with `analyze_baseline_law.py`'s `arm_cells_crossfit` filtered by task
(`score_run` already accepts `task_filter`).

---

## 5. Figures — caption vs. what is actually plotted

- **Figure 1 (`make_figure_f1.py`)** — caption is factually correct but omits
  five marks visibly in the panel: the solid line is an OLS fit over the
  *doubled* fold-direction points (232/762, not 116/381); the dotted line
  with white markers is a binned-mean overlay that is literally a redraw of
  appendix `tab:baseline`; the panel-title `r` is computed on 232 points but
  labelled "116 cells"; the inset repeats the `−11.0` that already appears in
  the abstract, the prose and `tab:robust`; and `node_count`/`cycle_check`
  are **excluded**, so it is a 4-task figure following a 6-task table.
- **Figure 1's ceiling sentence** — "so shared sampling noise cannot produce
  the slope" is true but does not address the ceiling bound: a cell at 0.99
  baseline can move at most +1 and at least −99, which is deterministic, not
  noise, and cross-fitting does not touch it. The paper *does* answer this
  (the logit and windowed rows of `tab:robust`); the caption points at the
  wrong defence.
- **Figure 3 (`make_figure_density.py`)** — the caption's "4× difference in y
  scale" measures **2.28×** from the PDF content stream (panel a ylim
  [−22.1, +43.1], panel b [−14.8, +13.8]); the 4× is the *effect* ratio from
  the script docstring, moved onto the wrong object, and the same sentence
  then says "an order of magnitude". Also **the two panels are two different
  models** (a: `qwen3-1.7b-think`, b: `qwen3-1.7b`), named only in the axes
  titles, so the caption's headline comparison is a cross-arm claim presented
  as within-figure. The "accuracy under `none`" secondary axis is in panel (a)
  only, and `sharex=True` invites reading it as figure-level.
- **Figure 4 (rewiring)** — plots accuracy while the claim it supports is a
  *difference*; `ylim` hard-coded to (0.4, 1.0) with no note; only the
  `n40k12` rung shown, though `qwen35-2b` also has `n60k12`/`n60k16`.

---

## 6. Redundancy and length

Body (Intro→Ethics) is **≈5.85 pages = 574 column-lines** in `acl.sty`
geometry. Floats are 154 of those, and **43 column-lines are caption text**.
So the length problem is mostly floats and captions restating body prose, not
the prose itself (body prose is only ~2,550 words).

- `tab:shortcut` (body) vs `tab:shortcut-published` (appendix): same 6×7
  shape, same headers; **30 of the body table's 42 cells are never mentioned**,
  and the body uses exactly two facts from it, both already written as
  sentences.
- `tab:baseline` (appendix) is what Figure 1's dotted line plots — same bin
  edges, by design. Between them they give three different correlations for
  "the same relation" (−0.417/−0.050, −0.31/+0.02, −0.35/−0.01) on three
  different cell counts (81/273, 116/381, 117/381), none reconciled.
- `tab:headline` rows 2–5 are re-aggregations of `tab:headline-density`. And
  **neither table contains the `+5.9` or `+5.5` the prose cites them for.**
- `tab:main`: 8 of its 24 rows are `node_count`/`cycle_check`, which have
  constant gold answers at n=40, which the paper itself flags as
  uninformative, and which are the source of its largest entries.
- `ci_table.tex` has **no `\label` at all** — four orphan appendix floats,
  referred to from §5.1 with no `\ref`. Its `zeroed` column is the only place
  the §3.4 truncation claim can be checked.

---

## 7. Build and hygiene

- **`talk_like_a_graph.log` is from an aborted run.** It stops at
  `\begin{document}` with `I can't write on file talk_like_a_graph.pdf` — a
  viewer held it locked. So the `.aux` is truncated to 3 lines, and the
  **zero overfull/underfull warnings mean nothing was typeset**, not that the
  paper is clean. `talk_like_a_graph.NEW.pdf` is a workaround build; the
  documented pipeline writes `talk_like_a_graph.pdf`. Worth settling which
  file is authoritative.
- **`talk_like_a_graph.bbl` is stale**: 25 `\bibitem` against the 23 in
  `NEW.pdf`. With natbib every `\bibitem` prints whether cited or not, so
  `guo2023gpt4graph` and `jin2024largelanguagemodelsongraphs` will reappear
  as phantom references if a rebuild skips bibtex.
- `custom.bib`: 25 entries, 23 cited, 0 missing. **`gemma4` and `qwen35` are
  placeholder-shaped** — `@misc`, dated 2026, no report/DOI, and `gemma4`'s
  URL is the HuggingFace *org landing page*, not either model card it names.
- Duplicate `\label` on §5.5 (`sec:results-thinking` + `sec:results-power`);
  the second has no `\ref`. Ten further dead labels; `tab:mae` is orphaned.
- Appendix floats are all `[H]`, which is what makes pages 9–10 ragged;
  `[htbp]` or `table*` would recover ~half a page.
- `\resizebox{\columnwidth}{!}` is on 10 tables and not on 6, so a table
  narrower than the column gets scaled *up* — effective font size varies
  table to table.
- **`paper/NUMBERS.md` has drifted**: it maps numbers to sections 5.6/5.7/5.8
  that no longer exist, cites prose that is gone, and points at
  `scripts/candidates/b_copying.py`, which is not in the working tree.
  `paper/README.md` still says "8-page body".

---

## 8. Data on disk that no document reports

- `runs/qwen3-8b.degfixdeg.*` — 6,400 rows. See §2b; largely ceiling-bound.
- `runs/qwen3-8b.retrieval.*` — 3,600 rows; the 1.7B twin is written up in
  `candidate-analyses.md`, this one is not.
- `runs/qwen3-1.7b.retrieval_threshold.*` — 1,800 rows, in no doc and not in
  `runs/README.md`.
- `runs/qwen35-2b.rewire_extra.jsonl` — 2,700 rows at rungs `n60k12`/`n60k16`;
  mentioned in `DATA.md`, analysed nowhere. Would widen the rewiring null
  past the single rung the paper plots.

---

## 9. Open questions

- Whether `tab:headline-density`'s per-density stars should stay labelled
  "McNemar p<0.05 (**uncorrected**)" while the paper BH-corrects everywhere
  else. Under Holm across that table's 14 tests the `+7.5` at p=.35 becomes
  0.107. The prose above the table already hedges correctly, so this is only
  a question of whether the table should match the paper's own standard.
- Whether a mechanism for the `clustering` effect can be stated at all.
  `docs/rq3-leads.md` proposes a retrieval/position account. Checked
  independently and it does **not** hold up: the favoured covariate loses to
  degree-z out of sample under the doc's own continuous model (pos −1.33
  p=.19 vs degree-z +3.00 p=.022); the "held-out" seed had already been used
  twice for other post-hoc choices, and git shows it sat in the repo nine
  days before the analysis that held it out; and `rwse` — which shares the
  proposed causal ingredient — shows the gradient *more* strongly and was
  never followed up. On the same corpus `rwse`−`none` is +4.01 against
  `clustering`'s +3.00. So §5.4's existing "Neither mechanism we can test
  explains the effect" looks right as written.
- `paper/cited2.txt` (23 lines, a list of cited bib keys) was left in the
  repo as scratch output during this audit. Nothing reads it; safe to delete.
