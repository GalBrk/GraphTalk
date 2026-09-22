# Consolidating `talk_like_a_graph.v2.tex` with the raw-trends rebuild

Written 2026-09-22. Audits `paper/talk_like_a_graph.v2.tex` against
`csv2/raw-trends/frame.csv` — the 84,000-row frame rebuilt directly from
`runs/*.jsonl` by `scripts/build_raw_frame.py`, independently of every existing
analysis script — and proposes how the two become one paper.

This supersedes nothing. `docs/paper-claim-audit.md` audits **v1** against the
repo's own artifacts; this file audits **v2** against a from-scratch rebuild of
the main sweep. Where they overlap it is noted.

**Nothing in `paper/` has been changed.** These are findings and a proposal.

---

## 0. The headline: the generated tables are clean

`tab:main` was regenerated from raw `runs/` — same pairing key, same
drop-truncated-pairwise policy — and **every one of its 24 × 7 entries matches
to the printed digit**, with three cells differing by 0.1 through rounding.

That is the same pattern the v1 audit found: the machine-written tables are
right, and the defects are in prose that drifted from them. So the consolidation
problem is not "the numbers are stale". It is that v2 answers a different set of
questions from the ones the rebuild chased, and five prose sentences need
correcting on the way through.

Reproduce with:

```bash
python scripts/raw_trends.py --question effects
```

---

## 1. Claims that hold exactly

Re-derived from the frame, cell by cell.

| v2 claim | Where | Rebuilt |
|---|---|---|
| `filler` on `qwen3-1.7b`/`edge_existence` runs −5, −20, −32, −6, −2, −1, −1 from p=.10 to .85 | §Length | identical, all seven densities |
| `node_count`, `qwen3-1.7b`: 0.315 accuracy; 68.5% of answers read the "nodes 0,…,39" preamble as the count | §Difficulty | 0.315 / 0.685 |
| `components` on `connected_nodes`, `qwen3-4b`: +9.2 over `none`, +21.8 over `filler`; `filler` itself costs 12.5 | §Cost | +9.2 / +21.8 / −12.5, n=400 |
| `edge_count` under `none`: 4b exact on 8/400 with median relative error 10.9%; 1.7b exact on 5 with 28.5%; thinking arms exact on 86% and 98% of terminating generations, truncating on 82% and 80% | §Difficulty | all six figures exact (28.5% is on terminating rows, which the sentence should say) |
| `connected_nodes` exact match falls 0.920 → 0.480 across the density range while set-F1 stays flat | §Difficulty | 0.92 / 0.91 / 0.59 / 0.48 against F1 0.970 / 0.987 / 0.969 / 0.963 |
| The length cost changes sign *within* every task | §Cost | holds for every task except `node_count`, which runs +0 to +71 and is one-sided |
| `qwen3-4b-think` has no headroom; the other three arms do | §Main | holds under the table's stated drop-truncated policy |

One note on the last row. Under the drop policy `cycle_check` is 0.997–1.000 for
all four arms, as v2 says. Scoring truncated generations as wrong instead,
`qwen3-1.7b` is at **0.923** — 6.1% of its `cycle_check` generations never
terminate on a task whose answer is always "Yes". The claim is correct as scoped;
the scoping is doing real work and should be visible where the claim is made.

---

## 2. Five numbers that are wrong

Each verified against the frame. All five are prose; none is in a table.

| # | v2 says | Actually | Fix |
|---|---|---|---|
| 1 | Set-F1 on `connected_nodes` is "$0.96$–$1.00$ in every cell" | range is **0.927–1.000**; 17 of 112 cells fall below 0.96 (lowest: `qwen3-1.7b-think`/`rwse`/p=.50 at 0.927) | bound becomes 0.93–1.00; the substance — F1 does not discriminate between conditions — survives |
| 2 | Outside the two flagged cells "every task truncates on fewer than $7\%$ of instances" | `qwen3-1.7b`/`edge_count` is **22.4%** | name the third cell, or raise the bound and flag it in `tab:main` |
| 3 | `edge_existence` "hit rate stays at $99$–$100\%$ under every condition" | `qwen3-1.7b`/`rwse` is **98.1%** on the main sweep | bound becomes 98–100% |
| 4 | "`qwen3-1.7b-think` is at $0.642$ on `node_degree`" | **0.768** | v2's own Table 1 prints 76.8 one paragraph earlier; the companion figure (0.797 on `connected_nodes`) is right |
| 5 | "On `edge_count` under `none` the plain arms almost never truncate" | true for 4b (0%), false for 1.7b (**30%** under `none`) | restrict the sentence to 4b, or state both rates |

Item 2 is the one defect that carried over from the v1 audit unchanged.

---

## 3. Claims the rebuild cannot check

Not contradicted — the rebuild does not have what they need. Listed so they are
not mistaken for verified.

- **`tab:lengthcost`(a)** is computed over an undocumented "mid-range cells"
  subset (11 / 7 / 11 / 1 cells). My all-cell per-task means are −11.4 / −9.4 /
  −0.5 / +24.5 against its −9.1 / −6.3 / −4.3 / +31.0. The qualitative claims the
  table supports — task-specific, sign-changing, no per-token rate — all
  reproduce. **Action:** state the filter in the caption, or recompute on all
  cells.
- **"|content gain| correlates $+0.46$ with headroom against $-0.21$ for
  density"** over 126 cells, partialling each on the other. My 66-cell *simple*
  correlations give **+0.31** and **+0.25** — density comes out positive.
  Partial-versus-simple plus a different cell set will explain most of this, but
  a sign flip on the density term should not be carried forward unexamined.
  **Action:** recheck before the sentence ships.
- **The cross-fitted $-11.0$ interaction, the MDEs, the shortcut bars, the
  rewiring corpus and `degfixdeg`** run on other corpora and other machinery.
  Outside what this rebuild touches. Leave them as they are.

---

## 4. What the rebuild adds

Eight results, none of which appears in v2 in any form. These are the spine of
the consolidated paper. Full derivations in `docs/raw-trends-large-graph.md`.

1. **`edge_existence` accuracy above p=.50 is the class prior, not skill.**
   `qwen3-1.7b` raw accuracy "recovers" 54 → 68 → 77 → **86%** from p=.50 to
   .85, while balanced accuracy sits at **52.9 / 52.1 / 53.3** — chance — and the
   model answers "yes" to **98–99%** of pairs. v2 has the qualitative half
   ("gone where the task collapses toward a constant answer") but never says the
   recovery is the prior. This is the single largest reading change.
2. **The primer switches the algorithm, and the switch is countable.**
   `edge_count`, `qwen3-1.7b`, p=.50: responses stating "sum the degrees, divide
   by two" go **7% → 100%** under `degree`, tokens 4036 → 1185, truncation
   29.8% → 9.5%. At p=.10 it is 67% → 99%.
3. **Where accuracy drops, reasoning length collapses.** `node_degree`,
   `qwen3-4b`, p=.50: 273 → **34** median tokens under `degree`, accuracy
   99 → 87, on a primer that states the answer verbatim.
4. **Serial position in the primer**, with the `none` and `filler` controls that
   make it interpretable. `node_degree` under `degree`, pooled p=.35–.85, by the
   target's line number: `qwen3-4b` 82.3 / 72.5 / 71.8 / **69.6** (primacy
   decay); `qwen3-1.7b` 31.2 / **18.3** / 19.4 / 24.3 (lost-in-the-middle).
5. **Capitulation** — response length collapsing together with majority-class
   guessing. The `(n, k)` ladder rules out both length and difficulty as the
   trigger; what enables it is the answer format.
6. **`all` is sub-additive, by roughly half.** `qwen3-4b`/`edge_count`: parts sum
   to +33, `all` gives +9. `qwen3-1.7b`/`edge_existence`: +50 against +26.
   `qwen3-4b`/`edge_existence`: −35 against −17. Gains *and* damage are both
   halved, which points at dilution rather than interference.
7. **Effects are one-directional.** Fix/break decomposition of every Δ:
   `degree` on `edge_count` fixes 33 and breaks 0; `rwse` on `edge_existence`
   fixes 0 and breaks 23. `clustering` is the only condition with two-way churn
   (8 fixed against 4 broken), so its small net is a different kind of effect,
   not a weak version of the same one.
8. **Relevance matching, and its counterexample.** Matched primer/task pairs
   give `degree`/`edge_count` +33, `components`/`connected_nodes` +9,
   `clustering`/`edge_existence` +4 — and `degree`/`node_degree` **−12**, where
   the primer states the answer outright. Maximal relevance, negative effect.

Two supporting results worth a sentence each:

- **Models do not ignore irrelevant primer content** — an irrelevant primer
  disturbs accuracy by 6.4–14.9 points — but it disturbs no more than a
  length-matched `filler` does. Irrelevance costs nothing beyond length.
- **The `node_count` off-by-one**: 98% of `none` errors answer 39 for a
  40-node graph. `clustering` rescues it to 95% correct, `degree` does not (3%).
  Reported as an observed pattern; no mechanism is claimed.

---

## 5. Proposed structure

Keep v2's controls apparatus, replace its results spine. v2's RQ2 — added text is
an active treatment — is the one headline the rebuild independently confirms, so
it survives as-is and carries over as a control for everything in §4.

| v2 material | Disposition |
|---|---|
| Design figure, four controls, shortcut solvers, answer-extraction validity | **Body, unchanged.** This is the paper's methodological contribution and nothing in the rebuild touches it. |
| RQ2: length is an active treatment (§Length) | **Body, unchanged.** Independently confirmed; becomes the control for every §4 result. |
| `tab:main` | **Body, unchanged.** Reproduces exactly. |
| "What makes these tasks hard" (§Difficulty) | **Body, merged** with the rebuild's difficulty results — same question, more evidence. Absorbs new results 1 and 8. |
| RQ1: route substitution, cross-fitted interaction | **Body, compressed.** Correct, but the behavioural evidence (new results 2–4) now explains the *same* phenomenon directly, so the interaction becomes support rather than headline. |
| RQ3: `clustering` on `node_degree` | **Body, compressed to one paragraph.** Still the one replicated non-answer content effect; the mechanism question is now answered differently by new result 4. |
| Rewiring, `degfixdeg`, screened arms, cross-fit figure | **Appendix, unchanged.** All correct; none on the new spine. |
| `tab:shortcut` (body copy) | **Cut.** 30 of its 42 cells are never mentioned and the body uses two facts from it, both already written as sentences. The appendix copy stays. |
| `tab:headline` rows 2–5 | **Cut.** Re-aggregations of `tab:headline-density`. |

New body sections, in order:

1. **What makes a task hard** — output operation with the graph held fixed, the
   generation budget, the class prior (new result 1), the metric.
2. **Which primer moves which task** — `tab:main` plus the per-task comparison
   tables against both `none` and `filler`, with the fix/break decomposition
   (new result 7) and relevance matching (new result 8).
3. **What the primer changes in the response** — the algorithm switch, the
   length collapse, capitulation (new results 2, 3, 5).
4. **Where the primer is read from** — serial position with its controls
   (new result 4).
5. **Composition** — sub-additivity of `all` (new result 6).

Recovering roughly the page budget this needs: the two cuts above, plus
`[htbp]` instead of `[H]` on the appendix floats.

---

## 6. One methods sentence to add

Not a correction — a reader otherwise cannot reconcile the tables with any other
reporting of these tasks:

> `connected_nodes` is scored on exact match rather than set-F1, and
> `edge_existence` on balanced accuracy rather than raw accuracy. Both choices
> change the numbers substantially: F1 sits at 0.93–1.00 in every cell while
> exact match spans 0.48–0.92, and raw accuracy on `edge_existence` rises with
> density while balanced accuracy falls to chance.

---

## 7. Build issues carried over from the v1 audit

Still present in the v2 tree, all cosmetic:

- `ci_table.tex` has no `\label`; four orphan appendix floats referenced with no
  `\ref`.
- Duplicate `\label` on the thinking section (`sec:results-thinking` +
  `sec:results-power`); the second is never referenced.
- `\resizebox{\columnwidth}{!}` is on some tables and not others, so effective
  font size varies table to table.
- `paper/NUMBERS.md` maps numbers to sections that no longer exist;
  `paper/README.md` still says "8-page body".
- `paper/cited2.txt` is scratch output; nothing reads it.
