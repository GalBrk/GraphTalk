# Two versions of the paper

| File | Pages | What it is |
|---|---|---|
| `talk_like_a_graph.tex` | 8 (body ends p.5) | The 5-page submission version. |
| `talk_like_a_graph_long.tex` | 17 (body ends p.12) | Everything, for you to cut down. |

Both compile with `latexmk -pdf`, both are clean: exit 0, zero overfull boxes, no
undefined references or citations. They share `acl.sty`, `acl_natbib.bst` and
`custom.bib`. Neither is anonymized (see the bottom of this file).

Three files in `paper/` are **generated, not hand-written**, so an appendix number
cannot drift from the run it came from. Regenerate from the repo root:

```bash
PYTHONPATH=. python paper/make_figure.py      # -> paper/density.pdf
PYTHONPATH=. python paper/make_tables.py      # -> paper/appendix_tables.tex
PYTHONPATH=. python paper/make_ci_table.py    # -> paper/ci_table.tex  (needs ci_all.json)
```

`ci_all.json` comes from `ci_all.py` in the repo root (~20 min: 144 cells ×
10,000 permutations × 10,000 bootstrap resamples).

---

## A. Restored — cut from the 5-page version purely for length

Bring any of these back by copying from the long file.

| Item | Where in the long version | Why it earns its place |
|---|---|---|
| **Truncation table** (`tab:hitcap`) | §5.1 | Gates the interpretation of `edge_count` everywhere. |
| **Density table** (`tab:density`) | §5.4 | The numbers behind the sign reversal. |
| **Primer-length table** (`tab:lengths`) | §3.1 | Makes the `filler` argument checkable. |
| **Chain-of-thought table** (`tab:cot`) | §5.6 | The largest effect in the study. |
| **Headline table** (`tab:headline`) | §5.7 | The one positive result. |
| **Thinking arms in the main table** | `tab:main-results` | The 5-page version splits them to an appendix. |
| Separate `\section{Conclusion}` | §7 | Merged into the Discussion at 5 pages. |
| Generator-family sweep withdrawal | §2, ¶2 | A real finding: the majority baseline swings 0.15→1.0, six times any primer effect. It explains why the design fixes the generator. |
| Seven-condition itemized list | §3.1 | Clearer than the run-on sentence the short version uses. |
| Results subsection headings | §5.5–5.7 | Demoted to `\paragraph` at 5 pages. |

## B. New — never in either earlier version

These are the ones I would actually argue for keeping if you can afford any of them.

1. **Shortcut-ceiling table** (`tab:shortcut`, §3.2). Contribution #2 had *no table* in
   any prior version — the ceilings existed only as prose asides. This is the 7×6 grid,
   with the cells a primer fully determines in bold. It also carries the point that a
   1.00 ceiling is what a *Python program* scores, not a model: `node_count` is a 1.00
   cell and Fatemi et al. report 18.8% for PaLM 2 on it.
2. **Blind-baseline table** (`tab:blind`, §4). Accuracy of always answering the modal
   gold value, per task per density. This is what makes "`node_count` and `cycle_check`
   measure nothing at n=40" a number rather than an assertion — both are exactly 1.000
   at every density.
3. **§3.5, Answer extraction and measurement validity.** The boolean-extraction bug, in
   full: models answered `"Yes, there is a cycle. A cycle is a path … with no repeated
   edges or nodes"` and the extractor took the trailing "no". I re-measured it this
   session — **225** of qwen3-4b's 5,568 scored boolean rows, all `cycle_check`, all
   wrong→right, none in the other direction. Before the fix the table showed a dramatic
   condition-dependent collapse (`filler` 0.140 at p=0.2); after it, nothing on that task
   is significant. Worth keeping because the failure mode generalizes: an extraction bug
   that correlates with response style is indistinguishable from a condition effect.
4. **Figure 1** (`fig:density`). node_degree accuracy vs density, all four arms, five
   conditions, with the blind baseline drawn in. Carries the ceiling/floor argument in
   one glance. The short version has no figure at all.
5. **Per-cell truncation table** (`tab:hitcap-detail`, §5.1). Shows the censoring is
   *condition-dependent*, not just severe — at p=0.5 `none` loses 100/100 rows for
   qwen3-4b-think while `degree` loses 60, so those two arms are scored on different
   graphs. That is a stronger reason to distrust the cell than "62–81% truncated".
6. **§5.8, sensitivity to the truncation convention** (`tab:sensitivity`). Drop-vs-zero
   flips the verdict on 1.7B `cycle_check`: `rwse` −7.5 (p<0.001) becomes −0.2 (p=1.0),
   while `clustering` +8.5 and `all` +7.0 appear. Reviewers ask about this; better to
   own it.
7. **Per-level headline breakdown** (`tab:headline`). All four density levels for both
   samples, not just the pooled row. The effect is positive at 8/8 levels — probability
   1/256 under a null — which is better evidence than the two pooled p-values alone.
8. **`components` density trend** (§5.7, last ¶). Inert pooled (+0.1, p=0.948) but the
   paired difference trends negative with density (slope −0.148, p=0.022). Worth a
   sentence because `components` is the only condition whose effect *cannot* be length.
9. **MAE evidence for the 4B `degree` harm** (§5.2). Mean absolute error rises 0.015 →
   0.138 under `degree`, which supports "the primer competes with the model's own
   computation" better than the raw miss count does.
10. **Appendix A** (`ci_table.tex`): Δ, 95% CI, permutation p, *and* the zeroed-convention
    Δ for **all 144 cells**. The short version reports intervals for 15.
11. **Appendix B** (`appendix_tables.tex`): exact-match accuracy for every (arm, task,
    density, primer) cell — 24 tables — with per-cell truncation counts.
12. **Longer Introduction and Discussion.** The intro now names the three-way confound
    (content / leakage / length) explicitly; the discussion adds the claim that a primer
    competes with rather than replaces the model's own computation, and the
    recommendation that primer gains reported against a no-primer baseline alone be
    assumed to be length effects until a placebo says otherwise.

## C. Still not in either version

- **The parked material** — difficulty ladder, retrieval probe, degree-preserving
  rewiring, Game-of-Thrones naming. Still in `PARKED.md`, still unrun, and named
  nowhere in either `.tex`. Both Limitations sections end on a generic "a design
  holding rendered length exactly constant would isolate content more sharply",
  which is ordinary future work and does not describe the parked machinery.
- **Anonymization.** Both files carry real names and GitHub handles and
  `\usepackage[final]{acl}`. For a double-blind venue: switch to `[review]` and replace
  the author block.
