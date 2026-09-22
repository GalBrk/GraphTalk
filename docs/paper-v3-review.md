# v3 review log

Tracking file for the 5-page rebuild. Target: body (Introduction through
Conclusion, excluding references and appendix) at 5 pages, organised around
which primers help and harm — for which task, which arm, on what terms, and
where the limit is.

Each round: review, list issues, fix, re-review. A round closes only when the
verifier block is green and no open issues remain.

## Verifier block

| check | command | target | round 2 |
|---|---|---|---|
| builds | `bash paper/make_v3.sh` | exit 0 | pass |
| body length | `sec:limitations` in `.aux` | page 5 or lower | **p5** |
| undefined refs | `grep -c 'LaTeX Warning: Reference'` | 0 | **0** |
| overfull boxes | final pass of the log | 0 | **0** |
| numbers traceable | every prose figure maps to a script + CSV | no orphans | round 3 |
| tests | `pytest -q tests/test_raw_frame.py tests/test_scoring.py` | pass | **19 passed** |

The page gate checks `sec:limitations`, not `sec:conclusion`: the body ends
where Limitations begins, and grepping where the Conclusion *starts* passes
while the Conclusion itself spills onto the next page.

## Triviality audit

The paper must carry findings a reviewer could not have predicted.

**Expected — machinery, never a headline**

| finding | why it is not a result |
|---|---|
| Primers help models with headroom, hurt models at ceiling | a ceiling effect; what earns its place is the *within-arm* sign flip and the leaky/non-leaky split, not the direction |
| Added text costs accuracy | it does not, here — see below — so the expected claim is itself refuted |
| Set-F1 hides the ceiling on `connected_nodes` | a measurement choice; one clause |
| Bigger model scores higher | not reported |

**Non-trivial — these carry the paper**

| finding | why it is surprising |
|---|---|
| A primer that **states the answer verbatim** costs `qwen3-4b` 12 points at p=.50, breaking 13 items and fixing 1 | the information is present and correct |
| The **same arm, task and primer** gain +27 at p=.85 and lose 12 at p=.50, tracking baseline monotonically | the effect belongs to the cell, not the primer |
| **Length is nearly free** (−0.3 pts per 1,000 chars, positive in 8 of 16 cells) while a degenerate preamble costs up to 15.7 | contradicts the expected long-prompt penalty |
| The primer **replaces the algorithm**: sum-of-degrees stated in 7% → 100% of responses | visible in the text, not inferred from scores |
| **Leakage is incomplete**: where the graph-blind solver scores 1.00, models recover 0.006 to 0.99 | containing the answer is not giving it |
| Raw accuracy **rises to 86% while balanced accuracy sits at chance** | the benchmark rewards giving up |
| **53% of side-information cells** sit above 0.90 and cannot show a gain | most of a standard sweep is uninformative |
| `all` ≈ its **largest single part** (1.00 [0.74, 1.20]), not a diluted sum | bundling buys one primer's worth of three |

---

## Round 1 — closed

| # | issue | resolution |
|---|---|---|
| 1 | Body 9 pages, target 5 | body rewritten; ends p5 |
| 2 | Related Work a full section | folded into the Introduction |
| 3 | Method had five subsections | two: controls, and corpus/measurement |
| 4 | Headroom crossover absent and unscripted | `raw_trends.py --question headroom` → `headroom.csv`; superseded in the body by the leaky/non-leaky split |
| 5 | `cycle_check` 90.7 in the budget comparison | section cut in the rewrite |
| 6 | `clustering` +5.9 cited without the main-sweep figure | section cut in the rewrite |
| 7 | Floor/ceiling null-cell claim unreproducible | cut |
| 8 | Paper led on the headroom law | leads on the answer-stating paradox |

## Round 2 — closed

Triggered by reconciling the body against `docs/paper-v3-review-fixes.md` (130
verified findings) and against the abstract, which had been revised to
stricter claims than the body carried.

| # | issue | resolution |
|---|---|---|
| 9 | **Decoding budget wrong by 4×.** Paper said 2,048 for plain arms; the raw generations cap at **8,192** on `densfull40` and 2,048 only on `densfull40hi` | corrected, and the non-comparability of truncation across the two bands is stated |
| 10 | **`filler` called "content-free"** — it states the vertex set in 40 id-ordered lines and a solver reading it alone scores 1.00 on `node_count` | renamed a *degenerate-content preamble of comparable length to `clustering`*; no longer the reference a content effect is defined against |
| 11 | **Length cost misattributed.** Fitted within (arm, task) the length term is −0.3 pts/1,000 chars, positive in 8 of 16 cells | claim replaced; the length-cost table, which is orphaned and refuted, dropped |
| 12 | **Serial position overclaimed.** Only 6 of 48 gradients differ from their own control; `none` itself slopes +25.0 on one cell | rewritten as unattributed; `_slope_diff` added to `raw_trends.py` so the count is reproducible |
| 13 | Additivity quoted 0.44 (7 densities) against a 4-density table | 0.50 [0.29, 0.71]; largest-part ratio 1.00 [0.74, 1.20] on the 19 discriminating cells |
| 14 | `158 of 289` cells above 0.90 | **172 of 324 (53%)** per `analyze_primer_window.py` |
| 15 | Answer-carrying primers "+11 to +13" | **+16** in the 0.25–0.75 bands |
| 16 | "next bar below 0.85 is 0.21" | **0.746**; the bars are bimodal, 18 at exactly 1.000 |
| 17 | Parse residual "confined to truncated `cycle_check`" | 128 of 133 are `cycle_check`; 5 are `edge_existence` |
| 18 | `v3_window_table` orphaned; `v3_headroom_table` unused | window table now carries §4.1; headroom table moved to the appendix |
| 19 | `v3_window_table` input twice | appendix duplicate removed |
| 20 | Three appendix tables overran the column (10 overfull boxes) | wrapped in `\resizebox` |

## Open — round 3

| # | issue |
|---|---|
| 21 | Traceability sweep: every numeral in the body must map to a script and a CSV in `paper/NUMBERS.md`; not yet done for the window, recovery and length-term figures |
| 22 | `docs/paper-v3-review-fixes.md` P1 items 2.6–2.16 not yet triaged against the rewritten body (several may no longer apply, since the sections they target were cut) |
