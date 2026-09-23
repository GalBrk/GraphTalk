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

## Round 3 — closed

| # | issue | resolution |
|---|---|---|
| 21 | Traceability sweep | `paper/NUMBERS.md` v3 section rewritten: every body figure maps to a CSV column and the command that writes it |
| 22 | Unified `4,706`/`4,691` for the same quantity, and `+11 to +13`/`+16` between abstract and body | both unified |
| 23 | **`components` is a constant string at p≥.20.** Every n=40 ER graph at p≥.20 is connected (verified: 0 of 100 have >1 component at .20/.35/.50; 52 of 100 at .10), yet the gain there (+11, +10, +9) exceeds the gain where the string varies (+7.0) | the paper now states that this gain is not graph-specific content, and links it to the `node_count` off-by-one that any interposed text interrupts |
| 24 | Additivity shortfall untested against a null; a ratio of noisy quantities is biased downward | matched-null simulation added to `print_additivity`: exact additivity plus the observed per-cell noise gives 1.00 [0.92, 1.08] against an observed 0.50, p<0.001 |

Review items 2.1–2.5, 2.10–2.16 of `docs/paper-v3-review-fixes.md` targeted
sections the rewrite removed, or were resolved by it: the −11.0 interaction is
gone, §5.4 is rebuilt on the tested difference, the leakage is named in the
abstract, and the shortcut bar is compared against model accuracy for the first
time.

## Round 4 — closed

Coherence read-through of the assembled body.

| # | issue | resolution |
|---|---|---|
| 25 | Conclusion still carried `+11 to +13` where abstract and body say `+16` | unified |
| 26 | §4.4 headed "Two effects that do not survive their controls", but bundling *does* survive its matched null | retitled "Position and composition" |
| 27 | §4.1 reports `components` at −0.3 while §4.3 reports it at +7…+9, which reads as a contradiction | forward reference added; the two are a mean over cells and one arm's format effect |

A cross-document sweep confirms one value per quantity: `+16` (4 sites),
`0.50`, `172`, `324`, `4,691` (3 sites), `6 of 48`, `0.746`; and no
occurrences of the superseded `+11 to 13`, `0.44` or `4,706`. The only
remaining instance of "content-free" is the sentence stating that `filler` is
not content-free.

## Round 5 — closed

Audit against the brief rather than against the prose: does the paper say
which primers help and harm, for which task *and which arm*, on what terms,
and where the limit is?

| # | issue | resolution |
|---|---|---|
| 28 | The body leaned on the two plain arms; the thinking arms appeared only in passing | per-arm paragraph added, with `print_per_arm` in `raw_trends.py` so it is reproducible |

The audit produced a result the paper did not have. **Thinking mode closes the
window**: median baseline rises 56.5 → 79.0 on the 1.7B checkpoint and
87.0 → 97.0 on the 4B one, leaving 60 of 110 cells in the 0.25–0.90 window for
`qwen3-1.7b` and only 15 for `qwen3-4b-think`. And **no primer is best twice**
— `all` leads on `qwen3-1.7b` (+9.8) and is worst on both 4B arms (−9.5,
−13.0); `degree` leads on `qwen3-1.7b-think`, `clustering` on `qwen3-4b`,
`components` on `qwen3-4b-think`. A primer chosen on one arm does not transfer.

Also checked: the body contains no process narration — no reference to
`docs/`, CSV files, scripts, review history or earlier versions — so it reads
as findings rather than as the path taken to them.

## Round 6 — closed

Audit of the appendix against the body: does every float support a claim the
paper still makes?

| # | issue | resolution |
|---|---|---|
| 29 | Nine appendix floats were orphaned, all supporting sections the rewrite removed (budget-matched thinking, MAE, signal detection, the `clustering` headline and its density table, rewiring, the density prior, the published-split bars, relevance matching) | dropped, with the `degfixdeg` and `published_ceiling` inputs |
| 30 | `tab:budget` carried two figures that do not reproduce: `qwen3-1.7b`/`cycle_check` read 90.7 against 92.2 from the frame, and `qwen3-4b` 99.0 against 99.8 | dropped with the table; no surviving float reports them |
| 31 | Eight surviving floats had no reference from the body | each now referenced from the claim it supports |
| 32 | `make_v3_tables.py` still generated `v3_relevance_table.tex`, which nothing inputs | generator call removed |

## Round 7 — closed

Run against the restructured paper (new title, ACL 2023 style, Related Work,
and results reorganised around the procedure account).

| # | issue | resolution |
|---|---|---|
| 33 | Page gate failed: it now requires Limitations and Ethics inside page 5 too, and Ethics ran to page 6 | Discussion and Ethics tightened without dropping content; both now end on page 5 |
| 34 | Citation integrity after the rewrite | clean: 24 cited, all present in `custom.bib`, `.bbl` carries exactly those 24, no phantom entries, 0 undefined-citation warnings |
| 35 | Two new abstract claims unverified | both confirmed, on the seven-density basis those two tasks carry: `qwen3-1.7b-think`/`node_degree`/`degree` **+20.1**; `qwen3-1.7b`/`edge_existence` false alarms **−18.6** under `all` and **+18.3** under `filler`. Checking them over the four shared densities gives +12.2/−15.7/+21.5, so the basis matters and the paper uses the right one |
| 36 | Reproducibility of the new floats | all six inputs (`v3_conditions_table`, `v3_main_table`, `v3_pertask_tables`, `v3_routes_table`, `v3_fig_window`, `v3_fig_edgeexist`) trace to `make_v3_tables.py` or `make_v3_figures.py`, and `make_v3.sh` runs both plus `primer_findings.py` |

### Round 7 — verification

Two verifiers compared the restructured paper with the version before it. Both
returned *approve with fixes*: better on every grading criterion, but three
statements contradicted by the data.

| # | issue | resolution |
|---|---|---|
| 37 | The Discussion's account (effect = accuracy of the adopted procedure minus that of the abandoned one) failed at p=.65: `retrieval` was any response that did not restate the neighbour list, so a response that stated the degree first and listed afterwards counted as counting | retrieval redefined as answer-first (`route()`, with a test); `qwen3-4b` under `degree` retrieves 47–97% at 57–97% against a count of 100–33%, and the sign matches at all seven densities. The primer also displaces enumeration (85–100% → at most 8%); `all` costs through less accurate retrieval and through non-retrieving responses that are 43 / 4 / 2% accurate from p=.65. Appendix Table 4 gains *enumerates* and *acc. of the rest* rows |
| 38 | "Thinking mode moves most cells out of this range" is false for the 1.7B model (38.8% vs 38.9% of cells in the band) | rewritten: thinking empties the range below 0.25 (28 and 20 cells → 0) and leaves the 1.7B model a similar share in the band |
| 39 | The Sanford link held by construction; "cf. Levy" pointed the wrong way | rewritten around what a stated answer still leaves undone (87% on `node_degree` at p=.50, 30% on `edge_count`); Levy contrasted on sign changes between models |
| 40 | "Answer leakage" in the title was never defined | defined in the Introduction |
| 41 | `qwen3-1.7b` "rarely uses the stated value" omitted its gains at p=.20 and .35 | +10 and +18 where its count is partly right, within 2 points from p=.50 (`[plain17]`) |
| 42 | The 400-per-density replication includes the main sweep's 100 graphs | reports the 300 new graphs: +4.0 [1.2, 6.8], p=.0047 |
| 43 | `all` "4.6 points below the best of its parts" compares with the maximum of three noisy estimates, which is biased upward | mean of parts only |
| 44 | Positive 1.7B length slopes rest on `all`, the longest primer | 3 and 1 of 8 positive without it; heading "No consistent gain from bundling or length" |
| 45 | set-F1 "compresses contrasts about sixfold" hid two reversals | "a sixth as much, and reverses two of the five contrasts above 5 points" |
| 46 | Figure 1: legend over the data, "significant after correction" unspecific | legend in the empty upper right, labelled *answer-carrying*; caption states q<.05 within each (arm, task) and "content primer" (the 393 cells exclude `filler`) |
| 47 | Ethics gave 217,000 generations without the analysed subset | 98,400 analysed (84,000 + 4,800 + 3,200 + 6,400) |
| 48 | After the fixes, Ethics ran to page 6 | tightened without dropping a finding: the Discussion no longer restates §4.2, the within-graph clustering spread moved from Limitations into §4.4 as a finding, the duplicated Shi/Levy sentence left Related Work, the Qwen3 model-card footnote became a bibliography entry, Table 1's forward reference to Table 2 became its own key, and both figures lost 0.1 in of height |
| 49 | All 25 cited references re-checked against the record of the version cited (ACL Anthology, NeurIPS and PMLR BibTeX, Crossref, arXiv, the ICLR proceedings, the Qwen3 model card) | two author lists corrected to the NeurIPS records: Dziri et al. (Welleck is seventh) and Rampášek et al. ("Michael Galkin"). Fatemi et al. stays 2024: arXiv October 2023, but the cited version is ICLR 2024. "Qwen Team" is the citation the Qwen3 model card gives. The other 21 match exactly |

## Status

`make_v3.sh` exit 0; body, Limitations and Ethics end on page 5 (8 pages with
references and appendix); A4, all fonts embedded, no Type 3; abstract 197
words; 0 undefined references, 0 overfull boxes, 0 undefined citations; every
float traces to a generator that `make_v3.sh` runs; all four arms reported.
