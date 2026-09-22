# v3 review log

Tracking file for the 5-page rebuild. Target: body (Introduction through
Conclusion, excluding references and appendix) at 5 pages, organised around
which primers help and harm — for which task, which arm, on what terms, and
where the limit is.

Each round: review, list issues, fix, re-review. A round closes only when the
verifier block below is all green and no open issues remain.

## Verifier block

| check | command | target |
|---|---|---|
| builds | `bash paper/make_v3.sh` | exit 0 |
| body length | `sec:conclusion` in `.aux` | **page 5 or lower** |
| undefined refs | `grep -c 'LaTeX Warning: Reference'` | 0 |
| overfull boxes | final pass of the log | 0 |
| numbers traceable | every figure in prose maps to a `csv2/raw-trends/` column in `paper/NUMBERS.md` | no orphans |
| tests | `pytest -q tests/test_raw_frame.py tests/test_scoring.py` | pass |

## Triviality audit

The paper must carry findings a reviewer could not have predicted. Rated
against "would a competent reader have guessed this before reading?"

**Expected — keep only as machinery, never as a headline**

| finding | why it is not a result |
|---|---|
| Primers help models with headroom, hurt models at ceiling | a ceiling effect; the interesting part is *where* the crossover sits and that the same primer flips sign, not that it flips |
| Added text costs accuracy | already established \citep{levy2024sametask,shi2023distracted}; ours is a control, not a contribution |
| Set-F1 hides the ceiling on `connected_nodes` | a measurement choice; one clause in Metrics |
| Bigger model scores higher | not reported |

**Non-trivial — these carry the paper**

| finding | why it is surprising |
|---|---|
| A primer that **states the answer verbatim** lowers accuracy (`degree` on `node_degree`, `qwen3-4b`: 99 → 87, breaking 13 items and fixing 1) | the information is present and correct, and the model does worse with it |
| **Where in the primer** the answer sits moves accuracy by 24.3 points — more than the primer's own total effect (−6.0) | position dominates presence |
| The primer **replaces the algorithm**: sum-of-degrees stated in 7% → 100% of responses | a content change, visible in the text, not inferred from scores |
| Where the primer hurts, **reasoning collapses** 273 → 34 tokens | the model trades a reliable long computation for an unreliable lookup |
| Raw accuracy on `edge_existence` **rises to 86% while balanced accuracy sits at chance** | the benchmark rewards giving up |
| **Capitulation is triggered by answer format**, not by prompt length or difficulty | both obvious causes are ruled out by the ladder |
| `all` ≈ its **largest single part**, not a diluted sum (0.80 [0.56, 1.08] vs 0.44 [0.33, 0.63]) | bundling buys one primer's worth out of three |
| **Relevance does not predict the effect** — mismatched content is worth +8.1 on `qwen3-1.7b`, beating two of four matched pairs | the obvious organising rule fails |
| Same primer, same task, **opposite sign**: `degree` on `edge_count` is −43 for `qwen3-1.7b-think` at p=.10 and +33 for `qwen3-4b` at p=.50 | not a property of the primer at all |

**Framing rule that follows:** the paper opens on the answer-stating paradox and
what the generated text shows, not on the headroom law. The headroom law is the
*explanation* offered afterwards, which is where its familiarity stops being a
liability.

## Open issues

### Round 1

| # | issue | status |
|---|---|---|
| 1 | Body is 9 pages; target 5 | open |
| 2 | Related Work is a full section for a 5-page paper | open |
| 3 | Method has five subsections; two would do | open |
| 4 | Headroom crossover (the "limit" result) is not in the paper and has no script | open |
| 5 | Plain-arm `cycle_check` in the budget comparison reads 90.7; the frame gives 92.2 under either truncation policy | open |
| 6 | `clustering` replication cites +5.9 from the dedicated corpus without the main-sweep figure (+4.0 on the same arm) beside it | open |
| 7 | Floor/ceiling null-cell claim ("42%", "all 33") does not reproduce and its cell definition is unstated | open |
| 8 | Paper still leads on the headroom law rather than the answer-stating paradox | open |
