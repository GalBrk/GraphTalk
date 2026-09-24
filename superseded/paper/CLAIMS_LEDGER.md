# Claims ledger — every paper version, checked against the data

Written 2026-09-24. Input to the next paper, which will be written from this ledger.

## 1. Versions audited

| Key | Source | Title | Rests on | Status |
|---|---|---|---|---|
| **v1** | `talk_like_a_graph.tex` (+ `.pdf`, `.NEW.pdf`) | Hints, Placebos and Shortcuts | `densfull40`/`hi`, plus 11 "held-out" arms, rewiring, GoT, 5–19-node split | Superseded |
| **v2** | `talk_like_a_graph.v2.tex` | Same title, 12 pp | v1 + `degfixdeg`, length/content split | Frozen |
| **S** | `short/short.tex` | What GraphQA Measures at n=40… | `densfull40`/`hi`, published split, `degfixdeg` | Draft |
| **v3** | `talk_like_a_graph.v3.tex` | When Do Structural Primers Help…? | `densfull40`/`hi` + 3 replication runs | README says "current" |
| **I** | `compare/independent/paper.tex` | Reliability, Consistency, Feature Resolution | `densfull40`/`hi`, independent `analyze.py` | Draft |
| **Y** | `synthesis/synthesis.tex` | Procedural Shortcuts… Invariants and Feature Resolution | `densfull40`/`hi`, CR + PF | Draft; `synthesis.pdf` is older than the `.tex` |
| **P** | `structural_primers_acl2023.pdf` (no source) | Structural Primers as Procedural Shortcuts… | same | Draft |
| **Q** | `Structural_Primers_Graph_Reasoning_ACL2023{,(1),(2)}.pdf` (no source; (1) and (2) are byte-identical) | Structural Primers Trigger Procedural Shortcuts… | same | Draft; Q0 has the solver table, Q1 replaces it with an exact rule |

## 2. How each claim was checked

The numbers below were re-derived on this laptop from `runs/` in three independent ways.

- **Own recount.** A fresh scoring pass written for this audit, sharing nothing with the existing analysis scripts except `graphtalk.scoring.extract_answer`:
  - 84,002 rows, 2 identical duplicates, **84,000 unique**;
  - 4 arms × 7 conditions × (4 densities × 6 tasks + 3 densities × 2 tasks) × 100 graphs.
  - Effects pair on the same graph and drop the pair if either side hit the budget; the CI is a bootstrap stratified by density.
- **Existing pipelines re-read.**
  - `superseded/csv2/raw-trends/primer_findings.txt`, "PF", which uses terminated pairs.
  - `compare/independent/results/`, "CR", which counts a capped generation as a failure.
  - `synthesis/PROVENANCE.md` records that both were re-run from raw on 2026-09-24 and matched.
- **Regenerated artefacts.**
  - All 700 graphs rebuilt from the seed formula (for RWSE resolution).
  - The n=40 solver bars re-run (`scripts/shortcut_table_n40.py` to scratch): **42/42 keys identical** to `shortcuts_n40_flat.json`.
  - The three `qwen3-1.7b` replication runs re-scored.
  - The 5–19-node baselines re-scored.
  - Raw responses read by hand for the route classifier and the discrepancy reports.

Verdicts:
- ✅ **re-derived**: my own recount or regeneration reproduces the number.
- ☑️ **source-matched**: the number matches the pipeline output, but that output was not independently recomputed.
- ⚠️ **reword**: the number is right but the wording claims more, or something different.
- ❌ **wrong**.
- ✂️ **cut**: not reproducible from the current data, superseded, or self-contradictory.

## 3. Setup facts

| # | Claim | Stated in | Verified value | Verdict |
|---|---|---|---|---|
| 1 | 84,000 generations; 700 graphs; 4 arms (Qwen3-1.7B/4B × thinking off/on); 7 conditions; G(40,p) at p∈{.10,.20,.35,.50} for six tasks, plus {.65,.75,.85} for `node_degree`/`edge_existence` | all | 84,000 unique (2 dup rows dropped); cell counts as stated | ✅ |
| 2 | Token budget: **8,192 for every arm in the main sweep**; 2,048 for the plain arms in the high-density extension | v3, I, Y, P | Max `n_new_tokens` among capped rows: plain arms, main sweep = 8,192; plain arms, `hi` = 2,048; thinking arms = 8,192 everywhere | ✅ |
| 2b | "2,048 for plain arms, 8,192 for thinking" (main sweep) | v1, v2, S | Contradicted by the rows | ❌ |
| 3 | `node_count` gold is always 40 and `cycle_check` gold is always "yes" at n=40, so both tasks are uninformative | all | 100% constant at every density | ✅ |
| 4 | Share of `edge_existence` queries that are true edges rises 10→85% with p | v3, Y, P, Q, S | 10/18/31/48/66/76/85% | ✅ |
| 5 | Truncation: 1.7B plain `edge_count` 22.4%; thinking-arm `edge_count` 81.0% / 62.3%; everything else ≤6.5% | v3, S | Same | ✅ |
| 5b | "Outside the two flagged cells, <7% truncates" | v1, v2 | 1.7B plain `edge_count` is 22.4% and is not flagged | ❌ |
| 6 | Primer lengths (chars): components 37, degree 891, clustering 1,629, rwse 2,949, all 4,691, filler 1,829 | v3, Y | PF `[length]` | ☑️ |
| 7 | `filler` is "content-free" | S | It names all 40 node ids ("Node N is simply present"), so it reveals `node_count` (1.7B `node_count` +39.8 under filler) | ⚠️ call it *length-matched, structure-free*, as I does |
| 8 | 5–19-node published split: without a primer, arms sit near ceiling on four tasks and only `edge_count` separates them | v2, S | Re-scored (30 graphs/cell). Most arms score 0.93–1.00 on `node_count`, `node_degree`, `cycle_check`, `connected_nodes`. Exceptions: gemma4-12b-think `connected_nodes` .82 (F1), gemma4-e4b `edge_count` .83. Plain Qwen3-8B/14B `edge_count` .43/.40 | ✅, with "most arms" |

## 4. Graph-blind solver (answer leakage)

| # | Claim | Stated in | Verified | Verdict |
|---|---|---|---|---|
| 9 | Solver: 16 exact rules, 1 heuristic, 8 fitted rules; fit and test on disjoint seeds | v1, v2, v3, Y | Counted in `graphtalk/shortcuts.py` | ✅ |
| 10 | `degree` and `all` carry the answer (solver = 100%) for `node_degree` and `edge_count`. No other (task, primer) cell exceeds 74.6% (`edge_existence`/degree), and `connected_nodes` never exceeds 0.8% | v3, Y, Q0 | Solver re-run: node_degree 14.4 → 21.0 (rwse) / 100; edge_count 3.0 → 10.4 (clustering) / 100; edge_existence 55–75 everywhere; connected_nodes ≤0.8 | ✅ |
| 11 | Any carrying threshold in (74.6, 100] yields the same split | v3, Y | Follows from #10 | ✅ |
| 12 | Route threshold 0.05 over the none bar, "gains ≤0.011 or ≥0.066" | v1, v2, S | True, but this is the older route definition, which v3 / #10 replaces | ✂️ keep #10's version instead |

## 5. Answer-carrying primers: the sign follows the baseline

| # | Claim | Stated in | Verified | Verdict |
|---|---|---|---|---|
| 13 | Plain 4B `node_degree` under `degree`, by density: −6, −1, −7, −12, −6, +7, +27, against a no-primer baseline of 100, 99, 99, 99, 82, 50, 33 | v3, Y, P, Q | PF `[flip]`. My pooled recount gives −6.5 at p≤.50 and +9.3 at p≥.65 | ✅ |
| 14 | At p=.50: 13 broken / 1 fixed (McNemar p=.0018) | v3, Y, Q | Same | ✅ |
| 15 | Under `degree`, plain 4B **retrieves** (answers without restating the neighbour list) in 44–74% of responses. Enumeration falls from 85–100% to ≤8% at p=.35–.75 | v3, Y, P, Q | Classifier read and spot-checked: retrieval responses are ~49 characters long, e.g. "The degree of node 13 is **20**. This is given directly…" | ✅ |
| 15b | P and Q describe responses as *retrieval vs enumeration* only | P, Q | A third class exists: "assert", 20–35% of responses. These open with a (usually wrong) value copied from the primer, notice the contradiction, and recount | ⚠️ name all three classes |
| 16 | Retrieval is only 73–98% accurate, and more accurate for nodes 0–9 (96% vs 83%, permutation p<.001) | v3, Y | Re-derived: 57 of 440 retrievals are wrong. Most errors are small (+1 in 18 cases, within ±3 in 39). This is a **copy error from a 40-line list**. The node-0–9 effect is confounded with single-digit ids and first-in-list position | ✅ (report as copy error; state the confound) |
| 17 | Plain 1.7B: `degree` +10 / +18 at p=.20/.35 (where its own count is 80% / 41%); within 2 points from p=.50 (baseline 5–30%); retrieves ≤4% | v3, Y | PF `[plain17]`. Recount pooled p≤.50: +7.5 [+3.0, +11.8] | ✅ |
| 18 | 1.7B-think: `degree` +19.9 [16.3, 23.6] over terminated pairs; **+16.0 [12.0, 19.7] with capped counted as errors**; +31.6 at p≥.65 | v3, Y, Q | Recount: p≤.50 +11.7, p≥.65 +31.6 [+24.7, +38.0] | ✅ |
| 18b | 1.7B-think "uses the primer **as a check**" | v3, Y, P, Q | Read the responses: it enumerates (76–98%), reports a conflict (23% at p≤.50, 49% at p≥.65), and then **defers to the primer**, typically writing "may be a typo in the connections, but the problem statement specifies…". It lands on the (correct) primer value in 61–69% of these | ⚠️ describe as *counts, then arbitrates toward the stated value*. The gain exists because the primer is correct |
| 18c | Budget: `degree` raises 1.7B-think truncation 1.1% → 7.6%; 96% of the truncated responses mention a discrepancy | v3, Y | PF `[trunc]` | ☑️ |
| 19 | Plain 4B `edge_count` under `degree`: 2.0 → 29.25 (+27.25 [22.75, 31.75], 114 fixed / 5 broken) | v3, Y, I, P, Q | Recount (terminated pairs): +27.7 [+23.1, +32.1], 114 / 5 | ✅ |
| 20 | Plain 4B already uses the degree-sum wording in 96–100% of `edge_count` responses without a primer, yet is exact on only 2% | v3, Y | PF `[edgecount]` | ☑️ |
| 21 | Band analysis (answer-carrying cells): +16.0 / +16.7 at 25–75% baseline, −5.8 at 75–90%, −1.4 at ≥90%. Split-half binning reproduces every band within 3 points | v3 | PF `[bands]`, `[splithalf]` | ☑️ |
| 22 | Cross-fitted route × baseline interaction −11.0 [−17.1, −5.7]; "replicates on eleven held-out arms" | v1, v2, S | Only in `superseded/review_checks.json`. The held-out CI is [−109, −3.8], and v1 says the held-out arms were "used for no claim". Superseded by #21 | ✂️ |

## 6. Joint correctness (degree vs neighbours on the same node)

| # | Claim | Stated in | Verified | Verdict |
|---|---|---|---|---|
| 23 | Plain 4B `degree` lowers joint correctness J 85.75 → 76.50 (−9.25 [−13.75, −4.75], 28 fixed / 65 broken), even though it raises `edge_count` by 27 | I, Y, P, Q | J re-derived exactly: 85.75 → 76.50 | ✅ |
| 24 | Plain 1.7B J 54.00 → 62.50; 10.5% of items are consistent but wrong under `degree` | I, Y, P, Q | J exact. My C is 73.00, which matches | ✅ |
| 25 | `components` J 95.00 for plain 4B; 4B-think J 93.75, `degree` +2.00 (n.s.) | I, Y | Exact | ✅ |

## 7. `edge_existence` is a response-bias measurement

| # | Claim | Stated in | Verified | Verdict |
|---|---|---|---|---|
| 26 | Plain arms say "yes" to 97–100% of true edges under every primer; 94–99% of errors are false alarms | v1, v2, v3, Y, S | PF `[fa]` | ☑️ |
| 27 | Plain 1.7B collapses from p=.65: yes-rate 98–99%, balanced accuracy 52–53%, median 48–120 tokens, while raw accuracy climbs to 86% | v3, Y, I, P, Q | Recomputed: yes-rates 98/99/99, balanced 53/52/53, tokens 50/47/120 | ✅ |
| 28 | `all` lowers 1.7B false alarms 51 → 33% (−18.6); `filler` raises them to 69% (+18.3). Balanced accuracy at p≥.65: `degree` +9.5, `all` +14.9 | v3, Y, P, Q | Balanced accuracy re-derived per density. False-alarm denominators are terminated pairs (n≈366) | ✅ (state the denominator) |
| 29 | Plain 4B: `clustering` is the only primer that lowers false alarms (16 → 8%) | v3, P | PF `[fa]`. My balanced accuracy: clustering 92–99 vs none 79–98 | ✅ |
| 30 | Signal-detection Δd′ tables | v1, v2 | The papers themselves say d′ is not identifiable at a ceiling hit rate | ✂️ |

## 8. Side-information primers (not answer-carrying)

| # | Claim | Stated in | Verified | Verdict |
|---|---|---|---|---|
| 31 | 4B-think does not move by more than 2.1 points under any primer on `node_degree` / `edge_existence` / `connected_nodes` | v3 | Recount: largest move is rwse on node_degree, −2.1 | ✅ |
| 32 | Plain 4B `node_degree` at p≥.65: `clustering` +11.3 [6.7, 16.3], `rwse` +7.7, `filler` +2.0 (n.s.), `all` −21.3 | v3, Y | Recount: +11.3 [+7.0, +16.0], +7.7, +2.0, −21.3 | ✅ |
| 33 | That `clustering` gain is non-specific: the printed coefficients barely vary within a graph (SD ≤ 0.021), and the procedure is unchanged (69% vs 74% enumerate) | v3, Y | PF `[cluster]`, `[clustproc]` | ☑️ |
| 34 | Plain 1.7B `clustering` on `node_degree`: +3.0 in the main sweep (n.s.); replicated at +4.0 (new graphs), +4.2 (fresh seeds), +6.2 (fixed mean degree, n=20–160) | v3 | Re-scored the raw runs: +4.0 [1.2, 6.6] p=.0047; +4.2 p=.00055; +6.2 p=3.5e-12 | ✅ |
| 34b | **Not stated anywhere:** the same effect is absent at high density. The dedicated 1.7B run at p≥.65 gives −0.7 [−2.8, +1.2] (n=1,200); the main sweep gives −3.3 [−7.0, +0.7] | none | Re-scored `qwen3-1.7b.degdens40hi` | new; must be stated (effect holds at p≤.50 only) |
| 34c | "+5.9 at p=.35–.50" window | v1, v2, S | The window was chosen post hoc, and #34 supersedes it | ✂️ |
| 35 | Rewiring (clustering .01 → .73 at fixed degree): no detectable clustering gain at any level | v1, v2, S | `rewiring_qwen3-1.7b.json`: +3.5 / +6.0 / +2.5, all n.s., n=200. This null is weak (CI ±7) | ⚠️ appendix at most; call it underpowered |
| 36 | Plain 4B `connected_nodes` under `components`: +9.3 (44 fixed / 7 broken). At p≥.20 every graph is connected, so the primer is a fixed 37-character sentence | v2, v3, S, I | Recount +9.2 [+6.0, +13.0], 44 / 7. Connectivity re-derived | ✅ |
| 37 | 4B-think `edge_count`: `components` and `clustering` both 20.0 → 44.75 success, through completion (20.5 → 45.5%); 97.6% of completed no-primer answers are correct | I, Y, Q | CR `pooled.csv` | ☑️ |
| 38 | `all` is no better than the mean of its parts (+0.2 [−1.1, 1.6]) | v3, Y | PF `[bundle]` | ☑️ |

## 9. Length and filler

| # | Claim | Stated in | Verified | Verdict |
|---|---|---|---|---|
| 39 | `filler` hurts both plain arms on `connected_nodes` and `edge_existence`: 1.7B −6.2 / −15.8, 4B −12.5 / −7.0 | v1, v2, S, I | Recount exact | ✅ |
| 40 | The length slope is positive for 1.7B in 6 of 7 fits and for 4B in 0 of 7; filler sits 18.3 / 13.8 points below the fitted line | v3 | PF `[length]` | ☑️ (appendix) |
| 41 | "Length cost grows with difficulty" | v1, v2 | v2 itself reports r = +0.55 / −0.09 / −0.44 by arm | ❌ |

## 10. Feature resolution

| # | Claim | Stated in | Verified | Verdict |
|---|---|---|---|---|
| 42 | RWSE printed to 2 dp gives 30.14 / 6.74 / 3.30 / 2.00 distinct pairs per graph (p=.10 / .35 / .50 / .85); the modal pair covers 62.68% of nodes at p=.50; from p=.35 this is fewer classes than degree in 100/100 graphs | Y, I, P, Q | Regenerated all 700 graphs: exact. The count is not monotone (2.08 at .65, 2.72 at .75) | ✅ |

## 11. Other

| # | Claim | Stated in | Verified | Verdict |
|---|---|---|---|---|
| 43 | 1.7B `node_count`: answers "39" on 68.5% of items without a primer (reads "nodes 0…39" as a count) | v1, v2, v3, S | PF `[nodecount]` | ☑️ (one line) |
| 44 | Regenerating 1,200 identical prompts changes 5.3% of outcomes | v3 | PF `[rerun]` | ☑️ (limitation) |
| 45 | MDE medians 5.6 / 10.0 | v1 | Stale; after the fix they are 5.5 / 5.4 | ❌ |
| 46 | 1.7B-think `node_degree` baseline 0.642 | v2 | The table says 76.8 | ❌ |
| 47 | "Both thinking arms at ceiling" | v1, v2 | 1.7B-think is at 76.8 / 79.7 on node_degree / connected_nodes | ❌ |
| 48 | "The one primer that beats both controls" | v2 abstract | v2's own body reports two such primers | ❌ |
| 49 | Set-F1 on `connected_nodes` is 0.96–1.00 "in every cell" | v1, v2, S | Holds pooled; per-density minimum is 0.931 | ⚠️ |
| 50 | "16 singleton responses conceal correct lists" | I | `new_success` sums to 15 | ❌ |
| 51 | 1.7B-think reports a mismatch in "40–59%" of density cells | P | Range is 39–59% | ❌ (minor) |
| 52 | GoT renaming null, neighbour-copy test (p=.002), negative-control instrument | v1, v2 | Source is the 5–19-node corpus and `candidates/` scripts. Not re-derived here, and not needed for the n=40 story | ✂️ |
| 53 | Ethics generation count ~200,000 | v1, v2 | 217,178 lines in `runs/` | ❌ (use 217,178) |

## 12. What the new paper should carry

The spine is the 40-node sweep. Every body claim below is ✅ or ☑️.

1. **Setup and leakage.** Six tasks, but two are constant at n=40 (#3). `edge_existence` needs balanced accuracy (#4, #27). The graph-blind solver sorts primers into answer-carrying and side information (#10).
2. **Answer-carrying primers change the procedure, and the sign follows the displaced procedure's accuracy.**
   - Plain 4B, −12 → +27 (#13–#16), through retrieval with copy errors.
   - 1.7B-think counts, then defers (#18, #18b).
   - Plain 1.7B helps only where its own count is middling (#17).
   - Bands (#21).
3. **A trade-off hidden by per-task accuracy:** `edge_count` +27 while degree/neighbour joint correctness falls 9 (#19, #23).
4. **`edge_existence` measures a yes-bias.** Collapse at p≥.65; primers act on false alarms (#26–#29).
5. **Side information is small and non-specific.**
   - `clustering` gains (#32–#34b, including the p≥.65 null).
   - `components` as a constant sentence (#36).
   - Completion effects on the thinking arms (#37).
   - Bundling (#38), filler costs (#39).
6. **Feature resolution:** 2-dp RWSE collapses (#42).
7. **Prior work, briefly.**
   - Fatemi et al. (the encodings, GraphQA).
   - Our 5–19-node pass: near ceiling, so the move to n=40 (#8).
   - The published-split solver bars do not transfer (edge_existence/none .498 → .735).

**Cut:** #12, #22, #30, #34c, #41, #45–#48, #52. **Appendix only:** #35, #40, #43, #44.
