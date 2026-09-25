# Paper v3 review — fix order

**Date:** 2026-09-22. **Status:** adversarial review of
`superseded/paper/talk_like_a_graph.v3.tex` against the repo's own artifacts. **Nothing in
the paper or in `csv2/` was changed.** Every finding names the file, the line,
and the recomputation that establishes it, so any of them can be disagreed with
on the data.

Method: nine independent audits (internal consistency, number traceability,
statistics, design validity, claims-vs-evidence, citations, code-vs-paper,
presentation, and a reject-case advocate), each verified twice — once by an
agent arguing the authors' defence, once by an agent redoing every check from
scratch — plus a completeness pass over the sections the nine did not reach.
130 findings survived verification; 2 were refuted and dropped.

Most numbers below were recomputed from `csv2/raw-trends/frame.csv` (84,000
rows) using `superseded/paper/make_v3_tables.py`'s own `paired()` logic: join on
`graph_id`, drop the pair if either side has `hit_cap`. Where a finding needed
the raw generations it says so.

Tags: **[edit]** text only · **[regen]** rerun a generator · **[analysis]** new
analysis on existing data · **[verify]** confirm a fact first.

---

## 0. Read this first

The experiment is sound and the data are real. Independent recomputation
reproduced roughly 45 of ~60 traced prose numerals *exactly*, including every
number in §5.1, every fix/break count, every response-length figure, every
serial-position slope and bootstrap interval, and every additivity count. The
feared v1/v3 pipeline split is **not** there: `superseded/paper/ci_table.tex` (v1
pipeline) and `superseded/paper/v3_main_table.tex` (v3 pipeline) agree in all 144 shared
cells to within rounding.

The "it's all noise" objection also fails. Simulating the null over the 240
usable cells (2,000 draws, discordant counts held fixed) gives an expected 3.7
cells with |Δ|≥10 and 0.8 with |Δ|≥20. Observed: **34 and 16**. This is not a
noise-mining paper, and the paper never says so — it should.

What sinks the current draft is that the write-up is not disciplined by the
data it sits on: a misstated decoding budget, two truncation conventions used
interchangeably across tables, a flagship inference that a content-free null
reproduces, and a "control" that controls neither length nor content. All of it
is fixable without a single new generation.

---

## 1. P0 — correctness. Nothing ships until these are done.

### 1.1 Decoding budget is wrong by 4×, and it changes mid-corpus

`:316`, `:697`, `:939`, `:1142-1143` · **[verify][edit][regen]**

> "Decoding is greedy with a budget of $2{,}048$ new tokens for plain arms and
> $8{,}192$ for thinking arms"

Truth from the raw generations: `densfull40` plain arms ran at **8,192** —
`runs/qwen3-1.7b.densfull40.shard*of25.jsonl` has 801 `hit_cap` rows, **all at
`n_new_tokens=8192`**, and 1,767 generations above 2,048, which is impossible
under a 2,048 cap. `qwen3-4b.densfull40`: 48 caps at 8192, 155 rows above
2,048. `densfull40hi` plain arms ran at **2,048** (caps `{2048: 5}` and
`{2048: 3}`). Thinking arms: 8,192 throughout, both corpora.

The paper contradicts itself on the page: §5.3 reports a median *terminating*
response of **3,737 new tokens** for `qwen3-1.7b`, impossible under 2,048.

Two consequences beyond the sentence:

- **`hit_cap` does not mean the same thing across densities inside one
  7-density series.** It is the variable driving pairwise dropping, the
  truncation table, and every response-length analysis. `effects.csv` and
  `serial_position.csv` mix two plain-arm budgets on `node_degree` and
  `edge_existence`. Practical damage is small (8 truncated plain-arm rows in
  the extension; responses above p=.50 run ~50 tokens) — say so rather than
  leaving a reader to assume the worst.
- **Table 12 is not budget-matched and currently handicaps the thinking arm.**
  The plain columns are observed-at-8,192; `think@2048` is rescored down.
  Rescore *both* at 2,048 and relabel, or drop the table. `:697` ("Rescored at
  the plain arms' $2{,}048$-token budget") rests on the false premise.

Also re-check §5.3's "budget-bound" framing for `edge_count`: with 8,192
available, "runs out of budget" needs restating.

### 1.2 "Changes the verdict on only three cells" is false

`:268-271` · **[edit]**

> "Truncated pairs are dropped; scoring them as incorrect changes the verdict
> on only three `qwen3-1.7b` `cycle_check` cells"

Recomputed over all 144 (arm, task, condition) cells under both conventions,
applying the paper's own BH-within-(arm, task): **24 cells flip at α=.05, 21
under BH**, spread over all four arms and five tasks. Three of the 24 are
`1.7b`/`cycle_check`. Nineteen cells move by ≥5 points. Three bolded Table 1
numbers **reverse sign**:

| cell | Table 1 (drop) | zeroed | n |
|---|---|---|---|
| `1.7b-think` / `edge_count` / degree | **−61.5** (p=.008) | −3.2 (n.s.) | 13 |
| `4b-think` / `edge_count` / all | **−41.5** (p<1e−4) | +1.8 (n.s.) | 41 |
| `4b-think` / `edge_count` / degree | **−23.5** (p=.007) | +8.5 (n.s.) | 34 |

`superseded/paper/ci_table.tex` already prints all of this — n, Δ, and the `zeroed`
column — for every cell. That makes it worse, not better: the appendix shows
the reader the refutation while §3.4 asserts the opposite.

Also delete **"the least favourable convention for the primer"** — false in 40
of 144 cells, sometimes by 21 points.

### 1.3 Table 1 needs an `n` column

`superseded/paper/make_v3_tables.py` → `main_table()` · **[regen]**

Twelve cells rest on fewer than 100 of 400 pairs; the smallest on **13** (8
discordant) and it is bolded significant. The caption says "$100$ graphs per
(task, primer, density) cell" and shows no n.

| arm | task | primer | n | Δ |
|---|---|---|---|---|
| `1.7b-think` | edge_count | degree | 13 | −61.5 |
| `1.7b-think` | edge_count | clustering | 23 | −8.7 |
| `1.7b-think` | edge_count | all | 24 | −8.3 |
| `1.7b-think` | edge_count | rwse | 25 | −4.0 |
| `1.7b-think` | edge_count | components | 27 | +7.4 |
| `1.7b-think` | edge_count | filler | 29 | +3.4 |
| `4b-think` | edge_count | degree | 34 | −23.5 |
| `4b-think` | edge_count | all | 41 | −41.5 |
| `4b-think` | edge_count | rwse | 49 | +0.0 |
| `4b-think` | edge_count | filler | 50 | −2.0 |
| `4b-think` | edge_count | clustering | 59 | +1.7 |
| `4b-think` | edge_count | components | 60 | +1.7 |

`paired()` already returns n; the call site discards it. Add the column and
suppress bolding below a floor (n≥100 is defensible). Note the surviving pairs
are selected on a **post-treatment** outcome — on `edge_count`, terminating
under `degree` means the model took the handshake route — so this is selection
on the mechanism under study, not missing-at-random.

### 1.4 One truncation convention across all tables

`superseded/paper/make_v3_tables.py`, `superseded/scripts/raw_trends.py:678` · **[regen]**

`tab:main` drops capped pairs (`paired()` filters `hit_cap_a==0 &
hit_cap_b==0`). But `tab:additivity`, `tab:relevance`, `tab:behaviour` and the
abstract's **0.44** descend from `superseded/csv2/raw-trends/effects.csv`, written by
`q_effects(df)` with the default `drop_capped=False` — truncated generations
scored **wrong**. `effects_capdropped.csv` exists and is unused.

Visible consequence today: the same quantity, same stated scope, printed twice
with a 16-point gap — `degree`/`edge_count` vs filler is **−5.5** (`1.7b-t`) in
`tab:relevance` and **−18.8** in `tab:main`.

Under the declared convention the headline moves: **0.44 → 0.50 [0.29, 0.71]**
(41 cells), and all/largest-part goes 0.80 [0.56, 1.08] → 1.00 [0.66, 1.15].

Regenerate the three tables from `effects_capdropped.csv` and re-derive the
abstract number. If the cap-as-wrong convention is deliberate for these tables,
say so in each caption and in §3.4.

### 1.5 Two claimed manual validations have no artifact

`:268`, `:754` · **[edit][analysis]**

- `superseded/csv2/raw-trends/marker_validation.csv` holds 72 sampled rows whose two
  hand-label columns are **100% empty**. `superseded/scripts/raw_trends.py:607` docstring:
  *"Stratified sample to hand-label; no Q5 claim ships before this is
  checked."* Lines 623-624 only ever emit blank columns. The Limitations claim
  "hand-validated on a stratified sample" has no artifact. Mitigating: a spot
  check of 200 `1.7b`/`edge_count`/p=.50 responses found the regex precise
  (107 hits: 100/100 under `degree`, 7/100 under `none` — matching the reported
  rates), so **filling in the 72 labels is the cheap path**, not deleting the
  claim.
- `:268` **"`<think>` blocks never reach the extractor" is false.** Nothing
  strips `<think>`: `scripts/build_raw_frame.py:218-219` passes the stored
  response straight to `scoring.extract_answer`, and `graphtalk/scoring.py` has
  no `</think>` handling. `scripts/audit_extraction.py`'s own docstring says
  the opposite of the paper. Over 6,720 thinking-arm rows, **14.5%** of
  `1.7b-think` and **10.3%** of `4b-think` responses contain no `</think>` at
  all. Where the block does close, full-text vs post-think extraction differs
  on 0.38%/0.96% of rows, every mismatch on a `hit_cap` row. `NUMBERS.md:63`
  hedges it correctly as "never reach the extractor *unstripped*"; the paper
  drops the word and asserts the reverse.
- "A manual check of $50$ rows per arm" traces, via `NUMBERS.md`, to
  `scripts/audit_extraction.py` — an automated check. Say automated, or do the
  manual one.

### 1.6 `filler` is neither length-matched nor content-free

`:44-45`, `:60-61`, `:98`, `:105`, `:194-196`, `:418`, `:599`, `:713`, `:733`
· **[edit][analysis]**

**It does not control length.** Fitting Δ_vs_none on primer length across the
five real-content primers (37 → 4,691 chars) within each (arm, task) gives a
length term of **−0.02 points per 1,000 characters** — 7 of 14 cells positive,
7 negative. Across the whole range the paper spans, added text costs nothing
systematic. `filler` sits **−4.2 points below that line on average, −18.3 at
worst**:

| arm | task | length term | filler predicted | filler observed | residual |
|---|---|---|---|---|---|
| `1.7b` | edge_existence | +3.4 | +2.5 | −15.7 | **−18.3** |
| `4b` | connected_nodes | −1.7 | +1.3 | −12.5 | **−13.8** |
| `1.7b-think` | node_degree | +1.6 | +4.8 | −3.5 | −8.4 |
| `1.7b` | connected_nodes | +0.0 | +1.7 | −6.2 | −7.9 |
| `1.7b` | node_degree | +1.3 | +5.0 | −0.5 | −5.5 |

So the paper's central methodological claim — *"a content-free preamble of
matched length is itself worth up to 15.7 points, so any effect measured
against `none` alone confounds content with length"* — **misattributes a
filler-specific effect to length**. The correct statement: a preamble of 40
repetitive contentless sentences damages accuracy; adding 4,700 characters of
informative text does not.

It is also not length-matched to anything but `clustering`. Mean `primer_chars`
from `frame.csv`: `components` 37, `degree` 891, `clustering` 1,629, `filler`
1,829, `rwse` 2,949, `all` 4,691. Relative to filler: components −98%, degree
−51%, clustering −11%, rwse +61%, all +157%. **The "Δ vs filler" column — the
half the caption tells readers to prefer — is a length-matched comparison for
one of five conditions.**

**It is not content-free either.** `graphtalk/primers.py:252-262` renders one
sentence per node in sorted id order: *"Node 0 is simply present within the
graph G. … Node 39 …"*. That states the vertex set exactly.
`graphtalk/shortcuts.py:868-869` has an **exact theorem rule**
(`count_sentences`) for it, and `tab:shortcut-published` bolds
`node_count`/`filler` at **1.00** against a `none` bar of 0.064. At n=40 the
constant gold answer hides it.

Fixes:
- Stop calling it "length-matched" and "content-free" (9 sites). Call it a
  *degenerate-content preamble of comparable length to `clustering`*.
- Replace the binary vs-filler contrast with the length-line residual
  (analysis A, §5) — it controls all five conditions instead of one.
- `:599` **"containing no per-node content whatsoever" must go.** Filler has 40
  per-node id-ordered lines — the same positional structure as `degree` — so
  its slope is a second retrieval condition, not a control. This is the
  load-bearing sentence for §5.4's main inference.

### 1.7 Stale and contradictory numbers

**[edit]**

| site | paper says | should be |
|---|---|---|
| `:57`, `:726` | "up to $21.6$ points" | Body gives 24.3 (`slope_pts`); `first10_minus_last10` gives 21.7. 21.6 matches no artifact. Pick one statistic and name it. |
| `:596` | "the steepest slope we measure" (−19.3) | −24.3 is reported three paragraphs earlier |
| `:488-489` | +26.0, +21.7 | `tab:main` gives +26.5, +21.8; `tab:relevance` gives +26.0, +21.7. The gap is the cap convention (§1.4). |
| `:475` | +9.2 | `tab:main` gives +9.3 |
| `:447` | `Tables~\ref{tab:main} and~\ref{tab:main}` | duplicated cross-reference |
| `:61`, `:98`, `:713` | filler "worth up to $15.7$ points" | Largest filler effect in `tab:main` is **+39.8** (`1.7b`/`node_count`), bolded, 2.5× larger and opposite in sign. Scope the claim. |
| `:268` | "82% and 80%" truncation | correct for `none` only; `tab:hitcap`'s 81.0/62.2 are condition-pooled. Scope is fine — say so. |

---

## 2. P1 — claims that do not survive. Rewrite or delete.

### 2.1 Drop the −11.0 route × baseline interaction

§5.6, `tab:robust`, `tab:baseline`, `fig:crossfit` · **[edit]**

Cross-fitting removes shared *sampling noise* between baseline and effect. It
does not remove the **bounded-outcome** confound: a cell at baseline 0.99 can
gain at most +1 and lose up to −99, and the reverse at floor. Route conditions
churn far more answers than no-route (mean discordant rate **16.5% vs 9.2%**),
so bound arithmetic alone converts "route primers change more answers" into
"route primers slope more negatively on baseline" — with no content, no route
and no reasoning involved.

Verification: the analysis was reimplemented from `frame.csv`, reproducing the
paper exactly (−11.0 [−17.0, −5.6], k_route=232, k_none=762, 83 blocks). The
identical pipeline was then run on a **content-free null** in which each cell's
outcomes under the condition are the `none` outcomes with a random subset
flipped at that cell's own observed discordance rate — flips independent of
correctness, so the primer carries zero information. Over five seeds:
**−11.5 [−17.4, −6.5], −12.9, −10.9, −12.0, −12.0.** Closed form under an
independent-flip null: −33.0 (route), −18.3 (no route), interaction −14.7.

*The content-free null predicts a slightly larger interaction than the one
observed.*

The paper's own windowed row confirms the mechanism: inside baseline
[0.10, 0.90] the route correlation is **r = −0.009** (k=78). 66% of route
points and 81% of no-route points sit outside that window.

Actions:
- Delete "the cross-fitted interaction test in the text addresses this" from
  the `tab:baseline` caption — it does not.
- Replace the result with the window table (analysis B, §5).
- The `.10–.90` rows are a **failed check reported as a shape finding**:
  `+3.0 [−29.1, +41.1]` comfortably contains −11.0, and
  `+0.22 [−1.36, +1.94]` contains −1.35. That is "unmeasured", not "it
  disappears". `superseded/scripts/analyze_review_checks.py:11-13` states the check's
  actual purpose, which the paper reframes.

### 2.2 Rebuild §5.4 around a tested difference

**[analysis][edit]**

The section infers "primers that state the answer have a steeper gradient than
the controls" from "the primer's CI excludes zero, `none`'s does not" — the
Gelman & Stern error. `gelman2006difference` is in `paper/custom.bib:161` and
cited **zero** times in v3.

Paired bootstrap (2,000 draws, resampling graphs so both conditions stay on the
same graphs) of slope(cond) − slope(none) on `node_degree`:

| contrast | Δslope | 95% CI |
|---|---|---|
| `4b` / degree | −10.8 | [−23.3, +1.8] |
| `1.7b-think` / degree | −11.9 | [−24.6, +0.9] |
| `1.7b-think` / filler | −9.3 | [−20.6, +1.8] |
| `1.7b-think` / rwse | +1.6 | [−10.5, +12.9] |
| `4b` / filler | +6.6 | [−0.9, +14.4] |
| `4b` / all | −9.2 | [−20.3, +1.7] |
| `4b` / rwse | +1.4 | [−6.0, +9.0] |
| **`1.7b-think` / all** | **−15.5** | **[−28.3, −2.5]** |

**7 of 8 include zero.** Only `all` on `1.7b-think` survives. Separately, the
56 slope CIs get no multiplicity correction; BH over the 28 `node_degree`
Spearman p-values leaves 5 of 8, and the specifically cited `4b`/degree cell
drops out (p=0.0136, q=0.063).

Delete *"Where in the primer the answer sits matters more than whether the
primer is there."*

### 2.3 Report the second half of `serial_position.csv`

**[edit]**

The file also holds `connected_nodes`. On `qwen3-4b` the slope there is
**+25.0 [14.5, 36.7] under `none`** (ρ=+0.222, p=7.3e−6) — reversed, larger,
significant, with no primer at all. `filler` gives +26.3 [12.0, 40.0].

§5.1 establishes that `node_degree`, `connected_nodes` and `edge_existence` ask
about the **same node of the same graph**. So the id-innocence argument ("ER
ids are assigned before edges are drawn, so the 40th line describes a node no
harder than the first") cannot cover it: same node, same graph, no primer,
+25 points on one task and ~0 on another. And *"under `none` no arm has an
interval excluding zero"* is true only of the task the paper reports.

Related: position is confounded with position in the **encoding**, which is
also node-id ordered (`graphtalk/prompts.py`, and the appendix example prompt
shows it). `none` is therefore not a positional control at all. The Limitations
name the node-identity confound but not this one.

### 2.4 Name the leakage in the abstract

`:41-58` · **[edit]**

§3.2 states the criterion: *"A primer's measured benefit is evidence of graph
reasoning only if it exceeds what a solver that never sees the graph can
recover from the primer text."* `tab:shortcut` gives the `degree` bar as
**1.00 on both `edge_count` and `node_degree`** — bolded, "the primer content
alone determines the answer". Both poles of the abstract's flagship reversal
are therefore pure answer leakage by the paper's own test. The words "leakage",
"route" and "substitute" appear nowhere in the abstract.

Suggested replacement: *"a primer that states the requested quantity outright
is worth +26.5 points on `edge_count`, where a graph-blind solver scores 1.00,
and −6.0 on `node_degree`, where it also scores 1.00 — the same leakage helps a
model at floor and hurts one at ceiling."*

### 2.5 Deliver contribution 1, or stop claiming it

Fig. 1 `:162-169`, §3.2, §5 · **[analysis][regen]**

Figure 1 makes the shortcut bar the organising device and states a decision
rule ("must exceed"). The rule is **never executed anywhere in the Results**.
Grepping the paper, `bar`/`bars` appears only in §3.2, the two appendix bar
tables, and the sentence defining the route/no-route label.

It is also not well-formed as written: a bar is an accuracy in [0,1], an effect
is a paired difference in points. Applied the only sensible way — is the model
doing better than a solver that reads the primer and nothing else? — it
inverts the paper's readings:

| cell | `none` | with primer | bar | gap |
|---|---|---|---|---|
| `4b` / edge_count / degree | 2.0% | 29.7% | 1.00 | **−70** |
| `1.7b` / node_degree / degree | 60.5% | 68.0% | 1.00 | −32 |
| `1.7b-think` / node_degree / degree | 76.8% | 88.5% | 1.00 | −12 |
| `4b` / node_degree / degree | 99.2% | 92.7% | 1.00 | −7 |

Add a bar column to Table 1, mark route cells, and add one paragraph: *the
model recovers 30% of what the primer hands it.* That is a publishable finding
and the strongest use of the solver in the paper. Reword "must exceed" into
something dimensionally coherent.

### 2.6 `components` is a constant string on 300 of 400 graphs

`:474-482` · **[edit][analysis]**

At n=40 every ER graph at p≥.20 is connected, so the `components` primer says
the same thing about every graph at three of the four densities (at p=.10 only
48/100 graphs have one component). It carries zero graph-specific information
there — which is exactly how `:105` describes it in the contributions list, as
a *control*.

The effect is **not larger where the string varies**: vs-`none` deltas for
`4b`/`connected_nodes`/`components` run +7.0 (p=.10, string varies), +11.0,
+10.0, +9.0 (p≥.20, string constant). Flat in information content — the
signature of a format effect.

The paper applies the opposite standard 90 lines earlier: at `:387-388` it
attributes the large `node_count` gains to *"almost any interposed text
interrupts that off-by-one"*. That account fits here and is never considered.

Either rename to "the largest effect of a short preamble we measure" and
disclose the constancy, or report p=.10 and p≥.20 separately. Drop
`components`/`connected_nodes` from the matched-pair analysis at p≥.20.

Note the `node_count` account is itself refutable by its own Table 1 row: the
six interposed texts give +2.0, +6.8, +21.2, +39.8, +47.0, +67.3 — a 65-point
spread, **anti-correlated with length** (`rwse`, +1,070 tokens, buys +2.0 and
is not significant; `components`, +8 tokens, buys +21.2). Neither
"interposition" nor "length" explains it.

### 2.7 Fix the matched-pair map

`superseded/scripts/raw_trends.py:64-70`, `tab:relevance` · **[edit]**

Bars from `shortcuts_n40.json`, averaged over the four evaluated densities:

| "matched" pair | bar gain over `none` |
|---|---|
| `degree` → `edge_count` | **+0.968** |
| `degree` → `node_degree` | **+0.853** |
| `components` → `connected_nodes` | **0.000** |
| `clustering` → `edge_existence` | **−0.008** |

Two of the four are matched only by a shared English word: a component count
cannot tell you which nodes neighbour node 16, and per-node clustering does not
determine whether (u,v) is an edge. Split the map into *determines the answer*
and *topically related*, and report them separately. Note that
`clustering`/`edge_existence` vs filler (+19.5) is carried almost entirely by
the control's damage — filler−none on that cell runs −5, −20, −32, −6.

### 2.8 Disclose the post-hoc density window

`:684` · **[edit]**

Clustering−none across seven densities: +0.3, +3.3, **+7.5, +4.3**, −2.0, +0.5,
−0.5. The `.35`–`.50` window is exactly the top two. `superseded/docs/rq3-leads.md:72-76`
already states it: *"The range was chosen after seeing the default seed."*

Add one sentence saying so; report the pre-specified all-density estimate
(**+1.9, p=0.022**) beside it; and reconcile the two replication figures — the
`tab:headline` row gives +4.2 over p=.10–.50 while the prose gives +5.5, which
is the replication restricted to the selected window. The independent-seed
+5.5 *is* a genuine out-of-sample test of an already-fixed window; say that, it
is the paper's best defence here.

Related: `:742` attaches "were not pre-registered" to the screened arms alone,
implying the main sweep was. Nothing in v3 was. The four
`analysis/confirmatory_*.json` files cover GoT naming and `qwen3-8b`, none of
which v3 reports.

### 2.9 Fix the additivity null

§5.5, `tab:additivity` · **[analysis][edit]**

Both ratios are tested against 1, but the estimator does not take the value 1
under the corresponding null — it is a noisy numerator over a noisy
denominator, with cells selected on the denominator being large (|sum| ≥ 4).

Matched simulation (resample within cell, recentre to exact additivity, keep n,
the numerator–denominator correlation and the screen):

- **Median ratio under exact additivity: 0.77 [0.64, 0.91]**, not 1.00.
  Observed 0.44 is still below it (0/400 replicates ≤ 0.44), so sub-additivity
  is real — but the shortfall is 0.44 vs **0.77**.
- For the largest-part ratio, the null "all == the true largest part" gives
  median **0.951 [0.800, 1.072]**; observed 0.800 sits at its 2.5th
  percentile — borderline evidence the bundle is *below* its largest part, the
  opposite of "indistinguishable".

Restate the abstract accordingly: "recovers a median 0.44 of what they are
worth separately" overstates the shortfall relative to what the estimator
returns under exact additivity.

### 2.10 Confront the thinking-arm `edge_count` reversal

§5.3 · **[edit]**

The paragraph recruits all four arms and leans hardest on the two thinking arms
(82%/80% truncation under `none`, i.e. the arms that "cannot finish inside" the
budget). The payoff paragraph then reports the mechanism on `qwen3-1.7b` alone
and never returns.

On those same two arms, on this same task, the same `degree` primer produces
the paper's two largest negative effects — **−61.5** and **−23.5**, both
bolded — plus **−41.5** for `all` on `4b-think`. No sentence in the body
mentions any of them.

Marker rates under `none` (`uses_degree_sum`, pooled over the four densities):
`1.7b` 0.42, `1.7b-think` 0.96, `4b` 0.99, `4b-think` 0.97. Under `degree`:
1.00, 1.00, 1.00, 0.98. So the "7% → 100% procedure switch" is the *minimum*
over all 16 (arm, density) cells; on three of four arms the procedure was
already universal and the primer changes nothing. The abstract's mechanism
claim ("What the primer changes is which procedure the model runs") is
dissociated from the arm that carries the accuracy result.

### 2.11 Define the "global correction"

`:281` · **[edit]**

It gates the paper's headline claims ("a claim that a condition beats both
controls also requires significance under a global correction") and is never
defined — no family, no q, and it is a different test on a different family
from the one §3.5 declares. State both.

### 2.12 `tab:mae` contradicts the flagship cell

`:1160-1180` · **[edit]**

On the arm and task carrying the abstract's headline, the natural metric for a
counting task says the primer does not help. `4b`/`edge_count`, `hit_cap==0`:

| condition | mean abs err | median | p90 | max | exact |
|---|---|---|---|---|---|
| `none` | 39.2 | 19.0 | 101.2 | 371 | 2.0% |
| `degree` | **40.4** | 6.0 | 117.4 | 408 | 29.8% |
| `filler` | **29.3** | 17.0 | 78.2 | 230 | 3.3% |

What `degree` does is **redistribute**, not reduce: median 19→6 and exact
2.0%→29.8%, while the tail gets heavier. Report that — it sharpens the
mechanism claim rather than weakening it. (On `1.7b` the MAE supports the story
hard and the paper under-sells it: 133.4 → 47.8, a 64% reduction, against
filler at 181.2.)

### 2.13 Soften two overgeneralisations

`:463-472`, `:712`, `:490-497` · **[analysis][edit]**

- *"Effects are close to one-directional"* rests on four hand-picked cells at
  one density and is contradicted by the statistic over all cells. Compute the
  fix/break distribution over every cell and report it, or restrict the claim
  and say so.
- *"reverse across model size"* — two sizes, one family. And several of the
  cells demonstrating the reversal are ones §5.1 disqualifies, or ceilings
  §5.6 explains.

### 2.14 The rewiring control has no effect to moderate

§3.3, §5.6, `fig:rewiring`, `tab:prior` · **[edit]**

On the rewiring corpus **not one** `clustering`-vs-`none` contrast is
significant on either 1.7B arm: `qwen3-1.7b` gives +3.5 [−3.0, +10.0] p=0.363,
+6.0 [−1.5, +13.5] p=0.148, +2.5 [−5.0, +10.0] p=0.599; `bh_reject: false`
everywhere. "The benefit stays flat" describes a corpus in which the benefit
was never established. With n=200 per level and CIs ~13 points wide, this
design could not detect a change in a 5.9-point effect if there were one. Say
that, or drop the claim that the rewiring rules the mechanism out.

Against `filler` the benefit is **not** flat: clustering−filler runs +13.5,
+11.0, +6.0 across levels.

### 2.15 A second density-prior test exists and is positive

`superseded/review_checks.json["prior"]["densfull40"]` · **[edit]**

`superseded/scripts/analyze_review_checks.py --question prior` runs **two** tests. The
paper reports only the rewiring one. The other — a paired "does the error point
toward C̄(n−1) more often under `clustering` than under `none`" rate, on the
corpus that supplies the +5.9 headline — is positive on `qwen3-4b`:
**+0.096 [+0.027, +0.178]** (n=73), against `filler` on the same arm at
**−0.046 [−0.092, −0.011]**. A 14-point contrast in exactly the direction the
density-prior account predicts.

§5.6 says "neither mechanism we can test explains it." Report it with the
counterweights: `1.7b` +0.037 [−0.013, +0.090] and `1.7b-think` −0.021
[−0.086, +0.043] are null, n=73 is small, and `filler` itself is significantly
positive on `1.7b-think`.

### 2.16 The route-threshold robustness claim fails at the evaluated densities

§3.2 `:225-233` · **[edit]**

> "every gain is either at most $0.011$ or at least $0.066$, so any threshold
> between these values yields the same split"

True only of bars averaged over all **seven** densities — and four of the six
tasks were never run beyond four. Recomputing over only the densities each task
was evaluated at, three pairs land inside the "empty" gap:
`edge_count`/`clustering` **0.0267**, `connected_nodes`/`all` **0.0142**,
`edge_existence`/`degree` **0.0142**. The upper edge is 0.027, not 0.011, and a
threshold of 0.013 would add two pairs to the route set.

---

## 3. P2 — build and presentation

| # | Item | Fix | Tag |
|---|---|---|---|
| 1 | **Body exceeds 8 pages.** Conclusion starts p8, spills ~10 lines onto p9 above the Limitations heading | Cut §5.1's "Difficulty is set by the answer" ¶ (`:400-415`) — prerequisite framing, and its fixed-degree half is already in `tab:degfixdeg`. Then fix `superseded/paper/make_v3.sh:37-40`: it greps `newlabel{sec:conclusion}`, i.e. where the last section *begins*, so it passes while the body overruns. Assert the Conclusion's **last** page = 8. | [edit] |
| 2 | **Tables 14 and 16 physically overprint on page 14.** The closing bracket of every CI in `tab:lengthcost` — `[-32.0, -1.0]`, `[-14.0, +3.1]`, `[-17.2, +4.0]`, `[+31.0, +31.0]` — is obliterated glyph-on-glyph by Table 16's task names; the header "range" is overprinted by "Task" | `length_cost_table.tex:4,13` set `\multicolumn{4}{l}` headings wider than the four columns they span → 34.7pt overfull (`v3.log:1261`). Shorten them into the caption, or `\resizebox{\columnwidth}{!}` as neighbouring tables do. | [edit] |
| 3 | Same page: 36.96pt vbox overflow drops Table 26's caption onto the folio | Replace `[H]` with `[tb]` in `published_ceiling.tex:1`, `ci_table.tex:13`, `v3.tex:1093` | [edit] |
| 4 | **Table 1 runs 2.1 cm past the right text margin**, to within 4 mm of the paper edge | Cutting the two constant-gold rows (#6) buys the width back | [regen] |
| 5 | **Non-anonymised `[preprint]` mode**; GitHub handles as the only affiliation, no institution or email | `[review]` for submission. Desk-reject condition at any double-blind venue. | [edit] |
| 6 | `node_count` and `cycle_check` in Table 1 | Constant gold at n=40. They supply the single largest number in the paper (+67.3) and contaminate every "largest effect" claim. Move to an appendix measurement table. | [regen][edit] |
| 7 | Three appendix pages of per-task tables **duplicate Table 1's left half verbatim** (every value, including bolding), and their captions declare a **different BH family** — "within the arm" vs Table 1's "within each (arm, task) family and baseline". None is cited from anywhere. | Delete `v3_pertask_tables.tex`, or keep and fix the caption | [edit] |
| 8 | Prose quotes `tab:cnmetric` and `tab:cncontrols`, which are **not `\input`** — the paper cites evidence it does not show (`:355-358`, `:474-482`) | `\input` them or stop quoting their numbers. Also: `:356` says F1 "sits at 0.93–1.00"; `connected_nodes_metric.tex` gives 0.963–0.993, and exact 0.480–0.930 against the prose's 0.48–0.92. | [edit] |
| 9 | Four figures downscaled 3.5–3.9×; five tables `\resizebox`'d to ~4pt | Split or cut. Nothing at 4pt is readable, and a reviewer will say so. | [edit] |
| 10 | `tab:sdt` prints d′ and c while the text says they are not separately identifiable | Cut | [edit] |
| 11 | **No reproducibility statement** — no seeds, hardware, software versions, checkpoint revisions, or code/data availability | Add. Scored checklist item at every ACL venue. | [edit] |
| 12 | `fig:balanced` caption asserts a claim the body refutes two ¶ later, using a series the figure does not plot | Redraw or rewrite the caption | [edit] |
| 13 | §5.3 reports as a headline the exact median its own appendix table refuses to print (`tab:behaviour` suppresses medians above 20% truncation) | Reconcile | [edit] |
| 14 | `fig:serial` caption says "$n{=}700$ per condition"; actual range is 647–700 | Say "647–700" | [edit] |

---

## 4. P3 — citations

- **`fatemi2023talklikeagraph` is cited as an arXiv preprint; it was published
  at ICLR 2024.** It is the paper's foundational citation. `custom.bib:1-6`.
- **Four Gemma 4 arms are reported in `tab:ceiling` and never named or cited**
  in the text, while the `gemma4` bib entry sits orphaned. Cite it or drop the
  arms.
- **`dwivedi2022graph` introduces RW*PE*** (random-walk *positional* encoding),
  not RWSE — that name is GraphGPS (Rampášek et al., NeurIPS 2022). Fix the
  attribution, or describe what `graphtalk/primers.py` actually computes
  (return probabilities at 2 and 3 steps).
- **ER generator**: confirm whether the corpus is G(n,p) (Gilbert 1959) or
  G(n,m) (Erdős–Rényi 1959) and cite accordingly. **[verify]**
- **Related Work omits the closest prior work in all four claim categories**,
  including a NeurIPS 2024 paper on this benchmark by overlapping authors.
- `gelman2006difference` is present and cited zero times — §2.2 fixes this.
- `kamradt2023needle` (a GitHub repo) is load-bearing alongside peer-reviewed
  work for the position claim; a better citation exists.
- Verified correct, do not re-check: `mcnemar1947note`,
  `benjaminihochberg1995controlling`, `geirhos2020shortcut`,
  `liu2024lostinthemiddle`, `levy2024sametask`, `shi2023distracted`,
  `fosdick2018configuring`, `chernozhukov2018double`, `hautus1995corrections`,
  `macmillan2005detection`, `dror2018hitchhikers`, `wang2023nlgraph`,
  `perozzi2024letyourgraph`, `qwen2025qwen3`, `wei2022chainofthought`,
  `dziri2023faithandfate`.

---

## 5. The rewrite: what the paper should claim

The result is in the data and not in the paper. Computed over all four arms of
`densfull40` + `densfull40hi`, dropping the two constant-gold tasks, and
splitting on the paper's own shortcut bars.

**The working window is baseline accuracy 0.25–0.90. 55% of non-leakage cells
(158 of 289) sit above 0.90 and cannot show a gain by construction.**

| mean Δ (pts) | baseline 0.25–0.75 | baseline > 0.90 |
|---|---|---|
| primer **states** the answer | **+11 to +13** | −1.3 |
| primer is **genuine side information** | **+2 to +4** | −0.8 |

Per primer, non-leaky, inside the window (baseline 0.25–0.90):

| primer | mean Δ | median Δ | cells |
|---|---|---|---|
| `all` | **+5.25** | +3.0 | 17 |
| `clustering` | **+2.91** | +4.0 | 28 |
| `degree` | +1.97 | +1.5 | 18 |
| `rwse` | +1.83 | +2.0 | 17 |
| `components` | −0.32 | −0.5 | 28 |

All 25 non-leaky gains ≥8 points have baseline 33–93% (clustered 50–80%);
11 of the 13 losses ≥8 points sit above 83%.

Best single figure available: **`qwen3-4b`, `node_degree`, `degree` primer —
−12 at p=.50 (baseline 99%) and +27 at p=.85 (baseline 33%).** Same arm, same
task, same primer; the sign flips with density purely through baseline. That
demonstrates the paper's own thesis *within one cell* instead of across arms,
and it is immune to the "two sizes, one family" objection.

Proposed thesis: *primers help mainly by leaking the answer, and only where the
model is not already right; genuine structural side information is worth ~3
points in a narrow difficulty window; and more than half of a standard sweep is
uninformative about either.*

Three things that are already strong and should be stated more loudly:

1. **§5.1 is the best section in the paper.** Disqualifying two of your own six
   tasks before reporting a result is rare and correct.
2. **The extraction pipeline is clean and under-claimed.** 99.84% of 84,000
   generations parse; the 0.16% residual is confined to truncated `cycle_check`
   rows. No condition-correlated extraction failure anywhere. That is the third
   confound in the introduction, and most papers in this area never check it.
3. **Two headline effects are density-robust and the paper never says so.**
   `components` on `4b`/`connected_nodes` runs +7.0, +11.0, +10.0, +9.0;
   `degree` on `4b`/`edge_count` runs +22.7, +30.3, +24.2, +33.7. Neither is a
   pooled average hiding one driving cell — which is the failure mode diagnosed
   elsewhere in the paper and the defence never mounted for its own best
   results.

---

## 6. Analyses to run (existing data, no new generations)

| | Analysis | Replaces / adds |
|---|---|---|
| **A** | Per (arm, task): fit Δ_vs_none on `primer_chars` across the five real-content conditions; report the residual as the content effect | Replaces the binary vs-filler contrast (§1.6). Controls all five conditions instead of one, and yields "length costs ≈0 over 37–4,691 chars" as a finding in its own right. |
| **B** | The baseline-window table of §5, with the leaky/non-leaky split from `shortcuts_n40.json` | Replaces the −11.0 interaction (§2.1) |
| **C** | Paired bootstrap of slope(cond) − slope(none), plus BH over the 28 cells | Rebuilds §5.4 (§2.2) |
| **D** | Matched-null simulation for the additivity ratio (recentre to exact additivity, keep n and the screen) | Fixes §2.9 |
| **E** | Model accuracy vs shortcut bar for all route cells | Delivers contribution 1 (§2.5) |
| **F** | Per-density profiles for `components`/`connected_nodes` and `degree`/`edge_count` | Pre-empts the pooling objection where it does not apply — four numbers each |
| **G** | Balanced accuracy for every `edge_existence` cell, not just `none` | The response-bias story is stronger than the accuracy story: hits sit at 98–100% under every condition, so accuracy moves only through false alarms |
| **H** | Fix/break distribution over all cells | Tests §2.13's claim instead of asserting it |
| **I** | `components` split p=.10 (string varies) vs p≥.20 (constant) | §2.6 |
| **J** | Report the second density-prior test already in `superseded/review_checks.json` | §2.15 |

---

## 7. Suggested order of work

1. **§1 (P0)** — mostly text edits plus two generator reruns. Until these are
   done, every number in the paper is suspect.
2. **Analyses A, B, E** — they unlock the rewrite and replace the two results
   that do not survive.
3. **§2 (P1)**, rewriting §5.4 around C and §5.6 around B.
4. **§3 and §4** last — mechanical.

---

## 8. Verified correct — do not re-audit

Checked during this review and found sound:

- The 81/273 vs 117/381 vs 116/381 cell counts are **not** an inconsistency:
  `tab:baseline` is main-sweep-only, `tab:robust` adds the high-density
  extension, and the captions say so.
- §5.1's "70% that terminate" and "82%/80%" are correctly scoped to `none` and
  correctly differ from `tab:hitcap`'s condition-pooled 22.4/81.0/62.2.
- `node_count`'s 0.315 + 68.5% = 1.000 is real arithmetic: every prediction in
  every cell is 39 or 40, no other error type exists.
- Ethics "approximately $200{,}000$ generations": `cat runs/*.jsonl | wc -l` =
  **217,178** over 388 files. "Approximately" covers it.
- "16,800 prompts per arm" (7×6×4×100) and the extension arithmetic are right;
  `frame.csv`'s 84,000 rows = 4 arms × 21,000 including the extension.
- "19,600 `node_degree` items" and the id-confound correlations (+0.019 degree,
  +0.020 clustering) reproduce exactly.
- The per-density slope series −9, −12, −24, −24, −12, −33, −18 reproduces
  exactly once capped rows are dropped, as the generator does.
- `tab:booleanbias` reproduces exactly from the four-density pool, despite
  `NUMBERS.md` mislabelling it as "p=0.50 rows".
- `ci_table.tex` (v1 pipeline) and `v3_main_table.tex` (v3 pipeline) agree in
  all 144 shared cells to within 0.1pp, every difference on a rounding
  boundary.
- All four abstract headline numbers recompute exactly: +26.463, −6.000,
  7%→100%, 273→34 with 99→87, median ratio 0.4422.
- `tab:shortcut` reproduces `shortcuts_n40_flat.json` cell for cell.
- `tab:prior`'s predicted shifts, the rewiring mean-clustering values
  (0.0139/0.3079/0.7347) and the observed shift (<0.3) all check out.
- The effects are **not** noise: expected 3.7 cells at |Δ|≥10 and 0.8 at
  |Δ|≥20 under the null, against 34 and 16 observed.

---

## 9. Housekeeping

An audit agent left `flips.csv` (145 rows, the truncation-convention flip
table) in the repo root, untracked. Delete it or move it to `csv2/one-offs/`.
Nothing else in the working tree was modified by this review.
