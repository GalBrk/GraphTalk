# `node_degree`: size vs. density, plain vs. thinking

Two separate experiments bear on the same task (`node_degree`, `qwen3-1.7b` /
`qwen3-1.7b-think`), varying two different things:

- **Part A — size sweep** (this project, `runs/qwen3-1.7b*.node_degree_n{20,40,60,80}.jsonl`):
  node count varies (20/40/60/80), graph density is *not* controlled — each graph draws
  its Erdős–Rényi edge probability uniformly from `[0, 1]` (`--graph-source diverse`
  default), so each size's 30-graphs-per-condition sample is a mix of easy (sparse) and
  hard (dense) graphs.
- **Part B — density sweep** (a collaborator's checkout,
  `/home/dcor/galbarak2/GraphTalk`, `docs/primer-effects-and-power.md` "Density at a
  fixed size"): node count is *pinned* at n=40, density is pinned per level
  (0.10–0.85, no sampling), 400 graphs/level — density is isolated as the only moving
  variable.

Numbers in Part A were scored directly from the raw run files in this repo (`hit_cap`/
`overflow` rows excluded from the success-rate denominator's correct count, but included
in the truncated-rate table). Numbers in Part B are transcribed from the collaborator's
already-analyzed and committed report, not re-scored here.

**Completeness note (Part A):** `qwen3-1.7b` is complete at all four sizes (n=30 per
condition). `qwen3-1.7b-think` is complete at n20/n60 but **still in flight** at n40
(14–15/30 rows per condition) and n80 (24–25/30 rows per condition) as of this write-up —
those cells are marked with their partial `n`.

## Part A — size sweep (uncontrolled density, this project)

### Success rate (exact match, `hit_cap`/`overflow` rows scored as wrong)

**qwen3-1.7b**

| condition | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 96.7% | 86.7% | 56.7% | 53.3% |
| components | 96.7% | 80.0% | 63.3% | 53.3% |
| degree | 96.7% | 86.7% | 66.7% | 56.7% |
| clustering | 90.0% | 76.7% | 63.3% | 50.0% |
| rwse | 96.7% | 86.7% | 76.7% | 53.3% |
| filler | 93.3% | 76.7% | 63.3% | 50.0% |
| all | 96.7% | 83.3% | 73.3% | 63.3% |

**qwen3-1.7b-think** *(n40/n80 partial — jobs still running)*

| condition | n20 | n40 (n=14–15) | n60 | n80 (n=24–25) |
|---|---|---|---|---|
| none | 96.7% | 71.4% | 70.0% | 54.2% |
| components | 93.3% | 73.3% | 66.7% | 54.2% |
| degree | 100.0% | 100.0% | 83.3% | 50.0% |
| clustering | 96.7% | 86.7% | 66.7% | 45.8% |
| rwse | 96.7% | 78.6% | 76.7% | 45.8% |
| filler | 93.3% | 85.7% | 60.0% | 45.8% |
| all | 100.0% | 93.3% | 70.0% | 64.0% |

### Truncated rate (`hit_cap` + `overflow`, of total rows per cell)

**qwen3-1.7b**

| condition | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 0.0% | 0.0% | 0.0% | 20.0% (6/30) |
| components | 0.0% | 0.0% | 0.0% | 16.7% (5/30) |
| degree | 0.0% | 0.0% | 0.0% | 16.7% (5/30) |
| clustering | 0.0% | 0.0% | 0.0% | 16.7% (5/30) |
| rwse | 0.0% | 0.0% | 0.0% | 16.7% (5/30) |
| filler | 0.0% | 0.0% | 0.0% | 16.7% (5/30) |
| all | 0.0% | 0.0% | 0.0% | 16.7% (5/30) |

**qwen3-1.7b-think** *(n40/n80 partial)*

| condition | n20 | n40 | n60 | n80 |
|---|---|---|---|---|
| none | 0.0% | 0.0% | 0.0% | 20.8% (5/24) |
| components | 0.0% | 0.0% | 0.0% | 20.8% (5/24) |
| degree | 0.0% | 0.0% | 0.0% | 25.0% (6/24) |
| clustering | 0.0% | 0.0% | 0.0% | 25.0% (6/24) |
| rwse | 0.0% | 0.0% | 3.3% (1/30) | 33.3% (8/24) |
| filler | 6.7% (2/30) | 0.0% | 0.0% | 25.0% (6/24) |
| all | 0.0% | 0.0% | 16.7% (5/30) | 28.0% (7/25) |

### Reading Part A

- **Truncation is a step function, not a slope.** It is exactly 0% at n20/40/60 for
  `qwen3-1.7b`, then jumps to 17–20% at n80 across every condition — the token-budget
  ceiling was fine until n80, then binding everywhere at once, not gradually.
- **The think arm starts truncating one size class earlier and worse** — `rwse`
  already shows 1 truncated row at n60 and climbs to 33% at n80, the worst cell in
  either table; `all` also picks up truncation at n60 (16.7%) where the plain arm has
  none.
- **Accuracy falls with size in both arms regardless of primer** — n20 is near-ceiling
  (90–100%) for every condition in both arms; by n80 every condition in both arms is
  in the 45–64% band. This is the same qualitative collapse Part B finds by raising
  density instead of size — both variables make the task harder, and (per this
  project's own `docs/plans/scale-vs-topology-investigation.md`) they are correlated
  in the uncontrolled corpus, since denser graphs also produce longer prompts.
- **No primer is consistently better than `none` at any size in the plain arm** — the
  ranking of conditions reorders at almost every size (`rwse` best at n60/n40,
  `all` best at n80 and n60-adjacent, `clustering` worst at n20/n40/n80). With n=30 per
  cell this is likely noise, not a reproducible primer effect — consistent with Part
  B's finding that `components` is inert and `clustering`'s true effect is only a few
  points once prompt length is controlled for.
- **At n80, `qwen3-1.7b-think`'s `degree` accuracy (50.0%) is *below* `none` (54.2%)**
  — with 25% of that cell truncated, this is likely truncation noise rather than a
  real reversal; not a claim to trust at this sample size.

## Part B — density sweep (n=40 fixed, `qwen3-1.7b` / `-think`, from galbarak2's checkout)

Source: jobs 866467 (plain, p≤0.50) + 866492 (plain, p≥0.65) + 866578 (think, all 7
levels), 400 graphs/density level — an order of magnitude more graphs per cell than
Part A's n=30.

### Success rate (exact match)

**qwen3-1.7b (plain)**

| p (density) | edges | none | components | clustering | degree |
|---|---|---|---|---|---|
| 0.10 | 78 | 92.2% | 93.8% | 92.5% | — |
| 0.20 | 153 | 76.5% | 78.2% | 79.7% | — |
| 0.35 | 269 | 39.3% | 41.5% | 46.8% | — |
| 0.50 | 390 | 29.5% | 24.5% | 33.8% | — |
| 0.65 | — | 14.0% | 12.2% | 12.0% | 12.2% |
| 0.75 | — | 9.0% | 6.5% | 9.5% | 12.2% |
| 0.85 | — | 5.2% | 6.2% | 4.8% | 5.8% |

**qwen3-1.7b-think**

| p (density) | none | components | clustering | degree |
|---|---|---|---|---|
| 0.10 | 95.2% | 95.4% | 96.7% | 98.2% |
| 0.20 | 87.7% | 89.9% | 92.5% | 94.7% |
| 0.35 | 63.2% | 66.5% | 66.7% | 83.0% |
| 0.50 | 58.5% | 58.0% | 57.8% | 81.6% |
| 0.65 | 43.9% | 44.0% | 41.7% | 74.2% |
| 0.75 | 44.2% | 38.7% | 35.5% | 74.2% |
| 0.85 | 49.4% | 46.5% | 43.1% | 84.6% |

`degree` states the queried node's degree verbatim in the primer text — its shortcut
bar on `node_degree` is 1.00, so its accuracy measures *whether the model uses a
stated fact*, not primer-aided reasoning. `rwse`/`filler` were not run at every density
level in this sweep (`filler` ran as a separate length-control job, see below); `all`
was not run.

### Truncated rate (`hit_cap`)

| arm | density range | truncated rate |
|---|---|---|
| plain, `none`/`components`/`clustering`, p≤0.50 | 0.10–0.50 | 0% (0/4,800) |
| plain, all four conditions, p≥0.65 | 0.65–0.85 | 0% (0/1,200 per level) — median output 262–278 tokens, max 732 |
| think, `none`/`components`/`clustering`, all 7 levels | 0.10–0.85 | 1.2%–3.1% |
| think, `degree`, all 7 levels | 0.10–0.85 | up to **12%** at p=0.65 (30–47 rows/cell at high density — `degree` has the longest prompts) |

### Reading Part B

- **Density collapses accuracy at fixed size, same shape as size collapses it at
  uncontrolled density**: plain `none` runs 0.922 → 0.052 as p goes 0.10 → 0.85 — a
  steeper, cleaner version of Part A's n20→n80 decline, with density isolated as the
  only cause (mean absolute error moves the same way: 0.09 → 0.32 → 1.02 → 1.88).
- **The thinking arm is close to immune.** At p=0.85 it scores 0.494 against the
  plain arm's 0.052 — paired on identical prompts, `think − plain` is positive at
  *every* density and grows with it: +0.029 (p=0.10) up to **+0.500** (p=0.85), all
  seven levels significant at p<0.002.
- **`degree` (the verbatim-answer control) separates "can't see it" from "won't use
  it."** Plain: worth only +0.7pp pooled at high density (p=0.59, not significant) —
  the model barely benefits from having the answer written down. Think: worth
  **+20.5pp pooled** (p<0.0001), rising to +34.2pp at p=0.85. Conclusion in the source
  document: the plain model's high-density failure is a *refusal* to consult a stated
  fact while enumerating and miscounting, not an inability to read it — reasoning mode
  is what lets it stop and check.
- **`components` is inert at every density** (pooled effect ≈0, p≈0.95 in the source
  document); **`clustering`'s usable-range effect (p≤0.50) is genuine but small**
  (+3.8pp plain p=0.0017, +2.3pp think p=0.036) and turns harmful against `none` on
  dense graphs in the thinking arm — though the source document's `filler`
  (length-only) control found that reversal is mostly a *prompt-length* tax
  (−11.7pp on dense graphs for any added text) rather than `clustering`'s content,
  which still helps by +5–9pp once length is controlled for.
- **Truncation is asymmetric between arms in the density sweep, same as Part A**:
  effectively 0% for plain at every density tested, but real and growing for think —
  concentrated in `degree` (the longest-prompt condition) specifically, reaching 12%
  at p=0.65. The source document explicitly flags this as non-neutral: dropping vs.
  scoring capped rows as zero changes the p=0.85 `degree` effect from +34.2pp to
  +29.5pp — smaller, same conclusion, but not free to ignore.

## Combined takeaways

| | Part A (size, mixed density) | Part B (density, fixed n=40) |
|---|---|---|
| plain `none` at the easiest cell | 96.7% (n20) | 92.2% (p=0.10) |
| plain `none` at the hardest cell | 53.3% (n80) | 5.2% (p=0.85) |
| think `none` at the hardest cell | 54.2% (n80, partial) | 49.4% (p=0.85) |
| plain truncation, hardest cell | 16.7–20.0% (n80) | 0% (p=0.85) |
| think truncation, hardest cell | 20.8–33.3% (n80, partial) | up to 12% (`degree`, p=0.65) |

Both size and density are real, independent difficulty knobs on `node_degree`, and
both hit the plain arm far harder than the thinking arm. The clean, controlled Part B
result is the trustworthy one for reading *primer* effects (large n per cell, density
isolated from size, a `filler` control for prompt length); Part A's own primer
rankings at n60/n80 should be read as noisy given n=30/cell and uncontrolled density —
useful for confirming the collapse itself and for sizing the token-budget fix, not for
claiming a primer effect at this sample size.

**Caveat carried over from Part B's own document:** all of the above is one task
(`node_degree`) and one model family (`qwen3-1.7b`), on Erdős–Rényi graphs only — the
one task/model pairing that cleared both the shortcut-bar filter and the headroom
filter at n=40. Part A extends the same task to a size axis Part B doesn't cover, but
neither extends to a different task or generator family.
