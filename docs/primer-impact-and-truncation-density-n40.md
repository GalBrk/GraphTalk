# Primer impact on success rate, and truncation among all rows — density at n=40

Companion to [`primer-impact-and-truncation.md`](primer-impact-and-truncation.md),
same questions (success rate, truncation-as-%-of-all-rows, per primer, and a
range recommendation) but varying **density** instead of **size**. Node count is
pinned at n=40; density (`p`) moves. Source data: a collaborator's checkout,
`/home/dcor/galbarak2/GraphTalk`, scored here directly from the raw run files
with this project's own `graphtalk.scoring.score_one`/`extract_answer` (not
transcribed from the collaborator's own write-up), so the numbers below are
independently computed and cross-checked against their published tables in
`docs/primer-effects-and-power.md` (they agree to within rounding, with one
expected exception noted under Part 1).

## Scope: one task, five primers — not "one table per task"

Unlike the size sweep, a clean, controlled density-at-n=40 grid exists **only
for `node_degree`**, for a reason worth stating plainly rather than glossing
over: every other task was excluded *in the source experiment's own design*,
not by a choice made in writing this document.

- **`edge_count`** truncates too heavily to test at n=40 densities (the same
  reason it's excluded from parts of the size sweep — see the companion doc).
- **`cycle_check`** degenerates — past the sparsest density level, essentially
  every graph has a cycle, so there's no headroom.
- **`node_count`** is flat at 1.0 regardless of density (density doesn't change
  the node list).
- **`edge_existence`** was never run at controlled density.
- **`connected_nodes`** was tried (job `density40`, `none`/`degree` only,
  4 density levels, 100 graphs/level) but is **excluded here**: the source
  document disqualifies it for lack of headroom (already 0.975 at n=40 in the
  size sweep, before this job was even submitted), and it only has 2 of the 5
  primers below. If you find `qwen3-1.7b.density40.shard*of3.jsonl` in
  galbarak2's `runs/`, that's the disqualified slice.

So this document has exactly one task's tables, for both `qwen3-1.7b` and
`qwen3-1.7b-think`, rather than seven.

**Primers actually run at density: `none`, `components`, `clustering`, `degree`,
`filler` — not `rwse` or `all`.** The size sweep's 7-primer set doesn't apply
here; don't read a missing `rwse`/`all` row as a zero.

**Two related datasets exist and are deliberately excluded from the tables
below** (footnoted here so a reader who finds these files on disk isn't left
guessing):
- `degdensrep` (`qwen3-1.7b.degdensrep.shard*of5.jsonl`) — an independent
  replication on **freshly-drawn graphs** (a different seed offset), used by the
  source document to check that the `clustering` effect reproduces. Not
  pairable with the main grid by `instance_id`, so it can't be merged into the
  same cells.
- `degfixdeg` (`qwen3-1.7b.degfixdeg.shard*of5.jsonl`) — a **different
  experimental design**: it varies node count at a *fixed mean degree* (not
  density at a fixed node count), specifically built to separate "density" from
  "edge count / prompt length" as competing explanations. Answers a related but
  different question than this document.

## Data assembled

Both arms cover all 7 density levels (p = 0.10, 0.20, 0.35, 0.50, 0.65, 0.75,
0.85; note p=0.05 and p=1.00 are excluded from the source design — p=1.00 is
the complete graph, a degenerate 1.0 majority-baseline case, and p=0.05 donates
free accuracy on `connected_nodes`-adjacent tasks, not relevant here but kept
consistent with the source design), 400 graphs/level, assembled from multiple
job files whose density ranges are non-overlapping (verified via each row's
`instance_id`, which embeds the density as `.../pX.X/...`):

- **plain (`qwen3-1.7b`)**: `degdens40` (p≤0.50, `none`/`components`/`clustering`)
  + `degdens40hi` (p≥0.65, `none`/`components`/`clustering`/`degree`) +
  `degceil` (`degree`, p≤0.50) + `degdensfill` (`filler`, all 7 levels) — 2,800
  rows per condition.
- **think (`qwen3-1.7b-think`)**: `degdensthink` (all 7 levels ×
  `none`/`components`/`clustering`/`degree`) + `degdensfillT` (`filler`, all 7
  levels) — 2,800 rows per condition.

Row-count sanity check passed: every condition in both arms totals exactly
2,800 rows (400 × 7 levels), confirming no file is double-counted or missing a
level.

## Part 1 — success rate by density, and Δ vs. `none` at that density

**Cross-check against the source document.** Every plain-arm number below
matches the collaborator's own published tables to the reported precision
(e.g. plain `none` at p=0.10/0.20/0.35/0.50 reads 92.2%/76.5%/39.2%/29.5% here
and there). **One expected discrepancy**: the think arm's `none` at p=0.85
reads 48.8% here vs. their published 49.4%. This is not a scoring
disagreement — their `scripts/score_density_sweep.py` **drops** `hit_cap` rows
from the accuracy denominator (scores only completed rows), while this
document scores every row, `hit_cap` or not, as its `primary` metric would
score it (i.e. counts a capped row as wrong if unparseable, consistent with
the size sweep's convention and with the "truncated-rate" metric being a
separate, explicit column rather than folded into accuracy). Recomputing with
their convention (195 correct / 395 completed = 49.4%) reproduces their number
exactly, confirming this is a stated convention difference, not an error.

**Read the bar column against the same caveat as the size sweep**: `degree`'s
bar is **1.00** on `node_degree` — it renders `"Node <queried> has degree
<gold>."` verbatim, so it states the answer rather than aiding reasoning, at
every density level in this table. `components`, `clustering`, and `filler` are
all bar 0.08, identical to `none` — clean at every density.

#### qwen3-1.7b (plain)

| primer | p=0.10 | p=0.20 | p=0.35 | p=0.50 | p=0.65 | p=0.75 | p=0.85 | bar |
|---|---|---|---|---|---|---|---|---|
| none | 92.2% | 76.5% | 39.2% | 29.5% | 14.0% | 9.0% | 5.2% | 0.08 |
| components | 93.8% | 78.2% | 41.5% | 24.5% | 12.2% | 6.5% | 6.2% | 0.08 |
| clustering | 92.5% | 79.8% | 46.8% | 33.8% | 12.0% | 9.5% | 4.8% | 0.08 |
| degree | 95.2% | 85.8% | 61.0% | 36.2% | 12.2% | 12.2% | 5.8% | 1.00 ⚠️ |
| filler | 95.2% | 77.0% | 40.2% | 27.5% | 4.5% | 2.8% | 1.8% | 0.08 |

**Δpp vs `none` at that density**

| primer | p=0.10 | p=0.20 | p=0.35 | p=0.50 | p=0.65 | p=0.75 | p=0.85 |
|---|---|---|---|---|---|---|---|
| components | +1.5 | +1.7 | +2.2 | -5.0 | -1.8 | -2.5 | +1.0 |
| clustering | +0.3 | +3.2 | +7.5 | +4.3 | -2.0 | +0.5 | -0.5 |
| degree | +3.0 | +9.3 | +21.7 | +6.8 | -1.8 | +3.2 | +0.5 |
| filler | +3.0 | +0.5 | +1.0 | -2.0 | -9.5 | -6.2 | -3.5 |

#### qwen3-1.7b-think

| primer | p=0.10 | p=0.20 | p=0.35 | p=0.50 | p=0.65 | p=0.75 | p=0.85 | bar |
|---|---|---|---|---|---|---|---|---|
| none | 93.8% | 87.5% | 63.2% | 58.5% | 43.8% | 44.2% | 48.8% | 0.08 |
| components | 94.0% | 89.8% | 66.5% | 58.0% | 44.0% | 38.8% | 45.2% | 0.08 |
| clustering | 96.0% | 92.5% | 66.5% | 57.8% | 41.8% | 35.5% | 43.0% | 0.08 |
| degree | 97.2% | 94.0% | 81.5% | 79.2% | 71.0% | 70.8% | 81.5% | 1.00 ⚠️ |
| filler | 96.8% | 90.2% | 59.0% | 49.0% | 34.8% | 30.0% | 37.5% | 0.08 |

**Δpp vs `none` at that density**

| primer | p=0.10 | p=0.20 | p=0.35 | p=0.50 | p=0.65 | p=0.75 | p=0.85 |
|---|---|---|---|---|---|---|---|
| components | +0.2 | +2.2 | +3.3 | -0.5 | +0.3 | -5.5 | -3.5 |
| clustering | +2.2 | +5.0 | +3.3 | -0.7 | -2.0 | -8.8 | -5.8 |
| degree | +3.5 | +6.5 | +18.2 | +20.8 | +27.2 | +26.5 | +32.7 |
| filler | +3.0 | +2.7 | -4.2 | -9.5 | -9.0 | -14.3 | -11.2 |

### Reading Part 1

- **`none` collapses with density in both arms, but the think arm collapses to
  a much higher floor.** Plain `none` runs 92.2%→5.2% (p=0.10→0.85); think
  `none` runs 93.8%→48.8% over the same range and never drops below the
  majority baseline (see the recommendation section) — it stays well above
  guessing even at the densest level tested, unlike the plain arm.
- **`clustering`'s clean (bar 0.08) effect is real but small, and concentrated
  in the middle of the range**: +7.5pp at p=0.35, +4.3pp at p=0.50 for the plain
  arm, tracking the same shape (+3.3pp/-0.7pp) for think — the source document's
  own pooled/trend analysis (not reproduced here) finds this the one primer
  effect on this task that's both clean and reaches significance, pooled across
  p≤0.50.
- **`filler`, despite an identical bar (0.08) to `clustering`, goes strongly
  *negative* at high density in both arms** (-9.5pp / -3.5pp plain, -9.0pp /
  -11.2pp think at p=0.65/0.85) — since `filler` carries no structure at all,
  this isolates a pure prompt-length cost that any primer this long would pay,
  separate from content. It's the reason `clustering`'s apparent high-density
  cost against `none` (e.g. -2.0pp plain, -5.8pp think at p=0.85) is at least
  partly a length tax rather than `clustering`'s content actively hurting.
- **`degree`'s huge Δ, especially in the think arm (+18 to +33pp from p=0.35
  up), is exactly the shortcut-bar-1.00 case Part 1's caveat warns about**: it
  measures whether the model consults a fact stated verbatim in the prompt, not
  primer-aided graph reasoning. Read as that, it's still an interesting result
  (the think arm consults the stated fact far more reliably than the plain arm
  does — the plain arm's Δ tops out at +9.3pp even though the answer is written
  down), just not a "primer helps reasoning" result.

## Part 2 — truncation among all rows, by primer × density

Cells show `pct% (trunc/n)`, `n = 400` in every cell; `trunc` = `hit_cap` or
`overflow` (`response is None`).

#### qwen3-1.7b (plain)

| primer | p=0.10 | p=0.20 | p=0.35 | p=0.50 | p=0.65 | p=0.75 | p=0.85 |
|---|---|---|---|---|---|---|---|
| none | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) |
| components | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) |
| clustering | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) |
| degree | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) |
| filler | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) |

#### qwen3-1.7b-think

| primer | p=0.10 | p=0.20 | p=0.35 | p=0.50 | p=0.65 | p=0.75 | p=0.85 |
|---|---|---|---|---|---|---|---|
| none | 1.5% (6/400) | 0.5% (2/400) | 0% (0/400) | 0% (0/400) | 0.2% (1/400) | 0.5% (2/400) | 1.2% (5/400) |
| components | 1.5% (6/400) | 0.5% (2/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 1.2% (5/400) | 2.8% (11/400) |
| clustering | 0.8% (3/400) | 0% (0/400) | 0.2% (1/400) | 0% (0/400) | 0% (0/400) | 0% (0/400) | 0.2% (1/400) |
| degree | 1.0% (4/400) | 1.2% (5/400) | 4.2% (17/400) | 4.8% (19/400) | **11.8% (47/400)** | 9.8% (39/400) | 7.5% (30/400) |
| filler | 1.0% (4/400) | 0.2% (1/400) | 0.2% (1/400) | 0% (0/400) | 0% (0/400) | 0.2% (1/400) | 0.8% (3/400) |

### Reading Part 2

- **The plain arm never truncates at any density, for any primer** — 0/2,800
  across every condition. Density alone, at n=40, doesn't push the plain arm's
  output length anywhere near the token cap; `none`'s collapse in Part 1 is
  entirely a reasoning failure, not a truncation artifact (median output ran
  262-278 tokens at the densest levels per the source document, against a
  2,048-token budget).
- **The think arm's truncation is almost entirely concentrated in the `degree`
  condition at high density** — 11.8% at p=0.65, 9.8% at p=0.75, 7.5% at
  p=0.85, against ≤2.8% for every other primer at every density. `degree`'s
  prompt is the longest of the five (it adds one sentence per node), so this is
  a primer-specific length effect layered on top of the density effect, not
  density-in-general causing truncation for the thinking arm.
- **This matters for reading `degree`'s huge Part-1 Δ at high density**: some
  of the think arm's advantage there is measured on a sample with up to ~12%
  of its rows truncated (and scored as wrong here, per this document's
  convention) — the true gap, scored only on completed rows, would be
  somewhat larger (the source document notes p=0.85's effect moves from
  +34.2pp to +29.5pp depending on whether capped rows are dropped or scored
  zero — smaller either way, same conclusion, but not free to ignore).

## Recommended density range

**Method**, same as the companion size-sweep document: classify `none`'s own
accuracy into ceiling (≥90%)/floor (≤ majority baseline + 2pp)/informative, and
flag truncated if any primer's truncated-rate-of-all-rows ≥10% at that density.
Majority baseline here comes from this run's own 400-graphs-per-level `none`
golds (property of the gold-degree distribution alone, independent of the
shortcut bar): **23.0% / 16.3% / 15.0% / 14.8% / 13.3% / 16.5% / 17.3%** at
p = 0.10 / 0.20 / 0.35 / 0.50 / 0.65 / 0.75 / 0.85.

| p | plain zone | think zone |
|---|---|---|
| 0.10 | ceiling (92%) | ceiling (94%) |
| 0.20 | informative | informative |
| 0.35 | informative | informative |
| 0.50 | informative | informative |
| 0.65 | floor (14% vs 13%+2) | informative + **trunc** (`degree`, 11.8%) |
| 0.75 | floor (9%) | informative |
| 0.85 | floor (5%) | informative |

**Recommendation: p = 0.20-0.50 is the range that's clean for both arms; the
thinking arm alone stays informative all the way to p = 0.85, but with the
`degree` condition specifically flagged for truncation from p = 0.65 up.**

- **p = 0.10 is a ceiling point for both arms** (92-94%) — too easy, no room for
  a primer to separate from `none`. Keep it as a reference point (shows where
  the task starts), not for detecting an effect.
- **p ≥ 0.65 is where the plain arm floors** — `none` falls to 14%/9%/5%, at or
  below the ~13-17% majority baseline, and per the source document's failure
  analysis the model isn't merely failing to compute there, it's actively
  defaulting to "this node connects to everyone" (predicting the complete-graph
  degree in 24-61% of rows) rather than reading the graph. A primer can't help
  a model that has stopped reading the graph; don't test the plain arm past
  p = 0.50 for a primer-effect question.
- **The thinking arm stays informative through p = 0.85** — `none` never drops
  below ~44%, comfortably clear of the ~13-17% floor at every density tested.
  If the project wants density coverage specifically for the thinking arm, the
  usable range is wider (p = 0.20-0.85) than for the plain arm — but the
  `degree` condition inside that extended range (p ≥ 0.65) carries a real
  truncation cost (7.5-11.8%) that isn't present for `none`/`components`/
  `clustering`/`filler` at the same densities, so a `degree`-vs-`none`
  comparison specifically should still be read cautiously past p = 0.50.
- **`clustering`'s clean, bar-0.08 effect — this run's one primer result that's
  both content-clean and reaches significance in the source document's own
  analysis — peaks exactly inside the p = 0.20-0.50 recommended window**
  (+7.5pp at p=0.35, the single largest clean Δ in either arm's table), which
  is corroborating evidence that this range is genuinely where "room to see a
  change" exists, not just where the ceiling/floor arithmetic happens to land.
- **The `degree` condition's large Δs at every density are shortcut-bar 1.00,
  not primer-aided reasoning** (see Part 1) — this applies uniformly across the
  whole recommended range and isn't something a different density choice can
  fix. `components`, `clustering`, and `filler` are bar 0.08 (clean, identical
  to `none`) at every density tested, so their Δs inside the recommended
  window aren't subject to the same caveat.
- **This is one task (`node_degree`), one model family, one graph generator
  (ER-only).** Nothing here says a different task would show the same
  p = 0.20-0.50 sweet spot — the companion size-sweep document's task-by-task
  variation (some tasks floor by n60, some stay informative to n80) suggests
  the density picture would likely vary by task too, if it were ever measured
  for a task other than `node_degree`.
