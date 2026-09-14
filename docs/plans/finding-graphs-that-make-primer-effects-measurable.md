# Finding the graphs that make primer effects measurable

> Self-contained brief. Repo: `C:\year3\ml-with-graphs\final\GraphTalk`, branch `main`.
> Read `CLAUDE.md` first. This merges two earlier plans: the "Smart Hybrid" grid search
> (whose infrastructure is already built and is reused as-is) and a structural-manipulation
> design (which supplies the grid basis, a free pre-GPU screen, and the final experiment).

## Goal

A clean, well-powered, shortcut-free demonstration of whether a primer helps `qwen3-1.7b` /
`qwen3-1.7b-think` answer GraphQA questions.

**Scope: only the graphs change.** Primers, tasks, prompts, encoders and scoring stay exactly as
they are. The `CLAUDE.md` "one renderer" invariant is not touched.

## Background an implementer needs

In Erdős–Rényi there are only two free parameters, `n` and `p`. Everything else is derived:

- **edge count** `m ≈ p·C(n,2)` → sets **prompt length** (the `incident` encoder emits
  `O(n + 2|E|)`: one line per node, listing every neighbour)
- **mean degree** `k = p·(n−1)` → sets **answer magnitude** for `node_degree`

### What the repo's own runs already settle

- **Density is not a difficulty driver.** Job 871262 held mean degree fixed and varied `n`, so
  density fell 8× while edges grew 8×. Accuracy fell monotonically in both blocks
  (0.912 → 0.624 at k=8; 0.575 → 0.224 at k=16). Had density driven difficulty it should have
  *risen*. Every "denser is harder" reading in the docs is density standing in for edge count.
- **Size matters only through edges.** `node_count` does not degrade at n=80; `node_degree`
  falls to 0.143 — it is the task whose work scales with edges.
- **Two things do drive difficulty:** edge count / prompt length, and answer magnitude. At
  matched edges, k=8 beats k=16 every time (640 edges: 0.624 vs 0.225).
- **Much of "difficulty" is a reading failure, not reasoning.** Job 871909 found the same
  collapse on a **graph-free** list of statements: perfect retrieval to ~1,500 tokens, then at
  6,305 tokens the model reads the *middle* only 19.5% of the time vs ~89% at the ends.
- **Length taxes primers directly.** `filler` (content-free, length-matched) costs −6.3pp pooled
  and −11.7pp on dense graphs. Any primer effect smaller than that is inside the noise floor of
  its own length.
- **Four of six tasks are already decided by a graph-blind solver** (`shortcuts.json`):
  `node_count`, `edge_count`, `node_degree` and `cycle_check` all have a condition scoring
  bar = 1.00. Read every effect against `bar(cond) − bar(none)`, never against zero.

### Two corrections this plan makes to the "Smart Hybrid" plan

**(1) The grid basis.** The Smart Hybrid grid is 5 sizes × 7 densities. In the `(n, p)` plane,
moving along *either* axis moves both `k` and `m` simultaneously — so **no cell-to-cell
comparison holds either causal driver fixed**. That is precisely the confound job 871262
falsified. Re-basing to `(k, m)` is a change of coordinates over the same plane, costs nothing,
and makes rows hold answer magnitude fixed while columns hold prompt length fixed.

**(2) The sample-size formula.** `required_n_closed_form` implements `N ≈ 7.84/δ`, which is the
`ψ → δ` limit — it assumes *every* discordant pair favours the primer. Job 871262 measured
discordance `ψ ≈ 0.30`. Against a proper McNemar sizing:

| δ | `required_n_closed_form` | proper (ψ=0.30) | undersized by |
|---|---|---|---|
| 3.3pp | 238 | 2,160 | 9.1× |
| 5.0pp | 157 | 940 | 6.0× |
| 10.0pp | 78 | 233 | 3.0× |
| 13.3pp | 59 | 131 | 2.2× |

**Use `required_sample_size_clustered` for every real sizing decision.** Treat
`required_n_closed_form` as the optimistic anchor its docstring says it is, and never quote it
as the headline number.

## Stage 0 — the free screen (NEW, no GPU time)

Two cell properties are computable from graph samples alone, before any model call:

- **`clu_sd`** — across-node spread of the *rendered, 2-dp* clustering value. This is literally
  what the clustering primer says. If it is ~0, the primer is a constant string and the cell
  cannot produce a primer effect no matter how many graphs you run.
- **`maj_base`** — frequency of the modal degree = what a blind guesser scores on `node_degree`.

Measured over the Smart Hybrid grid (80 graphs/cell), `clu_sd`:

| n \ p | 0.10 | 0.20 | 0.35 | 0.50 | 0.65 | 0.75 | 0.85 |
|---|---|---|---|---|---|---|---|
| 10 | 0.037 | 0.191 | 0.247 | 0.185 | 0.110 | 0.067 | 0.045 |
| 20 | 0.130 | 0.198 | 0.124 | 0.075 | 0.043 | 0.030 | 0.019 |
| 40 | 0.143 | 0.098 | 0.051 | 0.033 | 0.022 | 0.015 | 0.009 |
| 60 | 0.106 | 0.054 | 0.032 | 0.022 | 0.014 | 0.010 | 0.006 |
| 80 | 0.073 | 0.041 | 0.024 | 0.016 | 0.011 | 0.008 | 0.005 |

Under a joint screen (`clu_sd ≥ 0.08` and `maj_base ≤ 0.25`), **2 of 35 cells pass** (n=40/p=0.20
and n=60/p=0.10). Nothing at p ≥ 0.50 survives. The exact count depends on where the bar is set,
but the structure does not: **across the whole ER plane, primer informativeness and task
difficulty are anti-correlated.** Every cell with `clu_sd ≥ 0.10` sits at `k ≤ 6.7` — i.e. easy.

This screen is cheap and prunes most of the grid. Run it first, report it, and let it inform —
not silently override — the cell list.

## Stage 1 — scout, on a re-based grid

Same mechanism as the Smart Hybrid scout (plain arm, `none` only, cheap), but over `(k, m)` cells
rather than the `(n, p)` cross product. `n = 2m/k`, `p = k/(n−1)`:

| k | m | n | p | chars | maj_base | clu_sd | `none` |
|---|---|---|---|---|---|---|---|
| 4 | 160 | 80 | 0.0506 | 3,952 | 0.232 | 0.110 | ~0.85 est |
| 4 | 320 | 160 | 0.0252 | 8,237 | 0.222 | 0.074 | ~0.80 est |
| 8 | 160 | 40 | 0.2051 | 2,594 | 0.214 | 0.090 | **0.728 meas** |
| 8 | 320 | 80 | 0.1013 | 5,237 | 0.185 | 0.071 | **0.645 meas** |
| 8 | 640 | 160 | 0.0503 | 11,136 | 0.167 | 0.052 | **0.624 meas** |
| 12 | 320 | 53 | 0.2308 | 4,251 | 0.183 | 0.056 | ~0.52 est |
| 12 | 640 | 107 | 0.1132 | 8,805 | 0.153 | 0.044 | ~0.42 est |

Char counts are computed from the actual `incident` encoding and **validated against job 871262's
reported corpus sizes to within 1%** (n=80/k=8: 5,237 vs 5,254 reported; n=160/k=8: 11,136 vs
11,109; n=40/k=16: 3,796 vs 3,826). The k=8 row has measured `none` values, so the grid is
anchored and **replicates 871262 as a side effect** — if that row does not reproduce, stop and
investigate before interpreting anything else.

**k=16 is excluded, and that is a finding, not a convenience.** The cell that produced the largest
clustering gain in the project (+0.165 at n=20/k=16) has `maj_base = 0.333` and `clu_sd = 0.020` —
a near-silent primer in front of a task a blind guesser wins a third of the time.

Keep the Smart Hybrid's discipline verbatim: relaxed scout thresholds (drop only on truncation
> 30% or ceiling > 98%; floor rule standard), dropped cells written to `archive_filtered.csv`
never deleted, `--force-include` to bring any back.

## Stage 2 — full reporter, plus multiplicity control (NEW)

Unchanged from Smart Hybrid: all 7 conditions, both arms, per-cell derived N, zone /
shortcut-headroom / truncation / delta / CI / `arm_divergence` reported and nothing dropped.

**Add pre-registration and correction.** Stage 2 is roughly (surviving cells) × 6 non-`none`
conditions × 2 arms of tests — at 40 surviving cells that is ~480 comparisons, giving ~24 false
positives at α = 0.05 by construction. The repo already has one documented family-shopping
incident (a `components` density-slope at p = 0.017–0.022 that did not survive an honestly
specified family). Before running:

- Declare the **confirmatory** test: `clustering` vs `none` on `node_degree` at k=8/m=320
  (`hit_cap` rows dropped). This is the only zero-bar-delta cell in the suite — `clustering` and
  `none` both score 0.08 — so none of an effect there can be shortcut-explained.
- Everything else is **exploratory**, corrected with the existing
  `significance.benjamini_hochberg` within its declared family.
- Sizing comes from `required_sample_size_clustered`, not the closed form.

Not naming a "recommended range" in the output is fine and worth keeping — but **the decision
rule should be written down before the run** even though the decision stays with the user.

## Stage 3 — the structural experiment (NEW, this is the payoff)

Stages 0–2 locate a working difficulty cell. Stage 3 is what they exist to set up.

The ER plane cannot escape its own trade-off: the knobs that make the task hard also silence the
primers. **Degree-preserving rewiring escapes it.** Take a generated ER graph and apply repeated
double-edge swaps, accepting or rejecting each on whether it creates triangles. Measured at
n=80, m=334:

| graph | n | m | transitivity | clu_sd | degree sequence | encoding length |
|---|---|---|---|---|---|---|
| ER as generated | 80 | 334 | 0.099 | 0.067 | — | 5,332 chars |
| rewired low | 80 | 334 | **0.000** | 0.000 | identical | 5,332 chars |
| rewired high | 80 | 334 | 0.364 | 0.137 | identical | 5,332 chars |
| rewired high (longer) | 80 | 334 | **0.485** | **0.206** | identical | 5,332 chars |

Prompt length is invariant to **0.00%**, density is fixed, and the degree sequence is preserved
**exactly** — so every `node_degree` gold answer is literally unchanged and `maj_base` is
unchanged. The only thing that moves is triangle structure, i.e. exactly what the clustering
primer reports. This yields a **within-instance paired design: the same graph, rewired, asked the
same question, with the same answer.**

It also decouples the two properties the ER plane welds together: `clu_sd = 0.206` at `k = 8.35`,
whereas nowhere in the ER plane does `clu_sd ≥ 0.10` occur above `k = 6.7`.

**Sharp interpretation, decided in advance.** If `none` accuracy is flat across rewiring levels
while the clustering effect grows → causal evidence about primer *content*. If `none` also moves →
the finding becomes "clustering changes the task", which is real but a different claim. The `none`
arm decides which, and that must be stated before the run rather than after.

**Secondary (run only if the above works): component count.** Plant disjoint ER blocks at the same
`k`, so the degree distribution is unchanged. Measured at n=80, k=8:

| structure | m | components | degree sd | maj_base | chars |
|---|---|---|---|---|---|
| 1 component | 313 | 1 | 2.71 | 0.175 | 5,162 |
| 2 blocks of 40 | 348 | 2 | 2.55 | 0.212 | 5,446 |
| 4 blocks of 20 | 315 | 4 | 1.95 | 0.212 | 5,173 |

This is the only way to make the `components` primer informative without leaving ER-style
generation — in a single ER graph `P(connected) = 1.00` at every difficulty worth testing, which
is why every `components` result so far has been null by construction. Edge counts must be pinned
to match exactly (the 2-block draw came out 11% high).

**Two negative results already established — do not spend effort here.** Designed degree sequences
do *not* beat ER's `maj_base` (uniform 4–12 gives 0.175, identical to ER's 0.175; the
configuration model's multi-edge collapse eats the gain). Random regular graphs are disqualified
outright: every answer is 8, `maj_base` = **1.000**.

### Why this must be generated, not sliced from existing data

`scripts/extract_graph_topology.py` and `analysis/topology_drivers_report*.csv` already stratify
post-hoc on topology features (degree_std low/mid/high → +0.168/+0.319/+0.473; density
low/mid/high → +0.231/+0.348/+0.386). That analysis is what produced the "density drives
difficulty" claim job 871262 later falsified. In a natural ER corpus these features are all
mutually correlated, so stratification cannot identify any of them. Matched-pair *generation* is
the fix, and rewiring is its strongest form: the pair is the same graph.

---

# Implementation

## Reuse exactly as-is — all verified present, do not rewrite

| file | lines | what to use |
|---|---|---|
| `graphtalk/range_search.py` | 199 | `wilson_interval`, `scout_decision`, `zone_decision`, `shortcut_clean(bar=0.10)`, `truncation_clean(threshold=0.10)`, `classify_cell`, `arm_divergence` |
| `graphtalk/significance.py` | 640 | `paired_permutation_test_clustered`, `cluster_bootstrap_ci_clustered`, `minimum_detectable_effect_clustered`, **`required_sample_size_clustered`**, `benjamini_hochberg` |
| `scripts/score_density_size_grid.py` | 529 | `--stage scout\|full` scorer/reporter |
| `cluster/run_density_size_sweep.sh` | 213 | `--stage scout\|full`, `--force-include`, `--dry-run`; `SIZES`/`DENSITIES`/`TASKS`/`SCOUT_COUNT` env vars |
| `graphtalk/diverse_corpus.py` | 152 | `build_pool(..., algorithms=("er",))` — **already implemented**, with seed offsets keyed to `ALGORITHMS.index()` so filtering never changes a surviving algorithm's draw |
| `scripts/build_prompts.py` | 479 | `--graph-source diverse --algorithms er --node-count N --er-min-sparsity P --er-max-sparsity P --conditions ... --tasks ... --count N` |

`--algorithms` is already wired end-to-end (`build_prompts.py:387-393, 409-412, 428-430`;
`diverse_corpus.py:37, 78-87`). The earlier plan listed this as a gap — it is not.

## Change 1 — run the grid on `(k, m)` pairs, not the cross product

`run_density_size_sweep.sh` iterates `for N in $SIZES; for P in $DENSITIES` — a cross product. A
`(k, m)` grid is a *list of specific pairs*, not a cross product. Two options:

**Option A (zero code change, recommended to start).** Invoke once per pair with singleton env vars:

```bash
# k=4
SIZES="80"  DENSITIES="0.0506" cluster/run_density_size_sweep.sh --stage scout
SIZES="160" DENSITIES="0.0252" cluster/run_density_size_sweep.sh --stage scout
# k=8  (anchor row -- must reproduce 0.728 / 0.645 / 0.624)
SIZES="40"  DENSITIES="0.2051" cluster/run_density_size_sweep.sh --stage scout
SIZES="80"  DENSITIES="0.1013" cluster/run_density_size_sweep.sh --stage scout
SIZES="160" DENSITIES="0.0503" cluster/run_density_size_sweep.sh --stage scout
# k=12
SIZES="53"  DENSITIES="0.2308" cluster/run_density_size_sweep.sh --stage scout
SIZES="107" DENSITIES="0.1132" cluster/run_density_size_sweep.sh --stage scout
```

**Option B (cleaner if the loop is run often).** Add an optional `PAIRS="80:0.1013 160:0.0503 …"`
env var to `run_density_size_sweep.sh` that, when set, replaces the nested loop; and a matching
`--pairs n:p [n:p …]` to `scripts/score_density_size_grid.py` alongside `--sizes`/`--densities`.
Keep both existing flags working unchanged.

Note `--density-tolerance` defaults to `0.05`, which is larger than several of the `p` values
above. **Set `--density-tolerance 0.005` or smaller** for this grid, or `verify_density` will
wave through corpora that are not the requested cell.

## Change 2 — new `scripts/screen_grid.py` (Stage 0, no GPU)

Standalone. Generates sample graphs per candidate cell and reports the pre-GPU screen. No model,
no prompts.

```
--pairs n:p [n:p ...]        # or --sizes/--densities for a cross product
--samples 80                 # graphs per cell
--clu-sd-min 0.08            # report-only threshold
--maj-base-max 0.25
--out analysis/grid_screen.csv
```

Per cell emit: `n, p, k, m, chars, clu_sd, maj_base, n_components, p_connected, screen_pass`.
Use `nx.gnp_random_graph`, round clustering to 2 dp (matching `primers._fmt`) before taking the
spread, and build the encoding string the way `incident_encoder` does for the char count.
**Report, never auto-drop** — same discipline as `archive_filtered.csv`.

## Change 3 — new `graphtalk/structured_corpus.py` (Stage 3)

Confined to graph construction. Nothing downstream changes.

```python
def rewire_to_clustering(graph, target_transitivity, rng, max_swaps=20000):
    """Biased double-edge swaps toward/away from triangles.

    Preserves the degree sequence EXACTLY -- assert this before returning.
    Returns a graph with identical n, m and degree sequence, differing only
    in triangle structure. Reaches transitivity 0.000-0.485 at n=80, m=334.
    """

def plant_components(n, mean_degree, blocks, rng, target_edges=None):
    """`blocks` disjoint ER blocks of n/blocks nodes at the same mean degree.

    Pin total edge count to `target_edges` so arms are length-matched --
    an unpinned 2-block draw came out 11% high in testing.
    """
```

Reuse unchanged: `talk_like_a_graph.graph_generators.generate_graphs`
(`graph_generators.py:21-147`), `graphtalk.graphqa.canonical` and `graphtalk.graphqa.gold_answer`
(`graphqa.py:67-96`), `graphtalk.primers.render_primer`, `graphtalk.prompts`.

Wire into `build_prompts.py` as a `--graph-source structured` path with
`--rewire-transitivity FLOAT` and `--plant-blocks INT`, mirroring how `--graph-source diverse` is
dispatched (`build_prompts.py:410-414`).

## Sizing

- **Stage 0**: free.
- **Stage 1 scout**: 7 cells × 6 tasks × `SCOUT_COUNT` (40) graphs, plain arm, `none` only.
- **Stage 2 full**: surviving cells × 7 conditions × 2 arms, N per cell from
  `required_sample_size_clustered` at ψ from that cell's own scout discordance (871262 measured
  ~0.30; expect 233+ per cell for a 10pp effect, not 78).
- **Stage 3**: 3 rewiring levels × N × 5 conditions at the single best Stage 2 cell, plus the
  same for 3 component levels if Stage 3a works.

## Verification — run before spending GPU time

1. **Tests.** `rewire_to_clustering` preserves the degree sequence exactly and leaves every
   `node_degree` gold answer unchanged; rendered encoding length is **byte-identical** before and
   after (this is the claim Stage 3 rests on); `plant_components` hits `target_edges` exactly;
   `screen_grid.py` reproduces the `clu_sd` table above within sampling error.
   Run `uv run --no-sync pytest -q` — per `CLAUDE.md` the count must be the existing total plus
   the new tests; a different number means the environment is wrong, not the code.
2. **Corpus audit per cell before generating**: realised `k`, `m`, char length (against the Stage 1
   table), `maj_base`, `clu_sd`, component count. Reject any cell with `maj_base > 0.25`.
3. **Anchor check.** The k=8 row must reproduce 871262's 0.728 / 0.645 / 0.624. If not, stop.
4. **Token budget.** Histogram prompt tokens against the 2,048 plain / 8,192 think budgets.
   `ModelSpec.max_context_tokens` is a no-op for every model — do not rely on it.
5. **Reading results.** Drop `hit_cap` rows for primary comparisons but report the non-termination
   rate separately (that convention flipped `rwse` from p=0.11 to p=0.025). Read every effect
   against `bar(cond) − bar(none)` from `shortcuts.json`. `filler` bounds the length tax; a
   non-zero `components` effect in Stages 1–2 means length matching is broken.

## Risks

- **871262's clustering gains are point estimates on cells chosen for looking good.** The GOT
  replication shrank +7.8pp → +6.5pp under exactly that pressure. Treat +10.5pp as optimistic and
  size for ~8pp.
- **Stage 3 may move difficulty, not just primer content.** High clustering makes neighbour lists
  overlap, which could change the counting task on its own. The `none` arm measures this; the
  interpretation rule is fixed in advance above.
- **`clu_sd ≥ 0.08` is a judgment call.** The threshold is arbitrary; the monotone trade-off it
  reveals is not. Use it to rank and to report, not to silently delete cells.
- **k=4 and k=12 `none` values are interpolated** from the k∈{8,16} blocks. Stage 1 exists to
  catch a non-smooth surface.
- **Answer position stays uncontrolled**, as in every prior run. `graphqa.canonical()` sorts nodes
  and `incident_encoder` emits them in that order, so the answer-bearing line sits at relative
  depth `target_id / n` — uniformly random, across a region where job 871909 measured a 60pp
  retrieval swing. Accepted here as variance rather than bias (it is out of scope: controlling it
  means choosing which node is queried, not which graph is built), but it is the largest
  uncontrolled term in the experiment and matters when reading marginal results.
- **If the whole programme returns null, the honest conclusion is that it is null *for ER*.** The
  measured SBM alternative (`clu_sd` 0.156 with ~2.9 components at matched edge count) is the
  natural follow-up, not a discarded option.
