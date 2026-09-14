# What's on `main`, and why

This is the map of the repo's scope: what the vendored paper code covers,
where this project's own work picks up beyond it, and which of the parallel
branches fed this state of `main`. Read `README.md` first for the day-to-day
commands; this file is for orienting a new contributor (or reviewer) in the
history behind them.

## The paper this builds on, and where this repo goes past it

`talk_like_a_graph/` is a vendored, mostly-unmodified copy of Google
Research's reference implementation for
[Talk like a Graph: Encoding Graphs for Large Language Models](https://arxiv.org/abs/2310.04560)
(arXiv:2310.04560) — see `talk_like_a_graph/UPSTREAM.md` for the exact commit.
`graphtalk/` is this project's own package on top of it: primers (a short
preamble of structural facts before the graph encoding), the primer-only
shortcut solvers that bound how much of a primer effect is "free" information,
and the scoring/prompting machinery the sweeps run through.

**The paper's own benchmark never goes past 20 nodes.** Its GraphQA corpus is
Erdős–Rényi/Barabási–Albert/scale-free/stochastic-block-model graphs of 5–20
nodes, edge probability sampled continuously rather than swept at fixed
levels, and it runs no size-scaling experiment at all — `talk_like_a_graph`'s
own vendored generator caps node counts at 19
(`graph_generators._NUMBER_OF_NODES_RANGE`: small 5-9 / medium 10-14 / large
15-19). So the n=40+ work described below — the density sweep, the
ladder/retrieval/rewiring design, the size×density grid — is **this
project's own extension past the paper's tested range, not a replication of
one of its conditions.** `scripts/build_size_sweep.py`'s docstring says this
directly: "Nothing in the tracked corpus answers 'does this model degrade as
graphs get big?', because the corpus has no big graphs."

`docs/primer-effects-and-power.md` is the results document to read first for
what's actually been found; it supersedes `docs/sweep-findings.md` (the
original 5-19 node corpus). Two rules from it govern every number in this
repo: read a primer effect against `bar(cond) - bar(none)` from
`shortcuts.json`, not against zero (a primer-only solver that never sees the
graph already scores 1.00 on several cells), and against a length-matched
control rather than `none` — a content-free primer of the same length costs a
thinking model 11.7pp on dense graphs, larger than most measured primer
effects.

## The canonical n=40 experiment

The headline extension experiment: `n=40` fixed, ER density pinned to
`{0.10, 0.20, 0.35, 0.50}`, all 7 primer conditions (`none`, `components`,
`degree`, `clustering`, `rwse`, `filler`, `all`), all 6 tasks, 100 graphs per
(density, task) cell — 16,800 prompts, one shared file
(`prompts.densfull40.jsonl`) across every model arm so results are directly
comparable:

```bash
PYTHONPATH=. python scripts/build_size_sweep.py --sizes 40 --densities 0.10 0.20 0.35 0.50 --count 100
PYTHONPATH=. python scripts/score_density_sweep.py --responses "runs/qwen3-1.7b.degdens40.shard*of5.jsonl"
```

Four model arms: `qwen3-1.7b`, `qwen3-1.7b-think`, `qwen3-4b`,
`qwen3-4b-think` (see `cluster/run-4b-density-sweep.md` for the exact
submission recipe, including the `--max-new-tokens 8192` override
`edge_count` needs at this density).

**Read results with the same validity caveats `docs/primer-effects-and-power.md`
documents at this size/density:** `node_count` is contaminated by nearly every
primer (a per-node sentence lets a solver count sentences instead of reading
the graph), `cycle_check`'s gold is "yes" for almost every graph past the
sparsest level, and the `degree` primer states the `node_degree` answer
verbatim (shortcut bar 1.00). `node_degree` with
`{none, components, clustering, filler}` is the one cell already validated
clean at this size.

## The ladder, retrieval, and rewiring design

Three linked pieces, documented in full in `docs/ladder-and-rewiring.md` and
`docs/ladder-and-retrieval-results.md` — this is a pointer, not a restatement:

- **Ladder** (`scripts/build_ladder.py --stage screen`, `scripts/analyze_ladder.py`)
  screens each model across 18 `(n, k̄)` rungs to locate its informative band
  — not so easy the primer is at ceiling, not so hard the model can't read the
  prompt at all.
- **Retrieval** (`scripts/build_retrieval_probe.py`, `scripts/analyze_retrieval.py`)
  is a graph-free needle-in-haystack probe that independently locates each
  model's raw reading-token limit, so a bad ladder cell can be attributed to
  "can't read it" vs. "reads it but reasons wrong."
- **Rewiring** (`scripts/build_ladder.py --stage rewire`,
  `scripts/analyze_rewiring_sweep.py`) is the actual primer test: a
  degree-preserving double-edge-swap at the shared `n40k12` rung changes
  triangle count/clustering while holding degree sequence, node/edge count,
  and rendered prompt length **exactly** fixed — isolating a primer's content
  effect from the length and difficulty confounds `filler`/`degdens40`
  exposed.

`cluster/run_ladder.sh` orchestrates the cheap gating passes (`probe` +
`ladder`); the rewiring experiment itself is deliberately submitted
separately, only for rungs the first two passes have shown are valid for that
model.

## Why n=40, not a size sweep

A separate size-only sweep (`cluster/run_size_sweep.sh` +
`scripts/size_screen.py`, plus a never-executed 2D node-count × density grid,
`scripts/score_density_size_grid.py`/`cluster/run_density_size_sweep.sh`) was
tried and then removed from `main`. Neither served the project's actual
question: the size sweep's own primer-ranking result was an explicit
non-finding ("no primer is consistently better than `none` at any size...
likely noise"), and the ladder/rewiring design above already solves the
underlying problem those tools were reaching for — locating a valid operating
point — more rigorously, with a real executed primer test at the end rather
than an unrun exploratory grid. `docs/plans/finding-graphs-that-make-primer-effects-measurable.md`
has the fuller design history, marked superseded at its top.

## Significance-testing fixes

`scripts/check_significance.py`/`scripts/recommend_count.py` had three real
bugs, fixed and landed on `main`:

1. Clustered permutation/bootstrap tests keyed clusters on
   `(model, "<task>/<index>")` instead of `(model, graph)` — since each index
   is unique per task, this made clustering a silent no-op (`n_clusters ==
   n_pairs` on every row). Fixed via `graphtalk.analysis.graph_index`, the one
   shared definition `check_significance.py`, `mixed_models.py`'s GEE
   grouping, and their validators now all agree on.
2. The MDE (minimum detectable effect) injector moved pairs in one direction
   only, making every prior power estimate too optimistic.
3. `scripts/recommend_count.py` divided the swept MDE *parameter*
   (`mde_delta`) by an observed `delta`, instead of the *realized* difference
   that parameter actually produced (`mde_realized_diff`) — different units
   near the accuracy ceiling, which inflated sample-size recommendations
   53–163× on the affected cells.

`analysis/superseded/` holds the report artifacts these fixes made obsolete,
with a README explaining what each defect changed. **Any significance number
computed before these fixes should be treated as suspect** — re-derive it
rather than citing an old report.

## Branch provenance

This state of `main` reconciles four branches that had been developed in
parallel without ever merging into each other:

- `ladder-and-rewiring` — the ladder/retrieval/rewiring design above.
- `small-model-suite-and-primer-power` — the qwen3-4b model spec and the
  canonical n=40 density sweep.
- `statistical-significance` — the three bug fixes above.
- `analyzing-graph-features` — a `graphtalk/significance.py` extension
  (`required_n_closed_form`, `required_sample_size_clustered`) for
  prospective sample-size planning, and a real pre-existing bug fix in
  `graphtalk/diverse_corpus.py`'s `build_pool` (an explicit empty
  `algorithms` tuple silently fell back to the default instead of raising).
  Its size/density grid tooling was tried and removed — see "Why n=40, not a
  size sweep" above.

A fifth branch, `nitzan`, carried no unique content (verified: zero diff
against its own merge-base with `main`) and was retired without needing to
merge anything. The four branches above stay up on `origin` as historical
record rather than being force-deleted — ask before assuming one is safe to
remove.
