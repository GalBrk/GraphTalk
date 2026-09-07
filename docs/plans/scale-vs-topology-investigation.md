# Scale vs. topology: why `qwen3-8b`/`degree` (GOT) became globally significant at n=500

## Context

Commit `30fa9ed` replicated `qwen3-8b`/`degree` (GOT node-naming) at
`--count 500`: delta shrank +7.8pp -> +6.5pp, and `bh_significant_global`
flipped False -> True (n_clusters 180 -> 3,000; p 0.0018 -> 0.0001). The
committed narrative calls the shrinkage "the expected winner's-curse
correction" and treats this as pure added power --
`scripts/recommend_count.py`'s own sample-size formula structurally assumes
the observed +7.8pp is the stable true effect and only computes how much
data is needed to detect *that same effect*, with no mechanism for the
effect itself to change with scale.

That assumption had never actually been checked. Two diagnostic scripts
built for exactly this question (`scripts/check_old_vs_new_subsample.py`,
`scripts/diff_shared_instances.py`) were committed alongside the README
update but never run. This doc is that investigation, run to completion:
scale vs. topology, distribution shift, and driver analysis, plus a new
data point neither prior script surfaced (a Simpson's-paradox confound in
the naive driver analysis) and one they were built to catch (decoding
nondeterminism on the shared instances).

One structural fact bounds the whole investigation: the 500-row corpus is a
strict, byte-identical superset of the 30-row corpus (verified in commit
`13b7310`) -- both are the first N rows of one fixed HF split, itself one
fixed shuffle of one Erdos-Renyi generator run (`seed=1234`, no algorithm
variation anywhere in the published data; confirmed independently here via
`scripts/extract_graph_topology.py --verify-config edge_count`, 0/500
structural mismatches against a second task config). So "a new
graph-generation *process* was introduced at 500" was never on the table.
The live questions were narrower: was the original 30-row draw a
compositionally atypical sample of that same fixed population, and does the
`degree` primer's advantage concentrate on identifiable structural features?

## 1. Scale vs. topology: pure power, plus a small decoding-noise wrinkle

`scripts/check_old_vs_new_subsample.py --frame analysis/sweep_frame.count500.got.csv --model qwen3-8b --condition degree`:

| slice | n_clusters | delta | 95% CI | p |
|---|---|---|---|---|
| full (0-499) | 3,000 | +0.0647 | [+0.0533, +0.0763] | 0.0001 |
| original (0-29) | 180 | +0.0611 | [+0.0167, +0.1056] | 0.0125 |
| new (30-499) | 2,820 | +0.0649 | [+0.0532, +0.0766] | 0.0001 |

Original and new 95% CIs overlap, and the point estimates are close (+0.061
vs. +0.065) -- **consistent with one stable effect across the full sample.
The global-significance flip is pure added power** (MDE shrinking as
`n_clusters` grows from 180 to 3,000), not a change in what's being
measured.

One loose end the script itself flags: the "original" slice's delta here
(+0.0611, p=0.0125) doesn't quite match the standalone tracked `--count 30`
report's delta for the same cell (+0.0778, p=0.0018, from
`analysis/significance_report.got.csv`). `scripts/diff_shared_instances.py`
traces this exactly: of the 180 shared pairs, **5 flip** between the two
runs (`edge_count/1`, `edge_count/9`, `edge_count/18`, `edge_count/28`,
`cycle_check/5`), every one with byte-identical prompts in both runs. That
rules out a prompt/content change and points to decoding-level
nondeterminism between the two generation runs (both single-stream,
batching independently ruled out as a confound in commit `d409c11`) --
a real, if minor, methodological wrinkle worth knowing about (greedy
decoding across two separate cluster jobs is not perfectly reproducible
here), but it is a property of the *shared* 30 instances specifically, not
of the added 470, and does not bear on the topology question.

## 2. Distribution shift: none detected

`scripts/extract_graph_topology.py --count 500` computed 14 structural
features per graph (density, degree stats, component count, circuit rank,
tree/forest/bipartite/isolated-node status, triangle count, clustering,
size bucket) from the parsed graphs, joined by row index (every task shares
the same graph at a given index -- verified, not assumed).

`scripts/compare_old_vs_new_topology.py --features analysis/topology_features.csv --split-at 30`
ran an unpaired permutation test (new `graphtalk.significance
.unpaired_permutation_test`, hand-rolled to match the project's existing
scipy-free convention) on each feature between the original 30 and the new
470, BH-corrected across all 14:

**0/14 features differ significantly (BH q=0.05).** Size-bucket
proportions are near-identical too (small/medium/large: 30.0/33.3/36.7%
original vs. 30.4/34.7/34.9% new). The original 30-graph draw looks like an
unremarkable sample of the same fixed population -- there is no detectable
compositional skew to explain away. Full per-feature table:
`analysis/topology_old_vs_new_report.csv`; distribution plots:
`analysis/topology_distribution_plots/`.

This directly corroborates part 1's power-alone verdict from an
independent angle: not only does the *effect* look stable across the
old/new split, the *population it was measured on* is statistically
indistinguishable between the two slices.

## 3. Driver analysis: task dominance, and a Simpson's-paradox trap

`scripts/analyze_topology_drivers.py` (pooled across all 6 tasks, n=3,000
pairs) stratified the `degree` vs `none` effect by task and by structural
feature, reusing the exact same
`paired_permutation_test_clustered`/`cluster_bootstrap_ci_clustered`
machinery the headline result was computed with, BH-corrected across all
21 stratified tests.

**By task, `edge_count` overwhelmingly dominates:**

| task | delta | p |
|---|---|---|
| `edge_count` | **+0.3200** | 0.0001 |
| `cycle_check` | +0.0340 | 0.0063 |
| `node_degree` | +0.0140 | 0.0354 |
| `edge_existence` | +0.0060 | 0.25 (n.s.) |
| `connected_nodes` | +0.0120 | 0.54 (n.s.) |
| `node_count` | +0.0020 | 1.00 (n.s.) |

This resolves the open question from `analysis/README.md`'s "kept for the
historical record" note, which worried the n=30 attribution to `edge_count`
(+35.7pp there) might have shifted to `node_degree` at n=500 (its MAE
metric clears significance, p=0.014, while `edge_count`'s MAE narrowly
misses, p=0.050). **On the `exact`-match metric the headline result is
actually reported on, `edge_count` remains completely dominant at n=500
too** (+32.0pp vs. +1.4pp for `node_degree`) -- the MAE-metric shift was an
artifact of MAE penalizing by error *magnitude* on a different scale, not a
change in which task drives the accuracy effect.

**The naive pooled structural stratification looked informative, but was a
trap.** Pooled across all 6 tasks, several structural features showed a
clean-looking, monotonic-ish pattern -- sparser/simpler graphs (no cycle,
bipartite, has an isolated node, low density) showing roughly 1.5-2x the
effect of their complements:

| feature | complement | "sparse" side |
|---|---|---|
| `is_forest` | +0.057 (False) | **+0.101** (True) |
| `is_bipartite` | +0.056 (False) | **+0.101** (True) |
| `has_isolated_node` | +0.055 (False) | **+0.092** (True) |
| `density` tercile | +0.051 (high) | **+0.081** (low) |

Every one of these **reverses or collapses** when re-run with
`--task edge_count` (i.e. within the one task that actually carries the
pooled effect):

| feature | complement (within edge_count) | "sparse" side (within edge_count) |
|---|---|---|
| `is_forest` | **+0.363** (False) | +0.107 (True) |
| `is_bipartite` | **+0.358** (False) | +0.152 (True) |
| `has_isolated_node` | **+0.353** (False) | +0.225 (True) |
| `density` tercile | +0.231 (low) | **+0.386** (high) |

This is a textbook Simpson's paradox: pooling across tasks let a
structural feature's correlation with *which task dominates a stratum*
masquerade as a structural effect on the primer's usefulness. It is not.

**The real driver, found by checking the surviving candidates within
`edge_count` alone, is task difficulty -- specifically graph size and
density, monotonically:**

| stratum (within `edge_count`) | delta | p |
|---|---|---|
| `size_bucket=small` | +0.105 | 0.0010 |
| `size_bucket=medium` | +0.249 | 0.0001 |
| `size_bucket=large` | **+0.577** | 0.0001 |
| `degree_std=low` | +0.168 | 0.0001 |
| `degree_std=mid` | +0.319 | 0.0001 |
| `degree_std=high` | **+0.473** | 0.0001 |
| `density=low` | +0.231 | 0.0001 |
| `density=mid` | +0.348 | 0.0001 |
| `density=high` | **+0.386** | 0.0001 |

The `degree` primer's benefit on `edge_count` grows monotonically and
substantially with graph size, degree-sequence variance, and density --
i.e. it helps most exactly where manually counting edges from the encoded
graph text is hardest, and least where the no-primer baseline can already
get the count right by inspection on a small, sparse graph. That is a
coherent, mechanistically sensible story (the primer effectively hands the
model the "sum of degrees / 2" shortcut, which saves the most work exactly
when there's the most to sum), not a post-hoc pattern-match across
unrelated categorical labels.

**Read this as a hypothesis, not a second confirmed finding.** This is
still an exploratory analysis run against the same data that produced the
headline result -- 21+16 stratified tests were run in total across the two
passes above, most surviving BH correction only because the underlying
`edge_count` effect is enormous, not because each stratum is independently
well-powered evidence. A confirmatory check would pre-register "the
`degree` primer's `edge_count` benefit increases with graph size/density"
specifically (analogous to `analysis/confirmatory_got_degree.json`) before
collecting more data, rather than treating this write-up's numbers as
already establishing it.

## Reproducing this investigation

```bash
PYTHONPATH=. .venv/Scripts/python.exe scripts/check_old_vs_new_subsample.py \
    --frame analysis/sweep_frame.count500.got.csv --model qwen3-8b --condition degree
PYTHONPATH=. .venv/Scripts/python.exe scripts/diff_shared_instances.py \
    --old-frame analysis/sweep_frame.got.csv --new-frame analysis/sweep_frame.count500.got.csv \
    --model qwen3-8b --condition degree \
    --old-prompts prompts_got.jsonl --new-prompts prompts_got.count500.degree.jsonl
PYTHONPATH=. .venv/Scripts/python.exe scripts/extract_graph_topology.py \
    --count 500 --out analysis/topology_features.csv
PYTHONPATH=. .venv/Scripts/python.exe scripts/compare_old_vs_new_topology.py \
    --features analysis/topology_features.csv --split-at 30 \
    --out analysis/topology_old_vs_new_report.csv
PYTHONPATH=. .venv/Scripts/python.exe scripts/analyze_topology_drivers.py \
    --out analysis/topology_drivers_report.csv
PYTHONPATH=. .venv/Scripts/python.exe scripts/analyze_topology_drivers.py \
    --task edge_count --out analysis/topology_drivers_report.edge_count.csv
```
