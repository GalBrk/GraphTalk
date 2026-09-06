# Primer/task shortcut audit

`docs/plans/run_improved_tests.md` Phase 1, step 1: which `(condition,
task)` pairs let the primer answer the task directly, without the model
needing the graph encoding at all. This is a different, sharper confound
than "the task is trivial regardless of primer" (step 2, below) -- it is
about what the *treatment itself* hands the model, not about the task's
own difficulty.

**Method.** Every classification below is grounded in the actual
renderer (`graphtalk/primers.py::render_primer`) and the exhaustive
primer-only solver (`graphtalk/shortcuts.py`), not guessed from task
names -- reading the rule each cell's real, computed ceiling
(`shortcuts.json`, embedded in every frame row as `shortcut_score`)
traces back to, cross-checked against `tests/test_primers.py`/
`tests/test_shortcuts.py`. Three flags: **shortcut** (the primer text
alone determines the answer, or comes extremely close to it, via an
exact identity or a fitted rule that reaches ~ceiling on this corpus),
**partial** (a real, above-baseline signal from a one-directional
theorem or a moderate correlation, well short of ceiling), **none**
(no detectable leverage -- ceiling equals or barely exceeds the `none`
baseline). Full per-cell ceiling reference (`shortcuts.json`, all values
0-1):

| task | none | degree | clustering | rwse | filler | components | all |
|---|---|---|---|---|---|---|---|
| `node_count` | 0.064 | 1.000 | 1.000 | 1.000 | 1.000 | 0.076 | 1.000 |
| `edge_count` | 0.018 | 1.000 | 0.148 | 0.018 | 0.018 | 0.018 | 1.000 |
| `node_degree` | 0.082 | 1.000 | 0.082 | 0.616 | 0.082 | 0.082 | 1.000 |
| `cycle_check` | 0.832 | 0.946 | 0.832 | 0.832 | 0.832 | 1.000 | 0.946 |
| `edge_existence` | 0.498 | 0.794 | 0.720 | 0.642 | 0.498 | 0.498 | 0.794 |
| `connected_nodes` | 0.082 | 0.208 | 0.082 | 0.082 | 0.082 | 0.082 | 0.352 |

## Step 2 first (it bounds step 1): no task is ceiling-bound under `none`

The highest `none`-column value above is `cycle_check` at 0.832 -- high,
but not the ~100% the plan's step 2 describes. No task is trivially
solved by a graph-blind program *regardless of primer*. This exclusion
set is empty; every task remains in scope for step 1's per-condition
audit below (already noted in `scripts/task_scoped_screen.py`'s
docstring from before this audit existed -- confirmed again here, not
superseded).

## Step 1: per-(condition, task) classification

### `node_count` -- shortcut under FIVE of six conditions, not just `degree`

**The plan's own draft only names `degree x node_count`. That undercounts
it.** `render_primer` (`primers.py:245-272`) loops `for node in
sorted(graph.nodes())` and emits exactly one "Node X has/is ..." sentence
per node for **every** member of `NODE_PARTS = ("degree", "clustering",
"rwse", "filler")` -- not degree specifically. Counting node-sentences
(`shortcuts.py`'s `Context.n_from_sentences`, and the fitted
`node_count_from_max_degree` rule) recovers `n` exactly under any of
degree/clustering/rwse/filler, and `all` (which is degree+clustering+rwse
combined). Only `components` (one graph-level sentence, no per-node loop)
and `none` fail to reveal it -- matching their 0.076/0.064 ceilings,
both near the true baseline.

| condition | flag | mechanism |
|---|---|---|
| `degree` | shortcut | sentence count = n (exact, universal identity) |
| `clustering` | shortcut | same one-sentence-per-node rendering (exact, universal) |
| `rwse` | shortcut | same one-sentence-per-node rendering (exact, universal) |
| `filler` | shortcut | same one-sentence-per-node rendering (exact, universal) -- confirms this session's earlier finding that `qwen3-8b`/`filler`/`node_count` is shortcut-explainable, now traced to the renderer's actual loop structure rather than inferred from the ceiling number alone |
| `components` | none | single graph-level sentence carries no per-node count |
| `all` | shortcut | inherits degree/clustering/rwse |

### `edge_count` -- shortcut only under `degree`/`all`

`m_from_degrees` = sum(stated degrees) / 2, exact and universal, but only
`degree` (and `all`, which includes it) states degrees at all.

| condition | flag | mechanism |
|---|---|---|
| `degree` | shortcut | sum(degrees)/2 (exact, universal identity) |
| `clustering` | partial | ceiling 0.148 vs. 0.018 baseline -- `_fit_edge_count_density` regresses `m` from mean clustering coefficient (an ER density proxy), a real but approximate, corpus-fitted signal, not an identity |
| `rwse` | none | ceiling 0.018, exactly the baseline |
| `filler` | none | ceiling 0.018, exactly the baseline |
| `components` | none | ceiling 0.018, exactly the baseline |
| `all` | shortcut | inherits `degree` |

### `node_degree` -- shortcut under `degree`/`all`, real partial under `rwse`

| condition | flag | mechanism |
|---|---|---|
| `degree` | shortcut | the queried node's degree is stated verbatim -- a lookup, not even arithmetic (exact, universal) |
| `clustering` | none | ceiling 0.082, exactly the baseline -- a clustering coefficient alone doesn't determine degree |
| `rwse` | partial | ceiling 0.616 vs. 0.082 baseline -- substantial, via `_apply_stationary`'s fitted linear inversion of the return-probability diagonal (an approximate stationary-distribution relationship, not exact for a general graph) |
| `filler` | none | ceiling 0.082, exactly the baseline |
| `components` | none | ceiling 0.082, exactly the baseline |
| `all` | shortcut | inherits `degree` |

### `cycle_check` -- partial under `clustering`/`rwse`/`degree`/`all`; `components` is a real surprise

Two *exact but one-directional* theorems exist here (`_clustering_triangle`,
`_rwse_triangle`, `shortcuts.py:480-489`): a nonzero clustering
coefficient, or nonzero 3-step return probability, anywhere in the
primer proves a triangle exists ("Yes") but proves nothing when every
value is zero (can't rule a cycle out). A third, `_edges_at_least_nodes`
(m >= n implies a cycle -- a forest has at most n-1 edges), needs `known_m`,
which only resolves via `m_from_degrees` -- so it only fires when `degree`
is present.

**`components`/`cycle_check` = 1.0 needed real tracing, not a quick
read.** The comment at `shortcuts.py:496-498` calls the exact,
both-directions theorem `_circuit_rank` (`m - n + c > 0`) "the entire
justification for the `components` condition" -- but `_circuit_rank`
also needs `known_m`, and `known_m` only resolves through
`m_from_degrees`, which requires `degree` data that a bare `components`
primer does not carry. So `_circuit_rank` cannot actually fire on a
`components`-only primer, and the observed ceiling of 1.0 is much more
likely coming from the **fitted** `cycle_lookup_c` rule
(`shortcuts.py:1256-1261`: majority answer per stated component count,
fit and scored on disjoint splits per the project's `Split` discipline) --
a corpus-specific statistical regularity of this Erdos-Renyi generator's
parameter regime (e.g., "more than one component almost never happens
alongside a cycle at these n/p settings"), not a universal logical
guarantee the way `degree`/`edge_count`'s identity is. **Flagged as
`shortcut`, but with a corpus-fitted caveat, not an exact-theorem one --
and not fully traced to certainty in this pass** (would need to
instrument `rank_rules`'s actual winning-rule selection to be 100% sure
which rule is driving the 1.0, rather than inferring it from which rules
*could* fire).

| condition | flag | mechanism |
|---|---|---|
| `degree` | partial | `_edges_at_least_nodes` (exact, Yes-only) resolves many but not all cases; ceiling 0.946, high but short of 1.0 |
| `clustering` | partial | `_clustering_triangle` (exact, Yes-only); ceiling 0.832 == the `none` baseline exactly, i.e. this theorem adds nothing measurable on top of the majority-class floor at this corpus's parameters -- a real, if underwhelming, finding worth noting rather than assuming the theorem does nothing at all |
| `rwse` | partial | `_rwse_triangle` (exact, Yes-only); ceiling 0.832, same as `clustering` -- same underwhelming-in-practice note applies |
| `filler` | none | ceiling 0.832, exactly the baseline -- filler carries no relational content |
| `components` | shortcut (corpus-fitted, not a universal identity -- see above) | fitted majority-per-component-count lookup |
| `all` | partial | inherits degree's `_edges_at_least_nodes` (ceiling 0.946, matches `degree` alone -- `all` excludes `components` by definition, so it does NOT inherit the fitted components shortcut) |

### `edge_existence` -- partial under `degree`/`clustering`/`rwse`/`all`

| condition | flag | mechanism |
|---|---|---|
| `degree` | partial | `_degree_sum_margin`/`_apply_degree_sum` (`shortcuts.py:1247-1252`): a parameter-free Chung-Lu-style heuristic, `d_a + d_b > n-1` => "Yes" -- real (ceiling 0.794 vs. 0.498 baseline) but a heuristic, not an identity |
| `clustering` | partial | ceiling 0.720 vs. 0.498 baseline -- substantial, exact mechanism not fully traced in this pass (a plausible fitted/heuristic combination given the scale, not confirmed against a specific named rule) |
| `rwse` | partial | ceiling 0.642 vs. 0.498 baseline -- moderate, exact mechanism not fully traced in this pass |
| `filler` | none | ceiling 0.498, exactly the baseline |
| `components` | none | ceiling 0.498, exactly the baseline -- a single count says nothing about one specific pair |
| `all` | partial | ceiling 0.794, matches `degree` alone |

### `connected_nodes` -- weak partial at best; effectively the cleanest task

| condition | flag | mechanism |
|---|---|---|
| `degree` | partial (weak) | ceiling 0.208 vs. 0.082 baseline -- degree constrains *how many* nodes could be connected without saying *which*; real but modest leverage |
| `clustering` | none | ceiling 0.082, exactly the baseline |
| `rwse` | none | ceiling 0.082, exactly the baseline |
| `filler` | none | ceiling 0.082, exactly the baseline |
| `components` | none | ceiling 0.082, exactly the baseline |
| `all` | partial (weak) | ceiling 0.352, the highest of any condition here but still well short of ceiling -- inherits `degree`'s modest signal, `clustering`/`rwse` add nothing on their own |

## Reading the audit for Phase 5 prioritization

**`shortcut_flag == "none"` pairs** (the only ones that actually test
"does the primer help graph reasoning," not arithmetic/lookup execution):
`components` x every task except `cycle_check`, `filler` x every task
except `node_count`, `clustering`/`rwse` x `node_degree`/`edge_count`
(each one of that pair per condition -- see tables above),
`clustering`/`rwse` x `connected_nodes`. None of these appear in this
session's existing p<0.10 candidate list (Phase 1/2's screen) -- the
candidates found so far (`degree`/`edge_count`, `filler`/`node_count`,
and the borderline `gemma4-12b`/`degree`/`edge_count`) are **all**
`shortcut`-flagged. This is Phase 5's real, load-bearing implication: the
current candidate pool tests execution reliability, not reasoning, and a
genuine reasoning-focused follow-up needs a *new* screen pass restricted
to `shortcut_flag == "none"` cells specifically (see Phase 5, `analysis/
task_scoped_screen.got.csv`/`.csv`'s `shortcut_flag` column).
