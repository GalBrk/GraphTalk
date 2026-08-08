# Primer computation

## Context

The project's independent variable is a "primer": a short preamble of factual sentences
about each node's local structure, prepended before the graph encoding and the question.
This step builds the primer generation itself — the statistics, and the text they render
into. Prompt assembly, model querying, and scoring are all later steps.

The previous step proved we can recover a correct `networkx` graph from a GraphQA row
(`scripts/draw_graph.py`, verified on 600 rows). Everything here builds on that.

Local runs are spot checks on a handful of examples; full sweeps happen on the cluster.
So this lands as an importable module, not a script — cluster code must be able to
`from graphtalk.primers import ...` and get output identical to what we eyeballed
locally. Every function here is a pure function of the graph, with no randomness, so
that identity holds without seed management.

## Environment and repo state

- Python lives in `.venv/` (uv, Python 3.11). Run things as `.venv/bin/python ...`.
- Use `uv run --no-sync`, never bare `uv run` — a plain `uv run` re-syncs to the default
  dependencies and uninstalls the optional 2 GB `[pipeline]` TensorFlow extra.
- `talk_like_a_graph/` is vendored Google Research code (Apache 2.0, upstream
  `36af51e`). `graph_text_encoders.encode_graph(g, "incident")` is the base encoding
  this project uses. Do not reformat or restyle that directory.
- Anything importing `tensorflow_gnn` needs `TF_USE_LEGACY_KERAS=1`. Nothing in this
  step does.
- Existing Python in the repo uses 2-space indentation (upstream Google style); match it.

## Dataset facts worth not rediscovering

The published GraphQA dataset is `baharef/GraphQA` on HuggingFace (baharef = Bahare
Fatemi, the paper's first author). Verified properties:

- Fields are `algorithm, answer, nedges, nnodes, question, task_description,
  text_encoding` — **all strings**. There is no structured edge list; the graph exists
  only as rendered English inside `question`, which is why `parse_graph` exists.
- It ships **only the `adjacency` encoding** and **only Erdős–Rényi graphs**, despite the
  proposal saying otherwise. Incident encoding must be regenerated from the parsed graph.
- Test splits hold 500 rows per task. All six tasks the proposal names exist as configs.
- `connected_nodes` spells an isolated target as `" No nodes."`, not an empty list —
  about 7% of rows.
- `edge_count` and `node_count` answers carry a leading space (`" 115."`).
- `cycle_check` is 86% "Yes"; `edge_existence` is 58% "No".

Rows are fetched over the HTTP rows API rather than the `datasets` library, deliberately:
`datasets` brings its own `pyarrow`/`fsspec` pins and the venv holds a hand-tuned
TensorFlow 2.20 / tf-keras / tensorflow-gnn combination that is easy to disturb.

### Decisions already made

- **RWSE uses k=2..5**, with `k_min`/`k_max` configurable. k=1 is provably 0 for every
  node of every simple graph (`P = D⁻¹A` and `A` has a zero diagonal), so it is a
  constant column that costs tokens and carries nothing.
- **Report the RWSE/degree correlation.** Measured on 60 GraphQA graphs, per-node RWSE
  correlates with degree at r≈0.57 (k=2) rising to r≈0.94 (k=5), because the walk
  converges to stationarity where `P^k_ii → d_i/2m`. The "RWSE only" condition therefore
  partly restates degree. This is emitted as a diagnostic so it becomes a reported
  result rather than a hidden confound.
- **Per-node sentences**, matching the proposal's "one short factual sentence per node"
  and the surrounding graph prose.
- **An inert length control**, not a misinformation placebo. Wrong numbers would
  contradict the edge list in the same prompt, so a drop in accuracy could mean the
  model was misled, or merely confused by an inconsistent prompt — neither of which is
  the length effect the control exists to isolate. The control states only true,
  structurally vacuous facts, so there is nothing for it to contradict.
- **A seventh condition: number of connected components.** The rationale is below;
  `docs/features-considered.md` records the features that were evaluated and rejected.

### Correction to the proposal's task taxonomy

The proposal classifies cycle check as primer-agnostic. Measured on 100 `cycle_check`
rows, it is not:

- No acyclic graph has any node with clustering > 0 — necessarily, since a forest
  contains no triangle. So `clustering > 0` ⟹ a cycle exists, at **100% precision**.
- 83 of the 86 cyclic graphs have some node with clustering > 0 — **97% recall**.

A model reasoning "some node has clustering above zero, so there is a triangle, so there
is a cycle" therefore scores ~97% against an 86% majority-class baseline. `RWSE(k=3) > 0`
is the same triangle test and agreed with it on 100/100 graphs. (Even-k carries no such
signal: any node with a neighbour can walk out and back.)

`node_count` is not agnostic either, for a duller reason — every per-node primer
re-enumerates all the nodes, handing the model a second surface to count.

Of the three tasks the proposal calls primer-agnostic, only **edge_existence** actually
is. Anywhere this plan or the analysis refers to "the three agnostic tasks", read one.

## Approach

### 1. Lift shared code into a package

`parse_graph` currently lives in `scripts/draw_graph.py` and was written to be moved.
Create a `graphtalk/` package and move it, along with the HTTP row fetching:

- `graphtalk/graphqa.py` — `parse_graph(question)`, `fetch_rows(config, split, offset, length)`
- `graphtalk/primers.py` — statistics and renderers

Then update `scripts/draw_graph.py` to import from `graphtalk.graphqa` rather than
keeping its own copy. Its `expected_answer`/`check` logic stays where it is; that is
verification tooling, not pipeline code.

Add `graphtalk` to `packages` in `pyproject.toml`, and `tests` to `testpaths`.

### 2. Statistics (`graphtalk/primers.py`)

Three per-node functions, each returning a dict keyed by node id, nodes in sorted order:

- `degrees(graph) -> dict[int, int]` — thin wrapper over `graph.degree`
- `clustering(graph) -> dict[int, float]` — wrapper over `nx.clustering`
- `rwse(graph, k_min=2, k_max=5) -> dict[int, list[float]]` — diagonal of `P^k` for
  `P = D⁻¹A`, accumulated by repeated matrix multiply

and one graph-level function:

- `component_count(graph) -> int` — wrapper over `nx.number_connected_components`.
  Isolated nodes each count as their own component, which is what `networkx` already
  does and what the circuit-rank identity below requires.

Isolated nodes: degree-0 rows would divide by zero, so clamp the divisor to 1, which
yields an all-zero RWSE vector. This is a convention, not a fact — a walk from an
isolated node is undefined. Define it as a named constant with a comment, since ~7% of
`connected_nodes` rows have an isolated target and this case will occur.

Round to two decimals at render time, not in the statistics, so the diagnostic
correlation is computed on full precision.

### 3. Renderer

**One** renderer produces the text for every condition. There is no per-feature renderer
and no combining step.

**Naming.** The pieces a primer is built from are called **parts**, not "components" —
"connected components" is now one of the features, and the collision would be confusing
in code. The renderer parameter is `parts`.

Parts come in two tiers. **Node-level** parts each contribute an object phrase for a
given node:

| node-level part | phrase |
|---|---|
| `degree` | `degree 4` |
| `clustering` | `clustering coefficient 0.70` |
| `rwse` | `random-walk return probabilities 0.27, 0.15, 0.19, 0.18` |
| `filler` | `7 other nodes in this graph` (the length control) |

**Graph-level** parts contribute a whole sentence about the graph:

| graph-level part | sentence |
|---|---|
| `components` | `This graph has 3 connected components.` |

The renderer emits graph-level sentences first, then Oxford-joins each node's requested
phrases into one sentence per node under a shared verb, joining everything with a space:

```
Node 0 has degree 4.
Node 0 has degree 4, clustering coefficient 0.70, and random-walk return probabilities 0.27, 0.15, 0.19, 0.18.
Node 0 has 7 other nodes in this graph.
This graph has 3 connected components.
```

Use the singular when the count is 1 (`1 connected component`); a grammatical slip in a
condition that appears in every prompt of its arm is exactly the kind of thing that could
show up as a spurious effect.

```python
render_primer(graph, parts=(), k_min=2, k_max=5) -> str
```

`parts=()` yields the empty string — that is the `none` condition, not a special case.
`build_primer(graph, condition, ...)` is then a thin mapping from the seven condition
names onto part tuples.

This single code path is what makes the comparison mean anything. Fatemi et al.'s
central result is that phrasing alone moves accuracy by tens of points, so if conditions
differed in format as well as content, a difference between them would be
uninterpretable. With one renderer, format consistency is structural — there is no way
for two conditions to word themselves differently — rather than a convention someone has
to remember to maintain.

### 4. Length control

The `filler` component states a true fact that no task depends on:

```
Node 0 has 7 other nodes in this graph.
```

The phrasing is chosen to fit the shared `Node N has ...` template, so the control comes
out of the same renderer as everything else rather than needing its own. It contradicts
nothing in the encoding, and gives away nothing about structure — the node count is
already stated in the encoding's first line. Its only property is occupying prompt space
in the same rhythm as a real primer.

Exact length matching is not achievable with vacuous sentences, and chasing it would
make the text contrived. Instead: emit one sentence per node by default, accept an
optional `target_chars` to pad with a second vacuous clause when a closer match to a
longer condition is wanted, and **always report achieved character counts** so the
mismatch is visible in analysis rather than assumed away. Approximate matching that is
measured beats exact matching that is faked.

Note this control is only needed for comparisons against `none`. Contrasts among the real
primer conditions are already roughly format-matched. Given the taxonomy correction
above, the task where the information-vs-length ambiguity genuinely bites is
`edge_existence`, so that is where the control earns its cluster time first.

### 5. The components condition

```
This graph has 3 connected components.
```

Why this feature and not another (see `docs/features-considered.md` for the rejected
ones):

- **It is not the answer to any of the six tasks.** Degree gives away node_degree, edge
  count gives away edge_count, triangle count all but decides cycle_check. Component
  count gives away none of them, so a positive result cannot be dismissed as the primer
  trivially containing the answer.
- **It completes an exact inference.** A graph has a cycle iff its circuit rank
  `m − n + c > 0`. Checked against ground truth on 100 rows: **100/100**. The model must
  still count nodes and edges from the encoding and combine three numbers, which is
  precisely the "supplies usable ingredients but must still combine them" pattern the
  proposal is built around.
- **It costs one sentence, not one per node.** Token cost is negligible, so the length
  confound that motivated the inert control barely applies here. If a matched control is
  wanted for this arm specifically, it is a single vacuous graph-level sentence rather
  than the per-node filler.
- **The counts are not trivial.** 71% of the sampled graphs are connected, but the tail
  runs to 15 components, so the statement carries real variance.

The honest risk: models may simply not know the circuit-rank identity, in which case this
condition is a null. That is still an interesting null — "we supplied the exact
ingredients for a deterministic inference and the model could not use them" is a sharper
claim about black-box structural reasoning than another local statistic would produce.
Expect it to depend heavily on CoT, since the inference is multi-step.

### 6. Diagnostic

`rwse_degree_correlation(graph, k_min, k_max) -> dict[int, float]` returning Pearson r
per k, computed on unrounded values. Undefined when degree or RWSE is constant across
nodes (regular graphs, empty graphs) — return `None` for that k rather than a NaN.

Measured reference values, for sanity-checking the implementation: RWSE correlates with
degree at r≈0.57 (k=2) rising to r≈0.94 (k=5). Clustering, by contrast, is essentially
independent of degree (r≈−0.16) and varies more across nodes than degree does
(coefficient of variation 0.49 vs 0.38) — so clustering is the genuinely independent
feature of the three, and RWSE is the redundant one.

## Files

- `graphtalk/__init__.py`, `graphtalk/graphqa.py`, `graphtalk/primers.py` — new
- `scripts/draw_graph.py` — drop its local `parse_graph`/`fetch_rows`, import from `graphtalk.graphqa`
- `scripts/show_primers.py` — new; prints all seven conditions for a few real rows, plus
  character counts and the correlation diagnostic, for eyeballing
- `tests/test_primers.py` — new
- `pyproject.toml` — add `graphtalk` to packages, `tests` to testpaths

## Verification

Analytic tests, no network, in `tests/test_primers.py`:

- **Triangle** — degree 2, clustering 1.0, RWSE k=2 → 0.5 and k=3 → 0.25 (closed form,
  already confirmed by hand)
- **Star and path** — clustering 0 everywhere; both bipartite, so all odd-k RWSE is
  exactly 0
- **K4** — clustering 1.0, degree 3
- **Isolated node** — degree 0, clustering 0, RWSE all zeros
- **k=1 is 0** whenever `k_min=1` is requested
- **Component count** — connected graph → 1; two disjoint triangles → 2; a graph with an
  isolated node counts that node as its own component
- **Circuit-rank identity** — across a spread of small graphs (trees, forests, cycles,
  disjoint unions, graphs with isolated nodes), `m − n + c > 0` agrees with
  `nx.find_cycle`. This is the load-bearing property for the components condition: if it
  does not hold, the condition supplies no valid inference
- **Singular/plural** — the sentence reads `1 connected component`, not
  `1 connected components`
- **Brute-force cross-check** — enumerate every walk of length k on a small graph and
  compare the return fraction against the matrix-power result. This is the load-bearing
  test: it checks the fast implementation against an independent one rather than against
  itself.
- **Determinism** — the same graph produces byte-identical primer text across runs, so
  cluster output can be compared against local spot checks
- **Length control is inert** — its text contains no degree, clustering, or RWSE value,
  and is identical for any two graphs with the same node count
- **Rendering** — two-decimal formatting, one sentence per node, sorted node order,
  and correct Oxford-comma joining at one, two, and three components
- **Format invariance** — every condition's sentences match the same
  `Node N has <phrases>.` shape, which is the property the single renderer exists to
  guarantee

```bash
uv run --no-sync pytest tests/ -q
```

Then a spot check against real data, which is where wording gets judged:

```bash
.venv/bin/python scripts/show_primers.py --config node_degree --count 3
```

Read the printed primers and confirm the sentences are well-formed, the numbers match a
hand-check on the smallest graph, the length control states nothing structural, and the
reported correlation falls in the r≈0.57 to r≈0.94 range already measured. Do not skip
this by assuming the tests cover it — the tests check arithmetic, not whether the English
reads like the surrounding graph prose.

Also re-run the previous step's verification, since `scripts/draw_graph.py` is being
refactored to import from the new package:

```bash
.venv/bin/python scripts/draw_graph.py --config connected_nodes --index 0 --count 20
```
