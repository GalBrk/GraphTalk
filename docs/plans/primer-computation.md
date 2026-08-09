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
locally.

That identity is **not** automatic, and an earlier draft of this plan claimed it was.
Two things are required for it, both implemented here:

- **Canonical graphs.** The vendored encoder iterates nodes and adjacency in insertion
  order, so the same graph built two ways produces different prose. `canonical()` at the
  parse boundary makes insertion order equal sorted order.
- **Pre-quantised rendering.** Float RWSE values that sit exactly on a two-decimal
  boundary land one bit either side of it depending on the matrix summation order, which
  differs across BLAS builds. A single `_fmt` helper removes that.

With those two in place, every function here is a pure function of the graph and identity
holds without seed management. Without them, it does not.

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
- **The editable install currently maps only `talk_like_a_graph`.** Adding `graphtalk`
  to `pyproject.toml` is necessary but not sufficient — the mapping file is written at
  install time, so `uv pip install -e ".[dev]"` must be re-run. Until then
  `import graphtalk` fails from `scripts/` (whose `sys.path[0]` is `scripts/`, not the
  repo root) and under the bare `pytest` console script (which, unlike `python -m
  pytest`, does not add the working directory). Both verification commands at the end of
  this document depend on the reinstall.

## Provenance of the numbers in this document

Network access to the HuggingFace rows API is blocked in the current environment
(403 at the proxy). Every measurement quoted below was therefore taken on
`graph_generators.generate_graphs(500, "er", False, random_seed=1234)` — the upstream
generator that produced the dataset, at the test-split seed, with node counts 5..19 and
sparsity uniform on (0, 1).

Treat these as provisional. The following need re-measuring on real rows when a
networked environment is available:

| quantity | generator value | published / recorded |
|---|---|---|
| `edge_existence` class balance | near-balanced: Yes 50.3% ± 2.0 across query resamples, so the majority baseline is ≈51.6% | paper says 53.96% No |
| `cycle_check` "Yes" rate | 83.2% | paper says 81.96% |
| `connected_nodes` isolated target | 9.0% | earlier draft recorded ~7% |
| RWSE/degree correlation | see §6 | earlier draft recorded 0.57 → 0.94 for k=2..5 |

An earlier draft of this plan recorded 58% and 86% for the first two. Both were wrong;
the paper's own figures are in `https:::arxiv.org:pdf:2310.04560.pdf`.

## Dataset facts worth not rediscovering

The published GraphQA dataset is `baharef/GraphQA` on HuggingFace (baharef = Bahare
Fatemi, the paper's first author). Verified properties:

- Fields are `algorithm, answer, nedges, nnodes, question, task_description,
  text_encoding` — **all strings**. There is no structured edge list; the graph exists
  only as rendered English inside `question`, which is why `parse_graph` exists.
- It ships **only the `adjacency` encoding** and **only Erdős–Rényi graphs**, despite the
  proposal saying otherwise. Incident encoding must be regenerated from the parsed graph.
- Test splits hold 500 rows per task. All six tasks the proposal names exist as configs.
- `connected_nodes` spells an isolated target as `" No nodes."`, not an empty list.
- `edge_count` and `node_count` answers carry a leading space (`" 115."`).

Facts about the vendored code that the primer design depends on:

- **`incident_encoder` emits no line at all for a degree-0 node.** It branches on
  `nedges > 1` and `nedges == 1` with no `else` (`graph_text_encoders.py:112-121`). An
  isolated node appears only in the header enumeration. Isolation is therefore conveyed
  **by omission**, which is the single hardest inference in the encoding — and the reason
  the degree and rwse primers change that task's difficulty. 26% of graphs contain an
  isolated node.
- **Question headers are always ascending**, because `create_node_string` iterates
  `sorted(name_dict.keys())` (`graph_text_encoders.py:8-14`). This is why `parse_graph`
  happens to insert nodes in sorted order today (0/300 exceptions). It is a property of
  Google's formatting code, not of ours, which is why `canonical()` exists.
- **Query sampling differs by task.** `edge_existence` draws a uniform pair
  (`graph_tasks.py:136`); `connected_nodes` and `node_degree` draw one uniform node
  (`:393`, `:264`); `node_count`, `edge_count` and `cycle_check` do no query sampling and
  emit one row per graph. Any rate quoted per-row must state which of these it used.
- **`create_node_string` is malformed for n = 1** ("among nodes and 0"). The generator's
  minimum is 5 nodes, so this cannot occur; recorded only so nobody rediscovers it.

Rows are fetched over the HTTP rows API rather than the `datasets` library, deliberately:
`datasets` brings its own `pyarrow`/`fsspec` pins and the venv holds a hand-tuned
TensorFlow 2.20 / tf-keras / tensorflow-gnn combination that is easy to disturb.

### Decisions already made

- **RWSE uses k=2..3**, with `k_min`/`k_max` configurable. k=1 is provably 0 for every
  node of every simple graph (`P = D⁻¹A` and `A` has a zero diagonal). k=4 and k=5 are
  dropped for measured redundancy: the share of each column's per-node variance already
  explained by degree and clustering within a graph is

  | k | from degree alone | from degree + clustering |
  |---|---|---|
  | 2 | 0.52 | 0.74 |
  | 3 | 0.80 | 0.93 |
  | 4 | 0.81 | 0.86 |
  | 5 | 0.89 | **0.96** |

  k=2 is the only column carrying substantial independent signal (average inverse
  neighbour degree). k=3 is the triangle test, which is the arm's own cycle mechanism and
  cannot be dropped even though clustering duplicates it in other arms. k=4 is even, so
  it carries no odd-cycle signal, and k=5 buys pentagon detection on 1% of graphs.
  Dropping both is a further truncation of a range this project has already edited twice
  on principled grounds, and it is reported as a result rather than an omission.

- **RWSE is rendered with explicit step counts.** Bare numbers do not say which walk
  length each belongs to, and the proposal defines RWSE as k=1..4, so a reader applying
  the published convention reads every value at the wrong offset. Measured cost of that
  misreading on `cycle_check`, using the rule "some node has nonzero odd-k return, so a
  cycle exists": 96.5% with the labels known, 85.5% misread, against an 83.0% majority
  baseline. The misreading collapses the rule because it lands on the k=2 column, which
  is nonzero for any node with a neighbour (97.5% of graphs). Labelling two columns costs
  +7% prompt length; labelling four would have cost +67%.

- **Report the RWSE/degree correlation**, with the aggregation named. See §6.

- **Per-node sentences**, matching the proposal's "one short factual sentence per node"
  and the surrounding graph prose.

- **An inert length control**, not a misinformation placebo. Wrong numbers would
  contradict the edge list in the same prompt, so a drop in accuracy could mean the
  model was misled, or merely confused by an inconsistent prompt — neither of which is
  the length effect the control exists to isolate. The control states only true,
  structurally vacuous facts. It survived three independent attempts to show it leaks
  node count or reads as a degree claim; the decisive counter is that the encoding's
  first line already ends `and <n-1>.`, so the filler introduces no numeral the `none`
  arm lacks.

- **A seventh condition: number of connected components**, with the caveats in §5.
  `docs/features-considered.md` records the features that were evaluated and rejected.

- **Canonicalise at the parse boundary.** `canonical()` rebuilds a graph so insertion
  order is sorted order for both nodes and adjacency. Across all 24 edge-insertion
  permutations of one 5-node graph the raw encoder produces 22 distinct texts; with
  `canonical()`, exactly 1.

- **`expected_answer` moves into the package.** An earlier draft kept it in
  `scripts/draw_graph.py` as "verification tooling, not pipeline code". That is no longer
  true: the shortcut-ceiling work (`docs/plans/shortcut-ceilings.md`) needs gold answers,
  so it becomes pipeline code.

### Correction to the proposal's task taxonomy

The proposal sorts tasks into primer-aligned, primer-adjacent and primer-agnostic. An
earlier draft of this plan corrected it once, concluding that only `edge_existence`
remained genuinely agnostic. **That conclusion is also wrong.** A systematic audit of all
seven conditions against all six tasks found deterministic or near-deterministic routes
into every task for the `degree` and `all` arms.

Verified routes, measured with each task's own query sampling:

| condition | task | what it gives away | rate |
|---|---|---|---|
| `degree`, `rwse`, `all` | `connected_nodes` | degree 0 (equivalently an all-zero RWSE vector) means the answer is `" No nodes."` | 9.0% of rows |
| `degree`, `all` | `edge_existence` | degree 0 forces No; degree n−1 forces Yes | 27.9% ± 1.3 of rows, 100% precision |
| `degree`, `all` | `edge_existence` | the rule "Yes iff d_a + d_b > n−1", one comparison on two stated numbers | 79.2% ± 1.7 accuracy vs a ≈51.6% baseline |
| `clustering`, `rwse`, `all` | `cycle_check` | clustering > 0 and RWSE(k=3) > 0 are the same triangle test; they agree on 100% of graphs | fires on 80.8% at 100% precision; 97.6% accuracy vs 83.2% baseline |
| `components` | `node_count` | c = n exactly when m = 0, so on an edgeless graph the primer prints the answer | 1.2% of rows, STATED |
| `components` | `connected_nodes` | c = 1 means no node is isolated, deterministically deleting the `" No nodes."` answer | 73.6% of rows |
| every per-node condition | `node_count` | the primer emits one sentence per node including isolated ones, while the encoding body omits them | changes a line-count from 74% to 100% correct |

For scale on the third row: the paper reports 45.1% for PaLM on ER `edge_existence`. A
one-comparison rule over two numbers the primer states scores 79.2%.

Two findings that bound the damage rather than extend it:

- **`connected_nodes` leak-hunting is closed.** Under the incident encoding, the gold
  answer string is byte-identical to the node list on the queried node's own line for
  every non-isolated target (4491/4491 rows). So rows partition into ~90% where the
  answer is a verbatim sentence of the prompt and ~10% where it is conveyed only by an
  absent sentence. There is no third kind, and only the second can be a primer effect.
- **The `filler` control is genuinely inert** on every route tested.

**Consequence for the design.** There is no reliable agnostic tier, so the taxonomy
cannot be asserted; it has to be measured. That is what `docs/plans/shortcut-ceilings.md`
does, and it replaces the sampling filter an earlier draft proposed. Nothing in the
statistics or the renderer changes as a result — suppressing the degree-0 sentence would
make primer content depend on an encoder quirk and would break the one-renderer property.

## Approach

### 1. Lift shared code into a package

`parse_graph` currently lives in `scripts/draw_graph.py` and was written to be moved.
Create a `graphtalk/` package and move it, along with the HTTP row fetching and the
gold-answer logic:

- `graphtalk/graphqa.py` — `parse_graph(question)`, `canonical(graph)`,
  `fetch_rows(config, split, offset, length)`, `expected_answer(graph, config,
  task_description)`, `normalize(answer)`
- `graphtalk/primers.py` — statistics and renderers

`parse_graph` returns a canonical graph. `canonical` is separately exported so that
graphs from other sources (tests, the generator) can be normalised the same way:

```python
def canonical(graph: nx.Graph) -> nx.Graph:
  """Rebuilds so insertion order is sorted order, for nodes and adjacency.

  The vendored encoder iterates graph.nodes() and graph.neighbors() in insertion
  order, so identical graphs built differently render differently. Normalising here
  means the encoder inherits sorted order without being modified.
  """
  out = nx.Graph()
  out.add_nodes_from(sorted(graph.nodes()))
  out.add_edges_from(sorted(tuple(sorted(e)) for e in graph.edges()))
  return out
```

Then update `scripts/draw_graph.py` to import from `graphtalk.graphqa`. Its `check` and
`draw` logic stays where it is.

Add `graphtalk` to `packages` and `tests` to `testpaths` in `pyproject.toml`, then
re-run `uv pip install -e ".[dev]"`.

### 2. Statistics (`graphtalk/primers.py`)

Three per-node functions, each returning a dict keyed by node id, **built in sorted node
order** so that the renderer's sentence order and the diagnostic's vector alignment both
follow for free:

- `degrees(graph) -> dict[int, int]` — thin wrapper over `graph.degree`
- `clustering(graph) -> dict[int, float]` — wrapper over `nx.clustering`
- `rwse(graph, k_min=2, k_max=3) -> dict[int, dict[int, float]]` — the diagonal of `P^k`
  for `P = D⁻¹A`, keyed by k so callers never index by position

and one graph-level function:

- `component_count(graph) -> int` — wrapper over `nx.number_connected_components`.
  Isolated nodes each count as their own component, which is what `networkx` already
  does and what the circuit-rank identity requires.

Two implementation requirements, both load-bearing:

**Pass `nodelist=sorted(graph.nodes())` to `nx.to_numpy_array`, and pair the diagonal
back against that same list.** Row *i* of the matrix is the *i*-th inserted node, not
node *i*. Getting this wrong attaches every node's values to a different node, produces
numbers that are all individually plausible, and shows up as a low or negative
correlation in §6 — which reads as a finding rather than a defect. `canonical()` makes
the default correct too; keep the explicit `nodelist` anyway so `primers.py` is correct
for any graph it is handed.

**Pin the accumulation order and comment it.** Accumulate as `M = M @ P`, never
`P @ M`. Both satisfy "repeated matrix multiply" and they disagree in the rendered text:
for the 8-node graph with edges `[(0,2),(0,3),(0,5),(1,4),(1,7),(2,3),(2,5),(3,4),(3,5),
(3,7)]`, node 4 at k=4 has exact value 53/200, and `M @ P` renders `0.26` while
`P @ M` renders `0.27`. The rendering fix in §3 makes this moot, but pinning the order
costs a comment and keeps the two defences independent.

Isolated nodes: degree-0 rows would divide by zero, so clamp the divisor to 1, which
yields an all-zero RWSE vector. Define it as a named constant with a comment. This is a
**convention that deliberately departs from the definition** — a walker on an isolated
node cannot move, so it is trivially still at its starting node, and a simulation of the
definition returns 1.0 where the clamp returns 0.0. 0.0 is chosen because it reads as
"no walk structure here"; assert it as a convention in the tests rather than letting it
fall out of the arithmetic.

Round only at render time, so the diagnostic is computed at full precision.

### 3. Renderer

**One** renderer produces the text for every condition. There is no per-feature renderer
and no combining step.

**Naming.** The pieces a primer is built from are called **parts**, not "components" —
"connected components" is one of the features, and the collision would be confusing in
code. The renderer parameter is `parts`.

**Every rendered float goes through one helper:**

```python
def _fmt(value: float) -> str:
  """Formats to two decimals, stably across BLAS builds and multiply orders.

  Values sitting exactly on a two-decimal boundary are odd/200, whose reduced
  denominator is 8, 40 or 200. Only 8 is a power of two, so only those are exactly
  representable; the rest land one bit either side of the boundary depending on the
  summation order, which differs across BLAS builds. Pre-rounding to six decimals
  removes that: a two-decimal tie needs at most 2**3 in its denominator and a
  six-decimal tie needs 2**7, so no value can be both, and the pre-round can never
  itself flip a digit.
  """
  return format(round(float(value), 6), ".2f")
```

Note that `_fmt(0.125)` is `"0.12"`, not `"0.13"` — Python rounds exact halves to even.
That is deterministic on every machine and is not a bug; it is recorded here so nobody
chases it during the hand-check.

**Node-level** parts each contribute an object phrase for a given node:

| node-level part | phrase |
|---|---|
| `degree` | `degree 4` |
| `clustering` | `clustering coefficient 0.70` |
| `rwse` | `return probability 0.27 after 2 steps and 0.15 after 3 steps` |
| `filler` | `7 other nodes in this graph` (the length control) |

**Graph-level** parts contribute a whole sentence:

| graph-level part | sentence |
|---|---|
| `components` | `This graph has 3 connected components.` |

The renderer emits graph-level sentences first, then Oxford-joins each node's requested
phrases into one sentence per node under a shared verb, joining everything with a space:

```
Node 0 has degree 4.
Node 0 has return probability 0.27 after 2 steps and 0.15 after 3 steps.
Node 0 has degree 12, clustering coefficient 0.70, and return probability 0.08 after 2 steps and 0.05 after 3 steps.
Node 0 has 7 other nodes in this graph.
This graph has 3 connected components.
```

Two phrases join with `and` and no comma; three or more take the Oxford comma. Because
the RWSE phrase now contains its own `and`, the `all` condition nests two — which parses
correctly, as the third example shows, and avoids breaking the one-sentence-per-node
property. Dropping to two k values also removed the comma-collision problem an earlier
draft worried about: there are no longer any commas inside the RWSE phrase.

Use the singular when a count is 1 (`1 connected component`, `1 other node`); a
grammatical slip in a condition that appears in every prompt of its arm is exactly the
kind of thing that could show up as a spurious effect.

```python
render_primer(graph, parts=(), k_min=2, k_max=3) -> str
```

`parts=()` yields the empty string — that is the `none` condition, not a special case.
`build_primer(graph, condition, ...)` is then a thin mapping from the seven condition
names onto part tuples.

This single code path is what makes the comparison mean anything. Fatemi et al.'s central
result is that phrasing alone moves accuracy by tens of points, so if conditions differed
in format as well as content, a difference between them would be uninterpretable. With
one renderer, format consistency is structural rather than a convention someone has to
remember.

Note what the invariant does **not** cover: the primer names every node, while the
encoding body names only non-isolated ones. That asymmetry is a property of the vendored
encoder, is the mechanism behind the `connected_nodes` and `edge_existence` routes above,
and is documented rather than fixed.

### 4. Length control

The `filler` part states a true fact that no task depends on:

```
Node 0 has 7 other nodes in this graph.
```

The phrasing fits the shared `Node N has ...` template, so the control comes out of the
same renderer as everything else. It contradicts nothing in the encoding and gives away
nothing structural — the encoding's first line already ends `and <n-1>.`, so the numeral
is not new.

Measured lengths, 500 graphs, final design:

| condition | mean primer chars |
|---|---|
| `none` | 0 |
| `components` | 37 |
| `degree` | 265 |
| `clustering` | 497 |
| `filler` | 507 |
| `rwse` | 905 |
| `all` | 1441 |

The control at 507 chars already exceeds `degree` (265) and matches `clustering` (497),
which is the safe direction: if 507 characters of inert text move accuracy by *d*, then
265 characters cannot have moved it by more than *d*. `rwse` and `all` are longer than
the control, which is what the optional `target_chars` padding exists for. Always report
achieved character counts so the mismatch is visible in analysis rather than assumed away.

The control also does a second job nobody designed it for. It emits a sentence for every
node, including isolated ones, without stating any structural fact about them. So
`filler` versus `degree` on isolated-target rows separates "the model was told node 3 is
isolated" from "node 3 was mentioned at all" — a salience control for the confound in the
taxonomy section.

### 5. The components condition

```
This graph has 3 connected components.
```

The case for it:

- **It completes an exact inference.** A graph has a cycle iff its circuit rank
  `m − n + c > 0`. Checked against `networkx` cycle detection on 500 graphs: **500/500**.
  The model must still count nodes and edges from the encoding and combine three numbers,
  which is the "supplies usable ingredients but must still combine them" pattern the
  proposal is built around. Note that recovering `m` from the incident encoding means
  counting neighbour-mentions and halving, because each edge is printed twice.
- **It costs one sentence, not one per node** — 37 characters, so the length confound
  barely applies. A matched control for this arm would be a single vacuous graph-level
  sentence, not the per-node filler.
- **The counts are not trivial.** 74% of graphs are connected, but the mean is 2.09 and
  the tail runs to 16 components, so the statement carries real variance.

Two caveats that an earlier draft did not have. The claim "it is not the answer to any of
the six tasks" — the reason a positive result could not be dismissed as trivial — does
not survive:

- **It states the `node_count` answer on edgeless graphs.** `c = n` exactly when `m = 0`,
  which is 1.2% of rows. And that is the worst case for the encoding: with no edges the
  encoder emits no body and not even the `In this graph:` line, so the prompt is one
  header line plus the primer.
- **It deletes the hard case of `connected_nodes`.** `c = 1` implies no node is isolated,
  so the `" No nodes."` answer is deterministically excluded on 73.6% of rows — and that
  is precisely the answer the encoding conveys only by omission.

Neither is a reason to drop the condition; both are reasons to report `node_count` and
`connected_nodes` under `components` with those strata marked. Restate the justification
as "not the answer on 98.8% of rows" rather than "not the answer to any task".

Note also why the variance in `c` cannot be preserved while excluding isolated nodes:
65% of the components in multi-component graphs are lone isolated nodes. Dropping graphs
that contain one collapses the distribution to mean 1.02, max 5, 98% connected.

The honest risk remains: models may simply not know the circuit-rank identity, in which
case this condition is a null. That is still an interesting null.

### 6. Diagnostic

```python
rwse_degree_correlation(graph, k_min=2, k_max=3) -> dict[int, float | None]
```

Pearson r per k between degree and RWSE, computed on unrounded values with
`numpy.corrcoef`. Returns `None` for a k where either vector is constant, rather than a
NaN. `scipy` is not a dependency and is not needed for this.

**This is a descriptive statistic, not an implementation check.** An earlier draft used it
as both, and it cannot do the second job: a low or negative r is also the symptom of the
node-mapping bug in §2, so the two are indistinguishable. The ordering test in
Verification does that job instead.

**Aggregation must be named, because the choice reverses the conclusion.** Use the mean
of per-graph r. Measured three ways on the same 1000 graphs at k=2:

```
mean of per-graph r = +0.62    Fisher-z mean = +0.87    pooling all nodes = -0.45
```

Pooling every node of every graph flips the sign, because within a graph higher degree
means higher return probability, while across graphs larger graphs have both higher
degrees and lower return probabilities (small graphs: mean degree 2.7, mean return 0.34;
large graphs: mean degree 8.8, mean return 0.15). Pooling measures graph size instead of
node degree.

Reference values, mean of per-graph r over 500 graphs:

| k | mean r | defined on | acceptance window (≈3 batch sd, batches of 100) |
|---|---|---|---|
| 2 | +0.65 | 481/500 (4% undefined) | [0.50, 0.75] |
| 3 | +0.89 | 391/500 (22% undefined) | [0.83, 0.94] |

The k=3 undefined rate is not noise: odd-k return is exactly 0 for every node of a
triangle-free graph, and 19% of graphs are triangle-free. So the k=2 and k=3 means are
computed over **different populations**, and any statement comparing them across k must
say so. Do not describe r as "rising" — with two values that framing invites a trend
claim the data does not support.

Clustering, by contrast, correlates with degree at r ≈ −0.24 and varies more across nodes
than degree does (coefficient of variation 0.44 vs 0.39) — so clustering is the genuinely
independent feature of the three, and RWSE is the redundant one.

## Files

- `graphtalk/__init__.py`, `graphtalk/graphqa.py`, `graphtalk/primers.py` — new
- `scripts/draw_graph.py` — drop its local `parse_graph`/`fetch_rows`/`expected_answer`/
  `normalize`, import from `graphtalk.graphqa`
- `scripts/show_primers.py` — new; prints all seven conditions for a few real rows, plus
  character counts and the correlation diagnostic, for eyeballing
- `tests/test_primers.py` — new
- `tests/golden/` — new; a handful of graphs and their exact expected primer strings
- `pyproject.toml` — add `graphtalk` to packages, `tests` to testpaths

## Verification

Analytic tests, no network, in `tests/test_primers.py`.

**Three implementations of RWSE, with distinct jobs.** This is the part an earlier draft
got wrong, so it is spelled out.

- `matrix` — the production code.
- `enumerate_weighted` — list every walk of length k and sum the walks that return,
  **each weighted by the product of 1/degree at every step it takes**. Exact; assert
  equality with the matrix version to 1e-12. An earlier draft specified the *unweighted*
  "return fraction", which is a different quantity: on the 5-node graph with edges
  `[(0,1),(0,2),(2,3),(2,4)]`, node 0 at k=2 has return fraction 0.50 and return
  probability 0.667. Implementing the wrong one would have made 33% of every RWSE number
  in the corpus wrong, on 81% of graphs.
- `simulate` — step a walker at random with a fixed seed and count returns. Approximate;
  assert agreement within 0.03. This is the only one of the three that can catch a
  *conceptual* error, because it never expresses the weighting as code — the weighting
  emerges from the walker's choices. Cost is 0.07 s for a 19-node graph, all nodes, both
  k, 5000 walks, with a worst observed disagreement of 0.013, so the tolerance has a
  fivefold margin and a fixed seed makes the test non-flaky.

**The load-bearing graph is lopsided, not regular.** Return fraction and return
probability coincide whenever every walk carries equal weight, which is guaranteed on
regular graphs. On the triangle, K4 and the star they agree at every node and every k, so
none of those can serve as the cross-check; only the path disagrees, and only at even k
at interior nodes. Include the 5-node lopsided graph above, and check every node rather
than one.

The rest:

- **Ordering** — build the same graph with shuffled node- and edge-insertion order and
  assert byte-identical primer text, and assert on an asymmetric graph (a star centred on
  a high-numbered node) that each node receives its own values. This is the test that
  catches the node-mapping bug; nothing else can.
- **Rendering stability** — assert `M @ P` and `P @ M` accumulation render identically on
  the 53/200 tie graph above. A dyadic tie such as 1/8 is bit-identical on every path and
  would pass on a broken implementation, so the case must be non-dyadic.
- **Golden file** — a committed fixture of graphs and their exact primer strings, so a
  numpy or platform change fails a test locally instead of surfacing as an unexplained
  cluster diff. Same-process repetition cannot detect that and must not be relied on.
- **Closed forms** — triangle: degree 2, clustering 1.0, RWSE k=2 → 0.5 and k=3 → 0.25.
  K4: clustering 1.0, degree 3. Star and path: clustering 0 everywhere; both bipartite,
  so k=3 is exactly 0.
- **Conventions, asserted as conventions** — isolated node: degree 0, clustering 0, RWSE
  all zeros, *with a comment that the definition-as-simulated gives 1.0*. And k=1 is 0
  whenever `k_min=1` is requested.
- **Component count** — connected graph → 1; two disjoint triangles → 2; a graph with an
  isolated node counts that node as its own component.
- **Circuit-rank identity** — across trees, forests, cycles, disjoint unions and graphs
  with isolated nodes, `m − n + c > 0` agrees with `nx.find_cycle`. Load-bearing for §5.
- **Singular/plural** — `1 connected component`, `1 other node`.
- **Length control is inert** — its text contains no degree, clustering or RWSE value,
  and is identical for any two graphs with the same node count.
- **Rendering** — two-decimal formatting via `_fmt` only, one sentence per node, sorted
  node order, `and` with no comma at two phrases, Oxford comma at three.
- **Format invariance** — every node-level condition's sentences match the same
  `Node N has <phrases>.` shape. The `components` sentence is graph-level and is
  deliberately exempt; assert that shape separately rather than asserting a single shape
  across all seven.

```bash
uv pip install -e ".[dev]"          # required: see Environment
uv run --no-sync pytest tests/ -q
```

Then a spot check against real data, which is where wording gets judged:

```bash
.venv/bin/python scripts/show_primers.py --config node_degree --count 3
```

Read the printed primers and confirm the sentences are well-formed, the numbers match a
hand-check on the smallest graph, and the length control states nothing structural. Do not
skip this by assuming the tests cover it — the tests check arithmetic, not whether the
English reads like the surrounding graph prose.

`show_primers.py` also prints per-graph correlation values **labelled as information, not
as a pass/fail check**, together with a count of how many were undefined. Per-graph r
ranges from −0.77 to +1.00 and is undefined on a fifth of graphs at k=3; a three-row
sample lands entirely inside the reference window 0.2% of the time, prints a negative
value on 20% of runs and an undefined one on 53%. The acceptance criterion is the
corpus-level window in §6 over at least 100 graphs, not the per-graph print.

Also re-run the previous step's verification, since `scripts/draw_graph.py` is being
refactored to import from the new package:

```bash
.venv/bin/python scripts/draw_graph.py --config connected_nodes --index 0 --count 20
```
