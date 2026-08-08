# Graph features considered, and why most were rejected

The primer carries node degree, local clustering coefficient, RWSE, and — added later —
the number of connected components. This records what else was evaluated and why it was
left out, so the choice reads as a decision rather than an oversight.

## The selection criterion

A feature earns a place in the primer only if it passes all four tests:

1. **Relevance** — it bears on at least one of the six tasks.
2. **Non-triviality** — it is not the answer to any of them.
3. **Non-redundancy** — it is not recoverable from a feature already included.
4. **Expressibility** — it means something when stated in English.

Test 1 is stricter here than it first appears. The six tasks are: count nodes, count
edges, degree of a node, list a node's neighbours, does an edge exist, is there a cycle.
**Five are counting or adjacency lookup; only cycle check requires global structure.**
Most of the graph-theoretic feature literature is about node position and importance,
which none of these tasks ask about.

## Rejected: gives away the answer

These fail test 2. Including them would turn a task into a manipulation check rather than
an experiment.

| Feature | Task it answers |
|---|---|
| Edge count `m` | `edge_count`, directly |
| Sum of degrees | `edge_count`, being `2m` |
| Node count `n` | `node_count` — and it is already in the encoding's first line |
| Global triangle count | `cycle_check`: any triangle implies a cycle |
| Clique number | `cycle_check`: a clique of size ≥ 3 is a triangle |
| `is_tree` / `is_forest` | `cycle_check`, literally |
| Per-node adjacency list | `connected_nodes` and `edge_existence` — and it *is* the Incident encoding |

Note that degree already fails this test for `node_degree`. That is deliberate and
acknowledged in the proposal: that task is the manipulation check confirming the model
reads the primer at all.

## Rejected: bears on none of the six tasks

These fail test 1. Each is a reasonable feature answering "how central, how far, or how
deeply embedded is this node" — a question no task in this suite asks.

- Betweenness centrality
- Closeness centrality
- Eigenvector centrality and PageRank
- k-core number
- Diameter, eccentricity, average shortest path length
- Degree assortativity
- Shortest-path distances from the queried node
- Two-hop neighbourhood size

Adding any of them would be fishing: no prediction motivates them, and each costs a
full cluster sweep of six tasks by four models by two prompting styles.

These become interesting the moment the task set widens. GraphQA also ships
`reachability`, `shortest_path`, `triangle_counting`, `maximum_flow`, and
`node_classification`, and distance- and centrality-based features bear directly on
those. If the project extends beyond the six basic tasks, revisit this section rather
than this file's conclusion.

## Rejected: redundant with what is already there

These fail test 3 — they are recoverable from features the primer already carries.

- **Per-node triangle count.** A deterministic function of the two features already
  present: `T_i = C_i · d_i(d_i − 1) / 2`.
- **Average neighbour degree / neighbour degree list.** RWSE at k=2 already summarizes
  this, being `(1/d_i) · Σ_{j~i} 1/d_j`.
- **RWSE beyond k=5.** Return probability converges to the stationary distribution
  `d_i/2m`; measured correlation with degree is already r≈0.94 at k=5. Longer walks
  restate degree with more decimals.
- **Local efficiency.** Monotone in clustering at these graph sizes.

Worth recording that the redundancy in the existing feature set is not where it was
expected. Clustering was assumed to be near-constant across nodes in Erdős–Rényi graphs
(expected value ≈ p for every node); measured, it varies more than degree does
(coefficient of variation 0.49 vs 0.38) and is essentially uncorrelated with degree
(r ≈ −0.16). **Clustering is the genuinely independent feature; RWSE is the redundant
one.**

## Rejected: does not survive the move to text

These fail test 4. They are real structural encodings that stop meaning anything once
written as a sentence.

- **Laplacian positional encoding (LapPE).** The natural sibling of RWSE, from the same
  GraphGPS paper the proposal cites, so its absence needs an explicit reason.
  Eigenvectors are defined only up to sign, and degenerate eigenvalues leave the basis
  non-unique — the same graph yields different coordinates on different runs. A number
  whose sign is arbitrary cannot be stated as a fact in a sentence. RWSE has no such
  problem: it is a probability, and "the chance of returning after three steps is 0.19"
  is a claim the reader can interpret.
- **Learned or spectral embeddings generally.** Same objection: the coordinates are
  meaningful relative to a basis the reader cannot see.
- **Graph tokens** (from *Let Your Graph Do the Talking*, the companion paper in the
  vendored repo). Requires access to model internals, which contradicts the black-box
  premise of the whole project.

This contrast is arguably a finding rather than a footnote: some structural encodings
transfer from learned embedding to natural language and some cannot, and the dividing
line is whether the number survives without its basis.

## Adopted: number of connected components

Passes all four tests, and is the only evaluated feature that does.

- **Relevant** — with `n` and `m`, which the model can count from the encoding, it
  determines cycle check exactly via the circuit rank `m − n + c > 0`. Verified against
  ground truth on 100 rows: 100/100.
- **Non-trivial** — it is not the answer to any of the six tasks.
- **Non-redundant** — no combination of degree, clustering, or RWSE recovers it.
- **Expressible** — "This graph has 3 connected components" needs no gloss.

It also costs one sentence rather than one per node, so it barely interacts with the
prompt-length confound that shaped the inert control.

## A caveat on the whole exercise

Feature selection turned out to matter less than task classification. The proposal calls
cycle check primer-agnostic, but the clustering primer already supplies a decision
procedure for it at 100% precision and 97% recall — see the taxonomy correction in
`docs/plans/primer-computation.md`. Fixing that classification costs no cluster time and
changes the interpretation of the results more than any additional feature would.
