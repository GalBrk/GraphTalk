# GraphTalk

Course project building on
[Talk like a Graph: Encoding Graphs for Large Language Models](https://arxiv.org/abs/2310.04560).

`talk_like_a_graph/` is a vendored copy of Google Research's reference
implementation. See [talk_like_a_graph/UPSTREAM.md](talk_like_a_graph/UPSTREAM.md)
for the exact upstream commit and our local changes.

`graphtalk/` is this project's own package: `graphqa.py` recovers a networkx
graph from a GraphQA row and recomputes its gold answer, `primers.py` holds the
primer statistics and the single renderer every condition goes through, and
`shortcuts.py` holds the primer-only solvers: a strict parser that reads a
rendered primer back out of its text, sixteen theorem rules, one heuristic and
eight fitted rules, and the exact enumeration bound for small graphs. See
[docs/plans/shortcut-ceilings.md](docs/plans/shortcut-ceilings.md) for what the
resulting numbers mean.

Three of those theorems *reconstruct* rather than compare: the stated degree
sequence constrains which graphs are possible, often to exactly one, so a primer
that states no adjacency at all still gives away whole neighbour lists. That is
what makes `connected_nodes` a 35.2% cell rather than the 8.2% one it was first
measured as.

## The shortcut table

```bash
PYTHONPATH=. .venv/bin/python scripts/shortcut_table.py --graphs 500
```

Scores a solver that reads only the primer and never the graph, across 7
conditions x 6 tasks x 3 rungs. That score is the bar a model has to clear: a
model at or below it did the primer arithmetic and nothing more. Four of the six
tasks turn out to be already decided this way.

The table sorts the sweep rather than pruning it. A 100% shortcut is what a
Python program scores, not what a model scores — on `node_count` the shortcut is
100% and the paper reports 18.8% for PaLM 2 on the encoding GraphQA ships. So a
decided cell still answers a question, just a narrower one: whether the model
uses a fact it was handed, rather than whether it reasoned about the graph.

## The sweep

Three stages, and only the middle one needs a GPU. See
[cluster/README.md](cluster/README.md) for running it on the TAU CS cluster.

```bash
# 1. build every prompt to a file, on the laptop
PYTHONPATH=. .venv/bin/python scripts/build_prompts.py --count 30

# 2. generate, on a GPU node, once per model
sbatch cluster/sweep.sbatch gemma4-12b

# 3. score, back on the laptop
PYTHONPATH=. .venv/bin/python scripts/shortcut_table.py --graphs 500 --json shortcuts.json
PYTHONPATH=. .venv/bin/python scripts/score_sweep.py --responses runs/*.jsonl --shortcuts shortcuts.json
```

Step 2 is written as one job per model for readability. On the TAU cluster a
model needs more than the 24-hour partition limit, so it is really a *chain* of
resuming jobs with a per-model memory request — see
[cluster/README.md](cluster/README.md), which is the authority on how the sweep
is actually launched.

The prompt set is written to a file first so it can be read and diffed before any
GPU time is spent, and so every model in the sweep is handed the identical file.
The design is paired — the same graph and query appear under all seven conditions
and both prompt styles — which is what the proposal's McNemar test requires.

**`zero_cot` is retired.** Chain-of-thought is measured by the thinking arm (the
`-think` specs, native reasoning at `zero_shot`), which superseded it. The 5,040
`zero_cot` rows in the sweep are kept as a record and are still read by the
analysis, but nothing new is generated in that style — pass `--styles zero_shot`
to `build_prompts.py`. See `graphtalk/prompts.py` for what that means when
reading those rows.

At the proposal's 30 rows per task that is 2,520 prompts per model: 180 instances
x 7 conditions x 2 prompt styles (`zero_shot` and `zero_cot`, both using the
published dataset's own wording).

## Setup

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
```

That covers the graph generators, tasks, text encoders, and metrics — which is
everything except `graph_tasks_utils.py`.

### Optional: the TensorFlow pipeline

`graph_tasks_utils.py` alone needs `tensorflow`, `tensorflow-gnn`, and `seqio`
(~2 GB installed):

```bash
uv pip install -e ".[pipeline]"
```

`tensorflow_gnn` 1.0.3 requires Keras 2, but TF 2.20 ships Keras 3, so
`TF_USE_LEGACY_KERAS=1` must be set before importing it. `.venv/bin/activate`
exports it already; set it manually if you run the interpreter without
activating.

Python is pinned to 3.11 because `seqio` and `tensorflow-gnn` do not resolve
cleanly on 3.12+.

## Tests

```bash
uv run --no-sync pytest -q
```

359 tests: 27 vendored ones covering graph generation, text encoders and
metrics, 138 covering the primer statistics, the renderer, and the committed
golden primer strings, 143 covering the shortcut solvers, 40 covering prompt
assembly and answer scoring, and 11 covering the sweep frame, the failure taxonomy, the
wording split, and how a row's non-termination flag was obtained.

The last of those need `pandas`, which is not in the base install: without
`pip install -e ".[analysis]"` the suite fails at *collection* rather than
skipping, so the whole run aborts and none of the other 348 report.

Two of those deserve mention because they are what the rest rests on. The
**round trip** renders a primer, parses it back, and requires the recovered
values to equal the rounded originals — the only check that notices a renderer
change nobody meant to make, which is why `shortcuts.py` shares no code with
`primers.py`. And **theorem precision** must be exactly 1.0 over both an ER
corpus and an adversarial one of trees, forests, cycles and complete bipartite
graphs; the adversarial corpus exists because the ER generator produces no tree
at all, so a false rule keyed on the m = n-1 boundary scored a clean 1.0 without
it.

Use `--no-sync`: a plain `uv run` re-syncs the environment to the default
dependencies and would uninstall the optional `pipeline` extras.
