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
PYTHONPATH=. .venv/bin/python scripts/score_sweep.py --responses $(ls runs/*.jsonl | grep -v '\.got\.') --shortcuts shortcuts.json
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

Chain-of-thought is measured by the thinking arm (the `-think` specs, native
reasoning at `zero_shot`) rather than by a separate prompt style.

At the proposal's 30 rows per task that is 1,260 prompts per model: 180 instances
x 7 conditions, at `zero_shot` (the published dataset's own wording).

## Node naming

Every prompt names nodes one of two ways, chosen with `--node-naming` on
`build_prompts.py`:

- **`integer`** (default) — nodes stay `0, 1, 2, ...`, the published
  dataset's own scheme and what every other section of this README assumes.
- **`got`** — nodes are renamed to Game-of-Thrones characters (`Ned`,
  `Catelyn`, `Daenerys`, ...) throughout the primer, the encoding, and the
  question, via `graphtalk/node_naming.py`. Additive rather than a variant
  code path: a named prompt is the ordinary integer prompt with node
  references substituted after the fact, and a named response is converted
  back to integers before it reaches the same, unmodified scorer — nothing
  in `primers.py`, `prompts.py`, `graphqa.py`, or `scoring.py` is touched.

```bash
# integer node ids (default)
PYTHONPATH=. .venv/bin/python scripts/build_prompts.py --count 30

# GoT character names -- --out keeps this from overwriting prompts.jsonl.
PYTHONPATH=. .venv/bin/python scripts/build_prompts.py --count 30 \
    --node-naming got --out prompts_got.jsonl
```

Generation is the same `cluster/sweep.sbatch` as [the sweep](#the-sweep)
above, pointed at the named prompt file and tagged so its output doesn't
collide with the integer run's — both env vars the script already supports:

```bash
GRAPHTALK_PROMPTS=prompts_got.jsonl GRAPHTALK_RUN_TAG=got \
    sbatch cluster/sweep.sbatch gemma4-12b
```

That writes `runs/gemma4-12b.got.jsonl` next to the integer run's
`runs/gemma4-12b.jsonl`, rather than replacing it.

**Or skip both manual steps with `cluster/submit_sweep.sh`**, which builds
`prompts_got.jsonl` first if it doesn't exist yet (reusing it otherwise) and
sets both env vars for you:

```bash
cluster/submit_sweep.sh --node-naming got cluster/sweep.sbatch gemma4-12b
```

Every other `sbatch` flag/positional passes straight through unchanged, so
this is a drop-in replacement for the word `sbatch` in any of the invocations
above or in [cluster/README.md](cluster/README.md) — `--dry-run` prints what
it would do without submitting anything. Building the prompt file happens in
the wrapper itself, on the login node, not inside the SLURM job: compute
nodes have no outbound network, which is why `sweep.sbatch` sets
`HF_HUB_OFFLINE=1` in the first place.

**Score the two schemes separately, not with one `runs/*.jsonl` glob** — no
flag needed to get this right, it's enforced. Every named response's
`node_naming` field is enough for `score_sweep.py`/`build_sweep_frame.py` to
desubstitute GoT names back to integers automatically before scoring, and
`build_sweep_frame.py`, `sample_failures.py`, and `check_significance.py`
all **infer the scheme from the data and raise rather than silently pooling**
if their input ever carries more than one — pooling both schemes' files for
the same model would otherwise put two rows under the identical
`(instance_id, condition, style, model)` key, which
[docs/DATA.md](docs/DATA.md#the-pairing-key) requires to be unique within a
scheme:

```bash
PYTHONPATH=. .venv/bin/python scripts/build_sweep_frame.py \
    --responses runs/gemma4-12b.jsonl --shortcuts shortcuts.json
# -> analysis/sweep_frame.csv

PYTHONPATH=. .venv/bin/python scripts/build_sweep_frame.py \
    --responses runs/gemma4-12b.got.jsonl --shortcuts shortcuts.json
# -> analysis/sweep_frame.got.csv
```

Each `--out` left unset lands at its own scheme-tagged filename automatically
(`analysis.tagged_path`) — `sweep_frame.csv`/`failure_sample.csv` for
`integer`, `sweep_frame.got.csv`/`failure_sample.got.csv` for `got`, and
likewise for `check_significance.py --out`. Run the two side by side to ask
whether accuracy depends on node identity rather than graph structure. The
existing `shortcuts.json` is still the bar for both — `shortcut_table.py`
generates its own graphs and integer primers internally and never imports
`node_naming`, so the ceiling it measures (how much of a primer's
degree/clustering/etc. facts a solver can recover) does not depend on how a
downstream prompt happens to label the nodes. `shortcuts.py` itself is
integer-only, though (`_NODE_RE` expects `Node (\d+) ...`), so it's the
model's GoT-worded *response* that needs desubstituting before scoring,
never a primer that needs running through `shortcuts.py` directly.

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

418 tests: 27 vendored ones covering graph generation, text encoders and
metrics, 138 covering the primer statistics, the renderer, and the committed
golden primer strings, 143 covering the shortcut solvers, 72 covering prompt
assembly and answer scoring, 27 covering node naming and the GoT round trip,
and 11 covering the sweep frame, the failure taxonomy, the wording split, and
how a row's non-termination flag was obtained.

The last of those need `pandas`, which is not in the base install: without
`pip install -e ".[analysis]"` the suite fails at *collection* rather than
skipping, so the whole run aborts and none of the other 407 report.

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
