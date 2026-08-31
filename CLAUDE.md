# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A course project (Machine Learning with Graphs) that builds on
[Talk like a Graph: Encoding Graphs for Large Language Models](https://arxiv.org/abs/2310.04560)
(arXiv:2310.04560). It tests whether prepending a *primer* — a short preamble of
factual sentences about each node's local structure — before the standard graph
text encoding and question improves an LLM's accuracy on GraphQA tasks, and
separately measures how much of that accuracy a primer-only (no-graph) solver can
already reach without seeing the graph at all.

## Setup

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
```

Python is pinned to `>=3.11,<3.12` (`pyproject.toml`) because `seqio` and
`tensorflow-gnn` don't resolve cleanly on 3.12+.

Optional TensorFlow pipeline (only `graph_tasks_utils.py` needs it, ~2 GB):

```bash
uv pip install -e ".[pipeline]"
```

`tensorflow_gnn` 1.0.3 requires Keras 2, but TF 2.20 ships Keras 3, so
`TF_USE_LEGACY_KERAS=1` must be set before importing it (`.venv/bin/activate`
exports it already).

## Commands

Run the full test suite:

```bash
uv run --no-sync pytest -q
```

Always use `--no-sync` — a plain `uv run` re-syncs to the default dependency set
and uninstalls the optional `pipeline` extras. 345 tests total, plus 23 more in
`tests/test_node_naming.py` (368). On the cluster, `pytest -q` must report
exactly that many passed; a different number means the env is wrong, not the
code.

Run a single test file or test:

```bash
uv run --no-sync pytest -q tests/test_primers.py
uv run --no-sync pytest -q tests/test_primers.py::test_round_trip
```

Score the shortcut-ceiling table (what a primer-only solver, with no access to the
graph, scores — the bar a model result has to clear):

```bash
PYTHONPATH=. .venv/bin/python scripts/shortcut_table.py --graphs 500
```

Full three-stage sweep (only stage 2 needs a GPU — see `cluster/README.md` for
running it on the TAU CS cluster):

```bash
# 1. build every prompt to a file, on the laptop/login node
PYTHONPATH=. .venv/bin/python scripts/build_prompts.py --count 30
# node identifiers default to plain integers; pass --node-naming got for
# Game-of-Thrones character names instead (see graphtalk/node_naming.py) --
# build and score each naming scheme as a separate run to compare them

# 2. generate, on a GPU node, once per model
sbatch cluster/sweep.sbatch gemma4-12b

# 3. score, back on the laptop
PYTHONPATH=. .venv/bin/python scripts/shortcut_table.py --graphs 500 --json shortcuts.json
PYTHONPATH=. .venv/bin/python scripts/score_sweep.py --responses runs/*.jsonl --shortcuts shortcuts.json
```

Check statistical significance beyond `score_sweep.py`'s per-cell McNemar (that test is
underpowered at 30 pairs/cell — see `docs/sweep-findings.md`). Needs the `analysis` extra
(`uv pip install -e ".[dev,analysis]"`) and the joined sweep table built first:

```bash
PYTHONPATH=. .venv/bin/python scripts/build_sweep_frame.py --responses runs/*.jsonl \
    --shortcuts shortcuts.json --truncated-keys analysis/truncated_keys.json
PYTHONPATH=. .venv/bin/python scripts/check_significance.py --frame analysis/sweep_frame.csv
```

`check_significance.py` pools pairs across task and style per (model, condition) instead of
testing 288 tiny cells, reporting a permutation p-value, a bootstrap CI on the effect size,
and a Benjamini-Hochberg correction — for both main-sweep accuracy and thinking-arm
non-termination rate. Pass `--out <path.csv>` to save the printed rows instead of only
seeing them in the terminal.

Other one-off scripts:

```bash
python scripts/draw_graph.py --config node_degree --index 0   # parse+draw a GraphQA row
python scripts/show_primers.py --generated 3                  # eyeball rendered primer text
python scripts/measure_real_rows.py                           # re-measures corpus stats against real HF rows
```

## Architecture

### Package layout

- `talk_like_a_graph/` — a **vendored, mostly-unmodified copy** of Google
  Research's reference implementation (graph generators, text encoders, task
  generators, metrics). See `talk_like_a_graph/UPSTREAM.md` for the exact commit
  and the two local modifications (test class MRO fixes). Treat this directory as
  third-party code; prefer changing `graphtalk/` over editing it.
- `graphtalk/` — this project's own package:
  - `graphqa.py` — fetches GraphQA rows over the HF `datasets-server` rows API
    (deliberately *not* the `datasets` library, which brings pyarrow/fsspec pins
    that would disturb the hand-tuned TF/tf-keras/tensorflow-gnn combination),
    parses a networkx graph back out of a row's `question` prose, and recomputes
    gold answers (`gold_answer` / `expected_answer`). `canonical()` normalizes
    node/edge insertion order so re-encoding the same graph is reproducible.
  - `primers.py` — primer statistics (degree, clustering, RWSE, connected
    components) and `render_primer` / `build_primer`, the **single renderer**
    every experimental condition goes through. Every rendered float goes through
    `_fmt` (round-to-6-then-format-to-2 decimals, to sidestep BLAS-order tie
    instability). Graphs must arrive already canonicalized.
  - `shortcuts.py` — the primer-only solvers, deliberately **sharing no code with
    `primers.py`**: a strict parser (`parse_primer`) that reads rendered primer
    text back into structured data (re-deriving the join/separator rules rather
    than importing `primers._join`, so the round-trip test is a real
    cross-check, not a tautology), 16 exact "theorem" rules, 1 heuristic, 8
    fitted rules (which must be fit/scored on disjoint graph sets via `Split`),
    and an exact enumeration bound (`exact_island`) for small graphs (≤6 nodes).
    Parsing is strict — an unrecognized or conflicting sentence raises rather
    than being silently skipped.
  - `prompts.py` — assembles one prompt as `primer + "\n\n" + encoding +
    task_description [+ CoT suffix]`. Uses the `incident` encoding (not
    `adjacency`, which is what the published dataset's `question` field
    contains — that's why prompts are rebuilt from the parsed graph rather than
    reused verbatim).
  - `scoring.py` — answer extraction from free model text (per-task regex logic,
    tuned for CoT responses that reason before concluding) and the metrics named
    in the proposal: exact match for integer/boolean tasks, set-F1 for
    `connected_nodes`, plus MAE, majority baseline, and exact McNemar.
  - `node_naming.py` — Game-of-Thrones node naming, additive on top of the
    integer pipeline rather than a change to it: nothing in `primers.py`,
    `prompts.py`, `graphqa.py`, or `scoring.py` is modified. A named prompt is
    built by rendering the ordinary integer primer/encoding/question exactly as
    today and substituting node-id references for names as a text pass
    (`build_named_prompt`); a named model response is desubstituted back to
    integers (`desubstitute_response`) before it reaches the existing,
    unmodified scorer. Also patches around a vendored bug where
    `incident_encoder`'s per-node sentence opener hardcodes the raw node id
    regardless of the name dict it's given.
  - `models.py` — model configs only (`ModelSpec`), deliberately free of `torch`/
    `transformers` so prompt-building and scoring stay importable without a GPU
    stack.
  - `hf_backend.py` — the only module that imports `torch`/`transformers`;
    loading and greedy generation. Imported only by `scripts/run_sweep.py`.
- `scripts/` — the three pipeline stages (`build_prompts.py`, `run_sweep.py`,
  `score_sweep.py`) plus `shortcut_table.py`, `draw_graph.py`,
  `show_primers.py`, `measure_real_rows.py`.
- `cluster/` — `sweep.sbatch` and `README.md`, the authority on how the sweep
  actually runs on the TAU CS cluster (partitions, memory sizing, driver
  incompatibilities, chained-job submission for jobs that exceed the 24h
  partition limit).
- `docs/` — `sweep-findings.md` (results and their caveats) and `docs/plans/`
  (`shortcut-ceilings.md`, `primer-computation.md`) which explain what the
  measured numbers mean; read these before interpreting a new sweep result.

### Core design invariants

These are asserted by tests and referenced throughout the code — don't casually
break them:

- **One renderer.** All seven primer conditions (`none`, `components`, `degree`,
  `clustering`, `rwse`, `filler`, `all`) go through `render_primer`, so they
  differ in content only, never in format. This is what makes a difference
  between conditions interpretable.
- **The shortcut solver never sees the graph.** `shortcuts.py` operates on
  rendered primer *text*, not on graph objects or full-precision statistics —
  structurally, not just by discipline (there's no graph parameter to pass). This
  is what makes the shortcut score a meaningful lower bound on primer-only
  performance rather than something that could cheat.
- **The round trip is the cross-check.** `render_primer` → `parse_primer` must
  recover the rounded originals exactly. Because the parser restates the
  renderer's join/format rules instead of importing them, this test catches
  renderer changes that would otherwise pass silently. If you change
  `render_primer`'s output format, update the parser in the same change and
  expect `test_primers.py`'s round-trip test to fail until you do.
- **Fitted rules must be fit/scored on disjoint graph sets.** `shortcuts.Split`
  enforces different fit/test seeds structurally (raises if they're equal).
  Fitting and scoring a rule on the same graphs inflates its accuracy and
  invalidates the shortcut bar as a fair comparison point for model results.
- **Golden primer strings.** `tests/golden/primers.json` pins exact rendered
  text. Regenerate deliberately (see `tests/test_primers.py`'s module docstring)
  and read the diff before committing — an unintended diff here means the
  renderer changed in a way that would also silently shift every downstream
  prompt and shortcut number.

### Testing conventions

- 345 tests: vendored generator/encoder/metric tests, primer statistics/renderer/
  golden-string tests, shortcut-solver tests, prompt-assembly/scoring tests.
- Theorem rule precision is asserted at exactly 1.0 over both an Erdős–Rényi
  corpus and an adversarial corpus (trees, forests, cycles, complete bipartite
  graphs) — the ER generator alone never produces a tree, so a rule that's
  secretly keyed on the `m = n-1` boundary can pass on ER data alone.
- Network access (`graphqa.fetch_rows`) is only exercised by scripts, not by the
  test suite — tests use the vendored generator for graphs.
