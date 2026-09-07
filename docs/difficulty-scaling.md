# Difficulty scaling: xlarge graphs, density, reachability, overflow guard

This documents four additive changes to the eval pipeline: bigger synthetic
graphs, denser/more complex topology, a new multi-hop `reachability` task,
and an input-token overflow guard. All four go through the *synthetic*
graph path (`--graph-source diverse`) rather than the fixed published
dataset, and none of them change any existing default's behavior — every
new parameter defaults to `None`/today's value, and every existing test
(606 passing before this work) still passes unchanged, plus the new tests
added alongside each change (607 total locally, `torch`/pytensor-dependent
tests aside — see "Testing" below).

## Why

Two independent findings motivated this:

- `analysis/README.md` flags `gemma4-12b`/`gemma4-e4b` as `near_ceiling=True`
  on the published dataset — 96-99% control accuracy, "almost no primer
  effect could show up there regardless of sample size." The published
  split is also capped at 5-19 nodes, Erdős–Rényi only, so there was no way
  to give those models a harder question without leaving the published
  data.
- `docs/plans/scale-vs-topology-investigation.md`'s driver analysis found
  the `degree` primer's benefit on `edge_count` "grows monotonically and
  substantially with graph size, degree-sequence variance, and density" —
  i.e. the primer helps most exactly where manual counting from the graph
  text is hardest. That is direct evidence that graph size/density is a
  real, load-bearing difficulty axis for this experiment, not a plausible
  guess.

Reachability was added because it's a strictly harder query than anything
currently asked (multi-hop path existence vs. `edge_existence`'s
single-edge check), and it was sitting fully implemented but unused in the
vendored `talk_like_a_graph/graph_tasks.py` (the `Reachability` class,
lines 574-652) — reusing its wording rather than inventing new phrasing.

## What changed

### 1. Xlarge graph-size bucket

`talk_like_a_graph/graph_generators.py::generate_graphs` (vendored) takes a
new optional `node_size_ranges: dict[str, np.ndarray] | None = None`
parameter. When omitted (every existing caller), it resolves to the
original `_NUMBER_OF_NODES_RANGE` (5-19 nodes across `small`/`medium`/
`large`) — **byte-identical to today**, verified by a new test
(`test_node_size_ranges_default_unchanged`) comparing generated edge lists
under the same seed. A caller that wants bigger graphs passes its own dict,
e.g. `{"xlarge": np.arange(20, 40)}`, without touching the vendored
module's own `_NUMBER_OF_NODES_RANGE` at all — the vendored dict was never
mutated, since every existing test/corpus (theorem-rule precision tests,
the shortcut-fitting corpus, `show_primers.py`) reads it with no override
and would have silently shifted size distribution otherwise.

One side effect had to be handled: the `sbm` algorithm additionally
indexes `_NUMBER_OF_COMMUNITIES_RANGE` by the same bucket name, which has
no `"xlarge"` entry. It now falls back to the `"large"` entry
(`_NUMBER_OF_COMMUNITIES_RANGE.get(bucket, _NUMBER_OF_COMMUNITIES_RANGE["large"])`)
instead of raising `KeyError`.

`graphtalk/diverse_corpus.py::build_pool` and
`scripts/build_prompts.py::build_diverse` both forward this parameter
through, ending in a `--xlarge` CLI flag.

### 2. Density and topology complexity

`graphtalk/diverse_corpus.py::build_pool` now forwards
`er_min_sparsity`/`er_max_sparsity` to `generate_graphs`, which already
supported them (the vendored function just was never asked for anything
but its 0.0-1.0 defaults). These only affect the `"er"` algorithm in the
7-algorithm pool — the vendored generator consumes them nowhere else, so
`ba`/`sbm`/`sfn`/`complete`/`star`/`path` graphs in the same pool are
unaffected. Reachable via `--er-min-sparsity`/`--er-max-sparsity`.

### 3. Reachability task

A new task, end to end:

- `graphtalk/graphqa.py::gold_answer` gained a `"reachability"` branch:
  `"Yes"` if `nx.has_path(graph, a, b)` else `"No"`.
- `graphtalk/diverse_corpus.py::make_row` gained a matching branch, wording
  copied verbatim from vendored `graph_tasks.py`'s `Reachability` class:
  `f"Q: Is there a path from node {source} to node {target}?\nA: "`.
- `graphtalk/scoring.py` deliberately does **not** add `"reachability"` to
  `TASKS` (still exactly the original 6 entries) — `TASKS` doubles as the
  default `--tasks` value that drives `graphqa.fetch_rows` against the real
  published HF dataset, which has no `reachability` config. Instead there's
  a new `ALL_TASKS = TASKS + ("reachability",)`, and the three places that
  used to gate on `task not in TASKS` (`extract_answer`,
  `extract_answer_first`, `score_one`) now gate on `ALL_TASKS`.
  `_BOOLEAN_TASKS` gained `"reachability"` so it's graded as Yes/No like
  `edge_existence`/`cycle_check`.
- `scripts/build_prompts.py --tasks` now accepts `scoring.ALL_TASKS`
  (default unchanged: still the original 6). Passing `--tasks reachability`
  with anything other than `--graph-source diverse` raises immediately with
  a clear message, rather than failing deep inside an HTTP call to a
  nonexistent dataset config.

### 4. Input-token overflow guard

Bigger, denser graphs make an old gap real for the first time: nothing in
the pipeline checked a prompt's *input* token length against a model's
context window — only output-side truncation (`hit_cap`/`n_new_tokens`)
was ever tracked.

- `graphtalk/models.py::ModelSpec` gained `max_context_tokens: int | None =
  None`. **Not yet populated for any model** — filling in each checkpoint's
  real published context length (Gemma 4 / Qwen3 model cards) is a
  necessary follow-up before this check has any real effect; until then
  it's a no-op, same as today.
- `graphtalk/hf_backend.py::generate`/`generate_batch` take a new
  `max_context_tokens` parameter and raise `ValueError` before generation
  if `prompt_len + max_new_tokens > max_context_tokens` — fails loudly
  rather than truncating or letting `model.generate` hit an opaque shape
  error. `scripts/run_sweep.py` passes `spec.max_context_tokens` at both
  call sites.
- `scripts/build_prompts.py --model <key>` prints an early, approximate
  warning (chars/4 heuristic, not a real tokenizer count — this script is
  deliberately `torch`/`transformers`-free) if any built prompt looks like
  it would overflow that model's budget. The exact, authoritative check is
  the one in `hf_backend.py`.

## Backward compatibility

Every new parameter defaults to `None`/today's value at every layer
(`generate_graphs` → `build_pool` → `build_diverse` → the CLI), so:
- `scripts/build_prompts.py --count 30` (no new flags) is byte-for-byte
  unchanged.
- `scoring.TASKS` is still exactly 6 entries; anything that assumed that
  (test parametrization, `build_diverse`'s Python-level default) is
  unaffected.
- The published (`--graph-source published`/`stratified`/`build_named`)
  paths never see `"reachability"` unless explicitly asked for it, and are
  guarded against being asked for it accidentally.

## Running locally

Build a harder synthetic prompt set (xlarge graphs, dense ER, plus the new
reachability task, alongside the original 6 tasks):

```bash
PYTHONPATH=. .venv/Scripts/python.exe scripts/build_prompts.py \
    --graph-source diverse \
    --xlarge --er-min-sparsity 0.5 \
    --tasks node_count edge_count node_degree connected_nodes edge_existence cycle_check reachability \
    --count 30 \
    --out prompts_hard.jsonl
```

`--count` is the total pool size shared across all 7 algorithms (not
per-task) — the number of generations is `count * len(tasks) *
len(conditions) * len(styles)`.

Dry-run/validate before spending any real generation time (no GPU, no
network):

```bash
# eyeball rendered primers and length stats for xlarge + dense graphs
python scripts/show_primers.py --corpus 100 --xlarge
python scripts/show_primers.py --generated 5 --er-min-sparsity 0.8 --er-max-sparsity 1.0

# eyeball reachability wording/gold across all 7 graph algorithms
python -c "
import random
from graphtalk import diverse_corpus, graphqa
pool = diverse_corpus.build_pool(7)
rng = random.Random(0)
for alg, graph in pool:
  row = diverse_corpus.make_row(graph, 'reachability', rng)
  print(alg, row['task_description'].strip(), '->', row['gold'])
"

# approximate overflow warning for a specific model (chars/4 heuristic --
# a no-op today since no ModelSpec has max_context_tokens filled in yet)
python scripts/build_prompts.py --graph-source diverse --xlarge \
    --model qwen3-14b-think --count 30 --out prompts_hard.jsonl
```

## Testing

```bash
uv run --no-sync pytest -q tests/test_diverse_corpus.py tests/test_scoring.py tests/test_build_prompts.py
uv run --no-sync pytest -q talk_like_a_graph/graph_generators_test.py
uv run --no-sync pytest -q
```

The full suite reports 607 passed locally (up from the pinned 606, plus
this work's new tests), aside from two pre-existing, unrelated failures on
a laptop with no `torch`/no `g++` toolchain installed
(`tests/test_prompts.py::test_run_sweep_row_carries_node_naming` and
`tests/test_hierarchical_model.py::test_fit_end_to_end_recovers_a_strong_effect_direction`
— confirmed identical via `git stash` against the pre-change tree, so they
are environment gaps, not regressions from this work). On the cluster,
where `torch` and a real compiler are present, expect the full clean count.

**Superseded after the merge into `small-model-suite-and-primer-power`
(2026-09-07).** The 606/607 figures above were measured on `main` alone, on a
laptop where `statsmodels` is installed. The merged branch reported **593
passed** at the merge and **603** now (the density-sweep scorer added ten)
on the cluster conda env with `tests/test_hierarchical_model.py` and
`tests/test_mixed_models.py` `--ignore`d, because neither `conda_envs/graphtalk`
nor `conda_envs/graphtalk-cu126` has `statsmodels` or `pymc`. The numbers are
not comparable and neither is wrong; see `CLAUDE.md` for the count that applies
to a run on this cluster.

## Running on the SLURM cluster

Nothing about `cluster/sweep.sbatch`, `scripts/run_sweep.py`, or stage 2/3
changed — a harder prompt set is just a different **stage 1** output file,
fed into the existing pipeline the same way a GoT-named or re-worded prompt
file already is (see `cluster/README.md`'s "Regenerating part of a sweep").

**Stage 1 — login node** (network access, no GPU, no torch needed):

```bash
PYTHONPATH=. .venv/bin/python scripts/build_prompts.py \
    --graph-source diverse \
    --xlarge --er-min-sparsity 0.5 \
    --tasks node_count edge_count node_degree connected_nodes edge_existence cycle_check reachability \
    --count 30 \
    --out prompts_hard.jsonl
```

**Stage 2 — compute node, per model, unchanged script.** Point
`run_sweep.py` at the new prompt file and tag the output so it lands in its
own `runs/<model>.hard.jsonl` rather than colliding with the tracked
`runs/<model>.jsonl` sweep — exactly the `GRAPHTALK_PROMPTS`/
`GRAPHTALK_RUN_TAG` mechanism `cluster/README.md`'s "Regenerating part of a
sweep" section already documents (never tag anything containing `redo` —
`sweep.sbatch` refuses it):

```bash
GRAPHTALK_PROMPTS=prompts_hard.jsonl GRAPHTALK_RUN_TAG=hard \
  sbatch --exclude=n-801,n-802,n-803,n-804 --mem=48G --time=24:00:00 \
  cluster/sweep.sbatch qwen3-14b-think
#   -> runs/qwen3-14b-think.hard.jsonl
```

Smoke-test first, same caveat as any other run — a small `--limit` sees
only whichever task sits first in the file, since the prompt file is
ordered by task:

```bash
GRAPHTALK_PROMPTS=prompts_hard.jsonl GRAPHTALK_RUN_TAG=hard \
  sbatch --time=00:40:00 --exclude=n-801 cluster/sweep.sbatch qwen3-14b-think 20
```

Submit a chain, not a single job, exactly as `cluster/README.md`'s
"Runtime: submit a chain, not a job" describes — xlarge/dense prompts are
*longer* than the tracked sweep's, so budget generously rather than reusing
that section's per-model hour estimates verbatim:

```bash
MODEL=qwen3-14b-think; MEM=48G
PREV=""
for LINK in 1 2 3 4; do
  if [ -z "$PREV" ]; then
    PREV=$(GRAPHTALK_PROMPTS=prompts_hard.jsonl GRAPHTALK_RUN_TAG=hard \
      sbatch --parsable --exclude=n-801,n-802,n-803,n-804 --mem=$MEM \
      cluster/sweep.sbatch $MODEL)
  else
    PREV=$(GRAPHTALK_PROMPTS=prompts_hard.jsonl GRAPHTALK_RUN_TAG=hard \
      sbatch --parsable --exclude=n-801,n-802,n-803,n-804 --mem=$MEM \
      --dependency=afterany:$PREV cluster/sweep.sbatch $MODEL)
  fi
  echo "link $LINK: $PREV"
done
```

**Before relying on the overflow guard on the cluster**: `hf_backend.py`'s
check is a no-op until each `ModelSpec.max_context_tokens` in
`graphtalk/models.py` is filled in with that checkpoint's real published
context length — do that first if xlarge/dense prompts are expected to
approach any model's context window, otherwise an oversized prompt will
reach `model.generate` exactly as it would have before this change.

**Stage 3 — login node, unchanged:**

```bash
python scripts/score_sweep.py --responses runs/*.jsonl --shortcuts shortcuts.json
```

`score_sweep.py`/`graphtalk.analysis` pool by each row's `model` field, so
`runs/qwen3-14b-think.hard.jsonl` joins normally; nothing about scoring
`reachability` rows needs new code (`_BOOLEAN_TASKS` covers it), but a
`reachability` cell has no shortcut-solver ceiling yet — `graphtalk/shortcuts.py`
keeps its own, separate task list and was not extended here, so
`shortcut_table.py` simply won't report a bar for it.
