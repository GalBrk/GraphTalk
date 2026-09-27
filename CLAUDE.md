# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

## What this project is

A course project (Machine Learning with Graphs) that builds on
[Talk like a Graph: Encoding Graphs for Large Language Models](https://arxiv.org/abs/2310.04560)
(arXiv:2310.04560). It tests whether prepending a *primer* — a short preamble of
factual sentences about each node's local structure — before the standard graph
text encoding and question improves an LLM's accuracy on GraphQA tasks, and
separately measures how much of that accuracy a primer-only (no-graph) solver can
already reach without seeing the graph at all.

## Where things are

Every folder has a README; start at [README.md](README.md)'s repository map.

| Question | Read |
|---|---|
| What was found, and the status of each doc | [docs/README.md](docs/README.md) |
| Which script makes which output; the reproduce commands | [scripts/README.md](scripts/README.md) |
| File schemas and the pairing key | [data/README.md](data/README.md) |
| Building the paper and where each of its numbers comes from | [paper/README.md](paper/README.md) |
| Generating responses on the GPU cluster | [cluster/README.md](cluster/README.md) |
| The pilot and screens that chose the settings | [preliminary/README.md](preliminary/README.md) |

## Setup

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"      # add ,gpu for scripts/run_sweep.py
```

**On the lab machines there is no `.venv`.** Docstrings spell commands as
`PYTHONPATH=. .venv/bin/python ...`, which is right for a fresh clone. There, use
the conda env instead (`cluster/README.md` documents how it was built):

```bash
/home/dcor/galbarak2/conda_envs/graphtalk/bin/python
```

## Commands

```bash
uv run --no-sync pytest -q                                   # the whole suite, preliminary/ included
uv run --no-sync pytest -q tests/test_primers.py
uv run --no-sync pytest -q tests/test_primers.py::test_round_trip
```

Always use `--no-sync`; a plain `uv run` re-syncs the environment. The
commands that reproduce every output are in
[scripts/README.md](scripts/README.md#reproduce-everything). After a
regeneration, `git diff outputs/` should show no change except lines that print
a path.

## Rules for results and docs

- **Current claims only.** State what the current output shows; never correct,
  hedge against or mention an earlier version. Replaced drafts live only in git
  tag `pre-cleanup`.
- **Every number is tagged.** A number in a doc or the paper is copied as printed
  from a script's output and followed by its tag (`+7.5 [main]`);
  `tests/test_results_docs.py` checks the results docs. The conventions R1–R4
  (truncation is its own outcome, the metrics) are in
  [docs/README.md](docs/README.md#conventions).
- **Never cite `preliminary/`** in a results doc or the paper.
- **Stage commits by explicit file path.** Others may write into this working
  tree during a session.

## Package layout

- `talk_like_a_graph/` — a **vendored, mostly-unmodified copy** of Google
  Research's reference implementation (graph generators, text encoders). See
  `talk_like_a_graph/UPSTREAM.md` for the commit and the local modifications.
  Treat it as third-party code; prefer changing `graphtalk/`.
- `graphtalk/` — this project's package:
  - `primers.py` — primer statistics (degree, clustering, RWSE, connected
    components) and `render_primer` / `build_primer`, the **single renderer**
    every condition goes through. Every rendered float goes through `_fmt`
    (round-to-6-then-format-to-2, to sidestep BLAS-order tie instability).
    Graphs must arrive already canonicalized.
  - `shortcuts.py` — the primer-only solvers, deliberately **sharing no code with
    `primers.py`**: a strict parser (`parse_primer`) that re-derives the
    renderer's join rules, 16 exact "theorem" rules, 1 heuristic, 8 fitted rules
    (fit and scored on disjoint graph sets via `Split`), and an exact
    enumeration bound for small graphs. Parsing is strict: an unrecognized or
    conflicting sentence raises.
  - `prompts.py` — one prompt is `primer + "\n\n" + encoding + task_description`,
    with the `incident` encoding.
  - `scoring.py` — answer extraction from free model text and the metrics (exact
    match, set-F1, MAE, majority baseline, exact McNemar). `outcomes.py` applies
    rule R1: a truncated response is its own outcome.
  - `graphqa.py` — parses a graph out of a GraphQA row, recomputes gold answers;
    `canonical()` fixes node/edge order so re-encoding is reproducible.
    `diverse_corpus.py` generates the synthetic graphs.
  - `models.py` — model configs only, free of `torch`/`transformers`;
    `hf_backend.py` is the only module that imports them (used by
    `scripts/run_sweep.py`).
  - `node_naming.py` — Game-of-Thrones node names as a text pass over the integer
    prompt, desubstituted before scoring.
  - `significance.py`, `analysis.py`, `ladder.py`, `rewiring.py`, `cell_screen.py`
    serve the `preliminary/` work.

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

- Theorem rule precision is asserted at exactly 1.0 over both an Erdős–Rényi
  corpus and an adversarial corpus (trees, forests, cycles, complete bipartite
  graphs) — the ER generator alone never produces a tree, so a rule that's
  secretly keyed on the `m = n-1` boundary can pass on ER data alone.
- Network access (`graphqa.fetch_rows`) is only exercised by scripts, not by the
  test suite — tests use the vendored generator for graphs.
- `tests/test_results_docs.py` reads the committed outputs; a doc edit that
  changes a cited number must change the output first.
