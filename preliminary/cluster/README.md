# Preliminary cluster drivers

How the Game-of-Thrones and ladder/rewiring runs of the preliminary work were
submitted. [← preliminary/](../README.md) · general cluster setup:
[cluster/README.md](../../cluster/README.md)

Both drivers call `cluster/sweep.sbatch`, whose defaults are the main
experiment's: the 40-node prompts and `data/runs/`. So every job either driver
submits carries the preliminary paths explicitly, `GRAPHTALK_PROMPTS` (a file
under `preliminary/data/prompts/`) and `GRAPHTALK_RUNS_DIR=preliminary/data/runs`.
Run both from the repo root; `--dry-run` prints those variables with the
`sbatch` line.

## Running the GoT node-naming scheme

`sweep.sbatch`'s `GRAPHTALK_PROMPTS`/`GRAPHTALK_RUN_TAG` overrides are also how
a Game-of-Thrones-named arm is run (`graphtalk/node_naming.py` says what the
scheme is). `preliminary/cluster/submit_sweep.sh`
wraps that into one flag, and does stage 1 for you first if it hasn't run yet:

```bash
preliminary/cluster/submit_sweep.sh --node-naming got --exclude=n-801 --mem=32G \
    cluster/sweep.sbatch gemma4-12b
```

Every other `sbatch` flag or positional (`--array`, `--exclude`, the model
key, the smoke-test limit) passes straight through in whatever position it's
given -- only `--node-naming`, `--count`, and `--dry-run` are consumed by the
wrapper. `--count N` (GoT scheme only) requests a prompt file larger than the
tracked sweep's 30-per-task default -- e.g. for a targeted follow-up sized by
`preliminary/scripts/recommend_count.py` (see the "Track 2" section of
`git show pre-cleanup:analysis/README.md`) --
tagged into both the prompt filename and `GRAPHTALK_RUN_TAG` so it can't
collide with the tracked `--count 30` sweep's own files:

```bash
preliminary/cluster/submit_sweep.sh --node-naming got --count 500 \
    cluster/sweep.sbatch qwen3-8b
```

`--dry-run` prints what would run (and whether `preliminary/data/prompts/prompts_got.jsonl` would be
built) without touching anything, which is worth doing once before the real
submission since the wrapper still can't be tested on a scheduler you don't
have.

That's exactly the two-step recipe from `graphtalk/node_naming.py` collapsed
into one call: **stage 1 still runs on the login node**, not inside the
job -- `build_prompts.py` fetches over plain `urllib`, and compute nodes have
no outbound network (`HF_HUB_OFFLINE=1`, same reason as everywhere else in
this file). The wrapper builds `preliminary/data/prompts/prompts_got.jsonl` right there, before
`sbatch` is ever called, and reuses it on every later invocation rather than
rebuilding (`load_rows()`'s cache makes that safe -- see `graphtalk/node_naming.py`).
Omit `--node-naming` (or pass `--node-naming integer`) for the integer scheme:
the wrapper then submits `preliminary/data/prompts/prompts.jsonl`, the pilot's
file, which serves both the plain and the `-think` arms (`prompts_zero_shot.jsonl`
is byte-identical to it). Nothing is built for it.

## Running the ladder/rewiring sweep

The graph-structure ladder and the degree-preserving rewiring experiment
(`preliminary/docs/ladder-and-rewiring.md`) have their own driver, `preliminary/cluster/run_ladder.sh`,
rather than going through `sweep.sbatch` by hand:

```bash
preliminary/cluster/run_ladder.sh              # submits the probe + ladder-screen stages, both arms
preliminary/cluster/run_ladder.sh --dry-run    # print what would submit; build and submit nothing
STAGES=ladder MODELS=qwen3-1.7b preliminary/cluster/run_ladder.sh   # one stage, one model
```

It builds `preliminary/data/prompts/prompts.retrieval_locate.jsonl` / `prompts.ladder_screen.jsonl` if
they don't already exist, sizes each model's GPU tier and `--mem` itself (see
`tier_for()` in the script), and submits one job per model per stage --
writing to `<model>.retrieval_locate.jsonl` / `<model>.ladder_screen.jsonl`
(see [../data/README.md](../data/README.md)). The rewiring stage is deliberately **not** included in
the default run (`STAGES` defaults to `probe ladder`); it needs a rewire
prompt file built separately with `preliminary/scripts/build_ladder.py --stage rewire`,
restricted to the rungs that cleared both screens for the models being run --
read `preliminary/docs/ladder-and-rewiring.md` before spending GPU time on it.
