# Running the qwen3-4b arms of the n=40 density sweep

Gal is running `qwen3-1.7b` and `qwen3-1.7b-think` on this sweep; this is the
recipe for the two remaining arms, `qwen3-4b` and `qwen3-4b-think`, against the
**same prompt file** so all four models are directly comparable.

## The design

`n=40` fixed, ER density pinned to `{0.10, 0.20, 0.35, 0.50}`, all 7 primer
conditions (`none`, `components`, `degree`, `clustering`, `rwse`, `filler`,
`all`), all 6 tasks, 100 graphs per (density, task) cell -- 16,800 prompts.
Already built at `/home/dcor/galbarak2/GraphTalk/prompts.densfull40.jsonl`; you
do not need to rebuild it (it's tracked in git, or read it in place on the
cluster).

**Heads up on validity before you look at results:** `docs/primer-effects-and-power.md`
documents that at this size/density, `node_count` is contaminated by nearly
every primer (any per-node sentence lets you count sentences), `cycle_check`'s
gold is "yes" for almost every graph past the sparsest level, and the `degree`
primer states the `node_degree` answer verbatim (bar 1.00) -- read effects
against `shortcuts.json`'s bar, not against zero, especially for those cells.
`node_degree` with `{none, components, clustering, filler}` is the one cell
this project has already validated as clean.

## Submitting

Both models need `--max-new-tokens` raised from the 2048 default to 8192, or
`edge_count` truncates at these densities (390 edges at p=0.50 needs ~2,700
output tokens; some rows need more). `cluster/sweep.sbatch` now takes this via
`GRAPHTALK_MAX_NEW_TOKENS` (added for this sweep). Use `GRAPHTALK_ENV=graphtalk-cu126`
so the job isn't restricted to the cu130-only half of the partition (see
`cluster/README.md`'s driver table) -- that roughly doubles how many nodes can
pick up a shard.

Use a **run tag** so your output files don't collide with Gal's (`densfull40`,
already in use) or with the tracked sweep's own files:

```bash
cd /home/dcor/galbarak2/GraphTalk

# non-think -- fast, ~1 link should suffice (qwen3-1.7b's non-think chain
# finished its 25-way array comfortably inside one 24h link at this scale)
sbatch --array=0-24 --exclude=n-801 --mem=24G --time=24:00:00 \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=prompts.densfull40.jsonl,GRAPHTALK_RUN_TAG=densfull40-inbal,GRAPHTALK_MAX_NEW_TOKENS=8192 \
  --job-name=q4b-densfull cluster/sweep.sbatch qwen3-4b

# thinking -- much slower (the same design's qwen3-1.7b-think chain needed
# multiple 24h links), so chain it. Submit link 1, then once it's running
# add more links depending on the previous one (afterany):
sbatch --array=0-24 --exclude=n-801 --mem=24G --time=24:00:00 \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=prompts.densfull40.jsonl,GRAPHTALK_RUN_TAG=densfull40-inbal \
  --job-name=q4bT-densfull cluster/sweep.sbatch qwen3-4b-think

# capture the job id above as $PREV, then for each additional link:
sbatch --parsable --dependency=afterany:$PREV --array=0-24 --exclude=n-801 --mem=24G --time=24:00:00 \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=prompts.densfull40.jsonl,GRAPHTALK_RUN_TAG=densfull40-inbal \
  --job-name=q4bT-densfull cluster/sweep.sbatch qwen3-4b-think
```

Don't override `--max-new-tokens` on the `-think` arm -- it already defaults to
8192 (`models.THINK_MAX_NEW_TOKENS`), same value.

Output lands at `runs/qwen3-4b.densfull40-inbal.shard<i>of25.jsonl` and
`runs/qwen3-4b-think.densfull40-inbal.shard<i>of25.jsonl`.
`scripts/score_sweep.py`/`score_density_sweep.py` pool by each row's `model`
field, not by filename, so the `-inbal` tag rejoins the arm automatically --
nothing to reassemble.

## Two things that will bite you if skipped

- **The array width (25) must stay coprime with 42** (6 tasks x 7 conditions,
  the cycle length in the prompt file) -- otherwise some shards get a skewed
  subset of task/condition combinations instead of a proportional mix. If you
  need a different width, pick one not divisible by 2, 3, or 7 (e.g. 11, 13,
  25, 29).
- **There's a 100-job submit cap per user** (QOS `general`, `MaxSubmitPU=100`
  on this account). A 25-wide array plus a couple of chained links adds up
  fast -- check `squeue --me -r -h | wc -l` before adding another link, and if
  you hit `QOSMaxSubmitJobPerUserLimit`, wait for earlier shards to finish (or
  cancel a pending link) before resubmitting.

See `cluster/README.md` for the rest of the standing gotchas (n-801 is slow,
the page-cache warm-up, preemption/resume behaviour) -- all of it applies
unchanged here.
