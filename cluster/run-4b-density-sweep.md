# Running the qwen3-4b arms of the 40-node sweep

The commands that produce the committed files of the two Qwen3-4B arms,
`qwen3-4b` and `qwen3-4b-think`, under the names `scripts/build_raw_frame.py`
reads. The Qwen3-1.7B arms run the same way with the model key changed.
[← cluster/README.md](README.md)

## The design

`n=40` fixed, ER density pinned to `{0.10, 0.20, 0.35, 0.50}`, all 7 primer
conditions (`none`, `components`, `degree`, `clustering`, `rwse`, `filler`,
`all`), all 6 tasks, 100 graphs per (density, task) cell -- 16,800 prompts in
`data/prompts/prompts.densfull40.jsonl`. The high-density extension,
`data/prompts/prompts.densfull40hi.jsonl`, is `{0.65, 0.75, 0.85}` with
`node_degree` and `edge_existence` only, the same 7 conditions and 100 graphs per
cell -- 4,200 prompts. Both files are tracked, so there is nothing to rebuild
([scripts/README.md](../scripts/README.md) has the commands that built them).

Which cells a primer's effect can be read on, against the graph-blind solver's
bar in `data/shortcuts_n40_flat.json`: [docs/results/n40-sweep.md](../docs/results/n40-sweep.md).

## Submitting

Run from the root of a clone, after `mkdir -p out` (Slurm writes the job logs
there and does not create the directory). Output lands in that clone's
`data/runs/`.

Every command uses `GRAPHTALK_ENV=graphtalk-cu126` with `--exclude=n-801`. The
cu126 build runs on both driver generations, and an `--exclude` on the command
line replaces `sweep.sbatch`'s default list, so the 535.x nodes join the pool
while the slow n-801 stays out ([cluster/README.md](README.md)'s driver table).
That roughly doubles how many nodes can pick up a shard.

Budgets ([cluster/README.md](README.md#token-budgets)): 8192 for both arms on the
main sweep, because the plain arm's `edge_count` truncates at 2048 from
p = 0.35 (390 edges at p = 0.50 take ~2,700 output tokens); on the extension,
2048 for the plain arm and 8192 for the thinking arm.

```bash
cd <your clone>
mkdir -p out
COMMON="--exclude=n-801 --mem=24G --time=24:00:00"

# main sweep, plain -- one 24 h link is enough (a 25-way shard of the plain
# qwen3-4b arm is ~0.2M new tokens)
sbatch --array=0-24 $COMMON \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=data/prompts/prompts.densfull40.jsonl,GRAPHTALK_RUN_TAG=densfull40,GRAPHTALK_MAX_NEW_TOKENS=8192 \
  --job-name=q4b-densfull cluster/sweep.sbatch qwen3-4b

# main sweep, thinking -- ~1.5M new tokens per shard, so chain it: submit
# link 1, then add links that depend on the previous one (afterany)
PREV=$(sbatch --parsable --array=0-24 $COMMON \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=data/prompts/prompts.densfull40.jsonl,GRAPHTALK_RUN_TAG=densfull40,GRAPHTALK_MAX_NEW_TOKENS=8192 \
  --job-name=q4bT-densfull cluster/sweep.sbatch qwen3-4b-think)
PREV=$(sbatch --parsable --dependency=afterany:$PREV --array=0-24 $COMMON \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=data/prompts/prompts.densfull40.jsonl,GRAPHTALK_RUN_TAG=densfull40,GRAPHTALK_MAX_NEW_TOKENS=8192 \
  --job-name=q4bT-densfull cluster/sweep.sbatch qwen3-4b-think)
# ... repeat the last command for each further link

# high-density extension: 11 shards at 2048 for the plain arm, 25 at 8192 for
# the thinking arm
sbatch --array=0-10 $COMMON \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=data/prompts/prompts.densfull40hi.jsonl,GRAPHTALK_RUN_TAG=densfull40hi,GRAPHTALK_MAX_NEW_TOKENS=2048 \
  --job-name=q4b-densfullhi cluster/sweep.sbatch qwen3-4b
sbatch --array=0-24 $COMMON \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=data/prompts/prompts.densfull40hi.jsonl,GRAPHTALK_RUN_TAG=densfull40hi,GRAPHTALK_MAX_NEW_TOKENS=8192 \
  --job-name=q4bT-densfullhi cluster/sweep.sbatch qwen3-4b-think
```

Output lands at:

| Command | Files |
|---|---|
| main sweep | `data/runs/qwen3-4b.densfull40.shard<i>of25.jsonl`, `data/runs/qwen3-4b-think.densfull40.shard<i>of25.jsonl` |
| extension | `data/runs/qwen3-4b.densfull40hi.shard<i>of11.jsonl`, `data/runs/qwen3-4b-think.densfull40hi.shard<i>of25.jsonl` |

Each shard holds every n-th prompt of its file, as the committed shards do.
`build_raw_frame.py` selects runs by these file names
(`data/runs/<arm>.densfull40.shard*.jsonl` and `<arm>.densfull40hi.shard*.jsonl`),
not by the rows' `model` field, so a different run tag gives files the frame
does not read.

## Two limits that bite if ignored

- **The array width must stay coprime with the prompt file's cycle**: 42 rows
  (6 tasks x 7 conditions) in `densfull40`, 14 (2 x 7) in `densfull40hi`.
  Otherwise some shards get a skewed subset of task/condition combinations
  instead of a proportional mix. For another width, pick one not divisible by
  2, 3 or 7 (e.g. 11, 13, 25, 29).
- **There's a 100-job submit cap per user** (QOS `general`, `MaxSubmitPU=100`
  on this account), and every array task counts. A 25-wide array plus a couple
  of chained links adds up fast -- check `squeue --me -r -h | wc -l` before
  adding another link, and if you hit `QOSMaxSubmitJobPerUserLimit`, wait for
  earlier shards to finish (or cancel a pending link) before resubmitting.

See [cluster/README.md](README.md) for the rest of the standing gotchas (n-801
is slow, the page-cache warm-up, preemption and resume) -- all of it applies
here unchanged.
