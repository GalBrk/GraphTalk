# Running the GoT-naming arms of the n=40 density sweep

The integer-named n=40 density sweep (`cluster/run-4b-density-sweep.md`,
`docs/full-task-density-sweep.md`) asks whether primers help at this size and
density. This is the same question's node-naming axis (Stage 4b,
`scripts/naming_effect.py`): does renaming nodes from integers to
Game-of-Thrones characters change what `qwen3-1.7b`/`qwen3-4b` get right on
the *same* graphs? `graphtalk/node_naming.py`'s `GOT_NAMES` was extended from
20 to 40 entries to cover this corpus's node count -- see its module comment
for the added names and the collision screening applied to them.

## The design

Identical graphs, gold answers and conditions to `prompts.densfull40.jsonl`
(n=40, ER density pinned to `{0.10, 0.20, 0.35, 0.50}`, all 7 primer
conditions, all 6 tasks, 100 graphs per (density, task) cell, default seed) --
verified by cross-checking every `(instance_id, task, condition)` key's
`gold`/`edges`/`nodes` fields against the integer file. Only the node
references in the primer, encoding and question differ. Built with:

```bash
PYTHONPATH=. python scripts/build_size_sweep.py \
    --sizes 40 --densities 0.10 0.20 0.35 0.50 --count 100 \
    --conditions all clustering components degree filler none rwse \
    --tasks node_count edge_count node_degree connected_nodes edge_existence cycle_check \
    --node-naming got \
    --out prompts.densfull40.got.jsonl
```

**Not tracked in git** -- at n=40, GoT character names are longer strings than
the integers they replace, so this file is 125 MB, over GitHub's 100 MB push
limit (the trackable integer file is 86 MB). Run the command above once on
the login node before submitting anything below; it's deterministic, so
everyone building it gets the identical 16,800-row file.

## Submitting

Same recipe as `run-4b-density-sweep.md`, pointed at the GoT prompt file and
tagged `densfull40.got` (the `.got` segment is what lets both
`scripts/naming_effect.py`'s `arm_paths` and `scripts/score_sweep.py`'s
`desubstitute_named_responses` recognize these rows as GoT-named). Run all
four arms -- `qwen3-1.7b`, `qwen3-1.7b-think`, `qwen3-4b`, `qwen3-4b-think` --
against this same file so every arm pairs against its own integer counterpart
from the other sweep.

```bash
cd /home/dcor/galbarak2/GraphTalk

# qwen3-1.7b, non-think
sbatch --array=0-24 --exclude=n-801 --mem=24G --time=24:00:00 \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=prompts.densfull40.got.jsonl,GRAPHTALK_RUN_TAG=densfull40.got,GRAPHTALK_MAX_NEW_TOKENS=8192 \
  --job-name=q17b-densgot cluster/sweep.sbatch qwen3-1.7b

# qwen3-1.7b, thinking (chain -- don't override --max-new-tokens, it already
# defaults to 8192 via models.THINK_MAX_NEW_TOKENS)
sbatch --array=0-24 --exclude=n-801 --mem=24G --time=24:00:00 \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=prompts.densfull40.got.jsonl,GRAPHTALK_RUN_TAG=densfull40.got \
  --job-name=q17bT-densgot cluster/sweep.sbatch qwen3-1.7b-think

# qwen3-4b, non-think
sbatch --array=0-24 --exclude=n-801 --mem=24G --time=24:00:00 \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=prompts.densfull40.got.jsonl,GRAPHTALK_RUN_TAG=densfull40.got,GRAPHTALK_MAX_NEW_TOKENS=8192 \
  --job-name=q4b-densgot cluster/sweep.sbatch qwen3-4b

# qwen3-4b, thinking
sbatch --array=0-24 --exclude=n-801 --mem=24G --time=24:00:00 \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=prompts.densfull40.got.jsonl,GRAPHTALK_RUN_TAG=densfull40.got \
  --job-name=q4bT-densgot cluster/sweep.sbatch qwen3-4b-think

# for each thinking arm, capture the job id as $PREV and chain more links as needed:
sbatch --parsable --dependency=afterany:$PREV --array=0-24 --exclude=n-801 --mem=24G --time=24:00:00 \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=prompts.densfull40.got.jsonl,GRAPHTALK_RUN_TAG=densfull40.got \
  --job-name=<same-job-name> cluster/sweep.sbatch <model>
```

Output lands at `runs/<model>.densfull40.got.shard<i>of25.jsonl`.

## Analyzing

`qwen3-1.7b` and `qwen3-4b` (unlike the tracked ARMS models in
`scripts/naming_effect.py`) also carry runs from several unrelated
experiments under `runs/` -- `probe100`, `retrieval`, `ec500`, `size`,
`degdens40`, etc. -- so pooling by model key alone would mix corpora.
`naming_effect.py` gained `--tag` and `--models` for exactly this:

```bash
PYTHONPATH=. python scripts/naming_effect.py \
    --tag densfull40 \
    --models qwen3-1.7b qwen3-1.7b-think qwen3-4b qwen3-4b-think
```

`--tag densfull40` matches both `...densfull40.shard*` (integer) and
`...densfull40.got.shard*` (GoT) files for each model, and excludes every
other tag. An arm only reports once its GoT row count matches its integer
row count (16,800) -- otherwise it's listed under "skipped: incomplete".

## Two things that will bite you if skipped

Same as `run-4b-density-sweep.md`: the array width (25) must stay coprime
with 42 (6 tasks x 7 conditions), and there's a 100-job submit cap per user
(`squeue --me -r -h | wc -l` before adding another chained link). See
`cluster/README.md` for the rest of the standing cluster gotchas.
