# Generating responses on the TAU CS cluster

How every response in [`data/runs/`](../data/README.md) is generated, and how to
check that generation still reproduces them.
[← back to the repo](../README.md)

Partition `killable`, account `gpu-research`. The lab's conda envs and model
cache live under `/home/dcor/galbarak2/` (DCOR lab) and are readable in place:
[collaborator-access.md](collaborator-access.md) says how to use them without
building your own.

## Only generation needs a GPU

| Stage | Script | Where | Needs |
|---|---|---|---|
| 1. Build prompts | `scripts/build_size_sweep.py` | anywhere | no network, no torch |
| 2. Generate | `scripts/run_sweep.py`, via `cluster/sweep.sbatch` | compute node | GPU, torch, transformers |
| 3. Analyse | `scripts/build_raw_frame.py` and the rest ([scripts/README.md](../scripts/README.md)) | anywhere | no GPU |

The prompt set is a file you can read and diff before spending GPU time, every
model is handed the identical file, and the analysis can be rerun and changed
without regenerating anything.

## One-time setup

This builds the lab's envs and fills its model cache, as the `galbarak2`
account. Home is a **6 GB** quota with a 102k file cap, so everything goes on
the lab netapp. Anaconda is installed at `/home/dcor/galbarak2/anaconda3`.

```bash
source /home/dcor/galbarak2/anaconda3/etc/profile.d/conda.sh
conda create -y -p /home/dcor/galbarak2/conda_envs/graphtalk python=3.11
```

Use `-p` and the full path, not `-n graphtalk`. `envs_dirs` does not include
`conda_envs/`; it resolves to `anaconda3/envs`, so `-n` would silently put the
env somewhere `cluster/sweep.sbatch` does not look.

```bash
export PIP_CACHE_DIR=/home/dcor/galbarak2/pip_cache
export TMPDIR=/home/dcor/galbarak2/tmp
/home/dcor/galbarak2/conda_envs/graphtalk/bin/pip install -e ".[dev,gpu]"
/home/dcor/galbarak2/conda_envs/graphtalk/bin/python -m pytest -q
```

Redirect the pip cache before installing. The CUDA wheels are several GB and the
default `~/.cache/pip` would eat most of the 6 GB home quota.

`graphtalk` holds torch 2.13.0 built for CUDA 13.0; `graphtalk-cu126`, next to
it in `conda_envs/`, is the same env with torch 2.13.0 built for CUDA 12.6
(which of the two to use: [the driver section](#half-the-partition-has-a-driver-the-default-env-cannot-use)).
Both run transformers 5.15.0.

Then pre-download the models **on the login node**, because compute nodes run
with `HF_HUB_OFFLINE=1`:

```bash
export HF_HOME=/home/dcor/galbarak2/hf_cache
python -c "
from huggingface_hub import snapshot_download
for repo in ('Qwen/Qwen3-1.7B', 'Qwen/Qwen3-4B', 'Qwen/Qwen3-8B'):
    snapshot_download(repo)
"
```

Run it inside `tmux`, since a dropped SSH connection kills it.

## Running

Submit from the root of a clone. Slurm writes each job's log to `out/`, which
is gitignored and which Slurm does not create, so once per clone:

```bash
mkdir -p out
```

Without it the job fails before it starts and leaves no log. Output goes to the
clone's `data/runs/`, named `<model>.<run set>.shard<i>of<n>.jsonl` from the
model key, `GRAPHTALK_RUN_TAG` and the `--array` width. The analyses select
runs by that name ([Regenerating part of a run](#regenerating-part-of-a-run)).

[run-4b-density-sweep.md](run-4b-density-sweep.md) has the commands that
produce the Qwen3-4B arms' committed files. In short, for the main sweep:

```bash
sbatch --array=0-24 --exclude=n-801 --mem=24G --time=24:00:00 \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=data/prompts/prompts.densfull40.jsonl,GRAPHTALK_RUN_TAG=densfull40,GRAPHTALK_MAX_NEW_TOKENS=8192 \
  cluster/sweep.sbatch qwen3-4b
#   -> data/runs/qwen3-4b.densfull40.shard<i>of25.jsonl, i = 0..24
```

`--exclude=n-801` goes with `graphtalk-cu126` only: it replaces the script's
default exclude list, and so admits the 535.x-driver nodes, which that env can
use and the default env cannot
([below](#half-the-partition-has-a-driver-the-default-env-cannot-use)).

`sweep.sbatch` reads these variables (pass them with `--export=ALL,...`):

| Variable | Default | What it sets |
|---|---|---|
| `GRAPHTALK_PROMPTS` | `data/prompts/prompts.densfull40.jsonl` | the prompt file |
| `GRAPHTALK_RUN_TAG` | none | the run set in the output name; `redo` is refused |
| `GRAPHTALK_MAX_NEW_TOKENS` | `models.budget` | the token budget ([table](#token-budgets)) |
| `GRAPHTALK_RUNS_DIR` | `<clone>/data/runs` | the output directory |
| `GRAPHTALK_TASK_DIR` | none | a subdirectory of the output directory |
| `GRAPHTALK_ENV` | `graphtalk` | the conda env (`graphtalk-cu126`) |
| `GRAPHTALK_CONDA`, `GRAPHTALK_ENVS_DIR` | the lab's anaconda3 and `conda_envs/` | another conda install |
| `GRAPHTALK_HF_CACHE` | `/home/dcor/galbarak2/hf_cache/hub` (shared, read-only) | where weights are loaded from |
| `GRAPHTALK_HF_HOME` | `<clone>/.hf_home` | a writable `HF_HOME` of your own |
| `GRAPHTALK_BATCH_SIZE` | 1 | batched generation; not for a sweep ([Two levers](#two-levers-if-that-is-too-slow)) |

`HF_HOME` and the hub cache are separate on purpose: `huggingface_hub` writes a
`token` file at `HF_HOME`'s root, so pointing `HF_HOME` at the read-only shared
cache fails every load with `PermissionError`.

### Token budgets

The committed rows' budgets, read off `hit_cap` rows (whose `n_new_tokens` is
the budget):

| Run set | Arms | Shards | `GRAPHTALK_MAX_NEW_TOKENS` |
|---|---|---|---|
| `densfull40` (`data/prompts/prompts.densfull40.jsonl`) | all four | 25 | 8192 |
| `densfull40hi` (`data/prompts/prompts.densfull40hi.jsonl`) | `qwen3-1.7b`, `qwen3-4b` | 11 | 2048 |
| `densfull40hi` | `qwen3-1.7b-think`, `qwen3-4b-think` | 25 | 8192 |
| `density40`, `degfixdeg` | `qwen3-1.7b` | 3, 5 | 2048 |
| `degdensthink`, `degdensfillT` | `qwen3-1.7b-think` | 7 | 8192 |

The registry's defaults (`models.budget`) are the `densfull40` budgets, so only
the other rows need the variable; every command here passes it anyway. The
follow-up run sets `degdens40`, `degdens40hi`, `degdensfill`, `degdensrep` and
`degceil` (`qwen3-1.7b`) and `qwen3-8b.degfixdeg` have no row that reaches
2048 tokens, so any budget from 2048 up regenerates them; `density_followups.py`
rebuilds their prompts.

A plain arm needs 8192 on the main sweep because `edge_count` at 40 nodes
truncates at 2048 from p = 0.35 (390 edges at p = 0.50 take ~2,700 output
tokens). Generation still stops at EOS, so a higher cap only costs anything on
rows that actually run long.

In `preliminary/data/runs/`, `qwen3-1.7b-think`'s `ladder_screen` and
`retrieval_locate` rows ran at 16384, and `qwen3-1.7b`'s `ec500`, `probe100`
and `size` rows and `qwen3-4b`'s `probe100` rows at 2048; the other
preliminary run sets match their model's default.

### Smoke test

A second argument runs that many generations and writes them to
`data/runs/archive/smoke-<model>.jsonl` (under `GRAPHTALK_RUNS_DIR` if set),
which no analysis reads:

```bash
sbatch --time=00:40:00 --mem=24G cluster/sweep.sbatch qwen3-4b 20
```

**A short smoke test is not a representative one.** The main prompt file is
density-major: its first 4,200 rows are all at p = 0.10, cycling through the 6
tasks x 7 conditions every 42 rows, so the first 20 generations are
`node_count`, `edge_count` and `node_degree` on the sparsest graphs. A truncated
generation shows up as unparseable on `cycle_check` but as a confident *wrong
answer* on counting tasks, where the extractor picks an integer out of the
abandoned working. For a spread of tasks, conditions and densities, run the
check below instead.

### Check that generation still reproduces

`scripts/reproduce_rows.py` picks a small prompt subset whose committed
responses finished well inside the budget, and afterwards compares the
regenerated rows with the committed ones. Everything the check writes goes to a
scratch directory outside every checkout's `data/runs/`, under a run tag no
analysis reads. From the root of a clone:

```bash
mkdir -p out
PY=/home/dcor/galbarak2/conda_envs/graphtalk-cu126/bin/python
SCRATCH=/path/outside/any/checkout/gt-repro
PYTHONPATH=. $PY scripts/reproduce_rows.py subset --model qwen3-1.7b \
    --rows 48 --max-tokens 2000 --out $SCRATCH/repro48.jsonl

sbatch --time=06:00:00 --mem=16G \
  --export=ALL,GRAPHTALK_ENV=graphtalk-cu126,GRAPHTALK_PROMPTS=$SCRATCH/repro48.jsonl,GRAPHTALK_RUN_TAG=repro,GRAPHTALK_RUNS_DIR=$SCRATCH/runs,GRAPHTALK_HF_HOME=$SCRATCH/hf_home,GRAPHTALK_MAX_NEW_TOKENS=8192 \
  cluster/sweep.sbatch qwen3-1.7b
#   -> $SCRATCH/runs/qwen3-1.7b.repro.jsonl

# once the job has finished
PYTHONPATH=. $PY scripts/reproduce_rows.py compare --model qwen3-1.7b \
    --regenerated $SCRATCH/runs/qwen3-1.7b.repro.jsonl --subset $SCRATCH/repro48.jsonl
```

The subset is 48 prompts: 8 per task, 12 per density, every condition, 2 in
each (task, density) cell, about 21k committed tokens, so under an hour of
generation at 7 tok/s plus the warm-up. `subset` prints the budget to generate
at (the cap the committed rows reach, 8192 here). `compare` scores both sides
with `graphtalk.scoring` and `graphtalk.outcomes`, as the frame does, and
prints how often the text, the extracted answer and the outcome agree, per
task, then every row that differs with the length of the text the two share
before diverging.

Reading it:

- **Configuration drift** is a `FATAL` line in the job log, `compare` refusing
  a row (wrong gold or model), rows still missing after the job ended, a
  `<think>` block in a plain arm's regenerated text, or differing texts that
  share only a few characters. Divergence from the first tokens means a
  different prompt, chat template, dtype or budget.
- **Otherwise**, generation reproduces when the extracted answer and the
  outcome agree on at least 41 of the 48 rows. Exact text need not match:
  `data/runs/` does not record which card generated each row, and bf16
  arithmetic on another card or CUDA build flips near-tie tokens part way
  through a response (the batching check below shows the same mechanism, its
  mismatches sharing their first 57-942 characters).

### Regenerating part of a run

`run_sweep.py` skips keys the output already has, so a row left in place is
**not** regenerated. The analyses select runs by file name, not by the rows'
`model` field: `build_raw_frame.py` reads `data/runs/<arm>.densfull40.shard*.jsonl`
and `<arm>.densfull40hi.shard*.jsonl` (then the unsharded `<arm>.<run set>.jsonl`),
keeping the first row it sees for each key. So:

1. Strip the rows to regenerate from the `<arm>.<run set>.shard*.jsonl` files.
2. Regenerate them from a subset prompt file under a tag no analysis reads:

   ```bash
   sbatch --time=12:00:00 --export=ALL,GRAPHTALK_PROMPTS=subset.jsonl,GRAPHTALK_RUN_TAG=rerun,GRAPHTALK_MAX_NEW_TOKENS=8192 \
     cluster/sweep.sbatch qwen3-4b
   #   -> data/runs/qwen3-4b.rerun.jsonl
   ```

3. Append the regenerated rows to the shard files they were stripped from, and
   delete the `rerun` file.

A tagged file is read by no analysis until its rows are merged back. Never use a
tag that *starts* with a run set's name (`densfull40-x`): `primer_findings.py`
and the scripts that use its loader (`response_patterns.py`, `primer_flips.py`)
and `check_cycle_claims.py` glob `<arm>.densfull40*.shard*.jsonl` and would read
it, possibly in place of the committed rows, while `build_raw_frame.py` would
not. **Never tag a regeneration `redo`**:
`graphtalk.analysis` drops `.redo.shard` files, and `sweep.sbatch` refuses the
tag for that reason.

The Game-of-Thrones and ladder/rewiring drivers of the preliminary work are in
[preliminary/cluster/](../preliminary/cluster/README.md).

## Warm the page cache, or the job dies loading

`sweep.sbatch` reads the whole checkpoint with `cat` before starting Python.
This is not a nicety. `safetensors` mmaps the file and faults tensor offsets in
checkpoint order rather than file order, and those scattered reads are
pathological over NFS: without the warm-up a 16 GB checkpoint projected a
**nine-hour** load and died on its time limit having written no rows. One
sequential pass first costs about 20 minutes and drops the load to **two
seconds**.

The warm-up cost is paid per job on a cold node, and it dominates short
diagnostic runs — budget for it before submitting anything small.

## Half the partition has a driver the default env cannot use

`killable` spans two driver generations, and the default env's torch is a
**cu130** build that needs **580 or newer**.

Measured across the whole partition on 2026-09-17, one CPU-only `srun` per node
running `nvidia-smi --query-gpu=driver_version` — `nvidia-smi` reports the driver
without a GPU allocated, so this schedules even on a fully allocated node and
costs nothing:

| node | card | VRAM | driver | usable |
|---|---|---|---|---|
| n-301, n-303, n-304, n-306, n-307, n-350 | 3090 | 24 GB | 595.84 | yes |
| n-302, n-305 | 3090 | 24 GB | *unverified* (probe preempted) | presumed |
| n-602 | a6000 | 48 GB | 595.84 | yes |
| n-601 | a6000 | 48 GB | *unverified* (probe preempted) | presumed |
| n-503 | a5000 | 24 GB | 595.84 | yes |
| n-502 | a5000 | 24 GB | 580.173.02 | yes |
| n-801 | l40s | 48 GB | 580.126.09 | driver yes, **see below** |
| n-805 | l40s | 48 GB | 580.173.02 | yes |
| t-806 | l40s | 48 GB | 580.105.08 | yes |
| **n-802, n-803, n-804** | l40s | 48 GB | **535.183.01** (CUDA 12.2) | **no** |
| **n-501** | a5000 | 24 GB | **535.288.01** (CUDA 12.2) | **no** |

**Every 3090 node is on 595.84**, the newest driver in the partition, so the
24 GB half of the constraint needs no env change. **n-801's driver is fine**; it
is excluded for read throughput alone (next section), which is a different
failure with a different symptom, so do not reach for the cu126 env when a job
is slow on it.

**`n-501` is on the bad list and is easy to miss**: it is an a5000 node, so it
is not caught by thinking of the bad nodes as "the l40s ones". It cost three
separate job failures on 2026-09-05 before it was identified.

`sweep.sbatch` therefore carries a default `--exclude` of the four 535.x nodes
plus n-801, so ordinary placement is deterministic without anyone remembering
the table. **An `--exclude` on the command line replaces that list rather than
adding to it** -- `sbatch --exclude=n-801 ...` re-admits n-501, n-802, n-803
and n-804. With the default env the driver guard in `sweep.sbatch` then fails
such a job fast and visibly (~90 s, non-zero exit) rather than letting it fall
back to the CPU.

To use the 535.x nodes on purpose, switch the env rather than editing the
exclude:

```bash
# the cu126 build, which runs on BOTH driver generations
sbatch --export=ALL,GRAPHTALK_ENV=graphtalk-cu126 --exclude=n-801 ... \
    cluster/sweep.sbatch <model>

# or pin to one card type, when an arm is a headline result and should not
# straddle two CUDA builds
sbatch --constraint=a6000 ... cluster/sweep.sbatch <model>
```

Pinning to `a6000` keeps one card type and one CUDA build across an arm;
`graphtalk-cu126` places faster because it can use every node. `a6000` is only
n-601 and n-602 (16 GPUs, shared cluster-wide), so it can queue: on 2026-09-17
every a6000 and l40s GPU in `killable` was allocated while 25 GPUs sat free on
the 3090 and a5000 nodes.

### The constraint spans 24 GB and 48 GB cards

The default `--constraint` is `a6000|l40s|a5000|geforce_rtx_3090|h100`, so it
does not imply a 48 GB card. `qwen3-14b` and `gemma4-12b` (`min_vram_gb=48`) do
not fit a 24 GB one. `sweep.sbatch` reads `min_vram_gb` from the registry and
refuses a card that is too small, before the 20-minute page-cache warm-up rather
than after, so those models fail fast instead of OOMing an hour in. For a
big-model arm, narrow the constraint at submission time anyway and skip the
bounce:

```bash
sbatch --constraint='a6000|l40s' ... cluster/sweep.sbatch qwen3-14b
```

The constraint leaves out `geforce_rtx_2080` and the DGX `v100`/`quadro` nodes:
Turing and Volta have no bf16 tensor cores, so `device_map="auto"` would place
all or part of the model on CPU there rather than erroring.

On a node whose driver the env cannot use, `device_map="auto"` finds no usable
CUDA device and puts the model on the **CPU** — with no error and no warning, at
roughly a fortieth of the speed. Three jobs ran that way for sixteen hours
before it was spotted, and the symptom is indistinguishable from a busy filer or
a contended card, so it costs a long detour to diagnose. The tell is
`nvidia-smi` reporting **0 MiB used on your own assigned device** while the
process holds the weights in host RAM.

`sweep.sbatch` refuses to start on such a node, and excludes them by default so
the scheduler does not waste a link finding out. If you override `--exclude`
for another reason with the default env, carry the whole list, n-501 included:

```bash
sbatch --exclude=n-501,n-801,n-802,n-803,n-804 --mem=32G \
    cluster/sweep.sbatch qwen3-8b
```

Do not check the driver on the login node and assume it generalises — the login
node is on 580 while four compute nodes are not. A smoke test passing proves only
that *that* job's node was fine.

### n-801 is slow

Read throughput varies by node far more than expected. Measured with 2 GiB of
direct I/O, twice each:

| node | throughput |
|---|---|
| n-802, n-805 | ~31 MB/s |
| **n-801** | **12.4 MB/s idle, 3.4 MB/s under load** |

n-801 had a 195-day uptime and both stalls in this project landed on it, which
is why the default `--exclude` carries it and why the cu126 commands pass
`--exclude=n-801`.

### Do not put several checkpoint warm-ups on one node at once

Measured on 2026-08-28, when four plain arms were submitted together and the
scheduler put **three on n-602**, where they each began a sequential read of a
14.9 / 22.3 / 27.5 GB checkpoint at the same time. After **3.5 hours not one
had finished warming**, which puts each stream under **1.2 MB/s** and the node's
aggregate around 3.6 MB/s, the range of n-801 under load. The fourth arm, alone
on n-601, warmed 15.3 GB in 22 minutes (~11.8 MB/s) and finished the whole job
in 63 minutes.

The warm-up is bandwidth-bound and does not parallelise: N concurrent reads on one
node finish in the same total time as N sequential ones, except every job finishes
*late* instead of one finishing early and starting to generate. Stagger them, or
spread them with `--nodelist`, but do not submit several large arms and let the
scheduler pack them.

Two things make this hard to diagnose, both worth knowing before you go looking:

- **The warm-up prints nothing until it finishes.** `find ... -exec cat {} +` is a
  single opaque call, so "no output for three hours" is indistinguishable from a
  hang. It is almost always just slow.
- **`sstat` cannot tell you either.** On this cluster it returns the sentinel
  `213503982+` for CPU fields on a running job, so there is no way to confirm the
  process is alive from the login node. Reason about it from bytes and elapsed
  time instead.

### Size `--time` for a contended warm-up, not a measured-alone one

On the same day the plain arms were submitted with `--time=06:00:00`, sized from
~20 min of warm-up plus an hour of generation. Under the contention above the
warm-up alone was heading past 5 hours, so two of the three would have hit the
wall having written **zero rows**. `--time` is a ceiling, not a reservation --
the job releases the allocation when it exits -- so there is no reason to trim
it. Use 12 h for anything that has to warm a checkpoint it might be sharing
bandwidth for.

Slurm will not let you fix this after the fact: `scontrol update jobid=<j>
TimeLimit=...` upward returns `Access/permission denied` for an ordinary user. The
only remedy is `scancel` and resubmit.

A resubmit is a **full re-read** -- do not expect the node's page cache to help,
even on n-602 with 1 TB of RAM against ~65 GB of checkpoints. `qwen3-14b` had
read ~14 GB of its 27.5 GB checkpoint, was cancelled and resubmitted to the same
node, and then took almost exactly the time that reading all 27.5 GB from
scratch at the contended rate predicts. Whatever the reason -- NFS client
caching, or eviction under the other jobs on the node -- budget a restart as if
nothing were cached, and pick the node on current load rather than on history.

## `--array` sets the shard COUNT from the number of tasks, not the highest index

`sweep.sbatch` derives sharding from Slurm's array variables:

```sh
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
```

`SLURM_ARRAY_TASK_COUNT` is **how many tasks the array has**, not the largest id.
So resubmitting three failed shards of a five-way split with `--array=1,2,4`
gives `NSHARDS=3`, and each task then strides `records[i::3]` instead of
`records[i::5]`. The output is named `shard1of3.jsonl`, generates the **wrong
subset**, and **overlaps rows the surviving `*of5` shards already own** -- so a
later pooled scoring double-counts them. (The `shard4of3` task does fail loudly,
because `run_sweep.py` rejects `--shard 4 --num-shards 3`, but 1 and 2 run
happily and produce plausible-looking wrong data.) On 2026-09-05 this generated
28 rows wrongly strided, 12 of them duplicating rows owned by
`shard0of5`/`shard3of5`.

To resubmit a subset of shards, pass the count explicitly rather than relying on
an array:

```bash
for s in 3 17; do
  sbatch --job-name=q4bT-s${s} --mem=24G --time=24:00:00 \
    --export="ALL,SLURM_ARRAY_TASK_ID=${s},SLURM_ARRAY_TASK_COUNT=25,GRAPHTALK_RUN_TAG=densfull40,GRAPHTALK_MAX_NEW_TOKENS=8192" \
    cluster/sweep.sbatch qwen3-4b-think
done
```

**Keep the width coprime with the prompt file's cycle.** `densfull40` cycles
through its 6 tasks x 7 conditions every 42 rows and `densfull40hi` through its
2 tasks x 7 conditions every 14, and `run_sweep.py` strides the file. A width
sharing a factor with the cycle (2, 3 or 7 for 42) gives every shard a skewed
subset of the (task, condition) pairs: nothing is lost, since all rows are still
produced, but partial progress is unbalanced, so any mid-run comparison is across
different instance sets. The committed shards are 25-way, and 11-way for the
plain arms' `densfull40hi`; 11, 13, 25 and 29 are all safe.

## Memory is per-model, and it decides whether you are scheduled at all

The `--mem` request must hold the checkpoint in the page cache the warm-up fills,
plus the loader's buffers. The weights leave host memory for the GPU, so the
headroom above the checkpoint size is comfortable:

| model | checkpoint | `--mem` |
|---|---|---|
| `qwen3-1.7b` | ~4 GB | 16G |
| `qwen3-4b` | ~8 GB | 24G |
| `gemma4-e4b` | 15 GB | 32G |
| `qwen3-8b` | 16 GB | 32G |
| `gemma4-12b` | 23 GB | 40G |
| `qwen3-14b` | 28 GB | 48G |

Do not round these up "to be safe". The nodes are busy, and the ones with a spare
GPU are often the ones with least RAM free: a uniform 96G request left twelve
jobs sitting on `Reason=Resources` while three GPUs stood idle behind 39 GB and
47 GB of free memory. The sbatch default is 64G, which suits the largest model;
override it downward per model.

## Runtime: submit a chain, not a job

`killable`'s ceiling is 24 h. A plain arm of the main sweep, 16,800 prompts as a
25-way array, finishes inside one link; a thinking arm needs several.

`run_sweep.py` appends each response and skips work already present, so a later
job resumes rather than restarts. That logic was written for preemption and works
just as well for splitting: submit a chain against the same output names.

```bash
MODEL=qwen3-4b-think; MEM=24G
EXPORT=ALL,GRAPHTALK_RUN_TAG=densfull40,GRAPHTALK_MAX_NEW_TOKENS=8192
PREV=""
for LINK in 1 2 3; do
  if [ -z "$PREV" ]; then
    PREV=$(sbatch --parsable --array=0-24 --mem=$MEM --export=$EXPORT \
                  cluster/sweep.sbatch $MODEL)
  else
    PREV=$(sbatch --parsable --array=0-24 --mem=$MEM --export=$EXPORT \
                  --dependency=afterany:$PREV cluster/sweep.sbatch $MODEL)
  fi
  echo "link $LINK: $PREV"
done
```

`afterany` starts the next link whenever the previous one ends — completed,
preempted, or out of wall clock. Links that find the file already complete count
the remaining work and exit *before* the warm-up, so an over-long chain costs
seconds rather than 20 minutes each.

Keep the output names stable across links and requeues; a `%j` in the path would
make every one of them start over.

**There is a 100-job submit cap per user** (QOS `general`, `MaxSubmitPU=100`),
and every array task counts: three linked 25-way arrays are 75. Check
`squeue --me -r -h | wc -l` before adding a link; on
`QOSMaxSubmitJobPerUserLimit`, wait for earlier shards to finish or cancel a
pending link.

## Sizing

The main sweep is 16,800 prompts per arm (`densfull40`) plus 4,200 in the
high-density extension (`densfull40hi`), run as the arrays in the budget table.
One 25-way shard of `densfull40` generates about 0.6M new tokens for
`qwen3-1.7b`, 0.2M for `qwen3-4b`, 1.9M for `qwen3-1.7b-think` and 1.5M for
`qwen3-4b-think` (the committed shards' `n_new_tokens`, summed). Measured
single-stream throughput is 7.1-7.6 tok/s on an l40s for `gemma4-e4b` and
`qwen3-8b`; the plain `qwen3-1.7b` shards each finished inside one 24 h link,
and a thinking shard needs several.

### Two levers if that is too slow

- **Batch the generation.** Single-stream leaves most of the GPU idle, but a
  batch runs until its *longest* member finishes and these completion lengths
  are ragged. Batching needs **left** padding for these decoder-only models, and
  this is a live hazard rather than a theoretical one: `gemma-4-E4B-it`
  defaults to `padding_side='left'`, but **`Qwen3-8B` defaults to `'right'`**, so
  a naive implementation would corrupt half the sweep. Wrong padding produces
  fluent garbage, not an error.

  Implemented as `graphtalk.hf_backend.generate_batch` and
  `scripts/run_sweep.py --batch-size N` (forwarded here as
  `GRAPHTALK_BATCH_SIZE=N sbatch cluster/sweep.sbatch <model>`), handling
  both the padding-side hazard above and the per-row-length recovery a
  batch's ragged finish times require (see the function's docstring).

  **Validated on a GPU: do not use it for a sweep.** Measured with
  `cluster/validate_batching.sbatch` (git tag `pre-cleanup`) on 2026-09-04: L40S,
  `--batch-size 4`, the 24 budget-reference prompts in
  `preliminary/analysis/budget-*.jsonl`, both families, against single-stream
  output:

  | | gemma4-e4b | qwen3-8b |
  |---|---|---|
  | identical decoded text | 12/24 | 13/24 |
  | identical extracted answer | -- | 21/24 |
  | speedup over single-stream | -- | **1.44x** (0.15 -> 0.22 gen/s) |

  The gain is 1.44x and it is not free. On `qwen3-8b` three of 24 answers
  changed, one of them flipping a *correct* `cycle_check` response to a wrong
  one -- a ~4% perturbation of the score, against the pilot's
  `degree`-vs-`none` effect of 6.5 points. Paying 4% of the measurement to save
  31% of the wall clock is a bad trade, so the sweeps run single-stream.

  This is **not** the padding bug feared above: ten of eleven text mismatches
  agree on a long prefix (57-942 chars) before diverging, which is the
  floating-point non-associativity of batched vs. unbatched matmuls flipping a
  near-tie token -- wrong padding would have produced garbage from the first
  token everywhere. The default is `--batch-size 1`.
- **Ask for a faster card.** The h100s are **not** reachable from `killable` —
  n-102 and t-100 live in `gpu-h100-killable`, so the `h100` term in the
  `--constraint` can never match while `--partition` is `killable`. Override the
  partition to use them:

  ```bash
  sbatch --partition=gpu-h100-killable --constraint=h100 cluster/sweep.sbatch qwen3-14b
  ```

  It queues longer; the partition was 8 jobs deep when last checked.

## Preemption

`killable` means a higher-priority job can stop this one at any time; Slurm
requeues it. `run_sweep.py` flushes each response as it is produced, so a requeue
costs at most the row in flight.

Check on a run:

```bash
squeue --me -o "%.10i %.20j %.8T %.10M %R"
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,NodeList
wc -l data/runs/qwen3-4b.densfull40.shard*.jsonl
```
