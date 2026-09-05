# Running the sweep on the TAU CS cluster

Account `galbarak2`, DCOR lab, partition `killable` / account `gpu-research`.
The general cluster reference lives in the SlidesGen repo
(`training/CLUSTER.md`); this file covers only what GraphTalk needs on top of it.

## The pipeline is three stages, and only the middle one needs a GPU

| stage | script | where | needs |
|---|---|---|---|
| 1. build prompts | `scripts/build_prompts.py` | login node | network, no torch |
| 2. generate | `scripts/run_sweep.py` | compute node | GPU, torch, transformers |
| 3. score | `scripts/score_sweep.py` | login node | nothing |

Splitting it this way means the prompt set is a file you can read and diff before
spending GPU time, every model is handed the identical file, and the scoring can
be re-run and changed without regenerating anything.

Stage 1 runs on the **login node**, not a laptop: it fetches rows from the
HuggingFace datasets-server over plain `urllib`, and compute nodes have no
outbound network.

## One-time setup

Home is a **6 GB** quota with a 102k file cap, so everything goes on the lab
netapp. Anaconda is already installed at `/home/dcor/galbarak2/anaconda3` from
the SlidesGen work; only the env is new.

```bash
source /home/dcor/galbarak2/anaconda3/etc/profile.d/conda.sh
conda create -y -p /home/dcor/galbarak2/conda_envs/graphtalk python=3.11
```

Use `-p` and the full path, not `-n graphtalk`. `envs_dirs` does not include
`conda_envs/` — it resolves to `anaconda3/envs` — so `-n` would silently put the
env somewhere `cluster/sweep.sbatch` does not look.

```bash
export PIP_CACHE_DIR=/home/dcor/galbarak2/pip_cache
export TMPDIR=/home/dcor/galbarak2/tmp
/home/dcor/galbarak2/conda_envs/graphtalk/bin/pip install -e ".[dev,analysis]"
/home/dcor/galbarak2/conda_envs/graphtalk/bin/pip install \
    torch transformers accelerate huggingface_hub
```

Redirect the pip cache before installing. The CUDA wheels are several GB and the
default `~/.cache/pip` would eat most of the 6 GB home quota.

`pip install -e` works normally here; the `PYTHONPATH=.` prefix in the top-level
README is a macOS-only workaround for a broken editable install. Verify with
`pytest -q`, which must report **359 passed** — a different number means the env
is wrong, not the code. Install the `analysis` extra as above and not `[dev]`
alone: `tests/test_analysis.py` imports `pandas` at module scope, so without it
pytest aborts during *collection* and reports an error rather than 348 passes.

Then pre-download the models **on the login node**, because compute nodes run
with `HF_HUB_OFFLINE=1`:

```bash
export HF_HOME=/home/dcor/galbarak2/hf_cache
python -c "
from huggingface_hub import snapshot_download
for repo in ('google/gemma-4-E4B-it', 'google/gemma-4-12B-it',
             'Qwen/Qwen3-8B', 'Qwen/Qwen3-14B'):
    snapshot_download(repo)
"
```

Roughly 85 GB across the four. Run it inside `tmux` — a dropped SSH connection
kills it. None of the four is gated: the Gemma repos report `gated: false` and
download without a licence acceptance or a `huggingface-cli login`, though a
token from earlier work was present and does no harm.

## Running

```bash
# stage 1, on the login node
python scripts/build_prompts.py --count 30      # writes 1260 prompts

# stage 2, on the cluster, a chain per model (see below)
sbatch --exclude=n-801 --mem=32G cluster/sweep.sbatch qwen3-8b

# stage 3, on the login node
python scripts/score_sweep.py --responses runs/*.jsonl
```

Smoke-test first. Passing a second argument runs that many generations and
writes them to `runs/archive/smoke-<model>.jsonl` rather than the sweep's own file, so a
run that reveals a broken template cannot leave rows the real sweep then skips:

```bash
sbatch --time=00:40:00 --exclude=n-801 cluster/sweep.sbatch gemma4-e4b 20
```

**A 20-row smoke test is not a representative one.** The prompt file is ordered
by task, so the first 420 rows are all `node_count`, and a 20-row limit sees one
task out of six. That matters because a truncated generation shows up as
unparseable on `cycle_check` but as a confident *wrong answer* on `node_count`,
where the extractor picks an integer out of the abandoned working. The first
smoke test here reported a 100% parse rate while half the sweep was truncated.
Check a spread of tasks before trusting it.

### Regenerating part of a sweep

When a prompt changes, only the affected rows need re-running. Two overrides make
that a first-class operation rather than a hand-edit:

```bash
# 1. strip the affected rows from the arm -- run_sweep.py skips keys it already
#    has, so a row left in place will NOT be regenerated, silently.
# 2. point at a subset file and tag the output.
GRAPHTALK_PROMPTS=prompts.rerun.jsonl GRAPHTALK_RUN_TAG=rerun \
  sbatch --time=12:00:00 --mem=40G cluster/sweep.sbatch gemma4-12b
#   -> runs/gemma4-12b.rerun.jsonl   (or .rerun.shardNofM.jsonl under --array)
```

`score_sweep.py` pools by each row's `model` field, so a tagged file rejoins its
arm with no reassembly. Verify afterwards that the arm is back to its full unique
key count -- that is the check that catches a silent skip.

**Never tag a regeneration `redo`.** `analysis._EXCLUDE_SUBSTRINGS` matches
`.redo.shard`, so those rows would be dropped from every frame with no error;
`sweep.sbatch` refuses the tag outright for that reason. The existing
`runs/archive/*.redo.shard*.jsonl` are a different artefact that `docs/DATA.md`
deliberately excludes -- and since exclusion is now by directory, keeping
regenerated rows out of `runs/archive/` is what matters more than the tag.

### Running the GoT node-naming scheme

`GRAPHTALK_PROMPTS`/`GRAPHTALK_RUN_TAG` above are also how a
Game-of-Thrones-named arm is run (`graphtalk/node_naming.py`; see
`README.md#node-naming` for what the scheme is). `cluster/submit_sweep.sh`
wraps that into one flag, and does stage 1 for you first if it hasn't run yet:

```bash
cluster/submit_sweep.sh --node-naming got --exclude=n-801 --mem=32G \
    cluster/sweep.sbatch gemma4-12b
```

Every other `sbatch` flag or positional (`--array`, `--exclude`, the model
key, the smoke-test limit) passes straight through in whatever position it's
given -- only `--node-naming`, `--count`, and `--dry-run` are consumed by the
wrapper. `--count N` (GoT scheme only) requests a prompt file larger than the
tracked sweep's 30-per-task default -- e.g. for a targeted follow-up sized by
`scripts/recommend_count.py` (see `analysis/README.md`'s Track 2 section) --
tagged into both the prompt filename and `GRAPHTALK_RUN_TAG` so it can't
collide with the tracked `--count 30` sweep's own files:

```bash
cluster/submit_sweep.sh --node-naming got --count 500 \
    cluster/sweep.sbatch qwen3-8b
```

`--dry-run` prints what would run (and whether `prompts_got.jsonl` would be
built) without touching anything, which is worth doing once before the real
submission since the wrapper still can't be tested on a scheduler you don't
have.

That's exactly the two-step recipe from `README.md#node-naming` collapsed
into one call: **stage 1 still runs on the login node**, not inside the
job -- `build_prompts.py` fetches over plain `urllib`, and compute nodes have
no outbound network (`HF_HUB_OFFLINE=1`, same reason as everywhere else in
this file). The wrapper builds `prompts_got.jsonl` right there, before
`sbatch` is ever called, and reuses it on every later invocation rather than
rebuilding (`load_rows()`'s cache makes that safe -- see `README.md#node-naming`).
Omit `--node-naming` (or pass `--node-naming integer`) for the plain scheme;
nothing else about the wrapper's behavior changes.

## Warm the page cache, or the job dies loading

`sweep.sbatch` reads the whole checkpoint with `cat` before starting Python.
This is not a nicety. `safetensors` mmaps the file and faults tensor offsets in
checkpoint order rather than file order, and those scattered reads are
pathological over NFS: the first attempt projected a **nine-hour** load for a
16 GB checkpoint and died on its time limit having written no rows. One
sequential pass first costs about 20 minutes and drops the load to **two
seconds**.

The warm-up cost is paid per job on a cold node, and it dominates short
diagnostic runs — budget for it before submitting anything small.

## Half the partition has a driver this torch build cannot use

`killable` spans two driver generations, and the env's torch is a **cu130** build
that needs **580 or newer**:

| node | card | driver | usable |
|---|---|---|---|
| n-601, n-602 | a6000 | 595.84 (n-602) | yes |
| n-805 | l40s | 580.173.02 | yes |
| t-806 | l40s | 580.105.08 | yes |
| n-502, n-503 | a5000 | 580+ | yes |
| **n-802, n-803, n-804** | l40s | **535.183.01** (CUDA 12.2) | **no** |
| **n-501** | a5000 | **535.x** (CUDA 12.2) | **no** |

**`n-501` is on this list and is easy to miss** -- it is an a5000 node, so it is
not caught by thinking of the bad nodes as "the l40s ones". It cost three
separate job failures on 2026-09-05 before it was identified. The default
`--constraint=a6000|l40s|h100` spans both driver generations, so **every
submission is a dice roll**; the driver guard in `sweep.sbatch` turns that into a
fast, visible failure (~90 s, non-zero exit) rather than a silent CPU fallback,
but it does not prevent it. Two ways to make placement deterministic:

```bash
# pin to nodes known good for the cu130 env
sbatch --constraint=a6000 ... cluster/sweep.sbatch <model>

# or use the cu126 build, which runs on BOTH driver generations
sbatch --export=ALL,GRAPHTALK_ENV=graphtalk-cu126 ... cluster/sweep.sbatch <model>
```

Pinning to `a6000` keeps one card type and one CUDA build across an arm, which
matters when the arm is a headline result; `graphtalk-cu126` places faster
because it can use every node. Note `a6000` is only n-601 and n-602 (16 GPUs,
shared cluster-wide), so it can queue.

On an old node `device_map="auto"` finds no usable CUDA device and puts the model
on the **CPU** — with no error and no warning, at roughly a fortieth of the
speed. Three jobs ran that way for sixteen hours before it was spotted, and the
symptom is indistinguishable from a busy filer or a contended card, so it costs a
long detour to diagnose. The tell is `nvidia-smi` reporting **0 MiB used on your
own assigned device** while the process holds the weights in host RAM.

`sweep.sbatch` now refuses to start on such a node. Submit with the old nodes
excluded so the scheduler does not waste a link finding out:

```bash
sbatch --exclude=n-801,n-802,n-803,n-804 --mem=32G cluster/sweep.sbatch qwen3-8b
```

Do not check the driver on the login node and assume it generalises — the login
node is on 580 while three compute nodes are not, and that mistake is what let
this through in the first place. A smoke test passing proves only that *that*
job's node was fine.

The longer-term fix is a cu12 torch build, which runs on both generations and
would restore the full node pool; it means reinstalling into the env and
re-running the 359 tests. The `graphtalk-cu126` env is that build, and is
already in use — see the env note in `cluster/sweep.sbatch`.

### n-801 is slow; exclude it

Read throughput varies by node far more than expected. Measured with 2 GiB of
direct I/O, twice each:

| node | throughput |
|---|---|
| n-802, n-805 | ~31 MB/s |
| **n-801** | **12.4 MB/s idle, 3.4 MB/s under load** |

n-801 had a 195-day uptime and both stalls in this project landed on it. Pass
`--exclude=n-801` until someone reboots it.

### Do not put several checkpoint warm-ups on one node at once

Measured during the 2026-08-28 prompt-rewording re-run. Four plain arms were
submitted together; the scheduler put **three on n-602**, where they each began a
sequential read of a 14.9 / 22.3 / 27.5 GB checkpoint at the same time. After
**3.5 hours not one had finished warming**, which puts each stream under
**1.2 MB/s** and the node's aggregate around 3.6 MB/s -- the same range this file
already flags n-801 for. The fourth arm, alone on n-601, warmed 15.3 GB in 22
minutes (~11.8 MB/s) and finished the whole job in 63 minutes.

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

The same re-run submitted the plain arms with `--time=06:00:00`, sized from ~20 min
of warm-up plus an hour of generation. Under the contention above the warm-up alone
was heading past 5 hours, so two of the three would have hit the wall having written
**zero rows**. `--time` is a ceiling, not a reservation -- the job releases the
allocation when it exits -- so there is no reason to trim it. Use 12 h for anything
that has to warm a checkpoint it might be sharing bandwidth for.

Slurm will not let you fix this after the fact: `scontrol update jobid=<j>
TimeLimit=...` upward returns `Access/permission denied` for an ordinary user. The
only remedy is `scancel` and resubmit.

A resubmit is a **full re-read** -- do not expect the node's page cache to help.
It is tempting to send the job back to the same node on the grounds that n-602 has
1 TB of RAM against ~65 GB of checkpoints, so the bytes already read should still be
cached. That was tried here and did not work: `qwen3-14b` had read ~14 GB of its
27.5 GB checkpoint, was cancelled and resubmitted to the same node, and then took
almost exactly the time that reading all 27.5 GB from scratch at the contended rate
predicts. Whatever the reason -- NFS client caching, or eviction under the other
jobs on the node -- budget a restart as if nothing were cached, and pick the node on
current load rather than on history.

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
happily and produce plausible-looking wrong data.)

This happened on 2026-09-05: 28 rows were generated wrongly-strided, 12 of them
duplicating rows owned by `shard0of5`/`shard3of5`.

To resubmit a subset of shards, pass the count explicitly rather than relying on
an array:

```bash
for s in 1 2 4; do
  sbatch --job-name=ec8b-s${s} --constraint=a6000 \
    --export="ALL,SLURM_ARRAY_TASK_ID=${s},SLURM_ARRAY_TASK_COUNT=5,GRAPHTALK_PROMPTS=...,GRAPHTALK_RUN_TAG=..." \
    cluster/sweep.sbatch qwen3-8b
done
```

**Also prefer an ODD shard count.** The prompt file alternates conditions within
each task block, so with 2 conditions an even `--array` count preserves stride
parity: every even shard generates only `none` and every odd shard only the
treatment. Nothing is lost -- all rows are still produced -- but partial progress
is unpaired, so any mid-run comparison is across different instance sets and
meaningless. An odd count mixes both conditions into every shard.

## Memory is per-model, and it decides whether you are scheduled at all

The `--mem` request must hold the checkpoint in the page cache the warm-up fills,
plus the loader's buffers. The weights leave host memory for the GPU, so the
headroom above the checkpoint size is comfortable:

| model | checkpoint | `--mem` |
|---|---|---|
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

At the measured `zero_shot` budget of 2048 tokens a model needs roughly **22
hours**, close enough to `killable`'s 24 h cap that a single job is not a safe
bet -- chain anyway, as below.

`run_sweep.py` appends each response and skips work already present, so a later
job resumes rather than restarts. That logic was written for preemption and works
just as well for splitting: submit a chain against the same `--out`.

```bash
MODEL=qwen3-8b; MEM=32G
PREV=""
for LINK in 1 2 3; do
  if [ -z "$PREV" ]; then
    PREV=$(sbatch --parsable --exclude=n-801 --mem=$MEM cluster/sweep.sbatch $MODEL)
  else
    PREV=$(sbatch --parsable --exclude=n-801 --mem=$MEM \
                  --dependency=afterany:$PREV cluster/sweep.sbatch $MODEL)
  fi
  echo "link $LINK: $PREV"
done
```

`afterany` starts the next link whenever the previous one ends — completed,
preempted, or out of wall clock. Links that find the file already complete count
the remaining work and exit *before* the warm-up, so an over-long chain costs
seconds rather than 20 minutes each.

Keep `--out` stable across links and requeues; a `%j` in the path would make
every one of them start over.

## Sizing

At 30 rows per task the prompt file is **1,260 prompts** per model (180 instances
x 7 conditions), so 5,040 generations across the four models.

Generation runs freely at 2048 new tokens because these instruction-tuned
models narrate their working before answering. See `graphtalk/models.py` for
the measurement behind that number.

Measured single-stream throughput is 7.1-7.6 tok/s on an l40s for the smaller two
models; the 12B and 14B are slower per token, so treat 45 h as optimistic for
them and add links to the chain rather than assuming three is enough.

### Two levers if that is too slow

- **Batch the generation.** Single-stream leaves most of the GPU idle. Worth
  perhaps 3-5x here rather than the headline 8-10x, because a batch runs until
  its *longest* member finishes and these completion lengths are ragged (median
  271, max 1974). Batching needs **left** padding for these decoder-only models,
  and this is a live hazard rather than a theoretical one: `gemma-4-E4B-it`
  defaults to `padding_side='left'`, but **`Qwen3-8B` defaults to `'right'`**, so
  a naive implementation would corrupt half the sweep. Wrong padding produces
  fluent garbage, not an error. Verify against the single-stream responses in
  `analysis/budget-*.jsonl`: decoding is greedy, so a correct
  batched implementation reproduces them near-identically.

  Implemented as `graphtalk.hf_backend.generate_batch` and
  `scripts/run_sweep.py --batch-size N` (forwarded here as
  `GRAPHTALK_BATCH_SIZE=N sbatch cluster/sweep.sbatch <model>`), handling
  both the padding-side hazard above and the per-row-length recovery a
  batch's ragged finish times require (see the function's docstring).

  **Now validated on a GPU, and the answer is: do not use it.** Run with
  `cluster/validate_batching.sbatch` (2026-09-04, L40S, `--batch-size 4`,
  the 24 budget-reference prompts, both families):

  | | gemma4-e4b | qwen3-8b |
  |---|---|---|
  | identical decoded text | 12/24 | 13/24 |
  | identical extracted answer | -- | 21/24 |
  | speedup over single-stream | -- | **1.44x** (0.15 -> 0.22 gen/s) |

  The **3-5x above was optimistic**: the measured gain is 1.44x. And it is
  not free. On `qwen3-8b` three of 24 answers changed, one of them flipping
  a *correct* `cycle_check` response to a wrong one -- a ~4% perturbation of
  the score, against a `degree`-vs-`none` effect size of only 6.5 points.
  Paying 4% of your measurement to save 31% of your wall clock is a bad
  trade, and the GoT `--count 500` replication was run single-stream for
  exactly this reason.

  To be fair to the implementation, this is **not** the padding bug feared
  above: ten of eleven text mismatches agree on a long prefix (57-942
  chars) before diverging, which is the floating-point non-associativity of
  batched vs. unbatched matmuls flipping a near-tie token -- wrong padding
  would have produced garbage from the first token everywhere. The code
  looks correct; batching is simply not worth its cost *here*, at these
  budgets and this effect size. It may be worth revisiting for a run where
  throughput matters more than a few points of per-row fidelity.

  Default stays `--batch-size 1` (today's exact single-stream path), so
  nothing about an ordinary invocation changes until this flag is opted
  into.
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
wc -l runs/*.jsonl
```
