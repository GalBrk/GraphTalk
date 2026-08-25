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
/home/dcor/galbarak2/conda_envs/graphtalk/bin/pip install -e ".[dev]"
/home/dcor/galbarak2/conda_envs/graphtalk/bin/pip install \
    torch transformers accelerate huggingface_hub
```

Redirect the pip cache before installing. The CUDA wheels are several GB and the
default `~/.cache/pip` would eat most of the 6 GB home quota.

`pip install -e` works normally here; the `PYTHONPATH=.` prefix in the top-level
README is a macOS-only workaround for a broken editable install. Verify with
`pytest -q`, which must report **345 passed** — a different number means the env
is wrong, not the code.

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
python scripts/build_prompts.py --count 30      # writes 2520 prompts

# stage 2, on the cluster, a chain per model (see below)
sbatch --exclude=n-801 --mem=32G cluster/sweep.sbatch qwen3-8b

# stage 3, on the login node
python scripts/score_sweep.py --responses runs/*.jsonl
```

Smoke-test first. Passing a second argument runs that many generations and
writes them to `runs/smoke-<model>.jsonl` rather than the sweep's own file, so a
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

| node | driver | usable |
|---|---|---|
| n-602 | 595.84 | yes |
| n-805 | 580.173.02 | yes |
| t-806 | 580.105.08 | yes |
| **n-802, n-803, n-804** | **535.183.01** (CUDA 12.2) | **no** |

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
re-running the 345 tests.

### n-801 is slow; exclude it

Read throughput varies by node far more than expected. Measured with 2 GiB of
direct I/O, twice each:

| node | throughput |
|---|---|
| n-802, n-805 | ~31 MB/s |
| **n-801** | **12.4 MB/s idle, 3.4 MB/s under load** |

n-801 had a 195-day uptime and both stalls in this project landed on it. Pass
`--exclude=n-801` until someone reboots it.

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

At the measured `zero_shot` budget of 2048 tokens a model needs roughly **45
hours** — about 22 h for the zero_shot half and a comparable amount for zero_cot.
`killable` caps at 24 h, so no single job finishes a model.

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

At 30 rows per task the prompt file is **2,520 prompts** per model (180 instances
x 7 conditions x 2 styles), so 10,080 generations across the four models.

Both styles now generate freely — `zero_shot` at 2048 new tokens and `zero_cot`
at 1024 — because these instruction-tuned models narrate their working before
answering. See `graphtalk/models.py` for the measurement behind those numbers and
for what it does to the contrast between the two styles.

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
