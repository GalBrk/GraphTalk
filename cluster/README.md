# Running the sweep on the TAU CS cluster

Account `galbarak2`, DCOR lab, partition `killable` / account `gpu-research`.
The general cluster reference lives in the SlidesGen repo
(`training/CLUSTER.md`); this file covers only what GraphTalk needs on top of it.

## The pipeline is three stages, and only the middle one needs a GPU

| stage | script | where | needs |
|---|---|---|---|
| 1. build prompts | `scripts/build_prompts.py` | laptop | network, no torch |
| 2. generate | `scripts/run_sweep.py` | cluster | GPU, torch, transformers |
| 3. score | `scripts/score_sweep.py` | laptop | nothing |

Splitting it this way means the prompt set is a file you can read and diff before
spending GPU time, every model is handed the identical file, and the scoring can
be re-run and changed without regenerating anything.

## One-time setup

The cluster has no conda and a 2 GB home quota, so everything goes on the lab
netapp. Anaconda is already installed at `/home/dcor/galbarak2/anaconda3` from the
SlidesGen work; only the env is new.

```bash
conda create -y -n graphtalk python=3.11
conda activate /home/dcor/galbarak2/conda_envs/graphtalk
pip install torch transformers accelerate networkx numpy
```

Then pre-download the models **on the login node**, because compute nodes run with
`HF_HUB_OFFLINE=1`:

```bash
export HF_HOME=/home/dcor/galbarak2/hf_cache
python -c "
from huggingface_hub import snapshot_download
for repo in ('google/gemma-4-E4B-it', 'google/gemma-4-12B-it',
             'Qwen/Qwen3-8B', 'Qwen/Qwen3-14B'):
    snapshot_download(repo)
"
```

Gemma is a gated repo: accept the licence on the model page and
`huggingface-cli login` first, or the download 401s. Run the download inside
`tmux` — it is tens of GB and a dropped SSH connection kills it.

## Running

```bash
# stage 1, on the laptop, then copy prompts.jsonl to the cluster
PYTHONPATH=. .venv/bin/python scripts/build_prompts.py --count 30

# stage 2, on the cluster, once per model
sbatch cluster/sweep.sbatch gemma4-e4b
sbatch cluster/sweep.sbatch gemma4-12b
sbatch cluster/sweep.sbatch qwen3-8b
sbatch cluster/sweep.sbatch qwen3-14b

# stage 3, back on the laptop
PYTHONPATH=. .venv/bin/python scripts/score_sweep.py --responses runs/*.jsonl
```

Smoke-test first — `--limit 20` on `run_sweep.py` generates twenty rows and exits,
which catches a bad chat template or an OOM in two minutes instead of two hours.

## Sizing

At the proposal's 30 rows per task the prompt file is **2,520 prompts** per model
(180 instances x 7 conditions x 2 styles), so 10,080 generations across the four
models.

Half of those are `zero_cot` at up to 1024 new tokens and half are `zero_shot` at
64, so the CoT half dominates the runtime. Budget roughly 4-6 hours per 12B-class
model at single-stream generation. `--time=12:00:00` in the sbatch is deliberately
generous; `killable` caps at 1 day.

Two levers if that is too slow:

- **Batch the generation.** Single-stream leaves most of the GPU idle. Batching
  needs left-padding for these decoder-only models, and getting the padding side
  wrong produces fluent garbage rather than an error, so it is worth doing
  carefully rather than quickly.
- **Ask for a faster card.** The `--constraint` allows `a6000|l40s|h100`; pinning
  `h100` alone queues longer but runs several times faster.

## Preemption

`killable` means a higher-priority job can stop this one at any time; Slurm
requeues it. `run_sweep.py` appends each response as it is produced and skips
completed work on restart, so a requeue costs one row. Keep `--out` stable across
requeues — putting `%j` in the path would make every requeue start over.

Check on a run:

```bash
squeue --me -o "%.10i %.20j %.8T %.10M %R"
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,NodeList
```
