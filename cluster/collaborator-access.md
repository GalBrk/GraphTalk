# Getting at the data

Two routes. Cloning gives you the finished results anywhere; the cluster gives you
the models and the environments as well, without re-downloading 111 GB.

## Off-cluster: clone

```bash
git clone https://github.com/GalBrk/GraphTalk.git
```

Everything needed to rerun the analyses is tracked: the main experiment's
prompts, raw responses and solver bars in `data/`
([data/README.md](../data/README.md)), and the pilot's in `preliminary/data/`
([preliminary/data/README.md](../preliminary/data/README.md)).
[scripts/README.md](../scripts/README.md) reruns every number from them;
[docs/README.md](../docs/README.md) indexes the results.

## On the TAU CS cluster: read in place

Everything below is world-readable. Nothing needs to be copied and no permission
has to be requested.

```bash
REPO=/home/dcor/galbarak2/GraphTalk

ls $REPO/data/runs/                      # raw responses
cat $REPO/data/shortcuts_n40_flat.json   # the graph-blind solver's bars
```

That checkout is the lab's working copy: read it, but clone your own to run
anything that writes (`git pull`, jobs, analyses that write `outputs/`).

### Score without building an environment

Both conda envs are readable, so use one rather than installing your own:

```bash
source /home/dcor/galbarak2/anaconda3/etc/profile.d/conda.sh
conda activate /home/dcor/galbarak2/conda_envs/graphtalk-cu126

cd <your clone>
PYTHONPATH=. python scripts/build_raw_frame.py   # then the rest of scripts/README.md
```

`graphtalk` is a cu130 build and needs driver 580+; the nodes without it are the
535.x ones -- n-501, n-802, n-803, n-804 -- where `sweep.sbatch`'s own CUDA
guard fails the job in about 90 seconds rather than silently falling back to
CPU. [README.md](README.md)'s driver table lists every node. `graphtalk-cu126`
runs on both driver generations and is the one to prefer if you do not want to
think about it -- it cannot drive a B200, which `killable` does not have.

### Run models without downloading them

The checkpoint cache is reachable read-only. Load weights from its hub
directory, and give `HF_HOME` a writable directory of your own: `huggingface_hub`
writes a `token` file at `HF_HOME`'s root, so an `HF_HOME` inside the read-only
cache fails every load with `PermissionError`.

```bash
# interactive
export HF_HUB_CACHE=/home/dcor/galbarak2/hf_cache/hub
export HF_HOME=<a writable directory of your own>
export HF_HUB_OFFLINE=1
```

`cluster/sweep.sbatch` sets these itself: `HF_HUB_CACHE` from
`GRAPHTALK_HF_CACHE` (default the shared hub above) and `HF_HOME` from
`GRAPHTALK_HF_HOME` (default `.hf_home/` in the clone you submit from), so a job
needs neither exported.

Seven GraphTalk checkpoints are cached: `google/gemma-4-E4B-it`,
`google/gemma-4-12B-it`, `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`,
`Qwen/Qwen3-14B` and `Qwen/Qwen3.5-2B`. [README.md](README.md)'s one-time setup
downloads `Qwen/Qwen3-4B`, the main sweep's other model, into the same cache;
before a `qwen3-4b` job, confirm it is there with
`ls -d /home/dcor/galbarak2/hf_cache/hub/models--Qwen--Qwen3-4B`, because a
cache miss fails on the node after queueing. The cache is shared with other
projects on this account and holds 14 repos / 141 GB in total, so do not read
its size as this project's footprint. The cache root,
`/home/dcor/galbarak2/hf_cache`, is mode 711 -- you can traverse to the models
but not list the directory, which keeps the owner's API token private.

You cannot write to that cache. To fetch a checkpoint that is not there,
download it on the login node into a hub directory of your own and point
`HF_HUB_CACHE` (interactive) or `GRAPHTALK_HF_CACHE` (sbatch) at it.

## Submitting your own runs

Do not point a second job at a model already being generated into the same
file: `run_sweep.py` reads the completed set once at startup, so two concurrent
jobs on one output regenerate each other's rows and interleave duplicates into
it. Use a distinct `GRAPHTALK_RUN_TAG` or `GRAPHTALK_RUNS_DIR`, or shard with a
job array (`--array=0-24`), which splits the prompt file and gives each task its
own output. See [README.md](README.md).
