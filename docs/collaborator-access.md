# Getting at the data

Two routes. Cloning gives you the finished results anywhere; the cluster gives you
the models and the environments as well, without re-downloading 111 GB.

## Off-cluster: clone

```bash
git clone git@github.com:ArnavShahor/GraphTalk.git
```

**[DATA.md](DATA.md) documents every file's schema and how they join.** Everything needed to score and analyse is tracked: `runs/*.jsonl` (the raw model
responses), `prompts.jsonl` (the exact prompts they answer), and `shortcuts.json`
(the primer-only solver score each cell is read against). See `runs/README.md` for
the row schema and `docs/sweep-findings.md` for what the numbers do and do not
support -- particularly that the McNemar analysis as specified is underpowered.

## On the TAU CS cluster: read in place

Everything below is world-readable. Nothing needs to be copied and no permission
has to be requested.

```bash
REPO=/home/dcor/galbarak2/GraphTalk

ls $REPO/runs/                       # raw responses, including live shard files
cat $REPO/shortcuts.json             # the interpretation bar
```

### Score without building an environment

Both conda envs are readable, so activate one rather than installing your own:

```bash
source /home/dcor/galbarak2/anaconda3/etc/profile.d/conda.sh
conda activate /home/dcor/galbarak2/conda_envs/graphtalk-cu126

cd $REPO
python scripts/score_sweep.py --responses runs/*.jsonl --shortcuts shortcuts.json
```

`graphtalk` is a cu130 build and needs driver 580+, so it only runs on n-602,
n-805 and t-806. `graphtalk-cu126` runs on every GPU node in `killable` and is
the one to prefer -- it cannot drive a B200, which `killable` does not have.

### Run models without downloading them

The checkpoint cache is reachable read-only:

```bash
export HF_HOME=/home/dcor/galbarak2/hf_cache
export HF_HUB_OFFLINE=1
```

Four checkpoints are cached: `google/gemma-4-E4B-it`, `google/gemma-4-12B-it`,
`Qwen/Qwen3-8B`, `Qwen/Qwen3-14B` (111 GB). `HF_HOME` is mode 711 -- you can
traverse to the models but not list the directory, which keeps the owner's API
token private. Point `HF_HOME` at it and the libraries find the models by path.

You cannot write to that cache. To fetch a checkpoint that is not there, set
`HF_HOME` somewhere of your own.

## Submitting your own runs

Do not point a second job at a model already being generated: `run_sweep.py`
reads the completed set once at startup, so two concurrent jobs on one model
regenerate each other's rows and interleave duplicates into the same file. Use a
distinct `--out`, or shard with a job array (`--array=0-3`), which splits the
prompt file and gives each task its own output. See `cluster/README.md`.
