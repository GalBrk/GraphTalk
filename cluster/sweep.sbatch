#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gpus=1
# The Ampere/Ada cards with bf16 tensor cores: 24 GB (3090, a5000) and 48 GB
# (a6000, l40s). The main sweep's largest model is 4B (~8 GB of bf16 weights),
# so it fits anywhere in this pool. Every card here has at least 24 GB, which
# covers the registry up to `min_vram_gb=24`; the 48 GB models (qwen3-14b,
# gemma4-12b) do NOT fit the 24 GB half, so the VRAM guard further down refuses
# them on a too-small card rather than letting them OOM an hour in -- that
# guard, not this line, is what keeps the wide pool safe for the big models.
# Driver versions were measured across the whole partition (the table in
# cluster/README.md); all six reachable 3090 nodes are on 595.84, past the 580+
# the default cu130 env needs.
#
# The 2080 nodes and the DGX v100/quadro nodes stay out: Turing and Volta have
# no bf16 tensor cores, so `device_map="auto"` would place all or part of the
# model on CPU there -- silently, per the driver check below, not with an error.
#
# The `h100` term never matches from `killable`: n-102 and t-100 live in
# `gpu-h100-killable` (see the override below).
#SBATCH --constraint="a6000|l40s|a5000|geforce_rtx_3090|h100"
# The 535.x-driver nodes the default cu130 env cannot use (n-501, an a5000, and
# the l40s nodes n-802, n-803, n-804), plus n-801, whose read throughput
# collapses to 3.4 MB/s under load (its driver is fine at 580.126.09; it is the
# I/O that is not). Measured 2026-09-17.
#
# An --exclude on the sbatch command line REPLACES this list rather than adding
# to it, so `--exclude=n-801` re-admits all four 535.x nodes. That is right for
# GRAPHTALK_ENV=graphtalk-cu126, which runs on both driver generations, and
# wrong for the default env.
#SBATCH --exclude=n-501,n-801,n-802,n-803,n-804
#SBATCH --cpus-per-task=8
# Sized to hold the largest checkpoint in the page cache the warm-up below fills:
# 29.6 GB for Qwen3-14B, plus the loader's buffers. The weights land on the GPU
# rather than staying resident here, so the headroom above 30 GB is comfortable.
#
# Not larger, and override it downward per model (cluster/README.md's --mem
# table): the nodes with a spare GPU are often the ones with the least RAM free,
# so a request larger than the checkpoint needs costs queue time.
#SBATCH --mem=64G
# The partition ceiling. A 25-way array finishes a plain arm of the main sweep
# inside one 24 h link; a thinking arm needs several, so submit a chain and let
# the resume logic below carry the work across them.
#SBATCH --time=24:00:00
#SBATCH --job-name=graphtalk_sweep
# Slurm does not create this directory: run `mkdir -p out` in the clone root
# before the first submission, or the job fails before it starts, with no log.
#SBATCH --output=out/slurm-%j.out
#SBATCH --signal=USR1@120

# One model over one prompt file. Submit from the root of your clone, once per
# model (cluster/README.md has the command for every committed run set):
#   mkdir -p out
#   sbatch --array=0-24 --export=ALL,GRAPHTALK_RUN_TAG=densfull40 cluster/sweep.sbatch qwen3-1.7b
#
# An optional second argument runs a smoke test of that many generations:
#   sbatch --time=00:40:00 cluster/sweep.sbatch qwen3-1.7b 20
#
# The h100 cards are NOT reachable from this partition: n-102 and t-100 live in
# `gpu-h100-killable`, so the `h100` term in the --constraint above can never
# match while --partition is `killable`. To use them, override the partition:
#   sbatch --partition=gpu-h100-killable --constraint=h100 cluster/sweep.sbatch qwen3-14b
#
# `killable` is preemptible, so this job can be stopped and requeued at any
# moment. run_sweep.py appends to its output and skips work already present, so a
# requeue resumes rather than restarting -- which is why --out must be a stable
# path across requeues and not something with a job id in it.
#
# Environment overrides, all optional (sbatch --export=ALL,NAME=value,...):
#   GRAPHTALK_PROMPTS         prompt file (default data/prompts/prompts.densfull40.jsonl)
#   GRAPHTALK_RUN_TAG         run-set name in the output file, <model>.<tag>.jsonl
#   GRAPHTALK_MAX_NEW_TOKENS  token budget, in place of models.budget
#   GRAPHTALK_RUNS_DIR        output directory (default <submit dir>/data/runs)
#   GRAPHTALK_TASK_DIR        a subdirectory of the output directory to write into
#   GRAPHTALK_ENV             conda env (default graphtalk; graphtalk-cu126)
#   GRAPHTALK_CONDA           conda install whose etc/profile.d/conda.sh to source
#   GRAPHTALK_ENVS_DIR        directory holding the conda envs
#   GRAPHTALK_HF_CACHE        HF hub cache to load weights from (default: the lab's shared one)
#   GRAPHTALK_HF_HOME         writable HF_HOME (default <submit dir>/.hf_home)
#   GRAPHTALK_BATCH_SIZE      batched generation; not for a sweep (see below)

set -euo pipefail

MODEL="${1:?usage: sbatch cluster/sweep.sbatch <model-key> [limit]}"
LIMIT="${2:-}"

# Where the job was submitted from: the clone whose scripts run and whose
# data/runs/ receives the output. Anchored explicitly because RUNS_DIR is built
# before the `cd` below; $PWD is the fallback for running the script by hand.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# The prompt file is overridable so an arm can run another run set (the
# high-density extension is data/prompts/prompts.densfull40hi.jsonl) or a
# subset (scripts/reproduce_rows.py subset). Also how a GoT-named run points at
# preliminary/data/prompts/prompts_got.jsonl -- see
# preliminary/cluster/submit_sweep.sh, which builds it before ever setting this.
PROMPTS="${GRAPHTALK_PROMPTS:-data/prompts/prompts.densfull40.jsonl}"
if [[ ! -f "$PROMPTS" ]]; then
  # Fails already without this -- the REMAINING check below opens the file --
  # but with a bare Python traceback instead of a message that says what to
  # do about it, and this way it fails before conda even activates.
  echo "FATAL: prompt file '$PROMPTS' does not exist. Build it first" >&2
  echo "(e.g. preliminary/cluster/submit_sweep.sh --node-naming got ...) or point" >&2
  echo "GRAPHTALK_PROMPTS at an existing file." >&2
  exit 1
fi

# The submitting clone's data/runs/ unless GRAPHTALK_RUNS_DIR names another
# directory, e.g. a scratch directory for a check whose rows must stay out of
# every checkout.
RUNS_DIR="${GRAPHTALK_RUNS_DIR:-${SUBMIT_DIR}/data/runs}"

TASK_DIR="${GRAPHTALK_TASK_DIR:-}"
if [[ -n "$TASK_DIR" ]]; then
  RUNS_DIR="${RUNS_DIR}/${TASK_DIR}"
fi

# A smoke test writes to its own file. Sharing the sweep's output would make its
# rows count as done, so a smoke run that revealed a broken chat template would
# leave twenty bad generations that the full sweep then skips.
OUT="${RUNS_DIR}/${MODEL}.jsonl"
EXTRA=()
if [[ -n "$LIMIT" ]]; then
  # Straight into archive/: a smoke run is never part of the sweep. No analysis
  # reads it -- the main scripts glob data/runs/ itself, not its subdirectories,
  # and graphtalk.analysis excludes archive/ by directory.
  mkdir -p "${RUNS_DIR}/archive"
  OUT="${RUNS_DIR}/archive/smoke-${MODEL}.jsonl"
  EXTRA=(--limit "$LIMIT")
fi

# An optional run tag names the run set, so the output is
# <model>.<tag>.jsonl (plus a .shard<i>of<n> suffix under --array):
#
#   GRAPHTALK_RUN_TAG=densfull40 sbatch --array=0-24 ... qwen3-4b
#     -> data/runs/qwen3-4b.densfull40.shard0of25.jsonl ... shard24of25.jsonl
#
# Analyses select runs by this file name, not by the rows' `model` field:
# scripts/build_raw_frame.py reads data/runs/<arm>.densfull40.shard*.jsonl and
# <arm>.densfull40hi.shard*.jsonl only. Any other tag gives a file the frame
# never sees -- and one that merely starts with `densfull40` is read by
# primer_findings.py's `<arm>.densfull40*.shard*.jsonl` glob but not by the
# frame, so never use one (cluster/README.md, "Regenerating part of a run").
#
# `redo` is refused: graphtalk.analysis drops every `.redo.shard` path, so rows
# written under that tag would vanish from every preliminary frame in silence.
RUN_TAG="${GRAPHTALK_RUN_TAG:-}"
if [[ -n "$RUN_TAG" ]]; then
  if [[ "$RUN_TAG" == *redo* ]]; then
    echo "FATAL: run tag '$RUN_TAG' contains 'redo'. analysis.py excludes" >&2
    echo "'.redo.shard' paths, so these rows would vanish from every frame." >&2
    exit 1
  fi
  OUT="${OUT%.jsonl}.${RUN_TAG}.jsonl"
fi

# Submitting as a job array shards one model across that many GPUs. Each task
# takes every n-th prompt (run_sweep.py strides the file) and writes its own
# <out>.shard<i>of<n>.jsonl:
#   sbatch --array=0-24 --export=ALL,GRAPHTALK_RUN_TAG=densfull40 cluster/sweep.sbatch qwen3-1.7b-think
#
# Keep the width coprime with the prompt file's (task, condition) cycle -- 42
# rows in densfull40, 14 in densfull40hi -- or each shard gets a skewed subset of
# the pairs. The committed shards are 25-way, and 11-way for the plain arms'
# densfull40hi.
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
# If Slurm ever hands us a task id without a count, every task would fall back to
# NSHARDS=1, write to the same unsuffixed file, and regenerate each other's work
# in silence. Refuse instead.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "$NSHARDS" == "1" ]]; then
  echo "FATAL: SLURM_ARRAY_TASK_ID set but SLURM_ARRAY_TASK_COUNT is not;" >&2
  echo "shards would collide on one output file. Pass --num-shards by hand." >&2
  exit 1
fi
if [[ "$NSHARDS" -gt 1 ]]; then
  OUT="${OUT%.jsonl}.shard${SHARD}of${NSHARDS}.jsonl"
  EXTRA+=(--shard "$SHARD" --num-shards "$NSHARDS")
fi

# Batched generation (graphtalk.hf_backend.generate_batch), opt-in and unset by
# default, so an ordinary invocation runs the single-stream path the sweep was
# generated with. Validated on a GPU and not used for the sweep: batch size 4
# changed 3 of 24 answers for a 1.44x speedup (cluster/README.md, "Two levers").
BATCH_SIZE="${GRAPHTALK_BATCH_SIZE:-}"
if [[ -n "$BATCH_SIZE" ]]; then
  EXTRA+=(--batch-size "$BATCH_SIZE")
fi

# Override the spec's token budget (models.budget). The main sweep's four arms
# default to the 8192 their densfull40 rows were generated with; cluster/README.md
# lists the run sets generated at another budget, e.g. the plain arms'
# densfull40hi at 2048:
#   GRAPHTALK_MAX_NEW_TOKENS=2048 sbatch cluster/sweep.sbatch <model>
# Applies to every row in this job -- harmless for short answers, since
# generation still stops at EOS, not at the cap.
MAX_NEW_TOKENS="${GRAPHTALK_MAX_NEW_TOKENS:-}"
if [[ -n "$MAX_NEW_TOKENS" ]]; then
  EXTRA+=(--max-new-tokens "$MAX_NEW_TOKENS")
fi

# Which env, and therefore which part of the partition is reachable.
#
# The default `graphtalk` env is torch 2.13.0 built for CUDA 13.0. It needs
# driver 580+, which the 535.x nodes (n-501, n-802, n-803, n-804) do not have;
# the default --exclude above keeps jobs off them.
#
# graphtalk-cu126 is the same torch 2.13.0 built against CUDA 12.6, and CUDA
# minor-version compatibility means it runs on both driver generations. Verified
# on all three 535.183.01 l40s nodes: cuda.is_available() true, L40S detected at
# sm_89, bf16 matmul finite. Pair it with --exclude=n-801 to admit the 535.x
# nodes as well:
#
#   sbatch --exclude=n-801 --export=ALL,GRAPHTALK_ENV=graphtalk-cu126 cluster/sweep.sbatch <model>
#
# It cannot drive a B200: the cu126 build stops at sm_90 and Blackwell is sm_100.
# Irrelevant on `killable`, which has no B200, but it rules the env out if this
# ever moves to gpu-b200.
#
# Both envs are the lab's, readable in place (cluster/collaborator-access.md);
# GRAPHTALK_CONDA and GRAPHTALK_ENVS_DIR point at another install.
GRAPHTALK_ENV="${GRAPHTALK_ENV:-graphtalk}"
CONDA_ROOT="${GRAPHTALK_CONDA:-/home/dcor/galbarak2/anaconda3}"
ENVS_DIR="${GRAPHTALK_ENVS_DIR:-/home/dcor/galbarak2/conda_envs}"
set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENVS_DIR}/${GRAPHTALK_ENV}"
set -u
echo "env: $GRAPHTALK_ENV"

# Models are pre-fetched on the login node; compute nodes have no outbound
# network, and a job that tries to download blocks until it times out.
# Weights come from the lab-shared, read-only cache by default
# (cluster/collaborator-access.md lists what it holds).
#
# HF_HOME and HF_HUB_CACHE are set SEPARATELY and that split is load-bearing.
# Pointing HF_HOME at the shared cache fails: huggingface_hub writes a `token`
# file at HF_HOME's root, and the shared tree is read-only to everyone but its
# owner, so every load dies with `PermissionError at .../token`. HF_HOME must
# therefore be somewhere writable -- by default .hf_home/ in the submitting
# clone, the submitter's own -- with only HF_HUB_CACHE aimed at the shared
# blobs. Benign `Ignoring corrupted tree cache file ... Permission denied`
# warnings are expected from the same read-only-ness and are not an error.
#
# Override either:
#   GRAPHTALK_HF_CACHE=/path/to/hub   a hub cache holding your own weights
#   GRAPHTALK_HF_HOME=/path/to/dir    a writable HF_HOME somewhere else
#
# Offline is NOT negotiable: the GPU nodes have no route to huggingface.co, so a
# cache miss fails on the node after queueing rather than at submit time.
export HF_HOME="${GRAPHTALK_HF_HOME:-${SUBMIT_DIR}/.hf_home}"
export HF_HUB_CACHE="${GRAPHTALK_HF_CACHE:-/home/dcor/galbarak2/hf_cache/hub}"
mkdir -p "$HF_HOME"
export HF_HUB_OFFLINE=1
# Reclaims fragmented VRAM, which is what pushed a 12B run over the edge on a
# 48 GB card.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SUBMIT_DIR"
export PYTHONPATH="$SUBMIT_DIR"

mkdir -p "$RUNS_DIR" out
nvidia-smi

# Nothing left to do? Exit before warming. A chain is submitted with more links
# than the work usually needs, so the last ones find the file already complete --
# and the warm-up costs ~20 min whether or not there is anything to generate.
REMAINING="$(python - "$OUT" "$PROMPTS" "$SHARD" "$NSHARDS" <<'PY'
import json, os, sys

out, prompts = sys.argv[1], sys.argv[2]
shard, nshards = int(sys.argv[3]), int(sys.argv[4])
with open(prompts) as handle:
  records = [json.loads(line) for line in handle if line.strip()]
# Same stride the run itself uses, or a shard would count the whole file as
# outstanding and never take the early exit.
if nshards > 1:
  records = records[shard::nshards]
todo = {(r["instance_id"], r["condition"], r["style"]) for r in records}
if os.path.exists(out):
  with open(out) as handle:
    for line in handle:
      try:
        record = json.loads(line)
      except json.JSONDecodeError:
        continue  # a preempted job leaves one truncated line; it is regenerated
      todo.discard(
          (record["instance_id"], record["condition"], record["style"]))
print(len(todo))
PY
)"
echo "$REMAINING generations still to do for $MODEL"
if [[ "$REMAINING" == "0" ]]; then
  echo "nothing to do; exiting before the page-cache warm-up"
  exit 0
fi

# Refuse to run on a node whose driver this torch build cannot use.
#
# `device_map="auto"` falls back to CPU when no CUDA device is usable, silently:
# no error, no warning, just a model doing transformer inference on the host at
# roughly a fortieth of the speed. Three jobs ran that way for sixteen hours here
# before anyone noticed, and the symptom -- everything is mysteriously slow --
# looks exactly like a busy filer or an oversubscribed card, so it sends you
# chasing the wrong thing.
#
# The cause is that `killable` spans two driver generations: n-501, n-802, n-803
# and n-804 are on 535.x (CUDA 12.2) while every other node measured is on
# 580+/595 (cluster/README.md's table). The default env's torch is a cu130 build
# and needs 580 or newer, and a smoke test that passes proves only that its own
# node was fine.
#
# Checking costs one torch import against hours of silent waste. Exiting non-zero
# is deliberate: the job fails visibly and the chain's next link gets another
# roll of the scheduler, rather than the sweep filling up with CPU-generated rows.
python - <<'PY' || exit 1
import sys
import torch

if torch.cuda.is_available():
  print(f"cuda ok: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
  sys.exit(0)
print(f"FATAL: torch {torch.__version__} sees no usable CUDA device on this node.",
      file=sys.stderr)
print("The driver is probably too old for this build; check `nvidia-smi` and",
      file=sys.stderr)
print("exclude the 535.x nodes (n-501, n-802, n-803, n-804).", file=sys.stderr)
sys.exit(1)
PY

# Refuse to run a model that does not fit the card we landed on.
#
# The --constraint above spans 24 GB cards (3090, a5000) and 48 GB ones (a6000,
# l40s), because the small models this sweep runs fit anywhere and pinning them
# to the 48 GB half leaves them queueing behind a full partition. The registry's
# `min_vram_gb` is the field for this, and this check enforces it: it is what
# makes the wide constraint safe for qwen3-14b and gemma4-12b (48 GB) rather
# than a way to OOM them on a 3090.
#
# Checked here, before the page-cache warm-up: the warm-up costs ~20 min and
# reads the whole checkpoint, so discovering the mismatch after it is 20 wasted
# minutes plus a pointless hammering of the filer.
#
# Two details matter:
#
# `min_vram_gb` is written in spec-sheet GB (decimal), not GiB, so the byte
# count divides by 1e9 and not by 1024**3. Measured totals, 2026-09-17:
#
#   3090   24576 MiB    a5000  24564 MiB
#   a6000  49140 MiB    l40s   46068 MiB
#
# An l40s is sold as 48 GB and reports 45.0 GiB. Dividing by 1024**3 and asking
# for `>= 48` would therefore REFUSE qwen3-14b and gemma4-12b on the l40s nodes --
# the exact cards those models are meant to run on, and the only cards in
# `killable` that can. A fixed slack cannot paper over this either, because the
# nominal-to-reported gap is 0% on a 3090 and 6% on an l40s.
#
# The 10% is then for the driver/ECC reserve that makes torch's total_memory a
# little under the board total. It is nowhere near tight: the nearest refusal
# boundary is a 24 GB card reaching 27 GB against a 48 GB model.
python - "$MODEL" <<'PY' || exit 1
import sys

import torch

from graphtalk import models

spec = models.MODELS[sys.argv[1]]
props = torch.cuda.get_device_properties(0)
total_gb = props.total_memory / 1e9
if total_gb * 1.10 < spec.min_vram_gb:
  print(f"FATAL: {spec.key} needs ~{spec.min_vram_gb} GB of VRAM; this node's "
        f"{props.name} has {total_gb:.1f} GB.", file=sys.stderr)
  print("The --constraint in sweep.sbatch spans 24 GB and 48 GB cards. Narrow",
        file=sys.stderr)
  print("it for this model, e.g. --constraint='a6000|l40s'.", file=sys.stderr)
  sys.exit(1)
print(f"vram ok: {props.name} has {total_gb:.1f} GB, "
      f"{spec.key} needs ~{spec.min_vram_gb} GB")
PY

# Warm the page cache with one sequential pass over the checkpoint before the
# loader touches it.
#
# `safetensors` mmaps the file and reads tensor offsets in checkpoint order
# rather than file order, and those scattered reads are pathological over NFS:
# gemma4-e4b's 16 GB projected to a nine-hour load and the job died on its time
# limit before writing a row. Reading the blobs end to end first costs one
# sequential pass and turns every later mmap fault into a RAM hit -- loading then
# takes about two seconds.
#
# Read throughput varies by node far more than expected, so the warm-up is worth
# 20 min on a good node and much more on a bad one. Measured with 2 GiB of direct
# I/O, twice each: n-802 and n-805 served ~31 MB/s, while n-801 managed 12.4 MB/s
# idle and 3.4 MB/s under load, on a 195-day uptime. Both stalls in this project
# landed on n-801, which is why the default --exclude carries it.
#
# `cat` and not python: this has to happen before `import torch`, and the point
# is the kernel's cache, not anything in the process.
REPO="$(python -c "from graphtalk import models; print(models.MODELS['$MODEL'].repo_id)")"
BLOBS="${HF_HUB_CACHE}/models--${REPO//\//--}/blobs"
echo "warming page cache from $BLOBS"
WARM_START=$SECONDS
if find "$BLOBS" -type f -size +1M -exec cat {} + > /dev/null; then
  echo "page cache warmed in $(( (SECONDS - WARM_START) / 60 ))m $(( (SECONDS - WARM_START) % 60 ))s"
else
  # `find` needs list (r) permission on $BLOBS to enumerate it; actual
  # loading below only needs traverse (x) permission to open a file it
  # already knows the name of via the snapshots/ symlink, so a listing
  # failure here -- e.g. reading someone else's shared cache -- doesn't
  # mean loading will fail too. This step is a speed optimization only.
  echo "page cache warm-up skipped (couldn't list $BLOBS, likely a permissions gap on a shared cache); continuing without it"
fi

python scripts/run_sweep.py \
  --model "$MODEL" \
  --prompts "$PROMPTS" \
  --out "$OUT" \
  "${EXTRA[@]}"
