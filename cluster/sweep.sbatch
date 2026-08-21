#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gpus=1
#SBATCH --constraint="a6000|l40s|h100"
#SBATCH --cpus-per-task=8
# Sized to hold the largest checkpoint in the page cache the warm-up below fills:
# 29.6 GB for Qwen3-14B, plus the loader's buffers. The weights land on the GPU
# rather than staying resident here, so the headroom above 30 GB is comfortable.
#
# Not larger. An earlier 96G was guesswork and cost real queue time: the nodes
# with a spare GPU had 40-47 GB of RAM free while the ones with RAM free had no
# GPU, so nothing could satisfy gpu=1 + mem=96G and all twelve jobs sat on
# Reason=Resources. 64G is what the budget measurement actually ran at.
#SBATCH --mem=64G
# The partition ceiling. A full model needs roughly 45 h at the measured
# zero_shot budget, so no single job finishes one: submit a chain and let the
# resume logic below carry the work across them.
#SBATCH --time=24:00:00
#SBATCH --job-name=graphtalk_sweep
#SBATCH --output=out/slurm-%j.out
#SBATCH --signal=USR1@120

# One model over one prompt file. Submit once per model:
#   sbatch cluster/sweep.sbatch gemma4-12b
#
# An optional second argument runs a smoke test of that many generations:
#   sbatch --time=00:30:00 cluster/sweep.sbatch gemma4-e4b 20
#
# The h100 cards are NOT reachable from this partition: n-102 and t-100 live in
# `gpu-h100-killable`, so the `h100` term in the --constraint below can never
# match while --partition is `killable`. To use them, override the partition:
#   sbatch --partition=gpu-h100-killable --constraint=h100 cluster/sweep.sbatch qwen3-14b
#
# `killable` is preemptible, so this job can be stopped and requeued at any
# moment. run_sweep.py appends to its output and skips work already present, so a
# requeue resumes rather than restarting -- which is why --out must be a stable
# path across requeues and not something with a job id in it.

set -euo pipefail

MODEL="${1:?usage: sbatch cluster/sweep.sbatch <model-key> [limit]}"
LIMIT="${2:-}"

# The prompt file is overridable so an arm can run a subset. The thinking arm
# uses a zero_shot-only file whose 1260 rows are byte-identical to the zero_shot
# half of the main one, so the two arms are comparable row for row.
PROMPTS="${GRAPHTALK_PROMPTS:-prompts.jsonl}"

# A smoke test writes to its own file. Sharing the sweep's output would make its
# rows count as done, so a smoke run that revealed a broken chat template would
# leave twenty bad generations that the full sweep then skips.
OUT="runs/${MODEL}.jsonl"
EXTRA=()
if [[ -n "$LIMIT" ]]; then
  OUT="runs/smoke-${MODEL}.jsonl"
  EXTRA=(--limit "$LIMIT")
fi

# Submitting as a job array shards one model across that many GPUs:
#   sbatch --array=0-7 cluster/sweep.sbatch qwen3-8b-think
#
# Each shard writes its own file. score_sweep.py groups by the `model` field in
# each row rather than by filename, so the shards need no reassembly -- point it
# at runs/*.jsonl and they are pooled automatically.
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
if [[ "$NSHARDS" -gt 1 ]]; then
  OUT="${OUT%.jsonl}.shard${SHARD}of${NSHARDS}.jsonl"
  EXTRA+=(--shard "$SHARD" --num-shards "$NSHARDS")
fi

source /home/dcor/galbarak2/anaconda3/etc/profile.d/conda.sh
conda activate /home/dcor/galbarak2/conda_envs/graphtalk

# Models are pre-fetched on the login node; compute nodes may have no outbound
# network, and a job that tries to download blocks until it times out.
export HF_HOME=/home/dcor/galbarak2/hf_cache
export HF_HUB_OFFLINE=1
# Reclaims fragmented VRAM, which is what pushed the SlidesGen 12B run over the
# edge on a 48 GB card.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR"

mkdir -p runs out
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
# The cause is that `killable` spans two driver generations: n-802, n-803 and
# n-804 are on 535.x (CUDA 12.2) while n-602, n-805 and t-806 are on 580+/595.
# The env's torch is a cu130 build and needs 580 or newer. Landing on an old node
# is luck of the scheduler, which is why the first smoke test passed.
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
print("exclude the 535.x nodes (n-802, n-803, n-804).", file=sys.stderr)
sys.exit(1)
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
# landed on n-801. Pass `--exclude=n-801` until someone reboots it.
#
# `cat` and not python: this has to happen before `import torch`, and the point
# is the kernel's cache, not anything in the process.
REPO="$(python -c "from graphtalk import models; print(models.MODELS['$MODEL'].repo_id)")"
BLOBS="${HF_HOME}/hub/models--${REPO//\//--}/blobs"
echo "warming page cache from $BLOBS"
WARM_START=$SECONDS
find "$BLOBS" -type f -size +1M -exec cat {} + > /dev/null
echo "page cache warmed in $(( (SECONDS - WARM_START) / 60 ))m $(( (SECONDS - WARM_START) % 60 ))s"

python scripts/run_sweep.py \
  --model "$MODEL" \
  --prompts "$PROMPTS" \
  --out "$OUT" \
  "${EXTRA[@]}"
