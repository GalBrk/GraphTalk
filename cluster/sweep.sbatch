#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gpus=1
#SBATCH --constraint="a6000|l40s|h100"
#SBATCH --cpus-per-task=8
# 96G rather than 48G: the weights are read through the page cache by the warm-up
# below, and the cache has to hold the whole checkpoint (29.6 GB for Qwen3-14B)
# alongside the loader's own buffers. At 48G the cache is evicted while it is
# still being filled and the random-access reads go back to the netapp.
#SBATCH --mem=96G
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

# A smoke test writes to its own file. Sharing the sweep's output would make its
# rows count as done, so a smoke run that revealed a broken chat template would
# leave twenty bad generations that the full sweep then skips.
OUT="runs/${MODEL}.jsonl"
EXTRA=()
if [[ -n "$LIMIT" ]]; then
  OUT="runs/smoke-${MODEL}.jsonl"
  EXTRA=(--limit "$LIMIT")
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
REMAINING="$(python - "$OUT" <<'PY'
import json, os, sys

out = sys.argv[1]
with open("prompts.jsonl") as handle:
  todo = {(r["instance_id"], r["condition"], r["style"])
          for r in map(json.loads, filter(str.strip, handle))}
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
  --prompts prompts.jsonl \
  --out "$OUT" \
  "${EXTRA[@]}"
