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
#SBATCH --time=12:00:00
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

# Warm the page cache with one sequential pass over the checkpoint before the
# loader touches it.
#
# The lab netapp serves this tree at ~29 MB/s sequential, and `safetensors`
# mmaps the file and reads tensor offsets in checkpoint order rather than in
# file order. Those scattered reads measured ~3.6 MB/s over NFS, which put
# gemma4-e4b's 16 GB at a nine-hour load and made the job die on its time limit
# before writing a single row. Reading the blobs end to end first costs one
# sequential pass and turns every later mmap fault into a RAM hit.
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
