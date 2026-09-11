#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gpus=1
#SBATCH --constraint="a6000|l40s|h100"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Both validated families are ~16 GB checkpoints, so this is 20 min of warm-up
# plus ~25 min of generation on an uncontended node -- but the warm-up is
# bandwidth-bound and shares a filer with whatever else is running (see
# cluster/README.md, "Do not put several checkpoint warm-ups on one node at
# once"). --time is a ceiling and the allocation is released on exit, so the
# headroom costs nothing and a contended warm-up does not kill the job.
#SBATCH --time=08:00:00
#SBATCH --job-name=graphtalk_batchval
#SBATCH --output=out/slurm-%j.out

# Validates batched generation against single-stream for ONE model family, and
# exits non-zero if they disagree -- so a real sweep can chain on
# `--dependency=afterok:<this job>` rather than waiting on someone to read a log.
#
#   sbatch cluster/validate_batching.sbatch qwen3-8b [batch_size]
#
# Run it for BOTH families before trusting --batch-size anywhere. They need
# opposite padding sides (gemma defaults left, Qwen3 defaults right), so one
# passing says nothing about the other -- see graphtalk/hf_backend.py's
# generate_batch docstring and scripts/validate_batch_generation.py.
#
# Prompts come from scripts/validate_batch_generation.py --build-prompts, which
# runs on the login node first (it only reads tracked files, but keeping the
# split identical to the rest of the pipeline is what makes it reviewable).

set -euo pipefail

MODEL="${1:?usage: sbatch cluster/validate_batching.sbatch <model-key> [batch-size]}"
BATCH_SIZE="${2:-4}"
PROMPTS="${GRAPHTALK_PROMPTS:-batch_validation_prompts.jsonl}"

GRAPHTALK_ENV="${GRAPHTALK_ENV:-graphtalk}"
source /home/dcor/galbarak2/anaconda3/etc/profile.d/conda.sh
conda activate "/home/dcor/galbarak2/conda_envs/${GRAPHTALK_ENV}"
echo "env: $GRAPHTALK_ENV  model: $MODEL  batch-size: $BATCH_SIZE"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR"

if [[ ! -f "$PROMPTS" ]]; then
  echo "FATAL: '$PROMPTS' does not exist. Build it on the login node first:" >&2
  echo "  python scripts/validate_batch_generation.py --build-prompts $PROMPTS" >&2
  exit 1
fi

mkdir -p runs/archive out
nvidia-smi

# Same driver guard as cluster/sweep.sbatch: `device_map="auto"` falls back to
# the CPU with no error when no CUDA device is usable, and a validation run that
# silently generated on the host would still "pass" while proving nothing about
# the GPU code path this exists to check.
python - <<'PY' || exit 1
import sys
import torch

if torch.cuda.is_available():
  print(f"cuda ok: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
  sys.exit(0)
print(f"FATAL: torch {torch.__version__} sees no usable CUDA device on this node.",
      file=sys.stderr)
sys.exit(1)
PY

# Under runs/archive/ so these can never reach a frame: graphtalk/analysis.py
# excludes that directory wholesale, and these rows are a diagnostic, not sweep
# data. Removed first because run_sweep.py resumes by key -- a leftover file from
# an earlier attempt would be reported as "nothing to do" and the comparison
# would silently re-check stale text.
SINGLE="runs/archive/batchval-${MODEL}.single.jsonl"
BATCHED="runs/archive/batchval-${MODEL}.batched.jsonl"
rm -f "$SINGLE" "$BATCHED"

REPO="$(python -c "from graphtalk import models; print(models.MODELS['$MODEL'].repo_id)")"
BLOBS="${HF_HOME}/hub/models--${REPO//\//--}/blobs"
echo "warming page cache from $BLOBS"
WARM_START=$SECONDS
find "$BLOBS" -type f -size +1M -exec cat {} + > /dev/null
echo "page cache warmed in $(( (SECONDS - WARM_START) / 60 ))m $(( (SECONDS - WARM_START) % 60 ))s"

# Single-stream first: it is the reference the batched run has to reproduce, and
# it is the code path every row on disk so far was generated with.
echo "=== single-stream (--batch-size 1) ==="
python scripts/run_sweep.py --model "$MODEL" --prompts "$PROMPTS" \
    --out "$SINGLE" --batch-size 1

echo "=== batched (--batch-size $BATCH_SIZE) ==="
python scripts/run_sweep.py --model "$MODEL" --prompts "$PROMPTS" \
    --out "$BATCHED" --batch-size "$BATCH_SIZE"

# Runs here rather than back on the login node so the exit status is the job's
# own -- that is what makes --dependency=afterok a real gate on the result.
echo "=== comparison ==="
python scripts/validate_batch_generation.py \
    --prompts "$PROMPTS" \
    --compare "$SINGLE" "$BATCHED"
