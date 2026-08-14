#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gpus=1
#SBATCH --constraint="a6000|l40s|h100"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=graphtalk_sweep
#SBATCH --output=out/slurm-%j.out
#SBATCH --signal=USR1@120

# One model over one prompt file. Submit once per model:
#   sbatch cluster/sweep.sbatch gemma4-12b
#
# `killable` is preemptible, so this job can be stopped and requeued at any
# moment. run_sweep.py appends to its output and skips work already present, so a
# requeue resumes rather than restarting -- which is why --out must be a stable
# path across requeues and not something with a job id in it.

set -euo pipefail

MODEL="${1:?usage: sbatch cluster/sweep.sbatch <model-key>}"

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

python scripts/run_sweep.py \
  --model "$MODEL" \
  --prompts prompts.jsonl \
  --out "runs/${MODEL}.jsonl"
