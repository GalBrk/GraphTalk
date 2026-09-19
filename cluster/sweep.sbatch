#!/bin/bash
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gpus=1
# Widened from `a6000|l40s|h100` on 2026-09-17. That set had stopped being a
# filter and become a bottleneck: every a6000 and l40s GPU in `killable` was
# allocated, while 25 GPUs sat free on the 3090 and a5000 nodes the constraint
# excluded -- for a sweep whose largest live model is 4B (~8 GB of bf16
# weights). Driver versions were measured across the whole partition before
# widening rather than assumed (the table in cluster/README.md); all six
# reachable 3090 nodes are on 595.84, well past the 580+ the default cu130 env
# needs, so this costs no env change.
#
# Every card named here has at least 24 GB, which covers the registry up to
# `min_vram_gb=24`. The 48 GB models (qwen3-14b, gemma4-12b) are NOT safe on
# the 24 GB half of that pool, so the VRAM guard further down refuses them on a
# too-small card rather than letting them OOM an hour in -- that guard, not
# this line, is what keeps the wider pool honest for the big models.
#
# The 2080 nodes are deliberately still out even though 30 of them are idle:
# sm_75 has no bf16, so they need an fp16 path first, not just a wider
# constraint.
#SBATCH --constraint="a6000|l40s|a5000|geforce_rtx_3090|h100"
# The 535.x-driver nodes the default cu130 env cannot use, plus n-801, whose
# read throughput collapses to 3.4 MB/s under load (both stalls in this project
# landed there; its driver is fine at 580.126.09, it is the I/O that is not).
# Measured 2026-09-17. Note n-501 is an a5000, not an l40s, and only became
# reachable at all when `a5000` joined the constraint above.
#
# An --exclude on the sbatch command line REPLACES this list rather than adding
# to it, so `--exclude=n-801` silently re-admits all four 535.x nodes. To use
# those nodes on purpose, pass GRAPHTALK_ENV=graphtalk-cu126 instead, which
# runs on both driver generations.
#SBATCH --exclude=n-501,n-801,n-802,n-803,n-804
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
# The default --constraint below is the 48 GB tier (a6000|l40s|h100), sized for
# the largest model in graphtalk/models.py (min_vram_gb=48). A model with a
# smaller min_vram_gb can also run on the Ampere 24 GB tier -- same generation
# as a6000, bf16 tensor cores, just less VRAM -- which is a much bigger, often
# much less contended pool (`killable` has 2 a6000 nodes and up to 6 l40s nodes
# vs. 3 a5000 + 7 geforce_rtx_3090 nodes). Override --constraint (and size
# --mem down to the checkpoint, not the 64G default) for such a model:
#   sbatch --constraint="a5000|geforce_rtx_3090|a6000|l40s|h100" --mem=16G \
#       cluster/sweep.sbatch qwen3-1.7b
# Deliberately excludes geforce_rtx_2080 and the DGX v100/quadro nodes: Turing
# and Volta lack bf16 tensor cores, so `device_map="auto"` would place all or
# part of the model on CPU there -- silently, per the driver check below, not
# with an error. cluster/run_size_sweep.sh applies this override already.
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
# half of the main one, so the two arms are comparable row for row. Also how a
# GoT-named run points at prompts_got.jsonl instead -- see
# cluster/submit_sweep.sh, which builds it before ever setting this.
PROMPTS="${GRAPHTALK_PROMPTS:-prompts.jsonl}"
if [[ ! -f "$PROMPTS" ]]; then
  # Fails already without this -- the REMAINING check below opens the file --
  # but with a bare Python traceback instead of a message that says what to
  # do about it, and this way it fails before conda even activates.
  echo "FATAL: prompt file '$PROMPTS' does not exist. Build it first" >&2
  echo "(e.g. cluster/submit_sweep.sh --node-naming got ...) or point" >&2
  echo "GRAPHTALK_PROMPTS at an existing file." >&2
  exit 1
fi

# Absolute path for runs directory to ensure correct permissions and storage location
RUNS_DIR="/home/dcor/avivyossef/inbal/GraphTalk/runs"

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
  # Straight into the archive: a smoke run is never part of the sweep, and
  # runs/archive/ is excluded by directory rather than by the `smoke-` prefix.
  mkdir -p "${RUNS_DIR}/archive"
  OUT="${RUNS_DIR}/archive/smoke-${MODEL}.jsonl"
  EXTRA=(--limit "$LIMIT")
fi

# An optional run tag gives a partial regeneration its own output files, so the
# operation is auditable and reversible instead of appending into the arm:
#
#   GRAPHTALK_RUN_TAG=rerun sbatch --array=0-3 ... <model>
#     -> runs/<model>.rerun.shard0of4.jsonl
#
# score_sweep.py pools by each row's `model` field rather than by filename, so a
# tagged file joins its arm with no reassembly.
#
# `redo` is still refused. Exclusion is now by directory (runs/archive/) rather
# than by filename, so the old footgun is mostly gone -- but ".redo.shard" is kept
# as a safety net in analysis.is_excluded for paths predating that move, and a file
# named that way would still be dropped from every frame in silence.
RUN_TAG="${GRAPHTALK_RUN_TAG:-}"
if [[ -n "$RUN_TAG" ]]; then
  if [[ "$RUN_TAG" == *redo* ]]; then
    echo "FATAL: run tag '$RUN_TAG' contains 'redo'. analysis.py excludes" >&2
    echo "'.redo.shard' paths, so these rows would vanish from every frame." >&2
    exit 1
  fi
  OUT="${OUT%.jsonl}.${RUN_TAG}.jsonl"
fi

# Submitting as a job array shards one model across that many GPUs:
#   sbatch --array=0-7 cluster/sweep.sbatch qwen3-8b-think
#
# Each shard writes its own file. score_sweep.py groups by the `model` field in
# each row rather than by filename, so the shards need no reassembly -- point it
# at runs/*.jsonl and they are pooled automatically.
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

# Track 2.3: batched generation (graphtalk.hf_backend.generate_batch), opt-in
# and unset by default so every existing invocation keeps using the exact
# single-stream code path every row on disk so far was generated with.
# NOT YET VALIDATED against a GPU -- see generate_batch's docstring for the
# reference check (analysis/budget-gemma4-e4b.jsonl /
# budget-qwen3-8b.jsonl) to run before using this for a real sweep, not a
# smoke test.
BATCH_SIZE="${GRAPHTALK_BATCH_SIZE:-}"
if [[ -n "$BATCH_SIZE" ]]; then
  EXTRA+=(--batch-size "$BATCH_SIZE")
fi

# Override the spec's max_new_tokens budget, e.g. for a prompt file that mixes
# in edge_count at a density/size where the default 2048 truncates it:
#   GRAPHTALK_MAX_NEW_TOKENS=8192 sbatch cluster/sweep.sbatch <model>
# Applies to every row in this job, not just the ones that need it -- harmless
# for short-answer tasks since generation still stops at EOS, not at the cap.
MAX_NEW_TOKENS="${GRAPHTALK_MAX_NEW_TOKENS:-}"
if [[ -n "$MAX_NEW_TOKENS" ]]; then
  EXTRA+=(--max-new-tokens "$MAX_NEW_TOKENS")
fi

# Which env, and therefore which half of the cluster is reachable.
#
# The default is the cu130 build the first sweep was generated with. It needs
# driver 580+, which n-602, n-805 and t-806 have and n-802, n-803 and n-804 do
# not -- so it can only ever use half the partition.
#
# graphtalk-cu126 is the same torch 2.13.0 built against CUDA 12.6, and CUDA
# minor-version compatibility means it runs on both driver generations. Verified
# on all three 535.183.01 nodes: cuda.is_available() true, L40S detected at
# sm_89, bf16 matmul finite. Using it doubles the reachable pool to 48 GPUs.
#
#   sbatch --export=ALL,GRAPHTALK_ENV=graphtalk-cu126 cluster/sweep.sbatch <model>
#
# It cannot drive a B200: the cu126 build stops at sm_90 and Blackwell is sm_100.
# Irrelevant on `killable`, which has no B200, but it rules the env out if this
# ever moves to gpu-b200.
GRAPHTALK_ENV="${GRAPHTALK_ENV:-graphtalk}"
set +u
source /home/dcor/galbarak2/anaconda3/etc/profile.d/conda.sh
conda activate "/home/dcor/galbarak2/conda_envs/${GRAPHTALK_ENV}"
set -u
echo "env: $GRAPHTALK_ENV"

# Models are pre-fetched on the login node; compute nodes may have no outbound
# network, and a job that tries to download blocks until it times out.
# Weights come from the lab-shared cache by default: it holds Qwen3-0.6B/1.7B/
# 3.5-2B/8B/14B and gemma-4-E4B/12B, so nothing needs re-downloading.
#
# HF_HOME and HF_HUB_CACHE are set SEPARATELY and that split is load-bearing.
# Pointing HF_HOME at the shared cache fails: huggingface_hub writes a `token`
# file at HF_HOME's root, and the shared tree is read-only to everyone but its
# owner, so every load dies with `PermissionError at .../token`. HF_HOME must
# therefore be somewhere writable, with only HF_HUB_CACHE aimed at the shared
# blobs. Benign `Ignoring corrupted tree cache file ... Permission denied`
# warnings are expected from the same read-only-ness and are not an error.
#
# Override either for a checkout with its own weights:
#   GRAPHTALK_HF_CACHE=/path/to/hub sbatch ... <model>
#
# Offline is NOT negotiable: the GPU nodes have no route to huggingface.co, so a
# cache miss fails on the node after queueing rather than at submit time.
export HF_HOME="${GRAPHTALK_HF_HOME:-/home/dcor/avivyossef/inbal/GraphTalk/.hf_home}"
export HF_HUB_CACHE="${GRAPHTALK_HF_CACHE:-/home/dcor/galbarak2/hf_cache/hub}"
mkdir -p "$HF_HOME"
export HF_HUB_OFFLINE=1
# Reclaims fragmented VRAM, which is what pushed the SlidesGen 12B run over the
# edge on a 48 GB card.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR"

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
print("exclude the 535.x nodes (n-501, n-802, n-803, n-804).", file=sys.stderr)
sys.exit(1)
PY

# Refuse to run a model that does not fit the card we landed on.
#
# The --constraint above spans 24 GB cards (3090, a5000) and 48 GB ones (a6000,
# l40s), because the small models this sweep actually runs fit anywhere and
# pinning them to the 48 GB half is what left them queueing behind a full
# partition. `min_vram_gb` has been in the registry since the beginning,
# documented as the field used "to pick a --constraint rather than discovering
# an OOM an hour into the queue" -- but nothing read it, so nothing enforced it.
# This is that enforcement, and it is what makes the wider constraint safe for
# qwen3-14b and gemma4-12b (48 GB) rather than a way to OOM them on a 3090.
#
# Checked here, before the page-cache warm-up: the warm-up costs ~20 min and
# reads the whole checkpoint, so discovering the mismatch after it is 20 wasted
# minutes plus a pointless hammering of the filer.
#
# Two details that a first cut of this check got wrong, both worth keeping:
#
# `min_vram_gb` is written in spec-sheet GB (decimal), not GiB, so the byte
# count divides by 1e9 and not by 1024**3. Measured totals, 2026-09-17:
#
#   3090   24576 MiB    a5000  24564 MiB
#   a6000  49140 MiB    l40s   46068 MiB
#
# An l40s is sold as 48 GB and reports 45.0 GiB. Dividing by 1024**3 and asking
# for `>= 48` therefore REFUSES qwen3-14b and gemma4-12b on the l40s nodes --
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
# landed on n-801. Pass `--exclude=n-801` until someone reboots it.
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
