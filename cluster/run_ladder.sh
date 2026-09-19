#!/bin/bash
# Submits the two cheap, gating passes of the ladder design:
#
#   probe   -- the graph-free retrieval probe, which locates each model's
#              READING LIMIT. This gates everything: 11 of the 18 ladder rungs
#              sit past the 1,505-token clean-reading mark, and until the limit
#              is located per model there is no way to tell which rungs test
#              primers and which test eyesight.
#   ladder  -- `none` only, every rung, which locates each model's INFORMATIVE
#              BAND (ceiling / informative / floor).
#
# The two are independent and submit together: banding does not depend on the
# reading limit, which is applied later as a second filter in
# scripts/analyze_ladder.py. The rewiring experiment itself is deliberately NOT
# here -- it should only be spent on rungs that both passes have shown to be
# valid for that model.
#
#   cd /home/dcor/avivyossef/inbal/GraphTalk
#   cluster/run_ladder.sh                 # both stages, default models
#   cluster/run_ladder.sh --dry-run       # print, build nothing, submit nothing
#   STAGES=ladder MODELS=qwen3-1.7b cluster/run_ladder.sh
#
# OFFLINE CHECKPOINTS. sweep.sbatch exports HF_HUB_OFFLINE=1 and points
# HF_HUB_CACHE at the lab-shared cache, which holds all seven checkpoints -- so
# no model here needs downloading. A model NOT in that cache fails on the node
# after queueing rather than at submit time, so check before adding one.

set -euo pipefail

STAGES="${STAGES:-probe ladder}"
# All seven checkpoints in the lab-shared cache, both arms. Nothing needs
# downloading -- see cluster/sweep.sbatch's HF_HUB_CACHE.
MODELS="${MODELS:-qwen3-0.6b qwen3-1.7b qwen35-2b qwen3-8b gemma4-e4b gemma4-12b qwen3-14b \
qwen3-0.6b-think qwen3-1.7b-think qwen35-2b-think qwen3-8b-think gemma4-e4b-think \
gemma4-12b-think qwen3-14b-think}"
# Base graphs per rung for the ladder screen. 50 gives each rung a tight enough
# accuracy estimate to band it; banding needs far less precision than the
# primer test that follows.
LADDER_COUNT="${LADDER_COUNT:-50}"
# Retrieval-probe lengths. The existing probe measured 160/320/640 statements
# (1,505 / 3,105 / 6,305 tokens) and found the threshold somewhere in the first
# gap and the middle-collapse in the second. These intermediate values bracket
# both, which is the whole point of re-running it.
PROBE_STATEMENTS="${PROBE_STATEMENTS:-160 220 260 320 420 520 640}"
PROBE_COUNT="${PROBE_COUNT:-25}"
# Documented bad-driver / slow nodes (cluster/README.md). Every other submission
# path in this repo passes these; jobs that land there die on the CUDA check.
EXCLUDE="${EXCLUDE:-n-501,n-801,n-802,n-803,n-804}"

DRY_RUN="${DRY_RUN:-}"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "FATAL: unknown argument '$arg' (only --dry-run is accepted)" >&2; exit 1 ;;
  esac
done

if [[ ! -f cluster/sweep.sbatch ]]; then
  echo "FATAL: run this from the repo root (cluster/sweep.sbatch not found in $(pwd))." >&2
  exit 1
fi

if [[ -n "${GRAPHTALK_PYTHON:-}" ]]; then
  PYTHON="$GRAPHTALK_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
elif [[ -x /home/dcor/galbarak2/conda_envs/graphtalk/bin/python ]]; then
  PYTHON="/home/dcor/galbarak2/conda_envs/graphtalk/bin/python"
else
  echo "FATAL: no interpreter found. Set GRAPHTALK_PYTHON." >&2
  exit 1
fi

# Per-model GPU sizing. sweep.sbatch's header targets the 48 GB tier sized for
# Qwen3-14B; confining a 0.6B model there costs real queue time, because the
# 48 GB pool is small and heavily allocated while the Ampere 24 GB tier (a5000,
# geforce_rtx_3090 -- same generation, bf16 tensor cores) is much larger.
# Deliberately NOT geforce_rtx_2080 or the DGX v100/quadro nodes: Turing and
# Volta lack bf16, and `device_map="auto"` silently offloads to CPU rather than
# erroring, so a too-old card fails as "mysteriously slow", not loudly.
# --mem follows the checkpoint, not the 64G default sized to page-cache 28 GB.
tier_for () {   # tier_for <min_vram_gb> -> "<constraint>|<mem>"
  local vram="$1"
  if   (( vram <= 12 )); then echo "a5000|geforce_rtx_3090|a6000|l40s|h100;16G"
  elif (( vram <= 24 )); then echo "a6000|l40s|h100;32G"
  else                        echo "a6000|l40s|h100;64G"
  fi
}

submit () {   # submit <model> <prompts> <tag>
  local model="$1" prompts="$2" tag="$3"
  local vram tier constraint mem
  vram=$(PYTHONPATH=. "$PYTHON" -c \
    "from graphtalk import models; print(models.MODELS['$model'].min_vram_gb)" 2>/dev/null) || {
      echo "  SKIP $model -- not in graphtalk/models.py" >&2; return 0; }
  tier=$(tier_for "$vram"); constraint="${tier%;*}"; mem="${tier#*;}"
  echo "  submit $model (${vram}GB -> $mem)  <- $prompts  (tag $tag)"
  [[ -n "$DRY_RUN" ]] && return 0
  GRAPHTALK_PROMPTS="$prompts" GRAPHTALK_RUN_TAG="$tag" \
    sbatch --job-name="${model}_${tag}" \
      --constraint="$constraint" --mem="$mem" \
      --exclude="$EXCLUDE" \
      cluster/sweep.sbatch "$model"
}

for STAGE in $STAGES; do
  case "$STAGE" in
    probe)
      PROMPTS="prompts.retrieval_locate.jsonl"
      TAG="retrieval_locate"
      if [[ -f "$PROMPTS" ]]; then
        echo "reusing existing $PROMPTS"
      else
        echo "building $PROMPTS (statements: $PROBE_STATEMENTS, count $PROBE_COUNT)"
        [[ -z "$DRY_RUN" ]] && PYTHONPATH=. "$PYTHON" scripts/build_retrieval_probe.py \
            --statements $PROBE_STATEMENTS --positions 0.1 0.5 0.9 \
            --magnitudes small large --count "$PROBE_COUNT" --out "$PROMPTS"
      fi
      ;;
    ladder)
      PROMPTS="prompts.ladder_screen.jsonl"
      TAG="ladder_screen"
      if [[ -f "$PROMPTS" ]]; then
        echo "reusing existing $PROMPTS"
      else
        echo "building $PROMPTS (18 rungs x $LADDER_COUNT graphs, condition=none)"
        [[ -z "$DRY_RUN" ]] && PYTHONPATH=. "$PYTHON" scripts/build_ladder.py \
            --stage screen --count "$LADDER_COUNT" --out "$PROMPTS"
      fi
      ;;
    rewire)
      # Stage 3. Unlike the two gating stages this is NOT run on every rung:
      # REWIRE_PROMPTS must already have been built with build_ladder.py
      # --stage rewire --rungs <only the rungs that cleared BOTH gates for
      # these models>, read off analysis/ladder_matrix.limited.csv. Running it
      # on a rung the model ceilings or cannot read measures nothing.
      PROMPTS="${REWIRE_PROMPTS:?set REWIRE_PROMPTS to a built rewire file}"
      TAG="${REWIRE_TAG:-rewire}"
      [[ -f "$PROMPTS" ]] || { echo "FATAL: $PROMPTS not built" >&2; exit 1; }
      echo "using $PROMPTS ($(wc -l < "$PROMPTS") rows)"
      ;;
    *) echo "FATAL: unknown stage '$STAGE' (probe|ladder|rewire)" >&2; exit 1 ;;
  esac
  for MODEL in $MODELS; do
    submit "$MODEL" "$PROMPTS" "$TAG"
  done
done

echo
echo "when the jobs finish:"
echo "  PYTHONPATH=. $PYTHON scripts/analyze_ladder.py \\"
echo "      --responses 'runs/*.ladder_screen.jsonl' --out analysis/ladder_matrix.csv"
echo "  # then re-run it with the reading limits the probe gives you:"
echo "  #   --reading-limits qwen3-1.7b=2600 qwen3-1.7b-think=2600"
