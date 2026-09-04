#!/bin/bash
# Wraps `sbatch` to add one flag for the GoT node-naming scheme -- builds
# prompts_got.jsonl (login-node work: network, no torch, no GPU) if it
# doesn't already exist, then submits exactly as `sbatch` would.
#
# `build_prompts.py` fetches dataset rows over plain `urllib` and has to run
# on the login node -- compute nodes have no outbound network, which is why
# `cluster/sweep.sbatch` sets HF_HUB_OFFLINE=1. `sbatch` itself only queues a
# job; the job body runs later on a compute node. So building the prompt
# file has to happen here, at submission time, not inside sweep.sbatch.
#
# Original scheme, unchanged from today:
#   cluster/submit_sweep.sh cluster/sweep.sbatch gemma4-12b
#
# GoT scheme, one flag:
#   cluster/submit_sweep.sh --node-naming got cluster/sweep.sbatch gemma4-12b
#
# Every other sbatch flag/positional (--array, --exclude, --mem, the model
# key, the smoke-test limit) passes through untouched, in whatever position
# it's given -- only --node-naming and --dry-run are consumed here:
#
#   cluster/submit_sweep.sh --node-naming got --array=0-7 \
#       cluster/sweep.sbatch qwen3-8b-think
#
# --dry-run prints what would run instead of building anything or calling
# sbatch, for checking this script's own logic without cluster access.

set -euo pipefail

NODE_NAMING="integer"
DRY_RUN=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-naming) NODE_NAMING="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

case "$NODE_NAMING" in
  integer) ;;
  got)
    # Matches prompts.jsonl's own generation (`--count 30`, the proposal's
    # budget, every other default) -- the two schemes need identical
    # instances and conditions to be comparable at all.
    PROMPTS_FILE="prompts_got.jsonl"
    if [[ ! -f "$PROMPTS_FILE" ]]; then
      echo "building $PROMPTS_FILE (login node, --node-naming got)"
      if [[ -z "$DRY_RUN" ]]; then
        PYTHONPATH=. .venv/bin/python scripts/build_prompts.py --count 30 \
            --node-naming got --out "$PROMPTS_FILE"
      fi
    else
      echo "reusing existing $PROMPTS_FILE"
    fi
    export GRAPHTALK_PROMPTS="$PROMPTS_FILE"
    export GRAPHTALK_RUN_TAG="got"
    ;;
  *)
    echo "FATAL: unknown --node-naming '$NODE_NAMING'; known: integer, got" >&2
    exit 1
    ;;
esac

if [[ -n "$DRY_RUN" ]]; then
  ENV_PREFIX=""
  if [[ "$NODE_NAMING" == "got" ]]; then
    ENV_PREFIX="GRAPHTALK_PROMPTS=$GRAPHTALK_PROMPTS GRAPHTALK_RUN_TAG=$GRAPHTALK_RUN_TAG "
  fi
  echo "would run: ${ENV_PREFIX}sbatch ${ARGS[*]}"
else
  exec sbatch "${ARGS[@]}"
fi
