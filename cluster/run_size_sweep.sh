#!/bin/bash
# Automates the qwen3-1.7b node-size sweep: builds one prompt file per
# (task set, node size) (scripts/build_prompts.py --graph-source diverse
# --node-count) and submits both arms (plain + thinking) via
# cluster/sweep.sbatch for each.
#
# Filenames are tagged by task so rerunning with a different TASKS value
# never overwrites another task's prompts or mixes silently into another
# task's ambiguous output name: prompts_<tasks>_n<size>.jsonl and
# runs/<model>.<tasks>_n<size>.jsonl (<tasks> is TASKS with spaces turned
# into underscores, e.g. "node_degree" or "node_degree_edge_count" for a
# multi-task TASKS value). Primer condition never needs its own file --
# every condition is already a column within one file, for every task and
# model; splitting per condition would only add file-juggling with no
# benefit, since every script downstream already groups by the `condition`
# column.
#
# Must be run from the repo root on the cluster -- every path below is
# relative, and sbatch's own SLURM_SUBMIT_DIR is wherever it's invoked
# from (cluster/sweep.sbatch does `cd "$SLURM_SUBMIT_DIR"`), so results
# always land under this checkout:
#
#   cd /home/dcor/avivyossef/inbal/GraphTalk
#   cluster/run_size_sweep.sh
#
# Override sizes/tasks/pool size/models via env vars:
#
#   SIZES="20 40 80 160" TASKS="node_degree edge_count" COUNT=30 \
#       cluster/run_size_sweep.sh
#
# Preview without building or submitting anything:
#
#   cluster/run_size_sweep.sh --dry-run

set -euo pipefail

SIZES="${SIZES:-20 40 80}"
TASKS="${TASKS:-node_degree}"
COUNT="${COUNT:-30}"
MODELS="${MODELS:-qwen3-1.7b qwen3-1.7b-think}"
# n-801/802/803/804 are the documented bad-driver/slow nodes (cluster/README.md
# "Half the partition has a driver this torch build cannot use" / "n-801 is
# slow"); every other submission path in this repo passes --exclude for them,
# but this script didn't, so jobs landed there and failed immediately after the
# cu130 CUDA-availability check. n-501 (RTX A5000, in the 24 GB tier this
# script's own --constraint widens into) joined the list 2026-09-08: same
# 535.x-driver failure shape, just not yet surveyed when the table above was
# written.
EXCLUDE="${EXCLUDE:-n-501,n-801,n-802,n-803,n-804}"

DRY_RUN=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "FATAL: unknown argument '$arg' (only --dry-run is accepted)" >&2
        exit 1 ;;
  esac
done

# Refuse to run from the wrong place: everything below is a relative path,
# so this must be the repo root, not just any directory.
if [[ ! -f cluster/sweep.sbatch ]]; then
  echo "FATAL: run this from the repo root (cluster/sweep.sbatch not found" >&2
  echo "in the current directory: $(pwd))." >&2
  exit 1
fi

# Same interpreter fallback as cluster/submit_sweep.sh: .venv on a laptop,
# the lab conda env on this cluster (no .venv here, 6 GB home quota).
if [[ -n "${GRAPHTALK_PYTHON:-}" ]]; then
  PYTHON="$GRAPHTALK_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
elif [[ -x /home/dcor/galbarak2/conda_envs/graphtalk/bin/python ]]; then
  PYTHON="/home/dcor/galbarak2/conda_envs/graphtalk/bin/python"
else
  echo "FATAL: no interpreter found (.venv/bin/python, the cluster conda env)." >&2
  echo "Set GRAPHTALK_PYTHON to the python that has this package installed." >&2
  exit 1
fi

# TASKS is space-separated (argparse nargs="+"); underscores in the
# filename tag so multiple tasks stay one valid, readable path component
# instead of literal spaces.
TASK_TAG="${TASKS// /_}"

for N in $SIZES; do
  PROMPTS="prompts_${TASK_TAG}_n${N}.jsonl"
  RUN_TAG="${TASK_TAG}_n${N}"
  if [[ -f "$PROMPTS" ]]; then
    echo "reusing existing $PROMPTS"
  else
    echo "building $PROMPTS (--node-count $N --tasks $TASKS --count $COUNT)"
    if [[ -z "$DRY_RUN" ]]; then
      PYTHONPATH=. "$PYTHON" scripts/build_prompts.py --graph-source diverse \
          --node-count "$N" --tasks $TASKS --count "$COUNT" \
          --model qwen3-1.7b --out "$PROMPTS"
    fi
  fi

  for MODEL in $MODELS; do
    echo "submitting $MODEL on $PROMPTS (tag $RUN_TAG)"
    if [[ -z "$DRY_RUN" ]]; then
      # sweep.sbatch's own header targets the 48 GB tier (a6000|l40s|h100),
      # sized for the largest model in graphtalk/models.py (qwen3-14b,
      # min_vram_gb=48). The default MODELS here is qwen3-1.7b(-think),
      # min_vram_gb=8 -- an 8 GB model has no business being confined to
      # 48 GB cards, and as of 2026-09-08 every l40s in `killable` is fully
      # allocated (scontrol: gres/gpu=8/8 on n-801..805,t-806) while these
      # jobs sit `Priority`-pending days out on an 8-node pool.
      #
      # Widen to every bf16-capable card with enough VRAM for an 8 GB
      # model, i.e. add the Ampere 24 GB tier (a5000, geforce_rtx_3090) --
      # same generation as a6000, comfortable headroom over 8 GB. Deliberately
      # NOT geforce_rtx_2080 (Turing, no bf16 tensor cores, and only 8-11 GB
      # with zero headroom) or the DGX v100/quadro nodes (Volta/Turing, same
      # bf16 gap) -- `device_map="auto"` silently offloads to CPU rather than
      # erroring on a card that can't fit the model, so a too-small or too-old
      # card fails as "mysteriously slow", not loudly (see sweep.sbatch's
      # driver check for the same failure shape).
      #
      # --mem follows: sweep.sbatch's 64G default is sized to page-cache a
      # 28 GB checkpoint (qwen3-14b) before the warm-up; Qwen3-1.7B's is 3.9
      # GB on disk (.cache/hub/models--Qwen--Qwen3-1.7B/blobs), so 16G is
      # already 4x headroom and leaves far more nodes able to satisfy
      # gres/gpu=1 + mem simultaneously -- the exact `Reason=Resources`
      # trap cluster/README.md already documents for an oversized request.
      # A descriptive --job-name means a job that dies before writing a row
      # (e.g. landing on an excluded-but-not-yet-known-bad node) still shows
      # up in squeue/sacct as e.g. qwen3-1.7b_node_degree_n40 instead of the
      # generic graphtalk_sweep -- traceable without cross-referencing submit
      # timestamps against sacct history after the fact.
      GRAPHTALK_PROMPTS="$PROMPTS" GRAPHTALK_RUN_TAG="$RUN_TAG" \
        sbatch --job-name="${MODEL}_${RUN_TAG}" \
          --constraint="a5000|geforce_rtx_3090|a6000|l40s|h100" --mem=16G \
          --exclude="$EXCLUDE" \
          cluster/sweep.sbatch "$MODEL"
    fi
  done
done

echo
echo "once every job above finishes, score with:"
echo "  PYTHONPATH=. $PYTHON scripts/shortcut_table.py --graphs 500 --json shortcuts.json"
echo "  PYTHONPATH=. $PYTHON scripts/score_sweep.py --responses 'runs/qwen3-1.7b*.jsonl' --shortcuts shortcuts.json"
