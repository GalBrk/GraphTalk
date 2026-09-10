#!/bin/bash
# The "Smart Hybrid" size/density grid, two stages. Sibling to
# cluster/run_size_sweep.sh (same interpreter-resolution/--dry-run/EXCLUDE/
# sbatch plumbing) rather than an in-place extension of it: that script's
# prompts_<tasks>_n<size>.jsonl filename contract is depended on by
# scripts/size_screen.py's globs and by the existing size-sweep docs, and a
# density axis can't be bolted on without breaking that contract for every
# existing caller.
#
# --stage scout: cheap, fast, plain-arm, `none`-only pilot across the full
# node-count x density x task grid (scripts/score_density_size_grid.py
# --stage scout then applies the relaxed drop-only-absolute-losers screen).
#
#   cluster/run_density_size_sweep.sh --stage scout
#
# --stage full: deep-dive reporter (all 7 conditions, both arms, 100
# graphs/cell) on whichever (task, nodes, density) cells survived the scout
# screen -- read from SURVIVORS (default
# analysis/density_size_grid.survivors.json, written by
# `scripts/score_density_size_grid.py --stage scout`) -- plus anything named
# in --force-include (a copy of archive_filtered.csv, or any CSV/JSON with
# the same task/nodes/density columns, for pulling a scout-dropped cell back
# in).
#
#   cluster/run_density_size_sweep.sh --stage full
#   cluster/run_density_size_sweep.sh --stage full --force-include analysis/recovered.csv
#
# Override the grid/counts/models via env vars:
#
#   SIZES="10 20 40" DENSITIES="0.20 0.50" TASKS="node_degree cycle_check" \
#       SCOUT_COUNT=50 cluster/run_density_size_sweep.sh --stage scout
#
# Preview without building or submitting anything:
#
#   cluster/run_density_size_sweep.sh --stage scout --dry-run

set -euo pipefail

SIZES="${SIZES:-10 20 40 60 80}"
DENSITIES="${DENSITIES:-0.10 0.20 0.35 0.50 0.65 0.75 0.85}"
TASKS="${TASKS:-node_count edge_count node_degree connected_nodes edge_existence cycle_check}"
SCOUT_COUNT="${SCOUT_COUNT:-40}"
FULL_COUNT="${FULL_COUNT:-100}"
FULL_MODELS="${FULL_MODELS:-qwen3-1.7b qwen3-1.7b-think}"
SURVIVORS="${SURVIVORS:-analysis/density_size_grid.survivors.json}"
# n-801/802/803/804/n-501 are the documented bad-driver/slow nodes (see
# cluster/run_size_sweep.sh's own comment on this list).
EXCLUDE="${EXCLUDE:-n-501,n-801,n-802,n-803,n-804}"

STAGE=""
FORCE_INCLUDE=""
DRY_RUN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="${2:-}"; shift 2 ;;
    --force-include) FORCE_INCLUDE="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "FATAL: unknown argument '$1' (accepted: --stage scout|full," \
            "--force-include <path>, --dry-run)" >&2
       exit 1 ;;
  esac
done

if [[ "$STAGE" != "scout" && "$STAGE" != "full" ]]; then
  echo "FATAL: --stage scout|full is required." >&2
  exit 1
fi

# Refuse to run from the wrong place: everything below is a relative path.
if [[ ! -f cluster/sweep.sbatch ]]; then
  echo "FATAL: run this from the repo root (cluster/sweep.sbatch not found" >&2
  echo "in the current directory: $(pwd))." >&2
  exit 1
fi

# Same interpreter fallback as cluster/run_size_sweep.sh.
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

if [[ "$STAGE" == "scout" ]]; then
  for N in $SIZES; do
    for P in $DENSITIES; do
      PSTR="$(printf '%.2f' "$P")"
      PROMPTS="prompts_grid_scout_n${N}_p${PSTR}.jsonl"
      RUN_TAG="grid_scout_n${N}_p${PSTR}"
      if [[ -f "$PROMPTS" ]]; then
        echo "reusing existing $PROMPTS"
      else
        echo "building $PROMPTS (--node-count $N --er-min/max-sparsity $PSTR" \
             "--conditions none --tasks $TASKS --count $SCOUT_COUNT)"
        if [[ -z "$DRY_RUN" ]]; then
          PYTHONPATH=. "$PYTHON" scripts/build_prompts.py --graph-source diverse \
              --algorithms er --node-count "$N" \
              --er-min-sparsity "$P" --er-max-sparsity "$P" \
              --conditions none --tasks $TASKS --count "$SCOUT_COUNT" \
              --model qwen3-1.7b --out "$PROMPTS"
        fi
      fi
      echo "submitting qwen3-1.7b on $PROMPTS (tag $RUN_TAG)"
      if [[ -z "$DRY_RUN" ]]; then
        # See cluster/run_size_sweep.sh for why --constraint/--mem are
        # widened/shrunk from cluster/sweep.sbatch's own 48GB-tier default:
        # qwen3-1.7b only needs 8GB, and every l40s in `killable` has been
        # observed fully allocated while this pool sits idle.
        GRAPHTALK_PROMPTS="$PROMPTS" GRAPHTALK_RUN_TAG="$RUN_TAG" \
          sbatch --job-name="scout_${RUN_TAG}" \
            --constraint="a5000|geforce_rtx_3090|a6000|l40s|h100" --mem=16G \
            --exclude="$EXCLUDE" \
            cluster/sweep.sbatch qwen3-1.7b
      fi
    done
  done

  echo
  echo "once every job above finishes, score with:"
  echo "  PYTHONPATH=. $PYTHON scripts/score_density_size_grid.py --stage scout \\"
  echo "      --sizes $SIZES --densities $DENSITIES --tasks $TASKS"

else  # --stage full
  # Merge $SURVIVORS (scout survivors) and --force-include into one
  # (nodes, density) -> {tasks} map, in Python rather than bash: JSON/CSV
  # parsing and set-union-by-key are both a lot more error-prone as shell
  # text processing than as a five-line script, and this project already
  # embeds small Python heredocs in shell scripts for exactly this reason
  # (see cluster/sweep.sbatch's own REMAINING check).
  CELLS="$("$PYTHON" - "$SURVIVORS" "$FORCE_INCLUDE" <<'PY'
import csv
import json
import os
import sys

survivors_path, force_include_path = sys.argv[1], sys.argv[2]


def load(path):
  if not path or not os.path.exists(path):
    return []
  if path.endswith(".csv"):
    with open(path) as handle:
      rows = list(csv.DictReader(handle))
  else:
    with open(path) as handle:
      rows = json.load(handle)
    if not isinstance(rows, list):
      raise SystemExit(f"{path}: expected a JSON array of cell dicts")
  return [
      {"task": r["task"], "nodes": int(r["nodes"]), "density": float(r["density"])}
      for r in rows
  ]


cells = load(survivors_path) + load(force_include_path)
by_np = {}
for cell in cells:
  key = (cell["nodes"], cell["density"])
  by_np.setdefault(key, set()).add(cell["task"])

for (nodes, density), tasks in sorted(by_np.items()):
  print(f"{nodes}|{density:.2f}|{','.join(sorted(tasks))}")
PY
  )"

  if [[ -z "$CELLS" ]]; then
    echo "FATAL: no cells to build -- '$SURVIVORS' has none and no" >&2
    echo "--force-include was given (or it was also empty). Run" >&2
    echo "'scripts/score_density_size_grid.py --stage scout' first." >&2
    exit 1
  fi

  while IFS='|' read -r N PSTR TASKLIST; do
    [[ -z "$N" ]] && continue
    CELL_TASKS="${TASKLIST//,/ }"
    PROMPTS="prompts_grid_full_n${N}_p${PSTR}.jsonl"
    RUN_TAG="grid_full_n${N}_p${PSTR}"
    if [[ -f "$PROMPTS" ]]; then
      echo "reusing existing $PROMPTS"
    else
      echo "building $PROMPTS (--node-count $N --er-min/max-sparsity $PSTR" \
           "--tasks $CELL_TASKS --count $FULL_COUNT, all 7 conditions)"
      if [[ -z "$DRY_RUN" ]]; then
        PYTHONPATH=. "$PYTHON" scripts/build_prompts.py --graph-source diverse \
            --algorithms er --node-count "$N" \
            --er-min-sparsity "$PSTR" --er-max-sparsity "$PSTR" \
            --tasks $CELL_TASKS --count "$FULL_COUNT" \
            --model qwen3-1.7b --out "$PROMPTS"
      fi
    fi
    for MODEL in $FULL_MODELS; do
      echo "submitting $MODEL on $PROMPTS (tag $RUN_TAG)"
      if [[ -z "$DRY_RUN" ]]; then
        GRAPHTALK_PROMPTS="$PROMPTS" GRAPHTALK_RUN_TAG="$RUN_TAG" \
          sbatch --job-name="${MODEL}_${RUN_TAG}" \
            --constraint="a5000|geforce_rtx_3090|a6000|l40s|h100" --mem=16G \
            --exclude="$EXCLUDE" \
            cluster/sweep.sbatch "$MODEL"
      fi
    done
  done <<< "$CELLS"

  echo
  echo "once every job above finishes, score with:"
  echo "  PYTHONPATH=. $PYTHON scripts/score_density_size_grid.py --stage full \\"
  echo "      --sizes $SIZES --densities $DENSITIES --tasks $TASKS --shortcuts shortcuts.json"
fi
