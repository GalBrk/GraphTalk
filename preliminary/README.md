# Preliminary work

The pilot and the screens that chose the main experiment's settings: 40-node
graphs, the density range, and the Qwen3 models. [← back to the repo](../README.md)

The paper cites none of these numbers. It states only the pilot's outcome,
given below. Everything here still runs: the scripts read and write inside
`preliminary/`, and `preliminary/tests/` is part of the default test suite.

## What was run, and what it decided

| Stage | Runs (`data/runs/`) | Document | Status |
|---|---|---|---|
| **Pilot** on the published GraphQA graphs (5–19 nodes): Gemma 4 E4B and 12B, Qwen3-8B and 14B, each with and without thinking | `{gemma4-e4b,gemma4-12b,qwen3-8b,qwen3-14b}[-think]*.jsonl`, including `.rerun.` | this row | Without a primer most arms are near ceiling on four of the five tasks, and only `edge_count` separates them. **This is why the main experiment moved to 40-node graphs.** The earlier write-ups, [sweep-findings.md](docs/sweep-findings.md) (kept for its retractions) and [analysis/README.md](analysis/README.md), are not re-verified |
| Ladder screen and retrieval probe | `*.ladder_screen*`, `*.retrieval_locate*` | [ladder-and-retrieval-results.md](docs/ladder-and-retrieval-results.md) (design: [ladder-and-rewiring.md](docs/ladder-and-rewiring.md)) | current |
| Game-of-Thrones node naming | `*.got*` | [analysis/README.md](analysis/README.md), [analysis/task_scoped_screen_comparison.md](analysis/task_scoped_screen_comparison.md), [scale-vs-topology-investigation.md](docs/scale-vs-topology-investigation.md) | not re-verified |
| Published-split probes and clean-condition cells | `*.probe100*`, `*.cc500*`, `*.ec500*` | [primer-effects-and-power.md](docs/primer-effects-and-power.md) | not re-verified |
| Size sweep (20, 40 and 80 nodes, no primer) | `*.size*` | [primer-effects-and-power.md](docs/primer-effects-and-power.md), "Does size break them?" | not re-verified |
| Retrieval extensions | `*.retrieval_extend*`, `*.retrieval.*`, `*.retrieval_threshold*` | — | no results document |
| Rewiring | `*.rewire_shared*`, `qwen35-2b.rewire_extra.jsonl` | design: [ladder-and-rewiring.md](docs/ladder-and-rewiring.md) | no results document |

Other docs:
- [difficulty-scaling.md](docs/difficulty-scaling.md): the `--graph-source diverse` pipeline changes.
- [graph-corpus-status.md](docs/graph-corpus-status.md): which corpus suits which model.
- [run_improved_tests.md](docs/run_improved_tests.md): the statistical-power plan.
- [DATA.md](docs/DATA.md): the pilot data's schema reference.
- [primer-impact-and-truncation.md](docs/primer-impact-and-truncation.md): a size sweep whose runs were removed, so it cannot be reproduced.

## Layout

| Folder | What's inside |
|---|---|
| `scripts/` | The pilot pipeline: `build_prompts.py` (fetches the published split), `score_sweep.py` (per-cell McNemar), `build_sweep_frame.py`, `check_significance.py` (pooled permutation tests). Also the ladder, retrieval, rewiring and naming analyses |
| `tests/` | Their tests, which run in the default suite |
| `docs/`, `analysis/` | The write-ups listed above, and the significance and token-budget measurements |
| `data/` | The prompts, runs and published-split solver bars (`shortcuts.json`); see [data/README.md](data/README.md) |
| `outputs/` | `sweep-small-graph/` (the pilot's scored frames) and `ladder-retrieval/` |
| `cluster/` | `submit_sweep.sh` (GoT runs) and `run_ladder.sh`; see [cluster/README.md](cluster/README.md) |

## Rerun

From the repo root:

```bash
export PYTHONPATH=.
python preliminary/scripts/score_sweep.py --responses preliminary/data/runs/gemma4-12b.jsonl \
    --shortcuts preliminary/data/shortcuts.json
PILOT=$(ls preliminary/data/runs/{gemma4-e4b,gemma4-12b,qwen3-8b,qwen3-14b}{,-think}{.jsonl,.shard*,.rerun*} 2>/dev/null)
python preliminary/scripts/build_sweep_frame.py --responses $PILOT \
    --shortcuts preliminary/data/shortcuts.json --truncated-keys preliminary/analysis/truncated_keys.json
python preliminary/scripts/check_significance.py --frame preliminary/outputs/sweep-small-graph/sweep_frame.csv \
    --out preliminary/analysis/significance_report.csv
python preliminary/scripts/analyze_ladder.py --responses 'preliminary/data/runs/*.ladder_screen*.jsonl'
```

A plain `preliminary/data/runs/*.jsonl` glob mixes the pilot with the GoT,
ladder and probe families; the pilot's own files are the ones above (see
[data/README.md](data/README.md#scoring-them)). `recommend_count.py` reads the
significance report, so run `check_significance.py --out` first. `build_prompts.py` fetches rows from the HuggingFace
datasets-server and needs network access.

## Where things moved

Docs written before the cleanup use the old paths.

| Old path | Now |
|---|---|
| `runs/` (these families) | `preliminary/data/runs/` |
| `prompts.jsonl`, `prompts_got.jsonl`, … | `preliminary/data/prompts/` |
| `shortcuts.json` | `preliminary/data/shortcuts.json` |
| `csv2/sweep-small-graph/`, `csv2/ladder-retrieval/` | `preliminary/outputs/` |
| `analysis/` | `preliminary/analysis/` |
| `scripts/<one of the above>.py` | `preliminary/scripts/` |
| `docs/<one of the above>.md`, `docs/DATA.md` | `preliminary/docs/` |
| `runs/*.densfull40*` and the density follow-ups | `data/runs/` (the main experiment) |
| `csv2/raw-trends/`, `csv2/density-followups/` | `outputs/n40-sweep/`, `outputs/density-followups/` |

The one-off validation scripts these docs sometimes mention were removed after
they had done their job. So were `superseded/`, the replaced paper drafts and
their analyses. All of it is in git tag `pre-cleanup`:

- `validate_*`, `benchmark_mde.py`, `audit_extraction.py`, `backfill_hit_cap.py`,
  the topology investigation and the GEE/Bayesian cross-checks;
- `superseded/`, the replaced paper drafts and their analyses.

```bash
git show pre-cleanup:scripts/benchmark_mde.py      # or: git checkout pre-cleanup
```
