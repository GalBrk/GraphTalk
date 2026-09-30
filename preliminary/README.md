# Preliminary work

The pilot and the screens that chose the main experiment's settings: 40-node
graphs, the density range, and the Qwen3 models. [← back to the repo](../README.md)

The paper cites none of these numbers. It states only the pilot's outcome,
given below. The scripts read and write inside `preliminary/`,
`preliminary/tests/` is part of the default test suite, and every command in
this folder's READMEs runs as written from the repo root, under bash or zsh.
The older write-ups quote commands with the paths they were written against;
[Where things moved](#where-things-moved) translates them.

## What was run, and what it decided

| Stage | Runs (`data/runs/`) | Document | Status |
|---|---|---|---|
| **Pilot** on the published GraphQA graphs (5–19 nodes): Gemma 4 E4B and 12B, Qwen3-8B and 14B, each with and without thinking | `{gemma4-e4b,gemma4-12b,qwen3-8b,qwen3-14b}[-think]*.jsonl`, including `.rerun.` | this row | Without a primer most arms are near ceiling on five of the six tasks, and only `edge_count` separates them. **This is why the main experiment moved to 40-node graphs.** The earlier write-ups, [sweep-findings.md](docs/sweep-findings.md) (kept for its retractions) and the significance notes in `git show pre-cleanup:analysis/README.md`, are not re-verified |
| Ladder screen and retrieval probe | `*.ladder_screen*`, `*.retrieval_locate*` | [ladder-and-retrieval-results.md](docs/ladder-and-retrieval-results.md) (design: [ladder-and-rewiring.md](docs/ladder-and-rewiring.md)) | not re-verified; its tables omit the later ladder and probe runs that `outputs/ladder-retrieval/` includes |
| Game-of-Thrones node naming | `*.got*` | [analysis/task_scoped_screen_comparison.md](analysis/task_scoped_screen_comparison.md), [scale-vs-topology-investigation.md](docs/scale-vs-topology-investigation.md), [sweep-findings.md](docs/sweep-findings.md) "Renaming every node changes nothing", and "The GOT run" in `git show pre-cleanup:analysis/README.md` | not re-verified |
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
| `docs/`, `analysis/` | The write-ups listed above, and the significance and token-budget measurements; see [analysis/README.md](analysis/README.md) |
| `data/` | The prompts, runs and published-split solver bars (`shortcuts.json`); see [data/README.md](data/README.md) |
| `outputs/` | `sweep-small-graph/` (the pilot's scored frames and screens) and `ladder-retrieval/`; see [Outputs](#outputs) |
| `cluster/` | `submit_sweep.sh` (the pilot's integer and GoT runs) and `run_ladder.sh`; see [cluster/README.md](cluster/README.md) |

## Rerun

From the repo root. The quoted patterns are globbed by the scripts themselves,
so the braces expand the same way in bash and zsh:

```bash
export PYTHONPATH=.
python preliminary/scripts/score_sweep.py \
    --responses preliminary/data/runs/gemma4-12b.jsonl preliminary/data/runs/gemma4-12b.rerun.jsonl \
    --shortcuts preliminary/data/shortcuts.json
python preliminary/scripts/build_sweep_frame.py \
    --responses preliminary/data/runs/{gemma4-e4b,gemma4-12b,qwen3-8b,qwen3-14b}{,-think}{.jsonl,'.shard*','.rerun*'} \
    --shortcuts preliminary/data/shortcuts.json --truncated-keys preliminary/analysis/truncated_keys.json
python preliminary/scripts/build_sweep_frame.py \
    --responses 'preliminary/data/runs/*.got.jsonl' 'preliminary/data/runs/*.got.shard*.jsonl' \
    --shortcuts preliminary/data/shortcuts.json --truncated-keys preliminary/analysis/truncated_keys.json
python preliminary/scripts/build_sweep_frame.py --responses 'preliminary/data/runs/*.got.count500.*.jsonl' \
    --shortcuts preliminary/data/shortcuts.json --truncated-keys preliminary/analysis/truncated_keys.json \
    --out preliminary/outputs/sweep-small-graph/sweep_frame.count500.got.csv
python preliminary/scripts/task_scoped_screen.py \
    --frame preliminary/outputs/sweep-small-graph/sweep_frame.csv --n-perm 3000 --n-boot 3000
python preliminary/scripts/task_scoped_screen.py \
    --frame preliminary/outputs/sweep-small-graph/sweep_frame.got.csv --n-perm 3000 --n-boot 3000
python preliminary/scripts/check_significance.py \
    --frame preliminary/outputs/sweep-small-graph/sweep_frame.count500.got.csv \
    --confirmatory-config preliminary/analysis/confirmatory_got_degree.json --n-perm 10000 \
    --out preliminary/outputs/sweep-small-graph/significance_report.count500.csv
python preliminary/scripts/check_significance.py --frame preliminary/outputs/sweep-small-graph/sweep_frame.csv \
    --out preliminary/analysis/significance_report.csv
python preliminary/scripts/naming_effect.py
python preliminary/scripts/analyze_retrieval.py --responses 'preliminary/data/runs/*.retrieval_locate.jsonl' \
    --out preliminary/outputs/ladder-retrieval/retrieval_matrix.csv
python preliminary/scripts/analyze_ladder.py --responses 'preliminary/data/runs/*.ladder_screen*.jsonl' \
    --reading-limits qwen3-0.6b=0 qwen3-0.6b-think=0 qwen3-1.7b=1509 qwen3-1.7b-think=2449 \
    --out preliminary/outputs/ladder-retrieval/ladder_matrix.csv
```

These rewrite the committed files under `outputs/` unchanged; the table under
[Outputs](#outputs) gives the command for the rest. `check_significance.py`
tags its `--out` with the frame's naming scheme, so the `--count 500` report
lands at `significance_report.count500.got.csv`. The integer report it writes
to `preliminary/analysis/significance_report.csv` is not tracked; it is the
input `recommend_count.py` reads by default, so run `check_significance.py
--out` first.

A plain `preliminary/data/runs/*.jsonl` glob mixes the pilot with the GoT,
ladder and probe families; the pilot's own files are the ones above (see
[data/README.md](data/README.md#scoring-them)). `build_prompts.py` fetches
rows from the HuggingFace datasets-server and needs network access.

## Outputs

| File | Written by |
|---|---|
| `sweep-small-graph/sweep_frame.csv`, `sweep_frame.got.csv`, `sweep_frame.count500.got.csv` | `build_sweep_frame.py`, [Rerun](#rerun) |
| `sweep-small-graph/task_scoped_screen.csv`, `task_scoped_screen.got.csv` | `task_scoped_screen.py`, [Rerun](#rerun) |
| `sweep-small-graph/significance_report.count500.got.csv` | `check_significance.py`, [Rerun](#rerun) |
| `ladder-retrieval/ladder_matrix.csv`, `retrieval_matrix.csv` | `analyze_ladder.py`, `analyze_retrieval.py`, [Rerun](#rerun) |
| `ladder-retrieval/retrieval.qwen3-8b.csv` | `analyze_retrieval.py --responses 'preliminary/data/runs/qwen3-8b.retrieval.shard*.jsonl' --out …` |
| `ladder-retrieval/retrieval_threshold.qwen3-1.7b.csv` | `analyze_retrieval.py --responses 'preliminary/data/runs/qwen3-1.7b.retrieval_threshold.shard*.jsonl' --out …` |
| `ladder-retrieval/retrieval_extend_matrix.csv` | `analyze_retrieval.py --responses 'preliminary/data/runs/*.retrieval_extend.jsonl' --out …` |
| `ladder-retrieval/rewiring_qwen3-1.7b.json`, `rewiring_qwen3-1.7b-think.json` | `analyze_rewiring_sweep.py --responses preliminary/data/runs/<model>.rewire_shared.jsonl --json …` |
| `ladder-retrieval/rewiring_qwen35-2b.json` | `analyze_rewiring_sweep.py --responses 'preliminary/data/runs/qwen35-2b.rewire_*.jsonl' --json …` |
| `ladder-retrieval/ladder_matrix.limited.csv` | The two-gate matrix, over the three small Qwen models' arms, that the rewiring stage's shared rung (`n40k12`) was read off. It was cut from partial ladder files with other reading limits, so three rows' counts and nine `readable` flags differ from `ladder_matrix.csv`; no command in this tree rewrites it |
| `sweep-small-graph/topology_*.csv` | The topology investigation ([scale-vs-topology-investigation.md](docs/scale-vs-topology-investigation.md)), whose scripts are in tag `pre-cleanup` |
| `sweep-small-graph/sweep_frame.qwen3-1.7b.csv` | The size sweep in [primer-impact-and-truncation.md](docs/primer-impact-and-truncation.md), whose runs were removed |
| `sweep-small-graph/non_termination_sample.csv` | The non-termination sample cited in [sweep-findings.md](docs/sweep-findings.md); its script is in tag `pre-cleanup` |

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
| `tests/test_<one of those scripts>.py` | `preliminary/tests/` |
| `cluster/submit_sweep.sh`, `cluster/run_ladder.sh` | `preliminary/cluster/` |
| `docs/<one of the above>.md`, `docs/DATA.md` | `preliminary/docs/` |
| `docs/plans/run_improved_tests.md`, `docs/plans/scale-vs-topology-investigation.md` | `preliminary/docs/` |
| `docs/plans/shortcut-ceilings.md`, `docs/plans/primer-computation.md`, `docs/graph-design-requirements.md` | `docs/design/` |
| `runs/*.densfull40*` and the density follow-ups | `data/runs/` (the main experiment) |
| `csv2/raw-trends/`, `csv2/density-followups/` | `outputs/n40-sweep/`, `outputs/density-followups/` |

What these docs mention and this tree does not have is in git tag
`pre-cleanup`:

- the one-off scripts: `validate_*`, `benchmark_mde.py`, `audit_extraction.py`,
  `backfill_hit_cap.py`, `rewording_effect.py`, `characterize_non_termination.py`,
  the topology investigation and the GEE/Bayesian cross-checks;
- `docs/plans/finding-graphs-that-make-primer-effects-measurable.md`;
- `superseded/`, the replaced paper drafts and their analyses.

```bash
git show pre-cleanup:scripts/benchmark_mde.py      # or: git checkout pre-cleanup
```
