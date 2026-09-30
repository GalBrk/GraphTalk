# Measurement artefacts

> **Status: not re-verified.** Everything here predates the current scoring rule
> (a truncated response is its own outcome) and is not cited by the paper.
> Current results: [docs/README.md](../../docs/README.md).

Small, expensive-to-reproduce measurements that decisions in the pilot rest on.
Each cost GPU time a reader should not have to spend again to check the claim.

| File | What it is |
|---|---|
| `budget-gemma4-e4b.jsonl` | 24 `zero_shot` prompts spanning all six tasks, generated at a 2,048-token cap. It set `MAX_NEW_TOKENS["zero_shot"]`: at 64 tokens this model scored 3/24, at 2,048 it scored 24/24 with no row reaching the cap. Also the reference a batched implementation must reproduce |
| `budget-qwen3-8b.jsonl` | The same 24 prompts on Qwen3-8B with thinking off |
| `budget-qwen3-8b-THINKING.jsonl` | Three of those rows with thinking on: 1,179 mean tokens on a `node_count` question the same model answers in 105 without |
| `truncated_keys.json` | The 271 thinking-arm rows once hand-labelled non-terminating. No longer consulted (every row now carries `hit_cap`); kept as the provenance of that claim |
| `non_terminating_manifest.json` | Counts of non-terminating rows per thinking arm, by task and condition |
| `confirmatory_*.json` | Pre-registered cells for `check_significance.py --confirmatory-config`; see that script's docstring for the format |
| [primer_task_shortcut_audit.md](primer_task_shortcut_audit.md) | Which (primer, task) pairs let the primer text alone determine the answer |
| [task_scoped_screen_comparison.md](task_scoped_screen_comparison.md) | The per-task screen under Game-of-Thrones names against integer names |
| `rerun/probe100.qwen3-4b.txt` | Console output of `score_sweep.py --responses preliminary/data/runs/qwen3-4b.probe100.shard*.jsonl --shortcuts preliminary/data/shortcuts.json` |
| `rerun/retrieval.qwen3-8b.txt`, `rerun/retrieval_threshold.qwen3-1.7b.txt` | Console output of the two `analyze_retrieval.py` runs whose CSVs are in `../outputs/ladder-retrieval/` ([../README.md](../README.md#outputs) has the commands); only the printed `--out` path differs from a rerun |
| `rerun/size.qwen3-4b.txt` | Console output of `score_density_sweep.py` on `qwen3-4b.size.*`, from the version of that script in `superseded/scripts/` in tag `pre-cleanup` |
| `tables/published_split.cc500.txt`, `tables/published_split.ec500.txt` | Console output of `score_sweep.py --responses preliminary/data/runs/*.cc500.shard*.jsonl` (and `*.ec500.shard*.jsonl`), no `--shortcuts` |
| `topology_distribution_plots/` | The distribution plots of [../docs/scale-vs-topology-investigation.md](../docs/scale-vs-topology-investigation.md) |

The pilot's scored frames and per-task screens, and the significance report of
the GoT `--count 500` follow-up, are in `../outputs/sweep-small-graph/`. The
integer report is not tracked: the [Rerun](../README.md#rerun) block writes it
to `significance_report.csv` here.

The pilot's significance-testing notes, which older docs here cite by section
("Current significance results", "Phase A1 candidates", "Track 2", "Phase C"),
are in git tag `pre-cleanup`:

```bash
git show pre-cleanup:analysis/README.md
```
