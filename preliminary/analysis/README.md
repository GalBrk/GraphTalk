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

The pilot's scored frames and significance reports are in `../outputs/sweep-small-graph/`.

The pilot's significance-testing notes, which older docs here cite by section
("Current significance results", "Phase A1 candidates", "Track 2", "Phase C"),
are in git tag `pre-cleanup`:

```bash
git show pre-cleanup:analysis/README.md
```
