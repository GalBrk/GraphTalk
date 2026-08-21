# Sweep outputs

Raw model responses from the primer sweep, plus the shortcut table they are read
against. Tracked in git so a collaborator can clone them; also readable directly
on the TAU CS cluster at `/home/dcor/galbarak2/GraphTalk/runs/` (world-readable,
no permissions needed).

## Files

| file | rows | what |
|---|---|---|
| `gemma4-e4b.jsonl` | 2520 | `google/gemma-4-E4B-it` |
| `gemma4-12b.jsonl` | 2520 | `google/gemma-4-12B-it` |
| `qwen3-8b.jsonl` | 2520 | `Qwen/Qwen3-8B` |
| `qwen3-14b.jsonl` | 2520 | `Qwen/Qwen3-14B` |
| `smoke-gemma4-e4b.jsonl` | 20 | a smoke test; **not** part of the sweep |
| `../shortcuts.json` | 36 cells | primer-only solver score per (task, condition) |
| `../prompts.jsonl` | 2520 | the exact prompts these responses answer |

Every model saw the identical prompt file. Each row is one JSON object:

```json
{"instance_id": "node_count/7", "task": "node_count", "condition": "degree",
 "style": "zero_shot", "gold": " 18.", "model": "gemma4-12b", "response": "..."}
```

`instance_id` is the pairing key: the same graph and query appear under all seven
conditions and both styles, differing only in the primer. That pairing is what
the McNemar test is computed over.

## Scoring them

```bash
python scripts/score_sweep.py --responses runs/*.jsonl --shortcuts shortcuts.json
```

Shards from a job array (`runs/<model>.shardNofM.jsonl`) need no reassembly --
`score_sweep.py` groups by the `model` field on each row, not by filename.

## Read this before drawing conclusions

`docs/sweep-findings.md` covers what these numbers can and cannot support. In
short: the McNemar analysis the proposal specifies is **underpowered** -- 259 of
288 cells have fewer than 10 discordant pairs -- and for `gemma4-12b` that is a
ceiling (98.3% under `none`) rather than a sample-size problem. Two caveats
travel with the data: 955 rows were generated on CPU before a driver mismatch was
caught and have not been re-verified against GPU output, and at the 2048-token
budget both prompt styles reason, so `zero_shot` vs `zero_cot` is a contrast
about wording rather than about whether reasoning happens.
