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
| `archive/` | — | rows that carry a `model` field but are **not** part of the sweep: the smoke test and the 4x-cap probe. Excluded by directory, not by filename. |
| `../shortcuts.json` | 36 cells | primer-only solver score per (task, condition) |
| `<model>.rerun.shardNofM.jsonl` | 360/arm | the prompt-rewording regeneration; part of the arm |
| `<model>.got.jsonl` | — | the same arm, generated from a GoT-named prompt file (`--node-naming got`); `node_naming: "got"` on every row, see [../README.md#node-naming](../README.md#node-naming) |
| `../prompts.jsonl` | 2520 | the prompts these responses answer — **except** the 1,440 un-regenerated `zero_cot` rows |
| `../prompts.original-wording.jsonl` | 360 | the prompts those 1,440 rows actually answer |

Full schema, field semantics and join keys: **[../docs/DATA.md](../docs/DATA.md)**.

`zero_cot` rows are historical -- that prompt style is retired in favour of the
thinking arm, and its `filler` and `edge_existence` rows answer prompts that no
longer exist. Filter on `style == "zero_shot"` for anything current.

Every model saw the identical prompt file. Each row is one JSON object:

```json
{"instance_id": "node_count/7", "task": "node_count", "condition": "degree",
 "style": "zero_shot", "gold": " 18.", "model": "gemma4-12b", "response": "...",
 "n_new_tokens": 143, "hit_cap": false, "token_count_source": "retokenized"}
```

`n_new_tokens` and `hit_cap` are now on **every** row. Rows generated during or
after the prompt-rewording re-run carry the generator's own count; the rest were
backfilled by `scripts/backfill_hit_cap.py`, which re-tokenizes the response against
its budget and marks itself with `token_count_source: "retokenized"`. That method
reproduces the generator's flag on 45/45 capped and 2,835/2,835 non-capped rows.
`analysis/truncated_keys.json` is no longer consulted for any tracked row; it is kept
as the historical record of what was hand-labelled, including two rows it labels that
in fact terminated.

`instance_id` is the pairing key: the same graph and query appear under all seven
conditions and both styles, differing only in the primer. That pairing is what
the McNemar test is computed over.

## Scoring them

```bash
python scripts/score_sweep.py --responses runs/*.jsonl --shortcuts shortcuts.json
```

Shards from a job array (`runs/<model>.shardNofM.jsonl`) need no reassembly --
`score_sweep.py` groups by the `model` field on each row, not by filename. The same
is true of the `.rerun.` files: they are part of their arm, unlike
`runs/archive/*.redo.shard*.jsonl`, which `graphtalk/analysis.py` excludes by
directory.
That is why the regeneration is tagged `rerun` and not `redo` -- the exclusion
matches on the substring `.redo.shard`, so the wrong tag would drop every
regenerated row from the frame without raising anything.

`.got.jsonl` files are **not** excluded the way `archive/` is -- they are
part of the sweep, just a different node-naming scheme, so `runs/*.jsonl`
will glob them in once any exist. `scripts/build_sweep_frame.py`,
`scripts/sample_failures.py`, and `scripts/check_significance.py` all raise
if their input carries more than one scheme rather than silently pooling it
(`graphtalk.analysis.infer_node_naming`/`frame_node_naming`); score each
scheme with its own `--responses`/`--frame` and let each land at its own
`.got.`-tagged output file. See [../README.md#node-naming](../README.md#node-naming).

## Read this before drawing conclusions

`docs/sweep-findings.md` covers what these numbers can and cannot support. In
short: the McNemar analysis the proposal specifies is **underpowered** -- 259 of
288 cells have fewer than 10 discordant pairs -- and for `gemma4-12b` that is a
ceiling (98.3% under `none`) rather than a sample-size problem. Two caveats
travel with the data: 955 rows were generated on CPU before a driver mismatch was
caught and have not been re-verified against GPU output, and at the 2048-token
budget both prompt styles reason, so `zero_shot` vs `zero_cot` is a contrast
about wording rather than about whether reasoning happens.

A third now travels with it: `condition: filler` and the `edge_existence` question
mean **two different prompts** depending on style, because only the `zero_shot` rows
were regenerated after the rewording. Group by the frame's `wording` column before
comparing anything that touches those cells; see `docs/DATA.md`.
