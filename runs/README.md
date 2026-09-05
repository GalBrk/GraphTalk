# Sweep outputs

Raw model responses from the primer sweep, plus the shortcut table they are read
against. Tracked in git so a collaborator can clone them; also readable directly
on the TAU CS cluster at `/home/dcor/galbarak2/GraphTalk/runs/` (world-readable,
no permissions needed).

## Files

| file | rows | what |
|---|---|---|
| `gemma4-e4b.jsonl` | 900 | `google/gemma-4-E4B-it` |
| `gemma4-12b.jsonl` | 900 | `google/gemma-4-12B-it` |
| `qwen3-8b.jsonl` | 900 | `Qwen/Qwen3-8B` |
| `qwen3-14b.jsonl` | 900 | `Qwen/Qwen3-14B` |
| `archive/` | — | rows that carry a `model` field but are **not** part of the sweep: the smoke test and the 4x-cap probe. Excluded by directory, not by filename. |
| `../shortcuts.json` | 36 cells | primer-only solver score per (task, condition) |
| `<model>.rerun.shardNofM.jsonl` | 360/arm | the prompt-rewording regeneration; part of the arm |
| `<model>.got.jsonl` | — | the same arm, generated from a GoT-named prompt file (`--node-naming got`); `node_naming: "got"` on every row, see [../README.md#node-naming](../README.md#node-naming) |
| `<model>.probe100.jsonl` / `.probe100.shardNofM.jsonl` | 1200/arm | the small-model probe: 6 tasks x 100 graphs x {none, degree}. Arms: `qwen3-0.6b`, `qwen3-0.6b-think`, `qwen3-1.7b`, `qwen35-2b`, `qwen3-1.7b-think`. See [../docs/primer-effects-and-power.md](../docs/primer-effects-and-power.md) |
| `<model>.ec500.shardNof5.jsonl` | 2000/arm | the `edge_count` clean-condition experiment: 500 graphs x {none, components, clustering, rwse}, the first properly powered test of an uncontaminated primer cell |
| `../prompts.count100.none_degree.jsonl` | 1200 | prompts for the `probe100` arms |
| `../prompts.edgecount500.clean.jsonl` | 2000 | prompts for the `ec500` arms |
| `../prompts.jsonl` | 1260 | the prompts these responses answer |

Full schema, field semantics and join keys: **[../docs/DATA.md](../docs/DATA.md)**.

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
conditions, differing only in the primer. That pairing is what the McNemar test
is computed over.

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
will glob them in -- all eight arms exist as of 2026-09-02, 1,260 rows each. `scripts/build_sweep_frame.py`,
`scripts/sample_failures.py`, and `scripts/check_significance.py` all raise
if their input carries more than one scheme rather than silently pooling it
(`graphtalk.analysis.infer_node_naming`/`frame_node_naming`); score each
scheme with its own `--responses`/`--frame` and let each land at its own
`.got.`-tagged output file. See [../README.md#node-naming](../README.md#node-naming).

### Scoring `.got.` rows by hand silently breaks `connected_nodes`

A GoT-named response answers in **character names** ("Maester, Catelyn") while
its `gold` stays **integers**, so it must go through
`node_naming.desubstitute_response` before it reaches the scorer. The supported
scripts do this; a fresh analysis that calls `scoring.extract_answer` straight
over `runs/*.jsonl` does not.

It corrupts exactly one task -- `connected_nodes`, the only one whose *answer* is
a node list. The integer tasks and the yes/no tasks never compare a name to a
gold, so they look fine, which is what makes it hard to spot. On 2026-09-05 this
made `qwen3-8b` appear to score **0.080** on `connected_nodes` when it actually
scores **0.996**, and produced a "0.6B beats an 8B tenfold" result that survived
into a written draft before being caught.

Either exclude `.got.` files from an ad-hoc analysis, or use
`scripts/build_sweep_frame.py`, which handles the desubstitution.

## Read this before drawing conclusions

`docs/sweep-findings.md` covers what these numbers can and cannot support. In
short: the McNemar analysis the proposal specifies is **underpowered** -- every
one of 144 cells has fewer than 10 discordant pairs -- and for `gemma4-12b` and
`gemma4-e4b` that is a ceiling (98.9% and 96.7% under `none`) rather than a
sample-size problem. One caveat travels with the data: 955 rows were generated
on CPU before a driver mismatch was caught and have not been re-verified
against GPU output.
