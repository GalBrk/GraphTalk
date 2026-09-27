# Preliminary data

The prompts, raw responses and solver bars of the pilot and the screens.
[← preliminary/](../README.md)

The full schema, including the published split's formats and per-row caveats,
is in [../docs/DATA.md](../docs/DATA.md). The main experiment's data is in
[`data/`](../../data/README.md).

## Files

| File (`runs/`) | Rows | What |
|---|---|---|
| `{gemma4-e4b,gemma4-12b,qwen3-8b,qwen3-14b}.jsonl` | 900 each | The pilot, plain arms, on the published split |
| `<model>-think.shard<i>of<n>.jsonl` | — | The pilot's thinking arms |
| `<model>.rerun.*.jsonl` | 360/arm | The prompt-rewording regeneration; part of its arm |
| `<model>.got.jsonl`, `.got.shard*` | 1,260/arm | The same arms with Game-of-Thrones node names; `node_naming: "got"` on every row |
| `<model>.probe100.*` | 1,200/arm | The small-model probe: 6 tasks × 100 graphs × {`none`, `degree`}, for `qwen3-0.6b[-think]`, `qwen3-1.7b[-think]`, `qwen35-2b` |
| `<model>.cc500.*`, `<model>.ec500.*` | 2,000/arm | Clean-condition cells: `cycle_check` and `edge_count`, 500 graphs × {`none`, `components`, `clustering`, `rwse`} |
| `<model>.size.*` | 600/arm | The size sweep: 20/40/80-node ER graphs × 50 × 4 tasks, `none` only |
| `<model>.ladder_screen*` | 900/arm | The ladder screen: 18 (n, k̄) rungs × 50 graphs, `node_degree`, `none` |
| `<model>.retrieval_locate.jsonl` | 1,050/arm | The graph-free retrieval probe, `condition=retrieval` |
| `<model>.retrieval_extend.jsonl`, `.retrieval.*`, `.retrieval_threshold.*` | — | Retrieval extensions |
| `<model>.rewire_shared.jsonl`, `qwen35-2b.rewire_extra.jsonl` | 1,800 / 2,700 | The rewiring stage at the shared rung, and two extra rungs for `qwen35-2b` |

`prompts/` holds the prompt file each family answers (`prompts.jsonl` for the
pilot, `prompts_got.jsonl` for GoT, and one per screen). `shortcuts.json` is the
graph-blind solver's bar per `"<task>/<condition>"` on the published split.

## Scoring them

Score each family on its own. A plain `runs/*.jsonl` glob mixes them:
- `build_sweep_frame.py`, `sample_failures.py` and `check_significance.py` refuse
  to mix GoT and integer rows.
- `score_sweep.py` would silently pool the ladder and rewiring rows into the
  pilot's `node_degree` cells, because it groups only by (task, style,
  condition). Score those with `analyze_ladder.py` and
  `analyze_rewiring_sweep.py` instead.

A GoT response names nodes by character while its `gold` stays integers, so it
must pass through `graphtalk.node_naming.desubstitute_response` before scoring.
The scripts do this; a hand-rolled `scoring.extract_answer` loop does not, and
misreads `connected_nodes`.

## Caveats

- 955 pilot rows were generated on CPU before a driver mismatch was caught, and
  were not re-verified against GPU output.
- `n_new_tokens` and `hit_cap` are on every row. Older rows were backfilled by
  re-tokenizing, and carry `token_count_source: "retokenized"`.
- The pilot's McNemar tests are underpowered: every cell has fewer than 10
  discordant pairs. For the two Gemma models that is a ceiling (98.9% and 96.7%
  under `none`), not a sample-size problem.
