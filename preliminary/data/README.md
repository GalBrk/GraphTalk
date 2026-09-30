# Preliminary data

The prompts, raw responses and solver bars of the pilot and the screens.
[← preliminary/](../README.md)

The full schema, including the published split's formats and per-row caveats,
is in [../docs/DATA.md](../docs/DATA.md). The main experiment's data is in
[`data/`](../../data/README.md).

## Files

| File (`runs/`) | Rows | Prompt file (`prompts/`) | What |
|---|---|---|---|
| `{gemma4-e4b,gemma4-12b,qwen3-8b,qwen3-14b}.jsonl` | 900 each | `prompts.jsonl` | The pilot, plain arms, on the published split |
| `<model>-think.shard<i>of<n>.jsonl` | 900/arm | `prompts.jsonl` | The pilot's thinking arms |
| `<model>.rerun[.shard*].jsonl` | 360/arm | `prompts.jsonl` | The prompt-rewording regeneration; part of its arm |
| `<model>.got.jsonl`, `.got.shard*` | 1,260/arm | `prompts_got.jsonl` | The same arms with Game-of-Thrones node names; `node_naming: "got"` on every row |
| `qwen3-8b.got.count500.shard*` | 6,000 | `prompts_got.count500.degree.jsonl` | The GoT `--count 500` follow-up: 500 graphs × 6 tasks × {`none`, `degree`} |
| `<model>.probe100.*` | 1,200/arm | `prompts.count100.none_degree.jsonl` | The small-model probe: 6 tasks × 100 graphs × {`none`, `degree`}, for `qwen3-0.6b[-think]`, `qwen3-1.7b[-think]`, `qwen35-2b`, `qwen3-4b` |
| `<model>.cc500.*`, `<model>.ec500.*` | 2,000/arm; `qwen3-8b.ec500` 1,984 | `prompts.cyclecheck500.clean.jsonl`, `prompts.edgecount500.clean.jsonl` | Clean-condition cells: `cycle_check` and `edge_count`, 500 graphs × {`none`, `components`, `clustering`, `rwse`} |
| `<model>.size.*` | 600/arm | `prompts.sizesweep.jsonl` | The size sweep: 20/40/80-node ER graphs × 50 × 4 tasks, `none` only |
| `<model>.ladder_screen*` | 900/arm; `qwen3-14b-think` 450 (25 graphs per rung), `qwen35-2b-think` 868 | `prompts.ladder_screen.jsonl` | The ladder screen: 18 (n, k̄) rungs × 50 graphs, `node_degree`, `none` |
| `<model>.retrieval_locate.jsonl` | 1,050/arm; `qwen3-0.6b-think` 614 | `prompts.retrieval_locate.jsonl` | The graph-free retrieval probe, `condition=retrieval` |
| `<model>.retrieval_extend.jsonl` | 450/arm | `prompts.retrieval_extend.jsonl` | Retrieval extension for `qwen3-8b` and `qwen3-14b` |
| `<model>.retrieval.*`, `qwen3-1.7b.retrieval_threshold.*` | 3,600/arm, 1,800 | not tracked | Retrieval extensions: `.retrieval.` for `qwen3-1.7b` and `qwen3-8b`, `.retrieval_threshold.` for `qwen3-1.7b` |
| `<model>.rewire_shared.jsonl` | 1,800/arm | `prompts.rewire_shared.jsonl` | The rewiring stage at the shared rung, for `qwen3-1.7b[-think]` and `qwen35-2b` |
| `qwen35-2b.rewire_extra.jsonl` | 2,700 | `prompts.rewire_2b_extra.jsonl` | Two extra rewiring rungs for `qwen35-2b` |

Where a prompt file is tracked, every run file's `(instance_id, condition,
style)` keys are a subset of it. `prompts_zero_shot.jsonl` is byte-identical
to `prompts.jsonl`. `shortcuts.json` is the graph-blind solver's bar per
`"<task>/<condition>"` on the published split.

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
- The pilot's McNemar tests are underpowered: all but one of `score_sweep.py`'s
  288 cells over the 40 pilot files have fewer than 10 discordant pairs (the
  exception, `gemma4-12b-think` `connected_nodes`/`components`, has 11). For the
  two Gemma models that is a ceiling (98.9% and 96.7% under `none`, the mean of
  `score_sweep.py`'s six per-task figures), not a sample-size problem.
