# Data

The committed inputs to every analysis: the prompts, the models' raw responses,
and the graph-blind solver's bars. [← back to the repo](../README.md)

Everything here is read-only. The analyses write to `../outputs/`
([scripts/README.md](../scripts/README.md)).

## Files

| Path | Rows | What |
|---|---|---|
| `prompts/prompts.densfull40.jsonl` | 16,800 | The main sweep: 6 tasks × 7 conditions × 4 densities (p = .10, .20, .35, .50) × 100 graphs |
| `prompts/prompts.densfull40hi.jsonl` | 4,200 | Its high-density extension: `node_degree` and `edge_existence` × 7 conditions × 3 densities (p = .65, .75, .85) × 100 graphs |
| `prompts/prompts.density40.jsonl` | 2,400 | The `density40` follow-up: `node_degree` and `connected_nodes` × {`none`, `degree`} |
| `runs/` | 130,402 | Raw responses, one file per (model, run set, shard); families below |
| `shortcuts_n40_flat.json` | 42 keys | The graph-blind solver's accuracy per `"<task>/<condition>"`, averaged over densities; the bar an effect is read against |
| `shortcuts_n40.json` | — | The same, per density |

The other follow-ups' prompts are not committed: `scripts/density_followups.py`
rebuilds them in memory and checks every gold answer against the runs.

## Run families

| Run set | Models | Rows per model | Documented in |
|---|---|---|---|
| `densfull40` | `qwen3-1.7b`, `qwen3-1.7b-think`, `qwen3-4b`, `qwen3-4b-think` | 16,800 | [n40-sweep.md](../docs/results/n40-sweep.md) |
| `densfull40hi` | the same four | 4,200 | [n40-sweep.md](../docs/results/n40-sweep.md) |
| `degdens40`, `degceil`, `degdens40hi`, `degdensfill`, `degdensrep`, `degfixdeg`, `density40` | `qwen3-1.7b` | 4,800, 1,600, 4,800, 2,800, 3,200, 6,400, 2,400 | [density-followups.md](../docs/results/density-followups.md) |
| `degdensthink`, `degdensfillT` | `qwen3-1.7b-think` | 11,200, 2,800 | [density-followups.md](../docs/results/density-followups.md) |
| `degfixdeg` | `qwen3-8b` | 6,400 | [density-followups.md](../docs/results/density-followups.md) |

File names follow `runs/<model>.<run set>.shard<i>of<n>.jsonl`. Shards are
bookkeeping for resumable GPU jobs: every row carries its `model`, so shards
need no reassembly.

## Schemas

A **prompt** row:

| Field | Meaning |
|---|---|
| `instance_id` | `<task>/size<n>/p<density>/<index>`, e.g. `node_degree/size40/p0.1/0`. A fresh-seed replication adds `/s<seed>/` before the index, so it can never pool with the main corpus |
| `task`, `condition`, `style` | One of six tasks, one of seven conditions; `style` is always `zero_shot` |
| `prompt` | The exact text the model saw: primer, blank line, incident encoding, question |
| `gold` | The correct answer |
| `nodes`, `edges`, `size_class`, `density_class`, `density` | The graph's size and its ER density |

A **response** row has `instance_id`, `task`, `condition`, `style` and `gold`,
copied from the prompt, plus:
- `model`, the arm (`-think` marks thinking mode);
- `response`, the generated text;
- `n_new_tokens`;
- `hit_cap`, true when generation used the whole token budget;
- `overflow`, true when the prompt did not fit the model's context window, so
  nothing was generated and `response` is null. `scripts/run_sweep.py` writes it
  on every row; the committed rows do not carry it, and every one of them has a
  response.

The prompt text is not repeated; join on `(instance_id, condition)` to recover it.

**The pairing key.** Within one model, `(instance_id, condition)` is unique,
apart from the two duplicates noted under Caveats. The
same graph and question appear under every condition, differing only in the
primer, so every comparison is paired on the same graphs.

## Caveats

- `qwen3-1.7b.densfull40` holds 16,802 rows: two keys appear twice.
  `build_raw_frame.py` keeps one of each and says so when it runs.
- The plain arms ran `densfull40hi` with a 2,048-token budget, a quarter of the
  main sweep's (n40-sweep.md §10).
- Scoring reads the answer from the whole response, `<think>` block included.
  Across all 42,000 thinking-arm rows of the 40-node sweep, reading only the text
  after `</think>` changes no outcome from correct to wrong or back. The only
  disagreements are truncated responses, which are not correct either way.
- `build_raw_frame.py` regenerates every graph from its `instance_id` and stops
  if any regenerated gold answer differs from the recorded one.

The pilot's data (the published 5–19-node graphs) has its own reference:
[preliminary/data/README.md](../preliminary/data/README.md).
