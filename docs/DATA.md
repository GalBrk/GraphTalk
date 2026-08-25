# Data reference

Every dataset in this repository, its schema, and how the files join to each
other. `runs/README.md` covers day-to-day use; this file is the authority on
structure.

## Inventory

| path | rows | what |
|---|---|---|
| `prompts.jsonl` | 2,520 | every prompt in the main sweep |
| `prompts_zero_shot.jsonl` | 1,260 | the `zero_shot` half, byte-identical to those rows; the thinking arm's input |
| `runs/<model>.jsonl` | 2,520 × 4 | main sweep responses, one file per model |
| `runs/<model>-think.shard<i>of<n>.jsonl` | 1,260 × 4 | thinking-arm responses, split across shard files |
| `runs/<model>-think.redo.shard<i>of<n>.jsonl` | 67 | evidence only — see caveats |
| `runs/smoke-gemma4-e4b.jsonl` | 20 | a smoke test; **exclude from analysis** |
| `shortcuts.json` | 42 entries | the primer-only solver bar per (task, condition) |
| `analysis/*.jsonl` | 51 | token-budget measurements; see `analysis/README.md` |

15,187 response rows in total: 10,080 main sweep, 5,040 thinking arm, 67 redo.

## The pairing key

`instance_id` has the form `"<task>/<index>"`, e.g. `node_count/7`. **The same
graph and the same query appear under all seven conditions and both styles**,
differing only in the primer. That is what makes the design paired, and it is the
key the McNemar test is computed over.

A row is uniquely identified by `(instance_id, condition, style)` within a model.
No file contains a duplicate of that triple; this is asserted after every run.

## `prompts.jsonl` / `prompts_zero_shot.jsonl`

| field | type | meaning |
|---|---|---|
| `instance_id` | str | pairing key, `"<task>/<index>"` |
| `task` | str | one of the six tasks below |
| `condition` | str | one of the seven primer conditions below |
| `style` | str | `zero_shot` or `zero_cot` |
| `prompt` | str | the exact text handed to the model, primer included |
| `gold` | str | the published answer, dataset formatting preserved |
| `nodes` | int | node count of the underlying graph, 5–19 |
| `edges` | int | edge count of the underlying graph |

`nodes` and `edges` describe the graph, not the answer — they are present on every
row regardless of task, and are what you group by to ask whether an effect depends
on graph size.

## `runs/*.jsonl`

| field | type | meaning |
|---|---|---|
| `instance_id`, `task`, `condition`, `style`, `gold` | | copied from the prompt row |
| `model` | str | model key, e.g. `gemma4-12b` or `gemma4-12b-think` |
| `response` | str | the generated text, prompt stripped |

**The schema is identical for both arms**; the arm is identified by the `model`
field carrying the `-think` suffix. There is no `prompt` field — join on
`(instance_id, condition, style)` against the prompt file to recover it.

Because `model` is on every row, **sharded files need no reassembly**:
`score_sweep.py` groups by that field, so pointing it at `runs/*.jsonl` pools the
shards correctly. Shard filenames are bookkeeping for resumable jobs, not
meaningful divisions of the data.

## `shortcuts.json`

A flat object keyed `"<task>/<condition>"` with a float score in [0, 1]: what a
deterministic program scoring only the rendered primer text achieves on that cell.
It is the bar a model result is read against — see
`docs/plans/shortcut-ceilings.md`.

All **42** cells are present (6 tasks x 7 conditions). Note that
`shortcut-ceilings.md` speaks of *36* cells: that is 42 minus the six `none`
cells, which are excluded there as trivial because an empty primer can only score
the majority-class baseline. They are still written to the file, and their values
are that baseline — `cycle_check/none` is 0.832, `node_count/none` 0.064 — which
makes them a useful sanity check rather than padding: if a `none` cell ever
differs from the baseline, the solver is reading something it should not.

## Vocabularies

**Tasks (6)** — `node_count`, `edge_count`, `node_degree`, `connected_nodes`,
`edge_existence`, `cycle_check`.

**Conditions (7)** — `none` (no primer, the control), `degree`, `clustering`,
`rwse`, `components` (one statistic each), `all` (every statistic), `filler` (a
primer of the same shape carrying no structural information, which separates
"primer present" from "primer informative").

**Styles (2)** — `zero_shot`, `zero_cot`. The thinking arm is `zero_shot` only.

**`gold` formatting varies by task** and keeps the dataset's own punctuation, so
compare through `graphtalk.scoring`, never by string equality:

```
node_count       ' 18.'
edge_count       ' 115.'
node_degree      '13.'
connected_nodes  '0, 1, 2, 4, 5, 6, 10, 11, 12, 13, 15, 16, 17.'
edge_existence   'No.'
cycle_check      'Yes, there is a cycle.'
```

## Caveats that travel with specific rows

These are properties of the data, not of the analysis, so they belong here:

- **`runs/smoke-gemma4-e4b.jsonl` carries `model: gemma4-e4b`.** Globbing
  `runs/*.jsonl` pools its 20 rows into that model's results. Exclude it.
- **350 thinking-arm rows never terminate** and are truncated at the token cap.
  They still *parse*, because the extractor finds an integer in the abandoned
  working, so they score as confident wrong answers rather than as missing. Their
  exact keys are in `analysis/truncated_keys.json`. Filter them before reporting
  accuracy: on `gemma4-12b-think` the difference is 81.2% against 99.1%.
- **`runs/*.redo.shard*.jsonl` are evidence, not answers.** 67 of those rows
  regenerated at a 4x larger cap; 76% still hit it. They exist to show the cap was
  not the cause. Do not merge them into the arm.
- **955 main-sweep rows were generated on CPU** before a driver mismatch was
  found — 438 `gemma4-e4b`, 426 `qwen3-14b`, 91 `qwen3-8b`. Greedy decoding means
  they should match GPU output, but this is unverified.
- **The two arms used different torch builds**, cu130 for the main sweep and
  cu126 for the thinking arm. Same version, same transformers, greedy throughout.

## Reproducing

```bash
python scripts/build_prompts.py --count 30            # -> prompts.jsonl
python scripts/shortcut_table.py --graphs 500 --json shortcuts.json
python scripts/score_sweep.py --responses $(ls runs/*.jsonl | grep -v smoke) \
                             --shortcuts shortcuts.json
```

`--count` takes a *prefix* of each split, so a larger value is a strict superset:
`--count 40` contains all 2,520 `--count 30` rows with byte-identical prompt text.
Existing responses therefore remain valid when the sweep is grown, and
`run_sweep.py` regenerates only what is missing.
