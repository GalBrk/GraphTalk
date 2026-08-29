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

- **Anything under `runs/archive/` is not part of the sweep.** It holds the smoke
  test (20 rows carrying `model: gemma4-e4b`) and the 4x-cap regeneration probe.
  `runs/*.jsonl` no longer matches them, and `graphtalk/analysis.py` excludes the
  directory outright rather than matching on filenames.
- **679 rows never terminate** and are truncated at the token cap — 309 in the
  thinking arm and **370 in the main sweep**, which was previously believed to have
  none. Every row now carries `hit_cap`, measured on one instrument (see
  `scripts/backfill_hit_cap.py`); the main-sweep rows concentrate at `zero_cot`,
  which is given half the token budget, and materially inflate the `zero_cot`
  penalty reported in `docs/sweep-findings.md`.
  They still *parse*, because the extractor finds an integer in the abandoned
  working, so they score as confident wrong answers rather than as missing. Their
  Of those, 271 predate per-row token counts and their exact keys are in
  `analysis/truncated_keys.json`; the other 45 are recorded directly on the row as
  `hit_cap`, and `graphtalk/analysis.py` prefers that when present. Filter them
  before reporting accuracy: on `gemma4-12b-think` the difference is 81.2% against
  99.1%. The count fell from 350 because the reworded `filler` primer roughly
  halved non-termination in that condition -- see `docs/sweep-findings.md`.
- **`runs/archive/*.redo.shard*.jsonl` are evidence, not answers.** 67 of those
  rows regenerated at a 4x larger cap; 76% still hit it. They exist to show the cap
  was not the cause. Do not merge them into the arm.
- **955 main-sweep rows were generated on CPU** before a driver mismatch was
  found — 438 `gemma4-e4b`, 426 `qwen3-14b`, 91 `qwen3-8b`. Greedy decoding means
  they should match GPU output, but this is unverified.
- **The two arms used different torch builds**, cu130 for the main sweep and
  cu126 for the thinking arm. Same version, same transformers, greedy throughout.
- **2,880 rows are being regenerated under reworded prompts.** The `filler` primer
  and the `edge_existence` question were both reworded in `d7cdcf7..3545662`, so the
  360 affected `zero_shot` rows in each of the eight arms are generated from prompts
  the rest of the sweep never saw. The 1,440 affected rows in the obsolete `zero_cot`
  prompt style are **not** regenerated and keep the original wording — so
  `condition: filler` does not mean one thing across the whole sweep, and the frame
  carries a `wording` column to say which is which. Never pool the two.
- **`prompts.original-wording.jsonl` is what produced those un-regenerated rows.**
  `prompts.jsonl` now holds the revised wording throughout, including for the 360
  `zero_cot` rows whose responses were *not* regenerated — so for those rows the
  tracked prompt file no longer matches the tracked response. This file closes that
  gap: it is the 360 original-wording `zero_cot` prompts, verbatim, and it is the
  prompt of record for the 1,440 responses (360 x 4 plain models) that keep the old
  text. Rebuilding it from `build_prompts.py` is not possible; it exists because a
  response is only interpretable against the exact prompt that produced it.

## A prediction about the `edge_existence` rewording, recorded in advance

The question was reworded from *"Is node A connected to node B?"* to *"Does an edge
exist between Node A and Node B?"* on the argument that "connected to" is ambiguous
with reachability. Measured across all 30 `edge_existence` instances before the
rewording, that ambiguity was real but inert: 12 instances have gold `Yes`, 14 have
gold `No` with a path present, and 4 are genuinely unreachable — so 47% of instances
would flip under a reachability reading. On the 14 path-only pairs, where that reading
predicts a ~100% `Yes` rate, the eight models answered `Yes` on **5% to 34%**
(`gemma4-12b` 5%, `qwen3-8b` 34%). All eight already resolved the question the way the
gold does.

**So the prediction is that the rewording moves `edge_existence` accuracy very little.**
This is written down before the regenerated rows exist. If accuracy is flat, the
ambiguity was not what drove error on this task and the rewording bought comparability
loss for nothing; if it moves, the measurement above was wrong about what the models
were doing. Either outcome is a result — but only because the claim was staked first.

### Outcome: the prediction was wrong, and instructively so

Recorded 2026-08-29, after all eight arms were regenerated. **Accuracy moved a lot** —
mean **+10.2 points**, from +1.7 on `gemma4-12b` to +18.9 on `qwen3-8b`, with the
path-only "Yes" rate falling to exactly **0.0% in every arm**. The rewording was
justified.

The measurement above was not wrong; the inference from it was. Each of the numbers in
that table is reproduced by the re-run's baseline to within a couple of points, and the
gradient it noticed — weaker models drifting toward the reachability reading — is
exactly what predicts the effect size, at **r = +0.95** across the eight arms. What was
wrong was treating a 5–34% residual as negligible. On `gemma4-12b`, whose rate is 5%,
the prediction holds precisely: +1.7 points, near enough to nothing. The error was
generalising the strongest model's behaviour to a sweep whose error mass sits in the
weaker ones.

The narrower claim in that table survives intact: all eight models *do* resolve
"connected" as adjacency most of the time. It just turns out that "most of the time" is
where a tenth of the task's accuracy was hiding. Full analysis in
`docs/sweep-findings.md`; reproduce with `scripts/rewording_effect.py`.

## Reproducing

> Since the rewording, `build_prompts.py` reproduces `prompts.jsonl` exactly, but
> **not** the prompts behind every tracked response: the 1,440 un-regenerated
> `zero_cot` rows came from `prompts.original-wording.jsonl`. See the caveat above.

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
