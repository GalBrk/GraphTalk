# Where primers actually help, and why the design has not yet been able to tell

Written 2026-09-05, after adding `qwen3-0.6b`/`qwen3-1.7b`/`qwen35-2b` to the
suite and probing at `--count 100` on the `none`/`degree` conditions.

This document re-reads existing sweep data through the shortcut bar and reaches
a conclusion about the *design* rather than about primers.

> **Analysis pitfall this document was initially wrong about.** Scoring
> `runs/*.jsonl` by calling `scoring.extract_answer` directly is **incorrect for
> the 28 `.got.` files**: a GoT-named response answers in character names
> ("Maester, Catelyn") while `gold` stays integers, so it must go through
> `node_naming.desubstitute_response` first (see CLAUDE.md, "node_naming.py").
> Pooling them undesubstituted corrupts `connected_nodes` specifically -- the
> only task whose *answer* is a node list -- and made `qwen3-8b` appear to score
> 0.080 there when it actually scores 0.996. Every number below **excludes
> `.got.` files**. Anyone writing a fresh analysis script will hit this trap;
> `scripts/build_sweep_frame.py` is the supported path that handles it.

## Summary

1. **Five of six tasks are saturated.** Mean `none` accuracy across
   `qwen3-8b`/`qwen3-14b`/`gemma4-e4b`/`gemma4-12b`: `node_degree` 1.000,
   `connected_nodes` 1.000, `edge_existence` 1.000, `node_count` 0.992,
   `cycle_check` 0.975. Against Fatemi et al.'s 18.8% for PaLM 2 on
   `node_count`, the published setup no longer discriminates.

2. **`edge_count` is the only task with headroom** -- 0.670 mean, and only
   0.400-0.433 for `qwen3-8b`/`qwen3-14b`.

3. **`edge_count` is also where every large primer gain lives, and its `degree`
   bar is 1.00.** sum(degrees)/2 *is* the edge count, so a regex over the primer
   text scores 100% without seeing the graph. The four biggest effects in the
   project (+0.55, +0.45, +0.23, +0.20) are all there, and not one beats the
   regex solver.

4. **The one task with headroom does have clean conditions, and they were never
   powered.** `edge_count` x {`components` 0.02, `rwse` 0.02, `clustering` 0.15}
   are uncontaminated, show positive effects (+0.03 to +0.13), and sit at n=30
   with 1-7 discordant pairs.

The honest status of the proposal's question is not "primers do not help" --
that would be an unpowered null -- but **"the experiment that could answer it
has not been run."** Point 4 says exactly which experiment.

## The shortcut bar decides which cells are informative

`shortcuts.json` (`scripts/shortcut_table.py --graphs 500`) scores a primer-only
solver that never sees the graph:

| task | components | clustering | rwse | degree | all | filler | none |
|---|---|---|---|---|---|---|---|
| node_count | 0.08 | 1.00 | 1.00 | **1.00** | 1.00 | 1.00 | 0.06 |
| edge_count | **0.02** | **0.15** | **0.02** | **1.00** | 1.00 | 0.02 | 0.02 |
| node_degree | 0.08 | 0.08 | 0.62 | **1.00** | 1.00 | 0.08 | 0.08 |
| connected_nodes | 0.08 | 0.08 | 0.08 | 0.21 | 0.35 | 0.08 | 0.08 |
| edge_existence | 0.50 | 0.72 | 0.64 | 0.79 | 0.79 | 0.50 | 0.50 |
| cycle_check | 1.00 | 0.83 | 0.83 | 0.95 | 0.95 | 0.83 | 0.83 |

**`degree` is the worst condition for testing the hypothesis and the one the
sweep has most data on.** It gives the answer away on three of six tasks.

Cross that against headroom and only one row survives: **`edge_count`**, which
has both real headroom and three clean conditions.

## Every large primer gain is shortcut-explainable

`none` vs `degree`, GoT and capped rows excluded:

| model | task | none | degree | delta | n | disc | p | bar | beats bar? |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b-think | edge_count | 0.324 | 0.873 | +0.549 | 71 | 39 | 0.0000 | 1.00 | no |
| qwen3-1.7b | edge_count | 0.237 | 0.691 | +0.454 | 97 | 48 | 0.0000 | 1.00 | no |
| qwen3-8b | edge_count | 0.433 | 0.667 | +0.233 | 30 | 9 | 0.039 | 1.00 | no |
| qwen3-14b | edge_count | 0.400 | 0.600 | +0.200 | 30 | 8 | 0.070 | 1.00 | no |
| qwen3-1.7b | node_degree | 0.884 | 0.953 | +0.070 | 86 | 8 | 0.070 | 1.00 | no |
| gemma4-12b | edge_count | 0.933 | 1.000 | +0.067 | 30 | 2 | 0.500 | 1.00 | ties |
| qwen3-0.6b | connected_nodes | 0.834 | 0.875 | +0.041 | 100 | 8 | 1.000 | 0.21 | YES |

Highly significant, reproducible across five models spanning 0.6B-14B, and
uninterpretable as a reasoning claim: the primer contains the answer.

## The clean cells: positive, and never powered

Cells with bar < 0.35, GoT excluded:

| model | task | cond | none | primer | delta | n | disc | p | bar |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | edge_count | clustering | 0.433 | 0.567 | +0.133 | 30 | 6 | 0.219 | 0.15 |
| qwen3-14b | edge_count | components | 0.400 | 0.500 | +0.100 | 30 | 3 | 0.250 | 0.02 |
| qwen3-14b | edge_count | clustering | 0.400 | 0.467 | +0.067 | 30 | 2 | 0.500 | 0.15 |
| qwen3-0.6b | connected_nodes | degree | 0.834 | 0.875 | +0.041 | 100 | 8 | 1.000 | 0.21 |
| qwen3-8b | edge_count | components | 0.433 | 0.467 | +0.033 | 30 | 7 | 1.000 | 0.02 |
| qwen3-14b | edge_count | rwse | 0.400 | 0.433 | +0.033 | 30 | 1 | 1.000 | 0.02 |

Every top cell is `edge_count`. Effects are positive and clear their bars.
**Every one is at n=30 with 1-7 discordant pairs.** Power and clean cells are in
disjoint places: `degree` was scaled to n=500, the clean conditions never were.

## The recommended experiment

**`edge_count` x {`components`, `clustering`, `rwse`} vs `none`, at n=500, on
`qwen3-8b` and `qwen3-14b`.**

Everything lines up on this cell and nothing else does:

- the only task with headroom (0.400-0.433 baseline for these two models);
- bars of 0.02-0.15, so a gain cannot be shortcut-explained;
- observed effects already positive (+0.03 to +0.13) at n=30;
- discordance is high there (6-9 pairs at n=30, i.e. 20-30%), so n=500 projects
  to ~100-150 discordant pairs per cell -- comfortably powered, unlike anything
  in the current sweep.

`degree` should be dropped from the headline. Keep it as a documented positive
control for shortcut exploitation, which it demonstrates unusually well.

## Retracted from the first draft of this document

- ~~"`connected_nodes` scoring is broken; a 0.6B beats an 8B tenfold"~~ -- an
  artifact of pooling GoT rows without desubstitution. `qwen3-8b` scores 0.996
  and `qwen3-14b`/`gemma4-*` score 1.000. The task is saturated, not broken.
- ~~"`connected_nodes` is the only uncontaminated task with headroom; target
  it"~~ -- it has no headroom. `edge_count` is the target.
- ~~"qwen3-8b connected_nodes +0.053 at n=500, p<0.0001"~~ and ~~"qwen3-8b-think
  connected_nodes + rwse +0.149"~~ -- both GoT artifacts; gone after exclusion.
- The `sweep-findings.md:368` note about "resolving connected as adjacency"
  concerns **`edge_existence`**, not `connected_nodes`. Mis-cited here initially.

## The new small models

Probe: `prompts.count100.none_degree.jsonl`, 6 tasks x 100 graphs x
{none, degree}, 1,200 rows per arm, tag `probe100`.

`qwen3-0.6b` plain (95.5 min generation, 0.2% unparsed, 3 capped):

| task | none | degree | delta | disc | p |
|---|---|---|---|---|---|
| node_count | 0.900 | 0.550 | -0.350 | 39 | 0.0000 |
| edge_count | 0.070 | 0.090 | +0.020 | 14 | 0.79 |
| node_degree | 0.890 | 0.810 | -0.080 | 22 | 0.13 |
| connected_nodes | 0.834 | 0.875 | +0.041 | 8 | 1.00 |
| edge_existence | 0.680 | 0.620 | -0.060 | 10 | 0.11 |
| cycle_check | 0.856 | 0.814 | -0.041 | 12 | 0.39 |

pooled none 0.705 / degree 0.625 (**-0.079**); excluding the `node_count`
artifact below, **-0.025**.

`qwen3-0.6b` think (6 shards, 3.0% capped): pooled none 0.843 / degree 0.920
(**+0.078**), driven by `edge_count` 0.324 -> 0.873.

**The same model, same prompts, same instances: the primer hurts by 7.9 points
without the reasoning channel and helps by 7.8 with it.** No larger model can
show this -- all four are at ceiling in both arms. That contrast is the main
reason to keep a sub-2B model in the suite.

### The off-by-one artifact (do not report as a primer effect)

`qwen3-0.6b`'s -0.350 on `node_count` is not a reasoning failure. Of 100
`degree` rows: 55 exact, 45 off by exactly -1, **zero other error types**;
corrected for the convention it scores 1.000. The model identifies the node set
correctly every time and reports the highest node *label* (N-1) instead of |V|.

Confirmed two ways:

- The think arm verbalises the missing step on rows it gets right: *"The nodes
  are numbered from 0 to 17 ... so that's 18 nodes in total."*
- The rate scales with size: **0/31 (0%) below 10 nodes, 19/41 (46%) at 10-15,
  26/28 (93%) at 16+**. A pure labelling confusion would be flat; this is a
  counting-capacity limit, with the primer's per-node enumeration supplying the
  salient wrong answer.

**The existing sweep is not contaminated**: off-by-one rate on `node_count` is
0% for `gemma4-12b`, `gemma4-e4b`, `qwen3-8b` (560 rows) and `qwen3-14b`;
2-3% for `gemma4-e4b-think`. Same genre as the three cases in
`sweep-findings.md` where an artifact impersonated a finding.

`qwen3-0.6b` is genuinely too weak on `edge_count` in the plain arm (0.070/0.090,
diffuse errors), though its think arm reaches 0.324/0.873 on the same task.

### `qwen3-1.7b` clears the artifact and is the better suite member

Complete at 1,200 rows (0.4% capped, 0.3% unparsed):

| task | none | degree | delta | disc | p | bar |
|---|---|---|---|---|---|---|
| node_count | 0.950 | 0.960 | +0.010 | 7 | 1.000 | 1.00 |
| edge_count | **0.237** | 0.691 | +0.454 | 48 | 0.0000 | 1.00 |
| node_degree | 0.890 | 0.960 | +0.070 | 9 | 0.039 | 1.00 |
| connected_nodes | 0.963 | 0.984 | +0.021 | 3 | 1.000 | 0.21 |
| edge_existence | 0.890 | 0.890 | +0.000 | 8 | 1.000 | 0.79 |
| cycle_check | 0.888 | 0.918 | +0.031 | 5 | 0.375 | 0.95 |

pooled none 0.804 / degree 0.900 (**+0.096**).

**The off-by-one rate drops to 3% under `degree` (from 45%) and 5% under `none`
(from 10%).** This is direct confirmation of the counting-capacity explanation
above: 1.7B clears the limit the 0.6B falls off past ~15 nodes, and `node_count`
becomes an interpretable cell (+0.010) instead of an artifact reported as a
-0.350 primer effect.

**`qwen3-1.7b` has more `edge_count` headroom than `qwen3-8b`** -- 0.237 against
0.433 -- with a clean parse rate and no artifact. On the one task that is not
saturated it is the weakest baseline in the suite, which makes it the strongest
candidate for the n=500 clean-condition experiment, and roughly 4x cheaper per
row than `qwen3-8b`. Not yet submitted: the value of adding an arm depends on
whether the clean conditions produce a measurable effect at all, which
`ec500` will answer first.

## Operational notes

- **`--mem=64G` in `sweep.sbatch` is sized for Qwen3-14B's 29.6 GB checkpoint.**
  For a 1.2 GB model it cut eligible nodes from 5 to 1 and cost ~45 min of queue
  time. Right-size per model (`--mem=16G --cpus-per-task=4` worked).
- **n-501 has a 535.x driver (CUDA 12.2); the default cu130 env cannot use it.**
  `sweep.sbatch`'s comments name only n-802/803/804. Three shards died there in
  94 s (the driver guard working). Use `GRAPHTALK_ENV=graphtalk-cu126` or
  `--exclude=n-501`.
- **Use an ODD `--array` shard count.** With 2 conditions the stride preserves
  parity, so an even count sends `none` to even shards and `degree` to odd ones;
  partial progress is then unpaired and mid-run comparisons are meaningless.
- **`--array=1,2,4` does NOT mean "shards 1, 2 and 4 of 5".**
  `SLURM_ARRAY_TASK_COUNT` is the number of tasks (3), so `sweep.sbatch` computes
  `NSHARDS=3` and each task strides `records[i::3]`. Resubmitting failed shards
  that way produced 28 wrongly-strided rows, 12 of them duplicating rows owned by
  the surviving `*of5` shards. Pass `SLURM_ARRAY_TASK_ID`/`SLURM_ARRAY_TASK_COUNT`
  explicitly on separate non-array jobs instead; see `cluster/README.md`.
- **Page-cache warm-up cost ~44 min for a 1.2 GB checkpoint on n-602** vs ~9 min
  on n-502. It exists for gemma4-e4b's 16 GB checkpoint over NFS; at this size
  it plausibly costs more than it saves.
- **Qwen3.5 needs `AutoModelForImageTextToText`** -- declares
  `Qwen3_5ForConditionalGeneration`, multimodal even at 2B, like Gemma 4.
  Supported by the env's `transformers` 5.15.0.

## Caveats

- The `--count 100` probe corpus is a strict superset of the tracked `--count
  30` corpus (same split, first-N prefix), so instance ids are comparable, but
  it covers only `none` and `degree`.
- p-values are per (model, task) pooled over the instances shown, not the
  proposal's 36-cell grid. Discordance *rates* compare across documents; cell
  counts do not.
- `qwen3-1.7b` and `qwen35-2b` are complete (1,200 rows each).
  `qwen3-1.7b-think` was still generating when this was written.
  **`qwen35-2b-think` was cancelled at 73/1200 rows** -- it could only have
  confirmed a `degree`-condition result that is shortcut-explained anyway, and
  the plain-vs-think contrast it would have tested is covered by
  `qwen3-1.7b-think` at a size where the off-by-one artifact is absent. Its
  partial rows are in `runs/archive/cancelled-*` so they cannot reach a frame.
- The `edge_count` n=500 experiment (`ec500`) runs as jobs 853382+853499
  (`qwen3-8b`, 5 shards) and 853383 (`qwen3-14b`, 5 shards), all pinned to
  `--constraint=a6000` after three separate 535.x-driver failures.
- All GoT-named runs are excluded throughout. Re-including them requires
  desubstitution and would mainly affect `connected_nodes`.
