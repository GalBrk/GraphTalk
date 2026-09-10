# The shared ladder and the rewiring experiment

Design notes for `graphtalk/{rewiring,cell_screen,ladder}.py` and
`scripts/{build_ladder,analyze_ladder,analyze_rewiring_sweep}.py`. Read
`primer-effects-and-power.md` first — this builds on its findings and does not
restate them.

## What problem this solves

Earlier attempts to vary graph structure varied it by swapping the *generator*
(ER vs BA vs regular vs Watts–Strogatz vs SBM). That moves three things at once:
the shape, the degree distribution, and therefore both the `node_degree` gold
answers and `maj_base` — what a blind guesser scores. Measured across five
families at matched `(n, m)`, `maj_base` ranged **0.15 to 1.000**, a 6× swing
larger than any primer effect on record. Random regular graphs are the degenerate
case: every answer is the same integer, so a guesser scores 1.000.

**A family contrast is therefore not a structure contrast**, and a 72-cell
family grid was withdrawn on that basis.

## Degree-preserving rewiring

A double-edge swap removes `(a,b)` and `(c,d)` and adds `(a,d)` and `(c,b)`.
All four endpoints keep their degree, so:

- the degree sequence is preserved **exactly** — every gold answer and
  `maj_base` are unchanged;
- `n` and `m` are unchanged, and the `incident` encoder emits one line per node,
  so the rendered prompt is the **same string length**;
- what moves is the triangle count, which is exactly what `clustering` reports.

Measured at n=80, m=334: transitivity **0.000 → 0.630**, rendered clustering
spread 0.005 → 0.155, with `maj_base`, degree sequence and prompt length all
identical. `tests/test_rewiring.py` asserts each of those.

This gives a **within-instance paired design**: the same graph, rewired, same
question, same answer. It is what defuses the length confound *by construction* —
`filler` prices prompt length at −11.7pp on dense graphs, larger than most primer
effects, so a design that lets length drift inherits that confound.

### Two implementation points that are correctness, not tuning

**Acceptance uses the local triangle delta**, not a global `nx.triangles`
recount. The common-neighbour counts of the two removed and two added pairs give
the identical decision in O(deg) instead of O(m·k̄). This matters because a
global recount is slow enough to force a flat swap cap — and a flat cap
under-rewires large graphs so badly that a first screen wrongly concluded dense
cells could not carry an informative primer:

| k̄ | n | raw | flat cap 300–400 | ×3 of \|E\| |
| --- | --- | --- | --- | --- |
| 8 | 120 | 0.053 | 0.091 (rejected) | **0.210** |
| 16 | 40 | 0.048 | 0.088 (rejected) | **0.121** |
| 16 | 80 | 0.037 | — | **0.152** |

**Swap budget is a multiple of `|E|`** (`DEFAULT_MULT = 3`, measured — the curve
is flat past ×3). k̄=16 is the *magnitude-limited* regime where primers can
actually help, and the flat cap had excluded exactly it.

## The screen: three ways a cell is worthless

`cell_screen.py`, all computable before any GPU time:

- **blind** — `maj_base` > 0.25: a guesser wins too often for an effect to be legible.
- **silent** — `clu_sd` < 0.10 on the **rendered 2-dp** value: the primer is a
  constant string, and no sample size makes a constant informative.
- **unreadable** — prompt beyond the model's measured reading limit.

`clu_sd` is reported raw *and* rewired, because rewiring is part of the design:
screening on the raw value rejects precisely the cells worth running.
`n_graphs=1` is refused — single-seed screening flipped two cells across the
`maj_base` bar during development.

## The ladder

`base30` is 1.000 for qwen3-8b, gemma4-e4b, gemma4-12b and qwen3-14b at once —
saturated from 8B up. But `size20/40/80` puts qwen3-1.7b in its band at size40
(0.440) and qwen3-8b at size80 (0.479), on the *same graphs*. So: **one corpus,
every model runs all of it, each model analysed only on its own rungs.**

18 rungs, 1,691 → 15,136 tokens, over two axes — `n` (length) and `k̄`
(answer magnitude) — because the two produce difficulty with **opposite**
implications for primers:

- **Length-limited**: the model cannot read the graph, and cannot use a primer
  either. Where the plain arm collapses, the `degree` control that writes the
  answer *verbatim* into the prompt is worth **+0.7pp**.
- **Magnitude-limited**: the model reads everything and miscounts — the work a
  primer short-circuits. The same control is worth **+6.8pp** there.

A one-dimensional size ladder can only ever produce the first. Hence two axes.

Two boundaries from a 45-cell screen over n ∈ 20..400, k̄ ∈ 4..24:

- **n=20 is dead at every degree** (`maj_base` 0.287–0.325).
- **Structure stops binding above n≈120** — nearly every large cell passes both
  structural screens; what excludes them is reading limit and context window.
  Big graphs are structurally fine and simply too long.

Note that **adding triangles or cycles does not lengthen the prompt** — that is
the rewiring invariant. Prompt length is a function of `(n, m)` alone.

## Running it

```bash
cluster/run_ladder.sh              # probe + ladder screen, both arms
cluster/run_ladder.sh --dry-run
```

`sweep.sbatch` exports `HF_HUB_OFFLINE=1`, so a model whose weights are not
under `.cache/hub/` **cannot run** — it fails on the node after queueing, not at
submit time. Only `Qwen/Qwen3-1.7B` is cached here; adding a model means
fetching its checkpoint on the login node first.

Analysis:

```bash
PYTHONPATH=. python scripts/analyze_ladder.py --responses 'runs/*.ladder_screen.jsonl' \
    --reading-limits qwen3-1.7b=<from probe> --out analysis/ladder_matrix.csv
PYTHONPATH=. python scripts/analyze_rewiring_sweep.py --responses 'runs/*.rewire.jsonl'
```

`analyze_rewiring_sweep.py` asks **"does `none` move across rewiring levels?"
first and reports it first**, because that decides what the primer result means:
`none` flat → evidence about primer *content*; `none` moving → "clustering
changes the task", real but a different claim. Deciding that after seeing the
primer result would be choosing the interpretation to fit the answer.

## Gotchas

- **Gold carries a trailing period** in some corpora (`'13.'`), which silently
  zeroes a naive string comparison — it made four models look like they scored
  0.000 during development. Both analysis scripts normalise.
- **Shard count must be coprime with the condition count.** Records cycle
  through conditions innermost, so a 3-shard array on 3 conditions gives shard 0
  every `none` row and nothing else. The rewire stage has 3 conditions — use 5 or 7.
- **Reading limits are per model AND per arm.** `analyze_ladder.py` refuses to
  substitute a default; reusing one model's limit for another is the error the
  whole design exists to avoid.
