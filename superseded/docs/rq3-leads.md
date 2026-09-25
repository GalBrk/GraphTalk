# RQ3 follow-ups: what the `clustering` gain on `node_degree` is (and is not)

**Date:** 2026-09-18. **Status:** CPU-only analyses of existing runs; no new
generations. Reproduce with

```bash
PYTHONPATH=. python superseded/scripts/analyze_rq3_leads.py          # -> superseded/rq3_leads.json
PYTHONPATH=. python superseded/scripts/analyze_rq3_leads.py --test selection
```

Data: `qwen3-1.7b`, `node_degree`, the dedicated sweep (`degdens40`,
`degdens40hi`, `degdensfill`; 400 graphs per density, p = .10–.85) plus the
independent-seed replication (`degdensrep`, seed 20760906, p = .10–.50).
Run files store no prompt, so the script rebuilds every graph and queried
node from its seed with `build_size_sweep`'s recipe and asserts that the
rebuilt gold matches the stored gold. All 4,400 graphs match.

The finding under test (paper §5.4): `clustering` beats `none` by +1.9
points pooled over seven densities, and beats both `none` and `filler` at
p = .35 and .50. The two mechanisms the paper tested (the stated values
matter; mean clustering as a density prior) both failed.

## 1. Selection: is the effect an artifact of how the cell was chosen?

| Quantity | Value |
|---|---|
| Non-answer, no-route cells in the main-sweep screen (4 arms × task × condition) | 40 (160 per density) |
| Dedicated-sweep graphs identical to main-sweep graphs | **700 of 2,800** (indices 0–99 at every density) |
| `clustering` − `none`, the 700 screened graphs | +1.3 [−1.6, +4.3], p = .47 |
| `clustering` − `none`, the 2,100 unscreened graphs | **+2.1 [+0.2, +3.9], p = .031** |
| `clustering` − `filler`, unscreened graphs | +4.8 [+3.0, +6.5], p = 8×10⁻⁸ |
| Replication (fresh seed), `clustering` − `none` | +4.2 [+1.9, +6.8], p = .00055 |
| Replication p, Bonferroni over all 40 screened cells | **.022** |

**Result.** The dedicated sweep is not an independent sample: a quarter of
its graphs are the main sweep's own graphs, because both were built with
the same seed and keys. This does not inflate the effect, which is *larger*
on the unscreened graphs. The fresh-seed replication survives a correction
for every cell the screen could have picked. So the pooled effect is real,
not a product of selection.

**Per-density claims do not survive correction.** Holm across the 14
per-density tests (7 densities × {vs `none`, vs `filler`}):

| Test | Δ | p | Holm p |
|---|---|---|---|
| p=.65 vs filler | +7.5 | 2.4×10⁻⁵ | .0003 |
| p=.75 vs filler | +6.8 | 4.2×10⁻⁵ | .0005 |
| p=.35 vs none | +7.5 | .0089 | .107 |
| p=.50 vs filler | +6.3 | .022 | .22 |
| p=.35 vs filler | +6.5 | .027 | .24 |
| p=.50 vs none | +4.3 | .16 | 1.0 |

The two tests that survive, at p ≥ .65, reflect `filler`'s own cost, not a
`clustering` gain (`clustering` − `none` there is −2.0 to +0.5). The earlier
paper wording, that `clustering` "exceeds both controls only at p = .35
and .50", rested on uncorrected per-density p-values.

**What the data do support** is one pooled test over the intermediate
range (`selection.mid_pooled` in `superseded/rq3_leads.json`; this is what the paper
now reports):

| p = .35 and .50 pooled | Δ | 95% CI | p | n |
|---|---|---|---|---|
| `clustering` − `none` | +5.9 | [+2.1, +9.8] | .0036 | 800 |
| `clustering` − `filler` | +6.4 | [+2.8, +10.0] | .0012 | 800 |
| `filler` − `none` | −0.5 | [−3.9, +3.1] | .84 | 800 |
| replication seed, `clustering` − `none` | +5.5 | [+1.5, +9.5] | .0083 | 800 |

The range was chosen after seeing the default seed. The replication
seed played no part in choosing it, so it is the out-of-sample test of the
range, and it holds. The per-density estimates are too imprecise to narrow
the range further.

## 2. Heterogeneity: which instances gain?

Paired `clustering` − `none`, split into within-density terciles of a
property of the queried node (so density cannot drive the split). The
default seed covers all seven densities; "mid" is p = .35 and .50.

| Queried node | low tercile | mid tercile | high tercile |
|---|---|---|---|
| **list position (node id)**, all densities | **+5.0** [+2.1, +7.8] | +0.8 | −0.1 |
| list position, mid densities | **+11.6** [+5.2, +18.3] | +3.4 | +2.6 |
| list position, **replication seed** (held out) | **+7.5** [+3.4, +11.4] | +3.0 | +2.3 |
| degree relative to expectation (z), all | 0.0 | +3.0 | +2.7 |
| own clustering − graph mean, all | −0.2 | +3.9 | +2.0 |

OLS on all densities with density fixed effects (HC1 errors, per 10
positions / per unit z / per 0.1 clustering): position −1.7 (p = .018),
degree z +2.0 (p = .012), own clustering +0.7 (p = .50).

Controls, the same position split:

| | low | mid | high |
|---|---|---|---|
| `filler` − `none` | −1.5 | −3.8 | −1.9 |
| `components` − `none` | −0.6 | −0.6 | +0.1 |
| `none` accuracy (%) | 38.0 | 37.3 | 38.6 |

**Result.** The gain is concentrated on nodes **early in the list** (ids
0–13), whose sentences come first in both the primer and the encoding. The
pattern was found on the default seed and **replicates on the held-out
seed**. It is specific to `clustering`: `filler`, which also names every node
in the same order, and `components` show no position pattern. It is not
where `none` is weakest, since `none` accuracy is flat across position. The
queried node's own clustering value does not predict the gain, which is
consistent with the rewiring null: the model does not seem to use the value
it is given.

## 3. Error shape

Per condition, share of all responses (non-truncated), default seed.

| Density band | Condition | acc | off by 1 | off by 2 | off by ≥3 | under | over | mean signed |
|---|---|---|---|---|---|---|---|---|
| mid (.35–.50) | none | 34.4 | 37.8 | 14.8 | 13.1 | **51.0** | 14.6 | −0.21 |
| | clustering | **40.2** | 35.5 | 12.1 | 12.1 | **42.5** | 17.2 | +0.12 |
| | filler | 33.9 | 34.8 | 11.6 | 19.8 | 43.8 | 22.4 | +0.72 |
| | components | 33.0 | 38.6 | 12.5 | 15.9 | 47.4 | 19.6 | −0.14 |
| high (.65–.85) | none | 9.4 | 14.7 | 13.0 | 62.9 | 17.5 | 73.1 | +3.60 |
| | clustering | 8.8 | 15.7 | 13.0 | 62.6 | 19.8 | 71.5 | +3.35 |
| | filler | 3.0 | 7.3 | 6.8 | **82.8** | 5.5 | 91.5 | +6.56 |

**Result.** At intermediate density the model mostly **undercounts** under
`none` (51% of answers are too low). `clustering` cuts undercounting by 8.5
points and converts it into correct answers, not into large errors. `filler`
cuts undercounting by a similar amount but converts it into overcounting and
large misses, so accuracy does not improve. At high density, `filler`'s cost
is inflated overcounting.

## 4. Response behaviour: transcription or counting?

Every response writes out a neighbour list before answering ("Node v is
connected to …"). The script parses the first such list and classifies each
error: **bad transcription** (the written list is not the true neighbour
set) or **miscount** (list correct, count wrong).

| mid densities | correct | bad transcription | miscount | median tokens |
|---|---|---|---|---|
| none | 34.4 | 54.9 | 10.8 | 176 |
| clustering | 40.2 | **47.1** | 12.6 | **204** |
| filler | 33.9 | 56.5 | 9.6 | 147 |
| components | 33.0 | 52.4 | 14.6 | 172 |

Among bad transcriptions at mid density, the share that only *drop*
neighbours (omission) falls from 65% (`none`) to 52% (`clustering`), and the
mean number of true neighbours missing from a response falls from 1.49 to
0.86 (`filler` 1.05, `components` 0.94). Paired, all densities:
`clustering` fixes 224 transcription and 61 counting errors that `none`
made, and introduces 176 and 57. The net gain (+48 vs. +4) comes almost
entirely from **transcription**. No response under any condition mentions
clustering.

**Result.** The model's bottleneck is copying the queried node's line from
the encoding, not counting. Its typical error is dropping neighbours from
the line. Under `clustering` it writes fuller, longer neighbour lists (+28
median tokens). The effect is on retrieval of the queried node's line, not
on reasoning with the clustering values.

## Leads

Taken together: the `clustering` gain is real (it survives selection and
replicates), but it looks like a **retrieval aid for early-listed nodes**,
not use of clustering information. Candidate explanation: a numeric,
per-node primer that repeats "Node v …" in encoding order helps the model
find and copy a node's line, mostly for lines early in a long prompt.
`filler` repeats the node names too, without numbers, and does not help.

GPU tests that would confirm or kill this, in priority order. Full design,
time estimate and CPU prep in `docs/plans/rq3-gpu-tests.md`:

1. **Shuffled-value primer** (same template and values, permuted across
   nodes). Predicts the same gain if the mechanism is format/retrieval, and
   also the same position gradient.
2. **Reversed-order primer** (node 39 first). If the gain follows the
   primer's first sentences, it should move to high-id nodes. If it stays
   on low ids, it is tied to the encoding position instead.
3. **Numeric placebo** (random values in [0,1], same template). Separates
   "per-node numbers" from "real statistics".
4. **Primer after the encoding.** A retrieval-anchor account predicts the
   gain changes with where the index sits relative to the lines it indexes.
5. **Pre-registered confirmation** at p = .35/.50 on a fresh seed, with the
   position interaction as the primary test (fixed in advance, so it is not
   subject to the 14-test Holm problem above).

CPU follow-ups not done here: the same position split on `qwen3-4b`'s
high-density cells (densfull40hi, 100 graphs per cell, likely
underpowered); and transcription analysis on `connected_nodes`, which asks
for the same neighbour list and so is a direct test of the retrieval
account.

## Caveats

- The position, degree and clustering splits are post hoc (3 covariates ×
  2 comparisons × 2 scopes). The position pattern is the only one tested
  out of sample (replication seed), and it held.
- The transcription parser reads the *first* neighbour list a response
  writes. Parser checks are in `superseded/tests/test_analyze_rq3_leads_superseded.py`. It
  classifies every non-truncated response in these runs, but a response
  that revises its list later is judged on the first version.
- One model, one task, Erdős–Rényi graphs only.
