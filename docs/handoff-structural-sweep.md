# Handoff: structural-difficulty sweep design

Written 2026-09-09 from a design session on a Windows laptop; **re-verified 2026-09-10
on the TAU CS cluster** (`/home/dcor/avivyossef/inbal/GraphTalk`, as user `avivyossef`)
against the actual repo and environment. Every number below was re-measured there unless
marked otherwise. Corrections found during that pass are marked **[CORRECTED]**; the
environment section (§9) changed substantially and now describes what is actually on the
box rather than what the laptop session assumed.

## 1. What this session was doing

Designing — not yet building — an experiment to find which **structural properties of
the input graph** make a GraphQA task hard enough that Qwen3-1.7B fails often *without*
a primer, so that adding a primer yields a measurable, statistically defensible lift.

Two arms are treated as separate conditions throughout: `qwen3-1.7b` (plain) and
`qwen3-1.7b-think`.

**Status:** repo investigation complete, four empirical findings established, build plan
drafted, **no code written**. Three decisions are still open (§4).

## 2. Decisions already locked

| Decision | Answer |
| --- | --- |
| Repo access | Read whatever's needed |
| How the model runs | Cluster: build prompts on login node, `sbatch` on GPU node, score back on login node |
| Token limits | Qwen3 published defaults — and the repo already agrees: `models.py` has `max_context_tokens=32768`, `max_new_tokens` 8192 plain / 16384 think |
| Task scope | A small task panel of 2–3. Proposed and accepted: `node_degree` (local, well-characterized), `cycle_check` (the only globally-structural task), `connected_nodes` (set-valued, F1-scored) |
| Stats defaults (stated, not asked) | alpha = 0.05 two-sided, power = 0.80, MDE = 15pp on primer lift — all to be configurable |

## 3. The four findings, with reproduction

Run these from the repo root, with the cluster env **by absolute path**:

```bash
PY=/home/dcor/galbarak2/conda_envs/graphtalk/bin/python
export HF_HOME=$PWD/.cache HF_HUB_OFFLINE=1     # needed by anything that loads the tokenizer
```

There is no `.venv` and no `uv` on this box, and the `python` on PATH is an unrelated
3.13 install with none of the dependencies — see §9. All four findings below were re-run
this way on 2026-09-10 and the results are what is reported here.

### Finding 1 — `build_prompts.py`'s chars/4 heuristic is wrong by ~3x

Real ratio for graph-encoding text is **1.2–2.3 chars/token, never 4**, because the
content is digit-dense.

Full re-measured table (the laptop's three rows all reproduced to the digit):

| file | condition | n | chars | real tokens | chars/token |
| --- | --- | --- | --- | --- | --- |
| n20 | none | 20 | 1,278 | 780 | 1.64 |
| n20 | degree | 20 | 1,715 | 957 | 1.79 |
| n20 | all | 20 | 3,615 | 1,577 | 2.29 |
| n40 | none | 40 | 3,460 | 2,529 | 1.37 |
| n40 | degree | 40 | 4,349 | 2,898 | 1.50 |
| n40 | all | 40 | 8,149 | 4,138 | 1.97 |
| n80 | none | 80 | 10,775 | 8,969 | 1.20 |
| n80 | degree | 80 | 12,567 | 9,721 | 1.29 |
| n80 | all | 80 | 20,167 | 12,201 | 1.65 |

The ratio *falls* as n grows (edge lists get more digit-dense) and *rises* with the primer
(more prose), so no single constant can work — `token_budget.py` needs the real tokenizer,
or at minimum a per-(n, condition) table.

The offending constant is `_APPROX_CHARS_PER_TOKEN = 4` at `scripts/build_prompts.py:38`,
used by the `--model` overflow warning at `scripts/build_prompts.py:464-470`. It
under-counts by **1.7–3.3x** [CORRECTED: the laptop session said 2.5–3.3x; the true floor
is 4/2.29 = 1.7x, at n20 with the `all` primer].

```python
# Run under the cluster env with the repo-local HF cache -- see section 9:
#   env HF_HOME=$PWD/.cache HF_HUB_OFFLINE=1 \
#       /home/dcor/galbarak2/conda_envs/graphtalk/bin/python - <<'EOF'
import json, statistics
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')
for f in ['prompts_node_degree_n20.jsonl','prompts_node_degree_n40.jsonl','prompts_node_degree_n80.jsonl']:
    rows = [json.loads(l) for l in open(f)]
    for cond in ['none','degree','all']:
        sel = [r for r in rows if r['condition']==cond]
        if not sel: continue
        ch = statistics.mean(len(r['prompt']) for r in sel)
        tk = statistics.mean(len(tok(r['prompt'])['input_ids']) for r in sel)
        print(f, cond, sel[0]['nodes'], round(ch), round(tk), round(ch/tk,3))
```

**Tokenizer note [CORRECTED].** On the Windows laptop `AutoTokenizer.from_pretrained`
failed with `SSLCertVerificationError` and needed a curl workaround. **On TAU neither the
workaround nor any network access is needed** — but the naive call still fails, for a
different reason. The checkpoint is cached *inside the repo*, and `cluster/sweep.sbatch`
lines 182-183 set:

```bash
export HF_HOME=/home/dcor/avivyossef/inbal/GraphTalk/.cache
export HF_HUB_OFFLINE=1
```

`.cache/hub/models--Qwen--Qwen3-1.7B/` holds `tokenizer.json`, `tokenizer_config.json`,
`vocab.json`, `merges.txt`, `config.json` and both weight shards. **Any snippet in this
document must export those two variables**, or it will try to reach huggingface.co from a
node with no route to it.

Loaded tokenizer identifies as **`Qwen2Tokenizer`** — the slow one; `use_fast=True` still
returns it in this env, so expect slow tokenization on large batches. Vocab 151,643. Chat
wrapper overhead on empty content: **plain = 12 tokens, think = 8** (confirmed; plain is
larger because the template injects an empty `<think>\n\n</think>\n\n` block when
`enable_thinking=False`). Measure it the way `hf_backend.generate` does —
`apply_chat_template(..., add_generation_prompt=True, return_dict=True,
return_tensors="pt")` — since dropping `return_dict` silently returns a different shape
and yields a nonsense count.

### Finding 2 — the n80 "truncation" is input overflow, not output truncation

`docs/node_degree-density-and-size.md` reports 16.7–20% truncation at n80 and reads it as
output truncation. It isn't.

```python
import json
from collections import Counter
rows = [json.loads(l) for l in open('runs/qwen3-1.7b.node_degree_n80.jsonl')]
print(Counter(r.get('overflow') for r in rows))   # {False: 175, True: 35}
print(Counter(r.get('hit_cap') for r in rows))    # {False: 209, True: 1}
```

35/210 rows are `overflow: true` (generation skipped entirely) against exactly 1
`hit_cap`. Overflow is set at `scripts/run_sweep.py:188` as `completion is None`, which
happens when `hf_backend.generate` raises the context guard at
`graphtalk/hf_backend.py:103`. **That doc section needs correcting.**

### Finding 3 — the real input budgets, derived empirically

Measuring the overflowed vs surviving prompts brackets the limit exactly:

- overflowed prompts: 25,136–28,738 tokens (n=35)
- surviving prompts: 1,521–19,421 tokens (n=175)
- therefore the limit sits at **24,576 = 32768 - 8192**, confirming the guard formula
  `prompt_len + max_new_tokens > max_context_tokens`

Usable input budgets:

| arm | max_new_tokens | input budget |
| --- | --- | --- |
| plain | 8,192 | 24,576 |
| think | 16,384 | 16,384 |

The think arm has **half** the input room. That asymmetry — not accuracy — is what will
bound the feasible (n, density) region.

To reproduce, join runs to prompts on `(task, condition, style, instance_id)` — there is
no `index` field in either file — and tokenize with `enable_thinking=False`.

**Unverified aside worth checking:** the downloaded `config.json` reports
`max_position_embeddings: 40960`, not 32768. The repo's 32768 is the
conservative/published-recommended figure. Raising it would buy 8,192 tokens per arm, but
this was not validated on the checkpoint, and YaRN is presumably off. Treat as a lead, not
a fact.

### Finding 4 — measured output need, so budgets come from data

```python
import json, glob, statistics as st
from collections import defaultdict
agg = defaultdict(list); cap = defaultdict(int)
for f in glob.glob('runs/qwen3-1.7b*node_degree_n*.jsonl'):
    arm = 'think' if '-think' in f else 'plain'
    size = next((s for s in ('n20','n40','n60','n80') if s in f), None)
    if not size: continue
    for l in open(f):
        r = json.loads(l)
        k = (arm, size, r.get('condition'))
        if r.get('n_new_tokens'): agg[k].append(r['n_new_tokens'])
        if r.get('hit_cap'): cap[k] += 1
for k in sorted(agg):
    v = sorted(agg[k]); n = len(v)
    print(k, n, st.median(v), v[int(.9*(n-1))], max(v), cap[k])
```

Re-measured over all 8 `runs/qwen3-1.7b*node_degree_n*.jsonl` files, 28 cells per arm:

- **plain:** per-cell median 79–241, per-cell p90 178–496. Over all 805 plain rows: p99 =
  572, **uncapped max = 2,433** [CORRECTED: the laptop session said max 790], plus exactly
  **1 row that hit the 8,192 cap**.
- **think:** per-cell median 964–2,771, per-cell p90 1,619–5,707, plus **5 rows at exactly
  16,384**. Those runaways are **non-termination, not length-scaling** — the repo tracks
  this separately via `scripts/characterize_non_termination.py` and
  `analysis/non_termination_sample.csv` [CORRECTED: there is no
  `non_terminating_manifest.json` anywhere in the repo].

**Implication [CORRECTED].** 8,192 is still heavily over-provisioned for plain, but the
laptop session's proposed 2,048 is **too aggressive** — it would have truncated the
2,433-token tail row on top of the one already-capped row. Set the plain arm to **3,072**
(covers the observed max with headroom, still frees 5,120 tokens of input budget), or
4,096 for margin on the denser structural graphs this sweep will introduce, whose tail is
not yet measured. Either way, re-measure the tail on the screen run before locking it for
the confirmatory run.

## 4. The three open questions (not yet answered — re-ask these)

1. **Write permission** — options were: (a) new files + fix the chars/4 heuristic in
   `build_prompts.py`; (b) new files only, leave the heuristic; (c) write one module first
   for style review.
2. **Which primer is the "with primer" treatment** — options: (a) `all` + `filler` as
   length control [recommended]; (b) `all` vs `none` only, cheaper but leaves lift
   confounded with prompt length; (c) add `degree` as a 4th for the manipulation check.
3. **Staging** — options: (a) screen-then-confirm [recommended]; (b) one full sweep at
   k=53; (c) screen only for now.

The user rejected the question card and asked to clarify first; no answers were given. The
clarification never happened — the session was redirected to producing this handoff.

## 5. The methodological catch (must survive the handoff)

The stated constraint is "vary ONLY graph structure." That is not fully achievable and the
design must say so:

- Structural properties are **mathematically coupled** — density cannot move while degree
  variance, clustering and diameter stay fixed.
- Prompt length is a function of n and m, so **every density sweep is also a prompt-length
  sweep**.
- The repo already measured that confound: `filler` (length-only control) costs **-11.7pp**
  on dense graphs, per `docs/node_degree-density-and-size.md` Part B.

**Mitigation planned:** carry prompt tokens as a covariate, keep `filler` as the length
control, and have the design module emit a report of which property pairs are
decorrelatable at a given n and which are not. That report is a deliverable, not a caveat.

## 6. Prediction to test, not assume

From Part B at n=40: plain `none` is at 39.3% by p=0.35 and 14.0% by p=0.65; think `none`
is still at 43.9% at p=0.65. Where plain has hit the floor, think is only entering its
usable band. **Expect two separate sweet spots rather than one** — the user explicitly
asked for this to be flagged if true. Test it; don't assert it.

## 7. What exists in the repo to reuse (do not rebuild)

| Need | Already there |
| --- | --- |
| Clustered permutation tests, bootstrap CIs, BH correction | `graphtalk/significance.py` — **stdlib-only** (`itertools`, `math`, `random`), runs anywhere |
| Underpowered vs null separation | `significance.minimum_detectable_effect_clustered` |
| Sample-size calculation | `significance.required_sample_size_clustered`, `required_n_closed_form`, `scripts/recommend_count.py` |
| Ceiling/floor/informative zoning, arm divergence | `graphtalk/range_search.py` (`classify_cell`, `scout_decision`, `arm_divergence`) — **stdlib-only** (`math`) |
| GEE clustered by instance | `graphtalk/mixed_models.py` — **needs `statsmodels`, NOT installed in the cluster env; see §9** |
| Per-graph structural features | `scripts/extract_graph_topology.py` |
| Size x density grid + screens | `scripts/size_screen.py`, `score_density_size_grid.py`, `task_scoped_screen.py` |
| Graph pool generation | `graphtalk/diverse_corpus.py` — but only controls algorithm, n, and ER density; **this is the main gap** |
| Primer rendering (single renderer, 7 conditions) | `graphtalk/primers.py` — `CONDITIONS = none/components/degree/clustering/rwse/filler/all` |
| Prompt assembly | `graphtalk/prompts.py` — `primer + "\n\n" + encoding + task_description`, `incident` encoding, `zero_shot` style |

All module, function and script names in this table were confirmed present on 2026-09-10.
`CONDITIONS` in `primers.py` is exactly the seven named, with
`all = ("degree", "clustering", "rwse")` — note `all` does **not** include `components`.

## 8. Proposed build (drafted, unbuilt)

New modules:

- `graphtalk/structural_corpus.py` — targeted generators (regular, BA/power-law,
  Watts–Strogatz for clustering x diameter, trees/forests with controlled branching factor,
  SBM for component count, bipartite, planar, DAG), rejection sampling to hit targets,
  ~25-feature `measure()`. Must canonicalize via `graphqa.canonical()` like
  `diverse_corpus` does.
- `graphtalk/token_budget.py` — real-tokenizer counting through the actual chat template
  per mode; measured per-condition chars/token table as offline fallback.
- `graphtalk/structural_design.py` — property ranges, feasibility/decorrelation checks,
  k-per-cell from `required_sample_size_clustered`.
- `graphtalk/structural_analysis.py` — GEE logistic `success ~ property * primer +
  prompt_tokens` per arm, three-way term for the arm comparison, BH across all tests,
  analytic sweet-spot search generalizing `range_search.classify_cell`.
  **Blocked as specified:** this leans on `graphtalk/mixed_models.py`, which imports
  `statsmodels` — not installed in the cluster env, which is not writable by this user
  (§9). Resolve that before committing to GEE, or fall back to `significance.py`'s
  clustered permutation + bootstrap, which is stdlib-only and already covers the
  primary effect test. The GEE is what buys the *covariate adjustment for prompt
  tokens* (§5), so dropping it is not free — it is the difference between adjusting for
  the length confound and only controlling it by design via `filler`.

New scripts: `scripts/build_structural_sweep.py`, `scripts/analyze_structural_sweep.py`,
`cluster/run_structural_sweep.sh`, `docs/structural-sweep.md`, plus tests per module.

**Sizing.** Confirmed on the cluster 2026-09-10:
`significance.required_n_closed_form(delta)` returns **79 at 10pp, 53 at 15pp, 40 at
20pp** (two-sided alpha=0.05, 80% power). Full factorial at k=53 x 4 conditions x 2 arms is ~173k generations — too much.
Proposed staging mirrors the repo's existing pattern: plain-arm `none`-vs-`all` screen at
k=20 over the whole grid (~16k short generations), then full 4-condition x 2-arm
confirmatory at k=53 only on cells surviving `range_search.scout_decision` (~15k, half
think-mode). Survivor selection is by threshold, **computed — never by eyeballing cells**,
which was an explicit user constraint.

## 9. TAU environment notes [REWRITTEN 2026-09-10 — verified on the box]

The laptop session wrote this section from `cluster/README.md`. Checked against the actual
machine, several things differ. **Read this whole section before running anything.**

### Which Python

```bash
/home/dcor/galbarak2/conda_envs/graphtalk/bin/python
```

Python 3.11.15, transformers 5.15.0, torch 2.13.0+cu130, networkx 3.6.1, pandas 3.0.5.
This env is owned by `galbarak2` but is **readable and executable** by `avivyossef` —
verified working.

Three traps:

- **`.venv/` does not exist** in this checkout. `CLAUDE.md`'s `.venv/bin/python` is wrong
  here.
- **`uv` is not on PATH.** `CLAUDE.md`'s `uv run --no-sync pytest -q` cannot be run on
  this box at all. Use `<conda-env>/bin/python -m pytest -q` instead.
- **The default `python` on PATH is a trap.** It resolves to
  `/vol/joberant_nobck/data/NLP_368307701_2526a/avivyossef/miniconda3/bin/python`, which
  is **Python 3.13** — violating the repo's `>=3.11,<3.12` pin — and has neither
  `networkx` nor `transformers` installed. Always call the conda env by absolute path.

### The env is NOT writable, and three analysis deps are missing

```
statsmodels  MISSING
scipy        MISSING
arviz        MISSING
```

`/home/dcor/galbarak2/conda_envs/graphtalk` is `drwxr-sr-x galbarak2` — **this user cannot
pip-install into it.** Consequences:

- `graphtalk/mixed_models.py` (GEE) cannot import → §8's `structural_analysis.py` is
  blocked as designed.
- `tests/test_mixed_models.py` and `tests/test_hierarchical_model.py` **error during
  collection**, not merely fail.
- `graphtalk/significance.py` and `graphtalk/range_search.py` are **stdlib-only** and work
  fine, so the sizing, zoning and permutation-test reuse in §7 is unaffected.

Options, in order of preference: ask `galbarak2` to install into the shared env; or
`pip install --user statsmodels scipy` (check it does not shadow the env's numpy/pandas);
or clone the env to a writable path and set `GRAPHTALK_ENV` accordingly — `sweep.sbatch`
line 176 activates `/home/dcor/galbarak2/conda_envs/${GRAPHTALK_ENV}`, so a clone must
live under that directory or the script needs editing.

### Actual test count — the four-way disagreement, settled

The docs disagree; none of them is right:

| source | claims |
| --- | --- |
| `CLAUDE.md:43-45` | 345, +23 = 368 |
| `cluster/README.md:51` | 359 passed |
| `docs/difficulty-scaling.md:186` | 607 passed |
| **measured 2026-09-10 on this box** | **617 collected; 616 passed, 1 failed, 2 files error on collection** |

```bash
/home/dcor/galbarak2/conda_envs/graphtalk/bin/python -m pytest -q \
    --ignore=tests/test_hierarchical_model.py --ignore=tests/test_mixed_models.py
# 1 failed, 616 passed in 294s
```

**Do not use any of the documented counts as a health check** — they are all stale.
The 2 collection errors are the missing `statsmodels`/`arviz` above.

**The 1 failure is pre-existing and is not env rot:**
`tests/test_diverse_corpus.py::test_algorithms_filter_empty_tuple_raises` — expects
`build_pool(10, algorithms=())` to raise `ValueError` matching `"non-empty"`, and it does
not raise. Both `graphtalk/diverse_corpus.py` and `tests/test_diverse_corpus.py` are
**modified in the working tree** (uncommitted), so this is in-flight work by someone else,
not something the new session broke. Do not chase it; do check whether it lands before
building `structural_corpus.py` on top of `diverse_corpus`.

### Running the sweep

- Account `galbarak2`, DCOR lab, partition `killable`, account `gpu-research`.
- Stage 2: `sbatch cluster/sweep.sbatch qwen3-1.7b`. Override via `GRAPHTALK_PROMPTS`
  (must point at an existing file — guarded at `sweep.sbatch:64-71`) and
  `GRAPHTALK_RUN_TAG`.
- **Never use a run tag containing `redo`** — `sweep.sbatch:109-113` hard-refuses it,
  because `analysis.py` excludes `.redo.shard` paths and the rows would silently vanish
  from every frame.
- `HF_HOME` is repo-local and `HF_HUB_OFFLINE=1` (`sweep.sbatch:182-183`); the checkpoint
  is already in `.cache/`, so GPU nodes need no network.
- Submit a **chain**, not a single job (24h partition ceiling); see `cluster/README.md`,
  "Runtime: submit a chain, not a job".
- `TF_USE_LEGACY_KERAS=1` before importing `tensorflow_gnn`.
- `cluster/README.md` remains the authority on partitions, memory sizing, driver
  incompatibilities, and node exclusions (n-801 is slow).

### Repo state

`docs/handoff-structural-sweep.md` is **untracked**, as are ~40 `prompts_*.jsonl` files and
several new docs and modules (`graphtalk/range_search.py` among them). Commit before
relying on any of it travelling.

## 10. Kickoff prompt for the new terminal

```
Continuing a design session. Read docs/handoff-structural-sweep.md first — it has four
empirical findings (all re-verified on the TAU cluster 2026-09-10), the locked decisions,
and three open questions.

Context: designing a sweep to find which structural graph properties make GraphQA tasks
hard enough without a primer that adding one produces a significant lift, for
qwen3-1.7b and qwen3-1.7b-think as separate arms, on the task panel
{node_degree, cycle_check, connected_nodes}.

Use /home/dcor/galbarak2/conda_envs/graphtalk/bin/python by absolute path — there is no
.venv, `uv` is not installed, and the `python` on PATH is a bare 3.13 with no deps.
Any tokenizer snippet needs HF_HOME=$PWD/.cache and HF_HUB_OFFLINE=1.

Start by re-asking the three open questions in section 4 (write permission, which primer
is the treatment, screen-then-confirm vs full sweep). Do not write code before they're
answered.

Then resolve the one open blocker in section 9: statsmodels/scipy/arviz are missing from
the cluster env and it is not writable by this user, so the GEE analysis in section 8
cannot run as designed. Decide between getting them installed and falling back to the
stdlib-only permutation/bootstrap path in significance.py.
```

## 12. Grid specification [added 2026-09-10]

### The design move, and the measurement that justifies it

Prompt length is essentially a function of `(n, m)` alone — the `incident` encoding lists
each node's neighbours, so token count tracks `n + 2m` plus digit width, and is almost
blind to *shape*. Measured on this box with the real tokenizer, `condition="none"`:

| n | family | m | degree_std | clustering | diameter | tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | BA (hub-heavy) | 51 | 3.30 | 0.405 | 3 | 583 |
| 20 | ER (uniform) | 51 | 2.32 | 0.305 | 4 | 599 |
| 40 | BA (hub-heavy) | 111 | 4.14 | 0.331 | 4 | 1,275 |
| 40 | ER (uniform) | 111 | 1.88 | 0.127 | 4 | 1,308 |
| 40 | WS (clustered) | 120 | 0.81 | 0.395 | 5 | 1,384 |
| 40 | regular (flat) | 100 | 0.00 | 0.128 | 4 | 1,230 |
| 80 | BA (hub-heavy) | 231 | 4.83 | 0.208 | 4 | 2,678 |
| 80 | ER (uniform) | 231 | 2.25 | 0.057 | disc. | 2,743 |

**At exactly equal `m` (BA vs ER) prompt length differs by 2.4-2.6% at every size**, while
degree_std moves 1.9 -> 4.1 and clustering 0.13 -> 0.33. So *fixing `(n, m)` and swapping
the generator* buys a large structural contrast at near-constant prompt length. This is
what defuses the -11.7pp length confound **by design** rather than by post-hoc regression
adjustment — which matters given that the GEE covariate path is currently blocked (§9).

The 12% spread in the raw table above is entirely WS and regular *missing* the `m` target:
regular needs `k * n` even, WS needs `k` even. **Reachable `m` is quantised per family** —
`structural_design.py` must solve for the nearest jointly-reachable `m` per `(n, k)` cell
and report the residual length mismatch, not assume the target is hit.

### Axes actually set

| axis | levels | note |
| --- | --- | --- |
| `n` | 20, 40, 60, 80 | matches the existing size sweep, so old rows stay comparable |
| mean degree `k = 2m/n` | 3, 5, 8 | pins `m`, and therefore prompt length |
| family | regular, ER, WS, BA, SBM | 5 shapes at matched `(n, m)` |

4 x 3 x 5 = **60 length-matched cells**, plus a **sparse annex** of 4 `n`-levels x
{tree, forest, ER-sparse} = **12 cells**, giving **72 structural cells**.

The annex exists because family x `k` is **not fully crossable**: a forest has `m = n - c`,
so it cannot reach `k >= 3` at any `n`. This is the coupling problem from §5 in its most
concrete form, and the design module reports it rather than hiding it.

### Properties measured per graph (not set)

The 17 features `scripts/extract_graph_topology.py` already computes — `num_nodes`,
`num_edges`, `density`, `degree_mean`, `degree_std`, `degree_min`, `degree_max`,
`component_count`, `circuit_rank`, `is_tree`, `is_forest`, `is_bipartite`,
`has_isolated_node`, `triangle_count`, `is_triangle_free`, `clustering_mean`,
`size_bucket` — plus 8 to add: `diameter`, `radius`, `avg_shortest_path`, `assortativity`,
`transitivity`, `max_component_frac`, `degree_entropy`, and `prompt_tokens`.

### Combinations tested

- **Primary:** family at fixed `(n, k)` — a 5-way contrast at each of 12 length-matched
  points. This is the only contrast that isolates shape from both size and length.
- **Secondary:** `k` at fixed `(n, family)`, and `n` at fixed `(k, family)` — reproduces
  and extends the existing 4x4 size x density grid.
- **Two-dimensional knob:** clustering x diameter, via WS rewiring probability, which
  moves the two in opposite directions.
- **Component count:** reachable only through SBM and the forest annex.

### Sample sizes, and why a result is a trend rather than an accident

| stage | cells | graphs/cell | distinct graphs | generations |
| --- | --- | --- | --- | --- |
| screen — plain arm, `none` vs `all`, 3 tasks | 72 | 20 | 1,440 | 8,640 |
| confirm — ~30 surviving (cell x task) pairs, 4 conditions x 2 arms | ~30 | 53 | ~1,590 | ~12,720 |
| **total** | | | **~3,000** | **~21,400** |

No trend claim rests on 20 graphs. 20 is the screen *nomination* depth; cells are never
concluded from at that depth. Each marginal pools across the other axes — every `n`-level
pools ~360 graphs, every `k`-level ~400, every family ~240 — and the trend test is a
regression slope across levels with the **graph as the clustering unit**, not a pairwise
cell comparison.

The guarantee comes from the confirmatory stage: 53 = `required_n_closed_form(0.15)`, i.e.
80% power at alpha=0.05 for a 15pp lift. Because the same graph is scored under `none` and
`all`, those are 53 **matched pairs**, which is exactly what `significance.py`'s clustered
permutation test consumes. A survivor that then fails to reach significance is
disambiguated by `minimum_detectable_effect_clustered` into "underpowered" vs "genuinely
no effect", so a null is reportable rather than silent.

### Known limit

`k=8` at `n=80` is ~3,400 tokens bare and ~7,000 with `all` — comfortable in both arms. If
the grid is later pushed past `n=120` at `k=8`, the think arm's 16,384-token input budget
binds well before the plain arm's 24,576, and the two arms stop seeing identical prompts.
Feasibility per cell is tabulated in §13.

## 11. Review log (2026-09-10)

A verification pass on the TAU box re-ran every claim in this document. Result:

**Reproduced exactly** — the full chars/token table; `build_prompts.py:38` and the
`464-470` warning; `run_sweep.py:188`; `hf_backend.py:103`; both `models.py` token caps
(8192/16384, 32768); 210 n80 rows at 35 overflow / 1 `hit_cap`; the 25,136–28,738 vs
19,421 bracket around 24,576; think-arm output percentiles; k = 79/53/40; Part B's 39.3% /
14.0% / 43.9% / −11.7pp; the 12/8 chat-template overhead; and every module, function and
script named in §7.

**Corrected** — plain-arm uncapped max is 2,433, not 790, which invalidates the proposed
2,048 plain budget (§3, Finding 4); the chars/4 undercount floor is 1.7x, not 2.5x;
`non_terminating_manifest.json` does not exist (`analysis/non_termination_sample.csv`
does); the tokenizer is `Qwen2Tokenizer`, not the Fast variant.

**Newly found** — the env is not writable and is missing `statsmodels`/`scipy`/`arviz`,
which blocks §8's GEE module; the real test count is 617 collected / 616 passed / 1 failed
/ 2 collection errors, against three mutually inconsistent documented figures; there is
one pre-existing test failure in uncommitted `diverse_corpus` work; `uv` and `.venv` are
both absent so `CLAUDE.md`'s documented test command does not run here; and every snippet
in this document needs `HF_HOME` and `HF_HUB_OFFLINE` set.
