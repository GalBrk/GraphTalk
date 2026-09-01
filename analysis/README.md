# Measurement artefacts

Small, expensive-to-reproduce measurements that decisions in this project rest
on. They live in git because the reasoning in `docs/sweep-findings.md` and
`cluster/README.md` cites them, and because each cost GPU time that a reader
should not have to spend again to check the claim.

| file | what it is |
|---|---|
| `budget-gemma4-e4b.jsonl` | 24 `zero_shot` prompts spanning all six tasks, generated at a 2048 cap with exact `n_new_tokens` per row. The measurement that set `MAX_NEW_TOKENS["zero_shot"]`: at 64 tokens this model scored 3/24, at 2048 it scored 24/24 with no row reaching the cap. |
| `budget-qwen3-8b.jsonl` | the same 24 prompts on Qwen3-8B with `enable_thinking=False` |
| `budget-qwen3-8b-THINKING.jsonl` | three of those rows with thinking **on**, kept as the before-picture: 1179 mean tokens on a `node_count` question the same model answers in 105 without |
| `truncated_keys.json` | historical record of the **271** thinking-arm rows that were hand-labelled non-terminating. No longer consulted: every tracked row now carries `hit_cap`. Kept because it is the provenance of a claim, and because two of its rows turn out to have terminated. |
| `sweep_frame.csv` | one row per scored response over the whole tracked sweep, 15,120 rows. **Stale — do not cite; see below.** |
| `failure_sample.csv` | the stratified manual-inspection sample, 56 rows with full response text. **Stale — do not cite; see below.** |
| `significance_report.csv`/`.txt` | `scripts/check_significance.py`'s pooled permutation/bootstrap/BH-corrected results, one row per (arm, group, condition). See "The significance report" below — this one has a real methodology fix behind its most recent regeneration, not just fresher input. |

## The two CSVs, and what is in them now

Both were regenerated on 2026-08-29, after the prompt-rewording re-run and with the
current `scoring._extract_boolean`. They supersede the versions committed at
`3545662`, which were written before both changes and understated parse rates
badly -- `unparsed` alone fell from 342 rows to 70.

`sweep_frame.csv` carries three columns the earlier version did not:

- `wording` -- `revised` / `original` / `unaffected`, marking which text produced the
  row. The obsolete `zero_cot` rows keep the original `filler` and `edge_existence`
  wording, so `condition` alone does not identify the prompt. Group by this before
  comparing anything that touches those cells.
- `non_terminating_source` -- `recorded` where the row carries its own `hit_cap`,
  `ground_truth_file` where `truncated_keys.json` is still the only record.
- `n_new_tokens` -- present only on regenerated rows; empty, not zero, elsewhere.

A fourth was added since, once `runs/<model>.got.jsonl` files exist:

- `node_naming` -- `integer` (the default, and the only value in the frame
  today) or `got`. `scripts/build_sweep_frame.py` refuses to build a frame
  mixing the two -- it raises rather than silently pooling `(instance_id,
  condition, style)` rows that only look like duplicates because the column
  distinguishing them didn't exist yet. Score each scheme separately and let
  each land at its own `.got.`-tagged CSV; see `README.md#node-naming`.

## The significance report

Regenerated with a corrected methodology, not just fresher `sweep_frame.csv`
input -- an audit of `scripts/check_significance.py` found it was pooling
correlated rows (the same graph instance recurs up to 12x per condition,
across styles and models) as if they were independent, which is anti-
conservative, and including `non_terminating` rows in the main-sweep
accuracy test, confounding "does this primer help reasoning" with "does
this primer change truncation rate" (`docs/DATA.md`/`runs/README.md`
already establish that non-terminating rows must be filtered before
reporting accuracy; this script wasn't).

The fix: `graphtalk/significance.py` gained `paired_permutation_test_clustered`/
`cluster_bootstrap_ci_clustered`, which flip/resample whole graph instances
rather than individual rows, and `check_significance.py` excludes
non-terminating rows from the main-sweep test (`n_excluded_non_terminating`
reports how many, per condition, so the confound's size stays visible)
and adds a uniqueness guard (`graphtalk.analysis.assert_unique_pairing_key`)
before pairing anything.

**The effect was not cosmetic**: three `bh_significant` flags flipped versus
the pre-fix regeneration, including `gemma4-12b`/`rwse` going from
"significant" (p=0.0072) to clearly not (p=1.0) -- exactly the kind of
false positive naive pooling produces. `n_pairs` (raw rows) and `n_clusters`
(real graph instances) are both reported now precisely so this gap stays
visible rather than being averaged away again.

## The batching baseline

`budget-gemma4-e4b.jsonl` and `budget-qwen3-8b.jsonl` are the reference for
verifying a batched implementation. Decoding is greedy, so a correct batch must
reproduce these responses near-identically. That check matters more than it
sounds: **Qwen3's tokenizer defaults to `padding_side='right'` while Gemma's
defaults to `'left'`**, and for a decoder-only model the wrong padding side
produces fluent, well-formed, entirely wrong text rather than an error.

## Not kept

The prompt subsets used to drive these runs (`diag_prompts`, `think_probe`,
`tail-probe`, `redo-*`) are regenerable from `prompts_zero_shot.jsonl` plus
`truncated_keys.json`, and are not tracked.
