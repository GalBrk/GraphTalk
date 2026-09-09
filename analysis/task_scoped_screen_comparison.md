# Task-scoped screen: GOT vs. integer naming, side by side

`docs/plans/run_improved_tests.md` Phase 2. Union of every `(model,
condition, task)` cell either scheme's `scripts/task_scoped_screen.py` run
flagged at `p <= 0.10` (the screening threshold, not a significance
claim), computed at `--n-perm 3000 --n-boot 3000` against
`analysis/sweep_frame.got.csv` and `analysis/sweep_frame.csv`
respectively:

| model | condition | task | GOT delta | GOT p | integer delta | integer p |
|---|---|---|---|---|---|---|
| `gemma4-12b` | `degree` | `edge_count` | +0.167 | 0.0620 | +0.067 | 0.4985 |
| `qwen3-14b` | `degree` | `edge_count` | +0.300 | 0.0193 | +0.200 | 0.0666 |
| `qwen3-8b` | `degree` | `edge_count` | +0.367 | 0.0030 | +0.233 | 0.0393 |
| `qwen3-8b` | `filler` | `node_count` | +0.033 | 1.0000 | -0.233 | 0.0157 |

## Reading this table

**`qwen3-8b`/`degree`/`edge_count` corroborates cleanly across both
schemes** (already the headline finding driving this whole investigation):
same sign, both under the 0.05 screening cut in their own scheme, GOT the
larger of the two (consistent with the winner's-curse-shrunk +6.5pp GOT
already measured at `--count 500` -- the smaller integer-scheme estimate
at n=30 is not obviously inconsistent with the same true effect).

**`qwen3-14b`/`degree`/`edge_count` is a new, previously-unflagged
cross-scheme signal.** Same direction in both schemes (+0.300 GOT,
+0.200 integer), both under the 0.10 screen; only the GOT side clears
0.05 alone. This is exactly the kind of corroboration
`analysis/README.md`'s "Phase A1 candidates" section registered
`qwen3-14b`/`degree` (GOT) as confirmatory for -- this comparison is the
first time the integer side of the same cell has actually been looked at,
and it points the same direction, not a contradiction.

**`gemma4-12b`/`degree`/`edge_count` does NOT replicate across schemes.**
GOT alone looked borderline (p=0.062); integer's point estimate is a
third the size and nowhere near significant (p=0.499). Read this as
consistent with the existing model-heterogeneity note in
`analysis/README.md` (`gemma4-e4b`'s near-zero effect on this same
condition) rather than as a second confirmed model -- the GOT number
alone was never strong enough to lean on, and this comparison removes
what little support it had.

> **Follow-up, 2026-09-09.** "Filler hurts" was retracted on this corpus as an
> artifact of the primer's false numeral, and the retraction stands here. But a
> corrected, content-free `filler` *does* hurt once prompts get long: 11.7
> points on a thinking model at n=40, p>=0.85. So a negative `filler` effect is
> not automatically an instrument bug -- on long prompts it is the expected
> length penalty. See `../docs/primer-effects-and-power.md`, "The `filler`
> control".

**`qwen3-8b`/`filler`/`node_count` reverses between schemes -- flagged,
not smoothed over.** Integer shows a real, negative effect (filler hurts,
delta -0.233, p=0.016, ceiling-bound at 1.0 -- see
`scripts/task_scoped_screen.py`'s docstring on why a ceiling-bound cell's
effect is about shortcut execution, not graph reasoning). GOT shows
essentially nothing (delta +0.033, p=1.0) for the *same* (model,
condition, task) cell. Two effects this size and opposite-adjacent in
sign, from the same 30 underlying graphs under a different node-naming
scheme, is either (a) a real interaction between node-naming and how
reliably a model tracks the `filler` primer's sentence-count shortcut
(GOT names may make per-node filler sentences harder to count
correctly than integers do, independent of node identity, in a way that
happens to cancel out the "filler hurts" pattern seen with integers), or
(b) just noise at n=30 in a scheme that was never large enough to state
either verdict with much confidence -- **this cell should NOT be
pre-registered as "filler hurts node_count" without qualifying it to
integer naming specifically**, unlike `degree`/`edge_count`, which is a
generalizes-across-naming-schemes finding. `analysis/
confirmatory_integer_qwen3-8b_filler.json` (Phase B) already only
targets the integer scheme, so this doesn't invalidate that
pre-registration -- it's a reason not to *extend* it to GOT.

## Reproducing this table

```bash
PYTHONPATH=. .venv/bin/python scripts/task_scoped_screen.py \
    --frame analysis/sweep_frame.got.csv --n-perm 3000 --n-boot 3000 \
    --out analysis/task_scoped_screen.got.csv
PYTHONPATH=. .venv/bin/python scripts/task_scoped_screen.py \
    --frame analysis/sweep_frame.csv --n-perm 3000 --n-boot 3000 \
    --out analysis/task_scoped_screen.csv
```
