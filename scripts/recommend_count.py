"""Track 2.1: translates Track 1's MDE numbers into a recommended `--count`
per (model, condition) cell -- no GPU time, no code change to the
statistical machinery, just a data-driven choice of how big a future sweep
should be, and only where scaling up is actually affordable.

**The formula, and why no new simulation is needed.** For a paired
permutation/binomial-style test, the minimum detectable effect at a fixed
power target scales asymptotically as `MDE ~ 1/sqrt(N)` (N = cluster
count). So if the current sweep's `minimum_detectable_effect_clustered`
found `mde_delta` at `n_clusters` clusters, and a cell's own *observed*
`delta` is smaller in magnitude than that (true for every non-significant
row, by construction -- if `|delta| >= mde_delta` the row would already be
significant), the sample size needed to make `delta` itself detectable at
80% power is approximately:

    n_clusters_needed = n_clusters * (mde_delta / delta) ** 2

This reuses Track 1's already-computed MDE rather than running a new
simulation -- it is a closed-form extrapolation of it, not a
re-derivation. `scripts/validate_recommend_count.py` checks the
extrapolation's accuracy against a real bootstrap-based power simulation
at the recommended size, for a handful of cells, before trusting it
further.

Positive-direction `delta` (the condition helps) is checked against
`mde_delta`; negative-direction `delta` (the condition hurts) against
`mde_delta_negative` -- matching the sign of the effect actually observed,
not always the positive-direction MDE. A cell is skipped (not extrapolated)
when: it's already **globally** significant (`bh_significant_global` --
nothing to add data for; see the note on family-significant-only cells
below), its observed `delta` is exactly zero (no effect to extrapolate
from -- infinite data would still not detect a truly null effect), or the
relevant MDE is `None` (the search didn't converge within `[0, 1]` at the
current sample size -- see `graphtalk.significance
.minimum_detectable_effect_clustered`'s docstring; a near-ceiling/
near-floor control usually lands here, and no finite `--count` fixes a
ceiling problem, only more headroom would).

**Family-significant, not globally significant, cells (`bh_significant`
True but `bh_significant_global` False or blank) are a real, distinct
case, not just "already significant".** This is exactly the situation a
result worth replicating sits in: real per-model evidence that hasn't
cleared the stricter whole-table bar. `check_significance.py` never
simulates an MDE for these rows (its own trigger is `not
{per-family significance}`, computed before the whole-table pass even
runs -- there is no earlier point in that script where
`bh_significant_global` exists yet to trigger on instead, so this could
not be fixed by widening a condition there without a larger restructuring
of its single-pass `main()`). Pass `--frame` (the `sweep_frame.csv` the
report was built from) and this script computes a real MDE for exactly
these cells, on demand, using the same
`graphtalk.significance.minimum_detectable_effect_clustered` call
`check_significance.py` itself makes -- see `_mde_for_family_significant_cell`.
Without `--frame`, these cells are skipped with a `skip_reason` saying so,
rather than silently treated as "nothing to do here" the way a plain
`bh_significant` check would.

  PYTHONPATH=. .venv/bin/python scripts/recommend_count.py \
      --report analysis/significance_report.csv

  # Also compute real MDEs for family-significant/not-global cells:
  PYTHONPATH=. .venv/bin/python scripts/recommend_count.py \
      --report analysis/significance_report.got.csv \
      --frame analysis/sweep_frame.got.csv
"""

import argparse

import pandas as pd

from graphtalk import significance
from scripts import check_significance as cs

_CURRENT_COUNT = 30
_PUBLISHED_SPLIT_CAP = 500
# Matches check_significance.py's own fast-approximate MDE preset
# (Phase 1.3.2) -- this is a dry-run planning tool, not a final reported
# number, so the same speed/precision tradeoff applies.
_MDE_REPLICATES = 50
_MDE_N_PERM = 200
_MDE_N_STEPS = 5


def _mde_for_family_significant_cell(
    frame: pd.DataFrame, model: str, condition: str, delta: float,
    seed: int = 1234,
) -> float | None:
  """On-demand MDE for one (model, condition) cell that
  `check_significance.py` never simulated one for, because the cell was
  already significant within its own per-family correction (see the
  module docstring). Mirrors `_report`'s own MDE call as closely as
  possible: same main-sweep scope (`~is_think`, non-terminating rows
  included -- `graphtalk.analysis.build_frame` already forces them to
  score as wrong, so nothing is excluded here either), same
  bootstrap-CI-width-seeded `initial_hi`, same direction convention
  (search the side matching the observed `delta`'s own sign). Returns
  `None` if the cell has no paired rows at all (shouldn't happen for a
  cell the report already scored, but checked rather than assumed).
  """
  cell_frame = frame[(frame["model_family"] == model) & (~frame["is_think"])]
  control, treatment, cluster_ids = cs._paired_values(cell_frame, condition, "exact")
  if not control:
    return None
  ci = significance.cluster_bootstrap_ci_clustered(
      control, treatment, cluster_ids, n_boot=1000, seed=seed,
  )
  ci_width = ci["ci_high"] - ci["ci_low"]
  direction = "positive" if delta > 0 else "negative"
  result = significance.minimum_detectable_effect_clustered(
      control, treatment, cluster_ids, initial_hi=max(0.05, ci_width),
      direction=direction, n_replicates=_MDE_REPLICATES, n_perm=_MDE_N_PERM,
      n_steps=_MDE_N_STEPS, seed=seed,
  )
  return result["delta"] if delta > 0 else result["delta_negative"]


def recommend(
    report: pd.DataFrame, current_count: int = _CURRENT_COUNT,
    frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
  """One row per non-*globally*-significant main-sweep `exact` cell
  (`bound == "excluded"`, not derived, not the pooled-across-models row --
  a per-model recommendation is the actionable unit; pooled rows are a
  different, larger-family question `check_significance.py` already
  reports separately) with a finite recommendation, plus `skip_reason` for
  every cell this can't extrapolate for.

  `frame` (optional, the `sweep_frame.csv` the report was built from):
  when given, a cell that's family-significant but not globally
  significant gets a real, freshly-simulated MDE via
  `_mde_for_family_significant_cell` instead of being skipped outright --
  see the module docstring.
  """
  scoped = report[
      (report["arm"] == "main_sweep") & (report["metric"] == "exact")
      & (report["bound"] == "excluded") & (~report["is_derived_condition"])
      & (report["group"] != "pooled across all models")
  ]
  rows = []
  for _, r in scoped.iterrows():
    # Every row gets the same keys regardless of branch -- a DataFrame
    # built from dicts with inconsistent keys silently upcasts columns
    # that are sometimes-missing to `object` dtype (bool mixed with the
    # filled-in NaN), which breaks `~column` boolean negation later; this
    # keeps every column's dtype consistent (float/bool/str + NaN, not
    # object) by always populating all of them.
    base = {
        "model": r["group"], "condition": r["condition"],
        "n_clusters": r["n_clusters"], "delta": r["delta"],
        "mde_used": None, "n_clusters_needed": None,
        "recommended_count": None, "exceeds_published_cap": None,
        "skip_reason": None,
    }
    # `bh_significant_global` is `NaN` for an ineligible row (already
    # excluded from `scoped` above -- pooled/derived) or a real True/False
    # for every row that reaches here; `pd.notna` guards the NaN case
    # rather than let it evaluate as truthy the way a bare `if` would.
    if pd.notna(r["bh_significant_global"]) and bool(r["bh_significant_global"]):
      rows.append({**base, "skip_reason": "already globally significant"})
      continue
    if r["delta"] == 0:
      rows.append({**base, "skip_reason": "observed delta is exactly zero"})
      continue
    if r["bh_significant"]:
      # Family-significant, not globally significant: check_significance.py
      # never simulated an MDE for this row (see the module docstring) --
      # compute one now, on demand, if a frame was given to compute it from.
      if frame is None:
        rows.append({**base, "skip_reason": "family-significant, not "
                     "globally significant -- pass --frame to compute a "
                     "real MDE for this cell instead of skipping it"})
        continue
      mde = _mde_for_family_significant_cell(
          frame, r["group"], r["condition"], r["delta"],
      )
    else:
      mde = r["mde_delta"] if r["delta"] > 0 else r["mde_delta_negative"]
    if pd.isna(mde):
      rows.append({**base, "skip_reason": "MDE did not converge at the "
                   "current sample size (near-ceiling/near-floor)"})
      continue
    ratio = (mde / r["delta"]) ** 2
    n_clusters_needed = r["n_clusters"] * ratio
    recommended_count = current_count * ratio
    rows.append({
        **base, "mde_used": mde, "n_clusters_needed": n_clusters_needed,
        "recommended_count": recommended_count,
        "exceeds_published_cap": recommended_count > _PUBLISHED_SPLIT_CAP,
    })
  return pd.DataFrame(rows)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--report", default="analysis/significance_report.csv")
  parser.add_argument("--current-count", type=int, default=_CURRENT_COUNT)
  parser.add_argument("--frame", default=None,
                       help="the sweep_frame.csv --report was built from -- "
                            "when given, family-significant-but-not-"
                            "globally-significant cells get a real,  "
                            "on-demand MDE instead of being skipped; see "
                            "the module docstring")
  parser.add_argument("--out", default=None)
  args = parser.parse_args()

  report = pd.read_csv(args.report)
  frame = pd.read_csv(args.frame) if args.frame else None
  result = recommend(report, current_count=args.current_count, frame=frame)

  finite = result[result["recommended_count"].notna()].sort_values("recommended_count")
  skipped = result[result["recommended_count"].isna()]

  pd.set_option("display.width", 200)
  print(f"=== {len(finite)} cells with a finite recommended --count "
        f"(published split caps at {_PUBLISHED_SPLIT_CAP}) ===")
  if not finite.empty:
    print(finite[["model", "condition", "n_clusters", "delta", "mde_used",
                   "recommended_count", "exceeds_published_cap"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    affordable = finite[~finite["exceeds_published_cap"].astype(bool)]
    print(f"\n  {len(affordable)}/{len(finite)} are affordable within the "
          f"{_PUBLISHED_SPLIT_CAP}-graph published split cap.")

  print(f"\n=== {len(skipped)} cells skipped ===")
  if not skipped.empty:
    print(skipped["skip_reason"].value_counts().to_string())

  if args.out:
    result.to_csv(args.out, index=False)
    print(f"\nwrote {len(result)} rows to {args.out}")


if __name__ == "__main__":
  main()
