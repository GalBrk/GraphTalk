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
when: it's already significant (nothing to add data for), its observed
`delta` is exactly zero (no effect to extrapolate from -- infinite data
would still not detect a truly null effect), or the relevant MDE is `None`
(the search didn't converge within `[0, 1]` at the current sample size --
see `graphtalk.significance.minimum_detectable_effect_clustered`'s
docstring; a near-ceiling/near-floor control usually lands here, and no
finite `--count` fixes a ceiling problem, only more headroom would).

  PYTHONPATH=. .venv/bin/python scripts/recommend_count.py \
      --report analysis/significance_report.csv
"""

import argparse

import pandas as pd

_CURRENT_COUNT = 30
_PUBLISHED_SPLIT_CAP = 500


def recommend(report: pd.DataFrame, current_count: int = _CURRENT_COUNT) -> pd.DataFrame:
  """One row per non-significant main-sweep `exact` cell (`bound ==
  "excluded"`, not derived, not the pooled-across-models row -- a
  per-model recommendation is the actionable unit; pooled rows are a
  different, larger-family question `check_significance.py` already
  reports separately) with a finite recommendation, plus `skip_reason` for
  every cell this can't extrapolate for.
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
    if r["bh_significant"]:
      rows.append({**base, "skip_reason": "already significant"})
      continue
    if r["delta"] == 0:
      rows.append({**base, "skip_reason": "observed delta is exactly zero"})
      continue
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
  parser.add_argument("--out", default=None)
  args = parser.parse_args()

  report = pd.read_csv(args.report)
  result = recommend(report, current_count=args.current_count)

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
