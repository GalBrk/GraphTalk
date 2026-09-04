"""Stage 4b: regression cross-checks for `scripts/check_significance.py`'s
permutation-test results. No GPU needed.

`--method gee` (default): `graphtalk.mixed_models.fit_gee_all_models` --
one identity-link binomial GEE per model, pooling all six conditions into a
single population-averaged fit instead of `check_significance.py`'s six
separate n=30 permutation tests per model. See
`graphtalk/mixed_models.py`'s module docstring for why GEE, why identity
link, and why an independence working correlation.

`--method bayes`: `graphtalk.hierarchical_model`'s partial-pooling
hierarchical logistic model, fit via NUTS -- see that module's docstring
for the model itself and why its numbers are on the log-odds scale rather
than `--method gee`'s directly-comparable probability-difference scale.
Slower than `--method gee` (real MCMC, not one GEE fit per model) --
`--draws`/`--tune`/`--chains` default to a real run, not a quick check; use
smaller values for fast iteration and check `r_hat`/`ess_bulk`/
`n_divergences` before trusting anything from a reduced run.

Reads the same joined table `scripts/build_sweep_frame.py` writes that
`check_significance.py` does (`analysis/sweep_frame.csv`) -- no re-scoring.

`--method gee`'s output is directly diffable, cell for cell, against
`analysis/significance_report.csv`'s `main_sweep`/`excluded`-bound rows:
same `model`/`condition` keys, `delta`/`ci_low`/`ci_high`/`p_value` in the
same units (a raw probability difference against `none`). `--method bayes`
is not -- its `total_mean` is a log-odds effect, not a probability
difference -- so `--compare-against` only checks *direction* agreement
(sign) for `bayes`, not a raw-p-value threshold comparison. Both methods
are cross-checks, not a replacement -- `check_significance.py` remains the
project's corrected-inference pipeline; this script exists to answer "does
an independent method agree", not to produce a second set of citable
numbers.

  PYTHONPATH=. .venv/bin/python scripts/check_significance_glmm.py \
      --frame analysis/sweep_frame.csv --method gee

  PYTHONPATH=. .venv/bin/python scripts/check_significance_glmm.py \
      --frame analysis/sweep_frame.csv --method bayes \
      --compare-against analysis/significance_report.csv

Pass `--compare-against analysis/significance_report.csv` to also print a
per-cell diff against the permutation test's own results (direction
agreement, and -- `--method gee` only -- which cells disagree on
significance at the same nominal alpha) rather than just this script's own
numbers in isolation.
"""

import argparse

import pandas as pd

from graphtalk import analysis
from graphtalk import hierarchical_model
from graphtalk import mixed_models


def _print_gee_table(result: pd.DataFrame) -> None:
  print(f"{'model':<16}{'condition':<12}{'delta':>10}{'95% CI':>22}"
        f"{'p-value':>10}  converged")
  for _, row in result.iterrows():
    ci = f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
    print(f"{row['model']:<16}{row['condition']:<12}{row['delta']:>+10.4f}"
          f"{ci:>22}{row['p_value']:>10.4f}  {row['converged']}")


def _print_bayes_table(result: pd.DataFrame) -> None:
  print(f"{'model':<16}{'condition':<12}{'total':>10}{'94% HDI':>22}"
        f"{'P(>0)':>8}{'r_hat':>8}{'ess':>8}")
  for _, row in result.iterrows():
    hdi = f"[{row['total_hdi_low']:+.3f}, {row['total_hdi_high']:+.3f}]"
    print(f"{row['model']:<16}{row['condition']:<12}{row['total_mean']:>+10.4f}"
          f"{hdi:>22}{row['prob_positive']:>8.3f}{row['r_hat']:>8.3f}"
          f"{row['ess_bulk']:>8.0f}")


def _flag_convergence_issues(result: pd.DataFrame) -> None:
  bad_rhat = result[result["r_hat"] > 1.01]
  low_ess = result[result["ess_bulk"] < 400]
  if result["n_divergences"].iloc[0] > 0 if len(result) else False:
    print(f"\n  WARNING: {result['n_divergences'].iloc[0]} divergent transitions -- "
          f"see graphtalk/hierarchical_model.py's module docstring before "
          f"trusting any of this run's numbers.")
  if not bad_rhat.empty:
    print(f"\n  WARNING: {len(bad_rhat)} rows have r_hat > 1.01 (not converged):")
    print(bad_rhat[["model", "condition", "r_hat"]].to_string(index=False))
  if not low_ess.empty:
    print(f"\n  WARNING: {len(low_ess)} rows have ess_bulk < 400 "
          f"(unreliable r_hat/HDI at this sample size):")
    print(low_ess[["model", "condition", "ess_bulk"]].to_string(index=False))


def _compare_bayes_against(result: pd.DataFrame, report_path: str) -> None:
  """Direction-only comparison for `--method bayes`: `total_mean` is a
  log-odds effect, not a probability difference, so a raw-p-value/alpha
  threshold comparison the way `_compare_against` does for `--method gee`
  isn't apples to apples -- sign agreement is."""
  report = pd.read_csv(report_path)
  perm = report[
      (report["arm"] == "main_sweep") & (report["bound"] == "excluded")
      & (report["metric"] == "exact")
      & (report["group"] != "pooled across all models")
  ]
  joined = result.merge(
      perm[["group", "condition", "delta"]].rename(
          columns={"group": "model", "delta": "perm_delta"}
      ),
      on=["model", "condition"], how="inner",
  )
  same_sign = (joined["total_mean"] * joined["perm_delta"]) >= 0
  print(f"\n=== Direction comparison against {report_path} "
        f"(main_sweep/excluded, sign only) ===")
  print(f"  direction agreement: {int(same_sign.sum())}/{len(joined)}")
  if not same_sign.all():
    print(joined[~same_sign][["model", "condition", "total_mean", "perm_delta"]]
          .to_string(index=False))


def _compare_against(result: pd.DataFrame, report_path: str, alpha: float) -> None:
  """Per-cell diff against `check_significance.py`'s own report: direction
  agreement (sign of `delta` vs `delta`) and which cells disagree on
  "significant at nominal alpha" (GEE's raw `p_value <= alpha` -- not
  `bh_significant`, which is corrected across a different family and isn't
  the right comparison for "did an independent method see the same thing";
  see the module docstring)."""
  report = pd.read_csv(report_path)
  # `metric == "exact"` matters, not just `arm`/`bound`: `mae` rows also
  # carry `arm="main_sweep", bound="excluded"` (see
  # `scripts/check_significance.py::_report_mae`), and share the same
  # (model, condition) key -- without this filter they'd multiply-match
  # against this script's rows in the merge below (one real exact match
  # plus up to three mae-row duplicates per cell, one per task).
  perm = report[
      (report["arm"] == "main_sweep") & (report["bound"] == "excluded")
      & (report["metric"] == "exact")
      & (report["group"] != "pooled across all models")
  ]
  joined = result.merge(
      perm[["group", "condition", "delta", "p_value"]].rename(
          columns={"group": "model", "delta": "perm_delta", "p_value": "perm_p"}
      ),
      on=["model", "condition"], how="left",
  )
  if joined["perm_delta"].isna().any():
    missing = joined[joined["perm_delta"].isna()][["model", "condition"]]
    print(f"\n  {len(missing)} (model, condition) cells in the GEE output have "
          f"no match in {report_path} -- skipped from the comparison below:")
    print(missing.to_string(index=False))
    joined = joined.dropna(subset=["perm_delta"])

  same_sign = (joined["delta"] * joined["perm_delta"]) >= 0
  gee_sig = joined["p_value"] <= alpha
  perm_sig = joined["perm_p"] <= alpha
  print(f"\n=== Comparison against {report_path} "
        f"(main_sweep/excluded, raw p <= {alpha}) ===")
  print(f"  direction agreement: {int(same_sign.sum())}/{len(joined)}")
  print(f"  raw-significance agreement: {int((gee_sig == perm_sig).sum())}/{len(joined)}")
  disagreements = joined[gee_sig != perm_sig]
  if not disagreements.empty:
    print(f"\n  {len(disagreements)} cells disagree on raw significance "
          f"(GEE pools all 6 conditions per model in one fit; the "
          f"permutation test tests each condition alone -- a disagreement "
          f"here is the kind of thing this cross-check exists to surface, "
          f"not necessarily an error in either method):")
    for _, row in disagreements.iterrows():
      print(f"    {row['model']:<16}{row['condition']:<12}"
            f"GEE p={row['p_value']:.4f} (delta={row['delta']:+.4f})  "
            f"perm p={row['perm_p']:.4f} (delta={row['perm_delta']:+.4f})")
  if not same_sign.all():
    flipped = joined[~same_sign]
    print(f"\n  {len(flipped)} cells disagree on DIRECTION (sign of delta) "
          f"-- inspect these closely, this is a stronger disagreement than "
          f"a significance-threshold difference:")
    print(flipped[["model", "condition", "delta", "perm_delta"]].to_string(index=False))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frame", default="analysis/sweep_frame.csv")
  parser.add_argument("--method", choices=("gee", "bayes"), default="gee")
  parser.add_argument("--metric", default="exact",
                       help="the sweep_frame.csv column to model (default 'exact')")
  parser.add_argument("--compare-against", default=None,
                       help="optional path to a check_significance.py --out "
                            "CSV (e.g. analysis/significance_report.csv) to "
                            "diff this script's results against, cell by cell")
  parser.add_argument("--alpha", type=float, default=0.05,
                       help="raw-significance threshold used only for the "
                            "--compare-against agreement summary, not a "
                            "multiplicity-corrected cutoff")
  parser.add_argument("--out", default=None,
                       help="optional path to write the results table as CSV "
                            "-- pass the base name (e.g. "
                            "analysis/glmm_report.csv); "
                            "graphtalk.analysis.tagged_path suffixes it "
                            "automatically for a non-integer --frame scheme "
                            "(e.g. .got.csv), same convention "
                            "check_significance.py's --out uses, so a GOT "
                            "run can never collide with or overwrite an "
                            "integer-scheme output written to the same base "
                            "name")
  parser.add_argument("--draws", type=int, default=1000,
                       help="--method bayes only: posterior draws per chain")
  parser.add_argument("--tune", type=int, default=1000,
                       help="--method bayes only: tuning steps per chain")
  parser.add_argument("--chains", type=int, default=2,
                       help="--method bayes only: number of MCMC chains")
  parser.add_argument("--target-accept", type=float, default=0.9,
                       help="--method bayes only: NUTS target acceptance rate")
  parser.add_argument("--seed", type=int, default=1234)
  args = parser.parse_args()

  frame = pd.read_csv(args.frame)
  # Same scheme-tagging convention as `check_significance.py`'s --out --
  # computed here, once, from the frame actually loaded, not assumed from
  # --frame's filename.
  scheme = analysis.frame_node_naming(frame)
  # Same scope as `graphtalk.mixed_models._main_sweep_scope` -- main sweep
  # only. Non-terminating rows are NOT dropped here (nor anywhere in this
  # script) -- `graphtalk.analysis.build_frame` already forces their
  # `exact`/`primary` to 0.0 upstream, so they enter both `--method gee`
  # and `--method bayes` as ordinary rows, matching
  # `check_significance.py`'s own main-sweep scope exactly (see that
  # module's docstring). `fit_gee_all_models` re-derives this same scope
  # itself per model; `hierarchical_model.fit` fits every model jointly in
  # one call and does no filtering of its own, so it has to happen here.
  main_sweep = frame[~frame["is_think"]]

  if args.method == "bayes":
    trace, index = hierarchical_model.fit(
        main_sweep, metric=args.metric, draws=args.draws, tune=args.tune,
        chains=args.chains, target_accept=args.target_accept, seed=args.seed,
    )
    result = hierarchical_model.fit_summary(trace, index)
    _print_bayes_table(result)
    _flag_convergence_issues(result)
    if args.compare_against:
      _compare_bayes_against(result, args.compare_against)
    if args.out:
      out = analysis.tagged_path(args.out, scheme)
      result.to_csv(out, index=False)
      print(f"\nwrote {len(result)} rows to {out}")
    return

  result = mixed_models.fit_gee_all_models(main_sweep, metric=args.metric)
  _print_gee_table(result)

  if not result["converged"].all():
    unconverged = result[~result["converged"]]
    print(f"\n  WARNING: {len(unconverged)} fits did not converge -- see "
          f"graphtalk/mixed_models.py's module docstring before trusting "
          f"their numbers:")
    print(unconverged[["model", "condition"]].to_string(index=False))

  if args.compare_against:
    _compare_against(result, args.compare_against, args.alpha)

  if args.out:
    out = analysis.tagged_path(args.out, scheme)
    result.to_csv(out, index=False)
    print(f"\nwrote {len(result)} rows to {out}")


if __name__ == "__main__":
  main()
