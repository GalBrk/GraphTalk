"""Track 2.1's required dry run: `scripts/recommend_count.py`'s
`n_clusters_needed = n_clusters * (mde_delta / delta) ** 2` is a closed-form
asymptotic extrapolation (`MDE ~ 1/sqrt(N)`), not a re-run of the real
simulation -- this script checks that extrapolation against an actual
bootstrap power simulation at the recommended size, for real cells, before
trusting it to guide a `--count` choice that costs real GPU time.

**Method.** For each cell, pulls its real `(control, treatment,
cluster_ids)` from `analysis/sweep_frame.csv` (same reconstruction
`scripts/benchmark_mde.py` uses), then runs many trials where each trial:
bootstrap-resamples `n_clusters_needed` clusters (not `n_clusters`,
i.e. genuinely more than currently exist -- oversampling with replacement
from the existing clusters as a proxy for what new clusters might look
like, the same proxy the existing MDE simulation already relies on for
*fewer*-than-current clusters), injects the observed `delta` as the true
effect the same way `minimum_detectable_effect_clustered`'s own
`_power_and_realized` does (a fresh Bernoulli draw at `clip(control +
delta)`, not a deterministic shift -- see that function's docstring for
why), and checks whether the resulting permutation test reaches `p <=
0.05`. The fraction of trials that do is the simulated power at the
recommended size; it should land near the 80% target the closed-form
formula was aiming for.

  PYTHONPATH=. .venv/bin/python scripts/validate_recommend_count.py --n-cells 4
"""

import argparse
import random

import pandas as pd

from graphtalk import significance
from scripts import check_significance as cs
from scripts import recommend_count


def _resample_n(clusters: list, n_draws: int, rng: random.Random):
  """Like `graphtalk.significance._resample_clusters`, but resamples
  `n_draws` clusters (which may exceed `len(clusters)`) instead of always
  exactly `len(clusters)` -- the primitive `_resample_clusters` doesn't
  expose, needed here specifically to simulate a *larger* sample than
  currently exists."""
  items, draw_ids = [], []
  for draw_idx in range(n_draws):
    drawn = clusters[rng.randrange(len(clusters))]
    items.extend(drawn)
    draw_ids.extend([draw_idx] * len(drawn))
  return items, draw_ids


def simulate_power_at_n(
    control, treatment, cluster_ids, delta: float, n_clusters_target: int,
    n_trials: int = 200, alpha: float = 0.05, n_perm: int = 300, seed: int = 1234,
) -> float:
  """Empirical power to detect `delta` at `n_clusters_target` clusters,
  bootstrap-simulated from the real `(control, treatment)` pairs given.
  """
  by_cluster: dict = {}
  for cid, c, t in zip(cluster_ids, control, treatment):
    by_cluster.setdefault(cid, []).append((c, t))
  clusters = list(by_cluster.values())
  rng = random.Random(seed)
  hits = 0
  for _ in range(n_trials):
    pairs, draw_ids = _resample_n(clusters, n_clusters_target, rng)
    c_star = [c for c, _ in pairs]
    p_star = [min(1.0, c + delta) if delta >= 0 else max(0.0, c + delta)
              for c in c_star]
    t_star = [1.0 if rng.random() < p else 0.0 for p in p_star]
    result = significance.paired_permutation_test_clustered(
        c_star, t_star, draw_ids, n_perm=n_perm, seed=rng.randrange(2**31)
    )
    if result["p_value"] <= alpha:
      hits += 1
  return hits / n_trials


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frame", default="analysis/sweep_frame.csv")
  parser.add_argument("--report", default="analysis/significance_report.csv")
  parser.add_argument("--n-cells", type=int, default=4)
  parser.add_argument("--n-trials", type=int, default=200)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument(
      "--max-n-clusters-target", type=float, default=10_000,
      help="skip recommendations whose n_clusters_needed exceeds this -- "
           "this script's own simulation cost scales roughly linearly in "
           "n_clusters_target (pure-Python resampling + permutation test, "
           "no vectorization), so the largest recommendations here (real "
           "runs have needed 30,000+ clusters) would take hours to "
           "simulate on their own and are already well beyond the "
           "published 500-graph cap regardless -- not a size anyone would "
           "actually request, so not worth validating precisely.")
  args = parser.parse_args()

  frame = pd.read_csv(args.frame)
  report = pd.read_csv(args.report)
  recommendations = recommend_count.recommend(report)
  finite = recommendations[
      recommendations["recommended_count"].notna()
      & (recommendations["n_clusters_needed"] <= args.max_n_clusters_target)
  ]
  if finite.empty:
    print(f"No recommendations with n_clusters_needed <= {args.max_n_clusters_target}; "
          f"nothing to validate.")
    return
  sample = finite.sample(n=min(args.n_cells, len(finite)), random_state=args.seed)

  print(f"Validating {len(sample)} recommendations against a real bootstrap "
        f"power simulation at the recommended size (target: ~80% power)\n")
  powers = []
  for _, row in sample.iterrows():
    cell = {
        "arm": "main_sweep", "group": row["model"], "condition": row["condition"],
        "bound": "excluded",
    }
    control, treatment, cluster_ids = cs._paired_values(
        frame[(~frame["is_think"]) & (frame["failure_type"] != "non_terminating")
              & (frame["model_family"] == row["model"])],
        row["condition"], "exact",
    )
    n_target = round(row["n_clusters_needed"])
    power = simulate_power_at_n(
        control, treatment, cluster_ids, row["delta"], n_target,
        n_trials=args.n_trials, seed=args.seed,
    )
    powers.append(power)
    verdict = "OK" if 0.65 <= power <= 0.95 else "CHECK -- outside a loose [0.65, 0.95] band around the 0.8 target"
    print(f"  {row['model']:<14}{row['condition']:<12} "
          f"n_clusters {row['n_clusters']:.0f} -> recommended {n_target} "
          f"(delta={row['delta']:+.4f})  simulated power at recommended N: "
          f"{power:.3f}  [{verdict}]")

  # Report which direction any misses lean -- overshoot (recommended count
  # is *more* conservative than the closed-form target implies) is a safe
  # failure mode for planning a real sweep; undershoot (recommended count
  # under-delivers power) is the one that would actually matter and should
  # be investigated before trusting `recommend_count.py`'s numbers as-is.
  over = sum(1 for p in powers if p > 0.95)
  under = sum(1 for p in powers if p < 0.65)
  print(f"\n{over}/{len(powers)} overshot the loose band (extrapolation was "
        f"conservative -- more power than the 80% target), "
        f"{under}/{len(powers)} undershot it (extrapolation was optimistic "
        f"-- less power than the 80% target).")


if __name__ == "__main__":
  main()
