"""Pooled significance tests over paired sweep rows.

`graphtalk.scoring.mcnemar` is exact but per-cell: at 30 paired instances per
(task, style, condition) cell there is often too little discordance to say
anything, and there is no correction for testing 288 such cells at once (see
docs/sweep-findings.md, "The McNemar analysis is underpowered"). The
functions here pool pairs across task and style instead of testing 288 tiny
groups, trading per-task granularity for statistical power.

Hand-rolled rather than built on scipy/statsmodels, for the same reason
`scoring.mcnemar` avoids scipy's chi-square approximation: this project has
deliberately kept scipy out of its dependency set (see
docs/plans/primer-computation.md), and a permutation test on a sign-flippable
paired difference needs nothing beyond a source of randomness.

Pooling across task and style (and, when comparing across models, across
model too) means the same graph instance recurs many times in one pooled
sample -- `paired_permutation_test`/`cluster_bootstrap_ci` treat every row as
an independent draw, which overstates the effective sample size whenever
rows sharing an `instance_id` are correlated (a graph the model finds easy,
or a condition that happens to suit its structure, moves every row sharing
that instance in the same direction). `paired_permutation_test_clustered`/
`cluster_bootstrap_ci_clustered` correct for that by resampling/sign-flipping
whole clusters (shared `instance_id`s) rather than individual rows -- see
`scripts/check_significance.py`, which threads `instance_id` through as the
cluster key. The unclustered functions are kept, not replaced: they're still
correct wherever no cluster_id repeats.
"""

import itertools
import random

# Below this many clusters, `*_clustered` functions enumerate every sign
# pattern exactly instead of drawing `n_perm`/`n_boot` random ones, matching
# `scoring.mcnemar`'s own preference for the exact test over an approximation
# at small n (there, the discordant count; here, the cluster count).
_EXACT_CLUSTER_THRESHOLD = 20


def paired_permutation_test(
    control, treatment, n_perm: int = 10_000, seed: int = 0
) -> dict:
  """Pooled generalization of `scoring.mcnemar`'s exact sign test.

  `control`/`treatment` are aligned sequences of paired outcomes (typically
  0/1, pooled across every task and style a pair was observed in, not just
  one task/style cell). The null hypothesis is that each pair's control and
  treatment values are exchangeable, so the null distribution of the mean
  difference is built by randomly flipping the sign of each pair's observed
  difference. Uses the add-one correction (North et al. 2002) so a Monte
  Carlo p-value is never reported as exactly 0.

  Raises on a length mismatch, matching `scoring.mcnemar`'s guard against a
  silent misalignment that would invert the pairing.
  """
  control, treatment = list(control), list(treatment)
  if len(control) != len(treatment):
    raise ValueError(
        f"paired test needs equal lengths, got {len(control)} "
        f"and {len(treatment)}"
    )
  n = len(control)
  if n == 0:
    return {"n_pairs": 0, "observed_diff": 0.0, "p_value": 1.0}
  diffs = [t - c for c, t in zip(control, treatment)]
  observed = sum(diffs) / n
  observed_abs = abs(observed)
  rng = random.Random(seed)
  at_least_as_extreme = 0
  for _ in range(n_perm):
    permuted = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
    if abs(permuted) >= observed_abs - 1e-12:
      at_least_as_extreme += 1
  p_value = (at_least_as_extreme + 1) / (n_perm + 1)
  return {"n_pairs": n, "observed_diff": observed, "p_value": p_value}


def paired_permutation_test_clustered(
    control, treatment, cluster_ids, n_perm: int = 10_000, seed: int = 0
) -> dict:
  """Like `paired_permutation_test`, but flips every pair sharing a
  `cluster_ids` value together, not independently.

  Pairs that share a cluster -- the same graph instance seen across
  multiple styles or models -- are not independent replicates: if that
  instance is unusually easy, or the condition happens to help on its
  particular structure, every pair sharing it tends to move together.
  Flipping cluster-by-cluster rather than pair-by-pair preserves that
  dependence under the null, which is what keeps the p-value from being
  anti-conservative on pooled data. Reduces to `paired_permutation_test`
  when every `cluster_ids` value is unique (each cluster then holds exactly
  one pair, so cluster-level and pair-level sign-flipping coincide).

  Below `_EXACT_CLUSTER_THRESHOLD` clusters, enumerates every sign pattern
  exactly instead of drawing `n_perm` random ones -- see the module-level
  comment on `_EXACT_CLUSTER_THRESHOLD`.
  """
  control, treatment = list(control), list(treatment)
  cluster_ids = list(cluster_ids)
  if not (len(control) == len(treatment) == len(cluster_ids)):
    raise ValueError(
        f"paired test needs equal lengths, got {len(control)} control, "
        f"{len(treatment)} treatment, {len(cluster_ids)} cluster_ids"
    )
  n = len(control)
  if n == 0:
    return {"n_pairs": 0, "n_clusters": 0, "observed_diff": 0.0, "p_value": 1.0}
  diffs = [t - c for c, t in zip(control, treatment)]
  # Sum of diffs per cluster -- the unit that actually gets sign-flipped.
  by_cluster: dict = {}
  for cluster_id, diff in zip(cluster_ids, diffs):
    by_cluster[cluster_id] = by_cluster.get(cluster_id, 0.0) + diff
  cluster_sums = list(by_cluster.values())
  m = len(cluster_sums)
  observed = sum(diffs) / n
  observed_abs = abs(observed)

  def _extreme(cluster_total: float) -> bool:
    return abs(cluster_total / n) >= observed_abs - 1e-12

  at_least_as_extreme = 0
  if m <= _EXACT_CLUSTER_THRESHOLD:
    total_patterns = 0
    for signs in itertools.product((1, -1), repeat=m):
      total = sum(s * v for s, v in zip(signs, cluster_sums))
      if _extreme(total):
        at_least_as_extreme += 1
      total_patterns += 1
    p_value = at_least_as_extreme / total_patterns
  else:
    rng = random.Random(seed)
    for _ in range(n_perm):
      total = sum(v if rng.random() < 0.5 else -v for v in cluster_sums)
      if _extreme(total):
        at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (n_perm + 1)
  return {"n_pairs": n, "n_clusters": m, "observed_diff": observed, "p_value": p_value}


def cluster_bootstrap_ci_clustered(
    control, treatment, cluster_ids, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> dict:
  """Resamples whole clusters (e.g. graph instances) with replacement,
  carrying every pair that shares a cluster along together -- a real
  cluster bootstrap, unlike `cluster_bootstrap_ci`'s per-pair resampling,
  which understates variance when pairs sharing a cluster are correlated
  (see `paired_permutation_test_clustered`). Reduces to
  `cluster_bootstrap_ci` when every `cluster_ids` value is unique.
  """
  control, treatment = list(control), list(treatment)
  cluster_ids = list(cluster_ids)
  if not (len(control) == len(treatment) == len(cluster_ids)):
    raise ValueError(
        f"paired CI needs equal lengths, got {len(control)} control, "
        f"{len(treatment)} treatment, {len(cluster_ids)} cluster_ids"
    )
  n = len(control)
  if n == 0:
    return {"point_estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_clusters": 0}
  diffs = [t - c for c, t in zip(control, treatment)]
  by_cluster: dict = {}
  for cluster_id, diff in zip(cluster_ids, diffs):
    by_cluster.setdefault(cluster_id, []).append(diff)
  clusters = list(by_cluster.values())
  m = len(clusters)
  point = sum(diffs) / n
  rng = random.Random(seed)
  boot_means = []
  for _ in range(n_boot):
    resampled = []
    for _ in range(m):
      resampled.extend(clusters[rng.randrange(m)])
    boot_means.append(sum(resampled) / len(resampled))
  boot_means.sort()
  lo = boot_means[int((alpha / 2) * n_boot)]
  hi = boot_means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
  return {"point_estimate": point, "ci_low": lo, "ci_high": hi, "n_clusters": m}


def cluster_bootstrap_ci(
    control, treatment, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> dict:
  """Bootstrap CI on the paired mean difference, resampling by pair.

  Resampling whole pairs (rather than control and treatment values
  independently) is what "cluster" means here -- each resample keeps a
  pair's control and treatment value moving together, since they come from
  the same instance and are not independent draws.
  """
  control, treatment = list(control), list(treatment)
  if len(control) != len(treatment):
    raise ValueError(
        f"paired CI needs equal lengths, got {len(control)} "
        f"and {len(treatment)}"
    )
  n = len(control)
  if n == 0:
    return {"point_estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
  diffs = [t - c for c, t in zip(control, treatment)]
  point = sum(diffs) / n
  rng = random.Random(seed)
  boot_means = []
  for _ in range(n_boot):
    boot_means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
  boot_means.sort()
  lo = boot_means[int((alpha / 2) * n_boot)]
  hi = boot_means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
  return {"point_estimate": point, "ci_low": lo, "ci_high": hi}


def benjamini_hochberg(p_values, q: float = 0.05) -> list:
  """Benjamini-Hochberg step-up FDR correction.

  Returns a same-length list of reject flags: True where the null is
  rejected at FDR level `q`. Standard step-up procedure -- sort ascending,
  find the largest rank `k` whose p-value is <= (k/m)*q, reject that one and
  every smaller one.
  """
  p_values = list(p_values)
  m = len(p_values)
  if m == 0:
    return []
  order = sorted(range(m), key=lambda i: p_values[i])
  cutoff_rank = 0
  for rank, idx in enumerate(order, start=1):
    if p_values[idx] <= (rank / m) * q:
      cutoff_rank = rank
  reject = [False] * m
  for idx in order[:cutoff_rank]:
    reject[idx] = True
  return reject
