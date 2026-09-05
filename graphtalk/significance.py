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

Pooling across task (and, when comparing across models, across model too)
means the same graph recurs many times in one pooled sample --
`paired_permutation_test`/`cluster_bootstrap_ci` treat every row as an
independent draw, which overstates the effective sample size whenever rows
sharing a graph are correlated (a graph the model finds easy, or a
condition that happens to suit its structure, moves every row sharing that
graph in the same direction). `paired_permutation_test_clustered`/
`cluster_bootstrap_ci_clustered` correct for that by resampling/sign-flipping
whole clusters rather than individual rows -- see `scripts/check_significance.py`,
which threads `(model, graph_index)` through as the cluster key.

What that key deliberately is and isn't. It is **not** `instance_id`: that
string is `"<task>/<index>"`, and the six tasks sharing an index are the
*same graph* asked six different questions, so keying on the full string
put exactly one pair in every cluster and made this module's clustered
variants no-ops on the real sweep. (The justification that used to sit here
was repetition across prompt *styles*, which was true until the `zero_cot`
purge left one style; nothing was updated to point at the per-task
repetition that remained.) It is also **not** the bare graph index: a
cluster never spans more than one model, since different model families'
errors on the same graph number are not assumed to correlate as strongly as
one model's own repeated answers to it.

The unclustered functions are kept, not replaced: they're still correct
wherever no cluster_id repeats.
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

  Pairs that share a cluster -- the same graph seen under several tasks,
  for one model (callers key clusters by `(model, graph_index)`, never by
  the `"<task>/<index>"` instance id, which would put one pair in each
  cluster) -- are not independent replicates: if that graph is unusually
  easy, or the condition happens to help on its particular structure, every
  pair sharing it tends to move together.
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


def _resample_clusters(clusters: list, rng: random.Random):
  """One bootstrap draw: resamples `len(clusters)` clusters with
  replacement from `clusters` (each element a list of same-cluster items --
  diffs, for `cluster_bootstrap_ci_clustered`, or `(control, treatment)`
  pairs, for `minimum_detectable_effect_clustered`).

  Returns `(items, draw_ids)`: `items` is every item from each drawn
  cluster concatenated, in draw order; `draw_ids` labels each item with
  which of the `m` draws (0..m-1) produced it, not the original cluster id
  -- two draws of the same original cluster must be treated as two separate
  clusters by anything that clusters on `draw_ids` afterward (a resampled
  duplicate is not more evidence about the same instance, it is two
  hypothetical instances that happened to look alike), which is exactly
  what `minimum_detectable_effect_clustered`'s inner significance test
  needs and `cluster_bootstrap_ci_clustered` doesn't (it only reads the
  pooled mean, so it discards `draw_ids`).
  """
  m = len(clusters)
  items, draw_ids = [], []
  for draw_idx in range(m):
    drawn = clusters[rng.randrange(m)]
    items.extend(drawn)
    draw_ids.extend([draw_idx] * len(drawn))
  return items, draw_ids


def cluster_bootstrap_ci_clustered(
    control, treatment, cluster_ids, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> dict:
  """Resamples whole clusters (e.g. one model's six per-task rows on one
  graph) with replacement, carrying every pair that shares a cluster along
  together -- a real
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
    resampled, _draw_ids = _resample_clusters(clusters, rng)
    boot_means.append(sum(resampled) / len(resampled))
  boot_means.sort()
  lo = boot_means[int((alpha / 2) * n_boot)]
  hi = boot_means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
  return {"point_estimate": point, "ci_low": lo, "ci_high": hi, "n_clusters": m}


def _search_one_direction(
    power_and_realized, initial_hi: float, power_target: float, n_steps: int, sign: int,
) -> dict:
  """Geometric-expansion-then-bisection search for the smallest `|delta|`
  in one direction (`sign=+1` for a candidate improvement, `sign=-1` for a
  candidate harm) reaching `power_target`, calling `power_and_realized
  (delta)` exactly the way the un-refactored single-direction search
  always did. Shared by both directions of
  `minimum_detectable_effect_clustered`'s bidirectional search -- this is
  the same expansion/bisection logic that direction always used, extracted
  so it can run twice instead of duplicated. `sign` only ever multiplies
  the magnitude passed to `power_and_realized`; the search itself doesn't
  know or care which direction it's sweeping.
  """
  hi = max(0.05, initial_hi)
  power_hi, realized_hi = power_and_realized(sign * hi)
  expansions = 0
  while power_hi < power_target and hi < 1.0 and expansions < 10:
    hi = min(1.0, hi * 2)
    power_hi, realized_hi = power_and_realized(sign * hi)
    expansions += 1
  if power_hi < power_target:
    return {
        "delta": None, "realized_diff": realized_hi,
        "note": "MDE exceeds 1.0 at this power target",
    }
  lo = 0.0
  for _ in range(n_steps):
    mid = (lo + hi) / 2
    power_mid, realized_mid = power_and_realized(sign * mid)
    if power_mid >= power_target:
      hi, realized_hi = mid, realized_mid
    else:
      lo = mid
  return {"delta": sign * hi, "realized_diff": realized_hi, "note": None}


def minimum_detectable_effect_clustered(
    control, treatment, cluster_ids, initial_hi: float, alpha: float = 0.05,
    power_target: float = 0.8, n_replicates: int = 200, n_perm: int = 500,
    n_steps: int = 8, seed=0, direction: str = "both",
) -> dict:
  """The smallest additive shift `delta` such that, if the true effect on
  data shaped like this row were `delta`, `paired_permutation_test_clustered`
  would detect it (`p <= alpha`) at least `power_target` of the time.

  `direction` controls which sign(s) of `delta` are searched:
  `"positive"` (a candidate *improvement*, the only direction this function
  originally searched), `"negative"` (a candidate *harm*), or `"both"`
  (the default -- runs both searches, since a near-ceiling or near-floor
  control has very different headroom in each direction and a caller
  reading only the positive side can't tell "no room to improve" apart
  from "no room to get worse either", which are very different claims;
  see `scripts/check_significance.py`'s `near_ceiling`). The positive
  direction's `delta`/`realized_diff`/`note` keys keep their original
  names and, for the same `seed`, their original values -- the positive
  search runs first and is unaffected by whether the negative search runs
  afterward, since they draw from the same `rng` stream in sequence rather
  than sharing draws. The negative direction's results are the same three
  fields suffixed `_negative`, `None` when not searched (`direction
  != "negative" and direction != "both"`) or when there were no paired
  rows to begin with.

  Answers a question `bh_significant=False` alone can't: is this a real
  null, or just not enough power to see one. Simulation-based, not a
  formula -- for a candidate `delta`, each of `n_replicates` trials
  bootstrap-resamples whole clusters from this row's own real
  `(control, treatment)` data (`_resample_clusters`, the same resampling
  unit `cluster_bootstrap_ci_clustered` uses), injects the shift by
  treating `clip(control* + delta, 0, 1)` as a *probability* and drawing a
  fresh Bernoulli outcome for `treatment*` from it -- not a deterministic
  `treatment* = clip(...)`, which would move every pair by exactly `delta`
  with no exceptions and make the permutation test read "every diff shares
  one sign" as maximally extreme regardless of how small `delta` was,
  collapsing the search toward implausibly tiny deltas (caught this way via
  a smoke test before it shipped). The fresh draw is what makes a smaller
  `delta` genuinely harder to detect than a larger one, which is the whole
  point of an MDE. `delta=0` reproduces a true null (the draw's probability
  is just `control*` itself). Reruns `paired_permutation_test_clustered` on
  the injected data, clustering on the resampled draw index (two resampled
  copies of the same original cluster are two hypothetical instances, not
  one). `power(delta)` is the fraction of trials with `p <= alpha`.

  Clipping is deliberate, not a limitation to work around: a near-ceiling
  or near-floor control (see `scripts/check_significance.py`'s
  `near_ceiling`) has little room to move, so even a large `delta`
  produces a small *realized* shift once clipping binds -- correctly
  reflecting reduced detectability there rather than hiding it. Returns
  both `delta` (the swept parameter at convergence) and `realized_diff`
  (the replicates' actual mean `treatment* - control*` at that `delta`),
  since the two can differ and the gap between them is itself informative.

  Search (`_search_one_direction`, run once per requested direction):
  expands `hi` geometrically from `max(0.05, initial_hi)` -- callers should
  pass their own bootstrap CI width as `initial_hi`, a good anchor that
  avoids wasting steps on obviously-always-detectable deltas near 1.0,
  computed once rather than redundantly inside this function -- until
  `power(hi) >= power_target` or `hi` reaches 1.0, then bisects `n_steps`
  times between 0 and that `hi`. The negative-direction search is the same
  procedure with every candidate delta negated before being handed to
  `_power_and_realized`; `initial_hi` (a magnitude, not a signed value) is
  reused as-is for both directions' starting point.

  Power estimates carry real Monte Carlo noise (SE ~= 0.03 at
  `n_replicates=200` near power=0.8), so bisection does not converge
  perfectly monotonically -- the same tradeoff this module already accepts
  for its own p-values, not a bug specific to this function. One `rng`
  stream drives every resample and every inner test's seed, so the whole
  search is deterministic given `seed`.
  """
  if direction not in ("positive", "negative", "both"):
    raise ValueError(
        f"unknown direction: {direction!r}; known: 'positive', 'negative', 'both'"
    )
  control, treatment = list(control), list(treatment)
  cluster_ids = list(cluster_ids)
  if not (len(control) == len(treatment) == len(cluster_ids)):
    raise ValueError(
        f"MDE needs equal lengths, got {len(control)} control, "
        f"{len(treatment)} treatment, {len(cluster_ids)} cluster_ids"
    )
  if len(control) == 0:
    return {
        "delta": None, "realized_diff": None, "power_target": power_target,
        "note": "no paired rows",
        "delta_negative": None, "realized_diff_negative": None,
        "note_negative": "no paired rows" if direction != "positive" else None,
    }

  by_cluster: dict = {}
  for cid, c, t in zip(cluster_ids, control, treatment):
    by_cluster.setdefault(cid, []).append((c, t))
  clusters = list(by_cluster.values())
  rng = random.Random(seed)

  def _power_and_realized(delta: float) -> tuple:
    hits = 0
    realized_total, realized_n = 0.0, 0
    for _ in range(n_replicates):
      pairs, draw_ids = _resample_clusters(clusters, rng)
      c_star = [c for c, _ in pairs]
      # `clip(c + delta)` is a *probability*, not a deterministic value: a
      # fresh Bernoulli draw at that probability is what makes this a
      # believable synthetic outcome instead of an artifact. A deterministic
      # shift (every pair moving by exactly `delta`, no exceptions) makes
      # every nonzero diff share one sign no matter how small `delta` is --
      # the permutation test then reads "all diffs agree in sign" as
      # maximally extreme regardless of magnitude, so the search collapses
      # toward implausibly tiny deltas (caught via a smoke test: a 50-pair
      # fixture with a strong true effect converged to delta=0.001, clearly
      # wrong). Real per-pair noise is what makes a smaller `delta` harder
      # to detect than a larger one, which is the entire point of an MDE.
      p_star = [min(1.0, c + delta) if delta >= 0 else max(0.0, c + delta)
                for c in c_star]
      t_star = [1.0 if rng.random() < p else 0.0 for p in p_star]
      result = paired_permutation_test_clustered(
          c_star, t_star, draw_ids, n_perm=n_perm, seed=rng.randrange(2**31)
      )
      if result["p_value"] <= alpha:
        hits += 1
      realized_total += sum(t - c for c, t in zip(c_star, t_star))
      realized_n += len(c_star)
    return hits / n_replicates, realized_total / realized_n

  # The positive search runs first (when requested) so its draws from
  # `rng` are identical, in the same order, to what the original
  # single-direction function always drew -- direction="both"'s positive
  # results are therefore byte-identical to a direction="positive"-only
  # call at the same seed, and to this function's pre-bidirectional
  # behavior.
  result = {"power_target": power_target}
  if direction in ("positive", "both"):
    positive = _search_one_direction(
        _power_and_realized, initial_hi, power_target, n_steps, sign=1
    )
    result["delta"] = positive["delta"]
    result["realized_diff"] = positive["realized_diff"]
    result["note"] = positive["note"]
  else:
    result["delta"] = None
    result["realized_diff"] = None
    result["note"] = None
  if direction in ("negative", "both"):
    negative = _search_one_direction(
        _power_and_realized, initial_hi, power_target, n_steps, sign=-1
    )
    result["delta_negative"] = negative["delta"]
    result["realized_diff_negative"] = negative["realized_diff"]
    result["note_negative"] = negative["note"]
  else:
    result["delta_negative"] = None
    result["realized_diff_negative"] = None
    result["note_negative"] = None
  return result


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
