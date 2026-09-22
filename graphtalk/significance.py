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
import math
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


def _resample_clusters(clusters: list, rng: random.Random, m: int | None = None):
  """One bootstrap draw: resamples `m` clusters (default `len(clusters)`,
  today's only use) with replacement from `clusters` (each element a list
  of same-cluster items -- diffs, for `cluster_bootstrap_ci_clustered`, or
  `(control, treatment)` pairs, for `minimum_detectable_effect_clustered`
  and `required_sample_size_clustered`).

  `m` is a separate parameter from `len(clusters)` specifically for
  `required_sample_size_clustered`, which asks "what if there were more
  (or fewer) clusters like these" -- resampling *more* draws than
  `len(clusters)` from the same pilot pool is exactly how that prospective
  question gets simulated from a smaller, already-collected sample.
  Defaulting to `len(clusters)` keeps every existing call site (which
  always wants "as many draws as clusters") unaffected.

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
  n = len(clusters)
  m = n if m is None else m
  items, draw_ids = [], []
  for draw_idx in range(m):
    drawn = clusters[rng.randrange(n)]
    items.extend(drawn)
    draw_ids.extend([draw_idx] * len(drawn))
  return items, draw_ids


def cluster_bootstrap_ci_clustered(
    control, treatment, cluster_ids, n_boot: int = 10_000, seed: int = 0,
    alpha: float = 0.05, min_discordant: int = 10,
) -> dict:
  """Resamples whole clusters (e.g. one model's six per-task rows on one
  graph) with replacement, carrying every pair that shares a cluster along
  together -- a real
  cluster bootstrap, unlike `cluster_bootstrap_ci`'s per-pair resampling,
  which understates variance when pairs sharing a cluster are correlated
  (see `paired_permutation_test_clustered`). Reduces to
  `cluster_bootstrap_ci` when every `cluster_ids` value is unique.

  **Returns `None` bounds when too few pairs disagree.** A percentile
  bootstrap needs enough distinct nonzero differences to resample; below
  `min_discordant` there aren't enough, and what comes back describes the
  resampling rather than the population. The degenerate end of that is
  visible in the superseded report: six rows published a 95% CI of
  `[0.000, 0.000]` -- every one a cell where no pair disagreed at all, so
  every resample returned the same zeros -- and cells as thin as 2
  disagreements in 180 pairs printed intervals that read as ordinary. The
  measured one-sided coverage in that regime is well under the nominal 95%,
  and the miss is on the harm side, which is exactly the side a reader uses
  such an interval as a safety bound.

  `None` rather than a zero-width interval because "no interval is
  estimable here" and "the effect is provably 0.000 either way" are
  completely different claims, and only the first is true. `n_discordant`
  is always reported so a caller can see why.
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
    return {"point_estimate": 0.0, "ci_low": None, "ci_high": None,
            "n_clusters": 0, "n_discordant": 0}
  diffs = [t - c for c, t in zip(control, treatment)]
  by_cluster: dict = {}
  for cluster_id, diff in zip(cluster_ids, diffs):
    by_cluster.setdefault(cluster_id, []).append(diff)
  clusters = list(by_cluster.values())
  m = len(clusters)
  point = sum(diffs) / n
  n_discordant = sum(1 for d in diffs if d != 0)
  if n_discordant < min_discordant:
    # Not enough distinct nonzero differences for a percentile bootstrap to
    # describe anything but its own resampling -- see the docstring.
    return {"point_estimate": point, "ci_low": None, "ci_high": None,
            "n_clusters": m, "n_discordant": n_discordant}
  rng = random.Random(seed)
  boot_means = []
  for _ in range(n_boot):
    resampled, _draw_ids = _resample_clusters(clusters, rng)
    boot_means.append(sum(resampled) / len(resampled))
  boot_means.sort()
  lo = boot_means[int((alpha / 2) * n_boot)]
  hi = boot_means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
  return {"point_estimate": point, "ci_low": lo, "ci_high": hi,
          "n_clusters": m, "n_discordant": n_discordant}


def _binom_tail_ge(x: int, n: int, p: float) -> float:
  """P(X >= x) for X ~ Binomial(n, p). Increasing in `p`."""
  return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
             for i in range(x, n + 1))


def _binom_tail_le(x: int, n: int, p: float) -> float:
  """P(X <= x) for X ~ Binomial(n, p). Decreasing in `p`."""
  return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
             for i in range(0, x + 1))


def _invert_monotone(fn, target: float, iters: int = 200) -> float:
  """The `p` in [0, 1] where the monotone `fn(p)` crosses `target`.

  200 halvings drive the bracket below double precision, so the answer is
  exact to the representation rather than to a tolerance picked here.
  """
  lo, hi = 0.0, 1.0
  increasing = fn(1.0) > fn(0.0)
  for _ in range(iters):
    mid = (lo + hi) / 2
    if (fn(mid) < target) == increasing:
      lo = mid
    else:
      hi = mid
  return (lo + hi) / 2


def exact_paired_ci(b: int, c: int, n_pairs: int, alpha: float = 0.05) -> dict:
  """Exact conditional CI for the paired difference `(c - b) / n_pairs`.

  `b` and `c` are the discordant counts in the two directions -- the same
  quantities `scoring.mcnemar` tests, `b` where the control was right and
  the treatment wrong, `c` the reverse. `n_pairs` counts every pair,
  concordant ones included, so the interval lands on the same scale as the
  point estimate a caller reports.

  **Prefer this to `cluster_bootstrap_ci` for paired binary outcomes.** That
  function resamples a difference taking only the three values -1, 0 and +1,
  so its percentile endpoints fall on a coarse lattice and its error rate
  misses nominal *non-monotonically*: simulated at n=380 with no true
  effect, a nominal 95% interval wrongly excluded zero 11.70% of the time at
  4 discordant pairs, 2.00% at 10, and 7.07% at 11. Because the misbehaviour
  does not decay with the count, no minimum-discordant threshold repairs it
  -- a cut at 10 suppresses cells at 6, 7, 9 and 10 that are conservative
  while still admitting 11.

  Conditioning removes the problem instead of thresholding it. Given
  `k = b + c` discordant pairs, `c ~ Binomial(k, theta)` and the null is
  `theta = 1/2`; a Clopper-Pearson interval on `theta` inverts the exact
  binomial test, so coverage is at least nominal at every `k` by
  construction, discreteness making it conservative rather than liberal. The
  same simulation puts this at or below 3.92% everywhere, and identical to
  the bootstrap at the counts where the bootstrap behaves. Mapping back,
  `delta = (k / n_pairs) * (2 * theta - 1)`.

  It is also the interval matching the test already used everywhere here:
  `scoring.mcnemar` is the exact binomial McNemar test on the same `b` and
  `c`. Pairing an exact test with a bootstrap interval was a mismatch.

  Hand-rolled rather than `scipy.stats.beta.ppf` because this module stays
  scipy-free (see `pyproject.toml`); it agrees with scipy to 1.3e-16 across
  every `(b, c)` with `b + c <= 28`.

  With no discordant pairs the interval is `[0.0, 0.0]`: conditional on
  nothing having disagreed there is no direction to bound. That is a fact
  about this sample, not about the population, which is why `n_discordant`
  is returned for the caller to render instead of a zero-width interval.
  """
  k = b + c
  if k == 0:
    return {"ci_low": 0.0, "ci_high": 0.0, "n_discordant": 0}
  lo_theta = (0.0 if c == 0 else
              _invert_monotone(lambda p: _binom_tail_ge(c, k, p), alpha / 2))
  hi_theta = (1.0 if c == k else
              _invert_monotone(lambda p: _binom_tail_le(c, k, p), alpha / 2))
  scale = k / n_pairs
  return {"ci_low": scale * (2 * lo_theta - 1),
          "ci_high": scale * (2 * hi_theta - 1),
          "n_discordant": k}


def _flip_rates(control, delta: float, disagreement: float) -> tuple:
  """The pair of per-row flip probabilities that inject a net shift of
  `delta` into `control` while keeping the simulated disagreement rate at
  `disagreement`.

  Writing `q` for the share of `control` rows that are 0, `up` for
  `P(treatment = 1 | control = 0)` and `down` for
  `P(treatment = 0 | control = 1)`, the two things being asked for are:

      net shift          q * up - (1 - q) * down = delta
      disagreement rate  q * up + (1 - q) * down = disagreement

  which solve to `q * up = (disagreement + delta) / 2` and
  `(1 - q) * down = (disagreement - delta) / 2`. Solving both at once is
  what makes `delta = 0` a genuine null -- equal expected movement in each
  direction, rather than the no-movement-at-all a monotone injector
  produces -- while still letting `delta` set the net effect the search is
  calibrating against.

  `disagreement` is raised to `|delta|` when it is smaller: a net shift can
  never exceed the total movement available, and at that boundary one
  direction's rate is legitimately 0 (the monotone case, correct here
  rather than assumed everywhere). Rates are clamped into [0, 1] for the
  degenerate cells -- every control row identical, so `q` is 0 or 1 and one
  of the two equations has no rows to act on.
  """
  n = len(control)
  if n == 0:
    return 0.0, 0.0
  q = sum(1 for c in control if c < 0.5) / n
  total = max(disagreement, abs(delta))
  up_mass = (total + delta) / 2
  down_mass = (total - delta) / 2
  up = up_mass / q if q > 0 else 0.0
  down = down_mass / (1 - q) if q < 1 else 0.0
  return min(1.0, max(0.0, up)), min(1.0, max(0.0, down))


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
  unit `cluster_bootstrap_ci_clustered` uses), injects the shift as a pair
  of per-row flip *probabilities* (`_flip_rates`) and draws a fresh
  Bernoulli outcome for `treatment*` from them.

  Two things about that injection, both load-bearing. It is a probability
  rather than a deterministic `treatment* = control* + delta`, which would
  move every pair by exactly `delta` with no exceptions and make the
  permutation test read "every diff shares one sign" as maximally extreme
  regardless of how small `delta` was, collapsing the search toward
  implausibly tiny deltas (caught via a smoke test before it shipped). And
  it is **two-sided**: pairs move both ways, at this row's own observed
  disagreement rate, with `delta` setting only the net. The earlier
  `clip(control* + delta, 0, 1)` was monotone -- a correct control row
  could never come back wrong at `delta >= 0`, a wrong one never come back
  right at `delta < 0` -- so it simulated a kind of effect real data never
  produces (one pooled cell here moves 10 pairs up and 21 down) and one
  much easier to detect than a two-signed mix of the same mean. Every MDE
  it reported was correspondingly too small, and every power estimate built
  on it too high. `delta=0` is a true null under either scheme, but only
  this one gives it realistic per-pair churn rather than no movement at
  all.

  Reruns `paired_permutation_test_clustered` on the injected data,
  clustering on the resampled draw index (two resampled copies of the same
  original cluster are two hypothetical instances, not one). `power(delta)`
  is the fraction of trials with `p <= alpha`.

  The rate clamp in `_flip_rates` is deliberate, not a limitation to work
  around: a near-ceiling or near-floor control (see
  `scripts/check_significance.py`'s `near_ceiling`) has few rows on the
  side a shift would have to move, so the required flip rate saturates at
  1.0 and even a large `delta` produces a small *realized* shift --
  correctly reflecting reduced detectability there rather than hiding it.
  Returns both `delta` (the swept parameter at convergence) and
  `realized_diff` (the replicates' actual mean `treatment* - control*` at
  that `delta`), since the two can differ by an order of magnitude at the
  ceiling and the gap between them is itself informative.

  **`delta` and `realized_diff` are not interchangeable, and a caller
  comparing an MDE against an observed accuracy difference wants
  `realized_diff`.** `delta` is the swept parameter; `realized_diff` is
  what it actually produced on this row's data, which is the same scale as
  `check_significance.py`'s `delta` column. Dividing one by the other
  inflates a sample-size extrapolation by `(1 / headroom)^2` -- 50-160x on
  a near-ceiling cell (see `scripts/recommend_count.py`).

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

  # The injector is binary end to end: `_flip_rates` splits `control` at 0.5
  # to get `q`, and `treatment*` comes back as a fresh Bernoulli 0/1 draw. Fed
  # a graded metric -- `connected_nodes` set-F1, whose control vector holds 52
  # distinct values averaging 0.97 -- it compares that 0/1 draw against
  # untouched floats, so `realized_diff` is pinned near `-mean(control)` for
  # every candidate `delta`. The search reads that as "detected" at each step
  # and bisects to its floor, reporting an MDE of `initial_hi / 2**n_steps`
  # with a large *negative* realized effect: a number that looks like a
  # precise answer and is pure artifact. Refuse rather than return it; a
  # caller that wants an MDE on a graded metric must binarize first and say
  # where it put the threshold.
  if any(v not in (0, 1) for v in control) or any(
      v not in (0, 1) for v in treatment):
    note = ("non-binary outcome: this MDE simulates Bernoulli draws and "
            "cannot describe a graded metric -- binarize before calling")
    return {
        "delta": None, "realized_diff": None, "power_target": power_target,
        "note": note,
        "delta_negative": None, "realized_diff_negative": None,
        "note_negative": note if direction != "positive" else None,
    }

  by_cluster: dict = {}
  for cid, c, t in zip(cluster_ids, control, treatment):
    by_cluster.setdefault(cid, []).append((c, t))
  clusters = list(by_cluster.values())
  rng = random.Random(seed)
  # How often this cell's real pairs disagree at all, in either direction --
  # the churn `_flip_rates` reproduces so a simulated effect is as hard to
  # detect as a real one of the same size. Measured from this row's own
  # data, not assumed: a near-ceiling cell that disagrees on 2 pairs in 180
  # and a cell that disagrees on 30 are very different detection problems.
  observed_disagreement = (
      sum(1 for c, t in zip(control, treatment) if c != t) / len(control)
  )

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
      #
      # The flip rates are two-sided, and that is the point. The obvious
      # `p = clip(c + delta)` is monotone in `c`: at `delta >= 0` a correct
      # control row gets `p = 1` and can never come back wrong, at
      # `delta < 0` a wrong one gets `p = 0` and can never come back right.
      # Every simulated pair then moves with `delta` or not at all, while
      # real pairs move both ways (one pooled cell here disagrees 10 up
      # against 21 down). One-signed differences are far easier for a
      # permutation test to detect than a two-signed mix with the same
      # mean, so power came out badly overstated and the MDE correspondingly
      # too small -- and the Bernoulli draw above, whose comment claims to
      # have made this believable, fixes only the magnitude problem, never
      # the sign one.
      #
      # `_flip_rates` solves for the pair of rates that give a *net* shift
      # of `delta` while disagreeing as often as this cell really does, so
      # `delta = 0` is a true null (equal expected movement each way, not
      # zero movement) and a nonzero `delta` is a net shift riding on
      # realistic two-way churn.
      up, down = _flip_rates(c_star, delta, observed_disagreement)
      p_star = [up if c < 0.5 else 1.0 - down for c in c_star]
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


def required_n_closed_form(delta: float) -> int:
  """Fast, formula-based anchor for how many paired instances a follow-up
  needs to detect an additive effect `delta`, two-sided alpha=0.05, 80%
  power.

  Derivation: a paired mean-difference test needs N ~= (z_(a/2) + z_beta)^2
  * Var(D) / delta^2 pairs, where D is the per-pair outcome difference.
  Approximating Var(D) ~= |delta| itself -- for a real, one-directional
  effect the paired difference is dominated by the fraction of pairs the
  effect actually flips, and `Var(D) = b + c - (c - b)^2` (b, c the two
  discordant-pair rates) collapses to ~|delta| when the effect is small and
  mostly one-directional -- turns the delta^2 in the denominator into a
  single delta: N ~= (z_(a/2) + z_beta)^2 / |delta| = 2.8^2 / |delta| =
  7.84/|delta|, using z_0.025=1.96 and z_0.20=0.84.

  This is a sanity anchor, not the trustworthy answer -- it assumes the
  Var(D)~=|delta| approximation and ignores this pilot's actual control
  base rate and cluster structure entirely.
  `required_sample_size_clustered` simulates directly from the pilot's own
  data instead and is expected to roughly, not exactly, agree with this.
  """
  if delta == 0:
    raise ValueError("delta must be nonzero")
  return math.ceil(7.84 / abs(delta))


def required_sample_size_clustered(
    control, treatment, cluster_ids, target_delta: float, alpha: float = 0.05,
    power_target: float = 0.8, n_replicates: int = 200, n_perm: int = 500,
    seed=0, max_multiplier: int = 50,
) -> dict:
  """Prospective inversion of `minimum_detectable_effect_clustered`: fixes
  `target_delta` (the effect size a follow-up run should be able to
  detect -- typically this pilot's own observed delta, floored at some
  minimum effect of interest) and searches the smallest cluster count `N`
  such that resampling `N` clusters (with replacement) from *this pilot's
  own* `(control, treatment)` pairs and injecting `target_delta` the same
  Bernoulli-probability way `minimum_detectable_effect_clustered` already
  does (never a deterministic shift -- see that function's own comment on
  why a fresh draw is what makes a smaller delta genuinely harder to
  detect) reaches `power_target` on `paired_permutation_test_clustered`.

  Answers "how many graphs would a follow-up need," the mirror image of
  `minimum_detectable_effect_clustered`'s "what could this many graphs
  have detected." Reuses the same resampling/injection machinery
  (`_resample_clusters`'s `m` parameter) rather than a closed-form formula,
  because the answer is bound to the pilot's own control base rate and
  cluster-size distribution the same way MDE is -- a near-ceiling pilot
  needs more clusters to reach the same power than a mid-range one at the
  same `target_delta`, and a fixed-variance formula would miss that.
  `required_n_closed_form` is the fast, formula-based anchor for when this
  simulation is too slow to run for every cell; the two are expected to
  roughly agree, not be identical.

  Search: doubles a candidate multiplier of the pilot's own
  `len(clusters)` (starting at 1x) until `power(multiplier) >= power_target`
  or the multiplier reaches `max_multiplier`, then bisects between the last
  failing and first succeeding multiplier -- same expand-then-bisect shape
  as `_search_one_direction`, swept over an integer cluster count instead
  of a continuous delta.

  Returns `{"required_n_clusters", "achieved_power", "pilot_n_clusters"}`;
  `required_n_clusters` is `None` if `power_target` isn't reached by
  `max_multiplier` times the pilot's own cluster count -- an honest "even a
  lot more data shaped like this wouldn't be enough" answer, not a
  silently wrong number.
  """
  if target_delta == 0:
    raise ValueError("target_delta must be nonzero")
  control, treatment = list(control), list(treatment)
  cluster_ids = list(cluster_ids)
  if not (len(control) == len(treatment) == len(cluster_ids)):
    raise ValueError(
        f"required sample size needs equal lengths, got {len(control)} "
        f"control, {len(treatment)} treatment, {len(cluster_ids)} cluster_ids"
    )
  if len(control) == 0:
    return {"required_n_clusters": None, "achieved_power": 0.0,
            "pilot_n_clusters": 0}

  by_cluster: dict = {}
  for cluster_id, c, t in zip(cluster_ids, control, treatment):
    by_cluster.setdefault(cluster_id, []).append((c, t))
  clusters = list(by_cluster.values())
  pilot_n_clusters = len(clusters)
  rng = random.Random(seed)
  sign = 1 if target_delta > 0 else -1
  delta = abs(target_delta)

  def _power_at(n_clusters: int) -> float:
    hits = 0
    for _ in range(n_replicates):
      pairs, draw_ids = _resample_clusters(clusters, rng, m=n_clusters)
      c_star = [c for c, _ in pairs]
      p_star = [min(1.0, c + sign * delta) if sign > 0
                else max(0.0, c + sign * delta) for c in c_star]
      t_star = [1.0 if rng.random() < p else 0.0 for p in p_star]
      result = paired_permutation_test_clustered(
          c_star, t_star, draw_ids, n_perm=n_perm, seed=rng.randrange(2**31)
      )
      if result["p_value"] <= alpha:
        hits += 1
    return hits / n_replicates

  multiplier = 1
  power = _power_at(pilot_n_clusters * multiplier)
  while power < power_target and multiplier < max_multiplier:
    multiplier *= 2
    power = _power_at(pilot_n_clusters * multiplier)
  if power < power_target:
    return {"required_n_clusters": None, "achieved_power": power,
            "pilot_n_clusters": pilot_n_clusters}

  lo_multiplier, hi_multiplier = multiplier // 2, multiplier
  while hi_multiplier - lo_multiplier > 1:
    mid = (lo_multiplier + hi_multiplier) // 2
    mid_power = _power_at(pilot_n_clusters * mid)
    if mid_power >= power_target:
      hi_multiplier, power = mid, mid_power
    else:
      lo_multiplier = mid

  return {
      "required_n_clusters": pilot_n_clusters * hi_multiplier,
      "achieved_power": power,
      "pilot_n_clusters": pilot_n_clusters,
  }


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


def unpaired_permutation_test(
    group_a, group_b, n_perm: int = 10_000, seed: int = 0
) -> dict:
  """Two-sample permutation test on the difference of means, for comparing
  two *unpaired* samples -- e.g. a structural feature's distribution across
  two different sets of graph instances (old vs. new), where there is no
  instance-to-instance pairing the way there is between a `condition` and
  `CONTROL` row for the same graph.

  Every other test in this module is paired (the same instance appears on
  both sides), which is why they sign-flip a per-pair difference under the
  null of exchangeability. Here the null is that `group_a` and `group_b` are
  drawn from the same distribution, so the null is built by repeatedly
  reshuffling the *pooled* values into two groups of the original sizes and
  recomputing the mean difference -- the standard permutation test for this
  design. Works identically for a continuous feature (e.g. density) or a
  0/1 indicator (e.g. `has_isolated_node`), where the difference of means is
  a difference of proportions.

  Uses the same add-one correction (North et al. 2002) as
  `paired_permutation_test`, so a Monte Carlo p-value is never reported as
  exactly 0.
  """
  group_a, group_b = list(group_a), list(group_b)
  n_a, n_b = len(group_a), len(group_b)
  if n_a == 0 or n_b == 0:
    return {"n_a": n_a, "n_b": n_b, "observed_diff": 0.0, "p_value": 1.0}
  observed = sum(group_b) / n_b - sum(group_a) / n_a
  observed_abs = abs(observed)
  pooled = group_a + group_b
  rng = random.Random(seed)
  at_least_as_extreme = 0
  for _ in range(n_perm):
    rng.shuffle(pooled)
    shuffled_b = pooled[n_a:]
    shuffled_a = pooled[:n_a]
    permuted = sum(shuffled_b) / n_b - sum(shuffled_a) / n_a
    if abs(permuted) >= observed_abs - 1e-12:
      at_least_as_extreme += 1
  p_value = (at_least_as_extreme + 1) / (n_perm + 1)
  return {"n_a": n_a, "n_b": n_b, "observed_diff": observed, "p_value": p_value}


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
