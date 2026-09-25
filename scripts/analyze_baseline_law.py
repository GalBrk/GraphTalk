"""Shared helpers for the analyses of the 40-node and density follow-up runs:
the graph-blind solver's route split (`route_gain`, `offers_route`), instance-id
parsing (`density_of`, `is_replication_seed`), and a scipy-free Pearson r.

The tests this script used to run are in superseded/scripts/analyze_baseline_law.py.
The density continuum it last ran is printed by scripts/score_density_sweep.py
(`continuum`) under rule R1, through scripts/density_followups.py.
"""

import math

# Tasks whose gold answer is effectively constant at n=40 -- `node_count` is
# always 40 because n is fixed, and `cycle_check` is "yes" for every graph
# past the sparsest level. A blind solver scores ~1.00 on them under EVERY
# condition, so a shortcut bar fitted on the published split's small graphs
# reports content the primer does not actually add here. Both are excluded
# from the split rather than mis-classified by it.
DEGENERATE_TASKS = ("node_count", "cycle_check")

# A primer is treated as offering a substitute route when the blind solver
# recovers meaningfully more from it than from `none`. The threshold is a
# rounding guard, not a tuned parameter: every gain in `shortcuts.json` is
# either <= 0.012 or >= 0.114, so anything in [0.02, 0.10] gives this split.
ROUTE_GAIN_THRESHOLD = 0.05

DENSFULL_ARMS = ("qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think")


def route_gain(bars: dict, task: str, condition: str) -> float:
  """How much more of the answer the blind solver recovers from `condition`
  than from `none`, on `task`. Degenerate-answer tasks return 0.0: their
  bars were fitted on the published split's small graphs and overstate the
  added content at n=40 (see DEGENERATE_TASKS).
  """
  if task in DEGENERATE_TASKS:
    return 0.0
  return bars[f"{task}/{condition}"] - bars[f"{task}/none"]


def offers_route(bars: dict, task: str, condition: str) -> bool:
  return route_gain(bars, task, condition) > ROUTE_GAIN_THRESHOLD


def density_of(instance_id: str) -> float | None:
  """Parse the pinned density out of `<task>/size40/p<density>/<i>`."""
  for part in instance_id.split("/"):
    if len(part) > 1 and part[0] == "p":
      try:
        return float(part[1:])
      except ValueError:
        continue
  return None


def is_replication_seed(instance_id: str) -> bool:
  """True for the seed-offset replication corpus (`/s500000/`), which must
  never be pooled with the default-seed graphs it was built to be independent
  of. `size40` must not match, which is why the digits are required.
  """
  for part in instance_id.split("/"):
    if len(part) > 1 and part[0] == "s" and part[1:].isdigit():
      return True
  return False


def pearson(xs, ys) -> tuple[float, float]:
  """Pearson r and a two-sided p-value, via the exact t transform.

  Hand-rolled rather than imported: this script is run on machines where the
  `analysis` extra (scipy) is not installed, and r on a few hundred points
  does not need it.
  """
  n = len(xs)
  if n < 3:
    return float("nan"), float("nan")
  mx, my = sum(xs) / n, sum(ys) / n
  sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
  sxx = sum((x - mx) ** 2 for x in xs)
  syy = sum((y - my) ** 2 for y in ys)
  if sxx <= 0 or syy <= 0:
    return float("nan"), float("nan")
  r = sxy / math.sqrt(sxx * syy)
  r = max(-0.999999999, min(0.999999999, r))
  t = r * math.sqrt((n - 2) / (1 - r * r))
  return r, _t_sf(abs(t), n - 2) * 2


def _t_sf(t: float, dof: int) -> float:
  """Upper tail of Student's t, via the regularized incomplete beta."""
  x = dof / (dof + t * t)
  return 0.5 * _betainc(dof / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
  """Regularized incomplete beta I_x(a, b), continued fraction (Lentz)."""
  if x <= 0:
    return 0.0
  if x >= 1:
    return 1.0
  lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
  front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
  if x > (a + 1) / (a + b + 2):
    return 1.0 - _betainc(b, a, 1 - x)
  f, c, d = 1.0, 1.0, 0.0
  for i in range(0, 300):
    m = i // 2
    if i == 0:
      num = 1.0
    elif i % 2 == 0:
      num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
    else:
      num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
    d = 1.0 + num * d
    d = 1e-30 if abs(d) < 1e-30 else d
    d = 1.0 / d
    c = 1.0 + num / c
    c = 1e-30 if abs(c) < 1e-30 else c
    f *= c * d
    if abs(1.0 - c * d) < 1e-12:
      break
  return front * (f - 1.0)
