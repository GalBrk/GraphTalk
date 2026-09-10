"""Classification rules for the size/density grid reporter
(`scripts/score_density_size_grid.py`), applied uniformly to every task and
condition -- nothing here ever drops a row; it only labels one.

Two separate thresholds live here on purpose, not one function with
different arguments: `scout_decision` (Stage 2's relaxed screen -- the only
place a cell can be excluded from further generation, and even then only
archived, not deleted) and `zone_decision`/`classify_cell` (Stage 3's
descriptive classification, at the stricter thresholds
`docs/primer-impact-and-truncation.md` validated by hand). Conflating the two
into one parameterized function would make it easy to accidentally screen
with the strict bar or report with the relaxed one; keeping them as
different functions makes that a type error, not a bug waiting to happen.

No I/O, no task list, no filtering: every function here takes numbers in and
returns a descriptive dict out. `node_count` needs no special case anywhere
in this module -- pinning `n` per cell (as the grid does) drives its
none-accuracy to ~1.0, which trips the ordinary `ceiling` branch of
`zone_decision` on its own.
"""

import math

# This project deliberately keeps scipy out of its dependency set (see
# graphtalk/significance.py's module docstring), so the Wilson-interval
# z-score is a hand-rolled table rather than a stats-library lookup -- only
# 95% is needed anywhere this module is used.
_Z_BY_CONFIDENCE = {0.95: 1.959963985}


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple:
  """Wilson score interval for a binomial proportion.

  More accurate than the normal approximation near 0 or 1, which is exactly
  the regime every threshold in this module lives in (0.98/0.90 ceilings,
  10%/30% truncation bars) -- the normal approximation can extend past 0 or
  1 there and understates how wide the interval really is.
  """
  if n == 0:
    return (0.0, 1.0)
  if confidence not in _Z_BY_CONFIDENCE:
    raise ValueError(
        f"unsupported confidence level: {confidence!r}; known: "
        f"{sorted(_Z_BY_CONFIDENCE)}"
    )
  z = _Z_BY_CONFIDENCE[confidence]
  phat = successes / n
  denom = 1 + z * z / n
  center = phat + z * z / (2 * n)
  margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
  lo = (center - margin) / denom
  hi = (center + margin) / denom
  return (max(0.0, lo), min(1.0, hi))


def scout_decision(
    none_accuracy: float, majority_baseline: float, truncation_rate: float,
    ceiling: float = 0.98, floor_margin: float = 0.02,
    truncation_drop: float = 0.30,
) -> dict:
  """Stage 2's relaxed screen: drop only the absolute losers.

  Deliberately loosened from Stage 3's `zone_decision` thresholds (0.98 vs
  0.90 ceiling, 30% vs 10% truncation) -- a cell between 90-98% none-
  accuracy, or truncating between 10-30% on the cheap plain/`none` scout,
  survives to the full reporter, where both arms and all 7 conditions get
  a real look, instead of being cut on a coarse, single-arm, single-
  condition read. The floor rule is NOT loosened: a cell with zero room
  above the majority guess isn't rescued by relaxing this one, since no
  amount of primer or arm choice creates headroom that isn't there.

  Returns `{"decision": "survive"|"drop", "zone": str|None, "reason": str}`
  -- `zone` is `None` when the drop reason is truncation rather than a
  none-accuracy zone, since the two are independent checks.
  """
  if none_accuracy > ceiling:
    return {
        "decision": "drop", "zone": "ceiling",
        "reason": f"none accuracy {none_accuracy:.1%} > {ceiling:.0%} ceiling",
    }
  if none_accuracy <= majority_baseline + floor_margin:
    return {
        "decision": "drop", "zone": "floor",
        "reason": (
            f"none accuracy {none_accuracy:.1%} <= majority baseline "
            f"{majority_baseline:.1%} + {floor_margin:.0%}"
        ),
    }
  if truncation_rate > truncation_drop:
    return {
        "decision": "drop", "zone": None,
        "reason": f"truncation {truncation_rate:.1%} > {truncation_drop:.0%}",
    }
  return {"decision": "survive", "zone": "informative", "reason": ""}


def zone_decision(
    none_accuracy: float, majority_baseline: float, ceiling: float = 0.90,
    floor_margin: float = 0.02,
) -> str:
  """Stage 3's descriptive none-baseline zone -- `"ceiling"`, `"floor"`, or
  `"informative"`. Never used to drop a cell, only to label it; the
  thresholds are the ones `docs/primer-impact-and-truncation.md` already
  validated by hand (deliberately not `scripts/check_significance.py`'s
  0.95 near-ceiling threshold, which was tuned for different, already-
  collected data).
  """
  if none_accuracy >= ceiling:
    return "ceiling"
  if none_accuracy <= majority_baseline + floor_margin:
    return "floor"
  return "informative"


def shortcut_clean(shortcut_score: float | None, bar: float = 0.10) -> bool:
  """Whether a condition has enough headroom above the primer-only
  shortcut solver's score (rung 3, from `shortcuts.json`) to be usable
  evidence of reasoning. `None` (no shortcut score recorded for this
  (task, condition) cell) is never clean -- absence of a bar is not the
  same as clearing it.
  """
  return shortcut_score is not None and (1.0 - shortcut_score) >= bar


def truncation_clean(truncation_rate: float | None, threshold: float = 0.10) -> bool:
  """Whether a cell's combined `hit_cap`/`overflow` rate is low enough to
  trust. `None` (no truncation rate available) is never clean.
  """
  return truncation_rate is not None and truncation_rate < threshold


def classify_cell(
    none_accuracy: float, majority_baseline: float,
    shortcut_score: float | None, truncation_rate: float | None,
    ceiling: float = 0.90, floor_margin: float = 0.02,
    shortcut_bar: float = 0.10, truncation_threshold: float = 0.10,
) -> dict:
  """The full Stage-3 descriptive classification for one
  (task, condition, n, density) cell on one arm -- always a dict of labels,
  never a boolean "keep this or not."

  Returns `{"zone", "shortcut_clean", "truncation_clean", "passes",
  "reason"}`. `passes` is `True` only when the zone is `"informative"` and
  both cleanliness checks hold -- reported as a column, never used by this
  module or its callers to omit the row from any output.
  """
  zone = zone_decision(none_accuracy, majority_baseline, ceiling, floor_margin)
  sc_clean = shortcut_clean(shortcut_score, shortcut_bar)
  trunc_clean = truncation_clean(truncation_rate, truncation_threshold)
  passes = zone == "informative" and sc_clean and trunc_clean

  reasons = []
  if zone != "informative":
    reasons.append(f"zone={zone}")
  if not sc_clean:
    reasons.append(
        f"shortcut headroom {(1.0 - shortcut_score):.1%}"
        if shortcut_score is not None else "no shortcut score"
    )
  if not trunc_clean:
    reasons.append(
        f"truncation {truncation_rate:.1%}"
        if truncation_rate is not None else "no truncation rate"
    )
  return {
      "zone": zone,
      "shortcut_clean": sc_clean,
      "truncation_clean": trunc_clean,
      "passes": passes,
      "reason": "; ".join(reasons) if reasons else "clean",
  }


def arm_divergence(plain: dict, think: dict) -> dict:
  """Compares two `classify_cell` results (one per arm) and flags whenever
  they disagree -- on the combined `passes` verdict, or on any individual
  check, even when `passes` happens to coincide. The two arms are never
  collapsed into one verdict upstream of this function specifically so
  "one arm raises a risk the other doesn't" stays visible in the report
  instead of being averaged away.

  Returns `{"diverges": bool, "reason": str}`.
  """
  reasons = []
  if plain["passes"] != think["passes"]:
    reasons.append(f"passes: plain={plain['passes']} vs think={think['passes']}")
  if plain["zone"] != think["zone"]:
    reasons.append(f"zone: plain={plain['zone']} vs think={think['zone']}")
  if plain["shortcut_clean"] != think["shortcut_clean"]:
    reasons.append(
        f"shortcut_clean: plain={plain['shortcut_clean']} vs "
        f"think={think['shortcut_clean']}"
    )
  if plain["truncation_clean"] != think["truncation_clean"]:
    reasons.append(
        f"truncation_clean: plain={plain['truncation_clean']} vs "
        f"think={think['truncation_clean']}"
    )
  return {"diverges": bool(reasons), "reason": "; ".join(reasons)}
