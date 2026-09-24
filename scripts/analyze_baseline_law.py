"""The density continuum: one arm, one task, one primer, seven pinned densities
(`--test continuum`), so difficulty is manipulated rather than observed.

The route-split, held-out, instrument, cross-fit and ceiling tests that this
script also ran are in superseded/scripts/analyze_baseline_law.py; they back
claims the results no longer make. The helpers here are imported by other
live scripts.

    PYTHONPATH=. python scripts/analyze_baseline_law.py --shortcuts shortcuts.json

Scoring rules match `scripts/score_full_density_sweep.py`: `hit_cap` rows are
dropped rather than scored zero, dropped on both sides of a pair, and cells
are keyed by (arm, task, density, condition).
"""

import argparse
import collections
import glob
import hashlib
import json
import math
import os

from graphtalk import scoring

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

# Arms held out of the paper's development entirely. The first eight answer
# the published GraphQA split; the rest answer the 100-graph none/degree probe.
# Each of the first eight also has a `.rerun` file (filler + edge_existence,
# rerun after those conditions were reworded) that a single-pattern glob
# never matches, since it inserts a second `.rerun` segment before `.shard`
# or before `.jsonl` -- both patterns are listed so those two conditions are
# no longer silently missing from these arms.
HELDOUT_GLOBS = {
    "qwen3-8b": ["runs/qwen3-8b.jsonl", "runs/qwen3-8b.rerun.jsonl"],
    "qwen3-8b-think": ["runs/qwen3-8b-think.shard*.jsonl",
                       "runs/qwen3-8b-think.rerun.shard*.jsonl"],
    "qwen3-14b": ["runs/qwen3-14b.jsonl", "runs/qwen3-14b.rerun.jsonl"],
    "qwen3-14b-think": ["runs/qwen3-14b-think.shard*.jsonl",
                        "runs/qwen3-14b-think.rerun.shard*.jsonl"],
    "gemma4-e4b": ["runs/gemma4-e4b.jsonl", "runs/gemma4-e4b.rerun.jsonl"],
    "gemma4-e4b-think": ["runs/gemma4-e4b-think.shard*.jsonl",
                         "runs/gemma4-e4b-think.rerun.shard*.jsonl"],
    "gemma4-12b": ["runs/gemma4-12b.jsonl", "runs/gemma4-12b.rerun.jsonl"],
    "gemma4-12b-think": ["runs/gemma4-12b-think.shard*.jsonl",
                         "runs/gemma4-12b-think.rerun.shard*.jsonl"],
    "qwen3-0.6b": ["runs/qwen3-0.6b.probe100*.jsonl"],
    "qwen3-0.6b-think": ["runs/qwen3-0.6b-think.probe100*.jsonl"],
    "qwen35-2b": ["runs/qwen35-2b.probe100*.jsonl"],
}

PARAMS_B = {"qwen3-0.6b": 0.6, "qwen3-1.7b": 1.7, "qwen35-2b": 2.0,
            "qwen3-4b": 4.0, "gemma4-e4b": 4.0, "qwen3-8b": 8.0,
            "gemma4-12b": 12.0, "qwen3-14b": 14.0}

# The `node_degree` density continuum: one task, one primer at a time, seven
# pinned densities, up to 400 paired graphs each. The `.degdensrep.` seed-
# offset replication is excluded -- its instance ids carry an /s<seed>/
# segment precisely so it can never be pooled with the default-seed corpus.
CONTINUUM_GLOBS = {
    "qwen3-1.7b": ["runs/qwen3-1.7b.degdens40.shard*of5.jsonl",
                   "runs/qwen3-1.7b.degdens40hi.shard*of5.jsonl",
                   "runs/qwen3-1.7b.degdensfill.shard*of5.jsonl"],
    "qwen3-1.7b-think": ["runs/qwen3-1.7b-think.degdensthink.shard*of7.jsonl",
                         "runs/qwen3-1.7b-think.degdensfillT.shard*of7.jsonl"],
}


# --------------------------------------------------------------------------
# pure helpers (unit-tested in tests/test_analyze_baseline_law.py)
# --------------------------------------------------------------------------

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


def fit_line(xs, ys) -> tuple[float, float]:
  """Least-squares slope and intercept."""
  n = len(xs)
  mx, my = sum(xs) / n, sum(ys) / n
  sxx = sum((x - mx) ** 2 for x in xs)
  if sxx <= 0:
    return float("nan"), float("nan")
  slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
  return slope, my - slope * mx


def ols(rows, names) -> list[tuple[str, float, float, float]]:
  """Multiple regression with classical SEs. `rows` is a list of
  (x_vector_without_intercept, y). Returns (name, coef, se, t) per term,
  intercept first.
  """
  xs = [[1.0] + list(x) for x, _ in rows]
  ys = [y for _, y in rows]
  k = len(xs[0])
  xtx = [[sum(r[i] * r[j] for r in xs) for j in range(k)] for i in range(k)]
  xty = [sum(r[i] * y for r, y in zip(xs, ys)) for i in range(k)]
  inv = _invert(xtx)
  beta = [sum(inv[i][j] * xty[j] for j in range(k)) for i in range(k)]
  resid = [y - sum(b * v for b, v in zip(beta, r)) for r, y in zip(xs, ys)]
  dof = len(ys) - k
  s2 = sum(e * e for e in resid) / dof
  out = []
  for i, name in enumerate(["intercept"] + list(names)):
    se = math.sqrt(inv[i][i] * s2)
    out.append((name, beta[i], se, beta[i] / se if se else float("nan")))
  return out


def _invert(m):
  """Gauss-Jordan inverse of a small square matrix."""
  n = len(m)
  a = [list(row) + [1.0 if i == j else 0.0 for j in range(n)]
       for i, row in enumerate(m)]
  for col in range(n):
    piv = max(range(col, n), key=lambda r: abs(a[r][col]))
    if abs(a[piv][col]) < 1e-12:
      raise ValueError("singular design matrix")
    a[col], a[piv] = a[piv], a[col]
    p = a[col][col]
    a[col] = [v / p for v in a[col]]
    for r in range(n):
      if r == col:
        continue
      f = a[r][col]
      if f:
        a[r] = [v - f * w for v, w in zip(a[r], a[col])]
  return [row[n:] for row in a]


def _fold(instance_id: str) -> int:
  """A deterministic 2-way split on instance id, stable across processes.

  Python's builtin `hash()` on a str is randomized per-process (PYTHONHASHSEED)
  unless disabled, which would make the cross-fit split a coin flip on every
  invocation rather than a fixed, reproducible partition. `hashlib` is not.
  """
  return hashlib.md5(instance_id.encode()).digest()[0] % 2


def cells_from_scores_crossfit(scores, control="none", min_pairs=10):
  """Like `cells_from_scores`, but the baseline and the delta are estimated
  from disjoint halves of the paired graphs.

  `cells_from_scores` computes both `baseline` (mean of `b`) and `delta`
  (mean of `v - b`) from the SAME pairs, so a pair with an unusually low `b`
  pulls the cell toward a low baseline AND toward a high delta by
  construction -- regression to the mean, not signal. Splitting the paired
  instance ids into two folds and taking baseline from one, delta from the
  other, breaks that shared sampling noise. Both fold assignments are
  reported (baseline from fold A + delta from fold B, and the swap), which
  doubles k per cell rather than halving it, at the cost of each half being
  noisier -- the paper reports both this and the naive version side by side
  rather than picking one.
  """
  grouped = collections.defaultdict(dict)
  for (task, dens, iid, cond), value in scores.items():
    grouped[(task, dens, cond)][iid] = value
  out = []
  for (task, dens, cond), values in sorted(grouped.items(), key=repr):
    if cond == control:
      continue
    base = grouped.get((task, dens, control), {})
    pairs = [(i, base[i], v) for i, v in values.items()
             if base.get(i) is not None and v is not None]
    if len(pairs) < min_pairs:
      continue
    fold_a = [(b, v) for i, b, v in pairs if _fold(i) == 0]
    fold_b = [(b, v) for i, b, v in pairs if _fold(i) == 1]
    if len(fold_a) < min_pairs // 2 or len(fold_b) < min_pairs // 2:
      continue
    for base_fold, delta_fold, tag in ((fold_a, fold_b, "a>b"),
                                        (fold_b, fold_a, "b>a")):
      out.append(dict(
          task=task, density=dens, condition=cond,
          n=len(base_fold) + len(delta_fold), n_baseline_fold=len(base_fold),
          n_delta_fold=len(delta_fold), fold=tag,
          baseline=sum(b for b, _ in base_fold) / len(base_fold),
          delta=100.0 * sum(v - b for b, v in delta_fold) / len(delta_fold)))
  return out


def cells_from_scores(scores, control="none", min_pairs=10):
  """Turn {(task, density, instance_id, condition): score|None} into one
  record per (task, density, condition != control), paired on instance_id
  with `hit_cap` (None) rows dropped from BOTH sides.
  """
  grouped = collections.defaultdict(dict)
  for (task, dens, iid, cond), value in scores.items():
    grouped[(task, dens, cond)][iid] = value
  out = []
  for (task, dens, cond), values in sorted(grouped.items(), key=repr):
    if cond == control:
      continue
    base = grouped.get((task, dens, control), {})
    pairs = [(base[i], v) for i, v in values.items()
             if base.get(i) is not None and v is not None]
    if len(pairs) < min_pairs:
      continue
    out.append(dict(
        task=task, density=dens, condition=cond, n=len(pairs),
        baseline=sum(b for b, _ in pairs) / len(pairs),
        delta=100.0 * sum(v - b for b, v in pairs) / len(pairs)))
  return out


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def score_run(patterns, by_density=True, task_filter=None,
              skip_replication_seed=False):
  """-> {(task, density, instance_id, condition): score | None}

  None marks a `hit_cap` row, so callers can drop it from both sides of its
  pair rather than scoring it zero.
  """
  out, seen = {}, set()
  for pattern in patterns:
    for path in sorted(glob.glob(pattern)):
      if ".got." in os.path.basename(path):
        continue  # a different node-naming scheme; never pool it
      with open(path, encoding="utf-8") as handle:
        for line in handle:
          if not line.strip():
            continue
          row = json.loads(line)
          task = row["task"]
          if task_filter and task != task_filter:
            continue
          if skip_replication_seed and is_replication_seed(row["instance_id"]):
            continue
          key = (task, density_of(row["instance_id"]) if by_density else None,
                 row["instance_id"], row["condition"])
          if key in seen:
            continue  # duplicate rows survive a preemption/resume boundary
          seen.add(key)
          out[key] = None if row.get("hit_cap") else scoring.score_one(
              scoring.extract_answer(row["response"], task),
              row["gold"], task)["primary"]
  return out


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------

def test_continuum(args, bars):
  print("\nTEST 2  difficulty manipulated, not observed"
        " (node_degree density continuum)")
  for arm, patterns in CONTINUUM_GLOBS.items():
    scores = score_run([f"{args.runs}/{os.path.basename(p)}"
                        for p in patterns],
                       task_filter="node_degree", skip_replication_seed=True)
    cells = cells_from_scores(scores)
    if not cells:
      print(f"  {arm}: no runs found; skipped")
      continue
    conds = sorted({c["condition"] for c in cells})
    levels = sorted({c["density"] for c in cells})
    # The printed `none` column is the plain accuracy over every scored
    # `none` row at that level, for context. The correlations below use each
    # condition's own paired baseline, which is what its delta is against.
    unpaired = collections.defaultdict(list)
    for (task, dens, _, cond), value in scores.items():
      if cond == "none" and value is not None:
        unpaired[dens].append(value)
    print(f"\n  {arm}")
    print("    " + f"{'p':>6}{'none':>8}" + "".join(f"{c:>12}" for c in conds))
    series = collections.defaultdict(list)
    for level in levels:
      seen_here = unpaired.get(level, [])
      row = (f"    {level:>6.2f}"
             f"{(sum(seen_here) / len(seen_here) if seen_here else float('nan')):>8.3f}")
      for cond in conds:
        hit = [c for c in cells
               if c["density"] == level and c["condition"] == cond]
        if not hit:
          row += f"{'-':>12}"
          continue
        cell = hit[0]
        row += f"{cell['delta']:>+12.1f}"
        series[cond].append((cell["baseline"], cell["delta"]))
      print(row)
    print("    r(delta, baseline) across levels:")
    for cond in conds:
      pts = series[cond]
      if len(pts) >= 4:
        r, p = pearson([b for b, _ in pts], [d for _, d in pts])
        print(f"      {cond:<12} r={r:+.3f}  p={p:.3g}"
              f"   baseline {min(b for b, _ in pts):.2f}"
              f"-{max(b for b, _ in pts):.2f}")


TESTS = {"continuum": test_continuum}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--shortcuts", default="shortcuts.json",
                      help="blind-solver table from scripts/shortcut_table.py")
  parser.add_argument("--runs", default="runs", help="directory of run files")
  parser.add_argument("--test", action="append", choices=sorted(TESTS),
                      help="run only these (default: all four)")
  args = parser.parse_args()

  with open(args.shortcuts, encoding="utf-8") as handle:
    bars = json.load(handle)
  for name in args.test or ["continuum"]:
    TESTS[name](args, bars)


if __name__ == "__main__":
  main()
