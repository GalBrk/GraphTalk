"""Does a primer effect track edge density, or the headroom the density leaves?

The paper organises its results by density: the `clustering` effect "peaks at
intermediate density", the length cost is read off a density sweep, and
`densfull40hi` exists to extend the density range. But density is not a
treatment the model sees. It changes the answer distribution, which changes
how well the model already does, which is what decides whether any primer can
move accuracy at all. The two are collinear across the sweep, so "the effect
peaks at p=.35-.50" and "the effect peaks where the model is mid-range" are
the same sentence until they are separated.

Two tests, both on `csv2/sweep-large-graph/primer_decomposition.csv`:

  --test partial    Partial correlations of effect magnitude with headroom
                    (controlling density) and with density (controlling
                    headroom), over every non-degenerate cell. Headroom is
                    `min(acc, 1-acc)` on the `none` column -- read off the
                    control alone, never the treatment, so it cannot select
                    for an effect.

  --test orthogonal The clean identification. On `edge_existence` the
                    majority-class prior is U-shaped in density (.90 .82 .69
                    .52 .66 .76 .85) while density is monotone, so the two are
                    near-orthogonal *within that task* and a simple
                    correlation with each already answers the question.

Effect magnitude is |content gain| (condition - `filler`), the length-controlled
term, so a length cost that itself tracks density cannot masquerade as a
content effect. `--metric net` uses the uncorrected term instead.

CIs resample whole (arm, task, density) blocks with replacement via
`significance._resample_clusters`. That is the graph-sharing unit -- the three
conditions in one such block are answers about the same 100 graphs -- so a
per-cell bootstrap would understate the variance. Blocking any coarser (on
(arm, task), say) leaves two clusters for the `edge_existence` test, which is
too few for a percentile interval to mean anything.

  PYTHONPATH=. python scripts/analyze_effect_drivers.py
"""
import argparse
import collections
import csv
import math
import random

from graphtalk import significance

DEFAULT_CSV = "csv2/sweep-large-graph/primer_decomposition.csv"

# Constant-gold at n=40: a majority baseline of 1.000 leaves no headroom to
# measure and every condition "helps" by breaking an off-by-one.
DEGENERATE = {"node_count", "cycle_check"}

# Conditions whose primer neither states nor trivially implies any answer.
UNCONTAMINATED = {"components", "clustering", "rwse"}

PLAIN = ["qwen3-1.7b", "qwen3-4b"]


def pearson(xs, ys):
  n = len(xs)
  if n < 3:
    return float("nan")
  mx, my = sum(xs) / n, sum(ys) / n
  num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
  den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
  return num / den if den else float("nan")


def partial(xs, ys, zs):
  """r(x, y) with z held fixed."""
  rxy, rxz, ryz = pearson(xs, ys), pearson(xs, zs), pearson(ys, zs)
  den = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
  return (rxy - rxz * ryz) / den if den else float("nan")


def load(path, arms, uncontaminated_only=True, metric="content"):
  """One row per cell: (block, effect magnitude, headroom, density)."""
  field = "content_gain_pp" if metric == "content" else "net_pp"
  rows = []
  for r in csv.DictReader(open(path, encoding="utf-8")):
    if r["arm"] not in arms or r["task"] in DEGENERATE:
      continue
    if uncontaminated_only and r["condition"] not in UNCONTAMINATED:
      continue
    acc = float(r["none_acc"])
    rows.append({
        "block": (r["arm"], r["task"], r["density"]),
        "task": r["task"], "arm": r["arm"], "condition": r["condition"],
        "y": abs(float(r[field])),
        "headroom": min(acc, 1.0 - acc),
        "acc": acc,
        "density": float(r["density"]),
    })
  return rows


def block_ci(rows, stat, n_boot=10_000, seed=0, alpha=0.05):
  """Percentile CI, resampling whole (arm, task, density) blocks."""
  blocks = collections.defaultdict(list)
  for r in rows:
    blocks[r["block"]].append(r)
  clusters = list(blocks.values())
  rng = random.Random(seed)
  draws = []
  for _ in range(n_boot):
    items, _ = significance._resample_clusters(clusters, rng)
    value = stat(items)
    if not math.isnan(value):
      draws.append(value)
  draws.sort()
  if not draws:
    return (float("nan"), float("nan"))
  lo = draws[int(alpha / 2 * len(draws))]
  hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
  return (lo, hi)


def test_partial(rows, n_boot):
  print(f"cells={len(rows)}  blocks={len({r['block'] for r in rows})}")
  print(f"  collinearity r(headroom, density) = "
        f"{pearson([r['headroom'] for r in rows], [r['density'] for r in rows]):+.3f}")

  def r_head(items):
    return partial([r["y"] for r in items], [r["headroom"] for r in items],
                   [r["density"] for r in items])

  def r_dens(items):
    return partial([r["y"] for r in items], [r["density"] for r in items],
                   [r["headroom"] for r in items])

  for label, fn in [("headroom | density", r_head), ("density | headroom", r_dens)]:
    point = fn(rows)
    lo, hi = block_ci(rows, fn, n_boot=n_boot)
    star = " *" if (lo > 0 or hi < 0) else ""
    print(f"  partial r({label:<18}) = {point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]{star}")


def test_orthogonal(rows, n_boot):
  """edge_existence only: prior is U-shaped in density, density is monotone."""
  ee = [r for r in rows if r["task"] == "edge_existence"]
  if not ee:
    print("  no edge_existence cells")
    return
  print(f"edge_existence only: cells={len(ee)}  "
        f"blocks={len({r['block'] for r in ee})}")
  print(f"  r(headroom, density) within task = "
        f"{pearson([r['headroom'] for r in ee], [r['density'] for r in ee]):+.3f}"
        f"   (near zero => the two are separable here)")
  for label, key in [("headroom", "headroom"), ("density", "density")]:
    fn = (lambda k: lambda items: pearson([r["y"] for r in items],
                                          [r[k] for r in items]))(key)
    point = fn(ee)
    lo, hi = block_ci(ee, fn, n_boot=n_boot)
    star = " *" if (lo > 0 or hi < 0) else ""
    print(f"  r(|effect|, {label:<8}) = {point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]{star}")


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--csv", default=DEFAULT_CSV)
  ap.add_argument("--test", choices=["partial", "orthogonal", "all"], default="all")
  ap.add_argument("--metric", choices=["content", "net"], default="content")
  ap.add_argument("--all-conditions", action="store_true",
                  help="include the answer-stating primers as well")
  ap.add_argument("--arms", nargs="+", default=PLAIN)
  ap.add_argument("--n-boot", type=int, default=10_000)
  args = ap.parse_args()

  rows = load(args.csv, args.arms, not args.all_conditions, args.metric)
  print(f"{args.csv}  arms={','.join(args.arms)}  metric={args.metric}  "
        f"conditions={'all' if args.all_conditions else 'uncontaminated'}\n")
  if args.test in ("partial", "all"):
    print("== effect magnitude vs headroom and density, partialled ==")
    test_partial(rows, args.n_boot)
    print()
  if args.test in ("orthogonal", "all"):
    print("== clean identification on edge_existence ==")
    test_orthogonal(rows, args.n_boot)


if __name__ == "__main__":
  main()
