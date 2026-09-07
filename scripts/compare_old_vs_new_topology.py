"""Was the original 30-graph draw a compositionally atypical sample of the
fixed 500-graph published population -- and how do the two populations'
structural distributions actually differ?

Complements `scripts/check_old_vs_new_subsample.py` (which shows the old-30
and new-470 slices give overlapping-CI, similar-magnitude accuracy deltas for
qwen3-8b/degree -- consistent with pure power, not a different effect). This
script answers the structural half of the same investigation directly: does
the topology of the first 30 published graphs differ from the topology of the
other 470, or of the full 500, in any feature `scripts/extract_graph_topology.py`
computed?

Every published-split graph shares the same index across every task config
(verified by `extract_graph_topology.py --verify-config`), so `--split-at`
here means the same thing it does in `check_old_vs_new_subsample.py`: indices
below it are "original" (identical to the tracked --count 30 sweep), at or
above it are "new".

Hand-rolled `graphtalk.significance.unpaired_permutation_test` rather than
scipy's Mann-Whitney/KS, for the same reason the rest of this project's
significance code stays scipy-free (see `graphtalk/significance.py`'s module
docstring). `graphtalk.significance.benjamini_hochberg` corrects across the
whole feature list, consistent with the project's existing FDR convention
(`scripts/check_significance.py`) rather than introducing a second,
uncorrected multiple-testing surface.

    PYTHONPATH=. .venv/Scripts/python.exe scripts/compare_old_vs_new_topology.py \
        --features analysis/topology_features.csv --split-at 30
"""

import argparse
import os

import numpy as np
import pandas as pd

from graphtalk import significance

# (column, is_boolean) -- boolean columns are compared as proportions
# (mean of 0/1), continuous columns as a difference of means.
_FEATURES = (
    ("num_nodes", False),
    ("num_edges", False),
    ("density", False),
    ("degree_mean", False),
    ("degree_std", False),
    ("component_count", False),
    ("circuit_rank", False),
    ("triangle_count", False),
    ("clustering_mean", False),
    ("is_tree", True),
    ("is_forest", True),
    ("is_bipartite", True),
    ("has_isolated_node", True),
    ("is_triangle_free", True),
)


def _values(frame: pd.DataFrame, column: str, is_boolean: bool) -> list:
  if is_boolean:
    return frame[column].astype(bool).astype(float).tolist()
  return frame[column].astype(float).tolist()


def _describe(values: list) -> str:
  array = np.asarray(values, dtype=float)
  return f"mean={array.mean():.3f} median={np.median(array):.3f} sd={array.std(ddof=1) if len(array) > 1 else 0.0:.3f}"


def compare(frame: pd.DataFrame, split_at: int, n_perm: int, seed: int) -> pd.DataFrame:
  original = frame[frame["index"] < split_at]
  new = frame[frame["index"] >= split_at]

  rows = []
  for column, is_boolean in _FEATURES:
    original_values = _values(original, column, is_boolean)
    new_values = _values(new, column, is_boolean)
    full_values = _values(frame, column, is_boolean)
    result = significance.unpaired_permutation_test(
        original_values, new_values, n_perm=n_perm, seed=seed
    )
    rows.append({
        "feature": column,
        "is_boolean": is_boolean,
        "original_mean": float(np.mean(original_values)),
        "new_mean": float(np.mean(new_values)),
        "full_mean": float(np.mean(full_values)),
        "observed_diff": result["observed_diff"],
        "p_value": result["p_value"],
        "n_original": result["n_a"],
        "n_new": result["n_b"],
    })
  report = pd.DataFrame(rows)
  report["bh_significant"] = significance.benjamini_hochberg(
      report["p_value"].tolist(), q=0.05
  )
  return report


def print_report(frame: pd.DataFrame, split_at: int, report: pd.DataFrame) -> None:
  original = frame[frame["index"] < split_at]
  new = frame[frame["index"] >= split_at]
  print(f"original (index < {split_at}): n={len(original)}")
  print(f"new (index >= {split_at}): n={len(new)}")
  print()
  print(f"{'feature':<20} {'original':>28} {'new':>28} {'diff(new-orig)':>15} "
        f"{'p':>8}  bh_sig")
  for column, is_boolean in _FEATURES:
    row = report[report["feature"] == column].iloc[0]
    original_values = _values(original, column, is_boolean)
    new_values = _values(new, column, is_boolean)
    print(
        f"{column:<20} {_describe(original_values):>28} {_describe(new_values):>28} "
        f"{row['observed_diff']:>+15.4f} {row['p_value']:>8.4f}  {row['bh_significant']}"
    )

  size_bucket_original = original["size_bucket"].value_counts(normalize=True)
  size_bucket_new = new["size_bucket"].value_counts(normalize=True)
  print("\nsize_bucket proportions (original vs new):")
  for bucket in ("small", "medium", "large"):
    print(f"  {bucket:<8} original={size_bucket_original.get(bucket, 0.0):.3f} "
          f"new={size_bucket_new.get(bucket, 0.0):.3f}")

  n_significant = int(report["bh_significant"].sum())
  print(f"\n{n_significant}/{len(report)} features differ significantly "
        f"(BH q=0.05) between the original {split_at} and the new "
        f"{len(new)} instances.")
  if n_significant == 0:
    print(
        "-> no detected compositional difference: the original draw looks "
        "like an unremarkable sample of the same fixed population -- "
        "supports 'scale, not topology' as the explanation for the "
        "significance flip."
    )
  else:
    print(
        "-> at least one feature differs -- worth checking as a candidate "
        "driver in scripts/analyze_topology_drivers.py before concluding "
        "'pure power'."
    )


def maybe_plot(frame: pd.DataFrame, split_at: int, out_dir: str) -> None:
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("\nmatplotlib not installed -- skipping distribution plots")
    return

  os.makedirs(out_dir, exist_ok=True)
  original = frame[frame["index"] < split_at]
  new = frame[frame["index"] >= split_at]
  continuous = [c for c, is_boolean in _FEATURES if not is_boolean]
  for column in continuous:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bins = np.histogram_bin_edges(frame[column].astype(float), bins=15)
    ax.hist(original[column].astype(float), bins=bins, alpha=0.5,
            density=True, label=f"original (n={len(original)})")
    ax.hist(new[column].astype(float), bins=bins, alpha=0.5,
            density=True, label=f"new (n={len(new)})")
    ax.set_title(column)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{column}.png"), dpi=120)
    plt.close(fig)
  print(f"\nwrote {len(continuous)} distribution plots to {out_dir}/")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--features", default="analysis/topology_features.csv")
  parser.add_argument("--split-at", type=int, default=30)
  parser.add_argument("--n-perm", type=int, default=10_000)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out", default=None,
                      help="optional path to write the per-feature report CSV")
  parser.add_argument("--plots-dir", default="analysis/topology_distribution_plots")
  parser.add_argument("--no-plots", action="store_true")
  args = parser.parse_args()

  frame = pd.read_csv(args.features)
  report = compare(frame, args.split_at, args.n_perm, args.seed)
  print_report(frame, args.split_at, report)
  if args.out:
    report.to_csv(args.out, index=False)
    print(f"\nwrote per-feature report to {args.out}")
  if not args.no_plots:
    maybe_plot(frame, args.split_at, args.plots_dir)


if __name__ == "__main__":
  main()
