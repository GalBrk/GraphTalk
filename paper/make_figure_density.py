"""Build the two density figures: the clustering headline split by density
(paper/headline.pdf) and the degree/filler density continuum
(paper/continuum.pdf).

Both are paired differences against `none` with 95% percentile-bootstrap
intervals over graphs. They read the scored runs through the same loaders
the tables use (analyze_headline_robustness, analyze_baseline_law), so the
figures cannot drift from the numbers in the text.

  PYTHONPATH=. python paper/make_figure_density.py
"""
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
import analyze_baseline_law as abl  # noqa: E402
import analyze_headline_robustness as ahr  # noqa: E402
import analyze_review_checks as arc  # noqa: E402

# One colour per condition across every figure (validated categorical slots
# 1-3: blue, orange, aqua); marker and dash carry identity without colour.
STYLE = {
    "clustering": ("#2a78d6", "^", "-"),
    "filler":     ("#eb6834", "s", "--"),
    "degree":     ("#1baf7a", "D", "-"),
}
INK, MUTED = "#0b0b0b", "#52514e"


def boot_ci(diffs, n_boot=2000, seed=0):
  rng = random.Random(seed)
  n = len(diffs)
  means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                 for _ in range(n_boot))
  return 100 * means[int(0.025 * (n_boot - 1))], 100 * means[int(0.975 * (n_boot - 1))]


def _style(ax, xs, labels):
  ax.axhline(0, color=MUTED, linewidth=0.8, zorder=0)
  ax.set_xticks(xs)
  ax.set_xticklabels(labels, fontsize=6.5, color=MUTED)
  ax.tick_params(axis="y", labelsize=7, colors=MUTED)
  ax.grid(axis="y", alpha=0.25, linewidth=0.5)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color("#bbbbbb")


def _series(ax, xs, points, cond, offset, end_label=True):
  colour, marker, ls = STYLE[cond]
  ys = [p[0] for p in points]
  lo = [p[0] - p[1] for p in points]
  hi = [p[2] - p[0] for p in points]
  xo = [x + offset for x in xs]
  ax.errorbar(xo, ys, yerr=[lo, hi], color=colour, marker=marker,
              linestyle=ls, linewidth=1.4, markersize=4.5, capsize=0,
              elinewidth=0.9, label=cond)
  if end_label:
    ax.annotate(cond, (xo[-1], ys[-1]), xytext=(5, 0), textcoords="offset points",
                fontsize=7, color=INK, va="center")


def headline():
  scores = {**ahr._load(ahr.MAIN_GLOBS), **ahr._load(ahr.FILLER_GLOB)}
  levels = sorted({abl.density_of(i) for i, _ in scores})
  xs = list(range(len(levels)))
  fig, ax = plt.subplots(figsize=(3.4, 2.15))
  for cond, off in (("clustering", -0.08), ("filler", 0.08)):
    pts = []
    for lv in levels:
      diffs = [scores[(i, cond)] - scores[(i, "none")] for i, c in scores
               if c == "none" and abl.density_of(i) == lv and (i, cond) in scores]
      lo, hi = boot_ci(diffs)
      pts.append((100 * sum(diffs) / len(diffs), lo, hi))
    _series(ax, xs, pts, cond, off, end_label=False)
  _style(ax, xs, [f"{p:.2f}".lstrip("0") for p in levels])
  ax.set_xlim(-0.4, len(levels) - 0.6)
  ax.set_xlabel("edge density $p$", fontsize=7.5, color=MUTED)
  ax.set_ylabel("$\\Delta$ vs. none (points)", fontsize=7.5, color=MUTED)
  ax.legend(fontsize=6.5, loc="lower left", frameon=False)
  fig.tight_layout()
  fig.savefig("paper/headline.pdf", bbox_inches="tight")
  print("wrote paper/headline.pdf")


def continuum():
  arm = "qwen3-1.7b-think"
  scores = abl.score_run(abl.CONTINUUM_GLOBS[arm], task_filter="node_degree",
                         skip_replication_seed=True)
  cells = arc.paired_cells(scores)
  levels = sorted({c["density"] for c in cells})
  none_acc = {}
  for lv in levels:
    vals = [v for (_, d, _, c), v in scores.items()
            if c == "none" and d == lv and v is not None]
    none_acc[lv] = sum(vals) / len(vals)
  xs = list(range(len(levels)))
  fig, ax = plt.subplots(figsize=(3.4, 2.15))
  for cond, off in (("degree", -0.08), ("filler", 0.08)):
    pts = []
    for lv in levels:
      cell = next(c for c in cells if c["density"] == lv and c["condition"] == cond)
      diffs = [v - b for _, b, v in cell["pairs"]]
      lo, hi = boot_ci(diffs)
      pts.append((100 * sum(diffs) / len(diffs), lo, hi))
    _series(ax, xs, pts, cond, off)
  _style(ax, xs, [f"{p:.2f}".lstrip("0") + f"\n{none_acc[p]:.2f}" for p in levels])
  ax.set_xlim(-0.4, len(levels) - 0.1)
  ax.set_xlabel("edge density $p$ / accuracy under none", fontsize=7.5, color=MUTED)
  ax.set_ylabel("$\\Delta$ vs. none (points)", fontsize=7.5, color=MUTED)
  ax.legend(fontsize=6.5, loc="upper left", frameon=False)
  fig.tight_layout()
  fig.savefig("paper/continuum.pdf", bbox_inches="tight")
  print("wrote paper/continuum.pdf")


if __name__ == "__main__":
  headline()
  continuum()
