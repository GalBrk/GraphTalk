"""Build the density figure (superseded/paper/effects.pdf): the degree/filler continuum
above the clustering/filler headline, stacked on a shared density axis.

Both panels are paired differences against `none` with 95% percentile-
bootstrap intervals over graphs, on node_degree, over the same seven
densities -- which is why they share an x axis. Stacking them aligns the
two effects density by density: the answer-stating primer (a) climbs
monotonically as the task gets harder, the non-answer primer (b) peaks in
the middle and is roughly four times smaller, which is why the panels keep
separate y scales.

They read the scored runs through the same loaders the tables use
(analyze_headline_robustness, analyze_baseline_law), so the figures cannot
drift from the numbers in the text.

  PYTHONPATH=. python superseded/paper/make_figure_density.py
"""
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path[:0] = ["superseded/scripts", "scripts"]  # archived full copies first
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


def headline(ax):
  scores = {**ahr._load(ahr.MAIN_GLOBS), **ahr._load(ahr.FILLER_GLOB)}
  levels = sorted({abl.density_of(i) for i, _ in scores})
  xs = list(range(len(levels)))
  for cond, off in (("clustering", -0.08), ("filler", 0.08)):
    pts = []
    for lv in levels:
      diffs = [scores[(i, cond)] - scores[(i, "none")] for i, c in scores
               if c == "none" and abl.density_of(i) == lv and (i, cond) in scores]
      lo, hi = boot_ci(diffs)
      pts.append((100 * sum(diffs) / len(diffs), lo, hi))
    _series(ax, xs, pts, cond, off)
  _style(ax, xs, [f"{p:.2f}".lstrip("0") for p in levels])
  ax.set_xlim(-0.4, len(levels) + 0.35)
  ax.set_xlabel("edge density $p$", fontsize=7.5, color=MUTED)
  ax.set_title("(b) clustering, qwen3-1.7b", fontsize=7.5, color=INK,
               loc="left", pad=3)
  return levels


def continuum(ax):
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
  for cond, off in (("degree", -0.08), ("filler", 0.08)):
    pts = []
    for lv in levels:
      cell = next(c for c in cells if c["density"] == lv and c["condition"] == cond)
      diffs = [v - b for _, b, v in cell["pairs"]]
      lo, hi = boot_ci(diffs)
      pts.append((100 * sum(diffs) / len(diffs), lo, hi))
    _series(ax, xs, pts, cond, off)
  _style(ax, xs, [f"{p:.2f}".lstrip("0") for p in levels])
  ax.set_xlim(-0.4, len(levels) + 0.35)
  ax.set_title("(a) degree, qwen3-1.7b-think", fontsize=7.5, color=INK,
               loc="left", pad=12)
  # Baseline accuracy rides the top axis rather than a second tick row under
  # the density labels: the shared bottom axis belongs to both arms, this
  # arm's accuracies do not.
  top = ax.secondary_xaxis("top")
  top.set_xticks(xs)
  top.set_xticklabels([f"{none_acc[p]:.2f}" for p in levels], fontsize=6,
                      color=MUTED)
  top.tick_params(length=0, pad=1)
  top.set_xlabel("accuracy under none", fontsize=6.5, color=MUTED, labelpad=1)
  for side in ("top",):
    top.spines[side].set_visible(False)
  return levels


def main():
  fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(3.4, 3.3), sharex=True)
  continuum(ax_top)
  headline(ax_bot)
  ax_top.tick_params(axis="x", labelbottom=False)
  ax_top.set_xlabel("")
  for ax in (ax_top, ax_bot):
    ax.set_ylabel("$\\Delta$ vs. none (points)", fontsize=7.5, color=MUTED)
  fig.tight_layout(h_pad=1.4)
  fig.savefig("superseded/paper/effects.pdf", bbox_inches="tight")
  print("wrote superseded/paper/effects.pdf")


if __name__ == "__main__":
  main()
