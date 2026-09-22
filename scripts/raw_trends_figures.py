"""Figures for docs/raw-trends-large-graph.md, from csv2/raw-trends/*.csv.

  PYTHONPATH=. python scripts/raw_trends_figures.py

Writes PNGs to analysis/raw-trends/.

Colour follows the job, not taste:
  * signed effects (help/hurt) are a POLARITY job -> diverging blue<->red with a
    neutral gray midpoint, symmetric limits so zero is always the same colour.
  * conditions are an IDENTITY job -> a fixed categorical order, never cycled,
    never re-assigned when a filter drops a series.
  * magnitude-only panels use one hue, light->dark.
No dual axes anywhere; two measures of different scale get two panels.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUT = "analysis/raw-trends"
CSV = "csv2/raw-trends"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Categorical slots, fixed order (blue, orange, aqua, yellow, magenta, green).
SERIES = {
    "none": "#52514e",          # the control reads as ink, not as a series hue
    "filler": "#898781",        # the other control, muted
    "degree": "#2a78d6",
    "clustering": "#eb6834",
    "rwse": "#1baf7a",
    "components": "#eda100",
    "all": "#4a3aa7",
}
CONDS = ["none", "filler", "degree", "clustering", "rwse", "components", "all"]
PRIMERS = [c for c in CONDS if c != "none"]
DENS = ["0.1", "0.2", "0.35", "0.5", "0.65", "0.75", "0.85"]
ARMS = ["qwen3-1.7b", "qwen3-4b", "qwen3-1.7b-think", "qwen3-4b-think"]
TASKS = ["node_count", "cycle_check", "edge_existence", "node_degree",
         "connected_nodes", "edge_count"]

# Diverging ramp: blue <-> neutral gray <-> red. Never a rainbow, never a hue
# at the midpoint.
# Red at the NEGATIVE pole and blue at the positive one, so "red = the primer
# hurt" matches the reading everyone brings to a red cell. The neutral midpoint
# is gray, never a hue.
DIVERGING = LinearSegmentedColormap.from_list(
    "rd_gray_bl",
    ["#7d1f1f", "#d03b3b", "#ef9a99", "#f0efec", "#86b6ef", "#256abf", "#0d366b"])


def _style(ax, title=None, xlabel=None, ylabel=None):
  ax.set_facecolor(SURFACE)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color("#c3c2b7")
    ax.spines[side].set_linewidth(0.8)
  ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
  ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
  ax.set_axisbelow(True)
  if title:
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
  if xlabel:
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
  if ylabel:
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _fig(nrows, ncols, w, h):
  fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
  fig.patch.set_facecolor(SURFACE)
  return fig, np.atleast_1d(axes).ravel()


def _save(fig, name):
  os.makedirs(OUT, exist_ok=True)
  path = os.path.join(OUT, name)
  fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
  plt.close(fig)
  print("  ->", path)


# ---------------------------------------------------------------- figures

def fig_density_curves(eff):
  """Q7b: the 7-level curve. Only node_degree/edge_existence reach p=0.85."""
  for task in ("node_degree", "edge_existence"):
    fig, axes = _fig(1, 4, 16, 3.6)
    for ax, arm in zip(axes, ARMS):
      s = eff[(eff.arm == arm) & (eff.task == task)]
      for cond in PRIMERS:
        r = s[s.condition == cond].set_index("density").reindex(DENS)
        y = r["vs_filler_delta"] * 100
        ax.plot(range(len(DENS)), y, color=SERIES[cond], linewidth=2,
                marker="o", markersize=4.5, label=cond,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
      ax.axhline(0, color="#c3c2b7", linewidth=1.2, zorder=1)
      ax.set_xticks(range(len(DENS)))
      ax.set_xticklabels(DENS)
      _style(ax, arm, "edge probability p", None)
      ax.set_ylim(-50, 60)
    axes[0].set_ylabel("effect vs filler (pts)", color=INK2, fontsize=9)
    axes[-1].legend(frameon=False, fontsize=8, labelcolor=INK2,
                    loc="upper left", ncol=2)
    fig.suptitle("%s - primer effect against a length-matched control, "
                 "p = 0.1 to 0.85" % task, color=INK, fontsize=12, x=0.09,
                 ha="left", y=1.04)
    _save(fig, "fig_density_%s.png" % task)


def fig_effect_heatmap(eff):
  """Q1/Q2: where each primer helps and hurts. Polarity -> diverging."""
  fig, axes = _fig(2, 2, 13, 8)
  norm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
  for ax, arm in zip(axes, ARMS):
    grid = np.full((len(TASKS), len(PRIMERS)), np.nan)
    for i, task in enumerate(TASKS):
      for j, cond in enumerate(PRIMERS):
        r = eff[(eff.arm == arm) & (eff.task == task)
                & (eff.condition == cond) & (eff.density.isin(DENS))]
        if len(r):
          grid[i, j] = r["vs_filler_delta"].mean() * 100
    im = ax.imshow(grid, cmap=DIVERGING, norm=norm, aspect="auto")
    ax.set_xticks(range(len(PRIMERS)))
    ax.set_xticklabels(PRIMERS, rotation=30, ha="right")
    ax.set_yticks(range(len(TASKS)))
    # Only the left column carries task labels; repeating them in the right
    # column collided with the left panel's cells.
    ax.set_yticklabels(TASKS if arm in (ARMS[0], ARMS[2]) else [])
    ax.grid(False)
    for i in range(len(TASKS)):
      for j in range(len(PRIMERS)):
        if not np.isnan(grid[i, j]):
          v = grid[i, j]
          ax.text(j, i, "%+.0f" % v, ha="center", va="center", fontsize=8,
                  color="#ffffff" if abs(v) > 16 else INK)
    _style(ax, arm)
    ax.grid(False)
  cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
  cb.set_label("mean effect vs filler (pts), averaged over density",
               color=INK2, fontsize=9)
  cb.ax.tick_params(colors=MUTED, labelsize=8)
  cb.outline.set_visible(False)
  fig.suptitle("Primer effect by task and condition - blue helps, red hurts.\n"
               "Averaged over density; `node_count`/`cycle_check` have a "
               "constant gold answer, so their cells measure distraction.",
               color=INK, fontsize=12, x=0.09, ha="left", y=1.0)
  fig.subplots_adjust(wspace=0.12, hspace=0.55)
  _save(fig, "fig_effect_heatmap.png")


def fig_fix_break(eff):
  """Q1/Q2: a net effect hides whether a primer fixed items or broke them."""
  cells = [("qwen3-4b", "edge_count", "0.5", "degree"),
           ("qwen3-4b", "edge_count", "0.5", "all"),
           ("qwen3-1.7b", "edge_existence", "0.5", "degree"),
           ("qwen3-1.7b", "edge_existence", "0.5", "all"),
           ("qwen3-4b", "edge_existence", "0.5", "clustering"),
           ("qwen3-4b", "node_degree", "0.5", "degree"),
           ("qwen3-4b", "node_degree", "0.5", "all"),
           ("qwen3-4b", "edge_existence", "0.5", "rwse")]
  labels, fixed, broke = [], [], []
  for arm, task, dens, cond in cells:
    r = eff[(eff.arm == arm) & (eff.task == task) & (eff.density == dens)
            & (eff.condition == cond)]
    if r.empty:
      continue
    r = r.iloc[0]
    labels.append("%s\n%s / %s" % (cond, task, arm.replace("qwen3-", "")))
    fixed.append(r["vs_none_fixed"])
    broke.append(-r["vs_none_broke"])
  fig, axes = _fig(1, 1, 11, 4.6)
  ax = axes[0]
  y = np.arange(len(labels))
  ax.barh(y, fixed, color="#256abf", height=0.62, label="items fixed",
          edgecolor=SURFACE, linewidth=1.6)
  ax.barh(y, broke, color="#d03b3b", height=0.62, label="items broken",
          edgecolor=SURFACE, linewidth=1.6)
  for i, (f, b) in enumerate(zip(fixed, broke)):
    if f:
      ax.text(f + 0.8, i, "%d" % f, va="center", fontsize=8, color=INK2)
    if b:
      ax.text(b - 0.8, i, "%d" % -b, va="center", ha="right", fontsize=8,
              color=INK2)
  ax.axvline(0, color="#c3c2b7", linewidth=1.2)
  ax.set_yticks(y)
  ax.set_yticklabels(labels, fontsize=8)
  ax.invert_yaxis()
  _style(ax, "Primer effects are one-directional (vs none, p=0.5, n=100)",
         "items out of 100")
  ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
  _save(fig, "fig_fix_break.png")


def fig_serial_position(sp):
  """Q4/M2: the answer is in the primer - does position in it matter?"""
  show = ["none", "filler", "clustering", "degree", "all"]
  arms = ["qwen3-4b", "qwen3-1.7b", "qwen3-1.7b-think"]
  fig, axes = _fig(1, 3, 13, 3.8)
  buckets = ["acc_0_9", "acc_10_19", "acc_20_29", "acc_30_39"]
  xt = ["0-9", "10-19", "20-29", "30-39"]
  for ax, arm in zip(axes, arms):
    for cond in show:
      r = sp[(sp.arm == arm) & (sp.task == "node_degree")
             & (sp.condition == cond)]
      if r.empty:
        continue
      y = [r.iloc[0][b] * 100 for b in buckets]
      dashed = cond in ("none", "filler", "clustering")
      ax.plot(range(4), y, color=SERIES[cond], linewidth=2.2 if not dashed else 1.6,
              linestyle="--" if dashed else "-",
              marker="o", markersize=5, label=cond,
              markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(xt)
    _style(ax, arm, "target node's line in the 40-line primer")
  axes[0].set_ylabel("accuracy (%)", color=INK2, fontsize=9)
  axes[-1].legend(frameon=False, fontsize=8, labelcolor=INK2,
                  loc="center left", bbox_to_anchor=(1.02, 0.5))
  # Honest caption: the controls are NOT perfectly flat (1.7b dips mid-list and
  # recovers; filler on 1.7b-think slopes +12.1 pts), so the claim is about
  # steepness and monotonicity, not about the controls having zero slope.
  fig.suptitle(
      "node_degree: accuracy falls with the target's position in the primer -\n"
      "steeply and monotonically when the primer states the answer (solid); "
      "weakly or non-monotonically otherwise (dashed). Free y-axis per panel.",
      color=INK, fontsize=11, x=0.09, ha="left", y=1.12)
  _save(fig, "fig_serial_position.png")


def fig_balanced(bal):
  """Q4/M3: raw accuracy recovers at high density; the model stops answering."""
  fig, axes = _fig(1, 3, 13, 3.8)
  arm = "qwen3-1.7b"
  s = bal[(bal.arm == arm) & (bal.condition == "none")].set_index("density")
  s = s.reindex(DENS)
  x = range(len(DENS))
  ax = axes[0]
  ax.plot(x, s["acc"] * 100, color="#2a78d6", linewidth=2.2, marker="o",
          markersize=5, label="raw accuracy", markeredgecolor=SURFACE)
  ax.plot(x, s["balanced_acc"] * 100, color="#d03b3b", linewidth=2.2,
          marker="o", markersize=5, label="balanced accuracy",
          markeredgecolor=SURFACE)
  ax.axhline(50, color=MUTED, linewidth=1, linestyle=":")
  ax.text(0.1, 51.5, "chance", fontsize=8, color=MUTED)
  ax.set_xticks(x); ax.set_xticklabels(DENS)
  _style(ax, "%s, edge_existence, no primer" % arm, "edge probability p", "%")
  ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")

  ax = axes[1]
  ax.plot(x, s["pred_yes_rate"] * 100, color="#eb6834", linewidth=2.2,
          marker="o", markersize=5, label="model says yes",
          markeredgecolor=SURFACE)
  ax.plot(x, s["gold_yes_rate"] * 100, color=INK2, linewidth=1.6,
          linestyle="--", marker="o", markersize=4, label="actually yes",
          markeredgecolor=SURFACE)
  ax.set_xticks(x); ax.set_xticklabels(DENS)
  _style(ax, "the yes-rate runs away", "edge probability p", "% of pairs")
  ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")

  ax = axes[2]
  ax.plot(x, s["recall_yes"] * 100, color="#1baf7a", linewidth=2.2, marker="o",
          markersize=5, label="recall on real edges", markeredgecolor=SURFACE)
  ax.plot(x, s["recall_no"] * 100, color="#d03b3b", linewidth=2.2, marker="o",
          markersize=5, label="recall on non-edges", markeredgecolor=SURFACE)
  ax.set_xticks(x); ax.set_xticklabels(DENS)
  _style(ax, "every error is a false positive", "edge probability p", "%")
  ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center left")
  fig.suptitle("Raw accuracy hides a collapse: at p>=0.65 the model answers "
               "'yes' to 98% of pairs", color=INK, fontsize=12, x=0.09,
               ha="left", y=1.04)
  _save(fig, "fig_balanced_accuracy.png")


def fig_additivity(add):
  """Q6b: `all` bundles degree+clustering+rwse. Does it add up?"""
  fig, axes = _fig(1, 1, 6.2, 5.6)
  ax = axes[0]
  s = add[add.sum_parts.abs() > 0.02]
  ax.scatter(s.sum_parts * 100, s.d_all * 100, s=26, alpha=0.75,
             color="#2a78d6", edgecolor=SURFACE, linewidth=0.7,
             label="one (arm, task, density) cell")
  lim = 70
  ax.plot([-lim, lim], [-lim, lim], color=MUTED, linestyle="--", linewidth=1.2,
          label="additive (all = sum of parts)")
  ax.plot([-lim, lim], [-lim * 0.37, lim * 0.37], color="#d03b3b",
          linewidth=1.8, label="observed median (37% survives)")
  ax.axhline(0, color="#c3c2b7", linewidth=1)
  ax.axvline(0, color="#c3c2b7", linewidth=1)
  ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
  _style(ax, "Bundling dilutes: `all` keeps ~a third of its parts",
         "sum of degree + clustering + rwse effects (pts)",
         "effect of `all` (pts)")
  ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
  _save(fig, "fig_additivity.png")


def fig_behaviour(beh):
  """Q5a: the primer changes how much work the model does - both directions."""
  fig, axes = _fig(1, 3, 13, 3.8)
  panels = [("qwen3-4b", "node_degree", "the model stops counting"),
            ("qwen3-1.7b", "edge_count", "the model stops enumerating"),
            ("qwen3-1.7b", "edge_existence", "the model does more")]
  for ax, (arm, task, sub) in zip(axes, panels):
    s = beh[(beh.arm == arm) & (beh.task == task)]
    for cond in CONDS:
      r = s[s.condition == cond].set_index("density").reindex(DENS[:4])
      ax.plot(range(4), r["median_tokens"], color=SERIES[cond], linewidth=2,
              marker="o", markersize=4.5, label=cond,
              markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(DENS[:4])
    _style(ax, "%s / %s\n%s" % (task, arm.replace("qwen3-", ""), sub),
           "edge probability p")
  axes[0].set_ylabel("median new tokens (uncapped rows)", color=INK2, fontsize=9)
  axes[-1].legend(frameon=False, fontsize=7.5, labelcolor=INK2, ncol=2,
                  loc="upper left")
  fig.suptitle("Primers change effort in both directions, depending on the task",
               color=INK, fontsize=12, x=0.09, ha="left", y=1.06)
  _save(fig, "fig_behaviour_length.png")


def fig_capitulation(beh, ladder):
  """Why the small model quits - and why it is the answer format, not the load.

  Panel 1: same graphs, same model, same densities; only the required output
  differs. The binary task collapses to ~48 tokens, the integer task does not.
  Panel 2: the (n, k) ladder separates prompt length from task difficulty, and
  neither on its own produces the collapse.
  """
  fig, axes = _fig(1, 2, 13, 4.2)

  ax = axes[0]
  for task, hue, lab in (("edge_existence", "#d03b3b", "edge_existence (yes/no)"),
                         ("node_degree", "#256abf", "node_degree (integer)"),
                         ("connected_nodes", "#1baf7a", "connected_nodes (set)")):
    s = beh[(beh.arm == "qwen3-1.7b") & (beh.task == task)
            & (beh.condition == "none")].set_index("density").reindex(DENS)
    y = s["median_tokens"].to_numpy(dtype=float)
    keep = ~np.isnan(y)
    ax.plot(np.arange(len(DENS))[keep], y[keep], color=hue, linewidth=2.4,
            marker="o", markersize=5, label=lab, markeredgecolor=SURFACE,
            markeredgewidth=1)
  ax.set_xticks(range(len(DENS))); ax.set_xticklabels(DENS)
  _style(ax, "Same graphs, different required answer",
         "edge probability p", "median new tokens")
  ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
  ax.annotate("model bails to \"yes\"", xy=(5, 48), xytext=(3.1, 150),
              fontsize=8.5, color="#d03b3b",
              arrowprops=dict(arrowstyle="->", color="#d03b3b", lw=1.2))

  ax = axes[1]
  NS = [40, 60, 80, 120, 160, 300]
  for arm, hue in (("qwen3-1.7b", "#256abf"), ("qwen3-0.6b", "#eb6834"),
                   ("qwen3-8b", "#1baf7a")):
    s = ladder[(ladder.arm == arm) & (ladder.k == 8)].set_index("n").reindex(NS)
    ax.plot(range(len(NS)), s["median_tokens"], color=hue, linewidth=2.4,
            marker="o", markersize=5, label=arm.replace("qwen3-", ""),
            markeredgecolor=SURFACE, markeredgewidth=1)
  ax.set_xticks(range(len(NS)))
  ax.set_xticklabels([str(n) for n in NS])
  ax.set_ylim(0, 320)
  _style(ax, "Prompt 7.5x longer, difficulty held at k=8",
         "nodes (edges 160 → 1200)", "median new tokens")
  ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
  fig.suptitle("Capitulation is enabled by the answer format, not by load: a long prompt "
               "at fixed difficulty (right)\nnever collapses, but a binary answer under the "
               "same load (left) does.",
               color=INK, fontsize=11, x=0.06, ha="left", y=1.08)
  _save(fig, "fig_capitulation.png")


def main():
  eff = pd.read_csv(os.path.join(CSV, "effects.csv"))
  eff["density"] = eff["density"].astype(str)
  bal = pd.read_csv(os.path.join(CSV, "edge_existence_balanced.csv"))
  bal["density"] = bal["density"].astype(str)
  sp = pd.read_csv(os.path.join(CSV, "serial_position.csv"))
  add = pd.read_csv(os.path.join(CSV, "additivity.csv"))
  beh = pd.read_csv(os.path.join(CSV, "behaviour.csv"))
  beh["density"] = beh["density"].astype(str)

  print("writing figures:")
  fig_density_curves(eff)
  fig_effect_heatmap(eff)
  fig_fix_break(eff)
  fig_serial_position(sp)
  fig_balanced(bal)
  fig_additivity(add)
  fig_behaviour(beh)
  ladder = pd.read_csv(os.path.join(CSV, "ladder_length_vs_difficulty.csv"))
  fig_capitulation(beh, ladder)
  print("done")


if __name__ == "__main__":
  main()
