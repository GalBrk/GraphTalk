"""Figures 2 and 3 of the paper, one column each, from outputs/n40-sweep/primer_cells.csv
(primer_findings.py --csv-dir writes it).

fig_headroom.pdf (Figure 2): the degree primer's effect on node degree against the no-primer
correct share, per arm and density; also docs/img/headroom.png for the front README.
fig_addstats.pdf (Figure 3): what adding clustering and RWSE to the same degrees changes
(all three statistics minus the degree primer), per arm and density.

Usage: python paper/fig_headroom.py   (writes both next to this file)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
CELLS = HERE.parent / "outputs" / "n40-sweep" / "primer_cells.csv"
ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"]
LABEL = {"qwen3-1.7b": "1.7P", "qwen3-1.7b-think": "1.7T", "qwen3-4b": "4P", "qwen3-4b-think": "4T"}
COLOR = {"qwen3-1.7b": "#2a78d6", "qwen3-1.7b-think": "#4a3aa7",
         "qwen3-4b": "#eb6834", "qwen3-4b-think": "#1baf7a"}
MARK = {"qwen3-1.7b": "o", "qwen3-1.7b-think": "^", "qwen3-4b": "o", "qwen3-4b-think": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

t = pd.read_csv(CELLS)
t = t[(t.task == "node_degree") & t.condition.isin(["degree", "all"])]
w = t.pivot_table(index=["arm", "density"], columns="condition", values="delta").reset_index()
w = w.join(t[t.condition == "degree"].set_index(["arm", "density"])[["baseline", "flagged"]],
           on=["arm", "density"])
w["diff"] = w["all"] - w["degree"]
# Hollow: flagged, or a plain arm's high-density run at the smaller token budget.
w["hollow"] = w.flagged | (~w.arm.str.endswith("think") & (w.density >= .65))


def axes(height=1.35):
    fig, ax = plt.subplots(figsize=(3.2, height))
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.tick_params(labelsize=7.5, colors=MUTED, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    return fig, ax


def points(ax, s, x, y):
    arm = s.arm.iloc[0]
    for h in (False, True):
        v = s[s.hollow == h]
        ax.scatter(v[x], v[y], s=28, marker=MARK[arm], edgecolors=COLOR[arm], linewidths=1.2,
                   zorder=3, facecolors="white" if h else COLOR[arm], label=None if h else LABEL[arm])


fig, a = axes()
for arm in ARMS:
    points(a, w[w.arm == arm].assign(pct=lambda d: 100 * d.baseline), "pct", "degree")
a.set_xlabel("No-primer correct share (%)", fontsize=7.5, color=INK)
a.set_ylabel("Effect (points)", fontsize=7.5, color=INK)
a.legend(fontsize=7, frameon=False, loc="upper right", ncol=2, handletextpad=0.2, columnspacing=0.8)
fig.tight_layout()
fig.savefig(HERE / "fig_headroom.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(HERE.parent / "docs" / "img" / "headroom.png", dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")

fig, b = axes(1.7)  # taller: its y-label is longer than a 1.35-inch axis
for arm in ARMS:
    s = w[w.arm == arm].sort_values("density")
    b.plot(s.density, s["diff"], color=COLOR[arm], lw=1.0, zorder=2)
    points(b, s, "density", "diff")
dens = sorted(w.density.unique())
b.set_xticks(dens, [f"{d:.2f}".lstrip("0") for d in dens])
b.set_xlabel("Edge density $p$", fontsize=7.5, color=INK)
b.set_ylabel("Change (points)", fontsize=7.5, color=INK)
b.legend(fontsize=7, frameon=False, loc="lower center", ncol=2, handletextpad=0.2, columnspacing=0.8)
fig.tight_layout()
fig.savefig(HERE / "fig_addstats.pdf", bbox_inches="tight", pad_inches=0.02)
