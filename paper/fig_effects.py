"""The effects figure of the paper, one column: every unflagged cell of Table 1 (each primer
against no primer, per arm and task), grouped by what the primer states. Filled: q < .05.
Values are read from the [main] block of outputs/n40-sweep/primer_findings.txt.

Usage: python paper/fig_effects.py   (writes fig_effects.pdf next to this file)
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "outputs" / "n40-sweep" / "primer_findings.txt"
ARMS = {"qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"}
TASKS = {"node_degree", "connected_nodes", "edge_count", "edge_existence"}
CONDS = {"degree", "clustering", "rwse", "all", "components", "filler"}
ROWS = ["States the answer", "Other statistics", "Controls"]
SOLID, PALE, BAND, INK, MUTED, GRID = "#2a78d6", "#b9d3f2", "#f1f0ec", "#0b0b0b", "#6b6a66", "#e6e5e1"


def group(task, cond):
  if cond in ("components", "filler"):
    return "Controls"
  if cond in ("degree", "all") and task in ("node_degree", "edge_count"):
    return "States the answer"
  return "Other statistics"


main = SOURCE.read_text(encoding="utf-8").split("\n[main]", 1)[1].split("\n[", 1)[0]
cells, key = [], None
for line in main.splitlines():
  if m := re.match(r"^  (\S+) (\S+) vs (\S+):$", line):
    key = m.groups()
  elif key and (m := re.match(r"^    (\w+)\s+[\d.]+ -> [\d.]+: ([+-][\d.]+) .* q=(\S+) ", line)):
    arm, task, control = key
    cond, x, q = m.groups()
    if control == "none" and arm in ARMS and task in TASKS and cond in CONDS and "FLAGGED" not in line:
      cells.append(dict(x=float(x), sig=float(q) < .05, g=group(task, cond)))
counts = [sum(c["g"] == g for c in cells) for g in ROWS]
assert counts == [10, 42, 26], counts  # Table 1's unflagged cells: edge count is flagged except in 4P

fig, ax = plt.subplots(figsize=(3.2, 1.55))
for x in (-10, 10, 20, 30):
  ax.axvline(x, color=GRID, lw=0.5, zorder=0)
ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
for ri, g in enumerate(ROWS):
  cs = [c for c in cells if c["g"] == g]
  ax.plot([min(c["x"] for c in cs), max(c["x"] for c in cs)], [-ri, -ri], color=BAND, lw=13,
          solid_capstyle="round", zorder=1)
  placed = []
  for c in sorted(cs, key=lambda c: (not c["sig"], abs(c["x"]))):
    for k in range(12):  # swarm: nearest free vertical slot
      dy = (k + 1) // 2 * 0.075 * (1 if k % 2 else -1)
      if all(abs(c["x"] - px) > 1.0 or abs(dy - pdy) > 0.08 for px, pdy in placed):
        break
    placed.append((c["x"], dy))
    ax.scatter(c["x"], -ri + dy, s=15, marker="o", linewidths=0.5, edgecolors="white",
               zorder=4 if c["sig"] else 3, facecolors=SOLID if c["sig"] else PALE)  # significant on top

ax.set_yticks([0, -1, -2], ROWS, fontsize=7, color=INK)
ax.set_xticks([-10, 0, 10, 20, 30], ["−10", "0", "+10", "+20", "+30"])
ax.tick_params(axis="both", length=0, labelsize=7)
ax.tick_params(axis="x", colors=MUTED)
ax.set_xlim(-18, 31)
ax.set_ylim(-2.5, 0.5)
ax.set_xlabel("Change against no primer (points)", fontsize=7, color=INK)
for s in ax.spines.values():
  s.set_visible(False)
handles = [plt.Line2D([], [], ls="", marker="o", ms=4, mfc=SOLID, mec="white", label="q < .05"),
           plt.Line2D([], [], ls="", marker="o", ms=4, mfc=PALE, mec="white", label="not significant")]
ax.legend(handles=handles, fontsize=6.5, frameon=False, ncol=2, loc="lower right",
          bbox_to_anchor=(1.0, 0.95), handletextpad=0.1, columnspacing=0.9)
fig.tight_layout()
fig.savefig(HERE / "fig_effects.pdf", bbox_inches="tight", pad_inches=0.02)
