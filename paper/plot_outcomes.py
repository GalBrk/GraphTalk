"""Reproduce the thinking-4B edge-count figure from [edgecount] output.

Usage: python paper/plot_outcomes.py   (writes edge_count_outcomes.pdf next to this file)
"""

from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "outputs/n40-sweep/primer_findings.txt"
block = SOURCE.read_text(encoding="utf-8").split("[edgecount]", 1)[1].split("\n[", 1)[0]
pattern = re.compile(r"^\s*qwen3-4b-think\s+(\w+)\s+correct ([\d.]+)% "
                     r"wrong ([\d.]+)% truncated ([\d.]+)%", re.M)
data = {condition: tuple(map(float, values)) for condition, *values in pattern.findall(block)}
order = [("none", "None"), ("components", "Components"),
         ("degree", "Degree"), ("all", "All three")]
if not all(condition in data for condition, _ in order):
    raise ValueError(f"Missing thinking-4B edge-count conditions in {SOURCE}")
conditions = [label for _, label in order]
correct, wrong, truncated = (
    np.array([data[condition][i] for condition, _ in order]) for i in range(3)
)
assert np.allclose(correct + wrong + truncated, 100)

fig, ax = plt.subplots(figsize=(3.12, 1.9))
y = np.arange(len(conditions))
ax.barh(y, correct, color="#333333", height=0.65, label="Correct")
ax.barh(y, wrong, left=correct, color="#aaaaaa", height=0.65, label="Wrong")
ax.barh(
    y,
    truncated,
    left=correct + wrong,
    color="#ffffff",
    edgecolor="#888888",
    linewidth=0.45,
    hatch="////",
    height=0.65,
    label="Truncated",
)
for index, value in enumerate(correct):
    ax.text(value / 2, index, f"{value:g}", va="center", ha="center", color="white", fontsize=7)
for index, value in enumerate(truncated):
    ax.text(100 - value / 2, index, f"{value:g}", va="center", ha="center", fontsize=7)
ax.set_yticks(y, conditions, fontsize=7)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Share of all responses (%)", fontsize=7, labelpad=1)
ax.tick_params(axis="x", labelsize=7, length=2)
ax.tick_params(axis="y", length=0)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, fontsize=6.5, frameon=False)
fig.subplots_adjust(left=0.26, right=0.99, top=0.77, bottom=0.23)
fig.savefig(HERE / "edge_count_outcomes.pdf", bbox_inches="tight", pad_inches=0.01)
