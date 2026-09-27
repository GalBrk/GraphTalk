"""Reproduce the Qwen3-4B thinking edge-count outcome figure.

Values are copied from GraphTalk/results-sot at
csv2/raw-trends/primer_findings.txt, [edgecount].
"""

import matplotlib.pyplot as plt
import numpy as np

conditions = ["None", "Components", "Degree", "All three"]
correct = np.array([20.00, 44.75, 28.50, 21.75])
wrong = np.array([0.50, 0.75, 12.75, 22.50])
truncated = np.array([79.50, 54.50, 58.75, 55.75])
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
fig.savefig("edge_count_outcomes.pdf", bbox_inches="tight", pad_inches=0.01)
