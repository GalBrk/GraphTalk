"""Build figure F3: rewiring -- accuracy by triangle level x condition.

Reads the JSON scripts/analyze_rewiring_sweep.py already wrote (one record
per model/rung/level/condition, each a paired test against that level's own
`none` baseline), so this figure cannot drift from the numbers in those
files. Restricted to the `n40k12` rung, the one shared by all three arms
(qwen35-2b also has n60k12/n60k16, a separate size/degree extension not
comparable to the two qwen3-1.7b arms).

  PYTHONPATH=. python superseded/paper/make_figure_f3.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen35-2b"]
LEVELS = ["low", "base", "high"]
STYLE = {
    # Same condition colours as make_figure_density.py.
    "none":       ("#52514e", "o", "-"),
    "clustering": ("#2a78d6", "^", "-"),
    "filler":     ("#eb6834", "s", "--"),
}


def main():
    fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.75), sharey=True)
    for ax, arm in zip(axes, ARMS):
        with open(f"rewiring_{arm}.json", encoding="utf-8") as fh:
            records = [r for r in json.load(fh) if r["rung"] == "n40k12"]

        none_by_level = {r["level"]: r["control"]
                         for r in records if r["level"] is not None}
        series = {"none": [none_by_level[lv] for lv in LEVELS]}
        for cond in ("clustering", "filler"):
            series[cond] = [next(r["treatment"] for r in records
                                 if r["level"] == lv and r["condition"] == cond)
                            for lv in LEVELS]

        for cond, ys in series.items():
            colour, marker, ls = STYLE[cond]
            ax.plot(LEVELS, ys, color=colour, marker=marker, linestyle=ls,
                    linewidth=1.2, markersize=3, label=cond)
        ax.set_title(arm, fontsize=6.5)
        ax.set_xlabel("triangle level", fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.set_ylim(0.4, 1.0)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("accuracy", fontsize=6)
    axes[2].legend(fontsize=5.5, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig("superseded/paper/rewiring.pdf", bbox_inches="tight")
    print("wrote superseded/paper/rewiring.pdf")


if __name__ == "__main__":
    main()
