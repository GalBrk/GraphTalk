"""Build figure F1: cross-fitted delta vs baseline, route vs no-route,
all four densfull40 arms plus the densfull40hi extension.

This is the scatter behind analyze_baseline_law.py's `--test crossfit`:
baseline and delta are estimated from disjoint halves of the paired graphs
(see that script's module docstring for why), so the negative slope in the
route group can't be regression to the mean by construction. Reads the
scored runs directly, via the same functions the table numbers come from,
so the figure cannot drift from them.

Route classification uses shortcuts_n40_flat.json (the n=40 refit, mean
over densities -- scripts/shortcut_table_n40.py), not the published-split
shortcuts.json: at n=40 edge_existence's `none` baseline is already close
to every other condition's blind-solver bar (class imbalance alone gets
most of the way there), so every edge_existence condition reclassifies
from "route" to "no route" once the bars are refit on the right graph
size. See docs/paper-revision-handoff.md plan step 2.

  PYTHONPATH=. python paper/make_figure_f1.py
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
import analyze_baseline_law as abl

CORPORA = ("densfull40", "densfull40hi")
MARKERS = {"qwen3-1.7b": "o", "qwen3-1.7b-think": "s",
          "qwen3-4b": "^", "qwen3-4b-think": "D"}


def route_only(cells):
    return [c for c in cells
            if c["task"] not in abl.DEGENERATE_TASKS and c["route"]]


def no_route_only(cells):
    return [c for c in cells
            if c["task"] not in abl.DEGENERATE_TASKS and not c["route"]]


def main():
    with open("shortcuts_n40_flat.json", encoding="utf-8") as fh:
        bars = json.load(fh)

    cross = []
    for arm in abl.DENSFULL_ARMS:
        for corpus in CORPORA:
            patterns = [f"runs/{arm}.{corpus}.shard*.jsonl"]
            cross += abl.arm_cells_crossfit(arm, patterns, bars)

    route, no_route = route_only(cross), no_route_only(cross)

    fig, (ax_r, ax_n) = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, cells, title in ((ax_r, route, "substitute route"),
                             (ax_n, no_route, "no route")):
        for arm in abl.DENSFULL_ARMS:
            pts = [c for c in cells if c["arm"] == arm]
            if not pts:
                continue
            ax.scatter([c["baseline"] for c in pts], [c["delta"] for c in pts],
                       marker=MARKERS[arm], s=14, alpha=0.55, label=arm,
                       edgecolors="none")
        if len(cells) >= 4:
            xs, ys = [c["baseline"] for c in cells], [c["delta"] for c in cells]
            slope, intercept = abl.fit_line(xs, ys)
            r, p = abl.pearson(xs, ys)
            lo, hi = min(xs), max(xs)
            ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept],
                    color="black", linewidth=1.2, zorder=0)
            # Each cell appears once per fold direction, so len(cells) double-
            # counts and a Pearson p on it is anticonservative; label unique
            # cells and leave inference to the block bootstrap in the text.
            n_cells = len({(c["arm"], c["task"], c["density"], c["condition"])
                           for c in cells})
            ax.text(0.03, 0.03, f"r={r:+.2f}  cells={n_cells}",
                    transform=ax.transAxes, fontsize=7.5, va="bottom")
        ax.axhline(0, color="#bbbbbb", linewidth=0.7, zorder=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("cross-fitted baseline accuracy", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.25, linewidth=0.5)
    ax_r.set_ylabel("cross-fitted paired delta (pp)", fontsize=8.5)
    ax_r.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig("paper/crossfit.pdf", bbox_inches="tight")
    print("wrote paper/crossfit.pdf")


if __name__ == "__main__":
    main()
