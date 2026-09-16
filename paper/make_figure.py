"""Build figure 1: node_degree accuracy vs density, all four arms.

Reads the scored runs directly so the figure cannot drift from the tables.

  PYTHONPATH=. python paper/make_figure.py
"""
import collections
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"]
SHOW = ["none", "filler", "clustering", "degree", "all"]
STYLE = {
    "none":       ("#444444", "o", "-",  2.2),
    "filler":     ("#999999", "s", "--", 1.4),
    "clustering": ("#1f77b4", "^", "-",  1.4),
    "degree":     ("#d62728", "D", "-",  1.4),
    "all":        ("#2ca02c", "v", ":",  1.4),
}


def density(iid):
    for part in iid.split("/"):
        if len(part) > 1 and part[0] == "p":
            try:
                return float(part[1:])
            except ValueError:
                pass
    return None


def main():
    from graphtalk import scoring

    fig, axes = plt.subplots(1, 4, figsize=(11, 2.7), sharey=True)
    for ax, arm in zip(axes, ARMS):
        acc = collections.defaultdict(lambda: [0.0, 0])
        seen = set()
        for path in sorted(glob.glob(f"runs/{arm}.densfull40.shard*of25.jsonl")):
            for line in open(path):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["task"] != "node_degree" or r["condition"] not in SHOW:
                    continue
                k = (r["instance_id"], r["condition"])
                if k in seen or r.get("hit_cap"):
                    continue
                seen.add(k)
                e = scoring.score_one(
                    scoring.extract_answer(r["response"], "node_degree"),
                    r["gold"], "node_degree")["exact"]
                cell = acc[(r["condition"], density(r["instance_id"]))]
                cell[0] += e
                cell[1] += 1
        for cond in SHOW:
            xs = sorted({d for (c, d) in acc if c == cond})
            ys = [acc[(cond, d)][0] / acc[(cond, d)][1] for d in xs]
            colour, marker, ls, lw = STYLE[cond]
            ax.plot(xs, ys, color=colour, marker=marker, linestyle=ls,
                    linewidth=lw, markersize=4, label=cond)
        ax.axhline(0.19, color="#bbbbbb", linewidth=0.8, zorder=0)
        ax.set_title(arm, fontsize=9)
        ax.set_xlabel("edge density $p$", fontsize=8)
        ax.set_xticks([0.1, 0.2, 0.35, 0.5])
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("accuracy", fontsize=8)
    axes[0].legend(fontsize=6.5, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig("paper/density.pdf", bbox_inches="tight")
    print("wrote paper/density.pdf")


if __name__ == "__main__":
    main()
