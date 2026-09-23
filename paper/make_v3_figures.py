"""Draw the two body figures of talk_like_a_graph.v3.tex into paper/.

  v3_fig_window.pdf     every cell's effect against its accuracy without a primer
  v3_fig_edgeexist.pdf  qwen3-1.7b on edge_existence: balanced accuracy and
                        response length by density, four conditions

Reads the CSVs scripts/primer_findings.py writes (--figure-data), so run that
first; paper/make_v3.sh does. Sized for one ACL column (3.03 in) so that text
prints at 7-8 pt; serif to match the body; TrueType fonts in the PDF.

  PYTHONPATH=. python paper/make_v3_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SRC = "csv2/raw-trends/"
OUT = "paper/"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def window():
    t = pd.read_csv(SRC + "primer_cells.csv")
    fig, ax = plt.subplots(figsize=(3.03, 2.0))
    ax.axvspan(0.25, 0.75, color="#f1f0ec", lw=0, zorder=0)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    for carries, col, mk, lab in ((False, ORANGE, "o", "side information"),
                                  (True, BLUE, "^", "states the answer")):
        s = t[t.carries == carries]
        sig = s.q < 0.05
        ax.scatter(s[~sig].baseline, s[~sig].delta, s=11, marker=mk,
                   facecolors="none", edgecolors=col, linewidths=0.6, zorder=2)
        ax.scatter(s[sig].baseline, s[sig].delta, s=11, marker=mk, color=col,
                   edgecolors="white", linewidths=0.3, zorder=3, label=lab)
    tr = t[(t.arm == "qwen3-4b") & (t.task == "node_degree")
           & (t.condition == "degree")].sort_values("density")
    ax.plot(tr.baseline, tr.delta, color=INK, lw=0.8, zorder=4)
    first, last = tr.iloc[0], tr.iloc[-1]
    ax.annotate(r"$p{=}.10$", (first.baseline, first.delta), xytext=(-6, 13),
                textcoords="offset points", fontsize=6.5, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=INK))
    ax.annotate(r"$p{=}.85$", (last.baseline, last.delta), xytext=(-8, 5),
                textcoords="offset points", fontsize=6.5, color=INK)
    ax.annotate("qwen3-4b, node_degree,\n" + r"$\mathtt{degree}$, $p{=}.10$ to $.85$",
                xy=tuple(tr.iloc[4][["baseline", "delta"]]), xytext=(0.30, -30),
                fontsize=6.3, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("accuracy without a primer")
    ax.set_ylabel("effect of the primer (points)")
    ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.2,
              borderaxespad=0.1, markerscale=1.1)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT + "v3_fig_window.pdf")
    plt.close(fig)
    print("wrote " + OUT + "v3_fig_window.pdf")


def edge_existence():
    c = pd.read_csv(SRC + "edge_existence_collapse.csv")
    series = [("none", INK2, "o", "-"), ("filler", ORANGE, "s", "-"),
              ("degree", BLUE, "^", "-"), ("all", AQUA, "D", "-")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.03, 1.65))
    for cond, col, mk, ls in series:
        s = c[c.condition == cond].sort_values("density")
        kw = dict(color=col, marker=mk, ms=3, lw=1.0, ls=ls, mew=0)
        a.plot(s.density, s.bacc, label=cond, **kw)
        b.plot(s.density, s.tokens, **kw)
    a.axhline(0.5, color=INK2, lw=0.5, ls=(0, (2, 2)))
    a.text(0.09, 0.505, "chance", fontsize=6, color=INK2, ha="left", va="bottom")
    a.set_ylim(0.45, 1.0)
    a.set_ylabel("balanced accuracy")
    b.set_ylabel("median new tokens")
    b.set_ylim(0, 430)
    for ax, tag in ((a, "(a)"), (b, "(b)")):
        ax.set_xticks([0.1, 0.35, 0.65, 0.85])
        ax.set_xticklabels([".10", ".35", ".65", ".85"])
        ax.set_xlabel(r"edge density $p$")
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.text(0.02, 1.02, tag, transform=ax.transAxes, fontsize=7,
                va="bottom", ha="left")
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=4,
               frameon=False, handlelength=1.6, columnspacing=1.0,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.9))
    fig.savefig(OUT + "v3_fig_edgeexist.pdf")
    plt.close(fig)
    print("wrote " + OUT + "v3_fig_edgeexist.pdf")


if __name__ == "__main__":
    window()
    edge_existence()
