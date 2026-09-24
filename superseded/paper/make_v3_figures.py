"""Draw the two body figures of talk_like_a_graph.v3.tex into paper/.

  v3_fig_window.pdf     every cell's effect against its accuracy without a primer
  v3_fig_edgeexist.pdf  qwen3-1.7b on edge_existence: balanced accuracy and
                        response length by density, four conditions

Reads the CSVs scripts/primer_findings.py writes (--figure-data), so run that
first; superseded/paper/make_v3.sh does. Sized for one ACL column (3.03 in) so that text
prints at 7-8 pt; serif to match the body; TrueType fonts in the PDF.

  PYTHONPATH=. python superseded/paper/make_v3_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SRC = "superseded/csv2/raw-trends/"  # the snapshot this build was made from
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
    t["base"] = 100 * t.baseline
    fig, ax = plt.subplots(figsize=(3.03, 1.75))
    ax.axvspan(25, 75, color="#f1f0ec", lw=0, zorder=0)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    for carries, col, mk, lab in ((False, ORANGE, "o", "side information"),
                                  (True, BLUE, "^", "answer-carrying")):
        s = t[t.carries == carries]
        sig = s.q < 0.05
        ax.scatter(s[~sig].base, s[~sig].delta, s=11, marker=mk,
                   facecolors="none", edgecolors=col, linewidths=0.6, zorder=2)
        ax.scatter(s[sig].base, s[sig].delta, s=11, marker=mk, color=col,
                   edgecolors="white", linewidths=0.3, zorder=3, label=lab)
    tr = t[(t.arm == "qwen3-4b") & (t.task == "node_degree")
           & (t.condition == "degree")].sort_values("density")
    ax.plot(tr.base, tr.delta, color=INK, lw=0.8, zorder=4)
    first, last = tr.iloc[0], tr.iloc[-1]
    ax.annotate(r"$p$=.10", (first.base, first.delta), xytext=(-8, 13),
                textcoords="offset points", fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=INK))
    ax.annotate(r"$p$=.85", (last.base, last.delta), xytext=(-9, 5),
                textcoords="offset points", fontsize=7, color=INK)
    ax.annotate("qwen3-4b, node_degree,\ndegree, p=.10 to .85",
                xy=tuple(tr.iloc[4][["base", "delta"]]), xytext=(29, -31),
                fontsize=7, color=INK, family=["Courier New", "monospace"],
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    ax.set_xlim(-2, 102)
    ax.set_xlabel("accuracy without a primer (%)")
    ax.set_ylabel("effect of the primer (points)")
    ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.2,
              borderaxespad=0.1, markerscale=1.1)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT + "v3_fig_window.pdf")
    plt.close(fig)
    print("wrote " + OUT + "v3_fig_window.pdf")


def edge_existence():
    c = pd.read_csv(SRC + "edge_existence_collapse.csv")
    # A distinct dash per condition, so the lines stay apart in greyscale print.
    series = [("none", INK2, "o", "-"), ("filler", ORANGE, "s", (0, (1, 1))),
              ("degree", BLUE, "^", (0, (4, 1.5))), ("all", AQUA, "D", (0, (6, 1, 1, 1)))]
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.03, 1.45))
    for cond, col, mk, ls in series:
        s = c[c.condition == cond].sort_values("density")
        kw = dict(color=col, marker=mk, ms=3, lw=1.0, ls=ls, mew=0)
        a.plot(s.density, 100 * s.bacc, label=cond, **kw)
        b.plot(s.density, s.tokens, **kw)
    a.axhline(50, color=INK2, lw=0.5, ls=(0, (2, 2)))
    a.text(0.09, 50.5, "chance", fontsize=7, color=INK2, ha="left", va="bottom")
    a.set_ylim(45, 100)
    a.set_ylabel("balanced accuracy (%)")
    b.set_ylabel("median new tokens")
    b.set_ylim(0, 430)
    dens = sorted(c.density.unique())
    for ax, tag in ((a, "(a)"), (b, "(b)")):
        # a tick at every density; labels on four so that 7-pt text fits
        ax.set_xticks(dens)
        ax.set_xticklabels([f"{d:.2f}".lstrip("0") if d in (0.10, 0.35, 0.65, 0.85)
                            else "" for d in dens])
        ax.set_xlabel(r"edge density $p$")
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.text(0.02, 1.02, tag, transform=ax.transAxes, fontsize=7,
                va="bottom", ha="left")
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=4,
               frameon=False, handlelength=2.4, columnspacing=1.0,
               bbox_to_anchor=(0.5, 1.02), prop={"family": "monospace", "size": 7})
    fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.9))
    fig.savefig(OUT + "v3_fig_edgeexist.pdf")
    plt.close(fig)
    print("wrote " + OUT + "v3_fig_edgeexist.pdf")


if __name__ == "__main__":
    window()
    edge_existence()
