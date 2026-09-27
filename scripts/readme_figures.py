"""The two figures on the repo's front page, written to docs/img/.

  primer_effect.png  node_degree without a primer and with the degree primer
                     (states the answer) or clustering (states none), per arm;
                     every value is read from the [main] block of
                     outputs/n40-sweep/primer_findings.txt.
  primer_usage.png   how a primer is used: the seven conditions, a small graph,
                     and the prompt they make; every sentence is rendered by
                     graphtalk/primers.py.

  PYTHONPATH=. python scripts/readme_figures.py
"""
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from graphtalk import graphqa, primers
from talk_like_a_graph import graph_text_encoders

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
REPORT = ROOT / "outputs" / "n40-sweep" / "primer_findings.txt"
ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"]
NAMES = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-1.7b-think": "Qwen3-1.7B thinking",
         "qwen3-4b": "Qwen3-4B", "qwen3-4b-think": "Qwen3-4B thinking"}
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#d6d5d0"
ACCENT, ACCENT_BG, GRAY_BG, CARD = "#2a78d6", "#e9f2fc", "#f1f0ec", "#fbfbfa"
MONO = "DejaVu Sans Mono"


def main_rows(report):
    """(arm, condition) -> (none, with, effect, q) for node_degree vs none, from [main]."""
    rows, arm, on = {}, None, False
    for line in report.splitlines():
        if re.match(r"^\[[a-z0-9]+\]", line):
            on = line.startswith("[main]")
            continue
        if not on:
            continue
        m = re.match(r"^  (qwen3-[\w.-]+) node_degree vs none:", line)
        if m or re.match(r"^  \S", line):
            arm = m.group(1) if m else None
            continue
        m = re.match(r"^    (degree|clustering)\s+([\d.]+) -> ([\d.]+): ([+-][\d.]+) .*? q=(\S+)", line)
        if arm and m:
            rows[(arm, m.group(1))] = tuple(float(g) for g in m.groups()[1:])
    return rows


def primer_effect(rows):
    """Dumbbells: no primer (dot) to the degree primer (arrow); clustering as a gray ghost.
    Blue gains and red losses have q < .05; gray ones do not."""
    gain, loss, flat, ghost = "#2a78d6", "#e34948", "#9a9994", "#c9c8c3"
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    for y, a in enumerate(reversed(ARMS)):
        none, cl, _, _ = rows[(a, "clustering")]
        ax.plot([none, cl], [y - .22] * 2, color=ghost, lw=2, solid_capstyle="round", zorder=1)
        ax.scatter([cl], [y - .22], s=26, facecolors="white", edgecolors=ghost, linewidths=1.5, zorder=2)
        none, deg, d, q = rows[(a, "degree")]
        color = flat if q >= .05 else gain if d > 0 else loss
        ax.annotate("", xy=(deg, y), xytext=(none, y), zorder=3,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.3", color=color, lw=2.4,
                                    shrinkA=0, shrinkB=0))
        ax.scatter([none], [y], s=46, color=INK, zorder=4)
        label = f"{d:+.1f}".replace("-", "−") + ("" if q < .05 else " (n.s.)")
        ax.text(max(none, deg) + 1.2, y, label, va="center", fontsize=9, color=INK,
                fontweight="bold" if q < .05 else "normal")
    ax.set_yticks(range(len(ARMS)), [NAMES[a] for a in reversed(ARMS)], fontsize=9, color=INK)
    ax.set_xlim(55, 106)
    ax.set_xticks([60, 70, 80, 90, 100], ["60%", "70%", "80%", "90%", "100%"])
    ax.set_xlabel("Node-degree accuracy (correct share of all responses)", fontsize=8.5, color=INK)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.grid(axis="x", color="#ecebe7", lw=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", color=INK, lw=0, markersize=6, label="no primer"),
               plt.Line2D([], [], color=gain, lw=2.4, marker=">", markersize=6,
                          label="+ degree primer (states the answer)"),
               plt.Line2D([], [], color=ghost, lw=2, marker="o", markerfacecolor="white", markersize=5,
                          label="+ clustering primer (states none)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, 1.2), ncol=3, frameon=False,
              fontsize=8.5, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout()
    fig.savefig(OUT / "primer_effect.png", dpi=200, bbox_inches="tight", pad_inches=0.08, facecolor="white")


def primer_usage():
    """Four cards: the seven conditions and the graph feed the prompt, which goes to the model."""
    g = graphqa.canonical(nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]))
    q = 2  # the queried node
    primer = {c: primers.build_primer(g, c, k_min=2, k_max=3) for c in primers.CONDITIONS}
    encoding = graph_text_encoders.encode_graph(g, "incident").strip().splitlines()

    def about_q(condition):
        """The primer's sentence about node q (the whole primer when it has one sentence)."""
        for s in primer[condition].split(". "):
            if s.startswith(f"Node {q} "):
                return s.rstrip(".") + "."
        return primer[condition]

    w_in, h_in = 12.0, 5.6
    unit = h_in * 72 / 56                      # points per data unit on both axes

    def lh(size):                              # one line at linespacing 1.3, in data units
        return size * 1.3 / unit

    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def box(x, top, w, h, fc, ec, lw=1.2, r=1.0):
        ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw, zorder=0 if fc == CARD else 1))

    def text(x, y, s, size=7.6, **kw):
        ax.text(x, y, s, fontsize=size, va="top", zorder=5, **{"color": INK, "linespacing": 1.3, **kw})

    def card(x, top, w, h, title):
        box(x, top, w, h, CARD, LINE, lw=1.0, r=1.6)
        text(x + 1.5, top - 1.3, title, size=9.5, fontweight="bold")

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>,head_length=6,head_width=4", color=INK, lw=1.5,
                                     zorder=4))

    # Top left: the seven conditions, each with its sentence about node q.
    cx, cw, ctop, ch = 1, 42, 54.5, 31
    card(cx, ctop, cw, ch, "The 7 primer conditions")
    text(cx + 1.5, ctop - 3.8, f"each one's sentence about node {q}", color=MUTED)
    rows = [("none", "(no primer)"), ("components", about_q("components")), ("degree", about_q("degree")),
            ("clustering", about_q("clustering")), ("rwse", about_q("rwse")),
            ("all", "degree, clustering and RWSE in one sentence"), ("filler", about_q("filler"))]
    y, degree_y = ctop - 7.2, None
    for name, s in rows:
        star = name in ("degree", "all")
        lines = textwrap.wrap(s, 44)
        h = lh(7.2) * len(lines)
        if name == "degree":
            box(cx + 0.8, y + 0.6, cw - 1.6, h + 1.1, ACCENT_BG, ACCENT_BG, lw=0, r=0.6)
            degree_y = y - h / 2 + 0.05
        text(cx + 1.5, y, name + (" ★" if star else ""), size=8, fontweight="bold",
             color=ACCENT if star else INK)
        text(cx + 12.5, y + 0.05, "\n".join(lines), size=7.2, family=MONO, color=INK if star else MUTED)
        y -= h + 1.45
    text(cx + 1.5, ctop - ch + 2.4, "★ states the answer to a node-degree question", size=7.2, color=MUTED)

    # Bottom left: the graph.
    gx, gw, gtop, gh = 1, 42, 22, 20.5
    card(gx, gtop, gw, gh, "The graph")
    pos = {0: (9, 16.2), 1: (9, 7.4), 2: (16.5, 11.8), 3: (24, 11.8), 4: (31.5, 11.8)}
    for u, v in g.edges():
        ax.plot(*zip(pos[u], pos[v]), color=MUTED, lw=1.6, zorder=2)
    for n, (x, yy) in pos.items():
        queried = n == q
        ax.scatter([x], [yy], s=420, color=ACCENT if queried else "white",
                   edgecolors=ACCENT if queried else MUTED, linewidths=1.6, zorder=3)
        ax.text(x, yy, str(n), ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white" if queried else INK, zorder=4)
    text(gx + 1.5, 4.3, f"node {q} (filled) is the one the question asks about", size=7.4, color=MUTED)

    # Middle: the prompt the model reads = primer + encoding + question.
    px, pw, ptop, ph = 50, 45, 50, 37
    card(px, ptop, pw, ph, "The prompt the model reads")
    x, w, top = px + 1.5, pw - 3, ptop - 4.6
    mids = {}
    for kind, head, lines, fc, ec, hc in (
            ("primer", "PRIMER  ·  one of the 7 conditions (here: degree)",
             textwrap.wrap(primer["degree"], 52), ACCENT_BG, ACCENT, ACCENT),
            ("graph", "GRAPH  ·  the graph written as text (incident encoding)", encoding,
             GRAY_BG, LINE, MUTED),
            ("question", "QUESTION  ·  one of 6 tasks", [f"Q: What is the degree of node {q}?", "A:"],
             "white", LINE, MUTED)):
        h = 3.0 + lh(7.4) * len(lines) + 0.9
        box(x, top, w, h, fc, ec, lw=1.6 if kind == "primer" else 1.1)
        text(x + 1.1, top - 0.8, head, size=7.2, color=hc, fontweight="bold")
        text(x + 1.1, top - 2.8, "\n".join(lines), size=7.4, family=MONO)
        mids[kind] = top - h / 2
        top -= h + 1.2
    text(px, ptop - ph - 1.4, "Same graph and question under all 7 conditions,\nso every comparison is paired.",
         color=MUTED)

    # Right: the model and its answer.
    mx, mw = 101, 17
    card(mx, 40, mw, 26, "The model")
    box(mx + 2, 34.5, mw - 4, 8.5, "white", INK, lw=1.3)
    ax.text(mx + mw / 2, 30.25, "Qwen3\n1.7B · 4B\n± thinking", ha="center", va="center", fontsize=8.3,
            linespacing=1.3, zorder=5)
    arrow((mx + mw / 2, 25.6), (mx + mw / 2, 21.4))
    ax.text(mx + mw / 2, 19.3, "“3”", ha="center", va="center", fontsize=13, fontweight="bold", zorder=5)
    text(mx + mw / 2, 17.2, "scored against\nthe true answer", size=7.2, color=MUTED, ha="center")

    # How the cards connect.
    arrow((cx + cw + 0.3, degree_y), (x - 0.3, mids["primer"]))
    arrow((gx + gw + 0.3, 12.5), (x - 0.3, mids["graph"] - 3))
    arrow((px + pw + 0.3, 30.25), (mx + 1.7, 30.25))
    fig.savefig(OUT / "primer_usage.png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.12)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    primer_effect(main_rows(REPORT.read_text(encoding="utf-8")))
    primer_usage()
    print(f"wrote {OUT / 'primer_effect.png'} and {OUT / 'primer_usage.png'}")
