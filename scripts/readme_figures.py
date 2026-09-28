"""The two figures on the repo's front page, written to docs/img/.

  primer_effect.png  node_degree without a primer and with the degree primer
                     (states the answer) or clustering (states none), per arm;
                     every value is read from the [main] block of
                     outputs/n40-sweep/primer_findings.txt.
  primer_usage.png   how a primer is used: the seven conditions, each in its own
                     colour, and the prompt one opens; primer_usage.pdf is the
                     one-column version for the paper. Every sentence is
                     rendered by graphtalk/primers.py.

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


# One colour per condition; `all` shows the three statistics it combines.
COND_COLORS = {"none": "#8a8f98", "components": "#8e5cd9", "degree": "#2a78d6", "clustering": "#1a9c73",
               "rwse": "#e8812d", "all": "#2b2d42", "filler": "#d4587e"}
BLOCKS = {"primer": ("#e9f2fc", "#2a78d6"), "graph": ("#f1f0ec", "#9a958a"), "question": ("#fff4dc", "#d99a1e")}


def primer_usage():
    """The seven conditions, each in its own colour, and the prompt one of them opens (primer, the graph
    as text, the question): a wide PNG for the front page and a one-column PDF for the paper."""
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

    rows = [("none", "(no primer)"), ("components", about_q("components")), ("degree", about_q("degree")),
            ("clustering", about_q("clustering")), ("rwse", about_q("rwse")),
            ("all", "degree, clustering and RWSE in one sentence"), ("filler", about_q("filler"))]
    prompt = [("primer", "PRIMER · one of the seven (here: degree)", primer["degree"]),
              ("graph", "GRAPH · incident encoding", encoding),
              ("question", "QUESTION · one of six tasks", [f"Q: What is the degree of node {q}?", "A:"])]
    _usage_figure(True, q, rows, prompt).savefig(OUT / "primer_usage.png", dpi=200, facecolor="white",
                                                 bbox_inches="tight", pad_inches=0.06)
    _usage_figure(False, q, rows, prompt).savefig(OUT / "primer_usage.pdf", facecolor="white",
                                                  bbox_inches="tight", pad_inches=0.02)


def _usage_figure(wide, q, rows, prompt):
    """Draw in points: the conditions card and the prompt card, side by side (wide) or stacked."""
    s = 1.0 if wide else 0.88                        # type scale
    title, body, mono, small = 9.5 * s, 7.0 * s, 6.8 * s, 6.4 * s
    lh = lambda size: size * 1.38                    # one line, in points
    pad, gap = 9 * s, 12 * s
    card_w = 330 if wide else 210
    pill_w = 62 * s
    wrap_row = 50 if wide else 35
    wrap_prompt = 70 if wide else 48

    # `all` loses room to its three colour squares.
    row_lines = [textwrap.wrap(t, wrap_row - (7 if n == "all" else 0)) for n, t in rows]
    row_h = [max(lh(body) * 1.25, lh(mono) * len(ls)) for ls in row_lines]
    cond_h = pad + lh(title) + lh(small) + 4 + sum(h + 4 for h in row_h) + lh(small) + pad
    block_lines = [(k, head, textwrap.wrap(t, wrap_prompt) if isinstance(t, str) else t) for k, head, t in prompt]
    block_h = [lh(small) + 0.92 * lh(mono) * len(ls) + 10 * s for _, _, ls in block_lines]
    prompt_h = pad + lh(title) + 4 + sum(h + 6 for h in block_h) + lh(small) * 2 + pad
    W = 2 * card_w + 40 if wide else card_w
    H = max(cond_h, prompt_h) if wide else cond_h + gap + 14 + prompt_h

    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def rect(x, top, w, h, fc, ec="none", lw=0.0, r=4.0, z=1):
        ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw, zorder=z))

    def text(x, y, t, size, **kw):
        ax.text(x, y, t, fontsize=size, va="top", zorder=5, **{"color": INK, "linespacing": 1.38, **kw})

    # The conditions card.
    cx, ctop = 0, H
    rect(cx, ctop, card_w, cond_h, CARD, LINE, 0.8, 7)
    text(cx + pad, ctop - pad, "The seven primer conditions", title, fontweight="bold")
    text(cx + pad, ctop - pad - lh(title), f"each one's sentence about node {q}", small, color=MUTED)
    y = ctop - pad - lh(title) - lh(small) - 4
    degree_mid = None
    for (name, _), ls, h in zip(rows, row_lines, row_h):
        star = name in ("degree", "all")
        pill_h = lh(body) * 1.2
        rect(cx + pad, y, pill_w, pill_h, COND_COLORS[name], r=pill_h / 2, z=2)
        ax.text(cx + pad + pill_w / 2, y - pill_h / 2, name + (" ★" if star else ""), fontsize=body,
                ha="center", va="center", color="white", fontweight="bold", zorder=5)
        tx = cx + pad + pill_w + 8 * s
        if name == "all":                            # the three statistics it combines
            for i, c in enumerate(("degree", "clustering", "rwse")):
                rect(tx + i * 7 * s, y - 2.2 * s, 5.5 * s, 5.5 * s, COND_COLORS[c], r=1.2, z=2)
            tx += 24 * s
        text(tx, y - 1.2 * s, "\n".join(ls), mono, family=MONO, color=INK if name != "none" else MUTED)
        if name == "degree":
            degree_mid = y - pill_h / 2
        y -= h + 4
    text(cx + pad, ctop - cond_h + pad + lh(small) - 1, "★ states the answer to a node-degree question",
         small, color=MUTED)

    # The prompt card.
    px, ptop = (card_w + 40, H) if wide else (0, H - cond_h - gap - 14)
    rect(px, ptop, card_w, prompt_h, CARD, LINE, 0.8, 7)
    text(px + pad, ptop - pad, "The prompt the model reads", title, fontweight="bold")
    y, primer_mid = ptop - pad - lh(title) - 4, None
    for (kind, head, ls), h in zip(block_lines, block_h):
        fill, edge = BLOCKS[kind]
        rect(px + pad, y, card_w - 2 * pad, h, fill, r=4, z=2)
        rect(px + pad, y, 3.2 * s, h, edge, r=1.5, z=3)
        text(px + pad + 8 * s, y - 4 * s, head, small, color=edge, fontweight="bold")
        text(px + pad + 8 * s, y - 4 * s - lh(small), "\n".join(ls), mono, family=MONO)
        if kind == "primer":
            primer_mid = y - h / 2
        y -= h + 6
    text(px + pad, ptop - prompt_h + pad + 2 * lh(small) - 1,
         "The same graph and question under all seven conditions,\nso every comparison is paired.",
         small, color=MUTED)

    # One condition opens the prompt.
    head = "-|>,head_length=5,head_width=3"
    if wide:
        ax.add_patch(FancyArrowPatch((card_w + 3, degree_mid), (px + pad - 3, primer_mid), arrowstyle=head,
                                     color=COND_COLORS["degree"], lw=1.6, zorder=6))
    else:
        mid = card_w / 2
        ax.add_patch(FancyArrowPatch((mid, H - cond_h - 2), (mid, ptop + 2), arrowstyle=head,
                                     color=COND_COLORS["degree"], lw=1.6, zorder=6))
    return fig


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    primer_effect(main_rows(REPORT.read_text(encoding="utf-8")))
    primer_usage()
    print(f"wrote {OUT / 'primer_effect.png'} and {OUT / 'primer_usage.png'}")
