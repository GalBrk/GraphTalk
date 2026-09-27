"""Generate the paper's Results floats from the repo's tagged outputs.

Usage: python paper/floats.py   (reads ../outputs; --repo points elsewhere)

Writes, next to this file:
  fig_headroom.pdf   Figure 1: node_degree effect against the no-primer correct
                     share, per arm and density, degree and all (primer_cells.csv,
                     which primer_findings.py --csv-dir writes).
  fig_clustering.pdf Figure 2: the clustering effect on node_degree, every arm on the same
                     graphs ([clustarms]), then the follow-ups ([replic], [fixdeg17],
                     [ddplainfill], [ddplain], [ddthink], [fixdeg8]).
  table_cycles.tex   Table 2: what the correct cycle_check answers rest on ([ccanswer]),
                     bold where [cctest] finds a change against no primer.
Every value is copied as printed; nothing is recomputed from responses.
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from matrix_audit import ARMS, ARM_LABELS

HERE = Path(__file__).resolve().parent
LABEL = {"degree": "degree", "all": "all", "clustering": "clustering", "rwse": "RWSE",
         "components": "components", "none": "none"}
INK, MUTED = "#0b0b0b", "#52514e"
# One color per arm (blue, violet, orange, aqua: validated all-pairs on the light surface);
# the shape repeats plain (circle) vs thinking (triangle).
COLOR = {"qwen3-1.7b": "#2a78d6", "qwen3-1.7b-think": "#4a3aa7",
         "qwen3-4b": "#eb6834", "qwen3-4b-think": "#1baf7a"}
MARK = {"qwen3-1.7b": "o", "qwen3-1.7b-think": "^", "qwen3-4b": "o", "qwen3-4b-think": "^"}
NUM = r"([+-]\d+\.\d) \[([+-]\d+\.\d), ([+-]\d+\.\d)\]"


def block(text, tag):
    """Every line of every block that opens with [tag] (as test_results_docs reads it)."""
    out, on = [], False
    for line in text.splitlines():
        if re.match(r"^\[[a-z0-9]+\]", line):
            on = line.startswith(f"[{tag}]")
        if on:
            out.append(line)
    return "\n".join(out)


def interval(text, tag, key):
    """(effect, lo, hi) from the first line of block [tag] that contains key."""
    line = next(l for l in block(text, tag).splitlines() if key in l)
    return tuple(map(float, re.search(NUM, line).groups()))


def fig_headroom(cells_path):
    t = pd.read_csv(cells_path)
    t = t[(t.task == "node_degree") & t.condition.isin(["degree", "all"])]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for ax, cond in zip(axes, ["degree", "all"]):
        ax.axhline(0, color=MUTED, lw=0.6)
        for arm in ARMS:
            s = t[(t.condition == cond) & (t.arm == arm)]
            # Hollow: flagged, or a plain arm's high-density run at the 2048-token budget.
            hollow = s.flagged | (~s.arm.str.endswith("think") & (s.density >= .65))
            for h in (False, True):
                x = s[hollow == h]
                ax.scatter(100 * x.baseline, x.delta, s=34, marker=MARK[arm],
                           facecolors="white" if h else COLOR[arm], edgecolors=COLOR[arm],
                           linewidths=1.3, label=None if h else ARM_LABELS[arm], zorder=3)
        ax.set_title({"degree": "degree primer", "all": "all three statistics"}[cond], fontsize=8, color=INK)
        ax.set_xlabel("No-primer correct share (%)", fontsize=7.5, color=INK)
        ax.tick_params(labelsize=7.5, colors=MUTED, length=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", lw=0.5)
    axes[0].set_ylabel("Effect on correct share (points)", fontsize=7.5, color=INK)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left", handletextpad=0.2)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(HERE / "fig_headroom.pdf", bbox_inches="tight", pad_inches=0.02)


def fig_clustering(report, followups):
    rows = [("Same graphs, all four models", None, None)]
    for arm in ARMS:
        for band, key, n in (("low density", "p<=.50", 400), ("high density", "p>=.65", 300)):
            rows.append((f"{ARM_LABELS[arm]} {band} ({n})",
                         interval(report, "clustarms", f"{arm:17s} {key}"), arm))
    rows += [
        ("More graphs (follow-ups)", None, None),
        ("1.7P new graphs (1,200)", interval(report, "replic", "the 300 new graphs"), "qwen3-1.7b"),
        ("1.7P fresh seed (1,600)", interval(report, "replic", "fresh seeds"), "qwen3-1.7b"),
        ("1.7P fixed mean degree (3,200)", interval(followups, "fixdeg17", "all levels clustering"), "qwen3-1.7b"),
        ("1.7P against filler (1,600)", interval(followups, "ddplainfill", "p<=.50 clustering"), "qwen3-1.7b"),
        ("1.7P high density (1,200)", interval(followups, "ddplain", "p>=.65 clustering"), "qwen3-1.7b"),
        ("1.7T low density (1,600)", interval(followups, "ddthink", "p<=.50 clustering"), "qwen3-1.7b-think"),
        ("1.7T high density (1,200)", interval(followups, "ddthink", "p>=.65 clustering"), "qwen3-1.7b-think"),
        ("Qwen3-8B fixed mean degree (3,200)", interval(followups, "fixdeg8", "all levels clustering"), None),
    ]
    fig, ax = plt.subplots(figsize=(3.0, 3.9))
    ax.axvline(0, color=MUTED, lw=0.6)
    for i, (label, v, arm) in enumerate(rows):
        if v is None:
            continue
        d, lo, hi = v
        c = COLOR.get(arm, INK)
        ax.plot([lo, hi], [i, i], color=c, lw=1.3, solid_capstyle="round")
        ax.plot([d], [i], MARK.get(arm, "s"), color=c, ms=4.5)
        ax.text(1.02, i, f"{d:+.1f}".replace("-", "−"), va="center", ha="left",
                fontsize=7, color=MUTED, transform=ax.get_yaxis_transform())
    ax.set_xlim(-16, 17)
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    for tick, (_, v, _) in zip(ax.get_yticklabels(), rows):
        if v is None:
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("Clustering effect on node degree (points)", fontsize=7, color=INK)
    ax.tick_params(axis="x", labelsize=7, colors=MUTED, length=2)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(HERE / "fig_clustering.pdf", bbox_inches="tight", pad_inches=0.02)


def cctest_stars(claims):
    """[cctest]: {(what, arm, primer)} whose change against none has BH q < .05."""
    out, what = set(), None
    for line in block(claims, "cctest").splitlines():
        if "rests on a real cycle" in line:
            what = "real"
        elif "rests on an invented cycle" in line:
            what = "invented"
        elif "asserts" in line:
            what = None
        m = re.match(r"^    (\S+)\s+(.*)$", line)
        if m and what:
            for cond, star in re.findall(r"(\w+) [\d.]+->[\d.]+ \([+-][\d.]+(\*?),", m.group(2)):
                if star:
                    out.add((what, m.group(1), cond))
    return out


def table_cycles(claims):
    stars = cctest_stars(claims)
    arm_label = {"1.7B": "1.7P", "1.7B-T": "1.7T", "4B": "4P", "4B-T": "4T"}
    rows, last = [], None
    for line in block(claims, "ccanswer").splitlines():
        m = re.match(r"^  (\S+)\s+(\w+)\s+n=(\d+)\s+([\d.]+) /\s+([\d.]+) /\s+([\d.]+) /\s+([\d.]+)"
                     r" \(\s*([\d.]+)\)", line)
        if not m or m.group(2) not in ("none", "degree", "rwse", "clustering", "all"):
            continue
        arm, cond, n, real, inv, notc, named, edgecount = m.groups()
        bold = lambda v, what: f"\\textbf{{{v}}}" if (what, arm, cond) in stars else v
        if last is not None and arm != last:
            rows.append(r"\addlinespace[2pt]")
        rows.append(f"{arm_label[arm] if arm != last else ''} & {LABEL[cond]} & {n} & "
                    f"{bold(real, 'real')} & {bold(inv, 'invented')} & {notc} & {named} & "
                    f"{edgecount}\\\\ % [ccanswer] [cctest]")
        last = arm
    lines = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{2.1pt}",
             r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & & Correct & Valid & \multicolumn{2}{c}{False} & \multicolumn{2}{c}{No cycle}\\",
             r"\cmidrule(lr){4-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
             r"Arm & Primer & ``yes'' & Real & Invented & Not a & All & Edge\\",
             r" & & ($n$) & cycle & cycle & cycle & & count\\", r"\midrule",
             *rows, r"\bottomrule", r"\end{tabular}",
             r"\caption{What the correct cycle-check answers rest on (main sweep; every correct "
             r"answer is ``yes''). \emph{Correct ($n$)}: count of correct, finished answers; the "
             r"other columns are \% of those $n$, and Real, Invented, Not a cycle and All sum to "
             r"100. \emph{Valid}: a cycle whose every step is an edge. \emph{False}: "
             r"\emph{Invented} has a step that is not an edge; \emph{Not a cycle} is a length-2 or "
             r"retraced walk. \emph{No cycle}: \emph{All} names none; \emph{Edge count} argues "
             r"from the number of edges (valid). Bold (Real, Invented only): differs from the same "
             r"model without a primer (paired, BH $q<.05$).} % [ccanswer] [cctest]",
             r"\label{tab:cycles}", r"\end{table}"]
    (HERE / "table_cycles.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(HERE.parent))
    repo = Path(ap.parse_args().repo)
    report = (repo / "outputs/n40-sweep/primer_findings.txt").read_text(encoding="utf-8")
    fig_headroom(repo / "outputs/n40-sweep/primer_cells.csv")
    fig_clustering(report, (repo / "outputs/density-followups/density_followups.txt").read_text(encoding="utf-8"))
    table_cycles((repo / "outputs/n40-sweep/check_cycle_claims.txt").read_text(encoding="utf-8"))
