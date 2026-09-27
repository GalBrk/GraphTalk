"""Generate the paper's Results floats from the repo's tagged outputs.

Usage: python floats.py --repo /path/to/GraphTalk

Writes, next to this file:
  table_effects.tex  Table 1: every primer effect with q < .05 on the four tasks
                     whose answer varies ([main] vs none and vs filler, [bars],
                     [eemain]); cells at or above the truncation flag go in the caption.
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

from matrix_audit import ARMS, ARM_LABELS, parse_tagged_main

HERE = Path(__file__).resolve().parent
TASKS = {"node_degree": "Degree", "connected_nodes": "Neighbors",
         "edge_count": "Edge count", "edge_existence": "Edge exists"}
PRIMERS = ["degree", "all", "clustering", "rwse", "components"]
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


def qfmt(q):
    return "$<$.001" if q < .001 else f"{q:.2g}".lstrip("0")


def solver_bars(report):
    bars = {}
    for line in block(report, "bars").splitlines():
        m = re.match(r"^  (\w+): (.*)$", line)
        if m and m.group(1) in TASKS:
            for cond, val in re.findall(r"(\w+) ([\d.]+)", m.group(2)):
                bars[(m.group(1), cond)] = float(val)
    return bars


def discloses(bars, task, primer):
    if bars[(task, primer)] >= 85:
        return "answer"
    if task == "connected_nodes" and primer in ("degree", "all"):
        return "its size"
    if bars[(task, primer)] - bars[(task, "none")] >= 5:
        return "part"
    return "--"


def balanced(report):
    """[eemain]: arm -> primer -> (change, q)."""
    out, arm = {}, None
    for line in block(report, "eemain").splitlines():
        m = re.match(r"^  (qwen3-[\w.-]+)\s", line)
        if m:
            arm = m.group(1)
        m = re.match(r"^    BA (\w+)\s+vs none ([+-][\d.]+) .*? q=(\S+)", line)
        if m:
            out.setdefault(arm, {})[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def table_effects(report_path, report):
    none, filler = parse_tagged_main(report_path), parse_tagged_main(report_path, "filler")
    bars, ba = solver_bars(report), balanced(report)
    rows, flagged = [], []
    for task, tlabel in TASKS.items():
        for arm in ARMS:
            for c in PRIMERS:
                before, after, d, q, dt, flag = none[(arm, task, c)]
                if q >= .05:
                    continue
                if flag:
                    flagged.append(f"{ARM_LABELS[arm]} {tlabel.lower()} {LABEL[c]} ${d:+.1f}$ "
                                   f"(truncated ${dt:+.1f}$)")
                    continue
                fd, fq = filler[(arm, task, c)][2:4]
                bal = ""
                if task == "edge_existence":
                    bd, bq = ba[arm][c]
                    bal = f"${bd:+.1f}$" + ("$^{*}$" if bq < .05 else "")
                rows.append(f"{tlabel} & {ARM_LABELS[arm]} & {LABEL[c]} & {discloses(bars, task, c)} & "
                            f"{bars[(task, c)]:.1f} & {before:.2f}$\\to${after:.2f} & ${d:+.1f}$ & "
                            f"{qfmt(q)} & ${dt:+.1f}$ & ${fd:+.1f}$" + ("$^{*}$" if fq < .05 else "")
                            + f" & {bal}\\\\ % [main] [bars]" + (" [eemain]" if bal else ""))
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lllcrrrrrrr}", r"\toprule",
             r"Task & Arm & Primer & States & Solver & None$\to$primer & $\Delta$ & $q$ & "
             r"$\Delta$trunc. & $\Delta$ vs filler & $\Delta$BA\\", r"\midrule", *rows,
             r"\bottomrule", r"\end{tabular}",
             r"\caption{Every primer effect with $q<.05$ on the four tasks whose answer varies "
             r"(main sweep, $p\leq.50$, 400 paired graphs per row). Correct shares are percent "
             r"of all responses; $\Delta$ is in points. \emph{States}: what the primer states "
             r"about the answer, from the graph-blind solver (\emph{Solver}, percent correct from "
             r"the primer text alone). $\Delta$ vs filler compares the primer with the "
             r"structure-free filler on the same graphs; $\Delta$BA is the change in balanced "
             r"accuracy. $^{*}$: $q<.05$. P/T: plain/thinking. At or above 15\% truncation, "
             r"not shown: " + "; ".join(flagged) + r".} % [main] [bars] [eemain]",
             r"\label{tab:effects}", r"\end{table*}"]
    (HERE / "table_effects.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


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
    ap.add_argument("--repo", required=True)
    repo = Path(ap.parse_args().repo)
    report_path = repo / "csv2/raw-trends/primer_findings.txt"
    report = report_path.read_text(encoding="utf-8")
    table_effects(report_path, report)
    fig_headroom(repo / "csv2/raw-trends/primer_cells.csv")
    fig_clustering(report, (repo / "csv2/density-followups/density_followups.txt").read_text(encoding="utf-8"))
    table_cycles((repo / "csv2/raw-trends/check_cycle_claims.txt").read_text(encoding="utf-8"))
