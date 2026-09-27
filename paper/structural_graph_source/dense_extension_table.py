"""Audit the complete 40-node dense-extension primer comparisons.

Usage: python dense_extension_table.py --frame PATH --output dense_extension.tex
"""

import argparse
from pathlib import Path

import pandas as pd

from matrix_audit import ARMS, ARM_LABELS, CONDITIONS


def signed(x):
    """One decimal with a sign; a change that rounds to zero prints as +0.0."""
    s = f"{x:+.1f}"
    return "+0.0" if s == "-0.0" else s


def dense_rows(frame, task, arm):
    f = frame[(frame.task == task) & (frame.arm == arm)
              & (frame.density_class >= .65)]
    matched = {}
    for condition in ("none", *CONDITIONS):
        rows = f[f.condition == condition].set_index("instance_id").sort_index()
        assert len(rows) == 300 and rows.index.is_unique, (task, arm, condition)
        matched[condition] = rows
    control = matched["none"]
    assert all(r.index.equals(control.index) and (r.gold == control.gold).all()
               for r in matched.values())
    return matched


def panel(frame, measure, title):
    lines = [rf"\textbf{{{title}}}\par\smallskip",
             r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Task & Arm & None & Degree & Cluster. & RWSE & All & Comp. & Filler\\",
             r"\midrule"]
    for task, title in (("node_degree", "Degree"), ("edge_existence", "Edge exists")):
        for arm in ARMS:
            matched = dense_rows(frame, task, arm)
            values = {condition: 100 * measure(rows) for condition, rows in matched.items()}
            changes = " & ".join(f"${signed(values[c] - values['none'])}$" for c in CONDITIONS)
            lines.append(f"{title if arm == ARMS[0] else ''} & {ARM_LABELS[arm]} & "
                         f"{values['none']:.2f} & {changes}" + r"\\ % [dense-frame]")
        if task == "node_degree":
            lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def balanced(rows):
    """Balanced accuracy; a truncated response is not correct."""
    good = rows.hit_cap.eq(0) & rows.exact.eq(1)
    gold_yes = rows.gold_is_yes.astype(bool)
    return (good[gold_yes].mean() + good[~gold_yes].mean()) / 2


def yes_rate(rows):
    """Share of finished answers that say yes."""
    finished = rows[rows.hit_cap.eq(0)]
    assert len(finished) > 0
    return finished.pred.astype(str).str.strip().str.lower().eq("yes").mean()


def edge_panel(frame):
    """Edge existence in the layout of (a) and (b): each measure is averaged over
    the three levels, as [collapse] averages its per-level changes."""
    lines = [r"\textbf{(c) Edge-existence decisions}\par\smallskip",
             r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Measure & Arm & None & Degree & Cluster. & RWSE & All & Comp. & Filler\\",
             r"\midrule"]
    for measure, title in ((balanced, "Balanced acc."), (yes_rate, "Yes-rate")):
        for arm in ARMS:
            matched = dense_rows(frame, "edge_existence", arm)
            values = {c: 100 * sum(measure(g) for _, g in rows.groupby("density_class")) / 3
                      for c, rows in matched.items()}
            changes = " & ".join(f"${signed(values[c] - values['none'])}$" for c in CONDITIONS)
            lines.append(f"{title if arm == ARMS[0] else ''} & {ARM_LABELS[arm]} & "
                         f"{values['none']:.2f} & {changes}" + r"\\ % [dense-frame]")
        if measure is balanced:
            lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def latex(frame):
    lines = [r"\begin{table*}[t]", r"\centering\scriptsize",
             r"\setlength{\tabcolsep}{3.4pt}"]
    lines += panel(frame, lambda r: (r.exact * (1-r.hit_cap)).mean(),
                   "(a) Correct share")
    lines += [r"\par\medskip"]
    lines += panel(frame, lambda r: r.hit_cap.mean(), "(b) Truncated share")
    lines += [r"\par\medskip"]
    lines += edge_panel(frame)
    lines += [r"\caption{Complete dense-extension comparison at $p\in\{.65,.75,.85\}$ for degree and edge existence: no-primer value (\%) and paired change (points). (a,b) Correct and truncated share of all $300$ prompts; the wrong share is the remainder. (c) Balanced accuracy (a truncated response is not correct) and yes-rate of finished answers, each averaged over the three levels. The contrasts cited in the text are tested; the other cells are descriptive. The plain models' budget here is $2{,}048$ tokens. P/T: plain/thinking.} % [dense-frame]",
              r"\label{tab:densematrix}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(latex(pd.read_csv(args.frame)))
