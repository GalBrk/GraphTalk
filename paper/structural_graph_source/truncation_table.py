"""Regenerate the complete main-sweep truncation matrix from saved responses.

Usage: python truncation_table.py --frame PATH --output main_truncation.tex
The paired baseline and conditions are verified on the same instance IDs.
"""

import argparse
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

import pandas as pd

from matrix_audit import ARMS, ARM_LABELS, CONDITIONS, TASKS, TASK_LABELS


def f1(x):
    """One decimal, ties to even on the decimal value, as primer_findings prints [main]."""
    return f"{Decimal(repr(round(float(x), 9))).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN):+}"


def latex(frame):
    lines = [
        r"\begin{table*}[t]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Task & Arm & None & Degree & Cluster. & RWSE & All & Comp. & Filler\\",
        r"\midrule",
    ]
    for i, (task, task_label) in enumerate(zip(TASKS, TASK_LABELS)):
        for arm in ARMS:
            subset = frame[(frame.arm == arm) & (frame.task == task) & (frame.density_class <= .50)]
            matched = {}
            for condition in ("none", *CONDITIONS):
                rows = subset[subset.condition == condition].set_index("instance_id").sort_index()
                assert len(rows) == 400 and rows.index.is_unique, (task, arm, condition)
                matched[condition] = rows
            control = matched["none"]
            assert all(rows.index.equals(control.index) and (rows.gold == control.gold).all()
                       for rows in matched.values())
            base = 100 * control.hit_cap.mean()
            effects = [100 * (matched[condition].hit_cap.mean() - control.hit_cap.mean())
                       for condition in CONDITIONS]
            cells = " & ".join(f"${f1(effect)}$" for effect in effects)
            label = task_label if arm == ARMS[0] else ""
            lines.append(f"{label} & {ARM_LABELS[arm]} & {base:.2f} & {cells}"
                         + r"\\ % [main] [trunc]")
        if i == 3:
            lines.append(r"\midrule")
        elif i < len(TASKS) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Complete main-sweep truncated shares: no-primer baseline (percent of all prompts) and paired changes (percentage points), for the same queries as Table~\ref{tab:matrix}. Each of its cells' \emph{wrong} share equals $100\%$ minus its correct and truncated shares. P/T: plain/thinking; components and filler are diagnostics. No significance marks are assigned to truncation contrasts.} % [setup] [main]",
        r"\label{tab:truncmatrix}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(latex(pd.read_csv(args.frame)))
