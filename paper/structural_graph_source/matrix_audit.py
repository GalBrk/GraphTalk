"""Generate the paper's complete paired main-sweep exact-correctness matrix.

Usage: python matrix_audit.py --frame PATH --report PATH --output main_matrix.tex
The report is the tagged [main] output on GraphTalk/results-sot. We check every
displayed cell against its recorded baseline/effect/q before writing LaTeX.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"]
TASKS = ["node_degree", "connected_nodes", "edge_count", "edge_existence", "node_count", "cycle_check"]
CONDITIONS = ["degree", "clustering", "rwse", "all", "components", "filler"]
ARM_LABELS = dict(zip(ARMS, ["1.7P", "1.7T", "4P", "4T"]))
TASK_LABELS = ["Degree", "Neighbors", "Edge count", "Edge exists", "Node count", "Cycle"]


def parse_tagged_main(path, control="none"):
    """[main] rows against `control`: (arm, task, condition) -> (before, after,
    delta, q, truncated change, flagged), as printed."""
    report = Path(path).read_text()
    section = report.split("[main] ", 1)[1].split("\n[", 1)[0]
    header = re.compile(r"^  (qwen3-[\w.-]+) (\w+) vs (none|filler):$")
    row = re.compile(r"^    (\w+)\s+([\d.]+) -> ([\d.]+): ([+-][\d.]+) .*? q=([\deE.+-]+) "
                     r".*?truncated ([+-][\d.]+)")
    pairs = {}
    current = None
    for line in section.splitlines():
        h = header.match(line)
        if h:
            arm, task, ctl = h.groups()
            current = (arm, task) if ctl == control else None
        m = row.match(line)
        if m and current is not None:
            cond, *nums = m.groups()
            pairs[(current[0], current[1], cond)] = (*map(float, nums), "FLAGGED" in line)
    assert len(pairs) == 4 * 6 * (6 if control == "none" else 5), len(pairs)
    return pairs


def cell(frame, tagged, arm, task, cond):
    rows = frame[(frame.arm == arm) & (frame.task == task) & (frame.density_class <= .50)]
    subsets = {condition: rows[rows.condition == condition].set_index("instance_id").sort_index()
               for condition in ("none", cond)}
    a, b = subsets["none"], subsets[cond]
    assert len(a) == len(b) == 400 and a.index.is_unique and a.index.equals(b.index)
    assert (a.gold == b.gold).all()
    base = 100 * (a.exact * (1 - a.hit_cap)).mean()
    primed = 100 * (b.exact * (1 - b.hit_cap)).mean()
    eff = primed - base
    old, new, reported, q, _, _ = tagged[(arm, task, cond)]
    assert abs(base - old) < .006 and abs(primed - new) < .006
    assert abs(eff - reported) <= .051, (arm, task, cond, eff, reported)
    flagged = max(a.hit_cap.mean(), b.hit_cap.mean()) >= .15
    # The checks above tie the frame to the tag; the table prints the tag's values.
    return old, reported, q, flagged


def latex(frame, report):
    tagged = parse_tagged_main(report)
    lines = [
        r"\begin{table*}[t]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Task & Arm & None & Degree & Cluster. & RWSE & All & Comp. & Filler\\",
        r"\midrule",
    ]
    for idx, (task, label) in enumerate(zip(TASKS, TASK_LABELS)):
        if idx == 4:
            lines.append(r"\midrule")
        for arm in ARMS:
            data = {c: cell(frame, tagged, arm, task, c) for c in CONDITIONS}
            baseline = data[CONDITIONS[0]][0]
            assert all(abs(v[0] - baseline) < 1e-9 for v in data.values())
            values = []
            for cond in CONDITIONS:
                _, effect, q, flagged = data[cond]
                num = f"{effect:+.1f}"
                shown = "$" + (rf"\mathbf{{{num}}}" if q < .05 else num) \
                    + (r"^{\dagger}" if flagged else "") + "$"
                # Shaded where the primer states the answer: [bars] scores 100.0.
                if task in ("node_degree", "edge_count") and cond in ("degree", "all"):
                    shown = r"\cellcolor{black!12}" + shown
                values.append(shown)
            task_cell = label if arm == ARMS[0] else ""
            lines.append(f"{task_cell} & {ARM_LABELS[arm]} & {baseline:.2f} & "
                         + " & ".join(values) + r"\\ % [main] [bars]")
        if idx not in (3, len(TASKS) - 1):
            lines.append(r"\addlinespace[2pt]")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Main-sweep correct shares: no-primer baseline (\% of all prompts) and paired change (points) on the same $40$-node graphs, $p\leq.50$. Bold: $q<.05$ within (model, task, control); $^{\dagger}$: at least $15\%$ truncation on either side; shaded: the primer states the answer. Neighbor sets: exact set match (set-F1 in Table~\ref{tab:primarymetrics}(a)); edge existence: raw correctness (balanced accuracy in Table~\ref{tab:primarymetrics}(b)). Node count and cycle check have a constant answer at $40$ nodes; truncation shares are in Table~\ref{tab:truncmatrix}. P/T: plain/thinking.} % [setup] [main] [bars]",
        r"\label{tab:matrix}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(latex(pd.read_csv(args.frame), args.report))
