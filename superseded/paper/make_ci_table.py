"""Emit the all-cells effect-size table from superseded/ci_all.json.

  PYTHONPATH=. python superseded/paper/make_ci_table.py
"""
import json

ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"]
CONDS = ["components", "clustering", "rwse", "degree", "filler", "all"]
TASKS = ["connected_nodes", "cycle_check", "edge_count", "edge_existence",
         "node_count", "node_degree"]


def tex(s):
    return s.replace("_", r"\_")


def ci_cell(v):
    """The CI column, or an explicit "not estimable" marker.

    `significance.exact_paired_ci` conditions on the discordant pairs, so
    with none of them the interval is a point at zero. Printing that as
    `[+0.0, +0.0]` -- which this table did on seventeen cells -- reads as a
    confident null measured to the decimal, when it means nothing disagreed
    and there was no direction to bound. Say so instead; `n` and the
    permutation p-value in the neighbouring columns still carry the
    evidence that the cell is at ceiling.
    """
    lo, hi = v["ci"]
    if lo is None or hi is None or v.get("n_discordant", 0) == 0:
        return r"n/a ($%d$ disc.)" % v.get("n_discordant", 0)
    return "$[%+.1f, %+.1f]$" % (lo, hi)


def main():
    d = json.load(open("superseded/ci_all.json"))
    out = [
        "Paired effect sizes for every (arm, task, primer) cell: $\\Delta$ in",
        "percentage points against \\texttt{none}, an exact $95\\%$ interval",
        "conditional on the discordant pairs (Clopper--Pearson, inverting the",
        "same exact McNemar test used for the per-cell $p$-values), the",
        "permutation $p$-value, and $\\Delta$ under the alternative",
        "convention that scores truncated generations incorrect rather than",
        "dropping the pair (Section~\\ref{sec:extraction}). $n$ is the",
        "number of pairs in which neither arm truncated. ``n/a'' marks a cell",
        "in which no pair disagreed: the conditional estimate is then exactly",
        "zero with no width, which is a statement about this sample and not a",
        "bound on the effect, so the discordant count is reported instead.",
        "",
    ]
    for arm in ARMS:
        out += [
            r"\begin{table}[H]", r"\centering\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{llrrrrr}", r"\toprule",
            r"Task & Primer & $n$ & $\Delta$ & 95\% CI & perm $p$ & zeroed \\",
            r"\midrule",
        ]
        for ti, task in enumerate(TASKS):
            if ti:
                out.append(r"\midrule")
            for ci, cond in enumerate(CONDS):
                v = d.get(f"{arm}|{task}|{cond}")
                if v is None:
                    continue
                label = r"\texttt{%s}" % tex(task) if ci == 0 else ""
                out.append(
                    "%s & %s & %d & $%+.1f$ & %s & %.4f & $%+.1f$ \\\\"
                    % (label, tex(cond), v["n_drop"], v["delta_drop"],
                       ci_cell(v), v["p_perm"], v["delta_zero"]))
        out += [
            r"\bottomrule", r"\end{tabular}}",
            r"\caption{\texttt{%s}: all cells.}" % tex(arm),
            r"\end{table}", "",
        ]

    with open("superseded/paper/ci_table.tex", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote superseded/paper/ci_table.tex")


if __name__ == "__main__":
    main()
