"""Emit the all-cells effect-size table from ci_all.json.

  PYTHONPATH=. python paper/make_ci_table.py
"""
import json

ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"]
CONDS = ["components", "clustering", "rwse", "degree", "filler", "all"]
TASKS = ["connected_nodes", "cycle_check", "edge_count", "edge_existence",
         "node_count", "node_degree"]


def tex(s):
    return s.replace("_", r"\_")


def main():
    d = json.load(open("ci_all.json"))
    out = [
        "Paired effect sizes for every (arm, task, primer) cell: $\\Delta$ in",
        "percentage points against \\texttt{none}, a $95\\%$ paired-bootstrap",
        "interval, the permutation $p$-value, and $\\Delta$ under the alternative",
        "convention that scores truncated generations incorrect rather than",
        "dropping the pair (Section~\\ref{sec:extraction}). $n$ is the",
        "number of pairs in which neither arm truncated.",
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
                    "%s & %s & %d & $%+.1f$ & $[%+.1f, %+.1f]$ & %.4f & $%+.1f$ \\\\"
                    % (label, tex(cond), v["n_drop"], v["delta_drop"],
                       v["ci"][0], v["ci"][1], v["p_perm"], v["delta_zero"]))
        out += [
            r"\bottomrule", r"\end{tabular}}",
            r"\caption{\texttt{%s}: all cells.}" % tex(arm),
            r"\end{table}", "",
        ]

    with open("paper/ci_table.tex", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote paper/ci_table.tex")


if __name__ == "__main__":
    main()
