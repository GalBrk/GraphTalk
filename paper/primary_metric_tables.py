"""Generate the appendix secondary-metrics table: (a) from metric_audit.txt after
metric_audit.py runs, (b) from the [eemain] block of the repo's primer_findings.txt."""
import re
from pathlib import Path

root = Path(__file__).resolve().parent
lines = (root / "metric_audit.txt").read_text().splitlines()
report = (root.parent / "outputs" / "n40-sweep" / "primer_findings.txt").read_text(encoding="utf-8")
arm_names = {"qwen3-1.7b": "1.7P", "qwen3-1.7b-think": "1.7T",
             "qwen3-4b": "4P", "qwen3-4b-think": "4T"}
f1_pattern = re.compile(
    r"^\s+(qwen3-[\w.-]+)\s+(\w+)\s+([0-9.]+)->([0-9.]+)"
    r"\s+delta=([+-][0-9.]+).* q=([0-9.]+)$")
edge_pattern = re.compile(
    r"^  (qwen3-[\w.-]+)\s+(\w+)\s+raw ([0-9.]+) BA ([0-9.]+) yes ([0-9.]+) truncated ([0-9.]+)$")
f1_rows = [f1_pattern.match(s).groups() for s in lines if f1_pattern.match(s)]
eemain = report.split("\n[eemain] ", 1)[1].split("\n[", 1)[0].splitlines()
edge_rows = [edge_pattern.match(s).groups() for s in eemain if edge_pattern.match(s)]
assert len(f1_rows) == 24 and len(edge_rows) == 28

out = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\begin{minipage}[t]{.48\textwidth}",
    r"\centering\footnotesize\setlength{\tabcolsep}{2.1pt}",
    r"\textbf{(a) Neighbor-set Set-F1}\par\smallskip",
    r"\begin{tabular}{llrrr}",
    r"\toprule",
    r"Arm & Primer & None & With & $\Delta$; $q$\\",
    r"\midrule",
]
for i, (arm, primer, before, after, delta, q) in enumerate(f1_rows):
    if i and i % 6 == 0:
        out.append(r"\addlinespace[2pt]")
    out.append(
        f"{arm_names[arm]} & {primer} & {float(before):.4f} & {float(after):.4f}"
        + " & $" + f"{float(delta):+.4f}" + "$; " + f"{float(q):.4f}"
        + r"\\ % [metric-audit-f1]")
out += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{minipage}\hfill",
    r"\begin{minipage}[t]{.48\textwidth}",
    r"\centering\footnotesize\setlength{\tabcolsep}{2.0pt}",
    r"\textbf{(b) Edge-existence decisions}\par\smallskip",
    r"\begin{tabular}{llrrrr}",
    r"\toprule",
    r"Arm & Primer & Raw & Bal. & Yes & Trunc.\\",
    r"\midrule",
]
for i, (arm, primer, raw, ba, yes, trunc) in enumerate(edge_rows):
    if i and i % 7 == 0:
        out.append(r"\addlinespace[2pt]")
    out.append(
        f"{arm_names[arm]} & {primer} & {raw} & {ba} & {yes} & {trunc}"
        + r"\\ % [eemain]")
out += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{minipage}",
    r"\caption{Secondary metrics on the same $400$ main-sweep prompts per "
    r"model and condition. (a) Neighbor-set set-F1, with truncations scored "
    r"zero: each row gives the no-primer baseline, the primer score and "
    r"paired difference; $q$ adjusts six contrasts per model (paired "
    r"random-sign tests). (b) Edge-existence raw and balanced accuracy "
    r"(Bal., \%) and truncated share (Trunc.) use all prompts; yes-rate (Yes) uses "
    r"finished answers. Compare each row with its model's none row; "
    r"changes in balanced accuracy are tested in the text (\S\ref{sec:results}). "
    r"P/T: plain/thinking.} "
    r"% [metric-audit-f1] [eemain]",
    r"\label{tab:primarymetrics}",
    r"\end{table*}",
]
(root / "primary_metric_tables.tex").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
print("Generated appendix tables:", len(f1_rows), "F1 and", len(edge_rows), "edge rows")
