"""Generate appendix tables from metric_audit.txt after metric_audit.py runs."""
import re
from pathlib import Path

root = Path(__file__).resolve().parent
lines = (root / "metric_audit.txt").read_text().splitlines()
arm_names = {"qwen3-1.7b": "1.7P", "qwen3-1.7b-think": "1.7T",
             "qwen3-4b": "4P", "qwen3-4b-think": "4T"}
f1_pattern = re.compile(
    r"^\s+(qwen3-[\w.-]+)\s+(\w+)\s+([0-9.]+)->([0-9.]+)"
    r"\s+delta=([+-][0-9.]+).* q=([0-9.]+)$")
edge_pattern = re.compile(
    r"^\s+(qwen3-[\w.-]+)\s+(\w+)\s+raw=([0-9.]+)"
    r"\s+BA=([0-9.]+)\s+yes=([0-9.]+)\s+truncated=([0-9.]+)$")
f1_rows = [f1_pattern.match(s).groups() for s in lines if f1_pattern.match(s)]
start = lines.index("[metric-audit-edge-main] Edge existence, main sweep, all model arms;")
edge_rows = [edge_pattern.match(s).groups() for s in lines[start:] if edge_pattern.match(s)]
assert len(f1_rows) == 24 and len(edge_rows) == 20

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
    if i and i % 5 == 0:
        out.append(r"\addlinespace[2pt]")
    out.append(
        f"{arm_names[arm]} & {primer} & {float(raw):.3f} & {float(ba):.3f}"
        f" & {float(yes):.3f} & {float(trunc):.3f}"
        + r"\\ % [metric-audit-edge-main]")
out += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{minipage}",
    r"\caption{Primary-metric audit on the same $400$ main-sweep prompts per "
    r"arm and condition. (a) Neighbor-set Set-F1, with truncations scored "
    r"zero: each row gives the no-primer baseline, the primer score and "
    r"paired difference; $q$ adjusts six contrasts per arm. Components and "
    r"filler are diagnostics. (b) Edge-existence raw and balanced accuracy "
    r"(Bal.) and truncation (Trunc.) use all prompts; yes-rate (Yes) uses "
    r"finished answers. Compare each primer row with its arm's none row. "
    r"Only raw-accuracy contrasts have BH tests in Table~\ref{tab:matrix}; "
    r"other edge measures are descriptive. P/T: plain/thinking.} "
    r"% [metric-audit-f1] [metric-audit-edge-main]",
    r"\label{tab:primarymetrics}",
    r"\end{table*}",
]
(root / "primary_metric_tables.tex").write_text("\n".join(out) + "\n")
print("Generated appendix tables:", len(f1_rows), "F1 and", len(edge_rows), "edge rows")
