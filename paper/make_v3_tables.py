"""Emit the five float files the v3 additions need, into paper/.

Writes new files only; nothing v2 inputs is touched. Every output is prefixed
`v3_` -- note that paper/degfixdeg_table.tex and paper/short/degfixdeg_table.tex
are already two different files from two different generators, and this script
must not add a third collision.

  tab:vsfiller    delta against the length control, 6 tasks x 6 conditions (body)
  tab:additivity  `all` against the sum of its parts (body)
  tab:pertask     per-task accuracy and both deltas, all four arms (appendix)
  tab:behaviour   response length, strategy markers, truncation (appendix)
  tab:relevance   matched vs mismatched primer/task pairs (appendix)

Reads csv2/raw-trends/ only. frame.csv is the per-response frame that
scripts/build_raw_frame.py rebuilds from runs/ (and whose golds it verifies
against the stored ones); the aggregate CSVs come from scripts/raw_trends.py.
Pairing here mirrors paper/make_main_table.py: pair on the shared graph, drop a
pair if either side hit the generation budget, exact match, exact McNemar,
Benjamini-Hochberg within each (arm, task) family.

  PYTHONPATH=. python paper/make_v3_tables.py
"""
import numpy as np
import pandas as pd

from graphtalk import scoring

OUT = "paper/"
SRC = "csv2/raw-trends/"
ARMS = ["qwen3-1.7b", "qwen3-4b", "qwen3-1.7b-think", "qwen3-4b-think"]
SHORT = {"qwen3-1.7b": "1.7b", "qwen3-4b": "4b",
         "qwen3-1.7b-think": "1.7b-t", "qwen3-4b-think": "4b-t"}
CONDS = ["components", "clustering", "rwse", "degree", "all"]
HEADS = {"components": "comp.", "clustering": "clust.", "rwse": "rwse",
         "degree": "degree", "all": "all", "filler": "filler", "none": "none"}
# The four densities every task was run at. node_degree and edge_existence also
# have .65/.75/.85; pooling those in would make the columns mean different
# things task to task, so the pooled tables stay on the shared four.
DENS4 = [0.10, 0.20, 0.35, 0.50]
TASKS = ["node_count", "cycle_check", "edge_existence", "node_degree",
         "connected_nodes", "edge_count"]
# Gold is the same for every graph at n=40, so a shift on these need not be
# graph reading. Flagged in the tables rather than dropped.
CONST_GOLD = {"node_count", "cycle_check"}


def tex(s):
    return str(s).replace("_", r"\_")


def write(name, lines):
    with open(OUT + name, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote " + OUT + name)


def bh(pvals):
    """Benjamini-Hochberg q-values, order preserved. NaN passes through."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return q
    pv, n = p[ok], int(ok.sum())
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p down, so a small p can never get a
    # larger q than a bigger one.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def load_frame():
    f = pd.read_csv(SRC + "frame.csv")
    return f[f.density_class.isin(DENS4)]


def paired(frame, arm, task, baseline, cond):
    """Pooled paired contrast on the shared graphs. Returns (delta_pts, p, n)."""
    d = frame[(frame.arm == arm) & (frame.task == task)]
    a = d[d.condition == baseline].set_index("graph_id")
    b = d[d.condition == cond].set_index("graph_id")
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[(j.hit_cap_a == 0) & (j.hit_cap_b == 0)]
    if len(j) < 10:
        return np.nan, np.nan, len(j)
    m = scoring.mcnemar(j.exact_a.astype(bool), j.exact_b.astype(bool))
    delta = 100.0 * (j.exact_b.mean() - j.exact_a.mean())
    return delta, m["p_value"], len(j)


def cell(delta, q):
    if np.isnan(delta):
        return "--"
    body = f"{delta:+.1f}"
    if not np.isnan(q) and q < 0.05:
        return r"$\mathbf{" + body + "}$"
    return f"${body}$"


# ------------------------------------------------------------------ tab:main
def main_table(frame):
    """Both comparisons in one float: every delta against `none` and `filler`.

    Against `none` alone, content is confounded with the cost of the added
    text; against `filler` alone, `filler`'s own (large, task-specific) cost is
    invisible. The two halves disagree often enough that the paper needs both,
    and putting them side by side costs one float instead of two.
    """
    base, dn, dnq, df, dfq = {}, {}, {}, {}, {}
    for arm in ARMS:
        for task in TASKS:
            d = frame[(frame.arm == arm) & (frame.task == task)]
            b = d[(d.condition == "none") & (d.hit_cap == 0)]
            base[(arm, task)] = 100.0 * b.exact.mean()
            vn = ["filler"] + CONDS
            ds, ps = {}, {}
            for c in vn:
                ds[c], ps[c], _ = paired(frame, arm, task, "none", c)
            dn[(arm, task)] = ds
            dnq[(arm, task)] = dict(zip(vn, bh([ps[c] for c in vn])))
            ds, ps = {}, {}
            for c in CONDS:
                ds[c], ps[c], _ = paired(frame, arm, task, "filler", c)
            df[(arm, task)] = ds
            dfq[(arm, task)] = dict(zip(CONDS, bh([ps[c] for c in CONDS])))

    vn = ["filler"] + CONDS
    lines = [r"% Auto-generated by paper/make_v3_tables.py -- do not hand-edit.",
             r"\begin{table*}[!ht]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3.1pt}",
             r"\renewcommand{\arraystretch}{0.94}",
             r"\begin{tabular}{llr" + "r" * len(vn) + "r" * len(CONDS) + "}",
             r"\toprule",
             r"& & & \multicolumn{" + str(len(vn)) +
             r"}{c}{$\Delta$ vs \texttt{none}} & \multicolumn{" +
             str(len(CONDS)) + r"}{c}{$\Delta$ vs \texttt{filler}} \\",
             r"\cmidrule(lr){4-" + str(3 + len(vn)) + r"}" +
             r"\cmidrule(lr){" + str(4 + len(vn)) + "-" +
             str(3 + len(vn) + len(CONDS)) + "}",
             "Arm & Task & none & " + " & ".join(HEADS[c] for c in vn) +
             " & " + " & ".join(HEADS[c] for c in CONDS) + r" \\",
             r"\midrule"]
    for i, arm in enumerate(ARMS):
        if i:
            lines.append(r"\midrule")
        lines.append(r"\multirow{6}{*}{" + tex(arm) + "}")
        for task in TASKS:
            k = (arm, task)
            flag = r"$^{\dagger}$" if task in CONST_GOLD else ""
            lines.append(
                " & " + tex(task) + flag + f" & {base[k]:.1f} & " +
                " & ".join(cell(dn[k][c], dnq[k][c]) for c in vn) + " & " +
                " & ".join(cell(df[k][c], dfq[k][c]) for c in CONDS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Main sweep, $n{=}40$, pooled over densities "
              r"$p \in \{0.10, 0.20, 0.35, 0.50\}$, $100$ graphs per (task, "
              r"primer, density) cell. The \texttt{none} column is "
              r"exact-match accuracy (\%); the rest are paired differences in "
              r"percentage points, against \texttt{none} on the left and "
              r"against the length-matched \texttt{filler} on the right, with "
              r"truncated generations excluded pairwise. Reading a primer "
              r"against \texttt{none} alone charges it for the text it adds; "
              r"reading it against \texttt{filler} alone hides what that text "
              r"costs, which the \texttt{filler} column gives. "
              r"$^{\dagger}$: gold answer is constant at $n{=}40$, so a shift "
              r"need not reflect graph reading "
              r"(Section~\ref{sec:results-difficulty}). \textbf{Bold}: "
              r"significant, exact McNemar, Benjamini--Hochberg within each "
              r"(arm, task) family and baseline, $q=0.05$.}",
              r"\label{tab:main}", r"\end{table*}"]
    write("v3_main_table.tex", lines)


# ----------------------------------------------------------- tab:additivity
def additivity():
    """Is `all` the sum of degree + clustering + rwse? It is not."""
    a = pd.read_csv(SRC + "additivity.csv")
    a = a[a.density.isin(DENS4) & ~a.task.isin(CONST_GOLD)]
    g = a.groupby(["arm", "task"]).agg(parts=("sum_parts", "mean"),
                                       allv=("d_all", "mean"))
    g = 100 * g
    # A ratio of two null effects is noise. Four points is roughly the smallest
    # pooled difference the design can resolve, so below that no ratio is shown.
    g = g[g.parts.abs() >= 4.0]
    g["ratio"] = g.allv / g.parts

    lines = [r"% Auto-generated by paper/make_v3_tables.py -- do not hand-edit.",
             r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{llrrr}", r"\toprule",
             r"Arm & Task & $\sum$ parts & \texttt{all} & ratio \\",
             r"\midrule"]
    for group, rows in (("other tasks", g[g.index.get_level_values("task")
                                          != "edge_count"]),
                        (r"\texttt{edge\_count}",
                         g[g.index.get_level_values("task") == "edge_count"])):
        lines.append(r"\multicolumn{5}{l}{\emph{" + group + r"}} \\")
        for (arm, task), row in rows.iterrows():
            lines.append(f"{tex(SHORT[arm])} & {tex(task)} & "
                         f"${row.parts:+.1f}$ & ${row.allv:+.1f}$ & "
                         f"${row.ratio:.2f}$ \\\\")
        lines.append(r"\midrule")
    # Headline statistic is the per-cell median, not a ratio of these averaged
    # rows: averaging the numerator and denominator over densities first is a
    # ratio of means, which the largest cells dominate. The per-cell median is
    # stable across effect magnitudes (0.43/0.33/0.48 in the [4,8), [8,15) and
    # [15,100) point bands), the ratio of means is not.
    cells = a[(100 * a.sum_parts).abs() >= 4.0]
    percell = (cells.d_all / cells.sum_parts).median()
    lines += [r"\multicolumn{4}{l}{\emph{median over cells}} & " +
              f"${percell:.2f}$" + r" \\",
              r"\bottomrule", r"\end{tabular}",
              r"\caption{\texttt{all} renders \texttt{degree}, "
              r"\texttt{clustering} and \texttt{rwse} in one primer, so the "
              r"parts and the bundle are both measured. The first two columns "
              r"are paired differences against \texttt{none} in percentage "
              r"points, averaged over $p \in \{0.10, 0.20, 0.35, 0.50\}$; "
              r"rows whose parts sum to under $4$ points are omitted, since a "
              r"ratio of two null effects is not informative. The final row "
              r"is the median over the $48$ individual (arm, task, density) "
              r"cells rather than over the averaged rows above, because "
              r"averaging numerator and denominator over densities first "
              r"gives a ratio of means that the largest cells dominate; "
              r"weighted the other ways the estimate runs from $0.29$ to "
              r"$0.71$, so it bounds a tendency and is not a coefficient. "
              r"Either way the bundle delivers well under the sum of its "
              r"parts, and the shortfall is the same whether the parts help "
              r"or hurt, which is what dilution rather than interference "
              r"would predict. \texttt{edge\_count} is the extreme case: "
              r"there the bundle keeps almost nothing and in two arms moves "
              r"against its parts.}",
              r"\label{tab:additivity}", r"\end{table}"]
    write("v3_additivity_table.tex", lines)


# -------------------------------------------------------------- tab:pertask
def per_task(frame):
    """One block per task: accuracy under none, then both deltas, per arm."""
    lines = [r"% Auto-generated by paper/make_v3_tables.py -- do not hand-edit."]
    for task in TASKS:
        lines += [r"\begin{table}[htbp]", r"\centering", r"\small",
                  r"\setlength{\tabcolsep}{3.4pt}",
                  r"\begin{tabular}{lrrrrrrr}", r"\toprule",
                  r"Arm & none & " +
                  " & ".join(HEADS[c] for c in ["filler"] + CONDS) + r" \\"]
        for arm in ARMS:
            d = frame[(frame.arm == arm) & (frame.task == task)]
            base = d[d.condition == "none"]
            base = 100.0 * base[base.hit_cap == 0].exact.mean()
            ds, ps = {}, {}
            for c in ["filler"] + CONDS:
                ds[c], ps[c], _ = paired(frame, arm, task, "none", c)
            order = ["filler"] + CONDS
            qs = dict(zip(order, bh([ps[c] for c in order])))
            lines += [r"\midrule" if arm == ARMS[0] else "",
                      f"{tex(SHORT[arm])} & {base:.1f} & " +
                      " & ".join(cell(ds[c], qs[c]) for c in order) + r" \\"]
        lines = [x for x in lines if x != ""]
        flag = (r" Gold is constant at $n{=}40$ on this task, so a shift need "
                r"not reflect graph reading." if task in CONST_GOLD else "")
        lines += [r"\bottomrule", r"\end{tabular}",
                  r"\caption{\texttt{" + tex(task) + r"}, main sweep, pooled "
                  r"over $p \in \{0.10, 0.20, 0.35, 0.50\}$. The "
                  r"\texttt{none} column is exact-match accuracy (\%); the "
                  r"rest are paired differences against it in percentage "
                  r"points, truncated generations excluded pairwise." + flag +
                  r" \textbf{Bold}: exact McNemar, Benjamini--Hochberg within "
                  r"the arm, $q=0.05$.}",
                  r"\label{tab:pertask-" + task.replace("_", "") + "}",
                  r"\end{table}", ""]
    write("v3_pertask_tables.tex", lines)


# ------------------------------------------------------------ tab:behaviour
def behaviour():
    """What the primer changes in the response, not in the score.

    Length is measured on generations that terminated: a capped response is
    truncated at the budget, not long, so a median over all rows would
    understate the spread. The cap rate is printed beside it.
    """
    b = pd.read_csv(SRC + "behaviour.csv")
    picks = [("qwen3-1.7b", "edge_count", 0.50),
             ("qwen3-4b", "node_degree", 0.50),
             ("qwen3-1.7b", "edge_existence", 0.50),
             ("qwen3-4b", "connected_nodes", 0.50)]
    conds = ["none", "filler", "degree", "all"]
    lines = [r"% Auto-generated by paper/make_v3_tables.py -- do not hand-edit.",
             r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{3.6pt}",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Arm / task & & " + " & ".join(HEADS[c] for c in conds) +
             r" \\", r"\midrule"]
    for arm, task, dens in picks:
        s = b[(b.arm == arm) & (b.task == task) & (b.density == dens)]
        if s.empty:
            continue
        g = s.set_index("condition")
        lines.append(r"\multirow{3}{*}{\shortstack[l]{" + tex(SHORT[arm]) +
                     r"\\" + tex(task) + r"}}")
        for label, col, fmt in [("median tokens", "median_tokens", "{:.0f}"),
                                (r"\% sum-of-deg.", "uses_degree_sum", "{:.0%}"),
                                (r"\% truncated", "hit_cap_rate", "{:.0%}")]:
            vals = []
            for c in conds:
                v = g.loc[c, col] if c in g.index else np.nan
                # Suppressed above 20% truncation upstream: a median over the
                # survivors of a heavily truncated cell samples the short
                # responses only. Render it as missing, not as "nan".
                vals.append("--" if pd.isna(v)
                            else fmt.format(v).replace("%", r"\%"))
            lines.append(" & " + label + " & " + " & ".join(vals) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Response behaviour at $p{=}0.50$. Median new tokens "
              r"is computed over generations that terminated, since a capped "
              r"response is truncated at the budget rather than long, and is "
              r"reported as \mbox{--} where more than $20\%$ of generations "
              r"truncated, because the surviving median is then a sample of "
              r"the short responses; read the truncation row for those cells. "
              r"``sum-of-deg.'' is the share of responses that state the "
              r"handshake shortcut (sum the degrees, divide by two) rather "
              r"than enumerating edges. The primer moves length in both "
              r"directions depending on the task, and on \texttt{edge\_count} "
              r"it replaces the method outright.}",
              r"\label{tab:behaviour}", r"\end{table}"]
    write("v3_behaviour_table.tex", lines)


# ------------------------------------------------------------ tab:relevance
def relevance():
    """Does a primer help when and only when it carries what the task needs?"""
    r = pd.read_csv(SRC + "relevance.csv")
    r = r[r.density.isin(DENS4) & ~r.task.isin(CONST_GOLD)
          & (r.condition != "filler")]

    def row(label, sub):
        cells = []
        for arm in ARMS:
            s = sub[sub.arm == arm]
            cells.append("--" if s.empty
                         else f"${100 * s.delta_vs_filler.mean():+.1f}$")
        return label + " & " + " & ".join(cells) + r" \\"

    lines = [r"% Auto-generated by paper/make_v3_tables.py -- do not hand-edit.",
             r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Primer / task & " + " & ".join(tex(SHORT[a]) for a in ARMS) +
             r" \\", r"\midrule",
             r"\multicolumn{5}{l}{\emph{matched: the primer carries what the "
             r"task needs}} \\"]
    matched = r[r.matched == 1]
    for (cond, task), g in matched.groupby(["condition", "task"]):
        lines.append(row(f"{tex(cond)} / {tex(task)}", g))
    lines += [r"\midrule",
              r"\multicolumn{5}{l}{\emph{mismatched, mean over all pairs}} \\",
              row("---", r[r.matched == 0]),
              r"\bottomrule", r"\end{tabular}",
              r"\caption{Each primer reports one graph feature and each task "
              r"needs one, so a pair is \emph{matched} when the primer carries "
              r"what the task asks for. Entries are paired differences against "
              r"\texttt{filler} in percentage points, averaged over "
              r"$p \in \{0.10, 0.20, 0.35, 0.50\}$; constant-gold tasks are "
              r"excluded. The three largest gains we measure are all matched "
              r"pairs, but matching is not sufficient and not necessary: "
              r"mismatched content is still worth $+8.1$ points on "
              r"\texttt{qwen3-1.7b}, more than two of the matched pairs, and "
              r"the fourth matched pair reverses with model size. That pair is "
              r"\texttt{degree} on \texttt{node\_degree}, the one case where "
              r"the primer states the requested number outright: it gains "
              r"$8$ points on the 1.7B arm and loses $6$ on the 4B arm "
              r"(Section~\ref{sec:results-retrieval}).}",
              r"\label{tab:relevance}", r"\end{table}"]
    write("v3_relevance_table.tex", lines)


if __name__ == "__main__":
    frame = load_frame()
    main_table(frame)
    additivity()
    per_task(frame)
    behaviour()
    relevance()
