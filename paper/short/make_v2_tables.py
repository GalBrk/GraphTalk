"""Emit the four float files the v2 additions need, into paper/.

Writes new files only; nothing existing in paper/ is modified.

  tab:degfixdeg     fixed-mean-degree grid      (magnitude-vs-size, clustering)
  tab:cncontrols    connected_nodes vs both controls, qwen3-4b
  tab:lengthcost    length cost by task + content gain by arm
  tab:cnmetric      set-F1 against exact match on connected_nodes

  PYTHONPATH=. python paper/short/make_v2_tables.py
"""
import collections
import csv
import glob
import json
import statistics

from graphtalk import scoring, significance

OUT = "paper/"
PLAIN = ["qwen3-1.7b", "qwen3-4b"]
ARMS3 = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b"]
DENS4 = ["p0.1", "p0.2", "p0.35", "p0.5"]
CELLS = [((20, 0.421), 8), ((40, 0.205), 8), ((80, 0.101), 8),
         ((160, 0.05), 8), ((20, 0.842), 16), ((40, 0.41), 16),
         ((80, 0.203), 16), ((160, 0.101), 16)]


def tex(s):
    return str(s).replace("_", r"\_")


def write(name, lines):
    with open(OUT + name, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote " + OUT + name)


# ---------------------------------------------------------------- degfixdeg
def degfixdeg():
    def paired(arm):
        out = collections.defaultdict(lambda: collections.defaultdict(dict))
        for path in sorted(glob.glob(f"runs/{arm}.degfixdeg.shard*.jsonl")):
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("hit_cap"):
                    continue
                p = str(r["instance_id"]).split("/")
                cell = (int(p[1].replace("size", "")), float(p[2][1:]))
                out[cell][p[-1]][r["condition"]] = scoring.score_one(
                    scoring.extract_answer(r["response"], r["task"]),
                    r["gold"], r["task"])["exact"]
        return out

    small, large = paired("qwen3-1.7b"), paired("qwen3-8b")

    def arms(d, cell):
        c, t = [], []
        for by in d[cell].values():
            if "none" in by and "clustering" in by:
                c.append(by["none"])
                t.append(by["clustering"])
        return c, t

    rows, pc, pt = [], [], []
    for cell, deg in CELLS:
        c, t = arms(small, cell)
        lc, lt = arms(large, cell)
        rows.append((cell[0], deg, len(c), sum(c) / len(c), sum(t) / len(t),
                     100 * (sum(t) - sum(c)) / len(c),
                     scoring.mcnemar(c, t)["p_value"],
                     sum(lc) / len(lc), 100 * (sum(lt) - sum(lc)) / len(lc)))
        pc += c
        pt += t
    pm = scoring.mcnemar(pc, pt)
    delta = 100 * (sum(pt) - sum(pc)) / len(pc)
    assert len(pc) == 3191 and abs(delta - 6.20) < 0.05, (len(pc), delta)
    flags = significance.benjamini_hochberg([r[6] for r in rows])

    out = [r"\begin{table}[t]\centering\small",
           r"\setlength{\tabcolsep}{3.5pt}",
           r"\begin{tabular}{rrrrrrr}", r"\toprule",
           r"& & \multicolumn{3}{c}{\texttt{qwen3-1.7b}} "
           r"& \multicolumn{2}{c}{\texttt{qwen3-8b}} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
           r"$n$ & $\bar{d}$ & \texttt{none} & \texttt{clust.} & $\Delta$ "
           r"& \texttt{none} & $\Delta$ \\", r"\midrule"]
    for i, r in enumerate(rows):
        star = r"$^{*}$" if flags[i] else ""
        out.append(f"${r[0]}$ & ${r[1]}$ & ${r[3]:.3f}$ & ${r[4]:.3f}$ & "
                   f"${r[5]:+.1f}${star} & ${r[7]:.3f}$ & ${r[8]:+.1f}$ \\\\")
    out += [r"\midrule",
            r"\multicolumn{2}{l}{pooled} & "
            f"${sum(pc) / len(pc):.3f}$ & ${sum(pt) / len(pt):.3f}$ & "
            f"$\\mathbf{{{delta:+.1f}}}$ & "
            f"\\multicolumn{{2}}{{c}}{{$n{{=}}{len(pc)}$, $p{{<}}10^{{-4}}$}} \\\\",
            r"\bottomrule", r"\end{tabular}",
            r"\caption{Fixed-mean-degree sweep on \texttt{node\_degree}: mean "
            r"degree $\bar{d}$ is held while $n$ varies, so density falls as "
            r"the prompt lengthens. Growing $n$ eightfold at $\bar{d}{\approx}8$ "
            r"costs $29$ points and saturates after $n{=}80$; doubling "
            r"$\bar{d}$ costs $33$--$42$ points at every $n$. $\Delta$ is "
            r"\texttt{clustering} against \texttt{none}, exact match. "
            r"$^{*}$: Benjamini--Hochberg at $q{=}0.05$ across the eight "
            r"cells. This corpus has no \texttt{filler} arm, so $\Delta$ is a "
            r"net effect. \texttt{qwen3-8b} is a ceiling control.}",
            r"\label{tab:degfixdeg}", r"\end{table}"]
    write("degfixdeg_table.tex", out)
    print(f"  pooled {delta:+.2f} n={len(pc)} b={pm['b']} c={pm['c']} "
          f"BH {sum(flags)}/8")


# ------------------------------------------------- connected_nodes controls
def cn_controls():
    rows = json.load(open("vs_controls_densfull40.json", encoding="utf-8"))
    by = collections.defaultdict(dict)
    for r in rows:
        if r["arm"] == "qwen3-4b" and r["task"] == "connected_nodes":
            by[r["condition"]][r["control"]] = r
    out = [r"\begin{table}[t]\centering\small",
           r"\begin{tabular}{lrrrr}", r"\toprule",
           r"& \multicolumn{2}{c}{vs \texttt{none}} "
           r"& \multicolumn{2}{c}{vs \texttt{filler}} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"Cond. & $\Delta$ & $p$ & $\Delta$ & $p$ \\", r"\midrule"]

    def fmt(p):
        return r"$10^{-4}$" if p < 2e-4 else f"${p:.3f}$"

    for c in ["components", "clustering", "rwse", "degree", "all", "filler"]:
        n_, f_ = by[c].get("none"), by[c].get("filler")
        if not n_:
            continue
        bold = (f_ and n_["bh_global_reject"] and f_["bh_global_reject"])
        d1 = (f"$\\mathbf{{{n_['delta']:+.1f}}}$" if bold
              else f"${n_['delta']:+.1f}$")
        if f_:
            d2 = (f"$\\mathbf{{{f_['delta']:+.1f}}}$" if bold
                  else f"${f_['delta']:+.1f}$")
            out.append(f"\\texttt{{{tex(c)}}} & {d1} & {fmt(n_['p_perm'])} & "
                       f"{d2} & {fmt(f_['p_perm'])} \\\\")
        else:
            out.append(f"\\texttt{{{tex(c)}}} & {d1} & {fmt(n_['p_perm'])} & "
                       f"--- & --- \\\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{\texttt{connected\_nodes}, \texttt{qwen3-4b}, main "
            r"sweep pooled ($n{=}400$), exact match, points. Every condition "
            r"beats \texttt{filler}, because \texttt{filler} itself costs "
            r"$12.5$ points; only \texttt{components} also beats "
            r"\texttt{none}, and it adds $8$ tokens where \texttt{filler} "
            r"adds $470$ (Figure~\ref{fig:design}). \textbf{Bold}: "
            r"survives the global correction against both controls.}",
            r"\label{tab:cncontrols}", r"\end{table}"]
    write("connected_nodes_controls.tex", out)


# -------------------------------------------------------------- length cost
def length_cost():
    rows = list(csv.DictReader(
        open("csv2/sweep-large-graph/primer_decomposition.csv", encoding="utf-8")))
    mid = [x for x in rows if x["mid_range"] == "True"]
    bytask = collections.defaultdict(list)
    for x in mid:
        if x["condition"] == "clustering":     # one row per cell
            bytask[x["task"]].append(float(x["length_cost_pp"]))
    byarm = collections.defaultdict(list)
    for x in mid:
        byarm[(x["condition"], x["arm"])].append(float(x["content_gain_pp"]))

    out = [r"\begin{table}[t]\centering\small",
           r"\begin{tabular}{lrrr}", r"\toprule",
           r"\multicolumn{4}{l}{\emph{(a) length cost "
           r"(\texttt{filler}$-$\texttt{none}) by task}} \\",
           r"Task & cells & mean & range \\", r"\midrule"]
    for t, v in sorted(bytask.items(), key=lambda kv: statistics.mean(kv[1])):
        out.append(f"\\texttt{{{tex(t)}}} & ${len(v)}$ & "
                   f"${statistics.mean(v):+.1f}$ & "
                   f"$[{min(v):+.1f}, {max(v):+.1f}]$ \\\\")
    out += [r"\midrule",
            r"\multicolumn{4}{l}{\emph{(b) content gain "
            r"(cond.$-$\texttt{filler}) by arm}} \\",
            r"Cond. & \multicolumn{3}{c}{"
            + " / ".join(r"\texttt{%s}" % tex(a.replace("qwen3-", ""))
                         for a in ARMS3) + r"} \\", r"\midrule"]
    for c in ["components", "clustering", "rwse"]:
        vals = " / ".join(
            f"${statistics.mean(byarm[(c, a)]):+.1f}$" if byarm[(c, a)]
            else "---" for a in ARMS3)
        out.append(f"\\texttt{{{tex(c)}}} & \\multicolumn{{2}}{{c}}{{{vals}}} \\\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Mid-range cells. (a) The cost of added text is "
            r"task-specific, not a per-token rate, and changes sign within "
            r"tasks, so it cannot be transferred across them. "
            r"\texttt{node\_count} rests on a single cell and is a "
            r"constant-gold task; it is shown for completeness only. "
            r"(b) Only "
            r"\texttt{components} holds its content gain across arms; the "
            r"ranking of the other two reverses between \texttt{1.7b} and "
            r"\texttt{4b}, so no single ordering of primers by length "
            r"survives disaggregation.}",
            r"\label{tab:lengthcost}", r"\end{table}"]
    write("length_cost_table.tex", out)


# ------------------------------------------------------- F1 vs exact metric
def cn_metric():
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for arm in PLAIN:
        for path in glob.glob(f"runs/{arm}.densfull40.shard*.jsonl"):
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if (r["task"] != "connected_nodes" or r["condition"] != "none"
                        or r.get("hit_cap")):
                    continue
                d = str(r["instance_id"]).split("/")[2]
                res = scoring.score_one(
                    scoring.extract_answer(r["response"], r["task"]),
                    r["gold"], r["task"])
                a = agg[(arm, d)]
                a[0] += 1
                a[1] += res["primary"]
                a[2] += res["exact"]
    out = [r"\begin{table}[t]\centering\small",
           r"\begin{tabular}{l" + "rr" * len(DENS4) + "}", r"\toprule",
           r"& " + " & ".join(r"\multicolumn{2}{c}{$p{=}%s$}" % d[1:]
                              for d in DENS4) + r" \\",
           "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i)
                   for i in range(len(DENS4))),
           r"Arm & " + " & ".join(["F1 & exact"] * len(DENS4)) + r" \\",
           r"\midrule"]
    for arm in PLAIN:
        cells = []
        for d in DENS4:
            n, f1, ex = agg[(arm, d)]
            cells += [f"${f1 / n:.3f}$", f"${ex / n:.3f}$"]
        out.append(r"\texttt{%s} & " % tex(arm) + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{\texttt{connected\_nodes} under \texttt{none}: set-F1, "
            r"the task's default metric, against exact match. F1 stays at "
            r"$0.96$--$0.99$ everywhere while exact match falls by $44$ points "
            r"for \texttt{qwen3-1.7b}, so the task looks ceilinged under F1 "
            r"and is not.}",
            r"\label{tab:cnmetric}", r"\end{table}"]
    write("connected_nodes_metric.tex", out)


# --------------------------------------------------- published-split ceiling
PUBLISHED = ["gemma4-12b", "gemma4-12b-think", "gemma4-e4b", "gemma4-e4b-think",
             "qwen3-8b", "qwen3-8b-think", "qwen3-14b", "qwen3-14b-think"]
PUB_TASKS = ["node_count", "node_degree", "edge_count", "connected_nodes",
             "cycle_check"]


def published_ceiling():
    """Why the 5-19 node published split was left behind: it has no headroom."""
    head = " & ".join(r"\texttt{%s}" % tex(t) for t in PUB_TASKS)
    out = [r"\begin{table}[H]", r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{3pt}",
           r"\resizebox{\columnwidth}{!}{%",
           r"\begin{tabular}{l" + "r" * (len(PUB_TASKS) + 2) + "}",
           r"\toprule",
           "Arm & " + head + r" & all & cap \\",
           r"\midrule"]
    for arm in PUBLISHED:
        per, capped, total = collections.defaultdict(list), 0, 0
        for path in (glob.glob(f"runs/{arm}.jsonl")
                     + glob.glob(f"runs/{arm}.shard*of*.jsonl")):
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["condition"] != "none":
                    continue
                total += 1
                if r.get("hit_cap") in (True, "True"):
                    capped += 1
                    continue
                per[r["task"]].append(scoring.score_one(
                    scoring.extract_answer(r["response"], r["task"]),
                    r["gold"], r["task"])["exact"])
        flat = [x for v in per.values() for x in v]
        cells = [f"${sum(per[t]) / len(per[t]):.2f}$" for t in PUB_TASKS]
        out.append(r"\texttt{%s} & " % tex(arm) + " & ".join(cells)
                   + f" & ${sum(flat) / len(flat):.3f}$ & ${capped}/{total}$"
                   + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}",
            r"\caption{The published GraphQA split ($5$--$19$ nodes, $30$ "
            r"graphs per cell) under \texttt{none}, exact match. Every arm "
            r"scores $0.97$--$1.00$ on four of the five tasks; only "
            r"\texttt{edge\_count} separates them, and only for the two plain "
            r"Qwen arms ($0.40$--$0.43$ against $0.93$--$1.00$ for the rest). "
            r"\emph{cap} counts "
            r"generations that hit the token budget, excluded from the "
            r"accuracies. This corpus cannot measure a primer effect, which is "
            r"why the main sweep moves to $n{=}40$.}",
            r"\label{tab:ceiling}", r"\end{table}"]
    write("published_ceiling.tex", out)


if __name__ == "__main__":
    degfixdeg()
    cn_controls()
    length_cost()
    cn_metric()
    published_ceiling()
