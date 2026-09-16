"""Emit the appendix LaTeX tables straight from the scored runs.

Generated rather than hand-written so a number in the appendix cannot drift
from the runs it came from.

  PYTHONPATH=. python paper/make_tables.py
"""
import collections
import glob
import json

from graphtalk import scoring

ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"]
CONDS = ["none", "components", "clustering", "rwse", "degree", "filler", "all"]
TASKS = ["connected_nodes", "cycle_check", "edge_count", "edge_existence",
         "node_count", "node_degree"]
DENS = [0.1, 0.2, 0.35, 0.5]


def density(iid):
    for part in iid.split("/"):
        if len(part) > 1 and part[0] == "p":
            try:
                return float(part[1:])
            except ValueError:
                pass
    return None


def load(arm):
    seen, rows = set(), []
    for path in sorted(glob.glob(f"runs/{arm}.densfull40.shard*of25.jsonl")):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["instance_id"], r["condition"], r["style"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    return rows


def tex(name):
    return name.replace("_", r"\_")


def main():
    out = []
    for arm in ARMS:
        rows = load(arm)
        acc = collections.defaultdict(lambda: [0.0, 0])   # (task,d,cond)->[sum,n]
        cap = collections.Counter()                        # (task,d,cond)
        for r in rows:
            key = (r["task"], density(r["instance_id"]), r["condition"])
            if r.get("hit_cap"):
                cap[key] += 1
                continue
            e = scoring.score_one(
                scoring.extract_answer(r["response"], r["task"]),
                r["gold"], r["task"])["exact"]
            acc[key][0] += e
            acc[key][1] += 1

        out.append(r"\subsection{%s}" % tex(arm))
        for task in TASKS:
            out.append(r"\begin{table}[H]")
            out.append(r"\centering\footnotesize")
            out.append(r"\setlength{\tabcolsep}{3.4pt}")
            out.append(r"\resizebox{\columnwidth}{!}{%")
            out.append(r"\begin{tabular}{l" + "r" * len(CONDS) + "}")
            out.append(r"\toprule")
            out.append("$p$ & " + " & ".join(c[:6] for c in CONDS) + r" \\")
            out.append(r"\midrule")
            for d in DENS:
                cells = []
                for c in CONDS:
                    s, n = acc[(task, d, c)]
                    cells.append(f"{s / n:.3f}" if n else "--")
                out.append(f"{d} & " + " & ".join(cells) + r" \\")
            if sum(cap[(task, d, c)] for d in DENS for c in CONDS):
                out.append(r"\midrule")
                out.append(r"\multicolumn{%d}{l}{\emph{truncated rows dropped}} \\"
                           % (len(CONDS) + 1))
                for d in DENS:
                    cells = [str(cap[(task, d, c)]) for c in CONDS]
                    out.append(f"{d} & " + " & ".join(cells) + r" \\")
            out.append(r"\bottomrule")
            out.append(r"\end{tabular}}")
            out.append(r"\caption{\texttt{%s}, \texttt{%s}: exact-match accuracy "
                       r"by density and primer, truncated rows excluded.}"
                       % (tex(arm), tex(task)))
            out.append(r"\end{table}")
            out.append("")
        print(f"built {arm}", flush=True)

    with open("paper/appendix_tables.tex", "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote paper/appendix_tables.tex")


if __name__ == "__main__":
    main()
