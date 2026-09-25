"""Emit the 5-page paper's Table 2: the fixed-mean-degree sweep.

`degfixdeg` holds mean degree at ~8 or ~16 while n ranges over {20,40,80,160},
so density falls as prompt length rises. It separates two drivers of difficulty
that a density sweep confounds, and it tests `clustering` on a grid that played
no part in selecting that condition. It appears in no document.

Numbers come from `graphtalk.scoring.mcnemar`, the same function
`superseded/scripts/score_fixed_degree_sweep.py` uses; `main()` asserts the pooled figure
against that script's committed output (`superseded/analysis/tables/degfixdeg.1.7b.txt`)
so this table cannot silently disagree with the scorer.

Cells are keyed on the (size, density) pair, not density alone: two of the
eight share a density value (0.101 at size80/deg~8 and size160/deg~16), and
grouping by density would pool them -- see that script's docstring.

  PYTHONPATH=. python superseded/paper/short/make_degfixdeg_table.py
"""
import collections
import glob
import json

from graphtalk import scoring

ARM = "qwen3-1.7b"
CEILING_ARM = "qwen3-8b"
TREATMENT = "clustering"
CONTROL = "none"
# (size, density) -> nominal mean degree, in the order the table prints.
CELLS = [((20, 0.421), 8), ((40, 0.205), 8), ((80, 0.101), 8),
         ((160, 0.05), 8), ((20, 0.842), 16), ((40, 0.41), 16),
         ((80, 0.203), 16), ((160, 0.101), 16)]


def cell_of(iid):
    parts = str(iid).split("/")
    return int(parts[1].replace("size", "")), float(parts[2][1:])


def paired(arm):
    """(size, density) -> {index -> {condition -> exact}}, hit_cap dropped."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for path in sorted(glob.glob(f"runs/{arm}.degfixdeg.shard*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("hit_cap"):
                    continue
                e = scoring.score_one(
                    scoring.extract_answer(r["response"], r["task"]),
                    r["gold"], r["task"])["exact"]
                out[cell_of(r["instance_id"])][
                    str(r["instance_id"]).split("/")[-1]][r["condition"]] = e
    return out


def arms(by_cell, cell):
    control, treatment = [], []
    for by_cond in by_cell[cell].values():
        if CONTROL in by_cond and TREATMENT in by_cond:
            control.append(by_cond[CONTROL])
            treatment.append(by_cond[TREATMENT])
    return control, treatment


def main():
    small, large = paired(ARM), paired(CEILING_ARM)
    rows, pooled_c, pooled_t = [], [], []
    for cell, deg in CELLS:
        c, t = arms(small, cell)
        m = scoring.mcnemar(c, t)
        lc, lt = arms(large, cell)
        rows.append({
            "size": cell[0], "p": cell[1], "deg": deg, "n": len(c),
            "none": sum(c) / len(c), "clust": sum(t) / len(t),
            "delta": 100 * (sum(t) - sum(c)) / len(c),
            "p_value": m["p_value"],
            "large_none": (sum(lc) / len(lc)) if lc else float("nan"),
            "large_delta": (100 * (sum(lt) - sum(lc)) / len(lc))
                           if lc else float("nan"),
        })
        pooled_c += c
        pooled_t += t
    pm = scoring.mcnemar(pooled_c, pooled_t)
    pooled_delta = 100 * (sum(pooled_t) - sum(pooled_c)) / len(pooled_c)

    # The scorer is the authority; this table must agree with it.
    assert len(pooled_c) == 3191, len(pooled_c)
    assert abs(pooled_delta - 6.20) < 0.05, pooled_delta
    assert pm["b"] == 306 and pm["c"] == 504, (pm["b"], pm["c"])

    flags = dict(zip(range(len(rows)), __import__(
        "graphtalk.significance", fromlist=["x"]).benjamini_hochberg(
            [r["p_value"] for r in rows])))

    out = [r"\begin{table}[t]", r"\centering\small",
           r"\setlength{\tabcolsep}{3.5pt}",
           r"\begin{tabular}{rrrrrrr}", r"\toprule",
           r"& & \multicolumn{3}{c}{\texttt{qwen3-1.7b}} "
           r"& \multicolumn{2}{c}{\texttt{qwen3-8b}} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
           r"$n$ & $\bar{d}$ & \texttt{none} & \texttt{clust.} & $\Delta$ "
           r"& \texttt{none} & $\Delta$ \\", r"\midrule"]
    for i, r in enumerate(rows):
        star = r"$^{*}$" if flags[i] else ""
        out.append(
            f"${r['size']}$ & ${r['deg']}$ & ${r['none']:.3f}$ & "
            f"${r['clust']:.3f}$ & ${r['delta']:+.1f}${star} & "
            f"${r['large_none']:.3f}$ & ${r['large_delta']:+.1f}$ \\\\")
    out += [r"\midrule",
            r"\multicolumn{2}{l}{pooled} & "
            f"${sum(pooled_c) / len(pooled_c):.3f}$ & "
            f"${sum(pooled_t) / len(pooled_t):.3f}$ & "
            f"$\\mathbf{{{pooled_delta:+.1f}}}$ & \\multicolumn{{2}}{{c}}{{"
            f"$n{{=}}{len(pooled_c)}$}} \\\\",
            r"\bottomrule", r"\end{tabular}",
            r"\caption{Fixed-mean-degree sweep: mean degree $\bar{d}$ held at "
            r"$\approx 8$ or $\approx 16$ while $n$ varies, so density falls as "
            r"the prompt lengthens. $\Delta$ is \texttt{clustering} against "
            r"\texttt{none} in points. $^{*}$: exact McNemar surviving "
            r"Benjamini--Hochberg at $q{=}0.05$ across the eight cells. "
            r"\texttt{qwen3-8b} is shown as a ceiling control. This corpus has "
            r"no \texttt{filler} arm, so $\Delta$ is a net effect.}",
            r"\label{tab:degfixdeg}", r"\end{table}"]

    with open("superseded/paper/short/degfixdeg_table.tex", "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote superseded/paper/short/degfixdeg_table.tex")
    print(f"pooled: n={len(pooled_c)} delta={pooled_delta:+.2f} "
          f"b={pm['b']} c={pm['c']} p={pm['p_value']:.3g}  "
          f"BH-significant cells: {sum(flags.values())}/8")


if __name__ == "__main__":
    main()
