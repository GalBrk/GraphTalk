"""Emit the 5-page paper's Table 1 (task baselines) and its prose macros.

Table 1 is the majority-class baseline for each (task, density) beside the
measured accuracy under `none`. Nothing in the 8-page paper reports the class
prior per task: `tab:baseline` bins the route/no-route effect, and
`score_full_density_sweep.py`'s `blind_bar` prints an equivalent number to a
`.txt` only, pooled over all seven conditions.

Reads the runs directly, like `paper/make_tables.py`, so a number here cannot
drift from the generations it came from. The loader is deliberately a copy of
that file's rather than an import -- `paper/` is not a package, and this
directory is kept self-contained so the 8-page build is never touched.

Also emits two numbers the prose cites that no existing script produces per
density: `node_count`'s "39" rate (`analyze_error_taxonomy.py` pools densities)
and `edge_count`'s median relative error (`score_full_density_sweep.py`
accumulates absolute error but never prints it).

  PYTHONPATH=. python paper/short/make_baseline_table.py
"""
import collections
import glob
import json
import statistics

from graphtalk import scoring

ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"]
PLAIN = ["qwen3-1.7b", "qwen3-4b"]
TASKS = ["node_count", "cycle_check", "edge_existence", "node_degree",
         "edge_count", "connected_nodes"]
DENS = [0.1, 0.2, 0.35, 0.5, 0.65, 0.75, 0.85]


def density(iid):
    for part in str(iid).split("/"):
        if len(part) > 1 and part[0] == "p":
            try:
                return float(part[1:])
            except ValueError:
                pass
    return None


def load(arm):
    """Every densfull40 + densfull40hi row for one arm, deduped."""
    seen, rows = set(), []
    for pattern in (f"runs/{arm}.densfull40.shard*.jsonl",
                    f"runs/{arm}.densfull40hi.shard*.jsonl"):
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
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


def collect(rows_by_arm):
    """-> (prior[(task,d)], acc[(arm,task,d)], capped[(arm,task,d)])."""
    golds = collections.defaultdict(dict)     # (task,d) -> iid -> gold
    acc = collections.defaultdict(lambda: [0.0, 0])
    capped = collections.Counter()
    for arm, rows in rows_by_arm.items():
        for r in rows:
            d = density(r["instance_id"])
            key = (r["task"], d)
            # One gold per graph, not per condition: the answer does not
            # depend on which primer was prepended.
            golds[key].setdefault(r["instance_id"], r["gold"])
            if r["condition"] != "none":
                continue
            if r.get("hit_cap"):
                capped[(arm, r["task"], d)] += 1
                continue
            e = scoring.score_one(
                scoring.extract_answer(r["response"], r["task"]),
                r["gold"], r["task"])["exact"]
            acc[(arm, r["task"], d)][0] += e
            acc[(arm, r["task"], d)][1] += 1
    prior = {k: scoring.majority_baseline(list(v.values()))[1]
             for k, v in golds.items()}
    return prior, acc, capped


def baseline_table(prior, acc):
    out = [r"\begin{table*}[t]", r"\centering\small",
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{ll" + "r" * len(DENS) + "}", r"\toprule",
           r"Task & & " + " & ".join(f"${d}$" for d in DENS) + r" \\",
           r"\midrule"]
    for task in TASKS:
        cells = [f"{prior[(task, d)]:.3f}" if (task, d) in prior else "--"
                 for d in DENS]
        out.append(r"\multirow{3}{*}{\texttt{%s}} & prior & " % tex(task)
                   + " & ".join(cells) + r" \\")
        for arm in PLAIN:
            cells = []
            for d in DENS:
                s, n = acc[(arm, task, d)]
                cells.append(f"{s / n:.3f}" if n else "--")
            out.append(r" & \texttt{%s} & " % tex(arm.replace("qwen3-", ""))
                       + " & ".join(cells) + r" \\")
        out.append(r"\addlinespace[2pt]")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Majority-class baseline (\emph{prior}: the share of the "
            r"modal gold answer, i.e. what a graph-blind constant answer "
            r"scores) against measured exact-match accuracy under "
            r"\texttt{none}, by edge density. Two of six tasks have a prior of "
            r"$1.000$ at $n{=}40$. Truncated generations excluded.}",
            r"\label{tab:priors}", r"\end{table*}"]
    return "\n".join(out)


_DIGIT = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
          "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def alpha(text):
    """LaTeX control sequences may contain letters only, so spell out digits."""
    return "".join(_DIGIT.get(ch, ch) for ch in str(text) if ch.isalnum())


def gap_numbers(rows_by_arm):
    """The two per-density figures the prose cites."""
    macros = []
    # node_count "39" rate under `none`, qwen3-1.7b, by density.
    rate39 = {}
    for r in rows_by_arm["qwen3-1.7b"]:
        if r["task"] != "node_count" or r["condition"] != "none":
            continue
        d = density(r["instance_id"])
        p = scoring.extract_answer(r["response"], "node_count")
        hit, n = rate39.get(d, (0, 0))
        rate39[d] = (hit + (p == "39"), n + 1)
    for d, (hit, n) in sorted(rate39.items()):
        macros.append(r"\newcommand{\ncThirtyNine%s}{%.0f\%%}"
                      % (alpha(str(d).replace("0.", "p")), 100 * hit / n))
    # edge_count median relative error under `none`, plain arms, by density.
    for arm in PLAIN:
        rel = collections.defaultdict(list)
        for r in rows_by_arm[arm]:
            if r["task"] != "edge_count" or r["condition"] != "none":
                continue
            if r.get("hit_cap"):
                continue
            res = scoring.score_one(
                scoring.extract_answer(r["response"], "edge_count"),
                r["gold"], "edge_count")
            if res["absolute_error"] is not None:
                rel[density(r["instance_id"])].append(
                    res["absolute_error"] / int(r["gold"]))
        for d, vals in sorted(rel.items()):
            macros.append(
                r"\newcommand{\ecRel%s%s}{%.1f\%%}"
                % (alpha(arm.replace("qwen3-", "")),
                   alpha(str(d).replace("0.", "p")),
                   100 * statistics.median(vals)))
    return "\n".join(macros)


def main():
    rows_by_arm = {}
    for arm in ARMS:
        rows_by_arm[arm] = load(arm)
        print(f"loaded {arm}: {len(rows_by_arm[arm])} rows", flush=True)
    prior, acc, capped = collect(rows_by_arm)

    with open("paper/short/baseline_table.tex", "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(baseline_table(prior, acc) + "\n")
    print("wrote paper/short/baseline_table.tex")

    with open("paper/short/macros.tex", "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(gap_numbers(rows_by_arm) + "\n")
    print("wrote paper/short/macros.tex")

    print("\nprior vs none accuracy (plain arms):")
    for task in TASKS:
        for d in DENS:
            if (task, d) not in prior:
                continue
            cells = []
            for arm in PLAIN:
                s, n = acc[(arm, task, d)]
                cells.append(f"{arm}={s / n:.3f}" if n else f"{arm}=--")
            print(f"  {task:<17} p={d:<5} prior={prior[(task, d)]:.3f}  "
                  + "  ".join(cells))


if __name__ == "__main__":
    main()
