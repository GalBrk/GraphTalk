"""Emit the main-sweep LaTeX table (all four arms) straight from the scored
densfull40 runs, so the paper's headline table cannot drift from a
hand-typed number.

Exact match is used for every task, including `connected_nodes` (the
published-split shortcut table uses set-F1 there, but F1 is 0.96-1.00
everywhere at n=40 and the paper's own tables always reported exact match;
see docs/paper-revision-handoff.md's review). Truncated (`hit_cap`) rows are
dropped pairwise. Significance is the exact McNemar test, Benjamini-Hochberg
corrected within each (arm, task) family of six conditions at q=0.05 --
the convention the table caption states.

  PYTHONPATH=. python paper/make_main_table.py
"""
import collections
import glob
import json

from graphtalk import scoring, significance

PLAIN_ARMS = ["qwen3-1.7b", "qwen3-4b"]
THINK_ARMS = ["qwen3-1.7b-think", "qwen3-4b-think"]
ARMS = PLAIN_ARMS + THINK_ARMS
# Gold answer is constant at n=40 (always 40, always "yes"), so a condition can
# move accuracy without reading the graph. Flagged in the table rather than
# explained away in the prose.
CONSTANT_TASKS = {"node_count", "cycle_check"}
# >=62% of these generations reach the output budget, so the cell is reported
# but carries no verdict.
TRUNCATED_CELLS = {("qwen3-1.7b-think", "edge_count"),
                  ("qwen3-4b-think", "edge_count")}
CONDS =["components", "clustering", "rwse", "degree", "filler", "all"]
COND_HEAD = ["comp.", "clust.", "rwse", "degree", "filler", "all"]
TASKS = ["connected_nodes", "cycle_check", "edge_count", "edge_existence",
        "node_count", "node_degree"]
TASK_HEAD = {"connected_nodes": "connected\\_nodes", "cycle_check": "cycle\\_check",
            "edge_count": "edge\\_count", "edge_existence": "edge\\_existence",
            "node_count": "node\\_count", "node_degree": "node\\_degree"}


def load(arm):
  """{(task, condition, instance_id): score|None}; None marks hit_cap."""
  out, seen = {}, set()
  for path in sorted(glob.glob(f"runs/{arm}.densfull40.shard*of25.jsonl")):
    for line in open(path, encoding="utf-8"):
      if not line.strip():
        continue
      r = json.loads(line)
      k = (r["task"], r["condition"], r["instance_id"])
      if k in seen:
        continue
      seen.add(k)
      out[k] = None if r.get("hit_cap") else scoring.score_one(
          scoring.extract_answer(r["response"], r["task"]),
          r["gold"], r["task"])["exact"]
  return out


def cell(scores, task, cond):
  """(none_acc, delta_pp, p_value, n_pairs) for one (task, cond), pooled over
  density. Pairs where either side hit_cap are dropped."""
  base = {iid: v for (t, c, iid), v in scores.items()
          if t == task and c == "none" and v is not None}
  treat = {iid: v for (t, c, iid), v in scores.items()
          if t == task and c == cond and v is not None}
  shared = sorted(base.keys() & treat.keys())
  b = [base[i] for i in shared]
  v = [treat[i] for i in shared]
  none_acc = 100.0 * sum(base.values()) / len(base) if base else float("nan")
  delta = 100.0 * (sum(v) - sum(b)) / len(shared) if shared else float("nan")
  p = scoring.mcnemar(b, v)["p_value"] if shared else float("nan")
  return none_acc, delta, p, len(shared)


def build_arm_block(arm, scores):
  """-> (rows, n_range) where rows is [(task, none_acc, {cond: (delta, sig)})]."""
  p_values, index = [], []
  raw = {}
  for task in TASKS:
    for cond in CONDS:
      none_acc, delta, p, n = cell(scores, task, cond)
      raw[(task, cond)] = (none_acc, delta, p, n)
  for task in TASKS:
    ps = [raw[(task, c)][2] for c in CONDS]
    sig = significance.benjamini_hochberg(ps, q=0.05)
    for cond, s in zip(CONDS, sig):
      raw[(task, cond)] = raw[(task, cond)] + (s,)
  rows = []
  ns = []
  for task in TASKS:
    none_acc = raw[(task, CONDS[0])][0]
    cells = {}
    for cond in CONDS:
      _, delta, _, n, sig = raw[(task, cond)]
      cells[cond] = (delta, sig)
      ns.append(n)
    rows.append((task, none_acc, cells))
  return rows, (min(ns), max(ns))


def fmt_delta(delta, sig):
  s = f"${delta:+.1f}$"
  return r"$\mathbf{%s}$" % f"{delta:+.1f}" if sig else s


def render_block(arms, label, caption):
  out = [r"\begin{table*}[!ht]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4.2pt}", r"\renewcommand{\arraystretch}{0.94}",
         r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Arm & Task & none & " + " & ".join(COND_HEAD) + r" \\", r"\midrule"]
  all_ns = []
  for ai, arm in enumerate(arms):
    scores = load(arm)
    rows, n_range = build_arm_block(arm, scores)
    all_ns.append(n_range)
    out.append(r"\multirow{%d}{*}{%s}" % (len(rows), arm))
    for task, none_acc, cells in rows:
      mark = r"$^{\dagger}$" if task in CONSTANT_TASKS else ""
      if (arm, task) in TRUNCATED_CELLS:
        mark += r"$^{\ddagger}$"
      line = f" & {TASK_HEAD[task]}{mark} & {none_acc:.1f}"
      for cond in CONDS:
        delta, sig = cells[cond]
        line += " & " + fmt_delta(delta, sig)
      out.append(line + r" \\")
    if ai < len(arms) - 1:
      out.append(r"\midrule")
  lo = min(r[0] for r in all_ns)
  hi = max(r[1] for r in all_ns)
  out += [r"\bottomrule", r"\end{tabular}",
         r"\caption{%s $n \in [%d, %d]$ paired instances/cell. "
         r"\textbf{Bold}: significant, exact McNemar, "
         r"Benjamini--Hochberg within each (arm, task) family, $q=0.05$.}"
         % (caption, lo, hi),
         r"\label{%s}" % label, r"\end{table*}"]
  return "\n".join(out)


def main():
  tex = render_block(
      ARMS, "tab:main",
      r"Main sweep, $n{=}40$, pooled over densities "
      r"$p \in \{0.10, 0.20, 0.35, 0.50\}$, $100$ graphs per (task, primer, "
      r"density) cell. The \texttt{none} column is absolute accuracy (\%); "
      r"all other columns are paired differences against \texttt{none} in "
      r"percentage points, truncated generations excluded pairwise. "
      r"$^{\dagger}$: gold answer is constant at $n{=}40$, so a shift need "
      r"not reflect graph reading (Section~\ref{sec:results-difficulty}). "
      r"$^{\ddagger}$: $\ge{}62\%$ of generations reach the output budget "
      r"(Table~\ref{tab:hitcap}). \texttt{qwen3-4b-think} is at ceiling on "
      r"every task, leaving no headroom for a primer effect; the other three "
      r"arms are not (Section~\ref{sec:results-main}).")
  with open("paper/main_table.tex", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("% Auto-generated by paper/make_main_table.py -- do not hand-edit.\n")
    fh.write(tex + "\n")
  print("wrote paper/main_table.tex")


if __name__ == "__main__":
  main()
