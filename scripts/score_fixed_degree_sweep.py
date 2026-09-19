"""Score a fixed-mean-degree sweep: accuracy and paired tests per (size, density)
cell.

`score_density_sweep.py` groups rows by the density value alone, which is right
for a single-size density sweep but wrong for the fixed-mean-degree design in
`docs/primer-effects-and-power.md` ("Density was never the driver"): that design
pins a *different* density per size to hold mean degree constant as `n` varies
(job 871262, tag `degfixdeg`), and two of its eight cells share a density value
by coincidence -- 0.101 at both size=80/mean-degree~8 and size=160/mean-degree~16.
Grouping by density alone would silently pool those two cells into one blended
row; `run_sweep.py`'s output records don't carry `size_class`/`density_class` as
separate fields (only `build_size_sweep.py`'s own prompt file does), so the fix
has to read both segments out of `instance_id`, not just the `p` one.

This script groups by the (size, density) pair instead
(`node_degree/size80/p0.101/17` -> (80, 0.101)), which is unique across every
cell in this design even though density alone is not.

  PYTHONPATH=. python scripts/score_fixed_degree_sweep.py \
      --responses "runs/qwen3-1.7b.degfixdeg.shard*of5.jsonl" \
                  "runs/qwen3-1.7b.degfixdegfill.shard*of5.jsonl"

`hit_cap` rows are dropped rather than scored zero, for the same reason
`score_density_sweep.py` drops them: a truncated generation is a budget failure,
not a wrong answer. The count dropped is reported per cell.
"""

import argparse
import collections
import glob
import json

from graphtalk import scoring
from graphtalk import significance

CONTROL = "none"


def cell_of(instance_id: str) -> tuple[int, float] | None:
  """The (size, density) cell `build_size_sweep.py --densities` encoded in an
  instance id.

  Returns None when either segment is missing (a tracked-corpus row, or a plain
  size sweep with no `--densities`), so a mixed set of files degrades to a
  single unlabelled group rather than raising.
  """
  size = density = None
  for part in instance_id.split("/"):
    if part.startswith("size") and part[4:].isdigit():
      size = int(part[4:])
    elif len(part) > 1 and part[0] == "p":
      try:
        density = float(part[1:])
      except ValueError:
        continue
  return (size, density) if size is not None and density is not None else None


def mean_degree(cell: tuple[int, float]) -> float:
  """Approximate target mean degree of a cell, for grouping cells in a report.

  ER sparsity p has expected mean degree p * (n - 1); the fixed-mean-degree
  design chooses p per size to hold this near a target (8 or 16), so rounding
  recovers which block a cell belongs to.
  """
  size, density = cell
  return round(density * (size - 1))


def blind_bar(golds) -> tuple[str, float]:
  """Best constant answer over these rows, and what it scores."""
  counts = collections.Counter(golds)
  answer, hits = counts.most_common(1)[0]
  return answer, hits / len(golds)


def load(patterns) -> list[dict]:
  records = []
  for pattern in patterns:
    paths = sorted(glob.glob(pattern)) or [pattern]
    for path in paths:
      with open(path) as handle:
        for line in handle:
          if line.strip():
            records.append(json.loads(line))
  return records


def summarize(records) -> dict:
  """Per-cell aggregates and per-pair scores, hit_cap rows dropped."""
  cells = collections.defaultdict(
      lambda: {"kept": 0, "capped": 0, "total": 0.0, "parsed": 0}
  )
  paired = collections.defaultdict(dict)   # (cell, instance_id) -> cond -> score
  golds = collections.defaultdict(list)    # cell -> golds
  for record in records:
    cell = cell_of(record["instance_id"])
    bucket = cells[(cell, record["condition"])]
    if record.get("hit_cap"):
      bucket["capped"] += 1
      continue
    result = scoring.score_one(
        scoring.extract_answer(record["response"], record["task"]),
        record["gold"], record["task"],
    )
    bucket["kept"] += 1
    bucket["total"] += result["primary"]
    bucket["parsed"] += int(result["parsed"])
    paired[(cell, record["instance_id"])][record["condition"]] = result["primary"]
    golds[cell].append(record["gold"])
  return {"cells": cells, "paired": paired, "golds": golds}


def paired_arms(paired, cells, condition, control=CONTROL):
  """Aligned (control, treatment) hit vectors for the given cell(s).

  `cells=None` pools every cell; pass a set/list of (size, density) tuples to
  restrict to one cell or to one mean-degree block.
  """
  control_hits, treatment_hits = [], []
  for (cell, _), by_condition in paired.items():
    if cells is not None and cell not in cells:
      continue
    if control in by_condition and condition in by_condition:
      control_hits.append(by_condition[control])
      treatment_hits.append(by_condition[condition])
  return control_hits, treatment_hits


def report(summary, control=CONTROL) -> None:
  cells, paired, golds = summary["cells"], summary["paired"], summary["golds"]
  cell_keys = sorted({c for c, _ in cells if c is not None},
                     key=lambda c: (mean_degree(c), c[0]))
  conditions = sorted({cond for _, cond in cells})
  header = "  size   p       d~   " + "".join(f"{c:>13}" for c in conditions)

  print("blind bar (best constant answer, per cell):")
  for cell in cell_keys:
    answer, bar = blind_bar(golds[cell])
    print(f"  size={cell[0]:<4} p={cell[1]!s:<7} d~{mean_degree(cell):<4} "
          f"modal gold {answer!r:>6}  bar {bar:.3f}  (n={len(golds[cell])})")

  print("\nmean score by cell x condition (hit_cap dropped):")
  print(header)
  for cell in cell_keys:
    line = f"  {cell[0]:<6} {cell[1]!s:<7} {mean_degree(cell):<4}"
    for condition in conditions:
      bucket = cells[(cell, condition)]
      value = bucket["total"] / bucket["kept"] if bucket["kept"] else float("nan")
      line += f"{value:>13.3f}"
    print(line)

  print("\ncapped rows dropped, by cell x condition:")
  print(header)
  for cell in cell_keys:
    line = f"  {cell[0]:<6} {cell[1]!s:<7} {mean_degree(cell):<4}"
    for condition in conditions:
      line += f"{cells[(cell, condition)]['capped']:>13}"
    print(line)

  def run(cells_subset, label):
    out = []
    for condition in conditions:
      if condition == control:
        continue
      control_hits, treatment_hits = paired_arms(
          paired, cells_subset, condition, control)
      if not control_hits:
        continue
      test = scoring.mcnemar(control_hits, treatment_hits)
      delta = (test["c"] - test["b"]) / len(control_hits)
      out.append((label, condition, len(control_hits), test, delta))
    return out

  def show(rows, reject) -> None:
    for (label, condition, n, test, delta), keep in zip(rows, reject):
      print(f"  {label:<26} {condition:>11} - {control}: n={n:>5} "
            f"win {test['c']:>4} lose {test['b']:>4} delta {delta:+.4f} "
            f"p={test['p_value']:.4f}{'  *' if keep else ''}")

  print(f"\npooled across every cell, paired vs {control!r} "
        "(exact McNemar on rows sharing an instance_id):")
  pooled = run(None, "POOLED")
  show(pooled, significance.benjamini_hochberg([t["p_value"] for *_, t, _ in pooled]))

  print("\npooled per mean-degree block:")
  blocks = sorted({mean_degree(c) for c in cell_keys})
  by_block = []
  for d in blocks:
    block_cells = {c for c in cell_keys if mean_degree(c) == d}
    by_block += run(block_cells, f"d~{d} pooled")
  show(by_block, significance.benjamini_hochberg([t["p_value"] for *_, t, _ in by_block]))

  print("\nper cell (descriptive -- these are a family, correct before "
        "quoting any one):")
  per_cell = []
  for cell in cell_keys:
    per_cell += run({cell}, f"size={cell[0]} p={cell[1]:g} (d~{mean_degree(cell)})")
  show(per_cell,
       significance.benjamini_hochberg([t["p_value"] for *_, t, _ in per_cell]))
  if per_cell:
    print("  (* = survives Benjamini-Hochberg at q=0.05 within its own family)")

  if "clustering" in conditions and "degree" in conditions:
    print("\nheadroom captured by clustering, per cell "
          "((clustering - none) / (degree - none)):")
    for cell in cell_keys:
      none_v, clustering_v, ceiling_v = (
          cells[(cell, "none")], cells[(cell, "clustering")],
          cells[(cell, "degree")])
      if not (none_v["kept"] and clustering_v["kept"] and ceiling_v["kept"]):
        continue
      none_mean = none_v["total"] / none_v["kept"]
      clustering_mean = clustering_v["total"] / clustering_v["kept"]
      ceiling_mean = ceiling_v["total"] / ceiling_v["kept"]
      gap = ceiling_mean - none_mean
      captured = (clustering_mean - none_mean) / gap if gap else float("nan")
      print(f"  size={cell[0]:<4} p={cell[1]!s:<7} d~{mean_degree(cell):<4} "
            f"ceiling {ceiling_mean:.3f}  none {none_mean:.3f}  "
            f"clustering {clustering_mean:.3f}  captured {captured:+.1%}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--responses", nargs="+", required=True)
  parser.add_argument("--control", default=CONTROL)
  args = parser.parse_args()

  report(summarize(load(args.responses)), control=args.control)


if __name__ == "__main__":
  main()
