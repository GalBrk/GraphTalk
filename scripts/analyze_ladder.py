"""Turn ladder screen runs into the model x rung matrix.

This is the artefact the whole design exists to produce: for every model and
arm, which rungs it is at ceiling on, which at floor, and which are
*informative* -- the only ones where a primer test can be interpreted. A null on
a ceiling rung means "no room to improve" and a null on a floor rung means "no
room to fall"; neither is evidence about primers.

Readability is applied as a second, independent filter, because an informative
rung above the model's reading limit measures reading rather than primer
content: at the density where the plain arm collapses, the `degree` control --
which writes the answer verbatim into the prompt -- was worth +0.7pp. Rungs that
are informative but unreadable are reported in their own column rather than
dropped, since a model with many of those and no valid rung is itself the
finding.

Reading limits are per model AND per arm and must be supplied (`--reading-limits`
as `model=tokens` pairs, from the retrieval probe). Omitting them reports bands
only and says so, rather than silently substituting qwen3-1.7b's brackets --
reusing one model's limit for another is precisely the error this design exists
to avoid.
"""

import argparse
import glob
import json
import re
from collections import defaultdict

from graphtalk import ladder
from graphtalk import scoring

# Gold carries a trailing period in some corpora ("13."), which silently zeroes
# a naive string comparison -- it made four models look like they scored 0.000
# during this module's development. Normalise both sides, always.
_TRAILING = re.compile(r"[\s.]+$")


def _norm(value) -> str:
  return _TRAILING.sub("", str(value).strip())


def _truthy(value) -> bool:
  return str(value).lower() == "true"


def load(patterns, task="node_degree", condition="none"):
  """Accuracy per (model, rung), plus the capped/overflow counts behind it."""
  hits = defaultdict(int)
  totals = defaultdict(int)
  dropped = defaultdict(int)
  for pattern in patterns:
    for path in glob.glob(pattern):
      if "/archive/" in path or ".got." in path:
        continue
      with open(path, encoding="utf-8") as handle:
        for line in handle:
          try:
            row = json.loads(line)
          except json.JSONDecodeError:
            continue
          if row.get("task") != task or row.get("condition") != condition:
            continue
          rung = row.get("rung")
          if rung is None:
            parts = str(row.get("instance_id", "")).split("/")
            rung = parts[1] if len(parts) > 2 else None
          if rung is None:
            continue
          key = (row.get("model"), rung)
          # Capped and overflowed rows are excluded from accuracy but counted,
          # because non-termination hiding inside an accuracy number is exactly
          # what the repo's own rule warns about.
          if _truthy(row.get("hit_cap")) or _truthy(row.get("overflow")):
            dropped[key] += 1
            continue
          predicted = scoring.extract_answer(row.get("response") or "", task)
          hits[key] += predicted is not None and _norm(predicted) == _norm(row.get("gold"))
          totals[key] += 1
  return hits, totals, dropped


def parse_limits(values):
  limits = {}
  for item in values or []:
    if "=" not in item:
      raise SystemExit(f"--reading-limits wants model=tokens, got {item!r}")
    model, tokens = item.split("=", 1)
    limits[model.strip()] = int(tokens)
  return limits


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--responses", nargs="+", required=True)
  parser.add_argument("--task", default="node_degree")
  parser.add_argument("--condition", default="none")
  parser.add_argument("--reading-limits", nargs="*", default=None,
                      help="model=tokens pairs from the retrieval probe")
  parser.add_argument("--out", default=None, help="write the matrix as CSV")
  args = parser.parse_args()

  hits, totals, dropped = load(args.responses, args.task, args.condition)
  limits = parse_limits(args.reading_limits)
  models = sorted({model for model, _ in totals})
  if not models:
    raise SystemExit("no matching rows found")

  rung_tags = [(f"n{n}k{k}", n, k, tokens) for n, k, tokens in ladder.RUNGS]

  print(f"ladder screen: task={args.task} condition={args.condition}")
  if not limits:
    print("NO reading limits supplied -- bands only; readability NOT assessed.\n")
  else:
    print(f"reading limits: {limits}\n")

  header = f"{'rung':<10}{'tokens':>8}  " + "".join(f"{m[:14]:>16}" for m in models)
  print(header)
  print("-" * len(header))
  rows_out = []
  for tag, n, degree, tokens in rung_tags:
    line = f"{tag:<10}{tokens:>8}  "
    for model in models:
      total = totals[(model, tag)]
      if not total:
        line += f"{'':>16}"
        continue
      accuracy = hits[(model, tag)] / total
      band = ladder.classify_band(accuracy)
      limit = limits.get(model)
      readable = limit is None or tokens <= limit
      mark = {"ceiling": "^", "floor": "_", "informative": "*"}[band]
      if band == "informative" and not readable:
        mark = "x"
      line += f"{accuracy:>11.3f} {mark}  "
      rows_out.append({
          "rung": tag, "n": n, "mean_degree": degree, "tokens": tokens,
          "model": model, "accuracy": round(accuracy, 4), "n_rows": total,
          "dropped": dropped[(model, tag)], "band": band,
          "readable": readable, "valid": band == "informative" and readable,
      })
    print(line)
  print("\n  * informative and readable (VALID primer test)")
  print("  x informative but past the reading limit (measures reading, not primers)")
  print("  ^ ceiling   _ floor\n")

  for model in models:
    valid = [r for r in rows_out if r["model"] == model and r["valid"]]
    unreadable = [r for r in rows_out
                  if r["model"] == model and r["band"] == "informative"
                  and not r["readable"]]
    if valid:
      print(f"{model:<22} VALID: {', '.join(r['rung'] for r in valid)}")
    elif unreadable:
      print(f"{model:<22} NO VALID RUNG -- informative only where unreadable "
            f"({', '.join(r['rung'] for r in unreadable)}). This is a finding: "
            f"the task only gets hard past where the model can read.")
    else:
      print(f"{model:<22} NO VALID RUNG -- never leaves ceiling/floor on this ladder.")

  if args.out:
    import csv
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
      writer.writeheader()
      writer.writerows(rows_out)
    print(f"\nwrote {len(rows_out)} rows -> {args.out}")


if __name__ == "__main__":
  main()
