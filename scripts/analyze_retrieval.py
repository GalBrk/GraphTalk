"""Turn retrieval-probe runs into per-model reading limits.

`scripts/build_retrieval_probe.py` states one "Node X has degree Y." fact
among `k` distractor statements and asks for it back -- no graph, nothing to
compute. A model's accuracy here as `k` grows is therefore a pure measure of
*reading*, not of graph reasoning, and the brackets it produces are exactly
the `--reading-limits model=tokens` input `scripts/analyze_ladder.py` needs to
tell "hard because the model can't read the graph" apart from "hard because
the model reads it fine and still gets it wrong" (`docs/ladder-and-rewiring.md`).

Two failure shapes are reported, because they call for different fixes:

  * **length** -- pooled accuracy (all positions/magnitudes) falling as `k`
    grows. `reading_limit` is the largest tested `k` at or below which pooled
    accuracy is still >= `--clean-threshold` (default 0.90); everything past
    it is "degraded" for that model.
  * **lost-in-the-middle** -- accuracy at the middle position (0.5) falling
    behind the two edge positions (0.1, 0.9) specifically, which a pure
    length effect would not produce. `middle_gap` is `edge_accuracy -
    middle_accuracy` at each `k`; `middle_collapse_onset` is the first `k`
    where that gap exceeds `--middle-gap-threshold` (default 0.15).

Token counts are `chars // 4` (`_APPROX_CHARS_PER_TOKEN`, the same heuristic
`scripts/build_prompts.py` and `docs/difficulty-scaling.md` already use for an
overflow warning) applied to a prompt reconstructed with
`build_retrieval_probe.build` -- not a real tokenizer count, since no model's
actual tokenizer is available without `transformers`/GPU. Good enough for a
bracket; do not cite these as exact token counts the way `graphtalk/ladder.py`'s
`RUNGS` are (those came from the real tokenizer).
"""

import argparse
import glob
import json
import re
import statistics
from collections import defaultdict

from graphtalk import scoring
from scripts.build_retrieval_probe import build

_APPROX_CHARS_PER_TOKEN = 4
_INSTANCE = re.compile(r"^retrieval/k(\d+)/pos([\d.]+)/(small|large)/(\d+)$")
_TRAILING = re.compile(r"[\s.]+$")


def _norm(value) -> str:
  return _TRAILING.sub("", str(value).strip())


def _truthy(value) -> bool:
  return str(value).lower() == "true"


def load(patterns, task="node_degree", condition="retrieval"):
  """Accuracy per (model, k, position, magnitude), plus dropped-row counts."""
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
          match = _INSTANCE.match(str(row.get("instance_id", "")))
          if not match:
            continue
          k, position, magnitude, _ = match.groups()
          key = (row.get("model"), int(k), position, magnitude)
          if _truthy(row.get("hit_cap")) or _truthy(row.get("overflow")):
            dropped[key] += 1
            continue
          predicted = scoring.extract_answer(row.get("response") or "", task)
          hits[key] += predicted is not None and _norm(predicted) == _norm(row.get("gold"))
          totals[key] += 1
  return hits, totals, dropped


def approx_tokens_by_k(ks, positions, magnitudes, samples=5, seed=20260909):
  """Median chars//4 over a few reconstructed prompts, per k -- a bracket, not
  a real tokenizer count (see module docstring)."""
  tokens = {}
  for k in ks:
    records = build([k], positions, magnitudes, samples, seed)
    chars = [len(r["prompt"]) for r in records]
    tokens[k] = int(statistics.median(chars)) // _APPROX_CHARS_PER_TOKEN
  return tokens


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--responses", nargs="+", required=True)
  parser.add_argument("--task", default="node_degree")
  parser.add_argument("--clean-threshold", type=float, default=0.90)
  parser.add_argument("--middle-gap-threshold", type=float, default=0.15)
  parser.add_argument("--out", default=None, help="write the (model, k, position, "
                       "magnitude) matrix as CSV")
  args = parser.parse_args()

  hits, totals, dropped = load(args.responses, args.task)
  if not totals:
    raise SystemExit("no matching rows found")

  models = sorted({model for model, *_ in totals})
  ks = sorted({k for _, k, _, _ in totals})
  positions = sorted({p for _, _, p, _ in totals})
  magnitudes = sorted({m for *_, m in totals})
  tokens_by_k = approx_tokens_by_k(ks, [float(p) for p in positions], magnitudes)

  rows_out = []
  print(f"retrieval probe: task={args.task}  ~tokens = chars/{_APPROX_CHARS_PER_TOKEN} (approx, see module docstring)\n")

  for model in models:
    print(f"{'='*74}\n{model}\n{'='*74}")
    pooled_by_k = {}
    edge_by_k = {}
    middle_by_k = {}
    for k in ks:
      pooled_hits = pooled_total = 0
      edge_hits = edge_total = middle_hits = middle_total = 0
      for position in positions:
        for magnitude in magnitudes:
          key = (model, k, position, magnitude)
          total = totals.get(key, 0)
          if not total:
            continue
          accuracy = hits[key] / total
          rows_out.append({
              "model": model, "k": k, "approx_tokens": tokens_by_k[k],
              "position": position, "magnitude": magnitude,
              "accuracy": round(accuracy, 4), "n": total, "dropped": dropped[key],
          })
          pooled_hits += hits[key]; pooled_total += total
          if position == "0.5":
            middle_hits += hits[key]; middle_total += total
          else:
            edge_hits += hits[key]; edge_total += total
      if pooled_total:
        pooled_by_k[k] = pooled_hits / pooled_total
      if edge_total:
        edge_by_k[k] = edge_hits / edge_total
      if middle_total:
        middle_by_k[k] = middle_hits / middle_total

    print(f"{'k':>6} {'~tok':>7} {'pooled':>8} {'edge':>8} {'middle':>8} {'gap':>7}")
    for k in ks:
      if k not in pooled_by_k:
        continue
      gap = (edge_by_k[k] - middle_by_k[k]) if k in edge_by_k and k in middle_by_k else None
      gap_str = f"{gap:+.3f}" if gap is not None else "   n/a"
      print(f"{k:>6} {tokens_by_k[k]:>7} {pooled_by_k[k]:>8.3f} "
            f"{edge_by_k.get(k, float('nan')):>8.3f} {middle_by_k.get(k, float('nan')):>8.3f} {gap_str:>7}")

    clean_ks = [k for k in ks if pooled_by_k.get(k, 0) >= args.clean_threshold]
    reading_limit = tokens_by_k[max(clean_ks)] if clean_ks else None
    collapse_ks = [k for k in ks
                   if k in edge_by_k and k in middle_by_k
                   and edge_by_k[k] - middle_by_k[k] > args.middle_gap_threshold]
    collapse_tokens = tokens_by_k[min(collapse_ks)] if collapse_ks else None

    if reading_limit is not None:
      print(f"\n  reading_limit (pooled accuracy >= {args.clean_threshold}): "
            f"~{reading_limit} tokens (k={max(clean_ks)})")
    else:
      print(f"\n  reading_limit: never reaches {args.clean_threshold} pooled "
            f"accuracy at any tested k -- degraded even at the shortest prompt")
    if collapse_tokens is not None:
      print(f"  middle_collapse_onset (edge-middle gap > {args.middle_gap_threshold}): "
            f"~{collapse_tokens} tokens (k={min(collapse_ks)})")
    else:
      print("  middle_collapse_onset: no lost-in-the-middle gap seen at any tested k")
    print(f"  --reading-limits {model}={reading_limit if reading_limit is not None else 0}\n")

  if args.out:
    import csv
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
      writer.writeheader()
      writer.writerows(rows_out)
    print(f"wrote {len(rows_out)} rows -> {args.out}")


if __name__ == "__main__":
  main()
