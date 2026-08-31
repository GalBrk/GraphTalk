"""Objective 4: pulls a stratified sample of failure cases from the canonical
sweep frame for manual visual inspection, with full response text and the
underlying graph's size joined back in. No GPU needed.

Reads the CSV `scripts/build_sweep_frame.py` wrote -- does not re-score.
`response` text is re-joined from `runs/*.jsonl` and `nodes`/`edges` from
`prompts.jsonl`/`prompts_zero_shot.jsonl` on `(instance_id, condition,
style[, model]), node_naming`, since neither lives in the canonical frame
(see `graphtalk/analysis.py` for why). `node_naming` is part of both join
keys so a `--responses`/`--prompts` glob that also catches the other
scheme's files simply matches nothing for those rows, rather than joining in
the wrong text. `--frame` must carry a single scheme -- raises otherwise.

  PYTHONPATH=. .venv/bin/python scripts/sample_failures.py \
      --frame analysis/sweep_frame.csv --responses runs/*.jsonl \
      --prompts prompts.jsonl prompts_zero_shot.jsonl \
      --out analysis/failure_sample.csv
"""

import argparse
import glob
import json

import pandas as pd

from graphtalk import analysis
from scripts import score_sweep

# For quick spreadsheet scanning. The tail matters as much as the head: for a
# non-terminating row the diagnostic content (the repeated re-verification,
# "Wait, let me re-check Node X's connections one more time...") is at the
# *end* of the truncated text, not the start.
_PREVIEW_CHARS = 400


def _load_paths(patterns: list[str]) -> list[str]:
  # Sorted: `glob.glob` returns filesystem order, so without this the row order
  # of the exported CSV depends on how the directory happens to be laid out. It
  # changed under a `git checkout`, which made the committed artefact differ from
  # a fresh rebuild on content that was identical as a set -- a reproducibility
  # claim that held only by luck.
  paths = sorted(p for pattern in patterns for p in glob.glob(pattern))
  return [p for p in paths if not analysis.is_excluded(p)]


def _load_responses(paths: list[str]) -> dict[tuple, str]:
  records = score_sweep.load(paths)
  return {
      (r["model"], r["instance_id"], r["condition"], r["style"],
       r.get("node_naming", "integer")): r["response"]
      for r in records
  }


def _load_prompt_sizes(paths: list[str]) -> dict[tuple, tuple]:
  sizes = {}
  for path in paths:
    with open(path) as handle:
      for line in handle:
        if not line.strip():
          continue
        row = json.loads(line)
        key = (row["instance_id"], row["condition"], row["style"],
               row.get("node_naming", "integer"))
        sizes[key] = (row["nodes"], row["edges"])
  return sizes


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frame", required=True,
                       help="canonical CSV from scripts/build_sweep_frame.py")
  parser.add_argument("--responses", nargs="+", required=True)
  parser.add_argument("--prompts", nargs="+", default=[],
                       help="prompts.jsonl / prompts_zero_shot.jsonl, for the "
                            "nodes/edges columns")
  parser.add_argument("--n-per-stratum", type=int, default=3,
                       help="max sampled rows per (model, failure_type)")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--out", default=None,
                       help="default analysis/failure_sample.csv, or "
                            "analysis/failure_sample.<scheme>.csv for a "
                            "non-integer node_naming scheme -- see "
                            "README.md#node-naming")
  args = parser.parse_args()

  frame = pd.read_csv(args.frame)
  scheme = analysis.frame_node_naming(frame)
  sample = analysis.sample_failures(
      frame, n_per_stratum=args.n_per_stratum, seed=args.seed
  )
  if sample.empty:
    print("no failure rows to sample")
    return

  responses = _load_responses(_load_paths(args.responses))
  sizes = _load_prompt_sizes(args.prompts) if args.prompts else {}

  full, previews, tails, lengths, nodes, edges = [], [], [], [], [], []
  for _, row in sample.iterrows():
    naming = row["node_naming"] if "node_naming" in row else "integer"
    text = responses.get(
        (row["model"], row["instance_id"], row["condition"], row["style"], naming), ""
    )
    full.append(text)
    previews.append(text[:_PREVIEW_CHARS])
    tails.append(text[-_PREVIEW_CHARS:])
    lengths.append(len(text))
    size = sizes.get((row["instance_id"], row["condition"], row["style"], naming))
    nodes.append(size[0] if size else None)
    edges.append(size[1] if size else None)

  sample = sample.assign(
      response_full=full,
      response_preview=previews,
      response_tail_preview=tails,
      response_len_chars=lengths,
      nodes=nodes,
      edges=edges,
  )
  out = args.out or analysis.tagged_path("analysis/failure_sample.csv", scheme)
  sample.to_csv(out, index=False)
  print(f"wrote {len(sample)} sampled failure rows to {out} (node_naming: {scheme})")
  print(sample["failure_type"].value_counts().to_string())


if __name__ == "__main__":
  main()
