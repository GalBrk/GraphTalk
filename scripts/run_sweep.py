"""Stage 2: run one model over a prompt file. The only stage that needs a GPU.

Writes each response to the output file as it is produced and skips work already
present on restart. That is not tidiness -- the cluster's default partition is
`killable`, meaning preemptible: a higher-priority job can stop this one at any
point and Slurm requeues it. A run that only wrote its results at the end would
lose hours of generation every time that happened.

  python scripts/run_sweep.py --model gemma4-12b \\
      --prompts prompts.jsonl --out runs/gemma4-12b.jsonl
"""

import argparse
import json
import os
import time

from graphtalk import hf_backend
from graphtalk import models


def load_prompts(path: str) -> list[dict]:
  with open(path) as handle:
    return [json.loads(line) for line in handle if line.strip()]


def done_keys(path: str) -> set:
  """Keys already generated, so a requeued job resumes instead of restarting.

  Reads defensively: a job killed mid-write leaves a truncated final line, and
  that one unparseable line must not abort the resume. Anything unreadable is
  simply regenerated.
  """
  if not os.path.exists(path):
    return set()
  keys = set()
  with open(path) as handle:
    for line in handle:
      try:
        record = json.loads(line)
      except json.JSONDecodeError:
        continue
      keys.add((record["instance_id"], record["condition"], record["style"]))
  return keys


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", required=True, choices=sorted(models.MODELS))
  parser.add_argument("--prompts", default="prompts.jsonl")
  parser.add_argument("--out", required=True)
  parser.add_argument("--limit", type=int, default=None,
                      help="stop after this many generations, for smoke tests")
  parser.add_argument("--max-new-tokens", type=int, default=None,
                      help="override the spec's budget; for regenerating rows "
                           "that were truncated at a smaller one")
  parser.add_argument("--shard", type=int, default=0,
                      help="which shard of the prompt file this job generates")
  parser.add_argument("--num-shards", type=int, default=1,
                      help="split the prompts across this many concurrent jobs")
  args = parser.parse_args()

  if not 0 <= args.shard < args.num_shards:
    parser.error(f"--shard must be in [0, {args.num_shards})")

  spec = models.MODELS[args.model]
  records = load_prompts(args.prompts)
  # Stride, not contiguous blocks. The prompt file is ordered by task, so a block
  # split would hand one shard every `edge_count` row -- the task that runs an
  # order of magnitude longer than the rest -- and that shard would still be
  # generating long after the others had finished. Striding gives every shard the
  # same task mix, so they finish together.
  if args.num_shards > 1:
    records = records[args.shard::args.num_shards]
  already = done_keys(args.out)
  todo = [
      r for r in records
      if (r["instance_id"], r["condition"], r["style"]) not in already
  ]
  if args.limit is not None:
    todo = todo[: args.limit]

  print(f"model      {spec.repo_id}", flush=True)
  if args.num_shards > 1:
    print(f"shard      {args.shard} of {args.num_shards}", flush=True)
  print(f"prompts    {len(records)} total, {len(already)} done, {len(todo)} to run",
        flush=True)
  if not todo:
    print("nothing to do", flush=True)
    return

  tokenizer, model = hf_backend.load(spec)
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  started = time.time()
  with open(args.out, "a") as handle:
    for index, record in enumerate(todo, 1):
      completion = hf_backend.generate(
          tokenizer, model, record["prompt"],
          args.max_new_tokens or models.budget(spec, record["style"]),
          spec.chat_kwargs,
      )
      # `n_new_tokens`/`hit_cap` are new as of the prompt-rewording re-run; rows
      # generated before it do not carry them, so anything reading these must
      # treat absence as "unknown" and fall back to
      # `analysis/truncated_keys.json` -- see `graphtalk/analysis.py`.
      row = {
          "instance_id": record["instance_id"],
          "task": record["task"],
          "condition": record["condition"],
          "style": record["style"],
          "gold": record["gold"],
          "model": args.model,
          "response": completion.text,
          "n_new_tokens": completion.n_new_tokens,
          "hit_cap": completion.hit_cap,
      }
      # The prompt file's node-naming scheme has to travel with the response.
      # Everything downstream keys off this field on the *response* row --
      # `scripts/score_sweep.py` converts GoT names back to integers before
      # scoring, and `graphtalk/analysis.py` reads it for the frame's column and
      # for its mixed-scheme guard. All three default a missing field to
      # `integer`, so dropping it here does not raise: a GoT run would simply be
      # scored against integer gold and come out near-zero, with the guard unable
      # to fire because absence is not a conflict. Copied rather than defaulted,
      # so an integer prompt file stays byte-identical to what it wrote before.
      if "node_naming" in record:
        row["node_naming"] = record["node_naming"]
      handle.write(json.dumps(row) + "\n")
      # Flushed every row so a preemption loses at most the row in flight.
      handle.flush()
      if index % 25 == 0 or index == len(todo):
        rate = index / (time.time() - started)
        remaining = (len(todo) - index) / rate if rate else 0
        print(f"  {index}/{len(todo)}  {rate:.2f} gen/s  "
              f"~{remaining/60:.1f} min left", flush=True)

  print(f"wrote {args.out} in {(time.time()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
  main()
