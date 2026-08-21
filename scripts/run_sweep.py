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
  args = parser.parse_args()

  spec = models.MODELS[args.model]
  records = load_prompts(args.prompts)
  already = done_keys(args.out)
  todo = [
      r for r in records
      if (r["instance_id"], r["condition"], r["style"]) not in already
  ]
  if args.limit is not None:
    todo = todo[: args.limit]

  print(f"model      {spec.repo_id}", flush=True)
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
      response = hf_backend.generate(
          tokenizer, model, record["prompt"],
          models.budget(spec, record["style"]),
          spec.chat_kwargs,
      )
      handle.write(json.dumps({
          "instance_id": record["instance_id"],
          "task": record["task"],
          "condition": record["condition"],
          "style": record["style"],
          "gold": record["gold"],
          "model": args.model,
          "response": response,
      }) + "\n")
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
