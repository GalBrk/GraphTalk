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
import collections
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


def _group_by_budget(records: list[dict], spec, max_new_tokens_override) -> dict:
  """Buckets `records` by the `max_new_tokens` each one actually needs.

  A batched `model.generate` call takes one `max_new_tokens` for the whole
  batch, so rows needing different budgets (different `style`s, or a
  `--max-new-tokens` override applied selectively) can't share a batch --
  see `models.budget`. Today's data is `style="zero_shot"` only (post-purge),
  so this always yields a single group in practice, but stays correct if
  that ever changes rather than silently mixing budgets.
  """
  groups = collections.defaultdict(list)
  for record in records:
    budget = max_new_tokens_override or models.budget(spec, record["style"])
    groups[budget].append(record)
  return groups


def _chunk(items: list, size: int) -> list[list]:
  return [items[i:i + size] for i in range(0, len(items), size)]


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
  parser.add_argument("--batch-size", type=int, default=1,
                      help="rows generated per forward pass (Track 2.3). "
                           "Default 1 keeps today's exact single-stream code "
                           "path (hf_backend.generate); >1 routes through "
                           "hf_backend.generate_batch, which is NOT YET "
                           "VALIDATED against a GPU -- see its docstring. "
                           "Do not use >1 for a real sweep before that "
                           "validation has been done.")
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

  # One list of (budget, batch) work units, computed up front so the
  # progress counter below can report against the true total regardless of
  # how budget-grouping split it up.
  work: list[tuple[int, list[dict]]] = []
  for budget, group in _group_by_budget(todo, spec, args.max_new_tokens).items():
    if args.batch_size > 1:
      # Sorted by prompt length so a batch's members are close in length --
      # `model.generate` runs a batch until its *longest* member finishes
      # (see hf_backend.generate_batch's docstring), so grouping similar
      # lengths together avoids padding every short row out to one long
      # outlier's length. Changes output file row order relative to the
      # prompt file; harmless, since `done_keys` resumes by key, not order.
      group = sorted(group, key=lambda r: len(r["prompt"]))
    for batch in _chunk(group, args.batch_size):
      work.append((budget, batch))

  started = time.time()
  done_count = 0
  with open(args.out, "a") as handle:
    for budget, batch in work:
      if args.batch_size == 1:
        # Unchanged from before batching existed: `generate_batch` at
        # batch size 1 should be equivalent, but this path is the one
        # every row on disk so far was generated with, so it stays the
        # default rather than being replaced by an unvalidated one.
        completions = [hf_backend.generate(
            tokenizer, model, batch[0]["prompt"], budget, spec.chat_kwargs,
        )]
      else:
        completions = hf_backend.generate_batch(
            tokenizer, model, [r["prompt"] for r in batch], budget,
            spec.chat_kwargs,
        )
      for record, completion in zip(batch, completions):
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
      # Flushed once per *batch*, not per row: at batch_size 1 this is
      # exactly the old per-row flush (a preemption loses at most the row
      # in flight); at batch_size > 1 a preemption can lose up to one
      # batch, traded deliberately for the forward-pass savings batching
      # exists for.
      handle.flush()
      done_count += len(batch)
      if done_count % 25 < len(batch) or done_count == len(todo):
        rate = done_count / (time.time() - started)
        remaining = (len(todo) - done_count) / rate if rate else 0
        print(f"  {done_count}/{len(todo)}  {rate:.2f} gen/s  "
              f"~{remaining/60:.1f} min left", flush=True)

  print(f"wrote {args.out} in {(time.time()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
  main()
