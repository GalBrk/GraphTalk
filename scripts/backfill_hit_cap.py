"""Backfills `n_new_tokens` and `hit_cap` onto rows generated before run_sweep recorded them.

Non-termination was originally known only from `analysis/truncated_keys.json`, a
hand-curated list. Since the 2026-08-29 re-run `scripts/run_sweep.py` records the
fact per row, but only rows generated from that point carry it -- which left the
sweep measuring one quantity two ways, and a comparison between a regenerated
condition and an original one confounded with the instrument. This puts every row
on the same instrument without regenerating anything.

**The method, and why it is trustworthy here.** A response is re-tokenized with its
own model's tokenizer and compared against the budget from `graphtalk/models.py`.
That undercounts slightly, because `run_sweep` decodes with
`skip_special_tokens=True` and the trailing EOS is gone from the text -- so a row
that stopped naturally loses a token, while a row cut off at the cap never had one.
Measured against the 2,880 rows that carry the generator's own count, capped rows
land 0-1 tokens below budget and the nearest non-capped row is 29 below. The rule
`retokenized >= budget - 1` reproduces the recorded flag on **45/45 capped and
2,835/2,835 non-capped rows, with no misses and no false alarms**.

Backfilled rows are marked `token_count_source: "retokenized"`. Rows that already
carry the generator's count are left untouched, so ground truth is never
overwritten by an estimate.

The tokenizer lives here rather than in `graphtalk/analysis.py` on purpose: that
module is deliberately importable without `transformers`, and says so.

  HF_HOME=... PYTHONPATH=. python scripts/backfill_hit_cap.py --runs runs

**`--verify-recorded`**: a second, read-only mode that answers a question the
backfill above never touches. `filler`'s rows from the 2026-08-29 prompt-rewording
re-run already carry a generator-recorded `hit_cap` -- exactly the condition
`backfill` skips ("never overwrite... ground truth") -- so the agreement between
`filler`'s generator-recorded flag and an independent retokenization has never
actually been checked, unlike the other six conditions (see
docs/sweep-findings.md, "The filler instrument confound is reduced, not fully
closed"). This mode retokenizes every already-recorded row anyway, using the exact
same tokenizer/budget/`_EOS_SLACK` logic as `backfill` (kept in one place so the two
modes can't silently drift apart), and reports whether the recorded flag agrees --
without writing anything back to disk.

  HF_HOME=... PYTHONPATH=. python scripts/backfill_hit_cap.py --runs runs --verify-recorded
"""

import argparse
import collections
import glob
import json
import os

import transformers

from graphtalk import models

# Slack between the re-tokenized count and the budget.
#
# 1 would cover the dropped EOS, and every value from 1 to 25 scores 100% against
# the 2,880 rows carrying the generator's own count -- that set does not
# discriminate, because its nearest non-capped row sits 29 tokens below budget.
# The older data does discriminate. `gemma4-12b-think edge_count/8` re-tokenizes to
# 8190 of 8192 and is plainly truncated (it stops mid-enumeration, "- Node 3: 5"),
# so a slack of 1 misses it; `edge_count/25` sits 25 below and ends with a complete
# "A: 6", so anything at or above 25 would wrongly flag it. 5 is comfortably inside
# both bounds rather than on either edge.
_EOS_SLACK = 5


class _Tokenizers:
  """One tokenizer per checkpoint, shared across the `-think` / plain pair."""

  def __init__(self):
    self._by_repo = {}

  def count(self, model_key: str, text: str) -> int:
    repo = models.MODELS[model_key].repo_id
    if repo not in self._by_repo:
      self._by_repo[repo] = transformers.AutoTokenizer.from_pretrained(repo)
    return len(self._by_repo[repo](text, add_special_tokens=False)["input_ids"])


def backfill(path: str, tokenizers: _Tokenizers) -> tuple[int, int]:
  """Rewrites one run file in place. Returns (rows written, rows backfilled)."""
  with open(path) as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
  if not rows:
    return 0, 0

  filled = 0
  for row in rows:
    if row.get("hit_cap") is not None:
      continue                      # the generator's own count; never overwrite
    budget = models.budget(models.MODELS[row["model"]], row["style"])
    count = tokenizers.count(row["model"], row["response"])
    row["n_new_tokens"] = count
    row["hit_cap"] = count >= budget - _EOS_SLACK
    row["token_count_source"] = "retokenized"
    filled += 1

  tmp = path + ".tmp"
  with open(tmp, "w") as handle:
    for row in rows:
      handle.write(json.dumps(row) + "\n")
  os.replace(tmp, path)             # atomic: a crash cannot truncate the arm
  return len(rows), filled


def verify_recorded(
    path: str, tokenizers: _Tokenizers
) -> tuple[collections.Counter, collections.Counter, list]:
  """Read-only counterpart to `backfill`: retokenizes rows `backfill` skips
  (`hit_cap` already recorded) and checks the recorded flag against the
  retokenized one, without writing anything back to `path`.

  Returns `(checked_by_condition, agreeing_by_condition, disagreements)` --
  the first two let a caller merge per-condition tallies across many files
  without re-reading them; `disagreements` is a list of `(path, instance_id,
  condition, style, recorded, retokenized, n_new_tokens)` tuples, everything
  needed to look a mismatched row up by hand.
  """
  with open(path) as handle:
    rows = [json.loads(line) for line in handle if line.strip()]

  checked_by_condition = collections.Counter()
  agreeing_by_condition = collections.Counter()
  disagreements = []
  for row in rows:
    recorded = row.get("hit_cap")
    if recorded is None:
      continue                        # backfill's job, not this mode's
    condition = row["condition"]
    budget = models.budget(models.MODELS[row["model"]], row["style"])
    count = tokenizers.count(row["model"], row["response"])
    retokenized = count >= budget - _EOS_SLACK
    checked_by_condition[condition] += 1
    if retokenized == recorded:
      agreeing_by_condition[condition] += 1
    else:
      disagreements.append((
          path, row["instance_id"], condition, row["style"],
          recorded, retokenized, count,
      ))
  return checked_by_condition, agreeing_by_condition, disagreements


def _run_verify_recorded(runs_dir: str) -> None:
  """`--verify-recorded`'s driver: prints a per-condition agreement
  breakdown and every disagreement found, across every `runs/*.jsonl` file.
  Never writes to any file. Exits non-zero on any disagreement, so this is
  scriptable (e.g. CI) rather than needing a human to read the printout.
  """
  tokenizers = _Tokenizers()
  checked = collections.Counter()
  agreeing = collections.Counter()
  disagreements = []
  for path in sorted(glob.glob(os.path.join(runs_dir, "*.jsonl"))):
    if "smoke-" in path:
      continue
    file_checked, file_agreeing, file_disagreements = verify_recorded(
        path, tokenizers
    )
    checked.update(file_checked)
    agreeing.update(file_agreeing)
    disagreements.extend(file_disagreements)

  print("condition        checked  agreeing")
  for condition in sorted(checked):
    print(f"  {condition:<14} {checked[condition]:>7}  {agreeing[condition]:>8}")
  print(f"\n{len(disagreements)} disagreements found")
  for dis_path, instance_id, condition, style, recorded, retokenized, count \
      in disagreements:
    print(f"  {os.path.basename(dis_path)} {instance_id} {condition}/{style}: "
          f"recorded={recorded} retokenized={retokenized} (n_new_tokens={count})")

  if disagreements:
    raise SystemExit(1)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--runs", default="runs")
  parser.add_argument("--truncated-keys", default="analysis/truncated_keys.json",
                      help="cross-checked against the backfilled flag, not used to set it")
  parser.add_argument("--verify-recorded", action="store_true",
                       help="read-only: check already-recorded hit_cap rows "
                            "(e.g. filler's) against a fresh retokenization "
                            "instead of backfilling missing ones -- see the "
                            "module docstring")
  args = parser.parse_args()

  if args.verify_recorded:
    _run_verify_recorded(args.runs)
    return

  tokenizers = _Tokenizers()
  total = filled = 0
  for path in sorted(glob.glob(os.path.join(args.runs, "*.jsonl"))):
    if "smoke-" in path:
      continue
    rows, new = backfill(path, tokenizers)
    total += rows
    filled += new
    if new:
      print(f"  {os.path.basename(path):<45} {new:>5} backfilled of {rows}")
  print(f"\n{filled} rows backfilled, {total - filled} already carried a recorded count")

  # Cross-check against the hand-curated file. This is the first time the two can
  # be compared on the same rows, and a disagreement is worth knowing about.
  if os.path.exists(args.truncated_keys):
    with open(args.truncated_keys) as handle:
      labelled = {(m, i, c, s)
                  for m, keys in json.load(handle).items() for i, c, s in keys}
    flagged = set()
    for path in glob.glob(os.path.join(args.runs, "*.jsonl")):
      if "smoke-" in path or ".redo.shard" in path:
        continue
      with open(path) as handle:
        for line in handle:
          if not line.strip():
            continue
          row = json.loads(line)
          if row.get("hit_cap"):
            flagged.add((row["model"], row["instance_id"], row["condition"],
                         row["style"]))
    agree = labelled & flagged
    print(f"\ncross-check against {os.path.basename(args.truncated_keys)}:")
    print(f"  labelled {len(labelled)}, flagged {len(flagged)}, agreeing {len(agree)}")
    print(f"  labelled but not flagged: {len(labelled - agree)}")
    print(f"  flagged but not labelled: {len(flagged - agree)}")


if __name__ == "__main__":
  main()
