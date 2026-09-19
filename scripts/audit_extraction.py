"""Step 0 of the paper-revision plan: does `extract_answer` ever read the wrong
side of a `<think>...</think>` boundary?

Nothing in the codebase strips `<think>` tags before scoring (verified by
grepping for the literal tag across graphtalk/ and scripts/ -- no hits). Every
thinking-arm response is scored as raw text, `<think>` markup and all.
`extract_answer` picks the LAST "answer is/answer:" marker in the whole text,
so a stray "answer is" mentioned mid-reasoning is safe as long as it's not the
textually-last one. But it is worth checking directly rather than trusting
that argument, because three shapes could break it:

  1. `no_close`: the response never emits `</think>` at all (mid-thought
     truncation). Any extraction here is reading unfinished reasoning.
  2. `empty_after_close`: `</think>` appears but nothing (or only whitespace)
     follows it -- the model closed thinking and then generation stopped.
  3. `marker_before_close`: `has_answer_marker` is True, but the LAST marker
     match sits before the LAST `</think>` -- i.e. the extractor's chosen
     marker is inside the reasoning trace, not the stated conclusion after it.
     This is the one case that could silently misscore a row that has a
     perfectly good post-think answer sitting after it, if that answer never
     used the word "answer".

Usage:
  python scripts/audit_extraction.py runs/qwen3-1.7b-think.densfull40hi.shard*.jsonl
"""

import argparse
import glob
import json
import random
import sys

sys.path.insert(0, ".")
from graphtalk import scoring


def classify(text: str) -> str:
  close = text.rfind("</think>")
  if "<think>" not in text:
    return "no_think_tag"
  if close == -1:
    return "no_close"
  after = text[close + len("</think>"):].strip()
  if not after:
    return "empty_after_close"
  if scoring.has_answer_marker(text):
    marker_pos = text.rfind("answer")  # cheap proxy; _MARKER is anchored on this word
    if marker_pos != -1 and marker_pos < close:
      return "marker_before_close"
  return "clean"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("responses", nargs="+")
  ap.add_argument("--sample", type=int, default=50)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  files = []
  for pattern in args.responses:
    files.extend(sorted(glob.glob(pattern)))

  rows = []
  for path in files:
    with open(path, encoding="utf-8") as f:
      for line in f:
        rows.append(json.loads(line))

  think_rows = [r for r in rows if "<think>" in r.get("response", "")]
  if not think_rows:
    print(f"No <think> rows found across {len(files)} file(s) / {len(rows)} row(s).")
    return

  counts = {}
  flagged = {"no_close": [], "empty_after_close": [], "marker_before_close": []}
  for r in think_rows:
    label = classify(r["response"])
    counts[label] = counts.get(label, 0) + 1
    if label in flagged:
      flagged[label].append(r)

  print(f"{len(think_rows)} thinking-arm rows across {len(files)} file(s).")
  for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {label:22s} {n:6d}  ({100*n/len(think_rows):5.1f}%)")

  # hit_cap cross-check: no_close/empty_after_close should track hit_cap.
  hit_cap_true = sum(1 for r in think_rows if r.get("hit_cap"))
  no_close_not_capped = sum(1 for r in flagged["no_close"] if not r.get("hit_cap"))
  print(f"\nhit_cap=True: {hit_cap_true} of {len(think_rows)}")
  print(f"no_close but hit_cap=False (real anomaly, not just budget truncation): "
        f"{no_close_not_capped}")

  # The real ground-truth check, run over EVERY row with a non-empty
  # post-</think> tail (not just the noisy 'marker_before_close' heuristic
  # bucket, which just detects the word "answer" occurring anywhere in the
  # think trace -- true on 85% of rows regardless of risk): does
  # extract_answer on the full text (what every scoring script actually
  # calls) agree with extract_answer restricted to only the text after
  # </think> (what a human reading just the stated conclusion would get)?
  checked = 0
  mismatches = []
  for r in think_rows:
    text = r["response"]
    close = text.rfind("</think>")
    if close == -1:
      continue
    after = text[close + len("</think>"):].strip()
    if not after:
      continue
    checked += 1
    full = scoring.extract_answer(text, r["task"])
    post = scoring.extract_answer(after, r["task"])
    if full != post:
      mismatches.append((r, full, post, after))

  print(f"\nGround-truth check: full-text vs. post-</think>-only extraction, "
        f"{checked} rows with non-empty post-think content.")
  print(f"  mismatches: {len(mismatches)} ({100*len(mismatches)/max(checked,1):.2f}%)")

  rng = random.Random(args.seed)
  k = min(args.sample, len(mismatches))
  if k:
    print(f"\n=== {k} of {len(mismatches)} mismatches ===")
    for r, full, post, after in rng.sample(mismatches, k):
      print(f"\n[{r['instance_id']}] task={r['task']} cond={r['condition']} "
            f"gold={r['gold']!r} hit_cap={r.get('hit_cap')}")
      print(f"  full-text extraction:          {full!r}")
      print(f"  post-</think>-only extraction: {post!r}")
      print(f"  post-</think> tail: {after[:300]!r}")


if __name__ == "__main__":
  main()
