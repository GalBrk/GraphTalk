"""Validates `hf_backend.generate_batch` against the single-stream `generate`.

`graphtalk/hf_backend.py`'s `generate_batch` and `run_sweep.py --batch-size`
were written without a GPU to run them on, and both say so in their docstrings:
batching is the lever that makes Track 2.1's larger `--count`s affordable, but
using it unvalidated risks the one failure mode that does not announce itself.
Decoder-only generation must **left**-pad -- the next token is predicted from
the *last* position of each row's input, so right-padding predicts from a pad
token for every row shorter than the batch's longest. That produces fluent,
well-formed, entirely wrong text rather than an error, and `gemma-4-E4B-it`
defaults to `padding_side='left'` while `Qwen3-8B` defaults to `'right'` -- so
one family passing proves nothing about the other. Both must be checked.

Three modes, matching the three places this runs:

  # login node: reconstruct the prompts the tracked budget references came from
  python scripts/validate_batch_generation.py --build-prompts out.jsonl

  # compute node: two run_sweep.py runs over that file, --batch-size 1 and >1
  # login node: same prompts, two decodings -- do they agree?
  python scripts/validate_batch_generation.py --compare single.jsonl batched.jsonl

  # login node: cross-check either run against the tracked single-stream text
  python scripts/validate_batch_generation.py \
      --reference analysis/budget-qwen3-8b.jsonl --against single.jsonl

**Why `--compare` is the real test and `--reference` only a cross-check.**
`--compare` holds everything but batching fixed: same node, same env, same
weights, same prompts, in the same job. A disagreement there is batching's
fault and nothing else's. The tracked `analysis/budget-*.jsonl` responses were
generated on another day against a prompt file that has since been reworded
(`prompts.original-wording.jsonl` is the record of that), so a `--reference`
mismatch is evidence of drift somewhere in the pipeline, not necessarily of a
padding bug -- useful to know, but not the thing gating a real sweep.

**Which 24 prompts, and how they are recovered.** The budget files record
`task`/`condition`/`style`/`gold` but not `instance_id`, and that tuple is
*not* unique -- ('edge_existence', 'none', 'zero_shot', 'No.') matches 18 rows
of `prompts.jsonl`, so joining on it recovers only half the set unambiguously.
Their row order does identify them exactly, though: six tasks in prompt-file
order, the first two instances of each, `all` then `none` within each instance.
That reconstruction reproduces all 24 golds exactly for both families, which is
what `_reconstruct` asserts rather than assumes.

Exit status is 0 on pass and 1 on failure, so a cluster job can run the compare
itself and a follow-up sweep can chain on `--dependency=afterok` without a human
in between.
"""

import argparse
import json
import os
import sys

# Conditions and per-task instance count behind the tracked budget references;
# see the module docstring for how these were recovered.
_BUDGET_CONDITIONS = ("all", "none")
_BUDGET_INSTANCES = 2
_STYLE = "zero_shot"

# A wrong padding side corrupts every row, so exact agreement collapses to
# roughly zero -- it does not land just under a threshold. The margin here is
# for the floating-point non-associativity of batched vs. unbatched matmuls,
# which can flip a token at a near-tie and send one row down another path.
_MIN_EXACT_FRACTION = 0.75
# A row that diverges from fp drift agrees on a long prefix first; one that
# diverges from bad padding is wrong from its first token.
_MIN_COMMON_PREFIX = 20


def _load(path: str) -> list[dict]:
  with open(path) as handle:
    return [json.loads(line) for line in handle if line.strip()]


def _key(row: dict) -> tuple:
  return (row["instance_id"], row["condition"], row["style"])


def _reconstruct(prompts_path: str) -> list[dict]:
  """The 24 prompt records the budget references were generated from.

  Ordered exactly as the budget files are, so a reader can put the two side by
  side, and verified against both families' recorded golds before being
  returned -- if `prompts.jsonl` is ever rebuilt from a different split or
  ordering, this raises rather than silently validating batching against a
  different 24 prompts than the references describe.
  """
  records = _load(prompts_path)
  by_key = {(r["instance_id"], r["condition"], r["style"]): r for r in records}

  tasks = []
  for record in records:
    if record["task"] not in tasks:
      tasks.append(record["task"])

  selected = []
  for task in tasks:
    for index in range(_BUDGET_INSTANCES):
      for condition in _BUDGET_CONDITIONS:
        key = (f"{task}/{index}", condition, _STYLE)
        if key not in by_key:
          raise SystemExit(f"FATAL: {prompts_path} has no row for {key}")
        selected.append(by_key[key])
  return selected


def _verify_against_budgets(selected: list[dict], analysis_dir: str) -> None:
  """Cross-checks the reconstruction against every tracked budget file's golds."""
  for family in ("gemma4-e4b", "qwen3-8b"):
    path = os.path.join(analysis_dir, f"budget-{family}.jsonl")
    if not os.path.exists(path):
      print(f"  (no {path}; skipping its gold cross-check)")
      continue
    rows = _load(path)
    if len(rows) != len(selected):
      raise SystemExit(
          f"FATAL: {path} has {len(rows)} rows, reconstruction has "
          f"{len(selected)}; the budget files are not the 24 rows this "
          f"script assumes.")
    for index, (row, record) in enumerate(zip(rows, selected)):
      if (row["task"], row["condition"], row["gold"]) != (
          record["task"], record["condition"], record["gold"]):
        raise SystemExit(
            f"FATAL: {path} row {index} is "
            f"({row['task']}, {row['condition']}, {row['gold']!r}) but the "
            f"reconstruction gives "
            f"({record['task']}, {record['condition']}, {record['gold']!r}). "
            f"The row-order recovery in this script's docstring no longer "
            f"holds; do not trust a comparison built on it.")
    print(f"  {path}: all {len(rows)} golds match the reconstruction")


def _common_prefix(left: str, right: str) -> int:
  limit = min(len(left), len(right))
  for index in range(limit):
    if left[index] != right[index]:
      return index
  return limit


def _report(pairs: list[tuple], label_a: str, label_b: str) -> bool:
  """Prints per-row agreement and returns whether the run passes.

  Prints every mismatch rather than only counting them: the length of the
  agreeing prefix is what distinguishes "one token flipped at a near-tie, then
  the texts diverge" from "wrong from the first token", and that distinction is
  the whole point of the check.
  """
  exact = 0
  short_prefix = []
  for key, text_a, text_b in pairs:
    if text_a == text_b:
      exact += 1
      continue
    shared = _common_prefix(text_a, text_b)
    if shared < _MIN_COMMON_PREFIX:
      short_prefix.append(key)
    print(f"  DIFF {key}: agree on first {shared} chars")
    print(f"    {label_a}: {text_a[:160]!r}")
    print(f"    {label_b}: {text_b[:160]!r}")

  total = len(pairs)
  fraction = exact / total if total else 0.0
  print(f"\n  {exact}/{total} exact ({fraction:.1%})")

  passed = True
  if fraction < _MIN_EXACT_FRACTION:
    print(f"  FAIL: below the {_MIN_EXACT_FRACTION:.0%} exact-match floor.")
    passed = False
  if short_prefix:
    print(f"  FAIL: {len(short_prefix)} row(s) diverge within the first "
          f"{_MIN_COMMON_PREFIX} chars, which is what a wrong padding side "
          f"looks like: {short_prefix[:5]}")
    passed = False
  return passed


def _pair_runs(path_a: str, path_b: str) -> list[tuple]:
  rows_a = {_key(r): r for r in _load(path_a)}
  rows_b = {_key(r): r for r in _load(path_b)}
  shared = sorted(set(rows_a) & set(rows_b))
  only_a, only_b = sorted(set(rows_a) - set(rows_b)), sorted(set(rows_b) - set(rows_a))
  if only_a or only_b:
    # Not fatal on its own -- a preempted job leaves a short file -- but it
    # silently shrinks the sample the verdict rests on, so it is stated.
    print(f"  WARNING: {len(only_a)} row(s) only in {path_a}, "
          f"{len(only_b)} only in {path_b}")
  return [(key, rows_a[key]["response"], rows_b[key]["response"])
          for key in shared]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--build-prompts", metavar="OUT",
                      help="write the 24 budget-reference prompts to OUT")
  parser.add_argument("--prompts", default="prompts.jsonl",
                      help="the prompt file to reconstruct them from")
  parser.add_argument("--analysis-dir", default="analysis")
  parser.add_argument("--compare", nargs=2, metavar=("SINGLE", "BATCHED"),
                      help="compare two run_sweep.py outputs over the same prompts")
  parser.add_argument("--reference", metavar="BUDGET_JSONL",
                      help="a tracked analysis/budget-*.jsonl to cross-check against")
  parser.add_argument("--against", metavar="RUN_JSONL",
                      help="the run_sweep.py output to compare with --reference")
  args = parser.parse_args()

  if not (args.build_prompts or args.compare or args.reference):
    parser.error("pass one of --build-prompts, --compare, or --reference")

  if args.build_prompts:
    selected = _reconstruct(args.prompts)
    print(f"reconstructed {len(selected)} prompts from {args.prompts}")
    _verify_against_budgets(selected, args.analysis_dir)
    with open(args.build_prompts, "w") as handle:
      for record in selected:
        handle.write(json.dumps(record) + "\n")
    print(f"wrote {args.build_prompts}")

  passed = True

  if args.compare:
    single, batched = args.compare
    print(f"\n=== batched vs single-stream: {single} vs {batched} ===")
    passed &= _report(_pair_runs(single, batched), "single ", "batched")

  if args.reference:
    if not args.against:
      parser.error("--reference needs --against <run_sweep output>")
    print(f"\n=== tracked reference: {args.reference} vs {args.against} ===")
    # The budget file carries no instance_id, so it is paired by the same row
    # order --build-prompts wrote, not by key.
    reference_rows = _load(args.reference)
    run_rows = {_key(r): r for r in _load(args.against)}
    selected = _reconstruct(args.prompts)
    pairs = []
    for record, row in zip(selected, reference_rows):
      key = (record["instance_id"], record["condition"], record["style"])
      if key in run_rows:
        pairs.append((key, row["response"], run_rows[key]["response"]))
    # Reported, never gating: see the module docstring on prompt rewording.
    _report(pairs, "tracked", "this run")
    print("  (cross-check only -- does not affect exit status)")

  print("\nPASS" if passed else "\nFAIL")
  sys.exit(0 if passed else 1)


if __name__ == "__main__":
  main()
