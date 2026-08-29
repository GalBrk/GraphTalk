"""Builds the canonical scored table over the whole tracked sweep (main sweep
+ thinking arm) and prints the objective 1/2/3/5 summary tables: the
Zeroshot/none baseline, the cross-condition comparison, the thinking-arm
non-termination breakdown, and an overall failure_type tally. No GPU needed.

Rows that are not part of the tracked sweep live in `runs/archive/` and are
excluded by directory, so a plain `runs/*.jsonl` glob does not reach them.

  PYTHONPATH=. .venv/bin/python scripts/build_sweep_frame.py \
      --responses runs/*.jsonl --shortcuts shortcuts.json \
      --truncated-keys analysis/truncated_keys.json \
      --out analysis/sweep_frame.csv

The written CSV is the input `scripts/sample_failures.py` reads back in to
pull the manual-inspection sample (objective 4) without re-scoring.
"""

import argparse
import glob

from graphtalk import analysis
from scripts import score_sweep


def _load_paths(patterns: list[str]) -> list[str]:
  paths = [p for pattern in patterns for p in glob.glob(pattern)]
  return [p for p in paths if not analysis.is_excluded(p)]


def _print_baseline(frame) -> None:
  print("\n=== Objective 1: Zeroshot / none baseline ===")
  baseline = frame[(frame["condition"] == "none") & (frame["style"] == "zero_shot")]
  if baseline.empty:
    print("  no zero_shot/none rows found")
    return
  for is_think, group in baseline.groupby("is_think"):
    label = "thinking arm" if is_think else "main sweep"
    print(f"  -- {label} --")
    for model, g in group.groupby("model"):
      print(f"    {model:<20} primary={g['primary'].mean():>6.1%}  "
            f"exact={g['exact'].mean():>6.1%}  parsed={g['parsed'].mean():>6.1%}  "
            f"n={len(g)}")


def _print_condition_comparison(frame) -> None:
  print("\n=== Objective 2: cross-condition comparison "
        "(primary metric, mean over tasks) ===")
  table = frame.groupby(["model", "condition"])["primary"].mean().unstack("condition")
  print(table.round(3).to_string())


def _print_non_termination(frame) -> None:
  print("\n=== Objective 3: thinking-arm non-termination ===")
  think = frame[frame["is_think"]]
  if think.empty:
    print("  no thinking-arm rows found")
    return
  for label, key in (("by model", "model"), ("by task", "task"),
                      ("by condition", "condition")):
    print(f"  {label}:")
    counts = think.groupby(key)["non_terminating"].agg(["sum", "mean", "count"])
    print(counts.rename(columns={"sum": "n", "mean": "rate"}).to_string())
  extra = think[think["length_outlier"] & ~think["non_terminating"]]
  ground_truth_n = int(think["non_terminating"].sum())
  if len(extra):
    print(f"\n  {len(extra)} additional rows flagged by the length-outlier "
          f"heuristic beyond the {ground_truth_n} ground-truth rows -- "
          "UNVALIDATED, inspect before trusting.")
  else:
    print(f"\n  no additional rows flagged beyond the {ground_truth_n} "
          "ground-truth rows.")


def _print_failure_breakdown(frame) -> None:
  print("\n=== Objective 5: failure_type breakdown ===")
  table = frame.groupby(["model", "failure_type"]).size().unstack(fill_value=0)
  for column in analysis.FAILURE_TYPES:
    if column not in table.columns:
      table[column] = 0
  print(table[list(analysis.FAILURE_TYPES)].to_string())


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--responses", nargs="+", required=True)
  parser.add_argument("--shortcuts", default=None,
                       help="optional shortcuts.json for the shortcut_score column")
  parser.add_argument("--truncated-keys", default=None,
                       help="optional analysis/truncated_keys.json for the "
                            "ground-truth non_terminating column")
  parser.add_argument("--out", default="analysis/sweep_frame.csv")
  args = parser.parse_args()

  paths = _load_paths(args.responses)
  if not paths:
    print("no responses found after exclusions")
    return

  records = score_sweep.score_records(score_sweep.load(paths))
  shortcuts_by_cell = (
      analysis.load_shortcuts(args.shortcuts) if args.shortcuts else {}
  )
  truncated_keys = (
      analysis.load_truncated_keys(args.truncated_keys)
      if args.truncated_keys else set()
  )

  frame = analysis.build_frame(records, truncated_keys, shortcuts_by_cell)
  frame.to_csv(args.out, index=False)
  print(f"wrote {len(frame)} rows to {args.out}")

  _print_baseline(frame)
  _print_condition_comparison(frame)
  _print_non_termination(frame)
  _print_failure_breakdown(frame)


if __name__ == "__main__":
  main()
