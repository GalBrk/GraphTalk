"""Pin down exactly which of the "same" 30 instances disagree between the
tracked standalone --count 30 GOT sweep and the first 30 instances embedded
in the --count 500 replication -- rather than inferring a flip count from
the aggregate delta gap (see check_old_vs_new_subsample.py's sanity-check
line, which only tells you the two deltas differ, not which instances or
why).

Two distinct things can produce a paired-outcome flip on a "same" instance,
and this script is built to tell them apart:

  1. The prompt text itself changed between the two runs (e.g. a code
     change to build_prompts.py/primers.py/node_naming.py landed in
     between -- graphtalk/node_naming.py's `GOT_NAMES[1] = "Catelyn"`
     override, fixing a desubstitution collision with the vendored "Cat",
     is a real, dated example of exactly this kind of change). If the
     prompt differs, a different answer is expected, not surprising.
  2. The prompt is byte-identical but the model's response differed --
     decoding-level nondeterminism (temperature, sampling, batching,
     hardware/driver differences across cluster jobs -- this project has
     an existing precedent: rows generated on CPU due to a driver mismatch
     were flagged as unverified against GPU numbers). If prompts match
     and outcomes still differ, this is the remaining explanation.

    PYTHONPATH=. .venv/bin/python scripts/diff_shared_instances.py \
        --old-frame analysis/sweep_frame.got.csv \
        --new-frame analysis/sweep_frame.count500.got.csv \
        --model qwen3-8b --condition degree \
        --old-prompts prompts_got.jsonl \
        --new-prompts prompts_got.count500.jsonl
"""

import argparse
import json
import re

import pandas as pd

from scripts.check_significance import CONTROL

_INSTANCE_INDEX_RE = re.compile(r"/(\d+)$")


def _instance_index(instance_id: str) -> int:
  match = _INSTANCE_INDEX_RE.search(str(instance_id))
  if not match:
    raise ValueError(f"instance_id {instance_id!r} doesn't end in /<index>")
  return int(match.group(1))


def _load_frame(path: str, model: str, node_naming: str, split_at: int) -> pd.DataFrame:
  frame = pd.read_csv(path)
  frame = frame[(frame["model"] == model) & (~frame["is_think"])]
  if "node_naming" in frame.columns:
    frame = frame[frame["node_naming"] == node_naming]
  frame = frame.copy()
  frame["_index"] = frame["instance_id"].map(_instance_index)
  return frame[frame["_index"] < split_at]


def _pair_diffs(frame: pd.DataFrame, condition: str, metric: str) -> pd.DataFrame:
  """One row per (instance_id, style): `metric`(condition) - `metric`(none)."""
  keys = ["instance_id", "style"]
  control = frame[frame["condition"] == CONTROL].set_index(keys)[metric]
  treatment = frame[frame["condition"] == condition].set_index(keys)[metric]
  joined = pd.concat(
      [control.rename("control"), treatment.rename("treatment")],
      axis=1, join="inner",
  )
  joined["pair_diff"] = joined["treatment"] - joined["control"]
  return joined


def _load_prompts(path: str | None) -> dict[str, str]:
  if not path:
    return {}
  prompts = {}
  with open(path) as handle:
    for line in handle:
      record = json.loads(line)
      prompts[(record["instance_id"], record["condition"], record["style"])] = (
          record["prompt"]
      )
  return prompts


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--old-frame", default="analysis/sweep_frame.got.csv")
  parser.add_argument("--new-frame", default="analysis/sweep_frame.count500.got.csv")
  parser.add_argument("--model", default="qwen3-8b")
  parser.add_argument("--condition", default="degree")
  parser.add_argument("--metric", default="exact")
  parser.add_argument("--node-naming", default="got")
  parser.add_argument("--split-at", type=int, default=30)
  parser.add_argument("--old-prompts", default=None,
                      help="optional: prompts.jsonl the ORIGINAL --count 30 "
                           "run was built from, to check byte-identity")
  parser.add_argument("--new-prompts", default=None,
                      help="optional: prompts.jsonl the --count 500 run was "
                           "built from")
  args = parser.parse_args()

  old = _load_frame(args.old_frame, args.model, args.node_naming, args.split_at)
  new = _load_frame(args.new_frame, args.model, args.node_naming, args.split_at)

  old_diffs = _pair_diffs(old, args.condition, args.metric)
  new_diffs = _pair_diffs(new, args.condition, args.metric)

  merged = old_diffs[["pair_diff"]].join(
      new_diffs[["pair_diff"]], lsuffix="_old", rsuffix="_new", how="inner"
  )
  mismatched = merged[merged["pair_diff_old"] != merged["pair_diff_new"]]

  n_pairs = len(merged)
  implied_delta_gap = (merged["pair_diff_new"] - merged["pair_diff_old"]).mean()
  print(f"{args.model}/{args.condition}: {n_pairs} shared pairs compared "
        f"(instances 0-{args.split_at - 1})")
  print(f"  mismatched pairs: {len(mismatched)}")
  print(f"  mean(new - old) pair_diff = {implied_delta_gap:+.4f}  "
        f"(should match the two runs' reported delta gap)")

  if mismatched.empty:
    print("\n  No mismatches found -- the earlier delta gap must be coming "
          "from outside this model/condition/instance-range slice; double "
          "check --model/--condition/--split-at against what was compared.")
    return

  prompts_old = _load_prompts(args.old_prompts)
  prompts_new = _load_prompts(args.new_prompts)
  check_prompts = bool(prompts_old and prompts_new)

  print(f"\n  mismatched instances (task/index, style, old->new pair_diff"
        f"{', prompt match' if check_prompts else ''}):")
  for (instance_id, style), row in mismatched.iterrows():
    line = (f"    {instance_id:<20} {style:<12} "
            f"{row['pair_diff_old']:+.0f} -> {row['pair_diff_new']:+.0f}")
    if check_prompts:
      none_same = (
          prompts_old.get((instance_id, CONTROL, style))
          == prompts_new.get((instance_id, CONTROL, style))
      )
      cond_same = (
          prompts_old.get((instance_id, args.condition, style))
          == prompts_new.get((instance_id, args.condition, style))
      )
      line += f"  prompts identical: none={none_same} {args.condition}={cond_same}"
    print(line)

  if check_prompts:
    all_prompts_matched = all(
        prompts_old.get((iid, cond, style)) == prompts_new.get((iid, cond, style))
        for (iid, style) in mismatched.index
        for cond in (CONTROL, args.condition)
    )
    print("\n  verdict:")
    if all_prompts_matched:
      print("    Every mismatched instance had byte-identical prompts in "
            "both runs -> the flips are in the model's output, not the "
            "input. Points to decoding/serving-environment nondeterminism, "
            "not a code/content change between the two runs.")
    else:
      print("    At least one mismatched instance had DIFFERENT prompt "
            "text between the two runs -> at least part of the gap is a "
            "real content change (e.g. a build_prompts.py/node_naming.py "
            "edit landing between the two runs), not decoding noise. "
            "Worth `git log` on the relevant module between the two runs' "
            "dates.")
  else:
    print("\n  Pass --old-prompts/--new-prompts to check whether these "
          "instances' prompt text actually changed between the two runs, "
          "before attributing the flips to decoding nondeterminism.")


if __name__ == "__main__":
  main()
