"""Stage 4: pooled significance tests over the tracked sweep. No GPU needed.

`scripts/score_sweep.py` runs exact McNemar per (task, style, condition)
cell -- 288 cells across 4 models, most with fewer than 10 discordant pairs
out of 30, and no correction for testing 288 of them at once (see
docs/sweep-findings.md, "The McNemar analysis is underpowered"). This script
pools pairs across task and style instead, per `graphtalk/significance.py`:
a permutation p-value, a bootstrap CI on the effect size, and a
Benjamini-Hochberg correction across the conditions tested for one model.
It reuses the same pooling for the thinking arm's non-termination rate,
replacing the ad hoc p-values quoted in prose there.

Reads the already-scored, already-joined table `scripts/build_sweep_frame.py`
writes -- no re-scoring, no re-reading raw runs/*.jsonl.

  PYTHONPATH=. .venv/bin/python scripts/check_significance.py \
      --frame analysis/sweep_frame.csv
"""

import argparse

import pandas as pd

from graphtalk import significance

CONTROL = "none"


def _paired_values(frame: pd.DataFrame, condition: str, metric: str):
  """Aligned (control, treatment) lists for `condition` vs `CONTROL`.

  Paired on (model, instance_id, style): instance_id alone repeats across
  styles for the same task, so style has to be part of the key or a
  zero_shot control row could pair against a zero_cot treatment row: model
  has to be part of it too, since the same (instance_id, style) key recurs
  once per model when `frame` pools rows across models. Pooling across
  every task and style present in `frame` is what makes these pairs more
  numerous than a single `score_sweep.py` cell.
  """
  keys = ["model", "instance_id", "style"]
  control = frame[frame["condition"] == CONTROL].set_index(keys)[metric]
  treatment = frame[frame["condition"] == condition].set_index(keys)[metric]
  joined = pd.concat(
      [control.rename("control"), treatment.rename("treatment")],
      axis=1, join="inner",
  )
  return joined["control"].tolist(), joined["treatment"].tolist()


def _report(
    frame: pd.DataFrame, metric: str, label: str, args, arm: str, records: list
) -> None:
  print(f"\n  {label}")
  conditions = sorted(c for c in frame["condition"].unique() if c != CONTROL)
  rows = []
  for condition in conditions:
    control, treatment = _paired_values(frame, condition, metric)
    if not control:
      print(f"    {condition:<12} -- no paired rows found")
      continue
    perm = significance.paired_permutation_test(
        control, treatment, n_perm=args.n_perm, seed=args.seed
    )
    boot = significance.cluster_bootstrap_ci(
        control, treatment, n_boot=args.n_boot, seed=args.seed, alpha=args.alpha
    )
    rows.append((condition, perm, boot))

  if not rows:
    return
  reject = significance.benjamini_hochberg(
      [perm["p_value"] for _, perm, _ in rows], q=args.q
  )
  print(f"    {'condition':<12}{'n_pairs':>9}{'delta':>10}"
        f"{'95% CI':>22}{'p (perm)':>10}  BH-sig")
  for (condition, perm, boot), sig in zip(rows, reject):
    ci = f"[{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}]"
    print(f"    {condition:<12}{perm['n_pairs']:>9}"
          f"{perm['observed_diff']:>+10.3f}{ci:>22}"
          f"{perm['p_value']:>10.4f}  {'yes' if sig else 'no'}")
    records.append({
        "arm": arm,
        "group": label,
        "condition": condition,
        "n_pairs": perm["n_pairs"],
        "delta": perm["observed_diff"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "p_value": perm["p_value"],
        "bh_significant": sig,
    })


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frame", default="analysis/sweep_frame.csv")
  parser.add_argument("--n-perm", type=int, default=10_000,
                       help="permutations for the pooled p-value")
  parser.add_argument("--n-boot", type=int, default=10_000,
                       help="resamples for the pooled bootstrap CI")
  parser.add_argument("--alpha", type=float, default=0.05,
                       help="bootstrap CI level (default 95%% CI)")
  parser.add_argument("--q", type=float, default=0.05,
                       help="Benjamini-Hochberg FDR level")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--out", default=None,
                       help="optional path to write every row printed above "
                            "as CSV, for reuse beyond this terminal session")
  args = parser.parse_args()

  frame = pd.read_csv(args.frame)
  records = []

  main_sweep = frame[~frame["is_think"]]
  think = frame[frame["is_think"]]

  print("=" * 78)
  print("Main sweep: accuracy (exact) vs `none`, pooled across task + style")
  for model_family, group in main_sweep.groupby("model_family"):
    _report(group, "exact", model_family, args, "main_sweep", records)
  _report(main_sweep, "exact", "pooled across all models", args,
          "main_sweep", records)

  if not think.empty:
    print(f"\n{'=' * 78}")
    print("Thinking arm: non-termination rate vs `none`, pooled across task")
    for model_family, group in think.groupby("model_family"):
      _report(group, "non_terminating", model_family, args,
              "thinking_arm", records)
    _report(think, "non_terminating", "pooled across all models", args,
            "thinking_arm", records)

  if args.out:
    pd.DataFrame(records).to_csv(args.out, index=False)
    print(f"\nwrote {len(records)} rows to {args.out}")


if __name__ == "__main__":
  main()
