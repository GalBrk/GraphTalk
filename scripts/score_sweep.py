"""Stage 3: score a model's responses and run the paired tests. No GPU needed.

Reports, per (task, style):

  * the majority baseline over the sampled rows, which the proposal asks for on
    `cycle_check` and `edge_existence` and which is worth printing everywhere,
    since four of the six tasks have a baseline a model can fail to reach;
  * each condition's score under the metric the proposal names for that task;
  * the shortcut score for that cell, so a result can be read against the bar
    rather than only against the control;
  * McNemar against the no-primer control, on the paired rows.

  PYTHONPATH=. .venv/bin/python scripts/score_sweep.py --responses runs/*.jsonl
"""

import argparse
import collections
import json

from graphtalk import scoring

CONTROL = "none"


def load(paths) -> list[dict]:
  records = []
  for path in paths:
    with open(path) as handle:
      for line in handle:
        if line.strip():
          records.append(json.loads(line))
  return records


def score_records(records) -> list[dict]:
  for record in records:
    predicted = scoring.extract_answer(record["response"], record["task"])
    record["predicted"] = predicted
    record["score"] = scoring.score_one(predicted, record["gold"], record["task"])
  return records


def report(records, shortcuts_by_cell) -> None:
  by_model = collections.defaultdict(list)
  for record in records:
    by_model[record["model"]].append(record)

  for model in sorted(by_model):
    print(f"\n{'=' * 78}\n{model}")
    if shortcuts_by_cell:
      # Stated once rather than per block: the shortcut table scores
      # `connected_nodes` by exact match, so on that task the comparable column
      # is `exact` and not the F1 the proposal reports the task on.
      print("  shortcut is an exact-match figure; compare it against the "
            "'exact' column,\n  not against F1.")
    grouped = collections.defaultdict(list)
    for record in by_model[model]:
      grouped[(record["task"], record["style"], record["condition"])].append(record)

    tasks = sorted({t for t, _, _ in grouped})
    for task in tasks:
      styles = sorted({s for t, s, _ in grouped if t == task})
      for style in styles:
        cells = {c: rows for (t, s, c), rows in grouped.items()
                 if t == task and s == style}
        if not cells:
          continue
        golds = [r["gold"] for r in next(iter(cells.values()))]
        answer, share = scoring.majority_baseline(golds)
        metric = "F1" if task == "connected_nodes" else "acc"
        # On `connected_nodes` almost every gold string is unique, so the modal
        # answer covers a couple of rows and is not a predictor anyone would use.
        # Saying so is better than printing a number that invites comparison.
        note = "" if share >= 0.15 else "  -- degenerate, no useful constant"
        print(f"\n  {task} / {style}   baseline {share:.1%} "
              f"(always {answer!r}){note}")
        print(f"    {'condition':<12} {metric:>7} {'exact':>7} {'parsed':>7} "
              f"{'shortcut':>9} {'McNemar p':>10}  vs control")

        control = {r["instance_id"]: r for r in cells.get(CONTROL, [])}
        for condition in sorted(cells):
          rows = sorted(cells[condition], key=lambda r: r["instance_id"])
          summary = scoring.aggregate(r["score"] for r in rows)
          shortcut = shortcuts_by_cell.get((task, condition))
          shortcut_text = f"{shortcut:.1%}" if shortcut is not None else "-"

          if condition == CONTROL or not control:
            verdict, p_text = "", ""
          else:
            paired = [(control[r["instance_id"]], r) for r in rows
                      if r["instance_id"] in control]
            test = scoring.mcnemar(
                [c["score"]["exact"] > 0.5 for c, _ in paired],
                [t["score"]["exact"] > 0.5 for _, t in paired],
            )
            p_text = f"{test['p_value']:.3f}"
            delta = test["c"] - test["b"]
            verdict = (f"{'+' if delta > 0 else ''}{delta} net "
                       f"({test['discordant']} discordant)")
          print(f"    {condition:<12} {summary['primary']:>6.1%} "
                f"{summary['exact']:>6.1%} {summary['parsed']:>6.1%} "
                f"{shortcut_text:>9} {p_text:>10}  {verdict}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--responses", nargs="+", required=True)
  parser.add_argument("--shortcuts", default=None,
                      help="optional JSON of {'task/condition': score} to print "
                           "each cell's bar beside the model's result")
  args = parser.parse_args()

  shortcuts_by_cell = {}
  if args.shortcuts:
    with open(args.shortcuts) as handle:
      for key, value in json.load(handle).items():
        task, condition = key.split("/")
        shortcuts_by_cell[(task, condition)] = value

  records = score_records(load(args.responses))
  if not records:
    print("no responses found")
    return
  report(records, shortcuts_by_cell)

  unparsed = [r for r in records if not r["score"]["parsed"]]
  print(f"\n{len(unparsed)}/{len(records)} responses could not be parsed")
  if unparsed:
    # Unparsed rows are an extraction or truncation problem, not a wrong answer,
    # so they are surfaced rather than folded into the accuracy.
    worst = collections.Counter((r["task"], r["style"]) for r in unparsed)
    for (task, style), count in worst.most_common(5):
      print(f"  {task} / {style}: {count}")


if __name__ == "__main__":
  main()
