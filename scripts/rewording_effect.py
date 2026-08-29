"""What the 2026-08-28 prompt rewordings did, measured against the rows they replaced.

Two prompt strings changed in `d7cdcf7..3545662`: the `edge_existence` question
("Is node A connected to node B?" -> "Does an edge exist between Node A and Node
B?") and the `filler` primer ("Node N has <n-1> other nodes in this graph." ->
"Node N is simply present within the graph G."). The affected `zero_shot` rows
were regenerated; this script compares them against the originals.

Three things make the comparison honest, and all three are easy to get wrong:

  * **The two rewordings overlap.** `filler` spans every task and
    `edge_existence` spans every condition, so a naive per-task table mixes them
    -- a `node_count` number moves because 30 of its 210 rows are `filler`. The
    groups below are disjoint by construction.
  * **The extractor changed in the same merge.** `scoring._extract_boolean`
    gained a fallback that rescues 276 boolean rows sweep-wide, which would
    otherwise be credited to the rewording. Both sides are scored with the
    *current* extractor so the comparison isolates the prompt.
  * **`edge_existence` instances are not interchangeable.** The task is graded on
    a single edge, but 14 of the 30 instances have gold `No` with a path present.
    Only those can express a reachability misreading, so the per-class split is
    what distinguishes "the ambiguity was real" from "the new wording is just
    easier to read".

Baseline rows come from a pre-re-run copy of `runs/`; see `docs/DATA.md`.

  PYTHONPATH=. python scripts/rewording_effect.py --baseline <dir> --runs runs
"""

import argparse
import collections
import glob
import json
import os

import networkx as nx

from graphtalk import graphqa
from graphtalk import scoring
from scripts import build_prompts

ARMS = ("gemma4-e4b", "gemma4-12b", "qwen3-8b", "qwen3-14b",
        "gemma4-e4b-think", "gemma4-12b-think", "qwen3-8b-think",
        "qwen3-14b-think")


def load_arm(root: str, model: str) -> list[dict]:
  """Every tracked row for one arm, pooling shards and the `.rerun.` files.

  `.redo.shard` and `smoke-` are excluded for the reasons `docs/DATA.md` gives;
  `graphtalk.analysis.is_excluded` holds the same rule for the frame builder.
  """
  paths = sorted(glob.glob(os.path.join(root, f"{model}.jsonl"))
                 + glob.glob(os.path.join(root, f"{model}.shard*.jsonl"))
                 + glob.glob(os.path.join(root, f"{model}.rerun*.jsonl")))
  rows = []
  for path in paths:
    if ".redo.shard" in path or "smoke-" in path:
      continue
    with open(path) as handle:
      rows += [json.loads(line) for line in handle if line.strip()]
  return rows


def classify_instances(count: int, cache: str) -> dict[str, str]:
  """Each `edge_existence` instance by what a reachability reading would do to it.

  `path only` is the diagnostic class: gold is `No` because there is no edge, but
  a model reading "connected" as "reachable" would say `Yes`. `edge` and
  `unreachable` instances answer the same either way, so they are the control.
  """
  out = {}
  for index, row in enumerate(build_prompts.load_rows(
      "edge_existence", count, build_prompts.SPLIT, cache)):
    graph = graphqa.parse_graph(row["question"])
    source, target = graphqa._target_nodes(row["task_description"])
    if graph.has_edge(source, target):
      label = "edge (gold Yes)"
    elif nx.has_path(graph, source, target):
      label = "path only (gold No)"
    else:
      label = "unreachable (gold No)"
    out[f"edge_existence/{index}"] = label
  return out


def _score(rows, predicate) -> tuple[int, float, float]:
  """(n, accuracy, share answering Yes) over `zero_shot` rows matching.

  Returns n=0 for "no rows", which callers must distinguish from "scored 0%".
  Mid-re-run this is not hypothetical: an arm whose affected rows have been
  stripped but not yet regenerated matches nothing, and reporting that as 0.0%
  would read as a catastrophic regression rather than as missing data.
  """
  chosen = [r for r in rows if r["style"] == "zero_shot" and predicate(r)]
  if not chosen:
    return 0, 0.0, 0.0
  hits = yes = 0.0
  for row in chosen:
    predicted = scoring.extract_answer(row["response"], row["task"])
    hits += scoring.score_one(predicted, row["gold"], row["task"])["primary"]
    yes += (predicted or "").strip().lower().startswith("yes")
  return len(chosen), hits / len(chosen), yes / len(chosen)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--baseline", required=True,
                      help="directory of pre-re-run runs/*.jsonl")
  parser.add_argument("--runs", default="runs")
  parser.add_argument("--count", type=int, default=30)
  parser.add_argument("--cache", default=".cache/sweep_rows")
  args = parser.parse_args()

  klass = classify_instances(args.count, args.cache)
  counts = collections.Counter(klass.values())
  print("edge_existence instances: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

  groups = (
      ("QUESTION", lambda r: r["task"] == "edge_existence"
       and r["condition"] != "filler"),
      ("PRIMER", lambda r: r["condition"] == "filler"
       and r["task"] != "edge_existence"),
  )
  print(f"\n{'arm':<19}{'group':<10}{'n':>5}{'old':>9}{'new':>9}{'delta':>9}")
  for model in ARMS:
    old, new = load_arm(args.baseline, model), load_arm(args.runs, model)
    if not old or not new:
      continue
    for label, predicate in groups:
      n, a, _ = _score(old, predicate)
      m, b, _ = _score(new, predicate)
      if not n:
        continue
      if m != n:
        print(f"{model:<19}{label:<10}{n:>5}"
              f"{a:>8.1%}{'--':>9}{'--':>9}   incomplete ({m}/{n} regenerated)")
        continue
      print(f"{model:<19}{label:<10}{n:>5}{a:>8.1%}{b:>9.1%}{b - a:>+9.1%}")

  print(f"\nedge_existence by instance class (non-`filler` conditions)")
  print(f"{'arm':<19}{'class':<24}{'n':>4}{'acc old':>9}{'acc new':>9}"
        f"{'Yes old':>9}{'Yes new':>9}")
  for model in ARMS:
    old, new = load_arm(args.baseline, model), load_arm(args.runs, model)
    if not old or not new:
      continue
    for label in ("edge (gold Yes)", "path only (gold No)",
                  "unreachable (gold No)"):
      pred = (lambda r, _l=label: r["task"] == "edge_existence"
              and r["condition"] != "filler" and klass[r["instance_id"]] == _l)
      n, a, ya = _score(old, pred)
      m, b, yb = _score(new, pred)
      if not n or m != n:
        continue
      print(f"{model:<19}{label:<24}{n:>4}{a:>8.1%}{b:>9.1%}{ya:>9.1%}{yb:>9.1%}")


if __name__ == "__main__":
  main()
