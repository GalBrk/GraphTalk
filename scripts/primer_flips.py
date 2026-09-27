"""What a primer changes, for reading by hand: every (arm, question, primer) of the
40-node sweep whose outcome differs from the same question under `none`, with both
answers side by side.

Outcomes follow R1 (graphtalk/outcomes.py): correct, wrong or truncated, and a
truncated response is never correct. Rows are sorted by task, primer, direction
and arm, and in a seeded random order within each of those groups; `rank`
numbers them, so filtering `rank <= 3` gives a random three from every group.

  primer_says   the primer's sentences about the queried node(s), and any
                sentence about the whole graph (the `components` primer)
  start_*       how the response opens (the thinking tag dropped)
  end_*         what it concludes: the text after </think> when there is one,
                else the response; its last END characters either way

Whitespace is collapsed so every row is one line in a spreadsheet. The full text
is in data/runs/<arm>.densfull40*.jsonl under (instance_id, condition).

  PYTHONPATH=. python scripts/primer_flips.py

Writes outputs/n40-sweep/primer_flips.csv.
"""
import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_raw_frame as brf  # noqa: E402  (Corpus: the true graphs)
import primer_findings as pf  # noqa: E402

from graphtalk import outcomes, primers  # noqa: E402

CONDS = ["filler"] + pf.PRIMERS
START, END = 300, 500
# Primer sentences start with "Node <id> " or "This graph "; a float's period is
# followed by a digit, never by whitespace, so it does not end a sentence.
_SENTENCE = re.compile(r"(?<=\.)\s+(?=Node \d+ |This graph )")


def flat(s):
  return " ".join(s.split())


def primer_says(primer, targets):
  keep = tuple(f"Node {t} " for t in targets)
  return " ".join(s for s in _SENTENCE.split(primer)
                  if s.startswith(keep) or not s.startswith("Node "))


def start(text):
  return flat(text.removeprefix("<think>"))[:START]


def end(text):
  return flat(text.rpartition("</think>")[2])[-END:]


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--out", default="outputs/n40-sweep/primer_flips.csv")
  args = ap.parse_args()

  f = pd.read_csv(pf.FRAME, dtype={"pred": str, "gold": str})
  f["outcome"] = outcomes.outcome(f.exact, f.hit_cap)
  key = ["arm", "instance_id"]
  none = f[f.condition == "none"].set_index(key)[["outcome", "pred", "resp_chars"]]
  d = f[f.condition != "none"].join(none, on=key, rsuffix="_none")
  d = d[d.outcome != d.outcome_none].copy()
  d["direction"] = d.outcome_none + " -> " + d.outcome
  d["condition"] = pd.Categorical(d.condition, CONDS, ordered=True)
  groups = ["task", "condition", "direction", "arm"]
  d = (d.sample(frac=1, random_state=pf.SEED)
       .sort_values(groups, kind="stable").reset_index(drop=True))
  d["rank"] = d.groupby(groups, observed=True).cumcount() + 1

  runs = {a: pf.load_runs(f"data/runs/{a}.densfull40*.shard*.jsonl") for a in pf.ARMS}
  corpus, primer_text = brf.Corpus(), {}
  cols = {k: [] for k in ("primer_says", "start_none", "start_primer",
                          "end_none", "end_primer")}
  for r in d.itertuples():
    k = (r.density_class, r.index, r.condition)
    if k not in primer_text:
      g = corpus.graph(r.density_class, r.index)[0]
      primer_text[k] = primers.build_primer(g, r.condition)
    targets = corpus.row(r.density_class, r.index, r.task)["targets"]
    cols["primer_says"].append(primer_says(primer_text[k], targets))
    for side, cond in (("none", "none"), ("primer", r.condition)):
      text = runs[r.arm][(r.instance_id, cond)]["response"] or ""
      cols["start_" + side].append(start(text))
      cols["end_" + side].append(end(text))

  out = pd.DataFrame({
      "task": d.task, "condition": d.condition, "direction": d.direction,
      "arm": d.arm, "density": d.density_class, "instance_id": d.instance_id,
      "rank": d["rank"], "gold": d.gold, "pred_none": d.pred_none,
      "pred_primer": d.pred, "primer_says": cols["primer_says"],
      "chars_none": d.resp_chars_none, "chars_primer": d.resp_chars,
      "start_none": cols["start_none"], "start_primer": cols["start_primer"],
      "end_none": cols["end_none"], "end_primer": cols["end_primer"],
      "label": "", "notes": ""})
  # utf-8-sig: Excel reads a BOM-less file as the local code page and garbles
  # the responses' non-ASCII characters; pandas skips the BOM.
  out.to_csv(args.out, index=False, encoding="utf-8-sig")
  print(f"wrote {len(out)} rows to {args.out}")
  print(out.groupby("direction").size().to_string())


if __name__ == "__main__":
  main()
