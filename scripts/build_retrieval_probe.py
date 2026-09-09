"""Build a pure retrieval probe: can the model read one stated fact out of a long prompt?

Three separate findings in `docs/primer-effects-and-power.md` point at one
mechanism rather than three:

- the `degree` ceiling (job 871263) -- with the answer written verbatim the
  model reaches only 0.610 at p=0.35 and gains +6.8 pp at p=0.50;
- the high-density collapse (job 866492) -- it enumerates 35 neighbours
  correctly, answers 39, and never consults the `"Node 29 has degree 32."`
  sentence sitting in its own prompt;
- the `filler` penalty -- 1,831 characters of irrelevant text cost the thinking
  arm 11.7 pp on dense graphs.

All three are consistent with "this model loses a stated fact as the prompt
grows". That hypothesis is about *reading*, not about graphs, so it can be
tested without a graph anywhere in the design -- which is the point of this
script. Every prompt states the answer explicitly; nothing has to be computed,
parsed, or reasoned about. If accuracy still collapses with length, the primer
question is downstream of a reading limit and the density story is a symptom.

The statement format is `render_primer`'s `degree` wording verbatim ("Node X has
degree Y."), so a hit here is the same operation the `degree` condition asks for
in the real sweep, minus the graph.

Three factors:
  --statements   how many "Node X has degree Y." sentences surround the target
                 (context length; 10 -> ~70 tokens, 1280 -> ~9,000)
  position       where the queried node sits in the list (0.1 / 0.5 / 0.9),
                 since a length effect and a lost-in-the-middle effect are
                 different failures with different fixes
  magnitude      whether the stated degree is a small (1-9) or large (20-39)
                 number, because job 871262 found answer magnitude to be a
                 difficulty driver in its own right and it must not be
                 confounded with length here

Distractor degrees are drawn from the same band as the target, so the answer is
never identifiable by being the only number of its size.
"""

import argparse
import json
import random

QUESTION = "Q: What is the degree of node {node}?\nA: "
BANDS = {"small": (1, 9), "large": (20, 39)}


def build(statement_counts, positions, magnitudes, count, seed):
  records = []
  for k in statement_counts:
    for position in positions:
      for magnitude in magnitudes:
        low, high = BANDS[magnitude]
        for index in range(count):
          # One seed per cell and index, so a prompt is reproducible from its
          # instance_id alone -- the same contract build_size_sweep.py keeps.
          rng = random.Random(seed + 100003 * k + int(1000 * position) * 97
                              + (0 if magnitude == "small" else 7919) + index)
          degrees = [rng.randint(low, high) for _ in range(k)]
          target = min(k - 1, max(0, int(round(position * (k - 1)))))
          sentences = [f"Node {i} has degree {d}." for i, d in enumerate(degrees)]
          prompt = " ".join(sentences) + "\n\n" + QUESTION.format(node=target)
          records.append({
              "instance_id": f"retrieval/k{k}/pos{position:g}/{magnitude}/{index}",
              # Tagged as node_degree so scoring.extract_answer's existing
              # per-task logic applies unchanged -- the question wording is
              # identical, and a probe that needed its own extractor would be
              # measuring the extractor rather than the model.
              "task": "node_degree",
              "condition": "retrieval",
              "style": "zero_shot",
              "prompt": prompt,
              "gold": str(degrees[target]),
              "statements": k,
              "position": position,
              "magnitude": magnitude,
          })
  return records


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--statements", type=int, nargs="+",
                      default=[10, 40, 160, 320, 640, 1280])
  parser.add_argument("--positions", type=float, nargs="+", default=[0.1, 0.5, 0.9])
  parser.add_argument("--magnitudes", nargs="+", default=["small", "large"],
                      choices=sorted(BANDS))
  parser.add_argument("--count", type=int, default=100, help="items per cell")
  parser.add_argument("--seed", type=int, default=20260909)
  parser.add_argument("--out", default="prompts.retrieval.jsonl")
  args = parser.parse_args()

  records = build(args.statements, args.positions, args.magnitudes,
                  args.count, args.seed)
  with open(args.out, "w") as handle:
    for record in records:
      handle.write(json.dumps(record) + "\n")
  print(f"wrote {len(records)} prompts to {args.out}")
  for k in args.statements:
    rows = [r for r in records if r["statements"] == k]
    chars = sorted(len(r["prompt"]) for r in rows)
    print(f"  {k:>5} statements: {len(rows):>5} prompts, "
          f"chars med {chars[len(chars)//2]:>7,} max {chars[-1]:>7,}")


if __name__ == "__main__":
  main()
