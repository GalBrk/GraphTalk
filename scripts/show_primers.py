"""Print every primer condition for a few graphs, for eyeballing.

The tests in tests/test_primers.py check arithmetic. This script checks the
thing they cannot: whether the English reads like the surrounding graph prose,
whether the numbers survive a hand-check, and whether the length control really
states nothing structural. It prints the incident encoding alongside each primer
so the two can be read together, as the model will see them.

Usage:
  python scripts/show_primers.py --config node_degree --count 3
  python scripts/show_primers.py --generated 3     # no network needed
"""

import argparse
import textwrap

import numpy as np

from graphtalk import graphqa
from graphtalk import primers
from talk_like_a_graph import graph_generators
from talk_like_a_graph import graph_text_encoders

WRAP = 96


def _wrapped(text: str, indent: str) -> str:
  return textwrap.fill(
      text, width=WRAP, initial_indent=indent, subsequent_indent=indent
  )


def show(graph, label: str, k_min: int, k_max: int, target_chars: int | None):
  """Prints one graph's encoding, every condition's primer, and the diagnostic."""
  print(
      f"\n=== {label}: {graph.number_of_nodes()} nodes,"
      f" {graph.number_of_edges()} edges,"
      f" {primers.component_count(graph)} components ==="
  )
  print("  encoding (incident):")
  encoding = graph_text_encoders.encode_graph(graph, "incident")
  for line in encoding.strip().splitlines():
    print(_wrapped(line, "    "))

  lengths = {}
  for condition in primers.CONDITIONS:
    text = primers.build_primer(
        graph, condition, k_min=k_min, k_max=k_max, target_chars=target_chars
    )
    lengths[condition] = len(text)
    print(f"  [{condition}] {len(text)} chars")
    if text:
      print(_wrapped(text, "    "))

  # Information, not a pass/fail check. Per-graph r ranges from -0.77 to +1.00
  # and is undefined on a fifth of graphs at k=3, so a three-row sample says
  # nothing: the acceptance criterion is the corpus-level window in
  # docs/plans/primer-computation.md over at least 100 graphs.
  correlation = primers.rwse_degree_correlation(graph, k_min=k_min, k_max=k_max)
  rendered = " ".join(
      f"k={k}: {'undefined' if r is None else format(r, '+.2f')}"
      for k, r in sorted(correlation.items())
  )
  print(f"  rwse/degree r (information only): {rendered}")
  return lengths, correlation


def corpus_report(count: int, seed: int, k_min: int, k_max: int) -> None:
  """Prints the corpus-level statistics that are the actual acceptance criterion.

  The per-graph correlations printed by `show` are information only: per-graph r
  ranges from -0.77 to +1.00 and is undefined on a fifth of graphs at k=3. The
  criterion in docs/plans/primer-computation.md is the mean of per-graph r over
  at least 100 graphs, against the windows below. Aggregation matters: pooling
  all nodes of all graphs flips the sign, because it measures graph size rather
  than node degree.
  """
  windows = {2: (0.50, 0.75), 3: (0.83, 0.94)}
  graphs = [
      graphqa.canonical(graph)
      for graph in graph_generators.generate_graphs(
          count, "er", False, random_seed=seed
      )
  ]
  print(
      f"corpus of {count} graphs from generate_graphs(er, seed={seed});"
      " aggregation: mean of per-graph r"
  )
  for k in range(k_min, k_max + 1):
    values = [
        primers.rwse_degree_correlation(graph, k_min=k, k_max=k)[k]
        for graph in graphs
    ]
    defined = [value for value in values if value is not None]
    mean = np.mean(defined) if defined else float("nan")
    window = windows.get(k)
    verdict = ""
    if window:
      inside = window[0] <= mean <= window[1]
      verdict = (
          f"  window [{window[0]:.2f}, {window[1]:.2f}]:"
          f" {'inside' if inside else 'OUTSIDE'}"
      )
    print(
        f"  k={k}: mean r {mean:+.3f}  defined on {len(defined)}/{count}"
        f" ({100 * (count - len(defined)) / count:.0f}% undefined){verdict}"
    )
  print("  (different k are averaged over different populations; odd-k return is")
  print("   exactly 0 for every node of a triangle-free graph)")

  for condition in primers.CONDITIONS:
    lengths = [
        len(primers.build_primer(graph, condition, k_min=k_min, k_max=k_max))
        for graph in graphs
    ]
    print(f"  {condition:11s} mean {np.mean(lengths):7.1f} chars")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", default="node_degree")
  parser.add_argument("--split", default="zero_shot_test")
  parser.add_argument("--index", type=int, default=0)
  parser.add_argument("--count", type=int, default=3)
  parser.add_argument(
      "--generated",
      type=int,
      default=0,
      help=(
          "show this many graphs from the upstream generator instead of dataset"
          " rows; use where the HuggingFace rows API is unreachable"
      ),
  )
  parser.add_argument(
      "--corpus",
      type=int,
      default=0,
      help=(
          "instead of printing primers, report the corpus-level correlation and"
          " length statistics over this many generated graphs (use >= 100)"
      ),
  )
  parser.add_argument("--random-seed", type=int, default=1234)
  parser.add_argument("--k-min", type=int, default=2)
  parser.add_argument("--k-max", type=int, default=3)
  parser.add_argument(
      "--target-chars",
      type=int,
      default=None,
      help="pad every primer with inert filler up to this many characters",
  )
  args = parser.parse_args()

  if args.corpus:
    corpus_report(args.corpus, args.random_seed, args.k_min, args.k_max)
    return

  if args.generated:
    graphs = [
        (f"generated {i}", graphqa.canonical(graph))
        for i, graph in enumerate(
            graph_generators.generate_graphs(
                args.generated, "er", False, random_seed=args.random_seed
            )
        )
    ]
    print(
        f"{args.generated} graphs from generate_graphs(er, seed="
        f"{args.random_seed}); no dataset rows involved"
    )
  else:
    rows = graphqa.fetch_rows(args.config, args.split, args.index, args.count)
    graphs = [
        (f"{args.config}/{args.split} row {args.index + offset}",
         graphqa.parse_graph(row["question"]))
        for offset, row in enumerate(rows)
    ]
    print(f"{len(graphs)} rows from {args.config}/{args.split}")

  all_lengths, undefined = [], {}
  for label, graph in graphs:
    lengths, correlation = show(
        graph, label, args.k_min, args.k_max, args.target_chars
    )
    all_lengths.append(lengths)
    for k, value in correlation.items():
      undefined[k] = undefined.get(k, 0) + (value is None)

  print("\n=== achieved primer lengths (chars) ===")
  for condition in primers.CONDITIONS:
    values = [lengths[condition] for lengths in all_lengths]
    print(
        f"  {condition:11s} mean {np.mean(values):7.1f}"
        f"  min {min(values):5d}  max {max(values):5d}"
    )
  for k in sorted(undefined):
    print(
        f"  rwse/degree r undefined at k={k} on {undefined[k]}/{len(graphs)}"
        " graphs shown"
    )


if __name__ == "__main__":
  main()
