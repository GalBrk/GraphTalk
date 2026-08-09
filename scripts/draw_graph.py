"""Parse a graph out of a GraphQA example and draw it.

The published GraphQA dataset ships no edge lists -- every field is a string, and
the graph exists only as rendered English inside the `question` field. This script
recovers a networkx graph from that prose, checks the result against the row's own
metadata and ground-truth answer, and renders it to a PNG.

The parsing, row fetching and gold-answer logic live in `graphtalk.graphqa`, which
is shared with the primer pipeline; only the checking and drawing are here.

Usage:
  python scripts/draw_graph.py --config node_degree --index 0
"""

import argparse
from pathlib import Path

import matplotlib
import networkx as nx

from graphtalk.graphqa import expected_answer
from graphtalk.graphqa import fetch_rows
from graphtalk.graphqa import normalize
from graphtalk.graphqa import parse_graph

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)


def check(graph: nx.Graph, row: dict, config: str) -> list[str]:
  """Returns a list of mismatch descriptions; empty means the parse is sound."""
  problems = []
  if graph.number_of_nodes() != int(row["nnodes"]):
    problems.append(
        f"nnodes {graph.number_of_nodes()} != {row['nnodes']} (dataset)"
    )
  if graph.number_of_edges() != int(row["nedges"]):
    problems.append(
        f"nedges {graph.number_of_edges()} != {row['nedges']} (dataset)"
    )

  want = normalize(row["answer"])
  got = normalize(expected_answer(graph, config, row["task_description"]))
  if want != got:
    problems.append(f"answer {got!r} != {want!r} (dataset)")
  return problems


def draw(graph: nx.Graph, row: dict, config: str, out_path: Path) -> None:
  degrees = dict(graph.degree())
  question = row["task_description"].replace("\n", " ").strip()

  fig, ax = plt.subplots(figsize=(9, 8))
  pos = nx.spring_layout(graph, seed=7)
  nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.35, width=1.0)
  nx.draw_networkx_nodes(
      graph,
      pos,
      ax=ax,
      node_size=[220 + 90 * degrees[n] for n in graph.nodes()],
      node_color=[degrees[n] for n in graph.nodes()],
      cmap="viridis",
      linewidths=0.5,
      edgecolors="white",
  )
  nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9, font_color="white")

  ax.set_title(
      f"{config}  |  {row['algorithm']}  |  "
      f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges\n"
      f"{question}  ->  {row['answer'].strip()}",
      fontsize=10,
  )
  ax.axis("off")
  fig.tight_layout()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, dpi=150)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", default="node_degree")
  parser.add_argument("--split", default="zero_shot_test")
  parser.add_argument("--index", type=int, default=0)
  parser.add_argument(
      "--count",
      type=int,
      default=5,
      help="how many rows to verify; only --index is drawn",
  )
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()

  rows = fetch_rows(args.config, args.split, args.index, args.count)
  print(f"{args.config}/{args.split}: verifying {len(rows)} rows from index {args.index}")

  failures = 0
  for offset, row in enumerate(rows):
    graph = parse_graph(row["question"])
    problems = check(graph, row, args.config)
    status = "ok" if not problems else "FAIL"
    print(
        f"  [{args.index + offset:>3}] {graph.number_of_nodes():>3} nodes"
        f" {graph.number_of_edges():>4} edges  {status}"
    )
    for problem in problems:
      print(f"        - {problem}")
    failures += bool(problems)

  if failures:
    raise SystemExit(f"{failures}/{len(rows)} rows failed verification; not drawing")

  row = rows[0]
  out_path = args.out or Path("figures") / f"{args.config}_{args.index}.png"
  draw(parse_graph(row["question"]), row, args.config, out_path)
  print(f"\nwrote {out_path}")


if __name__ == "__main__":
  main()
