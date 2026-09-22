"""Raw `runs/*.jsonl` -> one tidy per-response frame for the 40-node density sweep.

Everything downstream (`scripts/raw_trends.py`) reads this one CSV. Nothing here
consults an existing analysis CSV or doc -- the point of this frame is to be an
independent rebuild from the raw generations plus `graphtalk/`.

The corpus is regenerated, not looked up. `scripts/build_size_sweep.py` derives its
ER seed from (density, size, index) with no task term, so every graph, every queried
node and every gold answer is recoverable from `instance_id` alone. We assert the
regenerated gold matches the gold recorded in the jsonl for *every* row; a single
mismatch means the seed formula drifted and the frame is abandoned rather than
silently built against the wrong graphs.

  PYTHONPATH=. python scripts/build_raw_frame.py

Writes csv2/raw-trends/frame.csv.
"""

import argparse
import collections
import csv
import glob
import json
import os
import random
import re
import sys

import networkx as nx

from graphtalk import diverse_corpus
from graphtalk import graphqa
from graphtalk import primers
from graphtalk import prompts
from graphtalk import scoring

SEED = 20260906           # scripts/build_size_sweep.py DEFAULT_SEED
SIZE = 40
ARMS = ("qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think")
TAGS = ("densfull40", "densfull40hi")

_KEY = re.compile(r"^(.+)/size(\d+)/p([\d.]+)/(\d+)$")

# Behaviour markers. `narrates_neighbors` was originally written as `cites_primer`;
# hand-validation showed 20/20 of its positives cite the *edge list* ("Node N is
# connected to ...") and none cite the primer, so it is named for what it measures.
# Deliberately conservative: each is validated against a
# hand-labelled sample by `raw_trends.py --question markers`, and any marker under
# 0.9 precision is dropped rather than loosened until it agrees.
_MARKERS = {
    "uses_degree_sum": re.compile(
        r"divide[d]?\s+by\s+2|divided by two|sum of (?:the )?degrees|handshak", re.I),
    "narrates_neighbors": re.compile(
        r"\baccording to\b|\bas (?:given|stated|listed|provided)\b"
        r"|\bthe (?:list|description|primer) (?:says|states)\b", re.I),
}
_NODE_MENTION = re.compile(r"\bNode\s+\d+", re.I)

_INT_TASKS = ("node_count", "edge_count", "node_degree")


def seed_of(density, index):
  """The seed `build_size_sweep.build` uses for a pinned-density graph."""
  return SEED + 1000000 * (1 + int(round(density * 1000))) + 1000 * SIZE + index


class Corpus:
  """Regenerates and caches each graph, its topology, and its per-task rows.

  One graph serves all six tasks at a given (density, index) -- the seed has no
  task term -- so `graph_id` is (density, index) and is what a clustered bootstrap
  must resample. `instance_id` prepends the task and is NOT a graph identity.
  """

  def __init__(self):
    self._graphs = {}
    self._rows = {}
    self._primer_chars = {}

  def graph(self, density, index):
    key = (density, index)
    if key not in self._graphs:
      s = seed_of(density, index)
      g = graphqa.canonical(nx.erdos_renyi_graph(SIZE, density, seed=s))
      n = g.number_of_nodes()
      m = g.number_of_edges()
      clustering = nx.clustering(g)
      topo = {
          "n_nodes": n,
          "n_edges": m,
          "density_real": round(2 * m / (n * (n - 1)), 6) if n > 1 else 0.0,
          "mean_degree": round(2 * m / n, 4) if n else 0.0,
          "n_components": nx.number_connected_components(g),
          "largest_cc": len(max(nx.connected_components(g), key=len)) if n else 0,
          "n_triangles": sum(nx.triangles(g).values()) // 3,
          "mean_clustering": round(sum(clustering.values()) / n, 6) if n else 0.0,
          "encoding_chars": len(prompts.encode(g)),
      }
      self._graphs[key] = (g, topo, clustering)
    return self._graphs[key]

  def row(self, density, index, task):
    key = (density, index, task)
    if key not in self._rows:
      self.graph(density, index)
      g = self._graphs[(density, index)][0]
      self._rows[key] = diverse_corpus.make_row(
          g, task, random.Random(seed_of(density, index)))
    return self._rows[key]

  def primer_chars(self, density, index, condition):
    key = (density, index, condition)
    if key not in self._primer_chars:
      g = self.graph(density, index)[0]
      self._primer_chars[key] = len(primers.build_primer(g, condition))
    return self._primer_chars[key]


def target_features(g, clustering, task, targets, gold):
  """Per-item features -- the queried node/pair, not the graph as a whole.

  These are what make Q3 answerable: `node_degree`, `connected_nodes` and
  `edge_existence` all query the *same* node of the same graph, so holding the
  graph fixed while the required output changes isolates the cost of the output
  operation from the cost of the graph.
  """
  out = {"target_id": "", "target_degree": "", "target_clustering": "",
         "gold_is_yes": "", "pair_common_neighbors": "", "pair_distance": "",
         "gold_magnitude": ""}
  if task in ("node_degree", "connected_nodes") and targets:
    v = targets[0]
    out["target_id"] = v
    out["target_degree"] = g.degree(v)
    out["target_clustering"] = round(clustering.get(v, 0.0), 6)
  elif task == "edge_existence" and len(targets) == 2:
    a, b = targets
    out["target_id"] = a
    out["target_degree"] = g.degree(a)
    out["target_clustering"] = round(clustering.get(a, 0.0), 6)
    out["pair_common_neighbors"] = len(list(nx.common_neighbors(g, a, b)))
    out["gold_is_yes"] = int(str(gold).strip().lower().startswith("y"))
    try:
      out["pair_distance"] = nx.shortest_path_length(g, a, b)
    except nx.NetworkXNoPath:
      out["pair_distance"] = -1        # -1 == endpoints in different components
  if task == "cycle_check":
    out["gold_is_yes"] = int(str(gold).strip().lower().startswith("y"))
  if task in _INT_TASKS:
    try:
      out["gold_magnitude"] = int(str(gold).strip())
    except ValueError:
      pass
  return out


FIELDS = [
    "arm", "model_size", "thinking", "task", "condition", "density_class",
    "index", "instance_id", "graph_id",
    "exact", "f1", "abs_error", "signed_error", "parsed", "hit_cap",
    "pred", "gold",
    "n_nodes", "n_edges", "density_real", "mean_degree", "n_components",
    "largest_cc", "n_triangles", "mean_clustering",
    "target_id", "target_degree", "target_clustering", "gold_is_yes",
    "pair_common_neighbors", "pair_distance", "gold_magnitude",
    "primer_chars", "encoding_chars",
    "n_new_tokens", "resp_chars", "n_node_mentions",
    "uses_degree_sum", "narrates_neighbors", "enumerates_nodes",
]


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--out", default="csv2/raw-trends/frame.csv")
  ap.add_argument("--arms", nargs="+", default=list(ARMS))
  args = ap.parse_args()

  corpus = Corpus()
  seen = set()
  n_rows = n_dupes = 0
  cells = collections.Counter()

  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  with open(args.out, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()

    for arm in args.arms:
      for tag in TAGS:
        paths = sorted(glob.glob("runs/%s.%s.shard*.jsonl" % (arm, tag)))
        paths += sorted(glob.glob("runs/%s.%s.jsonl" % (arm, tag)))
        for path in paths:
          for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            key = (arm, rec["instance_id"], rec["condition"], rec["style"])
            if key in seen:
              n_dupes += 1
              continue
            seen.add(key)

            m = _KEY.match(rec["instance_id"])
            if not m:
              sys.exit("unparseable instance_id: " + rec["instance_id"])
            task, size = m.group(1), int(m.group(2))
            dens, index = float(m.group(3)), int(m.group(4))
            if size != SIZE:
              sys.exit("unexpected size %d in %s" % (size, rec["instance_id"]))

            g, topo, clustering = corpus.graph(dens, index)
            row = corpus.row(dens, index, task)
            # The frame is abandoned, not repaired, if the corpus drifted.
            if str(row["gold"]) != str(rec["gold"]):
              sys.exit(
                  "gold mismatch at %s: regenerated %r != recorded %r. The seed "
                  "formula no longer reproduces this corpus; fix that before "
                  "trusting any number built from it."
                  % (rec["instance_id"], row["gold"], rec["gold"]))

            text = rec["response"] or ""
            pred = scoring.extract_answer(text, task)
            sc = scoring.score_one(pred, rec["gold"], task)

            signed = ""
            if task in _INT_TASKS and pred is not None:
              try:
                signed = (int(scoring.normalize(pred))
                          - int(scoring.normalize(rec["gold"])))
              except (ValueError, TypeError):
                signed = ""

            mentions = len(_NODE_MENTION.findall(text))
            dens_class = "%g" % dens
            out = {
                "arm": arm,
                "model_size": "4b" if "4b" in arm else "1.7b",
                "thinking": int(arm.endswith("-think")),
                "task": task,
                "condition": rec["condition"],
                "density_class": dens_class,
                "index": index,
                "instance_id": rec["instance_id"],
                # (density, index) -- the true graph identity. Six tasks share it.
                "graph_id": "p%s/%d" % (dens_class, index),
                "exact": sc["exact"],
                "f1": sc["primary"],
                "abs_error": ("" if sc["absolute_error"] is None
                              else sc["absolute_error"]),
                "signed_error": signed,
                "parsed": int(sc["parsed"]),
                "hit_cap": int(bool(rec.get("hit_cap"))),
                "pred": pred if pred is not None else "",
                "gold": rec["gold"],
                "primer_chars": corpus.primer_chars(dens, index, rec["condition"]),
                "n_new_tokens": rec.get("n_new_tokens", ""),
                "resp_chars": len(text),
                "n_node_mentions": mentions,
                "enumerates_nodes": int(mentions >= 0.5 * topo["n_nodes"]),
            }
            out.update(topo)
            out.update(target_features(g, clustering, task, row["targets"],
                                       rec["gold"]))
            for name, rx in _MARKERS.items():
              out[name] = int(bool(rx.search(text)))
            writer.writerow(out)
            n_rows += 1
            cells[(arm, task, rec["condition"], dens_class)] += 1

  print("wrote %d rows to %s (%d duplicate keys dropped)"
        % (n_rows, args.out, n_dupes))
  print("regenerated %d graphs; every regenerated gold matched the recorded gold"
        % len(corpus._graphs))

  bad = {k: v for k, v in cells.items() if v != 100}
  if bad:
    print("WARNING: %d cells are not exactly n=100:" % len(bad))
    for k, v in sorted(bad.items())[:20]:
      print("  %s -> %d" % (k, v))
  else:
    print("all %d (arm, task, condition, density) cells are n=100" % len(cells))


if __name__ == "__main__":
  main()
