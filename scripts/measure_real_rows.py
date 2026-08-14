"""Re-measures the provisional generator-derived statistics on real GraphQA rows.

Every quantity in docs/plans/primer-computation.md and
docs/plans/shortcut-ceilings.md was originally measured on
`graph_generators.generate_graphs(500, "er", False, random_seed=1234)`, because
the HuggingFace rows API was unreachable (403 at the proxy) when those documents
were written. Both flagged the numbers as provisional. This script is the
re-measurement.

The generator turned out to be the dataset. `generate_graphs(500, "er", False,
random_seed=1234)` and the published `zero_shot_test` split are the same multiset
of graphs -- 492 distinct graphs with identical multiplicities, nothing in one
and not the other. The published rows are a shuffle: only 2 of 500 sit at the
same index, which is why an index-by-index spot check looks like a total mismatch
and proves nothing. Section 0 asserts the multiset equality rather than assuming
it, and prints which way it went; if it ever stops holding, every "exact" claim
downstream drops back to being a proxy measurement.

Two decompositions matter for reading the output:

  - Query draw vs graph draw. `edge_existence`, `connected_nodes` and
    `node_degree` sample query nodes per row, so a rate quoted per row mixes the
    graph distribution with one particular query draw. Every per-row rate here is
    reported three ways: on the real rows with their real published queries, on
    the real graphs with resampled queries, and on generated graphs with
    resampled queries. The middle column is what isolates the two.
  - Aggregation of the RWSE/degree correlation. Pooling every node of every graph
    reverses the sign of the result; use the mean of per-graph r. Section 1.4
    prints all three aggregations so the reversal stays visible.

Rows are cached under .cache/graphqa_rows so re-runs do not re-fetch.

  PYTHONPATH=. .venv/bin/python scripts/measure_real_rows.py
  PYTHONPATH=. .venv/bin/python scripts/measure_real_rows.py --shortcut-table
"""

import argparse
import collections
import json
import os
import random
import sys
import time
import urllib.error

import networkx as nx
import numpy as np

from graphtalk import graphqa
from graphtalk import primers
from graphtalk import shortcuts

BAR = "-" * 78

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(REPO_ROOT, ".cache", "graphqa_rows")

# The rows API caps `length` per request; 100 is comfortably inside it.
PAGE = 100

CONFIGS = (
    "node_count",
    "edge_count",
    "node_degree",
    "connected_nodes",
    "edge_existence",
    "cycle_check",
)

# The paper's own figures, for the two quantities it reports (arXiv:2310.04560).
PAPER = {"edge_existence_no": 0.5396, "cycle_check_yes": 0.8196}

# What docs/plans/primer-computation.md and docs/plans/shortcut-ceilings.md
# recorded from the generator, keyed the way the report prints them. Held here so
# the script prints the comparison rather than leaving it to a reader with two
# documents open.
GENERATOR_CLAIM = {
    "edge_existence_yes": 0.503,
    "cycle_check_yes": 0.832,
    "connected_nodes_isolated": 0.090,
    "rwse_r_k2": 0.65,
    "rwse_r_k3": 0.89,
    "connected": 0.740,
    "mean_components": 2.09,
    "max_components": 16,
    "has_isolated": 0.26,
    "edgeless": 0.012,
    "single_component": 0.736,
    "triangle_free": 0.19,
}

GENERATOR_PRIMER_CHARS = {
    "none": 0,
    "components": 37,
    "degree": 265,
    "clustering": 497,
    "filler": 507,
    "rwse": 905,
    "all": 1441,
}

# Acceptance windows from section 6 of the primer plan (about 3 batch sd).
RWSE_WINDOWS = {2: (0.50, 0.75), 3: (0.83, 0.94)}


# --- fetching -------------------------------------------------------------


def _fetch_page(config, split, offset, length, attempts=6, pause=1.0):
  """One page, retried with exponential backoff.

  The rows API answers 429 well before the row budget of this script is spent, so
  a plain loop over pages fails partway through and loses everything fetched so
  far. Backing off is cheaper than re-fetching.
  """
  delay = pause
  for attempt in range(attempts):
    try:
      return graphqa.fetch_rows(config, split, offset, length)
    except urllib.error.HTTPError as error:
      if error.code not in (429, 500, 502, 503) or attempt == attempts - 1:
        raise
      print(
          f"    {config}/{split} offset {offset}: HTTP {error.code},"
          f" retrying in {delay:.0f}s",
          file=sys.stderr,
      )
      time.sleep(delay)
      delay *= 2
  raise RuntimeError("unreachable")


def load_rows(config, split, count, cache_dir, pause=1.0):
  """Returns `count` rows, fetching them page by page and caching to disk.

  The cache is keyed on config/split/count so a re-run costs no requests. It is
  plain JSON because the rows are plain JSON: no schema of ours sits in between,
  which keeps the cached bytes checkable against the API by hand. Pages are
  cached individually as well, so a run interrupted by rate limiting resumes
  rather than starting over.
  """
  os.makedirs(cache_dir, exist_ok=True)
  path = os.path.join(cache_dir, f"{config}.{split}.{count}.json")
  if os.path.exists(path):
    with open(path, encoding="utf-8") as handle:
      return json.load(handle)

  rows = []
  while len(rows) < count:
    length = min(PAGE, count - len(rows))
    page_path = os.path.join(
        cache_dir, f"{config}.{split}.page{len(rows)}-{length}.json"
    )
    if os.path.exists(page_path):
      with open(page_path, encoding="utf-8") as handle:
        page = json.load(handle)
    else:
      page = _fetch_page(config, split, len(rows), length)
      with open(page_path, "w", encoding="utf-8") as handle:
        json.dump(page, handle)
      time.sleep(pause)
    if not page:
      break
    rows.extend(page)
  with open(path, "w", encoding="utf-8") as handle:
    json.dump(rows, handle)
  return rows


def parse_rows(rows):
  """Recovers one canonical graph per row, in row order."""
  return [graphqa.parse_graph(row["question"]) for row in rows]


# --- section 0: does the corpus behave the way the documents assume ----------


def verify_corpus(rows_by_config, graphs_by_config):
  """Parse fidelity, gold-answer agreement, and cross-config graph alignment.

  Any mismatch here is a finding about the dataset or about `parse_graph`, not a
  nuisance, so it is printed loudly rather than silenced. The design documents
  assume row i of every config is the same graph; that is asserted here because
  every structural statistic below is computed once and quoted for all six tasks.
  """
  print(f"\n{BAR}\n0. CORPUS VERIFICATION")
  failures = 0
  for config in CONFIGS:
    rows, graphs = rows_by_config[config], graphs_by_config[config]
    bad_n = sum(g.number_of_nodes() != int(r["nnodes"]) for r, g in zip(rows, graphs))
    bad_m = sum(g.number_of_edges() != int(r["nedges"]) for r, g in zip(rows, graphs))
    bad_answer = []
    for index, (row, graph) in enumerate(zip(rows, graphs)):
      recomputed = graphqa.expected_answer(graph, config, row["task_description"])
      if graphqa.normalize(recomputed) != graphqa.normalize(row["answer"]):
        bad_answer.append((index, recomputed))
    failures += bad_n + bad_m + len(bad_answer)
    print(
        f"  {config:<17} {len(rows):>4} rows   nnodes/nedges mismatches"
        f" {bad_n}/{bad_m}   gold-answer mismatches {len(bad_answer)}"
    )
    for index, recomputed in bad_answer[:3]:
      print(
          f"      row {index}: dataset {rows[index]['answer']!r}"
          f" vs recomputed {recomputed!r}"
      )

  reference = graphs_by_config[CONFIGS[0]]
  for config in CONFIGS[1:]:
    aligned = sum(
      sorted(a.nodes()) == sorted(b.nodes()) and sorted(a.edges()) == sorted(b.edges())
      for a, b in zip(reference, graphs_by_config[config])
    )
    if aligned != len(reference):
      failures += 1
      print(
          f"  ALIGNMENT: {config} shares only {aligned}/{len(reference)} graphs"
          f" with {CONFIGS[0]}"
      )
  if not failures:
    print(
        f"  all six configs parse cleanly, recompute their own gold answers, and"
        f" share the same {len(reference)} graphs row for row"
    )
  return failures


def _graph_key(graph):
  return (
      graph.number_of_nodes(),
      tuple(sorted(tuple(sorted(edge)) for edge in graph.edges())),
  )


def compare_to_generator(real_graphs, seed, size):
  """Whether the generator at the test seed reproduces the published graphs.

  Row order is not the interesting question -- the published rows are shuffled
  relative to the generator's emission order, so an index-by-index comparison
  says almost nothing. The question is whether the two are the same *multiset* of
  graphs. If they are, every structural statistic taken on the generator is not a
  proxy for the published split but literally the same measurement, and the only
  quantities that can move are the ones that depend on the per-row query draw.
  """
  generated = shortcuts.generate_corpus(size, seed)
  aligned = sum(
      _graph_key(a) == _graph_key(b) for a, b in zip(real_graphs, generated)
  )
  real_counts = collections.Counter(_graph_key(g) for g in real_graphs)
  gen_counts = collections.Counter(_graph_key(g) for g in generated)
  only_real = sum((real_counts - gen_counts).values())
  only_gen = sum((gen_counts - real_counts).values())

  print(f"\n  generate_graphs({size}, 'er', False, random_seed={seed}) vs the"
        " published rows")
  print(f"    same graph at the same index : {aligned}/{size}")
  print(f"    in the rows but not generated: {only_real}")
  print(f"    generated but not in the rows: {only_gen}")
  print(f"    distinct graphs              : {len(real_counts)} real,"
        f" {len(gen_counts)} generated")
  if real_counts == gen_counts:
    print(
        "    IDENTICAL AS A MULTISET. The published split is a shuffle of the\n"
        "    generator's output at this seed, so every graph-structural number\n"
        "    the design documents took from the generator is exact, not a proxy.\n"
        "    Only quantities that depend on the per-row query draw can move."
    )
  else:
    print(
        "    NOT the same multiset -- the generator is a distributional proxy\n"
        "    only, and every structural rate below is a genuine re-measurement."
    )
  return generated


# --- helpers ---------------------------------------------------------------


def _fmt_pct(value):
  return "n/a" if value is None else f"{100 * value:5.1f}%"


def _row(label, real, resampled, generated, claim=None, paper=None):
  """One comparison line: real rows, real graphs resampled, generator, claims."""
  print(
      f"  {label:<34} {_fmt_pct(real):>6}  {resampled:>22}  {generated:>20}"
      f"  {'-' if claim is None else _fmt_pct(claim):>9}"
      f"  {'-' if paper is None else _fmt_pct(paper):>6}"
  )


def _mean_sd(values):
  array = np.asarray(values, dtype=float)
  return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _resample_rate(graphs, task, predicate, draws, seed=0):
  """Mean and sd of a per-row rate over `draws` independent query resamples.

  The generator's per-row rates depend on which query nodes were drawn, so a
  single draw conflates the query distribution with the graph distribution.
  Reporting mean and sd over resamples separates them.
  """
  rates = []
  for draw in range(draws):
    rng = random.Random(seed + draw)
    hits = 0
    for graph in graphs:
      targets = shortcuts.sample_query(graph, task, rng)
      hits += predicate(graph, targets)
    rates.append(hits / len(graphs))
  return _mean_sd(rates)


def _sd_text(pair):
  mean, sd = pair
  return f"{100 * mean:.1f}% +/- {100 * sd:.1f}"


# --- section 1: the four flagged quantities --------------------------------


def flagged_quantities(rows_by_config, graphs_by_config, generated, draws, extra):
  """The four quantities the provenance table lists as needing re-measurement."""
  print(f"\n{BAR}\n1. THE FOUR FLAGGED QUANTITIES")
  print(
      f"  {'quantity':<34} {'real':>6}  {'real graphs, resampled':>22}"
      f"  {'generator, resampled':>20}  {'gen claim':>9}  {'paper':>6}"
  )

  graphs = graphs_by_config["edge_existence"]

  # 1.1 edge_existence class balance.
  answers = collections.Counter(
      graphqa.normalize(row["answer"]) for row in rows_by_config["edge_existence"]
  )
  total = sum(answers.values())
  real_yes = answers["Yes"] / total
  has_edge = lambda g, t: g.has_edge(t[0], t[1])
  real_resampled = _resample_rate(graphs, "edge_existence", has_edge, draws)
  gen_resampled = _resample_rate(generated, "edge_existence", has_edge, draws)
  _row(
      "edge_existence Yes rate",
      real_yes,
      _sd_text(real_resampled),
      _sd_text(gen_resampled),
      GENERATOR_CLAIM["edge_existence_yes"],
      1 - PAPER["edge_existence_no"],
  )
  _row(
      "edge_existence majority baseline",
      max(real_yes, 1 - real_yes),
      _fmt_pct(max(real_resampled[0], 1 - real_resampled[0])),
      _fmt_pct(max(gen_resampled[0], 1 - gen_resampled[0])),
      0.516,
      max(PAPER["edge_existence_no"], 1 - PAPER["edge_existence_no"]),
  )

  # 1.2 cycle_check Yes rate. No query sampling, so one row per graph and the
  # resampled column is the same number as the real column by construction.
  cycle_rows = rows_by_config["cycle_check"]
  real_cycle = sum(
      graphqa.normalize(row["answer"]).startswith("Yes") for row in cycle_rows
  ) / len(cycle_rows)
  gen_cycle = sum(
      graphqa.gold_answer(g, "cycle_check").startswith("Yes") for g in generated
  ) / len(generated)
  _row(
      "cycle_check Yes rate",
      real_cycle,
      "(no query sampling)",
      _fmt_pct(gen_cycle),
      GENERATOR_CLAIM["cycle_check_yes"],
      PAPER["cycle_check_yes"],
  )

  # 1.3 connected_nodes isolated-target rate.
  connected_rows = rows_by_config["connected_nodes"]
  real_isolated = sum(
      graphqa.normalize(row["answer"]) == "No nodes" for row in connected_rows
  ) / len(connected_rows)
  isolated = lambda g, t: g.degree(t[0]) == 0
  real_iso_resampled = _resample_rate(
      graphs_by_config["connected_nodes"], "connected_nodes", isolated, draws
  )
  gen_iso_resampled = _resample_rate(generated, "connected_nodes", isolated, draws)
  _row(
      "connected_nodes 'No nodes.' rate",
      real_isolated,
      _sd_text(real_iso_resampled),
      _sd_text(gen_iso_resampled),
      GENERATOR_CLAIM["connected_nodes_isolated"],
  )

  if extra:
    print("\n  the same two paper-comparable rates over all three zero_shot splits")
    print(f"  {'split':<22} {'rows':>5}  {'edge_existence Yes':>19}"
          f"  {'cycle_check Yes':>16}")
    for split, ee_rows, cc_rows in extra:
      ee = sum(graphqa.normalize(r["answer"]) == "Yes" for r in ee_rows) / len(ee_rows)
      cc = sum(
          graphqa.normalize(r["answer"]).startswith("Yes") for r in cc_rows
      ) / len(cc_rows)
      print(f"  {split:<22} {len(ee_rows):>5}  {100 * ee:>18.1f}%  {100 * cc:>15.1f}%")

  # 1.4 RWSE/degree correlation. Aggregation is the whole point: pooling every
  # node of every graph reverses the sign, so all three are printed.
  print(f"\n{BAR}\n1.4 RWSE/DEGREE CORRELATION")
  print("  aggregation is load-bearing -- pooling reverses the sign")
  print(
      f"  {'corpus':<12} {'k':>2} {'mean of per-graph r':>20} {'defined on':>12}"
      f" {'Fisher-z':>9} {'pooled':>8}  window"
  )
  for label, corpus in (("real rows", graphs), ("generator", generated)):
    for k in (2, 3):
      per_graph, pooled_degree, pooled_rwse = [], [], []
      for graph in corpus:
        value = primers.rwse_degree_correlation(graph, k_min=k, k_max=k)[k]
        if value is not None:
          per_graph.append(value)
        table = primers.rwse(graph, k_min=k, k_max=k)
        for node in sorted(graph.nodes()):
          pooled_degree.append(graph.degree(node))
          pooled_rwse.append(table[node][k])
      mean = float(np.mean(per_graph))
      clipped = np.clip(per_graph, -0.999999, 0.999999)
      fisher = float(np.tanh(np.mean(np.arctanh(clipped))))
      pooled = float(np.corrcoef(pooled_degree, pooled_rwse)[0, 1])
      low, high = RWSE_WINDOWS[k]
      verdict = "inside" if low <= mean <= high else "OUTSIDE"
      print(
          f"  {label:<12} {k:>2} {mean:>+20.3f} {len(per_graph):>7}/{len(corpus):<4}"
          f" {fisher:>+9.3f} {pooled:>+8.3f}  [{low:.2f}, {high:.2f}] {verdict}"
      )
  print(
      "  the k=2 and k=3 means are over different populations: odd-k return is"
      "\n  exactly 0 for every node of a triangle-free graph, so k=3 is undefined"
      "\n  there. Do not describe r as 'rising' across k."
  )


# --- section 2: the structural statistics the design rests on ---------------


def structure(graphs, generated):
  """Connectivity, isolation, edgelessness, size and density.

  These are the rates the design decisions cite: the components arm's variance
  claim, the `connected_nodes` hard-case share, the `node_count` leak on edgeless
  graphs, and the claim that the encoding omits isolated nodes often enough to
  matter.
  """
  print(f"\n{BAR}\n2. STRUCTURAL STATISTICS")
  print(f"  {'quantity':<40} {'real rows':>10} {'generator':>10} {'gen claim':>10}")

  def line(label, real, gen, claim, spec="pct"):
    if spec == "pct":
      print(
          f"  {label:<40} {100 * real:>9.1f}% {100 * gen:>9.1f}%"
          f" {100 * claim:>9.1f}%"
      )
    else:
      print(f"  {label:<40} {real:>10{spec}} {gen:>10{spec}} {claim:>10{spec}}")

  def counts(corpus):
    return [primers.component_count(g) for g in corpus]

  real_c, gen_c = counts(graphs), counts(generated)
  line(
      "connected (c = 1)",
      np.mean([c == 1 for c in real_c]),
      np.mean([c == 1 for c in gen_c]),
      GENERATOR_CLAIM["connected"],
  )
  line(
      "mean component count",
      np.mean(real_c),
      np.mean(gen_c),
      GENERATOR_CLAIM["mean_components"],
      spec=".2f",
  )
  line(
      "max component count",
      max(real_c),
      max(gen_c),
      GENERATOR_CLAIM["max_components"],
      spec="d",
  )
  line(
      "contains an isolated node",
      np.mean([min(dict(g.degree()).values()) == 0 for g in graphs]),
      np.mean([min(dict(g.degree()).values()) == 0 for g in generated]),
      GENERATOR_CLAIM["has_isolated"],
  )
  line(
      "edgeless (m = 0, so c = n)",
      np.mean([g.number_of_edges() == 0 for g in graphs]),
      np.mean([g.number_of_edges() == 0 for g in generated]),
      GENERATOR_CLAIM["edgeless"],
  )
  line(
      "triangle-free (k=3 RWSE undefined)",
      np.mean([sum(nx.triangles(g).values()) == 0 for g in graphs]),
      np.mean([sum(nx.triangles(g).values()) == 0 for g in generated]),
      GENERATOR_CLAIM["triangle_free"],
  )
  print(
      "\n  c = 1 is also the share of connected_nodes rows on which the components"
      "\n  primer deterministically excludes the 'No nodes.' answer:"
      f" {100 * np.mean([c == 1 for c in real_c]):.1f}% real vs"
      f" {100 * GENERATOR_CLAIM['single_component']:.1f}% claimed."
  )

  print("\n  node count and density (generator draws n uniform 5..19, sparsity U(0,1))")
  print(f"  {'corpus':<12} {'n range':>10} {'mean n':>7} {'mean density':>13}"
        f" {'sd':>6} {'deciles of density':>34}")
  for label, corpus in (("real rows", graphs), ("generator", generated)):
    sizes = [g.number_of_nodes() for g in corpus]
    density = [nx.density(g) for g in corpus]
    deciles = np.percentile(density, [10, 30, 50, 70, 90])
    print(
        f"  {label:<12} {min(sizes):>4}..{max(sizes):<4} {np.mean(sizes):>7.2f}"
        f" {np.mean(density):>13.3f} {np.std(density):>6.3f}"
        f"   {' '.join(f'{d:.2f}' for d in deciles):>30}"
    )
  real_sizes = collections.Counter(g.number_of_nodes() for g in graphs)
  gen_sizes = collections.Counter(g.number_of_nodes() for g in generated)
  print("  n histogram (real / generator):")
  print(
      "    "
      + "  ".join(
          f"{n}:{real_sizes.get(n, 0)}/{gen_sizes.get(n, 0)}"
          for n in range(min(real_sizes), max(real_sizes) + 1)
      )
  )


def primer_lengths(graphs, generated):
  """Mean rendered primer characters per condition, against section 4's table."""
  print(f"\n{BAR}\n3. PRIMER LENGTH BY CONDITION (mean characters)")
  print(f"  {'condition':<12} {'real rows':>10} {'generator':>10} {'doc table':>10}"
        f" {'delta vs doc':>13}")
  for condition in primers.CONDITIONS:
    real = np.mean([len(primers.build_primer(g, condition)) for g in graphs])
    gen = np.mean([len(primers.build_primer(g, condition)) for g in generated])
    claimed = GENERATOR_PRIMER_CHARS[condition]
    print(
        f"  {condition:<12} {real:>10.0f} {gen:>10.0f} {claimed:>10}"
        f" {real - claimed:>+13.0f}"
    )
  print(
      "\n  What matters is the ordering, not the absolute counts: the `filler`"
      "\n  control has to sit at or above `degree` and `clustering` for the length"
      "\n  argument in section 4 of the primer plan to hold."
  )


# --- priority 3: the headline shortcut table on real graphs -----------------


def _best_on_small(task, small, fit_graphs):
  """Our strongest solver, scored on the same small graphs the exact bound covers.

  Mirrors `best_on_small` in scripts/shortcut_table.py deliberately rather than
  importing it: scripts/ is not a package, and a cross-script import would be a
  worse dependency than twelve duplicated lines. Keep the two in step.
  """
  best = 0.0
  for condition in ("degree", "all"):
    for rung in shortcuts.RUNGS:
      fit_rows = shortcuts.build_rows(fit_graphs, condition, task, rung, seed=5)
      rows = shortcuts.build_rows(small, condition, task, rung, seed=5)
      fitted = shortcuts.rank_rules(
          shortcuts.fit_rules(
              shortcuts.HEURISTICS + shortcuts.FITTED, fit_rows, task
          ),
          fit_rows,
          task,
      )
      fallback = shortcuts.majority_answer([gold for _, gold in fit_rows])
      best = max(best, shortcuts.score_solver(rows, task, fitted, fallback))
  return best


def shortcut_table(real_graphs, fit_seed, size):
  """Re-runs the headline shortcut cells with real rows as the evaluation set.

  Fitting stays on generated graphs at a different seed. That keeps the
  train/test discipline the shortcut plan insists on, and it is now disjoint by
  provenance as well as by seed, since section 0 shows the generator and the
  published rows share no graph.
  """
  print(f"\n{BAR}\n4. SHORTCUT TABLE, EVALUATED ON REAL ROWS")
  print(f"  fitted on generate_graphs(seed={fit_seed}), scored on {len(real_graphs)}"
        " published zero_shot_test graphs")
  print("  queries are resampled by shortcuts.build_rows, as in"
        " scripts/shortcut_table.py -- not the queries the dataset ships")
  fit_graphs = shortcuts.generate_corpus(size, fit_seed)

  cells = {}
  for condition in primers.CONDITIONS:
    fit_parsed = shortcuts.parse_corpus(fit_graphs, condition)
    test_parsed = shortcuts.parse_corpus(real_graphs, condition)
    for task in shortcuts.TASKS:
      for rung in shortcuts.RUNGS:
        cells[(condition, task, rung)] = shortcuts.score_cell(
            condition, task, rung, fit_graphs, real_graphs,
            fit_parsed=fit_parsed, test_parsed=test_parsed,
        )

  bad_precision = [
      (key, stat)
      for key, cell in cells.items()
      for stat in cell.theorems
      if stat["precision"] is not None and stat["precision"] < 1.0
  ]
  for key, stat in bad_precision:
    print(f"  THEOREM PRECISION BELOW 1.0 on real rows: {key} {stat}")
  if not bad_precision:
    print("  every theorem rule keeps precision 1.0 on real rows")

  print(f"\n  {'task':<17} {'baseline':>9} {'best arm':<22} {'shortcut':>9}  verdict")
  for task in shortcuts.TASKS:
    best_condition, best = None, -1.0
    baseline = cells[(next(iter(primers.CONDITIONS)), task, 3)].baseline
    for condition in primers.CONDITIONS:
      score = cells[(condition, task, 3)].shortcut
      if score > best + 1e-12:
        best, best_condition = score, [condition]
      elif abs(score - best) <= 1e-12:
        best_condition.append(condition)
    verdict = "decided" if best >= 0.99 else (
        "wide open" if best <= baseline + 1e-9 else "real headroom"
    )
    arms = ", ".join(best_condition[:3]) + ("..." if len(best_condition) > 3 else "")
    print(f"  {task:<17} {100 * baseline:>8.1f}% {arms:<22}"
          f" {100 * best:>8.1f}%  {verdict}")

  print("\n  the ladder on components x cycle_check, the one cell where rungs separate")
  base = cells[("components", "cycle_check", 1)].baseline
  rungs = [cells[("components", "cycle_check", r)].shortcut for r in shortcuts.RUNGS]
  print(
      f"    baseline {100 * base:.1f}%   rung 1 {100 * rungs[0]:.1f}%"
      f"   rung 2 {100 * rungs[1]:.1f}%   rung 3 {100 * rungs[2]:.1f}%"
  )

  print("\n  the exact island on n <= 6")
  print(f"  {'task':<17} {'rows':>5} {'determined':>11} {'bayes':>7} {'our best':>9}"
        "  verdict")
  small = [g for g in real_graphs if g.number_of_nodes() <= 6]
  for task in shortcuts.TASKS:
    result = shortcuts.exact_island(real_graphs, task)
    if not result["rows"]:
      continue
    ours = _best_on_small(task, small, fit_graphs)
    bayes = result["bayes_optimal"]
    if ours > bayes + 1e-9:
      verdict = "ABOVE BOUND -- rule reads something unstated"
    elif bayes - ours > 0.05:
      verdict = f"loose by {100 * (bayes - ours):.0f}pp: rules remain unfound"
    else:
      verdict = "tight"
    print(
        f"  {task:<17} {result['rows']:>5} {result['determined']:>10.1%}"
        f" {bayes:>6.1%} {ours:>8.1%}  {verdict}"
    )
    coverage = result["coverage"]
  print(f"  (small graphs are {coverage:.1%} of real rows)")


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--split", default="zero_shot_test")
  parser.add_argument("--rows", type=int, default=500)
  parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
  parser.add_argument("--test-seed", type=int, default=1234,
                      help="generator seed the documents' numbers came from")
  parser.add_argument("--fit-seed", type=int, default=999)
  parser.add_argument("--resamples", type=int, default=200,
                      help="query resamples for the per-row rates")
  parser.add_argument(
      "--no-extra-splits", action="store_true",
      help="skip the train/validation splits used for the paper comparison")
  parser.add_argument("--shortcut-table", action="store_true",
                      help="also re-run the shortcut cells on real graphs (priority 3)")
  args = parser.parse_args(argv)

  rows_by_config, graphs_by_config = {}, {}
  for config in CONFIGS:
    rows_by_config[config] = load_rows(
        config, args.split, args.rows, args.cache_dir
    )
    graphs_by_config[config] = parse_rows(rows_by_config[config])

  extra = []
  if not args.no_extra_splits:
    for split, count in (("zero_shot_train", 1000), ("zero_shot_validation", 500)):
      extra.append((
          split,
          load_rows("edge_existence", split, count, args.cache_dir),
          load_rows("cycle_check", split, count, args.cache_dir),
      ))
    extra.append((
        args.split, rows_by_config["edge_existence"], rows_by_config["cycle_check"]
    ))

  print(f"{BAR}\nGraphQA {args.split}: {args.rows} rows x {len(CONFIGS)} configs")
  failures = verify_corpus(rows_by_config, graphs_by_config)
  generated = compare_to_generator(
      graphs_by_config[CONFIGS[0]], args.test_seed, args.rows
  )

  flagged_quantities(
      rows_by_config, graphs_by_config, generated, args.resamples, extra
  )
  structure(graphs_by_config[CONFIGS[0]], generated)
  primer_lengths(graphs_by_config[CONFIGS[0]], generated)

  if args.shortcut_table:
    shortcut_table(graphs_by_config[CONFIGS[0]], args.fit_seed, args.rows)

  print(f"\n{BAR}")
  if failures:
    print(f"{failures} verification failures above -- read them before the numbers")
  return 1 if failures else 0


if __name__ == "__main__":
  raise SystemExit(main())
