# Results

The current results of this project, one document per family of runs. Each
document states only what its pipeline's committed output shows, cites every
number with the tag it is printed under, and is checked against that output by
`tests/test_results_docs.py`. Earlier analyses and paper drafts are in
[`superseded/`](../../superseded/README.md).

## The study

The project asks whether prepending a *primer* (short factual sentences about
each node's local structure: its degree, clustering coefficient, or random-walk
return probabilities) before a graph's text encoding and question improves an
LLM's accuracy on the six GraphQA tasks of Fatemi et al. (2024), and whether any
benefit depends on how closely the primer's content matches the task and on the
model's capacity.

**The main experiment** is the 40-node sweep: Qwen3-1.7B and Qwen3-4B, each with
and without thinking, on Erdős–Rényi graphs with 40 nodes at seven edge
densities.

## Documents

| Family of runs | Runs | Document | Pipeline | Output |
|---|---|---|---|---|
| 40-node sweep (main experiment) | `runs/*.densfull40*`, `runs/*.densfull40hi*`; its `clustering` replication also reads `runs/qwen3-1.7b.{degdens40,degdens40hi,degdensrep,degfixdeg}.*` | [n40-sweep.md](n40-sweep.md) | `scripts/build_raw_frame.py` → `scripts/primer_findings.py`, `scripts/legacy_claims.py` | `csv2/raw-trends/primer_findings.txt`, `csv2/raw-trends/legacy_claims.txt` |

## From the proposal to what was run

| Proposal | What was run in the main experiment | Where |
|---|---|---|
| Models: Gemma 4 (4B, 12B) and Qwen3 (8B, 14B) | A pilot ran `gemma4-e4b`, `gemma4-12b`, `qwen3-8b` and `qwen3-14b`, each with and without thinking; the main experiment runs Qwen3-1.7B and Qwen3-4B, each with and without thinking | [n40-sweep.md §1](n40-sweep.md#1-setup-and-measurement) |
| Graphs: the published GraphQA graphs, 30 per task | The pilot used the published graphs (5–19 nodes); the main experiment generates Erdős–Rényi graphs with 40 nodes at seven edge densities, 100 per density | [n40-sweep.md §1](n40-sweep.md#1-setup-and-measurement) |
| Primer conditions: none, degree, clustering, RWSE, all three | The same five, plus `components` (one sentence stating the number of connected components) and, as a control, `filler` (a preamble that names every node and states no structure; its length lies between those of the single-feature node-level primers) | [n40-sweep.md §1](n40-sweep.md#1-setup-and-measurement) |
| RWSE: return probabilities at walk lengths 1–4, two decimals | Return probabilities after 2 and 3 steps, two decimals (`graphtalk/primers.py`, `k_min=2, k_max=3`) | [n40-sweep.md §8](n40-sweep.md#8-feature-resolution) |
| Encoding: incident encoding, fixed | Incident encoding, fixed (`graphtalk/prompts.py`) | — |
| Tasks: all six GraphQA tasks | All six; at 40 nodes `node_count` and `cycle_check` have a constant gold answer | [n40-sweep.md §1](n40-sweep.md#1-setup-and-measurement) |
| Prompting: zero-shot and chain-of-thought | Zero-shot, with and without Qwen3's native thinking mode | [n40-sweep.md §1](n40-sweep.md#1-setup-and-measurement) |
| Metrics: exact match; MAE secondary; set-F1 for connected nodes; majority-class baseline; McNemar per cell | Exact match for every task (for `connected_nodes`, exact set match, with set-F1 as a secondary column); MAE secondary; balanced accuracy and yes-rate for `edge_existence`; the majority-class baseline; exact McNemar with Benjamini–Hochberg correction; a response that exhausts its token budget is reported as truncated, its own outcome | [n40-sweep.md §3](n40-sweep.md#3-every-primer-against-no-primer) |
| Controls beyond the paired comparison | A graph-blind solver that reads only the primer text, to separate primers that state the answer from those that do not | [n40-sweep.md §2](n40-sweep.md#2-which-primers-state-the-answer) |
