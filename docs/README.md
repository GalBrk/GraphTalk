# Docs

What the experiments found, and the design behind them.
[← back to the repo](../README.md)

## The study

The project asks whether prepending a *primer* before a graph's text encoding and
question improves an LLM's accuracy on the six GraphQA tasks of Fatemi et al.
(2024). A primer is a few short factual sentences about each node's local
structure: its degree, clustering coefficient, or random-walk return
probabilities. The project also asks whether any benefit depends on how closely
the primer's content matches the task, and on the model's capacity.

**The main experiment** is the 40-node sweep: Qwen3-1.7B and Qwen3-4B, each with
and without thinking, on Erdős–Rényi graphs with 40 nodes at seven edge
densities. **Its follow-ups** re-run `node_degree` with:
- 400 graphs per density;
- a thinking arm;
- a length-matched control;
- a fresh seed;
- a fixed-mean-degree grid, with Qwen3-8B.

## Results

Each results doc states only what its pipeline's committed output shows, and
cites every number with the tag it is printed under.
`tests/test_results_docs.py` checks each one against that output.

| Document | What it covers | Output it cites |
|---|---|---|
| [results/n40-sweep.md](results/n40-sweep.md) | **The main experiment**: every primer against no primer, which primers state the answer, how a stated answer changes the procedure, `edge_existence`'s yes-bias, truncation | `outputs/n40-sweep/primer_findings.txt` |
| [results/density-followups.md](results/density-followups.md) | The `node_degree` follow-ups: the `clustering` effect, its replication, thinking against plain, the `filler` control, fixed mean degree, the forensics | `outputs/density-followups/density_followups.txt` |

## Analyses

These go past accuracy into what the responses say. The paper's discussion and
future work draw on them.

| Document | What it covers | Status |
|---|---|---|
| [investigate_connections_and_cycles.md](investigate_connections_and_cycles.md) | Whether correct answers rest on true claims about the graph: invented cycles, fabricated edges, misread neighbour lists, the `node_count` off-by-one. Opens with conclusions C1–C5 | current |
| [primer-robustness.md](primer-robustness.md) | How much of each effect is prompt churn, whether the arms fail on the same questions, why responses truncate. Opens with conclusions R1–R5 | current |
| [primer-directions.md](primer-directions.md) | Nine further questions about what primers do to the responses | exploratory; its numbers are test-checked |
| [primer-directions-validation.md](primer-directions-validation.md) | The blind hand-validation of primer-directions' text detectors | the five pass/fail checks pass; C2 is descriptive |
| [density-interaction.md](density-interaction.md) | Whether graph density changes each primer's effect, and whether it matters beyond task difficulty | exploratory |

## Design

| Document | What it covers |
|---|---|
| [design/features-considered.md](design/features-considered.md) | Which graph statistics were considered for the primer, and why these four |
| [design/primer-computation.md](design/primer-computation.md) | How `graphtalk/primers.py` computes and renders each statistic; its measurements are on the pilot's 5–19-node graphs |
| [design/shortcut-ceilings.md](design/shortcut-ceilings.md) | The graph-blind solver: how to tell a primer that states the answer from one that helps. Its numbers are the pilot's; the 40-node bars are in [n40-sweep.md §2](results/n40-sweep.md#2-which-primers-state-the-answer) |
| [design/graph-design-requirements.md](design/graph-design-requirements.md) | The four requirements the preliminary difficulty ladder set for a graph corpus. The main 40-node corpus meets the first (n ≥ 40) and is plain G(n,p), without the ladder's rewiring |

Also here:
[plans/rq3-gpu-tests.md](plans/rq3-gpu-tests.md), a planned GPU test of the
`clustering` effect, not yet run.

## Conventions

These rules govern the results docs and the analyses; R3 says which of them
are test-checked.

**R1: truncation is its own outcome.** Every response is exactly one of
`correct`, `wrong` or `truncated`. A response that hit the token budget is
`truncated`, whatever its abandoned text says; it is never labelled wrong and
never dropped (`graphtalk/outcomes.py`).
- Each cell reports the three outcomes as shares of all its responses.
- A primer's effect is the paired change in the correct share on the same
  graphs, with a bootstrap interval (graphs resampled within density) and an
  exact McNemar test. The change in the truncated share is reported beside it.
- Descriptions of answers use finished responses only and state their n. These
  include error size, yes-rate, false alarms, and descriptions of the response
  text.
- A cell whose truncated share is 15% or more is flagged.

**R2: metrics.** Exact match is the primary metric for every task.
`connected_nodes` is scored by exact set match, with set-F1 as a secondary
column. `edge_existence` also reports balanced accuracy and the yes-rate beside
raw accuracy, because the share of true edges rises from 10% to 85% with
density.

**R3: every number is tagged.** Each pipeline writes its full output to a
committed file in `outputs/`. A results doc names that file in a `Source:` line
and cites each number exactly as printed, followed by its tag, e.g.
`+27.3 [edgecount]`. A tag's block runs from the line that starts with `[tag]`
to the next line that starts with a tag. `tests/test_results_docs.py` checks
every cited number in `docs/results/`, `primer-directions.md` and
`density-interaction.md`. `investigate_connections_and_cycles.md` and
`primer-robustness.md` tag their numbers against a table of sources; these two
are not test-checked.

**R4: current claims only.** A doc states what the current output shows. It
does not mention earlier versions, corrections or retracted numbers.

## From the proposal to what was run

| Proposal | What was run in the main experiment | Where |
|---|---|---|
| Models: Gemma 4 (4B, 12B) and Qwen3 (8B, 14B) | A pilot ran `gemma4-e4b`, `gemma4-12b`, `qwen3-8b` and `qwen3-14b`, each with and without thinking. The main experiment runs Qwen3-1.7B and Qwen3-4B, each with and without thinking | [n40-sweep.md §1](results/n40-sweep.md#1-setup-and-measurement), [preliminary/](../preliminary/README.md) |
| Graphs: the published GraphQA graphs, 30 per task | The pilot used the published graphs (5–19 nodes). The main experiment generates Erdős–Rényi graphs with 40 nodes at seven edge densities, 100 per density | [n40-sweep.md §1](results/n40-sweep.md#1-setup-and-measurement) |
| Primer conditions: none, degree, clustering, RWSE, all three | The same five, plus two: `components`, one sentence stating the number of connected components; and the control `filler`, a preamble that names every node and states no structure, sized between the single-feature node-level primers | [n40-sweep.md §1](results/n40-sweep.md#1-setup-and-measurement) |
| RWSE: return probabilities at walk lengths 1–4, two decimals | Return probabilities after 2 and 3 steps, two decimals (`graphtalk/primers.py`, `k_min=2, k_max=3`) | [n40-sweep.md §8](results/n40-sweep.md#8-feature-resolution) |
| Encoding: incident encoding, fixed | Incident encoding, fixed (`graphtalk/prompts.py`) | — |
| Tasks: all six GraphQA tasks | All six; at 40 nodes `node_count` and `cycle_check` have a constant gold answer | [n40-sweep.md §1](results/n40-sweep.md#1-setup-and-measurement) |
| Prompting: zero-shot and chain-of-thought | Zero-shot, with and without Qwen3's native thinking mode | [n40-sweep.md §1](results/n40-sweep.md#1-setup-and-measurement) |
| Metrics: exact match; MAE secondary; set-F1 for connected nodes; majority-class baseline; McNemar per cell | Exact match for every task; `connected_nodes` uses exact set match, with set-F1 as a secondary column. Also: MAE, secondary; balanced accuracy and yes-rate for `edge_existence`; the majority-class baseline; exact McNemar with Benjamini–Hochberg correction. A response that exhausts its token budget is its own outcome (R1) | [n40-sweep.md §3](results/n40-sweep.md#3-every-primer-against-no-primer) |
| Controls beyond the paired comparison | A graph-blind solver that reads only the primer text, to separate primers that state the answer from those that do not | [n40-sweep.md §2](results/n40-sweep.md#2-which-primers-state-the-answer) |
