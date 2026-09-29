<h1 align="center">Talk Like a Structured Graph?</h1>

<p align="center">
  <em>When a black-box language model reads a textual graph,<br>
  do explicit structural statistics improve execution of graph queries?</em>
</p>

<p align="center">
  <a href="paper/paper.pdf"><img alt="Paper" src="https://img.shields.io/badge/paper-PDF-b31b1b?style=flat-square"></a>
  <a href="https://arxiv.org/abs/2310.04560"><img alt="Builds on arXiv:2310.04560" src="https://img.shields.io/badge/builds%20on-arXiv%3A2310.04560-555?style=flat-square"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776ab?style=flat-square">
  <img alt="Models: Qwen3" src="https://img.shields.io/badge/models-Qwen3%201.7B%20%7C%204B-6f42c1?style=flat-square">
</p>

A course project for *Machine Learning with Graphs* (Tel Aviv University) that
extends [Talk like a Graph](https://arxiv.org/abs/2310.04560) (Fatemi et al.).
We prepend a **primer** (short factual sentences about each node's local structure)
to GraphQA's text encoding of a graph, and compare the same graphs with and without
it in paired tests. The paper: **Talk Like a Structured Graph? Structural Statistics
Help Small LLMs Mainly by Stating the Answer** ([PDF](paper/paper.pdf)).

## The experiment

<p align="center">
  <img src="docs/img/primer_usage.png" width="900" alt="The seven primer conditions, each in its own colour with its sentence about node 2, and the prompt the model reads: one primer, the incident encoding and the question">
  <br>
  <sub>How a primer is used. Left: the seven conditions, each with its sentence about node 2 of a
  five-node example (the study's graphs have 40 nodes). Right: the prompt the model reads, one primer
  followed by GraphQA's incident encoding and the question. Degree and all state the answer to this
  example question. Paper Figure 1.</sub>
</p>

Before the encoding, we place either no primer; one sentence per node stating its
degree, local clustering coefficient, or two- and three-step random-walk return
probabilities (a truncated RWSE); or a sentence combining all three statistics.
Controls give the component count or name each node without stating a statistic
(filler). The graph, encoding, and question remain fixed across primer conditions.

We evaluate Qwen3-1.7B and Qwen3-4B, each with thinking off (P) and on (T), on
40-node independent-edge graphs G(n,p) at p ∈ {.10, .20, .35, .50}, extending
node-degree and edge-existence tests to p ∈ {.65, .75, .85}. Every condition and arm
receives the same 100 graphs per task and density, allowing paired comparisons. A
rule-based graph-blind solver tests what can be answered from primer text alone.

## Key findings

<p align="center">
  <b>Explicit statistics can help by supplying answers or changing how a model uses the prompt,<br>
  but do not consistently improve graph-query execution.</b>
</p>

<p align="center">
  <img src="docs/img/primer_effect.png" width="760" alt="Dumbbell chart of node-degree accuracy per model. The degree primer, which states the answer, raises Qwen3-1.7B from 60.50% to 68.00% and Qwen3-1.7B thinking from 76.25% to 85.00%, lowers Qwen3-4B from 99.25% to 92.75%, and moves Qwen3-4B thinking from 97.00% to 99.50%, not significantly. The clustering primer moves each model by at most 3 points, none significantly.">
  <br>
  <sub>Node degree on 40-node graphs, 400 paired graphs per model. Blue and red changes are significant
  (Benjamini–Hochberg q &lt; .05); gray ones are not.
  Paper Figure 3; details: <a href="docs/results/n40-sweep.md#3-every-primer-against-no-primer">n40-sweep §3</a>.</sub>
</p>

- **Stated degrees improve node-degree accuracy when unaided counting is unreliable,
  but reduce it for a model that already counts accurately.** The degree primer raises
  node-degree accuracy by 7.5 points for plain 1.7B and 8.8 for thinking 1.7B, but
  lowers it by 6.5 for plain 4B, whose unaided accuracy is 99.25%.
  [n40-sweep §3](docs/results/n40-sweep.md#3-every-primer-against-no-primer)
- **This benefit does not establish improved graph reading:** degrees directly supply
  the queried value, and responses often use a lookup rather than the incident edges.
  At p = .50, 62% of plain 4B's responses have the direct-answer pattern rather than
  an explicit neighbor list.
  [n40-sweep §4](docs/results/n40-sweep.md#4-a-primer-that-states-the-answer-changes-the-procedure)
- **Clustering and return probabilities, which do not directly state the answer, yield
  smaller effects that vary by model and task**, with no significant gain in the joint
  accuracy of degree and neighbor-set queries.
  [n40-sweep §7](docs/results/n40-sweep.md#7-side-information-is-small-and-non-specific)
- **A component-count control also changes how a model reads the graph** despite
  conveying little new information on these graphs: responses restate the queried
  node's incident line much more often with it than with no primer or filler (98.5%,
  63.0%, and 21.8%).
  [n40-sweep §7](docs/results/n40-sweep.md#7-side-information-is-small-and-non-specific)
- **Even a correct cycle label can cite a nonexistent edge.** Among plain 1.7B's
  correct, finished responses, 43.5% cite an invented cycle containing a nonexistent
  edge.
  [cycles §5](docs/investigate_connections_and_cycles.md#5-cycle_check-do-correct-answers-rest-on-real-cycles)

<p align="center">
  <img src="docs/img/headroom.png" width="560" alt="Scatter plot of the degree primer's effect on node degree against the no-primer correct share, one point per model and edge density. Gains are largest where a model is right about half the time and turn into losses where it is right most of the time.">
  <br>
  <sub>The degree primer's effect on node degree against the no-primer correct share, per arm and
  density. Hollow: plain arm at p ≥ .65 (smaller budget). Paper Figure 4; details:
  <a href="docs/results/n40-sweep.md#4-a-primer-that-states-the-answer-changes-the-procedure">n40-sweep §4</a>.</sub>
</p>

<details>
<summary><b>Every primer against no primer</b> (paper Table 1)</summary>

| Task | Arm | none | degree | clustering | RWSE | all | components | filler |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| Degree ★ | 1.7P | 60.50 | **+7.5**<sup>f</sup> | +3.0 | +4.0 | **+10.2**<sup>f</sup> | +1.8 | −0.5 |
| | 1.7T | 76.25 | **+8.8**<sup>f</sup> | +1.5<sup>f</sup> | +2.5<sup>f</sup> | **+12.2**<sup>f</sup> | −1.0 | −3.0 |
| | 4P | 99.25 | **−6.5**<sup>f</sup> | −0.5 | **−2.2** | **−8.0**<sup>f</sup> | +0.2 | −0.5 |
| | 4T | 97.00 | +2.5 | +1.8 | −1.2<sup>f</sup> | +0.5 | +0.2 | +2.0 |
| Neighbors | 1.7P | 72.50 | +3.5<sup>f</sup> | +1.2<sup>f</sup> | +1.0<sup>f</sup> | +2.0<sup>f</sup> | +0.8<sup>f</sup> | **−6.2** |
| | 1.7T | 78.75 | **+6.2** | +2.0 | +0.2 | **+6.2** | +2.5 | +2.5 |
| | 4P | 86.00 | −3.0<sup>f</sup> | +1.0<sup>f</sup> | +0.0<sup>f</sup> | −2.8<sup>f</sup> | **+9.2**<sup>f</sup> | **−12.5** |
| | 4T | 95.50 | +0.5 | −0.8 | +0.5 | −1.2 | +0.5 | +1.2 |
| Edge count ★ | 1.7P | 1.25 | +3.0† | −0.5† | −0.8† | +0.8† | −1.0† | −0.5† |
| | 1.7T | 15.75 | −3.2† | −1.2† | −3.2† | +5.0† | −0.8† | −1.5† |
| | 4P | 2.00 | **+27.2**<sup>f</sup> | −1.2<sup>f</sup> | −1.5<sup>f</sup> | **+3.8** | −1.2<sup>f</sup> | +1.2 |
| | 4T | 20.00 | +8.5† | +24.8† | +10.2† | +1.8† | +24.8† | +12.2† |
| Edge exists | 1.7P | 70.50 | +4.0<sup>f</sup> | +3.8<sup>f</sup> | +5.0<sup>f</sup> | **+11.5**<sup>f</sup> | **−8.2**<sup>f</sup> | **−15.8** |
| | 1.7T | 99.00 | −2.5 | −1.0 | −1.2 | −1.8 | −0.8 | −0.5 |
| | 4P | 90.50 | **−4.8** | **+4.0**<sup>f</sup> | **−10.0** | **−10.0** | −2.8<sup>f</sup> | **−7.0** |
| | 4T | 99.75 | +0.0 | −0.8 | −1.2 | −1.0 | −0.5 | +0.0 |

Primary result: correct share without a primer (% of all prompts) and paired change
under each primer (points), on the same 400 graphs per row (p ≤ .50). ★: `degree` and
`all` supply the degree answer or every term needed to compute edge count. Bold:
q < .05 against no primer (BH within arm, task and control); <sup>f</sup>: q < .05
against filler; †: at least 15% truncation on either side, and no significance marks.
P/T: plain/thinking. Intervals:
[n40-sweep §3](docs/results/n40-sweep.md#3-every-primer-against-no-primer).

</details>

## Repository map

| Folder | What's inside | Start here |
|---|---|---|
| [`paper/`](paper/) | The paper, its LaTeX source, and the scripts behind every table and figure | [paper/README.md](paper/README.md) |
| [`docs/`](docs/) | The results, one doc per family of runs, plus the analyses behind the paper's discussion | [docs/README.md](docs/README.md) |
| [`scripts/`](scripts/) | The pipeline, from prompt building to every analysis | [scripts/README.md](scripts/README.md) |
| [`data/`](data/) | The committed inputs: prompts, the models' raw responses, the solver bars | [data/README.md](data/README.md) |
| [`outputs/`](outputs/) | What the scripts print and write; the docs and paper cite these | [scripts/README.md](scripts/README.md#outputs) |
| [`graphtalk/`](graphtalk/) | The package: primers, the graph-blind solver, prompts, scoring | module docstrings |
| [`cluster/`](cluster/) | How responses were generated on the TAU GPU cluster | [cluster/README.md](cluster/README.md) |
| [`preliminary/`](preliminary/) | The pilot and screens that chose the 40-node settings | [preliminary/README.md](preliminary/README.md) |
| [`talk_like_a_graph/`](talk_like_a_graph/) | Google Research's reference code, vendored | [UPSTREAM.md](talk_like_a_graph/UPSTREAM.md) |

## Quickstart

Only generation needs a GPU; everything else reruns on a laptop from the committed
responses.

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
uv run --no-sync pytest -q                     # ~830 tests
```

Reproduce the main experiment's numbers from the committed responses (no GPU):

```bash
export PYTHONPATH=. PYTHONUTF8=1
uv run --no-sync python scripts/build_raw_frame.py
uv run --no-sync python scripts/primer_findings.py --csv-dir outputs/n40-sweep > outputs/n40-sweep/primer_findings.txt
uv run --no-sync pytest -q tests/test_results_docs.py
```

<details>
<summary><b>Reproduce every number</b> (about an hour on a laptop)</summary>

The full list, in order, is in [scripts/README.md](scripts/README.md#reproduce-everything).
It rebuilds the frame, reruns each analysis into `outputs/`, reruns the density
follow-ups, and then the paper's table and figure scripts.

</details>

<details>
<summary><b>Generate responses on a GPU</b></summary>

`scripts/run_sweep.py` is the only stage that needs a GPU (`pip install -e ".[gpu]"`).
[cluster/README.md](cluster/README.md) covers how every response in `data/runs/`
was generated on the TAU CS cluster, and
[cluster/run-4b-density-sweep.md](cluster/run-4b-density-sweep.md) is the exact
recipe for one arm.

</details>

<details>
<summary><b>Build the paper</b></summary>

```bash
cd paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

[paper/README.md](paper/README.md) also regenerates each table and figure, and
says where every number in the paper comes from.

</details>

## Earlier work

The 40-node design came out of earlier screens:
- a pilot on GraphQA's published 5–19-node graphs, where most models were
  near ceiling without a primer;
- Game-of-Thrones node names;
- a size sweep;
- a difficulty ladder and a retrieval probe.

These live in [`preliminary/`](preliminary/README.md) with their own scripts,
data and tests, all still runnable. Replaced drafts of the paper and the
analyses are kept in git tag `pre-cleanup` (`git checkout pre-cleanup`).

## Credits

Gal Barak ([@GalBrk](https://github.com/GalBrk)), Nitzan Zacharia
([@NitzanZacharia](https://github.com/NitzanZacharia)) and Inbal Moryles
([@iinbal](https://github.com/iinbal)). The graph generators and
text encoders in `talk_like_a_graph/` are Google Research's (Apache-2.0); see
[UPSTREAM.md](talk_like_a_graph/UPSTREAM.md) for the commit and the local changes.
