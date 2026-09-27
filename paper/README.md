# Paper

**Talk Like a Structured Graph: Augmenting Text Encodings with Structural
Statistics**: [`structured_graph.pdf`](structured_graph.pdf).
[← back to the repo](../README.md)

## Build

ACL 2023 style (`acl2023.sty`, `acl_natbib.bst`, `custom.bib`). From this folder:

```sh
pdflatex structured_graph.tex
bibtex structured_graph
pdflatex structured_graph.tex
pdflatex structured_graph.tex
```

## Regenerate the tables and figures

Every float is generated from the repo's outputs. The one exception is Figure 3,
whose values are copied from the `[edgecount]` block. Run these from this folder
once the pipeline in [`scripts/README.md`](../scripts/README.md) has written
`../outputs/`:

```sh
python metric_audit.py --frame ../outputs/n40-sweep/frame.csv > metric_audit.txt
python primary_metric_tables.py
python matrix_audit.py --frame ../outputs/n40-sweep/frame.csv \
    --report ../outputs/n40-sweep/primer_findings.txt --output main_matrix.tex
python truncation_table.py --frame ../outputs/n40-sweep/frame.csv --output main_truncation.tex
python dense_extension_table.py --frame ../outputs/n40-sweep/frame.csv --output dense_extension.tex
python robustness_audit.py > robustness_audit.txt
python floats.py
python plot_outcomes.py
```

| File | Written by | Reads |
|---|---|---|
| `main_matrix.tex` (Results matrix) | `matrix_audit.py` | `frame.csv`, and checks all 144 contrasts against `[main]` |
| `fig_headroom.pdf` (Figure 1) | `floats.py` | `primer_cells.csv`; also writes `docs/img/fig_headroom.png` |
| `fig_clustering.pdf` (Figure 2) | `floats.py` | `[clustarms]`, `[replic]`, `density_followups.txt` |
| `table_cycles.tex` (Table 2) | `floats.py` | `[ccanswer]`, `[cctest]` in `check_cycle_claims.txt` |
| `edge_count_outcomes.pdf` (Figure 3) | `plot_outcomes.py` | values copied from `[edgecount]` |
| `primary_metric_tables.tex` | `primary_metric_tables.py` | `metric_audit.txt`, `[eemain]` |
| `main_truncation.tex`, `dense_extension.tex` | `truncation_table.py`, `dense_extension_table.py` | `frame.csv` (`[dense-frame]`) |
| `metric_audit.txt`, `robustness_audit.txt` | `metric_audit.py`, `robustness_audit.py` | `frame.csv`; the density follow-up runs |

`metric_audit.py` runs 50,000 seeded paired random-sign draws for set-F1 and
applies Benjamini–Hochberg within each arm's six primer contrasts.
`robustness_audit.py`'s `[interaction]` applies BH within three exploratory
density interactions, and `[bundle-direct]` within six paired high-density
alternatives. Both count a truncated response as its own outcome. No new model
inference was run for any of these.

The paper cites the results at `results-sot` commit `746d2ab`
(`graphtalkresults2026`), where these outputs were still under
`csv2/raw-trends/`. They now live in `outputs/`, with the same content.

## Where each part of the paper comes from

| Paper part | Source |
|---|---|
| Question and motivation | [results/README.md](../docs/results/README.md), "The study" |
| What the proposal planned and what was run (models, graphs, primers, tasks, metrics) | [results/README.md](../docs/results/README.md), "From the proposal to what was run"; primer definitions in `graphtalk/primers.py` |
| The pilot, and why the main experiment moved to 40-node graphs | [`preliminary/README.md`](../preliminary/README.md). Say it in words; the paper cites no pilot numbers |
| Setup and measurement | [n40-sweep.md](../docs/results/n40-sweep.md) §1 |
| Which primers state the answer (the graph-blind solver) | n40-sweep.md §2; design in [plans/shortcut-ceilings.md](../docs/plans/shortcut-ceilings.md) |
| Main results | n40-sweep.md §3–9 |
| The `node_degree` follow-ups: the `clustering` effect, its replication, thinking against plain, the `filler` control, fixed mean degree, the forensics | [density-followups.md](../docs/results/density-followups.md) §2–8 |
| Limitations | n40-sweep.md §10, density-followups.md §9 |
| Future work | n40-sweep.md, density-followups.md, [primer-directions.md](../docs/primer-directions.md), [primer-robustness.md](../docs/primer-robustness.md), [investigate_connections_and_cycles.md](../docs/investigate_connections_and_cycles.md) |

## Rules for numbers

1. **Every number comes from a tagged citation**, copied as printed. The
   citation lives in one of three places:
   - a doc in `docs/results/`;
   - one of the analysis docs: `primer-directions.md`, `primer-robustness.md`,
     `investigate_connections_and_cycles.md` or `density-interaction.md`;
   - this folder's `metric_audit.txt` or `robustness_audit.txt`.

   Put the tag in a LaTeX comment on the same line, e.g. `+3.8~points % [ddplain]`,
   so each number can be traced to its block in the script output. If the paper
   needs a number nothing prints yet, add it to the script and the doc first,
   with a tag. `tests/test_results_docs.py` checks the numbers in `docs/results/`
   and `primer-directions.md` against their outputs. Never cite `preliminary/`.
2. **Truncation is its own outcome.** A response is correct, wrong or truncated.
   An effect is the paired change in the correct share of *all* responses, in
   percentage points, with the change in the truncated share beside it when
   that change is not small. A share computed over finished responses only
   (error size, tokens, the "answers 39" share) must say so, and is never
   called accuracy.
3. **Exact match is the primary metric.** Set-F1 on `connected_nodes` is
   secondary. For `edge_existence` report balanced accuracy and the yes-rate
   next to raw accuracy.
4. **Significance means q < .05**, with the Benjamini–Hochberg families the
   docs name. A per-level effect whose q is above .05 is reported as a
   direction, not a finding.
5. **Current claims only.** Do not correct, hedge against or mention an
   earlier draft's numbers. Earlier drafts are in git tag `pre-cleanup`; use
   them for wording, never for numbers.
