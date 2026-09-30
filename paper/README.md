# Paper

**Talk Like a Structured Graph? Structural Statistics Help Small LLMs Mainly by
Stating the Answer**: [`paper.pdf`](paper.pdf).
[← back to the repo](../README.md)

`tests/test_paper_numbers.py` checks every tagged number in `paper.tex` against
`outputs/`. `python paper/build_tables.py --check` (from the repo root) also
checks Table 1 against the saved responses in `frame.csv`, and that Tables 5 and
10 match their sources.

## Build

ACL 2023 style (`acl2023.sty`, `acl_natbib.bst`, `custom.bib`). From this folder:

```sh
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Regenerate the figures and tables

Run these from the repo root once the pipeline in
[`scripts/README.md`](../scripts/README.md) has written `outputs/`:

```sh
PYTHONPATH=. python scripts/readme_figures.py && cp docs/img/primer_usage.pdf paper/fig_primer_usage.pdf && cp docs/img/primer_effect.pdf paper/fig_primer_effect.pdf
python paper/fig_effects.py
python paper/fig_headroom.py
python paper/plot_outcomes.py
python paper/build_tables.py
```

| File | Written by | Reads |
|---|---|---|
| `fig_primer_usage.pdf` (Figure 1) | `scripts/readme_figures.py`, as `docs/img/primer_usage.pdf` | a five-node example graph |
| `fig_effects.pdf` (Figure 2) | `fig_effects.py` | `[main]` in `primer_findings.txt` |
| `fig_primer_effect.pdf` (Figure 3) | `scripts/readme_figures.py`, as `docs/img/primer_effect.pdf` | `[main]` in `primer_findings.txt` |
| `fig_headroom.pdf`, `fig_addstats.pdf` (Figures 4 and 5) | `fig_headroom.py` | `primer_cells.csv` |
| `edge_count_outcomes.pdf` (Figure 6) | `plot_outcomes.py` | `[edgecount]` in `primer_findings.txt` |
| rows of `table_cycles.tex` (Table 10) | `build_tables.py` | `[ccanswer]` and `[cctest]` in `check_cycle_claims.txt` |
| rows of `main_truncation.tex` (Table 5) | `build_tables.py` | `frame.csv` (main sweep, truncated responses) |

The other tables, and the captions of Tables 5 and 10, are written by hand.

The paper's numbers are the tagged blocks of `outputs/`.

## Where each part of the paper comes from

| Paper part | Source |
|---|---|
| Question and motivation | [docs/README.md](../docs/README.md), "The study" |
| What the proposal planned and what was run (models, graphs, primers, tasks, metrics) | [docs/README.md](../docs/README.md), "From the proposal to what was run"; primer definitions in `graphtalk/primers.py` |
| The pilot, and why the main experiment moved to 40-node graphs | [`preliminary/README.md`](../preliminary/README.md). Say it in words; the paper cites no pilot numbers |
| Setup and measurement | [n40-sweep.md](../docs/results/n40-sweep.md) §1 |
| Which primers state the answer (the graph-blind solver) | n40-sweep.md §2; design in [design/shortcut-ceilings.md](../docs/design/shortcut-ceilings.md) |
| Main results | n40-sweep.md §3–9 |
| The `node_degree` follow-ups: the `clustering` effect, its replication, thinking against plain, the `filler` control, fixed mean degree, the forensics | [density-followups.md](../docs/results/density-followups.md) §2–8 |
| Limitations | n40-sweep.md §10, density-followups.md §9 |
| Future work | n40-sweep.md, density-followups.md, [primer-directions.md](../docs/primer-directions.md), [primer-robustness.md](../docs/primer-robustness.md), [investigate_connections_and_cycles.md](../docs/investigate_connections_and_cycles.md) |

## Rules for numbers

1. **Every number comes from a tagged citation**, copied as printed. The
   citation lives in one of two places:
   - a doc in `docs/results/`;
   - one of the analysis docs: `primer-directions.md`, `primer-robustness.md`,
     `investigate_connections_and_cycles.md` or `density-interaction.md`.

   Put the tag in a LaTeX comment on the same line, e.g. `+3.8~points % [ddplain]`,
   so each number can be traced to its block in the script output. If the paper
   needs a number nothing prints yet, add it to the script and the doc first,
   with a tag. `tests/test_results_docs.py` checks the numbers in `docs/results/`
   and `primer-directions.md` against their outputs, and
   `tests/test_paper_numbers.py` the paper's. Never cite `preliminary/`.
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
