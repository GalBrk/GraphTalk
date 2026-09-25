# Paper

There is no current paper source here yet. This directory holds what the next
one needs, and this file says how to write it from the repo.

- `acl2023.sty`, `acl.sty`, `acl_natbib.bst`: the ACL style files.
- `custom.bib`: the bibliography.

Build with `latexmk -pdf` from this directory. An ACL long paper has 8 pages of
body; references, limitations and appendices do not count.

## Where each part of the paper comes from

| Paper part | Source |
|---|---|
| Question and motivation | [results/README.md](../docs/results/README.md), "The study" |
| What the proposal planned and what was run (models, graphs, primers, tasks, metrics) | [results/README.md](../docs/results/README.md), "From the proposal to what was run"; primer definitions in `graphtalk/primers.py` |
| The pilot, and why the main experiment moved to 40-node graphs | [results/README.md](../docs/results/README.md), "Earlier stages", the pilot row. Say it in words; the paper cites no pilot numbers |
| Setup and measurement | [n40-sweep.md](../docs/results/n40-sweep.md) §1 |
| Which primers state the answer (the graph-blind solver) | n40-sweep.md §2; design in [plans/shortcut-ceilings.md](../docs/plans/shortcut-ceilings.md) |
| Main results | n40-sweep.md §3–9 |
| The `node_degree` follow-ups: the `clustering` effect, its replication, thinking against plain, the `filler` control, fixed mean degree, the forensics | [density-followups.md](../docs/results/density-followups.md) §2–8 |
| Limitations | n40-sweep.md §10, density-followups.md §9 |

The ladder, retrieval, rewiring, GoT-naming, probe and size-sweep runs are not
part of the paper (see "Earlier stages" in the results index).

## Rules for numbers

1. **Every number comes from a tagged citation** in `docs/results/*.md`, copied
   as printed. Put the tag in a LaTeX comment on the same line, e.g.
   `+3.8~points % [ddplain]`, so each number can be traced to its block in the
   script output. If the paper needs a number no doc cites, add it to the
   script and the doc first, with a tag. `tests/test_results_docs.py` then
   checks it.
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
   earlier draft's numbers.

## Earlier drafts

v1, v2, v3, the short draft, the independent reanalysis, the synthesis draft
and two drafts built outside the repo are in
[`superseded/paper/`](../superseded/paper/). None of them is current. Use them
for wording and structure, never for numbers.
[`superseded/paper/CLAIMS_LEDGER.md`](../superseded/paper/CLAIMS_LEDGER.md)
records where and why they disagreed.
