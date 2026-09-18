# Building the paper

`talk_like_a_graph.tex` is the paper (single ACL source, 8-page body plus
appendix). Everything else in this directory is either a LaTeX input generated
by a script, or the script that generates it — nothing here is hand-maintained
except the `.tex` prose itself. See `docs/paper-revision-handoff.md` for the
history of how it got to this shape.

## One command

```bash
bash paper/make_all.sh
```

Runs every analysis script that feeds the paper, in dependency order, then
`paper/make_main_table.py` / `make_ci_table.py` / `make_tables.py` /
`make_figure*.py` to regenerate the `\input`-ed tables (`main_table.tex`,
`ci_table.tex`, `appendix_tables.tex`) and figures (`headline.pdf`,
`continuum.pdf`, `crossfit.pdf`, `rewiring.pdf`, `density.pdf`), then
`latexmk -pdf` to build `talk_like_a_graph.pdf`. Read the script itself for
which analysis feeds which section — each step is labelled with the section
number it covers. Takes roughly 20 minutes, dominated by `ci_all.py`.

## Just the PDF

If the tables and figures are already up to date and only the prose changed:

```bash
cd paper && latexmk -pdf -interaction=nonstopmode talk_like_a_graph.tex
```

## Checking a specific number

`NUMBERS.md` maps every hand-typed number in the prose (not the numbers inside
`\input`-ed tables, which are self-evidently sourced from the script that
wrote them) to the exact command that produces it. Use it to audit one claim
without rerunning the full `make_all.sh`.

## Not part of the build

`density.pdf` and `appendix_tables.tex` are built by `make_all.sh` but not
`\includegraphics`/`\input` anywhere in the current `.tex` — kept because a
future extension of the paper may want them (see `docs/paper-revision-handoff.md`'s
"What is NOT in the final paper" note for why they were cut and what they'd
support).
