#!/usr/bin/env bash
# Build talk_like_a_graph.v3.tex -- the combined paper.
#
# Run from the repo root:
#
#   bash paper/make_v3.sh
#
# Deliberately does NOT call make_all.sh; nothing v3 uses needs its
# ~20-minute ci_all.py leg. Everything v3 generates comes from
# csv2/raw-trends/, which scripts/build_raw_frame.py and scripts/raw_trends.py
# produce from runs/ directly. To rebuild those first:
#
#   PYTHONPATH=. python scripts/build_raw_frame.py
#   PYTHONPATH=. python scripts/raw_trends.py --question all
set -euo pipefail

PY="${PY:-python}"

echo "== float data (from frame.csv and runs/) =="
PYTHONPATH=. "$PY" scripts/primer_findings.py --figure-data --csv-dir csv2/raw-trends

echo "== v3 tables =="
PYTHONPATH=. "$PY" paper/make_v3_tables.py

echo "== v3 figures =="
PYTHONPATH=. "$PY" paper/make_v3_figures.py

echo "== build the PDF =="
( cd paper && latexmk -pdf -interaction=nonstopmode talk_like_a_graph.v3.tex )

echo
echo "== page budget =="
# Target: a 5-page body, with Limitations and Ethics also inside page 5, so
# only references and the appendix follow. The body ends where Limitations
# begins; the label at the end of the Ethics statement is the last line
# before the references.
page_of() {
  grep -o "newlabel{$1}{{[0-9A-Z.]*}{[0-9]*}" paper/talk_like_a_graph.v3.aux \
    | sed 's/.*}{//; s/}//'
}
body_end=$(page_of sec:limitations)
front_end=$(page_of sec:ethics-end)
if [ -z "$body_end" ] || [ -z "$front_end" ]; then
  echo "  FAIL: a page-budget label is missing; page budget unchecked."
  exit 1
fi
echo "  body ends on page $body_end; Ethics ends on page $front_end"
if [ "$body_end" -gt 5 ] || [ "$front_end" -gt 5 ]; then
  echo "  FAIL: runs past the 5-page target."
  exit 1
fi
echo "  OK: within the 5-page target."

echo "done."
