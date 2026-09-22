#!/usr/bin/env bash
# Build talk_like_a_graph.v3.tex -- the combined paper.
#
# Run from the repo root:
#
#   bash paper/make_v3.sh
#
# Deliberately does NOT call make_all.sh. v3 inherits main_table.tex and
# ci_table.tex unchanged from that pipeline, and nothing added here needs the
# ~20-minute ci_all.py leg. Rerun make_all.sh only when the underlying runs/
# data changes.
#
# Everything v3 adds is generated from csv2/raw-trends/*.csv, which
# scripts/build_raw_frame.py and scripts/raw_trends.py produce from runs/
# directly. To rebuild those first:
#
#   PYTHONPATH=. python scripts/build_raw_frame.py
#   PYTHONPATH=. python scripts/raw_trends.py --question all
set -euo pipefail

PY="${PY:-python}"

echo "== v3 tables (from csv2/raw-trends/) =="
PYTHONPATH=. "$PY" paper/make_v3_tables.py

echo "== v3 figures (from csv2/raw-trends/) =="
# --only keeps paper/ to the four figures v3 embeds; the rest stay PNGs under
# analysis/raw-trends/ for the findings doc.
PYTHONPATH=. "$PY" scripts/raw_trends_figures.py \
    --format pdf --outdir paper --prefix v3_ \
    --only balanced_accuracy,serial_position,capitulation,effect_heatmap

echo "== build the PDF =="
( cd paper && latexmk -pdf -interaction=nonstopmode talk_like_a_graph.v3.tex )

echo
echo "== page budget =="
# Target: a 5-page body. The body ENDS where Limitations begins, so
# that is the label to check -- grepping where the Conclusion *starts* reports
# "page 8" and passes while the Conclusion itself spills onto page 9.
body_end=$(grep -o 'newlabel{sec:limitations}{{[0-9.]*}{[0-9]*}' \
    paper/talk_like_a_graph.v3.aux | sed 's/.*}{//')
if [ -z "$body_end" ]; then
    echo "  FAIL: sec:limitations label not found; page budget unchecked."
    exit 1
fi
echo "  body ends on page $body_end (the Limitations heading)"
if [ "$body_end" -gt 5 ]; then
    echo "  FAIL: body runs past the 5-page target."
    exit 1
fi
echo "  OK: within the 5-page body target."

echo "done."
