#!/usr/bin/env bash
# Regenerate every number and figure the paper cites, in dependency order.
# Run from the repo root:
#
#   PYTHONPATH=. .venv/bin/python -m pytest -q   # sanity check first
#   bash superseded/paper/make_all.sh
#
# Slow steps (shortcut refits, superseded/ci_all.py) are marked; the full run is
# roughly the same order of magnitude as superseded/ci_all.py alone (~20 min).
set -euo pipefail

PY="${PY:-.venv/bin/python}"

echo "== shortcut bars =="
PYTHONPATH=. "$PY" scripts/shortcut_table.py --graphs 500 --json shortcuts.json
PYTHONPATH=. "$PY" scripts/shortcut_table_n40.py \
    --json shortcuts_n40.json --flat-json shortcuts_n40_flat.json

echo "== main scored table (superseded/ci_all.json, ~20 min) =="
PYTHONPATH=. "$PY" superseded/ci_all.py
PYTHONPATH=. "$PY" superseded/ci_all.py --corpus densfull40hi --out superseded/ci_all_hi.json

echo "== headline-cell robustness (Section 5.5, Table 7) =="
PYTHONPATH=. "$PY" scripts/analyze_headline_robustness.py

echo "== route split / crossfit / held-out / ceiling (Section 5.4, 5.7) =="
PYTHONPATH=. "$PY" superseded/scripts/analyze_baseline_law.py --shortcuts shortcuts_n40_flat.json \
    --test split --test crossfit
PYTHONPATH=. "$PY" superseded/scripts/analyze_baseline_law.py --shortcuts shortcuts.json \
    --test heldout --test ceiling --test continuum --test instrument

echo "== GoT naming null (Section 5.4, 5.7) =="
PYTHONPATH=. "$PY" scripts/naming_effect.py

echo "== route-substitution across naming schemes (Section 5.4) =="
PYTHONPATH=. "$PY" scripts/candidates/h_got.py

echo "== error taxonomy, churn/length, controls (Section 5.3, 5.5, 5.6) =="
PYTHONPATH=. "$PY" superseded/scripts/analyze_error_taxonomy.py
PYTHONPATH=. "$PY" superseded/scripts/analyze_churn_and_length.py
PYTHONPATH=. "$PY" superseded/scripts/test_vs_controls.py \
    --json superseded/vs_controls_densfull40.json
PYTHONPATH=. "$PY" superseded/scripts/test_vs_controls.py --corpus densfull40hi \
    --json superseded/vs_controls_densfull40hi.json
PYTHONPATH=. "$PY" superseded/scripts/analyze_nontermination.py
PYTHONPATH=. "$PY" scripts/analyze_rewiring_sweep.py \
    --responses runs/qwen3-1.7b.rewire_shared.jsonl \
    --json rewiring_qwen3-1.7b.json
PYTHONPATH=. "$PY" scripts/analyze_rewiring_sweep.py \
    --responses runs/qwen3-1.7b-think.rewire_shared.jsonl \
    --json rewiring_qwen3-1.7b-think.json
PYTHONPATH=. "$PY" scripts/analyze_rewiring_sweep.py \
    --responses runs/qwen35-2b.rewire_shared.jsonl runs/qwen35-2b.rewire_extra.jsonl \
    --json rewiring_qwen35-2b.json

echo "== review checks: interaction, logit, headline split, SDT, budget, prior, tokens =="
# --tokenizer needs a local Qwen3 tokenizer.json (Qwen/Qwen3-1.7B on the HF hub);
# without it the `tokens` check is skipped and the other checks still run.
PYTHONPATH=. "$PY" superseded/scripts/analyze_review_checks.py ${QWEN3_TOKENIZER:+--tokenizer "$QWEN3_TOKENIZER"}

echo "== RQ3 leads: selection, heterogeneity, error shape, transcription (docs/rq3-leads.md) =="
# The `hetero` test's regression needs statsmodels.
PYTHONPATH=. "$PY" scripts/analyze_rq3_leads.py

echo "== tables and figures =="
PYTHONPATH=. "$PY" superseded/paper/make_main_table.py
PYTHONPATH=. "$PY" superseded/paper/make_ci_table.py
PYTHONPATH=. "$PY" superseded/paper/make_tables.py
PYTHONPATH=. "$PY" superseded/paper/make_figure.py
PYTHONPATH=. "$PY" superseded/paper/make_figure_f1.py
PYTHONPATH=. "$PY" superseded/paper/make_figure_f3.py
PYTHONPATH=. "$PY" superseded/paper/make_figure_density.py

echo "== build the PDF =="
( cd superseded/paper && latexmk -pdf -interaction=nonstopmode talk_like_a_graph.tex )

echo "done."
