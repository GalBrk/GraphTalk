# Scripts

The pipeline, from graphs to the numbers in the docs and the paper.
[← back to the repo](../README.md)

Run everything from the repo root with `PYTHONPATH=.` and the venv active
(`source .venv/bin/activate`, or prefix each command with `uv run --no-sync`).
Only `run_sweep.py`
needs a GPU; every other stage runs on a laptop from the committed files in
`data/`.

```mermaid
flowchart TD
    B["build_size_sweep.py"] --> P[("data/prompts/")]
    P --> R["run_sweep.py (GPU)"] --> RUNS[("data/runs/")]
    SH["shortcut_table_n40.py"] --> BARS[("data/shortcuts_n40_flat.json")]
    RUNS --> F["build_raw_frame.py"] --> FR[("outputs/n40-sweep/frame.csv")]
    FR --> PF["primer_findings.py"]
    BARS --> PF
    FR --> A["response_patterns · check_cycle_claims · primer_directions<br/>validate_directions · primer_robustness · density_interaction"]
    RUNS --> DF["density_followups.py"]
    PF --> O[("outputs/")]
    A --> O
    DF --> O
```

## The pipeline

| Stage | Script | Reads | Writes |
|---|---|---|---|
| 1. Build prompts | `build_size_sweep.py` | — | `data/prompts/*.jsonl` |
| 2. Generate (GPU) | `run_sweep.py`, via [`cluster/sweep.sbatch`](../cluster/sweep.sbatch) | a prompt file | `data/runs/<model>.<run>.shard<i>of<n>.jsonl` |
| Solver bars | `shortcut_table_n40.py` | — (generates its own graphs) | `data/shortcuts_n40.json`, `data/shortcuts_n40_flat.json` |
| 3. Frame | `build_raw_frame.py` | `data/runs/`, regenerated graphs | `outputs/n40-sweep/frame.csv` |
| 4. Analyses | see [Outputs](#outputs) | the frame, the runs, the prompts | `outputs/` |

These commands rebuild the committed prompt files and solver bars in `data/`.
Each reproduces its file byte for byte, so `git status` stays clean (or write
to another `--out` and `cmp` it against the committed file). Keep the order of
`--tasks` and `--conditions`: rows are written in that order, and `run_sweep.py`
assigns rows to shards by position.

```bash
PYTHONPATH=. python scripts/build_size_sweep.py --sizes 40 --densities 0.10 0.20 0.35 0.50 --count 100 \
    --tasks node_count edge_count node_degree connected_nodes edge_existence cycle_check \
    --conditions none components degree clustering rwse filler all --out data/prompts/prompts.densfull40.jsonl
PYTHONPATH=. python scripts/build_size_sweep.py --sizes 40 --densities 0.65 0.75 0.85 --count 100 \
    --tasks node_degree edge_existence \
    --conditions none components degree clustering rwse filler all --out data/prompts/prompts.densfull40hi.jsonl
PYTHONPATH=. python scripts/build_size_sweep.py --sizes 40 --densities 0.05 0.1 0.2 0.35 0.5 0.75 --count 100 \
    --tasks node_degree connected_nodes --conditions none degree --out data/prompts/prompts.density40.jsonl
PYTHONPATH=. python scripts/shortcut_table_n40.py --json data/shortcuts_n40.json \
    --flat-json data/shortcuts_n40_flat.json
```

`density_followups.py` rebuilds the follow-ups' prompts itself and checks each
gold answer against the runs, so only `density40`'s prompt file is committed.

## Outputs

| Script | Output (in `outputs/n40-sweep/` unless noted) | Cited by |
|---|---|---|
| `build_raw_frame.py` | `frame.csv`: one row per response, 84,000 rows | every analysis below |
| `primer_findings.py` | `primer_findings.txt`, `primer_cells.csv`, `edge_existence_collapse.csv`, `node_degree_routes.csv` | [n40-sweep.md](../docs/results/n40-sweep.md), the paper |
| `blind_bars.py` | `blind_bars.txt`: a learned graph-blind bar for `clustering` and `rwse` | [n40-sweep.md](../docs/results/n40-sweep.md) §2, [primer-directions.md](../docs/primer-directions.md) B7 |
| `density_followups.py` | `outputs/density-followups/density_followups.txt` | [density-followups.md](../docs/results/density-followups.md), the paper |
| `check_cycle_claims.py` | `check_cycle_claims.txt`, `cycle_claims.csv`, `response_claims.csv` | [investigate_connections_and_cycles.md](../docs/investigate_connections_and_cycles.md), the paper |
| `primer_robustness.py` | `primer_robustness.txt`, `robustness_responses.csv` | [primer-robustness.md](../docs/primer-robustness.md), the paper |
| `primer_directions.py` | `primer_directions.txt` | [primer-directions.md](../docs/primer-directions.md) |
| `validate_directions.py --score` | `directions_validation.txt` | [primer-directions-validation.md](../docs/primer-directions-validation.md) |
| `response_patterns.py` | `response_patterns.txt`, `response_pattern_cells.csv`, `response_contrasts.csv`, `response_relation.csv` | reused by `check_cycle_claims.py`, `primer_directions.py`, `validate_directions.py` |
| `density_interaction.py` | `density_interaction.txt` | [density-interaction.md](../docs/density-interaction.md) |

Three files in `outputs/n40-sweep/` are *inputs*, not outputs, and no command
below rewrites them:
- `directions_validation_sheet.csv`, labelled by hand, and its blind key
  `directions_validation_key.csv`. `validate_directions.py --make` draws the
  pair and refuses to overwrite a labelled sheet.
- `response_pattern_validation.csv`, a seeded sample of the text patterns'
  matches, for labelling. Its `label` column is empty, so `[rpvalid]` in
  `response_patterns.txt` leaves every text pattern out.
  `response_patterns.py --sample` draws such a sample and refuses to overwrite
  an existing one.

Imported by the scripts above:
- `analyze_primer_window.py`: the cell table's rule;
- `analyze_baseline_law.py`: the solver's route split;
- `score_density_sweep.py`: scoring by density level;
- `analyze_rq3_leads.py`: the `clustering` forensics.

The last two also run on their own; see their docstrings.

## Reproduce everything

About 15 minutes on a laptop. Each command overwrites its
committed output; `git diff outputs/` should then show no change.

```bash
export PYTHONPATH=. PYTHONUTF8=1
python scripts/build_raw_frame.py
python scripts/primer_findings.py --csv-dir outputs/n40-sweep > outputs/n40-sweep/primer_findings.txt
python scripts/blind_bars.py > outputs/n40-sweep/blind_bars.txt
python scripts/response_patterns.py --csv-dir outputs/n40-sweep > outputs/n40-sweep/response_patterns.txt
python scripts/check_cycle_claims.py --csv-dir outputs/n40-sweep > outputs/n40-sweep/check_cycle_claims.txt
python scripts/primer_directions.py > outputs/n40-sweep/primer_directions.txt
python scripts/validate_directions.py --score > outputs/n40-sweep/directions_validation.txt
python scripts/primer_robustness.py --csv-dir outputs/n40-sweep > outputs/n40-sweep/primer_robustness.txt
python scripts/density_interaction.py > outputs/n40-sweep/density_interaction.txt
python scripts/density_followups.py > outputs/density-followups/density_followups.txt
```

Then the paper's tables and figures: see [paper/README.md](../paper/README.md).
A regenerated figure (in `docs/img/` or `paper/`) matches the committed one in
content, but not byte for byte when matplotlib differs from the version that
drew it; each committed PDF records that version (3.10.9, or 3.10.8 for
`edge_count_outcomes.pdf`). Check a regenerated figure by eye, not with
`git diff`.

## Utilities

| Script | What it does |
|---|---|
| `readme_figures.py` | Draws two of the front page's three figures into `docs/img/`: the primer effect (from `[main]`) and how a primer is used (rendered by `graphtalk/primers.py`). The third, `docs/img/headroom.png`, is written by `paper/fig_headroom.py` |
| `show_primers.py` | Prints every primer condition for a few graphs, to read by eye. By default it fetches published GraphQA rows from the Hugging Face datasets-server, which needs network; `--generated N` draws its own graphs and runs offline |
| `draw_graph.py` | Parses a graph out of a published GraphQA row and draws it; fetches the row from the Hugging Face datasets-server, so it needs network |
| `reproduce_rows.py` | Cluster check that regenerated rows match the committed ones; see [cluster/README.md](../cluster/README.md) |
| `primer_flips.py` | Writes every question a primer flips, with both answers side by side, for reading by hand (`outputs/n40-sweep/primer_flips.csv`, not committed) |
