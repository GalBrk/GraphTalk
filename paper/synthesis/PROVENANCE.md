# Provenance of `synthesis.tex`

Every number traces to one experiment: the 40-node `densfull40` + `densfull40hi` sweep.

## Data and freshness

- **Raw data:** 172 files, `runs/{qwen3-1.7b,qwen3-1.7b-think,qwen3-4b,qwen3-4b-think}.{densfull40,densfull40hi}.shard*.jsonl`. They hold 84,002 rows, of which 84,000 are unique (2 identical duplicates).
- **Hash check:** all 172 files match the SHA-256 hashes in `paper/compare/independent/results/manifest.json`. No run file has changed since `d605f73` (2026-09-18, high-density extension).
- **Runs not used:** every other Qwen 1.7B/4B run belongs to a separate experiment, and the paper uses none of them:
  - `degdens40`, `degdensrep`, `degfixdeg`, `degdensthink`, `degdensfillT`;
  - `probe100`, `size`, `ladder_screen`, `retrieval*`, `rewire_shared`, `ec500`, `density40`.
- **Rerun check:** both analysis paths were rerun from the raw files on 2026-09-24:
  - `scripts/build_raw_frame.py` → `scripts/primer_findings.py` (v3; terminated pairs). The rebuilt `frame.csv` is identical. `primer_findings.txt` is identical except for its output-path line. `node_degree_routes.csv`, `edge_existence_collapse.csv` and `primer_cells.csv` are byte-identical.
  - `paper/compare/independent/analyze.py` (S, with capped generations counted as failures). All 9 result CSVs match, with a largest numeric difference of 1e-15. The only text difference is path separators.

Abbreviations: `CR/` = `paper/compare/independent/results/`; `PF [tag]` = `csv2/raw-trends/primer_findings.txt`, printed under that tag.

## Tables and figure

| Item | Source and selection |
|---|---|
| Table 1: character counts | `PF [length]` |
| Table 1: solver accuracy (and the 0.81 / 74.6 caption bounds; max non-carrying scores 0.8095 and 74.571) | `shortcuts_n40_flat.json`, keys `{task}/{condition}` (from `scripts/shortcut_table_n40.py`) |
| 16 exact / 1 heuristic / 8 fitted rules | `graphtalk/shortcuts.py` (`THEOREMS`, `HEURISTICS`, `FITTED`) |
| Table 2 | `CR/paired_effects.csv`, `control=none`: `baseline`, `delta`; `*` = `qvalue<.05` |
| Table 3 | `csv2/raw-trends/node_degree_routes.csv` (`qwen3-4b`, `none`/`degree`); `PF [flip]`, `[route]` |
| Table 4 | `CR/consistency.csv` (`consistent`, `joint_success`) |
| Figure 1a | `CR/by_density.csv` (plain 4B, `node_degree`, `none`/`degree`) |
| Figure 1b | `CR/edge_balance.csv` (plain 1.7B, `none`) |
| Figure 1c | `CR/topology.csv` (`rwse_unique`, `_4dp`, `_6dp`); drawn by `make_figure.py` |

## Claims in the text

| Section | Claim | Source |
|---|---|---|
| §3.1 / abstract | 84,000 generations, 840 cells, golds verified | `CR/manifest.json`, `CR/duplicates.json` |
| §4.1 | Plain 4B: 2.00→29.25, +27.25 [22.75, 31.75], 114 fixed / 5 broken; plain 1.7B +3.00 (q=.101); 1.7B-T −3.25 | `CR/paired_effects.csv` |
| §4.1 | 96–100% of plain-4B responses sum degrees; MAE 136→40 | `PF [edgecount]` |
| §4.1 | 4B-T completion and success: 20.50/97.56/20.00; `degree` 41.25/28.50; 44.75 against 32.25; completion 45.50 against 33.50 | `CR/pooled.csv` (`success`, `cap`) |
| §4.1 | Intervals [6.75, 18.01] and [7.50, 17.75] | `CR/paired_effects.csv`, `control=filler` |
| §4.1 | 64/35/8 against 46/22/1 | `CR/by_density.csv` |
| §4.1 | One component at p≥.20 | `CR/topology.csv`, `PF [comp]` |
| §4.2 | Plain-4B profile −6 … +27; 13 broken / 1 fixed | `PF [flip]` |
| §4.2 | Retrieval 44–74% at 73–98%; enumeration; 68–92% open with an answer; the rest 79/45/28/16% | `PF [route]`, `node_degree_routes.csv` |
| §4.2 | `filler` −2 at p=.50 | `CR/by_density.csv` (99→97) |
| §4.2 | `all` retrieval 15–51% | `node_degree_routes.csv` (`share_retrieve`) |
| §4.2 | 1.7B-T: at most 2% retrieve, 76–98% enumerate, discrepancy 23/49 against 0.3/4, correct 61/69, +19.9 | `PF [verify]` |
| §4.2 | 1.7B-T: +16.0 with capped counted as errors; capping 1.1→7.6% | `PF [trunc]` |
| §4.2 | Plain 1.7B: +10/+18, within 2 pp from p=.50 | `PF [plain17]` |
| §4.2 | Plain 1.7B: at most 4% retrieve, own count 80/41%, baselines 5–30% | `node_degree_routes.csv` |
| §4.3 | J/C effects −9.25, −13.25, +9.25, +8.50, +2.00 | `CR/joint_effects.csv` |
| §4.3 | C−J = 10.50 | `CR/consistency.csv` |
| §4.3 | −6.50 against −3.00 | `CR/paired_effects.csv` |
| §4.4 | Edge share 10→85% | `PF [goldshare]` |
| §4.4 | Hit rates, false-alarm share, 51→33%, `filler` +18.3 | `PF [fa]` |
| §4.4 | 54/86% raw, 94/99% yes, balanced 55.77/53.33 | `CR/edge_balance.csv` |
| §4.4 | `clustering` +11.3; 9.3/11.3/7.7/−21.3 | `PF [clusthi]`, `[bundle]` |
| §4.4 | 69% against 74% enumerate | `PF [clustproc]` |
| §4.4 | Within-graph SD ≤ 0.021 | `PF [cluster]` |
| §4.5 | 30.14/6.74/3.30/2.00; modal 62.68%; degree 12.50; 40.00/39.99 at 6 dp | `CR/topology.csv`, mean per p |
| §4.5 / §5 / Contribution (4) | "From p=.35 fewer node classes than degree (first 1-WL round)"; 11.98 against 6.74 at p=.35, 12.50 against 3.30 at p=.50 | `CR/topology.csv` (`degree_unique` against `rwse_unique`, mean per p; holds in 100/100 graphs at each p ≥ .35, 0/100 at p ≤ .20) |
| Limitations | Nodes 0–9: 96% against 83% | `PF [position]` |
| Limitations | 52-response parser audit, at most 1.25 pp | `CR/reviewer_sample28.jsonl`, `CR/reviewer_singleton24.jsonl`, `CR/reviewer_consistency_sensitivity.csv` |

## Not regenerated from raw data

- **Solver scores:** `shortcuts_n40_flat.json` was not recomputed. It was last changed at `6a0878b` (2026-09-18, the n=40 refit fix), and its values match Table 1.
- **"300 held-out graphs":** this figure is taken from `paper/NUMBERS.md`.
