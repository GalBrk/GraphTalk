# Structural Primers for LLM Graph Reasoning

Independent, auditable analysis and ACL-format paper based on GraphTalk's saved 40-node Qwen3 experiments. All files in this directory were added for this analysis. Existing repository source code was not changed.

## Evidence and scope

The source-data snapshot is repository commit `4b45c2435abe5913a4d28fa19e49b42d9bf62ca0`. Its 172 raw `runs/*.densfull40[hi].shard*.jsonl` files contain 84,002 rows; there are 84,000 unique response keys after removing two identical duplicates. The program records SHA-256 hashes of every raw file, reconstructs all 700 graphs independently, verifies every saved gold against the reconstruction, and requires 100 unique rows per task/condition/model/density cell. The primary paper analyzes six graph queries at p=.10,.20,.35,.50; p=.65,.75,.85 extends only degree and edge existence. The four models are Qwen3-1.7B / 4B, each plain and native thinking. No models were rerun.

The primary outcome is exact match AND not capped. The repository's response extractor is reused for comparability; it has documented errors. `reviewer_audit.py` reproduces the independent review's 28 sampled neighborhood answers, the 24 targeted singleton cases, and the narrow manual correction sensitivity. The audit is not an exhaustive parser validation. The paper separates measured graph-query primitives from untested graph ML predictions.

## Regenerate

From the repository root (the raw runs are tracked under `runs/`). `results/responses.csv`, `joint_rows.csv` and `extraction_differences.jsonl` are git-ignored and are written by the first command:

```bash
python -m pip install 'numpy==2.3.5' 'pandas==2.2.3' 'scipy==1.17.0' 'matplotlib==3.10.8'
python superseded/paper/compare/independent/analyze.py
python superseded/paper/compare/independent/reviewer_audit.py
python superseded/paper/compare/independent/make_assets.py
cd superseded/paper/compare/independent
TEXINPUTS=..: BSTINPUTS=..: latexmk -pdf -interaction=nonstopmode paper.tex
```

`analyze.py` imports `graphtalk/scoring.py`; all graphs, tasks, and golds are reconstructed within the new script. `make_assets.py` generates `assets/*.tex` and `assets/density_diagnostics.pdf` directly from the audited output CSVs. A full checkout requires roughly 293 MB of raw JSONL files and TeX Live with BibTeX, the existing `paper/acl.sty`, and `paper/acl_natbib.bst`. The paper uses the original ACL template geometry, two columns, and references/appendix after a page break.

## Reproducing individual claims

- `results/manifest.json`: exact source commit, per-file SHA-256 and size, counts, duplicate keys, extractor sensitivity.
- `results/responses.csv`: every response's arm, task, condition, density, graph, gold, parsed prediction, cap, exact score, success, source path, and line number. These line numbers index the source JSONL.
- `results/paired_effects.csv`: all paired task comparisons against `none` and `filler`, with stratified bootstrap intervals, fixed/broken discordances, exact McNemar p, and within-family BH q.
- `results/joint_rows.csv`, `consistency.csv`, `joint_effects.csv`: graph/target-matched degree-neighborhood consistency and joint correctness.
- `results/edge_balance.csv`: edge-label prevalence, predicted Yes, per-class success, balanced success, and class counts.
- `results/topology.csv`: edge count, graph components, target properties, and distinct RWSE signatures at 2, 4, 6 decimals per graph.
- `results/observed_caps.csv`: observed capped token counts by arm and sweep.
- `results/reviewer_sample28.jsonl`, `reviewer_singleton24.jsonl`, and `reviewer_consistency_sensitivity.csv`: raw response text and exact lines behind the targeted parser audit.

The bootstrap unit is the paired graph, stratified by density. Six comparisons per model/task/control are BH-adjusted separately; these families are exploratory. `all_paired_intervals.csv` in `assets/` is a convenience copy. We retain no inherited analysis table or conclusions from earlier paper drafts.
