# Claim-to-data index

Each cited CSV is derived from raw responses at source commit `4b45c2435abe5913a4d28fa19e49b42d9bf62ca0` by `analyze.py`. `manifest.json` records the exact filenames and SHA-256s. Each `responses.csv` row retains its raw file and line number; golds were independently reconstructed and checked. No earlier paper draft or analysis table is an input.

| Claim in paper | Source and selection |
| --- | --- |
| 84,002 raw, 84,000 unique, 700 graphs, two identical duplicates, 840 cells of 100 | `results/manifest.json`, `results/duplicates.json`, grouped `results/responses.csv` |
| Main 40-node task success for four arms and seven conditions | `results/pooled.csv`, four nonconstant tasks |
| Table 1 paired changes and adjusted significance | `results/paired_effects.csv`, `control=none`, columns `baseline`, `treatment`, `delta`, `qvalue`; matched outcome counts `fixed`, `broken`, confidence limits `lo`, `hi` |
| Plain4B edge count no-primer→degree 2.00→29.25, +27.25 [22.75,31.75], fixed114/broken5 | `paired_effects.csv`, arm `qwen3-4b`, task `edge_count`, control `none`, condition `degree` |
| Thinking4B component/clustering edge count 44.75, filler32.25; all p>=.20 components=1 | `pooled.csv`, `paired_effects.csv` `control=filler`, `by_density.csv`, `topology.csv` |
| Completion versus conditional edge-count correctness | `pooled.csv`, task `edge_count`, condition `none`, completion `1-cap`, conditional `success/(1-cap)` |
| Cross-query agreement and joint correctness Table 3 | `results/consistency.csv` (`consistent`, `joint_success`, `consistent_wrong`), and `joint_rows.csv` |
| Plain4B degree reduces joint success by9.25 [-13.75,-4.75], components raise by9.25 [6.00,12.75] | `joint_effects.csv`, arm `qwen3-4b`, metric `joint_success` |
| Dense plain4B degree success p=.5 versus .85 | `by_density.csv`, task `node_degree`, arm `qwen3-4b`, conditions `none`/`degree`; dense extension has a different plain-arm cap |
| Plain1.7B edge queries: raw, balanced, prevalence and predictions | `edge_balance.csv`, arm `qwen3-1.7b`, condition `none`, p=.50/.85 |
| RWSE distinct pairs and modal share | `topology.csv` mean of `rwse_unique`, `rwse_unique_6dp`, `rwse_modal_share` for 100 rows per p; higher precision never went into a model |
| Targeted parser sensitivity ≤1.25 percentage points | `reviewer_singleton24.jsonl`, `reviewer_sample28.jsonl`, `reviewer_consistency_sensitivity.csv`, `reviewer_audit.py`; audit is not exhaustive |

To regenerate the statistics, run `python superseded/paper/compare/independent/analyze.py` from a source checkout containing the raw JSONL files. The README lists dependencies and build instructions.
