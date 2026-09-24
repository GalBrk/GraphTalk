"""Recomputes, from the archived outputs and code, the values earlier drafts
printed for quantities the claims ledger (superseded/paper/CLAIMS_LEDGER.md)
cut or found wrong, so each of those numbers can be traced to its data. The
same quantities under the current scoring rule are in scripts/legacy_claims.py.

Run from the repo root:

  PYTHONPATH=. python superseded/scripts/reproduce_cut_claims.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join("superseded", "scripts"))
import analyze_baseline_law as abl  # noqa: E402  (the archived full copy)

SNAP = "superseded/csv2/raw-trends/"
SWEEP = "superseded/csv2/sweep-large-graph/"


def analysed():
  parts = {"40-node sweep": 84000, "degdens40, new graphs": 2400,
           "re-runs of main-sweep prompts": 1200, "degdensrep": 3200, "degfixdeg": 6400}
  print(f"v3 '97,200 analysed' (v3.tex:429) is the sum {' + '.join(map(str, parts.values()))} "
        f"= {sum(parts.values())} ({', '.join(parts)}).")


def copy_test():
  e = json.load(open("superseded/error_taxonomy.json", encoding="utf-8"))["node_degree_errors"]
  b = e["qwen3-4b|degree"]
  p = stats.binomtest(b["adjacent_id_copy"], b["n_wrong"], 0.173, alternative="greater").pvalue
  print(f"v1 copy test (v1.tex:447-449): {b['adjacent_id_copy']} of {b['n_wrong']} wrong answers "
        f"equal node k-1's or k+1's stated degree (superseded/error_taxonomy.json); against the "
        f"17.3% background v1 states, one-sided binomial p={p:.2g}. The script that derived "
        "17.3% (scripts/candidates/b_copying.py) was never committed.")


def mde():
  rows = []
  for arm in ("qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"):
    t = pd.read_csv(f"{SWEEP}mde_{arm}.csv")
    null = t[~t.bh_significant.astype(bool)]
    edge = null.mde_note.notna() | null.mde_note_negative.notna()
    rows.append((arm, len(null), int(edge.sum()), null.mde_delta))
    print(f"v1/v2 null cells (v1.tex:496-500), {arm}: {int(edge.sum())} of {len(null)} at floor "
          f"or ceiling ({SWEEP}mde_{arm}.csv)")
  for size in ("1.7b", "4b"):
    vals = pd.concat([r[3] for r in rows if r[0].startswith("qwen3-" + size)]).dropna()
    print(f"v2 median detectable gain (v2.tex:560-563), qwen3-{size} arms pooled: "
          f"{100 * vals.median():.2f} points")
  print("v1's medians (5.6 and 10.0, v1.tex:500-502) came from pre-fix MDE files kept only in "
        "the untracked archive/pre-mde-fold/; they are not in git.")


def thinking_baseline():
  r = pd.read_csv(SNAP + "node_degree_routes.csv")
  s = r[(r.arm == "qwen3-1.7b-think") & (r.condition == "none")].sort_values("density")
  print(f"v2 qwen3-1.7b-think node_degree baseline (v2.tex:396-397): unweighted mean of "
        f"{', '.join(f'{v:.3f}' for v in s.acc)} = {s.acc.mean():.3f} ({SNAP}node_degree_routes.csv)")


def length_r():
  t = pd.read_csv(SWEEP + "primer_decomposition.csv")
  for label, s in (("all rows", t), ("mid-range rows", t[t.mid_range.astype(bool)])):
    out = []
    for arm, g in s.groupby("arm"):
      g = g.dropna(subset=["length_cost_pp", "none_acc"])
      if len(g) > 2:
        out.append(f"{arm} {stats.pearsonr(g.none_acc, g.length_cost_pp)[0]:+.2f}")
    print(f"v2 length cost vs no-primer accuracy (v2.tex:419-422), {label} of "
          f"{SWEEP}primer_decomposition.csv: " + ", ".join(out))
  print("  v2 printed +0.55 (1.7b-think), -0.09 (1.7b), -0.44 (4b) and did not state its row set; "
        "neither of these two row sets reproduces all three.")


def crossfit():
  bars = json.load(open("shortcuts_n40_flat.json", encoding="utf-8"))
  cells = []
  for arm in abl.DENSFULL_ARMS:
    for corpus in ("densfull40", "densfull40hi"):
      cells += abl.arm_cells_crossfit(arm, [f"runs/{arm}.{corpus}.shard*.jsonl"], bars)
  route = [c for c in cells if c["task"] not in abl.DEGENERATE_TASKS and c["route"]]
  for task in ("node_degree", "edge_count", None):
    s = [c for c in route if task is None or c["task"] == task]
    r, p = abl.pearson([c["baseline"] for c in s], [c["delta"] for c in s])
    slope, _ = abl.fit_line([c["baseline"] for c in s], [c["delta"] for c in s])
    print(f"v2/short cross-fit (v2.tex:477-481), {task or 'pooled'}: {len(s)} points, "
          f"r={r:+.3f} (p={p:.2g}), slope {slope:+.1f}")


def route_threshold():
  bars = json.load(open("shortcuts_n40_flat.json", encoding="utf-8"))
  gains = {k: v - bars[k.split("/")[0] + "/none"] for k, v in bars.items()
           if not k.endswith("/none") and k.split("/")[0] not in abl.DEGENERATE_TASKS}
  below = max(g for g in gains.values() if g <= abl.ROUTE_GAIN_THRESHOLD)
  above = min(g for g in gains.values() if g > abl.ROUTE_GAIN_THRESHOLD)
  print(f"short-draft route threshold (short.tex:110-111): largest gain at or below 0.05 is "
        f"{below:.4f}, smallest above {above:.4f} (shortcuts_n40_flat.json); the draft printed "
        "0.011 and 0.066")


if __name__ == "__main__":
  analysed()
  copy_test()
  mde()
  thinking_baseline()
  length_r()
  crossfit()
  route_threshold()
