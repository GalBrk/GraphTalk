"""The paired effect of every primer in every cell of the 40-node sweep, for
`scripts/primer_findings.py` ([bands], [window], [bandsens], [flagged]).

A cell is one (arm, task, density, primer): its paired effect against `none`
and its baseline (the `none` correct share). Cells are split by whether the
primer text alone determines the answer: a graph-blind solver reading only the
primer (`shortcuts_n40_flat.json`) scores near-perfectly on it. Pairs follow
rule R1 (graphtalk/outcomes.py): a truncated response counts as not correct and
is never dropped; cells whose truncated share reaches 15% are flagged.

The command-line report this module used to print is in
superseded/scripts/analyze_primer_window.py; every number it gave is printed by
primer_findings.py now.
"""
import numpy as np
import pandas as pd

from graphtalk import outcomes

ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"]
CONDS = ["components", "clustering", "rwse", "degree", "all"]
# Constant gold at n=40, so a shift there is not about reading a graph.
CONST_GOLD = ("node_count", "cycle_check")
# A primer "carries the answer" when a solver reading only the primer text
# scores near-perfectly. The highest bar below 0.85 is 0.746
# (edge_existence/degree) and the carrying cells score 1.00, so any threshold
# in (0.746, 1.0] gives the same split.
CARRIES = 0.85
BANDS = [(0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 0.90), (0.90, 1.01)]


def cells(frame, bars):
    """One row per (arm, task, density, condition): paired effect and baseline."""
    out = []
    for arm in ARMS:
        for task in sorted(frame.task.unique()):
            if task in CONST_GOLD:
                continue
            for dens in sorted(frame.density_class.unique()):
                d = frame[(frame.arm == arm) & (frame.task == task)
                          & (frame.density_class == dens)]
                if d.empty:
                    continue
                a = d[d.condition == "none"].set_index("graph_id")
                for c in CONDS:
                    b = d[d.condition == c].set_index("graph_id")
                    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
                    if j.empty:
                        continue
                    # R1: every pair is kept; a truncated response is not correct.
                    ca = outcomes.outcome(j.exact_a, j.hit_cap_a) == outcomes.CORRECT
                    cb = outcomes.outcome(j.exact_b, j.hit_cap_b) == outcomes.CORRECT
                    ta, tb = j.hit_cap_a.mean(), j.hit_cap_b.mean()
                    out.append(dict(
                        arm=arm, task=task, density=dens, condition=c,
                        n=len(j),
                        baseline=ca.mean(),
                        delta=100.0 * (cb.mean() - ca.mean()),
                        acc=cb.mean(),
                        trunc_a=ta, trunc_b=tb,
                        flagged=max(ta, tb) >= outcomes.FLAG,
                        bar=bars.get(f"{task}/{c}", float("nan")),
                        bar_none=bars.get(f"{task}/none", float("nan"))))
    t = pd.DataFrame(out)
    t["carries"] = t.bar >= CARRIES
    return t


def window(t):
    """Mean effect per baseline band, answer-carrying cells and side information."""
    rows = []
    for lo, hi in BANDS:
        s = t[(t.baseline >= lo) & (t.baseline < hi)]
        if s.empty:
            continue
        car, non = s[s.carries], s[~s.carries]
        rows.append(dict(band=f"{lo:.2f}-{min(hi, 1.0):.2f}",
                         n_carries=len(car),
                         d_carries=car.delta.mean() if len(car) else np.nan,
                         n_side=len(non),
                         d_side=non.delta.mean() if len(non) else np.nan))
    return pd.DataFrame(rows)
