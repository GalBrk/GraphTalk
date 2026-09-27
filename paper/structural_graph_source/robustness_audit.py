"""Recompute supplementary 40-node contrasts from saved GraphTalk responses.

Usage: python robustness_audit.py --repo /path/to/GraphTalk
This performs no new model inference. The BH families are defined below before
the tests: three density interactions, and six high-density all-vs-controls.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


LOW = (.10, .20, .35, .50)
HIGH = (.65, .75, .85)


def bh(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    q = np.empty(len(pvalues))
    order = np.argsort(pvalues)
    corrected = np.minimum.accumulate(
        (pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1))[::-1]
    )[::-1]
    q[order] = np.minimum(corrected, 1)
    return q


def bootstrap_by_level(level_diffs, draws=10000, seed=20260925):
    """Bootstrap graphs within density, retaining the observed level weights."""
    rng = np.random.default_rng(seed)
    sample = np.empty(draws)
    total = sum(len(a) for a in level_diffs.values())
    for draw in range(draws):
        sample[draw] = sum(a[rng.integers(len(a), size=len(a))].sum()
                           for a in level_diffs.values()) / total
    return 100 * np.percentile(sample, (2.5, 97.5))


def dedicated_pairs(sds, patterns, condition):
    data = sds.summarize(sds.load(patterns), "density")
    pairs = sds.pairs_for(data["paired"], condition, None)
    assert len(pairs) == 7 * 400, (patterns, condition, len(pairs))
    by_level = {}
    for density in LOW + HIGH:
        level = [r for r in pairs if r[0] == density]
        assert len(level) == 400
        assert max(sum(r[1][1] for r in level), sum(r[2][1] for r in level)) < 60
        a = np.array([r[1][0] for r in level], dtype=int)
        b = np.array([r[2][0] for r in level], dtype=int)
        by_level[density] = b - a
    return by_level


def interaction(by_level, seed):
    lo = np.concatenate([by_level[d] for d in LOW])
    hi = np.concatenate([by_level[d] for d in HIGH])
    assert len(lo) == 1600 and len(hi) == 1200
    delta = 100 * (hi.mean() - lo.mean())
    p = stats.ttest_ind(hi, lo, equal_var=False).pvalue
    low_ci = bootstrap_by_level({d: by_level[d] for d in LOW}, seed=seed)
    high_ci = bootstrap_by_level({d: by_level[d] for d in HIGH}, seed=seed + 1)
    # CI for a difference requires jointly resampling both density groups.
    rng = np.random.default_rng(seed + 2)
    samples = []
    for _ in range(10000):
        mean_lo = sum(by_level[d][rng.integers(400, size=400)].sum() for d in LOW) / 1600
        mean_hi = sum(by_level[d][rng.integers(400, size=400)].sum() for d in HIGH) / 1200
        samples.append(100 * (mean_hi - mean_lo))
    ci = np.percentile(samples, (2.5, 97.5))
    return dict(low=100 * lo.mean(), high=100 * hi.mean(), difference=delta,
                ci=ci, p=p, low_ci=low_ci, high_ci=high_ci)


def direct_comparisons(frame, scoring):
    dense = frame[(frame.arm == "qwen3-4b") & (frame.task == "node_degree")
                  & (frame.density_class.isin(HIGH))].copy()
    dense["correct"] = (dense.exact * (1 - dense.hit_cap)).astype(int)
    dense["truncated"] = dense.hit_cap.astype(int)
    assert len(dense) == 7 * 300
    by_condition = {}
    for condition, subset in dense.groupby("condition"):
        subset = subset.set_index("instance_id").sort_index()
        assert len(subset) == 300 and subset.index.is_unique
        by_condition[condition] = subset
    all_rows = by_condition["all"]
    results = []
    for control in ("none", "clustering", "degree", "rwse", "components", "filler"):
        a = by_condition[control]
        assert a.index.equals(all_rows.index) and (a.gold == all_rows.gold).all()
        assert not a.hit_cap.any() and not all_rows.hit_cap.any()
        d = all_rows.correct.to_numpy() - a.correct.to_numpy()
        tested = scoring.mcnemar(a.correct.to_numpy().astype(bool),
                                 all_rows.correct.to_numpy().astype(bool))
        by_level = {p: d[all_rows.density_class.to_numpy() == p] for p in HIGH}
        ci = bootstrap_by_level(by_level, seed=20260926)
        results.append(dict(control=control, baseline=100*a.correct.mean(),
                            all=100*all_rows.correct.mean(), delta=100*d.mean(),
                            ci=ci, p=tested["p_value"],
                            fixed=tested["c"], broke=tested["b"]))
    for result, q in zip(results, bh([r["p"] for r in results])):
        result["q"] = q
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "scripts"))
    import score_density_sweep as sds
    from graphtalk import scoring

    def runs(*names):
        return [str(repo / "runs" / f"{name}.shard*.jsonl") for name in names]

    plain = runs("qwen3-1.7b.degdens40", "qwen3-1.7b.degceil",
                 "qwen3-1.7b.degdens40hi", "qwen3-1.7b.degdensfill")
    thinking = runs("qwen3-1.7b-think.degdensthink",
                    "qwen3-1.7b-think.degdensfillT")
    specifications = [("1.7B plain: clustering", plain, "clustering"),
                      ("1.7B thinking: clustering", thinking, "clustering"),
                      ("1.7B thinking: degree", thinking, "degree")]
    contrasts = [(label, interaction(dedicated_pairs(sds, files, condition),
                                     seed=20260925 + idx * 10))
                 for idx, (label, files, condition) in enumerate(specifications)]
    print("[interaction] 40-node dedicated paired follow-ups; high minus low change in all-prompt correct share (points). Welch p and BH q over three exploratory interactions; bootstrap CI resamples graphs within density. Truncated answers are not correct.")
    for (label, r), q in zip(contrasts, bh([r["p"] for _, r in contrasts])):
        print(f"  {label}: low {r['low']:+.2f}, high {r['high']:+.2f}; "
              f"interaction {r['difference']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}], "
              f"p={r['p']:.5g}, q={q:.5g}; n=1600/1200")

    path = repo / "csv2/raw-trends/frame.csv"
    frame = pd.read_csv(path, usecols=["arm", "task", "condition", "density_class",
                                       "instance_id", "gold", "exact", "hit_cap"])
    print("[bundle-direct] Plain 4B node degree p>=.65 (n=300 paired per contrast): high-density all minus comparator, correct share of every prompt; exact McNemar p and BH q over six alternatives. All cells have zero truncations; the plain dense arm uses 2048 output tokens.")
    for r in direct_comparisons(frame, scoring):
        print(f"  all vs {r['control']}: {r['baseline']:.2f} -> {r['all']:.2f}; "
              f"delta {r['delta']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}], "
              f"p={r['p']:.5g} q={r['q']:.5g} fixed {r['fixed']} broke {r['broke']}")


if __name__ == "__main__":
    main()
