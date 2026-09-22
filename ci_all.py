"""Permutation p + bootstrap CI for every (arm, task, condition) cell.

Also emits the drop-vs-zero sensitivity analysis and MAE on integer tasks.

`--corpus densfull40hi` scores the high-density extension (p=0.65-0.85,
node_degree + edge_existence only) as its own file rather than pooling it
into the main p=0.10-0.50 sweep: the two corpora represent different density
regimes and the paper reports them as separate rows, not one averaged cell.
"""
import argparse
import collections
import glob
import json

from graphtalk import scoring, significance

ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"]
CONDS = ["components", "clustering", "rwse", "degree", "filler", "all"]
TASKS = ["connected_nodes", "cycle_check", "edge_count", "edge_existence",
         "node_count", "node_degree"]
INTEGER = {"node_count", "edge_count", "node_degree"}


def load(arm, corpus):
    seen, rows = set(), []
    for path in sorted(glob.glob(f"runs/{arm}.{corpus}.shard*.jsonl")):
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                k = (r["instance_id"], r["condition"], r["style"])
                if k in seen:
                    continue
                seen.add(k)
                rows.append(r)
    return rows


ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="densfull40")
ap.add_argument("--out", default=None)
args = ap.parse_args()
out_path = args.out or ("ci_all.json" if args.corpus == "densfull40"
                         else f"ci_all_{args.corpus.replace('densfull40', '')}.json")

out = {}
for arm in ARMS:
    rows = load(arm, args.corpus)
    scored = collections.defaultdict(dict)   # (task, iid) -> cond -> (capped, exact, abserr)
    for r in rows:
        t = r["task"]
        capped = bool(r.get("hit_cap"))
        if capped:
            scored[(t, r["instance_id"])][r["condition"]] = (True, 0.0, None)
            continue
        res = scoring.score_one(
            scoring.extract_answer(r["response"], t), r["gold"], t)
        scored[(t, r["instance_id"])][r["condition"]] = (
            False, res["exact"], res["absolute_error"])

    for t in TASKS:
        # MAE on integer tasks, none vs each condition, non-capped rows only
        mae = {}
        for c in ["none"] + CONDS:
            errs = [v[c][2] for k, v in scored.items()
                    if k[0] == t and c in v and not v[c][0] and v[c][2] is not None]
            mae[c] = (sum(errs) / len(errs), len(errs)) if errs else (None, 0)

        for c in CONDS:
            drop_ctrl, drop_treat, zero_ctrl, zero_treat = [], [], [], []
            for k, v in scored.items():
                if k[0] != t or "none" not in v or c not in v:
                    continue
                ncap, ne, _ = v["none"]
                ccap, ce, _ = v[c]
                zero_ctrl.append(0.0 if ncap else ne)
                zero_treat.append(0.0 if ccap else ce)
                if ncap or ccap:
                    continue
                drop_ctrl.append(ne)
                drop_treat.append(ce)
            if not drop_ctrl:
                continue
            perm = significance.paired_permutation_test(
                drop_ctrl, drop_treat, n_perm=10000, seed=0)
            # Exact conditional interval, not a percentile bootstrap. These
            # are paired binary outcomes, so the difference takes only three
            # values and a bootstrap's endpoints land on a coarse lattice:
            # simulated at this `n` with no true effect, its nominal 95%
            # interval wrongly excluded zero 11.70% of the time at 4
            # discordant pairs and 7.07% at 11, non-monotonically, so no
            # minimum-discordant threshold repairs it. It also over-claimed
            # on five real cells here. `exact_paired_ci` conditions on the
            # discordant count instead and is at or below nominal at every
            # count; it reuses `mc`'s `b`/`c`, so the interval and the
            # McNemar p-value below now come from the same two numbers.
            mc = scoring.mcnemar(drop_ctrl, drop_treat)
            ci = significance.exact_paired_ci(
                mc["b"], mc["c"], len(drop_ctrl))
            scale = lambda v: None if v is None else v * 100
            mcz = scoring.mcnemar(zero_ctrl, zero_treat)
            out[f"{arm}|{t}|{c}"] = {
                "n_drop": len(drop_ctrl),
                "delta_drop": (sum(drop_treat) - sum(drop_ctrl)) / len(drop_ctrl) * 100,
                "ci": [scale(ci["ci_low"]), scale(ci["ci_high"])],
                "n_discordant": ci["n_discordant"],
                "p_perm": perm["p_value"],
                "p_mcnemar": mc["p_value"],
                "n_zero": len(zero_ctrl),
                "delta_zero": (sum(zero_treat) - sum(zero_ctrl)) / len(zero_ctrl) * 100,
                "p_mcnemar_zero": mcz["p_value"],
                "mae_none": mae["none"][0],
                "mae_cond": mae[c][0],
            }
        print(f"done {arm} {t}", flush=True)

with open(out_path, "w") as fh:
    json.dump(out, fh, indent=1)
print(f"wrote {len(out)} cells to {out_path}")
