"""(c) Is Table 4's 'difficulty' graph structure, or just answer magnitude?"""
import json, glob, sys, collections
sys.path[:0] = ["superseded/scripts", "scripts"]
import analyze_baseline_law as abl
from graphtalk import scoring

def rows(patterns, task="node_degree"):
    seen, out = set(), []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            if ".got." in path:
                continue
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["task"] != task:
                    continue
                k = (r["instance_id"], r["condition"])
                if k in seen:
                    continue
                seen.add(k)
                if r.get("hit_cap"):
                    continue
                r["score"] = scoring.score_one(
                    scoring.extract_answer(r["response"], task), r["gold"], task)["primary"]
                out.append(r)
    return out

print("=== (c1) degfixdeg: mean degree pinned, n swept 20->160 (qwen3-1.7b, node_degree) ===")
rs = rows(["runs/qwen3-1.7b.degfixdeg*.jsonl"])
acc = collections.defaultdict(list)
for r in rs:
    _, size, dens, _ = r["instance_id"].split("/")
    n = int(size[4:]); p = float(dens[1:])
    acc[(round(n * p), n, r["condition"])].append((r["score"], int(r["gold"])))
print(f"{'mean deg':>9}{'n':>6}{'cond':>12}{'acc':>8}{'mean gold':>11}{'k':>6}")
for (md, n, cond), v in sorted(acc.items()):
    print(f"{md:>9}{n:>6}{cond:>12}{sum(s for s,_ in v)/len(v):>8.3f}"
          f"{sum(g for _,g in v)/len(v):>11.2f}{len(v):>6}")

print("\n=== (c2) continuum corpus: accuracy under `none` by gold degree, within density ===")
rs = rows(["runs/qwen3-1.7b-think.degdensthink.shard*of7.jsonl",
           "runs/qwen3-1.7b-think.degdensfillT.shard*of7.jsonl"])
none = [r for r in rs if r["condition"] == "none"]
by = collections.defaultdict(list)
for r in none:
    by[(abl.density_of(r["instance_id"]), int(r["gold"]) // 5 * 5)].append(r["score"])
dens = sorted({d for d, _ in by})
buckets = sorted({g for _, g in by})
print("gold deg  " + "".join(f"{d:>9}" for d in dens))
for g in buckets:
    line = f"{g:>3}-{g+4:<6}"
    for d in dens:
        v = by.get((d, g))
        line += (f"{sum(v)/len(v):>9.2f}" if v and len(v) >= 15 else f"{'.':>9}")
    print(line)
print("k per cell:")
for g in buckets:
    print(f"{g:>3}-{g+4:<6}" + "".join(f"{len(by.get((d,g),[])):>9}" for d in dens))

xs = [int(r["gold"]) for r in none]; ys = [r["score"] for r in none]
r_, p_ = abl.pearson(xs, ys)
print(f"\n  pooled r(gold degree, correct) = {r_:+.3f}  p={p_:.2g}  n={len(none)}")
ds = [abl.density_of(r["instance_id"]) for r in none]
r2, p2 = abl.pearson(ds, ys)
print(f"  pooled r(density,     correct) = {r2:+.3f}  p={p2:.2g}")
rows_ols = [([int(r["gold"]), abl.density_of(r["instance_id"])], r["score"]) for r in none]
for name, coef, se, t in abl.ols(rows_ols, ["gold degree", "density"]):
    print(f"    {name:<13} {coef:+9.5f}  se {se:.5f}  t={t:+7.2f}")
