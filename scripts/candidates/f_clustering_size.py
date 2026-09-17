"""Bonus: does the `clustering` headline replicate across graph SIZE?
degfixdeg pins mean degree and sweeps n = 20..160."""
import json, glob, sys, collections, math
sys.path.insert(0, "scripts")
from graphtalk import scoring

rows = {}
for path in sorted(glob.glob("runs/qwen3-1.7b.degfixdeg*.jsonl")):
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        k = (r["instance_id"], r["condition"])
        if k in rows:
            continue
        rows[k] = None if r.get("hit_cap") else scoring.score_one(
            scoring.extract_answer(r["response"], r["task"]), r["gold"], r["task"])["primary"]

def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(lo + 1)) / 2 ** n)

cells = collections.defaultdict(lambda: [0, 0, 0])
for (iid, cond), s in rows.items():
    if cond != "clustering":
        continue
    base = rows.get((iid, "none"))
    if base is None or s is None:
        continue
    _, size, dens, _ = iid.split("/")
    key = (int(size[4:]), float(dens[1:]))
    cells[key][2] += 1
    if s > base:
        cells[key][0] += 1
    elif s < base:
        cells[key][1] += 1

print("qwen3-1.7b, node_degree, clustering vs none, mean degree pinned")
print(f"{'n':>5}{'p':>8}{'mean deg':>10}{'pairs':>7}{'help':>6}{'hurt':>6}{'delta':>8}{'McNemar':>10}")
B = C = N = 0
for (n, p), (b, c, k) in sorted(cells.items()):
    B += b; C += c; N += k
    print(f"{n:>5}{p:>8}{round(n*p):>10}{k:>7}{b:>6}{c:>6}"
          f"{100.0*(b-c)/k:>+7.1f}{mcnemar(b, c):>10.4f}")
print(f"{'POOLED':>13}{'':>10}{N:>7}{B:>6}{C:>6}{100.0*(B-C)/N:>+7.1f}{mcnemar(B, C):>10.2g}")
pos = sum(1 for v in cells.values() if v[0] > v[1])
print(f"\npositive at {pos}/{len(cells)} design points")
