"""(f) The never-reported retrieval probe: how reliable IS the primer route?"""
import json, glob, collections, sys
sys.path.insert(0, "scripts")
import analyze_baseline_law as abl
from graphtalk import scoring

rows = {}
for path in sorted(glob.glob("runs/qwen3-1.7b.retrieval*.jsonl")):
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["instance_id"] in rows:
            continue
        rows[r["instance_id"]] = (
            None if r.get("hit_cap") else scoring.score_one(
                scoring.extract_answer(r["response"], r["task"]),
                r["gold"], r["task"])["primary"])

by_k, by_pos, by_size, joint = (collections.defaultdict(list) for _ in range(4))
capped = 0
for iid, s in rows.items():
    _, k, pos, size, _ = iid.split("/")
    if s is None:
        capped += 1
        continue
    by_k[int(k[1:])].append(s)
    by_pos[float(pos[3:])].append(s)
    by_size[size].append(s)
    joint[(int(k[1:]), float(pos[3:]))].append(s)

print(f"qwen3-1.7b retrieval probe: {len(rows)} instances, {capped} truncated\n")
for name, d in (("k (facts in the primer)", by_k), ("position of the needle", by_pos),
                ("size class", by_size)):
    print(f"  by {name}:")
    for key, v in sorted(d.items(), key=lambda kv: str(kv[0])):
        print(f"    {str(key):>8}  acc {sum(v)/len(v):.3f}   n={len(v)}")
    print()
ks = sorted({k for k, _ in joint}); ps = sorted({p for _, p in joint})
print("  accuracy by k x position:")
print("      k \ pos " + "".join(f"{p:>8}" for p in ps))
for k in ks:
    print(f"      {k:>7}  " + "".join(
        f"{(sum(joint[(k,p)])/len(joint[(k,p)]) if joint.get((k,p)) else float('nan')):>8.3f}"
        for p in ps))
xs = [int(i.split("/")[1][1:]) for i in rows if rows[i] is not None]
ys = [rows[i] for i in rows if rows[i] is not None]
r, p = abl.pearson(xs, ys)
print(f"\n  r(k, correct)        = {r:+.3f}  p={p:.2g}")
xs = [float(i.split("/")[2][3:]) for i in rows if rows[i] is not None]
r, p = abl.pearson(xs, ys)
print(f"  r(position, correct) = {r:+.3f}  p={p:.2g}")
