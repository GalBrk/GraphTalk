"""(b) When a `degree` primer makes the model wrong, is it wrong by COPYING?"""
import json, glob, re, sys, collections
sys.path.insert(0, "scripts")
import analyze_baseline_law as abl
from graphtalk import scoring

ADJ = re.compile(r"Node (\d+) is connected to nodes? ([\d, ]+)\.")
QRY = re.compile(r"What is the degree of node (\d+)\?")

def graphs(path):
    """instance_id -> (degree-by-node dict, queried node, neighbours of it)"""
    out = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if d["task"] != "node_degree" or d["instance_id"] in out:
            continue
        adj = {int(n): [int(x) for x in nb.replace(" ", "").split(",") if x]
               for n, nb in ADJ.findall(d["prompt"])}
        q = QRY.search(d["prompt"])
        if not q:
            continue
        k = int(q.group(1))
        deg = {n: len(v) for n, v in adj.items()}
        out[d["instance_id"]] = (deg, k, adj.get(k, []))
    return out

def responses(pattern):
    seen = {}
    for path in sorted(glob.glob(pattern)):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["task"] != "node_degree" or r.get("hit_cap"):
                continue
            seen.setdefault((r["instance_id"], r["condition"]), r)
    return seen

G = graphs("prompts.densfull40.jsonl")
print(f"parsed {len(G)} node_degree graphs from prompts.densfull40.jsonl\n")

print("=== (b) wrong answers under `degree` vs `none` (densfull40, node_degree) ===")
print(f"{'arm':<17}{'cond':<9}{'wrong':>7}{'in multiset':>13}{'= nbr deg':>11}"
      f"{'= k+-1 deg':>12}{'chance':>9}")
detail = {}
for arm in abl.DENSFULL_ARMS:
    R = responses(f"runs/{arm}.densfull40.shard*of*.jsonl")
    for cond in ("none", "degree", "all", "clustering", "filler", "rwse", "components"):
        tot = inset = nbr = adjid = 0
        chance = 0.0
        for (iid, c), r in R.items():
            if c != cond or iid not in G:
                continue
            deg, k, nbrs = G[iid]
            gold = int(r["gold"])
            got = scoring.extract_answer(r["response"], "node_degree")
            try:
                v = int(str(got).strip())
            except (TypeError, ValueError):
                continue
            if v == gold:
                continue
            tot += 1
            vals = list(deg.values())
            wrong_vals = [x for x in vals if x != gold]
            if v in set(wrong_vals):
                inset += 1
            if any(deg.get(n) == v for n in nbrs):
                nbr += 1
            if deg.get(k - 1) == v or deg.get(k + 1) == v:
                adjid += 1
            # chance: P(a uniform draw from 0..max_deg lands in the wrong-value set)
            hi = max(vals) if vals else 1
            chance += len({x for x in wrong_vals}) / (hi + 1)
        if tot:
            print(f"{arm:<17}{cond:<9}{tot:>7}{100.0*inset/tot:>12.1f}%"
                  f"{100.0*nbr/tot:>10.1f}%{100.0*adjid/tot:>11.1f}%"
                  f"{100.0*chance/tot:>8.1f}%")
            detail[(arm, cond)] = (tot, inset, nbr, adjid)
json.dump({f"{a}|{c}": v for (a, c), v in detail.items()},
          open(sys.argv[1], "w"), indent=0)
