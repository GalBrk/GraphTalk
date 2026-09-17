"""(d) discordant-pair decomposition and (e) output length, one pass over a corpus."""
import json, glob, sys, collections, os
sys.path.insert(0, "scripts")
import analyze_baseline_law as abl
from graphtalk import scoring

def load(patterns):
    """-> {(task, dens, cond, iid): (score|None, n_new_tokens)}"""
    out, seen = {}, set()
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            if ".got." in os.path.basename(path):
                continue
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                k = (r["task"], abl.density_of(r["instance_id"]),
                     r["condition"], r["instance_id"])
                if k in seen:
                    continue
                seen.add(k)
                s = None if r.get("hit_cap") else scoring.score_one(
                    scoring.extract_answer(r["response"], r["task"]),
                    r["gold"], r["task"])["primary"]
                out[k] = (s, r.get("n_new_tokens"))
    return out

def cells(data, control="none"):
    g = collections.defaultdict(dict)
    for (task, dens, cond, iid), v in data.items():
        g[(task, dens, cond)][iid] = v
    rows = []
    for (task, dens, cond), vals in sorted(g.items(), key=repr):
        if cond == control:
            continue
        base = g.get((task, dens, control), {})
        b = c = n = 0
        dlen, nlen = 0.0, 0
        for iid, (s, tok) in vals.items():
            bs = base.get(iid)
            if bs is None:
                continue
            bscore, btok = bs
            if bscore is None or s is None:
                continue  # hit_cap: excluded from length too, 8192 is censored
            if btok is not None and tok is not None:
                dlen += tok - btok
                nlen += 1
            n += 1
            # binary tasks: exact 0/1. connected_nodes is set-F1 -> threshold.
            if s > bscore:
                b += 1
            elif s < bscore:
                c += 1
        if n < 10:
            continue
        rows.append(dict(task=task, density=dens, condition=cond, n=n,
                         helped=b, hurt=c, churn=b + c,
                         delta=100.0 * (b - c) / n,
                         dtokens=dlen / nlen if nlen else None))
    return rows

if __name__ == "__main__":
    allrows = {}
    for arm in abl.DENSFULL_ARMS:
        rows = cells(load([f"runs/{arm}.densfull40.shard*of*.jsonl"]))
        for r in rows:
            r["arm"] = arm
        allrows[arm] = rows
        print(f"loaded {arm}: {len(rows)} cells", flush=True)
    json.dump(allrows, open(sys.argv[1], "w"), indent=0)
