"""(f) Does the primer effect survive renaming nodes? GoT vs integer, same split."""
import json, glob, collections, sys
sys.path.insert(0, "scripts")
from graphtalk import scoring, node_naming

NODES = {}
for line in open("prompts_got.jsonl", encoding="utf-8"):
    d = json.loads(line)
    NODES[d["instance_id"]] = int(str(d["nodes"]).strip())

def score(patterns, got):
    out = {}
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["instance_id"], r["condition"])
                if key in out or str(r.get("hit_cap")).lower() == "true":
                    continue
                text = r["response"]
                if got:
                    n = NODES.get(r["instance_id"])
                    if n is None:
                        continue
                    text = node_naming.desubstitute_response(
                        text, {i: node_naming.GOT_NAMES[i] for i in range(n)})
                out[key] = scoring.score_one(
                    scoring.extract_answer(text, r["task"]), r["gold"], r["task"])["primary"]
    return out

ARMS = ["qwen3-8b", "qwen3-8b-think", "qwen3-14b", "qwen3-14b-think",
        "gemma4-e4b", "gemma4-e4b-think", "gemma4-12b", "gemma4-12b-think"]
TASKS = ["node_count", "edge_count", "node_degree", "connected_nodes",
         "edge_existence", "cycle_check"]

print("Paired delta vs `none`, published split. INT = integer ids, GOT = GoT names.")
print(f"{'arm':<19}{'task':<17}{'cond':<11}{'INT d':>8}{'GOT d':>8}{'n':>6}")
summary = collections.defaultdict(lambda: [[], []])
for arm in ARMS:
    ints = score([f"runs/{arm}.jsonl", f"runs/{arm}.shard*.jsonl"], got=False)
    gots = score([f"runs/{arm}.got.jsonl", f"runs/{arm}.got.shard*.jsonl"], got=True)
    if not gots:
        continue
    for task in TASKS:
        for cond in ("degree", "clustering", "components", "rwse", "all", "filler"):
            d = {}
            for tag, S in (("INT", ints), ("GOT", gots)):
                ids = [i for (i, c) in S if c == cond and i.startswith(task + "/")
                       and (i, "none") in S]
                if len(ids) < 10:
                    continue
                d[tag] = (100.0 * sum(S[(i, cond)] - S[(i, "none")] for i in ids) / len(ids),
                          len(ids))
            if len(d) == 2:
                print(f"{arm:<19}{task:<17}{cond:<11}{d['INT'][0]:>+8.1f}"
                      f"{d['GOT'][0]:>+8.1f}{d['INT'][1]:>6}")
                summary[cond][0].append(d["INT"][0])
                summary[cond][1].append(d["GOT"][0])

print("\nmean paired delta over all (arm, task) cells:")
print(f"{'cond':<12}{'INT':>8}{'GOT':>8}{'cells':>7}{'r(INT,GOT)':>12}")
import analyze_baseline_law as abl
allI, allG = [], []
for cond, (I, G) in summary.items():
    r = abl.pearson(I, G)[0] if len(I) >= 8 else float("nan")
    print(f"{cond:<12}{sum(I)/len(I):>+8.2f}{sum(G)/len(G):>+8.2f}{len(I):>7}{r:>12.3f}")
    allI += I; allG += G
print(f"{'ALL':<12}{sum(allI)/len(allI):>+8.2f}{sum(allG)/len(allG):>+8.2f}"
      f"{len(allI):>7}{abl.pearson(allI, allG)[0]:>12.3f}"
      f"   p={abl.pearson(allI, allG)[1]:.2g}")
