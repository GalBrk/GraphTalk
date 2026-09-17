import json, sys, collections
rows = json.load(open(sys.argv[1]))
flat = [r for v in rows.values() for r in v]

def pool(arm, task, cond):
    sel = [r for r in flat if r["arm"] == arm and r["task"] == task and r["condition"] == cond]
    n = sum(r["n"] for r in sel); b = sum(r["helped"] for r in sel); c = sum(r["hurt"] for r in sel)
    tk = [r["dtokens"] for r in sel if r["dtokens"] is not None]
    return n, b, c, (sum(tk)/len(tk) if tk else float("nan"))

print("=== (d) discordant pairs on the cells the paper highlights ===")
print(f"{'arm':<17}{'task':<17}{'cond':<11}{'n':>5}{'help':>6}{'hurt':>6}"
      f"{'net':>8}{'churn%':>8}{'sig/churn':>11}")
HIGH = [("qwen3-4b","edge_count","degree"), ("qwen3-1.7b","node_degree","degree"),
        ("qwen3-4b","node_degree","degree"), ("qwen3-1.7b","node_degree","all"),
        ("qwen3-4b","node_degree","all"), ("qwen3-1.7b","node_degree","clustering"),
        ("qwen3-1.7b","edge_existence","filler"), ("qwen3-1.7b","connected_nodes","filler"),
        ("qwen3-4b","connected_nodes","filler"), ("qwen3-4b","edge_existence","filler"),
        ("qwen3-1.7b","edge_existence","all"), ("qwen3-1.7b","node_count","filler"),
        ("qwen3-1.7b","node_count","clustering"), ("qwen3-4b","node_count","filler")]
for arm, task, cond in HIGH:
    n,b,c,tk = pool(arm, task, cond)
    if not n: continue
    print(f"{arm:<17}{task:<17}{cond:<11}{n:>5}{b:>6}{c:>6}"
          f"{100.0*(b-c)/n:>+7.1f}{100.0*(b+c)/n:>8.1f}{abs(b-c)/(b+c) if b+c else 0:>11.2f}")

print("\n=== (d) churn summary, all 546 cells ===")
tot_n = sum(r["n"] for r in flat); tot_ch = sum(r["churn"] for r in flat)
tot_net = sum(abs(r["helped"]-r["hurt"]) for r in flat)
print(f"  pairs {tot_n}, discordant {tot_ch} ({100.0*tot_ch/tot_n:.1f}%), "
      f"|net| {tot_net} = {100.0*tot_net/tot_ch:.1f}% of discordant pairs")
big = sorted((r for r in flat if r["churn"] >= 20), key=lambda r: abs(r["helped"]-r["hurt"])/r["churn"])[:8]
print("  most churn-per-net (>=20 discordant pairs), i.e. most 'randomising':")
for r in big:
    print(f"    {r['arm']:<17}{r['task']:<17}{r['condition']:<10}p={r['density']} "
          f"help {r['helped']:>3} hurt {r['hurt']:>3} net {r['delta']:+.1f} "
          f"ratio {abs(r['helped']-r['hurt'])/r['churn']:.2f}")

print("\n=== (e) mean paired change in generated tokens vs none ===")
print(f"{'arm':<17}" + "".join(f"{c:>12}" for c in
      ["components","clustering","rwse","degree","filler","all"]))
for arm in rows:
    line = f"{arm:<17}"
    for cond in ["components","clustering","rwse","degree","filler","all"]:
        sel = [r["dtokens"] for r in flat if r["arm"]==arm and r["condition"]==cond
               and r["dtokens"] is not None]
        line += f"{(sum(sel)/len(sel) if sel else float('nan')):>+12.1f}"
    print(line)
print("\n  per task, qwen3-1.7b (plain) and qwen3-1.7b-think:")
for arm in ["qwen3-1.7b", "qwen3-1.7b-think"]:
    for task in sorted({r["task"] for r in flat}):
        sel = {c: [r["dtokens"] for r in flat if r["arm"]==arm and r["task"]==task
                   and r["condition"]==c and r["dtokens"] is not None]
               for c in ["degree","filler","all"]}
        vals = {c: (sum(v)/len(v) if v else float("nan")) for c,v in sel.items()}
        print(f"    {arm:<18}{task:<17}" +
              "".join(f"{c}={vals[c]:+8.1f}  " for c in vals))
