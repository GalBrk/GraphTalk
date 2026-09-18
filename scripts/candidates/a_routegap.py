"""(a) Does the route GAP predict the effect better than baseline alone,
and does it transfer across corpora where baseline does not?"""
import json, sys, os
sys.path.insert(0, "scripts")
import analyze_baseline_law as abl

bars = json.load(open("shortcuts.json", encoding="utf-8"))

def enrich(cells):
    out = []
    for c in cells:
        if c["task"] in abl.DEGENERATE_TASKS:
            continue
        bar = bars.get(f'{c["task"]}/{c["condition"]}')
        if bar is None:
            continue
        c["bar"] = bar
        c["gap"] = bar - c["baseline"]
        out.append(c)
    return out

def load_densfull():
    cells = []
    for arm in abl.DENSFULL_ARMS:
        cells += abl.arm_cells(arm, [f"runs/{arm}.densfull40.shard*of*.jsonl"], bars)
    return enrich(cells)

def load_heldout():
    cells = []
    for arm, globs in abl.HELDOUT_GLOBS.items():
        cells += abl.arm_cells(arm, globs, bars, by_density=False)
    return enrich(cells)

def report(cells, name):
    for label, key in (("baseline", "baseline"), ("route gap (bar - baseline)", "gap")):
        for sub, tag in ((("route", True)), (("no route", False))):
            grp = [c for c in cells if c["route"] is tag]
            if len(grp) < 8:
                print(f"  {name:<12} {label:<26} {sub:<9} k={len(grp)} (too few)")
                continue
            r, p = abl.pearson([c[key] for c in grp], [c["delta"] for c in grp])
            slope, intercept = abl.fit_line([c[key] for c in grp], [c["delta"] for c in grp])
            cross = -intercept / slope if slope else float("nan")
            print(f"  {name:<12} {label:<26} {sub:<9} k={len(grp):>4} "
                  f"r={r:+.3f} p={p:<9.2g} slope={slope:+7.1f} crosses={cross:+.3f}")

def transfer(train, test, key):
    tr = [c for c in train if c["route"]]
    te = [c for c in test if c["route"]]
    slope, intercept = abl.fit_line([c[key] for c in tr], [c["delta"] for c in tr])
    hit = sum(1 for c in te if (slope * c[key] + intercept > 0) == (c["delta"] > 0))
    maj = max(sum(1 for c in te if c["delta"] > 0), sum(1 for c in te if c["delta"] <= 0))
    print(f"  fit on densfull40 ({key}) -> sign on heldout: {hit}/{len(te)} "
          f"= {hit/len(te):.3f}   majority = {maj/len(te):.3f}")

if __name__ == "__main__":
    d = load_densfull()
    h = load_heldout()
    print("=== (a) predictor comparison ===")
    report(d, "densfull40")
    report(h, "heldout")
    print()
    print("=== (a) cross-corpus sign transfer, route cells only ===")
    transfer(d, h, "baseline")
    transfer(d, h, "gap")
    json.dump({"densfull40": d, "heldout": h}, open(sys.argv[1], "w"), indent=0)
