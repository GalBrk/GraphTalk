import json, sys
sys.path[:0] = ["superseded/scripts", "scripts"]  # archived full copies first
import analyze_baseline_law as abl
d = json.load(open(sys.argv[1]))
tr = [c for c in d["densfull40"] if c["route"]]
te = [c for c in d["heldout"] if c["route"]]
for key in ("baseline", "gap"):
    s, i = abl.fit_line([c[key] for c in tr], [c["delta"] for c in tr])
    pp = sum(1 for c in te if s*c[key]+i > 0)
    print(f"{key:<9} fitted line predicts positive for {pp}/{len(te)} held-out route cells")
pos = sum(1 for c in te if c["delta"] > 0)
print(f"actual: {pos}/{len(te)} held-out route cells have delta > 0 "
      f"(majority class = {'neg' if pos < len(te)/2 else 'pos'})")
print("\nheld-out route cells, baseline distribution:")
bs = sorted(c["baseline"] for c in te)
print(f"  min {bs[0]:.3f}  q1 {bs[len(bs)//4]:.3f}  med {bs[len(bs)//2]:.3f}  max {bs[-1]:.3f}")
print(f"  {sum(1 for b in bs if b > 0.792)}/{len(bs)} sit above the densfull40 crossover 0.792")
