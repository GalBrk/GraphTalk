"""Does the paper's one surviving cell -- `clustering` on `node_degree` for
`qwen3-1.7b` -- also beat the length-matched `filler` control, not just
`none`? The main densfull40 sweep can't answer this: that corpus's own
clustering/node_degree cell is only +3.0pp and not significant (n=400,
pooled over p=0.10-0.50). The claim instead rests on a dedicated sweep at
400 graphs/density level, seven density levels, and that sweep only has a
filler arm (`degdensfill`) at all seven -- densfull40's `filler` doesn't
reach the extra densities added by `degdens40hi`.

  degdens40    (p=.10-.50): none, clustering, components
  degdens40hi  (p=.65-.85): none, components, clustering, degree
  degdensfill  (all 7 p):   filler
  degdensrep   (p=.10-.50, independent seed /s<seed>/): none, clustering
               -- the replication sample; never pooled with the above.

  PYTHONPATH=. python superseded/scripts/analyze_headline_robustness.py
"""
import glob
import json

from graphtalk import scoring

ARM = "qwen3-1.7b"
TASK = "node_degree"
MAIN_GLOBS = [
    f"runs/{ARM}.degdens40.shard*of5.jsonl",
    f"runs/{ARM}.degdens40hi.shard*of5.jsonl",
]
FILLER_GLOB = f"runs/{ARM}.degdensfill.shard*of5.jsonl"
REP_GLOB = f"runs/{ARM}.degdensrep.shard*of5.jsonl"


def _load(patterns):
  out, seen = {}, set()
  for pattern in patterns if isinstance(patterns, list) else [patterns]:
    for path in sorted(glob.glob(pattern)):
      with open(path, encoding="utf-8") as fh:
        for line in fh:
          if not line.strip():
            continue
          r = json.loads(line)
          k = (r["instance_id"], r["condition"])
          if k in seen or r.get("hit_cap"):
            continue
          seen.add(k)
          out[k] = scoring.score_one(
              scoring.extract_answer(r["response"], TASK), r["gold"], TASK
          )["exact"]
  return out


def paired_delta(scores, iids, cond_a, cond_b):
  """Mean(scores[cond_a] - scores[cond_b]) over ids present under both."""
  pairs = [(scores[(i, cond_a)], scores[(i, cond_b)]) for i in iids
           if (i, cond_a) in scores and (i, cond_b) in scores]
  a = [p[0] for p in pairs]
  b = [p[1] for p in pairs]
  delta = 100.0 * (sum(a) - sum(b)) / len(pairs) if pairs else float("nan")
  p = scoring.mcnemar(b, a)["p_value"] if pairs else float("nan")
  return delta, p, len(pairs)


def main():
  main_scores = _load(MAIN_GLOBS)
  filler_scores = _load(FILLER_GLOB)
  rep_scores = _load(REP_GLOB)
  scores = {**main_scores, **filler_scores}
  iids = {i for (i, _c) in scores}

  print(f"{ARM} / {TASK} / clustering: robustness beyond the main sweep\n")
  for label, cond_a, cond_b in (
      ("clustering vs none    (7 densities)", "clustering", "none"),
      ("clustering vs filler  (7 densities)", "clustering", "filler"),
  ):
    delta, p, n = paired_delta(scores, iids, cond_a, cond_b)
    print(f"  {label}: delta={delta:+.1f}pp  n={n}  p={p:.4g}")

  rep_iids = {i for (i, _c) in rep_scores}
  delta, p, n = paired_delta(rep_scores, rep_iids, "clustering", "none")
  print(f"\n  independent-seed replication (degdensrep, p=.10-.50):")
  print(f"    clustering vs none: delta={delta:+.1f}pp  n={n}  p={p:.4g}")


if __name__ == "__main__":
  main()
