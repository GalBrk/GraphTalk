"""Draw and score the hand-validation sheet for docs/primer-directions.md.

Six checks, each a seeded sample of the 40-node sweep's responses. What to label
and why is in docs/primer-directions-validation.md.

  C1  A1: does a response itself arrive at its final answer N ('own'), or not
      ('other')? Both classes are drawn and shown the same way.
  C2  A1/A3, edge_count, thinking arms: what does a reported conflict compare?
  C3  A3: is the first captured candidate an answer to the question asked?
  C4  A3: does a revision word start a revision of the model's own work?
  C5  A2: does the text state the marked value as that node's clustering
      coefficient / return probability? (the wrong quotes, which the misread
      finding rests on)
  C6  reports_conflict (response_patterns), which A1 relies on: does the text
      report an actual disagreement between two values?

The sheet (SHEET) shows each item's question and text, with the detector's
evidence marked «like this». Model, primer and the detector's verdict are only in
the key (KEY), so the labeller is blind to them.

  PYTHONPATH=. python scripts/validate_directions.py --make
  PYTHONPATH=. python scripts/validate_directions.py --score \\
      > outputs/n40-sweep/directions_validation.txt
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import primer_directions as pd_  # noqa: E402
import primer_findings as pf  # noqa: E402
import response_patterns as rp  # noqa: E402

SHEET = "outputs/n40-sweep/directions_validation_sheet.csv"
KEY = "outputs/n40-sweep/directions_validation_key.csv"
PASS = 0.9                                  # the repo's bar for a text marker
THINK, TASKS2 = pd_.THINK, ("node_degree", "edge_count")

QUESTION = {
    "C2": "What does the marked conflict compare?  a = a value given in the prompt vs the "
          "model's own count or sum;  b = two results of the model's own counting or "
          "arithmetic;  c = something else, or unclear;  d = no actual conflict is reported",
    "C4": "Does the marked word start a revision or re-check of the model's own work "
          "(going back over a count, a value or a step)?  y / n / ?",
    "C6": "Does the text at the marked word report an actual disagreement between two "
          "values (not only checking whether there is one)?  y / n / ?",
}
IF_FAILS = {
    "C1": "A1's own/other split (e.g. plain 4B keeps its own count in 41% of conflicts) "
          "cannot be reported; keep only 'primer's value vs not'.",
    "C2": "Rewrite A1/A3's edge_count reading to what the conflicts actually compare.",
    "C3": "A3's commitment numbers (first candidate position, first = final) are dropped.",
    "C4": "A3's revision-rate claim (doubles under degree on edge_count) is dropped.",
    "C5": "A2's neighbouring-line misread claim is dropped; 'quotes are accurate' stands "
          "only if the right quotes are also checked.",
    "C6": "Every A1 number rests on this marker: A1 is dropped until the marker is fixed.",
}


# ------------------------------------------------------------------ text helpers
def window(text, start, end, before=200, after=200):
  """One line of text around [start, end), that span marked «...»."""
  a, b = max(0, start - before), min(len(text), end + after)
  s = (("… " if a else "") + text[a:start] + "«" + text[start:end] + "»" + text[end:b]
       + (" …" if b < len(text) else ""))
  return " ".join(s.split())


def spread(items, k):
  """At most k items, evenly spaced from first to last, in order."""
  if len(items) <= k:
    return list(items)
  return [items[round(j * (len(items) - 1) / (k - 1))] for j in range(k)]


def conflict_span(text, targets):
  """The discrepancy word reports_conflict accepted."""
  m = pf._DISCREPANCY.search(text, rp._conflict(text, targets))
  return m.start(), m.end()


def asks(task, targets):
  return f"the degree of node {targets[0]}" if task == "node_degree" else \
      "the number of edges in the graph"


def pick(rng, index, n):
  return list(rng.choice(np.asarray(index), min(n, len(index)), replace=False))


# ------------------------------------------------------------------ the six samples
def c1(d, rng):
  x = d[d.task.isin(TASKS2) & d.condition.isin(["degree", "all"]) & (d.reports_conflict == 1)
        & (d.hit_cap == 0)].copy()
  x["kind"] = [pd_.source(pd_.to_int(p), pd_.to_int(g), t)
               for p, g, t in zip(x.pred, x.gold, x.text)]
  out = []
  for kind, n in (("own", 25), ("other", 15)):
    for i in pick(rng, x.index[x.kind == kind], n):
      r = x.loc[i]
      ans, gold = pd_.to_int(r.pred), pd_.to_int(r.gold)
      t = r.text.replace("*", "")
      # The places the detector read the answer as computed come first, so the
      # labeller judges that evidence; other mentions of the number fill up to six.
      evidence = [m for rx in (pd_._COUNTED, rp._HALF) for m in rx.finditer(t)
                  if ans in {int(v) for v in m.groups() if v}][:3]
      at = {m.start() for m in evidence}
      # A sentence-final "…is 28." counts; only a decimal like "28.5" is skipped.
      rest = [m for m in re.finditer(rf"(?<![\d.]){ans}(?!\d)(?!\.\d)", t)
              if not any(abs(m.start() - a) < 80 for a in at)]
      shown_m = sorted(evidence + spread(rest, 6 - len(evidence)), key=lambda m: m.start())
      shown = " ||| ".join(window(t, m.start(), m.end(), 150, 150) for m in shown_m)
      q = (f"The question asks for {asks(r.task, r.targets)}. The value given in the prompt is "
           f"{gold}; the final answer is {ans}. Below, separated by |||, are the places the "
           f"text writes {ans} (at most six, first to last). Apart from stating the final "
           f"answer, does the text arrive at {ans} as its own count or computation "
           f"('I count {ans}', 'that's {ans} connections', 'S / 2 = {ans}')?  y / n / ?")
      out.append((r, "C1", q, shown, "y" if kind == "own" else "n", kind))
  return out


def c2(d, rng):
  x = d[(d.task == "edge_count") & d.arm.isin(THINK) & (d.reports_conflict == 1)]
  out = []
  for c in ("none", "degree"):
    for i in pick(rng, x.index[x.condition == c], 20):
      r = x.loc[i]
      s, e = conflict_span(r.text, r.targets)
      out.append((r, "C2", QUESTION["C2"], window(r.text, s, e, 400, 300), "", c))
  return out


def c3(d, rng):
  out = []
  for task in TASKS2:
    for c in ("none", "degree"):
      x = d[(d.task == task) & d.arm.isin(THINK) & (d.condition == c) & (d.hit_cap == 0)]
      found = {}
      for i, text, targets in zip(x.index, x.text, x.targets):
        tr = pd_.trace_of(text).replace("*", "")
        cand = pd_.candidates(task, tr, targets[0] if targets else None)
        if cand:
          found[i] = (tr, cand[0])
      for i in pick(rng, list(found), 10):
        r, (tr, (at, value)) = x.loc[i], found[i]
        span = re.compile(rf"(?<!\d){value}(?!\d)").search(tr, at)
        s, e = (span.start(), span.end()) if span else (at, at + 1)
        q = (f"The question asks for {asks(task, r.targets)}. Is «{value}» the model's "
             f"candidate answer to that question at this point in its reasoning?  y / n / ?")
        out.append((r, "C3", q, window(tr, s, e), "y", c))
  return out


def c4(d, rng):
  out = []
  x = d[d.task.isin(TASKS2) & d.arm.isin(THINK)]
  for c in ("none", "degree"):
    y = x[x.condition == c]
    has = [i for i, t in zip(y.index, y.text) if pd_._REVISE.search(pd_.trace_of(t))]
    for i in pick(rng, has, 15):
      r = y.loc[i]
      tr = pd_.trace_of(r.text)
      ms = list(pd_._REVISE.finditer(tr))
      m = ms[rng.integers(len(ms))]
      out.append((r, "C4", QUESTION["C4"], window(tr, m.start(), m.end()), "y", c))
  return out


def c5(d, rng):
  """Wrong quotes of 4B-T under `all`, where the misread finding is."""
  x = d[(d.arm == "qwen3-4b-think") & (d.condition == "all")]
  found = {"near": [], "other": []}
  for i, text, dens, idx in zip(x.index, x.text, x.density_class, x["index"]):
    values, t, seen = pd_.facts(dens, idx)[1], text.replace("*", ""), set()
    for feature in ("clustering", "rwse"):
      for rx in pd_._QUOTE[feature]:
        for m in rx.finditer(t):
          col = pd_.FEATURE[feature] + (m.groupdict().get("s") == "3")
          k, v = int(m["k"]), m["v"]
          kind = pd_.classify_quote(k, v, values, col)
          if kind in found and (k, v, col) not in seen:
            seen.add((k, v, col))
            what = ("clustering coefficient" if feature == "clustering" else
                    f"return probability after {3 if col == 3 else 2} steps")
            found[kind].append((i, t, m.start("v"), m.end("v"), k, v, what))
  out = []
  for kind, n in (("near", 30), ("other", 15)):
    for j in pick(rng, range(len(found[kind])), n):
      i, t, s, e, k, v, what = found[kind][j]
      q = f"Does the text state «{v}» as node {k}'s {what}?  y / n / ?"
      out.append((x.loc[i], "C5", q, window(t, s, e), "y", kind))
  return out


def c6(d, rng):
  x = d[(d.reports_conflict == 1)]
  out = []
  for arm in pd_.ARMS:
    for i in pick(rng, x.index[x.arm == arm], 8):
      r = x.loc[i]
      s, e = conflict_span(r.text, r.targets)
      out.append((r, "C6", QUESTION["C6"], window(r.text, s, e, 300, 200), "y", r.task))
  return out


def make():
  if os.path.exists(SHEET) and pd.read_csv(SHEET, dtype=str).label.notna().any():
    sys.exit(f"{SHEET} already has labels; move it aside before drawing a new sheet")
  d = rp.load()
  rng = np.random.default_rng(pf.SEED)
  sheet, key = [], []
  for draw in (c1, c2, c3, c4, c5, c6):
    items = draw(d, rng)
    for j in rng.permutation(len(items)):
      r, check, q, text, expected, detail = items[j]
      item = f"{check}-{sum(k['check'] == check for k in key) + 1:02d}"
      sheet.append(dict(id=item, check=check, question=q, text=text, label="", note=""))
      key.append(dict(id=item, check=check, arm=r.arm, task=r.task, condition=r.condition,
                      instance_id=r.instance_id, truncated=int(r.hit_cap),
                      expected=expected, detail=detail))
  pd.DataFrame(sheet).to_csv(SHEET, index=False, encoding="utf-8-sig")
  pd.DataFrame(key).to_csv(KEY, index=False)
  counts = pd.DataFrame(key).check.value_counts().sort_index()
  print(f"wrote {len(sheet)} items to {SHEET} and the key to {KEY}: "
        + ", ".join(f"{c} {n}" for c, n in counts.items()))


# ------------------------------------------------------------------ scoring
def clopper_pearson(k, n, alpha=0.05):
  """Exact 95% interval for k successes out of n."""
  from scipy.stats import beta
  lo = beta.ppf(alpha / 2, k, n - k + 1) if k else 0.0
  hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
  return lo, hi


def normalize(label):
  """'y', 'n', 'a'-'d' or '' (unlabelled or '?'), from whatever was typed."""
  s = str(label).strip().lower() if pd.notna(label) else ""
  s = {"yes": "y", "no": "n", "1": "y", "0": "n", "true": "y", "false": "n"}.get(s, s)
  return s[:1] if s[:1] in ("y", "n", "a", "b", "c", "d") else ""


def agreement(x):
  """(agreeing, labelled) for rows with y/n labels against the detector's claim."""
  x = x[x.label.isin(["y", "n"])]
  return int((x.label == x.expected).sum()), len(x)


def score():
  sheet = pd.read_csv(SHEET, dtype=str, encoding="utf-8-sig")
  key = pd.read_csv(KEY, dtype=str)
  m = sheet[["id", "label", "note"]].merge(key, on="id")
  m["label"] = m.label.map(normalize)
  print(f"[dvsummary] {int((m.label != '').sum())} of {len(m)} items labelled; a check "
        f"passes when the detector agrees with the labels on {100 * PASS:.0f}% or more")
  for check in ("C1", "C2", "C3", "C4", "C5", "C6"):
    x = m[m.check == check]
    tag = "dv" + check.lower()
    if check == "C2":
      print(f"[{tag}] C2 what the edge_count conflicts compare (a prompt value vs own count "
            f"/ b own results / c other / d no conflict), by primer:")
      for c, y in x.groupby("detail"):
        lab = y.label[y.label.isin(list("abcd"))]
        print(f"  {c:7s} labelled {len(lab)} of {len(y)}: " + " / ".join(
            f"{v} {int((lab == v).sum())}" for v in "abcd"))
      continue
    k, n = agreement(x)
    if not n:
      print(f"[{tag}] {check}: not labelled yet ({len(x)} items)")
      continue
    lo, hi = clopper_pearson(k, n)
    verdict = "PASS" if k / n >= PASS else "FAIL -- " + IF_FAILS[check]
    print(f"[{tag}] {check}: detector agrees with {k} of {n} labels = {100 * k / n:.0f}% "
          f"[{100 * lo:.0f}, {100 * hi:.0f}]; {verdict}")
    for g, y in x.groupby("detail"):
      gk, gn = agreement(y)
      if gn:
        print(f"  {g:12s} {gk} of {gn}")


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  g = ap.add_mutually_exclusive_group(required=True)
  g.add_argument("--make", action="store_true", help=f"draw {SHEET} and {KEY}")
  g.add_argument("--score", action="store_true", help="score the labelled sheet")
  args = ap.parse_args()
  sys.stdout.reconfigure(encoding="utf-8")
  if args.make:
    make()
  else:
    score()


if __name__ == "__main__":
  main()
