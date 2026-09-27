# One source of truth — Phase 1 implementation plan (groundwork + 40-node sweep)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the 40-node sweep one pipeline, one results doc checked by a test, and move every other analysis of those runs to a tracked `superseded/` mirror, building the shared pieces (the truncation rule, the doc test, the move tooling) that Phases 2–6 reuse.

**Architecture:** A new `graphtalk/outcomes.py` holds the one truncation rule (R1). `scripts/primer_findings.py` and `scripts/analyze_primer_window.py` switch to it and gain the analyses the ledger still needs; their printed output, `csv2/raw-trends/primer_findings.txt`, is the only source for `docs/results/n40-sweep.md`, and `tests/test_results_docs.py` checks every number the doc cites against it. Everything else about these runs is moved with `git mv` into `superseded/`, and a throwaway script rewrites references to moved paths.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy (existing); pytest. On this laptop the interpreter is `C:\Users\Inbal\miniconda3\python.exe` (below: `$PY`, i.e. `/c/Users/Inbal/miniconda3/python` in Git Bash). Always run with `PYTHONPATH=.` from the repo root.

**Spec:** `docs/plans/2026-09-24-single-source-of-truth.md` (approved). Read its rules R1–R6 before starting.

## Global Constraints

- R1: every response is exactly one of `correct` / `wrong` / `truncated`; a response that hit the budget is `truncated` whatever its text says, never labelled wrong, never dropped. Effects are paired changes in the shares of all responses; the headline is the correct share; the truncated-share change is printed next to it. Answer descriptions (MAE, yes-rate, false alarms, route, wording) use finished responses only. A cell whose truncated share is 15% or more is flagged.
- R2: exact match primary; `connected_nodes` exact set match (F1 secondary); `edge_existence` with balanced accuracy and yes-rate.
- R3: every number in a results doc is written exactly as the output prints it, followed by ` [tag]`; `tests/test_results_docs.py` enforces it.
- R4: results docs state current claims only — no history, no mention of earlier versions.
- R6: moves use `git mv`; nothing is deleted except git-ignored build products and the byte-identical `Structural_Primers_Graph_Reasoning_ACL2023 (2).pdf`; the gitignored root `archive/` is not touched.
- The user's standing rule: ask for explicit permission immediately before editing or writing any file. Ask once per task, naming the files the task changes.
- Commit once, at the end of the phase, after verification is clean (spec: "Only then is the phase committed"). Stage as you go.
- The peer session `graphtalk-63` shares this working tree; it must stay idle while this runs.
- Baseline before any change: `pytest` gives 804 passed, 1 failed; the failure is `tests/test_prompts.py::test_run_sweep_row_carries_node_naming`, which fails on this laptop because `torch`'s `c10.dll` cannot load. That failure predates this work; do not try to fix it.

## Review Focus

1. A cell where one condition truncates on almost every graph (the thinking arms' `edge_count`, 62–81%): it must appear in `cells()` with a correct share near 0 and be flagged, not skipped. Test in Task 3.
2. A doc written with a Unicode minus (`−6.5`) against output printed with ASCII `-6.5`: the doc test must treat them as equal. Test in Task 2.
3. A cited value that is a prefix of a printed one (`2` vs `2.5`, `27.2` vs `27.25`): must not count as found. Test in Task 2.
4. `connected_nodes` predictions that are `No nodes`, empty, or unparsed, in the joint-correctness measure: an empty set is a valid answer only when parsed; an unparsed answer is never consistent. Test in Task 4.
5. A rendered RWSE primer that does not have exactly 40 node sentences: the count must stop with an error naming the instance, not report a wrong class count. Test in Task 4.

---

### Task 1: The truncation rule, `graphtalk/outcomes.py`

**Files:**
- Create: `graphtalk/outcomes.py`
- Test: `tests/test_outcomes.py`

**Interfaces:**
- Produces: `outcomes.CORRECT`, `WRONG`, `TRUNCATED` (str), `OUTCOMES` (tuple), `FLAG = 0.15`; `outcome(exact, hit_cap) -> np.ndarray[str]` (elementwise); `shares(outcomes) -> dict[str, float]`; `flagged(outcomes) -> bool`.

- [ ] **Step 1: Write the failing test** — `tests/test_outcomes.py`:

```python
"""R1: a response that hit the token budget is its own outcome."""
import pandas as pd
import pytest

from graphtalk import outcomes as oc


def test_truncated_wins_over_the_text_it_stopped_on():
  assert list(oc.outcome([1, 0, 1, 0], [True, True, False, False])) == [
      "truncated", "truncated", "correct", "wrong"]


def test_outcome_accepts_pandas_columns():
  f = pd.DataFrame({"exact": [1.0, 0.0], "hit_cap": [0, 1]})
  assert list(oc.outcome(f.exact, f.hit_cap)) == ["correct", "truncated"]


def test_shares_are_of_all_responses_and_sum_to_one():
  s = oc.shares(["correct", "wrong", "truncated", "correct"])
  assert s == {"correct": 0.5, "wrong": 0.25, "truncated": 0.25}


def test_shares_of_no_responses_is_an_error():
  with pytest.raises(ValueError):
    oc.shares([])


def test_flag_is_inclusive_at_fifteen_percent():
  assert oc.flagged(["truncated"] * 3 + ["correct"] * 17)
  assert not oc.flagged(["truncated"] * 2 + ["correct"] * 18)
```

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONPATH=. $PY -m pytest -q tests/test_outcomes.py -p no:cacheprovider`
Expected: FAIL, `ImportError: cannot import name 'outcomes'`.

- [ ] **Step 3: Write the module** — `graphtalk/outcomes.py`:

```python
"""The one rule for a response that hit the token budget.

Every response has exactly one outcome: correct, wrong or truncated. A response
that used the whole budget is truncated whatever its abandoned text says (the
text it stopped on can match the gold answer by accident), and it is never
labelled wrong or dropped. Analyses report the three outcomes as shares of all
responses and a primer's effect as the paired change in each share. Quantities
that describe an answer (error size, yes-rate, false alarms, response text) use
finished responses only. A cell whose truncated share is FLAG or more is
flagged. Rule R1 of docs/plans/2026-09-24-single-source-of-truth.md.
"""
import numpy as np

CORRECT, WRONG, TRUNCATED = "correct", "wrong", "truncated"
OUTCOMES = (CORRECT, WRONG, TRUNCATED)
FLAG = 0.15


def outcome(exact, hit_cap):
  """Elementwise outcome: `exact` is the scorer's 1/0, `hit_cap` whether the
  generation used the whole budget."""
  exact = np.asarray(exact, dtype=float)
  cap = np.asarray(hit_cap, dtype=bool)
  return np.where(cap, TRUNCATED, np.where(exact == 1, CORRECT, WRONG))


def shares(outcomes):
  """Share of each outcome among all the responses given; they sum to 1."""
  o = np.asarray(outcomes)
  if o.size == 0:
    raise ValueError("no responses to take shares of")
  return {k: float(np.mean(o == k)) for k in OUTCOMES}


def flagged(outcomes):
  """True when the truncated share reaches FLAG."""
  return shares(outcomes)[TRUNCATED] >= FLAG
```

- [ ] **Step 4: Run it to see it pass**

Run: `PYTHONPATH=. $PY -m pytest -q tests/test_outcomes.py -p no:cacheprovider`
Expected: 5 passed.

- [ ] **Step 5: Stage** — `git add graphtalk/outcomes.py tests/test_outcomes.py`

---

### Task 2: The results-doc test, `tests/test_results_docs.py`

**Files:**
- Create: `tests/test_results_docs.py`

**Interfaces:**
- Produces: `check(doc_text: str, read: Callable[[str], str]) -> list[str]` (problems; empty means clean), `blocks(text) -> dict[str, str]`, `citations(text) -> list[tuple[str, str]]`. The doc format every later phase uses: one or more lines `Source: \`<path>\`` and citations `<value> [<tag>]`.

- [ ] **Step 1: Write the test file** (the unit tests below exercise `check` on inline text, so they fail until `check` exists; the parametrized test runs over `docs/results/*.md` and has no cases yet):

```python
"""Every number a results doc cites must appear, as printed, in the output it
names (rule R3 of docs/plans/2026-09-24-single-source-of-truth.md).

A doc in docs/results/ other than README.md names its sources on lines
"Source: `<path>`" and cites a number as "<value> [<tag>]". The value must occur
in that tag's block of one of the sources: every line that starts with "[tag]",
plus the lines after it up to the next line that starts with a tag.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = sorted(p for p in (ROOT / "docs" / "results").glob("*.md")
              if p.name != "README.md")
SOURCE = re.compile(r"^Source: `([^`]+)`", re.M)
TAG_LINE = re.compile(r"^\[([a-z0-9]+)\]")
CITE = re.compile(r"(?<![\w.])([+\-\u2212]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)%?) "
                  r"\[([a-z0-9]+)\](?![(\[:])")


def blocks(text):
  out, tag = {}, None
  for line in text.splitlines():
    m = TAG_LINE.match(line)
    if m:
      tag = m.group(1)
    if tag:
      out.setdefault(tag, []).append(line)
  return {k: "\n".join(v) for k, v in out.items()}


def citations(text):
  return [(v.replace("\u2212", "-"), t) for v, t in CITE.findall(text)]


def check(doc_text, read):
  sources = SOURCE.findall(doc_text)
  if not sources:
    return ["names no Source"]
  found = {}
  for s in sources:
    for tag, block in blocks(read(s)).items():
      found[tag] = found.get(tag, "") + "\n" + block
  cites = citations(doc_text)
  if not cites:
    return ["cites no numbers"]
  problems = []
  for value, tag in cites:
    if tag not in found:
      problems.append(f"[{tag}] is not a tag in {sources}")
    elif not re.search(rf"(?<![\d.]){re.escape(value)}(?!\d|\.\d)", found[tag]):
      problems.append(f"{value} not found under [{tag}]")
  return problems


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_cited_number_is_in_its_source(doc):
  assert check(doc.read_text(encoding="utf-8"),
               lambda s: (ROOT / s).read_text(encoding="utf-8")) == []


SRC = ("[flip] qwen3-4b node_degree\n  degree p=0.50 base 99.0 delta -12.0\n"
       "[route] retrieves 44% at 27.25\n")
DOC = ("Source: `out.txt`\n\nAt p=.50 the primer costs -12.0 [flip]; "
       "it retrieves 44% [route].\n")


def test_check_accepts_numbers_the_source_prints():
  assert check(DOC, lambda s: SRC) == []


def test_check_reports_a_number_the_source_does_not_print():
  assert check(DOC.replace("-12.0", "-13.0"), lambda s: SRC) == [
      "-13.0 not found under [flip]"]


def test_check_reports_an_unknown_tag_and_an_empty_doc():
  assert check(DOC.replace("[route]", "[routes]"), lambda s: SRC)[0].startswith(
      "[routes] is not a tag")
  assert check("Source: `out.txt`\n", lambda s: SRC) == ["cites no numbers"]
  assert check("no source line", lambda s: SRC) == ["names no Source"]


def test_unicode_minus_matches_ascii_output():
  assert check(DOC.replace("-12.0", "\u221212.0"), lambda s: SRC) == []


def test_a_prefix_of_a_printed_number_is_not_found():
  assert check(DOC.replace("44%", "27.2"), lambda s: SRC) == [
      "27.2 not found under [route]"]
  assert check(DOC.replace("-12.0", "-12"), lambda s: SRC) == [
      "-12 not found under [flip]"]
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=. $PY -m pytest -q tests/test_results_docs.py -p no:cacheprovider`
Expected: 5 passed, 1 skipped (the empty parametrization over `docs/results/`). If a unit test fails, fix `check`/`CITE`, not the test.

- [ ] **Step 3: Stage** — `git add tests/test_results_docs.py`

---

### Task 3: Switch the 40-node pipeline to R1

**Files:**
- Modify: `scripts/primer_findings.py` (whole file; functions named below)
- Modify: `scripts/analyze_primer_window.py:43-72` (`cells`)
- Test: `tests/test_primer_findings.py` (replace `test_pairs_drop_a_pair_when_either_side_truncates`; add two tests)

**Interfaces:**
- Consumes: `graphtalk.outcomes.outcome`, `CORRECT`, `TRUNCATED`, `FLAG`.
- Produces: `primer_findings.with_outcomes(f) -> DataFrame` (adds 0/1 columns `correct`, `truncated`); `pairs(f, arm, task, a, b, dens)` now keeps every pair; `finished(j)` (pairs where both responses finished); `effect(j, col="correct") -> (d, lo, hi, broke, fixed, p, n, d_truncated)` where `d_truncated` is NaN unless `col == "correct"`; `apw.cells()` rows gain `trunc_a`, `trunc_b`, `flagged`.

- [ ] **Step 1: Replace the pairing test and add two** in `tests/test_primer_findings.py`. Delete `test_pairs_drop_a_pair_when_either_side_truncates` and add, after the imports, `import pytest` and `import analyze_primer_window as apw  # noqa: E402` (after the existing `import primer_findings as pf`), then:

```python
def _four_graphs():
  rows = []
  # (exact_none, cap_none, exact_degree, cap_degree) per graph
  for g, (ex_a, cap_a, ex_b, cap_b) in enumerate(
      [(1, 0, 0, 0), (0, 0, 1, 0), (1, 1, 1, 0), (0, 0, 1, 1)]):
    rows += [dict(arm="a", task="t", density_class=0.1, graph_id=g,
                  condition="none", exact=ex_a, hit_cap=cap_a),
             dict(arm="a", task="t", density_class=0.1, graph_id=g,
                  condition="degree", exact=ex_b, hit_cap=cap_b)]
  return pf.with_outcomes(pd.DataFrame(rows))


def test_pairs_keep_every_pair_and_mark_truncation():
  j = pf.pairs(_four_graphs(), "a", "t", "none", "degree", [0.1])
  assert sorted(j.index.get_level_values("graph_id")) == [0, 1, 2, 3]
  g2 = j.xs(2, level="graph_id").iloc[0]
  # The control hit the budget on a text that matched the gold: truncated, not correct.
  assert g2.correct_a == 0 and g2.truncated_a == 1
  assert len(pf.finished(j)) == 2


def test_effect_is_on_the_correct_share_with_the_truncated_change_beside_it():
  j = pf.pairs(_four_graphs(), "a", "t", "none", "degree", [0.1])
  d, lo, hi, broke, fixed, p, n, dt = pf.effect(j)
  assert d == pytest.approx(25.0)      # correct 1/4 -> 2/4
  assert dt == pytest.approx(0.0)      # truncated 1/4 -> 1/4
  assert (broke, fixed, n) == (1, 2, 4)
  assert "truncated +0.0, wrong -25.0" in pf.fmt(pf.effect(j))


def test_cells_keep_a_mostly_truncated_cell_and_flag_it():
  rows = []
  for g in range(10):
    rows.append(dict(arm="qwen3-4b", task="edge_count", density_class=0.1,
                     graph_id=g, condition="none", exact=1, hit_cap=0))
    rows.append(dict(arm="qwen3-4b", task="edge_count", density_class=0.1,
                     graph_id=g, condition="degree", exact=1, hit_cap=int(g < 9)))
  t = apw.cells(pd.DataFrame(rows), {"edge_count/degree": 1.0})
  row = t[t.condition == "degree"].iloc[0]
  assert row.n == 10 and row.baseline == 1.0
  assert row.delta == pytest.approx(-90.0)
  assert row.trunc_b == pytest.approx(0.9) and bool(row.flagged)
```

- [ ] **Step 2: Run them to see them fail**

Run: `PYTHONPATH=. $PY -m pytest -q tests/test_primer_findings.py -p no:cacheprovider`
Expected: the three new tests FAIL (`with_outcomes` missing; `cells` has no `trunc_b`).

- [ ] **Step 3: Change `analyze_primer_window.cells()`.** Add `from graphtalk import outcomes` to the imports, and replace the body of the inner `for c in CONDS:` loop (lines 57–69) with:

```python
                for c in CONDS:
                    b = d[d.condition == c].set_index("graph_id")
                    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
                    if j.empty:
                        continue
                    # R1: every pair is kept; a truncated response is not correct.
                    ca = outcomes.outcome(j.exact_a, j.hit_cap_a) == outcomes.CORRECT
                    cb = outcomes.outcome(j.exact_b, j.hit_cap_b) == outcomes.CORRECT
                    ta, tb = j.hit_cap_a.mean(), j.hit_cap_b.mean()
                    out.append(dict(
                        arm=arm, task=task, density=dens, condition=c,
                        n=len(j),
                        baseline=ca.mean(),
                        delta=100.0 * (cb.mean() - ca.mean()),
                        acc=cb.mean(),
                        trunc_a=ta, trunc_b=tb,
                        flagged=max(ta, tb) >= outcomes.FLAG,
                        bar=bars.get(f"{task}/{c}", float("nan")),
                        bar_none=bars.get(f"{task}/none", float("nan"))))
```

Also change the docstring line 18 area: add one sentence, "Pairs follow rule R1 (graphtalk/outcomes.py): a truncated response counts as not correct and is never dropped; cells whose truncated share reaches 15% are flagged."

- [ ] **Step 4: Change `primer_findings.py`.**

1. Imports: add `from graphtalk import outcomes` after `from graphtalk import scoring`.
2. Module docstring, the "Conventions" paragraph (lines 12–15), becomes: "Conventions: rule R1 (graphtalk/outcomes.py). Two conditions are paired on the shared graph within (arm, task, density) and every pair is kept; a response that hit the budget is truncated, never correct; an effect is the change in the correct share, printed with the change in the truncated share. Answer descriptions (MAE, yes-rate, false alarms, routes, wording) use finished responses only. Exact McNemar; 95% intervals from a bootstrap over graphs, stratified by density."
3. Replace `pairs()` and `pairs_as_error()` (lines 91–105) with:

```python
def with_outcomes(f):
  """Add R1's 0/1 columns: `correct` and `truncated`."""
  f = f.copy()
  o = outcomes.outcome(f.exact, f.hit_cap)
  f["correct"] = (o == outcomes.CORRECT).astype(int)
  f["truncated"] = (o == outcomes.TRUNCATED).astype(int)
  return f


def pairs(f, arm, task, a, b, dens):
  """Every graph that has both conditions, joined as *_a / *_b. Nothing is
  dropped; code that describes answers calls finished() itself."""
  d = f[(f.arm == arm) & (f.task == task) & f.density_class.isin(dens)]
  x = d[d.condition == a].set_index(["density_class", "graph_id"])
  y = d[d.condition == b].set_index(["density_class", "graph_id"])
  return x.join(y, lsuffix="_a", rsuffix="_b", how="inner")


def finished(j):
  """The pairs in which both responses finished within the budget."""
  return j[(j.hit_cap_a == 0) & (j.hit_cap_b == 0)]
```

4. Replace `effect()` and `fmt()` (lines 119–130) with:

```python
def effect(j, col="correct"):
  """Paired effect in points, its interval, broke/fixed, exact McNemar p, n, and
  (for the correct share) the change in the truncated share in points."""
  a, b = j[col + "_a"].astype(bool), j[col + "_b"].astype(bool)
  m = scoring.mcnemar(a.to_numpy(), b.to_numpy())
  d = 100 * (b.mean() - a.mean())
  lo, hi = boot(j, lambda s: 100 * (s[col + "_b"].mean() - s[col + "_a"].mean()))
  dt = (100 * (j.truncated_b.mean() - j.truncated_a.mean())
        if col == "correct" else np.nan)
  return d, lo, hi, m["b"], m["c"], m["p_value"], len(j), dt


def fmt(e):
  d, lo, hi, broke, fixed, p, n, dt = e
  s = f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}] broke {broke} fixed {fixed} p={p:.2g} n={n}"
  if not np.isnan(dt):
    s += f" | truncated {dt:+.1f}, wrong {-d - dt:+.1f}"
  return s
```

5. Every remaining `exact_a` becomes `correct_a` and every `exact_b` becomes `correct_b` (they all come from `pairs()` joins): `sed -i 's/exact_a/correct_a/g; s/exact_b/correct_b/g' scripts/primer_findings.py`. Then check nothing else matched: `grep -n 'correct_[ab]' scripts/primer_findings.py` must show only paired uses.
6. `main()`: right after `f = pd.read_csv(args.frame)` add `f = with_outcomes(f)`.
7. `band_table()`: replace the first print with
   `print(f"[cells] {len(t)} cells; {int(t.flagged.sum())} flagged (truncated share >= 15%); "`
   `      f"{int(t.carries.sum())} answer-carrying, {int((~t.carries).sum())} side")`.
8. `procedure()`: delete the three lines that print "degree vs none, 7 densities, truncation counted as an error" (the `pairs_as_error` call); the preceding "degree vs none, 7 densities" line now is that number.
9. `edge_count()`: the MAE line uses finished pairs. Replace `j = pairs(f, "qwen3-1.7b", "edge_count", "none", "degree", DENS4)` before the MAE print with `j = finished(pairs(f, "qwen3-1.7b", "edge_count", "none", "degree", DENS4))` and change the printed label to `"  qwen3-1.7b mean absolute error, both finished, none "`.
10. `edge_existence()`: in the per-primer loop, the false-alarm pairs are finished pairs: `j = finished(pairs(f.assign(fa=(f.pred == "Yes").astype(int)), arm, "edge_existence", "none", c, DENS4 + DENSHI))`, then `j = j[j.gold_is_yes_a == 0]` as before.
11. `measurement()`: `j = finished(pairs(f, arm, "connected_nodes", "none", c, DENS4))` (F1 and exact compared on finished pairs; on finished rows `correct` equals `exact`); add `"(finished pairs)"` to its first print.
12. `length()`: replace the pivot/masking block and `fit` (lines 521–538) with:

```python
      W = d.pivot_table(index=["density_class", "graph_id"], columns="condition",
                        values="correct")
      T = d.pivot_table(index=["density_class", "graph_id"], columns="condition",
                        values="truncated")
      # R1: a combination where any condition reaches the truncation flag is not
      # interpreted (its correct share moves with the budget, not the primer).
      if (T[["none", "filler"] + PRIMERS].mean() >= outcomes.FLAG).any():
        continue

      def fit(w):
        ds = [100 * (w[c].mean() - w["none"].mean()) for c in PRIMERS]
        slope, icpt = np.polyfit(x, ds, 1)
        fil = 100 * (w["filler"].mean() - w["none"].mean())
        return slope, fil - (icpt + slope * kch["filler"]), np.polyfit(x[:4], ds[:4], 1)[0]
```

   and in its summary print replace `(Table 2 columns with >=100 pairs)` with `(combinations below the truncation flag)`.

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=. $PY -m pytest -q tests/test_primer_findings.py tests/test_outcomes.py -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 6: Smoke-run the script end to end** (catches a column renamed in one place only)

Run: `PYTHONPATH=. $PY scripts/primer_findings.py > $SCRATCH/pf_r1.txt 2>&1; echo exit=$?; grep -c '^\[' $SCRATCH/pf_r1.txt`
(`$SCRATCH` is the session scratchpad.) Expected: `exit=0`, and a tag count at least the old file's (`grep -c '^\[' csv2/raw-trends/primer_findings.txt`). Do not write into `csv2/` yet.

- [ ] **Step 7: Stage** — `git add scripts/primer_findings.py scripts/analyze_primer_window.py tests/test_primer_findings.py`

---

### Task 4: The analyses the ledger still needs

**Files:**
- Modify: `scripts/primer_findings.py` (add `joint`, `rwse_resolution`, `setup_facts`, `solver_bars`; extend `edge_existence`'s `[collapse]` and `figure_data`; call them from `main`)
- Test: `tests/test_primer_findings.py`

**Interfaces:**
- Consumes: `with_outcomes`, `pairs`, `effect`, `fmt`, `finished` from Task 3.
- Produces: tags `[joint]`, `[rwse]`, `[setup]`, `[bars]`; `[collapse]` now covers both plain arms (this is the spec's `[balanced]`); helpers `neighbour_set(s) -> set[int] | None` and `rwse_pairs(prompt, instance_id) -> list[tuple[str, str]]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_primer_findings.py`):

```python
def test_neighbour_set_reads_lists_and_no_nodes():
  assert pf.neighbour_set("15, 39") == {15, 39}
  assert pf.neighbour_set("No nodes") == set()
  assert pf.neighbour_set("") is None
  assert pf.neighbour_set(float("nan")) is None


def test_rwse_pairs_needs_all_forty_sentences():
  s = " ".join(f"Node {i} has return probability 0.2{i % 3} after 2 steps "
               f"and 0.0{i % 2} after 3 steps." for i in range(40))
  assert len(set(pf.rwse_pairs(s, "x"))) == 6
  with pytest.raises(ValueError, match="node_degree/size40/p0.5/7"):
    pf.rwse_pairs(s.split(" Node 39")[0], "node_degree/size40/p0.5/7")
```

- [ ] **Step 2: Run to see them fail** — `PYTHONPATH=. $PY -m pytest -q tests/test_primer_findings.py -p no:cacheprovider` → the two new tests FAIL (`neighbour_set`, `rwse_pairs` missing).

- [ ] **Step 3: Add the helpers and analyses** to `primer_findings.py`, in the "analyses" section after `leakage()`:

```python
def neighbour_set(s):
  """A connected_nodes answer as a set of node ids; None when there is no answer."""
  if not isinstance(s, str) or not s.strip():
    return None
  return set() if "no nodes" in s.lower() else set(map(int, re.findall(r"-?\d+", s)))


def joint(f):
  """Degree and neighbour answers for the same queried node, main sweep (p<=.50).
  J: both correct (a truncated side is not correct). C: both finished and
  parsed, a valid neighbour set, and the stated degree equals its size."""
  cols = ["arm", "condition", "density_class", "graph_id"]
  nd = f[(f.task == "node_degree") & f.density_class.isin(DENS4)]
  cn = f[(f.task == "connected_nodes") & f.density_class.isin(DENS4)]
  j = nd.merge(cn, on=cols, suffixes=("_d", "_n"), validate="one_to_one")
  gold_sets = j.gold_n.map(neighbour_set)
  assert (j.gold_d.astype(int) == gold_sets.map(len)).all(), "tasks query different nodes"
  sets = j.pred_n.map(neighbour_set)
  valid = sets.map(lambda s: s is not None and all(0 <= v < 40 for v in s))
  size = sets.map(lambda s: len(s) if s is not None else -1)
  j["J"] = ((j.correct_d == 1) & (j.correct_n == 1)).astype(int)
  j["C"] = ((j.hit_cap_d == 0) & (j.hit_cap_n == 0) & (j.parsed_d == 1)
            & (j.parsed_n == 1) & valid
            & (pd.to_numeric(j.pred_d, errors="coerce") == size)).astype(int)
  print("[joint] degree and neighbours of the same node, p<=.50: J both correct, "
        "C consistent, % of items; then J against none")
  for arm in ARMS:
    s = j[j.arm == arm]
    print(f"  {arm:17s} " + ", ".join(
        f"{c} J {100 * s[s.condition == c].J.mean():.2f} C {100 * s[s.condition == c].C.mean():.2f}"
        for c in ["none", "filler"] + PRIMERS))
    base = s[s.condition == "none"].set_index(["density_class", "graph_id"])
    for c in ["filler"] + PRIMERS:
      x = base.join(s[s.condition == c].set_index(["density_class", "graph_id"]),
                    lsuffix="_a", rsuffix="_b", how="inner")
      print(f"    J {c:10s} " + fmt(effect(x, "J")))


def rwse_pairs(prompt, instance_id):
  """The (2-step, 3-step) return probabilities the rwse primer prints, one per node."""
  found = re.findall(r"Node (\d+) has return probability (\d+\.\d+) after 2 steps "
                     r"and (\d+\.\d+) after 3 steps", prompt)
  if len(found) != 40:
    raise ValueError(f"{instance_id}: {len(found)} rwse sentences, expected 40")
  return [(a, b) for _, a, b in found]


def rwse_resolution():
  """Distinct printed return-probability pairs per graph against distinct stated
  degrees, from the saved prompts (the text the model saw), by density."""
  deg = re.compile(r"Node (\d+) has degree (\d+)")
  seen = {}
  for path in ("prompts.densfull40.jsonl", "prompts.densfull40hi.jsonl"):
    for line in open(path, encoding="utf-8"):
      r = json.loads(line)
      if r["task"] != "node_degree" or r["condition"] not in ("rwse", "degree"):
        continue
      seen.setdefault(r["instance_id"], {})[r["condition"]] = r["prompt"]
  by = {}
  for iid, p in seen.items():
    pr = rwse_pairs(p["rwse"], iid)
    counts = pd.Series(pr).value_counts()
    degrees = {int(d) for _, d in deg.findall(p["degree"])}
    dens = float(re.search(r"/p([\d.]+)/", iid).group(1))
    by.setdefault(dens, []).append((len(counts), counts.iloc[0] / 40, len(degrees)))
  print("[rwse] per graph, by density: distinct printed (2-step, 3-step) pairs / "
        "modal pair's share of nodes / distinct stated degrees / graphs with fewer "
        "rwse classes than degrees")
  for dens, v in sorted(by.items()):
    a = np.array(v, dtype=float)
    print(f"  p={dens:.2f}: {a[:, 0].mean():.2f} / {100 * a[:, 1].mean():.2f}% / "
          f"{a[:, 2].mean():.2f} / {int((a[:, 0] < a[:, 2]).sum())}/{len(a)}")


def setup_facts(f):
  """The facts the setup section states: constant golds, budgets, truncation."""
  print("[setup] distinct gold values: node_count "
        f"{sorted(f[f.task == 'node_count'].gold.astype(str).unique())}, cycle_check "
        f"{sorted(f[f.task == 'cycle_check'].gold.astype(str).unique())}")
  hi = f.density_class.isin(DENSHI)
  for label, part in (("main sweep", f[~hi]), ("high-density extension", f[hi])):
    cap = part[part.hit_cap == 1].groupby("arm").n_new_tokens.max()
    print(f"  budget reached, {label}: " + ", ".join(f"{a} {int(v)}" for a, v in cap.items()))
  t = f[~hi].groupby(["arm", "task"]).truncated.mean()
  print("  truncated share, main sweep (arm/task): " + ", ".join(
      f"{a}/{k} {100 * v:.1f}%" for (a, k), v in t.items() if v > 0))


def solver_bars(bars):
  """The graph-blind solver's accuracy per (task, primer), from shortcuts_n40_flat.json."""
  print("[bars] graph-blind solver accuracy (%), n=40: task: none, then each primer")
  for task in TASKS4:
    print(f"  {task}: none {100 * bars[f'{task}/none']:.1f}, " + ", ".join(
        f"{c} {100 * bars[f'{task}/{c}']:.1f}" for c in ["filler"] + PRIMERS))
  side = {k: v for k, v in bars.items()
          if k.split("/")[0] in TASKS4 and v < apw.CARRIES}
  top = max(side, key=side.get)
  print(f"  carrying (>= {apw.CARRIES}): " + ", ".join(
      sorted(k for k, v in bars.items() if k.split("/")[0] in TASKS4 and v >= apw.CARRIES))
      + f"; highest other: {top} {100 * side[top]:.1f}")
```

Before relying on `solver_bars`, confirm every `f"{task}/{c}"` key exists: `$PY -c "import json;b=json.load(open('shortcuts_n40_flat.json'));print(sorted(b))"`. If `filler` keys are absent, drop `"filler"` from that list.

- [ ] **Step 4: Extend `[collapse]`** — in `edge_existence()`, replace the block from `s = d[d.arm == "qwen3-1.7b"]` through the `for c in [...]` print loop with (the balanced-accuracy effects loop after it stays):

```python
  e = f[f.task == "edge_existence"]
  for arm in ["qwen3-1.7b", "qwen3-4b"]:
    print(f"[collapse] {arm} edge_existence by density: yes-rate of finished / "
          "balanced accuracy (truncated not correct) / truncated share / median "
          "tokens of finished")
    for c in ["none", "filler", "components", "clustering", "rwse", "degree", "all"]:
      parts = []
      for dens, g in e[(e.arm == arm) & (e.condition == c)].groupby("density_class"):
        fin = g[g.hit_cap == 0]
        pos, neg = g[g.gold_is_yes == 1], g[g.gold_is_yes == 0]
        parts.append(f"{(fin.pred == 'Yes').mean():.2f}/"
                     f"{0.5 * (pos.correct.mean() + neg.correct.mean()):.2f}/"
                     f"{g.truncated.mean():.2f}/{fin.n_new_tokens.median():.0f}")
      print(f"  {c:10s} " + " ".join(parts))
```

and in `figure_data()` replace the `edge_existence_collapse.csv` block with the same definitions, one row per (arm, density, condition), columns `arm, density, condition, yes, bacc, truncated, tokens`, for both plain arms.

- [ ] **Step 5: Call them from `main()`**: after `t = band_table(f, bars)` add `setup_facts(f)` and `solver_bars(bars)`; after `edge_existence(f)` add `joint(f)`; after `clustering_spread()` add `rwse_resolution()`.

- [ ] **Step 6: Run the tests and the script**

Run: `PYTHONPATH=. $PY -m pytest -q tests/test_primer_findings.py -p no:cacheprovider` → all pass.
Run: `PYTHONPATH=. $PY scripts/primer_findings.py > $SCRATCH/pf_r1.txt 2>&1; echo exit=$?; grep -E '^\[(joint|rwse|setup|bars|collapse)\]' $SCRATCH/pf_r1.txt`
Expected: `exit=0`; the five tags print. Spot-check against the ledger (these do not depend on truncation): `[rwse]` p=0.50 is `3.30 / 62.68%`; `[joint]` plain 4B none J `85.75`, degree J `76.50`; `[setup]` budgets 8192 everywhere except `qwen3-1.7b`/`qwen3-4b` 2048 in the extension. A mismatch is a bug in the new code; stop and find it.

- [ ] **Step 7: Stage** — `git add scripts/primer_findings.py tests/test_primer_findings.py`

---

### Task 5: Snapshot what the archived v3 build reads, then regenerate

**Files:**
- Create (copies): `superseded/csv2/raw-trends/{primer_cells,edge_existence_collapse,node_degree_routes}.csv`, `superseded/csv2/raw-trends/primer_findings.txt`
- Modify (regenerated): `csv2/raw-trends/primer_findings.txt`, `csv2/raw-trends/{primer_cells,edge_existence_collapse,node_degree_routes}.csv`

- [ ] **Step 1: Snapshot before anything overwrites them** (the ledger and v3's `NUMBERS.md` cite the current text; v3's build reads the current CSVs):

```bash
mkdir -p superseded/csv2/raw-trends
for f in primer_cells.csv edge_existence_collapse.csv node_degree_routes.csv primer_findings.txt; do
  cp csv2/raw-trends/$f superseded/csv2/raw-trends/$f
done
git add superseded/csv2/raw-trends
```

- [ ] **Step 2: Confirm the frame is still what the runs give** (it is the pipeline's first stage and must not have drifted):

Run: `PYTHONPATH=. $PY scripts/build_raw_frame.py && git status --short csv2/raw-trends/frame.csv`
Expected: the script reports no gold mismatch, and `git status` shows no change to `frame.csv`.

- [ ] **Step 3: Regenerate the output**

Run: `PYTHONPATH=. $PY scripts/primer_findings.py --csv-dir csv2/raw-trends > csv2/raw-trends/primer_findings.txt`
Expected: exit 0. `git diff --stat csv2/raw-trends` shows the txt and three CSVs changed.

- [ ] **Step 4: Stage** — `git add csv2/raw-trends`

---

### Task 6: Checkpoint — compare every ledger claim with the new output (stop for the user)

**Files:**
- Create: `$SCRATCH/ledger_diff.md` (not in the repo)

- [ ] **Step 1:** For each numbered claim in `paper/CLAIMS_LEDGER.md` §3–§11 whose verdict is ✅, ☑️ or ⚠️, find the tag that now prints it in `csv2/raw-trends/primer_findings.txt` and write one row: `# | claim | ledger value | new value [tag] | same / moved / reversed`. "Moved" is a change of 1 point or more; "reversed" is a change of sign or of significance at .05.
- [ ] **Step 2:** List any ledger claim that no tag prints (for example, #37's completion numbers, which came from the independent pipeline). For each, either name the tag that should print it and add that print to `primer_findings.py` (then rerun Task 5 Step 3), or mark it "not carried".
- [ ] **Step 3: Stop.** Show the user every row marked moved or reversed, and every "not carried" claim, and ask how each should be stated. Do not write the results doc until they answer.

---

### Task 7: `docs/results/n40-sweep.md` and the index

**Files:**
- Create: `docs/results/n40-sweep.md`
- Create: `docs/results/README.md`

- [ ] **Step 1: Write the doc.** Header, exactly:

```markdown
# The 40-node sweep

Source: `csv2/raw-trends/primer_findings.txt`

Runs: `runs/{qwen3-1.7b,qwen3-1.7b-think,qwen3-4b,qwen3-4b-think}.{densfull40,densfull40hi}.shard*.jsonl`
(84,000 responses). Reproduce:

    PYTHONPATH=. python scripts/build_raw_frame.py
    PYTHONPATH=. python scripts/primer_findings.py --csv-dir csv2/raw-trends > csv2/raw-trends/primer_findings.txt

Scoring follows rule R1 (`graphtalk/outcomes.py`): every response is correct,
wrong or truncated; effects are paired changes in the share of all responses
that is correct, with the change in the truncated share beside them.
```

Then these sections, in this order, each claim followed by its number(s) in the `value [tag]` form, values copied from the output exactly (the checkpoint's answers decide wording where a number moved):

1. *Setup and measurement* — the corpus size `[extract]`; the two constant tasks `[setup]`; budgets and truncation `[setup]`; the share of `edge_existence` queries that are edges `[goldshare]`; primer lengths `[length]`; `filler` is length-matched and structure-free and names all 40 nodes `[nodecount]`.
2. *Answer leakage* — solver bars and the carrying/side split `[bars]`; what the saved prompts give away `[leak]`.
3. *Answer-carrying primers change the procedure* — plain 4B by density `[flip]`; routes and retrieval accuracy `[route]`; copy errors `[copyerr]`; the node-0–9 difference with its confound stated `[position]`; 1.7B-think counts, reports a conflict and defers `[verify]`, `[trunc]`; plain 1.7B `[plain17]`, `[plaincite]`; baseline bands `[bands]`, `[splithalf]`, `[window]`.
4. *A trade-off per-task accuracy hides* — `edge_count` under `degree` `[edgecount]`; joint correctness falls `[joint]`.
5. *`edge_existence` measures a yes-bias* — hits and false alarms `[fa]`; the plain-1.7B collapse and both arms' balanced accuracy `[collapse]`.
6. *Side information is small and non-specific* — 4B-think null `[null4bt]`; `clustering` at high density `[clusthi]`, `[cluster]`, `[clustproc]`; the 1.7B `clustering` replication and its absence at p≥.65 `[replic]`; `components` as a fixed sentence `[comp]`, `[compproc]`; the bundle `[bundle]`; filler and length `[length]`.
7. *Feature resolution* — RWSE pairs vs degrees `[rwse]`.
8. *Limits* — rerun noise `[rerun]`; detectable effect size `[power]`; extraction `[extract]`; `node_count` answers of 39 `[nodecount]`; F1 vs exact `[f1]`.

No sentence mentions an earlier version, a correction, or a retracted number (R4).

- [ ] **Step 2: Write the index** `docs/results/README.md`:

```markdown
# Results

One document per family of runs. Each states only what its pipeline's committed
output shows; `tests/test_results_docs.py` checks every number it cites.

| Family | Runs | Doc | Pipeline | Output |
|---|---|---|---|---|
| 40-node sweep | `*.densfull40*`, `*.densfull40hi*` | [n40-sweep.md](n40-sweep.md) | `scripts/build_raw_frame.py` → `scripts/primer_findings.py` | `csv2/raw-trends/primer_findings.txt` |
```

(Later phases add a row each.)

- [ ] **Step 3: Run the doc test**

Run: `PYTHONPATH=. $PY -m pytest -q tests/test_results_docs.py -p no:cacheprovider`
Expected: all pass, including `test_every_cited_number_is_in_its_source[n40-sweep.md]`. A failure names the value and tag; fix the doc, not the test.

- [ ] **Step 4: Stage** — `git add docs/results`

---

### Task 8: Move everything else about these runs to `superseded/`

**Files:** moves only (`git mv`), plus `superseded/README.md`.

- [ ] **Step 1: Remove LaTeX build products and the duplicate.** Only LaTeX
  products and `__pycache__` are deleted; other ignored files (the independent
  pipeline's three large results) are kept and moved in Step 2. First confirm the
  duplicate: `cmp "paper/Structural_Primers_Graph_Reasoning_ACL2023 (1).pdf" "paper/Structural_Primers_Graph_Reasoning_ACL2023 (2).pdf" && echo identical` must print `identical`. Then:

```bash
find paper \( -name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o -name '*.fdb_latexmk' \
  -o -name '*.fls' -o -name '*.log' -o -name '*.out' \) -type f -print -delete
find paper -name __pycache__ -type d -prune -print -exec rm -rf {} +
git rm -q "paper/Structural_Primers_Graph_Reasoning_ACL2023 (2).pdf"
git status --short --ignored paper    # what remains untracked or ignored
```

- [ ] **Step 2: Commit-track the untracked records, then move `paper/`**:

```bash
git add paper/compare/Structural_Primers_GraphTalk_ACL.pdf paper/compare/acl.sty \
        paper/compare/acl_natbib.bst \
        paper/compare/independent/Structural_Primers_GraphTalk_ACL.pdf \
        paper/synthesis/synthesis.pdf
git ls-files paper | grep -v -E '^paper/(acl\.sty|acl2023\.sty|acl_natbib\.bst|custom\.bib|README\.md)$' |
while IFS= read -r p; do
  mkdir -p "superseded/$(dirname "$p")"; git mv "$p" "superseded/$p"
done
# Ignored, untracked files left behind (the independent pipeline's large results):
# move them next to their tracked siblings, where the moved .gitignore still covers them.
git ls-files --others --ignored --exclude-standard paper | while IFS= read -r p; do
  mkdir -p "superseded/$(dirname "$p")"; mv "$p" "superseded/$p"
done
find paper -type d -empty -delete
```

Check: `git ls-files paper` lists exactly the five kept files, and
`git ls-files --others paper` lists nothing.

- [ ] **Step 3: Move scripts and their tests**:

```bash
mkdir -p superseded/scripts/candidates superseded/tests
for s in analyze_churn_and_length analyze_effect_drivers analyze_error_taxonomy \
         analyze_nontermination analyze_primer_survival test_vs_controls raw_trends \
         raw_trends_figures analyze_review_checks; do git mv scripts/$s.py superseded/scripts/$s.py; done
git mv scripts/candidates/a_routegap.py superseded/scripts/candidates/a_routegap.py
git mv scripts/candidates/a_transfer.py superseded/scripts/candidates/a_transfer.py
git mv ci_all.py superseded/ci_all.py
cp scripts/analyze_baseline_law.py superseded/scripts/analyze_baseline_law.py
git add superseded/scripts/analyze_baseline_law.py
for t in test_analyze_error_taxonomy test_analyze_nontermination test_analyze_churn_and_length \
         test_test_vs_controls test_analyze_review_checks; do git mv tests/$t.py superseded/tests/$t.py; done
```

- [ ] **Step 4: Move outputs**:

```bash
mkdir -p superseded/csv2/one-offs superseded/analysis/rerun superseded/analysis/tables
git mv csv2/sweep-large-graph superseded/csv2/sweep-large-graph
for f in primer_window serial_slope_diffs truncation_flips; do git mv csv2/one-offs/$f.csv superseded/csv2/one-offs/$f.csv; done
for f in effects effects_capdropped edge_existence_balanced difficulty output_operation behaviour \
         strategy_vs_accuracy serial_position error_shape relevance additivity moderators headroom \
         marker_validation ladder_length_vs_difficulty; do
  git mv csv2/raw-trends/$f.csv superseded/csv2/raw-trends/$f.csv; done
git mv analysis/raw-trends superseded/analysis/raw-trends
for f in $(git ls-files analysis/rerun | grep -E 'densfull40|primer_survival'); do git mv $f superseded/$f; done
for f in $(git ls-files analysis/tables | grep -E 'densfull40|primer_survival'); do git mv $f superseded/$f; done
for f in ci_all.json ci_all_hi.json vs_controls_densfull40.json vs_controls_densfull40hi.json \
         churn_len_densfull40.json error_taxonomy.json review_checks.json; do git mv $f superseded/$f; done
```

Check: `git ls-files analysis/rerun analysis/tables` lists only the degfixdeg, probe100, retrieval, retrieval_threshold, size and published_split files.

- [ ] **Step 5: Move docs**:

```bash
mkdir -p superseded/docs
for d in primer-effects-paper-draft raw-trends-large-graph full-task-density-sweep candidate-analyses \
         paper-claim-audit paper-revision-handoff paper-v2-consolidation paper-v3-review \
         paper-v3-review-fixes; do git mv docs/$d.md superseded/docs/$d.md; done
```

- [ ] **Step 6: Write `superseded/README.md`** — a title, one paragraph ("Everything here was replaced by a current version; nothing here is current. An exact rebuild of anything in this directory as it stood is `git checkout 7037749`. `paper/CLAIMS_LEDGER.md` explains why the paper versions disagreed."), then one table row per moved item or group: `| original path | replaced by | reproduce |`. Group rows by the Step 2–5 lists (e.g. one row for the 16 `analysis/rerun/*.densfull40*.vs_*.txt` files). "Replaced by" is `docs/results/n40-sweep.md` for every 40-node output, analysis and doc. "Reproduce" is the path-fixed command (Task 9) or "`git checkout 7037749`, then …" where the path-fixed run is not exact.

- [ ] **Step 7: Stage** — `git add superseded`

---

### Task 9: Fix every path that pointed at a moved file

**Files:**
- Create: `$SCRATCH/rewrite_paths.py`, `$SCRATCH/check_paths.py` (throwaway, not committed)
- Modify: every tracked text file that references a moved path; `scripts/analyze_baseline_law.py`; the moved files listed in Step 3

- [ ] **Step 1: Write `$SCRATCH/rewrite_paths.py`**:

```python
"""Rewrite references to files `git mv` staged, in tracked text files."""
import pathlib
import re
import subprocess

TEXT = {".md", ".py", ".sh", ".tex", ".toml", ".sbatch", ".cfg", ".gitignore"}


def git(*a):
  return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout


moves = [line.split("\t")[1:] for line in
         git("diff", "--cached", "--name-status", "-M", "HEAD").splitlines()
         if line.startswith("R")]
moves.sort(key=lambda m: -len(m[0]))
files = [p for p in git("ls-files").splitlines()
         if not p.startswith("runs/") and pathlib.Path(p).suffix in TEXT | {""}
         and pathlib.Path(p).is_file()]
changed = 0
for path in files:
  p = pathlib.Path(path)
  # Bytes in, bytes out: keeps each file's own line endings.
  text = p.read_bytes().decode("utf-8", "surrogateescape")
  new = text
  for old, dst in moves:
    new = re.sub(rf"(?<![\w./-]){re.escape(old)}(?![\w/-])", dst, new)
  if new != text:
    p.write_bytes(new.encode("utf-8", "surrogateescape"))
    changed += 1
    print(path)
print(f"{changed} files rewritten over {len(moves)} moves")
```

- [ ] **Step 2: Run it** — `$PY $SCRATCH/rewrite_paths.py`. Read `git diff` for every live (non-`superseded/`) file it changed and confirm each rewrite points somewhere real.

- [ ] **Step 3: Hand fixes the rewrite cannot make.** For each, grep, edit, and note it:
  - Moved scripts that import `analyze_baseline_law` or another moved module through `sys.path` must find the full copy first. In `superseded/scripts/analyze_churn_and_length.py`, `superseded/scripts/analyze_review_checks.py`, `superseded/scripts/candidates/a_routegap.py`, `superseded/scripts/candidates/a_transfer.py`, `superseded/paper/make_figure_f1.py`, `superseded/paper/make_figure_density.py`: find the `sys.path.insert` line (`grep -n 'sys.path' <file>`) and make `superseded/scripts` the first entry, keeping `scripts` after it (the live `analyze_headline_robustness.py` is still imported from there).
  - Moved scripts that locate the repo root with `parents[k]`: `grep -n 'parents\[' $(git ls-files superseded | grep '\.py$')`; add one level for each (for example `superseded/paper/compare/independent/analyze.py`: `parents[3]` → `parents[4]`).
  - Output paths built with f-strings the rewrite cannot see: `grep -n 'ci_all_\|vs_controls_\|churn_len_\|error_taxonomy\|review_checks\|sweep-large-graph\|raw-trends' $(git ls-files superseded | grep -E '\.(py|sh)$')` — every *output* a moved script writes goes under `superseded/`.
  - `superseded/paper/make_v3.sh`: drop the `scripts/primer_findings.py --figure-data` step and read the snapshot `superseded/csv2/raw-trends/*.csv`; its `cd paper` becomes `cd superseded/paper`. Same `cd` fix in `superseded/paper/make_all.sh`.
  - `scripts/build_raw_frame.py` lines 3 and 48 name `raw_trends.py`: reword to say only that `primer_findings.py` reads the frame.
  - Files under `superseded/` cite the *old* output of the live pipeline: `csv2/raw-trends/primer_findings.txt`, `primer_cells.csv`, `edge_existence_collapse.csv`, `node_degree_routes.csv` were regenerated in Task 5, so those references must point at the Task 5 snapshot. In every file under `superseded/` only: `git ls-files superseded | grep -E '\.(md|py|sh|tex)$' | xargs grep -l 'csv2/raw-trends/\(primer_findings.txt\|primer_cells.csv\|edge_existence_collapse.csv\|node_degree_routes.csv\)'`, and in each, prefix those four paths with `superseded/` (skip occurrences already prefixed). Live files keep pointing at the live output.
- [ ] **Step 4: Strip the cut tests from the live `scripts/analyze_baseline_law.py`.** Delete `test_split`, `test_heldout`, `test_instrument`, `test_crossfit`, `test_ceiling`, `ceiling_by_arm`, and `_describe`/`_split_report` if `grep -n '_describe\|_split_report' scripts/analyze_baseline_law.py` shows no other caller. Set `TESTS = {"continuum": test_continuum}` and the `main()` default to `["continuum"]`. Rewrite the module docstring's test list to the continuum test only, plus one line: "The route-split, held-out, instrument, cross-fit and ceiling tests are in `superseded/scripts/analyze_baseline_law.py`." Keep every helper (live scripts import them).
- [ ] **Step 5: Write `$SCRATCH/check_paths.py`**:

```python
"""List references to repo paths that do not exist, per tracked text file."""
import pathlib
import re
import subprocess

EXT = r"(?:md|py|sh|tex|csv|json|jsonl|txt|pdf|png|sty|bib|toml)"
REF = re.compile(rf"(?<![\w/.-])((?:[\w.-]+/)+[\w.() -]*?\.{EXT})(?![\w/])")
files = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                       check=True).stdout.splitlines()
for path in files:
  p = pathlib.Path(path)
  if path.startswith("runs/") or p.suffix not in {".md", ".py", ".sh", ".tex"}:
    continue
  for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
    for ref in REF.findall(line):
      if any(ch in ref for ch in "*{<$") or ref.startswith(("http", "~")):
        continue
      if not (pathlib.Path(ref).exists() or (p.parent / ref).exists()):
        print(f"{path}:{n}: {ref}")
```

- [ ] **Step 6: Run it** — `$PY $SCRATCH/check_paths.py > $SCRATCH/broken.txt; wc -l < $SCRATCH/broken.txt`. Every line under a live path must be fixed. A line under `superseded/` is fixed when the target moved; it may stay only if the target was deleted from git before `7037749` (confirm with `git log --oneline -1 --diff-filter=D -- <target>`), and then gets a note in `superseded/README.md`.
- [ ] **Step 7: Stage the path fixes** before anything regenerates files, so a restore below can never undo them: `git add -A superseded scripts tests docs csv2 analysis paper *.md`.
- [ ] **Step 8: Prove the moved material still reproduces.**
  - `PYTHONPATH=. $PY -m pytest -q superseded/tests -p no:cacheprovider` → all pass.
  - Rerun each moved analysis script from the repo root with its documented arguments (docstring usage lines), writing to its now-`superseded/` default; then `git status --short superseded` must show no content change to the outputs it rewrote (a changed output is a path bug or a non-deterministic script; find which). `superseded/ci_all.py` takes ~20 minutes; run it in the background.
  - `PYTHONPATH=. $PY superseded/paper/compare/independent/analyze.py` → its tracked result CSVs unchanged.
  - Build each archived paper from its directory: `cd superseded/paper && latexmk -pdf -interaction=nonstopmode talk_like_a_graph.v3.tex` (and `talk_like_a_graph.v2.tex`, `talk_like_a_graph.tex`, `short/short.tex`, `synthesis/synthesis.tex`, `compare/independent/paper.tex` with its `TEXINPUTS`); each must compile.
  - Then put the tracked files back to their staged state and remove the build products: `git diff --name-only -- superseded | while IFS= read -r p; do git checkout -- "$p"; done` (restores from the index, which holds the staged fixes and the original PDFs), then the Step 1 `find … -delete` commands with `superseded/paper` in place of `paper`. `git status --short superseded` must then be empty.
- [ ] **Step 9: Full test suite** — `PYTHONPATH=. $PY -m pytest -q --ignore=tests/test_hierarchical_model.py --ignore=tests/test_mixed_models.py -p no:cacheprovider` → all pass except the known `torch` DLL failure.
- [ ] **Step 10: Stage** — `git add -A superseded scripts tests docs csv2 analysis paper *.md`

---

### Task 10: Independent verification until clean, then commit

- [ ] **Step 1: Dispatch a fresh verification agent** (general-purpose, no access to this conversation) with this brief: re-derive every number in `docs/results/n40-sweep.md` from `runs/*.densfull40*.jsonl` and the saved prompts with **its own** code written in the scratchpad (it may import only `graphtalk.scoring.extract_answer`, `graphtalk.scoring.score_one` and `graphtalk.outcomes`), under R1–R2 as stated in the spec; run `$SCRATCH/check_paths.py`; run the full test suite; confirm `git diff --cached --name-status -M 7037749` shows only R, M and A apart from the one duplicate PDF; confirm no live file contains any of `-11.0`, `−11.0`, `eleven held-out`, `+5.9`, `content-free`, `2,048 for plain`, `both thinking arms at ceiling` (`git grep` excluding `superseded/`). Report every discrepancy with file:line and the value it found.
- [ ] **Step 2:** Fix each reported item (ask the user first where the fix changes a stated claim). Then dispatch a **new** agent with the same brief. Repeat until an agent reports nothing.
- [ ] **Step 3: Commit the phase**:

```bash
git commit -q -m "$(cat <<'EOF'
Give the 40-node sweep one pipeline and one results doc

Truncation is its own outcome everywhere (graphtalk/outcomes.py): effects
are paired changes in the correct share, with the truncated share beside
them. primer_findings.py is the only analysis of the 40-node runs; it adds
joint correctness, both arms' balanced accuracy, RWSE resolution, setup
facts and the solver bars. docs/results/n40-sweep.md states what its output
shows, and tests/test_results_docs.py checks every number it cites. Every
other paper version, analysis, output and doc about these runs moved to
superseded/, with the paths that pointed at them fixed.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4:** Report to the user: the commit, the verification rounds and what each found, and the Phase 2 plan as the next step.
