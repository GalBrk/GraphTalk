"""Q1-Q7 over `csv2/raw-trends/frame.csv` -- the independent rebuild of the
40-node density sweep from raw generations.

One script with `--question`, not seven one-off files; the repo already carries
~40 `analyze_*.py` and does not need more. Every question writes one CSV into
csv2/raw-trends/ and prints the same table.

  PYTHONPATH=. python scripts/raw_trends.py --question all

Questions:
  effects      Q1/Q2  which primer helps/hurts, by arm x task x density, with the
                      fix/break decomposition and both baselines (none, filler)
  difficulty   Q3     what makes a task hard: between-task, output-operation
                      (graph and queried node held fixed), within-task, calibration
  behaviour    Q5     response length and solution strategy by condition
  mechanism    Q4     serial position (with its control), yes-bias, error shape
  composition  Q6     relevance matching and additivity of `all` vs its parts
  moderators   Q7     primer x thinking interaction, full p=0.1..0.85 curve
  markers             stratified sample for hand-validating the regex markers

Statistical conventions, all forced by what the data turned out to be:
  * exact match is the outcome for every task; set-F1 is reported for
    `connected_nodes` only, as a secondary column, because F1 hides most of the
    spread there (94% F1 vs 72% exact in one cell).
  * every effect carries BOTH baselines: vs `none` (net) and vs `filler`
    (content, length-matched). `filler` is itself a large effect in places.
  * `hit_cap` rows are scored wrong AND the analysis is repeated with them
    dropped; length statistics use non-capped rows only, because a capped
    response is truncated, not long.
  * bootstrap CIs resample `graph_id`, not rows: one graph serves all six tasks.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from graphtalk import scoring

FRAME = "csv2/raw-trends/frame.csv"
OUTDIR = "csv2/raw-trends"

TASKS = ["node_count", "cycle_check", "edge_existence", "node_degree",
         "connected_nodes", "edge_count"]
CONDS = ["none", "filler", "degree", "clustering", "rwse", "components", "all"]
PRIMERS = [c for c in CONDS if c != "none"]
ARMS = ["qwen3-1.7b", "qwen3-1.7b-think", "qwen3-4b", "qwen3-4b-think"]
DENS = ["0.1", "0.2", "0.35", "0.5", "0.65", "0.75", "0.85"]
# The four levels every task was run at; .65/.75/.85 cover two tasks only, so
# pooling them in would make a cross-task summary mean different things per row.
DENS4 = DENS[:4]
# Gold is the same for every graph at n=40, so a shift on these need not be
# graph reading, and a ratio against a ceiling-bound part-sum is uninformative.
CONST_GOLD = {"node_count", "cycle_check"}

# Tasks whose gold is a single constant at n=40 -- `node_count` is always 40 and
# `cycle_check` is always "Yes". A deficit from 100% there measures distraction,
# not graph reasoning, and they are labelled as such everywhere they appear.
CONSTANT_GOLD = ("node_count", "cycle_check")

# Which primer feature answers which task, for Q6's relevance test.
MATCHED = {
    "edge_count": "degree",        # sum(deg)/2
    "node_degree": "degree",       # stated verbatim
    "connected_nodes": "components",
    "edge_existence": "clustering",  # triadic closure
}


def load(path=FRAME):
  df = pd.read_csv(path, low_memory=False)
  df["density_class"] = df["density_class"].astype(str)
  return df


# --------------------------------------------------------------------------
# paired effect machinery
# --------------------------------------------------------------------------

def _paired(sub, cond, base):
  """Aligned (base, cond) outcome vectors over the graphs both were run on."""
  a = sub[sub.condition == base].set_index("index")["exact"]
  b = sub[sub.condition == cond].set_index("index")["exact"]
  common = a.index.intersection(b.index)
  return a.loc[common].to_numpy(), b.loc[common].to_numpy()


def _boot_ci(base_v, cond_v, n_boot=2000, seed=0):
  """Percentile CI on the paired mean difference.

  Resamples pairs. Within a single (arm, task, density) cell each pair is a
  distinct graph already, so pair-resampling *is* graph-clustered here; the
  cross-task pooling in `moderators` is where clustering would otherwise bite.
  """
  if len(base_v) == 0:
    return float("nan"), float("nan")
  rng = np.random.default_rng(seed)
  d = cond_v - base_v
  idx = rng.integers(0, len(d), size=(n_boot, len(d)))
  means = d[idx].mean(axis=1)
  return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bh(pvals):
  """Benjamini-Hochberg q-values, order preserved."""
  p = np.asarray(pvals, dtype=float)
  ok = ~np.isnan(p)
  q = np.full_like(p, np.nan)
  if ok.sum() == 0:
    return q
  pv = p[ok]
  n = len(pv)
  order = np.argsort(pv)
  ranked = pv[order] * n / (np.arange(n) + 1)
  ranked = np.minimum.accumulate(ranked[::-1])[::-1]
  out = np.empty(n)
  out[order] = np.clip(ranked, 0, 1)
  q[ok] = out
  return q


def wilson(k, n, z=1.96):
  if n == 0:
    return float("nan"), float("nan")
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return c - h, c + h


# --------------------------------------------------------------------------
# Q1/Q2 -- effects
# --------------------------------------------------------------------------

def q_effects(df, drop_capped=False):
  """Per (arm, task, density, condition): accuracy, both baselines, fix/break."""
  work = df[df.hit_cap == 0] if drop_capped else df
  rows = []
  for arm in ARMS:
    for task in TASKS:
      for dens in DENS:
        sub = work[(work.arm == arm) & (work.task == task)
                   & (work.density_class == dens)]
        if sub.empty:
          continue
        n_none = (sub.condition == "none").sum()
        for cond in CONDS:
          cell = sub[sub.condition == cond]
          if cell.empty:
            continue
          acc = cell["exact"].mean()
          lo, hi = wilson(cell["exact"].sum(), len(cell))
          rec = {
              "arm": arm, "task": task, "density": dens, "condition": cond,
              "n": len(cell), "acc": acc, "acc_lo": lo, "acc_hi": hi,
              "f1": cell["f1"].mean(),
              "hit_cap_rate": df[(df.arm == arm) & (df.task == task)
                                 & (df.density_class == dens)
                                 & (df.condition == cond)]["hit_cap"].mean(),
              "parsed_rate": cell["parsed"].mean(),
              "constant_gold": int(task in CONSTANT_GOLD),
          }
          for base, tag in (("none", "vs_none"), ("filler", "vs_filler")):
            bv, cv = _paired(sub, cond, base)
            if cond == base or len(bv) == 0:
              rec.update({tag + "_delta": 0.0 if cond == base else float("nan"),
                          tag + "_fixed": 0, tag + "_broke": 0,
                          tag + "_p": float("nan"),
                          tag + "_lo": float("nan"), tag + "_hi": float("nan")})
              continue
            mc = scoring.mcnemar(bv, cv)
            lo_d, hi_d = _boot_ci(bv, cv)
            rec.update({
                tag + "_delta": float(cv.mean() - bv.mean()),
                # mcnemar's c == treatment right where control wrong == fixed
                tag + "_fixed": mc["c"], tag + "_broke": mc["b"],
                tag + "_churn": mc["discordant"], tag + "_p": mc["p_value"],
                tag + "_lo": lo_d, tag + "_hi": hi_d,
            })
          rows.append(rec)
  out = pd.DataFrame(rows)
  # BH within each (arm, task) family -- the six tasks are the paper's families.
  for tag in ("vs_none", "vs_filler"):
    out[tag + "_q"] = np.nan
    for (arm, task), g in out.groupby(["arm", "task"]):
      mask = (out.arm == arm) & (out.task == task)
      out.loc[mask, tag + "_q"] = bh(out.loc[mask, tag + "_p"].to_numpy())
  return out


def q_balanced(df):
  """`edge_existence` only: accuracy conflates skill with a moving class prior."""
  rows = []
  ee = df[df.task == "edge_existence"]
  for arm in ARMS:
    for dens in DENS:
      for cond in CONDS:
        cell = ee[(ee.arm == arm) & (ee.density_class == dens)
                  & (ee.condition == cond)]
        if cell.empty:
          continue
        gold_yes = cell["gold_is_yes"] == 1
        pred_yes = cell["pred"].astype(str).str.strip().str.lower().str.startswith("y")
        rp = (cell["exact"][gold_yes]).mean() if gold_yes.any() else float("nan")
        rn = (cell["exact"][~gold_yes]).mean() if (~gold_yes).any() else float("nan")
        rows.append({
            "arm": arm, "density": dens, "condition": cond, "n": len(cell),
            "acc": cell["exact"].mean(),
            "balanced_acc": np.nanmean([rp, rn]),
            "recall_yes": rp, "recall_no": rn,
            "pred_yes_rate": pred_yes.mean(),
            "gold_yes_rate": gold_yes.mean(),
            "majority_baseline": max(gold_yes.mean(), 1 - gold_yes.mean()),
        })
  return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Q3 -- difficulty
# --------------------------------------------------------------------------

def q_difficulty(df):
  rows = []
  for arm in ARMS:
    for task in TASKS:
      for dens in DENS:
        cell = df[(df.arm == arm) & (df.task == task)
                  & (df.density_class == dens) & (df.condition == "none")]
        if cell.empty:
          continue
        rec = {
            "arm": arm, "task": task, "density": dens, "n": len(cell),
            "acc": cell["exact"].mean(),
            "constant_gold": int(task in CONSTANT_GOLD),
            "hit_cap_rate": cell["hit_cap"].mean(),
            "mean_edges": cell["n_edges"].mean(),
            "encoding_chars": cell["encoding_chars"].mean(),
            "gold_magnitude": pd.to_numeric(cell["gold_magnitude"],
                                            errors="coerce").mean(),
            "mean_signed_error": pd.to_numeric(cell["signed_error"],
                                               errors="coerce").mean(),
            "mean_abs_error": pd.to_numeric(cell["abs_error"],
                                            errors="coerce").mean(),
        }
        # Within-task: does the per-item feature predict correctness?
        for feat in ("target_degree", "n_edges", "pair_common_neighbors"):
          v = pd.to_numeric(cell[feat], errors="coerce")
          if v.notna().sum() > 10 and v.nunique() > 2 and cell["exact"].nunique() > 1:
            rho, p = stats.spearmanr(v, cell["exact"], nan_policy="omit")
            rec["rho_" + feat] = rho
            rec["p_" + feat] = p
        rows.append(rec)
  return pd.DataFrame(rows)


def q_output_operation(df):
  """Same graph, same queried node, different required output.

  `node_degree` (count), `connected_nodes` (reproduce the set) and
  `edge_existence` (boolean) all query the identical node of the identical graph,
  so the gap between them is the cost of the output operation with topology and
  target held fixed.
  """
  rows = []
  three = ["node_degree", "connected_nodes", "edge_existence"]
  for arm in ARMS:
    for dens in DENS:
      for cond in CONDS:
        rec = {"arm": arm, "density": dens, "condition": cond}
        ok = True
        for task in three:
          cell = df[(df.arm == arm) & (df.task == task)
                    & (df.density_class == dens) & (df.condition == cond)]
          if cell.empty:
            ok = False
            break
          rec["acc_" + task] = cell["exact"].mean()
          rec["n_" + task] = len(cell)
        if not ok:
          continue
        rec["count_minus_list"] = rec["acc_node_degree"] - rec["acc_connected_nodes"]
        rows.append(rec)
  return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Q5 -- behaviour
# --------------------------------------------------------------------------

def q_behaviour(df):
  rows = []
  for arm in ARMS:
    for task in TASKS:
      for dens in DENS:
        for cond in CONDS:
          cell = df[(df.arm == arm) & (df.task == task)
                    & (df.density_class == dens) & (df.condition == cond)]
          if cell.empty:
            continue
          free = cell[cell.hit_cap == 0]      # length is censored by the cap
          tok = pd.to_numeric(free["n_new_tokens"], errors="coerce")
          rows.append({
              "arm": arm, "task": task, "density": dens, "condition": cond,
              "n": len(cell), "n_uncapped": len(free),
              "hit_cap_rate": cell["hit_cap"].mean(),
              # suppressed where the cap censors too much to read a length
              "median_tokens": (float(tok.median())
                                if cell["hit_cap"].mean() <= 0.2 else float("nan")),
              "p90_tokens": (float(tok.quantile(0.9))
                             if cell["hit_cap"].mean() <= 0.2 else float("nan")),
              "median_node_mentions": float(free["n_node_mentions"].median())
              if len(free) else float("nan"),
              "uses_degree_sum": cell["uses_degree_sum"].mean(),
              "narrates_neighbors": cell["narrates_neighbors"].mean(),
              "enumerates_nodes": cell["enumerates_nodes"].mean(),
              "acc": cell["exact"].mean(),
              "parsed_rate": cell["parsed"].mean(),
          })
  return pd.DataFrame(rows)


def q_strategy_accuracy(df):
  """Q5(c): among primer-present responses, is the switched strategy the wrong one?

  Observational: strategy is chosen by the model, not assigned, so this is an
  association and is reported as one -- never as a controlled mediation claim.
  """
  rows = []
  for arm in ARMS:
    for task, marker in (("edge_count", "uses_degree_sum"),
                         ("node_degree", "narrates_neighbors"),
                         ("connected_nodes", "narrates_neighbors")):
      for dens in DENS:
        for cond in CONDS:
          cell = df[(df.arm == arm) & (df.task == task)
                    & (df.density_class == dens) & (df.condition == cond)]
          if len(cell) < 20 or cell[marker].nunique() < 2:
            continue
          on = cell[cell[marker] == 1]
          off = cell[cell[marker] == 0]
          rows.append({
              "arm": arm, "task": task, "density": dens, "condition": cond,
              "marker": marker, "share_switched": cell[marker].mean(),
              "n_switched": len(on), "n_not": len(off),
              "acc_switched": on["exact"].mean(),
              "acc_not_switched": off["exact"].mean(),
              "gap": on["exact"].mean() - off["exact"].mean(),
          })
  return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Q4 -- mechanism
# --------------------------------------------------------------------------

def q_serial_position(df):
  """Does accuracy depend on WHERE in the 40-line primer the target sits?

  The primer lists nodes in id order, so the target's id is its line number. That
  also confounds position with id magnitude, which is why `none` and `filler` --
  conditions with no per-node primer content to retrieve from -- are reported
  beside `degree`. A slope under `degree` and a flat curve under `none` is
  retrieval position; a slope everywhere is node-id difficulty and the finding
  is dropped.

  Two statistics per cell, because the buckets alone mislead. The quartile
  columns show the shape; `slope_pts` is an OLS fit of correctness on line
  number, rescaled to points from the first line to the last, which uses all
  700 items rather than quartiles of ~175 and comes with a bootstrap interval.
  Read `slope_lo`/`slope_hi`: several bucket gradients that look clean have
  intervals straddling zero.

  The id-magnitude confound is also testable directly, and `print_serial`
  reports it: in an Erdos-Renyi graph a node's id carries no information about
  its degree, so if corr(target_id, target_degree) is ~0 then "later nodes are
  harder nodes" is not available as an explanation.
  """
  rows = []
  buckets = [(0, 9), (10, 19), (20, 29), (30, 39)]
  rng = np.random.default_rng(20260906)
  for arm in ARMS:
    for task in ("node_degree", "connected_nodes"):
      for cond in CONDS:
        cell = df[(df.arm == arm) & (df.task == task) & (df.condition == cond)]
        cell = cell[pd.to_numeric(cell["target_id"], errors="coerce").notna()]
        # A capped generation is a wrong answer for a reason unrelated to where
        # the node sits, so it is dropped rather than scored against position.
        cell = cell[cell["hit_cap"] == 0]
        if len(cell) < 100:
          continue
        tid = pd.to_numeric(cell["target_id"], errors="coerce")
        rec = {"arm": arm, "task": task, "condition": cond, "n": len(cell)}
        for lo, hi in buckets:
          m = (tid >= lo) & (tid <= hi)
          rec["acc_%d_%d" % (lo, hi)] = cell["exact"][m].mean() if m.any() else np.nan
          rec["n_%d_%d" % (lo, hi)] = int(m.sum())
        rho, p = stats.spearmanr(tid, cell["exact"], nan_policy="omit")
        rec["rho_position"] = rho
        rec["p_position"] = p
        rec["first10_minus_last10"] = rec["acc_0_9"] - rec["acc_30_39"]
        rec["slope_pts"], rec["slope_lo"], rec["slope_hi"] = _id_slope(
            tid.to_numpy(float), cell["exact"].to_numpy(float), rng)
        rows.append(rec)
  return pd.DataFrame(rows)


def print_id_confound(df):
  """Is a node's position in the primer confounded with how hard that node is?

  Position and node id are the same variable here, so the whole serial-position
  finding collapses if later-numbered nodes are intrinsically harder. In an
  Erdos-Renyi graph they should not be: ids are assigned before edges are drawn.
  This prints the check rather than asserting it, because a non-zero value would
  not be a bug -- it would mean the finding has to be dropped.
  """
  nd = df[(df.task == "node_degree")]
  tid = pd.to_numeric(nd["target_id"], errors="coerce")
  deg = pd.to_numeric(nd["target_degree"], errors="coerce")
  clu = pd.to_numeric(nd["target_clustering"], errors="coerce")
  ok = tid.notna() & deg.notna()
  print("  id/difficulty confound over %d items:" % int(ok.sum()))
  print("    corr(target_id, target_degree)     = %+.4f" % tid[ok].corr(deg[ok]))
  print("    corr(target_id, target_clustering) = %+.4f"
        % tid[ok].corr(clu[ok]))


def _id_slope(tid, exact, rng, boots=1000):
  """OLS slope of correctness on line number, in points from line 0 to line 39.

  Returns (slope, lo, hi) with a percentile bootstrap interval. A cell with no
  variation in correctness has no slope to report, so it yields NaN rather than
  a spurious zero.
  """
  if len(tid) < 50 or exact.std() == 0:
    return np.nan, np.nan, np.nan
  span = 39.0
  point = np.polyfit(tid, exact, 1)[0] * span * 100
  boot = []
  for _ in range(boots):
    i = rng.integers(0, len(tid), len(tid))
    if exact[i].std() == 0:
      continue
    boot.append(np.polyfit(tid[i], exact[i], 1)[0] * span * 100)
  if not boot:
    return point, np.nan, np.nan
  lo, hi = np.percentile(boot, [2.5, 97.5])
  return point, lo, hi


def q_error_shape(df):
  """M1/M2: is the error an arithmetic slip or a wild miss?"""
  rows = []
  for arm in ARMS:
    for task in ("edge_count", "node_degree"):
      for dens in DENS:
        for cond in CONDS:
          cell = df[(df.arm == arm) & (df.task == task)
                    & (df.density_class == dens) & (df.condition == cond)]
          if cell.empty:
            continue
          err = cell[cell.exact == 0]
          ae = pd.to_numeric(err["abs_error"], errors="coerce").dropna()
          se = pd.to_numeric(err["signed_error"], errors="coerce").dropna()
          gold = pd.to_numeric(err["gold_magnitude"], errors="coerce").dropna()
          rows.append({
              "arm": arm, "task": task, "density": dens, "condition": cond,
              "n": len(cell), "n_wrong": len(err),
              "median_abs_error": float(ae.median()) if len(ae) else np.nan,
              "median_signed_error": float(se.median()) if len(se) else np.nan,
              "frac_undercount": float((se < 0).mean()) if len(se) else np.nan,
              "median_rel_error": (float((ae / gold).median())
                                   if len(ae) and len(gold) == len(ae) else np.nan),
          })
  return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Q6 -- composition
# --------------------------------------------------------------------------

def q_composition(eff):
  """Relevance matching and additivity, both read off the effects table."""
  rel, add = [], []
  for arm in ARMS:
    for task in TASKS:
      for dens in DENS:
        cell = eff[(eff.arm == arm) & (eff.task == task) & (eff.density == dens)]
        if cell.empty:
          continue
        d = dict(zip(cell.condition, cell.vs_filler_delta))
        dn = dict(zip(cell.condition, cell.vs_none_delta))
        for cond in PRIMERS:
          if cond not in d:
            continue
          rel.append({
              "arm": arm, "task": task, "density": dens, "condition": cond,
              "matched": int(MATCHED.get(task) == cond),
              "delta_vs_filler": d[cond], "delta_vs_none": dn.get(cond, np.nan),
          })
        parts = [dn.get(c, np.nan) for c in ("degree", "clustering", "rwse")]
        if not np.isnan(parts).any() and "all" in dn:
          total = float(np.sum(parts))
          biggest = max(parts, key=abs)
          add.append({
              "arm": arm, "task": task, "density": dens,
              "d_degree": parts[0], "d_clustering": parts[1], "d_rwse": parts[2],
              "sum_parts": total, "d_all": dn["all"],
              "gap": dn["all"] - total,
              "ratio": dn["all"] / total if abs(total) > 0.02 else np.nan,
              # The rival account to "the bundle is a diluted sum" is "the
              # bundle is whichever single primer does the most" -- so carry
              # the largest part and the ratio against it too.
              "max_part": biggest,
              "ratio_max": (dn["all"] / biggest
                            if abs(biggest) > 0.02 else np.nan),
              # When one part carries nearly all of the sum the two accounts
              # predict the same number, so only cells below this share can
              # tell them apart.
              "max_share": (abs(biggest) / abs(total)
                            if abs(total) > 0.02 else np.nan),
          })
  return pd.DataFrame(rel), pd.DataFrame(add)


def print_additivity(add, boots=2000):
  """Is the bundle a shrunk sum of its parts, or just its largest part?

  Both ratios are reported with a bootstrap interval on the median. The
  discriminating cells are the ones where no single part dominates: when
  `max_share` is near 1 the sum and the largest part are the same number and
  the comparison is vacuous.
  """
  rng = np.random.default_rng(20260906)
  a = add[add.density.isin(DENS4) & ~add.task.isin(CONST_GOLD)]
  a = a[(100 * a.sum_parts).abs() >= 4.0]

  def med(v):
    v = np.asarray(v.dropna(), dtype=float)
    if len(v) < 5:
      return (np.nan,) * 3
    b = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(boots)]
    return np.median(v), np.percentile(b, 2.5), np.percentile(b, 97.5)

  same = (np.sign(a.d_all) == np.sign(a.sum_parts)) & \
         (a.d_all.abs() < a.sum_parts.abs())
  flip = np.sign(a.d_all) != np.sign(a.sum_parts)
  print("  additivity over %d cells: %d sub-additive, %d sign-flipped"
        % (len(a), int(same.sum()), int(flip.sum())))
  for label, sub in (("all cells", a),
                     ("parts share the effect (max < 75% of sum)",
                      a[a.max_share < 0.75])):
    print("  %s  n=%d" % (label, len(sub)))
    print("    all / sum(parts)   %5.2f [%.2f, %.2f]" % med(sub.ratio))
    print("    all / largest part %5.2f [%.2f, %.2f]" % med(sub.ratio_max))
  # Dilution predicts one shrink factor; report each side rather than assume it.
  for label, m in (("parts help", a.sum_parts > 0), ("parts hurt", a.sum_parts < 0)):
    x, y = a.sum_parts[m].to_numpy(), a.d_all[m].to_numpy()
    if len(x) < 5:
      continue
    k = (x * y).sum() / (x * x).sum()
    bs = []
    for _ in range(boots):
      i = rng.integers(0, len(x), len(x))
      bs.append((x[i] * y[i]).sum() / (x[i] * x[i]).sum())
    print("    slope, %s (n=%d): %.2f [%.2f, %.2f]"
          % (label, len(x), k, np.percentile(bs, 2.5), np.percentile(bs, 97.5)))


# --------------------------------------------------------------------------
# Q7 -- moderators
# --------------------------------------------------------------------------

def q_moderators(eff):
  """Primer x thinking interaction, and the 7-level density curve."""
  rows = []
  for size in ("1.7b", "4b"):
    base, think = "qwen3-" + size, "qwen3-" + size + "-think"
    for task in TASKS:
      for dens in DENS:
        for cond in PRIMERS:
          a = eff[(eff.arm == base) & (eff.task == task)
                  & (eff.density == dens) & (eff.condition == cond)]
          b = eff[(eff.arm == think) & (eff.task == task)
                  & (eff.density == dens) & (eff.condition == cond)]
          if a.empty or b.empty:
            continue
          a, b = a.iloc[0], b.iloc[0]
          rows.append({
              "model_size": size, "task": task, "density": dens,
              "condition": cond,
              "delta_nonthink": a.vs_filler_delta,
              "delta_think": b.vs_filler_delta,
              "interaction": b.vs_filler_delta - a.vs_filler_delta,
              "acc_none_nonthink": a.acc if a.condition == "none" else np.nan,
              "cap_nonthink": a.hit_cap_rate, "cap_think": b.hit_cap_rate,
          })
  return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# marker validation sample
# --------------------------------------------------------------------------

def q_markers(df, n_per=3, seed=0):
  """Stratified sample to hand-label; no Q5 claim ships before this is checked."""
  rng = np.random.default_rng(seed)
  picks = []
  for arm in ("qwen3-1.7b", "qwen3-4b"):
    for task in ("edge_count", "node_degree", "connected_nodes", "edge_existence"):
      for cond in ("none", "degree", "all"):
        cell = df[(df.arm == arm) & (df.task == task) & (df.condition == cond)]
        if cell.empty:
          continue
        take = cell.sample(min(n_per, len(cell)),
                           random_state=int(rng.integers(1 << 30)))
        picks.append(take[["arm", "task", "condition", "density_class",
                           "instance_id", "uses_degree_sum", "narrates_neighbors",
                           "enumerates_nodes", "n_node_mentions", "exact"]])
  out = pd.concat(picks, ignore_index=True)
  out["hand_uses_degree_sum"] = ""
  out["hand_narrates_neighbors"] = ""
  return out


# --------------------------------------------------------------------------

def save(frame, name):
  os.makedirs(OUTDIR, exist_ok=True)
  path = os.path.join(OUTDIR, name)
  frame.to_csv(path, index=False)
  print("  -> %s (%d rows)" % (path, len(frame)))
  return path


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--frame", default=FRAME)
  ap.add_argument("--question", default="all",
                  choices=["all", "effects", "difficulty", "behaviour",
                           "mechanism", "composition", "moderators", "markers"])
  args = ap.parse_args()

  df = load(args.frame)
  print("loaded %d rows, %d arms, %d graphs"
        % (len(df), df.arm.nunique(), df.graph_id.nunique()))
  want = args.question

  eff = None
  if want in ("all", "effects", "composition", "moderators"):
    print("\n[Q1/Q2] effects")
    eff = q_effects(df)
    save(eff, "effects.csv")
    save(q_effects(df, drop_capped=True), "effects_capdropped.csv")
    save(q_balanced(df), "edge_existence_balanced.csv")

  if want in ("all", "difficulty"):
    print("\n[Q3] difficulty")
    save(q_difficulty(df), "difficulty.csv")
    save(q_output_operation(df), "output_operation.csv")

  if want in ("all", "behaviour"):
    print("\n[Q5] behaviour")
    save(q_behaviour(df), "behaviour.csv")
    save(q_strategy_accuracy(df), "strategy_vs_accuracy.csv")

  if want in ("all", "mechanism"):
    print("\n[Q4] mechanism")
    print_id_confound(df)
    save(q_serial_position(df), "serial_position.csv")
    save(q_error_shape(df), "error_shape.csv")

  if want in ("all", "composition"):
    print("\n[Q6] composition")
    rel, add = q_composition(eff)
    print_additivity(add)
    save(rel, "relevance.csv")
    save(add, "additivity.csv")

  if want in ("all", "moderators"):
    print("\n[Q7] moderators")
    save(q_moderators(eff), "moderators.csv")

  if want in ("all", "markers"):
    print("\n[markers] hand-validation sample")
    save(q_markers(df), "marker_validation.csv")

  print("\ndone")


if __name__ == "__main__":
  main()
