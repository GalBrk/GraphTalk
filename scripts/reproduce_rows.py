"""Check that generation still reproduces the committed rows, on a small subset.

Two steps around one GPU job (cluster/README.md, "Check that generation still
reproduces"). CPU only; neither step needs torch.

  subset   picks prompts from the main prompt file, spread over every task,
           condition and density, whose committed responses for one model
           finished below a token cap without reaching the budget, and writes
           them as a prompt file for cluster/sweep.sbatch (GRAPHTALK_PROMPTS).
           The file must go outside data/runs/.
  compare  scores a regenerated responses file and the committed rows for the
           same keys exactly as the analysis does (graphtalk.scoring's
           extractor and scorer, graphtalk.outcomes' rule R1), and reports, per
           task, how often the response text, the extracted answer and the
           outcome agree, then lists every row where they do not.

Committed rows are read the way scripts/build_raw_frame.py reads them: the
`<model>.<run set>.shard*.jsonl` files, then `<model>.<run set>.jsonl`, for the
run sets densfull40 and densfull40hi, the first row seen for a key winning.
Only rows that finished are picked, so any budget at or above their length
regenerates them unchanged; generate at the budget `subset` prints (the cap
the committed rows reach) so a row that diverges and runs long is capped where
the committed rows were.

  PYTHONPATH=. python scripts/reproduce_rows.py subset --model qwen3-1.7b \\
      --rows 48 --max-tokens 2000 --out "$SCRATCH/repro48.jsonl"
  PYTHONPATH=. python scripts/reproduce_rows.py compare --model qwen3-1.7b \\
      --regenerated "$SCRATCH/runs/qwen3-1.7b.repro.jsonl" --subset "$SCRATCH/repro48.jsonl"
"""

import argparse
import collections
import glob
import json
import os
import statistics
import sys

from graphtalk import models
from graphtalk import outcomes
from graphtalk import scoring

PROMPTS = "data/prompts/prompts.densfull40.jsonl"
RUNS_DIR = "data/runs"
# The run sets scripts/build_raw_frame.py reads (its TAGS), in its order.
RUN_SETS = ("densfull40", "densfull40hi")


def key_of(record):
  """The pairing key every run file and run_sweep.py's resume use."""
  return (record["instance_id"], record["condition"], record["style"])


def density_of(instance_id):
  """`p0.35` from `<task>/size<n>/p<density>/<index>`."""
  return instance_id.split("/")[2]


def committed_paths(model, run_sets=RUN_SETS, runs_dir=RUNS_DIR):
  """The committed files of one arm, in the order build_raw_frame.py reads them."""
  paths = []
  for run_set in run_sets:
    paths += sorted(glob.glob(os.path.join(runs_dir, f"{model}.{run_set}.shard*.jsonl")))
    paths += sorted(glob.glob(os.path.join(runs_dir, f"{model}.{run_set}.jsonl")))
  return paths


def load_rows(paths):
  """Rows by key; the first row seen for a key wins, as in build_raw_frame.py."""
  rows = {}
  for path in paths:
    with open(path, encoding="utf-8") as handle:
      for line in handle:
        if line.strip():
          record = json.loads(line)
          rows.setdefault(key_of(record), record)
  return rows


def load_lines(path):
  """(raw line, parsed record) for every non-empty line, raw text kept so a
  subset is byte-identical to the prompt file it came from."""
  with open(path, encoding="utf-8", newline="") as handle:
    return [(line, json.loads(line)) for line in handle if line.strip()]


def finished_below(row, max_tokens):
  """True when a committed row ended by itself within `max_tokens`."""
  return (row is not None and not row.get("hit_cap")
          and row.get("n_new_tokens") is not None
          and row["n_new_tokens"] <= max_tokens)


def _in_order(values):
  return list(dict.fromkeys(values))


def pick(records, committed, rows, max_tokens):
  """Indices into `records` of `rows` prompts, spread over tasks, conditions
  and densities, each with a committed row that finished within `max_tokens`.

  Greedy and deterministic: each slot takes the (task, condition, density)
  whose task is least used so far, then whose (task, density) cell is, then
  density, then condition, and within it the earliest eligible prompt of the
  file. When every combination has an eligible prompt, every task, condition
  and density is therefore covered once `rows` reaches the largest of their
  counts, and every (task, density) cell once it reaches their product.
  """
  tasks = _in_order(r["task"] for r in records)
  conditions = _in_order(r["condition"] for r in records)
  densities = _in_order(density_of(r["instance_id"]) for r in records)
  pools = collections.defaultdict(collections.deque)
  for i, r in enumerate(records):
    if finished_below(committed.get(key_of(r)), max_tokens):
      pools[(r["task"], r["condition"], density_of(r["instance_id"]))].append(i)
  eligible = sum(len(p) for p in pools.values())
  if eligible < rows:
    raise ValueError(f"only {eligible} prompts have a committed row that finished "
                     f"within {max_tokens} tokens; asked for {rows}")

  used = collections.Counter()
  order = {tcd: n for n, tcd in enumerate((t, c, d) for t in tasks for c in conditions
                                          for d in densities)}
  chosen = []
  for _ in range(rows):
    triple = min((tcd for tcd, pool in pools.items() if pool),
                 key=lambda tcd: (used["task", tcd[0]],
                                  used["cell", tcd[0], tcd[2]],
                                  used["density", tcd[2]],
                                  used["condition", tcd[1]],
                                  used["triple", tcd],
                                  order[tcd]))
    t, c, d = triple
    chosen.append(pools[triple].popleft())
    for k in (("task", t), ("cell", t, d), ("density", d), ("condition", c),
              ("triple", triple)):
      used[k] += 1
  return sorted(chosen)


def outcome_of(row):
  """(extracted answer, outcome) exactly as the frame and the analyses score it."""
  pred = scoring.extract_answer(row["response"] or "", row["task"])
  exact = scoring.score_one(pred, row["gold"], row["task"])["exact"]
  return pred, str(outcomes.outcome(exact, bool(row.get("hit_cap"))))


def compare(regenerated, committed):
  """Per-row agreement of regenerated rows with the committed row of each key.

  Raises on a regenerated row with no committed counterpart, or whose gold or
  model differs from it: those are a wrong prompt file or a wrong model, not a
  reproduction result.
  """
  results = []
  for g in regenerated:
    c = committed.get(key_of(g))
    if c is None:
      raise ValueError(f"no committed row for {key_of(g)}")
    if str(c["gold"]) != str(g["gold"]) or c["model"] != g["model"]:
      raise ValueError(f"{key_of(g)}: committed gold/model {c['gold']!r}/{c['model']} "
                       f"!= regenerated {g['gold']!r}/{g['model']}")
    pred_c, out_c = outcome_of(c)
    pred_g, out_g = outcome_of(g)
    text_c, text_g = c["response"] or "", g["response"] or ""
    results.append({
        "task": c["task"], "instance_id": c["instance_id"],
        "condition": c["condition"],
        "same_text": text_c == text_g,
        "same_answer": pred_c == pred_g,
        "same_outcome": out_c == out_g,
        "prefix": len(os.path.commonprefix([text_c, text_g])),
        "answer": (pred_c, pred_g), "outcome": (out_c, out_g),
        "tokens": (c.get("n_new_tokens"), g.get("n_new_tokens")),
        "think": ("<think>" in text_c, "<think>" in text_g),
        "overflow": bool(g.get("overflow")),
    })
  return results


def _share(k, n):
  return f"{k}/{n} ({100.0 * k / n:.1f}%)" if n else "0/0"


def report_compare(results, missing, sources, broken=0):
  n = len(results)
  print(f"regenerated rows {n}, compared against {sources}")
  if broken:
    print(f"  unparseable lines skipped: {broken}")
  for field, label in (("same_text", "exact text"), ("same_answer", "same extracted answer"),
                       ("same_outcome", "same outcome")):
    print(f"  {label:22s} {_share(sum(r[field] for r in results), n)}")
  print(f"  {'task':16s} {'n':>4s} {'text':>5s} {'answer':>7s} {'outcome':>8s}")
  by_task = collections.defaultdict(list)
  for r in results:
    by_task[r["task"]].append(r)
  for task in sorted(by_task):
    rs = by_task[task]
    print(f"  {task:16s} {len(rs):4d} {sum(r['same_text'] for r in rs):5d} "
          f"{sum(r['same_answer'] for r in rs):7d} {sum(r['same_outcome'] for r in rs):8d}")
  prefixes = [r["prefix"] for r in results if not r["same_text"]]
  if prefixes:
    print(f"  common prefix of differing texts, chars: min {min(prefixes)}, "
          f"median {statistics.median(prefixes):g}, max {max(prefixes)}")
  print(f"  <think> in committed / regenerated text: "
        f"{sum(r['think'][0] for r in results)} / {sum(r['think'][1] for r in results)}")
  print(f"  regenerated rows reaching the budget: "
        f"{sum(r['outcome'][1] == outcomes.TRUNCATED for r in results)}; "
        f"not generated (overflow): {sum(r['overflow'] for r in results)}")
  if missing is not None:
    print(f"  subset rows with no regenerated row: {len(missing)}")
    for key in missing:
      print(f"    missing {key[0]} {key[1]}")
  differing = [r for r in results if not (r["same_text"] and r["same_outcome"])]
  if differing:
    print("  rows that differ (outcome committed -> regenerated, answer, tokens, "
          "common prefix):")
  for r in differing:
    kind = "OUTCOME" if not r["same_outcome"] else "ANSWER" if not r["same_answer"] else "text"
    print(f"    {kind:7s} {r['instance_id']} {r['condition']}: "
          f"{r['outcome'][0]} -> {r['outcome'][1]}, "
          f"{r['answer'][0]!r} -> {r['answer'][1]!r}, "
          f"{r['tokens'][0]} -> {r['tokens'][1]} tokens, prefix {r['prefix']}")


def _inside(path, directory):
  path, directory = os.path.abspath(path), os.path.abspath(directory)
  return os.path.commonpath([path, directory]) == directory


def run_subset(args):
  if _inside(args.out, args.runs_dir):
    sys.exit(f"--out {args.out} is inside {args.runs_dir}/: a prompt subset "
             f"belongs outside every checkout's runs directory")
  paths = committed_paths(args.model, (args.run_set,), args.runs_dir)
  if not paths:
    sys.exit(f"no committed files for {args.model}.{args.run_set} in {args.runs_dir}/")
  committed = load_rows(paths)
  lines = load_lines(args.prompts)
  records = [r for _, r in lines]
  try:
    chosen = pick(records, committed, args.rows, args.max_tokens)
  except ValueError as error:
    sys.exit(str(error))
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  with open(args.out, "w", encoding="utf-8", newline="") as handle:
    handle.writelines(lines[i][0] for i in chosen)

  picked = [records[i] for i in chosen]
  rows = [committed[key_of(r)] for r in picked]
  caps = sorted({r["n_new_tokens"] for r in committed.values() if r.get("hit_cap")})
  budget = models.budget(models.MODELS[args.model], "zero_shot")
  eligible = sum(finished_below(committed.get(key_of(r)), args.max_tokens) for r in records)
  print(f"wrote {len(chosen)} prompts to {args.out}")
  print(f"  from {args.prompts}; committed rows: {len(paths)} files "
        f"{args.runs_dir}/{args.model}.{args.run_set}.*")
  print(f"  eligible: {eligible} of {len(records)} prompts finished within "
        f"{args.max_tokens} tokens")
  for label, values, every in (
      ("tasks", [r["task"] for r in picked], [r["task"] for r in records]),
      ("conditions", [r["condition"] for r in picked], [r["condition"] for r in records]),
      ("densities", [density_of(r["instance_id"]) for r in picked],
       [density_of(r["instance_id"]) for r in records])):
    counts = collections.Counter(values)
    print(f"  {label:10s} " + ", ".join(f"{k} {counts[k]}" for k in _in_order(every)))
    absent = [k for k in _in_order(every) if not counts[k]]
    if absent:
      print(f"  WARNING: no {label} {', '.join(absent)} in the subset; raise --rows "
            f"or --max-tokens")
  tokens = [r["n_new_tokens"] for r in rows]
  print(f"  committed tokens: {sum(tokens)} total, max {max(tokens)}")
  print(f"  committed rows reach the budget at: {caps or 'no row reached it'}; "
        f"models.budget({args.model}): {budget}")
  if caps and caps != [budget]:
    print(f"  WARNING: the committed cap differs from models.budget; generate with "
          f"GRAPHTALK_MAX_NEW_TOKENS={caps[-1]}")
  print(f"  generate with GRAPHTALK_PROMPTS={args.out} "
        f"GRAPHTALK_MAX_NEW_TOKENS={caps[-1] if caps else budget}")


def run_compare(args):
  patterns = args.committed or committed_paths(args.model, RUN_SETS, args.runs_dir)
  paths = []
  for pattern in patterns:
    paths += sorted(glob.glob(pattern)) or [pattern]
  committed = load_rows(paths)
  regenerated, broken = [], 0
  with open(args.regenerated, encoding="utf-8") as handle:
    for line in handle:
      if line.strip():
        try:
          regenerated.append(json.loads(line))
        except json.JSONDecodeError:
          broken += 1    # a preempted job's half-written line, as run_sweep.py skips it
  seen = {}
  for g in regenerated:
    if key_of(g) in seen and seen[key_of(g)] != g:
      sys.exit(f"{args.regenerated}: two different rows for {key_of(g)}")
    seen[key_of(g)] = g
  if any(g["model"] != args.model for g in seen.values()):
    sys.exit(f"{args.regenerated} holds rows of another model than {args.model}")
  try:
    results = compare(list(seen.values()), committed)
  except ValueError as error:
    sys.exit(str(error))
  missing = None
  if args.subset:
    missing = [key_of(r) for _, r in load_lines(args.subset) if key_of(r) not in seen]
  report_compare(results, missing, f"{len(paths)} committed files of {args.model}", broken)


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  common = argparse.ArgumentParser(add_help=False)
  common.add_argument("--model", required=True, choices=sorted(models.MODELS))
  common.add_argument("--runs-dir", default=RUNS_DIR,
                      help=f"where the committed runs are (default {RUNS_DIR})")
  sub = parser.add_subparsers(dest="command", required=True)

  s = sub.add_parser("subset", parents=[common],
                     help="write a prompt subset whose committed rows finished")
  s.add_argument("--rows", type=int, required=True, help="how many prompts to pick")
  s.add_argument("--max-tokens", type=int, required=True,
                 help="pick only prompts whose committed response finished within "
                      "this many new tokens (bounds the GPU time)")
  s.add_argument("--out", required=True, help="the subset prompt file; not under data/runs/")
  s.add_argument("--prompts", default=PROMPTS, help=f"prompt file (default {PROMPTS})")
  s.add_argument("--run-set", default=RUN_SETS[0], choices=RUN_SETS,
                 help="committed run set whose rows decide eligibility (default "
                      f"{RUN_SETS[0]}); it must match --prompts")

  c = sub.add_parser("compare", parents=[common],
                     help="compare regenerated rows with the committed ones")
  c.add_argument("--regenerated", required=True, help="the regenerated responses JSONL")
  c.add_argument("--committed", nargs="+", default=None,
                 help="committed run files or globs (default: the files "
                      "build_raw_frame.py reads for --model)")
  c.add_argument("--subset", default=None,
                 help="the subset prompt file, to list rows not regenerated yet")

  args = parser.parse_args()
  if args.command == "subset":
    run_subset(args)
  else:
    run_compare(args)


if __name__ == "__main__":
  main()
