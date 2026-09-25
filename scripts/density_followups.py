"""The density follow-up runs, scored in one pass: the output that
docs/results/density-followups.md cites.

Every block is printed under a [tag] by scripts/score_density_sweep.py (rule
R1: a truncated response is its own outcome, never correct and never dropped;
flagged levels are left out of pooled effects and trends) or by
scripts/analyze_rq3_leads.py (the clustering forensics). The setup blocks
([ddsetup], [fixsetup], [d40setup]) rebuild each run set's prompts with
scripts/build_size_sweep.py, assert that every rebuilt gold equals the run's,
and print edges and prompt characters per level. Run sets are loaded one at a
time.

  PYTHONPATH=. python scripts/density_followups.py > csv2/density-followups/density_followups.txt
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rq3_leads as rq3  # noqa: E402
import build_size_sweep as bss  # noqa: E402
import score_density_sweep as sds  # noqa: E402

LO = [0.1, 0.2, 0.35, 0.5]
HI = [0.65, 0.75, 0.85]
POOLS = [("all levels", None), ("p<=.50", LO), ("p>=.65", HI)]
FIXDEG = [(20, 0.421), (40, 0.205), (80, 0.101), (160, 0.05),
          (20, 0.842), (40, 0.41), (80, 0.203), (160, 0.101)]
D40 = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75]  # tasks are pooled apart, never together


def runs(*names):
  return [f"runs/{name}.shard*.jsonl" for name in names]


PLAIN = runs("qwen3-1.7b.degdens40", "qwen3-1.7b.degceil", "qwen3-1.7b.degdens40hi",
             "qwen3-1.7b.degdensfill")
THINK = runs("qwen3-1.7b-think.degdensthink", "qwen3-1.7b-think.degdensfillT")


def block(tag, title, patterns, group="density", **kw):
  sds.report(sds.summarize(sds.load(patterns), group), tag, title + " -- runs: " + ", ".join(patterns), **kw)


def check_golds(records, patterns):
  """Every run row must have a prompt record with the same instance, condition
  and gold; stop otherwise."""
  built = {(r["instance_id"], r["condition"]): r["gold"] for r in records}
  for r in sds.load(patterns):
    key = (r["instance_id"], r["condition"])
    if built.get(key) != r["gold"]:
      raise SystemExit(f"{patterns}: {key} has gold {r['gold']!r}, the prompts "
                       f"{built.get(key)!r}")


def rebuilt(pattern_sets, cells, conditions):
  """The run sets' prompts, rebuilt with the builder that wrote them and
  checked gold for gold against every run set given."""
  records = []
  for size, density in cells:
    records += bss.build([size], 400, conditions, bss.DEFAULT_SEED, densities=[density],
                         tasks=("node_degree",))
  for patterns in pattern_sets:
    check_golds(records, patterns)
  return records


def setup():
  sds.report_prompts(
      rebuilt([PLAIN, THINK], [(40, d) for d in LO + HI],
              ("none", "components", "clustering", "degree", "filler")),
      "density", "ddsetup", "prompt design, 400 graphs per density at n=40 (the graphs of "
      "the plain and thinking runs; rebuilt golds equal both arms' golds)")
  sds.report_prompts(
      rebuilt([runs("qwen3-1.7b.degfixdeg"), runs("qwen3-8b.degfixdeg")], FIXDEG,
              ("none", "clustering")),
      "cell", "fixsetup", "prompt design at fixed mean degree, 400 graphs per cell "
      "(rebuilt golds equal the 1.7B and 8B runs' golds)")
  with open("prompts.density40.jsonl", encoding="utf-8") as fh:
    d40 = [json.loads(line) for line in fh if line.strip()]
  check_golds(d40, runs("qwen3-1.7b.density40"))
  sds.report_prompts(d40, "task", "d40setup", "prompt design of the density40 pilot "
                     "(prompts.density40.jsonl; its golds equal the runs' golds)")


def main():
  setup()
  block("ddplain", "qwen3-1.7b node_degree, plain: seven densities, vs none",
        PLAIN, pools=POOLS, trend=True, continuum=True, headroom=LO)
  block("ddplainfill", "qwen3-1.7b node_degree, plain: vs the filler control",
        PLAIN, control="filler", pools=POOLS)
  block("ddthink", "qwen3-1.7b-think node_degree: seven densities, vs none",
        THINK, pools=POOLS, trend=True, continuum=True)
  block("ddthinkfill", "qwen3-1.7b-think node_degree: vs the filler control",
        THINK, control="filler", pools=POOLS)
  sds.report_versus(sds.summarize(sds.load(PLAIN), "density"),
                    sds.summarize(sds.load(THINK), "density"), "ddgap",
                    "qwen3-1.7b node_degree: thinking minus plain on the same graphs and "
                    "condition -- runs: " + ", ".join(THINK) + " minus " + ", ".join(PLAIN))
  block("ddrep", "qwen3-1.7b node_degree, fresh seeds (p<=.50), vs none",
        runs("qwen3-1.7b.degdensrep"))
  block("fixdeg17", "qwen3-1.7b node_degree at fixed mean degree, per (size, density) cell",
        runs("qwen3-1.7b.degfixdeg"), group="cell", blocks=True, gold_means=True)
  block("fixdeg8", "qwen3-8b node_degree at fixed mean degree, per (size, density) cell",
        runs("qwen3-8b.degfixdeg"), group="cell", blocks=True, gold_means=True)
  block("d40", "qwen3-1.7b pilot: node_degree and connected_nodes at six densities, vs none",
        runs("qwen3-1.7b.density40"), group="task",
        pools=[(task, [(task, d) for d in D40]) for task in ("node_degree", "connected_nodes")])
  rows, graphs = rq3.load()
  for name, test in rq3.TESTS.items():
    print(f"[rq{name}] {rq3.TITLES[name]}\n" + json.dumps(rq3.tidy(test(rows, graphs)), indent=1))


if __name__ == "__main__":
  main()
