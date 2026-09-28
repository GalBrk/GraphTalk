"""Build the cycle and truncation tables and check the main matrix.

All values come from committed 40-node main-sweep data. No model generation is
needed. Run from anywhere: ``python paper/build_tables.py`` or ``--check``.
The latter checks the generated tables and every keyed numeric cell in the
paper's primary matrix against the saved response frame.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper"
FRAME = ROOT / "outputs/n40-sweep/frame.csv"
CYCLES = ROOT / "outputs/n40-sweep/check_cycle_claims.txt"
ARMS = [("qwen3-1.7b", "1.7P", "1.7B"),
        ("qwen3-1.7b-think", "1.7T", "1.7B-T"),
        ("qwen3-4b", "4P", "4B"),
        ("qwen3-4b-think", "4T", "4B-T")]
TASKS = [("node_degree", "Degree"), ("connected_nodes", "Neighbors"),
         ("edge_count", "Edge count"), ("edge_existence", "Edge exists"),
         ("node_count", "Node count"), ("cycle_check", "Cycle")]
CONDITIONS = ("degree", "clustering", "rwse", "all", "components", "filler")
ARM_SHORT = {a: b for a, b, _ in ARMS}
TASK_SHORT = dict(TASKS)


def response_shares():
    counts = defaultdict(lambda: [0, 0, 0])  # total, correct, truncated
    for row in csv.DictReader(FRAME.open(newline="", encoding="utf-8")):
        if float(row["density_class"]) > 0.50:
            continue
        key = row["arm"], row["task"], row["condition"]
        c = counts[key]
        c[0] += 1
        capped = int(row["hit_cap"])
        c[2] += capped
        c[1] += int(float(row["exact"]) == 1 and not capped)
    for key, (total, _, _) in counts.items():
        if total != 400:
            raise ValueError(f"Expected 400 paired responses at {key}, found {total}")
    if len(counts) != 4 * 6 * 7:
        raise ValueError(f"Incomplete main sweep: {len(counts)} cells")
    return counts


def replace_body(path: Path, rows: str, *, check: bool) -> None:
    old = path.read_text(encoding="utf-8")
    start = old.index("\\midrule") + len("\\midrule")
    end = old.rindex("\\bottomrule")
    fresh = old[:start] + "\n" + rows + "\n" + old[end:]
    if check:
        if fresh != old:
            raise AssertionError(f"Stale table: {path}; run python paper/build_tables.py")
    else:
        path.write_text(fresh, encoding="utf-8", newline="\n")


def make_truncation(counts, *, check: bool) -> None:
    lines = []
    for ti, (task, label) in enumerate(TASKS):
        if ti:
            lines.append("\\midrule" if ti == 4 else "\\addlinespace[2pt]")
        for ai, (arm, short, _) in enumerate(ARMS):
            none = counts[arm, task, "none"][2] / 4
            values = [f"{none:.2f}"]
            for cond in CONDITIONS:
                delta = counts[arm, task, cond][2] / 4 - none
                values.append(f"${delta:+.1f}$")
            prefix = f"{label} & {short}" if ai == 0 else f" & {short}"
            lines.append(prefix + " & " + " & ".join(values)
                         + r"\\ % [main] [trunc]")
    replace_body(HERE / "main_truncation.tex", "\n".join(lines), check=check)


ROW = re.compile(
    r"^\s*(1\.7B(?:-T)?|4B(?:-T)?)\s+"
    r"(none|filler|components|clustering|rwse|degree|all)\s+"
    r"n=(\d+)\s+([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*"
    r"([\d.]+)\s*\(\s*([\d.]+)\)"
)


def cycle_rows():
    source = CYCLES.read_text(encoding="utf-8")
    answer = source.split("[ccanswer]", 1)[1].split("[ccinvent]", 1)[0]
    rows = {}
    for line in answer.splitlines():
        m = ROW.match(line)
        if m:
            arm, cond, n, *values = m.groups()
            rows[arm, cond] = int(n), values
    if len(rows) != 28:
        raise ValueError(f"Expected 28 cycle rows, found {len(rows)}")

    test = source.split("[cctest]", 1)[1].split("[ccthinktest]", 1)[0]
    bold = {"real": set(), "invented": set()}
    for kind, heading in (("real", "rests on a real cycle:"),
                          ("invented", "rests on an invented cycle:")):
        section = test.split("  " + heading, 1)[1].split("\n  ", 1)[1]
        section = section.split("\n  rests on", 1)[0]
        for line in section.splitlines():
            arm_match = re.match(r"\s*(1\.7B(?:-T)?|4B(?:-T)?)\s+", line)
            if not arm_match:
                continue
            arm = arm_match.group(1)
            for segment in line[arm_match.end():].split("|"):
                if "*, n=" in segment:
                    cond = segment.strip().split()[0]
                    bold[kind].add((arm, cond))

    lines = []
    order = ("none", "degree", "clustering", "rwse", "all", "components", "filler")
    for ai, (_, short, label) in enumerate(ARMS):
        if ai:
            lines.append("\\addlinespace[2pt]")
        for ci, cond in enumerate(order):
            n, values = rows[label, cond]
            real, invented, not_cycle, no_cycle, edge = values
            if (label, cond) in bold["real"]:
                real = rf"\textbf{{{real}}}"
            if (label, cond) in bold["invented"]:
                invented = rf"\textbf{{{invented}}}"
            prefix = f"{short} &" if ci == 0 else " &"
            shown = "RWSE" if cond == "rwse" else cond
            lines.append(f"{prefix} {shown} & {n} & {real} & {invented} & "
                         f"{not_cycle} & {no_cycle} & {edge}"
                         + r"\\ % [ccanswer] [cctest]")
    return "\n".join(lines)


def check_primary_matrix(counts) -> None:
    tex = (HERE / "paper.tex").read_text(encoding="utf-8")
    block = tex.split("\\begin{table*}", 1)[1].split("\\label{tab:matrix}", 1)[0]
    task = None
    seen = set()
    for line in block.splitlines():
        if "% [main] [bars]" not in line:
            continue
        row = re.sub(r"\\cellcolor\{[^}]+\}", "", line.split("%", 1)[0])
        cells = [c.strip() for c in row.rsplit(r"\\", 1)[0].split("&")]
        if cells[0]:
            task = next((t for t, label in TASKS if label == cells[0]), None)
        arm = next((a for a, short, _ in ARMS if short == cells[1]), None)
        if task is None or arm is None or len(cells) != 9:
            raise AssertionError(f"Unrecognized primary matrix row: {line}")
        key = arm, task
        seen.add(key)
        expected = [100 * counts[arm, task, "none"][1] / 400]
        expected.extend(100 * (counts[arm, task, c][1]
                        - counts[arm, task, "none"][1]) / 400 for c in CONDITIONS)
        # The matrix orders degree, clustering, RWSE, all, components, filler.
        actual = []
        for cell in cells[2:]:
            m = re.search(r"[-+]?\d+(?:\.\d+)?", cell)
            if not m:
                raise AssertionError(f"No number in {cell!r} for {key}")
            actual.append(float(m.group()))
        for cond, got, want in zip(("none", *CONDITIONS), actual, expected):
            if abs(got - want) > (0.006 if cond == "none" else 0.051):
                raise AssertionError(f"{key} {cond}: paper {got}, saved responses {want}")
    if len(seen) != 16:
        raise AssertionError(f"Expected 16 keyed matrix rows, found {len(seen)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    counts = response_shares()
    make_truncation(counts, check=args.check)
    replace_body(HERE / "table_cycles.tex", cycle_rows(), check=args.check)
    check_primary_matrix(counts)
    print("Verified 16 primary rows and regenerated/verified two appendix tables")


if __name__ == "__main__":
    main()
