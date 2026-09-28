"""Every number on a tagged line of the paper must be printed in its tag's output block.

A line of the paper's LaTeX that ends in "% [tag] [tag2]" is checked: each number in
the text before the comment must equal, at the paper's precision, a number in one of
the named blocks of outputs/*/*.txt (a block runs from a line that starts with
"[tag]" to the next such line). A percentage may also be printed as a fraction
(43% for 0.43); a table row whose comment ends in "(%)" shows every number on it
as a percentage. Whole numbers that define the design rather than report a result
(the 40 nodes, the 15% flag, band edges) are in DESIGN and not checked.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "paper" / "paper.tex"
DESIGN = {0, 5, 9, 10, 15, 19, 20, 25, 40, 50, 75, 90, 100, 160, 400}
DENSITIES = {".10", ".20", ".35", ".50", ".65", ".75", ".85"}
TAG_LINE = re.compile(r"^\[([\w-]+)\]")
NUM = re.compile(r"(?<![\w.])\d*\.?\d+")


def blocks(paths):
  out, tag = {}, None
  for path in paths:
    tag = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
      m = TAG_LINE.match(line)
      if m:
        tag = m.group(1)
      if tag:
        out.setdefault(tag, []).append(re.sub(r"(?<=\d),(?=\d{3})", "", line))
  return out


def clean(text):
  """The prose of a line without what is not a reported number: references, model
  names, hypothesis labels, thresholds after < or >, LaTeX spacing."""
  text = re.sub(r"\\(ref|citep|citet|citealp|label)\{[^}]*\}", " ", text)
  text = re.sub(r"Qwen3-(1\.7B|4B|8B/14B|8B)|Gemma~?4|H[1-4]|\\S\d|n=40|\\dagger", " ", text)
  text = re.sub(r"(?<=\d)(\{,\}|,)(?=\d{3})", "", text).replace("\\,", "")
  return re.sub(r"(<|>|\\leq|\\geq)\s*\$?\s*[-+]?\.?\d+", " ", text)


def problems(tex, found):
  out = []
  for i, line in enumerate(tex.splitlines(), 1):
    if "% [" not in line:
      continue
    body, comment = line.split("% [", 1)
    tags = re.findall(r"\[([\w-]+)\]", "[" + comment)
    out += [f"line {i}: [{t}] is not a tag in outputs/" for t in tags if t not in found]
    pool = [float(n) for t in tags for l in found.get(t, []) for n in NUM.findall(l)]
    prose = clean(body)
    for m in NUM.finditer(prose):
      text, x = m.group(), float(m.group())
      if text in DENSITIES or ("." not in text and x in DESIGN):
        continue
      tol = 0.5 * 10 ** -(len(text.split(".")[1]) if "." in text else 0) + 1e-9
      percent = "(%)" in comment or prose[m.end():m.end() + 3].startswith(("\\%", "$\\%"))
      if not any(abs(x - y) <= tol or (percent and abs(x - 100 * y) <= tol) for y in pool):
        out.append(f"line {i}: {text} not found under {tags}")
  return out


@pytest.mark.skipif(not PAPER.exists(), reason="no paper at paper/paper/paper.tex")
def test_every_tagged_number_in_the_paper_is_in_its_output():
  found = blocks(sorted((ROOT / "outputs").glob("*/*.txt")))
  assert problems(PAPER.read_text(encoding="utf-8"), found) == []


SRC = {"flip": ["[flip] p=0.50 delta -12.0 base 99.0"], "recover": ["[recover] 0.43 n=700"]}


def test_reports_a_number_the_output_does_not_print():
  assert problems("costs $12.0$ points % [flip]\n", SRC) == []
  assert problems("costs $13.0$ points % [flip]\n", SRC) == ["line 1: 13.0 not found under ['flip']"]


def test_accepts_a_percentage_printed_as_a_fraction_and_skips_design_numbers():
  assert problems("accuracy is $43\\%$ on $40$ nodes % [recover]\n", SRC) == []
  assert problems("Plain Qwen3-1.7B scores $43\\%$ % [recover]\n", SRC) == []
  assert problems("q $<.05$ at $p=.50$ % [flip]\n", SRC) == []
  assert problems("none & 43.0 & 12.0\\\\ % [recover] [flip] (%)\n", SRC) == []
  assert problems("none & 43.0\\\\ % [recover]\n", SRC) == ["line 1: 43.0 not found under ['recover']"]


def test_reports_an_unknown_tag():
  assert problems("$12.0$ % [flips]\n", SRC)[0] == "line 1: [flips] is not a tag in outputs/"
