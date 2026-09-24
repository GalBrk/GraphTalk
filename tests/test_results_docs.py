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
CITE = re.compile(r"(?<![\w.])([+\-−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)%?) "
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
  return [(v.replace("−", "-"), t) for v, t in CITE.findall(text)]


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
  assert check(DOC.replace("-12.0", "−12.0"), lambda s: SRC) == []


def test_a_prefix_of_a_printed_number_is_not_found():
  assert check(DOC.replace("44%", "27.2"), lambda s: SRC) == [
      "27.2 not found under [route]"]
  assert check(DOC.replace("-12.0", "-12"), lambda s: SRC) == [
      "-12 not found under [flip]"]
