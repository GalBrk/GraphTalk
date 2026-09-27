# Hand validation for `primer-directions.md`

**Status: human-labelled; every check passes.** The user labelled the sheet by
hand, and three independent LLM labellers labelled it blind as well. C1's texts
were relabelled after a fix that also shows mentions of N at the end of a
sentence. The user decided the 8 items where the LLM labellers disagreed (each
row's `note` says which), and the user's own labels agree with the final labels
in the sheet. The scores are in `csv2/raw-trends/directions_validation.txt`, and
what they mean for the findings is in `primer-directions.md` under Validation.
The rules below are the ones used, with the user's C3 rule. A second human
labeller, working on a copy, would give an inter-rater agreement score.

**What this is.** The findings in [`primer-directions.md`](primer-directions.md)
A1, A2 and A3 rest on text detectors (regular expressions) that nobody has checked
by hand. This page lists what to check, where, and what each result changes. A4
and B5–B9 are numeric and need no labels.

**What you do** (about 226 items, roughly an hour):

1. Open `csv2/raw-trends/directions_validation_sheet.csv` in Excel, LibreOffice
   or Google Sheets.
2. For each row, read `question` and `text`, and type the answer in `label`:
   `y` or `n` (C2: `a`, `b`, `c` or `d`); `?` if you cannot tell. `note` is
   optional. Save as CSV, keeping the file name.
3. Run the scorer and paste its output back to Claude (or save it):

       PYTHONPATH=. python scripts/validate_directions.py --score \
           > csv2/raw-trends/directions_validation.txt

Do **not** open `csv2/raw-trends/directions_validation_key.csv`. It holds each
item's model, primer and the detector's verdict. The sheet hides them so your
labels are blind; the scorer joins them back.

In every `text`, the detector's evidence is marked «like this». `…` means the
text is cut there; `|||` separates excerpts from one response. The excerpts come
from the models' own reasoning, so they are often repetitive or wrong; judge only
what the question asks.

## The six checks

| Check | Items | Question (short) | Finding that depends on it | If it fails (under 90% agreement) |
|---|---|---|---|---|
| **C1** | 40 | Does the response itself arrive at its final answer N, apart from stating it? | A1: when a conflict is reported, the answer takes the primer's value, the model's own value, or neither (e.g. plain 4B keeps its own count in 41%) | Report only "primer's value vs not"; drop the own/other split |
| **C2** | 40 | What does the reported conflict compare: `a` a prompt value vs own count, `b` two own results, `c` other/unclear, `d` no conflict | A1/A3 on `edge_count`: conflicts rise from 12–33% to 65–70% under `degree` and end truncated. Is the rise about the stated degrees? | Not pass/fail: the a/b split under `none` vs `degree` becomes the doc's reading |
| **C3** | 40 | Is the marked number the model's candidate answer to the question at that point? | A3: the first candidate comes earlier under `degree` (27% vs 44% of the trace); first = final 92–100% | Drop A3's commitment numbers |
| **C4** | 30 | Does the marked word ("wait", "let me recount", "mistake"…) start a revision of the model's own work? | A3: revision words double under `degree` on `edge_count` | Drop the revision-rate claim |
| **C5** | 45 | Does the text state the marked value as that node's clustering coefficient / return probability? | A2: 4B-T's wrong quotes under `all` take the neighbouring line's value about twice as often as chance | Drop the misread claim |
| **C6** | 31 | Does the text report an actual disagreement between two values, not only check for one? | Every A1 number: this is the `reports_conflict` marker, so far checked only by an LLM labeller (19/20) | A1 is dropped until the marker is fixed |

The scorer reports, per check, how often the detector agrees with your labels,
with an exact 95% interval, and splits it by the group the detector assigned
(C1 own/other, C3/C4 no primer vs `degree`, C5 neighbouring-line/other, C6 task).
C2 is reported as the a/b/c/d counts under `none` and under `degree`.

## How to label, check by check

- **C1.** You see the question, the value given in the prompt, the final answer N,
  and up to six excerpts where the text writes N. Answer `y` if, besides the final
  statement, the text reaches N by its own counting or arithmetic ("that's N
  connections", "S / 2 = N", "I count N"). Answer `n` if N appears only as the
  final answer, as a node id, or as someone else's value. A count written in
  words ("that's six connections") counts as `y`.
- **C2.** Read the conflict. `a`: it sets a number the prompt gives (a degree in the
  list) against the model's own count or sum. `b`: it sets two of the model's own
  results against each other (two sums, a recount). `c`: something else, or you
  cannot tell. `d`: the word is there but no disagreement is reported ("to see if
  there is any conflict").
- **C3.** `y` if the marked number is what the model, at that point, gives as the
  answer to the question shown (even if it later changes it). `n` if it is a
  different quantity: another node's degree, a partial sum, a node id. When the
  marked number is the model reading or quoting the prompt's stated degree for
  the queried node (reciting the degree list), the user's rule applies: `y` if
  the model then compares it with its own count and decides; `n` if it only
  recites it.
- **C4.** `y` if the marked word begins going back over the model's own work
  (recounting, re-checking a value, correcting a step). `n` if it is filler or
  refers to something else ("wait for", "it's not a mistake to").
- **C5.** `y` if the text attributes the marked value to that node and that
  quantity, whatever the value is. The check is whether the detector read the
  quote right, not whether the model read the primer right.
- **C6.** `y` if the text states that two values disagree. `n` if it only wonders
  or checks whether they do.

## After labelling

The scorer prints PASS or FAIL per check, and for a FAIL, the change it implies
(the "If it fails" column above). Paste the output to Claude to update
`primer-directions.md`: which findings stand, which go, and how C2 reads.

A second labeller makes the result stronger. Copy the sheet, label it
independently, and ask for the agreement between the two sheets; the scorer reads
one sheet at a time.

## Also labelled only by an LLM (optional)

`scripts/response_patterns.py`'s text markers (`reports_conflict`, `degree_sum`,
`enumerate`, `restates_line`, `lookup_any`/`lookup_both`, `id_range` and the
`names_*` vocabulary markers) were validated on
`csv2/raw-trends/response_pattern_validation.csv` by an LLM labeller only. Only
`reports_conflict` feeds the primer-directions findings (check C6 above). The rest
matter if the `response_patterns` shifts are ever reported. Label that file's
`label` column (`1`/`0`) the same way to confirm them.
