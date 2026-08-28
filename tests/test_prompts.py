"""Prompt assembly and answer scoring.

The scoring half carries most of the weight here. Extraction sits between the
model and every number the study reports, and it fails silently: an extractor
that misses a correct answer records a wrong one, so a bug looks like a result.
The cases below are the response shapes models actually produce, including the
two that caused real defects -- a queried node's own id being read as its
neighbour list, and an Oxford comma truncating that list one element early.
"""

import networkx as nx
import pytest

from graphtalk import graphqa
from graphtalk import primers
from graphtalk import prompts
from graphtalk import scoring

QUESTION = "Q: Is there a cycle in this graph?\nA: "

GRAPH = graphqa.canonical(nx.path_graph(5))


# --- prompt assembly ------------------------------------------------------


def test_none_condition_is_encoding_and_question_only():
  """The control arm must be byte-identical to an unprimed prompt."""
  built = prompts.build_prompt(GRAPH, "none", QUESTION)
  assert built == prompts.encode(GRAPH) + QUESTION
  assert not built.startswith("\n")


def test_primer_precedes_the_encoding():
  built = prompts.build_prompt(GRAPH, "degree", QUESTION)
  primer = primers.build_primer(GRAPH, "degree")
  assert built.startswith(primer)
  assert built.index(primer) < built.index("G describes a graph")


def test_cot_appends_the_datasets_own_continuation():
  """Copied from a zero_cot_test row rather than invented, so it is comparable."""
  zero = prompts.build_prompt(GRAPH, "degree", QUESTION, style="zero_shot")
  cot = prompts.build_prompt(GRAPH, "degree", QUESTION, style="zero_cot")
  assert cot == zero + prompts.COT_SUFFIX
  assert cot.endswith("Let's think step by step. ")


@pytest.mark.parametrize("condition", sorted(primers.CONDITIONS))
def test_every_condition_keeps_encoding_and_question_intact(condition):
  """Only the primer may vary between arms.

  Fatemi et al.'s central result is that phrasing alone moves accuracy by tens of
  points, so a condition differing in format as well as content would be
  uninterpretable.
  """
  built = prompts.build_prompt(GRAPH, condition, QUESTION)
  assert prompts.encode(GRAPH) + QUESTION in built


def test_unknown_style_raises():
  with pytest.raises(ValueError, match="unknown prompt style"):
    prompts.build_prompt(GRAPH, "degree", QUESTION, style="few_shot")


# --- edge_existence rewording -----------------------------------------------


def test_reword_edge_existence_asks_about_an_edge_explicitly():
  """"Connected to" is ambiguous with reachability; the task grades one edge."""
  reworded = graphqa.reword_edge_existence(
      "Q: Is node 14 connected to node 3?\nA: "
  )
  assert reworded == "Q: Does an edge exist between Node 14 and Node 3?\nA: "


@pytest.mark.parametrize("text", [
    "Q: Is there a cycle in this graph?\nA: ",
    "Q: Is node 14 connected to node three?\nA: ",
    "Is node 14 connected to node 3?",
])
def test_reword_edge_existence_rejects_unrecognised_wording(text):
  """A future dataset revision that changes the phrasing must be caught here,
  not silently reworded into something wrong."""
  with pytest.raises(ValueError, match="unexpected edge_existence wording"):
    graphqa.reword_edge_existence(text)


def test_reworded_question_reaches_the_built_prompt():
  # The graph encoding body legitimately says "is connected to node Y" for
  # each real edge (talk_like_a_graph's incident encoder), so the "not asked
  # as a connectivity question" check below has to look only at the question
  # itself, not the whole built prompt.
  question = graphqa.reword_edge_existence(
      "Q: Is node 14 connected to node 3?\nA: "
  )
  built = prompts.build_prompt(GRAPH, "none", question)
  assert built.endswith("Does an edge exist between Node 14 and Node 3?\nA: ")
  assert "Is node 14 connected to node 3" not in built


# --- answer extraction ----------------------------------------------------


@pytest.mark.parametrize("text,want", [
    ("15.", "15"),
    ("There are 15 nodes in this graph.", "15"),
    ("Counting 0,1,2,3,4 and so on. In total there are 12 nodes.", "12"),
    ("Step 1: nodes 0 through 9. Step 2: that is 10.\nThe answer is 10.", "10"),
])
def test_extracts_integers(text, want):
  assert scoring.extract_answer(text, "node_count") == want


@pytest.mark.parametrize("text,want", [
    ("Yes, there is a cycle.", "Yes"),
    ("No.", "No"),
    ("Tracing 0 to 1 to 2 back to 0. Yes.", "Yes"),
    ("No nodes repeat, so there is no cycle. Answer: No", "No"),
])
def test_extracts_booleans(text, want):
  assert scoring.extract_answer(text, "cycle_check") == want


@pytest.mark.parametrize("text,want", [
    ("1, 2, 3.", "1, 2, 3"),
    ("they are 4 and 7", "4, 7"),
    ("Node 5 has no neighbours. No nodes.", "No nodes"),
    ("Step 1: check node 0.\nThe answer is 4, 6, 11.", "4, 6, 11"),
    # The queried node's own id must not be read as its neighbour list.
    ("Node 3 is connected to nodes 0, 2, and 9.", "0, 2, 9"),
    # Tie on integer count, so the later run wins.
    ("Node 12 is connected to node 5.", "5"),
])
def test_extracts_node_lists(text, want):
  assert scoring.extract_answer(text, "connected_nodes") == want


def test_no_nodes_is_never_read_as_a_boolean_no():
  """`No nodes` shares a prefix with the boolean answer and must not collide."""
  assert scoring.extract_answer("No nodes.", "connected_nodes") == "No nodes"
  assert scoring.extract_answer("There is a cycle. Yes", "cycle_check") == "Yes"


@pytest.mark.parametrize("text", ["", "   ", "I cannot answer that."])
def test_unreadable_responses_return_none(text):
  """None is not a wrong answer, and the scorer counts the two separately.

  A run full of Nones is a truncated generation or an extractor bug; folding them
  into the accuracy would hide that behind a plausible-looking low score.
  """
  assert scoring.extract_answer(text, "node_count") is None
  assert scoring.score_one(None, "5", "node_count")["parsed"] is False


# --- metrics --------------------------------------------------------------


def test_set_f1_edges():
  assert scoring.set_f1("1, 2, 3", "1, 2, 3.") == 1.0
  assert scoring.set_f1("No nodes", "No nodes.") == 1.0
  assert scoring.set_f1("No nodes", "1, 2.") == 0.0
  assert scoring.set_f1("1, 2, 9", "1, 2, 3.") == pytest.approx(2 / 3)
  assert scoring.set_f1(None, "1, 2.") == 0.0


def test_connected_nodes_reports_both_f1_and_exact():
  """F1 is the proposal's metric; exact match is what the shortcut table uses.

  Both are kept so a model result can be read against the control under F1 and
  against the shortcut bar under exact match, without comparing different
  quantities.
  """
  result = scoring.score_one("1, 2, 9", "1, 2, 3.", "connected_nodes")
  assert result["primary"] == pytest.approx(2 / 3)
  assert result["exact"] == 0.0


def test_integer_tasks_report_absolute_error():
  assert scoring.score_one("13", "15", "node_count")["absolute_error"] == 2
  assert scoring.score_one("15", "15", "node_count")["primary"] == 1.0


def test_majority_baseline():
  answer, share = scoring.majority_baseline(
      ["Yes, there is a cycle."] * 8 + ["No, there is no cycle."] * 2
  )
  assert answer == "Yes, there is a cycle"
  assert share == pytest.approx(0.8)


def test_mcnemar_counts_only_discordant_pairs():
  # Four pairs where the treatment wins, one where the control does.
  result = scoring.mcnemar([1, 1, 0, 0, 1], [1, 0, 1, 1, 1])
  assert (result["b"], result["c"], result["discordant"]) == (1, 2, 3)

  # Perfect agreement carries no information, so p is 1 rather than 0/0.
  assert scoring.mcnemar([1, 0, 1], [1, 0, 1])["p_value"] == 1.0

  # Five discordant pairs all in one direction: 2 * C(5,0) / 2**5.
  assert scoring.mcnemar([1] * 5, [0] * 5)["p_value"] == pytest.approx(0.0625)


def test_mcnemar_rejects_misaligned_pairs():
  """A silent length mismatch would invert the pairing and the conclusion."""
  with pytest.raises(ValueError, match="equal lengths"):
    scoring.mcnemar([1, 0, 1], [1, 0])
