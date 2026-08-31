"""Tests for graphtalk/node_naming.py. No network.

Nothing existing changed to add GoT naming (see node_naming.py's module
docstring), so these tests only need to cover the new module itself: nothing
in primers.py/prompts.py/graphqa.py/scoring.py needs new cases here.
"""

import networkx as nx
import pytest

from graphtalk import graphqa
from graphtalk import node_naming
from graphtalk import primers
from graphtalk import prompts

GRAPH = graphqa.canonical(nx.Graph([(0, 1), (1, 2), (2, 3)]))
GRAPH.add_node(4)  # isolated, contiguous ids 0..4
NAME_MAP = node_naming.build_name_map(GRAPH, "got")

NON_CONTIGUOUS = graphqa.canonical(nx.Graph([(7, 0), (7, 1), (7, 2)]))


def test_build_name_map_integer_is_none():
  assert node_naming.build_name_map(GRAPH, "integer") is None


def test_build_name_map_positional_against_got_names():
  assert NAME_MAP == {i: node_naming.GOT_NAMES[i] for i in range(5)}


def test_build_name_map_rejects_non_contiguous_ids():
  with pytest.raises(ValueError, match="contiguous"):
    node_naming.build_name_map(NON_CONTIGUOUS, "got")


def test_build_name_map_rejects_too_many_nodes():
  big = graphqa.canonical(nx.empty_graph(len(node_naming.GOT_NAMES) + 1))
  with pytest.raises(ValueError, match="GoT names"):
    node_naming.build_name_map(big, "got")


def test_build_name_map_rejects_unknown_naming():
  with pytest.raises(ValueError, match="unknown node naming"):
    node_naming.build_name_map(GRAPH, "south_park")


def test_substitute_node_refs_renames_both_cases():
  text = "Node 3 has degree 2. node 3 is also queried."
  out = node_naming.substitute_node_refs(text, NAME_MAP)
  assert out == f"Node {NAME_MAP[3]} has degree 2. node {NAME_MAP[3]} is also queried."


def test_substitute_node_refs_respects_word_boundaries():
  # "node 12" must not be corrupted by the substitution rule for node 1.
  name_map = {1: "Ned", 12: "Cat"}
  text = "Node 1 and Node 12 are different."
  out = node_naming.substitute_node_refs(text, name_map)
  assert out == "Node Ned and Node Cat are different."


def test_substitute_node_refs_leaves_unrelated_numbers_alone():
  text = "Node 3 has degree 2 and clustering coefficient 0.33."
  out = node_naming.substitute_node_refs(text, NAME_MAP)
  assert out == f"Node {NAME_MAP[3]} has degree 2 and clustering coefficient 0.33."


def test_rename_task_description_handles_node_prefixed_wording():
  text = "Q: What is the degree of node 3?\nA: "
  out = node_naming.rename_task_description(text, NAME_MAP)
  assert out == f"Q: What is the degree of node {NAME_MAP[3]}?\nA: "


def test_rename_task_description_handles_unprefixed_wording():
  # connected_nodes's real wording: no "node" immediately before the id, which
  # substitute_node_refs (anchored to that word) would miss entirely.
  text = "Q: List all the nodes connected to 3 in alphabetical order.\nA: "
  out = node_naming.rename_task_description(text, NAME_MAP)
  assert out == f"Q: List all the nodes connected to {NAME_MAP[3]} in alphabetical order.\nA: "


def test_rename_task_description_handles_two_targets():
  text = "Q: Does an edge exist between Node 1 and Node 3?\nA: "
  out = node_naming.rename_task_description(text, NAME_MAP)
  assert out == f"Q: Does an edge exist between Node {NAME_MAP[1]} and Node {NAME_MAP[3]}?\nA: "


def test_rename_task_description_leaves_digit_free_wording_alone():
  text = "Q: Is there a cycle in this graph?\nA: "
  assert node_naming.rename_task_description(text, NAME_MAP) == text


def test_encode_incident_named_uses_names_everywhere():
  text = node_naming.encode_incident_named(GRAPH, NAME_MAP)
  # The node-list header and neighbour lists (already correct via name_dict).
  for name in NAME_MAP.values():
    assert name in text
  assert "0" not in text.split("\n")[0]  # header no longer says a bare "0"
  # The per-node sentence opener bug this function patches around: without
  # the fix-up this line would read "Node 1 is connected to..." (raw int).
  assert f"Node {NAME_MAP[1]} is connected to nodes {NAME_MAP[0]}, {NAME_MAP[2]}." in text


def test_build_named_primer_matches_encoding_names():
  primer = node_naming.build_named_primer(GRAPH, "degree", NAME_MAP)
  encoding = node_naming.encode_incident_named(GRAPH, NAME_MAP)
  for name in NAME_MAP.values():
    assert f"Node {name} has degree" in primer
    assert name in encoding


def test_build_named_primer_none_condition_is_empty():
  assert node_naming.build_named_primer(GRAPH, "none", NAME_MAP) == ""


def test_build_named_prompt_is_internally_consistent():
  task_description = "Q: What is the degree of node 1?\nA: "
  prompt = node_naming.build_named_prompt(GRAPH, "degree", task_description, NAME_MAP)
  # The primer, the encoding, and the question must all name node 1 the same way.
  assert prompt.count(f"Node {NAME_MAP[1]} has degree") == 1
  assert f"Node {NAME_MAP[1]} is connected to nodes" in prompt
  assert f"degree of node {NAME_MAP[1]}?" in prompt


def test_build_named_prompt_renames_unprefixed_task_description_too():
  # Regression test: connected_nodes's real wording has no "node" before the
  # id, which is exactly what rename_task_description exists to still catch.
  task_description = "Q: List all the nodes connected to 1 in alphabetical order.\nA: "
  prompt = node_naming.build_named_prompt(GRAPH, "none", task_description, NAME_MAP)
  assert f"connected to {NAME_MAP[1]} in alphabetical order" in prompt
  assert "connected to 1 " not in prompt


def test_build_named_prompt_matches_unnamed_assembly_under_identity_map():
  # An identity name map (every node maps to its own str(id)) is the same
  # information as "integer" naming, so the two assembly paths must agree.
  identity_map = {n: str(n) for n in GRAPH.nodes()}
  task_description = "Q: What is the degree of node 1?\nA: "
  named = node_naming.build_named_prompt(GRAPH, "degree", task_description, identity_map)
  plain = prompts.build_prompt(GRAPH, "degree", task_description)
  assert named == plain


def test_build_named_prompt_appends_cot_suffix():
  task_description = "Q: How many nodes are in this graph?\nA: "
  prompt = node_naming.build_named_prompt(
      GRAPH, "none", task_description, NAME_MAP, style="zero_cot"
  )
  assert prompt.endswith(prompts.COT_SUFFIX)


def test_build_named_prompt_rejects_unknown_style():
  with pytest.raises(ValueError, match="unknown prompt style"):
    node_naming.build_named_prompt(GRAPH, "none", "Q: ?\nA: ", NAME_MAP, style="bogus")


def test_desubstitute_response_round_trips_a_node_list():
  answer = f"{NAME_MAP[0]}, {NAME_MAP[1]}, and {NAME_MAP[3]}."
  assert node_naming.desubstitute_response(answer, NAME_MAP) == "0, 1, and 3."


def test_desubstitute_response_leaves_no_nodes_alone():
  assert node_naming.desubstitute_response("No nodes.", NAME_MAP) == "No nodes."


def test_desubstitute_response_leaves_plain_numbers_and_booleans_alone():
  assert node_naming.desubstitute_response("3.", NAME_MAP) == "3."
  assert node_naming.desubstitute_response("Yes.", NAME_MAP) == "Yes."


def test_got_names_uses_catelyn_not_cat():
  """"Cat" collides with a common English word; the full name doesn't."""
  assert node_naming.GOT_NAMES[1] == "Catelyn"


def test_desubstitute_response_does_not_collide_with_the_word_cat():
  """Regression guard for the collision the "Catelyn" override closes: a
  response mentioning the unrelated capitalized word "Cat" must pass
  through unchanged, not get silently rewritten to a node id.
  """
  text = "The Cat was clearly visible in the diagram."
  assert node_naming.desubstitute_response(text, NAME_MAP) == text


def test_desubstitute_response_recognizes_catelyn():
  assert (node_naming.desubstitute_response("Catelyn is isolated.", NAME_MAP)
          == "1 is isolated.")


def test_build_named_prompt_uses_catelyn_in_the_encoding():
  task_description = "Q: What is the degree of node 1?\nA: "
  prompt = node_naming.build_named_prompt(GRAPH, "degree", task_description, NAME_MAP)
  assert "Catelyn" in prompt
  assert "Cat " not in prompt and not prompt.endswith("Cat")
