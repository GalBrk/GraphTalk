"""Game-of-Thrones node naming, layered on top of the integer pipeline.

Additive by design: every function here is new, and nothing in `primers.py`,
`prompts.py`, `graphqa.py`, or `scoring.py` is touched. A named prompt is built
by rendering the ordinary integer primer/encoding/question exactly as today
and then substituting node-id references for names; a named model response is
converted back to integers before it reaches the existing, unmodified scorer.

Two things make the substitution safe rather than a blanket find-and-replace
over numbers (which would corrupt degree counts, RWSE decimals, and neighbour
lists):

  * In primer text and in every task description, a node id only ever appears
    immediately after the literal word "node" or "Node" -- never a count, a
    coefficient, or a list entry.
  * The `incident` encoder's node-list header and neighbour lists already
    receive real names correctly when the encoder is called with a real name
    dict (they come from `name_dict[...]` lookups); the encoder's own
    per-node sentence opener does not
    (`talk_like_a_graph/graph_text_encoders.py`'s `incident_encoder` hardcodes
    the raw `source_node` int there regardless of `name_dict`, a vendored bug
    left alone since `talk_like_a_graph/` is third-party). The same "node "-
    prefixed substitution patches exactly that opener.

GoT names are a fixed, graph-independent list: `name_dictionaries` assigns
them by list position, not by anything about the graph, so `GOT_NAMES` is one
constant reused for every graph, and desubstituting a response needs no
graph-specific state either.
"""

import re

from talk_like_a_graph import graph_text_encoders
from talk_like_a_graph import name_dictionaries

from graphtalk import primers
from graphtalk import prompts

# {0: "Ned", 1: "Catelyn", ..., 19: "Osha"} -- see the module docstring on why
# this is graph-independent.
GOT_NAMES: dict[int, str] = name_dictionaries.create_name_dict(None, "got")
# The vendored list spells this character "Cat" -- a common English word,
# and the only one of the 20 short forms here that collides with one. Every
# other function in this module keys off `GOT_NAMES`'s values, so a values-
# only override is already correctly threaded through prompt-building and
# `desubstitute_response` alike, with no other special-casing needed. Not
# edited in the vendored `talk_like_a_graph/name_dictionaries.py` itself --
# see the module docstring above on treating that directory as third-party.
# Deliberately not kept as a dual alias for "Cat": a model that still
# abbreviates "Catelyn" that way would just miss desubstitution for that one
# mention, which is a safe miss, not a reintroduction of the collision this
# closes.
GOT_NAMES[1] = "Catelyn"

NAMINGS = ("integer", "got")

_NODE_REF_RE = re.compile(r"\b([Nn]ode) (\d+)\b")


def build_name_map(graph, naming: str) -> dict[int, str] | None:
  """`None` for `"integer"` (today's behavior); an id->name dict for `"got"`.

  Raises if the graph's node ids aren't exactly `0..n-1` or if it has more
  nodes than there are GoT names: both are silent-wrong-name failure modes
  otherwise, since `GOT_NAMES` is assigned by list position.
  """
  if naming == "integer":
    return None
  if naming != "got":
    raise ValueError(f"unknown node naming: {naming!r}; known: {NAMINGS}")
  n = graph.number_of_nodes()
  if set(graph.nodes()) != set(range(n)):
    raise ValueError("got naming requires contiguous 0..n-1 node ids")
  if n > len(GOT_NAMES):
    raise ValueError(f"graph has {n} nodes but only {len(GOT_NAMES)} GoT names are defined")
  return {i: GOT_NAMES[i] for i in range(n)}


def substitute_node_refs(text: str, name_map: dict[int, str]) -> str:
  """Replaces a `node <id>`/`Node <id>` self-reference with its display name."""
  return _NODE_REF_RE.sub(lambda m: f"{m.group(1)} {name_map[int(m.group(2))]}", text)


def encode_incident_named(graph, name_map: dict[int, str]) -> str:
  """The `incident` encoding with every node reference in `name_map`'s names."""
  text = graph_text_encoders.EDGE_ENCODER_FN["incident"](graph, name_map)
  return substitute_node_refs(text, name_map)


_DIGIT_RE = re.compile(r"\d+")


def rename_task_description(task_description: str, name_map: dict[int, str]) -> str:
  """Replaces every node-id digit in a task description with its display name.

  Broader than `substitute_node_refs` on purpose: a task description's only
  numeric content is ever its query node id(s) -- `graphqa._target_nodes`
  already relies on that same fact to recover query targets from any task
  description -- so a blanket digit substitution is safe here specifically.
  It has to be broader, too: `connected_nodes`'s wording ("...connected to
  14 in alphabetical order.") names its query node without the word "node"
  in front of it, which the narrower, "node "-anchored
  `substitute_node_refs` would miss.
  """
  return _DIGIT_RE.sub(lambda m: name_map[int(m.group())], task_description)


def build_named_primer(
    graph, condition: str, name_map: dict[int, str],
    k_min: int = 2, k_max: int = 3, target_chars: int | None = None,
) -> str:
  """The named counterpart of `primers.build_primer`."""
  text = primers.build_primer(
      graph, condition, k_min=k_min, k_max=k_max, target_chars=target_chars
  )
  return substitute_node_refs(text, name_map) if text else text


def build_named_prompt(
    graph, condition: str, task_description: str, name_map: dict[int, str],
    style: str = "zero_shot", k_min: int = 2, k_max: int = 3,
    target_chars: int | None = None,
) -> str:
  """The named counterpart of `prompts.build_prompt`: same assembly, GoT names."""
  if style not in prompts.PROMPT_STYLES:
    raise ValueError(f"unknown prompt style: {style}; known: {list(prompts.PROMPT_STYLES)}")

  primer = build_named_primer(
      graph, condition, name_map, k_min=k_min, k_max=k_max, target_chars=target_chars
  )
  body = encode_incident_named(graph, name_map) + rename_task_description(
      task_description, name_map
  )
  return f"{primer}\n\n{body}" if primer else body


def desubstitute_response(text: str, name_map: dict[int, str]) -> str:
  """Replaces known GoT names in a model response with their node id.

  Lets a GoT-worded response be scored by the existing, unmodified
  `scoring.extract_answer`/`scoring.score_one`, which only understand integer
  node ids.
  """
  inverse = {name: str(i) for i, name in name_map.items()}
  pattern = re.compile(
      r"\b(" + "|".join(re.escape(n) for n in sorted(inverse, key=len, reverse=True)) + r")\b"
  )
  return pattern.sub(lambda m: inverse[m.group(1)], text)
