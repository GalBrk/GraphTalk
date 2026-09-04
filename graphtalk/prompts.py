"""Prompt assembly: primer, then the graph encoding, then the question.

The order is the proposal's: the primer is "a short preamble of factual sentences
about each node's local structure, prepended before the standard graph text
encoding and the task question". Nothing else about the prompt varies between
conditions -- same encoding, same question, same wording -- because Fatemi et
al.'s central result is that phrasing alone moves accuracy by tens of points, so
a condition that differed in format as well as content would be uninterpretable.

The prompt style is `zero_shot`, taken from the published dataset rather than
invented here, so a number can be compared with the paper's without arguing
about wording: the question ends at `A: `. Verified against `zero_shot_test`
rows of `baharef/GraphQA`.

Chain-of-thought in this project is measured by the thinking arm -- the
`-think` model specs in `graphtalk/models.py`, which enable the model's native
reasoning channel at `zero_shot` -- rather than by a separate prompt style.

The encoding is regenerated rather than taken from the row. The published
dataset ships only the `adjacency` encoding and this project uses `incident`, so
the row's `question` field is the wrong encoding and only its `task_description`
is reusable.
"""

from talk_like_a_graph import graph_text_encoders

from graphtalk import primers

ENCODING = "incident"

PROMPT_STYLES = ("zero_shot",)


def encode(graph) -> str:
  """The graph encoding every condition shares."""
  return graph_text_encoders.encode_graph(graph, ENCODING)


def build_prompt(
    graph, condition: str, task_description: str,
    style: str = "zero_shot", k_min: int = 2, k_max: int = 3,
    target_chars: int | None = None,
) -> str:
  """Assembles one prompt: primer, encoding, question.

  `condition="none"` yields an empty primer and therefore the encoding and
  question alone -- the control arm, produced by the same code path as every
  other arm rather than as a special case.
  """
  if style not in PROMPT_STYLES:
    raise ValueError(f"unknown prompt style: {style}; known: {list(PROMPT_STYLES)}")

  primer = primers.build_primer(
      graph, condition, k_min=k_min, k_max=k_max, target_chars=target_chars
  )
  body = encode(graph) + task_description
  # A blank line rather than a space: the primer is a preamble, and running it
  # into `G describes a graph...` would read as one sentence stream and make the
  # primer's boundary invisible to both the model and anyone reading a log.
  return f"{primer}\n\n{body}" if primer else body
