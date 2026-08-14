"""Prompt assembly: primer, then the graph encoding, then the question.

The order is the proposal's: the primer is "a short preamble of factual sentences
about each node's local structure, prepended before the standard graph text
encoding and the task question". Nothing else about the prompt varies between
conditions -- same encoding, same question, same wording -- because Fatemi et
al.'s central result is that phrasing alone moves accuracy by tens of points, so
a condition that differed in format as well as content would be uninterpretable.

Two prompt styles, both taken from the published dataset rather than invented
here, so a number can be compared with the paper's without arguing about
wording. `zero_shot` ends the question at `A: `; `zero_cot` appends the dataset's
own `Let's think step by step. `. Verified against `zero_shot_test` and
`zero_cot_test` rows of `baharef/GraphQA`.

The encoding is regenerated rather than taken from the row. The published
dataset ships only the `adjacency` encoding and this project uses `incident`, so
the row's `question` field is the wrong encoding and only its `task_description`
is reusable.
"""

from talk_like_a_graph import graph_text_encoders

from graphtalk import primers

ENCODING = "incident"

# The dataset's own zero-shot-CoT continuation, copied from a `zero_cot_test`
# row. The trailing space is part of it: the question ends `A: ` and the model
# continues from `A: Let's think step by step. `.
COT_SUFFIX = "Let's think step by step. "

PROMPT_STYLES = ("zero_shot", "zero_cot")


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
  if style == "zero_cot":
    body += COT_SUFFIX
  # A blank line rather than a space: the primer is a preamble, and running it
  # into `G describes a graph...` would read as one sentence stream and make the
  # primer's boundary invisible to both the model and anyone reading a log.
  return f"{primer}\n\n{body}" if primer else body
