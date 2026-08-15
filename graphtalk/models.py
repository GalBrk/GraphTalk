"""The models the proposal names, as configuration only.

Deliberately free of `torch` and `transformers` so that prompt building and
scoring stay importable on a laptop with no GPU stack. The loading and generation
live in `graphtalk/hf_backend.py`, which the cluster job imports and nothing else
does.

Two families at two sizes each, per the proposal: a within-family capacity
comparison and a cross-family check. All queried greedily -- the proposal fixes
temperature at 0, which in `transformers` means `do_sample=False` rather than
`temperature=0.0`, since a literal zero temperature is a division by zero in the
sampling path.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelSpec:
  """What the backend needs to load one model.

  `loader` names the `transformers` auto-class. Gemma 4 checkpoints are
  multimodal and resolve through `AutoModelForImageTextToText` even for text-only
  use -- `AutoModelForCausalLM` does not resolve them. Qwen3 is text-only and
  takes the ordinary causal class.

  `min_vram_gb` is the measured bf16 weight footprint plus working room, used to
  pick a `--constraint` on the cluster rather than discovering an OOM an hour into
  the queue.

  `chat_kwargs` are passed to the tokenizer's `apply_chat_template`. Qwen3 needs
  `enable_thinking=False` there: its template otherwise opens a `<think>` block
  and the model reasons before answering on *both* prompt styles, which spends
  1529 tokens on a node_count question Gemma answers in 167 and, worse, makes
  `zero_shot` and `zero_cot` the same condition for that family -- collapsing the
  pairing the McNemar test is computed over. Gemma's template has no such switch
  and takes an empty dict.
  """

  key: str
  repo_id: str
  family: str
  size: str
  loader: str
  min_vram_gb: int
  chat_kwargs: dict = dataclasses.field(default_factory=dict)


MODELS = {
    spec.key: spec
    for spec in (
        # Gemma 4 12B in bf16 is ~24 GB of weights and is verified working on a
        # 48 GB A6000 in the SlidesGen setup on this same cluster and account.
        #
        # The proposal names "Gemma 4 4B", but no such checkpoint exists: the
        # small Gemma 4 releases are the E2B/E4B variants, whose "E" size is an
        # *effective* parameter count rather than a raw one. E4B is the closest
        # thing to the proposal's intent and is what the sweep uses; say so in the
        # write-up rather than calling it 4B.
        ModelSpec("gemma4-e4b", "google/gemma-4-E4B-it", "gemma4", "E4B",
                  "AutoModelForImageTextToText", 24),
        ModelSpec("gemma4-12b", "google/gemma-4-12B-it", "gemma4", "12B",
                  "AutoModelForImageTextToText", 48),
        ModelSpec("qwen3-8b", "Qwen/Qwen3-8B", "qwen3", "8B",
                  "AutoModelForCausalLM", 24,
                  {"enable_thinking": False}),
        ModelSpec("qwen3-14b", "Qwen/Qwen3-14B", "qwen3", "14B",
                  "AutoModelForCausalLM", 48,
                  {"enable_thinking": False}),
    )
}

# Generation length. A truncated response loses its conclusion specifically --
# which the extractor reads as unparseable, or worse, reads as a wrong answer
# after picking up an integer from the abandoned working.
#
# `zero_shot` was 64 on the assumption that a zero-shot answer is a few tokens
# long. Measured on 24 rows spanning all six tasks, that is false for these
# instruction-tuned chat models: given room, they narrate their working and then
# answer, so 64 tokens truncated ~90% of rows mid-sentence. The measured
# distribution (gemma4-e4b, cap 2048) is median 271 and max 1974, with
# `edge_count` the tail -- it enumerates and sums every edge, median 1390. At 64
# tokens gemma4-e4b scored 3/24; at 2048 it scored 24/24, and no row hit the cap.
#
# Note what this does to the design: at 2048 the model reasons on the zero_shot
# prompt too, so the contrast against `zero_cot` narrows to the prompt wording
# rather than to whether reasoning happens at all. The 64-token cap suppressed
# that, but by truncation rather than by design, which is not a contrast worth
# keeping.
MAX_NEW_TOKENS = {"zero_shot": 2048, "zero_cot": 1024}
