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
  """

  key: str
  repo_id: str
  family: str
  size: str
  loader: str
  min_vram_gb: int


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
                  "AutoModelForCausalLM", 24),
        ModelSpec("qwen3-14b", "Qwen/Qwen3-14B", "qwen3", "14B",
                  "AutoModelForCausalLM", 48),
    )
}

# Generation length. Zero-shot answers are a few tokens; a CoT answer needs room
# to reason and then state a conclusion, and a truncated CoT response loses the
# conclusion specifically -- which the extractor reads as unparseable and the
# scorer would otherwise read as a wrong answer. Sized per style rather than once,
# because paying the CoT budget on every zero-shot row would multiply the sweep's
# runtime for nothing.
MAX_NEW_TOKENS = {"zero_shot": 64, "zero_cot": 1024}
