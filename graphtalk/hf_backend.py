"""HuggingFace generation for the sweep. Imported only by the cluster job.

Kept apart from `graphtalk/models.py` so that everything else in the package --
prompt building, scoring, the shortcut table -- stays importable without `torch`
or `transformers`. That split is what lets stages 1 and 3 of the pipeline run and
be tested on a laptop while only stage 2 needs a GPU.

Mirrors the loading pattern already verified on this cluster in the SlidesGen
`gemma_play` setup: bf16 weights, `device_map="auto"`, the tokenizer's own chat
template, and greedy decoding.
"""

import dataclasses

import torch
import transformers

from graphtalk import models


@dataclasses.dataclass(frozen=True)
class Completion:
  """One generation, plus the two facts about *how* it ended.

  Recorded per row because non-termination is a measurement, not a hunch. A
  response cut off at the cap still parses -- the extractor finds an integer in
  the abandoned working -- so it scores as a confident wrong answer rather than
  as missing data, and `docs/DATA.md` puts the difference on `gemma4-12b-think`
  at 81.2% against 99.1%. Until now the only record of which rows those were was
  the hand-maintained `analysis/truncated_keys.json`, derived by a route nothing
  in the repo scripts; the generator states it directly instead.

  `n_new_tokens` counts generated ids including a trailing EOS, which
  `skip_special_tokens=True` drops from `text` -- so it can exceed what the
  visible text accounts for by one. That is the right count for `hit_cap`, which
  is the question being asked of it.
  """

  text: str
  n_new_tokens: int
  hit_cap: bool


def load(spec: models.ModelSpec):
  """Loads one model in bf16 and returns (tokenizer, model).

  bf16 rather than a quantised checkpoint: the SlidesGen run found that a w4a16
  Gemma checkpoint is decompressed back to full bf16 on the first forward pass by
  `compressed-tensors`, so quantisation cost VRAM instead of saving it. Plain bf16
  is both simpler and what fits.
  """
  tokenizer = transformers.AutoTokenizer.from_pretrained(spec.repo_id)
  loader = getattr(transformers, spec.loader)
  model = loader.from_pretrained(
      spec.repo_id, dtype=torch.bfloat16, device_map="auto"
  )
  model.eval()
  return tokenizer, model


def generate(tokenizer, model, prompt: str, max_new_tokens: int,
             chat_kwargs: dict | None = None) -> Completion:
  """One greedy completion, with the prompt stripped from the return value.

  `do_sample=False` is the proposal's temperature 0. Slicing the generated ids
  past `prompt_len` rather than string-stripping the prompt afterwards avoids a
  whole class of off-by-one bugs when the chat template rewrites whitespace.

  `chat_kwargs` comes from the model's spec and reaches the template unchanged;
  see `ModelSpec.chat_kwargs` for why Qwen3 must be asked not to think.
  """
  messages = [{"role": "user", "content": prompt}]
  inputs = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      tokenize=True,
      return_dict=True,
      return_tensors="pt",
      **(chat_kwargs or {}),
  ).to(model.device)

  prompt_len = inputs["input_ids"].shape[-1]
  with torch.inference_mode():
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
  n_new_tokens = out.shape[-1] - prompt_len
  return Completion(
      text=tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True),
      n_new_tokens=n_new_tokens,
      # `generate` stops *at* the budget, so equality is the cap being reached.
      hit_cap=n_new_tokens >= max_new_tokens,
  )
