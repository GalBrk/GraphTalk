"""HuggingFace generation for the sweep. Imported only by the cluster job.

Kept apart from `graphtalk/models.py` so that everything else in the package --
prompt building, scoring, the shortcut table -- stays importable without `torch`
or `transformers`. That split is what lets stages 1 and 3 of the pipeline run and
be tested on a laptop while only stage 2 needs a GPU.

Mirrors the loading pattern already verified on this cluster in the SlidesGen
`gemma_play` setup: bf16 weights, `device_map="auto"`, the tokenizer's own chat
template, and greedy decoding.
"""

import torch
import transformers

from graphtalk import models


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
             chat_kwargs: dict | None = None) -> str:
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
  return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
