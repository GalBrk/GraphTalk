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


def generate_batch(tokenizer, model, prompts: list[str], max_new_tokens: int,
                    chat_kwargs: dict | None = None) -> list[Completion]:
  """Like `generate`, but one forward pass for the whole `prompts` list
  instead of one call per prompt -- Track 2.3, the infrastructure 2.1/2.2's
  larger recommended `--count`s need to be affordable at all (single-stream
  leaves most of the GPU idle; see `cluster/README.md`'s "Two levers"
  section, which already measured the padding-side hazard this function
  has to get right).

  **NOT YET VALIDATED ON A GPU** -- this dev environment has no `torch`
  install and no CUDA device, so this function has only been checked by
  reading, not by running. Before trusting it for a real sweep: run it
  against the same prompts `analysis/budget-gemma4-e4b.jsonl` and
  `analysis/budget-qwen3-8b.jsonl` came from and confirm the decoded text
  matches `generate`'s single-stream output near-identically (greedy
  decoding, so it should be exact modulo the known floating-point
  non-associativity of batched vs. unbatched matmuls) -- for *both*
  families, since they need opposite padding sides (below) and only
  testing one would leave the other's hazard unchecked.

  **Padding side.** Decoder-only generation must left-pad: the model
  predicts each batch member's next token from the *last* position of its
  input, so right-padding would have it predict from a pad token instead
  of the real last prompt token for every row shorter than the batch's
  longest. `gemma-4-E4B-it` already defaults to `padding_side='left'`, but
  **`Qwen3` defaults to `'right'`** -- wrong padding produces fluent,
  well-formed, entirely wrong text rather than an error, so this is set
  explicitly here rather than trusted to the tokenizer's default for
  either family.

  **Missing pad token.** Several causal-LM tokenizers (Qwen3 among them)
  ship no `pad_token` at all, which left-padding requires; falls back to
  the model's own `eos_token` (the standard workaround -- an extra
  padding-shaped "end of sequence" costs nothing the model wasn't already
  trained to emit).

  **Recovering each row's true length.** `model.generate` runs the whole
  batch until every member has produced an EOS or the batch hits
  `max_new_tokens`; a row that finishes earlier than the batch's longest
  gets `pad_token_id`-filled for the remaining steps rather than truly
  stopping there. So `out.shape[-1] - prompt_len` (the single-stream
  formula) is the *batch's* length, not each row's -- reusing it directly
  would report every early-finishing row as having hit the cap. Instead,
  each row's true `n_new_tokens` is the index of the first `pad_token_id`
  in its generated slice, plus one (matching `generate`'s own convention
  of counting the terminating token itself, see `Completion.n_new_tokens`'s
  docstring) -- or the full slice length, with `hit_cap=True`, if no pad
  id appears (that row used the entire budget without producing its own
  stop). This assumes `pad_token_id` never appears as *real* generated
  content, which is true whenever it is a genuine special/reserved token
  (always true for the `pad_token = eos_token` fallback above, since a
  content token identical to EOS would have stopped generation already).
  """
  original_padding_side = tokenizer.padding_side
  tokenizer.padding_side = "left"
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
  pad_token_id = tokenizer.pad_token_id
  try:
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    inputs = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        **(chat_kwargs or {}),
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
      out = model.generate(
          **inputs, max_new_tokens=max_new_tokens, do_sample=False,
          pad_token_id=pad_token_id,
      )
  finally:
    # Restored even on failure -- `generate` (single-stream) is called on
    # the same shared tokenizer object elsewhere in the same process and
    # does not expect `padding_side` to have been changed out from under it.
    tokenizer.padding_side = original_padding_side

  completions = []
  for row in out[:, prompt_len:]:
    row = row.tolist()
    pad_positions = [i for i, tok in enumerate(row) if tok == pad_token_id]
    if pad_positions:
      n_new_tokens = pad_positions[0] + 1
      hit_cap = False
    else:
      n_new_tokens = len(row)
      hit_cap = n_new_tokens >= max_new_tokens
    completions.append(Completion(
        text=tokenizer.decode(row[:n_new_tokens], skip_special_tokens=True),
        n_new_tokens=n_new_tokens,
        hit_cap=hit_cap,
    ))
  return completions
