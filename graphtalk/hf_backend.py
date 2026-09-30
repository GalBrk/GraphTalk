"""HuggingFace generation for the sweep. Imported only by `scripts/run_sweep.py`.

Kept apart from `graphtalk/models.py` so that everything else in the package --
prompt building, scoring, the shortcut table -- stays importable without `torch`
or `transformers`. That split is what lets stages 1 and 3 of the pipeline run and
be tested on a laptop while only stage 2 needs a GPU.

The loading pattern is the one verified on the TAU cluster: bf16 weights,
`device_map="auto"`, the tokenizer's own chat template, and greedy decoding.
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
  the abandoned working -- so it would score as a confident wrong answer rather
  than as missing data; `preliminary/docs/DATA.md` puts the difference on
  `gemma4-12b-think` at 81.2% against 99.1%. The generator therefore states
  which rows reached the cap, and `graphtalk/outcomes.py` counts them as
  truncated.

  `n_new_tokens` counts generated ids including a trailing EOS, which
  `skip_special_tokens=True` drops from `text` -- so it can exceed what the
  visible text accounts for by one. That is the right count for `hit_cap`, which
  is the question being asked of it.
  """

  text: str
  n_new_tokens: int
  hit_cap: bool


class PromptOverflowError(ValueError):
  """A prompt's token count leaves no room for `max_new_tokens` under the
  model's `max_context_tokens`.

  A `ValueError` subclass (not a bare `ValueError`) specifically so
  `scripts/run_sweep.py` can catch this one condition -- a config/build
  problem the caller should record and move past -- without also
  swallowing an unrelated `ValueError` raised elsewhere in the same call
  (e.g. from a malformed `chat_kwargs`).
  """


def load(spec: models.ModelSpec):
  """Loads one model in bf16 and returns (tokenizer, model).

  bf16 rather than a quantised checkpoint: a w4a16 Gemma checkpoint is
  decompressed back to full bf16 on the first forward pass by
  `compressed-tensors`, so quantisation costs VRAM instead of saving it. Plain
  bf16 is both simpler and what fits.

  `dtype=` is the `from_pretrained` keyword from transformers 4.56 on (older
  releases take only `torch_dtype=` and reject `dtype=` when the model is
  built), which is why the `gpu` extra in pyproject.toml requires 4.56.
  """
  tokenizer = transformers.AutoTokenizer.from_pretrained(spec.repo_id)
  loader = getattr(transformers, spec.loader)
  model = loader.from_pretrained(
      spec.repo_id, dtype=torch.bfloat16, device_map="auto"
  )
  model.eval()
  return tokenizer, model


def generate(tokenizer, model, prompt: str, max_new_tokens: int,
             chat_kwargs: dict | None = None,
             max_context_tokens: int | None = None) -> Completion:
  """One greedy completion, with the prompt stripped from the return value.

  `do_sample=False` is the proposal's temperature 0. Slicing the generated ids
  past `prompt_len` rather than string-stripping the prompt afterwards avoids a
  whole class of off-by-one bugs when the chat template rewrites whitespace.

  `chat_kwargs` comes from the model's spec and reaches the template unchanged;
  see `ModelSpec.chat_kwargs` for why Qwen3 must be asked not to think.

  `max_context_tokens` (from `ModelSpec.max_context_tokens`) is checked against
  the tokenized prompt length before generation starts -- an oversized prompt
  is a config/build problem, not a modeled phenomenon, so it raises rather
  than silently truncating or letting `model.generate` fail with an opaque
  shape error. `None` (the default until a spec's context length is measured
  and filled in) makes this a no-op, same as today.
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
  if max_context_tokens and prompt_len + max_new_tokens > max_context_tokens:
    raise PromptOverflowError(
        f"prompt ({prompt_len} tokens) + max_new_tokens ({max_new_tokens}) "
        f"exceeds max_context_tokens ({max_context_tokens})"
    )
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
                    chat_kwargs: dict | None = None,
                    max_context_tokens: int | None = None) -> list[Completion]:
  """Like `generate`, but one forward pass for the whole `prompts` list
  instead of one call per prompt (`scripts/run_sweep.py --batch-size`).

  **Validated on a GPU, and not used for the sweep.** Against
  single-stream output on the 24 budget-reference prompts
  (`preliminary/analysis/budget-gemma4-e4b.jsonl`,
  `budget-qwen3-8b.jsonl`; L40S, batch size 4), 12/24 (gemma4-e4b) and 13/24
  (qwen3-8b) decoded texts were identical and qwen3-8b changed 3 of 24
  extracted answers, for a 1.44x speedup. The mismatches share a long prefix
  before diverging, which is batched-matmul floating-point order flipping a
  near-tie token, not a padding bug; but changing 3 answers in 24 is too
  large a perturbation next to the primer effects the sweep measures.
  `cluster/README.md` ("Two levers") has the table. The sweep runs at batch
  size 1, through `generate`.

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
    if max_context_tokens and prompt_len + max_new_tokens > max_context_tokens:
      raise PromptOverflowError(
          f"batch's longest prompt ({prompt_len} tokens) + max_new_tokens "
          f"({max_new_tokens}) exceeds max_context_tokens ({max_context_tokens})"
      )
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
