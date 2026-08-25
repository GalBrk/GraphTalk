# Measurement artefacts

Small, expensive-to-reproduce measurements that decisions in this project rest
on. They live in git because the reasoning in `docs/sweep-findings.md` and
`cluster/README.md` cites them, and because each cost GPU time that a reader
should not have to spend again to check the claim.

| file | what it is |
|---|---|
| `budget-gemma4-e4b.jsonl` | 24 `zero_shot` prompts spanning all six tasks, generated at a 2048 cap with exact `n_new_tokens` per row. The measurement that set `MAX_NEW_TOKENS["zero_shot"]`: at 64 tokens this model scored 3/24, at 2048 it scored 24/24 with no row reaching the cap. |
| `budget-qwen3-8b.jsonl` | the same 24 prompts on Qwen3-8B with `enable_thinking=False` |
| `budget-qwen3-8b-THINKING.jsonl` | three of those rows with thinking **on**, kept as the before-picture: 1179 mean tokens on a `node_count` question the same model answers in 105 without |
| `truncated_keys.json` | the exact `(instance_id, condition, style)` of all 350 non-terminating thinking-arm rows, per model |

## The batching baseline

`budget-gemma4-e4b.jsonl` and `budget-qwen3-8b.jsonl` are the reference for
verifying a batched implementation. Decoding is greedy, so a correct batch must
reproduce these responses near-identically. That check matters more than it
sounds: **Qwen3's tokenizer defaults to `padding_side='right'` while Gemma's
defaults to `'left'`**, and for a decoder-only model the wrong padding side
produces fluent, well-formed, entirely wrong text rather than an error.

## Not kept

The prompt subsets used to drive these runs (`diag_prompts`, `think_probe`,
`tail-probe`, `redo-*`) are regenerable from `prompts_zero_shot.jsonl` plus
`truncated_keys.json`, and are not tracked.
