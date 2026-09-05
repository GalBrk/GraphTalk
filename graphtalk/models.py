"""The models the proposal names, as configuration only.

Deliberately free of `torch` and `transformers` so that prompt building and
scoring stay importable on a laptop with no GPU stack. The loading and generation
live in `graphtalk/hf_backend.py`, which the cluster job imports and nothing else
does.

Two families at two sizes each, per the proposal: a within-family capacity
comparison and a cross-family check. Plus `qwen3-0.6b`, added afterwards and
outside the proposal's grid, to put a point below the ceiling the first sweep ran
into -- it extends the Qwen ladder to a third size rather than opening a third
family, so the within-family comparison stays a comparison. All queried greedily
-- the proposal fixes temperature at 0, which in `transformers` means
`do_sample=False` rather than `temperature=0.0`, since a literal zero temperature
is a division by zero in the sampling path.
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

  `chat_kwargs` are passed to the tokenizer's `apply_chat_template`. Both families
  have a thinking mode and their defaults are opposite: Gemma 4 defaults to
  thinking off, Qwen3 defaults to thinking on. Left alone that is an accidental
  asymmetry -- Qwen would reason in a hidden `<think>` channel even in the plain
  arm, spending 1179 tokens on a node_count question it answers in 105 without,
  and collapsing the plain and thinking arms into one condition for that family
  only, which is the pairing the McNemar test is computed over. The main specs
  therefore pin both families to thinking off; the `-think` specs pin both to on,
  as a separate arm rather than a confound inside the first.

  `max_new_tokens` overrides the module-level budget for this model. Thinking
  mode needs far more room than the same model without it, and a truncated
  `<think>` block loses the answer entirely -- the conclusion comes after the
  reasoning, so the cost of being wrong here is the whole row.
  """

  key: str
  repo_id: str
  family: str
  size: str
  loader: str
  min_vram_gb: int
  chat_kwargs: dict = dataclasses.field(default_factory=dict)
  max_new_tokens: dict | None = None


# Budget for the thinking arm. Placeholder until measured -- see
# scripts/measure_budget.py; the number below is replaced by the measured one
# before the arm is launched, because a truncated <think> block loses the answer
# rather than shortening it.
THINK_MAX_NEW_TOKENS = {"zero_shot": 8192}

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

        # The headroom probe. Every model above sits at 83-100% pooled exact
        # match, and `docs/sweep-findings.md` shows the consequence: all 144
        # McNemar cells have fewer than 10 discordant pairs, and for
        # `gemma4-12b` that is a ceiling, not a sample size -- there is nothing
        # left for a primer to flip. That document's "Target the headroom" names
        # harder instances as the fix; a weaker model is the other lever on the
        # same problem, and much the cheaper one, since it reuses the existing
        # prompt file rather than requiring a new corpus.
        #
        # Qwen3 rather than the smallest Gemma available. The choice matters for
        # two reasons the size alone does not capture:
        #   - It extends a *within-family* capacity ladder, 14B -> 8B -> 0.6B,
        #     which is the comparison the proposal's design is built around. A
        #     small model from another family (or another Gemma generation) would
        #     be a fifth unpaired point instead.
        #   - It has a native thinking mode, so the `-think` arm below exists for
        #     it on the same terms as for every other spec. A model without one
        #     could only ever fill half the design's rows.
        #
        # ~1.2 GB of bf16 weights, so `min_vram_gb` is far under the smallest
        # card in sweep.sbatch's constraint set; it is recorded for consistency
        # with the specs above, not because any card in `killable` could fail it.
        #
        # What this spec cannot tell you on its own: whether the model is weak at
        # *graph reasoning* or merely weak at *emitting a parseable answer*.
        # `scoring.extract_answer` counts an unreadable response as `None`,
        # separately from a wrong one -- read the parse rate before reading the
        # accuracy, because at this size the two failure modes are easy to
        # confuse and only one of them is about primers.
        ModelSpec("qwen3-0.6b", "Qwen/Qwen3-0.6B", "qwen3", "0.6B",
                  "AutoModelForCausalLM", 4,
                  {"enable_thinking": False}),

        # The middle rung. `qwen3-0.6b`'s probe left two of six tasks unusable
        # for opposite reasons: `edge_count` floored at 0.08 in the plain arm,
        # and `node_count` was dominated by an off-by-one that is a labelling
        # error rather than a graph-reasoning one (the model reports the highest
        # node label, N-1, instead of |V|). That artifact is size-dependent --
        # 0% on graphs under 10 nodes, 93% on 16+ -- so it reads as a counting
        # capacity limit rather than anything about primers, and it does not
        # appear at all in the 8B or 14B. 1.7B is the next rung up the same
        # ladder and the cheapest test of whether clearing that limit restores
        # `node_count` as an interpretable cell while keeping the headroom that
        # made the 0.6B worth running.
        ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B", "qwen3", "1.7B",
                  "AutoModelForCausalLM", 8,
                  {"enable_thinking": False}),

        # A newer generation, and deliberately a *pair* candidate rather than a
        # fifth unpaired point: Qwen3.5 (Feb 2026) is a different family from
        # Qwen3, so one size of it buys no within-family comparison. This spec
        # exists to answer one question before any of that is worth building
        # out -- does a generation-newer small model keep the headroom, or does
        # it hand back the ceiling that made this whole exercise necessary?
        # qwen3-8b already scores 0.997 on node_count and edge_existence, so
        # that risk is real and 2B is the conservative size to test it at.
        #
        # `AutoModelForImageTextToText`, not the causal class: Qwen3.5
        # checkpoints declare `Qwen3_5ForConditionalGeneration` and are
        # multimodal even at 2B, exactly the situation the Gemma 4 specs above
        # document. Verified supported by the env's transformers 5.15.0.
        #
        # Key is `qwen35-2b`, not `qwen3.5-2b`. The key becomes a filename
        # component (`runs/<key>.jsonl`) and this repo delimits run tags and
        # shard suffixes with dots (`.probe100.`, `.shard0of6.`). `qwen3-0.6b`
        # already carries one dot harmlessly, but a second in the family
        # position is where that convention would start to be ambiguous.
        ModelSpec("qwen35-2b", "Qwen/Qwen3.5-2B", "qwen35", "2B",
                  "AutoModelForImageTextToText", 12,
                  {"enable_thinking": False}),

        # The thinking arm. Same checkpoints, same prompt file, same instances --
        # the only difference is that the model is allowed its native reasoning
        # channel, so this is comparable row-for-row against the specs above.
        #
        # Separate keys rather than a flag because the output path is derived from
        # the key: `runs/<key>.jsonl`. Sharing a path would let the resume logic
        # treat a thinking row as satisfying a non-thinking one and silently mix
        # the two arms in a file nothing could unmix afterwards.
        #
        # THINK_MAX_NEW_TOKENS is measured, not guessed; see below.
        ModelSpec("gemma4-e4b-think", "google/gemma-4-E4B-it", "gemma4", "E4B",
                  "AutoModelForImageTextToText", 24,
                  {"enable_thinking": True}, THINK_MAX_NEW_TOKENS),
        ModelSpec("gemma4-12b-think", "google/gemma-4-12B-it", "gemma4", "12B",
                  "AutoModelForImageTextToText", 48,
                  {"enable_thinking": True}, THINK_MAX_NEW_TOKENS),
        ModelSpec("qwen3-8b-think", "Qwen/Qwen3-8B", "qwen3", "8B",
                  "AutoModelForCausalLM", 24,
                  {"enable_thinking": True}, THINK_MAX_NEW_TOKENS),
        ModelSpec("qwen3-14b-think", "Qwen/Qwen3-14B", "qwen3", "14B",
                  "AutoModelForCausalLM", 48,
                  {"enable_thinking": True}, THINK_MAX_NEW_TOKENS),

        # The probe's thinking twin. Budget deliberately left at the shared
        # THINK_MAX_NEW_TOKENS rather than tuned down for a small model: the
        # arms are compared row for row, and `budget`'s docstring is explicit
        # that a per-model budget change makes new rows incomparable with the
        # ones already on disk.
        #
        # Expect this arm to truncate far more than the larger ones do. Small
        # models repeat rather than stop, and truncation has impersonated a
        # finding in this project three separate times (see "Truncation
        # impersonates a finding" in docs/sweep-findings.md) -- a capped response
        # still parses, so it scores as a confident wrong answer rather than as
        # missing data. Filter on `hit_cap` before comparing anything here.
        ModelSpec("qwen3-0.6b-think", "Qwen/Qwen3-0.6B", "qwen3", "0.6B",
                  "AutoModelForCausalLM", 4,
                  {"enable_thinking": True}, THINK_MAX_NEW_TOKENS),
        ModelSpec("qwen3-1.7b-think", "Qwen/Qwen3-1.7B", "qwen3", "1.7B",
                  "AutoModelForCausalLM", 8,
                  {"enable_thinking": True}, THINK_MAX_NEW_TOKENS),
        ModelSpec("qwen35-2b-think", "Qwen/Qwen3.5-2B", "qwen35", "2B",
                  "AutoModelForImageTextToText", 12,
                  {"enable_thinking": True}, THINK_MAX_NEW_TOKENS),
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
MAX_NEW_TOKENS = {"zero_shot": 2048}


def budget(spec: ModelSpec, style: str) -> int:
  """New-token budget for one (model, style).

  Falls back to the module-level table, so the non-thinking specs keep exactly
  the numbers the first sweep was generated with -- changing that silently would
  make new rows incomparable with the 10,080 already on disk.
  """
  table = spec.max_new_tokens or MAX_NEW_TOKENS
  return table[style]
