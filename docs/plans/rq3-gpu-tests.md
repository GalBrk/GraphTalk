# RQ3 GPU tests: shuffled/reversed/placebo primers on `clustering`

**Status:** planned, not run. No GPU work has happened yet; this documents
the design so it is reachable outside any one conversation. The CPU-only
findings that motivate this plan are in `docs/results/density-followups.md`
(§2 for the effect, §8 for its selection check and the position mechanism);
read that first. The earlier notes, with the numbered leads this plan
implements, are in `superseded/docs/rq3-leads.md`.

## Why 1.7B, plain arm only

Yes, this is still worth running even restricted to one arm:

- The `clustering` effect on `node_degree` is a 1.7B plain-arm effect. The
  1.7B thinking arm gains +2.5 at p≤.50, carried by p=.20 (+5.2), and nothing
  at p=.35 or .50 (+3.2 and −0.8, neither significant;
  `docs/results/density-followups.md` §4). 4B shows an effect only on the dense extension (+11.3), not the
  main sweep.
- Skipping thinking mode is a large saving, not just a convenience: a
  thinking run needs an 8,192-token budget against ~141 tokens/answer for
  the plain arm, and the thinking arm has no effect at these densities to
  explain.
- The cost of restricting to one arm is generality, not correctness: this
  tells you the mechanism in one model, not whether it holds elsewhere.
  The paper already limits the claim to "one model, one task."
- A later check on 4B's existing dense-graph rows (position split) needs
  no new GPU time — see "CPU follow-ups" in `superseded/docs/rq3-leads.md`.

## Design

**Graphs:** 600 new-seed graphs at each of p=.35 and p=.50 (1,200 total).
**Seed:** 20510906 — 250,000 away from both the default seed (20260906)
and the replication seed (20760906), which should avoid graph reuse;
verify zero `instance_id` collisions before running, as
`build_size_sweep.py`'s existing guard did for the replication.

**Conditions:** `none` and `clustering` as anchors, plus the new primers
below. `filler` is dropped — it already equals `none` at these densities
(−0.5, `mid_pooled.filler_none` under `[rqselection]` in
`csv2/density-followups/density_followups.txt`).

**Why rerun the anchors instead of pairing with existing rows:**
`graphtalk/hf_backend.py` hardcodes `dtype=torch.bfloat16`. Older GPUs
either lack native bf16 (Pascal) or only emulate it (Turing/Volta), so an
old-GPU run would use fp16 or fp32 and produce different greedy outputs —
the existing bf16 rows can't be paired with them. Since the anchors have
to be regenerated anyway, running them on the fresh seed above gives a
third out-of-sample check of the pooled effect, with the hypotheses fixed
before the run (satisfying lead #5 in `superseded/docs/rq3-leads.md`, the
pre-registered confirmation).

### New primer conditions

| # | Condition | Stage | Tests |
|---|---|---|---|
| 1 | **Shuffled values** — same clustering numbers, permuted across nodes | 1 | Does the node↔value pairing matter, or just the format? |
| 2 | **Reversed order** — node 39's sentence first, node 0's last | 1 | Does the gain track the primer's sentence order or the encoding's node order? |
| 3 | **Numeric placebo** — random values in [0,1], same template | 2, only if #1 persists | Real statistics vs. any per-node number |
| 4 | **Primer after the encoding** | 2, lowest priority | Does position relative to the encoding matter? |

### What each result would mean

- **#1 persists** (shuffled ≈ clustering): the node–value pairing doesn't
  matter; the paper can say content isn't used, stronger than "two
  mechanisms not supported."
  **#1 vanishes**: values do matter, contradicting the rewiring and
  own-value nulls already in the paper — the account needs rethinking.
- **#2 gain moves to high ids**: tied to the primer's own sentence order.
  **#2 gain stays on low ids**: tied to the encoding's node order instead
  (an alignment account, since `clustering`'s sentences and the encoding's
  lines are both in node-id order today).
- **#3** separates "per-node numbers help" from "these particular
  statistics help."
- **#4** tests whether the primer needs to come first to work as a
  retrieval aid.

### Primary tests (fixed before the run)

- H1: `clustering` − `none` > 0 (replicates the anchor effect).
- H2: `shuffled` − `none` > 0, and `shuffled` ≈ `clustering` within a
  pre-set equivalence margin.
- H3: the slope of the paired gain against node id (0–39) is negative
  under `clustering` and its sign flips under `reversed`.

### Power

- 1,200 pairs/condition gives a 95% CI of about ±3.4 points on
  `shuffled` − `clustering` — enough to rule out losing the full effect at
  these densities (5.88 points, `docs/results/density-followups.md` §8).
- The reversed-order test has ~0.8 power to detect a full sign flip of
  the position gradient, ~0.6 power to detect the original low/high
  tercile gap (11.57 against 2.63 points, `docs/results/density-followups.md`
  §8) reappearing. A continuous slope model gains
  power over the tercile split used in the CPU analysis.

## Time estimate on an old GPU

- Existing 1.7B plain-arm answers: median 108 new tokens, mean 141, max
  567 (`n_new_tokens` in the existing `degdens40`/`degdensrep` rows).
- Single-stream HF generation on an old card (prompt read + decode)
  should run roughly **5–15 s/row**; the cluster itself measured only
  ~7 tok/s for an 8B model on an L40S, so assume the slow end.
- **Stage 1** (4 conditions × 1,200 graphs = 4,800 rows): **~7–20 hours.**
  Each stage-2 condition adds ~2–5 hours (1,200 rows).
- Pascal cards (GTX 10xx, P100) need fp32 and may run 1.5–2x slower than
  this estimate.
- **Run a 40-row pilot first** to get real seconds/row and to confirm
  fp16 doesn't produce garbage (Qwen3 can overflow in fp16 on some ops).
- **Keep `--batch-size 1`.** `cluster/README.md`'s batching validation
  measured only a 1.44x speedup and a ~4% answer-changing perturbation —
  not worth it at this effect size.

## CPU work before any GPU time (~half a day)

- Add the new conditions through `render_primer` (the "one renderer"
  invariant — see `CLAUDE.md`), with a per-graph seeded shuffle for the
  shuffled-value condition.
- Confirm `parse_primer` round-trips the reversed-order primer, and that
  the new conditions' shortcut bars equal `none`'s (they must add no
  graph-blind signal, or they aren't valid placebos).
- Add a dtype override (env var) to `graphtalk/hf_backend.py` for
  non-bf16 cards.
- Extend `scripts/analyze_rq3_leads.py` with the position-slope test so
  the new runs can be scored the same way as the existing dedicated
  sweep.
- Build and spot-check the new prompts before submitting the pilot.

## Relationship to `superseded/docs/rq3-leads.md`

This plan implements leads 1–4 of that document's "Leads" section; lead 5
(pre-registered confirmation) is folded into this design via the fixed
seed and the H1–H3 hypotheses above, rather than run separately.
