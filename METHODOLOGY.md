# Methodology, results, and limitations

## 1. Motivation

#43374's `evict_session_token_range()` is explicit and caller-driven, motivated by streaming
video, where a frame boundary is a natural external eviction signal. Long-running text sessions
(chat, agent loops) typically have no equivalent signal. This work explores a second mode —
eviction triggered autonomously once a session's window exceeds a configured threshold — and,
separately, tests which position-correction scheme to pair it with.

## 2. Three schemes compared

Given a continuous cache `[sink][evicted history][surviving window]`:

- **A — Full replay (control).** Flush history, re-run a full forward pass over surviving tokens
  with a freshly assigned, tightly packed position index. Exact but expensive — cost scales with
  the retained window on every eviction.
- **B (corrected) — Renumber + re-rotate.** Keep the cached K/V, counter-rotate the surviving keys
  by `-δ` using the RoPE rotation itself (no forward pass):

  ```python
  def apply_rope(x, positions, freqs):
      half = x.shape[-1] // 2
      x1, x2 = x[..., :half], x[..., half:]
      angles = positions.unsqueeze(-1) * freqs
      cos, sin = torch.cos(angles), torch.sin(angles)
      return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
  ```

  `K_corrected = apply_rope(K_cached, -δ, FREQS)`, applied to keys only, never values, never the
  sink.
- **B (uncorrected) — Leave gap.** Drop the evicted blocks' physical pointers; leave the surviving
  keys' position values untouched. Zero compute, a permanent position discontinuity between sink
  and surviving window.

## 3. Testbed

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` — GQA, 24 layers, 2 KV heads, head dim 64,
  `rope_theta=1e6`.
- **Task:** synthetic multi-fact recall. Sink tokens, then 3 named facts ("Alice's secret number
  is 4."), then `n_cycles × 16` filler tokens (triggering eviction cycles along the way), then a
  query for one of the three facts. Correctness = argmax logprob on the correct digit.
- **Config:** `sink_size=6, window_size=24, block_size=8`, 50 trials × 3 seeds = 150 per
  `n_cycles` point, for `n_cycles ∈ {2, 4, 8, 12, 16}`.
- **Design note:** all three strategies share the same random stream per trial (paired design),
  and the two B variants share a cloned pre-eviction cache to avoid duplicating the shared
  forward-pass prefix. Full replay (A) uses its own trigger (`horizon=48`) rather than the
  block-count trigger B uses, so eviction *cadence* differs between A and B even though final
  retained context size is comparable — worth keeping in mind when comparing A directly to B.

## 4. Results

| n_cycles | mean evictions | A: full replay | B: re-rotate | B: leave gap |
|---|---|---|---|---|
| 2 | ~0 | 88.7% | 92.0% | 92.0% |
| 4 | 4.0 | 84.0% | 82.7% | 90.0% |
| 8 | 12.0 | 90.0% | 89.3% | 95.3% |
| 12 | 20.0 | 87.3% | 88.7% | 96.0% |
| 16 | 28.0 | 90.7% | 91.3% | 95.3% |

The `n_cycles=2` row logs close to zero evictions and is a pre-eviction consistency check, not a
real data point — worth a quick sanity check that the window-token count actually stayed under
threshold there rather than assuming it (the sweep script does print a warning when mean evictions
< 1 per cell; if that warning didn't fire for this row, that's good evidence it's fine).

**Observed pattern:** re-rotate tracks full replay within ~1–4 points at every row. Leave-gap
outperforms both by 5–13 points at every row with evictions. Neither of these should be read as
statistically confirmed yet — see Limitations.

## 5. Precision check on the rotation math (preliminary — one data point)

To separate "rotation math is imprecise" from "attention propagation degrades under gaps," a
single check compares composing two `apply_rope` calls (rotate to `P`, then `-δ`) against
embedding fresh content directly at `P - δ`:

```
P=40, δ=8, synthetic torch.randn content: max abs diff = 4.77e-7
```

This is at the float32 machine-epsilon noise floor, and is consistent with the accuracy gap being
an attention-propagation effect rather than a rotation-arithmetic error — **at this one point**.

**This is not yet a full verification**, and the accuracy-gap interpretation above should be read
as provisional until it is. Specifically still open:

- only `rerotate_cache`'s *underlying* rotation was checked, not `rerotate_cache` as called in the
  eviction path itself
- only one `(P, δ)` pair was run — no sweep, and no large-δ point (the regime where float32
  argument-reduction error on `cos`/`sin` of large angles is a known risk)
- run in float32; production inference typically runs bf16/fp16, which has far less mantissa
  precision and could show measurable drift at far smaller magnitudes than float32 would
- run on synthetic (`torch.randn`) content, not real forward-pass activations

Broadening this (swept `(P, δ)` including large δ, real content, production dtypes,
`rerotate_cache` itself) is listed as open work below, not assumed.

## 6. Important caveat: what this experiment does and doesn't test

In every strategy, the three named facts are held in a separate cache region (`fact_caches`) that
is **never subject to eviction**. Only the filler tokens between the facts and the query are
evicted and corrected/left-gapped. This isolates the effect of a positional discontinuity
downstream of a fact from the effect of losing the fact's own cache entries — a legitimate thing
to isolate on its own, but it is a narrower claim than "multi-fact recall survives autonomous
eviction" might suggest, and narrower than what the scheduler sketch in Section 8 would actually
do (it evicts by raw sequence position with no concept of pinning). A companion run where the fact
itself is eligible for eviction is needed before this generalizes to a real long-running session.

## 7. Related work

- **OpenVINO GenAI's KV-cache eviction** ships exactly this choice as a toggle
  (`CacheEvictionConfig.apply_rotation`), **off by default** — i.e., leave-gap is already a
  production default elsewhere, for the same reason cited here (rotation cost), with the same
  caveat that it's scoped to "regular, linear LLaMa-like RoPE." This result is consistent with
  that existing default, not a new discovery — worth stating that way rather than as a novel
  finding.
- **StreamingLLM** (Xiao et al., 2023) — the original attention-sink streaming design — applies
  position transforms to re-index the rolling cache contiguously, i.e., closer to the "renumber"
  family than to "leave gap." "Leave gap" is a further relaxation beyond StreamingLLM's own
  design, which is worth being explicit about in framing.
- **Counter-evidence to engage with:** "Stateful KV Cache Management for LLMs" (arXiv 2511.04686)
  argues the opposite for its settings — that disrupting positional coherence via non-contiguous
  eviction can paradoxically *worsen* performance, and that retaining contiguous blocks is safer.
  This work doesn't yet reconcile that finding; it may come down to task type, eviction pattern
  (contiguous block vs. scattered), or model scale, and is a good next thing to test against
  directly rather than omit.
- A separate line of eviction work (e.g., work on prompt-pinning in scored eviction methods) finds
  that once important content is pinned by rule, most of the difference between eviction policies
  disappears — relevant context for why the pinned-fact design here isn't unusual, but also a
  reason the non-pinned case matters.

## 8. Open questions

- Full rerotation verification (swept `(P, δ)`, large δ, production dtype, real content, the
  actual `rerotate_cache` call path).
- Facts inside the eviction-eligible region, not pinned.
- Paired significance testing (McNemar / paired bootstrap) on the existing per-trial log — same
  stream is scored by all three strategies per trial, so this is available without new runs.
- Amortized precomputation cost under production conditions (larger models, concurrent
  multi-session load, tail latency) — untested here.
- Shared-block eviction under prefix caching (`ref_cnt > 1`) — #43374 already flags this as
  unsolved; not addressed here.
- Model-scale and config generalization — only one 0.5B model and one hyperparameter point tested.
  A `WINDOW_VALUES × CYCLE_VALUES` sweep at fixed `sink_size` (the "front-loaded slice" idea) is a
  reasonable next step, but hasn't been run yet — any claim about `sink_size` being a "flat lever"
  should wait until it's actually varied.

## 9. Future-work sketch: what a vLLM integration might look like (unimplemented, unassessed)

This is a sketch only — no implementation work has been done, and the snippets below have not
been checked against vLLM's actual scheduler/config APIs.

```
[ vLLM Scheduler Loop ]
        │
        ▼
[ sequence length check ] ── exceeds threshold? ── yes ──▶ [ compute compaction range ]
        │ no                                                        │
        ▼                                                            ▼
[ standard ingestion ]                              [ instruct block manager: release
                                                       historical blocks, leave logical
                                                       indices un-shifted ]
```

Sketch config surface: `--experimental-autonomous-text-eviction-threshold`,
`--experimental-text-eviction-window`. Sketch scheduler hook: a threshold check in
`_schedule_default()` that calls `evict_session_token_range()` with `apply_rotation=False`. None
of this has been prototyped against vLLM internals; treat it as a placeholder for a later phase,
not a design that's been validated.