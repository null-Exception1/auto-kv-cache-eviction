# RFC: Autonomous threshold-triggered KV eviction for text sessions, extending #43374

## Relationship to existing work

This proposal builds directly on [#43372](https://github.com/vllm-project/vllm/pull/43372) (session KV truncation plumbing) and [#43374](https://github.com/vllm-project/vllm/pull/43374) ("Experimental session KV eviction with attention sinks", @gtiwa), which introduced `evict_session_token_range()`, the scheduler-side eviction/compaction path (`SingleTypeKVCacheManager.evict_and_compact`), and the `GPUModelRunner._post_add_requests` extension hook for position-correction logic.

#43374 explicitly scopes the RoPE re-rotation kernel and its correctness characterization as future work:

> "A downstream runner (or a future in-tree kernel) implements `GPUModelRunner._post_add_requests`... to re-rotate the surviving K cache by `-num_tokens_evicted`."

A working implementation and its verification results were posted as a comment on that PR ([link]). This RFC also engages directly with [#51948](https://github.com/vllm-project/vllm/issues/51948) (@nvbfalk), which independently reaches the opposite default — evict without re-rotating, and leave the resulting position gaps in place — but scopes that claim explicitly:

> "Gaps are benign for the multimodal-RoPE model class we target, and we measured it."

This RFC tests whether that claim holds outside the model class it was scoped to. See "Scope" below.

Covers, beyond both threads' existing scope:

1. An autonomous, threshold-triggered eviction mode as an addition to #43374's explicit caller-driven API
2. Empirical results comparing three eviction-correction schemes on a text-session task, outside the model class #51948's "gaps are benign" claim was measured on
3. Two explicitly open questions this work does not resolve

## Scope

**In scope:** text sessions using standard (1D) RoPE, as implemented in models like Qwen2.5, Llama 3, Mistral.

**Explicitly out of scope:** multimodal sessions using M-RoPE. #43374's `_reindex_mm_features` and atomic-multimodal-item eviction exist because M-RoPE splits position into temporal/height/width components — a materially different correctness problem than the 1D case this proposal validates. #51948's "gaps are benign" result was measured on that M-RoPE model class; this RFC does not extend or contest that finding on its own terms, only tests whether the same design choice holds for 1D RoPE text models, which #51948 does not claim to cover.

## Motivation: why an autonomous trigger

#43374's `evict_session_token_range(request_id, num_tokens_to_evict, num_sink_tokens)` is explicit and caller-driven, motivated by streaming video, where a frame boundary is a natural, externally-known eviction signal. Long-running text sessions (chat, agent loops) typically have no equivalent external signal — nothing outside the KV cache itself knows when eviction should happen.

This proposal adds a second mode: eviction triggered autonomously by the scheduler once a session's window exceeds a configured size/block threshold, requiring no caller involvement. This is additive to #43374's API, not a replacement for it — the explicit-range API remains the right fit for its motivating case.

## Results

Tested on Qwen2.5-0.5B-Instruct (GQA, 24 layers, 2 KV heads), a synthetic multi-fact recall task (3 named facts, 1 queried per trial), `sink_size=6, window_size=24, block_size=8`, n=150 trials per point across 3 seeds:

| n_cycles | mean evictions | A: full replay | B: renumber + re-rotate | B: leave gap, no re-rotation |
|---|---|---|---|---|
| 4 | 4.0 | 84.0% | 82.7% | 90.0% |
| 8 | 12.0 | 90.0% | 89.3% | 95.3% |
| 12 | 20.0 | 87.3% | 88.7% | 96.0% |
| 16 | 28.0 | 90.7% | 91.3% | 95.3% |

(n_cycles=2 excluded: mean evictions ≈ 0, eviction never triggered at that window/cycle-length combination.)

Two findings, reported as measured rather than as originally hypothesized:

- **Renumber + re-rotate tracks full replay closely** (within ~1-4 points at every point), while avoiding full replay's cost — full forward passes over every retained token, vs. re-rotation's closed-form rotation of already-computed K vectors. This is the core cost/accuracy tradeoff the eviction mechanism was meant to validate, and it holds.
- **Leave-gap (no renumbering, no re-rotation) consistently outperforms both other schemes**, by 5-13 points, at every eviction count tested, with no narrowing or widening trend as eviction count grows. This is the same design choice made in #51948, now measured directly on a 1D-RoPE text model, outside the M-RoPE scope that RFC's own claim was measured on. We report this as the honest result rather than the one the project set out to find.

**Rotation-math correctness (preliminary).** A single-point check compared composing two `apply_rope` calls (rotate to position P, then re-rotate by `-evict_n`) against embedding the same content fresh, directly at position `P - evict_n`. At P=40, evict_n=8, on synthetic (`torch.randn`) content: max absolute difference across all K channels was `4.77e-7`, at the float32 precision noise floor. This supports rotation-composition exactness at this point but is not yet a full verification: it exercises the underlying `apply_rope` composition rather than the `rerotate_cache` function as shipped, at a single `(P, δ)` pair rather than a swept range (notably not yet including a large-δ point, where float32 argument-reduction error is a known risk), and on synthetic rather than model-produced content. Broadening this check — `rerotate_cache` itself, multiple `(P, δ)` pairs including large δ, real forward-pass content — is listed as an open item below rather than assumed.

The accuracy gap above therefore reflects content-propagation effects through attention (not position-rotation error, per the preliminary check above) — a limitation acknowledged in the original StreamingLLM re-rotation design and now measured layer-by-layer on a real model.

## Open questions, explicitly not resolved by this work

- **Full rerotation verification.** The preliminary single-point rotation check above needs to be broadened to the actual `rerotate_cache` function, a swept range of `(P, δ)` pairs including large δ, and real (non-synthetic) cached content before the correctness claim can be considered settled.
- **Amortized precomputation.** The original design motivation for this line of work assumed re-rotation cost was significant enough to justify spreading it across token ingestion rather than paying it at eviction time. Measured lazily (rotation applied at eviction), the cost is small on this model/scale/single-session setup. Whether this changes under production conditions this testbed can't reproduce — larger models, concurrent multi-session load, tail latency rather than throughput — is untested and remains open.
- **Shared-block eviction under prefix caching.** #43374's `evict_and_compact()` currently raises rather than evicting when a surviving block is shared (`ref_cnt > 1`), and lists a copy-on-write path as unsolved future work. This proposal does not resolve it.

## Summary of what this adds beyond #43374 and #51948

- A working re-rotation implementation for the `_post_add_requests` hook (posted separately as a PR comment), with a preliminary (single-point) correctness check reported above and full verification still open
- An autonomous threshold-triggered eviction mode, addressing sessions without external eviction signals
- A direct empirical comparison between renumber+rotate and leave-gap schemes, on a 1D-RoPE text task outside the model class #51948's "gaps are benign" result was scoped to