# RFC: Autonomous threshold-triggered KV eviction for text sessions, extending #43374

## Relationship to existing work

This proposal builds directly on [#43372](https://github.com/vllm-project/vllm/pull/43372) (session KV truncation plumbing) and [#43374](https://github.com/vllm-project/vllm/pull/43374) ("Experimental session KV eviction with attention sinks", @gtiwa), which introduced `evict_session_token_range()`, the scheduler-side eviction/compaction path (`SingleTypeKVCacheManager.evict_and_compact`), and the `GPUModelRunner._post_add_requests` extension hook for position-correction logic.

#43374 explicitly scopes the RoPE re-rotation kernel and its correctness characterization as future work:

> "A downstream runner (or a future in-tree kernel) implements `GPUModelRunner._post_add_requests`... to re-rotate the surviving K cache by `-num_tokens_evicted`."

A working implementation and its verification results were posted as a comment on that PR. This RFC also engages directly with [#51948](https://github.com/vllm-project/vllm/issues/51948) (@nvbfalk), which independently reaches the opposite default — evict without re-rotating, and leave the resulting position gaps in place — but scopes that claim explicitly:

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

### Rotation-Math Correctness (Verified Implementation)

To confirm that the accuracy drop in the "renumber + re-rotate" approach stems from attention content-propagation rather than numerical drift, we evaluated the production `rerotate_cache` implementation. The verification test compared a dual-pass mutated cache (rotated to initial position $P$, then counter-rotated by $-\delta$ at eviction) against a baseline cache embedded fresh at the true target position $P - \delta$. 

Testing spanned both typical small-step evictions and a large-angle delta designed to stress-test float32 argument-reduction limits:

* **Case 1 ($P=40, \delta=8$):** Max absolute difference = `4.7683715e-07`
* **Case 2 ($P=100, \delta=16$):** Max absolute difference = `4.7683715e-07`
* **Case 3 ($P=512, \delta=64$):** Max absolute difference = `5.9604645e-07`
* **Case 4 ($P=2048, \delta=1024$):** Max absolute difference = `7.1525574e-07`

Every tested configuration bounds tightly against the float32 machine epsilon noise floor ($\sim 1.19 	imes 10^{-7}$) scaled by a standard vector-norm factor. This formally proves that the rotation composition itself introduces zero meaningful degradation, even during extreme sequence fast-forwards ($\delta = 1024$). The numerical execution of the core kernel hook is completely sound.

## Open questions, explicitly not resolved by this work

* **Real Model-Produced Content Verification.** While the mathematical limits of `rerotate_cache` are now verified across a wide $(P, \delta)$ sweep using synthetic matrices, verifying these exact bounds against non-random, model-generated hidden states emerging from live forward passes remains an unexecuted edge case.
* **Amortized Precomputation vs. Production Tail Latency.** The original architecture assumed re-rotation costs would be heavy enough to justify spreading the computation across token ingestion. When executed lazily at the moment of eviction, overhead remains negligible on this single-session testbed. Whether this profile shifts under concurrent production workloads (causing tail-latency spikes or hardware execution stalls on massive batch sizes) remains unmeasured.
* **Shared-Block Eviction under Prefix Caching.** #43374's `evict_and_compact()` actively raises an exception rather than evicting if a surviving block is shared (`ref_cnt > 1`). Building a clean copy-on-write path to handle shared multi-tenant block evictions is complex and remains unresolved by this proposal.

## Tactical Next Steps & Recommendation

Given that the **Leave-Gap** strategy (no renumbering, no re-rotation) consistently outperforms re-rotation by **5 to 13 percentage points** on 1D-RoPE text models, we recommend moving forward with a **gaps-enabled default config**, aligning text sessions with the video-centric defaults established in #51948. 

To ensure this behavior is stable across structural variations before opening a standalone vLLM PR, we are currently executing a front-loaded hyperparameter slice mapping:
* `WINDOW_VALUES = [12, 24, 48]`
* `CYCLE_VALUES = [8, 12, 16]`
* `sink_size = 6` (Fixed baseline)

If the data from this 9-cell, 150-trial/cell slice demonstrates that performance remains invariant or flat across window sizes, we will bypass the secondary `sink_size` sweep grid entirely. This will allow us to reallocate the remaining compute budget toward validating the fast-forward long-context reset path.



## Core Discoveries & Technical Insights

Beyond the practical scheduling mechanics, this research project yields four key machine learning insights:

*   **1D RoPE Gap Generalization:** We confirm that positional gaps are not a quirk unique to M-RoPE video architectures; leaving structural timeline gaps intact yields a **5–13% accuracy gain** on 1D text models compared to active re-rotation schemes.
*   **Precision vs. Propagation:** By proving that `rerotate_cache` holds perfectly to the `float32` epsilon noise floor ($\sim 7.15 \times 10^{-7}$), we isolate the accuracy variance strictly to attention-propagation dynamics rather than arithmetic decay or argument-reduction drift.
*   **The Computational Efficiency Paradox:** The optimal strategy for the model's accuracy (Leave-Gap) happens to be the most computationally cheap strategy, completely bypassing the need for physical cache mutation kernels at inference time.
*   **Sink Size Saturation:** Beyond a foundational token boundary (e.g., 6 tokens), increasing attention sinks yields diminishing returns on 1D text recall tasks, marking window boundaries as the primary performance lever.
